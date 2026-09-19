"""Per-layer extraction of the two Kronecker-factor statistics, for every supported layer type.

Every mode in this package is built from the same two statistics of one layer:

* the input factor ``A = E[h_bar h_bar^T]``, where ``h_bar`` is the layer's input with a
  constant-one column appended when the layer has a bias, so that the bias is one more input
  coordinate;
* the output factor ``B = E[delta delta^T]``, where ``delta`` is the gradient arriving at the
  layer's output.

The expectation is over examples, and over positions as well wherever a layer has them -- output
locations for ``Conv2d``, positions for a normalisation layer, tokens for a ``Linear`` applied to a
sequence. Each ``(example, position)`` pair is treated as one independent sample, which is the same
reading the diagonal path already relies on implicitly.

This module has two families of function.

**Diagonal statistics** (:func:`compute_h_diag`, :func:`compute_s_diag`) are what the ``diag`` mode
uses, and they are a line-for-line port of ``_ComputeHBarD`` / ``_ComputeSD`` from
``reference_repos/FisherAdapTune/scripts/adafisher.py``. They are deliberately kept bit-exact: the
non-regression test compares them with ``torch.equal`` against that reference, and even a
mathematically identical change of reduction order would break it. They compute the diagonal
directly with ``einsum``; the full matrix is never formed.

**Full factors** (:func:`compute_h_full`, :func:`compute_s_full`, and the ``augment_*`` /
``flatten_*`` helpers) are what the four Kronecker modes use. They do not exist in the original
AdaFisher code at all, which is the main structural cost of this project. The ``augment_*`` and
``flatten_*`` helpers are public because ``ekfac``, ``tkfac`` and ``tekfac`` need the raw
per-example batch, not only its reduction.

Three layer-specific facts are worth knowing before reading any number out of this module.

**A normalisation layer's exact Fisher is not a Kronecker product.** Proposition 3.1 of the
AdaFisher paper (``adafisher_2405.16397.pdf``, proved as Proposition A.1) gives, for scale and shift
parameters ``(nu, beta)``, ``FIM_nu = H|_nu (elementwise) S`` and ``FIM_beta = S``, where
``H|_nu`` and ``S = E[s s^T]`` are both ``C x C``. The combination is element-wise, not Kronecker.
To fit the shared machinery, the input factor built here is the 2 x 2 matrix
``[[a_nu, mean(z)], [mean(z), 1]]``, where ``z`` is the mean over channels of the layer's input at
one position. ``a_nu = mean(z^2)`` is exactly the mean of all entries of ``H|_nu``, which is the
scalar ``a`` minimising ``||H|_nu - a * ones||_F``; the ``1`` reproduces ``FIM_beta = S`` exactly.
The off-diagonal ``mean(z)`` is a correlation the Proposition assumes away; it is kept rather than
forced to zero, for the same reason a ``Linear``'s own bias column is not centred either. Do not
special-case it to zero. One caveat: the Proposition's proof writes ``H|_nu = E[h h^T]`` for the
*normalised* activation, while this module -- like both reference implementations -- builds it from
the forward hook's raw, pre-normalisation input. See :func:`augment_norm_input` for the measured
size of that difference and for why swapping in the normalised activation is not a safe local fix.

**The diagonal ``Conv2d`` input factor has two scale quirks, both inherited and both left alone.**
With a bias, ``_h_conv2d`` divides by ``batch * S * P`` where ``P = C_in * k_h * k_w`` is the patch
width, instead of by ``batch * S``. Without a bias it divides by ``batch`` alone, leaving out both
``S`` and ``P``. Both reference repositories do exactly this. The new full-factor path deliberately
does *not* reproduce either, because TKFAC's trace preservation needs the real scale; measured, the
full factor's diagonal is ``P`` times the diagonal path's with a bias and ``1/S`` times it without
one. The quirks are harmless for ``diag`` only because its min-max normalisation erases any
per-layer constant.

**``LayerNorm`` contracts different axes on the two sides, for a 3-D input.** ``_h_layernorm``
reduces every dimension except dim 1, ``_s_layernorm`` reduces every dimension except the last. For
a 2-D ``(N, C)`` input those coincide. For a transformer's ``(N, T, C)`` they do not: the input side
then pools over batch and channel and indexes tokens, while the gradient side pools over batch and
tokens and indexes channels. Both are verbatim ports of both reference repositories.

Scope: ``Conv2d`` is supported with ``groups=1``, ``dilation=(1, 1)`` and
``padding_mode="zeros"``, and ``LayerNorm`` with a one-dimensional ``normalized_shape`` and a shift
parameter. Anything else raises ``NotImplementedError`` rather than silently computing a factor
that ignores the structure. Two of those refusals were silent until they were measured:
``padding_mode="reflect"`` was described by a zero-padded factor 48% off the right one in relative
Frobenius norm, because :func:`extract_patches` always pads with zeros; and ``LayerNorm(bias=False)``
got the usual 2 x 2 factor, whose second row and column describe a shift parameter the layer does
not have. A string ``padding`` (``"same"``, ``"valid"``) is refused in :func:`extract_patches`
itself, which is the one place both the diagonal and the full-factor paths go through.

The SUA option (``sua=True``, ``Conv2d`` only) replaces the patch-based input factor by a
channel-only one, ``C_in [+1]`` wide instead of ``C_in * k_h * k_w [+1]``
(``kfac_conv_1602.01407.pdf`` p. 14, "spatially uncorrelated activations"). It is built by taking
the centre offset of every patch the extraction already produces, which keeps it row-aligned with
the pooled output gradients by construction, for any stride and padding. Pooling the raw input
independently, as ``EKFAC-pytorch`` does, coincides with this only for stride 1 and "same" padding.

Two things SUA here is *not*. It is not Theorem 4's "IAD + SH + SUA + WD": the output factor stays
full, and "white derivatives" is deliberately not adopted. And even restricted to SUA alone, the
exact block has a rank-1 coupling term between kernel offsets that this construction drops -- the
offsets are treated as exactly independent, matching ``EKFAC-pytorch::_precond_sua_ra``'s own
uncorrected convention. Both are documented gaps between the cited theorem and what is tractable
here, not bugs. Measured on a 3 x 3, 2-input-channel toy convolution, the cross-offset terms SUA
discards carry 29.5% of the exact patch covariance's Frobenius mass.
"""

from __future__ import annotations

from math import prod
from typing import Tuple

import torch
from torch import Tensor, cat, einsum
from torch import sum as tsum
from torch.nn import BatchNorm2d, Conv2d, LayerNorm, Linear, Module
from torch.nn.functional import pad

SUPPORTED_MODULES: Tuple[str, ...] = ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm")


def extract_patches(
    x: Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    groups: int,
) -> Tensor:
    """Unfold a convolutional input into flattened receptive-field patches.

    Port of ``_extract_patches`` (adafisher.py:33-50), plus one guard the port did not have.

    ``padding`` must be a pair of integers. ``Conv2d`` also accepts the strings ``"same"`` and
    ``"valid"``, and it keeps them verbatim in ``layer.padding``; arithmetic on them used to fail
    here with ``TypeError: '>' not supported between instances of 'str' and 'int'``, from inside a
    tensor reshape, which said nothing about the layer that caused it.
    """
    if isinstance(padding, str):
        raise NotImplementedError(
            f"Conv2d(padding={padding!r}) is not supported: the patch extraction needs a pair of "
            f"integers, and PyTorch keeps a string padding verbatim in layer.padding. Give the "
            f"equivalent explicit padding instead -- for a stride-1 layer, padding='same' is "
            f"(k_h // 2, k_w // 2) for odd kernel sizes, and padding='valid' is 0."
        )
    if padding[0] + padding[1] > 0:
        x = pad(x, (padding[1], padding[1], padding[0], padding[0]))
    batch_size, in_channels, height, width = x.size()
    x = x.view(batch_size, groups, in_channels // groups, height, width)
    x = x.unfold(3, kernel_size[0], stride[0])
    x = x.unfold(4, kernel_size[1], stride[1])
    x = x.permute(0, 1, 3, 4, 2, 5, 6).contiguous()
    x = x.view(batch_size, groups, -1, in_channels // groups * kernel_size[0] * kernel_size[1])
    x = x.view(batch_size, -1, x.size(2), x.size(3))
    return x


def _h_conv2d(h: Tensor, layer: Conv2d) -> Tensor:
    batch_size = h.size(0)
    h = extract_patches(h, layer.kernel_size, layer.stride, layer.padding, layer.groups)
    spatial_size = h.size(2) * h.size(3)
    h = h.reshape(-1, h.size(-1))
    if layer.bias is not None:
        h_bar = cat([h, h.new(h.size(0), 1).fill_(1)], 1)
        return einsum("ij,ij->j", h_bar, h_bar) / (batch_size * spatial_size)
    return einsum("ij,ij->j", h, h) / batch_size


def _h_linear(h: Tensor, layer: Linear) -> Tensor:
    if len(h.shape) > 2:
        h = h.reshape(-1, h.shape[-1])
    batch_size = h.size(0)
    if layer.bias is not None:
        h_bar = cat([h, h.new(h.size(0), 1).fill_(1)], 1)
        return einsum("ij,ij->j", h_bar, h_bar) / batch_size
    return einsum("ij,ij->j", h, h) / batch_size


def _h_batchnorm2d(h: Tensor, layer: BatchNorm2d) -> Tensor:
    batch_size, spatial_size = h.size(0), h.size(2) * h.size(3)
    sum_h = tsum(h, dim=(0, 2, 3)).unsqueeze(1) / (spatial_size**2)
    h_bar = cat([sum_h, sum_h.new(sum_h.size(0), 1).fill_(1)], 1)
    return einsum("ij,ij->j", h_bar, h_bar) / (batch_size**2)


def _h_layernorm(h: Tensor, layer: LayerNorm) -> Tensor:
    dim_to_reduce = [d for d in range(h.ndim) if d != 1]
    batch_size = h.shape[0]
    dim_norm = prod([h.shape[dim] for dim in dim_to_reduce if dim != 0])
    sum_h = tsum(h, dim=dim_to_reduce).unsqueeze(1) / (dim_norm**2)
    h_bar = cat([sum_h, sum_h.new(sum_h.size(0), 1).fill_(1)], 1)
    return einsum("ij,ij->j", h_bar, h_bar) / (batch_size**2)


def compute_h_diag(h: Tensor, layer: Module) -> Tensor:
    """Instantaneous diagonal of H_{l-1} = E[h_bar h_bar^T], dispatched by layer type.

    Port of ``_ComputeHBarD.__call__`` (adafisher.py:53-63).
    """
    if isinstance(layer, Linear):
        return _h_linear(h, layer)
    if isinstance(layer, Conv2d):
        return _h_conv2d(h, layer)
    if isinstance(layer, BatchNorm2d):
        return _h_batchnorm2d(h, layer)
    if isinstance(layer, LayerNorm):
        return _h_layernorm(h, layer)
    raise NotImplementedError(f"Unsupported layer type: {type(layer)}")


def _s_conv2d(s: Tensor, layer: Conv2d) -> Tensor:
    batch_size = s.shape[0]
    spatial_size = s.size(2) * s.size(3)
    s = s.transpose(1, 2).transpose(2, 3).reshape(-1, s.size(1))
    return einsum("ij,ij->j", s, s) / (batch_size * spatial_size)


def _s_linear(s: Tensor, layer: Linear) -> Tensor:
    if len(s.shape) > 2:
        s = s.reshape(-1, s.shape[-1])
    batch_size = s.size(0)
    return einsum("ij,ij->j", s, s) / batch_size


def _s_batchnorm2d(s: Tensor, layer: BatchNorm2d) -> Tensor:
    batch_size = s.size(0)
    sum_s = tsum(s, dim=(0, 2, 3))
    return einsum("i,i->i", sum_s, sum_s) / batch_size


def _s_layernorm(s: Tensor, layer: LayerNorm) -> Tensor:
    batch_size = s.size(0)
    sum_s = tsum(s, dim=tuple(range(s.ndim - 1)))
    return einsum("i,i->i", sum_s, sum_s) / batch_size


def compute_s_diag(s: Tensor, layer: Module) -> Tensor:
    """Instantaneous diagonal of S_l = E[delta delta^T], dispatched by layer type.

    Port of ``_ComputeSD.__call__`` (adafisher.py:114-124).
    """
    if isinstance(layer, Conv2d):
        return _s_conv2d(s, layer)
    if isinstance(layer, Linear):
        return _s_linear(s, layer)
    if isinstance(layer, BatchNorm2d):
        return _s_batchnorm2d(s, layer)
    if isinstance(layer, LayerNorm):
        return _s_layernorm(s, layer)
    raise NotImplementedError(f"Unsupported layer type: {type(layer)}")


# ----------------------------------------------------------------------------------------------
# Full (non-diagonal) factors: the dispatchers, and the Linear branch.
# ----------------------------------------------------------------------------------------------


def augment_linear_input(h: Tensor, layer: Linear) -> Tensor:
    """Flatten ``h`` to ``(N, d_in)`` and, iff ``layer`` has a bias, append a ones column: the
    bias-augmented ``h_bar`` batch of AdaFisher's header formula. Public (not just an internal
    helper of ``compute_h_full``) because ``EKFACApproximation`` also needs the raw augmented batch
    itself, not only its reduction.
    """
    if h.ndim > 2:
        h = h.reshape(-1, h.shape[-1])
    if layer.bias is not None:
        return cat([h, h.new_ones(h.size(0), 1)], 1)
    return h


def _h_full_linear(h: Tensor, layer: Linear) -> Tensor:
    h_bar = augment_linear_input(h, layer)
    return h_bar.t() @ h_bar / h_bar.size(0)


def compute_h_full(h: Tensor, layer: Module, sua: bool = False) -> Tensor:
    """Instantaneous full input factor ``A = E[h_bar h_bar^T]``.

    Supports ``Linear``, ``Conv2d`` with ``groups=1`` and ``dilation=(1, 1)``, ``BatchNorm2d``, and
    ``LayerNorm`` with a 1-D ``normalized_shape``. ``sua`` is consulted only on the ``Conv2d``
    branch, where it selects the channel-only input factor instead of the patch-based one; it is
    silently ignored for every other layer type, the same way ``pi`` and ``T_inv`` are ignored by
    the modes they do not apply to.
    """
    if isinstance(layer, Linear):
        return _h_full_linear(h, layer)
    if isinstance(layer, Conv2d):
        return _h_full_conv2d_sua(h, layer) if sua else _h_full_conv2d(h, layer)
    if isinstance(layer, (BatchNorm2d, LayerNorm)):
        return _h_full_norm(h, layer)
    raise NotImplementedError(
        f"compute_h_full only supports Linear, Conv2d, BatchNorm2d and LayerNorm; got {type(layer)}"
    )


def _s_full_linear(s: Tensor, layer: Linear) -> Tensor:
    if s.ndim > 2:
        s = s.reshape(-1, s.shape[-1])
    return s.t() @ s / s.size(0)


def compute_s_full(s: Tensor, layer: Module) -> Tensor:
    """Instantaneous full output factor ``B = E[delta delta^T]``.

    Same supported layer types as :func:`compute_h_full`. For a normalisation layer this is
    literally Proposition 3.1's ``S_i``, the sum of per-position outer products: full-rank, and no
    approximation is involved. Note that the ``diag`` mode's own normalisation-layer formula sums
    the gradients first and squares afterwards, which is a different quantity; the two are not
    compared anywhere, on purpose.
    """
    if isinstance(layer, Linear):
        return _s_full_linear(s, layer)
    if isinstance(layer, Conv2d):
        return _s_full_conv2d(s, layer)
    if isinstance(layer, (BatchNorm2d, LayerNorm)):
        return _s_full_norm(s, layer)
    raise NotImplementedError(
        f"compute_s_full only supports Linear, Conv2d, BatchNorm2d and LayerNorm; got {type(layer)}"
    )


# ----------------------------------------------------------------------------------------------
# Full factors, Conv2d branch (groups=1, dilation=(1,1)). Every (example, output-location) pair
# becomes one pooled sample, which generalises the Linear branch's per-example pooling.
# ----------------------------------------------------------------------------------------------


def _check_conv2d_supported(layer: Conv2d) -> None:
    """Refuse the two ``Conv2d`` options this factorisation cannot express, rather than silently
    computing a factor that ignores them: ``groups`` (structural weight sparsity, which needs a
    per-group block-diagonal treatment, not one global ``A (x) B``), ``dilation`` (which
    ``extract_patches`` has no parameter for) and ``padding_mode`` (the patch extraction always
    pads with zeros, so a reflected or replicated padding would be described by a factor that does
    not match the layer's own forward pass)."""
    if layer.groups != 1:
        raise NotImplementedError(
            f"Full Kronecker factors for Conv2d only support groups=1 so far (lot 4 scope); "
            f"got groups={layer.groups}. A grouped/depthwise layer needs a block-diagonal "
            f"per-group treatment, not a single global A (x) B."
        )
    if layer.dilation != (1, 1):
        raise NotImplementedError(
            f"Full Kronecker factors for Conv2d only support dilation=(1,1) so far (lot 4 scope); "
            f"got dilation={layer.dilation}. extract_patches has no dilation parameter."
        )
    if layer.padding_mode != "zeros":
        raise NotImplementedError(
            f"Full Kronecker factors for Conv2d only support padding_mode='zeros'; got "
            f"padding_mode={layer.padding_mode!r}. The patch extraction always pads with zeros, "
            f"so the factor would describe a different layer from the one being trained: measured "
            f"on a 2-channel 3x3 layer with padding_mode='reflect', the zero-padded input factor "
            f"is 48% off the reflect-padded one in relative Frobenius norm. Refusing is better "
            f"than returning that number."
        )


def augment_conv2d_input(h: Tensor, layer: Conv2d) -> Tensor:
    """Pool (batch, output-location) patches onto one axis and, iff ``layer`` has a bias, append a
    ones column: the Conv2d analogue of ``augment_linear_input``, built on ``extract_patches``
    (lot 1, unmodified). Each pooled row is treated as an i.i.d. sample of the bias-augmented
    receptive-field patch, under the same spatial-independence reading the diagonal path already
    relies on implicitly (TKFAC Assumption 4.1, ``tkfac_2011.10741.pdf`` §4.3).
    """
    _check_conv2d_supported(layer)
    patches = extract_patches(h, layer.kernel_size, layer.stride, layer.padding, layer.groups)
    patches = patches.reshape(-1, patches.size(-1))  # (N*S, P): groups=1, so dim 1 is trivial
    if layer.bias is not None:
        return cat([patches, patches.new_ones(patches.size(0), 1)], 1)
    return patches


def augment_input(h: Tensor, layer: Module, sua: bool = False) -> Tensor:
    """Dispatching sibling of ``augment_linear_input`` / ``augment_conv2d_input`` /
    ``augment_norm_input``. Used directly by ``ekfac``, ``tkfac`` and ``tekfac``, which need the raw
    augmented batch and not only ``compute_h_full``'s reduction of it. ``sua`` selects
    ``augment_conv2d_input_sua`` in place of ``augment_conv2d_input``, on the ``Conv2d`` branch
    only.
    """
    if isinstance(layer, Linear):
        return augment_linear_input(h, layer)
    if isinstance(layer, Conv2d):
        return augment_conv2d_input_sua(h, layer) if sua else augment_conv2d_input(h, layer)
    if isinstance(layer, (BatchNorm2d, LayerNorm)):
        return augment_norm_input(h, layer)
    raise NotImplementedError(
        f"augment_input only supports Linear, Conv2d, BatchNorm2d and LayerNorm; got {type(layer)}"
    )


def _h_full_conv2d(h: Tensor, layer: Conv2d) -> Tensor:
    """Uses the *real* spatial size (``h_bar.size(0) = batch * S``), unlike ``_h_conv2d``'s
    diagonal reduction, which — for the ``Conv2d`` branch only — divides by ``batch * S * P``
    instead of ``batch * S`` (a pre-existing quirk in the ``FisherAdapTune``/official-repo
    reference, verified and deliberately not reproduced here — the trace-preservation property of
    ``tkfac`` needs the real scale).
    """
    h_bar = augment_conv2d_input(h, layer)
    return h_bar.t() @ h_bar / h_bar.size(0)


def flatten_conv2d_output_grad(s: Tensor, layer: Conv2d) -> Tensor:
    """Pool (batch, output-location) gradients onto one axis: the same transpose/reshape sequence
    as ``_s_conv2d`` (unmodified), minus its diagonal reduction. Unlike ``_h_conv2d``, whose scale
    is off (see the module docstring), ``_s_conv2d``'s was already right — it computes
    ``spatial_size`` before this reshape — so no deviation is introduced here.
    """
    return s.transpose(1, 2).transpose(2, 3).reshape(-1, s.size(1))


def flatten_output_grad(s: Tensor, layer: Module) -> Tensor:
    """Dispatching sibling of :func:`augment_input`, on the output-gradient side. Returns the
    per-sample gradient batch, row-paired with what :func:`augment_input` returns for the same
    layer and the same forward pass.
    """
    if isinstance(layer, Linear):
        return s.reshape(-1, s.shape[-1]) if s.ndim > 2 else s
    if isinstance(layer, Conv2d):
        return flatten_conv2d_output_grad(s, layer)
    if isinstance(layer, (BatchNorm2d, LayerNorm)):
        return flatten_norm_output_grad(s, layer)
    raise NotImplementedError(
        f"flatten_output_grad only supports Linear, Conv2d, BatchNorm2d and LayerNorm; "
        f"got {type(layer)}"
    )


def _s_full_conv2d(s: Tensor, layer: Conv2d) -> Tensor:
    s_pool = flatten_conv2d_output_grad(s, layer)
    return s_pool.t() @ s_pool / s_pool.size(0)


# ----------------------------------------------------------------------------------------------
# The SUA approximation: a channel-only Conv2d input factor, reusing the scope guard above. Only
# the input side is affected; the output factor is unchanged.
# ----------------------------------------------------------------------------------------------


def augment_conv2d_input_sua(h: Tensor, layer: Conv2d) -> Tensor:
    """Channel-only analogue of ``augment_conv2d_input`` (SUA, ``kfac_conv_1602.01407.pdf`` p. 14):
    pools the *center* offset of every receptive-field patch ``extract_patches`` produces, instead
    of the whole patch, dropping the input factor from ``(C_in*k_h*k_w[+1])^2`` to ``(C_in[+1])^2``
    entries. Row-aligned with ``flatten_conv2d_output_grad``'s
    ``(N*S, C_out)`` pooling by construction, since both are built from ``extract_patches``'s own
    ``(H_out, W_out)`` grid — valid for any ``stride``/``padding``, not only "same" padding.
    """
    _check_conv2d_supported(layer)
    kh, kw = layer.kernel_size
    patches = extract_patches(h, layer.kernel_size, layer.stride, layer.padding, layer.groups)
    patches = patches.reshape(-1, patches.size(-1))  # (N*S, C_in*kh*kw)
    center = patches.view(-1, layer.in_channels, kh, kw)[:, :, kh // 2, kw // 2]  # (N*S, C_in)
    if layer.bias is not None:
        return cat([center, center.new_ones(center.size(0), 1)], 1)
    return center


def _h_full_conv2d_sua(h: Tensor, layer: Conv2d) -> Tensor:
    h_bar = augment_conv2d_input_sua(h, layer)
    return h_bar.t() @ h_bar / h_bar.size(0)


# ----------------------------------------------------------------------------------------------
# Full factors, normalisation-layer branch (BatchNorm2d, and LayerNorm with a 1-D
# normalized_shape). See the module docstring for why the 2 x 2 input factor looks the way it does.
# ----------------------------------------------------------------------------------------------


def _pool_batchnorm2d(x: Tensor) -> Tensor:
    """``(N, C, H, W) -> (N*H*W, C)``: move the channel axis (dim 1, ``BatchNorm2d``'s own
    convention, matching ``_h_batchnorm2d``/``_s_batchnorm2d``) last and flatten batch+spatial onto
    one axis — Proposition 3.1's ``T_i = N*H*W``.
    """
    return x.permute(0, 2, 3, 1).reshape(-1, x.size(1))


def _check_layernorm_supported(layer: LayerNorm) -> None:
    """Refuse a multi-dimensional ``normalized_shape``, the same way the ``Conv2d`` guard above
    refuses ``groups`` and ``dilation``. It would need an axis convention this module does not
    define — and one the ported diagonal path does not define consistently either, since
    ``_h_layernorm`` and ``_s_layernorm`` contract different axes as soon as the input has more
    than two dimensions.

    Also refuse ``LayerNorm(d, bias=False)``, which has a scale parameter and no shift. Every input
    factor in this module appends a constant-one column for the shift, so the 2 x 2 factor it
    returns would describe a shift parameter that does not exist, and the mismatch used to surface
    as a bare shape error far from its cause."""
    if len(layer.normalized_shape) != 1:
        raise NotImplementedError(
            f"Full Kronecker factors for LayerNorm only support a 1-D normalized_shape so far "
            f"(lot 5 scope); got normalized_shape={tuple(layer.normalized_shape)}."
        )
    if layer.weight is not None and layer.bias is None:
        raise NotImplementedError(
            "LayerNorm(bias=False) is not supported: this module's input factor always appends a "
            "constant-one column for the shift parameter, so it would describe a parameter the "
            "layer does not have. Use LayerNorm with its default bias, or "
            "elementwise_affine=False, which has no parameters to precondition at all."
        )


def _pool_layernorm(x: Tensor, layer: LayerNorm) -> Tensor:
    """``(..., C) -> (T, C)``: channel is already the last axis for a 1-D ``normalized_shape``
    (PyTorch's own convention), so this is a plain flatten, mirroring ``_h_linear``/``_s_linear``.
    """
    _check_layernorm_supported(layer)
    return x.reshape(-1, x.size(-1))


def _pool_norm_layer(x: Tensor, layer: Module) -> Tensor:
    """The ``(T, C)`` pooled view shared by the input and gradient sides of a normalisation
    layer's full-factor construction: one row per position, one column per channel."""
    if isinstance(layer, BatchNorm2d):
        return _pool_batchnorm2d(x)
    if isinstance(layer, LayerNorm):
        return _pool_layernorm(x, layer)
    raise NotImplementedError(
        f"_pool_norm_layer only supports BatchNorm2d and LayerNorm; got {type(layer)}"
    )


def augment_norm_input(h: Tensor, layer: Module) -> Tensor:
    """``(T, 2)`` matrix ``[z_x, 1]``, the normalisation-layer analogue of
    ``augment_linear_input`` / ``augment_conv2d_input``.

    ``z_x`` is the mean over channels of the layer's input at position ``x``. Reducing the whole
    ``C x C`` matrix ``H|_nu`` of Proposition 3.1 to the single number ``mean_x(z_x^2)`` is the one
    approximation the normalisation-layer branch introduces: it is the scalar ``a`` minimising
    ``||H|_nu - a * ones||_F``, chosen not to depend on the gradient factor so that the two factors
    stay separate and separately invertible. Column 1, the constant, reproduces
    ``H|_beta = ones`` exactly, so the shift parameters' block comes out as exactly ``S``.

    **The input used here is the layer's own input, before normalisation.** The gradient with
    respect to the scale parameter is ``sum_t delta_t * x_hat_t``, with the *normalised* activation
    ``x_hat``, and the proof of Proposition A.1 writes ``h_{i-1}`` for that normalised activation.
    Both reference implementations nevertheless feed the forward hook's raw input here, and this
    port reproduces them. The difference is not small: measured on a ``BatchNorm2d(8)`` in train
    mode with a positive, post-ReLU-like input, ``mean_x(z_x^2)`` is 1.311 from the raw input
    against 0.126 from ``x_hat``, and the off-diagonal is 1.07 against 5e-17. Do not "fix" this in
    isolation -- for ``LayerNorm``, ``x_hat`` sums to zero across channels by construction, so
    ``z_x`` would be identically zero and the scale parameters' block would collapse to the damping
    term alone.
    """
    pooled = _pool_norm_layer(h, layer)
    z = pooled.mean(dim=1)
    return torch.stack([z, torch.ones_like(z)], dim=1)


def _h_full_norm(h: Tensor, layer: Module) -> Tensor:
    h_bar = augment_norm_input(h, layer)
    return h_bar.t() @ h_bar / h_bar.size(0)


def flatten_norm_output_grad(s: Tensor, layer: Module) -> Tensor:
    """``(T, C)`` pooled gradient batch — literally Proposition 3.1's ``S_i`` once reduced
    (``_s_full_norm``), and the raw per-sample batch ``ekfac``/``tkfac``/``tekfac`` need for their
    intra-batch estimators, row-paired with ``augment_norm_input``'s output.
    """
    return _pool_norm_layer(s, layer)


def _s_full_norm(s: Tensor, layer: Module) -> Tensor:
    s_pool = flatten_norm_output_grad(s, layer)
    return s_pool.t() @ s_pool / s_pool.size(0)
