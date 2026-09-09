"""Per-layer-type extraction of the diagonal Kronecker-factor statistics H_D, S_D.

This is a line-for-line port of ``_ComputeHBarD`` / ``_ComputeSD`` and ``_extract_patches`` from
``reference_repos/FisherAdapTune/scripts/adafisher.py`` (lines 33-155), kept bit-exact on purpose:
``tests/test_diag_bitexact.py`` checks ``torch.equal`` against that reference, and any deviation in
reduction order (e.g. building the full Kronecker factor and slicing its diagonal instead of using
``einsum`` directly) can change the floating-point result even though it is mathematically identical.

Scope note (lot 1): only the diagonal path is implemented. The full (non-diagonal) factors needed by
K-FAC / EKFAC / TKFAC / TEKFAC are introduced in lot 2, as siblings of ``compute_h_diag`` /
``compute_s_diag`` that reuse ``extract_patches``. See docs/reports/plan_lot1.md §1.

Lot 2 adds ``augment_linear_input``, ``compute_h_full`` and ``compute_s_full`` for ``Linear`` only —
the full ``A = E[h_bar h_bar^T]`` / ``B = E[delta delta^T]`` factors K-FAC and EKFAC need. ``Conv2d``,
``BatchNorm2d`` and ``LayerNorm`` are lots 4-5 (docs/reports/plan_lot2.md §0.1): both functions raise
``NotImplementedError`` for those types rather than silently falling back to something incorrect.

Lot 4 adds the ``Conv2d`` branch (``groups=1``, ``dilation=(1,1)`` only — docs/reports/plan_lot4.md
§0.4), reusing ``extract_patches`` unmodified: every ``(example, output-location)`` pair becomes one
pooled i.i.d. sample, exactly generalising the diagonal path's own pooling (§0.1). ``augment_input``
and ``flatten_output_grad`` are new, layer-dispatching public helpers — the raw-batch analogues of
``compute_h_full``/``compute_s_full`` that ``ekfac``/``tkfac``/``tekfac`` need (they cache the raw
augmented batch, not just its reduction; see plan_lot2.md §0.3). See plan_lot4.md §0.2 for a scale
discrepancy found in the existing, *unmodified* ``_h_conv2d`` diagonal path, deliberately not
reproduced by the new ``compute_h_full`` ``Conv2d`` branch below.

Lot 5 adds ``BatchNorm2d``/``LayerNorm`` (``normalized_shape`` a 1-tuple only — docs/reports/
plan_lot5.md §0.4). Proposition 3.1's exact FIM for a normalisation layer is *Hadamard*-, not
Kronecker-, structured (``FIM_nu = H|_nu (Hadamard) S``, ``FIM_beta = S`` exactly — plan_lot5.md
§0.1); ``_h_full_norm``/``augment_norm_input`` build a ``2x2`` factor ``A = diag-ish(a_nu, 1)`` (a
Frobenius-optimal, ``S``-independent scalar surrogate for the exact, full ``H|_nu``, plus a small,
deliberately-kept, honest ``nu``-``beta`` coupling term — plan_lot5.md §0.2) so that the *same*
``kron(A, B)`` machinery every other layer type already uses applies with zero changes to any
``approximations/*.py`` file.

Lot 6 adds the SUA approximation for ``Conv2d`` (``kfac_conv_1602.01407.pdf`` p. 14, "spatially
uncorrelated activations" — docs/reports/plan_lot6.md §0.1-§0.3): ``augment_conv2d_input_sua``/
``_h_full_conv2d_sua`` replace the patch-based input factor (``C_in*k_h*k_w[+1]`` wide) by a
channel-only one (``C_in[+1]`` wide), pooling the *center* offset of every patch ``extract_patches``
already produces rather than the whole patch — row-aligned with ``flatten_conv2d_output_grad``'s own
pooling by construction (plan_lot6.md §0.3), for any ``stride``/``padding``. ``compute_h_full``/
``augment_input`` gain an optional ``sua`` flag, consulted only on their ``Conv2d`` branch; the
output-factor side (``compute_s_full``/``flatten_output_grad``) is untouched, since SUA only concerns
the input factor (plan_lot6.md §0.2).
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

    Port of ``_extract_patches`` (adafisher.py:33-50).
    """
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
# Lot 2: full (non-diagonal) factors, Linear only. See docs/reports/plan_lot2.md §0.1, §1.1.
# ----------------------------------------------------------------------------------------------


def augment_linear_input(h: Tensor, layer: Linear) -> Tensor:
    """Flatten ``h`` to ``(N, d_in)`` and, iff ``layer`` has a bias, append a ones column: the
    bias-augmented ``h_bar`` batch of AdaFisher's header formula. Public (not just an internal
    helper of ``compute_h_full``) because ``EKFACApproximation`` also needs the raw augmented batch
    itself, not only its reduction — see plan_lot2.md §0.3.
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
    """Instantaneous full input factor A = E[h_bar h_bar^T] (Linear, Conv2d with groups=1 and
    dilation=(1,1) — lot 4, docs/reports/plan_lot4.md §1.1; BatchNorm2d, LayerNorm with a 1-D
    normalized_shape — lot 5, docs/reports/plan_lot5.md §1.1). ``sua`` (lot 6, docs/reports/
    plan_lot6.md §0.3) is consulted only on the Conv2d branch: it selects the channel-only SUA
    input factor instead of the patch-based one; meaningless (silently ignored) for every other
    layer type, exactly like ``pi``/``T_inv`` are meaningless for layer types they don't apply to.
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
    """Instantaneous full output factor B = E[delta delta^T] (Linear, Conv2d with groups=1 and
    dilation=(1,1) — lot 4, docs/reports/plan_lot4.md §1.1; BatchNorm2d, LayerNorm with a 1-D
    normalized_shape — lot 5, docs/reports/plan_lot5.md §1.1). For a normalisation layer this is
    literally Proposition 3.1's S_i (full-rank, "square-then-sum"), no approximation involved.
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
# Lot 4: full (non-diagonal) factors, Conv2d (groups=1, dilation=(1,1)). See
# docs/reports/plan_lot4.md §0.1-§0.4, §1.1.
# ----------------------------------------------------------------------------------------------


def _check_conv2d_supported(layer: Conv2d) -> None:
    """Explicit, typed scope guard (plan_lot4.md §0.4) rather than silently computing a Kronecker
    factor that ignores structural weight sparsity (``groups``) or an unfolding that ``extract_patches``
    cannot express (``dilation``)."""
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


def augment_conv2d_input(h: Tensor, layer: Conv2d) -> Tensor:
    """Pool (batch, output-location) patches onto one axis and, iff ``layer`` has a bias, append a
    ones column: the Conv2d analogue of ``augment_linear_input``, built on ``extract_patches``
    (lot 1, unmodified). Each pooled row is treated as an i.i.d. sample of the bias-augmented
    receptive-field patch, under the same spatial-independence reading the diagonal path already
    relies on implicitly (TKFAC Assumption 4.1, ``tkfac_2011.10741.pdf`` §4.3 — see
    docs/reports/plan_lot4.md §0.1).
    """
    _check_conv2d_supported(layer)
    patches = extract_patches(h, layer.kernel_size, layer.stride, layer.padding, layer.groups)
    patches = patches.reshape(-1, patches.size(-1))  # (N*S, P): groups=1, so dim 1 is trivial
    if layer.bias is not None:
        return cat([patches, patches.new_ones(patches.size(0), 1)], 1)
    return patches


def augment_input(h: Tensor, layer: Module, sua: bool = False) -> Tensor:
    """Dispatching sibling of ``augment_linear_input``/``augment_conv2d_input``/``augment_norm_input``
    (docs/reports/plan_lot4.md §0.5, plan_lot5.md §0.5): used directly by ``ekfac``/``tkfac``/
    ``tekfac``, which need the raw augmented batch, not just ``compute_h_full``'s reduction
    (plan_lot2.md §0.3). ``sua`` (lot 6, docs/reports/plan_lot6.md §0.3) selects
    ``augment_conv2d_input_sua`` in place of ``augment_conv2d_input`` on the Conv2d branch only.
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
    reference, verified and deliberately not reproduced here — docs/reports/plan_lot4.md §0.2).
    """
    h_bar = augment_conv2d_input(h, layer)
    return h_bar.t() @ h_bar / h_bar.size(0)


def flatten_conv2d_output_grad(s: Tensor, layer: Conv2d) -> Tensor:
    """Pool (batch, output-location) gradients onto one axis: the same transpose/reshape sequence
    as ``_s_conv2d`` (unmodified), minus its diagonal reduction. Unlike ``_h_conv2d``
    (``augment_conv2d_input``'s docstring / plan_lot4.md §0.2), ``_s_conv2d``'s scale was already
    correct — its ``spatial_size`` is computed before this reshape — so no deviation is introduced
    here.
    """
    return s.transpose(1, 2).transpose(2, 3).reshape(-1, s.size(1))


def flatten_output_grad(s: Tensor, layer: Module) -> Tensor:
    """Dispatching sibling, output-gradient side (docs/reports/plan_lot4.md §0.5, plan_lot5.md §0.5).
    The ``Linear`` branch is the same reshape ``ekfac``/``tkfac``/``tekfac`` each inlined before lot
    4; extracted here now that ``Conv2d``/normalisation-layer branches must exist alongside it.
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
# Lot 6: the SUA approximation, Conv2d input factor only (groups=1, dilation=(1,1), reusing lot
# 4's own scope guard). See docs/reports/plan_lot6.md §0.1-§0.3, §1.1.
# ----------------------------------------------------------------------------------------------


def augment_conv2d_input_sua(h: Tensor, layer: Conv2d) -> Tensor:
    """Channel-only analogue of ``augment_conv2d_input`` (SUA, ``kfac_conv_1602.01407.pdf`` p. 14):
    pools the *center* offset of every receptive-field patch ``extract_patches`` produces, instead
    of the whole patch, dropping the input factor from ``(C_in*k_h*k_w[+1])^2`` to ``(C_in[+1])^2``
    entries (docs/reports/plan_lot6.md §0.3). Row-aligned with ``flatten_conv2d_output_grad``'s
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
# Lot 5: full (non-diagonal) factors, BatchNorm2d / LayerNorm (normalized_shape a 1-tuple). See
# docs/reports/plan_lot5.md §0.1-§0.4, §1.1.
# ----------------------------------------------------------------------------------------------


def _pool_batchnorm2d(x: Tensor) -> Tensor:
    """``(N, C, H, W) -> (N*H*W, C)``: move the channel axis (dim 1, ``BatchNorm2d``'s own
    convention, matching ``_h_batchnorm2d``/``_s_batchnorm2d``) last and flatten batch+spatial onto
    one axis — Proposition 3.1's ``T_i = N*H*W`` (docs/reports/plan_lot5.md §0.3).
    """
    return x.permute(0, 2, 3, 1).reshape(-1, x.size(1))


def _check_layernorm_supported(layer: LayerNorm) -> None:
    """Explicit, typed scope guard (plan_lot5.md §0.4), mirroring lot 4's ``groups``/``dilation``
    guards: a multi-dimensional ``normalized_shape`` would need an axis convention this lot does not
    define (and that the *existing* ``_h_layernorm``/``_s_layernorm`` port does not consistently
    define either, for ``h.ndim > 2`` — plan_lot5.md §0.3)."""
    if len(layer.normalized_shape) != 1:
        raise NotImplementedError(
            f"Full Kronecker factors for LayerNorm only support a 1-D normalized_shape so far "
            f"(lot 5 scope); got normalized_shape={tuple(layer.normalized_shape)}."
        )


def _pool_layernorm(x: Tensor, layer: LayerNorm) -> Tensor:
    """``(..., C) -> (T, C)``: channel is already the last axis for a 1-D ``normalized_shape``
    (PyTorch's own convention), so this is a plain flatten, mirroring ``_h_linear``/``_s_linear``.
    """
    _check_layernorm_supported(layer)
    return x.reshape(-1, x.size(-1))


def _pool_norm_layer(x: Tensor, layer: Module) -> Tensor:
    """Dispatching sibling: the ``(T, C)`` pooled view shared by the H and S sides of a
    normalisation layer's full-factor construction (docs/reports/plan_lot5.md §0.3)."""
    if isinstance(layer, BatchNorm2d):
        return _pool_batchnorm2d(x)
    if isinstance(layer, LayerNorm):
        return _pool_layernorm(x, layer)
    raise NotImplementedError(
        f"_pool_norm_layer only supports BatchNorm2d and LayerNorm; got {type(layer)}"
    )


def augment_norm_input(h: Tensor, layer: Module) -> Tensor:
    """``(T, 2)`` matrix ``[z_x, 1]``, the normalisation-layer analogue of
    ``augment_linear_input``/``augment_conv2d_input``. ``z_x`` is the per-position channel-mean
    pre-activation — the Frobenius-optimal, ``S``-independent scalar surrogate
    (docs/reports/plan_lot5.md §0.2) for Proposition 3.1's exact ``H_{i-1}|_{nu_i}`` (a full ``C x C``
    matrix, Hadamard- not Kronecker-combined with ``S`` there — plan_lot5.md §0.1), fit into this
    codebase's shared ``kron(A, B)`` machinery. Column 1 (the constant) reproduces
    ``H_{i-1}|_{beta_i} = 11^T``'s exact contribution (``= S``) once run through that same machinery.
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
    intra-batch estimators, row-paired with ``augment_norm_input``'s output (plan_lot5.md §0.3).
    """
    return _pool_norm_layer(s, layer)


def _s_full_norm(s: Tensor, layer: Module) -> Tensor:
    s_pool = flatten_norm_output_grad(s, layer)
    return s_pool.t() @ s_pool / s_pool.size(0)
