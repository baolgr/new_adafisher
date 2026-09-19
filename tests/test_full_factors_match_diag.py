"""``compute_h_full``/``compute_s_full`` (lot 2) vs. the already-validated ``compute_h_diag``/
``compute_s_diag`` (lot 1): the diagonal of the full factor must equal the diagonal statistic,
``allclose`` (not ``equal`` — matmul- vs. einsum-order reductions are not bit-identical, per
docs/reports/plan_lot2.md §0.1/§1.1's own remark). Also checks the ``bias=False`` branch.

Lot 4 (docs/reports/plan_lot4.md §2.1) adds the ``Conv2d`` (``groups=1``, ``dilation=(1,1)``) checks:
shapes, the gradient-reconstruction identity (the primary correctness anchor — a real forward/
backward, not a toy), the documented ``_h_conv2d`` legacy-scale-quirk regression (plan_lot4.md §0.2 —
``Conv2d``'s ``H`` diagonal is *not* expected to match ``compute_h_diag`` up to a constant factor
``P``, unlike every other supported layer type), and the ``groups``/``dilation`` scope guards.

Lot 5 (docs/reports/plan_lot5.md §2.1) adds the ``BatchNorm2d``/``LayerNorm`` checks: shapes, the
exact (not merely ``allclose``-to-a-tolerance) ``A[1,1] == 1`` identity (plan_lot5.md §0.2's ``a_beta``,
a literal ``mean_x(1**2)``), ``A[0,0]`` matching ``augment_norm_input``'s own pooled statistic,
symmetry, and the ``LayerNorm`` ``normalized_shape`` scope guard. There is no "diagonal of the full
factor matches ``compute_h_diag``/``compute_s_diag``" check for these two types (unlike every other
supported layer type) — plan_lot5.md §0.6 explains why the two paths do not estimate the same
quantity even up to a reduction-order difference for a normalisation layer.

The full-repository audit added three groups of checks at the bottom of this file.

* The ``Conv2d`` scale quirk **without a bias**. The existing check covers ``bias=True``, where the
  ratio is the patch width ``P``; without a bias it is exactly ``1/S``, and since every
  convolution in every ResNet and CCT in this repository is ``bias=False``, that was the branch
  with no test.
* Three layer configurations that used to be accepted silently or to fail far from their cause:
  a non-zero ``padding_mode``, a string ``padding`` (``"same"``/``"valid"``), and
  ``LayerNorm(d, bias=False)``.
* Which activation a normalisation layer's input factor is built from. It is the layer's **raw**
  input, not the normalised activation the derivative with respect to the scale parameter actually
  pairs with. That is pinned here, with the measured size of the gap, not changed — see
  ``augment_norm_input``'s own docstring for why swapping in the normalised activation is not a
  safe local fix.

The ``A[0,0]`` checks of the two normalisation-layer shape tests were computing their expected
value by calling ``augment_norm_input``, the function under test. Those assertions are kept, and an
independent recomputation from the layer's input alone is now asserted beside them.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from adafisher_modes.factors import (
    augment_conv2d_input,
    augment_conv2d_input_sua,
    augment_input,
    augment_norm_input,
    compute_h_diag,
    compute_h_full,
    compute_s_diag,
    compute_s_full,
    flatten_conv2d_output_grad,
)
from conftest import seed_all


@pytest.mark.parametrize("bias", [True, False])
def test_full_input_factor_diagonal_matches_diag_mode(bias: bool) -> None:
    seed_all(0)
    layer = nn.Linear(7, 5, bias=bias)
    h = torch.randn(64, 7)

    A = compute_h_full(h, layer)
    H_diag = compute_h_diag(h, layer)

    assert A.shape == (7 + int(bias), 7 + int(bias))
    assert torch.allclose(torch.diagonal(A), H_diag, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bias", [True, False])
def test_full_output_factor_diagonal_matches_diag_mode(bias: bool) -> None:
    seed_all(1)
    layer = nn.Linear(7, 5, bias=bias)
    s = torch.randn(64, 5)

    B = compute_s_full(s, layer)
    S_diag = compute_s_diag(s, layer)

    assert B.shape == (5, 5)
    assert torch.allclose(torch.diagonal(B), S_diag, rtol=1e-5, atol=1e-6)


def test_full_factors_are_symmetric() -> None:
    seed_all(2)
    layer = nn.Linear(7, 5, bias=True)
    h, s = torch.randn(64, 7), torch.randn(64, 5)
    A, B = compute_h_full(h, layer), compute_s_full(s, layer)
    assert torch.allclose(A, A.t())
    assert torch.allclose(B, B.t())


# --------------------------------------------------------------------------------------------
# Lot 4: Conv2d (groups=1, dilation=(1,1)). See docs/reports/plan_lot4.md §0.1-§0.4, §2.1.
# --------------------------------------------------------------------------------------------

# Matches conftest.py's tiny_batch/TinyMultiLayerNet convention: 2 input channels, 5x5 spatial.
CONV_IN, CONV_OUT, CONV_K, CONV_PAD, CONV_HW, CONV_BATCH = 2, 3, 3, 1, 5, 6


def _conv_layer(bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(CONV_IN, CONV_OUT, kernel_size=CONV_K, padding=CONV_PAD, bias=bias)


@pytest.mark.parametrize("bias", [True, False])
def test_full_input_factor_conv2d_shape(bias: bool) -> None:
    seed_all(10)
    layer = _conv_layer(bias)
    x = torch.randn(CONV_BATCH, CONV_IN, CONV_HW, CONV_HW)
    A = compute_h_full(x, layer)
    d_in_aug = CONV_IN * CONV_K * CONV_K + int(bias)
    assert A.shape == (d_in_aug, d_in_aug)
    assert torch.allclose(A, A.t())


@pytest.mark.parametrize("bias", [True, False])
def test_full_output_factor_conv2d_shape(bias: bool) -> None:
    seed_all(11)
    layer = _conv_layer(bias)
    s = torch.randn(CONV_BATCH, CONV_OUT, CONV_HW, CONV_HW)
    B = compute_s_full(s, layer)
    assert B.shape == (CONV_OUT, CONV_OUT)
    assert torch.allclose(B, B.t())


@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_pooled_patches_reconstruct_true_gradient(bias: bool) -> None:
    """plan_lot4.md §0.7b: Sum_{(n,t)} delta_{n,t} a_{n,t}^T is, by the conv backprop chain rule,
    *exactly* the weight/bias gradient — a sharp identity, not an approximation. This is the primary
    correctness anchor for the new Conv2d pooling: a wrong patch layout or a mismatched
    (example, location) pairing between augment_conv2d_input and flatten_conv2d_output_grad breaks
    it immediately, rather than only degrading a Frobenius-norm comparison.
    """
    seed_all(12)
    layer = _conv_layer(bias)
    x = torch.randn(CONV_BATCH, CONV_IN, CONV_HW, CONV_HW, requires_grad=True)

    out = layer(x)
    loss = (out * torch.randn_like(out)).sum()
    (grad_out,) = torch.autograd.grad(loss, out, retain_graph=True)
    loss.backward()

    h_bar = augment_conv2d_input(x.detach(), layer)
    s_pool = flatten_conv2d_output_grad(grad_out, layer)
    recon = s_pool.t() @ h_bar  # (C_out, P[+1])

    if bias:
        recon_w, recon_b = recon[:, :-1], recon[:, -1]
        assert torch.allclose(recon_w.reshape(layer.weight.shape), layer.weight.grad, atol=1e-4, rtol=1e-4)
        assert torch.allclose(recon_b, layer.bias.grad, atol=1e-4, rtol=1e-4)
    else:
        assert torch.allclose(recon.reshape(layer.weight.shape), layer.weight.grad, atol=1e-4, rtol=1e-4)


def test_conv2d_h_diag_scale_quirk_is_a_constant_factor() -> None:
    """plan_lot4.md §0.2: the existing, unmodified _h_conv2d divides by batch*S*P instead of
    batch*S -- verified here as a constant ratio (=P=C_in*k_h*k_w), locked as a regression rather
    than left as an untested surprise. compute_s_full has no analogous quirk (checked directly,
    matching lot 2's own Linear-case assertion).
    """
    seed_all(13)
    layer = _conv_layer(bias=True)
    x = torch.randn(CONV_BATCH, CONV_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_BATCH, CONV_OUT, CONV_HW, CONV_HW)

    A_diag = torch.diagonal(compute_h_full(x, layer))
    H_diag = compute_h_diag(x, layer)
    ratio = A_diag / H_diag
    P = CONV_IN * CONV_K * CONV_K
    assert torch.allclose(ratio, torch.full_like(ratio, float(P)), rtol=1e-4, atol=1e-4)

    B_diag = torch.diagonal(compute_s_full(s, layer))
    S_diag = compute_s_diag(s, layer)
    assert torch.allclose(B_diag, S_diag, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "kernel,stride,padding,size",
    [(3, 1, 1, 5), (2, 1, 0, 5), (3, 2, 0, 7)],
)
def test_conv2d_h_diag_scale_quirk_without_bias_is_one_over_S(
    kernel: int, stride: int, padding: int, size: int
) -> None:
    """The other branch of the same quirk -- and the one that actually runs on this repository's
    convolutional benchmarks, since every Conv2d in every ResNet and CCT here is ``bias=False``.

    ``_h_conv2d`` divides by ``batch * S * P`` when the layer has a bias (the sibling test above)
    and by ``batch`` alone when it does not, leaving out both the spatial count ``S`` and the patch
    width ``P``. The full factor divides by ``batch * S`` in both cases, so the ratio
    ``diag(compute_h_full) / compute_h_diag`` is ``P`` with a bias and exactly ``1/S`` without one,
    where ``S = H_out * W_out``. Measured at 1/25, 1/16 and 1/9 for the three shapes below.
    """
    seed_all(18)
    layer = nn.Conv2d(CONV_IN, CONV_OUT, kernel_size=kernel, stride=stride, padding=padding,
                      bias=False)
    x = torch.randn(CONV_BATCH, CONV_IN, size, size)
    out_size = (size + 2 * padding - kernel) // stride + 1
    spatial = out_size * out_size

    ratio = torch.diagonal(compute_h_full(x, layer)) / compute_h_diag(x, layer)
    assert torch.allclose(ratio, torch.full_like(ratio, 1.0 / spatial), rtol=1e-5, atol=1e-8), (
        f"expected a constant ratio of 1/S = 1/{spatial}, got min {ratio.min():.8f} "
        f"max {ratio.max():.8f}"
    )
    # and it is emphatically not the with-bias value, which is what the sibling test pins
    assert not torch.allclose(ratio, torch.full_like(ratio, float(CONV_IN * kernel * kernel)))


def test_conv2d_groups_not_supported() -> None:
    layer = nn.Conv2d(4, 4, kernel_size=3, groups=2)
    h = torch.randn(2, 4, 5, 5)
    with pytest.raises(NotImplementedError):
        compute_h_full(h, layer)
    with pytest.raises(NotImplementedError):
        augment_input(h, layer)


def test_conv2d_dilation_not_supported() -> None:
    layer = nn.Conv2d(2, 2, kernel_size=3, dilation=2)
    h = torch.randn(2, 2, 7, 7)
    with pytest.raises(NotImplementedError):
        compute_h_full(h, layer)
    with pytest.raises(NotImplementedError):
        augment_input(h, layer)


# --------------------------------------------------------------------------------------------
# Lot 5: BatchNorm2d / LayerNorm (normalized_shape a 1-tuple). See docs/reports/plan_lot5.md
# §0.1-§0.4, §2.1.
# --------------------------------------------------------------------------------------------


def test_full_input_factor_batchnorm2d_shape_and_a_beta_exact() -> None:
    seed_all(30)
    layer = nn.BatchNorm2d(3)
    h = torch.randn(4, 3, 5, 5)

    A = compute_h_full(h, layer)
    assert A.shape == (2, 2)
    assert torch.allclose(A, A.t())
    # plan_lot5.md §0.2: a_beta = mean_x(1**2) = 1 exactly, not merely close to a tolerance.
    assert A[1, 1].item() == 1.0

    z = augment_norm_input(h, layer)[:, 0]
    assert torch.allclose(A[0, 0], (z**2).mean())

    # The assertion above compares the function against itself: both sides go through
    # augment_norm_input. So compute the same three entries here, from the layer's input alone,
    # without calling augment_norm_input or _pool_norm_layer. A[0,0] is the mean over positions of
    # the squared channel-mean, A[0,1] the mean of the channel-mean, A[1,1] the mean of 1.
    z_independent = h.mean(dim=1).reshape(-1)  # (N*H*W,), channels pooled at each position
    assert torch.allclose(A[0, 0], (z_independent**2).mean(), rtol=1e-6, atol=0)
    assert torch.allclose(A[0, 1], z_independent.mean(), rtol=1e-6, atol=0)
    assert torch.allclose(A[1, 0], A[0, 1], rtol=0, atol=0)
    # and the row order is the one the gradient side pools in, not merely the same multiset
    assert torch.equal(z, z_independent)


def test_full_output_factor_batchnorm2d_shape_and_symmetry() -> None:
    seed_all(31)
    layer = nn.BatchNorm2d(3)
    s = torch.randn(4, 3, 5, 5)

    B = compute_s_full(s, layer)
    assert B.shape == (3, 3)
    assert torch.allclose(B, B.t())


def test_full_input_factor_layernorm_shape_and_a_beta_exact() -> None:
    seed_all(32)
    layer = nn.LayerNorm(8)
    h = torch.randn(4, 8)

    A = compute_h_full(h, layer)
    assert A.shape == (2, 2)
    assert torch.allclose(A, A.t())
    assert A[1, 1].item() == 1.0

    z = augment_norm_input(h, layer)[:, 0]
    assert torch.allclose(A[0, 0], (z**2).mean())

    # Same independent recomputation as the BatchNorm2d case above: no call to
    # augment_norm_input or _pool_norm_layer on this side of the comparison. LayerNorm's channel
    # axis is already last for a 1-D normalized_shape, so pooling it is a mean over dim -1.
    z_independent = h.mean(dim=-1).reshape(-1)
    assert torch.allclose(A[0, 0], (z_independent**2).mean(), rtol=1e-6, atol=0)
    assert torch.allclose(A[0, 1], z_independent.mean(), rtol=1e-6, atol=0)
    assert torch.allclose(A[1, 0], A[0, 1], rtol=0, atol=0)
    assert torch.equal(z, z_independent)


def test_full_output_factor_layernorm_shape_and_symmetry() -> None:
    seed_all(33)
    layer = nn.LayerNorm(8)
    s = torch.randn(4, 8)

    B = compute_s_full(s, layer)
    assert B.shape == (8, 8)
    assert torch.allclose(B, B.t())


def test_layernorm_multi_dim_normalized_shape_not_supported() -> None:
    layer = nn.LayerNorm((4, 5))
    h = torch.randn(2, 4, 5)
    with pytest.raises(NotImplementedError):
        compute_h_full(h, layer)
    with pytest.raises(NotImplementedError):
        compute_s_full(h, layer)
    with pytest.raises(NotImplementedError):
        augment_input(h, layer)


# --------------------------------------------------------------------------------------------
# Lot 6: the SUA approximation, Conv2d input factor only. See docs/reports/plan_lot6.md
# §0.1-§0.3, §2.1.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bias", [True, False])
def test_augment_conv2d_input_sua_shape_with_without_bias(bias: bool) -> None:
    seed_all(14)
    layer = _conv_layer(bias)
    x = torch.randn(CONV_BATCH, CONV_IN, CONV_HW, CONV_HW)
    h_bar = augment_conv2d_input_sua(x, layer)
    assert h_bar.shape == (CONV_BATCH * CONV_HW * CONV_HW, CONV_IN + int(bias))


def test_augment_conv2d_input_sua_center_slice_equals_raw_pixel() -> None:
    """plan_lot6.md §0.3: for padding=(k-1)/2, stride=1 (this file's own CONV_PAD=1, CONV_K=3,
    matching every Conv2d fixture in this codebase), the center offset of every patch is *exactly*
    the corresponding raw input pixel -- no padding-zero ever contributes, at any output location,
    including boundary ones. The concrete, numeric version of §0.3's derivation.
    """
    seed_all(15)
    layer = _conv_layer(bias=True)
    x = torch.randn(CONV_BATCH, CONV_IN, CONV_HW, CONV_HW)

    h_bar = augment_conv2d_input_sua(x, layer)
    raw_pixels = x.permute(0, 2, 3, 1).reshape(-1, CONV_IN)  # (N*H*W, C_in), same (N,H,W) order
    assert torch.equal(h_bar[:, :-1], raw_pixels)
    assert torch.equal(h_bar[:, -1], torch.ones(h_bar.size(0)))


def test_augment_conv2d_input_sua_row_alignment_with_output_grad() -> None:
    """plan_lot6.md §0.3: row count matches flatten_conv2d_output_grad's own (N*H_out*W_out) count
    even when H_in != H_out (stride=2, no padding here) -- the case EKFAC-pytorch's own "pool the
    raw input independently" SUA construction would misalign.
    """
    seed_all(16)
    layer = nn.Conv2d(CONV_IN, CONV_OUT, kernel_size=CONV_K, stride=2, padding=0, bias=True)
    x = torch.randn(CONV_BATCH, CONV_IN, 7, 7)
    out = layer(x)
    s = torch.randn_like(out)

    h_bar = augment_conv2d_input_sua(x, layer)
    s_pool = flatten_conv2d_output_grad(s, layer)
    assert h_bar.size(0) == s_pool.size(0)


def test_augment_conv2d_input_sua_matches_full_for_1x1_kernel() -> None:
    """plan_lot6.md §0.6: with a 1x1 kernel, patches already are channel-only, so SUA and the
    full patch-based input factor coincide exactly.
    """
    seed_all(17)
    layer = nn.Conv2d(CONV_IN, CONV_OUT, kernel_size=1, bias=True)
    x = torch.randn(CONV_BATCH, CONV_IN, CONV_HW, CONV_HW)
    assert torch.equal(augment_conv2d_input_sua(x, layer), augment_conv2d_input(x, layer))


def test_augment_conv2d_input_sua_groups_dilation_not_supported() -> None:
    h = torch.randn(2, 4, 5, 5)
    with pytest.raises(NotImplementedError):
        augment_conv2d_input_sua(h, nn.Conv2d(4, 4, kernel_size=3, groups=2))
    h = torch.randn(2, 2, 7, 7)
    with pytest.raises(NotImplementedError):
        augment_conv2d_input_sua(h, nn.Conv2d(2, 2, kernel_size=3, dilation=2))


# --------------------------------------------------------------------------------------------
# Scope guards added after the full-repository audit: three layer configurations that used to be
# accepted silently, or to fail with an error naming neither the layer nor the reason.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("padding_mode", ["reflect", "replicate", "circular"])
def test_conv2d_non_zero_padding_mode_not_supported(padding_mode: str) -> None:
    """``extract_patches`` always pads with zeros, so the factor of a reflect-, replicate- or
    circular-padded convolution used to describe a different layer from the one being trained.
    Measured on a 2-channel 3x3 layer with ``padding_mode='reflect'``: the zero-padded input factor
    is 48% off the reflect-padded one in relative Frobenius norm. It is refused now.
    """
    layer = nn.Conv2d(CONV_IN, CONV_OUT, kernel_size=CONV_K, padding=CONV_PAD,
                      padding_mode=padding_mode)
    h = torch.randn(CONV_BATCH, CONV_IN, CONV_HW, CONV_HW)
    for call in (compute_h_full, augment_input):
        with pytest.raises(NotImplementedError, match="padding_mode"):
            call(h, layer)
    with pytest.raises(NotImplementedError, match="padding_mode"):
        augment_conv2d_input_sua(h, layer)


def test_conv2d_zeros_padding_mode_is_still_accepted() -> None:
    """The guard must not refuse the default, which is what every model in this repository uses."""
    layer = nn.Conv2d(CONV_IN, CONV_OUT, kernel_size=CONV_K, padding=CONV_PAD)
    assert layer.padding_mode == "zeros"
    h = torch.randn(CONV_BATCH, CONV_IN, CONV_HW, CONV_HW)
    assert compute_h_full(h, layer).shape == (CONV_IN * CONV_K * CONV_K + 1,) * 2


@pytest.mark.parametrize("padding", ["same", "valid"])
def test_conv2d_string_padding_not_supported(padding: str) -> None:
    """``Conv2d`` accepts ``padding='same'`` and ``padding='valid'`` and keeps the string verbatim
    in ``layer.padding``. Arithmetic on it used to raise ``TypeError: '>' not supported between
    instances of 'str' and 'int'`` from inside the patch extraction, in **both** the diagonal and
    the full-factor paths, naming neither the layer nor the option.
    """
    layer = nn.Conv2d(CONV_IN, CONV_OUT, kernel_size=CONV_K, padding=padding)
    h = torch.randn(CONV_BATCH, CONV_IN, CONV_HW, CONV_HW)
    for call in (compute_h_full, compute_h_diag, augment_input):
        with pytest.raises(NotImplementedError, match=f"padding={padding!r}"):
            call(h, layer)


def test_layernorm_without_bias_not_supported() -> None:
    """``LayerNorm(d, bias=False)`` has a scale parameter and no shift. Every input factor here
    appends a constant-one column for the shift, so the 2 x 2 matrix would describe a parameter the
    layer does not have.
    """
    layer = nn.LayerNorm(8, bias=False)
    h = torch.randn(4, 8)
    for call in (compute_h_full, compute_s_full, augment_input, augment_norm_input):
        with pytest.raises(NotImplementedError, match="bias=False"):
            call(h, layer)


def test_layernorm_with_its_default_bias_is_still_accepted() -> None:
    layer = nn.LayerNorm(8)
    assert layer.bias is not None
    assert compute_h_full(torch.randn(4, 8), layer).shape == (2, 2)


# --------------------------------------------------------------------------------------------
# Which activation a normalisation layer's input factor is built from. Pinned, not fixed --
# see the "deliberately not changed" note in augment_norm_input's own docstring.
# --------------------------------------------------------------------------------------------


def test_norm_input_factor_ignores_the_normalisation_itself() -> None:
    """A normalisation layer computes ``y = gamma * x_hat + beta``, so the derivative with respect
    to ``gamma`` pairs the output gradient with the **normalised** activation ``x_hat``. The
    forward hook sees the layer's input, before normalisation, and that is what is pooled -- as in
    both reference implementations.

    The discriminating check: drive a real ``BatchNorm2d`` through real hooks, on the same input,
    once in train mode and once in eval mode. ``x_hat`` uses batch statistics in train and running
    statistics in eval, and here the two are deliberately far apart. The **input** factor comes out
    bit-identical, which is only possible if it never looks at the normalisation; the **output**
    factor, which does depend on it, differs. Measured on ``BatchNorm2d(8)`` with a post-ReLU
    input: ``A[0,0]`` is 4.3991 from the raw input against 0.1296 from the train-mode ``x_hat``, a
    factor of 33.9, and the scale-shift coupling entry ``A[0,1]`` is 2.0377 against 4.3e-08.
    """
    from adafisher_modes import AdaFisherMulti  # noqa: PLC0415

    seed_all(40)
    factors = {}
    for training in (True, False):
        seed_all(40)
        model = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.BatchNorm2d(8))
        with torch.no_grad():  # running statistics deliberately far from any batch's
            model[1].running_mean.fill_(-3.0)
            model[1].running_var.fill_(9.0)
        opt = AdaFisherMulti(model, lr=0.0, fisher_mode="kfac", TCov=1, T_inv=1)
        model.train(training)
        x = torch.relu(torch.randn(8, 2, 6, 6, generator=torch.Generator().manual_seed(2)) + 1.0)
        opt.zero_grad()
        model(x).pow(2).sum().backward()
        bn = model[1]
        factors[training] = (opt.approx._A[bn].clone(), opt.approx._B[bn].clone())

    A_train, B_train = factors[True]
    A_eval, B_eval = factors[False]
    assert torch.equal(A_train, A_eval), (
        "the input factor changed with the normalisation mode, so it is no longer built from the "
        "raw input -- which is a change of definition, not a refactor"
    )
    assert not torch.allclose(B_train, B_eval), (
        "the output factor should depend on the normalisation; if it does not, this test is no "
        "longer discriminating"
    )


def test_norm_input_factor_coupling_entry_is_far_from_the_normalised_value() -> None:
    """The same finding as a number rather than an invariance. Built from ``x_hat``, the pooled
    channel-mean is zero by construction, so ``A[0,1]`` would be ~0 and ``A[0,0]`` would collapse:
    measured at 6.3e-15 for ``LayerNorm(16)``, which is exactly the reason swapping in ``x_hat``
    is not a safe local fix -- the whole scale block would become the damping term alone.
    """
    seed_all(41)
    layer = nn.LayerNorm(16)
    h = torch.relu(torch.randn(32, 16) * 1.5 + 2.0)
    A_raw = compute_h_full(h, layer)

    x_hat = (h - h.mean(-1, keepdim=True)) / (h.var(-1, unbiased=False, keepdim=True) + layer.eps).sqrt()
    A_hat = compute_h_full(x_hat, layer)

    assert A_raw[0, 1].abs() > 1.0, f"expected a large coupling entry, got {A_raw[0, 1].item()}"
    assert A_hat[0, 1].abs() < 1e-6
    assert A_hat[0, 0].abs() < 1e-6, (
        "LayerNorm's x_hat sums to zero across channels, so an x_hat-based scale block is exactly "
        "the damping term -- the documented reason this is not swapped in"
    )
    assert A_raw[0, 0] > 1.0
