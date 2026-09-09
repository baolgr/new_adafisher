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
