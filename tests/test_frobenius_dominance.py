"""Exit criteria 1 and 2 of lot 2 (docs/reports/plan_lot2.md §2.2, docs/reports/plan.md §6.1):

1. ``||F - F~_EKFAC||_F <= ||F - F~_KFAC||_F``          (EKFAC Thm 2/3, Appendix A.1)
2. ``Q_A``, ``Q_B`` orthogonal.

``F`` is the exact per-layer Fisher block, computable by per-sample accumulation on a toy
dimension: no real network or autograd is needed, since ``h`` and ``s`` are exactly what
``update_input_factor``/``update_output_factor`` consume — they are sampled directly. A throwaway
``nn.Linear`` supplies ``module`` (hook-dict key, ``.bias is not None`` check) and is never
forwarded.

Convention (plan_lot2.md §0.4): per-example ``vec_r(delta_n h_bar_n^T) = delta_n (x) h_bar_n``,
hence ``F = E_n[(delta_n delta_n^T) (x) (h_bar_n h_bar_n^T)]`` — B outer, A inner. Both
``KFACApproximation.f_tilde``/``EKFACApproximation.f_tilde`` use this same ordering, so the
dominance comparison is self-consistent even if this were not "the" literature convention (see the
internal-consistency check below, which verifies ``f_tilde`` and ``precondition`` agree
independently of that).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from adafisher_modes.approximations.ekfac import EKFACApproximation
from adafisher_modes.approximations.kfac import KFACApproximation
from adafisher_modes.approximations.tekfac import TEKFACApproximation
from adafisher_modes.approximations.tkfac import TKFACApproximation
from adafisher_modes.factors import augment_conv2d_input, augment_conv2d_input_sua
from conftest import seed_all

D_IN, D_OUT, N = 7, 5, 64
D_IN_AUG = D_IN + 1  # bias-augmented


def _exact_fisher_block(h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """F = E_n[ (s_n s_n^T) (x) (h_bar_n h_bar_n^T) ], by explicit per-sample accumulation."""
    F = torch.zeros(D_OUT * D_IN_AUG, D_OUT * D_IN_AUG)
    for n in range(h.size(0)):
        h_bar_n = torch.cat([h[n], h.new_ones(1)])
        F += torch.kron(torch.outer(s[n], s[n]), torch.outer(h_bar_n, h_bar_n))
    return F / h.size(0)


def _build_kfac_and_ekfac(h: torch.Tensor, s: torch.Tensor, Lambda: float = 1e-8):
    layer = nn.Linear(D_IN, D_OUT, bias=True)

    # gammas=(1.0, 1.0) makes update_running_avg an exact assignment (current <- new), regardless
    # of the eye/ones bootstrap — the cleanest way to get EMA-artifact-free factors out of the real,
    # stateful classes (plan_lot2.md §2.2), rather than a parallel pure-math reimplementation.
    kfac = KFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_inv=1, pi=True)
    kfac.update_input_factor(layer, h, step=0)
    kfac.update_output_factor(layer, s, step=0)
    kfac.refresh(layer, step=0)

    ekfac = EKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_eig=1)
    ekfac.update_input_factor(layer, h, step=0)
    ekfac.update_output_factor(layer, s, step=0)  # no eigenbasis yet -> s* not estimated
    ekfac.refresh(layer, step=0)  # Q_A, Q_B from the exact A, B; s* bootstrapped to ones
    # Re-drive the SAME (h, s) samples now that the eigenbasis exists, so s* is estimated from
    # exactly the data that defines F (the theorem is about one closed data-generating process —
    # plan_lot2.md §2.2's explicit warning against estimating s* from a different batch).
    ekfac.update_input_factor(layer, h, step=1)
    ekfac.update_output_factor(layer, s, step=1)

    return layer, kfac, ekfac


def test_ekfac_dominates_kfac_in_frobenius_norm() -> None:
    seed_all(0)
    h, s = torch.randn(N, D_IN), torch.randn(N, D_OUT)

    F_exact = _exact_fisher_block(h, s)
    layer, kfac, ekfac = _build_kfac_and_ekfac(h, s)

    F_kfac = kfac.f_tilde(layer)
    F_ekfac = ekfac.f_tilde(layer)

    err_kfac = torch.linalg.matrix_norm(F_exact - F_kfac)
    err_ekfac = torch.linalg.matrix_norm(F_exact - F_ekfac)

    assert err_ekfac <= err_kfac + 1e-4, (
        f"EKFAC Thm 2/3 violated: ||F-F~_EKFAC||_F={err_ekfac:.6f} > "
        f"||F-F~_KFAC||_F={err_kfac:.6f}"
    )


def test_eigenbases_are_orthogonal() -> None:
    seed_all(1)
    h, s = torch.randn(N, D_IN), torch.randn(N, D_OUT)
    layer, _kfac, ekfac = _build_kfac_and_ekfac(h, s)

    Q_A, Q_B = ekfac._Q_A[layer], ekfac._Q_B[layer]
    assert torch.allclose(Q_A.t() @ Q_A, torch.eye(D_IN_AUG), atol=1e-5)
    assert torch.allclose(Q_B.t() @ Q_B, torch.eye(D_OUT), atol=1e-5)


def test_f_tilde_matches_precondition_for_kfac() -> None:
    """Internal consistency: precondition()'s factored application and f_tilde()'s dense
    reconstruction must agree, independently of the §0.4 rvec/cvec convention being "the" one in
    the literature — this is what would actually break if they disagreed on ordering.
    """
    seed_all(2)
    h, s = torch.randn(N, D_IN), torch.randn(N, D_OUT)
    layer, kfac, _ekfac = _build_kfac_and_ekfac(h, s)

    weight_direction = torch.randn(D_OUT, D_IN)
    bias_direction = torch.randn(D_OUT)
    w_out, b_out = kfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction, bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(kfac.f_tilde(layer), M.flatten()).reshape(D_OUT, D_IN_AUG)

    got = torch.cat([w_out, b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


def test_f_tilde_matches_precondition_for_ekfac() -> None:
    seed_all(3)
    h, s = torch.randn(N, D_IN), torch.randn(N, D_OUT)
    layer, _kfac, ekfac = _build_kfac_and_ekfac(h, s)

    weight_direction = torch.randn(D_OUT, D_IN)
    bias_direction = torch.randn(D_OUT)
    w_out, b_out = ekfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction, bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(ekfac.f_tilde(layer), M.flatten()).reshape(D_OUT, D_IN_AUG)

    got = torch.cat([w_out, b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


def test_measured_rescaled_error_report(capsys) -> None:
    """Informational only (plan.md §6.1: "measured, not asserted"): report e(F~) = min_c ||F -
    c*F~||_F for kfac and ekfac via the closed form c* = <F, F~>/||F~||_F^2. No assertion — this is
    a reported measurement, not a pass/fail gate.
    """
    seed_all(4)
    h, s = torch.randn(N, D_IN), torch.randn(N, D_OUT)
    F_exact = _exact_fisher_block(h, s)
    layer_a, kfac, ekfac = _build_kfac_and_ekfac(h, s)
    layer_b, tkfac, tekfac = _build_tkfac_and_tekfac(h, s)

    for name, F_tilde in (
        ("kfac", kfac.f_tilde(layer_a)),
        ("ekfac", ekfac.f_tilde(layer_a)),
        ("tkfac", tkfac.f_tilde(layer_b)),
        ("tekfac", tekfac.f_tilde(layer_b)),
    ):
        c_star = torch.sum(F_exact * F_tilde) / torch.sum(F_tilde * F_tilde)
        e = torch.linalg.matrix_norm(F_exact - c_star * F_tilde)
        print(f"e(F~_{name}) after optimal rescaling (c*={c_star:.4f}) = {e:.6f}")


# --------------------------------------------------------------------------------------------
# Lot 3: tkfac / tekfac (docs/reports/plan_lot3.md §2.2). Exit criteria 1 (trace preservation,
# TKFAC Thm 4.1 / Lemma 4.1) and 2 (Frobenius dominance, TEKFAC Thm 3.1).
# --------------------------------------------------------------------------------------------


def _build_tkfac_and_tekfac(h: torch.Tensor, s: torch.Tensor, Lambda: float = 1e-8):
    layer = nn.Linear(D_IN, D_OUT, bias=True)

    tkfac = TKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_inv=1)
    tkfac.update_input_factor(layer, h, step=0)
    tkfac.update_output_factor(layer, s, step=0)
    tkfac.refresh(layer, step=0)

    tekfac = TEKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_eig=1, T_re=1)
    tekfac.update_input_factor(layer, h, step=0)
    tekfac.update_output_factor(layer, s, step=0)  # no eigenbasis yet -> Theta not touched
    tekfac.refresh(layer, step=0)  # Q_Phi, Q_Psi from the exact Phi_raw/Psi_raw; Theta bootstrapped
    # Re-drive the SAME (h, s) now that the eigenbasis exists, so Theta is estimated from exactly
    # the data that defines F (plan_lot2.md §2.2's own warning against a mismatched batch).
    tekfac.update_input_factor(layer, h, step=1)
    tekfac.update_output_factor(layer, s, step=1)

    return layer, tkfac, tekfac


def test_tkfac_trace_matches_exact_fisher() -> None:
    """TKFAC Thm 4.1 / Lemma 4.1: tr(F~_TKFAC) = tr(F), exactly (up to Lambda's perturbation and
    float rounding) — see docs/reports/plan_lot3.md §0.2 for why this holds regardless of the EMA
    scheme, verified here on the exact per-sample oracle.

    Undamped, this would be an equality to float-rounding precision: with gammas=(1,1), delta_l is
    exactly tr(F_exact) (§0.2's per-batch trace identity), and tr(Phi~)=tr(Psi~)=1 exactly. The
    damping term (§0.3) perturbs this by a computable amount — tr(Phi~)=1+damp*d_in_aug,
    tr(Psi~)=1+damp*d_out with damp=sqrt(Lambda/delta) — so Lambda is set very small (1e-12, not
    0, purely as a numerical safety margin, mirroring plan_lot2.md §2.2's own choice) to keep that
    perturbation far below the assertion's tolerance rather than inflating the tolerance to hide it.
    """
    seed_all(5)
    h, s = torch.randn(N, D_IN), torch.randn(N, D_OUT)
    F_exact = _exact_fisher_block(h, s)
    layer, tkfac, _tekfac = _build_tkfac_and_tekfac(h, s, Lambda=1e-12)

    F_tkfac = tkfac.f_tilde(layer)
    assert torch.allclose(F_tkfac.trace(), F_exact.trace(), atol=1e-4, rtol=1e-5)


def test_tekfac_dominates_tkfac_in_frobenius_norm() -> None:
    """TEKFAC Thm 3.1: ||F - F~_TEKFAC||_F <= ||F - F~_TKFAC||_F."""
    seed_all(6)
    h, s = torch.randn(N, D_IN), torch.randn(N, D_OUT)
    F_exact = _exact_fisher_block(h, s)
    layer, tkfac, tekfac = _build_tkfac_and_tekfac(h, s)

    F_tkfac = tkfac.f_tilde(layer)
    F_tekfac = tekfac.f_tilde(layer)

    err_tkfac = torch.linalg.matrix_norm(F_exact - F_tkfac)
    err_tekfac = torch.linalg.matrix_norm(F_exact - F_tekfac)

    assert err_tekfac <= err_tkfac + 1e-4, (
        f"TEKFAC Thm 3.1 violated: ||F-F~_TEKFAC||_F={err_tekfac:.6f} > "
        f"||F-F~_TKFAC||_F={err_tkfac:.6f}"
    )


def test_f_tilde_matches_precondition_for_tkfac() -> None:
    seed_all(7)
    h, s = torch.randn(N, D_IN), torch.randn(N, D_OUT)
    layer, tkfac, _tekfac = _build_tkfac_and_tekfac(h, s)

    weight_direction = torch.randn(D_OUT, D_IN)
    bias_direction = torch.randn(D_OUT)
    w_out, b_out = tkfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction, bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(tkfac.f_tilde(layer), M.flatten()).reshape(D_OUT, D_IN_AUG)

    got = torch.cat([w_out, b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


def test_f_tilde_matches_precondition_for_tekfac() -> None:
    seed_all(8)
    h, s = torch.randn(N, D_IN), torch.randn(N, D_OUT)
    layer, _tkfac, tekfac = _build_tkfac_and_tekfac(h, s)

    weight_direction = torch.randn(D_OUT, D_IN)
    bias_direction = torch.randn(D_OUT)
    w_out, b_out = tekfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction, bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(tekfac.f_tilde(layer), M.flatten()).reshape(D_OUT, D_IN_AUG)

    got = torch.cat([w_out, b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


# --------------------------------------------------------------------------------------------
# Lot 4: Conv2d (KFC), docs/reports/plan_lot4.md §0.7, §2.2. Replays exit criteria 1-4 of
# docs/reports/plan.md §6.1 on a toy 3x3, 2-input-channel conv (groups=1, dilation=(1,1)).
#
# F here is the *pooled-sample* exact Fisher block, E_{(n,t)}[(s_{n,t}s_{n,t}^T) (x)
# (a_{n,t}a_{n,t}^T)] over (example, output-location) pairs treated as i.i.d. -- the literal
# generalisation of the Linear-case F above under the same spatial-independence reading
# (TKFAC Assumption 4.1) the implementation itself relies on; see plan_lot4.md §0.1/§0.7a for why
# this, and not a fully spatially-cross-correlated block, is the right comparison target.
# --------------------------------------------------------------------------------------------

CONV_C_IN, CONV_C_OUT, CONV_K, CONV_PAD, CONV_HW, CONV_N = 2, 3, 3, 1, 5, 8
CONV_D_IN_AUG = CONV_C_IN * CONV_K * CONV_K + 1  # 19, bias-augmented


def _conv_layer() -> nn.Conv2d:
    return nn.Conv2d(CONV_C_IN, CONV_C_OUT, kernel_size=CONV_K, padding=CONV_PAD, bias=True)


def test_unfold_oracle_matches_conv2d_identity() -> None:
    """Cross-checks the oracle's patch layout (torch.nn.functional.unfold, independent of this
    project's own extract_patches) against Conv2d's own weight layout, before trusting it as ground
    truth below -- plan_lot4.md §0.7b.
    """
    seed_all(20)
    layer = _conv_layer()
    x = torch.randn(4, CONV_C_IN, CONV_HW, CONV_HW)
    out = layer(x)

    patches = F.unfold(x, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
    w2d = layer.weight.reshape(layer.weight.size(0), -1)
    out_manual = (w2d @ patches) + layer.bias.unsqueeze(0).unsqueeze(-1)
    out_manual = out_manual.reshape(out.shape)

    assert torch.allclose(out_manual, out, atol=1e-4, rtol=1e-4)


def _exact_fisher_block_conv2d(x: torch.Tensor, s: torch.Tensor, layer: nn.Conv2d) -> torch.Tensor:
    """Independent oracle (plan_lot4.md §0.7): does NOT call extract_patches/augment_conv2d_input/
    flatten_conv2d_output_grad -- uses torch.nn.functional.unfold for the patch side instead, and
    re-derives the output-side pooling locally rather than importing flatten_conv2d_output_grad.
    """
    patches = F.unfold(x, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
    patches = patches.transpose(1, 2).reshape(-1, patches.size(1))  # (N*S, P)
    a = torch.cat([patches, patches.new_ones(patches.size(0), 1)], dim=1)  # (N*S, P+1)
    d = s.transpose(1, 2).transpose(2, 3).reshape(-1, s.size(1))  # (N*S, C_out)

    dense_dim = CONV_C_OUT * CONV_D_IN_AUG
    F_exact = torch.zeros(dense_dim, dense_dim)
    for n in range(a.size(0)):
        F_exact += torch.kron(torch.outer(d[n], d[n]), torch.outer(a[n], a[n]))
    return F_exact / a.size(0)


def _build_conv_kfac_and_ekfac(x: torch.Tensor, s: torch.Tensor, Lambda: float = 1e-8):
    layer = _conv_layer()

    kfac = KFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_inv=1, pi=True)
    kfac.update_input_factor(layer, x, step=0)
    kfac.update_output_factor(layer, s, step=0)
    kfac.refresh(layer, step=0)

    ekfac = EKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_eig=1)
    ekfac.update_input_factor(layer, x, step=0)
    ekfac.update_output_factor(layer, s, step=0)  # no eigenbasis yet -> s* not estimated
    ekfac.refresh(layer, step=0)  # Q_A, Q_B from the exact A, B; s* bootstrapped
    # Re-drive the SAME (x, s) now that the eigenbasis exists (plan_lot2.md §2.2's own warning
    # against estimating s* from a batch different from the one defining F).
    ekfac.update_input_factor(layer, x, step=1)
    ekfac.update_output_factor(layer, s, step=1)

    return layer, kfac, ekfac


def test_ekfac_dominates_kfac_in_frobenius_norm_conv2d() -> None:
    seed_all(21)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)

    layer, kfac, ekfac = _build_conv_kfac_and_ekfac(x, s)
    F_exact = _exact_fisher_block_conv2d(x, s, layer)

    F_kfac, F_ekfac = kfac.f_tilde(layer), ekfac.f_tilde(layer)
    err_kfac = torch.linalg.matrix_norm(F_exact - F_kfac)
    err_ekfac = torch.linalg.matrix_norm(F_exact - F_ekfac)

    assert err_ekfac <= err_kfac + 1e-4, (
        f"EKFAC Thm 2/3 violated on Conv2d: ||F-F~_EKFAC||_F={err_ekfac:.6f} > "
        f"||F-F~_KFAC||_F={err_kfac:.6f}"
    )


def test_eigenbases_are_orthogonal_conv2d() -> None:
    seed_all(22)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    layer, _kfac, ekfac = _build_conv_kfac_and_ekfac(x, s)

    Q_A, Q_B = ekfac._Q_A[layer], ekfac._Q_B[layer]
    assert torch.allclose(Q_A.t() @ Q_A, torch.eye(CONV_D_IN_AUG), atol=1e-5)
    assert torch.allclose(Q_B.t() @ Q_B, torch.eye(CONV_C_OUT), atol=1e-5)


def test_f_tilde_matches_precondition_for_kfac_conv2d() -> None:
    seed_all(23)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    layer, kfac, _ekfac = _build_conv_kfac_and_ekfac(x, s)

    weight_direction = torch.randn(CONV_C_OUT, CONV_C_IN, CONV_K, CONV_K)
    bias_direction = torch.randn(CONV_C_OUT)
    w_out, b_out = kfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction.reshape(CONV_C_OUT, -1), bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(kfac.f_tilde(layer), M.flatten()).reshape(CONV_C_OUT, CONV_D_IN_AUG)

    got = torch.cat([w_out.reshape(CONV_C_OUT, -1), b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


def test_f_tilde_matches_precondition_for_ekfac_conv2d() -> None:
    seed_all(24)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    layer, _kfac, ekfac = _build_conv_kfac_and_ekfac(x, s)

    weight_direction = torch.randn(CONV_C_OUT, CONV_C_IN, CONV_K, CONV_K)
    bias_direction = torch.randn(CONV_C_OUT)
    w_out, b_out = ekfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction.reshape(CONV_C_OUT, -1), bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(ekfac.f_tilde(layer), M.flatten()).reshape(CONV_C_OUT, CONV_D_IN_AUG)

    got = torch.cat([w_out.reshape(CONV_C_OUT, -1), b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


def _build_conv_tkfac_and_tekfac(x: torch.Tensor, s: torch.Tensor, Lambda: float = 1e-8):
    layer = _conv_layer()

    tkfac = TKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_inv=1)
    tkfac.update_input_factor(layer, x, step=0)
    tkfac.update_output_factor(layer, s, step=0)
    tkfac.refresh(layer, step=0)

    tekfac = TEKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_eig=1, T_re=1)
    tekfac.update_input_factor(layer, x, step=0)
    tekfac.update_output_factor(layer, s, step=0)  # no eigenbasis yet -> Theta not touched
    tekfac.refresh(layer, step=0)  # Q_Phi, Q_Psi from the exact Phi_raw/Psi_raw; Theta bootstrapped
    tekfac.update_input_factor(layer, x, step=1)
    tekfac.update_output_factor(layer, s, step=1)

    return layer, tkfac, tekfac


def test_tkfac_trace_matches_exact_fisher_conv2d() -> None:
    seed_all(25)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    F_exact = _exact_fisher_block_conv2d(x, s, _conv_layer())
    layer, tkfac, _tekfac = _build_conv_tkfac_and_tekfac(x, s, Lambda=1e-12)

    F_tkfac = tkfac.f_tilde(layer)
    assert torch.allclose(F_tkfac.trace(), F_exact.trace(), atol=1e-4, rtol=1e-5)


def test_tekfac_dominates_tkfac_in_frobenius_norm_conv2d() -> None:
    seed_all(26)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    layer, tkfac, tekfac = _build_conv_tkfac_and_tekfac(x, s)
    F_exact = _exact_fisher_block_conv2d(x, s, layer)

    F_tkfac, F_tekfac = tkfac.f_tilde(layer), tekfac.f_tilde(layer)
    err_tkfac = torch.linalg.matrix_norm(F_exact - F_tkfac)
    err_tekfac = torch.linalg.matrix_norm(F_exact - F_tekfac)

    assert err_tekfac <= err_tkfac + 1e-4, (
        f"TEKFAC Thm 3.1 violated on Conv2d: ||F-F~_TEKFAC||_F={err_tekfac:.6f} > "
        f"||F-F~_TKFAC||_F={err_tkfac:.6f}"
    )


def test_f_tilde_matches_precondition_for_tkfac_conv2d() -> None:
    seed_all(27)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    layer, tkfac, _tekfac = _build_conv_tkfac_and_tekfac(x, s)

    weight_direction = torch.randn(CONV_C_OUT, CONV_C_IN, CONV_K, CONV_K)
    bias_direction = torch.randn(CONV_C_OUT)
    w_out, b_out = tkfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction.reshape(CONV_C_OUT, -1), bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(tkfac.f_tilde(layer), M.flatten()).reshape(CONV_C_OUT, CONV_D_IN_AUG)

    got = torch.cat([w_out.reshape(CONV_C_OUT, -1), b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


def test_f_tilde_matches_precondition_for_tekfac_conv2d() -> None:
    seed_all(28)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    layer, _tkfac, tekfac = _build_conv_tkfac_and_tekfac(x, s)

    weight_direction = torch.randn(CONV_C_OUT, CONV_C_IN, CONV_K, CONV_K)
    bias_direction = torch.randn(CONV_C_OUT)
    w_out, b_out = tekfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction.reshape(CONV_C_OUT, -1), bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(tekfac.f_tilde(layer), M.flatten()).reshape(CONV_C_OUT, CONV_D_IN_AUG)

    got = torch.cat([w_out.reshape(CONV_C_OUT, -1), b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


# --------------------------------------------------------------------------------------------
# Lot 5: normalisation layers (BatchNorm2d), docs/reports/plan_lot5.md §0.1-§0.2, §2.2. Replays
# exit criteria 1-4 of docs/reports/plan.md §6.1 on a toy BatchNorm2d(3).
#
# F here is the pooled-sample block E_x[(s_x s_x^T) (x) (h_bar_x h_bar_x^T)] with h_bar_x = [z_x, 1],
# z_x = mean_c(h_{c,x}) -- the literal generalisation of the Linear/Conv2d-case F under plan_lot5.md
# §0.2's Frobenius-optimal, S-independent scalar surrogate for Proposition 3.1's exact H|_nu (a full
# C x C matrix there, Hadamard- not Kronecker-combined with S). This is exactly the quantity
# compute_h_full/compute_s_full + kfac/ekfac/tkfac/tekfac are designed to approximate for a
# normalisation layer, not Proposition 3.1's own literal (Hadamard-structured) FIM -- see
# plan_lot5.md §0.1-§0.2 for why the two are deliberately not the same target.
# --------------------------------------------------------------------------------------------

NORM_C, NORM_HW, NORM_N = 3, 4, 6  # BatchNorm2d(3), (N, 3, 4, 4) -> T = N*H*W = 96
NORM_D_IN_AUG = 2


def _norm_layer() -> nn.BatchNorm2d:
    return nn.BatchNorm2d(NORM_C)


def _exact_fisher_block_norm(h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Independent oracle (plan_lot5.md §2.2): does NOT call _pool_norm_layer/augment_norm_input/
    flatten_norm_output_grad -- re-derives BatchNorm2d's own channel-dim-1 pooling locally.
    """
    pooled_h = h.permute(0, 2, 3, 1).reshape(-1, NORM_C)
    pooled_s = s.permute(0, 2, 3, 1).reshape(-1, NORM_C)
    z = pooled_h.mean(dim=1)
    h_bar = torch.stack([z, torch.ones_like(z)], dim=1)

    dense_dim = NORM_C * NORM_D_IN_AUG
    F_exact = torch.zeros(dense_dim, dense_dim)
    for x in range(h_bar.size(0)):
        F_exact += torch.kron(torch.outer(pooled_s[x], pooled_s[x]), torch.outer(h_bar[x], h_bar[x]))
    return F_exact / h_bar.size(0)


def _build_norm_kfac_and_ekfac(h: torch.Tensor, s: torch.Tensor, Lambda: float = 1e-8):
    layer = _norm_layer()

    kfac = KFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_inv=1, pi=True)
    kfac.update_input_factor(layer, h, step=0)
    kfac.update_output_factor(layer, s, step=0)
    kfac.refresh(layer, step=0)

    ekfac = EKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_eig=1)
    ekfac.update_input_factor(layer, h, step=0)
    ekfac.update_output_factor(layer, s, step=0)  # no eigenbasis yet -> s* not estimated
    ekfac.refresh(layer, step=0)  # Q_A, Q_B from the exact A, B; s* bootstrapped
    ekfac.update_input_factor(layer, h, step=1)
    ekfac.update_output_factor(layer, s, step=1)

    return layer, kfac, ekfac


def test_ekfac_dominates_kfac_in_frobenius_norm_norm() -> None:
    seed_all(30)
    h = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    s = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)

    layer, kfac, ekfac = _build_norm_kfac_and_ekfac(h, s)
    F_exact = _exact_fisher_block_norm(h, s)

    F_kfac, F_ekfac = kfac.f_tilde(layer), ekfac.f_tilde(layer)
    err_kfac = torch.linalg.matrix_norm(F_exact - F_kfac)
    err_ekfac = torch.linalg.matrix_norm(F_exact - F_ekfac)

    assert err_ekfac <= err_kfac + 1e-4, (
        f"EKFAC Thm 2/3 violated on a normalisation layer: "
        f"||F-F~_EKFAC||_F={err_ekfac:.6f} > ||F-F~_KFAC||_F={err_kfac:.6f}"
    )


def test_eigenbases_are_orthogonal_norm() -> None:
    seed_all(31)
    h = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    s = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    layer, _kfac, ekfac = _build_norm_kfac_and_ekfac(h, s)

    Q_A, Q_B = ekfac._Q_A[layer], ekfac._Q_B[layer]
    assert torch.allclose(Q_A.t() @ Q_A, torch.eye(NORM_D_IN_AUG), atol=1e-5)
    assert torch.allclose(Q_B.t() @ Q_B, torch.eye(NORM_C), atol=1e-5)


def test_f_tilde_matches_precondition_for_kfac_norm() -> None:
    seed_all(32)
    h = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    s = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    layer, kfac, _ekfac = _build_norm_kfac_and_ekfac(h, s)

    weight_direction, bias_direction = torch.randn(NORM_C), torch.randn(NORM_C)
    w_out, b_out = kfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction.unsqueeze(1), bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(kfac.f_tilde(layer), M.flatten()).reshape(NORM_C, NORM_D_IN_AUG)

    got = torch.cat([w_out.unsqueeze(1), b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


def test_f_tilde_matches_precondition_for_ekfac_norm() -> None:
    seed_all(33)
    h = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    s = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    layer, _kfac, ekfac = _build_norm_kfac_and_ekfac(h, s)

    weight_direction, bias_direction = torch.randn(NORM_C), torch.randn(NORM_C)
    w_out, b_out = ekfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction.unsqueeze(1), bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(ekfac.f_tilde(layer), M.flatten()).reshape(NORM_C, NORM_D_IN_AUG)

    got = torch.cat([w_out.unsqueeze(1), b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


def _build_norm_tkfac_and_tekfac(h: torch.Tensor, s: torch.Tensor, Lambda: float = 1e-8):
    layer = _norm_layer()

    tkfac = TKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_inv=1)
    tkfac.update_input_factor(layer, h, step=0)
    tkfac.update_output_factor(layer, s, step=0)
    tkfac.refresh(layer, step=0)

    tekfac = TEKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_eig=1, T_re=1)
    tekfac.update_input_factor(layer, h, step=0)
    tekfac.update_output_factor(layer, s, step=0)  # no eigenbasis yet -> Theta not touched
    tekfac.refresh(layer, step=0)  # Q_Phi, Q_Psi from the exact Phi_raw/Psi_raw; Theta bootstrapped
    tekfac.update_input_factor(layer, h, step=1)
    tekfac.update_output_factor(layer, s, step=1)

    return layer, tkfac, tekfac


def test_tkfac_trace_matches_exact_fisher_norm() -> None:
    seed_all(34)
    h = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    s = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    F_exact = _exact_fisher_block_norm(h, s)
    layer, tkfac, _tekfac = _build_norm_tkfac_and_tekfac(h, s, Lambda=1e-12)

    F_tkfac = tkfac.f_tilde(layer)
    assert torch.allclose(F_tkfac.trace(), F_exact.trace(), atol=1e-4, rtol=1e-5)


def test_tekfac_dominates_tkfac_in_frobenius_norm_norm() -> None:
    seed_all(35)
    h = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    s = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    layer, tkfac, tekfac = _build_norm_tkfac_and_tekfac(h, s)
    F_exact = _exact_fisher_block_norm(h, s)

    F_tkfac, F_tekfac = tkfac.f_tilde(layer), tekfac.f_tilde(layer)
    err_tkfac = torch.linalg.matrix_norm(F_exact - F_tkfac)
    err_tekfac = torch.linalg.matrix_norm(F_exact - F_tekfac)

    assert err_tekfac <= err_tkfac + 1e-4, (
        f"TEKFAC Thm 3.1 violated on a normalisation layer: "
        f"||F-F~_TEKFAC||_F={err_tekfac:.6f} > ||F-F~_TKFAC||_F={err_tkfac:.6f}"
    )


def test_f_tilde_matches_precondition_for_tkfac_norm() -> None:
    seed_all(36)
    h = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    s = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    layer, tkfac, _tekfac = _build_norm_tkfac_and_tekfac(h, s)

    weight_direction, bias_direction = torch.randn(NORM_C), torch.randn(NORM_C)
    w_out, b_out = tkfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction.unsqueeze(1), bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(tkfac.f_tilde(layer), M.flatten()).reshape(NORM_C, NORM_D_IN_AUG)

    got = torch.cat([w_out.unsqueeze(1), b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


def test_f_tilde_matches_precondition_for_tekfac_norm() -> None:
    seed_all(37)
    h = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    s = torch.randn(NORM_N, NORM_C, NORM_HW, NORM_HW)
    layer, _tkfac, tekfac = _build_norm_tkfac_and_tekfac(h, s)

    weight_direction, bias_direction = torch.randn(NORM_C), torch.randn(NORM_C)
    w_out, b_out = tekfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction.unsqueeze(1), bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(tekfac.f_tilde(layer), M.flatten()).reshape(NORM_C, NORM_D_IN_AUG)

    got = torch.cat([w_out.unsqueeze(1), b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


# --------------------------------------------------------------------------------------------
# Lot 6: the SUA approximation, Conv2d input factor only. See docs/reports/plan_lot6.md
# §0.1-§0.4, §0.7, §2.2. Replays exit criteria 1-4 of docs/reports/plan.md §6.1 on the same toy
# Conv2d as lot 4 (padding=(k-1)/2, stride=1), at the SUA-consistent scale.
#
# F_sua here is the *SUA-consistent* exact Fisher block: E_{(n,t)}[(s_{n,t}s_{n,t}^T) (x)
# (a_bar_{n,t}a_bar_{n,t}^T)] with a_bar_{n,t} = [x_{n,t}, 1] -- the same pooled-sample
# construction as lot 4's own F, but with the *center-pixel* channel vector standing in for the
# whole patch (plan_lot6.md §0.1's IAD+SH+SUA target), not the fully spatially-cross-correlated
# block lot 4's own oracle uses. Valid as a raw-pixel-pooling oracle specifically because
# CONV_PAD=(CONV_K-1)//2, stride=1 (plan_lot6.md §0.3's identity, independently verified
# numerically in test_full_factors_match_diag.py's own
# test_augment_conv2d_input_sua_center_slice_equals_raw_pixel).
# --------------------------------------------------------------------------------------------

CONV_SUA_D_IN_AUG = CONV_C_IN + 1  # 3, vs. lot 4's CONV_D_IN_AUG = 19


def _exact_fisher_block_conv2d_sua(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    a = x.permute(0, 2, 3, 1).reshape(-1, CONV_C_IN)
    a = torch.cat([a, a.new_ones(a.size(0), 1)], dim=1)
    d = s.permute(0, 2, 3, 1).reshape(-1, CONV_C_OUT)

    dense_dim = CONV_C_OUT * CONV_SUA_D_IN_AUG
    F_exact = torch.zeros(dense_dim, dense_dim)
    for n in range(a.size(0)):
        F_exact += torch.kron(torch.outer(d[n], d[n]), torch.outer(a[n], a[n]))
    return F_exact / a.size(0)


def _build_conv_sua_kfac_and_ekfac(x: torch.Tensor, s: torch.Tensor, Lambda: float = 1e-8):
    layer = _conv_layer()

    kfac = KFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_inv=1, pi=True, conv_sua=True)
    kfac.update_input_factor(layer, x, step=0)
    kfac.update_output_factor(layer, s, step=0)
    kfac.refresh(layer, step=0)

    ekfac = EKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_eig=1, conv_sua=True)
    ekfac.update_input_factor(layer, x, step=0)
    ekfac.update_output_factor(layer, s, step=0)  # no eigenbasis yet -> s* not estimated
    ekfac.refresh(layer, step=0)  # Q_A, Q_B from the exact A, B; s* bootstrapped
    # Re-drive the SAME (x, s) now that the eigenbasis exists (plan_lot2.md §2.2's own warning
    # against estimating s* from a batch different from the one defining F).
    ekfac.update_input_factor(layer, x, step=1)
    ekfac.update_output_factor(layer, s, step=1)

    return layer, kfac, ekfac


def test_ekfac_sua_dominates_kfac_sua_in_frobenius_norm() -> None:
    seed_all(40)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)

    layer, kfac, ekfac = _build_conv_sua_kfac_and_ekfac(x, s)
    F_exact = _exact_fisher_block_conv2d_sua(x, s)

    F_kfac, F_ekfac = kfac.f_tilde(layer), ekfac.f_tilde(layer)
    err_kfac = torch.linalg.matrix_norm(F_exact - F_kfac)
    err_ekfac = torch.linalg.matrix_norm(F_exact - F_ekfac)

    assert err_ekfac <= err_kfac + 1e-4, (
        f"EKFAC Thm 2/3 violated on Conv2d-SUA: ||F-F~_EKFAC||_F={err_ekfac:.6f} > "
        f"||F-F~_KFAC||_F={err_kfac:.6f}"
    )


def test_eigenbases_are_orthogonal_conv2d_sua() -> None:
    seed_all(41)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    layer, _kfac, ekfac = _build_conv_sua_kfac_and_ekfac(x, s)

    Q_A, Q_B = ekfac._Q_A[layer], ekfac._Q_B[layer]
    assert torch.allclose(Q_A.t() @ Q_A, torch.eye(CONV_SUA_D_IN_AUG), atol=1e-5)
    assert torch.allclose(Q_B.t() @ Q_B, torch.eye(CONV_C_OUT), atol=1e-5)


def _build_conv_sua_tkfac_and_tekfac(x: torch.Tensor, s: torch.Tensor, Lambda: float = 1e-8):
    layer = _conv_layer()

    tkfac = TKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_inv=1, conv_sua=True)
    tkfac.update_input_factor(layer, x, step=0)
    tkfac.update_output_factor(layer, s, step=0)
    tkfac.refresh(layer, step=0)

    tekfac = TEKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_eig=1, T_re=1, conv_sua=True)
    tekfac.update_input_factor(layer, x, step=0)
    tekfac.update_output_factor(layer, s, step=0)  # no eigenbasis yet -> Theta not touched
    tekfac.refresh(layer, step=0)  # Q_Phi, Q_Psi from the exact Phi_raw/Psi_raw; Theta bootstrapped
    tekfac.update_input_factor(layer, x, step=1)
    tekfac.update_output_factor(layer, s, step=1)

    return layer, tkfac, tekfac


def test_tkfac_sua_trace_matches_exact_fisher() -> None:
    seed_all(42)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    F_exact = _exact_fisher_block_conv2d_sua(x, s)
    layer, tkfac, _tekfac = _build_conv_sua_tkfac_and_tekfac(x, s, Lambda=1e-12)

    F_tkfac = tkfac.f_tilde(layer)
    assert torch.allclose(F_tkfac.trace(), F_exact.trace(), atol=1e-4, rtol=1e-5)


def test_tekfac_sua_dominates_tkfac_sua_in_frobenius_norm() -> None:
    seed_all(43)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    layer, tkfac, tekfac = _build_conv_sua_tkfac_and_tekfac(x, s)
    F_exact = _exact_fisher_block_conv2d_sua(x, s)

    F_tkfac, F_tekfac = tkfac.f_tilde(layer), tekfac.f_tilde(layer)
    err_tkfac = torch.linalg.matrix_norm(F_exact - F_tkfac)
    err_tekfac = torch.linalg.matrix_norm(F_exact - F_tekfac)

    assert err_tekfac <= err_tkfac + 1e-4, (
        f"TEKFAC Thm 3.1 violated on Conv2d-SUA: ||F-F~_TEKFAC||_F={err_tekfac:.6f} > "
        f"||F-F~_TKFAC||_F={err_tkfac:.6f}"
    )


def _assert_precondition_matches_f_tilde_per_position(approx, layer, weight_direction, bias_direction):
    """plan_lot6.md §0.4: precondition()'s per-kernel-offset application must agree with
    f_tilde()'s dense reconstruction independently *at every offset*, and the returned bias only
    at the center offset -- the other offsets' own solved bias column are expected to disagree (a
    property of the block-diagonal-across-offsets approximation, not a bug), checked explicitly
    below rather than merely assumed.
    """
    kh, kw = layer.kernel_size
    w_out, b_out = approx.precondition(layer, weight_direction, bias_direction)
    F_dense = approx.f_tilde(layer)
    center = (kh // 2) * kw + (kw // 2)

    any_bias_mismatch_off_center = False
    for p in range(kh * kw):
        khi, kwi = divmod(p, kw)
        M_p = torch.cat([weight_direction[:, :, khi, kwi], bias_direction.unsqueeze(1)], dim=1)
        expected_p = torch.linalg.solve(F_dense, M_p.flatten()).reshape(weight_direction.size(0), -1)
        assert torch.allclose(w_out[:, :, khi, kwi], expected_p[:, :-1], rtol=1e-3, atol=1e-5)
        if p == center:
            assert torch.allclose(b_out, expected_p[:, -1], rtol=1e-3, atol=1e-5)
        elif not torch.allclose(b_out, expected_p[:, -1], rtol=1e-3, atol=1e-5):
            any_bias_mismatch_off_center = True

    assert any_bias_mismatch_off_center, (
        "expected at least one non-center kernel offset's own solved bias to disagree with the "
        "returned (center-offset) bias, otherwise this is not a discriminating test of §0.4's "
        "center convention"
    )


def test_f_tilde_matches_precondition_for_kfac_conv2d_sua() -> None:
    seed_all(44)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    layer, kfac, _ekfac = _build_conv_sua_kfac_and_ekfac(x, s)

    weight_direction = torch.randn(CONV_C_OUT, CONV_C_IN, CONV_K, CONV_K)
    bias_direction = torch.randn(CONV_C_OUT)
    _assert_precondition_matches_f_tilde_per_position(kfac, layer, weight_direction, bias_direction)


def test_f_tilde_matches_precondition_for_ekfac_conv2d_sua() -> None:
    seed_all(45)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    layer, _kfac, ekfac = _build_conv_sua_kfac_and_ekfac(x, s)

    weight_direction = torch.randn(CONV_C_OUT, CONV_C_IN, CONV_K, CONV_K)
    bias_direction = torch.randn(CONV_C_OUT)
    _assert_precondition_matches_f_tilde_per_position(ekfac, layer, weight_direction, bias_direction)


def test_f_tilde_matches_precondition_for_tkfac_conv2d_sua() -> None:
    seed_all(46)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    layer, tkfac, _tekfac = _build_conv_sua_tkfac_and_tekfac(x, s)

    weight_direction = torch.randn(CONV_C_OUT, CONV_C_IN, CONV_K, CONV_K)
    bias_direction = torch.randn(CONV_C_OUT)
    _assert_precondition_matches_f_tilde_per_position(tkfac, layer, weight_direction, bias_direction)


def test_f_tilde_matches_precondition_for_tekfac_conv2d_sua() -> None:
    seed_all(47)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    layer, _tkfac, tekfac = _build_conv_sua_tkfac_and_tekfac(x, s)

    weight_direction = torch.randn(CONV_C_OUT, CONV_C_IN, CONV_K, CONV_K)
    bias_direction = torch.randn(CONV_C_OUT)
    _assert_precondition_matches_f_tilde_per_position(tekfac, layer, weight_direction, bias_direction)


@pytest.mark.parametrize(
    "mode_cls,mode_kwargs",
    [
        (KFACApproximation, {"T_inv": 1, "pi": True}),
        (EKFACApproximation, {"T_eig": 1}),
        (TKFACApproximation, {"T_inv": 1}),
        (TEKFACApproximation, {"T_eig": 1, "T_re": 1}),
    ],
)
def test_conv2d_sua_matches_full_for_1x1_kernel(mode_cls, mode_kwargs) -> None:
    """plan_lot6.md §0.6: with a 1x1 kernel, SUA and the patch-based path coincide exactly, so
    conv_sua=True and conv_sua=False must agree on precondition()'s output, for every mode.
    """
    seed_all(48)
    layer = nn.Conv2d(CONV_C_IN, CONV_C_OUT, kernel_size=1, bias=True)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    s = torch.randn(CONV_N, CONV_C_OUT, CONV_HW, CONV_HW)
    weight_direction = torch.randn(CONV_C_OUT, CONV_C_IN, 1, 1)
    bias_direction = torch.randn(CONV_C_OUT)

    approx_full = mode_cls(Lambda=1e-3, gammas=(1.0, 1.0), conv_sua=False, **mode_kwargs)
    approx_full.update_input_factor(layer, x, step=0)
    approx_full.update_output_factor(layer, s, step=0)
    approx_full.refresh(layer, step=0)
    if mode_cls in (EKFACApproximation, TEKFACApproximation):
        approx_full.update_input_factor(layer, x, step=1)
        approx_full.update_output_factor(layer, s, step=1)

    approx_sua = mode_cls(Lambda=1e-3, gammas=(1.0, 1.0), conv_sua=True, **mode_kwargs)
    approx_sua.update_input_factor(layer, x, step=0)
    approx_sua.update_output_factor(layer, s, step=0)
    approx_sua.refresh(layer, step=0)
    if mode_cls in (EKFACApproximation, TEKFACApproximation):
        approx_sua.update_input_factor(layer, x, step=1)
        approx_sua.update_output_factor(layer, s, step=1)

    w_full, b_full = approx_full.precondition(layer, weight_direction, bias_direction)
    w_sua, b_sua = approx_sua.precondition(layer, weight_direction, bias_direction)
    assert torch.allclose(w_full, w_sua, rtol=1e-4, atol=1e-6)
    assert torch.allclose(b_full, b_sua, rtol=1e-4, atol=1e-6)


def test_sua_discards_offblock_frobenius_mass(capsys) -> None:
    """Informational only (plan.md §6.1's "measured, not asserted" precedent, applied to
    plan_lot6.md §0.7's re-statement of kfac_conv_1602.01407.pdf §5.1's own empirical finding that
    SUA "loses a lot of information"): report the fraction of the exact patch-based input
    covariance's Frobenius mass that lives in the cross-kernel-offset correlations SUA's
    block-diagonal-across-offsets approximation discards. No threshold assertion -- only a
    non-degeneracy sanity check that this toy input has *some* such correlation for SUA to discard.
    """
    seed_all(49)
    x = torch.randn(CONV_N, CONV_C_IN, CONV_HW, CONV_HW)
    layer = _conv_layer()

    patches = F.unfold(x, kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
    patches = patches.transpose(1, 2).reshape(-1, patches.size(1))  # (N*S, C_in*kh*kw)
    omega_full = patches.t() @ patches / patches.size(0)  # weight-weight block only, no bias

    kh, kw = layer.kernel_size
    omega_full_6d = omega_full.reshape(CONV_C_IN, kh, kw, CONV_C_IN, kh, kw)
    same_position = torch.eye(kh * kw).reshape(kh, kw, kh, kw)
    mask = same_position.reshape(1, kh, kw, 1, kh, kw)
    omega_block_diag = (omega_full_6d * mask).reshape(omega_full.shape)

    discarded_fraction = (
        torch.linalg.matrix_norm(omega_full - omega_block_diag) / torch.linalg.matrix_norm(omega_full)
    ).item()
    print(f"SUA discards {discarded_fraction:.4%} of Omega's Frobenius mass (cross-offset terms)")
    assert discarded_fraction > 0, "toy input has no cross-position correlation -- vacuous measurement"


def test_sua_memory_footprint_resnet18_scale(capsys) -> None:
    """plan.md §6.3/§7's cited "4609 -> 512" reduction, measured on ResNet-18's largest conv
    factor's exact dimensions (a synthetic Conv2d(512, 512, 3, padding=1), not an actual
    torchvision.models.resnet18 instantiation -- avoids a hard torchvision test dependency;
    plan_lot6.md §3).
    """
    layer = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
    h = torch.randn(2, 512, 4, 4)

    d_in_full = augment_conv2d_input(h, layer).size(1)
    d_in_sua = augment_conv2d_input_sua(h, layer).size(1)
    assert d_in_full == 512 * 3 * 3 + 1  # 4609
    assert d_in_sua == 512 + 1  # 513

    byte_ratio = (d_in_full / d_in_sua) ** 2
    print(f"SUA input-factor byte reduction at ResNet-18 scale: {byte_ratio:.1f}x ({d_in_full} -> {d_in_sua})")
    assert byte_ratio > 50  # loose sanity bound around the ~80.7x figure plan.md/plan_lot6.md cite
