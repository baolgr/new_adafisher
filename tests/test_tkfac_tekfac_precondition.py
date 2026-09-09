"""Unit-level checks for ``TKFACApproximation``/``TEKFACApproximation``, independent of the
brute-force oracle in ``test_frobenius_dominance.py``: output shapes, the trace invariant that
makes ``tr(F~_TKFAC)=tr(F)`` hold (docs/reports/plan_lot3.md §0.2), and the
``T_inv``/``T_eig``/``T_re`` amortisation cadences (plan_lot3.md §2.1).

Lot 4 (docs/reports/plan_lot4.md §2.3) adds the ``Conv2d`` (``groups=1``, ``dilation=(1,1)``) analogue
of the shape/trace-invariant/cadence checks below. Lot 5 (docs/reports/plan_lot5.md §2.3) adds the
analogous ``BatchNorm2d``/``LayerNorm`` checks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from adafisher_modes.approximations.tekfac import TEKFACApproximation
from adafisher_modes.approximations.tkfac import TKFACApproximation
from conftest import seed_all


def _drive(approx, layer, h, s, step):
    approx.update_input_factor(layer, h, step)
    approx.update_output_factor(layer, s, step)
    approx.refresh(layer, step)


def test_precondition_shapes_with_bias():
    seed_all(0)
    layer = nn.Linear(6, 4, bias=True)
    h, s = torch.randn(16, 6), torch.randn(16, 4)
    for approx in (TKFACApproximation(T_inv=1), TEKFACApproximation(T_eig=1, T_re=1)):
        _drive(approx, layer, h, s, 0)
        w, b = approx.precondition(layer, torch.randn(4, 6), torch.randn(4))
        assert w.shape == (4, 6)
        assert b.shape == (4,)


def test_precondition_shapes_without_bias():
    seed_all(1)
    layer = nn.Linear(6, 4, bias=False)
    h, s = torch.randn(16, 6), torch.randn(16, 4)
    for approx in (TKFACApproximation(T_inv=1), TEKFACApproximation(T_eig=1, T_re=1)):
        _drive(approx, layer, h, s, 0)
        w = approx.precondition(layer, torch.randn(4, 6), None)
        assert isinstance(w, torch.Tensor)
        assert w.shape == (4, 6)


def test_trace_invariant_holds_under_real_gammas():
    """docs/reports/plan_lot3.md §0.2: tr(Phi_raw) == delta and tr(Psi_raw) == delta must hold
    *exactly* (up to float rounding) at every step, under the project's real, non-summing-to-1
    ``gammas=(0.92, 0.008)`` — not just the test-only ``gammas=(1,1)`` special case. This is what
    would silently break if a future edit normalized Phi/Psi before accumulating them.
    """
    seed_all(2)
    layer = nn.Linear(5, 3, bias=True)
    tkfac = TKFACApproximation(T_inv=1)  # default gammas=(0.92, 0.008)

    for step in range(10):
        h, s = torch.randn(20, 5), torch.randn(20, 3)
        _drive(tkfac, layer, h, s, step)
        delta = tkfac._delta[layer]
        assert torch.allclose(tkfac._Phi_raw[layer].trace(), delta, atol=1e-4, rtol=1e-5)
        assert torch.allclose(tkfac._Psi_raw[layer].trace(), delta, atol=1e-4, rtol=1e-5)


def test_tkfac_inverse_cadence_respects_t_inv():
    seed_all(3)
    layer = nn.Linear(5, 3, bias=True)
    tkfac = TKFACApproximation(T_inv=3)

    for step in range(6):
        h, s = torch.randn(20, 5), torch.randn(20, 3)
        _drive(tkfac, layer, h, s, step)
        if step % 3 == 0:
            cached_Phi_inv = tkfac._Phi_inv[layer].clone()

    # After the loop, the cache reflects the last refresh step (3), not step 5.
    assert torch.equal(tkfac._Phi_inv[layer], cached_Phi_inv)


def test_tekfac_eigenbasis_cadence_respects_t_eig():
    seed_all(4)
    layer = nn.Linear(5, 3, bias=True)
    tekfac = TEKFACApproximation(T_eig=3, T_re=1)

    q_phi_at_refresh = {}
    for step in range(6):
        h, s = torch.randn(20, 5), torch.randn(20, 3)
        _drive(tekfac, layer, h, s, step)
        if step % 3 == 0:
            q_phi_at_refresh[step] = tekfac._Q_Phi[layer].clone()

    assert torch.equal(tekfac._Q_Phi[layer], q_phi_at_refresh[3])
    assert not torch.equal(q_phi_at_refresh[0], q_phi_at_refresh[3])


def test_tekfac_theta_untouched_before_eigenbasis_exists():
    seed_all(5)
    layer = nn.Linear(5, 3, bias=True)
    # T_eig=2: no eigenbasis at step 0, so step 0's update_output_factor must not create/touch
    # _Theta at all (mirrors ekfac's own bootstrap gating, plan_lot2.md §0.3).
    tekfac = TEKFACApproximation(T_eig=2, T_re=1)
    h, s = torch.randn(20, 5), torch.randn(20, 3)
    tekfac.update_input_factor(layer, h, 0)
    tekfac.update_output_factor(layer, s, 0)
    assert layer not in tekfac._Theta
    tekfac.refresh(layer, 0)
    assert layer in tekfac._Theta
    theta_bootstrap = tekfac._Theta[layer].clone()

    h, s = torch.randn(20, 5), torch.randn(20, 3)
    tekfac.update_input_factor(layer, h, 1)
    tekfac.update_output_factor(layer, s, 1)  # eigenbasis exists now, T_re=1 -> Theta updates
    assert not torch.equal(tekfac._Theta[layer], theta_bootstrap)


def test_tekfac_theta_cadence_respects_t_re():
    seed_all(6)
    layer = nn.Linear(5, 3, bias=True)
    tekfac = TEKFACApproximation(T_eig=1, T_re=3)

    theta_at_re = {}
    for step in range(6):
        h, s = torch.randn(20, 5), torch.randn(20, 3)
        _drive(tekfac, layer, h, s, step)
        if step == 0:
            theta_at_re[0] = tekfac._Theta[layer].clone()
        if step % 3 == 0 and step > 0:
            theta_at_re[step] = tekfac._Theta[layer].clone()

    # Theta only updates at steps 0 (bootstrap) and 3 (T_re-gated); steps 1,2,4,5 must not move it.
    assert torch.equal(theta_at_re[0], theta_at_re[3]) is False
    assert torch.equal(tekfac._Theta[layer], theta_at_re[3])


def test_f_tilde_matches_precondition_for_tkfac():
    seed_all(7)
    layer = nn.Linear(5, 3, bias=True)
    tkfac = TKFACApproximation(T_inv=1)
    h, s = torch.randn(20, 5), torch.randn(20, 3)
    _drive(tkfac, layer, h, s, 0)

    weight_direction, bias_direction = torch.randn(3, 5), torch.randn(3)
    w_out, b_out = tkfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction, bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(tkfac.f_tilde(layer), M.flatten()).reshape(3, 6)

    got = torch.cat([w_out, b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


def test_f_tilde_matches_precondition_for_tekfac():
    seed_all(8)
    layer = nn.Linear(5, 3, bias=True)
    tekfac = TEKFACApproximation(T_eig=1, T_re=1)
    h, s = torch.randn(20, 5), torch.randn(20, 3)
    _drive(tekfac, layer, h, s, 0)
    h, s = torch.randn(20, 5), torch.randn(20, 3)
    _drive(tekfac, layer, h, s, 1)  # eigenbasis now exists from step 0 -> Theta gets estimated

    weight_direction, bias_direction = torch.randn(3, 5), torch.randn(3)
    w_out, b_out = tekfac.precondition(layer, weight_direction, bias_direction)

    M = torch.cat([weight_direction, bias_direction.unsqueeze(1)], dim=1)
    expected = torch.linalg.solve(tekfac.f_tilde(layer), M.flatten()).reshape(3, 6)

    got = torch.cat([w_out, b_out.unsqueeze(1)], dim=1)
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-5)


# --------------------------------------------------------------------------------------------
# Lot 4: Conv2d (groups=1, dilation=(1,1)). See docs/reports/plan_lot4.md §2.3.
# --------------------------------------------------------------------------------------------


def test_precondition_shapes_with_bias_conv2d():
    seed_all(10)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
    for approx in (TKFACApproximation(T_inv=1), TEKFACApproximation(T_eig=1, T_re=1)):
        _drive(approx, layer, h, s, 0)
        w, b = approx.precondition(layer, torch.randn(3, 2, 3, 3), torch.randn(3))
        assert w.shape == (3, 2, 3, 3)
        assert b.shape == (3,)


def test_precondition_shapes_without_bias_conv2d():
    seed_all(11)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
    h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
    for approx in (TKFACApproximation(T_inv=1), TEKFACApproximation(T_eig=1, T_re=1)):
        _drive(approx, layer, h, s, 0)
        w = approx.precondition(layer, torch.randn(3, 2, 3, 3), None)
        assert isinstance(w, torch.Tensor)
        assert w.shape == (3, 2, 3, 3)


def test_trace_invariant_holds_under_real_gammas_conv2d():
    seed_all(12)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    tkfac = TKFACApproximation(T_inv=1)  # default gammas=(0.92, 0.008)

    for step in range(10):
        h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
        _drive(tkfac, layer, h, s, step)
        delta = tkfac._delta[layer]
        assert torch.allclose(tkfac._Phi_raw[layer].trace(), delta, atol=1e-4, rtol=1e-5)
        assert torch.allclose(tkfac._Psi_raw[layer].trace(), delta, atol=1e-4, rtol=1e-5)


def test_tkfac_inverse_cadence_respects_t_inv_conv2d():
    seed_all(13)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    tkfac = TKFACApproximation(T_inv=3)

    for step in range(6):
        h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
        _drive(tkfac, layer, h, s, step)
        if step % 3 == 0:
            cached_Phi_inv = tkfac._Phi_inv[layer].clone()

    assert torch.equal(tkfac._Phi_inv[layer], cached_Phi_inv)


def test_tekfac_eigenbasis_cadence_respects_t_eig_conv2d():
    seed_all(14)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    tekfac = TEKFACApproximation(T_eig=3, T_re=1)

    q_phi_at_refresh = {}
    for step in range(6):
        h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
        _drive(tekfac, layer, h, s, step)
        if step % 3 == 0:
            q_phi_at_refresh[step] = tekfac._Q_Phi[layer].clone()

    assert torch.equal(tekfac._Q_Phi[layer], q_phi_at_refresh[3])
    assert not torch.equal(q_phi_at_refresh[0], q_phi_at_refresh[3])


# --------------------------------------------------------------------------------------------
# Lot 5: BatchNorm2d / LayerNorm (normalized_shape a 1-tuple). See docs/reports/plan_lot5.md §2.3.
# --------------------------------------------------------------------------------------------


def test_precondition_shapes_batchnorm2d():
    seed_all(20)
    layer = nn.BatchNorm2d(3)
    h, s = torch.randn(4, 3, 5, 5), torch.randn(4, 3, 5, 5)
    for approx in (TKFACApproximation(T_inv=1), TEKFACApproximation(T_eig=1, T_re=1)):
        _drive(approx, layer, h, s, 0)
        w, b = approx.precondition(layer, torch.randn(3), torch.randn(3))
        assert w.shape == (3,)
        assert b.shape == (3,)


def test_precondition_shapes_layernorm():
    seed_all(21)
    layer = nn.LayerNorm(8)
    h, s = torch.randn(16, 8), torch.randn(16, 8)
    for approx in (TKFACApproximation(T_inv=1), TEKFACApproximation(T_eig=1, T_re=1)):
        _drive(approx, layer, h, s, 0)
        w, b = approx.precondition(layer, torch.randn(8), torch.randn(8))
        assert w.shape == (8,)
        assert b.shape == (8,)


def test_trace_invariant_holds_under_real_gammas_batchnorm2d():
    seed_all(22)
    layer = nn.BatchNorm2d(3)
    tkfac = TKFACApproximation(T_inv=1)  # default gammas=(0.92, 0.008)

    for step in range(10):
        h, s = torch.randn(4, 3, 5, 5), torch.randn(4, 3, 5, 5)
        _drive(tkfac, layer, h, s, step)
        delta = tkfac._delta[layer]
        assert torch.allclose(tkfac._Phi_raw[layer].trace(), delta, atol=1e-4, rtol=1e-5)
        assert torch.allclose(tkfac._Psi_raw[layer].trace(), delta, atol=1e-4, rtol=1e-5)


def test_tkfac_inverse_cadence_respects_t_inv_layernorm():
    seed_all(23)
    layer = nn.LayerNorm(8)
    tkfac = TKFACApproximation(T_inv=3)

    for step in range(6):
        h, s = torch.randn(16, 8), torch.randn(16, 8)
        _drive(tkfac, layer, h, s, step)
        if step % 3 == 0:
            cached_Phi_inv = tkfac._Phi_inv[layer].clone()

    assert torch.equal(tkfac._Phi_inv[layer], cached_Phi_inv)


def test_tekfac_eigenbasis_cadence_respects_t_eig_layernorm():
    seed_all(24)
    layer = nn.LayerNorm(8)
    tekfac = TEKFACApproximation(T_eig=3, T_re=1)

    q_phi_at_refresh = {}
    for step in range(6):
        h, s = torch.randn(16, 8), torch.randn(16, 8)
        _drive(tekfac, layer, h, s, step)
        if step % 3 == 0:
            q_phi_at_refresh[step] = tekfac._Q_Phi[layer].clone()

    assert torch.equal(tekfac._Q_Phi[layer], q_phi_at_refresh[3])
    assert not torch.equal(q_phi_at_refresh[0], q_phi_at_refresh[3])


# --------------------------------------------------------------------------------------------
# Lot 6: the SUA approximation, Conv2d input factor only. See docs/reports/plan_lot6.md §2.3.
# --------------------------------------------------------------------------------------------


def test_precondition_shapes_with_bias_conv2d_sua():
    seed_all(50)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
    for approx in (
        TKFACApproximation(T_inv=1, conv_sua=True),
        TEKFACApproximation(T_eig=1, T_re=1, conv_sua=True),
    ):
        _drive(approx, layer, h, s, 0)
        w, b = approx.precondition(layer, torch.randn(3, 2, 3, 3), torch.randn(3))
        assert w.shape == (3, 2, 3, 3)
        assert b.shape == (3,)


def test_precondition_shapes_without_bias_conv2d_sua():
    seed_all(51)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
    h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
    for approx in (
        TKFACApproximation(T_inv=1, conv_sua=True),
        TEKFACApproximation(T_eig=1, T_re=1, conv_sua=True),
    ):
        _drive(approx, layer, h, s, 0)
        w = approx.precondition(layer, torch.randn(3, 2, 3, 3), None)
        assert isinstance(w, torch.Tensor)
        assert w.shape == (3, 2, 3, 3)


def test_trace_invariant_holds_under_real_gammas_conv2d_sua():
    seed_all(52)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    tkfac = TKFACApproximation(T_inv=1, conv_sua=True)  # default gammas=(0.92, 0.008)

    for step in range(10):
        h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
        _drive(tkfac, layer, h, s, step)
        delta = tkfac._delta[layer]
        assert torch.allclose(tkfac._Phi_raw[layer].trace(), delta, atol=1e-4, rtol=1e-5)
        assert torch.allclose(tkfac._Psi_raw[layer].trace(), delta, atol=1e-4, rtol=1e-5)


def test_tkfac_sua_inverse_cadence_respects_t_inv_conv2d():
    seed_all(53)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    tkfac = TKFACApproximation(T_inv=3, conv_sua=True)

    for step in range(6):
        h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
        _drive(tkfac, layer, h, s, step)
        if step % 3 == 0:
            cached_Phi_inv = tkfac._Phi_inv[layer].clone()

    assert torch.equal(tkfac._Phi_inv[layer], cached_Phi_inv)
    assert tkfac._Phi_inv[layer].shape == (3, 3)  # C_in+1, not C_in*k_h*k_w+1


def test_tekfac_sua_eigenbasis_cadence_respects_t_eig_conv2d():
    seed_all(54)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    tekfac = TEKFACApproximation(T_eig=3, T_re=1, conv_sua=True)

    q_phi_at_refresh = {}
    for step in range(6):
        h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
        _drive(tekfac, layer, h, s, step)
        if step % 3 == 0:
            q_phi_at_refresh[step] = tekfac._Q_Phi[layer].clone()

    assert torch.equal(tekfac._Q_Phi[layer], q_phi_at_refresh[3])
    assert not torch.equal(q_phi_at_refresh[0], q_phi_at_refresh[3])
    assert q_phi_at_refresh[3].shape == (3, 3)
