"""Unit-level checks for ``KFACApproximation``/``EKFACApproximation``, independent of the
brute-force oracle in ``test_frobenius_dominance.py``: output shapes, the ``pi`` toggle, and the
``T_inv``/``T_eig`` amortisation cadence (docs/reports/plan_lot2.md §2.3).

Lot 4 (docs/reports/plan_lot4.md §2.3) adds the ``Conv2d`` (``groups=1``, ``dilation=(1,1)``) analogue
of the shape/cadence checks below. Lot 5 (docs/reports/plan_lot5.md §2.3) adds the analogous
``BatchNorm2d``/``LayerNorm`` checks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from adafisher_modes.approximations.ekfac import EKFACApproximation
from adafisher_modes.approximations.kfac import KFACApproximation
from conftest import seed_all


def _drive(approx, layer, h, s, step):
    approx.update_input_factor(layer, h, step)
    approx.update_output_factor(layer, s, step)
    approx.refresh(layer, step)


def test_precondition_shapes_with_bias():
    seed_all(0)
    layer = nn.Linear(6, 4, bias=True)
    h, s = torch.randn(16, 6), torch.randn(16, 4)
    for approx in (KFACApproximation(T_inv=1), EKFACApproximation(T_eig=1)):
        _drive(approx, layer, h, s, 0)
        w, b = approx.precondition(layer, torch.randn(4, 6), torch.randn(4))
        assert w.shape == (4, 6)
        assert b.shape == (4,)


def test_precondition_shapes_without_bias():
    seed_all(1)
    layer = nn.Linear(6, 4, bias=False)
    h, s = torch.randn(16, 6), torch.randn(16, 4)
    for approx in (KFACApproximation(T_inv=1), EKFACApproximation(T_eig=1)):
        _drive(approx, layer, h, s, 0)
        w = approx.precondition(layer, torch.randn(4, 6), None)
        assert isinstance(w, torch.Tensor)
        assert w.shape == (4, 6)


def test_pi_false_reduces_to_one():
    seed_all(2)
    layer = nn.Linear(5, 3, bias=True)
    h, s = torch.randn(20, 5), torch.randn(20, 3)
    kfac = KFACApproximation(T_inv=1, pi=False)
    _drive(kfac, layer, h, s, 0)
    assert torch.equal(kfac._pi(kfac._A[layer], kfac._B[layer]), torch.tensor(1.0))


def test_pi_true_changes_damping_split():
    seed_all(3)
    layer = nn.Linear(5, 3, bias=True)
    h, s = torch.randn(20, 5), torch.randn(20, 3)

    kfac_pi = KFACApproximation(T_inv=1, pi=True, Lambda=1e-2)
    kfac_flat = KFACApproximation(T_inv=1, pi=False, Lambda=1e-2)
    for approx in (kfac_pi, kfac_flat):
        _drive(approx, layer, h, s, 0)

    A_tilde_pi, B_tilde_pi = kfac_pi._damped_factors(layer)
    A_tilde_flat, B_tilde_flat = kfac_flat._damped_factors(layer)
    assert not torch.allclose(A_tilde_pi, A_tilde_flat)
    assert not torch.allclose(B_tilde_pi, B_tilde_flat)


def test_kfac_inverse_cadence_respects_t_inv():
    seed_all(4)
    layer = nn.Linear(5, 3, bias=True)
    kfac = KFACApproximation(T_inv=3)

    for step in range(6):
        h, s = torch.randn(20, 5), torch.randn(20, 3)
        _drive(kfac, layer, h, s, step)
        if step % 3 == 0:
            cached_A_inv = kfac._A_inv[layer].clone()

    # After the loop, the cache reflects the last refresh step (3), not step 5.
    assert torch.equal(kfac._A_inv[layer], cached_A_inv)


def test_ekfac_eigenbasis_cadence_respects_t_eig():
    seed_all(5)
    layer = nn.Linear(5, 3, bias=True)
    ekfac = EKFACApproximation(T_eig=3)

    q_a_at_refresh = {}
    for step in range(6):
        h, s = torch.randn(20, 5), torch.randn(20, 3)
        _drive(ekfac, layer, h, s, step)
        if step % 3 == 0:
            q_a_at_refresh[step] = ekfac._Q_A[layer].clone()

    # Between refreshes (steps 1, 2 and 4, 5), the cached eigenbasis must not move.
    assert torch.equal(ekfac._Q_A[layer], q_a_at_refresh[3])
    assert not torch.equal(q_a_at_refresh[0], q_a_at_refresh[3])


# --------------------------------------------------------------------------------------------
# Lot 4: Conv2d (groups=1, dilation=(1,1)). See docs/reports/plan_lot4.md §2.3.
# --------------------------------------------------------------------------------------------


def test_precondition_shapes_with_bias_conv2d():
    seed_all(10)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
    for approx in (KFACApproximation(T_inv=1), EKFACApproximation(T_eig=1)):
        _drive(approx, layer, h, s, 0)
        w, b = approx.precondition(layer, torch.randn(3, 2, 3, 3), torch.randn(3))
        assert w.shape == (3, 2, 3, 3)
        assert b.shape == (3,)


def test_precondition_shapes_without_bias_conv2d():
    seed_all(11)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
    h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
    for approx in (KFACApproximation(T_inv=1), EKFACApproximation(T_eig=1)):
        _drive(approx, layer, h, s, 0)
        w = approx.precondition(layer, torch.randn(3, 2, 3, 3), None)
        assert isinstance(w, torch.Tensor)
        assert w.shape == (3, 2, 3, 3)


def test_kfac_inverse_cadence_respects_t_inv_conv2d():
    seed_all(12)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    kfac = KFACApproximation(T_inv=3)

    for step in range(6):
        h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
        _drive(kfac, layer, h, s, step)
        if step % 3 == 0:
            cached_A_inv = kfac._A_inv[layer].clone()

    assert torch.equal(kfac._A_inv[layer], cached_A_inv)


def test_ekfac_eigenbasis_cadence_respects_t_eig_conv2d():
    seed_all(13)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    ekfac = EKFACApproximation(T_eig=3)

    q_a_at_refresh = {}
    for step in range(6):
        h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
        _drive(ekfac, layer, h, s, step)
        if step % 3 == 0:
            q_a_at_refresh[step] = ekfac._Q_A[layer].clone()

    assert torch.equal(ekfac._Q_A[layer], q_a_at_refresh[3])
    assert not torch.equal(q_a_at_refresh[0], q_a_at_refresh[3])


# --------------------------------------------------------------------------------------------
# Lot 5: BatchNorm2d / LayerNorm (normalized_shape a 1-tuple). See docs/reports/plan_lot5.md §2.3.
# --------------------------------------------------------------------------------------------


def test_precondition_shapes_batchnorm2d():
    seed_all(20)
    layer = nn.BatchNorm2d(3)
    h, s = torch.randn(4, 3, 5, 5), torch.randn(4, 3, 5, 5)
    for approx in (KFACApproximation(T_inv=1), EKFACApproximation(T_eig=1)):
        _drive(approx, layer, h, s, 0)
        w, b = approx.precondition(layer, torch.randn(3), torch.randn(3))
        assert w.shape == (3,)
        assert b.shape == (3,)


def test_precondition_shapes_layernorm():
    seed_all(21)
    layer = nn.LayerNorm(8)
    h, s = torch.randn(16, 8), torch.randn(16, 8)
    for approx in (KFACApproximation(T_inv=1), EKFACApproximation(T_eig=1)):
        _drive(approx, layer, h, s, 0)
        w, b = approx.precondition(layer, torch.randn(8), torch.randn(8))
        assert w.shape == (8,)
        assert b.shape == (8,)


def test_kfac_inverse_cadence_respects_t_inv_batchnorm2d():
    seed_all(22)
    layer = nn.BatchNorm2d(3)
    kfac = KFACApproximation(T_inv=3)

    for step in range(6):
        h, s = torch.randn(4, 3, 5, 5), torch.randn(4, 3, 5, 5)
        _drive(kfac, layer, h, s, step)
        if step % 3 == 0:
            cached_A_inv = kfac._A_inv[layer].clone()

    assert torch.equal(kfac._A_inv[layer], cached_A_inv)


def test_ekfac_eigenbasis_cadence_respects_t_eig_layernorm():
    seed_all(23)
    layer = nn.LayerNorm(8)
    ekfac = EKFACApproximation(T_eig=3)

    q_a_at_refresh = {}
    for step in range(6):
        h, s = torch.randn(16, 8), torch.randn(16, 8)
        _drive(ekfac, layer, h, s, step)
        if step % 3 == 0:
            q_a_at_refresh[step] = ekfac._Q_A[layer].clone()

    assert torch.equal(ekfac._Q_A[layer], q_a_at_refresh[3])
    assert not torch.equal(q_a_at_refresh[0], q_a_at_refresh[3])


# --------------------------------------------------------------------------------------------
# Lot 6: the SUA approximation, Conv2d input factor only. See docs/reports/plan_lot6.md §2.3.
# --------------------------------------------------------------------------------------------


def test_precondition_shapes_with_bias_conv2d_sua():
    seed_all(40)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
    for approx in (
        KFACApproximation(T_inv=1, conv_sua=True),
        EKFACApproximation(T_eig=1, conv_sua=True),
    ):
        _drive(approx, layer, h, s, 0)
        w, b = approx.precondition(layer, torch.randn(3, 2, 3, 3), torch.randn(3))
        assert w.shape == (3, 2, 3, 3)
        assert b.shape == (3,)


def test_precondition_shapes_without_bias_conv2d_sua():
    seed_all(41)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
    h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
    for approx in (
        KFACApproximation(T_inv=1, conv_sua=True),
        EKFACApproximation(T_eig=1, conv_sua=True),
    ):
        _drive(approx, layer, h, s, 0)
        w = approx.precondition(layer, torch.randn(3, 2, 3, 3), None)
        assert isinstance(w, torch.Tensor)
        assert w.shape == (3, 2, 3, 3)


def test_kfac_sua_inverse_cadence_respects_t_inv_conv2d():
    seed_all(42)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    kfac = KFACApproximation(T_inv=3, conv_sua=True)

    for step in range(6):
        h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
        _drive(kfac, layer, h, s, step)
        if step % 3 == 0:
            cached_A_inv = kfac._A_inv[layer].clone()

    assert torch.equal(kfac._A_inv[layer], cached_A_inv)
    assert kfac._A_inv[layer].shape == (3, 3)  # C_in+1, not C_in*k_h*k_w+1


def test_ekfac_sua_eigenbasis_cadence_respects_t_eig_conv2d():
    seed_all(43)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
    ekfac = EKFACApproximation(T_eig=3, conv_sua=True)

    q_a_at_refresh = {}
    for step in range(6):
        h, s = torch.randn(6, 2, 5, 5), torch.randn(6, 3, 5, 5)
        _drive(ekfac, layer, h, s, step)
        if step % 3 == 0:
            q_a_at_refresh[step] = ekfac._Q_A[layer].clone()

    assert torch.equal(ekfac._Q_A[layer], q_a_at_refresh[3])
    assert not torch.equal(q_a_at_refresh[0], q_a_at_refresh[3])
    assert q_a_at_refresh[3].shape == (3, 3)
