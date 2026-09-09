"""Exit criterion 3 of lot 3 (docs/reports/plan_lot3.md §2.3, docs/reports/plan.md §6.2):

Degenerate-case consistency between EKFAC and TEKFAC. Forcing ``tr(Phi)=tr(Psi)=1`` then
substituting ``Phi := A``, ``Psi := B``, ``delta := 1`` gives ``Q_Phi = Q_A`` and ``Q_Psi = Q_B``,
so TEKFAC's re-scaling factor ``Theta`` (Eq. 3.2 of ``tekfac_2011.13609.pdf``) and EKFAC's ``s*``
(``ekfac_1806.03884.pdf`` §3.2) become **the same quantity**, estimated by the same intra-batch
formula in the same eigenbasis. The two preconditioners must then agree exactly.

This substitution is applied directly to ``TEKFACApproximation``'s internal ``_Phi_raw``/``_Psi_raw``
(bypassing its own weighted `(delta, Phi, Psi)` estimator entirely — that estimator is TKFAC's own
concern, already covered by ``test_frobenius_dominance.py``; this test isolates the eigenbasis
correction and its application, which is what TEKFAC actually adds on top of TKFAC).

Precaution (``plan.md``'s own listed testing pitfall): ``torch.linalg.eigh`` fixes neither the sign
nor the ordering of eigenvectors in general — a non-issue here because both ``ekfac`` and ``tekfac``
``eigh`` the exact same matrix values (``_Phi_raw``/``_Psi_raw`` are set to clones of ``_A``/``_B``),
so both calls make the identical sign/order choice. The comparison below is on the **applied
preconditioners**, never on the bases themselves, per that same pitfall.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from adafisher_modes.approximations.ekfac import EKFACApproximation
from adafisher_modes.approximations.tekfac import TEKFACApproximation
from adafisher_modes.ema import update_running_avg
from adafisher_modes.factors import augment_linear_input
from conftest import seed_all

D_IN, D_OUT, N = 7, 5, 64


def test_ekfac_tekfac_agree_in_degenerate_case() -> None:
    seed_all(0)
    h, s = torch.randn(N, D_IN), torch.randn(N, D_OUT)
    layer = nn.Linear(D_IN, D_OUT, bias=True)

    ekfac = EKFACApproximation(Lambda=1e-3, gammas=(1.0, 1.0), T_eig=1)
    ekfac.update_input_factor(layer, h, step=0)
    ekfac.update_output_factor(layer, s, step=0)
    ekfac.refresh(layer, step=0)  # Q_A, Q_B from the exact A, B; s* bootstrapped to ones
    ekfac.update_input_factor(layer, h, step=1)
    ekfac.update_output_factor(layer, s, step=1)  # s* <- exact intra-batch estimate on (h, s)

    tekfac = TEKFACApproximation(Lambda=1e-3, gammas=(1.0, 1.0), T_eig=1, T_re=1)
    # Degenerate substitution (plan.md §6.2): Phi := A, Psi := B (delta implicitly 1 -- neither
    # f_tilde nor precondition ever multiplies Theta by delta, plan_lot3.md §0.4), bypassing
    # TEKFACApproximation's own (delta, Phi, Psi) estimator entirely.
    tekfac._Phi_raw[layer] = ekfac._A[layer].clone()
    tekfac._Psi_raw[layer] = ekfac._B[layer].clone()
    tekfac.refresh(layer, step=0)  # Q_Phi = Q_A, Q_Psi = Q_B (eigh on identical matrices); Theta -> ones

    # Manually replicate Theta's intra-batch update (Eq. 3.2) on the SAME (h, s) ekfac used for
    # s*, without going through update_output_factor (which would also re-estimate _Phi_raw/
    # _Psi_raw via TEKFAC's own weighted formula, moving them away from the injected A, B).
    h_bar = augment_linear_input(h, layer)
    h_kfe = h_bar @ tekfac._Q_Phi[layer]
    s_kfe = s @ tekfac._Q_Psi[layer]
    theta_i = (s_kfe.t() ** 2) @ (h_kfe**2) / h_bar.size(0)
    update_running_avg(theta_i, tekfac._Theta[layer], tekfac.beta_theta)

    weight_direction = torch.randn(D_OUT, D_IN)
    bias_direction = torch.randn(D_OUT)
    w_ekfac, b_ekfac = ekfac.precondition(layer, weight_direction, bias_direction)
    w_tekfac, b_tekfac = tekfac.precondition(layer, weight_direction, bias_direction)

    assert torch.allclose(w_ekfac, w_tekfac, rtol=1e-5, atol=1e-6)
    assert torch.allclose(b_ekfac, b_tekfac, rtol=1e-5, atol=1e-6)
