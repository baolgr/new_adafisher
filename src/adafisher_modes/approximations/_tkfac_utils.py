"""The three per-batch statistics the ``tkfac`` and ``tekfac`` modes share.

Both modes build their curvature from the same trace-restricted triple of TKFAC's Theorem 4.1 /
eq. (4.9) (``tkfac_2011.10741.pdf``), which TEKFAC restates as its own eq. (2.9)-(2.10)
(``tekfac_2011.13609.pdf``)::

    delta   = E_n[ ||h_bar_n||^2 * ||delta_n||^2 ]
    Phi_raw = E_n[ ||delta_n||^2 * h_bar_n h_bar_n^T ]     ( = delta * Phi )
    Psi_raw = E_n[ ||h_bar_n||^2 * delta_n delta_n^T ]     ( = delta * Psi )

``h_bar_n`` is one example's bias-augmented input and ``delta_n`` the gradient at the layer's output
for the same example; the two are row-paired.

**The numerators are returned undivided on purpose.** ``tr(Phi_raw) = tr(Psi_raw) = delta`` holds
term by term, so it survives any running average that scales both sides by the same coefficients --
including this package's, whose coefficients do not sum to 1. Dividing by ``delta`` before
accumulating would break that identity and with it TKFAC's trace-preservation theorem. Do not
"simplify" this by storing ``Phi`` and ``Psi`` directly.

:func:`bootstrap_raw_factors` supplies the step-0 starting values, ``delta_0 = d_in * d_out``,
``Phi_raw_0 = d_out * I`` and ``Psi_raw_0 = d_in * I``. They are the unique choice that both
satisfies the same trace identity and makes the resulting preconditioner the identity, so the
bootstrap is inert in the same way ``kfac``'s identity seed and ``diag``'s all-ones seed are.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def instantaneous_raw_factors(h_bar: Tensor, s: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """``(delta_i, Phi_raw_i, Psi_raw_i)``, the un-normalized numerators of TKFAC Eq. (4.9):

        delta_i   = mean_n[ tr(Lambda_n)*tr(Gamma_n) ] = mean_n[ ||h_bar_n||^2 * ||s_n||^2 ]
        Phi_raw_i = mean_n[ tr(Gamma_n)*Lambda_n ]      = h_bar^T diag(||s_n||^2) h_bar / N
        Psi_raw_i = mean_n[ tr(Lambda_n)*Gamma_n ]      = s^T diag(||h_bar_n||^2) s / N

    with ``Lambda_n = h_bar_n h_bar_n^T``, ``Gamma_n = s_n s_n^T`` (per-example, paired). ``Phi_l =
    Phi_raw/delta`` and ``Psi_l = Psi_raw/delta`` are Theorem 4.1's actual factors; the division is
    left to the caller so that the running average accumulates the numerators, not the ratio.
    """
    norm_a = (h_bar**2).sum(dim=1)  # tr(Lambda_n) = ||h_bar_n||^2
    norm_g = (s**2).sum(dim=1)  # tr(Gamma_n)  = ||s_n||^2
    n = h_bar.size(0)
    delta_i = (norm_a * norm_g).mean()
    phi_raw_i = (h_bar * norm_g.unsqueeze(1)).t() @ h_bar / n
    psi_raw_i = (s * norm_a.unsqueeze(1)).t() @ s / n
    return delta_i, phi_raw_i, psi_raw_i


def bootstrap_raw_factors(
    d_in: int, d_out: int, dtype: torch.dtype, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor]:
    """``delta_0 = d_in*d_out``, ``Phi_raw_0 = d_out*I_{d_in}``, ``Psi_raw_0 = d_in*I_{d_out}``.

    The unique bootstrap that (a) satisfies ``tr(Phi_raw_0) = tr(Psi_raw_0) = delta_0`` exactly —
    the same trace identity every subsequent EMA'd triple satisfies by construction — and (b)
    reduces to ``F~ = delta_0 * (Phi_raw_0/delta_0) (x) (Psi_raw_0/delta_0) = I``, an inert
    preconditioner, matching ``kfac``/``ekfac``'s own ``eye``/``ones`` bootstraps.
    """
    delta_0 = torch.tensor(float(d_in * d_out), dtype=dtype, device=device)
    phi_raw_0 = d_out * torch.eye(d_in, dtype=dtype, device=device)
    psi_raw_0 = d_in * torch.eye(d_out, dtype=dtype, device=device)
    return delta_0, phi_raw_0, psi_raw_0
