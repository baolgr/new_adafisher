"""Shared statistics for the two trace-restricted modes (TKFAC, TEKFAC): the per-batch,
un-normalized numerators entering TKFAC's Theorem 4.1 / Eq. (4.9) (``tkfac_2011.10741.pdf``) and
TEKFAC's identical Eq. (2.9)-(2.10) (``tekfac_2011.13609.pdf``), and their bootstrap.

Kept un-normalized (not divided by ``delta``) on purpose, and shared between ``tkfac.py`` and
``tekfac.py`` rather than duplicated: dividing before accumulating would silently break the exact
``tr(Phi)=tr(Psi)=1`` invariant these un-normalized numerators guarantee under this project's own
EMA (``ema.py``, coefficients that do not sum to 1). See ``docs/reports/plan_lot3.md`` §0.2 for the
full derivation. This is the lot-3 analogue of ``_kron_utils.py`` (``plan_lot2.md`` §0.2): a small
private module justified by two consumers sharing non-trivial, easy-to-drift statistics.
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
    Phi_raw/delta`` and ``Psi_l = Psi_raw/delta`` are Theorem 4.1's actual factors; kept divided
    apart here so the caller's EMA accumulates the numerators, not the ratio (plan_lot3.md §0.2).
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
