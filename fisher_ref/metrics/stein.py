"""M3 — the Stein / KL gap between the two preconditioner Gaussians (``plan_exp_draft.md`` §5).

``D_lam(K || R) = 0.5 [ tr(K_lam R_lam^{-1}) - P - logdet K_lam + logdet R_lam ]``, the KL between
``N(0, R_lam^{-1})`` and ``N(0, K_lam^{-1})``. It is the **affine-invariant** comparison: unlike
``e_F`` it does not care how the two operators are scaled relative to each other, only how one
would precondition the other's geometry.

Regime A computes it densely, and that is affordable for a reason worth stating: a Cholesky is
``P^3/3`` and threads, against ``syevd``'s ``(4/3)P^3`` which — measured in ``plan_exp_lot1.md``
§6.1 — does not. At A1's ``P = 26 634`` the eigendecomposition took 1 280 s; the Cholesky is minutes
(``plan_exp_lot2.md`` §0.5). §5.1's Woodbury form is regime B's, i.e. lot 4's.

T7 is the self-consistency check: ``D_lam(R || R) = 0`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import Tensor

from ..approx.base import BlockOps


@dataclass(frozen=True)
class SteinReport:
    kl: float
    trace_term: float
    logdet_K: float
    logdet_R: float
    P: int
    lam: float

    def as_rows(self, **keys: Any) -> list:
        return [{**keys, "metric": name, "value": value} for name, value in
                (("stein_kl", self.kl), ("stein_trace", self.trace_term))]


def dense_logdet(matrix: Tensor, lam: float) -> Tensor:
    """``logdet(M + lam I)`` by Cholesky — the cheap half of M3 (``plan_exp_lot2.md`` §0.5)."""
    damped = matrix + lam * torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    factor = torch.linalg.cholesky(damped)
    return 2.0 * torch.log(factor.diagonal()).sum()


def stein_kl(R: Tensor, K: BlockOps, lam: float) -> SteinReport:
    """``D_lam(K || R)`` in regime A.

    ``tr(K_lam R_lam^{-1})`` is computed by solving ``R_lam X = K_lam`` once — a single Cholesky
    reused for the solve and for ``logdet R_lam``, rather than two factorisations.
    """
    size = R.shape[0]
    identity = torch.eye(size, dtype=R.dtype, device=R.device)
    damped_R = R + lam * identity
    factor = torch.linalg.cholesky(damped_R)
    logdet_R = 2.0 * torch.log(factor.diagonal()).sum()

    damped_K = K.to_dense() + lam * identity
    solved = torch.cholesky_solve(damped_K, factor)
    trace_term = solved.diagonal().sum()
    logdet_K = K.logdet(lam)

    kl = 0.5 * (trace_term - size - logdet_K + logdet_R)
    return SteinReport(kl=float(kl), trace_term=float(trace_term), logdet_K=float(logdet_K),
                       logdet_R=float(logdet_R), P=size, lam=lam)


def lambda_grid(R: Tensor, alphas: Any = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)) -> Dict[float, float]:
    """``lam = alpha * lam_bar`` with ``lam_bar = tr(R)/P`` (``plan_exp_draft.md`` §3.3).

    No metric that goes through an inverse is ever reported at a single ``lam``: §10.3 makes the
    sweep a reading rule. It matters more on A1 than the plan expected — ``plan_exp_lot1.md`` §6.4
    measured ``F``'s kernel at 4 805 dimensions, so every damped inverse reads ``lam I`` on a fifth
    of the space whatever ``alpha`` is, and the sweep is what separates "the approximation is wrong"
    from "the damping is doing the work".
    """
    lam_bar = float(R.diagonal().sum() / R.shape[0])
    return {float(alpha): float(alpha) * lam_bar for alpha in alphas}


__all__ = ["SteinReport", "dense_logdet", "lambda_grid", "stein_kl"]
