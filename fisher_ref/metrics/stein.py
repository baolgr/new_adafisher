"""The Kullback-Leibler gap between the two preconditioner Gaussians.

    D_lam(K || R) = 0.5 [ tr(K_lam R_lam^{-1}) - P - logdet K_lam + logdet R_lam ]

which is the KL divergence from ``N(0, R_lam^{-1})`` to ``N(0, K_lam^{-1})``. It is the
**affine-invariant** comparison: unlike a Frobenius gap it does not care how the two operators are
scaled relative to each other, only how one would precondition the other's geometry. Applied to the
reference against itself it is exactly zero, which is the self-consistency check.

On a materialised reference this is affordable for a reason worth stating: a Cholesky is ``P^3/3``
and threads well, against the ``(4/3) P^3`` of a symmetric eigendecomposition, which on the
machines this campaign runs on does not thread at all. At ``P`` around 27 000 the
eigendecomposition took 1 280 s while the Cholesky takes minutes.

Public API
----------

:func:`dense_logdet`  ``logdet(M + lam I)`` by Cholesky.

:func:`stein_kl`  the gap. ``tr(K_lam R_lam^{-1})`` is computed by solving ``R_lam X = K_lam`` once,
with a single Cholesky reused for the solve and for ``logdet R_lam``. ``k_lam`` is the damping
applied to ``K`` when it is not the reference's; it exists so that an operator carrying its own
damping is described by the **same** matrix in both halves of the expression, since overriding only
the solve would leave the log-determinant talking about a different object.

:func:`lambda_grid`  ``lam = alpha * tr(R)/P`` over a list of ``alpha``. No metric that goes through
an inverse is reported at a single damping: on a real network a fifth of the directions have
exactly zero curvature, so every damped inverse reads ``lam I`` there whatever ``alpha`` is, and the
sweep is what separates "the approximation is wrong" from "the damping is doing the work".

Dependencies: :mod:`fisher_ref.approx.base`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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


def stein_kl(R: Tensor, K: BlockOps, lam: float, *, k_lam: Optional[float] = None) -> SteinReport:
    """``D_lam(K || R)`` in regime A.

    ``tr(K_lam R_lam^{-1})`` is computed by solving ``R_lam X = K_lam`` once — a single Cholesky
    reused for the solve and for ``logdet R_lam``, rather than two factorisations.

    ``k_lam`` is the damping applied to ``K``, when it is not the reference's; ``None`` (the
    default) means ``k_lam = lam`` and is bit-identical to this function before lot 5. It exists so
    that an operator carrying its own damping — P2's, ``plan_exp_lot5.md`` §0.3 — is described by
    the **same** matrix in both halves of the expression: overriding only ``K.solve`` would leave
    ``logdet K`` talking about a different object from the trace term.
    """
    size = R.shape[0]
    k_damping = lam if k_lam is None else k_lam
    identity = torch.eye(size, dtype=R.dtype, device=R.device)
    damped_R = R + lam * identity
    factor = torch.linalg.cholesky(damped_R)
    logdet_R = 2.0 * torch.log(factor.diagonal()).sum()

    damped_K = K.to_dense() + k_damping * identity
    solved = torch.cholesky_solve(damped_K, factor)
    trace_term = solved.diagonal().sum()
    logdet_K = K.logdet(k_damping)

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
