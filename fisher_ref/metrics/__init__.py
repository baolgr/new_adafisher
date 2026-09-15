"""The metrics (``plan_exp_draft.md`` §5, lot 2 of its §9).

``plan_exp_draft.md`` §0.11 fixes the priority: **M1, M5, M7, M8 core, M3 where a Cholesky is
affordable**; M2, M4 and M6 are options that need Lanczos machinery to answer what M1 and M5
already answer, and are not built here.

Each metric takes a dense reference ``R`` and a ``CurvatureBlock`` ``K``, so it is written once for
every structure — and, for the ones that go through an inverse, a damping ``lam`` from
:func:`~fisher_ref.metrics.stein.lambda_grid`, never a single fixed value (§3.3, §10.3).
"""

from __future__ import annotations

from .coupling import coupling_matrix, offdiagonal_mass
from .frobenius import FrobeniusReport, dense_gap, frobenius
from .kron_diag import KroneckerReport, best_kronecker_fit, kronecker_structure
from .ngd import NgdReport, probe_gradient, rho
from .noise_floor import NoiseFloor, convergence_curve, noise_floor
from .stein import SteinReport, dense_logdet, lambda_grid, stein_kl

__all__ = ["FrobeniusReport", "KroneckerReport", "NgdReport", "NoiseFloor", "SteinReport",
           "best_kronecker_fit", "convergence_curve", "coupling_matrix", "dense_gap",
           "dense_logdet", "frobenius", "kronecker_structure", "lambda_grid", "noise_floor",
           "offdiagonal_mass", "probe_gradient", "rho", "stein_kl"]
