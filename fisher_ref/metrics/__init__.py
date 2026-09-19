"""The comparisons between a curvature reference and a structured approximation.

Five families are built, chosen because they answer different questions and because each is
affordable on a materialised reference:

``frobenius``  entry-wise fidelity, split into a raw gap, a pure-direction cosine, and the gap that
survives the best rescaling.
``ngd``  the fraction of the optimal quadratic decrease obtained along the preconditioned step.
``stein``  the Kullback-Leibler gap between the two preconditioner Gaussians, which is invariant
under a change of parameter basis.
``kron_diag``  how far the exact block is from *any* Kronecker product, and from any diagonal.
``coupling``  how much curvature lives between two layers rather than inside one.
``noise_floor``  the reference's own sampling error, below which no difference is interpretable.

Every metric takes a dense reference and a block from :mod:`fisher_ref.approx`, so it is written
once for every structure. The ones that go through an inverse take a damping, and are never
reported at a single value: :func:`fisher_ref.metrics.stein.lambda_grid` produces the sweep.

Metrics that would need Lanczos machinery to answer what these already answer are deliberately not
built.
"""

from __future__ import annotations

from .coupling import coupling_matrix, offdiagonal_mass
from .frobenius import FrobeniusReport, dense_gap, frobenius
from .kron_diag import (
    KroneckerReport,
    best_kronecker_fit,
    kronecker_analysis,
    kronecker_structure,
)
from .ngd import NgdReport, damped_cholesky, probe_gradient, rho
from .noise_floor import NoiseFloor, convergence_curve, noise_floor
from .stein import SteinReport, dense_logdet, lambda_grid, stein_kl

__all__ = ["FrobeniusReport", "KroneckerReport", "NgdReport", "NoiseFloor", "SteinReport",
           "best_kronecker_fit", "convergence_curve", "coupling_matrix", "damped_cholesky",
           "dense_gap", "dense_logdet", "frobenius", "kronecker_analysis", "kronecker_structure", "lambda_grid",
           "noise_floor", "offdiagonal_mass", "probe_gradient", "rho", "stein_kl"]
