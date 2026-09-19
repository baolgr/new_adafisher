"""How far an exact block is from any Kronecker product, and from any diagonal.

All of it reads off the **rearrangement** ``R(B)`` of :func:`fisher_ref.approx.base.rearrange`:

* ``sigma_2 / sigma_1`` and the tail mass ``sum_{i>1} sigma_i^2 / sum_i sigma_i^2`` of ``R(B)``
  measure how far the exact block is from *any* Kronecker product, the best rank-one fit being the
  top singular pair (Van Loan and Pitsianis; used on auto-encoder curvature by Koroko et al.,
  arXiv:2201.10285). A block that **is** a Kronecker product has ``sigma_2 / sigma_1 = 0``.
* ``||B - B^exp||_F / ||B||_F`` is the weight-sharing share: what dropping the cross-position terms
  costs. It is zero on a model with one position per example.
* ``||diag B - diag(A) (x) diag(G)|| / ||diag B||`` is the diagonal bias. Without weight sharing it
  is entirely the covariance of ``a_j^2`` and ``g_i^2``.

Public API
----------

:class:`KroneckerReport`  the ratio, the tail mass, the best rank-one gap, and optionally the
diagonal bias and the sharing share.

:func:`kronecker_structure`  the report from a full singular-value decomposition of ``R(B)``.

:func:`best_kronecker_fit`  ``(A, G)`` minimising ``||B - G (x) A||_F``, the top singular pair
reshaped.

:func:`kronecker_analysis`  both of the above from **one** spectral decomposition. The Gram of the
*smaller* side of ``R(B)`` is diagonalised once, which yields every squared singular value and the
top singular pair; a thin decomposition of the rearrangement itself also returns a right-singular
matrix nobody reads, gigabytes wide on a convolution. Precision: squaring costs half the digits on
the *small* singular values only, so ``sigma_2/sigma_1`` is exact to about
``eps (sigma_1/sigma_2)^2``, and the tail mass is taken as ``1 - sigma_1^2/||R(B)||_F^2`` with the
norm summed directly, so it is exact to ``eps`` over the tail.

Dependencies: :mod:`fisher_ref.approx.base`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor

from ..approx.base import Diag, rearrange


@dataclass(frozen=True)
class KroneckerReport:
    sigma_ratio: float
    tail_mass: float
    best_rank1_gap: float
    diagonal_bias: Optional[float] = None
    sharing_share: Optional[float] = None

    def as_rows(self, **keys: Any) -> list:
        rows = [{**keys, "metric": name, "value": value} for name, value in
                (("sigma2_over_sigma1", self.sigma_ratio), ("kron_tail_mass", self.tail_mass),
                 ("best_rank1_gap", self.best_rank1_gap))]
        if self.diagonal_bias is not None:
            rows.append({**keys, "metric": "diagonal_bias", "value": self.diagonal_bias})
        if self.sharing_share is not None:
            rows.append({**keys, "metric": "sharing_share", "value": self.sharing_share})
        return rows


def kronecker_structure(block: Tensor, d_out: int, d_in: int,
                        af_raw: Optional[Diag] = None) -> KroneckerReport:
    """M7 on one exact block, already in ``rvec([W | b])`` order."""
    rearranged = rearrange(block, d_out, d_in)
    singular = torch.linalg.svdvals(rearranged)
    total = float((singular * singular).sum())
    top = float(singular[0])
    ratio = float(singular[1] / singular[0]) if singular.numel() > 1 and top > 0 else 0.0
    tail = float((singular[1:] * singular[1:]).sum() / total) if total > 0 else 0.0
    # ||B - best rank-1 Kronecker||_F = the tail's own norm, by the rearrangement isometry.
    rank1_gap = float((singular[1:] * singular[1:]).sum() ** 0.5 / total ** 0.5) if total else 0.0

    diagonal_bias = None
    if af_raw is not None:
        exact_diagonal = block.diagonal()
        norm = float(exact_diagonal.norm())
        diagonal_bias = float((exact_diagonal - af_raw.values).norm() / norm) if norm else None
    return KroneckerReport(sigma_ratio=ratio, tail_mass=tail, best_rank1_gap=rank1_gap,
                           diagonal_bias=diagonal_bias)


def best_kronecker_fit(block: Tensor, d_out: int, d_in: int) -> "tuple":
    """``(A, G)`` minimising ``||B - G (x) A||_F`` — the top singular pair of ``R(B)``, reshaped.

    Reported alongside K-FAC's own factors: the gap between the two is how much of K-FAC's error is
    the *independence assumption* rather than the Kronecker form itself.
    """
    rearranged = rearrange(block, d_out, d_in)
    u, s, vh = torch.linalg.svd(rearranged, full_matrices=False)
    scale = s[0].sqrt()
    G = (u[:, 0] * scale).reshape(d_out, d_out)
    A = (vh[0] * scale).reshape(d_in, d_in)
    return A, G


def kronecker_analysis(rearranged: Tensor, d_out: int, d_in: int, *,
                       exact_diagonal: Optional[Tensor] = None,
                       kfac_diagonal: Optional[Tensor] = None) -> "tuple":
    """M7 and the best Kronecker fit from **one** spectral decomposition of ``R(B)``.

    Lot 2 called ``svdvals`` for M7 and a full ``svd`` for the best fit: two decompositions of a
    ``(d_out^2, d_in^2)`` matrix — ``4 096 x 83 521`` on A2's last convolution, whose thin ``svd``
    also returns a 2.7 GB ``V^H`` nobody reads. Here the Gram of the *smaller* side is diagonalised
    once (``4 096^2``), which yields every squared singular value and the top singular pair.

    Precision: squaring costs half the digits on the *small* singular values only. ``sigma_2/sigma_1``
    is exact to ``~eps (sigma_1/sigma_2)^2`` (irrelevant above ``1e-4``), and the tail mass is taken as
    ``1 - sigma_1^2/||R||_F^2`` with ``||R||_F^2`` summed directly, so it is exact to ``eps/tail``.

    Returns ``(KroneckerReport, A_best, G_best)``; ``diagonal_bias`` compares ``exact_diagonal`` to
    ``kfac_diagonal`` (K-FAC-expand's own diagonal, ``plan_exp_lot3.md`` §0.4) when both are given.
    """
    rows, cols = rearranged.shape
    total = float((rearranged * rearranged).sum())
    if rows <= cols:
        values, vectors = torch.linalg.eigh(rearranged @ rearranged.T)
    else:
        values, vectors = torch.linalg.eigh(rearranged.T @ rearranged)
    values = values.flip(0).clamp_min(0.0)
    sigma1 = float(values[0].sqrt())
    top = vectors[:, -1]
    if rows <= cols:
        left, right = top, (rearranged.T @ top) / sigma1 if sigma1 > 0 else top
    else:
        left, right = (rearranged @ top) / sigma1 if sigma1 > 0 else top, top
    scale = sigma1 ** 0.5
    G = (left * scale).reshape(d_out, d_out)
    A = (right * scale).reshape(d_in, d_in)

    ratio = float((values[1] / values[0]).sqrt()) if values.numel() > 1 and sigma1 > 0 else 0.0
    tail = max(1.0 - sigma1 * sigma1 / total, 0.0) if total > 0 else 0.0
    diagonal_bias = None
    if exact_diagonal is not None and kfac_diagonal is not None:
        norm = float(exact_diagonal.norm())
        diagonal_bias = float((exact_diagonal - kfac_diagonal).norm() / norm) if norm else None
    report = KroneckerReport(sigma_ratio=ratio, tail_mass=tail, best_rank1_gap=tail ** 0.5,
                             diagonal_bias=diagonal_bias)
    return report, A, G


__all__ = ["KroneckerReport", "best_kronecker_fit", "kronecker_analysis", "kronecker_structure"]
