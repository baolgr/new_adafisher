"""M7 — the independence bias, the weight-sharing share, and the diagonal bias
(``plan_exp_draft.md`` §5).

All three read off the **rearrangement** ``R(B)`` of ``approx.base.rearrange``:

* ``sigma_2 / sigma_1`` and the tail mass ``sum_{i>1} sigma_i^2 / sum sigma_i^2`` of ``R(B_l)``
  measure how far the exact block is from *any* Kronecker product — the independence bias, with the
  best rank-1 fit being the top singular pair (Van Loan-Pitsianis; Koroko et al. arXiv:2201.10285
  did exactly this on auto-encoders). A block that **is** Kronecker has ``sigma_2/sigma_1 = 0``.
* ``||B_l - B_l^exp||_F / ||B_l||_F`` is the weight-sharing share: what dropping the cross-position
  terms costs. Zero without sharing, which is A1 — so this column is lot 3's, on A2/A3.
* ``||diag B_l - diag(A) (x) diag(G)||/||diag B_l||`` is the diagonal bias, i.e. **HF3**: without
  sharing it is entirely ``Cov(a_j^2, g_i^2)`` (``plan_exp_draft.md`` §3.1).
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


__all__ = ["KroneckerReport", "best_kronecker_fit", "kronecker_structure"]
