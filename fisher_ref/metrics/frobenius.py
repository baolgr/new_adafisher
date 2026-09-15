"""M1 — entry-wise fidelity (``plan_exp_draft.md`` §5).

``e_F = ||R - K||_F / ||R||_F`` is the headline, but on its own it conflates two different errors,
and ``plan_exp_lot2.md`` §5.1 is why that matters: two references built on different probe sets
differ in *norm* by 30-60 % before any structure is involved. So M1 reports three numbers:

* ``e_F``     — the raw relative gap, scale included;
* ``cos_F``   — ``<R, K> / (||R||_F ||K||_F)``, pure direction, scale removed;
* ``e_F_star``— ``sqrt(1 - cos_F^2)``, the gap that survives the best rescaling ``c*``, i.e.
  ``min_c ||R - cK||_F / ||R||_F``.

``e_F`` large with ``cos_F`` near 1 means "the right shape, the wrong size" — a damping or scaling
question. ``e_F ~ e_F_star`` means the structure genuinely points elsewhere. **HF1 is settled by
this distinction**, not by ``e_F`` (``plan_exp_lot2.md`` §5.1).

**One numerical caveat, and it bites exactly where the campaign looks.** ``e_F_star`` is
``sqrt(1 - cos_F^2)``, which loses half its significant digits as ``cos_F -> 1``: a ``1e-16``
relative error in ``cos_F`` becomes ``1.5e-8`` in ``e_F_star``. So **fp64 floors ``e_F_star`` at
about ``sqrt(eps) ~ 1.5e-8``**, and a reported value below ``1e-7`` means "indistinguishable from a
pure rescaling", not a measurement. The cancellation is intrinsic to the quantity — writing it as
``sqrt(||R||^2 - <R,K>^2/||K||^2)/||R||`` is algebraically identical and cancels identically — so it
is documented rather than worked around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch import Tensor

from ..approx.base import BlockOps


@dataclass(frozen=True)
class FrobeniusReport:
    e_F: float
    cos_F: float
    e_F_star: float
    c_star: float
    fro_R: float
    fro_K: float
    inner: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_rows(self, **keys: Any) -> list:
        """One ``metrics.csv`` row per metric name (``plan_exp_draft.md`` §13's long format)."""
        return [{**keys, "metric": name, "value": value} for name, value in
                (("e_F", self.e_F), ("cos_F", self.cos_F), ("e_F_star", self.e_F_star),
                 ("c_star", self.c_star), ("fro_R", self.fro_R), ("fro_K", self.fro_K))]


def frobenius(R: Tensor, K: BlockOps, *, fro_R: Optional[float] = None) -> FrobeniusReport:
    """M1, without materialising ``K``.

    ``||R - K||_F^2 = ||R||^2 - 2<R,K> + ||K||^2`` — three quantities every ``CurvatureBlock``
    exposes in closed form, which is the whole reason ``inner_dense`` is part of the protocol.
    """
    inner = float(K.inner_dense(R))
    fro_k2 = float(K.fro2())
    fro_r2 = float((R * R).sum()) if fro_R is None else fro_R ** 2
    gap2 = max(fro_r2 - 2.0 * inner + fro_k2, 0.0)
    fro_r = fro_r2 ** 0.5
    fro_k = fro_k2 ** 0.5
    cos = inner / (fro_r * fro_k) if fro_r > 0 and fro_k > 0 else float("nan")
    return FrobeniusReport(
        e_F=gap2 ** 0.5 / fro_r if fro_r > 0 else float("nan"),
        cos_F=cos,
        e_F_star=max(1.0 - cos * cos, 0.0) ** 0.5,
        c_star=inner / fro_k2 if fro_k2 > 0 else float("nan"),
        fro_R=fro_r, fro_K=fro_k, inner=inner,
    )


def dense_gap(A: Tensor, B: Tensor, block: int = 4096) -> Dict[str, float]:
    """``||A - B||_F`` with **both** norms, accumulated over row blocks.

    Never returns a bare ratio: ``plan_exp_lot1.md`` §6.3 was unresolvable precisely because three
    gaps were reported against two denominators and only one norm was kept
    (``plan_exp_lot2.md`` §0.1).
    """
    gap2 = a2 = b2 = 0.0
    for start in range(0, A.shape[0], block):
        stop = min(start + block, A.shape[0])
        rows_a, rows_b = A[start:stop], B[start:stop]
        gap2 += float((rows_a - rows_b).pow(2).sum())
        a2 += float(rows_a.pow(2).sum())
        b2 += float(rows_b.pow(2).sum())
    inner = 0.5 * (a2 + b2 - gap2)
    return {"abs": gap2 ** 0.5, "fro_a": a2 ** 0.5, "fro_b": b2 ** 0.5,
            "rel_to_a": (gap2 / a2) ** 0.5 if a2 else float("nan"),
            "rel_to_b": (gap2 / b2) ** 0.5 if b2 else float("nan"),
            "rel_to_geom": gap2 ** 0.5 / (a2 * b2) ** 0.25 if a2 and b2 else float("nan"),
            "cos": inner / (a2 * b2) ** 0.5 if a2 and b2 else float("nan")}


__all__ = ["FrobeniusReport", "dense_gap", "frobenius"]
