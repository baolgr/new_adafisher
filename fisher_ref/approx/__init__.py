"""The approximation zoo (``plan_exp_draft.md`` §4, lot 2 of its §9).

P1 owns these implementations rather than calling ``adafisher_modes``, for the reason §7.2 of the
plan gives: the optimizer computes EMA'd, damped, hook-driven factors at training precision and has
no *source* parameter, while P1 needs the structure alone, in fp64, from type-2 vectors. **T13** is
what makes that safe — each structure here is pinned to the optimizer's own at the degenerate
setting (one update, identity EMA, empirical source, ``lambda = 0``).
"""

from __future__ import annotations

from .base import (
    BlockDiag,
    BlockOps,
    CurvatureBlock,
    Dense,
    Diag,
    block_diagonal_of,
    optimal_scale,
    rearrange,
)
from .kfac import Kron, af_raw_from_factors, kfac_from_factors

__all__ = ["BlockDiag", "BlockOps", "CurvatureBlock", "Dense", "Diag", "Kron",
           "af_raw_from_factors", "block_diagonal_of", "kfac_from_factors", "optimal_scale",
           "rearrange"]
