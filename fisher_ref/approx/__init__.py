"""The approximation zoo (``plan_exp_draft.md`` §4, lot 2 of its §9).

P1 owns these implementations rather than calling ``adafisher_modes``, for the reason §7.2 of the
plan gives: the optimizer computes EMA'd, damped, hook-driven factors at training precision and has
no *source* parameter, while P1 needs the structure alone, in fp64, from type-2 vectors. **T13** is
what makes that safe — each structure here is pinned to the optimizer's own at the degenerate
setting (one update, identity EMA, empirical source, ``lambda = 0``).

Only :class:`~fisher_ref.approx.ekfac.EKFAC` and the four of ``base`` are block *classes*: TKFAC is
a :class:`~fisher_ref.approx.kfac.Kron` with its scalar folded into one factor, and a normalisation
layer's readings are ``Dense``/``BlockDiag``/``Diag``. Fewer representations, fewer places for the
``rvec`` convention to be got wrong.
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
from .ekfac import EKFAC, ekfac_eigenbases
from .factors import (
    KFAC_MODES,
    LayerFactors,
    accumulate_ekfac,
    accumulate_factors,
    augmented_input,
    output_grad,
)
from .kfac import Kron, af_raw_from_factors, kfac_from_factors
from .norm_layers import NormStats, cross_term_share, exact_separate
from .tkfac import TkfacStats

__all__ = ["EKFAC", "KFAC_MODES", "BlockDiag", "BlockOps", "CurvatureBlock", "Dense", "Diag",
           "Kron", "LayerFactors", "NormStats", "TkfacStats", "accumulate_ekfac",
           "accumulate_factors", "af_raw_from_factors", "augmented_input", "block_diagonal_of",
           "cross_term_share", "ekfac_eigenbases", "exact_separate", "kfac_from_factors",
           "optimal_scale", "output_grad", "rearrange"]
