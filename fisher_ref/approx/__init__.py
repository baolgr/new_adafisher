"""The approximation zoo: structured stand-ins for an exact curvature block.

This package owns its own implementations rather than calling ``adafisher_modes``, because the two
answer different questions. The optimizer computes running-averaged, damped, hook-driven factors at
training precision and has no notion of a curvature *source*; the structural protocol needs the
structure alone, in float64, from type-2 backprop vectors. The two are tied together by a
conformance test that pins each structure here against the optimizer's own at the degenerate
setting (one update, identity running average, empirical source, zero damping) -- and by
:mod:`fisher_ref.approx.adafisher_state`, which reads the optimizer's live state as a block.

Only :class:`~fisher_ref.approx.ekfac.EKFAC` and the four containers of
:mod:`fisher_ref.approx.base` are block *classes*. TKFAC is a
:class:`~fisher_ref.approx.kfac.Kron` with its scalar folded into one factor, and a normalisation
layer's readings are ``Dense`` / ``BlockDiag`` / ``Diag``. Fewer representations means fewer places
for the row-major Kronecker convention to be got wrong.

Modules
-------

``base``  the ``BlockOps`` protocol and the four generic containers, plus the rearrangement
primitive ``R(B)``.
``kfac``  ``K = G (x) A`` and AdaFisher's raw diagonal estimator.
``ekfac``  the optimal diagonal in K-FAC's eigenbasis.
``tkfac``  the trace-preserving factorisation.
``factors``  accumulating everything the zoo needs from one traversal of the probes.
``norm_layers``  the four readings of a ``(gamma, beta)`` block, and the optimizer's own formulas
called directly.
``sharing``  the weight-sharing decomposition of an exact block.
``embed``  the structures of a raw positional-embedding block.
``adafisher_state``  the optimizer's live per-module state, as a block.
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
