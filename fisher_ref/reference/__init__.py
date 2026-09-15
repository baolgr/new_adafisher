"""Exact curvature references for the Fisher-drift campaign (``plan_exp_draft.md`` §2, §7).

One module per computational regime. Lot 1 ships ``dense`` (regime A, a materialised ``P x P``
matrix in fp64); ``factor`` (regime B, per-layer Grams) is lot 4 and ``matfree`` (regime C) lot 6.
"""

from __future__ import annotations

from .dense import DenseReference, ParamLayout, build_dense_reference, symmetrize_, to_augmented

__all__ = ["DenseReference", "ParamLayout", "build_dense_reference", "symmetrize_",
           "to_augmented"]
