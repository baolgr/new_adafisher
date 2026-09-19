"""Exact curvature references, one module per computational regime.

``dense`` is the small-model regime: a materialised ``P x P`` matrix in float64, built as
``F = U^T U`` from per-sample parameter gradients. Regimes that never form ``P x P`` -- per-layer
Gram matrices, and a matrix-free operator -- are not implemented here yet.

Re-exports the dense builder and its supporting types; see :mod:`fisher_ref.reference.dense`.
"""

from __future__ import annotations

from .dense import (
    DenseAccumulator,
    DenseReference,
    ParamLayout,
    RowCheckError,
    build_dense_reference,
    check_rows_against_autograd,
    reference_parameter_names,
    symmetrize_,
    to_augmented,
)

__all__ = ["DenseAccumulator", "DenseReference", "ParamLayout", "RowCheckError", "build_dense_reference",
           "check_rows_against_autograd", "reference_parameter_names", "symmetrize_",
           "to_augmented"]
