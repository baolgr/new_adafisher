"""How much curvature lives between two layers rather than inside one.

For a dense reference ``R`` and a list of per-layer column ranges, :func:`coupling_matrix` returns

    C[l, l'] = ||R[l, l']||_F / sqrt(||R[l, l]||_F * ||R[l', l']||_F)

with ones on the diagonal by construction, and an entry near zero meaning the two layers'
curvatures are orthogonal in the Frobenius sense. That is what "the exact block diagonal loses
nothing" would mean.

**What this is, precisely.** Writing ``R = U^T U / N``, layer ``l``'s tangent kernel is
``K_l = U_l U_l^T / N``, and the uncentred kernel alignment between two layers is
``<K_l, K_l'>_F / (||K_l||_F ||K_l'||_F) = ||R[l,l']||_F^2 / (||R[l,l]||_F ||R[l',l']||_F)``. The
entry returned here is the **square root** of that alignment. The two agree at 0 and at 1 and
differ everywhere in between, so a reported value of 0.5 is an alignment of 0.25. Read the numbers
this function returns as the square root, or square them first.

Public API
----------

:func:`coupling_matrix`  ``(names, C)``.

:func:`offdiagonal_mass`  what fraction of ``||R||_F^2`` the exact block diagonal keeps, and what
fraction it throws away -- the same question as one number.

:func:`as_rows`  the off-diagonal entries in the long result-file format.

This module imports nothing from the rest of ``fisher_ref``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import Tensor


def coupling_matrix(R: Tensor, ranges: Sequence[Tuple[str, slice]]) -> Tuple[List[str], Tensor]:
    """``(names, C)`` with ``C[l, l'] = ||R[l, l']||_F / (||R[l,l]||_F ||R[l',l']||_F)^(1/2)``.

    The diagonal is 1 by construction; an entry near 0 means the two layers' curvatures are
    orthogonal in the Frobenius sense, which is what "the block-diagonal loses nothing" means.
    """
    names = [name for name, _ in ranges]
    size = len(names)
    norms = torch.tensor([float(torch.linalg.matrix_norm(R[columns, columns]))
                          for _, columns in ranges], dtype=R.dtype)
    out = torch.zeros(size, size, dtype=R.dtype)
    for i, (_, rows) in enumerate(ranges):
        for j, (_, columns) in enumerate(ranges):
            scale = float((norms[i] * norms[j]) ** 0.5) ** 2
            value = float(torch.linalg.matrix_norm(R[rows, columns]))
            out[i, j] = value / scale ** 0.5 if scale > 0 else float("nan")
    return names, out


def offdiagonal_mass(R: Tensor, ranges: Sequence[Tuple[str, slice]]) -> Dict[str, float]:
    """How much of ``||R||_F^2`` the exact block-diagonal throws away — HF5 as one number."""
    total = float((R * R).sum())
    kept = sum(float((R[columns, columns] * R[columns, columns]).sum()) for _, columns in ranges)
    return {"block_diagonal_share": kept / total if total else float("nan"),
            "offdiagonal_share": 1.0 - kept / total if total else float("nan")}


def as_rows(names: Sequence[str], matrix: Tensor, **keys: Any) -> list:
    return [{**keys, "layer": names[i], "metric": f"coupling::{names[j]}",
             "value": float(matrix[i, j])}
            for i in range(len(names)) for j in range(len(names)) if i != j]


__all__ = ["as_rows", "coupling_matrix", "offdiagonal_mass"]
