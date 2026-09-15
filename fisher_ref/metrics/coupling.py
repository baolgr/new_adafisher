"""M8 — inter-block coupling (``plan_exp_draft.md`` §5).

``c_{ll'} = <K_l, K_{l'}>_F / (||K_l||_F ||K_{l'}||_F)``, an uncentred CKA between the layers'
tangent kernels. In regime A the blocks are submatrices of the dense reference, so the off-diagonal
block ``R[l, l']`` *is* the coupling and no Gram is needed.

This is HF5's measurement: if ``c`` is negligible everywhere, the exact block-diagonal loses
nothing and the inter-layer directions can be dropped.
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
