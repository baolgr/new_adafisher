"""Normalisation layers: the four readings of a ``(gamma, beta)`` block, which is what HF4 compares
(``plan_exp_draft.md`` §4, "Normalisations").

The exact per-sample gradients are ``d/d gamma = sum_t g_t * x_hat_t`` and ``d/d beta = sum_t g_t``,
so the exact block is ``2C x 2C`` and is **always** affordable — 64 x 64 for A1's two
``LayerNorm(32)``. There is nothing to approximate for memory here; the question is whether the
approximations the optimizer actually makes are *right*.

Four structures, and none of them needs a new block class:

1. **exact joint** — the ``2C x 2C`` block itself (a ``Dense``);
2. **exact separate** — the same with the ``gamma``-``beta`` cross terms dropped, which is what
   AdaFisher does. The gap between 1 and 2 **is** HF4;
3. **Hadamard**, Proposition 3.1 as written: ``gamma`` gets ``H (.) S`` with
   ``H = (1/|T|) sum_t x_hat x_hat^T`` and ``S = (1/|T|) sum_t g g^T``, and ``beta`` gets ``S``;
4. **Prop. 3.1 as implemented** — the diagonal ``[H_D]_c [S_D]_c`` this repository's ``diag.py``
   actually computes, a documented "sum-then-square" deviation from the Proposition's
   "square-then-sum" (``plan_lot5.md`` §0.6).

A caution this repository has already paid for twice: a normalisation block is **Hadamard-, not
Kronecker-structured** (``plan_lot5.md`` §0.1), so none of these is a ``Kron`` and
``ParamLayout.augmented_permutation`` refuses such a layer outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from .base import BlockDiag, Dense, Diag


@dataclass(frozen=True)
class NormStats:
    """Second moments of the normalised input and of the output gradient, per position."""

    H: Tensor        # (C, C), mean_t x_hat x_hat^T
    S: Tensor        # (C, C), mean_{n,c,t} g g^T
    C: int

    def hadamard(self) -> BlockDiag:
        """Proposition 3.1 as written: ``gamma -> H (.) S``, ``beta -> S``, cross terms zero."""
        return BlockDiag([("gamma", slice(0, self.C), Dense(self.H * self.S)),
                          ("beta", slice(self.C, 2 * self.C), Dense(self.S))], P=2 * self.C)

    def as_implemented_diagonal(self) -> Diag:
        """The diagonal ``diag.py`` computes: ``[H_D]_c [S_D]_c`` for ``gamma``, ``[S_D]_c`` for
        ``beta``. Compared here against the exact block, never against :meth:`hadamard` — the two
        do not estimate the same quantity even up to a reduction order (``plan_lot5.md`` §0.6).
        """
        return Diag(torch.cat([self.H.diagonal() * self.S.diagonal(), self.S.diagonal()]))


def exact_separate(block: Tensor, C: int) -> BlockDiag:
    """The exact ``2C x 2C`` block with the ``gamma``-``beta`` cross terms dropped.

    Its gap to the joint block is HF4, measured rather than argued: it is the only structure in the
    zoo whose error comes from *ignoring a coupling the exact object has*, not from a factorisation.
    """
    return BlockDiag([("gamma", slice(0, C), Dense(block[:C, :C])),
                      ("beta", slice(C, 2 * C), Dense(block[C:, C:]))], P=2 * C)


def cross_term_share(block: Tensor, C: int) -> Tuple[float, float]:
    """``(||cross||_F / ||block||_F, ||cross||_F / ||diagonal blocks||_F)`` — HF4 as one number.

    The first is the fraction of the block's Frobenius mass that "exact separate" throws away.
    """
    cross = block[:C, C:]
    total = float(torch.linalg.matrix_norm(block))
    diagonal = float((torch.linalg.matrix_norm(block[:C, :C]) ** 2
                      + torch.linalg.matrix_norm(block[C:, C:]) ** 2) ** 0.5)
    off = float(torch.linalg.matrix_norm(cross)) * (2.0 ** 0.5)   # both triangles
    return off / total, off / diagonal


__all__ = ["NormStats", "cross_term_share", "exact_separate"]
