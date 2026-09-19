"""Raw positional parameters: the structures of a ``pos_embed`` block.

``pos_embed`` has shape ``(1, T, D)`` and is *added* to the token stream, so its per-sample gradient
at position ``t`` and width ``d`` is simply the gradient at that token, and its block is
``(T*D) x (T*D)`` in row-major order ``t*D + d``. Read as a linear layer, its "input" is the
constant one-hot ``e_t``, which is worth stating rather than special-casing silently: every
position has its own parameters, so there is **no weight sharing** here, and what plays the role of
the cross-position-free block is the block with the cross-position terms dropped.

Three structures, all read off the exact block, so no extra accumulation is needed:

* ``position_blockdiag`` -- the exact ``D x D`` block of each position, cross-position terms
  dropped;
* ``kfac_onehot`` -- K-FAC-expand with the one-hot input: ``A = I_T / T`` and
  ``G = (1/N) sum_{n,c,t} g g^T = sum_t B_tt``, so every position gets the *average* of the
  per-position blocks. In this package's row-major convention the **outer** Kronecker factor is the
  position axis and the inner one the embedding width, which is the ordering that matches the
  ``t*D + d`` layout;
* the controls (the exact diagonal, and the identity -- ``pos_embed`` belongs to no hooked module,
  so the optimizer's own treatment of it *is* the identity).

Public API
----------

:func:`position_blocks`  the ``(T, D, D)`` stack of per-position blocks.

:func:`position_blockdiag`, :func:`kfac_onehot`  the two structures.

:func:`position_coupling_share`  ``(||B - blockdiag||_F / ||B||_F, the same squared)``: how much of
the block lives in the cross-position terms.

Dependencies: :mod:`fisher_ref.approx.base`, :mod:`fisher_ref.approx.kfac`.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from .base import BlockDiag, Dense
from .kfac import Kron


def _check(block: Tensor, positions: int, width: int) -> None:
    if block.shape != (positions * width, positions * width):
        raise ValueError(f"expected a ({positions}*{width})^2 block; got {tuple(block.shape)}")


def position_blocks(block: Tensor, positions: int, width: int) -> Tensor:
    """``(T, D, D)``: the exact block of every position."""
    _check(block, positions, width)
    per = block.reshape(positions, width, positions, width)
    index = torch.arange(positions, device=block.device)
    return per[index, :, index, :]


def position_blockdiag(block: Tensor, positions: int, width: int) -> BlockDiag:
    blocks = position_blocks(block, positions, width)
    return BlockDiag([(str(t), slice(t * width, (t + 1) * width), Dense(blocks[t].clone()))
                      for t in range(positions)], P=positions * width)


def kfac_onehot(block: Tensor, positions: int, width: int) -> Kron:
    """``kron(I_T / T, sum_t B_tt)`` — in :class:`Kron`'s convention the *outer* factor is the
    position axis (``G``) and the inner one the embedding width (``A``)."""
    blocks = position_blocks(block, positions, width)
    eye = torch.eye(positions, dtype=block.dtype, device=block.device) / positions
    return Kron(A=blocks.sum(dim=0), G=eye)


def position_coupling_share(block: Tensor, positions: int, width: int) -> Tuple[float, float]:
    """``(||B - blockdiag||_F / ||B||_F, same squared)``: the cross-position share of the block."""
    blocks = position_blocks(block, positions, width)
    total2 = float((block * block).sum())
    kept2 = float((blocks * blocks).sum())
    if total2 <= 0:
        return float("nan"), float("nan")
    share2 = max(total2 - kept2, 0.0) / total2
    return share2 ** 0.5, share2


__all__ = ["kfac_onehot", "position_blockdiag", "position_blocks", "position_coupling_share"]
