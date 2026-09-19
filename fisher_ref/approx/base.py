"""The common interface every structured approximation exposes, and four generic containers.

Every metric is written once against this protocol, so it applies to every structure. The protocol
is the usual one -- apply, solve, trace, squared Frobenius norm, log-determinant, diagonal, dense
form -- with one addition: ``inner_dense(R)``, the inner product against a dense reference. Without
it, comparing a Kronecker structure to a reference would have to materialise ``G (x) A``, which is
gigabytes for a single wide layer, just to take a scalar product. Every implementation inherits a
generic, always-correct ``inner_dense`` from :class:`BlockOps`, computed by sweeping the identity
through in column blocks; the structures that can do better override it with a closed form, and the
conformance tests check that the two agree.

The rearrangement
-----------------

The one primitive worth naming is ``R(B)`` of :func:`rearrange`. For a block ``B`` of size
``(d_out*d_in)^2`` in row-major (``rvec``) order it satisfies

    <B, G (x) A>       = vec(G)^T R(B) vec(A)
    ||B - G (x) A||_F  = ||R(B) - vec(G) vec(A)^T||_F

so the best rank-one Kronecker fit of ``B`` is the top singular pair of ``R(B)`` (Van Loan and
Pitsianis; used on auto-encoder curvature by Koroko et al., arXiv:2201.10285). That makes it the
shared engine of two things at once: the Kronecker-bias metric, whose ``sigma_2 / sigma_1`` *is*
the departure from any Kronecker product, and the Frobenius comparison against any Kronecker
structure.

Public API
----------

:func:`rearrange`  ``R(B)``, shape ``(d_out^2, d_in^2)``.

:class:`CurvatureBlock`  the protocol, for type checking.

:class:`BlockOps`  everything derivable from ``matmat``; implementations override what they can do
in closed form. ``solve`` and ``logdet`` have no generic fallback on purpose -- a column sweep
would cost ``P`` solves, and every structure here has a closed form.

:class:`Dense`  the reference itself as a block, so a metric can take ``R`` as ``K`` (the
self-consistency check).

:class:`Diag`  a diagonal structure: the exact diagonal control, AdaFisher's raw estimator, Adam's
second moment.

:class:`BlockDiag`  a block diagonal over contiguous parameter ranges. Its ``logdet`` counts the
coordinates no block covers explicitly, since each would contribute ``log(lam)``.

:func:`block_diagonal_of`  the exact block diagonal of a dense reference.

:func:`optimal_scale`  ``c* = <R, K> / ||K||_F^2``, the scalar that minimises ``||R - cK||_F``.
Required for any structure that makes no claim on scale, and for any comparison between references
built on different probe sets, whose norms differ.

This module imports nothing from the rest of ``fisher_ref``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol, Tuple, runtime_checkable

import torch
from torch import Tensor


def rearrange(block: Tensor, d_out: int, d_in: int) -> Tensor:
    """``R(B)``, shape ``(d_out^2, d_in^2)``, for a block in ``rvec`` order (row index
    ``i * d_in + j`` with ``i`` the output and ``j`` the input coordinate).
    """
    if block.shape != (d_out * d_in, d_out * d_in):
        raise ValueError(f"expected a ({d_out * d_in})^2 block; got {tuple(block.shape)}")
    return (block.reshape(d_out, d_in, d_out, d_in)
                 .permute(0, 2, 1, 3)
                 .reshape(d_out * d_out, d_in * d_in))


@runtime_checkable
class CurvatureBlock(Protocol):
    """``plan_exp_draft.md`` §7.1."""

    P: int

    def matmat(self, X: Tensor) -> Tensor: ...
    def matvec(self, v: Tensor) -> Tensor: ...
    def solve(self, v: Tensor, lam: float) -> Tensor: ...
    def trace(self) -> Tensor: ...
    def fro2(self) -> Tensor: ...
    def logdet(self, lam: float) -> Tensor: ...
    def diag(self) -> Tensor: ...
    def to_dense(self) -> Tensor: ...
    def inner_dense(self, R: Tensor) -> Tensor: ...


class BlockOps:
    """Everything derivable from ``matmat``. Implementations provide ``matmat`` and, where they can
    do better than a column sweep, override the rest.
    """

    P: int

    def matmat(self, X: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def solve(self, v: Tensor, lam: float) -> Tensor:  # pragma: no cover - abstract
        """``(K + lam I)^{-1} v``. No generic fallback: a column sweep would cost ``P`` solves, and
        every structure here has a closed form (a Kronecker eigenbasis, a diagonal, a Cholesky).
        """
        raise NotImplementedError

    def logdet(self, lam: float) -> Tensor:  # pragma: no cover - abstract
        """``log det(K + lam I)``, M3's second term."""
        raise NotImplementedError

    def matvec(self, v: Tensor) -> Tensor:
        return self.matmat(v.reshape(-1, 1)).reshape(-1)

    def apply_rows(self, U: Tensor) -> Tensor:
        """Rows ``u_i -> K u_i``, i.e. ``U K`` for a symmetric ``K`` (``plan_exp_draft.md`` §5.1)."""
        return self.matmat(U.T).T

    def to_dense(self, block: int = 1024) -> Tensor:
        columns = []
        for start in range(0, self.P, block):
            stop = min(start + block, self.P)
            basis = torch.zeros(self.P, stop - start, dtype=self._dtype(), device=self._device())
            basis[torch.arange(start, stop), torch.arange(stop - start)] = 1.0
            columns.append(self.matmat(basis))
        return torch.cat(columns, dim=1)

    def inner_dense(self, R: Tensor, block: int = 1024) -> Tensor:
        """``<R, K>`` by sweeping the identity in column blocks — never materialises ``K``."""
        total = torch.zeros((), dtype=R.dtype, device=R.device)
        for start in range(0, self.P, block):
            stop = min(start + block, self.P)
            basis = torch.zeros(self.P, stop - start, dtype=self._dtype(), device=self._device())
            basis[torch.arange(start, stop), torch.arange(stop - start)] = 1.0
            total = total + (R[:, start:stop] * self.matmat(basis)).sum()
        return total

    def diag(self) -> Tensor:
        return self.to_dense().diagonal().clone()

    def trace(self) -> Tensor:
        return self.diag().sum()

    def fro2(self) -> Tensor:
        dense = self.to_dense()
        return (dense * dense).sum()

    def _dtype(self) -> torch.dtype:  # pragma: no cover - trivial
        raise NotImplementedError

    def _device(self) -> object:  # pragma: no cover - trivial
        raise NotImplementedError


@dataclass
class Dense(BlockOps):
    """The reference itself, as a ``CurvatureBlock`` — so a metric can take ``R`` as ``K``
    (``D_lambda(R||R) = 0`` is T7's self-consistency check).
    """

    matrix: Tensor

    def __post_init__(self) -> None:
        self.P = int(self.matrix.shape[0])

    def matmat(self, X: Tensor) -> Tensor:
        return self.matrix @ X

    def to_dense(self, block: int = 1024) -> Tensor:
        return self.matrix

    def diag(self) -> Tensor:
        return self.matrix.diagonal().clone()

    def trace(self) -> Tensor:
        return self.matrix.diagonal().sum()

    def fro2(self) -> Tensor:
        return (self.matrix * self.matrix).sum()

    def inner_dense(self, R: Tensor, block: int = 1024) -> Tensor:
        return (R * self.matrix).sum()

    def solve(self, v: Tensor, lam: float) -> Tensor:
        damped = self.matrix + lam * torch.eye(self.P, dtype=self.matrix.dtype,
                                               device=self.matrix.device)
        return torch.linalg.solve(damped, v)

    def logdet(self, lam: float) -> Tensor:
        eigenvalues = torch.linalg.eigvalsh(self.matrix)
        return torch.log(eigenvalues + lam).sum()

    def _dtype(self) -> torch.dtype:
        return self.matrix.dtype

    def _device(self) -> object:
        return self.matrix.device


@dataclass
class Diag(BlockOps):
    """A diagonal structure: the exact diagonal control, AdaFisher's raw estimator, Adam's ``v``."""

    values: Tensor

    def __post_init__(self) -> None:
        self.P = int(self.values.numel())

    def matmat(self, X: Tensor) -> Tensor:
        return self.values.reshape(-1, 1) * X

    def matvec(self, v: Tensor) -> Tensor:
        return self.values * v

    def solve(self, v: Tensor, lam: float) -> Tensor:
        return v / (self.values + lam)

    def trace(self) -> Tensor:
        return self.values.sum()

    def fro2(self) -> Tensor:
        return (self.values * self.values).sum()

    def logdet(self, lam: float) -> Tensor:
        return torch.log(self.values + lam).sum()

    def diag(self) -> Tensor:
        return self.values

    def to_dense(self, block: int = 1024) -> Tensor:
        return torch.diag(self.values)

    def inner_dense(self, R: Tensor, block: int = 1024) -> Tensor:
        return (R.diagonal() * self.values).sum()

    def _dtype(self) -> torch.dtype:
        return self.values.dtype

    def _device(self) -> object:
        return self.values.device


@dataclass
class BlockDiag(BlockOps):
    """``blkdiag_l(K_l)`` over contiguous parameter ranges — the exact block-diagonal control of
    ``plan_exp_draft.md`` §4, and the container every per-layer structure is assembled into.
    """

    blocks: List[Tuple[str, slice, BlockOps]]
    P: int = 0

    def __post_init__(self) -> None:
        if not self.P:
            self.P = max(s.stop for _, s, _ in self.blocks) if self.blocks else 0

    def matmat(self, X: Tensor) -> Tensor:
        out = torch.zeros(self.P, X.shape[1], dtype=X.dtype, device=X.device)
        for _, columns, block in self.blocks:
            out[columns] = block.matmat(X[columns])
        return out

    def solve(self, v: Tensor, lam: float) -> Tensor:
        out = torch.zeros_like(v)
        for _, columns, block in self.blocks:
            out[columns] = block.solve(v[columns], lam)
        return out

    def trace(self) -> Tensor:
        return sum((block.trace() for _, _, block in self.blocks), start=torch.zeros(()))

    def fro2(self) -> Tensor:
        return sum((block.fro2() for _, _, block in self.blocks), start=torch.zeros(()))

    def logdet(self, lam: float) -> Tensor:
        """Only correct when the blocks partition all ``P`` coordinates; the uncovered ones would
        contribute ``log(lam)`` each and are counted explicitly.
        """
        covered = sum(s.stop - s.start for _, s, _ in self.blocks)
        total = sum((block.logdet(lam) for _, _, block in self.blocks), start=torch.zeros(()))
        import math  # noqa: PLC0415
        return total + (self.P - covered) * math.log(lam)

    def diag(self) -> Tensor:
        first = self.blocks[0][2]
        out = torch.zeros(self.P, dtype=first._dtype(), device=first._device())
        for _, columns, block in self.blocks:
            out[columns] = block.diag()
        return out

    def inner_dense(self, R: Tensor, block: int = 1024) -> Tensor:
        total = torch.zeros((), dtype=R.dtype, device=R.device)
        for _, columns, sub in self.blocks:
            total = total + sub.inner_dense(R[columns, columns])
        return total

    def to_dense(self, block: int = 1024) -> Tensor:
        first = self.blocks[0][2]
        out = torch.zeros(self.P, self.P, dtype=first._dtype(), device=first._device())
        for _, columns, sub in self.blocks:
            out[columns, columns] = sub.to_dense()
        return out

    def named(self) -> Dict[str, BlockOps]:
        return {name: block for name, _, block in self.blocks}

    def _dtype(self) -> torch.dtype:
        return self.blocks[0][2]._dtype()

    def _device(self) -> object:
        return self.blocks[0][2]._device()


def block_diagonal_of(matrix: Tensor, ranges: Iterable[Tuple[str, slice]]) -> BlockDiag:
    """The exact block-diagonal of a dense reference — ``blkdiag_l(B_l)``, HF5's control."""
    return BlockDiag([(name, columns, Dense(matrix[columns, columns]))
                      for name, columns in ranges], P=int(matrix.shape[0]))


def optimal_scale(R: Tensor, K: BlockOps) -> Tensor:
    """``c* = <R, K> / ||K||_F^2`` (``plan_exp_draft.md`` §3.3).

    Required for any structure that makes no claim on scale — AdaFisher's ``diag``, Adam's free
    diagonal — and, since ``plan_exp_lot2.md`` §5.1, for *any* comparison between references built
    on different probe sets, whose norms differ by 30-60 %.
    """
    return K.inner_dense(R) / K.fro2()


__all__ = ["BlockDiag", "BlockOps", "CurvatureBlock", "Dense", "Diag", "block_diagonal_of",
           "optimal_scale", "rearrange"]
