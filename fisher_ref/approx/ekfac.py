"""EKFAC: K-FAC's eigenbasis with the **optimal** diagonal in it (``plan_exp_draft.md`` §4).

``K = (Q_G (x) Q_A) diag(s) (Q_G (x) Q_A)^T`` with
``s_ij = (1/N) sum_{n,c} [(Q_G^T G_{n,c} Q_A)_ij]^2`` — the exact second moment of the per-sample
gradients *projected into* K-FAC's eigenbasis, which is why it is the best diagonal approximation
in that basis and why it dominates K-FAC in Frobenius norm.

Two consequences the campaign uses:

* **It preserves the trace by construction** (T9). ``sum_ij s_ij`` is the mean of
  ``||Q_G^T G Q_A||_F^2``, and an orthogonal change of basis leaves the Frobenius norm alone, so it
  equals the mean of ``||G||_F^2 = tr(B_l)``. No estimator quality enters — if T9 fails, the
  eigenbases are not orthonormal.
* **It needs a second pass over the probes.** ``Q_A`` and ``Q_G`` are only known once ``A`` and
  ``G`` have been accumulated, so ``s`` cannot be built in the same sweep. That is why
  ``capture.iter_probe_columns`` is a generator (``plan_exp_lot2.md`` §0.2).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .base import BlockOps, rearrange


@dataclass
class EKFAC(BlockOps):
    """``(Q_G (x) Q_A) diag(s) (Q_G (x) Q_A)^T``, never materialised."""

    QA: Tensor          # (d_in, d_in), eigenvectors of A
    QG: Tensor          # (d_out, d_out), eigenvectors of G
    s: Tensor           # (d_out, d_in), the optimal diagonal in that basis

    def __post_init__(self) -> None:
        self.d_in = int(self.QA.shape[0])
        self.d_out = int(self.QG.shape[0])
        self.P = self.d_in * self.d_out
        if self.s.shape != (self.d_out, self.d_in):
            raise ValueError(f"s must be ({self.d_out}, {self.d_in}); got {tuple(self.s.shape)}")

    def _apply(self, X: Tensor, scale: Tensor) -> Tensor:
        batch = X.shape[1]
        M = X.T.reshape(batch, self.d_out, self.d_in)
        # Q_G^T M Q_A, rescale, then Q_G (...) Q_A^T. The second contraction takes ``Q_G[o, i]``,
        # not ``Q_G[i, o]``: the eigenvectors are the *columns*, so writing the transpose here
        # silently applies the operator in a mirrored basis (how the conformance test first failed).
        projected = torch.einsum("oi,boj,jk->bik", self.QG, M, self.QA)
        scaled = projected * scale
        out = torch.einsum("oi,bij,kj->bok", self.QG, scaled, self.QA)
        return out.reshape(batch, self.P).T

    def matmat(self, X: Tensor) -> Tensor:
        return self._apply(X, self.s)

    def solve(self, v: Tensor, lam: float) -> Tensor:
        return self._apply(v.reshape(-1, 1), 1.0 / (self.s + lam)).reshape(-1)

    def trace(self) -> Tensor:
        return self.s.sum()

    def fro2(self) -> Tensor:
        return (self.s * self.s).sum()

    def logdet(self, lam: float) -> Tensor:
        return torch.log(self.s + lam).sum()

    def diag(self) -> Tensor:
        """``K_kk = sum_ij s_ij Q_G[k,i]^2 Q_A[l,j]^2`` for ``k = (k, l)`` in ``rvec`` order."""
        return ((self.QG * self.QG) @ self.s @ (self.QA * self.QA).T).reshape(-1)

    def to_dense(self, block: int = 1024) -> Tensor:
        basis = torch.kron(self.QG, self.QA)
        return (basis * self.s.reshape(-1)) @ basis.T

    def inner_dense(self, R: Tensor, block: int = 1024) -> Tensor:
        """``<R, K> = sum_ij s_ij (q_Gi (x) q_Aj)^T R (q_Gi (x) q_Aj)``, through the rearrangement.

        Contracting in two steps keeps the intermediate at ``d_out * d_in^2`` — 158 MB for A1's
        widest layer — instead of the ``d_in * d_in^2`` (3.9 GB) a one-shot bilinear form would need,
        and never forms the ``P x P`` eigenbasis.
        """
        tensor = rearrange(R, self.d_out, self.d_in).reshape(self.d_out, self.d_out,
                                                             self.d_in, self.d_in)
        partial = torch.einsum("ki,mi,kmlp->ilp", self.QG, self.QG, tensor)
        projected = torch.einsum("ilp,lj,pj->ij", partial, self.QA, self.QA)
        return (self.s * projected).sum()

    def _dtype(self) -> torch.dtype:
        return self.QA.dtype

    def _device(self) -> object:
        return self.QA.device


def ekfac_eigenbases(A: Tensor, G: Tensor) -> "tuple":
    """``(Q_A, Q_G)`` — K-FAC's eigenbasis, which EKFAC rescales inside.

    ``torch.linalg.eigh`` fixes neither the sign nor the ordering of eigenvectors, so never compare
    two bases: compare the applied operators (``CLAUDE.md``, "Known testing pitfalls").
    """
    _, q_a = torch.linalg.eigh(A)
    _, q_g = torch.linalg.eigh(G)
    return q_a, q_g


__all__ = ["EKFAC", "ekfac_eigenbases"]
