"""K-FAC: the Kronecker structure ``K = G (x) A``, and the two ways of building its factors from a
weight-shared layer (``plan_exp_draft.md`` §4).

Convention, and it is the one this repository already fixed in ``conventions.kron_rvec``: with a
direction ``M`` shaped ``(d_out, d_in)`` and ``rvec`` flattening, ``K rvec(M) = rvec(G M A^T)``,
i.e. the papers' ``A (x) B`` is ``kron(G, A)`` here — **output factor outer, input factor inner**.
``A`` is the input second moment (bias-augmented), ``G`` the output-gradient one.

The two normalisations differ only in what they average over, and the difference *is* HF2
(Eschenhagen et al., arXiv:2311.00636):

* **expand**: ``A = (1/NT) sum_{n,t} a a^T``, ``G = (1/N) sum_{n,c,t} g g^T`` — exact in the expand
  setting for a deep linear network (Prop. 1);
* **reduce**: the factors are built on the *summed* statistics ``sum_t a_{n,t}`` and
  ``sum_t g_{n,c,t}`` — exact in the reduce setting, e.g. mean pooling (Prop. 2).

A1 has no weight sharing (``T = 1``), so the two coincide there and HF2 is lot 3's question, on
A2/A3. Both are implemented now because they are one branch apart and the accumulator has to choose
at capture time, not afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .base import BlockOps, rearrange


@dataclass
class Kron(BlockOps):
    """``K = G (x) A`` in ``rvec`` order, never materialised."""

    A: Tensor            # (d_in, d_in), input factor
    G: Tensor            # (d_out, d_out), output factor
    _eig: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None

    def __post_init__(self) -> None:
        self.d_in = int(self.A.shape[0])
        self.d_out = int(self.G.shape[0])
        self.P = self.d_in * self.d_out

    # -- the cheap closed forms ------------------------------------------------------------------

    def matmat(self, X: Tensor) -> Tensor:
        # X: (P, b). Each column is rvec(M); K rvec(M) = rvec(G M A^T).
        batch = X.shape[1]
        M = X.T.reshape(batch, self.d_out, self.d_in)
        return torch.einsum("oi,bij,jk->bok", self.G, M, self.A.T).reshape(batch, self.P).T

    def trace(self) -> Tensor:
        return self.G.diagonal().sum() * self.A.diagonal().sum()

    def fro2(self) -> Tensor:
        return (self.G * self.G).sum() * (self.A * self.A).sum()

    def diag(self) -> Tensor:
        return torch.kron(self.G.diagonal(), self.A.diagonal())

    def to_dense(self, block: int = 1024) -> Tensor:
        return torch.kron(self.G, self.A)

    def inner_dense(self, R: Tensor, block: int = 1024) -> Tensor:
        """``<R, G (x) A> = vec(G)^T R(R) vec(A)`` — the rearrangement identity, so the Kronecker
        product is never formed (``plan_exp_lot2.md`` §0.4).
        """
        rearranged = rearrange(R, self.d_out, self.d_in)
        return self.G.reshape(-1) @ rearranged @ self.A.reshape(-1)

    # -- everything needing an inverse goes through the Kronecker eigenbasis ----------------------

    def _eigen(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(lam_A, Q_A, lam_G, Q_G)``, cached.

        ``(G (x) A + lam I)^{-1} = (Q_G (x) Q_A) diag(1/(lam_G lam_A + lam)) (Q_G (x) Q_A)^T`` —
        the only exact way to damp a Kronecker product, since ``G (x) A + lam I`` is not Kronecker.
        """
        if self._eig is None:
            lam_a, q_a = torch.linalg.eigh(self.A)
            lam_g, q_g = torch.linalg.eigh(self.G)
            self._eig = (lam_a, q_a, lam_g, q_g)
        return self._eig

    def eigenvalues(self) -> Tensor:
        """``(d_out, d_in)`` grid ``lam_G_i * lam_A_j`` — K-FAC's spectrum, and EKFAC's basis."""
        lam_a, _, lam_g, _ = self._eigen()
        return torch.outer(lam_g, lam_a)

    def solve(self, v: Tensor, lam: float) -> Tensor:
        lam_a, q_a, lam_g, q_g = self._eigen()
        M = v.reshape(self.d_out, self.d_in)
        projected = q_g.T @ M @ q_a
        scaled = projected / (torch.outer(lam_g, lam_a) + lam)
        return (q_g @ scaled @ q_a.T).reshape(-1)

    def logdet(self, lam: float) -> Tensor:
        return torch.log(self.eigenvalues() + lam).sum()

    def _dtype(self) -> torch.dtype:
        return self.A.dtype

    def _device(self) -> object:
        return self.A.device


def kfac_from_factors(A: Tensor, G: Tensor) -> Kron:
    return Kron(A=A, G=G)


def af_raw_from_factors(A: Tensor, G: Tensor, positions: int = 1):
    """AdaFisher's raw estimator: ``diag(A) (x) diag(G)``.

    ``diag(A) (x) diag(G) = diag(A (x) G)``, so it **is** K-FAC's diagonal — which is the whole of
    HF3 (``plan_exp_draft.md`` §3.1): the two paths from the exact block ``B_l`` to this object
    differ, entry-wise and without sharing, by ``Cov(a_j^2, g_i^2)``. The ``/|T|`` of App. A.3
    applies to convolutions.
    """
    from .base import Diag  # noqa: PLC0415
    values = torch.kron(G.diagonal(), A.diagonal())
    return Diag(values / positions if positions != 1 else values)


__all__ = ["Kron", "af_raw_from_factors", "kfac_from_factors"]
