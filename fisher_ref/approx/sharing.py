"""The weight-sharing decomposition of an exact curvature block.

On a layer whose weight is applied at ``T > 1`` positions the per-example gradient is
``G_n = sum_t g_t a_t^T``, and K-FAC-expand reaches the exact block in two steps::

    B  --(cross-position terms t != t' dropped)-->  B^exp  --(independence)-->  G (x) A

with ``B^exp = (1/N) sum_{n,c,t} (g_t g_t^T) (x) (a_t a_t^T)`` in row-major order (``a``
bias-augmented). The first arrow is the **sharing** error, the second the **independence** error.
Splitting them is what says whether K-FAC's error on a shared layer is a Kronecker-form problem or
a weight-sharing problem.

``B^exp`` is a sum of ``N*C*T`` rank-one Kronecker terms, so it is held as its **rearrangement**
``R(B^exp) = (1/N) sum vec(g g^T) vec(a a^T)^T``, never as a ``P x P`` matrix. The rearrangement is
an isometry, so ``||B^exp||_F = ||R(B^exp)||_F``, ``<B, B^exp> = <R(B), R(B^exp)>``,
``<B^exp, G (x) A> = vec(G)^T R(B^exp) vec(A)``, and the top singular pair of ``R(B^exp)`` is its
best Kronecker fit. Both ``vec(g g^T)`` and ``vec(a a^T)`` are symmetric, so the accumulation is
done in **half-vectorised** coordinates (off-diagonal entries weighted by ``sqrt(2)``, which is an
isometry on symmetric matrices), four times cheaper, and unpacked once.

**The half-vectorisation weights must be built in the working dtype.**
``torch.where(mask, 1.0, math.sqrt(2.0))`` is float32 even when everything around it is float64,
and carries a 1e-8 relative error into every norm -- eight orders of magnitude above round-off, and
far below anything a smoke run would notice.

For a normalisation layer ``B^exp`` is simply the dense ``2C x 2C`` sum of ``v v^T`` with
``v = [g_t * x_hat_t ; g_t]``, i.e. what the Hadamard form approximates by independence.

Public API
----------

:func:`half_vec_weights`, :func:`half_vec_outer`, :func:`unpack_half_vec`  the half-vectorisation.

:class:`SharingSums` / :func:`sharing_sums_for`  the additive sufficient statistic of ``B^exp`` for
one layer, with ``consume`` (chunked to bound the transient half-vectorised rows), ``rearranged``
and ``dense`` readers, and ``to`` / ``merge`` for the fold engine.

:func:`linear_decomposition`  the decomposition rows for a shared linear or convolutional layer:
``B^exp``'s norm and its cosine with the exact block, the sharing share, the independence share
against both denominators, the same two on the diagonal only, and the Kronecker tail of ``B^exp``.
The two differences of norms are computed from inner products, so they inherit the usual
square-root-of-epsilon floor when the two operands nearly coincide.

:func:`norm_decomposition`  the same two arrows on a ``2C x 2C`` normalisation block, with the
Hadamard form playing the independence step. Note that the Hadamard form is built from
expand-style statistics, so it approximates ``B^exp`` and not the exact block.

:func:`rearranged_block`  ``R(B)`` as a contiguous tensor, built once per layer and shared by every
contraction.

Dependencies: :mod:`fisher_ref.capture`, :mod:`fisher_ref.approx.base`,
:mod:`fisher_ref.approx.factors`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..capture import CapturedLayer
from .base import BlockOps, rearrange
from .factors import augmented_input


def half_vec_weights(d: int, dtype: torch.dtype, device: Any) -> Tuple[Tensor, Tensor, Tensor]:
    """``(rows, cols, weights)`` of the upper triangle, with ``weights`` **in ``dtype``**."""
    rows, cols = torch.triu_indices(d, d, device=device)
    weights = torch.full((rows.numel(),), math.sqrt(2.0), dtype=dtype, device=device)
    weights[rows == cols] = 1.0
    return rows, cols, weights


def half_vec_outer(x: Tensor) -> Tensor:
    """Rows ``x_r -> hvec(x_r x_r^T)``, ``(M, d) -> (M, d(d+1)/2)``, an isometry on the outer products."""
    rows, cols, weights = half_vec_weights(int(x.shape[1]), x.dtype, x.device)
    return x[:, rows] * x[:, cols] * weights


def unpack_half_vec(raw: Tensor, d_out: int, d_in: int) -> Tensor:
    """Half-vectorised ``(d_out(d_out+1)/2, d_in(d_in+1)/2)`` -> full ``(d_out^2, d_in^2)``."""
    def index_and_weights(d: int) -> Tuple[Tensor, Tensor]:
        rows, cols, weights = half_vec_weights(d, raw.dtype, raw.device)
        index = torch.empty(d, d, dtype=torch.long, device=raw.device)
        positions = torch.arange(rows.numel(), device=raw.device)
        index[rows, cols] = positions
        index[cols, rows] = positions
        return index.reshape(-1), weights

    index_out, weights_out = index_and_weights(d_out)
    index_in, weights_in = index_and_weights(d_in)
    scaled = raw / weights_out[:, None] / weights_in[None, :]
    return scaled[index_out][:, index_in]


@dataclass
class SharingSums:
    """The additive sufficient statistic of ``B^exp`` for one layer (raw sums, not divided)."""

    name: str
    kind: str
    d_in: int
    d_out: int
    positions: int
    n_probes: int = 0
    raw: Optional[Tensor] = None

    def consume(self, layer: CapturedLayer, *, first_column: bool,
                chunk_bytes: int = 2 ** 28) -> None:
        """Add one ``(micro-batch, column)``. ``chunk_bytes`` bounds the transient half-vectorised rows:
        ``x[:, rows] * x[:, cols] * w`` holds three ``(M, d(d+1)/2)`` tensors at once, i.e. 4 GB at
        ``M = 4096`` rows of A2's last convolution — on a 10 GB slice already holding a 4.8 GB
        reference.
        """
        if first_column:
            self.n_probes += layer.n_examples
        if self.kind == "norm":
            v = torch.cat([layer.g * layer.a, layer.g], dim=2).reshape(-1, 2 * layer.g.shape[-1])
            contribution = v.T @ v
            self.raw = contribution if self.raw is None else self.raw + contribution
            return
        a = augmented_input(layer, "expand").reshape(-1, self.d_in)
        g = layer.g.reshape(-1, self.d_out)
        if self.raw is None:
            self.raw = torch.zeros(self.d_out * (self.d_out + 1) // 2,
                                   self.d_in * (self.d_in + 1) // 2, dtype=a.dtype, device=a.device)
        width = self.raw.shape[0] + self.raw.shape[1]
        chunk_rows = max(1, chunk_bytes // (3 * width * a.element_size()))
        for start in range(0, a.shape[0], chunk_rows):
            stop = min(start + chunk_rows, a.shape[0])
            self.raw.addmm_(half_vec_outer(g[start:stop]).T, half_vec_outer(a[start:stop]))

    # -- the normalised object ---------------------------------------------------------------------

    def rearranged(self, n_probes: Optional[int] = None) -> Tensor:
        """``R(B^exp)``, ``(d_out^2, d_in^2)``, divided by ``n_probes`` (default: its own count)."""
        if self.kind == "norm":
            raise TypeError("a normalisation layer's B^exp is dense; use dense()")
        assert self.raw is not None
        return unpack_half_vec(self.raw, self.d_out, self.d_in) / (n_probes or self.n_probes)

    def dense(self, n_probes: Optional[int] = None) -> Tensor:
        if self.kind != "norm":
            raise TypeError("only a normalisation layer's B^exp is held densely")
        assert self.raw is not None
        return self.raw / (n_probes or self.n_probes)

    # -- folds -------------------------------------------------------------------------------------

    def to(self, device: Any) -> "SharingSums":
        moved = SharingSums(self.name, self.kind, self.d_in, self.d_out, self.positions,
                            self.n_probes)
        for spec in fields(self):
            value = getattr(self, spec.name)
            setattr(moved, spec.name, value.to(device) if isinstance(value, Tensor) else value)
        return moved

    @classmethod
    def merge(cls, entries: Sequence["SharingSums"]) -> "SharingSums":
        first = entries[0]
        raw = first.raw
        for entry in entries[1:]:
            assert raw is not None and entry.raw is not None
            raw = raw + entry.raw
        return SharingSums(first.name, first.kind, first.d_in, first.d_out, first.positions,
                           sum(entry.n_probes for entry in entries), raw)


def sharing_sums_for(layer: CapturedLayer) -> SharingSums:
    a = augmented_input(layer, "expand")
    return SharingSums(name=layer.name, kind=layer.kind, d_in=int(a.shape[-1]),
                       d_out=int(layer.g.shape[-1]), positions=layer.positions)


# ------------------------------------------------------------------------------------------------
# The decomposition itself
# ------------------------------------------------------------------------------------------------


def _gap(fro_a2: float, inner: float, fro_b2: float) -> float:
    return max(fro_a2 - 2.0 * inner + fro_b2, 0.0) ** 0.5


def linear_decomposition(block_rearranged: Tensor, bexp_rearranged: Tensor, kfac: BlockOps,
                         exact_diagonal: Tensor, d_out: int, d_in: int) -> List[Dict[str, Any]]:
    """The rows of ``plan_exp_lot3.md`` §0.5 for a shared linear/conv layer, all by contraction.

    ``kfac`` is K-FAC-**expand** (a ``Kron``); ``exact_diagonal`` is ``diag B`` in ``rvec([W|b])``.
    The two differences of norms are computed from inner products, so they inherit the usual
    ``sqrt(eps)`` floor when the two operands nearly coincide (``frobenius.py``'s caveat).
    """
    fro_b2 = float((block_rearranged * block_rearranged).sum())
    fro_e2 = float((bexp_rearranged * bexp_rearranged).sum())
    inner_be = float((block_rearranged * bexp_rearranged).sum())
    A, G = kfac.A, kfac.G  # type: ignore[attr-defined]
    inner_ek = float(G.reshape(-1) @ bexp_rearranged @ A.reshape(-1))
    fro_k2 = float(kfac.fro2())
    fro_b, fro_e = fro_b2 ** 0.5, fro_e2 ** 0.5
    independence = _gap(fro_e2, inner_ek, fro_k2)

    out_index = torch.arange(d_out, device=bexp_rearranged.device) * (d_out + 1)
    in_index = torch.arange(d_in, device=bexp_rearranged.device) * (d_in + 1)
    diag_bexp = bexp_rearranged[out_index][:, in_index].reshape(-1)
    diag_kfac = kfac.diag()
    norm_diag_b = float(exact_diagonal.norm())
    norm_diag_e = float(diag_bexp.norm())

    gram = (bexp_rearranged @ bexp_rearranged.T if bexp_rearranged.shape[0]
            <= bexp_rearranged.shape[1] else bexp_rearranged.T @ bexp_rearranged)
    squared = torch.linalg.eigvalsh(gram).flip(0).clamp_min(0.0)
    total = float(squared.sum())
    rows = {
        "bexp_fro": fro_e,
        "bexp_cos": inner_be / (fro_b * fro_e) if fro_b and fro_e else float("nan"),
        "sharing_share": _gap(fro_b2, inner_be, fro_e2) / fro_b if fro_b else float("nan"),
        "independence_share_exp": independence / fro_e if fro_e else float("nan"),
        "independence_share_total": independence / fro_b if fro_b else float("nan"),
        "diag_sharing": (float((exact_diagonal - diag_bexp).norm()) / norm_diag_b
                         if norm_diag_b else float("nan")),
        "diag_independence": (float((diag_bexp - diag_kfac).norm()) / norm_diag_e
                              if norm_diag_e else float("nan")),
        "bexp_sigma2_over_sigma1": (float((squared[1] / squared[0]).sqrt())
                                    if squared.numel() > 1 and total > 0 else 0.0),
        "bexp_kron_tail_mass": float(squared[1:].sum() / total) if total > 0 else 0.0,
    }
    return [{"metric": name, "value": value} for name, value in rows.items()]


def norm_decomposition(block: Tensor, bexp: Tensor, hadamard: BlockOps) -> List[Dict[str, Any]]:
    """The same two arrows on a ``2C x 2C`` normalisation block, with the Hadamard form as the
    independence step (Proposition 3.1 is built from expand-style statistics)."""
    fro_b = float(torch.linalg.matrix_norm(block))
    fro_e = float(torch.linalg.matrix_norm(bexp))
    had = hadamard.to_dense()
    rows = {
        "bexp_fro": fro_e,
        "bexp_cos": float((block * bexp).sum()) / (fro_b * fro_e) if fro_b and fro_e else
        float("nan"),
        "sharing_share": float(torch.linalg.matrix_norm(block - bexp)) / fro_b if fro_b else
        float("nan"),
        "independence_share_exp": float(torch.linalg.matrix_norm(bexp - had)) / fro_e if fro_e else
        float("nan"),
        "independence_share_total": float(torch.linalg.matrix_norm(bexp - had)) / fro_b if fro_b
        else float("nan"),
    }
    return [{"metric": name, "value": value} for name, value in rows.items()]


def rearranged_block(block: Tensor, d_out: int, d_in: int) -> Tensor:
    """``R(B)`` as a contiguous tensor — built once per layer and shared by every contraction."""
    return rearrange(block, d_out, d_in).contiguous()


__all__ = ["SharingSums", "half_vec_outer", "half_vec_weights", "linear_decomposition",
           "norm_decomposition", "rearranged_block", "sharing_sums_for", "unpack_half_vec"]
