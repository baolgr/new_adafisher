"""Building the zoo from the shared traversal (``plan_exp_lot2.md`` §0.2, §1.2).

One sweep of ``capture.iter_probe_columns`` accumulates, per layer, everything every structure
needs: the K-FAC factors ``A`` and ``G``, TKFAC's un-normalised numerators, and a normalisation
layer's ``H``/``S``. EKFAC needs a **second** sweep, because ``s`` is defined in an eigenbasis that
only exists once ``A`` and ``G`` do.

The reference and the structures therefore see the same probes and the same Monte-Carlo draws by
construction, which is the point: with two independent loops ``||R - K||`` would carry a sampling
difference between ``R`` and ``K`` that no metric could tell from structure error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..capture import CapturedLayer, iter_probe_columns
from .ekfac import EKFAC, ekfac_eigenbases
from .kfac import Kron
from .norm_layers import NormStats
from .tkfac import TkfacStats

KFAC_MODES = ("expand", "reduce")


def augmented_input(layer: CapturedLayer, mode: str = "expand") -> Tensor:
    """``(N, T, d_in[+1])`` for ``expand``; ``(N, 1, d_in[+1])`` of summed statistics for ``reduce``.

    The ones column is appended *after* the reduction for ``reduce``, so the bias sees ``T`` rather
    than ``1`` — Eschenhagen et al.'s Prop. 2 normalisation, not a re-derivation
    (``plan_exp_draft.md`` §4).
    """
    a = layer.a
    if mode == "reduce":
        a = a.sum(dim=1, keepdim=True)
    if getattr(layer.module, "bias", None) is not None:
        ones = a.new_full((a.shape[0], a.shape[1], 1), float(layer.positions) if mode == "reduce"
                          else 1.0)
        a = torch.cat([a, ones], dim=2)
    return a


def output_grad(layer: CapturedLayer, mode: str = "expand") -> Tensor:
    return layer.g.sum(dim=1, keepdim=True) if mode == "reduce" else layer.g


@dataclass
class LayerFactors:
    """Everything the zoo needs for one layer, accumulated over the whole probe set."""

    name: str
    kind: str
    d_in: int
    d_out: int
    positions: int
    n_probes: int
    a_rows: int = 0
    g_rows: int = 0
    A_raw: Optional[Tensor] = None
    G_raw: Optional[Tensor] = None
    tkfac_delta: Tensor = field(default_factory=lambda: torch.zeros(()))
    tkfac_phi: Optional[Tensor] = None
    tkfac_psi: Optional[Tensor] = None
    tkfac_count: int = 0
    norm_h: Optional[Tensor] = None
    norm_s: Optional[Tensor] = None
    norm_h_rows: int = 0

    # -- the two K-FAC factors -------------------------------------------------------------------

    @property
    def A(self) -> Tensor:
        """``E[a a^T]`` — averaged over ``(example, position)``, K-FAC-expand's normalisation."""
        assert self.A_raw is not None
        return self.A_raw / self.a_rows

    @property
    def G(self) -> Tensor:
        """``E[g g^T]``, averaged over examples and columns but **summed** over positions."""
        assert self.G_raw is not None
        return self.G_raw / self.n_probes

    def kfac(self) -> Kron:
        return Kron(A=self.A, G=self.G)

    def tkfac(self) -> Kron:
        assert self.tkfac_phi is not None and self.tkfac_psi is not None
        return TkfacStats(delta=self.tkfac_delta, phi_raw=self.tkfac_phi, psi_raw=self.tkfac_psi,
                          count=self.tkfac_count).build()

    def to(self, device: object) -> "LayerFactors":
        """Every accumulated tensor on ``device``.

        The accumulation runs where the traversal runs (the GPU), while the metrics run against the
        host-resident dense reference — so the two must be brought together explicitly. Leaving
        them apart is not a crash at the boundary but a crash *later*, deep inside a metric's
        einsum, which is how the first P1 cluster job died.
        """
        moved = LayerFactors(name=self.name, kind=self.kind, d_in=self.d_in, d_out=self.d_out,
                             positions=self.positions, n_probes=self.n_probes,
                             a_rows=self.a_rows, g_rows=self.g_rows,
                             tkfac_count=self.tkfac_count, norm_h_rows=self.norm_h_rows)
        for field_name in ("A_raw", "G_raw", "tkfac_delta", "tkfac_phi", "tkfac_psi",
                           "norm_h", "norm_s"):
            value = getattr(self, field_name)
            setattr(moved, field_name, None if value is None else value.to(device))
        return moved

    def norm_stats(self) -> NormStats:
        assert self.norm_h is not None and self.norm_s is not None
        return NormStats(H=self.norm_h / self.norm_h_rows, S=self.norm_s / self.n_probes,
                         C=self.d_out)


def _accumulate(store: Dict[str, LayerFactors], layer: CapturedLayer, n_probes: int,
                mode: str, first_column: bool) -> None:
    a = augmented_input(layer, mode)
    g = output_grad(layer, mode)
    entry = store.get(layer.name)
    if entry is None:
        entry = LayerFactors(name=layer.name, kind=layer.kind, d_in=int(a.shape[-1]),
                             d_out=int(g.shape[-1]), positions=layer.positions,
                             n_probes=n_probes)
        store[layer.name] = entry

    flat_a = a.reshape(-1, a.shape[-1])
    flat_g = g.reshape(-1, g.shape[-1])
    # The input does not depend on the backprop column: accumulate it once per micro-batch, or A
    # would be inflated by a factor C and every Kronecker structure silently rescaled.
    if first_column:
        entry.A_raw = flat_a.T @ flat_a if entry.A_raw is None else entry.A_raw + flat_a.T @ flat_a
        entry.a_rows += flat_a.shape[0]
    entry.G_raw = flat_g.T @ flat_g if entry.G_raw is None else entry.G_raw + flat_g.T @ flat_g
    entry.g_rows += flat_g.shape[0]

    # TKFAC's un-normalised numerators, per (example, column): tr(Lambda) = ||a||^2 summed over
    # positions, tr(Gamma) = ||g||^2. Keeping them raw is what makes tr(K) = tr(B_l) exact.
    trace_lambda = (a * a).sum(dim=(1, 2))
    trace_gamma = (g * g).sum(dim=(1, 2))
    entry.tkfac_delta = entry.tkfac_delta + (trace_lambda * trace_gamma).sum()
    weighted_a = torch.einsum("n,ntd,nte->de", trace_gamma, a, a)
    weighted_g = torch.einsum("n,ntd,nte->de", trace_lambda, g, g)
    entry.tkfac_phi = weighted_a if entry.tkfac_phi is None else entry.tkfac_phi + weighted_a
    entry.tkfac_psi = weighted_g if entry.tkfac_psi is None else entry.tkfac_psi + weighted_g
    # Counted per *probe*, not per (probe, column): the reference normalises B_l by N while summing
    # over the C columns, so dividing delta by N*C instead would scale tr(K) down by exactly C —
    # which is how T8 first failed, with a gap of precisely the class count.
    if first_column:
        entry.tkfac_count += int(a.shape[0])

    if layer.kind == "norm":
        # layer.a is x_hat for a normalisation layer (capture.py recomputes it in closed form).
        x_hat = layer.a.reshape(-1, layer.a.shape[-1])
        raw_g = layer.g.reshape(-1, layer.g.shape[-1])
        if first_column:
            entry.norm_h = (x_hat.T @ x_hat if entry.norm_h is None
                            else entry.norm_h + x_hat.T @ x_hat)
            entry.norm_h_rows += x_hat.shape[0]
        entry.norm_s = raw_g.T @ raw_g if entry.norm_s is None else entry.norm_s + raw_g.T @ raw_g


def accumulate_factors(model: nn.Module, inputs: Tensor, targets: Tensor, *, source: str,
                       modules: Mapping[str, nn.Module], mode: str = "expand",
                       **traversal: object) -> Dict[str, LayerFactors]:
    """One sweep: ``A``, ``G``, TKFAC's numerators and the normalisation statistics, per layer."""
    if mode not in KFAC_MODES:
        raise ValueError(f"mode must be one of {KFAC_MODES}; got {mode!r}")
    store: Dict[str, LayerFactors] = {}
    n_probes = int(inputs.shape[0])
    for step in iter_probe_columns(model, inputs, targets, source=source, modules=modules,
                                   **traversal):  # type: ignore[arg-type]
        for layer in step.capturer:
            _accumulate(store, layer, n_probes, mode, first_column=(step.column == 0))
    return store


def accumulate_ekfac(model: nn.Module, inputs: Tensor, targets: Tensor, *, source: str,
                     modules: Mapping[str, nn.Module],
                     factors: Mapping[str, LayerFactors], mode: str = "expand",
                     **traversal: object) -> Dict[str, EKFAC]:
    """The **second** sweep: ``s_ij = (1/N) sum_{n,c} [(Q_G^T G_{n,c} Q_A)_ij]^2``.

    Only the layers whose ``A``/``G`` were built in the first sweep are eligible, and the bases come
    from those factors — so the two sweeps must have traversed the same probes, which is what
    ``iter_probe_columns`` guarantees.
    """
    bases = {name: ekfac_eigenbases(entry.A, entry.G) for name, entry in factors.items()
             if entry.kind != "norm"}
    sums: Dict[str, Tensor] = {}
    n_probes = int(inputs.shape[0])
    for step in iter_probe_columns(model, inputs, targets, source=source, modules=modules,
                                   **traversal):  # type: ignore[arg-type]
        for layer in step.capturer:
            if layer.name not in bases:
                continue
            q_a, q_g = bases[layer.name]
            a = augmented_input(layer, mode)
            g = output_grad(layer, mode)
            per_sample = torch.einsum("ntd,nte->ned", a, g)      # (N, d_out, d_in), G_{n,c}
            projected = torch.einsum("oi,boj,jk->bik", q_g, per_sample, q_a)
            contribution = (projected * projected).sum(dim=0)
            sums[layer.name] = (contribution if layer.name not in sums
                                else sums[layer.name] + contribution)
    return {name: EKFAC(QA=bases[name][0], QG=bases[name][1], s=value / n_probes)
            for name, value in sums.items()}


__all__ = ["KFAC_MODES", "LayerFactors", "accumulate_ekfac", "accumulate_factors",
           "augmented_input", "output_grad"]
