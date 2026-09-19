"""Building the whole zoo from one traversal of the probe set.

One sweep of :func:`fisher_ref.capture.iter_probe_columns` accumulates, per layer, everything every
structure needs: the K-FAC factors ``A`` and ``G``, TKFAC's un-normalised numerators, and a
normalisation layer's second moments of the normalised input and of the output gradient. EKFAC
needs a **second** sweep, because its eigenvalues are defined in an eigenbasis that only exists once
``A`` and ``G`` do.

The reference and the structures therefore see the same probes by construction, which is the point:
with two independent loops the gap ``||R - K||`` would carry a sampling difference between the two
that no metric could tell apart from structure error. That guarantee holds for the deterministic
sources; with the Monte-Carlo source the two sweeps draw independently unless the caller re-seeds
between them.

Normalisations
--------------

``A`` and ``G`` are accumulated for a normalisation layer too, but they are not what the
normalisation readings use: ``A`` there is the ``(C+1) x (C+1)`` second moment of the *normalised*
input, which does not even have the size of a ``2C``-parameter block. The readings use
``norm_stats()`` instead, built from the raw ``x_hat`` and ``g``. The reduce mode does not apply to
a normalisation layer, so its statistics are always the expand-style ones.

Public API
----------

:data:`KFAC_MODES`  ``("expand", "reduce")``.

:func:`augmented_input`, :func:`output_grad`  the per-mode pooled statistics. In ``reduce`` mode the
input is **averaged** over positions and the gradient **summed**, following Eschenhagen et al.
(arXiv:2311.00636 §3.3, Eq. 10); the bias entry of the mean of ``[a_t; 1]`` is ``1``.

:func:`per_sample_gradient`  ``G_n = sum_t g_t a_bar_t^T``, the **true** per-example gradient in
``rvec([W | b])`` layout, whatever mode supplies the basis it is projected into. EKFAC's optimal
diagonal is defined on the per-example gradient, so a reduced statistic must not be substituted
here.

:func:`project_per_sample`  ``sum_n [(Q_G^T G_n Q_A)_ij]^2`` for one traversal step.

:class:`LayerFactors`  one layer's accumulated sums, with ``A`` and ``G`` as properties that apply
the right normalisation, ``kfac()`` / ``tkfac()`` / ``tkfac_eigenbases()`` / ``norm_stats()`` as
constructors, and ``to()`` / ``merge()`` for the fold engine. ``merge`` adds every raw sum and every
count, so merging folds equals a single accumulation over their union up to reduction order.

:func:`accumulate_factors`  the first sweep.

:func:`accumulate_ekfac`  the second sweep.

Dependencies: :mod:`fisher_ref.capture`, :mod:`fisher_ref.approx.kfac`,
:mod:`fisher_ref.approx.ekfac`, :mod:`fisher_ref.approx.tkfac`,
:mod:`fisher_ref.approx.norm_layers`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Dict, Mapping, Optional, Sequence

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
    """``(N, T, d_in[+1])`` for ``expand``; ``(N, 1, d_in[+1])`` of position-**averaged** statistics
    for ``reduce``.

    K-FAC-reduce's input factor is ``1/(N R^2) sum_n (sum_r a_{n,r})(sum_r a_{n,r})^T``
    (arXiv:2311.00636 §3.3, Eq. 10) — the mean over the ``R`` shared positions — while its gradient
    factor keeps the **sum** (:func:`output_grad`). The bias entry of the mean of ``[a_t; 1]`` is
    ``1``. Lot 2 summed here and appended ``T``, which made every reduce structure exactly ``T^2``
    times too large; invisible on A1 (``T = 1``), measured at 24.000 relative error for ``T = 5``
    (``plan_exp_lot3.md`` §0.1, pinned by T5).
    """
    a = layer.a
    if mode == "reduce":
        a = a.mean(dim=1, keepdim=True)
    if getattr(layer.module, "bias", None) is not None:
        a = torch.cat([a, a.new_ones((a.shape[0], a.shape[1], 1))], dim=2)
    return a


def output_grad(layer: CapturedLayer, mode: str = "expand") -> Tensor:
    """``(N, T, d_out)`` for ``expand``; ``(N, 1, d_out)`` summed over positions for ``reduce``."""
    return layer.g.sum(dim=1, keepdim=True) if mode == "reduce" else layer.g


def per_sample_gradient(layer: CapturedLayer) -> Tensor:
    """``G_n = sum_t g_t a_bar_t^T``, ``(N, d_out, d_in[+1])`` — the **true** per-example gradient in
    ``rvec([W | b])`` layout, whatever K-FAC mode supplies the basis it is projected into.

    EKFAC's ``s* = E[((Q_G (x) Q_A)^T grad)^2]`` is defined on the per-example gradient
    (``ekfac_1806.03884.pdf`` §3.2). Lot 2's reduce branch projected ``(sum_t a)(sum_t g)^T``
    instead, which is not a gradient unless the layer is in the reduce setting, so ``s`` was not the
    optimal diagonal and ``tr(K) = tr(B)`` failed (``plan_exp_lot3.md`` §0.2, T9-shared).
    """
    return torch.einsum("ntd,nte->ned", augmented_input(layer, "expand"),
                        output_grad(layer, "expand"))


def project_per_sample(layer: CapturedLayer, q_a: Tensor, q_g: Tensor) -> Tensor:
    """``sum_n [(Q_G^T G_n Q_A)_ij]^2`` for one ``(micro-batch, column)`` step, ``(d_out, d_in)``."""
    projected = torch.einsum("oi,boj,jk->bik", q_g, per_sample_gradient(layer), q_a)
    return (projected * projected).sum(dim=0)


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

    def tkfac_eigenbases(self) -> "tuple":
        """``(Q_Phi, Q_Psi)`` — TKFAC's own eigenbasis, which TEKFAC rescales inside (lot 5,
        ``plan_exp_lot5.md`` §0.5).

        Read off the **un-normalised** numerators: dividing a symmetric matrix by the positive
        scalar ``delta`` leaves its eigenvectors alone, so ``eigh(Phi_raw)`` and ``eigh(Phi)`` give
        the same basis — which is also the shortcut ``adafisher_modes/approximations/tekfac.py``
        already takes in its own ``refresh``.
        """
        assert self.tkfac_phi is not None and self.tkfac_psi is not None
        return ekfac_eigenbases(self.tkfac_phi, self.tkfac_psi)

    def to(self, device: object) -> "LayerFactors":
        """Every accumulated tensor on ``device``.

        The accumulation runs where the traversal runs (the GPU), while the metrics run against the
        host-resident dense reference — so the two must be brought together explicitly. Leaving
        them apart is not a crash at the boundary but a crash *later*, deep inside a metric's
        einsum, which is how the first P1 cluster job died. The fields are enumerated from the
        dataclass itself (lot 3), so a tensor field added later cannot be forgotten here.
        """
        moved = LayerFactors(name=self.name, kind=self.kind, d_in=self.d_in, d_out=self.d_out,
                             positions=self.positions, n_probes=self.n_probes)
        for spec in fields(self):
            value = getattr(self, spec.name)
            setattr(moved, spec.name, value.to(device) if isinstance(value, Tensor) else value)
        return moved

    @classmethod
    def merge(cls, entries: Sequence["LayerFactors"]) -> "LayerFactors":
        """The factors of the union of several disjoint probe sets: every raw sum and every count
        added, so ``merge(folds)`` equals a single accumulation over their probes up to reduction
        order (``plan_exp_lot3.md`` §0.8). The entries must be on one device.
        """
        if not entries:
            raise ValueError("nothing to merge")
        first = entries[0]
        merged = LayerFactors(name=first.name, kind=first.kind, d_in=first.d_in,
                              d_out=first.d_out, positions=first.positions, n_probes=0)
        for spec in fields(first):
            if spec.name in ("name", "kind", "d_in", "d_out", "positions"):
                continue
            values = [getattr(entry, spec.name) for entry in entries]
            if all(value is None for value in values):
                setattr(merged, spec.name, None)
                continue
            if any(value is None for value in values):
                raise ValueError(f"cannot merge {first.name!r}: {spec.name} missing on some folds")
            total = values[0]
            for value in values[1:]:
                total = total + value
            setattr(merged, spec.name, total)
        return merged

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

    # TKFAC's un-normalised numerators, one (Lambda, Gamma) = (a a^T, g g^T) per (example, column,
    # position): positions are flattened into the batch, exactly as adafisher_modes'
    # instantaneous_raw_factors does. Then tr(K) = delta = (1/N) sum ||a_t||^2 ||g_t||^2 = tr(B^exp)
    # (T8-exp); in reduce mode T = 1 after the reduction, so the same lines give the reduced block's
    # trace. Lot 2 used per-*example* traces sum_t ||a_t||^2, whose product carries every
    # cross-position pair and matches neither tr(B) nor tr(B^exp) (plan_exp_lot3.md §0.3); at T = 1
    # the two coincide. Keeping the numerators raw is what makes tr(K) exact.
    trace_lambda = (a * a).sum(dim=2)
    trace_gamma = (g * g).sum(dim=2)
    entry.tkfac_delta = entry.tkfac_delta.to(a) + (trace_lambda * trace_gamma).sum()
    weighted_a = torch.einsum("nt,ntd,nte->de", trace_gamma, a, a)
    weighted_g = torch.einsum("nt,ntd,nte->de", trace_lambda, g, g)
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
    from those factors (of whichever ``mode`` they were built in) — so the two sweeps must have
    traversed the same probes, which is what ``iter_probe_columns`` guarantees. ``G_{n,c}`` is
    always the true per-sample gradient (:func:`per_sample_gradient`), never a reduced statistic.
    """
    if mode not in KFAC_MODES:
        raise ValueError(f"mode must be one of {KFAC_MODES}; got {mode!r}")
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
            contribution = project_per_sample(layer, q_a, q_g)
            sums[layer.name] = (contribution if layer.name not in sums
                                else sums[layer.name] + contribution)
    return {name: EKFAC(QA=bases[name][0], QG=bases[name][1], s=value / n_probes)
            for name, value in sums.items()}


__all__ = ["KFAC_MODES", "LayerFactors", "accumulate_ekfac", "accumulate_factors",
           "augmented_input", "output_grad", "per_sample_gradient", "project_per_sample"]
