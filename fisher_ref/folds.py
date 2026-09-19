"""One traversal of the probes, several consumers, and per-layer error bars for free.

Every statistic the structural protocol accumulates -- the dense reference, the K-FAC factors in
both weight-sharing modes, TKFAC's numerators, a normalisation layer's second moments, the
position-wise block ``B^exp``, EKFAC's projected second moments -- is a **sum over probes**. So
instead of accumulating each once, this module accumulates them separately on ``K`` contiguous
folds of the probe order (already a seeded random permutation, from the training harness's own
split), offloads each fold's sums to the host at its boundary, and assembles any union of folds by
addition. One traversal then gives the full-set result **and** every balanced half/half partition
of the folds, where a naive implementation rebuilds a half-size reference from scratch per
partition.

Two passes, as EKFAC requires (its eigenbasis only exists once the factors do):

* **pass 1** feeds the dense accumulator, the expand and reduce factor stores, and the ``B^exp``
  sums;
* **pass 2** feeds EKFAC's ``sum_n [(Q_G^T G_n Q_A)_ij]^2`` for every basis, the bases being the
  **full-set** ones for every fold -- so a half's EKFAC carries the eigenvalue noise but not the
  basis noise. That is a declared approximation, not an oversight.

Device discipline: accumulation happens on the traversal's device, one fold at a time, and
everything stored is a host copy. The dense matrix is ``P x P`` per fold, which is why the host,
not a small GPU slice, holds them.

**The two passes are paired, including under ``source="mc"``.** ``type2`` and ``empirical`` are
deterministic given the probes, so they pair themselves. Monte-Carlo does not: it draws labels from
each sample's predicted distribution, and pass 1 builds the reference while pass 2 builds EKFAC's
rescaling against it. Different draws in the two passes put a *sampling* difference between the
reference and the approximation that no metric can tell from structure error -- EKFAC's
``sum_ij s_ij = tr(B)`` identity stops holding, and with it EKFAC's Frobenius dominance over K-FAC,
so the runner's ``invariant_violation`` guard fires on what is really a pairing bug.

Both passes therefore build their own :class:`torch.Generator` from the **same** ``mc_seed``
(:func:`mc_generator`) rather than sharing one object: a generator advances as it is used, so one
shared object would have handed pass 2 the state pass 1 left behind. Seeding twice restores the
same state by construction. ``mc_seed`` is the knob for sweeping draws; ``k`` is the number of
labels drawn per sample, which no caller could set before. Both are inert on the deterministic
sources, which ignore ``k`` and the generator entirely.

Public API
----------

:class:`FoldPlan`  contiguous micro-batches grouped into folds of equal size. It refuses a probe
count that does not divide evenly, because unequal folds make the two halves of a partition unequal
and a half's variance scales as one over its size.

:class:`FoldSums`  one fold's contribution, on the host, plus ``n_columns``: how many root columns
one probe contributed, which is the rank budget a reference's own metadata quotes.

:func:`first_pass`, :func:`second_pass`  the two traversals.

:func:`mc_generator`  the seeded generator both passes draw from, or ``None`` off ``source="mc"``.

:func:`eigenbases`  ``{basis: {layer: (Q_A, Q_G)}}`` from the full-set factors. ``tkfac_basis=True``
adds one more basis whose eigenvectors are TKFAC's own factors rather than K-FAC's; projecting the
per-sample gradients into it gives TEKFAC.

:func:`assemble`  a probe subset's normalised objects, built from its folds by addition.

:func:`partitions`  distinct balanced two-way partitions of the folds.

:func:`interval`  ``(low, high, sd)`` with ``value -/+ 1.96 sd / sqrt(2)``. ``sd`` is the sample
standard deviation of the half-size values; the ``1/sqrt(2)`` converts it to the spread of a
full-size estimate, since a variance that scales as one over the sample size is halved when the
sample doubles. It is deliberately a standard deviation and not a standard error: the interval
describes where one full-set estimate would land, not how well their mean is known.

Dependencies: :mod:`fisher_ref.capture`, :mod:`fisher_ref.reference.dense`,
:mod:`fisher_ref.approx.factors`, :mod:`fisher_ref.approx.ekfac`, :mod:`fisher_ref.approx.sharing`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .approx.ekfac import EKFAC, ekfac_eigenbases
from .approx.factors import LayerFactors, _accumulate, project_per_sample
from .approx.sharing import SharingSums, sharing_sums_for
from .capture import iter_probe_columns
from .reference.dense import DenseAccumulator, ParamLayout, symmetrize_

MODES = ("expand", "reduce")


@dataclass(frozen=True)
class FoldPlan:
    """Contiguous micro-batches grouped into ``folds`` folds of (nearly) equal size."""

    n_probes: int
    batch_size: int
    folds: int

    def __post_init__(self) -> None:
        if self.folds < 1:
            raise ValueError(f"folds must be >= 1; got {self.folds}")
        if self.folds > 1 and self.n_probes % (self.folds * self.batch_size):
            # Unequal folds make the two halves of a partition unequal too, and a half's variance
            # scales as 1/size — the interval would silently mix two sample sizes.
            raise ValueError(
                f"{self.folds} folds of whole micro-batches need n_probes divisible by "
                f"folds * batch_size = {self.folds * self.batch_size}; got {self.n_probes}. For "
                f"45 000 probes and 10 folds, batch 250 or 500 works (plan_exp_lot3.md §0.8)."
            )

    @property
    def n_batches(self) -> int:
        return math.ceil(self.n_probes / self.batch_size)

    def fold_of(self, start: int) -> int:
        return (start // self.batch_size) * self.folds // self.n_batches

    def sizes(self) -> List[int]:
        sizes = [0] * self.folds
        for start in range(0, self.n_probes, self.batch_size):
            sizes[self.fold_of(start)] += min(self.batch_size, self.n_probes - start)
        return sizes


@dataclass
class FoldSums:
    """Everything additive one fold contributed, on the host.

    ``n_columns`` is how many root columns each probe contributed (``C`` for the type-2 softmax
    root, ``k`` for Monte-Carlo, 1 for the empirical source). It travels with the fold because the
    accumulator that knows it is local to :func:`first_pass`, and a reference's own metadata quotes
    ``n_probes * n_columns`` as the number of rows of ``U`` -- the rank budget behind "this
    reference is not rank-limited by its probe count".
    """

    n_probes: int
    dense: Optional[Tensor] = None
    n_columns: int = 0
    factors: Dict[str, Dict[str, LayerFactors]] = field(default_factory=dict)
    sharing: Dict[str, SharingSums] = field(default_factory=dict)
    ekfac: Dict[str, Dict[str, Tensor]] = field(default_factory=dict)


def _host(tensor: Tensor) -> Tensor:
    return tensor.detach().to("cpu", copy=True)


def mc_generator(source: str, seed: int, device: Any) -> Optional[torch.Generator]:
    """The seeded generator the Monte-Carlo draws come from, or ``None`` for the other sources.

    Built fresh from ``seed`` by each pass. That is the whole pairing mechanism: a generator
    advances as it is used, so handing both passes one object would give pass 2 the state pass 1
    finished on, and the two would sample different labels for the same probe.

    ``None`` off ``source="mc"`` is not a shortcut. ``type2`` and ``empirical`` never touch a
    generator (:func:`fisher_ref.sources.output_root`), so passing one would be inert -- and
    returning ``None`` makes that inertness visible at the call site instead of implied.
    """
    if source != "mc":
        return None
    # ``device`` reaches here as whatever the caller had -- a string, a ``torch.device``, an index,
    # or nothing at all. One empty allocation turns all of those into a real device, which is what
    # a generator needs: ``multinomial`` refuses one that is not on the probabilities' own device.
    generator = torch.Generator(device=torch.empty(0, device=device).device)
    generator.manual_seed(seed)
    return generator


def first_pass(model: nn.Module, inputs: Tensor, targets: Tensor, *, source: str,
               modules: Mapping[str, nn.Module], layout: ParamLayout, plan: FoldPlan,
               raw_parameters: Sequence[str] = (), loss: str = "cross_entropy",
               dtype: torch.dtype = torch.float64, device: Any = "cpu",
               sharing: bool = True, check_rows: bool = True, k: int = 1, mc_seed: int = 0,
               log: Any = None) -> Tuple[List[FoldSums], Optional[Dict[str, float]]]:
    """Pass 1 over the probes: the dense reference, both K-FAC modes, ``B^exp``, per fold.

    Reduce factors are accumulated only for linear/conv layers with ``T > 1`` (at ``T = 1`` they are
    expand's, identically); ``B^exp`` only for layers with ``T > 1`` (at ``T = 1`` it is ``B``).
    Returns the folds and the row-check report of the first step.

    ``k`` is the Monte-Carlo sample count and ``mc_seed`` seeds its draws; both are ignored by
    ``type2`` and ``empirical``. :func:`second_pass` must be given the **same** two values, or the
    two passes see different labels -- see this module's header.
    """
    sizes = plan.sizes()
    accumulator = DenseAccumulator(layout, dtype=dtype, device=device)
    folds: List[FoldSums] = []
    current = -1
    stores: Dict[str, Dict[str, LayerFactors]] = {}
    shares: Dict[str, SharingSums] = {}

    def flush() -> None:
        if current < 0:
            return
        fold = FoldSums(n_probes=sizes[current], dense=accumulator.offload(),
                        n_columns=accumulator.n_columns)
        fold.factors = {mode: {name: entry.to("cpu") for name, entry in store.items()}
                        for mode, store in stores.items()}
        fold.sharing = {name: entry.to("cpu") for name, entry in shares.items()}
        folds.append(fold)

    for step in iter_probe_columns(model, inputs, targets, source=source, modules=modules,
                                   loss=loss, k=k, batch_size=plan.batch_size, dtype=dtype,
                                   device=device,
                                   generator=mc_generator(source, mc_seed, device),
                                   raw_parameters=raw_parameters):
        fold_index = plan.fold_of(step.start)
        if fold_index != current:
            flush()
            current = fold_index
            stores = {mode: {} for mode in MODES}
            shares = {}
            if log is not None:
                log(f"    fold {fold_index + 1}/{plan.folds} ({sizes[fold_index]} probes)")
        accumulator.consume(step, model=model, check=check_rows)
        first_column = step.column == 0
        for layer in step.capturer:
            _accumulate(stores["expand"], layer, sizes[current], "expand", first_column)
            if layer.positions > 1 and layer.kind != "norm":
                _accumulate(stores["reduce"], layer, sizes[current], "reduce", first_column)
            if sharing and layer.positions > 1:
                if layer.name not in shares:
                    shares[layer.name] = sharing_sums_for(layer)
                shares[layer.name].consume(layer, first_column=first_column)
    flush()
    if len(folds) != plan.folds:
        raise RuntimeError(f"expected {plan.folds} folds, traversed {len(folds)}")
    return folds, accumulator.check_report


#: The basis key of lot 5's TEKFAC structure (``plan_exp_lot5.md`` §0.5). Not a K-FAC *mode*: it
#: reuses the expand factors' TKFAC numerators, so it never appears in :data:`MODES`.
TKFAC_BASIS = "tkfac_expand"


def eigenbases(factors: Mapping[str, Mapping[str, LayerFactors]], *, tkfac_basis: bool = False
               ) -> Dict[str, Dict[str, Tuple[Tensor, Tensor]]]:
    """``{basis: {layer: (Q_A, Q_G)}}`` from full-set factors (norm layers have no EKFAC).

    ``tkfac_basis`` adds one more basis, :data:`TKFAC_BASIS`, whose eigenvectors are TKFAC's own
    ``Phi``/``Psi`` rather than K-FAC's ``A``/``G``. Projecting the per-sample gradients into it —
    which :func:`second_pass` then does for free, since it iterates over whatever bases it is handed
    — gives **TEKFAC**, the one optimizer mode P1's zoo had no counterpart for (lot 5,
    ``plan_exp_lot5.md`` §0.5). Off by default, so lot 3's code path and its published numbers are
    untouched.
    """
    bases = {mode: {name: ekfac_eigenbases(entry.A, entry.G) for name, entry in store.items()
                    if entry.kind != "norm"}
             for mode, store in factors.items()}
    if tkfac_basis and "expand" in factors:
        bases[TKFAC_BASIS] = {name: entry.tkfac_eigenbases()
                              for name, entry in factors["expand"].items()
                              if entry.kind != "norm"}
    return bases


def second_pass(model: nn.Module, inputs: Tensor, targets: Tensor, folds: List[FoldSums], *,
                source: str, modules: Mapping[str, nn.Module],
                bases: Mapping[str, Mapping[str, Tuple[Tensor, Tensor]]], plan: FoldPlan,
                loss: str = "cross_entropy", dtype: torch.dtype = torch.float64,
                device: Any = "cpu", k: int = 1, mc_seed: int = 0) -> None:
    """Pass 2: EKFAC's projected second moments for both bases, per fold, into ``folds``.

    No raw parameter is substituted: EKFAC is defined on linear/conv layers only, so the forward is
    the plain one (the per-sample gradients of the modules are the same either way).

    ``k`` and ``mc_seed`` must repeat what :func:`first_pass` was given. Re-seeding from the same
    ``mc_seed`` is what makes this pass sample the same labels pass 1 did, so EKFAC's rescaling is
    measured against the reference it is compared with rather than against a second draw.
    """
    on_device = {mode: {name: (q_a.to(device), q_g.to(device)) for name, (q_a, q_g) in per.items()}
                 for mode, per in bases.items()}
    current = -1
    sums: Dict[str, Dict[str, Tensor]] = {}

    def flush() -> None:
        if current >= 0:
            folds[current].ekfac = {mode: {name: _host(value) for name, value in per.items()}
                                    for mode, per in sums.items()}

    for step in iter_probe_columns(model, inputs, targets, source=source, modules=modules,
                                   loss=loss, k=k, batch_size=plan.batch_size, dtype=dtype,
                                   device=device,
                                   generator=mc_generator(source, mc_seed, device),
                                   check_independence=False):
        fold_index = plan.fold_of(step.start)
        if fold_index != current:
            flush()
            current = fold_index
            sums = {mode: {} for mode in on_device}
        for layer in step.capturer:
            for mode, per in on_device.items():
                if layer.name not in per:
                    continue
                q_a, q_g = per[layer.name]
                contribution = project_per_sample(layer, q_a, q_g)
                sums[mode][layer.name] = (contribution if layer.name not in sums[mode]
                                          else sums[mode][layer.name] + contribution)
    flush()


# ------------------------------------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------------------------------------


@dataclass
class Assembled:
    """A probe subset's normalised objects, built from its folds by addition."""

    n_probes: int
    matrix: Tensor
    factors: Dict[str, Dict[str, LayerFactors]]
    sharing: Dict[str, SharingSums]
    ekfac: Dict[str, Dict[str, EKFAC]]


def assemble(folds: Sequence[FoldSums], indices: Iterable[int],
             bases: Optional[Mapping[str, Mapping[str, Tuple[Tensor, Tensor]]]] = None,
             symmetrize: bool = True) -> Assembled:
    chosen = [folds[i] for i in indices]
    n = sum(fold.n_probes for fold in chosen)
    denses = [fold.dense for fold in chosen]
    if any(dense is None for dense in denses):
        raise ValueError("a fold carries no dense sum; was first_pass run on it?")
    matrix = denses[0].clone()  # type: ignore[union-attr]
    for dense in denses[1:]:
        matrix += dense
    matrix /= n
    if symmetrize:
        symmetrize_(matrix)
    factors = {mode: {name: LayerFactors.merge([fold.factors[mode][name] for fold in chosen])
                      for name in chosen[0].factors.get(mode, {})}
               for mode in chosen[0].factors}
    sharing = {name: SharingSums.merge([fold.sharing[name] for fold in chosen])
               for name in chosen[0].sharing}
    ekfac: Dict[str, Dict[str, EKFAC]] = {}
    if bases is not None:
        for mode, per in bases.items():
            ekfac[mode] = {}
            for name, (q_a, q_g) in per.items():
                if not all(name in fold.ekfac.get(mode, {}) for fold in chosen):
                    continue
                total = sum((fold.ekfac[mode][name] for fold in chosen[1:]),
                            start=chosen[0].ekfac[mode][name].clone())
                ekfac[mode][name] = EKFAC(QA=q_a, QG=q_g, s=total / n)
    return Assembled(n_probes=n, matrix=matrix, factors=factors, sharing=sharing, ekfac=ekfac)


def partitions(folds: int, count: int, seed: int = 0) -> List[Tuple[List[int], List[int]]]:
    """``count`` distinct balanced partitions of ``range(folds)`` into two halves (a half never
    appears with its complement twice). ``folds`` must be even."""
    if folds % 2 or folds < 2:
        raise ValueError(f"balanced partitions need an even number of folds; got {folds}")
    available = math.comb(folds, folds // 2) // 2
    if count > available:
        raise ValueError(f"only {available} distinct balanced partitions of {folds} folds; "
                         f"asked for {count}")
    generator = torch.Generator().manual_seed(seed)
    seen: set = set()
    out: List[Tuple[List[int], List[int]]] = []
    while len(out) < count:
        order = torch.randperm(folds, generator=generator).tolist()
        first = sorted(order[: folds // 2])
        second = sorted(order[folds // 2:])
        key = tuple(min(first, second))
        if key in seen:
            continue
        seen.add(key)
        out.append((first, second))
    return out


def interval(full_value: float, half_values: Sequence[float]) -> Tuple[float, float, float]:
    """``(low, high, sd_half)`` with ``value -/+ 1.96 sd_half / sqrt(2)``: a half has ``N/2`` probes,
    and a variance that scales as ``1/N`` is halved at ``N`` (``plan_exp_lot3.md`` §0.8)."""
    values = [v for v in half_values if v == v]  # drop NaN
    if len(values) < 2:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / len(values)
    sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
    half_width = 1.96 * sd / math.sqrt(2.0)
    return full_value - half_width, full_value + half_width, sd


__all__ = ["Assembled", "FoldPlan", "FoldSums", "MODES", "TKFAC_BASIS", "assemble",
           "eigenbases", "first_pass", "interval", "mc_generator", "partitions", "second_pass"]
