"""The structural protocol: every approximation family against the exact references, at fixed weights.

It answers "what is the best approximation *possible* in each family": same weights, same probes,
no running average, no min-max renormalisation, damping swept. The operational counterpart, which
reads the preconditioner the optimizer actually holds, is :mod:`fisher_ref.runners.p2_operational`.

This is a real command line. It sweeps
``(model, arm, seed, fraction) x structure x source x damping`` and writes a long-format
``metrics.csv`` plus a ``meta.json`` sidecar under
``fisher_ref/outputs/<model>/<arm>/seed<n>/<fraction>/``::

    PYTHONPATH=src:. python -m fisher_ref.runners.p1_structural --model mlp_ln_mnist \
        --arm diag --fractions 0,0.01,0.1,0.5,1 --probes 4000

What it computes, per parameter block
-------------------------------------

The structures are assembled by :mod:`fisher_ref.approx`: the exact diagonal and a trace-matched
identity as controls; K-FAC, TKFAC, EKFAC, TEKFAC and AdaFisher's raw diagonal on a linear or
convolutional layer, in both weight-sharing modes where the layer has more than one position; the
best rank-one Kronecker fit; the four normalisation-layer readings; and the positional-embedding
structures. Every one is compared by the Frobenius metric always, and by the Stein/KL gap and the
natural-gradient step ratio under their own separate size gates -- separate because they cost
separate things: the Stein gap densifies the structure and solves with a ``P x P`` right-hand side
per (structure, damping), while the step ratio's only ``P x P`` work is one Cholesky per damping,
shared by every structure.

Some orderings between structures are **theorems**, not findings: the exact diagonal is the
Frobenius-optimal diagonal, an optimal diagonal in a basis beats any other diagonal in that basis,
and the best rank-one Kronecker fit beats any Kronecker product. They are checked at runtime and a
violation is written out as an ``invariant_violation`` row, because it would be a bug.

Error bars come from the fold engine (``--folds``): one traversal accumulates on ``K`` folds, and
balanced two-way partitions of those folds give the whole-matrix noise floor, a per-layer floor,
and an interval on every full-set row. The paired differences between a reduce structure and its
expand counterpart get an interval on the **difference**, because both are judged against the same
reference halves and their marginal intervals are strongly correlated.

Structures whose scale is arbitrary -- the readings built from the optimizer's own formulas, which
differentiate the *mean* loss and therefore carry a factor of one over the squared batch size --
are rescaled by ``c*`` before the step ratio, or their result would be the plain-gradient step's
whatever their shape.

Extension point
---------------

:class:`ExtraStructures` injects blocks built **outside** the probe traversal, with their own
protocol label and their own damping convention. That is how the operational runner reuses this one
instead of copying it: the structural numbers and the operational ones then come from **one**
reference build, on the same probes at the same weights.

Dependencies: :mod:`fisher_ref.approx`, :mod:`fisher_ref.metrics`, :mod:`fisher_ref.folds`,
:mod:`fisher_ref.probes`, :mod:`fisher_ref.registry`, :mod:`fisher_ref.checkpoints`,
:mod:`fisher_ref.reference`, :mod:`fisher_ref.conventions`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks.common.runner import discover_benchmarks  # noqa: E402
from fisher_ref import conventions, metrics, probes, registry  # noqa: E402
from fisher_ref import folds as fold_engine  # noqa: E402
from fisher_ref.approx import (  # noqa: E402
    Dense,
    Diag,
    Kron,
    af_raw_from_factors,
    cross_term_share,
    exact_separate,
)
from fisher_ref.approx.base import BlockOps, rearrange  # noqa: E402
from fisher_ref.approx.embed import (  # noqa: E402
    kfac_onehot,
    position_blockdiag,
    position_coupling_share,
)
from fisher_ref.approx.norm_layers import DIAG_PY_TYPES, diag_py_reading  # noqa: E402
from fisher_ref.approx.sharing import linear_decomposition, norm_decomposition  # noqa: E402
from fisher_ref.capture import (  # noqa: E402
    capturable_modules,
    layer_kind,
    prepare_model,
    raw_parameter_names,
)
from fisher_ref.checkpoints import discover_runs, load_theta  # noqa: E402
from fisher_ref.metrics.coupling import as_rows as coupling_rows  # noqa: E402
from fisher_ref.reference import (  # noqa: E402
    DenseReference,
    ParamLayout,
    build_dense_reference,
    reference_parameter_names,
    to_augmented,
)

#: The result file's columns, in order. Long format: one row per measured number, so a new
#: structure or a new metric never changes the schema.
COLUMNS = ("layer", "layer_type", "structure", "source", "reference", "lambda_alpha", "metric",
           "value", "ci_low", "ci_high", "N", "probe_split", "seed", "protocol", "metrics_version")

#: ``(a, b)``: ``e_F(a) <= e_F(b)`` is a theorem, not a measurement. A violation is a bug; it is
#: recorded as an ``invariant_violation`` row and printed, never reported as a finding.
THEOREMS: Tuple[Tuple[str, str], ...] = (
    ("ekfac", "kfac"), ("ekfac_reduce", "kfac_reduce"),
    # TEKFAC Thm 3.1: the optimal diagonal in TKFAC's own eigenbasis beats any other diagonal in
    # that basis, and TKFAC's own eigenvalues are one such other diagonal.
    ("tekfac", "tkfac"),
    ("best_kron", "kfac"), ("best_kron", "tkfac"), ("best_kron", "kfac_reduce"),
    ("best_kron", "tkfac_reduce"),
    ("exact_diag", "af_raw"), ("exact_diag", "identity"), ("exact_diag", "hadamard_diag"),
    ("exact_diag", "diag_py"), ("exact_separate", "hadamard"), ("exact_separate", "exact_diag"),
    ("position_blockdiag", "kfac_onehot"), ("position_blockdiag", "exact_diag"),
)
#: Paired differences, ``(a, b)`` meaning "report ``metric(a) - metric(b)`` with an interval on
#: the difference". The two members of a pair are judged against the same reference halves, so
#: their marginal intervals are strongly correlated and comparing them separately would be the
#: wrong, over-conservative test.
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("kfac_reduce", "kfac"), ("ekfac_reduce", "ekfac"), ("tkfac_reduce", "tkfac"),
    # Lot 5's HF7 pairs. All five P2 structures are judged against the *same* reference halves, so
    # their fold intervals are strongly positively correlated and comparing two marginal intervals
    # is the wrong, over-conservative test; the interval on the **difference** is the right one, and
    # this is the machinery lot 3 already built for HF2 (plan_exp_lot5.md §0.9). Inert on a lot-3
    # run, where none of these structures exists.
    ("p2_kfac", "kfac"), ("p2_ekfac", "ekfac"), ("p2_tkfac", "tkfac"), ("p2_tekfac", "tekfac"),
    ("p2_diag", "diag_py_factors"),
    # The operator against its own second draw: the degeneracy gate's own error bar.
    ("p2rep_kfac", "p2_kfac"), ("p2rep_ekfac", "p2_ekfac"), ("p2rep_tkfac", "p2_tkfac"),
    ("p2rep_tekfac", "p2_tekfac"), ("p2rep_diag", "p2_diag"),
)
#: The structures whose scale is arbitrary: M5 is evaluated on ``c* K`` (``plan_exp_lot3.md`` §0.7).
#: Lot 5's two other readings of the optimizer's own formulas join ``diag_py``: all three are built
#: from the gradient of the **batch-mean** loss, so their scale carries ``CLAUDE.md`` §4.3's
#: ``1/batch^2`` — measured at ``c* = 1.6e4`` against ``batch = 128``'s ``16 384``. Without the
#: rescaling their ``rho`` would be the plain-gradient step's, whatever their shape.
#: The **P2** structures are deliberately *not* here: their scale is the object under measurement.
SCALE_FREE = ("diag_py", "diag_py_factors", "kron_py", "kron_py_1batch")


def rescaled(structure: BlockOps, factor: float) -> BlockOps:
    """``factor * K``, for the scale-free readings of :data:`SCALE_FREE`.

    A scalar multiple of a Kronecker product is still one (fold it into either factor), and of an
    eigen-decomposition still one (scale the diagonal) — so nothing is densified and ``solve`` stays
    closed-form.
    """
    from ..approx.ekfac import EKFAC  # noqa: PLC0415

    if isinstance(structure, Diag):
        return Diag(structure.values * factor)
    if isinstance(structure, Kron):
        return Kron(A=structure.A, G=structure.G * factor)
    if isinstance(structure, EKFAC):
        return EKFAC(QA=structure.QA, QG=structure.QG, s=structure.s * factor)
    if isinstance(structure, Dense):
        return Dense(structure.matrix * factor)
    raise TypeError(f"no rescaling defined for {type(structure).__name__}")


@dataclass(frozen=True)
class ExtraStructures:
    """Structures measured alongside the zoo's, built **outside** the probe traversal.

    Lot 5's P2 rungs (``plan_exp_lot5.md`` §0.10). ``None`` everywhere is P1 exactly as lots 2 and
    3 ran it; the three fields are what a non-P1 structure needs and a P1 one does not:

    * ``blocks[layer][structure]`` — the operator, already in the layout P1 holds that block in;
    * ``self_damped`` — structures whose damping is already inside them, so M5 and M3 must apply
      them with ``k_lam = 0`` rather than adding the sweep's ``lambda`` on top (§0.3);
    * ``protocol[structure]`` — the row label (``"P2"``, ``"P2-raw"``, ``"P1-py"``).

    They are **probe-independent**, which is what makes their fold intervals meaningful: the same
    fixed operator is evaluated against each half's reference, so the interval carries the
    reference's sampling noise alone.
    """

    blocks: Mapping[str, Mapping[str, BlockOps]] = field(default_factory=dict)
    self_damped: FrozenSet[str] = frozenset()
    protocol: Mapping[str, str] = field(default_factory=dict)
    #: M3 on an extra structure costs a dense ``K`` and a Cholesky solve with a ``P x P`` right-hand
    #: side, and M3 is not in HF7's verdict — so it is skipped above this width and the skip is
    #: written down as a row (``plan_exp_lot5.md`` §0.13). ``0`` = always skip.
    stein_max_p: int = 0

    def for_layer(self, name: str) -> Dict[str, BlockOps]:
        return dict(self.blocks.get(name, {}))

    def names(self) -> FrozenSet[str]:
        return frozenset(n for per in self.blocks.values() for n in per)


EMPTY_EXTRAS = ExtraStructures()


def _rows(records: Sequence[Dict[str, Any]], **defaults: Any) -> List[Dict[str, Any]]:
    out = []
    for record in records:
        row = {column: None for column in COLUMNS}
        row.update(defaults)
        row.update(record)
        out.append(row)
    return out


def _layer_types(model: torch.nn.Module, example: torch.Tensor) -> Dict[str, str]:
    return {info.name: info.layer_type for info in registry.classify(model, example)}


# ------------------------------------------------------------------------------------------------
# One block, one probe subset
# ------------------------------------------------------------------------------------------------


@dataclass
class Block:
    """Everything the structures of one parameter block are built from, for one probe subset."""

    name: str
    kind: str                 # "linear" | "conv" | "norm" | "embed"
    positions: int
    exact: torch.Tensor       # rvec([W|b]) for linear/conv; named_parameters() order otherwise
    expand: Any = None        # LayerFactors
    reduce: Any = None
    ekfac_expand: Any = None
    ekfac_reduce: Any = None
    tekfac_expand: Any = None
    sharing: Any = None       # SharingSums
    diag_py: Optional[Diag] = None
    shape: Tuple[int, int] = (0, 0)   # (d_out, d_in) for the rearrangement

    @property
    def P(self) -> int:
        return int(self.exact.shape[0])


def make_block(name: str, kind: str, positions: int, layout: ParamLayout, matrix: torch.Tensor,
               assembled: fold_engine.Assembled, raw_shapes: Mapping[str, torch.Size],
               diag_py: Mapping[str, Diag]) -> Block:
    columns = layout.block_slice(name)
    if kind in ("linear", "conv"):
        exact = to_augmented(matrix[columns, columns], layout.augmented_permutation(name))
        expand = assembled.factors["expand"][name]
        block = Block(name, kind, positions, exact, expand=expand,
                      reduce=assembled.factors.get("reduce", {}).get(name),
                      ekfac_expand=assembled.ekfac.get("expand", {}).get(name),
                      ekfac_reduce=assembled.ekfac.get("reduce", {}).get(name),
                      tekfac_expand=assembled.ekfac.get(fold_engine.TKFAC_BASIS, {}).get(name),
                      sharing=assembled.sharing.get(name),
                      diag_py=diag_py.get(name),
                      shape=(expand.d_out, expand.d_in))
        return block
    if kind == "norm":
        expand = assembled.factors["expand"][name]
        return Block(name, kind, positions, matrix[columns, columns].clone(), expand=expand,
                     sharing=assembled.sharing.get(name), diag_py=diag_py.get(name),
                     shape=(expand.d_out, expand.d_out))
    shape = raw_shapes[name]
    width = int(shape[-1])
    return Block(name, "embed", 1, matrix[columns, columns].clone(),
                 shape=(int(math.prod(shape[1:-1])), width))


def structures_of(block: Block, rearranged: Optional[torch.Tensor],
                  with_diag_py: bool = True, extra: Optional[ExtraStructures] = None
                  ) -> Tuple[Dict[str, BlockOps], List[Dict[str, Any]]]:
    """The structures of one block, and the block-level rows (M7, cross terms, decomposition)."""
    exact = block.exact
    trace = float(exact.diagonal().sum())
    structures: Dict[str, BlockOps] = {
        "exact_diag": Diag(exact.diagonal().clone()),
        "identity": Diag(torch.full((block.P,), trace / block.P, dtype=exact.dtype)),
    }
    rows: List[Dict[str, Any]] = []
    d_out, d_in = block.shape
    if block.kind in ("linear", "conv"):
        entry = block.expand
        kfac = entry.kfac()
        structures["kfac"] = kfac
        structures["tkfac"] = entry.tkfac()
        # App. A.3 divides both diagonal factors by |T|; K-FAC-expand only the input one (§0.4).
        structures["af_raw"] = af_raw_from_factors(entry.A, entry.G, positions=block.positions)
        if with_diag_py and block.diag_py is not None:
            # Lot 5's P1-py rung on a linear/conv layer: diag.py's own formula, called not
            # re-implemented, with no running average, no min-max and no damping. Empty unless
            # --diag-py-types asks for it, so lot 3's runs are unchanged (plan_exp_lot5.md §0.4).
            structures["diag_py"] = block.diag_py
        if block.ekfac_expand is not None:
            structures["ekfac"] = block.ekfac_expand
        if block.tekfac_expand is not None:
            structures["tekfac"] = block.tekfac_expand
        assert rearranged is not None
        report, A_best, G_best = metrics.kronecker_analysis(
            rearranged, d_out, d_in, exact_diagonal=exact.diagonal(), kfac_diagonal=kfac.diag())
        structures["best_kron"] = Kron(A=A_best, G=G_best)
        rows += report.as_rows(structure="exact_block")
        if block.positions > 1 and block.reduce is not None:
            structures["kfac_reduce"] = block.reduce.kfac()
            structures["tkfac_reduce"] = block.reduce.tkfac()
            if block.ekfac_reduce is not None:
                structures["ekfac_reduce"] = block.ekfac_reduce
        if block.positions > 1 and block.sharing is not None:
            rows += [{**r, "structure": "b_exp"} for r in linear_decomposition(
                rearranged, block.sharing.rearranged(), kfac, exact.diagonal(), d_out, d_in)]
    elif block.kind == "norm":
        stats = block.expand.norm_stats()
        channels = d_out
        structures["hadamard"] = stats.hadamard()
        structures["exact_separate"] = exact_separate(exact, channels)
        structures["hadamard_diag"] = stats.hadamard_diagonal()
        if with_diag_py and block.diag_py is not None:
            structures["diag_py"] = block.diag_py
        total_share, diagonal_share = cross_term_share(exact, channels)
        rows += [{"structure": "exact_block", "metric": "cross_term_share_total",
                  "value": total_share},
                 {"structure": "exact_block", "metric": "cross_term_share_diagonal",
                  "value": diagonal_share}]
        if block.positions > 1 and block.sharing is not None:
            rows += [{**r, "structure": "b_exp"} for r in norm_decomposition(
                exact, block.sharing.dense(), structures["hadamard"])]
    else:
        structures["position_blockdiag"] = position_blockdiag(exact, d_out, d_in)
        structures["kfac_onehot"] = kfac_onehot(exact, d_out, d_in)
        share, _ = position_coupling_share(exact, d_out, d_in)
        rows.append({"structure": "exact_block", "metric": "position_coupling_share",
                     "value": share})
        if rearranged is not None:
            report, _, _ = metrics.kronecker_analysis(rearranged, d_out, d_in)
            rows += report.as_rows(structure="exact_block")
    if extra is not None:
        structures.update(extra.for_layer(block.name))
    return structures, rows


def _rearranged(block: Block) -> Optional[torch.Tensor]:
    if block.kind in ("linear", "conv", "embed"):
        return rearrange(block.exact, *block.shape).contiguous()
    return None


def analyse_block(block: Block, *, alphas: Sequence[float], gradient: Optional[torch.Tensor],
                  stein_max_p: int, rho_max_p: int, common: Dict[str, Any],
                  extra: Optional[ExtraStructures] = None
                  ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Every structure of one block at every damping — M1 always, M3/M5 under their size gates.

    M3 and M5 have **separate** gates because they cost separate things: M3 densifies ``K`` and
    solves with a ``P x P`` right-hand side per ``(structure, lambda)``; M5's only ``P x P`` work is
    one Cholesky per ``lambda``, shared by every structure (lot 2, ``p1_features0_a1.sh``).
    """
    extra = extra or EMPTY_EXTRAS
    rearranged = _rearranged(block)
    structures, block_rows = structures_of(block, rearranged, extra=extra)
    rows = _rows(block_rows, **common)
    fro_r = float(torch.linalg.matrix_norm(block.exact))
    e_f: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    for name, structure in structures.items():
        report = metrics.frobenius(block.exact, structure, fro_R=fro_r, rearranged=rearranged)
        e_f[name] = report.e_F
        scales[name] = report.c_star
        rows += _rows(report.as_rows(), **common, structure=name)
    del rearranged

    for smaller, larger in THEOREMS:
        if smaller in e_f and larger in e_f and e_f[smaller] > e_f[larger] + 1e-9:
            print(f"  !! invariant violated on {block.name}: e_F({smaller})={e_f[smaller]:.12g} > "
                  f"e_F({larger})={e_f[larger]:.12g}", flush=True)
            rows += _rows([{"metric": "invariant_violation",
                            "value": e_f[smaller] - e_f[larger]}],
                          **common, structure=f"{smaller}<={larger}")

    grid = metrics.lambda_grid(block.exact, alphas)
    extra_names = extra.names()
    for name, structure in structures.items():
        # M3 costs a dense K plus a Cholesky solve with a P x P right-hand side, per (structure,
        # lambda), and it is not in HF7's verdict -- so lot 5's structures have their own, far
        # tighter gate, and the skip is a row rather than a silent absence (plan_exp_lot5.md §0.13).
        limit = extra.stein_max_p if name in extra_names else stein_max_p
        if block.P > limit:
            metric = "stein_kl_skipped_protocol" if name in extra_names else "stein_kl_skipped_P"
            rows += _rows([{"metric": metric, "value": float(block.P)}], **common, structure=name)
            continue
        for alpha, lam in grid.items():
            k_lam = 0.0 if name in extra.self_damped else None
            rows += _rows(metrics.stein_kl(block.exact, structure, lam, k_lam=k_lam).as_rows(),
                          **common, structure=name, lambda_alpha=alpha)

    if gradient is not None:
        if block.P > rho_max_p:
            rows += _rows([{"metric": "rho_skipped_P", "value": float(block.P)}
                           for _ in structures], **common)
            for row, name in zip(rows[-len(structures):], structures):
                row["structure"] = name
        else:
            for alpha, lam in grid.items():
                factor = metrics.damped_cholesky(block.exact, lam)
                for name, structure in structures.items():
                    used = (rescaled(structure, scales[name])
                            if name in SCALE_FREE and scales[name] == scales[name] else structure)
                    k_lam = 0.0 if name in extra.self_damped else None
                    rows += _rows(metrics.rho(block.exact, used, gradient, lam, factor=factor,
                                              k_lam=k_lam).as_rows(),
                                  **common, structure=name, lambda_alpha=alpha)
                del factor
    # The row label is per structure, not per run: one CSV carries P1, P1-py and the P2 rungs, on
    # the same probes at the same theta, which is what makes the HF7 comparison paired.
    if extra.protocol:
        for row in rows:
            row_structure = row.get("structure")
            label = (extra.protocol.get(row_structure)
                     if isinstance(row_structure, str) else None)
            if label is not None:
                row["protocol"] = label
    return rows, e_f


def half_values(block: Block, *, alphas: Sequence[float], gradient: Optional[torch.Tensor],
                rho_max_p: int, decomposition_max_p: int,
                extra: Optional[ExtraStructures] = None
                ) -> Dict[Tuple[str, str, Optional[float]], float]:
    """The per-half numbers an interval is built from: M1 for every structure, ``rho`` and the
    decomposition under their own size gates. ``diag_py`` is excluded (one full-set reading, no
    half); ``rho`` uses the full-set gradient, so its interval carries the noise of ``R`` and ``K``
    only (``plan_exp_lot3.md`` §0.8)."""
    extra = extra or EMPTY_EXTRAS
    out: Dict[Tuple[str, str, Optional[float]], float] = {}
    rearranged = _rearranged(block)
    sharing = block.sharing if block.P <= decomposition_max_p else None
    structures, block_rows = structures_of(
        Block(**{**block.__dict__, "sharing": sharing}), rearranged, with_diag_py=False,
        extra=extra)
    for row in block_rows:
        out[(row["structure"], row["metric"], None)] = float(row["value"])
    fro_r = float(torch.linalg.matrix_norm(block.exact))
    scales: Dict[str, float] = {}
    for name, structure in structures.items():
        report = metrics.frobenius(block.exact, structure, fro_R=fro_r, rearranged=rearranged)
        scales[name] = report.c_star
        for metric in ("e_F", "cos_F", "e_F_star"):
            out[(name, metric, None)] = float(getattr(report, metric))
    del rearranged
    if gradient is not None and block.P <= rho_max_p:
        for alpha, lam in metrics.lambda_grid(block.exact, alphas).items():
            factor = metrics.damped_cholesky(block.exact, lam)
            for name, structure in structures.items():
                # The same c* rescaling analyse_block applies, or a scale-free reading's interval
                # would be computed on the degenerate object (rho = the plain-gradient step's,
                # whatever its shape) and pasted onto a rescaled point value.
                scale = scales.get(name, float("nan"))
                used = (rescaled(structure, scale)
                        if name in SCALE_FREE and scale == scale else structure)
                k_lam = 0.0 if name in extra.self_damped else None
                out[(name, "rho", alpha)] = metrics.rho(block.exact, used, gradient, lam,
                                                        factor=factor, k_lam=k_lam).rho
            del factor
    return out


# ------------------------------------------------------------------------------------------------
# One source at one checkpoint
# ------------------------------------------------------------------------------------------------


def _log(message: str) -> None:
    print(message, flush=True)


def run_source(*, source: str, model: torch.nn.Module, inputs: torch.Tensor,
               targets: torch.Tensor, modules: Mapping[str, torch.nn.Module],
               raw: Sequence[str], layout: ParamLayout, kinds: Mapping[str, str],
               positions: Mapping[str, int], gradient: torch.Tensor,
               diag_py: Mapping[str, Diag], defaults: Dict[str, Any], args: argparse.Namespace,
               meta: Dict[str, Any], fraction: float, digest: str, loss: str,
               extra: Optional[ExtraStructures] = None) -> List[Dict[str, Any]]:
    n = int(inputs.shape[0])
    n_folds = args.folds if (source in args.fold_sources and args.folds > 1) else 1
    plan = fold_engine.FoldPlan(n_probes=n, batch_size=args.batch, folds=n_folds)
    started = time.perf_counter()
    # k and mc_seed are repeated verbatim in pass 2: the two passes pair themselves by re-seeding
    # from one number, never by sharing a generator object (fisher_ref.folds's header).
    mc = {"k": args.mc_samples, "mc_seed": args.mc_seed}
    folds, report = fold_engine.first_pass(
        model, inputs, targets, source=source, modules=modules, layout=layout, plan=plan,
        raw_parameters=raw, loss=loss, dtype=conventions.REFERENCE_DTYPE, device=args.device,
        sharing=source in args.sharing_sources, check_rows=args.check_rows, log=_log, **mc)
    _log(f"  [{source}] pass 1 ({n_folds} fold(s)): {time.perf_counter() - started:.1f}s; "
         f"row checks {report}")
    started = time.perf_counter()
    factors_only = fold_engine.assemble(folds, range(n_folds), symmetrize=False)
    bases = fold_engine.eigenbases(factors_only.factors, tkfac_basis=args.tekfac)
    del factors_only
    fold_engine.second_pass(model, inputs, targets, folds, source=source, modules=modules,
                            bases=bases, plan=plan, loss=loss,
                            dtype=conventions.REFERENCE_DTYPE, device=args.device, **mc)
    full = fold_engine.assemble(folds, range(n_folds), bases=bases)
    _log(f"  [{source}] pass 2 + assembly: {time.perf_counter() - started:.1f}s")

    reference_name = "E_hat" if source == "empirical" else "F"
    common_source = {**defaults, "reference": reference_name, "source": source}
    # n_columns is the traversal's own count (C for type2, k for mc, 1 for empirical), carried out
    # of pass 1 on the folds. It was hardcoded to 0, which made the recorded n_rows = 0 too — and
    # n_rows is the rank budget behind "this reference is not rank-limited by its probe count".
    reference = DenseReference(matrix=full.matrix, layout=layout, source=source, loss=loss,
                               n_probes=n, n_columns=folds[0].n_columns, probe_digest=digest)
    meta.setdefault("references", {})[f"{fraction}/{source}"] = {
        **reference.metadata(), "row_checks": report, "folds": n_folds}

    raw_shapes = {name: dict(model.named_parameters())[name].shape for name in raw}
    names = list(modules) + list(raw)
    # An injected structure whose layer is not analysed here would be re-warmed, snapshotted and
    # then dropped with no row and no warning -- which happens the moment --modules restricts the
    # P1 side. Say so rather than lose it quietly.
    if extra is not None:
        orphans = sorted(set(extra.blocks) - set(names))
        if orphans:
            _log(f"  !! {len(orphans)} injected structure(s) have no analysed block and are "
                 f"dropped: {orphans[:8]}{'...' if len(orphans) > 8 else ''}")
            meta.setdefault("dropped_extras", {})[f"{fraction}/{source}"] = orphans
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for name in names:
        kind = "embed" if name in raw else str(layer_kind(modules[name]))
        block = make_block(name, kind, positions.get(name, 1), layout, full.matrix, full,
                           raw_shapes, diag_py)
        columns = layout.block_slice(name)
        layer_gradient = gradient[columns]
        if kind in ("linear", "conv"):
            layer_gradient = layer_gradient[layout.augmented_permutation(name)]
        common = {**common_source, "layer": name, "layer_type": kinds.get(name, "other")}
        block_rows, _ = analyse_block(block, alphas=args.alphas, gradient=layer_gradient,
                                      stein_max_p=args.stein_max_p, rho_max_p=args.rho_max_p,
                                      common=common, extra=extra)
        rows += block_rows
        del block
    _log(f"  [{source}] per-block metrics: {time.perf_counter() - started:.1f}s")

    ranges = [(name, layout.block_slice(name)) for name in names]
    coupling_names, coupling = metrics.coupling_matrix(full.matrix, ranges)
    coupling_records = coupling_rows(coupling_names, coupling)
    for record in coupling_records:
        record["layer_type"] = kinds.get(record["layer"], "other")
    rows += _rows(coupling_records, **common_source, structure="exact_block")
    rows += _rows([{"metric": key, "value": value} for key, value in
                   metrics.offdiagonal_mass(full.matrix, ranges).items()],
                  **common_source, structure="block_diagonal")

    if n_folds > 1:
        started = time.perf_counter()
        rows += fold_intervals(folds, bases, rows, names=names, raw=raw, modules=modules,
                               positions=positions, layout=layout, gradient=gradient,
                               kinds=kinds, common_source=common_source, args=args, meta=meta,
                               fraction=fraction, source=source, raw_shapes=raw_shapes,
                               extra=extra)
        _log(f"  [{source}] fold intervals ({args.partitions} partitions): "
             f"{time.perf_counter() - started:.1f}s")
    del folds, full
    gc.collect()
    return rows


def fold_intervals(folds: List[fold_engine.FoldSums], bases: Any, rows: List[Dict[str, Any]], *,
                   names: Sequence[str], raw: Sequence[str], modules: Mapping[str, Any],
                   positions: Mapping[str, int], layout: ParamLayout, gradient: torch.Tensor,
                   kinds: Mapping[str, str], common_source: Dict[str, Any],
                   args: argparse.Namespace, meta: Dict[str, Any], fraction: float, source: str,
                   raw_shapes: Mapping[str, torch.Size],
                   extra: Optional[ExtraStructures] = None) -> List[Dict[str, Any]]:
    """Whole-matrix floor, per-layer floors, and ``ci_low``/``ci_high`` on the full-set rows, from
    ``args.partitions`` balanced partitions of the folds (``plan_exp_lot3.md`` §0.8, §0.10)."""
    splits = fold_engine.partitions(len(folds), args.partitions, seed=args.seed)
    whole: List[float] = []
    layer_floor: Dict[str, List[float]] = {name: [] for name in names}
    halves: Dict[str, Dict[Tuple[str, str, Optional[float]], List[float]]] = {
        name: {} for name in names}
    for index, (first, second) in enumerate(splits):
        pair = [fold_engine.assemble(folds, first, bases=bases),
                fold_engine.assemble(folds, second, bases=bases)]
        whole.append(metrics.dense_gap(pair[0].matrix, pair[1].matrix)["rel_to_geom"])
        for name in names:
            kind = "embed" if name in raw else str(layer_kind(modules[name]))
            columns = layout.block_slice(name)
            layer_floor[name].append(metrics.dense_gap(pair[0].matrix[columns, columns],
                                                       pair[1].matrix[columns, columns])[
                "rel_to_geom"])
            layer_gradient = gradient[columns]
            if kind in ("linear", "conv"):
                layer_gradient = layer_gradient[layout.augmented_permutation(name)]
            for half in pair:
                block = make_block(name, kind, positions.get(name, 1), layout, half.matrix, half,
                                   raw_shapes, {})
                measured = half_values(block, alphas=args.alphas, gradient=layer_gradient,
                                       rho_max_p=args.noise_rho_max_p,
                                       decomposition_max_p=args.noise_decomposition_max_p,
                                       extra=extra)
                for key, value in measured.items():
                    halves[name].setdefault(key, []).append(value)
                del block
        del pair
        gc.collect()
        _log(f"    partition {index + 1}/{len(splits)}: whole-matrix split {whole[-1]:.4f}")

    out: List[Dict[str, Any]] = []
    floor = _floor(whole)
    out += _rows([{"metric": key, "value": value} for key, value in floor.items()],
                 **common_source, structure="reference")
    meta.setdefault("noise_floor", {})[f"{fraction}/{source}"] = {
        **floor, "partitions": len(splits), "folds": len(folds), "mode": "folds",
        "splits": whole}
    for name in names:
        stats = _floor(layer_floor[name])
        common = {**common_source, "layer": name, "layer_type": kinds.get(name, "other"),
                  "structure": "reference"}
        out += _rows([{"metric": "layer_noise_floor", "value": stats["noise_floor"],
                       "ci_low": stats["noise_floor_low"], "ci_high": stats["noise_floor_high"]},
                      {"metric": "layer_sigma_n", "value": stats["sigma_n"]},
                      {"metric": "layer_null_two_independent",
                       "value": stats["null_two_independent"]}], **common)

    # ci on the full-set rows this source produced
    for row in rows:
        layer = row.get("layer")
        if layer not in halves:
            continue
        key = (row["structure"], row["metric"], row["lambda_alpha"])
        values = halves[layer].get(key)
        if values and row["value"] is not None:
            low, high, _ = fold_engine.interval(float(row["value"]), values)
            row["ci_low"], row["ci_high"] = low, high

    # HF2's paired differences
    full_values = {(row.get("layer"), row["structure"], row["metric"], row["lambda_alpha"]):
                   row["value"] for row in rows if row.get("layer") in halves}
    for name in names:
        common = {**common_source, "layer": name, "layer_type": kinds.get(name, "other")}
        for reduced, expanded in PAIRS:
            for metric, alpha in [("e_F", None)] + [("rho", float(a)) for a in args.alphas]:
                a_full = full_values.get((name, reduced, metric, alpha))
                b_full = full_values.get((name, expanded, metric, alpha))
                a_half = halves[name].get((reduced, metric, alpha))
                b_half = halves[name].get((expanded, metric, alpha))
                if a_full is None or b_full is None:
                    continue
                delta = float(a_full) - float(b_full)
                low = high = float("nan")
                if a_half and b_half and len(a_half) == len(b_half):
                    low, high, _ = fold_engine.interval(
                        delta, [x - y for x, y in zip(a_half, b_half)])
                out += _rows([{"metric": f"delta_{metric}", "value": delta, "ci_low": low,
                               "ci_high": high, "lambda_alpha": alpha}], **common,
                             structure=f"{reduced}-{expanded}")
    return out


def _floor(values: Sequence[float]) -> Dict[str, float]:
    """The same summary :func:`fisher_ref.metrics.noise_floor.noise_floor` reports, over splits
    already measured on the folds instead of over rebuilt references.

    ``noise_floor_low``/``_high`` are the observed **range** of the splits at the 20 partitions
    every run of this package has used, not a 95 % quantile interval; ``noise_floor`` is the upper
    of the two central values at an even count. The index arithmetic, and its one-index outward
    bias against a textbook nearest-rank percentile, are documented once in that module's header.
    """
    ordered = sorted(values)
    median = ordered[len(ordered) // 2]
    low = ordered[max(int(0.025 * len(ordered)) - 1, 0)]
    high = ordered[min(int(0.975 * len(ordered)), len(ordered) - 1)]
    return {"noise_floor": median, "noise_floor_low": low, "noise_floor_high": high,
            "sigma_n": median / 2.0, "null_two_independent": median / math.sqrt(2.0)}


# ------------------------------------------------------------------------------------------------
# One checkpoint
# ------------------------------------------------------------------------------------------------


def run_fraction(bench, run, fraction: float, args, meta: Dict[str, Any],
                 write: Any = None, extra: Optional[ExtraStructures] = None
                 ) -> List[Dict[str, Any]]:
    loaded = load_theta(run, fraction, device="cpu")
    meta.setdefault("checkpoints", {})[str(fraction)] = loaded.metadata()
    # Prepared once, here: the traversal casts the batch, not the network, and the reference and
    # the accumulators must share one model object (capture.prepare_model's docstring).
    model = prepare_model(loaded.model, conventions.REFERENCE_DTYPE, args.device)
    train = probes.build_probe_set(bench, split="train", n=args.probes, seed=args.seed,
                                   data_root=args.data_root)
    inputs, targets = train.as_model_batch(bench)
    modules = capturable_modules(model)
    if args.modules:
        modules = {name: modules[name] for name in args.modules if name in modules}
    raw = _raw_parameters(model, args)
    layout = ParamLayout.of(model, reference_parameter_names(model, modules, raw))
    # device *and* dtype: casting only the dtype left the example on the host while the model
    # had moved to the GPU, which is what killed cluster job 21125955 after 36 s.
    example = inputs[:2].to(device=args.device, dtype=conventions.REFERENCE_DTYPE)
    infos = registry.classify(model, example)
    kinds = {info.name: info.layer_type for info in infos}
    positions = {info.name: int(info.positions or 1) for info in infos}
    # setdefault on the *outer* key, then assign the fraction. Spelling it
    # `setdefault("blocks", {str(fraction): ...})` built the inner dict every time and threw it
    # away from the second fraction on, so a five-checkpoint run recorded only the first one's
    # layer inventory.
    meta.setdefault("blocks", {})[str(fraction)] = {
        name: {"kind": kinds.get(name), "positions": positions.get(name, 1)}
        for name in list(modules) + raw}
    loss = "cross_entropy" if isinstance(bench.loss_fn, torch.nn.CrossEntropyLoss) else "mse"

    defaults = {"N": len(train), "probe_split": "train", "seed": run.seed, "protocol": "P1",
                "metrics_version": conventions.METRICS_VERSION, "reference": "F"}

    started = time.perf_counter()
    # M5 needs the true gradient at this theta, in the reference's layout — micro-batched: one
    # forward over 45 000 CIFAR images is ~90 GB of fp64 activations (plan_exp_lot3.md §0.9).
    gradient = metrics.probe_gradient(model, inputs, targets, bench.loss_fn,
                                      names=set(layout.names), batch_size=args.batch,
                                      device=args.device,
                                      dtype=conventions.REFERENCE_DTYPE).to("cpu")
    diag_py: Dict[str, Diag] = {}
    if args.diag_py and len(train) < bench.batch_size:
        # diag.py's statistic is defined per training-size micro-batch; with fewer probes there is
        # not one, and a smaller batch would be a different estimator (plan_exp_lot3.md §0.7).
        _log(f"  diag_py skipped: {len(train)} probes < one training batch of {bench.batch_size}")
        meta.setdefault("diag_py_skipped", {})[str(fraction)] = len(train)
    elif args.diag_py:
        diag_py = {name: Diag(block.values.to("cpu")) for name, block in diag_py_reading(
            model, inputs, targets, modules, loss_fn=bench.loss_fn, batch_size=bench.batch_size,
            dtype=conventions.REFERENCE_DTYPE, device=args.device,
            types=getattr(args, "diag_py_types", None) or DIAG_PY_TYPES).items()}
    _log(f"  gradient + diag_py readings: {time.perf_counter() - started:.1f}s")

    rows: List[Dict[str, Any]] = []
    for source in args.sources:
        rows += run_source(source=source, model=model, inputs=inputs, targets=targets,
                           modules=modules, raw=raw, layout=layout, kinds=kinds,
                           positions=positions, gradient=gradient, diag_py=diag_py,
                           defaults=defaults, args=args, meta=meta, fraction=fraction,
                           digest=train.digest[:12], loss=loss, extra=extra)
        if write is not None:
            write(rows)

        wanted = args.noise_at is None or fraction in args.noise_at
        if source == "type2" and args.noise_partitions and wanted:
            block_defaults = {**defaults, "reference": "F", "source": source}

            def build(indices, _source=source):
                return build_dense_reference(model, inputs[indices], targets[indices],
                                             source=_source, batch_size=args.batch,
                                             device=args.device, modules=list(modules),
                                             raw_parameters=raw).matrix.to("cpu")

            floor = metrics.noise_floor(build, len(train), partitions=args.noise_partitions,
                                        seed=args.seed)
            rows += _rows(floor.as_rows(), **block_defaults, structure="reference")
            meta.setdefault("noise_floor", {})[str(fraction)] = {
                "median": floor.median, "low": floor.low, "high": floor.high,
                "sigma_n": floor.sigma_n, "null_two_independent": floor.null_two_independent,
                "partitions": floor.n_partitions, "mode": "rebuild"}
            if write is not None:
                write(rows)
    return rows


def _raw_parameters(model: torch.nn.Module, args: argparse.Namespace) -> List[str]:
    """``auto``: every batch-broadcast raw parameter when the whole model is analysed, none when
    ``--modules`` restricts it; ``none``; or explicit names."""
    available = raw_parameter_names(model)
    chosen = args.raw_parameters
    if chosen == ["auto"]:
        return [] if args.modules else available
    if chosen == ["none"]:
        return []
    unknown = sorted(set(chosen) - set(available))
    if unknown:
        raise SystemExit(f"--raw-parameters {unknown} are not batch-broadcast raw parameters of "
                         f"this model; available: {available}")
    return list(chosen)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="mlp_ln_mnist",
                        help="the run directory's label (e.g. cnn_gn_cifar_bn); the bench is read "
                             "from its manifest")
    parser.add_argument("--arm", default="diag")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fractions", default="0.5",
                        help="comma-separated checkpoint fractions, e.g. 0,0.01,0.1,0.5,1")
    parser.add_argument("--sources", nargs="+", default=["type2", "empirical"])
    parser.add_argument("--mc-samples", type=int, default=1, metavar="K",
                        help="labels drawn per probe for --sources mc; the columns carry the "
                             "K^-1/2 that makes the estimator unbiased. Ignored by type2 and "
                             "empirical, which draw nothing")
    parser.add_argument("--mc-seed", type=int, default=0,
                        help="seeds the Monte-Carlo draws. Both probe passes re-seed from it, "
                             "which is what makes them sample the same labels; change it to sweep "
                             "draws. Ignored by type2 and empirical")
    parser.add_argument("--probes", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--alphas", default="1e-4,1e-3,1e-2,1e-1,1",
                        help="lambda = alpha * tr(R)/P (plan_exp_draft.md §3.3); never a single one")
    parser.add_argument("--noise-partitions", type=int, default=20,
                        help="lot 2's floor by REBUILDING half-size references (2 x partitions "
                             "builds); 0 to skip. Lot 3 uses --folds instead")
    parser.add_argument("--noise-at", default=None,
                        help="comma-separated fractions for the rebuild floor (default: all)")
    parser.add_argument("--stein-max-p", type=int, default=4096,
                        help="skip M3 on blocks wider than this: its Cholesky is P^3/3 per "
                             "(structure, lambda) and would dominate the job at A1's first layer")
    parser.add_argument("--rho-max-p", type=int, default=None,
                        help="skip M5 on blocks wider than this (default: --stein-max-p)")
    parser.add_argument("--modules", nargs="*", default=None)
    parser.add_argument("--raw-parameters", nargs="+", default=["auto"],
                        help="'auto' (every batch-broadcast raw parameter, e.g. pos_embed, unless "
                             "--modules restricts the model), 'none', or names (plan_exp_lot3.md "
                             "§0.6)")
    parser.add_argument("--folds", type=int, default=1,
                        help="accumulate on K contiguous folds and derive intervals from "
                             "--partitions balanced partitions of them (plan_exp_lot3.md §0.8); "
                             "1 = no intervals")
    parser.add_argument("--tekfac", dest="tekfac", action="store_true", default=False,
                        help="also build TEKFAC -- EKFAC in TKFAC's own eigenbasis, the one "
                             "optimizer mode P1's zoo had no counterpart for (plan_exp_lot5.md "
                             "§0.5). Off by default, so lot 3's runs stay reproducible")
    parser.add_argument("--fold-sources", nargs="+", default=["type2"])
    parser.add_argument("--partitions", type=int, default=20)
    parser.add_argument("--sharing-sources", nargs="*", default=["type2", "empirical"],
                        help="sources on which the B^exp decomposition is accumulated "
                             "(plan_exp_lot3.md §0.5)")
    parser.add_argument("--diag-py", dest="diag_py", action="store_true", default=True)
    parser.add_argument("--no-diag-py", dest="diag_py", action="store_false")
    parser.add_argument("--check-rows", dest="check_rows", action="store_true", default=True)
    parser.add_argument("--no-check-rows", dest="check_rows", action="store_false")
    parser.add_argument("--noise-rho-max-p", type=int, default=8192,
                        help="M5 intervals only on blocks up to this width (a Cholesky per "
                             "lambda per half)")
    parser.add_argument("--noise-decomposition-max-p", type=int, default=8192,
                        help="B^exp decomposition intervals only on blocks up to this width")
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", default=str(ROOT / "benchmarks" / "data"))
    parser.add_argument("--outputs-root", default=str(ROOT / "benchmarks" / "outputs"))
    parser.add_argument("--out-dir", default=str(ROOT / "fisher_ref" / "outputs"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.alphas = [float(value) for value in str(args.alphas).split(",") if value]
    args.noise_at = ([float(v) for v in str(args.noise_at).split(",") if v]
                     if args.noise_at else None)
    args.rho_max_p = args.stein_max_p if args.rho_max_p is None else args.rho_max_p
    fractions = [float(value) for value in str(args.fractions).split(",") if value]
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    state = conventions.configure()
    torch.manual_seed(args.seed)
    runs = discover_runs(args.outputs_root, model=args.model, arm=args.arm, seed=args.seed)
    if len(runs) != 1:
        raise SystemExit(f"expected exactly one run for {args.model}/{args.arm} seed {args.seed}; "
                         f"found {[str(r.directory) for r in runs]}")
    run = runs[0]
    # The bench comes from the run's manifest, not from --model: `cnn_gn_cifar_bn` is a directory
    # label with no benchmarks/<name>/ folder (fisher_ref/checkpoints.py, RunRef.bench_name).
    bench = discover_benchmarks()[run.bench_name]
    print(f"P1 {args.model}/{args.arm} seed {run.seed} (bench {bench.name}, "
          f"{run.model_kwargs()}) | device={args.device} TF32 off={state.tf32_is_off()} | "
          f"folds={args.folds} partitions={args.partitions}", flush=True)

    meta: Dict[str, Any] = conventions.run_metadata({"protocol": "P1", "regime": "A",
                                                     "args": vars(args),
                                                     "bench": bench.name,
                                                     "model_kwargs": run.model_kwargs()})
    for fraction in fractions:
        started = time.perf_counter()
        out = Path(args.out_dir) / args.model / args.arm / f"seed{run.seed}" / str(fraction)
        out.mkdir(parents=True, exist_ok=True)

        def write(rows: Sequence[Dict[str, Any]], _out: Path = out) -> None:
            _write_csv(_out / "metrics.csv", rows)
            (_out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True,
                                                       default=str))

        _log(f"fraction {fraction}:")
        rows = run_fraction(bench, run, fraction, args, meta, write=write)
        write(rows)
        print(f"  fraction {fraction}: {len(rows)} rows in {time.perf_counter() - started:.1f}s "
              f"-> {out}", flush=True)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    import csv  # noqa: PLC0415
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in COLUMNS})
    temporary.replace(path)


if __name__ == "__main__":
    main()
