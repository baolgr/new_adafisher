"""The operational protocol: the optimizer's own preconditioner against the exact references.

The structural protocol asks *"how good can this family of approximations possibly be?"*. This one
asks *"how good is the thing the optimizer actually divides by?"* -- after the running average,
after the min-max renormalisation, at its own damping, estimated in float32 from augmented training
batches with normalisation layers in train mode. Same weights, same probes, same references: the
two protocols differ in exactly one thing, which is the object being judged::

    PYTHONPATH=src:. python -m fisher_ref.runners.p2_operational --model cnn_gn_cifar \
        --arm diag --fractions 1,0,0.5,0.1,0.01 --probes 45000 --batch 250 --folds 10

**It is the structural runner with four rungs instead of one.** Everything expensive -- the
reference build, the fold engine, the metrics -- is :mod:`fisher_ref.runners.p1_structural`'s,
called with an ``ExtraStructures``. Copying that code would have reintroduced the divergence the
package spent effort removing, and worse: "the structural number" and "the operational number"
would come from two builds nobody compared. Here they come from one traversal.

The four rungs, all in one ``metrics.csv``, distinguished by the ``protocol`` column:

========= ==================================================================================
``P1``     the structure alone: float64, per-example backprop vectors, no averaging
``P1-py``  the optimizer's **own formulas** on the same probes, still with no averaging, no
           min-max renormalisation and no damping
``P2-raw`` those formulas after the running average, the min-max, float32 and the training
           data, but **without** the optimizer's damping
``P2``     the operator the optimizer inverts, its own damping folded in
========= ==================================================================================

and, at one checkpoint per model, two diagnostics: a **replica**, a second re-warm on an
independent batch order, which is the operator's *own* draw-to-draw spread (the fold intervals
carry the reference's noise, not the operator's); and a **clean** re-warm fed the campaign's probe
tensors in eval mode, which separates "the data the optimizer sees" from "the running average".

The replica's seed offset has to exceed the step count, because the re-warm's batch stream reseeds
the global generator with ``seed + steps_produced`` before every pass over the loader: an offset of
one would replay the main run's own stream shifted by one step, reusing its augmentation and its
dropout masks.

A third reading is emitted alongside: the same Kronecker formulas on **one** micro-batch. Without
it the ``P1-py -> P2-raw`` step would attribute to "the running average and the min-max" what is
mostly plain sampling variance, since the running average's effective sample size is about 1.2
factor updates.

Everything the optimizer does **not** hook takes the identity preconditioner exactly, because its
fallback is plain momentum stochastic gradient descent. That is a result about the model, not a
hole in the table, and it is emitted once rather than once per mode.

Dependencies: :mod:`fisher_ref.runners.p1_structural`, :mod:`fisher_ref.rewarm`,
:mod:`fisher_ref.approx.adafisher_state`, :mod:`fisher_ref.approx.norm_layers`,
:mod:`fisher_ref.probes`, :mod:`fisher_ref.checkpoints`, :mod:`fisher_ref.conventions`.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks.common.runner import discover_benchmarks  # noqa: E402
from fisher_ref import conventions, probes  # noqa: E402
from fisher_ref.approx.adafisher_state import (  # noqa: E402
    MODES,
    least_check,
    snapshot,
    worst_check,
)
from fisher_ref.approx.base import BlockOps, Diag  # noqa: E402
from fisher_ref.approx.norm_layers import (  # noqa: E402
    HOOKED_TYPES,
    diag_py_factor_reading,
    kron_py_reading,
)
from fisher_ref.capture import capturable_modules, prepare_model, raw_parameter_names  # noqa: E402
from fisher_ref.checkpoints import discover_runs  # noqa: E402
from fisher_ref.rewarm import (  # noqa: E402
    MIN_REWARM_FACTOR_UPDATES,
    RewarmSpec,
    hparams_of,
    loader_settings,
    rewarm,
)
from fisher_ref.runners import p1_structural  # noqa: E402
from fisher_ref.runners.p1_structural import ExtraStructures  # noqa: E402

#: ``structure name -> protocol`` for the rungs that are not P1.
PROTOCOLS = {"P1-py": ("diag_py", "diag_py_factors", "kron_py", "kron_py_1batch")}

#: The re-warm variants and the structure-name prefix each writes.
VARIANT_PREFIX = {"main": "p2", "replica": "p2rep", "clean": "p2clean"}

#: Seed offset of the replica re-warm. **Not 1**: ``rewarm._train_batches`` reseeds the global RNG
#: with ``seed + produced`` before every epoch pass, so ``seed + 1`` would replay the main run's own
#: stream shifted by one step — reusing its dropout masks and its augmentation. The whole purpose of
#: the replica is an *independent* draw, so the offset has to exceed the step count.
REPLICA_SEED_OFFSET = 10_000


def _log(message: str) -> None:
    print(message, flush=True)


def build_extras(bench, run, fraction: float, args: argparse.Namespace, hparams,
                 meta: Dict[str, Any]) -> ExtraStructures:
    """Re-warm every requested mode at this checkpoint and read its state as curvature blocks.

    Runs **before** the reference build and frees its models afterwards, so the re-warm's fp32
    network and the reference's fp64 accumulator are never on the device together.
    """
    blocks: Dict[str, Dict[str, BlockOps]] = {}
    self_damped: set = set()
    protocol: Dict[str, str] = {}
    checks: Dict[str, Any] = {}

    def put(layer: str, name: str, block: BlockOps, label: str, damped: bool) -> None:
        blocks.setdefault(layer, {})[name] = block
        protocol[name] = label
        if damped:
            self_damped.add(name)

    variants = ["main"]
    if fraction in args.variants_at:
        variants += [v for v in ("replica", "clean") if v in args.variants]

    hooked_names: set = set()
    probe_tensors = None
    if "clean" in variants:
        probe_set = probes.build_probe_set(bench, split="train", n=args.probes, seed=args.seed,
                                           data_root=args.data_root)
        probe_tensors = probe_set.as_model_batch(bench)

    for variant in variants:
        prefix = VARIANT_PREFIX[variant]
        for mode in args.modes:
            started = time.perf_counter()
            spec = RewarmSpec(
                mode=mode, steps=args.rewarm_steps, lr=args.rewarm_lr,
                data="probes" if variant == "clean" else "train",
                seed=args.rewarm_seed + (REPLICA_SEED_OFFSET if variant == "replica" else 0),
            )
            model, optimizer = rewarm(
                bench, run, fraction, spec, hparams=hparams, data_root=args.data_root,
                device=args.device, num_workers=args.rewarm_workers, probes=probe_tensors,
                allow_short=args.allow_short_rewarm,
            )
            state = snapshot(model, optimizer, mode=mode,
                             dtype=conventions.REFERENCE_DTYPE, device="cpu")
            for layer, block in state.items():
                put(layer, f"{prefix}_{mode}", block.damped, "P2", damped=True)
                put(layer, f"{prefix}_raw_{mode}", block.undamped, "P2-raw", damped=False)
            # The identity seed's residue after k factor updates. For the *damped* operator the
            # rule is 0.08^k << lambda; for the P2-raw rung, whose lambda is removed, what it has to
            # be small against is the accumulated factor's own smallest eigenvalue -- so the ratio
            # is recorded per snapshot rather than assumed (plan_exp_lot5.md §0.6.3b).
            floor = least_check(state, "min_eigenvalue_undamped")
            residue = (1.0 - hparams.gammas[0]) ** (args.rewarm_steps // hparams.tcov)
            checks[f"{fraction}/{variant}/{mode}"] = {
                key: worst_check(state, key)
                for key in ("inverse_consistency_A", "inverse_consistency_B",
                            "inverse_consistency_Phi", "inverse_consistency_Psi",
                            "delta_vintage_gap", "min_s_star", "min_theta")
            }
            checks[f"{fraction}/{variant}/{mode}"].update({
                "min_eigenvalue_undamped": floor,
                "identity_seed_residue": residue,
                "residue_over_undamped_floor": residue / floor if floor and floor == floor
                else float("nan"),
                "residue_over_lambda": residue / hparams.lam if hparams.lam else float("nan"),
            })
            _log(f"  re-warm {variant}/{mode}: {spec.steps} steps, "
                 f"{len(state)} hooked modules, {time.perf_counter() - started:.1f}s")
            # The union, not the last write: which modules the optimizer preconditions decides
            # which blocks are labelled "it preconditions nothing here" (§0.7), and a last-write-wins
            # assignment turns into a wrong classification the first time a mode is dropped from
            # --modes or the loop is reordered. Every mode must agree, and that is asserted rather
            # than assumed.
            if hooked_names and set(state) != hooked_names:
                raise RuntimeError(
                    f"mode {mode!r} hooks {sorted(set(state) ^ hooked_names)} differently from the "
                    f"modes before it; the unhooked set (§0.7) would depend on the loop order"
                )
            hooked_names |= set(state)
            del model, optimizer, state
            gc.collect()
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    if not args.modes:
        # Without a re-warm there is no `hooked_names` to subtract, so every capturable module would
        # be labelled "the optimizer does not precondition this" -- which is false.
        raise SystemExit("--modes is empty: there is no operational state to read")

    # Everything the optimizer does NOT hook takes the identity preconditioner exactly: step()'s
    # fallback is plain momentum SGD (plan_exp_lot5.md §0.7). That is a result about the model, not
    # a hole in the table, and it is emitted once rather than five identical times.
    reference_model = bench.build_model(**run.model_kwargs())
    unhooked = [name for name in capturable_modules(reference_model) if name not in hooked_names]
    unhooked += raw_parameter_names(reference_model)
    parameters = dict(reference_model.named_parameters())
    for name in unhooked:
        size = sum(p.numel() for n, p in parameters.items()
                   if n == name or n.startswith(f"{name}."))
        if size:
            put(name, "p2_identity", Diag(torch.ones(size, dtype=conventions.REFERENCE_DTYPE)),
                "P2", damped=True)
    meta.setdefault("p2", {})[str(fraction)] = {
        "rewarm_steps": args.rewarm_steps, "rewarm_lr": args.rewarm_lr,
        "rewarm_seed": args.rewarm_seed, "variants": variants, "modes": list(args.modes),
        "unhooked": unhooked, "hparams": hparams.__dict__, "checks": checks,
        "loader": loader_settings(run, bench),
    }
    protocol.update({name: label for label, names in PROTOCOLS.items() for name in names})
    return ExtraStructures(blocks=blocks, self_damped=frozenset(self_damped), protocol=protocol,
                           stein_max_p=args.stein_extras_max_p)


def add_p1_py(bench, run, fraction: float, args: argparse.Namespace,
              extra: ExtraStructures, meta: Dict[str, Any]) -> ExtraStructures:
    """The P1-py rung's two readings, on the same probes the references use.

    P1 builds **no** Kronecker structure on a normalisation layer — its own ``A`` there is the
    ``(C+1) x (C+1)`` second moment of the *normalised* input, which does not have the size of a
    ``2C``-parameter block — so without this, three of HF7's five mode pairs would have nothing to
    compare against on any normalisation layer (``plan_exp_lot5.md`` §0.4).
    """
    from fisher_ref.checkpoints import load_theta  # noqa: PLC0415

    loaded = load_theta(run, fraction, device="cpu")
    model = prepare_model(loaded.model, conventions.REFERENCE_DTYPE, args.device)
    probe_set = probes.build_probe_set(bench, split="train", n=args.probes, seed=args.seed,
                                       data_root=args.data_root)
    inputs, targets = probe_set.as_model_batch(bench)
    modules = capturable_modules(model)
    if len(probe_set) < bench.batch_size:
        # The reading's statistic is defined per training-size micro-batch; a smaller one would be a
        # different estimator (plan_exp_lot3.md §0.7's rule, applied to kron_py too).
        meta.setdefault("p1_py_skipped", {})[str(fraction)] = len(probe_set)
        _log(f"  P1-py readings skipped: {len(probe_set)} probes < one training batch "
             f"of {bench.batch_size}")
        return extra
    started = time.perf_counter()
    shared = dict(loss_fn=bench.loss_fn, batch_size=bench.batch_size,
                  dtype=conventions.REFERENCE_DTYPE, device=args.device, types=HOOKED_TYPES)
    reading = kron_py_reading(model, inputs, targets, modules, **shared)
    # The sample-size control (plan_exp_lot5.md §0.4c). With gammas = (0.92, 0.008) the normalised
    # weight on the factor update j steps back is 0.92 * 0.08^j, so sum w_j^2 = 0.852 and the
    # effective sample size is 1.17 factor updates -- about 150 examples at batch 128, against
    # 45 000 probes on the P1 side. Without this reading the P1-py -> P2-raw rung would attribute
    # to "the running average and the min-max" what is mostly plain sampling variance. Same formula,
    # same eval-mode forward, ONE micro-batch.
    one_batch = {}
    if fraction in args.variants_at:
        one_batch = kron_py_reading(model, inputs[:bench.batch_size], targets[:bench.batch_size],
                                    modules, **shared)
    # Two ways of averaging the micro-batches, both emitted so the gap between them is a measured
    # number: `diag_py` averages the product outer(S_D, H_D), `diag_py_factors` averages each factor
    # and multiplies at the end -- which is what the optimizer's running average does
    # (plan_exp_lot5.md §5). Lot 3's convention is the first, and is left untouched.
    factor_diag = diag_py_factor_reading(model, inputs, targets, modules, **shared)
    blocks = {layer: dict(per) for layer, per in extra.blocks.items()}
    for layer, block in reading.items():
        blocks.setdefault(layer, {})["kron_py"] = _to_host(block)
    for layer, block in factor_diag.items():
        blocks.setdefault(layer, {})["diag_py_factors"] = _to_host(block)
    for layer, block in one_batch.items():
        blocks.setdefault(layer, {})["kron_py_1batch"] = _to_host(block)
    _log(f"  P1-py readings ({len(reading)} modules): {time.perf_counter() - started:.1f}s")
    del model
    gc.collect()
    return ExtraStructures(blocks=blocks, self_damped=extra.self_damped, protocol=extra.protocol,
                           stein_max_p=extra.stein_max_p)


def _to_host(block: BlockOps) -> BlockOps:
    """Move a freshly-built block to the host, where every metric runs.

    Lot 2's first cluster job died 36 s in on exactly this: the accumulation happens on the device
    and the dense reference lives on the host, so a structure left on the GPU crashes inside an
    einsum minutes later (``plan_exp_lot2.md`` §5.8).
    """
    for field in ("A", "G", "QA", "QG", "s", "values", "matrix"):
        tensor = getattr(block, field, None)
        if isinstance(tensor, torch.Tensor):
            setattr(block, field, tensor.to("cpu"))
    if hasattr(block, "_eig"):
        block._eig = None
    return block


def build_parser() -> argparse.ArgumentParser:
    parser = p1_structural.build_parser()
    parser.description = __doc__.splitlines()[0]
    parser.set_defaults(out_dir=str(ROOT / "fisher_ref" / "outputs" / "p2"), tekfac=True,
                        noise_partitions=0)
    group = parser.add_argument_group("P2 (lot 5)")
    group.add_argument("--modes", nargs="+", default=list(MODES),
                       help="the Fisher modes to re-warm and read (default: all five)")
    group.add_argument("--rewarm-steps", type=int, default=1000,
                       help=f"steps of the arm's own configuration before the state is read; at "
                            f"least {MIN_REWARM_FACTOR_UPDATES} x TCov, because the identity each "
                            f"mode seeds its running average with survives as 0.08^k and acts as a "
                            f"spurious extra damping on top of lambda (plan_exp_draft.md §3.2)")
    group.add_argument("--rewarm-lr", type=float, default=0.0,
                       help="0 freezes theta during the re-warm, so every factor is estimated at "
                            "exactly the checkpoint's weights; anything else measures staleness")
    group.add_argument("--rewarm-seed", type=int, default=1000)
    group.add_argument("--rewarm-workers", type=int, default=2)
    group.add_argument("--allow-short-rewarm", action="store_true", default=False)
    group.add_argument("--variants", nargs="*", default=["replica", "clean"],
                       choices=["replica", "clean"],
                       help="the §0.4b diagnostics, computed at --variants-at only")
    group.add_argument("--variants-at", default="1",
                       help="comma-separated fractions where the diagnostics run; empty = never")
    group.add_argument("--stein-extras-max-p", type=int, default=0,
                       help="M3 on a P2 or P1-py structure only up to this block width. 0 (the "
                            "default) skips it and records a stein_kl_skipped_protocol row: M3 "
                            "costs a dense K plus a P x P solve per (structure, lambda) and is not "
                            "in HF7's verdict (plan_exp_lot5.md §0.13)")
    group.add_argument("--no-p1-py", dest="kron_py", action="store_false", default=True,
                       help="skip the P1-py rung (kron_py, diag_py_factors)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.alphas = [float(value) for value in str(args.alphas).split(",") if value]
    args.noise_at = ([float(v) for v in str(args.noise_at).split(",") if v]
                     if args.noise_at else None)
    args.variants_at = [float(v) for v in str(args.variants_at).split(",") if v]
    args.rho_max_p = args.stein_max_p if args.rho_max_p is None else args.rho_max_p
    args.diag_py_types = HOOKED_TYPES     # the P1-py rung reads every hooked layer, not only norms
    fractions = [float(value) for value in str(args.fractions).split(",") if value]
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    state = conventions.configure()
    torch.manual_seed(args.seed)
    runs = discover_runs(args.outputs_root, model=args.model, arm=args.arm, seed=args.seed)
    if len(runs) != 1:
        raise SystemExit(f"expected exactly one run for {args.model}/{args.arm} seed {args.seed}; "
                         f"found {[str(r.directory) for r in runs]}")
    run = runs[0]
    bench = discover_benchmarks()[run.bench_name]
    hparams = hparams_of(run, bench)
    print(f"P2 {args.model}/{args.arm} seed {run.seed} (bench {bench.name}, "
          f"{run.model_kwargs()}) | device={args.device} TF32 off={state.tf32_is_off()} | "
          f"modes={args.modes} rewarm={args.rewarm_steps} steps at lr={args.rewarm_lr} | "
          f"lam={hparams.lam:g} gammas={hparams.gammas} TCov={hparams.tcov}", flush=True)

    meta: Dict[str, Any] = conventions.run_metadata({"protocol": "P2", "regime": "A",
                                                     "args": {k: str(v) for k, v in
                                                              vars(args).items()},
                                                     "bench": bench.name,
                                                     "model_kwargs": run.model_kwargs()})
    for fraction in fractions:
        started = time.perf_counter()
        out = Path(args.out_dir) / args.model / args.arm / f"seed{run.seed}" / str(fraction)
        out.mkdir(parents=True, exist_ok=True)

        def write(rows: Sequence[Dict[str, Any]], _out: Path = out) -> None:
            p1_structural._write_csv(_out / "metrics.csv", rows)
            (_out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True,
                                                       default=str))

        _log(f"fraction {fraction}:")
        extra = build_extras(bench, run, fraction, args, hparams, meta)
        if args.kron_py:
            extra = add_p1_py(bench, run, fraction, args, extra, meta)
        rows = p1_structural.run_fraction(bench, run, fraction, args, meta, write=write,
                                          extra=extra)
        write(rows)
        print(f"  fraction {fraction}: {len(rows)} rows in {time.perf_counter() - started:.1f}s "
              f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
