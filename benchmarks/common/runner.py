"""The runner every bench calls: argument parsing, the arm loop, seeding and reporting.

A model's ``bench.py`` is a :class:`Benchmark` literal plus ``main(BENCH)`` — no control flow, no
argument parsing, no reporting of its own. Everything else happens here.

The wall-clock-time protocol
----------------------------

AdaFisher's own (``papers/adafisher_2405.16397.pdf`` §5, p. 8: *"We employ the Wall-Clock-Time
(WCT) method with a cutoff of 200 epochs for AdaFisher's training"*). One reference arm — ``diag``
by default, ``--reference-arm`` — runs a fixed ``--epochs`` epochs unbudgeted. Its measured
elapsed time becomes every other arm's budget, and each of those runs as many epochs as fit. The
budget is derived in-process; ``--wct-budget`` supplies a pre-measured number instead, so one arm
can be one independent cluster job. ``--budget-mode epochs`` disables the protocol and gives every
arm the same epoch count, which is the comparison the protocol exists to avoid: an equal-epoch
comparison is rigged in favour of the modes with the most expensive step.

``--max-epoch-factor`` (default 3) bounds how far a *budgeted* arm may run when the reference
arm's budget turns out generous. It never applies to the reference arm, which must run exactly the
nominal epoch count — it is what defines the budget.

Learning rate
-------------

``--lr-schedule`` picks between the two readings of AdaFisher Appendix D's "a cosine annealing
learning rate decay strategy [...] aligning with the number of training epochs specified for each
optimizer": ``nominal`` anneals over ``--epochs``, shared by every arm and clamped at the floor
past it; ``budget`` anneals each budgeted arm over its own wall-clock budget, so every arm
completes one full cosine. ``benchmarks/common/schedules.py`` derives both and records the two
biases the shared-``T_max`` reading was measured to introduce. The reference arm is unbudgeted and
always uses ``nominal``.

Seeding
-------

``run_arm`` seeds the global generator once, before building the model, so every arm of a model
starts from identical weights and sees identical batch orderings. The train/val split takes the
same seed through its own generator.

Arm isolation
-------------

``run_arm`` also tears the arm down before returning: it unregisters the optimizer's hooks and
collects, because a hook-based optimizer and its model form a reference cycle that plain reference
counting cannot free. Without that, every arm stayed resident for the rest of the process and each
following arm measured its peak device memory on top of its predecessors — which is what made the
reported ``peak VRAM`` column rise monotonically in the order the arms ran, with ``adam`` shown as
the most expensive of the seven. Only that column and real memory pressure change; the weights,
the records and the report are already final when the teardown runs.

What a run writes
-----------------

Into ``--output-dir`` (default ``benchmarks/outputs/<output_group>/<name>/``): ``records.csv``,
``epochs.csv``, ``summary.md``, ``manifest.json`` and, unless ``--no-plot``, a set of PNG curves.
The report is rewritten after *every* completed arm, not once at the end, so a crash in a later
arm cannot lose the arms that already finished. ``--checkpoints`` additionally dumps model weights
at fixed fractions of the nominal trajectory (``benchmarks/common/checkpoints.py``).

The registry
------------

``discover_benchmarks()`` imports every ``benchmarks/models/<name>/bench.py`` that exposes a
``BENCH``. The folder list is the registry; nothing else has to be edited to add a model.
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    get_args,
    get_origin,
    get_type_hints,
)

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .checkpoints import CheckpointWriter, parse_fractions
from .loop import (
    default_prepare_batch,
    evaluate,
    peak_memory_bytes,
    release_optimizer,
    reset_peak_memory,
    train_under_budget,
)
from .optimizers import ARMS, HParams, build_optimizer, resolve_device
from .records import (
    ArmResult,
    format_bytes,
    write_epoch_csv,
    write_manifest,
    write_plots,
    write_step_csv,
    write_summary,
)
from .schedules import BudgetCosine, NominalCosine

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

DataBuilder = Callable[..., Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]]


@dataclass(frozen=True)
class Benchmark:
    """Everything that distinguishes one model's bench from another's."""

    name: str  # = folder name = output directory
    build_model: Callable[..., nn.Module]
    build_data: DataBuilder
    loss_fn: Callable[[Tensor, Tensor], Tensor]
    hparams: HParams
    display_name: str = ""
    prepare_batch: Callable[[Any], Tuple[Tensor, Tensor]] = default_prepare_batch
    metric_fn: Optional[Callable[[Tensor, Tensor], float]] = None  # None => no accuracy
    epochs: int = 50
    batch_size: int = 128
    arms: Tuple[str, ...] = ("diag", "kfac", "ekfac", "tkfac", "tekfac", "adam", "adamw")
    # Architecture variants a model declares for itself, ``{keyword: allowed values}``; each
    # becomes a ``--<keyword>`` CLI choice defaulting to its first value, and is passed to
    # ``build_model`` as a keyword. ``cnn_gn_cifar``'s ``--norm {gn,bn}`` is its only user; every
    # other bench leaves it empty and ``build_model`` is called with no arguments.
    model_choices: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    # Optional one-level subdirectory grouping this bench's results and SLURM jobs, i.e.
    # ``outputs/<output_group>/<name>/`` and ``slurm/<output_group>/``. Empty is still the flat
    # ``outputs/<name>/`` layout, used by nothing today. The campaign's eight original models set
    # it to ``"mnist"`` / ``"cifar10"`` and the CIFAR-100 / ImageNet-1K benches to ``"cifar100"`` /
    # ``"imagenet"``, so every dataset's results and jobs sit together instead of colliding in one
    # flat directory. ``fisher_ref/checkpoints.py`` discovers runs through each bench's own
    # ``output_group`` (``CAMPAIGN_GROUPS``), not by walking arbitrary directory names — a group not
    # in that set (``cifar100``, ``imagenet``) stays invisible to it, deliberately.
    output_group: str = ""

    def title(self) -> str:
        return self.display_name or self.name

    def output_root(self, outputs_dir: Path) -> Path:
        return outputs_dir / self.output_group / self.name if self.output_group else \
            outputs_dir / self.name


def discover_benchmarks() -> Dict[str, Benchmark]:
    """The folder list *is* the registry: every ``benchmarks/models/<x>/bench.py`` exposing a
    ``BENCH``. Adding a model folder adds its bench, with no edit here.
    """
    found: Dict[str, Benchmark] = {}
    for path in sorted(BENCHMARKS_DIR.glob("models/*/bench.py")):
        module = importlib.import_module(f"benchmarks.models.{path.parent.name}.bench")
        found[path.parent.name] = module.BENCH
    return found


# ----------------------------------------------------------------------------------------------
# CLI: one flag per HParams field, generated rather than hand-written
# ----------------------------------------------------------------------------------------------


def add_hparam_arguments(parser: argparse.ArgumentParser) -> None:
    hints = get_type_hints(HParams)
    for f in fields(HParams):
        flag = "--" + f.name.replace("_", "-")
        annotation = hints[f.name]
        if annotation is bool:
            parser.add_argument(flag, dest=f.name, default=None,
                                action=argparse.BooleanOptionalAction)
        elif get_origin(annotation) is tuple:
            parser.add_argument(flag, dest=f.name, type=float,
                                nargs=len(get_args(annotation)), default=None)
        elif annotation is int or int in get_args(annotation):
            parser.add_argument(flag, dest=f.name, type=int, default=None)
        else:
            parser.add_argument(flag, dest=f.name, type=float, default=None)


def resolve_hparams(bench: Benchmark, args: argparse.Namespace) -> HParams:
    overrides = {f.name: getattr(args, f.name) for f in fields(HParams)
                 if getattr(args, f.name) is not None}
    if "gammas" in overrides:
        overrides["gammas"] = tuple(overrides["gammas"])
    return replace(bench.hparams, **overrides)


def build_parser(bench: Benchmark) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"benchmarks.models.{bench.name}.bench",
        description=f"{bench.title()} — {len(ARMS)} available arms under the WCT protocol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(bench.arms))
    parser.add_argument("--epochs", type=int, default=bench.epochs,
                        help="nominal epoch count; also the cosine schedule's T_max")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=bench.batch_size)
    parser.add_argument("--budget-mode", dest="budget_mode", choices=["wct", "epochs"],
                        default="wct")
    parser.add_argument("--reference-arm", dest="reference_arm", default="diag",
                        help="the arm whose measured elapsed time becomes every other arm's "
                             "budget (unrelated to the 'reference' arm, which is FisherAdapTune's "
                             "own AdaFisher)")
    parser.add_argument("--wct-budget", dest="wct_budget", type=float, default=None,
                        help="pre-measured reference elapsed time (s); skips running the "
                             "reference arm, so one arm can be one independent SLURM job")
    parser.add_argument("--lr-schedule", dest="lr_schedule", choices=["nominal", "budget"],
                        default="nominal",
                        help="what the cosine anneals over. 'nominal': --epochs, shared by every "
                             "arm, clamped at the floor past it. 'budget': each budgeted arm's own "
                             "wall-clock budget, so every arm completes one full cosine "
                             "(common/schedules.py). The reference arm is unbudgeted and always "
                             "uses 'nominal'.")
    parser.add_argument("--max-epoch-factor", dest="max_epoch_factor", type=float, default=3.0)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=10**9)
    add_hparam_arguments(parser)
    for keyword, values in bench.model_choices.items():
        parser.add_argument(f"--{keyword.replace('_', '-')}", dest=keyword,
                            choices=list(values), default=values[0])
    parser.add_argument("--cutout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-root", dest="data_root", type=str,
                        default=str(BENCHMARKS_DIR / "data"))
    parser.add_argument("--allow-download", dest="allow_download",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--train-subset", dest="train_subset", type=int, default=None)
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="comma-separated fractions of the run's steps at which to dump "
                             "theta, e.g. '0,0.01,0.1,0.5,1' (plan_exp_step1.md D5)")
    parser.add_argument("--checkpoint-optimizer-state", dest="checkpoint_optimizer_state",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="also dump the optimizer's own state in each checkpoint (its "
                             "state_dict, plus AdaFisherMulti's EMA'd Fisher factors keyed by "
                             "module name). Off by default: it is what protocol P2 of "
                             "plan_exp_draft.md would otherwise re-warm from theta, and it costs "
                             "~1.8 GB over a 6-model x 7-arm x 5-checkpoint campaign")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--no-plot", dest="no_plot", action="store_true", default=False)
    return parser


# ----------------------------------------------------------------------------------------------
# One arm
# ----------------------------------------------------------------------------------------------


def total_s_of(steps: Sequence[Any]) -> float:
    return float(steps[-1].elapsed_s) if steps else 0.0


def run_arm(
    bench: Benchmark, arm: str, args: argparse.Namespace, hp: HParams,
    device: torch.device, budget_s: float, output_dir: Path,
) -> ArmResult:
    torch.manual_seed(args.seed)  # identical init and data order across every arm of this model
    model = bench.build_model(**{k: getattr(args, k) for k in bench.model_choices}).to(device)
    optimizer = build_optimizer(arm, model, hp)

    train_loader, val_loader, test_loader = bench.build_data(
        args.data_root, batch_size=args.batch_size, seed=args.seed,
        num_workers=args.num_workers, cutout=args.cutout,
        allow_download=args.allow_download, train_subset=args.train_subset,
    )
    # AdaFisher Appendix D: "A cosine annealing learning rate decay strategy was employed, aligning
    # with the number of training epochs specified for each optimizer". Under the WCT protocol the
    # arms complete different epoch counts inside one budget, so that sentence has two readings and
    # --lr-schedule picks between them; common/schedules.py derives both, and documents the two
    # measured biases the first campaign's single shared T_max introduced.
    #
    # The reference arm is unbudgeted by construction — it is what *defines* the budget — so it
    # always anneals over the nominal epoch count, whichever mode is requested.
    scheduler: Any = None
    lr_hook: Optional[BudgetCosine] = None
    if args.lr_schedule == "budget" and budget_s != float("inf"):
        lr_hook = BudgetCosine(optimizer, budget_s=budget_s)
    else:
        # Clamped past T_max, unlike torch's CosineAnnealingLR, whose periodicity made every
        # overshooting arm's LR climb back towards base_lr (schedules.py's docstring). Identical to
        # CosineAnnealingLR for every epoch within T_max.
        scheduler = NominalCosine(optimizer, t_max=args.epochs)
    # --max-epoch-factor bounds how far a *budgeted* arm may run when the reference arm's budget
    # turns out generous. The reference arm itself is unbudgeted and must run exactly the nominal
    # epoch count — it is what defines the budget (AdaFisher §5's WCT protocol). Do not apply the
    # factor to it.
    max_epochs = (args.epochs if budget_s == float("inf")
                  else max(args.epochs, int(args.epochs * args.max_epoch_factor)))

    writer: Optional[CheckpointWriter] = None
    if args.checkpoints:
        # The checkpoint schedule is expressed against the **nominal** trajectory (``--epochs``),
        # never against ``max_epochs``. Under the WCT protocol every arm of a model shares one
        # nominal length, so ``ckpt_0.5`` means the same amount of training in every arm and the
        # fractions are comparable across them, which is the whole point of the dumps. Using
        # ``max_epochs`` instead put a budgeted arm's ``ckpt_0.1`` at ~31% of its own run and made
        # ``ckpt_0.5`` unreachable. Do not "simplify" this back.
        writer = CheckpointWriter(
            output_dir=output_dir,
            fractions=parse_fractions(args.checkpoints),
            total_steps=min(args.max_steps, args.epochs * len(train_loader)),
            seed=args.seed,
            optimizer=optimizer,
            save_optimizer_state=args.checkpoint_optimizer_state,
        )

    eval_fn = None
    if val_loader is not None:
        eval_fn = lambda: evaluate(  # noqa: E731 — a one-line closure over the loop's own state
            model, val_loader, bench.loss_fn, device,
            prepare_batch=bench.prepare_batch, metric_fn=bench.metric_fn,
        )

    budget_label = "unbounded" if budget_s == float("inf") else f"{budget_s:.1f}s"
    schedule_label = "budget-cosine" if lr_hook is not None else f"nominal-cosine/{args.epochs}ep"
    print(f"[{arm}] budget={budget_label} max_epochs={max_epochs} lr={schedule_label} "
          f"device={device}")
    reset_peak_memory(device)  # the arm's peak covers training *and* evaluation
    steps, epochs = train_under_budget(
        model, optimizer, train_loader, bench.loss_fn,
        budget_s=budget_s, max_epochs=max_epochs, device=device, scheduler=scheduler,
        eval_fn=eval_fn, on_step=writer, max_steps=args.max_steps,
        prepare_batch=bench.prepare_batch, log_fn=print, lr_schedule=lr_hook,
    )
    if writer is not None:
        writer.save_final(len(steps), epochs[-1].epoch if epochs else 0, model)

    # An arm capped by --max-epoch-factor stops part-way up its own cosine and is NOT annealed —
    # the very thing --lr-schedule budget exists to prevent. Only worth warning about when the cap
    # bound *early*: hitting it with the budget essentially spent is the benign case. No arm of the
    # first campaign came close (the cheapest, resnet20_cifar/adam, used 59 of its 150 allowed
    # epochs), so this guards against a mis-set budget rather than an expected condition.
    unspent = budget_s - total_s_of(steps)
    if lr_hook is not None and len(epochs) >= max_epochs and unspent > 0.01 * budget_s:
        print(f"[{arm}] WARNING: stopped at the --max-epoch-factor cap ({max_epochs} epochs) with "
              f"{unspent:.1f}s of {budget_s:.1f}s budget unspent, so its cosine did not reach the "
              f"floor (lr={float(optimizer.param_groups[0]['lr']):.3e}). Raise "
              f"--max-epoch-factor, or treat this arm's final metrics as under-annealed.")

    test_loss, test_acc = (float("nan"), float("nan"))
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, bench.loss_fn, device,
                                       prepare_batch=bench.prepare_batch,
                                       metric_fn=bench.metric_fn)
    total_s = total_s_of(steps)
    peak_vram = peak_memory_bytes(device)
    print(f"[{arm}] done: {len(steps)} steps, {len(epochs)} epochs, "
          f"test_acc={test_acc * 100:.2f}%, peak VRAM={format_bytes(peak_vram)}")
    result = ArmResult(arm, steps, epochs, test_acc, test_loss, total_s,
                       dict(writer.written) if writer is not None else {}, peak_vram)

    # Tear the arm down before returning, so the *next* arm resets its peak-memory counter on an
    # empty device. An AdaFisherMulti arm is a reference cycle — the optimizer holds each hooked
    # module, and each module's hook dictionary holds a bound method of the optimizer — which plain
    # reference counting cannot break, so without this every arm's model, optimizer and curvature
    # factors stay resident for the rest of the process. The symptom in the shipped reports is a
    # `peak VRAM` column that rises monotonically in the order the arms ran, with `adam` (which has
    # no factors at all) reported as the most expensive of the seven. Nothing here touches the
    # weights or the records: `result` is already built, so no number a run produced can change.
    # Every other reference to the model and the optimizer — the checkpoint writer, the eval
    # closure, the local names — is acyclic and dies with this frame; the cycle was the only thing
    # refcounting could not undo. ``test_no_arm_survives_run_arm`` holds weak references and
    # proves nothing is left, for all seven arms.
    release_optimizer(model, optimizer)
    return result


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------


def main(bench: Benchmark, argv: Sequence[str] | None = None) -> None:
    args = build_parser(bench).parse_args(argv)
    hp = resolve_hparams(bench, args)
    device = resolve_device(args.device)
    root = args.output_dir or bench.output_root(BENCHMARKS_DIR / "outputs")
    print(f"model={bench.name} arms={args.arms} device={device} hparams={hp}")

    results: List[ArmResult] = []
    budget = float("inf") if args.budget_mode == "epochs" else args.wct_budget

    if args.budget_mode == "wct" and budget is None:
        if args.reference_arm not in args.arms:
            raise ValueError(f"--budget-mode wct needs either --wct-budget or --reference-arm "
                             f"({args.reference_arm!r}) among --arms")
        reference = run_arm(bench, args.reference_arm, args, hp, device, float("inf"),
                            root / args.reference_arm)
        results.append(reference)
        budget = reference.total_s
        print(f"[wct] reference arm {args.reference_arm!r} took {budget:.1f}s — "
              f"that is every other arm's budget")

    def report() -> str:
        """Rewrite the whole report from the arms finished so far.

        Called after *every* arm, not once at the end: a grouped job's report used to be written
        only after the last arm, so a crash in a later arm lost every arm that had completed
        (it cost `resnet20_cifar_all` and `mlp_ln_mnist_all` two finished arms each — the cause is
        in `_eigh_utils.py`'s docstring). Rewriting is cheap next to training and idempotent.
        """
        root.mkdir(parents=True, exist_ok=True)
        write_step_csv(results, root / "records.csv")
        write_epoch_csv(results, root / "epochs.csv")
        text = write_summary(results, root / "summary.md")
        # ``model``/``output_group`` identify the *bench* that produced this run. They are not in
        # ``vars(args)`` and cannot be recovered from the directory name: ``--output-dir`` is
        # free-form, and A2's BatchNorm variant is deliberately written to
        # ``outputs/cnn_gn_cifar_bn/`` by ``--norm bn``. Without them ``fisher_ref/checkpoints.py``
        # cannot rebuild the network (it resolves ``benchmarks/<name>/bench.py`` by name).
        write_manifest(
            results,
            vars(args) | {"model": bench.name, "output_group": bench.output_group,
                          "hparams": asdict(hp)},
            root / "manifest.json",
        )
        return text

    if results:
        report()  # the reference arm, before the budgeted ones start
    for arm in args.arms:
        if any(r.arm == arm for r in results):
            continue
        results.append(run_arm(bench, arm, args, hp, device, budget, root / arm))
        report()

    summary = report()
    print("\n" + summary)
    if not args.no_plot and write_plots(results, root, bench.title()):
        print(f"plots written to {root}")
