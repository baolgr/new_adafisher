"""The single runner (``plan_exp_step1.md`` D3): arm loop, WCT protocol, seeding, reporting, CLI.

A model's ``bench.py`` is a ``Benchmark`` literal plus ``main(BENCH)`` — no control flow, no
argument parsing, no reporting of its own (target: 50 lines).

The wall-clock-time protocol is AdaFisher's own (``adafisher_2405.16397.pdf`` §5, p. 8: *"We employ
the Wall-Clock-Time (WCT) method with a cutoff of 200 epochs for AdaFisher's training"*): one
reference arm runs a fixed epoch count, every other arm gets that arm's measured wall-clock time
and runs as many epochs as fit. ``plan.md`` §6.3: an epoch-fixed comparison is "rigged in favour of
the expensive modes".
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
    # ``build_model`` as a keyword. A2's ``--norm {gn,bn}`` (plan_exp_step1.md §4) is its only
    # user; every other bench leaves it empty and ``build_model`` is called with no arguments.
    model_choices: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    def title(self) -> str:
        return self.display_name or self.name


def discover_benchmarks() -> Dict[str, Benchmark]:
    """The folder list *is* the registry (``plan_exp_step1.md`` §5): every ``benchmarks/<x>/bench.py``
    exposing a ``BENCH``.
    """
    found: Dict[str, Benchmark] = {}
    for path in sorted(BENCHMARKS_DIR.glob("*/bench.py")):
        module = importlib.import_module(f"benchmarks.{path.parent.name}.bench")
        found[path.parent.name] = module.BENCH
    return found


# ----------------------------------------------------------------------------------------------
# CLI: one flag per HParams field, generated (D4) rather than hand-written
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
        prog=f"benchmarks.{bench.name}.bench",
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--no-plot", dest="no_plot", action="store_true", default=False)
    return parser


# ----------------------------------------------------------------------------------------------
# One arm
# ----------------------------------------------------------------------------------------------


def run_arm(
    bench: Benchmark, arm: str, args: argparse.Namespace, hp: HParams,
    device: torch.device, budget_s: float, output_dir: Path,
) -> ArmResult:
    torch.manual_seed(args.seed)  # identical init + data order across arms (plan_lot8.md §0.12)
    model = bench.build_model(**{k: getattr(args, k) for k in bench.model_choices}).to(device)
    optimizer = build_optimizer(arm, model, hp)

    train_loader, val_loader, test_loader = bench.build_data(
        args.data_root, batch_size=args.batch_size, seed=args.seed,
        num_workers=args.num_workers, cutout=args.cutout,
        allow_download=args.allow_download, train_subset=args.train_subset,
    )
    # AdaFisher Appendix D: "A cosine annealing learning rate decay strategy was employed, aligning
    # with the number of training epochs specified for each optimizer" — T_max is the *nominal*
    # epoch count (--epochs), shared by every arm, so the schedule's shape is shared even though
    # the arms complete different numbers of epochs within the same wall-clock budget.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # --max-epoch-factor bounds how far a *budgeted* arm may run when the reference arm's budget
    # turns out generous (plan_lot8.md §0.7 step 3). The reference arm itself is unbudgeted and
    # must run exactly the nominal epoch count — it is what defines the budget (AdaFisher §5's WCT
    # protocol). Lot 8's `main` applied the factor to both, which only ever mattered for the
    # in-process reference arm, since its SLURM jobs run that arm as --budget-mode epochs.
    max_epochs = (args.epochs if budget_s == float("inf")
                  else max(args.epochs, int(args.epochs * args.max_epoch_factor)))

    writer: Optional[CheckpointWriter] = None
    if args.checkpoints:
        writer = CheckpointWriter(
            output_dir=output_dir,
            fractions=parse_fractions(args.checkpoints),
            total_steps=min(args.max_steps, max_epochs * len(train_loader)),
            seed=args.seed,
        )

    eval_fn = None
    if val_loader is not None:
        eval_fn = lambda: evaluate(  # noqa: E731 — a one-line closure over the loop's own state
            model, val_loader, bench.loss_fn, device,
            prepare_batch=bench.prepare_batch, metric_fn=bench.metric_fn,
        )

    budget_label = "unbounded" if budget_s == float("inf") else f"{budget_s:.1f}s"
    print(f"[{arm}] budget={budget_label} max_epochs={max_epochs} device={device}")
    reset_peak_memory(device)  # the arm's peak covers training *and* evaluation
    steps, epochs = train_under_budget(
        model, optimizer, train_loader, bench.loss_fn,
        budget_s=budget_s, max_epochs=max_epochs, device=device, scheduler=scheduler,
        eval_fn=eval_fn, on_step=writer, max_steps=args.max_steps,
        prepare_batch=bench.prepare_batch, log_fn=print,
    )
    if writer is not None:
        writer.save_final(len(steps), epochs[-1].epoch if epochs else 0, model)

    test_loss, test_acc = (float("nan"), float("nan"))
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, bench.loss_fn, device,
                                       prepare_batch=bench.prepare_batch,
                                       metric_fn=bench.metric_fn)
    total_s = steps[-1].elapsed_s if steps else 0.0
    peak_vram = peak_memory_bytes(device)
    print(f"[{arm}] done: {len(steps)} steps, {len(epochs)} epochs, "
          f"test_acc={test_acc * 100:.2f}%, peak VRAM={format_bytes(peak_vram)}")
    return ArmResult(arm, steps, epochs, test_acc, test_loss, total_s,
                     dict(writer.written) if writer is not None else {}, peak_vram)


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------


def main(bench: Benchmark, argv: Sequence[str] | None = None) -> None:
    args = build_parser(bench).parse_args(argv)
    hp = resolve_hparams(bench, args)
    device = resolve_device(args.device)
    root = args.output_dir or (BENCHMARKS_DIR / "outputs" / bench.name)
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

    for arm in args.arms:
        if any(r.arm == arm for r in results):
            continue
        results.append(run_arm(bench, arm, args, hp, device, budget, root / arm))

    root.mkdir(parents=True, exist_ok=True)
    write_step_csv(results, root / "records.csv")
    write_epoch_csv(results, root / "epochs.csv")
    summary = write_summary(results, root / "summary.md")
    write_manifest(results, vars(args) | {"hparams": asdict(hp)}, root / "manifest.json")
    print("\n" + summary)
    if not args.no_plot and write_plots(results, root, bench.title()):
        print(f"plots written to {root}")
