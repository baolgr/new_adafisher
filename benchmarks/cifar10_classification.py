"""CIFAR-10 from-scratch classification bench, ResNet-50 / ViT-S/4, five Fisher approximation
modes + Adam + AdamW (docs/reports/plan.md §8's deferred row, §6.3; design in
docs/reports/plan_lot8.md).

Seven arms per model — ``diag``, ``kfac``, ``ekfac``, ``tkfac``, ``tekfac`` (all ``AdaFisherMulti``),
``adam``, ``adamw`` — compared under AdaFisher's own **wall-clock-time (WCT) protocol**
(``adafisher_2405.16397.pdf`` §5, p. 8: *"We employ the Wall-Clock-Time (WCT) method with a cutoff
of 200 epochs for AdaFisher's training"*): one reference arm runs a fixed number of epochs, every
other arm gets that arm's measured wall-clock time and runs as many epochs as fit. This is the
classification-task form of ``plan.md`` §6.3's requirement that the budget be fixed in wall-clock
time, "otherwise the comparison is rigged in favour of the expensive modes".

Outputs mirror the MNIST bench's (``benchmarks/equal_wallclock_bench.py``), whose ``StepRecord``,
``_sync``, ``_median`` and ``write_csv`` are imported rather than duplicated, plus what a
classification run has that a reconstruction run does not: per-epoch validation accuracy, a
one-shot test accuracy, and validation-error-vs-{epoch, wall-clock} plots — the AdaFisher paper's
own figure set (Figs. 16-18: *"WCT training loss, test error, for CNNs and ViTs on CIFAR10"*).

Usage (from the repository root, with the project venv):
    # local smoke, seconds
    PYTHONPATH=src .venv/bin/python benchmarks/cifar10_classification.py \
        --model resnet50 --arms diag adam --epochs 1 --train-subset 512 --budget-mode epochs
    # one reference arm (its elapsed time becomes the WCT budget for the others)
    PYTHONPATH=src .venv/bin/python benchmarks/cifar10_classification.py \
        --model resnet50 --arms diag --epochs 50 --budget-mode epochs
    # one budgeted arm, e.g. as one SLURM job
    PYTHONPATH=src .venv/bin/python benchmarks/cifar10_classification.py \
        --model resnet50 --arms ekfac --epochs 50 --budget-mode wct --wct-budget 4321.0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adafisher_modes import AdaFisherMulti  # noqa: E402
from cifar10_data import build_dataloaders  # noqa: E402
from cifar10_models import MODELS  # noqa: E402
from equal_wallclock_bench import StepRecord, _median, _sync  # noqa: E402

FISHER_ARMS = ("diag", "kfac", "ekfac", "tkfac", "tekfac")
BASELINE_ARMS = ("adam", "adamw")
ALL_ARMS = FISHER_ARMS + BASELINE_ARMS

# Per-model hyperparameters (plan_lot8.md §0.8): AdaFisher's own tuned operating point, from its
# Appendix D "HP Tuning" / Table 9 and from the official repository's shipped configs
# (reference_repos/AdaFisher/Image_Classification/configs/AdaFisher{CNN,ViT}.yaml,
# adam{CNN,ViT}.yaml). Not re-tuned here — per-arm tuning is out of scope (plan_lot8.md §4).
MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "resnet50": {
        "lr": 1e-3,  # Table 9, CNNs / AdaFisher
        "baseline_lr": 1e-3,  # Table 9, CNNs / Adam
        "weight_decay": 5e-4,  # Appendix D: "uniform weight decay of 5e-4 ... for CIFAR-10"
        "lam": 1e-3,  # AdaFisherCNN.yaml: Lambda
        "decoupled_wd": False,  # "Adam and AdaFisher were used for all CNN architectures"
        "conv_sua": True,  # mandatory at this scale — plan_lot8.md §0.5
        "fisher_batch_samples": 32,  # plan_lot8.md §0.4
    },
    "vit_small": {
        "lr": 1e-3,  # Table 9, ViTs / AdaFisherW
        "baseline_lr": 1e-4,  # Table 9, ViTs / AdamW ("adopted from the original publications")
        "weight_decay": 1e-2,  # AdaFisherViT.yaml: weight_decay
        "lam": 3e-3,  # AdaFisherViT.yaml: Lambda
        "decoupled_wd": True,  # "AdamW and AdaFisherW were applied for all ViT experiments"
        "conv_sua": False,  # one 4x4 patch-embedding conv; SUA would change nothing
        "fisher_batch_samples": None,
    },
}


# ----------------------------------------------------------------------------------------------
# Model- and dataset-agnostic training harness (plan_lot8.md §0.7, §1.4)
# ----------------------------------------------------------------------------------------------


@dataclass
class EpochRecord:
    epoch: int
    steps: int
    elapsed_s: float
    train_loss: float
    val_loss: float
    val_acc: float
    lr: float


def train_classifier_under_budget(
    model: nn.Module,
    optimizer: Any,
    train_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    budget_s: float,
    max_epochs: int,
    device: torch.device = torch.device("cpu"),
    scheduler: Any = None,
    eval_fn: Optional[Callable[[], Tuple[float, float]]] = None,
    max_steps: int = 10**9,
    log_fn: Callable[[str], None] = print,
) -> Tuple[List[StepRecord], List[EpochRecord]]:
    """Train until ``budget_s`` of wall-clock time is spent or ``max_epochs`` epochs complete.

    Budget semantics are lot 7's, unchanged (``plan_lot7.md`` §0.1): the check happens **before**
    each batch's processing, after that batch has been fetched, so a run overshoots ``budget_s`` by
    at most one batch's own processing duration and never by more.

    One addition over lot 7's ``train_under_time_budget``: an optional ``eval_fn``, run at every
    epoch boundary, whose duration is **subtracted from the clock** (plan_lot8.md §0.7). Validation
    is not part of the optimizer's cost, and charging it to the budget would penalize an arm purely
    for completing more epochs.

    Model- and dataset-agnostic on purpose, so ``tests/test_cifar10_bench.py`` can drive it on a
    tiny synthetic task with no CIFAR-10 download (the discipline of ``plan_lot7.md`` §0.8).
    """
    step_records: List[StepRecord] = []
    epoch_records: List[EpochRecord] = []
    batches_per_epoch = len(train_loader)
    if batches_per_epoch == 0:
        raise ValueError("train_loader must yield at least one batch")

    step = 0
    excluded_s = 0.0
    t0 = perf_counter()

    def elapsed() -> float:
        return perf_counter() - t0 - excluded_s

    for epoch in range(max_epochs):
        model.train()
        epoch_loss, epoch_batches, completed = 0.0, 0, True
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if elapsed() >= budget_s or step >= max_steps:
                completed = False
                break
            inputs, targets = inputs.to(device), targets.to(device)

            t_fb = perf_counter()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            _sync(device)
            fwd_bwd_s = perf_counter() - t_fb

            t_step = perf_counter()
            optimizer.step()
            _sync(device)
            step_s = perf_counter() - t_step

            loss_value = float(loss.item())
            epoch_loss += loss_value
            epoch_batches += 1
            step_records.append(
                StepRecord(
                    step=step,
                    epoch=epoch + batch_idx / batches_per_epoch,
                    elapsed_s=elapsed(),
                    loss=loss_value,
                    fwd_bwd_s=fwd_bwd_s,
                    step_s=step_s,
                )
            )
            step += 1

        if completed and scheduler is not None:
            scheduler.step()

        val_loss, val_acc = float("nan"), float("nan")
        if eval_fn is not None and epoch_batches > 0:
            t_eval = perf_counter()
            val_loss, val_acc = eval_fn()
            excluded_s += perf_counter() - t_eval

        if epoch_batches > 0:
            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_records.append(
                EpochRecord(
                    epoch=epoch,
                    steps=step,
                    elapsed_s=elapsed(),
                    train_loss=epoch_loss / epoch_batches,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    lr=current_lr,
                )
            )
            log_fn(
                f"  epoch {epoch:3d} | steps {step:6d} | {elapsed():8.1f}s | "
                f"train {epoch_loss / epoch_batches:.4f} | val_acc {val_acc * 100:5.2f}% | "
                f"lr {current_lr:.2e}"
            )

        if not completed or elapsed() >= budget_s or step >= max_steps:
            break

    return step_records, epoch_records


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
) -> Tuple[float, float]:
    """``(mean loss, top-1 accuracy)`` over ``loader``."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        total_loss += float(loss_fn(outputs, targets).item()) * targets.size(0)
        correct += int((outputs.argmax(dim=1) == targets).sum().item())
        seen += targets.size(0)
    model.train()
    if seen == 0:
        return float("nan"), float("nan")
    return total_loss / seen, correct / seen


# ----------------------------------------------------------------------------------------------
# Arms (plan_lot8.md §0.12)
# ----------------------------------------------------------------------------------------------


def make_optimizer(arm: str, model: nn.Module, args: argparse.Namespace) -> Any:
    """The seven arms of plan_lot8.md §0.12, at §0.8's per-model hyperparameters.

    The two baselines take ``--baseline-lr`` (Table 9 gives Adam/AdamW a different tuned learning
    rate from AdaFisher's on ViTs); ``weight_decay`` is shared, its *convention* being what differs
    (``Adam``/``AdaFisher`` couple it into the gradient, ``AdamW``/``AdaFisherW`` decouple it —
    plan_lot8.md §0.6).
    """
    if arm == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=args.baseline_lr, weight_decay=args.weight_decay
        )
    if arm == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=args.baseline_lr, weight_decay=args.weight_decay
        )
    if arm not in FISHER_ARMS:
        raise ValueError(f"Unknown arm {arm!r}; available: {list(ALL_ARMS)}")

    kwargs: Dict[str, Any] = dict(
        lr=args.lr,
        beta=args.beta,
        Lambda=args.lam,
        gammas=list(args.gammas),
        TCov=args.tcov,
        weight_decay=args.weight_decay,
        fisher_mode=arm,
        fisher_batch_samples=args.fisher_batch_samples,
        decoupled_weight_decay=args.decoupled_wd,
    )
    if arm == "diag":
        # Min-max ON: the paper-faithful default an actual user gets (plan_lot7.md §0.10), not
        # tests/test_diag_bitexact.py's bit-exactness-check configuration.
        kwargs["minmax_normalization"] = args.minmax_normalization
    else:
        kwargs["conv_sua"] = args.conv_sua
        if arm in ("kfac", "tkfac"):
            kwargs["T_inv"] = args.t_inv
        if arm in ("ekfac", "tekfac"):
            kwargs["T_eig"] = args.t_eig
        if arm == "tekfac":
            kwargs["T_re"] = args.t_re
    return AdaFisherMulti(model, **kwargs)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class ArmResult:
    arm: str
    steps: List[StepRecord]
    epochs: List[EpochRecord]
    test_acc: float
    test_loss: float
    total_s: float


def run_arm(arm: str, args: argparse.Namespace, device: torch.device, budget_s: float) -> ArmResult:
    torch.manual_seed(args.seed)  # identical init + data order across arms (plan_lot8.md §0.12)
    model = MODELS[args.model].build(args.num_classes).to(device)
    optimizer = make_optimizer(arm, model, args)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        cutout=args.cutout,
        allow_download=args.allow_download,
        train_subset=args.train_subset,
    )
    # Appendix D: "A cosine annealing learning rate decay strategy was employed, aligning with the
    # number of training epochs specified for each optimizer" — T_max is the *nominal* epoch count
    # (--epochs), the same for every arm, so the schedule shape is shared even though the arms
    # complete different numbers of epochs within the shared wall-clock budget.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    max_epochs = (
        args.epochs
        if args.budget_mode == "epochs"
        else max(args.epochs, int(args.epochs * args.max_epoch_factor))
    )

    budget_label = "unbounded" if budget_s == float("inf") else f"{budget_s:.1f}s"
    print(f"[{arm}] budget={budget_label} max_epochs={max_epochs} device={device}")
    steps, epochs = train_classifier_under_budget(
        model,
        optimizer,
        train_loader,
        loss_fn,
        budget_s=budget_s,
        max_epochs=max_epochs,
        device=device,
        scheduler=scheduler,
        eval_fn=lambda: evaluate(model, val_loader, loss_fn, device),
        log_fn=print,
    )
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    total_s = steps[-1].elapsed_s if steps else 0.0
    print(f"[{arm}] done: {len(steps)} steps, {len(epochs)} epochs, test_acc={test_acc * 100:.2f}%")
    return ArmResult(arm, steps, epochs, test_acc, test_loss, total_s)


# ----------------------------------------------------------------------------------------------
# Reporting (plan_lot8.md §0.10)
# ----------------------------------------------------------------------------------------------


def write_step_csv(results: List[ArmResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "step", "epoch", "elapsed_s", "loss", "fwd_bwd_s", "step_s"])
        for result in results:
            for r in result.steps:
                writer.writerow(
                    [result.arm, r.step, r.epoch, r.elapsed_s, r.loss, r.fwd_bwd_s, r.step_s]
                )


def write_epoch_csv(results: List[ArmResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["arm", "epoch", "steps", "elapsed_s", "train_loss", "val_loss", "val_acc", "lr"]
        )
        for result in results:
            for e in result.epochs:
                writer.writerow(
                    [
                        result.arm,
                        e.epoch,
                        e.steps,
                        e.elapsed_s,
                        e.train_loss,
                        e.val_loss,
                        e.val_acc,
                        e.lr,
                    ]
                )


def write_summary(results: List[ArmResult], path: Path, skip_first: int = 5) -> str:
    """Lot 7's ``summary.md`` table (steps, epochs, final loss, median fwd+bwd, median step,
    step/fwd+bwd ratio) plus the classification columns. Timing medians skip each run's first
    ``skip_first`` steps, excluding one-time hook/lazy-init cost.
    """
    lines = [
        "| arm | steps | epochs | final train loss | best val acc | test acc | "
        "median fwd+bwd | median step | step/fwd+bwd | total wall-clock |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        if not result.steps:
            lines.append(f"| {result.arm} | 0 | - | - | - | - | - | - | - | - |")
            continue
        steady = result.steps[skip_first:] if len(result.steps) > skip_first else result.steps
        fwd_bwd = _median([r.fwd_bwd_s for r in steady])
        step = _median([r.step_s for r in steady])
        ratio = step / fwd_bwd if fwd_bwd > 0 else float("nan")
        final_loss = sum(r.loss for r in result.steps[-50:]) / min(50, len(result.steps))
        best_val = max((e.val_acc for e in result.epochs), default=float("nan"))
        lines.append(
            f"| {result.arm} | {len(result.steps)} | {len(result.epochs)} | {final_loss:.4f} | "
            f"{best_val * 100:.2f}% | {result.test_acc * 100:.2f}% | {fwd_bwd * 1e3:.2f} ms | "
            f"{step * 1e3:.2f} ms | {ratio:.2f}x | {result.total_s:.1f}s |"
        )
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def write_plots(results: List[ArmResult], output_dir: Path, model_name: str, window: int = 50) -> bool:
    """Four figures: training loss and validation error, each against both epoch and wall-clock
    time. Returns False (skipping plots, never failing) if matplotlib is missing — plotting is an
    optional bench dependency, as in lot 7 (plan_lot7.md §0.9).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots (CSV/summary still written)", file=sys.stderr)
        return False

    def smoothed(values: List[float]) -> List[float]:
        out = []
        for i in range(len(values)):
            lo = max(0, i - window + 1)
            out.append(sum(values[lo : i + 1]) / (i - lo + 1))
        return out

    for x_attr, xlabel, filename in [("epoch", "epoch", "loss_vs_epoch.png"),
                                     ("elapsed_s", "wall-clock time (s)", "loss_vs_time.png")]:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for result in results:
            if not result.steps:
                continue
            ax.plot(
                [getattr(r, x_attr) for r in result.steps],
                smoothed([r.loss for r in result.steps]),
                label=result.arm,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"training loss ({window}-step moving average)")
        ax.set_title(f"CIFAR-10, {model_name}: training loss vs. {xlabel}", fontsize=11)
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    for x_attr, xlabel, filename in [("epoch", "epoch", "valerr_vs_epoch.png"),
                                     ("elapsed_s", "wall-clock time (s)", "valerr_vs_time.png")]:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for result in results:
            if not result.epochs:
                continue
            ax.plot(
                [getattr(e, x_attr) for e in result.epochs],
                [100.0 * (1.0 - e.val_acc) for e in result.epochs],
                marker="o",
                markersize=3,
                label=result.arm,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("validation error (%)")
        ax.set_title(f"CIFAR-10, {model_name}: validation error vs. {xlabel}", fontsize=11)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
    return True


def write_manifest(results: List[ArmResult], args: argparse.Namespace, path: Path) -> None:
    """Everything needed to re-run this exactly, plus the headline numbers, in one JSON file."""
    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "arms": {
            r.arm: {
                "steps": len(r.steps),
                "epochs": len(r.epochs),
                "total_s": r.total_s,
                "test_acc": r.test_acc,
                "test_loss": r.test_loss,
                "best_val_acc": max((e.val_acc for e in r.epochs), default=None),
                "final_epoch": asdict(r.epochs[-1]) if r.epochs else None,
            }
            for r in results
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=sorted(MODELS), default="resnet50")
    parser.add_argument("--arms", nargs="+", choices=list(ALL_ARMS), default=list(ALL_ARMS))
    parser.add_argument("--epochs", type=int, default=50, help="nominal epoch count; also T_max")
    parser.add_argument("--budget-mode", dest="budget_mode", choices=["wct", "epochs"], default="wct")
    parser.add_argument("--reference-arm", dest="reference_arm", default="diag")
    parser.add_argument("--wct-budget", dest="wct_budget", type=float, default=None,
                        help="pre-measured reference elapsed time (s); skips running the reference "
                             "arm, so one arm can be one independent SLURM job")
    parser.add_argument("--max-epoch-factor", dest="max_epoch_factor", type=float, default=3.0)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=10)
    # None => resolved from MODEL_DEFAULTS[--model] (plan_lot8.md §0.8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--baseline-lr", dest="baseline_lr", type=float, default=None)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--fisher-batch-samples", dest="fisher_batch_samples", type=int, default=None)
    parser.add_argument("--conv-sua", dest="conv_sua", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--decoupled-wd", dest="decoupled_wd", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--tcov", type=int, default=100)
    parser.add_argument("--gammas", type=float, nargs=2, default=(0.92, 0.008))
    parser.add_argument("--t-inv", dest="t_inv", type=int, default=100)
    parser.add_argument("--t-eig", dest="t_eig", type=int, default=100)
    parser.add_argument("--t-re", dest="t_re", type=int, default=1)
    parser.add_argument("--minmax-normalization", dest="minmax_normalization",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cutout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-root", dest="data_root", type=str,
                        default=str(REPO_ROOT / "benchmarks/data"))
    parser.add_argument("--allow-download", dest="allow_download",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--train-subset", dest="train_subset", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--no-plot", dest="no_plot", action="store_true", default=False)
    return parser


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    for key, value in MODEL_DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / f"benchmarks/outputs/lot8_cifar10/{args.model}"
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = resolve_defaults(build_parser().parse_args(argv))
    device = resolve_device(args.device)
    print(f"model={args.model} arms={args.arms} device={device} "
          f"lr={args.lr} baseline_lr={args.baseline_lr} wd={args.weight_decay} "
          f"decoupled_wd={args.decoupled_wd} conv_sua={args.conv_sua} "
          f"fisher_batch_samples={args.fisher_batch_samples}")

    results: List[ArmResult] = []
    budget = float("inf") if args.budget_mode == "epochs" else args.wct_budget

    if args.budget_mode == "wct" and budget is None:
        # No pre-measured budget: run the reference arm first, at a fixed epoch count, and adopt
        # its elapsed time as every other arm's budget (plan_lot8.md §0.7, AdaFisher §5's WCT).
        if args.reference_arm not in args.arms:
            raise ValueError(
                f"--budget-mode wct needs either --wct-budget or --reference-arm "
                f"({args.reference_arm!r}) among --arms"
            )
        reference = run_arm(args.reference_arm, args, device, budget_s=float("inf"))
        results.append(reference)
        budget = reference.total_s
        print(f"[wct] reference arm {args.reference_arm!r} took {budget:.1f}s — "
              f"that is every other arm's budget")

    for arm in args.arms:
        if any(r.arm == arm for r in results):
            continue
        results.append(run_arm(arm, args, device, budget_s=budget))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_step_csv(results, args.output_dir / "records.csv")
    write_epoch_csv(results, args.output_dir / "epochs.csv")
    summary = write_summary(results, args.output_dir / "summary.md")
    write_manifest(results, args, args.output_dir / "manifest.json")
    print("\n" + summary)
    if not args.no_plot and write_plots(results, args.output_dir, MODELS[args.model].display_name):
        print(f"plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
