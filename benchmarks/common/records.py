"""Per-step / per-epoch records and the report writers (``plan_exp_step1.md`` §5).

Merge of ``equal_wallclock_bench.{StepRecord,_median,write_csv,write_summary,write_plots}`` and
``cifar10_classification.{EpochRecord,ArmResult,write_*}``: the lot-7 names lose their underscore
and the writers keep lot 8's column schema verbatim, so a CSV produced here is column-identical to
``benchmarks/outputs/lot8_cifar10/*``'s.

A reconstruction task (``metric_fn=None``) leaves the accuracy columns at ``nan``; the schema is
the same either way, so one reader handles every bench.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class StepRecord:
    step: int
    epoch: float
    elapsed_s: float
    loss: float
    fwd_bwd_s: float
    step_s: float


@dataclass
class EpochRecord:
    epoch: int
    steps: int
    elapsed_s: float
    train_loss: float
    val_loss: float
    val_acc: float
    lr: float


@dataclass
class ArmResult:
    arm: str
    steps: List[StepRecord]
    epochs: List[EpochRecord]
    test_acc: float
    test_loss: float
    total_s: float
    checkpoints: Dict[str, str] = field(default_factory=dict)
    # Peak *device* memory over the whole arm (training and evaluation), in bytes. ``None`` on
    # any backend where PyTorch exposes no peak-memory counter — CPU, and MPS, which has
    # ``current_allocated_memory`` but no high-water mark.
    peak_vram_bytes: Optional[int] = None


def format_bytes(value: Optional[int]) -> str:
    """``None`` -> ``-`` (no peak counter on this backend), otherwise MiB or GiB."""
    if value is None:
        return "-"
    mib = value / 1024**2
    return f"{mib / 1024:.2f} GiB" if mib >= 1024 else f"{mib:.0f} MiB"


def median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def write_step_csv(results: Sequence[ArmResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["arm", "step", "epoch", "elapsed_s", "loss", "fwd_bwd_s", "step_s"])
        for result in results:
            for r in result.steps:
                writer.writerow(
                    [result.arm, r.step, r.epoch, r.elapsed_s, r.loss, r.fwd_bwd_s, r.step_s]
                )


def write_epoch_csv(results: Sequence[ArmResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["arm", "epoch", "steps", "elapsed_s", "train_loss", "val_loss", "val_acc", "lr"]
        )
        for result in results:
            for e in result.epochs:
                writer.writerow(
                    [result.arm, e.epoch, e.steps, e.elapsed_s, e.train_loss, e.val_loss,
                     e.val_acc, e.lr]
                )


def write_summary(results: Sequence[ArmResult], path: Path, skip_first: int = 5) -> str:
    """Lot 7's timing table (steps, epochs, final loss, median fwd+bwd, median step, their ratio)
    plus lot 8's classification columns. Timing medians skip each run's first ``skip_first`` steps,
    excluding one-time hook / lazy-initialization cost.
    """
    lines = [
        "| arm | steps | epochs | final train loss | best val acc | test acc | "
        "median fwd+bwd | median step | step/fwd+bwd | total wall-clock | peak VRAM |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        if not result.steps:
            lines.append(f"| {result.arm} | 0 | - | - | - | - | - | - | - | - | - |")
            continue
        steady = result.steps[skip_first:] if len(result.steps) > skip_first else result.steps
        fwd_bwd = median([r.fwd_bwd_s for r in steady])
        step = median([r.step_s for r in steady])
        ratio = step / fwd_bwd if fwd_bwd > 0 else float("nan")
        final_loss = sum(r.loss for r in result.steps[-50:]) / min(50, len(result.steps))
        best_val = max((e.val_acc for e in result.epochs), default=float("nan"))
        lines.append(
            f"| {result.arm} | {len(result.steps)} | {len(result.epochs)} | {final_loss:.4f} | "
            f"{best_val * 100:.2f}% | {result.test_acc * 100:.2f}% | {fwd_bwd * 1e3:.2f} ms | "
            f"{step * 1e3:.2f} ms | {ratio:.2f}x | {result.total_s:.1f}s | "
            f"{format_bytes(result.peak_vram_bytes)} |"
        )
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def write_plots(
    results: Sequence[ArmResult], output_dir: Path, title: str, window: int = 50
) -> bool:
    """Training loss against epoch and wall-clock time; validation loss against epoch; test vs.
    validation per arm (error when the task has an accuracy, loss otherwise); plus, when the task
    has an accuracy, validation error against both epoch and wall-clock time. Returns ``False``
    (skipping plots, never failing) if matplotlib is missing: plotting is an optional bench
    dependency (``plan_lot7.md`` §0.9).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots (CSV/summary still written)",
              file=sys.stderr)
        return False

    def smoothed(values: List[float]) -> List[float]:
        out = []
        for i in range(len(values)):
            lo = max(0, i - window + 1)
            out.append(sum(values[lo : i + 1]) / (i - lo + 1))
        return out

    output_dir.mkdir(parents=True, exist_ok=True)
    for x_attr, xlabel, filename in [("epoch", "epoch", "loss_vs_epoch.png"),
                                     ("elapsed_s", "wall-clock time (s)", "loss_vs_time.png")]:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for result in results:
            if not result.steps:
                continue
            ax.plot([getattr(r, x_attr) for r in result.steps],
                    smoothed([r.loss for r in result.steps]), label=result.arm)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"training loss ({window}-step moving average)")
        ax.set_title(f"{title}: training loss vs. {xlabel}", fontsize=11)
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for result in results:
        if not result.epochs:
            continue
        ax.plot([e.epoch for e in result.epochs], [e.val_loss for e in result.epochs],
                marker="o", markersize=3, label=result.arm)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss")
    ax.set_title(f"{title}: validation loss vs. epoch", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "valloss_vs_epoch.png", dpi=150)
    plt.close(fig)

    has_accuracy = any(e.val_acc == e.val_acc for r in results for e in r.epochs)
    if has_accuracy:
        for x_attr, xlabel, filename in [("epoch", "epoch", "valerr_vs_epoch.png"),
                                         ("elapsed_s", "wall-clock time (s)", "valerr_vs_time.png")]:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            for result in results:
                if not result.epochs:
                    continue
                ax.plot([getattr(e, x_attr) for e in result.epochs],
                        [100.0 * (1.0 - e.val_acc) for e in result.epochs],
                        marker="o", markersize=3, label=result.arm)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("validation error (%)")
            ax.set_title(f"{title}: validation error vs. {xlabel}", fontsize=11)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / filename, dpi=150)
            plt.close(fig)

    arms_with_epochs = [r for r in results if r.epochs]
    if arms_with_epochs:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        x = range(len(arms_with_epochs))
        width = 0.35
        if has_accuracy:
            val_vals = [100.0 * (1.0 - max((e.val_acc for e in r.epochs), default=float("nan")))
                        for r in arms_with_epochs]
            test_vals = [100.0 * (1.0 - r.test_acc) for r in arms_with_epochs]
            ylabel, metric, fmt = "error (%)", "error", "%.1f"
        else:
            val_vals = [r.epochs[-1].val_loss for r in arms_with_epochs]
            test_vals = [r.test_loss for r in arms_with_epochs]
            ylabel, metric, fmt = "loss", "loss", "%.3f"
        bars_val = ax.bar([i - width / 2 for i in x], val_vals, width, label="validation")
        bars_test = ax.bar([i + width / 2 for i in x], test_vals, width, label="test")
        for bars in (bars_val, bars_test):
            ax.bar_label(bars, fmt=fmt, fontsize=7, padding=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels([r.arm for r in arms_with_epochs])
        ax.set_xlabel("arm")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: test vs. validation {metric}", fontsize=11)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "test_vs_val.png", dpi=150)
        plt.close(fig)

    return True


def write_manifest(results: Sequence[ArmResult], config: Dict[str, Any], path: Path) -> None:
    """Everything needed to re-run this exactly, plus the headline numbers, in one JSON file."""
    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()},
        "arms": {
            r.arm: {
                "steps": len(r.steps),
                "epochs": len(r.epochs),
                "total_s": r.total_s,
                "test_acc": r.test_acc,
                "test_loss": r.test_loss,
                "best_val_acc": max((e.val_acc for e in r.epochs), default=None),
                "final_epoch": asdict(r.epochs[-1]) if r.epochs else None,
                "checkpoints": r.checkpoints,
                "peak_vram_bytes": r.peak_vram_bytes,
            }
            for r in results
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
