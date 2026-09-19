"""The record types every bench produces, and the four report writers that consume them.

Three dataclasses: :class:`StepRecord` (one optimizer step), :class:`EpochRecord` (one epoch) and
:class:`ArmResult` (one arm's whole run, plus its final test numbers, its total wall-clock time,
the checkpoints it wrote and its peak device memory).

Four writers, all taking a sequence of :class:`ArmResult` and a path:

* ``write_step_csv`` -> ``records.csv``: ``arm, step, epoch, elapsed_s, loss, fwd_bwd_s, step_s,
  data_s``. New columns are appended on the right, never inserted, so a reader that looks columns
  up by name keeps working unchanged.
* ``write_epoch_csv`` -> ``epochs.csv``: ``arm, epoch, steps, elapsed_s, train_loss, val_loss,
  val_acc, lr``.
* ``write_summary`` -> ``summary.md``: one Markdown row per arm — steps, epochs, the mean loss over
  the last 50 steps, best validation accuracy, test accuracy, the median forward+backward and
  optimizer-step times, their ratio, total wall-clock time, peak device memory and the *compute
  share*. The timing medians skip each run's first few steps, which carry one-time hook and
  lazy-initialization cost.

  The compute share is ``sum(fwd_bwd_s + step_s) / total_s``: how much of an arm's wall-clock
  budget was spent optimizing rather than fetching and copying batches. Under the wall-clock-time
  protocol every arm of a model gets the same budget, but the non-compute cost is paid once per
  *epoch* and the arms deliberately complete different epoch counts — so the cheap arms pay it
  most often. Measured on ``mnist_autoencoder``: 47.4% for ``adam`` against 78.5% for ``ekfac``,
  i.e. ``ekfac`` received 1.67x the optimization time inside an "equal" budget. On
  ``resnet50_cifar`` every arm is at 99.5-99.6% and the effect is negligible. The column does not
  change the protocol; it makes the bias readable off each run.
* ``write_manifest`` -> ``manifest.json``: the full run configuration plus the headline numbers,
  which is what lets another tool rebuild the network a checkpoint belongs to.

``write_plots`` additionally writes loss and validation curves with matplotlib, and returns
``False`` without failing when matplotlib is not installed — plotting is an optional dependency.

A reconstruction task has no accuracy: its bench passes ``metric_fn=None`` and the accuracy columns
are ``nan``. The schema is the same either way, so one reader handles every bench.
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
    # Time from the end of the previous optimizer step to the start of this step's forward pass:
    # the batch fetch, the host-to-device copy, and whatever ``on_step`` did after the previous
    # step. Charged to the wall-clock budget like everything else, and invisible until now.
    # Defaulted, so a record built without it (an older caller, a test fixture) still constructs.
    data_s: float = 0.0


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
        # ``data_s`` is appended on the right: a reader that resolves columns by name (the only
        # kind in this repository) is unaffected, and one that reads by position still finds the
        # seven columns it knew where it expected them.
        writer.writerow(
            ["arm", "step", "epoch", "elapsed_s", "loss", "fwd_bwd_s", "step_s", "data_s"]
        )
        for result in results:
            for r in result.steps:
                writer.writerow(
                    [result.arm, r.step, r.epoch, r.elapsed_s, r.loss, r.fwd_bwd_s, r.step_s,
                     r.data_s]
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
    """One Markdown row per arm: steps, epochs, the mean loss over the last 50 steps, best
    validation accuracy, test accuracy, the median forward+backward and optimizer-step times, their
    ratio, total wall-clock time, peak device memory and the compute share. The timing medians skip
    each run's first ``skip_first`` steps, which carry one-time hook and lazy-initialization cost.

    ``compute share`` is ``sum(fwd_bwd_s + step_s) / total_s`` over *all* of the arm's steps: the
    fraction of its wall-clock budget spent optimizing rather than fetching and copying batches.
    Nothing skipped there — the question is how the whole budget was spent, first step included.
    """
    lines = [
        "| arm | steps | epochs | final train loss | best val acc | test acc | "
        "median fwd+bwd | median step | step/fwd+bwd | total wall-clock | peak VRAM | "
        "compute share |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        if not result.steps:
            lines.append(f"| {result.arm} | 0 | - | - | - | - | - | - | - | - | - | - |")
            continue
        steady = result.steps[skip_first:] if len(result.steps) > skip_first else result.steps
        fwd_bwd = median([r.fwd_bwd_s for r in steady])
        step = median([r.step_s for r in steady])
        ratio = step / fwd_bwd if fwd_bwd > 0 else float("nan")
        final_loss = sum(r.loss for r in result.steps[-50:]) / min(50, len(result.steps))
        best_val = max((e.val_acc for e in result.epochs), default=float("nan"))
        compute_s = sum(r.fwd_bwd_s + r.step_s for r in result.steps)
        share = compute_s / result.total_s if result.total_s > 0 else float("nan")
        lines.append(
            f"| {result.arm} | {len(result.steps)} | {len(result.epochs)} | {final_loss:.4f} | "
            f"{best_val * 100:.2f}% | {result.test_acc * 100:.2f}% | {fwd_bwd * 1e3:.2f} ms | "
            f"{step * 1e3:.2f} ms | {ratio:.2f}x | {result.total_s:.1f}s | "
            f"{format_bytes(result.peak_vram_bytes)} | {share * 100:.1f}% |"
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
    dependency.
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
