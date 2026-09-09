"""Equal-wall-clock-budget convergence bench across all five Fisher approximation modes
(docs/reports/plan.md §6.3, §8 lot 7; design rationale in docs/reports/plan_lot7.md).

Trains ``diag``, ``kfac``, ``ekfac``, ``tkfac`` and ``tekfac`` on the 8-layer MNIST auto-encoder
(``benchmarks/mnist_autoencoder.py``), each under an identical wall-clock time budget rather than an
identical epoch count — ``plan.md`` §6.3: "the equal-budget requirement ... must therefore be fixed
in wall-clock time, otherwise the comparison is rigged in favour of the expensive modes." Reports
training loss as a function of both epoch and wall-clock time (``plan_lot7.md`` §0.3), and the
measured per-step "projection overhead" ratio ``step_s / fwd_bwd_s`` for every mode — an empirical
counterpart to ``plan.md`` §6.3's FLOP-based "≈2.1× overhead" estimate, which was never previously
measured (``plan_lot7.md`` §0.4).

The training loop itself (``train_under_time_budget``) is dataset- and model-agnostic — no
reference to ``AutoEncoder`` or MNIST anywhere in it (``plan_lot7.md`` §0.8) — so that
``tests/test_equal_wallclock_bench.py`` can exercise it directly, offline, against a tiny synthetic
task, at sub-second budgets.

Usage (from the repository root, with the project venv):
    PYTHONPATH=src .venv/bin/python benchmarks/equal_wallclock_bench.py --time-budget 60
    PYTHONPATH=src .venv/bin/python benchmarks/equal_wallclock_bench.py --modes diag kfac --time-budget 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mnist_autoencoder import (  # noqa: E402
    AutoEncoder,
    _load_reference_adafisher,
    _make_optimizer,
    _mnist_loader,
)

ALL_MODES = ("diag", "kfac", "ekfac", "tkfac", "tekfac")  # plan_lot7.md §0.10 — no "reference"


# ----------------------------------------------------------------------------------------------
# Dataset- and model-agnostic training loop (plan_lot7.md §0.1-§0.3, §0.8)
# ----------------------------------------------------------------------------------------------


@dataclass
class StepRecord:
    step: int
    epoch: float
    elapsed_s: float
    loss: float
    fwd_bwd_s: float
    step_s: float


def _sync(device: torch.device) -> None:
    """No-op on CPU; correctness guard for a future GPU run (plan_lot7.md §0.2) — CUDA kernel
    launches are asynchronous, so an un-synchronized perf_counter() bracket would time only launch
    overhead, not actual compute.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _default_prepare_batch(batch: Any) -> Tuple[Tensor, Tensor]:
    """Auto-encoder convention: reconstruct the flattened input (batch[0] is the image tensor,
    batch[1] its MNIST label, unused here — matches ``mnist_autoencoder.py::_train_one``).
    """
    x = batch[0].view(batch[0].size(0), -1)
    return x, x


def train_under_time_budget(
    model: nn.Module,
    optimizer: Any,
    data_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    time_budget_s: float,
    *,
    device: torch.device = torch.device("cpu"),
    max_steps: int = 10_000,
    prepare_batch: Callable[[Any], Tuple[Tensor, Tensor]] = _default_prepare_batch,
) -> List[StepRecord]:
    """Run ``optimizer`` on ``model`` over ``data_loader``, re-iterating it (i.e. starting a fresh
    epoch) for as long as needed, until either ``time_budget_s`` of wall-clock time has elapsed or
    ``max_steps`` batches have run (plan_lot7.md §0.11's defensive bound; not expected to trigger
    under a sane budget).

    The budget is checked *before* each batch's processing starts, but *after* that batch has
    already been fetched from ``data_loader`` (plain ``for batch in data_loader:`` iteration calls
    ``__next__`` before entering the loop body, so the fetch's cost is already reflected in the
    ``perf_counter()`` read the check uses) — plan_lot7.md §0.1. A run's total elapsed time
    therefore overshoots ``time_budget_s`` by at most one batch's own *processing* duration, never
    by more; that bound is not small and constant — a step landing on a ``T_inv``/``T_eig``
    amortised refresh can be an order of magnitude slower than the steady-state median
    (plan_lot7.md §6).
    """
    batches_per_epoch = len(data_loader)
    if batches_per_epoch == 0:
        raise ValueError("data_loader must yield at least one batch")

    records: List[StepRecord] = []
    step = 0
    t0 = perf_counter()
    epoch = 0
    while step < max_steps:
        for batch_idx, batch in enumerate(data_loader):
            if perf_counter() - t0 >= time_budget_s or step >= max_steps:
                return records

            x, target = prepare_batch(batch)
            x, target = x.to(device), target.to(device)

            t_fwd_bwd_0 = perf_counter()
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, target)
            loss.backward()
            _sync(device)
            fwd_bwd_s = perf_counter() - t_fwd_bwd_0

            t_step_0 = perf_counter()
            optimizer.step()
            _sync(device)
            step_s = perf_counter() - t_step_0

            records.append(
                StepRecord(
                    step=step,
                    epoch=epoch + batch_idx / batches_per_epoch,
                    elapsed_s=perf_counter() - t0,
                    loss=float(loss.item()),
                    fwd_bwd_s=fwd_bwd_s,
                    step_s=step_s,
                )
            )
            step += 1
        epoch += 1
    return records


# ----------------------------------------------------------------------------------------------
# MNIST auto-encoder driver (plan_lot7.md §1.2)
# ----------------------------------------------------------------------------------------------


def run_all_modes(
    modes: Sequence[str], args: argparse.Namespace, reference_module
) -> Dict[str, List[StepRecord]]:
    results: Dict[str, List[StepRecord]] = {}
    for mode in modes:
        torch.manual_seed(args.seed)  # plan_lot7.md §0.5 — identical init + minibatch order
        model = AutoEncoder()
        optimizer = _make_optimizer(mode, model, args, reference_module)
        loader = _mnist_loader(args.batch_size, train=True)
        loss_fn = nn.MSELoss()
        print(f"[{mode}] training for up to {args.time_budget:.1f}s (max_steps={args.max_steps})")
        records = train_under_time_budget(
            model, optimizer, loader, loss_fn, args.time_budget, max_steps=args.max_steps
        )
        if records:
            print(
                f"[{mode}] completed {len(records)} steps, "
                f"{records[-1].epoch:.2f} epochs, {records[-1].elapsed_s:.1f}s, "
                f"final loss={sum(r.loss for r in records[-10:]) / min(10, len(records)):.6f}"
            )
        else:
            print(f"[{mode}] completed 0 steps — time_budget_s too small for one batch")
        results[mode] = records
    return results


# ----------------------------------------------------------------------------------------------
# Reporting (plan_lot7.md §1.2)
# ----------------------------------------------------------------------------------------------


def _median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def write_csv(results: Dict[str, List[StepRecord]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "step", "epoch", "elapsed_s", "loss", "fwd_bwd_s", "step_s"])
        for mode, records in results.items():
            for r in records:
                writer.writerow([mode, r.step, r.epoch, r.elapsed_s, r.loss, r.fwd_bwd_s, r.step_s])


def write_summary(results: Dict[str, List[StepRecord]], path: Path, skip_first: int = 5) -> str:
    """Steady-state medians, skipping each run's first ``skip_first`` steps (plan_lot7.md §1.2 —
    excludes one-time hook / lazy-initialization cost from the timing medians).
    """
    lines = [
        "| mode | steps | epochs | final loss (last 10) | median fwd+bwd (s) | median step (s) | "
        "step/fwd+bwd ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, records in results.items():
        if not records:
            lines.append(f"| {mode} | 0 | - | - | - | - | - |")
            continue
        steady = records[skip_first:] if len(records) > skip_first else records
        fwd_bwd = _median([r.fwd_bwd_s for r in steady])
        step = _median([r.step_s for r in steady])
        ratio = step / fwd_bwd if fwd_bwd > 0 else float("nan")
        final_loss = sum(r.loss for r in records[-10:]) / min(10, len(records))
        lines.append(
            f"| {mode} | {len(records)} | {records[-1].epoch:.2f} | {final_loss:.6f} | "
            f"{fwd_bwd * 1e3:.3f} ms | {step * 1e3:.3f} ms | {ratio:.2f}x |"
        )
    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def write_plots(results: Dict[str, List[StepRecord]], output_dir: Path, smoothing_window: int = 10) -> bool:
    """Loss-vs-epoch and loss-vs-wall-clock-time plots, one line per mode. Returns False (and skips
    plotting) if matplotlib is not importable — plotting is an optional bench dependency
    (plan_lot7.md §0.9); the CSV/summary outputs never depend on it.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots (CSV/summary still written)", file=sys.stderr)
        return False

    def _smoothed(values: List[float], window: int) -> List[float]:
        out = []
        for i in range(len(values)):
            lo = max(0, i - window + 1)
            out.append(sum(values[lo : i + 1]) / (i - lo + 1))
        return out

    for x_attr, xlabel, filename in [
        ("epoch", "epoch", "loss_vs_epoch.png"),
        ("elapsed_s", "wall-clock time (s)", "loss_vs_time.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for mode, records in results.items():
            if not records:
                continue
            xs = [getattr(r, x_attr) for r in records]
            ys = _smoothed([r.loss for r in records], smoothing_window)
            ax.plot(xs, ys, label=mode)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"training loss ({smoothing_window}-step moving average)")
        ax.set_title(f"MNIST auto-encoder, loss vs. {xlabel}", fontsize=11)
        ax.legend()
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
    return True


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", choices=list(ALL_MODES), default=list(ALL_MODES))
    parser.add_argument("--time-budget", dest="time_budget", type=float, default=60.0)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--tcov", type=int, default=100)
    parser.add_argument("--gammas", type=float, nargs=2, default=(0.92, 0.008))
    parser.add_argument("--t-inv", dest="t_inv", type=int, default=100)
    parser.add_argument("--t-eig", dest="t_eig", type=int, default=100)
    parser.add_argument("--t-re", dest="t_re", type=int, default=1)
    parser.add_argument(
        "--minmax-normalization",
        dest="minmax_normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="diag mode only; default True — the paper-faithful configuration an actual user "
        "gets (plan_lot7.md §0.10), not mnist_autoencoder.py's own bit-exactness-check default "
        "of False.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "benchmarks/outputs/lot7_equal_wallclock"
    )
    parser.add_argument("--no-plot", dest="no_plot", action="store_true", default=False)
    args = parser.parse_args(argv)

    reference_module = _load_reference_adafisher()
    results = run_all_modes(args.modes, args, reference_module)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results, args.output_dir / "records.csv")
    summary_text = write_summary(results, args.output_dir / "summary.md")
    print("\n" + summary_text)

    if not args.no_plot:
        wrote_plots = write_plots(results, args.output_dir)
        if wrote_plots:
            print(f"plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
