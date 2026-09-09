"""Lot 7 (docs/reports/plan_lot7.md §1.3, §2): exercises the dataset- and model-agnostic
``train_under_time_budget`` harness (``benchmarks/equal_wallclock_bench.py``) directly, against a
tiny synthetic in-memory task — never MNIST, never network I/O (plan_lot7.md §0.8). The real §6.3
deliverable (five loss-vs-epoch / loss-vs-time curves on the actual MNIST auto-encoder) is produced
by running the script itself, not by this test (plan_lot7.md §2).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from conftest import seed_all
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

from adafisher_modes import AdaFisherMulti  # noqa: E402
from equal_wallclock_bench import train_under_time_budget  # noqa: E402

# Small enough that a sub-second budget still crosses several cadence periods (plan_lot7.md §1.3).
_MODE_KWARGS = {
    "diag": {},
    "kfac": {"T_inv": 2},
    "ekfac": {"T_eig": 2},
    "tkfac": {"T_inv": 2},
    "tekfac": {"T_eig": 2, "T_re": 1},
}


def _toy_task(d_in: int = 6, d_hidden: int = 5, d_out: int = 3, n: int = 64, batch_size: int = 8):
    seed_all(0)
    x = torch.randn(n, d_in)
    y = torch.randn(n, d_out)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    model = nn.Sequential(nn.Linear(d_in, d_hidden), nn.Sigmoid(), nn.Linear(d_hidden, d_out))
    return model, loader


def _prepare_batch(batch):
    return batch[0], batch[1]


@pytest.mark.parametrize("mode", list(_MODE_KWARGS))
def test_budget_is_respected_and_run_completes(mode: str) -> None:
    seed_all(0)
    model, loader = _toy_task()
    optimizer = AdaFisherMulti(model, lr=1e-3, TCov=1, fisher_mode=mode, **_MODE_KWARGS[mode])
    loss_fn = nn.MSELoss()

    budget = 0.3
    records = train_under_time_budget(
        model, optimizer, loader, loss_fn, budget, prepare_batch=_prepare_batch
    )

    assert len(records) >= 1, f"{mode}: no step completed within a {budget}s budget"
    assert all(math.isfinite(r.loss) for r in records), f"{mode}: non-finite loss recorded"

    # plan_lot7.md §0.1: checked before each batch, so total elapsed overshoots the budget by at
    # most one batch's own duration.
    max_single_step = max((r.fwd_bwd_s + r.step_s) for r in records)
    assert records[-1].elapsed_s <= budget + max_single_step + 1e-3, (
        f"{mode}: budget overshot by more than one batch's duration"
    )

    elapsed = [r.elapsed_s for r in records]
    assert elapsed == sorted(elapsed), f"{mode}: elapsed_s is not monotonically increasing"
    assert [r.step for r in records] == list(range(len(records))), f"{mode}: step is not contiguous"


def test_max_steps_guard_is_effective() -> None:
    """plan_lot7.md §0.11: an (effectively) unbounded time budget still terminates at exactly
    ``max_steps``.
    """
    seed_all(0)
    model, loader = _toy_task()
    optimizer = AdaFisherMulti(model, lr=1e-3, TCov=1, fisher_mode="diag")
    loss_fn = nn.MSELoss()

    records = train_under_time_budget(
        model, optimizer, loader, loss_fn, time_budget_s=1e6, max_steps=5, prepare_batch=_prepare_batch
    )
    assert len(records) == 5
    assert [r.step for r in records] == [0, 1, 2, 3, 4]


def test_eigenbasis_modes_pay_more_per_step_than_diag(capsys) -> None:
    """Informational (plan.md §6.1's "measured, not asserted" precedent, applied to plan_lot7.md
    §0.4): a cheap, CI-safe stand-in for the real, wall-clock-dependent "≈2.1x" overhead measurement
    on the full MNIST bench.

    Empirically (recorded while writing this test, plan_lot7.md §5 point 5), the claim "every
    non-diag mode costs strictly more per step than diag" does **not** hold robustly at toy scale
    for ``kfac``/``tkfac``: their ``precondition`` is two matmuls against cached inverses, a cost
    comparable to (and, run to run, sometimes below) ``diag``'s own multi-op min-max normalisation
    pipeline (several elementwise ``min``/``max``/``where`` calls plus a ``kron`` reconstruction) —
    a genuine small-scale finding about per-torch-op dispatch overhead, not measurement noise to be
    papered over with a looser tolerance. ``ekfac``/``tekfac`` additionally project into and out of
    an eigenbasis (two more matmuls each way, ``Q_B^T M Q_A`` then ``Q_B (...) Q_A^T``, roughly
    double ``kfac``'s matmul count) and *do* show a large, reproducible (~1.5-1.8x observed across
    repeated runs at this scale), machine-plausible gap over ``diag``. Only that comparison is
    asserted; ``kfac``/``tkfac`` are reported for information only, alongside ``ekfac``/``tekfac``.
    """
    # Large enough that matmul/eigh cost is measurable above per-call dispatch-overhead noise
    # (plan_lot7.md §5 point 5), still small enough to run in well under a second per mode.
    n_steps = 80
    mode_kwargs = {
        "diag": {},
        "kfac": {"T_inv": 10},
        "ekfac": {"T_eig": 10},
        "tkfac": {"T_inv": 10},
        "tekfac": {"T_eig": 10, "T_re": 1},
    }
    medians: dict[str, float] = {}
    for mode, kwargs in mode_kwargs.items():
        seed_all(0)
        model, loader = _toy_task(d_in=256, d_hidden=192, d_out=96, n=512, batch_size=32)
        optimizer = AdaFisherMulti(model, lr=1e-3, TCov=1, fisher_mode=mode, **kwargs)
        loss_fn = nn.MSELoss()
        records = train_under_time_budget(
            model, optimizer, loader, loss_fn,
            time_budget_s=1e6, max_steps=n_steps, prepare_batch=_prepare_batch,
        )
        assert len(records) == n_steps
        steady = records[10:]
        step_times = sorted(r.step_s for r in steady)
        medians[mode] = step_times[len(step_times) // 2]
        print(f"[{mode}] median step_s over {len(steady)} steady-state steps = {medians[mode]:.6e}s")

    for mode in ("kfac", "tkfac"):
        print(f"[{mode}] vs [diag] ratio = {medians[mode] / medians['diag']:.3f}x (informational only)")

    for mode in ("ekfac", "tekfac"):
        assert medians[mode] > medians["diag"], (
            f"{mode}'s median step_s ({medians[mode]:.6e}s) is not larger than diag's "
            f"({medians['diag']:.6e}s) — expected, since {mode} additionally projects into and out "
            "of an eigenbasis (two more matmuls each way than kfac/tkfac's cached-inverse multiply)"
        )
