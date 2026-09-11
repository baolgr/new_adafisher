"""The cosine schedule under the WCT protocol (``benchmarks/common/schedules.py``).

Two biases were *measured* on the first real campaign, both traceable to one shared
``T_max = --epochs`` driving ``torch.optim.lr_scheduler.CosineAnnealingLR``:

1. **A cheap arm's LR climbs back up.** ``CosineAnnealingLR`` is periodic, so past ``T_max`` it
   returns towards ``base_lr``. Every arm that overshot the nominal epoch count was ``adam`` or
   ``adamw`` (7 of the 8 models); ``resnet20_cifar/adam`` hit ``lr = 0`` at epoch 49, then trained
   9 more epochs with the LR rising back to ``6.2e-5``, and its best val acc decayed from 88.82%
   to 88.06%.
2. **An expensive arm never reaches the floor.** ``resnet50_cifar/ekfac`` stopped at epoch 38 of
   50 with ``lr`` still at ``1.6e-4``; across the five Fisher arms,
   ``corr(epochs completed, best val acc)`` was +0.98 / +0.92 / +0.92 on the three long models.

``NominalCosine`` fixes (1) without moving any LR an in-budget arm ever saw; ``BudgetCosine``
fixes (2) by annealing over the arm's own wall-clock budget.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.loop import train_under_budget  # noqa: E402
from benchmarks.common.schedules import BudgetCosine, NominalCosine  # noqa: E402

BASE_LR = 1e-3


def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=BASE_LR)


def _lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


# ---------------------------------------------------------------------------------------------
# NominalCosine: identical to CosineAnnealingLR inside T_max, clamped outside it
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("t_max", [20, 30, 50])
def test_nominal_cosine_matches_torch_inside_t_max(t_max: int) -> None:
    """No arm that stayed within the nominal length changes: same LR at every epoch.

    This is what preserves the first campaign's bit-exactly reproducible ``diag`` trajectories
    (verified at 2200/2200, 8580/8580 and 10530/10530 identical steps across two campaigns).
    """
    mine, theirs = _optimizer(), _optimizer()
    ours = NominalCosine(mine, t_max=t_max)
    torchs = torch.optim.lr_scheduler.CosineAnnealingLR(theirs, T_max=t_max)
    for _ in range(t_max):
        ours.step()
        torchs.step()
        assert _lr(mine) == pytest.approx(_lr(theirs), rel=1e-12, abs=1e-18)
    assert _lr(mine) == pytest.approx(0.0, abs=1e-18)  # the floor is reached exactly at T_max


def test_nominal_cosine_is_clamped_past_t_max_where_torch_rises() -> None:
    """The regression itself: ``resnet20_cifar/adam`` ran 59 epochs of a 50-epoch schedule.

    58 scheduler steps, not 59: the loop steps the scheduler once per *completed* epoch and that
    arm's last epoch was cut short by the budget, which is exactly what its epochs.csv shows
    (epoch 57 and epoch 58 both recorded at ``6.18e-05``).
    """
    t_max, overshoot = 50, 8
    mine, theirs = _optimizer(), _optimizer()
    ours = NominalCosine(mine, t_max=t_max)
    torchs = torch.optim.lr_scheduler.CosineAnnealingLR(theirs, T_max=t_max)
    for _ in range(t_max + overshoot):
        ours.step()
        torchs.step()
    assert _lr(mine) == pytest.approx(0.0, abs=1e-18), "must hold at the floor, not rise"
    # Non-vacuous: torch really does climb back, and to a value worth caring about.
    assert _lr(theirs) > 1e-5
    assert _lr(theirs) == pytest.approx(6.18e-5, rel=1e-2)  # resnet20_cifar/adam's measured value


def test_nominal_cosine_rejects_nonpositive_t_max() -> None:
    with pytest.raises(ValueError):
        NominalCosine(_optimizer(), t_max=0)


# ---------------------------------------------------------------------------------------------
# BudgetCosine: one full cosine over the arm's own budget
# ---------------------------------------------------------------------------------------------


def test_budget_cosine_spans_base_lr_to_floor_over_the_budget() -> None:
    optimizer = _optimizer()
    schedule = BudgetCosine(optimizer, budget_s=100.0)
    schedule(0.0)
    assert _lr(optimizer) == pytest.approx(BASE_LR)
    schedule(50.0)
    assert _lr(optimizer) == pytest.approx(BASE_LR / 2)  # cos(pi/2) = 0
    schedule(100.0)
    assert _lr(optimizer) == pytest.approx(0.0, abs=1e-18)
    schedule(140.0)  # an arm may overshoot by one batch; it must not rise
    assert _lr(optimizer) == pytest.approx(0.0, abs=1e-18)


def test_budget_cosine_is_monotone_decreasing() -> None:
    optimizer = _optimizer()
    schedule = BudgetCosine(optimizer, budget_s=7.0)
    seen = []
    for i in range(71):
        schedule(i * 0.1)
        seen.append(_lr(optimizer))
    assert all(a >= b for a, b in zip(seen, seen[1:]))
    assert seen[0] == pytest.approx(BASE_LR)


def test_budget_cosine_shape_is_the_same_cosine_as_nominal() -> None:
    """Same closed form, different abscissa: a fraction of the budget rather than of T_max."""
    optimizer = _optimizer()
    schedule = BudgetCosine(optimizer, budget_s=1.0)
    for fraction in (0.0, 0.13, 0.37, 0.5, 0.82, 1.0):
        schedule(fraction)
        expected = BASE_LR * 0.5 * (1.0 + math.cos(math.pi * fraction))
        assert _lr(optimizer) == pytest.approx(expected, abs=1e-18)


def test_budget_cosine_rejects_an_unbounded_budget() -> None:
    """The reference arm is unbudgeted, so it must be driven by ``NominalCosine`` instead —
    ``elapsed / inf`` would pin the LR at ``base_lr`` for the whole run.
    """
    with pytest.raises(ValueError):
        BudgetCosine(_optimizer(), budget_s=float("inf"))
    with pytest.raises(ValueError):
        BudgetCosine(_optimizer(), budget_s=0.0)


# ---------------------------------------------------------------------------------------------
# Through the real loop
# ---------------------------------------------------------------------------------------------


def _tiny_loader(batches: int = 8):
    data = [(torch.randn(4, 2), torch.randn(4, 2)) for _ in range(batches)]

    class _Loader(list):
        def __len__(self) -> int:  # DataLoader-shaped enough for the loop
            return batches

    return _Loader(data)


def test_loop_drives_budget_cosine_to_the_floor_by_the_budget() -> None:
    """The property the protocol needs: however many epochs an arm fits, it finishes annealed."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR)
    schedule = BudgetCosine(optimizer, budget_s=0.30)
    steps, _ = train_under_budget(
        model, optimizer, _tiny_loader(), nn.MSELoss(),
        budget_s=0.30, max_epochs=10**9, lr_schedule=schedule, log_fn=lambda _m: None,
    )
    assert steps, "the run must have taken at least one step"
    # The LR in force for the last batch is set from the elapsed time *before* it, so it is within
    # one batch of the floor rather than exactly at it.
    assert _lr(optimizer) < BASE_LR * 0.02


def test_loop_without_lr_schedule_leaves_the_lr_untouched() -> None:
    """``lr_schedule=None`` is the default and must stay inert — every pre-existing test relies
    on this loop being bit-identical to its previous behaviour.
    """
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR)
    train_under_budget(
        model, optimizer, _tiny_loader(), nn.MSELoss(),
        budget_s=float("inf"), max_epochs=2, log_fn=lambda _m: None,
    )
    assert _lr(optimizer) == BASE_LR


# ---------------------------------------------------------------------------------------------
# Through the whole runner: which arm gets which schedule
# ---------------------------------------------------------------------------------------------


def _synthetic_bench(epochs: int, n: int, batch: int):
    from torch.utils.data import DataLoader, TensorDataset

    from benchmarks.common.optimizers import HParams
    from benchmarks.common.runner import Benchmark

    def build_data(data_root, *, batch_size=8, seed=0, num_workers=0, cutout=True,
                   allow_download=False, train_subset=None):
        torch.manual_seed(0)
        dataset = TensorDataset(torch.randn(n, 6), torch.randint(0, 3, (n,)))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True), None, None

    return Benchmark(
        name="synthetic",
        build_model=lambda: nn.Sequential(nn.Linear(6, 5), nn.Sigmoid(), nn.Linear(5, 3)),
        build_data=build_data,
        loss_fn=nn.CrossEntropyLoss(),
        hparams=HParams(lr=1e-2, baseline_lr=1e-2, tcov=1),
        epochs=epochs,
        batch_size=batch,
        arms=("diag", "adam"),
    )


def _epoch_lrs(path):
    import csv

    rows = list(csv.DictReader(open(path)))
    by: dict = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(float(r["lr"]))
    return by


@pytest.mark.parametrize("mode", ["budget", "nominal"])
def test_runner_never_lets_any_arm_lr_rise_again(tmp_path, mode: str) -> None:
    """The whole runner, offline: under either schedule, no arm's LR may increase.

    ``adam`` is the cheap arm here, as it was on every real model, so under ``--budget-mode wct``
    it fits more epochs than the nominal count into the reference arm's time — the exact situation
    in which the unclamped ``CosineAnnealingLR`` used to climb back towards ``base_lr``. Measured
    on this fixture: ``diag`` runs its 4 nominal epochs, ``adam`` runs 12, so the overshoot the
    assertion targets is genuinely exercised (12 is the ``--max-epoch-factor`` cap, which binds
    before the sub-second budget does — that is why ``adam``'s final LR here is *not* near the
    floor, and why ``run_arm`` warns about it).
    """
    from benchmarks.common.runner import main as run_main

    epochs, n, batch = 4, 64, 8
    run_main(_synthetic_bench(epochs, n, batch),
             argv=["--output-dir", str(tmp_path), "--budget-mode", "wct", "--device", "cpu",
                   "--no-plot", "--num-workers", "0", "--lr-schedule", mode])

    lrs = _epoch_lrs(tmp_path / "epochs.csv")
    assert set(lrs) == {"diag", "adam"}
    for arm, series in lrs.items():
        assert all(a >= b - 1e-15 for a, b in zip(series, series[1:])), (arm, mode, series)


def test_runner_anneals_the_unbudgeted_reference_arm_under_budget_mode(tmp_path) -> None:
    """``--lr-schedule budget`` must not leave the reference arm at ``base_lr``.

    The reference arm is unbudgeted (``budget_s = inf``) because it is what *defines* the budget,
    so ``elapsed / budget_s`` would be 0 for its whole run. ``run_arm`` therefore falls back to
    ``NominalCosine`` for it, and the floor must still be reached at the nominal epoch count.
    """
    from benchmarks.common.runner import main as run_main

    epochs = 4
    run_main(_synthetic_bench(epochs, 64, 8),
             argv=["--output-dir", str(tmp_path), "--budget-mode", "wct", "--device", "cpu",
                   "--no-plot", "--num-workers", "0", "--lr-schedule", "budget",
                   "--reference-arm", "diag"])

    reference = _epoch_lrs(tmp_path / "epochs.csv")["diag"]
    assert len(reference) == epochs, "the reference arm runs exactly the nominal epoch count"
    assert reference[-1] == pytest.approx(0.0, abs=1e-18), "it must reach the annealing floor"
    assert reference[0] > 0.0
