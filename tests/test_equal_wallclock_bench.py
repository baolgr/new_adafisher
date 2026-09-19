"""Lot 7 (docs/reports/plan_lot7.md §1.3, §2): exercises ``benchmarks/common/loop.py`` directly —
the dataset- and model-agnostic ``train_under_budget`` harness, its ``data_s`` accounting, and the
``evaluate``/``top1`` pair that measures every run — against a tiny synthetic in-memory task, never
MNIST and never network I/O (plan_lot7.md §0.8). The real §6.3 deliverable (five loss-vs-epoch /
loss-vs-time curves on the actual MNIST auto-encoder) is produced by running the script itself, not
by this test (plan_lot7.md §2).
"""

from __future__ import annotations

import math
import sys
import weakref
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from conftest import seed_all
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adafisher_modes import AdaFisherMulti  # noqa: E402

from benchmarks.common.loop import (  # noqa: E402
    evaluate,
    release_optimizer,
    top1,
    train_under_budget,
)

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
    records, _ = train_under_budget(
        model, optimizer, loader, loss_fn, budget_s=budget, prepare_batch=_prepare_batch,
        log_fn=lambda _: None,
    )

    assert len(records) >= 1, f"{mode}: no step completed within a {budget}s budget"
    assert all(math.isfinite(r.loss) for r in records), f"{mode}: non-finite loss recorded"

    # plan_lot7.md §0.1: the budget is checked before each batch, so the run overshoots by at most
    # the LAST batch's own cycle — not by the largest cycle anywhere in the run.
    #
    # This used to be asserted against ``max(fwd_bwd_s + step_s)`` over *all* steps, which is the
    # first step's one-time hook and lazy-initialization cost: measured here at 18.9 ms for
    # ``diag`` against a real overshoot of 0.10 ms, i.e. a bound 185x too generous, and 6.3% of the
    # whole budget. ``bound_is_small`` below is what rules that form out.
    #
    # The bound is the last record's ``data_s + fwd_bwd_s + step_s``, and each of the three terms
    # is needed. Write ``c`` for the elapsed time at the last budget check, which the loop passed,
    # so ``c < budget``. Between that check and the record the loop does the transfer (inside
    # ``data_s``, which starts at the previous step's record), then the forward+backward, then the
    # optimizer step. Dropping ``data_s`` breaks the derivation, and it breaks in practice too:
    # over 20 repetitions x 5 modes the ``fwd_bwd_s + step_s`` form went negative by 1.4 us on
    # ``kfac`` (worst slack over all 100 runs), so it is only assertable with an epsilon larger
    # than itself. With ``data_s`` the worst slack over those same 100 runs was +20.2 us.
    last = records[-1]
    bound = last.data_s + last.fwd_bwd_s + last.step_s
    overshoot = last.elapsed_s - budget
    assert overshoot <= bound + 1e-4, (
        f"{mode}: overshot the budget by {overshoot * 1e6:.1f} us, more than the last batch's own "
        f"cycle ({bound * 1e6:.1f} us)"
    )
    # Non-vacuity: a bound worth a fraction of the budget would make the line above unfailable.
    assert bound < 0.02 * budget, (
        f"{mode}: the overshoot bound is {bound / budget:.1%} of the budget — too large to "
        "constrain anything"
    )

    elapsed = [r.elapsed_s for r in records]
    assert elapsed == sorted(elapsed), f"{mode}: elapsed_s is not monotonically increasing"
    assert [r.step for r in records] == list(range(len(records))), f"{mode}: step is not contiguous"

    # data_s accounts for the rest of the clock: fetch, host-to-device copy, and whatever on_step
    # did. Measured over 100 runs at this scale, the three timings cover 99.75-99.80% of the last
    # record's elapsed time; the remainder is the ``loss.item()`` and record-building between the
    # optimizer step and the record, which no timer here brackets.
    assert all(r.data_s >= 0.0 for r in records), f"{mode}: negative data_s"
    accounted = sum(r.data_s + r.fwd_bwd_s + r.step_s for r in records)
    assert 0.99 <= accounted / last.elapsed_s <= 1.0, (
        f"{mode}: data_s + fwd_bwd_s + step_s covers {accounted / last.elapsed_s:.4f} of the "
        "elapsed clock"
    )


def test_max_steps_guard_is_effective() -> None:
    """plan_lot7.md §0.11: an (effectively) unbounded time budget still terminates at exactly
    ``max_steps``.
    """
    seed_all(0)
    model, loader = _toy_task()
    optimizer = AdaFisherMulti(model, lr=1e-3, TCov=1, fisher_mode="diag")
    loss_fn = nn.MSELoss()

    records, _ = train_under_budget(
        model, optimizer, loader, loss_fn, budget_s=1e6, max_steps=5,
        prepare_batch=_prepare_batch, log_fn=lambda _: None,
    )
    assert len(records) == 5
    assert [r.step for r in records] == [0, 1, 2, 3, 4]


# ----------------------------------------------------------------------------------------------
# evaluate() and top1() — the two functions that produce every validation and test number
#
# Until now the whole suite executed neither: every runner-driven test passes ``eval_fn=None`` or
# supplies its own stub. Nothing would have noticed if evaluation had stopped calling
# ``model.eval()``, started updating BatchNorm's running statistics, or mis-weighted a ragged final
# batch.
# ----------------------------------------------------------------------------------------------


class _BNNet(nn.Module):
    """A net whose forward pass *records* the mode it was called in, and whose BatchNorm would
    visibly move if it were ever run in training mode.
    """

    def __init__(self, d_in: int = 4, d_out: int = 3) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(d_in)
        self.fc = nn.Linear(d_in, d_out)
        self.observed: list = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.observed.append((self.training, self.bn.training, torch.is_grad_enabled(),
                              x.requires_grad))
        return self.fc(self.bn(x))


def _ragged_loader(n: int = 11, batch_size: int = 4, d_in: int = 4, d_out: int = 3):
    """4 / 4 / 3 — a deliberately ragged last batch, which is what a batch-size-weighted mean is
    for. ``drop_last`` is False on every validation and test loader this project builds.
    """
    seed_all(3)
    x = torch.randn(n, d_in, dtype=torch.float64)
    y = torch.randn(n, d_out, dtype=torch.float64)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size), x, y


def test_evaluate_weights_a_ragged_last_batch_by_its_size() -> None:
    """The loss must be the whole-set mean, not the mean of the per-batch means.

    On a 4/4/3 split the two differ: the unweighted mean over-counts the 3-example batch by 4/3.
    """
    loader, x, y = _ragged_loader()
    seed_all(4)
    model = nn.Linear(4, 3).double()
    loss_fn = nn.MSELoss()

    loss, metric = evaluate(model, loader, loss_fn, torch.device("cpu"))

    with torch.no_grad():
        reference = float(loss_fn(model(x), y).item())  # one pass over the whole set
    assert loss == pytest.approx(reference, rel=1e-12, abs=1e-14)
    assert math.isnan(metric), "no metric_fn means no accuracy, not zero"

    # Non-vacuous: the unweighted mean of the batch means is a genuinely different number here.
    with torch.no_grad():
        per_batch = [float(loss_fn(model(xb), yb).item()) for xb, yb in loader]
    unweighted = sum(per_batch) / len(per_batch)
    assert abs(unweighted - reference) > 1e-3, (
        "the fixture no longer distinguishes the weighted mean from the unweighted one"
    )


def test_evaluate_accuracy_is_correct_over_the_right_total() -> None:
    """``metric_fn`` returns a batch *count*; ``evaluate`` divides by the number of examples seen,
    which on a ragged split is not ``batches * batch_size``.
    """
    seed_all(5)
    n, classes = 11, 3
    logits = torch.randn(n, classes)
    targets = torch.randint(0, classes, (n,))
    expected_correct = int((logits.argmax(dim=1) == targets).sum())
    assert 0 < expected_correct < n, "the fixture must be neither all right nor all wrong"

    # An identity "model", so the loader's inputs *are* the logits and the expected count is known.
    loader = DataLoader(TensorDataset(logits, targets), batch_size=4)
    _loss, acc = evaluate(nn.Identity(), loader, nn.CrossEntropyLoss(), torch.device("cpu"),
                          metric_fn=top1)
    assert acc == pytest.approx(expected_correct / n, rel=0, abs=1e-15)
    assert acc != pytest.approx(expected_correct / (3 * 4)), "divided by batches * batch_size"


def test_top1_returns_a_count_not_a_rate() -> None:
    logits = torch.tensor([[2.0, 1.0], [0.0, 5.0], [3.0, 1.0], [1.0, 0.0]])
    targets = torch.tensor([0, 1, 1, 0])
    assert top1(logits, targets) == 3
    assert isinstance(top1(logits, targets), int)


def test_evaluate_runs_in_eval_mode_without_grad_on_every_batch() -> None:
    loader, _x, _y = _ragged_loader()
    model = _BNNet().double()
    model.train()
    running = {k: v.clone() for k, v in model.bn.state_dict().items()}

    evaluate(model, loader, nn.MSELoss(), torch.device("cpu"))

    assert len(model.observed) == len(loader) == 3, "evaluate skipped a batch"
    assert all(obs == (False, False, False, False) for obs in model.observed), (
        f"evaluate must run with training=False and grad disabled on every batch: {model.observed}"
    )
    # BatchNorm in train mode would move all three of these.
    for key, before in running.items():
        assert torch.equal(model.bn.state_dict()[key], before), f"bn.{key} moved during evaluation"
    assert model.training is True, "evaluate must restore the mode it found"


def test_evaluate_restores_eval_mode_too() -> None:
    """The other half of ``model.train(was_training)``: a model handed over in eval mode must not
    come back in train mode.
    """
    loader, _x, _y = _ragged_loader()
    model = _BNNet().double()
    model.eval()
    evaluate(model, loader, nn.MSELoss(), torch.device("cpu"))
    assert model.training is False


def test_evaluate_on_an_empty_loader_is_nan_not_a_division_by_zero() -> None:
    empty = DataLoader(TensorDataset(torch.zeros(0, 4), torch.zeros(0, 3)), batch_size=4)
    loss, acc = evaluate(nn.Linear(4, 3), empty, nn.MSELoss(), torch.device("cpu"),
                         metric_fn=top1)
    assert math.isnan(loss) and math.isnan(acc)


# ----------------------------------------------------------------------------------------------
# Arm teardown: nothing may survive run_arm
# ----------------------------------------------------------------------------------------------


def test_release_optimizer_removes_only_its_own_hooks() -> None:
    seed_all(0)
    model, _loader = _toy_task()
    foreign = []
    model[0].register_forward_hook(lambda m, i, o: foreign.append(1))
    optimizer = AdaFisherMulti(model, lr=1e-3, TCov=1, fisher_mode="ekfac", T_eig=1)
    hooked = len(optimizer.modules)
    assert hooked == 2, "the toy net has two Linear layers to hook"

    removed = release_optimizer(model, optimizer)
    assert removed == 2 * hooked, "one forward and one full-backward hook per hooked module"
    # The unrelated hook is untouched and still fires.
    model(torch.randn(2, 6)).sum().backward()
    assert foreign == [1]


def test_no_arm_survives_run_arm() -> None:
    """The whole runner, all seven arms: every model and optimizer must be unreachable once its
    arm has returned.

    ``AdaFisherMulti`` puts a bound method of itself into each hooked module's hook dictionary
    while holding the module, so the two form a cycle that reference counting cannot break. Left
    alone, every arm's weights and curvature factors stay resident for the rest of the process and
    the next arm's ``reset_peak_memory`` starts from a baseline that includes them — which is how
    the shipped ``mnist_autoencoder`` report came to show a ``peak VRAM`` column rising
    monotonically in the order the arms ran, with ``adam`` (no factors at all) the most expensive
    of the seven.
    """
    from benchmarks.common import runner as runner_module
    from benchmarks.common.optimizers import ARMS, HParams, build_optimizer
    from benchmarks.common.runner import Benchmark
    from benchmarks.common.runner import main as run_main

    alive: dict = {}

    def build_model():
        model = nn.Sequential(nn.Linear(6, 5), nn.Sigmoid(), nn.Linear(5, 3))
        alive.setdefault("models", []).append(weakref.ref(model))
        return model

    def spy_build_optimizer(arm, model, hp, **kwargs):
        optimizer = build_optimizer(arm, model, hp, **kwargs)
        alive.setdefault("optimizers", []).append((arm, weakref.ref(optimizer)))
        return optimizer

    def build_data(data_root, *, batch_size=8, seed=0, num_workers=0, cutout=True,
                   allow_download=False, train_subset=None):
        seed_all(0)
        dataset = TensorDataset(torch.randn(32, 6), torch.randint(0, 3, (32,)))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True), None, None

    bench = Benchmark(
        name="synthetic", build_model=build_model, build_data=build_data,
        loss_fn=nn.CrossEntropyLoss(), hparams=HParams(lr=1e-2, tcov=1), epochs=1, batch_size=8,
        arms=("diag", "kfac", "ekfac", "tkfac", "tekfac", "adam", "adamw"),
    )
    assert set(bench.arms) < set(ARMS) and len(bench.arms) == 7

    # Automatic collection is switched off for the duration, so the result does not depend on
    # whether an unrelated allocation happens to trigger a sweep. ``gc.collect()`` still runs when
    # called explicitly, which is what ``release_optimizer`` does — so the fix works here and its
    # absence leaks all seven arms rather than a gc-timing-dependent subset.
    import gc
    import tempfile

    original = runner_module.build_optimizer
    runner_module.build_optimizer = spy_build_optimizer  # type: ignore[assignment]
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with tempfile.TemporaryDirectory() as out:
            run_main(bench, argv=["--output-dir", out, "--budget-mode", "epochs",
                                  "--device", "cpu", "--no-plot", "--num-workers", "0"])
        # Read the weak references while collection is still off. Re-enabling it first would let
        # the very next allocation sweep the generation and free the cycles, which is exactly the
        # late, incidental collection this test exists to rule out.
        leaked = [arm for arm, ref in alive["optimizers"] if ref() is not None]
        surviving_models = sum(1 for ref in alive["models"] if ref() is not None)
    finally:
        runner_module.build_optimizer = original  # type: ignore[assignment]
        if gc_was_enabled:
            gc.enable()

    assert len(alive["optimizers"]) == 7 and len(alive["models"]) == 7
    assert leaked == [], f"optimizers still alive after run_arm returned: {leaked}"
    assert surviving_models == 0, f"{surviving_models} model(s) outlived their arm"


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
        records, _ = train_under_budget(
            model, optimizer, loader, loss_fn,
            budget_s=1e6, max_steps=n_steps, prepare_batch=_prepare_batch,
            log_fn=lambda _: None,
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
