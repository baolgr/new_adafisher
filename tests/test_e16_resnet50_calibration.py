"""``fisher_ref/experiments/e16_resnet50_calibration.py``: the ResNet-50 calibration of E16's fourth
amendment. What is pinned is the part that turns the scan into E16's grid without anyone choosing
it -- the lattice, the snapping and the five-value window -- and the plan of the jobs: the scan
covers E14's ResNet-20 window exactly once, and every cell carries the two settings that differ from
the benchmark's own (``fisher_batch_samples=None``, ``norm_exact_rescaling=True``)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


@pytest.fixture
def cal(monkeypatch):
    monkeypatch.setenv("E16_MODEL", "resnet50_cifar")
    monkeypatch.setenv("E16_SEED", "0")
    return importlib.import_module("fisher_ref.experiments.e16_resnet50_calibration")


@pytest.fixture
def bench():
    from benchmarks.common.runner import discover_benchmarks
    return discover_benchmarks()["resnet50_cifar"]


def test_the_grid_rule_on_worked_examples(cal) -> None:
    # Both modes agree: the window is ResNet-20's own E16 grid.
    assert cal.grid_from_scan({"ekfac": 1e-11, "tekfac": 1e-11}) == pytest.approx(
        [1e-10, 3e-11, 1e-11, 3e-12, 1e-12], rel=1e-12)
    # 1e-10 and 1e-11: geometric mean 3.16e-11, nearest lattice value 3e-11 (0.023 decades away).
    assert cal.grid_from_scan({"ekfac": 1e-10, "tekfac": 1e-11}) == pytest.approx(
        [3e-10, 1e-10, 3e-11, 1e-11, 3e-12], rel=1e-12)
    # 3e-10 and 3e-12: geometric mean 3e-11 exactly.
    assert cal.grid_from_scan({"ekfac": 3e-10, "tekfac": 3e-12}) == pytest.approx(
        [3e-10, 1e-10, 3e-11, 1e-11, 3e-12], rel=1e-12)


def test_snap_takes_the_nearest_lattice_value_in_log(cal) -> None:
    assert cal.snap(2e-11) == pytest.approx(3e-11)      # 0.18 decades from 3e-11, 0.30 from 1e-11
    assert cal.snap(1.5e-11) == pytest.approx(1e-11)    # 0.18 from 1e-11, 0.30 from 3e-11
    assert cal.snap(1e-11) == pytest.approx(1e-11)


def test_the_scan_covers_e14s_window_exactly_once(cal) -> None:
    covered = [v for part in cal.PARTS.values() for v in part]
    assert covered == cal.SCAN_GRID
    assert len(cal.PARTS) == 5 and all(len(p) == 2 for p in cal.PARTS.values())
    assert cal.SCAN_GRID[0] == pytest.approx(1e-8) and cal.SCAN_GRID[-1] == pytest.approx(3e-13)


def test_the_scan_cells_are_e16s_add_arm(cal, bench) -> None:
    hp = bench.hparams
    for part, lams in cal.PARTS.items():
        cells = cal.plan(part, bench)
        assert [c["arm"] for c in cells] == ["add"] * len(lams)
        for c, lam in zip(cells, lams):
            assert c["lam"] == lam and c["lr"] == hp.lr * (lam / hp.lam)   # E16's cap * lambda
            assert c["wd"] == hp.weight_decay and c["freeze"] is None
            assert c["overrides"] == {"norm_exact_rescaling": True, "fisher_batch_samples": None}


def test_the_timing_cells_are_e16s_arms_at_one_q(cal, bench) -> None:
    import fisher_ref.experiments.e16_floor_clip as e16
    cells = cal.plan("timing", bench)
    assert [c["arm"] for c in cells] == list(cal.TIMING_ARMS)
    for c in cells[1:]:
        o = c["overrides"]
        assert o["rescale_form"] == "clip" and o["clip_fraction"] == cal.TIMING_Q
        assert all(o[k] == v for k, v in e16.CLIP_ARMS[c["arm"]].items())
        assert o["norm_exact_rescaling"] is True and o["fisher_batch_samples"] is None
    with pytest.raises(SystemExit):
        cal.plan("scan9", bench)


def _plan(monkeypatch, model: str, seed: int, grid=None):
    import fisher_ref.experiments.e16_floor_clip as e16
    from benchmarks.common.runner import discover_benchmarks
    from fisher_ref.experiments.e5_unhooked_freeze_control import unhooked_parameter_names
    monkeypatch.setattr(e16, "MODE", "ekfac")
    monkeypatch.setattr(e16, "SEED", seed)
    if grid is not None:
        monkeypatch.setitem(e16.LAMBDA_GRID, model, grid)
        monkeypatch.setitem(e16.CHECK_LAMBDA, (model, "ekfac"), grid[2])
    b = discover_benchmarks()[model]
    hp = b.hparams
    return e16.plan_cells(model, hp.lam, hp.lr, hp.weight_decay, hp.decoupled_wd,
                          unhooked_parameter_names(b))


def test_the_reduced_design_matches_the_calibration_drivers_constants(cal) -> None:
    import fisher_ref.experiments.e16_floor_clip as e16
    r = e16.REDUCED["resnet50_cifar"]
    assert list(r["clip_grid"]) == cal.REDUCED_Q and list(r["seeds"]) == cal.REDUCED_SEEDS


def test_resnet50_plans_its_reduced_design(monkeypatch) -> None:
    grid = [1e-10, 3e-11, 1e-11, 3e-12, 1e-12]
    for seed, extra in ((0, 1), (1, 0)):
        cells = _plan(monkeypatch, "resnet50_cifar", seed, grid)
        arms = [c["arm"] for c in cells]
        assert arms.count("dupcheck") == extra and "repro" not in arms
        assert "floor" not in arms and "cliplr" not in arms
        assert sorted(c["value"] for c in cells if c["arm"] == "add") == sorted(grid)
        for arm in ("clipema", "clip", "clipfixed"):
            assert [c["value"] for c in cells if c["arm"] == arm] == [0.95, 0.7, 0.3]
        assert len(cells) == 5 + 9 + extra
        assert all(c["overrides"]["fisher_batch_samples"] is None
                   and c["overrides"]["norm_exact_rescaling"] is True for c in cells)


def test_the_full_design_is_unchanged_on_every_other_network(monkeypatch) -> None:
    for model in ("cnn_gn_cifar", "vit_micro_cifar", "cct_2_3x2_cifar", "resnet20_cifar"):
        for seed, n in ((0, 36), (1, 34)):
            cells = _plan(monkeypatch, model, seed)
            assert len(cells) == n
            assert all("fisher_batch_samples" not in c["overrides"] for c in cells)


def test_the_driver_grid_is_the_rule_applied_to_the_scan(cal) -> None:
    """The calibration (jobs 21546221-32) put both modes' best validation lambda at 3e-12; the
    pre-registered rule turns that into the grid the driver runs, and nothing else does."""
    import fisher_ref.experiments.e16_floor_clip as e16
    assert e16.RESNET50_GRID == pytest.approx(cal.grid_from_scan({"ekfac": 3e-12, "tekfac": 3e-12}))
    assert e16.CHECK_LAMBDA[("resnet50_cifar", "ekfac")] == pytest.approx(3e-12)
