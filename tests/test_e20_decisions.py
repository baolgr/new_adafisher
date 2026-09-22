"""E20's decision script and driver, checked on synthetic output files and against E15.

The decision script must never read an incomplete or invalid set of jobs as a verdict, must apply
the pre-registered rules exactly (``docs/reports/plan_e20_s1b_vit_small.md``, section 2.1), and the
driver's constants must be the pre-registered ones -- the grids and the configuration E15 used.
All offline: no training run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import fisher_ref.experiments.e15_layer_damping as e15  # noqa: E402
from fisher_ref.experiments import e20_decisions as dec  # noqa: E402
from fisher_ref.experiments import e20_s1b_vit_small as drv  # noqa: E402

PROV = {"git_commit": "abc", "git_dirty": False, "smoke": False, "epochs": 15, "num_workers": 4,
        "batch": 32}
NOISE = [0.0, 0.3, -0.2, 0.1, -0.15]


def _acc(job: str, arm: str, value: float, seed: int, effects) -> float:
    """Test accuracy in [0, 1]: 0.5 plus the arm's effect in points plus a per-seed wobble."""
    return (50.0 + effects(job, arm, value) + NOISE[seed]) / 100


def _default_effects(job, arm, value):
    if arm == "s1b":
        return {1.0: 3.0, 0.3: 5.0, 0.1: 6.0, 0.03: 5.5, 0.01: 3.0, 0.003: 1.0, 0.001: 0.0,
                3e-4: -2.0}.get(value, -5.0)
    if arm == "single":
        return {1e-10: 2.0}.get(value, 1.0)
    if arm == "adamw":
        return {1e-3: 4.0}.get(value, 2.0)
    return 0.0   # reference


def write_set(tmp: Path, effects=_default_effects, *, drop=None, bridge_ok=True, dirty=None,
              commit_of=None, crash=None, extra=None) -> Path:
    for seed in range(5):
        for job in ("ekfac", "tekfac", "adamw"):
            if drop == (seed, job):
                continue
            cells = {}
            for key in dec._planned(job):
                _, arm, v = key.split("|")
                value = float(v)
                a = _acc(job, arm, value, seed, effects)
                cells[key] = {"arm": arm, "value": value, "test_acc": a, "epoch_val_acc": [a]}
            if crash == (seed, job):
                cells[next(iter(cells))]["error"] = "RuntimeError: singular"
            prov = dict(PROV, git_dirty=dirty == (seed, job),
                        git_commit=(commit_of or {}).get((seed, job), "abc"))
            f = {"cells": cells, "provenance": prov, "problems": [],
                 "bridge": {"reproduced": bridge_ok} if job != "adamw" and seed == 0 else None}
            (tmp / f"e20_{drv.MODEL}_s{seed}_{job}.json").write_text(json.dumps(f))
            if extra and job in extra:
                ext_cells = {}
                for value in extra[job]:
                    arm = "adamw" if job == "adamw" else "s1b"
                    key = f"{job}|{arm}|{value:g}"
                    a = _acc(job, arm, value, seed, effects)
                    ext_cells[key] = {"arm": arm, "value": value, "test_acc": a,
                                      "epoch_val_acc": [a]}
                (tmp / f"e20_{drv.MODEL}_s{seed}_{job}_ext.json").write_text(
                    json.dumps({"cells": ext_cells, "provenance": prov, "problems": []}))
    return tmp


# --------------------------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------------------------


def test_a_clear_set_gives_the_expected_verdicts(tmp_path: Path) -> None:
    res = dec.decide(write_set(tmp_path))
    assert res["valid"]
    assert res["verdict_per_layer"] == "confirmed"
    assert res["verdict_transfer"] == "transfers"
    r = res["modes"]["ekfac"]
    assert r["rule1_helps"]["verdict"] == "win" and r["rule1_helps"]["mean"] == pytest.approx(6.0)
    assert r["rule3_s1b_vs_single_transferred"]["mean"] == pytest.approx(4.0)
    assert r["rule4_s1b_vs_single_tuned"]["tau"] == 0.1
    assert r["rule5_s1b_vs_adamw_tuned"]["adamw_lr"] == 1e-3
    assert r["rule5_s1b_vs_adamw_tuned"]["mean"] == pytest.approx(2.0)


def test_a_constant_offset_is_a_win_but_a_noisy_zero_is_a_tie() -> None:
    assert dec.paired([1.0, 1.1, 0.9, 1.05, 0.95], [0.0] * 5)["verdict"] == "win"
    assert dec.paired([0.3, -0.3, 0.2, -0.2, 0.0], [0.0] * 5)["verdict"] == "tie"
    assert dec.paired([-1.0, -1.1, -0.9, -1.05, -0.95], [0.0] * 5)["verdict"] == "loss"
    # 1.76 standard errors above zero: a tie at the pre-registered 2, a win at any laxer threshold
    assert dec.paired([1.0, -0.2, 0.9, -0.1, 0.6], [0.0] * 5)["verdict"] == "tie"


@pytest.mark.parametrize("gap, verdict", [(0.05, "transfers"), (0.3, "does not transfer")])
def test_the_plateau_is_one_standard_error_wide(tmp_path: Path, gap: float, verdict: str) -> None:
    """The per-seed wobble has a standard error of about 0.09 points: tau = 0.1 at 0.05 below the
    best (0.55 SE) is inside the plateau, at 0.3 below (3.3 SE) it is outside."""
    def eff(job, arm, value):
        if arm == "s1b":
            return {0.03: 8.0, 0.1: 8.0 - gap}.get(value, 2.0)
        return _default_effects(job, arm, value)
    res = dec.decide(write_set(tmp_path, eff))
    assert res["modes"]["ekfac"]["rule2_transfer"]["verdict"] == verdict


def test_no_per_layer_advantage_is_refuted(tmp_path: Path) -> None:
    def eff(job, arm, value):
        return 2.0 if arm in ("s1b", "single") else _default_effects(job, arm, value)
    res = dec.decide(write_set(tmp_path, eff))
    assert res["verdict_per_layer"] == "refuted"


def test_tau_outside_the_plateau_does_not_transfer(tmp_path: Path) -> None:
    def eff(job, arm, value):
        if arm == "s1b":
            return {0.001: 8.0, 0.003: 7.95}.get(value, 2.0)
        return _default_effects(job, arm, value)
    res = dec.decide(write_set(tmp_path, eff))
    assert res["modes"]["ekfac"]["rule2_transfer"]["verdict"] == "does not transfer"
    assert res["verdict_transfer"] == "fails to transfer"


def test_best_tau_at_an_edge_asks_for_the_extension_then_reads_it(tmp_path: Path) -> None:
    def eff(job, arm, value):
        if arm == "s1b":
            return {3e-4: 9.0, 1e-4: 10.0, 3e-5: 7.0}.get(value, 2.0)
        return _default_effects(job, arm, value)
    res = dec.decide(write_set(tmp_path, eff))
    assert res["modes"]["ekfac"]["rule2_transfer"]["verdict"] == "needs the pre-registered extension"
    # with the extension the best (1e-4) is interior, and 0.1 is not in its plateau
    res = dec.decide(write_set(tmp_path, eff, extra={"ekfac": [1e-4, 3e-5],
                                                     "tekfac": [1e-4, 3e-5]}))
    assert res["modes"]["ekfac"]["rule2_transfer"]["verdict"] == "does not transfer"


def test_an_extension_still_at_the_edge_is_unresolved(tmp_path: Path) -> None:
    def eff(job, arm, value):
        return {3e-5: 12.0}.get(value, 2.0) if arm == "s1b" else _default_effects(job, arm, value)
    res = dec.decide(write_set(tmp_path, eff, extra={"ekfac": [1e-4, 3e-5],
                                                     "tekfac": [1e-4, 3e-5]}))
    assert res["modes"]["ekfac"]["rule2_transfer"]["verdict"] == "unresolved"


def test_adamw_at_the_top_of_its_grid_is_flagged(tmp_path: Path) -> None:
    def eff(job, arm, value):
        return {3e-2: 9.0}.get(value, 2.0) if arm == "adamw" else _default_effects(job, arm, value)
    res = dec.decide(write_set(tmp_path, eff))
    assert res["adamw"]["at_edge"] and res["adamw"]["chosen_lr"] == 3e-2


# --------------------------------------------------------------------------------------------
# No verdict on an invalid set
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs, needle", [
    (dict(drop=(3, "tekfac")), "missing file"),
    (dict(bridge_ok=False), "bridge"),
    (dict(dirty=(2, "adamw")), "dirty"),
    (dict(commit_of={(4, "ekfac"): "def"}), "more than one git commit"),
    (dict(crash=(1, "ekfac")), "crashed"),
])
def test_an_invalid_set_gives_no_verdict(tmp_path: Path, kwargs, needle: str) -> None:
    res = dec.decide(write_set(tmp_path, **kwargs))
    assert not res["valid"]
    assert any(needle in p for p in res["problems"])
    assert "verdict_per_layer" not in res


# --------------------------------------------------------------------------------------------
# The driver's constants are the pre-registered ones, and E15's
# --------------------------------------------------------------------------------------------


def test_grids_and_transferred_values_are_e15s() -> None:
    assert drv.TAU_GRID == e15.TAU_GRID["ekfac"] == e15.TAU_GRID["tekfac"]
    assert drv.SINGLE_GRID == e15.SINGLE_GRID[("vit_micro_cifar", "ekfac")]
    assert drv.TAU_TRANSFERRED in drv.TAU_GRID and drv.LAMBDA_TRANSFERRED in drv.SINGLE_GRID
    # the geometric mean of E15's four validation-selected single lambdas, rounded to the grid
    import math
    g = math.exp(sum(map(math.log, [1e-10, 3e-11, 1e-10, 1e-10])) / 4)
    assert min(drv.SINGLE_GRID, key=lambda v: abs(math.log(v / g))) == drv.LAMBDA_TRANSFERRED


def test_held_configuration_is_built_with_e15s_arithmetic() -> None:
    """The ViT-S Fisher cells use E15's held configuration; checked on vit_micro_cifar, where
    E15's own planner can build the same cell."""
    from benchmarks.common.runner import discover_benchmarks
    from fisher_ref.experiments.e5_unhooked_freeze_control import unhooked_parameter_names
    bench = discover_benchmarks()["vit_micro_cifar"]
    hp = bench.hparams
    saved = e15.ARMS, e15.GRID_LIMIT
    e15.ARMS, e15.GRID_LIMIT = ["s1b"], None
    try:
        (theirs,) = [c for c in e15.plan_cells("vit_micro_cifar", "ekfac", hp.lr / hp.lam, hp.lam,
                                               hp.lr, hp.weight_decay, hp.decoupled_wd,
                                               unhooked_parameter_names(bench))
                     if c["value"] == 0.1]
    finally:
        e15.ARMS, e15.GRID_LIMIT = saved
    held = drv._held(bench)
    assert (held["lr"], held["wd"], held["freeze"]) == (theirs["lr"], theirs["wd"], theirs["freeze"])


def test_bridge_values_are_e15s_stored_cells() -> None:
    path = ROOT / "fisher_ref/outputs/e15_layer_damping_vit_micro_cifar_s0.json"
    if not path.exists():
        pytest.skip("E15's outputs are not in this checkout")
    cells = json.load(open(path))["cells"]
    for mode, (acc, move) in drv.BRIDGE.items():
        c = cells[f"{mode}|s1b|0.1"]
        assert (c["test_acc"], c["theta_move_from_init"]) == (acc, move)
