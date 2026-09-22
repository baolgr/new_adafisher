"""E19: the driver's plan of cells and the decision script's rules, on synthetic runs.

What is pinned here is what E19's reading depends on, and cannot be checked after the runs:
- the ``bridge`` cell is built exactly as E16's driver builds its ``add`` cell at the same lambda
  (rule 0a compares their outcomes bit for bit, so their settings must be equal first);
- every other cell has S1-b's settings (the cap held per layer, the unhooked parameters frozen, no
  decoupled decay except ``wdctrl``'s fixed rate), the corrected normalisation statistic, and on
  ResNet-50 E16's ``fisher_batch_samples=None``;
- the grids, seeds, shard counts and largest shards are the pre-registered ones;
- the decision script never turns an incomplete, crashed or mis-configured set of runs into a
  verdict, and applies rules 0b, 2, 3 and 4 as written on differences planted by construction.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "fisher_ref/experiments/e19_layer_damping_transfer.py"
SUBMIT = ROOT / "fisher_ref/slurm/e19_submit.sh"
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks.common.runner import discover_benchmarks  # noqa: E402
from fisher_ref.experiments import e16_floor_clip as e16d  # noqa: E402
from fisher_ref.experiments import e19_decisions as dec  # noqa: E402
from fisher_ref.experiments import e19_layer_damping_transfer as e19d  # noqa: E402
from fisher_ref.experiments.e5_unhooked_freeze_control import unhooked_parameter_names  # noqa: E402

NETWORKS = list(e19d.NETWORKS)
MODES = list(e19d.MODES)


def _plan(model: str, mode: str, seed: int) -> list:
    hp, frozen = dec._network(model)
    return e19d.plan_cells(model, mode, seed, hp.lam, hp.lr, hp.weight_decay, hp.decoupled_wd,
                           list(frozen))


def _submit_table() -> dict:
    """``{network: (nshards, --time)}`` as e19_submit.sh sets them."""
    rows = re.findall(r"(\w+)\)\s+short=\w+; limit=([\d:]+); nshards=(\d+)", SUBMIT.read_text())
    return {model: (int(n), limit) for model, limit, n in rows}


# --------------------------------------------------------------------------------------------
# The plan of cells
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", NETWORKS)
@pytest.mark.parametrize("mode", MODES)
def test_cell_counts_are_the_pre_registered_ones(model: str, mode: str) -> None:
    per_arm = {"cct_2_3x2_cifar": {"held": 1, "s1b": 8, "netadapt": 8, "wdctrl": 1},
               "resnet20_cifar": {"held": 1, "s1b": 8, "netadapt": 8},
               "resnet50_cifar": {"held": 1, "s1b": 5}}[model]
    for seed in e19d.SEEDS[model]:
        cells = _plan(model, mode, seed)
        counts: dict = {}
        for c in cells:
            counts[c["arm"]] = counts.get(c["arm"], 0) + 1
        expected = dict(per_arm, **({"bridge": 1} if seed == 0 else {}))
        assert counts == expected
        assert len({(c["arm"], c["value"]) for c in cells}) == len(cells)   # no duplicate key


def test_seeds_and_grids_are_the_pre_registered_ones() -> None:
    assert e19d.SEEDS == {"cct_2_3x2_cifar": (0, 1, 2, 3, 4), "resnet20_cifar": (0, 1, 2, 3, 4),
                          "resnet50_cifar": (0, 1, 2)}
    eight = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4]
    assert e19d.TAU_GRID == {"cct_2_3x2_cifar": eight, "resnet20_cifar": eight,
                             "resnet50_cifar": [1.0, 0.3, 0.1, 0.03, 0.01]}
    assert e19d.NETADAPT_NETWORKS == ("cct_2_3x2_cifar", "resnet20_cifar")
    assert e19d.WDCTRL_TAU == 0.1 and 0.1 in e19d.TAU_GRID["resnet50_cifar"]


@pytest.mark.parametrize("model", NETWORKS)
@pytest.mark.parametrize("mode", MODES)
def test_bridge_is_built_exactly_as_e16_builds_its_add_cell(
        model: str, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rule 0a compares outcomes bit for bit; that only means something if the inputs are equal."""
    monkeypatch.setattr(e16d, "MODE", mode)
    monkeypatch.setattr(e16d, "SEED", 0)
    bench = discover_benchmarks()[model]
    hp = bench.hparams
    lam = e16d.CHECK_LAMBDA[(model, mode)]
    add = next(c for c in e16d.plan_cells(model, hp.lam, hp.lr, hp.weight_decay, hp.decoupled_wd,
                                          unhooked_parameter_names(bench))
               if c["arm"] == "add" and c["value"] == lam)
    bridge = next(c for c in _plan(model, mode, 0) if c["arm"] == "bridge")
    for key in ("value", "lam", "lr", "wd", "freeze", "overrides"):
        assert bridge[key] == add[key], key
    assert e19d.BRIDGE_LAMBDA[(model, mode)] == lam


@pytest.mark.parametrize("model", NETWORKS)
@pytest.mark.parametrize("mode", MODES)
def test_every_other_cell_has_s1b_settings(model: str, mode: str) -> None:
    hp, frozen = dec._network(model)
    cap = hp.lr / hp.lam
    for c in _plan(model, mode, 0):
        o = c["overrides"]
        assert o["norm_exact_rescaling"] is True
        if model == "resnet50_cifar":
            assert "fisher_batch_samples" in o and o["fisher_batch_samples"] is None
        if c["arm"] == "bridge":
            continue
        assert o["hold_cap"] is True
        assert c["lr"] == cap
        assert c["freeze"] == list(frozen)
        if c["arm"] == "wdctrl":
            assert hp.decoupled_wd
            # E15's convention: lr * coefficient is the benchmark's base_lr * wd per step.
            assert c["lr"] * c["wd"] == pytest.approx(hp.lr * hp.weight_decay, rel=1e-12)
        else:
            assert c["wd"] == (0.0 if hp.decoupled_wd else hp.weight_decay)
        if c["arm"] == "held":
            assert c["lam"] == c["value"] == e19d.HELD_LAMBDA[(model, mode)]
            assert "damping" not in o
        else:
            assert c["lam"] == hp.lam
            damping = {"s1b": "layer_relative", "wdctrl": "layer_relative",
                       "netadapt": "network_relative"}[c["arm"]]
            assert (o["damping"], o["damping_tau"]) == (damping, c["value"])


def test_the_networks_are_what_the_plan_says() -> None:
    """CCT: pos_embed unhooked and decoupled decay; the ResNets: nothing unhooked, coupled decay,
    and ResNet-50 under SUA. The whole design of the held settings rests on this."""
    hp, frozen = dec._network("cct_2_3x2_cifar")
    assert frozen == ("pos_embed",) and hp.decoupled_wd and hp.weight_decay == 1e-2
    assert hp.lr / hp.lam == pytest.approx(1 / 3)
    for model in ("resnet20_cifar", "resnet50_cifar"):
        hp, frozen = dec._network(model)
        assert frozen == () and not hp.decoupled_wd and hp.weight_decay == 5e-4
    assert dec._network("resnet50_cifar")[0].conv_sua is True


def test_held_lambda_is_on_e16_grid() -> None:
    for (model, _), lam in e19d.HELD_LAMBDA.items():
        assert lam in e16d.LAMBDA_GRID[model]


@pytest.mark.parametrize("model", NETWORKS)
def test_shards_partition_every_job_and_match_the_cost_table(model: str) -> None:
    nshards, _ = _submit_table()[model]
    assert nshards == {"cct_2_3x2_cifar": 2, "resnet20_cifar": 2, "resnet50_cifar": 4}[model]
    largest = 0
    for mode in MODES:
        for seed in e19d.SEEDS[model]:
            cells = _plan(model, mode, seed)
            owners = e19d.shard_of(cells, nshards)
            assert sorted(set(owners)) == list(range(nshards))
            largest = max(largest, max(owners.count(s) for s in range(nshards)))
    assert largest == {"cct_2_3x2_cifar": 10, "resnet20_cifar": 9, "resnet50_cifar": 2}[model]


def test_submit_script_limits_are_the_pre_registered_ones() -> None:
    assert _submit_table() == {"cct_2_3x2_cifar": (2, "01:00:00"),
                               "resnet20_cifar": (2, "01:30:00"),
                               "resnet50_cifar": (4, "02:30:00")}


def _driver(env_extra: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E19_", "WARMUP_SGD_"))}
    env.update({"WARMUP_SGD_NUM_WORKERS": "4", "WARMUP_SGD_EPOCHS": "15",
                "WARMUP_SGD_DEVICE": "cpu", **env_extra})
    return subprocess.run([sys.executable, str(DRIVER)], env=env, capture_output=True, text=True,
                          timeout=300)


def test_driver_refuses_a_smoke_setting_without_e19_smoke() -> None:
    r = _driver({"E19_MODEL": "cct_2_3x2_cifar", "E19_MODE": "ekfac", "E19_SEED": "0",
                 "E19_GRID_LIMIT": "1", "E19_MERGE": "1"})
    assert r.returncode != 0 and "non-production settings" in r.stderr


def test_driver_refuses_a_seed_e16_did_not_run() -> None:
    r = _driver({"E19_MODEL": "resnet50_cifar", "E19_MODE": "ekfac", "E19_SEED": "3",
                 "E19_MERGE": "1"})
    assert r.returncode != 0 and "E19_SEED=3" in r.stderr


# --------------------------------------------------------------------------------------------
# The decision script
# --------------------------------------------------------------------------------------------


def _cell(acc: float, **extra) -> dict:
    return {"test_acc": acc, "epoch_val_acc": [acc - 0.05, acc], "test_loss": 1.0,
            "theta_move_from_init": 1.0, "n_steps": 21090, **extra}


NOISE = [0.004, -0.003, 0.001, -0.002, 0.000]     # per seed, shared by every cell of a seed
# Per seed and per family, so that no paired difference is exactly constant across seeds (a zero
# standard error is E16's "degenerate", not a verdict). None of them moves a mean by more than 0.02
# points.
JITTER = {"add": [0.0] * 5,
          "s1b": [0.0005, -0.0004, 0.0002, -0.0001, 0.0],
          "clip": [-0.0003, 0.0004, 0.0001, -0.0002, 0.0],
          "held": [-0.0002, 0.0003, -0.0004, 0.0001, 0.0002],
          "netadapt": [0.0001, -0.0002, 0.0003, 0.0, -0.0002]}


def _grid_series(mode: str, arm: str, grid: list, best, level: float, seeds, jitter: str) -> dict:
    """``{(mode, arm, v): {seed: cell}}``: ``best`` at ``level``, every other value 1 point lower."""
    out = {}
    for v in grid:
        out[(mode, arm, float(v))] = {
            s: _cell(level + NOISE[s] + JITTER[jitter][s] - (0 if v == best else 0.01))
            for s in seeds}
    return out


def _synthetic(model: str, mode: str, *, s1b: float, clipema: float, held: float,
               s1b_best: float = 0.1) -> tuple:
    """E16's add best at its selected lambda, at 0.80; every other family planted relative to it."""
    seeds = list(e19d.SEEDS[model])
    add_best = e19d.HELD_LAMBDA[(model, mode)]
    e16_data = _grid_series(mode, "add", e16d.LAMBDA_GRID[model], add_best, 0.80, seeds, "add")
    clip_grid = [0.95, 0.7, 0.3]
    for arm in ("clipema", "clip", "clipfixed"):
        e16_data.update(_grid_series(mode, arm, clip_grid, 0.7, 0.80 + clipema, seeds, "clip"))
    e19_data = _grid_series(mode, "s1b", e19d.TAU_GRID[model], s1b_best, 0.80 + s1b, seeds, "s1b")
    if model in e19d.NETADAPT_NETWORKS:
        e19_data.update(_grid_series(mode, "netadapt", e19d.TAU_GRID[model], 0.01, 0.80, seeds,
                                     "netadapt"))
    e19_data[(mode, "held", add_best)] = {
        s: _cell(0.80 + held + NOISE[s] + JITTER["held"][s]) for s in seeds}
    return e16_data, e19_data


@pytest.fixture
def small_clip_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every network reads the same three-value clip grid in these synthetic tests."""
    monkeypatch.setattr(dec.e16, "clip_grid_of", lambda model: [0.95, 0.7, 0.3])
    monkeypatch.setattr(dec.e16, "arms_of", lambda model: ("add", "clipema", "clip", "clipfixed"))


@pytest.mark.usefixtures("small_clip_grid")
def test_read_pair_applies_rules_2_to_5_when_0b_is_a_tie() -> None:
    model, mode = "cct_2_3x2_cifar", "ekfac"
    e16_data, e19_data = _synthetic(model, mode, s1b=0.01, clipema=-0.01, held=0.0)
    problems: list = []
    row = dec.read_pair(model, mode, e16_data, e19_data, None, problems)
    assert problems == []
    assert row["chosen"]["add"] == e19d.HELD_LAMBDA[(model, mode)] and row["chosen"]["s1b"] == 0.1
    assert row["0b_held_minus_add"]["verdict"] == "tie"
    assert row["rule2_partner"] == "add (E16)"
    assert row["rule2_s1b_minus_add"]["verdict"] == "win"
    assert row["rule2_s1b_minus_add"]["mean"] == pytest.approx(1.0, abs=0.02)
    assert row["rule3_s1b_minus_clipema"]["verdict"] == "win"
    assert row["rule3_s1b_minus_clipema"]["mean"] == pytest.approx(2.0, abs=0.02)
    assert row["rule4_plateau"]["tau_0.1_inside"] is True
    assert row["rule5_s1b_minus_netadapt"]["verdict"] == "win"
    assert row["0a_bridge"]["identical"] is False    # no bridge given: reported, decides nothing


@pytest.mark.usefixtures("small_clip_grid")
def test_a_non_tie_0b_moves_rule_2_to_held_and_withholds_rule_3() -> None:
    model, mode = "resnet20_cifar", "tekfac"
    e16_data, e19_data = _synthetic(model, mode, s1b=0.0, clipema=-0.01, held=0.005)
    row = dec.read_pair(model, mode, e16_data, e19_data, None, [])
    assert row["0b_held_minus_add"]["verdict"] == "win"
    assert row["rule2_partner"].startswith("held")
    assert row["rule2_s1b_minus_add"]["mean"] == pytest.approx(-0.5, abs=0.02)
    assert row["rule3_s1b_minus_clipema"]["verdict"].startswith("no verdict")


@pytest.mark.usefixtures("small_clip_grid")
def test_held_at_a_lambda_other_than_e16s_selection_is_a_problem() -> None:
    model, mode = "resnet50_cifar", "ekfac"
    e16_data, e19_data = _synthetic(model, mode, s1b=0.0, clipema=-0.05, held=0.0)
    # Move E16's best add value one step away from the held lambda.
    grid = e16d.LAMBDA_GRID[model]
    other = grid[grid.index(e19d.HELD_LAMBDA[(model, mode)]) - 1]
    e16_data.update(_grid_series(mode, "add", grid, other, 0.80, list(e19d.SEEDS[model]), "add"))
    problems: list = []
    dec.read_pair(model, mode, e16_data, e19_data, None, problems)
    assert any("held ran at" in p for p in problems)


@pytest.mark.usefixtures("small_clip_grid")
def test_a_missing_tau_makes_every_s1b_rule_incomplete() -> None:
    model, mode = "cct_2_3x2_cifar", "tekfac"
    e16_data, e19_data = _synthetic(model, mode, s1b=0.01, clipema=-0.01, held=0.0)
    del e19_data[(mode, "s1b", 3e-4)][4]
    row = dec.read_pair(model, mode, e16_data, e19_data, None, [])
    assert row["chosen"]["s1b"] is None
    assert row["rule2_s1b_minus_add"]["verdict"] == "incomplete"
    assert row["rule3_s1b_minus_clipema"]["verdict"] == "incomplete"
    assert row["rule4_plateau"]["tau_0.1_inside"] is None


@pytest.mark.parametrize("vs, expected", [
    ({}, "incomplete"),
    ({(n, m): "tie" for n in NETWORKS for m in MODES}, "equivalent"),
    ({**{(n, m): "win" for n in NETWORKS[:2] for m in MODES},
      **{(NETWORKS[2], m): "tie" for m in MODES}}, "better"),
    ({**{(n, m): "tie" for n in NETWORKS for m in MODES},
      (NETWORKS[0], "ekfac"): "lose", (NETWORKS[1], "tekfac"): "lose"}, "worse"),
    ({**{(n, m): "win" for n in NETWORKS for m in MODES}, (NETWORKS[2], "ekfac"): "lose"},
     "mixed"),
    ({**{(n, m): "win" for n in NETWORKS for m in MODES}, (NETWORKS[2], "ekfac"): "incomplete"},
     "incomplete"),
])
def test_summary_uses_e16_rule_3_words(vs: dict, expected: str) -> None:
    assert dec.summarise(vs).startswith(expected)


def _write_e19_file(tmp: Path, model: str, seed: int, mode: str, mutate=None) -> None:
    cells = {}
    for c in _plan(model, mode, seed):
        key = e19d.cell_key(mode, c["arm"], c["value"])
        cells[key] = {**_cell(0.8), "mode": mode, "arm": c["arm"], "value": c["value"],
                      **{k: c[k] for k in ("lam", "lr", "wd", "freeze", "overrides")}}
    payload = {"model": model, "seed": seed, "mode": mode, "batch_size": 32, "epochs": 15,
               "arms": ["bridge", "held", "s1b", "netadapt", "wdctrl"], "grid_limit": None,
               "train_subset": None, "missing_shards": [], "absent_cells": [],
               "unplanned_cells": [],
               "provenance": [{"smoke": False, "git_commit": "abc", "git_dirty": False,
                               "num_workers": 4}],
               "cells": cells}
    if mutate:
        mutate(payload)
    (tmp / f"e19_layer_damping_{model}_s{seed}_{mode}.json").write_text(json.dumps(payload))


def _load(tmp: Path, monkeypatch: pytest.MonkeyPatch, mutate=None) -> list:
    model = "cct_2_3x2_cifar"
    monkeypatch.setattr(dec, "DIR", tmp)
    for seed in e19d.SEEDS[model]:
        for mode in MODES:
            _write_e19_file(tmp, model, seed, mode, mutate if (seed, mode) == (1, "ekfac") else None)
    problems: list = []
    commits: set = set()
    data, bridges = dec.load_e19(model, problems, commits)
    assert set(bridges) == set(MODES) and commits == {"abc"}
    return problems


def test_load_accepts_a_complete_valid_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _load(tmp_path, monkeypatch) == []


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["cells"]["ekfac|s1b|0.1"].update(crashed=True, error="boom"), "crashed"),
    (lambda p: p["cells"].pop("ekfac|netadapt|0.003"), "planned cells absent"),
    (lambda p: p["cells"]["ekfac|held|3e-11"].update(wd=0.01), "settings other than planned"),
    (lambda p: p["cells"]["ekfac|s1b|0.3"]["overrides"].update(norm_exact_rescaling=False),
     "settings other than planned"),
    (lambda p: p["provenance"][0].update(smoke=True), "not a smoke"),
    (lambda p: p["provenance"][0].update(git_dirty=True), "clean tree"),
    (lambda p: p.update(grid_limit=1), "full grids"),
    (lambda p: p["provenance"][0].update(num_workers=0), "4 workers"),
])
def test_load_refuses_an_invalid_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      mutate, message: str) -> None:
    problems = _load(tmp_path, monkeypatch, mutate)
    assert any(message in p for p in problems), problems
