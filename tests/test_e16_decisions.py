"""``fisher_ref/experiments/e16_decisions.py``: E16's pre-registered rules on synthetic runs.

The script is what turns thirty jobs into verdicts, so what is pinned here is that it cannot turn
an incomplete or invalid set of jobs into a verdict silently: a crashed or absent cell, a smoke
file, a cell run without the corrected normalisation statistic, a clipfixed cell that never froze,
a failed determinism or inertness check makes the verdict ``incomplete`` (or the file invalid) and
the script exit with status 2, and the lr control qualifies the main arm's verdict. On complete data
it reproduces the rules' verdicts on differences planted by construction.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "fisher_ref/experiments/e16_decisions.py"
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fisher_ref.experiments.e16_floor_clip import (  # noqa: E402
    CHECK_LAMBDA,
    CLIP_ARMS,
    CLIP_GRID,
    CLIPLR_FACTORS,
    LAMBDA_GRID,
    NO_HOOKED_NORM,
)

MODELS = ["cnn_gn_cifar", "vit_micro_cifar", "cct_2_3x2_cifar"]
MODES = ["ekfac", "tekfac"]
SEEDS = [0, 1, 2, 3, 4]
NOISE = [0.004, -0.003, 0.001, -0.002, 0.000]     # per-seed, shared by every cell of a seed
JITTER = [0.0005, -0.0004, 0.0002, -0.0001, 0.0]  # per-seed, per-family differences


def _cell(acc: float, **extra) -> dict:
    return {"test_acc": acc, "epoch_val_acc": [acc - 0.05, acc], "test_loss": 1.0,
            "theta_move_from_init": 1.0, "n_steps": 21090, **extra}


def _acc(family: str, value: float, seed: int, planted: dict, grid: list) -> float:
    # The best value of every family is its second grid value; the others are 1 point below.
    penalty = 0.0 if value == grid[1] else 0.01
    jitter = JITTER[seed] if family != "add" else 0.0
    return 0.60 + NOISE[seed] + planted.get(family, 0.0) - penalty + jitter


def _write(tmp: Path, planted: dict, mutate=None) -> None:
    exact = {"norm_exact_rescaling": True}
    for model in MODELS:
        lam_grid = LAMBDA_GRID[model]
        for seed in SEEDS:
            for mode in MODES:
                cells = {}

                def put(arm, value, acc, overrides, **extra):
                    cells[f"{mode}|{arm}|{value:g}"] = dict(
                        _cell(acc, **extra), mode=mode, arm=arm, value=value,
                        overrides=overrides)

                for v in lam_grid:
                    put("add", v, _acc("add", v, seed, planted, lam_grid), dict(exact))
                    put("floor", v, _acc("floor", v, seed, planted, lam_grid),
                        {"rescale_form": "floor", **exact})
                for arm in CLIP_ARMS:
                    fixed = arm == "clipfixed"
                    for q in CLIP_GRID:
                        extra = ({"log": [{"step": 21000, "layers": {"head": {"calibrated": 1.0}}}]}
                                 if fixed else {})
                        put(arm, q, _acc(arm, q, seed, planted, CLIP_GRID),
                            {"rescale_form": "clip", "clip_threshold":
                             {"clipema": "ema", "clip": "quantile", "clipfixed": "fixed"}[arm],
                             **exact}, **extra)
                for f in CLIPLR_FACTORS:
                    acc = (_acc("clipema", 0.7, seed, planted, CLIP_GRID)
                           + planted.get("cliplr", 0.0) + 0.0003 * seed)
                    put("cliplr", f, acc, {"rescale_form": "clip", "clip_threshold": "ema",
                                           **exact})
                checks = {}
                if seed == 0:
                    lam = CHECK_LAMBDA[(model, mode)]
                    put("dupcheck", lam, 0.6, dict(exact))
                    put("repro", lam, 0.6, {})
                    checks = {"determinism": {"identical": True},
                              "repro_vs_e14": {"identical": True}}
                    if model in NO_HOOKED_NORM:
                        checks["norm_exact_inert"] = {"identical": True}
                e16 = {"model": model, "seed": seed, "mode": mode, "epochs": 15,
                       "batch_size": 32, "grid_limit": None, "train_subset": None,
                       "calibrate_at": 2000, "missing_shards": [], "checks": checks,
                       "provenance": [{"smoke": False, "git_dirty": False,
                                       "git_commit": "abc123"}] * 3,
                       "cells": cells}
                if mutate is not None:
                    mutate(model, seed, mode, e16)
                (tmp / f"e16_floor_clip_{model}_s{seed}_{mode}.json").write_text(json.dumps(e16))


def _run(tmp: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "E16_DIR": str(tmp), "E16_OUT": str(tmp / "decisions.json"),
           "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}"}
    env.pop("E16_ALLOW_INCOMPLETE", None)
    return subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True,
                          timeout=120)


def test_complete_data_gives_the_planted_verdicts(tmp_path: Path) -> None:
    _write(tmp_path, planted={"clipema": 0.02, "clip": -0.02})
    res = _run(tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    report = json.loads((tmp_path / "decisions.json").read_text())
    assert report["verdict_clipema"].startswith("better than the E14 fix")
    assert report["verdict_clip"].startswith("worse than the E14 fix")
    assert report["verdict_floor"].startswith("equivalent")
    assert report["problems"] == []
    assert report["transfer_clipema"] == [CLIP_GRID[1]]


@pytest.mark.parametrize("defect", ["crashed", "smoke", "absent", "uncalibrated",
                                    "shipped_estimator", "nondeterministic", "not_inert"])
def test_an_invalid_set_of_jobs_is_never_read_as_a_verdict(tmp_path: Path, defect: str) -> None:
    def mutate(model, seed, mode, e16):
        key = f"tekfac|clipfixed|{CLIP_GRID[1]:g}"
        if defect in ("crashed", "absent", "uncalibrated", "shipped_estimator", "smoke"):
            if (model, seed, mode) != ("vit_micro_cifar", 3, "tekfac"):
                return
        if defect == "crashed":
            e16["cells"][key].update(crashed=True, test_acc=None, epoch_val_acc=[])
        elif defect == "smoke":
            e16["provenance"][1] = dict(e16["provenance"][1], smoke=True)
        elif defect == "absent":
            del e16["cells"][key]
        elif defect == "uncalibrated":
            e16["cells"][key]["log"][-1]["layers"]["head"]["calibrated"] = 0.0
        elif defect == "shipped_estimator":
            e16["cells"][key]["overrides"].pop("norm_exact_rescaling")
        elif defect == "nondeterministic" and seed == 0 and model == "cct_2_3x2_cifar":
            e16["checks"]["determinism"] = {"identical": False, "mismatched": ["test_acc"]}
        elif defect == "not_inert" and seed == 0 and model == "cnn_gn_cifar":
            e16["checks"]["norm_exact_inert"] = {"identical": False}

    _write(tmp_path, planted={}, mutate=mutate)
    res = _run(tmp_path)
    assert res.returncode == 2, res.stdout + res.stderr
    assert "NOT A VALID E16 RESULT" in res.stdout
    report = json.loads((tmp_path / "decisions.json").read_text())
    if defect in ("crashed", "absent"):
        assert report["verdict_clipfixed"].startswith("incomplete")


def test_the_repro_diagnostic_does_not_vote(tmp_path: Path) -> None:
    def mutate(model, seed, mode, e16):
        if seed == 0:
            e16["checks"]["repro_vs_e14"] = {"identical": False, "mismatched": ["test_acc"]}

    _write(tmp_path, planted={}, mutate=mutate)
    res = _run(tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "repro_vs_e14" in res.stdout


def test_the_lr_control_qualifies_the_main_arm(tmp_path: Path) -> None:
    _write(tmp_path, planted={"clipema": 0.02, "cliplr": 0.02})
    res = _run(tmp_path)
    assert res.returncode == 0, res.stdout + res.stderr
    report = json.loads((tmp_path / "decisions.json").read_text())
    assert "LIMITED BY LR" in report["verdict_clipema"]
