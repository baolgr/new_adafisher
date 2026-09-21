"""E16, fourth amendment: calibrate ResNet-50 before its exploratory E16 runs.

``resnet50_cifar`` has never run under the E protocol (batch 32, lambda ~ 1e-11), so the lambda
grid E16's add arm needs is not known for it, and its per-run cost is only estimated. This driver
measures both, before any clip result exists (``docs/reports/plan_floor_clip.md`` §12):

  scan0..scan4   the add arm at seed 0, 15 epochs, E16's exact protocol (``norm_exact_rescaling``
                 on, E16's lr = cap * lambda), at the ten values of E14's ResNet-20 window,
                 1e-8 .. 3e-13, two per part. This is the only accuracy this driver records.
  timing         two epochs each of add (at 1e-11) and of the three clip arms at q = 0.7, as E16
                 would run them. Only wall time and step count are kept: no accuracy of a clip arm
                 on ResNet-50 is seen before its grid is fixed.

``E16C_SUMMARIZE=1`` reads the twelve files and prints the pre-registered grid: the five lattice
values (1 and 3 times a power of ten) centred on the geometric mean of the two modes' best-
validation lambda, snapped to the lattice (``grid_from_scan``). It also prints the measured seconds
per step of each arm and the projected cost of ResNet-50's reduced E16.

Two settings differ from the benchmark's ``HParams``, both recorded in every cell's ``overrides``:
``fisher_batch_samples=None`` (the benchmark's 32 is refused together with ``norm_exact_rescaling`` on
a network with BatchNorm, and at batch 32 it caps nothing: the estimator is the same), and
``norm_exact_rescaling=True``, as in every E16 arm. ``conv_sua=True`` stays, as the benchmark sets it.

Environment variables::

    E16_MODE        ekfac or tekfac (required unless summarizing)
    E16C_PART       scan0 .. scan4, or timing
    E16C_DIR        where the files go (default: fisher_ref/outputs)
    E16C_SMOKE      "1" allows non-production settings (a local smoke)
    E16C_SUMMARIZE  "1": read the files and print the grid and the costs instead of running
    WARMUP_SGD_*    as elsewhere; WARMUP_SGD_EPOCHS must be 15 for a scan part, 2 for timing
"""
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.setdefault("E16_MODEL", "resnet50_cifar")
os.environ.setdefault("E16_SEED", "0")

import fisher_ref.experiments.e16_floor_clip as e16  # noqa: E402
from benchmarks.common.runner import discover_benchmarks  # noqa: E402

MODEL = "resnet50_cifar"
MODES = ("ekfac", "tekfac")
SMOKE = os.environ.get("E16C_SMOKE", "0") == "1"
PART = os.environ.get("E16C_PART", "")
DIR = Path(os.environ.get("E16C_DIR", str(ROOT / "fisher_ref/outputs")))
SUMMARIZE = os.environ.get("E16C_SUMMARIZE", "0") == "1"

# E14's ResNet-20 window: every value it ran, half-decade spacing, 1e-8 down to 3e-13.
SCAN_GRID = [1e-8, 3e-9, 1e-9, 3e-10, 1e-10, 3e-11, 1e-11, 3e-12, 1e-12, 3e-13]
PARTS = {f"scan{i}": SCAN_GRID[2 * i:2 * i + 2] for i in range(5)}
TIMING_LAMBDA, TIMING_Q = 1e-11, 0.7
TIMING_ARMS = ("add", "clip", "clipema", "clipfixed")
EPOCHS_OF = {**{p: 15 for p in PARTS}, "timing": 2}
OVERRIDES = {"norm_exact_rescaling": True, "fisher_batch_samples": None}
STEPS_PER_EPOCH = 1406   # 45 000 / 32, the loader drops the last incomplete batch
# ResNet-50's exploratory E16, pre-registered with this driver: seeds 0-2, both modes, add on the
# five-value grid, and the three clip arms at these q (plus the seed-0 checks).
REDUCED_Q = [0.95, 0.7, 0.3]
REDUCED_SEEDS = [0, 1, 2]


def lattice(lo_exp: int = -16, hi_exp: int = -4) -> List[float]:
    """1 and 3 times every power of ten in range, descending: the half-decade lattice E-series grids
    live on."""
    vals = [m * 10.0 ** k for k in range(lo_exp, hi_exp + 1) for m in (1, 3)]
    return sorted(vals, reverse=True)


def snap(x: float) -> float:
    """The lattice value nearest to x in log10 (a tie goes to the larger value)."""
    lat = lattice()
    return min(lat, key=lambda v: (abs(math.log10(v) - math.log10(x)), -v))


def grid_from_scan(best_by_mode: Dict[str, float]) -> List[float]:
    """The pre-registered rule: five lattice values centred on the geometric mean of the modes'
    best-validation lambdas, snapped to the lattice. Descending, as E16's grids are."""
    logs = [math.log10(v) for v in best_by_mode.values()]
    centre = snap(10 ** (sum(logs) / len(logs)))
    lat = lattice()
    i = lat.index(centre)
    return lat[i - 2:i + 3]


def plan(part: str, bench) -> List[Dict[str, Any]]:
    hp = bench.hparams
    if part in PARTS:
        return [dict(arm="add", value=lam, lam=lam, lr=hp.lr * (lam / hp.lam), wd=hp.weight_decay,
                     freeze=None, overrides=dict(OVERRIDES)) for lam in PARTS[part]]
    if part != "timing":
        raise SystemExit(f"E16C_PART must be one of {[*PARTS, 'timing']}; got {part!r}")
    clip_wd = 0.0 if hp.decoupled_wd else hp.weight_decay
    cells = [dict(arm="add", value=TIMING_LAMBDA, lam=TIMING_LAMBDA,
                  lr=hp.lr * (TIMING_LAMBDA / hp.lam), wd=hp.weight_decay, freeze=None,
                  overrides=dict(OVERRIDES))]
    for arm in TIMING_ARMS[1:]:
        cells.append(dict(arm=arm, value=TIMING_Q, lam=hp.lam, lr=hp.lr, wd=clip_wd, freeze=[],
                          overrides={"rescale_form": "clip", "clip_fraction": TIMING_Q,
                                     **e16.CLIP_ARMS[arm], **OVERRIDES}))
    return cells


def out_path(mode: str, part: str) -> Path:
    return DIR / f"e16c_{MODEL}_{mode}_{part}.json"


def check_settings() -> None:
    problems = []
    if e16.MODEL != MODEL or e16.SEED != 0:
        problems.append(f"E16_MODEL/E16_SEED = {e16.MODEL}/{e16.SEED}")
    if e16.BATCH != 32 or e16.TRAIN_SUBSET is not None or e16.CALIBRATE_AT != 2000:
        problems.append(f"batch {e16.BATCH}, subset {e16.TRAIN_SUBSET}, calibrate {e16.CALIBRATE_AT}")
    if e16.NUM_WORKERS != e16.PRODUCTION_WORKERS:
        problems.append(f"workers {e16.NUM_WORKERS}")
    if PART in EPOCHS_OF and e16.EPOCHS != EPOCHS_OF[PART]:
        problems.append(f"WARMUP_SGD_EPOCHS={e16.EPOCHS} for {PART} (needs {EPOCHS_OF[PART]})")
    if problems and not SMOKE:
        raise SystemExit("E16C: non-production settings without E16C_SMOKE=1: " + "; ".join(problems))
    if e16.MODE not in MODES:
        raise SystemExit(f"E16C: E16_MODE must be one of {MODES}; got {e16.MODE!r}")


def run() -> None:
    check_settings()
    bench = discover_benchmarks()[MODEL]
    cells = plan(PART, bench)
    prov = e16.provenance()
    print(f"E16C | {MODEL} | mode={e16.MODE} | part={PART} | {len(cells)} cells | "
          f"epochs={e16.EPOCHS} | batch={e16.BATCH} | smoke={SMOKE}\n"
          f"torch {prov['torch']} gpu {prov['gpu']} commit {prov['git_commit']} "
          f"dirty={prov['git_dirty']}", flush=True)
    results: Dict[str, Any] = {"model": MODEL, "mode": e16.MODE, "part": PART,
                               "epochs": e16.EPOCHS, "batch_size": e16.BATCH, "smoke": SMOKE,
                               "provenance": prov, "cells": {}}
    for cell in cells:
        cell = dict(cell)
        arm, value = cell.pop("arm"), cell.pop("value")
        run_ = e16.safe_run_cell(bench, **cell)
        steps = run_.get("n_steps") or 0
        wall = run_.get("wall_s")
        record = {"arm": arm, "value": value, "lam": cell["lam"], "lr": cell["lr"],
                  "wd": cell["wd"], "overrides": cell["overrides"], "wall_s": wall,
                  "n_steps": steps, "s_per_step": wall / steps if wall and steps else None,
                  "crashed": bool(run_.get("crashed")), "error": run_.get("error")}
        if PART == "timing":
            print(f"  {e16.MODE:6s} {arm:9s} {value:9.3g}  {steps} steps  "
                  f"{record['s_per_step'] or float('nan'):.4f} s/step", flush=True)
        else:
            record.update({k: run_.get(k) for k in ("test_acc", "test_loss", "epoch_val_acc",
                                                    "epoch_train_loss", "theta_move_from_init",
                                                    "log")})
            print(f"  {e16.MODE:6s} add {value:9.3g}  val={100 * e16._final_val(run_):6.2f}  "
                  f"({wall or float('nan'):.0f}s)", flush=True)
        results["cells"][f"{e16.MODE}|{arm}|{value:g}"] = record
        e16._write(out_path(e16.MODE, PART), results)


def summarize() -> Dict[str, Any]:
    report: Dict[str, Any] = {"problems": []}
    best: Dict[str, float] = {}
    commits = set()
    for mode in MODES:
        vals: Dict[float, float] = {}
        for part in PARTS:
            path = out_path(mode, part)
            if not path.exists():
                report["problems"].append(f"{path.name}: missing")
                continue
            f = json.load(open(path))
            commits.add(f["provenance"].get("git_commit"))
            if f.get("smoke") or f["provenance"].get("git_dirty"):
                report["problems"].append(f"{path.name}: smoke or dirty tree")
            for c in f["cells"].values():
                v = e16._final_val(c)
                if c.get("crashed") or not math.isfinite(v):
                    report["problems"].append(f"{path.name}: {c['value']:g} unusable")
                    continue
                vals[float(c["value"])] = v
        report[f"{mode}|val"] = {f"{k:g}": 100 * v for k, v in sorted(vals.items(), reverse=True)}
        print(f"{mode}: final validation accuracy by lambda: "
              + ", ".join(f"{k:g} {100 * v:.2f}" for k, v in sorted(vals.items(), reverse=True)))
        if len(vals) == len(SCAN_GRID):
            best[mode] = max(vals, key=lambda k: vals[k])
            if best[mode] in (SCAN_GRID[0], SCAN_GRID[-1]):
                report["problems"].append(f"{mode}: best lambda {best[mode]:g} is at the scan's "
                                          "edge; the centre may lie outside it")
    if len(commits) > 1:
        report["problems"].append(f"the scan files come from {len(commits)} commits")
    report["best"] = {m: f"{v:g}" for m, v in best.items()}
    if len(best) == len(MODES):
        report["grid"] = [float(f"{v:.0e}") for v in grid_from_scan(best)]
        print(f"best on validation: {report['best']} -> pre-registered grid {report['grid']}")
    seconds: Dict[str, List[float]] = {}
    for mode in MODES:
        path = out_path(mode, "timing")
        if not path.exists():
            report["problems"].append(f"{path.name}: missing")
            continue
        for c in json.load(open(path))["cells"].values():
            if c.get("s_per_step"):
                seconds.setdefault(c["arm"], []).append(c["s_per_step"])
    # The slower mode of the two, times a full 15-epoch run.
    per_run = {arm: max(v) * 15 * STEPS_PER_EPOCH for arm, v in seconds.items()}
    report["projected_run_s"] = per_run
    if set(per_run) == set(TIMING_ARMS):
        per_job = 5 * per_run["add"] + len(REDUCED_Q) * sum(per_run[a] for a in TIMING_ARMS[1:])
        report["projected_seed_mode_h"] = per_job / 3600
        report["projected_total_h"] = per_job * len(REDUCED_SEEDS) * len(MODES) / 3600
        print("projected 15-epoch run (s): "
              + ", ".join(f"{a} {s:.0f}" for a, s in per_run.items())
              + f"\nreduced E16 on ResNet-50: {per_job / 3600:.1f} h per (seed, mode) without the "
              f"seed-0 checks, {report['projected_total_h']:.0f} h of 1g for "
              f"{len(REDUCED_SEEDS)} seeds x 2 modes")
    for p in report["problems"]:
        print(f"  - {p}")
    return report


def main() -> None:
    if SUMMARIZE:
        DIR.mkdir(parents=True, exist_ok=True)
        report = summarize()
        with open(DIR / f"e16c_{MODEL}_summary.json", "w") as f:
            json.dump(report, f, indent=1)
        sys.exit(2 if report["problems"] else 0)
    run()


if __name__ == "__main__":
    main()
