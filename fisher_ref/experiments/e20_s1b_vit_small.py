"""E20: E15's per-layer safety constant (S1-b) on ViT-S, a network nobody tuned it on.

``docs/reports/plan_e20_s1b_vit_small.md`` is the specification. This driver implements its stage 1
and nothing else; anything it does differently is a deviation and must be recorded there.
``fisher_ref/experiments/e20_decisions.py`` applies its rules.

**No computation path is rewritten here.** The Fisher cells are built by
``e15_layer_damping.run_cell``, called unmodified, with E15's held configuration (cap held per
layer, unhooked parameters frozen, decoupled decay at the benchmark's rate). The AdamW cells are
built by ``e16_baselines.run_cell``, called unmodified: nothing frozen, the benchmark's decoupled
decay applied by AdamW itself. This driver only chooses which cells to run and records them.

**Jobs.** One job is (seed, job), where ``job`` is:

  ekfac, tekfac   [bridge at seed 0] + reference + 5 single lambda + 8 S1-b tau, on vit_small_cifar
  adamw           6 learning rates on vit_small_cifar
  calib           one shipped reference cell (ekfac, lambda = 3e-3): the only cell E17 allows before
                  its addendum is committed. It exists to time a run.

The **bridge** re-runs E15's stored ``vit_micro_cifar`` S1-b cell at tau = 0.1 in the job's mode,
seed 0, through E15's own cell builder, and stops the job unless it reproduces E15's test accuracy
and distance travelled bit for bit.

**Extension jobs** (pre-registered edge rules 2 and 5): ``E20_VALUES`` lists the extra values, which
must come from ``EXTENSIONS``, and ``E20_TAG`` names the file.

Environment variables::

    E20_SEED      required, 0-4
    E20_JOB       required: ekfac | tekfac | adamw | calib
    E20_ARMS      "netadapt" runs amendment 1's network-wide relative arm instead of the three
                  arms above (same tau grid, same settings); default: reference, single, s1b
    E20_VALUES    extension job only: ':'-separated values from EXTENSIONS (tau, or AdamW lr)
    E20_TAG       extension job only: file suffix, e.g. "ext_hi"
    E20_OUT       output path (default: fisher_ref/outputs/e20_vit_small_cifar_s<seed>_<job>[_<tag>].json)
    E20_SMOKE     "1" allows non-production settings: WARMUP_SGD_EPOCHS/NUM_WORKERS, E20_TRAIN_SUBSET,
                  E20_GRID_LIMIT (first k values of every grid), and skipping the bridge check
"""
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch  # noqa: E402

import fisher_ref.experiments.e15_layer_damping as e15  # noqa: E402
import fisher_ref.experiments.e16_baselines as e16b  # noqa: E402
from benchmarks.common.runner import discover_benchmarks  # noqa: E402
from fisher_ref.experiments.e5_unhooked_freeze_control import unhooked_parameter_names  # noqa: E402
from fisher_ref.experiments.warmup_sgd_baseline import DEVICE, EPOCHS, NUM_WORKERS  # noqa: E402

MODEL = "vit_small_cifar"
BRIDGE_MODEL = "vit_micro_cifar"
BATCH = 32
SEED = int(os.environ["E20_SEED"]) if os.environ.get("E20_SEED") else None
JOB = os.environ.get("E20_JOB", "")
TAG = os.environ.get("E20_TAG", "")
ARMS = os.environ.get("E20_ARMS", "")
SMOKE = os.environ.get("E20_SMOKE", "0") == "1"
TRAIN_SUBSET = int(os.environ["E20_TRAIN_SUBSET"]) if os.environ.get("E20_TRAIN_SUBSET") else None
GRID_LIMIT = int(os.environ["E20_GRID_LIMIT"]) if os.environ.get("E20_GRID_LIMIT") else None
VALUES = [float(v) for v in (os.environ.get("E20_VALUES") or "").replace(",", ":").split(":") if v]

# Fixed by the pre-registration, section 2.
TAU_TRANSFERRED = 0.1
LAMBDA_TRANSFERRED = 1e-10
SINGLE_GRID = [1e-9, 3e-10, 1e-10, 3e-11, 1e-11]
TAU_GRID = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4]
ADAMW_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
# The one extension per edge the rules allow: two more values on the side where the best sits.
EXTENSIONS = {"s1b": [10.0, 3.0, 1e-4, 3e-5], "adamw": [1e-1, 3e-1, 3e-5, 1e-5]}
# E15's stored seed-0 cells the bridge must reproduce (fisher_ref/outputs/e15_layer_damping_
# vit_micro_cifar_s0.json, "<mode>|s1b|0.1"): test accuracy and distance travelled, exactly.
BRIDGE = {"ekfac": (0.6055, 19.733800888061523), "tekfac": (0.6009, 19.903472900390625)}


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def provenance() -> Dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "smoke": SMOKE, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "tf32_matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "tf32_cudnn": torch.backends.cudnn.allow_tf32, "device": str(DEVICE),
        "num_workers": NUM_WORKERS, "epochs": EPOCHS, "batch": BATCH,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(), "host": platform.node(),
    }


def _grid(values: List[float]) -> List[float]:
    return values if GRID_LIMIT is None else values[:GRID_LIMIT]


def _held(bench) -> Dict[str, Any]:
    """E15's held configuration, with E15's own arithmetic (``e15.plan_cells``)."""
    hp = bench.hparams
    cap = hp.lr / hp.lam
    wd = hp.weight_decay * hp.lr / cap if hp.decoupled_wd else hp.weight_decay
    return dict(lr=cap, wd=wd, freeze=unhooked_parameter_names(bench))


def _relative(arm: str, t: float, base_lam: float, held: Dict[str, Any]) -> Dict[str, Any]:
    damping = "layer_relative" if arm == "s1b" else "network_relative"
    return dict(arm=arm, value=t, lam=base_lam,
                overrides={"hold_cap": True, "damping": damping, "damping_tau": t}, **held)


def fisher_cells(bench, mode: str) -> List[Dict[str, Any]]:
    """Every Fisher cell of one (seed, mode) job on ViT-S, in the order they run."""
    base_lam = bench.hparams.lam
    held = _held(bench)
    if VALUES:   # an extension job: relative values only, in whichever arm was asked for
        arm = "netadapt" if ARMS == "netadapt" else "s1b"
        return [_relative(arm, t, base_lam, held) for t in VALUES]
    if ARMS == "netadapt":   # amendment 1: the network-wide relative arm, on the same tau grid
        return [_relative("netadapt", t, base_lam, held) for t in _grid(TAU_GRID)]
    cells = [dict(arm="reference", value=base_lam, lam=base_lam,
                  overrides={"hold_cap": True}, **held)]
    cells += [dict(arm="single", value=lam, lam=lam, overrides={"hold_cap": True}, **held)
              for lam in _grid(SINGLE_GRID)]
    cells += [_relative("s1b", t, base_lam, held) for t in _grid(TAU_GRID)]
    return cells


def bridge_cell(mode: str) -> Dict[str, Any]:
    """E15's own ``vit_micro_cifar`` S1-b cell at tau = 0.1, built by E15's planner."""
    bench = discover_benchmarks()[BRIDGE_MODEL]
    hp = bench.hparams
    saved = e15.ARMS, e15.GRID_LIMIT
    e15.ARMS, e15.GRID_LIMIT = ["s1b"], None
    try:
        cells = e15.plan_cells(BRIDGE_MODEL, mode, hp.lr / hp.lam, hp.lam, hp.lr, hp.weight_decay,
                               hp.decoupled_wd, unhooked_parameter_names(bench))
    finally:
        e15.ARMS, e15.GRID_LIMIT = saved
    (cell,) = [c for c in cells if c["value"] == TAU_TRANSFERRED]
    cell = dict(cell)
    cell.pop("arm"), cell.pop("value")
    return e15.safe_run_cell(bench, mode, **cell)


def _write(out: Path, res: Dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)


def main() -> None:
    if SEED is None or JOB not in ("ekfac", "tekfac", "adamw", "calib"):
        raise SystemExit("E20: need E20_SEED and E20_JOB in ekfac|tekfac|adamw|calib")
    if (EPOCHS != 15 or NUM_WORKERS != 4 or TRAIN_SUBSET is not None or GRID_LIMIT is not None) \
            and not SMOKE:
        raise SystemExit(f"E20: non-production settings (epochs {EPOCHS}, workers {NUM_WORKERS}, "
                         f"subset {TRAIN_SUBSET}, grid limit {GRID_LIMIT}) without E20_SMOKE=1")
    if ARMS not in ("", "netadapt"):
        raise SystemExit(f"E20: E20_ARMS is 'netadapt' or unset; got {ARMS!r}")
    if ARMS == "netadapt" and JOB not in ("ekfac", "tekfac"):
        raise SystemExit("E20: the netadapt arm runs in an ekfac or tekfac job")
    if VALUES:
        family = "adamw" if JOB == "adamw" else "s1b"
        if JOB == "calib" or not TAG or not set(VALUES) <= set(EXTENSIONS[family]):
            raise SystemExit(f"E20: an extension job needs E20_TAG and values from "
                             f"{EXTENSIONS[family]}; got {VALUES} tag {TAG!r}")
    # E15's and E16's cell builders read their module globals at call time.
    e15.SEED, e15.BATCH, e15.TRAIN_SUBSET, e15.LOG_EVERY = SEED, BATCH, TRAIN_SUBSET, 1000
    e16b.SEED, e16b.OPT, e16b.TRAIN_SUBSET, e16b.BATCH = SEED, "adamw", TRAIN_SUBSET, BATCH

    bench = discover_benchmarks()[MODEL]
    out = Path(os.environ.get("E20_OUT") or ROOT / "fisher_ref/outputs" /
               f"e20_{MODEL}_s{SEED}_{JOB}{'_' + TAG if TAG else ''}.json")
    prov = provenance()
    res: Dict[str, Any] = {"model": MODEL, "seed": SEED, "job": JOB, "tag": TAG, "arms": ARMS,
                           "values": VALUES, "grid_limit": GRID_LIMIT,
                           "train_subset": TRAIN_SUBSET, "hparams": asdict(bench.hparams),
                           "provenance": prov, "problems": [], "bridge": None, "cells": {}}
    print(f"E20 | {MODEL} | seed={SEED} | job={JOB}{' tag=' + TAG if TAG else ''} | "
          f"commit {prov['git_commit']} dirty={prov['git_dirty']} | gpu {prov['gpu']}\n"
          f"frozen in the Fisher cells: {', '.join(unhooked_parameter_names(bench))}", flush=True)

    if JOB in ("ekfac", "tekfac") and SEED == 0 and not VALUES and ARMS != "netadapt":
        t0 = time.time()
        b = bridge_cell(JOB)
        want_acc, want_move = BRIDGE[JOB]
        ok = b.get("test_acc") == want_acc and b.get("theta_move_from_init") == want_move
        res["bridge"] = {"test_acc": b.get("test_acc"), "theta_move": b.get("theta_move_from_init"),
                         "expected": [want_acc, want_move], "reproduced": ok}
        print(f"  bridge {BRIDGE_MODEL} {JOB} s1b tau=0.1: test {b.get('test_acc')} vs {want_acc}, "
              f"move {b.get('theta_move_from_init')} vs {want_move} -> "
              f"{'REPRODUCED' if ok else 'NOT REPRODUCED'}  ({time.time() - t0:.0f}s)", flush=True)
        if not ok and not SMOKE:
            res["problems"].append("bridge not reproduced: the job stops here")
            _write(out, res)
            raise SystemExit("E20: the bridge did not reproduce E15's cell; no E20 result is read "
                             "until that is explained")

    if JOB == "adamw":
        lrs = VALUES or _grid(ADAMW_GRID)
        for lr in lrs:
            t0 = time.time()
            try:
                cell = e16b.run_cell(bench, lr)
            except Exception as exc:  # noqa: BLE001 -- a crash is an outcome, recorded
                cell = {"error": f"{type(exc).__name__}: {exc}", "test_acc": float("nan"),
                        "epoch_val_acc": []}
            cell.update(arm="adamw", value=lr, mode="adamw")
            res["cells"][f"adamw|adamw|{lr:g}"] = cell
            print(f"  adamw  lr={lr:9.2e}  val={100 * (cell['epoch_val_acc'] or [float('nan')])[-1]:6.2f}"
                  f"  test={100 * cell['test_acc']:6.2f}  ({time.time() - t0:.0f}s)", flush=True)
            _write(out, res)
    else:
        mode = "ekfac" if JOB == "calib" else JOB
        cells = fisher_cells(bench, mode)
        if JOB == "calib":
            cells = cells[:1]
        for cell in cells:
            arm, value = cell.pop("arm"), cell.pop("value")
            t0 = time.time()
            run = e15.safe_run_cell(bench, mode, **cell)
            v = run.get("epoch_val_acc") or [float("nan")]
            print(f"  {mode:6s} {arm:9s} {value:9.2e}  val={100 * v[-1]:6.2f}  "
                  f"test={100 * run['test_acc']:6.2f}  ({time.time() - t0:.0f}s)", flush=True)
            res["cells"][f"{mode}|{arm}|{value:g}"] = {
                **run, "mode": mode, "arm": arm, "value": value, "lam": cell["lam"],
                "lr": cell["lr"], "wd": cell["wd"], "freeze": cell["freeze"],
                "overrides": cell["overrides"],
            }
            _write(out, res)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
