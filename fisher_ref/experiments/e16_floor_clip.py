"""E16: a floor or a clip instead of an added safety constant.

``docs/reports/plan_floor_clip.md`` is the specification, and ``docs/reports/plan_lambda_dominance.md``
section "E16 -- pre-registered" carries its decision rules. This driver implements them and nothing
else; anything it does differently is a deviation and must be recorded there.

**What one job runs.** One network, one seed, one mode (``ekfac`` or ``tekfac``); every run 15
epochs at batch 32 with the cosine schedule, ``eig_before_rescale`` on, and -- in every arm except
the repro diagnostic -- ``norm_exact_rescaling`` on, so a normalisation layer's eigen-rescaling is
estimated from its exact per-row gradient (``plan_floor_clip.md`` §11). E16 is self-contained: its
add baseline is rerun here, under the same estimator, code and hardware as the other arms.

  add        E14's fix: divide by s + lambda, lr = cap * lambda, nothing frozen, the benchmark's
             own weight decay. 5 values of lambda around E13/E14's optimum.
  floor      family A: divide by max(s, lambda); otherwise exactly as add, paired with it by seed.
  clipema    family B, the main arm ("ema"): clip the undamped curvature step; per module, the
             threshold is the q-quantile of the active coordinates rescaled by the momentum's size
             against its own bias-corrected running average over 1000 steps.
  clip       family B, control ("quantile"): the same threshold reset at every step -- a
             per-module normalisation of the step, which clipema lets the momentum's size undo.
  clipfixed  family B, the conditional clip ("fixed"): per module, the median of the quantile over
             steps 1000-1999, frozen at step 2000.
  cliplr     the lr control, on the main arm: clipema at q = 0.7, at lr/10, lr/3 and 3*lr.

Seed 0 adds two checks, both on E14/E13/E10's best lambda for the network:

  dupcheck   the add cell at that lambda run a second time, in another process: it must be
             bit-identical to the first (determinism of the whole job, concurrency included).
  repro      the same cell with the shipped estimator (``norm_exact_rescaling`` off), compared
             with the stored E14/E13/E10 cell: a diagnostic of how far this hardware and code
             reproduce E14, recorded, not a gate. On ``cnn_gn_cifar``, which has no hooked
             normalisation layer, it must also be bit-identical to the add cell -- the knob must be
             inert there -- and that is a gate.

**Sharding.** A job runs ``E16_NSHARDS`` copies of this script in parallel on one GPU slice, copy
``E16_SHARD`` taking every ``NSHARDS``-th cell of the job's ordered list (the dupcheck cell always
goes to another shard than its twin), each writing its own shard file after every run. Then
``E16_MERGE=1`` merges the shard files into the job's file, checks that every planned cell is there
exactly once, and evaluates the two seed-0 checks. The cells are independent runs, so the sharding
changes nothing in any of them.

**Two choices that make the clip arms comparable, both recorded in the plan.** (1) The parameters no
hooked module owns (``pos_embed``; ``cnn_gn_cifar``'s GroupNorm) are frozen in the clip arms. In the
add and floor cells at lambda ~ 1e-10 they move by ``cap * lambda`` ~ 3e-11 to 1e-10 times their
momentum, i.e. they are frozen in effect; a clip arm at lr = 1e-3 would otherwise give them momentum
steps 1e7 to 1e8 times larger. (2) On the two networks with decoupled weight decay
(``vit_micro_cifar``, ``cct_2_3x2_cifar``) the clip arms run with none: in the add cells the decay is
``lr * wd`` with ``lr = cap * lambda``, i.e. it has vanished. ``cnn_gn_cifar``'s decay is coupled --
added to the gradient -- and is kept everywhere.

**Recorded along the way.** Every 50 steps up to step 5000, then every ``E16_LOG_EVERY`` steps, for
every hooked layer: its mean stored curvature; for ``floor``, the fraction of directions whose
curvature is above lambda; for the clip arms, ``gamma``, the fraction clipped among the active
coordinates (per input-eigen column on a normalisation layer), the fraction of active coordinates,
the fraction where the guard binds, the share of the step carried by coordinates below 1e-2 x rms,
and for ``clipema`` the momentum's size against its running average. And once per shard: the
torch/CUDA/cuDNN versions, the GPU, the TF32 flags, the data-loader workers, the git commit and
whether the tree was dirty.

**Production settings are enforced.** Batch 32, 15 epochs, 4 data-loader workers per process (the
worker count changes the trajectory, so every cell of E16 uses the same), the full grids, the full
training split and the pre-registered calibration step. Anything else needs ``E16_SMOKE=1``.

Environment variables::

    E16_MODEL          one benchmark (default: cnn_gn_cifar)
    E16_SEED           seed (default: 0)
    E16_MODE           ekfac or tekfac (required)
    E16_NSHARDS        processes sharing the job (default: 1)
    E16_SHARD          this process's index, 0 .. NSHARDS-1 (default: 0)
    E16_MERGE          "1": merge the shard files instead of running
    E16_OUT            the job's file (default:
                       fisher_ref/outputs/e16_floor_clip_<model>_s<seed>_<mode>.json)
    E16_LOG_EVERY      steps between two logs after step 5000 (default: 1000)
    E16_SMOKE          "1" allows the smoke settings below
    E16_ARMS, E16_BATCH, E16_GRID_LIMIT, E16_TRAIN_SUBSET, E16_CALIBRATE_AT
                       smoke only (defaults: every arm / 32 / none / none / 2000)
    WARMUP_SGD_EPOCHS, WARMUP_SGD_DATA_ROOT, WARMUP_SGD_DEVICE, WARMUP_SGD_NUM_WORKERS as elsewhere.
"""
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from benchmarks.common.loop import evaluate, train_under_budget
from benchmarks.common.optimizers import build_optimizer
from benchmarks.common.runner import discover_benchmarks
from benchmarks.common.schedules import NominalCosine
from fisher_ref.experiments.e5_unhooked_freeze_control import unhooked_parameter_names
from fisher_ref.experiments.warmup_sgd_baseline import (
    ALLOW_DOWNLOAD,
    DATA_ROOT,
    DEVICE,
    EPOCHS,
    NUM_WORKERS,
    flatten_params,
)

SMOKE = os.environ.get("E16_SMOKE", "0") == "1"
MODEL = os.environ.get("E16_MODEL", "cnn_gn_cifar")
SEED = int(os.environ.get("E16_SEED", "0"))
MODE = os.environ.get("E16_MODE", "")
NSHARDS = int(os.environ.get("E16_NSHARDS", "1"))
SHARD = int(os.environ.get("E16_SHARD", "0"))
MERGE = os.environ.get("E16_MERGE", "0") == "1"
LOG_EVERY = int(os.environ.get("E16_LOG_EVERY", "1000"))
DENSE_LOG_UNTIL, DENSE_LOG_EVERY = 5000, 50
OUT = Path(os.environ.get(
    "E16_OUT", str(ROOT / f"fisher_ref/outputs/e16_floor_clip_{MODEL}_s{SEED}_{MODE}.json")))

# The production settings, pre-registered. The smoke variables may change them only under E16_SMOKE.
PRODUCTION = {"E16_ARMS": "dupcheck,repro,add,floor,clipema,clip,clipfixed,cliplr",
              "E16_BATCH": "32", "E16_GRID_LIMIT": "", "E16_TRAIN_SUBSET": "",
              "E16_CALIBRATE_AT": "2000"}
PRODUCTION_EPOCHS, PRODUCTION_WORKERS = 15, 4


def _setting(name: str) -> str:
    return os.environ.get(name, PRODUCTION[name])


ARMS = [a.strip() for a in _setting("E16_ARMS").split(",") if a.strip()]
BATCH = int(_setting("E16_BATCH"))
GRID_LIMIT = int(_setting("E16_GRID_LIMIT")) if _setting("E16_GRID_LIMIT") else None
TRAIN_SUBSET = int(_setting("E16_TRAIN_SUBSET")) if _setting("E16_TRAIN_SUBSET") else None
CALIBRATE_AT = int(_setting("E16_CALIBRATE_AT"))

MODES = ("ekfac", "tekfac")
# The grids, exactly as pre-registered. Lambda: 5 values at half-decade spacing around the add
# optimum each network's five-seed sweep located (E14, E13, E10), shared by add and floor.
LAMBDA_GRID = {
    "cnn_gn_cifar": [3e-10, 1e-10, 3e-11, 1e-11, 3e-12],
    "vit_micro_cifar": [1e-9, 3e-10, 1e-10, 3e-11, 1e-11],
    "cct_2_3x2_cifar": [3e-10, 1e-10, 3e-11, 1e-11, 3e-12],
}
# Clip: the fraction of each module's active coordinates that is clipped. Sophia's README targets a
# win_rate (the fraction NOT clipped) of 0.1-0.5, i.e. 0.5-0.9 here; the grid reaches past both
# ends: 0.99 is almost sign-momentum in the eigenbasis, 0.1 almost the plain undamped step.
CLIP_GRID = [0.99, 0.95, 0.9, 0.7, 0.5, 0.3, 0.1]
# The three thresholds, one arm each, sharing CLIP_GRID. "fixed" freezes, per module, the median of
# the quantile over the 1000 steps before step 2000: past plan_exp_draft.md §3.2's 10 * TCov = 1000
# steps for the identity seed to decay; that is epoch 2 of 15 at batch 32 (1 406 steps per epoch,
# the loader drops the last incomplete batch).
CLIP_ARMS = {
    "clipema": {"clip_threshold": "ema", "clip_ema_horizon": 1000},
    "clip": {"clip_threshold": "quantile"},
    "clipfixed": {"clip_threshold": "fixed", "clip_calibrate_at": CALIBRATE_AT,
                  "clip_calibration_window": 1000},
}
CLIPLR_ARM = "clipema"
CLIPLR_Q = 0.7
CLIPLR_FACTORS = [0.1, 1 / 3, 3.0]
# E14/E13/E10's located optimum per (network, mode): the lambda of the seed-0 checks. It is in the
# lambda grid, so the dupcheck and repro cells have an add twin.
CHECK_LAMBDA = {
    ("cnn_gn_cifar", "ekfac"): 3e-11, ("cnn_gn_cifar", "tekfac"): 1e-10,
    ("vit_micro_cifar", "ekfac"): 1e-10, ("vit_micro_cifar", "tekfac"): 1e-10,
    ("cct_2_3x2_cifar", "ekfac"): 3e-11, ("cct_2_3x2_cifar", "tekfac"): 3e-11,
}
BASELINE_FILE = {"cnn_gn_cifar": "e14_seeds_cnn_eigfix", "vit_micro_cifar": "e13_seeds_vit_eigfix",
                 "cct_2_3x2_cifar": "e10_seeds_cct_eigfix"}
# The one network without a hooked normalisation layer: there norm_exact_rescaling must be inert.
NO_HOOKED_NORM = {"cnn_gn_cifar"}
# Order in which the cells are listed (and dealt to the shards): the checks, then the arms by
# importance, so a wall-clock kill costs the controls first.
ARM_ORDER = ["dupcheck", "repro", "add", "floor", "clipema", "clip", "clipfixed", "cliplr"]
COMPARED = ("test_acc", "test_loss", "epoch_val_acc", "theta_move_from_init", "n_steps")


def check_production_settings() -> None:
    """Refuse a non-production configuration unless E16_SMOKE=1 says so explicitly."""
    problems = [f"{k}={os.environ[k]!r}" for k in PRODUCTION
                if k in os.environ and os.environ[k] != PRODUCTION[k]]
    if EPOCHS != PRODUCTION_EPOCHS:
        problems.append(f"WARMUP_SGD_EPOCHS={EPOCHS}")
    if NUM_WORKERS != PRODUCTION_WORKERS:
        problems.append(f"WARMUP_SGD_NUM_WORKERS={NUM_WORKERS}")
    if MODEL not in LAMBDA_GRID:
        problems.append(f"E16_MODEL={MODEL!r} (not one of {sorted(LAMBDA_GRID)})")
    if problems and not SMOKE:
        raise SystemExit(
            "E16: non-production settings without E16_SMOKE=1: " + ", ".join(problems)
            + ". Unset them (the job script does), or set E16_SMOKE=1 for a smoke run."
        )
    if MODE not in MODES:
        raise SystemExit(f"E16: E16_MODE must be one of {MODES}; got {MODE!r}")
    if not 0 <= SHARD < NSHARDS:
        raise SystemExit(f"E16: need 0 <= E16_SHARD < E16_NSHARDS; got {SHARD}, {NSHARDS}")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}"


def provenance() -> Dict[str, Any]:
    """What produced this file, so that a result can be traced and a difference explained."""
    cuda = torch.cuda.is_available()
    return {
        "smoke": SMOKE, "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "tf32_matmul": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "device": str(DEVICE), "num_workers": NUM_WORKERS, "epochs": EPOCHS,
        "shard": SHARD, "nshards": NSHARDS,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(), "host": platform.node(),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("E16_", "WARMUP_SGD_"))},
    }


def _grid(values: List[Any]) -> List[Any]:
    return values if GRID_LIMIT is None else values[:GRID_LIMIT]


def cell_key(arm: str, value: float) -> str:
    return f"{MODE}|{arm}|{value:g}"


def _log(opt, names: Dict[Any, str]) -> Dict[str, Dict[str, float]]:
    """Per hooked layer: what the floor or the clip is doing at this step."""
    approx = opt.approx
    rescaling = approx._s_star if MODE == "ekfac" else approx._Theta
    stats = approx._clip_stats if approx.rescale_form == "clip" else {}
    out: Dict[str, Dict[str, float]] = {}
    for module in opt.modules:
        if module not in rescaling:
            continue
        s = rescaling[module]
        row = {"mean_curvature": float(s.mean())}
        if approx.rescale_form == "floor":
            row["frac_above_lambda"] = float((s > approx.lambda_for(module)).double().mean())
        elif module in stats:
            row.update({k: float(v) for k, v in stats[module].items()})
        out[names[module]] = row
    return out


def _is_log_step(step: int) -> bool:
    return step % LOG_EVERY == 0 or (step < DENSE_LOG_UNTIL and step % DENSE_LOG_EVERY == 0)


def run_cell(bench, *, lam: float, lr: float, wd: float, freeze: Optional[List[str]],
             overrides: Dict[str, Any]) -> Dict[str, Any]:
    """One training run: E15's ``run_cell`` -- itself E4's ``run_one`` with overrides -- with the
    floor/clip log in place of the lambda log, and nothing else changed (checked bit for bit
    against E4's ``run_one`` over a full epoch). Identical initialisation and data order across
    every cell of a seed."""
    hp = replace(bench.hparams, lam=lam, lr=lr, weight_decay=wd)
    torch.manual_seed(SEED)
    model = bench.build_model(**{k: v[0] for k, v in bench.model_choices.items()}).to(DEVICE)
    if freeze:
        wanted = set(freeze)
        found = 0
        for name, p in model.named_parameters():
            if name in wanted:
                p.requires_grad_(False)
                found += 1
        if found != len(wanted):
            raise ValueError(f"freeze named {len(wanted)} parameters, matched {found}")
    theta0 = flatten_params(model)
    optimizer = build_optimizer(MODE, model, hp, eig_before_rescale=True, **overrides)

    names = {m: n for n, m in model.named_modules()}
    log: List[Dict[str, Any]] = []
    inner_step = optimizer.step

    def step_and_log(*args, **kwargs):
        inner_step(*args, **kwargs)
        if _is_log_step(optimizer.steps - 1):
            log.append({"step": optimizer.steps - 1, "layers": _log(optimizer, names)})

    optimizer.step = step_and_log  # type: ignore[method-assign]

    train_loader, val_loader, test_loader = bench.build_data(
        DATA_ROOT, batch_size=BATCH, seed=SEED, num_workers=NUM_WORKERS,
        cutout=True, allow_download=ALLOW_DOWNLOAD, train_subset=TRAIN_SUBSET,
    )
    scheduler = NominalCosine(optimizer, t_max=EPOCHS)
    eval_fn = None
    if val_loader is not None:
        eval_fn = lambda: evaluate(  # noqa: E731
            model, val_loader, bench.loss_fn, DEVICE,
            prepare_batch=bench.prepare_batch, metric_fn=bench.metric_fn,
        )
    t0 = time.time()
    steps, epochs = train_under_budget(
        model, optimizer, train_loader, bench.loss_fn,
        budget_s=float("inf"), max_epochs=EPOCHS, device=DEVICE, scheduler=scheduler,
        eval_fn=eval_fn, prepare_batch=bench.prepare_batch, log_fn=lambda _msg: None,
    )
    elapsed = time.time() - t0
    test_loss, test_acc = evaluate(model, test_loader, bench.loss_fn, DEVICE,
                                   prepare_batch=bench.prepare_batch, metric_fn=bench.metric_fn)
    return {
        "wall_s": elapsed, "n_steps": len(steps),
        "epoch_train_loss": [e.train_loss for e in epochs],
        "epoch_val_loss": [e.val_loss for e in epochs],
        "epoch_val_acc": [e.val_acc for e in epochs],
        "test_loss": test_loss, "test_acc": test_acc,
        "theta_move_from_init": float((flatten_params(model) - theta0).norm()),
        "log": log,
    }


def safe_run_cell(bench, **kw) -> Dict[str, Any]:
    """A crash is an outcome to record, not a reason to lose the rest of the grid. The record says
    so explicitly; e16_decisions.py refuses to read a verdict through it."""
    try:
        return run_cell(bench, **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"    [CRASHED] {type(exc).__name__}: {exc}", flush=True)
        return {"crashed": True, "error": f"{type(exc).__name__}: {exc}", "wall_s": None,
                "n_steps": 0, "epoch_train_loss": [], "epoch_val_loss": [], "epoch_val_acc": [],
                "test_loss": None, "test_acc": None, "theta_move_from_init": None, "log": []}


def plan_cells(model_name: str, base_lam: float, base_lr: float, wd: float, decoupled: bool,
               frozen: List[str]) -> List[Dict[str, Any]]:
    """Every cell of one (network, seed, mode) job, in ARM_ORDER."""
    exact = {"norm_exact_rescaling": True}
    check_lam = CHECK_LAMBDA[(model_name, MODE)]

    def lambda_cell(arm: str, lam: float, overrides: Dict[str, Any]) -> Dict[str, Any]:
        # Spelled exactly as e4_fixed_average.py spells it: cap * lambda can differ in the last bit.
        lr = base_lr * (lam / base_lam)
        return dict(arm=arm, value=lam, lam=lam, lr=lr, wd=wd, freeze=None, overrides=overrides)

    cells: List[Dict[str, Any]] = []
    if "dupcheck" in ARMS and SEED == 0:
        cells.append(lambda_cell("dupcheck", check_lam, dict(exact)))
    if "repro" in ARMS and SEED == 0:
        cells.append(lambda_cell("repro", check_lam, {}))       # the shipped estimator
    if "add" in ARMS:
        cells += [lambda_cell("add", lam, dict(exact)) for lam in _grid(LAMBDA_GRID[model_name])]
    if "floor" in ARMS:
        cells += [lambda_cell("floor", lam, {"rescale_form": "floor", **exact})
                  for lam in _grid(LAMBDA_GRID[model_name])]
    clip_wd = 0.0 if decoupled else wd
    for arm, threshold in CLIP_ARMS.items():
        if arm in ARMS:
            for q in _grid(CLIP_GRID):
                cells.append(dict(arm=arm, value=q, lam=base_lam, lr=base_lr, wd=clip_wd,
                                  freeze=frozen, overrides={"rescale_form": "clip",
                                                            "clip_fraction": q, **threshold,
                                                            **exact}))
    if "cliplr" in ARMS:
        for f in _grid(CLIPLR_FACTORS):
            cells.append(dict(arm="cliplr", value=f, lam=base_lam, lr=base_lr * f, wd=clip_wd,
                              freeze=frozen, overrides={"rescale_form": "clip",
                                                        "clip_fraction": CLIPLR_Q,
                                                        **CLIP_ARMS[CLIPLR_ARM], **exact}))
    return sorted(cells, key=lambda c: ARM_ORDER.index(c["arm"]))


def shard_of(cells: List[Dict[str, Any]], nshards: int) -> List[int]:
    """Deal the cells round-robin; the dupcheck cell goes to another shard than its add twin, so
    that the determinism check covers two processes running side by side."""
    owners = [i % nshards for i in range(len(cells))]
    dup = [i for i, c in enumerate(cells) if c["arm"] == "dupcheck"]
    if dup and nshards > 1:
        lam = cells[dup[0]]["value"]
        twin = next(i for i, c in enumerate(cells) if c["arm"] == "add" and c["value"] == lam)
        if owners[dup[0]] == owners[twin]:
            owners[dup[0]] = (owners[twin] + 1) % nshards
    return owners


def shard_path(shard: int) -> Path:
    return OUT.with_name(OUT.stem + f".shard{shard}of{NSHARDS}.json")


def stored_baseline_cell(lam: float) -> Optional[Dict[str, Any]]:
    """The E14/E13/E10 cell of this seed and mode at this lambda, or None if the file is not there."""
    path = ROOT / "fisher_ref/outputs" / f"{BASELINE_FILE[MODEL]}_s{SEED}.json"
    if not path.exists():
        return None
    return json.load(open(path))["results"][MODEL]["cells"].get(f"{MODE}|shipped|{lam:g}")


def same_run(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Field-by-field equality of two runs' outcomes."""
    if a is None or b is None:
        return {"identical": False, "reason": "missing"}
    mismatched = [f for f in COMPARED if a.get(f) != b.get(f)]
    return {"identical": not mismatched, "mismatched": mismatched,
            "test_acc": [a.get("test_acc"), b.get("test_acc")]}


def _sanitise(x: Any) -> Any:
    """NaN and inf become null, so the file is strict JSON."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _sanitise(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitise(v) for v in x]
    return x


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(_sanitise(payload), f, indent=1, allow_nan=False)
    tmp.replace(path)   # never a half-written file, even on a wall-clock kill


def _final_val(cell: Dict[str, Any]) -> float:
    v = [a for a in (cell.get("epoch_val_acc") or []) if a is not None]
    return v[-1] if v else float("nan")


def header(bench) -> Dict[str, Any]:
    hp0 = bench.hparams
    return {"model": MODEL, "seed": SEED, "mode": MODE, "batch_size": BATCH, "epochs": EPOCHS,
            "base_lambda": hp0.lam, "base_lr": hp0.lr, "weight_decay": hp0.weight_decay,
            "decoupled_wd": hp0.decoupled_wd, "grid_limit": GRID_LIMIT,
            "train_subset": TRAIN_SUBSET, "calibrate_at": CALIBRATE_AT, "nshards": NSHARDS}


def run_shard() -> None:
    bench = discover_benchmarks()[MODEL]
    hp0 = bench.hparams
    frozen = unhooked_parameter_names(bench)
    cells = plan_cells(MODEL, hp0.lam, hp0.lr, hp0.weight_decay, hp0.decoupled_wd, frozen)
    owners = shard_of(cells, NSHARDS)
    mine = [c for c, o in zip(cells, owners) if o == SHARD]
    prov = provenance()
    print(f"E16 | {MODEL} | seed={SEED} | mode={MODE} | shard {SHARD}/{NSHARDS}: {len(mine)} of "
          f"{len(cells)} cells | batch={BATCH} | epochs={EPOCHS} | device={DEVICE} | "
          f"smoke={SMOKE}\ntorch {prov['torch']} cuda {prov['cuda']} gpu {prov['gpu']} "
          f"tf32 matmul/cudnn {prov['tf32_matmul']}/{prov['tf32_cudnn']} workers {NUM_WORKERS} "
          f"commit {prov['git_commit']} dirty={prov['git_dirty']}", flush=True)
    results: Dict[str, Any] = {**header(bench), "shard": SHARD, "provenance": prov,
                               "planned": [cell_key(c["arm"], c["value"]) for c in mine],
                               "cells": {}}
    path = shard_path(SHARD) if NSHARDS > 1 else OUT
    for cell in mine:
        cell = dict(cell)
        arm, value = cell.pop("arm"), cell.pop("value")
        t0 = time.time()
        run = safe_run_cell(bench, **cell)
        acc = run.get("test_acc")
        print(f"  {MODE:6s} {arm:9s} {value:9.3g}  val={100*_final_val(run):6.2f}  "
              f"test={100*acc if acc is not None else float('nan'):6.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        results["cells"][cell_key(arm, value)] = {
            **run, "mode": MODE, "arm": arm, "value": value, "lam": cell["lam"],
            "lr": cell["lr"], "wd": cell["wd"], "freeze": cell["freeze"],
            "overrides": cell["overrides"],
        }
        _write(path, results)
    if NSHARDS == 1:
        merge([results])


def merge(shards: Optional[List[Dict[str, Any]]] = None) -> None:
    """Merge the shard files into the job's file, check completeness, and evaluate the two
    seed-0 checks. Exits non-zero if a shard file or a planned cell is missing."""
    bench = discover_benchmarks()[MODEL]
    hp0 = bench.hparams
    if shards is None:
        missing = [str(shard_path(i)) for i in range(NSHARDS) if not shard_path(i).exists()]
        shards = [json.load(open(shard_path(i))) for i in range(NSHARDS)
                  if shard_path(i).exists()]
    else:
        missing = []
    cells: Dict[str, Any] = {}
    for s in shards:
        for key, c in s["cells"].items():
            if key in cells:
                raise SystemExit(f"E16 merge: {key} appears in two shards")
            cells[key] = c
    planned = [cell_key(c["arm"], c["value"]) for c in plan_cells(
        MODEL, hp0.lam, hp0.lr, hp0.weight_decay, hp0.decoupled_wd, unhooked_parameter_names(bench))]
    absent = [k for k in planned if k not in cells]
    checks: Dict[str, Any] = {}
    if SEED == 0:
        lam = CHECK_LAMBDA[(MODEL, MODE)]
        add_twin = cells.get(cell_key("add", lam))
        checks["determinism"] = same_run(cells.get(cell_key("dupcheck", lam)), add_twin)
        repro = cells.get(cell_key("repro", lam))
        checks["repro_vs_e14"] = same_run(repro, stored_baseline_cell(lam))
        if MODEL in NO_HOOKED_NORM:
            checks["norm_exact_inert"] = same_run(repro, add_twin)
    merged = {**header(bench), "provenance": [s.get("provenance") for s in shards],
              "missing_shards": missing, "absent_cells": absent, "checks": checks,
              "cells": cells}
    _write(OUT, merged)
    print(f"E16 merge | {MODEL} | seed={SEED} | mode={MODE}: {len(cells)} of {len(planned)} "
          f"planned cells, {len(missing)} missing shard files\n  checks: "
          f"{json.dumps(checks)}\nwrote {OUT}", flush=True)
    if missing or absent:
        raise SystemExit(f"E16 merge: incomplete ({len(missing)} shard files, "
                         f"{len(absent)} cells missing)")


def main() -> None:
    check_production_settings()
    if MERGE:
        merge()
    else:
        run_shard()


if __name__ == "__main__":
    main()
