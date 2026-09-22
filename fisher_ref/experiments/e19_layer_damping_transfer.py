"""E19: E15's per-layer safety constant (S1-b) on CCT and the two ResNets, paired with E16's cells.

``docs/reports/plan_lambda_dominance.md``, section "E19 -- pre-registered", is the specification.
This driver implements it and nothing else; anything it does differently is a deviation and must be
recorded there. ``fisher_ref/experiments/e19_decisions.py`` applies its rules.

**What one job runs.** One network, one seed, one mode (``ekfac`` or ``tekfac``); every run 15
epochs at batch 32 with the clamped cosine, ``eig_before_rescale`` on and ``norm_exact_rescaling``
on -- E16's protocol exactly, so that every cell here pairs by seed with E16's stored cells, which
share its initialisation and data order. E19 does not rerun E16's single-lambda grid or its clips.

  bridge    seed 0 only: E16's add cell at E16's check lambda, built exactly as E16's driver builds
            it (lr = cap * lambda, no hold_cap, nothing frozen, the benchmark's weight decay). Rule
            0a compares it bit for bit with E16's stored cell.
  held      one lambda for the network, at E16's validation-selected add value, with S1-b's
            settings below. Rule 0b pairs it with E16's add cell at that value.
  s1b       damping="layer_relative": lambda_l = tau * c_l for each hooked layer.
  netadapt  damping="network_relative": one lambda(t) = tau * c_net(t) (not on resnet50_cifar).
  wdctrl    s1b at tau = 0.1 with the benchmark's decoupled decay at its fixed rate
            (cct_2_3x2_cifar, the one decoupled network).

**S1-b's settings** (``held``, ``s1b``, ``netadapt``, ``wdctrl``), those of E15 and of E16's clip
arms: the cap held per layer (``hold_cap=True``, ``lr = cap = base_lr / base_lambda``); the
parameters no hooked module owns frozen (``pos_embed`` on CCT; nothing on the ResNets); and, on a
decoupled-decay network, no decay -- except ``wdctrl``, whose coefficient ``wd * base_lr / cap``
gives the benchmark's ``base_lr * wd`` per step at the top of the cosine. A coupled decay (the
ResNets') is added to the gradient and kept everywhere. On ``resnet50_cifar`` every cell also takes
E16's ``fisher_batch_samples=None`` (its reduced design's override; ``conv_sua`` stays on).

**Recorded along the way.** Every ``E19_LOG_EVERY`` steps, for every hooked layer, E15's log
(``e15_layer_damping._lambda_log``): the lambda in effect, the layer's mean stored curvature, and
the fraction of its directions whose curvature exceeds that lambda. And once per shard: the
torch/CUDA/cuDNN versions, the GPU, the TF32 flags, the data-loader workers, the git commit and
whether the tree was dirty.

**Sharding.** As E16: a (network, seed, mode) runs ``E19_NSHARDS`` copies of this script, each its
own job on its own GPU slice (``fisher_ref/slurm/e19_submit.sh``), copy ``E19_SHARD`` taking every
``NSHARDS``-th cell of the job's ordered list and writing its shard file after every run. Then
``E19_MERGE=1`` merges the shard files into the job's file and checks that every planned cell is
there exactly once. The cells are independent runs, so the sharding changes nothing in any of them.

**Production settings are enforced.** Batch 32, 15 epochs, 4 data-loader workers per process (the
worker count changes the trajectory, and E16 used 4), every arm, the full grids and the full
training split. Anything else needs ``E19_SMOKE=1``.

Environment variables::

    E19_MODEL          one of NETWORKS (required)
    E19_SEED           seed (required; E16's seeds for that network)
    E19_MODE           ekfac or tekfac (required)
    E19_NSHARDS        processes sharing the job (default: 1)
    E19_SHARD          this process's index, 0 .. NSHARDS-1 (default: 0)
    E19_MERGE          "1": merge the shard files instead of running
    E19_OUT            the job's file (default:
                       fisher_ref/outputs/e19_layer_damping_<model>_s<seed>_<mode>.json)
    E19_LOG_EVERY      steps between two lambda logs (default: 1000, E15's)
    E19_SMOKE          "1" allows the smoke settings below
    E19_ARMS, E19_BATCH, E19_GRID_LIMIT, E19_TRAIN_SUBSET
                       smoke only (defaults: every arm / 32 / none / none)
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
from fisher_ref.experiments.e15_layer_damping import TAU_GRID as E15_TAU_GRID
from fisher_ref.experiments.e15_layer_damping import _lambda_log
from fisher_ref.experiments.e16_floor_clip import CHECK_LAMBDA as E16_CHECK_LAMBDA
from fisher_ref.experiments.e16_floor_clip import REDUCED as E16_REDUCED
from fisher_ref.experiments.warmup_sgd_baseline import (
    ALLOW_DOWNLOAD,
    DATA_ROOT,
    DEVICE,
    EPOCHS,
    NUM_WORKERS,
    flatten_params,
)

SMOKE = os.environ.get("E19_SMOKE", "0") == "1"
MODEL = os.environ.get("E19_MODEL", "")
SEED = int(os.environ.get("E19_SEED", "0"))
MODE = os.environ.get("E19_MODE", "")
NSHARDS = int(os.environ.get("E19_NSHARDS", "1"))
SHARD = int(os.environ.get("E19_SHARD", "0"))
MERGE = os.environ.get("E19_MERGE", "0") == "1"
LOG_EVERY = int(os.environ.get("E19_LOG_EVERY", "1000"))
OUT = Path(os.environ.get(
    "E19_OUT", str(ROOT / f"fisher_ref/outputs/e19_layer_damping_{MODEL}_s{SEED}_{MODE}.json")))

# The production settings, pre-registered. The smoke variables may change them only under E19_SMOKE.
PRODUCTION = {"E19_ARMS": "bridge,held,s1b,netadapt,wdctrl", "E19_BATCH": "32",
              "E19_GRID_LIMIT": "", "E19_TRAIN_SUBSET": ""}
PRODUCTION_EPOCHS, PRODUCTION_WORKERS = 15, 4


def _setting(name: str) -> str:
    return os.environ.get(name, PRODUCTION[name])


ARMS = [a.strip() for a in _setting("E19_ARMS").split(",") if a.strip()]
BATCH = int(_setting("E19_BATCH"))
GRID_LIMIT = int(_setting("E19_GRID_LIMIT")) if _setting("E19_GRID_LIMIT") else None
TRAIN_SUBSET = int(_setting("E19_TRAIN_SUBSET")) if _setting("E19_TRAIN_SUBSET") else None

MODES = ("ekfac", "tekfac")
NETWORKS = ("cct_2_3x2_cifar", "resnet20_cifar", "resnet50_cifar")
# E16's seeds for each network, since every cell pairs with E16's cell at the same seed.
SEEDS = {"cct_2_3x2_cifar": (0, 1, 2, 3, 4), "resnet20_cifar": (0, 1, 2, 3, 4),
         "resnet50_cifar": tuple(E16_REDUCED["resnet50_cifar"]["seeds"])}
# The tau grids, exactly as pre-registered: E15's eight values for ekfac/tekfac (one grid for both
# modes there), and on ResNet-50 the five half-decade values around E15's optimum, 0.1.
E15_TAU = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4]
assert E15_TAU_GRID["ekfac"] == E15_TAU_GRID["tekfac"] == E15_TAU
TAU_GRID = {"cct_2_3x2_cifar": E15_TAU, "resnet20_cifar": E15_TAU,
            "resnet50_cifar": [1.0, 0.3, 0.1, 0.03, 0.01]}
# netadapt is left out of ResNet-50: its cost, and under SUA num_directions counts one kernel offset
# only, so the network-wide mean would weigh a 3x3 convolution ~9 times too little.
NETADAPT_NETWORKS = ("cct_2_3x2_cifar", "resnet20_cifar")
WDCTRL_TAU = 0.1
# E16's validation-selected add lambda per (network, mode), which e16_decisions.select gives on
# E16's files. e19_decisions.py checks this table against that selection before reading anything.
HELD_LAMBDA = {
    ("cct_2_3x2_cifar", "ekfac"): 3e-11, ("cct_2_3x2_cifar", "tekfac"): 3e-11,
    ("resnet20_cifar", "ekfac"): 1e-11, ("resnet20_cifar", "tekfac"): 3e-11,
    ("resnet50_cifar", "ekfac"): 3e-12, ("resnet50_cifar", "tekfac"): 3e-12,
}
# The bridge runs at E16's own check lambda: E16's stored seed-0 add cell there is the one compared.
BRIDGE_LAMBDA = {(n, m): E16_CHECK_LAMBDA[(n, m)] for n in NETWORKS for m in MODES}
# Every E16 cell of a reduced design carried these overrides; so does every E19 cell of it.
EXTRA_OVERRIDES = {n: dict(E16_REDUCED[n].get("overrides", {})) for n in NETWORKS if n in E16_REDUCED}
# Order in which the cells are listed (and dealt to the shards).
ARM_ORDER = ["bridge", "held", "s1b", "netadapt", "wdctrl"]


def check_production_settings() -> None:
    """Refuse a non-production configuration unless E19_SMOKE=1 says so explicitly."""
    problems = [f"{k}={os.environ[k]!r}" for k in PRODUCTION
                if k in os.environ and os.environ[k] != PRODUCTION[k]]
    if EPOCHS != PRODUCTION_EPOCHS:
        problems.append(f"WARMUP_SGD_EPOCHS={EPOCHS}")
    if NUM_WORKERS != PRODUCTION_WORKERS:
        problems.append(f"WARMUP_SGD_NUM_WORKERS={NUM_WORKERS}")
    if MODEL in SEEDS and SEED not in SEEDS[MODEL]:
        problems.append(f"E19_SEED={SEED} (E16 ran {MODEL} at seeds {SEEDS[MODEL]})")
    if problems and not SMOKE:
        raise SystemExit(
            "E19: non-production settings without E19_SMOKE=1: " + ", ".join(problems)
            + ". Unset them (the job script does), or set E19_SMOKE=1 for a smoke run."
        )
    if MODEL not in NETWORKS:
        raise SystemExit(f"E19: E19_MODEL must be one of {NETWORKS}; got {MODEL!r}")
    if MODE not in MODES:
        raise SystemExit(f"E19: E19_MODE must be one of {MODES}; got {MODE!r}")
    if not 0 <= SHARD < NSHARDS:
        raise SystemExit(f"E19: need 0 <= E19_SHARD < E19_NSHARDS; got {SHARD}, {NSHARDS}")


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
        "env": {k: v for k, v in os.environ.items() if k.startswith(("E19_", "WARMUP_SGD_"))},
    }


def _grid(values: List[Any]) -> List[Any]:
    return values if GRID_LIMIT is None else values[:GRID_LIMIT]


def cell_key(mode: str, arm: str, value: float) -> str:
    return f"{mode}|{arm}|{value:g}"


def run_cell(bench, mode: str, *, lam: float, lr: float, wd: float,
             freeze: Optional[List[str]], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """One training run: E16's ``run_cell`` with E15's lambda log in place of E16's floor/clip log,
    and nothing else changed -- the bridge cell depends on it, since it must reproduce E16's stored
    cell bit for bit. Identical initialisation and data order across every cell of a seed, and
    across E16's cells of that seed. Both logs only read the optimizer's state."""
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
    optimizer = build_optimizer(mode, model, hp, eig_before_rescale=True, **overrides)

    names = {m: n for n, m in model.named_modules()}
    log: List[Dict[str, Any]] = []
    inner_step = optimizer.step

    def step_and_log(*args, **kwargs):
        inner_step(*args, **kwargs)
        if (optimizer.steps - 1) % LOG_EVERY == 0:
            log.append({"step": optimizer.steps - 1,
                        "layers": _lambda_log(optimizer, names, mode)})

    optimizer.step = step_and_log

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
        "lambda_log": log,
    }


def safe_run_cell(bench, mode: str, **kw) -> Dict[str, Any]:
    """A crash is an outcome to record, not a reason to lose the rest of the grid. The record says
    so explicitly; e19_decisions.py refuses to read a verdict through it."""
    try:
        return run_cell(bench, mode, **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"    [CRASHED] {type(exc).__name__}: {exc}", flush=True)
        return {"crashed": True, "error": f"{type(exc).__name__}: {exc}", "wall_s": None,
                "n_steps": 0, "epoch_train_loss": [], "epoch_val_loss": [], "epoch_val_acc": [],
                "test_loss": None, "test_acc": None, "theta_move_from_init": None,
                "lambda_log": []}


def plan_cells(model_name: str, mode: str, seed: int, base_lam: float, base_lr: float, wd: float,
               decoupled: bool, frozen: List[str]) -> List[Dict[str, Any]]:
    """Every cell of one (network, seed, mode) job, in ARM_ORDER."""
    cap = base_lr / base_lam
    exact = {"norm_exact_rescaling": True, **EXTRA_OVERRIDES.get(model_name, {})}
    # S1-b's settings: the cap held per layer, the unhooked parameters frozen, and no decoupled
    # decay (E16's clip arms' convention); a coupled decay is kept.
    held = dict(lr=cap, wd=0.0 if decoupled else wd, freeze=frozen)
    cells: List[Dict[str, Any]] = []
    if "bridge" in ARMS and seed == 0:
        lam = BRIDGE_LAMBDA[(model_name, mode)]
        # E16's lambda_cell exactly, spelled as E16 spells it (cap * lambda can differ in the last
        # bit): lr = base_lr * (lambda / base_lambda), the benchmark's decay, nothing frozen.
        cells.append(dict(arm="bridge", value=lam, lam=lam, lr=base_lr * (lam / base_lam), wd=wd,
                          freeze=None, overrides=dict(exact)))
    if "held" in ARMS:
        lam = HELD_LAMBDA[(model_name, mode)]
        cells.append(dict(arm="held", value=lam, lam=lam,
                          overrides={"hold_cap": True, **exact}, **held))
    for arm, damping in (("s1b", "layer_relative"), ("netadapt", "network_relative")):
        if arm not in ARMS or (arm == "netadapt" and model_name not in NETADAPT_NETWORKS):
            continue
        for tau in _grid(TAU_GRID[model_name]):
            cells.append(dict(arm=arm, value=tau, lam=base_lam,
                              overrides={"hold_cap": True, "damping": damping,
                                         "damping_tau": tau, **exact}, **held))
    if "wdctrl" in ARMS and decoupled:
        # E15's convention: with lr = cap, a decoupled decay of rate base_lr * wd needs the
        # coefficient wd * base_lr / cap.
        cells.append(dict(arm="wdctrl", value=WDCTRL_TAU, lam=base_lam, lr=cap,
                          wd=wd * base_lr / cap, freeze=frozen,
                          overrides={"hold_cap": True, "damping": "layer_relative",
                                     "damping_tau": WDCTRL_TAU, **exact}))
    return sorted(cells, key=lambda c: ARM_ORDER.index(c["arm"]))


def shard_of(cells: List[Dict[str, Any]], nshards: int) -> List[int]:
    """Deal the cells round-robin."""
    return [i % nshards for i in range(len(cells))]


def shard_path(shard: int) -> Path:
    return OUT.with_name(OUT.stem + f".shard{shard}of{NSHARDS}.json")


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


def _planned(bench) -> List[Dict[str, Any]]:
    hp0 = bench.hparams
    return plan_cells(MODEL, MODE, SEED, hp0.lam, hp0.lr, hp0.weight_decay, hp0.decoupled_wd,
                      unhooked_parameter_names(bench))


def header(bench) -> Dict[str, Any]:
    hp0 = bench.hparams
    return {"model": MODEL, "seed": SEED, "mode": MODE, "batch_size": BATCH, "epochs": EPOCHS,
            "base_lambda": hp0.lam, "base_lr": hp0.lr, "cap": hp0.lr / hp0.lam,
            "weight_decay": hp0.weight_decay, "decoupled_wd": hp0.decoupled_wd,
            "arms": ARMS, "grid_limit": GRID_LIMIT, "train_subset": TRAIN_SUBSET,
            "log_every": LOG_EVERY, "nshards": NSHARDS}


def run_shard() -> None:
    bench = discover_benchmarks()[MODEL]
    cells = _planned(bench)
    mine = [c for c, o in zip(cells, shard_of(cells, NSHARDS)) if o == SHARD]
    prov = provenance()
    print(f"E19 | {MODEL} | seed={SEED} | mode={MODE} | shard {SHARD}/{NSHARDS}: {len(mine)} of "
          f"{len(cells)} cells | batch={BATCH} | epochs={EPOCHS} | device={DEVICE} | "
          f"smoke={SMOKE}\ntorch {prov['torch']} cuda {prov['cuda']} gpu {prov['gpu']} "
          f"tf32 matmul/cudnn {prov['tf32_matmul']}/{prov['tf32_cudnn']} workers {NUM_WORKERS} "
          f"commit {prov['git_commit']} dirty={prov['git_dirty']}", flush=True)
    results: Dict[str, Any] = {**header(bench), "shard": SHARD, "provenance": prov,
                               "planned": [cell_key(MODE, c["arm"], c["value"]) for c in mine],
                               "cells": {}}
    path = shard_path(SHARD) if NSHARDS > 1 else OUT
    for cell in mine:
        cell = dict(cell)
        arm, value = cell.pop("arm"), cell.pop("value")
        t0 = time.time()
        run = safe_run_cell(bench, MODE, **cell)
        acc = run.get("test_acc")
        print(f"  {MODE:6s} {arm:9s} {value:9.3g}  val={100*_final_val(run):6.2f}  "
              f"test={100*acc if acc is not None else float('nan'):6.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        results["cells"][cell_key(MODE, arm, value)] = {
            **run, "mode": MODE, "arm": arm, "value": value, "lam": cell["lam"],
            "lr": cell["lr"], "wd": cell["wd"], "freeze": cell["freeze"],
            "overrides": cell["overrides"],
        }
        _write(path, results)
    if NSHARDS == 1:
        merge([results])


def merge(shards: Optional[List[Dict[str, Any]]] = None) -> None:
    """Merge the shard files into the job's file and check completeness. Exits non-zero if a shard
    file or a planned cell is missing. The comparison with E16's stored cells (rule 0a) is
    e19_decisions.py's, which reads E16's files where they are."""
    bench = discover_benchmarks()[MODEL]
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
                raise SystemExit(f"E19 merge: {key} appears in two shards")
            cells[key] = c
    planned = [cell_key(MODE, c["arm"], c["value"]) for c in _planned(bench)]
    absent = [k for k in planned if k not in cells]
    unplanned = [k for k in cells if k not in planned]
    merged = {**header(bench), "provenance": [s.get("provenance") for s in shards],
              "missing_shards": missing, "absent_cells": absent, "unplanned_cells": unplanned,
              "cells": cells}
    _write(OUT, merged)
    print(f"E19 merge | {MODEL} | seed={SEED} | mode={MODE}: {len(cells)} of {len(planned)} "
          f"planned cells, {len(missing)} missing shard files, {len(unplanned)} unplanned\n"
          f"wrote {OUT}", flush=True)
    if missing or absent or unplanned:
        raise SystemExit(f"E19 merge: incomplete ({len(missing)} shard files, {len(absent)} cells "
                         f"missing, {len(unplanned)} unplanned)")


def main() -> None:
    check_production_settings()
    if MERGE:
        merge()
    else:
        run_shard()


if __name__ == "__main__":
    main()
