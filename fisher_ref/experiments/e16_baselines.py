"""E16's Adam/AdamW baselines, under E16's exact protocol, for reference only.

E16 compares ways to protect the division in ``ekfac``/``tekfac`` with each other; it has no
first-order baseline. The campaign's ``adam``/``adamw`` runs used another protocol (batch 128, 50
epochs, a wall-clock budget), so they cannot sit next to E16's cells. This driver reruns them the way
every E16 cell ran: batch 32, 15 epochs, ``NominalCosine``, 4 data-loader workers, the same seeded
initialisation and data order, so each baseline cell is paired by seed with every E16 cell.

**What is compared, and how it is chosen.** ``adam`` and ``adamw`` at the benchmark's own weight
decay (coupled for ``adam``, decoupled for ``adamw``), nothing frozen, as the benchmarks run them.
Each E16 family chose its value from a grid of 5-7, so each baseline gets one too: the benchmark's
``baseline_lr`` times {1/3, 1, 3}, selected on final validation accuracy as E16's rule 1 does. This
is a reference added after E16's results were read; it votes in none of E16's rules.

One job = (network, seed, optimizer): 3 cells. Output:
``fisher_ref/outputs/e16b_<model>_s<seed>_<optimizer>.json``.

**Grid extension (after the first 40 jobs).** At batch 32 the selected lr was the grid's top (x3) in
7 of 8 (network, optimizer) pairs, still rising, so the optimum was not located. A second job per
(network, seed, optimizer) runs x{10, 30} (``E16B_FACTORS=10:30``, ``E16B_TAG=ext``). On ResNet-50 the opposite happened -- both
baselines were best at the grid's *bottom* (x1/3) and fell monotonically above it -- so a third job
runs x{1/10, 1/30} (``E16B_TAG=ext2``) into
``..._<optimizer>_ext.json``; the report reads both files as one grid.

Environment variables::

    E16B_MODEL, E16B_SEED, E16B_OPT (adam | adamw)   required
    E16B_FACTORS   ':'-separated lr factors (default: 1/3:1:3; also 10:30 and 1/10:1/30)
    E16B_TAG       file suffix, e.g. "ext" -> ..._<optimizer>_ext.json (default: none)
    E16B_DIR       output directory (default: fisher_ref/outputs)
    E16B_SMOKE     "1" allows non-production settings (WARMUP_SGD_*, E16B_TRAIN_SUBSET)
    WARMUP_SGD_*   as elsewhere; production is 15 epochs, 4 workers
"""
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch  # noqa: E402

from benchmarks.common.loop import evaluate, train_under_budget  # noqa: E402
from benchmarks.common.optimizers import build_optimizer  # noqa: E402
from benchmarks.common.runner import discover_benchmarks  # noqa: E402
from benchmarks.common.schedules import NominalCosine  # noqa: E402
from fisher_ref.experiments.warmup_sgd_baseline import (  # noqa: E402
    ALLOW_DOWNLOAD,
    DATA_ROOT,
    DEVICE,
    EPOCHS,
    NUM_WORKERS,
    flatten_params,
)

MODELS = ("cnn_gn_cifar", "vit_micro_cifar", "cct_2_3x2_cifar", "resnet20_cifar", "resnet50_cifar")
OPTS = ("adam", "adamw")
ALLOWED_FACTORS = (1 / 30, 1 / 10, 1 / 3, 1.0, 3.0, 10.0, 30.0)
LR_FACTORS = tuple(
    {"1/30": 1 / 30, "1/10": 1 / 10, "1/3": 1 / 3}.get(f.strip(), None) or float(f)
    # ":"-separated: sbatch --export splits its argument on commas
    for f in (os.environ.get("E16B_FACTORS") or "1/3:1:3").replace(",", ":").split(":") if f.strip())
TAG = os.environ.get("E16B_TAG", "")
BATCH = 32
MODEL = os.environ.get("E16B_MODEL", "")
SEED = int(os.environ.get("E16B_SEED", "0"))
OPT = os.environ.get("E16B_OPT", "")
SMOKE = os.environ.get("E16B_SMOKE", "0") == "1"
TRAIN_SUBSET = int(os.environ["E16B_TRAIN_SUBSET"]) if os.environ.get("E16B_TRAIN_SUBSET") else None
DIR = Path(os.environ.get("E16B_DIR", str(ROOT / "fisher_ref/outputs")))


def run_cell(bench, lr: float) -> Dict[str, Any]:
    """E16's ``run_cell`` with the optimizer swapped: same seed, init, data order, schedule."""
    hp = replace(bench.hparams, baseline_lr=lr)
    torch.manual_seed(SEED)
    model = bench.build_model(**{k: v[0] for k, v in bench.model_choices.items()}).to(DEVICE)
    theta0 = flatten_params(model)
    optimizer = build_optimizer(OPT, model, hp)
    train_loader, val_loader, test_loader = bench.build_data(
        DATA_ROOT, batch_size=BATCH, seed=SEED, num_workers=NUM_WORKERS,
        cutout=True, allow_download=ALLOW_DOWNLOAD, train_subset=TRAIN_SUBSET,
    )
    scheduler = NominalCosine(optimizer, t_max=EPOCHS)
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
    return {"wall_s": elapsed, "n_steps": len(steps),
            "epoch_train_loss": [e.train_loss for e in epochs],
            "epoch_val_loss": [e.val_loss for e in epochs],
            "epoch_val_acc": [e.val_acc for e in epochs],
            "test_loss": test_loss, "test_acc": test_acc,
            "theta_move_from_init": float((flatten_params(model) - theta0).norm())}


def main() -> None:
    problems = []
    if MODEL not in MODELS or OPT not in OPTS:
        raise SystemExit(f"E16B: need E16B_MODEL in {MODELS} and E16B_OPT in {OPTS}")
    if not LR_FACTORS or (not set(LR_FACTORS) <= set(ALLOWED_FACTORS) and not SMOKE):
        raise SystemExit(f"E16B: lr factors {LR_FACTORS} outside {ALLOWED_FACTORS}")
    if (EPOCHS != 15 or NUM_WORKERS != 4 or TRAIN_SUBSET is not None) and not SMOKE:
        raise SystemExit(f"E16B: non-production settings (epochs {EPOCHS}, workers {NUM_WORKERS}, "
                         f"subset {TRAIN_SUBSET}) without E16B_SMOKE=1")
    import fisher_ref.experiments.e16_floor_clip as e16  # provenance only
    bench = discover_benchmarks()[MODEL]
    base = bench.hparams.baseline_lr
    out = DIR / f"e16b_{MODEL}_s{SEED}_{OPT}{'_' + TAG if TAG else ''}.json"
    prov = e16.provenance()
    res: Dict[str, Any] = {"model": MODEL, "seed": SEED, "opt": OPT, "epochs": EPOCHS,
                           "batch_size": BATCH, "baseline_lr": base,
                           "weight_decay": bench.hparams.weight_decay, "smoke": SMOKE,
                           "lr_factors": list(LR_FACTORS), "tag": TAG,
                           "provenance": prov, "problems": problems, "cells": {}}
    print(f"E16B | {MODEL} | seed={SEED} | {OPT} | lr {base:g} x {LR_FACTORS} | "
          f"commit {prov['git_commit']} dirty={prov['git_dirty']} gpu {prov['gpu']}", flush=True)
    for f in LR_FACTORS:
        lr = base * f
        t0 = time.time()
        try:
            cell = run_cell(bench, lr)
        except Exception as exc:  # noqa: BLE001 -- a crash is an outcome, recorded
            cell = {"crashed": True, "error": f"{type(exc).__name__}: {exc}", "test_acc": None,
                    "epoch_val_acc": []}
        cell.update(opt=OPT, lr=lr, lr_factor=f)
        res["cells"][f"{OPT}|{lr:g}"] = cell
        print(f"  {OPT:5s} lr={lr:9.3g}  val={100 * (cell['epoch_val_acc'] or [float('nan')])[-1]:6.2f}"
              f"  test={100 * (cell['test_acc'] or float('nan')):6.2f}  ({time.time() - t0:.0f}s)",
              flush=True)
        e16._write(out, res)


if __name__ == "__main__":
    main()
