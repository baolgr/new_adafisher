"""Which of two corrections to the plain-momentum stand-in helped, and which hurt?

On one network of three, ``cnn_gn_cifar``, fixing both together made the mismatch between real
``kfac`` and the stand-in *worse*. Two independent changes went into that rerun, bundled into one
job:

(a) the stand-in's schedule: ``lam + decay^k`` (a guess at the general shape) became
    ``(decay^k + sqrt(lam))^2``, which is derived from ``kfac``'s actual two-factor damping;
(b) the hooked/unhooked boundary: every parameter used to get the schedule's divisor; now only
    parameters belonging to a module type ``AdaFisherMulti`` actually hooks get it. ``GroupNorm``
    -- the one unhooked type among these networks -- now gets divisor one, matching real ``kfac``'s
    own fallback path, instead of the same shrinking divisor as everything else.

This script runs the two combinations neither earlier run covered -- corrected schedule with no
hooked boundary (which isolates (a)), and the old schedule with the hooked boundary respected
(which isolates (b)) -- alongside the two already measured, so all four cells of the two-by-two
print side by side from one run.

It reuses ``warmup_sgd_baseline.py``'s own stand-in and hooked-parameter helper unchanged; this
script only supplies the ``hooked_ids`` argument that module's ``run_variant()`` does not expose as
a toggle. The real ``kfac`` baseline and the seed-noise floor are read back from that script's
already-written JSON rather than re-run: neither involves the stand-in, so nothing here can change
them.

Environment variables: the same as ``warmup_sgd_baseline.py`` -- ``WARMUP_SGD_EPOCHS``,
``WARMUP_SGD_SEEDS``, ``WARMUP_SGD_DEVICE``, ``WARMUP_SGD_DATA_ROOT``, ``WARMUP_SGD_NUM_WORKERS``,
``WARMUP_SGD_ALLOW_DOWNLOAD``.

Reads: ``fisher_ref/outputs/warmup_sgd_baseline.json``.
Writes: ``fisher_ref/outputs/cnn_gn_ablation.json`` plus a table on standard output.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from warmup_sgd_baseline import (
    ALLOW_DOWNLOAD,
    DATA_ROOT,
    DEVICE,
    EPOCHS,
    NUM_WORKERS,
    OUT,
    SEEDS,
    WarmupMomentumSGD,
    flatten_params,
    hooked_param_ids,
)

from benchmarks.common.loop import evaluate, train_under_budget
from benchmarks.common.runner import Benchmark, discover_benchmarks
from benchmarks.common.schedules import NominalCosine

MODEL = "cnn_gn_cifar"


def run(bench: Benchmark, schedule: str, respect_hooked: bool) -> Dict[str, Any]:
    hp = bench.hparams
    seed = SEEDS[0]
    torch.manual_seed(seed)
    model_kwargs = {k: v[0] for k, v in bench.model_choices.items()}
    model = bench.build_model(**model_kwargs).to(DEVICE)
    theta0 = flatten_params(model)
    hooked_ids = hooked_param_ids(model) if respect_hooked else None
    optimizer = WarmupMomentumSGD(
        model.parameters(), lr=hp.lr, beta=hp.beta, Lambda=hp.lam,
        gamma0=hp.gammas[0], TCov=hp.tcov, weight_decay=hp.weight_decay,
        schedule=schedule, hooked_ids=hooked_ids,
    )
    train_loader, val_loader, test_loader = bench.build_data(
        DATA_ROOT, batch_size=bench.batch_size, seed=seed, num_workers=NUM_WORKERS,
        cutout=True, allow_download=ALLOW_DOWNLOAD, train_subset=None,
    )
    scheduler = NominalCosine(optimizer, t_max=EPOCHS)
    eval_fn = lambda: evaluate(  # noqa: E731
        model, val_loader, bench.loss_fn, DEVICE,
        prepare_batch=bench.prepare_batch, metric_fn=bench.metric_fn,
    )
    t0 = time.time()
    steps, epochs = train_under_budget(
        model, optimizer, train_loader, bench.loss_fn, budget_s=float("inf"), max_epochs=EPOCHS,
        device=DEVICE, scheduler=scheduler, eval_fn=eval_fn, prepare_batch=bench.prepare_batch,
        log_fn=print,
    )
    test_loss, test_acc = evaluate(model, test_loader, bench.loss_fn, DEVICE,
                                   prepare_batch=bench.prepare_batch, metric_fn=bench.metric_fn)
    theta_final = flatten_params(model)
    return {
        "schedule": schedule,
        "respect_hooked": respect_hooked,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "final_train_loss": epochs[-1].train_loss if epochs else float("nan"),
        "theta_move_from_init": float((theta_final - theta0).norm()),
        "wall_s": time.time() - t0,
    }


def main() -> None:
    benches = discover_benchmarks()
    bench = benches[MODEL]

    payload = json.load(open(OUT))["results"][MODEL]
    by_label = {r["label"]: r for r in payload}
    kfac0 = by_label[f"kfac_seed{SEEDS[0]}"]
    kfac1 = by_label[f"kfac_seed{SEEDS[1]}"]
    noise_loss = abs(kfac0["test_loss"] - kfac1["test_loss"])
    noise_acc = abs(kfac0["test_acc"] - kfac1["test_acc"])
    print(f"cnn_gn_ablation | device={DEVICE} | epochs={EPOCHS} | seed={SEEDS[0]}", flush=True)
    print(f"real kfac_seed{SEEDS[0]}: test_loss={kfac0['test_loss']:.4f} "
          f"test_acc={kfac0['test_acc']*100:.2f}%  |  seed-noise floor: "
          f"test_loss={noise_loss:.4f} test_acc={noise_acc*100:.2f}pp\n", flush=True)

    configs = [
        ("A", "ekfac", False, "old formula, no hooked boundary (Step 9's original stand-in)"),
        ("B", "kfac", False, "corrected formula, no hooked boundary (isolates the formula fix)"),
        ("C", "ekfac", True, "old formula, hooked boundary respected (isolates the GroupNorm fix)"),
        ("D", "kfac", True, "corrected formula, hooked boundary respected (Step 11's rerun)"),
    ]
    results = []
    print(f"{'cell':>5s} {'test_loss':>10s} {'test_acc':>9s} {'ratio(loss)':>12s} "
          f"{'ratio(acc)':>11s}  description", flush=True)
    for cell, schedule, respect_hooked, desc in configs:
        r = run(bench, schedule, respect_hooked)
        r["cell"] = cell
        d_loss = abs(r["test_loss"] - kfac0["test_loss"])
        d_acc = abs(r["test_acc"] - kfac0["test_acc"])
        ratio_loss = d_loss / max(noise_loss, 1e-12)
        ratio_acc = d_acc / max(noise_acc, 1e-12)
        r["ratio_loss"] = ratio_loss
        r["ratio_acc"] = ratio_acc
        results.append(r)
        print(f"{cell:>5s} {r['test_loss']:10.4f} {r['test_acc']*100:8.2f}% "
              f"{ratio_loss:12.2f}x {ratio_acc:11.2f}x  {desc}", flush=True)

    out = ROOT / "fisher_ref/outputs/cnn_gn_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"real_kfac_seed0": kfac0, "real_kfac_seed1": kfac1,
                   "noise_floor": {"test_loss": noise_loss, "test_acc": noise_acc},
                   "cells": results}, f, indent=1)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
