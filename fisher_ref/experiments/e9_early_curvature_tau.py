"""E9: does the curvature EARLY in training tighten `tau`, or not?

``docs/reports/plan_lambda_dominance.md`` Part 6, E7, left one candidate explanation untested.

**The constant and its residual.** E4/E6/E7 located the best safety constant for each network by
sweeping it. Expressing that best value against the network's own mean curvature gives a
dimensionless number, `tau`. Across the 8 peaks that sit strictly inside their tested window, `tau`
still spreads by **10.8x**, against 33x for the raw best constant itself. So the relative rule
removes about a third of the variation and leaves the rest unexplained.

**The candidate.** Those mean curvatures all come from E1, which re-warms the optimizer's state from
a **half-trajectory checkpoint**. The sweeps, meanwhile, train from scratch. The curvature moves by
orders of magnitude over training (`early_curvature.py` measured that directly), so setting a
whole-run constant from a mid-training statistic may simply be reading the wrong number. If so,
the same `tau` computed against an **early** curvature should be tighter.

**What this measures.** For each network and each of the four Kronecker modes: train a fresh network
with that mode's own optimizer at the benchmark's own settings, at **batch 32** -- the batch the
sweeps used, which matters because the stored curvature goes as `1/batch^2` (E3) -- and snapshot the
undamped curvature spectrum at several points along the early trajectory. Then recompute `tau` from
each snapshot and report how much it spreads.

**Two choices that are not free, stated.**

*Why the first snapshot is at 1000 steps and not at step 0.* Every mode starts its running average
at the identity, and that seed fades as `0.08^k` per factor update. Below ten updates the state is
measuring the optimizer's start-up, not the network -- measured in `plan_lambda_dominance.md`'s E1
addendum, where `ekfac`'s spectrum at batch 512 reproduced `0.08^9` to three digits on six
unrelated networks. 1000 steps at `TCov=100` is ten updates, the same rule E1 itself follows. On
CIFAR at batch 32 that is 0.7 of an epoch out of the 15 the sweeps run, so it is genuinely early.

*Which safety constant to train with.* The default one. The point of a rule for setting the constant
is that you do not know the right value yet, so the only trajectory available in practice is the
default one. It also barely matters here: E1 measured the default constant to sit above every
direction of every network, so the trajectory is momentum either way.

Env vars: E9_MODELS, E9_MODES, E9_BATCH, E9_SNAPSHOTS, E9_SEED, E9_OUT, DATA_ROOT.
"""
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch
from adafisher_modes.optimizer import SUPPORTED_MODULES

from benchmarks.common.optimizers import build_optimizer
from benchmarks.common.runner import discover_benchmarks
from fisher_ref.experiments.e1_lambda_percentile import correction_factor, describe
from fisher_ref.experiments.lambda_vs_curvature import spectra

MODELS = [m.strip() for m in os.environ.get(
    "E9_MODELS", "cnn_gn_cifar,vit_micro_cifar,cct_2_3x2_cifar").split(",") if m.strip()]
MODES = [m.strip() for m in os.environ.get(
    "E9_MODES", "kfac,ekfac,tkfac,tekfac").split(",") if m.strip()]
BATCH = int(os.environ.get("E9_BATCH", "32"))
SNAPSHOTS = sorted(int(s) for s in os.environ.get(
    "E9_SNAPSHOTS", "1000,2000,4000,8000").split(",") if s.strip())
SEED = int(os.environ.get("E9_SEED", "0"))
DATA_ROOT = os.environ.get("DATA_ROOT", str(ROOT / "benchmarks/data"))
OUT = Path(os.environ.get("E9_OUT", str(ROOT / "fisher_ref/outputs/e9_early_curvature_tau.json")))

# The best safety constant each (network, mode) reached, from the sweeps. Only peaks that sit
# strictly INSIDE their tested window are listed: a peak at the edge of a grid is a bound, not a
# value, and cannot pin a constant. Source: plan_lambda_dominance.md Part 6, E4 and E7.
BEST_LAMBDA = {
    ("cnn_gn_cifar", "kfac"): 1e-8, ("cnn_gn_cifar", "ekfac"): 1e-8,
    ("cnn_gn_cifar", "tkfac"): 1e-8, ("cnn_gn_cifar", "tekfac"): 1e-8,
    ("vit_micro_cifar", "ekfac"): 1e-8, ("vit_micro_cifar", "tkfac"): 3e-9,
    ("vit_micro_cifar", "tekfac"): 1e-9,
    ("cct_2_3x2_cifar", "tkfac"): 3e-10,
}
# The same quantity computed from E1's mid-training curvature, for the comparison this run exists
# to make. plan_lambda_dominance.md Part 6, E7, Result 4.
MID_TRAINING_SPREAD = 10.8


def pooled_true_scale_mean(model, approx, mode, lam, batch_size):
    """Mean of the network's undamped curvature values, put back at the scale it estimates.

    Pooled over every hooked layer, which is the same summary E1 reports and the one E3 established
    as the usable one: the middle value of these lists sits inside a floor of exactly-zero entries
    and measures that floor instead of the curvature.
    """
    corr, _, _ = correction_factor(mode, approx, batch_size)
    pooled = []
    for m in model.modules():
        if type(m).__name__ not in SUPPORTED_MODULES:
            continue
        try:
            und, _ = spectra(approx, mode, m)
        except (KeyError, RuntimeError):
            continue
        pooled.append(und.clamp_min(0.0).numpy() * corr)
    if not pooled:
        return None
    return describe(np.concatenate(pooled), lam)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benches = discover_benchmarks()
    print(f"E9: does the early curvature tighten tau? | device={device} | batch={BATCH} | "
          f"seed={SEED}\nmodels={MODELS}\nmodes={MODES}\nsnapshots at steps {SNAPSHOTS}\n",
          flush=True)

    results: Dict[str, Any] = {}
    for model_name in MODELS:
        bench = benches[model_name]
        hp = bench.hparams
        train_loader, _, _ = bench.build_data(
            DATA_ROOT, batch_size=BATCH, seed=SEED, num_workers=2,
            cutout=True, allow_download=False, train_subset=None,
        )
        print(f"\n########## {model_name}  (lambda={hp.lam:g}, lr={hp.lr:g}, TCov={hp.tcov}, "
              f"batch {BATCH}) ##########", flush=True)
        results[model_name] = {"lambda": hp.lam, "batch_size": BATCH, "modes": {}}

        for mode in MODES:
            t0 = time.time()
            torch.manual_seed(SEED)     # same init and data order for every mode
            model = bench.build_model(**{k: v[0] for k, v in bench.model_choices.items()}).to(device)
            opt = build_optimizer(mode, model, hp)
            snaps: Dict[str, Any] = {}
            step, it = 0, iter(train_loader)
            while step < max(SNAPSHOTS):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(train_loader)
                    batch = next(it)
                x, y = bench.prepare_batch(batch)
                opt.zero_grad()
                bench.loss_fn(model(x.to(device)), y.to(device)).backward()
                opt.step()
                step += 1
                if step in SNAPSHOTS:
                    d = pooled_true_scale_mean(model, opt.approx, mode, hp.lam, BATCH)
                    if d is not None:
                        snaps[str(step)] = d
                        print(f"  {mode:7s} step {step:6d}  mean curvature "
                              f"{d['mean_over_lambda']*hp.lam:.4g}  (max {d['max_over_lambda']*hp.lam:.4g})",
                              flush=True)
            results[model_name]["modes"][mode] = snaps
            print(f"  {mode:7s} done ({time.time()-t0:.0f}s)", flush=True)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            with open(OUT, "w") as f:
                json.dump({"batch_size": BATCH, "seed": SEED, "snapshots": SNAPSHOTS,
                           "best_lambda": {f"{k[0]}|{k[1]}": v for k, v in BEST_LAMBDA.items()},
                           "results": results}, f, indent=1)

    print("\n===== tau = best lambda / mean curvature, at each point of the trajectory =====",
          flush=True)
    print("A rule works if tau is the SAME across every (network, mode) pair.\n", flush=True)
    header = "  ".join(f"{s:>10d}" for s in SNAPSHOTS)
    print(f"{'network':>18s} {'mode':7s} {'best lam':>9s}  {header}", flush=True)
    per_step: Dict[int, list] = {s: [] for s in SNAPSHOTS}
    for (net, mode), lam_best in BEST_LAMBDA.items():
        if net not in results or mode not in results[net]["modes"]:
            continue
        lam = results[net]["lambda"]
        row = []
        for s in SNAPSHOTS:
            d = results[net]["modes"][mode].get(str(s))
            if d is None:
                row.append(f"{'-':>10s}")
                continue
            mean_true = d["mean_over_lambda"] * lam
            tau = lam_best / mean_true if mean_true > 0 else float("nan")
            per_step[s].append(tau)
            row.append(f"{tau:10.3g}")
        print(f"{net:>18s} {mode:7s} {lam_best:9.0e}  " + "  ".join(row), flush=True)

    print(f"\n{'spread of tau (max/min)':>34s}  "
          + "  ".join(
              f"{(max(v)/min(v) if v and min(v) > 0 else float('nan')):10.1f}"
              for v in (per_step[s] for s in SNAPSHOTS)), flush=True)
    print(f"{'geometric mean of tau':>34s}  "
          + "  ".join(
              f"{math.exp(sum(map(math.log, v))/len(v)) if v and min(v) > 0 else float('nan'):10.3g}"
              for v in (per_step[s] for s in SNAPSHOTS)), flush=True)
    print(f"\nFor comparison, the same tau computed from E1's MID-TRAINING curvature spreads by "
          f"{MID_TRAINING_SPREAD:.1f}x over the same 8 pairs.", flush=True)
    print("If any column above is tighter than that, reading the curvature early is the better rule.",
          flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
