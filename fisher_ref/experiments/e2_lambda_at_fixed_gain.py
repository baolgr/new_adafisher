"""Sweep the damping constant while holding the step-size cap constant.

The point, in one paragraph. Wherever the curvature estimate is negligible next to the damping
constant -- which earlier measurements found to be everywhere -- one update is the learning rate
over the damping, times the momentum. So that ratio is not a detail: it is the **cap** on the step
size, the value the update takes in any direction the optimizer sees as flat. The damping alone
decides how much of the spectrum sits *below* that cap and therefore gets genuinely shrunk. An
earlier sweep lowered the damping from 1e-3 to 1e-8 with the learning rate held at 1e-3, which
raised the cap by a factor of 100 000 at the same time -- so when the networks stopped training,
there was no way to tell whether curvature had started to bite or the steps had simply exploded.
Two things changed at once.

This script separates them. At each damping value it runs two arms::

  fixed_lr     the learning rate stays at the benchmark's own value. This is the earlier protocol,
               reproduced so the comparison is against a measured baseline rather than a remembered
               one.
  fixed_gain   the learning rate is scaled with the damping so the cap never moves. Only the
               fraction of the spectrum below the cap changes. This is the one-variable experiment.

Prediction, and what falsifies it. If the earlier collapse to chance-level accuracy was an
exploding step size, ``fixed_gain`` still trains where ``fixed_lr`` collapses, and its gap from the
plain-momentum placebo grows smoothly as the damping falls. If ``fixed_gain`` collapses too, the
problem is not the step size but the estimator: either noise (the stored curvature is about 92 %
a single minibatch, and it is inverted without a square root) or the exact rank deficiency that
makes the two factored-inverse modes fail in ``linalg.inv``. Both outcomes are informative.

Note on the algebra this rests on: for all five modes, multiplying the stored factors by ``c`` is
identical to dividing the damping by ``c^2`` and the whole step by ``c^2``. So this sweep also
stands in for "what if the curvature were stored at its true scale" -- no change to any optimizer
file is needed to ask that question. The equivalence is exact only once the identity seed each mode
starts from has decayed, which is why the runs are fifteen epochs and not two.

A run that silently settles at chance-level accuracy is detected rather than assumed: ``collapsed``
flags a run that never improved its validation accuracy by even one point over its own first epoch,
or that produced a non-finite loss. Every ratio computed from such a run would be meaningless.

Environment variables::

    E2_MODELS     comma-separated benchmarks (default: mlp_ln_mnist,cnn_gn_cifar,resnet20_cifar)
    E2_MODES      comma-separated modes (default: all five)
    E2_LAMBDAS    comma-separated damping values (default: 1e-4,1e-6,1e-8,1e-10,1e-12)
    E2_PROTOCOLS  comma-separated arms (default: fixed_lr,fixed_gain)
    E2_SEED       seed (default: 0)
    E2_OUT        output path (default: fisher_ref/outputs/e2_lambda_at_fixed_gain.json)

The training knobs shared with ``warmup_sgd_baseline.py`` are read from that module's own
``WARMUP_SGD_*`` variables (epochs, device, data root, workers).

Output: the JSON at ``E2_OUT`` plus a table on standard output.
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

from benchmarks.common.runner import discover_benchmarks

# One training loop for the whole lambda-dominance thread: warmup_sgd_baseline's run_variant is
# what Steps 9-15 used, so anything measured here is comparable with them step for step. It grew
# an `lr_override` for this experiment, additive and inert by default.
from fisher_ref.experiments.warmup_sgd_baseline import (
    DEVICE,
    EPOCHS,
    run_variant,
    safe_run_variant,
)

MODELS = [m.strip() for m in os.environ.get(
    "E2_MODELS", "mlp_ln_mnist,cnn_gn_cifar,resnet20_cifar").split(",") if m.strip()]
MODES = [m.strip() for m in os.environ.get(
    "E2_MODES", "diag,kfac,ekfac,tkfac,tekfac").split(",") if m.strip()]
LAMBDAS = [float(x) for x in os.environ.get(
    "E2_LAMBDAS", "1e-4,1e-6,1e-8,1e-10,1e-12").split(",") if x.strip()]
PROTOCOLS = [p.strip() for p in os.environ.get(
    "E2_PROTOCOLS", "fixed_lr,fixed_gain").split(",") if p.strip()]
SEED = int(os.environ.get("E2_SEED", "0"))
OUT = Path(os.environ.get("E2_OUT", str(ROOT / "fisher_ref/outputs/e2_lambda_at_fixed_gain.json")))


def collapsed(run: Dict[str, Any]) -> bool:
    """Did this run stop learning? Step 15 found that three of the five modes never crash at a
    small Lambda -- they silently settle at chance-level accuracy and report a number, which is
    why every ratio computed from such a run is meaningless. Detected here, not assumed: a run has
    collapsed if it never improved its validation accuracy by even one point over its own first
    epoch, or if anything is not finite.
    """
    accs = [a for a in run.get("epoch_val_acc", []) if a == a]
    if not accs or not math.isfinite(run.get("test_loss", float("nan"))):
        return True
    if len(accs) < 3:
        # Too few epochs to apply the improvement rule -- with one epoch it is trivially true and
        # would flag a perfectly healthy run (measured on a 1-epoch smoke: 95.0% test accuracy,
        # reported as collapsed). Production runs are 15 epochs; short ones are simply not judged.
        return False
    return max(accs) <= accs[0] + 0.01


def main() -> None:
    benches = discover_benchmarks()
    print(f"E2: lambda sweep at a fixed step-size cap | device={DEVICE} | epochs={EPOCHS} | "
          f"seed={SEED}\nmodels={MODELS}\nmodes={MODES}\nlambdas={LAMBDAS}\n"
          f"protocols={PROTOCOLS}\n", flush=True)

    results: Dict[str, Any] = {}
    for model_name in MODELS:
        bench = benches[model_name]
        base_lam, base_lr = bench.hparams.lam, bench.hparams.lr
        cap = base_lr / base_lam
        print(f"\n########## {model_name}  (default lambda={base_lam:g}, lr={base_lr:g}, "
              f"cap lr/lambda={cap:g}) ##########", flush=True)
        results[model_name] = {"base_lambda": base_lam, "base_lr": base_lr, "cap": cap, "cells": {}}

        for mode in MODES:
            # The reference point: the benchmark's own settings, unchanged. Both protocols pass
            # through it by construction, so it is run once and shared.
            t0 = time.time()
            ref = safe_run_variant(bench, SEED, mode, warmup_sgd=False)
            ref_bad = collapsed(ref)
            print(f"  {mode:7s} lambda={base_lam:9.1e} (reference)        "
                  f"test_loss={ref['test_loss']:8.4f} test_acc={ref['test_acc']:7.4f}"
                  f"{'  [COLLAPSED]' if ref_bad else ''}  ({time.time()-t0:.0f}s)", flush=True)
            results[model_name]["cells"][f"{mode}|reference|{base_lam:g}"] = {
                **{k: ref[k] for k in ("label", "test_loss", "test_acc", "n_steps",
                                       "epoch_val_acc", "epoch_val_loss", "theta_move_from_init")},
                "mode": mode, "protocol": "reference", "lam": base_lam, "lr": base_lr,
                "cap": cap, "collapsed": ref_bad,
            }

            for lam in LAMBDAS:
                for proto in PROTOCOLS:
                    lr = base_lr if proto == "fixed_lr" else base_lr * (lam / base_lam)
                    t0 = time.time()
                    run = safe_run_variant_lr(bench, SEED, mode, lam, lr)
                    bad = collapsed(run)
                    print(f"  {mode:7s} lambda={lam:9.1e} {proto:10s} lr={lr:9.1e} "
                          f"cap={lr/lam:9.3g}  test_loss={run['test_loss']:8.4f} "
                          f"test_acc={run['test_acc']:7.4f}"
                          f"{'  [COLLAPSED]' if bad else ''}"
                          f"{'  [CRASHED]' if 'error' in run else ''}"
                          f"  ({time.time()-t0:.0f}s)", flush=True)
                    results[model_name]["cells"][f"{mode}|{proto}|{lam:g}"] = {
                        **{k: run[k] for k in ("label", "test_loss", "test_acc", "n_steps",
                                               "epoch_val_acc", "epoch_val_loss",
                                               "theta_move_from_init")},
                        "mode": mode, "protocol": proto, "lam": lam, "lr": lr,
                        "cap": lr / lam, "collapsed": bad, "error": run.get("error"),
                    }
            OUT.parent.mkdir(parents=True, exist_ok=True)
            with open(OUT, "w") as f:   # rewritten after every mode: a crash late in the grid
                json.dump({"epochs": EPOCHS, "seed": SEED, "device": str(DEVICE),
                           "results": results}, f, indent=1)   # must not cost the modes already done

    print("\n===== SUMMARY: does the network still train? =====", flush=True)
    print("(test accuracy; C = collapsed to chance, X = crashed)\n", flush=True)
    for model_name, blob in results.items():
        print(f"-- {model_name}", flush=True)
        header = "  ".join(f"{lam:>9.0e}" for lam in LAMBDAS)
        print(f"  {'mode':7s} {'protocol':10s} {'ref':>9s}  {header}", flush=True)
        for mode in MODES:
            for proto in PROTOCOLS:
                ref = blob["cells"].get(f"{mode}|reference|{blob['base_lambda']:g}")
                cells = []
                for lam in LAMBDAS:
                    c = blob["cells"].get(f"{mode}|{proto}|{lam:g}")
                    if c is None:
                        cells.append(f"{'-':>9s}")
                    elif c.get("error"):
                        cells.append(f"{'X':>9s}")
                    elif c["collapsed"]:
                        cells.append(f"{'C':>9s}")
                    else:
                        cells.append(f"{100*c['test_acc']:9.2f}")
                r = f"{100*ref['test_acc']:9.2f}" if ref and not ref["collapsed"] else f"{'C':>9s}"
                print(f"  {mode:7s} {proto:10s} {r}  " + "  ".join(cells), flush=True)
    print(f"\nwrote {OUT}", flush=True)


def safe_run_variant_lr(bench, seed, mode, lam, lr):
    """``safe_run_variant`` with an lr override. Kept here rather than widening that function's
    signature a second time: its own callers (Steps 9-15) never vary lr."""
    try:
        return run_variant(bench, seed, mode, warmup_sgd=False,
                           lam_override=lam, lr_override=lr)
    except Exception as exc:  # noqa: BLE001 -- a singular Kronecker factor must not abort the grid
        print(f"  [CRASHED] {mode} lambda={lam:g} lr={lr:g}: {type(exc).__name__}: {exc}",
              flush=True)
        return {"label": f"{mode}_seed{seed}", "seed": seed, "wall_s": float("nan"), "n_steps": 0,
                "step_loss": [], "epoch_train_loss": [], "epoch_val_loss": [], "epoch_val_acc": [],
                "test_loss": float("nan"), "test_acc": float("nan"),
                "theta_move_from_init": float("nan"), "theta_final_norm": float("nan"),
                "theta_final": None, "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    main()
