"""Is it the curvature that does not help, or the way it is being estimated?

``e2_lambda_at_fixed_gain.py`` lowered the damping constant across nine orders of magnitude with
the step size held still, and found that the five modes do not get better. But lowering the damping
is only half of what "let the curvature matter" means. It puts the curvature back at the right
*size* relative to the constant, and it leaves the *averaging* exactly as broken as it was: the
stored value is still about 92 % one batch of examples, and it still carries a residue of the
identity it was started from. So "curvature does not help" and "this way of estimating it is too
noisy to help" were still tangled together.

What this run separates. Two estimators, everything else identical::

  shipped     what ships today: gammas = (0.92, 0.008), i.e. 0.08 * old + 0.008 * new, started
              from the identity. 92 % of the state is the most recent batch, the state settles at
              1/115 of what it measures, and a residue of the identity is still there after k
              updates.
  corrected   a real average, started from a real observation. gamma = 0.8 collapses the same
              update to the AdaFisher paper's own Eq. (3), 0.8 * old + 0.2 * new, whose
              coefficients add up to one so the state settles at what it measures. Seeding from
              the first observation means no residue of the identity is ever in it.

Both are then swept over the same damping values at a fixed step-size cap, exactly as in the
previous experiment, so the two grids are directly comparable cell by cell.

What each outcome would mean. If the corrected estimator finds real gains where the shipped one
found at most about two accuracy points, then the curvature was never the problem and the averaging
was. If it gives the same picture -- flat, then collapsing -- then the curvature genuinely does not
pay for itself on these networks at this operating point, which is a finding about the method
rather than about this port.

Why batch 32. The stored curvature is proportional to one over the squared batch size, and at batch
32 a corrected ``diag`` is the one case in the whole study where a mode becomes genuinely
curvature-aware: its damping constant lands at the 22nd to 66th percentile of the spectrum instead
of the 85th to 100th.

Environment variables::

    E4_MODELS            comma-separated benchmarks
                         (default: mlp_ln_mnist,cnn_gn_cifar,resnet20_cifar)
    E4_MODES             comma-separated modes (default: all five)
    E4_LAMBDAS           comma-separated damping values (default: 1e-4,1e-6,1e-8,1e-10)
    E4_ESTIMATORS        comma-separated arms (default: shipped,corrected)
    E4_BATCH             batch size (default: 32)
    E4_GAMMA             the corrected average's single coefficient (default: 0.8)
    E4_FREEZE_UNHOOKED   "1" holds the parameters the optimizer does not precondition at their
                         initial values in EVERY arm, turning that confound into a constant
                         (default: "0")
    E4_SEED              seed (default: 0)
    E4_OUT               output path (default: fisher_ref/outputs/e4_fixed_average.json)

Output: the JSON at ``E4_OUT`` plus a table on standard output. ``collapsed`` and ``run_one`` are
imported from here by ``e5_unhooked_freeze_control.py``.
"""
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from benchmarks.common.loop import evaluate, train_under_budget
from benchmarks.common.optimizers import build_optimizer
from benchmarks.common.runner import discover_benchmarks
from benchmarks.common.schedules import NominalCosine
from fisher_ref.experiments.warmup_sgd_baseline import (
    ALLOW_DOWNLOAD,
    DATA_ROOT,
    DEVICE,
    EPOCHS,
    NUM_WORKERS,
    flatten_params,
)

MODELS = [m.strip() for m in os.environ.get(
    "E4_MODELS", "mlp_ln_mnist,cnn_gn_cifar,resnet20_cifar").split(",") if m.strip()]
MODES = [m.strip() for m in os.environ.get(
    "E4_MODES", "diag,kfac,ekfac,tkfac,tekfac").split(",") if m.strip()]
LAMBDAS = [float(x) for x in os.environ.get(
    "E4_LAMBDAS", "1e-4,1e-6,1e-8,1e-10").split(",") if x.strip()]
ESTIMATORS = [e.strip() for e in os.environ.get(
    "E4_ESTIMATORS", "shipped,corrected").split(",") if e.strip()]
BATCH = int(os.environ.get("E4_BATCH", "32"))
# Whether to hold the parameters the optimizer does not precondition at their initial values.
# Holding the cap means lr = lambda, so those parameters' step shrinks with lambda and they freeze
# on their own as the sweep goes down -- which confounds "curvature started to matter" with "those
# parameters stopped moving". The E5 control measured that confound at 0.9% of cnn_gn_cifar and
# found it worth -0.26 to +1.68 accuracy points there, against a +4.32 effect. It is 9.7% of
# vit_micro_cifar, so on that network it has to be an axis of the experiment rather than something
# the sweep does silently. Freezing in EVERY arm makes it a constant instead of a variable.
FREEZE_UNHOOKED = os.environ.get("E4_FREEZE_UNHOOKED", "0") == "1"
# 1 = rebuild ekfac/tekfac's eigenbasis inside the backward hook, before the gradient is projected
# into it, so the rescaling is measured in the basis precondition then uses. CLAUDE.md section 3
# measured that the two orderings move the applied step by 3.0e-3 at Lambda=1e-3 and by 8.3 at
# Lambda=1e-8, and says in so many words to fix the ordering before acting on fix S1, which lowers
# Lambda. This sweep goes to 1e-12, four orders below where the effect was measured, so for those
# two modes the ordering is not a detail here -- it is an axis. Ignored for the other three modes,
# where AdaFisherMulti raises on it.
EIG_BEFORE_RESCALE = os.environ.get("E4_EIG_BEFORE_RESCALE", "0") == "1"
GAMMA = float(os.environ.get("E4_GAMMA", "0.8"))       # the paper's own Eq. (3) value
SEED = int(os.environ.get("E4_SEED", "0"))
OUT = Path(os.environ.get("E4_OUT", str(ROOT / "fisher_ref/outputs/e4_fixed_average.json")))


def estimator_kwargs(estimator: str, mode: str = "") -> Dict[str, Any]:
    """The extra arguments to AdaFisherMulti that define an estimator arm."""
    if estimator == "shipped":
        kw: Dict[str, Any] = {}
    elif estimator == "corrected":
        kw = {"gamma": GAMMA, "ema_seed_first": True}
    else:
        raise ValueError(f"unknown estimator {estimator!r}")
    if EIG_BEFORE_RESCALE and mode in ("ekfac", "tekfac"):
        kw["eig_before_rescale"] = True
    return kw


def collapsed(run: Dict[str, Any]) -> bool:
    """Did this run stop learning? Three of the five modes never crash when the safety constant is
    too small: they settle quietly at chance-level accuracy and report a number. Detected, not
    assumed -- a run has collapsed if it never improved its validation accuracy by even one point
    over its own first epoch, or if anything is not finite."""
    accs = [a for a in run.get("epoch_val_acc", []) if a == a]
    if not accs or not math.isfinite(run.get("test_loss", float("nan"))):
        return True
    if len(accs) < 3:
        return False
    return max(accs) <= accs[0] + 0.01


def run_one(bench, mode: str, estimator: str, lam: float, lr: float,
            freeze_names=None) -> Dict[str, Any]:
    """One training run. Mirrors warmup_sgd_baseline.run_variant, with the batch size and the two
    estimator knobs added, and without the plain-momentum stand-in branch this experiment does not
    use.

    ``freeze_names`` holds parameters fixed at their initial values by clearing ``requires_grad``,
    which makes ``step()`` skip them (it already skips any parameter whose gradient is None). Used
    only by the E5 control, which asks whether freezing the parameters the optimizer does not
    precondition is what produces E4's gain. None reproduces the previous behaviour exactly.
    """
    hp = replace(bench.hparams, lam=lam, lr=lr)
    torch.manual_seed(SEED)   # identical init and data order across every cell
    model = bench.build_model(**{k: v[0] for k, v in bench.model_choices.items()}).to(DEVICE)
    if freeze_names:
        wanted = set(freeze_names)
        found = 0
        for name, p in model.named_parameters():
            if name in wanted:
                p.requires_grad_(False)
                found += 1
        if found != len(wanted):
            raise ValueError(f"freeze_names named {len(wanted)} parameters, matched {found}")
    theta0 = flatten_params(model)
    optimizer = build_optimizer(mode, model, hp, **estimator_kwargs(estimator, mode))

    train_loader, val_loader, test_loader = bench.build_data(
        DATA_ROOT, batch_size=BATCH, seed=SEED, num_workers=NUM_WORKERS,
        cutout=True, allow_download=ALLOW_DOWNLOAD, train_subset=None,
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
        eval_fn=eval_fn, prepare_batch=bench.prepare_batch,
        log_fn=lambda _msg: None,   # 240 runs x 15 epochs of per-epoch lines would
    )                               # bury the per-run lines that carry the result
    elapsed = time.time() - t0

    test_loss, test_acc = (float("nan"), float("nan"))
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, bench.loss_fn, DEVICE,
                                       prepare_batch=bench.prepare_batch,
                                       metric_fn=bench.metric_fn)
    return {
        "wall_s": elapsed, "n_steps": len(steps),
        "epoch_train_loss": [e.train_loss for e in epochs],
        "epoch_val_loss": [e.val_loss for e in epochs],
        "epoch_val_acc": [e.val_acc for e in epochs],
        "test_loss": test_loss, "test_acc": test_acc,
        "theta_move_from_init": float((flatten_params(model) - theta0).norm()),
    }


def safe_run_one(bench, mode, estimator, lam, lr, freeze_names=None):
    """A singular Kronecker factor at a small safety constant is an ordinary outcome here, not a
    reason to lose the rest of the grid."""
    try:
        return run_one(bench, mode, estimator, lam, lr, freeze_names=freeze_names)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        print(f"    [CRASHED] {mode}/{estimator} lambda={lam:g}: {type(exc).__name__}: {exc}",
              flush=True)
        return {"wall_s": float("nan"), "n_steps": 0, "epoch_train_loss": [],
                "epoch_val_loss": [], "epoch_val_acc": [], "test_loss": float("nan"),
                "test_acc": float("nan"), "theta_move_from_init": float("nan"),
                "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    benches = discover_benchmarks()
    print(f"E4: a fixed running average, at batch {BATCH} | device={DEVICE} | epochs={EPOCHS} | "
          f"seed={SEED}\nmodels={MODELS}\nmodes={MODES}\nlambdas={LAMBDAS}\n"
          f"estimators={ESTIMATORS} (corrected = gamma {GAMMA} + ema_seed_first)\n"
          f"freeze_unhooked={FREEZE_UNHOOKED}  "
          f"eig_before_rescale={EIG_BEFORE_RESCALE} (ekfac/tekfac only)\n", flush=True)

    results: Dict[str, Any] = {}
    for model_name in MODELS:
        bench = benches[model_name]
        base_lam, base_lr = bench.hparams.lam, bench.hparams.lr
        cap = base_lr / base_lam
        freeze_names = None
        if FREEZE_UNHOOKED:
            from fisher_ref.experiments.e5_unhooked_freeze_control import unhooked_parameter_names
            freeze_names = unhooked_parameter_names(bench)
        print(f"\n########## {model_name}  (default lambda={base_lam:g}, lr={base_lr:g}, "
              f"cap={cap:g}, batch {BATCH}, "
              f"unpreconditioned params {'FROZEN: ' + ', '.join(freeze_names) if freeze_names else 'trainable'}"
              f") ##########", flush=True)
        results[model_name] = {"base_lambda": base_lam, "base_lr": base_lr, "cap": cap,
                               "batch_size": BATCH, "freeze_unhooked": FREEZE_UNHOOKED,
                               "eig_before_rescale": EIG_BEFORE_RESCALE,
                               "frozen_names": freeze_names, "cells": {}}

        for mode in MODES:
            for estimator in ESTIMATORS:
                for lam in [base_lam] + LAMBDAS:
                    lr = base_lr * (lam / base_lam)      # the cap never moves
                    t0 = time.time()
                    run = safe_run_one(bench, mode, estimator, lam, lr, freeze_names)
                    bad = collapsed(run)
                    tag = "reference" if lam == base_lam else f"{lam:g}"
                    print(f"  {mode:7s} {estimator:9s} lambda={lam:9.1e} lr={lr:9.1e}  "
                          f"test_loss={run['test_loss']:8.4f} test_acc={run['test_acc']:7.4f}"
                          f"{'  [COLLAPSED]' if bad else ''}"
                          f"{'  [CRASHED]' if 'error' in run else ''}"
                          f"  ({time.time()-t0:.0f}s)", flush=True)
                    results[model_name]["cells"][f"{mode}|{estimator}|{tag}"] = {
                        **run, "mode": mode, "estimator": estimator, "lam": lam, "lr": lr,
                        "cap": lr / lam, "collapsed": bad,
                    }
                OUT.parent.mkdir(parents=True, exist_ok=True)
                with open(OUT, "w") as f:   # rewritten after every (mode, estimator): a crash
                    json.dump({"epochs": EPOCHS, "seed": SEED, "batch_size": BATCH,   # late in the
                               "gamma": GAMMA, "device": str(DEVICE),                  # grid must
                               "results": results}, f, indent=1)                       # cost little

    print("\n===== SUMMARY: test accuracy, and what the corrected average changes =====", flush=True)
    for model_name, blob in results.items():
        print(f"\n-- {model_name}  (batch {BATCH})", flush=True)
        head = "  ".join(f"{lam:>9.0e}" for lam in LAMBDAS)
        print(f"  {'mode':7s} {'estimator':9s} {'ref':>9s}  {head}", flush=True)
        for mode in MODES:
            for estimator in ESTIMATORS:
                cells = []
                for lam in LAMBDAS:
                    c = blob["cells"].get(f"{mode}|{estimator}|{lam:g}")
                    if c is None:
                        cells.append(f"{'-':>9s}")
                    elif c.get("error"):
                        cells.append(f"{'X':>9s}")
                    elif c["collapsed"]:
                        cells.append(f"{'C':>9s}")
                    else:
                        cells.append(f"{100*c['test_acc']:9.2f}")
                ref = blob["cells"].get(f"{mode}|{estimator}|reference")
                r = f"{100*ref['test_acc']:9.2f}" if ref and not ref["collapsed"] else f"{'C':>9s}"
                print(f"  {mode:7s} {estimator:9s} {r}  " + "  ".join(cells), flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
