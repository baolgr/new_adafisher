"""E15: does one safety constant per layer beat the best single one?

``docs/reports/plan_lambda_dominance.md``, section "E15 -- pre-registered", is the specification.
This driver implements it and nothing else; anything it does differently from that section is a
deviation and must be recorded there.

**What it runs.** For one network, one seed, and each of three modes (``kfac``, ``ekfac``,
``tekfac``), four arm families, every run 15 epochs at batch 32 with the cosine schedule, the
shipped estimator, and ``eig_before_rescale`` on for ``ekfac``/``tekfac``:

  reference   the default ``lambda``, i.e. what the benchmark ships
  single      one ``lambda`` for the whole network, 5 values around E13/E14's optimum
  s1b         ``lambda_l = tau * c_l`` for each layer (``damping="layer_relative"``), 8 values of tau
  netadapt    one ``lambda(t) = tau * c_net(t)`` for the network (``"network_relative"``), same tau

All four hold the step-size cap the same way: the optimizer's ``lr`` is the cap
(``base_lr / base_lambda``) and ``hold_cap=True`` multiplies each preconditioned direction by the
``lambda`` inside it. For ``single`` that is exactly E7-E14's ``lr = cap * lambda``
(``tests/test_relative_damping.py``). The parameters the optimizer does not precondition are frozen
in every arm. Decoupled weight decay is applied at the benchmark's own rate, ``base_lr * wd`` at the
top of the cosine, whatever ``lambda`` is (Part 5, rule 3): with ``lr = cap`` that means a decay
coefficient of ``wd * base_lr / cap``. Coupled decay is added to the gradient and needs nothing.

**Two protocol controls.**

  repro    seed 0 only: one single-``lambda`` cell per mode run exactly as E13/E14 ran it (unfrozen,
           ``lr = cap * lambda``, no ``hold_cap``, decoupled decay scaled by that ``lr``). It must
           reproduce E13/E14's seed-0 test accuracy to 0.00 points; the check is printed at the end.
  wdctrl   decoupled-decay networks only, every seed: the same cell with the decay at its fixed
           rate. Paired by seed with E13's cell at the same ``lambda``, it measures how much of
           E13's gain was the decay switching off.

**Recorded along the way.** Every ``E15_LOG_EVERY`` steps, for every hooked layer: the ``lambda``
in effect, the layer's mean stored curvature ``c_l``, and the fraction of its directions whose
curvature exceeds that ``lambda``.

Environment variables::

    E15_MODEL          one benchmark (default: cnn_gn_cifar)
    E15_MODES          comma-separated (default: kfac,ekfac,tekfac)
    E15_ARMS           comma-separated subset of reference,single,s1b,netadapt,repro,wdctrl
                       (default: all that apply)
    E15_SEED           seed (default: 0)
    E15_BATCH          batch size (default: 32)
    E15_LOG_EVERY      steps between two lambda logs (default: 1000)
    E15_GRID_LIMIT     keep only the first k values of every grid -- for a smoke run (default: all)
    E15_TRAIN_SUBSET   truncate the training split -- for a smoke run (default: none)
    E15_OUT            output path (default: fisher_ref/outputs/e15_layer_damping_<model>_s<seed>.json)
    WARMUP_SGD_EPOCHS, WARMUP_SGD_DATA_ROOT, WARMUP_SGD_DEVICE, WARMUP_SGD_NUM_WORKERS as elsewhere.
"""
import json
import math
import os
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
from fisher_ref.experiments.lambda_vs_curvature import spectra
from fisher_ref.experiments.warmup_sgd_baseline import (
    ALLOW_DOWNLOAD,
    DATA_ROOT,
    DEVICE,
    EPOCHS,
    NUM_WORKERS,
    flatten_params,
)

MODEL = os.environ.get("E15_MODEL", "cnn_gn_cifar")
MODES = [m.strip() for m in os.environ.get("E15_MODES", "kfac,ekfac,tekfac").split(",") if m.strip()]
ARMS = [a.strip() for a in os.environ.get(
    "E15_ARMS", "reference,single,s1b,netadapt,repro,wdctrl").split(",") if a.strip()]
SEED = int(os.environ.get("E15_SEED", "0"))
BATCH = int(os.environ.get("E15_BATCH", "32"))
LOG_EVERY = int(os.environ.get("E15_LOG_EVERY", "1000"))
GRID_LIMIT = int(os.environ["E15_GRID_LIMIT"]) if os.environ.get("E15_GRID_LIMIT") else None
TRAIN_SUBSET = int(os.environ["E15_TRAIN_SUBSET"]) if os.environ.get("E15_TRAIN_SUBSET") else None
OUT = Path(os.environ.get(
    "E15_OUT", str(ROOT / f"fisher_ref/outputs/e15_layer_damping_{MODEL}_s{SEED}.json")))

# The grids, exactly as pre-registered. Single-lambda: 5 values at half-decade spacing around
# E13/E14's five-seed optimum. Tau: 8 values reaching at least 4x past the network-level tau the
# single-lambda optima translate into (0.008-0.21 for ekfac/tekfac, 0.25-3.3 for kfac).
SINGLE_GRID = {
    ("cnn_gn_cifar", "kfac"): [1e-9, 3e-10, 1e-10, 3e-11, 1e-11],
    ("cnn_gn_cifar", "ekfac"): [3e-10, 1e-10, 3e-11, 1e-11, 3e-12],
    ("cnn_gn_cifar", "tekfac"): [3e-10, 1e-10, 3e-11, 1e-11, 3e-12],
    ("vit_micro_cifar", "kfac"): [1e-10, 3e-11, 1e-11, 3e-12, 1e-12],
    ("vit_micro_cifar", "ekfac"): [1e-9, 3e-10, 1e-10, 3e-11, 1e-11],
    ("vit_micro_cifar", "tekfac"): [1e-9, 3e-10, 1e-10, 3e-11, 1e-11],
}
TAU_GRID = {
    "kfac": [30.0, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01],
    "ekfac": [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4],
    "tekfac": [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4],
}
# The seed-0 cells the repro control must reproduce, with the test accuracy E13/E14 recorded.
REPRO = {
    ("cnn_gn_cifar", "kfac"): (1e-10, 65.18),
    ("cnn_gn_cifar", "ekfac"): (3e-11, 67.41),
    ("cnn_gn_cifar", "tekfac"): (1e-10, 68.55),
    ("vit_micro_cifar", "kfac"): (1e-11, 54.23),
    ("vit_micro_cifar", "ekfac"): (1e-10, 54.36),
    ("vit_micro_cifar", "tekfac"): (1e-10, 54.92),
}


def _grid(values: List[float]) -> List[float]:
    return values if GRID_LIMIT is None else values[:GRID_LIMIT]


def _lambda_log(opt, names: Dict[Any, str], mode: str) -> Dict[str, List[float]]:
    """``{layer: [lambda in effect, mean stored curvature, fraction of directions above lambda]}``."""
    out: Dict[str, List[float]] = {}
    for module in opt.modules:
        c = opt.approx.mean_curvature(module)
        if c is None:
            continue
        lam = float(opt.approx.lambda_for(module))
        try:
            und, _ = spectra(opt.approx, mode, module)
            frac = float((und > lam).double().mean())
        except (KeyError, RuntimeError):   # no eigenbasis / inverse yet
            frac = float("nan")
        out[names[module]] = [lam, float(c), frac]
    return out


def run_cell(bench, mode: str, *, lam: float, lr: float, wd: float, freeze: Optional[List[str]],
             overrides: Dict[str, Any]) -> Dict[str, Any]:
    """One training run: E4's ``run_one`` with the optimizer overrides and the lambda log added,
    and nothing else changed -- the repro control depends on it. Identical initialisation and data
    order across every cell of a seed. ``freeze`` names the parameters to hold at their initial
    values; it is computed once, outside, because computing it reseeds the global generator."""
    hp = replace(bench.hparams, lam=lam, lr=lr, weight_decay=wd)
    torch.manual_seed(SEED)
    model = bench.build_model(**{k: v[0] for k, v in bench.model_choices.items()}).to(DEVICE)
    if freeze:
        wanted = set(freeze)
        for name, p in model.named_parameters():
            if name in wanted:
                p.requires_grad_(False)
    theta0 = flatten_params(model)
    if mode in ("ekfac", "tekfac"):
        overrides = {"eig_before_rescale": True, **overrides}
    optimizer = build_optimizer(mode, model, hp, **overrides)

    names = {m: n for n, m in model.named_modules()}
    log: List[Dict[str, Any]] = []
    inner_step = optimizer.step

    def step_and_log(*args, **kwargs):
        inner_step(*args, **kwargs)
        if (optimizer.steps - 1) % LOG_EVERY == 0:
            log.append({"step": optimizer.steps - 1,
                        "layers": _lambda_log(optimizer, names, mode)})

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
        "lambda_log": log,
    }


def safe_run_cell(bench, mode, **kw) -> Dict[str, Any]:
    """A singular factor at a small constant is an ordinary outcome, not a reason to lose the grid."""
    try:
        return run_cell(bench, mode, **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"    [CRASHED] {mode}: {type(exc).__name__}: {exc}", flush=True)
        return {"test_acc": float("nan"), "test_loss": float("nan"), "epoch_val_acc": [],
                "error": f"{type(exc).__name__}: {exc}"}


def plan_cells(model_name: str, mode: str, cap: float, base_lam: float, base_lr: float,
               wd: float, decoupled: bool, frozen: List[str]) -> List[Dict[str, Any]]:
    """Every cell of one (network, mode) at this seed, in the order they run."""
    # With lr = cap, a decoupled decay of rate base_lr * wd needs a coefficient wd * base_lr / cap.
    wd_fixed = wd * base_lr / cap if decoupled else wd
    held = dict(lr=cap, wd=wd_fixed, freeze=frozen)
    cells: List[Dict[str, Any]] = []
    if "reference" in ARMS:
        cells.append(dict(arm="reference", value=base_lam, lam=base_lam,
                          overrides={"hold_cap": True}, **held))
    if "single" in ARMS:
        for lam in _grid(SINGLE_GRID[(model_name, mode)]):
            cells.append(dict(arm="single", value=lam, lam=lam,
                              overrides={"hold_cap": True}, **held))
    for arm, damping in (("s1b", "layer_relative"), ("netadapt", "network_relative")):
        if arm in ARMS:
            for tau in _grid(TAU_GRID[mode]):
                cells.append(dict(arm=arm, value=tau, lam=base_lam,
                                  overrides={"hold_cap": True, "damping": damping,
                                             "damping_tau": tau}, **held))
    repro_lam = REPRO[(model_name, mode)][0]
    # Spelled exactly as e4_fixed_average.py spells it: cap * lambda can differ in the last bit.
    repro_lr = base_lr * (repro_lam / base_lam)
    if "repro" in ARMS and SEED == 0:
        # E13/E14 exactly: unfrozen, lr = cap * lambda, no hold_cap, the benchmark's own decay.
        cells.append(dict(arm="repro", value=repro_lam, lam=repro_lam, lr=repro_lr,
                          wd=wd, freeze=None, overrides={}))
    if "wdctrl" in ARMS and decoupled:
        # The repro cell with one change: the decoupled decay at its fixed rate, base_lr * wd,
        # which under lr = cap * lambda needs the coefficient wd * base_lr / lr.
        cells.append(dict(arm="wdctrl", value=repro_lam, lam=repro_lam, lr=repro_lr,
                          wd=wd * base_lr / repro_lr, freeze=None, overrides={}))
    return cells


def _final_val(cell: Dict[str, Any]) -> float:
    v = cell.get("epoch_val_acc") or []
    return v[-1] if v else float("nan")


def main() -> None:
    bench = discover_benchmarks()[MODEL]
    hp0 = bench.hparams
    base_lam, base_lr, wd, decoupled = hp0.lam, hp0.lr, hp0.weight_decay, hp0.decoupled_wd
    cap = base_lr / base_lam
    frozen = unhooked_parameter_names(bench)
    print(f"E15: one safety constant per layer | {MODEL} | seed={SEED} | batch={BATCH} | "
          f"epochs={EPOCHS} | device={DEVICE}\nmodes={MODES} arms={ARMS}\n"
          f"cap={cap:g} (base lr {base_lr:g} / base lambda {base_lam:g}); weight decay {wd:g}, "
          f"{'decoupled, applied at base_lr*wd' if decoupled else 'coupled'}; "
          f"frozen in the held arms: {', '.join(frozen) or 'nothing'}\n", flush=True)

    results: Dict[str, Any] = {"model": MODEL, "seed": SEED, "batch_size": BATCH,
                               "epochs": EPOCHS, "cap": cap, "base_lambda": base_lam,
                               "base_lr": base_lr, "weight_decay": wd,
                               "decoupled_wd": decoupled, "grid_limit": GRID_LIMIT,
                               "train_subset": TRAIN_SUBSET, "cells": {}}
    for mode in MODES:
        for cell in plan_cells(MODEL, mode, cap, base_lam, base_lr, wd, decoupled, frozen):
            arm, value = cell.pop("arm"), cell.pop("value")
            t0 = time.time()
            run = safe_run_cell(bench, mode, **cell)
            print(f"  {mode:6s} {arm:9s} {value:9.2e}  val={100*_final_val(run):6.2f}  "
                  f"test={100*run['test_acc']:6.2f}  ({time.time()-t0:.0f}s)", flush=True)
            results["cells"][f"{mode}|{arm}|{value:g}"] = {
                **run, "mode": mode, "arm": arm, "value": value, "lam": cell["lam"],
                "lr": cell["lr"], "wd": cell["wd"], "freeze": cell["freeze"],
                "overrides": cell["overrides"],
            }
            OUT.parent.mkdir(parents=True, exist_ok=True)
            with open(OUT, "w") as f:   # after every run: a wall-clock kill costs one cell
                json.dump(results, f, indent=1)

    print("\n===== per mode: best value on VALIDATION, and its test accuracy =====", flush=True)
    for mode in MODES:
        for arm in ("reference", "single", "s1b", "netadapt"):
            cells = [c for c in results["cells"].values() if c["mode"] == mode and c["arm"] == arm]
            cells = [c for c in cells if not math.isnan(_final_val(c))]
            if not cells:
                continue
            best = max(cells, key=_final_val)
            print(f"  {mode:6s} {arm:9s} best {best['value']:9.2e}  val={100*_final_val(best):6.2f}"
                  f"  test={100*best['test_acc']:6.2f}   (one seed: not a verdict)", flush=True)

    if "repro" in ARMS and SEED == 0 and GRID_LIMIT is None and TRAIN_SUBSET is None:
        print("\n===== repro control: must reproduce E13/E14's seed-0 cell to 0.00 points =====",
              flush=True)
        for mode in MODES:
            lam, expected = REPRO[(MODEL, mode)]
            got = results["cells"].get(f"{mode}|repro|{lam:g}")
            if got is None:
                continue
            diff = 100 * got["test_acc"] - expected
            print(f"  {mode:6s} lambda={lam:g}: {100*got['test_acc']:.2f} vs {expected:.2f} "
                  f"-> {'PASS' if abs(diff) < 0.005 else 'FAIL'} ({diff:+.2f})", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
