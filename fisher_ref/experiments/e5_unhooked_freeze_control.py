"""Is the study's one real gain a curvature effect, or just frozen GroupNorm?

The only real gain in the whole damping study: ``cnn_gn_cifar`` in ``kfac`` mode improves by 4.32
accuracy points when the damping constant is lowered from 1e-3 to 1e-8 with the step-size cap held
fixed. Every other mode on that network gains one to three points at the same place, and the other
two networks gain nothing.

**There is a second explanation for that, and it has nothing to do with curvature.** The optimizer
only preconditions four layer types; everything else takes a plain momentum step. So inside one
network the two populations move at very different speeds: a preconditioned parameter steps by the
learning rate over the damping, times its momentum, while an unpreconditioned one steps by the
learning rate times it. In the sweep the cap is held at one, which means the learning rate equals
the damping -- so as the damping falls from 1e-3 to 1e-8, the *unpreconditioned* parameters have
their step shrunk by a factor of 100 000. They stop moving.

On ``cnn_gn_cifar`` those are the 224 affine parameters of its three ``GroupNorm`` layers, 0.9 % of
the network -- but each one scales an entire channel, so their influence is not proportional to
their count. If freezing them is what buys the 4.32 points, then the one positive result in this
investigation is a step-size artifact.

The test. Three arms, five modes, everything else identical -- same seed, same initial weights,
same order of examples, batch 32, fifteen epochs::

  baseline      the default damping, unpreconditioned parameters trainable. The reference.
  frozen        the default damping, unpreconditioned parameters frozen. If freezing alone
                reproduces the gain, this lands near the 1e-8 number and the premise collapses.
  small_frozen  the 1e-8 point with the cap held, unpreconditioned parameters frozen explicitly.
                A consistency check: at 1e-8 they are already near-frozen implicitly, so this
                should reproduce the earlier number.

Reading it: if ``frozen`` minus ``baseline`` is near zero, freezing does nothing and the gain is
real. If it is near +4, the gain was never about curvature.

Which parameters count as unpreconditioned is settled by asking a real ``AdaFisherMulti`` which
ones it owns, rather than by reasoning about module types.

Environment variables::

    E5_MODELS        comma-separated benchmarks (default: cnn_gn_cifar)
    E5_MODES         comma-separated modes (default: all five)
    E5_SMALL_LAMBDA  the small damping value (default: 1e-8)
    E5_OUT           output path (default: fisher_ref/outputs/e5_unhooked_freeze.json)

The batch size, seed, epochs and device come from ``e4_fixed_average.py`` and
``warmup_sgd_baseline.py``, whose runner this script reuses.

Output: the JSON at ``E5_OUT`` plus a table on standard output.
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from adafisher_modes.optimizer import AdaFisherMulti

from fisher_ref.experiments.e4_fixed_average import collapsed, run_one

MODELS = [m.strip() for m in os.environ.get("E5_MODELS", "cnn_gn_cifar").split(",") if m.strip()]
MODES = [m.strip() for m in os.environ.get(
    "E5_MODES", "diag,kfac,ekfac,tkfac,tekfac").split(",") if m.strip()]
SMALL_LAMBDA = float(os.environ.get("E5_SMALL_LAMBDA", "1e-8"))
OUT = Path(os.environ.get("E5_OUT", str(ROOT / "fisher_ref/outputs/e5_unhooked_freeze.json")))


def unhooked_parameter_names(bench) -> List[str]:
    """The parameters that belong to no layer type the optimizer hooks, by name.

    Built by asking a real ``AdaFisherMulti`` which parameters it owns, rather than by reasoning
    about layer types -- the ownership map is what ``step()`` actually consults.
    """
    torch.manual_seed(0)
    model = bench.build_model(**{k: v[0] for k, v in bench.model_choices.items()})
    opt = AdaFisherMulti(model, lr=1e-3)
    owned = set(opt._owner)
    return [n for n, p in model.named_parameters() if id(p) not in owned]


def main() -> None:
    from benchmarks.common.runner import discover_benchmarks
    benches = discover_benchmarks()
    print(f"E5 control: does freezing the unpreconditioned parameters reproduce E4's gain?\n"
          f"models={MODELS} modes={MODES} small lambda={SMALL_LAMBDA:g}\n", flush=True)

    results: Dict[str, Any] = {}
    for model_name in MODELS:
        bench = benches[model_name]
        base_lam, base_lr = bench.hparams.lam, bench.hparams.lr
        frozen_names = unhooked_parameter_names(bench)
        torch.manual_seed(0)
        probe = bench.build_model(**{k: v[0] for k, v in bench.model_choices.items()})
        n_frozen = sum(p.numel() for n, p in probe.named_parameters() if n in set(frozen_names))
        n_total = sum(p.numel() for p in probe.parameters())
        print(f"########## {model_name}  (default lambda={base_lam:g}, lr={base_lr:g}) ##########",
              flush=True)
        print(f"  unpreconditioned: {len(frozen_names)} tensors, {n_frozen} of {n_total} "
              f"parameters ({100*n_frozen/n_total:.1f}%): {', '.join(frozen_names)}\n", flush=True)
        results[model_name] = {"base_lambda": base_lam, "base_lr": base_lr,
                               "frozen_names": frozen_names, "n_frozen": n_frozen,
                               "n_total": n_total, "cells": {}}

        arms = [("baseline", base_lam, False),
                ("frozen", base_lam, True),
                ("small_frozen", SMALL_LAMBDA, True)]
        for mode in MODES:
            for arm, lam, freeze in arms:
                lr = base_lr * (lam / base_lam)      # the cap lr/lambda never moves
                t0 = time.time()
                try:
                    run = run_one(bench, mode, "shipped", lam, lr,
                                  freeze_names=frozen_names if freeze else None)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [CRASHED] {mode}/{arm}: {type(exc).__name__}: {exc}", flush=True)
                    run = {"test_loss": float("nan"), "test_acc": float("nan"), "n_steps": 0,
                           "epoch_val_acc": [], "epoch_val_loss": [], "epoch_train_loss": [],
                           "wall_s": float("nan"), "theta_move_from_init": float("nan"),
                           "error": f"{type(exc).__name__}: {exc}"}
                bad = collapsed(run)
                print(f"  {mode:7s} {arm:13s} lambda={lam:9.1e} lr={lr:9.1e}  "
                      f"test_loss={run['test_loss']:8.4f} test_acc={run['test_acc']:7.4f}"
                      f"{'  [COLLAPSED]' if bad else ''}  ({time.time()-t0:.0f}s)", flush=True)
                results[model_name]["cells"][f"{mode}|{arm}"] = {
                    **run, "mode": mode, "arm": arm, "lam": lam, "lr": lr, "collapsed": bad}
            OUT.parent.mkdir(parents=True, exist_ok=True)
            with open(OUT, "w") as f:
                json.dump({"small_lambda": SMALL_LAMBDA, "results": results}, f, indent=1)

    print("\n===== VERDICT =====", flush=True)
    print("If 'frozen - baseline' is near zero, freezing does nothing and E4's gain is real.",
          flush=True)
    print("If it is near the gain E4 measured, that gain was never about curvature.\n", flush=True)
    for model_name, blob in results.items():
        print(f"-- {model_name}", flush=True)
        print(f"  {'mode':7s} {'baseline':>9s} {'frozen':>9s} {'diff':>8s} {'small_frozen':>13s}",
              flush=True)
        for mode in MODES:
            c = blob["cells"]
            b = c.get(f"{mode}|baseline", {}).get("test_acc", float("nan"))
            f_ = c.get(f"{mode}|frozen", {}).get("test_acc", float("nan"))
            s = c.get(f"{mode}|small_frozen", {}).get("test_acc", float("nan"))
            print(f"  {mode:7s} {100*b:9.2f} {100*f_:9.2f} {100*(f_-b):+8.2f} {100*s:13.2f}",
                  flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
