"""Is curvature negligible next to Lambda in the FIRST few TCov cycles of a FRESH ``kfac`` run --
the assumption every scalar-schedule stand-in (Steps 9, 11, 12 of
``docs/reports/validation_noise_investigation.md``) makes, but that no measurement in this project
has actually checked?

Every earlier curvature-vs-Lambda measurement (Steps 7, 8, 10) was taken mid-training, at
``ckpt_0.5``, after a 1000-step re-warm -- an ALREADY-WARM network. None of it says anything about
steps 0-300 of a network at its random initialization, which is exactly the window Steps 9/11/12
retrain from scratch and compare against a scalar stand-in, and where Step 12 found the
mathematically-corrected schedule helping two networks (``mlp_ln_mnist``, ``resnet20_cifar``) and
hurting a third (``cnn_gn_cifar``) -- a difference a scalar formula alone cannot explain if
curvature really is negligible everywhere, on every network, the whole time.

This script trains a real, fresh ``AdaFisherMulti(fisher_mode="kfac")`` (no checkpoint) on the
same three networks and the same seed Steps 9/11/12 use, and snapshots the undamped curvature
spectrum, per hooked layer, immediately after the 1st, 2nd, 3rd, 4th and 5th EMA update (steps
1, 101, 201, 301, 401 -- ``k = 1..5`` in ``WarmupMomentumSGD``'s own schedule, ``docs`` handoff),
exactly the window where the schedule correction changes the divisor the most.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from lambda_vs_curvature import spectra  # reused, not duplicated

from benchmarks.common.loop import train_under_budget
from benchmarks.common.optimizers import build_optimizer
from benchmarks.common.runner import discover_benchmarks

MODELS = ["mlp_ln_mnist", "cnn_gn_cifar", "resnet20_cifar"]
SEED = 0
TCOV = 100
K_MAX = 5
# completed_steps right after the k-th EMA update: the 1st happens at absolute step 0 (the very
# first optimizer.step() call, i.e. completed_steps == 1), the 2nd at absolute step TCOV, etc.
SNAPSHOT_STEPS = {(k - 1) * TCOV + 1: k for k in range(1, K_MAX + 1)}
DATA_ROOT = os.environ.get("DATA_ROOT", str(ROOT / "benchmarks" / "data"))
HOOKED_TYPES = ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm")

BENCHES = discover_benchmarks()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dyn_range(und: torch.Tensor) -> float:
    """p99/p01 of the undamped spectrum -- 1 means flat (isotropic: a scalar schedule can match
    it, however big); >>1 means anisotropic (some directions genuinely stand out from others,
    which no scalar can reproduce, however well its overall *magnitude* is calibrated).
    """
    q = torch.quantile(und, torch.tensor([0.01, 0.99], dtype=und.dtype))
    return (q[1] / q[0]).item() if q[0] > 0 else float("inf")


def main() -> None:
    for model_name in MODELS:
        bench = BENCHES[model_name]
        hp = bench.hparams
        torch.manual_seed(SEED)
        model_kwargs = {k: v[0] for k, v in bench.model_choices.items()}
        model = bench.build_model(**model_kwargs).to(DEV)
        optimizer = build_optimizer("kfac", model, hp)

        train_loader, _, _ = bench.build_data(
            DATA_ROOT, batch_size=bench.batch_size, seed=SEED, num_workers=2,
            cutout=True, allow_download=False, train_subset=None,
        )
        names = {id(m): n for n, m in model.named_modules()}
        print(f"\n===== {model_name} (lambda={hp.lam}, device={DEV}) =====", flush=True)

        def on_step(completed_steps: int, epoch: int, model_) -> None:
            if completed_steps not in SNAPSHOT_STEPS:
                return
            k = SNAPSHOT_STEPS[completed_steps]
            worst = None  # by max curvature / lambda
            worst_aniso = None  # by dyn_range, independently
            for m in model_.modules():
                if type(m).__name__ not in HOOKED_TYPES:
                    continue
                try:
                    und, _ = spectra(optimizer.approx, "kfac", m)
                except (KeyError, RuntimeError):
                    continue
                ratio = (und.max() / hp.lam).item()
                dr = dyn_range(und)
                if worst is None or ratio > worst[1]:
                    worst = (names.get(id(m), "?"), ratio, dr)
                if worst_aniso is None or dr > worst_aniso[1]:
                    worst_aniso = (names.get(id(m), "?"), dr, ratio)
            print(f"  k={k} (absolute step {completed_steps - 1:4d}): "
                  f"worst max/lambda: {worst[0]:26s} {worst[1]:10.3g}x  "
                  f"(that layer's own dyn_range={worst[2]:8.3g})  |  "
                  f"worst dyn_range: {worst_aniso[0]:26s} {worst_aniso[1]:10.3g}  "
                  f"(that layer's own max/lambda={worst_aniso[2]:8.3g}x)", flush=True)

        train_under_budget(
            model, optimizer, train_loader, bench.loss_fn,
            budget_s=float("inf"), max_epochs=10**9, device=DEV,
            on_step=on_step, max_steps=max(SNAPSHOT_STEPS) + 1,
            prepare_batch=bench.prepare_batch, log_fn=lambda *_a, **_k: None,
        )


if __name__ == "__main__":
    main()
