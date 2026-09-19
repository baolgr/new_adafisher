"""Is curvature negligible next to the damping constant in the FIRST few factor updates of a fresh run?

Every scalar-schedule stand-in for a Fisher mode assumes so, and no measurement in this project had
checked it. Every earlier curvature-versus-damping measurement was taken mid-training, at a
half-trajectory checkpoint, after a thousand-step re-warm -- an already-warm network. None of it
says anything about the first few hundred steps of a network at its random initialisation, which is
exactly the window a stand-in retrains from scratch, and where a mathematically corrected schedule
was found to help two networks and hurt a third. A scalar formula alone cannot explain that
difference if curvature really is negligible everywhere, on every network, the whole time.

This script trains a real, fresh ``AdaFisherMulti`` in ``kfac`` mode (no checkpoint) on three
networks and snapshots the undamped curvature spectrum, per hooked layer, immediately after the
first five factor updates -- absolute steps 0, 100, 200, 300 and 400, exactly the window where a
start-up schedule changes the divisor the most. It reports, per snapshot, the layer with the worst
maximum-over-damping ratio and the layer with the worst dynamic range (the 99th over the 1st
percentile of the undamped spectrum), since those are two different failure modes: a large ratio
means curvature is not negligible, a large dynamic range means no scalar can reproduce it however
well its magnitude is calibrated.

Configuration: module-level constants (``MODELS`` = ``mlp_ln_mnist``, ``cnn_gn_cifar``,
``resnet20_cifar``; ``SEED = 0``; ``TCOV = 100``; five updates).

Environment variables: ``DATA_ROOT`` (default ``benchmarks/data``).

Output: a table on standard output. Nothing is written to disk.

It reuses the spectrum helper from ``lambda_vs_curvature`` rather than duplicating it, imported by
its full package path so that both ``python -m fisher_ref.experiments.early_curvature`` and running
this file by path work. The repository root is put on the import path above, which is what makes
the package path resolvable in the second form.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from fisher_ref.experiments.lambda_vs_curvature import spectra  # reused, not duplicated

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
