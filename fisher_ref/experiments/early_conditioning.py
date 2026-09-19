"""Why does a plain-momentum stand-in reproduce some networks and modes but not others?

A damped Kronecker factor can be badly conditioned -- a near-zero eigenvalue swamped by one huge
one -- which no scalar schedule can reproduce and which amplifies noise in the corresponding
direction. That mechanism had only been observed as an outright *crash*, once the damping constant
was pushed far below its default. At each network's own default the per-layer dynamic range is flat
everywhere (about 1.0 to 1.1) at a half-trajectory checkpoint -- but that is read off an
already-warm network. Early in a *fresh* run the dynamic range of the raw, undamped spectrum grows
by three to four orders of magnitude over the first few hundred steps, even while curvature's
overall magnitude fades towards the damping constant. What had never been looked at is the *damped*
spectrum, which is the matrix actually inverted and applied.

This script closes that gap: for three networks and all five modes, a real fresh
``AdaFisherMulti`` run, snapshotting both the undamped and the **damped** spectrum per hooked layer
at the first five factor updates (absolute steps 0, 100, 200, 300, 400).

Reading it: if one mode shows a damped-spectrum dynamic range on a layer that another mode does not
show on the same layer of the same network, that is the mechanism. If a network never shows a
comparably extreme value on any single layer in any mode, that is consistent with its gap coming
from many small, evenly spread imperfections rather than one bad layer.

Environment variables::

    EARLY_COND_MODELS  comma-separated run directories
                       (default: cnn_gn_cifar,cct_2_3x2_cifar,resnet20_cifar)
    EARLY_COND_MODES   comma-separated modes (default: diag,kfac,tkfac,ekfac,tekfac)
    DATA_ROOT          dataset root (default: benchmarks/data)

Output: a table on standard output. Nothing is written to disk.
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

MODELS = os.environ.get("EARLY_COND_MODELS",
                        "cnn_gn_cifar,cct_2_3x2_cifar,resnet20_cifar").split(",")
MODES = os.environ.get("EARLY_COND_MODES", "diag,kfac,tkfac,ekfac,tekfac").split(",")
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


def dyn_range(spectrum: torch.Tensor) -> float:
    """p99/p01 -- 1 means flat (isotropic: a scalar schedule can match it, however big);
    >>1 means anisotropic (some directions genuinely stand out from others).
    """
    q = torch.quantile(spectrum, torch.tensor([0.01, 0.99], dtype=spectrum.dtype))
    return (q[1] / q[0]).item() if q[0] > 0 else float("inf")


def main() -> None:
    for model_name in MODELS:
        bench = BENCHES[model_name]
        hp = bench.hparams

        for mode in MODES:
            torch.manual_seed(SEED)
            model_kwargs = {k: v[0] for k, v in bench.model_choices.items()}
            model = bench.build_model(**model_kwargs).to(DEV)
            optimizer = build_optimizer(mode, model, hp)

            train_loader, _, _ = bench.build_data(
                DATA_ROOT, batch_size=bench.batch_size, seed=SEED, num_workers=2,
                cutout=True, allow_download=False, train_subset=None,
            )
            names = {id(m): n for n, m in model.named_modules()}
            print(f"\n===== {model_name} / {mode} (lambda={hp.lam}, device={DEV}) =====",
                  flush=True)

            def on_step(completed_steps: int, epoch: int, model_) -> None:
                if completed_steps not in SNAPSHOT_STEPS:
                    return
                k = SNAPSHOT_STEPS[completed_steps]
                worst_dmp = None  # by damped dyn_range -- the applied, actually-used matrix
                worst_und = None  # by undamped dyn_range -- the raw curvature, for comparison
                for m in model_.modules():
                    if type(m).__name__ not in HOOKED_TYPES:
                        continue
                    try:
                        und, dmp = spectra(optimizer.approx, mode, m)
                    except (KeyError, RuntimeError):
                        continue
                    dr_dmp = dyn_range(dmp)
                    dr_und = dyn_range(und)
                    layer = names.get(id(m), "?")
                    if worst_dmp is None or dr_dmp > worst_dmp[1]:
                        worst_dmp = (layer, dr_dmp)
                    if worst_und is None or dr_und > worst_und[1]:
                        worst_und = (layer, dr_und)
                print(f"  k={k} (step {completed_steps - 1:4d}): "
                      f"worst APPLIED dyn_range: {worst_dmp[0]:26s} {worst_dmp[1]:10.3g}  |  "
                      f"worst undamped dyn_range: {worst_und[0]:26s} {worst_und[1]:10.3g}",
                      flush=True)

            train_under_budget(
                model, optimizer, train_loader, bench.loss_fn,
                budget_s=float("inf"), max_epochs=10**9, device=DEV,
                on_step=on_step, max_steps=max(SNAPSHOT_STEPS) + 1,
                prepare_batch=bench.prepare_batch, log_fn=lambda *_a, **_k: None,
            )


if __name__ == "__main__":
    main()
