"""MLP ``784-32-32-10`` with LayerNorm, 26 634 parameters, on 28x28 greyscale MNIST, 10 classes.

Seven arms (``diag``/``kfac``/``ekfac``/``tkfac``/``tekfac``/``adam``/``adamw``) under the
wall-clock-time protocol: ``diag`` runs the nominal 20 epochs and its measured elapsed time is
every other arm's budget.

Shared hyperparameters, untuned: ``lr = 1e-3`` for every arm, no weight decay, ``lam = 1e-3``,
batch 128, 20 nominal epochs, cosine annealing. This model exists to be measured against an exact
Fisher matrix, not to set an MNIST record.

    python -m benchmarks.models.mlp_ln_mnist.bench --epochs 20 --budget-mode wct \\
        --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from benchmarks.common.data import mnist  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.models.mlp_ln_mnist.model import build_mlp_ln_mnist  # noqa: E402

BENCH = Benchmark(
    name="mlp_ln_mnist",
    display_name="MNIST, MLP 784-32-32-10 + LayerNorm",
    build_model=build_mlp_ln_mnist,
    build_data=mnist,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # CLAUDE.md's "Selecting a mode" defaults, untuned: this model exists to be *measured against
    # an exact Fisher*, not to set an MNIST record (plan_exp_draft.md §4).
    hparams=HParams(lr=1e-3, baseline_lr=1e-3, weight_decay=0.0, lam=1e-3),
    epochs=20,
    batch_size=128,
    output_group="mnist",
)

if __name__ == "__main__":
    main(BENCH)
