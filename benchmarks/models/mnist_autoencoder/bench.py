"""The 8-layer MNIST auto-encoder (``784-1000-500-250-30`` and its untied mirror), 2 837 314
parameters, on 28x28 greyscale MNIST. Mean-squared reconstruction error, so no accuracy column.

Seven arms (``diag``/``kfac``/``ekfac``/``tkfac``/``tekfac``/``adam``/``adamw``) under the
wall-clock-time protocol: ``diag`` runs the nominal 20 epochs and its measured elapsed time is
every other arm's budget.

Shared hyperparameters, untuned: ``lr = 1e-3`` for every arm, no weight decay, AdaFisher damping
``lam = 1e-3``, batch 500, 20 nominal epochs, cosine annealing.

    python -m benchmarks.models.mnist_autoencoder.bench --epochs 20 --budget-mode wct \\
        --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1

The original AdaFisher of ``reference_repos/FisherAdapTune`` is available as an eighth arm: with
``--arms reference diag --no-minmax`` the two curves should superimpose, which is this project's
non-regression demo of its own port.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from benchmarks.common.data import mnist  # noqa: E402
from benchmarks.common.loop import flatten_autoencoder_batch  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.models.mnist_autoencoder.model import build_mnist_autoencoder  # noqa: E402

BENCH = Benchmark(
    name="mnist_autoencoder",
    display_name="MNIST auto-encoder 784-1000-500-250-30",
    build_model=build_mnist_autoencoder,
    build_data=mnist,
    loss_fn=nn.MSELoss(),
    prepare_batch=flatten_autoencoder_batch,  # reconstruct the flattened input
    metric_fn=None,  # reconstruction: no accuracy
    hparams=HParams(),  # CLAUDE.md's "Selecting a mode" defaults, untuned (plan_lot7.md §6)
    epochs=20,
    batch_size=500,  # lots 1 and 7's batch size
    output_group="mnist",
)

if __name__ == "__main__":
    main(BENCH)
