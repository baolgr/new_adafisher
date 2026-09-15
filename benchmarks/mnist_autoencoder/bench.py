"""MNIST auto-encoder bench (lots 1 and 7), on the shared harness.

Lot 1's non-regression demo is ``--arms reference diag``: with ``--no-minmax`` the two training
curves should be visually superimposed, a direct consequence of the bit-exact ``f_tilde``
construction ``tests/test_diag_bitexact.py`` already checks. Lot 7's equal-wall-clock-budget
comparison across the five modes is the default ``--budget-mode wct``.

    python -m benchmarks.mnist_autoencoder.bench --arms reference diag --epochs 5 --no-minmax
    python -m benchmarks.mnist_autoencoder.bench --budget-mode wct --epochs 20
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.common.data import mnist  # noqa: E402
from benchmarks.common.loop import flatten_autoencoder_batch  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.mnist_autoencoder.model import build_mnist_autoencoder  # noqa: E402

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
