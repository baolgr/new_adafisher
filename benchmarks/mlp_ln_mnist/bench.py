"""A1 — MLP 784-32-32-10 + LayerNorm, MNIST (``plan_exp_draft.md`` §4).

    python -m benchmarks.mlp_ln_mnist.bench --epochs 20 --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.common.data import mnist  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.mlp_ln_mnist.model import build_mlp_ln_mnist  # noqa: E402

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
)

if __name__ == "__main__":
    main(BENCH)
