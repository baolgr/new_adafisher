"""CIFAR-10 / ViT-S/4 bench (lot 8), on the shared harness.

    python -m benchmarks.vit_small_cifar.bench --arms diag --epochs 50 --budget-mode epochs
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.common.data import cifar10  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.vit_small_cifar.model import build_vit_small_cifar  # noqa: E402

BENCH = Benchmark(
    name="vit_small_cifar",
    display_name="CIFAR-10, ViT-S/4",
    build_model=build_vit_small_cifar,
    build_data=cifar10,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    hparams=HParams(
        lr=1e-3,  # Table 9, ViTs / AdaFisherW
        baseline_lr=1e-4,  # Table 9, ViTs / AdamW ("adopted from the original publications")
        weight_decay=1e-2,  # AdaFisherViT.yaml: weight_decay
        lam=3e-3,  # AdaFisherViT.yaml: Lambda
        decoupled_wd=True,  # "AdamW and AdaFisherW were applied for all ViT experiments"
        conv_sua=False,  # one 4x4 patch-embedding conv; SUA would change nothing
    ),
    epochs=50,
    batch_size=128,
)

if __name__ == "__main__":
    main(BENCH)
