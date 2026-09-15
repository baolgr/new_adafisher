"""CIFAR-100 / ResNet-50 (CIFAR stem) — the ``resnet50_cifar`` architecture, 100-way head
(23 705 252 parameters).

    python -m benchmarks.resnet50_cifar100.bench --arms diag --epochs 50 --budget-mode epochs
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.common.data import cifar100  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.resnet50_cifar.model import build_resnet50_cifar  # noqa: E402

BENCH = Benchmark(
    name="resnet50_cifar100",
    display_name="CIFAR-100, ResNet-50",
    build_model=partial(build_resnet50_cifar, num_classes=100),
    build_data=cifar100,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    hparams=HParams(
        lr=1e-3, baseline_lr=1e-3, weight_decay=5e-4, lam=1e-3, decoupled_wd=False,
        conv_sua=True,  # mandatory at this scale — plan_lot8.md §0.5
        fisher_batch_samples=32,  # bounds ekfac/tekfac's cached batch — plan_lot8.md §0.4
    ),
    epochs=50,
    batch_size=128,
    output_group="cifar100",
)

if __name__ == "__main__":
    main(BENCH)
