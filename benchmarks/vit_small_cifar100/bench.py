"""CIFAR-100 / ViT-S/4 — the ``vit_small_cifar`` configuration, 100-way head (2 710 948
parameters). An adaptation, not a paper variant: see ``vit_small_cifar/model.py``.

    python -m benchmarks.vit_small_cifar100.bench --arms diag --epochs 50 --budget-mode epochs
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
from benchmarks.vit_small_cifar.model import build_vit_small_cifar  # noqa: E402

BENCH = Benchmark(
    name="vit_small_cifar100",
    display_name="CIFAR-100, ViT-S/4",
    build_model=partial(build_vit_small_cifar, num_classes=100),
    build_data=cifar100,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    hparams=HParams(
        lr=1e-3, baseline_lr=1e-4, weight_decay=1e-2, lam=3e-3, decoupled_wd=True,
        conv_sua=False,  # one 4x4 patch-embedding conv; SUA would change nothing
    ),
    epochs=50,
    batch_size=128,
    output_group="cifar100",
)

if __name__ == "__main__":
    main(BENCH)
