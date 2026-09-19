"""ViT-micro (``D=32``, 2 blocks, 2 heads, ``MLP = 2D``, mean pooling), 21 098 parameters, on
32x32 CIFAR-10, 10 classes. A configuration of ``vit_small_cifar/model.py``'s ``ViTCIFAR``.
Seven arms (``diag``/``kfac``/``ekfac``/``tkfac``/``tekfac``/``adam``/``adamw``) under the
wall-clock-time protocol: ``diag`` runs 30 epochs, and its elapsed time is every other arm's
budget. AdaFisher's ViT operating point (Table 9), not re-tuned: ``lr = 1e-3`` for the Fisher
arms, ``1e-4`` for the baselines, decoupled weight decay ``1e-2``, ``lam = 3e-3``, batch 128.

    python -m benchmarks.models.vit_micro_cifar.bench --epochs 30 --budget-mode wct \\
        --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from benchmarks.common.data import cifar10  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.models.vit_small_cifar.model import ViTCIFAR  # noqa: E402

build_vit_micro_cifar = partial(
    ViTCIFAR, patch_size=4, embed_dim=32, depth=2, num_heads=2, mlp_ratio=2.0, pool="mean",
    drop=0.0, attn_drop=0.0,
)

BENCH = Benchmark(
    name="vit_micro_cifar",
    display_name="CIFAR-10, ViT-micro d=32",
    build_model=build_vit_micro_cifar,
    build_data=cifar10,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # AdaFisher's ViT operating point (Table 9, ViTs; AdaFisherViT.yaml), decoupled decay.
    hparams=HParams(lr=1e-3, baseline_lr=1e-4, weight_decay=1e-2, lam=3e-3, decoupled_wd=True),
    epochs=30,
    batch_size=128,
    output_group="cifar10",
)

if __name__ == "__main__":
    main(BENCH)
