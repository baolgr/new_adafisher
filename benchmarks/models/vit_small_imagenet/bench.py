"""ViT-S/16 at 224x224, 22 050 664 parameters, on ImageNet-1K, 1000 classes.
Seven arms (``diag``/``kfac``/``ekfac``/``tkfac``/``tekfac``/``adam``/``adamw``) under the
wall-clock-time protocol: ``diag`` runs 30 epochs, and its elapsed time is every other arm's
budget. AdaFisher's ViT operating point, not re-tuned: ``lr = 1e-3`` for the Fisher arms,
``1e-4`` for the baselines, decoupled weight decay ``1e-2``, ``lam = 3e-3``, batch 256. Stage the
dataset first: see ``benchmarks/slurm/imagenet/README.md``.

    python -m benchmarks.models.vit_small_imagenet.bench --epochs 30 --budget-mode wct \\
        --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from benchmarks.common.data import imagenet  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.models.vit_small_cifar.model import ViTCIFAR  # noqa: E402

build_vit_small_imagenet = partial(
    ViTCIFAR, img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
    num_classes=1000,
)

BENCH = Benchmark(
    name="vit_small_imagenet",
    display_name="ImageNet-1K, ViT-S/16",
    build_model=build_vit_small_imagenet,
    build_data=imagenet,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # conv_sua: one 16x16 patch conv, so SUA changes nothing. fisher_batch_samples: 197 tokens
    # x 48 Linears is 3.7 GB of cached layer inputs otherwise.
    hparams=HParams(lr=1e-3, baseline_lr=1e-4, weight_decay=1e-2, lam=3e-3, decoupled_wd=True,
                    conv_sua=False, fisher_batch_samples=32),
    epochs=30,  # compute-bounded, shared by all 7 arms
    batch_size=256,  # AdaFisherViT.yaml's own mini_batch_size
    output_group="imagenet",
)

if __name__ == "__main__":
    main(BENCH)
