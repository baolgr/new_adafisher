"""A3 on ImageNet-1K @ 32x32 — the ``vit_micro_cifar`` configuration, 1000-way head (53 768
params). ImageNet32 (Chrabaszcz et al. 2017 §2); see ``cnn_gn_imagenet/bench.py``'s docstring.

    python -m benchmarks.vit_micro_imagenet.bench --epochs 40 --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.common.data import imagenet32  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.vit_small_cifar.model import ViTCIFAR  # noqa: E402

build_vit_micro_imagenet = partial(
    ViTCIFAR, patch_size=4, embed_dim=32, depth=2, num_heads=2, mlp_ratio=2.0, pool="mean",
    drop=0.0, attn_drop=0.0, num_classes=1000,
)

BENCH = Benchmark(
    name="vit_micro_imagenet",
    display_name="ImageNet-1K @32px, ViT-micro d=32",
    build_model=build_vit_micro_imagenet,
    build_data=imagenet32,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # AdaFisher's ViT operating point (Table 9, ViTs; AdaFisherViT.yaml), decoupled decay.
    hparams=HParams(lr=1e-3, baseline_lr=1e-4, weight_decay=1e-2, lam=3e-3, decoupled_wd=True),
    epochs=40,
    batch_size=256,
    output_group="imagenet",
)

if __name__ == "__main__":
    main(BENCH)
