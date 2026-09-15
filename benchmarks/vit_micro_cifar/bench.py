"""A3 — ViT-micro (``d=32``, 2 blocks, 2 heads, ``MLP=2D``, mean pooling), CIFAR-10
(``plan_exp_draft.md`` §4). 21 098 parameters.

The architecture is ``benchmarks/vit_small_cifar/model.py``'s ``ViTCIFAR``, configured — D1's one
sanctioned cross-folder import, so no ViT is ever duplicated. ``plan_exp_step1.md`` §4 estimated
``≈ 21 162``; the exact value under this configuration is **21 098**, the difference being exactly
the ``cls_token`` (32) plus its position row (32) that mean pooling removes.

    python -m benchmarks.vit_micro_cifar.bench --epochs 30 --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.common.data import cifar10  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.vit_small_cifar.model import ViTCIFAR  # noqa: E402

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
