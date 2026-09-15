"""ImageNet-1K @ 224 px / ViT-S/16 (22 050 664 parameters).

``ViTCIFAR`` is generic in ``img_size``/``patch_size``/``embed_dim``; configured here as ViT-S/16
(``D=384``, ``depth=12``, ``heads=6`` so ``D/heads = 64``, ``MLP = 4D``) at ``vit_2010.11929.pdf``
Table 1's own 224 px / patch-16 resolution. "ViT-S" is still not a row of that table (Base/Large/
Huge only), so the configuration is an adaptation, as ``vit_small_cifar``'s is — never attribute
it to the paper.

    python -m benchmarks.vit_small_imagenet.bench --arms diag --epochs 30 --budget-mode epochs
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.common.data import imagenet  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.vit_small_cifar.model import ViTCIFAR  # noqa: E402

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
    # conv_sua: one 16x16 patch-embedding conv, so SUA would change nothing.
    # fisher_batch_samples: 197 tokens x 48 Linears is 3.7 GB of cached h_bar otherwise.
    hparams=HParams(lr=1e-3, baseline_lr=1e-4, weight_decay=1e-2, lam=3e-3, decoupled_wd=True,
                    conv_sua=False, fisher_batch_samples=32),
    epochs=30,  # compute-bounded, shared by all 7 arms
    batch_size=256,  # AdaFisherViT.yaml's own mini_batch_size
    output_group="imagenet",
)

if __name__ == "__main__":
    main(BENCH)
