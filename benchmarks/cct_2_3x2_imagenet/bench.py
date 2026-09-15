"""B1 on ImageNet-1K @ 32x32 — CCT-2/3x2, 1000-way head (411 433 params). ImageNet32 (Chrabaszcz
et al. 2017 §2); see ``cnn_gn_imagenet/bench.py``'s docstring.

The downsampling is not a convenience here but a necessity: at 224 px this tokenizer emits a
56x56 = 3136-token sequence, whose attention map alone is ~10 TB at batch 256.

    python -m benchmarks.cct_2_3x2_imagenet.bench --epochs 40 --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.cct_2_3x2_cifar.model import build_cct_2_3x2_cifar  # noqa: E402
from benchmarks.common.data import imagenet32  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402

BENCH = Benchmark(
    name="cct_2_3x2_imagenet",
    display_name="ImageNet-1K @32px, CCT-2/3x2",
    build_model=partial(build_cct_2_3x2_cifar, num_classes=1000),
    build_data=imagenet32,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # AdaFisher's ViT operating point (Table 9, ViTs; AdaFisherViT.yaml): AdaFisherW / AdamW.
    hparams=HParams(lr=1e-3, baseline_lr=1e-4, weight_decay=1e-2, lam=3e-3, decoupled_wd=True),
    epochs=40,
    batch_size=256,
    output_group="imagenet",
)

if __name__ == "__main__":
    main(BENCH)
