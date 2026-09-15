"""A2 on ImageNet-1K @ 32x32 — the ``cnn_gn_cifar`` architecture, 1000-way head (88 808 params).

The network is byte-for-byte the CIFAR one apart from the head width: the data is downsampled to
32x32 instead (``common/data.py``'s ``imagenet32``, the construction of Chrabaszcz et al. 2017
§2), which is what makes a 32x32-native architecture applicable to ImageNet-1K at all. Read any
result from this bench as "ImageNet32", never as ImageNet-1K at its native resolution.

    python -m benchmarks.cnn_gn_imagenet.bench --epochs 40 --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.cnn_gn_cifar.model import build_cnn_gn_cifar  # noqa: E402
from benchmarks.common.data import imagenet32  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402

BENCH = Benchmark(
    name="cnn_gn_imagenet",
    display_name="ImageNet-1K @32px, CNN 16-32-64 + GroupNorm",
    build_model=partial(build_cnn_gn_cifar, num_classes=1000),
    model_choices={"norm": ("gn", "bn")},
    build_data=imagenet32,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # AdaFisher's CNN operating point (Table 9, CNNs), not re-tuned for ImageNet. The batch size
    # is AdaFisherCNN.yaml's own ``mini_batch_size: 256``, not the CIFAR benches' 128.
    hparams=HParams(lr=1e-3, baseline_lr=1e-3, weight_decay=5e-4, lam=1e-3, decoupled_wd=False),
    epochs=40,
    batch_size=256,
    output_group="imagenet",
)

if __name__ == "__main__":
    main(BENCH)
