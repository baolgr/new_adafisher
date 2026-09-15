"""B2 on CIFAR-100 — the ``resnet20_cifar`` architecture, 100-way head (275 572 parameters).

    python -m benchmarks.resnet20_cifar100.bench --epochs 50 --checkpoints 0,0.01,0.1,0.5,1
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
from benchmarks.resnet20_cifar.model import build_resnet20_cifar  # noqa: E402

BENCH = Benchmark(
    name="resnet20_cifar100",
    display_name="CIFAR-100, ResNet-20",
    build_model=partial(build_resnet20_cifar, num_classes=100),
    build_data=cifar100,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # AdaFisher's CNN operating point (Table 9, CNNs; AdaFisherCNN.yaml's 5e-4 decay).
    # conv_sua stays off: the widest input factor here is 64*3*3+1 = 577, not ResNet-50's 4609.
    hparams=HParams(lr=1e-3, baseline_lr=1e-3, weight_decay=5e-4, lam=1e-3, conv_sua=False),
    epochs=50,
    batch_size=128,
    output_group="cifar100",
)

if __name__ == "__main__":
    main(BENCH)
