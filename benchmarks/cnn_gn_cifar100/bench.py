"""A2 on CIFAR-100 — the ``cnn_gn_cifar`` architecture, 100-way head (30 308 parameters).

Same 32x32 3-channel images, same 45k/5k split, same augmentation as CIFAR-10: the only
differences are the normalisation constants (``common/data.py``'s ``CIFAR100_SPEC``) and the label
cardinality, so this bench is directly comparable with ``cnn_gn_cifar`` arm by arm.

    python -m benchmarks.cnn_gn_cifar100.bench --epochs 30 --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.cnn_gn_cifar.model import build_cnn_gn_cifar  # noqa: E402
from benchmarks.common.data import cifar100  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402

BENCH = Benchmark(
    name="cnn_gn_cifar100",
    display_name="CIFAR-100, CNN 16-32-64 + GroupNorm",
    build_model=partial(build_cnn_gn_cifar, num_classes=100),
    model_choices={"norm": ("gn", "bn")},
    build_data=cifar100,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # AdaFisher's CNN operating point (Table 9, CNNs; AdaFisherCNN.yaml's 5e-4 decay, which its
    # own config file applies to CIFAR-100 as well as CIFAR-10). Not re-tuned for 100 classes.
    hparams=HParams(lr=1e-3, baseline_lr=1e-3, weight_decay=5e-4, lam=1e-3, decoupled_wd=False),
    epochs=30,
    batch_size=128,
    output_group="cifar100",
)

if __name__ == "__main__":
    main(BENCH)
