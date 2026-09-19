"""ResNet-50 with the ImageNet stem (7x7 stride-2 convolution + max-pool), 25 557 032 parameters,
on ImageNet-1K at its native 224x224, 1000 classes. Seven arms (``diag``/``kfac``/``ekfac``/
``tkfac``/``tekfac``/``adam``/``adamw``) under the wall-clock-time protocol: ``diag`` runs 30
epochs, and its elapsed time is every other arm's budget. AdaFisher's CNN operating point, not
re-tuned: ``lr = 1e-3`` everywhere, coupled weight decay ``1e-4``, ``lam = 1e-3``, batch 256, SUA
on and the Fisher statistic capped to 32 examples per batch. Stage the dataset first.

    python -m benchmarks.models.resnet50_imagenet.bench --epochs 30 --budget-mode wct \\
        --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from benchmarks.common.data import imagenet  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.models.resnet50_imagenet.model import build_resnet50_imagenet  # noqa: E402

BENCH = Benchmark(
    name="resnet50_imagenet",
    display_name="ImageNet-1K, ResNet-50",
    build_model=build_resnet50_imagenet,
    build_data=imagenet,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    hparams=HParams(
        lr=1e-3, baseline_lr=1e-3, lam=1e-3, decoupled_wd=False,
        # Appendix D's uniform 5e-4 is stated for CIFAR-10; 1e-4 is the ILSVRC value of
        # resnet_1512.03385.pdf §3.4 ("weight decay of 0.0001"), which is what this bench uses.
        weight_decay=1e-4,
        conv_sua=True,  # mandatory at this scale — plan_lot8.md §0.5
        fisher_batch_samples=32,  # bounds ekfac/tekfac's cached batch — plan_lot8.md §0.4
    ),
    epochs=30,  # not the paper's ~120: a compute-bounded choice, shared by all 7 arms
    batch_size=256,  # AdaFisherCNN.yaml's own mini_batch_size
    output_group="imagenet",
)

if __name__ == "__main__":
    main(BENCH)
