"""ImageNet-1K @ 224 px / ResNet-50, the paper's own configuration (25 557 032 parameters).

The only bench of this repository running an architecture at the resolution it was designed for:
Table 1's 7x7/stride-2 + max-pool stem, ``RandomResizedCrop(224)``, 1000 classes. Everything about
it is expensive — see ``benchmarks/slurm/imagenet/README.md`` for the measured-per-arm budget this
implies under the WCT protocol.

    python -m benchmarks.resnet50_imagenet.bench --arms diag --epochs 30 --budget-mode epochs
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.common.data import imagenet  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.resnet50_imagenet.model import build_resnet50_imagenet  # noqa: E402

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
