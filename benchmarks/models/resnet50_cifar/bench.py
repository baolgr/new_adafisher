"""ResNet-50 with the CIFAR stem, 23 520 842 parameters, on 32x32 CIFAR-10, 10 classes. Seven arms
(``diag``/``kfac``/``ekfac``/``tkfac``/``tekfac``/``adam``/``adamw``) under the wall-clock-time
protocol: ``diag`` runs the nominal 50 epochs and its elapsed time is every other arm's budget.
Hyperparameters are AdaFisher's own tuned CNN point, not re-tuned here.

    python -m benchmarks.models.resnet50_cifar.bench --epochs 50 --budget-mode wct \\
        --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from benchmarks.common.data import cifar10  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.models.resnet50_cifar.model import build_resnet50_cifar  # noqa: E402

# AdaFisher's own tuned operating point (Appendix D "HP Tuning" / Table 9 and the official
# repository's shipped configs), not re-tuned here — plan_lot8.md §0.8, §4.
BENCH = Benchmark(
    name="resnet50_cifar",
    display_name="CIFAR-10, ResNet-50",
    build_model=build_resnet50_cifar,
    build_data=cifar10,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    hparams=HParams(
        lr=1e-3,  # Table 9, CNNs / AdaFisher
        baseline_lr=1e-3,  # Table 9, CNNs / Adam
        weight_decay=5e-4,  # Appendix D: "uniform weight decay of 5e-4 ... for CIFAR-10"
        lam=1e-3,  # AdaFisherCNN.yaml: Lambda
        decoupled_wd=False,  # "Adam and AdaFisher were used for all CNN architectures"
        conv_sua=True,  # mandatory at this scale — plan_lot8.md §0.5
        fisher_batch_samples=32,  # bounds ekfac/tekfac's cached batch — plan_lot8.md §0.4
    ),
    epochs=50,
    batch_size=128,
    output_group="cifar10",
)

if __name__ == "__main__":
    main(BENCH)
