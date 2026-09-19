"""ResNet-20 with option-A shortcuts, 269 722 parameters, on 32x32 CIFAR-10, 10 classes.

Seven arms (``diag``/``kfac``/``ekfac``/``tkfac``/``tekfac``/``adam``/``adamw``) under the
wall-clock-time protocol: ``diag`` runs the nominal 50 epochs and its measured elapsed time is
every other arm's budget.

AdaFisher's CNN operating point (Table 9, and Appendix D's uniform CIFAR-10 decay): every arm
takes ``lr = 1e-3``, coupled weight decay ``5e-4``, ``lam = 1e-3``, batch 128, 50 epochs, cosine.
The SUA convolution approximation stays off -- the widest input factor here is ``64*3*3+1 = 577``,
not ResNet-50's 4609.

    python -m benchmarks.models.resnet20_cifar.bench --epochs 50 --budget-mode wct \\
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
from benchmarks.models.resnet20_cifar.model import build_resnet20_cifar  # noqa: E402

BENCH = Benchmark(
    name="resnet20_cifar",
    display_name="CIFAR-10, ResNet-20",
    build_model=build_resnet20_cifar,
    build_data=cifar10,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # AdaFisher's CNN operating point (Table 9, CNNs; Appendix D's uniform 5e-4 CIFAR-10 decay).
    # conv_sua stays off: the widest input factor here is 64*3*3+1 = 577, not ResNet-50's 4609.
    hparams=HParams(lr=1e-3, baseline_lr=1e-3, weight_decay=5e-4, lam=1e-3, conv_sua=False),
    epochs=50,
    batch_size=128,
    output_group="cifar10",
)

if __name__ == "__main__":
    main(BENCH)
