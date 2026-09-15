"""B2 — ResNet-20, CIFAR-10 (``plan_exp_draft.md`` §4). 269 722 parameters.

    python -m benchmarks.resnet20_cifar.bench --epochs 50 --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.common.data import cifar10  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.resnet20_cifar.model import build_resnet20_cifar  # noqa: E402

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
