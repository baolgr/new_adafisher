"""3-convolution CNN 16-32-64 with GroupNorm, 24 458 parameters, on 32x32 CIFAR-10, 10 classes.

``--norm bn`` builds the BatchNorm variant; ``--norm gn`` (the default) leaves the normalisation
outside ``AdaFisherMulti``'s hooked module types on purpose -- see ``model.py``'s docstring.

Seven arms (``diag``/``kfac``/``ekfac``/``tkfac``/``tekfac``/``adam``/``adamw``) under the
wall-clock-time protocol: ``diag`` runs the nominal 30 epochs and its measured elapsed time is
every other arm's budget.

AdaFisher's CNN operating point (Table 9, and Appendix D's uniform CIFAR-10 decay): every arm
takes ``lr = 1e-3``, coupled weight decay ``5e-4``, ``lam = 1e-3``, batch 128, 30 epochs, cosine.

    python -m benchmarks.models.cnn_gn_cifar.bench --epochs 30 --budget-mode wct \\
        --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from benchmarks.models.cnn_gn_cifar.model import build_cnn_gn_cifar  # noqa: E402
from benchmarks.common.data import cifar10  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402

BENCH = Benchmark(
    name="cnn_gn_cifar",
    display_name="CIFAR-10, CNN 16-32-64 + GroupNorm",
    build_model=build_cnn_gn_cifar,
    model_choices={"norm": ("gn", "bn")},
    build_data=cifar10,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # AdaFisher's CNN operating point (Table 9, CNNs; Appendix D's uniform 5e-4 CIFAR-10 decay).
    hparams=HParams(lr=1e-3, baseline_lr=1e-3, weight_decay=5e-4, lam=1e-3, decoupled_wd=False),
    epochs=30,
    batch_size=128,
    output_group="cifar10",
)

if __name__ == "__main__":
    main(BENCH)
