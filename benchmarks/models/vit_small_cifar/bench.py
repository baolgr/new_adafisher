"""ViT-S/4, 2 693 578 parameters, on 32x32 CIFAR-10, 10 classes.
Seven arms (``diag``/``kfac``/``ekfac``/``tkfac``/``tekfac``/``adam``/``adamw``) under the
wall-clock-time protocol: ``diag`` runs 50 epochs, and its elapsed time is every other arm's
budget. AdaFisher's tuned ViT operating point (Table 9), not re-tuned here: ``lr = 1e-3`` for the
Fisher arms, ``1e-4`` for the baselines, decoupled weight decay ``1e-2``, ``lam = 3e-3``,
batch 128.

    python -m benchmarks.models.vit_small_cifar.bench --epochs 50 --budget-mode wct \\
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
from benchmarks.models.vit_small_cifar.model import build_vit_small_cifar  # noqa: E402

BENCH = Benchmark(
    name="vit_small_cifar",
    display_name="CIFAR-10, ViT-S/4",
    build_model=build_vit_small_cifar,
    build_data=cifar10,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    hparams=HParams(
        lr=1e-3,  # Table 9, ViTs / AdaFisherW
        baseline_lr=1e-4,  # Table 9, ViTs / AdamW ("adopted from the original publications")
        weight_decay=1e-2,  # AdaFisherViT.yaml: weight_decay
        lam=3e-3,  # AdaFisherViT.yaml: Lambda
        decoupled_wd=True,  # "AdamW and AdaFisherW were applied for all ViT experiments"
        conv_sua=False,  # one 4x4 patch-embedding conv; SUA would change nothing
    ),
    epochs=50,
    batch_size=128,
    output_group="cifar10",
)

if __name__ == "__main__":
    main(BENCH)
