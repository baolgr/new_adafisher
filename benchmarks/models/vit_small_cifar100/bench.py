"""ViT-S/4, 2 710 948 parameters, on 32x32 CIFAR-100, 100 classes.
The architecture is ``benchmarks/models/vit_small_cifar``'s with a 100-way head; same images,
same seeded 45k/5k split, same augmentation, so the two benches are comparable arm by arm.

Seven arms (``diag``/``kfac``/``ekfac``/``tkfac``/``tekfac``/``adam``/``adamw``) under the
wall-clock-time protocol: ``diag`` runs the nominal 50 epochs and its measured elapsed time is
every other arm's budget.

Hyperparameters are the CIFAR-10 counterpart's, not re-tuned for 100 classes; batch 128.

    python -m benchmarks.models.vit_small_cifar100.bench --epochs 50 --budget-mode wct \\
        --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from benchmarks.common.data import cifar100  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402
from benchmarks.models.vit_small_cifar.model import build_vit_small_cifar  # noqa: E402

BENCH = Benchmark(
    name="vit_small_cifar100",
    display_name="CIFAR-100, ViT-S/4",
    build_model=partial(build_vit_small_cifar, num_classes=100),
    build_data=cifar100,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    hparams=HParams(
        lr=1e-3, baseline_lr=1e-4, weight_decay=1e-2, lam=3e-3, decoupled_wd=True,
        conv_sua=False,  # one 4x4 patch-embedding conv; SUA would change nothing
    ),
    epochs=50,
    batch_size=128,
    output_group="cifar100",
)

if __name__ == "__main__":
    main(BENCH)
