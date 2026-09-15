"""B1 on CIFAR-100 — CCT-2/3x2, 100-way head (295 333 parameters). AdaFisher's own model
(``adafisher_2405.16397.pdf`` Table 2, whose CIFAR-100 column reports it too).

    python -m benchmarks.cct_2_3x2_cifar100.bench --epochs 50 --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.cct_2_3x2_cifar.model import build_cct_2_3x2_cifar  # noqa: E402
from benchmarks.common.data import cifar100  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402

BENCH = Benchmark(
    name="cct_2_3x2_cifar100",
    display_name="CIFAR-100, CCT-2/3x2",
    build_model=partial(build_cct_2_3x2_cifar, num_classes=100),
    build_data=cifar100,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # AdaFisher's ViT operating point (Table 9, ViTs; AdaFisherViT.yaml): AdaFisherW / AdamW.
    hparams=HParams(lr=1e-3, baseline_lr=1e-4, weight_decay=1e-2, lam=3e-3, decoupled_wd=True),
    epochs=50,
    batch_size=128,
    output_group="cifar100",
)

if __name__ == "__main__":
    main(BENCH)
