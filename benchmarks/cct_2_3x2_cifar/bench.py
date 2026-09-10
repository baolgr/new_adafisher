"""B1 — CCT-2/3x2, CIFAR-10 (``plan_exp_draft.md`` §4). 283 723 parameters; AdaFisher's own model
(``adafisher_2405.16397.pdf`` Table 2).

    python -m benchmarks.cct_2_3x2_cifar.bench --epochs 50 --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from benchmarks.cct_2_3x2_cifar.model import build_cct_2_3x2_cifar  # noqa: E402
from benchmarks.common.data import cifar10  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402

BENCH = Benchmark(
    name="cct_2_3x2_cifar",
    display_name="CIFAR-10, CCT-2/3x2",
    build_model=build_cct_2_3x2_cifar,
    build_data=cifar10,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # AdaFisher's ViT operating point (Table 9, ViTs; AdaFisherViT.yaml): AdaFisherW / AdamW.
    hparams=HParams(lr=1e-3, baseline_lr=1e-4, weight_decay=1e-2, lam=3e-3, decoupled_wd=True),
    epochs=50,
    batch_size=128,
)

if __name__ == "__main__":
    main(BENCH)
