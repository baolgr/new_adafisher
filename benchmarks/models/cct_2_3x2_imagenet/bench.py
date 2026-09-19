"""CCT-2/3x2, 411 433 parameters, on **ImageNet32** -- ImageNet-1K with every image squashed
to 32x32 (Chrabaszcz, Loshchilov & Hutter 2017, arXiv:1707.08819 §2), 1000 classes. The network
is ``benchmarks/models/cct_2_3x2_cifar``'s, unchanged apart from the head. Report results as
"ImageNet32", never as ImageNet-1K at its native resolution: published ImageNet numbers are not
a reference for it.

Seven arms (``diag``/``kfac``/``ekfac``/``tkfac``/``tekfac``/``adam``/``adamw``) under the
wall-clock-time protocol: ``diag`` runs the nominal 40 epochs and its measured elapsed time is
every other arm's budget.

Hyperparameters are the CIFAR counterpart's, not re-tuned; batch 256. The dataset must be staged
by hand -- see ``benchmarks/slurm/imagenet/README.md``.

    python -m benchmarks.models.cct_2_3x2_imagenet.bench --epochs 40 --budget-mode wct \\
        --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from benchmarks.models.cct_2_3x2_cifar.model import build_cct_2_3x2_cifar  # noqa: E402
from benchmarks.common.data import imagenet32  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402

BENCH = Benchmark(
    name="cct_2_3x2_imagenet",
    display_name="ImageNet-1K @32px, CCT-2/3x2",
    build_model=partial(build_cct_2_3x2_cifar, num_classes=1000),
    build_data=imagenet32,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # AdaFisher's ViT operating point (Table 9, ViTs; AdaFisherViT.yaml): AdaFisherW / AdamW.
    hparams=HParams(lr=1e-3, baseline_lr=1e-4, weight_decay=1e-2, lam=3e-3, decoupled_wd=True),
    epochs=40,
    batch_size=256,
    output_group="imagenet",
)

if __name__ == "__main__":
    main(BENCH)
