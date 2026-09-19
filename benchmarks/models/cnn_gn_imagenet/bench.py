"""3-convolution CNN 16-32-64 with GroupNorm, 88 808 parameters, on **ImageNet32** --
ImageNet-1K with every image squashed to 32x32 (Chrabaszcz, Loshchilov & Hutter 2017,
arXiv:1707.08819 §2), 1000 classes. The network is ``benchmarks/models/cnn_gn_cifar``'s,
unchanged apart from the head. Report results as "ImageNet32", never as ImageNet-1K at its
native resolution: published ImageNet numbers are not a reference. Seven arms (``diag``/
``kfac``/``ekfac``/``tkfac``/``tekfac``/``adam``/``adamw``) under the wall-clock-time protocol:
``diag`` runs 40 epochs, and its elapsed time is every other arm's budget. Hyperparameters are
the CIFAR counterpart's, not re-tuned; batch 256. Stage the dataset first: see
``benchmarks/slurm/imagenet/README.md``.

    python -m benchmarks.models.cnn_gn_imagenet.bench --epochs 40 --budget-mode wct \\
        --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from benchmarks.models.cnn_gn_cifar.model import build_cnn_gn_cifar  # noqa: E402
from benchmarks.common.data import imagenet32  # noqa: E402
from benchmarks.common.loop import top1  # noqa: E402
from benchmarks.common.optimizers import HParams  # noqa: E402
from benchmarks.common.runner import Benchmark, main  # noqa: E402

BENCH = Benchmark(
    name="cnn_gn_imagenet",
    display_name="ImageNet-1K @32px, CNN 16-32-64 + GroupNorm",
    build_model=partial(build_cnn_gn_cifar, num_classes=1000),
    model_choices={"norm": ("gn", "bn")},
    build_data=imagenet32,
    loss_fn=nn.CrossEntropyLoss(),
    metric_fn=top1,
    # AdaFisher's CNN operating point (Table 9, CNNs), not re-tuned for ImageNet. The batch size
    # is AdaFisherCNN.yaml's own ``mini_batch_size: 256``, not the CIFAR benches' 128.
    hparams=HParams(lr=1e-3, baseline_lr=1e-3, weight_decay=5e-4, lam=1e-3, decoupled_wd=False),
    epochs=40,
    batch_size=256,
    output_group="imagenet",
)

if __name__ == "__main__":
    main(BENCH)
