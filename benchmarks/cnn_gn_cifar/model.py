"""A2 of ``plan_exp_draft.md`` §4: a 3-convolution CNN (16-32-64) with a normalisation after each
convolution, global average pooling and a linear head, on CIFAR-10. 24 458 parameters in either
normalisation variant (``GroupNorm`` and ``BatchNorm2d`` both carry ``2C`` affine parameters).

It reproduces AdaFisher's own toy CNN bench (``adafisher_2405.16397.pdf`` App. B.2) at a size where
an exact Fisher is affordable in regime A, and it exists to isolate two things at once: **spatial
weight sharing** (Conv) and **normalisation**.

**``GroupNorm`` is deliberately not one of ``AdaFisherMulti``'s ``SUPPORTED_MODULES``**
(``Linear``/``Conv2d``/``BatchNorm2d``/``LayerNorm``), so its parameters receive the identity
preconditioner: no hook fires on them and no Fisher factor is ever built for them. That is the
point of the variant, not an oversight — ``plan_exp_draft_v0.md`` §2.5 picks GroupNorm precisely to
obtain a normalisation whose exact block is defined without the batch coupling that BN in *train*
mode introduces (a per-sample gradient, hence ``F`` itself, is undefined there). The
``norm="bn"`` variant is the one protocol P2 (``plan_exp_draft.md`` §7) compares against.

Configuration choices this project makes, not read off any paper: ``3x3`` convolutions with bias
and ``padding=1``, ``ReLU``, a ``2x2`` max-pool after each convolution block (32 -> 16 -> 8 -> 4),
and ``num_groups=8`` for ``GroupNorm`` (a divisor of all three widths).
"""

from __future__ import annotations

import torch
import torch.nn as nn

CHANNELS = (16, 32, 64)
NUM_GROUPS = 8


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "gn":
        return nn.GroupNorm(NUM_GROUPS, channels)
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    raise ValueError(f"norm must be 'gn' or 'bn'; got {kind!r}")


class CnnNorm(nn.Module):
    def __init__(self, norm: str = "gn", in_chans: int = 3,
                 channels: tuple[int, ...] = CHANNELS, num_classes: int = 10) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        width = in_chans
        for out_channels in channels:
            blocks += [
                nn.Conv2d(width, out_channels, kernel_size=3, padding=1, bias=True),
                _norm(norm, out_channels),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(2),
            ]
            width = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)).flatten(1))


def build_cnn_gn_cifar(norm: str = "gn", num_classes: int = 10) -> CnnNorm:
    return CnnNorm(norm=norm, num_classes=num_classes)
