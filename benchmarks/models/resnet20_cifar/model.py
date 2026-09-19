"""ResNet-20 for CIFAR-10 (``papers/resnet_1512.03385.pdf`` §4.2), written from the paper.
**269 722 parameters** (counted) — the paper's own "0.27M" for its ``n = 3`` network.

Input: one 32x32 RGB image. Output: 10 class logits.

§4.2 verbatim: "The network inputs are 32x32 images [...] The first layer is 3x3 convolutions.
Then we use a stack of 6n layers with 3x3 convolutions on the feature maps of sizes {32, 16, 8}
respectively, with 2n layers for each feature map size. The numbers of filters are {16, 32, 64}
respectively. The subsampling is performed by convolutions with a stride of 2. The network ends
with a global average pooling, a 10-way fully-connected layer, and softmax."

Shortcuts are **option A** — "identity mapping [...] with extra zero entries padded for increasing
dimensions [...] this option introduces no extra parameter" (§3.3, used in §4.2: "we use identity
shortcuts in all cases (i.e. option A)"). This is what makes the count exactly 269 722; option B,
a projection shortcut, would add about 2 000 parameters and no longer match the published figure.
Weights are initialized as in §3.4, He fan-out normal for convolutions, scales at 1 and shifts at
0 for BatchNorm.

One constraint comes from this project rather than from the paper, and shapes the code.
``AdaFisherMulti`` registers a full backward hook on every ``Conv2d``/``BatchNorm2d``/``Linear``/
``LayerNorm``, and PyTorch rejects such a hook on a module whose output is later mutated in place.
The residual add is therefore ``out = out + shortcut``, never ``+=``, and every ``ReLU`` is
``inplace=False``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCKS_PER_STAGE = 3  # n = 3 -> 6n + 2 = 20 layers
STAGE_CHANNELS = (16, 32, 64)


class ZeroPadShortcut(nn.Module):
    """Option A: stride-2 subsampling of the identity, then zero-padding of the new channels.
    Parameter-free by construction (§3.3).
    """

    def __init__(self, stride: int, pad_channels: int) -> None:
        super().__init__()
        self.stride = stride
        self.pad_channels = pad_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :, :: self.stride, :: self.stride]
        left = self.pad_channels // 2
        return F.pad(x, (0, 0, 0, 0, left, self.pad_channels - left))


class BasicBlock(nn.Module):
    """The 2-layer ``3x3 -> 3x3`` residual block of Fig. 3 / §4.2, ``expansion = 1``."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ZeroPadShortcut(stride, out_channels - in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # never `+=`: full backward hooks reject in-place outputs
        return self.relu(out)


class ResNetCIFARSmall(nn.Module):
    def __init__(self, blocks_per_stage: int = BLOCKS_PER_STAGE,
                 channels: tuple[int, ...] = STAGE_CHANNELS, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=False)

        self._in_channels = channels[0]
        self.stages = nn.Sequential(*[
            self._make_stage(width, blocks_per_stage, stride=1 if i == 0 else 2)
            for i, width in enumerate(channels)
        ])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)
        self._init_weights()

    def _make_stage(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self._in_channels, out_channels, stride=stride)]
        self._in_channels = out_channels
        layers += [BasicBlock(out_channels, out_channels) for _ in range(1, num_blocks)]
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """§3.4: "We initialize the weights as in [13]" — He et al.'s fan-out normal init."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.stages(x)
        return self.fc(self.avgpool(x).flatten(1))


def build_resnet20_cifar(num_classes: int = 10) -> ResNetCIFARSmall:
    return ResNetCIFARSmall(num_classes=num_classes)
