"""ResNet-50 with the CIFAR stem (``papers/resnet_1512.03385.pdf`` §4.1 "Deeper Bottleneck
Architectures" + Table 1), written from the paper rather than imported from
``torchvision``/``timm``. 23.52 M parameters, 107 hooked modules.

§4.2's CIFAR stem is a first ``3x3`` convolution on the 32x32 input — not the ``7x7``/stride-2
+ max-pool of the ImageNet variant, which would collapse 32x32 to 8x8 before the first
residual block.

One constraint comes from this project rather than from the paper, and shapes the code
(``plan_lot8.md`` §0.1, restated as ``plan_exp_step1.md`` §4's rule 1 for every model here):
``AdaFisherMulti`` registers ``register_full_backward_hook`` on every ``Conv2d``/
``BatchNorm2d``/``Linear``/``LayerNorm``, and PyTorch rejects a full backward hook on a module
whose output is later mutated in place. The residual add is therefore ``out = out + identity``
(never ``+=``) and every ``ReLU`` is ``inplace=False`` — which is why
``torchvision.models.resnet50`` (whose ``Bottleneck.forward`` does ``out += identity``) is not
reused.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_RESNET50_BLOCKS: tuple[int, int, int, int] = (3, 4, 6, 3)  # Table 1, "50-layer" column


class Bottleneck(nn.Module):
    """The 3-layer ``1x1 -> 3x3 -> 1x1`` residual block of §4.1 (p. 6), ``expansion = 4``.

    BN goes "right after each convolution and before activation" (§3.4), and the shortcut is a
    projection (``1x1`` conv + BN, "option B") exactly when the shortcut has to change shape.
    """

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=False)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + identity  # never `+=`: see this module's docstring, constraint 1
        return self.relu(out)


class ResNetCIFAR(nn.Module):
    def __init__(
        self, num_classes: int = 10, blocks: tuple[int, int, int, int] = _RESNET50_BLOCKS
    ) -> None:
        super().__init__()
        # §4.2's CIFAR stem: "the first layer is 3x3 convolutions" on the 32x32 input.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)

        self._in_channels = 64
        self.layer1 = self._make_layer(64, blocks[0], stride=1)
        self.layer2 = self._make_layer(128, blocks[1], stride=2)
        self.layer3 = self._make_layer(256, blocks[2], stride=2)
        self.layer4 = self._make_layer(512, blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [Bottleneck(self._in_channels, out_channels, stride=stride)]
        self._in_channels = out_channels * Bottleneck.expansion
        layers += [Bottleneck(self._in_channels, out_channels) for _ in range(1, num_blocks)]
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """§3.4: "We initialize the weights as in [13]" — He et al.'s fan-out normal init, with BN
        scales at 1 and shifts at 0.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.avgpool(x).flatten(1))


def build_resnet50_cifar(num_classes: int = 10) -> ResNetCIFAR:
    return ResNetCIFAR(num_classes=num_classes, blocks=_RESNET50_BLOCKS)
