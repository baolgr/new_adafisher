"""ResNet-50 with the **ImageNet** stem (``papers/resnet_1512.03385.pdf`` Table 1, "50-layer"
column + §3.4), the only architectural difference from ``benchmarks/resnet50_cifar/model.py``.

Table 1's first two rows are ``7x7, 64, stride 2`` followed by a ``3x3`` max-pool of stride 2, so
a 224x224 input reaches ``conv2_x`` at 56x56. The CIFAR variant replaces both with a single
stride-1 ``3x3`` convolution (§4.2), because that stem would collapse a 32x32 image to 8x8 before
the first residual block — which is exactly why the two cannot share one builder, and the whole
content of this file.

Everything else — the ``Bottleneck`` block, ``(3, 4, 6, 3)``, He fan-out init, the never-``+=``
residual add and ``inplace=False`` ReLUs that ``register_full_backward_hook`` requires — is
inherited unchanged from the CIFAR module, which this file imports rather than copies. That
import is the second sanctioned cross-folder import in ``benchmarks/`` (the first being
``vit_micro_cifar`` -> ``vit_small_cifar``, ``plan_exp_step1.md`` D1); duplicating 100 lines of
bottleneck would be the alternative, and a second copy to keep in sync with the hook constraints.

25 557 032 parameters — the standard ResNet-50 count at 1000 classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from benchmarks.resnet50_cifar.model import _RESNET50_BLOCKS, ResNetCIFAR


class ResNetImageNet(ResNetCIFAR):
    """``ResNetCIFAR`` with Table 1's ``conv1``/``pool1`` rows in place of the §4.2 CIFAR stem."""

    def __init__(
        self, num_classes: int = 1000, blocks: tuple[int, int, int, int] = _RESNET50_BLOCKS
    ) -> None:
        super().__init__(num_classes=num_classes, blocks=blocks)
        # Table 1: "7x7, 64, stride 2" then "3x3 max pool, stride 2". Replacing conv1 after
        # super().__init__() keeps one definition of every other layer; the replacement is
        # re-initialized here because _init_weights has already run on the module it replaces.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.avgpool(x).flatten(1))


def build_resnet50_imagenet(num_classes: int = 1000) -> ResNetImageNet:
    return ResNetImageNet(num_classes=num_classes, blocks=_RESNET50_BLOCKS)
