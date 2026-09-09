"""The two CIFAR-10 networks of lot 8 (docs/reports/plan_lot8.md §0.1, §0.2), written from their
papers rather than imported from ``torchvision``/``timm``.

- ``build_resnet50_cifar`` — ResNet-50 (``papers/resnet_1512.03385.pdf`` §4.1 "Deeper Bottleneck
  Architectures" + Table 1) with the paper's own CIFAR stem (§4.2: a first ``3x3`` convolution on
  the 32x32 input, no ``7x7``/stride-2 + max-pool, which would collapse 32x32 to 8x8 before the
  first residual block). 23.52 M parameters, 107 hooked modules.
- ``build_vit_small_cifar`` — a Vision Transformer following ``papers/vit_2010.11929.pdf`` §3.1
  Eq. (1)-(4), in a 32x32-native configuration that is an **adaptation**, not one of the paper's
  Table 1 variants (which are all 224px/patch-16 and JFT-scale). See plan_lot8.md §0.2 for the
  invariants kept from Table 1 (``MLP = 4D``, ``D/heads = 64``, single-linear head). 2.69 M
  parameters, 39 hooked modules.

Two constraints come from this project rather than from either paper, and shape the code:

1. **Hook safety.** ``AdaFisherMulti`` registers ``register_full_backward_hook`` on every
   ``Conv2d``/``BatchNorm2d``/``Linear``/``LayerNorm``. A full backward hook on a module whose
   output is later mutated in place is exactly what PyTorch rejects, so the residual add is
   ``out = out + identity`` (never ``+=``) and every ``ReLU`` is ``inplace=False``. This is why
   ``torchvision.models.resnet50`` (whose ``Bottleneck.forward`` does ``out += identity``) is not
   reused.
2. **Every learnable projection must live inside one of those four module types.**
   ``nn.MultiheadAttention`` keeps Q/K/V in a single raw ``Parameter`` (``in_proj_weight``), which
   no hook ever sees: the whole attention mechanism would silently fall through to
   ``AdaFisherMulti``'s ``F~ = I`` path. The attention block below therefore exposes its projection
   as a single ``nn.Linear`` named ``qkv`` (plan_lot8.md §0.2).
"""

from __future__ import annotations

from typing import Callable, Dict, NamedTuple, Tuple

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------------------
# ResNet-50, CIFAR variant (resnet_1512.03385.pdf Table 1 + §4.1 + §4.2)
# ----------------------------------------------------------------------------------------------

_RESNET50_BLOCKS: Tuple[int, int, int, int] = (3, 4, 6, 3)  # Table 1, "50-layer" column


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
        self, num_classes: int = 10, blocks: Tuple[int, int, int, int] = _RESNET50_BLOCKS
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


# ----------------------------------------------------------------------------------------------
# ViT, 32x32 adaptation (vit_2010.11929.pdf §3.1 Eq. (1)-(4); plan_lot8.md §0.2)
# ----------------------------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Eq. (1)'s trainable linear projection of flattened ``P x P`` patches to width ``D``, written
    as the equivalent ``Conv2d(3, D, kernel_size=P, stride=P)``.
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} is not divisible by patch_size {patch_size}")
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, C, H', W') -> (B, N, D)


class Attention(nn.Module):
    """Multi-head self-attention (Appendix A). ``qkv`` is deliberately one ``nn.Linear`` rather
    than ``nn.MultiheadAttention`` — see this module's docstring, constraint 2.
    """

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class Mlp(nn.Module):
    """"The MLP contains two layers with a GELU non-linearity" (§3.1, p. 4)."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Eq. (2)-(3): "Layernorm (LN) is applied before every block, and residual connections after
    every block" (§3.1, p. 4) — i.e. pre-LN.
    """

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float, drop: float, attn_drop: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViTCIFAR(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Eq. (1): the prepended learnable [class] embedding and the *learnable 1-D* position
        # embeddings ("we have not observed significant performance gains from using more advanced
        # 2D-aware position embeddings", §3.1 p. 4). Both are raw Parameters belonging to no hooked
        # module — the case plan_lot8.md §0.3's optimizer fix exists for.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)  # §3.1: "a single linear layer"

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])  # Eq. (4): the [class] token's final state


def build_vit_small_cifar(num_classes: int = 10) -> ViTCIFAR:
    """ViT-S/4 for 32x32: ``D=192``, ``depth=6``, ``heads=3`` (``D/heads = 64``, as ViT-Base),
    ``MLP = 4D = 768`` (as Table 1). plan_lot8.md §0.2.
    """
    return ViTCIFAR(num_classes=num_classes)


# ----------------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------------


class ModelSpec(NamedTuple):
    build: Callable[[int], nn.Module]
    display_name: str


MODELS: Dict[str, ModelSpec] = {
    "resnet50": ModelSpec(build_resnet50_cifar, "ResNet50"),
    "vit_small": ModelSpec(build_vit_small_cifar, "ViT-S/4"),
}
