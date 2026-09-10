"""The Vision Transformer of ``papers/vit_2010.11929.pdf`` §3.1 Eq. (1)-(4), in a 32x32-native
configuration, written from the paper rather than imported from ``torchvision``/``timm``.

``build_vit_small_cifar`` (2.69 M parameters, 39 hooked modules) is an **adaptation**, not one of
the paper's Table 1 variants (which are all 224px/patch-16 and JFT-scale). See ``plan_lot8.md``
§0.2 for the invariants kept from Table 1 (``MLP = 4D``, ``D/heads = 64``, a single-linear head).
Never attribute the configuration itself to the paper.

``ViTCIFAR`` is parametrized by ``patch_size``/``embed_dim``/``depth``/``num_heads``/``mlp_ratio``/
``pool`` so that ``benchmarks/vit_micro_cifar`` is a *configuration* of this architecture rather
than a copy of it (``plan_exp_step1.md`` D1: no architecture is ever duplicated).

Two constraints come from this project rather than from the paper, and shape the code
(``plan_lot8.md`` §0.2, restated as ``plan_exp_step1.md`` §4):

1. **Hook safety.** ``AdaFisherMulti`` registers ``register_full_backward_hook`` on every
   ``Conv2d``/``BatchNorm2d``/``Linear``/``LayerNorm``, and PyTorch rejects such a hook on a module
   whose output is later mutated in place: every residual add is ``x = x + ...``, never ``+=``.
2. **Every learnable projection must live inside one of those four module types.**
   ``nn.MultiheadAttention`` keeps Q/K/V in a single raw ``Parameter`` (``in_proj_weight``), which
   no hook ever sees: the whole attention mechanism would silently fall through to
   ``AdaFisherMulti``'s ``F~ = I`` path. Attention therefore exposes its projection as a single
   ``nn.Linear`` named ``qkv``. ``cls_token``/``pos_embed`` are raw ``Parameter``s by construction
   and belong to no hooked module — the case ``plan_lot8.md`` §0.3's optimizer fix exists for.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
        pool: str = "cls",
    ) -> None:
        super().__init__()
        if pool not in ("cls", "mean"):
            raise ValueError(f"pool must be 'cls' or 'mean'; got {pool!r}")
        self.pool = pool
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Eq. (1): the prepended learnable [class] embedding and the *learnable 1-D* position
        # embeddings ("we have not observed significant performance gains from using more advanced
        # 2D-aware position embeddings", §3.1 p. 4). Both are raw Parameters belonging to no hooked
        # module — the case plan_lot8.md §0.3's optimizer fix exists for. ``pool="mean"`` drops the
        # [class] token entirely (§3.1's footnote reports mean pooling as an equally viable
        # alternative), so there is no token to prepend and one fewer position to embed.
        n_positions = num_patches + (1 if pool == "cls" else 0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if pool == "cls" else None
        self.pos_embed = nn.Parameter(torch.zeros(1, n_positions, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)  # §3.1: "a single linear layer"

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
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
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Eq. (4): the [class] token's final state, or the mean over patch tokens.
        return self.head(x[:, 0] if self.cls_token is not None else x.mean(dim=1))


def build_vit_small_cifar(num_classes: int = 10) -> ViTCIFAR:
    """ViT-S/4 for 32x32: ``D=192``, ``depth=6``, ``heads=3`` (``D/heads = 64``, as ViT-Base),
    ``MLP = 4D = 768`` (as Table 1). plan_lot8.md §0.2.
    """
    return ViTCIFAR(num_classes=num_classes)
