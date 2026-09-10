"""B1 of ``plan_exp_draft.md`` §4: CCT-2/3x2 for CIFAR-10 (Hassani et al., *Escaping the Big Data
Paradigm with Compact Transformers*, arXiv:2104.05704). 283 723 parameters — the paper's own
"0.28M" for its smallest variant, and one of AdaFisher's own models (``adafisher_2405.16397.pdf``
Table 2).

Read off the paper's §3: ``CCT-L/PxN`` names ``L`` transformer encoder layers and an ``N``-layer
convolutional tokenizer of kernel size ``P``. CCT-2/3x2 is therefore ``L=2``, ``P=3``, ``N=2``, at
the paper's stated small-model width ``d = 128``, 2 heads and ``MLP ratio = 1`` (Table 2's
"CCT-2" row). Two components distinguish CCT from a plain ViT and are the reason this model is in
the campaign at all:

- **the convolutional tokenizer** (§3.1): ``Conv2d -> ReLU -> MaxPool``, twice, replacing patch
  embedding; it downsamples 32x32 to an 8x8 = 64-token sequence, so weight sharing enters through
  a *tokenizer* rather than a stride-``P`` patch projection;
- **sequence pooling** (§3.2, Eq. (2)-(3)): ``z' = softmax(g(z)^T) z`` with ``g`` a single
  ``Linear(d, 1)``, replacing the ``[class]`` token entirely.

Deviations from the reference implementation, stated rather than discovered: stochastic depth
(``0.1`` in the authors' CIFAR configuration) is **not** implemented — it randomizes the
computational graph per sample, which a curvature-fidelity bench has no use for; dropout and
attention dropout keep their reference values. Neither affects the parameter count.

Hook safety (``plan_exp_step1.md`` §4): every residual add is ``x = x + ...``, never ``+=``; the
attention projection is a single hookable ``nn.Linear`` named ``qkv``, never
``nn.MultiheadAttention``. The learned position embedding is a raw ``Parameter`` belonging to no
hooked module, by construction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvTokenizer(nn.Module):
    """§3.1's convolutional tokenizer: ``N`` blocks of ``Conv2d(k=P) -> ReLU -> MaxPool(3, s=2)``,
    emitting a ``(B, sequence_length, d)`` token sequence. Convolutions are bias-free, as in the
    authors' implementation.
    """

    def __init__(self, in_chans: int = 3, embed_dim: int = 128, n_conv_layers: int = 2,
                 kernel_size: int = 3, in_planes: int = 64) -> None:
        super().__init__()
        widths = [in_chans] + [in_planes] * (n_conv_layers - 1) + [embed_dim]
        blocks: list[nn.Module] = []
        for i in range(n_conv_layers):
            blocks += [
                nn.Conv2d(widths[i], widths[i + 1], kernel_size=kernel_size, stride=1,
                          padding=kernel_size // 2, bias=False),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.n_conv_layers = n_conv_layers

    def sequence_length(self, img_size: int) -> int:
        side = img_size
        for _ in range(self.n_conv_layers):
            side = (side + 2 * 1 - 3) // 2 + 1
        return side * side

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x).flatten(2).transpose(1, 2)  # (B, C, H', W') -> (B, N, D)


class Attention(nn.Module):
    """Multi-head self-attention. ``qkv`` is one bias-free ``nn.Linear`` (the authors' choice, and
    the hookable form this project requires).
    """

    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.1,
                 proj_drop: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
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


class Block(nn.Module):
    """Pre-LN transformer encoder layer, ``MLP = mlp_ratio * d`` with a GELU non-linearity."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float,
                 attn_drop: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.drop(self.fc2(self.drop(self.act(self.fc1(self.norm2(x))))))


class CCT(nn.Module):
    def __init__(self, img_size: int = 32, in_chans: int = 3, num_classes: int = 10,
                 embed_dim: int = 128, depth: int = 2, num_heads: int = 2,
                 mlp_ratio: float = 1.0, n_conv_layers: int = 2, kernel_size: int = 3,
                 drop: float = 0.1, attn_drop: float = 0.1) -> None:
        super().__init__()
        self.tokenizer = ConvTokenizer(in_chans, embed_dim, n_conv_layers, kernel_size)
        sequence_length = self.tokenizer.sequence_length(img_size)

        # §3.2: a *learnable* positional embedding, added to the tokenizer's output. A raw
        # Parameter, belonging to no hooked module.
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length, embed_dim))
        self.pos_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # Eq. (2)-(3)'s SeqPool: g(z) in R^{N x 1}, softmax over the sequence, then a weighted sum.
        self.attention_pool = nn.Linear(embed_dim, 1)
        self.fc = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.2)
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
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_drop(self.tokenizer(x) + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        weights = self.attention_pool(x).transpose(-1, -2).softmax(dim=-1)  # (B, 1, N)
        return self.fc((weights @ x).squeeze(-2))


def build_cct_2_3x2_cifar(num_classes: int = 10) -> CCT:
    """CCT-2/3x2: 2 encoder layers, a 2-conv ``3x3`` tokenizer, ``d=128``, 2 heads,
    ``mlp_ratio=1``. 283 723 parameters (paper: "0.28M").
    """
    return CCT(num_classes=num_classes)
