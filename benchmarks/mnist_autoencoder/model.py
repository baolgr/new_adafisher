"""The 8-layer MNIST auto-encoder of ``ekfac_1806.03884.pdf`` §4.1 — the historical K-FAC / EKFAC
bench, and this repository's primary bench since lot 1 (``plan.md`` §6.3).

Encoder ``784-1000-500-250-30`` with sigmoid activations, symmetric decoder with untied weights;
a linear output layer reconstructing raw pixel intensities. 2 837 314 parameters, ``Linear`` only.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

ENCODER_DIMS = [784, 1000, 500, 250, 30]


class AutoEncoder(nn.Module):
    """8 Linear layers, sigmoid activations, untied symmetric decoder."""

    def __init__(self, dims: List[int] = ENCODER_DIMS) -> None:
        super().__init__()
        encoder = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        decoder_dims = list(reversed(dims))
        decoder = [nn.Linear(decoder_dims[i], decoder_dims[i + 1])
                   for i in range(len(decoder_dims) - 1)]
        self.layers = nn.ModuleList(encoder + decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = torch.sigmoid(layer(x))
        return self.layers[-1](x)  # linear output layer, reconstructs raw pixel intensities


def build_mnist_autoencoder() -> AutoEncoder:
    return AutoEncoder()
