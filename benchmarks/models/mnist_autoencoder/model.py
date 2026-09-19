"""The 8-layer MNIST auto-encoder of ``papers/ekfac_1806.03884.pdf`` §4.1 — the historical K-FAC
and EKFAC benchmark, and this repository's primary bench.

Input: one 28x28 greyscale MNIST image, flattened to 784 values. Output: a reconstruction of the
same 784 values; the task is mean-squared reconstruction error, so there is no accuracy.

Encoder ``784-1000-500-250-30``, symmetric decoder ``30-250-500-1000-784`` with untied weights,
sigmoid activations between every pair of layers and a linear output layer that reconstructs raw
pixel intensities. Eight ``nn.Linear`` layers, all of them hooked by ``AdaFisherMulti``, and no
other module type. **2 837 314 parameters** (counted, not quoted).

The saturating sigmoids are the point: this is a network on which a badly scaled preconditioner
visibly stalls, which is why it is the primary bench.
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
