"""A1 of ``plan_exp_draft.md`` §4: MLP ``784-32-32-10`` with a ``LayerNorm(32)`` after each hidden
``Linear``, on MNIST. 26 634 parameters.

The cleanest case K-FAC theory has: plain ``Linear`` layers with **no weight sharing**, so the
expand/reduce distinction of Eschenhagen et al. (arXiv:2311.00636) does not arise, plus one
normalisation type. It is the control model of the Fisher-drift campaign, and small enough
(``P <= 2.9e4``) for regime A — a dense ``P x P`` Fisher in fp64.

Configuration choices this project makes, not read off any paper: the hidden activation is
``ReLU`` and normalisation is applied **after** the linear map and **before** the activation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN_DIMS = (32, 32)


class MlpLN(nn.Module):
    def __init__(self, in_features: int = 784, hidden_dims: tuple[int, ...] = HIDDEN_DIMS,
                 num_classes: int = 10) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = in_features
        for hidden in hidden_dims:
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=False)]
            width = hidden
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.flatten(1)))


def build_mlp_ln_mnist(num_classes: int = 10) -> MlpLN:
    return MlpLN(num_classes=num_classes)
