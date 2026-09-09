"""End-to-end sanity check for lot 4 (docs/reports/plan_lot4.md §2.4): a real ``AdaFisherMulti``,
with real forward/backward hooks (not the direct ``update_input_factor``/``update_output_factor``
calls the other lot-4 tests use), running a few ``step()``s on a small ``Conv2d``+``Linear`` network
for each of the four non-diagonal modes.

None of ``test_full_factors_match_diag.py``, ``test_frobenius_dominance.py`` or
``test_{kfac_ekfac,tkfac_tekfac}_precondition.py`` exercise hook registration/firing or the
``step()`` loop's module/parameter pairing for a ``Conv2d`` layer under a non-``diag`` mode — this is
the practical claim lot 4 exists to make, so it gets its own smoke test rather than being left
implicit.

No ``BatchNorm2d``/``LayerNorm`` in ``TinyConvNet`` (unlike ``conftest.py``'s own
``TinyMultiLayerNet``): those two layer types still raise ``NotImplementedError`` for the four new
modes until lot 5.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from adafisher_modes import AdaFisherMulti
from conftest import seed_all


class TinyConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 5 * 5, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(x.flatten(1))


# Cadence kwargs genuinely differ per mode (plan_lot2.md/plan_lot3.md: T_inv for kfac/tkfac, T_eig
# for ekfac/tekfac, T_re for tekfac only) — see approximations/__init__.py's MODES factory typing.
_MODE_KWARGS = {
    "kfac": {"T_inv": 2},
    "ekfac": {"T_eig": 2},
    "tkfac": {"T_inv": 2},
    "tekfac": {"T_eig": 2, "T_re": 1},
}


@pytest.mark.parametrize("mode", ["kfac", "ekfac", "tkfac", "tekfac"])
def test_optimizer_runs_on_conv_net(mode: str) -> None:
    seed_all(0)
    model = TinyConvNet()
    initial = {name: p.clone() for name, p in model.named_parameters()}

    opt = AdaFisherMulti(
        model, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, **_MODE_KWARGS[mode]
    )

    for _ in range(6):
        x = torch.randn(4, 2, 5, 5)
        target = torch.randint(0, 3, (4,))
        opt.zero_grad()
        loss = F.cross_entropy(model(x), target)
        loss.backward()
        opt.step()

    moved = False
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"{name} has non-finite entries after {mode} training"
        if not torch.equal(p, initial[name]):
            moved = True
    assert moved, f"no parameter moved under {mode} — preconditioning was a silent no-op"
