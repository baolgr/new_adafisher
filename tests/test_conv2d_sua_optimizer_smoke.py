"""End-to-end sanity check for lot 6 (docs/reports/plan_lot6.md §2.4): a real ``AdaFisherMulti``,
with real forward/backward hooks, running a few ``step()``s with ``conv_sua=True`` for each of the
four non-diagonal modes -- the lot-4 pattern (``test_conv2d_optimizer_smoke.py``), extended.

None of ``test_full_factors_match_diag.py``, ``test_frobenius_dominance.py`` or
``test_{kfac_ekfac,tkfac_tekfac}_precondition.py`` exercise hook registration/firing or the
``step()`` loop's module/parameter pairing under ``conv_sua=True`` -- this is the practical claim
lot 6 exists to make, so it gets its own smoke test rather than being left implicit.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from adafisher_modes import AdaFisherMulti
from conftest import seed_all
from test_conv2d_optimizer_smoke import TinyConvNet

# Cadence kwargs genuinely differ per mode (plan_lot2.md/plan_lot3.md: T_inv for kfac/tkfac, T_eig
# for ekfac/tekfac, T_re for tekfac only) -- same convention as test_conv2d_optimizer_smoke.py.
_MODE_KWARGS = {
    "kfac": {"T_inv": 2},
    "ekfac": {"T_eig": 2},
    "tkfac": {"T_inv": 2},
    "tekfac": {"T_eig": 2, "T_re": 1},
}


@pytest.mark.parametrize("mode", ["kfac", "ekfac", "tkfac", "tekfac"])
def test_optimizer_runs_on_conv_net_with_sua(mode: str) -> None:
    seed_all(0)
    model = TinyConvNet()
    initial = {name: p.clone() for name, p in model.named_parameters()}

    opt = AdaFisherMulti(
        model,
        lr=1e-3,
        fisher_mode=mode,
        TCov=1,
        Lambda=1e-2,
        conv_sua=True,
        **_MODE_KWARGS[mode],
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
        assert torch.isfinite(p).all(), f"{name} has non-finite entries after {mode}+SUA training"
        if not torch.equal(p, initial[name]):
            moved = True
    assert moved, f"no parameter moved under {mode}+SUA -- preconditioning was a silent no-op"


class TinyLinearNet(nn.Module):
    """No Conv2d at all -- used to confirm conv_sua is properly scoped (plan_lot6.md §2.4): it is
    silently ignored (factors.py's compute_h_full/augment_input only consult it on the Conv2d
    branch), so it must not perturb a Conv2d-free network's trajectory at all.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, 5, bias=True)
        self.fc2 = nn.Linear(5, 3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.mark.parametrize("mode", ["kfac", "ekfac", "tkfac", "tekfac"])
def test_conv_sua_is_inert_without_any_conv2d_module(mode: str) -> None:
    seed_all(1)
    model_a = TinyLinearNet()
    model_b = TinyLinearNet()
    model_b.load_state_dict(model_a.state_dict())

    opt_a = AdaFisherMulti(
        model_a, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, conv_sua=False, **_MODE_KWARGS[mode]
    )
    opt_b = AdaFisherMulti(
        model_b, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, conv_sua=True, **_MODE_KWARGS[mode]
    )

    for _ in range(6):
        seed_all(2)
        x = torch.randn(4, 6)
        target = torch.randint(0, 3, (4,))

        opt_a.zero_grad()
        loss_a = F.cross_entropy(model_a(x), target)
        loss_a.backward()
        opt_a.step()

        opt_b.zero_grad()
        loss_b = F.cross_entropy(model_b(x), target)
        loss_b.backward()
        opt_b.step()

    for (name, p_a), (_, p_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
        assert torch.equal(p_a, p_b), f"{name} diverged between conv_sua=False and conv_sua=True"
