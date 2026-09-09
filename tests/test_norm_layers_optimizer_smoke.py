"""End-to-end sanity check for lot 5 (docs/reports/plan_lot5.md §2.4): a real ``AdaFisherMulti``,
with real forward/backward hooks (not the direct ``update_input_factor``/``update_output_factor``
calls the other lot-5 tests use), running a few ``step()``s on ``conftest.py``'s existing
``TinyMultiLayerNet`` — the one network in this codebase already covering all four
``SUPPORTED_MODULES``, including both ``BatchNorm2d`` and ``LayerNorm`` — for each of the four
non-diagonal modes. No new network is needed here, unlike lot 4's purpose-built ``TinyConvNet``,
which deliberately excluded normalisation layers because they were unsupported by these four modes
until this lot.

Also implements ``plan.md`` §8's own lot-5 exit criterion, verbatim: "loss non-regression vs. ``diag``
on a small CNN."
"""

from __future__ import annotations

import pytest
import torch
from adafisher_modes import AdaFisherMulti
from conftest import TinyMultiLayerNet, seed_all

# Cadence kwargs genuinely differ per mode (plan_lot2.md/plan_lot3.md: T_inv for kfac/tkfac, T_eig
# for ekfac/tekfac, T_re for tekfac only) — mirrors test_conv2d_optimizer_smoke.py's own dict.
_MODE_KWARGS = {
    "kfac": {"T_inv": 2},
    "ekfac": {"T_eig": 2},
    "tkfac": {"T_inv": 2},
    "tekfac": {"T_eig": 2, "T_re": 1},
}


@pytest.mark.parametrize("mode", ["kfac", "ekfac", "tkfac", "tekfac"])
def test_optimizer_runs_on_full_net(mode: str) -> None:
    seed_all(0)
    model = TinyMultiLayerNet()
    initial = {name: p.clone() for name, p in model.named_parameters()}

    opt = AdaFisherMulti(
        model, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, **_MODE_KWARGS[mode]
    )

    for _ in range(6):
        x = torch.randn(6, 2, 5, 5)
        opt.zero_grad()
        loss = model(x).pow(2).sum()
        loss.backward()
        opt.step()

    moved = False
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"{name} has non-finite entries after {mode} training"
        if not torch.equal(p, initial[name]):
            moved = True
    assert moved, f"no parameter moved under {mode} — preconditioning was a silent no-op"


@pytest.mark.parametrize("mode", ["kfac", "ekfac", "tkfac", "tekfac"])
def test_loss_non_regression_vs_diag(mode: str) -> None:
    """``plan.md`` §8's own lot-5 exit test, verbatim: with identical initial weights and identical
    minibatches, a few steps of the new mode must not diverge relative to ``diag`` — a sanity floor,
    not a claim that the new modes beat ``diag`` (``plan.md`` §6.1: dominance over ``diag`` is
    measured, not asserted, and that reasoning applies here too — ``F~_D`` is not of the form the
    four new modes share, and comparing final losses is not a Frobenius-norm statement in the first
    place).
    """
    seed_all(1)
    model_new = TinyMultiLayerNet()
    model_diag = TinyMultiLayerNet()
    model_diag.load_state_dict(model_new.state_dict())

    opt_new = AdaFisherMulti(
        model_new, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, **_MODE_KWARGS[mode]
    )
    opt_diag = AdaFisherMulti(model_diag, lr=1e-3, fisher_mode="diag", TCov=1, Lambda=1e-2)

    batches = [torch.randn(6, 2, 5, 5) for _ in range(6)]
    for x in batches:
        for model, opt in ((model_new, opt_new), (model_diag, opt_diag)):
            model.zero_grad()
            loss = model(x).pow(2).sum()
            loss.backward()
            opt.step()

    loss_new = model_new(batches[-1]).pow(2).sum()
    loss_diag = model_diag(batches[-1]).pow(2).sum()

    assert torch.isfinite(loss_new)
    assert loss_new <= 10 * loss_diag + 1.0, (
        f"{mode} diverged relative to diag: loss_{mode}={loss_new:.4f}, loss_diag={loss_diag:.4f}"
    )
