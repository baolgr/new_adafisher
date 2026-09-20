"""The ``official`` arm: the authors' own published optimizer, run unmodified.

``benchmarks/common/optimizers.py`` can build eight things. Two of them are not this project's code
at all: ``reference`` is ``reference_repos/FisherAdapTune/scripts/adafisher.py``'s ``AdaFisher``, and
``official`` is ``reference_repos/AdaFisher/optimizers/AdaFisher.py``'s — the optimizer of the
authors' published repository, the one that produced the paper's Table 2. Both are loaded by file
path, so the read-only reference repositories' package ``__init__`` files are never imported.

Why the arm exists. A negative result about AdaFisher is only worth stating if the authors' own code
shows it too. The two reference arms are how this repository checks that: ``reference`` is the
partner of ``diag --no-minmax`` (``test_diag_bitexact.py`` shows they agree bit for bit), and
``official`` is the partner of ``diag`` at its shipped default, because the published repository
applies Eq. (4)'s min-max normalisation and FisherAdapTune does not. On the EMA the two references
are numerically the same: the published code's single ``gamma = 0.8`` gives the coefficients
``(0.08, 0.008)``, which is exactly this project's default pair ``gammas = (0.92, 0.008)``
(``docs/reports/archives/plan.md`` §1.4).

The tolerance below is measured, not guessed. ``diag`` differs from the published code only in how
the update is applied — this project returns a direction and adds it, the published code uses the
fused ``addcdiv_`` (``CLAUDE.md`` §2.1) — so the gap is a few ULPs per step. Measured worst relative
parameter gap: ``4.6e-7`` on a Linear-only net over 40 steps and ``3.6e-7`` on ``TinyMultiLayerNet``
(all four hooked layer types, BatchNorm feedback included) over 20 steps, both at ``lr = 1e-2``. The
assertions use ``1e-5``, an order and a half above what was measured.

All offline; no dataset, no GPU.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from adafisher_modes import AdaFisherMulti
from conftest import TinyMultiLayerNet, seed_all

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.optimizers import (  # noqa: E402
    ARMS,
    OFFICIAL_ADAFISHER,
    HParams,
    build_optimizer,
    official_gamma,
)

AGREEMENT_RTOL = 1e-5


def _linear_only_net() -> nn.Module:
    return nn.Sequential(nn.Linear(12, 8), nn.Sigmoid(), nn.Linear(8, 5))


def _worst_relative_parameter_gap(factory, x, y, steps: int, lr: float, tcov: int) -> float:
    """Train one copy with the authors' optimizer and one with ``diag``, from identical weights."""
    seed_all(0)
    model_official = factory()
    model_diag = copy.deepcopy(model_official)

    hp = HParams(lr=lr, tcov=tcov)
    opt_official = build_optimizer("official", model_official, hp)
    opt_diag = AdaFisherMulti(
        model_diag, lr=hp.lr, beta=hp.beta, Lambda=hp.lam, gammas=list(hp.gammas),
        TCov=tcov, fisher_mode="diag", minmax_normalization=True,
    )

    worst = 0.0
    for _ in range(steps):
        for model, opt in ((model_official, opt_official), (model_diag, opt_diag)):
            opt.zero_grad()
            nn.functional.mse_loss(model(x), y).backward()
            opt.step()
        for a, b in zip(model_official.parameters(), model_diag.parameters()):
            scale = max(a.abs().max().item(), 1e-12)
            worst = max(worst, (a - b).abs().max().item() / scale)
    return worst


def test_the_arm_is_registered():
    assert "official" in ARMS


def test_the_arm_builds_the_authors_own_class_from_their_own_file():
    opt = build_optimizer("official", _linear_only_net(), HParams())
    assert type(opt).__name__ == "AdaFisher"
    assert type(opt).__module__ == "_ref_official_adafisher"
    assert OFFICIAL_ADAFISHER.is_file()
    assert OFFICIAL_ADAFISHER.parts[-3:] == ("AdaFisher", "optimizers", "AdaFisher.py")


def test_decoupled_weight_decay_selects_the_authors_adafisherw():
    opt = build_optimizer("official", _linear_only_net(), replace(HParams(), decoupled_wd=True))
    assert type(opt).__name__ == "AdaFisherW"


def test_official_gamma_is_the_scalar_behind_this_projects_default_pair():
    # The published EMA is (gamma*1e-1) * old + (gamma*1e-2) * new; this project stores
    # (1 - gammas[0]) * old + gammas[1] * new. gamma = 0.8 <-> gammas = (0.92, 0.008).
    assert official_gamma(HParams()) == pytest.approx(0.8)


def test_official_gamma_refuses_a_pair_no_single_gamma_can_express():
    with pytest.raises(ValueError, match="not reachable"):
        official_gamma(replace(HParams(), gammas=(0.5, 0.008)))


def test_official_gamma_refuses_the_single_gamma_flag():
    # --gamma is AdaFisherMulti's Eq. (3) correction, which the published code does not implement.
    with pytest.raises(ValueError, match="cannot take --gamma"):
        official_gamma(replace(HParams(), gamma=0.8))


def test_the_arm_takes_no_adafishermulti_overrides():
    with pytest.raises(ValueError, match="takes no AdaFisherMulti overrides"):
        build_optimizer("official", _linear_only_net(), HParams(), ema_seed_first=True)


def test_every_parameter_moves():
    net = _linear_only_net()
    before = [p.detach().clone() for p in net.parameters()]
    opt = build_optimizer("official", net, HParams(lr=1e-2, tcov=2))
    x, y = torch.randn(16, 12), torch.randn(16, 5)
    for _ in range(3):
        opt.zero_grad()
        nn.functional.mse_loss(net(x), y).backward()
        opt.step()
    for old, new in zip(before, net.parameters()):
        assert not torch.equal(old, new)
        assert torch.isfinite(new).all()


def test_agrees_with_diag_on_a_linear_only_net():
    torch.manual_seed(1)
    x, y = torch.randn(16, 12), torch.randn(16, 5)
    gap = _worst_relative_parameter_gap(_linear_only_net, x, y, steps=40, lr=1e-2, tcov=2)
    assert gap < AGREEMENT_RTOL, gap


def test_agrees_with_diag_on_all_four_hooked_layer_types():
    torch.manual_seed(1)
    x, y = torch.randn(8, 2, 5, 5), torch.randn(8, 4)
    gap = _worst_relative_parameter_gap(TinyMultiLayerNet, x, y, steps=20, lr=1e-2, tcov=2)
    assert gap < AGREEMENT_RTOL, gap
