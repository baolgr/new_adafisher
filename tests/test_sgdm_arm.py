"""The ``sgdm`` arm: ``AdaFisherMulti`` with every Fisher estimate replaced by ``lam * I``.

E18 of ``docs/reports/plan_lambda_dominance.md`` asks whether the Fisher arms' results on ViT-S are
the results of momentum SGD. That question is only as sharp as the control is exact, so the arm is
pinned here three ways:

1. **It is AdaFisherMulti's own update, minus the curvature.** With ``gammas = (1.0, 0.0)`` every
   running average in ``AdaFisherMulti`` is multiplied by 0 and receives 0 of each new observation,
   so the operator ``diag`` divides by is exactly ``lam`` everywhere. The arm must reproduce that
   optimizer **bit for bit**, on a net with all four hooked layer types and on a ViT whose
   ``cls_token``/``pos_embed`` belong to no hooked module (they must keep AdaFisher's own fallback,
   not be divided by ``lam``). Measured: ``torch.equal`` on every parameter, 12 configurations.
2. **It is torch's momentum SGD at ``lr * (1 - beta) / (lam * (1 - beta^t))``**, which is what gives
   it its name and its "0.033 on the ViT benches".
3. **The bound E18's prediction rests on.** At the shipped settings ``diag`` divides by
   ``lam + H'S'`` with ``H'``, ``S'`` running averages of min-max-normalised factors, so
   ``0 <= H'S' <= X_k^2`` with ``X_k = 0.08^k + (0.008 / 0.92) * (1 - 0.08^k)`` after ``k`` factor
   updates: ``7.7e-3``, ``2.3e-4``, ``8.5e-5``, then ``7.56e-5`` in the limit. At ``lam = 3e-3``
   ``diag``'s step is therefore at least 0.975 times this arm's, coordinate by coordinate, once
   three factor updates have happened.

All offline; no dataset, no GPU.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from adafisher_modes import AdaFisherMulti
from conftest import TinyMultiLayerNet

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.optimizers import (  # noqa: E402
    ARMS,
    HParams,
    LambdaLimitSGD,
    build_optimizer,
)
from benchmarks.common.schedules import BudgetCosine  # noqa: E402
from benchmarks.models.vit_small_cifar.model import ViTCIFAR  # noqa: E402

LR, LAM, BETA, STEPS = 1e-2, 3e-3, 0.9, 30


def _tiny_vit() -> nn.Module:
    return ViTCIFAR(img_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2, num_classes=3,
                    drop=0.0)


NETS = {
    "all_four_layer_types": (TinyMultiLayerNet, (6, 2, 5, 5), 4),
    "vit_with_unhooked_params": (_tiny_vit, (6, 3, 8, 8), 3),
}


def _batches(x_shape, n_classes, steps=STEPS):
    g = torch.Generator().manual_seed(1)
    return [(torch.randn(*x_shape, generator=g), torch.randint(0, n_classes, (x_shape[0],),
                                                                  generator=g))
            for _ in range(steps)]


def _train(model, optimizer, batches) -> None:
    for x, y in batches:
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()


@pytest.mark.filterwarnings("ignore:Full backward hook is firing")
@pytest.mark.parametrize("net", sorted(NETS))
@pytest.mark.parametrize("decoupled, wd", [(False, 0.0), (False, 1e-2), (True, 1e-2)])
@pytest.mark.parametrize("tcov", [1, 10])
def test_sgdm_is_adafisher_with_the_curvature_set_to_lambda(net, decoupled, wd, tcov):
    build, x_shape, n_classes = NETS[net]
    torch.manual_seed(0)
    a = build()
    b = copy.deepcopy(a)
    start = [p.detach().clone() for p in a.parameters()]
    batches = _batches(x_shape, n_classes)

    reference = AdaFisherMulti(a, lr=LR, beta=BETA, Lambda=LAM, gammas=[1.0, 0.0], TCov=tcov,
                               weight_decay=wd, fisher_mode="diag",
                               decoupled_weight_decay=decoupled)
    _train(a, reference, batches)
    _train(b, LambdaLimitSGD(b, lr=LR, beta=BETA, lam=LAM, weight_decay=wd,
                             decoupled_weight_decay=decoupled), batches)

    for (name, p), q, p0 in zip(a.named_parameters(), b.parameters(), start):
        assert not torch.equal(p, p0), f"{name} never moved: the comparison would be vacuous"
        assert torch.equal(p, q), f"{name}: sgdm differs from AdaFisherMulti with F~ = lam * I"


@pytest.mark.filterwarnings("ignore:Full backward hook is firing")
def test_sgdm_is_not_the_shipped_diag():
    """Non-vacuity of the test above: at the shipped ``gammas`` the curvature is not zero, so
    ``diag`` and ``sgdm`` must differ."""
    build, x_shape, n_classes = NETS["all_four_layer_types"]
    torch.manual_seed(0)
    a = build()
    b = copy.deepcopy(a)
    batches = _batches(x_shape, n_classes)
    _train(a, AdaFisherMulti(a, lr=LR, beta=BETA, Lambda=LAM, TCov=1, fisher_mode="diag"), batches)
    _train(b, LambdaLimitSGD(b, lr=LR, beta=BETA, lam=LAM), batches)
    assert any(not torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_unhooked_parameters_take_the_fallback_step_not_the_divided_one():
    """``pos_embed`` moves by ``lr * m_hat``, exactly as in AdaFisherMulti, not by ``lr*m_hat/lam``."""
    torch.manual_seed(0)
    model = _tiny_vit()
    pos0 = model.pos_embed.detach().clone()
    head0 = model.head.weight.detach().clone()
    optimizer = LambdaLimitSGD(model, lr=LR, beta=BETA, lam=LAM)
    x, y = _batches((6, 3, 8, 8), 3, steps=1)[0]
    nn.functional.cross_entropy(model(x), y).backward()
    pos_grad = model.pos_embed.grad.clone()
    head_grad = model.head.weight.grad.clone()
    optimizer.step()
    # One step: m_1 = (1 - beta) g, bias correction 1 - beta, so m_hat = g.
    torch.testing.assert_close(model.pos_embed, pos0 - LR * pos_grad, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(model.head.weight, head0 - LR * head_grad / LAM, rtol=1e-6,
                               atol=1e-9)


def test_sgdm_is_torch_momentum_sgd_at_the_derived_learning_rate():
    """``m_t = (1 - beta) b_t`` for torch's buffer ``b_t``, so the hooked update is
    ``torch.optim.SGD(momentum=beta, dampening=0)`` at ``lr (1 - beta) / (lam (1 - beta^t))``.
    Not bit-exact (different arithmetic), so a tolerance."""
    torch.manual_seed(0)
    # float64, so that what is left is the identity's own rounding and not fp32's, amplified over
    # 30 steps at an effective learning rate of 0.33 (measured in fp32: 3.7e-6 absolute).
    a = nn.Sequential(nn.Linear(12, 8), nn.Tanh(), nn.Linear(8, 5)).double()  # all hooked
    b = copy.deepcopy(a)
    batches = [(x.double(), y) for x, y in _batches((6, 12), 5)]
    _train(a, LambdaLimitSGD(a, lr=LR, beta=BETA, lam=LAM), batches)

    sgd = torch.optim.SGD(b.parameters(), lr=0.0, momentum=BETA, dampening=0.0)
    for t, (x, y) in enumerate(batches, start=1):
        for group in sgd.param_groups:
            group["lr"] = LR * (1 - BETA) / (LAM * (1 - BETA**t))
        sgd.zero_grad()
        nn.functional.cross_entropy(b(x), y).backward()
        sgd.step()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-10, atol=1e-12)


def test_vit_bench_learning_rate_is_one_thirtieth():
    """The number quoted in E18: ``lr (1 - beta) / lam`` at the ViT benches' operating point."""
    hp = HParams(lr=1e-3, lam=3e-3, beta=0.9)
    assert hp.lr * (1 - hp.beta) / hp.lam == pytest.approx(1 / 30)


@pytest.mark.filterwarnings("ignore:Full backward hook is firing")
def test_the_diag_operator_is_within_the_bound_e18_relies_on():
    """``0 <= F~_D - lam <= X_k^2`` at the shipped settings (min-max on, ``gammas = (0.92, 0.008)``),
    with ``X_k = 0.08^k + (0.008/0.92)(1 - 0.08^k)`` after ``k`` factor updates."""
    torch.manual_seed(0)
    model = TinyMultiLayerNet()
    optimizer = AdaFisherMulti(model, lr=LR, beta=BETA, Lambda=LAM, TCov=1, fisher_mode="diag")
    x_shape, n_classes = NETS["all_four_layer_types"][1:]
    for k, (x, y) in enumerate(_batches(x_shape, n_classes, steps=8), start=1):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        # The hooks have now folded their k-th observation into every running average.
        bound = (0.08**k + (0.008 / 0.92) * (1 - 0.08**k)) ** 2
        for module in optimizer.modules:
            excess = optimizer.approx.f_tilde(module) - LAM
            assert excess.min() >= -1e-9
            assert excess.max() <= bound * (1 + 1e-5), (k, module, excess.max().item(), bound)
        optimizer.step()
    assert bound == pytest.approx((0.008 / 0.92) ** 2, rel=1e-5)
    assert LAM / (LAM + bound) > 0.975


def test_sgdm_is_registered_and_built_from_the_fisher_arms_hparams():
    assert "sgdm" in ARMS
    hp = HParams(lr=1e-3, baseline_lr=1e-4, lam=3e-3, weight_decay=1e-2, decoupled_wd=True)
    optimizer = build_optimizer("sgdm", nn.Linear(3, 2), hp)
    assert isinstance(optimizer, LambdaLimitSGD)
    assert optimizer.param_groups[0]["lr"] == 1e-3  # the Fisher arms' lr, not baseline_lr
    assert optimizer.lam == 3e-3 and optimizer.decoupled_weight_decay
    with pytest.raises(ValueError, match="takes no AdaFisherMulti overrides"):
        build_optimizer("sgdm", nn.Linear(3, 2), hp, fisher_mode="kfac")


def test_the_budget_cosine_drives_its_learning_rate():
    """The campaign protocol anneals each budgeted arm with ``BudgetCosine``, which writes the
    group's ``lr``; the arm must read it from there at every step."""
    model = nn.Linear(3, 2)
    optimizer = LambdaLimitSGD(model, lr=1e-3, beta=0.9, lam=3e-3)
    schedule = BudgetCosine(optimizer, budget_s=100.0)
    schedule(50.0)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5e-3)
    w0 = model.weight.detach().clone()
    model(torch.ones(1, 3)).sum().backward()
    g = model.weight.grad.clone()
    optimizer.step()
    torch.testing.assert_close(model.weight, w0 - 0.5e-3 * g / 3e-3, rtol=1e-5, atol=1e-7)
