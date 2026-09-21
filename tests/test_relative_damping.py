"""``damping``, ``damping_tau`` and ``hold_cap``: the per-layer safety constant of fix S1, and the
step-size cap held while it moves.

These are the checks ``docs/reports/plan_lambda_dominance.md`` (E15) makes a precondition of
submitting any run:

* off by default, and bit-identical when off;
* ``hold_cap=True, lr=c`` is the update ``hold_cap=False, lr=c*Lambda`` already gives, which is how
  E7-E14 held the cap -- so the new arms and the old single-``lambda`` arm share one convention;
* a per-layer ``lambda_l`` that happens to be constant reproduces the single-``lambda`` path, so the
  only thing a relative damping changes is the value of ``lambda``;
* ``mean_curvature`` is the mean eigenvalue of the undamped operator, computed densely;
* under ``hold_cap`` the decoupled weight decay does not move with ``lambda`` (Part 5, rule 3).

Everything runs in float64, so the equivalences can be held to a tight tolerance.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from adafisher_modes import AdaFisherMulti
from conftest import TinyMultiLayerNet, seed_all

KRON_MODES = ["kfac", "ekfac", "tkfac", "tekfac"]
ALL_MODES = ["diag", *KRON_MODES]
MODE_KWARGS = {
    "diag": {},
    "kfac": {"T_inv": 2},
    "ekfac": {"T_eig": 2},
    "tkfac": {"T_inv": 2},
    "tekfac": {"T_eig": 2},
}
DTYPE = torch.float64


class _Net(TinyMultiLayerNet):
    """``TinyMultiLayerNet`` (all four hooked layer types) plus one parameter no hooked module owns,
    so the plain-momentum fallback is exercised too."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) * self.gain


def _run(mode: str, *, steps: int = 6, lr: float = 1e-3, Lambda: float = 1e-3, patch=None,
         zero_loss: bool = False, freeze_gain: bool = False, **kwargs):
    """A short float64 run. Returns ``(model, optimizer, displacement per parameter)``."""
    seed_all(0)
    model = _Net().to(DTYPE)
    if freeze_gain:
        model.gain.requires_grad_(False)
    opt = AdaFisherMulti(model, lr=lr, Lambda=Lambda, TCov=2, fisher_mode=mode,
                         **MODE_KWARGS[mode], **kwargs)
    if patch is not None:
        patch(opt)
    theta0 = [p.detach().clone() for p in model.parameters()]
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randn(6, 2, 5, 5, generator=gen, dtype=DTYPE)
        y = torch.randint(0, 4, (6,), generator=gen)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        (loss * 0.0 if zero_loss else loss).backward()
        opt.step()
    return model, opt, [p.detach() - t0 for p, t0 in zip(model.parameters(), theta0)]


def _assert_close(a, b, rtol: float = 1e-9) -> None:
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=rtol, atol=1e-15)


# --------------------------------------------------------------------------------------------
# Off by default
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ALL_MODES)
def test_explicit_defaults_are_bit_identical_to_not_passing_them(mode: str) -> None:
    model_a, _, _ = _run(mode)
    model_b, _, _ = _run(mode, damping="global", damping_tau=None, hold_cap=False)
    for p, q in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p, q)


# --------------------------------------------------------------------------------------------
# hold_cap is the E-protocol's convention, moved inside the optimizer
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ALL_MODES)
def test_hold_cap_is_lr_times_lambda(mode: str) -> None:
    """``hold_cap=True, lr=c`` against ``lr=c*Lambda``: the update E7-E14 used to hold the cap."""
    cap, lam = 0.5, 1e-4
    _, _, held = _run(mode, lr=cap, Lambda=lam, hold_cap=True)
    _, _, scaled = _run(mode, lr=cap * lam, Lambda=lam)
    assert any(d.abs().max() > 0 for d in held)
    _assert_close(held, scaled)


@pytest.mark.parametrize("mode", ["kfac", "ekfac"])
def test_hold_cap_makes_decoupled_decay_independent_of_lambda(mode: str) -> None:
    """With every gradient zero, only the decoupled decay moves the weights. Under ``hold_cap`` it
    is ``1 - lr*wd`` whatever ``Lambda`` is -- Part 5's rule 3, by construction."""
    kw = dict(lr=0.5, hold_cap=True, weight_decay=1e-2, decoupled_weight_decay=True,
              zero_loss=True)
    model_a, _, move_a = _run(mode, Lambda=1e-3, **kw)
    model_b, _, move_b = _run(mode, Lambda=1e-9, **kw)
    assert any(d.abs().max() > 0 for d in move_a)
    for p, q in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(p, q)


# --------------------------------------------------------------------------------------------
# A relative damping changes lambda and nothing else
# --------------------------------------------------------------------------------------------


def _constant_curvature(opt: AdaFisherMulti) -> None:
    opt.approx.mean_curvature = lambda module: torch.ones((), dtype=DTYPE)  # type: ignore[method-assign]


@pytest.mark.parametrize("hold_cap", [False, True], ids=["plain", "hold_cap"])
@pytest.mark.parametrize("damping", ["layer_relative", "network_relative"])
@pytest.mark.parametrize("mode", KRON_MODES)
def test_constant_per_layer_lambda_reproduces_the_single_lambda_path(
    mode: str, damping: str, hold_cap: bool
) -> None:
    """Force every module's mean curvature to 1 and set ``tau = 1e-5``, while the constructor's
    shared ``Lambda`` stays at 1e-3: each module then gets ``lambda_l = 1e-5``, and the trajectory
    must be the single-``lambda`` one *at 1e-5*. A mode that still read the shared ``Lambda``
    anywhere -- in its damped factors, in the inverses it caches, or in the value ``hold_cap``
    scales by -- would land on 1e-3 there and fail. (Under ``hold_cap`` the plain-momentum fallback
    scales by the shared ``Lambda`` by design, so the one parameter that uses it is frozen here.)"""
    lr = 0.5 if hold_cap else 1e-3
    _, _, relative = _run(mode, Lambda=1e-3, lr=lr, hold_cap=hold_cap, freeze_gain=True,
                          damping=damping, damping_tau=1e-5, patch=_constant_curvature)
    _, _, single = _run(mode, Lambda=1e-5, lr=lr, hold_cap=hold_cap, freeze_gain=True)
    _assert_close(relative, single)


@pytest.mark.parametrize("mode", KRON_MODES)
def test_layer_relative_gives_each_layer_tau_times_its_own_mean(mode: str) -> None:
    tau = 0.05
    model, opt, _ = _run(mode, damping="layer_relative", damping_tau=tau)
    lams = []
    for module in opt.modules:
        c = opt.approx.mean_curvature(module)
        # rtol only: these values are ~1e-7, so the float64 default atol of 1e-7 would pass anything
        torch.testing.assert_close(opt.approx.lambda_for(module), tau * c, rtol=1e-12, atol=0.0)
        lams.append(float(opt.approx.lambda_for(module)))
    assert len(set(lams)) > 1, "every layer got the same lambda: the test would be vacuous"


@pytest.mark.parametrize("mode", KRON_MODES)
def test_network_relative_gives_every_layer_one_direction_weighted_mean(mode: str) -> None:
    tau = 0.05
    _, opt, _ = _run(mode, damping="network_relative", damping_tau=tau)
    cs = [opt.approx.mean_curvature(m) for m in opt.modules]
    ns = [opt.approx.num_directions(m) for m in opt.modules]
    expected = tau * sum(c * n for c, n in zip(cs, ns)) / sum(ns)
    for module in opt.modules:
        torch.testing.assert_close(opt.approx.lambda_for(module), expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("damping", ["layer_relative", "network_relative"])
@pytest.mark.parametrize("mode", KRON_MODES)
def test_a_relative_damping_does_change_the_trajectory(mode: str, damping: str) -> None:
    _, _, relative = _run(mode, lr=0.5, hold_cap=True, damping=damping, damping_tau=0.05)
    _, _, single = _run(mode, lr=0.5, hold_cap=True)
    assert max((a - b).abs().max().item() for a, b in zip(relative, single)) > 1e-6


# --------------------------------------------------------------------------------------------
# mean_curvature is what it says
# --------------------------------------------------------------------------------------------


def _dense_undamped(opt: AdaFisherMulti, mode: str, module: nn.Module) -> torch.Tensor:
    ap = opt.approx
    if mode == "kfac":
        return torch.kron(ap._B[module], ap._A[module])
    if mode == "tkfac":
        delta = ap._delta[module]
        return delta * torch.kron(ap._Psi_raw[module] / delta, ap._Phi_raw[module] / delta)
    if mode == "ekfac":
        Q = torch.kron(ap._Q_B[module], ap._Q_A[module])
        return Q @ torch.diag(ap._s_star[module].flatten()) @ Q.t()
    Q = torch.kron(ap._Q_Psi[module], ap._Q_Phi[module])
    return Q @ torch.diag(ap._Theta[module].flatten()) @ Q.t()


@pytest.mark.parametrize("mode", KRON_MODES)
def test_mean_curvature_is_the_dense_mean_eigenvalue(mode: str) -> None:
    _, opt, _ = _run(mode, steps=5)
    for module in opt.modules:
        dense = _dense_undamped(opt, mode, module)
        assert opt.approx.num_directions(module) == dense.size(0)
        torch.testing.assert_close(
            opt.approx.mean_curvature(module), torch.linalg.eigvalsh(dense).mean(),
            rtol=1e-8, atol=1e-18,
        )


# --------------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(fisher_mode="diag", damping="layer_relative", damping_tau=0.1), "diag"),
        (dict(fisher_mode="ekfac", damping="layer_relative"), "damping_tau > 0"),
        (dict(fisher_mode="ekfac", damping="network_relative", damping_tau=0.0), "damping_tau > 0"),
        (dict(fisher_mode="ekfac", damping_tau=0.1), "relative damping"),
        (dict(fisher_mode="ekfac", damping="per_layer", damping_tau=0.1), "damping must be"),
    ],
)
def test_invalid_configurations_are_refused(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AdaFisherMulti(_Net(), **kwargs)
