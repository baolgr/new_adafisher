"""``norm_exact_rescaling``: a normalisation layer's eigen-rescaling from its exact per-row gradient.

For ``BatchNorm2d``/``LayerNorm`` the input factor is built from ``h_bar_t = [z_t, 1]``, with
``z_t`` the raw input's channel mean (``CLAUDE.md`` §4.6). By default ``ekfac``/``tekfac`` then
estimate their rescaling from the per-row product ``delta_t (x) h_bar_t = [z_t delta_t, delta_t]``,
which is not the layer's gradient: the scale parameter's per-row gradient is ``delta_t * x_hat_t``.
Measured on ``vit_micro_cifar``: the default statistic under-states the scale column 800-8 000x.
The knob estimates the rescaling from the exact per-row gradient ``[delta_t * x_hat_t, delta_t]``,
projected into the same eigenbasis -- EKFAC's Lemma 1 makes that the optimal diagonal there.

What is pinned here: the recomputed ``x_hat`` is exactly the layer's own normalised output, for
``LayerNorm`` on token inputs and ``BatchNorm2d`` in training and in evaluation; the per-row
gradients sum to the real ``weight.grad``/``bias.grad``, in the optimizer's column order; the
stored ``s*`` and ``Theta`` equal the mean of the squared projected per-row gradients computed
independently; the knob changes nothing on a layer that is not a normalisation layer, nothing on a
network without one, and nothing when off; and the meaningless combinations are refused.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from adafisher_modes import AdaFisherMulti
from adafisher_modes.factors import (
    flatten_norm_output_grad,
    norm_exact_kfe_squares,
    normalized_norm_input,
)
from conftest import TinyMultiLayerNet, seed_all

MODES = ["ekfac", "tekfac"]
# Stored rescaling, input-side basis, output-side basis, per mode.
ATTRS = {"ekfac": ("_s_star", "_Q_A", "_Q_B"), "tekfac": ("_Theta", "_Q_Phi", "_Q_Psi")}


def _affine_identity(layer: nn.Module) -> nn.Module:
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.zero_()
    return layer


@pytest.mark.parametrize("case", ["layernorm_tokens", "batchnorm_train", "batchnorm_eval"])
def test_x_hat_is_the_layers_own_normalised_output(case: str) -> None:
    """With scale 1 and shift 0 the layer outputs x_hat itself, so the recomputation must match it
    exactly (up to one rounding) -- in train mode with the batch's statistics, in eval mode with
    the running ones."""
    g = torch.Generator().manual_seed(0)
    if case == "layernorm_tokens":
        layer = _affine_identity(nn.LayerNorm(8, eps=1e-5).double())
        h = 3.0 + 2.0 * torch.randn(4, 5, 8, generator=g, dtype=torch.float64)
    else:
        layer = _affine_identity(nn.BatchNorm2d(6).double())
        h = 1.5 + torch.randn(4, 6, 3, 3, generator=g, dtype=torch.float64)
        if case == "batchnorm_eval":
            with torch.no_grad():
                layer.running_mean.copy_(torch.randn(6, generator=g, dtype=torch.float64))
                layer.running_var.copy_(torch.rand(6, generator=g, dtype=torch.float64) + 0.5)
            layer.eval()
    expected = flatten_norm_output_grad(layer(h).detach(), layer)
    torch.testing.assert_close(normalized_norm_input(h, layer), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", ["layernorm", "batchnorm"])
def test_per_row_gradients_sum_to_the_real_gradients(kind: str) -> None:
    """``[delta_t * x_hat_t, delta_t]`` summed over rows is ``[weight.grad, bias.grad]``: the
    column order is the optimizer's (scale in column 0, shift in column 1) and nothing is lost."""
    g = torch.Generator().manual_seed(1)
    if kind == "layernorm":
        layer = nn.LayerNorm(8).double()
        h = torch.randn(4, 5, 8, generator=g, dtype=torch.float64)
    else:
        layer = nn.BatchNorm2d(6).double()
        h = torch.randn(4, 6, 3, 3, generator=g, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(layer.weight.shape, generator=g, dtype=torch.float64))
        layer.bias.copy_(torch.randn(layer.bias.shape, generator=g, dtype=torch.float64))
    y = layer(h)
    y.retain_grad()
    r = torch.randn(y.shape, generator=g, dtype=torch.float64)
    (y * r).sum(dim=tuple(range(1, y.dim()))).mean().backward()
    delta = flatten_norm_output_grad(y.grad, layer)
    x_hat = normalized_norm_input(h.detach(), layer)
    torch.testing.assert_close((delta * x_hat).sum(0), layer.weight.grad, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(delta.sum(0), layer.bias.grad, rtol=1e-12, atol=1e-14)


def test_kfe_squares_match_a_row_by_row_projection() -> None:
    g = torch.Generator().manual_seed(2)
    T, C = 11, 5
    x_hat = torch.randn(T, C, generator=g, dtype=torch.float64)
    delta = torch.randn(T, C, generator=g, dtype=torch.float64)
    Q_out = torch.linalg.qr(torch.randn(C, C, generator=g, dtype=torch.float64))[0]
    Q_in = torch.linalg.qr(torch.randn(2, 2, generator=g, dtype=torch.float64))[0]
    brute = torch.zeros(C, 2, dtype=torch.float64)
    for t in range(T):
        G_t = torch.stack([delta[t] * x_hat[t], delta[t]], dim=1)      # (C, 2): scale, shift
        brute += (Q_out.t() @ G_t @ Q_in).pow(2)
    torch.testing.assert_close(norm_exact_kfe_squares(x_hat, delta, Q_out, Q_in), brute / T,
                               rtol=1e-12, atol=0)


def _one_step(mode: str, knob: bool, model: nn.Module, x: torch.Tensor, capture: dict):
    """One optimizer step at which every module's rescaling is observed for the first time and,
    under ema_seed_first, is exactly that step's statistic. Returns the optimizer."""
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, T_eig=1, ema_seed_first=True,
                         eig_before_rescale=True, norm_exact_rescaling=knob)
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            def fwd(mod, inp, out, name=name):
                capture[name] = {"h": inp[0].detach()}
                out.register_hook(lambda grad, name=name: capture[name].update(delta=grad))
            handles.append(module.register_forward_hook(fwd))
    model(x).pow(2).mean().backward()
    opt.step()
    for h in handles:
        h.remove()
    return opt


@pytest.mark.parametrize("mode", MODES)
def test_stored_rescaling_is_the_exact_statistic_on_every_norm_layer(mode: str) -> None:
    """On a real network (both normalisation types), after one step the stored rescaling of every
    normalisation layer is the mean of the squared exact per-row gradients projected into the basis
    the optimizer measured it in -- recomputed here from autograd, not from the optimizer."""
    seed_all(0)
    model = TinyMultiLayerNet().double()
    x = torch.randn(6, 2, 5, 5, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    capture: dict = {}
    opt = _one_step(mode, True, model, x, capture)
    s_name, qin_name, qout_name = ATTRS[mode]
    checked = 0
    for name, module in model.named_modules():
        if not isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            continue
        c = capture[name]
        delta = flatten_norm_output_grad(c["delta"], module)
        # The layer was in train mode during the forward: x_hat from the batch's statistics.
        x_hat = normalized_norm_input(c["h"], module)
        Q_in, Q_out = getattr(opt.approx, qin_name)[module], getattr(opt.approx, qout_name)[module]
        brute = torch.zeros_like(getattr(opt.approx, s_name)[module])
        for t in range(delta.size(0)):
            G_t = torch.stack([delta[t] * x_hat[t], delta[t]], dim=1)
            brute += (Q_out.t() @ G_t @ Q_in).pow(2)
        torch.testing.assert_close(getattr(opt.approx, s_name)[module], brute / delta.size(0),
                                   rtol=1e-10, atol=0)
        checked += 1
    assert checked == 2       # the BatchNorm2d and the LayerNorm


@pytest.mark.parametrize("mode", MODES)
def test_only_normalisation_layers_change(mode: str) -> None:
    """At the same step, from the same state, the knob changes the rescaling of the normalisation
    layers (non-vacuously) and leaves every other layer's bit-identical."""
    s_name = ATTRS[mode][0]
    stores = {}
    for knob in (False, True):
        seed_all(0)
        model = TinyMultiLayerNet()
        x = torch.randn(6, 2, 5, 5, generator=torch.Generator().manual_seed(3))
        opt = _one_step(mode, knob, model, x, {})
        stores[knob] = {n: getattr(opt.approx, s_name)[m].clone()
                        for n, m in model.named_modules() if m in getattr(opt.approx, s_name)}
    for name in stores[False]:
        same = torch.equal(stores[False][name], stores[True][name])
        assert same == (name not in ("bn", "ln")), name


@pytest.mark.parametrize("mode", MODES)
def test_inert_on_a_network_without_normalisation_layers(mode: str) -> None:
    def run(knob: bool) -> list:
        seed_all(0)
        model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1), nn.ReLU(), nn.Flatten(),
                              nn.Linear(75, 4))
        opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=2, T_eig=2,
                             eig_before_rescale=True, norm_exact_rescaling=knob)
        g = torch.Generator().manual_seed(1)
        for _ in range(8):
            opt.zero_grad()
            model(torch.randn(6, 2, 5, 5, generator=g)).pow(2).mean().backward()
            opt.step()
        return [p.detach().clone() for p in model.parameters()]
    for a, b in zip(run(False), run(True)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("mode", MODES)
def test_default_is_the_shipped_statistic(mode: str) -> None:
    def run(**kw) -> list:
        seed_all(0)
        model = TinyMultiLayerNet()
        opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=2, T_eig=2, **kw)
        g = torch.Generator().manual_seed(1)
        for _ in range(8):
            opt.zero_grad()
            model(torch.randn(6, 2, 5, 5, generator=g)).pow(2).mean().backward()
            opt.step()
        return [p.detach().clone() for p in model.parameters()]
    for a, b in zip(run(), run(norm_exact_rescaling=False)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("mode", MODES)
def test_runs_with_every_rescale_form(mode: str) -> None:
    for extra in ({}, {"rescale_form": "floor", "Lambda": 1e-9},
                  {"rescale_form": "clip", "clip_fraction": 0.5, "clip_threshold": "ema"}):
        seed_all(0)
        model = TinyMultiLayerNet()
        opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=2, T_eig=2,
                             eig_before_rescale=True, norm_exact_rescaling=True, **extra)
        g = torch.Generator().manual_seed(1)
        for _ in range(10):
            opt.zero_grad()
            model(torch.randn(6, 2, 5, 5, generator=g)).pow(2).mean().backward()
            opt.step()
        assert all(torch.isfinite(p).all() for p in model.parameters()), extra


@pytest.mark.parametrize("mode", ["diag", "kfac", "tkfac"])
def test_other_modes_refuse_the_knob(mode: str) -> None:
    with pytest.raises(ValueError, match="ekfac"):
        AdaFisherMulti(TinyMultiLayerNet(), fisher_mode=mode, norm_exact_rescaling=True)


def test_refused_with_fisher_batch_samples_on_a_batchnorm_network() -> None:
    with pytest.raises(ValueError, match="fisher_batch_samples"):
        AdaFisherMulti(TinyMultiLayerNet(), fisher_mode="ekfac", norm_exact_rescaling=True,
                       fisher_batch_samples=4)
    # Without a BatchNorm2d the combination is fine: LayerNorm's statistics are per row.
    AdaFisherMulti(nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)), fisher_mode="ekfac",
                   norm_exact_rescaling=True, fisher_batch_samples=4)
