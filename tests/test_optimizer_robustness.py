"""What ``AdaFisherMulti`` does with the awkward networks and loops, rather than the tidy ones.

Every case here was a silent wrong answer or an unreadable crash before it got a test. Each test
says which, and what the behaviour is now:

1. **Two parameter groups.** The standard decay/no-decay split gave every hooked module two steps
   per iteration, one from its weight's group and one from its bias's, so both parameters moved
   exactly twice as far as intended. A module is now stepped once per ``step()``, whatever the
   grouping.
2. **A normalisation layer with a scale and no shift** (``LayerNorm(d, bias=False)``) is refused
   when the optimizer is built, instead of failing later on a tensor shape.
3. **A trainable weight with a frozen bias** is refused with a message naming the module, instead
   of a bare assertion in ``diag`` and a matrix-shape error in the four Kronecker modes.
4. **A module first reached after step 0** -- behind a conditional branch, or dropped by stochastic
   depth -- now starts its running average on that first observation. It used to raise ``KeyError``.
5. **Two backward passes over one forward** are refused by the three modes that pair each example's
   input with its own gradient. ``diag`` and ``kfac`` cache nothing and cannot detect it; that gap
   is pinned here too, so it stays visible.
6. **A weight shared by two modules.** The behaviour is pinned, not endorsed: it is surprising, and
   a future change to it should be a decision rather than an accident.
7. **The preconditioned direction is a new tensor**, never a view of the momentum buffer the
   optimizer is about to reuse.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from adafisher_modes import AdaFisherMulti
from adafisher_modes.optimizer import SUPPORTED_MODULES
from conftest import TinyMultiLayerNet, seed_all

ALL_MODES = ["diag", "kfac", "ekfac", "tkfac", "tekfac"]
# Cadence kwargs differ per mode (T_inv for kfac/tkfac, T_eig for ekfac/tekfac, T_re for tekfac).
CADENCE = {
    "diag": {},
    "kfac": {"T_inv": 1},
    "ekfac": {"T_eig": 1},
    "tkfac": {"T_inv": 1},
    "tekfac": {"T_eig": 1, "T_re": 1},
}


def _small_net() -> nn.Module:
    return nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Linear(4, 3))


# --------------------------------------------------------------------------------------------
# 1. Two parameter groups: one step per module, not one per group.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ALL_MODES)
def test_two_parameter_groups_step_each_module_once(mode: str) -> None:
    """The decay/no-decay split is the standard way to build two groups, and it puts a module's
    weight in one and its bias in the other. Both parameters must end up where a single group would
    have put them. Before the fix they moved exactly twice as far.
    """

    def run(split: bool) -> list[torch.Tensor]:
        seed_all(0)
        model = _small_net()
        opt = AdaFisherMulti(model, lr=1e-2, fisher_mode=mode, TCov=1, **CADENCE[mode])
        if split:
            weights = [p for n, p in model.named_parameters() if n.endswith("weight")]
            biases = [p for n, p in model.named_parameters() if n.endswith("bias")]
            opt.param_groups.clear()
            for params in (weights, biases):
                opt.add_param_group({"params": params, "lr": 1e-2, "beta": 0.9, "weight_decay": 0.0})
        gen = torch.Generator().manual_seed(1)
        for _ in range(3):
            x = torch.randn(12, 6, generator=gen)
            opt.zero_grad()
            model(x).pow(2).sum().backward()
            opt.step()
        return [p.detach().clone() for p in model.parameters()]

    one_group, two_groups = run(split=False), run(split=True)
    for a, b in zip(one_group, two_groups):
        assert torch.equal(a, b), (
            f"{mode}: splitting the parameters into two groups changed the step. A module is "
            f"preconditioned once, with its weight and bias together, so it must be stepped once "
            f"however the parameters are grouped."
        )


def test_two_parameter_groups_do_not_double_the_displacement() -> None:
    """The size of the old defect, as a number: with the split, every parameter moved exactly
    2.0000 times as far from its initial value as it should have.
    """
    seed_all(0)
    model = _small_net()
    start = [p.detach().clone() for p in model.parameters()]
    opt = AdaFisherMulti(model, lr=1e-1, fisher_mode="diag", TCov=1)
    weights = [p for n, p in model.named_parameters() if n.endswith("weight")]
    biases = [p for n, p in model.named_parameters() if n.endswith("bias")]
    opt.param_groups.clear()
    for params in (weights, biases):
        opt.add_param_group({"params": params, "lr": 1e-1, "beta": 0.9, "weight_decay": 0.0})

    x = torch.randn(8, 6)
    opt.zero_grad()
    model(x).pow(2).sum().backward()
    opt.step()
    split_shift = sum(((p.detach() - q) ** 2).sum() for p, q in zip(model.parameters(), start)).sqrt()

    seed_all(0)
    ref = _small_net()
    ref_start = [p.detach().clone() for p in ref.parameters()]
    ref_opt = AdaFisherMulti(ref, lr=1e-1, fisher_mode="diag", TCov=1)
    ref_opt.zero_grad()
    ref(x).pow(2).sum().backward()
    ref_opt.step()
    ref_shift = sum(((p.detach() - q) ** 2).sum() for p, q in zip(ref.parameters(), ref_start)).sqrt()

    assert torch.allclose(split_shift, ref_shift, rtol=1e-6), (
        f"two groups moved the parameters {float(split_shift / ref_shift):.4f} times as far as one "
        f"group, where it must be 1.0000; before the fix this ratio was exactly 2.0000"
    )


# --------------------------------------------------------------------------------------------
# 2. A normalisation layer with a scale but no shift.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ALL_MODES)
def test_layernorm_without_bias_is_refused_when_the_optimizer_is_built(mode: str) -> None:
    """``LayerNorm(d, bias=False)`` keeps its scale and drops its shift. Every input factor in this
    package is two columns wide for a normalisation layer, one per parameter, so that layer cannot
    be described. It used to surface as ``RuntimeError: shape '[8]' is invalid for input of size
    16`` in ``diag`` and ``mat1 and mat2 shapes cannot be multiplied (8x1 and 2x2)`` in the other
    four -- both several calls away from the layer that caused them.
    """
    model = nn.Sequential(nn.Linear(6, 8), nn.LayerNorm(8, bias=False), nn.Linear(8, 3))
    with pytest.raises(NotImplementedError, match="scale parameter but no shift parameter"):
        AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, **CADENCE[mode])


def test_layernorm_without_affine_is_still_skipped_silently() -> None:
    """The neighbouring case must keep working: ``elementwise_affine=False`` has no parameters at
    all, so there is nothing to precondition and the layer is simply not hooked.
    """
    seed_all(0)
    model = nn.Sequential(nn.Linear(6, 8), nn.LayerNorm(8, elementwise_affine=False), nn.Linear(8, 3))
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode="kfac", TCov=1, T_inv=1)
    assert all(not isinstance(m, nn.LayerNorm) for m in opt.modules)
    opt.zero_grad()
    model(torch.randn(8, 6)).pow(2).sum().backward()
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


# --------------------------------------------------------------------------------------------
# 3. A trainable weight with a frozen bias.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ALL_MODES)
def test_frozen_bias_on_a_trained_weight_is_refused_by_name(mode: str) -> None:
    seed_all(0)
    model = _small_net()
    model[0].bias.requires_grad_(False)
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, **CADENCE[mode])
    opt.zero_grad()
    model(torch.randn(8, 6)).pow(2).sum().backward()
    with pytest.raises(RuntimeError, match="weight gradient but no bias gradient"):
        opt.step()


@pytest.mark.parametrize("mode", ALL_MODES)
def test_a_wholly_frozen_module_still_trains_the_rest(mode: str) -> None:
    """The neighbouring case, which must keep working: freezing *both* of a module's parameters is
    a normal thing to do, and the rest of the network must still step.
    """
    seed_all(0)
    model = _small_net()
    model[0].weight.requires_grad_(False)
    model[0].bias.requires_grad_(False)
    frozen = model[0].weight.detach().clone()
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, **CADENCE[mode])
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(8, 6)).pow(2).sum().backward()
        opt.step()
    assert torch.equal(model[0].weight, frozen)
    assert not torch.equal(model[2].weight, torch.zeros_like(model[2].weight))
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize("mode", ALL_MODES)
def test_frozen_weight_with_a_trained_bias_falls_back_to_momentum(mode: str) -> None:
    """The mirror image of the case above, and the one that must *not* raise: with the weight
    frozen there is no direction to precondition the module with, so its bias takes a plain
    momentum step -- the same fallback a parameter belonging to no hooked module gets. This branch
    had no test, so nothing would have noticed if the new refusal had been placed one line higher.
    """
    seed_all(0)
    model = _small_net()
    model[0].weight.requires_grad_(False)
    frozen = model[0].weight.detach().clone()
    bias_before = model[0].bias.detach().clone()
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, **CADENCE[mode])
    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(8, 6)).pow(2).sum().backward()
        opt.step()
    assert torch.equal(model[0].weight, frozen), "a frozen weight must not move"
    assert not torch.equal(model[0].bias, bias_before), "its trained bias must still step"
    assert all(torch.isfinite(p).all() for p in model.parameters())


# --------------------------------------------------------------------------------------------
# 4. A module first reached after step 0.
# --------------------------------------------------------------------------------------------


class _ConditionalNet(nn.Module):
    """``late`` is skipped for the first few iterations, the way stochastic depth or a warm-up
    branch skips a block. Its first observation therefore arrives at a step that is not 0.
    """

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(6, 4)
        self.late = nn.Linear(4, 4)
        self.out = nn.Linear(4, 3)
        self.use_late = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.first(x))
        if self.use_late:
            x = torch.relu(self.late(x))
        return self.out(x)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_module_first_reached_after_step_zero(mode: str) -> None:
    """It used to raise ``KeyError: Linear(in_features=4, out_features=4, bias=True)`` out of the
    running average, in all five modes, because the start-up branch was keyed on ``step == 0``.
    """
    seed_all(0)
    model = _ConditionalNet()
    late_before = model.late.weight.detach().clone()
    # lr/Lambda as in the repository's other optimizer smoke tests: at lr=1e-2 with the default
    # Lambda=1e-3, diag's min-max puts F~_D in [Lambda, 1+Lambda] and its first steps are ~1/Lambda
    # times the gradient, which diverges on this net for reasons unrelated to what is under test.
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, **CADENCE[mode])
    for i in range(5):
        model.use_late = i >= 2
        opt.zero_grad()
        model(torch.randn(8, 6)).pow(2).sum().backward()
        opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not torch.equal(model.late.weight, late_before), (
        f"{mode}: the late module never moved, so its preconditioner was a silent no-op"
    )


@pytest.mark.parametrize("mode", ["kfac", "ekfac", "tkfac", "tekfac"])
def test_late_module_works_when_the_refresh_cadence_misses_its_first_step(mode: str) -> None:
    """The amortised inverse or eigenbasis is rebuilt on multiples of ``T_inv``/``T_eig``. Step 0
    is a multiple of every cadence, so a module present from the start always has one. A module
    that arrives at step 3 with a cadence of 2 does not, and ``precondition`` would read a cache
    that was never written.
    """
    cadence = {"kfac": {"T_inv": 2}, "tkfac": {"T_inv": 2},
               "ekfac": {"T_eig": 2}, "tekfac": {"T_eig": 2, "T_re": 1}}[mode]
    seed_all(0)
    model = _ConditionalNet()
    late_before = model.late.weight.detach().clone()
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, **cadence)
    for i in range(6):
        model.use_late = i >= 3  # first seen at step 3, which is not a multiple of 2
        opt.zero_grad()
        model(torch.randn(8, 6)).pow(2).sum().backward()
        opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not torch.equal(model.late.weight, late_before)


# --------------------------------------------------------------------------------------------
# 5. Two backward passes over one forward.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["ekfac", "tkfac", "tekfac"])
def test_two_backwards_over_one_forward_are_refused(mode: str) -> None:
    """These three modes pair each example's input with that same example's output gradient. A
    second backward has no input to pair with: the output factor would get two observations and the
    input factor one. It used to be a bare ``KeyError`` in ``tkfac`` and ``tekfac``, and a silent
    skip of the rescaling update in ``ekfac``.
    """
    seed_all(0)
    model = _small_net()
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, **CADENCE[mode])
    assert opt.modules, "building the optimizer is what registers the hooks under test"
    out = model(torch.randn(8, 6))
    out.pow(2).sum().backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="No cached layer input"):
        out.pow(2).sum().backward()


@pytest.mark.parametrize("mode", ["diag", "kfac"])
def test_two_backwards_are_not_detected_by_the_two_cacheless_modes(mode: str) -> None:
    """Pinned so the gap stays visible. ``diag`` and ``kfac`` reduce each hook's statistic on the
    spot and hold no per-example cache, so there is nothing in them that could notice a second
    backward: it simply folds the output gradient into the running average a second time. If a
    future change gives them a cache, this test is the one that should be revisited.
    """
    seed_all(0)
    model = _small_net()
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, **CADENCE[mode])
    out = model(torch.randn(8, 6))
    out.pow(2).sum().backward(retain_graph=True)
    out.pow(2).sum().backward()  # accepted, silently
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize("mode", ALL_MODES)
def test_gradient_accumulation_is_not_refused(mode: str) -> None:
    """The neighbouring pattern, which must keep working: several forward/backward pairs before one
    ``step()``. Each backward has its own forward, so every input is paired.
    """
    seed_all(0)
    model = _small_net()
    before = [p.detach().clone() for p in model.parameters()]
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, **CADENCE[mode])
    for _ in range(3):
        opt.zero_grad()
        for _ in range(2):  # two micro-batches, one step
            model(torch.randn(8, 6)).pow(2).sum().backward()
        opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


# --------------------------------------------------------------------------------------------
# 6. A weight shared by two modules. Pinned, not endorsed.
# --------------------------------------------------------------------------------------------


class _TiedNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f1 = nn.Linear(6, 6)
        self.f2 = nn.Linear(6, 6)
        self.f2.weight = self.f1.weight  # one Parameter object, two modules
        self.out = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.f2(torch.relu(self.f1(x)))))


def test_tied_weight_is_owned_by_the_last_module_and_stepped_by_both() -> None:
    """Three facts, pinned exactly as they are today.

    ``_owner`` is keyed by the identity of the parameter, so a weight shared by two modules is
    owned by whichever of them ``model.modules()`` reaches last. Both modules are still hooked and
    both accumulate factors, and both are still reached through their own (unshared) bias, so the
    shared weight is preconditioned **twice** in one ``step()`` -- once with each module's
    curvature. There is no reading of AdaFisher under which that is the intended update; it is
    recorded here so that changing it is a decision rather than an accident.
    """
    seed_all(0)
    model = _TiedNet()
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode="kfac", TCov=1, T_inv=1)

    assert opt._owner[id(model.f1.weight)] is model.f2, "the later module should own the weight"
    assert model.f1 in opt.modules and model.f2 in opt.modules

    stepped: list[tuple[nn.Module, torch.nn.Parameter]] = []
    original = AdaFisherMulti._step_module

    def spy(self, hparams, module, weight_param, bias_param):  # noqa: ANN001
        stepped.append((module, weight_param))
        return original(self, hparams, module, weight_param, bias_param)

    AdaFisherMulti._step_module = spy  # type: ignore[method-assign]
    try:
        opt.zero_grad()
        model(torch.randn(8, 6)).pow(2).sum().backward()
        opt.step()
    finally:
        AdaFisherMulti._step_module = original  # type: ignore[method-assign]

    shared = [mod for mod, w in stepped if w is model.f1.weight]
    assert len(stepped) == 3, f"expected one step per hooked module, got {len(stepped)}"
    assert set(shared) == {model.f1, model.f2}, (
        "the shared weight is preconditioned once per sharing module -- twice in total"
    )
    assert model.f1 in opt.approx._A and model.f2 in opt.approx._A


def test_tied_weight_without_biases_leaves_one_modules_factors_unread() -> None:
    """The same sharing without biases. ``model.parameters()`` then yields the shared weight once,
    so only its owner is ever stepped and the other module's factors are accumulated every
    ``TCov`` steps and never read. Pinned for the same reason as above.
    """
    seed_all(0)
    f1, f2 = nn.Linear(6, 6, bias=False), nn.Linear(6, 6, bias=False)
    f2.weight = f1.weight
    model = nn.Sequential(f1, nn.ReLU(), f2, nn.ReLU(), nn.Linear(6, 3, bias=False))
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode="kfac", TCov=1, T_inv=1)

    stepped: list[nn.Module] = []
    original = AdaFisherMulti._step_module

    def spy(self, hparams, module, weight_param, bias_param):  # noqa: ANN001
        stepped.append(module)
        return original(self, hparams, module, weight_param, bias_param)

    AdaFisherMulti._step_module = spy  # type: ignore[method-assign]
    try:
        opt.zero_grad()
        model(torch.randn(8, 6)).pow(2).sum().backward()
        opt.step()
    finally:
        AdaFisherMulti._step_module = original  # type: ignore[method-assign]

    assert f1 in opt.approx._A and f2 in opt.approx._A, "both modules still accumulate factors"
    assert f1 not in stepped, "the first of the two sharing modules is never stepped"
    assert f2 in stepped, "only the owner -- the later module -- is stepped"


# --------------------------------------------------------------------------------------------
# 7. The preconditioned direction must not alias the momentum buffer.
# --------------------------------------------------------------------------------------------


class _BiasFreeNet(nn.Module):
    """The case where aliasing is actually reachable: with no bias to concatenate, the augmented
    direction a Kronecker mode builds *is* the weight direction it was handed, so any in-place
    write inside ``precondition`` would land straight in the momentum buffer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
        self.fc = nn.Linear(3 * 5 * 5, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.relu(self.conv(x)).flatten(1))


@pytest.mark.parametrize("mode", ALL_MODES)
@pytest.mark.parametrize("biased", [True, False], ids=["with-bias", "no-bias"])
def test_precondition_returns_a_new_tensor_not_a_view_of_exp_avg(mode: str, biased: bool) -> None:
    """``precondition`` is handed ``state["exp_avg"]`` itself, which the optimizer reuses on every
    later step. If it ever returned a view of that buffer, ``param.add_(direction, alpha=...)``
    would still be right, but the momentum would be silently corrupted the moment anything scaled
    or modified the returned direction -- and nothing in the suite would have noticed.

    Run on two networks. The biased one covers all four supported layer types. The bias-free one is
    the case that makes the check falsifiable at all: a Kronecker mode's augmented direction is
    then the input tensor itself, with no concatenation to allocate a fresh one.
    """
    seed_all(0)
    model = TinyMultiLayerNet() if biased else _BiasFreeNet()
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, **CADENCE[mode])
    opt.zero_grad()
    model(torch.randn(6, 2, 5, 5)).pow(2).sum().backward()
    opt.step()

    checked = 0
    for module in opt.modules:
        weight_avg = opt.state[module.weight]["exp_avg"]
        bias_avg = None if module.bias is None else opt.state[module.bias]["exp_avg"]
        for param, exp_avg in [(module.weight, weight_avg)] + (
            [] if module.bias is None else [(module.bias, bias_avg)]
        ):
            assert param is not None
            result = opt.approx.precondition(module, weight_avg, bias_avg)
            saved = exp_avg.detach().clone()
            for out in (result if isinstance(result, tuple) else (result,)):
                assert out.data_ptr() != exp_avg.data_ptr(), (
                    f"{mode}: precondition returned a view of exp_avg for {module}"
                )
                out.add_(1.0)  # the destructive check: writing to the result must not reach state
            assert torch.equal(exp_avg, saved), (
                f"{mode}: writing to the preconditioned direction changed exp_avg for {module}"
            )
            checked += 1
    # TinyMultiLayerNet: Conv2d + BatchNorm2d + LayerNorm + two Linears, all with biases -> 10.
    # _BiasFreeNet: one Conv2d and one Linear, neither with a bias -> 2.
    assert checked == (10 if biased else 2), f"checked {checked} (module, parameter) pairs"
    if biased:
        assert set(SUPPORTED_MODULES) == {type(m).__name__ for m in opt.modules}
