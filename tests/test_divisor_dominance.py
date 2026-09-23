"""E21: the divisor read by :mod:`fisher_ref.divisor`, and the experiment's own registry.

The measurement this locks down is the one E21 exists to make: for every mode and every rescale form,
the number the optimizer divides each direction by, and whether the curvature or the safety constant
sets it. The test that matters most is :func:`test_divisor_reproduces_the_applied_step`, which checks
the divisor against the step ``precondition`` actually takes -- basis, ordering and value at once. A
divisor list that is the right length, positive, and indexed in the wrong order would pass everything
else and be silently wrong, which is the trap ``plan_exp_lot2.md`` §5.2 paid for twice.

Everything here is offline: no dataset, no checkpoint, no cluster.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from adafisher_modes.approximations._kron_utils import augment_direction  # noqa: E402
from adafisher_modes.optimizer import AdaFisherMulti  # noqa: E402
from conftest import TinyMultiLayerNet, seed_all  # noqa: E402

from fisher_ref import divisor  # noqa: E402

MODES = ("diag", "kfac", "ekfac", "tkfac", "tekfac")
EIGEN = divisor.EIGEN_MODES


def _train(mode: str, *, lam: float = 1e-3, steps: int = 6, tcov: int = 2,
           **kwargs) -> tuple[nn.Module, AdaFisherMulti]:
    """A short real run of ``AdaFisherMulti``: hooks, backward passes and ``step()``, so the state
    read afterwards is the state the optimizer built rather than one assembled by the test."""
    seed_all(0)
    model = TinyMultiLayerNet()
    # The cadence keyword belongs to the mode: T_inv to the two that invert factors, T_eig to the two
    # that diagonalise them, neither to diag. Passing the wrong one is a TypeError, not a no-op.
    cadence = ({"T_inv": tcov} if mode in ("kfac", "tkfac") else
               {"T_eig": tcov} if mode in EIGEN else {})
    optimizer = AdaFisherMulti(model, lr=1e-3, Lambda=lam, TCov=tcov, fisher_mode=mode,
                               **cadence, **kwargs)
    for step in range(steps):
        inputs = torch.randn(7, 2, 5, 5, generator=torch.Generator().manual_seed(100 + step))
        targets = torch.randint(0, 4, (7,), generator=torch.Generator().manual_seed(200 + step))
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(inputs), targets).backward()
        optimizer.step()
    return model, optimizer


def _basis_of(layer: divisor.LayerDivisor, approx, module) -> tuple[torch.Tensor, torch.Tensor] | None:
    if layer.mode == "diag":
        return None
    if layer.mode == "ekfac":
        return approx._Q_A[module].double(), approx._Q_B[module].double()
    if layer.mode == "tekfac":
        return approx._Q_Phi[module].double(), approx._Q_Psi[module].double()
    factors = approx._damped_factors(module)
    left, right = (factors[0], factors[1]) if layer.mode == "kfac" else (factors[1], factors[2])
    return (torch.linalg.eigh(left.double())[1], torch.linalg.eigh(right.double())[1])


# --------------------------------------------------------------------------------------------------
# The divisor itself
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_divisor_reproduces_the_applied_step(mode: str) -> None:
    """``M_kfe / d`` is the step ``precondition`` takes, in the divisor's own basis and order.

    This is the test that catches a divisor indexed the wrong way round: the projection is rebuilt
    here from the optimizer's own factors, independently of :mod:`fisher_ref.divisor`, and compared
    with the step the optimizer returns for the same momentum.
    """
    model, optimizer = _train(mode)
    layers = divisor.measure(model, optimizer, mode=mode)
    assert layers, "no hooked module was read"
    approx = optimizer.approx
    modules = {name: m for name, m in model.named_modules()}
    for name, layer in layers.items():
        module = modules[name]
        weight = optimizer.state[module.weight]["exp_avg"].double()
        bias = (optimizer.state[module.bias]["exp_avg"].double()
                if getattr(module, "bias", None) is not None else None)
        applied = approx.precondition(module, optimizer.state[module.weight]["exp_avg"],
                                      None if bias is None else optimizer.state[module.bias]["exp_avg"])
        parts = applied if isinstance(applied, tuple) else (applied,)
        step = augment_direction(parts[0].double(),
                                 None if len(parts) == 1 else parts[1].double())
        basis = _basis_of(layer, approx, module)
        momentum = augment_direction(weight, bias)
        if basis is not None:
            Q_A, Q_B = basis
            momentum = Q_B.t() @ momentum @ Q_A
            step = Q_B.t() @ step @ Q_A
        expected = momentum / layer.divisor.reshape(momentum.shape)
        # Scaled by the tensor's own largest entry, NOT coordinate by coordinate. In the eigenbasis
        # many coordinates are zero in exact arithmetic and fp32 rounding noise in practice -- down
        # to 1e-13 relative, which is the whole reason ``_rescale_utils`` carries a guard -- so a
        # per-coordinate relative tolerance compares noise against noise and cannot pass. MEASURED
        # worst case over the five modes and all four layer types: 4.8e-8 (diag), 1.7e-7 (kfac),
        # 4.7e-7 (ekfac), 6.7e-7 (tekfac), 1.4e-6 (tkfac), i.e. the precision of the optimizer's own
        # fp32 round trip through the basis. The median ratio is 1.000000 everywhere.
        scaled = float((expected - step).abs().max() / step.abs().max())
        assert scaled < 1e-5, (
            f"{mode}/{name}: the divisor does not reproduce the applied step (scaled error "
            f"{scaled:.2e}); a wrong index order is the usual cause"
        )
        # And the value, not only the scale: half the coordinates must agree to fp32 precision.
        ratio = (expected / step)[step.abs() > step.abs().max() * 1e-3]
        assert float(ratio.median()) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("mode", EIGEN)
def test_add_form_divisor_is_the_stored_curvature_plus_lambda(mode: str) -> None:
    lam = 1e-3
    model, optimizer = _train(mode, lam=lam)
    layers = divisor.measure(model, optimizer, mode=mode)
    stored = optimizer.approx._s_star if mode == "ekfac" else optimizer.approx._Theta
    modules = {name: m for name, m in model.named_modules()}
    for name, layer in layers.items():
        s = stored[modules[name]].double().flatten()
        assert torch.allclose(layer.curvature, s, rtol=0, atol=0)
        assert torch.allclose(layer.divisor, s + lam, rtol=1e-12, atol=0)
        assert layer.form == "add"


@pytest.mark.parametrize("mode", EIGEN)
def test_floor_form_divisor_is_the_maximum_of_the_two(mode: str) -> None:
    lam = 1e-3
    model, optimizer = _train(mode, lam=lam, rescale_form="floor")
    layers = divisor.measure(model, optimizer, mode=mode)
    for layer in layers.values():
        assert layer.form == "floor"
        assert torch.allclose(layer.divisor, torch.maximum(layer.curvature,
                                                           torch.tensor(lam, dtype=torch.float64)))
        # Below lambda the divisor is exactly lambda, so nothing there is curvature-set.
        assert not bool(layer.curvature_set[layer.curvature < lam / 2].any())


@pytest.mark.parametrize("mode", EIGEN)
def test_clip_divisor_is_read_from_the_rule_and_not_recomputed(mode: str) -> None:
    """Under the clip the divisor is ``|M| / |u|``, which is a property of the step taken. It must
    come out equal to ``gamma * s`` exactly where the coordinate was not clipped -- the branch that
    makes the clip curvature-informed at all."""
    model, optimizer = _train(mode, rescale_form="clip", clip_fraction=0.5)
    layers = divisor.measure(model, optimizer, mode=mode)
    rule = optimizer.approx._clip
    modules = {name: m for name, m in model.named_modules()}
    for name, layer in layers.items():
        record = rule._records[modules[name]]
        gamma = float(record.gamma.double())
        unclipped = layer.curvature_set & torch.isfinite(layer.divisor)
        active = record.active.double().flatten() > 0.5
        both = unclipped & active
        if not bool(both.any()):
            continue
        assert torch.allclose(layer.divisor[both], gamma * layer.curvature[both],
                              rtol=1e-5, atol=0), (
            f"{mode}/{name}: an unclipped active coordinate must be divided by gamma * s"
        )


def test_clip_curvature_set_fraction_follows_its_own_hyperparameter() -> None:
    """The count of curvature-set directions under the clip is about ``1 - clip_fraction`` by
    construction. E21's decision rules depend on knowing this: it is why the clip is read on the
    share of the *step* those directions carry and not on how many of them there are."""
    for fraction in (0.3, 0.9):
        model, optimizer = _train("ekfac", rescale_form="clip", clip_fraction=fraction)
        layers = divisor.measure(model, optimizer, mode="ekfac")
        rule = optimizer.approx._clip
        modules = {name: m for name, m in model.named_modules()}
        for name, layer in layers.items():
            record = rule._records[modules[name]]
            active = record.active.flatten()
            if int(active.sum()) < 10:
                continue
            among_active = float(layer.curvature_set[active].to(torch.float64).mean())
            assert among_active == pytest.approx(1.0 - fraction, abs=0.15), (
                f"ekfac/{name}: {among_active:.3f} of the active coordinates are curvature-set at "
                f"clip_fraction={fraction}"
            )


# --------------------------------------------------------------------------------------------------
# The statistics
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_a_constant_far_above_the_curvature_crushes_it(mode: str) -> None:
    """Lot 5's reading, reproduced on a real optimizer: at a constant far above every stored value
    the divisor has no structure at all and the step is the plain momentum step.

    Only this half is measurable on the fixture network. The other half -- a constant far *below* the
    curvature -- is not, and not for want of a small enough number: this network's stored curvature is
    about 1e-21 (seven examples, and the two conventions of ``CLAUDE.md`` §4.3-§4.1 shrinking it by
    ~2e8), while ``diag``'s min-max collapses to ~0 because its factors are nearly constant here. A
    constant below all of that is inside fp32 noise. The statistics themselves are therefore checked
    on a constructed divisor, where both answers are known on paper:
    :func:`test_statistics_on_a_divisor_whose_answer_is_known`.
    """
    crushed = divisor.summarise(divisor.measure(*_train(mode, lam=1e6), mode=mode))
    assert crushed["frac_curvature_set"] == 0.0
    assert crushed["cond_divisor"] == pytest.approx(1.0, abs=1e-4)
    assert crushed["cos_to_plain"] == pytest.approx(1.0, abs=1e-6)
    assert crushed["non_curvature_share"] == pytest.approx(1.0, abs=1e-3)
    assert crushed["lambda_over_mean_divisor"] == pytest.approx(1.0, abs=1e-3)
    assert crushed["step_share_curvature_set"] == 0.0


def _constructed(lam: float, *, curvature: torch.Tensor,
                 momentum: torch.Tensor) -> divisor.LayerDivisor:
    """A ``LayerDivisor`` with ``d = s + lambda``, built by hand: the statistics then have an answer
    that can be written down instead of measured."""
    d = curvature + lam
    return divisor.LayerDivisor(
        name="constructed", kind="linear", mode="ekfac", form="add", divisor=d,
        curvature=curvature, lam=lam, curvature_set=curvature > lam,
        cos_to_plain=float("nan"), applied=momentum / d,
    )


def test_statistics_on_a_divisor_whose_answer_is_known() -> None:
    """Every statistic, against its value computed on paper, at the two extremes and in between."""
    s = torch.logspace(-8, 0, 9, dtype=torch.float64)      # 1e-8 ... 1, nine decades
    m = torch.ones(9, dtype=torch.float64)

    crushed = divisor.statistics(_constructed(1e6, curvature=s, momentum=m))
    assert crushed["frac_curvature_set"] == 0.0
    assert crushed["step_share_curvature_set"] == 0.0
    assert crushed["cond_divisor"] == pytest.approx(1.0, abs=1e-5)
    assert crushed["non_curvature_share"] == pytest.approx(1.0, abs=1e-6)

    informed = divisor.statistics(_constructed(1e-12, curvature=s, momentum=m))
    # Every one of the nine directions is above this constant, so all of them are curvature-set and
    # the divisor's condition number is the curvature's own: 1 / 1e-8.
    assert informed["frac_curvature_set"] == 1.0
    assert informed["step_share_curvature_set"] == pytest.approx(1.0, abs=1e-9)
    # (1 + 1e-12) / (1e-8 + 1e-12) = 9.9990e7: the constant is four orders below the smallest
    # curvature value, so it moves the condition number by one part in 1e4 and no more.
    assert informed["cond_divisor"] == pytest.approx(1e8, rel=1e-3)
    assert informed["cond_curvature"] == pytest.approx(1e8, rel=1e-12)
    assert informed["non_curvature_share"] == pytest.approx(0.0, abs=1e-9)
    assert informed["spearman_divisor_curvature"] == pytest.approx(1.0)

    # A constant inside the spectrum: the four values above 1e-4 are curvature-set, and because the
    # momentum is flat the smallest divisors carry the most step -- so the share of the step those
    # four carry is far below their share of the directions. That gap is the reason the two are
    # reported separately.
    middle = divisor.statistics(_constructed(1e-4, curvature=s, momentum=m))
    assert middle["frac_curvature_set"] == pytest.approx(4 / 9)
    assert middle["step_share_curvature_set"] < 0.01
    assert 1.0 < middle["cond_divisor"] < 1e8


@pytest.mark.parametrize("mode", MODES)
def test_lot5_statistic_matches_the_form_agnostic_one_for_an_added_constant(mode: str) -> None:
    """``lambda / mean(d)`` and ``1 - mean(s)/mean(d)`` are the same number whenever the damping is a
    constant that is added, which is what lets E21's gate be read against lot 5's published value.
    For ``kfac``/``tkfac``, whose damping is folded into the two factors, they are not -- and that is
    why the gate uses the first."""
    stats = divisor.summarise(divisor.measure(*_train(mode, lam=1e-4), mode=mode))
    if mode in ("diag",) + EIGEN:
        assert stats["lambda_over_mean_divisor"] == pytest.approx(stats["non_curvature_share"],
                                                                  rel=1e-9)
    else:
        assert math.isfinite(stats["lambda_over_mean_divisor"])


def test_step_share_is_defined_for_every_form_so_the_clip_can_be_compared() -> None:
    """Rule 3 subtracts the clip's share of the step from the additive form's. Both have to exist."""
    for kwargs in ({}, {"rescale_form": "floor"},
                   {"rescale_form": "clip", "clip_fraction": 0.5}):
        stats = divisor.summarise(divisor.measure(*_train("ekfac", **kwargs), mode="ekfac"))
        assert 0.0 <= stats["step_share_curvature_set"] <= 1.0


def test_statistics_report_nan_rather_than_a_stand_in_under_the_clip() -> None:
    layers = divisor.measure(*_train("ekfac", rescale_form="clip", clip_fraction=0.5), mode="ekfac")
    for layer in layers.values():
        stats = divisor.statistics(layer)
        assert math.isnan(stats["lambda_over_mean_divisor"]), "the clip adds no lambda"
        assert math.isnan(stats["frac_curvature_above_lambda"])
        assert layer.lam == 0.0
        assert "clip_clipped_fraction" in stats


def test_spearman_is_nan_when_the_divisor_has_no_variation() -> None:
    layers = divisor.measure(*_train("ekfac", lam=1e6), mode="ekfac")
    assert all(math.isnan(divisor.statistics(layer)["spearman_divisor_curvature"])
               or divisor.statistics(layer)["spearman_divisor_curvature"] == pytest.approx(1.0)
               for layer in layers.values())


def test_all_four_hooked_layer_types_are_read() -> None:
    layers = divisor.measure(*_train("ekfac"), mode="ekfac")
    assert {layer.kind for layer in layers.values()} == {"linear", "conv", "norm"}
    for layer in layers.values():
        assert layer.divisor.numel() == layer.curvature.numel()
        assert layer.divisor.ndim == 1


# --------------------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------------------

def test_clip_is_refused_for_a_mode_that_has_none() -> None:
    model, optimizer = _train("kfac")
    optimizer.approx.rescale_form = "clip"      # only reachable by hand: the optimizer refuses it
    with pytest.raises(ValueError, match="clip"):
        divisor.measure(model, optimizer, mode="kfac")


def test_sua_is_refused_rather_than_read_as_one_kronecker_product() -> None:
    model, optimizer = _train("ekfac", conv_sua=True, steps=4)
    with pytest.raises(NotImplementedError, match="conv_sua"):
        divisor.measure(model, optimizer, mode="ekfac")


# --------------------------------------------------------------------------------------------------
# The experiment's registry and its rules
# --------------------------------------------------------------------------------------------------

def _driver():
    import fisher_ref.experiments.e21_divisor_dominance as e21
    return e21


def test_every_registry_entry_names_a_real_benchmark_and_a_real_mode() -> None:
    from benchmarks.common.runner import discover_benchmarks
    e21 = _driver()
    known = set(discover_benchmarks())
    for table in (e21.TUNED_LAMBDA, e21.TUNED_TAU, e21.TUNED_CLIP_Q):
        for model, mode in table:
            assert model in known, f"{model} is not a benchmark"
            assert mode in MODES, f"{mode} is not a Fisher mode"
    # The clip exists for the two eigenbasis modes only, and a relative damping is refused for diag.
    assert all(mode in EIGEN for _, mode in e21.TUNED_CLIP_Q)
    assert all(mode != "diag" for _, mode in e21.TUNED_TAU)
    assert "diag" not in e21.ARM_MODES["layerrel"]
    assert set(e21.ARM_MODES["clipema"]) == set(EIGEN)


def test_the_residue_guard_is_what_lot_five_learned() -> None:
    """``0.08^k`` has to sit far below ``lambda``, not below 1. At E14's constants lot 5's own
    ``k = 10`` is not enough, and the guard has to say so instead of producing a number."""
    e21 = _driver()
    assert not e21.residue_check(1e-11, 1000, 100, "add")["ok"]
    assert e21.residue_check(1e-11, 2000, 100, "add")["ok"]
    assert e21.residue_check(1e-3, 1000, 100, "add")["ok"]
    # The clip uses no constant, so there is nothing for the residue to be small against here.
    assert e21.residue_check(None, 1000, 100, "clip")["ok"]
    assert e21.residue_check(1e-11, 2000, 100, "add")["over_lambda"] < 1e-3


def test_no_cell_is_planned_without_a_value_on_record(monkeypatch) -> None:
    """A (network, mode) E14, E15 or E16 never located a value for is skipped, not run at a
    guess."""
    e21 = _driver()
    assert e21.TUNED_LAMBDA[("resnet20_cifar", "kfac")] is None
    assert ("cct_2_3x2_cifar", "kfac") not in e21.TUNED_TAU


def test_decisions_refuse_to_read_a_verdict_through_a_missing_or_crashed_cell() -> None:
    e21 = _driver()
    planned = [e21.cell_key("shipped", "ekfac", 0.5), e21.cell_key("tuned", "ekfac", 0.5)]
    good = {"median": {"frac_curvature_set": 0.9, "cos_to_plain": 0.2,
                       "step_share_curvature_set": 0.9, "non_curvature_share": 0.1,
                       "cond_divisor": 500.0, "spread_curvature": 10.0,
                       "lambda_over_mean_divisor": 0.1}}
    # A planned cell that is absent.
    out = e21.decide("cnn_gn_cifar", {planned[0]: good}, planned)
    assert out["missing"] == [planned[1]]
    assert out["rules"]["2_3_4"]["readable"] is False
    # A cell that crashed is not silently dropped either.
    out = e21.decide("cnn_gn_cifar", {planned[0]: good, planned[1]: {"crashed": True,
                                                                     "median": {}}}, planned)
    assert out["crashed"] == [planned[1]]
    assert out["rules"]["2_3_4"]["readable"] is False
    assert out["table"]["tuned|ekfac"]["verdict"] == "inconclusive"


def test_the_gate_fails_when_the_bridge_arm_does_not_reproduce_lot_five() -> None:
    e21 = _driver()

    def bridge(cond: float, share: float):
        return {e21.cell_key("bridge", mode, 0.5): {
            "median": {"cond_divisor": cond, "lambda_over_mean_divisor": share}}
            for mode in e21.ARM_MODES["bridge"]}

    planned = list(bridge(1.0, 1.0))
    assert e21.decide("cnn_gn_cifar", bridge(1.02, 0.999), planned)["rules"]["1_gate"]["passes"]
    # Lot 5 measured cond(F~) between 1.0000 and 1.1035; a divisor with real structure there would
    # mean this job is not measuring lot 5's object.
    assert e21.decide("cnn_gn_cifar", bridge(50.0, 0.3),
                      planned)["rules"]["1_gate"]["passes"] is False
    # On a network lot 5 never ran, the bridge is a new number rather than a gate.
    assert e21.decide("cct_2_3x2_cifar", bridge(50.0, 0.3),
                      planned)["rules"]["1_gate"]["applies"] is False


def test_the_clip_row_is_never_labelled_lambda_set() -> None:
    """The clip uses no constant. Whatever its verdict is, calling it "lambda-set" would be wrong."""
    e21 = _driver()
    key = e21.cell_key("clipema", "ekfac", 0.5)
    cells = {key: {"median": {"frac_curvature_set": 0.01, "step_share_curvature_set": 0.004,
                             "cos_to_plain": 0.19, "non_curvature_share": 1.0,
                             "cond_divisor": 1e4, "spread_curvature": 10.0}}}
    row = e21.decide("cnn_gn_cifar", cells, [key])["table"]["clipema|ekfac"]
    assert row["verdict"] == "cap_set"
    assert row["read_on"] == "step_share_curvature_set"
