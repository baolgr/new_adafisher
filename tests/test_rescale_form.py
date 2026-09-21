"""``rescale_form``: how ``ekfac``/``tekfac`` divide by the stored curvature inside their eigenbasis.

Three forms (``src/adafisher_modes/approximations/_rescale_utils.py``, ``docs/reports/
plan_floor_clip.md``):

* ``"add"``   -- ``M / (s + lambda)``, the default and what both papers prescribe;
* ``"floor"`` -- ``M / max(s, lambda)``, family A of ``docs/reports/fr/etude_clipping_vs_damping.md``;
* ``"clip"``  -- family B (Sophia-type): an active coordinate moves by ``sign(M) min(r / gamma, 1)``
  with ``r = |M| / s``, an inactive one (``|M| <= guard``) by ``M / max(gamma s, guard)``. Three
  per-module thresholds: ``"quantile"`` (a fraction of the active coordinates clipped at every
  step), ``"ema"`` (the same, rescaled by the momentum's size against its bias-corrected running
  average) and ``"fixed"`` (the median of the module's quantile over a window, frozen at a
  calibration step).

What is pinned here: the default is bit-identical to not passing the knob and to the expression the
two modes used before it existed; ``"floor"`` is the linear operator ``f_tilde`` says it is, on
every layer type the E16 networks use; ``"quantile"`` clips exactly the fraction asked for among the
active coordinates, is invariant to the scale of the momentum and of the curvature, and never
produces a NaN or a full step out of a zero-curvature, rounding-level direction; ``"ema"`` reduces
to ``"quantile"`` at constant momentum size and follows the momentum's size otherwise, with no
single early step dominating its average; ``"fixed"`` is ``"quantile"`` until it freezes each
module's own window median, is continuous at the switch and is then a true conditional clip; the
optimizer hands the clip the bias-corrected momentum; and every meaningless combination is refused.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from adafisher_modes import AdaFisherMulti
from adafisher_modes.approximations._rescale_utils import ClipRule, active_quantile, rescale
from conftest import TinyMultiLayerNet, seed_all

MODES = ["ekfac", "tekfac"]
# Name of the stored rescaling and of the two eigenbases, per mode.
ATTRS = {"ekfac": ("_s_star", "_Q_A", "_Q_B"), "tekfac": ("_Theta", "_Q_Phi", "_Q_Psi")}


def _random_problem(seed: int = 0, shape=(7, 11), dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(shape, generator=g, dtype=dtype)
    # A curvature spread over six decades, as the real stored ones are.
    s = torch.exp(torch.empty(shape[-2:], dtype=dtype).uniform_(-14.0, 0.0, generator=g))
    return M, s


def _clip_by_definition(M: torch.Tensor, s: torch.Tensor, gamma: float,
                        guard_fraction: float) -> torch.Tensor:
    """The clip written out from its definition, independently of ``ClipRule``."""
    guard = guard_fraction * M.pow(2).mean().sqrt()
    active = M.abs() > guard
    r = M.abs() / s
    u_active = torch.sign(M) * torch.clamp(r / gamma, max=1.0)
    return torch.where(active, u_active, M / torch.maximum(gamma * s, guard))


def _quantile(M, s, q, guard=1e-12, module=None):
    rule = ClipRule("quantile", q, guard)
    module = module if module is not None else nn.Linear(1, 1)
    return rule.apply(module, M, s), rule.stats[module]


# ---------------------------------------------------------------------------------------------
# "add" and "floor", on the helper
# ---------------------------------------------------------------------------------------------


def test_add_is_the_pre_existing_expression_bit_for_bit() -> None:
    M, s = _random_problem()
    for lam in (1e-3, torch.tensor(3e-7, dtype=torch.float64)):
        assert torch.equal(rescale(M, s, lam, "add"), M / (s + lam))


def test_floor_is_max_of_curvature_and_lambda() -> None:
    M, s = _random_problem()
    lam = float(s.median())
    expected = torch.where(s > lam, M / s, M / lam)
    torch.testing.assert_close(rescale(M, s, lam, "floor"), expected, rtol=0, atol=0)
    lam_t = torch.tensor(lam, dtype=torch.float64)          # a relative damping's 0-dim tensor
    torch.testing.assert_close(rescale(M, s, lam_t, "floor"), expected, rtol=0, atol=0)
    assert (s > lam).any() and (s < lam).any()


def test_floor_and_add_agree_where_lambda_dominates_or_vanishes() -> None:
    """The two forms differ only for curvature values near lambda. Far above every value they are
    both M/lambda to within s/lambda; far below every value both are M/s to within lambda/s."""
    M, s = _random_problem()
    for lam in (float(s.max()) * 1e8, float(s.min()) * 1e-8):
        torch.testing.assert_close(rescale(M, s, lam, "floor"), rescale(M, s, lam, "add"),
                                   rtol=2e-8, atol=0)


def test_rescale_refuses_clip() -> None:
    M, s = _random_problem()
    with pytest.raises(ValueError, match="ClipRule"):
        rescale(M, s, 1e-3, "clip")


# ---------------------------------------------------------------------------------------------
# "quantile", on the rule itself
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.01, 0.1, 0.5, 0.7, 0.9, 0.99])
def test_quantile_clips_exactly_the_requested_count(q: float) -> None:
    M, s = _random_problem(shape=(40, 50))
    u, st = _quantile(M, s, q)
    n = M.numel()
    c = max(math.floor(q * n), 1)
    clipped = (M.abs() / s) >= st["gamma"]
    assert int(clipped.sum()) == c
    assert float(st["clipped_fraction"]) == pytest.approx(c / n, abs=0)
    assert u.abs().max() <= 1.0
    assert torch.equal(u[clipped].abs(), torch.ones(c, dtype=u.dtype))
    # Every unclipped coordinate is the undamped step divided by gamma, and keeps its sign.
    free = ~clipped
    torch.testing.assert_close(u[free], (M / (st["gamma"] * s))[free], rtol=1e-14, atol=0)
    assert torch.equal(torch.sign(u), torch.sign(M))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quantile_count_is_exact_on_many_random_problems(dtype: torch.dtype) -> None:
    """Clipped is defined by the very tensor gamma is selected from, so the count is c, not "c or
    c - 1 through one rounding", including in fp32. 400 problems of varied size and fraction."""
    for seed in range(400):
        g = torch.Generator().manual_seed(seed)
        n_rows, n_cols = 2 + seed % 13, 3 + (7 * seed) % 17
        q = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99][seed % 6]
        M, s = _random_problem(seed, (n_rows, n_cols), dtype)
        M = M * torch.exp(torch.randn(M.shape, generator=g, dtype=dtype))   # a wide |M| spread
        u, st = _quantile(M, s, q, guard=1e-3)
        active = M.abs() > 1e-3 * M.pow(2).mean().sqrt()
        c = max(math.floor(q * int(active.sum())), 1)
        clipped = active & ((M.abs() / s.clamp_min(torch.finfo(dtype).tiny)) >= st["gamma"])
        assert int(clipped.sum()) == c, seed
        assert float(u.abs().max()) <= 1.0


def test_active_quantile_matches_a_sort_of_the_active_values() -> None:
    """The sync-free selection (sort + gather) returns the c-th largest active value, and the
    largest finite float when nothing is active."""
    M, s = _random_problem(shape=(30, 20))
    r = M.abs() / s
    active = torch.rand(M.shape, generator=torch.Generator().manual_seed(2)) > 0.4
    for q in (0.05, 0.5, 0.95):
        c = max(math.floor(q * int(active.sum())), 1)
        expected = torch.sort(r[active], descending=True).values[c - 1]
        assert torch.equal(active_quantile(r, active, q), expected)
    none = torch.zeros_like(active)
    assert float(active_quantile(r, none, 0.5)) == torch.finfo(r.dtype).max


def test_quantile_matches_its_definition() -> None:
    M, s = _random_problem(shape=(20, 30))
    u, st = _quantile(M, s, 0.4, guard=1e-3)
    torch.testing.assert_close(u, _clip_by_definition(M, s, float(st["gamma"]), 1e-3),
                               rtol=1e-15, atol=0)


@pytest.mark.parametrize("c", [1e-9, 1e-3, 7.0, 1e6])
def test_quantile_is_invariant_to_the_scale_of_momentum_and_curvature(c: float) -> None:
    """The 2e8 scale error of the stored curvature, the 1/batch^2 rule, the per-layer 1/T of the
    shared layers and the momentum's own size cannot reach the step."""
    M, s = _random_problem(shape=(20, 30))
    ref, _ = _quantile(M, s, 0.5)
    torch.testing.assert_close(_quantile(c * M, s, 0.5)[0], ref, rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(_quantile(M, c * s, 0.5)[0], ref, rtol=1e-12, atol=1e-15)


def test_quantile_counts_only_active_coordinates() -> None:
    """Coordinates at rounding level are neither counted in the fraction nor clipped to a full
    step. Without that restriction gamma would sit inside the rounding noise."""
    M, s = _random_problem(shape=(20, 30))
    M = M.clone()
    M[:, 10:] = 1e-14 * torch.randn(20, 20, dtype=M.dtype,
                                    generator=torch.Generator().manual_seed(3))
    u, st = _quantile(M, s, 0.5, guard=1e-3)
    n_active = 20 * 10
    assert float(st["active_fraction"]) == pytest.approx(n_active / M.numel(), abs=0)
    assert int((u[:, :10].abs() == 1.0).sum()) == n_active // 2
    assert float(u[:, 10:].abs().max()) < 1e-6


def test_quantile_is_safe_on_zero_curvature_directions() -> None:
    """An empirical Fisher has exact null directions (the head's logit-shift kernel) where the
    momentum is zero up to rounding. Such a coordinate must not become a full step."""
    M, s = _random_problem(shape=(10, 10))
    s, M = s.clone(), M.clone()
    s[0, :5] = 0.0
    M[0, :3] = 0.0            # zero curvature and zero momentum: 0/0 must not give NaN
    M[0, 3:5] = 1e-12         # zero curvature and rounding-level momentum
    guard = 1e-3
    u, _ = _quantile(M, s, 0.5, guard=guard)
    assert torch.isfinite(u).all()
    assert torch.equal(u[0, :3], torch.zeros(3, dtype=u.dtype))
    rms = float(M.pow(2).mean().sqrt())
    assert float(u[0, 3:5].abs().max()) <= 1e-12 / (guard * rms) * (1 + 1e-12)
    # Without the guard the same coordinates are active, have r = inf, and are clipped to a full
    # step: the guard is what prevents it, not an accident of the numbers.
    u_no_guard, _ = _quantile(M, s, 0.5, guard=1e-30)
    assert torch.equal(u_no_guard[0, 3:5].abs(), torch.ones(2, dtype=u.dtype))


def test_quantile_all_zero_direction_gives_zero() -> None:
    _, s = _random_problem()
    M = torch.zeros(7, 11, dtype=torch.float64)
    assert torch.equal(_quantile(M, s, 0.5)[0], M)


def test_quantile_broadcasts_over_sua_offsets() -> None:
    """Under SUA the projected direction is (k_h*k_w, d_out, d_in) against a (d_out, d_in) s; the
    fraction is taken over the whole module, every offset included."""
    M, s = _random_problem(shape=(9, 6, 5))
    u, st = _quantile(M, s, 0.25)
    assert u.shape == M.shape
    assert int(((M.abs() / s) >= st["gamma"]).sum()) == math.floor(0.25 * M.numel())


def test_stats_are_read_lazily_and_split_the_two_columns_of_a_norm_layer() -> None:
    M, s = _random_problem(shape=(16, 2))
    _, st = _quantile(M, s, 0.5)
    for key in ("gamma", "clipped_fraction", "active_fraction", "guard_fraction",
                "u2_share_below_1e-2_rms", "clipped_fraction_col0", "clipped_fraction_col1"):
        assert key in st
    clipped = (M.abs() / s) >= st["gamma"]
    for j in (0, 1):
        assert float(st[f"clipped_fraction_col{j}"]) == pytest.approx(
            float(clipped[:, j].double().mean()), abs=1e-15)


# ---------------------------------------------------------------------------------------------
# "ema", on the rule itself (fp64)
# ---------------------------------------------------------------------------------------------


def test_ema_first_step_and_constant_size_equal_quantile() -> None:
    M, s = _random_problem(shape=(20, 30))
    rule = ClipRule("ema", 0.4, 1e-12, ema_horizon=10)
    module = nn.Linear(1, 1)
    ref, _ = _quantile(M, s, 0.4)
    for _ in range(5):   # same momentum size every step: mu = mu_bar
        torch.testing.assert_close(rule.apply(module, M, s), ref, rtol=1e-12, atol=0)


def test_ema_follows_the_momentum_size_against_its_history() -> None:
    """After K steps at one size, a momentum a times smaller or larger is compared with a threshold
    held at the history's scale. The horizon is huge, so the bias-corrected average is the plain
    mean of the K + 1 logs: mu_bar = mu * a^(1/(K+1)), and gamma = gamma_q(M) * a^(1/(K+1))."""
    M, s = _random_problem(shape=(20, 30))
    K = 999
    _, st_ref = _quantile(M, s, 0.4)
    n_ref = int(((M.abs() / s) >= st_ref["gamma"]).sum())
    for a in (0.1, 3.0):
        rule = ClipRule("ema", 0.4, 1e-12, ema_horizon=10**12)
        module = nn.Linear(1, 1)
        for _ in range(K):
            rule.apply(module, M, s)
        u = rule.apply(module, a * M, s)
        gamma = float(st_ref["gamma"]) * a ** (1.0 / (K + 1))
        torch.testing.assert_close(u, _clip_by_definition(a * M, s, gamma, 1e-12),
                                   rtol=1e-9, atol=0)
        n_clipped = int((u.abs() == 1.0).sum())
        assert (n_clipped < n_ref) if a < 1 else (n_clipped > n_ref)
        assert float(rule.stats[module]["momentum_ratio"]) == pytest.approx(
            a ** (K / (K + 1)), rel=1e-9)


def test_ema_is_not_dominated_by_its_first_step() -> None:
    """The first bias-corrected momentum is one raw gradient, several times larger than the
    momentum settles to. Seeded with it, the average stayed biased for 12-18 % of a real run; with
    Adam's bias correction the first step weighs about 1/k of the average after k steps, as every
    other step does."""
    M, s = _random_problem(shape=(20, 30))
    rule = ClipRule("ema", 0.4, 1e-12, ema_horizon=1000)
    module = nn.Linear(1, 1)
    rule.apply(module, 5.0 * M, s)                  # a first step five times too large
    for _ in range(99):
        rule.apply(module, M, s)
    mu = float(M.pow(2).mean().sqrt())
    w = 1e-3
    weights = [(1 - w) ** (99 - k) for k in range(100)]     # step k's weight after 100 steps
    expected = math.log(mu) + math.log(5.0) * weights[0] / sum(weights)
    assert float(rule.log_mu_bar(module)) == pytest.approx(expected, rel=1e-12)
    assert weights[0] / sum(weights) < 0.0101              # about 1/100, not e^(-0.1) = 0.90


@pytest.mark.parametrize("c", [1e-6, 50.0])
def test_ema_is_invariant_to_a_constant_scale_of_the_whole_history(c: float) -> None:
    g = torch.Generator().manual_seed(4)
    seq = [torch.randn(20, 30, generator=g, dtype=torch.float64) * f for f in (1.0, 0.5, 2.0, 0.3)]
    _, s = _random_problem(shape=(20, 30))
    r1 = ClipRule("ema", 0.4, 1e-12, ema_horizon=3)
    r2 = ClipRule("ema", 0.4, 1e-12, ema_horizon=3)
    module = nn.Linear(1, 1)
    for M in seq:
        torch.testing.assert_close(r2.apply(module, c * M, c * s), r1.apply(module, M, s),
                                   rtol=1e-12, atol=1e-15)


def test_ema_ignores_an_all_zero_momentum_including_as_its_first_step() -> None:
    """A zero momentum carries no size. It used to seed the average at log(tiny) = -87 when it came
    first, clipping everything for thousands of steps."""
    M, s = _random_problem(shape=(20, 30))
    rule = ClipRule("ema", 0.4, 1e-12, ema_horizon=2)
    module = nn.Linear(1, 1)
    assert torch.equal(rule.apply(module, torch.zeros_like(M), s), torch.zeros_like(M))
    assert rule.log_mu_bar(module) is None
    ref, _ = _quantile(M, s, 0.4)
    torch.testing.assert_close(rule.apply(module, M, s), ref, rtol=1e-12, atol=0)
    before = rule.log_mu_bar(module).clone()
    assert torch.equal(rule.apply(module, torch.zeros_like(M), s), torch.zeros_like(M))
    assert torch.equal(rule.log_mu_bar(module), before)


# ---------------------------------------------------------------------------------------------
# "fixed", on the rule itself (fp64)
# ---------------------------------------------------------------------------------------------


def _fixed(calibrate_at: int, window: int, q: float = 0.3) -> ClipRule:
    return ClipRule("fixed", q, 1e-12, calibrate_at=calibrate_at, calibration_window=window)


def test_fixed_is_quantile_until_it_freezes_the_window_median() -> None:
    """Steps 2, 3, 4 fall in the window of a calibration at step 5 with window 3: their per-step
    quantiles are kept, their lower median is frozen at step 5 and used from then on."""
    rule = _fixed(calibrate_at=5, window=3)
    module = nn.Linear(1, 1)
    problems = [_random_problem(seed, (8, 9)) for seed in range(8)]
    quantiles = []
    for step in range(5):
        rule.begin_step(step)
        M, s = problems[step]
        u = rule.apply(module, M, s)
        ref, st = _quantile(M, s, 0.3)
        assert torch.equal(u, ref)
        quantiles.append(st["gamma"])
        assert module not in rule.gamma_fixed
    expected = torch.stack(quantiles[2:5]).median()
    for step in (5, 6, 7):
        rule.begin_step(step)
        M, s = problems[step]
        u = rule.apply(module, M, s)
        assert torch.equal(rule.gamma_fixed[module], expected)
        torch.testing.assert_close(u, _clip_by_definition(M, s, float(expected), 1e-12),
                                   rtol=1e-15, atol=0)
        assert float(rule.stats[module]["calibrated"]) == 1.0


def test_fixed_is_per_module_and_invariant_to_one_module_s_scale() -> None:
    """Each module freezes its own threshold, so multiplying one module's stored curvature by any
    constant -- the per-layer 1/T of a shared layer, a LayerNorm's surrogate -- changes nothing,
    before or after calibration."""
    (M1, s1), (M2, s2) = _random_problem(0, (8, 9)), _random_problem(1, (5, 12))
    ra, rb = _fixed(3, 2), _fixed(3, 2)
    m1, m2 = nn.Linear(1, 1), nn.Linear(1, 1)
    for step in range(6):
        for rule in (ra, rb):
            rule.begin_step(step)
        k = 1.0 + 0.1 * step
        torch.testing.assert_close(rb.apply(m1, k * M1, s1), ra.apply(m1, k * M1, s1),
                                   rtol=0, atol=0)
        torch.testing.assert_close(rb.apply(m2, k * M2, 1024.0 * s2), ra.apply(m2, k * M2, s2),
                                   rtol=1e-12, atol=1e-15)
    assert not torch.equal(ra.gamma_fixed[m1], ra.gamma_fixed[m2])


def test_fixed_switch_is_continuous_and_then_a_conditional_clip() -> None:
    """With the same momentum and curvature throughout, the frozen threshold is the quantile the
    module had been using, so the step does not jump at the switch. After it, a smaller momentum
    moves the unclipped coordinates proportionally and clips fewer."""
    M, s = _random_problem(shape=(20, 30))
    rule = _fixed(calibrate_at=4, window=4)
    module = nn.Linear(1, 1)
    before = None
    for step in range(5):
        rule.begin_step(step)
        u = rule.apply(module, M, s)
        if step == 3:
            before = u
    assert torch.equal(u, before)
    rule.begin_step(5)
    u_half = rule.apply(module, 0.5 * M, s)
    free = u.abs() < 1.0
    torch.testing.assert_close(u_half[free], 0.5 * u[free], rtol=1e-15, atol=0)
    assert int((u_half.abs() == 1.0).sum()) < int((u.abs() == 1.0).sum())


def test_fixed_module_first_seen_after_calibration_freezes_its_first_quantile() -> None:
    rule = _fixed(calibrate_at=2, window=5)
    late = nn.Linear(1, 1)
    M, s = _random_problem(shape=(6, 7))
    rule.begin_step(9)
    u = rule.apply(late, M, s)
    ref, st = _quantile(M, s, 0.3)
    assert torch.equal(u, ref)
    assert torch.equal(rule.gamma_fixed[late], st["gamma"])


def test_fixed_outside_an_optimizer_stays_quantile() -> None:
    M, s = _random_problem(shape=(6, 7))
    rule = _fixed(calibrate_at=0, window=5)
    module = nn.Linear(1, 1)
    assert torch.equal(rule.apply(module, M, s), _quantile(M, s, 0.3)[0])
    assert module not in rule.gamma_fixed


# ---------------------------------------------------------------------------------------------
# Through the optimizer
# ---------------------------------------------------------------------------------------------


def _loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x).pow(2).mean()


def _run(mode: str, steps: int = 12, **kwargs):
    seed_all(0)
    model = TinyMultiLayerNet()
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=3, T_eig=3,
                         eig_before_rescale=True, **kwargs)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randn(6, 2, 5, 5, generator=g)
        opt.zero_grad()
        _loss(model, x).backward()
        opt.step()
    return model, opt


@pytest.mark.parametrize("mode", MODES)
def test_default_is_bit_identical_to_not_passing_the_knob(mode: str) -> None:
    a, _ = _run(mode)
    b, _ = _run(mode, rescale_form="add")
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


@pytest.mark.parametrize("mode", MODES)
def test_add_precondition_is_the_pre_existing_formula(mode: str) -> None:
    """``precondition`` under the default reproduces, bit for bit, the formula both modes spelled
    out inline before ``rescale_form`` existed."""
    model, opt = _run(mode, steps=6)
    s_name, qa_name, qb_name = ATTRS[mode]
    module = model.fc1
    s, Q_A, Q_B = (getattr(opt.approx, n)[module] for n in (s_name, qa_name, qb_name))
    g = torch.Generator().manual_seed(5)
    w = torch.randn(module.weight.shape, generator=g)
    b = torch.randn(module.bias.shape, generator=g)
    got_w, got_b = opt.approx.precondition(module, w, b)
    M = torch.cat([w, b[:, None]], dim=1)
    ref = Q_B @ ((Q_B.t() @ M @ Q_A) / (s + opt.approx.Lambda)) @ Q_A.t()
    assert torch.equal(got_w, ref[:, :-1]) and torch.equal(got_b, ref[:, -1])


@pytest.mark.parametrize("mode", MODES)
def test_floor_precondition_matches_its_dense_operator(mode: str) -> None:
    """``precondition`` under ``"floor"`` is the inverse of the matrix ``f_tilde`` builds, with
    lambda placed inside the stored spectrum so both branches of the max are exercised."""
    model, opt = _run(mode, steps=6, rescale_form="floor")
    s_name = ATTRS[mode][0]
    module = model.fc2
    s = getattr(opt.approx, s_name)[module]
    opt.approx.Lambda = float(s.median())
    assert (s > opt.approx.Lambda).any() and (s < opt.approx.Lambda).any()
    g = torch.Generator().manual_seed(6)
    w = torch.randn(module.weight.shape, generator=g, dtype=s.dtype)
    b = torch.randn(module.bias.shape, generator=g, dtype=s.dtype)
    got_w, got_b = opt.approx.precondition(module, w, b)
    F = opt.approx.f_tilde(module).double()
    M = torch.cat([w, b[:, None]], dim=1).double()
    solved = torch.linalg.solve(F, M.flatten()).reshape(M.shape)
    torch.testing.assert_close(got_w.double(), solved[:, :-1], rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(got_b.double(), solved[:, -1], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
def test_clip_receives_the_bias_corrected_momentum(mode: str) -> None:
    """The step is ``lr`` times the clip of ``m / (1 - beta^t)``, applied without dividing again:
    not ``lr / (1 - beta^t)`` times it, which is 10x at the first step. And the momentum buffer
    itself is left untouched."""
    seed_all(0)
    model = TinyMultiLayerNet()
    lr, beta = 1e-3, 0.9
    opt = AdaFisherMulti(model, lr=lr, beta=beta, fisher_mode=mode, TCov=1, T_eig=1,
                         eig_before_rescale=True, rescale_form="clip", clip_fraction=0.5)
    x = torch.randn(6, 2, 5, 5, generator=torch.Generator().manual_seed(1))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    _loss(model, x).backward()
    opt.step()
    module = model.fc1
    bc = 1 - beta
    m_w, m_b = opt.state[module.weight]["exp_avg"], opt.state[module.bias]["exp_avg"]
    torch.testing.assert_close(m_w, (1 - beta) * module.weight.grad, rtol=1e-6, atol=0)
    direction_w, direction_b = opt.approx.precondition(module, m_w / bc, m_b / bc)
    # atol: the parameters are fp32 of size ~0.1, so the difference of two of them is exact only to
    # about one ulp there (~1e-8), against a step of up to lr = 1e-3. Dividing by the bias
    # correction again would be off by a factor 10, i.e. by up to 9e-3.
    torch.testing.assert_close(model.fc1.weight - before["fc1.weight"], -lr * direction_w,
                               rtol=0, atol=2e-8)
    torch.testing.assert_close(model.fc1.bias - before["fc1.bias"], -lr * direction_b,
                               rtol=0, atol=2e-8)
    _, qa_name, qb_name = ATTRS[mode]
    Q_A, Q_B = getattr(opt.approx, qa_name)[module], getattr(opt.approx, qb_name)[module]
    M = torch.cat([direction_w, direction_b[:, None]], dim=1)
    assert float((Q_B.t() @ M @ Q_A).abs().max()) == pytest.approx(1.0, rel=1e-5)


THRESHOLDS = [
    ("quantile", {"clip_fraction": 0.7}),
    ("ema", {"clip_fraction": 0.7}),
    ("fixed", {"clip_fraction": 0.7, "clip_calibrate_at": 6, "clip_calibration_window": 3}),
]
THRESHOLD_IDS = [th for th, _ in THRESHOLDS]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("form,extra", [("floor", {})] + [
    ("clip", {"clip_threshold": th, **kw}) for th, kw in THRESHOLDS
], ids=["floor"] + THRESHOLD_IDS)
def test_runs_on_all_four_layer_types(mode: str, form: str, extra: dict) -> None:
    lam = 1e-9 if form == "floor" else 1e-3
    model, opt = _run(mode, steps=20, rescale_form=form, Lambda=lam, **extra)
    for p in model.parameters():
        assert torch.isfinite(p).all()
    if form == "clip":
        stats = opt.approx._clip_stats
        assert set(stats) == set(opt.modules)
        for st in stats.values():
            assert 0.0 <= float(st["clipped_fraction"]) <= 1.0
            assert float(st["active_fraction"]) > 0.0
        if extra["clip_threshold"] == "quantile":
            for module, st in stats.items():
                assert float(st["clipped_fraction"]) <= 0.7 + 1e-6, module


@pytest.mark.parametrize("th,kw", THRESHOLDS, ids=THRESHOLD_IDS)
def test_clip_trajectory_is_independent_of_lambda(th: str, kw: dict) -> None:
    a, _ = _run("ekfac", rescale_form="clip", clip_threshold=th, Lambda=1e-3, **kw)
    b, _ = _run("ekfac", rescale_form="clip", clip_threshold=th, Lambda=1e-11, **kw)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


@pytest.mark.parametrize("mode", MODES)
def test_fixed_freezes_every_module_once_through_the_optimizer(mode: str) -> None:
    model, opt = _run(mode, steps=6, rescale_form="clip", clip_threshold="fixed",
                      clip_fraction=0.7, clip_calibrate_at=6, clip_calibration_window=3)
    rule = opt.approx._clip
    assert rule.gamma_fixed == {}
    g = torch.Generator().manual_seed(9)
    frozen = None
    for _ in range(4):   # steps 6 (the calibration), 7, 8, 9
        x = torch.randn(6, 2, 5, 5, generator=g)
        opt.zero_grad()
        _loss(model, x).backward()
        opt.step()
        if frozen is None:
            frozen = {m: v.clone() for m, v in rule.gamma_fixed.items()}
            assert set(frozen) == set(opt.modules)
    for m, v in frozen.items():
        assert torch.equal(rule.gamma_fixed[m], v)
        assert float(opt.approx._clip_stats[m]["calibrated"]) == 1.0
    assert len({float(v) for v in frozen.values()}) == len(frozen)   # one threshold per module


# ---------------------------------------------------------------------------------------------
# The layer types of the E16 networks that TinyMultiLayerNet does not have
# ---------------------------------------------------------------------------------------------


def _narrow_cct() -> nn.Module:
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from benchmarks.models.cct_2_3x2_cifar.model import CCT

    # Bias-free Conv2d tokenizer, bias-free fused qkv on 3-D token inputs, LayerNorms, the
    # attention-pool Linear with d_out = 1, and the head. No dropout, for determinism.
    return CCT(img_size=16, embed_dim=16, num_heads=2, drop=0.0, attn_drop=0.0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("form,extra", [("floor", {})] + [
    ("clip", {"clip_threshold": th, **kw}) for th, kw in THRESHOLDS
], ids=["floor"] + THRESHOLD_IDS)
def test_every_cct_layer_type(mode: str, form: str, extra: dict) -> None:
    seed_all(0)
    # fp64 throughout: the floor check solves f_tilde densely at lambda = 1e-9, where the operator's
    # condition number is ~1e6 or more and an fp32 build would leave rounding at the 1e-2 level.
    model = _narrow_cct().double()
    lam = 1e-9 if form == "floor" else 1e-3
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=2, T_eig=2, Lambda=lam,
                         eig_before_rescale=True, rescale_form=form, **extra)
    g = torch.Generator().manual_seed(1)
    for _ in range(8):
        x = torch.randn(4, 3, 16, 16, generator=g, dtype=torch.float64)
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), torch.randint(0, 10, (4,), generator=g)).backward()
        opt.step()
    s_name, qa_name, qb_name = ATTRS[mode]
    kinds = set()
    for module in opt.modules:
        kinds.add((type(module).__name__, module.bias is None))
        s = getattr(opt.approx, s_name)[module]
        Q_A, Q_B = getattr(opt.approx, qa_name)[module], getattr(opt.approx, qb_name)[module]
        w = torch.randn(module.weight.shape, generator=g, dtype=torch.float64)
        b = (None if module.bias is None
             else torch.randn(module.bias.shape, generator=g, dtype=torch.float64))
        got = opt.approx.precondition(module, w, b)
        got_w, got_b = (got, None) if module.bias is None else got
        assert got_w.shape == module.weight.shape
        M = w.reshape(w.size(0), -1)
        if b is not None:
            M = torch.cat([M, b[:, None]], dim=1)
        M_kfe = Q_B.t() @ M @ Q_A
        assert M_kfe.shape == s.shape
        G = torch.cat([got_w.reshape(w.size(0), -1)]
                      + ([got_b[:, None]] if got_b is not None else []), dim=1)
        U = Q_B.t() @ G @ Q_A
        if form == "floor":
            F = opt.approx.f_tilde(module)
            solved = torch.linalg.solve(F, M.flatten()).reshape(M.shape)
            torch.testing.assert_close(G, solved, rtol=1e-7, atol=1e-9)
        else:
            gamma = float(opt.approx._clip_stats[module]["gamma"])
            torch.testing.assert_close(U, _clip_by_definition(M_kfe, s, gamma, 1e-3),
                                       rtol=1e-10, atol=1e-12)
            assert float(U.abs().max()) <= 1.0 + 1e-12
    assert ("Conv2d", True) in kinds and ("Linear", True) in kinds
    assert ("LayerNorm", False) in kinds
    assert any(m.weight.shape[0] == 1 for m in opt.modules if isinstance(m, nn.Linear))


# ---------------------------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["diag", "kfac", "tkfac"])
def test_other_modes_refuse_the_knob(mode: str) -> None:
    with pytest.raises(ValueError, match="ekfac"):
        AdaFisherMulti(TinyMultiLayerNet(), fisher_mode=mode, rescale_form="floor")
    with pytest.raises(ValueError, match="ekfac"):
        AdaFisherMulti(TinyMultiLayerNet(), fisher_mode=mode, rescale_form="clip",
                       clip_fraction=0.5)


@pytest.mark.parametrize("kwargs,match", [
    ({"rescale_form": "clip"}, "clip_fraction"),
    ({"rescale_form": "clip", "clip_fraction": 0.0}, "clip_fraction"),
    ({"rescale_form": "clip", "clip_fraction": 1.0}, "clip_fraction"),
    ({"rescale_form": "floor", "clip_fraction": 0.5}, "only applies"),
    ({"rescale_form": "add", "clip_fraction": 0.5}, "only applies"),
    ({"rescale_form": "add", "clip_threshold": "ema"}, "only applies"),
    ({"rescale_form": "cap"}, "must be one of"),
    ({"rescale_form": "clip", "clip_fraction": 0.5, "clip_guard": 0.0}, "clip_guard"),
    ({"rescale_form": "clip", "clip_fraction": 0.5, "hold_cap": True}, "lambda"),
    ({"rescale_form": "clip", "clip_fraction": 0.5, "damping": "layer_relative",
      "damping_tau": 0.1}, "lambda"),
    ({"rescale_form": "clip", "clip_threshold": "median", "clip_fraction": 0.5}, "must be one of"),
    ({"rescale_form": "clip", "clip_fraction": 0.5, "clip_ema_horizon": 10}, "does not apply"),
    ({"rescale_form": "clip", "clip_fraction": 0.5, "clip_calibrate_at": 10}, "does not apply"),
    ({"rescale_form": "clip", "clip_fraction": 0.5, "clip_calibration_window": 10},
     "does not apply"),
    ({"rescale_form": "clip", "clip_threshold": "ema", "clip_fraction": 0.5,
      "clip_ema_horizon": 0}, "clip_ema_horizon"),
    ({"rescale_form": "clip", "clip_threshold": "ema", "clip_fraction": 0.5,
      "clip_calibrate_at": 10}, "does not apply"),
    ({"rescale_form": "clip", "clip_threshold": "fixed", "clip_fraction": 0.5}, "clip_calibrate_at"),
    ({"rescale_form": "clip", "clip_threshold": "fixed", "clip_calibrate_at": 10}, "clip_fraction"),
    ({"rescale_form": "clip", "clip_threshold": "fixed", "clip_fraction": 0.5,
      "clip_calibrate_at": 10, "clip_calibration_window": 0}, "clip_calibration_window"),
    ({"rescale_form": "clip", "clip_threshold": "fixed", "clip_fraction": 0.5,
      "clip_calibrate_at": 10, "clip_ema_horizon": 5}, "does not apply"),
])
def test_meaningless_combinations_are_refused(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AdaFisherMulti(TinyMultiLayerNet(), fisher_mode="ekfac", **kwargs)


@pytest.mark.parametrize("mode", MODES)
def test_clip_has_no_dense_operator(mode: str) -> None:
    model, opt = _run(mode, steps=3, rescale_form="clip", clip_fraction=0.5)
    with pytest.raises(NotImplementedError, match="not a linear operator"):
        opt.approx.f_tilde(model.fc1)


@pytest.mark.parametrize("form,extra", [("floor", {}), ("clip", {"clip_fraction": 0.5})])
def test_p2_reader_refuses_a_non_additive_operator(form: str, extra: dict) -> None:
    """``fisher_ref``'s P2 reader rebuilds ``s + Lambda``; under the other two forms that reading
    would be silently wrong, so it is refused."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from fisher_ref.approx.adafisher_state import snapshot

    model, opt = _run("ekfac", steps=3, rescale_form=form, **extra)
    with pytest.raises(NotImplementedError, match="rescale_form"):
        snapshot(model, opt, mode="ekfac")
