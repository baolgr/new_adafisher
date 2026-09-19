"""Lot 5 of the Fisher-drift campaign: P2, the operational protocol
(``docs/reports/plan_exp_lot5.md``; ``plan_exp_draft.md`` §3.2, §9, §10.2).

**T12** is the exit criterion: the state reader reproduces the *applied* preconditioner of every
mode on every hooked layer type, and ``diag`` reproduces the upstream optimizer's. Everything else
here guards a specific way this lot could be silently wrong — above all the layout of a
normalisation block, which is interleaved in the optimizer and blocked in the campaign
(``plan_exp_lot5.md`` §0.2), the same class of bug as ``plan_exp_lot2.md`` §5.2's.

Every test is offline and CPU-only.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn
from adafisher_modes import AdaFisherMulti
from conftest import seed_all

from fisher_ref import conventions
from fisher_ref.approx.adafisher_state import (
    MODES,
    norm_permutation,
    snapshot,
    worst_check,
)
from fisher_ref.approx.base import Dense, Diag, rearrange
from fisher_ref.approx.kfac import Kron
from fisher_ref.approx.norm_layers import (
    DIAG_PY_TYPES,
    HOOKED_TYPES,
    diag_py_factor_reading,
    diag_py_reading,
    kron_py_reading,
)
from fisher_ref.conventions import REPO_ROOT as ROOT
from fisher_ref.metrics import lambda_grid, rho, stein_kl
from fisher_ref.rewarm import MIN_REWARM_FACTOR_UPDATES, RewarmSpec

FP64 = torch.float64


class FourKinds(nn.Module):
    """One forward pass over all four module types ``AdaFisherMulti`` hooks."""

    def __init__(self, channels: int = 3, classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.ln = nn.LayerNorm(channels * 4)
        self.fc = nn.Linear(channels * 4, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x)).flatten(1)
        return self.fc(self.ln(x))


def _warm(mode: str, *, steps: int = 3, dtype: torch.dtype = FP64, seed: int = 0,
          **opt_kwargs):
    """A tiny, fully warmed optimizer: every mode's state populated, every cache refreshed."""
    seed_all(seed)
    model = FourKinds().to(dtype)
    kwargs = {"diag": {}, "kfac": {"T_inv": 1}, "ekfac": {"T_eig": 1},
              "tkfac": {"T_inv": 1}, "tekfac": {"T_eig": 1, "T_re": 1}}[mode]
    defaults = {"lr": 0.0, "beta": 0.9, "Lambda": 1e-3, "TCov": 1}
    optimizer = AdaFisherMulti(model, fisher_mode=mode,
                               **{**defaults, **kwargs, **opt_kwargs})
    generator = torch.Generator().manual_seed(seed + 7)
    for _ in range(steps):
        x = torch.randn(8, 2, 2, 2, dtype=dtype, generator=generator)
        y = torch.randint(0, 4, (8,), generator=generator)
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
    return model, optimizer


def _campaign_vector(module: nn.Module, weight: torch.Tensor,
                     bias: torch.Tensor | None) -> torch.Tensor:
    """A (weight, bias) direction flattened into the layout P1 holds that block in.

    ``Linear``/``Conv2d``: the bias-augmented ``rvec([W | b])`` order.
    **Normalisation layer**: ``named_parameters()`` order, ``[gamma; beta]`` — *not* row-major.
    """
    if isinstance(module, DIAG_PY_TYPES):
        assert bias is not None
        return torch.cat([weight.reshape(-1), bias.reshape(-1)])
    flat = weight.reshape(weight.shape[0], -1)
    if bias is not None:
        flat = torch.cat([flat, bias.reshape(-1, 1)], dim=1)
    return flat.reshape(-1)


# ------------------------------------------------------------------------------------------------
# T12
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_t12_snapshot_reproduces_the_applied_preconditioner(mode: str) -> None:
    """T12. ``K.solve(v, 0)`` is exactly what ``approx.precondition`` returns, for every mode and
    every hooked layer type, in the layout P1 holds the block in.

    This compares the **applied** operators, never the bases — ``linalg.eigh`` fixes neither the
    sign nor the ordering of eigenvectors (``CLAUDE.md``, "Known testing pitfalls").
    """
    model, optimizer = _warm(mode)
    state = snapshot(model, optimizer, mode=mode, dtype=FP64, device="cpu")
    assert set(state) == {name for name, m in model.named_modules() if m in optimizer.modules}

    generator = torch.Generator().manual_seed(11)
    for name, module in model.named_modules():
        if name not in state:
            continue
        weight = torch.randn(module.weight.shape, dtype=FP64, generator=generator)
        bias = (None if module.bias is None
                else torch.randn(module.bias.shape, dtype=FP64, generator=generator))
        expected = optimizer.approx.precondition(module, weight, bias)
        parts = expected if isinstance(expected, tuple) else (expected, None)
        wanted = _campaign_vector(module, parts[0], parts[1])

        direction = _campaign_vector(module, weight, bias)
        got = state[name].damped.solve(direction, 0.0)
        gap = float((got - wanted).norm() / wanted.norm())
        assert gap < 1e-10, f"{mode}/{name}: applied preconditioner differs by {gap:.3e}"


def test_t12_diag_matches_the_upstream_optimizer(fisheradaptune_adafisher) -> None:
    """T12's second half: ``diag``'s snapshot reproduces ``FisherAdapTune``'s own ``F~``.

    ``minmax_normalization=False`` is the setting the port is bit-exact under
    (``tests/test_diag_bitexact.py``); the snapshot adds only the layout conversion, which is what
    this checks.
    """
    seed_all(0)
    reference_model = FourKinds().to(FP64)
    ported_model = copy.deepcopy(reference_model)
    kwargs = dict(lr=0.0, beta=0.9, Lambda=1e-3, TCov=1)
    reference = fisheradaptune_adafisher.AdaFisher(reference_model, **kwargs)
    ported = AdaFisherMulti(ported_model, fisher_mode="diag", minmax_normalization=False, **kwargs)

    generator = torch.Generator().manual_seed(3)
    for _ in range(3):
        x = torch.randn(8, 2, 2, 2, dtype=FP64, generator=generator)
        y = torch.randint(0, 4, (8,), generator=generator)
        for model, optimizer in ((reference_model, reference), (ported_model, ported)):
            optimizer.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            optimizer.step()

    state = snapshot(ported_model, ported, mode="diag", dtype=FP64, device="cpu")
    reference_modules = {name: m for name, m in reference_model.named_modules()
                         if m in reference.modules}
    assert set(state) == set(reference_modules)
    for name, module in reference_modules.items():
        upstream = reference._get_F_tilde(module)
        if isinstance(upstream, list):
            flat = torch.cat([upstream[0].reshape(upstream[0].shape[0], -1),
                              upstream[1].reshape(-1, 1)], dim=1)
        else:
            flat = upstream.reshape(upstream.shape[0], -1)
        wanted = _campaign_vector(module, flat[:, :-1] if module.bias is not None else flat,
                                  flat[:, -1] if module.bias is not None else None)
        got = state[name].damped.diag()
        assert torch.allclose(got, wanted, rtol=1e-7, atol=0), name


def test_t12_norm_layout_permutation_is_not_vacuous() -> None:
    """The interleaved -> blocked permutation of §0.2, and that removing it **fails**.

    A ``gamma``-only direction must come back as something whose ``beta`` half is what the optimizer
    says it is. Without the permutation the two halves are shuffled together, which is symmetric,
    positive definite and silently wrong — ``plan_exp_lot2.md`` §5.2's bug in a new place.
    """
    model, optimizer = _warm("kfac")
    state = snapshot(model, optimizer, mode="kfac", dtype=FP64, device="cpu")
    module = model.ln
    channels = int(module.weight.numel())

    generator = torch.Generator().manual_seed(5)
    gamma = torch.randn(channels, dtype=FP64, generator=generator)
    beta = torch.zeros(channels, dtype=FP64)
    expected_w, expected_b = optimizer.approx.precondition(module, gamma, beta)
    wanted = torch.cat([expected_w, expected_b])
    got = state["ln"].damped.solve(torch.cat([gamma, beta]), 0.0)
    assert torch.allclose(got, wanted, rtol=1e-10, atol=1e-14)

    # The same operator without the permutation: it must disagree, or the permutation is untested.
    A, B = optimizer.approx._damped_factors(module)
    unpermuted = Dense(torch.kron(B, A))
    naive = unpermuted.solve(torch.cat([gamma, beta]), 0.0)
    assert float((naive - wanted).norm() / wanted.norm()) > 1e-3

    perm = norm_permutation(channels)
    assert perm.tolist() == [c * 2 for c in range(channels)] + [c * 2 + 1 for c in range(channels)]


@pytest.mark.parametrize("mode", MODES)
def test_undamped_rung_differs_by_lambda_exactly_where_the_damping_is_additive(mode: str) -> None:
    """P2-raw against P2. ``diag``/``ekfac``/``tekfac`` damp additively, so the two differ by
    exactly ``lambda I``; ``kfac``/``tkfac`` use *factored* Tikhonov, which is not ``K + lam I`` for
    any ``lam`` — and the negative half is what makes this test able to catch a mis-wiring.
    """
    model, optimizer = _warm(mode)
    state = snapshot(model, optimizer, mode=mode, dtype=FP64, device="cpu")
    block = state["fc"]
    size = block.damped.P
    gap = block.damped.to_dense() - block.undamped.to_dense()
    additive = torch.eye(size, dtype=FP64) * optimizer.approx.Lambda
    relative = float((gap - additive).norm() / additive.norm())
    if mode in ("diag", "ekfac", "tekfac"):
        assert relative < 1e-8, f"{mode}: expected additive damping, off by {relative:.3e}"
    else:
        assert relative > 1e-6, f"{mode}: factored Tikhonov must not look additive"


def test_conv_sua_is_refused_rather_than_answered_wrongly() -> None:
    """Under SUA the operator is not one ``kron(B~, A~)`` over the full patch direction
    (``plan_lot6.md`` §0.4); producing a plausible wrong answer is the failure mode to avoid."""
    model, optimizer = _warm("kfac", conv_sua=True)
    with pytest.raises(NotImplementedError, match="conv_sua"):
        snapshot(model, optimizer, mode="kfac", dtype=FP64, device="cpu")


@pytest.mark.parametrize("mode", MODES)
def test_undamped_floor_is_recorded_for_every_mode(mode: str) -> None:
    """``0.08^k << lambda`` is the rule for the **damped** operator. P2-raw has ``lambda`` removed,
    so what the identity seed's residue has to be small against is the accumulated factor's own
    smallest eigenvalue — which is therefore measured per layer, not assumed
    (``plan_exp_lot5.md`` §0.6.3b)."""
    from fisher_ref.approx.adafisher_state import least_check, min_eigenvalue

    model, optimizer = _warm(mode)
    state = snapshot(model, optimizer, mode=mode, dtype=FP64, device="cpu")
    floor = least_check(state, "min_eigenvalue_undamped")
    assert floor == floor, f"{mode}: no undamped floor recorded"
    for name, block in state.items():
        direct = min_eigenvalue(block.undamped)
        dense = float(torch.linalg.eigvalsh(block.undamped.to_dense()).min())
        assert direct == pytest.approx(dense, abs=1e-9, rel=1e-6), name


def test_inverse_consistency_detects_a_stale_vintage_and_not_bad_conditioning() -> None:
    """``precondition`` uses the inverse cached at the last ``refresh``; the snapshot builds the
    operator from the current factors. With ``TCov == T_inv`` they are the same vintage — and the
    snapshot **measures** that rather than assuming it.

    The check must be blind to conditioning: this campaign's A1 has an input factor of rank 646 out
    of 785, and a guard that tripped on it would abort a twelve-hour job for a non-reason. So both
    halves are pinned: a same-vintage pair agrees even when the factor is singular, and a stale one
    is caught.
    """
    model, optimizer = _warm("kfac")
    state = snapshot(model, optimizer, mode="kfac", dtype=FP64, device="cpu")
    for key in ("inverse_consistency_A", "inverse_consistency_B"):
        assert worst_check(state, key) < 1e-8

    # Singular factors, same vintage: still consistent, because the comparison is against a fresh
    # inverse of the same matrix rather than against the identity.
    model, optimizer = _warm("kfac", Lambda=0.0, gammas=[1.0, 1.0], steps=1)
    singular = snapshot(model, optimizer, mode="kfac", dtype=FP64, device="cpu")
    assert worst_check(singular, "inverse_consistency_B") < 1e-8

    # A stale inverse is caught, and the snapshot refuses rather than reporting a wrong operator.
    module = model.fc
    optimizer.approx._B_inv[module] = torch.eye(
        optimizer.approx._B_inv[module].shape[0], dtype=FP64)
    with pytest.raises(RuntimeError, match="inverse_consistency"):
        snapshot(model, optimizer, mode="kfac", dtype=FP64, device="cpu")


# ------------------------------------------------------------------------------------------------
# k_lam: the one change to the metrics
# ------------------------------------------------------------------------------------------------


def test_k_lam_default_is_inert() -> None:
    """``rho``/``stein_kl`` with ``k_lam=None`` are bit-identical to passing ``lam`` explicitly,
    which is what they did before lot 5."""
    generator = torch.Generator().manual_seed(2)
    root = torch.randn(40, 12, dtype=FP64, generator=generator)
    R = root.T @ root / 40
    K = Diag(torch.rand(12, dtype=FP64, generator=generator) + 0.1)
    gradient = torch.randn(12, dtype=FP64, generator=generator)
    for _, lam in lambda_grid(R, (1e-3, 1e-1)).items():
        assert rho(R, K, gradient, lam).rho == rho(R, K, gradient, lam, k_lam=lam).rho
        assert stein_kl(R, K, lam).kl == stein_kl(R, K, lam, k_lam=lam).kl


def test_k_lam_zero_applies_the_operator_as_it_stands() -> None:
    """``k_lam=0`` makes ``rho`` use ``K^{-1} g`` while the reference stays damped at ``lam`` —
    the whole point of the argument (``plan_exp_lot5.md`` §0.3)."""
    generator = torch.Generator().manual_seed(4)
    root = torch.randn(40, 12, dtype=FP64, generator=generator)
    R = root.T @ root / 40
    values = torch.rand(12, dtype=FP64, generator=generator) + 0.1
    K = Diag(values)
    gradient = torch.randn(12, dtype=FP64, generator=generator)
    lam = 0.5
    report = rho(R, K, gradient, lam, k_lam=0.0)
    direction = gradient / values
    expected_cos = float(direction @ torch.linalg.solve(R + lam * torch.eye(12, dtype=FP64),
                                                        gradient))
    expected_cos /= float(direction.norm() * torch.linalg.solve(
        R + lam * torch.eye(12, dtype=FP64), gradient).norm())
    assert report.cos_directions == pytest.approx(expected_cos, rel=1e-12)


# ------------------------------------------------------------------------------------------------
# The P1-py rung
# ------------------------------------------------------------------------------------------------


def _probe_batch(n: int = 16):
    generator = torch.Generator().manual_seed(9)
    return (torch.randn(n, 2, 2, 2, dtype=FP64, generator=generator),
            torch.randint(0, 4, (n,), generator=generator))


def test_diag_py_reading_default_types_are_lot_threes() -> None:
    """Extending the reading to every hooked layer must not change what lot 3 asked for."""
    seed_all(0)
    model = FourKinds().to(FP64)
    inputs, targets = _probe_batch()
    modules = dict(model.named_modules())
    loss = nn.CrossEntropyLoss()
    default = diag_py_reading(model, inputs, targets, modules, loss_fn=loss, batch_size=8)
    assert set(default) == {"bn", "ln"}
    extended = diag_py_reading(model, inputs, targets, modules, loss_fn=loss, batch_size=8,
                               types=HOOKED_TYPES)
    assert set(extended) == {"conv", "bn", "ln", "fc"}
    for name in default:
        assert torch.equal(default[name].values, extended[name].values)


def test_kron_py_reading_layout_and_agreement_with_the_optimizer() -> None:
    """``kron_py`` is ``kron(compute_s_full, compute_h_full)`` in the campaign's layout, and on a
    ``Linear`` its diagonal is ``diag_py``'s reading of the same micro-batches."""
    seed_all(0)
    model = FourKinds().to(FP64)
    inputs, targets = _probe_batch()
    modules = dict(model.named_modules())
    loss = nn.CrossEntropyLoss()
    blocks = kron_py_reading(model, inputs, targets, modules, loss_fn=loss, batch_size=8)
    assert set(blocks) == {"conv", "bn", "ln", "fc"}
    assert isinstance(blocks["fc"], Kron) and isinstance(blocks["ln"], Dense)
    # A normalisation layer's input factor is 2 x 2 (plan_lot5.md §0.2), so the block is 2C x 2C.
    channels = int(model.ln.weight.numel())
    assert blocks["ln"].P == 2 * channels

    # On a `Linear`, compute_h_diag IS the diagonal of compute_h_full, so the factor-averaged
    # diagonal reading must be the diagonal of kron_py entry for entry. The *product*-averaged one
    # (lot 3's diag_py) must NOT be, with more than one micro-batch — that gap is a real difference
    # in how the micro-batches are combined, not a tolerance (plan_exp_lot5.md §5).
    factors = diag_py_factor_reading(model, inputs, targets, modules, loss_fn=loss, batch_size=8)
    assert torch.allclose(blocks["fc"].diag(), factors["fc"].values, rtol=1e-10, atol=0)
    product = diag_py_reading(model, inputs, targets, modules, loss_fn=loss, batch_size=8,
                              types=HOOKED_TYPES)
    gap = float((product["fc"].values - factors["fc"].values).norm()
                / factors["fc"].values.norm())
    assert gap > 1e-6, "with two micro-batches the two averaging orders must differ"


def test_diag_py_averaging_orders_coincide_on_one_micro_batch() -> None:
    """The two conventions are the same statistic seen once; with a single micro-batch there is
    nothing to average, so they must agree exactly."""
    seed_all(0)
    model = FourKinds().to(FP64)
    inputs, targets = _probe_batch(n=8)
    modules = dict(model.named_modules())
    loss = nn.CrossEntropyLoss()
    product = diag_py_reading(model, inputs, targets, modules, loss_fn=loss, batch_size=8,
                              types=HOOKED_TYPES)
    factors = diag_py_factor_reading(model, inputs, targets, modules, loss_fn=loss, batch_size=8)
    assert set(product) == set(factors)
    for name in product:
        assert torch.allclose(product[name].values, factors[name].values, rtol=1e-12, atol=0), name


def test_kron_py_normalisation_block_matches_the_optimizers_own_operator() -> None:
    """The only object in this campaign that reads a normalisation layer the way the optimizer
    does: with ``gammas=(1, 1)``, one micro-batch and ``Lambda=0``, ``kron_py`` *is* ``kfac``'s
    operational block (``plan_exp_lot5.md`` §0.4)."""
    seed_all(0)
    model = FourKinds().to(FP64)
    inputs, targets = _probe_batch(n=8)
    modules = dict(model.named_modules())
    loss = nn.CrossEntropyLoss()
    reading = kron_py_reading(model, inputs, targets, modules, loss_fn=loss, batch_size=8)

    seed_all(0)
    replica = FourKinds().to(FP64)
    optimizer = AdaFisherMulti(replica, lr=0.0, Lambda=0.0, gammas=[1.0, 1.0], TCov=1,
                               fisher_mode="kfac", T_inv=1)
    # eval mode, like every reference this campaign builds: in train mode a BatchNorm2d normalises
    # by batch statistics, so its output — and every activation and gradient downstream of it —
    # is a different function. That difference is exactly what the P1-py -> P2-raw rung measures;
    # here it would just make the two sides incomparable.
    replica.eval()
    optimizer.zero_grad()
    loss(replica(inputs), targets).backward()
    optimizer.step()
    state = snapshot(replica, optimizer, mode="kfac", dtype=FP64, device="cpu")
    for name in ("ln", "bn", "fc", "conv"):
        gap = float((reading[name].to_dense() - state[name].damped.to_dense()).norm()
                    / state[name].damped.to_dense().norm())
        assert gap < 1e-10, f"{name}: kron_py and the optimizer's own operator differ by {gap:.3e}"


# ------------------------------------------------------------------------------------------------
# The TEKFAC structure P1 was missing
# ------------------------------------------------------------------------------------------------


def _tiny_reference(model, inputs, targets, modules):
    from fisher_ref import folds as fold_engine
    from fisher_ref.reference import ParamLayout, reference_parameter_names

    layout = ParamLayout.of(model, reference_parameter_names(model, modules, []))
    plan = fold_engine.FoldPlan(n_probes=int(inputs.shape[0]), batch_size=8, folds=1)
    folds, _ = fold_engine.first_pass(model, inputs, targets, source="type2", modules=modules,
                                      layout=layout, plan=plan, dtype=FP64, device="cpu",
                                      sharing=False, check_rows=False)
    bases = fold_engine.eigenbases(fold_engine.assemble(folds, [0], symmetrize=False).factors,
                                   tkfac_basis=True)
    fold_engine.second_pass(model, inputs, targets, folds, source="type2", modules=modules,
                            bases=bases, plan=plan, dtype=FP64, device="cpu")
    return layout, fold_engine.assemble(folds, [0], bases=bases)


def test_tekfac_basis_is_off_by_default_and_preserves_the_trace() -> None:
    """T9 for lot 5's new basis: an orthogonal change of basis preserves the Frobenius norm, so
    ``tr(K_tekfac) = tr(B_l)`` — and the basis is absent unless asked for."""
    seed_all(0)
    model = FourKinds().to(FP64)
    inputs, targets = _probe_batch(n=16)
    modules = {name: m for name, m in model.named_modules()
               if isinstance(m, HOOKED_TYPES)}
    layout, assembled = _tiny_reference(model, inputs, targets, modules)
    from fisher_ref import folds as fold_engine

    plain = fold_engine.eigenbases(assembled.factors)
    assert fold_engine.TKFAC_BASIS not in plain
    assert fold_engine.TKFAC_BASIS in assembled.ekfac

    from fisher_ref.reference import to_augmented
    for name in ("fc", "conv"):
        columns = layout.block_slice(name)
        exact = to_augmented(assembled.matrix[columns, columns],
                             layout.augmented_permutation(name))
        tekfac = assembled.ekfac[fold_engine.TKFAC_BASIS][name]
        assert float(tekfac.trace()) == pytest.approx(float(exact.diagonal().sum()), rel=1e-10)


def test_tekfac_dominates_tkfac_in_frobenius_norm() -> None:
    """TEKFAC Thm 3.1: the optimal diagonal in TKFAC's own eigenbasis beats any other diagonal in
    that basis, and TKFAC's own eigenvalues are one such other diagonal. A theorem, so a violation
    is a bug — which is why it joins ``p1_structural.THEOREMS``."""
    from fisher_ref import folds as fold_engine
    from fisher_ref.metrics import frobenius
    from fisher_ref.reference import to_augmented
    from fisher_ref.runners.p1_structural import THEOREMS

    assert ("tekfac", "tkfac") in THEOREMS
    seed_all(0)
    model = FourKinds().to(FP64)
    inputs, targets = _probe_batch(n=32)
    modules = {name: m for name, m in model.named_modules() if isinstance(m, HOOKED_TYPES)}
    layout, assembled = _tiny_reference(model, inputs, targets, modules)
    for name in ("fc", "conv"):
        columns = layout.block_slice(name)
        exact = to_augmented(assembled.matrix[columns, columns],
                             layout.augmented_permutation(name))
        tkfac = assembled.factors["expand"][name].tkfac()
        tekfac = assembled.ekfac[fold_engine.TKFAC_BASIS][name]
        assert frobenius(exact, tekfac).e_F <= frobenius(exact, tkfac).e_F + 1e-12


# ------------------------------------------------------------------------------------------------
# ExtraStructures inertness
# ------------------------------------------------------------------------------------------------


def test_extra_structures_default_is_inert() -> None:
    """``analyse_block`` and ``half_values`` with no extras return exactly what they returned before
    lot 5 — the 609-test suite is the wider check, this is the direct one."""
    from fisher_ref.runners.p1_structural import EMPTY_EXTRAS, Block, analyse_block, structures_of

    generator = torch.Generator().manual_seed(6)
    root = torch.randn(60, 12, dtype=FP64, generator=generator)
    exact = root.T @ root / 60
    block = Block("layer", "embed", 1, exact, shape=(4, 3))
    rearranged = rearrange(exact, 4, 3)
    without, rows_without = structures_of(block, rearranged)
    with_empty, rows_with = structures_of(block, rearranged, extra=EMPTY_EXTRAS)
    assert set(without) == set(with_empty)
    assert [r["metric"] for r in rows_without] == [r["metric"] for r in rows_with]

    common = {"layer": "layer", "layer_type": "other"}
    gradient = torch.randn(12, dtype=FP64, generator=generator)
    base, _ = analyse_block(block, alphas=[1e-2], gradient=gradient, stein_max_p=4096,
                            rho_max_p=4096, common=dict(common))
    same, _ = analyse_block(block, alphas=[1e-2], gradient=gradient, stein_max_p=4096,
                            rho_max_p=4096, common=dict(common), extra=EMPTY_EXTRAS)
    assert [(r["structure"], r["metric"], r["value"]) for r in base] == \
           [(r["structure"], r["metric"], r["value"]) for r in same]


def test_extra_structures_are_measured_labelled_and_self_damped() -> None:
    """An injected structure appears in the rows, carries its own protocol label, and — when it is
    self-damped — is applied with ``k_lam=0`` rather than the sweep's ``lambda``."""
    from fisher_ref.runners.p1_structural import Block, ExtraStructures, analyse_block

    generator = torch.Generator().manual_seed(8)
    root = torch.randn(60, 12, dtype=FP64, generator=generator)
    exact = root.T @ root / 60
    block = Block("layer", "embed", 1, exact, shape=(4, 3))
    operator = Diag(torch.rand(12, dtype=FP64, generator=generator) + 0.5)
    extra = ExtraStructures(blocks={"layer": {"p2_kfac": operator}},
                            self_damped=frozenset({"p2_kfac"}),
                            protocol={"p2_kfac": "P2"})
    common = {"layer": "layer", "layer_type": "other", "protocol": "P1"}
    gradient = torch.randn(12, dtype=FP64, generator=generator)
    rows, _ = analyse_block(block, alphas=[1e-2], gradient=gradient, stein_max_p=4096,
                            rho_max_p=4096, common=dict(common), extra=extra)
    labels = {r["protocol"] for r in rows if r["structure"] == "p2_kfac"}
    assert labels == {"P2"}
    assert {r["protocol"] for r in rows if r["structure"] == "exact_diag"} == {"P1"}
    # M3 is skipped for the extras by default, and the skip is in the data.
    skips = [r for r in rows if r["metric"] == "stein_kl_skipped_protocol"]
    assert [r["structure"] for r in skips] == ["p2_kfac"]
    # rho used k_lam = 0: reproduce it by hand.
    alpha, lam = next(iter(lambda_grid(exact, [1e-2]).items()))
    wanted = rho(exact, operator, gradient, lam, k_lam=0.0).rho
    got = [r["value"] for r in rows
           if r["structure"] == "p2_kfac" and r["metric"] == "rho" and r["lambda_alpha"] == alpha]
    assert got == [wanted]


# ------------------------------------------------------------------------------------------------
# The re-warm
# ------------------------------------------------------------------------------------------------


def test_rewarm_refuses_a_short_warm_with_the_reason() -> None:
    """``0.08^k << lambda``, not ``0.08^k << 1``: at ``k = 3`` the identity seed is half of
    ``lambda`` and the applied preconditioner was measured 12-87 % wrong
    (``plan_exp_draft.md`` §3.2)."""
    from benchmarks.common.optimizers import HParams
    from fisher_ref.rewarm import rewarm

    with pytest.raises(ValueError, match="0.08"):
        rewarm(None, None, 1.0, RewarmSpec(mode="kfac", steps=300), hparams=HParams(tcov=100),
               data_root=".")
    assert MIN_REWARM_FACTOR_UPDATES == 10
    assert 0.08 ** MIN_REWARM_FACTOR_UPDATES < 1e-3 * 1e-5


def test_rewarm_spec_rejects_an_unknown_data_source() -> None:
    with pytest.raises(ValueError, match="data must be one of"):
        RewarmSpec(mode="kfac", steps=1000, data="augmented")


def test_identity_seed_residue_is_what_the_rule_says() -> None:
    """The arithmetic the re-warm length is set by, as a number rather than a claim."""
    assert 0.08 ** 3 == pytest.approx(5.12e-4, rel=1e-3)
    assert 0.08 ** 3 > 0.5 * 1e-3          # half of lambda at k = 3
    assert 0.08 ** 10 < 1e-3 * 1e-6        # negligible against lambda at k = 10
    assert math.log(1e-3 / 100) / math.log(0.08) < MIN_REWARM_FACTOR_UPDATES


def test_scale_free_rescaling_covers_every_structure_kind() -> None:
    """``rho`` on a scale-free reading is evaluated on ``c* K``; lot 5's readings are ``Kron`` and
    ``Dense``, not only ``Diag``, so the rescaling has to cover them.

    Measured on the real ``cnn_gn_cifar`` checkpoint: ``kron_py``'s ``c*`` is ``1.6e4``, i.e. its
    raw scale is ``1/batch^2 = 1/16 384`` of the reference's (``CLAUDE.md`` §4.3). Without the
    rescaling its ``rho`` is the plain-gradient step's, whatever its shape.
    """
    from fisher_ref.approx.ekfac import EKFAC
    from fisher_ref.runners.p1_structural import SCALE_FREE, rescaled

    assert set(SCALE_FREE) == {"diag_py", "diag_py_factors", "kron_py", "kron_py_1batch"}
    generator = torch.Generator().manual_seed(12)
    A = torch.rand(3, 3, dtype=FP64, generator=generator)
    A = A @ A.T + torch.eye(3, dtype=FP64)
    G = torch.rand(2, 2, dtype=FP64, generator=generator)
    G = G @ G.T + torch.eye(2, dtype=FP64)
    QA, QG = torch.linalg.eigh(A)[1], torch.linalg.eigh(G)[1]
    blocks = [Diag(torch.rand(6, dtype=FP64, generator=generator) + 0.1),
              Kron(A=A, G=G),
              EKFAC(QA=QA, QG=QG, s=torch.rand(2, 3, dtype=FP64, generator=generator) + 0.1),
              Dense(torch.eye(6, dtype=FP64) * 3.0)]
    for block in blocks:
        scaled = rescaled(block, 7.0)
        assert type(scaled) is type(block)
        assert torch.allclose(scaled.to_dense(), 7.0 * block.to_dense(), rtol=1e-12, atol=0)
    with pytest.raises(TypeError):
        rescaled(object(), 2.0)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------------
# The runner, end to end on a real checkpoint
# ------------------------------------------------------------------------------------------------


def _first_available_run():
    """The smallest real run this machine has, or ``None`` — these tests are skipped without data."""
    from fisher_ref.checkpoints import discover_runs

    for model in ("cnn_gn_cifar", "vit_micro_cifar", "mlp_ln_mnist"):
        runs = discover_runs(model=model, arm="diag", seed=0)
        if runs and 0.5 in runs[0].checkpoints:
            return runs[0]
    return None


def test_rewarm_freezes_theta_and_lets_batchnorm_buffers_move() -> None:
    """``lr = 0`` must leave every parameter bit-identical — which is what makes "the state was
    estimated at exactly the checkpoint's weights" a fact rather than a hope. The **buffers** are
    expected to move: a ``BatchNorm2d`` in train mode rewrites its running statistics on every
    forward, which is precisely why the reference uses a second model (``plan_exp_lot5.md`` §0.6.2).
    """
    from benchmarks.common.runner import discover_benchmarks
    from fisher_ref.checkpoints import load_theta
    from fisher_ref.rewarm import RewarmSpec, hparams_of, rewarm

    run = _first_available_run()
    if run is None:
        pytest.skip("no trajectory checkpoints on this machine")
    bench = discover_benchmarks()[run.bench_name]
    hparams = hparams_of(run, bench)
    before = {name: p.detach().clone()
              for name, p in load_theta(run, 0.5).model.named_parameters()}
    model, _optimizer = rewarm(bench, run, 0.5, RewarmSpec(mode="kfac", steps=4, seed=3),
                               hparams=hparams, data_root=str(ROOT / "benchmarks" / "data"),
                               device="cpu", num_workers=0, allow_short=True)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), before[name]), name


@pytest.mark.parametrize("mode", MODES)
def test_snapshot_covers_exactly_the_modules_the_optimizer_hooks(mode: str) -> None:
    """Every hooked module gets an operator; nothing else does. The failure mode is silence — a
    block that quietly never receives one is simply missing from the table."""
    model, optimizer = _warm(mode)
    state = snapshot(model, optimizer, mode=mode, dtype=FP64, device="cpu")
    hooked = {name for name, m in model.named_modules() if m in optimizer.modules}
    assert set(state) == hooked == {"conv", "bn", "ln", "fc"}
    for name, block in state.items():
        parameters = sum(p.numel() for p in dict(model.named_modules())[name]
                         .parameters(recurse=False))
        assert block.damped.P == parameters, name


def test_runner_writes_every_rung_and_gates_the_variants(tmp_path) -> None:
    """The schema §0.9's rules consume: all four ``protocol`` values, ``p2_identity`` on a block the
    optimizer does not hook, fold intervals on the P2 rows, and the diagnostics **only** at the
    fraction asked for. Runs the real CLI on a real checkpoint, restricted to stay fast."""
    from fisher_ref.runners import p2_operational

    run = _first_available_run()
    if run is None:
        pytest.skip("no trajectory checkpoints on this machine")
    if run.model != "cnn_gn_cifar":
        pytest.skip("this assertion names cnn_gn_cifar's own unhooked GroupNorm")
    out = tmp_path / "p2"
    p2_operational.main([
        "--model", "cnn_gn_cifar", "--arm", "diag", "--fractions", "0.5", "--sources", "type2",
        "--probes", "256", "--batch", "128", "--folds", "2", "--partitions", "1",
        "--modules", "features.0", "features.1", "--modes", "diag", "kfac",
        "--rewarm-steps", "2", "--allow-short-rewarm", "--rewarm-workers", "0",
        "--variants", "replica", "--variants-at", "0.5", "--alphas", "1e-1",
        "--device", "cpu", "--out-dir", str(out),
    ])
    import csv as _csv
    rows = list(_csv.DictReader(
        (out / "cnn_gn_cifar" / "diag" / "seed0" / "0.5" / "metrics.csv").open()))
    protocols = {row["protocol"] for row in rows}
    assert {"P1", "P1-py", "P2", "P2-raw"} <= protocols, protocols
    structures = {row["structure"] for row in rows}
    assert "p2_identity" in structures            # features.1 is a GroupNorm: not hooked
    assert {"p2_diag", "p2_kfac", "p2_raw_diag", "p2_raw_kfac"} <= structures
    assert any(s.startswith("p2rep_") for s in structures)
    assert {row["layer"] for row in rows if row["structure"] == "p2_identity"} == {"features.1"}
    with_ci = [row for row in rows
               if row["structure"].startswith("p2_") and row["metric"] == "e_F_star"
               and row["ci_low"]]
    assert with_ci, "P2 rows must carry the reference's fold interval"
    # The paired difference lot 3's machinery emits, now for HF7's pairs.
    assert any(row["structure"] == "p2_kfac-kfac" for row in rows)


# ------------------------------------------------------------------------------------------------
# HF7's decision rule (plan_exp_lot5.md §0.9)
# ------------------------------------------------------------------------------------------------


def _decision_cells(builder, count: int = 15, seed: int = 0):
    """``count`` cells built by ``builder(rng) -> (p1, p2, replica)``, folded into one record."""
    import random

    from fisher_ref.experiments.lot5_decisions import (
        PRIMARY_PAIRS,
        degenerate,
        kendall_tau_b,
        operator_noise,
    )
    from fisher_ref.experiments.lot5_decisions import (
        median as drop_nan_median,
    )

    rng = random.Random(seed)
    record = {"tau": [], "tau_reproducible": [], "degenerate": 0, "dropped": 0, "cells": 0,
              "delta": [], "spread": [], "floor": []}
    for _ in range(count):
        p1, p2, replica = builder(rng)
        values = {}
        for (a, b), x, y, z in zip(PRIMARY_PAIRS, p1, p2, replica):
            values[a], values[b] = x, y
            values[b.replace("p2_", "p2rep_", 1)] = z
        noise = operator_noise(values, PRIMARY_PAIRS)
        record["cells"] += 1
        if degenerate(values, PRIMARY_PAIRS, noise):
            record["degenerate"] += 1
            continue
        record["tau"].append(kendall_tau_b(p1, p2))
        record["tau_reproducible"].append(kendall_tau_b(p2, replica))
        record["delta"].append(drop_nan_median([values[b] - values[a] for a, b in PRIMARY_PAIRS]))
        record["spread"].append(max(p1) - min(p1))
    return record


def test_hf7_is_not_confirmed_by_a_degenerate_operational_operator() -> None:
    """The rule must not announce HF7 confirmed when the P2 side carries no ranking at all.

    Both `e_F_star` and `rho` are invariant under `K -> cK`, and `plan_exp_lot5.md` §5.4 **measured**
    `cond(F~)` between 1.00 and 1.11 — so the five modes' values collapse onto one and their order
    is a batch draw. A rank correlation against noise is ~0, which a naive rule reads as "the
    rankings differ": simulated, that gave CONFIRMED in 98.7 % of draws, with no evidence at all.
    The degeneracy gate and the reproducibility ceiling are what stop it.
    """
    from fisher_ref.experiments.lot5_decisions import verdict

    def degenerate_cell(rng):
        p1 = [rng.random() for _ in range(4)]
        p2 = [0.99 + rng.gauss(0, 1e-4) for _ in range(4)]
        replica = [0.99 + rng.gauss(0, 1e-4) for _ in range(4)]
        return p1, p2, replica

    confirmed = sum(verdict(_decision_cells(degenerate_cell, seed=s)).startswith("CONFIRMED")
                    for s in range(60))
    assert confirmed <= 3, f"the degenerate P2 side was called CONFIRMED in {confirmed}/60 draws"


def test_hf7_still_detects_a_real_disagreement_and_a_real_agreement() -> None:
    """The gates must not make the rule inert: a reproducible P2 ranking that differs from P1 is
    confirmed, and one that agrees is refuted."""
    from fisher_ref.experiments.lot5_decisions import verdict

    def differs(rng):
        p1 = [rng.random() for _ in range(4)]
        truth = [rng.random() for _ in range(4)]
        return p1, [v + rng.gauss(0, 1e-3) for v in truth], [v + rng.gauss(0, 1e-3) for v in truth]

    def agrees(rng):
        p1 = [rng.random() for _ in range(4)]
        return p1, [v + rng.gauss(0, 1e-3) for v in p1], [v + rng.gauss(0, 1e-3) for v in p1]

    assert sum(verdict(_decision_cells(differs, seed=s)).startswith("CONFIRMED")
               for s in range(20)) >= 12
    assert sum(verdict(_decision_cells(agrees, seed=s)).startswith("REFUTED")
               for s in range(20)) >= 18


def test_decision_median_drops_nan_instead_of_being_poisoned_by_it() -> None:
    """``statistics.median`` sorts, and a ``nan`` makes the order meaningless — it returns ``0.9``
    for a list whose finite median is ``0.85``, silently. A fully-tied P2 side produces exactly such
    a ``nan`` from ``kendall_tau_b`` (zero denominator)."""
    import statistics

    from fisher_ref.experiments.lot5_decisions import kendall_tau_b, median

    poisoned = [0.9, 0.8, float("nan"), 0.7, 0.95]
    assert statistics.median(poisoned) == 0.9          # the trap
    assert median(poisoned) == pytest.approx(0.85)     # the fix
    assert kendall_tau_b([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) != kendall_tau_b([1.0], [1.0])
    tied = kendall_tau_b([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    assert tied != tied, "an all-tied side must give nan, not a number"


def test_decision_cell_key_keeps_arm_and_seed() -> None:
    """Without them a second seed's or a second arm's CSV silently overwrites the first, and
    `plan_exp_draft.md` §10.3 requires every aggregate to carry the count it was built from."""
    from fisher_ref.experiments.lot5_decisions import index

    base = {"layer": "fc", "fraction": "1.0", "source": "E_hat", "metric": "e_F_star",
            "lambda_alpha": "", "structure": "p2_kfac", "value": "0.5"}
    rows = [{**base, "model": "m", "arm": "diag", "seed": "0"},
            {**base, "model": "m", "arm": "adamw", "seed": "0", "value": "0.7"},
            {**base, "model": "m", "arm": "diag", "seed": "1", "value": "0.9"}]
    assert len(index(rows)) == 3


def test_decision_vocabularies_match_what_the_runner_writes() -> None:
    """The reader's constants must speak the writer's language.

    Found the hard way: the first version compared the CSV's ``source`` column against ``"E_hat"``
    (which lives in the ``reference`` column) and restricted the primary verdict to
    ``("linear", "conv")`` — while ``registry.LAYER_TYPES`` calls an unshared ``Linear`` ``linear``,
    a token-wise one ``linear_shared`` and the output module ``head``. Both mistakes produce an
    **empty** report, not an error.
    """
    from fisher_ref import registry
    from fisher_ref.experiments.lot5_decisions import (
        PRIMARY_KINDS,
        PRIMARY_SOURCE,
        collect,
        index,
        layer_kinds,
        verdict,
    )
    from fisher_ref.sources import SOURCES

    assert PRIMARY_SOURCE in SOURCES
    assert set(PRIMARY_KINDS) <= set(registry.LAYER_TYPES)
    # Every hooked, non-normalisation layer kind of this lot's four models must be in PRIMARY_KINDS,
    # or its cells are dropped without a word.
    assert {"linear", "linear_shared", "conv", "head"} <= set(PRIMARY_KINDS)

    from fisher_ref.experiments.lot5_decisions import PRIMARY_PAIRS
    base = {"layer": "fc", "layer_type": "head", "fraction": "1.0", "source": PRIMARY_SOURCE,
            "metric": "e_F_star", "lambda_alpha": "", "model": "m", "arm": "diag", "seed": "seed0"}
    rows = []
    for index_, (p1_name, p2_name) in enumerate(PRIMARY_PAIRS):
        rows.append({**base, "structure": p1_name, "value": str(0.1 * index_)})
        rows.append({**base, "structure": p2_name, "value": str(0.9 - 0.1 * index_)})
        rows.append({**base, "structure": p2_name.replace("p2_", "p2rep_", 1),
                     "value": str(0.9 - 0.1 * index_ + 1e-4)})
    record = collect(index(rows), PRIMARY_PAIRS, layer_kinds(rows), PRIMARY_KINDS, PRIMARY_SOURCE)
    assert "m" in record and record["m"]["cells"] == 1, record
    # The P1 and P2 orderings are exactly reversed and the replica reproduces: HF7 confirmed.
    assert verdict(record["m"]).startswith("CONFIRMED")


def test_metrics_version_records_the_new_zoo() -> None:
    assert conventions.METRICS_VERSION == "fisher_ref/0.3"
