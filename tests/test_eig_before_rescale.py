"""``eig_before_rescale``: measure the rescaling in the basis that is about to be used, not the one
about to be thrown away.

EKFAC's Lemma 1 (``ekfac_1806.03884.pdf``, Appendix A.1) says the diagonal ``D_ii = E[(Q^T grad)_i^2]``
is the best one **for the orthogonal ``Q`` it was measured in**, and for no other. TEKFAC's eq. (3.2)
(``tekfac_2011.13609.pdf``) is the same statement in TKFAC's basis. Algorithm 1 of each paper
therefore orders one iteration eigenbasis first, rescaling second, as does
``reference_repos/EKFAC-pytorch/ekfac.py::step`` (``_compute_kfe`` before ``_precond_intra``).

This package orders it the other way round by default, because the hooks fire before ``step()``:

1. backward hook: project the gradient into the **current** basis, fold it into ``s*``/``Theta``;
2. ``step()`` calls ``refresh``: replace the basis, leave ``s*``/``Theta`` alone;
3. ``step()`` calls ``precondition``: divide using the **new** basis and the **old** ``s*``.

``eig_before_rescale=True`` rebuilds the basis inside the backward hook, before the projection, and
``refresh`` then does not redo it for that step. The knob is off by default, so every trajectory
this repository has produced is reproducible bit for bit.

**How much it matters, measured on a 24-16-6 network with a LayerNorm, two hundred steps.** The
stored rescaling differs between the two orderings by 91% to 141% in relative Frobenius norm. The
*applied* step differs by 3.0e-3 (60 steps, ``TCov=10``) and 3.7e-5 (200 steps, ``TCov=20``) at the
shipped ``Lambda=1e-3`` -- the damping swamps ``s*`` there, so no trained model is affected -- and
by 8.3 and 0.28 at ``Lambda=1e-8``, where it reaches the step in full.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from adafisher_modes import AdaFisherMulti
from adafisher_modes.approximations import EKFACApproximation, TEKFACApproximation
from conftest import seed_all

# (mode, approximation class, input-basis attribute, output-basis attribute, rescaling attribute)
EIGEN_MODES = [
    ("ekfac", EKFACApproximation, "_Q_A", "_Q_B", "_s_star"),
    ("tekfac", TEKFACApproximation, "_Q_Phi", "_Q_Psi", "_Theta"),
]
IDS = [m[0] for m in EIGEN_MODES]


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(12, 8)
        self.ln = nn.LayerNorm(8)
        self.b = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.ln(self.a(x))))


def _spy_class(base, q_in: str, q_out: str):
    """``base`` with one extra behaviour: remember which basis was current at the moment the
    gradient was projected into it. Snapshotting immediately after ``update_output_factor`` returns
    is exact -- nothing inside that call changes the basis after the projection.
    """

    class _Spy(base):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.basis_at_projection: dict = {}

        def update_output_factor(self, module, s, step):  # noqa: ANN001, ANN201
            super().update_output_factor(module, s, step)
            if module in getattr(self, q_in):
                self.basis_at_projection[module] = (
                    getattr(self, q_in)[module].clone(),
                    getattr(self, q_out)[module].clone(),
                )

    return _Spy


def _train(mode, cls, q_in, q_out, *, flag: bool, steps: int = 4, t_eig: int = 1):
    """One short run whose approximation object is a spy. Returns ``(model, optimizer, spy)``."""
    seed_all(0)
    model = _Net()
    kwargs = {"T_eig": t_eig} | ({"T_re": 1} if mode == "tekfac" else {})
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, Lambda=1e-2, TCov=1,
                         eig_before_rescale=flag, **kwargs)
    # The optimizer reads self.approx at call time, in both hooks and in step(), so swapping in a
    # subclass here is enough -- no need to touch the MODES registry.
    spy = _spy_class(cls, q_in, q_out)(Lambda=1e-2, gammas=(0.92, 0.008),
                                       ema_seed_first=False, eig_before_rescale=flag, **kwargs)
    opt.approx = spy
    gen = torch.Generator().manual_seed(5)
    for _ in range(steps):
        opt.zero_grad()
        model(torch.randn(32, 12, generator=gen)).pow(2).sum().backward()
        opt.step()
    return model, opt, spy


# --------------------------------------------------------------------------------------------
# Inertness: the knob must change nothing until it is asked for.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode,cls,q_in,q_out,scale", EIGEN_MODES, ids=IDS)
def test_flag_off_is_bit_identical_to_not_passing_it(mode, cls, q_in, q_out, scale) -> None:
    """Default inertness, on a real trajectory rather than on one call -- the same proof
    ``test_ema_seed_first.py`` demands of its own knob.
    """

    def run(**kwargs) -> list[torch.Tensor]:
        seed_all(0)
        model = _Net()
        cadence = {"T_eig": 2} | ({"T_re": 1} if mode == "tekfac" else {})
        opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, Lambda=1e-2, TCov=1,
                             **cadence, **kwargs)
        gen = torch.Generator().manual_seed(5)
        for _ in range(8):
            opt.zero_grad()
            model(torch.randn(32, 12, generator=gen)).pow(2).sum().backward()
            opt.step()
        return [p.detach().clone() for p in model.parameters()]

    for a, b in zip(run(), run(eig_before_rescale=False)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("mode", ["diag", "kfac", "tkfac"])
def test_flag_is_refused_by_the_modes_that_have_no_eigenbasis(mode: str) -> None:
    """A knob that silently does nothing is worse than one that says so. ``diag``, ``kfac`` and
    ``tkfac`` have no eigenbasis to order against a rescaling.
    """
    model = _Net()
    with pytest.raises(ValueError, match="eig_before_rescale only applies"):
        AdaFisherMulti(model, fisher_mode=mode, eig_before_rescale=True)
    AdaFisherMulti(model, fisher_mode=mode, eig_before_rescale=False)  # the default must be fine


# --------------------------------------------------------------------------------------------
# The property the knob exists for, asserted directly on the two bases.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode,cls,q_in,q_out,scale", EIGEN_MODES, ids=IDS)
def test_on_the_rescaling_is_measured_in_the_basis_precondition_uses(mode, cls, q_in, q_out, scale) -> None:
    """With the knob on, the basis the gradient was projected into is, bit for bit, the basis
    ``precondition`` divides in. Nothing modifies either basis after ``step()`` returns, so reading
    them then is exact rather than a proxy.
    """
    _model, opt, spy = _train(mode, cls, q_in, q_out, flag=True)
    assert spy.basis_at_projection, "the spy never saw a projection"
    for module, (q_in_proj, q_out_proj) in spy.basis_at_projection.items():
        assert torch.equal(q_in_proj, getattr(spy, q_in)[module]), (
            f"{mode}: the input-side basis moved between measuring the rescaling and using it"
        )
        assert torch.equal(q_out_proj, getattr(spy, q_out)[module]), (
            f"{mode}: the output-side basis moved between measuring the rescaling and using it"
        )
    assert len(spy.basis_at_projection) == len(opt.modules)


@pytest.mark.parametrize("mode,cls,q_in,q_out,scale", EIGEN_MODES, ids=IDS)
def test_off_the_rescaling_is_measured_in_a_basis_that_is_then_replaced(mode, cls, q_in, q_out, scale) -> None:
    """The other half of the same statement, and the reason the knob was written.

    This is the test that fails if the two orderings are ever swapped: it pins the default as the
    *stale-basis* one. If a future change silently made the default rebuild the basis before the
    projection, the two bases would agree here and this assertion would fire. The audit found that
    no existing test would have noticed such a swap, because every correctness fixture in this
    suite drives ``refresh`` before re-driving the hooks -- the papers' order, not the shipped one.
    """
    _model, _opt, spy = _train(mode, cls, q_in, q_out, flag=False)
    assert spy.basis_at_projection, "the spy never saw a projection"
    moved = [
        module
        for module, (_q_in_proj, q_out_proj) in spy.basis_at_projection.items()
        if not torch.equal(q_out_proj, getattr(spy, q_out)[module])
    ]
    assert moved, (
        f"{mode}: with eig_before_rescale off, the basis the rescaling was measured in is expected "
        f"to be replaced before precondition uses it. If this now holds, the default ordering has "
        f"been changed -- which is a decision, not a refactor."
    )


@pytest.mark.parametrize("mode,cls,q_in,q_out,scale", EIGEN_MODES, ids=IDS)
def test_the_two_orderings_store_a_different_rescaling(mode, cls, q_in, q_out, scale) -> None:
    """The size of the difference in the stored statistic. Measured at 91%-141% relative on a
    longer run; asserted here only as "clearly nonzero", so the test does not become a thermometer.
    """
    _m0, opt0, _s0 = _train(mode, cls, q_in, q_out, flag=False, steps=6)
    _m1, opt1, _s1 = _train(mode, cls, q_in, q_out, flag=True, steps=6)
    a = torch.cat([v.reshape(-1) for v in getattr(opt0.approx, scale).values()])
    b = torch.cat([v.reshape(-1) for v in getattr(opt1.approx, scale).values()])
    rel = ((b - a).norm() / a.norm()).item()
    assert rel > 1e-3, f"{mode}: the two orderings stored the same rescaling (rel diff {rel:.3e})"


# --------------------------------------------------------------------------------------------
# Mechanics: the knob must move the eigendecomposition, not add one.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode,cls,q_in,q_out,scale", EIGEN_MODES, ids=IDS)
def test_refresh_does_not_redo_an_eigenbasis_the_hook_already_built(mode, cls, q_in, q_out, scale) -> None:
    """``eig_before_rescale`` moves the rebuild earlier in the iteration; it must not run it twice.
    Counted by instrumenting the class's own rebuild, so the count is of real decompositions.
    """
    counts = {True: 0, False: 0}
    for flag in (False, True):
        seed_all(0)
        model = _Net()
        cadence = {"T_eig": 1} | ({"T_re": 1} if mode == "tekfac" else {})
        opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, Lambda=1e-2, TCov=1,
                             eig_before_rescale=flag, **cadence)
        original = type(opt.approx)._rebuild_eigenbasis
        calls = []

        def counted(self, module, _original=original, _calls=calls):  # noqa: ANN001, ANN202
            _calls.append(module)
            return _original(self, module)

        type(opt.approx)._rebuild_eigenbasis = counted  # type: ignore[method-assign]
        try:
            gen = torch.Generator().manual_seed(5)
            for _ in range(4):
                opt.zero_grad()
                model(torch.randn(32, 12, generator=gen)).pow(2).sum().backward()
                opt.step()
        finally:
            type(opt.approx)._rebuild_eigenbasis = original  # type: ignore[method-assign]
        counts[flag] = len(calls)
    assert counts[True] == counts[False], (
        f"{mode}: eig_before_rescale changed the number of eigendecompositions from "
        f"{counts[False]} to {counts[True]}; it is meant to move them, not add any"
    )


# --------------------------------------------------------------------------------------------
# How far the difference reaches, as a function of the damping.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode,cls,q_in,q_out,scale", EIGEN_MODES, ids=IDS)
def test_the_difference_reaches_the_applied_step_only_once_lambda_is_lowered(
    mode, cls, q_in, q_out, scale
) -> None:
    """Why this is a knob and not a fix applied outright.

    Both runs are driven over the same data with the weights resynced after every step, so the two
    orderings always see the same network and the only difference between them is the ordering.
    The applied direction is then read out at two damping levels; ``Lambda`` enters only in
    ``precondition``, so both readings come from the same state.

    At the shipped ``Lambda = 1e-3`` the damping swamps the rescaling and the difference in the
    applied direction is small. Lower ``Lambda`` and it is not. That is the whole content of the
    finding, and the reason fixing the ordering has to come before lowering the damping.
    """
    seed_all(0)
    ref = _Net()
    seed_all(0)
    alt = _Net()
    alt.load_state_dict(ref.state_dict())
    cadence = {"T_eig": 10} | ({"T_re": 1} if mode == "tekfac" else {})
    opt_ref = AdaFisherMulti(ref, lr=1e-3, fisher_mode=mode, Lambda=1e-3, TCov=10,
                             eig_before_rescale=False, **cadence)
    opt_alt = AdaFisherMulti(alt, lr=1e-3, fisher_mode=mode, Lambda=1e-3, TCov=10,
                             eig_before_rescale=True, **cadence)
    gen = torch.Generator().manual_seed(5)
    for _ in range(60):
        x = torch.randn(64, 12, generator=gen)
        y = torch.randint(0, 4, (64,), generator=gen)
        for model, opt in ((ref, opt_ref), (alt, opt_alt)):
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
        alt.load_state_dict(ref.state_dict())  # identical weights going into the next step
    assert all(torch.isfinite(p).all() for p in ref.parameters()), "the reference run diverged"

    def applied(opt, lam: float) -> torch.Tensor:
        saved, opt.approx.Lambda = opt.approx.Lambda, lam
        out = []
        for module in opt.modules:
            w = opt.state[module.weight]["exp_avg"]
            b = None if module.bias is None else opt.state[module.bias]["exp_avg"]
            result = opt.approx.precondition(module, w, b)
            out += [t.reshape(-1) for t in (result if isinstance(result, tuple) else (result,))]
        opt.approx.Lambda = saved
        return torch.cat(out)

    rel = {}
    for lam in (1e-3, 1e-8):
        d_ref, d_alt = applied(opt_ref, lam), applied(opt_alt, lam)
        rel[lam] = ((d_alt - d_ref).norm() / d_ref.norm()).item()

    assert rel[1e-3] < 1e-2, (
        f"{mode}: at the shipped Lambda=1e-3 the ordering should barely reach the applied step; "
        f"got {rel[1e-3]:.3e}"
    )
    assert rel[1e-8] > 100 * rel[1e-3], (
        f"{mode}: lowering Lambda to 1e-8 should let the ordering reach the applied step; got "
        f"{rel[1e-8]:.3e} against {rel[1e-3]:.3e}"
    )
