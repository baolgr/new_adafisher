"""``docs/reports/audit_step.md`` §4.7's two knobs: the ``gamma`` EMA correction and ``diag``'s
``minmax_after_average``. Both must (1) reproduce their intended semantics and (2) be fully inert
at their default, so the bit-exactness tests of ``test_diag_bitexact.py`` stay meaningful.
"""

from __future__ import annotations

import pytest
import torch
from adafisher_modes.approximations import MODES
from adafisher_modes.approximations.diag import DiagApproximation
from adafisher_modes.optimizer import AdaFisherMulti
from conftest import TinyMultiLayerNet, seed_all


def test_gamma_none_leaves_gammas_untouched() -> None:
    """Default: ``gamma=None`` must not perturb whatever ``gammas`` was passed."""
    model = torch.nn.Linear(4, 3)
    opt = AdaFisherMulti(model, fisher_mode="diag", gammas=(0.92, 0.008), gamma=None)
    assert opt.approx.gammas == (0.92, 0.008)


def test_gamma_overrides_gammas_into_eq3_form() -> None:
    """``gamma=g`` must produce ``gammas=(1-g, 1-g)`` — Eq. (3)'s single-coefficient EMA under the
    existing ``(1-gammas[0])*current + gammas[1]*new`` formula.
    """
    model = torch.nn.Linear(4, 3)
    opt = AdaFisherMulti(model, fisher_mode="diag", gammas=(0.92, 0.008), gamma=0.8)
    assert opt.approx.gammas == pytest.approx((0.2, 0.2))


def test_gamma_fixed_point_is_the_true_average_not_1_over_115() -> None:
    """Eq. (3)'s defining property: a constant statistic is reproduced exactly at the fixed point,
    unlike the implemented rule's ``n/115`` (audit_step.md §4.3).
    """
    from adafisher_modes.ema import update_running_avg

    n = torch.full((5,), 7.0)
    current_eq3 = torch.ones(5)
    current_implemented = torch.ones(5)
    gammas_eq3 = (0.2, 0.2)  # gamma = 0.8
    gammas_implemented = (0.92, 0.008)

    for _ in range(5000):  # far past both rules' own settling time
        update_running_avg(n.clone(), current_eq3, gammas_eq3)
        update_running_avg(n.clone(), current_implemented, gammas_implemented)

    assert torch.allclose(current_eq3, n, atol=1e-6), "Eq. (3) must converge to the true constant"
    assert torch.allclose(current_implemented, n / 115, atol=1e-2), (
        "sanity check on the *implemented* rule's own known fixed point, n/115"
    )


@pytest.mark.parametrize("mode", sorted(MODES))
def test_gamma_runs_across_all_five_modes(mode: str) -> None:
    """audit_step.md §4.7's last row: the corrected average applies uniformly to the four
    Kronecker modes too, since ``gamma`` flows through the same ``gammas`` parameter every mode
    already accepts.
    """
    seed_all(0)
    model = TinyMultiLayerNet()
    opt = AdaFisherMulti(model, fisher_mode=mode, gamma=0.8, TCov=1, lr=1e-4)
    x = torch.randn(6, 2, 5, 5)
    for _ in range(3):
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_minmax_after_average_false_is_default_and_bitexact() -> None:
    """Default (``minmax_after_average=False``) must be bit-identical to the pre-existing
    before-the-EMA order (already covered end-to-end by test_diag_eq4_semantics.py; this pins the
    flag's own default).
    """
    approx = DiagApproximation()
    assert approx.minmax_after_average is False


def test_minmax_after_average_erases_the_ema_contraction_before_does_not() -> None:
    """§4.4's core claim, isolated on a toy case, with the *implemented* (uncorrected) EMA rule
    (``gammas=(0.92, 0.008)``, retention 0.08 + weight 0.008 -- coefficients summing to 0.088, not
    1, audit_step.md §4.2): that rule squeezes whatever it accumulates towards ``input/115``.

    Under ``minmax_after_average=False`` (before, the official-code/default order) each step's
    diagonal is normalised into [0, 1] *before* hitting that broken EMA, and nothing renormalises
    the result afterwards -- so after many steps the accumulator itself stays compressed near
    ``1/115`` of its own natural [0, 1] range, and *that* compressed tensor is what ``f_tilde``
    uses directly.

    Under ``minmax_after_average=True`` (after, Algorithm 1's order) the EMA instead accumulates
    the *raw*, unnormalised diagonal -- still compressed by the same broken rule -- but min-max is
    only applied once, at read time (``f_tilde``), on the fully-accumulated result. Since min-max
    normalisation is (up to the epsilon floor) invariant to a uniform multiplicative shrinkage of
    its whole input, that single read-time normalisation restores the full [0, 1] range regardless
    of how compressed the EMA left the raw accumulator -- exactly audit_step.md §4.4's "the factor
    of 115 lands on the input of the normalisation and is erased".
    """
    fc = torch.nn.Linear(4, 8, bias=False)
    seed_all(0)
    raw_inputs = [torch.randn(6, 4) for _ in range(50)]  # past the EMA's own settling time

    approx_before = DiagApproximation(gammas=(0.92, 0.008), minmax_after_average=False)
    approx_after = DiagApproximation(gammas=(0.92, 0.008), minmax_after_average=True)
    for step, h in enumerate(raw_inputs):
        approx_before.update_input_factor(fc, h, step=step)
        approx_after.update_input_factor(fc, h, step=step)

    H_before = approx_before._H[fc]
    H_after_raw = approx_after._H[fc]

    # "before": stuck near the EMA's own compressed range (well under the 7.6% ceiling audit_step.md
    # §4.4 derives for the *combined* Kronecker factor; a single factor's own max is looser but
    # still far from spanning [0, 1]).
    assert H_before.max().item() < 0.5
    # "after": the raw accumulator is compressed identically (same broken EMA, same inputs)...
    assert H_after_raw.max().item() < 0.5
    # ...but a single min-max pass over it, taken at read time exactly as f_tilde() does, restores
    # the full [0, 1] range regardless.
    from adafisher_modes.minmax import min_max_normalization

    H_after_normalised = min_max_normalization(H_after_raw, approx_after.epsilon)
    assert H_after_normalised.max().item() == pytest.approx(1.0, abs=1e-3)
    assert H_after_normalised.min().item() == pytest.approx(0.0, abs=1e-3)


def test_minmax_after_average_runs_end_to_end() -> None:
    seed_all(0)
    model = TinyMultiLayerNet()
    opt = AdaFisherMulti(
        model, fisher_mode="diag", minmax_after_average=True, TCov=1, lr=1e-4, gamma=0.8
    )
    x = torch.randn(6, 2, 5, 5)
    for _ in range(3):
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
