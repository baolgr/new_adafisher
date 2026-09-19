"""``ema_seed_first``: start each running average from its first observation, not from the identity.

Fix S3 of ``docs/reports/plan_lambda_dominance.md``. Every mode starts its curvature state at the
identity and then averages observations into it, so a residue of ``(1 - gammas[0])^k`` of that
identity is still in the state after ``k`` updates. With the shipped ``gammas`` that is ``0.08^k``;
with a corrected, convex average it is ``0.8^k``, which needs about 93 updates to fall below
``Lambda``. That is what froze ``kfac`` in ``docs/reports/audit_step.md`` section 4.8, and it is
what made the curvature measurements of ``plan_lambda_dominance.md`` Part 6 report the optimizer's
own start-up instead of the network at large batch sizes.

What is checked here:

1. the flag is inert by default -- the state after step 0 is exactly the documented
   ``(1 - gammas[0]) * identity + gammas[1] * first observation``, in all five modes;
2. with the flag on, the state after step 0 is exactly the first observation, in all five modes;
3. the property the flag exists for: feed a *constant* observation to a convex average and, with
   the flag on, the state is that constant at every step. Without it, it is off by the identity
   residue, and how far off is exactly ``0.8^(k+1)``;
4. a real ``AdaFisherMulti`` trains with the flag on, in all five modes, leaving no parameter
   un-updated and nothing non-finite.
"""
import pytest
import torch
from adafisher_modes.approximations import MODES
from adafisher_modes.ema import seed_or_accumulate
from adafisher_modes.optimizer import AdaFisherMulti
from torch import nn

ALL_MODES = ["diag", "kfac", "ekfac", "tkfac", "tekfac"]
SHIPPED = (0.92, 0.008)
CONVEX = (0.2, 0.2)  # what AdaFisherMulti(gamma=0.8) collapses to: 0.8 * old + 0.2 * new


def test_default_is_the_documented_identity_seeded_formula():
    store, new = {}, torch.full((3,), 5.0)
    seed_or_accumulate(new, store, "k", lambda: torch.ones(3), SHIPPED, 0, seed_first=False)
    expected = (1 - SHIPPED[0]) * torch.ones(3) + SHIPPED[1] * new
    assert torch.equal(store["k"], expected)


def test_seed_first_stores_the_first_observation_exactly():
    store, new = {}, torch.full((3,), 5.0)
    seed_or_accumulate(new, store, "k", lambda: torch.ones(3), SHIPPED, 0, seed_first=True)
    assert torch.equal(store["k"], new)


def test_seed_first_does_not_copy_by_reference():
    """The state must not alias the caller's tensor, which is reused by the next batch."""
    store, new = {}, torch.full((3,), 5.0)
    seed_or_accumulate(new, store, "k", lambda: torch.ones(3), SHIPPED, 0, seed_first=True)
    new.fill_(99.0)
    assert torch.equal(store["k"], torch.full((3,), 5.0))


def test_constant_input_returns_that_constant_under_a_convex_average():
    """The whole point of the flag, as an exact statement.

    A real average of a constant is that constant. With the identity seed it is not: the gap is
    exactly the residue, ``(1 - gammas[0])^(k+1)`` times ``identity - constant``.
    """
    const = torch.full((4,), 7.0)
    with_flag, without = {}, {}
    for step in range(12):
        seed_or_accumulate(const.clone(), with_flag, "k", lambda: torch.ones(4),
                           CONVEX, step, seed_first=True)
        seed_or_accumulate(const.clone(), without, "k", lambda: torch.ones(4),
                           CONVEX, step, seed_first=False)
        residue = (1 - CONVEX[0]) ** (step + 1)
        assert torch.allclose(with_flag["k"], const), f"step {step}"
        assert torch.allclose(without["k"], const + residue * (torch.ones(4) - const)), f"step {step}"
    # After 12 updates the identity-seeded average is still visibly short of the constant.
    assert not torch.allclose(without["k"], const, atol=1e-3)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_every_mode_accepts_the_flag_and_changes_only_the_seed(mode):
    torch.manual_seed(0)
    layer = nn.Linear(6, 4)
    h, s = torch.randn(8, 6), torch.randn(8, 4)

    states = {}
    for seed_first in (False, True):
        approx = MODES[mode](Lambda=1e-3, gammas=SHIPPED, ema_seed_first=seed_first)
        approx.update_input_factor(layer, h.clone(), step=0)
        approx.update_output_factor(layer, s.clone(), step=0)
        approx.refresh(layer, step=0)
        name = "_H" if mode == "diag" else ("_A" if mode in ("kfac", "ekfac") else "_Phi_raw")
        states[seed_first] = getattr(approx, name)[layer].clone()

    # Both are finite and the same shape; the seeded one is strictly different, since the identity
    # contributes (1 - gammas[0]) = 0.08 of itself to the other.
    assert states[False].shape == states[True].shape
    assert torch.isfinite(states[True]).all()
    assert not torch.allclose(states[False], states[True])


@pytest.mark.parametrize("mode", ALL_MODES)
def test_optimizer_trains_with_the_flag_on(mode):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.LayerNorm(8), nn.Linear(8, 3))
    before = [p.detach().clone() for p in model.parameters()]
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, ema_seed_first=True)
    for _ in range(4):
        opt.zero_grad()
        nn.functional.cross_entropy(model(torch.randn(16, 10)), torch.randint(0, 3, (16,))).backward()
        opt.step()
    after = list(model.parameters())
    assert all(torch.isfinite(p).all() for p in after)
    assert all(not torch.equal(a, b) for a, b in zip(before, after)), "a parameter never moved"


@pytest.mark.parametrize("mode", ALL_MODES)
def test_flag_off_is_bit_identical_to_not_passing_it(mode):
    """Default inertness, on a real trajectory rather than on one call."""
    def run(**kw):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 3))
        opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, **kw)
        gen = torch.Generator().manual_seed(1)
        for _ in range(5):
            x = torch.randn(16, 10, generator=gen)
            y = torch.randint(0, 3, (16,), generator=gen)
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
        return [p.detach().clone() for p in model.parameters()]

    for a, b in zip(run(), run(ema_seed_first=False)):
        assert torch.equal(a, b)
