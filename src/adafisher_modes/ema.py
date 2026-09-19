"""Running average of a raw Kronecker-factor statistic (AdaFisher Eq. 3).

Every mode keeps its curvature statistics as a running average over training steps. All of them go
through the two functions here, so the smoothing rule is the same in every mode and only the
statistic being smoothed differs.

The update is ``current <- (1 - gammas[0]) * current + gammas[1] * new``. With the shipped default
``gammas = (0.92, 0.008)`` that is ``0.08 * current + 0.008 * new``.

**The two coefficients deliberately do not sum to 1.** 0.08 + 0.008 = 0.088, so a constant input
``x`` is not reproduced: the average settles at ``x * 0.008 / (1 - 0.08) = x / 115``. About 92% of
the stored value comes from the single most recent observation. This is what both reference
implementations compute -- ``reference_repos/FisherAdapTune/scripts/adafisher.py``'s
``_update_running_avg``, and the official AdaFisher repository's ``update_running_avg``, which
writes the same two numbers as ``gamma * 1e-1`` and ``gamma * 1e-2`` at its default ``gamma = 0.8``.
Equation (3) of the AdaFisher paper describes a single-coefficient average whose weights do sum to
1. The gap between the paper and the code is reproduced here on purpose; it is not a porting
mistake. ``AdaFisherMulti(gamma=g)`` is the opt-in knob that turns the rule back into a convex
average, by passing ``gammas = (1 - g, 1 - g)``.

:func:`seed_or_accumulate` adds the start-up rule on top: at step 0 there is no stored value yet, so
one has to be invented. Both reference implementations seed the identity (a vector of ones, or an
identity matrix) and then average the first observation into it, which leaves a residue of
``(1 - gammas[0]) ** k`` of that identity after ``k`` updates -- 0.08**k with the shipped
coefficients. ``seed_first=True`` instead stores the first observation itself, so no residue of
anything else is ever left. It is off by default and changes nothing when off.
"""

from __future__ import annotations

from typing import Callable, Dict, Hashable, Sequence

from torch import Tensor


def update_running_avg(new: Tensor, current: Tensor, gammas: Sequence[float]) -> None:
    """In-place: ``current <- (1 - gammas[0]) * current + gammas[1] * new``."""
    current *= 1 - gammas[0]
    current += new * gammas[1]


def seed_or_accumulate(
    new: Tensor,
    store: Dict,
    key: Hashable,
    identity: Callable[[], Tensor],
    gammas: Sequence[float],
    step: int,
    seed_first: bool,
) -> None:
    """Start the running average when there is nothing to fold into, then fold ``new`` into it.

    There is nothing to fold into in two cases: step 0, and the first time this ``key`` is seen at
    all. The second case is not hypothetical -- a module behind a conditional branch, or one skipped
    by stochastic depth, can first be reached at step 7, and keying the start-up on ``step == 0``
    alone made that a bare ``KeyError`` out of the running average. Step 0 is kept as a start-up
    trigger in its own right, and not folded into "first time seen", so that a module whose hooks
    fire more than once at step 0 -- gradient accumulation over several micro-batches -- restarts
    exactly as it always has.

    ``seed_first`` picks between the two ways of starting it:

    * ``False`` (the default, and what every mode has always done): start from the identity, then
      immediately average the first observation into it. Algorithm 1 of the AdaFisher paper does
      this, and it leaves a residue of ``(1 - gammas[0])^k`` of that identity in the state after
      ``k`` updates. With the shipped ``gammas`` that residue is ``0.08^k``; with a corrected,
      convex average it is ``0.8^k``, which needs about 93 updates to fall below ``Lambda``, and
      that is what froze the ``kfac`` mode in a measured experiment with a corrected average.
    * ``True``: start from the first observation itself. The state is then a plain average of
      observations from the very first step, with no residue of anything else in it, ever.

    It changes nothing unless asked for: with ``seed_first=False`` the arithmetic is byte-for-byte
    what it has always been.
    """
    if step == 0 or key not in store:
        store[key] = new.detach().clone() if seed_first else identity()
        if seed_first:
            return
    update_running_avg(new, store[key], gammas)
