"""Exponential moving average of a raw Kronecker-factor statistic (AdaFisher Eq. 3).

Bit-exact port of ``_update_running_avg`` (``reference_repos/FisherAdapTune/scripts/adafisher.py:
28-30``) — the two-parameter ``gammas = [1-gamma, gamma]`` formula, not the official repository's
single-scalar ``gamma`` (which is buggy, docs/reports/plan.md §1.4). Shared by every
``FisherApproximation`` that maintains an EMA'd statistic, not just ``diag`` — kept in its own module
so K-FAC/TKFAC (lot 2/3) reuse it rather than duplicating it.
"""

from __future__ import annotations

from typing import Sequence

from torch import Tensor


def update_running_avg(new: Tensor, current: Tensor, gammas: Sequence[float]) -> None:
    """In-place: ``current <- (1 - gammas[0]) * current + gammas[1] * new``."""
    current *= 1 - gammas[0]
    current += new * gammas[1]
