"""The noise floor of the reference itself (``plan_exp_draft.md`` §3.4).

**Any difference between approximations below this floor is not interpretable** (§10.3's reading
rule, HF8). Lot 1 measured it from a *single* two-way split, which gives a point estimate and no
error bar — and then had to read a "1.33x the null" result with no way to say whether 1.33 was
significant (``plan_exp_lot1.md`` §6.2). This module does what §3.4 actually asks: 20 random
partitions, a 95 % interval, and the ``d(F_{N'}, F_N)`` curve against ``N'``.

The scaling that turns a split into a floor, derived once here because lot 1 got it right only in
passing: ``F^(1) - F^(2)`` is the difference of two *independent* ``N/2`` estimates, so its size is
``sqrt(2) sigma_{N/2} = 2 sigma_N``. Hence

    sigma_N          = d_split / 2                 (one N-probe estimate's own error)
    null(two runs)   = sqrt(2) sigma_N = d_split / sqrt(2)

and a gap between two independent ``N``-probe references must clear ``null(two runs)``, not
``sigma_N``, before it means anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import torch
from torch import Tensor

from .frobenius import dense_gap


@dataclass(frozen=True)
class NoiseFloor:
    splits: List[float]
    median: float
    low: float
    high: float
    sigma_n: float
    null_two_independent: float
    n_partitions: int
    half: int

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def as_rows(self, **keys: Any) -> list:
        return [{**keys, "metric": name, "value": value} for name, value in
                (("noise_floor", self.median), ("noise_floor_low", self.low),
                 ("noise_floor_high", self.high), ("sigma_n", self.sigma_n),
                 ("null_two_independent", self.null_two_independent))]


def noise_floor(build: Callable[[Sequence[int]], Tensor], n_probes: int, *,
                partitions: int = 20, seed: int = 0) -> NoiseFloor:
    """``d(F^(1), F^(2))`` over ``partitions`` random halves → a 95 % interval.

    ``build(indices)`` returns the dense reference over those probes; the caller owns the model and
    the probe set, so this module stays a pure statistic.
    """
    generator = torch.Generator().manual_seed(seed)
    half = n_probes // 2
    values: List[float] = []
    for _ in range(partitions):
        order = torch.randperm(n_probes, generator=generator)
        first = build(order[:half].tolist())
        second = build(order[half:2 * half].tolist())
        values.append(dense_gap(first, second)["rel_to_geom"])
        del first, second
    ordered = sorted(values)
    low = ordered[max(int(0.025 * len(ordered)) - 1, 0)]
    high = ordered[min(int(0.975 * len(ordered)), len(ordered) - 1)]
    median = ordered[len(ordered) // 2]
    return NoiseFloor(splits=values, median=median, low=low, high=high,
                      sigma_n=median / 2.0, null_two_independent=median / (2.0 ** 0.5),
                      n_partitions=partitions, half=half)


def convergence_curve(build: Callable[[Sequence[int]], Tensor], n_probes: int,
                      fractions: Sequence[float] = (0.125, 0.25, 0.5, 1.0)) -> Dict[float, float]:
    """``d(F_{N'}, F_N)`` against ``N'`` — the curve §3.4 asks for alongside the floor.

    It is what would say whether the residual is a genuine systematic floor or a heavier-tailed
    variance: ``plan_exp_lot1.md`` §6.2 measured ``sigma_N`` falling by 2.92x for 13.75x the probes,
    where ``N^{-1/2}`` predicts 3.71x, and had no way to tell which.
    """
    full = build(list(range(n_probes)))
    out: Dict[float, float] = {}
    for fraction in fractions:
        count = max(int(fraction * n_probes), 2)
        out[float(fraction)] = dense_gap(build(list(range(count))), full)["rel_to_geom"]
    return out


__all__ = ["NoiseFloor", "convergence_curve", "noise_floor"]
