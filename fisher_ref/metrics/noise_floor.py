"""The reference's own sampling error: the floor below which no difference is interpretable.

Any gap between two approximations that is smaller than this floor is not a finding.

The scaling that turns a split into a floor. ``F^(1) - F^(2)`` is the difference of two
*independent* half-size estimates, so its size is ``sqrt(2) sigma_{N/2} = 2 sigma_N``. Hence

    sigma_N        = d_split / 2                  (one N-probe estimate's own error)
    null(two runs) = sqrt(2) sigma_N = d_split / sqrt(2)

and a gap between two independent ``N``-probe references must clear ``null(two runs)``, not
``sigma_N``, before it means anything.

**What the interval is.** :func:`noise_floor` sorts the measured split distances and reports the
element at index ``max(int(0.025 n) - 1, 0)`` as ``low`` and at ``min(int(0.975 n), n - 1)`` as
``high``. For fewer than 80 partitions -- which includes the default of 20 -- those indices are 0
and ``n - 1``, so ``low`` and ``high`` are the observed **minimum and maximum**, not a 95 %
quantile interval. ``median`` is the upper of the two central values at even ``n``. Read them as
the observed range of the splits.

That is a sample-size limit, not an arithmetic slip. The textbook nearest-rank 2.5th percentile of
``n`` sorted values sits at index ``ceil(0.025 n) - 1``, which is index 0 for every ``n`` up to 40:
with 20 splits there is simply no value below the minimum to report, whatever formula is used.

Against nearest rank the indices here differ by at most one, and **only outwards**. Measured over
every ``n`` from 2 to 5000: 0 values of ``n`` where this interval is narrower than nearest rank,
4960 where it is one index wider on one side or both. Side by side::

    n     low here / nearest rank    high here / nearest rank    covers
    10        0 / 0                     9 / 9                    100.0 %
    20        0 / 0                    19 / 19                   100.0 %
    40        0 / 0                    39 / 38                   100.0 %
    41        0 / 1                    39 / 39                    97.6 %
    80        1 / 1                    78 / 77                    97.5 %
    200       4 / 4                   195 / 194                   96.0 %

("covers" is the share of the sorted sample between the two indices, inclusive; a true 95 %
interval covers 95 %.) ``low`` first moves off the minimum at ``n = 80`` and ``high`` off the
maximum at ``n = 41``. The bias is towards a *wider* interval, so a difference that clears this
floor also clears the nearest-rank one -- the safe direction for a threshold something has to beat.
The arithmetic is left as it is for that reason, and because every result file this package has
produced used 20 partitions, where the two agree exactly.

Public API
----------

:class:`NoiseFloor`  the split distances, their median, the range, ``sigma_n``, the two-run null,
and the partition count.

:func:`noise_floor`  ``d(F^(1), F^(2))`` over random disjoint halves of the probes. ``build(indices)``
returns the dense reference over those probes, so this module stays a pure statistic and the caller
owns the model and the probe set. The distance used is the gap normalised by the geometric mean of
the two norms.

:func:`convergence_curve`  ``d(F_{N'}, F_N)`` against ``N'``, which is what says whether the
residual is a genuine systematic floor or a heavier-tailed variance.

Dependencies: :mod:`fisher_ref.metrics.frobenius`.
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
    """``d(F^(1), F^(2))`` over ``partitions`` random halves, with ``low``/``high`` around it.

    ``build(indices)`` returns the dense reference over those probes; the caller owns the model and
    the probe set, so this module stays a pure statistic.

    ``low`` and ``high`` are the observed **range** of the splits at the default 20 partitions, not
    a 95 % quantile interval -- 20 samples cannot resolve one. This module's header has the index
    table and the comparison against nearest rank.
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
