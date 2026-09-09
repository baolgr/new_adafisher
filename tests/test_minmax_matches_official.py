"""``min_max_normalization`` vs. the official repository's ``MinMaxNormalization``.

Deliberately isolated from the EMA (which the official repository gets wrong, docs/reports/
plan.md §1.4): this test only exercises the normalisation formula itself, so the official
repository's bug cannot contaminate the oracle. See docs/reports/plan_lot1.md §3.1.
"""

from __future__ import annotations

import torch
from adafisher_modes.minmax import min_max_normalization, smart_detect_inf


def test_matches_official_on_random_tensors(official_adafisher) -> None:
    torch.manual_seed(42)
    for _ in range(10):
        x = torch.randn(37)
        expected = official_adafisher.MinMaxNormalization(x.clone())
        actual = min_max_normalization(x.clone())
        assert torch.equal(actual, expected)


def test_matches_official_with_inf_entries(official_adafisher) -> None:
    x = torch.tensor([1.0, float("inf"), -float("inf"), -3.0, 0.5])
    expected = official_adafisher.MinMaxNormalization(x.clone())
    actual = min_max_normalization(x.clone())
    assert torch.equal(actual, expected)


def test_matches_official_on_constant_tensor(official_adafisher) -> None:
    # max == min: exercises the epsilon guard in the denominator.
    x = torch.full((5,), 3.0)
    expected = official_adafisher.MinMaxNormalization(x.clone())
    actual = min_max_normalization(x.clone())
    assert torch.equal(actual, expected)


def test_smart_detect_inf_matches_official(official_adafisher) -> None:
    x = torch.tensor([1.0, float("inf"), -float("inf"), -3.0])
    expected = official_adafisher.smart_detect_inf(x.clone())
    actual = smart_detect_inf(x.clone())
    assert torch.equal(actual, expected)


def test_does_not_mutate_input() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])
    x_clone = x.clone()
    min_max_normalization(x)
    assert torch.equal(x, x_clone)
