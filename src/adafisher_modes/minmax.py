"""Min-max normalisation of an instantaneous Kronecker-factor diagonal (AdaFisher Eq. 4).

Ports two pieces of the official AdaFisher repository (``reference_repos/AdaFisher/optimizers/
AdaFisher.py:13-45``), independently of its EMA (which has the bug documented in
docs/reports/plan.md §1.4): ``smart_detect_inf`` and ``MinMaxNormalization``. FisherAdapTune carries
an unused copy of the former (``_smart_detect_inf``, adafisher.py:21-26) but never applies the
latter — see docs/reports/plan.md §1.1, critical fact 2, and §5.1 for why the ``diag`` mode restores
it, opt-out via ``minmax_normalization=False``.
"""

from __future__ import annotations

from torch import Tensor, inf


def smart_detect_inf(tensor: Tensor) -> Tensor:
    """Replace +inf with 1. and -inf with 0., on a clone (input left untouched)."""
    result = tensor.clone()
    result[tensor == inf] = 1.0
    result[tensor == -inf] = 0.0
    return result


def min_max_normalization(tensor: Tensor, epsilon: float = 1e-6) -> Tensor:
    """Scale ``tensor`` to [0, 1] via (t - min) / (max - min + epsilon).

    Operates on the clone produced by ``smart_detect_inf``, so the caller's tensor is never
    mutated in place.
    """
    tensor = smart_detect_inf(tensor)
    min_t = tensor.min()
    max_t = tensor.max()
    return tensor.add_(-min_t).div_(max_t - min_t + epsilon)
