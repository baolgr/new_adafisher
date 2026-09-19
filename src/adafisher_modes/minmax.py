"""Min-max normalisation of an instantaneous factor diagonal (AdaFisher Eq. 4).

Equation (4) of the AdaFisher paper rescales each of the two diagonal factors to the range [0, 1]
before combining them, so that the second moment ``F~_D`` lives in ``[lambda, 1 + lambda]``
whatever the scale of the activations or the gradients. This module holds the two pieces of that
rescaling, ported from the official AdaFisher repository
(``reference_repos/AdaFisher/optimizers/AdaFisher.py``, lines 13-45):

* :func:`smart_detect_inf` replaces ``+inf`` by 1 and ``-inf`` by 0 before anything else. NaN is
  not handled, matching the reference exactly: one NaN makes the whole normalised factor NaN.
* :func:`min_max_normalization` maps ``t`` to ``(t - min) / (max - min + 1e-6)``. The ``1e-6``
  guards a constant factor, which then maps to all zeros.

Neither function mutates its argument; both work on the clone :func:`smart_detect_inf` makes.

Only the ``diag`` mode uses this, and only when ``minmax_normalization=True`` (its default). The
FisherAdapTune reference carries an unused copy of ``smart_detect_inf`` and never applies the
normalisation at all, so ``minmax_normalization=False`` is what reproduces that reference exactly.
The four Kronecker modes never apply it: rescaling each factor to [0, 1] destroys the scale that
TKFAC's trace-preservation theorem and every Frobenius-norm comparison depend on.

One consequence worth knowing before reading a ``diag`` number off a normalisation layer: for
``BatchNorm2d`` and ``LayerNorm`` the input-factor diagonal has exactly **two** entries, one for
the scale parameter and one for the shift. Min-max of a two-entry vector is exactly ``[0, 1]`` or
``[1, 0]``, so on those layers the input factor carries one bit -- which of the two blocks is the
larger -- and no magnitude at all.
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
