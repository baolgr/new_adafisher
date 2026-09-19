"""Registry of Fisher approximation modes, selected by ``AdaFisherMulti(fisher_mode=...)``.

``MODES`` maps a mode name to the class that builds AdaFisher's second moment that way. The five
entries are ``"diag"``, ``"kfac"``, ``"ekfac"``, ``"tkfac"`` and ``"tekfac"``; all five support
``Linear``, ``Conv2d`` (with ``groups=1`` and ``dilation=(1, 1)``), ``BatchNorm2d`` and
``LayerNorm`` with a one-dimensional ``normalized_shape``.

``AdaFisherMulti`` looks a name up here, calls the factory with the keyword arguments it was given,
and never touches the result again except through the :class:`FisherApproximation` interface. That
is the whole extension point: adding a sixth mode means adding a class and one line here.
"""

from __future__ import annotations

from typing import Callable, Dict

from .base import FisherApproximation
from .diag import DiagApproximation
from .ekfac import EKFACApproximation
from .kfac import KFACApproximation
from .tekfac import TEKFACApproximation
from .tkfac import TKFACApproximation

# Typed as a factory, not Type[FisherApproximation]: the ABC declares no __init__, and each mode's
# constructor kwargs genuinely differ (e.g. diag's minmax_normalization has no kfac/ekfac
# equivalent) — Type[FisherApproximation] would wrongly imply a shared constructor signature.
MODES: Dict[str, Callable[..., FisherApproximation]] = {
    "diag": DiagApproximation,
    "kfac": KFACApproximation,
    "ekfac": EKFACApproximation,
    "tkfac": TKFACApproximation,
    "tekfac": TEKFACApproximation,
}

__all__ = [
    "FisherApproximation",
    "DiagApproximation",
    "KFACApproximation",
    "EKFACApproximation",
    "TKFACApproximation",
    "TEKFACApproximation",
    "MODES",
]
