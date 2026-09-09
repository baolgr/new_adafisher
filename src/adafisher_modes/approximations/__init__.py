"""Registry of Fisher approximation modes, selected by ``AdaFisherMulti(fisher_mode=...)``.

Lot 1 registered only ``"diag"``. Lot 2 added ``"kfac"``, ``"ekfac"`` (``Linear`` only — see
docs/reports/plan_lot2.md); lot 3 adds ``"tkfac"``, ``"tekfac"`` (``Linear`` only — see
docs/reports/plan_lot3.md) — none of this requires any change to ``optimizer.py``.
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
