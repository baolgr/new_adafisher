"""AdaFisher with interchangeable Fisher approximation modes (K-FAC, EKFAC, TKFAC, TEKFAC).

See ``docs/reports/plan.md`` for the full design and ``CLAUDE.md`` for the operating manual. Lot 1
implements ``fisher_mode="diag"``; lot 2 adds ``"kfac"``/``"ekfac"``; lot 3 adds
``"tkfac"``/``"tekfac"`` (all ``Linear`` only so far).
"""

from __future__ import annotations

from adafisher_modes.approximations import (
    MODES,
    DiagApproximation,
    EKFACApproximation,
    FisherApproximation,
    KFACApproximation,
    TEKFACApproximation,
    TKFACApproximation,
)
from adafisher_modes.optimizer import AdaFisherMulti

__all__ = [
    "AdaFisherMulti",
    "FisherApproximation",
    "DiagApproximation",
    "KFACApproximation",
    "EKFACApproximation",
    "TKFACApproximation",
    "TEKFACApproximation",
    "MODES",
]
