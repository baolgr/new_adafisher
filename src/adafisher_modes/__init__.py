"""AdaFisher with an interchangeable Fisher approximation mode.

AdaFisher (Martins Gomes et al., ICLR 2025, ``adafisher_2405.16397.pdf``) is Adam with the second
moment replaced by a Kronecker-factored estimate of each layer's Fisher information. This package
keeps that optimizer as published and makes the *construction* of the second moment selectable.

All five modes are built from the same two per-layer statistics: the second moment of the layer's
bias-augmented input, and the second moment of the gradient arriving at the layer's output. They
differ only in what they do with them.

===========  ============================================================================
``diag``     AdaFisher's own diagonal approximation (its Prop. 3.2, Eq. 4).
``kfac``     Martens & Grosse's K-FAC (``kfac_1503.05671.pdf``).
``ekfac``    George et al.'s eigenvalue-corrected K-FAC (``ekfac_1806.03884.pdf``).
``tkfac``    Gao et al.'s trace-restricted K-FAC (``tkfac_2011.10741.pdf``).
``tekfac``   Gao et al.'s trace-restricted, eigenvalue-corrected K-FAC (``tekfac_2011.13609.pdf``).
===========  ============================================================================

Everything outside that construction is identical across the five: the forward/backward hooks, the
running average of the factors, the momentum on the first moment, and the absence of a square root
on the second moment. The mode is a constructor field, not a separate optimizer --
``AdaFisherMulti(model, fisher_mode="ekfac")``. Nothing else in a training loop needs to know which
mode is active.

The optimizer itself is in :mod:`adafisher_modes.optimizer`; the five modes are in
:mod:`adafisher_modes.approximations`. The names below are re-exports for convenience.
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
