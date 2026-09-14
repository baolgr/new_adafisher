"""``fisher_ref`` — the Fisher-drift reference bench (``docs/reports/plan_exp_draft.md``).

A **reader** of this repository's benchmark artifacts: it trains nothing and changes nothing in
``benchmarks/`` or ``src/adafisher_modes/``. Lot 0 ships the four modules every later lot needs —
the precision and vectorisation conventions, the probe sets, the layer-type registry, and the
bridge to the trajectory checkpoints.
"""

from __future__ import annotations

from .conventions import (
    CAPTURE_DTYPE,
    METRICS_VERSION,
    REFERENCE_DTYPE,
    PrecisionState,
    assert_sample_independent,
    configure,
    kron_rvec,
    precision_state,
    reference_mode,
    run_metadata,
    rvec,
    unrvec,
)

__all__ = [
    "CAPTURE_DTYPE",
    "METRICS_VERSION",
    "REFERENCE_DTYPE",
    "PrecisionState",
    "assert_sample_independent",
    "configure",
    "kron_rvec",
    "precision_state",
    "reference_mode",
    "run_metadata",
    "rvec",
    "unrvec",
]
