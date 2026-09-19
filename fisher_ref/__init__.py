"""``fisher_ref`` -- the Fisher-drift reference bench.

A **reader** of this repository's benchmark artifacts: it trains nothing and changes nothing in
``benchmarks/`` or ``src/adafisher_modes/``. It loads a trajectory checkpoint, builds exact and
approximate curvature matrices at those weights, and compares them.

Layout
------

``conventions``  precision policy, the row-major (``rvec``) Kronecker convention, eval-mode
references and the sample-independence check.
``probes``  fixed, augmentation-free, content-hashed probe sets on a run's own seeded split.
``registry``  which kind of parameter block each part of a network is, weight sharing included.
``checkpoints``  the bridge to ``benchmarks/outputs/``: which weights exist, and how to load one.
``sources``  the backprop vectors (type-2, Monte-Carlo, empirical) a reference is built from.
``capture``  per-example layer inputs and output gradients, and per-sample gradients per layer kind.
``reference``  the exact references; ``dense`` materialises ``F = U^T U`` in float64.
``approx``  the approximation zoo (K-FAC, EKFAC, TKFAC, the normalisation-layer readings, the
weight-sharing decomposition, and the optimizer's own live state).
``metrics``  the comparisons: Frobenius fidelity, the natural-gradient step ratio, the Stein/KL
gap, the Kronecker and diagonal biases, inter-block coupling, and the reference's own noise floor.
``folds``  one traversal over folds of the probe set, so per-layer error bars come for free.
``rewarm``  rebuilding the optimizer's running averages at a checkpoint's weights.
``runners``  the two command-line protocols: the structural one and the operational one.
``experiments``  measurement drivers, configured by environment variables, that answer one
protocol question each.

This ``__init__`` re-exports the conventions only; every other module is imported by its own path.
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
