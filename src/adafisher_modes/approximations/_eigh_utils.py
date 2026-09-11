"""Numerically robust eigenbasis extraction, shared by ``ekfac`` and ``tekfac``.

Both modes need only the *eigenvectors* of a symmetric positive-semidefinite factor, and both used
to call ``linalg.eigh`` on the raw EMA factor. On CUDA that crashes, reproducibly, on a factor that
is **exactly** rank-deficient:

    torch._C._LinAlgError: linalg.eigh: The algorithm failed to converge because the input matrix
    is ill-conditioned or has too many repeated eigenvalues (error code: 1)

Two measured instances, both from real cluster runs that died mid-training after 20 s and 4 min of
useful work respectively (``benchmarks/slurm/logs/{mlp_ln_mnist,resnet20_cifar}_all-*.out``):

- ``mlp_ln_mnist``'s first ``Linear``: MNIST's border pixels are identically zero across the whole
  dataset — 130 of 784 — so ``A`` is ``785 x 785`` of rank 646 with **136 exactly-zero
  eigenvalues** and a condition number of ``4e301``;
- ``resnet20_cifar``'s convolutions: the same structure reached by another route, a post-ReLU patch
  covariance with dead channels.

``kfac``/``tkfac`` never hit this because they damp *before* inverting (``kfac.py::_damped_factors``
builds ``A + sqrt(lambda*pi) I``, strictly positive definite by construction); the eigenbasis modes
damped only afterwards, at ``precondition`` time (``s* + lambda``).

**The ridge below is a conditioning device, not damping.** ``M + cI`` shifts every eigenvalue by
``c`` and leaves every eigenvector — and their ascending order — unchanged, so the returned basis is
mathematically identical to ``eigh(M)``'s. EKFAC's and TEKFAC's semantics are untouched, and
``CLAUDE.md``'s rule that the four non-diagonal modes damp with ``lambda`` only still holds: the
scaling applied to the projected gradient is still ``s* + lambda`` / ``Theta + lambda``, unchanged.
Only the path cuSOLVER takes changes.
"""

from __future__ import annotations

import warnings

from torch import Tensor, eye, linalg

# Relative to the factor's own mean diagonal entry, so it follows its scale. The bias-augmentation
# column guarantees a nonzero diagonal for every supported layer type (``A[-1, -1] == 1`` for
# Linear/Conv2d, ``A[1, 1] == 1`` for a normalisation layer), so the ridge is never zero.
EIGH_RIDGE = 1e-6


def eigenbasis(M: Tensor) -> Tensor:
    """Eigenvectors of a symmetric PSD ``M``, ascending by eigenvalue.

    Falls back to the CPU solver if the device solver still fails: LAPACK's direct ``syevd`` handles
    spectra that cuSOLVER's iterative path does not, and a slow refresh is worth more than a dead
    job. The warning is deliberate — a silent fallback would hide a degenerating factor.
    """
    ridge = EIGH_RIDGE * M.diagonal().mean()
    conditioned = M + ridge * eye(M.size(0), dtype=M.dtype, device=M.device)
    try:
        return linalg.eigh(conditioned)[1]
    except linalg.LinAlgError:
        warnings.warn(
            f"linalg.eigh failed on a {tuple(M.shape)} factor on device {M.device}; retrying on "
            "CPU. The factor is likely near-singular — see _eigh_utils.py's docstring.",
            RuntimeWarning,
            stacklevel=2,
        )
        return linalg.eigh(conditioned.cpu())[1].to(M.device)
