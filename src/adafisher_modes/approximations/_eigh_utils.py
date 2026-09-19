"""Numerically robust eigenvector extraction, shared by the ``ekfac`` and ``tekfac`` modes.

Both modes need only the *eigenvectors* of a symmetric positive-semidefinite factor, and both used
to call ``torch.linalg.eigh`` on the raw running-average factor. On CUDA that crashes, reproducibly,
on a factor that is **exactly** rank-deficient::

    torch._C._LinAlgError: linalg.eigh: The algorithm failed to converge because the input matrix
    is ill-conditioned or has too many repeated eigenvalues (error code: 1)

Two measured instances, both from real cluster runs that died mid-training after 20 s and 4 min of
useful work (``benchmarks/slurm/logs/{mlp_ln_mnist,resnet20_cifar}_all-*.out``):

* ``mlp_ln_mnist``'s first ``Linear``. MNIST's border pixels are identically zero across the whole
  dataset -- 130 of 784 -- so its input factor is 785 x 785 of rank 646, with 136 exactly-zero
  eigenvalues and a condition number of 4e301.
* ``resnet20_cifar``'s convolutions: the same structure reached by another route, a post-ReLU patch
  covariance with dead channels.

The ``kfac`` and ``tkfac`` modes never hit this because they add their damping *before* inverting,
which makes the matrix strictly positive definite. The eigenbasis modes damp only afterwards, when
rescaling the projected gradient.

:func:`eigenbasis` adds a small ridge before decomposing, and falls back to the CPU solver if the
device solver still raises. **The ridge is a conditioning device, not damping.** ``M + c*I`` shifts
every eigenvalue by ``c`` and leaves every eigenvector, and their ascending order, unchanged, so the
basis returned is the same one ``eigh(M)`` would give. Only the eigenvectors are returned; the
shifted eigenvalues are discarded, so nothing the modes use as curvature carries the ridge. The
rule that the four Kronecker modes damp with ``lambda`` alone still holds.

Not reproducible on a CPU-only test runner: LAPACK solves these matrices happily. Do not remove the
ridge because the suite is green locally.
"""

from __future__ import annotations

import warnings

from torch import Tensor, eye, linalg

# Relative to the factor's own mean diagonal entry, so it follows the factor's scale. It is not
# guaranteed nonzero: a bias-free Linear or Conv2d has no constant-one column, and the output
# factor never has one, so a layer whose input or whose output gradient is identically zero gets a
# ridge of zero and no conditioning at all. That case is degenerate anyway -- the factor is then the
# zero matrix, which eigh handles.
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
