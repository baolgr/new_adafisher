"""Precision policy, vectorisation convention and run metadata for the Fisher-drift campaign
(``docs/reports/plan_exp_draft.md`` §2.5, §8; lot 0 of its §9).

Three things live here, and they are the invariants every later lot inherits:

1. **Precision.** References (``F``, ``E_hat``, Grams, every metric) are fp64; the layer statistics
   and ``U`` may be accumulated in fp32. TF32 is turned **off** on both the matmul and the cuDNN
   path — the second defaults to ``True``, which is the trap: on A100/H100 convolutions then run in
   TF32 with a relative error around ``1e-3`` on per-sample gradients, enough to break every
   exactness test (``plan_exp_draft.md`` §2.5).
2. **Vectorisation.** PyTorch flattens row-major (``rvec``), the literature writes ``cvec``. The
   one identity that fixes everything is ``rvec(B M A^T) = (B (x) A) rvec(M)``, so the papers'
   ``A (x) B`` (input factor first) is this repository's ``B (x) A``. ``kron_rvec`` is that
   statement, once, with a numeric test behind it (T0.3).
3. **Reference mode.** Every reference is computed with the model in ``eval`` mode: in ``train``
   mode a ``BatchNorm2d`` output depends on the rest of the batch, so per-sample gradients — and
   therefore ``F`` and ``E_hat`` — are not defined at all. ``reference_mode`` is the context that
   guarantees it and restores whatever was there before; ``assert_sample_independent`` is the
   check that it worked.

``configure()`` deliberately *returns what it set*. The TF32 switches have three spellings across
torch versions (``allow_tf32``, and the newer ``fp32_precision``), so asserting one attribute is
not evidence that TF32 is off — the returned record goes into every result file's metadata, and
T0.1 checks that it agrees with the live state.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

METRICS_VERSION = "fisher_ref/0.1"

#: Every reference and every metric is computed in this dtype (``plan_exp_draft.md`` §2.2: Fisher
#: spectra span more than 1e8, and in fp32 eigenvalues below ~1e-7 * lambda_max are not meaningful).
REFERENCE_DTYPE = torch.float64
#: ``U`` and the captured layer statistics may be accumulated here; the reductions are fp64.
CAPTURE_DTYPE = torch.float32

REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------------------
# Precision
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionState:
    """What the backends are actually set to. ``None`` = the knob does not exist on this torch."""

    matmul_allow_tf32: Optional[bool]
    cudnn_allow_tf32: Optional[bool]
    matmul_fp32_precision: Optional[str]
    cudnn_fp32_precision: Optional[str]
    cudnn_deterministic: Optional[bool]
    cudnn_benchmark: Optional[bool]
    deterministic_algorithms: bool

    def tf32_is_off(self) -> bool:
        """True when every knob that exists says "no TF32"."""
        flags_off = all(v is not True for v in (self.matmul_allow_tf32, self.cudnn_allow_tf32))
        precisions_off = all(
            v is None or v in ("ieee", "none")
            for v in (self.matmul_fp32_precision, self.cudnn_fp32_precision)
        )
        return flags_off and precisions_off


def _get(obj: Any, name: str) -> Any:
    return getattr(obj, name, None)


def _set(obj: Any, name: str, value: Any) -> None:
    """Set an attribute only if it already exists; never create a new backend knob."""
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
        except (RuntimeError, ValueError, TypeError):  # a read-only or renamed knob
            pass


def precision_state() -> PrecisionState:
    """Read the live backend state without changing it."""
    return PrecisionState(
        matmul_allow_tf32=_get(torch.backends.cuda.matmul, "allow_tf32"),
        cudnn_allow_tf32=_get(torch.backends.cudnn, "allow_tf32"),
        matmul_fp32_precision=_get(torch.backends.cuda.matmul, "fp32_precision"),
        cudnn_fp32_precision=_get(torch.backends.cudnn, "fp32_precision"),
        cudnn_deterministic=_get(torch.backends.cudnn, "deterministic"),
        cudnn_benchmark=_get(torch.backends.cudnn, "benchmark"),
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
    )


def configure(*, deterministic_algorithms: bool = False) -> PrecisionState:
    """Apply the campaign's precision policy and return the resulting state.

    ``deterministic_algorithms`` is off by default on purpose: ``torch.use_deterministic_
    algorithms(True)`` turns a missing deterministic kernel into a hard error at an arbitrary later
    point, for a property fp64 references do not need (``plan_exp_lot0.md`` §0.8).
    """
    _set(torch.backends.cuda.matmul, "allow_tf32", False)
    _set(torch.backends.cudnn, "allow_tf32", False)
    _set(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    _set(torch.backends.cudnn, "fp32_precision", "ieee")
    _set(torch.backends.cudnn, "deterministic", True)
    _set(torch.backends.cudnn, "benchmark", False)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    return precision_state()


# ----------------------------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------------------------


def git_revision() -> Optional[str]:
    """The repository's current commit, or ``None`` if git is unavailable (a tarball on a node)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def run_metadata(extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The block written into every result file (``plan_exp_draft.md`` §13).

    ``metrics_version`` is an invariant: changing it invalidates comparisons with earlier results,
    which is why it is recorded rather than assumed.
    """
    meta: Dict[str, Any] = {
        "metrics_version": METRICS_VERSION,
        "torch_version": torch.__version__,
        "reference_dtype": str(REFERENCE_DTYPE),
        "capture_dtype": str(CAPTURE_DTYPE),
        "precision": asdict(precision_state()),
        "git_revision": git_revision(),
    }
    if extra:
        meta.update(dict(extra))
    return meta


# ----------------------------------------------------------------------------------------------
# Reference mode
# ----------------------------------------------------------------------------------------------


@contextmanager
def reference_mode(model: nn.Module) -> Iterator[nn.Module]:
    """Put every submodule in ``eval`` mode for the duration, then restore the previous modes.

    This is what makes ``F`` and ``E_hat`` *defined*: with ``BatchNorm2d`` in train mode a sample's
    output depends on the rest of the batch, so there is no per-sample gradient to sum over
    (``plan_exp_draft.md`` §2.5). It also disables dropout — ``cct_2_3x2_cifar`` and
    ``vit_small_cifar`` carry ``p = 0.1``, so this is not a formality on those two.
    """
    previous = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield model
    finally:
        for module, was_training in previous:
            module.training = was_training


#: Relative tolerance for :func:`assert_sample_independent`, by input dtype. These are *reduction
#: order* margins, not physics: measured on ``cnn_gn_cifar`` and ``resnet20_cifar`` with a batch of
#: 256 against a batch of 4, an eval-mode network differs by ``5e-7`` relative in fp32 and ``1e-15``
#: absolute in fp64 (different BLAS/cuDNN kernels for different batch sizes), while the same
#: ``resnet20_cifar`` in **train** mode differs by ``3.6e-2`` relative — five orders of magnitude
#: away, which is the separation this check lives on.
INDEPENDENCE_RTOL = {torch.float64: 1e-10, torch.float32: 1e-5, torch.float16: 1e-2,
                     torch.bfloat16: 1e-1}


def assert_sample_independent(
    model: nn.Module, inputs: Tensor, *, k: int = 2, rtol: Optional[float] = None
) -> None:
    """Check that ``model(inputs[:k]) == model(inputs)[:k]``, i.e. that no layer mixes examples.

    The direct test of the property :func:`reference_mode` exists to establish. The comparison is
    relative (to ``max(|f|, 1)``) and its default tolerance comes from the input dtype
    (:data:`INDEPENDENCE_RTOL`): in fp32 two batch sizes take different BLAS kernels, so a perfectly
    sample-independent network still differs in the last few significant digits, while a
    ``BatchNorm2d`` left in train mode differs by ~1e-2 relative. Raises ``AssertionError`` naming
    the measured gap.
    """
    if inputs.shape[0] <= k:
        raise ValueError(f"need more than k={k} examples to compare; got {inputs.shape[0]}")
    with torch.no_grad():
        full = model(inputs)[:k]
        part = model(inputs[:k])
    gap = float((full - part).abs().max())
    scale = max(float(full.abs().max()), 1.0)
    tolerance = rtol if rtol is not None else INDEPENDENCE_RTOL.get(inputs.dtype, 1e-5)
    if not gap / scale <= tolerance:
        raise AssertionError(
            f"model is not sample-independent: max |f(x)[:k] - f(x[:k])| / max(|f|, 1) = "
            f"{gap / scale:.3e} > {tolerance:.1e} (absolute gap {gap:.3e}). A normalisation layer "
            "is in train mode (batch statistics), so per-sample gradients — and therefore F and "
            "E_hat — are not defined (plan_exp_draft.md §2.5)."
        )


# ----------------------------------------------------------------------------------------------
# Vectorisation convention
# ----------------------------------------------------------------------------------------------


def rvec(matrix: Tensor) -> Tensor:
    """Row-major flattening, PyTorch's own (``W.flatten()``); ``rvec(u v^T) = u (x) v``."""
    return matrix.reshape(-1)


def unrvec(vector: Tensor, shape: Sequence[int]) -> Tensor:
    """Inverse of :func:`rvec`."""
    return vector.reshape(tuple(shape))


def kron_rvec(output_factor: Tensor, input_factor: Tensor) -> Tensor:
    """The dense Kronecker curvature in ``rvec`` order, for a direction shaped ``(d_out, d_in)``.

    ``rvec(B M A^T) = (B (x) A) rvec(M)``, so a paper writing ``F = A (x) B`` with
    ``A = E[h h^T]`` (input) and ``B = E[d d^T]`` (output) becomes ``kron(B, A)`` here — output
    factor first. This is the central pitfall of ``papers/kfac_from_scratch_2507.05127.pdf``
    (Def. 1/2, Def. 23), re-derived in ``plan_lot2.md`` §0.4, asserted numerically by T0.3.
    """
    return torch.kron(output_factor, input_factor)
