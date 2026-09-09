"""Abstract interface shared by every Fisher approximation mode.

See docs/reports/plan_lot1.md §0 for the rationale behind this exact signature (three corrections
to the sketch in docs/reports/plan.md §2.2, made precise while implementing lot 1):

1. ``update_input_factor`` / ``update_output_factor`` are two separate hook-driven calls, not one
   ``update_factors(A, B)`` call, because the forward and backward hooks fire independently and
   never have both raw factors available at the same time.
2. ``precondition`` is called once per module, receiving the weight direction and (if present) the
   bias direction together, so that a mode whose preconditioning is expensive (an eigenbasis
   projection or a matrix inverse, for K-FAC/EKFAC/TKFAC/TEKFAC) computes its shared state exactly
   once per module rather than once per parameter.
3. ``precondition`` receives the *raw* first-moment tensor(s) (``exp_avg``), not a bias-corrected
   ``m_hat``; the optimizer folds the bias-correction scalar into its own step size, mirroring
   ``AdaFisher._step`` (adafisher.py:258-273) as closely as possible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module


class FisherApproximation(ABC):
    """Builds AdaFisher's second moment v^(t) for one module, from raw per-hook factors."""

    @abstractmethod
    def update_input_factor(self, module: Module, h: Tensor, step: int) -> None:
        """Called from the forward hook with the raw layer input ``h`` (``input[0].data``)."""
        raise NotImplementedError

    @abstractmethod
    def update_output_factor(self, module: Module, s: Tensor, step: int) -> None:
        """Called from the backward hook with the raw output gradient ``s`` (``grad_output[0].data``)."""
        raise NotImplementedError

    @abstractmethod
    def refresh(self, module: Module, step: int) -> None:
        """Amortised re-estimation (matrix inverses, eigenbases). No-op for ``diag``."""
        raise NotImplementedError

    @abstractmethod
    def precondition(
        self,
        module: Module,
        weight_direction: Tensor,
        bias_direction: Optional[Tensor],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Apply the preconditioner to the raw first-moment direction(s) of one module.

        Returns the preconditioned weight direction alone if ``module.bias is None``, otherwise a
        ``(weight_direction, bias_direction)`` pair — mirroring the single-tensor-vs-list convention
        of the original ``_get_F_tilde`` (adafisher.py:209-223).
        """
        raise NotImplementedError
