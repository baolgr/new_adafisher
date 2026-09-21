"""The interface every Fisher approximation mode implements.

A mode is driven by four calls, in this order within one training step:

1. :meth:`FisherApproximation.update_input_factor`, from a forward hook, with the layer's raw input.
2. :meth:`FisherApproximation.update_output_factor`, from a backward hook, with the gradient
   arriving at the layer's output.
3. :meth:`FisherApproximation.refresh`, from ``step()``, to redo whatever is amortised over several
   steps (a matrix inverse, an eigendecomposition). A no-op for ``diag``.
4. :meth:`FisherApproximation.precondition`, from ``step()``, which applies the inverse second
   moment to the direction the optimizer is about to take.

Three details of this signature are deliberate.

**The two factor updates are separate calls, not one.** The forward and backward hooks fire at
different times and never hold both raw statistics at once. A mode that needs them paired (EKFAC
and TEKFAC need the per-example input and gradient together) caches the input in
``update_input_factor`` and consumes it in ``update_output_factor``. Those three modes consume
their cache through :func:`pop_cached_input`, which turns "the input is not there" into one clear
error instead of a bare ``KeyError`` in one mode and a silent skip in another.

**``precondition`` is called once per module, with the weight and bias directions together.** The
four Kronecker modes pay their dominant cost -- a projection into an eigenbasis, or two matrix
products against a cached inverse -- once per module rather than once per parameter. It returns a
single tensor when the module has no bias, and a ``(weight, bias)`` pair when it has one, mirroring
the single-tensor-or-list convention of the reference implementation's own ``_get_F_tilde``.

**``precondition`` receives the raw first moment, not a bias-corrected one.** The optimizer folds
the Adam bias-correction scalar ``1 - beta**t`` into its own step size instead, exactly as the
reference implementation does. The one exception is ``rescale_form="clip"``
(``consumes_bias_corrected_momentum``), whose threshold is compared with the momentum itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module


def pop_cached_input(cache: Dict[Module, Tensor], module: Module) -> Tensor:
    """Take back the input ``update_input_factor`` cached for ``module``, or say why it is missing.

    ``ekfac``, ``tkfac`` and ``tekfac`` pair each example's input with the gradient of the *same*
    example, so the backward hook consumes exactly what the forward hook cached. One backward pass
    per forward pass is what makes that pairing well defined, and there is no way to honour it
    otherwise: a second backward over the same forward hands the output factor a second observation
    while the input factor has only one, and the per-example pairing the eigen-rescaling and the
    trace-restricted numerators are built from no longer exists.

    So the second backward is refused here, for all three modes, with the same error. It used to be
    a bare ``KeyError`` naming the module in ``tkfac`` and ``tekfac``, and a silent skip of the
    rescaling update in ``ekfac`` -- three modes, three behaviours, none of them saying what was
    wrong.

    Note that ``diag`` and ``kfac`` cache nothing, so they cannot detect this and still accept a
    second backward silently, folding the output gradient into their running average twice.
    """
    h_bar = cache.pop(module, None)
    if h_bar is None:
        raise RuntimeError(
            f"No cached layer input for {module}: update_output_factor was reached without a "
            f"matching update_input_factor. The usual cause is two backward passes over one "
            f"forward pass (for example loss_a.backward(retain_graph=True) followed by "
            f"loss_b.backward()), which this mode cannot support: it pairs each example's input "
            f"with that same example's output gradient, and the second backward has no input to "
            f"pair with. Run one backward per forward, or accumulate the two losses and call "
            f"backward once."
        )
    return h_bar


class FisherApproximation(ABC):
    """Builds AdaFisher's second moment v^(t) for one module, from raw per-hook factors."""

    #: The damping shared by every module unless :meth:`set_lambda` gave one its own.
    Lambda: float

    # ------------------------------------------------------------------
    # Per-module damping. Unused unless AdaFisherMulti(damping=...) is set; every mode then reads
    # its damping through lambda_for(), which returns the shared ``Lambda`` itself -- the same
    # Python float, so the same arithmetic -- for any module that was never given its own.
    # ------------------------------------------------------------------

    def lambda_for(self, module: Module) -> Union[float, Tensor]:
        """The damping in effect for ``module``: its own value if :meth:`set_lambda` gave it one,
        otherwise the shared ``Lambda``."""
        overrides = self.__dict__.get("_lambda_override")
        if overrides is not None and module in overrides:
            return overrides[module]
        return self.Lambda

    def set_lambda(self, module: Module, value: Tensor) -> None:
        """Give ``module`` its own damping, used from the next time the mode reads it."""
        self.__dict__.setdefault("_lambda_override", {})[module] = value

    def applied_lambda(self, module: Module) -> Union[float, Tensor]:
        """The damping inside the operator ``precondition`` applies *now*. It is
        :meth:`lambda_for` for a mode that adds the damping at division time; ``kfac`` and
        ``tkfac`` bake it into inverses at ``refresh`` and override this to return that value."""
        return self.lambda_for(module)

    @property
    def consumes_bias_corrected_momentum(self) -> bool:
        """Whether :meth:`precondition` must receive the bias-corrected momentum
        ``m / (1 - beta^t)``, its result then being applied without further correction. True only
        for ``ekfac``/``tekfac`` under ``rescale_form="clip"``, which compares the momentum against
        a threshold; every other mode receives the raw momentum, as the note above says."""
        return False

    def begin_step(self, step: int) -> None:
        """Called by the optimizer before it walks the modules of one step, with the same step index
        the hooks used. No-op by default."""

    def mean_curvature(self, module: Module) -> Optional[Tensor]:
        """Mean eigenvalue of ``module``'s *undamped* stored curvature, i.e. its trace divided by
        its dimension, or ``None`` if nothing is stored for it yet. What a relative damping is
        proportional to."""
        raise NotImplementedError(f"{type(self).__name__} has no relative damping")

    def num_directions(self, module: Module) -> Optional[int]:
        """Dimension of ``module``'s curvature operator (``d_out * d_in_aug``), or ``None`` if
        nothing is stored for it yet. The weight of that module in a network-wide mean."""
        raise NotImplementedError(f"{type(self).__name__} has no relative damping")

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
