"""``diag`` mode: AdaFisher's own diagonal-Kronecker second moment (Prop. 3.2, Eq. 4).

    F~_D = H'_D (x) S'_D + lambda

``H_D`` is the diagonal of the second moment of the layer's bias-augmented input, ``S_D`` the
diagonal of the second moment of the gradient at the layer's output, and ``H'``/``S'`` are their
min-max normalisations to [0, 1]. The Kronecker product of two diagonals is an outer product, so
``F~_D`` has exactly the shape of the layer's gradient and the whole mode reduces to an element-wise
division. This is the one mode that is diagonal in the parameter basis.

The extraction of ``H_D`` and ``S_D`` is :mod:`adafisher_modes.factors`; the normalisation is
:mod:`adafisher_modes.minmax`; the running average is :mod:`adafisher_modes.ema`. Nothing is
amortised, so :meth:`DiagApproximation.refresh` does nothing: ``F~_D`` is rebuilt from the two
running averages every time it is applied.

**Where the min-max sits relative to the running average is a real choice, not a detail.** By
default this class normalises the *instantaneous* factor and then averages it, which is what the
official AdaFisher repository does (``AdaFisher.py:412`` and ``:431``). Algorithm 1 of the paper
does the opposite: it averages first (line 4) and normalises the result (line 5). The two are not
equivalent, because normalising destroys scale. Averaging first leaves the running average's own
mis-scaling in the result; normalising first erases it. Measured on a real layer: in the shipped
order ``F~_D`` spans ``[lambda, lambda + 7.6e-5]``, a 7.6% spread; in the paper's order it spans
``[lambda, 1 + lambda]``. ``minmax_after_average=True`` selects the paper's order; the default
``False`` is bit-identical to the behaviour this port has always had.

``minmax_normalization=False`` removes the normalisation entirely and reproduces
``reference_repos/FisherAdapTune/scripts/adafisher.py`` bit-for-bit. That is the setting the
non-regression test runs under.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from torch import Tensor, kron
from torch.nn import Module

from adafisher_modes.ema import seed_or_accumulate
from adafisher_modes.factors import compute_h_diag, compute_s_diag
from adafisher_modes.minmax import min_max_normalization

from .base import FisherApproximation


class DiagApproximation(FisherApproximation):
    def __init__(
        self,
        Lambda: float = 1e-3,
        gammas: Sequence[float] = (0.92, 0.008),
        minmax_normalization: bool = True,
        minmax_after_average: bool = False,
        epsilon: float = 1e-6,
        ema_seed_first: bool = False,
    ) -> None:
        self.Lambda = Lambda
        self.gammas = gammas
        self.minmax_normalization = minmax_normalization
        self.minmax_after_average = minmax_after_average
        self.ema_seed_first = ema_seed_first
        self.epsilon = epsilon
        self._H: Dict[Module, Tensor] = {}
        self._S: Dict[Module, Tensor] = {}

    def update_input_factor(self, module: Module, h: Tensor, step: int) -> None:
        H_i = compute_h_diag(h, module)
        if self.minmax_normalization and not self.minmax_after_average:
            H_i = min_max_normalization(H_i, self.epsilon)
        seed_or_accumulate(H_i, self._H, module, lambda: H_i.new_ones(H_i.size(0)),
                           self.gammas, step, self.ema_seed_first)

    def update_output_factor(self, module: Module, s: Tensor, step: int) -> None:
        S_i = compute_s_diag(s, module)
        if self.minmax_normalization and not self.minmax_after_average:
            S_i = min_max_normalization(S_i, self.epsilon)
        seed_or_accumulate(S_i, self._S, module, lambda: S_i.new_ones(S_i.size(0)),
                           self.gammas, step, self.ema_seed_first)

    def refresh(self, module: Module, step: int) -> None:
        # Nothing to amortise: F~_D is recombined from H, S on every call to precondition().
        pass

    def f_tilde(self, module: Module) -> Tensor:
        """Raw, unsplit ``F~_D = kron(H, S) + lambda``, shaped like the module's gradient with the
        bias column still attached.

        Port of ``AdaFisherBackbone._get_F_tilde`` before its weight/bias split
        (``adafisher.py:215-217``). Exposed as a test hook: the bit-exactness test compares it
        directly against that reference, and it is the quantity the cross-mode comparisons use.
        """
        H, S = self._H[module], self._S[module]
        if self.minmax_normalization and self.minmax_after_average:
            H = min_max_normalization(H, self.epsilon)
            S = min_max_normalization(S, self.epsilon)
        return kron(H.unsqueeze(1), S.unsqueeze(0)).t() + self.Lambda

    def precondition(
        self,
        module: Module,
        weight_direction: Tensor,
        bias_direction: Optional[Tensor],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        F = self.f_tilde(module)
        if module.bias is not None:
            assert bias_direction is not None
            F_w = F[:, :-1].reshape(weight_direction.shape)
            F_b = F[:, -1:].reshape(bias_direction.shape)
            return weight_direction / F_w, bias_direction / F_b
        return weight_direction / F.reshape(weight_direction.shape)
