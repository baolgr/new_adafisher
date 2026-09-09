"""``diag`` mode: AdaFisher's original diagonal-Kronecker approximation (Prop. 3.2, Eq. 4).

    F~_D = H'_{D,l-1} kron S'_{D,l} + lambda * I

where H', S' are the min-max normalisations of the instantaneous diagonals H_D, S_D, applied before
the EMA of Eq. (3). Reference: ``reference_repos/FisherAdapTune/scripts/adafisher.py``
(``AdaFisherBackbone._get_F_tilde``, lines 209-223) for the Kronecker/damping construction, and
``reference_repos/AdaFisher/optimizers/AdaFisher.py:412,431`` for where min-max is applied relative
to the EMA. See docs/reports/plan.md §5.1 for why min-max defaults to on here.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from torch import Tensor, kron
from torch.nn import Module

from adafisher_modes.ema import update_running_avg
from adafisher_modes.factors import compute_h_diag, compute_s_diag
from adafisher_modes.minmax import min_max_normalization

from .base import FisherApproximation


class DiagApproximation(FisherApproximation):
    def __init__(
        self,
        Lambda: float = 1e-3,
        gammas: Sequence[float] = (0.92, 0.008),
        minmax_normalization: bool = True,
        epsilon: float = 1e-6,
    ) -> None:
        self.Lambda = Lambda
        self.gammas = gammas
        self.minmax_normalization = minmax_normalization
        self.epsilon = epsilon
        self._H: Dict[Module, Tensor] = {}
        self._S: Dict[Module, Tensor] = {}

    def update_input_factor(self, module: Module, h: Tensor, step: int) -> None:
        H_i = compute_h_diag(h, module)
        if self.minmax_normalization:
            H_i = min_max_normalization(H_i, self.epsilon)
        if step == 0:
            self._H[module] = H_i.new_ones(H_i.size(0))
        update_running_avg(H_i, self._H[module], self.gammas)

    def update_output_factor(self, module: Module, s: Tensor, step: int) -> None:
        S_i = compute_s_diag(s, module)
        if self.minmax_normalization:
            S_i = min_max_normalization(S_i, self.epsilon)
        if step == 0:
            self._S[module] = S_i.new_ones(S_i.size(0))
        update_running_avg(S_i, self._S[module], self.gammas)

    def refresh(self, module: Module, step: int) -> None:
        # Nothing to amortise: F~_D is recombined from H, S on every call to precondition().
        pass

    def f_tilde(self, module: Module) -> Tensor:
        """Raw, unsplit F~_D = kron(H, S) + lambda * I for ``module``.

        Port of ``AdaFisherBackbone._get_F_tilde`` before the weight/bias split
        (adafisher.py:215-217). Exposed as a debug/test hook: the bit-exactness test compares it
        directly against the reference implementation, and it is the natural quantity the
        Frobenius-dominance tests of lot 2/3 (docs/reports/plan.md §6.1) will compare across modes.
        """
        H, S = self._H[module], self._S[module]
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
