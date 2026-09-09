"""``tkfac`` mode: Gao et al.'s Trace-restricted Kronecker-Factored Approximate Curvature (TKFAC).
Linear (lot 3, ``docs/reports/plan_lot3.md``) and Conv2d with ``groups=1``, ``dilation=(1,1)``
(lot 4, ``docs/reports/plan_lot4.md``).

    F_l = E[Lambda_{l-1} (x) Gamma_l],  Lambda_{l-1} = h_bar h_bar^T,  Gamma_l = delta delta^T

is approximated as ``F_l ~= delta_l * Phi_l (x) Psi_l`` (``tkfac_2011.10741.pdf`` Eq. 4.3), with the
"simplified formulas" of Theorem 4.1 / Eq. (4.9) (the paper's own stated choice, "in the rest of
this paper we all use the simplified formulas as (4.9)"):

    delta_l = E[tr(Lambda)tr(Gamma)],  Phi_l = E[tr(Gamma)Lambda]/delta_l,  Psi_l = E[tr(Lambda)Gamma]/delta_l

which keeps ``tr(Phi_l)=tr(Psi_l)=1`` and hence ``tr(F~) = delta_l = tr(F)`` **exactly** (Lemma 4.1,
the partial-trace argument of Theorem 4.1's proof) -- the headline property this mode exists for.

Damping follows the paper's own factored-Tikhonov trick for fully-connected layers (Eq. 5.14-5.15),
re-derived in ``docs/reports/plan_lot3.md`` §0.3 into the ``(1/delta)*Psi~^-1 M Phi~^-1`` form used
below: ``Phi~ = Phi + sqrt(Lambda/delta)*I``, ``Psi~ = Psi + sqrt(Lambda/delta)*I``. The paper's own
convolution-specific *adaptive* damping (the ``nu``-floor of Eq. 5.16) is not implemented — this
mode's existing, dimension-agnostic ``sqrt(Lambda/delta)`` scalar damping is reused unchanged for
Conv2d (lot 4, ``docs/reports/plan_lot4.md`` §0.6).

**Conv2d (lot 4, plan_lot4.md §0.5).** ``update_input_factor``/``update_output_factor`` build
``h_bar``/``s_flat`` via the layer-dispatching ``augment_input``/``flatten_output_grad`` (in place of
lot 3's ``Linear``-only ``augment_linear_input``/inline reshape); ``_tkfac_utils``'s
``instantaneous_raw_factors``/``bootstrap_raw_factors`` are already generic over any ``(h_bar, s)``
shape and need no change, and neither does ``precondition`` (``_kron_utils`` handles the ``Conv2d``
weight reshape).

**SUA (lot 6, docs/reports/plan_lot6.md).** ``conv_sua=True`` selects the channel-only SUA input
factor (``kfac_conv_1602.01407.pdf`` p. 14) for ``Conv2d`` modules: ``update_input_factor`` caches
``augment_input(h, module, sua=self.conv_sua)``; ``instantaneous_raw_factors``/
``bootstrap_raw_factors`` need no change (already generic over any ``(h_bar, s)`` shape).
``precondition`` applies the small operator independently at each of the ``k_h*k_w`` kernel offsets
instead of one flat matmul (plan_lot6.md §0.4).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from torch import Tensor, eye, kron
from torch.nn import Conv2d, Module

from adafisher_modes.ema import update_running_avg
from adafisher_modes.factors import augment_input, flatten_output_grad

from ._kron_utils import (
    augment_conv2d_direction_sua,
    augment_direction,
    split_conv2d_direction_sua,
    split_direction,
)
from ._tkfac_utils import bootstrap_raw_factors, instantaneous_raw_factors
from .base import FisherApproximation


class TKFACApproximation(FisherApproximation):
    def __init__(
        self,
        Lambda: float = 1e-3,
        gammas: Sequence[float] = (0.92, 0.008),
        T_inv: int = 100,
        conv_sua: bool = False,
    ) -> None:
        self.Lambda = Lambda
        self.gammas = gammas
        self.T_inv = T_inv
        self.conv_sua = conv_sua
        self._delta: Dict[Module, Tensor] = {}
        self._Phi_raw: Dict[Module, Tensor] = {}
        self._Psi_raw: Dict[Module, Tensor] = {}
        self._Phi_inv: Dict[Module, Tensor] = {}
        self._Psi_inv: Dict[Module, Tensor] = {}
        self._delta_at_refresh: Dict[Module, Tensor] = {}
        self._cached_h_bar: Dict[Module, Tensor] = {}

    def update_input_factor(self, module: Module, h: Tensor, step: int) -> None:
        # All of this mode's state needs both d_in_aug and d_out simultaneously (the bootstrap,
        # docs/reports/plan_lot3.md §0.2), which are only jointly known once the backward hook
        # fires. So the forward hook only caches h_bar for update_output_factor to consume
        # (forward always precedes backward for one loss.backward() call, plan_lot2.md §0.3).
        self._cached_h_bar[module] = augment_input(h, module, sua=self.conv_sua)

    def update_output_factor(self, module: Module, s: Tensor, step: int) -> None:
        h_bar = self._cached_h_bar.pop(module)
        s_flat = flatten_output_grad(s, module)
        delta_i, phi_raw_i, psi_raw_i = instantaneous_raw_factors(h_bar, s_flat)
        if step == 0:
            self._delta[module], self._Phi_raw[module], self._Psi_raw[module] = (
                bootstrap_raw_factors(h_bar.size(1), s_flat.size(1), h_bar.dtype, h_bar.device)
            )
        update_running_avg(delta_i, self._delta[module], self.gammas)
        update_running_avg(phi_raw_i, self._Phi_raw[module], self.gammas)
        update_running_avg(psi_raw_i, self._Psi_raw[module], self.gammas)

    def _damped_factors(self, module: Module) -> Tuple[Tensor, Tensor, Tensor]:
        # .clone(): self._delta[module] is mutated in place by update_running_avg (current *=...;
        # current +=...). refresh() stores this returned `delta` in _delta_at_refresh to freeze it
        # until the next T_inv-aligned refresh; without the clone it would silently keep tracking
        # every subsequent EMA update instead, corrupting precondition()'s "same vintage as
        # Phi_inv/Psi_inv" invariant (docs/reports/plan_lot3.md §0.3).
        delta = self._delta[module].clone()
        Phi = self._Phi_raw[module] / delta
        Psi = self._Psi_raw[module] / delta
        damp = (self.Lambda / delta).sqrt()
        Phi_tilde = Phi + damp * eye(Phi.size(0), dtype=Phi.dtype, device=Phi.device)
        Psi_tilde = Psi + damp * eye(Psi.size(0), dtype=Psi.dtype, device=Psi.device)
        return delta, Phi_tilde, Psi_tilde

    def refresh(self, module: Module, step: int) -> None:
        if step % self.T_inv != 0:
            return
        delta, Phi_tilde, Psi_tilde = self._damped_factors(module)
        self._delta_at_refresh[module] = delta
        self._Phi_inv[module] = Phi_tilde.inverse()
        self._Psi_inv[module] = Psi_tilde.inverse()

    def f_tilde(self, module: Module) -> Tensor:
        """Dense ``(d_out*d_in_aug)^2`` reconstruction of ``F~_TKFAC = delta * kron(Psi~, Phi~)``
        (B outer, A inner, i.e. output outer / input inner -- plan_lot2.md §0.4), from the
        *current* damped factors. ``precondition`` never forms this matrix; it exists only for the
        Frobenius-dominance / trace-preservation tests (plan_lot3.md §0.6). Reflects any EMA drift
        since the last ``refresh()``, exactly like ``kfac.f_tilde()``.
        """
        delta, Phi_tilde, Psi_tilde = self._damped_factors(module)
        return delta * kron(Psi_tilde, Phi_tilde)

    def precondition(
        self,
        module: Module,
        weight_direction: Tensor,
        bias_direction: Optional[Tensor],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        bias_shape = None if bias_direction is None else bias_direction.shape
        delta = self._delta_at_refresh[module]
        if self.conv_sua and isinstance(module, Conv2d):
            M = augment_conv2d_direction_sua(weight_direction, bias_direction)
            direction = (self._Psi_inv[module] @ M @ self._Phi_inv[module]) / delta
            return split_conv2d_direction_sua(direction, weight_direction.shape, bias_shape)
        M = augment_direction(weight_direction, bias_direction)
        direction = (self._Psi_inv[module] @ M @ self._Phi_inv[module]) / delta
        return split_direction(direction, weight_direction.shape, bias_shape)
