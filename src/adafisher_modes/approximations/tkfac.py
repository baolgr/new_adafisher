"""``tkfac`` mode: Gao et al.'s trace-restricted Kronecker-factored approximate curvature.

    F_l = E[Lambda (x) Gamma],   Lambda = h_bar h_bar^T,   Gamma = delta delta^T

is approximated by a Kronecker product carrying an explicit scalar
(``tkfac_2011.10741.pdf`` eq. 4.3)::

    F_l ~= delta_l * Phi_l (x) Psi_l

Theorem 4.1, in the simplified form of eq. (4.9) that the paper itself adopts for all its
experiments::

    delta_l = E[tr(Lambda) tr(Gamma)]
    Phi_l   = E[tr(Gamma) Lambda] / delta_l
    Psi_l   = E[tr(Lambda) Gamma] / delta_l

so that ``tr(Phi_l) = tr(Psi_l) = 1`` and therefore ``tr(F~) = delta_l = tr(F)`` **exactly**
(Lemma 4.1). Preserving the trace of the true block is the property this mode exists for.

**Why the numerators are stored undivided.** ``delta``, ``Phi * delta`` and ``Psi * delta`` are
averaged separately, and the division happens only when the factors are used. Averaging the ratio
instead would break the trace identity, because this package's running average does not have
coefficients summing to 1 (:mod:`adafisher_modes.ema`): ``tr(Phi_raw) = delta`` survives the
average only while both sides are scaled by the same coefficients. The three per-batch numerators
are built in :mod:`adafisher_modes.approximations._tkfac_utils`, shared with ``tekfac``.

**Damping.** Eq. (5.15) of the paper adds ``sqrt(lambda)`` to ``sqrt(delta) Phi`` and to
``sqrt(delta) Psi``, which expands to ``delta Phi (x) Psi + ... + lambda I``. Factoring
``sqrt(delta)`` out of both gives the equivalent form used here::

    Phi~ = Phi + sqrt(lambda / delta) * I,   Psi~ = Psi + sqrt(lambda / delta) * I
    F~^-1 M  =  (1 / delta) * Psi~^-1 M Phi~^-1

in the ``rvec`` convention (output factor on the left, input factor on the right -- see the
:mod:`~adafisher_modes.approximations.kfac` docstring for the derivation). The paper's separate,
convolution-only adaptive damping (its eq. 5.16) is deliberately not implemented; the same
dimension-agnostic scalar is used for every layer type.

**Cadence.** ``refresh`` recomputes the two inverses on steps that are multiples of ``T_inv``, and
freezes the value of ``delta`` alongside them so that ``precondition`` divides by the same
``delta`` the cached inverses were built from.

**SUA** (``conv_sua=True``, ``Conv2d`` only) makes the input factor channel-only instead of
patch-based (``kfac_conv_1602.01407.pdf`` p. 14); the statistics above are already generic over the
input width, and ``precondition`` applies the small operator independently at each kernel offset.

``f_tilde`` builds the dense operator and exists only for tests; ``precondition`` never forms it.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from torch import Tensor, eye, kron
from torch.nn import Conv2d, Module

from adafisher_modes.ema import seed_or_accumulate
from adafisher_modes.factors import augment_input, flatten_output_grad

from ._kron_utils import (
    augment_conv2d_direction_sua,
    augment_direction,
    split_conv2d_direction_sua,
    split_direction,
)
from ._tkfac_utils import bootstrap_raw_factors, instantaneous_raw_factors
from .base import FisherApproximation, pop_cached_input


class TKFACApproximation(FisherApproximation):
    def __init__(
        self,
        Lambda: float = 1e-3,
        gammas: Sequence[float] = (0.92, 0.008),
        T_inv: int = 100,
        conv_sua: bool = False,
        ema_seed_first: bool = False,
    ) -> None:
        self.Lambda = Lambda
        self.gammas = gammas
        self.T_inv = T_inv
        self.conv_sua = conv_sua
        self.ema_seed_first = ema_seed_first
        self._delta: Dict[Module, Tensor] = {}
        self._Phi_raw: Dict[Module, Tensor] = {}
        self._Psi_raw: Dict[Module, Tensor] = {}
        self._Phi_inv: Dict[Module, Tensor] = {}
        self._Psi_inv: Dict[Module, Tensor] = {}
        self._delta_at_refresh: Dict[Module, Tensor] = {}
        self._cached_h_bar: Dict[Module, Tensor] = {}

    def update_input_factor(self, module: Module, h: Tensor, step: int) -> None:
        # Every statistic this mode keeps mixes the input and the gradient of the same example, and
        # the step-0 bootstrap needs both widths at once, so nothing can be accumulated until the
        # backward hook fires. The forward hook therefore only caches h_bar; the forward pass always
        # precedes the backward pass of one loss.backward() call.
        self._cached_h_bar[module] = augment_input(h, module, sua=self.conv_sua)

    def update_output_factor(self, module: Module, s: Tensor, step: int) -> None:
        h_bar = pop_cached_input(self._cached_h_bar, module)
        s_flat = flatten_output_grad(s, module)
        delta_i, phi_raw_i, psi_raw_i = instantaneous_raw_factors(h_bar, s_flat)
        # The bootstrap is needed whenever the running average is starting, which is step 0 and
        # also the first time this module is seen at all -- it can first be reached at step 7,
        # behind a conditional branch. seed_or_accumulate starts on the same condition.
        boot = (
            bootstrap_raw_factors(h_bar.size(1), s_flat.size(1), h_bar.dtype, h_bar.device)
            if (step == 0 or module not in self._delta) and not self.ema_seed_first
            else (None, None, None)
        )
        for new, store, identity in (
            (delta_i, self._delta, boot[0]),
            (phi_raw_i, self._Phi_raw, boot[1]),
            (psi_raw_i, self._Psi_raw, boot[2]),
        ):
            seed_or_accumulate(new, store, module, lambda t=identity: t,
                               self.gammas, step, self.ema_seed_first)

    def _damped_factors(self, module: Module) -> Tuple[Tensor, Tensor, Tensor]:
        # .clone() is load-bearing: update_running_avg mutates self._delta[module] in place, and
        # refresh() stores what this returns in _delta_at_refresh to freeze it until the next
        # refresh. Without the clone that stored value would keep tracking every later update, and
        # precondition() would divide by a delta of a different vintage than its cached inverses.
        delta = self._delta[module].clone()
        Phi = self._Phi_raw[module] / delta
        Psi = self._Psi_raw[module] / delta
        damp = (self.Lambda / delta).sqrt()
        Phi_tilde = Phi + damp * eye(Phi.size(0), dtype=Phi.dtype, device=Phi.device)
        Psi_tilde = Psi + damp * eye(Psi.size(0), dtype=Psi.dtype, device=Psi.device)
        return delta, Phi_tilde, Psi_tilde

    def refresh(self, module: Module, step: int) -> None:
        # "or nothing cached yet" covers a module first reached after step 0, whose first step is
        # not necessarily a multiple of T_inv; precondition() would otherwise read an inverse that
        # was never built. It cannot fire on a module present from step 0, since step 0 is a
        # multiple of every T_inv, so no existing trajectory changes.
        if step % self.T_inv != 0 and module in self._Phi_inv:
            return
        delta, Phi_tilde, Psi_tilde = self._damped_factors(module)
        self._delta_at_refresh[module] = delta
        self._Phi_inv[module] = Phi_tilde.inverse()
        self._Psi_inv[module] = Psi_tilde.inverse()

    def f_tilde(self, module: Module) -> Tensor:
        """Dense ``(d_out * d_in_aug)^2`` reconstruction of ``F~ = delta * kron(Psi~, Phi~)``:
        output factor outer, input factor inner. Built from the *current* running averages, so it
        reflects any drift since the last ``refresh()``, exactly like ``kfac``'s. For tests and
        debugging only -- ``precondition`` never forms this matrix.
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
