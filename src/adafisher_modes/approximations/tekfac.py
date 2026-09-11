"""``tekfac`` mode: Gao et al.'s Trace-restricted Eigenvalue-corrected Kronecker Factorization
(TEKFAC). Linear (lot 3, ``docs/reports/plan_lot3.md``) and Conv2d with ``groups=1``,
``dilation=(1,1)`` (lot 4, ``docs/reports/plan_lot4.md``).

TEKFAC combines TKFAC's trace-restricted factors with EKFAC's eigenbasis correction
(``tekfac_2011.13609.pdf`` §3.1). It reuses **the same** ``(delta, Phi_raw, Psi_raw)`` triple
``tkfac.py`` maintains (Eq. 2.9-2.10, identical to TKFAC's own Eq. 4.9) -- see
``docs/reports/plan_lot3.md`` §0.4 -- rather than the undivided K-FAC factors ``A``, ``B``:

    F_l ~= delta_l * Phi_l (x) Psi_l = delta_l * (Q_Phi (x) Q_Psi)(Lambda_Phi (x) Lambda_Psi)(Q_Phi (x) Q_Psi)^T   (Eq. 3.1)

Exactly EKFAC's own Lemma 1 argument, applied to the orthogonal basis ``Q_Phi (x) Q_Psi`` instead
of ``Q_A (x) Q_B``, replaces the still-inexact rescaling ``delta_l*(Lambda_Phi (x) Lambda_Psi)`` by
the directly-estimated optimal diagonal (Eq. 3.2-3.3; note the absence of a separate ``delta``
multiplying ``Theta`` -- it is already the right scale, estimated directly from the raw gradient):

    Theta_ii = E[((Q_Phi (x) Q_Psi)^T grad_omega h)_i^2],   F_l ~= (Q_Phi (x) Q_Psi) Theta_l (Q_Phi (x) Q_Psi)^T

Theorem 3.1: ``||F - F~_TEKFAC||_F <= ||F - F~_TKFAC||_F``, by the same Lemma-1 optimality argument
EKFAC's Theorem 2/3 uses. ``Theta``'s estimator (below) is structurally identical to ``ekfac.py``'s
``s*`` estimator with ``(Q_A,Q_B) -> (Q_Phi,Q_Psi)`` -- the degenerate-case test
(``tests/test_ekfac_tekfac_equiv.py``) checks exactly this correspondence.

Damping (Eq. 3.4-3.5): a plain additive ``lambda`` to ``Theta`` for dense layers. The paper's own
conv-only trace-adaptive ``lambda``/CNN-wide ``beta`` rescaling of Eq. 3.5 is not implemented — this
mode's existing, dimension-agnostic additive ``lambda`` is reused unchanged for Conv2d (lot 4,
``docs/reports/plan_lot4.md`` §0.6).

Decoupled cadences (Algorithm 1, plan_lot3.md §0.4): ``T_eig`` gates the eigenbasis refresh (as in
``ekfac``); ``T_re`` independently gates ``Theta``'s own EMA update. No separate ``T_fim`` --
``(delta, Phi_raw, Psi_raw)`` update on every hook fire, like ``kfac``/``ekfac``'s ``A``/``B``.
``beta_factors``/``beta_theta`` are this project's own ``gammas``-shaped EMA rate for
``(delta,Phi_raw,Psi_raw)`` and for ``Theta`` respectively (TEKFAC's own beta_1/beta_2 of Eq.
3.6-3.8, renamed to avoid the notation collision with Adam/AdaFisher's beta_1 -- CLAUDE.md).

**Conv2d (lot 4, plan_lot4.md §0.5).** Same two swaps as ``tkfac.py``:
``augment_input``/``flatten_output_grad`` replace lot 3's ``Linear``-only
``augment_linear_input``/inline reshape in ``update_input_factor``/``update_output_factor``; nothing
else changes.

**SUA (lot 6, docs/reports/plan_lot6.md).** ``conv_sua=True`` selects the channel-only SUA input
factor (``kfac_conv_1602.01407.pdf`` p. 14) for ``Conv2d`` modules: ``update_input_factor`` caches
``augment_input(h, module, sua=self.conv_sua)``. ``precondition`` applies the small operator
independently at each of the ``k_h*k_w`` kernel offsets instead of one flat matmul (plan_lot6.md
§0.4).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from torch import Tensor, diag, kron
from torch.nn import Conv2d, Module

from adafisher_modes.ema import update_running_avg
from adafisher_modes.factors import augment_input, flatten_output_grad

from ._eigh_utils import eigenbasis
from ._kron_utils import (
    augment_conv2d_direction_sua,
    augment_direction,
    split_conv2d_direction_sua,
    split_direction,
)
from ._tkfac_utils import bootstrap_raw_factors, instantaneous_raw_factors
from .base import FisherApproximation


class TEKFACApproximation(FisherApproximation):
    def __init__(
        self,
        Lambda: float = 1e-3,
        gammas: Sequence[float] = (0.92, 0.008),
        beta_factors: Optional[Sequence[float]] = None,
        beta_theta: Optional[Sequence[float]] = None,
        T_eig: int = 100,
        T_re: int = 1,
        conv_sua: bool = False,
    ) -> None:
        self.Lambda = Lambda
        self.beta_factors = tuple(beta_factors) if beta_factors is not None else tuple(gammas)
        self.beta_theta = tuple(beta_theta) if beta_theta is not None else tuple(gammas)
        self.T_eig = T_eig
        self.T_re = T_re
        self.conv_sua = conv_sua
        self._delta: Dict[Module, Tensor] = {}
        self._Phi_raw: Dict[Module, Tensor] = {}
        self._Psi_raw: Dict[Module, Tensor] = {}
        self._Q_Phi: Dict[Module, Tensor] = {}
        self._Q_Psi: Dict[Module, Tensor] = {}
        self._Theta: Dict[Module, Tensor] = {}
        self._cached_h_bar: Dict[Module, Tensor] = {}

    def update_input_factor(self, module: Module, h: Tensor, step: int) -> None:
        self._cached_h_bar[module] = augment_input(h, module, sua=self.conv_sua)

    def update_output_factor(self, module: Module, s: Tensor, step: int) -> None:
        h_bar = self._cached_h_bar.pop(module)
        s_flat = flatten_output_grad(s, module)
        delta_i, phi_raw_i, psi_raw_i = instantaneous_raw_factors(h_bar, s_flat)
        if step == 0:
            self._delta[module], self._Phi_raw[module], self._Psi_raw[module] = (
                bootstrap_raw_factors(h_bar.size(1), s_flat.size(1), h_bar.dtype, h_bar.device)
            )
        update_running_avg(delta_i, self._delta[module], self.beta_factors)
        update_running_avg(phi_raw_i, self._Phi_raw[module], self.beta_factors)
        update_running_avg(psi_raw_i, self._Psi_raw[module], self.beta_factors)

        if module in self._Q_Phi and step % self.T_re == 0:
            h_kfe = h_bar @ self._Q_Phi[module]
            s_kfe = s_flat @ self._Q_Psi[module]
            # Intra-batch estimate of Theta (Eq. 3.2), same construction as ekfac's s* estimator
            # (plan_lot2.md §0.3) with (Q_A,Q_B) -> (Q_Phi,Q_Psi).
            theta_i = (s_kfe.t() ** 2) @ (h_kfe**2) / h_bar.size(0)
            update_running_avg(theta_i, self._Theta[module], self.beta_theta)

    def refresh(self, module: Module, step: int) -> None:
        if step % self.T_eig != 0:
            return
        # Eigenvectors of Phi_raw/Psi_raw == eigenvectors of Phi_raw/delta, Psi_raw/delta
        # (plan_lot3.md §0.4): dividing a symmetric matrix by a positive scalar leaves its
        # eigenvectors unchanged, so eigh can skip the division.
        Q_Phi = eigenbasis(self._Phi_raw[module])
        Q_Psi = eigenbasis(self._Psi_raw[module])
        if module not in self._Q_Phi:
            # Bootstrap, the direct generalisation of ekfac's `s*` bootstrap (plan_lot2.md §0.3):
            # no Theta estimate exists before an eigenbasis does.
            self._Theta[module] = Q_Psi.new_ones(Q_Psi.size(0), Q_Phi.size(0))
        self._Q_Phi[module] = Q_Phi
        self._Q_Psi[module] = Q_Psi

    def f_tilde(self, module: Module) -> Tensor:
        """Dense ``(d_out*d_in_aug)^2`` reconstruction of ``F~_TEKFAC = kron(Q_Psi, Q_Phi) @
        diag(Theta+Lambda) @ kron(Q_Psi, Q_Phi)^T`` (B outer, A inner -- plan_lot2.md §0.4), from
        the *current* eigenbasis/Theta. ``precondition`` never forms this matrix; it exists only
        for the Frobenius-dominance test (plan_lot3.md §0.6).
        """
        Q_Phi, Q_Psi = self._Q_Phi[module], self._Q_Psi[module]
        scale = (self._Theta[module] + self.Lambda).flatten()
        Q = kron(Q_Psi, Q_Phi)
        return Q @ diag(scale) @ Q.t()

    def precondition(
        self,
        module: Module,
        weight_direction: Tensor,
        bias_direction: Optional[Tensor],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        Q_Phi, Q_Psi = self._Q_Phi[module], self._Q_Psi[module]
        bias_shape = None if bias_direction is None else bias_direction.shape
        if self.conv_sua and isinstance(module, Conv2d):
            M = augment_conv2d_direction_sua(weight_direction, bias_direction)
            M_kfe = (Q_Psi.t() @ M @ Q_Phi) / (self._Theta[module] + self.Lambda)
            direction = Q_Psi @ M_kfe @ Q_Phi.t()
            return split_conv2d_direction_sua(direction, weight_direction.shape, bias_shape)
        M = augment_direction(weight_direction, bias_direction)
        M_kfe = (Q_Psi.t() @ M @ Q_Phi) / (self._Theta[module] + self.Lambda)
        direction = Q_Psi @ M_kfe @ Q_Phi.t()
        return split_direction(direction, weight_direction.shape, bias_shape)
