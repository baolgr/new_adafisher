"""``kfac`` mode: Martens & Grosse's Kronecker-factored approximate curvature (K-FAC). Linear
(lot 2, ``docs/reports/plan_lot2.md``) and Conv2d with ``groups=1``, ``dilation=(1,1)`` (lot 4,
``docs/reports/plan_lot4.md`` §0.5 — this file needed **no** logic change: ``update_input_factor``/
``update_output_factor`` already dispatch through ``factors.py``'s ``compute_h_full``/
``compute_s_full``, and ``precondition`` already goes through ``_kron_utils``, both of which gained
``Conv2d`` support underneath this file unchanged).

    F_l = E[h_bar h_bar^T (x) delta delta^T] ~= E[h_bar h_bar^T] (x) E[delta delta^T] = A (x) B

(independence assumption, ``kfac_1503.05671.pdf`` §3, eq. 1). Inverted via
``(A (x) B)^-1 = A^-1 (x) B^-1`` (§4.2, "Approximating F~^-1 as block-diagonal") and applied to a
direction ``M`` (shaped like the module's gradient) through the vec identity
``(A (x) B) vec(X) = vec(B X A^T)`` (§4.2, unnumbered display equation after eq. 6:
``U_i = G_{i,i}^-1 V_i A_bar_{i-1,i-1}^-1``), which in this project's row-major convention becomes
``M -> B^-1 M A^-1`` — derived from scratch in ``docs/reports/plan_lot2.md`` §0.4 (not just cited).

Damping follows the factored Tikhonov technique of §6.3 (unnumbered display equation, p. 23):

    pi_l = sqrt( (tr(A)/d_A) / (tr(B)/d_B) )
    A~ = A + pi_l * sqrt(lambda) * I
    B~ = B + (sqrt(lambda) / pi_l) * I

Reproduced (not copied) from ``EKFAC-pytorch/kfac.py::_inv_covs``'s ``pi=True`` branch, whose local
``pi`` variable equals this ``pi_l`` squared (its code adds ``sqrt(eps*pi)``/``sqrt(eps/pi)``, i.e.
``sqrt(eps)*sqrt(pi)`` — consistent with the paper once ``pi_l = sqrt(pi)``).

**SUA (lot 6, docs/reports/plan_lot6.md).** ``conv_sua=True`` selects the channel-only SUA input
factor (``kfac_conv_1602.01407.pdf`` p. 14) for ``Conv2d`` modules instead of the patch-based one —
``update_input_factor`` passes ``sua=self.conv_sua`` to ``compute_h_full``; ``precondition`` applies
the resulting small operator independently at each of the ``k_h*k_w`` kernel offsets instead of one
flat matmul (plan_lot6.md §0.4). ``update_output_factor``, ``refresh``, ``f_tilde`` are unchanged —
none of them care whether ``A`` came from patches or from SUA's channel-only pooling, only its size
differs.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from torch import Tensor, eye, kron
from torch.nn import Conv2d, Module

from adafisher_modes.ema import update_running_avg
from adafisher_modes.factors import compute_h_full, compute_s_full

from ._kron_utils import (
    augment_conv2d_direction_sua,
    augment_direction,
    split_conv2d_direction_sua,
    split_direction,
)
from .base import FisherApproximation


class KFACApproximation(FisherApproximation):
    def __init__(
        self,
        Lambda: float = 1e-3,
        gammas: Sequence[float] = (0.92, 0.008),
        T_inv: int = 100,
        pi: bool = True,
        conv_sua: bool = False,
    ) -> None:
        self.Lambda = Lambda
        self.gammas = gammas
        self.T_inv = T_inv
        self.pi = pi
        self.conv_sua = conv_sua
        self._A: Dict[Module, Tensor] = {}
        self._B: Dict[Module, Tensor] = {}
        self._A_inv: Dict[Module, Tensor] = {}
        self._B_inv: Dict[Module, Tensor] = {}

    def update_input_factor(self, module: Module, h: Tensor, step: int) -> None:
        A_i = compute_h_full(h, module, sua=self.conv_sua)
        if step == 0:
            self._A[module] = eye(A_i.size(0), dtype=A_i.dtype, device=A_i.device)
        update_running_avg(A_i, self._A[module], self.gammas)

    def update_output_factor(self, module: Module, s: Tensor, step: int) -> None:
        B_i = compute_s_full(s, module)
        if step == 0:
            self._B[module] = eye(B_i.size(0), dtype=B_i.dtype, device=B_i.device)
        update_running_avg(B_i, self._B[module], self.gammas)

    def _pi(self, A: Tensor, B: Tensor) -> Tensor:
        """Factored Tikhonov scalar pi_l (kfac_1503.05671.pdf §6.3). ``pi=False`` reduces to the
        plain, non-factored damping split (pi_l = 1).
        """
        if not self.pi:
            return A.new_tensor(1.0)
        return ((A.trace() / A.size(0)) / (B.trace() / B.size(0))).sqrt()

    def _damped_factors(self, module: Module) -> Tuple[Tensor, Tensor]:
        A, B = self._A[module], self._B[module]
        pi = self._pi(A, B)
        lambda_sqrt = self.Lambda**0.5
        A_tilde = A + (pi * lambda_sqrt) * eye(A.size(0), dtype=A.dtype, device=A.device)
        B_tilde = B + (lambda_sqrt / pi) * eye(B.size(0), dtype=B.dtype, device=B.device)
        return A_tilde, B_tilde

    def refresh(self, module: Module, step: int) -> None:
        if step % self.T_inv != 0:
            return
        A_tilde, B_tilde = self._damped_factors(module)
        self._A_inv[module] = A_tilde.inverse()
        self._B_inv[module] = B_tilde.inverse()

    def f_tilde(self, module: Module) -> Tensor:
        """Dense ``(d_out*d_in_aug)^2`` reconstruction of ``F~_KFAC = kron(B~, A~)`` (B outer, A
        inner — see plan_lot2.md §0.4), from the *current* damped factors. ``precondition`` never
        forms this matrix; it exists only for the Frobenius-dominance test (plan_lot2.md §0.5).
        Reflects any EMA drift since the last ``refresh()`` — call ``refresh()`` immediately before
        this for a matched comparison.
        """
        A_tilde, B_tilde = self._damped_factors(module)
        return kron(B_tilde, A_tilde)

    def precondition(
        self,
        module: Module,
        weight_direction: Tensor,
        bias_direction: Optional[Tensor],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        bias_shape = None if bias_direction is None else bias_direction.shape
        if self.conv_sua and isinstance(module, Conv2d):
            M = augment_conv2d_direction_sua(weight_direction, bias_direction)
            direction = self._B_inv[module] @ M @ self._A_inv[module]
            return split_conv2d_direction_sua(direction, weight_direction.shape, bias_shape)
        M = augment_direction(weight_direction, bias_direction)
        direction = self._B_inv[module] @ M @ self._A_inv[module]
        return split_direction(direction, weight_direction.shape, bias_shape)
