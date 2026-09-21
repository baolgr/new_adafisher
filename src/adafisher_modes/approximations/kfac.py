"""``kfac`` mode: Martens & Grosse's Kronecker-factored approximate curvature.

    F_l = E[h_bar h_bar^T (x) delta delta^T] ~= E[h_bar h_bar^T] (x) E[delta delta^T] = A (x) B

``h_bar`` is the layer's bias-augmented input and ``delta`` the gradient at its output. Replacing
the expectation of the Kronecker product by the Kronecker product of the expectations is K-FAC's
independence assumption (``kfac_1503.05671.pdf`` section 3, eq. 1). It buys
``(A (x) B)^-1 = A^-1 (x) B^-1`` (section 4.2), so a direction is preconditioned by two matrix
products instead of one large solve.

**Applied form.** PyTorch lays a weight out as ``(d_out, d_in)``, so flattening it row by row is the
``rvec`` convention, while the K-FAC literature uses the column-major ``cvec``. From
``vec_r(u v^T) = u (x) v`` one gets ``vec_r(B M A^T) = (B (x) A) vec_r(M)``, so the operator this
file must build is ``kron(B, A)`` -- output factor outer, input factor inner -- and applying its
inverse to a direction ``M`` is ``B^-1 M A^-1``. This is the transpose trap the K-FAC-from-scratch
notes (``kfac_from_scratch_2507.05127.pdf``, Def. 1/2 and Def. 23) warn about: on square factors a
swap is completely silent.

**Damping** follows the factored Tikhonov technique of section 6.3 (p. 23)::

    pi   = sqrt( (tr(A) / dim(A)) / (tr(B) / dim(B)) )
    A~   = A + pi * sqrt(lambda) * I
    B~   = B + (sqrt(lambda) / pi) * I

Expanding ``A~ (x) B~`` recovers ``A (x) B + lambda I`` plus two cross terms; ``pi`` is the value
that minimises the trace norm of those cross terms. ``pi=False`` sets it to 1, the unsplit version.
The same formula appears in ``EKFAC-pytorch/kfac.py::_inv_covs``, whose local ``pi`` variable is
this ``pi`` squared.

**Cadence.** ``update_input_factor`` and ``update_output_factor`` fold each new observation into the
running average (:mod:`adafisher_modes.ema`) from the raw statistics built by
:mod:`adafisher_modes.factors`. ``refresh`` recomputes the two inverses, but only on steps that are
multiples of ``T_inv``; ``precondition`` uses whatever inverses that last refresh left behind.

**SUA** (``conv_sua=True``, ``Conv2d`` only) replaces the patch-based input factor, of width
``C_in * k_h * k_w [+1]``, by a channel-only one of width ``C_in [+1]``
(``kfac_conv_1602.01407.pdf`` p. 14). Nothing in the rescaling changes; the same small operator is
applied independently at each of the ``k_h * k_w`` kernel offsets instead of once to a single flat
matrix. Measured at ResNet-18 scale, the widest input factor drops from 4609 to 513 entries per
side, about 80.7 times fewer bytes.

``f_tilde`` builds the dense operator and exists only for tests; ``precondition`` never forms it.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from torch import Tensor, eye, kron
from torch.nn import Conv2d, Module

from adafisher_modes.ema import seed_or_accumulate
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
        ema_seed_first: bool = False,
    ) -> None:
        self.Lambda = Lambda
        self.gammas = gammas
        self.T_inv = T_inv
        self.pi = pi
        self.conv_sua = conv_sua
        self.ema_seed_first = ema_seed_first
        self._A: Dict[Module, Tensor] = {}
        self._B: Dict[Module, Tensor] = {}
        self._A_inv: Dict[Module, Tensor] = {}
        self._B_inv: Dict[Module, Tensor] = {}
        self._lambda_at_refresh: Dict[Module, Union[float, Tensor]] = {}

    def update_input_factor(self, module: Module, h: Tensor, step: int) -> None:
        A_i = compute_h_full(h, module, sua=self.conv_sua)
        seed_or_accumulate(A_i, self._A, module,
                           lambda: eye(A_i.size(0), dtype=A_i.dtype, device=A_i.device),
                           self.gammas, step, self.ema_seed_first)

    def update_output_factor(self, module: Module, s: Tensor, step: int) -> None:
        B_i = compute_s_full(s, module)
        seed_or_accumulate(B_i, self._B, module,
                           lambda: eye(B_i.size(0), dtype=B_i.dtype, device=B_i.device),
                           self.gammas, step, self.ema_seed_first)

    def _pi(self, A: Tensor, B: Tensor) -> Tensor:
        """The factored Tikhonov scalar ``pi`` of ``kfac_1503.05671.pdf`` §6.3, the value that
        minimises the trace norm of the two cross terms ``A~ (x) B~`` introduces. ``pi=False``
        returns 1, the unsplit damping.
        """
        if not self.pi:
            return A.new_tensor(1.0)
        return ((A.trace() / A.size(0)) / (B.trace() / B.size(0))).sqrt()

    def _damped_factors(self, module: Module) -> Tuple[Tensor, Tensor]:
        A, B = self._A[module], self._B[module]
        pi = self._pi(A, B)
        lambda_sqrt = self.lambda_for(module)**0.5
        A_tilde = A + (pi * lambda_sqrt) * eye(A.size(0), dtype=A.dtype, device=A.device)
        B_tilde = B + (lambda_sqrt / pi) * eye(B.size(0), dtype=B.dtype, device=B.device)
        return A_tilde, B_tilde

    def refresh(self, module: Module, step: int) -> None:
        # "or nothing cached yet" covers a module first reached after step 0, whose first step is
        # not necessarily a multiple of T_inv; precondition() would otherwise read an inverse that
        # was never built. It cannot fire on a module present from step 0, since step 0 is a
        # multiple of every T_inv, so no existing trajectory changes.
        if step % self.T_inv != 0 and module in self._A_inv:
            return
        A_tilde, B_tilde = self._damped_factors(module)
        self._A_inv[module] = A_tilde.inverse()
        self._B_inv[module] = B_tilde.inverse()
        self._lambda_at_refresh[module] = self.lambda_for(module)

    def applied_lambda(self, module: Module) -> Union[float, Tensor]:
        """The damping baked into the cached inverses, which is the one ``precondition`` applies
        until the next refresh -- not whatever :meth:`lambda_for` says now."""
        return self._lambda_at_refresh[module]

    def mean_curvature(self, module: Module) -> Optional[Tensor]:
        """``(tr A / d_in) * (tr B / d_out)``: the eigenvalues of ``A (x) B`` are the products
        ``a_i b_j``, so their mean is the product of the two means."""
        if module not in self._A or module not in self._B:
            return None
        A, B = self._A[module], self._B[module]
        return (A.trace() / A.size(0)) * (B.trace() / B.size(0))

    def num_directions(self, module: Module) -> Optional[int]:
        if module not in self._A or module not in self._B:
            return None
        return self._A[module].size(0) * self._B[module].size(0)

    def f_tilde(self, module: Module) -> Tensor:
        """Dense ``(d_out * d_in_aug)^2`` reconstruction of ``F~ = kron(B~, A~)``: output factor
        outer, input factor inner, which is the ``rvec`` ordering the module docstring derives.

        Built from the *current* running averages, so it reflects any drift since the last
        ``refresh()``; call ``refresh()`` immediately before it for a matched comparison against
        what ``precondition`` would apply. For tests and debugging only -- ``precondition`` never
        forms this matrix.
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
