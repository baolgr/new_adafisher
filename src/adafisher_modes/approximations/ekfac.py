"""``ekfac`` mode: George et al.'s eigenvalue-corrected Kronecker factorization (EKFAC). Linear
(lot 2, ``docs/reports/plan_lot2.md``) and Conv2d with ``groups=1``, ``dilation=(1,1)`` (lot 4,
``docs/reports/plan_lot4.md``).

    F_l ~= (Q_B (x) Q_A) diag(s*) (Q_B (x) Q_A)^T,   s*_i = E[((Q_B (x) Q_A)^T vec(dW))_i^2]

``Q_A``, ``Q_B`` are the eigenvectors of the *same* ``A = E[h_bar h_bar^T]``, ``B = E[delta
delta^T]`` factors K-FAC uses (``ekfac_1806.03884.pdf`` §3.2). Lemma 1 (Appendix A.1) proves that
for *any* fixed orthogonal ``Q``, the diagonal ``D`` minimizing ``||G - QDQ^T||_F`` has
``D_ii = E[(Q^T grad)_i^2]``; Theorem 2 specializes this to ``Q = Q_A (x) Q_B`` (orthogonal, since
the Kronecker product of two orthogonal matrices is orthogonal); Theorem 3 concludes
``||G - EKFAC||_F <= ||G - KFAC||_F``, since KFAC's own ``D = S_A (x) S_B`` is one particular,
generally suboptimal diagonal choice.

**s* estimation** (``docs/reports/plan_lot2.md`` §0.3): tracked as an EMA — the same Eq.-3
mechanism every factor in this codebase uses — of the *intra-batch* estimate from Algorithm 1
(``ekfac_1806.03884.pdf`` §4): ``E_n[((Q_B (x) Q_A)^T (delta_n (x) h_bar_n))^2]``. This is a
deliberate third choice, distinct from both named paper variants ("EKFAC" recomputes s* from
scratch every minibatch; "EKFAC-ra", ``EKFAC-pytorch/ekfac.py::_precond_ra``, tracks squared
minibatch-*averaged* gradients): it uses Algorithm 1's real intra-batch statistic (available here
because the hooks already expose the raw batch, not just an averaged gradient) while keeping this
mode's smoothing cadence identical to every other mode's, per ``CLAUDE.md``'s "everything but
v^(t)'s construction is identical across the five modes".

**Conv2d (lot 4, plan_lot4.md §0.5).** ``update_input_factor``/``update_output_factor`` build
``h_bar``/``s_flat`` via the layer-dispatching ``augment_input``/``flatten_output_grad`` (in place of
lot 2's ``Linear``-only ``augment_linear_input``/inline reshape); every other line — the ``A_i``
reduction, the ``s*`` intra-batch estimator, ``refresh``, ``f_tilde``, ``precondition`` — is already
shape-agnostic and needs no change, since a ``Conv2d``'s pooled ``(example, output-location)`` batch
has exactly the same ``(N, d_in_aug)``/``(N, d_out)`` shape a ``Linear``'s batch does.

**SUA (lot 6, docs/reports/plan_lot6.md).** ``conv_sua=True`` selects the channel-only SUA input
factor (``kfac_conv_1602.01407.pdf`` p. 14) for ``Conv2d`` modules: ``update_input_factor`` passes
``sua=self.conv_sua`` to ``augment_input``. ``update_output_factor``'s intra-batch ``s*`` estimator
needs **no** change: ``h_bar`` (cached from ``update_input_factor``) is already row-aligned with
``s_flat`` regardless of ``conv_sua`` (plan_lot6.md §0.3 -- both are pooled over the same ``(N,
H_out, W_out)`` grid ``extract_patches`` defines), so ``h_kfe = h_bar @ Q_A`` and the ``g2`` outer
product work unchanged; only ``Q_A``'s size differs. ``precondition`` applies the small operator
independently at each of the ``k_h*k_w`` kernel offsets instead of one flat matmul (plan_lot6.md
§0.4).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from torch import Tensor, diag, eye, kron
from torch.nn import Conv2d, Module

from adafisher_modes.ema import update_running_avg
from adafisher_modes.factors import augment_input, compute_s_full, flatten_output_grad

from ._eigh_utils import eigenbasis
from ._kron_utils import (
    augment_conv2d_direction_sua,
    augment_direction,
    split_conv2d_direction_sua,
    split_direction,
)
from .base import FisherApproximation


class EKFACApproximation(FisherApproximation):
    def __init__(
        self,
        Lambda: float = 1e-3,
        gammas: Sequence[float] = (0.92, 0.008),
        T_eig: int = 100,
        conv_sua: bool = False,
    ) -> None:
        self.Lambda = Lambda
        self.gammas = gammas
        self.T_eig = T_eig
        self.conv_sua = conv_sua
        self._A: Dict[Module, Tensor] = {}
        self._B: Dict[Module, Tensor] = {}
        self._Q_A: Dict[Module, Tensor] = {}
        self._Q_B: Dict[Module, Tensor] = {}
        self._s_star: Dict[Module, Tensor] = {}
        self._cached_h_bar: Dict[Module, Tensor] = {}

    def update_input_factor(self, module: Module, h: Tensor, step: int) -> None:
        h_bar = augment_input(h, module, sua=self.conv_sua)
        A_i = h_bar.t() @ h_bar / h_bar.size(0)
        if step == 0:
            self._A[module] = eye(A_i.size(0), dtype=A_i.dtype, device=A_i.device)
        update_running_avg(A_i, self._A[module], self.gammas)
        # Cached for update_output_factor, which fires on the same batch right after (forward
        # always precedes backward for one loss.backward() call) — see plan_lot2.md §0.3.
        self._cached_h_bar[module] = h_bar

    def update_output_factor(self, module: Module, s: Tensor, step: int) -> None:
        B_i = compute_s_full(s, module)
        if step == 0:
            self._B[module] = eye(B_i.size(0), dtype=B_i.dtype, device=B_i.device)
        update_running_avg(B_i, self._B[module], self.gammas)

        h_bar = self._cached_h_bar.pop(module, None)
        if module in self._Q_A and h_bar is not None:
            s_flat = flatten_output_grad(s, module)
            h_kfe = h_bar @ self._Q_A[module]
            s_kfe = s_flat @ self._Q_B[module]
            # Intra-batch estimate of s* (Algorithm 1's COMPUTE SCALINGS step): the per-example
            # projected gradient in the KFE is (s_kfe_n (x) h_kfe_n); its squared entries, averaged
            # over the batch, are exactly (s_kfe_n_i)^2 * (h_kfe_n_j)^2 averaged over n.
            g2 = (s_kfe.t() ** 2) @ (h_kfe**2) / h_bar.size(0)
            update_running_avg(g2, self._s_star[module], self.gammas)

    def refresh(self, module: Module, step: int) -> None:
        if step % self.T_eig != 0:
            return
        Q_A = eigenbasis(self._A[module])
        Q_B = eigenbasis(self._B[module])
        if module not in self._Q_A:
            # Bootstrap, the direct generalisation of diag's `H.new_ones(...)` (a diagonal of ones
            # is the identity): no s* estimate exists before an eigenbasis does.
            self._s_star[module] = Q_B.new_ones(Q_B.size(0), Q_A.size(0))
        self._Q_A[module] = Q_A
        self._Q_B[module] = Q_B

    def f_tilde(self, module: Module) -> Tensor:
        """Dense ``(d_out*d_in_aug)^2`` reconstruction of ``F~_EKFAC = kron(Q_B, Q_A) @
        diag(s*+lambda) @ kron(Q_B, Q_A)^T`` (B outer, A inner — plan_lot2.md §0.4), from the
        *current* eigenbasis/s*. ``precondition`` never forms this matrix; it exists only for the
        Frobenius-dominance test (plan_lot2.md §0.5).
        """
        Q_A, Q_B = self._Q_A[module], self._Q_B[module]
        scale = (self._s_star[module] + self.Lambda).flatten()
        Q = kron(Q_B, Q_A)
        return Q @ diag(scale) @ Q.t()

    def precondition(
        self,
        module: Module,
        weight_direction: Tensor,
        bias_direction: Optional[Tensor],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        Q_A, Q_B = self._Q_A[module], self._Q_B[module]
        bias_shape = None if bias_direction is None else bias_direction.shape
        if self.conv_sua and isinstance(module, Conv2d):
            M = augment_conv2d_direction_sua(weight_direction, bias_direction)
            M_kfe = (Q_B.t() @ M @ Q_A) / (self._s_star[module] + self.Lambda)
            direction = Q_B @ M_kfe @ Q_A.t()
            return split_conv2d_direction_sua(direction, weight_direction.shape, bias_shape)
        M = augment_direction(weight_direction, bias_direction)
        M_kfe = (Q_B.t() @ M @ Q_A) / (self._s_star[module] + self.Lambda)
        direction = Q_B @ M_kfe @ Q_A.t()
        return split_direction(direction, weight_direction.shape, bias_shape)
