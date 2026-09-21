"""``ekfac`` mode: George et al.'s eigenvalue-corrected Kronecker factorization.

    F_l ~= (Q_B (x) Q_A) diag(s*) (Q_B (x) Q_A)^T,   s*_i = E[((Q_B (x) Q_A)^T vec(dW))_i^2]

``Q_A`` and ``Q_B`` are the eigenvectors of the *same* two factors K-FAC uses: ``A``, the second
moment of the bias-augmented input, and ``B``, the second moment of the gradient at the layer's
output (``ekfac_1806.03884.pdf`` section 3.2). What EKFAC changes is the rescaling inside that
basis. Lemma 1 (Appendix A.1) shows that for any fixed orthogonal ``Q`` the diagonal ``D``
minimising ``||G - Q D Q^T||_F`` has ``D_ii = E[(Q^T grad)_i^2]``; Theorem 2 applies it to
``Q_B (x) Q_A``, which is orthogonal because a Kronecker product of orthogonal matrices is; Theorem
3 concludes ``||F - EKFAC||_F <= ||F - KFAC||_F``, since K-FAC's own diagonal -- the product of the
two factors' eigenvalues -- is one particular, generally suboptimal choice.

**How ``s*`` is estimated here.** It is a running average (:mod:`adafisher_modes.ema`, the same
mechanism every other factor in this package uses) of Algorithm 1's *intra-batch* statistic
``E_n[((Q_B (x) Q_A)^T (delta_n (x) h_bar_n))^2]``. That is a deliberate third choice, different
from both variants the paper names: "EKFAC" recomputes ``s*`` from scratch every minibatch, and
"EKFAC-ra" (``EKFAC-pytorch/ekfac.py::_precond_ra``) tracks the square of the *batch-averaged*
gradient. The hooks here already expose the raw per-example batch, so the real intra-batch statistic
is available, and using a running average keeps this mode's smoothing cadence identical to every
other mode's. Do not "fix" this to match ``EKFAC-pytorch`` literally.

**Applied form.** Same ``rvec`` convention as ``kfac``: projecting is ``Q_B^T M Q_A``, rescaling is
an element-wise division by ``s* + lambda``, and projecting back is ``Q_B (...) Q_A^T``. ``s*`` is
stored as a ``(d_out, d_in_aug)`` matrix whose row-major flattening matches ``kron(Q_B, Q_A)``'s own
row order.

**Cadence and start-up.** ``refresh`` recomputes the two eigenbases, but only on steps that are
multiples of ``T_eig``, through :func:`~adafisher_modes.approximations._eigh_utils.eigenbasis`.
Until an eigenbasis exists there is nothing to project into, so ``s*`` is seeded to all ones the
first time ``refresh`` runs and the projected-gradient estimate only starts on the next hook fire.
For that first window the mode is therefore plain momentum SGD scaled by ``1 / (1 + lambda)``.

**Which basis ``s*`` is measured in (``eig_before_rescale``, off by default).** Within one
iteration the hooks fire first and ``step()`` second, so by default the order is: project the
gradient into the *current* eigenbasis and fold it into ``s*``; then ``refresh`` replaces that
eigenbasis, leaving ``s*`` untouched; then ``precondition`` divides using the **new** basis and the
**old** ``s*``. Lemma 1 makes ``D_ii = E[(Q^T grad)_i^2]`` optimal for the ``Q`` it was measured in
and for no other, and Algorithm 1 of the paper orders it the other way round -- eigenbasis first,
rescaling second, which is also what
``reference_repos/EKFAC-pytorch/ekfac.py::step`` does (``_compute_kfe`` before ``_precond_intra``).
``eig_before_rescale=True`` rebuilds the eigenbasis inside the backward hook, from ``A`` and ``B``
as they stand for that step, *before* projecting the gradient; ``refresh`` then does not redo it,
so ``s*`` and ``precondition`` share one basis. It is off by default because turning it on changes
every trajectory, and because at the shipped ``lambda = 1e-3`` the difference does not reach the
applied step -- ``lambda`` dominates ``s*`` there. It does reach it as ``lambda`` is lowered.

**SUA** (``conv_sua=True``, ``Conv2d`` only) makes the input factor channel-only instead of
patch-based (``kfac_conv_1602.01407.pdf`` p. 14). The ``s*`` estimator needs no change: the cached
input batch stays row-aligned with the pooled output gradients either way, because both are pooled
over the same output grid. Only ``Q_A``'s size changes, and ``precondition`` applies the resulting
small operator independently at each kernel offset.

**How the division is done (``rescale_form``, ``"add"`` by default).** ``"add"`` divides by
``s* + lambda`` as above. ``"floor"`` divides by ``max(s*, lambda)``, and ``"clip"`` replaces the
safety constant by a Sophia-type per-coordinate cap on the undamped step, with a threshold that is
recomputed every step (``"quantile"``), follows the momentum's recent size (``"ema"``), or is one
constant for the whole network (``"fixed"``). Both alternatives live in
:mod:`adafisher_modes.approximations._rescale_utils`; ``docs/reports/plan_floor_clip.md`` is the
experiment they exist for.

``f_tilde`` builds the dense operator and exists only for tests; ``precondition`` never forms it.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from torch import Tensor, diag, eye, kron
from torch.nn import BatchNorm2d, Conv2d, LayerNorm, Module

from adafisher_modes.ema import seed_or_accumulate, update_running_avg
from adafisher_modes.factors import (
    augment_input,
    compute_s_full,
    flatten_output_grad,
    norm_exact_kfe_squares,
    normalized_norm_input,
)

from ._eigh_utils import eigenbasis
from ._kron_utils import (
    augment_conv2d_direction_sua,
    augment_direction,
    split_conv2d_direction_sua,
    split_direction,
)
from ._rescale_utils import ClipRule, check_rescale_args, floored, rescale
from .base import FisherApproximation, pop_cached_input


class EKFACApproximation(FisherApproximation):
    def __init__(
        self,
        Lambda: float = 1e-3,
        gammas: Sequence[float] = (0.92, 0.008),
        T_eig: int = 100,
        conv_sua: bool = False,
        ema_seed_first: bool = False,
        eig_before_rescale: bool = False,
        rescale_form: str = "add",
        clip_fraction: Optional[float] = None,
        clip_guard: float = 1e-3,
        clip_threshold: str = "quantile",
        clip_ema_horizon: Optional[int] = None,
        clip_calibrate_at: Optional[int] = None,
        clip_calibration_window: Optional[int] = None,
        norm_exact_rescaling: bool = False,
    ) -> None:
        check_rescale_args(rescale_form, clip_fraction, clip_guard, clip_threshold,
                           clip_ema_horizon, clip_calibrate_at, clip_calibration_window)
        self.Lambda = Lambda
        self.gammas = gammas
        self.T_eig = T_eig
        self.conv_sua = conv_sua
        self.ema_seed_first = ema_seed_first
        self.eig_before_rescale = eig_before_rescale
        self.rescale_form = rescale_form
        self.clip_fraction = clip_fraction
        self.clip_guard = clip_guard
        self.norm_exact_rescaling = norm_exact_rescaling
        self._cached_x_hat: Dict[Module, Tensor] = {}
        self._clip: Optional[ClipRule] = None
        if rescale_form == "clip":
            assert clip_fraction is not None  # check_rescale_args
            self._clip = ClipRule(clip_threshold, clip_fraction, clip_guard, clip_ema_horizon,
                                  clip_calibrate_at, clip_calibration_window)
        self._s_star_observed: set = set()
        self._eig_step: Dict[Module, int] = {}
        self._A: Dict[Module, Tensor] = {}
        self._B: Dict[Module, Tensor] = {}
        self._Q_A: Dict[Module, Tensor] = {}
        self._Q_B: Dict[Module, Tensor] = {}
        self._s_star: Dict[Module, Tensor] = {}
        self._cached_h_bar: Dict[Module, Tensor] = {}

    def update_input_factor(self, module: Module, h: Tensor, step: int) -> None:
        h_bar = augment_input(h, module, sua=self.conv_sua)
        A_i = h_bar.t() @ h_bar / h_bar.size(0)
        seed_or_accumulate(A_i, self._A, module,
                           lambda: eye(A_i.size(0), dtype=A_i.dtype, device=A_i.device),
                           self.gammas, step, self.ema_seed_first)
        # Cached for update_output_factor, which fires on the same batch right after: the forward
        # pass always precedes the backward pass of one loss.backward() call. The intra-batch s*
        # estimator needs the input and the gradient row-paired, which is why this is cached rather
        # than recomputed.
        self._cached_h_bar[module] = h_bar
        if self.norm_exact_rescaling and isinstance(module, (BatchNorm2d, LayerNorm)):
            self._cached_x_hat[module] = normalized_norm_input(h, module)

    def update_output_factor(self, module: Module, s: Tensor, step: int) -> None:
        B_i = compute_s_full(s, module)
        seed_or_accumulate(B_i, self._B, module,
                           lambda: eye(B_i.size(0), dtype=B_i.dtype, device=B_i.device),
                           self.gammas, step, self.ema_seed_first)

        h_bar = pop_cached_input(self._cached_h_bar, module)
        x_hat = self._cached_x_hat.pop(module, None)
        if self.eig_before_rescale and step % self.T_eig == 0:
            # Rebuild the eigenbasis from A and B as they stand for *this* step, before the
            # gradient is projected into it, so that s* is measured in the basis precondition()
            # will then use. refresh() sees _eig_step and does not redo it. Off by default; see the
            # class docstring.
            self._rebuild_eigenbasis(module)
            self._eig_step[module] = step
        if module in self._Q_A:
            s_flat = flatten_output_grad(s, module)
            if x_hat is not None:
                # norm_exact_rescaling: the normalisation layer's exact per-row gradient
                # [delta * x_hat, delta], not the input factor's surrogate [z * delta, delta].
                g2 = norm_exact_kfe_squares(x_hat, s_flat, self._Q_B[module], self._Q_A[module])
            else:
                h_kfe = h_bar @ self._Q_A[module]
                s_kfe = s_flat @ self._Q_B[module]
                # Intra-batch estimate of s* (Algorithm 1's COMPUTE SCALINGS step): the per-example
                # projected gradient in the KFE is (s_kfe_n (x) h_kfe_n); its squared entries,
                # averaged over the batch, are exactly (s_kfe_n_i)^2 * (h_kfe_n_j)^2 averaged over n.
                g2 = (s_kfe.t() ** 2) @ (h_kfe**2) / h_bar.size(0)
            if self.ema_seed_first and module not in self._s_star_observed:
                # refresh() had to put *something* in s*, because precondition() reads it on the
                # very step the eigenbasis is created, before any gradient has been projected into
                # that basis. Under ema_seed_first the first projected gradient replaces it
                # outright rather than being averaged into it, so the identity is carried for
                # exactly one step and leaves no residue behind.
                self._s_star[module] = g2.detach().clone()
                self._s_star_observed.add(module)
            else:
                update_running_avg(g2, self._s_star[module], self.gammas)

    def _rebuild_eigenbasis(self, module: Module) -> None:
        """Eigenvectors of the current ``A`` and ``B``, and, the first time round, a seed for
        ``s*``. Split out of ``refresh`` because ``eig_before_rescale`` calls it from the backward
        hook instead.
        """
        Q_A = eigenbasis(self._A[module])
        Q_B = eigenbasis(self._B[module])
        if module not in self._Q_A:
            # No s* estimate can exist before an eigenbasis does, so seed it with ones -- the
            # direct generalisation of diag's own all-ones seed, and an inert preconditioner up to
            # the damping. Under ema_seed_first this value survives exactly one step; see
            # update_output_factor.
            self._s_star[module] = Q_B.new_ones(Q_B.size(0), Q_A.size(0))
        self._Q_A[module] = Q_A
        self._Q_B[module] = Q_B

    def refresh(self, module: Module, step: int) -> None:
        if self._eig_step.get(module) == step:
            return  # already rebuilt in this step's backward hook (eig_before_rescale)
        # "or no eigenbasis yet" covers a module first reached after step 0, whose first step is
        # not necessarily a multiple of T_eig; precondition() would otherwise read a basis that was
        # never built. It cannot fire on a module present from step 0, since step 0 is a multiple
        # of every T_eig, so no existing trajectory changes.
        if step % self.T_eig != 0 and module in self._Q_A:
            return
        self._rebuild_eigenbasis(module)

    def mean_curvature(self, module: Module) -> Optional[Tensor]:
        """``mean(s*)``: ``s*`` *is* the spectrum. Before the first ``refresh`` there is no ``s*``
        yet, but ``refresh`` is about to seed it with ones, so its mean is 1."""
        if module in self._s_star:
            return self._s_star[module].mean()
        if module in self._A and module in self._B:
            return self._A[module].new_ones(())
        return None

    def num_directions(self, module: Module) -> Optional[int]:
        if module not in self._A or module not in self._B:
            return None
        return self._A[module].size(0) * self._B[module].size(0)

    @property
    def consumes_bias_corrected_momentum(self) -> bool:
        """True under ``rescale_form="clip"``: the optimizer then hands ``precondition`` the
        bias-corrected momentum and applies the result as it is. See
        :mod:`adafisher_modes.approximations._rescale_utils`."""
        return self._clip is not None

    @property
    def _clip_stats(self) -> Dict[Module, Dict[str, Tensor]]:
        """Per-module diagnostics of the clip at its last step, computed on access (empty for the
        other forms)."""
        return {} if self._clip is None else self._clip.stats

    def begin_step(self, step: int) -> None:
        if self._clip is not None:
            self._clip.begin_step(step)

    def _rescale(self, M_kfe: Tensor, module: Module) -> Tensor:
        """The division by ``s*`` inside the eigenbasis, in the form ``rescale_form`` selects."""
        if self._clip is not None:
            return self._clip.apply(module, M_kfe, self._s_star[module])
        return rescale(M_kfe, self._s_star[module], self.lambda_for(module), self.rescale_form)

    def f_tilde(self, module: Module) -> Tensor:
        """Dense ``(d_out * d_in_aug)^2`` reconstruction of
        ``F~ = kron(Q_B, Q_A) diag(s* + lambda) kron(Q_B, Q_A)^T`` (``diag(max(s*, lambda))`` under
        ``rescale_form="floor"``): output factor outer, input
        factor inner, matching ``s*``'s own row-major flattening. Built from the current eigenbasis
        and ``s*``. For tests and debugging only -- ``precondition`` never forms this matrix.
        """
        Q_A, Q_B = self._Q_A[module], self._Q_B[module]
        if self.rescale_form == "clip":
            raise NotImplementedError(
                "rescale_form='clip' is not a linear operator: the coordinates whose undamped "
                "step exceeds the threshold are clipped, so there is no matrix to reconstruct"
            )
        if self.rescale_form == "floor":
            scale = floored(self._s_star[module], self.lambda_for(module)).flatten()
        else:
            scale = (self._s_star[module] + self.lambda_for(module)).flatten()
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
            M_kfe = self._rescale(Q_B.t() @ M @ Q_A, module)
            direction = Q_B @ M_kfe @ Q_A.t()
            return split_conv2d_direction_sua(direction, weight_direction.shape, bias_shape)
        M = augment_direction(weight_direction, bias_direction)
        M_kfe = self._rescale(Q_B.t() @ M @ Q_A, module)
        direction = Q_B @ M_kfe @ Q_A.t()
        return split_direction(direction, weight_direction.shape, bias_shape)
