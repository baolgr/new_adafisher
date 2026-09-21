"""``tekfac`` mode: Gao et al.'s trace-restricted, eigenvalue-corrected Kronecker factorization.

TEKFAC is TKFAC's pair of factors with EKFAC's eigenbasis correction on top
(``tekfac_2011.13609.pdf`` section 3.1). It eigendecomposes the *same* trace-restricted
``(delta, Phi, Psi)`` triple ``tkfac`` maintains, not the undivided K-FAC factors::

    F_l ~= delta * Phi (x) Psi
         = (Q_Phi (x) Q_Psi) (delta * Lambda_Phi (x) Lambda_Psi) (Q_Phi (x) Q_Psi)^T   (eq. 3.1)

and then replaces that still-inexact rescaling by the directly estimated optimal diagonal, exactly
EKFAC's Lemma 1 argument applied to a different orthogonal basis (eq. 3.2-3.3)::

    Theta_ii = E[((Q_Phi (x) Q_Psi)^T grad)_i^2]
    F_l      ~= (Q_Phi (x) Q_Psi) Theta (Q_Phi (x) Q_Psi)^T

There is no ``delta`` multiplying ``Theta``: ``Theta`` is estimated from the raw gradient and is
already on the right scale. Theorem 3.1 then gives ``||F - TEKFAC||_F <= ||F - TKFAC||_F``.

The eigenvectors of ``Phi`` are the eigenvectors of ``Phi * delta``, since dividing a symmetric
matrix by a positive scalar does not move them, so the decomposition can skip the division and run
straight on the stored numerators.

**Damping** (eq. 3.4) is a plain additive ``lambda`` on ``Theta``, as in ``ekfac``. The paper's
convolution-only trace-adaptive ``lambda`` and its network-wide rescaling of dense layers (eq. 3.5)
are not the default. ``AdaFisherMulti(damping="layer_relative", damping_tau=1)`` gives every layer
``lambda = mean(Theta) = tr(Theta)/dim(Theta)``, which is eq. (3.5) without its floor ``vartheta``
and without the dense-layer rescaling.

**Cadence.** Algorithm 1 of the paper gives the eigendecomposition and the rescaling separate
intervals, and so does this class: ``T_eig`` gates the eigenbasis refresh and ``T_re`` gates
``Theta``'s own running-average update. There is no separate interval for the ``(delta, Phi, Psi)``
triple; it is updated on every hook fire, like ``A`` and ``B`` in ``kfac`` and ``ekfac``.

**Which basis ``Theta`` is measured in (``eig_before_rescale``, off by default).** Algorithm 1 of
the paper orders one iteration eigenbasis first, rescaling second. By default this class does the
opposite, because the hooks fire before ``step()``: ``Theta`` is folded in against the *current*
basis, ``refresh`` then replaces that basis, and ``precondition`` divides using the new basis and
the old ``Theta``. Eq. (3.2)'s optimality argument -- EKFAC's Lemma 1 in TKFAC's basis -- holds for
the basis ``Theta`` was measured in and for no other. ``eig_before_rescale=True`` rebuilds the
basis inside the backward hook, before the projection, so the two agree. It is off by default
because turning it on changes every trajectory; see
:mod:`adafisher_modes.approximations.ekfac` for the same knob and the measured size of the
difference.

**Two smoothing rates, and a name clash to keep straight.** ``beta_factors`` smooths
``(delta, Phi, Psi)`` and ``beta_theta`` smooths ``Theta``; both default to the shared ``gammas``.
These are the paper's own ``beta_1`` and ``beta_2`` of eq. 3.6-3.8, renamed here because
Adam and AdaFisher already use ``beta_1`` for the momentum of the first moment, which is an
unrelated quantity.

**SUA** (``conv_sua=True``, ``Conv2d`` only) makes the input factor channel-only instead of
patch-based (``kfac_conv_1602.01407.pdf`` p. 14), and ``precondition`` then applies the small
operator independently at each kernel offset.

**How the division is done (``rescale_form``, ``"add"`` by default).** ``"add"`` divides by
``Theta + lambda`` as above. ``"floor"`` divides by ``max(Theta, lambda)``, and ``"clip"`` replaces the
safety constant by a Sophia-type per-coordinate cap on the undamped step, with a threshold that is
recomputed every step (``"quantile"``), follows the momentum's recent size (``"ema"``), or is one
constant for the whole network (``"fixed"``). Both alternatives live in
:mod:`adafisher_modes.approximations._rescale_utils`; ``docs/reports/plan_floor_clip.md`` is the
experiment they exist for.

``f_tilde`` builds the dense operator and exists only for tests; ``precondition`` never forms it.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from torch import Tensor, diag, kron
from torch.nn import BatchNorm2d, Conv2d, LayerNorm, Module

from adafisher_modes.ema import seed_or_accumulate, update_running_avg
from adafisher_modes.factors import (
    augment_input,
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
from ._tkfac_utils import bootstrap_raw_factors, instantaneous_raw_factors
from .base import FisherApproximation, pop_cached_input


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
        self.beta_factors = tuple(beta_factors) if beta_factors is not None else tuple(gammas)
        self.beta_theta = tuple(beta_theta) if beta_theta is not None else tuple(gammas)
        self.T_eig = T_eig
        self.T_re = T_re
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
        self._theta_observed: set = set()
        self._eig_step: Dict[Module, int] = {}
        self._delta: Dict[Module, Tensor] = {}
        self._Phi_raw: Dict[Module, Tensor] = {}
        self._Psi_raw: Dict[Module, Tensor] = {}
        self._Q_Phi: Dict[Module, Tensor] = {}
        self._Q_Psi: Dict[Module, Tensor] = {}
        self._Theta: Dict[Module, Tensor] = {}
        self._cached_h_bar: Dict[Module, Tensor] = {}

    def update_input_factor(self, module: Module, h: Tensor, step: int) -> None:
        self._cached_h_bar[module] = augment_input(h, module, sua=self.conv_sua)
        if self.norm_exact_rescaling and isinstance(module, (BatchNorm2d, LayerNorm)):
            self._cached_x_hat[module] = normalized_norm_input(h, module)

    def update_output_factor(self, module: Module, s: Tensor, step: int) -> None:
        h_bar = pop_cached_input(self._cached_h_bar, module)
        x_hat = self._cached_x_hat.pop(module, None)
        s_flat = flatten_output_grad(s, module)
        delta_i, phi_raw_i, psi_raw_i = instantaneous_raw_factors(h_bar, s_flat)
        # Same start-up condition as seed_or_accumulate's: step 0, or the first time this module is
        # seen at all, which can be any step if it sits behind a conditional branch.
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
                               self.beta_factors, step, self.ema_seed_first)

        if self.eig_before_rescale and step % self.T_eig == 0:
            # Rebuild the eigenbasis from Phi_raw and Psi_raw as they stand for *this* step, before
            # the gradient is projected into it, so that Theta is measured in the basis
            # precondition() will then use -- the order Algorithm 1 of the paper gives. refresh()
            # sees _eig_step and does not redo it. Off by default; see the class docstring.
            self._rebuild_eigenbasis(module)
            self._eig_step[module] = step
        if module in self._Q_Phi and step % self.T_re == 0:
            if x_hat is not None:
                # norm_exact_rescaling: the normalisation layer's exact per-row gradient, as in
                # ekfac, in TKFAC's basis.
                theta_i = norm_exact_kfe_squares(x_hat, s_flat, self._Q_Psi[module],
                                                 self._Q_Phi[module])
            else:
                h_kfe = h_bar @ self._Q_Phi[module]
                s_kfe = s_flat @ self._Q_Psi[module]
                # Intra-batch estimate of Theta (eq. 3.2), the same construction as ekfac's s*
                # estimator with (Q_A, Q_B) replaced by (Q_Phi, Q_Psi).
                theta_i = (s_kfe.t() ** 2) @ (h_kfe**2) / h_bar.size(0)
            if self.ema_seed_first and module not in self._theta_observed:
                # Same one-step handover as ekfac's s*: refresh() must leave a value behind for
                # precondition() to read on the step the eigenbasis is created, and the first
                # projected gradient then replaces it outright instead of averaging into it.
                self._Theta[module] = theta_i.detach().clone()
                self._theta_observed.add(module)
            else:
                update_running_avg(theta_i, self._Theta[module], self.beta_theta)

    def _rebuild_eigenbasis(self, module: Module) -> None:
        """Eigenvectors of the current ``Phi_raw`` and ``Psi_raw``, and, the first time round, a
        seed for ``Theta``. Split out of ``refresh`` because ``eig_before_rescale`` calls it from
        the backward hook instead.
        """
        # Dividing a symmetric matrix by a positive scalar leaves its eigenvectors unchanged, so
        # the eigenvectors of Phi_raw are those of Phi = Phi_raw/delta and the decomposition can run
        # straight on the stored numerators.
        Q_Phi = eigenbasis(self._Phi_raw[module])
        Q_Psi = eigenbasis(self._Psi_raw[module])
        if module not in self._Q_Phi:
            # No Theta estimate can exist before an eigenbasis does, so seed it with ones, exactly
            # as ekfac seeds s*. Under ema_seed_first this value survives exactly one step; see
            # update_output_factor.
            self._Theta[module] = Q_Psi.new_ones(Q_Psi.size(0), Q_Phi.size(0))
        self._Q_Phi[module] = Q_Phi
        self._Q_Psi[module] = Q_Psi

    def refresh(self, module: Module, step: int) -> None:
        if self._eig_step.get(module) == step:
            return  # already rebuilt in this step's backward hook (eig_before_rescale)
        # "or no eigenbasis yet" covers a module first reached after step 0, whose first step is
        # not necessarily a multiple of T_eig; precondition() would otherwise read a basis that was
        # never built. It cannot fire on a module present from step 0, since step 0 is a multiple
        # of every T_eig, so no existing trajectory changes.
        if step % self.T_eig != 0 and module in self._Q_Phi:
            return
        self._rebuild_eigenbasis(module)

    def mean_curvature(self, module: Module) -> Optional[Tensor]:
        """``mean(Theta)``, which is ``tr(Theta) / dim(Theta)`` -- the quantity TEKFAC's eq. (3.5)
        sets its convolution-layer damping to. Before the first ``refresh``, ``Theta`` is about to
        be seeded with ones, so its mean is 1."""
        if module in self._Theta:
            return self._Theta[module].mean()
        if module in self._delta:
            return self._delta[module].new_ones(())
        return None

    def num_directions(self, module: Module) -> Optional[int]:
        if module not in self._delta:
            return None
        return self._Phi_raw[module].size(0) * self._Psi_raw[module].size(0)

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
        """The division by ``Theta`` inside the eigenbasis, in the form ``rescale_form`` selects."""
        if self._clip is not None:
            return self._clip.apply(module, M_kfe, self._Theta[module])
        return rescale(M_kfe, self._Theta[module], self.lambda_for(module), self.rescale_form)

    def f_tilde(self, module: Module) -> Tensor:
        """Dense ``(d_out * d_in_aug)^2`` reconstruction of
        ``F~ = kron(Q_Psi, Q_Phi) diag(Theta + lambda) kron(Q_Psi, Q_Phi)^T``
        (``diag(max(Theta, lambda))`` under ``rescale_form="floor"``): output factor outer,
        input factor inner, matching ``Theta``'s own row-major flattening. Built from the current
        eigenbasis and ``Theta``. For tests and debugging only -- ``precondition`` never forms this
        matrix.
        """
        Q_Phi, Q_Psi = self._Q_Phi[module], self._Q_Psi[module]
        if self.rescale_form == "clip":
            raise NotImplementedError(
                "rescale_form='clip' is not a linear operator: the coordinates whose undamped "
                "step exceeds the threshold are clipped, so there is no matrix to reconstruct"
            )
        if self.rescale_form == "floor":
            scale = floored(self._Theta[module], self.lambda_for(module)).flatten()
        else:
            scale = (self._Theta[module] + self.lambda_for(module)).flatten()
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
            M_kfe = self._rescale(Q_Psi.t() @ M @ Q_Phi, module)
            direction = Q_Psi @ M_kfe @ Q_Phi.t()
            return split_conv2d_direction_sua(direction, weight_direction.shape, bias_shape)
        M = augment_direction(weight_direction, bias_direction)
        M_kfe = self._rescale(Q_Psi.t() @ M @ Q_Phi, module)
        direction = Q_Psi @ M_kfe @ Q_Phi.t()
        return split_direction(direction, weight_direction.shape, bias_shape)
