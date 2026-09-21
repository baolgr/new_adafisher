"""The division inside the eigenbasis, shared by ``ekfac`` and ``tekfac``.

Both modes project the direction into an eigenbasis, divide each coordinate by a stored curvature
value (``s*`` for ``ekfac``, ``Theta`` for ``tekfac``), and project back. This module holds that
middle step, in three forms selected by ``AdaFisherMulti(rescale_form=...)``:

``"add"`` (the default, and what both papers prescribe)
    ``M / (s + lambda)``. The safety constant is *added* to every curvature value, so it moves the
    well-estimated values as well as the tiny ones.

``"floor"`` (family A of ``docs/reports/fr/etude_clipping_vs_damping.md``)
    ``M / max(s, lambda)``. A curvature value above ``lambda`` is used as it is; one below it is
    replaced by ``lambda``. Where ``lambda`` dominates every value -- the shipped operating point,
    ``plan_lambda_dominance.md`` E1 -- this is the same as ``"add"`` to within ``s / lambda``. The
    two differ only for the values that sit near ``lambda``, and by at most a factor of 2 there.

``"clip"`` (family B, Sophia-type: Liu et al., arXiv:2305.14342)
    Divide by the undamped curvature, then cap every coordinate at a threshold ``gamma``. With
    ``r = |M| / s`` the size of each coordinate's undamped step, and ``guard = clip_guard * rms(M)``:

    * an *active* coordinate (``|M| > guard``) moves by ``u = sign(M) * min(r / gamma, 1)``: it is
      *clipped* (``|u| = 1``) when ``r >= gamma``, and otherwise takes the undamped step divided by
      ``gamma``;
    * an *inactive* coordinate (``|M| <= guard``) moves by ``u = M / max(gamma * s, guard)``, i.e. by
      at most ``|M| / guard <= 1``.

    For active coordinates this is ``clip(M / (gamma s), 1) = M / max(gamma s, |M|)``: a floor on the
    curvature proportional to the momentum itself (``|M| / gamma``), not a constant. ``lambda``
    plays no role. Clipped is *defined* by ``active and r >= gamma``, with ``r`` the same tensor
    ``gamma`` is selected from, so the count is exact by construction (ties aside).

    **Why the guard, and why the fraction counts only active coordinates.** In the eigenbasis many
    coordinates are zero in exact arithmetic -- the gradient lies in the span of the batch's inputs,
    ``A`` has few observed directions, and an empirical Fisher has exact null directions such as the
    head's logit-shift kernel (``plan_exp_lot1.md`` §6.4) -- and they are rounding noise in fp32.
    Measured on a 6-example batch: ``|M|`` down to ``1e-13``, with the median ``r`` set entirely by
    such coordinates. Counted in the fraction they would put ``gamma`` inside the noise and clip
    every real coordinate; left unguarded, each would take a full step driven by rounding. The guard
    level ``clip_guard = 1e-3`` is a choice, not a measured gap: on real layers genuine coordinates
    reach ``5e-6 x rms`` and exact-null rows ``9.8e-4 x rms`` (E16 audit), so every verdict on the
    clip is a verdict at ``clip_guard = 1e-3``.

    The threshold is where the three variants differ (``clip_threshold``). All three are *per
    module*: the stored curvature's scale error differs between layers by up to four orders of
    magnitude -- a layer applied at ``T`` positions stores ``s`` averaged over ``N*T`` rows
    (``plan_exp_lot3.md`` §5.2's ``1/T``), and a normalisation layer's input factor is built from the
    raw channel mean (``CLAUDE.md`` §4.6) -- so one ``gamma`` for the whole network would not be one
    threshold in consistent units (measured in the E16 audit: at a pooled calibration, a ViT's head
    was clipped on 3 % of its coordinates and its final LayerNorm on 97 %).

    ``"quantile"``
        ``gamma_q`` = the ``c``-th largest ``r`` among the module's ``n_a`` active coordinates at
        *this* step, ``c = max(floor(clip_fraction * n_a), 1)``: exactly ``c`` of them are clipped
        at every step. ``u`` is unchanged when ``M`` or ``s`` is multiplied by any positive
        constant, so no scale error of the stored curvature reaches the step -- but neither does
        the momentum's own size. Every step of every module is a full-size step: this is a
        per-module *normalisation* with a clip of its shape, not a clip that acts only beyond a
        ceiling. Kept as the control that isolates the normalisation.

    ``"ema"``
        The same quantile, rescaled by how large the momentum is compared with its own recent
        history::

            gamma = gamma_q * mu_bar / mu,   mu = rms(M),
            log mu_bar = bias-corrected running average of log mu, horizon H = clip_ema_horizon

        The average is Adam's: ``num <- (1-w) num + w log mu``, ``den <- (1-w) den + w``,
        ``log mu_bar = num / den`` with ``w = 1/H``, so no single early step dominates it (the first
        bias-corrected momentum is one raw gradient, measured 4.4-5x larger in rms than the
        momentum settles to; an average seeded with it stayed biased by >10 % for 12-18 % of a
        run). A step with ``mu = 0`` carries no size and is skipped. When ``mu = mu_bar`` this is
        ``"quantile"`` exactly; when the momentum is ``a`` times smaller than over the last ``~H``
        steps, the unclipped coordinates are ``a`` times smaller and fewer are clipped; a spike is
        clipped harder. The curvature's scale is still absorbed at every step: the average is on
        ``mu``, not on ``gamma``, because ``gamma`` also carries ``s``'s scale, which jumps at every
        factor update (``s`` is 92 % one minibatch). ``H = 1000`` by default, the memory of Adam's
        bias-corrected second moment at ``beta_2 = 0.999``.

    ``"fixed"``
        One ``gamma`` per module, held constant: a conditional clip, as Sophia's is -- it acts only
        where the undamped step exceeds the ceiling, and below it the step is proportional to the
        momentum. ``gamma`` is calibrated, not hand-set: until step ``clip_calibrate_at`` the rule
        is ``"quantile"``; the module's ``gamma_q`` of the last ``clip_calibration_window`` steps
        before it are kept, and at the first step ``>= clip_calibrate_at`` their (lower) median is
        frozen. A median over a window, because ``s`` is 92 % one minibatch and one step's quantile
        is one draw (measured: ``x3.6`` between factor updates). Per module, because of the scale
        error above; it also makes the switch continuous (the frozen value is what the module had
        just been using). A module first reached after ``clip_calibrate_at`` freezes its first
        ``gamma_q``. Sophia itself uses one ``rho`` for the whole network, which is meaningful there
        because its curvature estimate has the correct scale in every layer; here it does not.

    **The optimizer hands the clip the bias-corrected momentum** ``m / (1 - beta^t)`` and applies
    the result without further correction. For ``"quantile"`` this changes nothing (``u`` does not
    depend on ``M``'s scale). For ``"ema"`` and ``"fixed"`` it is the Adam convention for comparing
    a momentum with a threshold.

    **Diagnostics** (``ClipRule.stats``) are computed only when read, from references to the last
    step's tensors: they cost nothing on the steps where nobody reads them, and they describe the
    last step until the next backward pass updates the stored curvature in place.

    **The threshold state is not in the optimizer's** ``state_dict``: ``"ema"``'s running average
    and ``"fixed"``'s window and frozen values live in :class:`ClipRule`, like every mode's factors.
    A run that is checkpointed and resumed restarts them; E16 never resumes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

RESCALE_FORMS = ("add", "floor", "clip")
CLIP_THRESHOLDS = ("quantile", "ema", "fixed")


def check_rescale_args(
    form: str,
    clip_fraction: Optional[float],
    clip_guard: float,
    clip_threshold: str = "quantile",
    clip_ema_horizon: Optional[int] = None,
    clip_calibrate_at: Optional[int] = None,
    clip_calibration_window: Optional[int] = None,
) -> None:
    """Refuse an inconsistent combination at construction time, not on the first step."""
    if form not in RESCALE_FORMS:
        raise ValueError(f"rescale_form must be one of {RESCALE_FORMS}; got {form!r}")
    clip_only = {"clip_fraction": clip_fraction, "clip_ema_horizon": clip_ema_horizon,
                 "clip_calibrate_at": clip_calibrate_at,
                 "clip_calibration_window": clip_calibration_window}
    if form != "clip":
        given = [k for k, v in clip_only.items() if v is not None]
        if given or clip_threshold != "quantile":
            raise ValueError(
                f"{', '.join(given) or 'clip_threshold'} only applies to rescale_form='clip'; "
                f"rescale_form is {form!r}"
            )
        return
    if clip_threshold not in CLIP_THRESHOLDS:
        raise ValueError(f"clip_threshold must be one of {CLIP_THRESHOLDS}; got {clip_threshold!r}")
    if not clip_guard > 0.0:
        raise ValueError(f"clip_guard must be > 0; got {clip_guard!r}")
    if clip_fraction is None or not 0.0 < clip_fraction < 1.0:
        raise ValueError(
            f"rescale_form='clip' needs 0 < clip_fraction < 1 (the fraction of each module's active "
            f"coordinates that is clipped); got {clip_fraction!r}"
        )

    def refuse(**kwargs) -> None:
        given = [k for k, v in kwargs.items() if v is not None]
        if given:
            raise ValueError(
                f"{', '.join(given)} does not apply to clip_threshold={clip_threshold!r}"
            )

    if clip_threshold == "quantile":
        refuse(clip_ema_horizon=clip_ema_horizon, clip_calibrate_at=clip_calibrate_at,
               clip_calibration_window=clip_calibration_window)
    elif clip_threshold == "ema":
        refuse(clip_calibrate_at=clip_calibrate_at,
               clip_calibration_window=clip_calibration_window)
        if clip_ema_horizon is not None and not clip_ema_horizon >= 1:
            raise ValueError(f"clip_ema_horizon must be >= 1 step; got {clip_ema_horizon!r}")
    else:  # fixed
        refuse(clip_ema_horizon=clip_ema_horizon)
        if clip_calibrate_at is None or clip_calibrate_at < 0:
            raise ValueError(
                "clip_threshold='fixed' needs clip_calibrate_at >= 0, the step at which each "
                "module's threshold is frozen"
            )
        if clip_calibration_window is not None and not clip_calibration_window >= 1:
            raise ValueError(
                f"clip_calibration_window must be >= 1 step; got {clip_calibration_window!r}"
            )


def _count_dtype(device: torch.device) -> torch.dtype:
    """fp64 where the device has it, so that ``floor(q * n_a)`` is computed exactly as Python would
    (``0.7 * 10`` is ``7`` in fp64 but ``6.9999995`` in fp32). MPS has no fp64."""
    return torch.float32 if device.type == "mps" else torch.float64


def active_quantile(r: Tensor, active: Tensor, clip_fraction: float) -> Tensor:
    """The ``c``-th largest ``r`` among the active coordinates, ``c = max(floor(q * n_a), 1)``, as a
    0-dim tensor on ``r``'s device. Computed by a sort and a gather, never by indexing with a
    boolean mask, so the count ``n_a`` never has to reach the host: no synchronisation per module
    per step. With no active coordinate (``M = 0``) any value will do, since ``u = 0`` then; the
    largest finite float is returned."""
    finfo = torch.finfo(r.dtype)
    flat_r = r.flatten()
    flat_active = active.flatten()
    masked = torch.where(flat_active, flat_r, torch.full_like(flat_r, float("-inf")))
    ordered = torch.sort(masked, descending=True).values
    n_active = flat_active.sum()
    c = torch.floor(n_active.to(_count_dtype(r.device)) * clip_fraction).to(torch.long)
    c = torch.clamp(torch.minimum(c, n_active), min=1)
    gamma = ordered.gather(0, (c - 1).reshape(1)).reshape(())
    # An r that overflowed to inf (zero curvature under active momentum) can be the c-th largest;
    # the largest finite float keeps gamma * s finite and sends every finite-r coordinate to zero.
    gamma = torch.nan_to_num(gamma, posinf=finfo.max, neginf=finfo.max)
    return torch.where(n_active > 0, gamma, torch.full_like(gamma, finfo.max))


def floored(s: Tensor, lam: Union[float, Tensor]) -> Tensor:
    """``max(s, lambda)``. A Python-float ``lambda`` goes through ``clamp_min``, which passes it to
    the kernel as a scalar; building a device tensor for it would cost a host-to-device copy per
    module per step. Measured bit-identical to ``torch.maximum(s, as_tensor(lambda))``."""
    if isinstance(lam, Tensor):
        return torch.maximum(s, lam.to(dtype=s.dtype, device=s.device))
    return s.clamp_min(lam)


def rescale(M_kfe: Tensor, s: Tensor, lam: Union[float, Tensor], form: str) -> Tensor:
    """The two stateless forms, ``"add"`` and ``"floor"``. ``"clip"`` goes through
    :class:`ClipRule`."""
    if form == "add":
        # Kept as the exact expression both modes used before this module existed, so the default
        # path is bit-identical to the pre-rescale_form code.
        return M_kfe / (s + lam)
    if form == "floor":
        return M_kfe / floored(s, lam)
    raise ValueError(
        f"rescale() handles 'add' and 'floor'; 'clip' goes through ClipRule; got {form!r}")


class _Record:
    """References to one module's last clip step: enough to compute every diagnostic on demand."""

    __slots__ = ("abs_m", "s", "r", "active", "gamma", "guard", "rms", "u", "extra")

    def __init__(self, abs_m: Tensor, s: Tensor, r: Tensor, active: Tensor, gamma: Tensor,
                 guard: Tensor, rms: Tensor, u: Tensor, extra: Dict[str, Tensor]) -> None:
        self.abs_m, self.s, self.r, self.active = abs_m, s, r, active
        self.gamma, self.guard, self.rms, self.u, self.extra = gamma, guard, rms, u, extra

    def stats(self) -> Dict[str, Tensor]:
        dtype = self.u.dtype
        active = self.active
        clipped = active & (self.r >= self.gamma)
        n_active = active.sum().clamp_min(1).to(dtype)
        guard_binds = ~active & (self.gamma * self.s <= self.guard)
        u2 = self.u.pow(2)
        small = self.abs_m < 1e-2 * self.rms
        out = {
            "gamma": self.gamma.detach(),
            # clip_fraction refers to the active coordinates, so this is counted among them.
            "clipped_fraction": clipped.sum().to(dtype) / n_active,
            "active_fraction": active.to(dtype).mean(),
            "guard_fraction": guard_binds.to(dtype).mean(),
            # How much of the step is carried by coordinates just above the guard (|M| < 1e-2 rms):
            # the part of a clip verdict that depends on the choice clip_guard = 1e-3.
            "u2_share_below_1e-2_rms": (u2 * small).sum() / u2.sum().clamp_min(
                torch.finfo(dtype).tiny),
        }
        if self.u.dim() == 2 and self.u.size(-1) == 2:
            # A normalisation layer's two input-eigen columns (scale and shift, mixed by the 2x2
            # Q_A): where the clipped slots go, column by column.
            for j in (0, 1):
                out[f"clipped_fraction_col{j}"] = (
                    clipped[:, j].sum().to(dtype) / active[:, j].sum().clamp_min(1).to(dtype))
        out.update({k: v.detach() for k, v in self.extra.items()})
        return out


class ClipRule:
    """The ``"clip"`` form with its per-module threshold state. One instance per approximation
    object. The optimizer calls :meth:`begin_step` before walking the modules and :meth:`apply`
    exactly once per module per step: ``"ema"`` advances its running average and ``"fixed"`` its
    calibration window on every call."""

    def __init__(
        self,
        threshold: str,
        clip_fraction: float,
        clip_guard: float,
        ema_horizon: Optional[int] = None,
        calibrate_at: Optional[int] = None,
        calibration_window: Optional[int] = None,
    ) -> None:
        self.threshold = threshold
        self.clip_fraction = clip_fraction
        self.clip_guard = clip_guard
        self.ema_horizon = 1000 if ema_horizon is None else ema_horizon
        self.calibrate_at = calibrate_at
        self.calibration_window = 1000 if calibration_window is None else calibration_window
        self.step: Optional[int] = None
        # "ema": Adam-style bias-corrected running average of log rms(M), as (numerator, weight).
        self.log_mu_num: Dict[Module, Tensor] = {}
        self.log_mu_den: Dict[Module, Tensor] = {}
        # "fixed": each module's per-step quantiles inside the window, then its frozen threshold.
        self.window: Dict[Module, List[Tensor]] = {}
        self.gamma_fixed: Dict[Module, Tensor] = {}
        self._records: Dict[Module, _Record] = {}

    def begin_step(self, step: int) -> None:
        self.step = step

    @property
    def stats(self) -> Dict[Module, Dict[str, Tensor]]:
        """Per-module diagnostics of the last step, computed now from references to its tensors."""
        return {module: record.stats() for module, record in self._records.items()}

    def log_mu_bar(self, module: Module) -> Optional[Tensor]:
        """The bias-corrected running average of ``log rms(M)`` ("ema"), or ``None`` before the
        module's first nonzero momentum."""
        den = self.log_mu_den.get(module)
        if den is None or not bool(den > 0):
            return None
        return self.log_mu_num[module] / den

    def apply(self, module: Module, M_kfe: Tensor, s: Tensor) -> Tensor:
        tiny = torch.finfo(s.dtype).tiny
        abs_m = M_kfe.abs()
        rms = abs_m.pow(2).mean().sqrt()
        guard = self.clip_guard * rms
        active = abs_m > guard
        r = abs_m / s.clamp_min(tiny)
        extra: Dict[str, Tensor] = {}

        if self.threshold == "fixed" and module in self.gamma_fixed:
            gamma = self.gamma_fixed[module]
            extra["calibrated"] = torch.ones((), dtype=s.dtype, device=s.device)
        else:
            gamma = active_quantile(r, active, self.clip_fraction)
            if self.threshold == "fixed":
                gamma = self._calibrate(module, gamma)
                extra["calibrated"] = torch.full(
                    (), float(module in self.gamma_fixed), dtype=s.dtype, device=s.device)
            elif self.threshold == "ema":
                gamma, ratio = self._ema(module, gamma, rms, tiny)
                extra["momentum_ratio"] = 1.0 / ratio

        ratio_to_gamma = r / gamma.clamp_min(tiny)
        u_active = torch.sign(M_kfe) * ratio_to_gamma.clamp(max=1.0)
        u_inactive = M_kfe / torch.maximum(gamma * s, guard).clamp_min(tiny)
        u = torch.where(active, u_active, u_inactive)
        self._records[module] = _Record(abs_m, s, r, active, gamma, guard, rms, u, extra)
        return u

    def _ema(self, module: Module, gamma_q: Tensor, rms: Tensor,
             tiny: float) -> Tuple[Tensor, Tensor]:
        """``gamma_q * mu_bar / mu``, with ``log mu_bar`` Adam's bias-corrected running average of
        ``log mu``. Returns ``(gamma, mu_bar / mu)``."""
        if module not in self.log_mu_num:
            self.log_mu_num[module] = torch.zeros((), dtype=rms.dtype, device=rms.device)
            self.log_mu_den[module] = torch.zeros((), dtype=rms.dtype, device=rms.device)
        w = 1.0 / self.ema_horizon
        log_mu = rms.clamp_min(tiny).log().detach()
        nonzero = rms > 0
        num = torch.where(nonzero, self.log_mu_num[module] * (1.0 - w) + log_mu * w,
                          self.log_mu_num[module])
        den = torch.where(nonzero, self.log_mu_den[module] * (1.0 - w) + w,
                          self.log_mu_den[module])
        self.log_mu_num[module], self.log_mu_den[module] = num, den
        log_mu_bar = torch.where(den > 0, num / den.clamp_min(tiny), log_mu)
        ratio = torch.exp(log_mu_bar - log_mu)
        return gamma_q * ratio, ratio

    def _calibrate(self, module: Module, gamma_q: Tensor) -> Tensor:
        """Collect ``gamma_q`` inside the window before ``calibrate_at``; at the first step at or
        after it, freeze the lower median of the window (or ``gamma_q`` itself if the module was
        never seen inside it) and use it from this step on. Outside an optimizer, where no step is
        known, the rule stays ``"quantile"`` and nothing is collected."""
        assert self.calibrate_at is not None  # checked at construction
        if self.step is None:
            return gamma_q
        if self.step < self.calibrate_at:
            if self.step >= self.calibrate_at - self.calibration_window:
                self.window.setdefault(module, []).append(gamma_q.detach())
            return gamma_q
        samples = self.window.pop(module, [])
        frozen = torch.stack(samples).median() if samples else gamma_q.detach()
        self.gamma_fixed[module] = frozen
        return frozen
