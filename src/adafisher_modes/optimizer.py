"""``AdaFisherMulti``: AdaFisher with a selectable Fisher approximation mode.

This is a port of ``AdaFisherBackbone`` / ``AdaFisher`` from
``reference_repos/FisherAdapTune/scripts/adafisher.py``, with the construction of the second moment
moved behind the :class:`~adafisher_modes.approximations.base.FisherApproximation` interface so that
it can be swapped for K-FAC, EKFAC, TKFAC or TEKFAC. The optimizer itself is Adam's update with the
square root removed: momentum on the gradient, Adam's bias correction, and a division by a
Fisher estimate instead of by a running average of squared gradients (AdaFisher Table 1,
``adafisher_2405.16397.pdf``).

**What happens in one step.** Forward and backward hooks on every supported layer hand the raw input
and the raw output gradient to the active mode, but only on steps that are multiples of ``TCov``
(100 by default), so the curvature comes from one minibatch in every hundred. ``step()`` then walks
this optimizer's parameters, and for each hooked module calls ``refresh`` once (to redo any
amortised inverse or eigendecomposition) and ``precondition`` once, with the weight and bias
directions together. A parameter belonging to no hooked module -- a ViT's class token, a positional
embedding -- falls through to a plain momentum step, which is what dividing by an identity Fisher
would give.

Supported layer types are ``Linear``, ``Conv2d``, ``BatchNorm2d`` and ``LayerNorm``. A normalisation
layer with ``affine=False`` has no weight to precondition and is skipped.

**Three differences from the reference that are on by default.**

1. *The update is unfused.* ``precondition`` returns a direction and the optimizer applies it with
   ``param.add_(direction, alpha=-lr/bias_correction)``. The reference uses the fused ``addcdiv_``,
   which is an element-wise division and therefore assumes the preconditioner is diagonal in the
   parameter basis -- true only for ``diag``. The two paths differ by a few units in the last place
   per step on a ``Linear``; with ``BatchNorm2d`` in the network that gap is amplified to about
   1e-4 over six steps by the running statistics feeding back into the forward pass.
2. *Parameters are paired to modules by identity*, through a ``id(param) -> module`` map built once.
   **This is a bug fix, not a preference.** The reference walks both lists by position and runs
   exactly ``len(self.modules)`` iterations, spending one on each *unpaired* parameter, so a network
   with ``k`` parameters outside the four hooked types silently never steps its last ``k`` modules.
   Measured on a 32 x 32 ViT-S/4: the class token and the positional embedding cost the final
   ``LayerNorm`` and the whole classification head, which were never updated, with finite gradients
   and a still-falling loss. Do not restore the positional loop.
3. *``diag`` applies its min-max normalisation before the running average*, following the official
   AdaFisher repository rather than Algorithm 1 of the paper. See
   :mod:`adafisher_modes.approximations.diag`.

**Knobs that are off by default.** Each reproduces today's exact behaviour when left alone.

``gamma``
    Overrides ``gammas`` with ``(1 - gamma, 1 - gamma)``, which turns the running average into a
    convex one: ``gamma * old + (1 - gamma) * new``. The shipped ``gammas = (0.92, 0.008)`` has
    coefficients summing to 0.088, so the stored curvature settles at about 1/115 of what it
    estimates. Note that eq. (3) of the paper, read literally, puts ``gamma`` on the *new* term
    instead; both reference implementations put it on the history term, which is what this knob
    follows.
``fisher_batch_samples``
    Estimate the factors from the first ``k`` examples of each batch only. Dimension 0 is the
    example axis for all four supported layer types, so slicing it identically in both hooks keeps
    the input and gradient rows paired, which the intra-batch estimators of ``ekfac`` and ``tekfac``
    require. It changes the *estimator*, not the gradient that is applied. It exists for memory:
    every forward hook fires before any backward hook, so ``ekfac`` and ``tekfac`` hold every
    layer's cached input batch at once -- measured at 3.37 GB across ResNet-50 at batch 128, or
    1.51 GB with SUA on.
``decoupled_weight_decay``
    The official ``AdaFisherW`` rule: decay the parameter directly instead of adding
    ``weight_decay * param`` to the gradient. Needed to compare like-for-like against ``AdamW``.
``ema_seed_first``
    Start each running average from its first observation instead of from an identity, so no residue
    of that identity is left in the state.
``minmax_after_average``
    ``diag`` only; moves the min-max to where Algorithm 1 puts it.
``eig_before_rescale``
    ``ekfac`` and ``tekfac`` only. Rebuild the eigenbasis inside the backward hook, before the
    gradient is projected into it, so that the rescaling is measured in the basis ``precondition``
    then uses. By default the eigenbasis is replaced afterwards, in ``step()``, which leaves the
    two out of step -- the order both source papers' own algorithms and
    ``reference_repos/EKFAC-pytorch/ekfac.py`` put the other way round.
``damping``, ``damping_tau``
    The four Kronecker modes only. ``"global"`` (the default) adds the one shared ``Lambda`` to
    every module. ``"layer_relative"`` gives each module ``lambda_l = damping_tau * c_l``, where
    ``c_l`` is the mean eigenvalue of that module's own undamped stored curvature, recomputed at
    the start of every ``step()`` from the state as the hooks left it (fix S1 of
    ``docs/reports/plan_lambda_dominance.md``). ``"network_relative"`` gives every module the same
    ``damping_tau * c_net``, where ``c_net`` is the mean over all the network's curvature
    directions. ``kfac`` and ``tkfac`` bake the damping into inverses they rebuild every ``T_inv``
    steps, so for them a new ``lambda_l`` takes effect at the next rebuild.
``hold_cap``
    Multiply each preconditioned direction by the damping that went into it, so that ``lr`` is the
    step-size cap -- the largest multiple of the momentum any direction can move by -- whatever the
    damping is. ``hold_cap=True, lr=c`` is the same update as ``hold_cap=False, lr=c*Lambda``
    under a global damping; under a relative damping it holds that cap in every layer at once. A
    parameter with no curvature (the plain-momentum fallback) is scaled by the shared ``Lambda``.
    With decoupled weight decay the decay factor is ``1 - lr * weight_decay``, so under this knob
    it does not move with the damping.
``rescale_form``, ``clip_threshold``, ``clip_fraction``, ``clip_guard``, and three more
    ``ekfac`` and ``tekfac`` only: how the projected direction is divided by the stored curvature
    inside the eigenbasis. ``"add"`` (the default) is ``s + lambda``, what both papers prescribe.
    ``"floor"`` is ``max(s, lambda)``: values above ``lambda`` are used as they are. ``"clip"`` is
    a Sophia-type per-coordinate cap on the *undamped* step; ``lambda`` then plays no role. Its
    threshold is ``clip_threshold``, always per module: ``"quantile"`` clips a fraction
    ``clip_fraction`` of the module's active coordinates at every step (a per-module
    normalisation); ``"ema"`` does the same but lets the step follow the momentum's size relative
    to its bias-corrected average over ``clip_ema_horizon`` steps; ``"fixed"`` freezes, at step
    ``clip_calibrate_at``, the median of the module's quantile over the preceding
    ``clip_calibration_window`` steps (the three more: ``clip_ema_horizon``,
    ``clip_calibrate_at``, ``clip_calibration_window``). Under ``"clip"`` the mode receives the
    bias-corrected momentum. ``clip_guard`` keeps a coordinate of
    numerically zero curvature and near-zero momentum from being clipped to a full step. See
    :mod:`adafisher_modes.approximations._rescale_utils` and ``docs/reports/plan_floor_clip.md``.
``norm_exact_rescaling``
    ``ekfac`` and ``tekfac`` only. Estimate a normalisation layer's eigen-rescaling (``s*``,
    ``Theta``) from its exact per-row gradient ``[delta * x_hat, delta]`` instead of the input
    factor's surrogate ``[z * delta, delta]`` (``z`` = the raw input's channel mean, ``CLAUDE.md``
    §4.6). The factors and the eigenbasis are unchanged; EKFAC's Lemma 1 makes the new statistic the
    optimal diagonal in that basis. Measured before the fix: on ``vit_micro_cifar`` the surrogate
    under-states the scale column 800-8 000x. Inert on a network with no hooked normalisation
    layer. Refused with ``fisher_batch_samples`` on a network with a ``BatchNorm2d``, whose batch
    statistics cannot be recomputed from part of the batch.

Distributed training is not supported: the reference's all-reduce over the two factors has no
equivalent here, and silently accepting the flag would be worse than omitting it.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from torch import Tensor, is_grad_enabled, no_grad, zeros_like
from torch.nn import BatchNorm2d, LayerNorm, Module, Parameter
from torch.optim import Optimizer

from adafisher_modes.approximations import MODES

SUPPORTED_MODULES = ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm")


class AdaFisherMulti(Optimizer):
    """AdaFisher (Martins Gomes et al., ICLR 2025) with a selectable Fisher approximation mode.

    ``fisher_mode`` is one of ``"diag"`` (AdaFisher's own Prop. 3.2 / Eq. 4), ``"kfac"``,
    ``"ekfac"``, ``"tkfac"`` or ``"tekfac"``. Extra keyword arguments are forwarded to the mode:
    ``T_inv`` for ``kfac``/``tkfac``, ``T_eig`` for ``ekfac``/``tekfac``, ``T_re`` for ``tekfac``,
    ``pi`` for ``kfac``, ``conv_sua`` for all four. Passing one that the selected mode does not
    take is a ``TypeError`` rather than a silent no-op.
    """

    def __init__(
        self,
        model: Module,
        lr: float = 1e-3,
        beta: float = 0.9,
        Lambda: float = 1e-3,
        gammas: Sequence[float] = (0.92, 0.008),
        gamma: Optional[float] = None,
        TCov: int = 100,
        weight_decay: float = 0.0,
        fisher_mode: str = "diag",
        minmax_normalization: bool = True,
        minmax_after_average: bool = False,
        fisher_batch_samples: Optional[int] = None,
        decoupled_weight_decay: bool = False,
        ema_seed_first: bool = False,
        eig_before_rescale: bool = False,
        damping: str = "global",
        damping_tau: Optional[float] = None,
        hold_cap: bool = False,
        rescale_form: str = "add",
        clip_fraction: Optional[float] = None,
        clip_guard: float = 1e-3,
        clip_threshold: str = "quantile",
        clip_ema_horizon: Optional[int] = None,
        clip_calibrate_at: Optional[int] = None,
        clip_calibration_window: Optional[int] = None,
        norm_exact_rescaling: bool = False,
        **mode_kwargs,
    ) -> None:
        if fisher_mode not in MODES:
            raise ValueError(f"Unknown fisher_mode {fisher_mode!r}; available: {sorted(MODES)}")
        if fisher_batch_samples is not None and fisher_batch_samples < 1:
            raise ValueError(f"fisher_batch_samples must be >= 1 or None; got {fisher_batch_samples}")
        if damping not in ("global", "layer_relative", "network_relative"):
            raise ValueError(
                f"damping must be 'global', 'layer_relative' or 'network_relative'; got {damping!r}"
            )
        if damping == "global" and damping_tau is not None:
            raise ValueError("damping_tau only applies to a relative damping; damping is 'global'")
        if damping != "global":
            if damping_tau is None or not damping_tau > 0:
                raise ValueError(f"damping={damping!r} needs damping_tau > 0; got {damping_tau!r}")
            if fisher_mode == "diag":
                raise ValueError(
                    "a relative damping needs the curvature's own scale, which diag's min-max "
                    "normalisation removes; use one of the four Kronecker modes"
                )
        if gamma is not None:
            gammas = (1 - gamma, 1 - gamma)
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        self.damping = damping
        self.damping_tau = damping_tau
        self.hold_cap = hold_cap
        self.model = model
        self.TCov = TCov
        self.steps = 0
        self.fisher_batch_samples = fisher_batch_samples
        self.decoupled_weight_decay = decoupled_weight_decay
        self.modules: List[Module] = []
        self._owner: Dict[int, Module] = {}
        if fisher_mode == "diag":
            mode_kwargs = {
                "minmax_normalization": minmax_normalization,
                "minmax_after_average": minmax_after_average,
                **mode_kwargs,
            }
        # ema_seed_first applies to all five modes and is inert by default: it starts each running
        # average from its first observation instead of from an identity, so no residue of that
        # identity is ever left in the state.
        if fisher_mode in ("ekfac", "tekfac"):
            mode_kwargs = {"eig_before_rescale": eig_before_rescale, **mode_kwargs}
        elif eig_before_rescale:
            raise ValueError(
                f"eig_before_rescale only applies to the two modes that have an eigenbasis to "
                f"order against a rescaling, 'ekfac' and 'tekfac'; got fisher_mode={fisher_mode!r}."
            )
        clip_kwargs = {"clip_fraction": clip_fraction, "clip_ema_horizon": clip_ema_horizon,
                       "clip_calibrate_at": clip_calibrate_at,
                       "clip_calibration_window": clip_calibration_window}
        if norm_exact_rescaling:
            if fisher_mode not in ("ekfac", "tekfac"):
                raise ValueError(
                    f"norm_exact_rescaling only applies to the two modes that estimate an "
                    f"eigen-rescaling, 'ekfac' and 'tekfac'; got fisher_mode={fisher_mode!r}"
                )
            if fisher_batch_samples is not None and any(
                    isinstance(m, BatchNorm2d) for m in model.modules()):
                raise ValueError(
                    "norm_exact_rescaling recomputes a BatchNorm2d's normalised activation from "
                    "its batch statistics, which fisher_batch_samples would compute on part of "
                    "the batch only; use one or the other"
                )
            mode_kwargs = {"norm_exact_rescaling": True, **mode_kwargs}
        if (rescale_form != "add" or clip_threshold != "quantile"
                or any(v is not None for v in clip_kwargs.values())):
            if fisher_mode not in ("ekfac", "tekfac"):
                raise ValueError(
                    f"rescale_form and clip_fraction only apply to the two modes that divide "
                    f"coordinate by coordinate in an eigenbasis, 'ekfac' and 'tekfac'; got "
                    f"fisher_mode={fisher_mode!r}. kfac and tkfac invert their factors without "
                    f"ever diagonalising them, and diag's min-max removes the curvature's scale."
                )
            if rescale_form == "clip" and (hold_cap or damping != "global"):
                raise ValueError(
                    "rescale_form='clip' does not use lambda at all, so neither hold_cap (which "
                    "scales the step by lambda) nor a relative damping (which sets lambda) means "
                    "anything with it"
                )
            mode_kwargs = {"rescale_form": rescale_form, "clip_guard": clip_guard,
                           "clip_threshold": clip_threshold, **clip_kwargs, **mode_kwargs}
        self.approx = MODES[fisher_mode](
            Lambda=Lambda, gammas=gammas, ema_seed_first=ema_seed_first, **mode_kwargs
        )
        self._prepare_model()
        super().__init__(model.parameters(), defaults)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _fisher_slice(self, tensor: Tensor) -> Tensor:
        """First ``fisher_batch_samples`` examples of ``tensor``, or ``tensor`` itself when the
        knob is unset. Dimension 0 is the example axis for all four supported layer types, so
        applying this identically in both hooks selects the same examples in the same order on the
        input and gradient sides -- which is what ``ekfac`` and ``tekfac``'s intra-batch estimators
        need.
        """
        if self.fisher_batch_samples is None or tensor.size(0) <= self.fisher_batch_samples:
            return tensor
        return tensor[: self.fisher_batch_samples]

    def _save_input(self, module: Module, input, output) -> None:
        if is_grad_enabled() and self.steps % self.TCov == 0:
            self.approx.update_input_factor(module, self._fisher_slice(input[0].data), self.steps)

    def _save_grad_output(self, module: Module, grad_input, grad_output) -> None:
        if self.steps % self.TCov == 0:
            self.approx.update_output_factor(
                module, self._fisher_slice(grad_output[0].data), self.steps
            )

    def _prepare_model(self) -> None:
        for module in self.model.modules():
            if module.__class__.__name__ not in SUPPORTED_MODULES:
                continue
            if getattr(module, "weight", None) is None:
                # e.g. BatchNorm2d(affine=False) / LayerNorm(elementwise_affine=False): nothing to
                # precondition, and no weight to pair a parameter to.
                continue
            if isinstance(module, (BatchNorm2d, LayerNorm)) and module.bias is None:
                # A normalisation layer's input factor is always two columns wide, one for the
                # scale and one for the shift, in every mode. LayerNorm(d, bias=False) has the
                # scale and not the shift, so that factor describes a parameter the layer does not
                # have -- which used to surface only as a shape error inside the mode, several
                # calls away from its cause.
                raise NotImplementedError(
                    f"{module} has a scale parameter but no shift parameter, which this optimizer "
                    f"cannot precondition: the curvature factor of a normalisation layer covers "
                    f"the pair (scale, shift) jointly, in all five modes. Use the layer's default "
                    f"bias, or elementwise_affine=False, which leaves it with no parameters at all."
                )
            self.modules.append(module)
            self._owner[id(module.weight)] = module
            if module.bias is not None:
                self._owner[id(module.bias)] = module
            module.register_forward_hook(self._save_input)
            module.register_full_backward_hook(self._save_grad_output)

    # ------------------------------------------------------------------
    # Per-parameter first moment (Table 1's m^(t): momentum, bias correction, weight decay)
    # ------------------------------------------------------------------

    def _update_moment(self, hparams: Dict, param: Parameter, beta: float) -> tuple[Tensor, float]:
        state = self.state[param]
        if len(state) == 0:
            state["step"] = 0
            state["exp_avg"] = zeros_like(param)
        grad = param.grad
        if hparams["weight_decay"] != 0 and not self.decoupled_weight_decay:
            grad = grad.add(param, alpha=hparams["weight_decay"])
        state["exp_avg"].mul_(beta).add_(grad, alpha=1 - beta)
        state["step"] += 1
        bias_correction = 1 - beta ** state["step"]
        return state["exp_avg"], bias_correction

    @no_grad()
    def _step_module(
        self, hparams: Dict, module: Module, weight_param: Parameter, bias_param: Optional[Parameter]
    ) -> None:
        beta = hparams["beta"]
        weight_exp_avg, weight_bc = self._update_moment(hparams, weight_param, beta)
        bias_exp_avg, bias_bc = (
            self._update_moment(hparams, bias_param, beta) if bias_param is not None else (None, None)
        )

        if self.approx.consumes_bias_corrected_momentum:
            # The clip compares the momentum with a threshold, so it must see the bias-corrected
            # momentum; its result is then applied as it is. New tensors: never touch exp_avg.
            weight_exp_avg = weight_exp_avg / weight_bc
            bias_exp_avg = None if bias_exp_avg is None else bias_exp_avg / bias_bc
            weight_bc = 1.0
            bias_bc = None if bias_bc is None else 1.0
        direction = self.approx.precondition(module, weight_exp_avg, bias_exp_avg)
        if self.hold_cap:
            lam = self.approx.applied_lambda(module)
            direction = (
                tuple(d * lam for d in direction) if isinstance(direction, tuple) else direction * lam
            )

        if bias_param is not None:
            assert bias_bc is not None  # paired with bias_param by _update_moment above
            weight_direction, bias_direction = direction
            self._apply_update(hparams, weight_param, weight_direction, weight_bc)
            self._apply_update(hparams, bias_param, bias_direction, bias_bc)
        else:
            self._apply_update(hparams, weight_param, direction, weight_bc)

    def _apply_update(
        self, hparams: Dict, param: Parameter, direction: Tensor, bias_correction: float
    ) -> None:
        """``param -= lr * direction / bias_correction``, preceded by the decoupled decay factor
        when ``decoupled_weight_decay`` is on. ``p*(1 - lr*wd) - lr*d/bc`` is
        algebraically the official ``AdaFisherW._step`` expression
        ``p - lr*(d/bc + wd*p)``; with the flag off this is bit-identical to the pre-lot-8
        ``param.add_(direction, alpha=-lr/bc)``.
        """
        if self.decoupled_weight_decay and hparams["weight_decay"] != 0:
            param.mul_(1 - hparams["lr"] * hparams["weight_decay"])
        param.add_(direction, alpha=-hparams["lr"] / bias_correction)

    @no_grad()
    def _step_fallback(self, hparams: Dict, param: Parameter) -> None:
        """Plain SGD-with-momentum step (F~ = I), for a parameter not paired to any hooked
        module — same fallback as the original's ``F_tilde = ones_like(...)`` branch
        (adafisher.py:299), independent of ``fisher_mode``.
        """
        exp_avg, bias_correction = self._update_moment(hparams, param, hparams["beta"])
        if self.hold_cap:
            exp_avg = exp_avg * self.approx.Lambda   # a new tensor: never scale the momentum itself
        self._apply_update(hparams, param, exp_avg, bias_correction)

    def _set_relative_damping(self) -> None:
        """Give every module with stored curvature its damping for this step, from the state as the
        hooks left it: ``tau * c_l`` for ``layer_relative``, and ``tau * c_net`` for
        ``network_relative``, where ``c_net`` weighs each module by its number of directions.
        """
        tau = self.damping_tau
        assert tau is not None  # checked in __init__ for every damping other than "global"
        present = []
        for module in self.modules:
            if module.weight.grad is None:
                continue
            c = self.approx.mean_curvature(module)
            n = self.approx.num_directions(module)
            if c is not None and n is not None:
                present.append((module, c, n))
        if not present:
            return
        if self.damping == "layer_relative":
            for module, c, _ in present:
                self.approx.set_lambda(module, tau * c)
            return
        c_net = sum(c * n for _, c, n in present) / sum(n for _, _, n in present)
        for module, _, _ in present:
            self.approx.set_lambda(module, tau * c_net)

    @no_grad()
    def step(self, closure: Optional[Callable] = None) -> None:
        """Walk this optimizer's parameters in order, stepping each hooked module exactly once
        (weight and bias together, ``base.py``'s point 2) and falling back to plain
        momentum-SGD for every parameter that belongs to no hooked module.

        Replaces the reference's index-bookkeeping loop (``adafisher.py:275-307``), which paired
        parameters to modules positionally and by shape and silently skipped its last ``k`` modules
        on any model carrying ``k`` unpaired parameters. See this module's header.
        """
        if closure is not None:
            raise NotImplementedError("Closure not supported.")
        # One set for the whole walk, not one per parameter group. A module is preconditioned once,
        # with its weight and bias together, so "already stepped" is a property of the module and
        # not of the group the parameter happens to sit in. Built per group, the standard
        # decay/no-decay split stepped every hooked module twice -- once from its weight's group
        # and once from its bias's -- moving both parameters twice as far as intended. Measured on
        # a two-layer net with that split: the parameters moved exactly 2.0000 times as far as
        # under a single group.
        stepped: set[int] = set()
        if self.damping != "global":
            self._set_relative_damping()
        self.approx.begin_step(self.steps)
        for group in self.param_groups:
            hparams = {k: group[k] for k in ("lr", "beta", "weight_decay")}
            for param in group["params"]:
                if param.grad is None:
                    continue
                module = self._owner.get(id(param))
                if module is None:
                    self._step_fallback(hparams, param)
                    continue
                if id(module) in stepped:
                    continue  # the bias of a module already stepped alongside its weight
                if module.weight.grad is None:
                    # Weight frozen, bias not: no direction to precondition the module with.
                    self._step_fallback(hparams, param)
                    continue
                stepped.add(id(module))
                if module.bias is not None and module.bias.grad is None:
                    # The input factor of a module with a bias carries a constant-one column for
                    # it, so the direction handed to precondition() must carry a bias column too.
                    # Dropping it silently used to fail inside the mode: an assertion with no
                    # message in diag, a matrix-shape error in the four Kronecker modes.
                    raise RuntimeError(
                        f"{module} has a weight gradient but no bias gradient. A module cannot be "
                        f"preconditioned with only part of its direction: its Fisher factor treats "
                        f"the bias as one more input coordinate. The usual cause is "
                        f"bias.requires_grad = False on a layer whose weight is still trained. "
                        f"Freeze the whole module (weight and bias), or train the whole module."
                    )
                bias_param = module.bias
                self.approx.refresh(module, self.steps)
                self._step_module(hparams, module, module.weight, bias_param)
        self.steps += 1
