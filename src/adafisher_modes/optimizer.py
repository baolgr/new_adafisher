"""AdaFisher with an interchangeable Fisher approximation mode.

Ports ``AdaFisherBackbone`` / ``AdaFisher``
(``reference_repos/FisherAdapTune/scripts/adafisher.py:157-307``) almost verbatim: hook
registration, the ``_check_dim`` module/parameter pairing, and the index-bookkeeping ``step()``
loop are kept structurally unchanged (that bookkeeping is brittle and out of scope to rewrite, see
docs/reports/plan.md §2.1). Three deltas, all consequences of moving the second-moment construction
behind ``FisherApproximation`` (docs/reports/plan_lot1.md §0):

1. The forward/backward hooks delegate to ``approx.update_input_factor`` /
   ``update_output_factor`` instead of computing the EMA inline.
2. ``_step`` becomes ``_step_module``: weight and (if present) bias are preconditioned together in
   one ``approx.precondition`` call, so an expensive mode only pays its dominant cost once per
   module, not once per parameter.
3. ``precondition`` receives the raw ``exp_avg``; bias correction is folded into the optimizer's
   own step size (``alpha=-lr/bias_correction``), mirroring the original's ``step_size =
   lr/bias_correction`` (adafisher.py:272) as closely as possible.

Lot 8 (docs/reports/plan_lot8.md §0.3, §0.4, §0.6) makes three changes here, the first since lot 1:

4. The index-bookkeeping parameter/module pairing is replaced by an explicit, identity-keyed map
   built once in ``_prepare_model``. The ported loop ran exactly ``len(self.modules)`` iterations
   and consumed one iteration per *unpaired* parameter, so a model with ``k`` raw ``Parameter``s
   outside the four hooked module types (a ViT's ``cls_token``/``pos_embed``) silently never
   stepped its last ``k`` modules — measured on ViT-S/4: the final ``LayerNorm`` and the whole
   classification head were never updated (plan_lot8.md §0.3). Identity pairing cannot mis-pair,
   and reproduces the old loop exactly on every net where the old loop was correct.
5. ``fisher_batch_samples`` optionally restricts the *curvature statistic* to the first ``k``
   examples of each batch (plan_lot8.md §0.4): dimension 0 is the example axis for all four
   supported layer types, so slicing it identically in both hooks keeps ``h``/``delta`` row-paired
   by construction, which ``ekfac``/``tekfac``'s intra-batch estimators require. Default ``None``
   = unchanged behaviour.
6. ``decoupled_weight_decay`` implements the official ``AdaFisherW`` rule
   (``reference_repos/AdaFisher/optimizers/AdaFisher.py:685``, ``_step``:
   ``param -= lr * (exp_avg / bc / F_tilde + weight_decay * param)``), needed for a like-for-like
   comparison against ``AdamW`` on a ViT (plan_lot8.md §0.6). Default ``False`` = unchanged
   behaviour.

Lot-1 scope: no ``dist_training`` support yet (the original's distributed all-reduce over H, S,
adafisher.py:210-214, has no equivalent here) — silently accepting and ignoring that flag would be
worse than omitting it, so it is simply not part of this constructor yet.

``gamma`` (``docs/reports/audit_step.md`` §4, §4.7): the published Eq. (3) time-average is
``H^(t) = gamma*H^(t-1) + (1-gamma)*H_new``, one scalar with coefficients summing to 1. Both
reference repositories instead compute ``0.08*old + 0.008*new`` at their own tuned
``gammas=(0.92, 0.008)`` default — a discrepancy with the paper, not a porting bug (§4.2-§4.3).
Passing ``gamma`` here reproduces Eq. (3) exactly through the existing ``update_running_avg(new,
current, gammas)`` machinery unchanged: that function computes ``(1-gammas[0])*current +
gammas[1]*new``, so ``gammas=(1-gamma, 1-gamma)`` collapses it to Eq. (3)'s single-coefficient
form. ``gamma=None`` (default) leaves ``gammas`` exactly as passed — bit-identical to today's
behaviour. When both are given, ``gamma`` wins.

``minmax_after_average`` (``diag`` mode only, ``docs/reports/audit_step.md`` §4.4, §4.7): Algorithm
1 applies Eq. (4)'s min-max normalisation after the EMA (its line 4 then line 5); the official code
-- and this port's default -- does it before. Passed straight through to ``DiagApproximation``; see
its own docstring for why the two orders are not equivalent.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from torch import Tensor, is_grad_enabled, no_grad, zeros_like
from torch.nn import Module, Parameter
from torch.optim import Optimizer

from adafisher_modes.approximations import MODES

SUPPORTED_MODULES = ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm")


class AdaFisherMulti(Optimizer):
    """AdaFisher (Martins Gomes et al., ICLR 2025) with a selectable Fisher approximation mode.

    Only ``fisher_mode="diag"`` (AdaFisher's own Prop. 3.2 / Eq. 4) is implemented so far.
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
        **mode_kwargs,
    ) -> None:
        if fisher_mode not in MODES:
            raise ValueError(f"Unknown fisher_mode {fisher_mode!r}; available: {sorted(MODES)}")
        if fisher_batch_samples is not None and fisher_batch_samples < 1:
            raise ValueError(f"fisher_batch_samples must be >= 1 or None; got {fisher_batch_samples}")
        if gamma is not None:
            gammas = (1 - gamma, 1 - gamma)
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
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
        self.approx = MODES[fisher_mode](Lambda=Lambda, gammas=gammas, **mode_kwargs)
        self._prepare_model()
        super().__init__(model.parameters(), defaults)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _fisher_slice(self, tensor: Tensor) -> Tensor:
        """First ``fisher_batch_samples`` examples of ``tensor`` (plan_lot8.md §0.4), or ``tensor``
        itself when the knob is unset. Dimension 0 is the example axis for all four supported layer
        types, so applying this identically in both hooks selects the same examples in the same
        order on the ``h`` and ``delta`` sides — which is what ``ekfac``/``tekfac``'s intra-batch
        estimators need.
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

        direction = self.approx.precondition(module, weight_exp_avg, bias_exp_avg)

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
        when ``decoupled_weight_decay`` is on (plan_lot8.md §0.6). ``p*(1 - lr*wd) - lr*d/bc`` is
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
        self._apply_update(hparams, param, exp_avg, bias_correction)

    @no_grad()
    def step(self, closure: Optional[Callable] = None) -> None:
        """Walk this optimizer's parameters in order, stepping each hooked module exactly once
        (weight and bias together, ``base.py``'s point 2) and falling back to plain
        momentum-SGD for every parameter that belongs to no hooked module.

        Replaces the reference's index-bookkeeping loop (adafisher.py:275-307), which paired
        parameters to modules positionally and by shape and silently skipped its last ``k`` modules
        on any model carrying ``k`` unpaired parameters — see this module's header and
        docs/reports/plan_lot8.md §0.3.
        """
        if closure is not None:
            raise NotImplementedError("Closure not supported.")
        for group in self.param_groups:
            hparams = {k: group[k] for k in ("lr", "beta", "weight_decay")}
            stepped: set[int] = set()
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
                bias_param = (
                    module.bias
                    if module.bias is not None and module.bias.grad is not None
                    else None
                )
                self.approx.refresh(module, self.steps)
                self._step_module(hparams, module, module.weight, bias_param)
        self.steps += 1
