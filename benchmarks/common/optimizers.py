"""The single optimizer factory: nine arms, one hyperparameter record.

``build_optimizer(arm, model, hp)`` returns a ready optimizer for one of:

* the five ``AdaFisherMulti`` Fisher modes — ``diag``, ``kfac``, ``ekfac``, ``tkfac``, ``tekfac``;
* the two baselines, ``adam`` and ``adamw``;
* ``reference``, FisherAdapTune's own ``AdaFisher``
  (``reference_repos/FisherAdapTune/scripts/adafisher.py``), loaded by file path so that the
  read-only reference repository's package ``__init__`` is never imported. It is the original
  implementation this project's ``diag`` mode is a port of, and it stays available to every bench.
* ``official``, the *other* original — ``reference_repos/AdaFisher/optimizers/AdaFisher.py``, the
  optimizer of the authors' own published repository, loaded the same way and equally unmodified.
  It is the class that produced the paper's Table 2, and unlike ``reference`` it applies Eq. (4)'s
  min-max normalisation, so it is the like-for-like partner of ``diag`` at its default
  ``minmax=True`` (``reference`` is the partner of ``diag --no-minmax``, which it matches
  bit-exactly). Its ``gamma`` is one scalar rather than a pair; see :func:`official_gamma`.
* ``sgdm``, a control with no curvature at all: :class:`LambdaLimitSGD`, which is
  ``AdaFisherMulti``'s own update with every Fisher estimate replaced by ``lam * I``. It takes the
  Fisher arms' ``lr``, ``lam``, ``beta`` and decay convention, not ``baseline_lr``, because it
  exists to be compared with them: it is what they reduce to when ``lam`` dominates the stored
  curvature. E18 of ``docs/reports/plan_lambda_dominance.md``.

:class:`HParams` is the arm-independent operating point. One instance lives with each model, as a
literal in its ``bench.py``; this module owns only the field meanings and their defaults. Field
names are also the command-line flag names — ``runner.add_hparam_arguments`` generates one
``--<field>`` flag per field — so a new knob needs no CLI code.

Two conventions to know about. The baselines take ``baseline_lr`` rather than ``lr``, because
AdaFisher's Table 9 tunes Adam/AdamW separately from AdaFisher on ViTs. ``weight_decay`` is shared
by every arm and applied to every parameter, including biases and normalisation scales; what
differs between arms is the *convention* — ``Adam`` and ``AdaFisher`` fold the decay into the
gradient, ``AdamW`` and ``AdaFisherW`` (``decoupled_wd=True``) apply it directly to the weights.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from adafisher_modes import AdaFisherMulti
from adafisher_modes.factors import SUPPORTED_MODULES

REPO_ROOT = Path(__file__).resolve().parents[2]
FISHERADAPTUNE_ADAFISHER = REPO_ROOT / "reference_repos/FisherAdapTune/scripts/adafisher.py"
OFFICIAL_ADAFISHER = REPO_ROOT / "reference_repos/AdaFisher/optimizers/AdaFisher.py"

FISHER_ARMS = ("diag", "kfac", "ekfac", "tkfac", "tekfac")
BASELINE_ARMS = ("adam", "adamw")
ARMS = FISHER_ARMS + BASELINE_ARMS + ("reference", "official", "sgdm")


@dataclass(frozen=True)
class HParams:
    """One arm-independent hyperparameter set per model. Field names are also the CLI flag names
    (``runner.add_hparam_arguments`` generates them from ``fields(HParams)``).
    """

    lr: float = 1e-3
    baseline_lr: float = 1e-3  # adam / adamw: AdaFisher's Table 9 tunes them separately
    weight_decay: float = 0.0
    lam: float = 1e-3  # AdaFisherMulti's ``Lambda``
    beta: float = 0.9  # AdaFisher's beta_1, the momentum of m^(t)
    gammas: Tuple[float, float] = (0.92, 0.008)
    gamma: Optional[float] = None  # overrides gammas with Eq. (3)'s published single-gamma rule
    tcov: int = 100
    t_inv: int = 100  # kfac, tkfac
    t_eig: int = 100  # ekfac, tekfac
    t_re: int = 1  # tekfac, T_RE of Alg. 1
    minmax: bool = True  # diag only; True = faithful to AdaFisher Eq. (4)
    minmax_after_average: bool = False  # diag only; True = AdaFisher Algorithm 1's order
                                         # (normalise the running average) instead of the official
                                         # code's (normalise each instantaneous factor)
    conv_sua: bool = False  # the four Kronecker modes, Conv2d only
    fisher_batch_samples: Optional[int] = None
    decoupled_wd: bool = False  # True = AdaFisherW / AdamW convention


def _load_by_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_reference_adafisher() -> ModuleType:
    return _load_by_path("_ref_fisheradaptune_adafisher", FISHERADAPTUNE_ADAFISHER)


def load_official_adafisher() -> ModuleType:
    return _load_by_path("_ref_official_adafisher", OFFICIAL_ADAFISHER)


def official_gamma(hp: HParams) -> float:
    """The single ``gamma`` the official ``AdaFisher`` takes, from ``hp.gammas``.

    The official running average is ``current = (gamma * 1e-1) * current + (gamma * 1e-2) * new``
    (``AdaFisher.py::update_running_avg``), so one scalar sets both coefficients. This project
    stores the same two coefficients as a pair, ``(1 - gammas[0], gammas[1])``. The published
    ``gamma = 0.8`` is therefore exactly this project's default ``gammas = (0.92, 0.008)``.

    A pair that no single ``gamma`` can express is refused rather than silently rounded: the arm
    exists to run the authors' code at *their* operating point, so a hyperparameter that cannot
    reach it must be an error.
    """
    if hp.gamma is not None:
        raise ValueError(
            "the 'official' arm cannot take --gamma: that flag is AdaFisherMulti's Eq. (3) "
            "single-gamma EMA, which the published code does not implement (its own 'gamma' is "
            "the scalar behind the (0.08, 0.008) pair — see official_gamma)."
        )
    old, new = 1.0 - hp.gammas[0], hp.gammas[1]
    gamma = new * 1e2
    if abs(old - gamma * 1e-1) > 1e-12 or not 0.0 <= gamma < 1.0:
        raise ValueError(
            f"gammas={hp.gammas} is not reachable by the official AdaFisher, whose two EMA "
            f"coefficients are (gamma*1e-1, gamma*1e-2) for one scalar gamma; this pair asks for "
            f"({old}, {new}), i.e. gamma={gamma} and {gamma * 1e-1}."
        )
    return gamma


class LambdaLimitSGD(torch.optim.Optimizer):
    """``AdaFisherMulti`` with every Fisher estimate replaced by ``lam * I``: the ``sgdm`` arm.

    At the shipped damping the Fisher arms divide by a matrix that is almost exactly ``lam * I``
    (lot 5 measured ``cond`` of it at 1.0000-1.1035 on the regime-A models). This arm is that limit
    taken exactly, so that the only thing separating it from a Fisher arm is the curvature. Per
    parameter, with ``eta`` the group's current learning rate and ``t`` the step count:

        m_t   = beta * m_{t-1} + (1 - beta) * g_t          (g_t + wd * theta if the decay is coupled)
        theta <- theta * (1 - eta * wd)                    (only if the decay is decoupled)
        theta <- theta - eta / (1 - beta^t) * m_t / lam    (a parameter of a hooked module)
        theta <- theta - eta / (1 - beta^t) * m_t          (any other parameter)

    The last line is ``AdaFisherMulti``'s own fallback for parameters no hooked module owns (a ViT's
    ``cls_token`` and ``pos_embed``), kept as it is: dividing those by ``lam`` too would make this
    arm differ from the Fisher arms in a second way. Which parameters are hooked is decided exactly
    as ``AdaFisherMulti._prepare_model`` decides it, and a hooked bias whose module's weight has no
    gradient takes the fallback, as it does there.

    **Why the arm is called ``sgdm``.** Write torch's momentum buffer ``b_t = beta * b_{t-1} + g_t``
    (``torch.optim.SGD`` with ``dampening=0``, ``b_1 = g_1``). Then ``m_t = (1 - beta) * b_t``
    exactly, so the hooked update is ``torch.optim.SGD(momentum=beta)`` at the learning rate
    ``eta * (1 - beta) / (lam * (1 - beta^t))``. Once ``beta^t`` is negligible (``0.9^44 < 0.01``)
    that is ``eta * (1 - beta) / lam``: 0.033 at the ViT benches' ``lr = 1e-3``, ``lam = 3e-3``.

    No hook and no curvature statistic, so a step costs what a momentum-SGD step costs. Under the
    wall-clock-time protocol the arm therefore completes more steps than any Fisher arm.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        beta: float,
        lam: float,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
    ) -> None:
        if lam <= 0:
            raise ValueError(f"lam must be > 0; got {lam}")
        super().__init__(model.parameters(), dict(lr=lr, beta=beta, weight_decay=weight_decay))
        self.lam = lam
        self.decoupled_weight_decay = decoupled_weight_decay
        # id(param) -> the module that owns it, for the same modules AdaFisherMulti hooks.
        self._owner: Dict[int, nn.Module] = {}
        for module in model.modules():
            if module.__class__.__name__ not in SUPPORTED_MODULES:
                continue
            if getattr(module, "weight", None) is None:
                continue
            self._owner[id(module.weight)] = module
            if module.bias is not None:
                self._owner[id(module.bias)] = module

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> None:
        if closure is not None:
            raise NotImplementedError("Closure not supported.")
        for group in self.param_groups:
            lr, beta, wd = group["lr"], group["beta"], group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                grad = param.grad
                if wd != 0 and not self.decoupled_weight_decay:
                    grad = grad.add(param, alpha=wd)
                state["exp_avg"].mul_(beta).add_(grad, alpha=1 - beta)
                state["step"] += 1
                bias_correction = 1 - beta ** state["step"]
                module = self._owner.get(id(param))
                hooked = module is not None and module.weight.grad is not None
                direction = state["exp_avg"] / self.lam if hooked else state["exp_avg"]
                if self.decoupled_weight_decay and wd != 0:
                    param.mul_(1 - lr * wd)
                param.add_(direction, alpha=-lr / bias_correction)


def build_optimizer(arm: str, model: nn.Module, hp: HParams, **overrides: Any) -> Any:
    """The nine arms at ``hp``'s operating point.

    ``overrides`` are passed straight to ``AdaFisherMulti`` and win over anything derived from
    ``hp``. They exist for experiments that vary one optimizer knob outside the ``HParams`` set,
    such as sweeping ``gamma`` or ``ema_seed_first``. Passing one for the
    ``adam``/``adamw``/``reference`` arms, which do not take them, is an error rather than a silent
    no-op.

    The two baselines take ``baseline_lr``: AdaFisher's Table 9 gives Adam/AdamW a different tuned
    learning rate from AdaFisher's on ViTs. ``weight_decay`` is shared by every arm; what differs
    is the convention — ``Adam`` and ``AdaFisher`` couple it into the gradient, ``AdamW`` and
    ``AdaFisherW`` decouple it.
    """
    if overrides and arm in ("adam", "adamw", "reference", "official", "sgdm"):
        raise ValueError(f"arm {arm!r} takes no AdaFisherMulti overrides; got {sorted(overrides)}")
    if arm == "sgdm":
        return LambdaLimitSGD(model, lr=hp.lr, beta=hp.beta, lam=hp.lam,
                              weight_decay=hp.weight_decay,
                              decoupled_weight_decay=hp.decoupled_wd)
    if arm == "adam":
        return torch.optim.Adam(model.parameters(), lr=hp.baseline_lr,
                                weight_decay=hp.weight_decay)
    if arm == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=hp.baseline_lr,
                                 weight_decay=hp.weight_decay)
    if arm == "reference":
        return load_reference_adafisher().AdaFisher(
            model, lr=hp.lr, beta=hp.beta, Lambda=hp.lam, TCov=hp.tcov, gammas=list(hp.gammas),
            weight_decay=hp.weight_decay,
        )
    if arm == "official":
        official = load_official_adafisher()
        cls = official.AdaFisherW if hp.decoupled_wd else official.AdaFisher
        return cls(
            model, lr=hp.lr, beta=hp.beta, Lambda=hp.lam, gamma=official_gamma(hp),
            TCov=hp.tcov, weight_decay=hp.weight_decay,
        )
    if arm not in FISHER_ARMS:
        raise ValueError(f"Unknown arm {arm!r}; available: {list(ARMS)}")

    kwargs: Dict[str, Any] = dict(
        lr=hp.lr,
        beta=hp.beta,
        Lambda=hp.lam,
        gammas=list(hp.gammas),
        gamma=hp.gamma,
        TCov=hp.tcov,
        weight_decay=hp.weight_decay,
        fisher_mode=arm,
        fisher_batch_samples=hp.fisher_batch_samples,
        decoupled_weight_decay=hp.decoupled_wd,
    )
    if arm == "diag":
        kwargs["minmax_normalization"] = hp.minmax
        kwargs["minmax_after_average"] = hp.minmax_after_average
    else:
        kwargs["conv_sua"] = hp.conv_sua
        if arm in ("kfac", "tkfac"):
            kwargs["T_inv"] = hp.t_inv
        if arm in ("ekfac", "tekfac"):
            kwargs["T_eig"] = hp.t_eig
        if arm == "tekfac":
            kwargs["T_re"] = hp.t_re
    kwargs.update(overrides)

    return AdaFisherMulti(model, **kwargs)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
