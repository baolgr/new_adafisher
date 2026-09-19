"""The single optimizer factory: eight arms, one hyperparameter record.

``build_optimizer(arm, model, hp)`` returns a ready optimizer for one of:

* the five ``AdaFisherMulti`` Fisher modes — ``diag``, ``kfac``, ``ekfac``, ``tkfac``, ``tekfac``;
* the two baselines, ``adam`` and ``adamw``;
* ``reference``, FisherAdapTune's own ``AdaFisher``
  (``reference_repos/FisherAdapTune/scripts/adafisher.py``), loaded by file path so that the
  read-only reference repository's package ``__init__`` is never imported. It is the original
  implementation this project's ``diag`` mode is a port of, and it stays available to every bench.

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

REPO_ROOT = Path(__file__).resolve().parents[2]
FISHERADAPTUNE_ADAFISHER = REPO_ROOT / "reference_repos/FisherAdapTune/scripts/adafisher.py"

FISHER_ARMS = ("diag", "kfac", "ekfac", "tkfac", "tekfac")
BASELINE_ARMS = ("adam", "adamw")
ARMS = FISHER_ARMS + BASELINE_ARMS + ("reference",)


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


def load_reference_adafisher() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_ref_fisheradaptune_adafisher", FISHERADAPTUNE_ADAFISHER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_optimizer(arm: str, model: nn.Module, hp: HParams, **overrides: Any) -> Any:
    """The eight arms at ``hp``'s operating point.

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
    if overrides and arm in ("adam", "adamw", "reference"):
        raise ValueError(f"arm {arm!r} takes no AdaFisherMulti overrides; got {sorted(overrides)}")
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
