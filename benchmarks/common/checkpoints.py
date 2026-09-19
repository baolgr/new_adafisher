"""Trajectory checkpointing: the model weights at fixed fractions of a run.

Off by default. ``--checkpoints 0,0.01,0.1,0.5,1`` writes ``<arm dir>/ckpt_<fraction>.pt``, each a
``dict`` carrying the model state, the step and epoch it was taken at, the run's seed, the
fraction, the denominator that fraction is relative to, and whether it landed on schedule. That is
everything needed to reload it into a fresh model and to label it unambiguously in a result table.
These files are the input the curvature analysis in ``fisher_ref/`` consumes.

:class:`CheckpointWriter` is the ``on_step`` callback of ``benchmarks/common/loop.py``: it fires
after every optimizer step and writes whichever fractions fall due. ``t=0`` is the initialization,
because the loop calls ``on_step`` once with zero completed steps before the first batch.

The denominator is the nominal trajectory, not the arm's own
------------------------------------------------------------

The schedule is expressed in completed steps against ``--epochs`` times the batches per epoch.
Under the wall-clock-time protocol every arm of a model shares that one nominal length, so
``ckpt_0.5`` means the same amount of training in every arm and the fractions are comparable
across arms — which is the whole point of the dumps. Do not change this to the arm's own maximum
epoch count: doing so once put a budgeted arm's ``ckpt_0.1`` at about 31% of its own run and made
``ckpt_0.5`` unreachable on six arms out of seven.

A wall-clock-budgeted arm need not reach the nominal length: an expensive mode stops short, a
cheap one overshoots. ``save_final`` therefore pins the largest requested fraction to wherever the
arm actually ended, but only if it never fired on schedule, and marks that payload
``scheduled=False``. A consumer can then tell "the 100% point of the nominal trajectory" from
"this arm's own last step".

The optimizer's state, optionally
---------------------------------

``--checkpoint-optimizer-state`` (opt-in, off by default, so today's payloads are unchanged byte
for byte) adds the optimizer's own state next to the weights: ``torch.optim``'s ``state_dict()``
for any optimizer, plus — for ``AdaFisherMulti`` — its running-average per-module Fisher factors
re-keyed by module name, since the live dicts are keyed by ``nn.Module`` objects, which do not
survive a reload. Entries whose name starts with ``_cached`` are excluded: they hold one batch's
layer inputs, not state (3.37 GB across ResNet-50 at batch 128), and the next factor update
rebuilds them anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

DEFAULT_FRACTIONS: Tuple[float, ...] = (0.0, 0.01, 0.1, 0.5, 1.0)


def parse_fractions(spec: str) -> Tuple[float, ...]:
    """``"0,0.01,0.1,0.5,1"`` -> ``(0.0, 0.01, 0.1, 0.5, 1.0)``, sorted and de-duplicated."""
    fractions = sorted({float(part) for part in spec.split(",") if part.strip()})
    if any(not 0.0 <= f <= 1.0 for f in fractions):
        raise ValueError(f"checkpoint fractions must lie in [0, 1]; got {spec!r}")
    return tuple(fractions)


def label(fraction: float) -> str:
    return f"{fraction:g}"


#: Live per-module dicts whose name starts with one of these hold a *batch*, not state.
TRANSIENT_PREFIXES: Tuple[str, ...] = ("_cached",)


def optimizer_state_payload(optimizer: Any, model: nn.Module) -> Dict[str, Any]:
    """The optimizer's own state, serialisable and re-keyed by module name.

    Duck-typed on purpose: ``adam``/``adamw`` have no ``approx`` and contribute their
    ``state_dict()`` alone, so this module needs no import from ``adafisher_modes`` and no branch
    per arm. For ``AdaFisherMulti`` the Fisher factors are read generically out of
    ``vars(optimizer.approx)`` — every mode stores its state as ``{nn.Module: Tensor}`` dicts, so
    ``diag``'s ``_H``/``_S`` and ``tekfac``'s ``_Phi_raw``/``_Theta``/... are all covered without
    naming any of them here.
    """
    payload: Dict[str, Any] = {
        "optimizer_class": type(optimizer).__name__,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    approx = getattr(optimizer, "approx", None)
    if approx is None:
        return payload

    names = {id(module): name for name, module in model.named_modules()}
    fisher: Dict[str, Dict[str, Tensor]] = {}
    for family, mapping in vars(approx).items():
        if not isinstance(mapping, dict) or family.startswith(TRANSIENT_PREFIXES):
            continue
        by_name = {names[id(key)]: value.detach().to("cpu").clone()
                   for key, value in mapping.items()
                   if isinstance(key, nn.Module) and id(key) in names
                   and isinstance(value, Tensor)}
        if by_name:
            fisher[family] = by_name
    payload["fisher_state"] = fisher
    payload["fisher_approximation"] = type(approx).__name__
    payload["optimizer_steps"] = getattr(optimizer, "steps", None)
    return payload


@dataclass
class CheckpointWriter:
    """``on_step`` callback writing the scheduled checkpoints of one arm."""

    output_dir: Path
    fractions: Sequence[float]
    total_steps: int
    seed: int
    written: Dict[str, str] = field(default_factory=dict)
    #: Opt-in (``--checkpoint-optimizer-state``). Both must be set for the optimizer's state to be
    #: written; leaving them alone reproduces the previous payload exactly.
    optimizer: Optional[Any] = None
    save_optimizer_state: bool = False

    def _due_at(self, completed_steps: int) -> List[float]:
        return [f for f in self.fractions
                if round(f * self.total_steps) == completed_steps and label(f) not in self.written]

    def __call__(self, completed_steps: int, epoch: int, model: nn.Module) -> None:
        for fraction in self._due_at(completed_steps):
            self.save(fraction, completed_steps, epoch, model)

    def save_final(self, completed_steps: int, epoch: int, model: nn.Module) -> None:
        """A wall-clock-budgeted run can stop short of the nominal length; the largest requested
        fraction is then pinned to wherever the run actually ended, flagged ``scheduled=False``.
        An arm that overshot the nominal length already has that fraction on schedule, and keeps it.
        """
        if not self.fractions:
            return
        fraction = max(self.fractions)
        if label(fraction) not in self.written:
            self.save(fraction, completed_steps, epoch, model, scheduled=False)

    def save(self, fraction: float, completed_steps: int, epoch: int, model: nn.Module,
             scheduled: bool = True) -> None:
        path = self.output_dir / f"ckpt_{label(fraction)}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "fraction": fraction,
            "step": completed_steps,
            "epoch": epoch,
            "seed": self.seed,
            # The denominator the fraction is relative to, and whether this payload landed on the
            # scheduled step or was pinned to the arm's own end by ``save_final``.
            "total_steps": self.total_steps,
            "scheduled": scheduled,
        }
        if self.save_optimizer_state and self.optimizer is not None:
            payload["optimizer_state"] = optimizer_state_payload(self.optimizer, model)
        torch.save(payload, path)
        self.written[label(fraction)] = str(path)
