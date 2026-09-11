"""Trajectory checkpointing (``plan_exp_step1.md`` D5) — the input steps 2-5 of
``plan_exp_draft.md`` consume: ``theta`` at ``t in {0, 1%, 10%, 50%, 100%}`` of a run's steps
(``plan_exp_draft.md`` §7, "Checkpoints").

Off by default. ``--checkpoints 0,0.01,0.1,0.5,1`` writes ``<arm dir>/ckpt_<frac>.pt``, each a
``dict`` carrying the model state, the step and epoch it was taken at, the run's seed, and the
fraction — everything needed to reload it into a fresh model and to label it in a result table.

The schedule is expressed in *completed steps* against the run's **nominal** length (``--epochs``
times the batches per epoch), which under the WCT protocol is shared by every arm of a model — so
``ckpt_0.5`` means the same amount of training in every arm, and the fractions are comparable
across them. ``t=0`` is the initialization (the loop calls ``on_step`` once with ``0`` before the
first batch).

A wall-clock-budgeted arm need not reach the nominal length: an expensive mode stops short, a cheap
one overshoots it. ``save_final`` therefore pins the largest requested fraction to wherever the arm
actually ended, *if* it never fired on schedule — and marks that payload ``scheduled=False``, so a
consumer can tell "the 100% point of the nominal trajectory" from "this arm's own last step". Every
payload carries its ``step``, ``epoch`` and the ``total_steps`` denominator, so no fraction is ever
ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

DEFAULT_FRACTIONS: Tuple[float, ...] = (0.0, 0.01, 0.1, 0.5, 1.0)


def parse_fractions(spec: str) -> Tuple[float, ...]:
    """``"0,0.01,0.1,0.5,1"`` -> ``(0.0, 0.01, 0.1, 0.5, 1.0)``, sorted and de-duplicated."""
    fractions = sorted({float(part) for part in spec.split(",") if part.strip()})
    if any(not 0.0 <= f <= 1.0 for f in fractions):
        raise ValueError(f"checkpoint fractions must lie in [0, 1]; got {spec!r}")
    return tuple(fractions)


def label(fraction: float) -> str:
    return f"{fraction:g}"


@dataclass
class CheckpointWriter:
    """``on_step`` callback writing the scheduled checkpoints of one arm."""

    output_dir: Path
    fractions: Sequence[float]
    total_steps: int
    seed: int
    written: Dict[str, str] = field(default_factory=dict)

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
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "fraction": fraction,
                "step": completed_steps,
                "epoch": epoch,
                "seed": self.seed,
                # The denominator the fraction is relative to, and whether this payload landed on
                # the scheduled step or was pinned to the arm's own end by ``save_final``.
                "total_steps": self.total_steps,
                "scheduled": scheduled,
            },
            path,
        )
        self.written[label(fraction)] = str(path)
