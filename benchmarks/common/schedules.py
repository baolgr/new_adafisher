"""Cosine annealing driven by an explicit *progress fraction*, so the schedule can be aligned with
whatever a WCT-budgeted arm actually gets to run.

Why this module exists
----------------------

AdaFisher Appendix D specifies "a cosine annealing learning rate decay strategy [...] aligning with
the number of training epochs specified for each optimizer". Under the WCT protocol
(``runner.py``'s docstring) the arms of one model complete *different* epoch counts inside the same
wall-clock budget, so "the number of training epochs specified for each optimizer" has two readings,
and the first campaign used the one that biases the comparison at both ends:

``nominal`` — one shared ``T_max = --epochs``. Measured consequences on the first campaign:

* a **cheap** arm overshoots the nominal epoch count, and ``torch.optim.lr_scheduler.
  CosineAnnealingLR`` is *periodic*: past ``T_max`` it climbs back towards the base LR. Every
  overshooting arm was ``adam``/``adamw`` (7 models out of 8) — ``resnet20_cifar/adam`` reached
  ``lr = 0`` at epoch 49, then trained 9 further epochs with the LR rising back to ``6.2e-5``, and
  its best val acc (88.82%) decayed to 88.06% by the end;
* an **expensive** arm stops short and never reaches the annealing floor.
  ``resnet50_cifar/ekfac`` stopped at epoch 38 of 50 with ``lr`` still at ``1.6e-4``. Across the
  five Fisher arms, ``corr(epochs completed, best val acc)`` was **+0.98** (``resnet20_cifar``),
  **+0.92** (``resnet50_cifar``) and **+0.92** (``vit_small_cifar``) — i.e. on the long models the
  ranking of the modes was largely a ranking of how far each got through the shared schedule.

Both failure modes are removed here:

* :class:`NominalCosine` keeps the shared ``T_max = --epochs`` and is **clamped**: past ``T_max`` the
  LR holds at ``eta_min`` instead of rising. For every epoch ``e <= T_max`` it returns exactly what
  ``CosineAnnealingLR`` returned (same closed form, ``_get_closed_form_lr``), so no arm that stayed
  within the nominal length changes at all — the first campaign's bit-exactly reproducible ``diag``
  trajectories are preserved.
* :class:`BudgetCosine` anneals over the arm's **own wall-clock budget**: progress is
  ``elapsed / budget_s``, evaluated before each batch, so every budgeted arm completes one full
  cosine and is compared at the same point of its own schedule. This is the reading that matches
  what the protocol asks ("the best model obtainable in T seconds"), and it makes
  ``--max-epoch-factor`` a pure defensive bound rather than a schedule-shaping parameter.

One interaction to know about: :class:`BudgetCosine` assumes the arm actually spends its budget.
``--max-epoch-factor`` caps how many epochs a budgeted arm may run, and if that cap binds first
the arm stops part-way up its cosine, under-annealed — the very thing this class exists to
prevent. No arm of the first campaign came close (the cheapest, ``resnet20_cifar/adam``, used 59
of its 150 allowed epochs), so a bound cap means the budget is far too generous for that arm;
``runner.run_arm`` prints a warning naming the unspent budget and the LR it stopped at.

The one thing :class:`BudgetCosine` costs is bit-exact reproducibility: the LR now depends on
measured elapsed time, so two runs of the same arm agree statistically, not to the last bit. The
reference arm is unaffected — it is unbudgeted by construction (it is what *defines* the budget), so
``runner.py`` always drives it with :class:`NominalCosine`, exactly as before.
"""

from __future__ import annotations

import math
from typing import Any, List


class _ProgressCosine:
    """Shared machinery: set every param group's LR from a progress fraction in ``[0, 1]``.

    ``progress = 0`` restores the base LR captured at construction, ``progress = 1`` gives
    ``eta_min``; values outside the interval are clamped, which is what turns the periodic
    behaviour of ``CosineAnnealingLR`` into a floor.
    """

    def __init__(self, optimizer: Any, eta_min: float = 0.0) -> None:
        self.optimizer = optimizer
        self.eta_min = eta_min
        self.base_lrs: List[float] = [float(g["lr"]) for g in optimizer.param_groups]

    def set_progress(self, progress: float) -> None:
        p = min(max(progress, 0.0), 1.0)
        factor = 0.5 * (1.0 + math.cos(math.pi * p))
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.eta_min + (base - self.eta_min) * factor


class NominalCosine(_ProgressCosine):
    """Cosine over a fixed ``t_max`` epochs, clamped at ``eta_min`` past it.

    Exposes ``step()``, so it is a drop-in for the ``CosineAnnealingLR`` the loop already drives
    once per *completed* epoch (``loop.train_under_budget``); nothing about that contract changes.
    """

    def __init__(self, optimizer: Any, t_max: int, eta_min: float = 0.0) -> None:
        super().__init__(optimizer, eta_min)
        if t_max <= 0:
            raise ValueError(f"t_max must be positive; got {t_max}")
        self.t_max = t_max
        self.completed_epochs = 0

    def step(self) -> None:
        self.completed_epochs += 1
        self.set_progress(self.completed_epochs / self.t_max)


class BudgetCosine(_ProgressCosine):
    """Cosine over a wall-clock budget: ``progress = elapsed_s / budget_s``.

    Exposes ``__call__(elapsed_s)``, the signature of ``loop.train_under_budget``'s ``lr_schedule``
    hook, which fires before each batch's forward pass.
    """

    def __init__(self, optimizer: Any, budget_s: float, eta_min: float = 0.0) -> None:
        super().__init__(optimizer, eta_min)
        if not budget_s > 0 or budget_s == float("inf"):
            raise ValueError(f"budget_s must be finite and positive; got {budget_s}")
        self.budget_s = budget_s

    def __call__(self, elapsed_s: float) -> None:
        self.set_progress(elapsed_s / self.budget_s)
