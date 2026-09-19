"""Cosine learning-rate annealing driven by an explicit progress fraction in ``[0, 1]``.

Two classes, one shared rule: the learning rate is
``eta_min + (base_lr - eta_min) * (1 + cos(pi * p)) / 2``, with ``p`` clamped to ``[0, 1]``.
``base_lr`` is whatever each parameter group held when the schedule was built.

:class:`NominalCosine` takes ``p = completed_epochs / t_max`` and exposes ``step()``, so it is a
drop-in for ``torch.optim.lr_scheduler.CosineAnnealingLR`` in a loop that steps once per completed
epoch. :class:`BudgetCosine` takes ``p = elapsed_s / budget_s`` and exposes
``__call__(elapsed_s)``, the signature of ``train_under_budget``'s per-batch ``lr_schedule`` hook.

Why both exist
--------------

AdaFisher Appendix D (``papers/adafisher_2405.16397.pdf``) specifies "a cosine annealing learning
rate decay strategy [...] aligning with the number of training epochs specified for each
optimizer". Under the wall-clock-time protocol the arms of one model complete *different* epoch
counts inside the same budget, so that sentence has two readings, and the shared-``T_max`` reading
biases the comparison at both ends. Both biases were measured on this project's first campaign:

* A **cheap** arm overshoots the shared epoch count. ``CosineAnnealingLR`` is periodic, so past
  ``T_max`` its learning rate climbs back towards the base value. Every overshooting arm was
  ``adam`` or ``adamw``, on 7 models out of 8. ``resnet20_cifar/adam`` reached ``lr = 0`` at epoch
  49, then trained 9 further epochs with the learning rate rising back to ``6.2e-5``, and its best
  validation accuracy decayed from 88.82% to 88.06%.
* An **expensive** arm stops short and never reaches the floor. ``resnet50_cifar/ekfac`` stopped
  at epoch 38 of 50 with ``lr`` still at ``1.6e-4``. Across the five Fisher arms,
  ``corr(epochs completed, best val acc)`` was +0.98 on ``resnet20_cifar``, +0.92 on
  ``resnet50_cifar`` and +0.92 on ``vit_small_cifar`` — on the long models the ranking of the
  modes was largely a ranking of how far each got through the shared schedule.

:class:`NominalCosine` removes the first bias by clamping: past ``t_max`` the learning rate holds
at ``eta_min`` instead of rising. Inside ``t_max`` it is the same closed-form cosine
``CosineAnnealingLR`` computes, so an arm that never overshoots is unchanged up to floating-point
rounding (the two spellings of the formula differ in the last one or two bits of a float64, which
is enough that a long run is statistically, not bit-for-bit, reproducible against the old one).

:class:`BudgetCosine` removes the second by annealing over the arm's own budget, so every budgeted
arm completes exactly one full cosine and the arms are compared at the same point of their own
schedule. The price is that the learning rate now depends on measured elapsed time, so such a run
is no longer bit-reproducible. The reference arm is unaffected: it is unbudgeted by construction —
it is what *defines* the budget — so ``runner.py`` always drives it with :class:`NominalCosine`.

One interaction to know about: :class:`BudgetCosine` assumes the arm actually spends its budget.
``runner.py``'s ``--max-epoch-factor`` caps how many epochs a budgeted arm may run, and if that cap
binds first the arm stops part-way up its cosine, under-annealed — the very thing this class exists
to prevent. ``run_arm`` prints a warning naming the unspent budget and the learning rate it stopped
at. No arm of the first campaign came close: the cheapest, ``resnet20_cifar/adam``, used 59 of its
150 allowed epochs.
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
