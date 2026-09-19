"""The single training loop shared by every bench, plus its evaluation helper.

One function, ``train_under_budget``, trains until whichever comes first: a wall-clock budget is
spent, an epoch cap is reached, or a step cap is reached. It is deliberately dataset- and
model-agnostic — a task is described to it by two callables, ``prepare_batch`` (how to turn a
loader item into ``(inputs, targets)``) and, for evaluation, ``metric_fn`` — so the same loop
drives an MNIST auto-encoder, a CIFAR classifier and the synthetic tensors the unit tests use.

What the clock measures
-----------------------

The budget is checked once per batch, after the batch has been fetched and before it is
processed. A run therefore overshoots its budget by at most one batch's own processing time,
never more.

Charged to the budget: fetching a batch from the loader (including the cost of restarting the
worker processes at the top of every epoch), the forward and backward passes, the optimizer step,
and anything ``on_step`` does, such as writing a checkpoint.

Not charged: ``eval_fn``. Its duration is measured and subtracted, because validation is a
measurement of the run, not part of the optimizer's cost.

``sync`` is called after the backward pass and after the optimizer step so that the recorded
``fwd_bwd_s`` and ``step_s`` are real elapsed times on CUDA, where kernel launches are
asynchronous and an unsynchronized timer would measure launch overhead only.

The part of the budget that is *not* the optimizer is recorded too, as ``data_s``: the time from
the end of the previous optimizer step to the start of this step's forward pass. It is the batch
fetch, the host-to-device copy and whatever ``on_step`` did after the previous step. Together the
three account for essentially the whole clock — ``sum(data_s + fwd_bwd_s + step_s)`` was measured
at 99.75-99.80% of the last record's ``elapsed_s`` over 100 toy runs, the missing 0.2% being the
per-step bookkeeping that sits between the optimizer step and the record itself (``loss.item()``
and building the record), which no timer here brackets. This matters because the non-compute share
is paid once per *epoch*, while the arms
of one model deliberately complete different epoch counts inside the same budget — so a cheap arm
pays it more often and spends a smaller fraction of an "equal" budget on optimization. Measured on
``mnist_autoencoder``: 47.4% of the budget is compute for ``adam`` against 78.5% for ``ekfac``, so
``ekfac`` gets 1.67x the optimization time. On ``resnet50_cifar`` it is 99.5-99.6% for all seven
arms. ``data_s`` makes that share visible per run instead of leaving it to be reconstructed.

Extension points
----------------

``on_step(completed_steps, epoch, model)`` fires once with ``completed_steps=0`` before the first
batch — the state at initialization — and then after every optimizer step.
``benchmarks/common/checkpoints.py`` is its only user.

``lr_schedule(elapsed_s)`` is the wall-clock-driven alternative to a per-epoch ``scheduler``: it
fires before each batch's forward pass, so the learning rate can follow the budget actually
consumed rather than a shared nominal epoch count. ``benchmarks/common/runner.py`` passes one or
the other, never both. A ``scheduler``, when given, is stepped once per *completed* epoch; an
epoch cut short by the budget does not step it.

Records produced
----------------

``(step_records, epoch_records)``, the dataclasses of ``benchmarks/common/records.py``. Every
step record carries its own elapsed time, loss and the three timings above; every epoch record
carries the epoch's mean training loss, the validation numbers and the learning rate.

``EpochRecord.lr`` is **the rate the epoch actually ran at** — read off the optimizer when that
epoch's first step was taken, before any scheduler step. One rule, whichever schedule is in force:
a per-epoch ``scheduler`` holds the rate constant across the epoch, so the recorded number is the
rate every batch of it used; a per-batch ``lr_schedule`` moves the rate within the epoch, so the
recorded number is the rate the epoch *started* at. Reading it after ``scheduler.step()`` instead
— which is what this loop used to do — recorded the rate the *next* epoch would use, and a
budget-truncated final epoch (which never steps the scheduler) then meant a third thing again.
The consequence of the fix: under a cosine annealed over ``t_max`` epochs, the last recorded rate
is the one the last epoch ran at, not the floor the schedule reaches once that epoch is over.
"""

from __future__ import annotations

import gc
from time import perf_counter
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .records import EpochRecord, StepRecord

PrepareBatch = Callable[[Any], Tuple[Tensor, Tensor]]
OnStep = Callable[[int, int, nn.Module], None]


def sync(device: torch.device) -> None:
    """No-op on CPU; correctness guard on CUDA. Kernel launches are asynchronous, so an
    un-synchronized ``perf_counter()`` bracket would time the launch, not the work.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device) -> Optional[int]:
    """High-water mark of allocated device memory since the last ``reset_peak_memory``, or
    ``None`` where PyTorch exposes no such counter (CPU; MPS, which reports current allocation
    only). This is *allocator* memory, not the process's whole VRAM footprint — the CUDA context
    and cuDNN workspaces sit on top, so it under-reports what ``nvidia-smi`` shows.
    """
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


def release_optimizer(model: nn.Module, optimizer: Any) -> int:
    """Unregister every hook ``optimizer`` put on ``model``, and return how many were removed.

    A hook-based optimizer is a reference cycle: the optimizer holds the module, the module's hook
    dictionary holds a bound method of the optimizer. Plain reference counting can never free
    either, so an arm's model, its optimizer and every curvature factor in it stay alive after the
    function that built them has returned — until some later, unrelated garbage collection. The
    next arm then resets its peak-memory counter on a baseline that still contains its
    predecessors, which is why the reported ``peak VRAM`` column used to be strictly monotone in
    the order the arms ran, and why ``adam`` was reported as needing more memory than every Fisher
    mode.

    Only the optimizer's *own* hooks are removed: an entry is deleted when the callable it holds
    is a bound method of this optimizer. A hook registered by anything else — a model's own, or a
    test's spy — is left alone. Nothing about the model's weights is touched, so this changes no
    number a run produced; it changes only what is still resident afterwards.
    """
    removed = 0
    for module in model.modules():
        for name in ("_forward_pre_hooks", "_forward_hooks",
                     "_backward_hooks", "_backward_pre_hooks", "_full_backward_hooks"):
            hooks = getattr(module, name, None)
            if not hooks:
                continue
            for key in [k for k, fn in hooks.items()
                        if getattr(fn, "__self__", None) is optimizer]:
                del hooks[key]
                removed += 1
    gc.collect()  # the cycle is broken, but anything already in a generation still needs a sweep
    return removed


def default_prepare_batch(batch: Any) -> Tuple[Tensor, Tensor]:
    """Supervised convention: ``(inputs, targets)`` straight off the loader."""
    return batch[0], batch[1]


def flatten_autoencoder_batch(batch: Any) -> Tuple[Tensor, Tensor]:
    """Reconstruction convention: the flattened image is both input and target."""
    x = batch[0].flatten(1)
    return x, x


def top1(outputs: Tensor, targets: Tensor) -> int:
    """Number of correct top-1 predictions in a batch (a *count*, so ``evaluate`` can accumulate
    it across batches of unequal size).
    """
    return int((outputs.argmax(dim=1) == targets).sum().item())


def train_under_budget(
    model: nn.Module,
    optimizer: Any,
    loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    budget_s: float,
    max_epochs: int = 10**9,
    device: torch.device = torch.device("cpu"),
    scheduler: Any = None,
    eval_fn: Optional[Callable[[], Tuple[float, float]]] = None,
    on_step: Optional[OnStep] = None,
    max_steps: int = 10**9,
    prepare_batch: PrepareBatch = default_prepare_batch,
    log_fn: Callable[[str], None] = print,
    lr_schedule: Optional[Callable[[float], None]] = None,
) -> Tuple[List[StepRecord], List[EpochRecord]]:
    """Train until ``budget_s`` is spent, ``max_epochs`` epochs complete or ``max_steps`` steps run.

    ``on_step(completed_steps, epoch, model)`` is the loop's only extension point: it fires once
    with ``completed_steps=0`` before the first batch (the state at initialization) and then after
    every optimizer step, with the number of steps completed so far.
    ``benchmarks/common/checkpoints.py`` is its only user today.

    ``lr_schedule(elapsed_s)`` is the wall-clock-driven alternative to ``scheduler``
    (``benchmarks/common/schedules.py::BudgetCosine``): it fires before each batch's forward pass,
    so the schedule can follow the budget actually consumed rather than a shared nominal epoch
    count. The two are mutually exclusive by construction — ``runner.py`` passes one or the other.
    """
    step_records: List[StepRecord] = []
    epoch_records: List[EpochRecord] = []
    batches_per_epoch = len(loader)
    if batches_per_epoch == 0:
        raise ValueError("loader must yield at least one batch")

    step = 0
    excluded_s = 0.0
    t0 = perf_counter()
    # Elapsed time at the end of the previous optimizer step, i.e. where this step's ``data_s``
    # starts counting. Zero before the first step, so ``data_s`` of step 0 covers whatever ran
    # between ``t0`` and that first forward pass — including the ``on_step(0, ...)`` dump.
    previous_end_s = 0.0

    def elapsed() -> float:
        return perf_counter() - t0 - excluded_s

    if on_step is not None:
        on_step(0, 0, model)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss, epoch_batches, completed = 0.0, 0, True
        epoch_lr = float("nan")
        for batch_idx, batch in enumerate(loader):
            if elapsed() >= budget_s or step >= max_steps:
                completed = False
                break
            if lr_schedule is not None:
                # Before the forward pass, so this batch is taken with the LR its own position in
                # the budget prescribes. Excluded from the fwd+bwd and step timings below.
                lr_schedule(elapsed())
            if epoch_batches == 0:
                # The rate this epoch's first step runs at, read before any scheduler step — that
                # is what EpochRecord.lr means. See this module's docstring.
                epoch_lr = float(optimizer.param_groups[0]["lr"])
            inputs, targets = prepare_batch(batch)
            inputs, targets = inputs.to(device), targets.to(device)

            t_fb = perf_counter()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            sync(device)
            fwd_bwd_s = perf_counter() - t_fb

            t_step = perf_counter()
            optimizer.step()
            sync(device)
            step_s = perf_counter() - t_step

            # Everything between the previous step and this forward pass: the loader fetch, the
            # host-to-device copy, and whatever ``on_step`` did last time. Derived from timestamps
            # this loop already takes — it costs no extra call to the clock, which is why adding it
            # does not perturb the budget it measures. ``t_fb`` is converted to the same excluded-
            # eval-time scale ``elapsed()`` uses; ``excluded_s`` cannot have moved since ``t_fb``,
            # because evaluation only runs between epochs.
            data_s = (t_fb - t0 - excluded_s) - previous_end_s

            loss_value = float(loss.item())
            epoch_loss += loss_value
            epoch_batches += 1
            elapsed_s = elapsed()
            previous_end_s = elapsed_s
            step_records.append(
                StepRecord(
                    step=step,
                    epoch=epoch + batch_idx / batches_per_epoch,
                    elapsed_s=elapsed_s,
                    loss=loss_value,
                    fwd_bwd_s=fwd_bwd_s,
                    step_s=step_s,
                    data_s=data_s,
                )
            )
            step += 1
            if on_step is not None:
                on_step(step, epoch, model)

        if completed and scheduler is not None:
            scheduler.step()

        val_loss, val_acc = float("nan"), float("nan")
        if eval_fn is not None and epoch_batches > 0:
            t_eval = perf_counter()
            val_loss, val_acc = eval_fn()
            excluded_s += perf_counter() - t_eval

        if epoch_batches > 0:
            epoch_records.append(
                EpochRecord(
                    epoch=epoch,
                    steps=step,
                    elapsed_s=elapsed(),
                    train_loss=epoch_loss / epoch_batches,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    lr=epoch_lr,
                )
            )
            log_fn(
                f"  epoch {epoch:3d} | steps {step:6d} | {elapsed():8.1f}s | "
                f"train {epoch_loss / epoch_batches:.4f} | val_acc {val_acc * 100:5.2f}% | "
                f"lr {epoch_lr:.2e}"
            )

        if not completed or elapsed() >= budget_s or step >= max_steps:
            break

    return step_records, epoch_records


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    *,
    prepare_batch: PrepareBatch = default_prepare_batch,
    metric_fn: Optional[Callable[[Tensor, Tensor], float]] = None,
) -> Tuple[float, float]:
    """``(mean loss, mean metric)`` over ``loader``; the metric is ``nan`` when ``metric_fn`` is
    ``None`` (a reconstruction task has no accuracy). ``metric_fn`` returns a batch *total*, e.g.
    ``top1``'s correct count, so batches of unequal size average correctly.
    """
    was_training = model.training
    model.eval()
    total_loss, total_metric, seen = 0.0, 0.0, 0
    for batch in loader:
        inputs, targets = prepare_batch(batch)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        n = targets.size(0)
        total_loss += float(loss_fn(outputs, targets).item()) * n
        if metric_fn is not None:
            total_metric += float(metric_fn(outputs, targets))
        seen += n
    model.train(was_training)
    if seen == 0:
        return float("nan"), float("nan")
    return total_loss / seen, (total_metric / seen if metric_fn is not None else float("nan"))
