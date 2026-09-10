"""The single training loop (``plan_exp_step1.md`` D2), the merge of lot 7's
``train_under_time_budget`` and lot 8's ``train_classifier_under_budget``.

Semantics preserved exactly from both:

- the budget is checked **before** each batch is processed but **after** it has been fetched, so a
  run overshoots ``budget_s`` by at most one batch's own processing duration (``plan_lot7.md``
  §0.1);
- ``eval_fn``'s duration is subtracted from the clock — validation is not the optimizer's cost
  (``plan_lot8.md`` §0.7);
- one ``scheduler.step()`` per **completed** epoch;
- ``max_steps`` is a defensive bound independent of the clock (``plan_lot7.md`` §0.11);
- lot 7's "re-iterate the loader forever" behaviour is ``max_epochs=10**9``, the default.

Dataset- and model-agnostic on purpose (``plan_lot7.md`` §0.8): the two task shapes differ only by
``prepare_batch`` and ``metric_fn``, so ``tests/`` drives this loop on synthetic tensors.
"""

from __future__ import annotations

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
    """No-op on CPU; correctness guard on CUDA (``plan_lot7.md`` §0.2) — kernel launches are
    asynchronous, so an un-synchronized ``perf_counter()`` bracket would time launch overhead only.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
) -> Tuple[List[StepRecord], List[EpochRecord]]:
    """Train until ``budget_s`` is spent, ``max_epochs`` epochs complete or ``max_steps`` steps run.

    ``on_step(completed_steps, epoch, model)`` is the loop's only extension point: it fires once
    with ``completed_steps=0`` before the first batch (the ``t=0`` state) and then after every
    optimizer step, with the number of steps completed so far. ``common/checkpoints.py`` is its
    only user today.
    """
    step_records: List[StepRecord] = []
    epoch_records: List[EpochRecord] = []
    batches_per_epoch = len(loader)
    if batches_per_epoch == 0:
        raise ValueError("loader must yield at least one batch")

    step = 0
    excluded_s = 0.0
    t0 = perf_counter()

    def elapsed() -> float:
        return perf_counter() - t0 - excluded_s

    if on_step is not None:
        on_step(0, 0, model)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss, epoch_batches, completed = 0.0, 0, True
        for batch_idx, batch in enumerate(loader):
            if elapsed() >= budget_s or step >= max_steps:
                completed = False
                break
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

            loss_value = float(loss.item())
            epoch_loss += loss_value
            epoch_batches += 1
            step_records.append(
                StepRecord(
                    step=step,
                    epoch=epoch + batch_idx / batches_per_epoch,
                    elapsed_s=elapsed(),
                    loss=loss_value,
                    fwd_bwd_s=fwd_bwd_s,
                    step_s=step_s,
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
            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_records.append(
                EpochRecord(
                    epoch=epoch,
                    steps=step,
                    elapsed_s=elapsed(),
                    train_loss=epoch_loss / epoch_batches,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    lr=current_lr,
                )
            )
            log_fn(
                f"  epoch {epoch:3d} | steps {step:6d} | {elapsed():8.1f}s | "
                f"train {epoch_loss / epoch_batches:.4f} | val_acc {val_acc * 100:5.2f}% | "
                f"lr {current_lr:.2e}"
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
