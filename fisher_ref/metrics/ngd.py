"""The fraction of the optimal quadratic decrease obtained along the preconditioned direction.

    rho(K) = (g^T d)^2 / ((d^T R_lam d) (g^T R_lam^{-1} g)),   d = K_lam^{-1} g

with ``R_lam = R + lam I``. By Cauchy-Schwarz in the ``R_lam`` inner product, ``rho`` lies in
``[0, 1]``, and it is one exactly when ``d`` is parallel to the exact natural-gradient direction
``R_lam^{-1} g``. It is the metric that says whether an approximation is *useful for a step*,
rather than close in norm.

``g`` is the true gradient at those weights on the probes, so it is a property of the checkpoint
rather than of the metric; :func:`probe_gradient` builds it from the same probe set the references
use.

Public API
----------

:class:`NgdReport`  ``rho``, the damping it was evaluated at, and the cosine between the two
directions.

:func:`probe_gradient`  the mean-loss gradient at these weights, flattened in
``named_parameters()`` order, computed in eval mode so that the gradient and the reference are
statements about the same function. ``batch_size`` micro-batches the forward and weights each
batch's mean-loss gradient by its share of the probes, which is exact for a ``reduction="mean"``
criterion; without it a single forward over tens of thousands of images is tens of gigabytes of
float64 activations.

:func:`damped_cholesky`  the Cholesky factor of ``R + lam I`` without materialising the identity.
Adding ``lam`` to the diagonal of a clone is bit-identical, because every off-diagonal entry of
``lam I`` is exactly zero.

:func:`rho`  the metric. ``R_lam^{-1} g`` is solved once by Cholesky and reused for both
denominator terms, and ``factor`` lets a caller sweeping several structures at one damping
re-use the same factorisation.

``k_lam`` is the damping applied to ``K`` when it is not the reference's. ``None`` (the default)
means the same value, which is what a structural comparison wants. An operational operator passes
``0.0``: it already carries the optimizer's own damping, and for the two factored-inverse modes
that damping is *factored* Tikhonov (one shift on each Kronecker factor), which is not ``K + lam I``
for any ``lam``. The reference stays damped either way -- it is the yardstick, and an exact Fisher
on a real network has thousands of exactly-zero directions, so it has to be.

Dependencies: :mod:`fisher_ref.approx.base`, :mod:`fisher_ref.conventions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..approx.base import BlockOps
from ..conventions import reference_mode


@dataclass(frozen=True)
class NgdReport:
    rho: float
    lam: float
    cos_directions: float

    def as_rows(self, **keys: Any) -> list:
        return [{**keys, "metric": name, "value": value} for name, value in
                (("rho", self.rho), ("cos_ngd", self.cos_directions))]


def probe_gradient(model: nn.Module, inputs: Tensor, targets: Tensor,
                   loss_fn: Callable[[Tensor, Tensor], Tensor],
                   names: Any = None, batch_size: Optional[int] = None,
                   device: Any = None, dtype: Optional[torch.dtype] = None) -> Tensor:
    """The mean-loss gradient at this ``theta``, flattened in ``named_parameters()`` order.

    Computed under :func:`~fisher_ref.conventions.reference_mode`, i.e. the same eval-mode,
    dropout-free forward every reference uses, so ``g`` and ``R`` are statements about the same
    function.

    ``batch_size`` micro-batches the forward, weighting each batch's mean-loss gradient by
    ``n_b / N`` — exact for a ``reduction="mean"`` criterion. Lot 2's single forward over all probes
    fitted an MLP on MNIST; on 45 000 CIFAR images through a CNN it is ~90 GB of fp64 activations
    (``plan_exp_lot3.md`` §0.9). ``device``/``dtype`` move each micro-batch, never the whole set.
    """
    if batch_size is not None:
        return _batched_probe_gradient(model, inputs, targets, loss_fn, names, batch_size,
                                       device, dtype)
    model.zero_grad(set_to_none=True)
    with reference_mode(model):
        loss = loss_fn(model(inputs), targets)
    loss.backward()
    wanted = set(names) if names is not None else None
    parts = [parameter.grad.reshape(-1) for name, parameter in model.named_parameters()
             if wanted is None or name in wanted]
    gradient = torch.cat(parts).detach().clone()
    model.zero_grad(set_to_none=True)
    return gradient


def _batched_probe_gradient(model: nn.Module, inputs: Tensor, targets: Tensor,
                            loss_fn: Callable[[Tensor, Tensor], Tensor], names: Any,
                            batch_size: int, device: Any, dtype: Optional[torch.dtype]) -> Tensor:
    wanted = set(names) if names is not None else None
    selected = [(name, p) for name, p in model.named_parameters()
                if wanted is None or name in wanted]
    total = int(inputs.shape[0])
    out: Optional[Tensor] = None
    with reference_mode(model):
        for start in range(0, total, batch_size):
            batch = inputs[start:start + batch_size]
            labels = targets[start:start + batch_size]
            if device is not None or dtype is not None:
                batch = batch.to(device=device, dtype=dtype) if dtype else batch.to(device)
                labels = labels.to(device)
            loss = loss_fn(model(batch), labels)
            grads = torch.autograd.grad(loss, [p for _, p in selected], allow_unused=True)
            flat = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                              for (_, p), g in zip(selected, grads)]).detach()
            weighted = flat * (float(batch.shape[0]) / total)
            out = weighted if out is None else out + weighted
    assert out is not None
    return out


def damped_cholesky(R: Tensor, lam: float) -> Tensor:
    """The Cholesky factor of ``R + lam I``, without materialising the identity.

    ``R + lam * torch.eye(P)`` allocates a whole second ``P x P`` on top of the sum it returns; at
    A1's first layer (``P = 25 120``, fp64) that is 5.05 GB of pure waste. Adding ``lam`` to the
    diagonal of a clone is **bit-identical** — every off-diagonal entry of ``lam I`` is exactly
    ``0.0``, and ``x + 0.0 == x`` for every finite ``x``.
    """
    damped = R.clone()
    damped.diagonal().add_(lam)
    return torch.linalg.cholesky(damped)


def rho(R: Tensor, K: BlockOps, gradient: Tensor, lam: float, *,
        factor: Optional[Tensor] = None, k_lam: Optional[float] = None) -> NgdReport:
    """M5. ``R_lam^{-1} g`` is solved once by Cholesky and reused for both denominator terms.

    ``factor`` is that Cholesky, when the caller already holds it. It depends on ``(R, lam)``
    alone, so a caller sweeping the structures at a fixed ``lam`` otherwise re-factorises the same
    matrix once per structure — six times over at A1's first layer, where one factorisation is
    ``P^3/3 = 5.3e12`` flops and 5.05 GB.

    ``k_lam`` is the damping applied to ``K``, when it is not the reference's. ``None`` (the
    default) means ``k_lam = lam``, which is what every P1 structure wants and is bit-identical to
    this function before lot 5. **P2 passes ``0.0``**: the operational preconditioner already carries
    the optimizer's own damping, and for ``kfac``/``tkfac`` that damping is *factored* Tikhonov
    (``A + pi*sqrt(lam) I`` on one side, ``B + sqrt(lam)/pi I`` on the other), which is not
    ``K + lam I`` for any ``lam`` (``plan_exp_lot5.md`` §0.3). The reference stays damped at the
    sweep's ``lam`` either way — it is the yardstick, and ``plan_exp_lot1.md`` §6.4 measured ``F``'s
    kernel at 4 805 of A1's 26 634 directions, so it has to be.
    """
    if factor is None:
        factor = damped_cholesky(R, lam)
    exact_direction = torch.cholesky_solve(gradient.reshape(-1, 1), factor).reshape(-1)

    direction = K.solve(gradient, lam if k_lam is None else k_lam)
    numerator = float(gradient @ direction) ** 2
    quadratic = float(direction @ ((R @ direction) + lam * direction))
    optimal = float(gradient @ exact_direction)
    value = numerator / (quadratic * optimal) if quadratic > 0 and optimal > 0 else float("nan")

    norms = float(direction.norm() * exact_direction.norm())
    cos = float(direction @ exact_direction) / norms if norms > 0 else float("nan")
    return NgdReport(rho=value, lam=lam, cos_directions=cos)


__all__ = ["NgdReport", "damped_cholesky", "probe_gradient", "rho"]
