"""M5 — the fraction of the optimal quadratic decrease obtained along the preconditioned direction
(``plan_exp_draft.md`` §5).

    rho(K) = (g^T d)^2 / ((d^T R_lam d) (g^T R_lam^{-1} g)),   d = K_lam^{-1} g,   rho in [0, 1]

``rho = 1`` iff ``d`` is parallel to the exact natural-gradient direction ``R_lam^{-1} g``. It is
the metric that says whether an approximation is *useful for a step*, rather than close in norm —
and the one `plan_exp_lot1.md` §3.6 names as the cheapest decisive measurement on the
`mnist_autoencoder` stall, where the four Kronecker modes freeze while `adam` descends.

``g`` is the true gradient at that ``theta`` on the probes, so it is a property of the checkpoint
rather than of the metric; :func:`probe_gradient` builds it from the same probe set the references
use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

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
                   names: Any = None) -> Tensor:
    """The mean-loss gradient at this ``theta``, flattened in ``named_parameters()`` order.

    Computed under :func:`~fisher_ref.conventions.reference_mode`, i.e. the same eval-mode,
    dropout-free forward every reference uses, so ``g`` and ``R`` are statements about the same
    function.
    """
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


def rho(R: Tensor, K: BlockOps, gradient: Tensor, lam: float) -> NgdReport:
    """M5. ``R_lam^{-1} g`` is solved once by Cholesky and reused for both denominator terms."""
    size = R.shape[0]
    identity = torch.eye(size, dtype=R.dtype, device=R.device)
    factor = torch.linalg.cholesky(R + lam * identity)
    exact_direction = torch.cholesky_solve(gradient.reshape(-1, 1), factor).reshape(-1)

    direction = K.solve(gradient, lam)
    numerator = float(gradient @ direction) ** 2
    quadratic = float(direction @ ((R @ direction) + lam * direction))
    optimal = float(gradient @ exact_direction)
    value = numerator / (quadratic * optimal) if quadratic > 0 and optimal > 0 else float("nan")

    norms = float(direction.norm() * exact_direction.norm())
    cos = float(direction @ exact_direction) / norms if norms > 0 else float("nan")
    return NgdReport(rho=value, lam=lam, cos_directions=cos)


__all__ = ["NgdReport", "probe_gradient", "rho"]
