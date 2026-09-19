"""The backprop vectors a curvature reference is built from: the *source* axis of the campaign.

A reference curvature matrix is ``F = U^T U`` where each row of ``U`` is the parameter gradient of
one scalar ``<v, f(x_n)>``. This module produces the ``v``: the columns of a **root** of the
output-space matrix ``Lambda_n = d^2 l_n / d f_n^2``, one set of columns per source.

* ``type2``  the closed-form root of the true Fisher's output factor. For softmax plus cross
  entropy ``Lambda_n = diag(p_n) - p_n p_n^T``, and ``S_n[:, c] = sqrt(p_c) (e_c - p_n)`` satisfies
  ``S_n S_n^T = Lambda_n`` exactly (``papers/kfac_from_scratch_2507.05127.pdf``, the test-case
  section). It gives ``C`` columns for a rank of ``C - 1``: the null vector is ``sqrt(p_n)``.
* ``mc``  ``K`` labels drawn from ``p_n``, column ``j`` being ``K^{-1/2} (p_n - e_{y_j})``. The
  ``K^{-1/2}`` is that paper's own errata item; without it a Monte-Carlo K-FAC does not converge to
  the generalised Gauss-Newton matrix. The estimator is unbiased: averaging over draws gives
  ``diag(p) - p p^T``.
* ``empirical``  the single vector ``p_n - e_{y_n}``, i.e. ``d l_n / d f_n`` at the true label,
  which gives the empirical Fisher rather than the Fisher.

Two conventions this module fixes, because getting either wrong is invisible until a metric is
compared across models.

1. **``l_n`` is the per-sample loss, never the batch mean.** Every benchmark's ``loss_fn`` is a
   ``reduction="mean"`` criterion, so it is *not* the object differentiated here and is
   deliberately never called. For ``nn.MSELoss`` the per-sample loss consistent with that mean is
   ``l_n = (1/D) sum_d (f_d - y_d)^2``, hence ``Lambda_n = (2/D) I``: the loss's own ``1/D`` is
   inside ``Lambda``. There ``F`` is the generalised Gauss-Newton matrix by the canonical link
   (Martens, arXiv:1412.1193 §9).
2. **The columns are unscaled.** The ``N^{-1/2}`` that turns the stacked rows into an average is
   applied by the driver, which is the only place that knows the *total* probe count; a function
   that sees one micro-batch cannot apply it.

Public API
----------

:data:`SOURCES`, :data:`LOSSES`  the three sources and the two loss kinds.

:class:`OutputRoot`  the ``(N, R, d_out)`` columns, with ``column(i)`` returning the ``(N, d_out)``
backprop vector of column ``i``.

:func:`output_root`  build the root columns for one batch. ``generator`` seeds the Monte-Carlo
draws, so a caller sweeping seeds gets reproducible draws without touching the global RNG. **Two
traversals that must see the same draws have to be handed the same generator state**, not just the
same generator object, because a generator advances as it is used.

:func:`loss_kind`  reads the loss kind off a benchmark's criterion rather than from a second
model-to-loss table that could drift from the harness's own.

Dependencies: none from the rest of ``fisher_ref``. :mod:`fisher_ref.capture` imports this module
lazily, to avoid an import cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

SOURCES: Tuple[str, ...] = ("type2", "mc", "empirical")
LOSSES: Tuple[str, ...] = ("cross_entropy", "mse")


@dataclass(frozen=True)
class OutputRoot:
    """Columns ``(N, R, d_out)`` of a root of ``Lambda_n``, one row block of ``U`` per column."""

    columns: Tensor
    source: str
    loss: str

    @property
    def n_columns(self) -> int:
        return int(self.columns.shape[1])

    def column(self, index: int) -> Tensor:
        """The ``(N, d_out)`` backprop vector of column ``index``."""
        return self.columns[:, index]

    def metadata(self) -> dict:
        return {"source": self.source, "loss": self.loss, "n_columns": self.n_columns}


def loss_kind(bench: Any) -> str:
    """Which ``LOSSES`` entry a benchmark's criterion is.

    Read off ``bench.loss_fn`` rather than from a second model -> loss table in ``fisher_ref``: the
    drift between two such tables is exactly what ``plan_exp_step1.md`` D4 removed.
    """
    criterion = getattr(bench, "loss_fn", None)
    if isinstance(criterion, nn.CrossEntropyLoss):
        return "cross_entropy"
    if isinstance(criterion, nn.MSELoss):
        return "mse"
    raise NotImplementedError(
        f"no output-space root is defined for {type(criterion).__name__}; the campaign's models use "
        f"nn.CrossEntropyLoss or nn.MSELoss (plan_exp_draft.md §2.1)"
    )


def _cross_entropy_root(outputs: Tensor, targets: Tensor, source: str, k: int,
                        generator: Optional[torch.Generator]) -> Tensor:
    probabilities = torch.softmax(outputs, dim=-1)
    n, classes = probabilities.shape
    if source == "type2":
        # S[n, c, :] = sqrt(p_c) (e_c - p_n); S S^T = diag(p) - p p^T, rank C - 1.
        identity = torch.eye(classes, dtype=outputs.dtype, device=outputs.device)
        return probabilities.sqrt().unsqueeze(-1) * (identity - probabilities.unsqueeze(1))
    if source == "mc":
        sampled = torch.multinomial(probabilities, k, replacement=True, generator=generator)
        one_hot = F.one_hot(sampled, num_classes=classes).to(outputs.dtype)
        return (probabilities.unsqueeze(1) - one_hot) * (1.0 / math.sqrt(k))
    targets = targets.reshape(n)
    one_hot = F.one_hot(targets.long(), num_classes=classes).to(outputs.dtype)
    return (probabilities - one_hot).unsqueeze(1)


def _mse_root(outputs: Tensor, targets: Tensor, source: str) -> Tensor:
    flat = outputs.reshape(outputs.shape[0], -1)
    dimension = flat.shape[1]
    if source == "mc":
        raise NotImplementedError(
            "the MC source is defined for the categorical models only (plan_exp_draft.md §0.11: it "
            "tests HF6); for MSE, Lambda = (2/D) I is already exact and type2 is the root"
        )
    if source == "type2":
        # Lambda = (2/D) I, so the root is sqrt(2/D) e_c, the same for every sample: an expanded
        # view, never a materialised (N, D, D) tensor.
        root = math.sqrt(2.0 / dimension) * torch.eye(
            dimension, dtype=outputs.dtype, device=outputs.device
        )
        return root.expand(flat.shape[0], dimension, dimension)
    residual = flat - targets.reshape(flat.shape).to(flat.dtype)
    return ((2.0 / dimension) * residual).unsqueeze(1)


def output_root(outputs: Tensor, targets: Tensor, *, source: str, loss: str = "cross_entropy",
                k: int = 1, generator: Optional[torch.Generator] = None) -> OutputRoot:
    """Root columns of ``Lambda_n`` for every sample of a batch.

    ``outputs`` are the model's own outputs (logits for ``cross_entropy``), ``targets`` the batch's
    labels (or reconstruction targets for ``mse``). ``k`` is the number of Monte-Carlo samples and
    is ignored by the other two sources; ``generator`` seeds them, so a caller sweeping seeds gets
    reproducible draws without touching the global RNG.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss!r}; expected one of {LOSSES}")
    if source == "mc" and k < 1:
        raise ValueError(f"the MC source needs k >= 1 samples; got {k}")
    if loss == "cross_entropy":
        if outputs.ndim != 2:
            raise NotImplementedError(
                f"cross-entropy roots expect (N, C) logits; got {tuple(outputs.shape)}"
            )
        columns = _cross_entropy_root(outputs, targets, source, k, generator)
    else:
        columns = _mse_root(outputs, targets, source)
    return OutputRoot(columns=columns, source=source, loss=loss)
