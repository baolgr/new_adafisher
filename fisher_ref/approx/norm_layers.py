"""Normalisation layers: the four readings of a ``(gamma, beta)`` block, and the optimizer's own
formulas called directly.

The exact per-sample gradients of a normalisation layer are ``d/d gamma = sum_t g_t * x_hat_t`` and
``d/d beta = sum_t g_t``, so the exact block is ``2C x 2C`` and is **always** affordable -- 64 x 64
for a ``LayerNorm(32)``. There is nothing to approximate for memory here; the question is whether
the approximations the optimizer actually makes are right.

Four readings, and none of them needs a new block class:

1. **exact joint** -- the ``2C x 2C`` block itself (a ``Dense``);
2. **exact separate** -- the same with the ``gamma``-``beta`` cross terms dropped, which is what
   AdaFisher does. The gap between 1 and 2 is what that choice costs;
3. **Hadamard**, AdaFisher's Proposition 3.1 as written
   (``papers/adafisher_2405.16397.pdf``): ``gamma`` gets ``H (.) S`` with
   ``H = (1/|T|) sum_t x_hat x_hat^T`` and ``S = (1/|T|) sum_t g g^T``, and ``beta`` gets ``S``;
4. **the formula as implemented**, i.e. what the shipped ``diag.py`` actually computes, which is a
   documented "sum-then-square" deviation from the Proposition's "square-then-sum". It is read by
   **calling** the optimizer's own ``compute_h_diag`` / ``compute_s_diag``, never re-derived.

With more than one position per example (``GroupNorm``, ``BatchNorm2d`` on feature maps,
``LayerNorm`` over tokens) the ``beta`` block is **not** exact under Proposition 3.1:
``d/d beta = sum_t g_t`` makes it ``(1/N) sum (sum_t g_t)(sum_t g_t)^T``, while ``S`` sums
``g_t g_t^T`` position by position. "Exact on beta" holds only at one position per example.

A caution this repository has paid for twice: a normalisation block is **Hadamard-, not
Kronecker-structured**, so none of these is a ``Kron``, and the dense layout's bias-augmented
permutation refuses such a layer outright.

Public API
----------

:class:`NormStats`  ``H`` and ``S``, with ``hadamard()``, ``hadamard_diagonal()`` and the older
alias ``as_implemented_diagonal()`` kept so earlier outputs stay reproducible. Since the exact
diagonal is the Frobenius-optimal diagonal, ``hadamard_diagonal`` can never beat it -- that
ordering is a theorem, not a finding.

:func:`exact_separate`, :func:`cross_term_share`  reading 2 and the size of what it drops.

:data:`DIAG_PY_TYPES`  the normalisation types the shipped ``diag.py`` has a formula for. A
``GroupNorm`` is not one of them, so ``AdaFisherMulti`` does not hook it and it takes the identity
preconditioner.

:data:`HOOKED_TYPES`  every module type ``AdaFisherMulti`` hooks.

:func:`diag_py_reading`  the optimizer's ``F~_D = S_D (x) H_D`` per hooked module, **as the
optimizer computes it**: the raw module input, the gradient of the *mean* loss over a training-size
micro-batch with the true labels, averaged over full micro-batches, with no running average, no
min-max renormalisation and no damping. The statistic is batch-size dependent, so the last,
incomplete micro-batch is dropped rather than weighted. Its overall scale is arbitrary (mean-loss
gradients carry a ``1/batch^2``), so it must be read through a direction metric or after optimal
rescaling.

:func:`diag_py_factor_reading`  the same formula with the micro-batches averaged **factor by
factor**, which is how the optimizer's running average actually combines them.
:func:`diag_py_reading` averages the *product*. The two differ by the covariance of the two factors
across micro-batches, and both are emitted so that difference is a measured number rather than a
choice argued for here.

:func:`kron_py_reading`  ``kron(compute_s_full, compute_h_full)`` per hooked module: the optimizer's
**own** Kronecker factor formulas, on the campaign's probes, with no averaging and no damping. It
exists because the campaign's own structures build no Kronecker structure on a normalisation layer
at all, so without this reading three of the five optimizer modes would have nothing to compare
against there. On a ``Linear`` or ``Conv2d`` the result is in the bias-augmented ``rvec([W | b])``
order; on a normalisation layer it is permuted into ``named_parameters()`` order.

Dependencies: :mod:`fisher_ref.conventions`, :mod:`fisher_ref.approx.base`,
:mod:`fisher_ref.approx.kfac`, :mod:`fisher_ref.approx.adafisher_state` (for the normalisation
permutation), and ``adafisher_modes.factors``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Mapping, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..conventions import reference_mode
from .base import BlockDiag, Dense, Diag

if TYPE_CHECKING:  # pragma: no cover
    from .base import BlockOps


@dataclass(frozen=True)
class NormStats:
    """Second moments of the normalised input and of the output gradient, per position."""

    H: Tensor        # (C, C), mean_t x_hat x_hat^T
    S: Tensor        # (C, C), mean_{n,c,t} g g^T
    C: int

    def hadamard(self) -> BlockDiag:
        """Proposition 3.1 as written: ``gamma -> H (.) S``, ``beta -> S``, cross terms zero."""
        return BlockDiag([("gamma", slice(0, self.C), Dense(self.H * self.S)),
                          ("beta", slice(self.C, 2 * self.C), Dense(self.S))], P=2 * self.C)

    def hadamard_diagonal(self) -> Diag:
        """``diag(hadamard)``: ``H_cc S_cc`` for ``gamma``, ``S_cc`` for ``beta``.

        This is what lot 2 emitted as ``as_implemented`` and described as "the diagonal ``diag.py``
        computes". It is not: ``diag.py`` sums the **raw** input over batch and positions before
        squaring and yields a 2-entry ``H_D`` shared by all channels (:func:`diag_py_reading`). And
        since ``diag R`` is the Frobenius-optimal diagonal, ``e_F`` of this reading can never beat
        ``exact_diag`` — lot 2 §6.3's ordering (c) > (d) was a theorem (``plan_exp_lot3.md`` §0.7).
        """
        return Diag(torch.cat([self.H.diagonal() * self.S.diagonal(), self.S.diagonal()]))

    def as_implemented_diagonal(self) -> Diag:
        """Lot 2's name for :meth:`hadamard_diagonal`, kept so its A1 outputs stay reproducible;
        new code uses the accurate name (``plan_exp_lot3.md`` §0.7)."""
        return self.hadamard_diagonal()


def exact_separate(block: Tensor, C: int) -> BlockDiag:
    """The exact ``2C x 2C`` block with the ``gamma``-``beta`` cross terms dropped.

    Its gap to the joint block is HF4, measured rather than argued: it is the only structure in the
    zoo whose error comes from *ignoring a coupling the exact object has*, not from a factorisation.
    """
    return BlockDiag([("gamma", slice(0, C), Dense(block[:C, :C])),
                      ("beta", slice(C, 2 * C), Dense(block[C:, C:]))], P=2 * C)


def cross_term_share(block: Tensor, C: int) -> Tuple[float, float]:
    """``(||cross||_F / ||block||_F, ||cross||_F / ||diagonal blocks||_F)`` — HF4 as one number.

    The first is the fraction of the block's Frobenius mass that "exact separate" throws away.
    """
    cross = block[:C, C:]
    total = float(torch.linalg.matrix_norm(block))
    diagonal = float((torch.linalg.matrix_norm(block[:C, :C]) ** 2
                      + torch.linalg.matrix_norm(block[C:, C:]) ** 2) ** 0.5)
    off = float(torch.linalg.matrix_norm(cross)) * (2.0 ** 0.5)   # both triangles
    return off / total, off / diagonal


#: The normalisation types ``diag.py`` has a formula for; any other (``GroupNorm``) is not hooked by
#: ``AdaFisherMulti`` and takes the identity preconditioner.
DIAG_PY_TYPES = (nn.BatchNorm2d, nn.LayerNorm)

#: Every module type ``AdaFisherMulti`` hooks. Lot 5's P1-py rung (``plan_exp_lot5.md`` §0.4) reads
#: the optimizer's own formulas on **all** of them, not just the normalisations lot 3 needed: it is
#: what gives ``p2_diag`` and ``p2_kfac`` a like-for-like P1 partner everywhere.
HOOKED_TYPES = (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)


def _hooked(modules: Mapping[str, nn.Module], types: Tuple[type, ...]) -> Dict[str, nn.Module]:
    """The modules of ``types`` carrying a weight, with the one guard that matters.

    A normalisation layer's reading is laid out ``[gamma; beta]``, which presupposes it *has* a
    ``beta``: ``augment_norm_input`` always returns a ``(T, 2)`` matrix, so a layer with
    ``weight`` but no ``bias`` would be given a bias column that corresponds to no parameter. Lot 3
    excluded such a layer by requiring both; that exclusion is kept, and made explicit.
    """
    out: Dict[str, nn.Module] = {}
    for name, module in modules.items():
        if not isinstance(module, types) or getattr(module, "weight", None) is None:
            continue
        if isinstance(module, DIAG_PY_TYPES) and getattr(module, "bias", None) is None:
            continue
        out[name] = module
    return out


def diag_py_reading(model: nn.Module, inputs: Tensor, targets: Tensor,
                    modules: Mapping[str, nn.Module], *,
                    loss_fn: Callable[[Tensor, Tensor], Tensor], batch_size: int,
                    dtype: torch.dtype = torch.float64, device: object = "cpu",
                    types: Tuple[type, ...] = DIAG_PY_TYPES) -> Dict[str, Diag]:
    """``diag.py``'s own ``F~_D = S_D (x) H_D`` for every hooked normalisation layer, **as the
    optimizer computes it** — the raw module input, the gradient of the *mean* loss over a
    training-size micro-batch with the true labels — averaged over full micro-batches, with P1's
    conventions (no EMA, no min-max, no damping; ``plan_exp_lot3.md`` §0.7).

    ``compute_h_diag``/``compute_s_diag`` are called, not re-implemented (``plan_exp_draft.md``
    §0.12), and the ``(C, 2)`` result is laid out ``[gamma_1..C, beta_1..C]`` as in
    ``named_parameters()``. The statistic is batch-size dependent ("sum-then-square"), so the last,
    incomplete micro-batch is dropped rather than weighted. Its overall scale is arbitrary (mean-loss
    gradients, ``1/B^2`` constants): read it through ``cos_F`` and ``c*``.

    ``types`` restricts which modules are read. The default is lot 3's — the two normalisation types
    — so that call is unchanged. Lot 5 passes :data:`HOOKED_TYPES` to get the same reading on
    ``Linear`` and ``Conv2d`` as well, where the result is laid out in the bias-augmented
    ``rvec([W | b])`` order every Kronecker structure of this campaign uses. That is the object
    ``p2_diag`` is the EMA'd, min-maxed, damped version of (``plan_exp_lot5.md`` §0.4).
    """
    from adafisher_modes.factors import compute_h_diag, compute_s_diag  # noqa: PLC0415

    wanted = _hooked(modules, types)
    if not wanted:
        return {}
    raw_inputs: Dict[str, Tensor] = {}
    raw_grads: Dict[str, Tensor] = {}

    def make_hook(name: str):
        def hook(module: nn.Module, args: Tuple, output: Tensor) -> None:
            raw_inputs[name] = args[0].detach()
            output.register_hook(lambda grad: raw_grads.__setitem__(name, grad.detach()))
        return hook

    sums: Dict[str, Tensor] = {}
    batches = 0
    handles = [module.register_forward_hook(make_hook(name)) for name, module in wanted.items()]
    try:
        with reference_mode(model):
            for start in range(0, int(inputs.shape[0]) - batch_size + 1, batch_size):
                batch = inputs[start:start + batch_size].to(device=device, dtype=dtype)
                batch.requires_grad_(True)
                labels = targets[start:start + batch_size].to(device)
                raw_inputs.clear()
                raw_grads.clear()
                loss = loss_fn(model(batch), labels)
                # inputs=[batch]: every hooked module's output is on the path to it, so no tensor
                # hook is pruned (plan_exp_lot1.md §0.1).
                torch.autograd.grad(loss, [batch])
                for name, module in wanted.items():
                    h_diag = compute_h_diag(raw_inputs[name], module)
                    s_diag = compute_s_diag(raw_grads[name], module)
                    # (d_out, d_in_aug); for a normalisation layer that is (C, 2) = [gamma, beta].
                    f_tilde = torch.outer(s_diag, h_diag)
                    sums[name] = f_tilde if name not in sums else sums[name] + f_tilde
                batches += 1
    finally:
        for handle in handles:
            handle.remove()
    if batches == 0:
        raise ValueError(f"no full micro-batch of {batch_size} in {int(inputs.shape[0])} probes")
    return {name: Diag(_reference_layout_vector(value / batches, wanted[name]))
            for name, value in sums.items()}


def _reference_layout_vector(f_tilde: Tensor, module: nn.Module) -> Tensor:
    """Flatten a ``(d_out, d_in_aug)`` per-entry statistic into the layout P1 holds that block in.

    ``Linear``/``Conv2d``: the bias-augmented ``rvec([W | b])`` order, which is the plain row-major
    flattening. **Normalisation layer**: ``named_parameters()`` order, all of ``gamma`` then all of
    ``beta`` — *not* the row-major one, which interleaves them (``plan_exp_lot5.md`` §0.2).
    """
    if isinstance(module, DIAG_PY_TYPES):
        return torch.cat([f_tilde[:, 0], f_tilde[:, 1]])
    return f_tilde.reshape(-1)


def kron_py_reading(model: nn.Module, inputs: Tensor, targets: Tensor,
                    modules: Mapping[str, nn.Module], *,
                    loss_fn: Callable[[Tensor, Tensor], Tensor], batch_size: int,
                    dtype: torch.dtype = torch.float64, device: object = "cpu",
                    types: Tuple[type, ...] = HOOKED_TYPES) -> Dict[str, "BlockOps"]:
    """``kron(compute_s_full(s, m), compute_h_full(h, m))`` per hooked module — the P1-py rung
    (``plan_exp_lot5.md`` §0.4): the optimizer's **own** Kronecker factor formulas, on the campaign's
    probes, with no running average, no min-max and no damping.

    Same conventions as :func:`diag_py_reading`, same single hooked pass, same caveat: the gradient
    is the *mean* loss's over a training-size micro-batch, so the overall scale carries the
    ``1/batch^2`` of ``CLAUDE.md`` §4.3 and the reading is used through ``cos_F``, ``e_F_star`` and
    ``c*``, never a raw ``e_F``.

    It exists because P1 has **no Kronecker structure on a normalisation layer at all**: the
    campaign's own ``A`` for such a layer is the ``(C+1) x (C+1)`` second moment of the *normalised*
    input, which does not even have the size of a ``2C``-parameter block, while the optimizer uses
    the ``2 x 2`` surrogate of ``plan_lot5.md`` §0.2. Without this reading, three of HF7's five mode
    pairs would have nothing to compare against on any normalisation layer.
    """
    from adafisher_modes.factors import compute_h_full, compute_s_full  # noqa: PLC0415

    from .base import Dense  # noqa: PLC0415
    from .kfac import Kron  # noqa: PLC0415

    def combine(sums: Dict[str, Tuple[Tensor, Tensor]], batches: int
                ) -> Dict[str, "BlockOps"]:
        out: Dict[str, "BlockOps"] = {}
        for name, (a_sum, g_sum) in sums.items():
            module = modules[name]
            A, G = a_sum / batches, g_sum / batches
            if isinstance(module, DIAG_PY_TYPES):
                # (C, 2) direction -> interleaved rvec; P1 holds the block in named order.
                from .adafisher_state import norm_permutation  # noqa: PLC0415
                perm = norm_permutation(int(module.weight.numel()))
                dense = torch.kron(G, A)
                out[name] = Dense(dense[perm][:, perm].contiguous())
            else:
                out[name] = Kron(A=A, G=G)
        return out

    return _hooked_factor_pass(model, inputs, targets, modules, loss_fn=loss_fn,
                               batch_size=batch_size, dtype=dtype, device=device, types=types,
                               per_module=lambda h, s, m: (compute_h_full(h, m),
                                                           compute_s_full(s, m)),
                               combine=combine)


def diag_py_factor_reading(model: nn.Module, inputs: Tensor, targets: Tensor,
                           modules: Mapping[str, nn.Module], *,
                           loss_fn: Callable[[Tensor, Tensor], Tensor], batch_size: int,
                           dtype: torch.dtype = torch.float64, device: object = "cpu",
                           types: Tuple[type, ...] = HOOKED_TYPES) -> Dict[str, Diag]:
    """``diag.py``'s formula with the micro-batches averaged **factor by factor**, the way the
    optimizer's running average actually combines them (lot 5, ``plan_exp_lot5.md`` §5).

    :func:`diag_py_reading` averages the *product* ``outer(S_D, H_D)`` over micro-batches; the
    optimizer keeps one running average per factor and multiplies at the end, so the faithful
    EMA-free reading is ``outer(mean_b S_D, mean_b H_D)``. The two differ by the covariance of
    ``H_D`` and ``S_D`` across micro-batches, and both are emitted so that difference is a measured
    number in the CSV rather than a choice argued for here.

    Lot 3's convention is the default of :func:`diag_py_reading` and is left untouched, so its A1,
    A2 and A3 rows stay exactly what they were.
    """
    from adafisher_modes.factors import compute_h_diag, compute_s_diag  # noqa: PLC0415

    def combine(sums: Dict[str, Tuple[Tensor, Tensor]], batches: int) -> Dict[str, Diag]:
        return {name: Diag(_reference_layout_vector(
            torch.outer(second / batches, first / batches), modules[name]))
            for name, (first, second) in sums.items()}

    return _hooked_factor_pass(model, inputs, targets, modules, loss_fn=loss_fn,
                               batch_size=batch_size, dtype=dtype, device=device, types=types,
                               per_module=lambda h, s, m: (compute_h_diag(h, m),
                                                           compute_s_diag(s, m)),
                               combine=combine)


def _hooked_factor_pass(model: nn.Module, inputs: Tensor, targets: Tensor,
                        modules: Mapping[str, nn.Module], *,
                        loss_fn: Callable[[Tensor, Tensor], Tensor], batch_size: int,
                        dtype: torch.dtype, device: object, types: Tuple[type, ...],
                        per_module: Callable[[Tensor, Tensor, nn.Module], Tuple[Tensor, Tensor]],
                        combine: Callable[[Dict[str, Tuple[Tensor, Tensor]], int], Dict],
                        ) -> Dict:
    """One hooked pass over the probes in training-size micro-batches, accumulating a pair of
    statistics per module. The forward-hook + tensor-hook shape is :func:`diag_py_reading`'s, kept
    in one place so the ``inputs=[batch]`` subtlety of ``plan_exp_lot1.md`` §0.1 lives once.
    """
    wanted = _hooked(modules, types)
    if not wanted:
        return {}
    raw_inputs: Dict[str, Tensor] = {}
    raw_grads: Dict[str, Tensor] = {}

    def make_hook(name: str):
        def hook(module: nn.Module, args: Tuple, output: Tensor) -> None:
            raw_inputs[name] = args[0].detach()
            output.register_hook(lambda grad: raw_grads.__setitem__(name, grad.detach()))
        return hook

    sums: Dict[str, Tuple[Tensor, Tensor]] = {}
    batches = 0
    handles = [module.register_forward_hook(make_hook(name)) for name, module in wanted.items()]
    try:
        with reference_mode(model):
            for start in range(0, int(inputs.shape[0]) - batch_size + 1, batch_size):
                batch = inputs[start:start + batch_size].to(device=device, dtype=dtype)
                batch.requires_grad_(True)
                labels = targets[start:start + batch_size].to(device)
                raw_inputs.clear()
                raw_grads.clear()
                loss = loss_fn(model(batch), labels)
                torch.autograd.grad(loss, [batch])
                for name, module in wanted.items():
                    first, second = per_module(raw_inputs[name], raw_grads[name], module)
                    if name not in sums:
                        sums[name] = (first, second)
                    else:
                        sums[name] = (sums[name][0] + first, sums[name][1] + second)
                batches += 1
    finally:
        for handle in handles:
            handle.remove()
    if batches == 0:
        raise ValueError(f"no full micro-batch of {batch_size} in {int(inputs.shape[0])} probes")
    return combine(sums, batches)


__all__ = ["DIAG_PY_TYPES", "HOOKED_TYPES", "NormStats", "cross_term_share",
           "diag_py_factor_reading", "diag_py_reading", "exact_separate", "kron_py_reading"]
