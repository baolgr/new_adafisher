"""Shared plumbing for the four non-diagonal Fisher approximation modes (K-FAC, EKFAC, TKFAC,
TEKFAC): folding a (weight, bias) direction pair into one augmented-shape tensor before applying a
Kronecker-factored preconditioner, and splitting the result back.

This is the same "shape plumbing" as the reference AdaFisher's own weight/bias split
(``adafisher.py:218-222``), which ``diag.py`` already inlines directly. Factored out here (private,
leading underscore — not part of the public API) because K-FAC/EKFAC (lot 2) and TKFAC/TEKFAC
(lot 3, ``docs/reports/plan.md`` §2.1's file layout) all need it identically, whereas ``diag.py``'s
own inline version is left untouched per ``CLAUDE.md``'s "no unsolicited refactor of existing code"
convention. See ``docs/reports/plan_lot2.md`` §0.2.

Lot 4 (docs/reports/plan_lot4.md §0.3) extends ``augment_direction`` to a ``Conv2d`` weight
direction, shaped ``(C_out, C_in, k_h, k_w)`` rather than ``(d_out, d_in)``: flattened to
``(C_out, C_in*k_h*k_w)`` before the (optional) bias column is appended, matching the patch-dimension
layout ``extract_patches`` already produces. ``split_direction`` needs no change — reshaping a
``(C_out, C_in*k_h*k_w)`` slice back to ``weight_shape`` already recovers the ``(C_in, k_h, k_w)``
layout correctly, since that flattening order already matches ``Conv2d.weight``'s own.

Lot 6 (docs/reports/plan_lot6.md §0.4) adds ``augment_conv2d_direction_sua``/
``split_conv2d_direction_sua``: under the SUA approximation the input factor is channel-only
(``C_in[+1]``), smaller than the weight's own flattened width (``C_in*k_h*k_w[+1]``), so the single
flat matmul ``augment_direction``/``split_direction`` build is no longer shape-compatible. These two
new functions instead expose the weight direction as ``k_h*k_w`` independent ``(C_out, C_in[+1])``
slices — one per kernel offset, all preconditioned by the *same* small operator via ordinary batched-
matmul broadcasting (no explicit loop in the callers) — with the bias direction broadcast into every
slice and, on the way back, read off only the center slice's estimate (plan_lot6.md §0.4's bias
convention, inherited from ``EKFAC-pytorch::_precond_sua_ra`` for lack of a theorem-backed
alternative).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

from torch import Size, Tensor, cat


def augment_direction(weight_direction: Tensor, bias_direction: Optional[Tensor]) -> Tensor:
    """Concatenate weight/bias momentum directions into one ``(d_out, d_in [+1])`` matrix,
    mirroring the bias-augmented-input convention (a ones column appended) used to build the
    corresponding input factor. ``weight_direction`` with more than 2 dimensions (a ``Conv2d``
    weight, ``(C_out, C_in, k_h, k_w)``) is flattened to ``(C_out, C_in*k_h*k_w)`` first — lot 4,
    plan_lot4.md §0.3.
    """
    W = (
        weight_direction
        if weight_direction.ndim == 2
        else weight_direction.reshape(weight_direction.size(0), -1)
    )
    if bias_direction is None:
        return W
    return cat([W, bias_direction.unsqueeze(1)], dim=1)


def split_direction(
    M: Tensor, weight_shape: Size, bias_shape: Optional[Size]
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Inverse of ``augment_direction``: slice a jointly-preconditioned ``(d_out, d_in [+1])``
    matrix back into ``(weight, bias)``, reshaped to their original shapes.
    """
    if bias_shape is None:
        return M.reshape(weight_shape)
    return M[:, :-1].reshape(weight_shape), M[:, -1:].reshape(bias_shape)


def augment_conv2d_direction_sua(
    weight_direction: Tensor, bias_direction: Optional[Tensor]
) -> Tensor:
    """SUA analogue of ``augment_direction`` (docs/reports/plan_lot6.md §0.4): expose a Conv2d
    weight direction, shaped ``(C_out, C_in, k_h, k_w)``, as ``k_h*k_w`` independent
    ``(C_out, C_in [+1])`` slices (one per kernel offset, row-major in ``(k_h, k_w)`` — matching
    ``split_conv2d_direction_sua``'s inverse reshape) instead of one flat
    ``(C_out, C_in*k_h*k_w [+1])`` matrix. The bias direction, if present, is broadcast identically
    into every slice, mirroring how ``augment_conv2d_input_sua`` appends an offset-independent ones
    column to the input factor.
    """
    c_out, c_in, kh, kw = weight_direction.shape
    M = weight_direction.permute(2, 3, 0, 1).reshape(kh * kw, c_out, c_in)
    if bias_direction is None:
        return M
    b = bias_direction.reshape(1, c_out, 1).expand(kh * kw, c_out, 1)
    return cat([M, b], dim=2)


def split_conv2d_direction_sua(
    M: Tensor, weight_shape: Size, bias_shape: Optional[Size]
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Inverse of ``augment_conv2d_direction_sua``. The bias estimate is read off the *center*
    kernel offset only (docs/reports/plan_lot6.md §0.4) — the same reference offset
    ``augment_conv2d_input_sua`` uses to build the input factor, not an arbitrary choice — since the
    other ``k_h*k_w - 1`` slices' own bias estimates are expected to disagree (a property of this
    block-diagonal-across-offsets approximation, not a bug).
    """
    c_out, c_in, kh, kw = weight_shape
    W = M[:, :, :c_in].reshape(kh, kw, c_out, c_in).permute(2, 3, 0, 1)
    if bias_shape is None:
        return W
    center = (kh // 2) * kw + (kw // 2)
    bias = M[center, :, -1].reshape(bias_shape)
    return W, bias
