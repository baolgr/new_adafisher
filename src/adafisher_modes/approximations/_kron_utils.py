"""Shape plumbing shared by the four Kronecker modes: fold a (weight, bias) direction pair into one
matrix before preconditioning it, and split the result back.

The input factor of every supported layer type is built from the layer's input with a constant-one
column appended, so that the bias is treated as one more input coordinate. The direction the
optimizer wants to precondition therefore has to be laid out the same way: the weight as a
``(d_out, d_in)`` matrix with the bias appended as one more column. That is all
:func:`augment_direction` and :func:`split_direction` do. A ``Conv2d`` weight, shaped
``(C_out, C_in, k_h, k_w)``, is flattened to ``(C_out, C_in * k_h * k_w)`` first, which is the same
order the patch extraction produces, so the inverse reshape recovers the kernel layout unchanged.

The ``diag`` mode does the same split inline and is left alone; this module exists because the four
Kronecker modes need it identically.

**The SUA pair is different, and needs its own two functions.** Under SUA the input factor is
channel-only (``C_in [+1]``), narrower than the weight's own flattened width
(``C_in * k_h * k_w [+1]``), so one flat matrix product no longer type-checks.
:func:`augment_conv2d_direction_sua` instead exposes the weight as ``k_h * k_w`` independent
``(C_out, C_in [+1])`` slices, one per kernel offset in row-major ``(k_h, k_w)`` order, which a
batched matrix product preconditions with the same small operator and no explicit loop. The bias
direction is broadcast into every slice, mirroring the offset-independent constant-one column of the
input factor, and :func:`split_conv2d_direction_sua` reads the preconditioned bias back from the
centre offset only. The other ``k_h * k_w - 1`` offsets produce different bias estimates; that is a
property of treating the offsets as independent, not a bug. The centre is used because it is the
same reference offset the SUA input factor itself is built from, and it matches
``EKFAC-pytorch::_precond_sua_ra``; there is no theorem behind it.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

from torch import Size, Tensor, cat


def augment_direction(weight_direction: Tensor, bias_direction: Optional[Tensor]) -> Tensor:
    """Concatenate weight/bias momentum directions into one ``(d_out, d_in [+1])`` matrix,
    mirroring the bias-augmented-input convention (a ones column appended) used to build the
    corresponding input factor. ``weight_direction`` with more than 2 dimensions (a ``Conv2d``
    weight, ``(C_out, C_in, k_h, k_w)``) is flattened to ``(C_out, C_in*k_h*k_w)`` first, which is
    the order the patch extraction already produces.
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
    """SUA analogue of ``augment_direction``: expose a Conv2d
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
    kernel offset only — the same reference offset
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
