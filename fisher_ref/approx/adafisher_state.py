"""Reading ``AdaFisherMulti``'s live per-module state as a curvature block.

The structural protocol asks how good a family of approximations can possibly be. The operational
protocol asks how good the thing the optimizer actually divides by is. This module is the bridge:
it turns the optimizer's own state -- after the running average, after the min-max renormalisation,
at its own damping -- into the same block interface every metric already takes.

**No new block class is needed.** For one hooked module the optimizer folds the weight and bias
directions into one ``(d_out, d_in_aug)`` matrix ``M`` and returns ``F~^-1 M``. In this
repository's row-major convention (output factor outer, input factor inner) that operator is:

===========  ==========================================================  ===========================
mode         ``F~``                                                      state read
===========  ==========================================================  ===========================
``diag``     ``Diag(f_tilde.flatten())``, ``f_tilde = kron(H,S)^T + lam`` ``_H``, ``_S``
``kfac``     ``kron(B~, A~)``                                            ``_A``, ``_B``
``ekfac``    ``(Q_B (x) Q_A) diag(s* + lam) (Q_B (x) Q_A)^T``            ``_Q_A``, ``_Q_B``, ``_s_star``
``tkfac``    ``delta * kron(Psi~, Phi~)``                                ``_delta``, ``_Phi_raw``, ``_Psi_raw``
``tekfac``   ``(Q_Psi (x) Q_Phi) diag(Theta + lam) (Q_Psi (x) Q_Phi)^T`` ``_Q_Phi``, ``_Q_Psi``, ``_Theta``
===========  ==========================================================  ===========================

Measured on a four-layer float64 network covering all four hooked module types: solving
``F~ x = rvec(M)`` densely reproduces the optimizer's own ``precondition`` to a relative error
between 0.0 and 2.2e-15 in all twenty (mode, layer) combinations.

**The layout trap, and it is the dangerous part.** For a ``Linear`` or a ``Conv2d`` the operator is
natively in the bias-augmented ``rvec([W | b])`` order, so there is nothing to do. For a
**normalisation layer** there is: the optimizer's input factor there is ``2 x 2``, so ``M`` is
``(C, 2)`` and its row-major order is *interleaved* -- ``(gamma_0, beta_0, gamma_1, beta_1, ...)``
-- while the campaign keeps that block in ``named_parameters()`` order, all of ``gamma`` then all of
``beta``. A block that skips :func:`norm_permutation` is symmetric, positive definite and silently
wrong.

**Two vintages, both reproduced rather than tidied.** ``precondition`` uses the inverses and
eigenbases cached at the last refresh and the factors accumulated at the last hook fire. With the
factor cadence equal to the refresh cadence (this project's default) those coincide, and
:func:`snapshot` checks it by comparing the cached inverse against a freshly computed one of the
same matrix. EKFAC's ``s*`` is the one place they genuinely differ: it is updated inside the
backward hook, using the basis from the *previous* refresh, and only then does the optimizer
refresh the basis. That is what the shipped optimizer does, so it is what is read.

Public API
----------

:data:`MODES`  the five Fisher modes.

:func:`norm_permutation`  the interleaved-to-blocked index vector for a normalisation layer.

:class:`OperationalBlock`  one module's operational preconditioner in two rungs: ``damped`` (the
operator the optimizer inverts, its own damping folded in, so a metric must apply it with no extra
damping) and ``undamped`` (the same operator with the optimizer's ``lambda`` removed, judged at the
sweep's damping like any structural one), plus ``checks``.

:func:`snapshot`  every hooked module's operational preconditioner. It raises on a spatially
uncorrelated convolution approximation: under that approximation the operator is not one Kronecker
product over the full patch direction but the same small operator applied independently at each
kernel offset, and representing it as a single Kronecker product would be wrong in the silent way.

:func:`min_eigenvalue`, :func:`worst_check`, :func:`least_check`  diagnostics over a snapshot.
``nan`` when no module reports a given check, which is the honest answer for, say, a cached-inverse
check on a mode that caches no inverse.

:data:`INVERSE_CONSISTENCY_TOL`  the bound above which a cached inverse is of a different *vintage*
than the factors it is paired with. The obvious check, ``||A~ A_inv - I||``, is the wrong one: it
conflates a stale inverse with an ill-conditioned factor, and this campaign has both. Comparing the
cached inverse against a freshly computed one *of the same matrix* cancels the conditioning
entirely, so a same-vintage pair agrees to round-off however badly conditioned it is, while a
different vintage is order one away. What is left is the precision gap between the optimizer's
float32 inverse and a float64 one, measured at 5.2e-7 and 4.0e-8 on a real thousand-step re-warm.

Dependencies: :mod:`fisher_ref.capture` (layer classification), :mod:`fisher_ref.approx.base`,
:mod:`fisher_ref.approx.kfac`, :mod:`fisher_ref.approx.ekfac`, and the live optimizer object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..capture import layer_kind
from .base import BlockOps, Dense, Diag
from .ekfac import EKFAC
from .kfac import Kron

#: The five Fisher modes ``AdaFisherMulti`` can be built in.
MODES = ("diag", "kfac", "ekfac", "tkfac", "tekfac")

#: ``||A_inv - inverse(A~)||_F / ||inverse(A~)||_F`` above this means the cached inverse is of a
#: different **vintage** than the factors it is paired with — a real defect.
#:
#: The obvious check, ``||A~ A_inv - I||``, is the wrong one: it conflates a stale inverse with an
#: ill-conditioned factor, and this campaign has both (``mlp_ln_mnist``'s first input factor is
#: 785 x 785 of rank 646, with 136 exactly-zero eigenvalues). Comparing the cached inverse against a
#: freshly computed one *of the same matrix* cancels the conditioning entirely: same input, same
#: call, so a same-vintage pair agrees to round-off however badly conditioned it is, while a
#: different vintage is O(1) away.
#:
#: What is left is the precision gap: the optimizer inverts in fp32 and this compares against an
#: fp64 inverse. MEASURED on a real 1 000-step re-warm of ``cnn_gn_cifar/diag/ckpt_0.5``:
#: ``5.2e-7`` for ``A`` and ``4.0e-8`` for ``B``, i.e. five orders of magnitude of margin under the
#: bound, and it does not grow with the factor's rank deficiency.
INVERSE_CONSISTENCY_TOL = 1e-2


@dataclass(frozen=True)
class OperationalBlock:
    """One module's operational preconditioner, in the layout P1 uses for that block kind."""

    name: str
    mode: str
    kind: str
    #: ``F~``: the operator the optimizer inverts, its own damping folded in. Metrics must apply it
    #: with ``k_lam=0`` — the damping is inside it, and for ``kfac``/``tkfac`` it is *factored*
    #: Tikhonov, which is not ``K + lam I`` for any ``lam``.
    damped: BlockOps
    #: The same operator with the optimizer's own ``lambda`` removed, judged at the sweep's
    #: ``lambda`` like any structural approximation.
    undamped: BlockOps
    #: Diagnostics recorded in ``meta.json`` rather than asserted away.
    checks: Dict[str, float]


def norm_permutation(channels: int) -> Tensor:
    """Index vector taking a normalisation layer's operator from the optimizer's interleaved
    ``rvec((C, 2))`` order into ``named_parameters()`` order (all of ``gamma``, then all of
    ``beta``).

    ``K_named = K_rvec[perm][:, perm]`` with ``perm[j*C + c] = c*2 + j``.
    """
    rows = torch.arange(channels)
    return torch.cat([rows * 2, rows * 2 + 1])


def _permuted_dense(block: BlockOps, channels: int) -> Dense:
    """A normalisation layer's operator, materialised and reordered (§0.2). ``2C x 2C`` is at most
    ``128 x 128`` on every model this campaign runs, so materialising it costs nothing and removes
    every chance of applying the permutation lazily in the wrong place.
    """
    perm = norm_permutation(channels)
    dense = block.to_dense()
    return Dense(dense[perm][:, perm].contiguous())


def _cast(tensor: Tensor, dtype: torch.dtype, device: Any) -> Tensor:
    return tensor.detach().to(device=device, dtype=dtype).clone()


def _inverse_consistency(factor: Tensor, inverse: Optional[Tensor]) -> float:
    """Is ``inverse`` the inverse of **this** ``factor``, or of an older one?

    ``||cached - inverse(factor)||_F / ||inverse(factor)||_F``. Deliberately not
    ``||factor @ cached - I||``: see :data:`INVERSE_CONSISTENCY_TOL`. ``nan`` when the mode caches
    no inverse (``diag``, ``ekfac``, ``tekfac``).
    """
    if inverse is None:
        return float("nan")
    fresh = factor.to(torch.float64).inverse()
    scale = float(fresh.norm())
    return float((inverse.to(torch.float64) - fresh).norm() / scale) if scale else float("nan")


def _diag_blocks(approx: Any, module: nn.Module, dtype: torch.dtype,
                 device: Any) -> tuple[BlockOps, BlockOps, Dict[str, float]]:
    # f_tilde() *is* the optimizer's own construction, min-max placement included; the undamped
    # rung is it minus the scalar Lambda it adds to every entry.
    f_tilde = _cast(approx.f_tilde(module), dtype, device)
    return (Diag(f_tilde.flatten()),
            Diag((f_tilde - approx.Lambda).flatten()),
            {})


def _kfac_blocks(approx: Any, module: nn.Module, dtype: torch.dtype,
                 device: Any) -> tuple[BlockOps, BlockOps, Dict[str, float]]:
    A_tilde, B_tilde = approx._damped_factors(module)
    checks = {
        "inverse_consistency_A": _inverse_consistency(A_tilde, approx._A_inv.get(module)),
        "inverse_consistency_B": _inverse_consistency(B_tilde, approx._B_inv.get(module)),
    }
    damped = Kron(A=_cast(A_tilde, dtype, device), G=_cast(B_tilde, dtype, device))
    undamped = Kron(A=_cast(approx._A[module], dtype, device),
                    G=_cast(approx._B[module], dtype, device))
    return damped, undamped, checks


def _ekfac_blocks(approx: Any, module: nn.Module, dtype: torch.dtype,
                  device: Any) -> tuple[BlockOps, BlockOps, Dict[str, float]]:
    q_a = _cast(approx._Q_A[module], dtype, device)
    q_b = _cast(approx._Q_B[module], dtype, device)
    s = _cast(approx._s_star[module], dtype, device)
    return (EKFAC(QA=q_a, QG=q_b, s=s + approx.Lambda),
            EKFAC(QA=q_a, QG=q_b, s=s),
            {"min_s_star": float(s.min())})


def _tkfac_blocks(approx: Any, module: nn.Module, dtype: torch.dtype,
                  device: Any) -> tuple[BlockOps, BlockOps, Dict[str, float]]:
    delta, Phi_tilde, Psi_tilde = approx._damped_factors(module)
    frozen = approx._delta_at_refresh.get(module)
    checks = {
        "inverse_consistency_Phi": _inverse_consistency(Phi_tilde, approx._Phi_inv.get(module)),
        "inverse_consistency_Psi": _inverse_consistency(Psi_tilde, approx._Psi_inv.get(module)),
        # precondition() divides by the delta frozen at the last refresh, _damped_factors by the
        # current one. They coincide when TCov == T_inv, which is this project's default.
        "delta_vintage_gap": (float("nan") if frozen is None
                              else float((delta - frozen).abs() / delta.abs().clamp_min(1e-30))),
    }
    delta_d = _cast(delta, dtype, device)
    damped = Kron(A=_cast(Phi_tilde, dtype, device), G=_cast(Psi_tilde, dtype, device) * delta_d)
    Phi = _cast(approx._Phi_raw[module], dtype, device) / delta_d
    Psi = _cast(approx._Psi_raw[module], dtype, device) / delta_d
    return damped, Kron(A=Phi, G=Psi * delta_d), checks


def _tekfac_blocks(approx: Any, module: nn.Module, dtype: torch.dtype,
                   device: Any) -> tuple[BlockOps, BlockOps, Dict[str, float]]:
    q_phi = _cast(approx._Q_Phi[module], dtype, device)
    q_psi = _cast(approx._Q_Psi[module], dtype, device)
    theta = _cast(approx._Theta[module], dtype, device)
    return (EKFAC(QA=q_phi, QG=q_psi, s=theta + approx.Lambda),
            EKFAC(QA=q_phi, QG=q_psi, s=theta),
            {"min_theta": float(theta.min())})


_BUILDERS = {"diag": _diag_blocks, "kfac": _kfac_blocks, "ekfac": _ekfac_blocks,
             "tkfac": _tkfac_blocks, "tekfac": _tekfac_blocks}


def snapshot(model: nn.Module, optimizer: Any, *, mode: str,
             dtype: torch.dtype = torch.float64,
             device: Any = "cpu") -> Dict[str, OperationalBlock]:
    """Every hooked module's operational preconditioner, keyed by module name.

    ``model`` must be the network ``optimizer`` was built on (the state is keyed by module identity).
    The result holds plain, detached, host-side copies, so the re-warm's model and optimizer can be
    freed immediately afterwards.

    Raises on ``conv_sua=True`` with a ``Conv2d`` present: under SUA the operator is **not** one
    ``kron(B~, A~)`` over the full ``(C_out, C_in*k_h*k_w + 1)`` direction but the same small
    ``(C_in + 1)``-wide operator applied independently at each kernel offset, with the bias read back
    from the centre offset only (``plan_lot6.md`` §0.4). Representing that as a single ``Kron`` would
    be wrong in the silent way — right shape, wrong matrix — so it is refused instead
    (``plan_exp_lot5.md`` §0.11).
    """
    if mode not in _BUILDERS:
        raise ValueError(f"unknown fisher mode {mode!r}; available: {list(MODES)}")
    approx = optimizer.approx
    hooked = {id(m) for m in optimizer.modules}
    owners = getattr(optimizer, "_owner", {})
    names = {id(m): name for name, m in model.named_modules()}
    if getattr(approx, "conv_sua", False) and any(
            isinstance(m, nn.Conv2d) for m in optimizer.modules):
        raise NotImplementedError(
            "conv_sua=True: the operational preconditioner of a Conv2d is not a single "
            "kron(B~, A~) over the full patch direction but the same (C_in+1)-wide operator applied "
            "independently at each kernel offset, with the bias read from the centre offset only "
            "(plan_lot6.md §0.4). fisher_ref cannot represent it yet (plan_exp_lot5.md §0.11)."
        )

    out: Dict[str, OperationalBlock] = {}
    for module in optimizer.modules:
        name = names.get(id(module))
        if name is None:
            raise KeyError("a hooked module is not part of the model passed to snapshot()")
        kind = layer_kind(module)
        if kind is None:
            raise RuntimeError(f"module {name!r} is hooked by the optimizer but fisher_ref cannot "
                               f"classify it ({type(module).__name__})")
        damped, undamped, checks = _BUILDERS[mode](approx, module, dtype, device)
        checks["min_eigenvalue_undamped"] = min_eigenvalue(undamped)
        for key, value in checks.items():
            # A cached inverse of a different vintage than the factors it is paired with would make
            # `damped` describe one matrix while `precondition` applies another -- symmetric,
            # positive definite and wrong. The bound is generous (a 785x785 fp32 inverse of a
            # sqrt(lambda)-damped factor lands around 1e-4); anything near it means the cadences
            # are not aligned, which is a defect, not round-off.
            if key.startswith("inverse_consistency") and value == value and \
                    value > INVERSE_CONSISTENCY_TOL:
                raise RuntimeError(
                    f"{name}: {key} = {value:.3e} > {INVERSE_CONSISTENCY_TOL:g}. The inverse "
                    f"cached at the last refresh is not the inverse of the current factors, so the "
                    f"operator read here is not the one precondition() applies. Check that "
                    f"TCov, T_inv and T_eig are aligned and that the re-warm length is a multiple "
                    f"of TCov (plan_exp_lot5.md §0.1)."
                )
        if kind == "norm":
            channels = int(module.weight.numel())
            damped = _permuted_dense(damped, channels)
            undamped = _permuted_dense(undamped, channels)
        out[name] = OperationalBlock(name=name, mode=mode, kind=kind, damped=damped,
                                     undamped=undamped, checks=checks)
    # Not "did the loop above populate what it iterated" — it did, by construction. The check that
    # can fail is against the optimizer's own pairing map: a module whose *parameters* it steps but
    # which is absent from ``optimizer.modules`` would be preconditioned in training and missing
    # here, and the symptom would be a quietly incomplete table.
    stepped = {id(module) for module in owners.values()}
    missing = sorted(names[i] for i in stepped
                     if i not in hooked and names.get(i) is not None)
    if missing:
        raise RuntimeError(
            f"these modules own a parameter the optimizer pairs to them but are not in "
            f"optimizer.modules, so they have no operational state: {missing}"
        )
    return out


def min_eigenvalue(block: BlockOps) -> float:
    """The smallest eigenvalue of an **undamped** operational operator.

    The quantity the re-warm length actually has to be compared against for the P2-raw rung
    (``plan_exp_lot5.md`` §0.6.3b). ``0.08^k << lambda`` is the rule for the *damped* operator, where
    ``lambda`` dominates anyway; with ``lambda`` removed, what the identity seed's residue ``0.08^k``
    has to be small against is the accumulated factor's own floor. Recorded per layer rather than
    assumed.

    For a Kronecker product of two PSD factors the minimum is the product of the two minima.
    """
    if isinstance(block, Kron):
        return float(torch.linalg.eigvalsh(block.A).min() * torch.linalg.eigvalsh(block.G).min())
    if isinstance(block, EKFAC):
        return float(block.s.min())
    if isinstance(block, Diag):
        return float(block.values.min())
    if isinstance(block, Dense):
        return float(torch.linalg.eigvalsh(block.matrix).min())
    return float("nan")


def worst_check(snapshots: Dict[str, OperationalBlock], key: str) -> float:
    """The largest value of one diagnostic over a snapshot.

    ``nan`` when no module reports it — which is the honest answer for, say,
    ``inverse_consistency_A`` on ``ekfac``, a mode that caches no inverse at all. Returning ``0.0``
    there would read as "measured, and perfect".
    """
    values = [block.checks[key] for block in snapshots.values()
              if key in block.checks and block.checks[key] == block.checks[key]]
    return max(values) if values else float("nan")


def least_check(snapshots: Dict[str, OperationalBlock], key: str) -> float:
    """The *smallest* value of one diagnostic over a snapshot, ignoring ``nan``.

    The companion of :func:`worst_check` for quantities where small is the risk —
    ``min_eigenvalue_undamped``, against which the identity seed's residue has to be negligible.
    """
    values = [block.checks[key] for block in snapshots.values()
              if key in block.checks and block.checks[key] == block.checks[key]]
    return min(values) if values else float("nan")


__all__ = ["INVERSE_CONSISTENCY_TOL", "MODES", "OperationalBlock", "least_check",
           "min_eigenvalue", "norm_permutation", "snapshot", "worst_check"]
