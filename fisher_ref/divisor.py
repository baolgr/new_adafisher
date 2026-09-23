"""The number the optimizer divides each direction by, and whether the curvature or the safety
constant sets it.

Lot 5 measured the *operator* ``F~`` the optimizer inverts and found it to be a multiple of the
identity: ``cond(F~)`` between 1.0000 and 1.1035, with ``lambda`` at 98.9-100 % of its mean
eigenvalue. That measurement was made at the **shipped** safety constant and nowhere else
(``plan_exp_lot5.md`` §6.7: ``lambda`` is not swept on the P2 side). Since then three replacements
for that constant have been tried -- a tuned single value (E7-E14), one value per layer (E15), and a
clip that uses no constant at all (E16) -- and every one of them was judged on **accuracy**. None was
judged on the question lot 5 asked. This module is what asks it.

**What is measured.** For one hooked module, the optimizer divides the momentum by one number per
direction, in whatever basis that mode works in. Call that list ``d`` and the module's own stored,
undamped curvature ``s``, in the same layout:

=========== ============================== ==============================================
mode        ``s``                          ``d``
=========== ============================== ==============================================
``diag``    ``kron(H', S')^T``             ``s + lambda`` (``f_tilde``, min-max included)
``kfac``    ``eig(B) (x) eig(A)``          ``eig(B~) (x) eig(A~)`` -- *factored* Tikhonov
``ekfac``   ``s*``                         ``s + lambda``, ``max(s, lambda)``, or the clip's
``tkfac``   ``delta eig(Psi) (x) eig(Phi)``  ``delta eig(Psi~) (x) eig(Phi~)``
``tekfac``  ``Theta``                      as ``ekfac``
=========== ============================== ==============================================

Under ``rescale_form="clip"`` there is no fixed operator at all -- the map is not linear, so
``cond(F~)`` is not defined and this module does not pretend otherwise. But ``d`` still is: the clip
divides coordinate ``i`` by

* ``gamma * s_i``   where the coordinate is active and its undamped step is under the threshold,
* ``|M_i|``         where it is active and clipped -- the momentum's own size, carrying no curvature,
* ``max(gamma s_i, guard)``  where it is inactive.

so ``d = |M| / |u|`` exactly, read from :class:`ClipRule`'s own record of the step rather than
recomputed here. That is why :func:`measure` calls the optimizer's ``precondition``: the divisor
under a non-linear form is a property of the step that was taken, not of a matrix.

**The headline statistic, and why it is a fraction rather than a ratio.** The obvious summary is how
much of the curvature's dynamic range survives into ``d``. It is unusable here: several factors in
this model zoo are *exactly* rank-deficient (``mlp_ln_mnist``'s first input factor has 136 zero
eigenvalues, ``plan_lambda_dominance.md``), so the curvature's range is infinite and every ratio
against it collapses to zero whatever the damping does. :func:`statistics` therefore leads with
``frac_curvature_set``: the fraction of directions whose divisor is set by the curvature rather than
by the safety constant or by the clip's ceiling. It is E1's own criterion (``s > lambda``, so the two
tables can be read against each other), generalised to the forms E1 did not have:

============= =========================================================================
form          a direction is *curvature-set* when
============= =========================================================================
``add``       ``s > lambda``
``floor``     ``s > lambda``           (below it the divisor is exactly ``lambda``)
factored      ``s_i > d_i - s_i``      (the curvature beats the damping's own contribution
              (``kfac``/``tkfac``)     to that eigen-direction; the same thing as ``s > lambda``
                                       when the damping is additive)
``clip``      ``(active and r < gamma) or (not active and gamma s > guard)``
============= =========================================================================

Everything else is descriptive and reported beside it: ``lambda_over_mean_divisor`` (lot 5's
"``lambda`` is 98.9-100 % of the divisor's mean") and its form-agnostic companion
``non_curvature_share``, ``cond_divisor`` and ``spread_divisor`` (lot 5's ``cond(F~)``),
``cos_to_plain`` (1.0 means the step is the plain momentum step -- doing nothing), and the rank
correlation between ``d`` and ``s``.

**One statistic exists because the count is misleading under the clip.** The clip fixes how many
coordinates it clips -- that is what ``clip_fraction`` *is* -- so ``frac_curvature_set`` there comes
out at about ``1 - clip_fraction`` whatever the curvature does, and comparing it with an additive
form's count would compare a measurement with a hyperparameter. ``step_share_curvature_set``, the
share of the step's squared norm those coordinates carry, is not fixed by anything and is the number
to read for the clip.

**One side effect, declared.** :func:`measure` calls ``precondition`` once per module outside
``step()``. For ``clip_threshold="ema"`` that advances the threshold's running average by one
observation out of a horizon of 1000, and for ``"fixed"`` it re-reads a frozen value without
advancing the calibration window (``ClipRule._calibrate`` returns early when no step is set). Neither
changes the state being read; nothing is written back to the momentum.

Dependencies: :mod:`fisher_ref.capture` (layer classification) and the live optimizer object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from adafisher_modes.approximations._kron_utils import augment_direction
from torch import Tensor

from .capture import layer_kind

#: The quantile pair every spread is read on. Robust to the exact-zero directions that a
#: rank-deficient factor puts at the bottom of ``s``.
SPREAD_QUANTILES = (0.1, 0.9)

#: Modes whose divisor lives in an eigenbasis the mode itself stores, so ``d`` is a stored vector
#: rather than a spectrum that has to be computed.
EIGEN_MODES = ("ekfac", "tekfac")


@dataclass(frozen=True)
class LayerDivisor:
    """One module's divisor, its undamped curvature, and how the two were obtained."""

    name: str
    kind: str
    mode: str
    #: ``"add"``, ``"floor"`` or ``"clip"``.
    form: str
    #: The number each direction is divided by, as a flat float64 vector.
    divisor: Tensor
    #: The module's own stored, undamped curvature, in the same layout and order as ``divisor``.
    curvature: Tensor
    #: The damping in effect for this module (``applied_lambda``). ``0.0`` under ``"clip"``, which
    #: uses none -- recorded as a number rather than as ``nan`` so that a reader cannot average it
    #: into a table of real ones by accident.
    lam: float
    #: ``True`` where that direction's divisor is set by the curvature; see the module docstring.
    curvature_set: Tensor
    #: ``cos(applied direction, momentum)``. Exactly 1 when the divisor is a constant.
    cos_to_plain: float
    #: :class:`ClipRule`'s own diagnostics for this module, under ``"clip"`` only.
    clip: Dict[str, float] = field(default_factory=dict)
    #: The step in the divisor's own basis, so that the share of it the curvature-set directions
    #: carry can be measured and not only their number. Built as ``M_kfe / d`` for the linear forms
    #: and read from the clip's own record otherwise -- the same quantity either way, which is what
    #: makes the clip comparable with the constant it replaces.
    applied: Optional[Tensor] = None


def _fp64(tensor: Tensor) -> Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float64)


#: Relative ridge added before a decomposition, and subtracted from the eigenvalues afterwards. Same
#: device as ``approximations/_eigh_utils.EIGH_RIDGE`` and the same reason: an exactly rank-deficient
#: factor makes the solver fail outright (it killed two cluster runs in campaign 1, and it fails on
#: LAPACK too, not only on cuSOLVER). The difference is that here the eigenvalues **are** the answer,
#: so the shift cannot be left in: ``M + cI`` has every eigenvalue of ``M`` plus ``c`` and the same
#: eigenvectors, so subtracting ``c`` afterwards is exact rather than approximate.
EIGH_RIDGE = 1e-6


def _decompose(matrix: Tensor, *, vectors: bool) -> Tuple[Tensor, Optional[Tensor]]:
    """Ascending eigenvalues of a symmetric factor, in float64 on the host, with the eigenvectors
    when asked for. Retries once with a relative ridge, which is then removed from the values."""
    dense = _fp64(matrix)
    if not bool(torch.isfinite(dense).all()):
        raise ValueError(f"a {tuple(dense.shape)} factor holds non-finite entries, so its spectrum "
                         f"is not a divisor; the optimizer's own damped factor is what is read here")
    try:
        if vectors:
            values, basis = torch.linalg.eigh(dense)
            return values, basis
        return torch.linalg.eigvalsh(dense), None
    except torch.linalg.LinAlgError:
        ridge = EIGH_RIDGE * float(dense.diagonal().mean().abs())
        conditioned = dense + ridge * torch.eye(dense.size(0), dtype=dense.dtype)
        if vectors:
            values, basis = torch.linalg.eigh(conditioned)
            return values - ridge, basis
        return torch.linalg.eigvalsh(conditioned) - ridge, None


def _eig(matrix: Tensor) -> Tensor:
    """Ascending eigenvalues of a symmetric factor, in float64 on the host."""
    return _decompose(matrix, vectors=False)[0]


def _kron_spectrum(inner: Tensor, outer: Tensor) -> Tensor:
    """Eigenvalues of ``kron(outer, inner)``, flattened in this repository's row-major convention:
    output factor outer, input factor inner, matching ``s*``'s own flattening."""
    return torch.outer(outer, inner).flatten()


def _ranks(values: Tensor) -> Tensor:
    """Average ranks, so that the exact ties a rank-deficient factor produces are not ordered by
    whatever ``argsort`` happens to do with them."""
    order = torch.argsort(values)
    positions = torch.empty_like(values)
    positions[order] = torch.arange(values.numel(), dtype=values.dtype)
    unique, inverse, counts = torch.unique(values, return_inverse=True, return_counts=True)
    summed = torch.zeros(unique.numel(), dtype=values.dtype).scatter_add_(0, inverse, positions)
    return (summed / counts.to(values.dtype))[inverse]


def _spearman(a: Tensor, b: Tensor) -> float:
    """Rank correlation. ``nan`` when either side is constant -- which is the answer, not a failure:
    a divisor with no variation has no ordering to agree with."""
    if a.numel() < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    if float(ra.std()) == 0.0 or float(rb.std()) == 0.0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


#: The basis the divisor is diagonal in: ``(Q_A, Q_B)``, or ``None`` for the parameter basis.
Basis = Optional[Tuple[Tensor, Tensor]]


def _eigh(matrix: Tensor) -> Tuple[Tensor, Tensor]:
    """Ascending eigenvalues and their eigenvectors, float64 on the host. Same order as
    :func:`_eig`, so a spectrum built from one and a projection built from the other agree index by
    index."""
    values, vectors = _decompose(matrix, vectors=True)
    assert vectors is not None
    return values, vectors


def _diag_pair(approx: Any, module: nn.Module) -> Tuple[Tensor, Tensor, Basis]:
    # f_tilde() *is* the optimizer's own construction, min-max placement included. Its basis is the
    # parameter basis, so no projection is needed.
    damped = _fp64(approx.f_tilde(module))
    lam = float(approx.lambda_for(module))
    return (damped - lam).flatten(), damped.flatten(), None


def _kfac_pair(approx: Any, module: nn.Module) -> Tuple[Tensor, Tensor, Basis]:
    A_tilde, B_tilde = approx._damped_factors(module)
    values_A, Q_A = _eigh(A_tilde)
    values_B, Q_B = _eigh(B_tilde)
    undamped = _kron_spectrum(_eig(approx._A[module]), _eig(approx._B[module]))
    return undamped, _kron_spectrum(values_A, values_B), (Q_A, Q_B)


def _tkfac_pair(approx: Any, module: nn.Module) -> Tuple[Tensor, Tensor, Basis]:
    delta, Phi_tilde, Psi_tilde = approx._damped_factors(module)
    delta = float(_fp64(delta)) if isinstance(delta, Tensor) else float(delta)
    # The operator is delta * kron(Psi, Phi), with Phi = Phi_raw/delta and Psi = Psi_raw/delta:
    # exactly the pair adafisher_state._tkfac_blocks builds, read as a spectrum instead of a block.
    undamped = delta * _kron_spectrum(_eig(approx._Phi_raw[module]) / delta,
                                      _eig(approx._Psi_raw[module]) / delta)
    values_Phi, Q_Phi = _eigh(Phi_tilde)
    values_Psi, Q_Psi = _eigh(Psi_tilde)
    return undamped, delta * _kron_spectrum(values_Phi, values_Psi), (Q_Phi, Q_Psi)


def _stored_rescaling(approx: Any, mode: str, module: nn.Module) -> Tensor:
    return (approx._s_star if mode == "ekfac" else approx._Theta)[module]


def _clip_divisor(approx: Any, module: nn.Module
                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, float]]:
    """``(s, d, curvature_set, |u|, stats)`` for one module under ``rescale_form="clip"``, read from
    :class:`ClipRule`'s record of the step :func:`measure` just asked for.

    ``d = |M| / |u|`` is the division that was actually performed, whatever branch produced ``u``.
    A coordinate with ``M = 0`` has no divisor (``u`` is 0 too) and is dropped.
    """
    record = approx._clip._records.get(module)
    if record is None:
        raise RuntimeError(
            "no clip record for this module: precondition() was not reached for it, so the "
            "divisor under a non-linear rescale form cannot be read"
        )
    abs_m, u = _fp64(record.abs_m), _fp64(record.u).abs()
    s = _fp64(record.s)
    active = _fp64(record.active.to(torch.float64)) > 0.5
    gamma = float(_fp64(record.gamma))
    guard = float(_fp64(record.guard))
    r = _fp64(record.r)
    moving = u > 0
    divisor = torch.where(moving, abs_m / u.clamp_min(torch.finfo(torch.float64).tiny),
                          torch.full_like(abs_m, float("nan")))
    # Curvature-set: the coordinates whose divisor is proportional to s. An active coordinate is,
    # until it is clipped (then the divisor is |M|); an inactive one is, until the guard takes over.
    curvature_set = torch.where(active, r < gamma, gamma * s > guard) & moving
    stats = {key: float(_fp64(value)) for key, value in approx._clip.stats.get(module, {}).items()}
    return s.flatten(), divisor.flatten(), curvature_set.flatten(), u.flatten(), stats


def _eigen_pair(approx: Any, mode: str, module: nn.Module) -> Tuple[Tensor, Tensor, Basis]:
    s = _fp64(_stored_rescaling(approx, mode, module))
    lam = approx.lambda_for(module)
    lam = _fp64(lam) if isinstance(lam, Tensor) else float(lam)
    form = getattr(approx, "rescale_form", "add")
    basis = ((_fp64(approx._Q_A[module]), _fp64(approx._Q_B[module])) if mode == "ekfac"
             else (_fp64(approx._Q_Phi[module]), _fp64(approx._Q_Psi[module])))
    if form == "add":
        return s.flatten(), (s + lam).flatten(), basis
    if form == "floor":
        return (s.flatten(),
                torch.maximum(s, torch.as_tensor(lam, dtype=torch.float64)).flatten(), basis)
    raise ValueError(f"_eigen_pair handles the linear forms; {form!r} goes through _clip_divisor")


def _projected_momentum(module: nn.Module, weight: Tensor, bias: Optional[Tensor],
                        basis: Basis) -> Tensor:
    """The momentum in the basis the divisor is diagonal in, flattened in the divisor's own order.

    ``augment_direction`` is the optimizer's own weight-and-bias augmentation, imported rather than
    reproduced: a private re-spelling of it is how a block ends up the right shape and the wrong
    matrix, which this campaign has paid for twice (``plan_exp_lot2.md`` §5.2).
    """
    M = augment_direction(_fp64(weight), None if bias is None else _fp64(bias))
    if basis is None:
        return M.flatten()
    Q_A, Q_B = basis
    return (Q_B.t() @ M @ Q_A).flatten()


def _momentum(optimizer: Any, module: nn.Module) -> Optional[Tuple[Tensor, Optional[Tensor], float]]:
    """``(weight momentum, bias momentum, bias correction)`` as ``_step_module`` would see them, or
    ``None`` when the module has not been stepped yet."""
    weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    state = optimizer.state.get(weight) if weight is not None else None
    if state is None or "exp_avg" not in state:
        return None
    correction = 1.0 - optimizer.defaults["beta"] ** state["step"] \
        if "beta" in optimizer.defaults else 1.0 - 0.9 ** state["step"]
    bias_exp_avg = None
    if bias is not None and bias.requires_grad:
        bias_state = optimizer.state.get(bias)
        if bias_state is None or "exp_avg" not in bias_state:
            return None
        bias_exp_avg = bias_state["exp_avg"]
    return state["exp_avg"], bias_exp_avg, float(correction)


def _cos_to_plain(approx: Any, module: nn.Module, weight: Tensor, bias: Optional[Tensor],
                  correction: float) -> float:
    """``cos(precondition(m), m)``, with ``m`` handed over exactly as ``_step_module`` hands it.

    A divisor that is a constant multiple of the identity gives 1 exactly, whatever the constant, so
    this is the scale-free form of "the optimizer is taking the plain momentum step". ``hold_cap``
    and the step size are irrelevant to it by construction.
    """
    if approx.consumes_bias_corrected_momentum:
        weight = weight / correction
        bias = None if bias is None else bias / correction
    direction = approx.precondition(module, weight, bias)
    parts = direction if isinstance(direction, tuple) else (direction,)
    given = [weight] if bias is None else [weight, bias]
    flat_out = torch.cat([_fp64(p).flatten() for p in parts if p is not None])
    flat_in = torch.cat([_fp64(p).flatten() for p in given])
    norms = float(flat_out.norm()) * float(flat_in.norm())
    return float(torch.dot(flat_out, flat_in) / norms) if norms > 0 else float("nan")


def measure(model: nn.Module, optimizer: Any, *, mode: str,
            skipped: Optional[Dict[str, str]] = None) -> Dict[str, LayerDivisor]:
    """Every hooked module's divisor, keyed by module name.

    ``model`` must be the network ``optimizer`` was built on. The optimizer must have been stepped at
    least once, so that a momentum exists to divide: a module without one is skipped and its absence
    is visible in the returned keys rather than filled with a guess.

    ``skipped``, when given, is filled with ``{layer: why}`` for every module that could not be read,
    and those modules are left out instead of ending the whole read. One degenerate layer -- a factor
    the mode's own damping turned non-finite, say -- should cost that layer, not the other thirty-eight
    of a ResNet. Pass ``None`` to have such a layer raise instead.

    Refuses ``conv_sua=True`` with a ``Conv2d`` present, for the reason
    :func:`fisher_ref.approx.adafisher_state.snapshot` refuses it: the divisor is then applied
    independently at each kernel offset and a single flat list would be the wrong object. No model in
    this experiment's registry sets it.
    """
    approx = optimizer.approx
    form = getattr(approx, "rescale_form", "add")
    if getattr(approx, "conv_sua", False) and any(isinstance(m, nn.Conv2d)
                                                  for m in optimizer.modules):
        raise NotImplementedError(
            "conv_sua=True: the divisor of a Conv2d is applied independently at each kernel offset "
            "(plan_lot6.md §0.4), so one flat list per module is the wrong object here"
        )
    if form == "clip" and mode not in EIGEN_MODES:
        raise ValueError(f"rescale_form='clip' exists for {EIGEN_MODES}; mode is {mode!r}")

    names = {id(m): name for name, m in model.named_modules()}
    out: Dict[str, LayerDivisor] = {}
    for module in optimizer.modules:
        name = names.get(id(module))
        if name is None:
            raise KeyError("a hooked module is not part of the model passed to measure()")
        moment = _momentum(optimizer, module)
        if moment is None:
            if skipped is not None:
                skipped[name] = "no momentum yet: the optimizer has not stepped this module"
            continue
        weight, bias, correction = moment
        try:
            out[name] = _read_one(approx, module, name, mode, form, weight, bias, correction)
        except (ValueError, torch.linalg.LinAlgError, RuntimeError) as exc:
            if skipped is None:
                raise
            skipped[name] = f"{type(exc).__name__}: {exc}"
    return out


def _read_one(approx: Any, module: nn.Module, name: str, mode: str, form: str, weight: Tensor,
              bias: Optional[Tensor], correction: float) -> LayerDivisor:
    """One module's divisor. Split out of :func:`measure` so that a failure is attributable to a
    single layer and can be recorded as such."""
    cos = _cos_to_plain(approx, module, weight, bias, correction)
    clip_stats: Dict[str, float] = {}
    if form == "clip":
        # Read after the precondition() call above, so the record describes that call's tensors.
        s, d, curvature_set, applied, clip_stats = _clip_divisor(approx, module)
        lam = 0.0
    else:
        if mode == "diag":
            s, d, basis = _diag_pair(approx, module)
        elif mode == "kfac":
            s, d, basis = _kfac_pair(approx, module)
        elif mode == "tkfac":
            s, d, basis = _tkfac_pair(approx, module)
        else:
            s, d, basis = _eigen_pair(approx, mode, module)
        # The curvature beats the damping's own contribution to that direction. For an additive
        # lambda this is exactly E1's `s > lambda`; for kfac/tkfac, whose damping is folded into the
        # two factors, it is the only form of the question that is well posed per direction.
        curvature_set = s > (d - s)
        lam_applied = approx.applied_lambda(module)
        lam = (float(_fp64(lam_applied).mean()) if isinstance(lam_applied, Tensor)
               else float(lam_applied))
        # The step in the divisor's own basis, so that the share of it the curvature-set directions
        # carry is measured the same way here as it is under the clip.
        applied = _projected_momentum(module, weight, bias, basis) / d
    kind = layer_kind(module) or type(module).__name__
    return LayerDivisor(name=name, kind=kind, mode=mode, form=form, divisor=d, curvature=s,
                        lam=lam, curvature_set=curvature_set, cos_to_plain=cos, clip=clip_stats,
                        applied=applied)


def statistics(layer: LayerDivisor) -> Dict[str, float]:
    """The numbers one layer contributes to the table. ``nan`` wherever a quantity is not defined
    for that form, never a stand-in value."""
    finite = torch.isfinite(layer.divisor) & torch.isfinite(layer.curvature)
    d, s = layer.divisor[finite], layer.curvature[finite]
    curvature_set = layer.curvature_set[finite]
    low, high = SPREAD_QUANTILES

    def spread(values: Tensor) -> float:
        if values.numel() == 0:
            return float("nan")
        bottom = float(torch.quantile(values, low))
        top = float(torch.quantile(values, high))
        return top / bottom if bottom > 0 else float("inf")

    def cond(values: Tensor) -> float:
        if values.numel() == 0:
            return float("nan")
        bottom = float(values.min())
        return float(values.max()) / bottom if bottom > 0 else float("inf")

    mean_d = float(d.mean()) if d.numel() else float("nan")
    mean_s = float(s.mean()) if s.numel() else float("nan")
    # The share of the step's energy carried by the curvature-set directions, where the applied
    # direction is available in the divisor's own basis. It is the statistic to read for the clip,
    # where the *count* of curvature-set directions is not a measurement at all: the clip fixes the
    # clipped fraction at ``clip_fraction`` by construction, so `frac_curvature_set` there comes out
    # at about ``1 - clip_fraction`` whatever the curvature does. How much of the step those
    # coordinates carry is not fixed by anything.
    step_share = float("nan")
    if layer.applied is not None:
        energy = layer.applied[finite].to(torch.float64).pow(2)
        total = float(energy.sum())
        step_share = float(energy[curvature_set].sum()) / total if total > 0 else float("nan")
    out = {
        "n_directions": float(d.numel()),
        # THE headline: the share of directions whose divisor the curvature sets.
        "frac_curvature_set": float(curvature_set.to(torch.float64).mean()) if d.numel()
        else float("nan"),
        "step_share_curvature_set": step_share,
        # The share of the average divisor that is *not* curvature. Defined for every form,
        # including a factored damping and the clip -- under the clip what fills it is the momentum
        # ceiling rather than a constant, which is why this is not called a damping share.
        "non_curvature_share": 1.0 - mean_s / mean_d if mean_d else float("nan"),
        # Lot 5's literal statistic, "lambda is 98.9-100 % of the divisor's mean eigenvalue". Equal
        # to the line above whenever the damping is a constant that is added; ``nan`` under the clip,
        # which adds none.
        "lambda_over_mean_divisor": (layer.lam / mean_d if layer.form != "clip" and mean_d
                                     else float("nan")),
        "cond_divisor": cond(d),
        "spread_divisor": spread(d),
        "cond_curvature": cond(s),
        "spread_curvature": spread(s),
        "mean_divisor": mean_d,
        "mean_curvature": mean_s,
        "min_curvature": float(s.min()) if s.numel() else float("nan"),
        "max_curvature": float(s.max()) if s.numel() else float("nan"),
        "spearman_divisor_curvature": _spearman(d, s),
        "cos_to_plain": layer.cos_to_plain,
        "lam": layer.lam,
        # E1's own criterion, kept under its own name for the forms where lambda is a number that is
        # added: so that this table can be read directly against E1's percentiles.
        "frac_curvature_above_lambda": (float((s > layer.lam).to(torch.float64).mean())
                                        if layer.form != "clip" and s.numel() else float("nan")),
    }
    out.update({f"clip_{key}": value for key, value in layer.clip.items()})
    return out


def summarise(layers: Dict[str, LayerDivisor], keys: Optional[List[str]] = None
              ) -> Dict[str, float]:
    """Median over layers of each statistic. The median, not the mean: one rank-deficient layer
    otherwise decides a whole network's row."""
    rows = [statistics(layer) for layer in layers.values()]
    wanted = keys or sorted({key for row in rows for key in row})
    out: Dict[str, float] = {}
    for key in wanted:
        values = [row[key] for row in rows if key in row and row[key] == row[key]
                  and abs(row[key]) != float("inf")]
        out[key] = float(torch.tensor(values, dtype=torch.float64).median()) if values \
            else float("nan")
    out["n_layers"] = float(len(rows))
    return out


__all__ = ["EIGEN_MODES", "SPREAD_QUANTILES", "LayerDivisor", "measure", "statistics", "summarise"]
