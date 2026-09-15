"""Regime A: the dense, exact reference ``F = U^T U`` in fp64 (``plan_exp_draft.md`` §2.1-§2.2,
lot 1 of its §9).

``U`` stacks one row per (probe ``n``, root column ``c``), the row being the parameter gradient of
the scalar ``<Lambda_n^{1/2} e_c, f(x_n)>``, so ``F = U^T U`` holds exactly with a ``1/N`` in front
(applied once at the end, not as an ``N^{-1/2}`` per row: one division instead of ``m`` multiplies).
The same function at ``source="empirical"`` gives ``E_hat``, and ``B_l`` is a **slice** of the
result — there is no second construction to be wrong.

Two decisions, both load-bearing for every later lot.

**The column layout is ``model.named_parameters()`` order**, each parameter contributing its own
``numel`` in row-major (``rvec``) order — *not* the bias-augmented per-module ``rvec([W | b])``
layout of ``adafisher_modes``. Three reasons, in increasing order of importance: a parameter-space
vector (a gradient, a ``theta``, a random direction) needs no conversion, so the T1 oracle compares
like with like; ``weight`` is registered before ``bias`` in every relevant module, so a module's
block stays contiguous and ``B_l`` is a slice; and, decisively, ``F`` must span **every** parameter,
including the ones that have no ``[W | b]`` slot at all (``pos_embed``, ``cls_token``, a
``GroupNorm`` affine). Lot 2 adds the small permutation into ``rvec([W | b])`` when it has a K-FAC
to compare against.

**``U`` is never materialised.** ``F`` is accumulated as ``F += U_blk^T U_blk`` over probe
micro-batches; only ``U_blk`` (``batch_size x P``, 54 MB at ``P = 26 634``) and ``F`` itself
(5.68 GB for A1) are resident. `plan_exp_draft.md` §2.2 sizes the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..capture import (
    capturable_modules,
    coverage,
    iter_probe_columns,
    per_sample_gradients,
    prepare_model,
)
from ..conventions import REFERENCE_DTYPE, run_metadata


@dataclass(frozen=True)
class ParamLayout:
    """Which columns of ``U`` (and rows/columns of ``F``) belong to which parameter."""

    names: Tuple[str, ...]
    slices: Dict[str, slice]
    shapes: Dict[str, torch.Size]
    P: int

    @classmethod
    def of(cls, model: nn.Module, params: Optional[Iterable[str]] = None) -> "ParamLayout":
        """Layout over ``params`` (default: every parameter), in ``named_parameters()`` order."""
        selected = None if params is None else set(params)
        names: List[str] = []
        slices: Dict[str, slice] = {}
        shapes: Dict[str, torch.Size] = {}
        offset = 0
        for name, parameter in model.named_parameters():
            if selected is not None and name not in selected:
                continue
            names.append(name)
            slices[name] = slice(offset, offset + parameter.numel())
            shapes[name] = parameter.shape
            offset += parameter.numel()
        if selected is not None and len(names) != len(selected):
            missing = sorted(selected - set(names))
            raise KeyError(f"parameters not found on the model: {missing}")
        return cls(names=tuple(names), slices=slices, shapes=shapes, P=offset)

    def block_slice(self, module_name: str) -> slice:
        """The contiguous column range of one module's parameters.

        Raises if they are not contiguous — which would mean ``B_l`` is not a submatrix and every
        block metric silently measures the wrong thing.
        """
        prefix = f"{module_name}." if module_name else ""
        owned = [name for name in self.names
                 if name.startswith(prefix) and "." not in name[len(prefix):]]
        if not owned:
            raise KeyError(f"no parameter of module {module_name!r} is in this layout")
        start = min(self.slices[name].start for name in owned)
        stop = max(self.slices[name].stop for name in owned)
        if stop - start != sum(self.slices[name].stop - self.slices[name].start for name in owned):
            raise RuntimeError(
                f"the parameters of {module_name!r} are not contiguous in the layout: {owned}"
            )
        return slice(start, stop)

    def augmented_permutation(self, module_name: str) -> Tensor:
        """Index vector taking this module's block from ``named_parameters()`` order into the
        bias-augmented ``rvec([W | b])`` order the Kronecker formulas are written in.

        ``block_augmented = block_named[perm][:, perm]``. This is the permutation
        ``plan_exp_lot1.md`` §0.5 deferred to "the lot that has a K-FAC to compare against": a
        reference stacks ``rvec(W)`` then ``b``, two contiguous parameters, while ``G (x) A`` with a
        bias-augmented ``A`` interleaves the bias as the last *column* of every output row. Same
        entries, different order — which is exactly how T3 failed on its first run.

        Identity when the module has no bias. Raises for a normalisation layer, whose
        ``(gamma, beta)`` block is Hadamard-structured and has no ``[W | b]`` reading at all
        (``plan_lot5.md`` §0.1).
        """
        prefix = f"{module_name}." if module_name else ""
        weight_shape = self.shapes.get(prefix + "weight")
        if weight_shape is None:
            raise KeyError(f"module {module_name!r} has no weight in this layout")
        if len(weight_shape) < 2:
            raise NotImplementedError(
                f"{module_name!r} has a 1-D weight {tuple(weight_shape)}: a normalisation layer's "
                "block is Hadamard-structured, not rvec([W | b]) (plan_lot5.md §0.1)"
            )
        d_out = int(weight_shape[0])
        d_in = int(weight_shape.numel() // d_out)
        base = self.block_slice(module_name).start
        offset = self.slices[prefix + "weight"].start - base
        bias_slice = self.slices.get(prefix + "bias")
        if bias_slice is None:
            return torch.arange(offset, offset + d_out * d_in)
        bias_offset = bias_slice.start - base
        perm = torch.empty(d_out * (d_in + 1), dtype=torch.long)
        for row in range(d_out):
            span = slice(row * (d_in + 1), row * (d_in + 1) + d_in)
            perm[span] = torch.arange(offset + row * d_in, offset + (row + 1) * d_in)
            perm[row * (d_in + 1) + d_in] = bias_offset + row
        return perm


def to_augmented(block: Tensor, perm: Tensor) -> Tensor:
    """Reorder a block from ``named_parameters()`` order into ``rvec([W | b])`` order."""
    return block[perm][:, perm]


@dataclass
class DenseReference:
    """A materialised ``P x P`` reference, plus the provenance every result file must carry."""

    matrix: Tensor
    layout: ParamLayout
    source: str
    loss: str
    n_probes: int
    n_columns: int
    probe_digest: Optional[str] = None

    @property
    def P(self) -> int:
        return self.layout.P

    @property
    def n_rows(self) -> int:
        """Rows of ``U``. For the type-2 softmax root this is ``N*C`` for a **rank** of ``N(C-1)``:
        ``S_n`` has the null vector ``sqrt(p_n)``, which is what ``plan_exp_draft.md`` §2.2-§2.3's
        tables quote as ``m``.
        """
        return self.n_probes * self.n_columns

    def block(self, module_name: str) -> Tensor:
        """``B_l`` — a view into :attr:`matrix`, not a second construction."""
        columns = self.layout.block_slice(module_name)
        return self.matrix[columns, columns]

    def matvec(self, v: Tensor) -> Tensor:
        return self.matrix @ v

    def trace(self) -> Tensor:
        return self.matrix.diagonal().sum()

    def fro2(self) -> Tensor:
        return (self.matrix * self.matrix).sum()

    def diag(self) -> Tensor:
        return self.matrix.diagonal()

    def fro(self) -> Tensor:
        """``‖·‖_F``. Recorded in :meth:`metadata` for a reason: ``plan_exp_lot1.md`` §6.3 left HF1
        unresolved because three gaps were reported against two different denominators and only the
        train-side norm was stored, so no reader could re-normalise them. Every reference now
        carries its own trace and norm (``plan_exp_lot2.md`` §0.1).
        """
        return self.fro2().sqrt()

    def metadata(self, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        meta = run_metadata({
            "protocol": "P1", "regime": "A", "source": self.source, "loss": self.loss,
            "n_probes": self.n_probes, "n_columns": self.n_columns, "n_rows": self.n_rows,
            "P": self.P, "probe_digest": self.probe_digest,
            "trace": float(self.trace()), "fro": float(self.fro()),
        })
        if extra:
            meta.update(dict(extra))
        return meta


def symmetrize_(matrix: Tensor, block: int = 4096) -> Tensor:
    """In-place ``M <- (M + M^T)/2``, block by block.

    The obvious spelling ``M = 0.5 * (M + M.T)`` allocates **two** further ``P x P`` tensors, and
    ``M.addmm_``'s absence in the accumulation loop would allocate a third: at ``P = 26 634`` each
    is 5.68 GB, so the naive builder needs ~17 GB where the matrix itself needs 5.68 — which is
    precisely ``plan_exp_draft.md`` §2.2's ``P_max = sqrt(B/24)`` and §12's "the 10 GB MIG cannot
    allocate ``F``" risk, reproduced by the code rather than by the mathematics. Working on
    off-diagonal block *pairs* costs one ``block x block`` temporary instead.

    A Gram accumulated by ``gemm`` is symmetric only to round-off (the reduction order differs
    between the two triangles), so this is cosmetic — but it is cheap, and downstream code that
    reads both triangles should not see a `1e-16` asymmetry it has to explain.
    """
    size = matrix.shape[0]
    for start in range(0, size, block):
        stop = min(start + block, size)
        diagonal = matrix[start:stop, start:stop]
        diagonal.add_(diagonal.clone().T).mul_(0.5)
        for other in range(stop, size, block):
            other_stop = min(other + block, size)
            upper = matrix[start:stop, other:other_stop]
            lower = matrix[other:other_stop, start:stop]
            averaged = upper.add(lower.T).mul_(0.5)
            upper.copy_(averaged)
            lower.copy_(averaged.T)
    return matrix


def _selected_modules(model: nn.Module,
                      modules: Optional[Sequence[str]]) -> Dict[str, nn.Module]:
    available = capturable_modules(model)
    if modules is None:
        _, uncovered = coverage(model)
        if uncovered:
            raise NotImplementedError(
                f"{len(uncovered)} parameter(s) belong to no capturable module and would be all-zero "
                f"columns of U, leaving F symmetric, PSD and wrong: {uncovered}. Pass "
                "modules=[...] to build a restricted reference, or wait for the lot that covers "
                "raw parameters (plan_exp_draft.md §1, point 3)."
            )
        return available
    unknown = [name for name in modules if name not in available]
    if unknown:
        raise KeyError(f"not capturable modules of this model: {unknown}; "
                       f"available: {sorted(available)}")
    return {name: available[name] for name in modules}


def build_dense_reference(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    source: str,
    loss: str = "cross_entropy",
    k: int = 1,
    batch_size: int = 256,
    dtype: torch.dtype = REFERENCE_DTYPE,
    device: Any = "cpu",
    generator: Optional[torch.Generator] = None,
    modules: Optional[Sequence[str]] = None,
    probe_digest: Optional[str] = None,
    check_independence: bool = True,
) -> DenseReference:
    """Build ``F`` (``source="type2"``), ``E_hat`` (``source="empirical"``) or the Monte-Carlo
    Fisher (``source="mc"``) over ``inputs``/``targets``, in ``dtype``.

    ``inputs``/``targets`` are a model batch, i.e. already through ``bench.prepare_batch`` (what
    ``probes.ProbeSet.as_model_batch`` returns). ``modules`` restricts the columns to those
    modules' parameters — with ``None``, every parameter must be covered, and an uncovered one is
    an error rather than a silent block of zeros.

    ``check_independence`` runs :func:`~fisher_ref.conventions.assert_sample_independent` once, on
    the first micro-batch: with a ``BatchNorm`` left in train mode there are no per-sample
    gradients to stack and the whole object is undefined (``plan_exp_draft.md`` §2.5).
    """
    model = prepare_model(model, dtype, device)
    selected = _selected_modules(model, modules)
    parameter_names = [f"{name}.{p_name}" if name else p_name
                       for name, module in selected.items()
                       for p_name, _ in module.named_parameters(recurse=False)]
    layout = ParamLayout.of(model, parameter_names)

    matrix = torch.zeros(layout.P, layout.P, dtype=dtype, device=device)
    n_probes = int(inputs.shape[0])
    n_columns = 0
    rows: Optional[Tensor] = None

    for step in iter_probe_columns(model, inputs, targets, source=source, modules=selected,
                                   loss=loss, k=k, batch_size=batch_size, dtype=dtype,
                                   device=device, generator=generator,
                                   check_independence=check_independence):
        n_columns = step.n_columns
        if rows is None or rows.shape[0] != step.n_examples:
            rows = torch.empty(step.n_examples, layout.P, dtype=dtype, device=device)
        written = []
        for layer in step.capturer:
            for p_name, gradient in per_sample_gradients(layer).items():
                full = f"{layer.name}.{p_name}" if layer.name else p_name
                rows[:, layout.slices[full]] = gradient.reshape(step.n_examples, -1)
                written.append(full)
        if set(written) != set(layout.names):
            raise RuntimeError(
                "the capture did not fill every column of U — missing "
                f"{sorted(set(layout.names) - set(written))}, unexpected "
                f"{sorted(set(written) - set(layout.names))}"
            )
        matrix.addmm_(rows.T, rows)

    matrix /= n_probes
    symmetrize_(matrix)
    return DenseReference(matrix=matrix, layout=layout, source=source, loss=loss,
                          n_probes=n_probes, n_columns=n_columns, probe_digest=probe_digest)
