"""Per-example layer statistics: the inputs ``a``, the output gradients ``g``, and the *normalised*
input ``x_hat`` of a normalisation layer (``docs/reports/plan_exp_draft.md`` §7, lot 1 of its §9).

Everything the campaign builds — the exact references of regime A, the per-layer Grams of regime B,
the whole approximation zoo — is a contraction of ``(a, g)`` with the **example axis intact**. One
forward pass captures ``a``; one backward pass per root column of ``Lambda_n`` (``sources.py``)
captures ``g``, and the ``N`` probes of a batch share that pass (``plan_exp_draft.md`` §2.1).

Three things here are deliberate, and each of them was a bug first.

1. **No module backward hook.** ``register_full_backward_hook`` fires from the module's *input*-side
   node, so a module whose ``grad_input`` is not needed by the requested ``inputs=`` is pruned out
   of the backward graph and its hook silently never fires. Measured on a
   ``Linear -> LayerNorm -> ReLU -> Linear`` in fp64: ``autograd.grad(out, params, grad_outputs=V)``
   fires all three modules when nothing requires grad (with a ``UserWarning``), but only the last
   two as soon as the input requires grad. On ``mlp_ln_mnist`` the missing module is the first
   ``Linear``, i.e. **25 120 of 26 634 parameters** — whose ``U`` columns would simply be zero, with
   ``F`` still symmetric PSD and every exactness test still passing on the layers that did fire.
   So ``g`` is taken from a **tensor** hook on the module's output (``output.register_hook``), which
   is exactly ``grad_output[0]``, fires unconditionally, and survives both an in-place
   ``ReLU(inplace=True)`` on that output and a residual reuse of it (both checked). This is a
   deliberate divergence from ``adafisher_modes/optimizer.py``'s ``_save_grad_output``; do not
   "align" it back.
2. **``x_hat`` is recomputed, not hooked.** ``d/d gamma = sum_t g_t * x_hat_t`` needs the
   *normalised* activation, while a forward hook on ``nn.LayerNorm`` / ``nn.BatchNorm2d`` /
   ``nn.GroupNorm`` sees the input **before** normalisation. In :func:`~fisher_ref.conventions.
   reference_mode` (eval) all three are closed-form from the captured input plus the module's own
   ``eps`` and buffers, and the result is exact: ``(g * x_hat).sum(t)`` reproduces ``weight.grad``
   to ``1e-15`` on LayerNorm (2-D and 3-D input), BatchNorm2d-eval and GroupNorm. A ``BatchNorm*``
   in **train** mode is refused: ``x_hat`` would then depend on the rest of the batch and per-sample
   gradients would not exist at all (``plan_exp_draft.md`` §2.5).
3. **``a`` is not bias-augmented.** The column layout of ``U`` is ``model.named_parameters()``
   order (see ``reference/dense.py``), in which the bias gradient is ``g.sum(t)`` in its own slice.
   The appended ones column of ``factors.augment_*`` belongs to the ``rvec([W | b])`` layout, which
   this package does not use; ``factors.extract_patches`` is the one helper reused from there.
"""

from __future__ import annotations

import copy
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # the editable install's .pth is inert in this sandbox (CLAUDE.md)
    sys.path.insert(0, str(_SRC))

from adafisher_modes.factors import extract_patches  # noqa: E402

from .conventions import assert_sample_independent, reference_mode  # noqa: E402

KINDS: Tuple[str, ...] = ("linear", "conv", "norm")

_CHANNEL_FIRST_NORMS = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)


# ----------------------------------------------------------------------------------------------
# Which modules can be captured, and as what
# ----------------------------------------------------------------------------------------------


def layer_kind(module: nn.Module) -> Optional[str]:
    """``"linear"`` / ``"conv"`` / ``"norm"``, or ``None`` when the module carries no statistic.

    ``None`` covers both "not a parametrised layer" (``ReLU``, ``Flatten``) and "parametrised but
    with no own weight" (``LayerNorm(elementwise_affine=False)``): neither contributes a column to
    ``U``.
    """
    if not any(True for _ in module.parameters(recurse=False)):
        return None
    if isinstance(module, nn.Linear):
        return "linear"
    if isinstance(module, nn.Conv2d):
        return "conv"
    if isinstance(module, (nn.LayerNorm, *_CHANNEL_FIRST_NORMS)):
        return "norm"
    return None


def capturable_modules(model: nn.Module) -> Dict[str, nn.Module]:
    """Every module of ``model`` whose per-sample gradient this package can reconstruct."""
    return {name: module for name, module in model.named_modules()
            if layer_kind(module) is not None}


def coverage(model: nn.Module) -> Tuple[List[str], List[str]]:
    """``(covered, uncovered)`` parameter names, in ``named_parameters()`` order.

    "Uncovered" is not a detail: those columns of ``U`` would be identically zero, leaving ``F``
    symmetric, PSD and quietly wrong. The dense builder raises on a non-empty second list rather
    than emitting them. The known cases are the raw parameters of the transformer models
    (``pos_embed``, ``cls_token``), which ``registry.classify`` also reports as ``hooked=False``.
    """
    covered_set: set = set()
    for name, module in capturable_modules(model).items():
        prefix = f"{name}." if name else ""
        covered_set.update(prefix + p_name for p_name, _ in module.named_parameters(recurse=False))
    names = [name for name, _ in model.named_parameters()]
    return ([n for n in names if n in covered_set], [n for n in names if n not in covered_set])


# ----------------------------------------------------------------------------------------------
# x_hat, in closed form
# ----------------------------------------------------------------------------------------------


def normalized_input(module: nn.Module, x: Tensor) -> Tensor:
    """The normalised activation ``x_hat`` a normalisation layer multiplies by ``gamma``.

    Closed form from the module's own statistics, because a forward hook sees the *pre*-normalised
    input. Only defined when the layer's statistics do not mix examples, which is why a
    ``BatchNorm*`` in train mode is refused rather than approximated.
    """
    if isinstance(module, nn.LayerNorm):
        dims = tuple(range(x.ndim - len(module.normalized_shape), x.ndim))
        mean = x.mean(dims, keepdim=True)
        variance = x.var(dims, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(variance + module.eps)
    if isinstance(module, nn.GroupNorm):
        grouped = x.reshape(x.shape[0], module.num_groups, -1)
        mean = grouped.mean(-1, keepdim=True)
        variance = grouped.var(-1, unbiased=False, keepdim=True)
        return ((grouped - mean) / torch.sqrt(variance + module.eps)).reshape(x.shape)
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if module.training:
            raise RuntimeError(
                f"{type(module).__name__} is in train mode: x_hat would depend on the rest of the "
                "batch, so per-sample gradients — and therefore F and E_hat — are not defined. Use "
                "fisher_ref.conventions.reference_mode (plan_exp_draft.md §2.5)."
            )
        if module.running_mean is None or module.running_var is None:
            raise RuntimeError(
                f"{type(module).__name__} has track_running_stats=False, so in eval mode it still "
                "normalises with batch statistics and mixes examples (plan_exp_draft.md §2.5)."
            )
        shape = (1, -1) + (1,) * (x.ndim - 2)
        mean = module.running_mean.reshape(shape).to(x.dtype)
        variance = module.running_var.reshape(shape).to(x.dtype)
        return (x - mean) / torch.sqrt(variance + module.eps)
    raise NotImplementedError(
        f"no closed-form x_hat for {type(module).__name__}; the campaign's models use LayerNorm, "
        "BatchNorm2d and GroupNorm (plan_exp_draft.md §1)"
    )


# ----------------------------------------------------------------------------------------------
# Pooling onto (N, T, d), example axis first
# ----------------------------------------------------------------------------------------------


def _check_conv2d(module: nn.Conv2d) -> None:
    """The same scope as ``factors._check_conv2d_supported``, restated locally rather than importing
    a private name from a module pinned bit-exact by ``tests/test_diag_bitexact.py``.
    """
    if module.groups != 1:
        raise NotImplementedError(
            f"per-sample gradients from patches assume groups=1; got groups={module.groups}. A "
            "grouped/depthwise layer needs a per-group patch extraction."
        )
    if module.dilation != (1, 1):
        raise NotImplementedError(
            f"extract_patches has no dilation parameter; got dilation={module.dilation}."
        )


def _pool_channel_first(x: Tensor) -> Tensor:
    """``(N, C, *spatial) -> (N, T, C)``, ``T`` row-major over the spatial axes."""
    return x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)


def pool_input(module: nn.Module, kind: str, x: Tensor) -> Tensor:
    """The layer's input as ``(N, T, d_in)`` — for a norm layer, its ``x_hat``."""
    if kind == "linear":
        return x.reshape(x.shape[0], -1, x.shape[-1])
    if kind == "conv":
        _check_conv2d(module)
        patches = extract_patches(x, module.kernel_size, module.stride, module.padding,
                                  module.groups)
        return patches.reshape(x.shape[0], patches.shape[2], patches.shape[3])
    normalized = normalized_input(module, x)
    if isinstance(module, nn.LayerNorm):
        return normalized.reshape(normalized.shape[0], -1, normalized.shape[-1])
    return _pool_channel_first(normalized)


def pool_output_grad(module: nn.Module, kind: str, g: Tensor) -> Tensor:
    """The gradient at the layer's output as ``(N, T, d_out)``, row-aligned with :func:`pool_input`.

    For ``conv`` that alignment is ``extract_patches``' own: its ``T`` axis is row-major over
    ``(H_out, W_out)``, which is what ``_pool_channel_first`` produces here.
    """
    if kind == "linear":
        return g.reshape(g.shape[0], -1, g.shape[-1])
    if kind == "conv":
        return _pool_channel_first(g)
    if isinstance(module, nn.LayerNorm):
        return g.reshape(g.shape[0], -1, g.shape[-1])
    return _pool_channel_first(g)


# ----------------------------------------------------------------------------------------------
# Capture
# ----------------------------------------------------------------------------------------------


@dataclass
class CapturedLayer:
    """One layer's statistics for one probe batch and one backprop column."""

    name: str
    module: nn.Module
    kind: str
    a: Tensor   # (N, T, d_in); x_hat for a normalisation layer
    g: Tensor   # (N, T, d_out), from the most recent backward

    @property
    def n_examples(self) -> int:
        return int(self.a.shape[0])

    @property
    def positions(self) -> int:
        return int(self.a.shape[1])


class Capture:
    """Context manager registering the forward (and, through it, tensor) hooks.

    One forward fills ``a``; every backward overwrites ``g``, so the driver loops over the root's
    columns re-using a single forward (``retain_graph=True`` on all but the last).
    """

    def __init__(self, model: nn.Module,
                 modules: Optional[Mapping[str, nn.Module]] = None) -> None:
        self.model = model
        self.modules: Dict[str, nn.Module] = (
            dict(modules) if modules is not None else capturable_modules(model)
        )
        self.kinds: Dict[str, str] = {}
        for name, module in self.modules.items():
            kind = layer_kind(module)
            if kind is None:
                raise ValueError(f"module {name!r} ({type(module).__name__}) carries no statistic")
            self.kinds[name] = kind
        self._inputs: Dict[str, Tensor] = {}
        self._grads: Dict[str, Tensor] = {}
        self._calls: Dict[str, int] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # -- lifecycle ------------------------------------------------------------------------------

    def _make_hook(self, name: str):
        kind = self.kinds[name]

        def forward_hook(module: nn.Module, inputs: Tuple, output: Tensor) -> None:
            if not torch.is_grad_enabled():  # a no_grad probe forward captures nothing
                return
            self._calls[name] = self._calls.get(name, 0) + 1
            if self._calls[name] > 1:
                raise RuntimeError(
                    f"module {name!r} was called {self._calls[name]} times in one forward pass; a "
                    "reused module overwrites its own statistic, and its per-sample gradient is a "
                    "sum over calls this package does not form (plan_exp_draft.md §2.5)."
                )
            with torch.no_grad():
                self._inputs[name] = pool_input(module, kind, inputs[0].detach())

            def tensor_hook(grad: Tensor) -> None:
                self._grads[name] = pool_output_grad(module, kind, grad.detach())

            output.register_hook(tensor_hook)

        return forward_hook

    def __enter__(self) -> "Capture":
        for name, module in self.modules.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self) -> None:
        """Drop everything captured so far — call between probe micro-batches."""
        self._inputs.clear()
        self._grads.clear()
        self._calls.clear()

    # -- access ---------------------------------------------------------------------------------

    def layer(self, name: str) -> CapturedLayer:
        if name not in self._inputs:
            raise KeyError(f"no forward captured for {name!r}; did the module run under grad?")
        if name not in self._grads:
            raise KeyError(f"no backward captured for {name!r}; run a backward pass first")
        return CapturedLayer(name=name, module=self.modules[name], kind=self.kinds[name],
                             a=self._inputs[name], g=self._grads[name])

    def __iter__(self) -> Iterator[CapturedLayer]:
        return (self.layer(name) for name in self.modules)

    def __len__(self) -> int:
        return len(self.modules)


@contextmanager
def capture(model: nn.Module,
            modules: Optional[Mapping[str, nn.Module]] = None) -> Iterator[Capture]:
    """Functional spelling of :class:`Capture`."""
    with Capture(model, modules) as capturer:
        yield capturer


# ----------------------------------------------------------------------------------------------
# Per-sample gradients
# ----------------------------------------------------------------------------------------------


@dataclass
class ProbeColumn:
    """One ``(micro-batch, root column)`` step of a traversal, with the capture live.

    ``capturer`` holds every selected layer's ``(a, g)`` for exactly the examples ``start:stop`` and
    exactly this column of ``Lambda^{1/2}``; it is reused on the next iteration, so a consumer that
    needs the tensors beyond the current step must copy them.
    """

    capturer: "Capture"
    start: int
    stop: int
    column: int
    n_columns: int
    outputs: Tensor

    @property
    def n_examples(self) -> int:
        return self.stop - self.start


def prepare_model(model: nn.Module, dtype: torch.dtype, device: object) -> nn.Module:
    """``model`` in ``dtype`` on ``device``, deep-copied only when a cast is actually needed.

    A reader must not mutate the caller's model, and every consumer of :func:`iter_probe_columns`
    must hand it a model already in the reference dtype — the traversal casts the *batch*, not the
    network, so a mismatch surfaces as a bare ``mat1 and mat2 must have the same dtype`` from deep
    inside ``nn.Linear``. Prepare once and pass the same object to the reference builder and to the
    factor accumulators, which also guarantees they see the same parameters
    (``plan_exp_lot2.md`` §0.2).
    """
    parameter = next(model.parameters(), None)
    if parameter is not None and parameter.dtype == dtype and str(parameter.device) == str(device):
        return model
    return copy.deepcopy(model).to(device=device, dtype=dtype)


def iter_probe_columns(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    source: str,
    modules: Mapping[str, nn.Module],
    loss: str = "cross_entropy",
    k: int = 1,
    batch_size: int = 256,
    dtype: torch.dtype = torch.float64,
    device: object = "cpu",
    generator: Optional[torch.Generator] = None,
    check_independence: bool = True,
) -> Iterator[ProbeColumn]:
    """Walk the probe set once, yielding the live capture per ``(micro-batch, root column)``.

    The single definition of "how a reference or a structure is accumulated": one forward per
    micro-batch under :func:`~fisher_ref.conventions.reference_mode`, one backward per column of the
    root, seeded with ``inputs=[batch]`` — never ``inputs=params``, which silently prunes the first
    module's statistic (this module's header, point 1).

    It is a generator because two consumers need it and one of them needs it **twice**: EKFAC's
    eigenvalues ``s_ij`` can only be accumulated once ``Q_A`` and ``Q_G`` are known, i.e. after a
    first pass has built ``A`` and ``G``. And because both consumers must see the *same* probes and
    the same Monte-Carlo draws — with two independent loops, ``‖R − K‖`` would carry a sampling
    difference between reference and approximation that no metric could tell from structure error
    (``plan_exp_lot2.md`` §0.2).
    """
    from .sources import output_root  # noqa: PLC0415 - avoids a cycle at import time

    parameter = next(model.parameters(), None)
    if parameter is not None and parameter.dtype != dtype:
        raise TypeError(
            f"the model is {parameter.dtype} but the traversal runs in {dtype}: call "
            "capture.prepare_model(model, dtype, device) once and pass the result to every "
            "consumer, so the reference and the structures share one network."
        )
    n_probes = int(inputs.shape[0])
    checked = not check_independence
    with reference_mode(model), Capture(model, modules) as capturer:
        for start in range(0, n_probes, batch_size):
            stop = min(start + batch_size, n_probes)
            batch = inputs[start:stop].to(device=device, dtype=dtype).detach().requires_grad_(True)
            batch_targets = targets[start:stop].to(device=device)
            if not checked and stop - start > 2:
                assert_sample_independent(model, batch.detach(), k=2)
                checked = True

            capturer.reset()
            outputs = model(batch)
            root = output_root(outputs.detach(), batch_targets, source=source, loss=loss, k=k,
                               generator=generator)
            for column in range(root.n_columns):
                torch.autograd.grad(
                    outputs, [batch], grad_outputs=root.column(column),
                    retain_graph=(column < root.n_columns - 1),
                )
                yield ProbeColumn(capturer=capturer, start=start, stop=stop, column=column,
                                  n_columns=root.n_columns, outputs=outputs)


def per_sample_gradients(layer: CapturedLayer) -> Dict[str, Tensor]:
    """``{"weight": (N, *weight.shape), "bias": (N, *bias.shape)}`` for one captured layer.

    ``linear`` / ``conv`` contract the position axis into an outer product,
    ``G_n = sum_t g_{n,t} a_{n,t}^T``. A normalisation layer does **not**: its exact per-sample
    gradients are ``d/d gamma = sum_t g_t * x_hat_t`` and ``d/d beta = sum_t g_t``, element-wise
    (``plan_exp_draft.md`` §4, "Normalisations") — which is the same statement as ``plan_lot5.md``
    §0.1's "Hadamard-, not Kronecker-structured".
    """
    a, g = layer.a, layer.g
    if a.shape[:2] != g.shape[:2]:
        raise RuntimeError(
            f"{layer.name!r}: input and gradient disagree on (N, T) — {tuple(a.shape)} vs "
            f"{tuple(g.shape)}"
        )
    module = layer.module
    gradients: Dict[str, Tensor] = {}
    if layer.kind == "norm":
        gradients["weight"] = (g * a).sum(dim=1)
    else:
        weight_grad = torch.einsum("ntd,nte->nde", g, a)
        if layer.kind == "conv":
            weight_grad = weight_grad.reshape(weight_grad.shape[0], *module.weight.shape)
        gradients["weight"] = weight_grad
    if getattr(module, "bias", None) is not None:
        gradients["bias"] = g.sum(dim=1)
    return gradients
