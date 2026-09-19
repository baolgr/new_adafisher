"""Layer-type registry: what kind of parameter block each part of a benchmark network is.

Every per-layer-type conclusion the campaign draws is an aggregation over this classification, so
it has to be explicit, exhaustive and checked. Two properties matter more than the taxonomy itself.

* **It partitions the parameters.** Every ``named_parameter`` of a model belongs to exactly one
  record, including the ones no optimizer hook sees (``pos_embed``, ``cls_token``, a ``GroupNorm``
  affine). Those are marked ``hooked=False`` and still classified, because the campaign measures
  *curvature*, not what the optimizer happens to precondition. :func:`assert_partitions` is the
  check.
* **Weight sharing is a runtime property, not a class property.** ``nn.Linear`` is the same class
  whether its input is ``(N, 784)`` (one position per example) or ``(N, T, 32)`` (shared over ``T``
  tokens), and the distinction decides which K-FAC variant even applies (Eschenhagen et al.,
  arXiv:2311.00636 §3.2-§3.3). It is therefore read from **one forward pass**, not guessed from the
  module class: pass ``example_input`` to :func:`classify` to get it.

``head`` is assigned to the module whose output *is* the model's output, by tensor identity rather
than by shape. For an auto-encoder that is the last decoder ``Linear``.

Public API
----------

:data:`LAYER_TYPES`  the nine labels a block can carry.

:class:`LayerInfo`  one parameter block: its name, module type, layer type, the fully-qualified
names of its parameters, their count, whether an ``AdaFisherMulti`` hook sees it, the rank of its
input tensor, the number of shared positions ``T``, and whether it produced the model's output.

:func:`classify`  the classification of a whole model.

:func:`by_type`, :func:`type_counts`, :func:`unhooked_parameters`, :func:`assert_partitions`
aggregations and the partition check.

:func:`hooked_module_types`  the module types ``AdaFisherMulti`` hooks, read from the optimizer
itself when it can be imported and from a local fallback tuple otherwise.

Dependencies: :mod:`fisher_ref.conventions` (for ``reference_mode`` during the shape-probing
forward) and, optionally, ``adafisher_modes.optimizer``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .conventions import reference_mode

#: The AdaFisherMulti module types (``adafisher_modes.optimizer.SUPPORTED_MODULES``), duplicated
#: here as a fallback only; ``hooked_module_types()`` prefers the real thing.
_FALLBACK_SUPPORTED = ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm")

LAYER_TYPES: Tuple[str, ...] = (
    "linear",         # Linear, unshared input (rank-2 input)
    "linear_shared",  # Linear applied at several positions (tokens, sequence)
    "conv",           # Conv1d/2d/3d
    "norm",           # BatchNorm*, LayerNorm, GroupNorm, InstanceNorm*, RMSNorm
    "embed",          # nn.Embedding, and raw positional / class-token parameters
    "lora_a",
    "lora_b",
    "head",           # the module whose output is the model's output
    "other",
)

_NORM_MARKERS = ("norm",)  # LayerNorm, BatchNorm2d, GroupNorm, RMSNorm, InstanceNorm2d, ...
_EMBED_PARAM_MARKERS = ("pos_embed", "position_embed", "cls_token", "class_token", "embed")


def hooked_module_types() -> Tuple[str, ...]:
    try:
        from adafisher_modes.optimizer import SUPPORTED_MODULES  # noqa: PLC0415
    except Exception:  # pragma: no cover - only when adafisher_modes is not importable
        return _FALLBACK_SUPPORTED
    return tuple(SUPPORTED_MODULES)


@dataclass(frozen=True)
class LayerInfo:
    """One parameter block: a leaf module with its own parameters, or a raw ``Parameter``."""

    name: str                      # module name ("" for the root), or the parameter's own name
    module_type: str               # "Linear", "GroupNorm", ... or "Parameter"
    layer_type: str                # one of LAYER_TYPES
    params: Tuple[str, ...]        # fully-qualified parameter names
    n_params: int
    hooked: bool                   # is an AdaFisherMulti SUPPORTED_MODULES instance with a weight
    input_ndim: Optional[int] = None   # rank of the module's input tensor (needs example_input)
    positions: Optional[int] = None    # T: shared positions the weight is applied at
    is_output: bool = False            # its output is the model's output tensor

    @property
    def shared(self) -> Optional[bool]:
        """Whether the weight is applied at more than one position (``None`` = not probed)."""
        return None if self.positions is None else self.positions > 1


def _is_norm(module: nn.Module) -> bool:
    name = type(module).__name__.lower()
    return any(marker in name for marker in _NORM_MARKERS)


def _classify_module(name: str, module: nn.Module, *, input_ndim: Optional[int],
                     is_output: bool) -> str:
    lowered = name.lower()
    if "lora_a" in lowered or lowered.endswith(".lora_a"):
        return "lora_a"
    if "lora_b" in lowered or lowered.endswith(".lora_b"):
        return "lora_b"
    if isinstance(module, nn.Embedding):
        return "embed"
    if _is_norm(module):
        return "norm"
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return "conv"
    if isinstance(module, nn.Linear):
        if is_output:
            return "head"
        if input_ndim is not None and input_ndim > 2:
            return "linear_shared"
        return "linear"
    return "other"


def _classify_raw_parameter(name: str) -> str:
    lowered = name.lower()
    if "lora_a" in lowered:
        return "lora_a"
    if "lora_b" in lowered:
        return "lora_b"
    if any(marker in lowered for marker in _EMBED_PARAM_MARKERS):
        return "embed"
    return "other"


def _positions(input_shape: Sequence[int], module: nn.Module,
               output_shape: Optional[Sequence[int]]) -> Optional[int]:
    """Number of positions the weight is applied at, per example."""
    if isinstance(module, nn.Linear):
        return math.prod(input_shape[1:-1]) if len(input_shape) > 2 else 1
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and output_shape is not None:
        return math.prod(output_shape[2:]) if len(output_shape) > 2 else 1
    if _is_norm(module) and len(input_shape) > 2:
        # A normalisation's gamma/beta are per-channel and applied at every other position.
        if isinstance(module, nn.LayerNorm):
            normalized = len(module.normalized_shape)
            spatial = list(input_shape[1:len(input_shape) - normalized])
        else:  # BatchNorm2d / GroupNorm: (N, C, *spatial)
            spatial = list(input_shape[2:])
        return math.prod(spatial) if spatial else 1
    return 1


def _probe_shapes(model: nn.Module, example_input: Tensor) -> Tuple[Dict[str, Any], set]:
    """One forward pass under :func:`reference_mode`, recording per-module input/output shapes and
    which modules produced the model's own output tensor.
    """
    seen: Dict[str, Any] = {}
    outputs: Dict[str, Any] = {}
    handles = []

    def make_hook(module_name: str):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if module_name in seen:  # a reused module: first call wins, and it is flagged
                seen[module_name]["calls"] += 1
                return
            in_shape = tuple(inputs[0].shape) if inputs and isinstance(inputs[0], Tensor) else None
            out_shape = tuple(output.shape) if isinstance(output, Tensor) else None
            seen[module_name] = {"input_shape": in_shape, "output_shape": out_shape, "calls": 1}
            if isinstance(output, Tensor):
                outputs[module_name] = output
        return hook

    for module_name, module in model.named_modules():
        if module_name and not list(module.children()):
            handles.append(module.register_forward_hook(make_hook(module_name)))
    try:
        with reference_mode(model), torch.no_grad():
            model_output = model(example_input)
    finally:
        for handle in handles:
            handle.remove()
    produced_output = {
        module_name for module_name, tensor in outputs.items()
        if isinstance(model_output, Tensor) and tensor is model_output
    }
    return seen, produced_output


def classify(model: nn.Module, example_input: Optional[Tensor] = None) -> List[LayerInfo]:
    """Classify every parameter block of ``model``.

    Without ``example_input`` the shape-dependent fields stay ``None``: a shared ``Linear`` is
    reported as ``linear`` and no ``head`` is identified. Pass a real probe batch (already through
    the bench's ``prepare_batch``) for the full classification.
    """
    shapes: Dict[str, Any] = {}
    produced_output: set = set()
    if example_input is not None:
        shapes, produced_output = _probe_shapes(model, example_input)

    supported = hooked_module_types()
    infos: List[LayerInfo] = []
    for module_name, module in model.named_modules():
        own = [(p_name, p) for p_name, p in module.named_parameters(recurse=False)]
        if not own:
            continue
        prefix = f"{module_name}." if module_name else ""
        shape_info = shapes.get(module_name, {})
        input_shape = shape_info.get("input_shape")
        output_shape = shape_info.get("output_shape")
        input_ndim = len(input_shape) if input_shape else None
        is_output = module_name in produced_output

        is_leaf = not list(module.children())
        if is_leaf:
            layer_type = _classify_module(module_name, module, input_ndim=input_ndim,
                                          is_output=is_output)
            hooked = (type(module).__name__ in supported
                      and getattr(module, "weight", None) is not None)
            positions = (_positions(input_shape, module, output_shape)
                         if input_shape is not None else None)
            infos.append(LayerInfo(
                name=module_name, module_type=type(module).__name__, layer_type=layer_type,
                params=tuple(prefix + p_name for p_name, _ in own),
                n_params=sum(p.numel() for _, p in own), hooked=hooked,
                input_ndim=input_ndim, positions=positions, is_output=is_output,
            ))
            continue

        # A raw Parameter registered on a container module (pos_embed, cls_token): one record each,
        # since they are independent blocks with no shared statistic.
        for p_name, parameter in own:
            full = prefix + p_name
            infos.append(LayerInfo(
                name=full, module_type="Parameter", layer_type=_classify_raw_parameter(full),
                params=(full,), n_params=parameter.numel(), hooked=False,
            ))
    return infos


def by_type(infos: Iterable[LayerInfo]) -> Dict[str, List[LayerInfo]]:
    grouped: Dict[str, List[LayerInfo]] = {}
    for info in infos:
        grouped.setdefault(info.layer_type, []).append(info)
    return grouped


def type_counts(infos: Iterable[LayerInfo]) -> Dict[str, int]:
    return {key: len(value) for key, value in sorted(by_type(infos).items())}


def unhooked_parameters(infos: Iterable[LayerInfo]) -> List[str]:
    """Parameter names belonging to no ``AdaFisherMulti`` hooked module — they take the identity
    preconditioner, and the campaign measures their curvature anyway.
    """
    return [name for info in infos if not info.hooked for name in info.params]


def assert_partitions(model: nn.Module, infos: Iterable[LayerInfo]) -> None:
    """Every parameter of ``model`` appears in exactly one record. Raises with the offenders."""
    covered: List[str] = [name for info in infos for name in info.params]
    expected = [name for name, _ in model.named_parameters()]
    duplicates = sorted({name for name in covered if covered.count(name) > 1})
    missing = sorted(set(expected) - set(covered))
    extra = sorted(set(covered) - set(expected))
    if duplicates or missing or extra:
        raise AssertionError(
            f"registry does not partition the parameters: missing={missing} "
            f"duplicated={duplicates} unknown={extra}"
        )
