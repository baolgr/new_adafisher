"""Probe sets: the fixed, augmentation-free, versioned inputs every reference is computed on
(``docs/reports/plan_exp_draft.md`` §1.1, §3.4; lot 0 of its §9).

A probe set is an *invariant* of the campaign (``plan_exp_draft.md`` §8): change it and every
earlier comparison is void. Hence three properties, each of them the reason for a design choice:

* **No augmentation.** ``benchmarks/common/data.py::build_loaders`` applies the *train* transform
  to the train split, and on CIFAR that is ``RandomCrop(32, padding=4)`` + ``RandomHorizontalFlip``
  + optional ``Cutout`` — i.e. a probe built through it would be a different image on every call.
  Probes are built from the **eval** transform on both splits.
* **The run's own split.** The train/val partition is re-derived with the same
  ``seeded_train_val_split(n_total, val_size, seed)`` the training run used, so a "train probe" is a
  point that run actually trained on and a "val probe" is one it never saw — which is the whole
  content of HF1.
* **Hashed over its content.** The digest covers the tensor bytes, not just the recipe: a
  torchvision transform change or a different copy of the dataset on the cluster would otherwise
  pass unnoticed (``plan_exp_lot0.md`` §0.4).

Probes are the **first ``n`` indices** of the seeded permutation's split, so a 256-probe set is a
prefix of a 1 000-probe set (:meth:`ProbeSet.head`). That is what makes the ``N' < N`` noise-floor
curve of §3.4 free rather than a separate sweep.

The dataset is not re-declared here: each bench's ``build_data`` is a
``functools.partial(build_loaders, SPEC)``, and :func:`dataset_spec_of` reads the spec off it.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
from torch import Tensor

from benchmarks.common.data import DatasetSpec, build_transforms, seeded_train_val_split
from benchmarks.common.runner import BENCHMARKS_DIR, Benchmark

PROBE_VERSION = "probes/1"
DEFAULT_DATA_ROOT = BENCHMARKS_DIR / "data"
SPLITS = ("train", "val")


def dataset_spec_of(bench: Benchmark) -> DatasetSpec:
    """The ``DatasetSpec`` a bench's ``build_data`` is bound to."""
    builder: Any = bench.build_data
    if isinstance(builder, functools.partial) and builder.args:
        candidate = builder.args[0]
        if isinstance(candidate, DatasetSpec):
            return candidate
    raise TypeError(
        f"cannot recover the DatasetSpec of bench {bench.name!r}: its build_data is {builder!r}, "
        "not a functools.partial(build_loaders, SPEC). Either bind it that way (as "
        "benchmarks/common/data.py does for mnist/cifar10/cifar100) or pass spec= explicitly."
    )


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part)
        hasher.update(b"|")
    return hasher.hexdigest()


@dataclass(frozen=True)
class ProbeSet:
    """``N`` fixed inputs and their true labels, with everything needed to reproduce them."""

    model: str
    dataset: str
    split: str
    seed: int
    indices: Tuple[int, ...]
    inputs: Tensor
    targets: Tensor
    digest: str
    version: str = PROBE_VERSION

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def as_model_batch(self, bench: Benchmark) -> Tuple[Tensor, Tensor]:
        """``(model input, loss target)`` — the bench's own ``prepare_batch`` applied, so the
        auto-encoder's reconstruction target is handled without a special case here.
        """
        return bench.prepare_batch((self.inputs, self.targets))

    def to(self, device: Any = None, dtype: Any = None) -> "ProbeSet":
        inputs = self.inputs.to(device=device, dtype=dtype) if dtype else self.inputs.to(device)
        return ProbeSet(self.model, self.dataset, self.split, self.seed, self.indices,
                        inputs, self.targets.to(device), self.digest, self.version)

    def head(self, n: int) -> "ProbeSet":
        """The first ``n`` probes — a valid probe set, for §3.4's ``N' < N`` noise-floor curve."""
        if n > len(self):
            raise ValueError(f"cannot take {n} probes from a set of {len(self)}")
        return _finalize(self.model, self.dataset, self.split, self.seed,
                         self.indices[:n], self.inputs[:n], self.targets[:n])

    def metadata(self) -> dict:
        return {
            "probe_version": self.version, "probe_digest": self.digest, "model": self.model,
            "dataset": self.dataset, "split": self.split, "seed": self.seed, "n": len(self),
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model, "dataset": self.dataset, "split": self.split,
                    "seed": self.seed, "indices": list(self.indices), "inputs": self.inputs,
                    "targets": self.targets, "digest": self.digest, "version": self.version}, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ProbeSet":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        probes = cls(payload["model"], payload["dataset"], payload["split"], int(payload["seed"]),
                     tuple(int(i) for i in payload["indices"]), payload["inputs"],
                     payload["targets"], payload["digest"], payload["version"])
        recomputed = _content_digest(probes.model, probes.dataset, probes.split, probes.seed,
                                     probes.indices, probes.inputs, probes.targets)
        if recomputed != probes.digest:
            raise ValueError(
                f"probe set at {path} fails its own digest ({recomputed} != {probes.digest}): the "
                "file has been modified or was written by another version."
            )
        return probes


def _content_digest(model: str, dataset: str, split: str, seed: int,
                    indices: Sequence[int], inputs: Tensor, targets: Tensor) -> str:
    return _digest(
        PROBE_VERSION.encode(), model.encode(), dataset.encode(), split.encode(),
        str(seed).encode(), str(list(indices)).encode(),
        inputs.contiguous().cpu().numpy().tobytes(),
        targets.contiguous().cpu().numpy().tobytes(),
    )


def _finalize(model: str, dataset: str, split: str, seed: int, indices: Sequence[int],
              inputs: Tensor, targets: Tensor) -> ProbeSet:
    indices = tuple(int(i) for i in indices)
    return ProbeSet(model, dataset, split, seed, indices, inputs, targets,
                    _content_digest(model, dataset, split, seed, indices, inputs, targets))


def build_probe_set(
    bench: Benchmark,
    *,
    split: str,
    n: int,
    seed: int = 0,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    allow_download: bool = False,
    spec: Optional[DatasetSpec] = None,
) -> ProbeSet:
    """``n`` probes from ``split`` of the run's own seeded partition, eval transform only.

    ``allow_download`` is off by default, as everywhere in ``benchmarks/`` (compute nodes have no
    internet, and a missing dataset must be a clear error rather than a network timeout).
    ``spec`` overrides the dataset, which is how the offline tests inject a synthetic one.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
    spec = spec or dataset_spec_of(bench)
    _, eval_transform = build_transforms(spec, cutout=False, cutout_length=16)
    dataset = spec.dataset_cls(root=str(Path(data_root).expanduser()), train=True,
                               download=allow_download, transform=eval_transform)
    train_idx, val_idx = seeded_train_val_split(len(dataset), spec.val_size, seed)
    pool = train_idx if split == "train" else val_idx
    if n > len(pool):
        raise ValueError(f"asked for {n} probes but the {split} split holds {len(pool)}")
    indices = pool[:n]

    samples = [dataset[i] for i in indices]
    inputs = torch.stack([sample[0] for sample in samples])
    targets = torch.as_tensor([sample[1] for sample in samples])
    return _finalize(bench.name, spec.name, split, seed, indices, inputs, targets)


def build_probe_pair(bench: Benchmark, *, n: int, seed: int = 0, **kwargs: Any
                     ) -> Tuple[ProbeSet, ProbeSet]:
    """``(train probes, val probes)`` of the same size, seed and partition — HF1's two sets."""
    return (build_probe_set(bench, split="train", n=n, seed=seed, **kwargs),
            build_probe_set(bench, split="val", n=n, seed=seed, **kwargs))
