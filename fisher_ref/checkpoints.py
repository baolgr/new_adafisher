"""The bridge to ``benchmarks/outputs/``: which θ exist, and how to load one
(``docs/reports/plan_exp_draft.md`` §3.5; lot 0 of its §9).

The campaign does not train anything. Its θ axis is the trajectory checkpoints the benchmark
harness already writes, in three layouts:

```
benchmarks/outputs/<model>/<arm>/ckpt_{0,0.01,0.1,0.5,1}.pt          # seed 0, up to 7 arms
benchmarks/outputs/<group>/<model>/<arm>/ckpt_*.pt                   # ... grouped by dataset
benchmarks/outputs/seeds/<model>/seed<n>/<arm>/ckpt_*.pt             # extra seeds, diag + adamw
```

``<group>`` is a bench's ``output_group`` (``mnist``, ``cifar10``, ``cifar100``, ``imagenet``), and
only a directory named after a *declared* group is descended into — never a bare guess, so
``_calibration`` and legacy trees such as ``outputs/lot8_cifar10/`` cannot be mistaken for one.
Both layouts coexist on purpose: a model's results do not move when its bench gains a group.

Three rules, all of them consequences of the wall-clock-time protocol, and all enforced here rather
than left to each consumer:

1. **A fraction is relative to the *nominal* trajectory** (``--epochs`` x batches), shared by every
   arm of a model, so ``ckpt_0.5`` is the same amount of *training* in every arm — but not the same
   wall-clock time, and not the same position inside an arm's own run. Read ``progress`` (=
   ``step / total_steps``) when an x-axis has to be honest.
2. **``scheduled=False`` means "the arm's own end"**, not the nominal 100 %: a budgeted arm that
   stopped short has its largest fraction pinned there by ``CheckpointWriter.save_final``.
3. **Pre-fix payloads are rejected.** Every checkpoint written before the campaign-1 denominator
   fix used ``max_epochs`` as the denominator, i.e. is mislabelled on every arm but ``diag``
   (``CLAUDE.md``'s status block). Such a payload has neither ``total_steps`` nor ``scheduled``;
   ``load_theta`` refuses it unless ``strict=False``, which then marks the record
   ``mislabelled=True``.

The network is rebuilt through its own ``bench.build_model``, with the architecture variants
(``cnn_gn_cifar``'s ``--norm``) taken from the run's ``manifest.json`` — so a ``bn`` run cannot be
silently reloaded into a ``gn`` network. Which bench that is comes from the manifest too
(``config["model"]``), **not** from the run directory's name: ``--output-dir`` is free-form, and
A2's BatchNorm variant lives in ``outputs/cifar10/cnn_gn_cifar_bn/`` with no folder of that name
``benchmarks/``. :attr:`RunRef.model` remains the directory label a consumer groups by;
:attr:`RunRef.bench_name` is what rebuilds the network.

A run launched with ``--checkpoint-optimizer-state`` also carries the optimizer's own state
(``LoadedCheckpoint.optimizer_state``: its ``state_dict`` plus, for ``AdaFisherMulti``, the EMA'd
Fisher factors keyed by module name). It is ``None`` everywhere today — the flag is off by default
and no campaign has used it yet — so P2 re-warms from θ (§3.2) unless a future run opted in.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from benchmarks.common.optimizers import ARMS
from benchmarks.common.runner import BENCHMARKS_DIR, Benchmark, discover_benchmarks

DEFAULT_OUTPUTS_ROOT = BENCHMARKS_DIR / "outputs"
SEEDS_DIRNAME = "seeds"


def _output_groups() -> frozenset:
    """The ``output_group`` values the benches declare — the only directory names ``discover_runs``
    will treat as a dataset grouping rather than as a model.
    """
    if not _BENCH_CACHE:
        _BENCH_CACHE.update(discover_benchmarks())
    return frozenset(b.output_group for b in _BENCH_CACHE.values() if b.output_group)

_CKPT_RE = re.compile(r"^ckpt_(?P<fraction>[0-9]*\.?[0-9]+)\.pt$")
_SEED_RE = re.compile(r"^seed(?P<seed>\d+)$")
#: Keys only the post-denominator-fix writer emits (``plan_exp_draft.md`` §3.5, rule 3).
REQUIRED_PAYLOAD_KEYS = ("total_steps", "scheduled")

#: Run directories whose name is not a ``benchmarks/<name>/`` folder, written before
#: ``manifest.json`` recorded the bench that produced it. ``cnn_gn_cifar_bn`` is A2's BatchNorm
#: variant — ``--norm bn --output-dir outputs/cifar10/cnn_gn_cifar_bn`` — and it is precisely the
#: protocol P2 (``plan_exp_draft.md`` §7) compares the GroupNorm one against, so leaving its 28
#: checkpoints unloadable was not an option. Its manifest already carries ``norm: "bn"``, so
#: :meth:`RunRef.model_kwargs` rebuilds it correctly once the *bench* is resolved.
#:
#: **New runs need no entry here**: ``runner.main`` writes ``config["model"]``. This map is a
#: compatibility shim for result trees that already exist, not a naming convention to extend — a
#: new architecture variant is a ``model_choices`` entry on its bench, not a new directory name.
_LEGACY_BENCH_OF_DIRECTORY = {"cnn_gn_cifar_bn": "cnn_gn_cifar"}


@dataclass(frozen=True)
class RunRef:
    """One ``(model, arm, seed)`` run directory and the checkpoints it holds."""

    model: str
    arm: str
    seed: int
    directory: Path
    checkpoints: Dict[float, Path] = field(default_factory=dict)
    manifest: Optional[Mapping[str, Any]] = None

    @property
    def fractions(self) -> Tuple[float, ...]:
        return tuple(sorted(self.checkpoints))

    @property
    def config(self) -> Mapping[str, Any]:
        return ((self.manifest or {}).get("config", {}) or {}) if self.manifest else {}

    @property
    def bench_name(self) -> str:
        """The ``benchmarks/<name>/bench.py`` that produced this run.

        **Not** the same thing as :attr:`model`, which is the run *directory*'s name and stays the
        label a consumer groups and plots by (``cnn_gn_cifar`` and ``cnn_gn_cifar_bn`` are two
        different runs and must not collapse into one). ``--output-dir`` is free-form, so the
        directory name is a label, never an identifier: the manifest is the authority, and
        :data:`_LEGACY_BENCH_OF_DIRECTORY` covers the trees written before it recorded this.
        """
        recorded = self.config.get("model")
        if recorded:
            return str(recorded)
        return _LEGACY_BENCH_OF_DIRECTORY.get(self.model, self.model)

    def model_kwargs(self) -> Dict[str, Any]:
        """The architecture variants this run was built with, read from its manifest."""
        bench = _bench(self.bench_name)
        return {key: self.config[key] for key in bench.model_choices if key in self.config}

    def summary(self) -> Mapping[str, Any]:
        """This arm's entry in the run's ``manifest.json`` (steps, epochs, test accuracy, ...)."""
        return ((self.manifest or {}).get("arms", {}) or {}).get(self.arm, {})


@dataclass
class LoadedCheckpoint:
    """A rebuilt network at one checkpoint, with everything needed to label it."""

    model: nn.Module
    run: RunRef
    fraction: float
    step: int
    epoch: int
    total_steps: Optional[int]
    scheduled: bool
    mislabelled: bool
    path: Path
    #: Present only for a run launched with ``--checkpoint-optimizer-state`` (off by default).
    optimizer_state: Optional[Mapping[str, Any]] = None

    @property
    def progress(self) -> Optional[float]:
        """``step / total_steps``: where this θ really sits in its own arm's trajectory."""
        if not self.total_steps:
            return None
        return self.step / self.total_steps

    def metadata(self) -> Dict[str, Any]:
        return {
            "model": self.run.model, "arm": self.run.arm, "seed": self.run.seed,
            "fraction": self.fraction, "step": self.step, "epoch": self.epoch,
            "total_steps": self.total_steps, "scheduled": self.scheduled,
            "progress": self.progress, "mislabelled": self.mislabelled,
            "checkpoint": str(self.path), "has_optimizer_state": self.optimizer_state is not None,
        }


_BENCH_CACHE: Dict[str, Benchmark] = {}


def _bench(model: str) -> Benchmark:
    if not _BENCH_CACHE:
        _BENCH_CACHE.update(discover_benchmarks())
    if model not in _BENCH_CACHE:
        raise KeyError(
            f"no benchmarks/{model}/bench.py; known models: {sorted(_BENCH_CACHE)}. If this is a "
            f"run directory name rather than a bench name, its manifest.json predates "
            f"config['model'] — add it to fisher_ref.checkpoints._LEGACY_BENCH_OF_DIRECTORY."
        )
    return _BENCH_CACHE[model]


def _read_manifest(directory: Path) -> Optional[Mapping[str, Any]]:
    path = directory / "manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _collect_checkpoints(arm_dir: Path) -> Dict[float, Path]:
    found: Dict[float, Path] = {}
    for path in sorted(arm_dir.glob("ckpt_*.pt")):
        match = _CKPT_RE.match(path.name)
        if match:
            found[float(match.group("fraction"))] = path
    return found


def _runs_under(run_root: Path, model: str, seed: Optional[int]) -> List[RunRef]:
    """``run_root`` is one model's report directory: its subdirectories are arms."""
    manifest = _read_manifest(run_root)
    manifest_seed = None
    if manifest is not None:
        manifest_seed = (manifest.get("config", {}) or {}).get("seed")
    runs: List[RunRef] = []
    for arm_dir in sorted(p for p in run_root.iterdir() if p.is_dir()):
        if arm_dir.name not in ARMS:  # _calibration, and anything else that is not an arm
            continue
        checkpoints = _collect_checkpoints(arm_dir)
        if not checkpoints:
            continue
        run_seed = seed if seed is not None else int(manifest_seed or 0)
        runs.append(RunRef(model=model, arm=arm_dir.name, seed=run_seed, directory=arm_dir,
                           checkpoints=checkpoints, manifest=manifest))
    return runs


def discover_runs(
    outputs_root: Path | str = DEFAULT_OUTPUTS_ROOT,
    *,
    model: Optional[str] = None,
    arm: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[RunRef]:
    """Every ``(model, arm, seed)`` run holding at least one checkpoint, both layouts.

    Missing seeds are simply absent from the result — the extra-seed campaign is in flight, and a
    consumer must report the seeds it found rather than assume five (``plan_exp_draft.md`` §3.4).
    """
    root = Path(outputs_root)
    if not root.is_dir():
        return []
    runs: List[RunRef] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if model_dir.name == SEEDS_DIRNAME:
            for seeded_model_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                for seed_dir in sorted(p for p in seeded_model_dir.iterdir() if p.is_dir()):
                    match = _SEED_RE.match(seed_dir.name)
                    if match is None:
                        continue
                    runs += _runs_under(seed_dir, seeded_model_dir.name, int(match.group("seed")))
            continue
        if model_dir.name in _output_groups():
            for grouped in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                runs += _runs_under(grouped, grouped.name, None)
            continue
        runs += _runs_under(model_dir, model_dir.name, None)

    def keep(run: RunRef) -> bool:
        return ((model is None or run.model == model)
                and (arm is None or run.arm == arm)
                and (seed is None or run.seed == seed))

    return [run for run in runs if keep(run)]


def available_seeds(outputs_root: Path | str = DEFAULT_OUTPUTS_ROOT, *,
                    model: Optional[str] = None, arm: Optional[str] = None) -> List[int]:
    return sorted({run.seed for run in discover_runs(outputs_root, model=model, arm=arm)})


def load_theta(
    run: RunRef,
    fraction: float,
    *,
    strict: bool = True,
    device: Any = "cpu",
    build_model: bool = True,
) -> LoadedCheckpoint:
    """Rebuild ``run``'s network at ``fraction`` of the nominal trajectory.

    ``strict=True`` (the default) refuses a payload written before the checkpoint-denominator fix,
    whose fractions mean something else on every arm but ``diag``. ``strict=False`` loads it and
    sets ``mislabelled=True`` on the record.
    """
    if fraction not in run.checkpoints:
        raise KeyError(f"{run.model}/{run.arm} (seed {run.seed}) has fractions "
                       f"{list(run.fractions)}, not {fraction}")
    path = run.checkpoints[fraction]
    payload = torch.load(path, map_location="cpu", weights_only=True)

    missing = [key for key in REQUIRED_PAYLOAD_KEYS if key not in payload]
    if missing and strict:
        raise ValueError(
            f"{path} predates the checkpoint-denominator fix (missing {missing}). Its fractions "
            f"are relative to max_epochs, i.e. --max-epoch-factor times too long, and are wrong "
            f"for every arm but 'diag' (plan_exp_draft.md §3.5, rule 3). Re-run "
            f"{run.model}/{run.arm}, or pass strict=False to accept the mislabelling knowingly."
        )

    model: nn.Module
    if build_model:
        bench = _bench(run.bench_name)
        model = bench.build_model(**run.model_kwargs())
        model.load_state_dict(payload["model_state_dict"])
        model.to(device)
    else:  # metadata only — cheap enough to scan a whole campaign
        model = nn.Identity()

    return LoadedCheckpoint(
        model=model, run=run, fraction=float(payload.get("fraction", fraction)),
        step=int(payload.get("step", -1)), epoch=int(payload.get("epoch", -1)),
        total_steps=(int(payload["total_steps"]) if "total_steps" in payload else None),
        scheduled=bool(payload.get("scheduled", True)), mislabelled=bool(missing), path=path,
        optimizer_state=payload.get("optimizer_state"),
    )
