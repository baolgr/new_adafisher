"""Rebuilding the optimizer's memory at a checkpoint's weights.

The trajectory checkpoints hold **weights only**: the optimizer's running averages are gone when
the training job ends. So the operational protocol has to reconstruct them -- load the weights,
build a fresh ``AdaFisherMulti`` in the mode being measured, run it for a while, and read its state.

**How long.** Not "until the pre-checkpoint history is forgotten", which is the naive reading, but
until the *identity seed* is. Every mode seeds its running average with the identity at step 0, and
the update multiplies the stored state by ``1 - gammas[0] = 0.08`` every ``TCov`` steps, so after
``k`` factor updates a fresh optimizer still carries ``0.08^k * I`` -- a spurious extra damping on
top of ``lambda``. The condition is therefore ``0.08^k << lambda``, not ``0.08^k << 1``. At
``k = 3`` the residue is 5.1e-4, half of a typical ``lambda = 1e-3``, and the applied
preconditioner was measured 12 % to 87 % wrong across the four Kronecker modes; by ``k = 10`` it is
1.1e-11. :func:`rewarm` therefore **refuses** fewer than :data:`MIN_REWARM_FACTOR_UPDATES` times
``TCov`` steps unless told otherwise, with that sentence in the message.

**What is held fixed, and why.** The weights are frozen (``lr = 0``) by default. The staleness term
-- the effect of the weights moving while the memory warms -- was measured at the same order as the
estimator's own batch-draw noise, i.e. invisible. Freezing is therefore free, and it buys something
real: every factor is then estimated at exactly the checkpoint's weights, which is where the
reference is built. ``lr`` is exposed so the moving variant can still be measured.

**Two models, not one.** The operational protocol requires normalisation layers in **train** mode,
where a ``BatchNorm2d`` rewrites its running mean and variance on every forward. After a thousand
steps a re-warmed network's buffers would have nothing to do with the checkpoint's. Restoring them
afterwards would work; using a separate network for the reference cannot be got wrong, so
:func:`rewarm` always loads its own copy.

**Precision.** The re-warm runs in float32, the training dtype, because the object being measured is
the one the optimizer holds and that object is float32; the snapshot is cast to float64 once, at the
boundary. One declared deviation comes with it: the campaign's precision policy turns TF32 off,
which the training runs did not, so these convolutions are computed about 1e-3 relative *more*
accurately than the real ones -- four orders of magnitude below the estimator's own spread, whose
dominant term is that about 92 % of the state is one minibatch's factor.

Public API
----------

:data:`MIN_REWARM_FACTOR_UPDATES`  the minimum number of factor updates, in units of ``TCov``.

:class:`RewarmSpec`  which mode, how many steps, at what learning rate, on which data stream
(``"train"`` is the arm's own augmented loader with the network in train mode; ``"probes"`` feeds
the campaign's clean, augmentation-free probe tensors in eval mode), and from which seed. Two
re-warms differing only in the seed give the operator's own draw-to-draw spread.

:func:`hparams_of`  the arm's *own* hyperparameters, read from its ``manifest.json`` rather than
from the benchmark's defaults, because a run may have overridden any of them on the command line.

:func:`loader_settings`  batch size, augmentation and training subset, as the run itself used them.

:func:`rewarm`  the re-warm itself; returns ``(model, optimizer)`` with the memory warmed. With
``lr = 0`` it asserts that no parameter moved.

:func:`specs_for`  the variant menu: ``main`` always, plus ``replica`` (a second batch order) and
``clean`` (the probes in eval mode) where asked for.

Dependencies: :mod:`fisher_ref.checkpoints`, :mod:`fisher_ref.conventions`, and the training
harness's own optimizer factory and benchmark record.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Generator, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from benchmarks.common.optimizers import HParams, build_optimizer
from benchmarks.common.runner import Benchmark

from .checkpoints import RunRef, load_theta
from .conventions import reference_mode

#: ``plan_exp_draft.md`` §3.2: ``0.08^k << lambda``, met at ``k = 10`` (``1.1e-11``).
MIN_REWARM_FACTOR_UPDATES = 10

DATA_SOURCES = ("train", "probes")


def hparams_of(run: RunRef, bench: Benchmark) -> HParams:
    """The arm's *own* hyperparameters, from its ``manifest.json``, not the bench's defaults.

    A run may have overridden any of them from the command line (``--lam``, ``--gammas``, …), and
    two of this lot's four models do not share an operating point: ``vit_micro_cifar`` trains at
    ``lam = 3e-3`` with decoupled weight decay, ``cnn_gn_cifar`` at ``lam = 1e-3`` with coupled.
    Reading the bench default instead would measure an optimizer the run never used.

    Keys the recorded manifest does not carry (``gamma`` and ``minmax_after_average`` were added
    after campaign 1) fall back to the bench's own value, which is the default in both cases.
    """
    recorded = dict((run.config.get("hparams") or {}))
    known = {f.name for f in fields(HParams)}
    overrides: Dict[str, Any] = {}
    for key, value in recorded.items():
        if key not in known or value is None:
            continue
        overrides[key] = tuple(value) if key == "gammas" else value
    return replace(bench.hparams, **overrides)


def loader_settings(run: RunRef, bench: Benchmark) -> Dict[str, Any]:
    """Batch size, augmentation and subset, as the run itself used them."""
    config = run.config
    return {
        "batch_size": int(config.get("batch_size") or bench.batch_size),
        "cutout": bool(config.get("cutout", True)),
        "train_subset": config.get("train_subset"),
        "seed": int(config.get("seed") or run.seed),
    }


@dataclass(frozen=True)
class RewarmSpec:
    """One re-warm: which mode, how long, on what data, from which batch order."""

    mode: str
    steps: int
    #: ``0.0`` freezes the weights (the default, see the module docstring); anything else measures
    #: the staleness term instead.
    lr: float = 0.0
    #: ``"train"`` is the arm's own augmented loader with the network in train mode — what P2 is
    #: defined as. ``"probes"`` feeds the campaign's clean, augmentation-free probe tensors with the
    #: network in eval mode, which separates "the data the optimizer sees" from "the running
    #: average" (``plan_exp_lot5.md`` §0.4b).
    data: str = "train"
    #: Seeds the batch order (and, for ``"train"``, the augmentation). Two re-warms differing only
    #: here give the operator's own draw-to-draw spread.
    seed: int = 0

    def __post_init__(self) -> None:
        if self.data not in DATA_SOURCES:
            raise ValueError(f"data must be one of {DATA_SOURCES}; got {self.data!r}")


def _train_batches(bench: Benchmark, data_root: str, settings: Mapping[str, Any], *,
                   seed: int, num_workers: int, steps: int
                   ) -> Generator[Tuple[Tensor, Tensor], None, None]:
    """The arm's own training stream, cycled, with a reproducible batch order.

    ``torch.manual_seed`` before ``iter(loader)`` is what makes it reproducible: a ``DataLoader``
    with ``shuffle=True`` and no explicit generator seeds its sampler — and its workers' augmentation
    — from the global RNG at the moment iteration starts.
    """
    torch.manual_seed(seed)
    loader, _, _ = bench.build_data(
        data_root, batch_size=settings["batch_size"], seed=settings["seed"],
        num_workers=num_workers, cutout=settings["cutout"], allow_download=False,
        train_subset=settings["train_subset"],
    )
    produced = 0
    while produced < steps:
        torch.manual_seed(seed + produced)
        for batch in loader:
            yield bench.prepare_batch(batch)
            produced += 1
            if produced >= steps:
                return


def _probe_batches(inputs: Tensor, targets: Tensor, batch_size: int, *, seed: int,
                   steps: int) -> Generator[Tuple[Tensor, Tensor], None, None]:
    """The campaign's own probes, shuffled, cycled: clean images, the eval transform, no Cutout."""
    generator = torch.Generator().manual_seed(seed)
    produced = 0
    while produced < steps:
        order = torch.randperm(int(inputs.shape[0]), generator=generator)
        for start in range(0, int(inputs.shape[0]) - batch_size + 1, batch_size):
            index = order[start:start + batch_size]
            yield inputs[index], targets[index]
            produced += 1
            if produced >= steps:
                return


def rewarm(bench: Benchmark, run: RunRef, fraction: float, spec: RewarmSpec, *,
           hparams: HParams, data_root: str, device: Any = "cpu", num_workers: int = 2,
           probes: Optional[Tuple[Tensor, Tensor]] = None,
           allow_short: bool = False, batch_size: Optional[int] = None,
           optimizer_kwargs: Optional[Mapping[str, Any]] = None) -> Tuple[nn.Module, Any]:
    """Load ``theta`` at ``fraction``, run ``spec.steps`` steps of the arm's own configuration, and
    return the (model, optimizer) pair with its memory warmed.

    The returned model is this function's **own** copy, never the caller's — see the module
    docstring for why the reference needs a second one.

    ``batch_size`` overrides the run's own, and ``optimizer_kwargs`` are passed straight to
    ``build_optimizer``. Both default to "exactly what the run used", which is what lot 5 needs.
    They exist for an experiment that reads the state of an optimizer configured *differently* from
    the arm whose weights it starts from — E21 re-warms at the safety constant E14 tuned, which was
    tuned at batch 32 while these runs trained at 128, and the stored curvature carries the batch
    size through ``1/batch^2`` (``CLAUDE.md`` §4.3). Reading a batch-32 constant against a batch-128
    curvature would compare two operating points at once.
    """
    if spec.mode not in ("diag", "kfac", "ekfac", "tkfac", "tekfac"):
        raise ValueError(f"not a Fisher mode: {spec.mode!r}")
    minimum = MIN_REWARM_FACTOR_UPDATES * hparams.tcov
    if spec.steps < minimum and not allow_short:
        raise ValueError(
            f"a re-warm of {spec.steps} steps is {spec.steps / hparams.tcov:.1f} factor updates; "
            f"the identity each mode seeds its running average with then survives as "
            f"0.08^{spec.steps // hparams.tcov} of the state, acting as a spurious extra damping on "
            f"top of lambda={hparams.lam:g} (plan_exp_draft.md §3.2: the condition is "
            f"0.08^k << lambda, not 0.08^k << 1; at k=3 the applied preconditioner was measured "
            f"12-87 % wrong). Use at least {minimum} steps, or pass allow_short=True knowingly."
        )
    settings = loader_settings(run, bench)
    if batch_size is not None:
        settings = {**settings, "batch_size": int(batch_size)}
    loaded = load_theta(run, fraction, device=device)
    model = loaded.model
    before = copy.deepcopy({name: p.detach().clone() for name, p in model.named_parameters()})

    optimizer = build_optimizer(spec.mode, model, replace(hparams, lr=spec.lr),
                                **dict(optimizer_kwargs or {}))
    stream: Generator[Tuple[Tensor, Tensor], None, None]
    if spec.data == "probes":
        if probes is None:
            raise ValueError("spec.data='probes' needs the probe tensors")
        stream = _probe_batches(probes[0], probes[1], settings["batch_size"], seed=spec.seed,
                                steps=spec.steps)
    else:
        stream = _train_batches(bench, data_root, settings, seed=spec.seed,
                                num_workers=num_workers, steps=spec.steps)

    def run_steps() -> None:
        # The same order common/loop.py::train_under_budget uses: zero_grad, forward, loss,
        # backward, step. Nothing else of that loop applies here (no scheduler, no eval, no
        # records) and reproducing it would only add ways to diverge from it.
        for inputs, targets in stream:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device)
            optimizer.zero_grad()
            bench.loss_fn(model(inputs), targets).backward()
            optimizer.step()

    try:
        if spec.data == "probes":
            with reference_mode(model):
                run_steps()
        else:
            model.train()
            run_steps()
    finally:
        # Close the generator explicitly. A ``DataLoader`` with ``num_workers > 0`` holds worker
        # processes and their file descriptors until its iterator is finalised, and this function is
        # called 15 times per checkpoint x 5 checkpoints in a P2 job — relying on when the generator
        # happens to be collected is how a job runs out of descriptors several hours in.
        stream.close()

    if spec.lr == 0.0:
        drifted = [name for name, p in model.named_parameters()
                   if not torch.equal(p.detach(), before[name].to(p.device))]
        if drifted:
            raise RuntimeError(
                f"the re-warm was asked to freeze theta (lr=0) but these parameters moved: "
                f"{drifted[:5]}{'...' if len(drifted) > 5 else ''}"
            )
    return model, optimizer


def specs_for(modes: Sequence[str], steps: int, *, seed: int, replica_seed: Optional[int] = None,
              clean: bool = False) -> Dict[str, RewarmSpec]:
    """``{variant: spec}`` per mode is built by the runner; this is the variant *menu*
    (``plan_exp_lot5.md`` §0.4b): ``main`` always, plus ``replica`` (a second batch order, giving the
    operator's own spread) and ``clean`` (the probes in eval mode) where asked for.
    """
    out: Dict[str, RewarmSpec] = {}
    for mode in modes:
        out[f"main/{mode}"] = RewarmSpec(mode=mode, steps=steps, seed=seed)
        if replica_seed is not None:
            out[f"replica/{mode}"] = RewarmSpec(mode=mode, steps=steps, seed=replica_seed)
        if clean:
            out[f"clean/{mode}"] = RewarmSpec(mode=mode, steps=steps, seed=seed, data="probes")
    return out


__all__ = ["DATA_SOURCES", "MIN_REWARM_FACTOR_UPDATES", "RewarmSpec", "hparams_of",
           "loader_settings", "rewarm", "specs_for"]
