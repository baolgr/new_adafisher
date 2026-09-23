"""E21: is the divisor curvature-set once the safety constant is tuned, floored or replaced by a clip?

``docs/reports/plans/plan_lambda_dominance.md`` section "E21 -- pre-registered" is the specification
and carries the decision rules. This driver implements them and nothing else.

**The gap this fills.** Lot 5 of the drift campaign measured the operator the optimizer actually
divides by and found it to be a multiple of the identity: ``cond(F~)`` between 1.0000 and 1.1035,
``lambda`` at 98.9-100 % of its mean eigenvalue, the step within one part in 10^4-10^6 of the plain
gradient step. It made that measurement at the **shipped** constant only -- ``plan_exp_lot5.md`` §6.7
records that ``lambda`` is not swept on the P2 side by construction. E7-E14 then tuned the constant
down by seven orders of magnitude, E15 replaced it by one value per layer, E16 replaced it by a clip
that uses no constant at all; all three were judged on **accuracy**, and E0's curvature-free control
was run at the shipped constant, so it says nothing about them. So the question "does the curvature
now set the divisor, or does the constant still crush it?" has no measured answer at any of the
settings this project proposes to use. E21 is lot 5's own protocol, replayed at those settings.

**What one cell is.** Lot 5's P2 re-warm, unchanged: load ``theta`` at a checkpoint of the campaign's
``diag`` run, build a fresh ``AdaFisherMulti`` **in the setting under test**, run it with the weights
frozen (``lr = 0``) until the identity its running average is seeded with has decayed, then read the
divisor with :mod:`fisher_ref.divisor`. One cell is one (arm, mode, checkpoint). Nothing trains, so a
cell costs a re-warm.

**The five arms.**

``bridge``    the shipped constant at the **run's own** batch size, with the shipped estimator: lot
              5's own object. It is the gate -- it must reproduce lot 5's published numbers, or this
              job is not comparable with them and nothing else in it is read.
``shipped``   the shipped constant at batch 32, with E14/E16's estimator. The like-for-like control
              the three fixes are measured against: it differs from ``tuned`` in the constant alone.
``tuned``     the single constant E14/E13/E10 located for this (network, mode).
``layerrel``  E15's one constant per layer, at the dial E15/E19 selected for this (network, mode).
``clipema``   E16's main clip arm, at the fraction it selected for this (network, mode).

**Why batch 32 for everything except the gate.** The stored curvature carries the batch size through
``1/batch^2`` (``CLAUDE.md`` §4.3, measured 56 times in E3), and every constant E14, E15 and E16
selected was selected at batch 32 while these checkpoints come from batch-128 runs. Reading a
batch-32 constant against a batch-128 curvature would move two things at once, in the direction that
flatters the constant.

**Why the re-warm is longer than lot 5's.** Each mode seeds its running average with the identity, so
after ``k`` factor updates a fresh optimizer still carries ``0.08^k`` of it -- a spurious extra
damping. The condition is ``0.08^k << lambda`` (``plan_exp_draft.md`` §3.2), and at E14's constants
``lambda`` is 1e-10 to 1e-11, where lot 5's ``k = 10`` residue of 1.07e-11 is **the same size as
lambda itself**. ``CLAUDE.md`` says so in as many words: lengthen it when sweeping ``lambda``
downwards. The default here is 2 000 steps (``k = 20``, residue 1.2e-22), and every cell asserts
``0.08^k <= 1e-3 * lambda`` rather than noting it afterwards. Lot 5 measured that ``k = 20`` changes
nothing at the shipped constant, so the gate is unaffected by the change.

**Sharding**, as E16: a (model, seed) job runs ``E21_NSHARDS`` copies, each taking every
``NSHARDS``-th cell and writing its own shard file after every cell; then ``E21_MERGE=1`` merges
them, checks every planned cell is present exactly once, and applies the decision rules.

Environment variables::

    E21_MODEL          one benchmark (default: cnn_gn_cifar)
    E21_SEED           the seed of the run whose checkpoints are read (default: 0)
    E21_ARMS           comma-separated (default: every arm)
    E21_MODES          comma-separated (default: the arm's own modes)
    E21_FRACTIONS      checkpoints, comma-separated (default: 0.1,0.5,1)
    E21_REWARM_STEPS   steps of the setting under test before the state is read (default: 2000)
    E21_NSHARDS        processes sharing the job (default: 1)
    E21_SHARD          this process's index, 0 .. NSHARDS-1 (default: 0)
    E21_MERGE          "1": merge the shard files and read the rules instead of running
    E21_OUT            the job's file (default: fisher_ref/outputs/e21_divisor_<model>_s<seed>.json)
    E21_SMOKE          "1" allows a reduced configuration
    WARMUP_SGD_DEVICE, WARMUP_SGD_DATA_ROOT, WARMUP_SGD_NUM_WORKERS as elsewhere.
"""
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch

from benchmarks.common.runner import discover_benchmarks
from fisher_ref import conventions, divisor
from fisher_ref.checkpoints import discover_runs
from fisher_ref.experiments.e5_unhooked_freeze_control import unhooked_parameter_names
from fisher_ref.experiments.e16_floor_clip import CHECK_LAMBDA as E16_CHECK_LAMBDA
from fisher_ref.experiments.warmup_sgd_baseline import DATA_ROOT, DEVICE, NUM_WORKERS
from fisher_ref.rewarm import RewarmSpec, hparams_of, loader_settings, rewarm

SMOKE = os.environ.get("E21_SMOKE", "0") == "1"
MODEL = os.environ.get("E21_MODEL", "cnn_gn_cifar")
SEED = int(os.environ.get("E21_SEED", "0"))
NSHARDS = int(os.environ.get("E21_NSHARDS", "1"))
SHARD = int(os.environ.get("E21_SHARD", "0"))
MERGE = os.environ.get("E21_MERGE", "0") == "1"
REWARM_STEPS = int(os.environ.get("E21_REWARM_STEPS", "2000"))
OUT = Path(os.environ.get(
    "E21_OUT", str(ROOT / f"fisher_ref/outputs/e21_divisor_{MODEL}_s{SEED}.json")))

#: The arm whose checkpoints are read. Lot 5 read ``diag``'s trajectory; keeping it is what makes
#: the gate a comparison rather than a new measurement.
THETA_ARM = "diag"

#: The batch size every arm but ``bridge`` re-warms at: the one E14, E15 and E16 tuned at.
E_PROTOCOL_BATCH = 32

#: How far below ``lambda`` the identity seed's residue ``0.08^k`` has to sit. Asserted per cell.
RESIDUE_MARGIN = 1e-3

#: The single constant E14/E13/E10 located per (network, mode), five seeds at batch 32.
#: ``ekfac``/``tekfac`` come from E16's own table, so there is one source of truth for them;
#: ``kfac``/``tkfac`` are read from those sweeps' published best (``results.md`` §4.3) and are
#: ``None`` where no value was located -- E14 found ``resnet20_cifar`` flat in both (best gain +0.20
#: and -0.03), and ``tkfac`` was never run on ``cct_2_3x2_cifar``.
TUNED_LAMBDA: Dict[Tuple[str, str], Optional[float]] = {
    ("cnn_gn_cifar", "kfac"): 1e-10,      # 65.52 at 1e-10 (results.md §4.3)
    ("cnn_gn_cifar", "tkfac"): 3e-8,      # 63.04 at 3e-8
    ("vit_micro_cifar", "kfac"): 3e-11,   # 52.98 at 3e-11
    ("vit_micro_cifar", "tkfac"): 3e-10,  # 50.25 at 3e-10
    ("cct_2_3x2_cifar", "kfac"): 3e-12,   # 80.40 at 3e-12
    ("cct_2_3x2_cifar", "tkfac"): None,   # not run
    ("resnet20_cifar", "kfac"): None,     # flat: best gain +0.20
    ("resnet20_cifar", "tkfac"): None,    # flat: best gain -0.03
}
for (_model, _mode), _lam in E16_CHECK_LAMBDA.items():
    TUNED_LAMBDA[(_model, _mode)] = _lam
# The values this experiment was designed around. A change in E16's table must be seen here, not
# silently inherited: these four are the ones E21's registry covers.
assert E16_CHECK_LAMBDA[("cnn_gn_cifar", "ekfac")] == 3e-11
assert E16_CHECK_LAMBDA[("vit_micro_cifar", "ekfac")] == 1e-10
assert E16_CHECK_LAMBDA[("cct_2_3x2_cifar", "tekfac")] == 3e-11
assert E16_CHECK_LAMBDA[("resnet20_cifar", "tekfac")] == 1e-11

#: The dial E15 (``cnn_gn_cifar``, ``vit_micro_cifar``) and E19 (the other two) selected for S1-b,
#: per (network, mode). ``results.md`` §5.1, the value in brackets. E19 did not run ``kfac``.
TUNED_TAU: Dict[Tuple[str, str], Optional[float]] = {
    ("cnn_gn_cifar", "kfac"): 1.0, ("cnn_gn_cifar", "ekfac"): 0.1,
    ("cnn_gn_cifar", "tekfac"): 0.1,
    ("vit_micro_cifar", "kfac"): 3.0, ("vit_micro_cifar", "ekfac"): 0.1,
    ("vit_micro_cifar", "tekfac"): 0.1,
    ("cct_2_3x2_cifar", "ekfac"): 0.3, ("cct_2_3x2_cifar", "tekfac"): 0.3,
    ("resnet20_cifar", "ekfac"): 0.3, ("resnet20_cifar", "tekfac"): 0.3,
}

#: The clipped fraction E16's ``clipema`` arm selected on validation, per (network, mode).
#: ``results.md`` §5.2, the value in brackets in the ``clipema`` column.
TUNED_CLIP_Q: Dict[Tuple[str, str], float] = {
    ("cnn_gn_cifar", "ekfac"): 0.99, ("cnn_gn_cifar", "tekfac"): 0.7,
    ("vit_micro_cifar", "ekfac"): 0.95, ("vit_micro_cifar", "tekfac"): 0.99,
    ("cct_2_3x2_cifar", "ekfac"): 0.95, ("cct_2_3x2_cifar", "tekfac"): 0.99,
    ("resnet20_cifar", "ekfac"): 0.95, ("resnet20_cifar", "tekfac"): 0.9,
}
CLIP_EMA_HORIZON = 1000     # E16's own horizon

#: Which modes each arm can be read in. ``bridge`` covers all five, because lot 5 published all
#: five. The clip exists for the two eigenbasis modes only; a relative damping is refused for
#: ``diag``, whose min-max removes the scale it would be relative to.
ARM_MODES = {
    "bridge": ("diag", "kfac", "ekfac", "tkfac", "tekfac"),
    "shipped": ("kfac", "ekfac", "tkfac", "tekfac"),
    "tuned": ("kfac", "ekfac", "tkfac", "tekfac"),
    "layerrel": ("kfac", "ekfac", "tekfac"),
    "clipema": ("ekfac", "tekfac"),
}
ARM_ORDER = ["bridge", "shipped", "tuned", "layerrel", "clipema"]
ARMS = [a for a in (os.environ.get("E21_ARMS") or ",".join(ARM_ORDER)).split(",") if a]
MODE_FILTER = [m for m in (os.environ.get("E21_MODES") or "").split(",") if m]
FRACTIONS = [float(f) for f in (os.environ.get("E21_FRACTIONS") or "0.1,0.5,1").split(",") if f]

#: The two networks lot 5 published, i.e. the two where ``bridge`` is a gate rather than a new
#: number. ``cct_2_3x2_cifar`` and ``resnet20_cifar`` are regime-B models lot 5 never ran.
LOT5_PUBLISHED = ("cnn_gn_cifar", "vit_micro_cifar")
#: Lot 5's own published ranges, ``plan_exp_lot5.md`` §6 and ``results.md`` §6.3.
LOT5_COND_RANGE = (1.0, 1.15)
LOT5_MIN_DAMPING_SHARE = 0.98

#: Rule 2's cuts on the median ``frac_curvature_set``.
CURVATURE_SET_AT = 0.5
LAMBDA_SET_AT = 0.05
#: Rule 4: above this the applied step is the plain momentum step.
PLAIN_STEP_AT = 0.99


def _log(message: str) -> None:
    print(message, flush=True)


def check_settings() -> None:
    problems: List[str] = []
    if MODEL not in {m for m, _ in TUNED_LAMBDA}:
        problems.append(f"E21_MODEL={MODEL!r} has no tuned constant on record")
    if REWARM_STEPS != 2000:
        problems.append(f"E21_REWARM_STEPS={REWARM_STEPS}")
    if FRACTIONS != [0.1, 0.5, 1.0]:
        problems.append(f"E21_FRACTIONS={FRACTIONS}")
    unknown = [a for a in ARMS if a not in ARM_ORDER]
    if unknown:
        raise SystemExit(f"E21: unknown arms {unknown}; available: {ARM_ORDER}")
    if not 0 <= SHARD < NSHARDS:
        raise SystemExit(f"E21: need 0 <= E21_SHARD < E21_NSHARDS; got {SHARD}, {NSHARDS}")
    if problems and not SMOKE:
        raise SystemExit("E21: non-production settings without E21_SMOKE=1: " + ", ".join(problems))


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {type(exc).__name__}"


def provenance() -> Dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "smoke": SMOKE, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "device": str(DEVICE), "num_workers": NUM_WORKERS,
        "shard": SHARD, "nshards": NSHARDS, "rewarm_steps": REWARM_STEPS,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(), "host": platform.node(),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("E21_", "WARMUP_SGD_"))},
    }


def cell_key(arm: str, mode: str, fraction: float) -> str:
    return f"{arm}|{mode}|{fraction:g}"


def plan_cells(model_name: str, bench, run) -> List[Dict[str, Any]]:
    """Every cell of one (model, seed) job, in ``ARM_ORDER``.

    A cell carries the whole configuration of its re-warm, so that what was run can be read off the
    output file without re-deriving it from this module.
    """
    hparams = hparams_of(run, bench)
    own_batch = int(loader_settings(run, bench)["batch_size"])
    cells: List[Dict[str, Any]] = []
    for arm in sorted(set(ARMS), key=ARM_ORDER.index):
        modes = [m for m in ARM_MODES[arm] if not MODE_FILTER or m in MODE_FILTER]
        for mode in modes:
            eigen = mode in divisor.EIGEN_MODES
            # bridge is lot 5's object: the shipped constant, the run's own batch, and none of the
            # E-series knobs. Every other arm carries E14/E16's two corrections, so that `shipped`
            # differs from `tuned` in the constant alone.
            overrides: Dict[str, Any] = {}
            if arm != "bridge" and eigen:
                overrides.update({"eig_before_rescale": True, "norm_exact_rescaling": True})
            lam, batch, freeze = hparams.lam, E_PROTOCOL_BATCH, None
            wd = hparams.weight_decay
            if arm == "bridge":
                batch = own_batch
            elif arm == "tuned":
                lam = TUNED_LAMBDA.get((model_name, mode))
                if lam is None:
                    continue        # no constant was located for this (network, mode)
            elif arm == "layerrel":
                tau = TUNED_TAU.get((model_name, mode))
                if tau is None:
                    continue
                overrides.update({"damping": "layer_relative", "damping_tau": tau,
                                  "hold_cap": True})
            elif arm == "clipema":
                q = TUNED_CLIP_Q.get((model_name, mode))
                if q is None:
                    continue
                overrides.update({"rescale_form": "clip", "clip_threshold": "ema",
                                  "clip_fraction": q, "clip_ema_horizon": CLIP_EMA_HORIZON})
                # E16's own two choices for its clip arms, reproduced: the parameters no hooked
                # module owns are frozen, and a decoupled decay -- which in an add cell has
                # vanished with lr = cap * lambda -- is dropped. Neither moves the divisor (nothing
                # moves at lr = 0; a coupled decay does reach the momentum, so it is kept).
                freeze = unhooked_parameter_names(bench)
                if hparams.decoupled_wd:
                    wd = 0.0
            cells.append(dict(arm=arm, mode=mode, lam=lam, batch=batch, wd=wd,
                              freeze=freeze, overrides=overrides,
                              tau=overrides.get("damping_tau"),
                              clip_fraction=overrides.get("clip_fraction")))
    return [dict(cell, fraction=f) for cell in cells for f in FRACTIONS]


def residue_check(lam: Optional[float], steps: int, tcov: int, form: str) -> Dict[str, Any]:
    """``0.08^k`` against ``lambda``, the condition the re-warm length has to meet.

    Under the clip ``lambda`` is unused, so there is nothing to be small against except the
    curvature's own floor, which is a per-layer quantity recorded with the layer; the check then
    reports the residue and passes.
    """
    updates = steps // tcov
    residue = 0.08 ** updates
    if form == "clip" or not lam:
        return {"factor_updates": updates, "residue": residue, "over_lambda": None, "ok": True}
    return {"factor_updates": updates, "residue": residue, "over_lambda": residue / lam,
            "ok": residue <= RESIDUE_MARGIN * lam}


def run_cell(bench, run, cell: Dict[str, Any], hparams) -> Dict[str, Any]:
    """One re-warm, one read. Returns the per-layer statistics and their median."""
    form = cell["overrides"].get("rescale_form", "add")
    residue = residue_check(cell["lam"], REWARM_STEPS, hparams.tcov, form)
    if not residue["ok"]:
        raise ValueError(
            f"a re-warm of {REWARM_STEPS} steps leaves {residue['residue']:.2e} of the identity "
            f"seed, which is {residue['over_lambda']:.2f} x lambda={cell['lam']:g}: the spurious "
            f"damping would be the size of the thing being measured. Raise E21_REWARM_STEPS "
            f"(plan_exp_draft.md §3.2)."
        )
    spec = RewarmSpec(mode=cell["mode"], steps=REWARM_STEPS, lr=0.0, data="train", seed=1000)
    started = time.perf_counter()
    model, optimizer = rewarm(
        bench, run, cell["fraction"], spec,
        hparams=replace(hparams, lam=cell["lam"], weight_decay=cell["wd"]),
        data_root=DATA_ROOT, device=DEVICE, num_workers=NUM_WORKERS,
        batch_size=cell["batch"], optimizer_kwargs=cell["overrides"],
    )
    if cell["freeze"]:
        # Frozen *after* the optimizer is built, as E16 does it before: with lr = 0 nothing moves
        # either way, so this only records the arm's own configuration.
        wanted = set(cell["freeze"])
        for name, param in model.named_parameters():
            if name in wanted:
                param.requires_grad_(False)
    skipped: Dict[str, str] = {}
    layers = divisor.measure(model, optimizer, mode=cell["mode"], skipped=skipped)
    per_layer = {name: divisor.statistics(layer) for name, layer in layers.items()}
    kinds = {name: layer.kind for name, layer in layers.items()}
    out = {
        "wall_s": time.perf_counter() - started,
        "median": divisor.summarise(layers),
        "layers": per_layer, "kinds": kinds, "residue": residue,
        # A layer whose own damped factor came out degenerate costs that layer, not the cell -- and
        # it says so here rather than quietly shrinking the median's population.
        "n_hooked": len(optimizer.modules), "n_read": len(layers), "skipped": skipped,
    }
    del model, optimizer, layers
    if str(DEVICE).startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def safe_run_cell(bench, run, cell, hparams) -> Dict[str, Any]:
    """A crash is an outcome to record. ``decide`` refuses to read a verdict through one."""
    try:
        return run_cell(bench, run, cell, hparams)
    except Exception as exc:  # noqa: BLE001
        print(f"    [CRASHED] {type(exc).__name__}: {exc}", flush=True)
        return {"crashed": True, "error": f"{type(exc).__name__}: {exc}", "median": {},
                "layers": {}, "kinds": {}, "wall_s": None}


def shard_path(shard: int) -> Path:
    return OUT.with_name(OUT.stem + f".shard{shard}of{NSHARDS}.json")


def _sanitise(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    return value


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as handle:
        json.dump(_sanitise(payload), handle, indent=1, allow_nan=False)
    tmp.replace(path)


def _median_of(cells: Dict[str, Any], arm: str, mode: str, key: str) -> float:
    """The median of one statistic over this (arm, mode)'s checkpoints. ``nan`` if any cell of it
    crashed: a partial arm is not a verdict."""
    rows = [c for k, c in cells.items() if k.startswith(f"{arm}|{mode}|")]
    if not rows or any(row.get("crashed") for row in rows):
        return float("nan")
    values = [row["median"].get(key) for row in rows]
    values = [v for v in values if v is not None and v == v]
    return float(torch.tensor(values, dtype=torch.float64).median()) if values else float("nan")


def decide(model_name: str, cells: Dict[str, Any], planned: List[str]) -> Dict[str, Any]:
    """The pre-registered rules. Every one of them can come out "inconclusive", and a missing or
    crashed cell makes it so rather than being skipped."""
    missing = [key for key in planned if key not in cells]
    crashed = sorted(key for key, cell in cells.items() if cell.get("crashed"))
    out: Dict[str, Any] = {"missing": missing, "crashed": crashed, "rules": {}, "table": {}}

    # Rule 1, the gate: on the networks lot 5 published, the bridge arm must reproduce it.
    gate: Dict[str, Any] = {"applies": model_name in LOT5_PUBLISHED, "modes": {}}
    if gate["applies"]:
        for mode in ARM_MODES["bridge"]:
            cond = _median_of(cells, "bridge", mode, "cond_divisor")
            # Lot 5's literal statistic, not the form-agnostic one, because the gate is a
            # reproduction of lot 5's own published number.
            share = _median_of(cells, "bridge", mode, "lambda_over_mean_divisor")
            if cond != cond or share != share:
                gate["modes"][mode] = {"cond_divisor": None,
                                       "lambda_over_mean_divisor": None,
                                       "reproduces_lot5": None}
                continue
            gate["modes"][mode] = {
                "cond_divisor": cond, "lambda_over_mean_divisor": share,
                "reproduces_lot5": bool(LOT5_COND_RANGE[0] <= cond <= LOT5_COND_RANGE[1]
                                        and share >= LOT5_MIN_DAMPING_SHARE),
            }
        verdicts = [m["reproduces_lot5"] for m in gate["modes"].values()]
        gate["passes"] = None if any(v is None for v in verdicts) else all(verdicts)
    else:
        gate["passes"] = None
    out["rules"]["1_gate"] = gate

    # Rules 2-4, per (arm, mode).
    for arm in ARM_ORDER:
        for mode in ARM_MODES[arm]:
            key = f"{arm}|{mode}"
            if not any(k.startswith(f"{key}|") for k in cells):
                continue
            frac = _median_of(cells, arm, mode, "frac_curvature_set")
            step_share = _median_of(cells, arm, mode, "step_share_curvature_set")
            cos = _median_of(cells, arm, mode, "cos_to_plain")
            # Rule 2 is read on the count for the additive forms and on the share of the step for
            # the clip, whose count is its own hyperparameter (divisor.py). The two are not mixed:
            # `read_on` says which one this row's verdict came from, and rule 3's gain is taken on
            # that same statistic -- both are defined for every form, so the pair is like-for-like.
            clipped = arm == "clipema"
            read_on = "step_share_curvature_set" if clipped else "frac_curvature_set"
            statistic = step_share if clipped else frac
            # Only a fix has something to gain over the control; the control and the gate do not.
            base = (_median_of(cells, "shipped", mode, read_on)
                    if arm in ("tuned", "layerrel", "clipema") else float("nan"))
            verdict = "inconclusive"
            if statistic == statistic:
                verdict = ("curvature_set" if statistic >= CURVATURE_SET_AT else
                           ("cap_set" if clipped else "lambda_set")
                           if statistic <= LAMBDA_SET_AT else "partial")
            out["table"][key] = {
                "frac_curvature_set": frac if frac == frac else None,
                "step_share_curvature_set": step_share if step_share == step_share else None,
                "read_on": read_on,
                "cos_to_plain": cos if cos == cos else None,
                "non_curvature_share": (lambda v: v if v == v else None)(
                    _median_of(cells, arm, mode, "non_curvature_share")),
                "cond_divisor": (lambda v: v if v == v else None)(
                    _median_of(cells, arm, mode, "cond_divisor")),
                "spread_curvature": (lambda v: v if v == v else None)(
                    _median_of(cells, arm, mode, "spread_curvature")),
                # Rule 2.
                "verdict": verdict,
                # Rule 3: did the fix move it off the shipped constant's answer, on the statistic
                # its own verdict is read on?
                "gain_over_shipped": ((statistic - base)
                                      if (statistic == statistic and base == base) else None),
                "moved_off_lambda": (bool(statistic >= CURVATURE_SET_AT and base <= LAMBDA_SET_AT)
                                     if (statistic == statistic and base == base) else None),
                # Rule 4: is the step still the plain momentum step?
                "acts_on_the_step": bool(cos <= PLAIN_STEP_AT) if cos == cos else None,
            }
    out["rules"]["2_3_4"] = {
        "curvature_set_at": CURVATURE_SET_AT, "lambda_set_at": LAMBDA_SET_AT,
        "plain_step_at": PLAIN_STEP_AT,
        "readable": not missing and not crashed and gate["passes"] is not False,
    }
    return out


def main() -> None:
    check_settings()
    conventions.configure()
    runs = discover_runs(str(ROOT / "benchmarks" / "outputs"), model=MODEL, arm=THETA_ARM,
                         seed=SEED)
    if len(runs) != 1:
        raise SystemExit(f"expected exactly one {MODEL}/{THETA_ARM} run at seed {SEED}; found "
                         f"{[str(r.directory) for r in runs]}")
    run = runs[0]
    bench = discover_benchmarks()[run.bench_name]
    hparams = hparams_of(run, bench)
    cells = plan_cells(MODEL, bench, run)
    planned = [cell_key(c["arm"], c["mode"], c["fraction"]) for c in cells]

    if MERGE:
        merged: Dict[str, Any] = {}
        shards: Dict[str, Any] = {}
        for shard in range(NSHARDS):
            path = shard_path(shard)
            if not path.exists():
                raise SystemExit(f"E21 merge: {path} is missing")
            payload = json.load(open(path))
            for key, cell in payload["cells"].items():
                if key in merged:
                    raise SystemExit(f"E21 merge: {key} is in two shards")
                merged[key] = cell
            shards[str(shard)] = payload["provenance"]
        _write(OUT, {"model": MODEL, "seed": SEED, "theta_arm": THETA_ARM,
                     "fractions": FRACTIONS, "rewarm_steps": REWARM_STEPS,
                     "planned": planned, "cells": merged, "shards": shards,
                     "decisions": decide(MODEL, merged, planned)})
        _log(f"E21 merged {len(merged)} cells -> {OUT}")
        return

    owners = [index % NSHARDS for index in range(len(cells))]
    mine = [cell for index, cell in enumerate(cells) if owners[index] == SHARD]
    _log(f"E21 {MODEL} seed {SEED}: {len(mine)} of {len(cells)} cells (shard {SHARD}/{NSHARDS}), "
         f"re-warm {REWARM_STEPS} steps at lr=0, theta from {THETA_ARM}, "
         f"shipped lam={hparams.lam:g} TCov={hparams.tcov}")
    payload: Dict[str, Any] = {"model": MODEL, "seed": SEED, "theta_arm": THETA_ARM,
                               "planned": planned, "provenance": provenance(), "cells": {}}
    for index, cell in enumerate(mine, start=1):
        key = cell_key(cell["arm"], cell["mode"], cell["fraction"])
        _log(f"  [{index}/{len(mine)}] {key}: lam={cell['lam']} batch={cell['batch']} "
             f"overrides={cell['overrides']}")
        result = safe_run_cell(bench, run, cell, hparams)
        payload["cells"][key] = {**{k: v for k, v in cell.items() if k != "freeze"},
                                 "n_frozen": len(cell["freeze"] or []), **result}
        median = result.get("median") or {}
        _log(f"      frac_curvature_set={median.get('frac_curvature_set')} "
             f"step_share={median.get('step_share_curvature_set')} "
             f"non_curvature_share={median.get('non_curvature_share')} "
             f"cond_divisor={median.get('cond_divisor')} "
             f"cos_to_plain={median.get('cos_to_plain')} ({result.get('wall_s')}s)")
        _write(shard_path(SHARD), payload)
    _log(f"E21 shard {SHARD} done -> {shard_path(SHARD)}")


if __name__ == "__main__":
    main()
