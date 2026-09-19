"""Is a re-warm of ten factor updates enough on the operational protocol's own models?

``rewarm_fidelity.py`` answered that question once, on ``cnn_gn_cifar`` alone, from a 2 000-step
training run of its own. The operational protocol runs on four models and reads real checkpoints,
so the answer has to be re-measured where it will be used.

The shape of the measurement. From a checkpoint's weights, with the weights frozen:

* ``S_ref`` -- one long warm of thirty factor updates on batch order A: the converged state;
* ``S_B``, ``S_C`` -- two re-warms of ``k`` factor updates on batch orders B and C. Their distance
  is the estimator's **own noise floor**: two honest draws of the same quantity;
* ``S_M`` -- one re-warm of ``k`` updates with the weights **moving**: the staleness term.

Everything is reported on the **applied preconditioner** (the operator applied to a fixed random
direction, which is the object the protocol actually compares against the reference) as well as on
the raw state, because the two behave very differently: the raw factors sit at two to four times
their own floor whatever the length, while the applied operator collapses to 1.3 to 4.5 times it
by ten updates.

The reading is pre-registered so it cannot be fitted afterwards: ten updates is *confirmed
sufficient* on a model if, for all five modes, the re-warm gap on the applied preconditioner is
within five times its own noise floor **and** improves by less than a factor of two between ten and
twenty updates. A model where that fails has its operational numbers re-run at the length that
works.

Environment variables::

    P2F_MODELS        comma-separated run directories    (default: the four regime-A models)
    P2F_FRACTION      which checkpoint                   (default: 0.5)
    P2F_KS            comma-separated k values           (default: 3,10,20)
    P2F_REF_K         the converged warm's k             (default: 30)
    P2F_DATA_ROOT     dataset root                       (default: benchmarks/data)
    P2F_OUTPUTS_ROOT  where the checkpoints live         (default: benchmarks/outputs)
    P2F_OUT           where the JSON goes   (default: fisher_ref/outputs/rewarm_fidelity_lot5.json)
    P2F_DEVICE        cuda or cpu                        (default: cuda when available)
    P2F_WORKERS       data-loader workers                (default: 2)

Output: a table on standard output and the JSON at ``P2F_OUT``.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch  # noqa: E402

from benchmarks.common.runner import discover_benchmarks  # noqa: E402
from fisher_ref import conventions  # noqa: E402
from fisher_ref.approx.adafisher_state import MODES, snapshot  # noqa: E402
from fisher_ref.checkpoints import discover_runs  # noqa: E402
from fisher_ref.rewarm import RewarmSpec, hparams_of, rewarm  # noqa: E402

MODELS = os.environ.get(
    "P2F_MODELS", "cnn_gn_cifar,cnn_gn_cifar_bn,vit_micro_cifar,mlp_ln_mnist").split(",")
FRACTION = float(os.environ.get("P2F_FRACTION", "0.5"))
KS = [int(v) for v in os.environ.get("P2F_KS", "3,10,20").split(",")]
REF_K = int(os.environ.get("P2F_REF_K", "30"))
DATA_ROOT = os.environ.get("P2F_DATA_ROOT", str(ROOT / "benchmarks" / "data"))
#: Where the trajectory checkpoints live. Every other lot-5 entry point takes ``--outputs-root``;
#: relying on the default only works because the cluster's ``$SLURM_SUBMIT_DIR`` happens to be the
#: repository root, which is a coincidence, not a contract.
OUTPUTS_ROOT = os.environ.get("P2F_OUTPUTS_ROOT", str(ROOT / "benchmarks" / "outputs"))
OUT = Path(os.environ.get("P2F_OUT", str(ROOT / "fisher_ref" / "outputs" /
                                         "rewarm_fidelity_lot5.json")))
DEVICE = os.environ.get("P2F_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = int(os.environ.get("P2F_WORKERS", "2"))

#: §0.8's rule, written down before the numbers exist.
FLOOR_FACTOR = 5.0
IMPROVEMENT_FACTOR = 2.0


def _applied(state: Dict[str, Any], directions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """``F~^-1 v`` on a fixed random direction per layer — the object P2 compares to ``F``."""
    return {name: block.damped.solve(directions[name], 0.0) for name, block in state.items()}


def _raw(state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """The undamped operator's own entries, as a vector per layer: the *state*, not its effect."""
    return {name: block.undamped.diag() for name, block in state.items()}


def _relative(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    shared = sorted(set(a) & set(b))
    num = sum(float((a[name] - b[name]).pow(2).sum()) for name in shared)
    den = sum(float(a[name].pow(2).sum()) for name in shared)
    return (num / den) ** 0.5 if den else float("nan")


def _directions(state: Dict[str, Any], seed: int = 7) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {name: torch.randn(block.damped.P, dtype=conventions.REFERENCE_DTYPE,
                              generator=generator)
            for name, block in state.items()}


def _state_at(bench, run, hparams, mode: str, steps: int, seed: int, lr: float) -> Dict[str, Any]:
    model, optimizer = rewarm(bench, run, FRACTION, RewarmSpec(mode=mode, steps=steps, lr=lr,
                                                               seed=seed),
                              hparams=hparams, data_root=DATA_ROOT, device=DEVICE,
                              num_workers=WORKERS, allow_short=True)
    state = snapshot(model, optimizer, mode=mode, dtype=conventions.REFERENCE_DTYPE, device="cpu")
    del model, optimizer
    gc.collect()
    if str(DEVICE).startswith("cuda"):
        torch.cuda.empty_cache()
    return state


def main() -> None:
    conventions.configure()
    benches = discover_benchmarks()
    results: List[Dict[str, Any]] = []
    print(f"re-warm fidelity (lot 5) | models={MODELS} fraction={FRACTION} ks={KS} "
          f"reference={REF_K} device={DEVICE}", flush=True)
    print(f"{'model':<18}{'mode':<8}{'k':>4}{'gap(applied)':>14}{'floor':>12}{'gap/floor':>11}"
          f"{'staleness':>11}{'gap(state)':>12}{'floor':>11}", flush=True)

    for model_name in MODELS:
        runs = discover_runs(OUTPUTS_ROOT, model=model_name, arm="diag", seed=0)
        if len(runs) != 1:
            print(f"  !! {model_name}: {len(runs)} runs found, skipped", flush=True)
            continue
        run = runs[0]
        bench = benches[run.bench_name]
        hparams = hparams_of(run, bench)
        for mode in MODES:
            started = time.perf_counter()
            reference = _state_at(bench, run, hparams, mode, REF_K * hparams.tcov, 101, 0.0)
            directions = _directions(reference)
            reference_applied, reference_raw = _applied(reference, directions), _raw(reference)
            del reference
            row_by_k: Dict[int, Dict[str, float]] = {}
            for k in KS:
                steps = k * hparams.tcov
                b = _state_at(bench, run, hparams, mode, steps, 202, 0.0)
                c = _state_at(bench, run, hparams, mode, steps, 303, 0.0)
                m = _state_at(bench, run, hparams, mode, steps, 404, hparams.lr)
                row = {
                    "gap_applied": _relative(reference_applied, _applied(b, directions)),
                    "floor_applied": _relative(_applied(b, directions), _applied(c, directions)),
                    "staleness_applied": _relative(_applied(b, directions),
                                                   _applied(m, directions)),
                    "gap_state": _relative(reference_raw, _raw(b)),
                    "floor_state": _relative(_raw(b), _raw(c)),
                }
                row_by_k[k] = row
                ratio = row["gap_applied"] / row["floor_applied"] if row["floor_applied"] else \
                    float("nan")
                print(f"{model_name if k == KS[0] and mode == MODES[0] else '':<18}"
                      f"{mode if k == KS[0] else '':<8}{k:>4}{row['gap_applied']:>14.2e}"
                      f"{row['floor_applied']:>12.2e}{ratio:>11.2f}"
                      f"{row['staleness_applied']:>11.2e}{row['gap_state']:>12.2e}"
                      f"{row['floor_state']:>11.2e}", flush=True)
                del b, c, m
                gc.collect()
            results.append({"model": model_name, "mode": mode, "fraction": FRACTION,
                            "tcov": hparams.tcov, "lam": hparams.lam,
                            "by_k": {str(k): v for k, v in row_by_k.items()},
                            "seconds": time.perf_counter() - started})
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps({"verdicts": verdicts(results), "rows": results,
                                       "rule": {"floor_factor": FLOOR_FACTOR,
                                                "improvement_factor": IMPROVEMENT_FACTOR,
                                                "ks": KS, "reference_k": REF_K}},
                                      indent=2, default=str))

    print("\n-- §0.8's pre-registered verdict --", flush=True)
    for model_name, verdict in verdicts(results).items():
        print(f"  {model_name:<18} {verdict}", flush=True)
    print(f"\nwritten to {OUT}", flush=True)


def verdicts(results: List[Dict[str, Any]], target: int = 10) -> Dict[str, str]:
    """§0.8's rule, applied. Nothing here chooses a threshold; both come from the module constants.

    ``k = target`` is *confirmed sufficient* on a model when, for **every** mode, the applied
    preconditioner's re-warm gap is within ``FLOOR_FACTOR`` of its own batch-draw noise floor and
    improves by less than ``IMPROVEMENT_FACTOR`` between ``target`` and the next longer ``k``.
    """
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for row in results:
        by_model.setdefault(row["model"], []).append(row)
    out: Dict[str, str] = {}
    for model_name, rows in by_model.items():
        failures: List[str] = []
        for row in rows:
            at = row["by_k"].get(str(target))
            if at is None:
                failures.append(f"{row['mode']}: k={target} not measured")
                continue
            floor = at["floor_applied"]
            ratio = at["gap_applied"] / floor if floor else float("inf")
            if not ratio <= FLOOR_FACTOR:
                failures.append(f"{row['mode']}: gap/floor = {ratio:.1f} > {FLOOR_FACTOR}")
            longer = [k for k in sorted(int(k) for k in row["by_k"]) if k > target]
            if longer:
                later = row["by_k"][str(longer[0])]["gap_applied"]
                if later > 0 and at["gap_applied"] / later > IMPROVEMENT_FACTOR:
                    failures.append(f"{row['mode']}: still improving "
                                    f"{at['gap_applied'] / later:.1f}x to k={longer[0]}")
        out[model_name] = ("confirmed" if not failures
                           else "NOT confirmed -- " + "; ".join(failures))
    return out


if __name__ == "__main__":
    main()
