"""How big does curvature ever get, per layer, on every network tested?

Follow-up to ``lambda_vs_curvature.py``, which established that the **typical** (median) curvature
is a small fraction of the damping constant in every layer of every mode of every network -- but
only checked the **largest** value per layer on two networks. A median cannot see a handful of large
outlier directions, and those are exactly what would explain why a plain-momentum stand-in
reproduces one network's training almost exactly and neither convolutional network's.

This script closes that gap: the same per-layer maximum check, on every network
``lambda_vs_curvature.py`` covers. It reuses that module's re-warm and spectrum machinery
unchanged -- same checkpoint, same re-warm length, same five modes -- and only the summary changes,
from "median of the spectrum" to "how much of the spectrum's tail clears the damping constant".

Per (network, mode, layer): the maximum, the 99.9th, 99th and 50th percentile of the *undamped*
curvature, each read as a multiple of the damping constant, plus the fraction of directions whose
curvature alone exceeds it.

Environment variables: the same as ``lambda_vs_curvature.py`` -- ``REWARM_STEPS`` (default 1000),
``CKPT_FRACTION`` (default 0.5), ``DATA_ROOT`` (default ``benchmarks/data``), ``MODELS``.

Output: ``fisher_ref/outputs/curvature_max_per_layer.json`` plus a table on standard output.

It reuses ``lambda_vs_curvature``'s re-warm and spectrum machinery, imported by its full package
path, so both ``python -m fisher_ref.experiments.curvature_max_per_layer`` and running this file by
path work. The repository root and ``src`` go on the import path first, which is what makes the
package path resolvable in the second form.
"""
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path[:0] = [str(_ROOT), str(_ROOT / "src")]

from adafisher_modes.optimizer import SUPPORTED_MODULES
from fisher_ref.experiments.lambda_vs_curvature import (
    ALL_RUNS,
    BENCHES,
    DEV,
    FRACTION,
    HParams,
    MODES,
    REWARM_STEPS,
    ROOT,
    build_optimizer,
    build_loader,
    rewarm,
    spectra,
)


def summarise_max(und: torch.Tensor, lam: float) -> dict:
    ratio = und / lam
    q = torch.quantile(ratio, torch.tensor([0.5, 0.99, 0.999], dtype=ratio.dtype))
    return {
        "max_over_lambda": ratio.max().item(),
        "p999_over_lambda": q[2].item(),
        "p99_over_lambda": q[1].item(),
        "p50_over_lambda": q[0].item(),
        "frac_curvature_exceeds_lambda": (und > lam).double().mean().item(),
        "n": und.numel(),
    }


def main() -> None:
    print(f"curvature_max_per_layer | device={DEV} | re-warm={REWARM_STEPS} steps | "
          f"ckpt_{FRACTION}\n", flush=True)

    results: dict = {}
    worst_per_run_mode: list[dict] = []

    for bench_name, group, run_dir, build_kw in ALL_RUNS:
        root = ROOT / "benchmarks/outputs" / group / run_dir
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            print(f"[skip] {run_dir}: no manifest", flush=True)
            continue
        cfg = json.load(open(manifest_path))["config"]
        hp = HParams(**cfg["hparams"])
        bench = BENCHES[bench_name]
        loader = build_loader(bench, cfg.get("batch_size", 128))
        print(f"===== {run_dir} (bench={bench_name}, lambda={hp.lam}) =====", flush=True)
        results[run_dir] = {}

        for mode in MODES:
            ckpt = root / mode / f"ckpt_{FRACTION}.pt"
            if not ckpt.exists():
                print(f"  [skip] {mode}: no {ckpt.name}", flush=True)
                continue
            t0 = time.time()
            torch.manual_seed(0)
            model = bench.build_model(**build_kw).to(DEV)
            payload = torch.load(ckpt, map_location=DEV, weights_only=True)
            model.load_state_dict(payload["model_state_dict"])
            opt = build_optimizer(mode, model, hp)
            rewarm(model, opt, bench, loader, REWARM_STEPS)

            names = {id(m): n for n, m in model.named_modules()}
            per_layer = []
            worst = None
            for m in model.modules():
                if type(m).__name__ not in SUPPORTED_MODULES:
                    continue
                try:
                    und, _ = spectra(opt.approx, mode, m)
                except (KeyError, RuntimeError) as exc:
                    print(f"    [warn] {names.get(id(m))}: {type(exc).__name__}", flush=True)
                    continue
                s = summarise_max(und, hp.lam)
                s["layer"] = names.get(id(m), "?")
                s["kind"] = type(m).__name__
                per_layer.append(s)
                if worst is None or s["max_over_lambda"] > worst["max_over_lambda"]:
                    worst = s

            results[run_dir][mode] = per_layer
            n_layers = len(per_layer)
            n_over = sum(1 for s in per_layer if s["max_over_lambda"] >= 1.0)
            print(f"  {mode:7s} layers={n_layers:3d}  worst layer={worst['layer'][:30]:30s} "
                  f"({worst['kind']})  max/lambda={worst['max_over_lambda']:10.3g}  "
                  f"p99.9/lambda={worst['p999_over_lambda']:10.3g}  "
                  f"layers_with_any_dir_over_lambda={n_over}/{n_layers}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            worst_per_run_mode.append({
                "run": run_dir, "mode": mode, "n_layers": n_layers,
                "n_layers_with_dir_over_lambda": n_over, **worst,
            })
        print(flush=True)

    out = ROOT / "fisher_ref/outputs"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "curvature_max_per_layer.json", "w") as f:
        json.dump({"rewarm_steps": REWARM_STEPS, "fraction": FRACTION, "device": str(DEV),
                   "results": results}, f, indent=1)

    print("===== SUMMARY: worst single layer per (network, mode) =====", flush=True)
    print(f"{'run':>20s} {'mode':>7s} {'max/lambda':>11s} {'p99.9/lambda':>13s} "
          f"{'layer':>32s} {'kind':>13s} {'layers>lambda':>14s}", flush=True)
    for w in worst_per_run_mode:
        print(f"{w['run']:>20s} {w['mode']:>7s} {w['max_over_lambda']:11.3g} "
              f"{w['p999_over_lambda']:13.3g} {w['layer'][:32]:>32s} {w['kind']:>13s} "
              f"{w['n_layers_with_dir_over_lambda']:>6d}/{w['n_layers']:<7d}", flush=True)
    print(f"\nwrote {out / 'curvature_max_per_layer.json'}", flush=True)


if __name__ == "__main__":
    main()
