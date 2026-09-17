"""Is ``lambda`` negligible against the real curvature scale? Per layer, per model, per mode.

Follow-up to ``docs/reports/validation_noise_investigation.md``. That investigation established
(a) the Fisher modes' extra noise sits entirely on the held-out side, not the training side, and
(b) across nine runs the only variable tracking it is model *size* (corr -0.68), not the share of
normalisation layers (corr +0.20). The proposed reading of (b) is that "size" is a stand-in for
the ratio between the **absolute** constant ``Lambda = 1e-3`` and each layer's own curvature
scale, which depends on widths and activation magnitudes:

  * ``Lambda`` >> curvature  -> damping dominates every direction, every direction gets roughly the
    same step, the method degenerates towards plain momentum. Calm, barely second-order.
  * ``Lambda`` << curvature  -> the flattest directions are amplified by up to 1/Lambda. Aggressive,
    and bumpy on held-out data.

This script measures that ratio directly.

Three numbers per (model, mode, layer), all read off the spectrum of ``F~``:

  curv/damp        median(undamped spectrum) / median(damping contribution). >> 1 means Lambda is
                   negligible; << 1 means Lambda runs the show.
  frac_damp_dom    fraction of the spectrum where the damping term exceeds the curvature itself.
  dyn_range        p99 / p01 of the *applied* divisor -- how much more the flattest directions are
                   amplified than the steepest. This is the quantity the noise should track.

Damping is not a uniform additive shift in every mode, which is why the script works with both the
undamped and the damped spectrum rather than with a single scalar:

  diag    F~ = outer(S, H) + Lambda                     (H, S min-max'd into [0,1])
  kfac    F~ = kron(B + sqrt(L)/pi I, A + pi sqrt(L) I) (damping split across the two factors)
  ekfac   F~ = s* + Lambda                              (s* IS the spectrum, in its own eigenbasis)
  tkfac   F~ = delta * kron(Psi + d, Phi + d)           (d = sqrt(Lambda/delta))
  tekfac  F~ = Theta + Lambda                           (Theta IS the spectrum)

Protocol. The checkpoints hold theta only, so the EMA state must be re-warmed before it can be
read. ``rewarm_fidelity.py`` established the rule -- every mode seeds its EMA with the IDENTITY,
which acts as a spurious extra damping of ``0.08^k`` on top of Lambda, so the requirement is
``0.08^k << Lambda``, i.e. **k = 10 factor updates**, not 3. Hence REWARM_STEPS = 10 * TCov. Each
mode is re-warmed at *its own* ckpt_0.5, i.e. in the conditions that arm actually ran in, and with
that run's own hyperparameters read from its manifest.

Eigenvalues are computed on the CPU on purpose: cuSOLVER raises on the exactly rank-deficient
factors this project has already hit in production (see ``approximations/_eigh_utils.py``).

Env vars: REWARM_STEPS, CKPT_FRACTION, DATA_ROOT, MODELS (comma-separated run-directory names).
"""
import inspect
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
import torch.nn as nn

from adafisher_modes.optimizer import SUPPORTED_MODULES
from benchmarks.common.optimizers import HParams, build_optimizer
from benchmarks.common.runner import discover_benchmarks

REWARM_STEPS = int(os.environ.get("REWARM_STEPS", "1000"))   # 10 * TCov
FRACTION = os.environ.get("CKPT_FRACTION", "0.5")
DATA_ROOT = os.environ.get("DATA_ROOT", str(ROOT / "benchmarks/data"))
MODES = ["diag", "kfac", "ekfac", "tkfac", "tekfac"]
NORM_TYPES = (nn.BatchNorm2d, nn.LayerNorm)

# (bench name, output group, run-directory name, build_model kwargs). The run directory is what
# identifies the trajectory; the bench name is what builds the network.
ALL_RUNS = [
    ("mlp_ln_mnist",      "mnist",   "mlp_ln_mnist",      {}),
    ("mnist_autoencoder", "mnist",   "mnist_autoencoder", {}),
    ("cnn_gn_cifar",      "cifar10", "cnn_gn_cifar_cpu",  {}),
    ("cnn_gn_cifar",      "cifar10", "cnn_gn_cifar_bn",   {"norm": "bn"}),
    ("vit_micro_cifar",   "cifar10", "vit_micro_cifar",   {}),
    ("resnet20_cifar",    "cifar10", "resnet20_cifar",    {}),
    ("cct_2_3x2_cifar",   "cifar10", "cct_2_3x2_cifar",   {}),
]
if os.environ.get("MODELS"):
    want = {s.strip() for s in os.environ["MODELS"].split(",")}
    ALL_RUNS = [r for r in ALL_RUNS if r[2] in want]

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BENCHES = discover_benchmarks()


def build_loader(bench, hp_batch):
    sig = inspect.signature(bench.build_data).parameters
    kw = {"batch_size": hp_batch, "seed": 0, "num_workers": 2, "allow_download": False}
    if "cutout" in sig:
        kw["cutout"] = True
    train, _, _ = bench.build_data(DATA_ROOT, **{k: v for k, v in kw.items() if k in sig})
    return train


def rewarm(model, opt, bench, loader, steps):
    it, done = iter(loader), 0
    while done < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        x, y = bench.prepare_batch(batch)
        x, y = x.to(DEV), y.to(DEV)
        opt.zero_grad()
        bench.loss_fn(model(x), y).backward()
        opt.step()
        done += 1


def spectra(approx, mode, module):
    """(undamped spectrum of F~, damped/applied spectrum of F~), both 1-D, on the CPU."""
    cpu = lambda t: t.detach().to("cpu", torch.float64)
    lam = approx.Lambda
    if mode == "diag":
        H, S = cpu(approx._H[module]), cpu(approx._S[module])
        und = torch.outer(S, H).flatten()
        return und, und + lam
    if mode == "ekfac":
        und = cpu(approx._s_star[module]).flatten()
        return und, und + lam
    if mode == "tekfac":
        und = cpu(approx._Theta[module]).flatten()
        return und, und + lam
    if mode == "kfac":
        A, B = cpu(approx._A[module]), cpu(approx._B[module])
        pi = ((A.trace() / A.size(0)) / (B.trace() / B.size(0))).sqrt() if approx.pi else torch.tensor(1.0, dtype=A.dtype)
        sl = lam ** 0.5
        a = torch.linalg.eigvalsh(A).clamp_min(0.0)
        b = torch.linalg.eigvalsh(B).clamp_min(0.0)
        return torch.outer(b, a).flatten(), torch.outer(b + sl / pi, a + pi * sl).flatten()
    if mode == "tkfac":
        delta = cpu(approx._delta[module])
        Phi = cpu(approx._Phi_raw[module]) / delta
        Psi = cpu(approx._Psi_raw[module]) / delta
        d = (lam / delta).sqrt()
        p = torch.linalg.eigvalsh(Phi).clamp_min(0.0)
        q = torch.linalg.eigvalsh(Psi).clamp_min(0.0)
        return (delta * torch.outer(q, p)).flatten(), (delta * torch.outer(q + d, p + d)).flatten()
    raise ValueError(mode)


def summarise(und, dmp):
    damp = (dmp - und).clamp_min(0.0)
    med_u, med_d = und.median().item(), damp.median().item()
    q = torch.quantile(dmp, torch.tensor([0.01, 0.99], dtype=dmp.dtype))
    return {
        "curv_over_damp": (med_u / med_d) if med_d > 0 else float("inf"),
        "frac_damp_dom": (und < damp).double().mean().item(),
        "dyn_range": (q[1] / q[0]).item() if q[0] > 0 else float("inf"),
        "n": und.numel(),
    }


def geo(xs):
    xs = [x for x in xs if x == x and 0 < x < float("inf")]
    if not xs:
        return float("nan")
    return float(torch.tensor(xs, dtype=torch.float64).log().mean().exp())


def main():
    print(f"lambda vs curvature | device={DEV} | re-warm={REWARM_STEPS} steps | ckpt_{FRACTION}", flush=True)
    print(f"data_root={DATA_ROOT}\n", flush=True)

    results = {}
    for bench_name, group, run_dir, build_kw in ALL_RUNS:
        root = ROOT / "benchmarks/outputs" / group / run_dir
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            print(f"[skip] {run_dir}: no manifest", flush=True); continue
        cfg = json.load(open(manifest_path))["config"]
        hp = HParams(**cfg["hparams"])
        bench = BENCHES[bench_name]
        loader = build_loader(bench, cfg.get("batch_size", 128))
        print(f"===== {run_dir}  (bench={bench_name}, lambda={hp.lam}, TCov={hp.tcov}) =====", flush=True)
        results[run_dir] = {}

        for mode in MODES:
            ckpt = root / mode / f"ckpt_{FRACTION}.pt"
            if not ckpt.exists():
                print(f"  [skip] {mode}: no {ckpt.name}", flush=True); continue
            t0 = time.time()
            torch.manual_seed(0)
            model = bench.build_model(**build_kw).to(DEV)
            payload = torch.load(ckpt, map_location=DEV, weights_only=True)
            model.load_state_dict(payload["model_state_dict"])
            opt = build_optimizer(mode, model, hp)
            rewarm(model, opt, bench, loader, REWARM_STEPS)

            names = {id(m): n for n, m in model.named_modules()}
            per_layer = []
            for m in model.modules():
                if type(m).__name__ not in SUPPORTED_MODULES:  # a tuple of class names, not types
                    continue
                try:
                    und, dmp = spectra(opt.approx, mode, m)
                except (KeyError, RuntimeError) as exc:
                    print(f"    [warn] {names.get(id(m))}: {type(exc).__name__}", flush=True); continue
                s = summarise(und, dmp)
                s["layer"] = names.get(id(m), "?")
                s["kind"] = "norm" if isinstance(m, NORM_TYPES) else type(m).__name__
                per_layer.append(s)

            results[run_dir][mode] = per_layer
            agg = geo([s["curv_over_damp"] for s in per_layer])
            rng = geo([s["dyn_range"] for s in per_layer])
            fdd = sum(s["frac_damp_dom"] for s in per_layer) / max(len(per_layer), 1)
            print(f"  {mode:7s} layers={len(per_layer):3d}  curv/damp={agg:10.3g}  "
                  f"dyn_range={rng:10.3g}  frac_damp_dom={fdd:5.2f}  ({time.time()-t0:.0f}s)", flush=True)
            for s in sorted(per_layer, key=lambda s: s["curv_over_damp"])[:3]:
                print(f"      lowest curv/damp: {s['layer'][:38]:38s} {s['kind']:11s} "
                      f"{s['curv_over_damp']:10.3g}  dyn={s['dyn_range']:9.3g}  n={s['n']}", flush=True)
        print(flush=True)

    out = ROOT / "fisher_ref/outputs"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "lambda_vs_curvature.json", "w") as f:
        json.dump({"rewarm_steps": REWARM_STEPS, "fraction": FRACTION, "device": str(DEV),
                   "results": results}, f, indent=1)

    print("===== SUMMARY: geometric mean over layers, per model =====", flush=True)
    print(f"{'run':>20s} " + " ".join(f"{m:>12s}" for m in MODES), flush=True)
    for key, label in [("curv_over_damp", "curv/damp"), ("dyn_range", "dyn_range")]:
        print(f"-- {label}", flush=True)
        for run_dir, modes in results.items():
            cells = []
            for m in MODES:
                cells.append(f"{geo([s[key] for s in modes[m]]):12.4g}" if m in modes else f"{'-':>12s}")
            print(f"{run_dir:>20s} " + " ".join(cells), flush=True)
    print(f"\nwrote {out/'lambda_vs_curvature.json'}", flush=True)


# The guard is load-bearing, not boilerplate: on macOS the DataLoader's workers are started by
# `spawn`, which re-imports this file, and without it they hang at 0% CPU. Linux forks instead,
# which is why the cluster never saw it.
if __name__ == "__main__":
    main()
