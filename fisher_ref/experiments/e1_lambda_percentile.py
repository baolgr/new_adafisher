"""Where does the damping constant sit inside the curvature spectrum, as a percentile?

``lambda_vs_curvature.py`` reported the *median* curvature against the damping, and
``curvature_max_per_layer.py`` the single *largest* value per layer. Both answered "is the damping
bigger than the curvature?" -- yes, everywhere, by a lot. Neither answered the question a damping
constant actually poses, which is **where in the distribution it sits**: a floor at the 5th
percentile is doing the job it exists for; a floor at the 100th percentile *is* the optimizer.

Two scales are reported for every layer, because the interesting question is conditional.

``as shipped``  the spectrum the optimizer really uses today.

``corrected``   the same spectrum with the two known, purely multiplicative shrinkages undone. This
is an **analytic** rescaling, not a re-measurement: it assumes the running average has reached its
fixed point and that the statistic is stationary over the averaging window. ``e4_fixed_average.py``
is the version that actually retrains with the corrected conventions; this is the cheap preview,
and the exact Fisher built by :mod:`fisher_ref.reference.dense` is the independent anchor, since it
uses neither convention.

The two shrinkages, and why the exponents differ per mode::

  running average   stored <- (1-g0)*stored + g1*new settles at g1/g0 = 1/115 of what it
                    estimates. Applied ONCE per stored tensor, so the exponent is the number of
                    averaged tensors the undamped spectrum is built from:
                      diag    H and S            -> 2
                      kfac    A and B            -> 2
                      ekfac   s* only            -> 1   (the eigenbasis is scale-free)
                      tkfac   three averaged tensors, net -> 1
                      tekfac  Theta only         -> 1
  batch average     the backward signal is the gradient of the batch-MEAN loss, so each example
                    carries 1/batch; curvature is that signal squared -> 1/batch^2. Applies to
                    every mode EXCEPT diag with its renormalisation applied before the average,
                    where squeezing each factor into [0,1] destroys the scale before it is stored.

Outputs, per (run, mode), per layer and pooled over the whole network::

  lambda_pct     percentile of the damping constant within the undamped spectrum (100 means it sits
                 above everything, i.e. no direction is preconditioned at all)
  frac_above     fraction of directions whose curvature alone exceeds it
  lambda_star_p  the damping that WOULD sit at the p-th percentile, for p in {1, 10, 50}

The protocol matches ``lambda_vs_curvature.py`` exactly -- same re-warm length, same checkpoint,
same per-run hyperparameters read from the manifest -- so the two are directly comparable.

Environment variables: ``REWARM_STEPS`` (default 1000), ``CKPT_FRACTION`` (default 0.5),
``DATA_ROOT`` (default ``benchmarks/data``), ``MODELS``, ``MODES`` (default all five),
``BATCH_SIZES`` (comma-separated; empty means each run's own).

Output: ``fisher_ref/outputs/e1_lambda_percentile.json`` plus a table on standard output.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
import torch
from adafisher_modes.optimizer import SUPPORTED_MODULES

from benchmarks.common.optimizers import HParams, build_optimizer

# Reuse Step 7's machinery verbatim rather than re-deriving it: same spectra, same re-warm, same
# loader, so any difference between this file's numbers and lambda_vs_curvature.json's is the
# metric, never the setup.
from fisher_ref.experiments.lambda_vs_curvature import (
    ALL_RUNS,
    BENCHES,
    DEV,
    FRACTION,
    REWARM_STEPS,
    build_loader,
    rewarm,
    spectra,
)

MODES = [m.strip() for m in os.environ.get("MODES", "diag,kfac,ekfac,tkfac,tekfac").split(",")]
# E3 (plan_lambda_dominance.md Part 3): the stored curvature is proportional to 1/batch^2, so the
# batch size moves the very ratio this file measures, with no code change anywhere. Set
# BATCH_SIZES to a comma-separated list to re-warm the same checkpoint at several of them. The
# prediction is sharp and falsifiable: the "as shipped" spectrum moves as 1/batch^2, so the
# "corrected" spectrum -- which multiplies by batch^2 -- must come out INVARIANT. Empty = use the
# batch size the run itself was trained at, read from its manifest.
BATCH_SIZES = [int(b) for b in os.environ.get("BATCH_SIZES", "").split(",") if b.strip()]
PCTS = (1.0, 10.0, 50.0)

# Number of times the (1/115) running-average shrinkage is baked into the undamped spectrum.
EMA_EXPONENT = {"diag": 2, "kfac": 2, "ekfac": 1, "tkfac": 1, "tekfac": 1}

# Every mode seeds its state with the identity, and that seed decays by (1-gammas[0]) at each EMA
# update, so after k updates a residue of (1-gammas[0])^k is still sitting in the stored tensor.
# For ``ekfac``/``tekfac`` that residue lands DIRECTLY in the spectrum, additively, because ``s*``
# and ``Theta`` *are* the spectrum -- and it is exactly computable, which makes it a floor the
# measurement can be checked against rather than a vague worry.
#
# The count k is one lower for those two modes than for the other three, and not by accident:
# ``s*``/``Theta`` are only updated once an eigenbasis exists (``ekfac.py``'s
# ``if module in self._Q_A``), and the eigenbasis is created by the first ``refresh()``, which runs
# *after* the first set of hooks. So over a re-warm of ``REWARM_STEPS`` steps at cadence ``TCov``,
# ``A``/``B`` get ceil(steps/TCov) updates and ``s*``/``Theta`` get one fewer.
#
# Measured, cluster job 21283680, 1000 re-warm steps (k=10, so k-1=9 -> 0.08^9 = 1.342e-10): the
# mean of ekfac's and tekfac's spectrum at batch 512 came out at 1.34e-7, 1.35e-7, 1.59e-7, 1.69e-7
# times Lambda on the four models with Lambda=1e-3 and 4.51e-8, 4.85e-8 on the two with
# Lambda=3e-3 -- i.e. 1.00x to 1.26x the predicted 0.08^9/Lambda, tracking the SEED across two
# different values of Lambda rather than anything about the network. At that batch size those two
# modes are not measuring curvature at all. Hence the flag below, and hence a re-warm long enough
# that the residue falls under the quantity being measured -- NOT merely under Lambda, which is the
# criterion rewarm_fidelity.py established for the *applied preconditioner* and which is too weak
# for the spectrum itself.
SEED_CONTAMINATION_FACTOR = 3.0   # flagged when the measured mean is under this multiple of the floor


def seed_residue(mode, approx, rewarm_steps, tcov):
    """Exact leftover of the identity seed in the spectrum, or None when it is not additive there.

    Returned only for ``ekfac``/``tekfac``, where ``s*``/``Theta`` are the spectrum and the residue
    is a plain additive term. For ``diag``/``kfac``/``tkfac`` the seed sits inside factors that are
    multiplied together, so the pure-seed term is the square of this and the dominant contamination
    is a seed x data cross term whose size depends on the data -- real, but not exactly computable
    here, so it is not reported rather than reported wrongly.
    """
    if mode not in ("ekfac", "tekfac"):
        return None
    g0, _ = ema_rates(mode, approx)
    k = max(-(-rewarm_steps // max(tcov, 1)) - 1, 0)   # one fewer than A/B, see the comment above
    return (1.0 - g0) ** k


def ema_rates(mode, approx):
    """The (g0, g1) pair that actually averaged the tensor the spectrum is read from.

    ``tekfac`` renames its two EMA rates ``beta_factors``/``beta_theta`` to avoid the notation
    collision with Adam's betas (CLAUDE.md), and the spectrum it reports is ``Theta``, so it is
    ``beta_theta`` that shrinks it -- not the ``gammas`` every other mode exposes.
    """
    if mode == "tekfac":
        return approx.beta_theta[0], approx.beta_theta[1]
    return approx.gammas[0], approx.gammas[1]


def correction_factor(mode, approx, batch_size):
    """Multiplier taking the stored spectrum back to the scale it estimates. See the header."""
    g0, g1 = ema_rates(mode, approx)
    fixed_point = g1 / g0                      # what one averaged tensor settles at, per unit
    ema = fixed_point ** (-EMA_EXPONENT[mode])
    minmax_kills_scale = mode == "diag" and getattr(approx, "minmax_normalization", False) \
        and not getattr(approx, "minmax_after_average", False)
    batch = 1.0 if minmax_kills_scale else float(batch_size) ** 2
    return ema * batch, ema, batch


def describe(und, lam):
    """Where Lambda sits in this (1-D, non-negative) spectrum."""
    x = np.sort(np.asarray(und, dtype=np.float64))
    n = x.size
    below = float(np.searchsorted(x, lam, side="left")) / n
    out = {
        "n": int(n),
        "lambda_pct": 100.0 * below,
        "frac_above": float((x > lam).mean()),
        "max_over_lambda": float(x[-1] / lam),
        "median_over_lambda": float(x[n // 2] / lam),
        # The MEAN is the statistic to read for scale questions (E3). These spectra are massively
        # rank-deficient -- mlp_ln_mnist's first input factor has 136 exactly-zero eigenvalues out
        # of 785, because 130 MNIST pixels are identically zero -- so their median sits in the
        # numerical floor and moves for reasons that have nothing to do with the quantity being
        # estimated. The mean is the trace divided by the dimension, which is dominated by the
        # large eigenvalues and is exactly the quantity the 1/batch^2 argument is about. Step 7 of
        # validation_noise_investigation.md hit the same trap from the other side (its medians were
        # reporting the identity seed's residue); this is the same lesson in a new place.
        "mean_over_lambda": float(x.mean() / lam),
        "p90_over_lambda": float(np.quantile(x, 0.90) / lam),
        "p99_over_lambda": float(np.quantile(x, 0.99) / lam),
    }
    for p in PCTS:
        out[f"lambda_star_p{int(p)}"] = float(np.quantile(x, p / 100.0))
    return out


def main():
    print(f"E1: percentile of Lambda in the curvature spectrum | device={DEV} | "
          f"re-warm={REWARM_STEPS} | ckpt_{FRACTION}", flush=True)

    results = {}
    for bench_name, group, run_dir, build_kw in ALL_RUNS:
        root = ROOT / "benchmarks/outputs" / group / run_dir
        if not (root / "manifest.json").exists():
            print(f"[skip] {run_dir}: no manifest", flush=True)
            continue
        cfg = json.load(open(root / "manifest.json"))["config"]
        hp = HParams(**cfg["hparams"])
        bench = BENCHES[bench_name]
        native_batch = cfg.get("batch_size", 128)
        batches = BATCH_SIZES or [native_batch]
        print(f"\n===== {run_dir}  (lambda={hp.lam}, native batch={native_batch}, "
              f"TCov={hp.tcov}) =====", flush=True)
        results[run_dir] = {"lambda": hp.lam, "native_batch_size": native_batch, "by_batch": {}}

        for batch_size in batches:
            loader = build_loader(bench, batch_size)
            if len(batches) > 1:
                print(f"  --- batch {batch_size} ---", flush=True)
            results[run_dir]["by_batch"][str(batch_size)] = {}
            modes_blob = results[run_dir]["by_batch"][str(batch_size)]

            for mode in MODES:
                ckpt = root / mode / f"ckpt_{FRACTION}.pt"
                if not ckpt.exists():
                    print(f"  [skip] {mode}: no {ckpt.name}", flush=True)
                    continue
                t0 = time.time()
                torch.manual_seed(0)
                model = bench.build_model(**build_kw).to(DEV)
                model.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=True)["model_state_dict"])
                opt = build_optimizer(mode, model, hp)
                rewarm(model, opt, bench, loader, REWARM_STEPS)

                lam = opt.approx.Lambda
                corr, ema_part, batch_part = correction_factor(mode, opt.approx, batch_size)
                names = {id(m): n for n, m in model.named_modules()}
                per_layer, pooled_shipped, pooled_corrected = [], [], []
                for m in model.modules():
                    if type(m).__name__ not in SUPPORTED_MODULES:
                        continue
                    try:
                        und, _ = spectra(opt.approx, mode, m)
                    except (KeyError, RuntimeError) as exc:
                        print(f"    [warn] {names.get(id(m))}: {type(exc).__name__}", flush=True)
                        continue
                    und = und.clamp_min(0.0).numpy()
                    rec = {"layer": names.get(id(m), "?"), "kind": type(m).__name__,
                           "shipped": describe(und, lam), "corrected": describe(und * corr, lam)}
                    per_layer.append(rec)
                    pooled_shipped.append(und)
                    pooled_corrected.append(und * corr)

                if not per_layer:
                    continue
                pooled = {"shipped": describe(np.concatenate(pooled_shipped), lam),
                          "corrected": describe(np.concatenate(pooled_corrected), lam)}
                floor = seed_residue(mode, opt.approx, REWARM_STEPS, hp.tcov)
                contaminated = (floor is not None and
                                pooled["shipped"]["mean_over_lambda"] * lam
                                < SEED_CONTAMINATION_FACTOR * floor)
                modes_blob[mode] = {
                    "correction": {"total": corr, "ema": ema_part, "batch": batch_part},
                    "seed_floor": floor,
                    "seed_floor_over_lambda": None if floor is None else floor / lam,
                    "mean_over_seed_floor": None if floor is None else
                        pooled["shipped"]["mean_over_lambda"] * lam / floor,
                    "seed_contaminated": contaminated,
                    "pooled": pooled, "layers": per_layer,
                }
                s, c = pooled["shipped"], pooled["corrected"]
                note = ""
                if floor is not None:
                    ratio = pooled["shipped"]["mean_over_lambda"] * lam / floor
                    note = (f"  seed_floor={floor / lam:.3g}xlambda  mean/floor={ratio:6.2f}"
                            f"{'  [SEED-CONTAMINATED: not measuring curvature]' if contaminated else ''}")
                print(f"  {mode:7s} x{corr:9.3g} (ema x{ema_part:.4g}, batch x{batch_part:.4g})  "
                      f"layers={len(per_layer):3d}  n={s['n']:8d}{note}", flush=True)
                print(f"          as shipped : lambda at pct {s['lambda_pct']:6.2f}   "
                      f"frac above {s['frac_above']:.4f}   max/lambda {s['max_over_lambda']:9.3g}",
                      flush=True)
                print(f"          corrected  : lambda at pct {c['lambda_pct']:6.2f}   "
                      f"frac above {c['frac_above']:.4f}   max/lambda {c['max_over_lambda']:9.3g}   "
                      f"lambda*(p10)={c['lambda_star_p10']:.3g}  lambda*(p50)={c['lambda_star_p50']:.3g}"
                      f"   ({time.time()-t0:.0f}s)", flush=True)

    out = ROOT / "fisher_ref/outputs"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "e1_lambda_percentile.json"
    with open(dest, "w") as f:
        json.dump({"rewarm_steps": REWARM_STEPS, "fraction": FRACTION, "device": str(DEV),
                   "ema_exponent": EMA_EXPONENT, "results": results}, f, indent=1)

    print("\n===== SUMMARY: percentile of Lambda in the pooled spectrum =====", flush=True)
    print(f"{'run':>20s} {'batch':>6s} " + " ".join(f"{m:>16s}" for m in MODES), flush=True)
    for scale in ("shipped", "corrected"):
        print(f"-- {scale}  (percentile of lambda / fraction of directions above it)", flush=True)
        for run_dir, blob in results.items():
            for bs, modes_blob in blob["by_batch"].items():
                cells = []
                for m in MODES:
                    if m in modes_blob:
                        q = modes_blob[m]["pooled"][scale]
                        cells.append(f"{q['lambda_pct']:7.2f}/{q['frac_above']:<8.4f}")
                    else:
                        cells.append(f"{'-':>16s}")
                print(f"{run_dir:>20s} {bs:>6s} " + " ".join(cells), flush=True)

    # E3's own read-out: the median of the corrected spectrum must not move with the batch size.
    multi = {r: b for r, b in results.items() if len(b["by_batch"]) > 1}
    if multi:
        print("\n===== E3: is the corrected spectrum batch-size invariant? =====", flush=True)
        print("(MEAN of the pooled spectrum / lambda -- the trace over the dimension, see "
              "describe(); 'shipped' should move as 1/batch^2, 'corrected' should not move)",
              flush=True)
        for run_dir, blob in multi.items():
            for m in MODES:
                row = []
                for bs, modes_blob in blob["by_batch"].items():
                    if m not in modes_blob:
                        continue
                    row.append((bs, modes_blob[m]["pooled"]["shipped"]["mean_over_lambda"],
                                modes_blob[m]["pooled"]["corrected"]["mean_over_lambda"]))
                if not row:
                    continue
                ship = " ".join(f"b{bs}={v:9.3g}" for bs, v, _ in row)
                corr = " ".join(f"b{bs}={w:9.3g}" for bs, _, w in row)
                print(f"  {run_dir:>20s} {m:7s} shipped   {ship}", flush=True)
                print(f"  {'':>20s} {'':7s} corrected {corr}", flush=True)
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
