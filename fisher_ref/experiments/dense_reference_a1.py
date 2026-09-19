"""The dense reference of one small model at one checkpoint, in float64.

Builds the true Fisher, the empirical Fisher and the per-layer blocks of a regime-A model, and
reports the three distances the campaign's first questions need::

    PYTHONPATH=src:. python -u fisher_ref/experiments/dense_reference_a1.py

What it produces, at enough type-2 probes that the reference is not rank-limited by the probe
count:

* the true Fisher and the empirical Fisher, dense, float64;
* their spectra, and the per-block trace shares and ranks;
* the **source gap** between them, with no structure involved;
* the **train/validation gap**, i.e. how much the curvature differs on data the run saw and data
  it did not;
* the **noise floor** from a two-way split of the probes: any later difference below it is not
  interpretable.

Only summaries and the two spectra are written to disk. The matrices themselves are gigabytes and
recomputable in minutes.

Memory, which is the whole reason this has its own job. The accumulator is the only ``P x P``
tensor on the device, so the build peaks at about ``1.05 P^2`` and fits a 10 GB slice. Everything
with more than one ``P x P`` live at once -- the two references side by side, and the
eigendecomposition -- happens on the **host**, where memory is cheap. Measured on CPU at
``P = 12 030``: ``1.27 P^2`` for the whole build, against ``+1.16 P^2`` for the single expression
``0.5 * (M + M.T)`` this code no longer uses.

Result, ``mlp_ln_mnist`` at half its trajectory, 55 000 probes, one hour on an H100 slice, peak
device 6.06 GB::

    builds       train type2 147.4 s | empirical 15.0 s | val/test 13.5 s | halves 73.6 s
    noise floor  0.2986 at 27 500 probes per half  ->  sigma_N = 0.1493
    source gap   0.3751   =  2.51 sigma_N          ->  interpretable
    train/val    0.7121   ~ 1.4 x its null         ->  not settled
    val/test     0.7111   ~ 1.02 x its null        ->  indistinguishable...
    train/test   1.0222   =  1.44 x train/val      ->  ...but this does not fit
    F            lambda_max 9.2124e-01  rank 21 829/26 634  tr 16.68
    E_hat        lambda_max 7.8299e-01  rank 18 342/26 634  tr 11.95
    blocks       features.0 P=25 120 trace 0.627 deficiency 4 579 (MNIST's dead pixels)
                 features.3 P= 1 056 trace 0.208 deficiency    34
                 features.1 P=    64 trace 0.083 deficiency     0
                 head       P=   330 trace 0.066 deficiency    33 (= d_in + 1, the logit shift)
                 features.4 P=    64 trace 0.017 deficiency     0

Two things to carry forward. The probe count is nearly free -- build time is linear in it, 13.2x
for 13.75x the probes -- while one full spectrum costs 1 280 s and does not thread, so budget such
a job from the eigendecomposition and not from the references. And the noise floor fell by 2.92x
for 13.75x the probes where the ideal rate predicts 3.71x: it decays more slowly than
``N^{-1/2}``.

Environment variables::

    A1_MODEL          run directory                (default: mlp_ln_mnist)
    A1_ARM            arm                          (default: diag)
    A1_FRACTION       checkpoint fraction          (default: 0.5)
    A1_PROBES         probe count                  (default: 4000)
    A1_BATCH          traversal micro-batch        (default: 512)
    A1_SEED           probe and split seed         (default: 0)
    A1_DEVICE         cuda or cpu                  (default: cuda when available)
    A1_DATA_ROOT      dataset root                 (default: benchmarks/data)
    A1_OUTPUTS_ROOT   where the checkpoints live   (default: benchmarks/outputs)
    A1_OUT_DIR        where the results go         (default: fisher_ref/outputs)
    A1_MODULES        comma-separated module names to restrict the reference to; empty = the whole
                      model. For debugging on a laptop only.
    A1_BLOCK_SPECTRA  "0" skips the per-block eigendecompositions, keeping the trace shares and
                      dropping the ranks. The widest block alone is 83 % of a full-model
                      decomposition.

Output: ``<A1_OUT_DIR>/<model>/<arm>/<fraction>/reference_summary.json`` and ``spectrum.pt``,
written incrementally, plus a running log on standard output.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks.common.runner import discover_benchmarks  # noqa: E402
from fisher_ref import conventions, probes  # noqa: E402
from fisher_ref.checkpoints import discover_runs, load_theta  # noqa: E402
from fisher_ref.reference import build_dense_reference  # noqa: E402

MODEL = os.environ.get("A1_MODEL", "mlp_ln_mnist")
ARM = os.environ.get("A1_ARM", "diag")
FRACTION = float(os.environ.get("A1_FRACTION", "0.5"))
N_PROBES = int(os.environ.get("A1_PROBES", "4000"))
BATCH = int(os.environ.get("A1_BATCH", "512"))
DATA_ROOT = Path(os.environ.get("A1_DATA_ROOT", ROOT / "benchmarks" / "data"))
OUTPUTS_ROOT = Path(os.environ.get("A1_OUTPUTS_ROOT", ROOT / "benchmarks" / "outputs"))
OUT_DIR = Path(os.environ.get("A1_OUT_DIR", ROOT / "fisher_ref" / "outputs"))
DEVICE = os.environ.get("A1_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("A1_SEED", "0"))
#: Comma-separated module names to restrict the reference to; empty = the whole model. Only for
#: debugging the path on a laptop — at A1's full P the run holds 22 GB of host-side references.
MODULES = [name for name in os.environ.get("A1_MODULES", "").split(",") if name] or None
#: Per-block eigendecompositions. On A1 the widest block is 25 120^2, i.e. 83 % of a full-P
#: decomposition on its own — set to 0 to keep the trace shares and drop the ranks.
BLOCK_SPECTRA = os.environ.get("A1_BLOCK_SPECTRA", "1") != "0"


def gigabytes(matrix: torch.Tensor) -> float:
    return matrix.numel() * matrix.element_size() / 1e9


def peak_device_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if DEVICE.startswith("cuda") else float("nan")


def relative(a: torch.Tensor, b: torch.Tensor, block: int = 4096) -> float:
    """``||a - b||_F / ||b||_F``, accumulated over row blocks.

    The one-liner would allocate a third ``P x P`` tensor (5.68 GB here) at the exact moment four
    references are already live — the host-side twin of the device-side trap ``symmetrize_`` closes.
    """
    difference_sq, reference_sq = _gap_sums(a, b, block)
    return (difference_sq / reference_sq) ** 0.5


def _gap_sums(a: torch.Tensor, b: torch.Tensor, block: int) -> "tuple":
    difference_sq = 0.0
    reference_sq = 0.0
    for start in range(0, a.shape[0], block):
        stop = min(start + block, a.shape[0])
        rows_b = b[start:stop]
        difference_sq += float((a[start:stop] - rows_b).pow(2).sum())
        reference_sq += float(rows_b.pow(2).sum())
    return difference_sq, reference_sq


def gap(a: torch.Tensor, b: torch.Tensor, block: int = 4096) -> dict:
    """``||a - b||_F`` with **both** operands' norms, not just a ratio.

    ``plan_exp_lot1.md`` §6.3 reported three gaps against two different denominators and stored
    only one of them, which is why its val/test result could not be reconciled with its train/test
    one. Recording the absolute distance and both norms makes every ratio recoverable after the
    fact (``plan_exp_lot2.md`` §0.1).
    """
    difference_sq, b_sq = _gap_sums(a, b, block)
    a_sq = float(sum(a[s:s + block].pow(2).sum() for s in range(0, a.shape[0], block)))
    return {"abs": difference_sq ** 0.5, "fro_a": a_sq ** 0.5, "fro_b": b_sq ** 0.5,
            "rel_to_b": (difference_sq / b_sq) ** 0.5,
            "rel_to_a": (difference_sq / a_sq) ** 0.5,
            "rel_to_geom": difference_sq ** 0.5 / (a_sq * b_sq) ** 0.25}


#: Trace and Frobenius norm of every reference this script builds, keyed by the name the gaps use.
#: ``plan_exp_lot1.md`` §6.3 could not be resolved because only the train-side norms were kept, so
#: the three held-out gaps sat on two different denominators with no way to re-normalise them
#: (``plan_exp_lot2.md`` §0.1).
NORMS: dict = {}


def build(model, inputs, targets, source: str, tag: str, name: str) -> "tuple":
    started = time.perf_counter()
    reference = build_dense_reference(model, inputs, targets, source=source, batch_size=BATCH,
                                      device=DEVICE, probe_digest=tag, modules=MODULES)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    matrix = reference.matrix.to("cpu")
    reference.matrix = matrix                      # free the device copy before the next build
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()
    NORMS[name] = {"trace": float(reference.trace()), "fro": float(reference.fro()),
                   "n_probes": reference.n_probes, "source": source, "probe_digest": tag}
    print(f"  {name:<10} {source:<10} {tag:<12} P={reference.P}  rows={reference.n_rows:<7} "
          f"{elapsed:7.1f}s  ||.||_F={NORMS[name]['fro']:.4f}  "
          f"peak device {peak_device_gb():5.2f} GB", flush=True)
    return reference, elapsed


def main() -> None:
    state = conventions.configure()
    torch.manual_seed(SEED)
    print(f"device={DEVICE}  TF32 off={state.tf32_is_off()}  "
          f"metrics_version={conventions.METRICS_VERSION}", flush=True)

    bench = discover_benchmarks()[MODEL]
    # Filter on the seed explicitly. Since the extra-seed campaign landed, (model, arm) matches
    # five runs — seed 0 under outputs/<group>/<model>/ and seeds 1-4 under outputs/seeds/ — and
    # taking runs[0] would pick one by directory ordering ("mnist" < "seeds" today, but that is an
    # accident of the group name, not a rule).
    runs = discover_runs(OUTPUTS_ROOT, model=MODEL, arm=ARM, seed=SEED)
    if not runs:
        available = sorted({r.seed for r in discover_runs(OUTPUTS_ROOT, model=MODEL, arm=ARM)})
        raise SystemExit(
            f"no run for {MODEL}/{ARM} at seed {SEED} under {OUTPUTS_ROOT}; seeds found: {available}"
        )
    if len(runs) > 1:
        raise SystemExit(
            f"{len(runs)} runs match {MODEL}/{ARM} seed {SEED}: "
            f"{[str(r.directory) for r in runs]}. Refusing to guess which theta is meant."
        )
    run = runs[0]
    loaded = load_theta(run, FRACTION, device="cpu")
    print(f"theta: {MODEL}/{ARM} ckpt {FRACTION}  step={loaded.step} epoch={loaded.epoch} "
          f"progress={loaded.progress:.3f}  seed={run.seed}  dir={run.directory}", flush=True)

    # The val split is the run's own held-out partition and is small: MNIST reserves 5 000 images
    # (benchmarks/common/data.py::MNIST_SPEC). Raising N — which §6 says the noise floor demands —
    # therefore raises the train side only, and the train/val gap keeps its own, larger floor.
    spec = probes.dataset_spec_of(bench)
    val_probes = min(N_PROBES, spec.val_size)
    train = probes.build_probe_set(bench, split="train", n=N_PROBES, seed=SEED,
                                   data_root=DATA_ROOT)
    val = probes.build_probe_set(bench, split="val", n=val_probes, seed=SEED, data_root=DATA_ROOT)
    # The dataset's own held-out set, as a *second* out-of-sample source. Built at the same n as
    # the val set so the two are compared like for like (§6.4): if their gap sits inside their
    # combined floor they are interchangeable and a later lot may pool them for a lower HF1 floor;
    # if it does not, that is a distribution shift and pooling would manufacture HF1 signal.
    test = probes.build_probe_set(bench, split="test", n=val_probes, seed=SEED,
                                  data_root=DATA_ROOT)
    print(f"probes: {len(train)} train / {len(val)} val / {len(test)} test"
          f"{' (held-out sets capped by the val split size)' if val_probes < N_PROBES else ''}, "
          f"digests {train.digest[:12]} / {val.digest[:12]} / {test.digest[:12]}", flush=True)
    x_train, y_train = train.as_model_batch(bench)
    x_val, y_val = val.as_model_batch(bench)
    x_test, y_test = test.as_model_batch(bench)

    out = OUT_DIR / MODEL / ARM / f"seed{run.seed}" / str(FRACTION)
    out.mkdir(parents=True, exist_ok=True)

    model = loaded.model
    print("\nbuilds:", flush=True)
    fisher, t_fisher = build(model, x_train, y_train, "type2", train.digest[:12], "F_train")
    empirical, t_empirical = build(model, x_train, y_train, "empirical", train.digest[:12], "E_hat")

    summary = {
        "model": MODEL, "arm": ARM, "seed": run.seed, "fraction": FRACTION,
        "n_probes": N_PROBES, "n_val_probes": len(val), "P": fisher.P,
        "n_rows_type2": fisher.n_rows,
        "batch_size": BATCH, "device": DEVICE, "modules": MODULES,
        "seconds": {"type2": t_fisher, "empirical": t_empirical},
        "peak_device_gb": peak_device_gb(),
        "probe_digest": {"train": train.digest, "val": val.digest, "test": test.digest},
        # Every reference's own trace and norm, so any gap below can be re-normalised by a reader:
        # the failure plan_exp_lot1.md §6.3 could not diagnose (plan_exp_lot2.md §0.1).
        "norms": NORMS,
        "checkpoint": loaded.metadata(),
    }

    def checkpoint_summary() -> None:
        """Write what is known so far.

        The first real run of this job hit its time limit **inside the per-block loop**, having
        already produced every gap and both spectra — and wrote nothing, because the single write
        was at the end. Same lesson as `benchmarks/common/runner.py`'s "rewrite the report after
        every completed arm" (`benchmarks/slurm/README.md`).
        """
        (out / "reference_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    # Q1: the source error, with no structure involved.
    summary["gaps"] = {}
    summary["gaps"]["source"] = gap(empirical.matrix, fisher.matrix)
    summary["source_gap"] = summary["gaps"]["source"]["rel_to_b"]
    print(f"\nsource gap  ||E_hat - F||_F / ||F||_F = {summary['source_gap']:.4f}", flush=True)
    checkpoint_summary()

    # HF1: the same reference on probes the run never trained on.
    val_fisher, _ = build(model, x_val, y_val, "type2", val.digest[:12], "F_val")
    summary["gaps"]["train_val"] = gap(val_fisher.matrix, fisher.matrix)
    summary["train_val_gap"] = summary["gaps"]["train_val"]["rel_to_b"]
    print(f"train/val   ||F_val - F_train||_F / ||F_train||_F = "
          f"{summary['train_val_gap']:.4f}", flush=True)

    # Are val and test interchangeable? Both are out of sample in the same sense (the harness uses
    # neither to select anything), but val is a slice of the training set while the test set is a
    # separate collection. Same n on both sides, so this gap is read against the same floor as the
    # train/val one. Pooling them for a lower HF1 floor is justified only if it is small.
    test_fisher, _ = build(model, x_test, y_test, "type2", test.digest[:12], "F_test")
    summary["gaps"]["val_test"] = gap(test_fisher.matrix, val_fisher.matrix)
    summary["val_test_gap"] = summary["gaps"]["val_test"]["rel_to_b"]
    summary["gaps"]["train_test"] = gap(test_fisher.matrix, fisher.matrix)
    summary["train_test_gap"] = summary["gaps"]["train_test"]["rel_to_b"]
    print(f"val/test    ||F_test - F_val||_F / ||F_val||_F = {summary['val_test_gap']:.4f}"
          f"   (train/test = {summary['train_test_gap']:.4f})", flush=True)
    del val_fisher, test_fisher
    checkpoint_summary()

    # The noise floor of F itself (plan_exp_draft.md §3.4): one two-way split gives its scale.
    half = N_PROBES // 2
    first, _ = build(model, x_train[:half], y_train[:half], "type2", "half1", "F_half1")
    second, _ = build(model, x_train[half:], y_train[half:], "type2", "half2", "F_half2")
    summary["gaps"]["noise_floor"] = gap(first.matrix, second.matrix)
    summary["noise_floor"] = summary["gaps"]["noise_floor"]["rel_to_b"]
    # sigma_N, the error of a single N-probe estimate: F^(1) - F^(2) is the difference of two
    # independent N/2 estimates, so it is sqrt(2) * sigma_{N/2} = 2 * sigma_N. Two *independent*
    # N-probe estimates (train vs val) would differ by sqrt(2) * sigma_N under the null.
    summary["sigma_n"] = summary["noise_floor"] / 2.0
    summary["null_gap_two_independent"] = summary["noise_floor"] / (2.0 ** 0.5)
    print(f"noise floor ||F^(1) - F^(2)||_F / ||F^(2)||_F = {summary['noise_floor']:.4f} "
          f"(at N/2 = {half})  ->  sigma_N = {summary['sigma_n']:.4f}, and two independent "
          f"N-probe estimates differ by {summary['null_gap_two_independent']:.4f} under the null",
          flush=True)
    del first, second
    checkpoint_summary()

    # Spectra and per-block structure. Measured: torch's eigvalsh runs at ~20 GFLOP/s whatever
    # torch.set_num_threads says (19.8 at 1 thread, 18.6 at 8 — LAPACK's dsyevd does not thread
    # here), so a P x P decomposition costs (4/3)P^3 / 2e10 seconds: 1156 s at P = 26 634,
    # measured on the cluster. That, not the builds, is this job's wall clock.
    print("\nspectra (fp64, host):", flush=True)
    spectra = {}
    for name, reference in (("F", fisher), ("E_hat", empirical)):
        started = time.perf_counter()
        eigenvalues = torch.linalg.eigvalsh(reference.matrix)
        spectra[name] = eigenvalues
        top = float(eigenvalues.max())
        rank = int((eigenvalues > 1e-12 * top).sum())
        summary.setdefault("spectrum", {})[name] = {
            "lambda_max": top, "lambda_min": float(eigenvalues.min()),
            "rank_1e-12": rank, "condition_at_1e-12": top / float(eigenvalues[eigenvalues >
                                                                              1e-12 * top].min()),
            "seconds": time.perf_counter() - started,
        }
        print(f"  {name:<6} lambda_max={top:.4e}  lambda_min={float(eigenvalues.min()):+.2e}  "
              f"rank(1e-12)={rank}/{reference.P}  "
              f"{summary['spectrum'][name]['seconds']:.1f}s", flush=True)
        torch.save(spectra, out / "spectrum.pt")
        checkpoint_summary()

    # Cheapest block first: the loop is dominated by the single widest block (25 120^2 here, i.e.
    # 83 % of a full-P decomposition), so this ordering means a time-out costs that one block
    # rather than the whole table.
    print("\nper-block (type-2 F):", flush=True)
    blocks = {}
    names = sorted({n.rsplit(".", 1)[0] for n in fisher.layout.names},
                   key=lambda n: fisher.block(n).shape[0])
    for name in names:
        block = fisher.block(name)
        blocks[name] = {"P": block.shape[0], "trace": float(block.diagonal().sum()),
                        "trace_share": float(block.diagonal().sum() / fisher.trace())}
        if BLOCK_SPECTRA:
            eigenvalues = torch.linalg.eigvalsh(block)
            rank = int((eigenvalues > 1e-12 * eigenvalues.max()).sum())
            blocks[name].update(rank_1e_12=rank, deficiency=block.shape[0] - rank)
        summary["blocks"] = blocks
        checkpoint_summary()
        print(f"  {name:<14} P={block.shape[0]:<6} trace_share={blocks[name]['trace_share']:.3f}  "
              f"rank={blocks[name].get('rank_1e_12', '-')}/{block.shape[0]}  "
              f"deficiency={blocks[name].get('deficiency', '-')}", flush=True)
    summary["metadata"] = fisher.metadata()
    checkpoint_summary()
    print(f"\nwrote {out}/reference_summary.json and spectrum.pt", flush=True)


if __name__ == "__main__":
    main()
