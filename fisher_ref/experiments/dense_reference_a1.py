"""The A1 dense reference: `F`, `E_hat` and the per-layer blocks of `mlp_ln_mnist` at one
checkpoint, in fp64 (`docs/reports/plan_exp_lot1.md` §5, lot 1 of `plan_exp_draft.md` §9).

Run it:

    PYTHONPATH=src:. python -u fisher_ref/experiments/dense_reference_a1.py

or through `fisher_ref/slurm/dense_reference_a1.sh`, which is what produced the numbers below.
Every knob is an environment variable with a default, so the sbatch script sets them without this
file growing a CLI it does not want (the `experiments/` convention: constants, one question, the
answer pasted back into this docstring).

What it produces, at `N = 4000` type-2 probes (`N(C-1) = 36 000 >= P = 26 634`, so `F` is not
rank-limited by the probe count — `plan_exp_draft.md` §2.2):

* `F` (type-2, the true Fisher) and `E_hat` (empirical), dense, fp64, `26 634^2`;
* their spectra, and the per-block trace shares and ranks;
* the **source gap** `||E_hat - F||_F / ||F||_F` — the campaign's Q1, with no structure involved;
* the **train/val gap** `||F_val - F_train||_F / ||F_train||_F` — HF1;
* the **noise floor** `||F^(1) - F^(2)||_F / ||F||_F` from a two-way split of the probes
  (`plan_exp_draft.md` §3.4): any later difference below it is not interpretable.

Only summaries and the two spectra are written to disk. `F` itself is 5.68 GB and recomputable in
minutes; `plan_exp_draft.md` §12 says not to write it.

Memory, which is the whole reason this has its own job. The accumulator is the only `P x P` tensor
on the device (`addmm_`, and a block-wise in-place symmetrisation — `reference/dense.py`), so the
build peaks at `~1.05 x P^2 = 6.0 GB` and fits the same `h100_1g.10gb` slice the training jobs use.
Everything with more than one `P x P` live at once — the two references side by side, and
`eigvalsh` — happens on the **host**, where `--mem` is cheap. Measured multiplier on CPU at
`P = 12 030`: `1.27 x P^2` for the whole build, against `+1.16 x P^2` for the single expression
`0.5 * (M + M.T)` this code no longer uses.

Result: ------------------------------- not yet run -------------------------------
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
    difference_sq = 0.0
    reference_sq = 0.0
    for start in range(0, a.shape[0], block):
        stop = min(start + block, a.shape[0])
        rows_b = b[start:stop]
        difference_sq += float((a[start:stop] - rows_b).pow(2).sum())
        reference_sq += float(rows_b.pow(2).sum())
    return (difference_sq / reference_sq) ** 0.5


def build(model, inputs, targets, source: str, tag: str) -> "tuple":
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
    print(f"  {source:<10} {tag:<12} P={reference.P}  rows={reference.n_rows:<7} "
          f"{elapsed:7.1f}s  {gigabytes(matrix):5.2f} GB  peak device {peak_device_gb():5.2f} GB",
          flush=True)
    return reference, elapsed


def main() -> None:
    state = conventions.configure()
    torch.manual_seed(SEED)
    print(f"device={DEVICE}  TF32 off={state.tf32_is_off()}  "
          f"metrics_version={conventions.METRICS_VERSION}", flush=True)

    bench = discover_benchmarks()[MODEL]
    runs = discover_runs(OUTPUTS_ROOT, model=MODEL, arm=ARM)
    if not runs:
        raise SystemExit(f"no run found for {MODEL}/{ARM} under {OUTPUTS_ROOT}")
    loaded = load_theta(runs[0], FRACTION, device="cpu")
    print(f"theta: {MODEL}/{ARM} ckpt {FRACTION}  step={loaded.step} epoch={loaded.epoch} "
          f"progress={loaded.progress:.3f}  seed={runs[0].seed}", flush=True)

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

    out = OUT_DIR / MODEL / ARM / f"seed{runs[0].seed}" / str(FRACTION)
    out.mkdir(parents=True, exist_ok=True)

    model = loaded.model
    print("\nbuilds:", flush=True)
    fisher, t_fisher = build(model, x_train, y_train, "type2", train.digest[:12])
    empirical, t_empirical = build(model, x_train, y_train, "empirical", train.digest[:12])

    summary = {
        "model": MODEL, "arm": ARM, "seed": runs[0].seed, "fraction": FRACTION,
        "n_probes": N_PROBES, "n_val_probes": len(val), "P": fisher.P,
        "n_rows_type2": fisher.n_rows,
        "batch_size": BATCH, "device": DEVICE, "modules": MODULES,
        "seconds": {"type2": t_fisher, "empirical": t_empirical},
        "peak_device_gb": peak_device_gb(),
        "probe_digest": {"train": train.digest, "val": val.digest},
        "trace": {"F": float(fisher.trace()), "E_hat": float(empirical.trace())},
        "fro": {"F": float(torch.linalg.matrix_norm(fisher.matrix)),
                "E_hat": float(torch.linalg.matrix_norm(empirical.matrix))},
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
    summary["source_gap"] = relative(empirical.matrix, fisher.matrix)
    print(f"\nsource gap  ||E_hat - F||_F / ||F||_F = {summary['source_gap']:.4f}", flush=True)
    checkpoint_summary()

    # HF1: the same reference on probes the run never trained on.
    val_fisher, _ = build(model, x_val, y_val, "type2", val.digest[:12])
    summary["train_val_gap"] = relative(val_fisher.matrix, fisher.matrix)
    print(f"train/val   ||F_val - F_train||_F / ||F_train||_F = "
          f"{summary['train_val_gap']:.4f}", flush=True)

    # Are val and test interchangeable? Both are out of sample in the same sense (the harness uses
    # neither to select anything), but val is a slice of the training set while the test set is a
    # separate collection. Same n on both sides, so this gap is read against the same floor as the
    # train/val one. Pooling them for a lower HF1 floor is justified only if it is small.
    test_fisher, _ = build(model, x_test, y_test, "type2", test.digest[:12])
    summary["val_test_gap"] = relative(test_fisher.matrix, val_fisher.matrix)
    summary["train_test_gap"] = relative(test_fisher.matrix, fisher.matrix)
    print(f"val/test    ||F_test - F_val||_F / ||F_val||_F = {summary['val_test_gap']:.4f}"
          f"   (train/test = {summary['train_test_gap']:.4f})", flush=True)
    del val_fisher, test_fisher
    checkpoint_summary()

    # The noise floor of F itself (plan_exp_draft.md §3.4): one two-way split gives its scale.
    half = N_PROBES // 2
    first, _ = build(model, x_train[:half], y_train[:half], "type2", "half1")
    second, _ = build(model, x_train[half:], y_train[half:], "type2", "half2")
    summary["noise_floor"] = relative(first.matrix, second.matrix)
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
