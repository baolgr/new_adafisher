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

    train = probes.build_probe_set(bench, split="train", n=N_PROBES, seed=SEED,
                                   data_root=DATA_ROOT)
    val = probes.build_probe_set(bench, split="val", n=N_PROBES, seed=SEED, data_root=DATA_ROOT)
    print(f"probes: {len(train)} train / {len(val)} val, digests {train.digest[:12]} / "
          f"{val.digest[:12]}", flush=True)
    x_train, y_train = train.as_model_batch(bench)
    x_val, y_val = val.as_model_batch(bench)

    model = loaded.model
    print("\nbuilds:", flush=True)
    fisher, t_fisher = build(model, x_train, y_train, "type2", train.digest[:12])
    empirical, t_empirical = build(model, x_train, y_train, "empirical", train.digest[:12])

    summary = {
        "model": MODEL, "arm": ARM, "seed": runs[0].seed, "fraction": FRACTION,
        "n_probes": N_PROBES, "P": fisher.P, "n_rows_type2": fisher.n_rows,
        "batch_size": BATCH, "device": DEVICE, "modules": MODULES,
        "seconds": {"type2": t_fisher, "empirical": t_empirical},
        "peak_device_gb": peak_device_gb(),
        "probe_digest": {"train": train.digest, "val": val.digest},
        "trace": {"F": float(fisher.trace()), "E_hat": float(empirical.trace())},
        "fro": {"F": float(torch.linalg.matrix_norm(fisher.matrix)),
                "E_hat": float(torch.linalg.matrix_norm(empirical.matrix))},
        "checkpoint": loaded.metadata(),
    }

    # Q1: the source error, with no structure involved.
    summary["source_gap"] = relative(empirical.matrix, fisher.matrix)
    print(f"\nsource gap  ||E_hat - F||_F / ||F||_F = {summary['source_gap']:.4f}", flush=True)

    # HF1: the same reference on probes the run never trained on.
    val_fisher, _ = build(model, x_val, y_val, "type2", val.digest[:12])
    summary["train_val_gap"] = relative(val_fisher.matrix, fisher.matrix)
    print(f"train/val   ||F_val - F_train||_F / ||F_train||_F = "
          f"{summary['train_val_gap']:.4f}", flush=True)
    del val_fisher

    # The noise floor of F itself (plan_exp_draft.md §3.4): one two-way split gives its scale.
    half = N_PROBES // 2
    first, _ = build(model, x_train[:half], y_train[:half], "type2", "half1")
    second, _ = build(model, x_train[half:], y_train[half:], "type2", "half2")
    summary["noise_floor"] = relative(first.matrix, second.matrix)
    print(f"noise floor ||F^(1) - F^(2)||_F / ||F^(2)||_F = {summary['noise_floor']:.4f} "
          f"(at N/2 = {half}); any later difference below it is not interpretable", flush=True)
    del first, second

    # Spectra and per-block structure.
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

    print("\nper-block (type-2 F):", flush=True)
    blocks = {}
    for name in sorted({n.rsplit(".", 1)[0] for n in fisher.layout.names}):
        block = fisher.block(name)
        eigenvalues = torch.linalg.eigvalsh(block)
        rank = int((eigenvalues > 1e-12 * eigenvalues.max()).sum())
        blocks[name] = {"P": block.shape[0], "trace": float(block.diagonal().sum()),
                        "trace_share": float(block.diagonal().sum() / fisher.trace()),
                        "rank_1e-12": rank, "deficiency": block.shape[0] - rank}
        print(f"  {name:<14} P={block.shape[0]:<6} trace_share={blocks[name]['trace_share']:.3f}  "
              f"rank={rank}/{block.shape[0]}  deficiency={blocks[name]['deficiency']}", flush=True)
    summary["blocks"] = blocks
    summary["metadata"] = fisher.metadata()

    out = OUT_DIR / MODEL / ARM / f"seed{runs[0].seed}" / str(FRACTION)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reference_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    torch.save({name: value for name, value in spectra.items()}, out / "spectrum.pt")
    print(f"\nwrote {out}/reference_summary.json and spectrum.pt", flush=True)


if __name__ == "__main__":
    main()
