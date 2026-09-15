#!/bin/bash
# A1 (mlp_ln_mnist) dense reference: F, E_hat and the per-layer blocks in fp64 at one checkpoint.
# Lot 1 phase 5 of docs/reports/plan_exp_lot1.md §5; the campaign plan is plan_exp_draft.md §9.
# Submit from the repository root:  sbatch fisher_ref/slurm/dense_reference_a1.sh
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates the training benchmarks,
# and benchmarks/ is read-only for this campaign (plan_exp_draft.md §8). Hence a separate directory.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=fisher_ref_a1
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# ---------------------------------------------------------------------------------------------
# Why these numbers. plan_exp_draft.md §2.2 and §12 warn that "an analysis job that inherits the
# training job's --gpus line cannot allocate F at all": at P = 26 634 a dense fp64 F is 5.68 GB and
# the plan budgets P_max = sqrt(B/24), i.e. ~17 GB with the eigendecomposition. That constant 24 is
# a property of the *implementation*, not of the mathematics — it counts three P x P buffers. This
# job needs one, so the 10 GB slice is enough:
#
#   --gpus=h100_1g.10gb:1  The accumulator is the only P x P tensor on the device. It is filled with
#                          addmm_ (no product temporary) and symmetrised block-wise in place
#                          (reference/dense.py::symmetrize_), so the build peaks at ~1.05 x P^2 =
#                          6.0 GB of the 10 GB. The naive spellings would have needed ~17 GB —
#                          measured at P = 12 030: 1.27 x P^2 for this builder, against +1.16 x P^2
#                          for the single expression 0.5 * (M + M.T) alone. The same slice every
#                          training job of campaign 1 used, so there is no scheduling risk, and it
#                          is the cheapest shape the def-msh-ab_gpu allocation bills.
#   --mem=64G              Host side, where everything with two P x P live at once happens: up to
#                          four references during the noise-floor split (4 x 5.68 = 22.7 GB) and the
#                          two eigvalsh calls (F + a copy). Peak expected ~25 GB; 64G is headroom
#                          for a first run, to be lowered once the log reports the real MaxRSS.
#   --cpus-per-task=4      NOT 16, and this is measured, not tidied: torch's eigvalsh does not
#                          thread. 19.8 GFLOP/s at 1 thread, 18.6 at 8 (LAPACK dsyevd, laptop), and
#                          job 21077038 ran at 21.6 GFLOP/s on 16 cores — the same regime. Extra
#                          cores buy nothing here and only make the job harder to schedule; 4 is
#                          for the data loading and the GPU builds.
#   --time=01:30:00        MEASURED from job 21077038 (CANCELLED at the 00:50:00 limit, inside the
#                          per-block loop, after producing everything else):
#                            setup + probes                       ~120 s
#                            5 GPU builds                           34 s  (11.2 + 1.2 + 10.9 + 2x5.5)
#                            eigvalsh(F), eigvalsh(E_hat)         2311 s  (1155.6 + 1155.7)
#                            eigvalsh(features.0), 25 120^2        ~970 s  ((25120/26634)^3 x 1156)
#                            the other four blocks                  ~0 s
#                                                            total ~57 min
#                          Set A1_BLOCK_SPECTRA=0 to drop the last 970 s and keep the trace shares.
#
# Prerequisites on rorqual:/home/blgr/new_adafisher —
#   dataset/MNIST   (compute nodes have no internet)
#   a post-denominator-fix mlp_ln_mnist/diag checkpoint at fraction 0.5 (campaign 1), in EITHER
#   layout: benchmarks/outputs/mnist/mlp_ln_mnist/diag/ or the pre-migration
#   benchmarks/outputs/mlp_ln_mnist/diag/. Stated as a requirement rather than a path because the
#   driver locates it through fisher_ref.checkpoints.discover_runs, which reads both layouts —
#   so this line does not have to be resynchronised the day the cluster's outputs/ tree is
#   grouped by dataset, and cannot silently claim to have "verified" a path on the wrong side of
#   that migration. A missing checkpoint is a clear SystemExit from the driver, not a wrong result.
# ---------------------------------------------------------------------------------------------

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023 gcc/13.3 cuda/12.6 python/3.11.5

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements-cluster.txt

export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# N = 4000 type-2 probes: N(C-1) = 36 000 >= P = 26 634, so F is not rank-limited by the probe
# count (plan_exp_draft.md §2.2). The script's knobs are environment variables by design — the
# fisher_ref/experiments/ convention is one question per script, no CLI.
# Every one is "${VAR:-default}" so `sbatch --export=ALL,A1_PROBES=55000 ...` actually takes
# effect. A plain `export A1_PROBES=4000` would silently overwrite it — the same trap as the
# hand-copied --wct-budget that produced campaign 1's mislabelled checkpoints.
export A1_MODEL="${A1_MODEL:-mlp_ln_mnist}"
export A1_ARM="${A1_ARM:-diag}"
export A1_FRACTION="${A1_FRACTION:-0.5}"
export A1_PROBES="${A1_PROBES:-4000}"
export A1_BATCH="${A1_BATCH:-512}"
export A1_SEED="${A1_SEED:-0}"
export A1_DEVICE="${A1_DEVICE:-cuda}"
export A1_BLOCK_SPECTRA="${A1_BLOCK_SPECTRA:-1}"
export A1_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export A1_OUTPUTS_ROOT="$SLURM_SUBMIT_DIR/benchmarks/outputs"
export A1_OUT_DIR="$SLURM_SUBMIT_DIR/fisher_ref/outputs"

python -u fisher_ref/experiments/dense_reference_a1.py
