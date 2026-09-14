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
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:50:00
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
#   --cpus-per-task=16     The long pole is host-side and threaded: two 26 634^2 fp64 eigvalsh calls,
#                          ~2.5e13 flop each. 16 is Rorqual's documented per-GPU maximum.
#   --time=00:50:00        ESTIMATE, not a measurement — this job has never run. Basis: ~40 s setup
#                          (measured, benchmarks/slurm/README.md), ~40 s for the five GPU builds
#                          (1.8e14 fp64 flop total on a 1/7 slice), 8-17 min for the two eigvalsh.
#                          Replace with the measured Elapsed + headroom after the first COMPLETED
#                          run, as benchmarks/slurm/README.md requires of every --time.
#
# Prerequisites, both verified present on rorqual:/home/blgr/new_adafisher —
#   dataset/MNIST                                    (compute nodes have no internet)
#   benchmarks/outputs/mlp_ln_mnist/diag/ckpt_0.5.pt (campaign 1, post-denominator-fix)
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# N = 4000 type-2 probes: N(C-1) = 36 000 >= P = 26 634, so F is not rank-limited by the probe
# count (plan_exp_draft.md §2.2). The script's knobs are environment variables by design — the
# fisher_ref/experiments/ convention is one question per script, no CLI.
export A1_MODEL=mlp_ln_mnist
export A1_ARM=diag
export A1_FRACTION=0.5
export A1_PROBES=4000
export A1_BATCH=512
export A1_SEED=0
export A1_DEVICE=cuda
export A1_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export A1_OUTPUTS_ROOT="$SLURM_SUBMIT_DIR/benchmarks/outputs"
export A1_OUT_DIR="$SLURM_SUBMIT_DIR/fisher_ref/outputs"

python -u fisher_ref/experiments/dense_reference_a1.py
