#!/bin/bash
# E2 of docs/reports/plan_lambda_dominance.md: sweep Lambda while holding lr/Lambda -- the cap on
# the step size -- constant, against the same sweep at a fixed lr (which is what Step 15 of
# docs/reports/validation_noise_investigation.md actually ran, and why its collapse could not be
# attributed).
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks.
# Same convention as warmup_sgd_baseline.sh, whose training loop this reuses.
#
# Submit from the repository root:  sbatch fisher_ref/slurm/e2_lambda_at_fixed_gain.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e2_lambda_at_fixed_gain
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers.
#   grid: 3 models x 5 modes x (1 reference + 5 lambdas x 2 protocols) = 165 runs of 15 epochs.
#     The grid reaches 1e-12 because E1 measured that a Kronecker mode needs Lambda one to five
#     orders of magnitude below default before any direction's curvature reaches it, and E2's own
#     smoke confirmed nothing moves at 1e-6. At a fixed cap there is no reason not to go there.
#     Measured per-run cost on this hardware, from warmup_sgd_baseline.json: 19 s for
#     mlp_ln_mnist/kfac. resnet20_cifar is the expensive one (39 hooked layers, 270k parameters);
#     budget a few minutes there. 08:00:00 is generous on purpose -- the output JSON is rewritten
#     after every completed mode, so a wall-clock kill loses at most one mode, never the grid.
#   --gpus: a dedicated slice. These are timing-insensitive runs (fixed epoch count, not a
#     wall-clock budget), but the shared backfill queue's drift once cost this project a full
#     rerun (job 21122754) and the slice costs nothing extra here.
#   --mem: 24G. Each run holds one model plus its optimizer state; ekfac/tekfac additionally cache
#     every layer's input batch simultaneously (plan_lot8.md section 0.4), which is the memory
#     high-water mark and is ~100 MB at these model sizes.
#
# Expect crashes, and do not treat them as job failures: kfac and tkfac invert their damped factors
# directly and hit "linalg.inv: matrix is singular" at a Lambda as large as 3e-5 (Step 15). Every
# run is wrapped, so a crash becomes an X in the summary table and the grid continues.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023 gcc/13.3 cuda/12.6 python/3.11.5

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements-cluster.txt

export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export WARMUP_SGD_DATA_ROOT="${WARMUP_SGD_DATA_ROOT:-$SLURM_SUBMIT_DIR/dataset}"
export WARMUP_SGD_EPOCHS="${WARMUP_SGD_EPOCHS:-15}"
export WARMUP_SGD_DEVICE="${WARMUP_SGD_DEVICE:-auto}"
# NOTE: sbatch --export truncates any value containing a comma at the first comma (measured, Step
# 14 -- it silently dropped five of six models from a run). Set these here, not on the sbatch line.
export E2_MODELS="${E2_MODELS:-mlp_ln_mnist,cnn_gn_cifar,resnet20_cifar}"
export E2_MODES="${E2_MODES:-diag,kfac,ekfac,tkfac,tekfac}"
export E2_LAMBDAS="${E2_LAMBDAS:-1e-4,1e-6,1e-8,1e-10,1e-12}"
export E2_PROTOCOLS="${E2_PROTOCOLS:-fixed_lr,fixed_gain}"

python fisher_ref/experiments/e2_lambda_at_fixed_gain.py
