#!/bin/bash
# E4 of docs/reports/plan_lambda_dominance.md: is it the curvature that does not help, or the way
# it is being estimated?
#
# E2 lowered the safety constant across nine orders of magnitude with the step size held still and
# found the five modes do not get better. But that only fixes the *size* of the curvature relative
# to the constant; it leaves the *averaging* as broken as it was (92% one batch of examples, plus a
# residue of the identity it was started from). This job sweeps the same values of the safety
# constant twice: once with the estimator that ships, once with a real average started from a real
# observation (gamma = 0.8, which is the paper's own Eq. (3), plus ema_seed_first). At batch 32,
# because E3 measured that is where a corrected estimator has a chance of mattering at all.
#
# One job per network, because resnet20_cifar costs eight times what mlp_ln_mnist does and a single
# job would have to be sized for the worst one. Submit from the repository root:
#
#   sbatch --job-name=e4_mlp    --time=02:30:00 --export=ALL,E4_MODELS=mlp_ln_mnist   fisher_ref/slurm/e4_fixed_average.sh
#   sbatch --job-name=e4_cnn    --time=03:30:00 --export=ALL,E4_MODELS=cnn_gn_cifar   fisher_ref/slurm/e4_fixed_average.sh
#   sbatch --job-name=e4_resnet --time=08:00:00 --export=ALL,E4_MODELS=resnet20_cifar fisher_ref/slurm/e4_fixed_average.sh
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e4_fixed_average
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers.
#   grid: 5 modes x 2 estimators x (1 reference + 4 values of the constant) = 50 runs of 15 epochs
#     per network. Measured at batch 128 in E2 (job 21283681): 964 s for mlp_ln_mnist's 55 runs,
#     1495 s for cnn_gn_cifar's, 5041 s for resnet20_cifar's. Batch 32 is four times the steps at a
#     quarter the work each, which came out at roughly 2.7x the wall clock in a local smoke, hence
#     the three --time values above. The output JSON is rewritten after every (mode, estimator), so
#     a wall-clock kill costs at most one pair.
#   --gpus: a dedicated slice. These runs are not timed, but the shared backfill queue's drift once
#     cost this project a full rerun (job 21122754) and the slice costs nothing extra.
#   --mem: 24G. One model plus its optimizer state per run; ekfac and tekfac additionally hold every
#     layer's input batch at once, which is the high-water mark and is small at these model sizes.
#
# Expect crashes and do not read them as job failures: kfac and tkfac invert their factors directly
# and hit "linalg.inv: matrix is singular" at a small enough constant. Every run is wrapped, so a
# crash becomes an X in the table and the grid carries on.

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
# NOTE: sbatch --export truncates any value containing a comma at the first comma (measured: it
# silently dropped five of six models from an earlier run). Anything comma-separated is set here,
# not on the sbatch line. E4_MODELS is a single name, so it is safe to pass there.
export E4_MODELS="${E4_MODELS:-mlp_ln_mnist}"
export E4_MODES="${E4_MODES:-diag,kfac,ekfac,tkfac,tekfac}"
export E4_LAMBDAS="${E4_LAMBDAS:-1e-4,1e-6,1e-8,1e-10}"
export E4_ESTIMATORS="${E4_ESTIMATORS:-shipped,corrected}"
export E4_BATCH="${E4_BATCH:-32}"
export E4_GAMMA="${E4_GAMMA:-0.8}"
# Scalar on purpose. A seed axis is five submissions with a different value and a different
# output file, which needs no code and gives each seed its own reference arm -- and those
# references are what measure the seed-to-seed floor on the network being swept, instead of
# borrowing a floor measured on a different one.
export E4_SEED="${E4_SEED:-0}"
# 1 = hold the parameters the optimizer does not precondition at their initial values in
# EVERY arm, so the sweep does not silently freeze them as lambda falls. See the E5 control.
export E4_FREEZE_UNHOOKED="${E4_FREEZE_UNHOOKED:-0}"
# 1 = ekfac/tekfac measure their rescaling in the basis precondition uses. Matters only
# once Lambda is far below 1e-3; see CLAUDE.md section 3.
export E4_EIG_BEFORE_RESCALE="${E4_EIG_BEFORE_RESCALE:-0}"
export E4_OUT="${E4_OUT:-$SLURM_SUBMIT_DIR/fisher_ref/outputs/e4_fixed_average_${E4_MODELS}_frozen${E4_FREEZE_UNHOOKED}_eig${E4_EIG_BEFORE_RESCALE}.json}"

python fisher_ref/experiments/e4_fixed_average.py
