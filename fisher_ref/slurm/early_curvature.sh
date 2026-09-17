#!/bin/bash
# Is curvature negligible next to Lambda in the first few TCov cycles of a FRESH kfac run?
# Every earlier curvature-vs-Lambda measurement (lambda_vs_curvature.py, Steps 7/8/10) was taken
# mid-training, at ckpt_0.5. This checks steps 0-400 of a network at its random initialization,
# on the same three networks and seed as warmup_sgd_baseline.py (Steps 9/11/12) -- the window
# where the schedule correction helped two networks and hurt a third (cnn_gn_cifar).
# Runs fisher_ref/experiments/early_curvature.py.
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks,
# and this is an analysis job. Same convention as lambda_vs_curvature.sh / cnn_gn_ablation.sh.
#
# Submit from the repository root:  sbatch fisher_ref/slurm/early_curvature.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=early_curvature
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# --time: 3 networks x 400 steps of real training + 5 eigendecomposition snapshots each (CPU,
# small matrices); well under 5 min based on warmup_sgd_baseline.sh's own measured per-epoch
# timings for these three models.

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
export DATA_ROOT="${DATA_ROOT:-$SLURM_SUBMIT_DIR/dataset}"

python fisher_ref/experiments/early_curvature.py
