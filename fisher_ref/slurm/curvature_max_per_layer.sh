#!/bin/bash
# How big does curvature ever get, per layer, on every network already checked by
# lambda_vs_curvature.py -- not just the two (mlp_ln_mnist, cnn_gn_cifar/BatchNorm) that got a
# by-hand maximum check before. Runs fisher_ref/experiments/curvature_max_per_layer.py.
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks,
# and this is an analysis job. Same convention as lambda_vs_curvature.sh.
#
# Submit from the repository root:  sbatch fisher_ref/slurm/curvature_max_per_layer.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=curvature_max_per_layer
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Same numbers as lambda_vs_curvature.sh, which this reuses verbatim (7 runs x 5 modes x 1000
# re-warm steps, eigendecompositions on the CPU): under 15 min of compute, 01:00:00 generous.

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
export REWARM_STEPS="${REWARM_STEPS:-1000}"
export CKPT_FRACTION="${CKPT_FRACTION:-0.5}"

python fisher_ref/experiments/curvature_max_per_layer.py
