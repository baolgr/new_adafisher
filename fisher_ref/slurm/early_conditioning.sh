#!/bin/bash
# Why do kfac/cnn_gn_cifar and tkfac/cct_2_3x2_cifar fail Step 14's SGD-equivalence check while
# resnet20_cifar never fails outright on any of its five modes? Snapshots the APPLIED (damped)
# Kronecker-factor dyn_range, not just the raw curvature, in the first 400 steps of a fresh run --
# the window Step 13 showed the raw spectrum is still anisotropic in, but never checked on the
# actually-applied matrix, and never covered cct_2_3x2_cifar at all.
# Runs fisher_ref/experiments/early_conditioning.py.
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks,
# and this is an analysis job.
#
# Submit from the repository root:  sbatch fisher_ref/slurm/early_conditioning.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=early_conditioning
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# --time: 3 networks x 5 modes x 400 steps of real training + 5 spectrum snapshots each (CPU
# eigendecomposition, small matrices); well under 15 min based on early_curvature.py's own
# measured per-model timings for these networks.

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

python fisher_ref/experiments/early_conditioning.py
