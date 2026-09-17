#!/bin/bash
# Separates warmup_sgd_baseline.py's two Step-11 corrections on cnn_gn_cifar, the one network
# where fixing both together made the mismatch from real kfac WORSE, not better. Runs the 2x2:
# {old, corrected schedule} x {ignore, respect the hooked/unhooked boundary}.
# Runs fisher_ref/experiments/cnn_gn_ablation.py.
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks,
# and this is an analysis job. Same convention as warmup_sgd_baseline.sh, which this reads the
# real-kfac baseline and seed-noise floor back from (must already exist at
# fisher_ref/outputs/warmup_sgd_baseline.json -- run that job first if it does not).
#
# Submit from the repository root:  sbatch fisher_ref/slurm/cnn_gn_ablation.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=cnn_gn_ablation
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# --time: 4 short cnn_gn_cifar runs (15 epochs, ~25s each per warmup_sgd_baseline.sh's own
# measured timing) plus environment setup; generous headroom.

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
export WARMUP_SGD_EPOCHS="${WARMUP_SGD_EPOCHS:-15}"
export WARMUP_SGD_SEEDS="${WARMUP_SGD_SEEDS:-0,1}"
export WARMUP_SGD_DATA_ROOT="${WARMUP_SGD_DATA_ROOT:-$SLURM_SUBMIT_DIR/dataset}"
export WARMUP_SGD_NUM_WORKERS="${WARMUP_SGD_NUM_WORKERS:-8}"
export WARMUP_SGD_ALLOW_DOWNLOAD="0"

python fisher_ref/experiments/cnn_gn_ablation.py
