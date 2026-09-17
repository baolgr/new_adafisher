#!/bin/bash
# Is AdaFisherMulti/kfac, at the project's default operating point, indistinguishable from plain
# momentum-SGD with a matching warmup step size? Runs
# fisher_ref/experiments/warmup_sgd_baseline.py over mlp_ln_mnist, cnn_gn_cifar, resnet20_cifar:
# kfac at two seeds (the seed-noise floor) plus WarmupMomentumSGD at the first seed (identical init
# and data order, only the optimizer differs).
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks,
# and this is an analysis job. Same convention as lambda_vs_curvature.sh.
#
# Submit from the repository root:  sbatch fisher_ref/slurm/warmup_sgd_baseline.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=warmup_sgd_baseline
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers.
#   --gpus: a dedicated slice, not the shared "cpubackfill" queue where throughput drifts under
#     other users' load (the lesson of job 21122754, lambda_vs_curvature.sh's own note).
#   --time: 3 models x 3 runs (kfac seed0, kfac seed1, warmup_sgd seed0) x 15 epochs x ~350
#     steps/epoch. All three models are small (21k-270k parameters); generous headroom over
#     environment setup (~1-2 min) plus training.
#   --epochs=15: >=3 full TCov=100 warmup cycles ((1-gammas[0])^3 = 5.1e-4, well under
#     Lambda=1e-3) within the first epoch, then ~10 more epochs to see the post-warmup trajectory.

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
export WARMUP_SGD_MODELS="${WARMUP_SGD_MODELS:-mlp_ln_mnist,cnn_gn_cifar,resnet20_cifar}"
export WARMUP_SGD_EPOCHS="${WARMUP_SGD_EPOCHS:-15}"
export WARMUP_SGD_SEEDS="${WARMUP_SGD_SEEDS:-0,1}"
export WARMUP_SGD_DATA_ROOT="${WARMUP_SGD_DATA_ROOT:-$SLURM_SUBMIT_DIR/dataset}"
export WARMUP_SGD_NUM_WORKERS="${WARMUP_SGD_NUM_WORKERS:-8}"
export WARMUP_SGD_ALLOW_DOWNLOAD="0"

python fisher_ref/experiments/warmup_sgd_baseline.py
