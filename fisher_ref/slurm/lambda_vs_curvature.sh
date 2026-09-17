#!/bin/bash
# Is Lambda negligible against the real curvature scale? Per layer, per model, per mode.
# Runs fisher_ref/experiments/lambda_vs_curvature.py over the seven runs whose held-out noise was
# measured in docs/reports/validation_noise_investigation.md.
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks,
# and this is an analysis job. Same convention as dense_reference_a1.sh.
#
# Submit from the repository root:  sbatch fisher_ref/slurm/lambda_vs_curvature.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=lambda_vs_curvature
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers.
#   --gpus: a dedicated slice. Not for the arithmetic (the networks are 21k-284k parameters and the
#     eigendecompositions run on the CPU on purpose, see the script's docstring) but because a job
#     without one lands on the shared "cpubackfill" queue, where measured throughput drifts under
#     other users' load -- that is what invalidated job 21122754 and cost a full rerun.
#   --time: 7 runs x 5 modes x 1000 re-warm steps. The largest network here is cct_2_3x2_cifar
#     (284k parameters); at the ~10-30 ms/step these models cost on an H100 slice that is under
#     15 min of compute, plus ~2 min of environment setup. 01:00:00 is deliberately generous.
#   --mem: the spectra are formed as outer products of two factor spectra; the widest is
#     mnist_autoencoder's 1000 x 785 = 785k entries in fp64. Nothing here is memory-bound.
#
# The re-warm length is NOT a free choice: the checkpoints hold theta only, every mode seeds its
# EMA with the identity, and that seed acts as a spurious extra damping of 0.08^k on top of Lambda.
# rewarm_fidelity.py measured that k=3 leaves the applied preconditioner 12-87% wrong, and that
# k=10 lands within 1.3-4.5x of the estimator's own batch-draw noise floor. Hence 1000 = 10 * TCov.

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

python fisher_ref/experiments/lambda_vs_curvature.py
