#!/bin/bash
# E9 of docs/reports/plan_lambda_dominance.md: does reading the curvature EARLY in training tighten
# the constant `tau`, or not?
#
# E7 located the best safety constant for each network by sweeping it, and expressed it against the
# network's own mean curvature to get a dimensionless `tau`. That still spreads by 10.8x across the
# eight peaks that could be located. Every one of those mean curvatures came from a HALF-TRAJECTORY
# checkpoint, while the sweeps train from scratch -- so the leading candidate for the residual is
# that a mid-training statistic is the wrong thing to set a whole-run constant from. This job
# recomputes `tau` from the curvature measured along the early trajectory instead.
#
# Hand-written, NOT generated. Submit from the repository root:
#   sbatch fisher_ref/slurm/e9_early_curvature_tau.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e9_early_tau
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers.
#   work: 3 networks x 4 modes, each trained from scratch to 8000 steps at batch 32, with the
#     curvature read at 1000, 2000, 4000 and 8000. The sweeps measured cct_2_3x2_cifar, the slowest
#     of the three, at ~220 s for 15 epochs (about 21 000 steps at this batch), so 8000 steps is
#     about 85 s of training per mode plus four eigendecompositions. Budget under an hour; 02:30:00
#     is deliberately generous. The JSON is rewritten after every mode.
#   --gpus: a dedicated slice, for the reason every other analysis job here gives -- the shared
#     backfill queue's throughput drifts under other users' load and once cost a full rerun.
#   --mem: 24G. The pooled list of curvature values is one fp64 array of sum_l (d_in x d_out)
#     entries, 276 k for the largest of these three networks. Nothing here is memory-bound.
#
# The first snapshot is at 1000 steps and not earlier on purpose: every mode starts its running
# average at the identity and that seed fades as 0.08^k per factor update, so below ten updates the
# measurement reports the optimizer's start-up rather than the network. That is not a guess -- it
# was measured, and the driver's docstring carries the numbers.

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
# NOTE: sbatch --export truncates any value containing a comma at the first comma. Anything
# comma-separated is set here, not on the sbatch line.
export E9_MODELS="${E9_MODELS:-cnn_gn_cifar,vit_micro_cifar,cct_2_3x2_cifar}"
export E9_MODES="${E9_MODES:-kfac,ekfac,tkfac,tekfac}"
export E9_SNAPSHOTS="${E9_SNAPSHOTS:-1000,2000,4000,8000}"
export E9_BATCH="${E9_BATCH:-32}"
export E9_SEED="${E9_SEED:-0}"

python fisher_ref/experiments/e9_early_curvature_tau.py
