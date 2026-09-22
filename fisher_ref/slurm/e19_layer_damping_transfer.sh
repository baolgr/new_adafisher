#!/bin/bash
# E19 of docs/reports/plan_lambda_dominance.md: E15's per-layer safety constant (S1-b) on
# cct_2_3x2_cifar, resnet20_cifar and resnet50_cifar, paired by seed with E16's stored add and clip
# cells. Pre-registered there before this script existed; read that section first.
#
# ONE SHARD of one (network, seed, mode): E19_SHARD takes every E19_NSHARDS-th cell of that job's
# ordered list and writes fisher_ref/outputs/e19_layer_damping_<model>_s<seed>_<mode>.shard<i>of<n>.json
# after every cell. fisher_ref/slurm/e19_merge.sh then merges the shard files.
#
# Do not sbatch this by hand: fisher_ref/slurm/e19_submit.sh submits the shards and the merge of one
# (network, seed, mode) together, with the right --time and shard count, and refuses a modified
# tree. All of E19 must come from ONE clean checkout of ONE commit: e19_decisions.py refuses files
# from a dirty tree or from different commits.
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e19_layer_damping
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers.
#   --gpus / --cpus-per-task / --mem / OMP_NUM_THREADS: exactly E16's line
#     (fisher_ref/slurm/e16_floor_clip.sh), so every E19 cell runs on the hardware of the E16 cells
#     it is paired with, one process per 1g slice.
#   --time, set per network by e19_submit.sh: 1:00 / 1:30 / 2:30 for cct / resnet20 / resnet50. The
#     largest shard holds 10 / 9 / 2 cells: about 35 / 55 / 85 minutes, from the add cells' times
#     measured on 1g slices in E16 (190-202 / 321-335 / 2385-2425 s) plus E15's measured overhead of
#     a relative damping (+3 % for s1b, +4-6 % for netadapt). About 1.7x on top.

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

# Production settings, forced: --export=ALL would otherwise carry a smoke value from the submitting
# shell into the job. The driver refuses any of these unless E19_SMOKE=1, and this job never sets it.
unset E19_SMOKE E19_ARMS E19_BATCH E19_GRID_LIMIT E19_TRAIN_SUBSET E19_LOG_EVERY E19_MERGE
export WARMUP_SGD_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export WARMUP_SGD_EPOCHS=15
export WARMUP_SGD_DEVICE=auto
export WARMUP_SGD_NUM_WORKERS=4   # as in E16: the worker count changes the trajectory
export E19_MODEL="${E19_MODEL:?set E19_MODEL on the sbatch line}"
export E19_SEED="${E19_SEED:?set E19_SEED on the sbatch line}"
export E19_MODE="${E19_MODE:?set E19_MODE on the sbatch line}"
export E19_SHARD="${E19_SHARD:?set E19_SHARD on the sbatch line}"
export E19_NSHARDS="${E19_NSHARDS:?set E19_NSHARDS on the sbatch line}"
export E19_OUT="$SLURM_SUBMIT_DIR/fisher_ref/outputs/e19_layer_damping_${E19_MODEL}_s${E19_SEED}_${E19_MODE}.json"
mkdir -p "$SLURM_SUBMIT_DIR/fisher_ref/outputs"

python -u fisher_ref/experiments/e19_layer_damping_transfer.py
