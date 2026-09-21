#!/bin/bash
# E16 of docs/reports/plan_lambda_dominance.md: a floor (family A) or a clip (family B) instead of
# the added safety constant, with E16's own add baseline. Pre-registered there, and specified in
# docs/reports/plan_floor_clip.md, before this script existed; read both first.
#
# ONE SHARD of one (network, seed, mode): E16_SHARD in {0, 1, 2} takes every third cell of that
# job's ordered list and writes fisher_ref/outputs/e16_floor_clip_<model>_s<seed>_<mode>.shard<i>of3.json
# after every cell. fisher_ref/slurm/e16_floor_clip_merge.sh then merges the three shard files and
# runs the seed-0 checks. Every cell but the repro diagnostic runs with norm_exact_rescaling
# (plan_floor_clip.md §11).
#
# Do not sbatch this by hand: fisher_ref/slurm/e16_submit.sh submits the three shards and the merge
# of one (network, seed, mode) together, with the right --time, and refuses a modified tree. All 30
# (network, seed, mode) must be submitted from ONE clean checkout of ONE commit: e16_decisions.py
# refuses files from a dirty tree or from different commits.
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e16_floor_clip
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers.
#   --gpus / --cpus-per-task / --mem / OMP_NUM_THREADS: exactly the E-series line
#     (fisher_ref/slurm/e4_fixed_average.sh, which ran E13/E14), so every E16 cell runs on the
#     hardware E14's stored cells ran on, and the per-cell times below are measured ones.
#     One process per 1g slice: no two runs share a GPU. (A first design ran 3 processes on one
#     h100_3g.40gb slice; on 2026-09-21 no 3g slice was free for ~1.5 h while 143 of 160 1g slices
#     were, and the speed of 3 processes sharing a slice had never been measured.)
#   --time, set per network by e16_submit.sh: 0:35 / 1:15 / 1:30 / 2:30 for cnn / vit / cct /
#     resnet20. The largest shard holds 12 cells: 22 / 49 / 58 / 97 minutes, from the add cells'
#     times measured on 1g slices in E14/E13/E10 (75 / 152 / 212 / 290 s) and the clip arms' times
#     projected (the audit's per-operation costs; for resnet20, cnn's measured per-layer overhead
#     times its 39 hooked layers). About 1.5x on top. Measured on cnn seed 0 (21543912-14): the clip
#     cells ran at 0.80-0.84x their projection.

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
# shell into the job. The driver refuses any of these unless E16_SMOKE=1, and this job never sets it.
unset E16_SMOKE E16_ARMS E16_BATCH E16_GRID_LIMIT E16_TRAIN_SUBSET E16_CALIBRATE_AT E16_LOG_EVERY \
      E16_MERGE
export WARMUP_SGD_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export WARMUP_SGD_EPOCHS=15
export WARMUP_SGD_DEVICE=auto
export WARMUP_SGD_NUM_WORKERS=4   # as in E14: the worker count changes the trajectory
export E16_MODEL="${E16_MODEL:?set E16_MODEL on the sbatch line}"
export E16_SEED="${E16_SEED:?set E16_SEED on the sbatch line}"
export E16_MODE="${E16_MODE:?set E16_MODE on the sbatch line}"
export E16_SHARD="${E16_SHARD:?set E16_SHARD on the sbatch line}"
export E16_NSHARDS=3
export E16_OUT="$SLURM_SUBMIT_DIR/fisher_ref/outputs/e16_floor_clip_${E16_MODEL}_s${E16_SEED}_${E16_MODE}.json"
mkdir -p "$SLURM_SUBMIT_DIR/fisher_ref/outputs"

python -u fisher_ref/experiments/e16_floor_clip.py
