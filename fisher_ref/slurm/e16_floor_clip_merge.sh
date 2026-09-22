#!/bin/bash
# E16: merge the three shard files of one (network, seed, mode) into
# fisher_ref/outputs/e16_floor_clip_<model>_s<seed>_<mode>.json, check that every planned cell is
# there exactly once, and at seed 0 run the determinism and inertness checks (and the repro
# diagnostic). CPU only: it trains nothing. Submitted by fisher_ref/slurm/e16_submit.sh with
# --dependency=afterany on the three shards, so it also runs, and reports what is missing, when a
# shard failed. Its exit status is non-zero if a shard file or a planned cell is missing.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e16_merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

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

# The same production settings as the shards: the driver checks them before merging too.
unset E16_SMOKE E16_ARMS E16_BATCH E16_GRID_LIMIT E16_TRAIN_SUBSET E16_CALIBRATE_AT E16_LOG_EVERY \
      E16_SHARD
export WARMUP_SGD_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export WARMUP_SGD_EPOCHS=15
export WARMUP_SGD_DEVICE=auto
export WARMUP_SGD_NUM_WORKERS=4
export E16_MODEL="${E16_MODEL:?set E16_MODEL on the sbatch line}"
export E16_SEED="${E16_SEED:?set E16_SEED on the sbatch line}"
export E16_MODE="${E16_MODE:?set E16_MODE on the sbatch line}"
export E16_NSHARDS="${E16_NSHARDS:-3}"   # 3, or what e16_submit.sh passes (5 on resnet50)
export E16_MERGE=1
export E16_OUT="$SLURM_SUBMIT_DIR/fisher_ref/outputs/e16_floor_clip_${E16_MODEL}_s${E16_SEED}_${E16_MODE}.json"

python -u fisher_ref/experiments/e16_floor_clip.py
