#!/bin/bash
# E19: merge the shard files of one (network, seed, mode) into
# fisher_ref/outputs/e19_layer_damping_<model>_s<seed>_<mode>.json and check that every planned cell
# is there exactly once. CPU only: it trains nothing. Submitted by fisher_ref/slurm/e19_submit.sh
# with --dependency=afterany on the shards, so it also runs, and reports what is missing, when a
# shard failed. Its exit status is non-zero if a shard file or a planned cell is missing. The
# comparison with E16's stored cells (rule 0a) is e19_decisions.py's, not this job's.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e19_merge
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
unset E19_SMOKE E19_ARMS E19_BATCH E19_GRID_LIMIT E19_TRAIN_SUBSET E19_LOG_EVERY E19_SHARD
export WARMUP_SGD_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export WARMUP_SGD_EPOCHS=15
export WARMUP_SGD_DEVICE=auto
export WARMUP_SGD_NUM_WORKERS=4
export E19_MODEL="${E19_MODEL:?set E19_MODEL on the sbatch line}"
export E19_SEED="${E19_SEED:?set E19_SEED on the sbatch line}"
export E19_MODE="${E19_MODE:?set E19_MODE on the sbatch line}"
export E19_NSHARDS="${E19_NSHARDS:?set E19_NSHARDS on the sbatch line}"
export E19_MERGE=1
export E19_OUT="$SLURM_SUBMIT_DIR/fisher_ref/outputs/e19_layer_damping_${E19_MODEL}_s${E19_SEED}_${E19_MODE}.json"

python -u fisher_ref/experiments/e19_layer_damping_transfer.py
