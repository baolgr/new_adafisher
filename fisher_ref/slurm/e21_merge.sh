#!/bin/bash
# E21: merge the shard files of one (network, seed) into
# fisher_ref/outputs/e21_divisor_<model>_s<seed>.json, check that every planned cell is there exactly
# once, and apply the decision rules (the gate on the bridge arm, then rules 2-4). CPU only: it
# trains nothing and reads nothing from the GPU. Submitted by fisher_ref/slurm/e21_submit.sh with
# --dependency=afterany on the shards, so it also runs, and reports what is missing, when a shard
# failed. Its exit status is non-zero if a shard file is missing.
#
# A missing or crashed cell does not silently become a verdict: `decisions.rules.2_3_4.readable` is
# false and every affected row reads "inconclusive".

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e21_merge
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
unset E21_SMOKE E21_ARMS E21_MODES E21_FRACTIONS E21_REWARM_STEPS E21_SHARD
export WARMUP_SGD_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export WARMUP_SGD_DEVICE=cpu
export WARMUP_SGD_NUM_WORKERS=2
export E21_MODEL="${E21_MODEL:?set E21_MODEL on the sbatch line}"
export E21_SEED="${E21_SEED:?set E21_SEED on the sbatch line}"
export E21_NSHARDS="${E21_NSHARDS:?set E21_NSHARDS on the sbatch line}"
export E21_MERGE=1
export E21_OUT="$SLURM_SUBMIT_DIR/fisher_ref/outputs/e21_divisor_${E21_MODEL}_s${E21_SEED}.json"

python -u fisher_ref/experiments/e21_divisor_dominance.py
