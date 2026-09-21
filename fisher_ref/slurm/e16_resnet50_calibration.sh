#!/bin/bash
# E16, fourth amendment: calibrate ResNet-50 (docs/reports/plan_floor_clip.md §12) before its
# exploratory E16 runs. One job per (mode, part): five scan parts (the add arm at seed 0, two lambda
# values each, 15 epochs) and one timing part (two epochs of add and of each clip arm, wall time only)
# per mode -- 12 jobs. From the root of the checkout E16 runs from (the same clean commit):
#
#   for m in ekfac tekfac; do
#     for p in scan0 scan1 scan2 scan3 scan4; do
#       sbatch --job-name=e16c_r50_${m}_$p --time=02:30:00 \
#         --export=ALL,E16_MODE=$m,E16C_PART=$p fisher_ref/slurm/e16_resnet50_calibration.sh
#     done
#     sbatch --job-name=e16c_r50_${m}_timing --time=01:00:00 \
#       --export=ALL,E16_MODE=$m,E16C_PART=timing fisher_ref/slurm/e16_resnet50_calibration.sh
#   done
#
# Then, once the twelve files are in fisher_ref/outputs/, on any machine with the environment:
#   E16C_SUMMARIZE=1 python fisher_ref/experiments/e16_resnet50_calibration.py
# prints the pre-registered lambda grid and the measured cost of ResNet-50's reduced E16.
#
# Why these numbers. The E-series line (1g slice, 8 CPUs, 24G, OMP_NUM_THREADS=8), as every E16 job.
# --time: ResNet-50 has never run at batch 32. Campaign 2 measured 0.214 s per step for ekfac at
# batch 128 on a 1g slice (0.042 s of it in the optimizer), which scales to roughly 0.08-0.09 s per
# step at batch 32: ~30 min per 15-epoch add run, so ~1 h per scan part, and ~25 min for the timing
# part. Those are estimates, hence 2:30 and 1:00. The timing part is what replaces them by
# measurements.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e16c_r50
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:30:00
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

# Production settings, forced, as in e16_floor_clip.sh.
unset E16_SMOKE E16_ARMS E16_BATCH E16_GRID_LIMIT E16_TRAIN_SUBSET E16_CALIBRATE_AT E16_LOG_EVERY \
      E16_SHARD E16_MERGE E16_NSHARDS E16C_SMOKE E16C_SUMMARIZE E16C_DIR
export E16_MODEL=resnet50_cifar
export E16_SEED=0
export E16_MODE="${E16_MODE:?set E16_MODE on the sbatch line}"
export E16C_PART="${E16C_PART:?set E16C_PART on the sbatch line}"
export WARMUP_SGD_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export WARMUP_SGD_DEVICE=auto
export WARMUP_SGD_NUM_WORKERS=4
if [ "$E16C_PART" = timing ]; then export WARMUP_SGD_EPOCHS=2; else export WARMUP_SGD_EPOCHS=15; fi
mkdir -p "$SLURM_SUBMIT_DIR/fisher_ref/outputs"

python -u fisher_ref/experiments/e16_resnet50_calibration.py
