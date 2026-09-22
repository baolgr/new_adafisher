#!/bin/bash
# E16's Adam/AdamW baselines under E16's exact protocol (fisher_ref/experiments/e16_baselines.py):
# one job per (network, seed, optimizer), 3 cells each (baseline_lr x {1/3, 1, 3}). A reference
# added after E16's results; it votes in none of E16's rules. From the checkout's root:
#
#   for n in cnn_gn_cifar vit_micro_cifar cct_2_3x2_cifar resnet20_cifar; do
#     for s in 0 1 2 3 4; do for o in adam adamw; do
#       sbatch --job-name=e16b_${n%%_*}_s${s}_$o --time=$(case $n in cnn*) echo 00:20:00;;
#         vit*) echo 00:35:00;; cct*) echo 00:45:00;; *) echo 01:00:00;; esac) \
#         --export=ALL,E16B_MODEL=$n,E16B_SEED=$s,E16B_OPT=$o fisher_ref/slurm/e16_baselines.sh
#     done; done; done
#   for s in 0 1 2; do for o in adam adamw; do
#     sbatch --job-name=e16b_resnet50_s${s}_$o --time=03:00:00 \
#       --export=ALL,E16B_MODEL=resnet50_cifar,E16B_SEED=$s,E16B_OPT=$o fisher_ref/slurm/e16_baselines.sh
#   done; done
#
# --time: three runs per job. An Adam step costs no more than an E16 add step, whose 15-epoch runs
# were measured at 75 / 150 / 210 / 330 / 2 490 s on a 1g slice (cnn / vit / cct / resnet20 /
# resnet50), so the limits are ~2-4x three add runs. The E-series line, as every E16 job.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e16b
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
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
unset E16B_SMOKE E16B_TRAIN_SUBSET E16B_DIR
export WARMUP_SGD_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export WARMUP_SGD_EPOCHS=15
export WARMUP_SGD_DEVICE=auto
export WARMUP_SGD_NUM_WORKERS=4
export E16B_MODEL="${E16B_MODEL:?set E16B_MODEL}" E16B_SEED="${E16B_SEED:?set E16B_SEED}" \
       E16B_OPT="${E16B_OPT:?set E16B_OPT}"
mkdir -p "$SLURM_SUBMIT_DIR/fisher_ref/outputs"

python -u fisher_ref/experiments/e16_baselines.py
