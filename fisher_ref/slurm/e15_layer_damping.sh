#!/bin/bash
# E15 of docs/reports/plan_lambda_dominance.md: does one safety constant per layer beat the best
# single one? Pre-registered there before this script existed; read that section first.
#
# One job per (network, seed): each runs kfac, ekfac and tekfac through the four arm families
# (reference, single lambda, S1-b per layer, network-adaptive) plus the two protocol controls.
# Submit from the repository root:
#
#   for s in 0 1 2 3 4; do
#     sbatch --job-name=e15_cnn_s$s --time=02:15:00 \
#       --export=ALL,E15_MODEL=cnn_gn_cifar,E15_SEED=$s fisher_ref/slurm/e15_layer_damping.sh
#     sbatch --job-name=e15_vit_s$s --time=04:00:00 \
#       --export=ALL,E15_MODEL=vit_micro_cifar,E15_SEED=$s fisher_ref/slurm/e15_layer_damping.sh
#   done
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e15_layer_damping
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers.
#   work: 3 modes x (1 reference + 5 single + 8 S1-b + 8 network-adaptive) = 66 runs of 15 epochs
#     at batch 32, plus 3 repro cells at seed 0 and, on vit_micro_cifar only, 3 weight-decay
#     controls at every seed. Measured per run: 68-70 s on cnn_gn_cifar (E14, jobs 21498168-72),
#     136-155 s on vit_micro_cifar (E13, jobs 21468492-96). So about 1 h 20 per cnn_gn_cifar seed
#     and 2 h 50 to 3 h per vit_micro_cifar seed, hence the two --time values above. The relative
#     damping adds a few reductions per step and the lambda log an eigendecomposition per layer
#     every 1000 steps, both small at these sizes. The JSON is rewritten after every run, so a
#     wall-clock kill costs one cell.
#   --gpus, --mem: as e4_fixed_average.sh, whose runs these are, with the same networks.

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
export WARMUP_SGD_DATA_ROOT="${WARMUP_SGD_DATA_ROOT:-$SLURM_SUBMIT_DIR/dataset}"
export WARMUP_SGD_EPOCHS="${WARMUP_SGD_EPOCHS:-15}"
export WARMUP_SGD_DEVICE="${WARMUP_SGD_DEVICE:-auto}"
# NOTE: sbatch --export truncates any value containing a comma at the first comma. E15_MODES is
# comma-separated, so it is set here and not on the sbatch line; E15_MODEL and E15_SEED are safe.
export E15_MODEL="${E15_MODEL:-cnn_gn_cifar}"
export E15_SEED="${E15_SEED:-0}"
export E15_MODES="${E15_MODES:-kfac,ekfac,tekfac}"
export E15_OUT="${E15_OUT:-$SLURM_SUBMIT_DIR/fisher_ref/outputs/e15_layer_damping_${E15_MODEL}_s${E15_SEED}.json}"

python fisher_ref/experiments/e15_layer_damping.py
