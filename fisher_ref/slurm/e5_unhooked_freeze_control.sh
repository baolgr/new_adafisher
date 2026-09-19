#!/bin/bash
# E5 control of docs/reports/plan_lambda_dominance.md: is E4's one real gain a curvature effect, or
# just the parameters the optimizer does not precondition being frozen?
#
# E4 found +4.32 accuracy points on cnn_gn_cifar/kfac when the safety constant drops from 1e-3 to
# 1e-8 with the step-size cap held. But holding the cap means lr = lambda, and a parameter the
# optimizer does not precondition steps by lr times its momentum -- so those parameters have their
# step shrunk by 100 000 over that sweep. They stop moving. On cnn_gn_cifar they are the 224 affine
# parameters of its three GroupNorm layers. This job asks whether freezing them, at the DEFAULT
# safety constant, reproduces the gain on its own.
#
# It gates everything downstream: if it does, the one positive result in the investigation is a
# step-size artifact and the per-layer damping plan rests on nothing.
#
# Hand-written, NOT generated. Submit from the repository root:
#   sbatch fisher_ref/slurm/e5_unhooked_freeze_control.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e5_freeze_control
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers. 5 modes x 3 arms = 15 runs of 15 epochs at batch 32, the same protocol E4 used.
# E4's cnn_gn_cifar job did 50 such runs in 60 minutes, so 15 is about 20 minutes plus 2 for setup.
# 01:30:00 is deliberately generous. The JSON is rewritten after every mode.

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
export E5_MODELS="${E5_MODELS:-cnn_gn_cifar}"
export E5_MODES="${E5_MODES:-diag,kfac,ekfac,tkfac,tekfac}"
export E5_SMALL_LAMBDA="${E5_SMALL_LAMBDA:-1e-8}"
export E4_BATCH="${E4_BATCH:-32}"          # run_one is imported from the E4 driver
export E4_SEED="${E4_SEED:-0}"

python fisher_ref/experiments/e5_unhooked_freeze_control.py
