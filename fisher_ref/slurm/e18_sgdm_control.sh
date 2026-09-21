#!/bin/bash
# E18 of docs/reports/plan_lambda_dominance.md: is ViT-S's deficit against AdamW the deficit of
# momentum SGD? Pre-registered there before this script was submitted; read that section first.
#
# One job per (dataset, seed). Two invocations of the bench, in this order:
#   1. campaign 2's grouped job for this model, flag for flag (benchmarks/slurm/<group>/
#      train_<model>_all.sh), with the `sgdm` arm appended to --arms. `diag` is the unbudgeted
#      reference; its measured time is the budget of the other seven arms, sgdm included.
#   2. `sgdm` alone at matched steps: 50 epochs, the nominal cosine -- exactly diag's 17 550 steps
#      and diag's own learning-rate schedule, from the same initialisation and batch order.
# Submit from the repository root:
#
#   for s in 0 1 2; do
#     sbatch --job-name=e18_c10_s$s \
#       --export=ALL,E18_MODEL=vit_small_cifar,E18_GROUP=cifar10,E18_SEED=$s \
#       fisher_ref/slurm/e18_sgdm_control.sh
#     sbatch --job-name=e18_c100_s$s \
#       --export=ALL,E18_MODEL=vit_small_cifar100,E18_GROUP=cifar100,E18_SEED=$s \
#       fisher_ref/slurm/e18_sgdm_control.sh
#   done
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates the campaign's own jobs.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e18_sgdm_control
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:45:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers.
#   work: 8 arms under the wall-clock budget + 1 at matched steps = 9 ViT-S runs. Measured in
#     campaign 2: one arm = 997 s (CIFAR-10) and 990 s (CIFAR-100) of training, plus ~6 % for
#     validation, i.e. ~1 060 s. 9 x 1 060 s = 2.65 h, plus ~40 s of setup. 3:45 leaves ~40 %.
#   --gpus, --cpus-per-task, --mem: those of the campaign-2 job this one replays.
# The report of invocation 1 is rewritten after every arm (runner.main), so a wall-clock kill costs
# the arm in progress only.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023 gcc/13.3 cuda/12.6 python/3.11.5

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements-cluster.txt

export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR"
DATA_ROOT="${DATA_ROOT:-$SLURM_SUBMIT_DIR/dataset}"
E18_MODEL="${E18_MODEL:-vit_small_cifar}"
E18_GROUP="${E18_GROUP:-cifar10}"
E18_SEED="${E18_SEED:-0}"
OUT="$SLURM_SUBMIT_DIR/benchmarks/outputs/controls/e18_sgdm/$E18_GROUP/$E18_MODEL/seed$E18_SEED"
echo "E18: model=$E18_MODEL group=$E18_GROUP seed=$E18_SEED out=$OUT"

# 1. Campaign 2's protocol, verbatim, plus sgdm.
python -m "benchmarks.models.$E18_MODEL.bench" \
  --arms diag kfac ekfac tkfac tekfac adam adamw sgdm \
  --epochs 50 \
  --budget-mode wct \
  --reference-arm diag \
  --max-epoch-factor 3 \
  --lr-schedule budget \
  --num-workers 8 \
  --no-allow-download \
  --data-root "$DATA_ROOT" \
  --checkpoints 0,0.01,0.1,0.5,1 \
  --seed "$E18_SEED" \
  --output-dir "$OUT/wct"

# 2. sgdm at diag's step count and schedule.
python -m "benchmarks.models.$E18_MODEL.bench" \
  --arms sgdm \
  --epochs 50 \
  --budget-mode epochs \
  --lr-schedule nominal \
  --num-workers 8 \
  --no-allow-download \
  --data-root "$DATA_ROOT" \
  --checkpoints 0,0.01,0.1,0.5,1 \
  --seed "$E18_SEED" \
  --output-dir "$OUT/matched_steps"
