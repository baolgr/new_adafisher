#!/bin/bash
# Dispatcher for the WCT-budgeted training jobs (docs/reports/plan_lot8.md §0.7).
# Hand-maintained (unlike the scripts generate_jobs.py owns): it exists to remove the manual
# "read total_s back, then sbatch the remaining arms" step between the reference arms and
# everything else, so a whole model's submission can be queued unattended in one sitting.
#
# MODELS is the list of model folders to dispatch — the subset you have reference runs for, not
# necessarily all eight (benchmarks/slurm/README.md, "Submission order"). Override it at
# submission time:
#   sbatch --export=ALL,MODELS="resnet20_cifar cct_2_3x2_cifar" ... dispatch_remaining_arms.sh
#
# Usage, right after submitting the two diag reference jobs:
#   jid_resnet=$(sbatch --parsable benchmarks/slurm/train_resnet50_cifar_diag.sh)
#   jid_vit=$(sbatch --parsable benchmarks/slurm/train_vit_small_cifar_diag.sh)
#   sbatch --dependency=afterok:$jid_resnet:$jid_vit benchmarks/slurm/dispatch_remaining_arms.sh
#
# --dependency=afterok holds this job in the queue until BOTH reference jobs finish with exit
# code 0. If either fails or is cancelled, the dependency is never satisfied and this job sits as
# DependencyNeverSatisfied until manually cancelled (`scancel`) — check `sq` for that state if
# the remaining 12 arms never appear.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=dispatch_arms
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=benchmarks/slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023 python/3.11.5

IFS=' ' read -r -a MODELS <<< "${MODELS:-resnet50_cifar vit_small_cifar}"
REMAINING_ARMS=(kfac ekfac tkfac tekfac adam adamw)

for model in "${MODELS[@]}"; do
  manifest="benchmarks/outputs/${model}/diag/manifest.json"
  if [ ! -f "$manifest" ]; then
    echo "missing $manifest — the diag reference job did not write it, aborting" >&2
    exit 1
  fi
  budget=$(python -c "import json;print(json.load(open('$manifest'))['arms']['diag']['total_s'])")
  echo "$model diag reference total_s=$budget"
  for arm in "${REMAINING_ARMS[@]}"; do
    sbatch --export=ALL,WCT_BUDGET="$budget" "benchmarks/slurm/train_${model}_${arm}.sh"
  done
done
