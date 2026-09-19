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
#   jid_resnet=$(sbatch --parsable benchmarks/slurm/cifar10/train_resnet50_cifar_diag.sh)
#   jid_vit=$(sbatch --parsable benchmarks/slurm/cifar10/train_vit_small_cifar_diag.sh)
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
  # Jobs and results are grouped by dataset (benchmarks/slurm/<group>/, outputs/<group>/<model>/).
  # The group is read straight out of the bench's own `output_group=` line rather than imported,
  # because this job loads a bare python module with no venv and no torch.
  # The model folders live under benchmarks/models/ since the step-1 reorganisation. Checked
  # explicitly: `set -e` would otherwise abort on sed's own exit status, with no message saying
  # which file was missing.
  bench="benchmarks/models/${model}/bench.py"
  if [ ! -f "$bench" ]; then
    echo "$bench does not exist — is $model a benchmarks/models/ folder? aborting" >&2
    exit 1
  fi
  group=$(sed -n 's/.*output_group="\([^"]*\)".*/\1/p' "$bench")
  if [ -z "$group" ]; then
    echo "$bench declares no output_group — aborting" >&2
    exit 1
  fi
  manifest="benchmarks/outputs/${group}/${model}/diag/manifest.json"
  if [ ! -f "$manifest" ]; then
    echo "missing $manifest — the diag reference job did not write it, aborting" >&2
    exit 1
  fi
  budget=$(python -c "import json;print(json.load(open('$manifest'))['arms']['diag']['total_s'])")
  echo "$model ($group) diag reference total_s=$budget"
  for arm in "${REMAINING_ARMS[@]}"; do
    sbatch --export=ALL,WCT_BUDGET="$budget" "benchmarks/slurm/${group}/train_${model}_${arm}.sh"
  done
done
