#!/bin/bash
# Submit one (network, seed, mode) of E19: its shards, one h100_1g.10gb slice each
# (fisher_ref/slurm/e19_layer_damping_transfer.sh), and the merge that runs after them
# (fisher_ref/slurm/e19_merge.sh, --dependency=afterany).
#
#   usage, from the root of the checkout the jobs will run from:
#     bash fisher_ref/slurm/e19_submit.sh cct_2_3x2_cifar 0 ekfac
#   all of E19 (26 (network, seed, mode) = 64 shards + 26 merges), from ONE clean checkout:
#     for m in ekfac tekfac; do
#       for s in 0 1 2 3 4; do bash fisher_ref/slurm/e19_submit.sh cct_2_3x2_cifar $s $m; done
#       for s in 0 1 2 3 4; do bash fisher_ref/slurm/e19_submit.sh resnet20_cifar $s $m; done
#       for s in 0 1 2;     do bash fisher_ref/slurm/e19_submit.sh resnet50_cifar $s $m; done
#     done
#
# Shards and --time per network (plan_lambda_dominance.md, E19, "Cost"): the largest shard holds
# 10 / 9 / 2 cells, about 35 / 55 / 85 min, for cct / resnet20 / resnet50.
#
# Refuses a checkout whose tracked files are modified: e19_decisions.py refuses files from a dirty
# tree or from different commits. Untracked files (the dataset link, fisher_ref/outputs/) do not
# count, as in the driver's own provenance.

set -euo pipefail
model="${1:?usage: e19_submit.sh MODEL SEED MODE}"
seed="${2:?usage: e19_submit.sh MODEL SEED MODE}"
mode="${3:?usage: e19_submit.sh MODEL SEED MODE}"

case "$model" in
  cct_2_3x2_cifar) short=cct; limit=01:00:00; nshards=2; seeds="0 1 2 3 4" ;;
  resnet20_cifar)  short=r20; limit=01:30:00; nshards=2; seeds="0 1 2 3 4" ;;
  resnet50_cifar)  short=r50; limit=02:30:00; nshards=4; seeds="0 1 2" ;;
  *) echo "e19_submit: unknown model '$model'" >&2; exit 1 ;;
esac
case "$mode" in ekfac|tekfac) ;; *) echo "e19_submit: mode must be ekfac or tekfac" >&2; exit 1 ;; esac
case " $seeds " in *" $seed "*) ;; *) echo "e19_submit: $model runs seeds $seeds" >&2; exit 1 ;; esac

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "e19_submit: tracked files are modified in $(pwd); refusing (one clean commit for all of E19)" >&2
  exit 1
fi
test -e dataset || { echo "e19_submit: no dataset/ in $(pwd)" >&2; exit 1; }

vars="ALL,E19_MODEL=$model,E19_SEED=$seed,E19_MODE=$mode,E19_NSHARDS=$nshards"
ids=()
for shard in $(seq 0 $((nshards - 1))); do
  ids+=("$(sbatch --parsable --job-name="e19_${short}_s${seed}_${mode}_p${shard}" --time="$limit" \
           --export="$vars,E19_SHARD=$shard" fisher_ref/slurm/e19_layer_damping_transfer.sh)")
done
dep="afterany:$(IFS=:; echo "${ids[*]}")"
merge_id="$(sbatch --parsable --job-name="e19_${short}_s${seed}_${mode}_merge" --dependency="$dep" \
              --export="$vars" fisher_ref/slurm/e19_merge.sh)"
echo "e19 $model seed $seed $mode (commit $(git rev-parse --short HEAD)): shards ${ids[*]}, merge $merge_id"
