#!/bin/bash
# Submit one (network, seed, mode) of E16: its three shards, one h100_1g.10gb slice each
# (fisher_ref/slurm/e16_floor_clip.sh), and the merge that runs after them
# (fisher_ref/slurm/e16_floor_clip_merge.sh, --dependency=afterany).
#
#   usage, from the root of the checkout the jobs will run from:
#     bash fisher_ref/slurm/e16_submit.sh cnn_gn_cifar 0 ekfac
#   all thirty:
#     for s in 0 1 2 3 4; do for m in ekfac tekfac; do for n in cnn_gn_cifar vit_micro_cifar \
#       cct_2_3x2_cifar; do bash fisher_ref/slurm/e16_submit.sh $n $s $m; done; done; done
#
# Refuses a checkout whose tracked files are modified: e16_decisions.py refuses files from a dirty
# tree or from different commits, so all thirty must be submitted from one clean commit. Untracked
# files (the dataset link, fisher_ref/outputs/) do not count, as in the driver's own provenance.

set -euo pipefail
model="${1:?usage: e16_submit.sh MODEL SEED MODE}"
seed="${2:?usage: e16_submit.sh MODEL SEED MODE}"
mode="${3:?usage: e16_submit.sh MODEL SEED MODE}"

case "$model" in
  cnn_gn_cifar)    short=cnn; limit=00:35:00 ;;
  vit_micro_cifar) short=vit; limit=01:15:00 ;;
  cct_2_3x2_cifar) short=cct; limit=01:30:00 ;;
  *) echo "e16_submit: unknown model '$model'" >&2; exit 1 ;;
esac
case "$mode" in ekfac|tekfac) ;; *) echo "e16_submit: mode must be ekfac or tekfac" >&2; exit 1 ;; esac
case "$seed" in 0|1|2|3|4) ;; *) echo "e16_submit: seed must be 0-4" >&2; exit 1 ;; esac

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "e16_submit: tracked files are modified in $(pwd); refusing (one clean commit for all 30)" >&2
  exit 1
fi
test -e dataset || { echo "e16_submit: no dataset/ in $(pwd)" >&2; exit 1; }

vars="ALL,E16_MODEL=$model,E16_SEED=$seed,E16_MODE=$mode"
ids=()
for shard in 0 1 2; do
  ids+=("$(sbatch --parsable --job-name="e16_${short}_s${seed}_${mode}_p${shard}" --time="$limit" \
           --export="$vars,E16_SHARD=$shard" fisher_ref/slurm/e16_floor_clip.sh)")
done
dep="afterany:$(IFS=:; echo "${ids[*]}")"
merge_id="$(sbatch --parsable --job-name="e16_${short}_s${seed}_${mode}_merge" --dependency="$dep" \
              --export="$vars" fisher_ref/slurm/e16_floor_clip_merge.sh)"
echo "e16 $model seed $seed $mode (commit $(git rev-parse --short HEAD)): shards ${ids[*]}, merge $merge_id"
