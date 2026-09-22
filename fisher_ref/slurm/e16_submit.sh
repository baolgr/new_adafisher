#!/bin/bash
# Submit one (network, seed, mode) of E16: its three shards, one h100_1g.10gb slice each
# (fisher_ref/slurm/e16_floor_clip.sh), and the merge that runs after them
# (fisher_ref/slurm/e16_floor_clip_merge.sh, --dependency=afterany).
#
#   usage, from the root of the checkout the jobs will run from:
#     bash fisher_ref/slurm/e16_submit.sh cnn_gn_cifar 0 ekfac
#   the thirty that vote (submitted on 2026-09-21 from a clone at 7663d62, which is their commit):
#     for s in 0 1 2 3 4; do for m in ekfac tekfac; do for n in cnn_gn_cifar vit_micro_cifar \
#       cct_2_3x2_cifar; do bash fisher_ref/slurm/e16_submit.sh $n $s $m; done; done; done
#   the ten exploratory ResNet-20 ones, from a checkout at the fourth amendment's commit (one commit
#   per network is what e16_decisions.py requires of an exploratory network):
#     for s in 0 1 2 3 4; do for m in ekfac tekfac; do
#       bash fisher_ref/slurm/e16_submit.sh resnet20_cifar $s $m; done; done
#
# --time per shard: the largest shard, projected from the driver's own shard assignment, is 22 / 49
# / 58 / 97 min for cnn / vit / cct / resnet20 -- add times measured on 1g, clip times projected;
# measured on cnn seed 0 (job 21543912-14), the clip cells ran at 0.80-0.84x their projection.
#
# Refuses a checkout whose tracked files are modified: e16_decisions.py refuses files from a dirty
# tree or from different commits, so every job must be submitted from one clean commit. Untracked
# files (the dataset link, fisher_ref/outputs/) do not count, as in the driver's own provenance.

set -euo pipefail
model="${1:?usage: e16_submit.sh MODEL SEED MODE}"
seed="${2:?usage: e16_submit.sh MODEL SEED MODE}"
mode="${3:?usage: e16_submit.sh MODEL SEED MODE}"

case "$model" in
  cnn_gn_cifar)    short=cnn; limit=00:35:00 ;;
  vit_micro_cifar) short=vit; limit=01:15:00 ;;
  cct_2_3x2_cifar) short=cct; limit=01:30:00 ;;
  resnet20_cifar)  short=r20; limit=02:30:00 ;;   # exploratory (plan_floor_clip.md §12)
  # exploratory, reduced design at seeds 0-2 on its calibrated grid (plan_floor_clip.md §12): 5
  # shards, the largest 2.69 h from the calibration's MEASURED per-run times (add 2486 s, clips
  # 3320-3735 s on a 1g slice), so each job stays under 3 h instead of 5 with 3 shards
  resnet50_cifar)  short=r50; limit=04:00:00; nshards=5 ;;
  *) echo "e16_submit: unknown model '$model'" >&2; exit 1 ;;
esac
case "$mode" in ekfac|tekfac) ;; *) echo "e16_submit: mode must be ekfac or tekfac" >&2; exit 1 ;; esac
case "$seed" in 0|1|2|3|4) ;; *) echo "e16_submit: seed must be 0-4" >&2; exit 1 ;; esac

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "e16_submit: tracked files are modified in $(pwd); refusing (one clean commit for all of E16)" >&2
  exit 1
fi
test -e dataset || { echo "e16_submit: no dataset/ in $(pwd)" >&2; exit 1; }

nshards="${nshards:-3}"
vars="ALL,E16_MODEL=$model,E16_SEED=$seed,E16_MODE=$mode,E16_NSHARDS=$nshards"
ids=()
for shard in $(seq 0 $((nshards - 1))); do
  ids+=("$(sbatch --parsable --job-name="e16_${short}_s${seed}_${mode}_p${shard}" --time="$limit" \
           --export="$vars,E16_SHARD=$shard" fisher_ref/slurm/e16_floor_clip.sh)")
done
dep="afterany:$(IFS=:; echo "${ids[*]}")"
merge_id="$(sbatch --parsable --job-name="e16_${short}_s${seed}_${mode}_merge" --dependency="$dep" \
              --export="$vars" fisher_ref/slurm/e16_floor_clip_merge.sh)"
echo "e16 $model seed $seed $mode (commit $(git rev-parse --short HEAD)): shards ${ids[*]}, merge $merge_id"
