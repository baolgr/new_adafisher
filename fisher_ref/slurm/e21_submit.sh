#!/bin/bash
# Submit one (network, seed) of E21: its shards, one h100_1g.10gb slice each
# (fisher_ref/slurm/e21_divisor_dominance.sh), and the merge that runs after them
# (fisher_ref/slurm/e21_merge.sh, --dependency=afterany).
#
#   usage, from the root of the checkout the jobs will run from:
#     bash fisher_ref/slurm/e21_submit.sh cnn_gn_cifar 0
#   all of E21 (4 networks x 1 seed = 8 shards + 4 merges), from ONE clean checkout:
#     for m in cnn_gn_cifar vit_micro_cifar cct_2_3x2_cifar resnet20_cifar; do
#       bash fisher_ref/slurm/e21_submit.sh "$m" 0
#     done
#
# ONE seed per network, deliberately. E21 measures a property of the operator, not an outcome of
# training: its spread over the layers and checkpoints of one run is reported per cell, and a second
# seed would buy a second draw of the same quantity rather than the comparison the question needs.
# Every arm within a (network, seed) starts from the same weights, so the comparison between arms is
# paired and exact whatever the seed is.
#
# Shards and --time per network (plan_lambda_dominance.md, E21, "Cost"): 2 shards each; the largest
# holds 27 cells. The limits are PROJECTIONS from one measured cell, scaled by E16's own per-network
# step cost -- check the first job's Elapsed with `sacct` before submitting the other three.
#
# Refuses a checkout whose tracked files are modified, and a network whose theta checkpoints are not
# there: a cell cannot be re-warmed from weights that do not exist, and discovering that four jobs in
# is how an afternoon is lost. Untracked files (the dataset link, fisher_ref/outputs/) do not count.

set -euo pipefail
model="${1:?usage: e21_submit.sh MODEL SEED}"
seed="${2:?usage: e21_submit.sh MODEL SEED}"

case "$model" in
  cnn_gn_cifar)    short=cnn; limit=01:00:00; nshards=2 ;;
  vit_micro_cifar) short=vit; limit=01:00:00; nshards=2 ;;
  cct_2_3x2_cifar) short=cct; limit=01:30:00; nshards=2 ;;
  resnet20_cifar)  short=r20; limit=02:00:00; nshards=2 ;;
  *) echo "e21_submit: unknown model '$model' (E21 runs the four networks E14 tuned)" >&2; exit 1 ;;
esac

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "e21_submit: tracked files are modified in $(pwd); refusing (one clean commit for all of E21)" >&2
  exit 1
fi
test -e dataset || { echo "e21_submit: no dataset/ in $(pwd)" >&2; exit 1; }

# The weights every cell re-warms from. Seed 0 lives in the grouped tree; another seed in outputs/seeds/.
if [ "$seed" = "0" ]; then
  theta="benchmarks/outputs/cifar10/$model/diag"
else
  theta="benchmarks/outputs/seeds/$model/seed$seed/diag"
fi
for fraction in 0.1 0.5 1; do
  test -f "$theta/ckpt_$fraction.pt" || {
    echo "e21_submit: $theta/ckpt_$fraction.pt is missing; pull the checkpoints first (CLAUDE.md," \
         "'Cluster sync' -- the results filter does not include ckpt_*.pt)" >&2
    exit 1
  }
done

vars="ALL,E21_MODEL=$model,E21_SEED=$seed,E21_NSHARDS=$nshards"
ids=()
for shard in $(seq 0 $((nshards - 1))); do
  ids+=("$(sbatch --parsable --job-name="e21_${short}_s${seed}_p${shard}" --time="$limit" \
           --export="$vars,E21_SHARD=$shard" fisher_ref/slurm/e21_divisor_dominance.sh)")
done
dep="afterany:$(IFS=:; echo "${ids[*]}")"
merge_id="$(sbatch --parsable --job-name="e21_${short}_s${seed}_merge" --dependency="$dep" \
              --export="$vars" fisher_ref/slurm/e21_merge.sh)"
echo "e21 $model seed $seed (commit $(git rev-parse --short HEAD)): shards ${ids[*]}, merge $merge_id"
