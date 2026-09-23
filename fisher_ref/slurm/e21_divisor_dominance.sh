#!/bin/bash
# E21 of docs/reports/plans/plan_lambda_dominance.md: lot 5's operational protocol replayed at the
# settings E14, E15 and E16 selected, to ask whether the number the optimizer divides by is set by
# the curvature or still by the safety constant. Pre-registered there before this script existed;
# read that section first.
#
# ONE SHARD of one (network, seed): E21_SHARD takes every E21_NSHARDS-th cell of that job's ordered
# list and writes fisher_ref/outputs/e21_divisor_<model>_s<seed>.shard<i>of<n>.json after every cell.
# fisher_ref/slurm/e21_merge.sh then merges the shards and applies the decision rules.
#
# Nothing here trains. A cell is a re-warm at frozen weights (lr = 0) followed by a read of the
# optimizer's own state, so the cost is the re-warm and the output is a table, not a trajectory.
#
# Do not sbatch this by hand: fisher_ref/slurm/e21_submit.sh submits the shards and the merge of one
# (network, seed) together, with the right --time, and refuses a modified tree.
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e21_divisor
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers.
#   --gpus / --cpus-per-task / --mem / OMP_NUM_THREADS: the E-series line (e4_fixed_average.sh, which
#     ran E13/E14, and e16_floor_clip.sh), so the re-warm runs on the hardware whose cells this
#     experiment is about. One process per 1g slice. Memory is not the constraint here -- lot 5's own
#     P2 jobs needed 80 GB because they built an exact reference matrix, and E21 builds none: it
#     reads the optimizer's factors and their eigenvalues, the widest of which is 577 x 577
#     (resnet20_cifar's input factor).
#   --time, set per network by e21_submit.sh: 1:00 / 1:00 / 1:30 / 2:00 for cnn / vit / cct /
#     resnet20, on 2 shards each. Projected from a measured cell: 14 s at batch 32 and 44 s at batch
#     128 for a 1 400-step re-warm of cnn_gn_cifar on one laptop processor, scaled to 2 000 steps and
#     by E16's own measured per-network step cost (cnn 1.0, vit 2.1, cct 2.7, resnet20 4.6). The
#     largest shard holds 27 cells. These are projections, not measurements: check the first job's
#     Elapsed before submitting the rest.

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

# Production settings, forced: --export=ALL would otherwise carry a smoke value from the submitting
# shell into the job. The driver refuses any of these unless E21_SMOKE=1, and this job never sets it.
unset E21_SMOKE E21_ARMS E21_MODES E21_FRACTIONS E21_REWARM_STEPS E21_MERGE
export WARMUP_SGD_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export WARMUP_SGD_DEVICE=auto
export WARMUP_SGD_NUM_WORKERS=4   # as in E14/E16: the worker count changes the batch stream
export E21_MODEL="${E21_MODEL:?set E21_MODEL on the sbatch line}"
export E21_SEED="${E21_SEED:?set E21_SEED on the sbatch line}"
export E21_SHARD="${E21_SHARD:?set E21_SHARD on the sbatch line}"
export E21_NSHARDS="${E21_NSHARDS:-2}"
export E21_OUT="$SLURM_SUBMIT_DIR/fisher_ref/outputs/e21_divisor_${E21_MODEL}_s${E21_SEED}.json"
mkdir -p "$SLURM_SUBMIT_DIR/fisher_ref/outputs"

python -u fisher_ref/experiments/e21_divisor_dominance.py
