#!/bin/bash
# E16 of docs/reports/plan_lambda_dominance.md: a floor (family A) or a clip (family B) instead of
# the added safety constant, with E16's own add baseline. Pre-registered there, and specified in
# docs/reports/plan_floor_clip.md, before this script existed; read both first.
#
# One job per (network, seed, mode) -- 30 jobs. Each runs its 34 cells (36 at seed 0: the
# determinism and repro checks) as 3 processes sharing one h100_3g.40gb slice, then merges their
# files into fisher_ref/outputs/e16_floor_clip_<model>_s<seed>_<mode>.json. Every cell but the repro
# diagnostic runs with norm_exact_rescaling (plan_floor_clip.md §11).
#
# Commit everything first, then pull that commit on the cluster: e16_decisions.py refuses files
# from a dirty tree or from different commits. Submit ONE job first and read its timing and its
# seed-0 checks (the merge line prints them); then the rest. From the repository root:
#
#   for s in 0 1 2 3 4; do for m in ekfac tekfac; do
#     sbatch --job-name=e16_cnn_s${s}_$m --time=00:50:00 \
#       --export=ALL,E16_MODEL=cnn_gn_cifar,E16_SEED=$s,E16_MODE=$m fisher_ref/slurm/e16_floor_clip.sh
#     sbatch --job-name=e16_vit_s${s}_$m --time=01:40:00 \
#       --export=ALL,E16_MODEL=vit_micro_cifar,E16_SEED=$s,E16_MODE=$m fisher_ref/slurm/e16_floor_clip.sh
#     sbatch --job-name=e16_cct_s${s}_$m --time=02:00:00 \
#       --export=ALL,E16_MODEL=cct_2_3x2_cifar,E16_SEED=$s,E16_MODE=$m fisher_ref/slurm/e16_floor_clip.sh
#   done; done
#
# Then, once all thirty are in:  python fisher_ref/experiments/e16_decisions.py
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e16_floor_clip
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers.
#   --gpus: a 3g.40gb MIG slice (Rorqual offers 1g.10gb, 2g.20gb, 3g.40gb and whole H100s; sinfo,
#     2026-09-21). These networks are small and latency-bound at batch 32: one run leaves most of a
#     slice idle, so the job runs 3 independent cells at a time on 3/7 of an H100 instead of one at
#     a time on 1/7. Every cell of E16 runs this way -- including its add baseline -- so every
#     comparison is between runs made on the same hardware.
#   --cpus-per-task / E16_NSHARDS: 3 processes x (1 main + 4 data-loader workers) = 15 of 16 cores.
#     The worker count stays 4, as in E14: it changes the trajectory, so it is the same in every
#     cell. OMP_NUM_THREADS=1 per process, since the work is on the GPU.
#   --time, per job, from the per-run times measured on 1g slices (E14/E13/E10's add cells: 75 /
#     152 / 205 s; the clips projected at up to 1.9x that): about 65 / 145 / 170 minutes of cells
#     per (network, mode), over 3 processes about 22 / 48 / 57 minutes if the slice keeps each
#     process at its 1g speed. Not measured on a 3g slice yet -- hence ~2x on top, and one job first.
#     The 3 processes share the slice by time-slicing (no MPS), so how much they overlap is exactly
#     what the first job measures: every cell prints its own seconds in the log below. If they do not
#     overlap at all, a job takes its full cell time (65 / 145 / 170 min, above the limits): raise
#     --time to ~1.3x the measured job time before submitting the rest. Finished cells survive a
#     wall-clock kill (each is written as soon as it ends), but a resubmitted job reruns them all.
#   --mem: 3 processes, each holding CIFAR-10 in 5 processes; well under 48G.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023 gcc/13.3 cuda/12.6 python/3.11.5

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements-cluster.txt

export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1

# Production settings, forced: --export=ALL would otherwise carry a smoke value from the submitting
# shell into the job. The driver refuses any of these unless E16_SMOKE=1, and this job never sets it.
unset E16_SMOKE E16_ARMS E16_BATCH E16_GRID_LIMIT E16_TRAIN_SUBSET E16_CALIBRATE_AT E16_LOG_EVERY \
      E16_SHARD E16_MERGE
export WARMUP_SGD_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export WARMUP_SGD_EPOCHS=15
export WARMUP_SGD_DEVICE=auto
export WARMUP_SGD_NUM_WORKERS=4   # as in E14: the worker count changes the trajectory
export E16_MODEL="${E16_MODEL:?set E16_MODEL on the sbatch line}"
export E16_SEED="${E16_SEED:?set E16_SEED on the sbatch line}"
export E16_MODE="${E16_MODE:?set E16_MODE on the sbatch line}"
export E16_NSHARDS=3
export E16_OUT="$SLURM_SUBMIT_DIR/fisher_ref/outputs/e16_floor_clip_${E16_MODEL}_s${E16_SEED}_${E16_MODE}.json"

pids=()
for i in $(seq 0 $((E16_NSHARDS - 1))); do
  E16_SHARD=$i python fisher_ref/experiments/e16_floor_clip.py > "${E16_OUT%.json}.shard${i}.log" 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
for i in $(seq 0 $((E16_NSHARDS - 1))); do
  echo "===== shard $i ====="; cat "${E16_OUT%.json}.shard${i}.log"
done
# Merge whatever exists: it records missing shards and cells, and exits non-zero if any.
E16_MERGE=1 python fisher_ref/experiments/e16_floor_clip.py || status=1
exit $status
