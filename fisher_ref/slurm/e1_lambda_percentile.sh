#!/bin/bash
# E1 + E3 of docs/reports/plan_lambda_dominance.md, in one job.
#
#   E1  where does Lambda sit inside each layer's curvature spectrum -- as a PERCENTILE, not as a
#       maximum, and both at the scale the optimizer ships with and at the scale the two known
#       shrinkages (docs/reports/audit_step.md section 4.3) say it should be at.
#   E3  the same measurement at three batch sizes. The stored curvature goes as 1/batch^2, so the
#       batch size moves the very ratio E1 measures with no code change. Sharp prediction: the
#       "as shipped" spectrum moves as 1/batch^2 and the "corrected" one does not move at all.
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks
# and this is an analysis job. Same convention as lambda_vs_curvature.sh, whose machinery this
# script imports rather than duplicates.
#
# Submit from the repository root:  sbatch fisher_ref/slurm/e1_lambda_percentile.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e1_lambda_percentile
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# Why these numbers.
#   --gpus: a dedicated slice, for the reason lambda_vs_curvature.sh gives -- a job without one
#     lands on the shared backfill queue whose throughput drifts under other users' load, which is
#     what invalidated job 21122754. Not for speed.
#   --time: three times lambda_vs_curvature's workload (three batch sizes), and that job measured
#     7.5 min of compute for 7 runs x 5 modes at 1000 re-warm steps. The batch-512 pass costs about
#     4x the batch-128 pass per step in data, so budget ~6x the single-batch job, plus ~2 min of
#     setup. 03:00:00 is deliberately generous; check the log's per-mode timings before trusting a
#     tighter value on a rerun.
#   --mem: 32G, double lambda_vs_curvature's. This job POOLS every layer's spectrum to ask what
#     fraction of the whole network's directions sit above Lambda, so it holds one fp64 array of
#     sum_l (d_in_l x d_out_l) entries -- 1.06 M for resnet20_cifar, 2.5 M for cct_2_3x2_cifar, i.e.
#     tens of MB, but it holds the per-layer copies alongside it while concatenating.
#
# The re-warm length is not a free choice: the checkpoints hold theta only, every mode seeds its
# EMA with the identity, and 0.08^k must fall below Lambda before the reading means anything.
# rewarm_fidelity.py fixed that at k=10, hence 1000 = 10 * TCov. A shorter re-warm does not make
# the job cheaper, it makes it wrong: at 200 steps this script reports kfac's largest curvature at
# 14x Lambda, which is the identity seed, not curvature (measured, laptop, 2026-09-17).

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
export DATA_ROOT="${DATA_ROOT:-$SLURM_SUBMIT_DIR/dataset}"
export REWARM_STEPS="${REWARM_STEPS:-1000}"
export CKPT_FRACTION="${CKPT_FRACTION:-0.5}"
export BATCH_SIZES="${BATCH_SIZES:-32,128,512}"

python fisher_ref/experiments/e1_lambda_percentile.py
