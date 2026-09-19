#!/bin/bash
# Lot 3 P1 on A2 (cnn_gn_cifar, GroupNorm): every structure of the zoo against the exact references,
# the weight-sharing decomposition and per-layer fold intervals, at the five checkpoints of
# cnn_gn_cifar/diag seed 0. docs/reports/plan_exp_lot3.md §0.11; plan_exp_draft.md §9 (lot 3), §13.
# Submit from the repository root:  sbatch fisher_ref/slurm/p1_structural_a2_gn.sh
#
# Hand-written, like every job in this directory (fisher_ref/slurm/README.md).

#SBATCH --account=def-msh-ab
#SBATCH --job-name=fisher_ref_p1_a2_gn
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=09:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# ---------------------------------------------------------------------------------------------
# Sizing. P = 24 458; T = 1 024 / 256 / 64 on the three convolutions.
#
#   device    one P x P fp64 accumulator (4.79 GB), a micro-batch of 250 CIFAR images in fp64 with
#             its graph retained over the C = 10 backprop columns, and the B^exp chunks, bounded at
#             256 MB by SharingSums.consume.
#   host      ten P x P folds (47.9 GB) + ten half-vectorised B^exp of the last convolution (7.0 GB)
#             + the assembled full matrix (4.8 GB) + a half pair during the intervals (9.6 GB) + the
#             2.7 GB rearrangements of the last convolution -> ~77 GB peak. 128G is headroom.
#   cpus      the per-block metrics (Cholesky for M5 on the 18 496-wide block, Gram
#             eigendecompositions for M7) and the 40 half analyses are blocked BLAS3 on the host.
#   time      MEASURED from the lot-3 smoke (job 21276016, same slice, N = 2 000, 2 folds), scaled
#             by the probe count where the cost is linear in it: pass 1 type-2 107.6 s -> ~40 min at
#             N = 45 000, pass 2 ~1 min, the empirical source ~4.5 min, the per-block metrics 44 s
#             per source (N-independent, host-side), the fold intervals 24 s per partition -> ~10 min
#             for 20 partitions with ten folds to sum instead of two. ~60 min per checkpoint, ~5 h
#             for five. --time=09:00:00 is that plus the usual headroom.
#
# --probes 45000 --batch 250 --folds 10: the whole train split of the run's seeded partition, in ten
# folds of 4 500 probes (FoldPlan refuses unequal folds). --fractions starts with the two endpoints,
# so a timeout still leaves the most informative checkpoints written (every source is written as
# soon as it is done).
#
# Prerequisites, all on rorqual:/home/blgr/new_adafisher --
#   dataset/cifar-10-batches-py
#   benchmarks/outputs/cifar10/cnn_gn_cifar/diag/ckpt_{0,0.01,0.1,0.5,1}.pt
# ---------------------------------------------------------------------------------------------

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023 gcc/13.3 cuda/12.6 python/3.11.5

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements-cluster.txt

export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

P1_MODEL="${P1_MODEL:-cnn_gn_cifar}"
P1_FRACTIONS="${P1_FRACTIONS:-0,1,0.5,0.1,0.01}"

python -u -m fisher_ref.runners.p1_structural \
  --model "$P1_MODEL" \
  --arm diag \
  --seed 0 \
  --fractions "$P1_FRACTIONS" \
  --sources type2 empirical \
  --probes 45000 \
  --batch 250 \
  --alphas 1e-4,1e-3,1e-2,1e-1,1 \
  --folds 10 \
  --partitions 20 \
  --fold-sources type2 \
  --noise-partitions 0 \
  --stein-max-p 4096 \
  --rho-max-p 30000 \
  --noise-rho-max-p 8192 \
  --noise-decomposition-max-p 8192 \
  --device cuda \
  --data-root "$SLURM_SUBMIT_DIR/dataset" \
  --outputs-root "$SLURM_SUBMIT_DIR/benchmarks/outputs" \
  --out-dir "$SLURM_SUBMIT_DIR/fisher_ref/outputs"
