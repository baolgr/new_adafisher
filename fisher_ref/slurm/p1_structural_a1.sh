#!/bin/bash
# P1 on A1: every structure of the zoo against the exact references, at the five checkpoints of
# mlp_ln_mnist/diag. Lot 2 step 6 of docs/reports/plan_exp_lot2.md §3; the campaign plan is
# plan_exp_draft.md §9 (lot 2) and §13 (the output format).
# Submit from the repository root:  sbatch fisher_ref/slurm/p1_structural_a1.sh
#
# Hand-written, like dense_reference_a1.sh: benchmarks/slurm/generate_jobs.py enumerates the
# *training* benchmarks, and benchmarks/ is read-only for this campaign (plan_exp_draft.md §8).

#SBATCH --account=def-msh-ab
#SBATCH --job-name=fisher_ref_p1_a1
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# ---------------------------------------------------------------------------------------------
# Sizing, from dense_reference_a1's measurements (plan_exp_lot1.md §6.1) rather than extrapolated.
# A1 is P = 26 634; a type-2 reference over N = 55 000 probes costs 147 s, an empirical one 15 s,
# and a half-size one 74 s.
#
#   per fraction, per source:  3 traversals of the probes, not 1 --
#                                (1) the dense reference, (2) the factor accumulation,
#                                (3) EKFAC's second pass, which cannot start until Q_A and Q_G
#                                    exist (approx/ekfac.py's header).
#                              type-2 3 x 147 = 441 s, empirical 3 x 15 = 45 s.
#   M1 / M7 per fraction:      the rearrangement of the 25 120^2 block is a 5.05 GB reshape and a
#                              thin SVD of (1024, 616225); ~3 min both sources.
#   noise floor:               2 x 20 half-size builds = 2 944 s. Measured at ONE fraction only
#                              (--noise-at 1), because it is a property of the estimator at that
#                              theta and 5 of them would cost 4 h on their own.
#   5 fractions x (441 + 45 + 180) + 2 944  ~=  1.9 h.  --time=03:00:00 is that plus headroom, and
#   is an ESTIMATE: the first COMPLETED run replaces it, as benchmarks/slurm/README.md requires.
#
#   --gpus=h100_1g.10gb:1  The accumulator is still the only P x P tensor on the device; the
#                          dense_reference_a1 run measured 6.06 GB of the 10 GB slice at this N.
#   --mem=96G              Host side. Peak is the reference (5.68 GB) plus the rearranged block
#                          (5.05 GB) plus, during the floor, a second reference -- ~20 GB expected.
#                          96G is first-run headroom; lower it once the log reports MaxRSS (the
#                          dense job asked 64G and used 25.3).
#   --cpus-per-task=8      Unlike `syevd`, which lot 1 measured at ~20 GFLOP/s whatever the thread
#                          count, M3's `potrf` and M7's `gesdd` are blocked BLAS3 algorithms that do
#                          thread. Unverified at this scale -- the job prints its own per-fraction
#                          timings, so the next submission can be sized on them.
#
# --stein-max-p 4096: M3 needs a Cholesky of the block per (structure, lambda). At A1's first layer
# that is 25 120^3/3 flops and a 5 GB dense K, thirty times over per source -- it would dominate
# everything else by an order of magnitude. Skipped there and *recorded as skipped* (a
# `stein_kl_skipped_P` row), never silently absent. The other four blocks are 1 056 or smaller and
# cost nothing.
#
# Prerequisites, both already on rorqual:/home/blgr/new_adafisher --
#   dataset/MNIST
#   benchmarks/outputs/mnist/mlp_ln_mnist/diag/ckpt_{0,0.01,0.1,0.5,1}.pt
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# "${VAR:-default}" everywhere, so `sbatch --export=ALL,P1_PROBES=20000 ...` actually takes effect
# rather than being silently overwritten (the trap dense_reference_a1.sh documents).
P1_MODEL="${P1_MODEL:-mlp_ln_mnist}"
P1_ARM="${P1_ARM:-diag}"
P1_SEED="${P1_SEED:-0}"
P1_FRACTIONS="${P1_FRACTIONS:-0,0.01,0.1,0.5,1}"
P1_PROBES="${P1_PROBES:-55000}"
P1_BATCH="${P1_BATCH:-512}"
P1_ALPHAS="${P1_ALPHAS:-1e-4,1e-3,1e-2,1e-1,1}"
P1_NOISE_PARTITIONS="${P1_NOISE_PARTITIONS:-20}"
P1_NOISE_AT="${P1_NOISE_AT:-1}"
P1_STEIN_MAX_P="${P1_STEIN_MAX_P:-4096}"

python -u -m fisher_ref.runners.p1_structural \
  --model "$P1_MODEL" \
  --arm "$P1_ARM" \
  --seed "$P1_SEED" \
  --fractions "$P1_FRACTIONS" \
  --sources type2 empirical \
  --probes "$P1_PROBES" \
  --batch "$P1_BATCH" \
  --alphas "$P1_ALPHAS" \
  --noise-partitions "$P1_NOISE_PARTITIONS" \
  --noise-at "$P1_NOISE_AT" \
  --stein-max-p "$P1_STEIN_MAX_P" \
  --device cuda \
  --data-root "$SLURM_SUBMIT_DIR/dataset" \
  --outputs-root "$SLURM_SUBMIT_DIR/benchmarks/outputs" \
  --out-dir "$SLURM_SUBMIT_DIR/fisher_ref/outputs"
