#!/bin/bash
# P1's missing block: M5 (and M1/M7, rebuilt) on A1's first layer, the one `p1_structural_a1.sh`
# could not reach. Companion to that job, not a replacement -- see the OUT-DIR note below.
# Submit from the repository root:  sbatch fisher_ref/slurm/p1_features0_a1.sh
#
# Hand-written, like its two predecessors: benchmarks/slurm/generate_jobs.py enumerates the
# *training* benchmarks, and benchmarks/ is read-only for this campaign (plan_exp_draft.md §8).

#SBATCH --account=def-msh-ab
#SBATCH --job-name=fisher_ref_p1_f0_a1
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# ---------------------------------------------------------------------------------------------
# Why this job exists.
#
# `features.0` is 25 120 of A1's 26 634 parameters and carries 63 % of tr(F). Job 21128864 reported
# M1 and M7 on it but **no M5 and no M3**: `analyse_layer` had one size gate, `--stein-max-p 4096`,
# and it `continue`d past both metrics. M3's cost justifies that gate; M5's does not.
#
#   M3 per (structure, lambda):  densify K (5.05 GB) + solve R_lam X = K_lam with a P x P RHS
#                                = 2 P^3 = 3.2e13 flops, five live P x P buffers. Thirty times
#                                over per source: ~9.5e14 flops. Still skipped here.
#   M5 per lambda:               ONE Cholesky of R_lam (P^3/3 = 5.3e12) shared by every structure,
#                                then vector solves. Five per source: ~2.6e13 flops, one live
#                                factor. That is 36x less work than M3, hence a separate gate
#                                (`--rho-max-p`, added for this job; it defaults to --stein-max-p,
#                                so job 21128864 stays reproducible bit-for-bit).
#
# Sizing, MEASURED rather than extrapolated:
#   Cholesky, P = 25 120, fp64:  30 s (timed at P = 8 192 and 16 384, 156-176 GFLOP/s, scaled by
#                                P^3; both extrapolations agree to 12 %). 5 alphas x 2 sources
#                                = 10 per fraction = ~300 s.
#   base per fraction:           job 21128864 measured 474 s for the *whole* model, both sources,
#                                all five blocks. This job builds a one-module reference (P = 25 120
#                                instead of 26 634) and re-does M1/M7 on it: ~400 s.
#   5 fractions x ~700 s      ~= 1 h.  --time=02:30:00 is that plus headroom.
#
#   --mem=48G              Host side, where every Cholesky runs (the runner moves the reference to
#                          the CPU). Peak is the reference (5.05 GB) + the rearrangement M7 needs
#                          (5.05 GB, transient) + the Cholesky's clone and factor (5.05 each)
#                          ~= 20-25 GB. Job 21128864 measured MaxRSS 27.5 GB *including* the noise
#                          floor's two extra half-references, which this job does not build.
#   --cpus-per-task=16     The Cholesky is `potrf`, blocked BLAS3, and it threads -- unlike the
#                          `syevd` of plan_exp_lot1.md §6.1, which does not. This is the one job in
#                          the campaign where asking for cores actually buys time.
#   --gpus=h100_1g.10gb:1  Unchanged: only the reference/factor accumulation touches the device,
#                          and a 25 120^2 fp64 accumulator is 5.05 GB of the 10 GB slice (lot 1
#                          measured 6.06 GB peak at the larger P = 26 634).
#
# --noise-partitions 0: the floor is a property of the whole-matrix reference and job 21128864
# already measured it at fraction 1.0 (0.3431, CI [0.3323, 0.3520]). A *per-block* floor is a
# different statistic and a separate job; do not read this job's per-block errors against that
# whole-matrix number.
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# "${VAR:-default}" everywhere, so `sbatch --export=ALL,P1_PROBES=20000 ...` takes effect rather
# than being silently overwritten (the trap dense_reference_a1.sh documents).
P1_MODEL="${P1_MODEL:-mlp_ln_mnist}"
P1_ARM="${P1_ARM:-diag}"
P1_SEED="${P1_SEED:-0}"
P1_FRACTIONS="${P1_FRACTIONS:-0,0.01,0.1,0.5,1}"
P1_PROBES="${P1_PROBES:-55000}"
P1_BATCH="${P1_BATCH:-512}"
P1_ALPHAS="${P1_ALPHAS:-1e-4,1e-3,1e-2,1e-1,1}"

# OUT-DIR: deliberately NOT fisher_ref/outputs. `_write_csv` opens metrics.csv with "w", so writing
# a one-module run into job 21128864's tree would replace 1 200 rows per fraction with ~120 and
# destroy the only copy of the five-block result. The two trees are merged at analysis time.
P1_OUT="${P1_OUT:-$SLURM_SUBMIT_DIR/fisher_ref/outputs/features0}"

python -u -m fisher_ref.runners.p1_structural \
  --model "$P1_MODEL" \
  --arm "$P1_ARM" \
  --seed "$P1_SEED" \
  --fractions "$P1_FRACTIONS" \
  --sources type2 empirical \
  --probes "$P1_PROBES" \
  --batch "$P1_BATCH" \
  --alphas "$P1_ALPHAS" \
  --modules features.0 \
  --noise-partitions 0 \
  --stein-max-p 4096 \
  --rho-max-p 30000 \
  --device cuda \
  --data-root "$SLURM_SUBMIT_DIR/dataset" \
  --outputs-root "$SLURM_SUBMIT_DIR/benchmarks/outputs" \
  --out-dir "$P1_OUT"
