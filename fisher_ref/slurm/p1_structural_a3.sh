#!/bin/bash
# Lot 3 P1 on A3 (vit_micro_cifar): every structure of the zoo against the exact references over the
# WHOLE model — pos_embed included, as a raw parameter (plan_exp_lot3.md §0.6) — the weight-sharing
# decomposition on the ten token-wise layers, and per-layer fold intervals, at the five checkpoints
# of vit_micro_cifar/diag seed 0. docs/reports/plan_exp_lot3.md §0.11; plan_exp_draft.md §9, §13.
# Submit from the repository root:  sbatch fisher_ref/slurm/p1_structural_a3.sh
#
# Hand-written, like every job in this directory (fisher_ref/slurm/README.md).

#SBATCH --account=def-msh-ab
#SBATCH --job-name=fisher_ref_p1_a3
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# ---------------------------------------------------------------------------------------------
# Sizing. P = 21 098 (19 050 in hooked modules + 2 048 in pos_embed); T = 64 on every token-wise
# layer and LayerNorm; the widest block is qkv at 3 168.
#
#   device    one P x P fp64 accumulator (3.56 GB) and a micro-batch of 250 images in fp64 with its
#             graph retained over the C = 10 columns. B^exp is small here (<= 21 MB per layer).
#   host      ten P x P folds (35.6 GB) + the assembled full matrix (3.6 GB) + a half pair during the
#             intervals (7.1 GB) -> ~50 GB peak. 96G is headroom.
#   cpus      17 blocks x (M3 on every block <= 4 096 wide, M5 Cholesky, M7) x 2 sources, and the
#             40 half analyses — all host-side BLAS3.
#   time      MEASURED from the lot-3 smoke (job 21276016, same slice, N = 2 000, 2 folds): pass 1
#             type-2 21.2 s -> ~8 min at N = 45 000, the per-block metrics 34.5 s per source
#             (N-independent), the fold intervals 11.5 s per partition -> ~5 min for 20. ~17 min per
#             checkpoint, ~1.5 h for five. --time=04:00:00 is that plus headroom.
#
# Prerequisites, all on rorqual:/home/blgr/new_adafisher --
#   dataset/cifar-10-batches-py
#   benchmarks/outputs/cifar10/vit_micro_cifar/diag/ckpt_{0,0.01,0.1,0.5,1}.pt
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

P1_FRACTIONS="${P1_FRACTIONS:-0,1,0.5,0.1,0.01}"

python -u -m fisher_ref.runners.p1_structural \
  --model vit_micro_cifar \
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
  --raw-parameters auto \
  --noise-partitions 0 \
  --stein-max-p 4096 \
  --rho-max-p 30000 \
  --noise-rho-max-p 8192 \
  --noise-decomposition-max-p 8192 \
  --device cuda \
  --data-root "$SLURM_SUBMIT_DIR/dataset" \
  --outputs-root "$SLURM_SUBMIT_DIR/benchmarks/outputs" \
  --out-dir "$SLURM_SUBMIT_DIR/fisher_ref/outputs"
