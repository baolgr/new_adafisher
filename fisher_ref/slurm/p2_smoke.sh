#!/bin/bash
# Lot 5's GPU smoke: the P2 runner on all four regime-A models, one checkpoint each, N = 2 000,
# four folds (see the --folds note below), the REAL 1 000-step re-warm. Its metric values are NOT results (at that probe count
# everything sits under the noise floor -- plan_exp_lot2.md §5.6); it exists to exercise every
# device path the four real jobs will take, and to measure the per-stage timings they are sized
# from. Submit from the repository root:  sbatch fisher_ref/slurm/p2_smoke.sh
#
# Hand-written, like every job in this directory (fisher_ref/slurm/README.md).

#SBATCH --account=def-msh-ab
#SBATCH --job-name=fisher_ref_p2_smoke
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# ---------------------------------------------------------------------------------------------
# Why a smoke first, again. Lot 2's first P1 job died after 36 s on a device crossing a CPU-only
# laptop could not expose (plan_exp_lot2.md §5.8): with no CUDA locally, every smoke has
# device=cpu, where a host/device mismatch is invisible by construction. Lot 5 adds four new
# device paths -- the fp32 re-warm and its dataloader, the snapshot's cast to fp64 on the host,
# the P1-py hooked pass, and the TEKFAC basis in pass 2 -- so the same insurance applies.
#
# The re-warm is at its real length here (1 000 steps = 10 x TCov). It is cheap (cnn_gn_cifar's
# whole 30-epoch run is 10 530 steps in 46.8 s on a full H100) and it is the one stage whose cost
# on a MIG slice has never been measured.
#
# --folds 4, not 2: `partitions(folds, count)` refuses more balanced partitions than exist, and
# C(2,1)/2 = 1 -- a two-fold smoke cannot produce the 3 partitions asked for and would die on a
# ValueError several minutes in. C(4,2)/2 = 3 does. (Found locally before submission; the real jobs
# use 10 folds / 20 partitions, where C(10,5)/2 = 126, and A1 8 folds where C(8,4)/2 = 35.)
#
# --mem=64G: four folds of a 26 634^2 fp64 accumulator are 22.7 GB, plus the assembled matrix and
# one half pair. The real jobs need far more; this one deliberately does not.
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

for MODEL in vit_micro_cifar cnn_gn_cifar cnn_gn_cifar_bn mlp_ln_mnist; do
  echo "=============================== $MODEL ==============================="
  python -u -m fisher_ref.runners.p2_operational \
    --model "$MODEL" \
    --arm diag \
    --seed 0 \
    --fractions 0.5 \
    --sources type2 empirical \
    --probes 2000 \
    --batch 250 \
    --alphas 1e-4,1e-3,1e-2,1e-1,1 \
    --folds 4 \
    --partitions 3 \
    --fold-sources type2 empirical \
    --noise-partitions 0 \
    --stein-max-p 4096 \
    --rho-max-p 30000 \
    --noise-rho-max-p 8192 \
    --noise-decomposition-max-p 8192 \
    --rewarm-steps 1000 \
    --rewarm-workers 4 \
    --variants replica clean \
    --variants-at 0.5 \
    --device cuda \
    --data-root "$SLURM_SUBMIT_DIR/dataset" \
    --outputs-root "$SLURM_SUBMIT_DIR/benchmarks/outputs" \
    --out-dir "$SLURM_SUBMIT_DIR/fisher_ref/outputs/p2_smoke"
done
