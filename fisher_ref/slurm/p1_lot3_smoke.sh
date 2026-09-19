#!/bin/bash
# Lot 3's GPU smoke: the P1 runner on A2 (GroupNorm), A2 (BatchNorm-eval) and A3, one checkpoint
# each, at a small probe count — before the three full jobs are sized and submitted.
# docs/reports/plan_exp_lot3.md §0.11, §3 step 6. Submit from the repository root:
#   sbatch fisher_ref/slurm/p1_lot3_smoke.sh
#
# Hand-written, like every job in this directory: benchmarks/slurm/generate_jobs.py enumerates the
# *training* benchmarks, and benchmarks/ is read-only for this campaign (plan_exp_draft.md §8).

#SBATCH --account=def-msh-ab
#SBATCH --job-name=fisher_ref_p1_lot3_smoke
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# ---------------------------------------------------------------------------------------------
# Why this job exists.
#
# Lot 2's first cluster job died after 36 s on a device crossing the CPU-only laptop could not have
# exposed (plan_exp_lot2.md §5.8). Lot 3 adds more of exactly that surface: per-fold host offloads,
# eigenbases moved to the device for the second pass, raw-parameter leaves, the row checks' extra
# backwards, a diag.py reading hooked on the GPU. None of it runs on a GPU locally. This job runs
# every one of those paths on the real models, on the same 10 GB slice the A3 job will use, and
# prints the per-stage timings the full jobs are sized from.
#
# Sizing (the dense matrix is P x P whatever N is, so the host side is set by P, not by the probes):
#   --folds 2               two P x P folds on the host: 2 x 4.79 GB for A2 (P = 24 458), plus the
#                           assembled full matrix, a half pair, and the 2.7 GB rearrangements of
#                           A2's last convolution -> ~25 GB peak. 48G is headroom.
#   --probes 2000           divisible by folds x batch = 2 x 100 (FoldPlan refuses unequal folds).
#   --gpus h100_1g.10gb     one P x P fp64 accumulator (4.79 GB) + a micro-batch of 100 CIFAR images
#                           through the network in fp64 + the bounded B^exp chunks (256 MB).
#   --time 01:00:00         three models x one checkpoint x two sources at N = 2 000: the traversals
#                           are seconds; the CPU metrics on A2's 18 496-wide block dominate
#                           (a few Cholesky and Gram eigendecompositions). An ESTIMATE.
#
# Output goes to fisher_ref/outputs/_smoke_lot3/, never to the real tree.
# Prerequisites, all on rorqual:/home/blgr/new_adafisher --
#   dataset/cifar-10-batches-py
#   benchmarks/outputs/cifar10/{cnn_gn_cifar,cnn_gn_cifar_bn,vit_micro_cifar}/diag/ckpt_*.pt
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
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

for MODEL in cnn_gn_cifar cnn_gn_cifar_bn vit_micro_cifar; do
  python -u -m fisher_ref.runners.p1_structural \
    --model "$MODEL" \
    --arm diag \
    --seed 0 \
    --fractions 0.5 \
    --sources type2 empirical \
    --probes 2000 \
    --batch 100 \
    --alphas 1e-4,1e-3,1e-2,1e-1,1 \
    --folds 2 \
    --partitions 1 \
    --noise-partitions 0 \
    --stein-max-p 4096 \
    --rho-max-p 30000 \
    --device cuda \
    --data-root "$SLURM_SUBMIT_DIR/dataset" \
    --outputs-root "$SLURM_SUBMIT_DIR/benchmarks/outputs" \
    --out-dir "$SLURM_SUBMIT_DIR/fisher_ref/outputs/_smoke_lot3"
done
