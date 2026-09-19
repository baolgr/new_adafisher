#!/bin/bash
# Lot 5's P2 on A3 (vit_micro_cifar): the operational preconditioner of all five Fisher modes against the exact
# references, at the five checkpoints of vit_micro_cifar/diag seed 0, alongside the P1 and P1-py rungs it
# has to be compared with. docs/reports/plan_exp_lot5.md §3.5; plan_exp_draft.md §9 (lot 5), §13.
# Submit from the repository root:  sbatch fisher_ref/slurm/p2_operational_a3.sh
#
# Hand-written, like every job in this directory (fisher_ref/slurm/README.md).

#SBATCH --account=def-msh-ab
#SBATCH --job-name=fisher_ref_p2_a3
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=09:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# ---------------------------------------------------------------------------------------------
# Sizing. P = 21 098.
#
#   device    one P x P fp64 accumulator (3.6 GB) during the reference build, and -- separately in
#             time, never at once -- the fp32 re-warm's network and dataloader batch, which is the
#             size of one training step. The re-warm is torn down and the cache emptied before the
#             traversal starts.
#   host      10 P x P folds (35.6 GB) + the assembled matrix + a half pair during the
#             intervals + the per-block rearrangements. lot 3 measured 54.3 GB on this model; the extras add per-structure scalars, not buffers.
#   time      lot 3 measured ~12 min per checkpoint on this model at this probe count (job 21276896).
#             Lot 5 adds, per checkpoint: five re-warms of 1 000 steps (~5 min), one P1-py hooked
#             pass (~1 min), and about three times the structures in M1 and in the fold intervals,
#             whose 24 s per partition becomes ~60 s (+12 min at 20 partitions). M3 is NOT computed
#             for the new structures (plan_exp_lot5.md §0.13), which is what keeps this near lot 3's
#             cost instead of three times it. ~45 min per checkpoint, ~3.8 h for five; --time=09:00:00 is that plus headroom.
#             Three costs were added after the naive audit and are already in the figure above:
#             --fold-sources now includes `empirical`, i.e. one more 20-partition interval loop per
#             checkpoint (+~20 min, the loop being N-independent and host-side); the `replica` and
#             `clean` variants triple fraction 1's re-warms (15 instead of 5) and add ~20 structures
#             to its M1 and its partitions; and the P1-py rung now emits a one-micro-batch control
#             as well (plan_exp_lot5.md §0.4c).
#
# --probes/--batch/--folds are lot 3's own, deliberately: the recomputed P1 rows can then be checked against
# the published ones, a free regression test on the whole reuse. FoldPlan refuses unequal
# folds, so the probe count must divide folds x batch (45 000 / 250 / 10 does).
#
# --fractions starts with the two endpoints, so a timeout still leaves the most informative
# checkpoints written; every source is written to disk as soon as it is done.
#
# --fold-sources type2 empirical, not type2 alone: the P2 operator and the P1-py readings are all
# built from the EMPIRICAL, true-label, batch-mean gradient, so `E_hat` is their natural reference
# and is the one HF7's verdict is read against (plan_exp_lot5.md §0.9). Against `F` the P1 -> P1-py
# step would carry the whole type-2 -> empirical source change on top of the formula change it is
# meant to isolate -- lot 1 measured that source gap at 1.42 relative on A1, which is not a
# correction term. So the per-layer noise floors and the fold intervals have to exist on the
# empirical rows too, which is what this costs: one more 20-partition interval loop per checkpoint
# (~20 min, the loop being N-independent and host-side), already in the --time below.
#
# --variants-at 1: the two diagnostics of plan_exp_lot5.md §0.4b -- a second re-warm on a different
# batch order (the operator's OWN draw-to-draw spread, which the fold intervals do not contain) and
# one fed the clean probe tensors in eval mode (the augmentation and train-mode term) -- run at one
# checkpoint only, in the spirit of lot 2's --noise-at 1. The replica is not a nicety: HF7's rule
# **gates on it** (plan_exp_lot5.md §0.9), because a P2 ranking that does not reproduce across two
# draws of the same estimator cannot be said to differ from anything. It also triples that
# checkpoint's re-warms (15 instead of 5) and adds ~20 structures to its M1 and to its 20 fold
# partitions, which is why --fractions puts it first, where the budget is least likely to run out.
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

P2_MODEL="${P2_MODEL:-vit_micro_cifar}"
P2_FRACTIONS="${P2_FRACTIONS:-1,0,0.5,0.1,0.01}"

python -u -m fisher_ref.runners.p2_operational \
  --model "$P2_MODEL" \
  --arm diag \
  --seed 0 \
  --fractions "$P2_FRACTIONS" \
  --sources type2 empirical \
  --probes 45000 \
  --batch 250 \
  --alphas 1e-4,1e-3,1e-2,1e-1,1 \
  --folds 10 \
  --partitions 20 \
  --fold-sources type2 empirical \
  --noise-partitions 0 \
  --stein-max-p 4096 \
  --rho-max-p 30000 \
  --noise-rho-max-p 8192 \
  --noise-decomposition-max-p 8192 \
  --modes diag kfac ekfac tkfac tekfac \
  --rewarm-steps 1000 \
  --rewarm-workers 4 \
  --variants replica clean \
  --variants-at 1 \
  --device cuda \
  --data-root "$SLURM_SUBMIT_DIR/dataset" \
  --outputs-root "$SLURM_SUBMIT_DIR/benchmarks/outputs" \
  --out-dir "$SLURM_SUBMIT_DIR/fisher_ref/outputs/p2"
