#!/bin/bash
# Lot 5's exit criterion 5: is a re-warm of 10 x TCov steps enough on THIS lot's four models?
# plan_exp_draft.md §3.2 answered that once, on cnn_gn_cifar alone, from a 2 000-step training run
# of its own; lot 5 reads real checkpoints on four models, so it is re-measured where it is used
# (docs/reports/plan_exp_lot5.md §0.8). Submit from the repository root:
#   sbatch fisher_ref/slurm/rewarm_fidelity_lot5.sh
#
# Hand-written, like every job in this directory (fisher_ref/slurm/README.md).

#SBATCH --account=def-msh-ab
#SBATCH --job-name=fisher_ref_rewarm_lot5
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# ---------------------------------------------------------------------------------------------
# Sizing. This job trains; it never builds a reference, so P plays no part.
#
#   work    4 models x 5 modes x (one 30 x TCov reference warm + three k x TCov re-warms at
#           k in {3, 10, 20}) = 4 x 5 x (3 000 + 3 x 3 300) = 258 000 optimizer steps.
#           DERIVED from a measured rate: cnn_gn_cifar/diag's own 30-epoch run is 10 530 steps in
#           46.8 s on a full H100 (its manifest), so ~4.4 ms/step there; on a 1g.10gb slice with
#           the four Kronecker modes costing 1-2x more per step, budget ~30 ms/step -> ~2.2 h.
#   mem     32G: one fp32 network and one dataloader batch at a time, plus the fp64 snapshots,
#           which are factors (at most 785^2 on A1), not P x P matrices.
#   gpus    the same 1g.10gb slice: this is training-shaped work, not analysis-shaped.
#
# It should be submitted FIRST. Its verdict either confirms the 1 000-step re-warm the four P2
# jobs use, or names the model where it is not enough -- in which case that model's P2 job is
# re-run at the length that is, and the first numbers are reported as superseded.
#
# Prerequisites, on rorqual:/home/blgr/new_adafisher --
#   dataset/{cifar-10-batches-py,MNIST}
#   benchmarks/outputs/{cifar10,mnist}/<model>/diag/ckpt_0.5.pt
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

export P2F_MODELS="${P2F_MODELS:-cnn_gn_cifar,cnn_gn_cifar_bn,vit_micro_cifar,mlp_ln_mnist}"
export P2F_FRACTION="${P2F_FRACTION:-0.5}"
export P2F_KS="${P2F_KS:-3,10,20}"
export P2F_REF_K="${P2F_REF_K:-30}"
export P2F_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export P2F_OUTPUTS_ROOT="$SLURM_SUBMIT_DIR/benchmarks/outputs"
export P2F_DEVICE=cuda
export P2F_WORKERS=4
export P2F_OUT="$SLURM_SUBMIT_DIR/fisher_ref/outputs/rewarm_fidelity_lot5.json"

python -u -m fisher_ref.experiments.rewarm_fidelity_lot5
