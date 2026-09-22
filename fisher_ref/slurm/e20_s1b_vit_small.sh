#!/bin/bash
# One job of E20 (docs/reports/plan_e20_s1b_vit_small.md): one (seed, job) of stage 1 on ViT-S.
# Submitted by fisher_ref/slurm/e20_submit.sh, which sets E20_SEED, E20_JOB and --time; see there.
#
# Hand-written, NOT generated: benchmarks/slurm/generate_jobs.py enumerates *training* benchmarks.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=e20_s1b_vit_small
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=fisher_ref/slurm/logs/%x-%j.out

# --gpus, --mem: E15's (fisher_ref/slurm/e15_layer_damping.sh), whose cell builder these runs use
# and whose bridge cell must be reproduced on the same kind of slice. ViT-S at batch 32 is small
# (2.7 M parameters, 65 tokens of width 192).

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
export WARMUP_SGD_DATA_ROOT="$SLURM_SUBMIT_DIR/dataset"
export WARMUP_SGD_EPOCHS=15
export WARMUP_SGD_NUM_WORKERS=4
export WARMUP_SGD_DEVICE=auto

python fisher_ref/experiments/e20_s1b_vit_small.py
