#!/bin/bash
# Hand-written wrapper around stage_imagenet.sh, to run it as a batch job instead of interactively
# on the login node: untarring 1.28M files and resizing them with a multiprocess pool is CPU/IO
# work that belongs on a compute node, not the shared login node. No GPU needed.
#
# Compute nodes have no internet (see benchmarks/slurm/README.md) — the val split's reorganisation
# step needs valprep.sh, so fetch it once on the login node BEFORE submitting MODE=full:
#   wget -O "$HOME/valprep.sh" \
#     https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
#
# Usage (from the repository root, on the login node), two dependent jobs:
#   J1=$(sbatch --parsable --export=ALL,MODE=full,TAR_DIR=/path/to/tars,OUT_DIR=$SCRATCH/imagenet_staging,VALPREP_SCRIPT=$HOME/valprep.sh \
#        benchmarks/slurm/imagenet/stage_imagenet_job.sh)
#   sbatch --dependency=afterok:$J1 --export=ALL,MODE=resize,OUT_DIR=$SCRATCH/imagenet_staging \
#        benchmarks/slurm/imagenet/stage_imagenet_job.sh
#
# --time/--cpus-per-task/--mem below are untested guesses, not measured figures (unlike every
# other --time in this repository) — there is no calibration job for a one-off staging step.
# Override on the sbatch command line (takes precedence over the #SBATCH lines) if your
# allocation's max walltime is shorter, and split MODE=full into a second job if it does not
# finish in time (tar extraction resumes cheaply; already-extracted per-wnid directories are just
# re-skipped work, not corruption).

#SBATCH --account=def-msh-ab
#SBATCH --job-name=stage_imagenet
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=benchmarks/slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

MODE="${MODE:?set MODE=full or MODE=resize}"
OUT_DIR="${OUT_DIR:?set OUT_DIR, e.g. \$SCRATCH/imagenet_staging (needs ~350 GB free)}"

module purge
module load StdEnv/2023 gcc/13.3 python/3.11.5

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements-cluster.txt

if [ "$MODE" = "full" ]; then
  : "${TAR_DIR:?set TAR_DIR to the directory holding the two ILSVRC2012 tars}"
  time benchmarks/slurm/imagenet/stage_imagenet.sh full "$TAR_DIR" "$OUT_DIR"
elif [ "$MODE" = "resize" ]; then
  time benchmarks/slurm/imagenet/stage_imagenet.sh resize "$OUT_DIR" 32
else
  echo "MODE must be 'full' or 'resize'" >&2
  exit 1
fi
