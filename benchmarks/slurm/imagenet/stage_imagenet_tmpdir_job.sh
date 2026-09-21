#!/bin/bash
# Build both ImageNet ImageFolder trees and export them as two tars, doing all of the work in
# node-local $SLURM_TMPDIR. Hand-written, like stage_imagenet_job.sh beside it.
#
# WHY THIS EXISTS, rather than stage_imagenet_job.sh's two dependent jobs writing to a persistent
# OUT_DIR: the extracted tree is 1 281 167 train + 50 000 val files, and /scratch on this cluster
# has a **1 000 000 file quota** (`diskusage_report`, measured 2026-09-20: 4/1000K used). The full
# tree alone is 33% over it, and imagenet32 would double the count. /project/def-msh-ab is worse
# (316K of 500K files already used). $SLURM_TMPDIR has no inode quota and 375 GB of local NVMe,
# and the generated training jobs already expect a *tar*, not a tree -- they expand
# $IMAGENET_ARCHIVE into their own $SLURM_TMPDIR. So the tree never needs to persist: only the
# two tars do, which are 2 files.
#
# Peak $SLURM_TMPDIR use, measured against the tar sizes: ~140 GB for the train tree (the loop
# removes each per-wnid tar right after extracting it), +6 GB val, +6 GB imagenet32, +5 GB venv
# = ~160 GB of 375 GB. Both output tars are written straight to $ARCHIVE_DIR, never staged
# locally, which is what keeps that number flat.
#
# The whole node (192 cores) is requested on purpose: stage_imagenet.sh's resize step builds a
# bare `ProcessPoolExecutor()`, which sizes itself from os.cpu_count() -- the NODE's core count,
# not the cgroup's. On a 32-core allocation that is a 6x oversubscription. Asking for all 192
# cores makes the two numbers agree instead of editing the script.
#
# Fetch valprep.sh on the LOGIN node first (compute nodes have no internet):
#   wget -O "$HOME/valprep.sh" \
#     https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
#
# Usage, from the repository root on the login node:
#   sbatch --export=ALL,TAR_DIR=$SCRATCH/imagenet_tars,ARCHIVE_DIR=$SCRATCH,VALPREP_SCRIPT=$HOME/valprep.sh \
#          benchmarks/slurm/imagenet/stage_imagenet_tmpdir_job.sh
#
# Produces $ARCHIVE_DIR/imagenet32.tar (the four 32 px benches) and $ARCHIVE_DIR/imagenet.tar
# (resnet50_imagenet, vit_small_imagenet). imagenet32.tar is written FIRST, so a job that dies
# in the long final tar still leaves the four 32 px benches runnable.
#
# --time is a guess, not a measured figure: there is no calibration job for a one-off staging
# step. The pieces that dominate it are tar I/O, not compute.

#SBATCH --account=def-msh-ab
#SBATCH --job-name=stage_imagenet_tmpdir
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=benchmarks/slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TAR_DIR="${TAR_DIR:?set TAR_DIR to the directory holding the two ILSVRC2012 tars}"
ARCHIVE_DIR="${ARCHIVE_DIR:?set ARCHIVE_DIR to where the two output tars go, e.g. \$SCRATCH}"
VALPREP_SCRIPT="${VALPREP_SCRIPT:?set VALPREP_SCRIPT to a copy fetched on the login node}"
export VALPREP_SCRIPT

for f in ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar; do
  [ -f "$TAR_DIR/$f" ] || { echo "missing $TAR_DIR/$f" >&2; exit 1; }
done
[ -f "$VALPREP_SCRIPT" ] || { echo "missing $VALPREP_SCRIPT" >&2; exit 1; }
mkdir -p "$ARCHIVE_DIR"

WORK="$SLURM_TMPDIR/staging"
mkdir -p "$WORK"
echo "== local scratch before =="; df -h "$SLURM_TMPDIR"

module purge
module load StdEnv/2023 gcc/13.3 python/3.11.5

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements-cluster.txt

echo "== [1/4] full-resolution tree =="
time benchmarks/slurm/imagenet/stage_imagenet.sh full "$TAR_DIR" "$WORK"
df -h "$SLURM_TMPDIR"

echo "== [2/4] 32 px tree =="
time benchmarks/slurm/imagenet/stage_imagenet.sh resize "$WORK" 32
df -h "$SLURM_TMPDIR"

echo "== [3/4] imagenet32.tar -> $ARCHIVE_DIR =="
time tar -cf "$ARCHIVE_DIR/imagenet32.tar" -C "$WORK" imagenet32
ls -l "$ARCHIVE_DIR/imagenet32.tar"

echo "== [4/4] imagenet.tar -> $ARCHIVE_DIR =="
time tar -cf "$ARCHIVE_DIR/imagenet.tar" -C "$WORK" imagenet
ls -l "$ARCHIVE_DIR/imagenet.tar"

echo "== counts (sanity: 1000 train wnids, 1000 val wnids) =="
echo "train wnids: $(find "$WORK/imagenet/train" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "val   wnids: $(find "$WORK/imagenet/val"   -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "train JPEGs: $(find "$WORK/imagenet/train" -name '*.JPEG' | wc -l)"
echo "val   JPEGs: $(find "$WORK/imagenet/val"   -name '*.JPEG' | wc -l)"
echo "done"
