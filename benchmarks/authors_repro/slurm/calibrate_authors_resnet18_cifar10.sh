#!/bin/bash
# CALIBRATION, not a result: the authors' own train.py for 5 epochs, to measure seconds-per-epoch
# before asking for the 200-epoch walltime. Everything else is their AdaFisherCNN.yaml verbatim
# (ResNet18 / CIFAR-10 / AdaFisher / lr 1e-3 / batch 256 / cosine T_max=200), so this also proves
# the whole pipeline — wheelhouse environment, staged CIFAR-10, asdl stub, GPU — end to end.
#
# Submit from the repository root:
#   sbatch benchmarks/authors_repro/slurm/calibrate_authors_resnet18_cifar10.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=authors_r18_c10_calib
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=benchmarks/authors_repro/slurm/logs/%x-%j.out

# --time: ESTIMATED. ~4 min to build the environment from the wheelhouse, ~1 min to stage
# CIFAR-10, then 5 epochs of ResNet18 at batch 256 on a 1/7 H100 slice. 40 min is deliberately
# generous; this job exists precisely because the per-epoch cost is not yet known.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash benchmarks/authors_repro/slurm/run_authors_train.sh \
  AdaFisherCNN.yaml _calibration_resnet18_adafisher 5
