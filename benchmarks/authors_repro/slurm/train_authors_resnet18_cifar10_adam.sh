#!/bin/bash
# Reproduce one published line of the AdaFisher paper's Table 2 with the authors' own code.
#
#   docs/papers/adafisher_2405.16397.pdf, Table 2, CIFAR-10, ResNet18, Adam: 94.85 +/- 0.1
#   (mean, std over trials; batch size 256; 210-epoch schedule, Table 8)
#
# Config: reference_repos/AdaFisher/Image_Classification/configs/adamCNN.yaml, used VERBATIM —
# no override of any kind. It is already exactly the published setting: resnet18Cifar, CIFAR10,
# Adam, lr 1e-3, weight decay 5e-4, cosine annealing T_max=210,
# max_epochs 210, batch 256, aug + cutout (1 hole, 16 px). Table 9 confirms lr 1e-3 for CNNs.
#
# See run_authors_train.sh for what the job does and for the single deviation it makes (the asdl
# import stub, which no Adam code path can reach).
#
# One seed (42, the authors' own default), so the number to compare is one draw from the
# distribution the published mean/std summarises — not a mean over trials.
#
# Submit from the repository root:
#   sbatch benchmarks/authors_repro/slurm/train_authors_resnet18_cifar10_adam.sh

#SBATCH --account=def-msh-ab
#SBATCH --job-name=authors_r18_c10_adam
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:45:00
#SBATCH --output=benchmarks/authors_repro/slurm/logs/%x-%j.out

# --time: MEASURED. The calibration job (21449705, same GPU slice, same config) ran the
# authors' train.py at 19.3 s/epoch (median of its 5 epochs; the first is 21.5 s, warm-up
# included) and spent 49 s on the environment build and the data staging. 210 epochs is
# therefore about 68 min of training, and 01:45:00 leaves roughly 30% margin.
# Adam is not separately calibrated; it does strictly less work per step than AdaFisher, so the
# AdaFisher measurement bounds it.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
bash benchmarks/authors_repro/slurm/run_authors_train.sh \
  adamCNN.yaml resnet18_cifar10_adam
