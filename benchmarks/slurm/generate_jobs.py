"""Generate every SLURM job script under this directory, one per (model, arm) plus the grouped,
calibration, sweep and extra-seed jobs.

    python benchmarks/slurm/generate_jobs.py     # regenerates every .sh here; edit this file only

The model folder list is the registry: this script calls
``benchmarks.common.runner.discover_benchmarks`` and reads each ``Benchmark``'s own nominal epoch
count, arm list, dataset and output group, so adding a model folder adds its jobs with no edit
here — except in :data:`WALLTIME`, which has one required row per model and raises ``KeyError``
without it. The scripts are near-identical apart from the model, the arm and the ``--time``
budget, hence generated rather than hand-maintained.

What gets generated, per model
------------------------------

* ``calibrate_<model>.sh`` — two short epochs of every arm on a 5000-example training subset, to
  turn that model's ``--time`` value from an extrapolation into a measurement.
* ``train_<model>_<arm>.sh`` — one arm per job. The reference arm (``diag``) runs unbudgeted at a
  fixed epoch count; every other arm reads its budget from the ``WCT_BUDGET`` environment
  variable, and falls back to an equal-epoch run when that is unset.
* ``train_<model>_all.sh`` — all seven arms in one job, for the models listed in :data:`GROUPED`.
  This is the preferred form: the runner derives the wall-clock budget in-process, so there is no
  measured number to copy by hand between jobs.
* sweep and extra-seed jobs, from :data:`SWEEPS` and :data:`V0_SEEDS`.

Jobs land in ``benchmarks/slurm/<output_group>/``, matching where their results land
(``benchmarks/outputs/<output_group>/<model>/``).

Every cluster-specific line — account, GPU shape, module load, the no-internet assumption — is
copied from job scripts that have actually run on this cluster rather than re-derived. See
``README.md`` in this directory for the provenance of each.
"""

from __future__ import annotations

import sys
from pathlib import Path

SLURM_DIR = Path(__file__).resolve().parent
REPO_ROOT = SLURM_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.common.runner import discover_benchmarks  # noqa: E402

REFERENCE_ARM = "diag"

# Per-model ``--time`` for one training job and for the calibration job, plus the basis of the
# training figure. **Every row is measured from a completed run**, not extrapolated: the reference
# arm's own ``total_s`` (``benchmarks/outputs/<group>/<model>/[<arm>/]manifest.json``, or its
# ``resnet20_cifar``, whose grouped job died before writing one), on one ``h100_1g.10gb`` MIG slice.
#
# Three measurements bound the margin:
#   * **setup is ~40 s**, not the ~10 min an earlier draft of this table assumed:
#     ``cal_mlp_ln_mnist`` COMPLETED in ``00:00:44`` *including* module load, ``virtualenv``,
#     ``pip install --no-index`` and 7 arms x 2 epochs, and ``mlp_ln_mnist_diag`` in ``00:00:58``
#     for 21.6 s of training. The wheelhouse is node-local, so the install is a disk copy.
#   * **validation adds ~6%** on top of training: ``cct_2_3x2_cifar_all`` ran ``00:30:22`` against
#     1662.5 s of accounted training (eval time is excluded from the budget clock, not from the
#     job's wall-clock).
#   * under the WCT protocol **every arm of a model takes the reference arm's time**, so one
#     ``--time`` per model covers all seven; ``--max-epoch-factor`` only lets a cheap arm fit more
#     epochs into the same wall-clock, never overrun it.
# Each value below is the measured job duration plus roughly 40-60% headroom, rounded up.
WALLTIME = {
    "mnist_autoencoder": ("00:15:00", "00:10:00",
                          "MEASURED. T_diag = 13.7 s; 7 arms = 95.7 s of training, ~2.5 min of"
                          " job. Setup-dominated."),
    "mlp_ln_mnist": ("00:15:00", "00:10:00",
                     "MEASURED. T_diag = 21.6 s; 7 arms = 151 s of training, ~3.5 min of job."),
    "cnn_gn_cifar": ("00:20:00", "00:10:00",
                     "MEASURED. T_diag = 46.0 s; 7 arms = 322 s of training, ~6.5 min of job."),
    "vit_micro_cifar": ("00:25:00", "00:10:00",
                        "MEASURED. T_diag = 67.4 s; 7 arms = 472 s of training, ~9.5 min of job."),
    "resnet20_cifar": ("01:05:00", "00:10:00",
                       "MEASURED. T_diag = 360.3 s (resnet20_cifar_all-20851634.out); 7 arms ="
                       " 42 min of training, ~45 min of job."),
    "cct_2_3x2_cifar": ("00:45:00", "00:10:00",
                        "MEASURED. T_diag = 237.5 s; 7 arms = 27.7 min of training, and the job"
                        " itself ran 00:30:22 end to end."),
    "resnet50_cifar": ("01:15:00", "00:15:00",
                       "MEASURED. T_diag = 2822.8 s = 47.0 min per arm; ~50 min of job with"
                       " validation and setup. This is the PER-ARM figure; the grouped job for"
                       " this model is in GROUPED below and is now the default."),
    "vit_small_cifar": ("00:30:00", "00:10:00",
                        "MEASURED. T_diag = 987.8 s = 16.5 min per arm; ~18 min of job. This is"
                        " the PER-ARM figure; see GROUPED below for the default."),
    # ------------------------------------------------------------------------------------------
    # CIFAR-100. DERIVED, not measured, but on a firm basis: the images are the same 50 000
    # 32x32x3 samples under the same 45k/5k split and the same augmentation, and the architecture
    # is the CIFAR-10 one with a wider head (e.g. resnet20: 275 572 parameters against 269 722,
    # +2.2%). Every row below is therefore its CIFAR-10 counterpart's MEASURED value, unchanged;
    # the headroom already in those values covers the head. Run the calibration job first anyway.
    # ------------------------------------------------------------------------------------------
    "cnn_gn_cifar100": ("00:20:00", "00:10:00", "DERIVED from cnn_gn_cifar (MEASURED, 46.0 s)."),
    "vit_micro_cifar100": ("00:25:00", "00:10:00",
                           "DERIVED from vit_micro_cifar (MEASURED, 67.4 s)."),
    "resnet20_cifar100": ("01:05:00", "00:10:00",
                          "DERIVED from resnet20_cifar (MEASURED, 360.3 s)."),
    "cct_2_3x2_cifar100": ("00:45:00", "00:10:00",
                           "DERIVED from cct_2_3x2_cifar (MEASURED, 237.5 s)."),
    "resnet50_cifar100": ("01:15:00", "00:15:00",
                          "DERIVED from resnet50_cifar (MEASURED, 2822.8 s). PER-ARM figure."),
    "vit_small_cifar100": ("00:30:00", "00:10:00",
                           "DERIVED from vit_small_cifar (MEASURED, 987.8 s). PER-ARM figure."),
    # ------------------------------------------------------------------------------------------
    # ImageNet-1K. ESTIMATED — nothing here has been measured on this cluster, which is exactly
    # why every one of these models also gets a calibration job, and why the estimates are
    # deliberately generous. The reasoning, for the 32 px benches: 1 256 167 training images
    # against CIFAR's 45 000 is 27.9x the samples per epoch, and 40 epochs against 30 or 50, so
    # the GPU work is 22-37x its CIFAR counterpart's. That is NOT the binding constraint —
    # decoding 1.28 M JPEGs per epoch is, even pre-resized to 32 px, which is what --cpus-per-task
    # is raised for below. The figures assume the pre-resized `imagenet32` tree (stage_imagenet.sh
    # resize); reading full-resolution JPEGs instead costs roughly another 5-10x and will not fit.
    # For the two 224 px benches: ~1250 img/s fwd+bwd for fp32 ResNet-50 on one H100, i.e. ~17 min
    # an epoch and ~8.5 h for the 30-epoch reference arm, hence 12 h per arm with headroom.
    # ------------------------------------------------------------------------------------------
    "cnn_gn_imagenet": ("03:30:00", "00:30:00",
                        "ESTIMATED. Data-loader bound; ~2 h/arm expected at 40 epochs."),
    "vit_micro_imagenet": ("03:30:00", "00:30:00",
                           "ESTIMATED. Data-loader bound; ~2 h/arm expected at 40 epochs."),
    "resnet20_imagenet": ("05:00:00", "00:30:00",
                          "ESTIMATED. 27.9x resnet20_cifar's samples/epoch, 40 epochs."),
    "cct_2_3x2_imagenet": ("05:00:00", "00:30:00",
                           "ESTIMATED. 27.9x cct_2_3x2_cifar's samples/epoch, 40 epochs."),
    "resnet50_imagenet": ("12:00:00", "01:00:00",
                          "ESTIMATED. ~1250 img/s fp32 on one H100 = ~8.5 h for 30 epochs."),
    "vit_small_imagenet": ("12:00:00", "01:00:00",
                           "ESTIMATED. ViT-S/16 at 224 px, 197 tokens, fp32, 30 epochs."),
}

# ``(--gpus, --cpus-per-task, --mem)``. The default is the MIG slice every pre-existing job has
# always used and is left untouched. ImageNet needs two things it does not give: CPU cores, because
# 1.28 M JPEG decodes per epoch is the actual bottleneck for the small models, and — at 224 px — a
# whole H100, since a 10 GB slice does not hold ResNet-50's activations at batch 256, let alone
# ekfac's cached inputs on top.
DEFAULT_RESOURCES = ("h100_1g.10gb:1", 8, "16G")
RESOURCES = {
    "cnn_gn_imagenet": ("h100_1g.10gb:1", 16, "64G"),
    "vit_micro_imagenet": ("h100_1g.10gb:1", 16, "64G"),
    "resnet20_imagenet": ("h100_1g.10gb:1", 16, "64G"),
    "cct_2_3x2_imagenet": ("h100_1g.10gb:1", 16, "64G"),
    "resnet50_imagenet": ("h100:1", 16, "96G"),
    "vit_small_imagenet": ("h100:1", 16, "96G"),
}

# Models whose seven arms are ALSO emitted as a single job (``train_<model>_all.sh``), with the
# ``--time`` that job needs. Grouping trades parallelism for one setup instead of seven — a ~4 min
# saving per model, not the ~1 h an earlier draft of this file claimed, since setup is measured at
# ~40 s (see WALLTIME above). The real arguments for grouping are therefore fewer jobs to track and
# no dispatcher or ``--dependency`` to chain, not machine time.
#
# ``resnet50_cifar`` and ``vit_small_cifar`` used to be deliberately absent, for two reasons. One
# still holds and one no longer does:
#   * STILL TRUE: serialising ResNet-50's seven 47-minute arms is a ~6 h job, which backfills worse
#     than seven concurrent ones. Accepted here, because the per-arm alternative requires copying
#     the reference arm's measured ``total_s`` into six ``--wct-budget`` flags by hand, and that is
#     exactly how the first campaign produced a result set whose checkpoint fractions were wrong
#     (benchmarks/archives/2026-09-10_prefix_campaign/README.md). A grouped job derives the budget
#     in-process; there is no number to copy.
#   * NO LONGER TRUE: "a grouped job loses every completed arm if a later one crashes". It did —
#     ``resnet20_cifar_all`` and ``mlp_ln_mnist_all`` each died in their third arm and lost the two
#     that had finished (cause in ``_eigh_utils.py``'s docstring) — because ``main`` wrote its
#     report only after the last arm. ``main`` now rewrites the whole report after *every* arm.
GROUPED = {
    "mnist_autoencoder": ("00:15:00", "MEASURED. 7 x T_diag = 7 x 13.7 s = 95.7 s of training."),
    "mlp_ln_mnist": ("00:15:00", "MEASURED. 7 x T_diag = 7 x 21.6 s = 151 s of training."),
    "cnn_gn_cifar": ("00:20:00", "MEASURED. 7 x T_diag = 7 x 46.0 s = 322 s of training."),
    "vit_micro_cifar": ("00:25:00", "MEASURED. 7 x T_diag = 7 x 67.4 s = 472 s of training."),
    "cct_2_3x2_cifar": ("00:45:00", "MEASURED. The job ran 00:30:22 end to end for all 7 arms."),
    "resnet20_cifar": ("01:05:00", "MEASURED. 7 x T_diag = 7 x 360.3 s = 42 min of training."),
    "resnet50_cifar": ("07:00:00",
                       "MEASURED. 7 x T_diag = 7 x 2822.8 s = 5.49 h of training; +6% validation"
                       " and ~40 s setup = ~5.9 h. The longest job of the campaign."),
    "vit_small_cifar": ("02:45:00",
                        "MEASURED. 7 x T_diag = 7 x 987.8 s = 1.92 h of training; +6% validation"
                        " and ~40 s setup = ~2.1 h."),
    # CIFAR-100: 7 x the DERIVED per-arm figure, i.e. the CIFAR-10 grouped value unchanged.
    "cnn_gn_cifar100": ("00:20:00", "DERIVED from cnn_gn_cifar (MEASURED)."),
    "vit_micro_cifar100": ("00:25:00", "DERIVED from vit_micro_cifar (MEASURED)."),
    "resnet20_cifar100": ("01:05:00", "DERIVED from resnet20_cifar (MEASURED)."),
    "cct_2_3x2_cifar100": ("00:45:00", "DERIVED from cct_2_3x2_cifar (MEASURED)."),
    "resnet50_cifar100": ("07:00:00", "DERIVED from resnet50_cifar (MEASURED)."),
    "vit_small_cifar100": ("02:45:00", "DERIVED from vit_small_cifar (MEASURED)."),
    # ImageNet-1K @32px: 7 arms in one job, which is what keeps the WCT budget derived in-process
    # rather than hand-copied. The two 224 px models are deliberately absent — 7 x ~8.5 h does not
    # fit any reasonable queue, so they run as seven per-arm jobs and the reference arm's measured
    # total_s has to be passed to the other six by hand. See benchmarks/slurm/imagenet/README.md.
    "cnn_gn_imagenet": ("16:00:00", "ESTIMATED. 7 x ~2 h, data-loader bound."),
    "vit_micro_imagenet": ("16:00:00", "ESTIMATED. 7 x ~2 h, data-loader bound."),
    "resnet20_imagenet": ("23:00:00", "ESTIMATED. 7 x ~3 h."),
    "cct_2_3x2_imagenet": ("23:00:00", "ESTIMATED. 7 x ~3 h."),
}

# Hyperparameter sweeps: one job per swept model, running every arm once per value into
# ``outputs/sweeps/<model>_<field>/<value>/``. ``{field}`` is any ``HParams`` field, i.e. any
# generated ``--<field>`` flag (runner.add_hparam_arguments).
#
# ``mnist_autoencoder``'s ``lam``: on the first campaign all five Fisher modes froze at
# MSE ~= 0.7085 within 2 epochs of a 20-epoch run and never moved again, while ``adam`` descended
# monotonically to 0.4836. Measured on the checkpoints: between 10% and 100% of training the
# parameters moved by 0.2-0.5% for the five Fisher arms against 147% for ``adam``, having
# travelled ~1.0 from initialisation against 25.8. That is a stalled step size on an 8-layer
# all-sigmoid autoencoder, not a converged optimum, and ``lam`` is the parameter that sets it —
# hence a bracket around the default rather than a one-sided scan: 1e-1 damps the step towards
# Adam-like magnitudes, 1e-5 removes almost all damping. Cheap: 7 arms x T_diag = 96 s per value.
SWEEPS = {
    "mnist_autoencoder": [("lam", ["1e-5", "1e-3", "1e-1"], "00:20:00")],
}

# Extra training seeds for the Fisher-drift campaign (``plan_exp_draft_v0.md`` §3.4: "3 graines
# d'entrainement par modele (regime A : 5)"; §9's Lot 1 for A1 says "5 points de controle x 5
# graines"). Seed 0 already exists for all five models, so regime A needs 4 more and regime B 2.
#
# Only TWO arms are needed, not seven: §3.5 samples the trajectories of "AdamW (reference neutre)
# et AdaFisher (trajectoire propre)" only. That is what makes this cheap -- 2 x T_diag per model
# per seed, ~1 h of MIG slice for the whole table, against 8 h for the two large models that
# serve the lot-8 convergence question instead.
#
# --lr-schedule is passed EXPLICITLY, so the job says which protocol it ran under instead of
# leaving it to the CLI default. ``nominal`` is chosen because it is what this job has always done
# (the flag was absent, and ``nominal`` is the default), so the seeds already on disk and any
# future one stay one set.
#
# An earlier version of this comment claimed the flag was omitted so that these seeds would match
# seed 0's ``nominal`` schedule. Measured from the manifests on disk, that premise is false and
# seed 0 is not one protocol but three:
#   * ``mlp_ln_mnist``      seed 0: ``"lr_schedule": "budget"``
#   * the other four models seed 0: no ``lr_schedule`` key at all — those runs pre-date
#     ``common/schedules.py`` and used torch's *periodic* ``CosineAnnealingLR``, whose rate climbs
#     back up past ``T_max``. A missing value is not ``nominal``.
# So "match seed 0" was never achievable across the table, and the honest fix is to record what
# these runs do rather than to claim agreement with something that does not exist. The existing
# runs are left alone; their protocol is what their own manifests say.
#
# Use ``budget`` instead when the point is a convergence curve under the WCT protocol, which is
# what every generated *training* job passes. For drift checkpoints the schedule only has to be
# stated, not optimal.
V0_SEED_ARMS = ("diag", "adamw")
V0_SEEDS_LR_SCHEDULE = "nominal"
V0_SEEDS = {
    "mlp_ln_mnist": [1, 2, 3, 4],      # regime A -> 5 seeds total
    "cnn_gn_cifar": [1, 2, 3, 4],
    "vit_micro_cifar": [1, 2, 3, 4],
    "cct_2_3x2_cifar": [1, 2],         # regime B -> 3 seeds total
    "resnet20_cifar": [1, 2],
}
# MEASURED. Training is 2 x T_diag per (model, seed), exact by construction of the WCT protocol:
#   A1 2x21.6 + A2 2x46.0 + A3 2x67.4 + B1 2x237.5 + B2 2x360.3 = 24.4 min per full seed.
#   4 regime-A seeds + 2 regime-B seeds = 62 min of training, +~2 min of eval, +40 s of setup.
V0_SEEDS_TIME = "01:45:00"

HEADER = """#!/bin/bash
# {title}
# Generated by benchmarks/slurm/generate_jobs.py — edit that file, not this one.
# See docs/reports/plan_exp_step1.md §6 and benchmarks/slurm/README.md.
# Submit from the repository root:  sbatch benchmarks/slurm/{filename}

#SBATCH --account=def-msh-ab
#SBATCH --job-name={job_name}
#SBATCH --gpus={gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output=benchmarks/slurm/logs/%x-%j.out

# --time: {time_basis}
# Dataset: {dataset}, read from $DATA_ROOT (default $SLURM_SUBMIT_DIR/dataset; an ImageNet job
# overrides it with the node-local copy it stages below). Compute nodes have no internet, and
# --no-allow-download turns a missing dataset into a clear error rather than a network timeout.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023 gcc/13.3 cuda/12.6 python/3.11.5

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements-cluster.txt

export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR"
{data_staging}DATA_ROOT="${{DATA_ROOT:-$SLURM_SUBMIT_DIR/dataset}}"

"""

# Inserted into HEADER just before ``DATA_ROOT=``, per dataset. Only ImageNet needs one: a
# torchvision archive is a handful of files read once, whereas ImageNet is 1.28 M small files
# whose per-epoch random read pattern is what a shared parallel filesystem is worst at. Alliance
# Canada's guidance is to keep such a dataset as one archive on /project and expand it into the
# node-local $SLURM_TMPDIR at job start, which is what this does.
IMAGENET_STAGING = """
# ---------------------------------------------------------------------------------------------
# Stage ImageNet into node-local storage. IMAGENET_ARCHIVE must point at a tar of the ImageFolder
# tree stage_imagenet.sh (beside this script) produces: train/<wnid>/*.JPEG + val/<wnid>/*.JPEG
# under a single `imagenet/` or `imagenet32/` directory. The 32 px benches want the pre-resized
# `imagenet32` tar; the 224 px ones want the full-resolution `imagenet` tar.
#   sbatch --export=ALL,IMAGENET_ARCHIVE=/project/def-msh-ab/blgr/{archive}.tar ...
# Set IMAGENET_SKIP_STAGING=1 to read from $SLURM_SUBMIT_DIR/dataset instead (smoke runs).
# ---------------------------------------------------------------------------------------------
IMAGENET_ARCHIVE="${{IMAGENET_ARCHIVE:-}}"
if [ -z "${{IMAGENET_SKIP_STAGING:-}}" ] && [ -n "$IMAGENET_ARCHIVE" ]; then
  echo "staging $IMAGENET_ARCHIVE into $SLURM_TMPDIR/dataset ..."
  mkdir -p "$SLURM_TMPDIR/dataset"
  time tar -xf "$IMAGENET_ARCHIVE" -C "$SLURM_TMPDIR/dataset"
  export DATA_ROOT="$SLURM_TMPDIR/dataset"
  echo "staged: $(ls "$SLURM_TMPDIR/dataset")"
elif [ -z "${{IMAGENET_SKIP_STAGING:-}}" ]; then
  echo "IMAGENET_ARCHIVE unset - reading from the submit directory; slow. See the header." >&2
fi
"""

CALIBRATION_BODY = """# Calibration: 2 short epochs of every arm on a 5000-example training subset, printing per-arm
# steps/s, median step time and peak CUDA memory. Its job is to turn this model's --time value
# from an extrapolation into a measurement, and to confirm every arm fits the 10 GB MIG slice.
# Run this BEFORE the training jobs.
python -m benchmarks.models.{model}.bench \\
  --arms {arms} \\
  --epochs 2 \\
  --budget-mode epochs \\
  --train-subset 5000 \\
  --num-workers {workers} \\
  --no-allow-download \\
  --data-root "$DATA_ROOT" \\
  --output-dir "$SLURM_SUBMIT_DIR/benchmarks/outputs/{out}/_calibration"
"""

REFERENCE_BODY = """# Reference arm (plan_lot8.md §0.7): a fixed {epochs}-epoch run whose measured elapsed time is
# every other arm's wall-clock budget. Read it back from the run's manifest.json:
#   python -c "import json;print(json.load(open('benchmarks/outputs/{out}/{arm}/manifest.json'))['arms']['{arm}']['total_s'])"
# and pass it to the other jobs as --wct-budget (they default to WCT_BUDGET below).
python -m benchmarks.models.{model}.bench \\
  --arms {arm} \\
  --epochs {epochs} \\
  --budget-mode epochs \\
  --num-workers {workers} \\
  --no-allow-download \\
  --data-root "$DATA_ROOT" \\
  --checkpoints 0,0.01,0.1,0.5,1 \\
  --output-dir "$SLURM_SUBMIT_DIR/benchmarks/outputs/{out}/{arm}"
"""

BUDGETED_BODY = """# One WCT-budgeted arm (plan_lot8.md §0.7). WCT_BUDGET is the reference arm's measured elapsed
# time in seconds; override it at submission time once the reference job has run:
#   sbatch --export=ALL,WCT_BUDGET=6543 benchmarks/slurm/{filename}
# Leaving it unset falls back to --budget-mode epochs, i.e. the (deliberately rigged, plan.md
# §6.3) equal-epoch comparison — usable, but not this campaign's headline protocol.
WCT_BUDGET="${{WCT_BUDGET:-}}"
if [ -n "$WCT_BUDGET" ]; then
  BUDGET_ARGS=(--budget-mode wct --wct-budget "$WCT_BUDGET" --max-epoch-factor 3
               --lr-schedule budget)
else
  echo "WCT_BUDGET unset — falling back to --budget-mode epochs (see this script's header)" >&2
  BUDGET_ARGS=(--budget-mode epochs)
fi

python -m benchmarks.models.{model}.bench \\
  --arms {arm} \\
  --epochs {epochs} \\
  "${{BUDGET_ARGS[@]}}" \\
  --num-workers {workers} \\
  --no-allow-download \\
  --data-root "$DATA_ROOT" \\
  --checkpoints 0,0.01,0.1,0.5,1 \\
  --output-dir "$SLURM_SUBMIT_DIR/benchmarks/outputs/{out}/{arm}"
"""


GROUPED_BODY = """# All {n_arms} arms of one model, in ONE job (plan_exp_step1.md D3's runner does the whole WCT
# protocol in-process): the reference arm runs {epochs} epochs unbudgeted, its measured elapsed
# time becomes every other arm's budget, and one combined report is written at the end. No
# WCT_BUDGET to pass, no dispatcher, no --dependency.
#
# The per-arm alternative is benchmarks/slurm/{jobdir}train_{model}_<arm>.sh, one job each; use it if you
# need the arms to run concurrently. A crash in one arm no longer costs the others either way:
# main() rewrites records.csv/epochs.csv/summary.md/manifest.json after every completed arm.
#
# --lr-schedule budget: each budgeted arm anneals its cosine over its OWN wall-clock budget, so
# every arm completes one full cosine and is compared at the same point of its own schedule. With
# the previous shared T_max=--epochs, a cheap arm overshot the nominal epoch count and its LR
# climbed back up, while an expensive arm stopped before reaching the floor — both measured, both
# documented in benchmarks/common/schedules.py. The reference arm is unbudgeted and keeps the
# nominal schedule; it is what defines the budget.
python -m benchmarks.models.{model}.bench \\
  --arms {arms} \\
  --epochs {epochs} \\
  --budget-mode wct \\
  --reference-arm {reference} \\
  --max-epoch-factor 3 \\
  --lr-schedule budget \\
  --num-workers {workers} \\
  --no-allow-download \\
  --data-root "$DATA_ROOT" \\
  --checkpoints 0,0.01,0.1,0.5,1 \\
  --output-dir "$SLURM_SUBMIT_DIR/benchmarks/outputs/{out}"
"""


SWEEP_BODY = """# Hyperparameter sweep: the full {n_arms}-arm WCT protocol, once per --{flag} value, each into its
# own output directory. Every value re-derives its own budget from its own reference arm, so the
# arms of one value are comparable to each other; values are comparable to each other only in the
# sense that they share the model, the seed and the nominal epoch count.
#
# Why this sweep exists: see SWEEPS in benchmarks/slurm/generate_jobs.py.
for VALUE in {values}; do
  echo "=== --{flag} $VALUE ==="
  python -m benchmarks.models.{model}.bench \\
    --arms {arms} \\
    --epochs {epochs} \\
    --budget-mode wct \\
    --reference-arm {reference} \\
    --max-epoch-factor 3 \\
    --lr-schedule budget \\
    --{flag} "$VALUE" \\
    --num-workers {workers} \\
    --no-allow-download \\
    --data-root "$DATA_ROOT" \\
    --checkpoints 0,0.01,0.1,0.5,1 \\
    --output-dir "$SLURM_SUBMIT_DIR/benchmarks/outputs/sweeps/{model}_{flag}/$VALUE"
done
"""


V0_SEEDS_BODY = """# Extra training seeds for the Fisher-drift campaign (plan_exp_draft_v0.md §3.4). Only the two
# trajectories §3.5 samples are run -- AdamW (neutral reference) and AdaFisher (diag) -- so this is
# 2 x T_diag per (model, seed), not 7. Regime A gets seeds 1-4 (5 with the existing seed 0), regime
# B seeds 1-2 (3 with seed 0).
#
# --lr-schedule is stated explicitly (see V0_SEEDS_LR_SCHEDULE in generate_jobs.py) so this job is
# self-describing: 'nominal' is what it has always run, and a future reader does not have to know
# the CLI default to know what these checkpoints sit on. Seed 0 is NOT one protocol -- mlp_ln_mnist
# ran 'budget' and the other four pre-date schedules.py entirely -- so read each run's own
# manifest.json before comparing a seed against it.
#
# Writes to benchmarks/outputs/seeds/<model>/seed<n>/ -- deliberately ungrouped, its axis being
# the seed -- leaving outputs/<group>/<model>/ (seed 0) untouched.
# One failing run does not abort the rest; the job exits non-zero at the end if any failed.
FAILED=()
run_one () {{  # $1 = model, $2 = seed
  echo "=== $1  seed $2 ==="
  python -m "benchmarks.models.$1.bench" \\
    --arms {arms} \\
    --seed "$2" \\
    --budget-mode wct \\
    --reference-arm {reference} \\
    --max-epoch-factor 3 \\
    --lr-schedule {lr_schedule} \\
    --num-workers {workers} \\
    --no-allow-download \\
    --data-root "$DATA_ROOT" \\
    --checkpoints 0,0.01,0.1,0.5,1 \\
    --no-plot \\
    --output-dir "$SLURM_SUBMIT_DIR/benchmarks/outputs/seeds/$1/seed$2" \\
  || FAILED+=("$1:seed$2")
}}

{invocations}
if [ ${{#FAILED[@]}} -gt 0 ]; then
  echo "FAILED runs: ${{FAILED[*]}}" >&2
  exit 1
fi
echo "all seed runs completed"
"""


def dataset_of(bench) -> str:
    """``build_data`` is a ``functools.partial`` of ``common.data.build_loaders`` over one
    ``DatasetSpec``; its name is what has to be staged on the cluster.
    """
    return bench.build_data.args[0].name


def write(filename: str, text: str) -> None:
    """``filename`` may carry a one-level prefix (``cifar100/train_x_diag.sh``) — the bench's
    ``output_group``, so the jobs of a dataset sit together exactly as its results do.
    """
    path = SLURM_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    print(f"wrote {path.relative_to(SLURM_DIR.parents[1])}")


def workers_of(bench) -> int:
    """``--num-workers`` = ``--cpus-per-task``: the dataloader is the only multi-process consumer
    in these jobs, and on ImageNet it is the binding one (1.28 M JPEG decodes per epoch).
    """
    return RESOURCES.get(bench.name, DEFAULT_RESOURCES)[1]


def header(bench, dataset: str = "", **kwargs) -> str:
    """``HEADER`` with this bench's resource profile and dataset staging filled in.

    ``dataset`` overrides the name printed in the header comment, for the one job that spans
    several benches (``train_v0_seeds.sh``); the staging block still follows ``bench``.
    """
    gpus, cpus, mem = RESOURCES.get(bench.name, DEFAULT_RESOURCES)
    dataset = dataset or dataset_of(bench)
    staging = ""
    if dataset_of(bench) == "imagenet":
        # ``imagenet32`` for the downsampled benches, ``imagenet`` for the native ones — the same
        # directory name ``common.data.imagenet_root`` looks for, derived from the bench's own
        # configured resolution rather than from its name.
        img_size = bench.build_data.keywords["img_size"]
        staging = IMAGENET_STAGING.format(
            archive="imagenet" if img_size >= 64 else f"imagenet{img_size}"
        )
    return HEADER.format(gpus=gpus, cpus=cpus, mem=mem, dataset=dataset,
                         data_staging=staging, **kwargs)


def main() -> None:
    benches = discover_benchmarks()
    for name, bench in sorted(benches.items()):
        train_time, calib_time, basis = WALLTIME[name]
        # ``output_group`` is one string used three times: the job subdirectory, the results
        # subdirectory, and nothing else. Empty for every CIFAR-10/MNIST bench, which therefore
        # keeps the flat layout its existing results and jobs already use.
        jobdir = f"{bench.output_group}/" if bench.output_group else ""
        out = f"{bench.output_group}/{name}" if bench.output_group else name

        filename = f"{jobdir}calibrate_{name}.sh"
        write(
            filename,
            header(
                bench,
                title=f"{bench.title()}: calibration pass, all arms, 2 epochs on a 5k subset.",
                filename=filename, job_name=f"cal_{name}", time=calib_time,
                time_basis=("2 epochs x %d arms on a 5000-example subset; dominated by env setup "
                            "and the first-step hook/eigendecomposition warm-up, not by training. "
                            "Generous on purpose." % len(bench.arms)),
            )
            + CALIBRATION_BODY.format(model=name, out=out, workers=workers_of(bench),
                                     arms=" ".join(bench.arms)),
        )
        if name in GROUPED:
            grouped_time, grouped_basis = GROUPED[name]
            filename = f"{jobdir}train_{name}_all.sh"
            write(
                filename,
                header(
                    bench,
                    title=(f"{bench.title()}, all {len(bench.arms)} arms in one job. "
                           f"Writes to benchmarks/outputs/{out}/."),
                    filename=filename, job_name=f"{name}_all", time=grouped_time,
                    time_basis=grouped_basis,
                )
                + GROUPED_BODY.format(model=name, out=out, jobdir=jobdir,
                                      arms=" ".join(bench.arms), epochs=bench.epochs,
                                      workers=workers_of(bench),
                                      reference=REFERENCE_ARM, n_arms=len(bench.arms)),
            )

        for flag, values, sweep_time in SWEEPS.get(name, []):
            filename = f"{jobdir}sweep_{name}_{flag}.sh"
            write(
                filename,
                header(
                    bench,
                    title=(f"{bench.title()}: --{flag} sweep over {values}, all "
                           f"{len(bench.arms)} arms per value. Writes to "
                           f"benchmarks/outputs/sweeps/{name}_{flag}/<value>/."),
                    filename=filename, job_name=f"{name}_{flag}_sweep", time=sweep_time,
                    time_basis=(f"{len(values)} values x {len(bench.arms)} arms x T_reference; "
                                f"see SWEEPS in generate_jobs.py for the per-value measurement."),
                )
                + SWEEP_BODY.format(model=name, flag=flag, values=" ".join(values),
                                    arms=" ".join(bench.arms), epochs=bench.epochs,
                                    workers=workers_of(bench),
                                    reference=REFERENCE_ARM, n_arms=len(bench.arms)),
            )

        for arm in bench.arms:
            filename = f"{jobdir}train_{name}_{arm}.sh"
            body = (REFERENCE_BODY if arm == REFERENCE_ARM else BUDGETED_BODY).format(
                model=name, out=out, arm=arm, epochs=bench.epochs, filename=filename,
                workers=workers_of(bench),
            )
            write(
                filename,
                header(
                    bench,
                    title=(f"{bench.title()}, arm={arm}"
                           + (" (WCT reference arm)" if arm == REFERENCE_ARM else "")
                           + f". Writes to benchmarks/outputs/{out}/{arm}/."),
                    filename=filename, job_name=f"{name}_{arm}", time=train_time,
                    time_basis=basis,
                ) + body,
            )

    write_v0_seeds(benches)


def write_v0_seeds(benches) -> None:
    """One job for the whole extra-seed table: ~1 h, so a single setup and one job to track."""
    invocations = "\n".join(
        f'run_one {model} {seed}'
        for model in V0_SEEDS
        for seed in V0_SEEDS[model]
    )
    filename = "train_v0_seeds.sh"
    total = sum(len(v) for v in V0_SEEDS.values())
    write(
        filename,
        header(
            benches["mlp_ln_mnist"],  # every model in V0_SEEDS shares the default MIG profile
            dataset=" and ".join(sorted({dataset_of(benches[m]) for m in V0_SEEDS})),
            title=(f"Fisher-drift campaign: {total} extra training seeds over "
                   f"{len(V0_SEEDS)} models, arms={'+'.join(V0_SEED_ARMS)} only. "
                   f"Writes to benchmarks/outputs/seeds/<model>/seed<n>/."),
            filename=filename, job_name="v0_seeds", time=V0_SEEDS_TIME,
            time_basis=("MEASURED. 2 x T_diag per (model, seed): 24.4 min for one full sweep of "
                        "the five models; 4 regime-A seeds + 2 regime-B seeds = 62 min of "
                        "training, plus ~2 min of eval and ~40 s of setup."),
        )
        + V0_SEEDS_BODY.format(arms=" ".join(V0_SEED_ARMS), reference=REFERENCE_ARM,
                               lr_schedule=V0_SEEDS_LR_SCHEDULE,
                               workers=DEFAULT_RESOURCES[1], invocations=invocations),
    )


if __name__ == "__main__":
    main()
