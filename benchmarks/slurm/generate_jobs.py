"""Generate the SLURM job scripts, one calibration job plus one job per arm, for every model
folder (``plan_exp_step1.md`` §6: "jobs enumerate model folders").

The folder list *is* the registry: this script imports ``benchmarks.common.runner.discover_
benchmarks`` and reads each ``Benchmark``'s own nominal epoch count, arm list and dataset, so
adding a model folder adds its jobs with no edit here. The scripts are near-identical apart from
the model, the arm and the ``--time`` budget, hence generated rather than hand-maintained.

    python benchmarks/slurm/generate_jobs.py        # regenerates every .sh in this directory

Every cluster-specific line (account, GPU shape, module load, no-internet assumption) is copied
from job scripts that have actually run on this cluster
(``/Users/baolgr/Documents/AtlasAnalyticsLab/experiments/*/slurm/``), not re-derived — see
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
# arm's own ``total_s`` (``benchmarks/outputs/<model>/[<arm>/]manifest.json``, or its log for
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
# NOTE the deliberate absence of --lr-schedule budget. Seed 0 ran under the default ``nominal``
# schedule; these seeds must share that convention or they are not comparable with it, which is
# their entire purpose. The budget schedule is for convergence curves, not for drift checkpoints.
V0_SEED_ARMS = ("diag", "adamw")
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
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time={time}
#SBATCH --output=benchmarks/slurm/logs/%x-%j.out

# --time: {time_basis}
# Dataset: {dataset} must be staged under $SLURM_SUBMIT_DIR/dataset (compute nodes have no
# internet; --no-allow-download turns a missing dataset into a clear error, not a timeout).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load StdEnv/2023 gcc/13.3 cuda/12.6 python/3.11.5

virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r requirements-cluster.txt

export PYTHONPATH="$SLURM_SUBMIT_DIR/src:$SLURM_SUBMIT_DIR"

"""

CALIBRATION_BODY = """# Calibration: 2 short epochs of every arm on a 5000-example training subset, printing per-arm
# steps/s, median step time and peak CUDA memory. Its job is to turn this model's --time value
# from an extrapolation into a measurement, and to confirm every arm fits the 10 GB MIG slice.
# Run this BEFORE the training jobs.
python -m benchmarks.{model}.bench \\
  --arms {arms} \\
  --epochs 2 \\
  --budget-mode epochs \\
  --train-subset 5000 \\
  --num-workers 8 \\
  --no-allow-download \\
  --data-root "$SLURM_SUBMIT_DIR/dataset" \\
  --output-dir "$SLURM_SUBMIT_DIR/benchmarks/outputs/{model}/_calibration"
"""

REFERENCE_BODY = """# Reference arm (plan_lot8.md §0.7): a fixed {epochs}-epoch run whose measured elapsed time is
# every other arm's wall-clock budget. Read it back from the run's manifest.json:
#   python -c "import json;print(json.load(open('benchmarks/outputs/{model}/{arm}/manifest.json'))['arms']['{arm}']['total_s'])"
# and pass it to the other jobs as --wct-budget (they default to WCT_BUDGET below).
python -m benchmarks.{model}.bench \\
  --arms {arm} \\
  --epochs {epochs} \\
  --budget-mode epochs \\
  --num-workers 8 \\
  --no-allow-download \\
  --data-root "$SLURM_SUBMIT_DIR/dataset" \\
  --checkpoints 0,0.01,0.1,0.5,1 \\
  --output-dir "$SLURM_SUBMIT_DIR/benchmarks/outputs/{model}/{arm}"
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

python -m benchmarks.{model}.bench \\
  --arms {arm} \\
  --epochs {epochs} \\
  "${{BUDGET_ARGS[@]}}" \\
  --num-workers 8 \\
  --no-allow-download \\
  --data-root "$SLURM_SUBMIT_DIR/dataset" \\
  --checkpoints 0,0.01,0.1,0.5,1 \\
  --output-dir "$SLURM_SUBMIT_DIR/benchmarks/outputs/{model}/{arm}"
"""


GROUPED_BODY = """# All {n_arms} arms of one model, in ONE job (plan_exp_step1.md D3's runner does the whole WCT
# protocol in-process): the reference arm runs {epochs} epochs unbudgeted, its measured elapsed
# time becomes every other arm's budget, and one combined report is written at the end. No
# WCT_BUDGET to pass, no dispatcher, no --dependency.
#
# The per-arm alternative is benchmarks/slurm/train_{model}_<arm>.sh, one job each; use it if you
# need the arms to run concurrently. A crash in one arm no longer costs the others either way:
# main() rewrites records.csv/epochs.csv/summary.md/manifest.json after every completed arm.
#
# --lr-schedule budget: each budgeted arm anneals its cosine over its OWN wall-clock budget, so
# every arm completes one full cosine and is compared at the same point of its own schedule. With
# the previous shared T_max=--epochs, a cheap arm overshot the nominal epoch count and its LR
# climbed back up, while an expensive arm stopped before reaching the floor — both measured, both
# documented in benchmarks/common/schedules.py. The reference arm is unbudgeted and keeps the
# nominal schedule; it is what defines the budget.
python -m benchmarks.{model}.bench \\
  --arms {arms} \\
  --epochs {epochs} \\
  --budget-mode wct \\
  --reference-arm {reference} \\
  --max-epoch-factor 3 \\
  --lr-schedule budget \\
  --num-workers 8 \\
  --no-allow-download \\
  --data-root "$SLURM_SUBMIT_DIR/dataset" \\
  --checkpoints 0,0.01,0.1,0.5,1 \\
  --output-dir "$SLURM_SUBMIT_DIR/benchmarks/outputs/{model}"
"""


SWEEP_BODY = """# Hyperparameter sweep: the full {n_arms}-arm WCT protocol, once per --{flag} value, each into its
# own output directory. Every value re-derives its own budget from its own reference arm, so the
# arms of one value are comparable to each other; values are comparable to each other only in the
# sense that they share the model, the seed and the nominal epoch count.
#
# Why this sweep exists: see SWEEPS in benchmarks/slurm/generate_jobs.py.
for VALUE in {values}; do
  echo "=== --{flag} $VALUE ==="
  python -m benchmarks.{model}.bench \\
    --arms {arms} \\
    --epochs {epochs} \\
    --budget-mode wct \\
    --reference-arm {reference} \\
    --max-epoch-factor 3 \\
    --lr-schedule budget \\
    --{flag} "$VALUE" \\
    --num-workers 8 \\
    --no-allow-download \\
    --data-root "$SLURM_SUBMIT_DIR/dataset" \\
    --checkpoints 0,0.01,0.1,0.5,1 \\
    --output-dir "$SLURM_SUBMIT_DIR/benchmarks/outputs/sweeps/{model}_{flag}/$VALUE"
done
"""


V0_SEEDS_BODY = """# Extra training seeds for the Fisher-drift campaign (plan_exp_draft_v0.md §3.4). Only the two
# trajectories §3.5 samples are run -- AdamW (neutral reference) and AdaFisher (diag) -- so this is
# 2 x T_diag per (model, seed), not 7. Regime A gets seeds 1-4 (5 with the existing seed 0), regime
# B seeds 1-2 (3 with seed 0).
#
# No --lr-schedule flag on purpose: seed 0 ran under the default 'nominal' schedule and these must
# match it to be comparable, which is the whole point. Do not add --lr-schedule budget here.
#
# Writes to benchmarks/outputs/seeds/<model>/seed<n>/, leaving outputs/<model>/ (seed 0) untouched.
# One failing run does not abort the rest; the job exits non-zero at the end if any failed.
FAILED=()
run_one () {{  # $1 = model, $2 = seed
  echo "=== $1  seed $2 ==="
  python -m "benchmarks.$1.bench" \\
    --arms {arms} \\
    --seed "$2" \\
    --budget-mode wct \\
    --reference-arm {reference} \\
    --max-epoch-factor 3 \\
    --num-workers 8 \\
    --no-allow-download \\
    --data-root "$SLURM_SUBMIT_DIR/dataset" \\
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
    path = SLURM_DIR / filename
    path.write_text(text)
    path.chmod(0o755)
    print(f"wrote {path.relative_to(SLURM_DIR.parents[1])}")


def main() -> None:
    benches = discover_benchmarks()
    for name, bench in sorted(benches.items()):
        train_time, calib_time, basis = WALLTIME[name]
        dataset = dataset_of(bench)
        write(
            f"calibrate_{name}.sh",
            HEADER.format(
                title=f"{bench.title()}: calibration pass, all arms, 2 epochs on a 5k subset.",
                filename=f"calibrate_{name}.sh", job_name=f"cal_{name}", time=calib_time,
                dataset=dataset,
                time_basis=("2 epochs x %d arms on a 5000-example subset; dominated by env setup "
                            "and the first-step hook/eigendecomposition warm-up, not by training. "
                            "Generous on purpose." % len(bench.arms)),
            )
            + CALIBRATION_BODY.format(model=name, arms=" ".join(bench.arms)),
        )
        if name in GROUPED:
            grouped_time, grouped_basis = GROUPED[name]
            filename = f"train_{name}_all.sh"
            write(
                filename,
                HEADER.format(
                    title=(f"{bench.title()}, all {len(bench.arms)} arms in one job. "
                           f"Writes to benchmarks/outputs/{name}/."),
                    filename=filename, job_name=f"{name}_all", time=grouped_time,
                    time_basis=grouped_basis, dataset=dataset,
                )
                + GROUPED_BODY.format(model=name, arms=" ".join(bench.arms),
                                      epochs=bench.epochs, reference=REFERENCE_ARM,
                                      n_arms=len(bench.arms)),
            )

        for flag, values, sweep_time in SWEEPS.get(name, []):
            filename = f"sweep_{name}_{flag}.sh"
            write(
                filename,
                HEADER.format(
                    title=(f"{bench.title()}: --{flag} sweep over {values}, all "
                           f"{len(bench.arms)} arms per value. Writes to "
                           f"benchmarks/outputs/sweeps/{name}_{flag}/<value>/."),
                    filename=filename, job_name=f"{name}_{flag}_sweep", time=sweep_time,
                    time_basis=(f"{len(values)} values x {len(bench.arms)} arms x T_reference; "
                                f"see SWEEPS in generate_jobs.py for the per-value measurement."),
                    dataset=dataset,
                )
                + SWEEP_BODY.format(model=name, flag=flag, values=" ".join(values),
                                    arms=" ".join(bench.arms), epochs=bench.epochs,
                                    reference=REFERENCE_ARM, n_arms=len(bench.arms)),
            )

        for arm in bench.arms:
            filename = f"train_{name}_{arm}.sh"
            body = (REFERENCE_BODY if arm == REFERENCE_ARM else BUDGETED_BODY).format(
                model=name, arm=arm, epochs=bench.epochs, filename=filename
            )
            write(
                filename,
                HEADER.format(
                    title=(f"{bench.title()}, arm={arm}"
                           + (" (WCT reference arm)" if arm == REFERENCE_ARM else "")
                           + f". Writes to benchmarks/outputs/{name}/{arm}/."),
                    filename=filename, job_name=f"{name}_{arm}", time=train_time,
                    time_basis=basis, dataset=dataset,
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
    datasets = sorted({dataset_of(benches[m]) for m in V0_SEEDS})
    filename = "train_v0_seeds.sh"
    total = sum(len(v) for v in V0_SEEDS.values())
    write(
        filename,
        HEADER.format(
            title=(f"Fisher-drift campaign: {total} extra training seeds over "
                   f"{len(V0_SEEDS)} models, arms={'+'.join(V0_SEED_ARMS)} only. "
                   f"Writes to benchmarks/outputs/seeds/<model>/seed<n>/."),
            filename=filename, job_name="v0_seeds", time=V0_SEEDS_TIME,
            time_basis=("MEASURED. 2 x T_diag per (model, seed): 24.4 min for one full sweep of "
                        "the five models; 4 regime-A seeds + 2 regime-B seeds = 62 min of "
                        "training, plus ~2 min of eval and ~40 s of setup."),
            dataset=" and ".join(datasets),
        )
        + V0_SEEDS_BODY.format(arms=" ".join(V0_SEED_ARMS), reference=REFERENCE_ARM,
                               invocations=invocations),
    )


if __name__ == "__main__":
    main()
