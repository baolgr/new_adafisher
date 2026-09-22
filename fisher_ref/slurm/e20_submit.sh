#!/bin/bash
# Submit E20 (docs/reports/plan_e20_s1b_vit_small.md), from the root of ONE clean checkout.
#
#   1. the calibration, allowed at any time (the shipped setting only, which E17 allows):
#        bash fisher_ref/slurm/e20_submit.sh calib
#      Read the minutes of its one run in its log ("ekfac reference ... (NNNs)").
#
#   2. stage 1, only once E17's addendum naming its candidates is committed (E20, section 0):
#        E20_E17_ADDENDUM=<commit> E20_RUN_MIN=<minutes per run from step 1> \
#          bash fisher_ref/slurm/e20_submit.sh stage1
#      15 jobs: (seed 0-4) x (ekfac, tekfac, adamw).
#
#   3. an edge extension, only if e20_decisions.py asks for it (rules 2 and 5), one job per seed:
#        sbatch --job-name=e20_s<seed>_<job>_ext --time=<limit> \
#          --export=ALL,E20_SEED=<seed>,E20_JOB=<ekfac|tekfac|adamw>,E20_VALUES=<v1>:<v2>,E20_TAG=ext \
#          fisher_ref/slurm/e20_s1b_vit_small.sh
#
# Refuses a checkout whose tracked files are modified: e20_decisions.py refuses files from a dirty
# tree or from different commits. Untracked files (the dataset link, fisher_ref/outputs/) do not
# count, as in the driver's own provenance.

set -euo pipefail
what="${1:?usage: e20_submit.sh calib|stage1}"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "e20_submit: tracked files are modified in $(pwd); refusing" >&2; exit 1
fi
test -e dataset || { echo "e20_submit: no dataset/ in $(pwd)" >&2; exit 1; }
git ls-files --error-unmatch docs/reports/plan_e20_s1b_vit_small.md >/dev/null 2>&1 \
  || { echo "e20_submit: the pre-registration is not committed; refusing" >&2; exit 1; }
job=fisher_ref/slurm/e20_s1b_vit_small.sh
commit="$(git rev-parse --short HEAD)"

case "$what" in
  calib)
    id="$(sbatch --parsable --job-name=e20_calib --time=00:45:00 \
            --export=ALL,E20_SEED=0,E20_JOB=calib "$job")"
    echo "e20 calibration (commit $commit): job $id" ;;
  stage1)
    addendum="${E20_E17_ADDENDUM:?set E20_E17_ADDENDUM to the commit of the E17 candidate addendum}"
    run_min="${E20_RUN_MIN:?set E20_RUN_MIN to the minutes of one run, from the calibration}"
    git merge-base --is-ancestor "$addendum" HEAD \
      || { echo "e20_submit: $addendum is not an ancestor of HEAD" >&2; exit 1; }
    git show --name-only --format= "$addendum" | grep -qx docs/reports/plan_lambda_dominance.md \
      || { echo "e20_submit: $addendum does not touch plan_lambda_dominance.md" >&2; exit 1; }
    # 14 cells per Fisher job (+ the short bridge at seed 0), 6 per AdamW job; 1.5x margin + 10 min.
    fisher_min="$(python3 -c "import math; print(math.ceil(14 * $run_min * 1.5 + 10))")"
    adamw_min="$(python3 -c "import math; print(math.ceil(6 * $run_min * 1.5 + 10))")"
    for seed in 0 1 2 3 4; do
      for j in ekfac tekfac adamw; do
        m=$fisher_min; [ "$j" = adamw ] && m=$adamw_min
        id="$(sbatch --parsable --job-name="e20_s${seed}_${j}" --time="$m" \
                --export="ALL,E20_SEED=$seed,E20_JOB=$j" "$job")"
        echo "e20 seed $seed $j (commit $commit, --time ${m} min): job $id"
      done
    done ;;
  *) echo "e20_submit: unknown '$what' (calib|stage1)" >&2; exit 1 ;;
esac
