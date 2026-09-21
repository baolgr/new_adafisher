"""E16's pre-registered decision rules, applied to the finished runs.

``docs/reports/plan_lambda_dominance.md``, section "E16 -- pre-registered", states the rules; this
script applies them and chooses no threshold of its own. It reads the E16 files,
``fisher_ref/outputs/e16_floor_clip_<model>_s<seed>_<mode>.json``, one per (network, seed, mode)
job. E16 is self-contained: its add baseline is one of its own arms, run under the same estimator
(``norm_exact_rescaling``), code and hardware as the other arms, and every comparison pairs by seed
inside one job.

**Nothing incomplete is read as a verdict.** Every file is checked first: its network, seed and
mode match its name; it is not a smoke; production settings (15 epochs, batch 32, full grids and
split, calibration at step 2000); no missing shard, no absent cell, no crashed cell; every cell but
the repro diagnostic ran with ``norm_exact_rescaling``; every ``clipfixed`` cell froze every layer;
one git commit for all thirty files and a clean tree; and at seed 0 the determinism check (a cell
rerun in another process is bit-identical) and, on ``cnn_gn_cifar``, the inertness check (with no
hooked normalisation layer the corrected statistic changes nothing, bit for bit). A value is usable
only with a finite final validation and test accuracy at all five seeds. A family whose grid or
comparison has an unusable value gets the verdict ``incomplete`` -- never ``equivalent`` -- and the
script exits with status 2, unless ``E16_ALLOW_INCOMPLETE=1``.

**Reported, not voted:** the seed-0 repro diagnostic (the shipped estimator at E14/E13/E10's best
lambda, against the stored cell -- bit-identical only where the hardware and code match E14's).

**The plateau (rule 4).** Pre-registered, in E13's and E14's words: every value whose five-seed
mean test accuracy lies within one standard error of the best mean, the standard error being the
best value's own. That is the criterion voted on. E13's own reported plateaus were in fact computed
with a two-sample criterion (``m_best - m_v <= sqrt(SE_best^2 + SE_v^2)``); it is reported
alongside, not voted on.

**For E17.** E17 (same document) takes its candidates from this script's output: each clip arm and
the floor if rule 3 rates them "better" or "equivalent" and rule 4 finds a transferable value, and
add's transferable lambda or, failing one, the geometric mean of add's six selected lambdas. Those
are printed at the end, as E17 defines them; where several values transfer, the one with the best
validation accuracy averaged over the six pairs (E17's candidate-A rule).

**Exploratory networks (fourth amendment, ``plan_floor_clip.md`` §12).** ``resnet20_cifar`` runs the
same arms and grids but votes in none of the rules above and feeds nothing to E17. It is read in a
section of its own, by rules pre-registered before any of its runs: for each clip arm, rules 1-2
against add in each mode, and the arm is "robust" there if it loses in neither mode, "not robust"
if it loses in one, "incomplete" otherwise. Also reported: whether the q that transfers across the
three voting networks (rule 4) lies inside the network's own plateau. Its files go through the same
checks (one commit per network, not necessarily the voting networks' one: pairing never crosses
networks), but a defect there is listed under that section and does not make the voting verdicts
invalid; a network with no file at all is reported "not run".

Environment variables::

    E16_MODELS            comma-separated (default: cnn_gn_cifar,vit_micro_cifar,cct_2_3x2_cifar)
    E16_EXPLORATORY       comma-separated (default: resnet20_cifar)
    E16_SEEDS             comma-separated (default: 0,1,2,3,4)
    E16_DIR               where the JSON files are (default: fisher_ref/outputs)
    E16_OUT               output path (default: fisher_ref/outputs/e16_decisions.json)
    E16_ALLOW_INCOMPLETE  "1": exit 0 even when something is incomplete (the verdicts still say so)
"""
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fisher_ref.experiments.e16_floor_clip import (  # noqa: E402
    CHECK_LAMBDA,
    CLIP_ARMS,
    CLIP_GRID,
    CLIPLR_ARM,
    CLIPLR_FACTORS,
    CLIPLR_Q,
    EXPLORATORY,
    LAMBDA_GRID,
    NO_HOOKED_NORM,
    VOTING,
)

MODELS = [m.strip() for m in os.environ.get(
    "E16_MODELS", ",".join(VOTING)).split(",") if m.strip()]
EXPLORATORY_MODELS = [m.strip() for m in os.environ.get(
    "E16_EXPLORATORY", ",".join(EXPLORATORY)).split(",") if m.strip()]
SEEDS = [int(s) for s in os.environ.get("E16_SEEDS", "0,1,2,3,4").split(",") if s.strip()]
MODES = ["ekfac", "tekfac"]
DIR = Path(os.environ.get("E16_DIR", str(ROOT / "fisher_ref/outputs")))
OUT = Path(os.environ.get("E16_OUT", str(DIR / "e16_decisions.json")))
ALLOW_INCOMPLETE = os.environ.get("E16_ALLOW_INCOMPLETE", "0") == "1"
FAMILIES = ("floor", *CLIP_ARMS)       # every family judged against add (rules 2-3)
# Rule 6: pairs of families compared with each other, each at its own chosen value.
BETWEEN = [("clipema", "clip"),        # what the per-step normalisation costs or buys
           ("clipfixed", "clipema")]   # a frozen threshold against a momentum-following one
# E17 candidate B may only use lambdas common to every network's grid.
E17_FLOOR_COMMON = [3e-10, 1e-10, 3e-11, 1e-11]

Cell = Dict[str, Any]
Series = Dict[int, Cell]          # seed -> run
Key = Tuple[str, str, float]      # (mode, family, value)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _final_val(c: Cell) -> float:
    v = c.get("epoch_val_acc") or []
    return 100 * v[-1] if v and _finite(v[-1]) else float("nan")


def _test(c: Cell) -> float:
    t = c.get("test_acc")
    return 100 * t if _finite(t) else float("nan")


def _usable(series: Optional[Series]) -> bool:
    """Five seeds, each with a finite final validation and test accuracy, none crashed."""
    return (series is not None and sorted(series) == sorted(SEEDS)
            and all(not series[s].get("crashed") and _finite(_final_val(series[s]))
                    and _finite(_test(series[s])) for s in SEEDS))


def _mean_se(xs: List[float]) -> Tuple[float, float]:
    """Mean and standard error of a complete list. Callers pass only usable series."""
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, sd / math.sqrt(n)


def expected_keys(model: str, seed: int, mode: str) -> List[str]:
    """Every cell key the driver plans for one (network, seed, mode) job."""
    keys = []
    if seed == 0:
        keys += [f"{mode}|dupcheck|{CHECK_LAMBDA[(model, mode)]:g}",
                 f"{mode}|repro|{CHECK_LAMBDA[(model, mode)]:g}"]
    keys += [f"{mode}|{arm}|{v:g}" for arm in ("add", "floor") for v in LAMBDA_GRID[model]]
    for arm in CLIP_ARMS:
        keys += [f"{mode}|{arm}|{q:g}" for q in CLIP_GRID]
    keys += [f"{mode}|cliplr|{f:g}" for f in CLIPLR_FACTORS]
    return keys


def load(model: str, problems: List[str], commits: set,
         diagnostics: Dict[str, Any]) -> Dict[Key, Series]:
    """``{(mode, family, value): {seed: run}}`` for every E16 arm, with every file checked. Each
    defect found is appended to ``problems``."""
    out: Dict[Key, Series] = {}
    for seed in SEEDS:
        for mode in MODES:
            path = DIR / f"e16_floor_clip_{model}_s{seed}_{mode}.json"
            if not path.exists():
                problems.append(f"{path.name}: missing")
                continue
            e16 = json.load(open(path))
            provs = e16.get("provenance") or []
            checks = {
                "network/seed/mode": (e16.get("model"), e16.get("seed"), e16.get("mode"))
                == (model, seed, mode),
                "not a smoke": bool(provs) and all(p and p.get("smoke") is False for p in provs),
                "15 epochs": e16.get("epochs") == 15, "batch 32": e16.get("batch_size") == 32,
                "full grids": e16.get("grid_limit") is None,
                "full split": e16.get("train_subset") is None,
                "calibration at 2000": e16.get("calibrate_at") == 2000,
                "clean tree": bool(provs) and all(p and p.get("git_dirty") is False
                                                  for p in provs),
                "every shard merged": not e16.get("missing_shards"),
            }
            problems += [f"{path.name}: fails '{name}'" for name, ok in checks.items() if not ok]
            commits.update(p.get("git_commit") for p in provs if p)
            absent = [k for k in expected_keys(model, seed, mode) if k not in e16["cells"]]
            if absent:
                problems.append(f"{path.name}: {len(absent)} planned cells absent, e.g. {absent[:3]}")
            if seed == 0:
                gates = e16.get("checks", {})
                if not gates.get("determinism", {}).get("identical"):
                    problems.append(f"{path.name}: determinism check failed "
                                    f"({gates.get('determinism')})")
                if model in NO_HOOKED_NORM and not gates.get("norm_exact_inert", {}).get(
                        "identical"):
                    problems.append(f"{path.name}: norm_exact_rescaling not inert on {model} "
                                    f"({gates.get('norm_exact_inert')})")
                diagnostics[f"{model}|{mode}|repro_vs_e14"] = gates.get("repro_vs_e14")
            for key, c in e16["cells"].items():
                if c.get("crashed"):
                    problems.append(f"{path.name}: {key} crashed ({c.get('error')})")
                    continue
                if c["arm"] != "repro" and not c.get("overrides", {}).get("norm_exact_rescaling"):
                    problems.append(f"{path.name}: {key} ran without norm_exact_rescaling")
                if c.get("overrides", {}).get("clip_threshold") == "fixed":
                    last = (c.get("log") or [{}])[-1].get("layers", {})
                    if not last or any(row.get("calibrated") != 1.0 for row in last.values()):
                        problems.append(f"{path.name}: {key} never froze every layer's threshold")
                if c["arm"] in ("repro", "dupcheck"):
                    continue
                out.setdefault((c["mode"], c["arm"], float(c["value"])), {})[seed] = c
    return out


def select(data: Dict[Key, Series], mode: str, family: str,
           values: List[float]) -> Tuple[Optional[float], List[float]]:
    """Rule 1: the value with the best seed-mean final-epoch VALIDATION accuracy. Returns
    ``(value, unusable values)``; the choice is None if any value of the grid is unusable,
    because a selection over part of a grid is not the pre-registered selection."""
    unusable = [v for v in values if not _usable(data.get((mode, family, v)))]
    if unusable:
        return None, unusable
    means = {v: _mean_se([_final_val(data[(mode, family, v)][s]) for s in SEEDS])[0]
             for v in values}
    return max(values, key=lambda v: means[v]), []


def paired(a: Series, b: Series) -> Tuple[float, float, str]:
    """Rule 2: mean and standard error of test(a) - test(b), paired by seed, and the verdict. A
    zero standard error with a nonzero mean means identical runs somewhere: flagged, not voted."""
    m, se = _mean_se([_test(a[s]) - _test(b[s]) for s in SEEDS])
    if se == 0:
        return m, se, "tie" if m == 0 else "degenerate"
    return m, se, "win" if m > 2 * se else "lose" if m < -2 * se else "tie"


def plateaus(data: Dict[Key, Series], mode: str, family: str,
             values: List[float]) -> Tuple[List[float], List[float]]:
    """Rule 4 on TEST accuracy: (the pre-registered plateau -- within one SE of the best mean,
    SE of the best; the two-sample variant E13's own numbers used, reported only)."""
    stats = {v: _mean_se([_test(data[(mode, family, v)][s]) for s in SEEDS]) for v in values
             if _usable(data.get((mode, family, v)))}
    if len(stats) != len(values):
        return [], []
    best = max(stats, key=lambda v: stats[v][0])
    m_best, se_best = stats[best]
    one_se = [v for v in values if stats[v][0] >= m_best - se_best]
    two_sample = [v for v in values
                  if m_best - stats[v][0] <= math.sqrt(se_best ** 2 + stats[v][1] ** 2)]
    return one_se, two_sample


def family_verdict(vs: Dict[Tuple[str, str], str]) -> str:
    """Rule 3. Anything other than win/tie/lose makes the family incomplete."""
    if any(v not in ("win", "tie", "lose") for v in vs.values()):
        bad = sorted(f"{m}/{md}: {v}" for (m, md), v in vs.items()
                     if v not in ("win", "tie", "lose"))
        return "incomplete (" + "; ".join(bad) + ")"
    by_model: Dict[str, List[str]] = {}
    for (model, _), v in vs.items():
        by_model.setdefault(model, []).append(v)
    losses = sum(1 for m in by_model.values() if "lose" in m)
    wins_both = sum(1 for m in by_model.values() if m.count("win") == len(MODES))
    if losses == 0 and wins_both >= 2:
        return "better than the E14 fix"
    if losses >= 2:
        return "worse than the E14 fix"
    if losses == 0:
        return "equivalent to the E14 fix"
    return "mixed"


def exploratory(voting_commits: set, transfer: Dict[str, List[float]],
                diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """The fourth amendment's reading of the exploratory networks. Votes in nothing above."""
    print("\n===== exploratory: the clip on other architectures (does not vote; "
          "plan_floor_clip.md §12) =====")
    out: Dict[str, Any] = {}
    for model in EXPLORATORY_MODELS:
        if not any((DIR / f"e16_floor_clip_{model}_s{s}_{md}.json").exists()
                   for s in SEEDS for md in MODES):
            out[model] = "not run"
            print(f"{model}: not run")
            continue
        x_problems: List[str] = []
        x_commits: set = set()
        data = load(model, x_problems, x_commits, diagnostics)
        if len(x_commits) > 1:
            x_problems.append(f"the files come from {len(x_commits)} commits: {sorted(x_commits)}")
        grids = {"add": LAMBDA_GRID[model], "floor": LAMBDA_GRID[model],
                 **{a: CLIP_GRID for a in CLIP_ARMS}}
        # Pairing never crosses networks, so one commit per network is what matters; a commit
        # other than the voting networks' is reported, not a defect.
        row: Dict[str, Any] = {"problems": x_problems, "commits": sorted(x_commits),
                               "same_commit_as_voting": x_commits == voting_commits}
        per_arm: Dict[str, List[str]] = {a: [] for a in CLIP_ARMS}
        for mode in MODES:
            chosen = {family: select(data, mode, family, values)[0]
                      for family, values in grids.items()}
            row[f"{mode}|chosen"] = chosen
            print(f"{model:16s} {mode:6s}  chosen on validation: "
                  + ", ".join(f"{k} {v}" for k, v in chosen.items()))
            for family in FAMILIES:
                if chosen["add"] is None or chosen[family] is None:
                    v = "incomplete"
                else:
                    m, se, v = paired(data[(mode, family, chosen[family])],
                                      data[(mode, "add", chosen["add"])])
                    row[f"{mode}|{family}_minus_add"] = {"mean": m, "se": se, "verdict": v}
                    print(f"    {family:9s} - add       = {m:+6.2f} +- {se:4.2f}   -> {v}")
                if family in per_arm:
                    per_arm[family].append(v)
            for family in CLIP_ARMS:
                one_se, two_sample = plateaus(data, mode, family, CLIP_GRID)
                inside = [q for q in transfer.get(family, []) if q in one_se]
                row[f"{mode}|plateau_{family}"] = {
                    "one_se_of_best": one_se, "two_sample": two_sample,
                    "voting_transferable": transfer.get(family, []), "inside": inside}
                print(f"    {family:9s} plateau {one_se}; voting networks' transferable q "
                      f"{transfer.get(family, [])} inside it: {inside}")
        for family, vs in per_arm.items():
            if x_problems or any(v not in ("win", "tie", "lose") for v in vs):
                verdict = "incomplete"
            elif "lose" in vs:
                verdict = "not robust: loses to add in " + ", ".join(
                    md for md, v in zip(MODES, vs) if v == "lose")
            else:
                verdict = f"robust: loses to add in neither mode ({vs.count('win')} of 2 wins)"
            row[f"verdict_{family}"] = verdict
            print(f"  {model}, {family}: {verdict}")
        for p in x_problems:
            print(f"  - {p}")
        out[model] = row
    return out


def main() -> None:
    problems: List[str] = []
    commits: set = set()
    diagnostics: Dict[str, Any] = {}
    report: Dict[str, Any] = {"seeds": SEEDS, "pairs": {}, "problems": problems,
                              "diagnostics": diagnostics}
    verdicts: Dict[str, Dict[Tuple[str, str], str]] = {f: {} for f in FAMILIES}
    between: Dict[Tuple[str, str], Dict[Tuple[str, str], str]] = {p: {} for p in BETWEEN}
    plateau_sets: Dict[str, List[set]] = {f: [] for f in ("add", "floor", *CLIP_ARMS)}
    lr_limited: Dict[Tuple[str, str], str] = {}
    chosen_all: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    val_means: Dict[str, Dict[float, List[float]]] = {f: {} for f in ("add", "floor", *CLIP_ARMS)}
    print(f"E16 decisions | seeds {SEEDS}\n")
    for model in MODELS:
        data = load(model, problems, commits, diagnostics)
        lam_grid = LAMBDA_GRID[model]
        grids = {"add": lam_grid, "floor": lam_grid, **{a: CLIP_GRID for a in CLIP_ARMS}}
        for mode in MODES:
            row: Dict[str, Any] = {}
            chosen: Dict[str, Optional[float]] = {}
            for family, values in grids.items():
                chosen[family], unusable = select(data, mode, family, values)
                if unusable:
                    row[f"unusable_{family}"] = unusable
            chosen_all[(model, mode)] = chosen
            row["chosen"] = chosen
            print(f"{model:16s} {mode:6s}  chosen on validation: "
                  + ", ".join(f"{k} {v}" for k, v in chosen.items()))
            for family in FAMILIES:
                if chosen["add"] is None or chosen[family] is None:
                    verdicts[family][(model, mode)] = "incomplete"
                    continue
                m, se, v = paired(data[(mode, family, chosen[family])],
                                  data[(mode, "add", chosen["add"])])
                verdicts[family][(model, mode)] = v
                row[f"{family}_minus_add"] = {"mean": m, "se": se, "verdict": v}
                print(f"    {family:9s} - add       = {m:+6.2f} +- {se:4.2f}   -> {v}")
            for a, b in BETWEEN:
                if chosen[a] is None or chosen[b] is None:
                    between[(a, b)][(model, mode)] = "incomplete"
                    continue
                m, se, v = paired(data[(mode, a, chosen[a])], data[(mode, b, chosen[b])])
                between[(a, b)][(model, mode)] = v
                row[f"{a}_minus_{b}"] = {"mean": m, "se": se, "verdict": v}
                print(f"    {a:9s} - {b:9s} = {m:+6.2f} +- {se:4.2f}   -> {v}   (rule 6)")
            for family, values in grids.items():
                one_se, two_sample = plateaus(data, mode, family, values)
                plateau_sets[family].append(set(one_se))
                row[f"plateau_{family}"] = {"one_se_of_best": one_se, "two_sample": two_sample}
            for family in val_means:
                for v in grids[family]:
                    if _usable(data.get((mode, family, v))):
                        val_means[family].setdefault(v, []).append(
                            _mean_se([_final_val(data[(mode, family, v)][s]) for s in SEEDS])[0])
            # Rule 5: the lr control, against the main clip arm's cell at the same q.
            ref = data.get((mode, CLIPLR_ARM, CLIPLR_Q))
            ctrl = []
            status = "not limited by lr"
            for f in CLIPLR_FACTORS:
                cell = data.get((mode, "cliplr", float(f)))
                if not (_usable(ref) and _usable(cell)):
                    status = "incomplete"
                    ctrl.append((f, None, None, "incomplete"))
                    continue
                m, se, v = paired(cell, ref)
                ctrl.append((f, m, se, v))
                print(f"    lr x{f:<6.3g} at q={CLIPLR_Q} - lr x1 = {m:+6.2f} +- {se:4.2f}"
                      f"   -> {v}   (rule 5)")
                if v not in ("win", "tie", "lose"):
                    status = "incomplete"
                elif v == "win" and status != "incomplete":
                    status = "limited by lr"
            lr_limited[(model, mode)] = status
            row["lr_control"] = ctrl
            row[f"{CLIPLR_ARM}_rule5"] = status
            report["pairs"][f"{model}|{mode}"] = row
        print()
    if len(commits) > 1:
        problems.append(f"the files come from {len(commits)} different commits: {sorted(commits)}")

    print("\n===== verdicts =====")
    for family in FAMILIES:
        verdict = family_verdict(verdicts[family])
        if family == CLIPLR_ARM:
            limited = sorted(f"{m}/{md}" for (m, md), s in lr_limited.items()
                             if s == "limited by lr")
            unknown = sorted(f"{m}/{md}" for (m, md), s in lr_limited.items()
                             if s == "incomplete")
            if limited:
                verdict += f" -- LIMITED BY LR on {', '.join(limited)} (rule 5)"
            if unknown:
                verdict += f" -- rule 5 incomplete on {', '.join(unknown)}"
        report[f"verdict_{family}"] = verdict
        print(f"Rule 3, {family}: {verdict}")
    for a, b in BETWEEN:
        vs = between[(a, b)]
        report[f"rule6_{a}_minus_{b}"] = {f"{m}|{md}": v for (m, md), v in vs.items()}
        print(f"Rule 6, {a} - {b}: " + ", ".join(f"{m}/{md} {v}" for (m, md), v in vs.items()))
    transfer: Dict[str, List[float]] = {}
    for family, sets in plateau_sets.items():
        complete = len(sets) == len(MODELS) * len(MODES) and all(sets)
        common = sorted(set.intersection(*sets)) if complete else []
        transfer[family] = common
        report[f"transfer_{family}"] = common if complete else "incomplete"
        label = "(for E17)" if family == "floor" else ""
        print(f"Rule 4, {family} {label}: values inside every plateau: "
              f"{common if complete else 'incomplete'}")

    print("\n===== E17 candidates, as E17 defines them =====")
    e17: Dict[str, Any] = {}

    def best_by_validation(family: str, pool: List[float]) -> float:
        return max(pool, key=lambda v: sum(val_means[family][v]) / len(val_means[family][v]))

    for family in (*CLIP_ARMS, "floor"):
        ok_rule3 = report[f"verdict_{family}"].startswith(("better", "equivalent"))
        pool = transfer[family] if family != "floor" else [
            v for v in transfer["floor"] if v in E17_FLOOR_COMMON]
        e17[family] = best_by_validation(family, pool) if ok_rule3 and pool else None
        print(f"  {family:9s}: {e17[family] if e17[family] is not None else 'not a candidate'}")
    if transfer["add"]:
        e17["add"] = best_by_validation("add", transfer["add"])
        e17["add_rule"] = f"transferable (of {transfer['add']}, the best mean validation accuracy)"
    elif all(chosen_all[(m, md)]["add"] is not None for m in MODELS for md in MODES):
        logs = [math.log(chosen_all[(m, md)]["add"]) for m in MODELS for md in MODES]
        e17["add"] = math.exp(sum(logs) / len(logs))
        e17["add_rule"] = "geometric mean of the six selected lambdas"
    else:
        e17["add"], e17["add_rule"] = None, "incomplete"
    print(f"  add      : {e17['add']} ({e17['add_rule']})")
    e17["note"] = ("every E16 arm ran with norm_exact_rescaling=True; a candidate is run the way "
                   "E16 ran it")
    report["e17"] = e17

    report["exploratory"] = exploratory(commits, transfer, diagnostics)

    print("\n===== diagnostics (reported, not voted) =====")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")
    if problems:
        print("\n===== NOT A VALID E16 RESULT UNTIL THESE ARE RESOLVED =====")
        for p in problems:
            print(f"  - {p}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"\nwrote {OUT}")
    if problems and not ALLOW_INCOMPLETE:
        sys.exit(2)


if __name__ == "__main__":
    main()
