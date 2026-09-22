"""E19's pre-registered decision rules, applied to the finished runs.

``docs/reports/plan_lambda_dominance.md``, section "E19 -- pre-registered", states the rules; this
script applies them and chooses no threshold of its own. It reads E19's files,
``fisher_ref/outputs/e19_layer_damping_<model>_s<seed>_<mode>.json``, and E16's,
``e16_floor_clip_<model>_s<seed>_<mode>.json``: E19 reruns neither E16's single-lambda grid nor its
clips, and pairs its S1-b cells with them by seed. Every statistic is E16's own function
(``e16_decisions.select``, ``paired``, ``plateaus``), so the two experiments are read the same way.

**Rule 0, checked before anything is read.**
  0a  the seed-0 bridge cell against E16's stored seed-0 add cell at the same lambda, on E16's
      determinism fields: bit-identical or not. Reported; it decides nothing by itself.
  0b  held - add, paired over every seed, at E16's selected lambda: must be a tie at 2 SE. If it is,
      rules 2 and 3 use E16's add and clip cells. If not, on that network rule 2 compares S1-b with
      held instead, and rule 3 gives no verdict.
  0c  completeness: one commit and a clean tree for every E19 file; not a smoke; production
      settings; every planned cell present once, none unplanned, none crashed; every cell ran with
      exactly the settings the driver plans for it; the held lambda equals E16's selection. E16's
      files go through E16's own checks (``e16_decisions.load``). A family with a missing or
      unusable value gets "incomplete", never a verdict, and the script exits with status 2 unless
      ``E19_ALLOW_INCOMPLETE=1``.

**Rules.** 1 selection on validation, verdict on test. 2 S1-b - add. 3 S1-b - clipema (clip and
clipfixed reported beside it). 4 is tau = 0.1 inside S1-b's plateau (E16 rule 4's definition), on
the six new pairs, and, reported, on E15's four ekfac/tekfac pairs. 5 S1-b - netadapt (CCT,
ResNet-20). Summaries of rules 2 and 3 over the six pairs in E16 rule 3's words. Reported, not voted:
wdctrl - s1b at tau = 0.1 on CCT.

Environment variables::

    E19_DIR               where E19's JSON files are (default: fisher_ref/outputs)
    E16_DIR               where E16's are (read by e16_decisions; default: fisher_ref/outputs)
    E15_DIR               where E15's are (default: fisher_ref/outputs)
    E19_OUT               output path (default: <E19_DIR>/e19_decisions.json)
    E19_ALLOW_INCOMPLETE  "1": exit 0 even when something is incomplete (the verdicts still say so)
"""
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks.common.runner import discover_benchmarks  # noqa: E402
from fisher_ref.experiments import e16_decisions as e16  # noqa: E402
from fisher_ref.experiments.e5_unhooked_freeze_control import unhooked_parameter_names  # noqa: E402
from fisher_ref.experiments.e16_floor_clip import CLIP_ARMS, LAMBDA_GRID, same_run  # noqa: E402
from fisher_ref.experiments.e19_layer_damping_transfer import (  # noqa: E402
    BRIDGE_LAMBDA,
    HELD_LAMBDA,
    MODES,
    NETADAPT_NETWORKS,
    NETWORKS,
    SEEDS,
    TAU_GRID,
    WDCTRL_TAU,
    cell_key,
    plan_cells,
)

DIR = Path(os.environ.get("E19_DIR", str(ROOT / "fisher_ref/outputs")))
E15_DIR = Path(os.environ.get("E15_DIR", str(ROOT / "fisher_ref/outputs")))
OUT = Path(os.environ.get("E19_OUT", str(DIR / "e19_decisions.json")))
ALLOW_INCOMPLETE = os.environ.get("E19_ALLOW_INCOMPLETE", "0") == "1"
TRANSFER_TAU = 0.1
E15_NETWORKS = ("cnn_gn_cifar", "vit_micro_cifar")
E15_SEEDS = [0, 1, 2, 3, 4]
REPORTED_CLIPS = ("clip", "clipfixed")    # beside rule 3's clipema, not voted
SETTINGS = ("lam", "lr", "wd", "freeze", "overrides")
VERDICTS = ("win", "tie", "lose")

Cell = Dict[str, Any]
Series = Dict[int, Cell]          # seed -> run
Key = Tuple[str, str, float]      # (mode, arm, value)
Pair = Tuple[str, str]            # (network, mode)


@lru_cache(maxsize=None)
def _network(model: str) -> Tuple[Any, Tuple[str, ...]]:
    """A network's benchmark hyperparameters and the parameters no hooked module owns."""
    bench = discover_benchmarks()[model]
    return bench.hparams, tuple(unhooked_parameter_names(bench))


def planned_cells(model: str, seed: int, mode: str) -> Dict[str, Cell]:
    """``{cell key: planned settings}`` for one (network, seed, mode) job, from the driver itself."""
    hp, frozen = _network(model)
    cells = plan_cells(model, mode, seed, hp.lam, hp.lr, hp.weight_decay, hp.decoupled_wd,
                       list(frozen))
    return {cell_key(mode, c["arm"], c["value"]): c for c in cells}


def load_e19(model: str, problems: List[str], commits: set,
             plans: Optional[Dict[Tuple[int, str], Dict[str, Cell]]] = None
             ) -> Tuple[Dict[Key, Series], Dict[str, Cell]]:
    """``({(mode, arm, value): {seed: run}}, {mode: seed-0 bridge cell})``, every file checked. Each
    defect found is appended to ``problems``. ``plans`` may pass the planned cells in (tests)."""
    data: Dict[Key, Series] = {}
    bridges: Dict[str, Cell] = {}
    for seed in SEEDS[model]:
        for mode in MODES:
            path = DIR / f"e19_layer_damping_{model}_s{seed}_{mode}.json"
            if not path.exists():
                problems.append(f"{path.name}: missing")
                continue
            e19 = json.load(open(path))
            provs = e19.get("provenance") or []
            checks = {
                "network/seed/mode": (e19.get("model"), e19.get("seed"), e19.get("mode"))
                == (model, seed, mode),
                "not a smoke": bool(provs) and all(p and p.get("smoke") is False for p in provs),
                "15 epochs": e19.get("epochs") == 15, "batch 32": e19.get("batch_size") == 32,
                "every arm": e19.get("arms") == ["bridge", "held", "s1b", "netadapt", "wdctrl"],
                "full grids": e19.get("grid_limit") is None,
                "full split": e19.get("train_subset") is None,
                "4 workers": bool(provs) and all(p and p.get("num_workers") == 4 for p in provs),
                "clean tree": bool(provs) and all(p and p.get("git_dirty") is False
                                                  for p in provs),
                "every shard merged": not e19.get("missing_shards"),
                "nothing unplanned": not e19.get("unplanned_cells"),
            }
            problems += [f"{path.name}: fails '{name}'" for name, ok in checks.items() if not ok]
            commits.update(p.get("git_commit") for p in provs if p)
            plan = (plans or {}).get((seed, mode)) or planned_cells(model, seed, mode)
            absent = [k for k in plan if k not in e19["cells"]]
            if absent:
                problems.append(f"{path.name}: {len(absent)} planned cells absent, e.g. {absent[:3]}")
            for key, c in e19["cells"].items():
                if key not in plan:
                    problems.append(f"{path.name}: {key} is not a planned cell")
                    continue
                if c.get("crashed"):
                    problems.append(f"{path.name}: {key} crashed ({c.get('error')})")
                    continue
                differ = [s for s in SETTINGS if c.get(s) != plan[key][s]]
                if differ:
                    problems.append(f"{path.name}: {key} ran with settings other than planned "
                                    f"({', '.join(differ)})")
                if c["arm"] == "bridge":
                    bridges[mode] = c
                    continue
                data.setdefault((c["mode"], c["arm"], float(c["value"])), {})[seed] = c
    return data, bridges


def load_e15(model: str) -> Dict[Key, Series]:
    """E15's S1-b cells of one network, ``{(mode, "s1b", tau): {seed: run}}``. Empty if absent."""
    out: Dict[Key, Series] = {}
    for seed in E15_SEEDS:
        path = E15_DIR / f"e15_layer_damping_{model}_s{seed}.json"
        if not path.exists():
            continue
        for c in json.load(open(path))["cells"].values():
            if c.get("arm") == "s1b":
                out.setdefault((c["mode"], "s1b", float(c["value"])), {})[seed] = c
    return out


def summarise(vs: Dict[Pair, str]) -> str:
    """Rules 2 and 3 over the six pairs, in E16 rule 3's words. Anything but win/tie/lose in any
    pair makes the summary incomplete."""
    bad = sorted(f"{m}/{md}: {v}" for (m, md), v in vs.items() if v not in VERDICTS)
    if bad or len(vs) != len(NETWORKS) * len(MODES):
        return "incomplete (" + "; ".join(bad or ["pairs missing"]) + ")"
    by_model: Dict[str, List[str]] = {}
    for (model, _), v in vs.items():
        by_model.setdefault(model, []).append(v)
    losses = sum(1 for m in by_model.values() if "lose" in m)
    wins_both = sum(1 for m in by_model.values() if m.count("win") == len(MODES))
    if losses == 0 and wins_both >= 2:
        return "better"
    if losses >= 2:
        return "worse"
    if losses == 0:
        return "equivalent"
    return "mixed"


def _cmp(a: Optional[Series], b: Optional[Series], seeds: List[int]) -> Dict[str, Any]:
    """``a - b`` on test accuracy, paired by seed (E16's ``paired``), or incomplete."""
    if not (e16._usable(a, seeds) and e16._usable(b, seeds)):
        return {"verdict": "incomplete"}
    m, se, v = e16.paired(a, b, seeds)   # type: ignore[arg-type]
    return {"mean": m, "se": se, "verdict": v}


def read_pair(model: str, mode: str, e16_data: Dict[Key, Series], e19_data: Dict[Key, Series],
              bridge: Optional[Cell], problems: List[str]) -> Dict[str, Any]:
    """Rules 0a, 0b and 1-5 for one (network, mode)."""
    seeds = list(SEEDS[model])
    row: Dict[str, Any] = {}
    # Rule 1: every family at its validation-selected value. add and the clips are E16's own.
    chosen: Dict[str, Optional[float]] = {
        "add": e16.select(e16_data, mode, "add", LAMBDA_GRID[model], seeds)[0]}
    for arm in CLIP_ARMS:
        if arm in e16.arms_of(model):
            chosen[arm] = e16.select(e16_data, mode, arm, e16.clip_grid_of(model), seeds)[0]
    chosen["s1b"] = e16.select(e19_data, mode, "s1b", TAU_GRID[model], seeds)[0]
    if model in NETADAPT_NETWORKS:
        chosen["netadapt"] = e16.select(e19_data, mode, "netadapt", TAU_GRID[model], seeds)[0]
    row["chosen"] = chosen
    held_lam = HELD_LAMBDA[(model, mode)]
    if chosen["add"] is not None and chosen["add"] != held_lam:
        problems.append(f"{model}/{mode}: held ran at {held_lam:g}, but E16's selection is "
                        f"{chosen['add']:g}")

    def series(source: Dict[Key, Series], arm: str, value: Optional[float]) -> Optional[Series]:
        return None if value is None else source.get((mode, arm, float(value)))

    s1b = series(e19_data, "s1b", chosen["s1b"])
    add = series(e16_data, "add", chosen["add"])
    held = series(e19_data, "held", held_lam)

    # 0a: reported, decides nothing by itself.
    row["0a_bridge"] = same_run(bridge, (e16_data.get((mode, "add", BRIDGE_LAMBDA[(model, mode)]))
                                         or {}).get(0))
    # 0b: decides which cells rules 2 and 3 may use.
    row["0b_held_minus_add"] = ob = _cmp(held, add, seeds)
    if ob["verdict"] == "tie":
        row["rule2_s1b_minus_add"] = _cmp(s1b, add, seeds)
        row["rule2_partner"] = "add (E16)"
        row["rule3_s1b_minus_clipema"] = _cmp(s1b, series(e16_data, "clipema",
                                                          chosen.get("clipema")), seeds)
    elif ob["verdict"] in ("win", "lose"):
        row["rule2_s1b_minus_add"] = _cmp(s1b, held, seeds)
        row["rule2_partner"] = "held (E19; 0b is not a tie)"
        row["rule3_s1b_minus_clipema"] = {
            "verdict": f"no verdict (0b: held - add = {ob['mean']:+.2f} +- {ob['se']:.2f})"}
    else:
        row["rule2_s1b_minus_add"] = {"verdict": "incomplete"}
        row["rule2_partner"] = f"none (0b: {ob['verdict']})"
        row["rule3_s1b_minus_clipema"] = {"verdict": "incomplete"}
    for arm in REPORTED_CLIPS:
        if arm in chosen:
            row[f"reported_s1b_minus_{arm}"] = _cmp(s1b, series(e16_data, arm, chosen[arm]), seeds)
    # Rule 4: E16 rule 4's plateau on test accuracy.
    one_se, two_sample = e16.plateaus(e19_data, mode, "s1b", TAU_GRID[model], seeds)
    row["rule4_plateau"] = {"one_se_of_best": one_se, "two_sample": two_sample,
                            "tau_0.1_inside": (TRANSFER_TAU in one_se) if one_se else None}
    # Rule 5.
    if model in NETADAPT_NETWORKS:
        row["rule5_s1b_minus_netadapt"] = _cmp(s1b, series(e19_data, "netadapt",
                                                           chosen.get("netadapt")), seeds)
    # Reported: the benchmark's decoupled decay under S1-b.
    if (mode, "wdctrl", WDCTRL_TAU) in e19_data:
        row["reported_wdctrl_minus_s1b_at_0.1"] = _cmp(
            e19_data[(mode, "wdctrl", WDCTRL_TAU)], e19_data.get((mode, "s1b", WDCTRL_TAU)), seeds)
    return row


def _fmt(d: Dict[str, Any]) -> str:
    if "mean" in d:
        return f"{d['mean']:+6.2f} +- {d['se']:4.2f}  -> {d['verdict']}"
    return str(d.get("verdict"))


def main() -> None:
    problems: List[str] = []
    commits: set = set()
    diagnostics: Dict[str, Any] = {}
    report: Dict[str, Any] = {"pairs": {}, "problems": problems, "diagnostics": diagnostics}
    rule2: Dict[Pair, str] = {}
    rule3: Dict[Pair, str] = {}
    inside: Dict[Pair, Optional[bool]] = {}
    print("E19 decisions\n")
    for model in NETWORKS:
        e16_problems: List[str] = []
        e16_commits: set = set()
        e16_data = e16.load(model, e16_problems, e16_commits, diagnostics)
        if len(e16_commits) > 1:
            e16_problems.append(f"{model}: E16's files come from {len(e16_commits)} commits")
        problems += [f"E16 {p}" for p in e16_problems]
        e19_data, bridges = load_e19(model, problems, commits)
        for mode in MODES:
            row = read_pair(model, mode, e16_data, e19_data, bridges.get(mode), problems)
            report["pairs"][f"{model}|{mode}"] = row
            rule2[(model, mode)] = row["rule2_s1b_minus_add"]["verdict"]
            rule3[(model, mode)] = row["rule3_s1b_minus_clipema"]["verdict"]
            inside[(model, mode)] = row["rule4_plateau"]["tau_0.1_inside"]
            print(f"{model:16s} {mode:6s}  chosen on validation: "
                  + ", ".join(f"{k} {v}" for k, v in row["chosen"].items()))
            print(f"    0a bridge bit-identical to E16: {row['0a_bridge'].get('identical')}")
            print(f"    0b held - add          = {_fmt(row['0b_held_minus_add'])}")
            print(f"    2  s1b - {row['rule2_partner']:10s} = {_fmt(row['rule2_s1b_minus_add'])}")
            print(f"    3  s1b - clipema       = {_fmt(row['rule3_s1b_minus_clipema'])}")
            for arm in REPORTED_CLIPS:
                if f"reported_s1b_minus_{arm}" in row:
                    print(f"       s1b - {arm:9s}     = {_fmt(row[f'reported_s1b_minus_{arm}'])}"
                          "   (reported)")
            print(f"    4  plateau {row['rule4_plateau']['one_se_of_best']}; "
                  f"tau = 0.1 inside: {row['rule4_plateau']['tau_0.1_inside']}")
            if "rule5_s1b_minus_netadapt" in row:
                print(f"    5  s1b - netadapt      = {_fmt(row['rule5_s1b_minus_netadapt'])}")
            if "reported_wdctrl_minus_s1b_at_0.1" in row:
                print(f"       wdctrl - s1b(0.1)   = "
                      f"{_fmt(row['reported_wdctrl_minus_s1b_at_0.1'])}   (reported)")
        print()
    if len(commits) > 1:
        problems.append(f"E19's files come from {len(commits)} different commits: {sorted(commits)}")

    print("===== verdicts =====")
    report["rule2_summary"] = summarise(rule2)
    report["rule3_summary"] = summarise(rule3)
    print(f"Rule 2, S1-b against the tuned single lambda: {report['rule2_summary']}")
    print(f"Rule 3, S1-b against the clip:                {report['rule3_summary']}")
    if any(v is None for v in inside.values()):
        report["rule4"] = "incomplete"
    else:
        report["rule4"] = ("tau = 0.1 transfers" if all(inside.values()) else
                           "tau = 0.1 does not transfer; inside the plateau of: "
                           + ", ".join(f"{m}/{md}" for (m, md), v in inside.items() if v))
    print(f"Rule 4: {report['rule4']}")

    e15_rows: Dict[str, Any] = {}
    for model in E15_NETWORKS:
        data = load_e15(model)
        for mode in MODES:
            one_se, _ = e16.plateaus(data, mode, "s1b", sorted(
                {v for (md, _, v) in data if md == mode}, reverse=True), E15_SEEDS)
            e15_rows[f"{model}|{mode}"] = {"one_se_of_best": one_se,
                                           "tau_0.1_inside": (TRANSFER_TAU in one_se)
                                           if one_se else None}
    report["rule4_e15_pairs_reported"] = e15_rows
    print("Rule 4, E15's pairs (reported): " + ", ".join(
        f"{k} {v['tau_0.1_inside']}" for k, v in e15_rows.items()))

    if problems:
        print("\n===== NOT A VALID E19 RESULT UNTIL THESE ARE RESOLVED =====")
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
