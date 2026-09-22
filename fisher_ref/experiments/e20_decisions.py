"""E20's pre-registered rules, applied to its output files.

``docs/reports/plan_e20_s1b_vit_small.md``, section 2.1, is the specification. This script reads the
stage-1 files, refuses to give any verdict on an incomplete or invalid set, and otherwise prints
rules 1-5 per mode and the two verdicts.

Validity, checked first, all of it or no verdict:
  * every (seed, job) file for seeds 0-4 and jobs ekfac, tekfac, adamw exists and holds every
    planned cell, none crashed, none non-finite;
  * the seed-0 ekfac and tekfac files carry a bridge that reproduced E15 bit for bit;
  * one git commit across all files, no dirty tree, no smoke setting, 15 epochs, 4 workers,
    batch 32.

Extension files (``..._<job>_<tag>.json``) are read as part of their job's grid.

Usage: ``python fisher_ref/experiments/e20_decisions.py [directory]`` (default:
fisher_ref/outputs). Exit code 0 with verdicts, 2 when the set is invalid.
"""
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from fisher_ref.experiments.e20_s1b_vit_small import (  # noqa: E402
    ADAMW_GRID,
    LAMBDA_TRANSFERRED,
    MODEL,
    SINGLE_GRID,
    TAU_GRID,
    TAU_TRANSFERRED,
)

SEEDS = range(5)
MODES = ("ekfac", "tekfac")


def _load(directory: Path, seed: int, job: str) -> Tuple[Optional[Dict[str, Any]], List[Dict]]:
    """The job's main file, and its extension files (sorted by name)."""
    main = directory / f"e20_{MODEL}_s{seed}_{job}.json"
    exts = sorted(glob.glob(str(directory / f"e20_{MODEL}_s{seed}_{job}_*.json")))
    load = lambda p: json.load(open(p))  # noqa: E731
    return (load(main) if main.exists() else None), [load(p) for p in exts]


def _planned(job: str) -> List[str]:
    if job == "adamw":
        return [f"adamw|adamw|{lr:g}" for lr in ADAMW_GRID]
    return ([f"{job}|reference|0.003"] + [f"{job}|single|{v:g}" for v in SINGLE_GRID]
            + [f"{job}|s1b|{t:g}" for t in TAU_GRID])


def validate(directory: Path) -> Tuple[List[str], Dict[Tuple[int, str], Dict[str, Any]]]:
    """``(problems, cells)`` with ``cells[(seed, job)]`` the job's cells, extensions merged in."""
    problems: List[str] = []
    cells: Dict[Tuple[int, str], Dict[str, Any]] = {}
    commits = set()
    for seed in SEEDS:
        for job in (*MODES, "adamw"):
            main, exts = _load(directory, seed, job)
            if main is None:
                problems.append(f"missing file: seed {seed}, job {job}")
                continue
            merged = dict(main["cells"])
            for ext in exts:
                merged.update(ext["cells"])
            for f in [main, *exts]:
                p = f.get("provenance", {})
                commits.add(p.get("git_commit"))
                if p.get("git_dirty"):
                    problems.append(f"seed {seed} {job}: dirty tree")
                if p.get("smoke") or p.get("epochs") != 15 or p.get("num_workers") != 4 \
                        or p.get("batch") != 32 or f.get("train_subset") or f.get("grid_limit"):
                    problems.append(f"seed {seed} {job}: non-production settings")
                problems += [f"seed {seed} {job}: {x}" for x in f.get("problems", [])]
            for key in _planned(job):
                c = merged.get(key)
                if c is None:
                    problems.append(f"seed {seed} {job}: missing cell {key}")
                elif c.get("error") or not math.isfinite(c.get("test_acc", float("nan"))) \
                        or not c.get("epoch_val_acc"):
                    problems.append(f"seed {seed} {job}: cell {key} crashed or is not finite")
            if job in MODES and seed == 0 and not (main.get("bridge") or {}).get("reproduced"):
                problems.append(f"seed 0 {job}: the bridge did not reproduce E15")
            cells[(seed, job)] = merged
    if len(commits) > 1:
        problems.append(f"more than one git commit across files: {sorted(map(str, commits))}")
    return problems, cells


def _series(cells, job: str, key: str, metric: str) -> List[float]:
    out = []
    for seed in SEEDS:
        c = cells[(seed, job)][key]
        out.append(100 * (c["test_acc"] if metric == "test" else c["epoch_val_acc"][-1]))
    return out


def _mean(x: List[float]) -> float:
    return sum(x) / len(x)


def _se(x: List[float]) -> float:
    m = _mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1) / len(x))


def paired(a: List[float], b: List[float]) -> Dict[str, Any]:
    """E15 rule 2's criterion: win / tie / loss at 2 standard errors of the paired difference."""
    d = [x - y for x, y in zip(a, b)]
    m, s = _mean(d), _se(d)
    verdict = "win" if m > 2 * s else ("loss" if m < -2 * s else "tie")
    return {"mean": m, "se": s, "verdict": verdict, "per_seed": d}


def plateau(means: Dict[float, float], ses: Dict[float, float]) -> Tuple[List[float], float, bool]:
    """E13's plateau on test: ``(values within one SE of the best, best, best is at an edge)``."""
    ordered = sorted(means)
    best = max(ordered, key=lambda v: means[v])
    pl = [v for v in ordered if means[v] >= means[best] - ses[best]]
    return pl, best, best in (ordered[0], ordered[-1])


def _grid_keys(cells, job: str, arm: str) -> List[float]:
    return sorted({c["value"] for c in cells[(0, job)].values() if c.get("arm") == arm})


def _chosen_on_val(cells, job: str, arm: str) -> float:
    values = _grid_keys(cells, job, arm)
    key = lambda v: f"{job if arm != 'adamw' else 'adamw'}|{arm}|{v:g}"  # noqa: E731
    return max(values, key=lambda v: _mean(_series(cells, job, key(v), "val")))


def decide(directory: Path) -> Dict[str, Any]:
    problems, cells = validate(directory)
    if problems:
        return {"valid": False, "problems": problems}
    out: Dict[str, Any] = {"valid": True, "modes": {}}
    lr_star = _chosen_on_val(cells, "adamw", "adamw")
    lrs = _grid_keys(cells, "adamw", "adamw")
    adamw_at = _series(cells, "adamw", f"adamw|adamw|{lr_star:g}", "test")
    adamw_edge = lr_star in (lrs[0], lrs[-1])
    out["adamw"] = {"chosen_lr": lr_star, "test_mean": _mean(adamw_at), "at_edge": adamw_edge,
                    "grid": lrs}
    for mode in MODES:
        t = lambda arm, v: _series(cells, mode, f"{mode}|{arm}|{v:g}", "test")  # noqa: E731
        s1b_t = t("s1b", TAU_TRANSFERRED)
        taus = _grid_keys(cells, mode, "s1b")
        means = {v: _mean(t("s1b", v)) for v in taus}
        ses = {v: _se(t("s1b", v)) for v in taus}
        pl, best, edge = plateau(means, ses)
        extended = len(taus) > len(TAU_GRID)
        if edge:
            transfer = "unresolved" if extended else "needs the pre-registered extension"
        else:
            transfer = "transfers" if TAU_TRANSFERRED in pl else "does not transfer"
        tau_star = _chosen_on_val(cells, mode, "s1b")
        lam_star = _chosen_on_val(cells, mode, "single")
        out["modes"][mode] = {
            "rule1_helps": paired(s1b_t, t("reference", 3e-3)),
            "rule2_transfer": {"plateau": pl, "best": best, "at_edge": edge, "verdict": transfer},
            "rule3_s1b_vs_single_transferred": paired(s1b_t, t("single", LAMBDA_TRANSFERRED)),
            "rule4_s1b_vs_single_tuned": {**paired(t("s1b", tau_star), t("single", lam_star)),
                                          "tau": tau_star, "lambda": lam_star},
            "rule5_s1b_vs_adamw_tuned": {**paired(s1b_t, adamw_at),
                                         "adamw_lr": lr_star, "adamw_at_edge": adamw_edge},
            "means": {"reference": _mean(t("reference", 3e-3)), "s1b_0.1": _mean(s1b_t),
                      "single_1e-10": _mean(t("single", LAMBDA_TRANSFERRED)),
                      "adamw": _mean(adamw_at)},
        }
    r3 = [out["modes"][m]["rule3_s1b_vs_single_transferred"]["verdict"] for m in MODES]
    r2 = [out["modes"][m]["rule2_transfer"]["verdict"] for m in MODES]
    out["verdict_per_layer"] = ("confirmed" if all(v == "win" for v in r3)
                                else "refuted" if all(v != "win" for v in r3) else "mixed")
    out["verdict_transfer"] = ("transfers" if all(v == "transfers" for v in r2)
                               else "fails to transfer" if all(v == "does not transfer" for v in r2)
                               else "mixed or pending")
    return out


def main() -> None:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "fisher_ref/outputs"
    res = decide(directory)
    if not res["valid"]:
        print("E20: NO VERDICT -- the set of files is not valid:")
        for p in res["problems"]:
            print(f"  - {p}")
        raise SystemExit(2)
    print(f"E20 on {MODEL} | AdamW chosen lr {res['adamw']['chosen_lr']:g}"
          f"{' (AT AN EDGE: run the pre-registered extension)' if res['adamw']['at_edge'] else ''}")
    for mode, r in res["modes"].items():
        print(f"\n{mode}: test means {', '.join(f'{k} {v:.2f}' for k, v in r['means'].items())}")
        for rule in ("rule1_helps", "rule3_s1b_vs_single_transferred", "rule4_s1b_vs_single_tuned",
                     "rule5_s1b_vs_adamw_tuned"):
            x = r[rule]
            print(f"  {rule:34s} {x['mean']:+.2f} +- {x['se']:.2f}  -> {x['verdict']}")
        x = r["rule2_transfer"]
        print(f"  {'rule2_transfer':34s} plateau {x['plateau']} best {x['best']:g} -> {x['verdict']}")
    print(f"\nVERDICT per-layer damping on a held-out network: {res['verdict_per_layer']}")
    print(f"VERDICT tau = 0.1 transfers to ViT-S: {res['verdict_transfer']}")


if __name__ == "__main__":
    main()
