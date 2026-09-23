"""E16's results as tables and figures, for reading, not for deciding.

The verdicts are ``e16_decisions.py``'s; this script only lays the data out. It reads the merged
files ``fisher_ref/outputs/e16_floor_clip_<model>_s<seed>_<mode>.json`` (five networks) and
``e16_decisions.json``, and writes:

  docs/reports/figures/e16/*.png        the figures
  docs/reports/e16_results_tables.md    every cell's five-seed (three on ResNet-50) mean and
                                        standard error, test and final validation accuracy

Environment variables::

    E16_DIR      where the JSON files are (default: fisher_ref/outputs)
    E16_FIGDIR   default: docs/reports/figures/e16
    E16_TABLES   default: docs/reports/e16_results_tables.md
"""
import json
import math
import os
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIR = Path(os.environ.get("E16_DIR", str(ROOT / "fisher_ref/outputs")))
FIGDIR = Path(os.environ.get("E16_FIGDIR", str(ROOT / "docs/reports/figures/e16")))
TABLES = Path(os.environ.get("E16_TABLES", str(ROOT / "docs/reports/e16_results_tables.md")))

MODELS = ["cnn_gn_cifar", "vit_micro_cifar", "cct_2_3x2_cifar", "resnet20_cifar", "resnet50_cifar"]
SEEDS = {m: [0, 1, 2, 3, 4] for m in MODELS} | {"resnet50_cifar": [0, 1, 2]}
MODES = ["ekfac", "tekfac"]
LAMBDA_ARMS = ["add", "floor"]
CLIP_ARMS = ["clipema", "clip", "clipfixed"]
COLORS = {"add": "#1f4e79", "floor": "#7f7f7f", "clipema": "#c0392b", "clip": "#e67e22",
          "clipfixed": "#27ae60", "cliplr": "#8e44ad", "adam": "#17a2b8", "adamw": "#000000"}
BASELINES = ["adam", "adamw"]
LABEL = {"cnn_gn_cifar": "CNN-GN", "vit_micro_cifar": "ViT-micro", "cct_2_3x2_cifar": "CCT-2/3x2",
         "resnet20_cifar": "ResNet-20*", "resnet50_cifar": "ResNet-50*"}

Cells = Dict[Tuple[str, str, float], Dict[int, Dict[str, Any]]]


def load(model: str) -> Cells:
    out: Cells = {}
    for seed in SEEDS[model]:
        for mode in MODES:
            f = json.load(open(DIR / f"e16_floor_clip_{model}_s{seed}_{mode}.json"))
            for c in f["cells"].values():
                if c["arm"] in ("dupcheck", "repro"):
                    continue
                out.setdefault((mode, c["arm"], float(c["value"])), {})[seed] = c
    return out


def load_baselines(model: str) -> Cells:
    """``(mode, opt, lr)`` keyed like E16's cells, with the mode repeated for both E16 modes so
    every figure can index them the same way. Absent files give an empty dict."""
    out: Cells = {}
    for seed in SEEDS[model]:
        for opt in BASELINES:
            files = [DIR / f"e16b_{model}_s{seed}_{opt}{t}.json"
                     for t in ("", "_ext", "_ext2")]
            for c in (c for f in files if f.exists()
                      for c in json.load(open(f))["cells"].values()):
                if c.get("crashed"):
                    continue
                for mode in MODES:
                    out.setdefault((mode, opt, float(c["lr"])), {})[seed] = c
    # A value is kept only when every seed is there, and a baseline is used only when at least
    # three of its lr values are complete: selecting over a part of the grid would read as a
    # located optimum when the rest is still running.
    out = {k: v for k, v in out.items() if sorted(v) == SEEDS[model]}
    per_opt: Dict[str, int] = {}
    for (_, opt, _) in out:
        per_opt[opt] = per_opt.get(opt, 0) + 1
    return {k: v for k, v in out.items() if per_opt[k[1]] >= 3 * len(MODES)}


def baseline_choice(cells: Cells, mode: str, opt: str):
    """Rule 1 applied to a baseline: the lr with the best seed-mean final validation accuracy."""
    vs = values(cells, mode, opt)
    return max(vs, key=lambda v: stat(cells, (mode, opt, v), "val")[0]) if vs else None


def ms(xs: List[float]) -> Tuple[float, float]:
    m = sum(xs) / len(xs)
    return m, (st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan"))


def acc(c: Dict[str, Any], which: str) -> float:
    return 100 * (c["test_acc"] if which == "test" else c["epoch_val_acc"][-1])


def stat(cells: Cells, key, which="test") -> Tuple[float, float]:
    s = cells[key]
    return ms([acc(s[k], which) for k in sorted(s)])


def values(cells: Cells, mode: str, arm: str) -> List[float]:
    return sorted({v for (md, a, v) in cells if md == mode and a == arm}, reverse=True)


# ---------------------------------------------------------------- figures

def fig_grids(data: Dict[str, Cells], dec: Dict[str, Any]) -> None:
    """Test accuracy along each grid: lambda (add, floor) left, q (three clips) right."""
    fig, axes = plt.subplots(len(MODELS), 4, figsize=(17, 3.3 * len(MODELS)), squeeze=False)
    for i, model in enumerate(MODELS):
        cells = data[model]
        for j, mode in enumerate(MODES):
            axl, axq = axes[i, 2 * j], axes[i, 2 * j + 1]
            for arm in LAMBDA_ARMS:
                vs = values(cells, mode, arm)
                if not vs:
                    continue
                m = [stat(cells, (mode, arm, v))[0] for v in vs]
                e = [2 * stat(cells, (mode, arm, v))[1] for v in vs]
                axl.errorbar(vs, m, yerr=e, marker="o", ms=4, capsize=2, color=COLORS[arm],
                             label=arm)
            axl.set_xscale("log")
            axl.invert_xaxis()
            axl.set_xlabel("λ")
            axl.set_title(f"{LABEL[model]} · {mode} · add / floor", fontsize=9)
            for arm in CLIP_ARMS:
                vs = values(cells, mode, arm)
                m = [stat(cells, (mode, arm, v))[0] for v in vs]
                e = [2 * stat(cells, (mode, arm, v))[1] for v in vs]
                axq.errorbar(vs, m, yerr=e, marker="o", ms=4, capsize=2, color=COLORS[arm],
                             label=arm)
            # add's best test mean, as a reference line on the q panel
            best_add = max(stat(cells, (mode, "add", v))[0] for v in values(cells, mode, "add"))
            axq.axhline(best_add, color=COLORS["add"], ls="--", lw=1, label="best add")
            ch = chosen(dec, model, mode)
            for opt in BASELINES:
                if ch.get(opt) is not None:
                    m_b = stat(cells, (mode, opt, float(ch[opt])))[0]
                    for ax in (axl, axq):
                        ax.axhline(m_b, color=COLORS[opt], ls=":", lw=1.2,
                                   label=f"{opt} (lr {ch[opt]:.2g})")
            axq.set_xlabel("q (fraction clipped)")
            axq.set_title(f"{LABEL[model]} · {mode} · clips", fontsize=9)
            for ax in (axl, axq):
                ax.set_ylabel("test acc (%)")
                ax.grid(alpha=0.3)
            if i == 0:
                axl.legend(fontsize=7)
                axq.legend(fontsize=7)
    fig.suptitle("E16: test accuracy along every grid (mean ± 2 SE over seeds; "
                 "* = exploratory network)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGDIR / "grids_test_acc.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


BASE: Dict[str, Cells] = {}   # filled in main(): the baseline cells per network


def chosen(dec: Dict[str, Any], model: str, mode: str) -> Dict[str, float]:
    if model in dec.get("exploratory", {}) and isinstance(dec["exploratory"][model], dict):
        ch = dict(dec["exploratory"][model][f"{mode}|chosen"])
    else:
        ch = dict(dec["pairs"][f"{model}|{mode}"]["chosen"])
    for opt in BASELINES:
        b = baseline_choice(BASE.get(model, {}), mode, opt)
        if b is not None:
            ch[opt] = b
    return ch


def paired_diffs(data, dec) -> List[Tuple[str, str, str, float, float]]:
    rows = []
    for model in MODELS:
        cells = data[model]
        for mode in MODES:
            ch = chosen(dec, model, mode)
            base = cells[(mode, "add", float(ch["add"]))]
            for fam in ["floor", *CLIP_ARMS, *BASELINES]:
                if ch.get(fam) is None:
                    continue
                s = cells[(mode, fam, float(ch[fam]))]
                d = [acc(s[k], "test") - acc(base[k], "test") for k in SEEDS[model]]
                m, se = ms(d)
                rows.append((model, mode, fam, m, se))
    return rows


def fig_forest(rows) -> None:
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(rows) + 1.5))
    ys = []
    for k, (model, mode, fam, m, se) in enumerate(rows):
        y = len(rows) - k
        ys.append((y, f"{LABEL[model]} {mode} {fam}"))
        verdict = "win" if m > 2 * se else "lose" if m < -2 * se else "tie"
        ax.errorbar(m, y, xerr=2 * se, fmt="o", color=COLORS[fam], ms=5, capsize=2,
                    mfc="white" if verdict == "tie" else COLORS[fam])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks([y for y, _ in ys], [t for _, t in ys], fontsize=7)
    ax.set_xlabel("test accuracy, family − add at their validation-selected values "
                  "(points; bar = 2 SE, paired by seed; open = tie)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("E16 rules 1-2 (voting networks), the exploratory reading (*), and the Adam/AdamW "
                 "references (not voted)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "forest_family_minus_add.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_lr_control(data, dec) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for ax, model in zip(axes, MODELS[:3]):
        cells = data[model]
        for mode, mk in zip(MODES, ("o", "s")):
            fs = [0.1, 1 / 3, 1.0, 3.0]
            m, e = [], []
            for f in fs:
                key = (mode, "clipema", 0.7) if f == 1.0 else (mode, "cliplr", f)
                a, b = stat(cells, key)
                m.append(a)
                e.append(2 * b)
            ax.errorbar(fs, m, yerr=e, marker=mk, capsize=2, label=mode)
        ax.set_xscale("log")
        ax.set_xlabel("lr factor (clipema, q = 0.7)")
        ax.set_ylabel("test acc (%)")
        ax.set_title(LABEL[model])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Rule 5, the learning-rate control on the main clip arm")
    fig.tight_layout()
    fig.savefig(FIGDIR / "lr_control.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_curves(data, dec) -> None:
    """Validation accuracy per epoch, seed mean, at each family's chosen value."""
    fig, axes = plt.subplots(2, len(MODELS), figsize=(4 * len(MODELS), 6.5), squeeze=False)
    for j, model in enumerate(MODELS):
        cells = data[model]
        for i, mode in enumerate(MODES):
            ax = axes[i, j]
            ch = chosen(dec, model, mode)
            for fam in ["add", "floor", *CLIP_ARMS, *BASELINES]:
                if ch.get(fam) is None:
                    continue
                s = cells[(mode, fam, float(ch[fam]))]
                curves = [s[k]["epoch_val_acc"] for k in SEEDS[model]]
                mean = [100 * sum(c[e] for c in curves) / len(curves)
                        for e in range(len(curves[0]))]
                ax.plot(range(1, len(mean) + 1), mean, color=COLORS[fam],
                        ls=":" if fam in BASELINES else "-", label=f"{fam} {ch[fam]:.2g}")
            ax.set_title(f"{LABEL[model]} · {mode}", fontsize=9)
            ax.set_xlabel("epoch")
            ax.set_ylabel("val acc (%)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)
    fig.suptitle("Validation accuracy per epoch at the validation-selected value "
                 "(seed mean)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "val_curves_chosen.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_clip_diagnostics(data, dec) -> None:
    """Per-layer clipped fraction along training, seed 0, the chosen q, per clip arm."""
    fig, axes = plt.subplots(len(CLIP_ARMS), len(MODELS),
                             figsize=(4 * len(MODELS), 3 * len(CLIP_ARMS)), squeeze=False)
    for j, model in enumerate(MODELS):
        cells = data[model]
        ch = chosen(dec, model, "ekfac")
        for i, arm in enumerate(CLIP_ARMS):
            ax = axes[i, j]
            c = cells[("ekfac", arm, float(ch[arm]))][0]
            steps = [r["step"] for r in c["log"]]
            layers = list(c["log"][-1]["layers"])
            for name in layers:
                ys = [r["layers"].get(name, {}).get("clipped_fraction", float("nan"))
                      for r in c["log"]]
                is_norm = "norm" in name or name.startswith("bn") or ".bn" in name
                ax.plot(steps, ys, lw=0.8, alpha=0.8 if is_norm else 0.35,
                        color="#c0392b" if is_norm else "#555555")
            ax.axhline(float(ch[arm]), color="k", ls=":", lw=1)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{LABEL[model]} · ekfac · {arm} q={ch[arm]:g}", fontsize=8)
            ax.set_xlabel("step")
            ax.set_ylabel("clipped fraction")
            ax.grid(alpha=0.3)
    fig.suptitle("Fraction of active coordinates clipped, per hooked layer (seed 0; red = "
                 "normalisation layers; dotted = q)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "clipped_fraction_per_layer.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_cost(data) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    arms = ["add", "floor", *CLIP_ARMS, *BASELINES]
    width = 0.12
    for k, arm in enumerate(arms):
        ys = []
        for model in MODELS:
            w = [c["wall_s"] for (md, a, v), s in data[model].items() if a == arm
                 for c in s.values() if c.get("wall_s")]
            ys.append(st.median(w) / 60 if w else float("nan"))
        ax.bar([i + (k - 3) * width for i in range(len(MODELS))], ys, width,
               color=COLORS[arm], label=arm)
    ax.set_xticks(range(len(MODELS)), [LABEL[m] for m in MODELS])
    ax.set_ylabel("median wall time per 15-epoch run (min)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Cost per run on one h100_1g.10gb slice (measured)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "cost_per_run.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- tables

def tables(data, dec, rows) -> str:
    out = ["# E16 — results tables",
           "",
           "*Generated by `fisher_ref/experiments/e16_report.py` from the 46 merged files; the "
           "verdicts are `e16_decisions.py`'s. Accuracies in %, mean ± standard error over seeds "
           "(5; 3 on ResNet-50). \\* = exploratory network, votes in no rule. **Bold** = the "
           "value rule 1 selects on validation. `adam`/`adamw` are references run afterwards under E16's "
           "protocol (`e16_baselines.py`), lr = baseline_lr x {1/3, 1, 3}, selected the same way; "
           "they vote in no rule.*", ""]
    out += ["## Paired differences at the selected values (rules 1-2)", "",
            "| network | mode | family | selected | family − add (test) | verdict |",
            "|---|---|---|---|---|---|"]
    for model, mode, fam, m, se in rows:
        v = "win" if m > 2 * se else "lose" if m < -2 * se else "tie"
        ch = chosen(dec, model, mode)
        out.append(f"| {LABEL[model]} | {mode} | {fam} | {ch[fam]:g} (add {ch['add']:g}) | "
                   f"{m:+.2f} ± {se:.2f} | {v} |")
    for model in MODELS:
        cells = data[model]
        out += ["", f"## {LABEL[model]} (`{model}`), every cell", ""]
        for mode in MODES:
            ch = chosen(dec, model, mode)
            out += [f"**{mode}**", "", "| arm | value | test | final val | runs | wall (min) |",
                    "|---|---|---|---|---|---|"]
            for arm in ["add", "floor", *CLIP_ARMS, "cliplr", *BASELINES]:
                for v in values(cells, mode, arm):
                    t = stat(cells, (mode, arm, v), "test")
                    va = stat(cells, (mode, arm, v), "val")
                    s = cells[(mode, arm, v)]
                    w = st.median([c["wall_s"] for c in s.values()]) / 60
                    vv = f"**{v:g}**" if ch.get(arm) is not None and float(ch[arm]) == v else f"{v:g}"
                    out.append(f"| {arm} | {vv} | {t[0]:.2f} ± {t[1]:.2f} | "
                               f"{va[0]:.2f} ± {va[1]:.2f} | {len(s)} | {w:.1f} |")
            out.append("")
    return "\n".join(out) + "\n"


# ------------------------------------------------ does the clip pay off? (vs the tuned lambda)

E_FILES = {"cnn_gn_cifar": "e14_seeds_cnn_eigfix", "vit_micro_cifar": "e13_seeds_vit_eigfix",
           "cct_2_3x2_cifar": "e10_seeds_cct_eigfix", "resnet20_cifar": "e14_seeds_resnet20_eigfix"}
CAMPAIGN = {  # benchmark protocol (batch 128, wall-clock budget), shipped lambda: seeds 1-4 (1-2
    # on the regime-B models), seed 0 only for resnet50 (campaign 2's re-run)
    "cnn_gn_cifar": ("seeds/cnn_gn_cifar", [1, 2, 3, 4]),
    "vit_micro_cifar": ("seeds/vit_micro_cifar", [1, 2, 3, 4]),
    "cct_2_3x2_cifar": ("seeds/cct_2_3x2_cifar", [1, 2]),
    "resnet20_cifar": ("seeds/resnet20_cifar", [1, 2]),
    "resnet50_cifar": ("cifar10/resnet50_cifar", [None]),
}


def shipped_cells(model: str, mode: str) -> Dict[int, Dict[str, Any]]:
    """The shipped lambda = 1e-3 under the E protocol (E14/E13/E10's reference cells)."""
    out = {}
    if model not in E_FILES:
        return out
    for seed in SEEDS[model]:
        path = DIR / f"{E_FILES[model]}_s{seed}.json"
        if path.exists():
            c = json.load(open(path))["results"][model]["cells"].get(f"{mode}|shipped|reference")
            if c:
                out[seed] = c
    return out


def e15_cells(model: str, mode: str) -> Dict[int, Dict[str, Any]]:
    """E15's per-layer lambda (S1-b), tau selected on seed-mean final validation accuracy."""
    runs: Dict[float, Dict[int, Dict[str, Any]]] = {}
    for seed in SEEDS[model]:
        path = DIR / f"e15_layer_damping_{model}_s{seed}.json"
        if not path.exists():
            return {}
        for c in json.load(open(path))["cells"].values():
            if c["mode"] == mode and c["arm"] == "s1b":
                runs.setdefault(float(c["value"]), {})[seed] = c
    full = {t: r for t, r in runs.items() if len(r) == len(SEEDS[model])}
    if not full:
        return {}
    best = max(full, key=lambda t: ms([acc(c, "val") for c in full[t].values()])[0])
    return {k: dict(v, _tau=best) for k, v in full[best].items()}


def campaign_row(model: str) -> str:
    """Best Fisher mode's test accuracy against the better of adam/adamw, benchmark protocol."""
    import re
    sub, seeds = CAMPAIGN[model]
    gaps, fisher, base = [], [], []
    for sd in seeds:
        path = ROOT / "benchmarks/outputs" / sub / (f"seed{sd}" if sd is not None else "") / "summary.md"
        if not path.exists():
            return "n/a"
        rows = {}
        for line in path.read_text().splitlines():
            cols = [c.strip() for c in line.split("|")]
            if len(cols) > 7 and re.match(r"^[\d.]+%$", cols[6] or ""):
                rows[cols[1]] = float(cols[6].rstrip("%"))
        b = max(rows[a] for a in ("adam", "adamw") if a in rows)
        fisher.append({a: rows[a] for a in ("diag", "kfac", "ekfac", "tkfac", "tekfac") if a in rows})
        base.append(b)
    modes = sorted(fisher[0], key=lambda a: -sum(f[a] for f in fisher) / len(fisher))
    top = modes[0]
    gaps = [f[top] - b for f, b in zip(fisher, base)]
    m, se = ms(gaps)
    beat = [a for a in modes if all(f[a] > b for f, b in zip(fisher, base))]
    n = f"{len(seeds)} seed{'s' if len(seeds) > 1 else ''}"
    mean = {a: sum(f[a] for f in fisher) / len(fisher) for a in modes}
    return (f"**campaigns 1-2** (benchmark protocol, λ=1e-3, {n}): best mode **{top}** "
            f"{mean[top]:.1f} vs best Adam(W) {ms(base)[0]:.1f}: "
            f"{m:+.1f}{'' if math.isnan(se) else f' ± {se:.1f}'}; modes above Adam(W) on every seed: "
            f"{', '.join(beat) if beat else 'none'} ({', '.join(f'{a} {mean[a]:.1f}' for a in modes)})")


E_EXTRA = {"cct_2_3x2_cifar": ["e11_seeds_cct_kfac"]}   # E11 ran cct's kfac arm


def _select(runs: Dict[Any, Dict[int, Dict[str, Any]]], n: int):
    """Rule 1 over a grid: the value whose seed-mean final validation is best, among complete ones."""
    full = {v: r for v, r in runs.items() if len(r) == n}
    if not full:
        return None, {}
    best = max(full, key=lambda v: ms([acc(c, "val") for c in full[v].values()])[0])
    return best, full[best]


def e14_modes(model: str) -> Dict[str, Tuple[Any, Dict[int, Dict[str, Any]]]]:
    """Every mode E14/E13/E10(+E11) ran, each at its validation-selected lambda (shipped stat)."""
    if model not in E_FILES:
        return {}
    runs: Dict[str, Dict[float, Dict[int, Dict[str, Any]]]] = {}
    for stem in [E_FILES[model], *E_EXTRA.get(model, [])]:
        for seed in SEEDS[model]:
            path = DIR / f"{stem}_s{seed}.json"
            if not path.exists():
                continue
            for key, c in json.load(open(path))["results"][model]["cells"].items():
                mode, _, lam = key.split("|")
                if lam != "reference":
                    runs.setdefault(mode, {}).setdefault(float(lam), {})[seed] = c
    n = max((len(r) for m in runs.values() for r in m.values()), default=0)
    return {mode: _select(r, n) for mode, r in runs.items()}


def e15_modes(model: str) -> Dict[str, Tuple[Any, Dict[int, Dict[str, Any]]]]:
    """Every mode E15 ran with a per-layer lambda (S1-b), each at its validation-selected tau."""
    runs: Dict[str, Dict[float, Dict[int, Dict[str, Any]]]] = {}
    for seed in SEEDS[model]:
        path = DIR / f"e15_layer_damping_{model}_s{seed}.json"
        if not path.exists():
            return {}
        for c in json.load(open(path))["cells"].values():
            if c["arm"] == "s1b":
                runs.setdefault(c["mode"], {}).setdefault(float(c["value"]), {})[seed] = c
    return {mode: _select(r, len(SEEDS[model])) for mode, r in runs.items()}


def best_mode_line(label: str, per_mode, knob: str, best_b) -> str:
    """'label: best mode (score, knob) among N; the others ...; vs best Adam(W) ...'."""
    per_mode = {m: v for m, v in per_mode.items() if v[1]}
    if not per_mode:
        return ""
    by_val = sorted(per_mode, key=lambda m: -ms([acc(c, "val") for c in per_mode[m][1].values()])[0])
    top = by_val[0]
    parts = [f"{m} {ms([acc(c, 'test') for c in per_mode[m][1].values()])[0]:.1f}"
             for m in by_val]
    vs = paired_on(per_mode[top][1], best_b) if best_b else "Adam(W) at this protocol: pending"
    return (f"**{label}**: best mode **{top}** ({knob}={per_mode[top][0]:g}); "
            f"all: {', '.join(parts)}; {top} − best Adam(W): {vs}")


def paired_on(a: Dict[int, Dict[str, Any]], b: Dict[int, Dict[str, Any]]) -> str:
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return "—"
    m, se = ms([acc(a[k], "test") - acc(b[k], "test") for k in seeds])
    tag = "win" if m > 2 * se else "lose" if m < -2 * se else "tie"
    return f"{m:+.2f} ± {se:.2f} **{tag}**" + ("" if len(seeds) == 5 else f" ({len(seeds)} seeds)")


def baseline_note(model: str, opt: str, lr) -> str:
    """' (lr=..., EDGE)' when the selected lr is the top or bottom of the baseline grid."""
    lrs = sorted({float(c["lr"]) for (md, o, _), s_ in BASE.get(model, {}).items() if o == opt
                  for c in s_.values()})
    edge = lrs and (lr == lrs[0] or lr == lrs[-1])
    return f" (lr={lr:.2g}{', **grid edge**' if edge else ''})"


def fmt(cells: Dict[int, Dict[str, Any]], extra: str = "") -> str:
    if not cells:
        return "—"
    m, se = ms([acc(c, "test") for c in cells.values()])
    return f"{m:.2f} ± {se:.2f}{extra}"


def worth_it_table(data, dec) -> str:
    out = [
        "# E16 — does the clip pay off? Clipping against the adapted lambda",
        "",
        "*Generated by `fisher_ref/experiments/e16_report.py`. Test accuracy (%), mean ± standard "
        "error over seeds, all under the **E protocol** (batch 32, 15 epochs, clamped cosine, "
        "`eig_before_rescale`): 5 seeds, 3 on ResNet-20's shipped/E14 cells and on ResNet-50. Every "
        "run of a (network, seed) shares its initialisation and data order, so the differences are "
        "paired by seed; **win/lose** at 2 SE, otherwise tie. Each arm is at the value it selects "
        "on validation. The last column is context from another protocol and is not comparable "
        "number for number.*",
        "",
        "Columns:",
        "- **shipped λ = 1e-3**: the damping campaigns 1 and 2 ran with (E14/E13/E10's reference "
        "cells; shipped normalisation statistic).",
        "- **tuned λ (E14 fix)**: one λ per network, E16's `add` arm (with `norm_exact_rescaling`).",
        "- **per-layer λ (E15)**: `damping=\"layer_relative\"`, τ selected on validation; run on "
        "CNN-GN and ViT-micro only.",
        "- **clip**: `clipema` (E16's main arm) at its selected q; in brackets, the range over the "
        "three clip arms (`clip`, `clipema`, `clipfixed`) at their own selected q.",
        "- **Adam / AdamW**: E16's baselines (`e16_baselines.py`), lr = benchmark `baseline_lr` "
        "x {1/3, 1, 3}, selected on validation. **grid edge** = the selected lr is the grid's end: "
        "the optimum was not located, so the baseline is a lower bound on what Adam(W) reaches.",
        "- **context**, one cell per network: (1) campaigns 1-2, benchmark protocol (batch 128, 30-50 "
        "epochs, wall-clock budget, λ = 1e-3): the best of the five modes by mean test accuracy, its "
        "gap to the better of adam/adamw, the modes above Adam(W) on every seed, and every mode's "
        "mean; (2) E14/E13/E10 (+E11's kfac on CCT), E protocol, one tuned λ per mode: the mode "
        "best on validation, every mode's test mean at its own λ, and that mode against E16's best "
        "Adam(W) baseline, paired by seed; (3) the same for E15's per-layer λ (CNN-GN, ViT-micro). "
        "E14/E15 ran no Adam/AdamW arm; the comparison uses E16's baselines, same protocol and "
        "seeds. E14/E15 used the shipped normalisation statistic.",
        "",
        "| network | mode | shipped λ=1e-3 | tuned λ (E14) | per-layer λ (E15) | clip (3 arms) | "
        "Adam | AdamW | clip − shipped | clip − tuned λ | clip − per-layer λ | clip − best Adam(W) | "
        "campaigns 1-2 (context) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        cells = data[model]
        for mode in MODES:
            ch = chosen(dec, model, mode)
            tuned = cells[(mode, "add", float(ch["add"]))]
            clip = cells[(mode, "clipema", float(ch["clipema"]))]
            three = [stat(cells, (mode, a, float(ch[a])))[0] for a in CLIP_ARMS]
            shipped = shipped_cells(model, mode)
            e15 = e15_cells(model, mode)
            base = {o: cells[(mode, o, float(ch[o]))] for o in BASELINES if ch.get(o) is not None}
            best_b = (max(base.values(), key=lambda s: ms([acc(c, "val") for c in s.values()])[0])
                      if base else {})
            tau = " (τ=%g)" % next(iter(e15.values()))["_tau"] if e15 else ""
            lam_note = " (λ=%g)" % ch["add"]
            q_note = " (q=%g)" % ch["clipema"]
            if mode == "ekfac":
                lines = [campaign_row(model),
                         best_mode_line("E14 (tuned λ, E protocol)", e14_modes(model), "λ", best_b),
                         best_mode_line("E15 (per-layer λ, E protocol)", e15_modes(model), "τ",
                                        best_b)]
                context = "<br>".join(x for x in lines if x)
            else:
                context = "〃"
            cols = [LABEL[model], mode, fmt(shipped), fmt(tuned, lam_note), fmt(e15, tau),
                    fmt(clip, q_note) + " [%.2f–%.2f]" % (min(three), max(three)),
                    *(fmt(base[o], baseline_note(model, o, float(ch[o]))) if o in base else "—"
                      for o in BASELINES),
                    paired_on(clip, shipped), paired_on(clip, tuned), paired_on(clip, e15),
                    paired_on(clip, best_b), context]
            out.append("| " + " | ".join(cols) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    dec = json.load(open(DIR / "e16_decisions.json"))
    data = {m: load(m) for m in MODELS}
    for m in MODELS:
        BASE[m] = load_baselines(m)
        data[m].update(BASE[m])
    rows = paired_diffs(data, dec)
    fig_grids(data, dec)
    fig_forest(rows)
    fig_lr_control(data, dec)
    fig_curves(data, dec)
    fig_clip_diagnostics(data, dec)
    fig_cost(data)
    TABLES.write_text(tables(data, dec, rows))
    (TABLES.parent / "e16_vs_lambda.md").write_text(worth_it_table(data, dec))
    print(f"wrote {FIGDIR} and {TABLES}")


if __name__ == "__main__":
    main()
