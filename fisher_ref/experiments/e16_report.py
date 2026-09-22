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
          "clipfixed": "#27ae60", "cliplr": "#8e44ad"}
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


def chosen(dec: Dict[str, Any], model: str, mode: str) -> Dict[str, float]:
    if model in dec.get("exploratory", {}) and isinstance(dec["exploratory"][model], dict):
        return dec["exploratory"][model][f"{mode}|chosen"]
    return dec["pairs"][f"{model}|{mode}"]["chosen"]


def paired_diffs(data, dec) -> List[Tuple[str, str, str, float, float]]:
    rows = []
    for model in MODELS:
        cells = data[model]
        for mode in MODES:
            ch = chosen(dec, model, mode)
            base = cells[(mode, "add", float(ch["add"]))]
            for fam in ["floor", *CLIP_ARMS]:
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
    ax.set_title("E16 rules 1-2 (voting networks) and the exploratory reading (*)")
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
            for fam in ["add", "floor", *CLIP_ARMS]:
                if ch.get(fam) is None:
                    continue
                s = cells[(mode, fam, float(ch[fam]))]
                curves = [s[k]["epoch_val_acc"] for k in SEEDS[model]]
                mean = [100 * sum(c[e] for c in curves) / len(curves)
                        for e in range(len(curves[0]))]
                ax.plot(range(1, len(mean) + 1), mean, color=COLORS[fam],
                        label=f"{fam} {ch[fam]:g}")
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
    arms = ["add", "floor", *CLIP_ARMS]
    width = 0.16
    for k, arm in enumerate(arms):
        ys = []
        for model in MODELS:
            w = [c["wall_s"] for (md, a, v), s in data[model].items() if a == arm
                 for c in s.values() if c.get("wall_s")]
            ys.append(st.median(w) / 60 if w else float("nan"))
        ax.bar([i + (k - 2) * width for i in range(len(MODELS))], ys, width,
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
           "value rule 1 selects on validation.*", ""]
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
            for arm in ["add", "floor", *CLIP_ARMS, "cliplr"]:
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


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    dec = json.load(open(DIR / "e16_decisions.json"))
    data = {m: load(m) for m in MODELS}
    rows = paired_diffs(data, dec)
    fig_grids(data, dec)
    fig_forest(rows)
    fig_lr_control(data, dec)
    fig_curves(data, dec)
    fig_clip_diagnostics(data, dec)
    fig_cost(data)
    TABLES.write_text(tables(data, dec, rows))
    print(f"wrote {FIGDIR} and {TABLES}")


if __name__ == "__main__":
    main()
