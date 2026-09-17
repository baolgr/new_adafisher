"""Figures for the lambda-vs-curvature finding (validation_noise_investigation.md, Steps 7-8).

    PYTHONPATH=src .venv/bin/python docs/reports/figures/make_figures.py

Reads only files already on disk; writes fig*.png next to this script. Figure 4 is
skipped until fisher_ref/outputs/warmup_sgd_baseline_v2.json exists.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

# Matplotlib defaults: the default colour cycle, no custom style.
INK, INK2, MUTED, GRID = "black", "black", "gray", "lightgray"
MODE_COLOR = {"diag": "C0", "kfac": "C1", "ekfac": "C2", "tkfac": "C3", "tekfac": "C4"}
SGD_COLOR = "C5"
LAM_RAMP = {"1e-5": "C0", "1e-3": "C1", "1e-1": "C2"}
MODES = ["diag", "kfac", "ekfac", "tkfac", "tekfac"]
GAMMA0, GAMMA1 = 0.92, 0.008
RHO, C = 1 - GAMMA0, GAMMA1 / GAMMA0



def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png")


# --------------------------------------------------------------------------------------------
# Figure 1: largest curvature value / lambda, every model x mode
# --------------------------------------------------------------------------------------------
def fig1() -> None:
    d = json.load(open(ROOT / "fisher_ref/outputs/curvature_max_per_layer.json"))["results"]
    lam = {m: 1e-3 for m in d} | {"vit_micro_cifar": 3e-3, "cct_2_3x2_cifar": 3e-3}
    models = list(d)
    floor = 1e-8
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(models))[::-1]
    offsets = np.linspace(-0.28, 0.28, len(MODES))
    for j, mode in enumerate(MODES):
        vals = [max(r["max_over_lambda"] for r in d[m][mode]) for m in models]
        shown = [max(v, floor) for v in vals]
        ax.scatter(shown, y + offsets[j], s=46, color=MODE_COLOR[mode], zorder=3, label=mode)
    for i, m in enumerate(models):
        bound = C ** 2 / lam[m]
        ax.plot([bound, bound], [y[i] - 0.4, y[i] + 0.4], color=MODE_COLOR["diag"], lw=1,
                ls=(0, (2, 2)), zorder=2)
    ax.axvline(1.0, color=INK, lw=1.2, zorder=2)
    ax.text(1.12, y[0] + 0.45, "curvature = λ", color=INK, fontsize=9, va="bottom")
    ax.text(C ** 2 / 1e-3, y[-1] - 0.62, "diag ceiling (proved)", color=MODE_COLOR["diag"],
            fontsize=8.5, ha="center", va="top")
    ax.set_xscale("log")
    ax.set_xlim(floor / 2, 10)
    ax.set_ylim(-1, len(models))
    ax.set_yticks(y, [f"{m}  (λ={lam[m]:g})" for m in models])
    ax.set_xlabel("largest stored curvature value in the network ÷ λ   (log scale)")
    ax.set_title("No direction of any network reaches λ, in any mode\n"
                 + "checkpoint at 50% of training, 1000 re-warm steps, identity-seed residual "
            "removed; values below 1e-8 drawn at 1e-8", fontsize=10)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13), handletextpad=0.2,
              columnspacing=1.2)
    save(fig, "fig1_max_curvature_over_lambda")


# --------------------------------------------------------------------------------------------
# Figure 2: (a) proved diag ceiling vs number of factor updates, (b) applied divisor during warmup
# --------------------------------------------------------------------------------------------
def fig2() -> None:
    d = json.load(open(ROOT / "fisher_ref/outputs/curvature_max_per_layer.json"))["results"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    k = np.arange(1, 11)
    bk = RHO ** k + C * (1 - RHO ** k)
    a.plot(k, bk ** 2 / 1e-3, color=MODE_COLOR["diag"], marker="o", ms=5,
           label="proved ceiling  b_k² / λ")
    a.axhline(C ** 2 / 1e-3, color=MODE_COLOR["diag"], lw=1, ls=(0, (2, 2)))
    a.text(5.5, C ** 2 / 1e-3 * 1.35, "asymptote: 7.56 % of λ", color=MODE_COLOR["diag"],
           ha="center", va="bottom", fontsize=9)
    ae = max(r["max_over_lambda"] for r in d["mnist_autoencoder"]["diag"])
    a.annotate("mnist_autoencoder (stalled run)", (10, ae), xytext=(4.0, ae * 6), fontsize=8.5,
               color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    measured = [max(r["max_over_lambda"] for r in d[m]["diag"])
                for m in d if m not in ("vit_micro_cifar", "cct_2_3x2_cifar")]
    a.scatter([10] * len(measured), measured, s=40, color=INK2, zorder=4, label="measured max, 5 networks at λ=1e-3")
    a.axhline(1, color=INK, lw=1.2)
    a.text(1, 1.25, "curvature = λ", color=INK, fontsize=9)
    a.set_yscale("log")
    a.set_xlim(0.5, 11.6)
    a.set_xticks(k)
    a.set_xlabel("number of factor updates k  (one every 100 steps)")
    a.set_ylabel("largest possible curvature ÷ λ")
    a.set_title("(a) diag: a data-independent ceiling")
    a.legend(loc="upper right", fontsize=8.5)

    t = np.arange(0, 700)
    kk = t // 100 + 1
    lam = 1e-3
    div_kfac = (RHO ** kk + lam ** 0.5) ** 2
    div_ekfac = lam + RHO ** (t // 100)
    b.step(t, div_ekfac / lam, where="post", color=MODE_COLOR["ekfac"], label="ekfac / tekfac")
    b.step(t, div_kfac / lam, where="post", color=MODE_COLOR["kfac"], label="kfac")
    b.axhline(1, color=INK, lw=1.2)
    b.text(700, 1.3, "λ", color=INK, fontsize=10, ha="right")
    b.set_yscale("log")
    b.set_xlabel("training step")
    b.set_ylabel("applied divisor ÷ λ")
    b.set_title("(b) the start-up leftover, not λ, sets the step early on")
    b.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout(w_pad=3)
    save(fig, "fig2_diag_ceiling_and_warmup")


# --------------------------------------------------------------------------------------------
# Figure 3: lambda sweep on mnist_autoencoder, step-by-step training loss
# --------------------------------------------------------------------------------------------
def fig3() -> None:
    base = ROOT / "benchmarks/outputs/sweeps/mnist_autoencoder_lam"
    loss = {}
    for lam_dir in LAM_RAMP:
        per = defaultdict(list)
        with open(base / lam_dir / "records.csv") as f:
            for row in csv.DictReader(f):
                per[row["arm"]].append(float(row["loss"]))
        loss[lam_dir] = per
    arms = MODES + ["adam"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
    n_show, win = 800, 15
    for ax, arm in zip(axes.flat, arms):
        for lam_dir, color in LAM_RAMP.items():
            s = np.array(loss[lam_dir][arm][:n_show])
            sm = np.convolve(s, np.ones(win) / win, mode="valid")
            ax.plot(np.arange(len(sm)) + win // 2, sm, color=color, label=f"λ = {lam_dir}")
        for x in (100, 200, 300):
            ax.axvline(x, color=GRID, lw=0.8, zorder=0)
        ax.set_title(arm)
    for ax in axes[1]:
        ax.set_xlabel("training step")
    for ax in axes[:, 0]:
        ax.set_ylabel("training loss (MSE, 15-step mean)")
    axes[0, 0].legend(loc="upper right", fontsize=8.5)
    fig.suptitle("mnist_autoencoder, first 800 steps: λ=1e-5 and λ=1e-3 overlap; only λ=1e-1 "
                 "slows the start\n(λ=1e-5 is hidden under λ=1e-3 where they coincide; adam ignores λ and is the control)")
    fig.tight_layout()
    save(fig, "fig3_lambda_sweep_first_steps")


# --------------------------------------------------------------------------------------------
# Figure 4: momentum-SGD with matched warm-up vs the real Fisher modes
# --------------------------------------------------------------------------------------------
def _sd(x: np.ndarray) -> float:
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def _load_runs(files) -> dict:
    """model -> label -> run, merged over several warmup_sgd_baseline.py outputs."""
    res: dict = defaultdict(dict)
    for f in files:
        for model, runs in json.load(open(f))["results"].items():
            for r in runs:
                res[model].setdefault(r["label"], r)
    return res


def _arm(by: dict, prefix: str) -> list:
    return [by[k] for k in sorted(by) if k.startswith(prefix + "_seed")]


def fig4() -> None:
    """Fisher mode vs momentum-SGD with that mode's warm-up schedule, at each bench's own lambda
    (seed s of both arms shares init and data order)."""
    out = ROOT / "fisher_ref/outputs"
    files = sorted(out.glob("warmup_sgd_v3_*.json")) + sorted(out.glob("warmup_sgd_baseline_small.json"))
    if not files:
        print("skip fig4: no data")
        return
    res = _load_runs(files)
    models = [m for m in ("mlp_ln_mnist", "cnn_gn_cifar", "resnet20_cifar") if m in res]
    modes = [m for m in MODES if any(_arm(res[x], m) and _arm(res[x], f"sgd_{m}") for x in models)]
    fig, axes = plt.subplots(len(modes), len(models), figsize=(4 * len(models), 3.2 * len(modes)),
                             squeeze=False)
    rows = []
    for i, mode in enumerate(modes):
        for j, model in enumerate(models):
            ax = axes[i][j]
            by = res[model]
            fis, sgd = _arm(by, mode), _arm(by, f"sgd_{mode}")
            if not fis or not sgd:
                ax.set_visible(False)
                continue
            for runs, color, name in ((fis, MODE_COLOR[mode], mode),
                                      (sgd, SGD_COLOR, "SGD, same warm-up")):
                acc = np.array([r["epoch_val_acc"] for r in runs]) * 100
                ep = np.arange(1, acc.shape[1] + 1)
                if len(runs) > 1:
                    ax.fill_between(ep, acc.min(0), acc.max(0), color=color, alpha=0.12, lw=0)
                ax.plot(ep, acc.mean(0), color=color, label=f"{name} (n={len(runs)})")
            ax.set_title(f"{model} — {mode}")
            ax.set_xlabel("epoch")
            ax.set_ylabel("validation accuracy (%)")
            ax.legend(fontsize=7.5)
            fa = np.array([r["test_acc"] for r in fis]) * 100
            sa = np.array([r["test_acc"] for r in sgd]) * 100
            paired = np.array([by[f"sgd_{mode}_seed{r['seed']}"]["test_acc"] * 100 - r["test_acc"] * 100
                               for r in fis if f"sgd_{mode}_seed{r['seed']}" in by])
            se = np.sqrt(np.nan_to_num(_sd(fa) ** 2) / len(fa) + np.nan_to_num(_sd(sa) ** 2) / len(sa))
            z = (sa.mean() - fa.mean()) / se if len(fa) > 1 and len(sa) > 1 else float("nan")
            rows.append((model, mode, fa.mean(), _sd(fa), len(fa), sa.mean(), _sd(sa), len(sa),
                         sa.mean() - fa.mean(), z, paired.mean(), _sd(paired)))
    fig.tight_layout()
    save(fig, "fig4_sgd_equivalence_curves")

    fig, ax = plt.subplots(figsize=(12.5, 0.42 * len(rows) + 1.1))
    ax.axis("off")
    header = ["model", "mode", "Fisher test acc (n)", "SGD test acc (n)", "SGD − Fisher",
              "diff. ÷ s.e.", "paired gap"]
    f = lambda v, fmt: "—" if v != v else format(v, fmt)  # noqa: E731
    cells = [[m, mo, f"{fm:.2f} ± {f(fs, '.2f')} % ({nf})", f"{sm:.2f} ± {f(ss, '.2f')} % ({ns})",
              f"{dm:+.2f} pt", f(z, "+.1f"), f"{pm:+.2f} ± {f(ps, '.2f')} pt"]
             for m, mo, fm, fs, nf, sm, ss, ns, dm, z, pm, ps in rows]
    tab = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.45)
    for (r, _), cell in tab.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Test accuracy, mean ± sd (n seeds), each bench's own λ. |difference ÷ std. error| "
                 "below ~2: not distinguishable; — : needs n ≥ 2", fontsize=10)
    save(fig, "fig4_sgd_equivalence_table")


# --------------------------------------------------------------------------------------------
# Figure 5: cnn_gn_cifar, 2x2 ablation of the SGD stand-in (warm-up schedule x unhooked params)
# --------------------------------------------------------------------------------------------
def fig5() -> None:
    path = ROOT / "fisher_ref/outputs/cnn_gn_ablation.json"
    if not path.exists():
        print("skip fig5: no cnn_gn_ablation.json")
        return
    d = json.load(open(path))
    v3 = ROOT / "fisher_ref/outputs/warmup_sgd_v3_kfac.json"
    kfac = sgd = None
    if v3.exists():
        by = _load_runs([v3])["cnn_gn_cifar"]
        kfac = np.array([r["test_acc"] for r in _arm(by, "kfac")]) * 100
        sgd = np.array([r["test_acc"] for r in _arm(by, "sgd_kfac")]) * 100
    order = sorted(d["cells"], key=lambda c: c["cell"])
    labels = [f"{c['cell']}: {c['schedule']} schedule,\n"
              f"{'GroupNorm divisor 1' if c['respect_hooked'] else 'GroupNorm divisor ≈ λ'}"
              for c in order]
    acc = [c["test_acc"] * 100 for c in order]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(order))[::-1]
    ax.scatter(acc, y, color=SGD_COLOR, zorder=3, label="SGD stand-in, seed 0 (one run per cell)")
    if kfac is not None:
        ax.axvspan(kfac.min(), kfac.max(), color=MODE_COLOR["kfac"], alpha=0.2,
                   label=f"real kfac, min–max over {len(kfac)} seeds")
        ax.axvline(kfac.mean(), color=MODE_COLOR["kfac"], label="real kfac, mean")
        ax.scatter(sgd, [y[-1] - 0.25] * len(sgd), color=SGD_COLOR, marker="|", s=200,
                   label=f"cell D over {len(sgd)} seeds (5-seed job)")
    ax.scatter([d["real_kfac_seed0"]["test_acc"] * 100], [len(order) - 0.3], marker="x",
               color=MODE_COLOR["kfac"], label="real kfac, seed 0")
    ax.set_yticks(y, labels)
    ax.set_xlabel("test accuracy (%)")
    ax.set_title("cnn_gn_cifar: which change to the SGD stand-in moves it away from kfac?")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save(fig, "fig5_cnn_gn_ablation")


# --------------------------------------------------------------------------------------------
# Figure 6: lambda sweep on mlp_ln_mnist, Fisher mode vs SGD stand-in
# --------------------------------------------------------------------------------------------
def fig6() -> None:
    out = ROOT / "fisher_ref/outputs"
    sweep = sorted(out.glob("warmup_sgd_lambda_sweep_lam*.json"))
    if not sweep:
        print("skip fig6: no lambda sweep yet")
        return
    model = "mlp_ln_mnist"
    per_lam = {}
    for f in sweep:
        d = json.load(open(f))
        per_lam[float(d["lambda"])] = _load_runs([f]).get(model, {})
    default = _load_runs(sorted(out.glob("warmup_sgd_v3_*.json"))
                         + sorted(out.glob("warmup_sgd_baseline_small.json"))).get(model, {})
    per_lam.setdefault(1e-3, {}).update(default)  # mlp_ln_mnist's own lambda
    lams = sorted(per_lam)
    fig, axes = plt.subplots(1, len(MODES), figsize=(3.3 * len(MODES), 3.6), sharey=True)
    for ax, mode in zip(axes, MODES):
        for lam in lams:
            by = per_lam[lam]
            fis = [r["test_acc"] * 100 for r in _arm(by, mode)]
            sgd = [r["test_acc"] * 100 for r in _arm(by, f"sgd_{mode}")]
            ax.scatter([lam * 0.85] * len(fis), fis, color=MODE_COLOR[mode], s=18)
            ax.scatter([lam * 1.15] * len(sgd), sgd, color=SGD_COLOR, s=18, marker="s")
            if not fis:
                ax.scatter([lam], [2], color="gray", marker="x", s=30,
                           label="no run (crashed / not reached)" if lam == lams[0] else None)
        mean = lambda arm: [np.mean([r["test_acc"] * 100 for r in _arm(per_lam[l], arm)])  # noqa: E731
                            if _arm(per_lam[l], arm) else np.nan for l in lams]
        ax.plot(lams, mean(mode), color=MODE_COLOR[mode], label=mode)
        ax.plot(lams, mean(f"sgd_{mode}"), color=SGD_COLOR, ls="--", label="SGD, same warm-up")
        ax.set_xscale("log")
        ax.set_xlim(min(lams) / 3, max(lams) * 3)
        ax.set_title(mode)
        ax.set_xlabel("λ")
        ax.legend(fontsize=7.5, loc="center left")
    axes[0].set_ylabel("test accuracy (%)")
    fig.suptitle(f"{model}: λ sweep, test accuracy of each run (partial: jobs still running)")
    fig.tight_layout()
    save(fig, "fig6_lambda_sweep_mlp_ln_mnist")


# --------------------------------------------------------------------------------------------
# Figure 7: simplified SGD-equivalence summary (dot + error-bar, replaces fig4's busy grid/table)
# --------------------------------------------------------------------------------------------
def fig7() -> None:
    files = sorted(Path(ROOT / "fisher_ref/outputs").glob("warmup_sgd_v3_*.json"))
    if not files:
        print("skip fig7: no data")
        return
    res = _load_runs(files)
    models = [m for m in ("mlp_ln_mnist", "cnn_gn_cifar", "resnet20_cifar") if m in res]
    modes = [m for m in MODES if any(_arm(res[x], m) and _arm(res[x], f"sgd_{m}") for x in models)]
    rows = [(model, mode) for model in models for mode in modes
            if _arm(res[model], mode) and _arm(res[model], f"sgd_{mode}")]
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    for yi, (model, mode) in zip(y, rows):
        by = res[model]
        fa = np.array([r["test_acc"] for r in _arm(by, mode)]) * 100
        sa = np.array([r["test_acc"] for r in _arm(by, f"sgd_{mode}")]) * 100
        ax.plot([fa.mean(), sa.mean()], [yi, yi], color="lightgray", zorder=1, lw=1)
        ax.errorbar(fa.mean(), yi + 0.16, xerr=_sd(fa), color=MODE_COLOR[mode], fmt="o", ms=7,
                    capsize=3, zorder=3)
        ax.errorbar(sa.mean(), yi - 0.16, xerr=_sd(sa), color=SGD_COLOR, fmt="s", ms=7,
                    capsize=3, zorder=3)
    ax.set_yticks(y, [f"{m}, {mo}" for m, mo in rows])
    ax.set_xlabel("test accuracy (%), mean ± 1 std over 5 seeds")
    ax.set_title("Same warm-up schedule, with vs. without the Fisher preconditioner\n"
                 "circle = real Fisher mode, square = momentum-SGD given that mode's own learning-rate "
                 "schedule\n(same seed ⇒ same init and data order; overlapping bars = not "
                 "distinguishable)", fontsize=9.5)
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="black", ms=8,
                          label="Fisher mode (color = mode, see row label)"),
               plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=SGD_COLOR, ms=8,
                          label="SGD, same schedule")]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig7_sgd_equivalence_summary")


# --------------------------------------------------------------------------------------------
# Figure 8: simplified lambda sweep on mnist_autoencoder (final loss vs lambda, one panel)
# --------------------------------------------------------------------------------------------
def fig8() -> None:
    base = ROOT / "benchmarks/outputs/sweeps/mnist_autoencoder_lam"
    csv_lams = ["1e-5", "1e-3", "1e-1"]
    if not all((base / l / "records.csv").exists() for l in csv_lams):
        print("skip fig8: sweep data missing")
        return
    final = {arm: [] for arm in MODES + ["adam"]}
    for lam_dir in csv_lams:
        per = defaultdict(list)
        with open(base / lam_dir / "records.csv") as f:
            for row in csv.DictReader(f):
                per[row["arm"]].append(float(row["loss"]))
        for arm in MODES + ["adam"]:
            tail = per[arm][-50:]
            final[arm].append(sum(tail) / len(tail))
    lams = [float(l) for l in csv_lams]
    lam_labels = list(csv_lams)

    # Repatriated cluster job: mnist_autoencoder at lambda=1e-4, all five Kronecker modes,
    # 2 seeds each (warmup_sgd_lambda_sweep_lam1e-04.json). Slots in between 1e-5 and 1e-3.
    extra = ROOT / "fisher_ref/outputs/warmup_sgd_lambda_sweep_lam1e-04.json"
    if extra.exists():
        by_label = {r["label"]: r for r in json.load(open(extra))["results"]["mnist_autoencoder"]}
        insert_at = 1
        lams.insert(insert_at, 1e-4)
        lam_labels.insert(insert_at, "1e-4")
        for mode in MODES:
            seed_means = [sum(by_label[f"{mode}_seed{s}"]["step_loss"][-50:]) / 50
                          for s in (0, 1) if f"{mode}_seed{s}" in by_label]
            final[mode].insert(insert_at, sum(seed_means) / len(seed_means))

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for mode in MODES:
        ax.plot(lams, final[mode], color=MODE_COLOR[mode], marker="o", ms=6, label=mode)
    adam_mean = sum(final["adam"]) / len(final["adam"])
    ax.axhline(adam_mean, color=SGD_COLOR, ls="--", lw=1.5,
               label="adam (reference; λ does not apply to it)")
    ax.set_xscale("log")
    ax.set_xticks(lams)
    ax.set_xticklabels(lam_labels)
    ax.set_xlabel("damping λ")
    ax.set_ylabel("final training loss (MSE, mean of last 50 steps)")
    ax.set_title("mnist_autoencoder: only diag reacts to λ, and stays far above adam\n"
                 "(4 of 5 modes are flat across 4 orders of magnitude of λ)", fontsize=10.5)
    ax.legend(loc="center right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig8_lambda_sweep_final_loss")


# --------------------------------------------------------------------------------------------
# Figure 9: mnist_autoencoder training loss vs. epoch, all modes overlaid on one panel
# --------------------------------------------------------------------------------------------
def fig9() -> None:
    path = ROOT / "benchmarks/outputs/sweeps/mnist_autoencoder_lam/1e-3/records.csv"
    if not path.exists():
        print("skip fig9: no data")
        return
    per = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            per[row["arm"]].append((float(row["epoch"]), float(row["loss"])))
    win = 25
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for arm, color in list(MODE_COLOR.items()) + [("adam", SGD_COLOR)]:
        rows = per.get(arm)
        if not rows:
            continue
        epochs, losses = zip(*rows)
        sm = np.convolve(losses, np.ones(win) / win, mode="valid")
        ax.plot(epochs[win // 2: win // 2 + len(sm)], sm, color=color, label=arm)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (MSE, 25-step running mean)")
    ax.set_title("mnist_autoencoder (λ=1e-3): all five Kronecker modes plateau,\n"
                 "adam keeps descending for the whole budget", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig9_mnist_autoencoder_loss_vs_epoch")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
