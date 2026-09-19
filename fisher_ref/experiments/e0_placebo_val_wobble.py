"""Does the plain-momentum placebo reproduce the Fisher modes' held-out noise?

Costs nothing: it reads runs that already exist on disk and computes a metric they were never read
for.

Background. An earlier investigation established that the five Fisher modes' extra noise sits
entirely on the *held-out* side: their per-batch and per-epoch **training** losses are
indistinguishable from Adam's, while their validation loss and accuracy jump around 2.6 to 4.5
times more. A plain-momentum placebo was then built -- same momentum, same weight decay, same
start-up-then-constant step-size schedule, no curvature estimate at all -- and compared against the
real modes, but only on the *final* test loss and accuracy. The per-epoch validation curves were
recorded and never read.

This script reads them, and asks the original question of the placebo instead of of Adam: if the
placebo is as noisy on held-out data as the real mode, the noise is a property of the step size and
of not normalising the step, not of the curvature estimate, which the placebo does not have.

Metric. "Wobble" is the mean absolute change from one epoch to the next. It is reported over the
stable *second half* of training -- the first epochs are dominated by the descent itself, not by
noise -- and, for reference, over all epochs. Accuracy wobble is in percentage points; loss wobble
is divided by that curve's own mean level, so a mode sitting at a higher loss is not penalised for
it.

Floor. Each real mode was run at two seeds, so the absolute difference between the two seeds'
wobbles says how much the wobble itself moves from one run to another. A real-versus-placebo
difference smaller than that floor means nothing.

Environment variables::

    SOURCE  input JSON basename under fisher_ref/outputs
            (default: warmup_sgd_baseline_small.json)
    HALF    fraction of epochs treated as the stable tail (default: 0.5)

Reads: ``fisher_ref/outputs/<SOURCE>``, and, as an anchor against Adam, the ``epochs.csv`` of the
campaign's own training runs under ``benchmarks/outputs/``.
Writes: ``fisher_ref/outputs/e0_placebo_val_wobble.json`` plus a table on standard output.
"""
import json
import os
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fisher_ref/outputs"
SOURCE = os.environ.get("SOURCE", "warmup_sgd_baseline_small.json")
HALF = float(os.environ.get("HALF", "0.5"))
MODES = ["diag", "kfac", "tkfac", "ekfac", "tekfac"]


def wobble(series, tail_only=True):
    """Mean |x[t+1] - x[t]| over the series, restricted to its stable tail."""
    xs = list(series)
    if tail_only:
        xs = xs[int(len(xs) * HALF):]
    if len(xs) < 2:
        return float("nan")
    return mean(abs(b - a) for a, b in zip(xs, xs[1:]))


def rel_wobble(series, tail_only=True):
    """Same, divided by the curve's own level, so curves at different heights are comparable."""
    xs = list(series)
    if tail_only:
        xs = xs[int(len(xs) * HALF):]
    lvl = mean(xs)
    w = wobble(series, tail_only)
    return w / lvl if lvl > 0 else float("nan")



# ----------------------------------------------------------------------------------------------
# Anchor: is the wobble *excessive* in the first place? Step 4 measured that against Adam, on the
# campaign runs (benchmarks/outputs/), which the placebo runs do not contain. Recomputed here with
# the identical metric so the two halves of E0 can be read on one scale. Caveat: those runs follow
# the wall-clock-budget protocol, so different arms completed different numbers of epochs.
# ----------------------------------------------------------------------------------------------

CAMPAIGN = {
    "mlp_ln_mnist": "mnist/mlp_ln_mnist",
    "cnn_gn_cifar": "cifar10/cnn_gn_cifar",
    "resnet20_cifar": "cifar10/resnet20_cifar",
    "mnist_autoencoder": "mnist/mnist_autoencoder",
    "vit_micro_cifar": "cifar10/vit_micro_cifar",
    "cct_2_3x2_cifar": "cifar10/cct_2_3x2_cifar",
}


def read_epochs_csv(path):
    """{arm: {"val_acc": [...], "val_loss": [...]}} from a campaign run's epochs.csv.

    A row is taken whole or not at all. Parsing each field under its own guard let a row whose
    accuracy parsed but whose loss did not append to one list only, after which the two curves
    were one epoch out of step for every later row and every wobble read off them was wrong.
    """
    import csv
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                values = {k: float(row[k]) for k in ("val_acc", "val_loss")}
            except (ValueError, KeyError, TypeError):
                continue                       # skip the row, keeping the two curves aligned
            arm = out.setdefault(row["arm"], {"val_acc": [], "val_loss": []})
            for k, value in values.items():
                arm[k].append(value)
    return out


def campaign_anchor():
    print("===== ANCHOR: the same metric against Adam, on the campaign runs =====", flush=True)
    print("(benchmarks/outputs/, wall-clock-budget protocol -- arms differ in epoch count)\n", flush=True)
    print(f"  {'network':>20s} {'mode':7s} {'acc wobble':>11s} {'/adam':>7s} {'rel loss wobble':>16s} {'/adam':>7s}", flush=True)
    anchor = {}
    for net, rel in CAMPAIGN.items():
        csv_path = ROOT / "benchmarks/outputs" / rel / "epochs.csv"
        if not csv_path.exists():
            print(f"  {net:>20s} [no epochs.csv]", flush=True)
            continue
        arms = read_epochs_csv(csv_path)
        if "adam" not in arms:
            print(f"  {net:>20s} [no adam arm]", flush=True)
            continue
        a_adam = 100 * wobble(arms["adam"]["val_acc"]) if arms["adam"]["val_acc"] else float("nan")
        l_adam = rel_wobble(arms["adam"]["val_loss"])
        anchor[net] = {}
        for mode in MODES:
            if mode not in arms:
                continue
            a = 100 * wobble(arms[mode]["val_acc"]) if arms[mode]["val_acc"] else float("nan")
            l = rel_wobble(arms[mode]["val_loss"])
            anchor[net][mode] = {"acc_wobble": a, "acc_over_adam": a / a_adam if a_adam else float("nan"),
                                 "loss_relwobble": l, "loss_over_adam": l / l_adam if l_adam else float("nan")}
            print(f"  {net:>20s} {mode:7s} {a:11.3f} {a/a_adam if a_adam else float('nan'):7.2f} "
                  f"{l:16.4f} {l/l_adam if l_adam else float('nan'):7.2f}", flush=True)
    print(flush=True)
    return anchor


def main():
    src = OUT / SOURCE
    payload = json.load(open(src))
    print("E0: held-out wobble, real mode vs plain-momentum placebo", flush=True)
    print(f"source={src.name}  epochs={payload.get('epochs')}  seeds={payload.get('seeds')}  "
          f"tail=last {int((1-HALF)*100)}% of epochs\n", flush=True)

    results = {}
    for net, runs in payload["results"].items():
        by_label = {r["label"]: r for r in runs}
        results[net] = {}
        print(f"===== {net} =====", flush=True)
        print(f"  {'mode':7s} {'acc wobble (pts)':>26s}  {'rel. loss wobble':>26s}   verdict", flush=True)
        print(f"  {'':7s} {'real   placebo   ratio':>26s}  {'real   placebo   ratio':>26s}", flush=True)
        for mode in MODES:
            real = by_label.get(f"{mode}_seed0")
            real1 = by_label.get(f"{mode}_seed1")
            plac = by_label.get(f"sgd_{mode}_seed0")
            if real is None or plac is None:
                print(f"  {mode:7s} [missing]", flush=True)
                continue

            a_real = 100 * wobble(real["epoch_val_acc"])
            a_plac = 100 * wobble(plac["epoch_val_acc"])
            a_floor = abs(a_real - 100 * wobble(real1["epoch_val_acc"])) if real1 else float("nan")
            l_real = rel_wobble(real["epoch_val_loss"])
            l_plac = rel_wobble(plac["epoch_val_loss"])
            l_floor = abs(l_real - rel_wobble(real1["epoch_val_loss"])) if real1 else float("nan")

            # A gap counts as real only if it exceeds the seed-to-seed spread of the wobble itself.
            gap_a, gap_l = abs(a_real - a_plac), abs(l_real - l_plac)
            reproduced = (gap_a <= a_floor or a_floor != a_floor) and (gap_l <= l_floor or l_floor != l_floor)
            verdict = "placebo reproduces" if reproduced else "differs"

            results[net][mode] = {
                "acc_wobble_real": a_real, "acc_wobble_placebo": a_plac,
                "acc_wobble_seed_floor": a_floor,
                "loss_relwobble_real": l_real, "loss_relwobble_placebo": l_plac,
                "loss_relwobble_seed_floor": l_floor,
                "final_val_acc_real": real["epoch_val_acc"][-1],
                "final_val_acc_placebo": plac["epoch_val_acc"][-1],
                "reproduced": reproduced,
            }
            print(f"  {mode:7s} {a_real:8.3f} {a_plac:8.3f} {a_plac/a_real if a_real else float('nan'):7.2f}"
                  f"  {l_real:8.4f} {l_plac:8.4f} {l_plac/l_real if l_real else float('nan'):7.2f}"
                  f"   {verdict}  (seed floor {a_floor:.3f} pts / {l_floor:.4f})", flush=True)
        print(flush=True)

    import math
    print("===== VERDICT =====", flush=True)
    cells = [(n, m, d) for n, ms in results.items() for m, d in ms.items()]
    repro = sum(1 for _, _, d in cells if d["reproduced"])
    print(f"{repro} of {len(cells)} (network, mode) cells: the curvature-free placebo's held-out "
          f"wobble is within the real mode's own seed-to-seed spread.", flush=True)
    for key, label in [("acc_wobble", "accuracy"), ("loss_relwobble", "relative loss")]:
        rs = sorted(d[f"{key}_placebo"] / d[f"{key}_real"] for _, _, d in cells
                    if d[f"{key}_real"] > 0 and d[f"{key}_placebo"] == d[f"{key}_placebo"])
        if not rs:
            continue
        gmean = math.exp(sum(math.log(r) for r in rs) / len(rs))
        above = sum(1 for r in rs if r > 1)
        print(f"placebo/real {label}-wobble ratio over {len(rs)} cells: geometric mean {gmean:.2f}, "
              f"median {rs[len(rs)//2]:.2f}, quartiles {rs[len(rs)//4]:.2f}-{rs[3*len(rs)//4]:.2f}, "
              f"range {rs[0]:.2f}-{rs[-1]:.2f}; placebo noisier in {above}/{len(rs)}", flush=True)
    print(flush=True)

    anchor = campaign_anchor()

    with open(OUT / "e0_placebo_val_wobble.json", "w") as f:
        json.dump({"source": SOURCE, "half": HALF, "results": results,
                   "campaign_anchor": anchor}, f, indent=1)
    print(f"wrote {OUT/'e0_placebo_val_wobble.json'}", flush=True)


if __name__ == "__main__":
    main()
