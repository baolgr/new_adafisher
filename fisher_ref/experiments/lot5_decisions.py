"""Apply the pre-registered decision rules to the operational protocol's outputs.

Every constant below was fixed before the runs that produced the data, exactly as its structural
counterpart does. Run it over the CSVs the operational jobs write::

    PYTHONPATH=src:. python -m fisher_ref.experiments.lot5_decisions

**The hypothesis as stated**: the ranking of the approximations under real operating conditions is
*not* the ranking under idealised ones. So it predicts **disagreement**, and "refuted" means the
rankings agree.

**The trap that shaped this file, and it is the whole design.** Both the rescaling-invariant
Frobenius gap and the step ratio are invariant under multiplying the structure by a constant (the
step ratio because it is invariant under scaling the direction, and the operational operator is
applied with no extra damping). So if the operational preconditioner is proportional to the
identity -- which was measured, its condition number lying between 1.00 and 1.11 -- then all five
operational values collapse onto one and their order is noise. A rank correlation against noise is
about zero, and a rule that reads "correlation near zero" as "the rankings differ" would announce
the hypothesis **confirmed with no evidence at all**: simulated over 20 000 draws of a random
ranking of five, the median correlation over fifteen cells is at or below 0.2 in 98.7 % of them.

Two gates therefore stand in front of the verdict, and both use data the runner already produces --
a second re-warm on an independent batch order:

* **degeneracy** -- a cell whose operational values span less than the operator's own draw-to-draw
  spread has no ranking to speak of. It is counted and reported as "the operational preconditioner
  has no measurable preferred direction here", which is a finding in its own right, and it never
  votes;
* **reproducibility** -- the correlation between the two draws is the **ceiling** on what the
  structural-versus-operational correlation can mean. A ranking that does not reproduce across two
  draws of the same estimator cannot be said to differ from anything.

With those, "confirmed" needs the operational ranking to reproduce **and** to differ; "refuted"
needs it to agree. Neither outcome is free.

Three readings are emitted: the primary one over the four Kronecker modes whose structural
counterpart is the same object minus the operational compromises; a five-mode variant that adds
``diag`` against the reading of its own formula; and, for normalisation layers, a sign-agreement
rate with a binomial interval instead of a rank correlation, because over two items a rank
correlation is always plus or minus one and its median partitions at 50 %.

A fourth output is the **ladder**: the three gaps along
``structure -> the optimizer's own formulas -> after averaging -> after damping``, reported as
shares of the sum of their absolute values. The dominant rung is named only when every gap is
non-negative, because a later rung can fit better after optimal rescaling and an argmax over mixed
signs would name a winner among quantities that do not add up to anything.

Kendall's tau-b is implemented here rather than imported: ``scipy`` is in neither the project's
dependencies nor the cluster's offline wheelhouse, and for five items the definition is ten
comparisons.

Environment variables::

    P2D_OUTPUTS  root of the operational metrics trees
                 (default: fisher_ref/outputs/p2)
    P2D_OUT      output path (default: fisher_ref/outputs/lot5_decisions.json)

Output: a verdict table on standard output and the JSON at ``P2D_OUT``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

OUTPUTS = Path(os.environ.get("P2D_OUTPUTS", str(ROOT / "fisher_ref" / "outputs" / "p2")))
OUT = Path(os.environ.get("P2D_OUT", str(ROOT / "fisher_ref" / "outputs" / "lot5_decisions.json")))

#: The **primary** verdict ranks the four modes whose P1 counterpart is the same object minus the
#: operational compromises: fp64, per-example, no averaging. ``diag`` is deliberately not here —
#: its P1-rung structure ``af_raw`` is a *different estimator* (the diagonal of the campaign's own
#: type-2 factors), which is the very confusion ``plan_exp_lot5.md`` §0.14 item 1 records. It is
#: ranked in the five-mode variant instead, against the P1-py reading of its own formula.
PRIMARY_PAIRS = (("kfac", "p2_kfac"), ("ekfac", "p2_ekfac"),
                 ("tkfac", "p2_tkfac"), ("tekfac", "p2_tekfac"))
#: The five-mode reading, reported alongside. One of its five (``diag``) sits at the P1-py rung, so
#: it is not the like-for-like comparison the primary is — which is exactly why both are reported.
FIVE_PAIRS = PRIMARY_PAIRS + (("diag_py_factors", "p2_diag"),)
#: Normalisation layers: only two pairs exist there (§0.4), so a rank correlation over two items is
#: always +/-1 and a median of it partitions at 50 %. Reported as a sign-agreement rate with a
#: binomial interval instead (§0.9).
NORM_PAIRS = (("diag_py_factors", "p2_diag"), ("kron_py", "p2_kfac"))

#: §0.9's thresholds, all of them.
TAU_DIFFER = 0.2
TAU_AGREE = 0.6
#: A model on which more than this fraction of cells are degenerate is reported as "not decidable",
#: not as confirmed.
DEGENERATE_SHARE = 0.5
LADDER_MAJORITY = 2.0 / 3.0
#: The rungs, in order, for the ladder attribution (§0.4).
LADDER = {"diag": ("af_raw", "diag_py_factors", "p2_raw_diag", "p2_diag"),
          "kfac": ("kfac", "kron_py", "p2_raw_kfac", "p2_kfac")}
LADDER_STEPS = ("formula", "estimator", "damping")
#: The **source** the primary verdict is read on — the ``source`` column's own vocabulary
#: (``type2`` / ``empirical``), whose ``empirical`` rows are the ones compared against the reference
#: ``E_hat``. The operational state, ``kron_py`` and ``diag_py_*`` are all built from the
#: **empirical, true-label, batch-mean** gradient, so against ``F`` the ``P1 -> P1-py`` step would
#: carry the whole type-2 -> empirical source change on top of the formula change it is meant to
#: isolate — and lot 1 measured that source gap at 1.42 relative on A1, which is not a correction
#: term (§0.9).
PRIMARY_SOURCE = "empirical"
#: The reference that source is compared against, for the report's metadata.
PRIMARY_REFERENCE = "E_hat"

#: The layer kinds the **primary** verdict is read on: every kind where P1 builds all four Kronecker
#: structures. ``registry.LAYER_TYPES`` distinguishes an unshared ``Linear`` (``linear``), one
#: applied at several positions (``linear_shared``) and the output module (``head``) — all three are
#: ``Linear``, and leaving any of them out would silently drop cells. ``embed`` is excluded because
#: the optimizer hooks no such parameter, and ``norm`` has the secondary verdict of its own.
PRIMARY_KINDS = ("linear", "linear_shared", "conv", "head")


def kendall_tau_b(first: Sequence[float], second: Sequence[float]) -> float:
    """Kendall's ``tau_b`` between two small vectors, ties included; ``nan`` if either is all-tied.

    Written here rather than imported: ``scipy`` is in neither the project's dependencies nor the
    cluster's ``--no-index`` wheelhouse, and for ``n <= 5`` the definition is ten comparisons.
    """
    n = len(first)
    concordant = discordant = tied_first = tied_second = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = first[i] - first[j]
            b = second[i] - second[j]
            if a == 0 and b == 0:
                tied_first += 1
                tied_second += 1
            elif a == 0:
                tied_first += 1
            elif b == 0:
                tied_second += 1
            elif a * b > 0:
                concordant += 1
            else:
                discordant += 1
    pairs = n * (n - 1) / 2
    denominator = ((pairs - tied_first) * (pairs - tied_second)) ** 0.5
    return (concordant - discordant) / denominator if denominator else float("nan")


def _finite(values: Sequence[float]) -> List[float]:
    return [v for v in values if v == v and abs(v) != float("inf")]


def median(values: Sequence[float]) -> float:
    """``statistics.median`` with ``nan`` dropped first.

    Not a nicety: ``statistics.median`` sorts, and a ``nan`` makes the sort order meaningless —
    ``median([0.9, 0.8, nan, 0.7, 0.95])`` returns ``0.9`` instead of ``0.85``, silently.
    """
    finite = _finite(values)
    return statistics.median(finite) if finite else float("nan")


def load(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/*/*/*/metrics.csv")):
        model, arm, seed, fraction = path.parts[-5:-1]
        for row in csv.DictReader(path.open()):
            row.update(model=model, arm=arm, seed=seed, fraction=fraction)
            rows.append(row)
    return rows


def _value(row: Dict[str, Any]) -> Optional[float]:
    try:
        value = float(row["value"])
    except (TypeError, ValueError):
        return None
    return value if value == value else None


#: The cell key. ``arm`` and ``seed`` are part of it: without them a second seed's or a second arm's
#: CSV silently overwrites the first, and §10.3 requires every aggregate to carry the count it was
#: built from.
CellKey = Tuple[str, str, str, str, str, str, str, str]


def index(rows: Sequence[Dict[str, Any]]) -> Dict[CellKey, Dict[str, float]]:
    out: Dict[CellKey, Dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = _value(row)
        if value is None or not row.get("layer"):
            continue
        key = (row["model"], row["arm"], row["seed"], row["layer"], row["fraction"],
               row["source"], row["metric"], row["lambda_alpha"] or "")
        out[key][row["structure"]] = value
    return out


def layer_kinds(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
    return {(row["model"], row["layer"]): row["layer_type"]
            for row in rows if row.get("layer")}


def noise_floors(by_key: Dict[CellKey, Dict[str, float]]) -> Dict[Tuple[str, ...], float]:
    """``layer_noise_floor``, the reference's own split-half distance, per (model, layer, ...)."""
    out: Dict[Tuple[str, ...], float] = {}
    for key, values in by_key.items():
        if key[6] == "layer_noise_floor":
            out[key[:6]] = values.get("reference", float("nan"))
    return out


# ------------------------------------------------------------------------------------------------
# The two gates
# ------------------------------------------------------------------------------------------------


def operator_noise(values: Dict[str, float], pairs: Sequence[Tuple[str, str]]) -> float:
    """The operational operator's own draw-to-draw spread in this cell.

    ``median_m |e_F*(p2rep_m) - e_F*(p2_m)|`` over the modes for which both exist. ``nan`` when the
    replica was not run at this fraction — the caller then transfers the value measured at the
    fraction where it was (``plan_exp_lot5.md`` §0.9), which is declared, not silent.
    """
    gaps = []
    for _, p2 in pairs:
        replica = p2.replace("p2_", "p2rep_", 1)
        if p2 in values and replica in values:
            gaps.append(abs(values[replica] - values[p2]))
    return median(gaps) if gaps else float("nan")


def degenerate(values: Dict[str, float], pairs: Sequence[Tuple[str, str]], noise: float) -> bool:
    """Is there a P2 ranking to speak of at all?

    ``True`` when the modes' ``e_F_star`` span **less** than the operator's own noise — i.e. the
    preconditioner has no measurable preferred direction and their order is a batch draw. Such a
    cell never votes.
    """
    p2 = _finite([values[b] for _, b in pairs if b in values])
    if len(p2) < len(pairs) or noise != noise:
        return False
    return (max(p2) - min(p2)) <= noise


# ------------------------------------------------------------------------------------------------
# The verdict
# ------------------------------------------------------------------------------------------------


def collect(by_key: Dict[CellKey, Dict[str, float]], pairs: Sequence[Tuple[str, str]],
            kinds: Dict[Tuple[str, str], str], allowed_kinds: Optional[Sequence[str]],
            reference_source: str) -> Dict[str, Dict[str, Any]]:
    """Per model: the cell-level statistics the verdict is read from."""
    noise_by_layer: Dict[Tuple[str, str, str, str], float] = {}
    for key, values in by_key.items():
        model, arm, seed, layer, _fraction, source, metric, alpha = key
        if metric != "e_F_star" or source != reference_source or alpha:
            continue
        measured = operator_noise(values, pairs)
        if measured == measured:
            noise_by_layer[(model, arm, seed, layer)] = measured

    out: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"tau": [], "tau_reproducible": [], "degenerate": 0, "dropped": 0, "cells": 0,
                 "delta": [], "spread": []})
    for key, values in by_key.items():
        model, arm, seed, layer, _fraction, source, metric, alpha = key
        if metric != "e_F_star" or source != reference_source or alpha:
            continue
        if allowed_kinds is not None and kinds.get((model, layer)) not in allowed_kinds:
            continue
        raw_p1 = [values.get(a) for a, _ in pairs]
        raw_p2 = [values.get(b) for _, b in pairs]
        record = out[model]
        record["cells"] += 1
        if any(v is None or v != v for v in raw_p1 + raw_p2):
            record["dropped"] += 1
            continue
        p1 = [float(v) for v in raw_p1 if v is not None]
        p2 = [float(v) for v in raw_p2 if v is not None]
        noise = noise_by_layer.get((model, arm, seed, layer), float("nan"))
        if degenerate(values, pairs, noise):
            record["degenerate"] += 1
            continue
        tau = kendall_tau_b(p1, p2)
        if tau != tau:
            record["dropped"] += 1
            continue
        record["tau"].append(tau)
        raw_replicas = [values.get(b.replace("p2_", "p2rep_", 1)) for _, b in pairs]
        if all(v is not None and v == v for v in raw_replicas):
            ceiling = kendall_tau_b(p2, [float(v) for v in raw_replicas if v is not None])
            if ceiling == ceiling:
                record["tau_reproducible"].append(ceiling)
        record["delta"].append(median([values[b] - values[a] for a, b in pairs]))
        record["spread"].append(max(p1) - min(p1))
    return dict(out)


def verdict(record: Dict[str, Any]) -> str:
    """§0.9's verdict for one model, on ``e_F_star`` alone.

    ``rho`` is reported across the damping sweep and does **not** vote: the P2 operator has exactly
    one damping by construction (its own), so a `rho` verdict would be a conclusion about an inverse
    at a single lambda, which `plan_exp_draft.md` §10.3 forbids.
    """
    cells = record["cells"]
    if not cells:
        return "no cells"
    share = record["degenerate"] / cells
    if share > DEGENERATE_SHARE:
        return (f"NOT DECIDABLE -- {record['degenerate']}/{cells} cells degenerate: the "
                f"operational preconditioner's five modes span less than the operator's own "
                f"batch-draw spread, so there is no ranking to compare")
    votes = record["tau"]
    if not votes:
        return f"no admissible cells ({record['degenerate']} degenerate, {record['dropped']} dropped)"
    tau = median(votes)
    ceiling = median(record["tau_reproducible"])
    detail = (f"median tau(P1,P2) = {tau:+.2f} over {len(votes)} cells; "
              f"reproducibility ceiling tau(P2,P2rep) = {ceiling:+.2f}; "
              f"{record['degenerate']} degenerate, {record['dropped']} dropped")
    if tau <= TAU_DIFFER:
        if ceiling == ceiling and ceiling < TAU_AGREE:
            return (f"NOT DECIDABLE -- the P2 ranking does not reproduce across two draws "
                    f"({ceiling:+.2f} < {TAU_AGREE}), so its disagreement with P1 is not "
                    f"interpretable. {detail}")
        return f"CONFIRMED (rankings differ) -- {detail}"
    if tau >= TAU_AGREE:
        return f"REFUTED (rankings agree) -- {detail}"
    return f"mixed -- {detail}"


def magnitude(record: Dict[str, Any], floors: Dict[Tuple[str, ...], float],
              model: str) -> Dict[str, Any]:
    """§0.9's magnitude half, with **both** clauses: the operational cost has to exceed the layer's
    own noise floor *and* the spread between the modes it is supposed to dwarf."""
    if not record["delta"]:
        return {"cells": 0}
    model_floors = _finite([v for k, v in floors.items() if k[0] == model])
    floor = median(model_floors) if model_floors else float("nan")
    delta, spread = median(record["delta"]), median(record["spread"])
    return {
        "cells": len(record["delta"]),
        "median_operational_cost": delta,
        "median_spread_between_modes": spread,
        "median_layer_noise_floor": floor,
        "above_noise_floor": bool(floor == floor and delta > floor),
        "compromises_dominate": bool(delta > spread and (floor != floor or delta > floor)),
    }


def sign_agreement(by_key: Dict[CellKey, Dict[str, float]], pairs: Sequence[Tuple[str, str]],
                   kinds: Dict[Tuple[str, str], str], allowed_kinds: Sequence[str],
                   reference_source: str) -> Dict[str, Dict[str, Any]]:
    """The normalisation-layer reading: how often the two pairs order the same way, with a binomial
    interval. Not a tau — over two items ``tau_b`` is exactly +/-1 and its median partitions at
    50 %, which would be a coin flip dressed as a rank statistic (§0.9)."""
    counts: Dict[str, List[int]] = defaultdict(list)
    for key, values in by_key.items():
        model, _arm, _seed, layer, _fraction, source, metric, alpha = key
        if metric != "e_F_star" or source != reference_source or alpha:
            continue
        if kinds.get((model, layer)) not in allowed_kinds:
            continue
        raw = [values.get(a) for a, _ in pairs] + [values.get(b) for _, b in pairs]
        if any(v is None or v != v for v in raw):
            continue
        p1_a, p1_b, p2_a, p2_b = (float(v) for v in raw)  # type: ignore[arg-type]
        counts[model].append(1 if (p1_a - p1_b) * (p2_a - p2_b) > 0 else 0)
    out: Dict[str, Dict[str, Any]] = {}
    for model, hits in counts.items():
        n, k = len(hits), sum(hits)
        rate = k / n if n else float("nan")
        half = 1.96 * math.sqrt(rate * (1 - rate) / n) if n else float("nan")
        out[model] = {"cells": n, "agreement_rate": rate,
                      "ci": [max(0.0, rate - half), min(1.0, rate + half)],
                      "distinguishable_from_chance": bool(n and (rate - half > 0.5
                                                                 or rate + half < 0.5))}
    return out


def ladder(by_key: Dict[CellKey, Dict[str, float]], reference_source: str
           ) -> Dict[str, Dict[str, Any]]:
    """§0.9's ladder: the three gaps as shares of the sum of their absolute values.

    Reported as shares, and the dominant rung is named only when **every** gap is non-negative — a
    later rung can fit better after optimal rescaling, and an argmax over mixed signs would name a
    winner among quantities that do not add up to anything.
    """
    shares: Dict[str, List[List[float]]] = defaultdict(list)
    mixed: Dict[str, int] = defaultdict(int)
    for key, values in by_key.items():
        model, _arm, _seed, _layer, _fraction, source, metric, alpha = key
        if metric != "e_F_star" or source != reference_source or alpha:
            continue
        for mode, rungs in LADDER.items():
            if not all(name in values for name in rungs):
                continue
            gaps = [values[rungs[i + 1]] - values[rungs[i]] for i in range(3)]
            total = sum(abs(g) for g in gaps)
            if total <= 0:
                continue
            label = f"{model}/{mode}"
            if any(g < 0 for g in gaps):
                mixed[label] += 1
            shares[label].append([abs(g) / total for g in gaps])
    out: Dict[str, Dict[str, Any]] = {}
    for label, rows in shares.items():
        per_step = {LADDER_STEPS[i]: median([row[i] for row in rows]) for i in range(3)}
        clean = len(rows) - mixed[label]
        winner = max(per_step, key=lambda name: per_step[name])
        dominant = (winner if per_step[winner] >= LADDER_MAJORITY and clean >= LADDER_MAJORITY *
                    len(rows) else "mixed")
        out[label] = {"cells": len(rows), "cells_with_a_negative_gap": mixed[label],
                      "median_shares": per_step, "dominant": dominant}
    return out


def main() -> None:
    rows = load(OUTPUTS)
    if not rows:
        raise SystemExit(f"no metrics.csv under {OUTPUTS}")
    by_key = index(rows)
    kinds = layer_kinds(rows)
    floors = noise_floors(by_key)
    report: Dict[str, Any] = {
        "source": str(OUTPUTS), "rows": len(rows),
        "seeds": sorted({row["seed"] for row in rows}),
        "arms": sorted({row["arm"] for row in rows}),
        "models": sorted({row["model"] for row in rows}),
        "rule": {"tau_differ": TAU_DIFFER, "tau_agree": TAU_AGREE,
                 "degenerate_share": DEGENERATE_SHARE, "ladder_majority": LADDER_MAJORITY,
                 "primary_source": PRIMARY_SOURCE, "primary_reference": PRIMARY_REFERENCE,
                 "primary_kinds": list(PRIMARY_KINDS),
                 "primary_pairs": [list(p) for p in PRIMARY_PAIRS]},
    }
    print(f"models {report['models']} | arms {report['arms']} | seeds {report['seeds']} "
          f"| reference {PRIMARY_REFERENCE}", flush=True)

    for label, pairs, allowed in (
            ("primary (linear/conv/head, P1 rung, 4 Kronecker modes)", PRIMARY_PAIRS,
             PRIMARY_KINDS),
            ("five-mode variant (diag at the P1-py rung)", FIVE_PAIRS, PRIMARY_KINDS)):
        per_model = collect(by_key, pairs, kinds, allowed, PRIMARY_SOURCE)
        report[label] = {
            "verdicts": {model: verdict(record) for model, record in per_model.items()},
            "magnitude": {model: magnitude(record, floors, model)
                          for model, record in per_model.items()},
        }
        print(f"\n== HF7, {label} ==", flush=True)
        for model, text in sorted(report[label]["verdicts"].items()):
            print(f"  {model:<20} {text}", flush=True)
        for model, stats in sorted(report[label]["magnitude"].items()):
            if not stats.get("cells"):
                continue
            print(f"  {model:<20} operational cost {stats['median_operational_cost']:+.3f} vs "
                  f"spread between modes {stats['median_spread_between_modes']:.3f} vs floor "
                  f"{stats['median_layer_noise_floor']:.3f} -> compromises dominate: "
                  f"{stats['compromises_dominate']} ({stats['cells']} cells)", flush=True)

    report["normalisations (sign agreement)"] = sign_agreement(
        by_key, NORM_PAIRS, kinds, ("norm",), PRIMARY_SOURCE)
    print("\n== normalisation layers: sign agreement, not a rank correlation ==", flush=True)
    for model, stats in sorted(report["normalisations (sign agreement)"].items()):
        print(f"  {model:<20} {stats['agreement_rate']:.0%} "
              f"[{stats['ci'][0]:.0%}, {stats['ci'][1]:.0%}] over {stats['cells']} cells -> "
              f"distinguishable from chance: {stats['distinguishable_from_chance']}", flush=True)

    report["ladder"] = ladder(by_key, PRIMARY_SOURCE)
    print("\n== the ladder: P1 -> P1-py (formula) -> P2-raw (estimator) -> P2 (damping) ==",
          flush=True)
    for key, stats in sorted(report["ladder"].items()):
        shares = ", ".join(f"{k} {v:.0%}" for k, v in stats["median_shares"].items())
        print(f"  {key:<28} dominant: {stats['dominant']:<10} ({shares}; {stats['cells']} cells, "
              f"{stats['cells_with_a_negative_gap']} with a negative gap)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwritten to {OUT}", flush=True)


if __name__ == "__main__":
    main()
