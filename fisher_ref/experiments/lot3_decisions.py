"""Apply the pre-registered decision rules to the structural protocol's outputs.

A reader of ``metrics.csv`` files only: it computes nothing new and chooses no threshold. Every
number it compares against was fixed before the runs that produced the data. It prints one verdict
per hypothesis per model, together with the cells that produced it, so the verdict can be audited
rather than trusted.

The hypotheses it reads:

* **weight sharing, reduce against expand** -- per shared-layer cell, the interval on the
  *difference* of the two Frobenius gaps, so the vote is on a paired quantity rather than on two
  marginal intervals of strongly correlated numbers;
* **weight sharing, shared against unshared** -- the median Frobenius gap over shared layers
  against the head's, intervals included, plus the mechanism: every shared layer's sharing share
  above its own per-layer noise floor;
* **the diagonal bias** -- is the exact block's diagonal far from the product of the two factors'
  diagonals, on every layer but the first;
* **the cross terms of a normalisation block** -- is the coupling between the scale and the shift
  parameters large enough to matter;
* **the Hadamard reading against the exact diagonal** -- which of the two is closer;
* **does the step ratio fall along training** -- checked at both ends of the trajectory, for every
  layer and structure;
* **inter-layer coupling** -- is every off-diagonal coupling entry above 0.5. Note that the
  coupling matrix stores the **square root** of the uncentred kernel alignment, so a threshold of
  0.5 there is an alignment of 0.25.

A verdict is "confirmed" when at least two thirds of the admissible cells vote for it, "refuted"
when at least two thirds vote against, and "mixed" otherwise.

Environment variables::

    LOT3_OUTPUTS  root of the metrics trees (default: fisher_ref/outputs)
    LOT3_MODELS   comma-separated run directories with weight sharing
                  (default: cnn_gn_cifar,cnn_gn_cifar_bn,vit_micro_cifar)
    LOT3_A1       the unshared model, read for the sharing-free hypotheses
                  (default: mlp_ln_mnist)
    LOT3_JSON     optional path to dump the verdicts; unset prints only

Output: a verdict table on standard output, and the JSON at ``LOT3_JSON`` when set.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = Path(os.environ.get("LOT3_OUTPUTS", ROOT / "fisher_ref" / "outputs"))
MODELS = os.environ.get("LOT3_MODELS", "cnn_gn_cifar,cnn_gn_cifar_bn,vit_micro_cifar").split(",")
A1 = os.environ.get("LOT3_A1", "mlp_ln_mnist")
SHARED_TYPES = ("conv", "linear_shared")


def _float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _coalesce(*values: Optional[float], default: float) -> float:
    """The first value that is present, else ``default``.

    Spelled with explicit ``is None`` tests rather than ``a or b or default``, because a bound of
    exactly ``0.0`` is falsy: ``or`` would step over a real measured zero and use the next
    candidate, which is a different number. Everywhere below, "missing" must fall back and "zero"
    must not.
    """
    for value in values:
        if value is not None:
            return value
    return default


def load(model: str) -> Dict[float, List[Dict[str, str]]]:
    base = OUTPUTS / model / "diag" / "seed0"
    out: Dict[float, List[Dict[str, str]]] = {}
    if not base.is_dir():
        return out
    for directory in sorted(base.iterdir()):
        path = directory / "metrics.csv"
        if path.is_file():
            out[float(directory.name)] = list(csv.DictReader(path.open()))
    return out


def blocks(model: str, fraction: float) -> Dict[str, Dict[str, Any]]:
    path = OUTPUTS / model / "diag" / "seed0" / str(fraction) / "meta.json"
    if not path.is_file():
        return {}
    meta = json.loads(path.read_text())
    per = meta.get("blocks", {})
    return next(iter(per.values()), {}) if per else {}


def pick(rows: List[Dict[str, str]], **keys: Any) -> List[Dict[str, str]]:
    return [row for row in rows if all(row.get(k) == (v if v is None else str(v))
                                       for k, v in keys.items())]


def verdict(votes: List[str], positive: str, negative: Tuple[str, ...]) -> str:
    if not votes:
        return "no data"
    share = sum(v == positive for v in votes) / len(votes)
    against = sum(v in negative for v in votes) / len(votes)
    if share >= 2 / 3:
        return "confirmed"
    if against >= 2 / 3:
        return "refuted"
    return "mixed"


# ------------------------------------------------------------------------------------------------


def hf2_reduce_vs_expand(model: str, data: Dict[float, List[Dict[str, str]]]) -> Dict[str, Any]:
    """HF2 (ii): per shared-layer cell, CI95 of e_F(kfac_reduce) - e_F(kfac)."""
    cells: List[Tuple[str, float, str, float, Optional[float], Optional[float]]] = []
    for fraction, rows in sorted(data.items()):
        for row in pick(rows, source="type2", structure="kfac_reduce-kfac", metric="delta_e_F"):
            low, high, value = _float(row["ci_low"]), _float(row["ci_high"]), _float(row["value"])
            if value is None:
                continue
            if low is not None and high is not None and high < 0:
                vote = "reduce"
            elif low is not None and low > 0:
                vote = "expand"
            else:
                vote = "tie"
            cells.append((row["layer"], fraction, vote, value, low, high))
    by_type: Dict[str, List[str]] = defaultdict(list)
    for layer, _, vote, *_ in cells:
        by_type[layer].append(vote)
    return {"verdict": verdict([c[2] for c in cells], "reduce", ("expand", "tie")),
            "cells": cells, "by_layer": {k: verdict(v, "reduce", ("expand", "tie"))
                                         for k, v in by_type.items()}}


def hf2_shared_vs_unshared(model: str, data: Dict[float, List[Dict[str, str]]]) -> Dict[str, Any]:
    """HF2 (i): median e_F(kfac) over shared layers vs the head, CIs included, plus the mechanism:
    every shared layer's sharing_share above its own per-layer noise floor."""
    supports, against, details = 0, 0, []
    for fraction, rows in sorted(data.items()):
        types = {row["layer"]: row["layer_type"] for row in rows if row["layer_type"]}
        kfac = {row["layer"]: row for row in pick(rows, source="type2", structure="kfac",
                                                    metric="e_F")}
        shared = [row for layer, row in kfac.items() if types.get(layer) in SHARED_TYPES]
        heads = [row for layer, row in kfac.items() if types.get(layer) == "head"]
        if not shared or not heads:
            continue
        low = statistics.median([_coalesce(_float(r["ci_low"]), _float(r["value"]), default=0.0)
                                 for r in shared])
        high = statistics.median([_coalesce(_float(r["ci_high"]), _float(r["value"]), default=0.0)
                                  for r in shared])
        head = heads[0]
        head_low = _coalesce(_float(head["ci_low"]), _float(head["value"]), default=0.0)
        head_high = _coalesce(_float(head["ci_high"]), _float(head["value"]), default=0.0)
        floors = {row["layer"]: _float(row["value"]) for row in
                  pick(rows, source="type2", structure="reference", metric="layer_noise_floor")}
        shares = {row["layer"]: _float(row["value"]) for row in
                  pick(rows, source="type2", structure="b_exp", metric="sharing_share")}
        # A missing share must fail the clause (default 0) and a missing floor must too
        # (default 1). A *measured* zero on either side must be used as measured: with `or`, a
        # per-layer floor of exactly 0 became 1 and turned "any share at all clears the floor"
        # into "the share exceeds 1", which is the opposite reading.
        mechanism = all(_coalesce(shares.get(r["layer"]), default=0.0)
                        > _coalesce(floors.get(r["layer"]), default=1.0)
                        for r in shared)
        if low > head_high and mechanism:
            supports += 1
        elif high < head_low:
            against += 1
        details.append((fraction, round(statistics.median([_float(r["value"]) or 0 for r in
                                                           shared]), 4),
                        round(_float(head["value"]) or 0, 4), mechanism))
    n = len(details)
    if n == 0:
        return {"verdict": "no data", "details": details}
    if supports >= 4 and n >= 5 or (n < 5 and supports == n):
        state = "confirmed"
    elif against >= 4 and n >= 5 or (n < 5 and against == n):
        state = "refuted"
    else:
        state = "mixed"
    return {"verdict": state, "details": details, "supports": supports, "against": against}


def _q4_count(data: Dict[float, List[Dict[str, str]]], test) -> Tuple[int, int, List[Any]]:
    hits, total, details = 0, 0, []
    for fraction, rows in sorted(data.items()):
        outcome = test(rows)
        if outcome is None:
            continue
        total += 1
        hits += bool(outcome[0])
        details.append((fraction, outcome[1]))
    return hits, total, details


def q4_1_diagonal_bias(data, first_layer: str) -> Dict[str, Any]:
    def test(rows):
        values = [(_float(r["value"]) or 0.0, r["layer"]) for r in
                  pick(rows, source="type2", structure="exact_block", metric="diagonal_bias")
                  if r["layer"] != first_layer]
        if not values:
            return None
        return (min(v for v, _ in values) >= 0.10,
                {layer: round(v, 3) for v, layer in values})

    hits, total, details = _q4_count(data, test)

    def refuted(rows):
        values = [_float(r["value"]) or 0.0 for r in
                  pick(rows, source="type2", structure="exact_block", metric="diagonal_bias")
                  if r["layer"] != first_layer]
        return (bool(values) and max(values) <= 0.05, None) if values else None

    against, _, _ = _q4_count(data, refuted)
    return _threshold(hits, against, total, details)


def q4_2_cross_terms(data) -> Dict[str, Any]:
    def test(rows):
        values = {r["layer"]: _float(r["value"]) or 0.0 for r in
                  pick(rows, source="type2", structure="exact_block",
                       metric="cross_term_share_total")}
        if not values:
            return None
        return (min(values.values()) >= 0.10, {k: round(v, 3) for k, v in values.items()})

    hits, total, details = _q4_count(data, test)
    return _threshold(hits, total - hits, total, details)


def q4_3_hadamard_vs_diag(data) -> Dict[str, Any]:
    signs: List[str] = []
    details = []
    for fraction, rows in sorted(data.items()):
        e = {(r["layer"], r["structure"]): _float(r["value"]) for r in
             pick(rows, source="type2", metric="e_F")}
        layers = {layer for layer, s in e if s == "hadamard"}
        if not layers:
            continue
        per = {layer: (e[(layer, "hadamard")] or 0) - (e[(layer, "exact_diag")] or 0)
               for layer in layers if (layer, "exact_diag") in e}
        if all(v < 0 for v in per.values()):
            signs.append("hadamard_better")
        elif all(v > 0 for v in per.values()):
            signs.append("exact_diag_better")
        else:
            signs.append("mixed")
        details.append((fraction, {k: round(v, 4) for k, v in per.items()}))
    counts = {s: signs.count(s) for s in set(signs)}
    top = max(counts, key=lambda name: counts[name]) if counts else "no data"
    state = top if counts and counts[top] >= 4 and top != "mixed" else "mixed"
    return {"verdict": state, "counts": counts, "details": details}


def q4_4_rho_falls(data) -> Dict[str, Any]:
    fractions = sorted(data)
    if not fractions or fractions[0] != 0.0 or fractions[-1] != 1.0:
        return {"verdict": "no data"}
    first, last = data[0.0], data[1.0]
    outcomes = []
    for alpha in ("0.001", "0.1"):
        start = {(r["layer"], r["structure"]): _float(r["value"]) for r in
                 pick(first, source="type2", metric="rho", lambda_alpha=alpha)}
        end = {(r["layer"], r["structure"]): _float(r["value"]) for r in
               pick(last, source="type2", metric="rho", lambda_alpha=alpha)}
        for key in start.keys() & end.keys():
            before, after = start[key], end[key]
            if before is not None and after is not None:
                outcomes.append((alpha, key, after < before))
    holds = all(o[2] for o in outcomes) if outcomes else None
    return {"verdict": "no data" if holds is None else ("confirmed" if holds else "refuted"),
            "exceptions": [o for o in outcomes if not o[2]],
            "fraction_falling": (sum(o[2] for o in outcomes) / len(outcomes)) if outcomes else None}


def q4_5_coupling(data) -> Dict[str, Any]:
    def test(rows):
        values = [_float(r["value"]) or 0.0 for r in
                  pick(rows, source="type2", structure="exact_block")
                  if r["metric"].startswith("coupling::")]
        if not values:
            return None
        return (min(values) >= 0.5, round(min(values), 3))

    hits, total, details = _q4_count(data, test)
    return _threshold(hits, total - hits, total, details)


def _threshold(hits: int, against: int, total: int, details: List[Any]) -> Dict[str, Any]:
    if total == 0:
        state = "no data"
    elif hits >= min(4, total):
        state = "confirmed"
    elif against >= min(4, total):
        state = "refuted"
    else:
        state = "mixed"
    return {"verdict": state, "hits": hits, "total": total, "details": details}


FIRST_LAYER = {"mlp_ln_mnist": "features.0", "cnn_gn_cifar": "features.0",
               "cnn_gn_cifar_bn": "features.0", "vit_micro_cifar": "patch_embed.proj"}


def main() -> None:
    report: Dict[str, Any] = {}
    for model in [A1] + MODELS:
        data = load(model)
        if not data:
            print(f"{model}: no outputs under {OUTPUTS}")
            continue
        entry: Dict[str, Any] = {"fractions": sorted(data)}
        if model != A1:
            entry["HF2_ii_reduce_beats_expand"] = hf2_reduce_vs_expand(model, data)
            entry["HF2_i_shared_worse_than_unshared"] = hf2_shared_vs_unshared(model, data)
        entry["Q4.1_diagonal_bias"] = q4_1_diagonal_bias(data, FIRST_LAYER.get(model, ""))
        entry["Q4.2_cross_terms"] = q4_2_cross_terms(data)
        entry["Q4.3_hadamard_vs_exact_diag"] = q4_3_hadamard_vs_diag(data)
        entry["Q4.4_rho_falls"] = q4_4_rho_falls(data)
        entry["Q4.5_coupling"] = q4_5_coupling(data)
        report[model] = entry
        print(f"\n=== {model}  fractions {entry['fractions']}")
        for key, value in entry.items():
            if isinstance(value, dict):
                print(f"  {key:36s} {value.get('verdict')}")
                if key == "HF2_ii_reduce_beats_expand":
                    for layer, state in sorted(value["by_layer"].items()):
                        print(f"      {layer:28s} {state}")
    target = os.environ.get("LOT3_JSON")
    if target:
        Path(target).write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
