"""P1 — the structural protocol: every structure against the exact references, at fixed ``theta``
(``plan_exp_draft.md`` §3.2, §13; lot 2 of its §9).

"What is the best approximation *possible* in each family": same ``theta``, same probes, no EMA, no
min-max, damping swept. The operational counterpart (P2, reading ``AdaFisherMulti``'s live state) is
lot 5's.

Unlike ``experiments/``, this is a real CLI: it sweeps
``(model, arm, seed, fraction) x structure x source x lambda`` and writes ``plan_exp_draft.md``
§13's format — a long-format ``metrics.csv`` plus a ``meta.json`` sidecar — under
``fisher_ref/outputs/<model>/<arm>/seed<n>/<fraction>/``.

    PYTHONPATH=src:. python -m fisher_ref.runners.p1_structural --model mlp_ln_mnist \\
        --arm diag --fractions 0,0.01,0.1,0.5,1 --probes 4000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benchmarks.common.runner import discover_benchmarks  # noqa: E402
from fisher_ref import conventions, metrics, probes, registry  # noqa: E402
from fisher_ref.approx import (  # noqa: E402
    Dense,
    Diag,
    accumulate_ekfac,
    accumulate_factors,
    af_raw_from_factors,
    cross_term_share,
    exact_separate,
)
from fisher_ref.capture import capturable_modules, prepare_model  # noqa: E402
from fisher_ref.checkpoints import discover_runs, load_theta  # noqa: E402
from fisher_ref.metrics.coupling import as_rows as coupling_rows  # noqa: E402
from fisher_ref.reference import build_dense_reference, to_augmented  # noqa: E402

#: ``plan_exp_draft.md`` §13's columns, in order. Long format: one row per measured number, so a
#: new structure or metric never changes the schema.
COLUMNS = ("layer", "layer_type", "structure", "source", "reference", "lambda_alpha", "metric",
           "value", "ci_low", "ci_high", "N", "probe_split", "seed", "protocol", "metrics_version")


def _rows(records: Sequence[Dict[str, Any]], **defaults: Any) -> List[Dict[str, Any]]:
    out = []
    for record in records:
        row = {column: None for column in COLUMNS}
        row.update(defaults)
        row.update(record)
        out.append(row)
    return out


def _layer_types(model: torch.nn.Module, example: torch.Tensor) -> Dict[str, str]:
    return {info.name: info.layer_type for info in registry.classify(model, example)}


def analyse_layer(name: str, exact: torch.Tensor, entry, ekfac, kinds: Dict[str, str],
                  alphas: Sequence[float], source: str, defaults: Dict[str, Any],
                  ) -> List[Dict[str, Any]]:
    """Every structure available for one layer, at every damping — M1 always, M3/M5 per ``lambda``."""
    rows: List[Dict[str, Any]] = []
    common = {**defaults, "layer": name, "layer_type": kinds.get(name, "other"), "source": source}

    structures: Dict[str, Any] = {"exact_diag": Diag(exact.diagonal().clone())}
    if entry.kind != "norm":
        structures["kfac"] = entry.kfac()
        structures["tkfac"] = entry.tkfac()
        structures["af_raw"] = af_raw_from_factors(entry.A, entry.G)
        if ekfac is not None:
            structures["ekfac"] = ekfac
        A, G = metrics.best_kronecker_fit(exact, entry.d_out, entry.d_in)
        structures["best_kron"] = Dense(torch.kron(G, A))
        report = metrics.kronecker_structure(exact, entry.d_out, entry.d_in,
                                             af_raw=structures["af_raw"])
        rows += _rows(report.as_rows(structure="exact_block"), **common)
    else:
        stats = entry.norm_stats()
        channels = entry.d_out
        structures["hadamard"] = stats.hadamard()
        structures["exact_separate"] = exact_separate(exact, channels)
        structures["as_implemented"] = stats.as_implemented_diagonal()
        total_share, diagonal_share = cross_term_share(exact, channels)
        rows += _rows([{"metric": "cross_term_share_total", "value": total_share},
                       {"metric": "cross_term_share_diagonal", "value": diagonal_share}],
                      **common, structure="exact_block")

    for structure_name, block in structures.items():
        rows += _rows(metrics.frobenius(exact, block).as_rows(), **common,
                      structure=structure_name)
        for alpha, lam in metrics.lambda_grid(exact, alphas).items():
            rows += _rows(metrics.stein_kl(exact, block, lam).as_rows(), **common,
                          structure=structure_name, lambda_alpha=alpha)
    return rows


def run_fraction(bench, run, fraction: float, args, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    loaded = load_theta(run, fraction, device="cpu")
    # Prepared once, here: the traversal casts the batch, not the network, and the reference and
    # the accumulators must share one model object (capture.prepare_model's docstring).
    model = prepare_model(loaded.model, conventions.REFERENCE_DTYPE, args.device)
    train = probes.build_probe_set(bench, split="train", n=args.probes, seed=args.seed,
                                   data_root=args.data_root)
    inputs, targets = train.as_model_batch(bench)
    modules = capturable_modules(model)
    if args.modules:
        modules = {name: modules[name] for name in args.modules}
    kinds = _layer_types(model, inputs[:2].to(conventions.REFERENCE_DTYPE))

    defaults = {"N": len(train), "probe_split": "train", "seed": run.seed, "protocol": "P1",
                "metrics_version": conventions.METRICS_VERSION, "reference": "F"}
    rows: List[Dict[str, Any]] = []
    for source in args.sources:
        reference = build_dense_reference(model, inputs, targets, source=source,
                                          batch_size=args.batch, device=args.device,
                                          modules=list(modules), probe_digest=train.digest[:12])
        reference.matrix = reference.matrix.to("cpu")
        factors = accumulate_factors(model, inputs, targets, source=source, modules=modules,
                                     batch_size=args.batch, device=args.device)
        ekfacs = accumulate_ekfac(model, inputs, targets, source=source, modules=modules,
                                  factors=factors, batch_size=args.batch, device=args.device)
        reference_name = "E_hat" if source == "empirical" else "F"

        for name, entry in factors.items():
            block = reference.block(name)
            if entry.kind != "norm":
                block = to_augmented(block, reference.layout.augmented_permutation(name))
            rows += analyse_layer(name, block, entry, ekfacs.get(name), kinds, args.alphas,
                                  source, {**defaults, "reference": reference_name})

        ranges = [(name, reference.layout.block_slice(name)) for name in factors]
        names, coupling = metrics.coupling_matrix(reference.matrix, ranges)
        block_defaults = {**defaults, "reference": reference_name, "source": source}
        rows += _rows(coupling_rows(names, coupling), **block_defaults, structure="exact_block")
        rows += _rows([{"metric": key, "value": value} for key, value in
                       metrics.offdiagonal_mass(reference.matrix, ranges).items()],
                      **block_defaults, structure="block_diagonal")

        if source == "type2" and args.noise_partitions:
            def build(indices, _source=source):
                return build_dense_reference(model, inputs[indices], targets[indices],
                                             source=_source, batch_size=args.batch,
                                             device=args.device,
                                             modules=list(modules)).matrix.to("cpu")

            floor = metrics.noise_floor(build, len(train), partitions=args.noise_partitions,
                                        seed=args.seed)
            rows += _rows(floor.as_rows(), **block_defaults, structure="reference")
            meta.setdefault("noise_floor", {})[str(fraction)] = {
                "median": floor.median, "low": floor.low, "high": floor.high,
                "sigma_n": floor.sigma_n, "null_two_independent": floor.null_two_independent,
                "partitions": floor.n_partitions}
        meta.setdefault("references", {})[f"{fraction}/{source}"] = reference.metadata()
        del reference
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="mlp_ln_mnist")
    parser.add_argument("--arm", default="diag")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fractions", default="0.5",
                        help="comma-separated checkpoint fractions, e.g. 0,0.01,0.1,0.5,1")
    parser.add_argument("--sources", nargs="+", default=["type2", "empirical"])
    parser.add_argument("--probes", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--alphas", default="1e-4,1e-3,1e-2,1e-1,1",
                        help="lambda = alpha * tr(R)/P (plan_exp_draft.md §3.3); never a single one")
    parser.add_argument("--noise-partitions", type=int, default=20,
                        help="§3.4's random partitions for the noise-floor interval; 0 to skip")
    parser.add_argument("--modules", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", default=str(ROOT / "benchmarks" / "data"))
    parser.add_argument("--outputs-root", default=str(ROOT / "benchmarks" / "outputs"))
    parser.add_argument("--out-dir", default=str(ROOT / "fisher_ref" / "outputs"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.alphas = [float(value) for value in str(args.alphas).split(",") if value]
    fractions = [float(value) for value in str(args.fractions).split(",") if value]
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    state = conventions.configure()
    torch.manual_seed(args.seed)
    bench = discover_benchmarks()[args.model]
    runs = discover_runs(args.outputs_root, model=args.model, arm=args.arm, seed=args.seed)
    if len(runs) != 1:
        raise SystemExit(f"expected exactly one run for {args.model}/{args.arm} seed {args.seed}; "
                         f"found {[str(r.directory) for r in runs]}")
    run = runs[0]
    print(f"P1 {args.model}/{args.arm} seed {run.seed} | device={args.device} "
          f"TF32 off={state.tf32_is_off()}", flush=True)

    meta: Dict[str, Any] = conventions.run_metadata({"protocol": "P1", "regime": "A",
                                                     "args": vars(args)})
    for fraction in fractions:
        started = time.perf_counter()
        rows = run_fraction(bench, run, fraction, args, meta)
        out = Path(args.out_dir) / args.model / args.arm / f"seed{run.seed}" / str(fraction)
        out.mkdir(parents=True, exist_ok=True)
        _write_csv(out / "metrics.csv", rows)
        (out / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
        print(f"  fraction {fraction}: {len(rows)} rows in {time.perf_counter() - started:.1f}s "
              f"-> {out}", flush=True)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    import csv  # noqa: PLC0415
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in COLUMNS})


if __name__ == "__main__":
    main()
