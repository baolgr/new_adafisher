"""Is AdaFisherMulti, at its default operating point, indistinguishable from plain momentum-SGD
with a matching warmup step size? (docs handoff, "2. Le test SGD avec momentum".)

``docs/reports/validation_noise_investigation.md`` and ``lambda_vs_curvature.py`` measured that the
*applied* divisor (curvature + damping) sits within a few percent of ``Lambda`` almost everywhere,
for every Kronecker mode, once each factor's EMA has warmed up (``0.08^k`` residual from the
identity seed, ``k`` = number of ``TCov``-cadence updates). That is a claim about one scalar's
*magnitude*; this script tests its behavioural consequence directly: replace ``kfac``'s per-module
Kronecker-factored divisor with a single global scalar that follows the same warmup shape, and see
whether training looks the same.

WarmupMomentumSGD:  m <- beta*m + (1-beta)*g  (+ coupled weight decay, AdaFisherMulti's own
convention -- ``optimizer.py::_update_moment``), theta <- theta - (lr/divisor(t)) * m/(1-beta^t),
divisor(t) = Lambda + (1 - gammas[0]) ** (t // TCov). This literally replays kfac.py's own
identity-seeded-EMA shape (``_A``/``_B`` seeded to ``I`` at step 0, EMA'd by ``update_running_avg``
with gain ``1 - gammas[0]`` every ``TCov`` steps -- see ``ema.py``, ``kfac.py:74-84``) collapsed to
a scalar: divisor = 1 + Lambda for the first TCov steps, then multiplied by (1 - gammas[0]) at every
TCov boundary, converging to ~Lambda once (1-gammas[0])^k is negligible (k=3 -> 5.1e-4, half of the
project's own Lambda=1e-3 -- do not shorten the run below a few hundred steps or the two optimizers
are compared before kfac's own EMA has warmed up either).

Protocol: for each model, run kfac twice (seed 0, seed 1 -- the seed-noise floor) and
WarmupMomentumSGD once (seed 0, so its init and data order are IDENTICAL to kfac/seed0 -- the only
thing that differs is the optimizer). All three share one nominal-cosine LR schedule over the same
--epochs, the model's own bench.py hyperparameters (lr, weight_decay, Lambda, gammas, TCov),
--budget-mode epochs (no WCT confound: identical step count for every arm).

Success reads on (train loss trajectory, val accuracy trajectory, test accuracy, final-theta
distance from init): |kfac_seed0 - warmup_sgd_seed0| should be of the same order as
|kfac_seed0 - kfac_seed1|. If kfac-vs-SGD is instead a large multiple of kfac-vs-kfac (seed noise),
the "divisor ~= Lambda everywhere" measurement is missing something (a checkpoint region, a layer
type, ...).

Env vars: WARMUP_SGD_MODELS (comma-separated benchmarks/<name> folders, default
"mlp_ln_mnist,cnn_gn_cifar,resnet20_cifar"), WARMUP_SGD_MODES (comma-separated Fisher modes,
default "kfac"), WARMUP_SGD_EPOCHS (default 15), WARMUP_SGD_SEEDS (comma-separated, default "0,1"
-- the first is shared with WarmupMomentumSGD, the rest are extra seeds for the noise floor),
WARMUP_SGD_LAMBDAS (comma-separated Lambda overrides, e.g. "1e-4,1e-6,1e-8"; unset = each bench's
own Lambda, one output file per value otherwise), WARMUP_SGD_DEVICE (default "auto"),
WARMUP_SGD_DATA_ROOT, WARMUP_SGD_NUM_WORKERS (default 4), WARMUP_SGD_ALLOW_DOWNLOAD ("1" to allow).
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
import torch.nn as nn

from benchmarks.common.loop import evaluate, train_under_budget
from benchmarks.common.optimizers import build_optimizer, resolve_device
from benchmarks.common.runner import Benchmark, discover_benchmarks
from benchmarks.common.schedules import NominalCosine

MODELS = os.environ.get("WARMUP_SGD_MODELS", "mlp_ln_mnist,cnn_gn_cifar,resnet20_cifar").split(",")
EPOCHS = int(os.environ.get("WARMUP_SGD_EPOCHS", "15"))
SEEDS = [int(s) for s in os.environ.get("WARMUP_SGD_SEEDS", "0,1").split(",")]
DEVICE = resolve_device(os.environ.get("WARMUP_SGD_DEVICE", "auto"))
DATA_ROOT = os.environ.get("WARMUP_SGD_DATA_ROOT", str(ROOT / "benchmarks" / "data"))
NUM_WORKERS = int(os.environ.get("WARMUP_SGD_NUM_WORKERS", "4"))
ALLOW_DOWNLOAD = os.environ.get("WARMUP_SGD_ALLOW_DOWNLOAD", "0") == "1"

OUT = ROOT / "fisher_ref" / "outputs" / "warmup_sgd_baseline.json"


MODES = os.environ.get("WARMUP_SGD_MODES", "kfac").split(",")
SGD_ALL_SEEDS = os.environ.get("WARMUP_SGD_SGD_ALL_SEEDS", "0") == "1"  # SGD at every seed, not seed 0 only
OUT = Path(os.environ.get("WARMUP_SGD_OUT", str(OUT)))

# Optional Lambda sweep, overriding each bench's own hp.lam for BOTH the real arm and the stand-in
# (kept identical to each other so only Lambda itself varies). None (the default) keeps each
# bench's own Lambda, exactly as before. Raw curvature (A, B, s*, Theta, Phi/Psi) never depends on
# Lambda -- only the damping added on top of it does -- so shrinking Lambda is expected to bring
# the two closer to being comparable (docs handoff, the lambda-vs-curvature crossover estimates).
LAMBDAS: List[Optional[float]] = (
    [float(x) for x in os.environ["WARMUP_SGD_LAMBDAS"].split(",")]
    if os.environ.get("WARMUP_SGD_LAMBDAS") else [None]
)

# AdaFisherMulti._prepare_model's own ownership rule: parameters of any other module (GroupNorm,
# positional embeddings, ...) take the fallback step with divisor 1, never the preconditioned one.
HOOKED_TYPES = ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm")


def hooked_param_ids(model: nn.Module) -> set:
    ids = set()
    for m in model.modules():
        if type(m).__name__ in HOOKED_TYPES and getattr(m, "weight", None) is not None:
            ids.add(id(m.weight))
            if m.bias is not None:
                ids.add(id(m.bias))
    return ids


class WarmupMomentumSGD(torch.optim.Optimizer):
    """Momentum-SGD whose scalar divisor replays one Fisher mode's identity-seeded EMA warmup.

    ``schedule="ekfac"`` (also tekfac): s* is set to ones at the first refresh and first EMA'd at
    step TCov, so the applied divisor is ``Lambda + decay**(t // TCov)`` (1 + Lambda on steps
    0..TCov-1). ``schedule="kfac"``: A and B are seeded to I and EMA'd *at* step 0 before the first
    inverse, so after ``k = t // TCov + 1`` updates both are ~``decay**k * I`` and each receives
    ``sqrt(Lambda)`` of damping (pi ~ 1 while the seed dominates): divisor
    ``(decay**k + sqrt(Lambda))**2``, which tends to ``Lambda``. Both schedules ignore the data part
    of the factors, which the curvature measurement found negligible once the seed has decayed.
    Parameters outside hooked modules use divisor 1, as AdaFisherMulti's fallback step does.
    """

    def __init__(self, params, lr: float = 1e-3, beta: float = 0.9, Lambda: float = 1e-3,
                 gamma0: float = 0.92, TCov: int = 100, weight_decay: float = 0.0,
                 schedule: str = "kfac", hooked_ids: Optional[set] = None,
                 decoupled_weight_decay: bool = False) -> None:
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.Lambda = Lambda
        self.decay = 1 - gamma0  # 0.08 at the project's defaults
        self.TCov = TCov
        self.schedule = schedule
        self.hooked_ids = hooked_ids
        self.decoupled_weight_decay = decoupled_weight_decay
        self.steps = 0

    def divisor(self) -> float:
        if self.schedule == "kfac":
            k = self.steps // self.TCov + 1
            return (self.decay ** k + self.Lambda ** 0.5) ** 2
        return self.Lambda + self.decay ** (self.steps // self.TCov)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        d_hooked = self.divisor()
        for group in self.param_groups:
            lr, beta, wd = group["lr"], group["beta"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                d = d_hooked if self.hooked_ids is None or id(p) in self.hooked_ids else 1.0
                grad = p.grad
                if wd != 0 and not self.decoupled_weight_decay:
                    grad = grad.add(p, alpha=wd)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg"].mul_(beta).add_(grad, alpha=1 - beta)
                state["step"] += 1
                bias_correction = 1 - beta ** state["step"]
                if self.decoupled_weight_decay and wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(state["exp_avg"], alpha=-lr / (d * bias_correction))
        self.steps += 1


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def run_variant(bench: Benchmark, seed: int, optimizer_arm: Optional[str],
                 warmup_sgd: bool, schedule: str = "kfac",
                 lam_override: Optional[float] = None) -> Dict[str, Any]:
    """``optimizer_arm='kfac'`` uses the real production path (``build_optimizer``); ``warmup_sgd``
    builds :class:`WarmupMomentumSGD` at the same hyperparameters instead. Mirrors
    ``benchmarks/common/runner.py::run_arm`` (same seeding order, same NominalCosine schedule) minus
    checkpoints/plots, which this analysis does not need.
    """
    hp = bench.hparams if lam_override is None else replace(bench.hparams, lam=lam_override)
    torch.manual_seed(seed)  # identical init + data order for kfac/seed0 vs warmup_sgd/seed0
    model_kwargs = {k: v[0] for k, v in bench.model_choices.items()}
    model = bench.build_model(**model_kwargs).to(DEVICE)
    theta0 = flatten_params(model)

    if warmup_sgd:
        optimizer: Any = WarmupMomentumSGD(
            model.parameters(), lr=hp.lr, beta=hp.beta, Lambda=hp.lam,
            gamma0=hp.gammas[0], TCov=hp.tcov, weight_decay=hp.weight_decay,
            schedule=schedule, hooked_ids=hooked_param_ids(model),
            decoupled_weight_decay=hp.decoupled_wd,
        )
        label = f"sgd_{schedule}"
    else:
        assert optimizer_arm is not None
        optimizer = build_optimizer(optimizer_arm, model, hp)
        label = optimizer_arm

    train_loader, val_loader, test_loader = bench.build_data(
        DATA_ROOT, batch_size=bench.batch_size, seed=seed, num_workers=NUM_WORKERS,
        cutout=True, allow_download=ALLOW_DOWNLOAD, train_subset=None,
    )
    scheduler = NominalCosine(optimizer, t_max=EPOCHS)
    eval_fn = None
    if val_loader is not None:
        eval_fn = lambda: evaluate(  # noqa: E731
            model, val_loader, bench.loss_fn, DEVICE,
            prepare_batch=bench.prepare_batch, metric_fn=bench.metric_fn,
        )

    t0 = time.time()
    steps, epochs = train_under_budget(
        model, optimizer, train_loader, bench.loss_fn,
        budget_s=float("inf"), max_epochs=EPOCHS, device=DEVICE, scheduler=scheduler,
        eval_fn=eval_fn, prepare_batch=bench.prepare_batch, log_fn=print,
    )
    elapsed = time.time() - t0

    test_loss, test_acc = (float("nan"), float("nan"))
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, bench.loss_fn, DEVICE,
                                       prepare_batch=bench.prepare_batch,
                                       metric_fn=bench.metric_fn)

    theta_final = flatten_params(model)
    return {
        "label": f"{label}_seed{seed}",
        "seed": seed,
        "wall_s": elapsed,
        "n_steps": len(steps),
        "step_loss": [s.loss for s in steps],
        "epoch_train_loss": [e.train_loss for e in epochs],
        "epoch_val_loss": [e.val_loss for e in epochs],
        "epoch_val_acc": [e.val_acc for e in epochs],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "theta_move_from_init": float((theta_final - theta0).norm()),
        "theta_final_norm": float(theta_final.norm()),
        # kept so cross-run parameter distance is computable after the fact (small models only;
        # skip for anything with millions of params to keep the JSON small).
        "theta_final": theta_final.tolist() if theta_final.numel() <= 300_000 else None,
    }


def summarize(model_name: str, runs: List[Dict[str, Any]], mode: str = "kfac") -> None:
    by_label = {r["label"]: r for r in runs}
    kfac0 = by_label.get(f"{mode}_seed{SEEDS[0]}")
    sgd0 = by_label.get(f"sgd_{mode}_seed{SEEDS[0]}")
    other_seeds = [by_label[f"{mode}_seed{s}"] for s in SEEDS[1:] if f"{mode}_seed{s}" in by_label]
    runs = [r for r in runs if r["label"].startswith((f"{mode}_", f"sgd_{mode}_"))]
    if kfac0 is None or sgd0 is None:
        return
    print(f"\n=== {model_name} ===")
    print(f"{'':>24s} {'test_loss':>10s} {'test_acc':>9s} {'final_train_loss':>17s} "
          f"{'theta_move':>11s}")
    for r in runs:
        ftl = r["epoch_train_loss"][-1] if r["epoch_train_loss"] else float("nan")
        print(f"{r['label']:>24s} {r['test_loss']:10.4f} {r['test_acc']*100:8.2f}% "
              f"{ftl:17.4f} {r['theta_move_from_init']:11.4f}")
    for other in other_seeds:
        d_seed = abs(kfac0["test_loss"] - other["test_loss"])
        d_sgd = abs(kfac0["test_loss"] - sgd0["test_loss"])
        a_seed = abs(kfac0["test_acc"] - other["test_acc"])
        a_sgd = abs(kfac0["test_acc"] - sgd0["test_acc"])
        print(f"  |{mode}_seed{SEEDS[0]} - {other['label']}| test_loss={d_seed:.4f} "
              f"test_acc={a_seed*100:.2f}pp   vs.   |{mode} - sgd_{mode}| "
              f"test_loss={d_sgd:.4f} test_acc={a_sgd*100:.2f}pp   "
              f"ratio(loss)={d_sgd / max(d_seed, 1e-12):.2f}x  "
              f"ratio(acc)={a_sgd / max(a_seed, 1e-12):.2f}x")


def safe_run_variant(bench: Benchmark, seed: int, optimizer_arm: Optional[str], warmup_sgd: bool,
                      schedule: str = "kfac", lam_override: Optional[float] = None) -> Dict[str, Any]:
    """``run_variant``, but a crash (e.g. a singular Kronecker factor at a very small Lambda --
    measured, not hypothetical: ``kfac`` hit ``linalg.inv``'s "matrix is singular" on
    ``mlp_ln_mnist`` at Lambda as large as 3e-5) is caught and turned into a NaN-filled
    placeholder instead of aborting every model/mode/lambda still queued behind it. The next call
    still gets a fresh model and optimizer (``torch.manual_seed`` at the top of ``run_variant``),
    so nothing leaks across the boundary.
    """
    label_hint = f"sgd_{schedule}" if warmup_sgd else optimizer_arm
    try:
        return run_variant(bench, seed, optimizer_arm, warmup_sgd, schedule, lam_override)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        print(f"  [CRASHED] {label_hint}_seed{seed}: {type(exc).__name__}: {exc}", flush=True)
        return {
            "label": f"{label_hint}_seed{seed}", "seed": seed, "wall_s": float("nan"),
            "n_steps": 0, "step_loss": [], "epoch_train_loss": [], "epoch_val_loss": [],
            "epoch_val_acc": [], "test_loss": float("nan"), "test_acc": float("nan"),
            "theta_move_from_init": float("nan"), "theta_final_norm": float("nan"),
            "theta_final": None, "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    benches = discover_benchmarks()
    print(f"warmup_sgd_baseline | device={DEVICE} | epochs={EPOCHS} | seeds={SEEDS} | "
          f"models={MODELS} | lambdas={LAMBDAS}\n", flush=True)

    for lam in LAMBDAS:
        out_path = OUT if lam is None else OUT.with_stem(OUT.stem + f"_lam{lam:.0e}")
        results: Dict[str, List[Dict[str, Any]]] = {}
        if lam is not None:
            print(f"\n########## Lambda override = {lam:.3g} ##########\n", flush=True)

        for model_name in MODELS:
            bench = benches[model_name]
            runs: List[Dict[str, Any]] = []
            results[model_name] = runs
            for mode in MODES:
                for seed in SEEDS:
                    print(f"--- {model_name} {mode} seed={seed} (lambda={lam}) ---", flush=True)
                    runs.append(safe_run_variant(bench, seed, mode, warmup_sgd=False,
                                                 lam_override=lam))
                for seed in (SEEDS if SGD_ALL_SEEDS else SEEDS[:1]):
                    print(f"--- {model_name} sgd_{mode} seed={seed} (lambda={lam}) ---", flush=True)
                    runs.append(safe_run_variant(bench, seed, None, warmup_sgd=True, schedule=mode,
                                                 lam_override=lam))
                summarize(model_name, runs, mode)

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump({"epochs": EPOCHS, "seeds": SEEDS, "modes": MODES,
                               "lambda": lam, "device": str(DEVICE), "results": results},
                              f, indent=1)
                print(f"\nwrote {out_path}", flush=True)


# On macOS the DataLoader's workers are started by `spawn`, which re-imports this file, and
# without this guard they hang at 0% CPU (lambda_vs_curvature.py hit this first). Linux forks
# instead, which is why the cluster never saw it -- kept anyway for a local rerun.
if __name__ == "__main__":
    main()
