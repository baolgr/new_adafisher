"""How long does a fresh optimizer's IDENTITY seed dominate its own state? (cnn_gn_cifar)

update_input_factor sets the EMA to `I` at step 0, then contracts by 0.08 per TCov. So after k
factor updates the state is  0.08^k * I  +  0.008 * sum_{j<k} 0.08^j X_{t-j}.  The seed enters with
norm sqrt(d) while the accumulated part is only ~0.0087*||X||, so "how many updates until the seed
is negligible" is NOT answered by 0.08^k alone. Measured here, at frozen theta (lr=0), for the
widest layer of each mode's own state.

Result, 2026-09-11, cnn_gn_cifar's widest conv (``features.8``), as a fraction of ||state||:

    kfac   k=1 7.7e-1  k=2 1.0e-1  k=3 8.2e-3  k=5 5.2e-5  k=10 1.7e-10
    diag   k=1 9.7e-1  k=2 7.3e-1  k=3 1.6e-1  k=5 1.2e-3  k=10 3.8e-09

Read with ``rewarm_fidelity.py``: the seed is small in *norm* by k=3, yet the applied preconditioner
is still 12 % off there, because what matters is the inverse. The seed shifts every eigenvalue of
the factor by 0.08^k, i.e. it is an extra damping — at k=3 that is 5.1e-4, half of lambda=1e-3.
Hence the protocol rule 0.08^k << lambda (plan_exp_draft.md §3.2).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from adafisher_modes import AdaFisherMulti

from benchmarks.common.runner import discover_benchmarks

TCOV, BATCH, STEPS = 100, 128, 1200
BENCH = discover_benchmarks()["cnn_gn_cifar"]
FAMILY = {"diag": "_H", "kfac": "_A", "ekfac": "_A", "tkfac": "_Phi_raw", "tekfac": "_Phi_raw"}

torch.manual_seed(0)
train, _, _ = BENCH.build_data(str(ROOT / "benchmarks/data"), batch_size=BATCH, seed=0,
                              num_workers=0, cutout=True, allow_download=False)

print(f"{'mode':<8}{'module':<12}" + "".join(f"{'k=' + str(k):>10}" for k in range(1, 13)))
print("        (fraction of ||state|| explained by the identity seed 0.08^k I)")
for mode, family in FAMILY.items():
    torch.manual_seed(0)
    model = BENCH.build_model()
    extra = {"diag": {}, "kfac": {"T_inv": 100}, "ekfac": {"T_eig": 100},
             "tkfac": {"T_inv": 100}, "tekfac": {"T_eig": 100, "T_re": 1}}[mode]
    opt = AdaFisherMulti(model, lr=0.0, beta=0.9, Lambda=1e-3, gammas=[0.92, 0.008], TCov=TCOV,
                         weight_decay=5e-4, fisher_mode=mode, **extra)
    target = max(opt.modules, key=lambda m: m.weight.numel())
    name = {id(m): n for n, m in model.named_modules()}[id(target)]
    ratios, done, it = [], 0, iter(train)
    while done < STEPS:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train); batch = next(it)
        x, y = BENCH.prepare_batch(batch)
        opt.zero_grad(); BENCH.loss_fn(model(x), y).backward(); opt.step()
        done += 1
        if done % TCOV == 0:
            k = done // TCOV                       # updates applied: steps 0, TCov, ..., (k-1)TCov
            state = getattr(opt.approx, family)[target].double()
            seed = 0.08 ** k * (torch.eye(state.size(0), dtype=torch.float64) if state.dim() == 2
                                else torch.ones_like(state))
            ratios.append(float(seed.norm() / state.norm()))
    print(f"{mode:<8}{name:<12}" + "".join(f"{r:>10.2e}" for r in ratios))
