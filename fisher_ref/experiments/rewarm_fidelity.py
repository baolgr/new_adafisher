"""Re-warm fidelity on cnn_gn_cifar, v2 (plan_exp_draft.md §3.2).

v1 was invalidated by its own numbers: a "noise floor" of 4.6e-7 between two re-warms on *different*
batches is impossible unless the state is dominated by something batch-independent. It is — the EMA
is seeded with the IDENTITY at step 0 (kfac.py:77 and friends), and that seed decays as 0.08^k while
carrying norm sqrt(d), against an accumulated state of norm ~0.0087*||X||. At k = 3 updates the seed
still dominates. v1's "real" run had 4 updates; a real checkpoint (cnn_gn_cifar ckpt_0.5, step 5265)
has 52.

v2 therefore: trains 2000 steps (20 updates, seed residual 0.08^19 ~ 0) before snapshotting, and
sweeps the re-warm length k*TCov in {300, 600, 1000} inside ONE re-warm run per variant (the state
is snapshotted in passing). It also compares the *primary* EMA state only — the derived caches
(inverses, eigenbases) are recomputed by refresh() and would otherwise dominate the aggregate with
their 1/lambda scale — and reports per family.

  S_real  the state a real run holds at theta*
  S_f1    re-warm from theta*, theta FROZEN (lr=0), fresh optimizer, batch order A
  S_f2    same, batch order B      -> ||S_f1 - S_f2|| is the estimator's own NOISE FLOOR
  S_m     re-warm, theta MOVING    -> the staleness term

Result, 2026-09-11 (the table recorded in ``plan_exp_draft.md`` §3.2): at k = 3 updates the four
Kronecker modes are 12-87 % wrong on the *applied preconditioner* and tkfac/tekfac's raw state is
off by six orders of magnitude; at k = 10 every mode sits within 1.3-4.5x of its own batch-draw
noise floor. The cause is not the EMA forgetting its history (0.08^k) but the identity seed acting
as a **spurious extra damping of 0.08^k on top of lambda** — so the rule is ``0.08^k << lambda``,
not ``0.08^k << 1``. See ``identity_seed_residual.py`` for the seed's own decay.
"""
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import torch
from adafisher_modes import AdaFisherMulti

from benchmarks.common.runner import discover_benchmarks

TRAIN_STEPS = 2000
REWARM_AT = (300, 600, 1000)
TCOV, LR, BATCH = 100, 1e-3, 128
MODES = ["diag", "kfac", "ekfac", "tkfac", "tekfac"]
# The primary EMA'd state. Everything else (_A_inv, _Q_A, _Phi_inv, _delta_at_refresh, ...) is
# recomputed by refresh() from these, and carries a 1/lambda scale that would swamp the aggregate.
PRIMARY = {"diag": ("_H", "_S"), "kfac": ("_A", "_B"), "ekfac": ("_A", "_B", "_s_star"),
           "tkfac": ("_delta", "_Phi_raw", "_Psi_raw"),
           "tekfac": ("_delta", "_Phi_raw", "_Psi_raw", "_Theta")}

BENCH = discover_benchmarks()["cnn_gn_cifar"]
DEV = torch.device("cpu")


def loader(seed):
    torch.manual_seed(seed)
    train, _, _ = BENCH.build_data(str(ROOT / "benchmarks/data"), batch_size=BATCH, seed=0,
                                   num_workers=0, cutout=True, allow_download=False)
    return train


def run(model, optimizer, steps, data_seed, snapshot_at=(), snapshot_fn=None):
    train, done, it = loader(data_seed), 0, None
    it = iter(train)
    out = {}
    while done < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train); batch = next(it)
        x, y = BENCH.prepare_batch(batch)
        optimizer.zero_grad()
        BENCH.loss_fn(model(x), y).backward()
        optimizer.step()
        done += 1
        if done in snapshot_at:
            out[done] = snapshot_fn()
    return out


def make(mode, lr, state_dict=None):
    torch.manual_seed(0)
    model = BENCH.build_model().to(DEV)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    extra = {"diag": {}, "kfac": {"T_inv": 100}, "ekfac": {"T_eig": 100},
             "tkfac": {"T_inv": 100}, "tekfac": {"T_eig": 100, "T_re": 1}}[mode]
    return model, AdaFisherMulti(model, lr=lr, beta=0.9, Lambda=1e-3, gammas=[0.92, 0.008],
                                 TCov=TCOV, weight_decay=5e-4, fisher_mode=mode, **extra)


def state_of(model, opt, mode):
    names = {id(m): n for n, m in model.named_modules()}
    return {fam: {names[id(m)]: t.detach().clone().double()
                  for m, t in getattr(opt.approx, fam).items() if id(m) in names}
            for fam in PRIMARY[mode]}


def rel(a, b, per_family=False):
    fams = {}
    for fam in sorted(set(a) & set(b)):
        num = den = 0.0
        for name in sorted(set(a[fam]) & set(b[fam])):
            x, y = a[fam][name], b[fam][name]
            num += float((x - y).norm()) ** 2
            den += float(x.norm()) ** 2
        fams[fam] = num ** 0.5 / den ** 0.5 if den else float("nan")
    if per_family:
        return fams
    num = sum(float((a[f][n] - b[f][n]).norm()) ** 2 for f in fams for n in a[f] if n in b[f])
    den = sum(float(a[f][n].norm()) ** 2 for f in fams for n in a[f])
    return num ** 0.5 / den ** 0.5


def precond_of(model, opt, directions):
    out = {}
    for name, module in model.named_modules():
        if name not in directions:
            continue
        opt.approx.refresh(module, 0)
        m_w, m_b = directions[name]
        d = opt.approx.precondition(module, m_w, m_b)
        out[name] = torch.cat([t.flatten().double() for t in (d if isinstance(d, tuple) else (d,))])
    return {"precond": out}


print(f"cnn_gn_cifar | real run {TRAIN_STEPS} steps ({TRAIN_STEPS // TCOV} factor updates), "
      f"re-warm sweep {REWARM_AT}, batch {BATCH}, {DEV}", flush=True)
print(f"{'mode':<8}{'k*TCov':>8}{'re-warm gap':>13}{'noise floor':>13}{'staleness':>12}"
      f"{'| precond gap':>15}{'p-floor':>10}{'p-stale':>10}", flush=True)

for mode in MODES:
    t0 = time.time()
    model, opt = make(mode, LR)
    run(model, opt, TRAIN_STEPS, data_seed=0)
    theta = copy.deepcopy(model.state_dict())
    s_real = state_of(model, opt, mode)
    torch.manual_seed(7)
    directions = {n: (torch.randn_like(m.weight), None if m.bias is None else torch.randn_like(m.bias))
                  for n, m in model.named_modules() if m in opt.modules}
    p_real = precond_of(model, opt, directions)

    snaps, psnaps = {}, {}
    for tag, lr, seed in (("f1", 0.0, 101), ("f2", 0.0, 202), ("m", LR, 303)):
        m2, o2 = make(mode, lr, state_dict=theta)
        got = run(m2, o2, max(REWARM_AT), data_seed=seed, snapshot_at=REWARM_AT,
                  snapshot_fn=lambda: (state_of(m2, o2, mode), precond_of(m2, o2, directions)))
        if lr == 0.0:
            assert all(torch.equal(a, b) for a, b in zip(m2.state_dict().values(), theta.values()))
        snaps[tag] = {k: v[0] for k, v in got.items()}
        psnaps[tag] = {k: v[1] for k, v in got.items()}

    for k in REWARM_AT:
        print(f"{mode if k == REWARM_AT[0] else '':<8}{k:>8}"
              f"{rel(s_real, snaps['f1'][k]):>13.2e}{rel(snaps['f1'][k], snaps['f2'][k]):>13.2e}"
              f"{rel(snaps['f1'][k], snaps['m'][k]):>12.2e}"
              f"{rel(p_real, psnaps['f1'][k]):>15.2e}"
              f"{rel(psnaps['f1'][k], psnaps['f2'][k]):>10.2e}"
              f"{rel(psnaps['f1'][k], psnaps['m'][k]):>10.2e}", flush=True)
    print(f"   per-family at k*TCov={REWARM_AT[-1]}: gap="
          f"{ {f: f'{v:.1e}' for f, v in rel(s_real, snaps['f1'][REWARM_AT[-1]], True).items()} } "
          f"floor={ {f: f'{v:.1e}' for f, v in rel(snaps['f1'][REWARM_AT[-1]], snaps['f2'][REWARM_AT[-1]], True).items()} }"
          f"  [{time.time() - t0:.0f}s]", flush=True)
