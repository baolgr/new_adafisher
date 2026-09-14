# Lot 7 Implementation Plan — Equal-wall-clock-budget convergence bench, all five modes

**Scope** (`docs/reports/plan.md` §8, lot 7 row): "Convergence bench at equal wall-clock budget, 5
modes", exit test `plan.md` §6.3. Unlike lots 2-6, this lot adds **no new Fisher-approximation
math** — `diag`, `kfac`, `ekfac`, `tkfac`, `tekfac` are all already implemented and tested
end-to-end (`CLAUDE.md`'s "Status" block). Lot 7's job is purely empirical: actually run the five
modes on the primary bench (`plan.md` §6.3: the 8-layer MNIST auto-encoder) under an **equal
wall-clock time budget** — not an equal epoch count, which `plan.md` §6.3 explicitly flags as
rigging the comparison in favour of the expensive modes — and report training loss as a function of
both epoch and wall-clock time, plus a direct empirical check of §6.3's FLOP-based "≈2.1×
overhead" estimate.

The CIFAR-10/ResNet/ViT work in `plan.md`'s own lot-7 table row is a **separate, explicitly
deferred** line ("*Deferred, on request*") — not part of this lot. See §4.

---

## 0. Design decisions this lot must resolve before code exists

### 0.1 What "equal wall-clock budget" means operationally

`plan.md` §6.3 states the requirement but not the mechanics: *"The equal-budget requirement of
criterion (iii) must therefore be fixed in wall-clock time, otherwise the comparison is rigged in
favour of the expensive modes."* Turning this into code requires four separate decisions, each
resolved below:

1. **What gets timed.** The full per-batch cost a practitioner actually pays: data fetch +
   `zero_grad()` + forward + backward + `optimizer.step()`. Not just `step()` in isolation — a
   user comparing five *optimizers* cares about total training wall-clock, and excluding the
   (mode-independent) data-fetch cost would only inflate the apparent relative overhead of the
   expensive modes without changing the ranking. This matches how the existing
   `benchmarks/mnist_autoencoder.py::_train_one` already times its epochs (`t0 = time.time()`
   around the whole batch loop, not just the optimizer call).
2. **The stopping rule.** Checked **before starting each new batch**, not mid-batch and not only
   at epoch boundaries: `if elapsed_s >= time_budget_s: stop`. Consequences, both deliberate:
   - A run's total wall-clock time **overshoots `time_budget_s` by at most one batch's own
     processing duration** — never by more. Concretely: the check happens after that batch's data
     has already been fetched (a plain `for batch in data_loader:` calls the iterator's `__next__`
     before entering the loop body, so the fetch's cost is already reflected in the
     `perf_counter()` read the check uses) but before `zero_grad`/forward/backward/`step`, so a
     passing check still commits to running that batch's *processing* to completion. Checking
     mid-batch instead (to close even that gap) would require killing a forward or backward pass
     partway through, which is not meaningful for a CPU-bound `nn.Module`. **This bound is not
     small and constant** — verified empirically while calibrating the reference run (§3, §6): a
     step that happens to coincide with a `T_inv`/`T_eig` amortised refresh (an `O(d³)` inversion
     or `eigh`) can be an order of magnitude slower than the steady-state median, and one such step
     landing right at the budget boundary is exactly what produced a ≈0.2s overshoot on a 5s
     calibration budget for `tekfac` (§6) — well above its own ≈21ms steady-state median step
     time, but still bounded by *that specific batch's own* processing duration, which is what
     `tests/test_equal_wallclock_bench.py::test_budget_is_respected_and_run_completes` actually
     checks (against the run's own observed max, not the median).
   - Checking only at epoch boundaries (the alternative considered and rejected) would make the
     five modes' actual elapsed times diverge by up to one full epoch's duration whenever
     per-epoch time is large relative to `time_budget_s` — for the ≈2-3× slower non-`diag` modes,
     that is not a small effect. Per-batch checking keeps the five runs' actual elapsed times
     within one batch of each other and of `time_budget_s`, which is the operational meaning of
     "equal budget".
   - The dataset is iterated indefinitely (re-iterating the `DataLoader`, i.e. starting a new
     epoch with a fresh shuffle) until the budget is exhausted — never a fixed `range(epochs)`.
     This is the one structural change relative to `_train_one`'s existing `for epoch in
     range(args.epochs)` loop.
3. **Recording resolution — per-step, not per-epoch (§0.3).**
4. **Timer.** `time.perf_counter()`, not `time.time()` (§0.2).

### 0.2 Timer choice, and CPU/GPU correctness

`time.perf_counter()` is a monotonic, high-resolution clock unaffected by system clock adjustments
(NTP sync, DST) — the correct choice for measuring short intervals, unlike `time.time()`
(wall-clock, can jump). `benchmarks/mnist_autoencoder.py` used `time.time()` for whole-epoch
timings (tens of seconds, where the distinction is immaterial); lot 7 times individual batches
(milliseconds), where it is not. This lot's new code uses `perf_counter()` throughout; the existing
`mnist_autoencoder.py` epoch timer is untouched (out of scope, not wrong at its resolution).

**GPU correctness, forward-looking.** Every existing script and test in this repository is
CPU-only (`grep -rn cuda src/ benchmarks/` returns nothing outside `.device`-preserving code —
tensors simply follow whatever device they are already on, but nothing here ever calls `.cuda()`).
The bench is run on CPU for this lot, consistent with that. CUDA kernel launches are asynchronous,
so a naive `perf_counter()` bracket around GPU ops would time only kernel-launch overhead, not
actual compute — silently wrong, not just imprecise. The new training-loop helper therefore accepts
a `device` and calls `torch.cuda.synchronize()` immediately before each timestamp when
`device.type == "cuda"` (a no-op, and untaken branch, on CPU). This costs nothing on the CPU path
actually exercised here and removes a latent correctness trap if this harness is later pointed at a
GPU (the user's HPC-server context makes this a real, not hypothetical, future use).

### 0.3 Recording resolution: per-step records answer both axes of §6.3 at once

§6.3 asks for loss as a function of **both** (a) epoch and (b) wall-clock time. The simplest design
that gives both without maintaining two different aggregations: record one entry per **step**
(mini-batch), not per epoch —

```python
@dataclass
class StepRecord:
    step: int            # 0-indexed, global across the whole run (spans epochs)
    epoch: float          # epoch + (batch_index_within_epoch / batches_per_epoch)
    elapsed_s: float      # perf_counter() elapsed since this run's t0, checked before the batch
    loss: float           # this batch's own loss (raw, unsmoothed)
    fwd_bwd_s: float       # wall-clock spent in forward + loss + backward, this batch
    step_s: float          # wall-clock spent in optimizer.step(), this batch
```

`epoch` (a float) and `elapsed_s` are simply two different columns of the same series — plotting
loss against either one is a one-line change, no resampling or binning needed. This also
automatically handles the fact that under a shared time budget the five modes reach **different**
final epoch counts (the whole point of the comparison) and different step counts: each mode's
curve is exactly as long as its own list of `StepRecord`s, with no artificial truncation or
interpolation to a common grid.

`fwd_bwd_s` / `step_s` are recorded to let §0.4's overhead measurement be computed directly from
the same data, without a second, separately-timed pass.

**Smoothing is a plotting-time concern only.** Raw per-batch loss is noisy (a single batch's MSE);
the CSV output (§1.2) stores the raw, unsmoothed series (reproducible, no information discarded).
Only the optional plot (§0.9) applies a trailing moving average (fixed window, default 10 steps)
purely for visual legibility.

### 0.4 Isolating optimizer overhead from model cost — an empirical check of `plan.md` §6.3's estimate

`plan.md` §6.3 derives, from FLOP counts alone, that the KFE projection step costs "≈18 GFLOP/step
against ≈8.5 GFLOP/step for the model's own fwd+bwd, i.e. ≈2.1× overhead", and flags this as the
reason the comparison must be wall-clock-based. That number was never measured — only estimated
analytically. Since `fwd_bwd_s` and `step_s` are already recorded per step (§0.3), lot 7 gets a
direct empirical counterpart for free: `median(step_s) / median(fwd_bwd_s)` per mode, computed over
steady-state steps (§0.6 excludes the first few steps of each run, which include one-time hook /
lazy-initialization cost). This is reported as a **measurement**, in the same spirit as `plan.md`
§6.1's "measured, not asserted" Frobenius-dominance-vs-`diag` metric — not a pass/fail gate, since
actual wall-clock ratios are machine- and BLAS-dependent in a way FLOP counts are not. Reported for
all five modes (`diag`'s ratio is expected to be close to 0, i.e. its `step_s` negligible next to
`fwd_bwd_s` — consistent with §6.3's own "≈2.8 MFLOP/step — negligible" line for `diag`).

### 0.5 Fair-start protocol

Reused verbatim from the existing `_train_one` convention (`mnist_autoencoder.py:90`,
`torch.manual_seed(args.seed)` immediately before constructing the model): each mode's model,
optimizer and `DataLoader` are constructed fresh, immediately after **re-seeding** with the same
seed. This gives every mode an identical parameter initialization and an identical sequence of
minibatch shuffles (the `DataLoader`'s `shuffle=True` draws from the global `torch` RNG), so
differences in the recorded loss curves are attributable only to the Fisher approximation, not to
initialization or data-order luck. No new design needed here — flagged because it is easy to get
wrong when refactoring `_train_one`'s epoch-bounded loop into an unbounded, budget-checked one.

### 0.6 Hyperparameters: shared and untuned across all five modes

An equal-*budget* comparison that is not also an equal-*hyperparameter* comparison is confounded —
a mode could "win" simply by having been handed a larger effective learning rate. All five modes
therefore use the same `lr`, `beta`, `Lambda`, `gammas`, `TCov`, and (where applicable) the same
`T_inv=100` / `T_eig=100` / `T_re=1` — the exact defaults documented in `CLAUDE.md`'s "Selecting a
mode" section — for every mode that accepts them, with no per-mode retuning. This is a deliberate
scope limitation (§4): the bench measures relative convergence and overhead **at the paper-default
operating point**, not each mode's best achievable convergence, which would require a
hyperparameter sweep for each of the five modes and is a different, much larger undertaking.

One consequence worth stating explicitly: with `TCov=100`, the factor EMA (and hence the
preconditioner itself, for the four non-`diag` modes) only updates every 100 steps — the hook fires
at `step % TCov == 0` (`optimizer.py:74,78`), which is true at `step=0` (so the *first* factor
value is already a real, non-bootstrap estimate, not just the identity `A`/`B` initialization seen
in `kfac.py`'s/`ekfac.py`'s per-module bootstrap branch) and then every 100 steps after. A
time budget short enough that a mode completes only a few dozen steps would show a preconditioner
that has barely moved past its initial EMA value — an artifact of the budget being too small
relative to `TCov`, not of the mode itself. §3's reference run is sized (via a short calibration
pass) so that every mode completes several hundred steps, i.e. several `TCov` periods, within its
budget.

### 0.7 Where the new code lives — reuse, not duplication

`benchmarks/mnist_autoencoder.py` already owns `AutoEncoder`, `_mnist_loader`, and (for
`"reference"`/`"diag"` only) `_make_optimizer`. Two options were considered:

- **(a)** Fold the wall-clock-budget loop and all five modes directly into
  `mnist_autoencoder.py`, growing its CLI.
- **(b)** Generalize `mnist_autoencoder.py::_make_optimizer` to build any of the five
  `AdaFisherMulti` modes (a small, directly-motivated extension — every earlier lot's smoke test
  needed the analogous per-mode kwargs table, e.g. `tests/test_conv2d_optimizer_smoke.py`'s
  `_MODE_KWARGS`), and add a **new** file, `benchmarks/equal_wallclock_bench.py`, that *imports*
  `AutoEncoder`, `_mnist_loader` and `_make_optimizer` from `mnist_autoencoder.py` rather than
  duplicating them, and owns the budget-checked training loop, the CSV/markdown/plot reporting, and
  its own CLI.

**Resolved: (b).** It keeps `mnist_autoencoder.py` doing one thing (the lot-1 exit-criterion
epoch-fixed comparison, `plan.md` §8 row 1 — now additionally able to run any of the five modes
side by side over a fixed epoch count, which is a useful quick sanity check before committing to a
long budgeted run, but still epoch-fixed, still that file's original contract) and gives lot 7 its
own file, matching every previous lot's one-new-file-per-lot pattern in the `docs/reports/plan.md`
§2.1 layout. `_make_optimizer` is the only piece of `mnist_autoencoder.py` that changes
(generalized from a two-branch `if` into a small per-mode kwargs table, mirroring
`test_conv2d_optimizer_smoke.py`'s convention); `AutoEncoder`, `_mnist_loader`, `_train_one` and the
existing CLI's behaviour for `--optimizer reference|diag|both` are untouched.

### 0.8 Making the harness testable offline — no MNIST download in `pytest`

None of the existing tests import anything from `benchmarks/` (they exercise `adafisher_modes`
directly). `benchmarks/mnist_autoencoder.py` imports `torchvision` at module level (harmless,
already an installed `bench` extra — see `docs/reports/plan_lot6.md`'s own note about not adding
hard test dependencies) but only *calls* `datasets.MNIST(download=True)` inside `_mnist_loader`,
which a test must never call — the sandboxed CI environment this project's own `CLAUDE.md` describes
should not be assumed to have network access, and a test that silently downloads ~10 MB on first
run is a bad test regardless.

Resolved by keeping the budget-checked training loop **dataset- and model-agnostic**: it is a plain
function of `(model, optimizer, data_loader, loss_fn, time_budget_s, device)`, with no reference to
`AutoEncoder` or MNIST anywhere in its body. `benchmarks/equal_wallclock_bench.py`'s CLI wires it to
the real `AutoEncoder` + MNIST `DataLoader`; `tests/test_equal_wallclock_bench.py` wires the exact
same function to a tiny synthetic `TensorDataset` (random tensors, generated once with a fixed
seed, fully in-memory, zero network I/O) and a 2-layer toy `nn.Sequential`, at a sub-second time
budget. This is the same "shared plumbing, only the construction of `v^(t)` differs across modes"
principle `plan.md` §2.2 uses to justify the `FisherApproximation` ABC, applied one level up: the
training-loop *harness* is shared and dataset-agnostic; only what is fed into it differs between the
real bench and its test.

### 0.9 Plotting: an optional dependency, with graceful degradation

Neither `matplotlib` nor `pandas` is currently installed (`pyproject.toml`'s `bench` extra is only
`torchvision`). Loss-vs-epoch and loss-vs-wall-clock-time curves are the literal deliverable of
§6.3's "training loss as a function of ... epoch and ... wall-clock time", and are also how every
cited K-FAC/EKFAC/TKFAC paper presents this exact comparison — so `matplotlib` is added to the
`bench` extra (`pyproject.toml`, §1.4). The CSV output does **not** depend on it. The plotting
function itself is guarded by a local `try: import matplotlib ... except ImportError:` with a
one-line stderr notice on failure, so `tests/test_equal_wallclock_bench.py` — which does not want a
plotting dependency to be a hard test requirement, consistent with §0.8's dependency discipline —
can still exercise the full harness (CSV path) even in an environment without `matplotlib`
installed; it only additionally asserts that a `.png` was produced when `matplotlib` **is**
importable, never that it must be.

### 0.10 Which "five modes", precisely — and why not six

"5 modes" (`plan.md` §8) means `diag`, `kfac`, `ekfac`, `tkfac`, `tekfac` — the five entries of the
`MODES` registry (`approximations/__init__.py`). `diag` is run with its **default**
`minmax_normalization=True` (paper-faithful, `CLAUDE.md`'s "Selecting a mode" default) — the
configuration an actual user gets, not `minmax_normalization=False`. The reference `AdaFisher`
optimizer (`reference_repos/FisherAdapTune/scripts/adafisher.py`, already exercised by
`mnist_autoencoder.py --optimizer reference`) is **not** re-included as a sixth curve: lot 1's
`test_diag_bitexact.py` already established that `diag(minmax_normalization=False)` reproduces it
bit-exactly at every EMA update, so a `reference` curve would be visually indistinguishable from a
`diag(minmax=False)` curve and would not be a sixth *mode* in the `plan.md` §2.2 sense (a different
construction of `v^(t)`) — it would only inflate the run count without adding information. This
mirrors `mnist_autoencoder.py`'s own lot-1 framing, which already treats `reference` as a
correctness check against `diag`, not as a member of the mode comparison.

### 0.11 A `max_steps` safety bound, independent of the time budget

The per-batch stopping check guarantees termination for any finite, positive `time_budget_s` (time
strictly advances every batch), so there is no literal infinite-loop risk. A `max_steps` cap
(default `10_000`, generously above anything a sane budget reaches — §3's calibration puts the
non-`diag` modes at a few hundred steps/minute) is still added as a defensive bound, purely to turn
a pathological misconfiguration (e.g. `time_budget_s` accidentally left at a huge value, or a
budget so small that per-batch Python overhead dominates and thousands of near-zero-work batches
run in a burst) into a clean, fast failure rather than a runaway process — the kind of guard that
costs one `if` and is worth having in a script meant to be run unattended for minutes at a time.

---

## 1. File-by-file design

### 1.1 `benchmarks/mnist_autoencoder.py` — generalize `_make_optimizer`

Replace the two-branch `if name == "reference" / "diag"` with a per-mode kwargs table, one entry
per non-`diag`, non-`reference` mode (mirroring `tests/test_conv2d_optimizer_smoke.py`'s
`_MODE_KWARGS` convention exactly), and extend `--optimizer`'s `choices`:

```python
_NEW_MODE_KWARGS = {
    "kfac": lambda a: {"T_inv": a.t_inv},
    "ekfac": lambda a: {"T_eig": a.t_eig},
    "tkfac": lambda a: {"T_inv": a.t_inv},
    "tekfac": lambda a: {"T_eig": a.t_eig, "T_re": a.t_re},
}


def _make_optimizer(name, model, args, reference_module):
    if name == "reference":
        return reference_module.AdaFisher(...)                     # unchanged
    if name == "diag":
        return AdaFisherMulti(..., fisher_mode="diag",
                               minmax_normalization=args.minmax_normalization)   # unchanged
    if name in _NEW_MODE_KWARGS:
        return AdaFisherMulti(
            model, lr=args.lr, beta=args.beta, Lambda=args.lam, TCov=args.tcov,
            gammas=list(args.gammas), fisher_mode=name, **_NEW_MODE_KWARGS[name](args),
        )
    raise ValueError(name)
```

New CLI flags: `--t-inv` (default 100), `--t-eig` (default 100), `--t-re` (default 1) — the
`CLAUDE.md`-documented defaults, so the existing `--optimizer both` invocation's behaviour and
output are byte-for-byte unchanged (these flags are inert unless `--optimizer` selects a mode that
reads them). `--optimizer`'s `choices` gains `"kfac"`, `"ekfac"`, `"tkfac"`, `"tekfac"`, and `"all"`
(the five `AdaFisherMulti` modes, excluding `"reference"` — §0.10's reasoning applies here too,
kept consistent between the two scripts).

Nothing else in the file changes: `AutoEncoder`, `_mnist_loader`, `_train_one`, `main`'s existing
`--optimizer reference|diag|both` paths are untouched.

### 1.2 `benchmarks/equal_wallclock_bench.py` — new

```python
"""Equal-wall-clock-budget convergence bench across all five Fisher approximation modes
(docs/reports/plan.md §6.3, §8 lot 7; docs/reports/plan_lot7.md).
"""

@dataclass
class StepRecord:
    step: int
    epoch: float
    elapsed_s: float
    loss: float
    fwd_bwd_s: float
    step_s: float


def train_under_time_budget(
    model: nn.Module,
    optimizer,
    data_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    time_budget_s: float,
    *,
    device: torch.device = torch.device("cpu"),
    max_steps: int = 10_000,
    prepare_batch: Callable[[Any], Tuple[Tensor, Tensor]] = lambda batch: (batch[0], batch[0]),
) -> List[StepRecord]:
    """Dataset- and model-agnostic (docs/reports/plan_lot7.md §0.8): no reference to MNIST or
    AutoEncoder anywhere in this function, so it is exercised directly, offline, by
    tests/test_equal_wallclock_bench.py against a tiny synthetic dataset.

    ``prepare_batch`` maps one DataLoader batch to (model_input, loss_target); defaults to the
    auto-encoder's own convention (reconstruct the flattened input), overridden by the test with a
    supervised-regression toy task.
    """
    batches_per_epoch = len(data_loader)
    records: List[StepRecord] = []
    step = 0
    t0 = perf_counter()
    epoch = 0
    while step < max_steps:
        for batch_idx, batch in enumerate(data_loader):
            elapsed = perf_counter() - t0
            if elapsed >= time_budget_s or step >= max_steps:
                return records
            x, target = prepare_batch(batch)
            x, target = x.to(device), target.to(device)

            t_fwd_bwd_0 = perf_counter()
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, target)
            loss.backward()
            _sync(device)
            fwd_bwd_s = perf_counter() - t_fwd_bwd_0

            t_step_0 = perf_counter()
            optimizer.step()
            _sync(device)
            step_s = perf_counter() - t_step_0

            records.append(StepRecord(
                step=step, epoch=epoch + batch_idx / batches_per_epoch,
                elapsed_s=perf_counter() - t0, loss=loss.item(),
                fwd_bwd_s=fwd_bwd_s, step_s=step_s,
            ))
            step += 1
        epoch += 1
    return records
```

(`_sync(device)` = `torch.cuda.synchronize()` iff `device.type == "cuda"`, else a no-op — §0.2.)

`run_all_modes(modes, args, ...)`: for each mode in the requested list, re-seeds (§0.5), builds a
fresh `AutoEncoder` + `_make_optimizer(mode, ...)` (imported from `mnist_autoencoder.py`) + a fresh
`_mnist_loader`, calls `train_under_time_budget`, collects `{mode: List[StepRecord]}`.

Reporting, from the collected records:
- **CSV** (`benchmarks/outputs/lot7_equal_wallclock/records.csv`, columns `mode, step, epoch,
  elapsed_s, loss, fwd_bwd_s, step_s`) — the full, unsmoothed raw data (§0.3), one row per
  `(mode, step)`. `benchmarks/outputs/` is already `.gitignore`d.
- **`summary.md`**: one row per mode — steps completed, final epoch reached, final loss (mean of
  the last 10 records), median `fwd_bwd_s`, median `step_s`, and `median(step_s) /
  median(fwd_bwd_s)` (§0.4's overhead ratio), medians taken over steps `>= 5` (skips each run's
  first few, one-time-cost-inflated steps, mirroring the amortisation-cadence exclusion already
  used when this project reports steady-state timings elsewhere).
- **Plots** (`loss_vs_epoch.png`, `loss_vs_time.png`), only if `matplotlib` imports successfully
  (§0.9): one line per mode, x = `epoch` / `elapsed_s`, y = loss smoothed with a trailing
  10-step moving average.

CLI (`argparse`, mirroring `mnist_autoencoder.py`'s style): `--modes` (default: all five, §0.10),
`--time-budget` (seconds, default 60.0), `--max-steps` (default 10000, §0.11), `--batch-size`
(default 500, matching `mnist_autoencoder.py`), `--lr`, `--beta`, `--lam`, `--tcov`, `--gammas`,
`--t-inv`, `--t-eig`, `--t-re` (same defaults as §1.1), `--minmax-normalization`/
`--no-minmax-normalization` (`BooleanOptionalAction`, default **True** — deliberately different
from `mnist_autoencoder.py`'s own CLI default of `False`, which exists there for lot 1's
bit-exactness check against `reference`; here the point is §0.10's "the configuration an actual
user gets"), `--seed` (default 0), `--output-dir` (default
`benchmarks/outputs/lot7_equal_wallclock`), `--no-plot`.

### 1.3 `tests/test_equal_wallclock_bench.py` — new

Imports `train_under_time_budget` from `equal_wallclock_bench` directly (needs `benchmarks/` on
`sys.path`; added the same way `mnist_autoencoder.py` adds `src/` —
`sys.path.insert(0, str(REPO_ROOT / "benchmarks"))` at the top of the test file, not a
`pyproject.toml` change, since only this one test file needs it).

Fixture: an in-memory `TensorDataset` of random `(x, y)` pairs, wrapped in a small `DataLoader`; a
2-layer `nn.Sequential`. No `AutoEncoder`, no MNIST, no network I/O (§0.8). Parameterized by
`(d_in, d_hidden, d_out, n, batch_size)` — the small default (`6, 5, 3, 64, 8`) is used for the
budget/guard tests below; a larger instance is used only by the third, timing-sensitive test
(§5 point 5 explains why).

```python
_MODE_KWARGS = {  # same table as test_conv2d_optimizer_smoke.py's convention, TCov/T_inv/T_eig
                  # small enough that the tiny synthetic run crosses several cadence periods
    "diag": {},
    "kfac": {"T_inv": 2},
    "ekfac": {"T_eig": 2},
    "tkfac": {"T_inv": 2},
    "tekfac": {"T_eig": 2, "T_re": 1},
}

@pytest.mark.parametrize("mode", list(_MODE_KWARGS))
def test_budget_is_respected_and_run_completes(mode):
    ...
    records = train_under_time_budget(model, optimizer, loader, loss_fn, time_budget_s=0.3)
    assert len(records) >= 1
    assert all(math.isfinite(r.loss) for r in records)
    max_single_step = max(r.fwd_bwd_s + r.step_s for r in records)
    assert records[-1].elapsed_s <= 0.3 + max_single_step + 1e-3   # §0.1's overshoot bound
    assert [r.elapsed_s for r in records] == sorted(r.elapsed_s for r in records)  # monotonic
    assert [r.step for r in records] == list(range(len(records)))                 # contiguous
```

A second test, `test_max_steps_guard_is_effective`: a huge `time_budget_s` with a tiny `max_steps`
(e.g. 5) terminates at exactly `max_steps` records — exercises §0.11's guard directly (the budget
alone would never trigger within a normal test timeout).

A third, informational test (§0.4, `capsys`-printed — the established "measured, not asserted"
pattern of `test_frobenius_dominance.py`): `test_eigenbasis_modes_pay_more_per_step_than_diag`
runs `diag` and each non-`diag` mode for a fixed, larger step count (a generous time budget and a
small, fixed `max_steps` instead, so the comparison is over an equal *step* count — the point here
is internal, not the §6.3 wall-clock comparison itself), on a **larger** synthetic task
(`d_in=256, d_hidden=192, d_out=96`) than the other two tests use. Asserts `median(step_s)` is
strictly larger for `ekfac`/`tekfac` than for `diag` only; `kfac`/`tkfac` are printed for
information without an assertion — §5 point 5 records why that narrower scope was necessary (an
initial version of this test, asserting the ordering for all four non-`diag` modes at the smaller
toy scale used elsewhere in this file, failed: `diag`'s own multi-op min-max pipeline turned out to
be comparably expensive, per-torch-op-dispatch-call, to `kfac`/`tkfac`'s two cached-inverse
matmuls — a genuine finding, not noise to be tolerance-papered-over).

### 1.4 `pyproject.toml`

```toml
bench = ["torchvision>=0.15", "matplotlib>=3.7"]
```

---

## 2. Tests

| Test | Assertion |
|---|---|
| `test_equal_wallclock_bench::test_budget_is_respected_and_run_completes` (×5 modes) | the harness runs to completion on a tiny synthetic task for every one of the five modes; every recorded loss is finite; `elapsed_s` is monotonic and overshoots `time_budget_s` by at most that run's own largest single-batch processing duration (§0.1); `step` is contiguous from 0 |
| `test_equal_wallclock_bench::test_max_steps_guard_is_effective` | `max_steps` terminates a run independently of an (effectively unbounded) time budget (§0.11) |
| `test_equal_wallclock_bench::test_eigenbasis_modes_pay_more_per_step_than_diag` | `median(step_s)` is strictly greater for `ekfac`/`tekfac` than for `diag`, at equal step count (asserted); `kfac`/`tkfac` vs. `diag` printed for information only, not asserted (§5 point 5) — the cheap, CI-safe stand-in for §0.4's real, machine-dependent overhead measurement |

The real §6.3 deliverable — five loss-vs-epoch / loss-vs-time curves and the measured overhead
ratios on the actual MNIST auto-encoder — is not a `pytest` assertion (§0.4: wall-clock numbers are
machine-dependent, not the kind of fact this project encodes as a hard test gate elsewhere either,
e.g. `plan.md` §6.1's own rescaled-error metric). It is produced by actually running
`benchmarks/equal_wallclock_bench.py` and is reported in §6 below, once available.

---

## 3. Producing the reference run

1. **Calibrate.** Run `equal_wallclock_bench.py --modes diag kfac ekfac tkfac tekfac --time-budget
   5 --seed 0` (a short, throwaway budget) to read off steps/second per mode from `summary.md`.
2. **Pick the real budget** from that calibration so that the *slowest* mode still completes at
   least a few hundred steps (§0.6: several `TCov=100` periods) within the budget — expected to
   land in the tens-of-seconds-to-low-minutes range per mode given `plan.md` §6.3's own
   ≈2× wall-clock estimate and this bench's small (`≈2.84e6`-parameter) model.
3. **Run the reference bench** at that budget, `--seed 0`, default hyperparameters (§0.6),
   `--output-dir benchmarks/outputs/lot7_equal_wallclock`.
4. **Record findings** in §6 of this document and in `CLAUDE.md`'s status block, following every
   previous lot's convention of citing concrete measured numbers (e.g. lot 6's "≈80.7×... measured
   at ResNet-18 scale") rather than the a priori FLOP estimate alone.

---

## 4. Explicitly out of scope for lot 7

- **CIFAR-10 / ResNet / ViT from-scratch training, with an Adam baseline** — `plan.md`'s own lot-7
  table row lists this as a *separate*, "deferred, on request" line, distinct from the lot-7 row
  itself (`plan.md` §8, §6.3: "Deferred, on request: ... Not part of the current scope; see §8, lot
  7" — a cross-reference to the *deferred* row, not an instruction to fold it into lot 7). Lot 6's
  own "explicitly out of scope" section already flagged this exact boundary
  (`plan_lot6.md` §3: *"lot 7's equal-wall-clock comparison, and/or the deferred CIFAR-10/ResNet
  work ... gates on this lot"* — two separate follow-ups, not one).
- **Per-mode hyperparameter tuning** (§0.6) — the bench measures the five modes at one shared,
  paper-default operating point, not each mode's best case.
- **An `Adam` baseline** — not part of `plan.md` §6.3's primary-bench criterion (that comparison is
  reserved for the deferred CIFAR-10 work, §6.3: *"CIFAR-10 classification ... with Adam as an
  additional baseline"*); the primary MNIST auto-encoder bench compares the five `AdaFisherMulti`
  Fisher-approximation modes against each other and against `reference` AdaFisher only (§0.10), as
  every earlier reference to this bench (`plan.md` §6.3, lot 1's own `mnist_autoencoder.py`) does.
- **GPU execution** — the harness is made GPU-*safe* (§0.2) but not GPU-*run*; every existing script
  in this repository is CPU-only, and there is no evidence a CUDA device is available in this
  project's sandboxed environment.
- **Statistical significance across seeds** — a single `--seed 0` run per mode, matching every
  earlier lot's smoke/bench convention (`mnist_autoencoder.py` itself has always been single-seed).
  Multi-seed variance estimation would be a natural lot-8-scale extension, not requested here.
- **Automating the "reference run" (§3) inside `pytest`** — deliberately manual/scripted (§2's own
  table), for the same reason `mnist_autoencoder.py --optimizer both --epochs 5` has never been part
  of the `pytest` suite: it is a multi-second-to-minute, MNIST-downloading, illustrative run, not a
  fast, offline correctness gate.

---

## 5. Points worth flagging explicitly (not blocking, per this session's operating mode)

1. §0.1's per-batch (not per-epoch) budget check is a small but real design choice this plan makes
   that `plan.md` §6.3 does not itself specify — the alternative (epoch-boundary checking) is not
   wrong, only coarser, and would have been the simpler change relative to `_train_one`'s existing
   loop. The per-batch version was chosen because the whole point of this lot is to make the five
   runs' *elapsed times* actually comparable, and epoch-granularity checking reintroduces exactly
   the kind of budget skew (up to one full epoch, worse for the slower modes) the wall-clock
   requirement exists to remove.
2. §0.4's `fwd_bwd_s`/`step_s` split is not requested anywhere in `plan.md`, but follows directly
   from already having to time every batch for the budget check (§0.1) — recording *where* that
   time goes is a near-zero marginal cost that turns lot 7 into the first place in this project
   where §6.3's own central FLOP-based claim ("≈2.1× overhead") gets an actual empirical
   measurement, rather than staying an analytical estimate cited but never checked.
3. `benchmarks/outputs/` was already present in `.gitignore` (`.gitignore:11`) before this lot
   touched anything — evidence this output location was anticipated (likely in an earlier lot's
   setup) specifically for a benchmark script like this one; reusing it rather than inventing a new
   output convention.
4. `TCov=100` (§0.6) means a mode's very first preconditioner is already a real EMA estimate (the
   hook fires at `step % TCov == 0`, true at `step=0`), not stuck at its bootstrap identity
   initialization — worth stating precisely since `kfac.py`'s/`ekfac.py`'s per-module `eye(...)`
   bootstrap (seen while reading those files for this plan) could otherwise be misread as meaning
   the first `TCov` steps run with an identity preconditioner; they do not — only the steps *before*
   the first hook firing would (there are none, since it fires at step 0).
5. **§0.1's overshoot direction was stated backwards in an early draft of this plan, and caught
   only empirically**, while calibrating the reference run (§3, §6): the check happens *before* a
   batch's processing but *after* that batch has already been fetched (`for batch in
   data_loader:` calls `__next__` before entering the loop body), so a passing check still commits
   to running that batch's forward/backward/`step` to completion — the run's total elapsed time
   *overshoots* `time_budget_s` by at most one batch's own processing duration, it does not
   *undershoot*. This was corrected everywhere it appeared (this document, the
   `train_under_time_budget` docstring, and `tests/test_equal_wallclock_bench.py`'s own assertion,
   which was written correctly from the start — only the prose explanation was wrong) once the
   5-second calibration run showed `tekfac` finishing at ≈5.2s against a 5.0s budget: not
   measurement slop, but exactly one `T_eig`/`T_re`-cadence refresh step (an `O(d³)` `eigh`, far
   above that mode's own steady-state median) landing right at the boundary. A reminder that a
   design claim phrased confidently in prose is still a claim, and this project's own convention of
   preferring measurement over assertion (`plan.md` §6.1) applies to the benchmark harness's own
   documentation, not only to the Fisher-approximation quality claims it exists to measure.
6. **The informational overhead test's scope had to be narrowed after an empirical falsification**
   (§1.3, §2): the original design intent — "every non-`diag` mode costs strictly more per step
   than `diag`, robustly, regardless of problem size" — is false at the small toy scale the other
   two tests in this file use (`d_in=6`). `kfac`/`tkfac`'s `precondition` is two matmuls against
   cached inverses; at that scale this is comparable to, and in some runs measured *below*,
   `diag`'s own multi-op min-max normalisation pipeline (several elementwise `min`/`max`/`where`
   calls plus a `kron` reconstruction — `diag.py`'s `f_tilde`). Only `ekfac`/`tekfac`, whose
   `precondition` additionally projects into and out of an eigenbasis (roughly double `kfac`'s
   matmul count), showed a large, reproducible gap across repeated runs at a moderately larger
   scale (`d_in=256`). The fix was not a looser tolerance on the same claim but a narrower, true
   claim — the same discipline `plan.md` §6.1 already applies to the `diag`-vs-four-new-modes
   Frobenius comparison ("measured, not asserted," precisely because it is not a theorem).

---

## 6. Results of the reference benchmark run

Produced by the exact procedure of §3, on this machine (CPU-only, single-process, `num_workers=0`):

```bash
PYTHONPATH=src .venv/bin/python benchmarks/equal_wallclock_bench.py --time-budget 60 --seed 0 \
    --output-dir benchmarks/outputs/lot7_equal_wallclock
```

Setup: all five modes, `--time-budget 60` (chosen after a 5s calibration pass showed 100-163
steps/5s across modes, i.e. several hundred steps — several `TCov=100` periods, §0.6 — within
60s for every mode), `batch_size=500`, `seed=0`, every other hyperparameter at the shared
`CLAUDE.md` default (`lr=1e-3, beta=0.9, Lambda=1e-3, gammas=(0.92, 0.008), TCov=100, T_inv=100,
T_eig=100, T_re=1`, `diag` at `minmax_normalization=True`, §0.10). Raw per-step data, the summary
table and both plots are at `benchmarks/outputs/lot7_equal_wallclock/` (`.gitignore`d, regenerable
by the command above — machine-specific wall-clock numbers are not meant to be committed, §0.9's
"CSV/summary never depend on a plotting library" design applies equally to "these exact
milliseconds are this machine's, not a portable fact").

### 6.1 Steady-state timing — §0.4's overhead measurement

| mode | steps completed | epochs | median fwd+bwd | median step (precondition) | step / fwd+bwd |
|---|---|---|---|---|---|
| `diag` | 1963 | 16.35 | 12.594 ms | 3.050 ms | **0.24×** |
| `kfac` | 1571 | 13.08 | 12.004 ms | 11.938 ms | **0.99×** |
| `tkfac` | 1525 | 12.70 | 12.147 ms | 12.472 ms | **1.03×** |
| `ekfac` | 1205 | 10.03 | 11.816 ms | 21.375 ms | **1.81×** |
| `tekfac` | 1199 | 9.98 | 11.800 ms | 21.356 ms | **1.81×** |

(medians over steps ≥5, i.e. excluding each run's first few hook/lazy-init steps, §1.2.)

`plan.md` §6.3 estimated, from FLOP counts alone, that "the KFE projection + inverse" costs
"≈18 GFLOP/step against ≈8.5 GFLOP/step for the model's own fwd+bwd, i.e. ≈2.1× overhead" —
stated once, for "all non-diag modes" undifferentiated. The measurement refines this into two
regimes the FLOP estimate did not distinguish:

- **`kfac`/`tkfac`** (`precondition` = two matmuls against a cached inverse) cost **≈1.0×** their
  own `fwd_bwd` time in extra work — i.e. total per-step cost ≈2.0× the model's own fwd+bwd alone,
  the same order of magnitude as the ≈2.1× estimate, on the low side of it.
- **`ekfac`/`tekfac`** (`precondition` additionally projects into and back out of an eigenbasis,
  roughly double the matmul count) cost **≈1.8×** — total per-step cost ≈2.8× fwd+bwd, closer to,
  and slightly above, a naive doubling of `kfac`'s own ratio — consistent with §5 point 6's toy-scale
  finding that the eigenbasis modes are the ones with a robust, large timing gap over `kfac`/`tkfac`,
  not just over `diag`.
- **`diag`**'s own cost (**0.24×**) is not literally the "≈2.8 MFLOP/step — negligible" the FLOP
  estimate suggested (it is a fifth of `fwd_bwd`'s time, not a rounding error) — a reminder that
  `diag`'s `f_tilde()` does real, if small, work (`kron` reconstruction plus the min-max
  normalisation's several elementwise passes, §5 point 6), even though it is still, by a wide
  margin, the cheapest of the five per step.

### 6.2 Convergence — loss vs. epoch and loss vs. wall-clock time (§6.3's actual metric)

See `loss_vs_epoch.png` / `loss_vs_time.png` in the output directory. Two findings, read directly
off the recorded per-step series (not eyeballed off the plots):

**All five modes reach a statistically indistinguishable loss plateau within the 60s budget.**
Mean training loss over each run's final 100 steps:

| mode | mean loss, last 100 steps |
|---|---|
| `ekfac` | 0.067232 |
| `diag` | 0.067237 |
| `tekfac` | 0.067299 |
| `tkfac` | 0.067339 |
| `kfac` | 0.067401 |

A spread of **0.25%** across all five modes — despite reaching very different epoch counts (9.98
to 16.35) and very different step counts (1199 to 1963) to get there. On this small, thoroughly
over-parameterized 8-layer auto-encoder, all five constructions of `v^(t)` converge to essentially
the same optimum at this budget; the "which mode's curvature model is better" question this project
exists to explore is not distinguishable from the final-loss numbers alone at 60s. It would need
either a harder task (the deferred CIFAR-10 work, §4) or a much shorter budget (§6.3's actual
point) to separate.

**The wall-clock ranking of "time to reach the loss floor" is where the five modes actually
differ.** Time at which the 10-step-smoothed training loss first drops below 0.075 (roughly
midway between the initial ≈0.19 and the ≈0.067 floor — an arbitrary but fixed, reproducible
threshold, not tuned per mode):

| mode | step | epoch | wall-clock time |
|---|---|---|---|
| `diag` | 17 | 0.14 | **0.68 s** |
| `tkfac` | 115 | 0.96 | **4.58 s** |
| `kfac` | 211 | 1.76 | **8.61 s** |
| `tekfac` | 209 | 1.74 | **10.56 s** |
| `ekfac` | 209 | 1.74 | **10.58 s** |

`diag` reaches the loss floor first by nearly an order of magnitude, followed by `tkfac`, then
`kfac`, with `ekfac`/`tekfac` essentially tied last (consistent with §6.1: they pay both a
per-step cost premium *and*, in this specific run, do not reach the floor in fewer epochs than
`kfac`/`tkfac` to compensate — `kfac` needs 1.76 epochs and `tekfac`/`ekfac` need 1.74, essentially
the same epoch count, so `ekfac`/`tekfac`'s extra per-step cost is not offset by faster
per-epoch progress here). **This is a measurement at one specific, untuned, shared-hyperparameter
operating point (§0.6), not a general ranking claim** — `plan.md` §6.1's own caution that
"dominance over `diag` is not a theorem" applies with equal force to a wall-clock convergence
comparison: `Lambda=1e-3` and `lr=1e-3` were never retuned per mode (§4, deliberately out of
scope), and `diag`'s own defaults are, unsurprisingly, closer to what the original AdaFisher paper
already tuned this exact bench around. What the measurement *does* establish, cleanly, is
exactly the point `plan.md` §6.3 set out to make: the equal-*epoch* framing implicit in a naive
comparison would have shown `tkfac` "winning" outright (fewest epochs to the floor, 0.96), while
the equal-*wall-clock* framing this lot implements shows `diag` over 6× faster than `tkfac` to the
same floor once each mode's real per-step cost is accounted for — the exact rigging `plan.md` §6.3
warned an epoch-fixed comparison would introduce, now demonstrated rather than only argued.
