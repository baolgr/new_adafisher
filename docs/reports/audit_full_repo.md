# Full-repository debugging audit

*Written to be read by someone who has not followed the audit. Plain words, short sentences, every
term explained the first time it appears. Every number says whether it was measured here, measured
by one of the three audit agents, or worked out on paper.*

Date: 2026-09-19. Scope: every Python file in `src/adafisher_modes/`, `fisher_ref/` and
`benchmarks/` (148 files), plus all 24 test files. Method: three independent agents, one per
package, each required to confirm or refute every suspicion **numerically** against an oracle that
shares no code with the thing it checks. The orchestrator then re-verified the highest-impact
findings independently and checked the seams between the three packages.

---

## Short version

**The numerical core is sound.** The thing that would have been worst — a wrong formula quietly
producing wrong curvature — is not there. The exact Fisher reference, the per-example gradient
capture on all seven layer kinds, the row-major Kronecker convention in all five modes across all
four layer types, K-FAC's `pi`, TKFAC's trace identity, EKFAC's optimality, the fold arithmetic,
all five metrics, and the damping plumbing each agree with an independent brute-force oracle to
between 0.0 and 3.5e-15. One full optimizer step was rebuilt by hand from scratch and matched to
2.7e-16. **No headline conclusion of this project is overturned by a numerical bug.**

**Nine findings change how an existing number must be read, or are loaded guns.** Ranked by what
they touch:

| # | What | Status | What it touches |
|---|---|---|---|
| 1 | The inter-layer coupling metric returns the **square root** of the alignment its prose calls a CKA | measured here | Every reported coupling number |
| 2 | The wall-clock budget charges a per-epoch cost, so cheap arms get less optimization time | measured here | The WCT ranking on the four cheap models |
| 3 | `ekfac`/`tekfac` estimate their rescaling in the eigenbasis that is about to be replaced | measured here | P2's `ekfac`/`tekfac` numbers; bites if `λ` is lowered |
| 4 | A normalisation layer's input factor is built from the **raw** input, not the normalised one | measured here | Every BatchNorm/LayerNorm block in the four Kronecker modes |
| 5 | The `lr` column of `epochs.csv` is one epoch ahead | measured here | Any statement quoting an epoch index from that column |
| 6 | The multi-seed axis mixes three different learning-rate protocols | measured here | The `{diag, adamw} × seeds` checkpoint axis |
| 7 | Seed coverage is uneven: 3 of 5 models have 5 seeds, 2 have 3 | measured here | Any per-model error bar |
| 8 | The Monte-Carlo curvature source is unpaired between its two passes | agent-measured | Nothing yet — no `mc` row exists anywhere |
| 9 | The `peak VRAM` column is cumulative across arms, not per-arm | agent-measured | That column only |

**Nothing was changed except documentation.** All 148 files are AST-identical once docstrings are
stripped, the test suite is 674 passed / 41 skipped before and after, and the set of 715 test
node IDs is unchanged.

---

## The findings, one by one

### 1. The coupling metric is a square root

`fisher_ref/metrics/coupling.py` returns

```
C[l, l'] = ||R[l, l']||_F / sqrt(||R[l, l]||_F * ||R[l', l']||_F)
```

Write `R = U^T U / N`. Layer `l`'s tangent kernel is `K_l = U_l U_l^T / N`. Then
`<K_l, K_l'>_F = ||R[l, l']||_F^2` and `||K_l||_F = ||R[l, l]||_F`, so the uncentred kernel
alignment is `||R[l,l']||_F^2 / (||R[l,l]||_F ||R[l',l']||_F)`. **The code returns its square
root.** Measured here on a random 40x12 example: code 0.30124755, alignment 0.09075009, and
0.30124755 = sqrt(0.09075009) to ten digits.

The two agree at 0 and at 1 and differ everywhere in between, always upward. So every reported
coupling number overstates the coupling when read as a CKA:

| as reported | read as an alignment |
|---|---|
| "inter-layer coupling >= 0.5 everywhere" | 0.25 |
| "fails on the ViT (min 0.28)" | 0.078 |
| "A2-GN 0.44" | 0.194 |

**What this does and does not change.** The Q4.5 verdict is computed by comparing the measured
value against a 0.5 threshold on the *same* scale, so the verdict itself is unaffected. What is
affected is every sentence that calls the number a CKA or an alignment. The honest reading is
stronger than the reported one, not weaker: the layers are **less** entangled than the prose says.
The claim that already failed on the ViT fails harder.

Nothing in the test suite constrains this value — the only coupling test checks the diagonal,
symmetry and sign. That is why it went unnoticed.

### 2. The wall-clock budget is biased toward the expensive arms

Every arm of a model gets the same wall-clock budget. But part of that budget is spent outside the
optimizer — fetching batches, moving them to the device, and restarting the dataloader workers at
the top of each epoch. That cost is **per epoch**, and the arms deliberately complete different
epoch counts. The cheap arms complete more epochs, so they pay the tax more times.

Measured here on `mnist_autoencoder`, the project's primary bench, from the shipped
`records.csv`. All seven arms had the same 14.59 s budget:

| arm | steps | epochs | compute (s) | wall (s) | % of budget that is compute | compute vs `adam` |
|---|---|---|---|---|---|---|
| `adam` | 2073 | 20 | 6.92 | 14.59 | **47.4%** | 1.00x |
| `adamw` | 2070 | 20 | 6.89 | 14.58 | 47.3% | 1.00x |
| `diag` | 2200 | 21 | 8.66 | 14.59 | 59.4% | 1.25x |
| `kfac` | 1730 | 17 | 9.86 | 14.59 | 67.6% | 1.43x |
| `tkfac` | 1781 | 17 | 10.07 | 14.59 | 69.0% | 1.46x |
| `ekfac` | 1301 | 13 | 11.53 | 14.70 | **78.5%** | 1.67x |
| `tekfac` | 1321 | 13 | 11.36 | 14.59 | 77.9% | 1.64x |

So `ekfac` receives 1.67 times the actual optimization time `adam` does, inside a budget that is
supposed to be equal. The bias is monotone in how expensive the arm is, and it runs **against**
Adam.

It is negligible where the model is large: `resnet50_cifar` is 99.5-99.6% compute for all seven
arms. It matters on the four cheap models — `mnist_autoencoder`, `mlp_ln_mnist`, `cnn_gn_cifar`,
`vit_micro_cifar`.

**What this does and does not change.** It does **not** overturn the `mnist_autoencoder` stall
finding. Adam reaches MSE 0.4836 against the Fisher modes' 0.7085 *while receiving less
optimization time than every one of them*. Correcting the bias would widen that gap, not close it.
What it does change is the opposite kind of claim: any statement that a Fisher mode is competitive
with Adam under an equal wall-clock budget, on one of the four cheap models, is resting on a
budget that was not equal in the way that matters.

### 3. `ekfac` and `tekfac` measure their rescaling in a basis that is about to be thrown away

EKFAC's whole point is a rescaling that is optimal **in a fixed basis**. EKFAC's Lemma 1 says the
diagonal `D_ii = E[(Q^T grad)_i^2]` is optimal for the `Q` it was measured in, and for no other.

In this code the order inside one iteration is:

```
backward hook:  project the gradient into the CURRENT basis, fold it into s*
step():         refresh() -- replace the basis with a new one, leaving s* untouched
step():         precondition() -- divide using the NEW basis and the OLD s*
```

TEKFAC's own Algorithm 1 orders it the other way: eigenbasis first, rescaling second. The
reference implementation `reference_repos/EKFAC-pytorch/ekfac.py` does the same.

Measured here, on a real `AdaFisherMulti` over 700 steps: the output-factor basis moves a lot
between consecutive refreshes. The "column gap" below is `1 - mean|<q_old, q_new>|`, where 0 means
unchanged and 1 means completely different:

| step | `Q_A` | `Q_B` |
|---|---|---|
| 200 | 0.0000 | 0.1573 |
| 300 | 0.0000 | 0.6293 |
| 500 | 0.0000 | 0.6793 |
| 700 | 0.0000 | 0.5174 |

(`Q_A` is flat only because the audit fixture feeds a fixed input; in real training both move.)
Large movement is expected, not itself a bug — the running average is 92% one minibatch, so the
factor is essentially resampled every `TCov` steps. That is exactly what makes the ordering matter.

The audit agent measured the consequence: the stored `s*` differs from what the paper's ordering
would hold by 10.7% in relative Frobenius norm by step 700.

**What this does and does not change.** At the shipped `λ = 1e-3` the applied step is unchanged to
four decimal places, because `λ` dominates `s*` — the same fact `plan_lambda_dominance.md`
reports. **So no trained model in this repository is invalidated by this.** Two things are:

* `fisher_ref`'s P2 reader consumes `_s_star` and `_Theta` directly. The P2 numbers for
  `ekfac`/`tekfac` describe a statistic measured in a basis that no longer exists. The P2 report's
  own conclusion is that there **is** no readable ranking between the five modes at the
  operational point (the spread is 100-1000x below the noise floor), so this does not change that
  verdict — it reinforces why no such ranking should be read.
* The moment `λ` is lowered, the error reaches the step. That is precisely what fix S1 of
  `plan_lambda_dominance.md` proposes. Re-running at `λ = 1e-8` gives an applied direction that
  differs by 0.32 (`ekfac`) and 0.46 (`tekfac`) in relative Frobenius norm. **Fix the ordering
  before acting on S1.**

No test would fail if the two orderings were swapped: every correctness fixture calls `refresh`
first and then re-drives the hooks, which is the paper's order, not the shipped one.

### 4. A normalisation layer's input factor uses the wrong activation

A normalisation layer computes `y = γ * x̂ + β`, where `x̂` is the normalised input. So the
derivative with respect to `γ` pairs the output gradient with `x̂`, not with the layer's input `x`.
The proof of Proposition A.1 in the AdaFisher paper is explicit that the activations in question
are the normalised ones.

The code takes the input from a forward hook, which sees `x` **before** normalisation, and pools
that. `fisher_ref/capture.py` recomputes `x̂` for exactly this reason — so the two halves of this
repository already disagree about which object this is.

Measured here, on `BatchNorm2d(8)` with a realistic post-ReLU input and running statistics far from
the batch statistics:

* The factor is **bit-identical in train mode and eval mode**. That is only possible if it never
  looks at the normalisation at all, since `x̂` uses batch statistics in train and running
  statistics in eval.
* The scale-shift coupling entry `A[0,1]` reads **1.997**. Built from `x̂` it would be ~0, because
  the pooled `x̂` is zero to 1e-16 by construction.
* The agent measured the scalar `a_ν` as **10.4x too large** on that input, and the reconstructed
  `d/dγ` as 0.906 relative error against 4e-15 for the `x̂` version.

**What this touches.** The `BatchNorm2d` and `LayerNorm` blocks of `kfac`, `ekfac`, `tkfac` and
`tekfac` on every network that has them — `cnn_gn_cifar_bn`, `resnet20_cifar`, `resnet50_cifar`,
every ViT and every CCT. For `diag` the same input is inherited from both reference
implementations, so there it is a property of published AdaFisher; for the four new modes it is a
choice this repository made, and it is **not** in `CLAUDE.md`'s deviation table.

**Do not "fix" this by swapping in `x̂`.** For `LayerNorm`, `x̂` sums to zero across channels by
construction, so the scalar would be exactly 0 and the whole scale block would collapse to the
damping term. The scalar surrogate is a poor stand-in for the exact factor under either input.
This is a design question, not a one-line repair.

### 5. The `lr` column is one epoch ahead

`benchmarks/common/loop.py` steps the scheduler and *then* records the learning rate, so the value
written for epoch `k` is the one epoch `k+1` will use.

Measured here on `benchmarks/outputs/cifar10/cnn_gn_cifar/epochs.csv`, base 1e-3, `T_max = 30`:
epoch 0 records 0.0009972609, which is exactly the cosine value for epoch 1. The rate actually used
during epoch 0 was 1e-3, and it appears nowhere in the file. This holds for every epoch checked.

**What this touches.** Any statement quoting an epoch index from that column. In particular the
campaign-1 audit's "`resnet20_cifar/adam` hit `lr = 0` at epoch 49": the rate was set to zero
*after* epoch 48 finished. The direction of that finding — a periodic schedule climbing back up —
is real and unchanged; only the epoch indices are off by one.

Two further inconsistencies in the same column: under `--lr-schedule budget` no scheduler is used,
so the column holds the last batch's rate; and a budget-truncated final epoch never steps the
scheduler, so its row means a third thing.

### 6. The multi-seed axis mixes three learning-rate protocols

The generated seeds job deliberately omits `--lr-schedule`, with a comment saying seed 0 ran under
the default `nominal` schedule and these seeds must match it. Measured here from the manifests
actually on disk, that premise is false:

| model | seed 0 | seeds 1-4 |
|---|---|---|
| `mlp_ln_mnist` | **`budget`** | `nominal` |
| `cnn_gn_cifar` | **absent** (pre-clamp, periodic) | `nominal` |
| `vit_micro_cifar` | absent | `nominal` |
| `cct_2_3x2_cifar` | absent | `nominal` |
| `resnet20_cifar` | absent | `nominal` |

A missing value is not `nominal`: those runs pre-date `schedules.py` and used torch's **periodic**
cosine, whose rate climbs back up after `T_max`. For the four "absent" models the damage is bounded
— all five checkpoint fractions fire at or before `T_max`, where periodic and clamped agree, so the
*checkpoints* stay comparable and only the final model and the reported best accuracy differ. For
`mlp_ln_mnist` the rate differs at every step, so every checkpoint but `ckpt_0` sits on a different
protocol from the other seeds'.

This touches the `adamw` arm of the `{diag, adamw} × seeds` axis. The `diag` arm is the unbudgeted
reference and always uses the clamped schedule, so it is unaffected.

### 7. Seed coverage is uneven

`available_seeds()` returns `[0, 1, 2, 3, 4]`. That is the **union across all models**, and it is
true. Per model, measured here through the real bridge:

| model | seeds present |
|---|---|
| `cnn_gn_cifar`, `mlp_ln_mnist`, `vit_micro_cifar` | 0, 1, 2, 3, 4 |
| `cct_2_3x2_cifar`, `resnet20_cifar` | **0, 1, 2** |
| `cnn_gn_cifar_bn`, `mnist_autoencoder` | 0 |

The function does take a `model=` argument, so a careful consumer gets the right answer. This is a
reporting hazard rather than a code bug: any per-model error bar on `cct_2_3x2_cifar` or
`resnet20_cifar` rests on three seeds, not five.

### 8. The Monte-Carlo curvature source is unpaired (loaded gun, nothing hit)

The two probe passes draw their Monte-Carlo labels from the global RNG without a shared generator,
so with `source="mc"` the reference and the approximation see **different sampled labels**. The
agent measured EKFAC's trace identity `sum(s) = tr(B)` breaking by 5.5e-2 relative under `mc`
against exactly 0.0 under the type-2 control. The Monte-Carlo sample count also cannot be set from
either runner — it is stuck at 1.

**Nothing reported is contaminated.** Across all 45 `metrics.csv` files the `source` column holds
only `type2` (128 437 rows) and `empirical` (126 557 rows). There is no `mc` row anywhere. Both
sources used are deterministic given the probes, so they are paired by construction. This is a trap
for the first run that passes `--sources mc`.

### 9. The `peak VRAM` column is cumulative

Each `AdaFisherMulti` arm leaves its model and optimizer alive past the end of its run, because the
hooks form a reference cycle that plain reference counting cannot free. The next arm's memory
baseline therefore includes its predecessors. The agent measured 5 of 7 models still resident after
their runs returned, and the symptom is visible in the shipped data: on `mnist_autoencoder` the
reported peaks are strictly monotone in the order the arms ran, and `adam` is reported as needing
*more* memory than every Fisher mode, which is impossible. Four independent runs show the same
ladder. **No training number is affected** — only that column, and only where the leaked state is
comparable to the activations.

---

## What the test suite does not check

The suite is 674 passed, 41 skipped. The 41 are 40 `--runslow` gates and one missing optional
package; with `--runslow` everything passes. The gaps below are about what the passing tests
constrain, not about failures.

**The largest gap is in `benchmarks/`.** `loop.evaluate` and `loop.top1` are executed **zero
times** by the entire suite, and `build_loaders`, `mnist()`, `cifar10()` and `cifar100()` are never
called by any test. Every runner-driven test supplies its own stub or passes `None`. So nothing
would notice if evaluation stopped using `model.eval()`, started updating BatchNorm statistics, or
mis-weighted a ragged final batch. The agent checked all of this by hand and **the code is
correct** — on a ragged split the weighted mean is exact to eight digits and the module observes
`training=False, grad_enabled=False` on every batch. The behaviour is right; the suite simply does
not watch it.

**Assertions that are weaker than their names.** `test_all_modes_run` is 100 of the 194 items in
its file and asserts only that parameters are finite — a mode that silently updated *nothing* would
pass. The "no parameter left un-updated" check exists, but runs with `fisher_mode="diag"` only, so
the four Kronecker modes are never checked for it on any model. The wall-clock overshoot bound is
computed against the maximum over *all* steps, including the first slow one, which in several
measured cases makes it impossible to fail.

**Two tests that cover a branch the benchmarks do not use.** The documented `Conv2d` scale quirk —
that the full factor's diagonal is `P` times the diagonal one's — is asserted only with `bias=True`.
Measured here:

| kernel/stride/pad | bias=True | bias=False | `P` | `1/S` |
|---|---|---|---|---|
| k=3, s=1, p=1 | 18.0000 | 0.0123 | 18 | 0.0123 |
| k=2, s=1, p=0 | 8.0000 | 0.0156 | 8 | 0.0156 |
| k=3, s=2, p=0 | 18.0000 | 0.0625 | 18 | 0.0625 |

Without a bias the ratio is exactly `1/S`, not `P`. **Every convolution in every ResNet and CCT in
this repository is `bias=False`**, so the branch that actually runs on the CNN benchmarks is the
untested one. The port is faithful to the reference in both branches, so this is a documentation
and coverage gap, not a porting bug.

**Why finding 1 went unnoticed**: nothing constrains the coupling formula's value. **Why finding 3
went unnoticed**: every `ekfac`/`tekfac` correctness fixture uses the paper's ordering rather than
the shipped one. **Why finding 4 went unnoticed**: the BatchNorm oracle builds its statistic from
the raw pooled input too, so both sides share the assumption.

A handful of tolerances are loose enough that they cannot fail (a decoupled-weight-decay identity
with 140x slack, a Cutout area asserted only as an upper bound, a learning-rate floor checked at
2e-5 where the measured value is 1e-11). And the noise floor's "95% interval" is, at the default 20
partitions, exactly the observed minimum and maximum — the indices only move inside the sample at
80 partitions and above.

---

## What is verified correct

This bounds the audit. Each of these was checked against an oracle sharing no code with it.

* **The exact Fisher reference**, on a network covering strided `Conv2d`, `BatchNorm2d` in eval,
  `GroupNorm`, `Linear` shared and unshared, `LayerNorm` over tokens and a raw `pos_embed`:
  3.29e-16 (type-2), 4.10e-16 (empirical). End-to-end on a real trained checkpoint: 3.47e-15.
* **The per-example gradient capture** survives in-place activations, residual reuse of a captured
  output, and correctly refuses a module called twice. Worst error 1.7e-16.
* **The row-major Kronecker convention**, re-derived from `vec_r(uv^T) = u ⊗ v` and checked by
  rebuilding the dense preconditioner by hand for 38 combinations of mode, layer type, bias and
  SUA: worst relative error 4.5e-13.
* **One complete optimizer step**, rebuilt by hand from the augmented input factor through the
  running average, `pi`, both damped factors, the Kronecker solve and Adam's bias correction:
  2.7e-16.
* **K-FAC's `pi`** matches the paper's §6.3 including the bias-augmented dimension; **TKFAC's**
  trace identity holds to twelve digits at every step under the real coefficients; **TEKFAC's**
  equations match with no spurious factor; **EKFAC's** `sum(s) = tr(B)` holds to 7.9e-16.
* **The damping is applied exactly once**, on the right object, in every mode — checked explicitly
  for double damping.
* **The λ-dominance arithmetic**, verified here directly: driving the shipped running average to
  its fixed point with a constant observation of 1.0 gives 0.0086956522, which is 1/115.0000
  exactly; squared for the two stored factors that is 13 225; times 16 384 at batch 128 that is
  2.167e8. The report's "115", "13 225" and "about 200 million" are all correct.
* **All 20 model parameter counts** match `CLAUDE.md` exactly, three of them re-derived on paper.
* **The folder reorganisation** broke exactly one thing: a helper script that still reads the old
  `benchmarks/<model>/` path, and it fails loudly rather than silently.

---

## What this means for the experiments already run

**Unchanged and still supported:**

* The λ-dominance result. Its arithmetic is verified here, and findings 3 and 4 are *masked* by it,
  which is consistent rather than contradictory.
* The `mnist_autoencoder` stall. Finding 2 makes the comparison *less* favourable to Adam than it
  should be, and Adam still wins by a wide margin. The finding is strengthened.
* HF2 (K-FAC-reduce beats K-FAC-expand). It votes on a type-2 Frobenius error with no inverse and
  no coupling, so none of the nine findings touches it.
* The P2 conclusion that the applied preconditioner is numerically the identity. Finding 3 affects
  the *ranking between modes*, and the P2 report's own verdict is that no such ranking is readable.

**Must be re-read, not re-run:**

* Every coupling number, as a square root (finding 1).
* Every epoch index taken from the `lr` column (finding 5).
* Every `peak VRAM` figure (finding 9).
* Any per-model error bar on `cct_2_3x2_cifar` or `resnet20_cifar` — three seeds, not five
  (finding 7).

**Should be treated as provisional:**

* Any claim that a Fisher mode is competitive with Adam under an equal wall-clock budget on
  `mnist_autoencoder`, `mlp_ln_mnist`, `cnn_gn_cifar` or `vit_micro_cifar` (finding 2). The budget
  was not equal in the way that matters. The large models are unaffected.
* Any per-layer-type reading that involves a `BatchNorm2d` or `LayerNorm` block in one of the four
  Kronecker modes (finding 4). Those blocks describe a statistic paired with the wrong activation.
* The `adamw` arm of the multi-seed axis (finding 6).

**Must be fixed before the next step, not after:**

* Finding 3, if `plan_lambda_dominance.md`'s fix S1 is acted on. Lowering `λ` is exactly what makes
  the stale-basis error reach the applied step — 0.32 and 0.46 relative at `λ = 1e-8`. Fixing the
  ordering is not a one-liner: the hooks fire before `step()`, so the refresh has to move earlier
  in the iteration, and that changes the meaning of the cadence argument.
* Finding 8, before any run that passes `--sources mc`.

---

## Documentation

Every module-level docstring in all three packages was rewritten to be self-contained: what the
module computes, its public API, and its dependencies on other modules in this repository. A
machine check confirms **zero** module docstrings still cite an internal report. Every paper
citation was kept. The 32 inline comments that warn a future reader against "fixing" a deliberate
deviation were kept on purpose, with their justification now stated in the file rather than pointed
at. Three docstring claims that were false were corrected to the measured truth: the coupling
formula, the Monte-Carlo pairing, and the noise-floor interval.

**Nothing but documentation changed.** Proof: parsing all 148 files before and after, deleting every
docstring node, and comparing the abstract syntax trees gives 148 identical and 0 changed. The
checker was negative-controlled — it detects a `gammas[0]` -> `gammas[1]` index swap, which is
precisely the class of silent change at issue. The suite is 674 passed / 41 skipped before and
after, with the same 715 test node IDs, and `ruff` and `mypy` report the same pre-existing
diagnostics with only line numbers moved.

---

## Operational note

`.venv/bin/pytest`, `.venv/bin/ruff` and `.venv/bin/mypy` do not run: their shebang points at a
path with a trailing space that the checkout no longer has. The commands in `CLAUDE.md`'s "Running
the tests" section therefore fail as written. `.venv/bin/python -m pytest` works and is what this
audit used throughout.

---

# Part 2 — Remediation, and the final verification

*Added after the audit above. Same rules: every number here was measured, and it says by whom.*

## The rule the fixes followed

Two classes, kept strictly apart.

**Behaviour-preserving fixes** were applied directly. Such a fix must not change any number any
existing run would have produced, and each one had to prove it rather than argue it.

**Behaviour-changing fixes** were never applied directly. They ship as a keyword argument
defaulting to **exactly today's behaviour**, so every past experiment stays reproducible bit for
bit. This is the repository's own established pattern — `ema_seed_first`, `minmax_after_average`,
`gamma`, `decoupled_weight_decay` all exist for the same reason.

That rule is why the remediation does not invalidate a single result already on disk.

## What was fixed

**One behaviour-changing fix, shipped as an off-by-default knob.**

`eig_before_rescale` (finding 3, the stale eigenbasis). With it on, `ekfac`/`tekfac` rebuild the
eigenbasis inside the backward hook, before the gradient is projected into it, so the rescaling is
measured in the basis `precondition` then uses — the order TEKFAC's Algorithm 1 and
`EKFAC-pytorch` both use. Verified here directly: with the knob **off** the trajectory is
bit-identical to not passing the argument at all (`torch.equal` on every parameter), with it **on**
parameters move by 2.3e-1, and the three modes that have no eigenbasis raise `ValueError` rather
than silently ignoring it. Measured cost of the old order: the stored rescaling differs by
**91–141 %**, but the applied step differs by only **3.0e-3** at the shipped `λ = 1e-3` against
**8.3** at `λ = 1e-8`. So no trained model here is affected — and the ordering must be fixed
*before* acting on `plan_lambda_dominance.md`'s fix S1, which is precisely what lowers `λ`.

**Fifteen behaviour-preserving fixes.**

In `src/`: the multi-parameter-group double step (a module was stepped once per group, moving
parameters exactly **2.0000×** too far under the standard decay/no-decay split — no model here uses
more than one group, which is why it is behaviour-preserving); and six configurations that used to
fail with a confusing error now fail with a clear one naming the module — a non-zero `padding_mode`
(which silently returned a factor **48.3 %** wrong), a string `padding`, `LayerNorm(bias=False)`, a
frozen bias, a module first reached after step 0, and a second backward over one forward.

In `fisher_ref/`: the Monte-Carlo source is now paired between its two probe passes. Verified here
— two passes seeded from one number draw identically, while a *shared* generator object would not,
because it advances as it is used. Unpaired, a real run wrote **2** `invariant_violation` rows
claiming EKFAC's dominance theorem had failed; paired, **0**. Also: `k` is now settable, the
per-fraction layer inventory is no longer overwritten, the reference metadata records its real
column count instead of 0, two experiment drivers now run under `python -m`, and two
falsy-zero bugs in the pre-registered decision rules are gone.

In `benchmarks/`: the `lr` column now records the rate the epoch actually ran at — verified here end
to end, epoch 0 now reads `1e-3` where it used to read the epoch-1 value; each arm's model and
optimizer are released at the end of its run, so `peak VRAM` is per-arm rather than cumulative; the
stale dispatch path is fixed; and the seeds job now states its schedule explicitly instead of
relying on a premise that was measurably false.

**And the environment**: every text console script in `.venv/bin/` embedded a stale interpreter path
with a stray space, so `.venv/bin/pytest` died with "cannot execute" and **every command in this
file's own "Running the tests" section failed as written**. 22 scripts repaired; the documented
commands work again.

## What was deliberately not fixed

**The wall-clock budget itself** (finding 2). Changing what the budget charges would alter the
protocol and make new runs incomparable with the entire existing campaign. The bias is now
*measurable* instead: a per-step `data_s` column and a per-arm `compute share` in the summary.
Verified here on a real run — `data_s + fwd_bwd_s + step_s` accounts for **99.9–100.0 %** of the
clock, so the gap the budget charges is now visible in every future run rather than having to be
reconstructed.

**The coupling formula** (finding 1). It is a legitimate normalised coupling in `[0,1]`, and the
Q4.5 verdict compares it against a 0.5 threshold on the same scale, so changing it would silently
invalidate every reported coupling number. The real gap was that nothing constrained its value —
now closed. Verified here by mutation: against a version with the square root removed, the two new
tests **fail** while the pre-existing test **passes**, which is exactly the blindness that let this
through.

**The normalisation-layer input** (finding 4). Documented in §4.6 with its measured numbers and
pinned by two tests, but deliberately given no knob: for `LayerNorm`, the normalised activation sums
to zero across channels, so the "correct" setting collapses the whole scale block to the damping
term (measured: `a_ν = 6.3e-15`). A knob whose correct setting is degenerate is a trap, and
choosing a replacement is a design decision, not a repair.

**The noise-floor percentile.** Checked at n = 10, 20, 40, 41, 80, 200 and swept to 5000: at the
default 20 partitions the code *is* exactly nearest-rank, so the only defect was the "95 %" label,
which the docstring no longer claims. Away from the default it differs by one index, always
outwards — the safe direction for a threshold something must beat.

**Existing result files are not rewritten.** Everything in Part 1's "must be re-read" list still
applies to the data already on disk. The fixes are for future runs.

## The final verification

| check | before the audit | after the fixes |
|---|---|---|
| full suite, default | 674 passed, 41 skipped | **812 passed, 41 skipped, 0 failed** |
| full suite, `--runslow` | 714 passed, 1 skipped | **852 passed, 1 skipped, 0 failed** (4 min 50 s) |
| collected tests | 715 | **853** |
| skips | 41 (40 `--runslow`, 1 `curvlinops`) | **41, unchanged** |
| `ruff` (src + fisher_ref + benchmarks) | 12 | **11** — same set, one resolved, none new |
| `mypy` | 2 / 9 / clean | **2 / 9 / clean, unchanged** |
| generated SLURM jobs vs. the generator | — | **all match**, including the one intended change |
| `bench.py` 50-line spec | 20/20, 7 at exactly 50 | **20/20, 7 at exactly 50** |
| assertions removed or weakened | — | **zero** |

That last row is the one that matters most, and it was checked mechanically rather than taken on
trust: every test function whose assertions changed was diffed statement by statement. Nine changed.
All nine are tightenings — two parameter counts went from a ±0.4 % range to an exact equality; a
tautological `shape[0] == shape[0]` became a real head-width check; a tolerance whose effective value
was 1.37e-4 became 3e-6; an overshoot bound computed from the maximum over *all* steps (which in
several measured cases could not fail) became the last step's own cycle, with a guard that fails if
the bound ever exceeds 2 % of the budget. The one that looks like a weakening — an exact
annealing-floor check becoming an inequality — is not: the whole cosine series is now asserted
against an independently computed closed form at `rel=1e-12`, which subsumes the two spot checks it
replaced. The value changed because the `lr` fix changed what the column means.

**What the new tests actually buy.** The largest gap Part 1 found was that `loop.evaluate` and
`loop.top1` were executed zero times by the whole suite, and that `mnist()`, `cifar10()` and
`cifar100()` were never called. Both are now covered, and the coverage is real: each new test was
proven to fail against a deliberately broken copy of the code, and the induced failure messages are
recorded. `test_all_modes_run` — 100 of one file's items — asserted only that parameters were
finite, so a mode that silently updated nothing would have passed; it now asserts that no parameter
is left unchanged, for all five modes on all 20 models, which also closes the separate gap that the
"no parameter left un-updated" check had only ever run on `diag`.

## What this changes about the experiments

**Nothing in Part 1's reading of the existing results changes.** No fix altered what any past run
did. The findings that must be re-read — the coupling numbers as square roots, the epoch indices
from the `lr` column, the `peak VRAM` figures, the uneven seed coverage, the wall-clock bias on the
four cheap models, the normalisation-layer blocks — are re-read exactly as Part 1 describes.

**What changes is the next run.** A future run records an honest `lr`, a per-arm `peak VRAM`, and a
`compute share` that makes the wall-clock bias visible without reconstruction. A future
Monte-Carlo run is paired, so a sampling difference can no longer masquerade as a violated theorem.
And the ordering defect in `ekfac`/`tekfac` is now both fixable by a flag and pinned by a test that
fires if the two orderings are ever swapped — which matters specifically because the one change this
project is most likely to make next, lowering `λ`, is the change that would have let that defect
reach the applied step for the first time.
