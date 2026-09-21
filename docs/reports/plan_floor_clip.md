# A floor or a clip instead of an added safety constant (E16)

*Implementation plan and experiment specification. The code is done, tested and audited; the thirty
cluster jobs are written and **not yet submitted**. The decision rules are pre-registered in
[`plan_lambda_dominance.md`](plan_lambda_dominance.md), section "E16 — pre-registered", and are
repeated in §5 below so this document can be read alone.*

*Amended three times on 2026-09-21, every time before any submission.*
- *First (§9): family B went from one arm to three, because the single clip was a per-layer
  normalisation more than a clip.*
- *Second (§10): a six-agent audit found a design error (a threshold shared by the whole network),
  an estimator bias, an off-by-one, a cost that would have overrun the jobs, and an analysis script
  that could read incomplete data as a verdict. All fixed.*
- *Third (§11): the normalisation layers' curvature statistic, which the audit had recorded as a
  limit, is now corrected in every arm. E16 therefore reruns its own add baseline instead of reusing
  E14's cells. The jobs are split by mode and run three cells at a time on a larger GPU slice.*

---

## Short version

**The question.** The eigenbasis modes (`ekfac`, `tekfac`) divide each coordinate of the step by a
stored curvature value `s`, after adding a safety constant `λ` to it: `s + λ`. E14 found that
lowering `λ` to about 1e-11–1e-10 gains **+6.6 to +10 accuracy points** on three of four networks,
but that the best `λ` does not follow the curvature from one network to the next, so it has to be
tuned per network. The French feasibility study
[`fr/etude_clipping_vs_damping.md`](fr/etude_clipping_vs_damping.md) listed two other ways to protect
the division:

- **Family A, a floor:** divide by `max(s, λ)` instead of `s + λ`. Curvature values above `λ` are
  used as they are.
- **Family B, a clip (Sophia-type):** divide by `s` alone, then cap every coordinate of the result
  at a threshold. Three ways to set the threshold are built and tested, because they answer
  different questions (§2).

**What is built.**
- One option on the optimizer, `rescale_form = "add" | "floor" | "clip"`, for `ekfac` and `tekfac`
  only. For `"clip"`, a threshold rule `clip_threshold = "quantile" | "ema" | "fixed"`, always set
  per layer.
- A second option, `norm_exact_rescaling`. It estimates a normalisation layer's curvature statistic
  from the layer's true gradient instead of a surrogate that under-states it 80–989× (measured, §11).
- Both are off by default, and the default is bit-identical to the code before them.
- The algorithm is written out in LaTeX in [`fr/clipping_algorithme.tex`](fr/clipping_algorithme.tex).

**What will be run.** On three networks that respond to `λ` (`cnn_gn_cifar`, `vit_micro_cifar`,
`cct_2_3x2_cifar`), for `ekfac` and `tekfac`, five seeds, under E10/E13/E14's protocol:
- the E14 fix (`add`), rerun inside E16;
- the floor;
- the three clips.

Every arm uses the corrected statistic, so every comparison pairs by seed within one job, on the
same code and the same hardware.

**What it can decide.**
- Whether any of the four beats the tuned `λ`.
- The question family B exists for: whether one setting works on every network, where no single `λ`
  does.
- Two things inside family B: whether normalising the step at every step helps or hurts, and whether
  a frozen threshold does better than one that follows the momentum.

---

## 1. What was known before this plan

**The damping dominates at the shipped `λ`.** `λ = 1e-3` is above every stored curvature value, in
every layer, of every network measured (E1). The division then does the same thing in every
direction, and the five modes behave as momentum SGD (E0, lot 5 of the drift campaign).

**Lowering `λ` works, but the right value has to be found per network.** With the step-size cap held
fixed (E2's protocol), `ekfac`/`tekfac` peak at `λ` between 1e-11 and 1e-10:

| network | best `λ` (5 seeds) | gain over the shipped `λ` | source |
|---|---|---|---|
| `cct_2_3x2_cifar` | 3e-11 | +9.9 / +10.0 | E10 |
| `vit_micro_cifar` | 1e-10 | +9.0 / +8.8 | E13 |
| `cnn_gn_cifar` | 3e-11 / 1e-10 | +6.6 / +6.9 | E14 |
| `resnet20_cifar` | flat | +0.6 at best | E14 |

E14 then found that making `λ` proportional to the network's curvature (fix S1) does not predict
these values better than a fixed number does. The plateaus of these networks do not share a single
`λ` either: E13's plateau for `ekfac` on `vit_micro_cifar` is 1e-10 alone, E10's on
`cct_2_3x2_cifar` is 3e-11 alone.

**The feasibility study on clipping** (`fr/etude_clipping_vs_damping.md`, one network, one
checkpoint, no training run) concluded two things that still stand:

1. At the shipped `λ`, a clip set to reproduce today's step size clips **95–100 %** of the
   coordinates. The optimizer would become sign descent instead of momentum SGD.
2. A floor `max(s, λ)` is **provably identical** to `s + λ` wherever `λ` dominates every value.

Both conclusions are about the *shipped* operating point. E7–E14 have since found an operating point
where `λ` no longer dominates (it sits inside the list of curvature values). That is where the two
families become worth testing, and it is why this plan exists now.

**A correction to that study.** It says Sophia's authors recommend keeping the *clipped* fraction
between 0.1 and 0.5. Sophia's README
([github.com/Liuhong99/Sophia](https://github.com/Liuhong99/Sophia), read on 2026-09-21) asks for a
`train/win_rate` between 0.1 and 0.5 and states that "a large ρ will lead to a large
train/win_rate". In Sophia's update `clip(m / (ρ·h + ε), 1)`, a larger `ρ` clips *fewer*
coordinates. So `win_rate` is the fraction **not** clipped, and the recommendation is a clipped
fraction of **0.5 to 0.9**. The grid below covers both readings.

---

## 2. The forms, in one place

Write `M` for the step direction projected into the eigenbasis (one number per direction), and `s`
for the stored curvature in the same basis (`s*` in `ekfac`, `Θ` in `tekfac`). Then the optimizer
moves each coordinate by `lr × u`:

| form | `u` | what `λ` does |
|---|---|---|
| `"add"` (default, both papers) | `M / (s + λ)` | shifts **every** curvature value up by `λ` |
| `"floor"` (family A) | `M / max(s, λ)` | replaces only the values **below** `λ` |
| `"clip"` (family B) | active: `sign(M)·min(r/γ, 1)`; inactive: `M / max(γ·s, guard)` | none: `λ` is not used |

### The floor

Where `s ≫ λ`, the floor and `add` both give `M/s`. Where `s ≪ λ`, both give `M/λ`. They differ only
for values near `λ`, by at most a factor of 2 (at `s = λ`: `M/2λ` against `M/λ`). So family A asks
one narrow question: does it help to leave the well-estimated curvature values untouched instead of
shifting them by `λ`?

### The clip, and what all three thresholds share

`r = |M|/s` is the size each coordinate's step would have with no safety constant at all.
`guard = 10⁻³ × rms(M)`. A coordinate is **active** when its momentum is above the guard.
- An active coordinate moves by `u = sign(M)·min(r/γ, 1)`. It is *clipped* (`|u| = 1`, a step of
  exactly `lr`) when `r ≥ γ`. Otherwise it takes the undamped step divided by `γ`.
- An inactive coordinate moves by `u = M / max(γ·s, guard)`, so by at most `|M|/guard < 1`.

For an active coordinate this is `clip(M/(γs), 1) = M/max(γs, |M|)`: a floor on the curvature, but a
floor proportional to the momentum itself (`|M|/γ`), not a constant.

Clipped is *defined* as `r ≥ γ`, with `r` the same tensor `γ` is selected from. So the number of
clipped coordinates is exact by construction (ties aside).

**Why the guard.** Some directions have exactly zero curvature (a classification head's logit-shift
directions, `plan_exp_lot1.md` §6.4), and their momentum is zero up to rounding. Without the guard,
such a coordinate would move by a full `lr` driven by rounding noise.

**Why the fraction counts only active coordinates.** In the eigenbasis many coordinates are zero in
exact arithmetic, because the gradient lies in the span of the batch's inputs and `A` has few
observed directions; in fp32 they are rounding noise. Measured on a 6-example batch: `|M|` down to
1e-13, and the median of `r` set entirely by such coordinates. Counted in, they would put the
threshold inside the noise and clip every real coordinate.

**The guard level is a choice, not a measured gap.** On real layers, genuine coordinates go down to
5e-6 × rms and exact-null rows up to 9.8e-4 × rms (audit, §10). So **every clip verdict is a verdict
at `clip_guard = 1e-3`**. How much of each step is carried by coordinates just above the guard
(below 1e-2 × rms) is logged for every layer.

**Why every threshold is per layer.** The stored curvature's scale error is not the same in every
layer. A layer applied at `T` positions (a convolution, a token-wise Linear) stores `s` averaged over
`N·T` rows (`plan_exp_lot3.md` §5.2's `1/T`). Measured at step 2000: relative to the head, the error
is 1/422 to 1/7 400 on `cnn_gn_cifar`'s convolutions, 1/251 to 1/5 620 on `vit_micro_cifar` and
1/91 to 1/2 724 on `cct_2_3x2_cifar`. One threshold for the whole network is therefore not one
threshold in consistent units (§10, finding 1).

The clip receives the **bias-corrected** momentum `m/(1 − β^t)`, and its result is applied without
dividing again.

### Three ways to set the threshold

**`"quantile"`: reset at every step.** `γ` is the `c`-th largest `r` among the layer's `n_a` active
coordinates, with `c = max(⌊q·n_a⌋, 1)`. Exactly that many are clipped at every step.

- Multiplying `M` or `s` by any positive number leaves `u` unchanged. No scale error of the stored
  curvature can reach the step.
- But neither can the momentum's own size. Each layer's largest coordinate moves by exactly `lr` at
  every step, however small the momentum is.
- So this is **a per-layer normalisation of the step**, with a clip of its shape. It is kept as the
  control that isolates the normalisation.

**`"ema"`: the step follows the momentum again (main arm).** The same quantile, multiplied by how
large the momentum is compared with its own recent history:

```
γ = γ_quantile · μ̄ / μ,   μ = rms(M)
log μ̄ = Adam's bias-corrected running average of log μ:
   num ← (1 − 1/H)·num + (1/H)·log μ,   den ← (1 − 1/H)·den + 1/H,   log μ̄ = num / den,   H = 1000
```

- When the momentum has its usual size (`μ = μ̄`), this is exactly `"quantile"`.
- When it is `a` times smaller than over the last ~1 000 steps, the unclipped coordinates are `a`
  times smaller and fewer are clipped. A spike is clipped harder.
- The curvature's scale is still absorbed at every step. Only the momentum's size is lagged, and that
  is the one thing `"quantile"` throws away.
- **Why the average is on `μ` and not on `γ`.** `γ` also carries the scale of `s`, which jumps at
  every factor update: the stored `s` is 92 % one minibatch. Lagging that would clip for reasons
  unrelated to the momentum.
- **Why it is bias-corrected.** The first bias-corrected momentum is one raw gradient, measured
  4.4–5× larger in rms than the momentum settles to. An average seeded with it stayed more than 10 %
  off for 12–18 % of a run, clipping 0.27–0.36 instead of 0.7 early on (§10, finding 2).
- A step with zero momentum carries no size and is skipped.
- `H = 1000` is the memory of Adam's bias-corrected second moment at `β₂ = 0.999`. It is fixed, not
  swept. Beyond ~1 000 steps the size is still normalised, as in Adam.

**`"fixed"`: a conditional clip.** One `γ` per layer, held constant.

- The clip then acts only where the undamped step exceeds the ceiling. Below it, the step is
  proportional to the momentum, as in Sophia.
- `γ` is calibrated rather than hand-set. Until step 2 000 the rule is `"quantile"`. The layer's
  quantile at each of the 1 000 steps before step 2 000 is kept, and at step 2 000 their (lower)
  median is frozen.
- **Why a median over a window.** `s` is 92 % one minibatch, so one step's quantile is one draw: it
  varied ×3.6 between factor updates on `vit_micro_cifar` (§10, finding 3).
- **Why per layer.** Because of the per-layer scale error above. It also makes the switch continuous:
  the frozen value is what the layer had just been using. Measured on real networks: the step size
  changes by 0.92–1.04× at the switch.
- Step 2 000 is past the `10 × TCov = 1 000` steps the identity seed needs to decay
  (`plan_exp_draft.md` §3.2), and the window starts after them. That is epoch 2 of 15 at batch 32
  (1 406 steps per epoch).
- After calibration the clipped fraction is free to drift, and it is logged, which gives back
  Sophia's own diagnostic.
- Sophia itself uses one `ρ` for the whole network. That is meaningful there because its curvature
  estimate has the correct scale in every layer (arXiv:2305.14342, §2.3); here it does not.

**What each comparison isolates.**
- `"ema"` against `"quantile"`: the same clip with and without the per-step normalisation.
- `"fixed"` against `add`: a threshold proportional to the momentum against a constant floor. This is
  the cleanest statement of family B's own claim.
- `"fixed"` against `"ema"`: a frozen threshold against one that follows the momentum. Both are per
  layer, so this comparison is not mixed with "per layer against network-wide".

---

## 3. The code (done)

| file | change |
|---|---|
| `src/adafisher_modes/approximations/_rescale_utils.py` | **new.** `rescale()` for `"add"`/`"floor"`; `ClipRule` for the three thresholds and their per-layer state; `active_quantile()` (a sort and a gather, so the active count never reaches the host); `check_rescale_args()` refuses bad arguments at construction |
| `src/adafisher_modes/factors.py` | two new functions for `norm_exact_rescaling`: `normalized_norm_input()` recomputes a normalisation layer's normalised activation `x̂` at every row, with the statistics the layer itself used; `norm_exact_kfe_squares()` projects the exact per-row gradient into the eigenbasis |
| `approximations/ekfac.py`, `tekfac.py` | the division `(...) / (s + λ)` goes through `self._rescale(...)`, in both the plain and the SUA branch; `f_tilde` builds `max(s, λ)` under `"floor"` and raises under `"clip"`. Under `norm_exact_rescaling`, a normalisation layer's `s*`/`Θ` is estimated from its exact per-row gradient |
| `approximations/base.py` | a hook `begin_step`, no-op by default; the property `consumes_bias_corrected_momentum`, false by default |
| `optimizer.py` | eight new constructor arguments, all inert by default: `rescale_form`, `clip_threshold`, `clip_fraction`, `clip_guard`, `clip_ema_horizon`, `clip_calibrate_at`, `clip_calibration_window`, `norm_exact_rescaling`. `step()` calls `begin_step` before walking the layers. Under `"clip"` it hands the mode the bias-corrected momentum (new tensors; the buffer is untouched) and applies the result without dividing again |
| `fisher_ref/approx/adafisher_state.py` | the P2 reader rebuilds `s + λ`; it now **refuses** an optimizer running `"floor"` or `"clip"`, instead of returning a silently wrong operator |

**Refused at construction, each with a message naming the reason:**
- a non-default form, or `norm_exact_rescaling`, on `diag`, `kfac` or `tkfac`. The first has no
  curvature scale left after its min-max; the other two invert their factors without diagonalising
  them;
- `norm_exact_rescaling` with `fisher_batch_samples` on a network with a `BatchNorm2d`, whose batch
  statistics cannot be recomputed from part of the batch;
- any `clip_*` argument with a form other than `"clip"`;
- `"clip"` without `0 < clip_fraction < 1`;
- `"fixed"` without `clip_calibrate_at`;
- an argument that belongs to another threshold;
- `"clip"` together with `hold_cap` or a relative damping. Both act through `λ`, which `"clip"` does
  not use.

**Tests.** 120 tests, all offline: `tests/test_rescale_form.py` (90),
`tests/test_norm_exact_rescaling.py` (20) and `tests/test_e16_decisions.py` (10). The full suite
gives **1014 passed, 41 skipped**; no existing test was modified.
- **The defaults** are bit-identical to not passing the options. `"add"` is also bit-identical to
  the formula the two modes spelled out inline before, on a trained state. The audit checked it
  bit-identical, over 16 configurations, to the code at three earlier commits.
- **`"floor"`** is exactly `max(s, λ)`, with `λ` as a float and as a tensor. It agrees with `"add"`
  when `λ` is far above or far below every value. Its `precondition` is the inverse of its own dense
  `f_tilde`: on `TinyMultiLayerNet`, with `λ` at the median of the stored values so both branches
  run, and on every layer type of a narrow CCT in fp64 (bias-free convolutions and Linears, token
  inputs, LayerNorms, a `d_out = 1` Linear, the head).
- **`"quantile"`:**
  - it clips exactly `max(⌊q·n_a⌋, 1)` coordinates, for `q` from 0.01 to 0.99, and on 400 random
    problems in fp32 and in fp64;
  - the sort-and-gather selection equals a sort of the active values;
  - it matches its definition written out independently;
  - the result is unchanged when `M` or `s` is scaled by 1e-9 to 1e6;
  - rounding-level coordinates are neither counted nor clipped to a full step;
  - it produces no NaN, and no full step, on zero-curvature directions. It does produce a full step
    when the guard is removed, so the guard is what prevents it;
  - it handles the SUA layout, and splits its statistics by input-eigen column on a normalisation
    layer.
- **`"ema"`:**
  - it equals `"quantile"` at its first step and at every step while the momentum keeps one size;
  - after 999 steps at one size, a momentum 10× smaller or 3× larger is compared with the
    threshold the history predicts, to 1e-9;
  - a first step five times too large weighs about 1/100 of the average after 100 steps, not 0.90;
  - multiplying the whole history and the curvature by one constant changes nothing;
  - an all-zero momentum, first or later, does not move the average.
- **`"fixed"`:**
  - it is `"quantile"` bit for bit until the calibration step, then freezes the lower median of the
    window's quantiles;
  - it is per layer: multiplying one layer's `s` by 1 024 changes nothing, before or after
    calibration;
  - the switch is continuous at constant input, and afterwards the clip is conditional (a smaller
    momentum moves the unclipped coordinates proportionally and clips fewer);
  - a layer first reached after calibration freezes its first quantile;
  - through the optimizer, every layer freezes once, with its own value, and never moves again.
- **`norm_exact_rescaling`:**
  - the recomputed `x̂` is exactly the layer's own normalised output, for `LayerNorm` on token
    inputs and `BatchNorm2d` in training and in evaluation mode;
  - the per-row gradients sum to the real `weight.grad`/`bias.grad`, in the optimizer's column order;
  - the stored `s*` and `Θ` of both normalisation layers of a real network equal the mean of the
    squared projected per-row gradients, recomputed from autograd, to 1e-10;
  - at the same step, only the normalisation layers' statistics change; the option is inert,
    bit for bit, on a network without normalisation layers; it runs with every form.
- **Through the optimizer:** the parameters move by `lr` times the clip of the bias-corrected
  momentum, and the buffer stays the raw one. All four layer types train, under the floor and all
  three clips, in both modes. Every clip trajectory is bit-identical at `λ = 1e-3` and `1e-11`.
- **The decision script** (§5): it returns the planted verdicts on complete synthetic data. Any of
  seven defects makes it exit with status 2 and print "incomplete", never "equivalent": a crashed or
  absent cell, a smoke file, a clipfixed cell that never froze, a cell run without the corrected
  statistic, a failed determinism check, a failed inertness check. The lr control qualifies the main
  arm's verdict, and the repro diagnostic does not vote.

---

## 4. The experiment

**Networks.** The three whose accuracy responds to `λ` and on which E10/E13/E14 located the optimum
with five seeds: `cnn_gn_cifar` (GroupNorm CNN), `vit_micro_cifar` (small ViT), `cct_2_3x2_cifar`
(compact convolutional transformer). Not `resnet20_cifar`: E14 found it flat in `λ` to within 0.6
points, so it cannot tell two ways of handling `λ` apart.

**Modes.** `ekfac` and `tekfac`, the two that divide coordinate by coordinate in an eigenbasis and
the two with the largest E14 gains. The forms are not defined for the other three (§3).

**Seeds.** 0–4, as in E10/E13/E14. At a given seed, every cell shares its initialisation and data
order.

**Protocol, fixed across every arm.** E10/E13/E14's, with one change:
- batch 32, 15 epochs, the cosine schedule clamped at its end, 4 data-loader workers per process
  (the worker count changes the trajectory, so it is the same everywhere);
- `eig_before_rescale=True`, the Cutout augmentation;
- **`norm_exact_rescaling=True` in every arm** (§11). The shipped estimator otherwise;
- final-epoch validation accuracy on the run's own 5 000-image split, and test accuracy on the
  10 000 CIFAR-10 test images.

The driver refuses any other setting unless `E16_SMOKE=1`, which the job script never sets.

**Arms, per (network, mode, seed).** Cells are listed in this order and dealt to the job's three
processes in turn. The seed-0 checks therefore run first, and a wall-clock kill would cost the `lr`
control first.

| arm | what varies | values | runs |
|---|---|---|---|
| `dupcheck` (seed 0 only) | the `add` cell at E14's best `λ`, run a second time in another process | must be bit-identical to its twin | 1 |
| `repro` (seed 0 only) | the same cell with the shipped estimator | compared with the stored E14/E13/E10 cell: a diagnostic; on `cnn_gn_cifar`, also compared with its `add` twin, which must be bit-identical | 1 |
| `add` (the E14 fix) | `λ`, `lr = cap·λ` | the 5 values below | 5 |
| `floor` (family A) | `λ`, `lr = cap·λ` | the same 5 values | 5 |
| `clipema` (B, `"ema"`, **main arm**) | `q`, the clipped fraction | 0.99, 0.95, 0.9, 0.7, 0.5, 0.3, 0.1 | 7 |
| `clip` (B, `"quantile"`) | `q` | the same 7 values | 7 |
| `clipfixed` (B, `"fixed"`) | `q` over the calibration window | the same 7 values | 7 |
| `cliplr` (control) | `lr` of `clipema` at `q = 0.7` | `lr/10`, `lr/3`, `3·lr` | 3 |

`λ` grids, five values at half-decade spacing around each network's located optimum:

| network | `λ` values (both modes) |
|---|---|
| `cnn_gn_cifar` | 3e-10, 1e-10, 3e-11, 1e-11, 3e-12 |
| `vit_micro_cifar` | 1e-9, 3e-10, 1e-10, 3e-11, 1e-11 |
| `cct_2_3x2_cifar` | 3e-10, 1e-10, 3e-11, 1e-11, 3e-12 |

**Why E14's cells are not reused.** The corrected statistic changes the `add` trajectories on the
two networks with LayerNorms. On `cnn_gn_cifar`, which has none hooked, it does not: the `repro`
check verifies that bit for bit. But the jobs also run on another GPU slice, three cells at a time,
and that may change the floating-point order of operations. So the baseline is rerun everywhere.
Every comparison is then between cells from one job: same code, same estimator, same hardware, same
seed. The stored E14 cells survive as a diagnostic: `repro` says how far this hardware and code
reproduce them.

**The two checks at seed 0.**
- *Determinism.* `dupcheck` runs one `add` cell a second time, in another process of the same job.
  The two must agree on every recorded field (test accuracy and loss, validation curve, distance
  travelled, step count). A failure means the runs are not reproducible on this hardware under
  concurrency, and E16 is not read.
- *Inertness* (`cnn_gn_cifar` only). The same cell under the shipped estimator must be bit-identical
  to the `add` cell, since the network has no hooked normalisation layer. A failure means the new
  option changes something it should not.

**How the grids are balanced.**
- The `add` baseline is selected from the **same** five values as the floor, so the two `λ`-based
  families choose from one grid.
- Each clip gets seven values of `q` against the floor's five values of `λ`. That asymmetry favours
  the clip slightly in selection, which is why rule 1 selects on validation and judges on test.

**Two choices that make the clip arms comparable with the `add` cells.** Both are deliberate and both
would otherwise be confounds.

1. *The parameters the optimizer does not precondition are frozen in the clip arms.*
   - They are `pos_embed` on the two transformers (2 048 and 8 192 parameters) and `cnn_gn_cifar`'s
     GroupNorm layers (224 parameters).
   - In the `add` cells at `λ ≈ 1e-10` they receive `cap·λ` ≈ 3e-11 to 1e-10 times their momentum
     per step, so they are frozen in effect.
   - A clip arm runs at `lr = 1e-3`. Unfrozen, it would give them momentum steps 10⁷ to 10⁸ times
     larger than in the cell it is compared with.
2. *No weight decay in the clip arms of the two transformers.*
   - Their decay is decoupled: `θ ← θ(1 − lr·wd)`. In the `add` cells `lr = cap·λ`, so the decay has
     vanished there (Part 5, rule 3 of `plan_lambda_dominance.md` names exactly this).
   - At `lr = 1e-3` it would be active in the clip arms only.
   - `cnn_gn_cifar`'s decay is coupled, added to the gradient, and is kept in every arm. One side
     effect, measured: the decay gives the head's logit-shift row a real momentum, so on
     `cnn_gn_cifar` that loss-invariant row took up to 21 of 63 clipped slots between steps 1 100 and
     1 183, until the decay had driven it to zero.
   - How much the vanished decay is worth is E15's `wdctrl` question, not this one.

**Why `lr` is not tuned for the clip.** Under the clip, `lr` is the largest per-coordinate step (at
the calibration step, for `"fixed"`).
- It is set to the benchmark's own `lr`, 1e-3. At `q = 0.99` that is sign-momentum at an Adam-like
  step size.
- The `cliplr` control brackets it on the main arm at `q = 0.7`, from `lr/10` to `3·lr`. The low
  end covers the 1e-4 the benchmarks give Adam on the two transformers.
- The control does not vote. If it wins, the main arm's verdict is reported as "limited by `lr`"
  (rule 5).

**Recorded along the way.** Every 50 steps up to step 5 000, then every 1 000 steps, for every hooked
layer:
- its mean stored curvature;
- for the floor, the fraction of directions above `λ`, where the floor does nothing;
- for the clips: `γ`; the fraction clipped among the active coordinates, split by input-eigen column
  on a normalisation layer; the fraction of active coordinates; the fraction where the guard binds;
  the share of the step carried by coordinates below 1e-2 × rms;
- for `clipema`, the momentum's size against its running average; for `clipfixed`, whether `γ` is
  frozen yet.

Once per process: the torch, CUDA and cuDNN versions, the GPU, the TF32 flags, the worker count, the
git commit and whether the tree was dirty.

**The jobs.** One job per (network, seed, mode): 30 jobs. Each runs its cells as three processes
sharing one `h100_3g.40gb` slice (3/7 of an H100), then merges their files. These networks are
small and latency-bound at batch 32, so one run leaves most of a slice idle. Rorqual offers 1g.10gb,
2g.20gb and 3g.40gb slices and whole H100s (`sinfo`, 2026-09-21).

| network | `add`, measured on 1g | clip / clipema / clipfixed, projected upper bound | cells per job | cell-time per job | at 3 per slice | `--time` |
|---|---|---|---|---|---|---|
| `cnn_gn_cifar` | 72–75 s | ~120 / 140 / 110 s | 34 (+2 at seed 0) | ~65 min | ~22 min | 0:50 |
| `vit_micro_cifar` | 149–152 s | ~270 / 325 / 245 s | 34 (+2) | ~145 min | ~48 min | 1:40 |
| `cct_2_3x2_cifar` | 197–212 s | ~315 / 360 / 285 s | 34 (+2) | ~170 min | ~57 min | 2:00 |

The last column but one assumes each process keeps its 1g speed on the larger slice, which has not
been measured: hence `--time` about 2× that. The three processes share the slice by time-slicing
(no MPS), so how much they overlap is exactly what is unknown.
- If they do not overlap at all, a job takes its full cell time (the fifth column), which is above
  its limit.
- So submit one job first, and read its timing (every cell prints its own seconds) and its seed-0
  checks. If needed, raise `--time` to about 1.3× the measured job time, then submit the rest.
- A job killed at its limit keeps its finished cells, since each is written as soon as it ends. But
  a resubmitted job reruns them all. Summed over every cell, the whole experiment is about 62 hours of run time,
6 of them for the rerun `add` baseline (computed from the table). It runs as 30 jobs of under an
hour each (projected), instead of 15 jobs of 2 to 5 hours.

**Files.**
- Driver: `fisher_ref/experiments/e16_floor_clip.py` (a shard process, or the merge with
  `E16_MERGE=1`).
- Job: `fisher_ref/slurm/e16_floor_clip.sh`; the submission loop is in its header. Commit first: the
  decision script refuses files from a dirty tree or from different commits.
- Decisions: `fisher_ref/experiments/e16_decisions.py`, run once all thirty merged files are in
  `fisher_ref/outputs/`.

---

## 5. Decision rules (pre-registered; copied from `plan_lambda_dominance.md` E16)

**Nothing incomplete is read as a verdict.** Before any rule is applied, every file must be a
production run: not a smoke, 15 epochs, batch 32, full grids and split, calibration at 2000.
- Every shard must be merged, and every planned cell present and not crashed.
- Every cell but the repro diagnostic must have run with `norm_exact_rescaling`, and every
  `clipfixed` cell must have frozen every layer.
- All thirty files must come from one commit of a clean tree.
- At seed 0, the determinism check must pass everywhere, and the inertness check on `cnn_gn_cifar`.

A (network, mode, family, value) is usable only with a finite final validation and test accuracy at
all five seeds. A family whose grid or comparison has an unusable value is **incomplete**, never
"equivalent", and the script exits with status 2.

1. **Select on validation, judge on test.** Within each family, the chosen `λ` or `q` is the one with
   the best five-seed mean of the final-epoch *validation* accuracy. The comparison is then made on
   *test* accuracy at the chosen values.
2. **Per (network, mode):** `Δ` = test accuracy of the family minus test accuracy of `add`, paired
   by seed. The family **wins** if the mean of `Δ` exceeds 2 standard errors of `Δ`, **loses** if it
   is below −2 standard errors, and **ties** otherwise. A zero standard error with a nonzero mean is
   flagged as degenerate, not voted. Applied to `floor`, `clip`, `clipema` and `clipfixed`.
3. **Verdict per family.**
   - **Better than the E14 fix** if it loses nowhere and wins for both modes on at least two of the
     three networks.
   - **Worse** if it loses on at least two networks.
   - **Equivalent** if it loses nowhere and is not better.
   - **Mixed** otherwise.
4. **Transfer.** A value is **transferable** if it lies inside the plateau of all six
   (network, mode) pairs. The plateau voted on is the pre-registered one: every value whose five-seed
   mean test accuracy is within one standard error of the best mean, that standard error being the
   best value's own (E14's reading). E13's own reported plateaus were computed with a two-sample
   criterion, `m_best − m_v ≤ √(SE_best² + SE_v²)`. It is reported alongside and not voted on.
   The rule is applied to `q` for each clip, to `λ` for `add`, and, for E17, to `λ` for the floor.
   **Family B's case rests on this rule**: a clip is worth adopting even at "equivalent" under rule 3
   if some `q` transfers and no `λ` does, because it then removes a per-network search.
5. **`lr` control.** If any `cliplr` cell beats the `clipema` cell at the same `q = 0.7` under rule
   2's criterion, `clipema`'s rule-3 verdict is printed with **"limited by `lr`"** for that
   (network, mode). It is then not read as a verdict on clipping. With three cells each tested at
   2 SE on five seeds, a false "limited by lr" is expected up to 16 % of the time per
   (network, mode), from the one-sided t₄ tail (5.8 % per cell).
6. **Inside family B**, with rule 2's criterion, each arm at its own chosen `q`:
   - **`clipema` − `clip`** measures the per-step normalisation. A tie means it neither helps nor
     hurts, and the two arms' verdicts are about the clip. If `clip` wins, the normalisation is
     helping, and a `clip` win against `add` is not a win for clipping as such.
   - **`clipfixed` − `clipema`** measures a frozen threshold against one that follows the momentum.

**Reported, not voted:** the repro diagnostic at seed 0.

**For E17.** E17 (same document) takes its candidates from this script's output. The script prints
them as E17 defines them:
- each clip arm and the floor if rule 3 rates it "better" or "equivalent" and rule 4 finds a
  transferable value;
- add's transferable `λ`, or failing one the geometric mean of add's six selected `λ`;
- where several values are transferable, the one with the best validation accuracy averaged over
  the six pairs, which is E17's candidate-A rule.

Every E16 arm ran with `norm_exact_rescaling`, so a candidate runs that way too, to be "configured
exactly as its source experiment ran it".

---

## 6. Predictions, written before any run

- **Family A ties everywhere**, `|Δ| < 0.5` points, and its chosen `λ` equals `add`'s or is one grid
  step away. If it wins, the curvature values near `λ` matter more than their count suggests.
- **Family B is the open question.** Two readings compete:
  - *It wins or transfers.* What `λ` really set was a step-size cap. A curvature step capped
    against the momentum's own size is a better way to set it.
  - *It loses.* At `q ≥ 0.9` it is close to sign descent in the eigenbasis, which uses almost no
    curvature information. At `q ≤ 0.3` it is close to the undamped step, which the fall of every
    `ekfac`/`tekfac` curve below 1e-11 (E13, E14) says is too noisy. Nothing in the middle beats a
    tuned additive `λ`.
- **`clipema` and `clip` differ mostly early in training.** The audit measured them within 0.015–0.039
  of each other in clipped fraction after step ~3 000.
- **`clipfixed`'s clipped fraction drifts after calibration.** On a 300-step check it ranged 0.31–0.73
  across layers at `q = 0.7`, 100 steps after the switch. The exact curvature's overall size falls
  about 100-fold between the first and last checkpoint on A1 (drift campaign, `plan_exp_lot2.md`
  §6.4), so the drift over a full run can be large. Its direction is not predicted.
- **The corrected statistic does not move `add`'s optimum.** On every (network, mode), rule 1
  selects E14/E13/E10's `λ` for `add`, or a value one grid step away. The option changes the step
  only on the normalisation layers: by at most 2.45× in median at step 2 000 (§11), on 320 of
  21 098 parameters on `vit_micro_cifar` and 1 280 of 283 723 on `cct_2_3x2_cifar`.
- The chosen `q` falls inside Sophia's 0.5–0.9, if the published range carries over.

---

## 7. What this experiment will not settle, and its known limits

- **TF32.** Nothing in the training path disables TF32, and PyTorch's default lets cuDNN run
  convolutions in TF32 on the H100. Emulated on CPU, that makes conv weight gradients 0.6–1.1 % off
  on `cnn_gn_cifar` and `cct_2_3x2_cifar`, i.e. noise per eigen-coordinate of 1.4–9.2e-3 × rms,
  above the 1e-3 × rms guard. The clip's direction is then 4–19 % noisier than `add`'s on
  `cnn_gn_cifar`. Every arm runs under the same default, and the flags are recorded in every file.
  The clip's verdicts are verdicts on this hardware.
- **The guard, the horizon and the calibration.** `clip_guard = 1e-3`, `H = 1000` and the
  calibration at step 2 000 over 1 000 steps are fixed, not swept.
- **A frozen threshold meets a moving curvature.** After `clipfixed` freezes, its step still follows
  `s*`, which is 92 % one minibatch and jumps at every factor update: by a median of 0.04–0.16
  decades per update, 0.11–0.5 at the 90th percentile (audit, measured). `clipema` absorbs those
  jumps at every step; `clipfixed` does not. So `clipfixed − clipema` also measures exposure to that
  noise, not only "frozen against lagged".
- **The first ~800 steps.** Until the identity seed has decayed, the stored `s` is nearly uniform, so
  the clips take curvature-free, sign-like steps at full `lr`, while the paired `add` cells move at
  most 0.018 times the momentum per step. The two families differ in kind there.
- **The input factor of a normalisation layer is still the surrogate.** `norm_exact_rescaling`
  corrects the eigen-rescaling (`s*`, `Θ`), not the 2×2 input factor built from the raw channel mean
  (`CLAUDE.md` §4.6), which the eigenbasis comes from. Lemma 1 makes the corrected rescaling optimal
  in whatever basis it is measured in, so the statistic is right. The basis may still not be the best
  one for these layers.
- **Batch size.** Every run is at batch 32. The clips are insensitive to the `1/batch²` rule by
  construction (for `clipfixed`, once calibrated at the batch size it runs at); `add` and `floor` are
  not. The transfer of a fixed `λ` across batch sizes is open question 2 of
  `plan_lambda_dominance.md`, and is not tested here.
- **Anything about `kfac`, `tkfac`, `diag`,** for which the forms are not defined.
- **The shipped operating point.** Nothing here runs at `λ = 1e-3`, where the feasibility study
  already showed the floor is inert and the clip degenerates to sign descent.
- **Wall-clock cost.** The runs are fixed-epoch, not fixed-time. Wall time is recorded per run, so
  the clips' overhead can be read afterwards, but it does not enter any verdict.

---

## 8. Status

| step | state |
|---|---|
| optimizer options, refusals, P2 guard | **done** |
| audit (six agents: one per arm, layer types, protocol) and every fix | **done** (§10) |
| the corrected normalisation statistic, and E16's own baseline | **done** (§11) |
| tests: 120 new; full suite 1014 passed, 41 skipped | **done** |
| driver, decisions script, job script | **done**. Local smokes: every arm on all three networks; three processes sharing a job, merged, with the determinism check and the inertness check passing |
| the 30 cluster jobs | **not submitted**. Commit first; submit one job, read its timing and its seed-0 checks, then the rest |
| results | — |

---

## 9. First amendment (same day, before submission): one clip arm became three

The first version of this plan had one clip arm, what is now `clip` (`"quantile"`). Reviewing it
found that its threshold is reset at every step, so the momentum's size never reaches the step. It
is a per-layer normalisation with a clipped shape, not a clip that acts only beyond a ceiling. A
verdict on it could not have told the two effects apart.

**Added:** `clipema` (the main arm) and `clipfixed` (the conditional clip), rule 6, and the `lr`
control moved to the main arm. **Changed while implementing them:** the clipped fraction counts only
active coordinates, and the clip receives the bias-corrected momentum.

---

## 10. Second amendment (same day, before submission): the audit and its fixes

Six agents audited the code before submission, in parallel: one per arm (`floor`, `clip`, `clipema`,
`clipfixed`), one on every layer type of the three networks, one on the driver, the decision script
and the job. Each measured rather than argued, on the real networks, with the real optimizer. What
they found, and what was done:

1. **A threshold shared by the whole network was incoherent** (`clipfixed`; blocker). The stored
   curvature's scale error differs between layers by up to four orders of magnitude (§2). At a
   pooled calibration, the head of `vit_micro_cifar` was clipped on 3 % of its coordinates and its
   final LayerNorm on 97 %. The head's steps shrank 10³–10⁴× at the switch, and the split was
   immediate and stable, not a drift. *Fixed:* every threshold is per layer. `clipfixed` freezes each
   layer's own window median. The option to give one `γ` for the whole network was removed.
2. **`clipema`'s average was dominated by its first step** (blocker). Seeded with one raw gradient,
   4.4–5× larger in rms than the momentum settles to, the average stayed more than 10 % off until
   step 2 400–3 900, 12–18 % of a run. After step ~3 000, `clipema` and `clip` coincide, so the seed
   artefact was as large as the quantity rule 6 reads, and on `cnn_gn_cifar` it flipped its sign. An
   all-zero first momentum seeded `log(tiny) = −87` and clipped everything for thousands of steps.
   *Fixed:* Adam's bias correction, and zero momenta are skipped.
3. **`clipfixed` froze one draw** (major). Calibrating on the single step 2000 froze a quantile that
   varied ×3.6 between factor updates on `vit_micro_cifar`, about one grid step of `q`. *Fixed:* the
   median over the 1 000 steps before.
4. **The clipped count could be `c − 1`** (minor), through rounding: 3–6 % of module-steps on real
   states. *Fixed:* clipped is defined as `r ≥ γ` on the tensor `γ` is selected from. The count is now
   exact on 400 random problems in fp32.
5. **The jobs would have overrun** (blocker). One clip call was 44 operations per layer per step,
   against 2 for `add`, with one host–device synchronisation per layer per step, and statistics
   computed every step but read every 1 000. Projected from measured per-operation costs, the
   `vit_micro_cifar` jobs ran 35 minutes past their limit. *Fixed:* the quantile is a sort and a
   gather, with no synchronisation, and the statistics are computed only when read.
6. **The decision script could turn missing data into a verdict** (blocker). A family with no valid
   cell was rated "equivalent to the E14 fix". Crashed seeds were dropped without notice. Rule 5 was
   computed and never applied, and the plateau was ambiguous between E13's and E14's readings.
   *Fixed:* §5's gate, the "incomplete" verdict and exit status 2, rule 5 printed into rule 3, the
   plateau pinned with the other reading reported.
7. **Nothing stopped a leftover smoke setting or a changed environment** (major). The job passed
   `--export=ALL`, and the worker count, which changes the trajectory, was not pinned. No run recorded
   its software, GPU or commit. *Fixed:* the job unsets the smoke variables and pins the production
   values; the driver refuses anything else without `E16_SMOKE=1` and records all of the above.
8. **Smaller fixes.** A device tensor built for `λ` at every floor call; docstrings; "1 407 steps per
   epoch" (1 406); "a billion times" (10⁷–10⁸); "exactly c"; "sign(M)" (active coordinates only);
   "2 × 10⁸" (the scale error at batch 32 is not one number); `CLAUDE.md`'s statement that
   `precondition` receives `m̂`; strict JSON for crashed cells; eight new tests on the CCT layer types
   the tests had never exercised.

**What the audit verified as correct.** `add` is bit-identical to the pre-change code. The driver
reproduces E4's run path bit for bit. The freeze lists match the parameters no hooked module owns.
Hooks fire correctly under the freeze. On every layer type of the three networks, the layout, the
weight/bias split and the applied update are exact. There is no NaN anywhere, including over
2 601-step runs.

---

## 11. Third amendment (same day, before submission): the normalisation statistic, corrected

**The defect.** A normalisation layer (`LayerNorm`, `BatchNorm2d`) has two parameters per channel, a
scale and a shift. Its per-row gradient is `[δ_t ⊙ x̂_t, δ_t]`: the output gradient times the
*normalised* activation for the scale, the output gradient alone for the shift. The optimizer's input
factor for these layers is instead built from `h̄_t = [z_t, 1]`, with `z_t` the channel mean of the
*raw* input (`CLAUDE.md` §4.6, inherited from both reference repositories). `ekfac` and `tekfac` then
estimated their eigen-rescaling `s*`/`Θ` from the per-row product `[z_t·δ_t, δ_t]`, which is that
model's gradient, not the layer's. Measured on the real networks at step 300, the true statistic of
the scale-carrying eigen-column is **295–989×** the modelled one on `vit_micro_cifar`, and **80–106×**
on `cct_2_3x2_cifar`. The shift column is right, 0.96–1.03×. At step 2 000 the audit had measured
800–8 070× on the ViT's block LayerNorms.

**Why it mattered for E16.**
- Under `add` at E14's `λ`, both columns of these layers are dominated by `λ`, so the defect never
  reached E14's steps.
- Under the clip it does. On `vit_micro_cifar` at `q ≤ 0.3`, every clipped slot of a block LayerNorm
  went to the scale column, and the shift steps were throttled 3–117×.
- The second amendment recorded this as a limit. It is now corrected instead.

**The fix: `norm_exact_rescaling`,** an option of `ekfac`/`tekfac`, off by default.
- The forward hook also recomputes `x̂_t` from the layer's input, with the statistics the layer used:
  per token for `LayerNorm`; the batch's per-channel statistics for `BatchNorm2d` in training, its
  running statistics in evaluation.
- The rescaling is then the mean of the squared exact per-row gradient projected into the same
  eigenbasis: `E_t[(Q_outᵀ G_t Q_in)²]` with `G_t = [δ_t ⊙ x̂_t, δ_t]`.
- EKFAC's Lemma 1 (`ekfac_1806.03884.pdf`, App. A.1) makes that the optimal diagonal in that basis,
  whatever the basis. The factors and the eigenbasis are unchanged.
- The option is inert on a network with no hooked normalisation layer. It is refused with
  `fisher_batch_samples` on a `BatchNorm2d` network, whose batch statistics cannot be recomputed
  from part of the batch.
- Tested against autograd (§3).

**The correction also reaches `add`'s step.** Under `add` the divisor of each coordinate is
`s + λ`. Measured with `ekfac` at step 2 000, at E14's `λ`, from one initialisation and one data
order (seed 0), once with each statistic. The settings are the E-series ones (batch 32,
`eig_before_rescale`, `lr = cap·λ`), run on CPU without the cosine schedule, which moves `lr` by
1.1 % in 2 000 steps. The script is a one-off measurement, not part of the suite.

| network, `λ` | shipped statistic | corrected: first LayerNorm | corrected: final LayerNorm | corrected: the other three |
|---|---|---|---|---|
| `vit_micro_cifar`, 1e-10 | 1.00 | 2.45 (4.54) | 1.05 (1.11) | 1.00–1.07 (1.01–1.16) |
| `cct_2_3x2_cifar`, 3e-11 | 1.00 | 2.21 (2.87) | 1.94 (2.56) | 1.11–1.17 (1.16–1.27) |

The entries are `(s + λ)/λ` on the eigen-column that carries the scale (on `cct_2_3x2_cifar`, most
of it): the median over the layer's channels, with the 90th percentile in brackets. So a value of
1.00 means `λ` alone sets the divisor, and the stored statistic plays no part.
- On `vit_micro_cifar` the other column is unchanged.
- On `cct_2_3x2_cifar` it moves too (1.02–1.19 in median), because there the 2×2 eigenbasis sits at
  about 45° and both columns carry some of the scale.
- At step 300 the two runs are identical, and at step 1 000 within 12 % in median. Until about step
  1 000 the stored values are still dominated by the residue of the identity they start from
  (`0.08^k` after `k` factor updates: 4.1e-5 at step 300, 400 000 times `λ` on `vit_micro_cifar`).

**What changes in E16.** Every arm runs with the option on, including `add`. The E14 fix is
therefore rerun inside E16 on all three networks, rather than read from E14's files:
- on the two LayerNorm networks the corrected statistic changes `add`'s steps (table above);
- the jobs now also run on a larger slice, three cells at a time (below).

`addfill` is gone, since `add` now covers the full grid everywhere. The stored E14 cells serve as a
seed-0 diagnostic (`repro`), and the pipeline is checked by a determinism check and, on
`cnn_gn_cifar`, an inertness check (§4).

**The jobs.**
- One job per (network, seed, mode) instead of per (network, seed): 30 jobs.
- Each job runs three processes on one `h100_3g.40gb` slice instead of one process on a `1g.10gb`
  slice.
- Projected, each job takes under an hour instead of 2 to 5, for a similar total. The per-slice
  speed is unmeasured, so the first job is a timing check.
