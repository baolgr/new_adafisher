# Auditing one AdaFisher step, and justifying every constant in it

**Status.** §4 is a **finding**, verified against the paper and both reference implementations while
writing this report. §5-§11 are a **proposal**: none of it has been run.

## 1. Why this report exists

The supervisor's remark was: *"I'm not sure how much the AdaFisher framework can be trusted. You need
to do a sanity check on every step of the optimizer step and rationalize the parameter selections as
such."*

It is worth being precise about what is being asked, because it is not "find the bug". It is two
separate requests:

1. **Sanity-check each step.** One update of the optimizer is a chain of about seven distinct
   operations. Each one is a place where the code could be doing something other than what the papers
   describe, in a way that no test currently notices. The request is to check each link of the chain
   against its definition, rather than against another implementation of the same possible mistake.
2. **Justify the constants.** The optimizer is configured with roughly eight numbers
   (`lr`, `beta`, `Lambda`, `gammas`, `TCov`, `T_inv`, `T_eig`, `T_re`). Every one of them is
   currently inherited from the AdaFisher paper or from Adam's conventions. None of them is derived
   from a measurement made in this repository. The request is to be able to say, for each number,
   *what it controls, and what measurement would show that this value is a reasonable one*.

The remark is also well-founded on evidence already in this repository, not merely a matter of
prudence: on the project's primary benchmark (the MNIST autoencoder) all five Fisher modes stop
improving after two epochs at a mean-squared error of about 0.7085, while plain Adam, from the same
initialisation, descends to 0.4836. The parameters of the Fisher arms move by 0.2-0.5% between 10%
and 100% of training. Something in the chain produces steps of the wrong size, and until §4 nobody
knew which link.

## 2. What one step actually does, in plain terms

To audit a chain you first have to name its links. The vocabulary first:

- **`h`** — the *input* to the layer, captured by a forward hook. "Bias-augmented" (written `h̄`) just
  means a constant 1 is appended, so the layer's bias can be treated as one more column of its weight
  matrix.
- **`delta`** — the gradient of the loss with respect to the layer's *pre-activation* output, captured
  by a backward hook. It is the signal coming back down the network.
- **`A = E[h̄ h̄ᵀ]`** — the covariance of the layer's inputs, averaged over the batch. Size
  (inputs+1) x (inputs+1). The paper calls it `H`.
- **`B = E[delta deltaᵀ]`** — the covariance of the backpropagated signal. Size outputs x outputs.
  The paper calls it `S`. Together, `H` and `S` are the paper's "Kronecker factors" (KFs).
- **`F ≈ A ⊗ B`** — the Kronecker approximation of that layer's block of the Fisher information
  matrix. The Kronecker product is what makes this affordable: the true block is
  (params x params), unstorable; the approximation factorises it into the two small matrices above.
- **`F̃`** — whatever the active mode actually *applies*, after time-averaging, after damping, and (in
  `diag` mode) after min-max normalisation. **This is the object the audit is about.** `F` is what the
  papers define; `F̃` is what the code uses.
- **`m`** — the momentum buffer, exactly Adam's first moment: `m <- 0.9 m + 0.1 g`.
- **`lambda`** ("Tikhonov damping", the constructor's `Lambda`) — a small number added to `F̃` so it
  can be inverted where the curvature estimate vanishes. It also silently sets a **ceiling on the step
  size**: where the curvature estimate is negligible, the update becomes `lr/lambda` times the
  gradient. With the defaults that ceiling is a thousand-fold.
- **min-max normalisation** — rescaling a vector so its smallest entry becomes 0 and its largest 1:
  `(t - min) / (max - min + 1e-6)`. Note what this implies: it **destroys any overall scale**. Two
  inputs differing by a constant factor produce the same output.

And the chain, for one layer, during one call to `step()`:

| # | Stage | What it does | Where |
|---|---|---|---|
| S1 | capture | forward hook stores `h`, backward hook stores `delta` | [optimizer.py:133-134](../../src/adafisher_modes/optimizer.py#L133-L134) |
| S2 | factor build | turns `(h, delta)` into this batch's `A`, `B` | [factors.py](../../src/adafisher_modes/factors.py) |
| S3 | normalisation | `diag` mode only: rescales each factor's diagonal into `[0, 1]` | [minmax.py](../../src/adafisher_modes/minmax.py) |
| S4 | time averaging | folds the new batch's factor into a running estimate, every `TCov` steps | [ema.py](../../src/adafisher_modes/ema.py) |
| S5 | damping + inversion | adds `lambda`, inverts or eigendecomposes, every `T_inv`/`T_eig` steps | `approximations/*.py` |
| S6 | momentum | `m <- beta m + (1-beta) g`, then bias correction | [optimizer.py:148-151](../../src/adafisher_modes/optimizer.py#L148-L151) |
| S7 | update | `theta <- theta - lr * F̃⁻¹ m̂` | [optimizer.py:184](../../src/adafisher_modes/optimizer.py#L184) |

Three structural facts about this chain are worth stating early, because much of the audit follows
from them.

**There is no square root.** Adam divides the gradient by `sqrt(v)`; AdaFisher divides by `F̃` itself.
This is deliberate and defended at length by the paper (Appendix, *"Square Root Utilization"*), so it
is not a transcription question — but it changes the *units* of the update, with consequences nobody
here has accounted for. See §8.

**The curvature is refreshed a hundred times less often than the momentum.** `TCov = 100` means
stages S1-S5 run on one step in a hundred; S6-S7 run on every step. So for 99 steps out of 100 the
optimizer applies a **fixed linear operator** to a changing momentum vector. That is what makes the
audit method of §5 possible. It is also an assumption — that curvature barely moves over 100 steps —
that nobody has measured on these models, and it is **not in the paper**: Algorithm 1 recomputes the
factors and their average at *every* step (its lines 3-4, inside the `while` loop). `TCov` is an
addition made by the implementation.

**Min-max comes before the averaging, not after.** The paper's Algorithm 1 does line 4 "Compute EMAs
of `H_D` and `S_D` using Eq. (3)", then line 5 "Compute `F̃_D` using Eq. (4)" — and Eq. (4) is where
min-max lives. So the published order is *average, then normalise*. The official code, and therefore
this port, does the opposite ([AdaFisher.py:412](../../reference_repos/AdaFisher/optimizers/AdaFisher.py#L412),
[:431](../../reference_repos/AdaFisher/optimizers/AdaFisher.py#L431):
`update_running_avg(MinMaxNormalization(H_D_i), ...)`). This was a conscious choice in this repository
— `CLAUDE.md` records it as "the placement used by the official repository, not after accumulation" —
and on its own it looks like a detail. §4.4 shows it is not: it is what turns the defect of §4.2 from
harmless into dominant.

## 3. What is currently guaranteed — and what is not

The test suite is large (341 tests) and genuinely good at one thing. Naming exactly which thing is the
whole reason this report exists.

| Currently proven | Not proven by any test |
|---|---|
| **Faithfulness of the port.** `diag` with min-max off is bit-exact against `FisherAdapTune`; the min-max function is bit-exact against the official repository. | That either reference implementation matches the published algorithm. A bit-exact port of a mistake passes. **This is exactly what happened — see §4.** |
| **Internal consistency.** The factored `precondition()` agrees with the dense `f_tilde()`; EKFAC's error is no worse than K-FAC's; TKFAC preserves the trace; the eigenbases are orthonormal. | That the applied operator is `(F + lambda I)⁻¹` **at the right overall scale**. Every one of these properties is invariant to multiplying `F̃` by a constant. |
| **It runs.** Smoke tests on every layer type; every parameter is updated. | That the curvature information does any work. There is no control run where `F̃` is replaced by something uninformative. |
| Memory and cost behave as claimed (SUA's 80.7x reduction, etc.). | That a single number from the AdaFisher paper is reproducible with this code. |

In one sentence: **we tested equivalence to a reference, and never agreement with a definition.**

## 4. Finding: the implemented time-average is not the paper's Eq. (3)

### 4.1 What the paper says

Eq. (3), verbatim in structure:

```
H^(t) = γ · H^(t-1) + (1 − γ) · H_new ,      S^(t) = γ · S^(t-1) + (1 − γ) · S_new ,   0 < γ ≤ 1
```

An ordinary exponential moving average: **one scalar**, coefficients summing to 1, so a constant input
is reproduced exactly. Algorithm 1's stated defaults are `α = 0.001`, `λ = 0.001`, `γ = 0.8`,
`β = 0.9`, and the hyperparameter appendix says: *"The decay factor γ for AdaFisher was tuned within
{0.1, 0.2, …, 0.9, 0.99}. The optimal value is: γ = 0.8."* So the published rule at its published
default is `0.8 · old + 0.2 · new`.

The paper also states plainly what the average is *for*: it lets "the curvature estimation depend on
much more data than what can be reasonably processed in a single mini-batch."

### 4.2 What both implementations compute

Official repository,
[AdaFisher.py:48-58](../../reference_repos/AdaFisher/optimizers/AdaFisher.py#L48-L58):

```python
def update_running_avg(new, current, gamma: float):
    current *= gamma * 1e-1
    current += (gamma * 1e-2) * new
```

With `gamma = 0.8` — the paper's own tuned default — this is `0.08 · old + 0.008 · new`. There are two
stray decades relative to Eq. (3): the retention is `γ/10` instead of `γ`, and the weight on the new
observation is `γ/100` instead of `1 − γ`.

`FisherAdapTune`, the authoritative reference here, encodes the same thing in a two-parameter form,
[adafisher.py:28-30](../../reference_repos/FisherAdapTune/scripts/adafisher.py#L28-L30) with
`gammas = [0.92, 0.008]`:

```python
current *= 1 - gammas[0]     # 0.08
current += new * gammas[1]    # 0.008
```

`1 − 0.92 = 0.08 = 0.8 × 1e-1` and `0.008 = 0.8 × 1e-2`. **The two repositories are numerically
identical here, and neither implements Eq. (3).** This port reproduces them bit-exactly
([ema.py:19-21](../../src/adafisher_modes/ema.py#L19-L21)), which is precisely what it was asked to do.

So the earlier reading recorded in [plan.md §1.4](archives/plan.md) — that this is a *divergence
between the two repositories*, arbitrated in favour of FisherAdapTune — should be retired. It is not a
divergence between the repositories; it is a discrepancy between the paper and both of them.

### 4.3 The two consequences, as arithmetic

Write the rule as `c_{k+1} = 0.08 c_k + 0.008 n_k`.

**Consequence 1 — the estimate is 115 times too small.** Feed a constant statistic `n`. A genuine
average returns `n`. This returns its fixed point:

```
c* = 0.008 / (1 − 0.08) · n = 0.008696 · n ≈ n / 115
```

Each stored factor settles at about one hundred and fifteenth of the quantity it estimates. And
`F̃` multiplies two of them, so the compounded factor on the preconditioner is about
**115² ≈ 13 000**.

**Consequence 2 — there is no averaging left.** Unrolling, the weight on the observation from `j`
updates ago is proportional to `0.08^j`. Normalised:

| | most recent batch | one before | two before |
|---|---|---|---|
| implemented (`0.08`/`0.008`) | **92%** | 7.4% | 0.6% |
| Eq. (3) at `γ = 0.8` | 20% | 16% | 12.8% |

The implemented estimate is 92% a single minibatch. Combined with `TCov = 100` (itself absent from
Algorithm 1, §2), the curvature is estimated from **one minibatch out of every hundred steps** — which
defeats verbatim the purpose the paper gives for having an average at all.

**A note on the tuned value.** `γ = 0.8` was selected by grid search *against the implemented rule*,
so it is the optimum of `0.08/0.008`, not of `0.8/0.2`. The published number therefore cannot be read
as validating Eq. (3); and conversely, fixing the rule invalidates the tuned `γ`.

### 4.4 Why this is invisible in `diag` — and why the code's ordering makes it fatal instead

Min-max normalisation destroys overall scale. So:

- **In the paper's order** (average, then normalise — Algorithm 1 lines 4-5) the factor of 115 lands on
  the *input* of the normalisation and is **erased**. `F̃_D` lives in `[0, 1] + lambda`, and
  `lambda = 1e-3` is a genuinely small damping term. Consequence 1 is harmless; only consequence 2
  survives.
- **In the implemented order** (normalise, then average — §2) the factor of 115 lands on the *output*
  and nothing removes it:

```
normalised diagonals live in [0, 1]
=> after averaging, each lives in [0, 0.0087]
=> their Kronecker product lives in [0, 7.6e-5]
=> F̃_D = kron(H, S) + lambda, with lambda = 1e-3
=> F̃_D ranges over [1.0e-3, 1.076e-3]: a 7.6% spread across the entire layer
```

That is the whole point. In its paper-faithful default configuration, **`diag`'s preconditioner is
very nearly the constant `lambda`**, the curvature contributes at most 7.6% of variation, and the mode
reduces to momentum SGD with an effective learning rate of `lr / lambda = 1000 × lr = 1.0`.

For the four Kronecker modes there is no min-max at all (by design — it would destroy TKFAC's trace
preservation), so the 115x contraction of `A` and of `B` passes straight into `F̃`. There the damping is
split between the factors as `Ã = A + pi·sqrt(lambda) I` and `B̃ = B + sqrt(lambda)/pi I`
([kfac.py:96-99](../../src/adafisher_modes/approximations/kfac.py#L96-L99)), with
`sqrt(lambda) = 0.0316`. Shrinking the data-dependent part of each factor by 115 while leaving the
damping term alone pushes these modes **towards** damping dominance too — towards being scaled SGD.

**And for those four modes there is no defence from AdaFisher's own conventions.** They implement
K-FAC, EKFAC, TKFAC and TEKFAC, whose own papers and reference implementations use ordinary convex
running averages — `EKFAC-pytorch` does `m2.mul_(alpha).add_((1-alpha)*bs, ...)`
([ekfac.py:103](../../reference_repos/EKFAC-pytorch/ekfac.py#L103)) and
`addmm(..., beta=1-alpha, alpha=alpha/n)` ([kfac.py:175-176](../../reference_repos/EKFAC-pytorch/kfac.py#L175-L176)).
The four modes inherited AdaFisher's `gammas` because of this project's founding design rule —
"everything else is identical across the five modes" — which was the right call for comparability, but
it has propagated a transcription error into four implementations whose source papers do not contain
it. That decision now has to be made explicitly rather than by default.

### 4.5 The start-up transient nobody has looked at

Every mode seeds its state with the identity at step 0
([kfac.py:77](../../src/adafisher_modes/approximations/kfac.py#L77) and analogues), matching
Algorithm 1's `F̃_D = I`. With a retention of 0.08 applied only every `TCov = 100` steps, the seed's
weight is `0.08^k` after `k` factor updates:

| factor updates `k` | weight left on the identity seed | step in training |
|---|---|---|
| 1 | 0.08 | 100 |
| 2 | 0.0064 | 200 |
| 3 | 5.1e-4 | 300 |
| 10 | 1.1e-11 | 1000 |

So after the first update the state is `0.08 I + 0.008 A_raw`: a mixture whose composition depends on
the raw factor's own scale, and which is dominated by the seed wherever that factor is small —
typically the output factor `B`, built from backpropagated gradients. The effective step size then
changes by orders of magnitude over the first few hundred steps as the seed dies out. Under Eq. (3)'s
`γ = 0.8` the seed would decay as `0.8^k` instead, and would fade towards the *correct* fixed point
rather than towards one 115 times too small.

The re-warm study already recorded this arithmetic ("a fresh optimizer carries `0.08^k I`, a spurious
extra damping"), but treated it as a constraint on *offline* re-warming. It applies just as much to
every real training run, where nothing re-warms anything.

### 4.6 This explains four things already observed but never explained

None of the following is proof; but one mechanism accounts for all four, which is a reason to test it
first.

1. **`diag` takes huge early steps.** Recorded in the lot-8 notes as a curiosity: training loss 17.6
   after 4 steps for `diag` against ~2.3 for the Kronecker modes at `lr = 1e-3`. §4.4 predicts this
   *and its size*: the effective learning rate is `lr/lambda = 1.0`.
2. **`diag` is the only mode that responds to `lambda`, and only at `1e-5`.** The sweep over
   `{1e-5, 1e-3, 1e-1}` moved the four Kronecker modes by under 1% and moved `diag` only at `1e-5`
   (0.708 -> 0.578). §4.4 gives the crossover: `diag`'s data-dependent term maxes out at `7.6e-5`, so
   `lambda` dominates completely at `1e-3` and `1e-1`, and only at `1e-5` does the curvature estimate
   start to matter at all. The bracket straddles that crossover at exactly one of its three points.
3. **That sweep's conclusion is probably measuring the wrong thing.** It compared *final* losses on
   the benchmark where all four Kronecker modes collapse within two epochs. If the modes overshoot
   into a flat region in their first few hundred steps, no value of `lambda` can change the final
   loss, and "lambda is not the lever" becomes a statement about the plateau rather than about the
   optimizer. The repository currently records that sweep as having **exonerated** `lambda`; that
   reading should be weakened until the same sweep is examined over the first few hundred steps.
4. **The stall's shape fits an overshoot, not a small step.** The Fisher arms travel about 1.0 from
   initialisation and then stop, moving 0.2-0.5% over the last 90% of training; Adam travels 25.8 and
   keeps moving. A step size far too large early, in an all-sigmoid network, lands the parameters in
   saturation where gradients vanish and nothing moves again. That is a different diagnosis from "the
   steps are too small", and it implies a different fix.

### 4.7 What to run, in what order

These are cheap and they isolate one thing each. They should come **before** any learning-rate sweep,
because until they are done the quantity being swept is not well defined.

| Variant | What it isolates |
|---|---|
| `gammas = (0.2, 0.8)` in this port's parameterisation, i.e. `0.8·old + 0.2·new` | Eq. (3) as published, at its published `γ` |
| `diag` with min-max applied *after* the average | Algorithm 1's order (§4.4); should, on its own, remove `lambda`'s dominance |
| the 2x2 of the two above, for `diag` | separates "wrong scale" from "wrong ordering" — they interact, so neither can be read alone |
| `TCov = 1` against `TCov = 100`, with the corrected average | Algorithm 1 recomputes every step; `TCov` is an implementation addition |
| the corrected average for the four Kronecker modes | their own source papers' convention (§4.4) |

Also worth doing while at it: re-tune `γ` over the paper's own grid `{0.1, …, 0.99}` under the
corrected rule, since the published optimum belongs to the uncorrected one (§4.3).

## 5. Auditing the step by probing it, instead of reading it

From here on, the report is method. The central idea is to stop reading the code and start measuring
the function it computes.

One optimizer step is a function `Δtheta = Φ(g ; state)`. At any step that is *not* a multiple of
`TCov`, the state is frozen; and with `beta = 0` the function is **linear** in `g`. A linear function
can be extracted exactly: feed it the basis vector `e_i` as the gradient, read off column `i`.

This is cheaper than it sounds: setting `p.grad` by hand and calling `step()` needs no forward and no
backward pass through the model. So for a small model — `mlp_ln_mnist` has 26 634 parameters — the
entire matrix the optimizer actually applies, call it `M`, can be recovered column by column. Then,
with no appeal to what the code looks like:

- **Is `M` symmetric and positive definite?** It must be. A failure means the plumbing is wrong.
- **Is `M` block-diagonal by layer?** `∂Δtheta_i/∂g_j` must vanish across layers. This is the general
  form of the lot-8 bug, where a ViT's classification head was never updated at all while the loss
  still decreased.
- **Does `M⁻¹` equal `A_stored ⊗ B_stored + lambda I`,** with the factors read out of the optimizer's
  own state and the Kronecker product formed independently? This single comparison covers S4-S7 at
  once, and catches the whole family of errors the current tests cannot see: an overall scale factor
  (§4), an `rvec`/`cvec` transposition, a misplaced `pi`.
- **Where does `lambda` sit in the spectrum of `M⁻¹`?** Report, per layer and per model, the
  percentile of `lambda` among the eigenvalues. This distinguishes "a preconditioner" from "SGD with
  extra steps", and it is the direct answer to the second half of the supervisor's request. §4.4
  predicts that for `diag` it sits at the very top; that prediction should be confirmed by measurement
  rather than by arithmetic alone.
- **How good an approximation is it?** The exact dense Fisher already exists here for small models
  ([fisher_ref/reference/dense.py](../../fisher_ref/reference/dense.py)). Compute the angle between
  `M m̂` and `(F + lambda I)⁻¹ m̂`. If that angle is large, the update is not a natural-gradient step in
  any useful sense, whatever the mode is called.

Two stages sit upstream of what probing can see:

- **Are `h` and `delta` the right tensors?** Compare the hooks' captures against the layer input
  recomputed independently and against the exact autograd gradient. The machinery exists in
  [fisher_ref/capture.py](../../fisher_ref/capture.py).
- **Do the backward hooks always fire?** `optimizer.py` uses `register_full_backward_hook`, precisely
  the hook `fisher_ref/capture.py` deliberately refuses to use because it can be pruned out of the
  backward graph silently — measured at 25 120 of 26 634 parameters invisible on `mlp_ln_mnist` in the
  configuration where it triggers. In an ordinary training loop it should fire for every module, but
  this has never been *verified*. The check is a counter per module, asserted equal to the module count
  at every `TCov` step, on all eight benchmark models. It costs nothing, and the failure mode is not a
  crash but a silently zero factor.

## 6. Oracles: has any mode ever been shown to do the right thing?

No current test checks that the preconditioner *solves* anything. Three exact, deterministic oracles
would establish that baseline:

1. **One-step Newton.** A linear least-squares problem, one `Linear` layer, and a **single** example.
   The empirical Fisher block is then exactly `h̄h̄ᵀ ⊗ deltadeltaᵀ` — the factorisation is exact, not an
   approximation, because there is nothing to average over. With `lr = 1`, `lambda -> 0`, `beta = 0`,
   `TCov = 1`, K-FAC and EKFAC must reach the minimum in about one step. If they do not, the plumbing
   is wrong and every dominance and trace test has been passing in spite of it.
2. **A quadratic with a prescribed Hessian.** Check the convergence rate, and check that it is
   *independent of the conditioning* of that Hessian. Conditioning-independence is the defining
   property of a working preconditioner, and no current test examines it.
3. **Is the step even a descent direction?** On a real run, count the fraction of steps with
   `<F̃⁻¹ m̂, g> <= 0`, per mode and per layer. With `F̃` positive definite and `m̂ ~= g` this should be
   essentially zero. Separately: with `lr` small enough the loss must decrease monotonically. Two
   counters, no cost, and a non-zero result would be decisive.

Test-design note: use layers whose input and output dimensions **differ**. A transposition between the
`rvec` and `cvec` conventions — flagged in `CLAUDE.md` as the central pitfall of the project — is
invisible on a square layer.

## 7. Placebo controls: is the curvature doing any work?

This is, in my judgement, the most convincing single family of experiments for the question actually
asked. "Can the framework be trusted" reduces, operationally, to "does the Fisher information change
the outcome". The way to answer that is to replace `F̃` with something of the same shape carrying no
information, and see whether the results move. Given §4.4, `diag`'s answer may well be *no* — which is
exactly why this must be measured rather than argued.

| Placebo | What it isolates |
|---|---|
| `F̃ = I` | Reduces to momentum SGD with bias correction. The floor every mode must beat. |
| `F̃` frozen at step 0 | Does the *evolving* curvature matter, or only the initial rescaling? |
| `F̃` from a different seed, or from a 100x stale checkpoint | Sensitivity to where in parameter space we are — the drift campaign's question, turned into a trust test. |
| Same eigenvalues, **random orthogonal basis** | Is the information in the *directions*, or only in the overall scale? |
| Same basis, **eigenvalues shuffled** | The converse. |
| `diag` with its normalised diagonals replaced by constants | Given §4.4, is anything left of min-max beyond one scalar per layer? |

If a mode matches its own score under several of these, that is a publishable result as it stands, and
an honest one. It is also the only protocol that separates "AdaFisher works" from "AdaFisher is Adam
with a different effective learning rate".

## 8. Units, scale, and the missing square root

Adam divides by `sqrt(v)`; AdaFisher divides by `F̃`. The paper defends this at length and reports an
ablation in its favour, so it is a design choice, not a transcription issue. But it changes how the
optimizer responds to rescaling the loss. Multiply the loss by `c`:

- gradients scale by `c`, so Adam's `sqrt(v)` scales by `c`, and `g / sqrt(v)` is **unchanged**. Adam
  is invariant to the scale of the loss.
- gradients scale by `c` and `F̃`, built from products of two gradient-like quantities, scales by `c²`.
  So the update scales by `c / c² = 1/c`. **AdaFisher's step size is inversely proportional to the
  scale of the loss.**

Two consequences, both affecting results this repository has already produced:

- `lr = 1e-3` for AdaFisher is not comparable to `lr = 1e-3` for Adam, and is not even comparable
  between two models whose losses have different natural scales — a mean-squared error averaged over
  784 pixels versus a cross-entropy, say. The seven-arm comparison tables inherit this.
- **The test:** re-run one benchmark with the loss multiplied by 10 and by 0.1 (equivalently,
  `reduction='sum'` versus `'mean'`). Prediction: `adam` and `adamw` bit-identical across the three
  runs; the five Fisher modes shifted as though `lr` had been multiplied by `c²`.

Other invariances worth checking in the same pass — each one a property a curvature-aware method is
supposed to have and a first-order method is not: rescaling a layer pair (`W1 -> cW1`, `W2 -> W2/c`
leaves a ReLU network's function unchanged); permuting hidden units; changing the batch size (note that
without the square root, the *noise* in `F̃` enters the update linearly rather than as its square root,
so batch-size sensitivity should be **stronger** than Adam's, not weaker).

**A proposal for fair comparison.** Report the *effective gain*
`eta_eff = lr · ||F̃⁻¹ m̂|| / ||m̂||`, per layer, and match arms on `eta_eff` rather than on `lr`.
Comparing AdaFisher and Adam "at the same learning rate" compares two steps of arbitrarily different
sizes; that is not a comparison of preconditioners.

## 9. Justifying the constants

The deliverable for the second half of the request is this table, with the last column filled in.
Today that column is empty for every row.

| Constant | Value | Where it comes from | What it actually controls | The measurement that would justify it |
|---|---|---|---|---|
| `gammas` | `(0.92, 0.008)` | **not the paper.** Eq. (3) is `γ·old + (1−γ)·new` with `γ = 0.8`; this is `γ/10·old + γ/100·new` (§4) | the memory of the estimate **and its scale** | fixed point `n/115`; 92% of the estimate is one minibatch. Both are settled — what remains is §4.7's comparison against the published rule |
| `TCov` | 100 | **not the paper.** Algorithm 1 recomputes every step | how stale the curvature estimate is allowed to get | drift of `F` versus step separation; pick `TCov` from a tolerance. The drift campaign is already computing this |
| `Lambda` | `1e-3` | the paper (Eq. 4, Algorithm 1) | the floor of the spectrum, hence the **ceiling on the step size** (`lr/lambda`) | the percentile of `lambda` within each layer's spectrum (§5). §4.4 predicts it is at the top for `diag`. The existing sweep's null result is a fact to be explained, not a clearance — §4.6 point 3 |
| `lr` | `1e-3` | the paper (`α = 0.001`) | step size | a bracket from `1e-4` to `1e-1`, but only **after** §4.7 and §8, since until then the swept quantity is not well defined |
| `beta` | 0.9 | Adam, via the paper | smoothing of `m` | `m` is refreshed every step and `F̃` every 100; measure `cos(F̃_t⁻¹ m̂, F̃_{t+100}⁻¹ m̂)` to price that mismatch |
| `T_inv`, `T_eig`, `T_re` | 100, 100, 1 | K-FAC / TEKFAC | staleness of the inverse or eigenbasis, separately from the factors | degradation of the angle against a fresh `F⁻¹` as a function of staleness |
| min-max, and **where** | on, *before* the average | the function is Eq. (4); the placement is the official code's, against Algorithm 1 (§2) | not a tuning knob: it **changes the class of algorithm**, and its placement decides whether §4's scale error is erased or dominant | §4.7's 2x2. Should be reported as distinct arms, never as a default |
| `conv_sua` | off | lot 6 | memory on convolutions | already measured: 80.7x reduction, discarded off-block mass quantified |

Alongside the table, the most legible way to present "why these choices" is an **ablation ladder** from
Adam to AdaFisher, changing exactly one thing per rung:

```
SGD -> SGD+momentum -> Adam -> Adam without the square root
    -> the same, with v = diag(A) (x) diag(B) instead of g^2
    -> + min-max normalisation (after the average, per Algorithm 1)
    -> + min-max before the average (the official code's order)
    -> + the implemented time-average instead of Eq. (3)
    -> K-FAC -> EKFAC
```

The rung at which behaviour breaks *is* the explanation. This is also the cheapest way to answer
"which part of AdaFisher does the work", which is the first question a reviewer will ask.

## 10. External confidence: two things nobody has done

1. **Run the authors' own code on the failing benchmark.** Point `FisherAdapTune` (or the official
   repository) at the MNIST autoencoder. If it stalls at 0.7085 too, the stall belongs to AdaFisher as
   implemented and not to this port — which, given §4, is the expected outcome and worth having on
   record. If it does not stall, there is a porting bug that bit-exactness did not catch, which is
   entirely possible: the bit-exact test covers `diag` only, with min-max **off**, comparing `f_tilde`
   construction with the model resynchronised after every step. It covers neither a free-running
   trajectory nor any of the four Kronecker modes.
2. **Reproduce one published number.** One row of the paper's CIFAR-10 table. Until some published
   number is reproduced with this code, no negative conclusion about the modes is defensible — and
   note that §4 makes this more interesting, not less: the published numbers were produced *by the
   implemented rule*, so reproducing them is a test of the port, while the corrected rule of §4.7 is a
   different algorithm that may do better or worse.

There is also a question of principle the existing tooling can already answer: **all modes use the
empirical Fisher**, built from the gradient at the true label. Its known failure mode is that it is
largest precisely along the mean gradient direction, so `F_emp⁻¹ g` shrinks the one direction one wants
to move in. That is a candidate mechanism for a stall, and it is directly measurable by comparing the
empirical and the "type-2" Fisher at the stall point — both are implemented in
[fisher_ref/sources.py](../../fisher_ref/sources.py).

## 11. Suggested order, and what it costs

| # | Action | Cost | Why here |
|---|---|---|---|
| 1 | Spectrum of the stored factors and where `lambda` falls in it, per layer, at existing checkpoints, all five modes | laptop, hours | confirms §4.4 by measurement and explains the `lambda` sweep's asymmetry |
| 2 | §4.7's variants: Eq. (3)'s average, Algorithm 1's min-max order, the 2x2, `TCov = 1` | one benchmark, a handful of arms | the actual finding, turned into numbers |
| 3 | Extract `M` by probing; compare against `A ⊗ B + lambda I` and against the exact dense `F` | laptop | audits S4-S7 exactly, without reading code |
| 4 | One-step Newton and the prescribed-Hessian quadratic | laptop, minutes | correctness, currently untested |
| 5 | Loss-scale test (`c = 0.1, 1, 10`) | one benchmark, seven arms | validates or invalidates every `lr`-based comparison so far |
| 6 | Placebo controls (identity, frozen, randomised basis) | one benchmark, ~four placebos | "does the Fisher do anything" |
| 7 | The authors' own code on the same benchmark | hours | attributes the stall |
| 8 | Permanent assertions: hook-firing counters, descent-direction counter | negligible | cheap, and they guard against silent regressions |
| 9 | Learning-rate bracket, and re-tuning `γ` on the paper's own grid under the corrected rule | cluster | only after 2 and 5 |

Items 1, 3, 4 and 8 are **measurements on existing checkpoints and on a frozen optimizer**, not
training runs: no GPU hours, no new campaign. Items 1 and 2 together are probably enough to answer the
substance of the supervisor's question.

## 12. Consequences for the repository's own record

Three things this repository currently states should be revised. **No file has been edited** — this
list is the proposal.

1. [plan.md §1.4](archives/plan.md) frames the averaging rule as a *divergence between the official
   repository and FisherAdapTune*, arbitrated in favour of the latter. The two are numerically
   identical on this point (§4.2); the real discrepancy is with the paper's Eq. (3), and it is shared.
2. The `lambda` sweep is recorded as having **exonerated** `lambda`. It compared final losses on a
   benchmark where every arm had already collapsed (§4.6 point 3), so the null result is uninformative
   about the optimizer. The claim should be weakened to what was actually measured.
3. The min-max placement is recorded as following "the official repository, not after accumulation",
   as a fidelity choice. It is also a **deviation from Algorithm 1**, and per §4.4 it is the deviation
   that turns the scale error from harmless into dominant. It should be recorded as both.

## 13. Summary for the supervisor

The existing tests prove that this code is a faithful port of AdaFisher's *implementation*; they were
never designed to test it against AdaFisher's *published algorithm*, and they cannot see an error in
the overall scale of the preconditioner. Auditing the seven stages of one step against the paper found
one, at stage four: the published time-average is `γ·old + (1−γ)·new` with `γ = 0.8`, whereas both
reference repositories compute `γ/10·old + γ/100·new`, i.e. `0.08·old + 0.008·new`. Two consequences
follow arithmetically — each stored curvature factor settles at about 1/115 of the quantity it
estimates (and the preconditioner at about 1/13 000), and 92% of the estimate is a single minibatch
rather than an average over many. The paper's own Algorithm 1 would have hidden the first of these,
because it normalises *after* averaging and normalisation destroys scale; the implementation normalises
*before*, so instead the damping constant `lambda = 1e-3` ends up thirteen times larger than the entire
curvature term, and `diag` reduces to momentum SGD at an effective learning rate of 1.0. That single
mechanism quantitatively predicts three things already observed in this repository and previously
unexplained: `diag`'s unusually large early steps, the fact that `diag` is the only mode that responds
to `lambda`, and the fact that it responds only at the smallest value tested. Sections 5 to 11 lay out
the rest of the audit — probing the optimizer to recover exactly the matrix it applies, exact oracles
on problems whose answer is known, placebo preconditioners to test whether the curvature does any work
at all, and a table that pairs every constant with the measurement that would justify it.

