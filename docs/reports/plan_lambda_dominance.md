# The safety constant is running the optimizer, not the curvature

*Written to be read by someone who has not followed the investigation. Plain words, short
sentences, every term explained the first time it appears. Every number here was measured; where
something was worked out on paper instead, it says so.*

---

## Short version

**What these optimizers are supposed to do.** Before taking a step, look at the shape of the loss
around the current point. Where it bends sharply, take a small step. Where it is flat, take a big
one. This is done by dividing the step by a measure of how sharply the loss bends, direction by
direction. That measure is called the curvature.

**The problem.** Dividing by curvature goes wrong where the curvature is zero, so a small safety
constant is added first. It is called `λ` and it is 0.001 by default. We measured how big the
curvature estimate actually is, and `λ` turned out to be **bigger than it in every direction, in
every layer, of every network, in every mode**. So the division is the same everywhere. It no
longer depends on direction. All five methods are doing plain gradient descent with momentum and a
fixed step size. The curvature is computed at real cost at every step, and then the arithmetic
throws it away.

**Why that happens.** Two conventions inherited from the original code each shrink the stored
curvature, and they multiply together.

| What it is | How much it shrinks | Where in the code |
|---|---|---|
| The running average is not really an average. It does `new = 0.08 × old + 0.008 × measurement`. Those two numbers should add up to 1 and they add up to 0.088, so the stored value settles at 1/115 of what it is measuring. | 115 per stored quantity. Four of the five modes store two of them, so 13 225. | [ema.py:19-21](../../src/adafisher_modes/ema.py#L19-L21) |
| The signal used is the gradient of the loss *averaged* over the batch, so each example counts for 1/128 of itself. Curvature is built by squaring that signal, so each example's share shrinks by 128². | 16 384 at batch 128 | [optimizer.py:138-142](../../src/adafisher_modes/optimizer.py#L138-L142) |

Together: about 200 million. Against a `λ` of 0.001, the curvature never gets a vote. This is not a
bug in this project. It is a faithful copy of both reference implementations.

**What the four measurements found.** All four have been run. Details are in Part 6.

| | Question | Answer |
|---|---|---|
| **E0** | Is the noisy validation curve caused by the curvature estimate? | **No.** An optimizer that computes no curvature at all, but takes the same size of step, makes the same amount of noise. Checked on 30 network-and-mode pairs; the average difference is 4%. |
| **E1** | Where does `λ` sit among the curvature values? | **Above all of them**, in all 105 cases measured. Undo both shrinkages on paper and it is still above 93% to 100% of directions on every network, for four of the five modes. |
| **E3** | Does the batch size move the ratio, the way the arithmetic says? | **Yes.** 56 measurements of the "shrinks as 1/batch²" rule, all between 0.021 and 0.144 where the rule predicts 0.0625. But no batch size rescues the optimizer. |
| **E2** | What happens if `λ` really is made small? | An earlier study said the optimizer breaks. That was an artifact: lowering `λ` also enlarges the step, and the earlier study did not compensate. Compensate and **75 runs out of 75 train fine**, against 19 out of 75 before. But it **buys almost nothing**: the best gain anywhere in 75 runs is 2.05 accuracy points. |

**So the headline has changed.** `λ` really does dominate, for the arithmetic reasons above. The
dominance really can be removed. But removing it does not make these five methods better
optimizers. They are not being held back by the safety constant.

| **E4** | Is it the curvature that does not help, or the way it is being estimated? | **Not the estimating.** A real average, started from a real observation, gives the same result at the same effective damping: −0.37 points on average over the four Kronecker modes. But at batch 32 the study's one real gain shows up: **+4.32 accuracy points** on `cnn_gn_cifar`/`kfac`, at a safety constant of 10⁻⁸. |

**Where that left it, after E4.** The curvature does not pay for itself at the default operating
point, and fixing the estimator does not change that. But there *is* an operating point where it
pays, far below the default, and reaching it by hand per network is not a method.

**Where it stands now, after E14.** E7 to E14 swept `λ` further down, with the step size held still,
with the eigenbasis ordering corrected and with five seeds. That operating point is real and large:
**+6.6 to +10.0 accuracy points** for `ekfac`/`tekfac` on three of the four networks measured
(`cct_2_3x2_cifar`, `vit_micro_cifar`, `cnn_gn_cifar`), at `λ` between 1e-11 and 1e-10, i.e. 7 to 8
orders of magnitude below the default. E2's "buys almost nothing" was wrong: its window stopped too high.

The plan had been to reach that point without hand-tuning through **S1**: `λ` proportional to the
curvature, with one dimensionless constant `τ` that would transfer between networks. **That part
has lost its evidence.** Across four networks whose mean curvature spans 15 to 20 times, the best
`λ` for `ekfac`/`tekfac` moves by only 3 to 10 times, and a fixed number predicts it as well as the
proportional rule, or better (E14). What survives is narrower. First, a relative `λ` is still the
only form that is automatically right when the batch size or the running average changes. Second,
the *per-layer* version of S1, one `λ` per layer inside a network, had never been run: every
experiment until then used one `λ` for the whole network.

**E15 ran it, and it wins.** With `λ_l = τ × (mean curvature of layer l)` and the step-size cap held
in every layer, S1 per layer beats the best single `λ` in all six (network, mode) pairs tested, by
+1.9 to +6.1 points. It also beats a relative `λ` shared by the whole network, so the gain comes from
treating layers differently. Against the default `λ` it gains **+8.2 to +14.8 points**. For
`ekfac`/`tekfac` one value, `τ = 0.1`, is in the plateau on both networks. It mostly does two
things: it stops shrinking the classification head's steps, and it lets the curvature act in the
flattest layers. Two small networks at batch 32 so far. Part 6, E15, has the numbers.

---

## The findings, one by one

*The same story as the table above, told straight through. If you read only one section, read this
one.*

**1. The safety constant crushes everything else.** We measured, on the seven networks in this
project, layer by layer and direction by direction: in **105 cases out of 105**, `λ` is bigger than
the curvature estimate in **every** direction. Not "in most of them". In all of them. So the
division comes out the same everywhere, the adaptation does not exist, and all five methods reduce
to ordinary gradient descent with momentum and a fixed step size.

**2. Why.** Two conventions inherited from the original code shrink the curvature estimate, and they
multiply. The first: the running average is not one. Its two coefficients should add up to 1 and add
up to 0.088, so the stored value settles at 1/115 of what it measures. The second: the signal used is
the gradient of the loss *averaged* over the batch, so each example counts for 1/128 of itself, and
since curvature is built by squaring that signal, that is 1/16 000. Together: about 200 million times
too small. This is not a bug in this project — it is faithful to both reference implementations.

**3. The noise that started the investigation does not come from the curvature.** The earlier
investigation was looking for why these methods' validation curves jump around more than Adam's. We
compared each method against a twin that computes no curvature at all but takes the same size of
step. Over 30 comparisons, the twin makes **exactly the same amount of noise** (4% average
difference). So the noise comes from how the step is sized, not from the curvature.

**4. Correcting the size is not enough.** We recomputed where `λ` would sit if both shrinkages were
undone. On the smallest network, `λ` would land in the middle — which is what the paper wants. But
on the six others it would still be **above 93 to 100% of directions**. So fixing the conventions
does not make any of the five methods adaptive, except marginally on the smallest network.

**5. The batch size does move the ratio, but rescues nothing.** Since the stored curvature is
proportional to 1/batch², going from 128 down to 32 should multiply it by 16. Checked: 56
measurements, all consistent. But even at batch 32, `λ` is still above everything. One useful detail
in passing: small batch is the only place — and only there — where one of the modes (`diag`) becomes
genuinely adaptive once the size is corrected.

**6. An earlier experiment was misread.** An earlier study pushed `λ` down to 10⁻⁸, watched the
networks fall apart, and concluded that `λ` cannot be lowered. But lowering `λ` mechanically enlarges
the step by the same factor: the experiment was changing two things at once. We redid it,
compensating the learning rate so the step size stays identical. Result: **19 runs out of 75 survive
under the old protocol, 75 out of 75 under the new one.** The collapse was entirely an exploding
step, nothing to do with curvature.

**7. And here is the result that counts: lifting the barrier buys almost nothing.** With the right
protocol we lowered `λ` across nine orders of magnitude. The best gain found in **75 runs is +2.05
accuracy points**, on one network. Nine of the fifteen network-and-mode pairs do have *some* `λ` that
beats their default, but by a few tenths of a point. Everywhere else it is flat, and then it collapses
sharply.

In other words: **`λ` really does dominate, we now know how to remove it properly, and removing it
does not make these methods better.** They are not being held back by the safety constant.

**8. Two internal agreements that make the measurements trustworthy.** The point where the behaviour
starts to change (between 10⁻⁸ and 10⁻¹⁰) is exactly the one predicted by the measurement of the
directions. And the order in which the methods break follows exactly their ranking: `diag`, the least
damped, gains the most **and** breaks first; `kfac`, the most damped, is the most robust everywhere.

### What is still open

*Updated after E14. The question this section first asked, whether the averaging is to blame, was
answered by E4: it is not.*

1. **Does one `λ` per layer beat the best single `λ`?** Yes, on the two networks E15 measured, in
   all six (network, mode) pairs, and the gain is per-layer (E15). Open: whether `τ = 0.1` transfers
   to a third network for `ekfac`/`tekfac`, and whether it holds on `resnet20_cifar` and at another
   batch size.
2. **Does a fixed `λ` survive a change of batch size once S4 is applied?** Every run from E8 to E14
   is at batch 32, so "fixed" has only been shown at one batch size.
3. **How much of the E10 and E13 gains is weight decay switching off?** Answered by E15's control:
   almost none. With the decay held at its fixed rate, E13's cells move by −0.08 to −0.23 points,
   none more than 1.6 standard errors from zero.

### Two measuring traps we walked into

Both are worth remembering, because each one nearly produced a false result.

The first: summarizing these lists of numbers by their middle value gives nonsense, because many
directions have exactly zero curvature. On MNIST, 130 pixels are always black, which kills 136
directions out of 785 in the first layer. The middle value was measuring that floor of zeros, not the
curvature.

The second, and this one is prettier: for two of the five methods, one number kept coming back
**identical on networks with nothing in common**. It was the leftover of the optimizer's starting
value, which fades by a factor 0.08 at each update — after 9 updates, 1.34 × 10⁻¹⁰, which matched
what we were measuring to three digits, **and across two different values of `λ`**, which rules out
coincidence. The measurement was not measuring curvature, it was measuring the optimizer's start-up.
Fixed by doubling the warm-up: the symptom disappeared, and its disappearance revealed point 5's rule
for those two methods, which had been invisible until then.

---

## Part 1 — What was already known

Sources: [validation_noise_investigation.md](validation_noise_investigation.md), steps 7 to 16, and
[audit_step.md](audit_step.md), section 4.

### 1.1 The words, once

- **Gradient**: the direction in which the loss drops fastest, for the current batch of examples.
  Every optimizer starts from it.
- **Momentum**: a running average of recent gradients, used instead of the raw gradient so the path
  is smoother. Written `m`.
- **Curvature**: how fast the gradient itself changes as you move. Big curvature means the loss
  bends sharply here, so you should step carefully. The whole promise of this family of methods is
  to divide the step by the curvature, separately in each direction.
- **The curvature estimate**: what the optimizer actually builds and uses. It is not the true
  curvature. It is an approximation, averaged over time, with `λ` added, and in one of the five
  modes rescaled as well.
- **`λ`, the safety constant**: a small number added to the curvature estimate before dividing, so
  that a direction with almost no curvature does not produce an enormous step. It is meant to be a
  floor under the estimate. Here it is 0.001, or 0.003 on the two Transformers.
- **Directions, and the list of curvature values**: a network with 26 000 parameters has 26 000
  independent directions to move in. The curvature estimate gives one number per direction: how
  sharply the loss bends along it. We will talk about this list a lot.
- **Percentile**: where a number sits in a sorted list. `λ` at the 50th percentile means half the
  directions have more curvature than `λ` and half have less. `λ` at the 100th percentile means it
  is above all of them.
- **The update, written out**: `parameter ← parameter − learning rate × m / (curvature + λ)`, with
  the division done separately in each direction.
- **The step-size cap**: when the curvature is negligible, the division is just by `λ`, so the step
  is `learning rate / λ` times the momentum. With the defaults that is 0.001 / 0.001 = 1. The step
  *is* the momentum, unscaled. That is momentum SGD at a learning rate of 1, not of 0.001.

### 1.2 Where the shrinkage comes from

**The running average.** Every time the curvature is re-estimated, which is every 100 steps by
default, the code does:

```
stored ← 0.08 × stored + 0.008 × new measurement
```

A real running average has coefficients that add up to 1. Feed it the same number over and over and
it gives you that number back. These add up to 0.088. Feed this one a constant `n` and it settles
at `0.008 / (1 − 0.08) × n`, which is `n / 115`. Two things follow:

- **The size is wrong.** Every stored quantity is about 115 times smaller than what it measures.
  Four of the five modes multiply two of them together, so their curvature estimate is about 13 225
  times too small.
- **There is no averaging left.** 92% of the stored value is the single most recent measurement.
  That is the opposite of what an average is for.

This does not match the AdaFisher paper, whose equation (3) says `γ × old + (1−γ) × new` with
`γ = 0.8`. And it is the same in *both* reference implementations, so it is a property of AdaFisher
as published in code, not a mistake made while porting it here (audit_step, section 4.2).

**The batch average.** The hook that collects the backward signal collects the gradient of the loss
*averaged over the batch*. So each example contributes 1/128 of its own gradient. Curvature is built
by squaring that signal, so each example's contribution to the curvature is divided by
128² = 16 384 compared with the textbook quantity. The K-FAC reference implementation cancels this
by multiplying the backward signal by the batch size. This project deliberately follows AdaFisher's
convention instead.

**Why one mode is different.** In `diag` mode each layer's numbers are squeezed into the range 0 to
1 before the running average. Squeezing destroys the overall size, so it cancels the batch factor.
But it happens *before* the average, so the factor of 115 survives. The result is a ceiling that
does not depend on the data at all: `(1/115)² = 0.000076`, which is 7.6% of `λ`. The two biggest
values ever measured are 6.7% and 7.0%. The arithmetic and the measurement agree.

### 1.3 What had been measured before this plan

- **Every layer of seven networks, all five modes** (steps 7 and 10): the single most curved
  direction found anywhere reaches 7% of `λ` for `diag`, and under 0.2% of `λ` for the other four.
  There is no hidden pocket of sharply curved directions.
- **A stand-in test** (steps 9, 11, 12, 14): a plain momentum optimizer with no curvature estimate
  at all, given a step size matched to what `λ` imposes, was run against the real modes on 30
  network-and-mode pairs. 11 match closely, 12 are a few percent off, 2 clearly differ.
- **What was left over** (step 16): `kfac` is the only mode whose divisor genuinely varies from one
  direction to another early in training, by a factor of 8 to 16 over the first 200 steps or so.
  That lines up with where it differs from the stand-in. `tkfac`'s own case shows the opposite
  pattern and is still unexplained.
- **A start-up transient** (step 8): every mode starts its curvature estimate at the identity
  matrix. That starting value fades by a factor 0.08 at each re-estimation, so for the first few
  hundred steps the divisor is neither the curvature nor `λ` but the leftover of that start. This is
  why an earlier sweep of `λ` across four orders of magnitude changed almost nothing.
- **A correction, already tried** (audit_step, section 4.8, job 21274380): the paper's own averaging
  rule, the paper's own ordering of the rescaling step, and re-estimating every step instead of
  every 100. Six configurations. None changed the final loss on the main benchmark. One new effect
  appeared: the corrected average *freezes* `kfac`, because the starting value now fades by 0.8
  instead of 0.08 and needs about 9 300 steps to disappear instead of 1 000.

### 1.4 What was not explained

1. Where `λ` *should* sit. Everything measured so far reports the largest curvature value next to
   `λ`. Nobody had reported the whole list. Without that, no fix has a target.
2. Why `tkfac` differs from its stand-in on one particular network.
3. Why `resnet20_cifar` sits a few percent off its stand-in in *all five* modes.
4. **The original question**: why the validation curves jump around more than Adam's, while the
   training curves do not. Nothing since step 4 had touched it.

---

## Part 2 — Three things the earlier record did not say

### 2.1 Making `λ` smaller and making the curvature bigger are the same move

Suppose we multiply both stored quantities by a constant `c`, to put them back near their true size.
For each of the five modes, the divisor you end up with is exactly what you would get by leaving the
stored quantities alone, dividing `λ` by `c²`, and then dividing the whole step by `c²`:

| mode | the divisor | after multiplying the stored quantities by `c` |
|---|---|---|
| `diag` | `H⊗S + λ` | `c²(H⊗S) + λ`, which is `c²` times `[H⊗S + λ/c²]` |
| `kfac` | `(A + π√λ)⊗(B + √λ/π)` | `π` is a ratio of traces and does not change under a common rescaling, so this is `c²` times `[(A + π√(λ/c²))⊗(B + √(λ/c²)/π)]` |
| `tkfac` | `δ (Φ + √(λ/δ))⊗(Ψ + √(λ/δ))` | `Φ` and `Ψ` have a fixed total of 1 and `δ` becomes `c²δ`, so the added term becomes `√(λ/c²δ)` |
| `ekfac`, `tekfac` | `Θ + λ` | `c²Θ + λ`, which is `c²` times `[Θ + λ/c²]` |

**What that means for an experiment already done.** The earlier study that pushed `λ` from 0.001
down to 10⁻⁸ and watched the networks fall apart was not asking "what if the curvature mattered". It
was asking "what if the curvature mattered *and* every step were 100 000 times bigger". The
networks falling apart is what a step size exploding looks like. **The experiment has to be redone
with the step size held still**, which means moving `λ` and the learning rate together.

Two useful side effects. First, you do **not** need to change the running average or the batch
convention to ask the scientific question: moving `λ` and the learning rate together is the same
thing on paper and touches no optimizer code. Second, the equivalence is only exact once the
starting value each mode begins from has faded. One more reason to get rid of it (fix S3).

### 2.2 The correction already tried removes 13 225 of a 200 000 000 gap

Fixing the running average leaves the batch factor untouched. And the size of that first correction
is **not the same for all five modes**, which is easy to get wrong. The factor of 115 applies once
per *stored quantity*, and the modes do not store the same number of them.

| mode | what it stores | correction to the running average |
|---|---|---|
| `diag` | `H` and `S` | 115² = 13 225 |
| `kfac` | `A` and `B` | 115² = 13 225 |
| `ekfac` | `s*` only. The basis it is written in has no size of its own. | 115 |
| `tkfac` | `Ψ_raw ⊗ Φ_raw / δ`: three stored quantities, two on top and one underneath | 115 |
| `tekfac` | `Θ` only | 115 |

Apply that, and the batch factor where it survives, to the largest curvature value measured on
`mlp_ln_mnist`. The last column was later confirmed by measurement, in Part 6.

| mode | as it ships | running average fixed | both fixed |
|---|---|---|---|
| `diag` | 0.067 | about 880, the ceiling `[0,1]` divided by `λ` | the same. Squeezing into 0 to 1 already cancels the batch factor. |
| `kfac` | 0.000017 | 0.22 | about 3 600 |
| `ekfac` | 0.00034 | 0.039 | about 640 |
| `tkfac` | 0.0011 | 0.13 | about 2 100 |
| `tekfac` | 0.00047 | 0.054 | about 900 |

Read it two ways. **Fixing the average is enough for `diag`**: it puts back exactly the
thousand-to-one range of step sizes the paper intends. That predicts, and matches, what audit_step
section 4.8 measured — the corrected average visibly changed `diag`'s early behaviour without
changing where it ended up. **It is not enough for the other four**: even the largest curvature
value in the whole network stays at a fifth of `λ` or below, so every direction is still dominated
by the safety constant.

And the spread in the last column says something on its own. 640 for `ekfac` against 3 600 for
`kfac` is a factor of 5.6. **One shared numerical value of `λ` cannot be right for all five modes at
once**, because they store things on different scales.

### 2.3 There is a thousand-to-one split inside every network

[`_step_fallback`](../../src/adafisher_modes/optimizer.py#L209-L216) handles every parameter that
does not belong to one of the four layer types the optimizer hooks into. It does no division at all.
Those parameters get plain momentum. Parameters that *are* hooked get divided by something that
measurement says is essentially `λ`. So inside one network:

- hooked parameters move by `learning rate / λ` times the momentum, which is 1000 times the
  momentum. On the two Transformers, where `λ = 0.003`, it is 333 times.
- unhooked parameters move by `learning rate` times the momentum, which is 0.001 times it.

A thousand to one, between parts of the same network. Affected here: `cnn_gn_cifar`'s GroupNorm
layers, because the optimizer does not handle GroupNorm at all; every Vision Transformer's class
token and position table; and any layer type not on the list.

This is faithful to the reference implementation and is not a porting mistake. But it is the missing
number between two things already known: that GroupNorm gets no curvature treatment (step 6), and
that making the stand-in optimizer respect the same boundary improved its agreement with real
`kfac` by a factor of five (step 12). It is also a candidate for the original question. A network
whose normalization scales barely move while its weights move a thousand times faster is being
trained by neither Adam nor SGD, and that is a plausible source of jumpy validation numbers.

---

## Part 3 — The experiments

Ordered by how much they tell you for what they cost. E0 to E3 have been run; their results are in
Part 6.

### E0 — Re-read the runs we already have, using the metric we started from

**Cost: nothing.** `fisher_ref/outputs/warmup_sgd_baseline_small.json` already holds, for six
networks and five modes, the validation loss and accuracy at every epoch of three runs each: the
real mode at two random starting points, and the curvature-free stand-in at the first of them. The
earlier study read only the *final* number out of these files.

**What to compute.** How much the validation curve jumps from one epoch to the next, for the real
mode and for its stand-in, plus how much two runs of the same mode differ from each other by chance.

**What it settles.** If the stand-in jumps as much as the real mode, then the jumpy validation
curves this whole investigation was written to explain come from the size of the step, not from the
curvature. If the stand-in is calm where the real mode is jumpy, then the curvature *is* doing
something the size measurements cannot see, which would be far more interesting.

### E1 — Where does `λ` actually sit among the curvature values?

**The measurement this whole thread was missing.** Everything before it reports "the largest
curvature value, next to `λ`". What a safety floor actually needs is its **percentile**: out of all
the directions in this layer, what fraction has more curvature than `λ`? A floor at the 5th
percentile is doing its job. A floor at the 100th percentile *is* the optimizer.

**What to run.** Reload each mode's state from a mid-training checkpoint, let the running averages
fill back up, then for every layer report the whole list of curvature values, what fraction is above
`λ`, and the same again after multiplying by the two corrections we know about. There is a free
cross-check on `mlp_ln_mnist`: this repository can build the exact curvature matrix for that network
([fisher_ref/reference/dense.py](../../fisher_ref/reference/dense.py)), and it uses neither
convention.

**What it settles.** The value of `λ` that would put the floor at a chosen percentile, per mode and
per layer. That number is the target for fixes S1, S2 and S7. Without it they are guesses.

### E2 — Move `λ` with the step size held still

**The corrected version of the earlier sweep**, following Part 2.1. Move `λ` from 0.001 down to
10⁻¹², and at each point set the learning rate so that `learning rate / λ` — the cap on the step
size — never changes. Report, at each point, whether the network still trains and how well.

**What would prove it wrong.** If the earlier collapse was the step size exploding, it goes away
here, and the difference from the plain-momentum stand-in grows smoothly as `λ` falls. If the
networks still fall apart at a matched step size, then the problem is not the step size but the
estimate itself: either too noisy, because it is 92% one batch and is inverted without a square
root, or too full of dead directions, which is what makes two of the modes crash. Either answer
is useful.
Before this was run, neither could be ruled out.

### E3 — Change the batch size, which moves the same ratio for free

Under the current convention the stored curvature is proportional to 1/batch². So going from 128
down to 32 multiplies it by 16, and going up to 512 divides it by 16, **without touching a line of
code**.

**What it settles.** A cheap check of Part 1.2 that could fail. If the fraction of directions above
`λ` does not move as 1/batch², the arithmetic is wrong somewhere and everything downstream needs
re-reading. It is also a result on its own: it would mean `λ = 0.001` has no meaning independent of
the batch size, which matters for every comparison table in this repository.

### E4 — Separate the two shrinkages

A two-by-two: {running average as it ships, real average with a start-up correction} crossed with
{batch-averaged backward signal, per-example backward signal}. Measure E1's percentile in each
square, **not** the final loss.

Two requirements. Use a benchmark that is not already degenerate — **not** `mnist_autoencoder`,
where audit_step section 4.8 showed all six configurations land on the same number and nothing can
be told apart. And pair it with fix S3, or the corrected average will freeze `kfac` again for the
reason section 4.8 found. E3 adds a third: run it at batch 32.

### E5 — Measure how big the step actually is, per group of parameters

Record `learning rate × ‖divisor⁻¹ m‖ / ‖m‖` per layer and per step, keeping hooked and unhooked
parameters apart, on all eight networks. It costs nothing; it is a few lines inside `step()`.

**What it settles.** It puts a number on section 2.3 for the real networks. It also gives the
quantity that arms should be matched on when comparing against Adam: comparing two optimizers "at
the same learning rate" compares two steps of completely different sizes. And it is the monitor that
would have caught the earlier study's silent collapses, where a run reported a number while having
stopped learning.

### E6 — Test the one surviving mechanism properly

Step 16 found that `kfac`'s divisor varies from direction to direction early on, exactly where it
differs from its stand-in. That is a coincidence in time, not a demonstrated cause. The test that
turns it into one is already designed in
[validation_noise_investigation.md](validation_noise_investigation.md): force `kfac`'s divisor to be
the same in every direction for steps 0 to 400 only, then run it unmodified afterwards, same
starting point and same order of examples. Any difference in the final result can then only come
from that window. Worth doing after E0 to E2, and less interesting if E0 already closes the noise
question.

---

## Part 4 — The fixes

None of these is invented here. The first is asked for by two of the five source papers and is
explicitly not implemented in this port.

### S1 — Make the safety constant relative to each layer

Replace the one shared number `λ` by `λ_l = τ × (average curvature of layer l)`, computed per layer.
This is exactly TEKFAC's equation 3.5, recorded as not implemented at
[tekfac.py:24-26](../../src/adafisher_modes/approximations/tekfac.py#L24-L26). TKFAC has its own
adaptive floor, equation 5.16, also not implemented
([tkfac.py:19-21](../../src/adafisher_modes/approximations/tkfac.py#L19-L21)).

**Why it is the deep fix.** A relative floor does not care about either shrinkage, or about how the
loss itself is scaled. The question "is the running average a bug or a convention" stops mattering,
because the answer no longer changes what the optimizer does. `τ` becomes the real knob, and E1
measures what it should be.

**Status after E14: demoted, not refuted.** S1 makes two claims. They have to be kept apart, because
the experiments tested only one of them.

- **Claim 1, across networks: one `τ` fits every network.** E7 to E14 tested a *consequence* of it:
  they swept one `λ` for the whole network and compared the best value with the network's pooled
  mean curvature. The consequence did not hold. For `ekfac`/`tekfac` the best `λ` stays between
  1e-11 and 1e-10 while the curvature varies 15 to 20 times, and dividing by the curvature does not
  make the optimum more constant (E14, Result 3). E13's refinement, "one `τ` per damping rule", also
  fails: on `cnn_gn_cifar`, `kfac` wants the same `λ` as `ekfac`. So the promise that motivated S1,
  setting `λ` from a measurement instead of a search, has no support left.
- **Claim 2, inside a network: each layer needs its own `λ`.** Never tested: every run so far used
  one `λ` for all layers. It remains plausible, because the curvature is very uneven across layers.
  Measured at the half-way checkpoint (`fisher_ref/outputs/curvature_max_per_layer.json`, the
  99th-percentile curvature of each layer), the largest layer's value is **230 to 5 000 times** the
  smallest's for `ekfac`/`tekfac`: 4 800 to 5 000 times on `cnn_gn_cifar`, 520 to 560 on
  `vit_micro_cifar`, 230 on `cct_2_3x2_cifar`, 560 to 1 100 on `resnet20_cifar`. On all four
  networks the largest is the **classification head**. Between the 10th and the 90th percentile of
  layers, the three deeper networks' layers sit within **2 to 7 times** of each other. So in
  practice S1 per layer mostly means treating the head differently from the rest.
- **The invariance argument is untouched.** Every run from E8 to E14 is at batch 32, with the same
  running average and the same loss. A `λ` that is "fixed" in stored units is fixed only at those
  settings: E3 measured that the stored curvature scales as 1/batch². E14 says that `λ` should not
  follow the curvature *of the network*. It says nothing about the global scale factors.

What replaces S1 as the leading candidate is **S4 plus one fixed `λ`** (see S4). The per-layer
version stays on the list as a direct experiment. It has a trap: with one learning rate for the
network, a per-layer `λ_l` also changes each layer's step-size cap `lr/λ_l`, and that breaks
Part 5's rule 1 unless the cap is held per layer.

**Status after E15: claim 2 supported, and S1 per layer is the leading fix again.** Tested directly,
with the cap held per layer (`damping="layer_relative"`, `hold_cap=True`), it beats the best single
`λ` in all six pairs, and a network-wide relative `λ` in all six too (E15). For `ekfac`/`tekfac`,
`τ = 0.1` is in the plateau on both networks, so claim 1 comes back in a narrower form: one `τ` per
damping rule, *per layer*. That is two networks; a held-out one (E17) is needed before calling it a constant.
S4 plus a fixed `λ` stays worth testing as the simpler alternative, but on these two networks it is
bounded by the single-`λ` arm, which S1 per layer beats.

### S2 — Stop the step size from depending on `λ` at all

Rescale the divided direction back to the length of the momentum it came from: multiply by
`‖m‖ / ‖divisor⁻¹ m‖`, per layer. Then only the *shape* of the division affects the update. Its
overall size is set by the learning rate alone.

**Why we planned to do this first.** It removes, by construction, the confusion that made the
earlier `λ` sweep unreadable. It makes a `λ` sweep a genuinely one-knob experiment. And it makes the
comparison against Adam fair, since Adam normalizes its own step and these modes do not. It is also
the smallest of the three changes. (In the event E2 did not need it — see Part 5.)

### S3 — Remove the start-up transient

Every mode starts its curvature estimate at the identity matrix, and that starting value then fades
geometrically. For the first few hundred steps the divisor is neither the curvature nor `λ` but the
leftover of that start. Two standard remedies: divide the running average by `1 − decay^k`, which is
exactly what Adam does to its own moments, or start from the first real measurement instead of the
identity.

**What it buys.** It removes the confusion that made steps 8, 11, 12 and 13 hard to read. It removes
the `kfac` freeze that audit_step section 4.8 found, where the corrected average needs 9 300 steps
for the start to fade. It makes the "never re-warm for less than 10 cycles" rule unnecessary. And it
makes Part 2.1's equivalence exact from step 0, which is what E2 relies on.

### S4 — Put the backward signal back at per-example size

Multiply the captured backward signal by the batch size, which is the K-FAC reference
implementation's own convention. Then the stored object really is the empirical Fisher matrix, and
`λ = 0.001` means what it means in that literature. **Mostly redundant with S1**, since a relative
floor already absorbs the size. So this is for making the constants comparable with published ones,
not for fixing behaviour.

**Status after E14: promoted.** With S1's cross-network claim unsupported, S4 plus one fixed `λ` is
the simplest fix consistent with the data. S4 removes the batch factor. The running average's factor
(115 for `ekfac`/`tekfac`) is a known constant and can be divided out explicitly (fix S3 or
`gamma`). What remains is one number. For `ekfac`/`tekfac`, E10 to E14's best stored values
(1e-11 to 1e-10 at batch 32) correspond to about **1.2e-6 to 1.2e-5** at the true scale. That is
worked out on paper as 115 × 32² ≈ 1.2e5 times the stored value, not measured. The candidate can be
refuted cheaply. Without S4, the best stored `λ` must fall 16 times when the batch goes from 32 to
128. With S4, it must stay put.

### S5 — Numerical safety when `λ` is small

Needed before E2 and E4. `kfac` and `tkfac` invert their quantities directly with
`torch.linalg.inv` and crash when one is exactly singular. The small ridge in
[_eigh_utils.py](../../src/adafisher_modes/approximations/_eigh_utils.py) exists only for the two
modes that take an eigendecomposition. Make it symmetric. And add the guard the earlier study did
not have: a monitor on the actual step size (E5) turns a silent collapse to chance-level accuracy
into a visible failure.

### S6 — A square-root arm, as an instrument

AdaFisher divides by the curvature rather than by its square root, deliberately, with an ablation in
support. But without the square root, noise in the estimate enters the update in full rather than
halved, and the step size becomes inversely proportional to how the loss is scaled. A square-root
arm inside E2 would say whether the trouble at small `λ` is noise. This is a measuring instrument,
not a fix to adopt.

### S7 — Pick `λ` from a measured percentile

The practical form of S1: choose `λ_l` so it sits at the *p*-th percentile of layer *l*'s list, and
try `p` at 1, 10 and 50. A one-knob family whose two ends are today's behaviour (`p = 100`, plain
momentum SGD) and no floor at all (`p = 0`).

---

## Part 5 — Order, and three rules for the protocol

| # | Action | Status | Why here |
|---|---|---|---|
| 1 | **E0** | **done** | closed the original question: the noise is not the curvature |
| 2 | **E1** | **done** (jobs 21283680, 21285097) | gave the target: `λ` above everything, still above 93-100% of directions even corrected |
| 3 | **E3** | **done** (same jobs) | confirmed the arithmetic, and says to run E4 at batch 32 |
| 4 | **E2** | **done** (job 21283681) | ran before the fixes because moving `λ` and the learning rate together does what S2 was for, with no code change. Answered both halves: the barrier is the step size, and lifting it buys nothing |
| 5 | **E4 with S3**, at batch 32 | **done** (jobs 21295803-5) | answered it: **not** the averaging. And found the one real gain in the study, +4.32 points on `cnn_gn_cifar`/`kfac` at 10⁻⁸ |
| 6 | **E5, E6** | **done** (jobs 21374756, 21379909-12) | E4's gain is not the frozen GroupNorm, and the gain is real on two more architectures |
| 7 | **E7 to E14** | **done** | the operating point is real (+6.6 to +10.0 points on three networks of four) and sits between 1e-11 and 1e-10 for `ekfac`/`tekfac`. The eigenbasis ordering defect must be fixed there (E8, E10). The seed floor is 0.15 to 2.4 points depending on the network, not 0.04 to 0.18. `τ` does not transfer between networks better than a fixed number (E14) |
| 8 | **S1**, across networks (one `τ` for every network) | **demoted** after E14 | a fixed `λ` does as well. See S1's status paragraph |
| 9 | **S4 + one fixed `λ`**, tested by a change of batch | open, **demoted** by E15 | the simplest fix consistent with E14, but bounded by the best single `λ`, which S1 per layer beats in all six E15 pairs |
| 10 | **S1 per layer**, as a direct arm against the best single `λ` | **done: E15, adopted** (jobs 21523753-62) | wins 6/6 against the single `λ` and 6/6 against a network-wide relative `λ`; `τ = 0.1` fits `ekfac`/`tekfac` on both networks. Next: the held-out network is E17's (ViT-S), subject to how E17 reads E15 rule 4 (see E15's results); then a batch change and the head-only decomposition. Before E15 it had never been tested. The spread between layers is mostly the head (S1's status paragraph). Must hold the cap per layer and the weight-decay rate still (rules 1 and 3) |
| 11 | **A floor or a clip instead of the added `λ`** (families A and B of `fr/etude_clipping_vs_damping.md`) | **E16, pre-registered**, code done and audited | the feasibility study found both degenerate at the shipped `λ`; E7-E14 found an operating point where `λ` no longer dominates, which is where they can differ. The clip is scale-free, so it is the one candidate that could transfer between networks where no single `λ` does |
| 12 | **E15's and E16's verdicts, tested on a network nobody tuned them on** | **E17, pre-registered and amended**, waits for E16. From E15, candidate C does not qualify (rule 4 read literally); S1-b at `τ = 0.1` may run there only as an exploratory arm | their own transfer rules only ask whether one setting works on the networks it was chosen on. ViT-S is the network where the shipped Fisher arms lose to AdamW, and the λ work has never run it |
| 13 | **Is ViT-S's deficit against AdamW the deficit of momentum SGD?** | **E18, pre-registered, running** (jobs 21532555-60); reproduction gate passed | at the shipped `λ` every Fisher arm should reduce to momentum SGD at `lr(1−β)/λ`. An arm that *is* that limit settles it, and needs nothing from E15 or E16 |

**On the reordering.** S2 was planned as a prerequisite for E2. It turned out not to be needed:
moving `λ` and the learning rate together holds the cap exactly still, which is what S2 was for, and
does it without changing any optimizer code. S2 is still the right thing to ship, because it makes
the cap something you set rather than something that falls out of two constants. But it is no longer
blocking anything.

**Three rules, each one the lesson of a mistake already made.**

1. **Any sweep of `λ` must hold the step-size cap still.** Otherwise it moves two things at once and
   the result cannot be attributed to either (Part 2.1, and E2's result).
2. **Draw no conclusion from `mnist_autoencoder`.** Every arm of every configuration lands on the
   same number there. It cannot tell anything apart (audit_step, section 4.8).
3. **Hold the weight-decay rate still too.** Holding the cap means `lr = cap × λ`, so `lr` falls by
   the same 7 to 8 orders of magnitude as `λ`. Coupled decay (the CNNs) is added to the gradient and
   is unaffected. Decoupled decay (the two transformers, `decoupled_wd=True`) multiplies the weights
   by `1 − lr × wd`, so it vanishes in every lowered arm while the reference arm keeps it. E10 and
   E13 therefore changed two things at once on `cct_2_3x2_cifar` and `vit_micro_cifar`. E14's two
   networks use coupled decay and are not affected. Any new sweep on a transformer must apply the
   decay at the benchmark's own rate, `base_lr × wd`, whatever `λ` is.

---

## Part 6 — Results

### E0 — done. The jumpy validation curves do not need the curvature estimate.

Script: [e0_placebo_val_wobble.py](../../fisher_ref/experiments/e0_placebo_val_wobble.py). Nothing
new was computed: it reads `fisher_ref/outputs/warmup_sgd_baseline_small.json` (six networks × five
modes × {real mode at two starting points, curvature-free stand-in at the first of them}) and
`benchmarks/outputs/*/epochs.csv`. Output: `fisher_ref/outputs/e0_placebo_val_wobble.json`. "Jump"
means the average size of the change from one epoch to the next, over the second half of training,
which is how steps 3 and 4 defined it.

**First, is there anything to explain?** Recomputing step 4's metric on the campaign runs, as a
multiple of how much Adam's own curve jumps in the same run:

| network | how much more the five modes jump than Adam |
|---|---|
| `cnn_gn_cifar` | 3.7 to 7.2 times |
| `vit_micro_cifar` | 3.0 to 6.0 |
| `resnet20_cifar` | 2.6 to 4.9 |
| `cct_2_3x2_cifar` | 1.3 to 1.8 |
| `mlp_ln_mnist` | **0.47 to 1.9** — two of the modes are *calmer* than Adam |

**Then the answer.** Comparing each real mode against its own curvature-free stand-in. Same starting
point, same initial weights, same order of examples. The only difference is whether a curvature
estimate exists at all.

| | jump in accuracy | jump in loss, relative |
|---|---|---|
| stand-in ÷ real, averaged over 25-30 pairs | **0.96** | **1.15** |
| middle value | 0.98 | 1.03 |
| middle half of the pairs | 0.79 to 1.13 | 1.00 to 1.22 |
| times the stand-in was the jumpier one | 9 out of 25 | 19 out of 30 |

**The stand-in jumps as much as the real thing.** An optimizer that never estimates curvature, given
the same step size, reproduces the jumpy validation curves to within a few percent on average, with
no consistent direction. Whatever makes these curves jump, **it is not the curvature estimate**.
Which is what Part 1's measurement predicts, since the estimate is not affecting the step anyway.

Two things this does **not** show, and they matter:

- It does not show that the *step size* is the cause. The stand-in was built to carry the same step
  size, so it shares that suspect too. Telling them apart needs a run at a genuinely different step
  size. That is E2.
- The two sets of runs are not on one scale. The stand-in runs are 15 fixed epochs; the campaign
  runs use a wall-clock budget and differ in how many epochs each arm completed. Only comparisons
  within one set are used above.

One note for anyone re-running it. The strict per-pair test — is the real-versus-stand-in gap inside
the spread between two runs of the same mode? — passes in only 13 of 30 pairs. But that test cannot
be used as stated: with two runs the spread is itself a noisy number, and in several pairs it comes
out at 0.003 accuracy points, which nothing can fit inside. The spread of ratios is the honest
read-out. The per-pair verdict is kept in the JSON for completeness.

### E1 — done, all seven networks. `λ` is above every direction, everywhere.

Script: [e1_lambda_percentile.py](../../fisher_ref/experiments/e1_lambda_percentile.py), job script
[e1_lambda_percentile.sh](../../fisher_ref/slurm/e1_lambda_percentile.sh). Seven runs × five modes ×
three batch sizes, starting from each arm's own halfway checkpoint, with each run's settings read
from its own record. Same protocol as step 7, and the "as it ships" maxima reproduce step 10's
table.

**Run twice**, at 1 000 and then 2 000 warm-up steps. Jobs 21283680 (32 minutes) and 21285097
(64 minutes). The second was needed because the first turned out to be measuring its own start-up
transient for two of the five modes — see the addendum. **Every number below is the 2 000-step run**,
which flags no contaminated cells. The shorter run is kept as
`e1_lambda_percentile_rewarm1000.json`. Output: `fisher_ref/outputs/e1_lambda_percentile.json`. Peak
memory 3.1 GB against 32 GB asked for; a rerun can ask for 8.

**Result 1 — the strongest form the claim can take.** In **all 105 cases** (7 networks × 3 batch
sizes × 5 modes), `λ` is at the **100th percentile** and the fraction of directions above it is
**zero**. Not "above the typical direction". Not "above all but a few". Above every single direction
of every layer of every network, at every batch size tried. Step 10 reported the largest value per
layer. This reports the whole list, and there is nothing above `λ` anywhere in the model zoo.

**Result 2 — and this corrects what one network alone suggested.** Undoing both shrinkages on paper,
at each run's own batch size of 128, `λ` would sit at these percentiles:

| network | `diag` | `kfac` | `ekfac` | `tkfac` | `tekfac` |
|---|---|---|---|---|---|
| `mlp_ln_mnist` | **55.7** | 97.3 | 94.2 | 98.8 | 93.2 |
| `cnn_gn_cifar` (BatchNorm) | 85.4 | 98.5 | 97.8 | 98.7 | 97.9 |
| `resnet20_cifar` | 88.6 | 99.9 | 99.9 | 99.9 | 99.9 |
| `vit_micro_cifar` | 94.4 | 99.1 | 99.0 | 99.2 | 99.0 |
| `cnn_gn_cifar` (GroupNorm) | 97.3 | 99.1 | 98.4 | 99.1 | 98.5 |
| `cct_2_3x2_cifar` | 99.3 | 99.9 | 99.8 | 99.9 | 99.8 |
| `mnist_autoencoder` | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

**For the four Kronecker modes the corrected percentile is 93.2 to 100.0 on every network, without
exception.** Only `diag` ever goes below, and only on the two smallest. An earlier reading based on
one network alone concluded "correcting the size fixes `diag`", and that **does not generalize**:
`diag` lands between 85 and 100 on five of the seven. **Undoing the two inherited conventions is not
enough, by itself, to make any of these five modes curvature-aware on any network here.** `λ` has to
come down as well, by one to five orders of magnitude depending on the mode and the network. That is
the case for fixes S1 and S7, and the reason not to expect E4 alone to change an outcome.

`mnist_autoencoder`, the benchmark where all five modes get stuck, is the extreme: **100.0 in every
mode even after correcting**, at every batch size. And `diag` is the least dominated mode on all
seven networks, which fits its being the only mode that ever responded to a `λ` sweep.

**Result 3 — a percentile is not always a usable target.** For `kfac` on `mlp_ln_mnist`, the `λ`
that would sit at the 10th percentile came out **exactly zero**. At least 10% of its directions have
exactly zero estimated curvature. That is the rank deficiency this project has met before: the first
layer's input quantity has 136 exactly-zero values out of 785, because 130 MNIST pixels are always
black. It is the same structure that crashed the GPU's matrix solver in campaign 1. So fix S7 has
to pick its percentile **above** each layer's dead fraction, and report that fraction. A naive "put
`λ` at the 10th percentile" rule would set `λ = 0`.

### E1 addendum — a measurement artifact, found, explained exactly, and fixed

Reading E3's table turned up a number with a suspicious look. For `ekfac` and `tekfac`, the average
curvature stopped falling with the batch size and settled on a value that **repeated across networks
with nothing in common**: 1.34e-7, 1.35e-7, 1.59e-7, 1.69e-7 times `λ` on the four models with
`λ = 0.001`, and 4.51e-8, 4.85e-8 on the two with `λ = 0.003`. That is exactly the warning sign
[validation_noise_investigation.md](validation_noise_investigation.md) records in its own practical
notes: *when a summary number comes out identical across unrelated networks, suspect it is measuring
the setup, not the network.*

It was. Every mode starts its state at the identity, and that start fades by `0.08` at each update.
For `ekfac` and `tekfac` the leftover lands **directly** in the list of curvature values, because
for those two modes that list *is* the stored quantity. And those two get one update fewer than the
others over the same warm-up: their quantity is only updated once a basis exists, and the basis is
created by the first refresh, which happens *after* the first round of hooks. So over 1 000 steps at
one update per 100, the other quantities get 10 updates and these get 9. And `0.08⁹` is
`1.342 × 10⁻¹⁰`:

| network | `λ` | predicted `0.08⁹/λ` | measured `ekfac` | measured `tekfac` | ratio |
|---|---|---|---|---|---|
| `mnist_autoencoder` | 0.001 | 1.342e-7 | 1.340e-7 | 1.340e-7 | **1.00** |
| `resnet20_cifar` | 0.001 | 1.342e-7 | 1.350e-7 | 1.350e-7 | **1.01** |
| `cct_2_3x2_cifar` | 0.003 | 4.474e-8 | 4.510e-8 | 4.510e-8 | **1.01** |
| `vit_micro_cifar` | 0.003 | 4.474e-8 | 4.850e-8 | 4.850e-8 | 1.08 |
| `cnn_gn_cifar` | 0.001 | 1.342e-7 | 1.600e-7 | 1.590e-7 | 1.19 |
| `mlp_ln_mnist` | 0.001 | 1.342e-7 | 1.690e-7 | 1.630e-7 | 1.26 |

Three digits on four of six networks, **and it tracks two different values of `λ`**. That is the
decisive test, because the leftover is a property of how the optimizer starts, not of the data. At
batch 512 those two modes were not measuring curvature at all. The visible symptom was their
"`λ` at percentile 0.00, fraction above 1.0000" cells, where multiplying a floor by the batch
correction lifts all of it above `λ`.

**What it changes and what it does not.** This is an artifact of the warm-up, not of training: in a
real run the start keeps fading and is gone after twenty-odd cycles. But it does correct a rule this
repository states. `rewarm_fidelity.py`'s "never shorter than 10 cycles" comes from requiring
`0.08^k` to be well under `λ`. That is the right requirement for the *divisor the optimizer applies*,
which is dominated by `λ`. It is **too weak for the list of curvature values itself**, which is
orders of magnitude below `λ`. The right requirement there is `0.08^k` well under the thing being
measured.

The script now computes that floor exactly and prints how far above it each measurement is, flagging
anything under three times it. So the trap is self-documenting instead of something the next reader
has to rediscover.

**Fixed and confirmed.** Job 21285097 redid everything at 2 000 warm-up steps (`0.08¹⁹` is
1.7 × 10⁻²¹, dead), finished in 1 hour 4 minutes, and flags **no cells at all**. The symptom is gone
with it: the spurious cells at batch 512 are now 94 to 100, in line with every other mode. Result 1
is unchanged; it was never near the floor. Result 2's table has been updated to the 2 000-step
numbers, and the effect is to *strengthen* it — the four Kronecker modes now sit at 93 to 100
everywhere. One honest cost of the longer warm-up: the parameters drift further from the checkpoint
during it, because the warm-up takes real optimizer steps.

### E3 — done, all seven networks. No batch size rescues the optimizer, and the 1/batch² rule holds.

**The practical result is unambiguous.** At batch 32, 128 and 512, on all seven networks, in all
five modes, `λ` is at the **100th percentile**. Changing the batch size by a factor of sixteen does
not move it in a single one of the 105 cases. No batch size in the usable range makes this optimizer
curvature-aware.

**The check on the arithmetic.** The stored curvature should fall by a factor of 16 — a ratio of
0.0625 — each time the batch is quadrupled. Measured on the average, at 2 000 warm-up steps, for all
four Kronecker modes and both batch steps: **56 measurements, every one between 0.021 and 0.144**,
against 1.0 if no such rule existed. Per network, the range across its four modes and two steps:

| network | measured range | middle ÷ predicted |
|---|---|---|
| `cct_2_3x2_cifar` | 0.057 to 0.078 | 1.08 |
| `cnn_gn_cifar` (GroupNorm) | 0.063 to 0.078 | 1.13 |
| `resnet20_cifar` | 0.056 to 0.093 | 1.19 |
| `vit_micro_cifar` | 0.076 to 0.088 | 1.31 |
| `mlp_ln_mnist` | 0.021 to 0.144 | 1.29 |
| `mnist_autoencoder` | 0.080 to 0.140 | 1.76 |
| `cnn_gn_cifar` (BatchNorm) | 0.094 to 0.111 | 1.64 |

**The rule holds on every network and every Kronecker mode**, with a mild tendency to fall slightly
less than predicted, which is not chased here. Note what the shorter run could not see: at 1 000
steps `ekfac` and `tekfac` were pinned to their start-up floor at large batch, and the rule was
invisible for them. Removing the transient made it appear. That is a second, independent
confirmation of the addendum.

**`diag` is exempt, and that was predicted.** Squeezing each layer's numbers into 0 to 1 before
storing them destroys the size, so no 1/batch² rule can apply. Measured, its ratios are 0.18 to 0.82
where the other four are at 0.06 to 0.11. But it drifts in an interesting direction. `diag`'s
*corrected* percentile of `λ`, across batch 32 / 128 / 512:

| network | b32 | b128 | b512 |
|---|---|---|---|
| `cnn_gn_cifar` (BatchNorm) | **31.5** | 85.4 | 97.2 |
| `cnn_gn_cifar` (GroupNorm) | **57.9** | 97.3 | 97.7 |
| `resnet20_cifar` | **65.7** | 88.6 | 99.0 |
| `mlp_ln_mnist` | **65.9** | 55.7 | 96.2 |
| `vit_micro_cifar` | 79.2 | 94.4 | 98.0 |
| `cct_2_3x2_cifar` | 98.4 | 99.3 | 99.5 |
| `mnist_autoencoder` | 99.9 | 100.0 | 100.0 |

A bigger batch produces more extreme values inside each layer, the squeeze then pushes most of the
entries toward zero, and the safety constant dominates again. **A small batch plus a corrected
average is the one corner of this study where any mode becomes genuinely curvature-aware** — `diag`,
on four of seven networks, at batch 32. Two consequences. `λ` has no batch-independent meaning in
`diag` either, by a different route from the 1/batch² one. And **E4 should be run at batch 32, not
128**, if it is meant to give `diag` a chance to show anything.

**A lesson about measuring.** The first version of this measurement summarized each list by its
middle value and produced nonsense: ratios moving up and down and by four orders of magnitude
between batch sizes. The reason is result 3's dead directions — the middle value sits inside a floor
of zeros, so it measures the floor. The read-out is now the average. Together with the start-up
floor of the addendum, and with step 7's own version of the same trap, that is three separate
instances of one lesson: **on these lists, summarize with the average or a high percentile, never
with the middle — and always check the answer against the floors you can compute exactly.**

### E2 — done. The earlier collapse was entirely the step size; and removing it buys almost nothing.

Script: [e2_lambda_at_fixed_gain.py](../../fisher_ref/experiments/e2_lambda_at_fixed_gain.py), job
script [e2_lambda_at_fixed_gain.sh](../../fisher_ref/slurm/e2_lambda_at_fixed_gain.sh). Job
21283681, 2 hours 6 minutes. Full grid: 3 networks × 5 modes × (1 reference + 5 values of `λ` ×
2 protocols) = 165 runs of 15 epochs, same starting point, same initial weights, same order of
examples throughout. Output: `fisher_ref/outputs/e2_lambda_at_fixed_gain.json`.

**Result 1 — the protocol was the whole story, and the separation is total.**

| | runs that still train |
|---|---|
| learning rate left alone — the earlier study's protocol | **19 out of 75** |
| learning rate moved with `λ` so the step-size cap never changes | **75 out of 75** |

Broken down by `λ`, out of 15 cases each (3 networks × 5 modes):

| `λ` | learning rate left alone | cap held still |
|---|---|---|
| 0.0001 | 15 / 15 | 15 / 15 |
| 0.000001 | 3 / 15 | **15 / 15** |
| 10⁻⁸ | 0 / 15 | **15 / 15** |
| 10⁻¹⁰ | 0 / 15 | **15 / 15** |
| 10⁻¹² | 1 / 15 | **15 / 15** |

Leaving the learning rate alone fails in the two ways the earlier study catalogued: `kfac` and
`tkfac` crash on a singular matrix, and `diag`, `ekfac` and `tekfac` settle quietly at chance-level
accuracy. And it fails in **all** five modes on all three networks from `λ = 10⁻⁶` downwards.
Holding the cap still, **not one run out of seventy-five fails**, at any `λ` down to 10⁻¹². The
earlier conclusion — that `λ` cannot be pushed more than one order of magnitude below its default —
described that study's own protocol, not the optimizer. Part 2.1's algebra is now a measurement
across three architectures and five modes.

**Result 2 — and this is the one that matters. Removing the barrier buys almost nothing.**

Change in test accuracy against each mode's own default `λ`, with the cap held still:

| network | mode | 0.0001 | 10⁻⁶ | 10⁻⁸ | 10⁻¹⁰ | 10⁻¹² |
|---|---|---|---|---|---|---|
| `mlp_ln_mnist` | `diag` | +0.18 | −0.03 | −1.86 | −17.66 | −37.44 |
| | `kfac` | −0.12 | −0.27 | −0.26 | −0.83 | −2.28 |
| | `ekfac` | −0.16 | −0.19 | −0.13 | −2.39 | −63.68 |
| | `tkfac` | −0.07 | −0.26 | −0.02 | −0.70 | −52.71 |
| | `tekfac` | +0.02 | +0.00 | −0.11 | −2.54 | −62.95 |
| `cnn_gn_cifar` | `diag` | +1.73 | **+2.05** | −3.21 | −19.25 | −35.31 |
| | `kfac` | +0.40 | +0.68 | +0.48 | **+1.12** | −3.93 |
| | `ekfac` | −0.69 | −0.17 | −0.75 | −4.56 | −7.59 |
| | `tkfac` | **+1.32** | +0.05 | +0.10 | −3.49 | −10.49 |
| | `tekfac` | +0.56 | +1.01 | +0.21 | −3.71 | −6.91 |
| `resnet20_cifar` | `diag` | +0.15 | −0.61 | −3.53 | −26.08 | −45.23 |
| | `kfac` | +0.28 | +0.07 | −0.41 | −0.72 | −2.23 |
| | `ekfac` | −0.01 | −0.16 | −0.45 | −1.81 | −5.90 |
| | `tkfac` | +0.07 | +0.17 | −0.88 | −2.38 | −15.01 |
| | `tekfac` | −0.55 | −0.58 | −0.83 | −1.81 | −6.60 |

**The best gain anywhere in seventy-five runs is 2.05 accuracy points**, and the three biggest are
all on the same network: `cnn_gn_cifar`, where `diag` gains 2.05 at 10⁻⁶, `tkfac` gains 1.32 at
0.0001 and `kfac` gains 1.12 at 10⁻¹⁰. Nine of the fifteen network-and-mode pairs have *some* `λ`
that beats their default, by well under half a point in most cases. Everything else is flat, and
then falls off a cliff.

**Three readings, and the third is the point.**

1. **The crossover exists, and it is where E1 said it would be.** Nothing moves at all between 0.001
   and 10⁻⁶. E1 measured that `λ` is still above 93 to 100% of directions there, so there is nothing
   to move. Things start changing at 10⁻⁸ to 10⁻¹⁰, which is exactly the range E1's target values
   point at. Two independent measurements, one of the curvature values and one of the behaviour,
   agree on the location.
2. **The order in which the modes break follows E1's ordering, mode by mode.** `diag` is the least
   dominated mode on all seven networks (55th to 100th percentile, against 93rd to 100th for the
   others). It is both the mode that gains the most, 2.05 points, and the one that breaks first and
   hardest: −17.7 at 10⁻¹⁰, and −37 to −45 at 10⁻¹². The other four are flat until 10⁻¹⁰. `kfac` is
   the most robust anywhere: never worse than −3.93 in any cell, at any `λ`, on any network.
3. **Curvature, given a real vote, does not pay for itself here.** This is a negative result and it
   reframes the whole thread. The five modes are not being held back by `λ`. Take the safety
   constant away properly, with the step size held still, across nine orders of magnitude, and they
   do not become better optimizers. They stay where they were, and then they break.

**The one thing this does not settle, which is exactly E4's job.** Making `λ` small with the cap
held still is, on paper, the same as putting the curvature back at its true size (Part 2.1). So E2
*has* explored the corrected-size regime. But it has **not** corrected the *averaging*: the estimate
is still 92% one batch of examples, and still full of dead directions. So "curvature does not help"
is, at this point, tangled up with "this way of estimating it is too noisy, and has too many dead
directions, to help". The collapse at 10⁻¹², where an almost-singular estimate is inverted with almost nothing
added to it, is exactly what a noisy-estimate failure looks like. **E4 with fix S3 — a real average,
no start-up value — is the experiment that tells them apart**, and E2 has made it the most valuable
one left rather than an also-ran. Per E3, run it at batch 32.

### E4 — done. It is not the averaging. And there is one real gain, at batch 32.

Script: [e4_fixed_average.py](../../fisher_ref/experiments/e4_fixed_average.py), job script
[e4_fixed_average.sh](../../fisher_ref/slurm/e4_fixed_average.sh). Three jobs, one per network:
21295803 (`mlp_ln_mnist`, 50 min), 21295804 (`cnn_gn_cifar`, 60 min), 21295805 (`resnet20_cifar`,
4 h 02), all finished cleanly. Grid per network: 5 modes x 2 estimators x (1 reference + 4 values
of the safety constant) = 50 runs of 15 epochs at **batch 32**, same starting point and same order
of examples throughout. Outputs: `fisher_ref/outputs/e4_fixed_average_<network>.json`.

The two estimators, everything else identical:

| | what it does |
|---|---|
| `shipped` | `0.08 * old + 0.008 * new`, started from the identity. 92% of the state is the most recent batch, the state settles at 1/115 of what it measures, and a residue of the identity is still in it. |
| `corrected` | `0.8 * old + 0.2 * new`, started from the first observation. This is the paper's own Eq. (3) at its own `gamma`, plus fix S3. The state settles at what it measures, and no residue of anything else is ever in it. |

**Result 1 — the answer to the question, and it is no.** A nominal value of the safety constant does
not mean the same thing to the two estimators: the corrected one stores curvature 115 times larger
(13 225 for `diag` and `kfac`, which store two quantities), so the *same* number is a much smaller
*effective* damping for it. Comparing them at the same nominal value would be comparing two
different operating points. Compared at matched effective damping instead, over the 24 pairs where
that match lands inside the grid that was actually run:

| | pairs | mean | range |
|---|---|---|---|
| the four Kronecker modes | 21 | **−0.37 points** | −2.54 to +0.51 |
| `diag` | 3 | −1.92 points | −2.85 to −0.88 |

**A real average changes nothing.** Not better, not meaningfully worse: the same result, at the same
effective damping, from an estimator that is 92% one batch and one that is a genuine average over
many. So E2's finding stands as stated. **The curvature genuinely does not pay for itself on these
networks at this operating point**, and the way it was being averaged was not what was holding it
back.

**Result 2 — and there is one real gain, which E2 at batch 128 could not see.** Against each
estimator's own reference, the best any cell reaches:

| network | mode | best gain | at |
|---|---|---|---|
| `cnn_gn_cifar` | `kfac` | **+4.32 points** | 10⁻⁸ |
| `cnn_gn_cifar` | `ekfac` | +2.90 | 10⁻⁸ |
| `cnn_gn_cifar` | `tekfac` | +2.43 | 10⁻⁸ |
| `cnn_gn_cifar` | `tkfac` | +2.13 | 10⁻⁸ |
| `cnn_gn_cifar` | `diag` | +1.15 | 10⁻⁴ |
| `resnet20_cifar`, `mlp_ln_mnist` | every mode | +0.03 to +0.66 | — |

At batch 32, **all five modes improve on `cnn_gn_cifar`**, and the four Kronecker ones all peak at
the same place, 10⁻⁸. E2's best anywhere at batch 128 was +2.05; here `kfac` reaches +4.32. That is
the operating point where curvature pays, it is five orders of magnitude below the default, and it
is specific to this network — the other two are flat to within a point. Note it is the **shipped**
estimator that gets there, which is Result 1 again from the other side.

**Result 3 — `diag` is the fragile one, everywhere, and the corrected average makes it worse.**
`diag`'s corrected arm loses 4.6 points at 10⁻⁴ and collapses by 27 to 36 points at 10⁻⁶ on all
three networks, while its shipped arm is still within a point of its reference there. That is E1's
ordering once more: `diag` is the least damping-dominated mode, so it is the first to feel the
curvature — and correcting the average, which multiplies its stored curvature by 13 225, pushes it
straight past the useful range.

**A note on how this was nearly got wrong.** A local smoke of this experiment — 2 epochs, batch 256,
`mlp_ln_mnist` — showed the corrected estimator beating the shipped one by 3.4 points on `kfac` and
29 points on `ekfac`, which is the opposite sign to the real result. Two epochs is 4 updates of the
running average, where the corrected one is simply better conditioned; 15 epochs at batch 32 is 210
updates, and the picture reverses. The smoke was useful for finding two bugs and worthless as
evidence. It is recorded here so that nobody re-runs it and believes it.

**What this closes, and what is left.** The entanglement E2 named is resolved: it is not the
estimator's averaging. Three things remain open and none of them is now the obvious next step —
fix S1 (a safety constant relative to each layer, which is what two of the source papers ask for and
what would reach `cnn_gn_cifar`'s 10⁻⁸ operating point without hand-tuning), E5 (the thousand-to-one
split between hooked and unhooked parameters inside every network), and E6 (step 16's causal test).

### E5 control — done. E4's gain is not the frozen GroupNorm.

Script: [e5_unhooked_freeze_control.py](../../fisher_ref/experiments/e5_unhooked_freeze_control.py),
job script [e5_unhooked_freeze_control.sh](../../fisher_ref/slurm/e5_unhooked_freeze_control.sh).
Job 21374756, 43 minutes. Output: `fisher_ref/outputs/e5_unhooked_freeze.json`.

**Why it had to be run.** E4's only real gain came from lowering the safety constant with the
step-size cap held fixed. Holding the cap means `lr = λ`, and a parameter the optimizer does not
precondition steps by `lr` times its momentum. So over that sweep those parameters had their step
shrunk by 100 000. They stopped moving. On `cnn_gn_cifar` they are the 224 affine parameters of its
three GroupNorm layers, 0.9% of the network, but each one scales a whole channel. If freezing them
is what buys the 4.32 points, the one positive result in this investigation is a step-size artifact.

**Result.** Freezing them at the *default* safety constant, everything else identical:

| mode | baseline | frozen | difference | `λ = 10⁻⁸` and frozen |
|---|---|---|---|---|
| `diag` | 60.87 | 60.75 | **−0.12** | 56.90 |
| `kfac` | 61.26 | 61.36 | **+0.10** | 65.81 |
| `ekfac` | 60.25 | 61.93 | +1.68 | 62.99 |
| `tkfac` | 61.37 | 61.11 | **−0.26** | 63.22 |
| `tekfac` | 60.84 | 61.74 | +0.90 | 62.71 |

For `kfac`, the mode that carried the result, freezing on its own is worth **+0.10** against the
+4.32 that had to be explained. It accounts for 2% of it.

**The last column validates the mechanism on the way past.** Freezing explicitly at `λ = 10⁻⁸`
reproduces what E4 measured when the sweep froze them implicitly, to within 0.6 points on all five
modes: 65.81 against 65.58, 62.99 against 63.15, 63.22 against 63.50, 62.71 against 63.27. So the
reading was right — at `λ = 10⁻⁸` those parameters really are frozen — and it is not what produces
the gain.

**The cleanest reading** compares the two columns where the freeze is held constant, so that only
the safety constant changes:

| mode | effect of `λ` alone, 10⁻³ → 10⁻⁸, freeze held constant |
|---|---|
| `kfac` | **+4.45** |
| `tkfac` | **+2.11** |
| `ekfac` | +1.06 |
| `tekfac` | +0.97 |
| `diag` | −3.85 |

With the freeze neutralised, `kfac`'s effect is *larger* than E4 measured, not smaller. The premise
holds.

**One measured caveat.** For `ekfac` and `tekfac`, freezing accounts for a real share of their E4
gains: +1.68 of +2.90, and +0.90 of +2.43. Their gains were not pure. Only `kfac` and `tkfac` are.

**What it changed downstream.** `vit_micro_cifar` has **9.7%** of its parameters outside the four
layer types the optimizer handles — its position table — against 0.9% here. The same confound would
be ten times larger there, so the next sweeps carry the freeze as an **axis of the experiment**
rather than letting it happen silently: `E4_FREEZE_UNHOOKED` holds those parameters at their initial
values in *every* arm, which turns the confound into a constant. Both settings are run on both
networks (jobs 21379909-12, `cct_2_3x2_cifar` and `vit_micro_cifar`).

### E6 — done, two new architectures. The gain is real, it is bigger, and the freeze confound is zero.

Jobs 21379909-12, one hour each. `cct_2_3x2_cifar` and `vit_micro_cifar`, the same sweep as E4
(5 modes x 5 values of the safety constant, batch 32, 15 epochs, step-size cap held), run twice per
network: once with the unpreconditioned parameters trainable, once with them held at their initial
values in every arm. Outputs: `fisher_ref/outputs/e4_fixed_average_*_frozen{0,1}.json`.

**Result 1 — the confound the E5 control found on 0.9% of a network is zero on 9.7% of one.**
`vit_micro_cifar` keeps 9.7% of its parameters outside the four layer types the optimizer handles,
ten times `cnn_gn_cifar`'s share. Over 50 cells across the two networks, freezing them changes the
result by at most **0.67 accuracy points**, with a mean of **-0.006** and **+0.025**. The question is
closed, on the network where it was most likely to matter.

**Result 2 — the gain is real on both, and larger than on `cnn_gn_cifar`.** Best gain over each
mode's own reference:

| network | `kfac` | `ekfac` | `tkfac` | `tekfac` | at |
|---|---|---|---|---|---|
| `cct_2_3x2_cifar` | **+5.78** | **+7.55** | **+6.33** | **+7.12** | 10⁻¹⁰, the bottom of the grid |
| `vit_micro_cifar` | +6.49 | +2.19 | +3.42 | +2.98 | 10⁻¹⁰ / 10⁻⁸ |
| `cnn_gn_cifar` (E4) | +4.32 | +2.90 | +2.13 | +2.43 | 10⁻⁸ |

Three of the five networks now show a real gain from lowering the safety constant. The two that do
not are `resnet20_cifar`, where the effect is expected to be diluted across 39 layers, and
`mlp_ln_mnist`, which is saturated at 97.4% and cannot show anything.

`diag` is the exception on every network: it collapses at 10⁻¹⁰ (-26.9 on `cct_2_3x2_cifar`, -14.6 on
`vit_micro_cifar`, -23.2 on `cnn_gn_cifar`). That is E1's ordering again — it is the least
damping-dominated mode, so it is the first to enter the regime where the curvature commands, and
that regime is bad.

**Result 3 — and the reason a follow-up was needed. The grid is too coarse to locate the peak.** All
four Kronecker modes on `cct_2_3x2_cifar`, and two on `vit_micro_cifar`, peak at the *bottom* of the
grid. Their optimum is at or below 10⁻¹⁰ and cannot be read off. Since the whole point was to test
whether the constant `tau` transfers between architectures, and `tau` is computed from the peak
position, the coarse grid cannot answer the question it was run to answer.

### E7 — the pre-registered test of `tau`, running

`tau = 1.7e-5` was measured on `cnn_gn_cifar` alone. If it is a constant of the *method* rather than
a regularity of one network, it predicts every other network's best safety constant from that
network's own mean curvature, which E1 has already measured. Written down before the runs:

| network | mean curvature (true scale) | `tau` predicts | coarse grid said |
|---|---|---|---|
| `cnn_gn_cifar` | 5.7e-4 to 6.3e-4 | 9.6e-9 to 1.1e-8 | 10⁻⁸, dead on |
| `vit_micro_cifar` | 2.0e-4 to 2.4e-4 | 3.4e-9 to 4.0e-9 | 10⁻⁸ for two modes, 10⁻¹⁰ for two |
| `cct_2_3x2_cifar` | 2.5e-5 to 2.9e-5 | 4.2e-10 to 4.9e-10 | at or below 10⁻¹⁰ |
| `resnet20_cifar` | 2.0e-5 to 2.7e-5 | 3.4e-10 to 4.6e-10 | no peak |

At the coarse grid's resolution (two orders of magnitude between points) the prediction is consistent
with `cnn_gn_cifar` and `cct_2_3x2_cifar`, and half consistent with `vit_micro_cifar`. That is not a
test. Jobs 21386750-52 run a fine grid bracketing each prediction: `cct_2_3x2_cifar` over
{3e-9 … 3e-11}, `vit_micro_cifar` over {3e-8 … 3e-10}, `cnn_gn_cifar` over {1e-7 … 1e-9}. Only the
trainable arm is run, since Result 1 measured the freeze axis to be inert.

If each network's peak lands on its predicted value, `tau` is a property of the method and fix S1
has its constant. If the peaks land at the same *absolute* safety constant instead, regardless of
each network's curvature, then a relative damping buys nothing over a better-chosen fixed number,
and S1's whole premise is wrong.

### E7 — done. The fixed number is refuted, the proportional rule under-corrects, and E2's conclusion falls.

Jobs 21386750-52, 36 to 91 minutes. Fine grids bracketing each network's predicted best safety
constant, trainable arm only (E6 measured the freeze axis inert). Outputs:
`fisher_ref/outputs/e7_fine_{cnn,vit,cct}.json`.

**A determinism check first.** The fine runs re-ran every mode's reference arm. All 15 reproduce the
coarse runs' references to **0.0000 accuracy points**. Same seed, same protocol, same data order —
so every comparison below is between runs that differ only in the safety constant.

**Where the peaks are:**

| network | mode | reference | best | gain | at | inside the window? | predicted | ratio |
|---|---|---|---|---|---|---|---|---|
| `cnn_gn_cifar` | `kfac` | 61.26 | 65.58 | **+4.32** | 10⁻⁸ | yes | 9.8e-9 | 1.02 |
| | `ekfac` | 60.25 | 63.15 | +2.90 | 10⁻⁸ | yes | 1.0e-8 | 0.99 |
| | `tkfac` | 61.37 | 63.50 | +2.13 | 10⁻⁸ | yes | 1.1e-8 | 0.93 |
| | `tekfac` | 60.84 | 63.27 | +2.43 | 10⁻⁸ | yes | 9.6e-9 | 1.04 |
| `vit_micro_cifar` | `kfac` | 45.35 | 50.61 | **+5.26** | ≤3e-10 | **no, still rising** | 4.0e-9 | 0.07 |
| | `ekfac` | 46.63 | 48.82 | +2.19 | 10⁻⁸ | yes | 3.4e-9 | 2.91 |
| | `tkfac` | 46.06 | 50.90 | **+4.84** | 3e-9 | yes | 3.9e-9 | 0.76 |
| | `tekfac` | 46.64 | 48.30 | +1.66 | 1e-9 | yes | 3.7e-9 | 0.27 |
| `cct_2_3x2_cifar` | `kfac` | 73.05 | 80.16 | **+7.11** | ≤3e-11 | **no, still rising** | 4.9e-10 | 0.06 |
| | `ekfac` | 72.49 | 80.25 | **+7.76** | ≤3e-11 | **no, still rising** | 4.2e-10 | 0.07 |
| | `tkfac` | 72.68 | 79.62 | **+6.94** | 3e-10 | yes | 4.5e-10 | 0.66 |
| | `tekfac` | 72.98 | 80.54 | **+7.56** | ≤3e-11 | **no, still rising** | 4.2e-10 | 0.07 |

**Result 1 — E2's conclusion is wrong, and by a lot.** E2 concluded that curvature, given a real
vote, does not pay for itself: the best gain anywhere in 75 runs was +2.05 points. On
`cct_2_3x2_cifar` three of the four Kronecker modes reach **80.2 to 80.5%** against 72.5 to 73.1% at
the default — **+7.1 to +7.8 points** — and they are still improving at the bottom of the window.
E2 drew its conclusion from three networks, two of which cannot show the effect at all
(`mlp_ln_mnist` is saturated at 97.4%, `resnet20_cifar` dilutes it across 39 layers), and treated
the third's +4.32 as an isolated outlier. It was not an outlier.

**Result 2 — a single fixed safety constant is refuted.** The best value spans **3e-11 to 1e-8
across networks, a factor of at least 333**. And those are bounds: four peaks are still rising at
the bottom of their window, so the true span is wider. No one number serves three networks.

**Result 3 — the proportional rule points the right way and under-corrects, consistently.** Measured
on two independent network pairs, where a rule of `lambda` proportional to the mean curvature would
predict an exponent of 1:

| pair | curvature ratio | rule predicts | measured | exponent |
|---|---|---|---|---|
| `cnn_gn_cifar` / `vit_micro_cifar` | 2.7x | 2.7x | ≥5.8x | **≥1.78** |
| `cnn_gn_cifar` / `cct_2_3x2_cifar` | 21.9x | 21.9x | ≥187x | **≥1.70** |

Two pairs, two independent measurements, and the same exponent to within 5%. The safety constant
should fall faster than the curvature does, roughly as its square. There is no structural reason for
an exponent of 2 in any of the five modes' damping algebra, so this is an empirical fact without an
explanation yet.

**Result 4 — of five candidate summaries of the curvature, the mean is the best and is not good
enough.** Taking the 8 peaks that sit strictly inside their window, and asking which statistic makes
`lambda_best / statistic` constant:

| statistic | spread of the ratio |
|---|---|
| **mean** | **10.8x** |
| median | 17.2x |
| 90th percentile | 42.5x |
| 99th percentile | 70.3x |
| maximum | 59.4x |

The raw best `lambda` itself spans 33x over the same 8 peaks, so the mean-relative rule removes
about a third of the variation in log terms and leaves 10.8x. That is far too loose to call `tau` a
constant of the method. **Fix S1 in its textbook form — `lambda` proportional to the layer's mean
curvature, one dimensionless knob — is better than a fixed number and is not sufficient.**

**Result 5 — `diag` must not share the rule.** On all three networks it degrades monotonically as
the safety constant falls, its best value sits at the *top* of every window, and it loses 7.9 points
on `cct_2_3x2_cifar` where the other four gain 7. Whatever rule the four Kronecker modes get, `diag`
needs the opposite one.

**Limits of this measurement, stated.** One seed per cell. The gains of +2 to +7.8 points are far
above the two-seed floor this project has measured on the same protocol (0.04 to 0.18 points), but
differences of 0.2 to 0.5 between neighbouring values of the constant are not resolvable, so a peak
position carries about half a grid step of uncertainty. Four peaks are at the bottom of their window
and are bounds, not values, which makes Results 2 and 3 lower bounds. And the mean curvatures come
from E1, measured at a mid-training checkpoint, while these sweeps train from scratch — the leading
candidate explanation for the 10.8x residual in Result 4 is that a mid-training statistic is the
wrong thing to set a whole-run constant from.

### E9 — done. The candidate is refuted, and the check it forced is the real finding.

Script: [e9_early_curvature_tau.py](../../fisher_ref/experiments/e9_early_curvature_tau.py), job
script [e9_early_curvature_tau.sh](../../fisher_ref/slurm/e9_early_curvature_tau.sh). Job 21432483,
9 minutes 46. Output: `fisher_ref/outputs/e9_early_curvature_tau.json`.

**What it tested.** E7 left one candidate for the 10.8x residual in `tau`: every mean curvature it
used came from a half-trajectory checkpoint, while the sweeps train from scratch, so a mid-training
statistic might simply be the wrong number to set a whole-run constant from. This job trains each
network from scratch at batch 32 and re-reads the curvature at 1000, 2000, 4000 and 8000 steps,
then recomputes `tau` from each.

**Result 1 — the candidate is refuted, cleanly.**

| where the curvature is read | spread of `tau` over the 8 pairs |
|---|---|
| mid-training (E1, what E7 used) | 10.8x |
| step 1000 | 10.0x |
| step 2000 | 10.0x |
| step 4000 | 10.0x |
| step 8000 | 10.0x |

Reading the curvature early changes the spread by 7%. It is not where the residual comes from. The
value itself is stable — the geometric mean of `tau` moves only from 1.48e-5 to 2.05e-5 across the
whole trajectory, against 1.51e-5 mid-training — so the *level* of the constant is not in question.
Its *spread* is.

**Result 2 — the spread is not spread across the networks. It sits inside one of them.**

| group | step 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|
| `cnn_gn_cifar`, 4 modes | 1.1x | 1.3x | 1.3x | 1.4x |
| `vit_micro_cifar`, 3 modes | **10.0x** | **10.0x** | **10.0x** | **10.0x** |
| all 8 pairs | 10.0x | 10.0x | 10.0x | 10.0x |

And on `vit_micro_cifar` the three modes have **the same curvature** — 1.60e-4, 1.69e-4, 1.69e-4 —
while their best safety constants differ by a factor of ten: 1e-8, 3e-9, 1e-9. So the residual
cannot be a curvature-statistic problem. Three modes with one curvature cannot want three different
constants because of how the curvature was summarised.

**Result 3 — and this is the finding. None of the eight peaks is actually resolved.** Checking each
peak against its own runner-up in the same sweep:

| network | mode | best | runner-up | margin |
|---|---|---|---|---|
| `cnn_gn_cifar` | `kfac` | 65.58 | 65.11 | 0.47 |
| | `ekfac` | 63.15 | 63.07 | **0.08** |
| | `tkfac` | 63.50 | 63.16 | 0.34 |
| | `tekfac` | 63.27 | 63.00 | 0.27 |
| `vit_micro_cifar` | `ekfac` | 48.82 | 48.19 | 0.63 |
| | `tkfac` | 50.90 | 50.74 | **0.16** |
| | `tekfac` | 48.30 | 47.63 | 0.67 |
| `cct_2_3x2_cifar` | `tkfac` | 79.62 | 79.01 | 0.61 |

The seed-to-seed floor this project has measured on this protocol is **0.04 to 0.18 accuracy
points**. Four of the eight margins are at or below it. The other four are three to four times it,
which is not comfortable either at one seed. And one curve is visibly not a curve with a peak:
`vit_micro_cifar`/`tkfac` reads 48.66, 49.48, **50.90**, 47.47, 50.74 — a dip of three points
between two values that agree to 0.16.

**So `tau` has not been measured.** Every value of it in this document is computed from a peak
position that is not distinguishable from its neighbour at one seed. That includes the tight-looking
1.1x to 1.4x inside `cnn_gn_cifar`, which is close to circular: all four of its modes peaked at the
same grid point and have mean curvatures within 12% of each other, so their `tau` values must agree
whether or not any rule holds. Four modes there are close to one determination, not four.

**What this changes.** The blocking measurement for fix S1 is not a better curvature statistic and
not a finer grid. It is **seeds**. Until each peak is separated from its neighbour by more than the
seed-to-seed floor, the constant cannot be pinned, and neither the "10.8x is too loose" reading of
E7 Result 4 nor any tighter reading is supported by the data. E7's own limits section flagged the
half-grid-step uncertainty; this run is what forced it to be followed through.

The cheap version: keep only the two or three values of the safety constant bracketing each peak,
and run those at five seeds. On `cnn_gn_cifar` that is 3 x 4 x 5 = 60 runs, about the cost of two of
the sweeps already run.

### E8 — done. Three peaks finally located, and a defect that was understating two modes by up to 4.5 points.

Jobs 21432181 (the sweep, 1 h 15) and 21432595 (the ordering control, 32 min). `cct_2_3x2_cifar`,
window extended down to 1e-12, batch 32, 15 epochs, same protocol as E7. Outputs:
`fisher_ref/outputs/e8_deep_cct.json` and `e8_deep_cct_eigfix.json`. The four values E7 had already
run reproduce to **0.00 accuracy points**, so the two windows are one continuous curve.

**Result 1 — extending the window located three peaks properly.** E9 had just established that not
one peak in the study was separated from its neighbour by more than the seed floor. Three now are:

| mode | reference | 1e-10 | 1e-11 | 3e-12 | 1e-12 | best | gain | margin over runner-up |
|---|---|---|---|---|---|---|---|---|
| `tkfac` | 72.68 | **79.01** | 73.42 | 66.31 | 59.64 | 1e-10 | +6.33 | **5.59** |
| `tekfac` | 72.98 | **80.10** | 78.00 | 72.87 | 67.62 | 1e-10 | +7.12 | **2.10** |
| `ekfac` | 72.49 | **80.04** | 78.09 | 72.79 | 67.28 | 1e-10 | +7.55 | **1.95** |
| `kfac` | 73.05 | 78.83 | 80.90 | **81.17** | 80.27 | 3e-12 | +8.12 | 0.27 |
| `diag` | 73.05 | 46.15 | 37.58 | 33.72 | 30.29 | — | −26.90 | — |

Against a seed floor of 0.04 to 0.18 points, margins of 1.95, 2.10 and 5.59 are real. So the answer
to E9's blocking problem is not only seeds: **widen the window until the curve falls off on both
sides**, and the peak locates itself. E7's grids stopped while three of these were still climbing,
which is exactly why they could not be pinned.

`kfac` is the exception twice over. Its peak sits at 3e-12, thirty times below the other three
modes' on the same network, and its margin is 0.27 — still inside the noise band. It is also the
only one of the four with no eigenbasis, so Result 2 does not touch it.

**Result 2 — and this corrects earlier results in this document.** The eigenbasis-ordering defect
that `CLAUDE.md` section 3 documents costs real accuracy once the safety constant is lowered:

| mode | at the default constant | 1e-10 | 1e-11 | 3e-12 | 1e-12 |
|---|---|---|---|---|---|
| `ekfac` | +0.79 | **+1.79** | **+3.30** | **+3.89** | **+3.05** |
| `tekfac` | +0.03 | **+1.65** | **+3.23** | **+4.49** | **+3.19** |

At the shipped constant it is worth +0.79 and +0.03, which is why no trained model in this
repository was affected and why the audit could call it inert. Four to nine orders of magnitude
lower it is worth **1.6 to 4.5 accuracy points**. `CLAUDE.md` says in so many words to fix the
ordering before acting on fix S1, which lowers the constant; this is that warning, measured on a
real network.

**Every `ekfac` and `tekfac` number at a small safety constant in E7 and in Result 1 above is
therefore understated**, by 1.6 to 4.5 points. They are kept as measured, and this table is what
converts them.

**Result 3 — the best result in the study.**

| accuracy | mode | constant | ordering | its own reference | gain |
|---|---|---|---|---|---|
| **81.83%** | `ekfac` | 1e-10 | corrected | 73.28 | **+8.55** |
| 81.75% | `tekfac` | 1e-10 | corrected | 73.01 | +8.74 |
| 81.17% | `kfac` | 3e-12 | shipped | 73.05 | +8.12 |

Three different modes, two different orderings, two constants thirty times apart, all landing within
0.7 points of each other around 81.5%, from references around 73. Whatever the right rule for the
constant is, the *ceiling* it reaches on this network looks mode-independent.

One caveat on Result 3, since it invites a comparison it does not support: these are test
accuracies at a fixed 15 epochs and batch 32. The campaign's own numbers on this model are
validation accuracies under the wall-clock-budget protocol. The two are not the same measurement
and should not be put in one table.

**A note on the corrected ordering's own peaks.** With the ordering fixed, `ekfac` reads 81.83 at
1e-10 against 81.39 at 1e-11, and `tekfac` 81.75 against 81.23 — margins of 0.44 and 0.52, back
inside the band where a peak is not located. Fixing the defect raised the whole curve and flattened
its top. So Result 1's three located peaks are located for the *shipped* ordering, and locating them
for the corrected one needs either seeds or a finer grid between 1e-10 and 1e-11.

### E10-E12 — done. The first located peak, a floor eight times larger than assumed, and a defect that was hiding EKFAC's own theorem.

Eleven jobs (21446986-97), 15 to 48 minutes each, all COMPLETED. Five seeds on
`cct_2_3x2_cifar` with the corrected eigenbasis ordering (E10), five seeds on its `kfac` alone
(E11), and one seeded run extending `vit_micro_cifar`'s window downward (E12). Outputs:
`fisher_ref/outputs/e1{0,1,2}_*.json`. Seed 0 of E10 reproduces E8's single-seed numbers to
**0.0000 accuracy points**, so the two are one experiment.

**Result 1 — the seed floor on this network is eight to thirty-five times larger than the one this
document had been borrowing.** Each of the ten per-seed jobs re-ran its own reference arm:

| mode | s0 | s1 | s2 | s3 | s4 | spread | sd |
|---|---|---|---|---|---|---|---|
| `ekfac` | 73.28 | 73.03 | 72.42 | 73.21 | 72.72 | 0.86 | 0.36 |
| `tekfac` | 73.01 | 72.85 | 72.49 | 73.23 | 72.92 | 0.74 | 0.27 |
| `kfac` | 73.05 | 73.43 | 72.35 | 73.76 | 73.00 | **1.41** | 0.53 |

The floor is **0.74 to 1.41 points**, not the 0.04 to 0.18 measured on `cnn_gn_cifar` and
`resnet20_cifar` and used everywhere above for want of anything better. Every "resolved" verdict in
E8 has to be re-read against it: `tkfac`'s margin of 5.59 survives, `ekfac`'s 1.95 and `tekfac`'s
2.10 become marginal, and `kfac`'s 0.27 was never resolved. **A noise floor does not transfer
between networks**, and this is the second time in this document that borrowing one has misled.

**Result 2 — the first properly located peak in the study, and it is not where one seed said.**
With five seeds and the ordering corrected, on `cct_2_3x2_cifar`:

| mode | ref | 3e-10 | 1e-10 | **3e-11** | 1e-11 | 3e-12 | 1e-12 |
|---|---|---|---|---|---|---|---|
| `ekfac` | 72.93 | 78.85 ±0.19 | 81.57 ±0.10 | **82.85 ±0.14** | [81.39] | [76.68] | [70.33] |
| `tekfac` | 72.90 | 79.06 ±0.18 | 81.59 ±0.09 | **82.88 ±0.18** | [81.23] | [77.36] | [70.81] |

(± is the standard error over five seeds; bracketed values are E8's single seed.)

The peak is at **3e-11**, falling away on both sides, separated from its runner-up by **7.4 and 6.4
standard errors**. E8 put it at 1e-10 because its grid went 1e-10, 1e-11 and skipped the point
between. **+9.91 and +9.98 accuracy points over their own references — the largest gains in the
study**, and the first ones resting on a peak that is statistically located rather than assumed.

**Result 3 — `kfac` has no peak. It has a plateau, and the anomaly dissolves.**

| `λ` | 1e-11 | 3e-12 | 1e-12 | 3e-13 |
|---|---|---|---|---|
| accuracy, 5 seeds | 80.16 ±0.22 | **80.40 ±0.24** | 79.96 ±0.28 | 77.89 ±0.22 |

The top three sit within 0.44 points and the best beats its runner-up by **0.7 standard errors**:
not separable. So `kfac` does not want 3e-12 specifically, it is flat across a decade. The question
"why does this one mode want a constant thirty times smaller than the others" was built on a peak
that is not there.

**Result 4 — and this is the one that matters. The ordering defect was hiding EKFAC's own theorem.**
`kfac`'s ceiling on this network is **80.40**. `ekfac`'s and `tekfac`'s are **82.85 and 82.88** —
two and a half points higher. Before the ordering was corrected, at one seed, `kfac` read 81.17 and
`ekfac` 80.04: KFAC looked **better**. EKFAC exists because its rescaling is provably at least as
good as K-FAC's in the same basis (`ekfac_1806.03884.pdf` Theorems 2 and 3), and measuring that
rescaling in a basis that is about to be replaced was costing exactly that advantage. The property
only becomes visible once the safety constant is low enough for the rescaling to matter at all,
which is why nothing in this repository had ever seen it.

**Result 5 — E12 extended `vit_micro_cifar`'s window in the wrong direction.** With the ordering
corrected, three of its four modes peak at the **top** of the new window:

| mode | ref | 1e-10 | 1e-11 | 3e-12 | 1e-12 | best | gain |
|---|---|---|---|---|---|---|---|
| `kfac` | 45.35 | 51.84 | **54.23** | 53.08 | 50.92 | 1e-11, interior | +8.88 |
| `tekfac` | 46.33 | **54.92** | 49.35 | 43.52 | 39.15 | 1e-10, top edge | +8.59 |
| `ekfac` | 46.32 | **54.36** | 47.91 | 43.10 | 39.38 | 1e-10, top edge | +8.04 |
| `tkfac` | 46.06 | **49.20** | 40.74 | 36.53 | 31.32 | 1e-10, top edge | +3.14 |

Correcting the ordering moves this network's optimum *up*, so the window needed extending above
1e-10, not below. What it does establish: the gains on `vit_micro_cifar` went from E7's +2.19 and
+2.98 for `ekfac`/`tekfac` to **+8.04 and +8.59**. `kfac`'s interior peak at 1e-11 has a margin of
1.15 points at one seed, which is inside this study's measured floor, so it is not located either.

**Result 6 — `tau`, at the one peak that is located.** Reading the curvature at steps 1000 to 8000
gives `tau` between **1.0e-6 and 1.6e-6** for the located peak. The earlier figure of 1.7e-5, built
from peaks that E9 showed were not located and from `ekfac`/`tekfac` runs carrying the ordering
defect, is **twenty times larger**. One located determination is not a rule, but it does mean the
constant this document has been quoting was wrong.

**What is left, stated plainly.** One peak in the study is located. Locating a second needs the same
treatment somewhere else: the corrected ordering, five seeds, and a window wide enough that the
curve falls on both sides. `vit_micro_cifar` is the obvious candidate and needs its window moved up,
to roughly {1e-8, 3e-9, 1e-9, 3e-10, 1e-10}.

### E13 — done. The optimum is a plateau, not a point — and `tau` agrees across two architectures for the two modes that share a damping rule.

Five jobs (21468492-96), 1 h 27 each, all COMPLETED. `vit_micro_cifar`, corrected eigenbasis
ordering, five seeds, and a window widened to eight values at half-a-decade spacing:
{1e-8, 3e-9, 1e-9, 3e-10, 1e-10, 3e-11, 1e-11, 3e-12}. The half-decade spacing is the direct lesson
of E10, where the real peak sat at 3e-11, a point the earlier grid skipped by stepping a full decade.
Output: `fisher_ref/outputs/e13_seeds_vit_eigfix_s*.json`.

**Result 1 — a second network, a second floor, and it is larger again.**

| mode | s0 | s1 | s2 | s3 | s4 | spread |
|---|---|---|---|---|---|---|
| `tekfac` | 46.33 | 47.65 | 46.40 | 45.24 | 45.51 | **2.41** |
| `ekfac` | 46.32 | 47.54 | 46.40 | 45.24 | 45.52 | 2.30 |
| `tkfac` | 46.06 | 45.69 | 45.96 | 45.29 | 44.45 | 1.61 |
| `kfac` | 45.35 | 45.80 | 46.42 | 46.21 | 45.00 | 1.42 |

**1.42 to 2.41 points**, against `cct_2_3x2_cifar`'s 0.74 to 1.41 and the 0.04 to 0.18 this document
borrowed for most of its length. Two networks measured, two very different floors, both far above
the borrowed one. Treat every accuracy difference under about two points on these networks as
unresolved unless it has been measured with seeds.

**Result 2 — the curves, five-seed means with their standard error.**

| mode | ref | 1e-8 | 3e-9 | 1e-9 | 3e-10 | 1e-10 | 3e-11 | 1e-11 | 3e-12 |
|---|---|---|---|---|---|---|---|---|---|
| `kfac` | 45.76 | 45.59 | 46.68 | 48.73 | 49.82 | 51.68 | 52.43 | **52.98** ±0.51 | 52.83 |
| `ekfac` | 46.20 | 47.40 | 48.92 | 51.30 | 54.21 | **55.24** ±0.27 | 52.76 | 48.70 | 44.27 |
| `tkfac` | 45.49 | 49.37 | 50.15 | 49.74 | **50.25** ±0.61 | 48.46 | 44.30 | 39.87 | 35.63 |
| `tekfac` | 46.23 | 46.97 | 48.77 | 51.03 | 54.58 | **54.99** ±0.08 | 53.21 | 49.26 | 44.48 |

Best gains: `ekfac` **+9.03**, `tekfac` **+8.77**, `kfac` +7.23, `tkfac` +4.76. All far above the
2.41 floor.

**Result 3 — and this is why no peak here is "located": the optimum is a plateau.** Listing every
constant whose five-seed mean is within one standard error of the best:

| mode | plateau | width |
|---|---|---|
| `ekfac` | 1e-10 alone | 1x (nearest neighbour 1.6 standard errors below) |
| `tekfac` | 3e-10 .. 1e-10 | 3x |
| `kfac` | 3e-11 .. 3e-12 | 10x |
| `tkfac` | 3e-9 .. 3e-10 | 10x |

More seeds will not turn these into points, because the curve genuinely has a flat top. **`tau`
therefore cannot be pinned to better than the width of the plateau it is read from** — a factor of
three to ten — and that is a property of the problem, not of the measurement. Only `ekfac`'s optimum
is nearly sharp, and it is the one that would separate first with more seeds.

**Result 4 — the positive result. `tau` agrees across two architecture families, for the two modes
that share a damping rule.**

| network | mode | best `λ` | mean curvature | `tau` |
|---|---|---|---|---|
| `cct_2_3x2_cifar` | `ekfac` | 3e-11 | 2.21e-5 | 1.36e-6 |
| `cct_2_3x2_cifar` | `tekfac` | 3e-11 | 2.09e-5 | 1.44e-6 |
| `vit_micro_cifar` | `ekfac` | 1e-10 | 1.69e-4 | 5.91e-7 |
| `vit_micro_cifar` | `tekfac` | 1e-10 | 1.69e-4 | 5.91e-7 |

A compact transformer and a convolutional tokeniser network, curvatures a factor of eight apart,
and `tau` agrees to **2.4x** — inside the three-to-tenfold width of the plateaus it is read from.
Within each network the two modes agree exactly.

**Result 5 — but `tau` is not one constant across modes, and the split is structural.** On
`vit_micro_cifar`, where all four were measured at five seeds:

| mode | `tau` |
|---|---|
| `tkfac` | 1.87e-6 |
| `ekfac`, `tekfac` | 5.91e-7 |
| `kfac` | 6.03e-8 |

Thirty times apart, on one network, with curvatures within 6% of each other. So the four modes
genuinely want different constants, and the split follows the damping algebra: `ekfac` and `tekfac`
add `λ` to a per-direction rescaling and agree exactly; `kfac` and `tkfac` add it inside factors that
are then inverted, and land elsewhere. **Fix S1 needs one constant per damping rule, not one
constant.** That is a sharper and more implementable statement than anything earlier in this
document, and it is the first version of S1 that two independent networks support.

### E14 — pre-registered: does `tau` predict the best safety constant on two networks it was not fitted on?

Written and committed **before** the runs were submitted.

**The question.** E10 and E13 located the best safety constant on two networks and read `tau`
from it. `tau` is the best constant divided by the network's mean curvature. The two readings agree
across those two networks for `ekfac`/`tekfac`, and the four modes split along their damping rule.
Two networks are enough to *suggest* a rule, not to test one. This experiment tests it on two
networks that played no part in fitting `tau`: `cnn_gn_cifar` and `resnet20_cifar`.

**Where each `tau` comes from.** Every value below is a five-seed result from E10, E11 or E13, with
the ordering corrected, except the upper end of `tkfac`'s range. That one is E8's single-seed
`cct_2_3x2_cifar` peak, whose margin of 5.59 points survives E10's measured floor. `tkfac` has no
eigenbasis, so the ordering defect does not touch it.

| mode | `tau`, low end | `tau`, high end | read from |
|---|---|---|---|
| `ekfac` | 5.9e-7 | 1.4e-6 | `vit_micro_cifar` (E13), `cct_2_3x2_cifar` (E10) |
| `tekfac` | 5.9e-7 | 1.4e-6 | the same |
| `tkfac` | 1.9e-6 | 4.1e-6 | `vit_micro_cifar` (E13), `cct_2_3x2_cifar` (E8, one seed) |
| `kfac` | 6.0e-8 | 1.2e-7 | `vit_micro_cifar` (E13), `cct_2_3x2_cifar` (E11) |

**Which curvature the prediction uses.** It must be the same statistic `tau` was divided by, which is
E9's mean curvature at step 8000, at batch 32. E9 measured it on `cnn_gn_cifar`, per mode: `kfac`
4.09e-4, `ekfac` 3.33e-4, `tkfac` 4.58e-4, `tekfac` 4.16e-4. E9 did not measure `resnet20_cifar`.
For it, the only measurement is E1's mid-training range, 2.0e-5 to 2.7e-5. On the two networks where
both exist, E1 reads 1.3 to 1.8 times higher than E9 at step 8000. So the `resnet20_cifar`
predictions may be up to 1.8 times too high. That is less than one step of the grid below.

**The predictions.** Each is the `tau` range multiplied by that network's curvature.

| network | `ekfac` | `tekfac` | `tkfac` | `kfac` |
|---|---|---|---|---|
| `cnn_gn_cifar` | 2.0e-10 .. 4.5e-10 | 2.5e-10 .. 6.0e-10 | 8.6e-10 .. 1.9e-9 | 2.5e-11 .. 4.7e-11 |
| `resnet20_cifar` | 1.2e-11 .. 3.7e-11 | 1.2e-11 .. 3.9e-11 | 3.7e-11 .. 1.1e-10 | 1.2e-12 .. 3.1e-12 |

**The competing hypothesis.** It says the best constant is a fixed number, and not proportional to
the curvature. On `cnn_gn_cifar` that number is 1e-8: E4 and E7 put all four modes' peaks there, at
one seed, with the ordering defect still in place. This is 1.3 to 2.6 orders of magnitude away from
the predictions for `ekfac`, `tekfac` and `kfac`, and about one order away for `tkfac`. The two
hypotheses can therefore be told apart. On `resnet20_cifar`, E4 saw every mode flat to within a point
from 1e-4 down to 1e-10. That window stopped above every prediction in the table.

**The protocol.** Everything is E13's, except the network and the window:

- batch 32 and 15 epochs;
- the shipped estimator and the corrected eigenbasis ordering (`E4_EIG_BEFORE_RESCALE=1`);
- the learning rate moved together with the constant, so the step-size cap stays still (Part 5,
  rule 1);
- the parameters the optimizer does not precondition stay trainable;
- each seed is its own job, and each job re-runs its own reference arm.

| network | seeds | window, half-decade spacing | runs per seed | measured cost per run |
|---|---|---|---|---|
| `cnn_gn_cifar` | 0-4 | 1e-7, 3e-8, ..., 3e-12 (10 values) | 44 | ~71 s (E4, job 21295804) |
| `resnet20_cifar` | 0-2 | 1e-8, 3e-9, ..., 3e-13 (10 values) | 44 | ~280 s (E4, job 21295805) |

Each window contains the competing hypothesis's 1e-8 and reaches at least a factor of 3 below the
lowest prediction. The curve can therefore fall off on both sides, which E8 showed is what locates a
peak. `resnet20_cifar` gets three seeds rather than five because one seed costs about 3.4 hours.

**The decision rules, fixed now.** They use E13's definition of a plateau: every constant whose
seed-mean accuracy lies within one standard error of the best seed-mean.

1. **Per (network, mode):** the prediction is **confirmed** if the plateau intersects the predicted
   range widened by a factor of 3 on each side. That is one grid step, and it is also the width E13
   measured for the narrowest plateaus. It is **refuted** if the plateau does not intersect that
   range. It is **unresolved** if the best constant sits at the edge of the window.
2. **The competing hypothesis** is supported on `cnn_gn_cifar` for a mode if that mode's plateau
   contains 1e-8.
3. **The rule is supported** if the prediction is confirmed for `ekfac` and `tekfac` on both
   networks. This is the claim fix S1 would rest on. It is **refuted** if it fails for either of
   the two on either network.
4. **The split by damping rule is supported** if, on each network, `kfac`'s plateau lies entirely
   below `ekfac`'s and `tekfac`'s, and those two plateaus overlap each other.

What these rules do not settle: a network whose curve is flat across the whole window (as
`resnet20_cifar` was down to 1e-10) produces a plateau that intersects everything. That case is
reported as **uninformative**, not as a confirmation.

**Submitted** on 21 September 2026, after the commit above. Jobs 21498168-72 run `cnn_gn_cifar`,
seeds 0-4, with a 1:45 limit. Jobs 21498173-75 run `resnet20_cifar`, seeds 0-2, with a 6:00 limit.
Each log confirms its window and `eig_before_rescale=True`. Outputs:
`fisher_ref/outputs/e14_seeds_{cnn,resnet20}_eigfix_s<seed>.json`. The first `cnn_gn_cifar` runs
take 68 to 70 s each, as E4 measured.

### E14 — done. The pre-registered rule passes for `ekfac`/`tekfac`, but the test was too weak: the best constant barely follows the curvature.

All eight jobs COMPLETED: `cnn_gn_cifar` in 52 to 53 min per seed, `resnet20_cifar` in 3 h 32 to
3 h 35. Nothing crashed, including `kfac` and `tkfac` down to 3e-13.

**Result 1 — the curves.** Test accuracy in %, seed mean ± standard error. `ref` is each mode at
its default constant (1e-3).

`cnn_gn_cifar`, five seeds:

| mode | ref | 1e-7 | 3e-8 | 1e-8 | 3e-9 | 1e-9 | 3e-10 | 1e-10 | 3e-11 | 1e-11 | 3e-12 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kfac` | 61.10 | 63.74 | 64.00 | 64.47 | 64.46 | 65.04 | 64.93 | **65.52** ±0.18 | 64.64 | 63.40 | 61.40 |
| `ekfac` | 61.60 | 63.06 | 62.71 | 63.13 | 63.93 | 65.59 | 67.32 | 68.15 | **68.23** ±0.25 | 65.67 | 61.39 |
| `tkfac` | 62.04 | 62.52 | **63.04** ±0.27 | 62.39 | 62.33 | 61.08 | 59.68 | 58.16 | 57.08 | 55.81 | 52.69 |
| `tekfac` | 61.30 | 62.23 | 63.10 | 63.51 | 64.71 | 65.82 | 67.21 | **68.20** ±0.23 | 67.99 | 65.87 | 61.77 |

`resnet20_cifar`, three seeds:

| mode | ref | 1e-8 | 3e-9 | 1e-9 | 3e-10 | 1e-10 | 3e-11 | 1e-11 | 3e-12 | 1e-12 | 3e-13 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `kfac` | 86.68 | 86.87 | **86.88** ±0.18 | 86.55 | 86.47 | 86.75 | 86.75 | 86.87 | 86.71 | 86.76 | 86.62 |
| `ekfac` | 86.27 | 86.37 | 86.29 | 86.17 | 86.28 | 86.68 | 86.54 | **86.84** ±0.23 | 84.87 | 78.03 | 65.13 |
| `tkfac` | 86.60 | 86.36 | 86.20 | **86.57** ±0.03 | 86.36 | 86.31 | 85.96 | 84.65 | 80.04 | 70.52 | 59.72 |
| `tekfac` | 86.36 | 86.36 | 86.39 | 86.32 | 86.11 | 86.32 | 86.46 | **86.98** ±0.32 | 85.57 | 79.61 | 66.09 |

On `cnn_gn_cifar` the gains are large: **+6.6 and +6.9 points** for `ekfac` and `tekfac`, +4.4 for
`kfac` and +1.0 for `tkfac`. On `resnet20_cifar` they are small: **+0.6** at best, and zero for
`kfac` and `tkfac`. That matches E4, which found this network flat to within a point.

**Result 2 — the pre-registered verdicts.** A plateau is every constant whose seed mean lies within
one standard error of the best seed mean (rule 1).

| network | mode | plateau | predicted | predicted, widened 3x | verdict |
|---|---|---|---|---|---|
| `cnn_gn_cifar` | `ekfac` | 1e-10, 3e-11 | 2.0e-10 .. 4.5e-10 | 6.7e-11 .. 1.4e-9 | **confirmed**, at the lower edge |
| | `tekfac` | 1e-10, 3e-11 | 2.5e-10 .. 6.0e-10 | 8.3e-11 .. 1.8e-9 | **confirmed**, at the lower edge |
| | `kfac` | 1e-10 | 2.5e-11 .. 4.7e-11 | 8.3e-12 .. 1.4e-10 | **confirmed**, at the upper edge |
| | `tkfac` | 3e-8 | 8.6e-10 .. 1.9e-9 | 2.9e-10 .. 5.7e-9 | **refuted** |
| `resnet20_cifar` | `ekfac` | 1e-10, 1e-11 | 1.2e-11 .. 3.7e-11 | 4.0e-12 .. 1.1e-10 | **confirmed** |
| | `tekfac` | 1e-11 | 1.2e-11 .. 3.9e-11 | 4.0e-12 .. 1.2e-10 | **confirmed** |
| | `kfac` | 7 of 10 values, 1e-8 .. 1e-12 | 1.2e-12 .. 3.1e-12 | | **uninformative**: flat |
| | `tkfac` | 1e-9 | 3.7e-11 .. 1.1e-10 | 1.2e-11 .. 3.3e-10 | **refuted**, but fragile |

`kfac` on `resnet20_cifar` is the case the pre-registration set aside: all ten values lie within
0.41 points of each other, so its plateau intersects everything and says nothing. `tkfac` on
`resnet20_cifar` is refuted only because its best cell has an unusually small standard error
(0.03, from three seeds). With a two-sample criterion, which was *not* pre-registered, its plateau
becomes {1e-9, 3e-10} and the verdict flips to confirmed. Its substantive result is simpler: no
value of the constant does better than the default. On `cnn_gn_cifar` the refutation of `tkfac` is
clear: at the predicted value, 1e-9, it reads 61.08%, 2 points below its best and 1 point below its
own default.

- **Rule 2, the fixed-number hypothesis at 1e-8: refuted for all four modes.** No plateau on
  `cnn_gn_cifar` contains 1e-8. E4 and E7's single-seed peaks at 1e-8 came from windows that
  stopped at 1e-9 or 1e-10, and for `ekfac`/`tekfac` from the ordering defect as well.
- **Rule 3: the rule is supported** by the pre-registered criterion. `ekfac` and `tekfac` are
  confirmed on both networks.
- **Rule 4: the split by damping rule is not supported.** `ekfac` and `tekfac` share their plateau
  on both networks, as on the two earlier ones. But on `cnn_gn_cifar` `kfac`'s plateau (1e-10) is
  not below theirs (1e-10, 3e-11). On `vit_micro_cifar` and `cct_2_3x2_cifar` it was 10 to 30 times
  lower. On `resnet20_cifar` `kfac` is flat.

**Result 3 — and why the pass is weaker than it looks. The competitor was the wrong one.** This
comparison was *not* pre-registered, so read it as a finding to test, not as a verdict. The
competing hypothesis written down was a fixed constant of 1e-8. That number came from runs carrying
the ordering defect and windows that stopped too high, so it was easy to refute. The fair competitor
is a fixed constant fitted on the same two networks `tau` was fitted on. With the ordering
corrected, those two networks put `ekfac`/`tekfac` at 3e-11 and 1e-10, whose geometric mean is
5.5e-11. Here is how both rules predict the two new networks:

| mode | network | fixed 5.5e-11 is off by | proportional to curvature is off by |
|---|---|---|---|
| `ekfac` | `cnn_gn_cifar` | 1.0x (inside the plateau) | 3.0x |
| `tekfac` | `cnn_gn_cifar` | 1.0x (inside the plateau) | 3.8x |
| `ekfac` | `resnet20_cifar` | 1.0x (inside the plateau) | 1.0x (inside the plateau) |
| `tekfac` | `resnet20_cifar` | 5.5x | 2.1x |

`cnn_gn_cifar` is the network that tells the two rules apart. Its mean curvature is 15 to 20 times
`cct_2_3x2_cifar`'s, while `resnet20_cifar`'s is about equal to it. On `cnn_gn_cifar` the fixed
constant lands inside the plateau, and the proportional rule misses it by a factor of 3 to 4. The
proportional rule passed rule 1 only because of the factor-3 widening.

Across all four networks now measured, the same picture:

| mode | curvature, smallest to largest | best constant, smallest to largest | slope of log(best constant) against log(curvature) |
|---|---|---|---|
| `ekfac` | 15x | 3.3x (3e-11 .. 1e-10) | **+0.32** |
| `tekfac` | 20x | 10x (1e-11 .. 1e-10) | **+0.50** |

A proportional rule predicts a slope of 1, and a fixed constant predicts 0. The spread of the
optimum around a single fixed value is 0.21 decades for `ekfac` and 0.37 for `tekfac`. Around the
best proportional rule it is 0.38 and 0.37. **Dividing by the curvature does not make the optimum
more constant. For `ekfac` it makes it less constant.** Four points, each uncertain by roughly the
width of its plateau (one to ten times), cannot pin the slope. They do say that on these networks
the best constant for `ekfac`/`tekfac` sits between **1e-11 and 1e-10** whatever the curvature.

One qualification is essential. Every run of E8 to E14 used **batch 32**, and E3 showed that the
stored curvature scales as 1/batch². So a constant that is fixed in stored units is fixed only at
this batch size. What E14 argues against is scaling with the *network's* curvature, not scaling with
the batch.

**Result 4 — EKFAC's advantage over K-FAC holds on a third network.** The best seed mean for each
mode:

| network | `kfac` | `ekfac` | `tekfac` | `ekfac − kfac` |
|---|---|---|---|---|
| `cct_2_3x2_cifar` (E10, E11) | 80.40 | 82.85 | 82.88 | **+2.45** |
| `vit_micro_cifar` (E13) | 52.98 | 55.24 | 54.99 | **+2.26** |
| `cnn_gn_cifar` (E14) | 65.52 | 68.23 | 68.20 | **+2.71** |
| `resnet20_cifar` (E14) | 86.88 | 86.84 | 86.98 | −0.04 |

That is the same advantage, 2.3 to 2.7 points, on three networks. It is a tie on the one network
where nothing responds to the constant. EKFAC's Theorems 2 and 3 (`ekfac_1806.03884.pdf`) are
about approximation error, not accuracy, so this agrees with them without testing them.

**Result 5 — the seed floor, measured on two more networks.** Over five seeds, the four reference
arms of `cnn_gn_cifar` spread by **0.44 to 2.19 points**. Over three seeds, `resnet20_cifar`'s spread
by 0.15 to 0.45. The 0.04 to 0.18 this document borrowed for most of its length came from these
same two networks: two seeds of `kfac` at its default constant (`validation_noise_investigation.md`,
Step 9). For `kfac`, five seeds give 0.44 on `cnn_gn_cifar` against the 0.04 borrowed, i.e. ten
times larger, and 0.15 on `resnet20_cifar` against 0.18. So two seeds understated `cnn_gn_cifar`'s
floor tenfold, while `resnet20_cifar`'s held.

**What E14 changes.**

1. **Fix S1 has lost its evidence.** A safety constant proportional to each layer's mean curvature
   is not better supported than a fixed constant, for any mode. The simpler candidate is a fixed
   `λ` for `ekfac`/`tekfac` around 3e-11 to 1e-10 in stored units at batch 32. Fix S4 (restoring
   the per-example scale) would make that number independent of the batch size.
2. **The prediction was right where it mattered least.** `ekfac`/`tekfac` land in the predicted
   decade on all four networks. But a fixed constant lands there too, so this confirms nothing
   about curvature.
3. **`tkfac` is the outlier.** Its optimum ranges from 3e-10 (`vit_micro_cifar`) to 3e-8
   (`cnn_gn_cifar`). Where it gains at all, it gains least: +1.0 and 0.0 here, +4.8 on
   `vit_micro_cifar`.

**Limits.** `resnet20_cifar` has three seeds, and its curvature is E1's mid-training value, which
read 1.3 to 1.8 times higher than E9's step-8000 value on the two networks where both exist. The
plateau definition is sensitive to a single cell's standard error, as `tkfac` on `resnet20_cifar`
shows. `cct_2_3x2_cifar`'s five-seed window for `ekfac`/`tekfac` stops at 3e-11, its best point.
The drop below it rests on E8's single seed.

### E15 — pre-registered: does one safety constant per layer beat the best single one?

Written **before** any code for it exists and before any run. Anything below that changes after the
runs are submitted is a deviation, and will be recorded as one.

**The question.** E7 to E14 always used one `λ` for the whole network. S1's second claim is that
each layer needs its own (S1's status paragraph, Part 4). This experiment tests that claim directly.
Inside one network, at an equal step-size cap, does `λ_l = τ × c̄_l`, where `c̄_l` is layer `l`'s own
mean stored curvature, beat the best single `λ`?

**Two versions of S1, and which one is tested.** Write `m_i` for the momentum in direction `i`,
`s_i` for the stored curvature there, and `cap` for the step multiplier (`base_lr / base_λ`: 1 on
`cnn_gn_cifar`, 1/3 on `vit_micro_cifar`). Shown for `ekfac`/`tekfac`, where `λ` is added direction by
direction:

```
single λ, cap held (E7-E14):    step_i = cap · m_i / (1 + s_i / λ)
S1-b, cap held per layer:       step_i = cap · m_i / (1 + s_i / (τ · c̄_l))
S1-a, the papers' literal form: step_i = lr  · m_i / (s_i + τ · c̄_l)
```

- **S1-b is tested here.** Every layer keeps the same cap, so the only thing that changes compared
  with a single `λ` is where the threshold sits in each layer. It has one knob, `τ`, against the
  single-`λ` arm's one knob, `λ`. That is a fair comparison, and it respects Part 5's rule 1.
- **S1-a is not tested here.** With one learning rate for the network, a per-layer `λ_l` also gives
  each layer its own cap, `lr / λ_l`, so it needs a two-dimensional sweep over `τ` and `lr`. It is a
  follow-up, and only if S1-b wins.

**How `c̄_l` is read.** It is read from the state the optimizer already stores, at every refresh, so
it needs no extra decomposition. `ekfac`: `mean(s*)`. `tekfac`: `mean(Θ)`, which is exactly TEKFAC's
Eq. (3.5) without its floor `ϑ`. `kfac`: `(tr A / d_in) × (tr B / d_out)`, the mean eigenvalue of
`A ⊗ B`. `tkfac`: `δ / (d_in × d_out)`, since `tr Φ = tr Ψ = 1`. Early in training `c̄_l` still
contains the identity's residue, and so does the curvature it is compared with. That is equally true
of the single-`λ` arm, and is left as it ships.

**What `τ` should be, roughly.** The single-`λ` optima already measured translate into a
network-level `τ`: the optimal `λ` divided by the network's mean stored curvature (E9's step-8000
values, divided by 115 × 32² for `ekfac`/`tekfac` and by 115² × 32² for `kfac`). This comes out at
**0.008 to 0.035** on `cnn_gn_cifar` and **0.07 to 0.21** on `vit_micro_cifar` for `ekfac`/`tekfac`,
and at **0.25 to 3.3** for `kfac`. `kfac` is higher because its constant enters inside the two
factors, split as `π√λ` and `√λ/π`, not added once. S1-b's best `τ` need not equal these values,
because it divides by each layer's own mean and not by the network's. The grids are built to reach
at least a factor of 4 past these values on either side.

**The arms,** for each (network, mode, seed):

| arm | what varies | grid | runs |
|---|---|---|---|
| reference | nothing: the default `λ` | — | 1 |
| single `λ` | one `λ` for the network, cap held | 5 values at half-decade spacing, centred on E13/E14's best (below) | 5 |
| **S1-b** | `λ_l = τ · c̄_l` for each layer, cap held per layer | `ekfac`/`tekfac`: `τ` in {1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4}. `kfac`: {30, 10, 3, 1, 0.3, 0.1, 0.03, 0.01} | 8 |
| network-adaptive | one `λ(t) = τ · c̄_net(t)` for the whole network, where `c̄_net` is the mean over all its layers' directions, cap held | the same grids as S1-b | 8 |

`τ = 1` in the `ekfac`/`tekfac` grid is TEKFAC's own rule. The network-adaptive arm separates two
things S1-b does at once: following each *layer*, and following the curvature *over time*.

Single-`λ` grids: `cnn_gn_cifar` `kfac` {1e-9 .. 1e-11}, `ekfac`/`tekfac` {3e-10 .. 3e-12};
`vit_micro_cifar` `ekfac`/`tekfac` {1e-9 .. 1e-11}, `kfac` {1e-10 .. 1e-12}.

**Fixed across every arm.**
- Everything E13/E14 fixed: batch 32, 15 epochs, the cosine schedule, the shipped estimator,
  `eig_before_rescale=True` for `ekfac`/`tekfac`, five seeds 0-4 sharing initialisation and data
  order across arms.
- **The parameters the optimizer does not precondition are frozen in every arm**
  (`E4_FREEZE_UNHOOKED=1`). S1-b has no `λ_l` to give them, and with the cap convention they would
  otherwise step at `cap` times their momentum. E6 measured the freeze inert on `vit_micro_cifar`.
  On `cnn_gn_cifar` E5 measured it at −0.26 to +1.68 points, a constant here because it applies to
  every arm.
- **Weight decay at the benchmark's own rate, whatever `λ` is (rule 3).** `cnn_gn_cifar`'s decay is
  coupled and needs nothing. On `vit_micro_cifar`, every arm is built with a decay coefficient
  chosen so that `lr × wd` equals `base_lr × wd = 1e-5` per step at the top of the cosine.
- Modes: `ekfac` and `tekfac` (primary) and `kfac` (secondary). Not `tkfac`, which gained +1.0 and
  0.0 in E14. Not `diag`, which E7 showed wants the opposite of the others.
- Networks: `cnn_gn_cifar` (the largest spread between layers, 5 000×, and E14's five-seed single-`λ`
  curve to check against) and `vit_micro_cifar` (15 layers, 520 to 560×, dominated by the head).
  Not `resnet20_cifar`: it does not respond to `λ` at all, so it cannot tell two rules for `λ`
  apart.

**Controls that check the protocol itself.**
1. **Reproduction, seed 0.** For each mode, one single-`λ` cell run exactly as E13/E14 ran it
   (unfrozen, and on `vit_micro_cifar` with the decay that vanishes). It must reproduce E14's
   seed-0 test accuracy on `cnn_gn_cifar` (`kfac` 1e-10: 65.18, `ekfac` 3e-11: 67.41, `tekfac`
   1e-10: 68.55) and E13's on `vit_micro_cifar` (`kfac` 1e-11: 54.23, `ekfac` 1e-10: 54.36,
   `tekfac` 1e-10: 54.92), to 0.00 points. If one fails, the new code path changed something
   besides `λ`, and no result below is read until that is explained.
2. **The weight-decay confound, five seeds, `vit_micro_cifar` only.** For each mode, the best
   single-`λ` cell run unfrozen with the decay at its fixed rate. Paired by seed with E13's existing
   cell at the same `λ`, this measures how much of E13's gain was the decay switching off.

**The decision rules, fixed now.**
1. **Selection on validation, verdict on test.** Within each arm family, the chosen `λ` or `τ` is
   the one with the best five-seed mean of the final-epoch *validation* accuracy. The comparison is
   then made on *test* accuracy at those chosen values. Otherwise the best of eight values of `τ`
   would be compared with the best of five values of `λ` on the same numbers used to choose them,
   which favours S1-b.
2. **Per (network, mode):** `Δ` = test accuracy of S1-b minus test accuracy of the single `λ`, paired
   by seed. **S1-b wins** if the mean of `Δ` exceeds 2 standard errors of `Δ`, **loses** if it is
   below −2 standard errors, and **ties** otherwise.
3. **Verdict on S1 per layer.** **Adopted as the next fix** if S1-b wins for `ekfac` and for `tekfac`
   on both networks. **Dropped** if it ties or loses for both modes on both networks. Anything in
   between is **mixed**, and rule 5 decides what it means.
4. **Tuning-free?** S1-b is called tuning-free if one value of `τ` lies inside the plateau of every
   (network, mode) pair. The plateau uses E13's definition on test accuracy: every value whose
   five-seed mean lies within one standard error of the best. That is the property E14 found
   missing for a single `λ` across networks.
5. **Layer or time?** The same comparison as rule 2, between S1-b and the network-adaptive arm. If
   they tie, any gain of S1-b comes from following the curvature over time, not from treating
   layers differently.

**Recorded along the way, not voted on:** `c̄_l` and `λ_l` for every layer at every refresh, and the
fraction of each layer's directions whose curvature exceeds its `λ_l`. The expectation, written down
so it can fail: if S1-b wins, it is because the head gets a threshold far above the single-`λ`
optimum while the other layers stay near it. If that is what the logs show, the next arm is "single
`λ` for the body, relative `λ` for the head only".

**Cost, estimated from measured times per run** (68 to 70 s on `cnn_gn_cifar` in E14, 136 to 155 s
on `vit_micro_cifar` in E13). Per seed, 3 modes × 22 runs, plus the control cells. `cnn_gn_cifar` is
about 1 h 20 per seed, so a 2:15 limit. `vit_micro_cifar` is about 2 h 50 per seed, so a 4:00
limit. That is ten jobs and about 21 GPU-hours on `h100_1g.10gb` slices.

**Not submitted until three things exist:** the optimizer option (`damping="layer_relative"` /
`"network_relative"`, `damping_tau`), off by default and bit-identical when off; tests showing that
a constant `λ_l` reproduces the single-`λ` path and that `c̄_l` equals the mean eigenvalue computed
densely; and the driver `fisher_ref/experiments/e15_layer_damping.py`.

**Note added before submission, 21 September 2026. No rule changes.** All three now exist
(`ee8c86c`, `tests/test_relative_damping.py`, the driver). Locally, over one full epoch of
1 407 steps, the driver's repro cell is **bit-identical** to `e4_fixed_average.run_one` on both
networks: same test accuracy, same distance travelled by the parameters, same validation curve.

The local smoke run exposed one difference between the arms that the rules above do not name, so it
is written down here before any result exists. Every mode starts its stored curvature from the
identity, and that start fades as 0.08^k after `k` factor updates. A single `λ` of 1e-10 sits far
below that leftover until it has faded under the real curvature. Worked out on paper for
`cnn_gn_cifar`/`ekfac`, from a real stored curvature of about 3e-9: that takes about 8 updates, i.e.
about 800 steps, 4% of a run. Until then every direction moves by about `cap × λ / leftover`, which
is almost nothing. A *relative* `λ` scales with that leftover, so the S1-b and network-adaptive arms
move from the first step. In the smoke run (16 steps) they reached 17% to 19% on `vit_micro_cifar`
while every single-`λ` arm stayed at 11%. So a win of S1-b under rule 2 could come from being
relative at all, and not from being per-layer. **Rule 5 is the one that separates the two:** both of
the arms it compares are relative. A rule-2 win is read as "per-layer" only if rule 5 also favours
S1-b.

### E16 — pre-registered: a floor or a clip instead of the added constant?

**Amended on 2026-09-21, before any submission.** Family B now has three arms instead of one. The
first version had a single clip whose threshold was reset at every step (now the arm `clip`,
threshold `"quantile"`). Reviewing it found that the momentum's size then never reaches the step: it
is a per-layer normalisation with a clipped shape, and a verdict on it could not have said which of
the two mattered. Added: `clipema` (the main arm, whose step follows the momentum's size against its
own last ~1 000 steps) and `clipfixed` (Sophia's rule: one threshold for the whole network,
calibrated at step 2 000 and frozen). Also added: rule 6, and the `lr` control moved to the main
arm. Two changes came out of implementing them, both measured: the clipped fraction counts only the
coordinates whose momentum is above the guard, because rounding-level coordinates had placed the
threshold inside the noise; and the clip receives the bias-corrected momentum. Details in
`plan_floor_clip.md` §2 and §9.

**Second amendment, 2026-09-21, still before any submission: an audit.** Six agents audited the code,
the protocol and the analysis script on the real networks (`plan_floor_clip.md` §10). What changed
here, each for a measured reason:
- **Every clip threshold is per layer.** The stored curvature's scale error differs between layers by
  up to four orders of magnitude (the `1/T` of shared layers, the LayerNorm surrogate). A threshold
  shared by the whole network clipped a ViT's head on 3 % of its coordinates and its final LayerNorm
  on 97 %. So `clipfixed` now freezes, per layer, the median of its quantile over steps 1000-1999.
- **clipema's running average is bias-corrected.** Seeded with its first step, it stayed biased for
  12-18 % of a run, as large as the quantity rule 6 reads.
- **A repro gate at every seed.** Each job first reruns E14/E13/E10's best add cell of its own seed
  and stops unless it reproduces the stored cell bit for bit.
- **The decision script cannot turn missing data into a verdict** (the gate before rule 1, below).
  Rule 4's plateau is pinned to the reading E14 used, with E13's two-sample reading reported. Rule
  5's qualification is printed into rule 3.

**Third amendment, 2026-09-21, still before any submission: the normalisation statistic, corrected,
and E16's own baseline** (`plan_floor_clip.md` §11).
- **Every arm runs with `norm_exact_rescaling=True`.** The second amendment had recorded a limit:
  `ekfac`/`tekfac` estimate a normalisation layer's rescaling from the gradient of a surrogate built
  on the raw channel mean (`CLAUDE.md` §4.6), not from the layer's own gradient `[δ ⊙ x̂, δ]`. The
  option uses the layer's own gradient, projected into the same eigenbasis. EKFAC's Lemma 1 makes
  that the optimal diagonal in that basis. Measured at step 300, the scale-carrying column is
  295–989× larger than the surrogate's on `vit_micro_cifar` and 80–106× on `cct_2_3x2_cifar`. The
  shift column is unchanged (0.96–1.03×).
- **E16 therefore reruns add instead of reading E14's cells.** The correction reaches add's own
  step at E14's `λ`. Measured with `ekfac` at step 2 000 of the E-series protocol (CPU, seed 0):
  - under the shipped statistic, the divisor `s + λ` of every normalisation-layer coordinate is
    1.00 times `λ` (median), i.e. `λ` alone;
  - under the corrected one, on `vit_micro_cifar` at `λ = 1e-10`, the scale column's divisor is a
    median 2.45 times `λ` (90th percentile 4.5) on the first LayerNorm and at most 1.07 (1.16) on
    the other four;
  - on `cct_2_3x2_cifar` at `λ = 3e-11`, 2.21 (2.87) on the first LayerNorm, 1.94 (2.56) on the
    final one and 1.11-1.17 on the other three.

  The jobs also run on a different GPU slice, three runs at a time, which may change the
  floating-point order. So add runs in the same jobs as the families, on the same five-value grids,
  and `addfill` is gone.
- **The repro gate becomes a diagnostic, and two gates replace it, at seed 0.** *Determinism:* one
  add cell runs twice, in two processes of the same job, and the two must agree on every recorded
  field. *Inertness:* on `cnn_gn_cifar`, which has no hooked normalisation layer, the add cell must
  be bit-identical with and without the option. The comparison with E14's stored cell is reported
  and does not vote.
- **The jobs.** One per (network, seed, mode): 30 in all, each running three processes on one
  `h100_3g.40gb` slice. Projected under an hour each, against 2 to 5 hours before.

The rest of this section is the thrice-amended pre-registration.

Written **before** any cluster run. The code, its tests and the driver exist (they are what makes
the rules below checkable); nothing has been submitted. The full specification, the code changes
and the cost are in [`plan_floor_clip.md`](plan_floor_clip.md). Anything that changes after the runs
are submitted is a deviation, and will be recorded as one.

**The question.** `ekfac` and `tekfac` divide each coordinate of the step, in their eigenbasis, by
`s + λ`. Two other ways to protect that division were set aside by the feasibility study
`fr/etude_clipping_vs_damping.md`, because at the shipped `λ = 1e-3` both degenerate: the floor is
then identical to `s + λ`, and the clip clips 95 to 100 % of the coordinates. E7 to E14 found an
operating point, `λ` between 1e-11 and 1e-10, where `λ` sits inside the list of curvature values.
There, both can differ from E14's fix:

```
E14's fix (add):   u = M / (s + λ),                lr = cap · λ
family A (floor):  u = M / max(s, λ),              lr = cap · λ
family B (clip):   active: u = sign(M)·min(r/γ, 1), r = |M|/s;  inactive (|M| ≤ guard): M/max(γ·s, guard)
                   lr = the benchmark's own; γ always per layer:
   clip       γ reset at every step so a fraction q of the active coordinates is clipped
   clipema    the same γ, times μ̄/μ: μ = rms(M), μ̄ its bias-corrected running average (~1 000 steps)
   clipfixed  the median of the layer's γ over steps 1000-1999, frozen at step 2 000
```

Family B does not use `λ` at all. None of its three arms is sensitive to the scale error of the
stored curvature, which differs from layer to layer: `clip` and `clipema` absorb it at every step,
and `clipfixed` absorbs it once, per layer, when it calibrates. That is the property that could make one `q` work on every network, which E14
found no single `λ` does. The three differ in what they do with the momentum's size. `clip` throws
it away at every step. `clipema` keeps it relative to its recent history. `clipfixed` keeps it
entirely: its clip is conditional, and below the threshold the step is proportional to the
momentum.

**The arms.** `cnn_gn_cifar`, `vit_micro_cifar`, `cct_2_3x2_cifar`; `ekfac` and `tekfac`; seeds
0-4; batch 32, 15 epochs, the cosine schedule, the shipped estimator, `eig_before_rescale=True`:
E10/E13/E14's protocol, with one change. Every arm but the repro diagnostic runs with
`norm_exact_rescaling=True` (third amendment).

| arm | grid | runs per (network, mode, seed) |
|---|---|---|
| determinism check (seed 0) | the add cell at E14/E13/E10's best `λ`, run a second time in another process | 1 |
| repro diagnostic (seed 0) | the same cell without `norm_exact_rescaling`, compared with the stored cell | 1 |
| add (E14's fix) | 5 values of `λ` around each network's optimum | 5, rerun here |
| floor | the same 5 values | 5 |
| clip | `q` in {0.99, 0.95, 0.9, 0.7, 0.5, 0.3, 0.1} | 7 |
| clipema (main arm) | the same `q` | 7 |
| clipfixed | the same `q`, at the calibration step | 7 |
| clip-lr control | clipema at `q = 0.7`, at `lr/10`, `lr/3`, `3·lr` | 3 |

The `λ` grids: `cnn_gn_cifar` and `cct_2_3x2_cifar` {3e-10, 1e-10, 3e-11, 1e-11, 3e-12},
`vit_micro_cifar` {1e-9, 3e-10, 1e-10, 3e-11, 1e-11}. Every add cell is run here, so the add grid
is complete on all three networks.

**Fixed across the clip arms, so they are comparable with the add cells.** The parameters no hooked
module owns are frozen: in the add cells they move by `cap·λ ≈ 1e-10` times their momentum, i.e.
they are frozen in effect. On the two transformers the clip arms run with no weight decay: their
decay is decoupled, and in the add cells it vanishes with `lr = cap·λ` (this Part's rule 3). Both
choices are argued in `plan_floor_clip.md` §4.

**The decision rules, fixed now.**
0. **Nothing incomplete is read as a verdict.** Every file must be a production run: not a smoke;
   15 epochs, batch 32, full grids and split, calibration at 2000.
   - Every shard is merged, and every planned cell is present and not crashed.
   - Every cell but the repro diagnostic ran with `norm_exact_rescaling`, and every clipfixed cell
     froze its threshold in every layer.
   - At seed 0, the determinism check passed in every file, and the inertness check on
     `cnn_gn_cifar`.
   - All thirty files come from one commit of a clean tree.

   A value is usable only with finite final validation and test accuracy at all five seeds. A family
   whose grid or comparison has an unusable value is **incomplete**, never "equivalent".
1. **Select on validation, judge on test.** Within each family, the chosen `λ` or `q` is the one with
   the best five-seed mean of the final-epoch validation accuracy. The add baseline is selected from
   the same five values of `λ` as the floor. The comparison is made on test accuracy at the chosen
   values.
2. **Per (network, mode):** `Δ` = test accuracy of the family minus that of add, paired by seed,
   for each of floor, clip, clipema and clipfixed.
   **Win** if the mean of `Δ` exceeds 2 standard errors, **lose** if below −2 standard errors,
   **tie** otherwise. A zero standard error with a nonzero mean is flagged as degenerate, not voted.
3. **Per family.** **Better than the E14 fix** if it loses nowhere and wins for both modes on at
   least two of the three networks. **Worse** if it loses on at least two networks. **Equivalent** if
   it loses nowhere and is not better. **Mixed** otherwise.
4. **Transfer.** A value is transferable if it lies inside the plateau of all six (network, mode)
   pairs, with E13's plateau (every value whose five-seed mean test accuracy is within one standard
   error of the best). That standard error is the best value's own, as E14 read the rule; E13's own
   numbers were computed with the two-sample criterion `m_best − m_v ≤ √(SE_best² + SE_v²)`, which is
   reported alongside and not voted on. Applied to `q` for each clip, to `λ` for add, and for E17 to
   `λ` for the floor, over the same grids. The clip
   is worth adopting even at "equivalent" if some `q` transfers and no `λ` does, because it then
   removes a per-network search.
5. **`lr` control.** If a clip-lr cell beats the clipema cell at `q = 0.7` under rule 2, clipema's
   rule-3 verdict is reported as "limited by `lr`", not as a verdict on clipping. Three cells at 2 SE
   on five seeds give a false "limited" up to 16 % of the time per (network, mode).
6. **Inside family B**, with rule 2's criterion, each arm at its own chosen `q`. **clipema − clip**
   measures the per-step normalisation: a tie means it neither helps nor hurts; a clip win means a
   clip win against add is not a win for clipping as such. **clipfixed − clipema** measures a frozen
   threshold against one that follows the momentum.

**Checks on the pipeline, at seed 0.** They use the add cell at E14/E13/E10's best `λ`
(`cnn_gn_cifar`: `ekfac` 3e-11, `tekfac` 1e-10; `vit_micro_cifar`: 1e-10; `cct_2_3x2_cifar`: 3e-11).
Each compares test accuracy and loss, the validation curve, the distance travelled and the step
count.
- *Determinism.* That cell runs a second time, in another process of the same job. The two must be
  equal. If they are not, runs on this hardware are not reproducible under concurrency, and E16 is
  not read (rule 0).
- *Inertness*, on `cnn_gn_cifar` only. The same cell without `norm_exact_rescaling` must equal the
  add cell, since that network has no hooked normalisation layer. If it does not, the option
  changes something it should not (rule 0).
- *Repro, a diagnostic that does not vote.* The cell without the option is compared with the stored
  E14/E13/E10 cell of seed 0. It says whether this code on this hardware reproduces E14. A mismatch
  is reported, not fatal, because every comparison E16 makes is between cells of one job.

**Predictions, written now so they can fail.** The floor ties everywhere, within 0.5 points, at the
same `λ` as add or one grid step away. For the clip there are two competing readings, and nothing
measured so far favours either: it wins or transfers, because what `λ` really set was a step-size
cap; or it loses, because near `q = 1` it is sign descent and near `q = 0` it is the undamped step,
whose noise every curve below 1e-11 already shows. If Sophia's published range carries over, the
chosen `q` lies in 0.5 to 0.9. clipema and clip are close, within one seed floor: momentum at `β = 0.9` already
smooths over ~10 steps, and the gradient's size changes only moderately within 1 000 steps. clipfixed's
clipped fraction drifts after calibration, in a direction not predicted. The corrected statistic
does not move add's optimum: on every (network, mode), rule 1 selects E14/E13/E10's `λ` for add, or
a value one grid step away. It changes the step only on the normalisation layers, by at most 2.45×
in median at step 2 000, on 320 of 21 098 parameters on `vit_micro_cifar` and 1 280 of 283 723 on
`cct_2_3x2_cifar`.

**What this cannot separate, recorded before the result.** Every run is at batch 32, so the
batch-size question (open question 2 above) stays open for the floor and for add. Every clip verdict
is a verdict at `clip_guard = 1e-3`, on this hardware (TF32 convolutions on the H100 are noisier
than the guard on `cnn_gn_cifar`). The normalisation layers' rescaling is now exact (third
amendment), but their 2×2 input factor, from which their eigenbasis comes, is still built from the
raw channel mean (`CLAUDE.md` §4.6). The rescaling is optimal in that basis; the basis may not be
the best one (`plan_floor_clip.md` §7).

**Cost.** 34 runs per (network, seed, mode), 36 at seed 0. Projected from the add cells' measured
times on 1g slices and the clip's measured per-operation cost: about 65, 145 and 170 minutes of runs
per job on `cnn_gn_cifar`, `vit_micro_cifar` and `cct_2_3x2_cifar`. On one `h100_3g.40gb` slice
running three at a time, that is about 22, 48 and 57 minutes per job, if each run keeps its 1g
speed. That speed is not measured, so the limits are about twice that and one job is submitted
first. Thirty jobs, about 62 hours of run time summed over cells. Job script
`fisher_ref/slurm/e16_floor_clip.sh`, decisions `fisher_ref/experiments/e16_decisions.py`.

### E17 — pre-registered: ViT-S is the held-out network for E15's and E16's verdicts

**Amended on 21 September 2026, before any E17 run and before any E16 result existed.** Two changes,
and one fact about timing. The rest of this section is the text as first written.

1. **Candidate C does not qualify.** Its entry condition is "E15 rule 4 says tuning-free". Rule 4,
   as written, covers all six of E15's (network, mode) pairs, and there it fails. `kfac`'s plateaus
   are {1, 0.3} on `cnn_gn_cifar` and {3} on `vit_micro_cifar`. Neither contains the `τ = 0.1` that
   lies in all four `ekfac`/`tekfac` plateaus. C's tie-break clause mentions "the four pairs", which
   can be read as limiting rule 4 to `ekfac`/`tekfac`. That ambiguity was noticed only after E15's
   results had been read. So every reading chosen then was chosen after the fact, except the literal
   one. **The literal reading governs**, by the user's decision on 21 September 2026. S1-b at
   `τ = 0.1`, for `ekfac` and `tekfac`, may still run on ViT-S. It runs inside the stage-1 jobs, not
   before them, as an arm labelled **exploratory**. It is reported apart and casts no vote in E17. A
   good result can only motivate a new test, pre-registered on its own, on a fourth network.
2. **E16's candidates run with E16's final estimator.** E16's third amendment runs every cell with
   `norm_exact_rescaling=True`: the rescaling statistic uses the exact per-row gradient of a
   normalisation layer. E16 also re-runs its own add baseline under that option. "Configured exactly
   as its source experiment ran it" therefore includes `norm_exact_rescaling=True` for candidates A,
   B and D, in both stages, and D's `λ` comes from that re-run baseline. The candidates are what
   `fisher_ref/experiments/e16_decisions.py` prints from E16's thirty output files. The reference
   arm stays the optimizer as it ships: the bench's `λ = 3e-3`, with the shipped estimator.
3. **Timing, as it happened.** This section was first written while E15's `vit_micro_cifar` jobs
   were still running. Another session then read E15's results and wrote them up; it states that it
   did so after this section existed. The first commit containing this section came after that
   write-up, so the commit alone cannot prove the order. Apart from this amendment, the text below is
   unchanged since it was first written.

Written on 21 September 2026, **before any result of E15 or E16 was read**, and before any run
described here. What existed at that moment: E15's ten jobs had been submitted (21523753-62). The
five `cnn_gn_cifar` jobs had finished and their outputs were on the cluster. They were not opened
while this section was prepared. The five `vit_micro_cifar` jobs were still running. E16 had not been
submitted. Anything below that changes after this point is a deviation and will be recorded as one.

**The question.** E15 and E16 each pick a way to set the safety constant, or to replace it. They pick
it on the networks they tune on: `cnn_gn_cifar` and `vit_micro_cifar`, plus `cct_2_3x2_cifar` for
E16. Their transfer rules (E15 rule 4, E16 rule 4) ask whether one setting works on *all of those*
networks. They cannot ask whether it works on a network nobody tuned it on. E17 asks that. The network
is ViT-S: `vit_small_cifar` (2 693 578 parameters) first, and `vit_small_cifar100` second.

**Why ViT-S.** Three reasons.

1. It is where the shipped Fisher arms lose. In campaign 2 the best of them, `diag`, sits **4.10**
   points below AdamW in best validation accuracy on CIFAR-10 (67.68 % against 71.78 %) and **4.68**
   points below on CIFAR-100 (40.80 % against 45.48 %). In final test accuracy the gaps are 4.23
   (67.24 % against 71.47 %) and 4.80 (40.67 % against 45.47 %). One seed each.
2. It is a transformer 128 times larger than `vit_micro_cifar`, the only transformer the λ work has
   used, built from the same code with the same hyperparameters.
3. It is clean. No experiment in this document has run it at any `λ`, `τ` or `q` other than the
   shipped ones. The only trace of it in the E series is job 21190546: E0's curvature-free baseline,
   run on the shipped `diag`, cancelled at epoch 4, with no output saved. E1's seven networks do not
   include it.

**What gets transferred is decided by E15's and E16's own rules, not by a choice made later.** Each
candidate below is defined by the verdicts those two experiments' pre-registered rules produce.
Nothing is picked by looking at their curves. The modes are `ekfac` and `tekfac`, the two that E15
and E16 share. Each candidate is one arm per mode.

- **A. Each of E16's three clip arms** (`clip`, `clipema`, `clipfixed`), if E16 rule 3 rates it
  "better" or "equivalent" *and* E16 rule 4 finds at least one transferable `q` for it. The value
  sent to ViT-S is its transferable `q` with the highest validation accuracy, averaged over the six
  (network, mode) pairs of five-seed means. That is E16 rule 1's selection, applied across networks.
- **B. E16's floor**, under the same two conditions. E16 rule 4 is written for `q` and for add's `λ`.
  For this candidate its definition is applied to the floor's `λ`: a value inside the plateau of all
  six pairs. Only the four values common to every network's grid can qualify
  (3e-10, 1e-10, 3e-11, 1e-11).
- **C. E15's S1-b** (one `λ` per layer), if E15 rule 3 says "adopted" *and* E15 rule 4 says
  "tuning-free". The value sent is the `τ` that lies inside every plateau. If several do, it is the
  one with the highest validation accuracy averaged over the four pairs. **E15's network-adaptive
  arm** is a candidate under the same two conditions, with rules 2 to 4 applied to it in place of
  S1-b.
- **D. The added `λ`, always run.** If E16 rule 4 finds a transferable `λ` for add, that value is
  used. If not, the value is the geometric mean of the six `λ` values that E16 rule 1 selects for
  add, one per (network, mode). This arm is run whatever else qualifies. It measures what the best
  single network-independent `λ` gives. That is the baseline any other candidate has to beat to be
  worth its complexity.

The candidates and their values are written into this section as a dated addendum. The addendum
contains only the output of E15's and E16's decision rules (E16's are computed by
`fisher_ref/experiments/e16_decisions.py`), applied as above. It is written
before any E17 job is submitted.

**The one thing that would spoil the test.** A run on `vit_small_cifar` or `vit_small_cifar100` at
any `λ`, `τ`, `q` or damping form other than the shipped ones, before stage 1 is done. That is
forbidden until then. E18 does not break this rule: it runs only the shipped arms, plus a control
with no setting of its own (see E18).

**Stage 1: does the rule transfer?** The E-series protocol, so that the value means on ViT-S what it
meant where it was chosen. Batch 32, 15 epochs, the cosine schedule, the shipped estimator,
`eig_before_rescale=True`, seeds 0-4. Each candidate arm is configured exactly as its source
experiment ran it on `vit_micro_cifar`, the other transformer: the step-size cap, which parameters
are frozen, and how weight decay is applied. Each seed's job re-runs its own reference arm: the same
mode at the bench's own `λ = 3e-3`.

After the candidate arm, an **oracle sweep** is run on ViT-S. It covers the candidate's own family,
on its source experiment's grid: the seven values of `q`, or E15's eight values of `τ`, or five values
of `λ` at half-decade spacing centred on the transferred one. The oracle exists only to measure how
far the transferred value sits from ViT-S's own best. It cannot change the transferred value, which is
fixed before any ViT-S run.

**Stage 1 rules, per candidate and per mode.**

1. **Does it help?** `Δ` = test accuracy of the candidate minus that of the reference arm, paired by
   seed. It **helps** if the mean of `Δ` exceeds 2 standard errors. It **hurts** if the mean is below
   −2 standard errors. Otherwise it **ties**. This is E15 rule 2's criterion.
2. **Does it transfer?** The plateau is E13's: every value on the oracle grid whose five-seed mean
   test accuracy lies within one standard error of the best. The candidate **transfers** if its value
   lies inside ViT-S's plateau. This extends E16 rule 4 to a seventh (network, mode) pair. If the
   oracle's best value sits at an edge of its grid, the grid is extended by two values on that side,
   once. If the best is still at the edge, the verdict is **unresolved**.
3. **Transfer loss, reported and not voted on.** The oracle's value is chosen on validation, as in
   E16 rule 1. The loss is its test accuracy minus the candidate's, paired by seed.
4. **Per candidate.** It **transfers to ViT-S** if rule 2 says "transfers" for both modes. It
   **fails to transfer** if it lies outside the plateau for both. Anything else is **mixed**.

**Stage 2: does it close the campaign gap?** Stage 2 is run for every candidate that is not "fails
to transfer" in stage 1. It uses campaign 2's protocol exactly, the flags of
`benchmarks/slurm/cifar10/train_vit_small_cifar_all.sh`: batch 128, 50 nominal epochs, the
wall-clock-time protocol with `diag` as the reference arm, the budget cosine, `--max-epoch-factor 3`.
It runs on both datasets with seeds 0, 1 and 2. Each (dataset, seed) is one job that runs its own
`diag` reference, its own `adamw`, its own shipped `ekfac` and `tekfac`, and the candidates. Every
comparison is therefore made inside one job. Under this protocol only the reference arm is
reproducible across jobs; this repository measured that on the seed campaign, 158 of 160 checkpoints.

**Stage 2 conversions, fixed now.**

- **`λ`-based candidates (B and D)** are in stored units at batch 32. Stage 2 runs at batch 128, so
  it uses `λ × (32/128)² = λ/16`. The factor is E3's measured 1/batch² scaling of the stored
  curvature. That this carries the *optimum* across batch sizes is untested: it is Part 5's item 9.
  So a stage-2 failure of a `λ`-based candidate reads "fails at batch 128 under the 1/batch²
  conversion", and not "fails".
- **`q` and `τ` (A and C)** are free of scale and are used unchanged.
- **Every other setting keeps its value**, including the settings counted in steps
  (`clip_ema_horizon`, `clip_calibrate_at`, `T_eig`, `TCov`).
- **Decay:** decoupled, at the bench's own rate for the Fisher arms, `1e-3 × 1e-2 = 1e-5` per step at
  the top of the cosine, whatever `λ` is (Part 5, rule 3). That applies to the clip arms too, unlike
  in E16, because the partner here is AdamW, which decays.
- **Step-size cap:** as in stage 1. `lr = cap × λ` with `cap = 1/3` for the `λ`-based candidates, and
  the bench's `lr = 1e-3` for the clip.
- **Parameters no hooked module owns:** as in stage 1. On ViT-S these are `cls_token` and
  `pos_embed`, **0.47 %** of the parameters (12 672 of 2 693 578). E6 measured freezing them inert on
  `vit_micro_cifar`, where they are 9.7 %.

**Stage 2 rules, per dataset, per candidate and per mode.** The primary number is final test
accuracy, measured once after the last epoch, with nothing selected. Best validation accuracy is
reported next to it, for continuity with campaign 2's 4.10 and 4.68.

1. `gap` = test accuracy of the candidate minus that of `adamw` in the same job. The candidate
   **closes the gap** if `gap ≥ 0` on all three seeds.
2. It **narrows the gap** if its `gap` is larger than the shipped arm's (same mode, same job) on all
   three seeds.
3. Otherwise it **does not close the gap**.

Three seeds give weak evidence. Under a symmetric null, three results of the same sign occur with
probability 1/4. A pass says "worth five seeds", not "established".

**Not submitted until four things exist:** the addendum naming the candidates; `vit_small_cifar`
entries in the E15/E16 drivers; the harness fields that carry the transferred settings into
`HParams` for stage 2, inert and bit-identical when unset; and a one-run calibration of ViT-S at
batch 32. That last one is also where the cost comes from. It is not measured. The estimate from the
batch-128 time (997 s for 50 epochs, campaign 2) is 6 to 10 minutes per 15-epoch run.

**What E17 cannot settle.** One architecture, trained on two datasets. "Transfers to ViT-S" means
"transfers to one more network", not "transfers". And stage 2 for the `λ`-based candidates rests on
the 1/batch² conversion.

### E18 — pre-registered: is ViT-S's deficit against AdamW the deficit of momentum SGD?

Written on 21 September 2026, before any E18 job was submitted. Same state of E15 and E16 as at E17.

**The question.** Campaign 2's sixth finding (`CLAUDE.md`, and
`docs/reports/campaign2_cifar100_imagenet.md`) offers one reading, and says it is untested. At the
shipped `λ`, the operator each Fisher arm divides by is almost `λ I`. Each arm is then momentum SGD at
a learning rate of `lr (1 − β) / λ`. On the ViT benches that is `1e-3 × 0.1 / 3e-3 = 0.033`. If this
holds on ViT-S, its 4.10- and 4.68-point deficit against AdamW is what momentum SGD at that rate does
on this ViT. It would not then be evidence about curvature. E18 tests it with an arm that *is* that
limit.

**The arm, `sgdm`.** `benchmarks/common/optimizers.py::LambdaLimitSGD`. It is `AdaFisherMulti`'s own
update with every Fisher estimate replaced by `λ I`:

```
m_t   = β m_{t−1} + (1 − β) g_t
θ    ← θ (1 − lr × wd)                          decoupled decay, as AdaFisherW
θ    ← θ − lr / (1 − β^t) × m_t / λ             a parameter of a hooked module
θ    ← θ − lr / (1 − β^t) × m_t                 cls_token, pos_embed: AdaFisher's own fallback
```

It takes the Fisher arms' `lr`, `λ`, `β` and decay convention, not AdamW's `baseline_lr`. It has no
setting of its own. The fallback line is kept on purpose. Dividing `cls_token` and `pos_embed` by `λ`
too would make this arm differ from the Fisher arms in a second way. It computes no curvature, so a
step costs what a momentum-SGD step costs.

Two tests pin it (`tests/test_sgdm_arm.py`, 19 tests). **First, it is AdaFisher minus the
curvature.** `AdaFisherMulti` with `gammas = (1.0, 0.0)` multiplies every running average by 0, so
the operator it divides by is exactly `λ`. The arm reproduces that optimizer **bit for bit**
(`torch.equal` on every parameter). This holds on 12 configurations: all four hooked layer types, and
a ViT with `cls_token` and `pos_embed`, each under three decay settings and two `TCov`. **Second, it
is torch's momentum SGD.** Torch's buffer is `b_t = β b_{t−1} + g_t`, and `m_t = (1 − β) b_t` exactly.
So the arm equals `torch.optim.SGD(momentum=β)` at learning rate `lr (1 − β) / (λ (1 − β^t))`. The
test checks this in float64, to 1e-10.

**What is already known, and what is not.**

- **For `diag`, the reading is almost arithmetic.** Each of its two factors is min-max normalised to
  [0, 1] and then averaged with the coefficients (0.08, 0.008), starting from 1. So after `k` factor
  updates each is at most `X_k = 0.08^k + (0.008/0.92)(1 − 0.08^k)`. The amount `diag` adds above `λ`
  is then at most `X_k²`: 7.7e-3, then 2.3e-4, then 8.5e-5, and 7.56e-5 in the limit.
  `tests/test_sgdm_arm.py` checks that bound. At `λ = 3e-3`, `diag`'s step is at least these fractions
  of `sgdm`'s, coordinate by coordinate: **0.279** during steps 0-99, **0.930** during steps 100-199,
  and **0.973** from step 200 on. That holds on any network. So at an equal number of steps, the only
  open question for `diag` is whether a 2.5 % smaller step and the first 200 steps of 17 550 change
  the outcome.
- **For the four Kronecker modes nothing bounds it.** Their operator is built from stored factors
  whose size depends on the network. Lot 5 measured it at `cond ≤ 1.1035` on four regime-A models.
  E1 measured `λ` above every curvature direction on seven networks. **ViT-S is in neither
  measurement.**

**The runs.** Datasets `vit_small_cifar` and `vit_small_cifar100`, seeds 0, 1 and 2. One job per
(dataset, seed), with two invocations of the bench (`fisher_ref/slurm/e18_sgdm_control.sh`).

1. **Campaign 2's grouped job, verbatim, with `sgdm` added last to `--arms`.** The flags are those of
   `train_vit_small_cifar_all.sh`: `--epochs 50 --budget-mode wct --reference-arm diag
   --max-epoch-factor 3 --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1`. `diag` runs its 17 550
   steps unbudgeted. Its time becomes the budget of the other seven arms, `sgdm` included.
2. **`sgdm` alone at matched steps:** `--budget-mode epochs --epochs 50 --lr-schedule nominal`. That
   is exactly `diag`'s 17 550 steps under `diag`'s own schedule. Same initialisation, same batch
   order.

All seven shipped arms are re-run, not reused from campaign 2. Under the wall-clock-time protocol only
the reference arm reproduces across jobs, so every comparison is made inside one job. Seeds 1 and 2
exist for no ViT-S arm at all. Three seeds give a first measure of its seed-to-seed spread.

Results go to `benchmarks/outputs/controls/e18_sgdm/<dataset>/<model>/seed<n>/{wct,matched_steps}/`.
That tree is outside every directory `fisher_ref/checkpoints.py` reads.

**Reproduction control, a gate.** At seed 0, `diag`'s per-step training loss must be **bit-identical**
to campaign 2's over all 17 550 steps. Its best validation accuracy must be 67.68 % and 40.80 %. If
not, nothing else is read until the difference is explained. The six budgeted shipped arms at seed 0
are compared with campaign 2 as well. That comparison is reported as the cross-job noise of a
budgeted arm on this model. It is not a gate.

**Endpoint, and one margin fixed now.** The primary number is final test accuracy: measured once,
after the last epoch, with nothing selected. Best validation accuracy is reported next to it.
Comparisons are paired by seed, within one dataset. The margin is `m = 1.0` point, a quarter of the
smaller gap the reading has to explain.

**The decision rules, per dataset.**

1. **Does `sgdm` reproduce `diag` at matched steps?** `Δ₁` = test accuracy of `sgdm` at matched
   steps minus that of `diag`. They are **equivalent** if the mean of `Δ₁` ± 2 standard errors lies
   inside ±`m`. They are **different** if the mean of `Δ₁` is more than 2 standard errors from zero.
   Otherwise the result is **unresolved**.
2. **The reading.** Write `D_sgdm` = test accuracy of `sgdm` minus that of `adamw`, both in the
   wall-clock job. Write `D_diag` = test accuracy of `diag` minus that of `adamw`.
   - **Confirmed** if three things hold: rule 1 says "equivalent"; the mean of `D_sgdm` is below −2
     standard errors, so momentum SGD loses to AdamW too; and the mean of `D_sgdm − D_diag` is within
     ±`m`.
   - **Refuted** if rule 1 says "different", or if the mean of `D_diag` is below −2 standard errors
     while that of `D_sgdm` is not. Either way the Fisher arm then loses something momentum SGD does
     not.
   - **Unresolved** otherwise.
3. **Does curvature buy anything at equal wall-clock time?** For each Kronecker mode, `Δ_f` = its test
   accuracy minus that of `sgdm`, in the wall-clock job, with rule 1's 2-standard-error criterion.
   Eight comparisons (four modes, two datasets) make one 2-standard-error "win" by chance plausible.
   So a mode counts as "curvature helps here" only if it wins on both datasets.

**Predictions, written now so they can fail.**

- Rule 1: equivalent on both datasets.
- Rule 2: confirmed on both datasets.
- Rule 3: every Kronecker mode ties or loses against `sgdm`. Campaign 2 gave them 1.5 to 9.4 % fewer
  steps than `diag` in the same budget (seed 0: `kfac` 17 121 and 17 055, `tkfac` 16 975 and 17 287,
  `ekfac` 15 947 and 16 165, `tekfac` 15 901 and 16 128, against 17 550). `sgdm`'s step costs about
  what AdamW's does, so it should complete about as many steps as AdamW (18 526 and 18 954).

**What each outcome would mean.**

- **Confirmed:** at the shipped `λ`, the ViT-S results measure momentum SGD at 0.033, not curvature,
  so the deficit is not evidence against curvature. It would not show that curvature helps ViT-S at a
  lower `λ`. That is E17's question.
- **Refuted:** the Fisher arms lose something momentum SGD at the same rate does not. That has to be
  explained before any claim about ViT-S is made.

**Relation to E17.** `sgdm` uses the shipped `lr`, `λ` and `β` and has no setting of its own, so E18
does not spoil E17's held-out status. E18's runs are not E17's comparison partners, because E17's
stage 2 re-runs its own inside one job. They serve E17 as a cross-job check.

**Cost, from measured times.** In campaign 2 one ViT-S arm took 997 s of training on CIFAR-10 and
990 s on CIFAR-100, plus about 6 % for validation. One job runs 8 arms at wall-clock budget plus 1 at
matched steps: 9 × ~1 060 s, about 2.65 hours, plus about 40 s of setup. The time limit is 3:45. Six
jobs, about 16 GPU-hours on `h100_1g.10gb` slices.

**Submitted** on 21 September 2026, after the rules above were written. Jobs 21532555, 21532557 and
21532559 run `vit_small_cifar`, seeds 0, 1 and 2. Jobs 21532556, 21532558 and 21532560 run
`vit_small_cifar100`, seeds 0, 1 and 2. Each has a 3:45 limit. The code reached the cluster by copying
three files, not by a commit: `benchmarks/common/optimizers.py`, which the cluster held at `51bbe0f` and
which now differs from it only by the `sgdm` arm; `tests/test_sgdm_arm.py`; and this job script.
SHA-256 of the two files the jobs run, identical on both sides: `optimizers.py` `42eb27c8…403c9`, job
script `d010a91b…f358`. A local smoke run of both invocations on 512 images went through every arm,
the report, the plots and the checkpoints, and the matched-steps `sgdm` took exactly `diag`'s step
count.

**Reproduction gate: passed**, checked while the jobs ran by comparing only the `diag` rows. At
seed 0, `diag`'s per-step training loss is identical to campaign 2's at all 17 550 of 17 550 steps,
on both datasets. Its best validation accuracy is 67.68 % and 40.80 %, and its test accuracy 67.24 %
and 40.67 %, as in campaign 2.

**Code provenance, recorded when it changed.** Each job's first invocation started on `51bbe0f` plus
the three copied files. At about 17:22 EDT on 21 September, while those first invocations were
running, the cluster checkout was fast-forwarded to `178a758`. That commit contains the same three
files, byte for byte, and adds E16's code in `src/adafisher_modes/`. Each job's second invocation,
`sgdm` at matched steps, therefore starts on `178a758`. It runs nothing that changed. Between the
two commits, `benchmarks/common/`, `benchmarks/models/` and `requirements-cluster.txt` differ only by
`optimizers.py`, which the jobs already had in its committed form. The one thing the arm reads from
the package, `factors.SUPPORTED_MODULES`, is unchanged; `factors.py` only gained new functions.

### E15 — done. One safety constant per layer wins everywhere, and the win comes from the layers.

Ten jobs (21523753-62), all COMPLETED, no crashed run: `cnn_gn_cifar` in 1 h 18 to 1 h 31 per
seed, `vit_micro_cifar` in 2 h 56 to 3 h 04. Outputs:
`fisher_ref/outputs/e15_layer_damping_{cnn_gn_cifar,vit_micro_cifar}_s{0..4}.json`.

**The protocol held.** All six repro cells reproduce E13's and E14's seed-0 test accuracy to
**0.00 points** (65.18, 67.41, 68.55 on `cnn_gn_cifar`; 54.23, 54.36, 54.92 on `vit_micro_cifar`).
So the new code path changes nothing except what it was built to change.

**Result 1 — the pre-registered verdicts.** Each arm family's value is chosen on the five-seed mean
of the final validation accuracy (rule 1). Test accuracy in %, five-seed mean ± standard error. `Δ`
is paired by seed.

| network | mode | reference | best single `λ` | S1-b | network-adaptive | rule 2: S1-b − single | rule 5: S1-b − network-adaptive |
|---|---|---|---|---|---|---|---|
| `cnn_gn_cifar` | `kfac` | 61.38 | 65.27 (1e-10) | **69.58** (`τ`=1) | 65.52 (`τ`=1) | **+4.32 ± 0.30, win** | **+4.07 ± 0.34, win** |
| | `ekfac` | 60.75 | 68.24 (1e-10) | **70.20** (0.1) | 68.44 (3e-3) | **+1.96 ± 0.46, win** | **+1.76 ± 0.29, win** |
| | `tekfac` | 60.81 | 68.14 (3e-11) | **70.06** (0.1) | 68.38 (1e-2) | **+1.92 ± 0.28, win** | **+1.67 ± 0.33, win** |
| `vit_micro_cifar` | `kfac` | 45.81 | 52.76 (1e-11) | **57.33** (3) | 54.37 (1e-2, edge) | **+4.56 ± 0.41, win** | **+2.96 ± 0.21, win** |
| | `ekfac` | 46.14 | 54.90 (1e-10) | **60.95** (0.1) | 56.63 (3e-3) | **+6.05 ± 0.26, win** | **+4.32 ± 0.22, win** |
| | `tekfac` | 46.24 | 54.78 (1e-10) | **60.76** (0.1) | 56.67 (3e-3) | **+5.98 ± 0.27, win** | **+4.08 ± 0.32, win** |

- **Rule 2: S1-b wins in all six pairs**, by +1.9 to +6.1 points. The smallest margin is 4.3 paired
  standard errors, and S1-b is ahead on every one of the 30 (pair, seed) cells.
- **Rule 5: S1-b also beats the network-adaptive arm in all six pairs**, by +1.7 to +4.3 points. By
  the note added before submission, the win is therefore read as **per-layer**. It does not come
  from being relative, and it does not come from escaping the identity's start-up leftover. The
  network-adaptive arm itself only ties the single `λ` on `cnn_gn_cifar` (+0.21 to +0.25, none
  above rule 2's threshold of 2 standard errors). It beats it by +1.6 to +1.9 on `vit_micro_cifar`. Being relative helps a
  little on one network. Being per-layer helps a lot on both.
- **Rule 3: S1 per layer is adopted as the next fix.** It wins for `ekfac` and `tekfac` on both
  networks, and for `kfac` too.
- **Rule 4, as written: not tuning-free.** No single `τ` lies in the plateau of all six pairs.
  `ekfac`/`tekfac` have `τ = 0.1` in their plateau on both networks: {0.1}, {0.3, 0.1}, {0.1},
  {0.1}. `kfac` wants {1, 0.3} on `cnn_gn_cifar` and {3} on `vit_micro_cifar`. That is adjacent but
  disjoint, and ten to thirty times higher than the other two, as the pre-registration expected from
  its factored damping. So the reading that holds is "one `τ` per damping rule", and it holds for
  `ekfac`/`tekfac` on two networks. That is a finding to test on a third network, not a verdict,
  since rule 4 was written for all six pairs.

Against the default `λ`, S1-b gains **+8.2 to +14.8 points**: 70.2% against 60.8% on
`cnn_gn_cifar`/`ekfac`, and 61.0% against 46.1% on `vit_micro_cifar`/`ekfac`. The best single `λ`
gave +3.9 to +8.8.

One asterisk. The network-adaptive arm's best value on `vit_micro_cifar`/`kfac` is at the bottom of
its grid (`τ` = 0.01), so it is a bound. It rises slowly there, by about 0.3 points per half-decade
(53.75, 54.06, 54.37). That is far too slowly to close the 2.96 points rule 5 measured.

**Result 2 — the paper's own rule is not the best one, but it beats the best single `λ`.**
`τ = 1` is TEKFAC's eq. (3.5) without its floor. It reaches 68.74 / 68.77 on `cnn_gn_cifar` and
57.40 / 57.49 on `vit_micro_cifar` (`ekfac` / `tekfac`). That is 1.3 to 3.6 points under the best
`τ` (0.1), and still 0.5 to 2.7 points above the best single `λ`.

**Result 3 — what S1-b actually does to each layer.** From the per-layer logs: the time-average
from step 4 000 onwards, geometric mean over five seeds, at each family's chosen value. The first
number is `λ_l` divided by the best single `λ`. The two percentages are the fraction of that layer's
directions whose curvature exceeds its `λ`, under S1-b and under the best single `λ`.

| network, mode | head | the other layers | last block and final norm (`vit_micro_cifar` only) |
|---|---|---|---|
| `cnn_gn_cifar`, `ekfac` | **303×**; 15% / 57% | 0.36-0.55×; 36-58% / 15-22% | — |
| `cnn_gn_cifar`, `tekfac` | **986×**; 13% / 66% | 1.2-1.7×; 35-55% / 39-62% | — |
| `cnn_gn_cifar`, `kfac` | **26×**; 3% / 27% | 0.017-0.032×; 10-15% / 0-0.1% | — |
| `vit_micro_cifar`, `ekfac` | **370×**; 38% / 86% | 0.24-2.7×, patch embedding 11× | 0.05-0.5×; 17-56% / 0-4% |
| `vit_micro_cifar`, `tekfac` | **342×**; 36% / 86% | 0.24-2.6×, patch embedding 10× | 0.04-0.5×; 17-54% / 0-4% |
| `vit_micro_cifar`, `kfac` | **593×**; 8% / 73% | 0.2-2.7×, patch embedding 15× | 0.16-0.8×; 5-10% / 0% |

The pre-registered expectation holds **in part**. The head does get a constant hundreds of times
above the single optimum in five pairs of six (26 times with `kfac` on `cnn_gn_cifar`). Under a
single `λ`, 27% to 86% of the head's directions were being shrunk; under S1-b, 3% to 38% are. But
the other layers do not all stay near the single optimum. On `vit_micro_cifar` the last block and
the final norm get a constant **2 to 20 times lower**. There, a single `λ` shrank almost nothing (0% to
4% of directions above it) and S1-b lets the curvature act on 17% to 56% of them. With `kfac` on
`cnn_gn_cifar`, the whole body gets a constant 30 to 60 times lower.

So S1-b does two things at once. It **stops shrinking the head's steps**, and it **lets the
curvature act in the flattest layers**, which a single `λ` leaves as plain momentum. Put simply, it
evens out across layers the share of directions that the curvature actually reaches. A "relative
`λ` for the head only" arm, the follow-up the pre-registration named, would test the first effect
alone. The logs predict it would not recover all of the gain.

**Result 4 — the weight-decay confound of E10/E13 is negligible.** Same cell as E13, same `λ`,
paired by seed, with the decay at its fixed rate instead of vanishing: `kfac` −0.23 ± 0.25,
`ekfac` −0.08 ± 0.21, `tekfac` −0.16 ± 0.10. None is more than 1.6 standard errors from zero. E13's
gains stand. Part 5's rule 3 stays, as protocol hygiene.

**Consequence for E17, stated and not resolved here.** E17, pre-registered in this document before
E15's results were read, sends S1-b to ViT-S only if E15 rule 3 says "adopted" *and* E15 rule 4 says
"tuning-free". Rule 3 says adopted. Rule 4 **as written** covers every (network, mode) pair E15 ran,
which is six pairs including `kfac`, and it fails there. E17's own text, however, averages "over the
four pairs", which reads as `ekfac`/`tekfac` only, the two modes E17 transfers. Over those four pairs
rule 4 holds, with `τ = 0.1` the only value inside every plateau. Which reading governs is E17's
decision to make, recorded before any E17 job is submitted, and not something to settle by looking
at these numbers.

**What E15 does not establish.**
- Two networks, both small (24 k and 21 k parameters), at batch 32 for 15 epochs, with the shipped
  estimator. This is the E-protocol, not the benchmark's wall-clock protocol.
- Whether `τ = 0.1` transfers to a third network, for `ekfac`/`tekfac`.
- Whether S1-b survives a change of batch size. It should, since a relative `λ` absorbs the 1/batch²
  factor by construction, but that has not been run.
- Whether it helps on `resnet20_cifar`, where a single `λ` bought nothing.
