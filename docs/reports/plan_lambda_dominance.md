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

**Where that leaves it.** The curvature genuinely does not pay for itself on these networks at the
default operating point, and no amount of fixing the estimator changes that. But there *is* an
operating point where it pays, five orders of magnitude below the default, and reaching it by hand
per network is not a method. Making the safety constant relative to each layer (**S1**) is what
would reach it without hand-tuning, and it is what two of the five source papers already ask for.

The other fixes: make the step size stop depending on `λ` (**S2**), remove the start-up
transient (**S3**), and make the safety constant relative to each layer instead of one shared
number (**S1**). Two of the five source papers already ask for S1 and this port does not do it.

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

Lowering `λ` with the step held still is mathematically the same as putting the curvature back at its
true size — but it does not fix the *average*, which is still 92% a single batch of examples. So "the
curvature does not help" and "this particular estimate is too noisy to help" have not been told apart
yet. The collapse at 10⁻¹² looks a lot like an estimation-noise failure. The experiment that settles
it: fix the average properly, and rerun at batch 32.

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

## Part 5 — Order, and two rules for the protocol

| # | Action | Status | Why here |
|---|---|---|---|
| 1 | **E0** | **done** | closed the original question: the noise is not the curvature |
| 2 | **E1** | **done** (jobs 21283680, 21285097) | gave the target: `λ` above everything, still above 93-100% of directions even corrected |
| 3 | **E3** | **done** (same jobs) | confirmed the arithmetic, and says to run E4 at batch 32 |
| 4 | **E2** | **done** (job 21283681) | ran before the fixes because moving `λ` and the learning rate together does what S2 was for, with no code change. Answered both halves: the barrier is the step size, and lifting it buys nothing |
| 5 | **E4 with S3**, at batch 32 | **done** (jobs 21295803-5) | answered it: **not** the averaging. And found the one real gain in the study, +4.32 points on `cnn_gn_cifar`/`kfac` at 10⁻⁸ |
| 6 | **S1** as a full arm | **next** | a safety constant relative to each layer would reach E4's 10⁻⁸ operating point without hand-tuning it per network |
| 7 | **E5, E6** | open | the thousand-to-one split inside each network, and step 16's causal test |

**On the reordering.** S2 was planned as a prerequisite for E2. It turned out not to be needed:
moving `λ` and the learning rate together holds the cap exactly still, which is what S2 was for, and
does it without changing any optimizer code. S2 is still the right thing to ship, because it makes
the cap something you set rather than something that falls out of two constants. But it is no longer
blocking anything.

**Two rules, each one the lesson of a mistake already made.**

1. **Any sweep of `λ` must hold the step-size cap still.** Otherwise it moves two things at once and
   the result cannot be attributed to either (Part 2.1, and E2's result).
2. **Draw no conclusion from `mnist_autoencoder`.** Every arm of every configuration lands on the
   same number there. It cannot tell anything apart (audit_step, section 4.8).

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
