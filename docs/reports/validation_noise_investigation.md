# Why the Fisher modes' validation curves are noisy

*A record of one investigation, written to be re-read months later. Plain language throughout:
every term is explained where it first appears. All numbers here were measured, none estimated.*

---

## The question we started from

Across the benchmark runs, the five Fisher-based optimizer modes (`diag`, `kfac`, `ekfac`,
`tkfac`, `tekfac`) produce validation curves that wobble much more from one epoch to the next
than plain `Adam` does — but only on some models. On ResNet-20 the wobble is obvious; on the
small Vision Transformer it is not, and the Fisher curves are actually steadier than Adam's.

The first guess was that **BatchNorm** was responsible. ResNet-20 uses it, the Transformer does
not. BatchNorm is the one component that keeps a memory of past batches and feeds it back into
the network, so it was the natural suspect.

The repository already contains the perfect test for this: one small CNN (`cnn_gn_cifar`) that
can be built with either GroupNorm or BatchNorm, changing nothing else. Both variants had already
been run.

---

## Summary of what we found

1. **The first comparison was not valid.** The two runs differed in three ways at once, not one.
2. **A first attempt to fix it produced garbage**, because the machine we ran on was being shared
   with someone else's work. This was detected, diagnosed and thrown away.
3. **The clean rerun gave a weak, mixed answer**: BatchNorm makes things somewhat worse, but not
   uniformly, and not by much.
4. **The decisive finding came from a different question.** The extra noise is almost entirely on
   the *evaluation* side, not the *training* side. The optimizer's own trajectory is as smooth as
   Adam's; it is the performance on unseen data that jumps around.
5. Two follow-up checks eliminated two more suspects: the 100-step refresh cycle, and the share
   of normalization layers in the network.
6. What seemed to remain standing was a property of curvature-based optimization itself, and a
   single constant (`λ`) that controls how far it can go.
7. **Measuring that constant overturned point 6.** In every model and every mode, `λ` is far
   larger than the curvature the optimizer has estimated, in *every* direction. The division by
   curvature, which is the whole idea of these methods, has almost no effect: all five modes
   behave like plain momentum with a fixed step multiplier. The noise therefore cannot come from
   "large steps in flat directions", and its real cause is open again (Step 7).
8. **One loose end from Step 7 turned out to have a clean cause.** A 10 000-times sweep of `λ`
   had barely moved the final result, which looked contradictory if the step size really is
   `learning rate ÷ λ`. It is not a contradiction: for roughly the first 300 steps, `λ` is not the
   number that sets the step size at all — a leftover of how the optimizer starts up is, and that
   leftover does not depend on `λ` (Step 8).
9. **Testing whether that picture ("these methods are secretly all plain momentum") actually
   predicts real training gave a mixed answer.** It does, almost exactly, on a network with no
   convolutions. On two convolutional networks it clearly does not: swapping the real curvature
   estimate for nothing, everything else identical, moves the final result 8 to 33 times more than
   two ordinary random-seed runs of the same method differ from each other (Step 9).
10. **The obvious explanation for point 9 — a few outlier directions hiding in the tail — does not
    hold up.** Checked on every layer of all seven networks, not just the two spot-checked before:
    the single most-curved direction found anywhere, in any layer, in any mode, on any network,
    tops out at 7% of `λ` (`diag` only; under 0.2% for every other mode). There is no hidden
    high-curvature tail for the two convolutional networks to be using (Step 10).
11. **Point 9's own stand-in turned out to have a bug of its own — fixing it changed the answer on
    two of the three networks.** The stand-in's step-size formula was a plausible guess, not
    derived from `kfac`'s actual two-factor math; correcting it (and a second, separate fix for a
    layer type `kfac` does not handle at all) makes `mlp_ln_mnist` match exactly and
    `resnet20_cifar` a close call instead of a clear failure. `cnn_gn_cifar` gets *worse*, and
    which of the two fixes is responsible has not been separated out (Step 11).
12. **Separating those two fixes on `cnn_gn_cifar` shows they disagree.** The layer-type-boundary
    fix helps a lot (its best-measured configuration on this network, 7× the noise floor on
    accuracy); the step-size-formula fix — the same one that helped the other two networks —
    consistently hurts on this one, in every configuration tested. A likely reason, not yet
    checked: every "curvature is negligible" measurement in this document was taken mid-training,
    never in the first 200 steps of a fresh network, which is exactly where these stand-ins are
    actually compared against real `kfac` (Step 12).
13. **Checked directly — that lead does not hold up.** Curvature in the first 0-300 steps of a
    fresh network is real and genuinely uneven across directions (confirmed on both
    `cnn_gn_cifar` and `resnet20_cifar`, not assumed), but the two networks show essentially the
    *same* early pattern — a non-negligible, anisotropic classification head fading over ~300
    steps — even though one network's schedule fix helped and the other's hurt. Early curvature
    is not what makes `cnn_gn_cifar` the exception (Step 13).
14. **Widening the check to all five modes on six networks changes the shape of the answer.**
    "Behaves as plain momentum" passes cleanly on 11 of 30 (network, mode) pairs and is a few
    percent off on another 12; only 2 are clear failures, and `cnn_gn_cifar`/`kfac` is joined by
    a new one, `cct_2_3x2_cifar`/`tkfac`. `resnet20_cifar` turns out to be a different kind of
    exception again: every one of its five modes is a small, consistent, several-percent gap —
    never a clean pass, never a clear failure (Step 14).
15. **Pushing `λ` down confirms the theory in a narrow window, then breaks the optimizer instead
    of teaching anything new.** One order of magnitude below default, most networks still train
    and several modes show a real, moderate divergence from plain momentum — the predicted
    effect, seen directly. Two more orders down, nothing new is revealed: `kfac`/`tkfac` crash
    outright on a singular matrix, and `diag`/`ekfac`/`tekfac` silently collapse to chance-level
    performance instead — both a dead end, not a deeper signal (Step 15).
16. **The two Step 14 failures turn out to have different causes, not one shared mechanism.**
    `kfac`'s failure on `cnn_gn_cifar` has a strong correlational candidate, not yet shown to be
    causal: it is the only mode whose *applied* step is genuinely anisotropic in the first ~200
    steps of training, on either network tested — precisely where it fails. `tkfac`'s failure on
    `cct_2_3x2_cifar` shows the opposite pattern — flat, isotropic
    early conditioning where it fails, real anisotropy on the network where it passes — so it
    needs its own explanation, not yet found. `resnet20_cifar`'s uniform few-percent gap on every
    mode is only half-explained: `kfac`'s own mild anisotropy there fits a dilution story (its one
    anisotropic layer is a much smaller share of a 39-layer network than of `cnn_gn_cifar`'s
    4-layer one), but the other four modes show no early anisotropy at all and still land in the
    same band (Step 16).

---

## Step 1 — the first comparison was not a comparison

The two existing runs (GroupNorm and BatchNorm) differed in **three** ways simultaneously:

| | GroupNorm run | BatchNorm run |
|---|---|---|
| learning-rate schedule | old, buggy version | fixed version |
| hardware | cluster GPU | laptop processor |
| normalization type | GroupNorm | BatchNorm |

The learning-rate schedule difference is the serious one. The schedule is supposed to lower the
learning rate smoothly to zero over the course of training. In the old version, a fast optimizer
that finished its allotted time early would run *past* the end of the schedule — and because the
schedule is periodic, its learning rate would start **climbing back up**. We confirmed this
directly in the data: in the GroupNorm run, `adam` and `adamw` each show two epochs where the
learning rate increases instead of decreasing. In the BatchNorm run, every arm decreases
monotonically.

So any difference in wobble between these two runs could equally well have come from the
schedule bug or the hardware, not from the normalization layer. The comparison had to be redone.

---

## Step 2 — the rerun that had to be discarded

We resubmitted the GroupNorm variant on the cluster, matched to the BatchNorm run's settings
(same schedule, same processor-only computation). **Job 21122754. It completed without error and
the results were nonsense.**

How the comparison works: every optimizer gets the same amount of *real time* to train. One
reference optimizer runs a full training first, and however long it took becomes the time budget
for all the others. Each one then completes as many passes over the data as its speed allows.

The results made no sense: `ekfac` completed only 8 passes over the data and `tekfac` only 11,
where the others did 21 to 35 — far outside what their known computational cost predicts. And
`adam` (35 passes) hugely outpaced `adamw` (24), although the two are nearly identical in cost.

**Cause, found in the timing data.** We looked at how long each single pass over the data took.
For `diag` — the very first optimizer to run — a pass took 15 to 20 seconds at the start of the
job and 26 to 37 seconds by the end. The same computation, on the same data, taking twice as
long later in the same run. Nothing in the algorithm can do that. The job had been placed on a
low-priority queue that borrows spare capacity on machines that other people are also using, so
the amount of computing power actually available to us drifted up and down over the 82 minutes
the job lasted. `ekfac` and `tekfac` happened to run during the worst stretch (70 to 90 seconds
per pass) and were starved of training time through pure bad luck.

**Fix.** Every other job in this project reserves a small slice of a graphics card, even when the
heavy work happens on the processor. That reservation is what gets you a machine that is yours
alone. We resubmitted with that reservation while still doing all computation on the processor,
so the numbers stay comparable with the BatchNorm run.

**Job 21130847: 44 minutes, 93.76% of the reserved capacity actually used, all seven optimizers
completed 30 or 31 passes, and the per-pass timing was flat.** Clean data.

---

## Step 3 — the clean GroupNorm vs BatchNorm comparison

Now everything except the normalization layer is identical. Measuring wobble as the typical
epoch-to-epoch change in validation accuracy:

| optimizer | GroupNorm | BatchNorm |
|---|---|---|
| diag | 1.9 pts | 4.0 pts |
| kfac | 3.3 | 4.2 |
| ekfac | 2.9 | 5.2 |
| tkfac | 2.7 | 4.4 |
| tekfac | 2.9 | 4.6 |
| adam | 1.5 | 2.2 |
| adamw | 1.6 | 2.3 |

BatchNorm is noisier in absolute terms — but it is noisier for **Adam too**, and it also reaches
much higher accuracy (73-74% versus 64-65%), which changes the scale on which wobble is measured.
Dividing each Fisher mode's wobble by Adam's own wobble in the same run removes both effects:

| optimizer | GroupNorm | BatchNorm |
|---|---|---|
| kfac | 2.15 | 1.87 |
| ekfac | 1.88 | 2.30 |
| tkfac | 1.76 | 1.94 |
| tekfac | 1.84 | 2.04 |

**Mixed result.** Three of the four modes get modestly worse with BatchNorm; `kfac` gets better.
The effect sizes are small, and each configuration was run once, so a difference of 0.2 could be
run-to-run chance. This did not settle the question.

---

## Step 4 — the finding that changed everything: *where* is the noise?

Instead of asking *which network component* causes the noise, we asked *which part of the
process* shows it. If the optimizer were taking erratic steps, the training loss would be
erratic too. If only the measurement on held-out data is erratic, the optimizer is fine and
something else is going on.

Measured on the stable second half of training, comparing like with like:

| what is measured | GroupNorm: Fisher ÷ Adam | BatchNorm: Fisher ÷ Adam |
|---|---|---|
| loss on each individual batch, step by step | **1.07** | **1.14** |
| training loss averaged per epoch | 1.41 | 1.13 |
| loss on held-out data | **3.67** | **4.52** |
| accuracy on held-out data | **2.58** | **3.22** |

**On the training side the Fisher modes are indistinguishable from Adam. On the held-out side
they are three to four and a half times noisier.** The optimization trajectory is smooth; only
generalization jumps around.

(One caveat: batch-to-batch loss carries a large shared noise floor — roughly 10% of the loss
level for every optimizer — which could hide a small training-side difference. But the
epoch-averaged training loss says the same thing.)

**This rules out the original hypothesis as the main cause.** The training-versus-held-out gap is
already large under **GroupNorm**, and GroupNorm keeps no memory across batches at all — it
normalizes each example on its own, so the function evaluated on held-out data is exactly the one
that was trained. The mechanism is present without BatchNorm.

### What is left as an explanation

The model is moving a lot in directions the training loss does not care about, but that change
its behaviour on unseen data. This is exactly what curvature-based optimization does **by
design**: it divides the step by the local curvature, so it takes *large* steps where the loss is
flat. Flat on the training data does not mean flat everywhere.

This matters more than it sounds, because the curvature estimate is extremely rank-deficient: an
earlier measurement in this project found the empirical Fisher matrix on one model to have rank
3 511 out of 26 634, meaning that in roughly 23 000 directions the estimate is essentially zero
and the step size there is set entirely by the safety constant `λ`. With `λ = 0.001` and a
learning rate of `0.001`, the step in such a direction can be as large as one full unit of
momentum — a thousand times the naive step.

Consistent with this reading: the Fisher modes reach **both** a lower training loss (1.04-1.07
versus 1.16) **and** a higher accuracy (64-65% versus 60-61%) than Adam here. They are not
overfitting. They travel faster and further, on a bumpier path.

*Later note: the explanation in this subsection did not survive measurement. Step 7 shows that
the steps are not larger in flat directions than in steep ones — they are the same size in every
direction. The facts of Step 4 (noise on the held-out side only) still stand; this reading of
them does not.*

### What BatchNorm does after all

It is not innocent, just demoted. It roughly doubles or triples the held-out wobble of
**everyone**, Adam included (×1.68 for Adam, ×2.10 for the Fisher modes on accuracy) — which is
the signature of its lagging internal statistics, an evaluation-side effect unrelated to the
optimizer. And it raises the Fisher modes' *relative* excess by about 25%.

---

## Step 5 — is it the 100-step refresh cycle? No.

The curvature estimate is recomputed only every 100 steps and held fixed in between. With about
352 steps per pass over the data, the optimizer only gets three or four different curvature
estimates per pass — so one unlucky estimate governs a third of a pass. A plausible source of
lumpy behaviour.

We measured the average size of the loss jump as a function of position in the 100-step cycle. If
refreshes caused jolts, the steps right after a refresh would stand out.

They do not. The excess near a refresh ranges from −6.8% to +9.0%, all within statistical noise
(largest z-score 1.8). And the small positive excess seen in the BatchNorm run appears equally in
`adam` and `adamw`, **which have no refresh cycle at all** — so it is a shared artefact, not a
refresh effect.

---

## Step 6 — is it the normalization layers? No.

The theory here was that normalization layers get a much cruder curvature model than the other
layers. For a dense or convolutional layer, the curvature is approximated by a structure that
matches the true object's shape. For a normalization layer it does not: the true curvature has a
different shape entirely, and to fit it into the shared machinery the whole input side is
compressed into **one single number** per layer. On top of that, these layers have outsized
influence — one of their parameters rescales an entire feature channel — so the crudest
approximation lands on the most sensitive parameters.

If that were the driver, networks with a larger share of normalization layers should be noisier.
Across nine runs:

| model | share of normalization layers | parameters | held-out noise, Fisher ÷ Adam |
|---|---|---|---|
| vit_micro_cifar | 33% | 21 k | 6.56 |
| cnn_gn_cifar (GroupNorm) | **0%** | 24 k | 3.67 |
| cnn_gn_cifar (BatchNorm) | 43% | 24 k | 4.67 |
| mlp_ln_mnist | 40% | 27 k | 1.56 |
| resnet20_cifar | 49% | 270 k | 3.32 |
| cct_2_3x2_cifar | 29% | 284 k | 1.74 |
| vit_small_cifar | 33% | 2.7 M | 2.04 |
| resnet50_cifar | 50% | 23.5 M | 1.32 |

Correlation: **+0.20**. None.

**The decisive counter-example came from a discovery about the code**, made while running this
test: **GroupNorm is not one of the layer types the optimizer handles at all.** Its parameters
receive no curvature treatment whatsoever — they fall back to plain momentum. So `cnn_gn_cifar`
has **zero** approximated normalization layers, and it is still among the noisiest models.

This also reframes the whole GroupNorm-versus-BatchNorm comparison of Step 3. It was never
"batch statistics versus per-example statistics". It was: *three layers left on plain momentum*
versus *three layers put through the crude approximation* (plus BatchNorm's own effects).
Switching them on costs about 25-30% more noise — a real effect, matching Step 4's independent
25% figure, but a small one.

---

## An unexpected lead: model size

The one variable that does track the noise is **how big the model is**: correlation **−0.68**
between the logarithm of the parameter count and the held-out noise ratio (−0.62 excluding a
degenerate run, −0.41 measured on accuracy instead of loss). The three smallest models
(21-27 k parameters) are the noisiest; the two largest (2.7 M, 23.5 M) are the calmest.
`mlp_ln_mnist` breaks the pattern.

This connects directly to Step 4's conclusion, which is why it is worth pursuing rather than
filing as a curiosity. The safety constant `λ = 0.001` is an **absolute** number, while the
actual scale of the estimated curvature depends on layer widths and activation magnitudes. What
governs the behaviour is the *ratio* between the two:

- If `λ` is large compared with typical curvature, it dominates everywhere, every direction gets
  roughly the same step, and the method quietly degenerates into plain momentum — calm, and not
  very second-order.
- If `λ` is negligible compared with typical curvature, the flattest directions get amplified by
  up to the full factor of a thousand — aggressive, and bumpy on held-out data.

So "model size" is probably standing in for "how `λ` compares to the real curvature scale". That
is the next measurement.

*Later note: it was measured (Step 7), and only the first of the two cases above ever occurs.*

---

## Step 7 — measuring `λ` against the curvature: `λ` wins everywhere, by a lot

### What was measured, in plain terms

Each Fisher mode keeps a running estimate of how curved the loss is in each direction of
parameter space. To take a step, it divides the gradient (averaged over recent steps) by
**curvature + `λ`**. The intent: where the loss is sharply curved, the divisor is large and the
step small; where it is flat, the divisor is small and the step large. `λ` is only meant as a
safety floor, so that a direction of near-zero curvature does not produce an enormous step.

The question was simply: **how big is the estimated curvature compared with `λ`?** If curvature is
usually much bigger, `λ` is a minor safety floor and the method really adapts to each direction.
If `λ` is usually much bigger, the divisor is essentially `λ` everywhere and the method does not
adapt at all.

### How

The saved checkpoints contain the network's weights but not the optimizer's running estimates, so
those estimates were rebuilt first: load the weights from halfway through training, run 1 000
training steps so the estimates fill up again (the length an earlier measurement in this project
showed to be necessary), then read them off. This was done for the five modes on seven trained
networks: `mlp_ln_mnist`, `mnist_autoencoder`, `cnn_gn_cifar` with GroupNorm and with BatchNorm,
`vit_micro_cifar`, `resnet20_cifar` and `cct_2_3x2_cifar` (cluster job 21178722, 7.5 minutes).
Script: `fisher_ref/experiments/lambda_vs_curvature.py`; raw results:
`fisher_ref/outputs/lambda_vs_curvature.json`.

### What came out

**On all seven networks, in all five modes, for 99% of directions, the estimated curvature is
below 3% of `λ`.** Put differently: once `λ` is added, the most-damped 1% and the least-damped 1%
of directions differ by at most 3% in how much their step is shrunk.

That already answers the question, but the 1% left over could in principle hold a handful of
strongly curved directions — often the ones that matter most. So on two of the networks the check
was repeated looking at the single **largest** curvature value in each layer:

| largest curvature in the whole network, as a fraction of `λ` | `diag` | `kfac` | `ekfac` | `tkfac` | `tekfac` |
|---|---|---|---|---|---|
| `mlp_ln_mnist` | 0.067 | 0.000017 | 0.00034 | 0.0011 | 0.00047 |
| `cnn_gn_cifar` (BatchNorm) | 0.070 | 0.000006 | 0.00048 | 0.00064 | 0.00058 |

**Not a single direction, in any layer, reaches `λ`.** Even the most curved direction the
optimizer knows about changes its step by at most 7% for `diag` and at most about 0.1% for the
four other modes.

### Why the curvature numbers are so small

This is not a bug in this project's code: it reproduces the original AdaFisher implementation
faithfully. Two inherited conventions each shrink the stored curvature, and they multiply.

1. **The running average keeps only a small fraction of what it sees.** At each update the stored
   estimate is multiplied by 0.08 and then gets 0.008 times the new measurement added. Repeat that
   long enough and the stored value settles at 0.008 ÷ 0.92 ≈ **1/115 of the actual quantity**. The
   four non-diagonal modes store the curvature as the product of *two* such averages, so they carry
   this shrinkage twice: about **1/13 000**.
2. **The gradient used is that of the batch-average loss.** Averaging over a batch of 128 examples
   divides each example's contribution to the gradient by 128, and curvature is built from the
   gradient *squared*, so that contribution shrinks by 128² ≈ **1/16 000**.

For `diag`, a rescaling step (each layer's values squeezed into the range 0 to 1 before
averaging) cancels the second effect but not the first. That gives a hard ceiling that does not
depend on the data at all: the curvature `diag` stores can never exceed (1/115)² ≈ 0.000076,
which is **7.6% of `λ = 0.001`**. The measured maxima, 6.7% and 7.0%, sit just under it.

As a rough order of magnitude only: without these two shrinkages, the largest curvature values
would be hundreds to thousands of times `λ`, and `λ` would sit in the middle of the curvature
range, which is where a safety floor is supposed to be.

### What this means

- **The five modes are, in effect, the same optimizer.** Since the divisor is essentially `λ`
  everywhere, each step is just the averaged gradient times `learning rate ÷ λ`, which is 1 at the
  default settings (1/3 for the two Transformers, which use `λ = 0.003`). That is plain momentum
  with a fixed step multiplier. The curvature estimates that distinguish the modes from one another
  are computed at every step, at real cost, and then barely used.
- **The explanation from Step 4 is refuted.** It needed large steps in flat directions and small
  steps in curved ones. There are no such differences: every direction gets the same step.
- **The model-size lead cannot work the way it was proposed.** It needed some networks to be in the
  regime where `λ` is negligible. None of them are.
- **An open lead, not tested:** Adam scales every parameter's step to a similar size whatever the
  gradient's magnitude. Plain momentum does not: its step is as large as the gradient is. So the
  Fisher modes' step sizes depend directly on how large gradients are in a given network, which
  could vary with network size. "Model size" might be standing in for gradient magnitude rather than
  for curvature.
- **A tension with an earlier result, not yet resolved.** On `mnist_autoencoder`, changing `λ` from
  0.00001 to 0.1 moved the four non-diagonal modes' final loss by less than 1%. If the step is
  simply `learning rate ÷ λ` times the gradient, that change multiplies the step size by 10 000 and
  should have mattered a lot. One possible reconciliation, not verified, is that those runs were
  stuck at a point where the gradient itself is almost zero, so that no step size helps.

### A trap in the first version of the measurement

The script as first run summarised each layer by the **middle** value (the median) of its
curvature. For `ekfac` and `tekfac` that median came out almost identical, to three digits, in
nearly every layer of every network. Such a coincidence signals that something other than the
network is being measured. The running estimates start from 1, and after the rebuild that
starting value has been multiplied by 0.08 nine times, leaving 0.08⁹ ≈ 1.3 × 10⁻¹⁰. The real
curvature in the middle of the range is smaller still, so the median was reporting what is left
of the starting value, not the curvature. The conclusion does not change, because that leftover
is itself far below `λ`. The two-network check above removes the leftover exactly before
measuring, and looks at the largest values rather than the median.

---

## Step 8 — the `λ` sweep re-read step by step: why it barely moved anything

### The loose end

Step 7 left one tension unresolved. On `mnist_autoencoder`, an existing run had already tried
three very different values of `λ` (0.00001, 0.001, and 0.1 — each 100 times the last) and found
that the four non-diagonal modes' final loss barely changed, by under 1%. If the step size really
is `learning rate ÷ λ`, multiplying `λ` by 10 000 should multiply the step by 10 000 too, and that
should be impossible to miss. It was not clear whether this meant the "step ≈ `learning rate ÷ λ`"
picture from Step 7 was itself wrong, or whether something else was going on.

### What was checked

No new training was needed — the three runs already existed, just not as saved output; they were
regenerated once (cluster job 21181255, about 8 minutes) and their step-by-step training loss was
read off and lined up side by side, step for step, for each of the five modes.

### What it showed

The three curves are **not** all identical, and they are **not** all spread apart proportionally
to `λ` either. They split unevenly:

- The two *smaller* values of `λ` (0.00001 and 0.001, still 100 times apart) track each other
  almost exactly, from the very first step.
- The *largest* value (0.1) is clearly slower to bring the loss down, for the first couple of
  hundred steps.
- After a few hundred steps, all three land on the same point and stay there for the rest of
  training.

For example, on `kfac`, training loss at step 100 is 0.882 (`λ`=0.00001), 0.921 (`λ`=0.001) and
1.031 (`λ`=0.1) — the two small values are close, the large one stands apart. By step 300 all
three read 0.71, and they stay there. The same pattern shows up in every mode.

### Why

This is the same "leftover of the starting value" mechanism flagged as a trap at the end of
Step 7. The optimizer's curvature estimate does not start at zero; it starts at a fixed default,
and that default is only reduced by a fixed multiplier (0.08) each time the estimate is refreshed
(roughly every 100 steps). Early in training, the number the optimizer actually divides the step
by is not `λ` — it is **whichever is bigger, `λ` or the leftover default**, and for the first few
refreshes the leftover default (starting at 1, then 0.08, then roughly 0.006, and so on) is far
bigger than a `λ` of 0.00001 or 0.001. Both of those runs are therefore governed by the *same*
`λ`-independent number for their first couple of hundred steps, which is exactly why their curves
overlap. A `λ` of 0.1, by contrast, is close enough to that leftover default to actually compete
with it from the first refresh onward, which is why only that run visibly slows down.

Once training has gone on long enough for the leftover default to fade below even the smallest
`λ` tested, `λ` should finally be in charge — and Step 7 already established that at that point
the real curvature is negligible next to `λ` for every mode tested, so what happens next is a
step size of `learning rate ÷ λ`, the same for every direction. That would still predict some
lasting difference between the three runs late in training, which is not what is seen (all three
end at the same point) — so this does not yet fully close the loose end, but it removes the part
of it that looked like a contradiction: the sweep's *lack of early effect* between 0.00001 and
0.001 is exactly what the startup mechanism predicts, not a sign that `λ` does not matter.

### What this means

- **The intuition "step ≈ `learning rate ÷ λ`, so a 10 000x sweep should move things a lot" was
  missing a term.** The actual divisor is closer to `λ` plus a startup leftover that fades on its
  own schedule and does not depend on `λ` at all. For a `λ` small enough to be dominated by that
  leftover during the part of training that decides where the loss ends up, changing `λ` further
  is invisible.
- **This sharpens, rather than replaces, Step 7's finding.** Step 7 showed that later in training
  `λ` dwarfs the real curvature; Step 8 shows that *early* in training, on top of that, `λ` itself
  can be dwarfed by this unrelated startup number. Two separate reasons for the same symptom —
  curvature-blindness — stacked on top of each other.
- **Next check, already running as of this writing:** build a plain momentum optimizer with no
  curvature estimate at all, but with a step size forced to follow the exact same startup-then-`λ`
  schedule, and see whether it reproduces a real Fisher mode's training closely — closer than the
  normal run-to-run variation between two random seeds of that same mode. If it does, every claim
  in this document about "the Fisher modes" at this project's default settings is really a claim
  about plain momentum with an unusual step-size schedule, not about curvature at all.

---

## Step 9 — does plain momentum with the matched schedule actually reproduce `kfac`? A mixed answer

### What was checked

The plain-momentum optimizer proposed at the end of Step 8 — same momentum, same weight decay,
same startup-then-`λ` step-size schedule, but with the curvature estimate replaced by nothing —
was built (`fisher_ref/experiments/warmup_sgd_baseline.py`) and run against real `kfac` on three
networks: the small MLP (`mlp_ln_mnist`, no convolutions), and two convolutional networks
(`cnn_gn_cifar`, `resnet20_cifar`). For each network: `kfac` twice, from two different random
starting points, to see how much two runs of the *same* method normally differ by chance; and the
plain-momentum stand-in once, from the *same* starting point and the same order of training
examples as the first `kfac` run, so the only thing that differs between that pair is whether
curvature is used at all (cluster job 21182482, 15 passes over the data each, about 4 minutes
total).

If plain momentum reproduces `kfac`, the gap between `kfac` and plain-momentum should be about the
same size as the gap between two `kfac` runs that only differ by chance.

### What came out

| network | kfac vs. kfac (different starting points, chance alone) | kfac vs. plain-momentum stand-in (same starting point) | how much bigger |
|---|---|---|---|
| `mlp_ln_mnist` (MLP, no convolutions) | test accuracy 97.56% vs. 97.45% (0.11 points) | test accuracy 97.56% vs. 97.45% (0.11 points) | **same size** |
| `cnn_gn_cifar` (small CNN) | test accuracy 61.32% vs. 61.36% (0.04 points) | test accuracy 61.32% vs. 62.64% (1.32 points) | **33 times bigger** |
| `resnet20_cifar` (bigger CNN) | test accuracy 86.27% vs. 86.09% (0.18 points) | test accuracy 86.27% vs. 84.78% (1.49 points) | **8 times bigger** |

**On the plain MLP, the stand-in is indistinguishable from real `kfac`** — the gap is no larger
than two `kfac` runs differ from each other purely by chance. Step 7 and Step 8's picture — that
at this project's default settings, curvature is drowned out and the method behaves as plain
momentum on a fixed schedule — holds up exactly here, on the one network with no convolutions.

**On both convolutional networks, it does not hold.** The stand-in is noticeably worse on
`resnet20_cifar` and, oddly, noticeably *better* on `cnn_gn_cifar` — not a consistent direction,
but in both cases a real, non-chance difference from real `kfac`. So the curvature estimate is
doing *something* on these two networks that plain momentum with the same schedule does not
capture, even though Step 7 measured its size as negligible almost everywhere.

### Why this is not a contradiction of Step 7

Step 7 measured a **size**: in over 99% of directions, curvature is a small fraction of `λ`. It
never measured whether the *shape* — which directions get slightly more or slightly less step, not
how much — still matters. A plain scalar step size, like the stand-in's, can only ever act the
same in every direction, whereas `kfac`'s preconditioner, even when it is nearly a fixed multiple
of the identity almost everywhere, is built from a matrix, not a number — the small departures
from "nearly a fixed multiple" are exactly the departures a plain scalar cannot reproduce. On a
network with no convolutions, those departures turned out not to matter. On both convolutional
networks, they did, in both directions (better on one, worse on the other) — consistent with a
handful of directions or layers carrying real, non-negligible curvature, small enough not to show
up in Step 7's "over 99% of directions" summary, but large enough to move final accuracy by more
than a percentage point on these two architectures.

### What this means

- **The "these methods are just plain momentum" reading from Step 7/8 does not generalize past the
  network it was cleanest on.** It holds essentially exactly for the MLP; it clearly does not for
  either convolutional network tried.
- **This reopens, rather than closes, the search for what makes curvature matter or not.** The
  natural next step is the one Step 7's own summary already flagged as the biggest blind spot in
  its own measurement: it reported the *typical* (99th-percentile) curvature-to-`λ` ratio, never
  the handful of largest values *per layer* on these two CNNs specifically (it was only checked on
  `mlp_ln_mnist` and `cnn_gn_cifar` with BatchNorm, not with GroupNorm, and not on
  `resnet20_cifar` at all) — a small number of outlier directions, invisible in a percentile
  summary, is exactly what could explain a result that is mostly "plain momentum" with a few
  directions that are not.
- **This is a separate question from the original one this document was written to answer** (why
  the *held-out* curves are noisier than the *training* curves). It does not change anything in
  the "Settled" list below. It changes how much weight Step 7/8's "these are all secretly the same
  optimizer" reading should carry when reasoning about convolutional networks specifically.

---

## Step 10 — checking every layer's largest value, on every network: the outlier-direction idea is ruled out

### The question

Step 9's own next step: Step 7 measured the *typical* (median) curvature everywhere, and Step 9's
CNN failures could plausibly come from the untested tail — a handful of directions or layers
whose curvature is not negligible, invisible in a median, but real enough to bend the trajectory
away from plain momentum's. Step 7 had only checked the single *largest* value per layer on two
networks (`mlp_ln_mnist`, `cnn_gn_cifar` with BatchNorm), by hand. This step checks all seven
networks `lambda_vs_curvature.py` already covers, not just those two, and specifically includes
`cnn_gn_cifar` with GroupNorm and `resnet20_cifar` — the two networks that actually failed Step 9.

### How

Same procedure as Step 7's two-network check, generalised to all seven networks and all five
modes: load each network's `ckpt_0.5`, re-warm the optimizer's running estimates for 1000 steps
(the length Step 7 already established as sufficient), then, for every hooked layer, take the
single largest value in its curvature spectrum and express it as a multiple of `λ`
(`fisher_ref/experiments/curvature_max_per_layer.py`, cluster job 21183572, under 3 minutes).

### What came out

**Largest curvature value found, as a multiple of `λ`, for the single worst layer of each network
and mode (35 (network, mode) pairs; "layers > λ" counts layers with even one direction whose
curvature alone, before `λ` is added, exceeds `λ`):**

| network | mode | max ÷ λ | worst layer | layer kind | layers with any direction > λ |
|---|---|---|---|---|---|
| `mlp_ln_mnist` | diag | 0.067 | `features.1` | LayerNorm | 0 / 5 |
| `mlp_ln_mnist` | kfac | 1.7 × 10⁻⁵ | `features.0` | Linear | 0 / 5 |
| `mlp_ln_mnist` | ekfac | 3.4 × 10⁻⁴ | `features.3` | Linear | 0 / 5 |
| `mlp_ln_mnist` | tkfac | 1.3 × 10⁻³ | `features.0` | Linear | 0 / 5 |
| `mlp_ln_mnist` | tekfac | 4.8 × 10⁻⁴ | `features.0` | Linear | 0 / 5 |
| `mnist_autoencoder` | diag | 4.2 × 10⁻⁶ | `layers.7` | Linear | 0 / 8 |
| `mnist_autoencoder` | kfac | 5.1 × 10⁻⁸ | `layers.7` | Linear | 0 / 8 |
| `mnist_autoencoder` | ekfac | 2.9 × 10⁻⁶ | `layers.7` | Linear | 0 / 8 |
| `mnist_autoencoder` | tkfac | 6.4 × 10⁻⁸ | `layers.7` | Linear | 0 / 8 |
| `mnist_autoencoder` | tekfac | 2.9 × 10⁻⁶ | `layers.7` | Linear | 0 / 8 |
| `cnn_gn_cifar` (GroupNorm) | diag | 0.052 | `head` | Linear | 0 / 4 |
| `cnn_gn_cifar` (GroupNorm) | kfac | 1.5 × 10⁻⁵ | `head` | Linear | 0 / 4 |
| `cnn_gn_cifar` (GroupNorm) | ekfac | 1.1 × 10⁻³ | `head` | Linear | 0 / 4 |
| `cnn_gn_cifar` (GroupNorm) | tkfac | 1.7 × 10⁻³ | `head` | Linear | 0 / 4 |
| `cnn_gn_cifar` (GroupNorm) | tekfac | 1.1 × 10⁻³ | `head` | Linear | 0 / 4 |
| `cnn_gn_cifar` (BatchNorm) | diag | 0.069 | `features.1` | BatchNorm2d | 0 / 7 |
| `cnn_gn_cifar` (BatchNorm) | kfac | 6.3 × 10⁻⁶ | `head` | Linear | 0 / 7 |
| `cnn_gn_cifar` (BatchNorm) | ekfac | 4.9 × 10⁻⁴ | `head` | Linear | 0 / 7 |
| `cnn_gn_cifar` (BatchNorm) | tkfac | 6.0 × 10⁻⁴ | `head` | Linear | 0 / 7 |
| `cnn_gn_cifar` (BatchNorm) | tekfac | 5.7 × 10⁻⁴ | `head` | Linear | 0 / 7 |
| `vit_micro_cifar` | diag | 0.024 | `norm` | LayerNorm | 0 / 15 |
| `vit_micro_cifar` | kfac | 3.4 × 10⁻⁷ | `head` | Linear | 0 / 15 |
| `vit_micro_cifar` | ekfac | 4.4 × 10⁻⁵ | `head` | Linear | 0 / 15 |
| `vit_micro_cifar` | tkfac | 4.5 × 10⁻⁵ | `head` | Linear | 0 / 15 |
| `vit_micro_cifar` | tekfac | 5.2 × 10⁻⁵ | `head` | Linear | 0 / 15 |
| `resnet20_cifar` | diag | 0.071 | `bn1` | BatchNorm2d | 0 / 39 |
| `resnet20_cifar` | kfac | 3.8 × 10⁻⁶ | `fc` | Linear | 0 / 39 |
| `resnet20_cifar` | ekfac | 3.1 × 10⁻⁴ | `fc` | Linear | 0 / 39 |
| `resnet20_cifar` | tkfac | 3.4 × 10⁻⁴ | `fc` | Linear | 0 / 39 |
| `resnet20_cifar` | tekfac | 2.9 × 10⁻⁴ | `fc` | Linear | 0 / 39 |
| `cct_2_3x2_cifar` | diag | 0.022 | `norm` | LayerNorm | 0 / 17 |
| `cct_2_3x2_cifar` | kfac | 4.8 × 10⁻⁷ | `fc` | Linear | 0 / 17 |
| `cct_2_3x2_cifar` | ekfac | 3.9 × 10⁻⁵ | `fc` | Linear | 0 / 17 |
| `cct_2_3x2_cifar` | tkfac | 6.1 × 10⁻⁵ | `fc` | Linear | 0 / 17 |
| `cct_2_3x2_cifar` | tekfac | 4.2 × 10⁻⁵ | `fc` | Linear | 0 / 17 |

**Not one single layer, on any of the seven networks, in any of the five modes, has even one
direction where curvature alone reaches `λ`.** The largest value found anywhere, across every
layer of every network checked — `diag` on `resnet20_cifar`'s first BatchNorm layer — is still
only 7% of `λ`. Every non-diagonal mode is far smaller still: their worst case anywhere is `kfac`
on `mlp_ln_mnist` at 0.0017% of `λ`, and every one of `resnet20_cifar`'s 39 layers and
`cnn_gn_cifar`'s 4-7 layers — the two networks where Step 9's stand-in optimizer actually
disagreed with real `kfac` — tops out at 0.001-0.17% of `λ` for the four non-diagonal modes.

### What this means

**The outlier-direction hypothesis from Step 9 is ruled out, cleanly.** It is not merely that a
*typical* direction is dominated by `λ` (Step 7) — the single most-curved direction found in
*any* layer of *any* network tested is too. There is no hidden tail of large-curvature directions
for the four non-diagonal modes to be finding and using on `cnn_gn_cifar` or `resnet20_cifar` that
a plain scalar step size would miss. Whatever made Step 9's stand-in disagree with real `kfac` on
those two networks, it is not "a few directions with real curvature, invisible in a median."

This reopens Step 9's question rather than closing it. Plausible remaining candidates, none
checked yet:
- **A per-layer, not per-direction, effect.** `kfac`'s divisor is a separate near-`λ` number for
  *each* layer, not one number for the whole network; Step 9's stand-in used one global number.
  Even if every layer's divisor is individually within a fraction of a percent of `λ`, deep
  networks are known to be sensitive to layer-to-layer step-size imbalance, and `resnet20_cifar`
  (39 layers) and `cnn_gn_cifar` have many more layers than `mlp_ln_mnist` (5) — where the stand-in
  worked.
- **Accumulation, not magnitude.** A systematic departure too small to move any one step
  noticeably could still compound over thousands of correlated steps, especially where a network
  has residual connections or batch statistics that carry information forward, which
  `mlp_ln_mnist` does not have and both CNNs do.
- **Only two seeds were used for each network's noise floor** (Step 9), so how tight that floor
  happens to be — `cnn_gn_cifar`'s was unusually tight, 0.04 accuracy points — is itself only
  known to the precision two runs can give it; a wider seed sweep would make the "8-33 times
  bigger than chance" comparison more trustworthy.

---

## Step 11 — the SGD stand-in's own schedule was itself only approximate; fixing it changes two of three verdicts

### What was wrong with the stand-in

Step 9's plain-momentum stand-in used a schedule guessed from the general *shape* of the
identity-seeded warmup (`Lambda + (1-gammas[0])^k`), not derived from `kfac`'s actual formula.
Real `kfac` preconditions with two *separately* damped factors multiplied together
(`B~⁻¹ M A~⁻¹`), not one divide, so once the data-driven part of each factor is negligible — which
Steps 7 and 10 established everywhere, not just typically — its true scalar limit is
`(decay^k + √λ)²`, not `λ + decay^k`. The two agree once `k` is a few `TCov` cycles in (both →
`λ`), but disagree substantially over the first 200-300 steps. A second, separate fix: on any
network with a layer type this optimizer does not handle at all (`cnn_gn_cifar`'s GroupNorm is
the one case among these three networks — see Step 6), real `kfac` gives those parameters plain
momentum with no division at all, not the warmup divisor; the stand-in now does the same, where
before it wrongly gave GroupNorm the same shrinking divisor as every handled layer.

### What changed in the result

Same protocol as Step 9, same three networks, same two seeds, only the stand-in's formula
corrected (`fisher_ref/experiments/warmup_sgd_baseline.py`, cluster job 21184339):

| network | old formula: gap vs. seed noise (loss / accuracy) | corrected formula: gap vs. seed noise (loss / accuracy) |
|---|---|---|
| `mlp_ln_mnist` | 0.26× / 1.00× | 1.04× / **0.00×** |
| `resnet20_cifar` | 5.41× / 8.28× | **1.91× / 2.83×** |
| `cnn_gn_cifar` | 134.82× / 33.00× | 175.85× / 46.25× |

**`mlp_ln_mnist`**, already a clean pass, is now essentially exact — zero difference in test
accuracy between real `kfac` and the stand-in. **`resnet20_cifar`** moves from a clear failure to
a close call, within about 2-3× of ordinary seed-to-seed variation. **`cnn_gn_cifar`** gets
*worse*, not better.

### What this means

Two of the three "failures" behind Step 9 and Step 10 were, at least partly, an artifact of
comparing real `kfac` against an imprecise stand-in — not a real property of those networks.
`resnet20_cifar` no longer clearly refutes "these methods are secretly all plain momentum"; it is
now borderline rather than a clear failure.

`cnn_gn_cifar` still refutes it, and more so after the fix. It is also the only one of the three
networks where the two corrections both apply at once (it is the only one with any
optimizer-unhandled layer type), so this result does not yet say which of the two fixes — the
formula, or the GroupNorm boundary — is responsible, or whether they partly cancel or reinforce
each other. That separation has not been done.

---

## Step 12 — separating Step 11's two fixes on `cnn_gn_cifar`: one helps, the other actively hurts

### The question

Step 11 bundled two independent fixes into one rerun and could not tell which one made
`cnn_gn_cifar` disagree with real `kfac` even more than before. This step reruns all four
combinations — {old, corrected} step-size formula crossed with {ignoring, respecting} the
hooked/unhooked layer-type boundary — on that one network, holding everything else fixed
(`fisher_ref/experiments/cnn_gn_ablation.py`, cluster job 21187037, under 3 minutes).

### What came out

Gap from real `kfac`, as a multiple of the two-seed noise floor (test loss / test accuracy):

| | old formula (`λ + decay^k`) | corrected formula (`(decay^k+√λ)²`) |
|---|---|---|
| **boundary ignored** (Step 9's original) | 134.8× / 33.0× | 194.5× / 54.0× |
| **boundary respected** (the GroupNorm fix) | **83.8× / 7.0×** | 175.9× / 46.3× (Step 11's rerun) |

**The two fixes pull in opposite directions, and cleanly so.** Respecting the hooked/unhooked
boundary — giving `cnn_gn_cifar`'s GroupNorm layers plain, undivided momentum instead of the same
shrinking divisor as everything else, which is what real `kfac`'s own fallback path already does
— helps substantially in both rows (134.8×→83.8×, 33.0×→7.0× in the top-to-bottom comparison
holding the old formula fixed). Correcting the step-size formula to `kfac`'s real two-factor math
*hurts* in both columns (comparing left-to-right at either row): it made `mlp_ln_mnist` and
`resnet20_cifar` better (Step 11) but makes `cnn_gn_cifar` worse, consistently, regardless of the
boundary fix.

### What this means

The anomaly is real, not a bookkeeping mix-up: it is specifically the *formula* correction that
disagrees with `cnn_gn_cifar`, in exactly the network where Step 9 already found the largest
mismatch. One candidate explanation, not yet checked: the corrected formula gives a noticeably
*larger* step than the old one for roughly the first two `TCov` cycles (steps 0-200) — every
measurement of "curvature is negligible next to `λ`" in this document (Steps 7, 8, 10) was taken
**mid-training**, at `ckpt_0.5`, on an already-warmed-up network. None of it says anything about
the first 200 steps of a *freshly initialized* network, which is exactly the phase where this
whole family of stand-ins is compared against real `kfac` (Steps 9, 11, 12 all retrain from
scratch). If `cnn_gn_cifar` — the smallest and shallowest of the three, at 4 layers — has genuinely
non-negligible curvature specifically in that early window, unlike the other two, then *no* scalar
`λ`-limit schedule can model it correctly there, and the old (slower-stepping) formula's better
showing would be a coincidence of that specific error happening to matter less, not evidence that
it is the more correct model.

This is a plausible lead, not a confirmed one — checking it needs a different measurement from
every one in this document so far (curvature at steps 0-200 of a fresh run, not at a mid-training
checkpoint), and is a fork away from the question this document was originally written to
answer. Recorded here as the concrete next step if this particular thread is picked back up; not
pursued further for now.

---

## Step 13 — checking the early-curvature lead directly: it does not explain `cnn_gn_cifar` either

### The question, and a needed correction to how it was framed

Step 12 left one lead: maybe curvature is not negligible next to `λ` in the first ~200-300 steps
of a *fresh* network, unlike every measurement in this document taken mid-training. That framing
turns out to be slightly wrong on its own terms: a scalar step-size schedule does not need
curvature to be *small* relative to `λ` — it needs curvature to be *uniform across directions*
(isotropic). A huge but perfectly uniform curvature is still exactly representable by a scalar.
The right question is whether curvature is **anisotropic** early on — whether some directions get
noticeably more or less shrinkage than others — not just how big it is.

### What was measured

A real, fresh `AdaFisherMulti(fisher_mode="kfac")` (no checkpoint) trained on each of the three
networks Steps 9/11/12 use, same seed, snapshotting the *undamped* curvature spectrum of every
hooked layer right after the 1st through 5th EMA update (absolute steps 0, 100, 200, 300, 400 —
`fisher_ref/experiments/early_curvature.py`, cluster job 21189379). Two numbers per layer per
snapshot: the largest value as a multiple of `λ` (magnitude), and the spread between its
99th and 1st percentile (`dyn_range`; 1 means flat/isotropic, larger means anisotropic).

| network | k=1 (step 0) | k=2 (step 100) | k=3 (step 200) | k=4 (step 300) | k=5 (step 400) |
|---|---|---|---|---|---|
| `mlp_ln_mnist`, worst-magnitude layer (`features.0`) | 165× / dyn_range 2.9 | 14.3× / 28 | 1.1× / 346 | 0.09× / 4 020 | 0.008× / 49 100 |
| `cnn_gn_cifar`, worst-magnitude layer (`features.8`) | 85× / dyn_range 2.0 | 6.2× / 13 | 0.43× / 121 | 0.03× / 672 | 0.003× / 1 150 |
| `cnn_gn_cifar`, **most anisotropic** layer (`head`) | 41.8× / dyn_range 6.5 | 2.6× / 62 | 0.18× / 696 | 0.013× / 7 040 | 0.001× / 41 800 |
| `resnet20_cifar`, worst-magnitude layer (`stages.2.2.conv1`) | 420× / dyn_range 2.7 | 38.5× / 29 | 2.75× / 175 | 0.19× / 487 | 0.009× / 509 |
| `resnet20_cifar`, **most anisotropic** layer (`fc`) | 80.1× / dyn_range 12.5 | 6.7× / 163 | 0.49× / 1 860 | 0.031× / 15 100 | 0.0005× / 35 800 |

Two things hold on every network: magnitude decays from tens-to-hundreds of `λ` at `k=1` to well
under `λ` by `k=4`-`k=5` (confirming Step 8's mechanism directly, not just inferring it from a
`λ` sweep) — and, at the same time, anisotropy climbs by several orders of magnitude, because the
uniform identity-seed leftover is what is fading, leaving the genuinely uneven, data-driven part
behind at ever-smaller absolute scale.

### What this rules out

**`cnn_gn_cifar` does not have a uniquely bad early-curvature problem.** Its worst-magnitude
layer is, if anything, *less* extreme at `k=1` than the other two networks (85× against `resnet20`
`'s 420× and `mlp_ln_mnist`'s 165×), and its most-anisotropic layer (`head`, the final classifier)
tracks `resnet20_cifar`'s equivalent layer (`fc`) closely in both shape and rough magnitude (e.g.
`k=2`: 2.6×/62 against 6.7×/163; `k=3`: 0.18×/696 against 0.49×/1860). Both networks show the same
qualitative picture — a genuinely non-negligible, genuinely anisotropic classification head for
roughly the first 200-300 steps — yet Step 11/12 found the schedule correction *helps*
`resnet20_cifar` and *hurts* `cnn_gn_cifar`. Early-training curvature, measured directly rather
than assumed, does not distinguish the two networks in the direction the anomaly needs.

### What is left

The one structural difference the data points to, not yet tested: **`cnn_gn_cifar` has 4 hooked
layers, `resnet20_cifar` has 39.** If a per-layer schedule mismatch is genuinely present in *both*
networks' classification head (which this step suggests), a 39-layer network has 38 other layers
across which that one layer's error can be diluted in the overall trajectory, while in a 4-layer
network the head *is* a quarter of the whole optimized network — the same-sized mismatch would be
expected to move the total result much more. This is a plausible reading of the measurements
above, not a confirmed one; it would need its own check (e.g. comparing the *fraction* of the
network's parameters or update norm that sits in the anisotropic layer, across networks) that has
not been run.

This closes the specific lead Step 12 proposed (mid-training curvature measurements being blind
to an early-training effect): the early effect is real and now measured directly, but it is
present on both the network that passed and the network that failed, so it is not, by itself, the
explanation for `cnn_gn_cifar` being the exception.

---

## Step 14 — the full picture: all five modes, on six of the eight benchmark networks

### The question

Steps 9, 11, 12 and 13 all used `kfac` as the one representative curvature-based mode, on three
networks. This step runs the same comparison — a real Fisher mode against the plain-momentum
stand-in with a matched step-size schedule, same seed, same init, same data order — for **all
five modes** (`diag`, `kfac`, `tkfac`, `ekfac`, `tekfac`) on **six** of the project's eight
implemented benchmark networks (`mlp_ln_mnist`, `cnn_gn_cifar`, `resnet20_cifar`,
`mnist_autoencoder`, `vit_micro_cifar`, `cct_2_3x2_cifar` — the two largest, `resnet50_cifar` and
`vit_small_cifar`, are still pending). Each network at its own default `λ` (1e-3 or 3e-3), 15
epochs, cluster job 21192489.

Getting a clean run took two corrections, both mechanical rather than scientific: `sbatch
--export=...` truncates any value containing a comma at the first comma, which silently dropped
five of the six networks and four of the five modes from an earlier attempt (only
`mlp_ln_mnist`/`diag` actually ran); and `WarmupMomentumSGD` needed decoupled-weight-decay support
to compare fairly against the three networks (`vit_micro_cifar`, `cct_2_3x2_cifar`, and later
`vit_small_cifar`) that use the AdaFisherW convention.

### What came out

30 (network, mode) cells (`mnist_autoencoder` excluded from the verdict below — every mode
collapses to the same stuck point regardless of optimizer, established earlier in this document,
so it cannot distinguish anything here). Classified as **PASS** when the stand-in's relative test-
loss gap from the real mode is under 1%, **FAIL** when that gap exceeds 1% *and* is more than 10x
the two-seed noise floor, everything else **borderline**:

| network | diag | kfac | tkfac | ekfac | tekfac |
|---|---|---|---|---|---|
| `mlp_ln_mnist` | borderline (1.9%) | borderline (5.1%) | borderline (6.5%) | PASS (0.2%) | PASS (0.3%, ratio inflated by a near-zero seed gap) |
| `cnn_gn_cifar` | borderline (3.0%) | **FAIL (4.3%, 176x noise)** | PASS (0.5%) | PASS (0.2%) | PASS (1.0%) |
| `resnet20_cifar` | borderline (3.6%) | borderline (1.8%) | borderline (2.1%) | borderline (3.9%) | borderline (2.0%) |
| `vit_micro_cifar` | borderline (2.2%) | borderline (1.1%) | PASS (0.3%) | PASS (~0%) | PASS (~0%) |
| `cct_2_3x2_cifar` | borderline (1.4%) | PASS (0.5%) | **FAIL (1.6%, 11x noise)** | PASS (0.3%) | PASS (0.7%) |

(percentages are the relative test-loss gap between the real mode and its stand-in; "ratio" is
that gap divided by the gap between two seeds of the real mode)

**Eleven cells pass cleanly, twelve are a small (1-6%) but not clearly-resolved gap, and two are
unambiguous failures** — `cnn_gn_cifar`/`kfac` (already known since Step 9) and, newly found here,
`cct_2_3x2_cifar`/`tkfac`. No third network/mode combination reaches that clarity of failure.

### What this means

- **"AdaFisherMulti behaves as plain momentum with a fixed step multiplier" is the majority
  finding, not a universal one.** It holds cleanly on 11 of 30 cells and approximately (a few
  percent off) on another 12; it clearly does not hold on 2.
- **`resnet20_cifar` is a new, different kind of exception from `cnn_gn_cifar`.** Every one of its
  five modes lands in "borderline", never a clean pass — a small, *consistent*, mode-independent
  gap (1.8-3.9%), unlike `cnn_gn_cifar` and `cct_2_3x2_cifar`, where four of five modes pass
  cleanly and exactly one fails outright. This is a different shape of exception (spread evenly
  across every mode) from the other two (concentrated in one specific mode) and is not explained
  by anything measured so far in this document.
- **The `cnn_gn_cifar`/`kfac` anomaly is not just about the schedule formula.** `tkfac` — which
  shares `kfac`'s two-factor-product structure and uses the *same*, mathematically-derived
  schedule formula in the stand-in — passes cleanly on `cnn_gn_cifar` (0.5%) while failing on
  `cct_2_3x2_cifar` (1.6%, 11x noise) instead. Whatever is behind these two failures, it is not a
  property of the two-factor schedule formula in general; each failure is specific to its own
  (network, mode) pair.
- **`diag` is borderline almost everywhere it is informative** (5 of 5 non-stuck networks), never
  a clean pass, never a clear fail — consistent with Step 7's own finding that `diag`'s curvature
  ceiling (up to 7% of `λ`) is structurally higher than the other four modes' (under 0.2%): a
  small, real, mode-specific gap is exactly what that difference in ceiling predicts.

---

## Step 15 — pushing `λ` down: real divergence at first, then just breakage

### The question

If curvature is negligible next to `λ` at the project's default (Steps 7, 8, 10), shrinking `λ`
should let curvature start to matter — and, per Step 9-12's own machinery, should widen the gap
between a real Fisher mode and the plain-momentum stand-in. Does it, and how far can `λ` be
pushed before something else takes over?

### What was measured

The same real-mode-vs-stand-in comparison as Steps 9/11/12/14, at four values of `λ` (1e-4, 3e-5,
1e-6, 1e-8 — each roughly 10x smaller than the last, `1e-4` being the mildest, about 10x below
the project's own default), on the same six networks and five modes as Step 14 (cluster jobs
21201679/82/84/86, after two fixes: `sbatch --export` truncating comma-separated values at the
first comma — which silently dropped every network but the first from two earlier attempts — and
a crash-resilience wrapper around each individual run, so one singular matrix no longer aborts
every model and mode still queued behind it).

### What came out: two different failure modes, and a clean boundary

| | `λ=1e-4` | `λ=3e-5` | `λ=1e-6` | `λ=1e-8` |
|---|---|---|---|---|
| `mlp_ln_mnist` / `diag`, `ekfac`, `tekfac` | trains normally | trains normally | trains normally | **collapses** to chance level |
| `mlp_ln_mnist` / `kfac` | trains normally | **crashes** | crashes | crashes |
| `mlp_ln_mnist` / `tkfac` | trains normally | trains normally | **crashes** | crashes |
| `cnn_gn_cifar`, `resnet20_cifar` (all 5 modes) | trains normally | mixed | mostly crashes or collapses | crashes/collapses |
| `vit_micro_cifar` (all 5 modes) | trains normally | mostly crashes/collapses | crashes/collapses | crashes/collapses |
| `cct_2_3x2_cifar` (all 5 modes) | **already** crashes or collapses | crashes/collapses | crashes/collapses | crashes/collapses |

Two distinct things happen, not one:

- **`kfac` and `tkfac` crash outright** — a real Python exception, `torch.linalg.inv`: "the input
  matrix is singular", from a Kronecker factor with an exact zero on the diagonal (the same
  exactly-rank-deficient structure this project has hit before, e.g. `mlp_ln_mnist`'s dead MNIST
  pixels). Both modes invert their factors directly with no conditioning step; `_eigh_utils.py`'s
  ridge exists for `ekfac`/`tekfac`'s eigendecomposition, not for these two.
- **`diag`, `ekfac`, `tekfac` never crash — they silently collapse** to chance-level performance
  (test loss pinned at `ln(number of classes)`) once `λ` is too small. No exception, no warning:
  the run "succeeds" and reports a number, which is why this needed checking directly rather than
  trusting the earlier ratio-only view.

**That collapse breaks the ratio metric itself.** Once both the real mode and its stand-in have
collapsed to the same chance-level point, the gap between them is small *because both are stuck at
the same trivial place*, not because the stand-in is a good model. `mlp_ln_mnist`/`diag` at
`λ=3e-5` reports a 0.20x ratio and a 0.00% relative gap — worse in every practical sense than
`λ=1e-4`'s 11.95x/17.9%, but it *looks* like a clean pass by the numbers alone. Every ratio in
Steps 9-15 that comes from a collapsed run is meaningless in this way, and must be read against
the actual test-loss/accuracy value, not the ratio in isolation.

`cct_2_3x2_cifar` stands out again: it is the only network that already fails at the *mildest*
value tested, `λ=1e-4` — consistent with Step 14's finding that this network's `tkfac` cell was
already a borderline failure at its own default `λ` (3e-3), before any reduction at all.

### The one informative value: `λ=1e-4`

At `λ=1e-4` (about 10x below default), four of the six networks are still training to a
reasonable accuracy on every mode, and most of those cells now show a real, moderate divergence
from the stand-in — a few percent to about 18% relative gap on `mlp_ln_mnist`, single-digit-to-
double-digit ratios on several `cnn_gn_cifar`/`resnet20_cifar` cells (Step 14's own table).
That is the predicted effect, seen directly rather than argued from Step 7's magnitudes: shrinking
`λ` by one order of magnitude does start to reveal curvature-driven behaviour, mildly.

### What this means

- **The original hypothesis is confirmed, in a narrow band.** `λ` an order of magnitude below
  default is where curvature starts to matter (matching Step 10's own crossover estimates, which
  put most modes' crossover one to three orders of magnitude below default). Below that band, the
  sweep stops answering the question it was built to answer.
- **Two further orders of magnitude down does not reveal *more* curvature-driven adaptivity — it
  just breaks the optimizer**, in one of two ways depending on the mode's own numerical structure
  (crash for the two directly-inverted modes, silent collapse for the other three). This is a
  genuine, previously-undocumented boundary of the current implementation, not a property of the
  underlying curvature.
- **This closes the "would a smaller `λ` show more?" thread from earlier in this document with a
  concrete answer**, rather than leaving it as a hypothetical: yes, mildly, at `λ~1e-4`; no further
  useful signal below that.

---

## Step 16 — the mechanism explains `kfac`/`cnn_gn_cifar`, not `tkfac`/`cct_2_3x2_cifar`

### The question

Step 14 found two clear failures — `kfac` on `cnn_gn_cifar`, `tkfac` on `cct_2_3x2_cifar` — and one
network, `resnet20_cifar`, that never fails outright on any of its five modes, just sits a few
percent off on all of them. Step 13 measured that curvature is genuinely anisotropic in the first
few hundred steps of a fresh run, but never on the *applied* (damped) spectrum — the matrix a
scalar stand-in actually cannot reproduce — and never on `cct_2_3x2_cifar` at all. This step closes
that gap.

### What was measured

A real, fresh `AdaFisherMulti` for every one of the five modes, on `cnn_gn_cifar`,
`cct_2_3x2_cifar` and `resnet20_cifar`, snapshotting the *applied* spectrum's `dyn_range` (p99/p01
of the damped, actually-inverted-and-used matrix) at the same five early checkpoints as Step 13
(`fisher_ref/experiments/early_conditioning.py`, cluster job 21214956, under 5 minutes). `dyn_range
= 1` means the applied step is isotropic everywhere — exactly what a scalar stand-in can match,
however large; `> 1` means some directions get a genuinely different amount of shrinkage than
others, which no scalar can reproduce.

### What came out

Worst-layer applied `dyn_range`, `k=1..5` (steps 0, 100, 200, 300, 400):

| network / mode | k=1 | k=2 | k=3 | k=4 | k=5 | Step 14 verdict |
|---|---|---|---|---|---|---|
| `cnn_gn_cifar` / `kfac` | 4.92 (`head`) | **8.44** | 3.99 | 1.84 | 1.23 | **FAIL** (4.3%, 176x) |
| `cnn_gn_cifar` / `tkfac` | 1.00 | 1.00 | 1.00 | 1.00 | 1.01 | PASS (0.5%) |
| `cnn_gn_cifar` / `ekfac`, `tekfac` | 1.00 | 1.00 | 1.00 | 1.00 | ~1.0 | PASS |
| `cct_2_3x2_cifar` / `kfac` | 2.08 (`tokenizer.blocks.0`) | 2.84 | 1.73 | 1.22 | 1.06 | PASS (0.5%) |
| `cct_2_3x2_cifar` / `tkfac` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **FAIL** (1.6%, 11x) |
| `resnet20_cifar` / `kfac` | 9.05 (`fc`) | **16.0** | 6.11 | 2.28 | 1.27 | borderline (1.8%) |
| `resnet20_cifar` / `tkfac`, `ekfac`, `tekfac` | ~1.00 | ~1.00 | ~1.00 | ~1.00 | ~1.0 | borderline (2.0-2.1%) |
| `resnet20_cifar` / `diag` | 1.16 | 1.14 | 1.06 | 1.06 | 1.06 | borderline (3.6%) |

**`kfac` cleanly explains itself.** On both networks where it is tested, `kfac` is the one mode
whose applied step is measurably anisotropic in the first ~200 steps (peaking 8-16x); every other
mode's applied step is flat (`dyn_range≈1`) at every checkpoint, on every network checked. This
lines up exactly with `cnn_gn_cifar`/`kfac` being the clear failure there.

**`tkfac`/`cct_2_3x2_cifar` is not explained by this mechanism — it is the opposite pattern.**
`tkfac`'s applied step is flat (1.00 at every checkpoint) on `cct_2_3x2_cifar`, the network where
it *fails*, and `kfac` — which shows real, if modest, anisotropy there (peaking 2.84x) — is the
mode that *passes*. Early-training conditioning does not distinguish these two the way it
distinguishes `cnn_gn_cifar`'s pass/fail split. Whatever makes `tkfac` fail specifically on
`cct_2_3x2_cifar`, it either happens after step 400 or is not a conditioning effect at all.

**`resnet20_cifar`'s uniform borderline-ness is only half-explained.** `kfac` shows anisotropy of
a similar shape and magnitude to `cnn_gn_cifar`'s (peak 16.0x against 8.44x) — consistent with the
same mechanism being present but *diluted*: `fc` is 1 of `resnet20_cifar`'s 39 hooked layers
(~2.6% of them) against `cnn_gn_cifar`'s `head` being 1 of 4 (25%), a ~10x difference in the
layer's share of the network, in the same direction as the ~90x difference in outcome severity
(176x noise vs. 1.91x). This is consistent with, though does not by itself prove, the
per-layer-count dilution idea Step 12/13 could not confirm. But it only accounts for `kfac`: the
other four modes on `resnet20_cifar` show **no** early anisotropy at all (flat `dyn_range≈1`,
same as their flat behaviour on the other two networks) and still land in the same few-percent
borderline band as `kfac` does. Whatever makes `diag`/`tkfac`/`ekfac`/`tekfac` a few percent off
on `resnet20_cifar`, it is not early-training conditioning either — consistent with something
that accumulates over the *whole* run across many layers, rather than a single early anisotropic
step, but that accumulation itself has not been measured.

### What this means

- The Step 14 anomalies are **not one mechanism wearing two faces.** `kfac`/`cnn_gn_cifar` has a
  strong correlational candidate (early applied-step anisotropy in one layer of a shallow
  network, present exactly where it fails and absent everywhere it does not). `tkfac`/
  `cct_2_3x2_cifar` does not share it, and needs its own explanation, not yet found.
- `resnet20_cifar`'s mode-independent borderline gap is real and still open. The one candidate
  this step rules out is early-training conditioning on any single layer, for any of the four
  modes that show no anisotropy there — leaving whole-run accumulation across its 39 layers as
  the remaining, still-untested candidate.
- **Correlation, not yet causation.** That `kfac` is anisotropic exactly where it fails does not
  by itself show the anisotropy is *why* it fails. A same-seed ablation to test this properly —
  force `kfac` isotropic for steps 0-400 only, then switch to unmodified `kfac` for the rest of
  training, so any difference in the final outcome can only trace back to that window — was
  designed but deliberately not run in this pass (see the "Open, parked" list below for the
  design and why a simpler, whole-run version of the same ablation was rejected as confounded).

---

## What is settled, and what is not

**Settled:**
- The extra noise is on the generalization side, not the optimization side.
- It is not caused by the 100-step refresh cycle.
- It is not proportional to the share of normalization layers.
- BatchNorm contributes about 25-30%, partly through an evaluation-side effect that hits Adam too.
- The Fisher modes are *better* than Adam here on both training loss and final accuracy.
- `λ` dwarfs the estimated curvature in every direction, every mode, every network measured, so
  all five modes behave as plain momentum with a step multiplier of `learning rate ÷ λ`. The
  cause is two inherited conventions that shrink the stored curvature (Step 7), not a bug.
- Consequently, the noise is **not** caused by larger steps in flat directions, and the
  `λ`-to-curvature ratio cannot explain the spread across models.
- The `λ` sweep on `mnist_autoencoder` barely moving the results is not a contradiction of "step
  ≈ `learning rate ÷ λ`". For roughly the first 300 steps the step size is set by a startup
  leftover that fades on its own schedule, independent of `λ`; only a `λ` large enough to compete
  with that leftover (0.1, not 0.00001 or 0.001) has a visible early effect (Step 8). What still
  is not explained is why the runs converge to the same point late in training too.
- A plain-momentum optimizer given the same startup-then-`λ` step-size schedule and no curvature
  estimate at all reproduces real `kfac` almost exactly on a network with no convolutions
  (`mlp_ln_mnist`), within normal seed-to-seed variation. It does **not** reproduce `kfac` on
  either convolutional network tried (`cnn_gn_cifar`, `resnet20_cifar`): the gap from real `kfac`
  is 8 to 33 times larger than two seeds of `kfac` differ from each other (Step 9). "These methods
  are secretly all plain momentum" is confirmed on one architecture and refuted on two others.
- That refutation is not explained by a hidden tail of high-curvature directions. Checked on
  every layer of all seven networks: nowhere does even the single most-curved direction reach
  `λ` — 7% of it at most, for `diag`; under 0.2% for every other mode, on every network, including
  the two that refuted Step 9 (Step 10).
- Part of Step 9's refutation was itself a measurement artifact: the stand-in's own step-size
  formula was an approximation, not derived from `kfac`'s real two-factor math. Correcting it
  makes `mlp_ln_mnist` match exactly and turns `resnet20_cifar` from a clear failure into a close
  call. `cnn_gn_cifar` is unaffected by this correction — if anything it disagrees more (Step 11).
- On `cnn_gn_cifar` specifically, that formula correction and the separate layer-type-boundary fix
  pull in opposite directions: the boundary fix helps (best case 7× the noise floor on accuracy),
  the formula fix hurts, in every combination of the two tested (Step 12).
- Curvature in the first 0-300 steps of a fresh network is real and genuinely anisotropic, on
  both `cnn_gn_cifar` and `resnet20_cifar` — but the two networks' early patterns look alike, so
  this is not what makes `cnn_gn_cifar` react the opposite way to the schedule fix (Step 13).
- Widened to all five modes on six of the eight benchmark networks: "behaves as plain momentum"
  passes cleanly on 11 of 30 (network, mode) pairs, is a few percent off on 12, and is a clear
  failure on 2 — `cnn_gn_cifar`/`kfac` (known) and `cct_2_3x2_cifar`/`tkfac` (new). `resnet20_cifar`
  is not a `kfac`-specific exception: all five of its modes land in the same small, few-percent
  "borderline" band, never a clean pass (Step 14).
- Shrinking `λ` one order of magnitude below default (to `1e-4`) reproduces the predicted effect
  directly: most networks still train, and several modes show a real, moderate divergence from
  the plain-momentum stand-in. Two further orders down reveals nothing more — `kfac`/`tkfac` hit
  an exactly-singular Kronecker factor and crash (`torch.linalg.inv`: "matrix is singular", as
  high as `λ=3e-5`, far above the ~1e-8 a pure curvature-vs-`λ` estimate predicted), while `diag`/
  `ekfac`/`tekfac` never crash but silently collapse to chance-level performance instead. Ratios
  computed from a collapsed run are meaningless — both sides are stuck at the same trivial point
  — which is why some earlier-looking "good" ratios in this document need re-reading against the
  actual loss/accuracy value, not the ratio alone (Step 15).
- `kfac`'s failure on `cnn_gn_cifar` has a found, measured cause: it is the only mode, on either
  network checked, whose *applied* step is genuinely anisotropic (up to 8-16x) in the first ~200
  steps of a fresh run; every other mode's applied step is flat there. `tkfac`'s failure on
  `cct_2_3x2_cifar` is **not** explained by the same mechanism — its applied step is flat exactly
  where it fails, and `kfac` (which passes there) is the one showing mild early anisotropy instead
  (Step 16).

**Open, deliberately parked for now — not being chased in this pass:**
- **Step 16's finding is correlation, not yet demonstrated causation.** `kfac` is the only mode
  showing early applied-step anisotropy on `cnn_gn_cifar`, and the only mode that fails there;
  that is not the same as showing the anisotropy is *why* it fails. A same-seed, same-data-order
  ablation was designed to test this properly but not run: force `kfac`'s damped factors to their
  own trace-matched, direction-blind average (same overall magnitude, no direction-dependence)
  for steps 0-400 only, then switch to unmodified `kfac` for every step after — so any difference
  in the *final* outcome between that run and real `kfac` can only be attributed to the forced
  window, since both runs use identical code from step 400 onward (an earlier, simpler design —
  forcing isotropy for the whole run — was rejected: it would permanently change `kfac`'s
  behaviour everywhere, confounding "the early window mattered" with "some other, later effect of
  the same change mattered instead"). If the ablated run ends up close to real `kfac`, the early
  window is not load-bearing and the true cause is later in training; if it ends up close to the
  plain-momentum stand-in instead, the early window is confirmed as causal. Needs a small,
  temporary subclass or monkeypatch of `KFACApproximation` — not implemented.
- What makes `tkfac` fail specifically on `cct_2_3x2_cifar`, now that early-training conditioning
  has been ruled out as the cause (Step 16) — it must be something that happens later in training,
  or is not a conditioning effect at all.
- What makes `resnet20_cifar` sit a few percent off on *every* mode. `kfac`'s own contribution
  fits a per-layer-count dilution story (Step 16: its one anisotropic layer, `fc`, is ~2.6% of
  `resnet20_cifar`'s 39 hooked layers against ~25% of `cnn_gn_cifar`'s 4, in the same direction as
  the ~90x gap in outcome severity) — but `diag`/`tkfac`/`ekfac`/`tekfac` show **no** early
  anisotropy on `resnet20_cifar` at all and still land in the same borderline band, so dilution of
  a single early effect cannot be the whole story for those four. The remaining candidate,
  unchecked: something that accumulates across many layers over the *whole* run rather than
  showing up as one anisotropic layer early on.
- What does cause the held-out noise, the question this document was originally written to
  answer. The untested lead: plain momentum's step grows with the gradient's magnitude, which
  Adam's does not. Everything from Step 9 onward is a detour into whether the five Fisher modes
  are secretly plain momentum at all, not yet a return to this original question.
- Whether the modes become genuinely different, and better or worse, once the curvature is kept
  at its true scale in a *stable* regime, rather than the unstable one Step 15 found starting two
  orders of magnitude below default. `λ~1e-4` is the one value measured so far that is both
  informative and (mostly) stable; nothing between the default and total breakdown has been swept
  finely.
- The two largest benchmark networks, `resnet50_cifar` and `vit_small_cifar`, have never been run
  through the Step 14/15 comparisons at all (cost, not a decision that they are uninteresting).
  Six of the project's eight networks are covered; two are not.

**Caveats that apply to everything above:**
- One random seed per configuration. No error bars anywhere.
- Five of the nine runs in the model comparison predate the learning-rate schedule fix.
- `mnist_autoencoder` appears in some tables as a near-zero outlier; its Fisher modes are known
  to be stuck, so its curves are flat for reasons unrelated to this investigation.

---

## Practical notes worth keeping

- **Never submit a timing-sensitive job to a shared, low-priority queue.** Reserve a graphics-card
  slice even for processor-only work; on this account that is the only way to get a machine to
  yourself. Compare the two attempts: 86.91% capacity used with drifting per-pass times, versus
  93.76% and flat.
- **The symptom to watch for** is per-pass duration changing over the life of a single job. If the
  same computation gets slower, the measurement is contaminated, whatever the exit code says.
- **GroupNorm parameters are not curvature-preconditioned** by this optimizer. Any result
  involving `cnn_gn_cifar` in its default configuration must be read with that in mind.
- **When a summary number comes out identical across unrelated layers or networks, suspect it is
  measuring the setup rather than the network.** Here it was the leftover of the running
  estimates' starting value (Step 7).
- **Compare `λ` with the curvature the optimizer actually stores, not with the textbook
  quantity.** The stored curvature is smaller by a factor of about 115 per running average and
  128² for the batch average, so a `λ` that looks small on paper can dominate completely.
