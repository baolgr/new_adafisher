# Campaign 2: CIFAR-100, ImageNet, and the two large CIFAR-10 models re-run

*The results of the training jobs that ran on 20 and 21 September 2026, read in one place. Plain
language throughout. Every number here was measured on the cluster, unless it is labelled otherwise.
Every result rests on **one seed** (seed 0), and every section says what that allows.*

---

## The short version

We trained six image classifiers on CIFAR-100 and six on ImageNet. We also re-ran the two large
CIFAR-10 models, whose first run had been made before two protocol fixes. Each benchmark has seven
arms: the five Fisher modes (`diag`, `kfac`, `ekfac`, `tkfac`, `tekfac`) plus `adam` and `adamw`.
Every arm gets the same wall-clock time. That makes 14 benchmarks and 98 arms, all complete.

1. **The data is sound.** Over 8.4 million recorded steps there is not one non-finite loss. The
   clock never runs backwards. No arm overshoots its time budget by more than 0.35 s, which is less
   than one of its own steps.
2. **The Fisher modes beat the better of the two Adam baselines on 9 of the 14 benchmarks.** The
   margins run from +0.7 to +21.0 points of best validation accuracy. The largest single result is
   CCT on CIFAR-100: 55.8% against 34.8%.
3. **They lose on 5 benchmarks, and the losses have a pattern.** ViT-S loses to `adamw` on all three
   datasets: CIFAR-10 (−4.1 points), CIFAR-100 (−4.7) and ImageNet at 224 px (−6.8). Two of the four
   small networks trained on ImageNet downsampled to 32x32 also lose: the small GroupNorm CNN (−5.7)
   and ResNet-20 (−1.5). The last one is inside the seed-to-seed noise; the other four are not.
4. **ResNet-50 trained on ImageNet at 224 px for 30 epochs reaches 74.2% on the official ILSVRC
   validation set with `diag`, against 70.9% for `adamw` and 65.0% for `adam`.** The gap between
   `diag` and `adam` is +9.25 points. The AdaFisher paper reports +9.17 for the same pair, after 90
   epochs rather than 30 (Table 3). The two protocols differ, so this is context and not a
   reproduction. But the size of the gap comes back.
5. **The five modes split into the same two groups as in the seed table.** `{diag, kfac, tkfac}`
   rank ahead of `{ekfac, tekfac}` in 72 of the 84 possible cross-group comparisons. `tkfac` beats
   `tekfac` on 14 benchmarks out of 14. The two losing modes are again the two that fit the fewest
   steps into the same time: 0.88 of `diag`'s count, on average, against 0.95 to 0.96 for the
   others.
6. **The learning-rate schedule fix did what it was predicted to do on ResNet-50/CIFAR-10.** Before
   the fix, `ekfac` and `tekfac` were cut off mid-way down the cosine and read 90.0% and 90.8%.
   After it they read 94.1% and 93.8%, that is +4.1 and +3.1 points. The spread between the five
   modes on that model shrank from 3.9 points to 0.5.
7. **Between the two Adam baselines, `adam` collapses where the weight decay is large.** On the
   three architectures run with weight decay 0.01 (the two ViTs and CCT), `adam` trails `adamw` by
   3.8 to 53.5 points. On ImageNet it reaches only 0.8% on the micro ViT, where guessing gives 0.1%,
   and 6.3% on ViT-S against 59.7% for `adamw`. `adam` adds the weight decay to the gradient, and
   `adamw` applies it separately. That this causes the collapse is a likely explanation, but it has
   not been tested. It does mean `adamw` is the baseline to compare against on those benchmarks.

**What this campaign does not say.** All seven arms run at their default settings, including the
default safety constant `λ` (the number added to every curvature value before dividing by it).
`plan_lambda_dominance.md` and `plan_exp_lot5.md` measured that, at this `λ`, the curvature
estimate barely changes the step on the four networks where it was measured: the modes behave
almost like plain momentum. So "the Fisher modes win" is a statement about these optimizers **as
shipped**, not about curvature. And one seed cannot resolve differences smaller than the
seed-to-seed spread. That spread has been measured between 0.04 and 2.4 points depending on the
network, so any single gap under about 2 points below is not a finding.

---

## 1. What was run

### 1.1 The protocol

Every benchmark follows AdaFisher's own wall-clock-time protocol (`adafisher_2405.16397.pdf` §5).
The reference arm, `diag`, trains for a fixed number of epochs. Its measured time becomes the budget
for the other six arms. Each of those trains until the budget runs out, however many steps that is.

Two details matter for reading the numbers.

- **Every budgeted arm uses `--lr-schedule budget`.** Its cosine learning-rate curve is stretched
  over its own time budget, so each arm completes exactly one full cosine. The learning rate
  recorded at the last epoch of every budgeted arm is between `5e-15` and `8e-9` on CIFAR and at
  most `2e-6` on ImageNet, starting from `1e-3` or `1e-4`. So no arm is cut off mid-way. That is the fix the campaign-1
  audit asked for (`CLAUDE.md`, "Campaign-1 audit").
- **"Validation" and "test" are two different held-out sets.** On CIFAR, validation is 5 000 images
  carved out of the training set by a seeded split, and test is the official 10 000-image test set.
  On ImageNet, validation is 25 000 images carved out of the 1.28 M training images, and test is
  the official 50 000-image ILSVRC validation set. The published ImageNet numbers, such as the
  AdaFisher paper's Table 3, are therefore comparable with our **test** column.

### 1.2 The settings

The settings are each benchmark's own defaults, identical to its CIFAR-10 counterpart except for
the number of classes. Two families of settings exist.

| family | benchmarks | Fisher `lr` | baseline `lr` | weight decay | `λ` |
|---|---|---|---|---|---|
| CNN | `cnn_gn`, `resnet20`, `resnet50` | 1e-3 | 1e-3 | 5e-4, coupled (1e-4 on ImageNet `resnet50`) | 1e-3 |
| Transformer | `vit_micro`, `vit_small`, `cct_2_3x2` | 1e-3 | 1e-4 | 0.01, decoupled for the Fisher modes | 3e-3 |

`resnet50_*` additionally runs with `conv_sua=True` and `fisher_batch_samples=32`. `vit_small_imagenet`
runs with `fisher_batch_samples=32`. The batch size is 128 on CIFAR and 256 on ImageNet.

### 1.3 The jobs

| dataset | benchmark | parameters | reference epochs | job | elapsed | limit |
|---|---|---|---|---|---|---|
| CIFAR-100 | `cnn_gn_cifar100` | 30 308 | 30 | 21447340 | 0:07:14 | 0:20 |
| | `vit_micro_cifar100` | 24 068 | 30 | 21447341 | 0:09:11 | 0:25 |
| | `resnet20_cifar100` | 275 572 | 50 | 21447342 | 0:44:53 | 1:05 |
| | `cct_2_3x2_cifar100` | 295 333 | 50 | 21447343 | 0:30:27 | 0:45 |
| | `resnet50_cifar100` | 23 705 252 | 50 | 21447345 | 5:43:21 | 7:00 |
| | `vit_small_cifar100` | 2 710 948 | 50 | 21447344 | 2:01:07 | 2:45 |
| CIFAR-10 (re-run) | `resnet50_cifar` | 23 520 842 | 50 | 21447469 | 5:43:04 | 7:00 |
| | `vit_small_cifar` | 2 693 578 | 50 | 21447468 | 2:02:00 | 2:45 |
| ImageNet32 | `cnn_gn_imagenet` | 88 808 | 40 | 21450214 | 2:31:49 | 16:00 |
| | `vit_micro_imagenet` | 53 768 | 40 | 21450215 | 3:03:58 | 16:00 |
| | `resnet20_imagenet` | 334 072 | 40 | 21450216 | 14:00:24 | 23:00 |
| | `cct_2_3x2_imagenet` | 411 433 | 40 | 21450217 | 9:16:57 | 23:00 |
| ImageNet-1K, 224 px | `resnet50_imagenet` | 25 557 032 | 30 | 21450222 (`diag`), 21468627-32 | 6:24:54, then 6:23:40 to 6:24:50 per arm | 12:00 each |
| | `vit_small_imagenet` | 22 050 664 | 30 | 21450223 (`diag`), 21496658-63 | 9:29:17, then 9:28:30 to 9:29:26 per arm | 12:00 each |

Each grouped job runs all seven arms one after the other. The two 224 px models run one arm per job:
the `diag` job measures the budget, then `dispatch_arms` submits the six others with that budget
(21450224 for ResNet-50 and 21496598 for ViT-S). The six ViT-S arms ran overnight, 20-21 September.

**The four "ImageNet32" benchmarks are not ImageNet-1K.** They train on the full 1.28 M ImageNet
images squashed to 32x32 (Chrabaszcz et al. 2017, arXiv:1707.08819 §2), so that the four
32x32-native architectures can run unchanged. Published ImageNet numbers are not a reference for
them.

---

## 2. Is the data sound?

Checked mechanically, on every recorded step of every arm (8 416 589 steps):

- **Zero non-finite losses.**
- **Zero backwards steps of the clock**: within each arm the elapsed time only increases.
- **Every budget is respected.** The worst overshoot on each benchmark is between +0.00 s and
  +0.35 s (`tekfac` on both re-run CIFAR-10 models). Each is smaller than that arm's own longest
  single step, which is the bound the training loop guarantees.
- **The data pipeline is not the bottleneck on ImageNet.** The "compute share" column (the fraction
  of an arm's time spent in its forward, backward and optimizer step) is 92.5% to 99.0% on all 35
  ImageNet32 arms and 97.3% to 98.5% on the fourteen arms at 224 px. The job headers had estimated the
  ImageNet32 jobs as "data-loader bound" at about 2 h per arm. They measured 21 and 26 min per arm
  for the two smallest networks (the job's elapsed time divided by 7).

---

## 3. The Fisher modes against the baselines

Best validation accuracy, in %, one seed. "Fisher best" is the best of the five modes, and "baseline
best" the better of `adam` and `adamw`. The last column is the same comparison on the test set.

| benchmark | `diag` | `kfac` | `ekfac` | `tkfac` | `tekfac` | `adam` | `adamw` | Fisher best − baseline best | same, on test |
|---|---|---|---|---|---|---|---|---|---|
| `cnn_gn_cifar100` | 28.86 | **29.72** | 28.92 | 29.16 | 27.80 | 27.16 | 27.84 | +1.88 | +2.28 |
| `vit_micro_cifar100` | **18.76** | 18.66 | 16.58 | 17.64 | 16.56 | 5.78 | 9.60 | **+9.16** | +8.98 |
| `resnet20_cifar100` | 65.28 | 64.92 | 64.32 | **65.44** | 63.94 | 60.88 | 57.20 | **+4.56** | +4.81 |
| `cct_2_3x2_cifar100` | **55.82** | 54.58 | 53.92 | 53.50 | 53.30 | 24.28 | 34.80 | **+21.02** | +21.36 |
| `resnet50_cifar100` | 76.92 | **77.72** | 76.70 | 77.50 | 77.12 | 71.18 | 73.10 | **+4.62** | +3.99 |
| `vit_small_cifar100` | 40.80 | 38.62 | 36.42 | 38.18 | 36.38 | 18.76 | **45.48** | **−4.68** | −4.80 |
| `resnet50_cifar` | 93.94 | 94.26 | 94.12 | **94.34** | 93.84 | 92.56 | 93.66 | +0.68 | +1.29 |
| `vit_small_cifar` | 67.68 | 67.22 | 65.82 | 66.72 | 65.80 | 45.66 | **71.78** | **−4.10** | −4.23 |
| `cnn_gn_imagenet` (32 px) | 6.23 | 6.46 | 6.78 | 6.66 | 6.57 | 8.34 | **12.47** | **−5.70** | −4.60 |
| `vit_micro_imagenet` (32 px) | 10.09 | **10.39** | 9.84 | 10.09 | 9.72 | 0.82 | 8.56 | +1.82 | +1.55 |
| `resnet20_imagenet` (32 px) | 26.24 | 26.51 | 26.17 | 26.48 | 26.27 | 26.73 | **28.04** | −1.54 | −1.35 |
| `cct_2_3x2_imagenet` (32 px) | 26.83 | 26.53 | 26.64 | **27.08** | 26.81 | 3.63 | 23.58 | **+3.51** | +2.99 |
| `resnet50_imagenet` (224 px) | **77.56** | 77.28 | 76.87 | 77.14 | 76.96 | 68.22 | 74.29 | **+3.28** | +3.52 |
| `vit_small_imagenet` (224 px) | 52.98 | 52.70 | 52.01 | 52.76 | 52.07 | 6.27 | **59.74** | **−6.77** | −6.57 |

In bold, in the accuracy columns: the best of the seven arms. In bold, in the margin column: margins
larger than 2.4 points, the largest seed-to-seed spread this project has measured (`vit_micro_cifar`,
E13 in `plan_lambda_dominance.md`). Only those margins are read as findings at one seed.

**Where the Fisher modes win clearly.** Six benchmarks: CCT on CIFAR-100 (+21.0) and on
ImageNet32 (+3.5), the micro ViT on CIFAR-100 (+9.2), ResNet-20 on CIFAR-100 (+4.6), ResNet-50 on
CIFAR-100 (+4.6) and on ImageNet at 224 px (+3.3). On all six, even the *worst* of the five modes
beats the better baseline, by +2.6 to +18.5 points.

**Where the result is inside the noise.** Four benchmarks have margins of −1.5 to +1.9:
`cnn_gn_cifar100`, `resnet50_cifar`, `vit_micro_imagenet` and `resnet20_imagenet`. One seed cannot
say who wins there.

**Where the Fisher modes lose clearly.** Four benchmarks.

- **ViT-S on CIFAR-10, CIFAR-100 and ImageNet at 224 px**, every time against `adamw`, by 4.1, 4.7
  and 6.8 points. This is the one architecture where the loss repeats across datasets, and it grows
  with the dataset: on ImageNet `adamw` is ahead by 6.6 points on the official validation set too
  (56.23% against 49.66%). It is also a new pattern: on
  the six classification models of the seed table, the Fisher modes beat the better baseline in 34
  of 35 (mode, seed) cells (`CLAUDE.md`, "The seed axis completed"). The micro ViT, which shares
  ViT-S's code and its hyperparameters, wins by +9.2 on CIFAR-100. What differs between the two is
  size: 2.7 M parameters against 24 k, width 192 against 32, 6 blocks against 2, 3 heads against 2,
  and an MLP 4 times the width against 2 times. Which of these reverses the result has not been
  measured.
- **The GroupNorm CNN on ImageNet32**, by 5.7 points. At that accuracy (6% to 12% over 1 000
  classes) all seven arms are barely learning. One measured fact is worth recording: the
  1000-class output layer holds 65 000 of the network's 88 808 parameters, i.e. **73%**, against
  3% on CIFAR-10. This is now mostly a linear classifier on a small feature extractor.

**The CIFAR-10 to CIFAR-100 comparison, where it is fair.** Only ResNet-50 and ViT-S have a CIFAR-10
run under the same protocol (the re-runs above). ResNet-50's Fisher margin grows from +0.7 on
CIFAR-10 to +4.6 on CIFAR-100. ViT-S's stays at about −4 on both. The four smaller models' CIFAR-10
numbers come from the seed table, which used a different learning-rate schedule (`nominal`). In
that table the margin on CIFAR-10 is +12.6 to +14.4 for the micro ViT, +10.1 to +11.1 for CCT,
+1.9 to +3.0 for `cnn_gn` and +1.2 to +2.3 for ResNet-20. On CIFAR-100 all four keep the same sign.

---

## 4. Which baseline to compare against

Where the weight decay is 0.01, `adam` and `adamw` are not two equally good baselines.

| benchmark | weight decay | `adam` | `adamw` | `adam − adamw` |
|---|---|---|---|---|
| `vit_small_cifar` | 0.01 | 45.66 | 71.78 | **−26.12** |
| `vit_small_imagenet` | 0.01 | 6.27 | 59.74 | **−53.47** |
| `vit_small_cifar100` | 0.01 | 18.76 | 45.48 | **−26.72** |
| `cct_2_3x2_imagenet` | 0.01 | 3.63 | 23.58 | **−19.94** |
| `cct_2_3x2_cifar100` | 0.01 | 24.28 | 34.80 | **−10.52** |
| `vit_micro_imagenet` | 0.01 | 0.82 | 8.56 | **−7.75** |
| `vit_micro_cifar100` | 0.01 | 5.78 | 9.60 | −3.82 |
| `resnet50_imagenet` | 1e-4 | 68.22 | 74.29 | −6.07 |
| `cnn_gn_imagenet` | 5e-4 | 8.34 | 12.47 | −4.13 |
| the five other CNN benchmarks | 5e-4 | | | −1.92 to +3.68 |

`adam` here is `torch.optim.Adam(weight_decay=...)`, which adds the weight-decay term to the
gradient before Adam divides by its running scale. `adamw` applies it to the weights directly
(`benchmarks/common/optimizers.py:138-143`). The gap is large on every benchmark with the larger
weight decay. That is consistent with this difference being the cause, but no run has changed the
weight decay alone, so it is not established. It also means the ViT and CCT rows of section 3 must
be read against `adamw`. That is what "baseline best" does: `adamw` is the better of the two on 13
benchmarks of 14, the exception being `resnet20_cifar100` (`adam` ahead by 3.7).

---

## 5. The five modes split into two groups again

The seed table found, over 24 paired cells on CIFAR-10 and MNIST, that `{diag, kfac, tkfac}` beat
`{ekfac, tekfac}`, and that the two losing modes are the two that fit the fewest steps into the same
time. The 14 benchmarks here are a new sample: different datasets, one seed each, with all arms of
one benchmark sharing their initialisation and data order.

**The ranking replicates.** Mean rank among the five modes, by best validation accuracy (1 is best):

| mode | mean rank here (14 benchmarks) | mean rank in the seed table (24 cells) | in the bottom two here |
|---|---|---|---|
| `kfac` | **2.21** | 2.08 | 2 / 14 |
| `tkfac` | **2.29** | 2.42 | 1 / 14 |
| `diag` | **2.36** | 2.42 | 5 / 14 |
| `ekfac` | 3.86 | 4.17 | 10 / 14 |
| `tekfac` | 4.29 | 3.92 | 10 / 14 |

Of the 84 possible comparisons between a mode of the first group and a mode of the second,
**72 go to the first group** (86%). Head to head, with an exact two-sided sign test over the 14
benchmarks: `tkfac > tekfac` **14/14** (p = 1.2e-4); `kfac > ekfac`, `kfac > tekfac` and
`tkfac > ekfac` 12/14 each (p = 0.013); `diag > ekfac` and `diag > tekfac` 11/14 (p = 0.057). Inside
the first group no pair separates (`diag` vs `kfac` 8/14, `kfac` vs `tkfac` 9/14). Inside the
second neither (`ekfac` vs `tekfac` 9/14).

**The mechanism is the same one.** Steps completed inside the same time budget, relative to `diag`,
averaged over the 14 benchmarks:

| `diag` | `kfac` | `tkfac` | `ekfac` | `tekfac` | `adam` | `adamw` |
|---|---|---|---|---|---|---|
| 1.00 | 0.96 | 0.95 | **0.88** | **0.88** | 1.11 | 1.11 |

The largest deficits are on the two ResNet-50 CIFAR benchmarks, where `ekfac` and `tekfac` complete
0.76 of `diag`'s steps: 38 epochs against 45 for `kfac`/`tkfac` and 50 for `diag`. This is the cost
of projecting every step into and out of an eigenbasis (`plan_lot7.md` §6).

**Why the correlation between steps and accuracy now means something.** Across the four budgeted
Fisher arms of a benchmark, the number of epochs completed correlates with best validation accuracy
at +0.56 to +0.99 on 12 of the 14 benchmarks. In campaign 1 a similar correlation (+0.92 to +0.98,
over all five Fisher arms) could not be read, because an expensive arm also stopped earlier on its cosine and so trained at a
higher final learning rate. Here every arm finishes its own cosine, so that confound is gone.
What remains is the number of steps. Two caveats: with four or five points per benchmark, this
correlation mostly restates the two-group split, and it is not an equal-step comparison. It does
not show that `ekfac`/`tekfac` approximate the curvature worse.

---

## 6. The schedule fix, measured on the model it was meant for

`resnet50_cifar` and `vit_small_cifar` were first run before two fixes: the checkpoint denominator
and the `budget` learning-rate schedule. Both were re-run here as one grouped job each. The
reference arm `diag` is not budgeted, so it is reproducible, and it reproduces: **93.94%** and
**67.68%**, identical to the first run.

| model | arm | before the fix | after | change | epochs completed |
|---|---|---|---|---|---|
| `resnet50_cifar` | `ekfac` | 90.04 | 94.12 | **+4.08** | 38 → 39 |
| | `tekfac` | 90.76 | 93.84 | **+3.08** | 39 → 38 |
| | `kfac` | 93.60 | 94.26 | +0.66 | 45 → 45 |
| | `tkfac` | 93.92 | 94.34 | +0.42 | 45 → 45 |
| | `adamw` | 93.44 | 93.66 | +0.22 | 52 → 52 |
| `vit_small_cifar` | `ekfac` | 67.16 | 65.82 | −1.34 | 47 → 46 |
| | `tekfac` | 67.02 | 65.80 | −1.22 | 46 → 46 |
| | `kfac` | 67.46 | 67.22 | −0.24 | 50 → 49 |
| | `tkfac` | 67.28 | 66.72 | −0.56 | 49 → 49 |
| | `adamw` | 71.38 | 71.78 | +0.40 | 54 → 53 |

On ResNet-50 the fix moves exactly the arms it was predicted to move. Before the fix `ekfac` and
`tekfac` stopped at epochs 38 and 39 of 50, `ekfac` with its learning rate still at `1.6e-4`. Now they complete their
cosine, and gain 3 to 4 points. `ekfac` goes from last of the seven arms to third, and the spread
between the five modes shrinks from **3.88 points to 0.50**. So campaign 1's ranking of the modes
on this model was mostly a ranking of how far down the schedule each one got. That is what the
campaign-1 audit concluded, and it is now measured.

On ViT-S nothing moves beyond noise. Before the fix every Fisher arm already reached epoch 46 to 50
of 50, i.e. near the bottom of the cosine, so there was little to recover. The changes of −1.3 to
+0.4 points are what two runs of a budgeted arm differ by. They are not reproducible run to run,
because their step count depends on the node's load (`CLAUDE.md`, "only the reference arm is
reproducible across campaigns").

---

## 7. ResNet-50 on ImageNet-1K, next to the AdaFisher paper

The AdaFisher paper's Table 3 reports ResNet-50 on ImageNet-1K at batch 256, one trial (§D.2), under
a wall-clock budget equal to 90 epochs of Adam. Its top-1 accuracy on the ILSVRC validation set is
76.95% for AdaFisher and 67.78% for Adam, a gap of **+9.17** points.

Ours, on the same official validation set (our "test" column), after a budget equal to 30 epochs of
`diag`:

| arm | top-1, ILSVRC val | epochs completed |
|---|---|---|
| `kfac` | **74.45** | 30 |
| `diag` | 74.24 | 30 |
| `tkfac` | 74.24 | 30 |
| `ekfac` | 73.91 | 27 |
| `tekfac` | 73.64 | 27 |
| `adamw` | 70.93 | 32 |
| `adam` | 64.99 | 32 |

The `diag` − `adam` gap is **+9.25** points, against the paper's +9.17. Our `diag` sits 2.7 points
below the paper's AdaFisher, with a third of its training time. These are **not the same
experiment**: the budget is a third as long, and the reference arm is `diag` rather than Adam. Our
`adam` also carries the coupled weight decay of section 4. So this is context, not a reproduction.
Still, the size of AdaFisher's advantage over Adam on this benchmark comes back.

Two further points. First, the five modes are within 0.81 points of each other on this benchmark,
and the four Kronecker modes run here with `conv_sua=True`. So this is a result about SUA-K-FAC and
its variants, not about full-patch K-FAC (`CLAUDE.md`, "`conv_sua=True` is not optional on
ResNet-50"). Second, validation accuracy on the held-out training images is 2.8 to 3.4 points
higher than on the official set for every arm, e.g. 77.56% against 74.24% for `diag`. Rank the arms
on the official set.

---

## 8. What this campaign does not establish

- **One seed.** The seed-to-seed spread measured in this project runs from 0.04 points
  (`cnn_gn_cifar`, `resnet20_cifar`, at batch 128) to 2.4 points (`vit_micro_cifar`, E13). No
  per-benchmark difference under about 2.4 points in section 3 is a finding. The two-group result
  of section 5 does not depend on this, because it pools 14 benchmarks.
- **The default `λ`.** At `λ = 1e-3` or `3e-3`, the preconditioner the optimizer actually divides by
  is within about 10% of a multiple of the identity (`plan_exp_lot5.md` §6, on four regime-A
  networks). And the E-series of `plan_lambda_dominance.md` gains 5 to 10 points on
  `cct_2_3x2_cifar` and `vit_micro_cifar` by lowering `λ` 7 to 8 orders of magnitude, under its
  own protocol (batch 32, 15 epochs, test accuracy). So these runs
  compare the five optimizers *as shipped*. They do not measure what curvature buys at a working
  `λ`. That comparison needs fix S1 (a `λ` relative to each layer's curvature), which is not
  implemented yet.
- **Untuned baselines.** `adam`/`adamw` run at one learning rate per family (1e-3 on CNNs, 1e-4 on
  transformers), fixed in each benchmark's `HParams`. They were not swept here.
- **ImageNet32 is not ImageNet-1K** (section 1.3).

---

## 9. For the next job generation

The ImageNet `--time` values in `benchmarks/slurm/generate_jobs.py` were estimates. They are now
measured, all with a wide margin: 2:32 and 3:04 for the two small ImageNet32 grouped jobs, against
16:00 requested; 9:17 and 14:00 for CCT and ResNet-20, against 23:00; 6:25 per arm for ResNet-50
against 12:00; and 9:29 for ViT-S's reference arm against 12:00, which is the tightest. The CIFAR-100
limits, carried over from CIFAR-10, were safe: every job used 36% to 82% of its limit. None of
these values has been changed yet.
