# Running the authors' own code: the stalling bench, and two published CIFAR-10 numbers

**Why this exists.** This repository has a negative result about AdaFisher: on its primary bench, the
8-layer MNIST auto-encoder, all five Fisher modes freeze at a mean-squared error of about 0.708 while
Adam descends to 0.48 (`CLAUDE.md`'s status block, "the project's **primary** bench is the only
failure"). Every number behind that statement was produced by this repository's own port. That is not
enough. A port can be wrong in a way that no internal test catches, and until the authors' own code
has been run — on the same bench, and on a setting whose answer is published — a negative conclusion
is not defensible.

So this lot does two things, in the only order that makes sense:

1. **Run the authors' own optimizer, unmodified, on the bench that stalls.** If their code stalls too,
   the stall is a property of AdaFisher and not of this port. **It does.**
2. **Reproduce one published number with the authors' own training script.** If a line of their
   Table 2 comes back, then their code, their data pipeline and this cluster together are known to
   work — which is what licenses reading step 1 as a statement about the method. **Two lines came
   back.**

## Short version

Step 1 is **done, and it answers the question.** The optimizer of the authors' published repository,
loaded from `reference_repos/AdaFisher/optimizers/AdaFisher.py` and not modified in any way, freezes
on the MNIST auto-encoder at exactly the same place this project's `diag` mode does. Over a full
20-epoch run of 2 200 steps the two disagree by at most **2.6e-7 in relative terms** — the few
floating-point bits the fused `addcdiv_` costs (`CLAUDE.md` §2.1), and nothing else. Both end at a
train loss of **0.7083**; Adam, same net, same data, same learning rate, ends at **0.4712**. The other
implementation by the same authors, FisherAdapTune's, stalls at the same 0.7083 as well.

**The stall is therefore not a defect of this port.** It is what AdaFisher, as published, does on that
network at that operating point.

Two things sharpen it, both measured here rather than assumed:

* **The plateau is the network's, not the optimizer's.** Adam lands on the very same 0.7083 as soon as
  its learning rate is raised to `1e-2`. So 0.7083 is where this all-sigmoid auto-encoder sits when an
  optimizer fails to get off the initial plateau, whichever optimizer it is.
* **AdaFisher sits there at every learning rate tried**: `1e-4`, `1e-3`, `1e-2` and `1e-1` all end
  within 7e-6 of 0.70828, while Adam's ending loss moves across that range (0.6090, 0.4719, 0.7083,
  0.7085 — it escapes at the two small rates and not at the two large ones). The AdaFisher
  trajectories *do* differ — the authors' code at `1e-4` and at `1e-3` are up to 0.28 apart mid-run —
  they simply arrive at the same place. A learning-rate bracket is thus ruled out as the explanation,
  the way the damping bracket already was (`CLAUDE.md`'s `lam` paragraph).

Step 2 is **done, and it lands.** Running the authors' `train.py` on their own
`AdaFisherCNN.yaml`, verbatim, one seed, gives **best test top-1 = 96.32 %** against the published
**96.25 ± 0.2**; their `adamCNN.yaml` gives **94.78 %** against the published **94.85 ± 0.1**. Both
are inside their published band, at 0.35 and 0.70 of the respective standard deviation, and the
published gap between the two optimizers (1.40 points) comes back as **1.54**. The positive control
therefore exists: the authors' code, their data pipeline and this cluster together reproduce a
published number, which is what licenses reading step 1 as a statement about the method rather than
about the setup.

## 0. Decisions, and the reason for each

### 0.1 Which published line, and why that one

`docs/papers/adafisher_2405.16397.pdf`, **Table 2, CIFAR-10, ResNet18, AdaFisher: 96.25 ± 0.2**
(batch size 256; 200-epoch cutoff, confirmed by Table 8; learning rate `1e-3`, confirmed by Table 9).

It is the right line for one reason above all others: **the authors ship exactly that setting as a
config file.** `reference_repos/AdaFisher/Image_Classification/configs/AdaFisherCNN.yaml` is
`resnet18Cifar` on `CIFAR10` with `AdaFisher`, `init_lr: 0.001`, `weight_decay: 5e-4`, `gamma: 0.8`,
`Lambda: 1e-3`, `CosineAnnealingLR` with `T_max: 200`, `max_epochs: 200`, `mini_batch_size: 256`,
augmentation and cutout with one 16-pixel hole. Nothing has to be inferred from the paper and nothing
has to be overridden, so there is no room for a transcription mistake to be blamed afterwards.

The **Adam row of the same table, 94.85 ± 0.1**, is run as well, from their `adamCNN.yaml` (210
epochs, per Table 8). It costs roughly the same and it is the control that matters: a single number
near 96 proves little on its own, whereas the published *gap* of about 1.4 points between the two
optimizers is a claim the reproduction either recovers or does not.

### 0.2 The authors have two implementations, and both are needed

* `reference_repos/FisherAdapTune/scripts/adafisher.py` — authoritative for this project
  (`CLAUDE.md`), already available as the `reference` arm, and **bit-exactly** matched by
  `diag --no-minmax` (`tests/test_diag_bitexact.py`). It does **not** apply Eq. (4)'s min-max
  normalisation.
* `reference_repos/AdaFisher/optimizers/AdaFisher.py` — the published repository, the code that
  produced Table 2, and it **does** apply the min-max normalisation.

`diag`'s shipped default is min-max **on**, so its like-for-like partner is the published repository,
not FisherAdapTune. That partner had never been run. This lot adds it as the **`official` arm** of
`benchmarks/common/optimizers.py`, loaded by file path exactly as `reference` is, so that neither
read-only repository's package `__init__` is ever imported.

On the exponential moving average the two references agree: the published code takes one scalar
`gamma` and forms the coefficients `(gamma·1e-1, gamma·1e-2)`, so its published `gamma = 0.8` is
`(0.08, 0.008)`, which is this project's default pair `gammas = (0.92, 0.008)` — the same numbers,
written differently (`docs/reports/archives/plan.md` §1.4). `official_gamma` performs that
translation, and **refuses** a pair no single scalar can express rather than rounding one silently: an
arm that exists to run the authors' code at their operating point must not quietly run at a different
one.

### 0.3 The one deviation from the authors' code: an `asdl` import stub

`Image_Classification/src/train.py` line 33 imports `asdl` unconditionally. The three names it imports
are used **only** by its Shampoo and K-FAC arms (lines 229-244 and 406-409). `asdl` is a git
dependency in the authors' `requirements.txt`, it is absent from the Alliance Canada wheelhouse, and
compute nodes have no internet.

`benchmarks/authors_repro/asdl_stub/` therefore provides the three names, each of which raises on
first use. This is the **only** deviation in the whole of `benchmarks/authors_repro/`. It cannot
affect an AdaFisher or Adam run, because no line of either path reaches those names, and
`run_authors_train.sh` refuses outright to launch a config whose `optimizer` is `Shampoo` or `kfac`,
so the stub cannot be reached by accident.

### 0.4 The authors' own pinned environment, and one pin that is load-bearing

Their `requirements.txt` pins `torch 2.3.0`, `torchvision 0.18.0`, `numpy 1.26.4`. **All three are in
the cluster's wheelhouse**, verified by building the environment on the login node before submitting
anything, so the reproduction runs on the authors' own versions rather than on today's.

`numpy 1.26.4` is not cosmetic. `Image_Classification/src/utils/early_stop.py` uses `np.Inf`, which
numpy 2 removed; the wheelhouse default is numpy 2.4.2, so an unpinned install crashes in the
authors' own early-stopping object. `einops` and `timm` are left unpinned on purpose: the authors'
`models/__init__.py` imports every architecture, two of which need them, and ResNet18 needs neither —
so pinning them to their old versions would risk a dependency conflict for no gain.

### 0.5 What counts as reproduced, fixed in advance

**Best test top-1 accuracy over the run**, which is what the authors' own `train.py` tracks as
`best_acc1` and what Table 9 calls "final validation top-1 accuracy". Their `validate()` evaluates on
the CIFAR-10 test set after every epoch.

The target is the published mean ± its published standard deviation: **96.25 ± 0.2** for AdaFisher and
**94.85 ± 0.1** for Adam. One seed is run (42, the authors' default), so the number produced is a
single draw from the distribution the published figures summarise, not a mean over trials. Being
inside the published band is the check; being outside it by less than a point would still leave the
published *ordering* recoverable and will be reported as such rather than dressed up either way.

### 0.6 Two guard-rails that a failed first submission put in

The first attempt (job 21449006) failed in two seconds. `reference_repos/` is gitignored, and the
cluster checkout held the **directory skeleton with zero files in it** — the authors' code had never
been synced there at all. The script's integrity check, a `diff -r` between the read-only tree and the
copy it makes in `$SLURM_TMPDIR`, passed happily: `diff -r` on two empty trees is clean.

Both holes are now closed. The script counts the files it copied and refuses to continue below a
floor, and it names the missing config file (and lists what is there) instead of dying on a bare
`test -f`. Separately, the sync was verified by comparing SHA-256 sums of all 73 files of the authors'
repository on both machines: identical, both sides.

## 1. What was built

| Path | What it is |
|---|---|
| `benchmarks/common/optimizers.py` | the new `official` arm, `load_official_adafisher`, `official_gamma`; `ARMS` grows from 8 to 9 |
| `tests/test_official_adafisher_arm.py` | 10 offline tests: the arm is the authors' class from the authors' file, `AdaFisherW` under `decoupled_wd`, the `gamma` translation and its two refusals, every parameter moves, and the measured agreement with `diag` on a Linear-only net and on all four hooked layer types |
| `benchmarks/authors_repro/asdl_stub/` | the import stub of §0.3 |
| `benchmarks/authors_repro/slurm/run_authors_train.sh` | copies the authors' tree to `$SLURM_TMPDIR`, checks the copy, stages CIFAR-10, builds their pinned environment, runs their `train.py` verbatim, then reads best/final accuracy off their own log files |
| `benchmarks/authors_repro/slurm/calibrate_authors_resnet18_cifar10.sh` | 5 epochs, to measure seconds per epoch before asking for a 200-epoch walltime |
| `benchmarks/authors_repro/slurm/train_authors_resnet18_cifar10_{adafisher,adam}.sh` | the two full runs, their configs verbatim |

Nothing under `src/adafisher_modes/`, `fisher_ref/` or `reference_repos/` was touched.

## 2. Step 1: the authors' code on the bench that stalls

The authors have no auto-encoder script, so their *optimizer* is run inside this repository's own
training loop — the same loop, the same data, the same seed and the same learning rate for every arm.
That is the controlled comparison; and the loop itself is not in question, since Adam trains fine in
it and this project's `diag` mode matches FisherAdapTune bit for bit inside it.

**Four arms, 20 epochs each, equal epochs (not the wall-clock protocol), batch 500, `lr = 1e-3`, CPU**
(`benchmarks/outputs/authors_repro/mnist_autoencoder_e20/`):

| arm | what it is | steps | final train loss |
|---|---|---|---|
| `diag` | this project's port, min-max on (its default) | 2 200 | **0.7083** |
| `official` | the authors' published optimizer, unmodified | 2 200 | **0.7083** |
| `reference` | FisherAdapTune's `AdaFisher`, unmodified | 2 200 | **0.7083** |
| `adam` | `torch.optim.Adam` | 2 200 | **0.4712** |

Agreement between implementations, over all 2 200 steps of the run:

| pair | largest absolute loss gap | largest relative gap |
|---|---|---|
| `diag` vs `official` | 1.79e-7 | **2.58e-7** |
| `diag` vs `reference` | 1.05e-3 | 1.31e-3 |
| `official` vs `reference` | 1.05e-3 | 1.31e-3 |

The first row is the point of the exercise: this project's `diag` mode and the authors' published
optimizer produce the same trajectory to seven digits. The second and third rows are the min-max
normalisation being on in one implementation and off in the other — worth a thousandth of the loss
here, and not worth the difference between stalling and not.

### 2.1 The learning-rate bracket

Same bench, same 20 epochs, `official` against `adam`, four learning rates
(`benchmarks/outputs/authors_repro/mnist_autoencoder_lr/`):

| learning rate | `official` final train loss | `adam` final train loss |
|---|---|---|
| `1e-4` | 0.708274 | 0.609006 |
| `1e-3` | 0.708280 | **0.471945** |
| `1e-2` | 0.708281 | 0.708275 |
| `1e-1` | 0.708276 | 0.708504 |

Six digits, on purpose: the authors' code ends within **7e-6** of the same loss at all four learning
rates, which is a tighter agreement than between two learning rates of Adam anywhere in the table.

Two readings, and they pull in different directions, so both are stated.

**The learning rate is not the lever.** The authors' code ends at 0.7083 across four decades. The
flag does reach it — the `1e-4` and `1e-3` trajectories differ by up to 0.28 in the middle of the run
— so this is four different journeys to one destination, not one run repeated.

**But 0.7083 is the network's plateau, not a number AdaFisher invents.** Adam lands on it too at
`1e-2`. This auto-encoder is eight `Linear` layers with sigmoids throughout and a 30-unit bottleneck,
chosen precisely because it punishes a badly scaled step; 0.7083 is where an optimizer stays when it
cannot leave the initial plateau. What distinguishes the two optimizers here is that Adam has a
window of learning rates that escapes and AdaFisher, at this damping, has none among those tried.

Which side of the plateau AdaFisher is stuck on is already known and is not re-measured here:
`CLAUDE.md` records parameters travelling about **1.0** from initialisation for the Fisher arms
against **25.8** for Adam, i.e. steps that are far too small, which is the expected consequence of
`lambda` sitting above every curvature direction (`docs/reports/plan_lambda_dominance.md`).

## 3. Step 2: the published CIFAR-10 line — protocol

One calibration job and two full runs, all three invoking the authors' `train.py` with one of their
own config files and no override:

| job | config | epochs | published target | job id | state |
|---|---|---|---|---|---|
| `authors_r18_c10_calib` | `AdaFisherCNN.yaml`, `max_epochs` replaced by 5 | 5 | none; it measures seconds per epoch | 21449705 | COMPLETED, 00:02:18 |
| `authors_r18_c10_adafisher` | `AdaFisherCNN.yaml`, verbatim | 200 | 96.25 ± 0.2 | 21449869 | COMPLETED, 01:04:52 — **96.32 %** |
| `authors_r18_c10_adam` | `adamCNN.yaml`, verbatim | 210 | 94.85 ± 0.1 | 21449870 | COMPLETED, 01:04:33 — **94.78 %** |

The calibration job exists because this repository budgets walltime from a measurement and not from
an extrapolation (`CLAUDE.md`, the step-1 follow-up). It also exercises the whole path — wheelhouse
environment, staged CIFAR-10, the `asdl` stub, the GPU — for the price of five epochs.

**It did both.** Measured on a `h100_1g.10gb` slice: **19.2 s per epoch** (median of five; the first
is 21.5 s with warm-up in it) and **49 s** for the environment build and the data staging together.
So 200 epochs is about 64 minutes of training, which is what the two full jobs ask for with 30% of
margin — `01:30:00` and `01:45:00`, measured rather than extrapolated. And the five epochs train the
way a healthy ResNet18 trains: test top-1 **38.72 → 52.10 → 58.00 → 63.26 → 67.96 %**, train loss
1.899 → 0.944. That is not a result — it is five epochs of a 200-epoch cosine schedule — but it rules
out a broken pipeline before an hour of GPU time is spent on it.

Data: CIFAR-10 is staged in `dataset/`, copied to node-local storage, and `download=True` finds it
already verified. Results go to `benchmarks/outputs/authors_repro/`, which is gitignored like every
other output tree, and which `fisher_ref`'s run discovery ignores because `authors_repro` is not a
dataset group any bench declares.

## 4. What this establishes, and what it does not

**Established.** The MNIST auto-encoder stall is reproduced by the authors' own published optimizer,
run unmodified, and this project's `diag` mode agrees with it to 2.6e-7. Any explanation of the stall
that appeals to a porting mistake is now excluded.

**Established.** That the authors' code reaches its published accuracy in this environment: two rows
of Table 2, both inside their published standard deviation, from their config files with no override
(§6.2). So the negative result on the auto-encoder is not a broken setup talking: the same code, the
same machine and the same person's hands produce a published number on CIFAR-10 and a stall on the
auto-encoder.

**Still not established.** That AdaFisher stalls on *anything else*. The two results together say
that AdaFisher works as published on ResNet18/CIFAR-10 and fails on this auto-encoder — which is a
statement about where the method works, not a verdict on it. What sits between the two settings is
untested here: the CIFAR-10 net is convolutional with batch normalisation, trained with augmentation
at batch 256 under a 200-epoch cosine; the auto-encoder is eight sigmoid `Linear` layers at batch
500 for 20 epochs. Any of those could be the discriminating factor.

**Out of scope.** The four Kronecker modes are this project's own, have no counterpart in the authors'
code, and are untouched by either step. The auto-encoder comparison uses one seed and this
repository's training loop; a plateau shared by three implementations and reached from four learning
rates is not fragile, but it is one bench. The CIFAR-10 runs use one seed each, so they land a single
draw inside a published band rather than reproducing the band itself.

## 5. Findings during implementation

1. **The cluster had none of the authors' code.** `reference_repos/` is gitignored, and the remote
   checkout held 0 of its 73 files inside an otherwise correct directory skeleton — so every
   integrity check that compared the remote tree with itself passed. Fixed by syncing and verifying
   SHA-256 sums on both sides, and by making the job count the files it copies (§0.6).
2. **`numpy 2` breaks the authors' code before it trains anything**, via `np.Inf` in their
   `early_stop.py`. Their own pin, `1.26.4`, is in the wheelhouse; so are `torch 2.3.0` and
   `torchvision 0.18.0`, so the reproduction runs on their versions and not on today's.
3. **The published code trains under a gradient scaler by default.** `train.py` sets
   `autocast = True` whenever CUDA is available, and `config['precision']: fp32` then makes
   `torch.autocast(device_type='cuda', dtype=torch.float32)` disable itself with a warning while
   `GradScaler(enabled=True)` stays on. The arithmetic is fp32 with the loss scaled up and the
   gradients unscaled before the step. It is worth knowing that the Fisher factors are collected
   inside that scaled backward pass, and worth not worrying about: min-max normalisation makes the
   `diag` preconditioner invariant to a constant factor on those statistics. Their code, run as they
   run it.
4. **Same behaviour, three implementations, and the min-max ordering barely matters here.** The two
   published implementations differ exactly by the min-max normalisation, and on this bench that is
   worth 1.3e-3 of relative loss. On a bench where Eq. (4)'s normalisation does matter — it puts
   `F~_D` on `[lambda, 1+lambda]`, which is the whole subject of `plan_lambda_dominance.md` — the two
   would not sit this close, so this is a fact about the auto-encoder and not a general one.

## 6. Results

### 6.1 The learning-rate bracket: complete

Done, four points, both arms, 20 epochs each; the table in §2.1 is the result. The last point behaves
like the others: at `lr = 1e-1` the authors' code ends at 0.708276 and Adam at 0.708504, both on the
plateau.

So Adam escapes the plateau at `1e-4` and `1e-3` and fails to at `1e-2` and `1e-1`; the authors' code
fails to at all four. The plateau itself is shared, which is why it is described as the network's.

### 6.2 The CIFAR-10 runs: both published lines come back inside their published band

Jobs **21449869** (AdaFisher, 200 epochs) and **21449870** (Adam, 210 epochs), both COMPLETED in
`01:04:52` and `01:04:33` on a `h100_1g.10gb` slice, one seed (42), the authors' `train.py` on the
authors' configs with no override.

| | AdaFisher | Adam |
|---|---|---|
| **best test top-1, measured** | **96.32 %** (epoch 198) | **94.78 %** (epoch 179) |
| **published (Table 2)** | 96.25 ± 0.2 | 94.85 ± 0.1 |
| difference from the published mean | **+0.07**, i.e. 0.35 of its std | **−0.07**, i.e. 0.70 of its std |
| final test top-1 | 96.28 % | 94.71 % |
| final train top-1 | 99.30 % | 97.94 % |
| final train loss | 0.0221 | 0.0610 |
| epochs | 200 | 210 |
| median seconds per epoch | 19.15 | 18.17 |
| total training time | 3 834.5 s | 3 818.3 s |

**Both rows are reproduced**, each within a fraction of its own published standard deviation, and the
published *ordering* comes with them: the measured gap is **1.54 points** against the published 1.40.

Three further things this run measures, none of which needed an extra job.

1. **Table 8's epoch counts are the equal-wall-clock protocol, and the arithmetic checks out.** The
   paper gives AdaFisher 200 epochs and Adam 210 and says the comparison is at equal wall-clock time
   (§5). Here the two runs took **3 834.5 s and 3 818.3 s** — **0.4 % apart**. That is not a
   coincidence to be admired, it is a confirmation that the published epoch counts were derived the
   way the paper says they were, on hardware fast enough for the ratio to hold.
2. **AdaFisher's own per-step overhead on ResNet18 is small: 5.3 %** (19.15 s/epoch against Adam's
   18.17 s). Worth recording next to this repository's own overhead measurements for the four
   Kronecker modes, which are one to two times the network's own forward+backward
   (`plan_lot7.md` §6) — `diag`, which is what AdaFisher is, is the cheap one, and this is the
   authors' code confirming it on their own benchmark.
3. **AdaFisher reaches Adam's best accuracy at epoch 158**, after 50.5 minutes of the 63.6 minutes
   Adam's whole run takes, and Adam never reaches AdaFisher's best. So on this setting the paper's
   convergence claim reproduces too, not only its final number.
