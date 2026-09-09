# Lot 8 Implementation Plan — CIFAR-10 from-scratch training, ResNet-50 / ViT, with Adam & AdamW baselines

**Scope** (`docs/reports/plan.md` §8, the table's last row): *"Deferred, on request: CIFAR-10
from-scratch training, ResNet / ViT, with Adam baseline"*, cross-referenced from §6.3 (*"CIFAR-10
classification on ResNet / ViT, trained from scratch, with Adam as an additional baseline"*). This
row has been deferred since lot 1 (`plan_lot1.md` §"out of scope"), was re-confirmed out of scope by
lot 6 (`plan_lot6.md` §3) and lot 7 (`plan_lot7.md` §4), and is now **requested**. Its two stated
prerequisites are in place: the SUA approximation (lot 6 — `plan.md` §6.3: *"Practical consequence:
ResNet/ViT work will require the SUA approximation"*) and the equal-wall-clock harness (lot 7 —
`plan.md` §6.3: *"the equal-budget requirement … must therefore be fixed in wall-clock time"*).

Like lot 7, this lot adds **no new Fisher-approximation math**: `diag`, `kfac`, `ekfac`, `tkfac`,
`tekfac` are implemented and tested end-to-end. Unlike lot 7, it is not purely a bench script — it
needs two networks written from their papers, a CIFAR-10 pipeline, two first-order baselines, and
**three genuine `src/` changes** that this lot's models are the first in the project to require
(§0.3, §0.4, §0.6). It also needs a cluster path (§0.11): a single 7-arm ResNet-50 comparison is
roughly three orders of magnitude past what the MNIST auto-encoder bench costs.

The user's brief for this lot, verbatim in substance: implement the two models from their papers
(and, where useful, their official repositories), train them fully on CIFAR-10, compare against Adam
and AdamW as baselines, produce the **same kind of results as the MNIST bench** (the five Fisher
approximations compared in steps, wall-clock time, etc.), and — if the runs are too heavy locally —
write SLURM jobs for the same HPC as `/Users/baolgr/Documents/AtlasAnalyticsLab`.

---

## 0. Design decisions this lot must resolve before code exists

### 0.1 ResNet-50 for 32×32 inputs — what exactly gets built, and why it is not `torchvision`'s

**Architecture, from the paper.** `papers/resnet_1512.03385.pdf` §4.1, "Deeper Bottleneck
Architectures" (p. 6) + Table 1: the 50-layer net replaces each 2-layer block of ResNet-34 by a
3-layer bottleneck — `1×1` (reduce), `3×3`, `1×1` (restore, `expansion = 4`) — stacked
`(3, 4, 6, 3)` times at widths `(64, 128, 256, 512)`, with stride 2 at the entry of stages 2-4 and a
projection (`1×1` conv + BN) shortcut whenever the shortcut has to change shape. BN goes *"right
after each convolution and before activation"* (§3.4, p. 4), weights are initialized as in He et al.
[13] (`kaiming_normal_`, fan-out), no dropout.

**The CIFAR stem, also from the paper.** §4.2 (p. 7) defines the paper's own CIFAR-10 protocol: the
network *"first"* applies a `3×3` convolution to the 32×32 input — not §4.1's `7×7`/stride-2 +
`3×3` max-pool ImageNet stem, which would take a 32×32 image down to 8×8 before the first residual
block. This lot therefore builds the standard "CIFAR ResNet-50": `conv1 = 3×3, 64, stride 1,
padding 1`, no max-pool, everything downstream unchanged from Table 1. **23.52 M parameters**, 107
hooked modules (53 `Conv2d`, 53 `BatchNorm2d`, 1 `Linear`), 161 parameter tensors — measured, not
estimated.

**Why hand-written rather than `torchvision.models.resnet50`.** `torchvision`'s `Bottleneck.forward`
does `out += identity`, an in-place mutation of the output of a hooked `BatchNorm2d`. This project's
optimizer registers `register_full_backward_hook` on every `BatchNorm2d`
(`optimizer.py::_prepare_model`), and a full backward hook on a module whose output is later mutated
in place is exactly the case PyTorch errors on. Same reason for `nn.ReLU(inplace=False)` throughout.
This is a hook-safety-by-construction requirement of *this* project, not an architectural deviation:
the resulting network is identical to `torchvision`'s modulo the stem and the in-place ops.

**Data augmentation, from the paper.** §4.2: *"4 pixels are padded on each side, and a 32×32 crop is
randomly sampled from the padded image or its horizontal flip"*, on a **45k/5k train/val split**
(§4.2, p. 7: *"which is determined on a 45k/5k train/val split"*), testing on the single 32×32 view.
See §0.9 for the one addition (Cutout) this lot takes from AdaFisher's own configs instead.

### 0.2 ViT for 32×32 inputs — an adaptation, stated as one

**Architecture, from the paper.** `papers/vit_2010.11929.pdf` §3.1, Eq. (1)-(4): split the image
into `P×P` patches, flatten and linearly project them to a constant width `D` (Eq. 1, the "patch
embedding" — implementable as a `Conv2d(3, D, kernel_size=P, stride=P)`, which is the projection of
Eq. 1 written as a convolution); prepend a learnable `[class]` embedding whose final state is the
image representation (Eq. 4); add **learnable 1-D** position embeddings (§3.1, p. 4: *"We use
standard learnable 1D position embeddings, since we have not observed significant performance gains
from using more advanced 2D-aware position embeddings"*); then `L` encoder blocks alternating MSA
and MLP, with **LayerNorm before every block and residual connections after every block** (§3.1,
p. 3, pre-LN), the MLP being *"two layers with a GELU non-linearity"*. The classification head is
*"a single linear layer"* in the fine-tuning regime (§3.1, p. 3); this lot uses the single linear
layer, since a from-scratch CIFAR-10 run has no pre-training/fine-tuning split for the two-regime
distinction to apply to.

**Why the configuration is not one of the paper's.** Table 1 defines only Base (`L=12, D=768,
MLP=3072, h=12`), Large and Huge — all designed for 224×224 inputs at patch 16 (196 tokens) and for
JFT-scale pre-training. There is no "ViT-Small" in this paper; inventing one and attributing it to
the paper would violate this repository's citation rule. What this lot builds is therefore stated as
an **adaptation for 32×32 CIFAR-10**, preserving the paper's structural invariants and nothing more:

| Quantity | Paper (ViT-Base, Table 1 / §3.1) | This lot (`ViT-S/4`, 32×32) | Invariant preserved |
|---|---|---|---|
| patch size `P` | 16 (224 px → 196 tokens) | **4** (32 px → 64 tokens + `[class]` = 65) | token count of the same order |
| width `D` | 768 | **192** | — |
| MLP width | 3072 | **768** | `= 4·D`, exactly as Table 1 |
| heads `h` | 12 | **3** | `D/h = 64`, exactly as Table 1 |
| depth `L` | 12 | **6** | — |
| head | single linear (§3.1) | single linear | identical |

**2.69 M parameters**, 39 hooked modules (1 `Conv2d` patch embedding, 13 `LayerNorm` = 6 blocks × 2
+ the final norm, 25 `Linear` = 6 blocks × (`qkv`, `proj`, `fc1`, `fc2`) + the head), 80 parameter
tensors — of which **two, `cls_token` and `pos_embed`, belong to no hooked
module at all**. That is the direct cause of §0.3.

**Attention must expose its projection as a `Linear`.** `nn.MultiheadAttention` keeps Q/K/V in a
single raw `Parameter` (`in_proj_weight`), never wrapped in any of this project's four
`SUPPORTED_MODULES`. Under `AdaFisherMulti` its weights would silently receive the `F̃ = I` fallback
path (`optimizer.py::_step_fallback`) — i.e. the entire attention mechanism, the majority of a ViT's
non-MLP parameters, would be optimized by plain momentum SGD while the bench claims to be measuring
a Fisher preconditioner. The attention block is therefore written out, with a single
`self.qkv = nn.Linear(D, 3D)` (one hooked module, one `(3D, D)` factor pair) and
`self.proj = nn.Linear(D, D)`.

**Expected accuracy, and what this bench is for.** The ViT paper's own headline caveat (§1, p. 1;
§3.1 "Inductive bias", p. 4; §4.2/Fig. 3) is that ViTs *"lack some of the inductive biases inherent
to CNNs … and therefore do not generalize well when trained on insufficient amounts of data"*.
CIFAR-10's 45k training images is exactly that regime. A from-scratch ViT-S/4 on CIFAR-10 lands well
below both a CIFAR ResNet and a pre-trained ViT, and this lot does **not** treat that as a defect to
tune away: the deliverable is a *relative* comparison of seven optimizers on one fixed architecture,
not a CIFAR-10 accuracy record. This must be stated in the report so the numbers are not misread.

### 0.3 `optimizer.py`'s parameter↔module pairing is wrong on ViT — measured, then fixed

`AdaFisherMulti.step()` is a near-verbatim port of the reference's index-bookkeeping loop
(`adafisher.py:275-307`), which `plan.md` §2.1 deliberately kept *"structurally unchanged"* as
*"brittle and out of scope to rewrite"*. Lot 8 is the first lot whose models break it.

The loop iterates **exactly `len(self.modules)` times**, walking `params` and `self.modules` in
parallel and consuming one parameter per iteration on the shape-mismatch (`_check_dim`) fallback
branch. Any parameter that belongs to *no* hooked module therefore costs one module its turn, and
the last `k` modules are silently never stepped, where `k` is the number of such parameters.

Measured on the ViT of §0.2 (2 optimizer steps, `fisher_mode="diag"`, all five modes behave
identically here since the defect is in the outer loop):

```
[vit_small/diag] n_params=80 n_modules=39 unchanged=4 nograd=0
   unchanged: ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']
[resnet50/diag] n_params=161 n_modules=107 unchanged=0 nograd=0
```

`cls_token` and `pos_embed` (both direct `Parameter`s of the top-level module, hence *first* in
`model.parameters()` order) consume two iterations, so the final `LayerNorm` and **the entire
classification head are never updated** — silently, with finite gradients, no exception, and a loss
that still decreases. ResNet-50 is unaffected (every one of its 161 parameters belongs to one of its
107 hooked modules), which is why no earlier lot could have caught this.

**Resolved: replace the index bookkeeping by an explicit identity-keyed pairing**, built once in
`_prepare_model`:

- `self._owner: Dict[int, Module]` maps `id(module.weight)` and `id(module.bias)` to the module.
- `step()` walks `group["params"]` in order. A parameter that owns no entry, or whose module was
  already stepped this call, is handled by `_step_fallback` / skipped respectively; otherwise the
  module is `refresh`ed and stepped once, consuming both its weight and its bias.
- `_check_dim` (shape-heuristic pairing) becomes dead and is removed with the loop it served.

Two properties this must have, both checked (§2): (i) on every network where the old loop was
correct — `TinyMultiLayerNet`, the MNIST auto-encoder, lots 4/6's conv nets, ResNet-50 — the new
pairing produces **the same modules, the same `refresh` calls, and the same updates in the same
order**, so all 145 existing tests including the bit-exactness ones keep passing unchanged; (ii) on
ViT, all 80 parameters are updated.

The considered-and-rejected minimal alternative was to change `for _ in range(len(self.modules))`
into a `while idx_param < len(params)` loop: it fixes this specific symptom in one line but keeps
pairing parameters to modules by *shape matching*, which silently mis-pairs whenever a stray
`Parameter` happens to match the current module's weight or bias shape. Identity pairing cannot
mis-pair by construction, and is strictly simpler than the code it replaces.

### 0.4 Hook memory at CIFAR scale — measured, and bounded by a new `fisher_batch_samples` knob

The four non-diagonal modes reduce *factors*, but the reductions run on the full pooled
`(example, output-location)` batch, and `ekfac`/`tekfac` additionally **cache the augmented input
batch** (`_cached_h_bar`, needed for the intra-batch `s*`/`Θ` estimator, `plan_lot2.md` §0.3). Every
forward hook fires before any backward hook, so on a `TCov` step **every layer's cache is live at
once**. Measured (analytically exact from the real module shapes, batch 128, fp32):

| Quantity | ResNet-50, `conv_sua=False` | ResNet-50, `conv_sua=True` | ViT-S/4 |
|---|---|---|---|
| `A` factors, all layers | 498.5 MB | **96.1 MB** | 17.0 MB |
| `B` factors, all layers | 225.7 MB | 225.7 MB | 26.0 MB |
| `s*` / `Θ` (`ekfac`/`tekfac`) | 94.1 MB | 53.8 MB | 10.7 MB |
| **`_cached_h_bar` peak (`ekfac`/`tekfac`)** | **3371.3 MB** | **1513.2 MB** | 271.7 MB |
| `eigh`/`inverse` per refresh | 495.4 GFLOP | 122.3 GFLOP | 6.9 GFLOP |
| per-step projection | 287.7 GFLOP | 107.3 GFLOP | 9.2 GFLOP |
| widest module | `layer4.*.conv2`, `d_in=4608` | `fc`, `d_in=2049` | `mlp.fc2`, `d_in=769` |

`A`/`B` are doubled by their inverses (`kfac`/`tkfac`) or eigenbases (`ekfac`/`tekfac`). At the
AdaFisher paper's own batch size of 256 every batch-dependent row doubles again. On the 10 GB MIG
slice this project's sibling repository is allocated (§0.11), `ekfac` on ResNet-50 does not fit —
before counting the network's own activations.

**Resolved: `fisher_batch_samples: Optional[int] = None` on `AdaFisherMulti`.** When set to `k`, the
two hooks pass `input[0].data[:k]` / `grad_output[0].data[:k]` to the approximation instead of the
whole batch. Three reasons this particular form:

1. **Example-axis slicing keeps `h`/`δ` row-paired by construction.** Dimension 0 is the example
   axis for all four supported layer types (`Linear` `(B,…,C)`, `Conv2d` `(B,C,H,W)`, `BatchNorm2d`
   `(B,C,H,W)`, `LayerNorm` `(B,…,C)`), so slicing it identically in both hooks selects *the same
   examples in the same order* on both sides — which is what `ekfac`'s and `tekfac`'s intra-batch
   estimators require. A random row-level subsample of the *pooled* rows would need shared RNG
   state between two independently-firing hooks; a deterministic prefix slice needs none.
2. **It is a statement about the estimator, not about training.** The gradient the optimizer applies
   is still the full-batch gradient; only the curvature *statistic* is estimated from `k` examples,
   on 1 step in `TCov`. This is a smaller change of estimator variance than `TCov=100` itself
   already is.
3. **Default `None` is exactly the current behaviour**, so no existing test or bench changes by one
   ULP; the knob is inert unless a bench sets it.

Chosen bench defaults: `fisher_batch_samples=32` for ResNet-50 (`_cached_h_bar` peak 378 MB at
`conv_sua=True`), `None` for ViT-S/4 (272 MB at batch 128 is already affordable). Both are
overridable; §3's calibration job measures the real peak before the full runs are submitted.

### 0.5 `conv_sua=True` is mandatory for ResNet-50, meaningless for ViT

`plan.md` §6.3 predicted this (*"ResNet/ViT work will require the SUA approximation"*) for ResNet-18;
§0.4's table confirms it for ResNet-50 and quantifies it on the real architecture: the widest input
factor drops from `d_in = 512·3·3 = 4608` to `d_in = 512`, total `A` storage from 498.5 MB to
96.1 MB (5.2×), the per-refresh `eigh`/`inverse` cost from 495 to 122 GFLOP (4.1×) and the per-step
projection from 288 to 107 GFLOP (2.7×). The non-SUA ResNet-50 arms are therefore **not run**; the
flag is set for all four non-diagonal modes on the CNN. On ViT-S/4 the model has exactly one
`Conv2d` (the patch embedding, `4×4`, `d_in = 3·16 = 48`), so `conv_sua` changes essentially nothing
and is left at its default `False` — the same "inert without a Conv2d worth the flag" situation
lot 6's own `test_conv_sua_is_inert_without_any_conv2d_module` locks down.

Consequence for the report, stated up front: the ResNet-50 CNN arms measure **`kfac`/`ekfac`/
`tkfac`/`tekfac` *under SUA*** — `plan_lot6.md` §0.1's documented approximation gap (no
Sherman-Morrison correction for the cross-offset mean-coupling term) and §0.4's bias convention
(read back from the center offset only) are part of what is being measured, not incidental.

### 0.6 Weight decay: `AdaFisherW`'s decoupled variant is needed for the ViT arm

The AdaFisher paper's own experimental note (Table 2's footnote, p. 8): *"Adam and AdaFisher were
used for all CNN architectures, while AdamW and AdaFisherW were applied for all ViT experiments."*
`AdaFisherMulti` currently implements only the coupled (L2-into-the-gradient) rule ported from
`FisherAdapTune`'s `AdaFisher`; the official repository's `AdaFisherW`
(`reference_repos/AdaFisher/optimizers/AdaFisher.py:685`, `_step`) is

```
param -= lr * ( exp_avg / bias_correction / F_tilde  +  weight_decay * param )
```

i.e. AdamW-style decoupling, with the decay term scaled by `lr` and evaluated at the **pre-update**
parameter. Comparing a coupled-decay AdaFisher against AdamW on a ViT would confound the
preconditioner with the decay convention, which is precisely the confound the paper's own footnote
avoids.

**Resolved: `decoupled_weight_decay: bool = False` on `AdaFisherMulti`.** `False` (default) is
today's exact behaviour, byte for byte. `True` skips the `grad.add(param, alpha=wd)` in
`_update_moment` and applies `param.mul_(1 - lr·wd)` before the preconditioned step — algebraically
identical to the official expression above (`p·(1−lr·wd) − lr·d/bc = p − lr·wd·p − lr·d/bc`), in
both `_step_module` and `_step_fallback`. Bench usage: `True` for every AdaFisher arm on ViT,
`False` for every AdaFisher arm on ResNet-50.

### 0.7 What "equal budget" means here — AdaFisher's own WCT protocol, not a fixed epoch count

`plan.md` §6.3 requires a wall-clock budget; lot 7 implemented one with a fixed
`--time-budget` in seconds. For a classification run the AdaFisher paper's own protocol is more
directly comparable and is adopted verbatim (§5, p. 8): *"We employ the Wall-Clock-Time (WCT) method
with a cutoff of 200 epochs for AdaFisher's training"* — i.e. one **reference arm** is trained for a
fixed number of epochs, and every other arm gets **that arm's measured wall-clock time**, running as
many epochs as it fits (the paper's own baseline configs then run 210 epochs where AdaFisher runs
200: `reference_repos/AdaFisher/Image_Classification/configs/adamCNN.yaml` vs `AdaFisherCNN.yaml`).

Operationally, `--budget-mode wct` (the default):

1. The reference arm (`--reference-arm diag`, matching `plan.md`'s framing of `diag` as the
   AdaFisher baseline the four new modes are alternatives to) runs `--epochs N`.
2. Its total elapsed training time `T` becomes the budget for every other arm.
3. Each other arm trains until `elapsed ≥ T`, checked **before each batch** — reusing lot 7's
   semantics exactly (`plan_lot7.md` §0.1: overshoot bounded by one batch's own processing
   duration), and capped at `--max-epoch-factor × N` epochs so a much cheaper arm cannot run
   unboundedly longer if `T` turns out generous.
4. Every arm additionally reports its **per-epoch** curve, so the epoch-fixed comparison (the
   "rigged in favour of the expensive modes" one, `plan.md` §6.3) is still readable off the same
   run — `--budget-mode epochs` forces it for all arms, for a sanity check.

Validation happens on the 5k held-out split at every epoch boundary and is **excluded from the
budget clock** (it is not part of the optimizer's cost and would otherwise penalize an arm purely
for completing more epochs). The 10k test set is evaluated **once**, at the end of each arm.

### 0.8 Hyperparameters: taken from AdaFisher's own tuned table, not re-tuned here

Per-arm hyperparameter tuning is explicitly out of scope (§4), exactly as in lot 7. The shared
operating point comes from the AdaFisher paper's Appendix D ("HP Tuning", Table 9 — learning rates
tuned by grid search on CIFAR-10 at batch 256) and from the official repository's shipped configs
(`Image_Classification/configs/*.yaml`):

| | ResNet-50 (CNN arms) | ViT-S/4 (ViT arms) | Source |
|---|---|---|---|
| `lr`, AdaFisher arms | `1e-3` | `1e-3` | Table 9, "CNNs"/"ViTs" × AdaFisher/AdaFisherW |
| `lr`, Adam/AdamW arms | `1e-3` | `1e-4` | Table 9 ("CNNs"/Adam; "ViTs"/AdamW, *"adopted from the original publications"*) |
| weight decay | `5e-4`, coupled | `1e-2`, decoupled | Appendix D ("uniform weight decay of `5×10⁻⁴` … for CIFAR-10"); `AdaFisherViT.yaml` |
| `Lambda` (damping) | `1e-3` | `3e-3` | `AdaFisherCNN.yaml` / `AdaFisherViT.yaml` |
| batch size | 128 (see below) | 128 | deviation, stated |
| schedule | `CosineAnnealingLR`, `T_max = --epochs` | idem | Appendix D (*"cosine annealing learning rate decay"*); `scheduler_kwargs.T_max` |
| `TCov` | 100 | 100 | `CLAUDE.md` default, unchanged across lots 1-7 |
| `gammas` | `(0.92, 0.008)` | `(0.92, 0.008)` | this project's default (`CLAUDE.md`); *not* the paper's scalar `γ=0.8`, see below |
| `T_inv` / `T_eig` / `T_re` | 100 / 100 / 1 | idem | `CLAUDE.md` defaults |

Two deliberate deviations, both flagged rather than hidden:

- **Batch size 128, not the paper's 256.** Halves every batch-dependent row of §0.4's memory table
  and the network's own activation footprint, which is what makes a 10 GB MIG slice viable (§0.11).
  It is also the batch size the official repo's own ViT+AdamW config uses
  (`adamViT.yaml: mini_batch_size: 128`). Since all seven arms share it, it cannot bias the
  comparison; it only makes the absolute accuracies non-comparable with the paper's Table 2.
- **`gammas = (0.92, 0.008)`, not the paper's tuned `γ = 0.8`.** This repository's whole test suite,
  and every lot from 1 to 7, is built on the two-element `gammas` convention of
  `FisherAdapTune/scripts/adafisher.py` (the authoritative reference, `CLAUDE.md`'s hard
  constraints). Silently switching to the official repo's scalar `γ` here would change the EMA
  semantics under the four new modes for the first time in the project, in the one lot least able
  to isolate the effect. Kept, and noted as a difference from the paper's tuned CNN setting.

### 0.9 Data pipeline

- **Split.** 45k train / 5k val from the official 50k train set, seeded and reproducible; the
  official 10k test set held out entirely. This is the ResNet paper's own CIFAR protocol (§4.2,
  *"determined on a 45k/5k train/val split"*), not an invention.
- **Train transform.** `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` (ResNet §4.2, verbatim)
  → `ToTensor` → `Normalize` (per-channel CIFAR-10 mean/std) → **Cutout** (1 hole, 16 px).
- **Cutout** is not in the ResNet paper; it is in every one of AdaFisher's own shipped configs
  (`cutout: True, n_holes: 1, cutout_length: 16`), citing DeVries & Taylor (2017), and the paper
  reports both with- and without-Cutout figures (Figs. 16-18). Implemented as an explicit
  `--cutout/--no-cutout` flag defaulting to **on** (reference-faithful), so the choice is visible
  in the CLI rather than baked in.
- **Eval transform.** `ToTensor` + `Normalize` only — ResNet §4.2's *"we only evaluate the single
  view of the original 32×32 image"*.
- **Download policy.** `--data-root` with `download` attempted only when the dataset is absent
  *and* `--allow-download` is set (default on locally, **off** in every SLURM script): Alliance
  Canada's Rorqual compute nodes have no internet, and a silent network call there fails as an
  opaque timeout rather than a clear "dataset not staged" error.

### 0.10 What the bench reports — the MNIST bench's outputs, plus classification metrics

The brief asks for *"the same type of results as the MNIST one"*. Lot 7's outputs are `records.csv`,
`summary.md` and two plots (loss vs. epoch, loss vs. wall-clock) built from a `StepRecord`
dataclass. This lot **reuses `equal_wallclock_bench.py`'s `StepRecord`, `_sync` and `_median`
directly** (import, not copy) and adds exactly what a classification run has that a reconstruction
run does not:

| Output | Content |
|---|---|
| `records.csv` | per step: `arm, step, epoch, elapsed_s, loss, fwd_bwd_s, step_s` — identical schema to lot 7 |
| `epochs.csv` | per epoch: `arm, epoch, elapsed_s, train_loss, val_acc, lr` |
| `summary.md` | per arm: steps, epochs, final train loss, **best val acc**, **final test acc**, median fwd+bwd (ms), median step (ms), **step/fwd+bwd ratio** (lot 7's §6.3-overhead measurement, replayed on real architectures), total wall-clock |
| `loss_vs_epoch.png`, `loss_vs_time.png` | lot 7's two plots, one line per arm |
| `valerr_vs_epoch.png`, `valerr_vs_time.png` | validation *error* against both axes — the AdaFisher paper's own figure set (Figs. 16-18: *"WCT training loss, test error, for CNNs and ViTs on CIFAR10"*) |

The `step/fwd+bwd` ratio column is the point of contact with lot 7's headline finding (`plan_lot7.md`
§6: `kfac`/`tkfac` ≈ 1.0×, `ekfac`/`tekfac` ≈ 1.8× on the MNIST auto-encoder, against `plan.md`
§6.3's a-priori ≈ 2.1× estimate for "all non-diag modes"). Two architectures with a completely
different layer mix — 53 convolutions under SUA, 13 `LayerNorm`s + 25 `Linear`s — are the natural
test of how architecture-dependent that split is. **Predicted, and recorded here before the runs so
the prediction is falsifiable**: the ratio should fall on both models relative to MNIST, because the
auto-encoder's fwd+bwd is unusually cheap relative to its factor dimensions
(`d_in` up to 1001 for a 2.84 M-parameter model), whereas ResNet-50's fwd+bwd at batch 128 dominates
its 107 GFLOP/step of SUA projection.

### 0.11 Local feasibility, and the cluster

**Not runnable locally at full scale.** Seven arms × ResNet-50 × 50 epochs on this machine's MPS
backend is on the order of days, and the `ekfac`/`tekfac` arms need ~2 GB of factor state on top of
training. `benchmarks/equal_wallclock_bench.py`'s reference run took 60 s per mode; this is ~4
orders of magnitude more work. The bench is therefore built to run **locally in smoke form**
(`--epochs 1 --train-subset 512 --arms diag adam`, seconds) and **on the cluster in full**.

**The cluster** is the one `/Users/baolgr/Documents/AtlasAnalyticsLab` targets — an Alliance Canada
H100 cluster (Fir or Rorqual), as documented in that repository's
`experiments/default/slurm/README.md` (verified there against `docs.alliancecan.ca` and live
`module spider`, 2026-07-19) and used by its 48 already-submitted jobs. The invariants this lot
copies from working, already-run scripts there rather than re-deriving:

- `#SBATCH --account=def-msh-ab` (confirmed via `sshare -U`; `rrg-msh` has no GPU allocation).
- `#SBATCH --gpus=h100_1g.10gb:1` — a **MIG slice** (1/8 compute, 10 GB), the only GPU shape that
  account's allocation supports; `--gres=gpu:` is deprecated in favour of `--gpus`/`--gpus-per-node`.
- `--cpus-per-task=8`, `--mem=16G` (`--mem=0` is harmful on a shared MIG node), **no `--partition`**
  (the docs' own guidance is to let the scheduler place the job).
- `module purge && module load StdEnv/2023 gcc/13.3 cuda/12.6 python/3.11.5`, then a
  `$SLURM_TMPDIR` virtualenv with `pip install --no-index` against the Alliance wheelhouse.
- `set -euo pipefail; cd "$SLURM_SUBMIT_DIR"` and submission from the repository root; `--output`
  into a `logs/` directory that must already exist.
- No internet on the compute node: `dataset/` staged by `rsync` beforehand, loaders with
  `download=False`.

The 10 GB ceiling is what §0.4's `fisher_batch_samples` and §0.5's mandatory `conv_sua` are for, and
what §3's calibration job verifies before the long jobs are queued. A single-line comment in each
script documents the `--gpus=h100:1` / `h100_3g.40gb:1` switch for an account with a full-GPU
allocation.

**`--time` budgets are unmeasured extrapolations** until §3's calibration job runs — flagged in each
script's header exactly as the sibling repository flags its own (*"UNMEASURED extrapolation —
re-tighten from real sacct data once this job has actually run once"*).

### 0.12 Seeds, arms, and what a "run" is

- **Arms**: `diag, kfac, ekfac, tkfac, tekfac` (`AdaFisherMulti`) + `adam` (`torch.optim.Adam`) +
  `adamw` (`torch.optim.AdamW`) = **7 per model**, 14 jobs for the two models.
- **One seed** (`--seed 0`) per arm, matching every earlier lot (`plan_lot7.md` §4: *"a single
  --seed 0 run per mode, matching every earlier lot's smoke/bench convention"*). The AdaFisher paper
  averages 5 runs; doing the same here would be 70 jobs. `--seed` is a CLI argument and each SLURM
  script is one arm, so a multi-seed replication is a resubmission, not a code change. Stated as a
  limitation in the report rather than silently implied away.
- Every arm starts from **identical initial weights** (`torch.manual_seed(seed)` immediately before
  model construction, lot 7's `plan_lot7.md` §0.5 convention) and the same data order.
- One arm = one process = one output directory. Arms are independent and can run concurrently.

---

## 1. Implementation

### 1.1 `src/adafisher_modes/optimizer.py` — three changes, all default-inert

| Change | §  | Default | Blast radius |
|---|---|---|---|
| identity-keyed `_owner` pairing replacing the index loop + `_check_dim` | §0.3 | — (always on) | must leave all 145 existing tests bit-identical (§2) |
| `fisher_batch_samples: Optional[int] = None` | §0.4 | `None` = today's behaviour | two hook bodies |
| `decoupled_weight_decay: bool = False` | §0.6 | `False` = today's behaviour | `_update_moment`, `_step_module`, `_step_fallback` |

No other file in `src/` is touched. In particular no `approximations/*.py` and no `factors.py`
change: both models are built entirely from the four `SUPPORTED_MODULES`, all within lots 4-6's
declared scope (`Conv2d` `groups=1`/`dilation=(1,1)`; `LayerNorm` with a 1-tuple `normalized_shape`
— ViT's `LayerNorm(192)` qualifies, and its `(B, N, C)` input is already handled by
`_pool_layernorm`'s flatten).

### 1.2 `benchmarks/cifar10_models.py`

`build_resnet50_cifar(num_classes=10)` (§0.1) and `build_vit_small_cifar(num_classes=10)` (§0.2),
plus a `MODELS` registry `{name → (builder, display_name)}`. Self-contained, no import from `src/`,
no dependency on either reference repository. Every architectural constant carries its paper
citation in a comment.

### 1.3 `benchmarks/cifar10_data.py`

`build_dataloaders(...) -> (train, val, test)` implementing §0.9: seeded 45k/5k split, the two
transforms, `Cutout` (own class, ~15 lines), `download` gating, `--train-subset` support for smoke
runs, `pin_memory` only under CUDA.

### 1.4 `benchmarks/cifar10_classification.py` — the entrypoint

Imports `StepRecord`, `_sync`, `_median`, `write_csv` from `equal_wallclock_bench` (§0.10) and adds:

- `train_classifier_under_budget(model, optimizer, train_loader, val_loader, loss_fn, *, budget_s,
  max_epochs, scheduler, device, eval_fn, max_steps)` — lot 7's per-batch budget check
  (`plan_lot7.md` §0.1) with an epoch-boundary validation pass whose duration is **subtracted from
  the clock**, returning `(step_records, epoch_records)`. Model- and dataset-agnostic, so
  `tests/test_cifar10_bench.py` can drive it on a synthetic task (lot 7's §0.8 pattern).
- `make_optimizer(arm, model, args)` — the seven arms of §0.12, with §0.8's per-model
  hyperparameters and §0.5's/§0.6's per-model flags resolved from `--model`.
- `evaluate(model, loader, device) -> (loss, top1)`.
- `run_all_arms(...)` implementing §0.7's two budget modes.
- `write_summary` / `write_plots` for §0.10's outputs (plots skipped, not failed, without
  `matplotlib`, per `plan_lot7.md` §0.9).

CLI (defaults shown are the ResNet-50 ones; `--model vit_small` swaps in §0.8's ViT column):

```
--model {resnet50,vit_small}  --arms diag kfac ekfac tkfac tekfac adam adamw
--epochs 50  --budget-mode {wct,epochs}  --reference-arm diag  --max-epoch-factor 3
--batch-size 128  --lr ...  --baseline-lr ...  --weight-decay ...  --lam ...
--tcov 100 --gammas 0.92 0.008 --t-inv 100 --t-eig 100 --t-re 1
--conv-sua/--no-conv-sua  --decoupled-wd/--no-decoupled-wd  --fisher-batch-samples 32
--cutout/--no-cutout  --data-root ... --allow-download/--no-allow-download
--device {auto,cpu,cuda,mps}  --train-subset N  --seed 0  --output-dir ...
```

### 1.5 `benchmarks/slurm/` — 14 job scripts + calibration + README

- `calibrate_<model>.sh` (2 jobs, `--time=00:40:00`): 2 epochs × all 7 arms at `--train-subset 5000`,
  printing steps/s, median step time and `torch.cuda.max_memory_allocated()` per arm. **Run first**
  (§3).
- `train_<model>_<arm>.sh` (14 jobs): one arm each, `--budget-mode wct` for the four non-reference
  AdaFisher arms and the two baselines, plain `--epochs N` for the reference arm.
- `README.md`: submission order, the staging step, what each `#SBATCH` line requests and why (with
  the source, following the sibling repository's own README convention), and an explicit "these
  `--time` values are unmeasured until the calibration job has run" warning.
- `requirements-cluster.txt` at the repository root: minimum-version (not pinned) `torch`,
  `torchvision`, `numpy`, `matplotlib` — pinned versions from a macOS dev box routinely do not exist
  in the Alliance wheelhouse.

---

## 2. Tests (`tests/test_cifar10_bench.py`, new — offline, no CIFAR download)

| Test | Assertion | Source |
|---|---|---|
| `test_resnet50_shapes_and_param_count` | forward `(2,3,32,32) → (2,10)`; 23.52 M params; 53 `Conv2d` + 53 `BatchNorm2d` + 1 `Linear` hooked; stem is `3×3`/stride 1 with no max-pool | §0.1 |
| `test_vit_shapes_and_param_count` | forward `(2,3,32,32) → (2,10)`; 2.69 M params; 65 tokens; `qkv` is a single `nn.Linear` (not `MultiheadAttention`) | §0.2 |
| `test_every_parameter_is_updated[resnet50,vit_small]` | after 2 real `AdaFisherMulti` steps, **no** parameter is bit-identical to its initial value — the §0.3 regression, on both models | §0.3 |
| `test_pairing_matches_legacy_loop_on_reference_nets` | on `TinyMultiLayerNet` + the lot-4 conv net, the new pairing selects exactly the `(module, weight, bias)` triples the old index loop did | §0.3 |
| `test_fisher_batch_samples_is_inert_when_larger_than_batch` | `fisher_batch_samples=1000` on a batch of 8 gives a bit-identical trajectory to `None`, all five modes | §0.4 |
| `test_fisher_batch_samples_bounds_the_cached_batch` | with `k=2`, `ekfac`'s cached `h_bar` has exactly the rows `k=2` examples produce | §0.4 |
| `test_decoupled_weight_decay_matches_official_rule` | one step of `decoupled_weight_decay=True` equals `p·(1−lr·wd) − lr·d/bc` computed by hand from the official `AdaFisherW._step` expression | §0.6 |
| `test_coupled_weight_decay_unchanged` | `decoupled_weight_decay=False` is bit-identical to the pre-lot-8 code path | §0.6 |
| `test_budget_harness_excludes_eval_time` | on a synthetic task with a deliberately slow `eval_fn`, elapsed budget accounting ignores the eval | §0.7 |
| `test_wct_budget_gives_every_arm_the_reference_time` | in `wct` mode all non-reference arms stop within one batch of the reference's elapsed time, or at the epoch cap | §0.7 |
| `test_cutout_masks_one_square_region` | `Cutout(1, 16)` zeroes exactly one ≤16×16 axis-aligned region, and is identity when disabled | §0.9 |
| `test_summary_and_csv_schema` | `summary.md`/`epochs.csv` contain the columns §0.10 promises, on a 2-arm synthetic run | §0.10 |

Plus the non-regression requirement, run as a gate, not written as a new test: **`.venv/bin/pytest
tests/ -v` must still be 145 passed** before the new file is added, and 145 + new after.

No test in this file downloads CIFAR-10, builds a full ResNet-50 training run, or touches the
network — the same discipline as `tests/test_equal_wallclock_bench.py` (`plan_lot7.md` §0.8). The
two model-construction tests do instantiate the real models (a fraction of a second, no data).

---

## 3. Producing the reference runs

1. **Local smoke** (minutes): `--model resnet50 --arms diag adam --epochs 1 --train-subset 512
   --budget-mode epochs`, then the same for `vit_small`. Confirms the whole pipeline end to end,
   including plots and CSVs.
2. **Stage and sync**: `rsync` the repository (respecting `.gitignore`) and the CIFAR-10 `dataset/`
   to the cluster; ensure `benchmarks/slurm/logs/` exists.
3. **Calibration jobs** (2 jobs, ~40 min): read off, per arm, steps/s, median `step_s`/`fwd_bwd_s`,
   and peak CUDA memory. Confirms §0.4's `fisher_batch_samples=32` and §0.5's `conv_sua=True` keep
   every arm inside 10 GB, and turns every `--time` in the 14 training scripts from an extrapolation
   into a measurement.
4. **Reference arms** (2 jobs): `diag` at `--epochs 50`, one per model. Their measured elapsed times
   are the WCT budgets.
5. **Remaining 12 arms**, submitted with `--budget-mode wct --wct-budget <T>` from step 4.
6. **Record findings** in §6 of this document, in `CLAUDE.md`'s status block, and in a short report,
   citing measured numbers (every previous lot's convention) — in particular the §0.10 prediction
   about the `step/fwd+bwd` ratio, whether it held.

`--epochs 50` rather than the paper's 200: at 200 epochs × 7 arms × 2 models the ResNet-50 arms alone
would need days of H100 time per arm, and the comparison this lot is asked for (which optimizer gets
further per unit wall-clock) is fully readable at 50. Stated as a deviation; `--epochs` is a CLI flag.

---

## 4. Explicitly out of scope for lot 8

- **Per-arm hyperparameter tuning.** §0.8's operating point is AdaFisher's own tuned one, applied
  unchanged to all seven arms; it is not each arm's best case (lot 7 made the same call, §4).
  Notably `Lambda`, `TCov`, `T_inv`/`T_eig` and `gammas` are *not* swept, and the four new modes have
  never been tuned by anyone, on any task.
- **Multi-seed statistics.** One seed per arm (§0.12).
- **CIFAR-100, Tiny ImageNet, ImageNet-1k.** The brief says CIFAR-10; the data layer is written so a
  second dataset is a new module plus a registry entry, but none is added here.
- **SGD, K-FAC-as-a-baseline, AdaHessian, Shampoo.** The brief names Adam and AdamW; the AdaFisher
  paper's other four baselines are a different (and much larger) comparison.
- **Non-SUA ResNet-50 arms** (§0.5) and **mixed precision / `torch.compile` / DDP**: `AdaFisherMulti`
  has no distributed path at all (`optimizer.py`'s own lot-1 scope note), and every timing claim in
  this project so far is single-device fp32.
- **New Fisher math.** No `approximations/*.py` or `factors.py` change (§1.1).
- **Actually submitting the SLURM jobs.** The scripts, the README and the calibration job are the
  deliverable; queueing them needs cluster credentials and a staged dataset, and the sibling
  repository's own convention is that submission is a human step.

---

## 5. Points worth flagging explicitly

1. **§0.3 is a real bug fix in shipped code, not a new feature.** Every ViT-shaped model — anything
   with a raw `Parameter` outside the four hooked module types — has been silently skipping its last
   `k` modules since lot 1. It has never been observable on any net in this repository's tests. The
   fix's correctness rests on a claim (existing tests stay bit-identical) that is checked, not
   assumed.
2. **The four new modes have never been trained on a real classification task by anyone.** Lots 2-6
   validated them against theorems on toy blocks; lot 7 ran them on a 2.84 M-parameter
   auto-encoder for 60 s. A divergence, a NaN, or a plainly worse curve on ResNet-50 would be a
   *finding*, not a failed lot — and the honest framing of the exit criterion is "seven curves,
   correctly measured", not "the new modes win".
3. **SUA is part of what is measured on the CNN** (§0.5), including its documented, uncorrected
   cross-offset gap (`plan_lot6.md` §0.1). A ResNet-50 result therefore cannot be read as a
   statement about EKFAC-with-full-patch-factors.
4. **`fisher_batch_samples` changes the estimator** (§0.4). It is defaulted off, set to 32 only on
   the CNN, and reported in the summary so no curve is read without it. If the calibration job (§3)
   shows the full batch fits, it should be turned off.
5. **The `diag` arm here runs with min-max normalisation ON** (its paper-faithful default, and what
   `plan_lot7.md` §0.10 chose for the same reason), so it is the AdaFisher of the paper, not the
   bit-exactness-check configuration of `tests/test_diag_bitexact.py`.
6. **`--epochs 50` and batch 128 both differ from the paper's protocol** (§0.8, §3); absolute
   accuracies are not comparable to the paper's Table 2, only the seven arms to each other.

---

## 6. Results

### 6.1 What has actually been measured (implementation session)

The convergence comparison itself is **not** in this section: §3's cluster runs have not been
submitted, so **no lot-8 accuracy or convergence number exists**. What follows is everything this
lot measured while building it — all of it reproducible from the repository, none of it a training
result.

**Networks** (`tests/test_cifar10_bench.py`, and the numbers §0.1/§0.2 quote):

| | ResNet-50 (CIFAR stem) | ViT-S/4 |
|---|---|---|
| parameters | 23.52 M | 2.69 M |
| hooked modules | 107 = 53 `Conv2d` + 53 `BatchNorm2d` + 1 `Linear` | 39 = 1 `Conv2d` + 13 `LayerNorm` + 25 `Linear` |
| parameter tensors | 161, all owned by a hooked module | 80, of which 2 (`cls_token`, `pos_embed`) owned by none |

**The §0.3 pairing bug**, before the fix, 2 steps, `fisher_mode="diag"`:

```
[vit_small] n_params=80 n_modules=39 unchanged=4    ['norm.weight','norm.bias','head.weight','head.bias']
[resnet50]  n_params=161 n_modules=107 unchanged=0
```

after the fix, `unchanged=0` on both, with the 145 pre-lot-8 tests (bit-exactness included) still
passing and `test_pairing_matches_legacy_loop_on_reference_nets` asserting the new pairing
reproduces the removed loop's selection exactly on `TinyMultiLayerNet` and on a conv net.

**Hook memory and cost at CIFAR scale** — §0.4's table, computed exactly from the real module
shapes at batch 128, fp32. The two decisions it forced: `conv_sua=True` on ResNet-50 (§0.5) and
`fisher_batch_samples=32` there (§0.4).

**A 4-step ResNet-50 smoke** (MPS, batch 64, `--train-subset 256`, `conv_sua=True`,
`fisher_batch_samples=32`), whose only purpose was to prove all five modes run end to end on this
architecture:

```
| arm    | final train loss (4 steps) | test acc | median fwd+bwd | median step | step/fwd+bwd |
| diag   | 17.5591                    | 10.00%   | 126.73 ms      |   2.73 ms   | 0.02x |
| kfac   |  2.3279                    | 10.19%   | 127.93 ms      | 173.46 ms   | 1.36x |
| ekfac  |  2.3407                    |  9.42%   | 127.18 ms      | 279.73 ms   | 2.20x |
| tkfac  |  2.4313                    | 10.11%   | 123.03 ms      | 172.22 ms   | 1.40x |
| tekfac |  2.3477                    |  9.33%   | 133.07 ms      | 266.40 ms   | 2.00x |
```

**Neither column is a measurement of anything.** With 4 steps, `write_summary`'s `skip_first=5`
cannot skip anything, so every median includes step 0's one-off `eigh`/`inverse` refresh — the
ratios are inflated by construction, and are not comparable to lot 7's steady-state 1.0×/1.8×.
Accuracies at 10% are chance on CIFAR-10, as expected after 4 steps.

One thing in that table *is* worth recording, as an observation to not re-discover: `diag`'s train
loss after 4 steps (17.6) is an order of magnitude above the four Kronecker modes' (≈2.3). This is
the expected consequence of Eq. (4)'s min-max normalisation — which is ON by default, deliberately
(§5 point 5) — putting `F̃_D` in `[0,1] + λ`, so `m̂/F̃_D` can reach ~`1/λ = 1000×` the gradient
early in training. Four steps at a constant `lr` says nothing about 50 epochs under a cosine
schedule; if the real ResNet-50 `diag` arm turns out unstable, this is the first place to look, and
the first thing to check is whether `λ` (not the mode) is miscalibrated for a 23.5 M-parameter CNN.

### 6.2 The full comparison

*(To be filled once §3's cluster runs complete — measured numbers only, following every previous
lot. §0.10's prediction about the `step/fwd+bwd` ratio falling on both architectures relative to
MNIST is recorded before the fact and should be answered here explicitly, whichever way it goes.)*
