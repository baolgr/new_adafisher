# CLAUDE.md — AdaFisher with interchangeable Fisher approximation modes

## Purpose

Add four **alternative modes** for building AdaFisher's second moment `v^(t)` (Martins Gomes et al.,
ICLR 2025): K-FAC, EKFAC, TKFAC, TEKFAC. All five modes share the same exact per-layer Fisher block

```
F_l = E[ h̄_{l-1} h̄_{l-1}ᵀ  ⊗  δ_l δ_lᵀ ]
```

(`h̄` = bias-augmented layer input, `δ` = gradient backpropagated to the pre-activation), and differ
only in what they do with it. Everything else — hooks, factor EMA, `β₁` momentum, absence of a square
root on `v^(t)` — is **identical across the five modes**.

> **Status: lots 1-7 complete.** Lot 1: `diag` mode, bit-exact against FisherAdapTune with min-max
> off, Eq. (4) semantics verified with min-max on. Lot 2: `kfac`/`ekfac` (`Linear` only), Frobenius
> dominance of EKFAC over KFAC verified on a toy per-sample-exact Fisher block, `Q_A`/`Q_B`
> orthogonality verified. Lot 3: `tkfac`/`tekfac` (`Linear` only), `tr(F̃_TKFAC)=tr(F)` and
> Frobenius dominance of TEKFAC over TKFAC verified on the same toy block, EKFAC↔TEKFAC agreement
> verified in the degenerate case. Lot 4: `Conv2d` (`groups=1`, `dilation=(1,1)`) for all four
> non-diagonal modes, reusing lot 1's `extract_patches` unmodified; the same four §6.1 assertions
> (EKFAC/KFAC and TEKFAC/TKFAC dominance, `tr(F̃_TKFAC)=tr(F)`, eigenbasis orthogonality) replayed
> on a toy `3×3`, 2-input-channel conv against an independent `torch.nn.functional.unfold`-based
> oracle; a real `AdaFisherMulti` (hooks, not direct calls) verified end-to-end on a small
> `Conv2d`+`Linear` network for all four modes. Lot 5: `BatchNorm2d`/`LayerNorm` (`normalized_shape`
> a 1-tuple) for all four non-diagonal modes — Proposition 3.1's exact FIM for a normalisation layer
> is Hadamard-, not Kronecker-, structured; fit into the shared `kron(A,B)` machinery via a
> Frobenius-optimal, `S`-independent `2×2` scalar-ish factor, needing **zero** changes to any
> `approximations/*.py` file (only `factors.py` gained new branches); the same four §6.1 assertions
> replayed on a toy `BatchNorm2d(3)` against an independent per-position oracle, plus a real
> `AdaFisherMulti` end-to-end smoke test and the literal "loss non-regression vs. `diag`" exit
> criterion on `TinyMultiLayerNet` (the net already covering all four `SUPPORTED_MODULES`). Design
> decisions are settled in `docs/reports/plan.md` §5 and §9; lot 1's implementation-precision
> refinements (three ABC signature corrections) are in `docs/reports/plan_lot1.md` §0; lot 2's
> (full-factor extraction, the `s*` estimator, the `rvec` derivation, shared weight/bias plumbing)
> are in `docs/reports/plan_lot2.md` §0; lot 3's (the un-normalized-numerator EMA needed for exact
> trace preservation, the `(1/δ)Ψ̃⁻¹MΦ̃⁻¹` damping derivation, TEKFAC's reuse of TKFAC's factors,
> decoupled `T_eig`/`T_re` cadences) are in `docs/reports/plan_lot3.md` §0; lot 4's (a verified scale
> quirk in the existing, unmodified diagonal `Conv2d` path deliberately not reproduced,
> `groups=1`/`dilation=(1,1)` scope guards, why `kfac.py` needed zero logic changes, the
> pooled-`(example, output-location)` reading of the exit test's oracle) are in
> `docs/reports/plan_lot4.md` §0; lot 5's (the Hadamard-vs-Kronecker structure of Proposition 3.1,
> the Frobenius-optimal scalar surrogate `a_ν` for the exact `H|_ν`, why the small `ν`-`β` coupling
> term is kept rather than forced to zero) are in `docs/reports/plan_lot5.md` §0. Lot 6: the SUA
> approximation (`conv_sua=True`) for `Conv2d`'s input factor, all four non-diagonal modes —
> collapses the patch-based factor from `(C_in·k_h·k_w[+1])²` to `(C_in[+1])²` entries (the "4609 →
> 512" reduction `plan.md` §6.3/§7 flag) by pooling the *center* offset of every patch instead of
> the whole patch, a center-slice construction that is row-aligned with the output factor's own
> pooling by construction (unlike `EKFAC-pytorch`'s own SUA, which pools the raw input
> independently and can misalign for `stride≠1`/non-"same" padding); `precondition` applies the
> resulting small operator independently at each kernel offset (`_kron_utils`'s new
> `augment_conv2d_direction_sua`/`split_conv2d_direction_sua`), with the bias direction broadcast
> into every offset and read back from the center offset only. The same four §6.1 assertions
> replayed at the SUA-consistent scale on the lot-4 toy conv, plus the per-offset `f_tilde`/
> `precondition` consistency check, the `plan.md` §5.1-cited "SUA loses information" finding
> measured (not asserted) on that same toy conv, and the `4609→513`/`≈80.7×` memory reduction
> measured at ResNet-18 scale.
>
> **Lot 7: done.** Equal-wall-clock-budget convergence bench, all five modes, on the primary
> MNIST auto-encoder — `plan.md` §6.3's exit test, run for real (`--time-budget 60 --seed 0`,
> shared untuned hyperparameters at the `CLAUDE.md` defaults, `docs/reports/plan_lot7.md` §6).
> Two findings, both measured rather than assumed: (1) `plan.md` §6.3's own FLOP-based "≈2.1×
> overhead" estimate, lumped across "all non-diag modes," splits empirically into two real
> regimes it did not distinguish — `kfac`/`tkfac` (two matmuls against a cached inverse) cost
> `≈1.0×` their own fwd+bwd time in extra per-step work, `ekfac`/`tekfac` (which additionally
> project into and out of an eigenbasis) cost `≈1.8×`; (2) at this shared, untuned operating
> point all five modes converge to a statistically indistinguishable final loss (`0.25%` spread)
> within the 60s budget, but reach it at very different wall-clock times (`diag` ≈0.68s, `tkfac`
> ≈4.6s, `kfac` ≈8.6s, `ekfac`/`tekfac` ≈10.6s) — over `6×` apart between the fastest and the
> epoch-count "winner" (`tkfac`, fewest epochs to the floor), which is exactly the equal-epoch
> rigging `plan.md` §6.3 predicted an epoch-fixed comparison would introduce, now demonstrated
> rather than only argued. `benchmarks/models/mnist_autoencoder.py::_make_optimizer` was generalized to
> build all five modes (`--optimizer kfac|ekfac|tkfac|tekfac|all`) for a quick epoch-fixed sanity
> check; the wall-clock-budget bench itself is the new `benchmarks/equal_wallclock_bench.py`, built
> around a dataset- and model-agnostic training-loop helper so its mechanics (budget adherence,
> the `max_steps` guard) are covered by a fast, offline `pytest` test
> (`tests/test_equal_wallclock_bench.py`) that never touches MNIST or the network. Design
> decisions, and two things an early draft of this plan got wrong and corrected empirically (the
> budget-overshoot direction; the scope of the "non-diag costs more per step" claim), are in
> `docs/reports/plan_lot7.md` §0 and §5; the reference run's full numbers are in its §6.
>
> **Lot 8: code done, cluster runs pending.** `plan.md` §8's last row — formerly *"Deferred, on
> request: CIFAR-10 from-scratch training, ResNet / ViT, with Adam baseline"* — requested, planned
> in `docs/reports/plan_lot8.md`, and fully implemented: two networks written from their papers
> (ResNet-50 with the paper's own CIFAR stem, `resnet_1512.03385.pdf` §4.1/§4.2, 23.52 M params,
> 107 hooked modules; a 32x32 `ViT-S/4` adaptation of `vit_2010.11929.pdf` §3.1 Eq. (1)-(4), 2.69 M
> params, 39 hooked modules — an adaptation because Table 1 defines only Base/Large/Huge, keeping
> that table's `MLP = 4D` and `D/heads = 64`), a CIFAR-10 pipeline on the ResNet paper's own 45k/5k
> protocol, and a **7-arm** bench (the five Fisher modes + `Adam` + `AdamW`) under **AdaFisher's own
> wall-clock-time protocol** (`adafisher_2405.16397.pdf` §5: one reference arm runs a fixed epoch
> count, every other arm gets its measured wall-clock time). The full runs go to SLURM (16 generated
> jobs, `benchmarks/slurm/`, Alliance Canada H100); only a local smoke has been run, so **no lot-8
> convergence result is claimed yet** — §6 of `plan_lot8.md` is deliberately empty. Three genuine
> `src/optimizer.py` changes came out of it, each verified against the 145 pre-lot-8 tests staying
> green: (1) a **real bug fix** — the ported index-bookkeeping `step()` loop ran exactly
> `len(self.modules)` iterations and consumed one per *unpaired* parameter, so `ViT-S/4`'s
> `cls_token`/`pos_embed` silently cost the final `LayerNorm` and **the whole classification head**
> every update (measured, then fixed by identity-keyed pairing — `plan_lot8.md` §0.3); (2)
> `fisher_batch_samples`, bounding `ekfac`/`tekfac`'s cached input batch, measured at **3.37 GB**
> across ResNet-50's layers at batch 128 (1.51 GB even under SUA), since every forward hook fires
> before any backward hook (§0.4); (3) `decoupled_weight_decay`, the official `AdaFisherW` rule,
> without which the ViT arms could not be compared like-for-like with `AdamW` (§0.6, and the
> AdaFisher paper's own Table 2 footnote). All three default to today's exact behaviour. Measured at
> ResNet-50 scale, confirming on the real architecture what `plan.md` §6.3 predicted from ResNet-18:
> SUA takes the widest input factor from `d_in = 4608` to `512`, total `A` storage from 498.5 to
> 96.1 MB, and the per-step projection from 288 to 107 GFLOP — `conv_sua=True` is not optional here.
>
> **Step 1 of the Fisher-drift campaign (`docs/reports/plan_exp_step1.md`, the repository's lot 9):
> done.** `benchmarks/` became a package: one shared harness in `benchmarks/common/` and one folder
> per tested model, replacing the five flat modules lots 1/7/8 had grown (`mnist_autoencoder.py`,
> `equal_wallclock_bench.py`, `cifar10_classification.py`, `cifar10_models.py`, `cifar10_data.py`,
> all deleted, every line of them migrated). The three training loops became one
> (`common/loop.py::train_under_budget`, the merge of lot 7's and lot 8's with both sets of
> semantics preserved — the overshoot bound, the excluded eval time, one scheduler step per
> completed epoch, the `max_steps` guard, all still asserted by their original, unweakened tests);
> the two optimizer factories became one (`common/optimizers.py`, 8 arms including `reference`);
> per-model hyperparameters moved out of the runner into an `HParams` literal in each model's own
> `bench.py`, whose CLI overrides are generated by one loop over `fields(HParams)`. Five new
> networks were written from their papers (`plan_exp_draft.md` §4's A1/A2/A3/B1/B2), each hitting
> its published or derived parameter count **exactly**: `mlp_ln_mnist` 26 634, `cnn_gn_cifar`
> 24 458, `vit_micro_cifar` 21 098, `resnet20_cifar` 269 722 (option-A shortcuts — option B would
> not match the paper's "0.27M"), `cct_2_3x2_cifar` 283 723 (paper: "0.28M"). `--checkpoints
> 0,0.01,0.1,0.5,1` (`common/checkpoints.py`) is the new input steps 2-5 consume. Three numeric
> corrections came out of building it, each measured rather than assumed: `vit_micro_cifar` is
> **21 098**, not `plan_exp_step1.md` §4's estimated 21 162 (mean pooling removes the `cls_token`
> and its position row, 64 parameters); the verbatim-migrated `resnet50_cifar` and `vit_small_cifar`
> are **23 520 842** and **2 693 578**, not §4's transcribed 23 519 178 and 2 685 898 (both round
> to plan_lot8's own "23.52 M" / "2.69 M"). One inherited defect was fixed in the merged runner: in
> `--budget-mode wct` the *reference* arm was given `max_epoch_factor x --epochs`, so it would have
> set the WCT budget from a 3x-long run; the factor now applies to budgeted arms only, which is
> what `plan_lot8.md` §0.7 step 3 describes. Nothing in `src/adafisher_modes/` changed.
>
> **Step-1 follow-up, after the first real cluster campaign: done.** 6 of 8 models completed all 7
> arms (`mnist_autoencoder`, `cnn_gn_cifar`, `vit_micro_cifar`, `cct_2_3x2_cifar` via one grouped
> job each; `resnet50_cifar`, `vit_small_cifar` via 7 per-arm jobs). `mlp_ln_mnist` and
> `resnet20_cifar` died in their `ekfac` arm with `FAILED 1:0`, and the tracebacks
> (`benchmarks/slurm/logs/*_all-*.out`) put both on the same line: `linalg.eigh` on the raw EMA
> factor, `_LinAlgError: ill-conditioned or has too many repeated eigenvalues`. **Three fixes came
> out of it, each measured rather than assumed.** (1) `src/adafisher_modes/` reopened for the first
> time since lot 8: the new `approximations/_eigh_utils.py::eigenbasis` adds a relative
> conditioning ridge before decomposing and falls back to the CPU solver on failure, used by both
> `ekfac.py` and `tekfac.py`. It is inert on the returned basis (`M + cI` shares every eigenvector
> with `M`), so no EKFAC/TEKFAC semantics change and all 238 pre-existing tests stayed green
> unmodified; the cause is measured — `mlp_ln_mnist`'s first `Linear` has `A` of rank 646 out of
> 785 with 136 exactly-zero eigenvalues, because 130 of MNIST's 784 pixels are identically zero.
> The crash is cuSOLVER-specific (replaying the failing arm on CPU runs clean and reaches 97.01%),
> so `tests/test_eigh_conditioning.py` pins the fix's *contract* rather than the crash. (2) The
> checkpoint schedule's denominator was `max_epochs x batches`, i.e. `--max-epoch-factor` times too
> long for every budgeted arm: measured on a completed `cct_2_3x2_cifar` run, `ckpt_0.1` sat at 31%
> of the arm's own run and `ckpt_0.5` never fired at all, on 6 of 7 arms. It is now the *nominal*
> trajectory, shared by every arm of a model, and each payload carries its `total_steps` denominator
> plus a `scheduled` flag. **Every checkpoint produced before this fix is mislabelled except the
> `diag` arms'.** (3) Every SLURM `--time` is now measured from a completed run rather than
> extrapolated, which also corrected an earlier claim in this repository's own SLURM README: setup
> is **~40 s** (`cal_mlp_ln_mnist` COMPLETED in `00:00:44` including `pip install --no-index`), not
> the ~10 min that had been assumed, so "setup costs more than the compute" was false and grouping
> the small models buys ~4 min per model, not ~1 h — it is worth doing for having fewer jobs to
> track, not for machine time. Two non-failures were also cleared up: the missing
> `resnet20_cifar`/`cct_2_3x2_cifar` per-arm `diag` reports are `CANCELLED+` at an identical
> `Elapsed`, i.e. a deliberate `scancel` when the submission strategy changed, not a bug.

> **Campaign-1 audit + the LR-schedule fix: done. Re-runs pending.** All 8 models x 7 arms of the
> first real campaign were verified mechanically — **0** non-finite losses across ~700 k recorded
> steps, `elapsed_s` monotone everywhere, the WCT budget respected within its one-batch overshoot
> bound on all 54 budgeted arms (worst case `+0.22 s`), and `adam` bit-identical to `adamw` on the
> two models with `wd=0`. The strongest check came free: the `diag` arm ran twice, in two
> campaigns a day apart under different job layouts, and its per-step losses are **bit-identical**
> (2200/2200, 8580/8580, 10530/10530, 10530/10530) — the harness's seeding is exact. **Three
> findings, each measured rather than assumed.** (1) A **protocol defect in the cosine schedule**,
> biasing both ends of the WCT comparison: `T_max = --epochs` is shared, but a cheap arm overshoots
> it and `CosineAnnealingLR` is *periodic*, so its LR climbs back up (every overshooting arm was
> `adam`/`adamw`, 7 models of 8; `resnet20_cifar/adam` hit `lr = 0` at epoch 49 then trained 9 more
> epochs with the LR rising to `6.2e-5`, best val acc decaying 88.82% -> 88.06%), while an
> expensive arm stops before the floor (`resnet50_cifar/ekfac` at epoch 38 of 50, `lr` still
> `1.6e-4`; `corr(epochs completed, best val acc)` over the five Fisher arms was **+0.98 /
> +0.92 / +0.92** on the three long models, i.e. the ranking of the modes was largely a ranking of
> schedule position). Fixed by `benchmarks/common/schedules.py` and `--lr-schedule`: `NominalCosine`
> is clamped past `T_max` and provably identical to torch's inside it (so the bit-exact `diag`
> trajectories survive), `BudgetCosine` anneals each budgeted arm over its own budget so every arm
> completes one full cosine. (2) `resnet50_cifar` and `vit_small_cifar` were run **entirely
> pre-fix**, as per-arm jobs with a hand-copied `--wct-budget`, so their checkpoint fractions are
> the mislabelled ones (`ckpt_0.01` at 3% of the trajectory, `ckpt_0.5` never written on 6 of 7
> arms) — being re-run as one grouped job each, which derives the budget in-process. (3) The
> project's **primary** bench is the only failure, and it is unambiguous: on `mnist_autoencoder`
> all five Fisher modes freeze at MSE ~= 0.7085 within 2 epochs of a 20-epoch run while `adam`
> descends monotonically to 0.4836. Measured on the checkpoints, between 10% and 100% of training
> the parameters move **0.2-0.5%** for the five Fisher arms against **147%** for `adam`, having
> travelled **~1.0** from initialisation against **25.8** — a collapsed step size on an 8-layer
> all-sigmoid autoencoder, not a converged optimum. This **reframes lot 7's own conclusion**: "all
> five modes converge to a statistically indistinguishable final loss (0.25% spread)" was true but
> read the wrong way, because lot 7 had no `adam` arm to show that the shared point is a bad one.
> `lam` has since been **measured and exonerated** (`sweep_mnist_autoencoder_lam.sh`, the full WCT
> protocol at `lam in {1e-5, 1e-3, 1e-1}`): four orders of magnitude of damping move the four
> Kronecker modes by **under 1%** (0.7038 to 0.7106, against `adam`'s 0.478 in the same budget),
> and raising `lam` does nothing in any mode. Only `diag` responds, and only at `1e-5`
> (0.708 -> 0.578, still worse than `adam`) — consistent with `lam` capping the `1/lam`
> amplification Eq. (4)'s min-max allows, which is a `diag`-only mechanism. So the stall is **not**
> a damping problem, and the `lam=1e-3`-is-too-large reading this file briefly carried is wrong for
> the four Kronecker modes. Saturating sigmoids are also disqualified as the sole explanation:
> `adam` escapes the same point from the same initialisation. The untested candidates are the
> shared `lr=1e-3` (a `lr` bracket is the natural next sweep) and the geometry of `A (x) B` on this
> net, which the per-layer ratio `||F~^-1 m_hat|| / ||m_hat||` at the stall point would settle. Two
> incidental repairs: `main` now rewrites the report after **every** completed arm (a grouped job
> used to lose every finished arm if a later one crashed, which cost `resnet20_cifar_all` and
> `mlp_ln_mnist_all` two arms each), which is what makes grouping the two large models safe; and
> the superseded/duplicated result trees moved to `benchmarks/archives/` so `outputs/<model>/`
> holds exactly one report plus one checkpoint-only directory per arm.

> **CIFAR-100 and ImageNet-1K: done — campaign 2 has run all 14 benchmarks.** Results in `docs/reports/campaign2_cifar100_imagenet.md`, summarised at the end
> of this block. The six
> image-classification architectures (`cnn_gn`, `vit_micro`, `resnet20`, `cct_2_3x2`, `resnet50`,
> `vit_small`) now exist on three datasets instead of one: **12 new `benchmarks/<model>/` folders**
> (`*_cifar100`, `*_imagenet`), each with all 7 arms under the same WCT protocol, plus **107 new
> SLURM jobs** under `benchmarks/slurm/{cifar100,imagenet}/` and results under
> `benchmarks/outputs/{cifar100,imagenet}/`. `mnist_autoencoder` and `mlp_ln_mnist` are
> deliberately **not** extended: their input dimensionality is part of the architecture (`784-...`,
> and the auto-encoder has no label head at all), and an A1 with a 3072-dim input would be 98 k
> parameters — past `plan_exp_draft.md` §2.2's `P_max` for regime A, i.e. it would stop being the
> model whose exact Fisher is affordable, which is the only reason A1 exists.
>
> **Four decisions, each with a reason.** (1) **CIFAR-100 is free**: `CIFAR100_SPEC` has been in
> `common/data.py` since step 1, the images are the same 50 000 32x32x3 under the same 45k/5k
> split and the same augmentation, so each bench is its CIFAR-10 counterpart with `num_classes=100`
> and the `--time` values are its counterpart's *measured* figures, transferred rather than
> re-derived (+2.2% parameters at worst, `resnet20`: 275 572 against 269 722). (2) **ImageNet-1K
> runs at two resolutions**, because this repository has two families of model: `resnet50_imagenet`
> (the paper's own 7x7/s2 + max-pool stem, `resnet_1512.03385.pdf` Table 1, **25 557 032**
> parameters) and `vit_small_imagenet` (ViT-S/16, **22 050 664**) at the native 224 px; the four
> 32x32-native architectures on **downsampled ImageNet** (Chrabaszcz et al. 2017, arXiv:1707.08819
> §2 — the whole image squashed to 32x32), structurally unmodified apart from the head. For
> `cct_2_3x2` that is not a convenience: at 224 px its tokenizer emits a 3136-token sequence whose
> attention map is ~10 TB at batch 256. **Report those four as "ImageNet32", never as ImageNet-1K.**
> (3) The transforms are **AdaFisher's own**, not re-derived
> (`reference_repos/AdaFisher/Image_Classification/src/utils/data.py:157-194`): `RandomResizedCrop`
> + flip + `Normalize(0.485/0.456/0.406, 0.229/0.224/0.225)` + Cutout at the same `cutout_length:
> 16` its CIFAR configs use, `Resize(256)` + `CenterCrop(224)` on eval. (4) `Benchmark.output_group`
> routes both the results and the jobs into a one-level dataset subdirectory. It was introduced
> empty-by-default for the eight existing benches and then, in a follow-up, set on all twenty:
> `outputs/{mnist,cifar10,cifar100,imagenet}/<model>/` and `slurm/<group>/`, with the eight
> original result trees migrated and `fisher_ref/checkpoints.py` taught to read both layouts.
>
> **What is measured and what is not.** Every new model builds, forwards, and runs all five Fisher
> modes for two real `AdaFisherMulti` steps with no parameter left un-updated — including ResNet-50
> and ViT-S/16 at 224 px (`tests/test_benchmark_models.py`, which picked the folders up the moment
> they existed; **341 -> 524** tests passing in the default run, **565** with `--runslow`, none
> pre-existing modified except the three model-folder registries that exist to fail on a new
> folder). `tests/test_dataset_benches.py` (54
> tests, new) covers the half that is genuinely new: the `ImageFolder` pipeline on a synthetic
> 2-class tree, both transform regimes, the seeded split, the train/val/test wiring, and the
> class-mapping check that catches the realistic staging mistake (ILSVRC's val tar is *flat*).
> ImageNet-1K needs an accepted image-net.org agreement and 155 GB. It is now staged on the cluster
> (job 21447397, `stage_imagenet_tmpdir`, 26 min), and `benchmarks/slurm/imagenet/README.md` is still
> the file to read before submitting anything there.
>
> **Campaign 2, run on 20-21 September 2026 (one seed, seed 0; `docs/reports/campaign2_cifar100_imagenet.md`).**
> Six CIFAR-100 benchmarks, the four ImageNet32 ones, ResNet-50 and ViT-S/16 on ImageNet at 224 px, and a re-run of
> `resnet50_cifar`/`vit_small_cifar` under the fixed protocol (they had been run before the
> checkpoint and schedule fixes). All jobs `COMPLETED`; 8.4 M recorded steps, **zero** non-finite
> losses, every budget respected within `+0.35 s`.
> Five things to carry:
> 1. **The Fisher modes beat the better of `adam`/`adamw` clearly on 6 of 14 benchmarks** (+3.3 to
>    **+21.0** points, CCT on CIFAR-100: 55.8% against 34.8%), sit inside the one-seed noise on 4,
>    and **lose clearly on 4**: ViT-S against `adamw` on all three datasets (CIFAR-10 −4.1,
>    CIFAR-100 −4.7, ImageNet 224 px −6.8: 52.98% against 59.74%), and `cnn_gn` on ImageNet32 (−5.7,
>    all arms at 6-12%). ViT-S is the one architecture where the loss repeats across datasets; the
>    micro ViT, same code and hyperparameters, wins by +9.2 on CIFAR-100.
> 2. **ResNet-50, ImageNet-1K, 30 epochs: `diag` 74.24% on the ILSVRC val set**, `kfac` 74.45,
>    `adamw` 70.93, `adam` 64.99. The `diag − adam` gap is **+9.25**, against **+9.17** in the
>    AdaFisher paper's Table 3 at 90 epochs. Different protocols, so context, not a reproduction.
> 3. **The two groups of the seed table replicate on 14 new benchmarks**: `{diag, kfac, tkfac}`
>    ahead of `{ekfac, tekfac}` in 72 of 84 cross-group pairs, `tkfac > tekfac` 14/14; mean ranks
>    `kfac` 2.21, `tkfac` 2.29, `diag` 2.36, `ekfac` 3.86, `tekfac` 4.29. Same mechanism: steps
>    completed in the same budget, relative to `diag`, are 0.96 / 0.95 / 0.88 / 0.88.
> 4. **The schedule fix is measured on `resnet50_cifar`**: `ekfac` 90.04 -> 94.12 and `tekfac`
>    90.76 -> 93.84 once they complete their cosine; the spread between the five modes goes from
>    3.88 to 0.50 points. `diag`, unbudgeted, reproduces to the digit (93.94, and 67.68 on ViT-S).
> 5. **`adam` collapses wherever the weight decay is 0.01** (the ViTs and CCT): 3.8 to 26.7 points
>    behind `adamw` (up to 53.5 on `vit_small_imagenet`), 0.82% on `vit_micro_imagenet`. `adam` is `torch.optim.Adam(weight_decay=...)`,
>    i.e. decay added to the gradient; that this is the cause is likely and **untested**. Compare
>    those benchmarks against `adamw`.
> 6. **The Fisher arms gain most of their accuracy while the learning rate anneals, and `adamw`
>    does not.** Measured on every local result tree of campaign 2 and the CIFAR-10 runs (20 trees,
>    seed 0): the best validation accuracy each arm adds during the **last 25 % of its own run**.
>    In **19 of 20** trees every one of the five Fisher modes gains more than `adamw`; the one
>    exception is `cnn_gn_cifar100` (Fisher +0.54 at least, `adamw` +0.86). The gap is largest on the
>    ResNets: `resnet20_imagenet` Fisher **+7.0 to +9.4** points against `adamw` +0.4,
>    `resnet50_imagenet` **+5.4 to +8.4** against +0.6, `resnet50_cifar100` +4.8 to +8.1 against +0.7.
>    On ResNet-50 at 224 px, `adamw` leads for most of the run (52.6 % against 38-42 % at a tenth of
>    it) and the Fisher arms overtake it only at the end, finishing with a *higher* training loss
>    (1.09 against 0.91) and a *higher* validation accuracy. **One reading, not tested:** this is
>    what large-step momentum SGD does — noisy while the learning rate is high, converging as it
>    decays, generalising better than Adam on a ResNet and worse on a ViT. It is the reading
>    `plan_lambda_dominance.md` §1.1 predicts: once `lam` dominates the curvature the update is
>    `lr/lam` times the momentum. Converted to `torch.optim.SGD(momentum=0.9)`, whose buffer *sums*
>    gradients where AdaFisher's *averages* them, that is a learning rate of
>    **`(lr/lam)(1 - beta) = 0.1`** at `lam = 1e-3` — the textbook ResNet-on-ImageNet value — and
>    **0.033** at `lam = 3e-3` (the ViT/CCT benches). Two reasons it is only a reading: `lam`
>    dominance was measured on the regime-A models only, and ResNet-50 here also runs SUA and
>    `fisher_batch_samples=32`. **The test that settles it:** an SGD-momentum arm at that learning
>    rate, on the cheap 32 px benches first. If it tracks the Fisher arms, these results measure
>    SGD, not curvature. A curvature-free stand-in already exists for six small networks
>    (`plan_lambda_dominance.md` E0, `fisher_ref/outputs/warmup_sgd_baseline_small.json`); none has
>    been run at this scale. An alternative the gains alone do not exclude: an arm that is further
>    from its plateau at 75 % simply has more left to gain. **Tested on ViT-S by E18** (see the
   E17/E18 block below): momentum SGD at 0.033 does *not* track the Fisher arms there. It does
   worse than `diag` by 6.2 points on CIFAR-10 and 1.6 on CIFAR-100, and the gap is made in the
   first 200 steps. Not yet tested on the ResNets, where the reading was first proposed.
> All of it is at the **shipped** `lam` (1e-3 or 3e-3), where lot 5 measured the applied
> preconditioner to be almost a multiple of the identity: it says how these optimizers do as
> shipped, not what curvature buys. The ImageNet `--time` values in `generate_jobs.py` were
> estimates and are now measured with wide margins (e.g. 2:32 against 16:00 for `cnn_gn_imagenet`,
> 14:00 against 23:00 for `resnet20_imagenet`, 9:29 against 12:00 for `vit_small_imagenet/diag`, the
> tightest); they have not been changed yet. Local copy: the whole `outputs/imagenet/` tree, **including its 210
> checkpoints** (7.8 GB, 390 files), was pulled on 2026-09-21 and checked byte-for-byte against the
> cluster.

> **Fisher-drift campaign, plan v1 + lot 0: done.** `docs/reports/plan_exp_draft.md` is now **the**
> campaign plan: `plan_exp_draft_v0.md` (the original French draft, kept verbatim) adapted to what
> this repository actually produces, with every change listed in its own §0. It also supersedes
> the simplified rewrite that used to sit at the repository **root** as `plan_exp_draft.md`
> (added by `6dd981c`, absent from the working tree since 2026-09-11, recoverable with
> `git show 6dd981c:plan_exp_draft.md`): that file's **scope cuts** (regimes A/B first, metrics
> M1/M5/M7/M8 first, sources type-2 + empirical first, four hypotheses) are kept as v1's §0.11
> priority order, its "reuse `adafisher_modes`, do not rebuild" rule as §0.12, and — because
> comments in `benchmarks/` and `tests/` cite *its* section numbers — §0.13 maps them (its §4
> Models -> v1 §1, its §7 Checkpoints -> v1 §3.5, its §8 Steps -> v1 §9), so no code comment
> needs rewriting. The substantive changes from v0:
> the campaign's models are the **eight `benchmarks/<model>/` folders** with their measured
> parameter counts, not plan-internal definitions (A4/B3 stay deferred, and v0's B4 "realistic
> width d=192" role is already filled by `vit_small_cifar`); its θ axis is the **checkpoints the
> harness already writes** (`{0, 1%, 10%, 50%, 100%} x {diag, adamw} x the seeds that exist`, plus
> five more arms at seed 0), which buys a question v0 could not ask — *does the drift at a given θ
> depend on which optimizer produced that θ?*; **P2 reads `AdaFisherMulti`'s own state for all five
> modes** rather than an upstream adapter for `diag` only, because the port's fidelity (the open
> question v0 made P2 wait on) is settled by `test_diag_bitexact.py`; and, since the checkpoints
> hold **θ only**, P2 must re-warm the EMA — bounded, and that is why it is acceptable: the update
> multiplies the existing state by `1-gamma_0 = 0.08` at every `TCov`, so after `k` factor updates
> the pre-checkpoint history contributes `0.08^k` (`5.1e-4` at `k=3`, i.e. ~300 steps re-warm the
> operational state to better than `1e-3` **whatever the trajectory before it was**). Two
> corrections to v0 worth keeping: its lot-0 exit criterion ("T1-T6 pass") was **unsatisfiable** —
> T1 needs a dense `F`, T3-T5 need K-FAC, none of which lot 0 builds — so lot 0 gained its own
> T0.1-T0.8 and T1-T6 moved to the lots that build their prerequisites; and the campaign's analysis
> jobs must **not** inherit the training jobs' `h100_1g.10gb` line, because regime A is
> `P_max = sqrt(B/24)` and at 10 GB that is 20 412, below `mlp_ln_mnist`'s own `P = 26 634`.
>
> **Lot 0 is implemented** (`docs/reports/plan_exp_lot0.md`): the new top-level `fisher_ref/`
> package — a *reader*, which trains nothing and changed **no** file in `benchmarks/` or `src/` —
> with `conventions.py`, `probes.py`, `registry.py` and the `checkpoints.py` bridge, plus
> `tests/test_fisher_ref_lot0.py` (27 tests, 261 -> 288 passing, no pre-existing test modified).
> Two findings, both measured: (1) an **absolute** tolerance for the per-sample-independence check
> is wrong — `cnn_gn_cifar` (GroupNorm, eval, reloaded from its own 50% checkpoint) differs by
> `6.3e-6` between a batch of 256 and a batch of 4 purely from fp32 reduction order, so the check
> is relative with a dtype-keyed default, and the margin it lives in is five orders of magnitude
> wide (`5e-7` eval vs `3.6e-2` for `resnet20_cifar` with BN in **train** mode — the case that
> makes `F` and `E_hat` undefined at all); (2) the bridge discovers **42 runs** today (6 models x 7
> arms, seed 0), all post-denominator-fix, and `available_seeds()` returns `[0]` everywhere — the
> honest answer while `train_v0_seeds.sh` is in flight, which is exactly what every consumer must
> report rather than assume the plan's 3-5 seeds.
>
> **Re-warm fidelity, measured — and the answer to "should we re-run to save the optimizer state?"
> is no.** P2 needs the optimizer's internal state at a checkpoint's θ, but the checkpoints hold θ
> only. Measured on `cnn_gn_cifar`, all five modes (`fisher_ref/experiments/rewarm_fidelity.py`,
> numbers in `plan_exp_draft.md` §3.2): a re-warm of **10·TCov steps** reproduces the *applied*
> preconditioner to within **1.3-4.5× the estimator's own batch-draw noise floor**, and `ekfac`'s
> `s*` lands *below* its floor. Re-running the 6 models x 7 arms would cost only 87 min of training
> and 1.8 GB — but it would **replace** campaign 1 rather than augment it (under WCT only the
> reference arm is bit-reproducible; the others' step counts follow a *measured* budget), to
> preserve a state that is 92 % one minibatch's factor. **One correction to the plan's own
> arithmetic came out of it**: the re-warm length is set by `0.08^k << lambda`, **not** by
> `0.08^k << 1`. Every mode seeds its EMA with the **identity** at step 0 (`kfac.py:77` and
> analogues), so a fresh optimizer carries `0.08^k I` — a *spurious extra damping*. At `k=3`
> (the 300 steps the `0.08^k` arithmetic alone suggested) that is `5.1e-4`, half of `lam=1e-3`, and
> the applied preconditioner is **12-87 % wrong** on the four Kronecker modes (`tkfac`/`tekfac`'s
> un-normalized numerators are off by **six orders of magnitude**); by `k=10` the gap has collapsed
> by 3-6 orders. Do not shorten a P2 re-warm below `10·TCov`, and lengthen it when sweeping `lam`
> downwards. `--checkpoint-optimizer-state` (opt-in, OFF by default) exists for the *next* campaign
> that runs anyway, not for a re-run.
>
> **Lot 1 of the campaign: code done, the A1 cluster run pending** (`docs/reports/plan_exp_lot1.md`).
> The first curvature matrices: `fisher_ref/sources.py` (the three output-space roots, CE and MSE),
> `fisher_ref/capture.py` (per-example `(a, g)` and the recomputed `x_hat`, per-sample gradients for
> `Linear` shared and unshared, `Conv2d`, `BatchNorm2d`-eval, `LayerNorm`, `GroupNorm`) and
> `fisher_ref/reference/dense.py` (regime A: `F = U^T U`, `E_hat`, `B_l` in fp64, `U` streamed), plus
> `tests/test_fisher_ref_lot1.py` (44 tests, **297 -> 341** passing, no pre-existing test modified;
> `ruff`/`mypy` clean). T1 `3.8e-16` against an independent `jvp -> Λ -> vjp` oracle (T1 asks
> `1e-10`), T2's `O(K^{-1/2})` over 5 seeds and `K` to `1e4`, T6 to `1e-6` against central finite
> differences on all three norm types. **Three findings, each measured.** (1) A **silent** 94%-of-`P`
> hole in the obvious design: `register_full_backward_hook` fires from the module's *input*-side
> node, so `autograd.grad(out, params, ...)` with a `requires_grad` input **prunes the first module
> out** and its hook never fires — on `mlp_ln_mnist` that is the first `Linear`, 25 120 of 26 634
> parameters, whose `U` columns would simply be zero with `F` still symmetric, PSD and every
> exactness test still green. `capture.py` therefore uses a *tensor* hook on the module output and
> the driver backpropagates with `inputs=[batch]`; both halves are locked by a regression test.
> (2) The type-2 `F` of a classification **head** is rank-deficient at *any* `N`, by exactly
> `d_in + 1`: adding a constant to every logit leaves the softmax unchanged, so `W += 1_C v^T`,
> `b += c·1_C` is in `ker F` — verified at `9.1e-18` against `9.1e-2` for a random direction, and
> measured as 297/330 on the real A1 head. Unlike regime B's `m < P` deficiency this one does not go
> away, so every damped-inverse metric on a head block reads `λI` there. (3) `curvlinops` is
> **not adopted** (`plan_exp_draft.md` §2.4 is updated): absent from the venv and from the cluster's
> `--no-index` list, and unnecessary, since T1's own oracle shares no code with the build it checks.
> On the real `mlp_ln_mnist/diag/ckpt_0.5` with 256 train probes, `||E_hat - F||_F / ||F||_F = 1.42`
> on the head + LayerNorm blocks — the source error is not a correction term. **Phase 5 is written
> but not submitted**: `fisher_ref/experiments/dense_reference_a1.py` + `fisher_ref/slurm/
> dense_reference_a1.sh`, `N = 4000`. Sizing it produced a fourth measured finding — `plan_exp_draft.md`
> §2.2's `P_max = sqrt(B/24)` counts **three** `P x P` buffers, which is what the obvious spellings
> (`M += rows.T @ rows`, `M = 0.5*(M + M.T)`) allocate and what the lot's first draft had. With
> `addmm_` and a block-wise in-place `symmetrize_` the build needs **one**, measured at `1.27 x P^2`
> against `+1.16 x P^2` for that symmetrisation alone — so A1 fits the **same `h100_1g.10gb` slice
> the training jobs use** (~6 GB of 10), and §12's "an analysis job cannot allocate `F`" risk
> becomes a rule instead: hold one `P x P` on the device, put everything needing two on the host.
>
> **The A1 run is done** — job 21082966, `h100_1g.10gb`, `N = 55 000` (the whole MNIST train pool),
> `COMPLETED` in `01:06:33`, MaxRSS 25.3 GB, peak device **6.06 GB** against the 6.0 predicted
> (`plan_exp_lot1.md` §6). A first attempt (21077038, `N=4000`, 00:50:00) was cancelled in the
> *per-block* loop having produced everything else and **written nothing** — its only write was at
> the end; now incremental, cheapest-block-first, `A1_BLOCK_SPECTRA=0` to skip.
> **Budget these jobs from the spectrum, not the references:** a full type-2 `F` over 55 000 probes
> costs **147 s** (linear in `N`), one `26 634^2` `eigvalsh` costs **1280 s** and does **not**
> thread (19.8 GFLOP/s at 1 thread, 18.6 at 8, 21.6 on 16 cluster cores) — never ask for cores to
> speed up a spectrum. **Results:** noise floor `0.2986` on halves of 27 500, i.e.
> `sigma_N = 0.1493`; **source gap `0.3751` = 2.51 sigma_N, so Q1 is interpretable** (it was 0.76
> and 1.74 sigma at `N=4000` — most of that was noise). **HF1 is not settled**: train/val `0.7121`
> at ~1.4x its null, and the new val/test check says `0.7111` = 1.02x *its* null (indistinguishable)
> while train/test `1.0222` is 1.44x train/val — a tension the recorded data **cannot** resolve,
> because the three gaps use two denominators and only the train-side norms are stored. Lot 2's
> first job on HF1 is to record `||.||_F` for every reference and re-read all three against a common
> denominator; until then do not quote val/test as licence to pool the two held-out sets.
> Also measured: `sigma_N` fell 2.92x for 13.75x the probes where `N^{-1/2}` predicts 3.71x, so the
> floor decays *more slowly* than the ideal rate and MNIST has no probes left to give; `rank(F) =
> 21 829/26 634` and `rank(E_hat) = 18 342/26 634` — at `N > P` the empirical Fisher is no longer
> rank-limited by `N`, and is **more** deficient than `F` (Kunstner et al. arXiv:1905.12558, sharper
> than the first run could show). Per block, every deficiency is structural and two were predicted:
> the **head's is exactly `33 = d_in + 1`** (the logit-shift kernel of §4 — adding a constant to
> every logit leaves the softmax unchanged), and `features.0`'s **4 579 ~ 32 x 143** is MNIST's dead
> pixels (that layer's input factor is rank 646/785, `_eigh_utils.py` — the fact that crashed
> cuSOLVER in campaign 1). `features.0` also carries **63% of `tr(F)`**: A1's curvature is
> overwhelmingly in its first layer, which conditions any per-layer-type reading of this model.

> **Lot 3 of the campaign (regime A with weight sharing): code done, cluster runs submitted**
> (`docs/reports/plan_exp_lot3.md`). A1 has `T = 1` on every layer, so every K-FAC branch written
> for shared weights had never been *run* — and three of them were wrong, each measured rather than
> suspected. (1) **K-FAC-reduce was `T²` too large**: it summed the input over positions where
> Eschenhagen et al. (arXiv:2311.00636 §3.3, Eq. 10) average it — relative error **24.000** at
> `T = 5` and **80.000** at `T = 9`, i.e. exactly `T² − 1`, and biased in the direction that favours
> expand, which is HF2's own question. (2) **EKFAC-reduce projected a statistic that is not a
> gradient** (`(Σ_t a)(Σ_t g)ᵀ` instead of the per-example `Σ_t g_t a_tᵀ`), so its `s` was not the
> optimal diagonal and `tr(K) = tr(B)` failed. (3) **TKFAC at `T > 1`** used per-*example* summed
> traces, whose product carries every cross-position pair and matches neither `tr(B)` nor
> `tr(B^exp)` nor `adafisher_modes`' own; it is now the per-`(n, c, t)` flattening the optimizer
> uses, and preserves `tr(B^exp)` exactly. **T5** (expand exact in the expand setting, reduce in the
> reduce setting, and each *not* exact in the other), **T8-exp**, **T9-shared** and **T13-shared**
> pin all three. T13-shared also measures a declared constant worth carrying into P2: fed identical
> gradients, the optimizer's Kronecker operator on a shared layer is **K-FAC-expand divided by `T`**
> (`A` agrees exactly, `G = T · compute_s_full`), so a fixed `λ` weighs `T` times more there — 64 on
> a ViT token layer, 1 024 on A2's first convolution.
> **Two lot-2 readings are corrected** (erratum added to `plan_exp_lot2.md` §6.3 and its French
> translation): its `as_implemented` normalisation structure is `diag(hadamard)`, **not** `diag.py`'s
> formula (which sums the *raw* input over batch and positions before squaring, and yields a
> 2-entry `H_D` shared by all channels) — lot 3 renames it `hadamard_diag` and adds the shipped
> reading as `diag_py`, computed by **calling** `compute_h_diag`/`compute_s_diag` on training-size
> micro-batches; and three of §6.3's four orderings are **theorems** (the exact diagonal is the
> Frobenius-optimal diagonal; `exact_separate` is optimal among matrices with no `γ`-`β` cross
> terms), now asserted in the tests and checked at runtime, so a violation is a bug rather than a
> finding.
> **A3's `pos_embed` (2 048 of 21 098 parameters) was silently outside lot 2's reference**, whose
> runner passed `modules=...` and labelled the restricted result `F`. It is now captured by
> substituting a batch-expanded leaf for the traversal (and listing it in `inputs=`, or the engine
> prunes it — the lot-1 pruning trap in a new place), and **every reference build now checks its own
> rows against autograd twice**: the sum rule, and one example against a one-example backward (the
> second catches what the first cannot, a statistic that mixes examples). Measured on the three real
> checkpoints: `≤ 5.3e-16`, GroupNorm, BatchNorm-eval, stride-4 patch embedding and `pos_embed`
> included. New modules: `fisher_ref/folds.py` (one traversal, `K` folds offloaded to the host,
> halves assembled by addition — per-layer intervals *and* the whole-matrix floor for no extra
> traversal, where lot 2 rebuilt forty references for the floor alone), `approx/sharing.py` (the
> `B → B^exp → K-FAC` decomposition, held as a half-vectorised rearrangement so nothing `P × P` is
> formed), `approx/embed.py`, and `experiments/lot3_decisions.py`, which applies §0.10's
> **pre-registered** decision rules. `METRICS_VERSION` is `fisher_ref/0.2`. When lot 3 closed the suite was
> **609 passing** — the 579 that existed, unmodified, plus its 30 — and it has grown since with work
> outside this lot.
>
> **The three runs are done** (`21276896` A3 `01:02:33`; `21276894` A2-GN `04:30:41`; `21276895`
> A2-BN `04:20:50`; `N = 45 000`, ten folds, 20 partitions, five checkpoints, both sources, host
> peaks 54-81 GB, **zero** `invariant_violation` rows, row checks `≤ 5.3e-16` throughout), and the
> pre-registered rules give **HF2 confirmed on all three models** (`plan_exp_lot3.md` §6). Four
> results to carry:
> 1. **K-FAC-reduce beats K-FAC-expand almost everywhere**: 14/15, 15/15 and 40/45 cells, with
>    reduce's wins at `Δe_F = −0.08 … −0.94` against expand's small, early-checkpoint wins
>    (`+0.02 … +0.18`, plus the ViT patch embedding at `θ = 0`, `+0.78`). The whole-matrix noise
>    floor at this `N` is 0.005-0.039, so these are not noise (A1's floor was 0.343).
> 2. **K-FAC's error on a shared layer is first of all a *weight-sharing* error, not the
>    independence assumption**: the median `sharing_share` is **0.971 on all three models** —
>    dropping the cross-position terms alone already costs ~97 % of the exact block — while the
>    independence step on top costs 0.45-0.60. At initialisation the block is *nearly* Kronecker
>    (`best_kron` 0.05 on the ViT's `qkv`, 0.22-0.37 on A2's convolutions) while K-FAC-expand sits
>    at 0.95-0.97: the Kronecker form is not what fails, K-FAC's choice of factors is.
> 3. **`e_F` and `ρ` disagree systematically on that same question**: `ρ(reduce) < ρ(expand)` in
>    70/75, 62/75 and 162/225 cells. Reduce is closer in norm and usually *worse* for the step it
>    would produce, so the expand/reduce choice **for an optimizer** must be read on `ρ`, not `e_F`
>    (which is why §0.10 votes on `e_F` and reports `ρ` across the sweep rather than mixing them).
> 4. **Proposition 3.1 under weight sharing: right direction, 20-38× too small.** The Hadamard
>    reading's optimal rescaling is 0.95 on A1 (no sharing) but 27.8 / 20.4 / 37.9 on A2-GN / A2-BN /
>    A3, because it is built from expand-style statistics and therefore approximates `B^exp`, not the
>    exact `(γ, β)` block. Direction-wise the ranking is identical on all four models. And the
>    shipped `diag_py` reading is last by direction in 13/15 (A2-BN) and 15/25 (A3) cells — the claim
>    lot 2 §6.3 made about the wrong object now holds, measured, for the right one.
>
> Two of lot 2's own Q4 conclusions **do not transport** and are recorded as failing: "`ρ` falls for
> every structure along training" is a tendency (95 % of pairs on A1, but 52-71 % here), and
> "inter-layer coupling ≥ 0.5 everywhere" fails on the ViT (min 0.28) and on A2-GN (0.44) — layers
> are *less* entangled where weights are shared, the opposite of what an MLP-only study suggests.

> **Lot 5 of the campaign — P2, the operational protocol: DONE, all six jobs COMPLETED**
> (`docs/reports/plan_exp_lot5.md` §6; jobs **21388028** fidelity `00:28`, **21388029** smoke
> `00:42`, **21388030/31/32/33** for A3 `01:58` / A2-GN `06:58` / A2-BN `07:25` / A1 `02:04`, the
> last four chained `afterok` to the smoke). 181 574 rows, **zero** invariant violations, **zero**
> non-finite values, row checks against autograd `≤ 1.04e-15`, host peaks 77.9-81.6 GB against the
> 80.7 predicted.
>
> **The result, in one line: the preconditioner the optimizer actually divides by is the identity.**
> `cond(F~)` is **1.0000-1.1035** over 25 (mode, block) pairs and `lambda` is **98.9-100.0 %** of
> its mean eigenvalue; over 7 750
> (mode, block, checkpoint, source, damping) cells `rho(P2 mode)` equals `rho(identity)` — the
> plain-gradient step — to a median of one part in `10^4` to `10^6` (p95 `<= 1.2 %`). The same
> structures in their P1 form deliver **0.58-0.88** of the ideal quadratic decrease where the
> operational ones deliver **0.18-0.42**, which is what doing nothing delivers. This is
> `plan_lambda_dominance.md` seen from the other end: that report measured the *state* sitting
> `~2e8` below what it estimates, lot 5 measures the *consequence* for the operator and for the step.
>
> **And the damping is what does it, not the averaging** — the separation §11's decision rule does
> not make. Before the optimizer's own `lambda` the operational operator still carries **0.01-0.43**
> of structure (after the EMA, the min-max, train mode, the augmentation, fp32 and an effective
> sample of ~150 examples); adding `lambda` takes it to `1e-8`-`5e-4`, i.e. **139x to 5.8 Mx closer
> to the identity**. The four-rung ladder (P1 -> P1-py -> P2-raw -> P2, with a one-micro-batch
> control) attributes the rest: the optimizer's own **formulas cost nothing** directionally
> (`kron_py` = `kfac` to three decimals); the **effective sample size** costs 0.275 on A1's
> `785 x 785` input factor but only 0.001-0.013 on the CIFAR convolutions, because 128 examples are
> 128 x 1024 *patches* — **weight sharing is what makes the operational estimator viable at all**.
>
> **HF7 is "confirmed" by the pre-registered rule and the honest reading is different, and
> stronger.** Median `tau(P1,P2)` is `-0.67 / -0.33 / -0.33 / -0.33` with a reproducibility ceiling
> of `+0.67`-`+1.00` and **0 degenerate, 0 dropped** cells — the gate §0.9 was built around had a
> real chance to fire and did not. But the spread between the five modes at the operational rung is
> **0.0000-0.0003** against a per-layer noise floor of **0.026-0.167**, i.e. **100-1000x below it**,
> while at the structural rung it is 2-16x *above*. By `plan_exp_draft.md` §10.3's own reading rule
> that difference is not reportable: **there is no P2 ranking, so HF7 as worded is ill-posed at the
> operational point.** §0.9's gate used the operator's own replica noise (`1e-8`) as its yardstick,
> which is too permissive; §10.3's floor is the binding one and the verdict is reported under it.
>
> **The one mode that survives its own damping is the one whose `lambda` is relative.** Ranked by
> departure from doing nothing: `ekfac` `1.3e-7`, `tekfac` `1.5e-7`, `kfac` `1.8e-5`, `diag`
> `2.9e-5`, **`tkfac` `6.8e-5`** — 500x further than `ekfac`, growing along training, and holding the
> single largest departure of all 7 750 cells (`+11.6 %` over the plain gradient on A1's
> `features.0` at initialisation). `tkfac` is the only mode damping with `sqrt(lambda/delta)` against
> `tr(Phi) = tr(Psi) = 1`, i.e. **relative to the curvature's own trace**; every other adds an
> absolute `lambda`. That is quantitative support for fix **S1** of `plan_lambda_dominance.md`,
> arriving from a new direction — and it is corroborated independently by the condition numbers,
> whose per-mode ranges reproduce the same ordering: `ekfac`/`tekfac` 1.0000-1.0013, `kfac`
> 1.0003-1.0206, `diag` 1.0001-1.0568, **`tkfac` 1.0010-1.1035**.
>
> **Exit criterion 5, the re-warm, is confirmed on all four models**: at `k = 10` the gap on the
> applied preconditioner is within **0.37-1.82x** its own batch-draw floor on all 20 (model, mode)
> pairs and `k = 20` changes nothing; at `k = 3` the four Kronecker modes sit **4-8 x 10^5** above
> their floor with the applied preconditioner **52-87 % wrong**, because the floor *collapses* when
> two replicas share the same deterministic identity seed. **Exit criterion 8** (the free regression
> test on reusing the P1 runner) holds on type-2 to `3.47e-13` over 33 065 rows with zero above
> `1e-10`; HF7's own P1 structures to `<= 5.5e-10` and the verdict statistic to `<= 1.5e-13`.
>
> **A finding that belongs to lot 3, found chasing the one exception.** On the *empirical* source
> `ekfac_reduce`'s inverse metrics differ from lot 3's by up to `0.376`. Cause: the reduce output
> factor `G_reduce` is **exactly rank-deficient** on any layer followed by a normalisation —
> `rank = channels - num_groups`, measured **8 of 16** on `cnn_gn_cifar`'s `features.0` and 24 of 32
> on `features.4` — because a normalisation's backward makes the gradient sum to zero within each
> group and `reduce` sums over exactly those positions. Inside that degenerate null space the
> eigenvectors are numerically arbitrary: K-FAC is invariant (constant eigenvalue there), EKFAC is
> not (it assigns a data-estimated `s`). Measured: a `3.2e-16` perturbation rotates the null-space
> eigenvectors by **0.53**, moves `kfac_reduce`'s step by `1.2e-12` and **`ekfac_reduce`'s by
> `0.144`**. **HF2's verdict stands** (it votes on type-2 `e_F`, no inverse), but no `rho`, `cos_ngd`
> or Stein-KL number for `ekfac_reduce` should be quoted to better than ~10 %.
>
> **What lot 5 does not establish**: one arm (`diag`) and one seed; regime A only, so §10.3's
> regime-B clause is unmet and every per-layer-type reading is provisional until lot 4; and `lambda`
> is not swept on the P2 side by construction. Read `plan_exp_lot5.md` §6.7 before extending any of
> it. P1 asks how good a family of approximations can
> possibly be; P2 asks how good the thing the optimizer actually divides by is. The new
> `fisher_ref/approx/adafisher_state.py` reads `AdaFisherMulti`'s live per-module state — after the
> running average, after `diag`'s min-max, at the optimizer's own `lambda`, estimated in fp32 from
> augmented training batches with normalisation layers in train mode — and `fisher_ref/rewarm.py`
> rebuilds that state from a checkpoint's weights, since the checkpoints hold weights only.
> **It is not on B1/B2**: `cct_2_3x2_cifar` and `resnet20_cifar` are regime-B models, so both halves
> of the comparison need lot 4, which does not exist; lot 5 runs on **every regime-A model instead**
> (A1 `mlp_ln_mnist`, A2-GN `cnn_gn_cifar`, A2-BN `cnn_gn_cifar_bn`, A3 `vit_micro_cifar`), which is
> exactly where P1 has been measured (§0.0 there).
>
> **The design decision that makes it cheap and paired**: `p2_operational.py` is the *P1* runner
> with extra structures injected (`ExtraStructures`), not a second runner — so P1 and P2 come from
> **one** reference build, on the same probes at the same weights. Checked directly rather than
> argued: on a real `cnn_gn_cifar/diag/ckpt_0.5`, **340 of 340 P1 rows are bit-identical** between a
> pure-P1 run and the P2 run, and the only extra structure is the `tekfac` the P2 job enables on
> purpose. 657 tests pass (the 628 that existed, unmodified, plus lot 5's 29).
>
> **Six things the plan's own adversarial re-read caught before any code was written**, each of
> which changed the design (§0.14 there): there was no like-for-like P1 partner for `p2_diag`;
> **P1 builds no Kronecker structure on a normalisation layer at all** (its `A` there is the
> `(C+1)x(C+1)` second moment of the *normalised* input, the wrong size for a `2C` block), so three
> of HF7's five mode pairs had nothing to compare against; the P2 operator's **own** sampling noise
> was nowhere (fold intervals carry the *reference's*, and the operator is a draw of something that
> is 92 % one minibatch); the augmentation and train-mode term was silently folded into "the EMA";
> M3 would have tripled the job for a metric not in the verdict; and there was no sizing table.
> The fix is a **four-rung ladder** — P1 (structure alone, fp64, per-example) -> P1-py (the
> optimizer's *own formulas*, called not re-implemented, still without averaging or damping) ->
> P2-raw (after the running average, min-max, fp32 and the training data) -> P2 (plus the
> optimizer's damping) — so the gap is attributed rather than lumped.
>
> **Four measured findings during implementation**, each recorded in §5 there. (1) The premise was
> verified first: solving `F~ x = rvec(M)` densely reproduces `approx.precondition` to **0.0-2.2e-15**
> in all twenty (mode, layer) combinations, so no new block class was needed. (2) `diag_py`
> **averages the product** `outer(S_D, H_D)` over micro-batches while the optimizer keeps one
> running average per **factor**; lot 3's function is left untouched and `diag_py_factor_reading` is
> added beside it, so the gap is a number in the CSV rather than a choice. (3) On a real block the
> P1-py readings sit at **`c* = 1.6e4`** against the exact Fisher — that is `CLAUDE.md` §4.3's
> `1/batch^2 = 16 384` at batch 128, confirmed end to end on a trained network for the first time;
> its consequence is that `rho` on a scale-free reading must be evaluated on `c*K`, which lot 3 did
> only for a `Diag`. (4) **At the correct re-warm length the operational preconditioner is,
> numerically, a multiple of the identity**: `cond(F~)` on `cnn_gn_cifar`'s head is 1.00-1.11 for
> `ekfac`/`tkfac`/`tekfac`, 1.02 for `kfac`, 1.05 for `diag`, and `lambda` is **37x** the mean
> eigenvalue of the exact Fisher there. That is `plan_lambda_dominance.md` seen from the other side,
> and it means `rho(P2) ~= rho(identity)` in the real runs will be a measurement, not a bug. One
> layer, one checkpoint, one model — the four jobs are what settle it.
>
> **Two traps this lot had to avoid, both of a kind this repository has paid for before.** A
> normalisation layer's block is **interleaved** in the optimizer (`gamma_0, beta_0, gamma_1, ...`)
> and **blocked** in the campaign (`[gamma; beta]`), so a block that skips the permutation is
> symmetric, positive definite and silently wrong — `plan_exp_lot2.md` §5.2's bug in a new place,
> pinned by a test that fails when the permutation is removed. And `conv_sua=True` is **refused**
> rather than answered: under SUA the operator is not one `kron(B~, A~)` over the full patch
> direction but the same small operator applied at each kernel offset (`plan_lot6.md` §0.4).

> **The authors' own code, on the bench that stalls and on two published CIFAR-10 numbers: both
> steps done** (`docs/reports/plan_authors_repro.md`). Every number behind this
> file's "the primary bench is the only failure" paragraph came from this repository's port, so a
> porting mistake was not excluded. It is now. The optimizer of the authors' **published**
> repository (`reference_repos/AdaFisher/optimizers/AdaFisher.py`, unmodified, the new **`official`
> arm** of `benchmarks/common/optimizers.py`) freezes on the MNIST auto-encoder exactly where
> `diag` does: over a full 20-epoch run of 2 200 steps the two trajectories differ by at most
> **2.58e-7 relative** — the few bits the fused `addcdiv_` costs (§2.1), and nothing else — both
> ending at train loss **0.70828** against `adam`'s **0.47194**. FisherAdapTune's `AdaFisher` (the
> `reference` arm, min-max off) stalls at the same 0.70828, 1.3e-3 away. **The stall is AdaFisher's,
> not the port's.** Two measurements sharpen it. (1) **A learning-rate bracket is ruled out, as the
> `lam` bracket already was**: the authors' code ends within **7e-6 of 0.70828 at `lr` in
> {1e-4, 1e-3, 1e-2, 1e-1}** — four different journeys (the `1e-4` and `1e-3` trajectories are up to
> 0.28 apart mid-run) to one destination. Do not re-run it. (2) **0.70828 is the network's plateau,
> not a number AdaFisher invents**: `adam` lands on it too, at `lr = 1e-2` (0.70828) and `1e-1`
> (0.70850), and escapes only at `1e-3` (0.4719) and `1e-4` (0.6090). So the honest statement is that
> Adam has a window of learning rates that leaves this all-sigmoid plateau and AdaFisher, at this
> damping, has none among those tried — which is what `plan_lambda_dominance.md` predicts, and
> consistent with the already-measured parameter travel (~1.0 against Adam's 25.8), i.e. steps far
> too small. **Step 2, the positive control, is in, and both published rows come back**: the authors'
> own `train.py` on their own `AdaFisherCNN.yaml`/`adamCNN.yaml` **verbatim** (jobs 21449869/21449870,
> COMPLETED in `01:04:52`/`01:04:33`, seed 42, their pinned `torch 2.3.0`/`torchvision
> 0.18.0`/`numpy 1.26.4`, all three in the wheelhouse) gives Table 2's CIFAR-10 ResNet18 best test
> top-1 at **96.32 %** against the published **96.25 ± 0.2** (+0.07, i.e. 0.35 of its std) and, for
> Adam, **94.78 %** against **94.85 ± 0.1** (−0.07, 0.70 of its std) — the published gap of 1.40
> points reproducing as **1.54**. So the negative result above is not a broken setup talking: the same
> code on the same machine reproduces a published number on CIFAR-10 and stalls on the auto-encoder.
> Three measurements worth keeping from it: **Table 8's 200-vs-210 epoch split really is the
> equal-wall-clock protocol** (the two runs took 3 834.5 s and 3 818.3 s, **0.4 % apart**);
> AdaFisher's per-epoch overhead over Adam on ResNet18 is **5.3 %** (19.15 s against 18.17 s), which
> is `diag` being the cheap mode, next to the four Kronecker modes' 1-2x of fwd+bwd
> (`plan_lot7.md` §6); and **AdaFisher reaches Adam's best accuracy at epoch 158**, 50.5 min into the
> 63.6 min Adam's whole run takes, while Adam never reaches AdaFisher's best — the paper's convergence
> claim, not only its final number. What is **still not established** is that AdaFisher stalls on
> anything between those two settings: a convolutional net with BatchNorm, augmentation, batch 256 and
> a 200-epoch cosine works; eight sigmoid `Linear` layers at batch 500 for 20 epochs do not, and which
> of those differences is the discriminating one is untested. Three things that cost a job or would
> have: `reference_repos/` is gitignored and the cluster checkout held **0 of its 73 files** inside a
> correct directory skeleton, so the job's `diff -r` integrity check passed on two empty trees (it now
> counts files too); `numpy 2` breaks the authors' code before it trains anything (`np.Inf` in their
> `early_stop.py`); and their `train.py` imports `asdl` unconditionally for its Shampoo/K-FAC arms
> only, which `benchmarks/authors_repro/asdl_stub/` satisfies and nothing on an AdaFisher or Adam
> path can reach — **the single deviation in the whole reproduction**.

> **The seed axis completed to all seven arms: done, and it changes a reading.** The extra-seed
> table (`train_v0_seeds.sh`) only ever ran **two** arms, `diag` and `adamw`, because
> `plan_exp_draft_v0.md` §3.5 samples two trajectories for the *drift* question. That left a
> different hole, measured from the bridge before anything was launched: `kfac`, `ekfac`, `tkfac`,
> `tekfac` and `adam` existed at **seed 0 and nowhere else, on every model**, so every statement
> comparing the five modes *to each other* rested on one draw. Seven new jobs
> (`benchmarks/slurm/train_seeds_<model>.sh`, from `SEED_COMPLETION` in `generate_jobs.py`) fill it
> at the seeds the plan asks for — regime A 1-4, regime B 1-2 — and add the two models
> `V0_SEEDS` never covered at all: `cnn_gn_cifar_bn` (one of lot 5's four regime-A models) and
> `mnist_autoencoder` (the primary bench, the one carrying the stall finding). All seven
> `COMPLETED`, exit 0, 9 min to 1 h 31 each. The tree is now **24 (model, seed) runs x 7 arms x 5
> fractions = 840 checkpoints**, every cell filled; `available_seeds()` returns `[0,1,2,3,4]` on the
> five regime-A models and `[0,1,2]` on the two regime-B ones **for all seven arms**, and
> `discover_runs()` finds **217** runs against 81 before.
>
> **The re-run of `diag`/`adamw` was checked, not assumed: 158 of 160 pre-existing checkpoints are
> bit-identical.** The two exceptions are both `cnn_gn_cifar/adamw/ckpt_1.pt`, at seeds 1 and 3, and
> the cause is the WCT protocol rather than any non-determinism. `diag` is the *reference* arm, so it
> is unbudgeted and ran **exactly 10 530 steps in all 8 cases**, bit-identical both times. `adamw` is
> budgeted by measured time: 11 317 / 11 013 / 11 683 / 11 314 steps in the old campaign against
> 10 490 / 10 576 / 10 272 / 10 618 in the new one — **7-12 % fewer steps for a budget only 1-4 %
> smaller** (46.4 s -> 45.3 s and so on), i.e. the node was busier and each step cost more.
> `ckpt_1.pt` is written at the nominal step 10 530 when the arm reaches it and pinned to the arm's
> own end otherwise: the old runs passed 10 530 at all four seeds, the new ones only at seeds 2 and 4
> (10 576, 10 618) and fell short at seeds 1 and 3 (10 490, 10 272). Exactly the two that differ.
> **The rule to carry: under WCT only the reference arm is reproducible across campaigns; a budgeted
> arm's step count follows node contention, so two campaigns are not comparable step-for-step there.**
>
> **The finding one seed could not reach: the five modes split into two groups, and the split is
> significant.** Pairing is exact — within a `(model, seed)` cell every arm shares the initialisation
> and the data order — so the comparison is an exact paired sign test over the **24** cells. Mean
> rank of 5: `kfac` **2.08**, `diag` **2.42**, `tkfac` **2.42**, `tekfac` **3.92**, `ekfac` **4.17**.
> Inside `{diag, kfac, tkfac}` no pair separates (p >= 0.31); inside `{ekfac, tekfac}` neither
> (p = 0.84); **all six cross-group pairs do** — `kfac > ekfac` 22/24 (p = 4e-5), `kfac > tekfac`
> 21/24 (p = 2.8e-4), `tkfac > ekfac` 22/24 (p = 4e-5), `diag > ekfac`, `diag > tekfac` and
> `tkfac > tekfac` all 19/24 (p = 0.0066). Bottom-2 occupancy: `ekfac` **19/24**, `tekfac` **17/24**,
> against `kfac` 2/24, `tkfac` 4/24, `diag` 6/24.
>
> **And the mechanism is measured, not inferred: the two losing modes are the two that buy fewest
> steps.** Steps completed inside the *same* wall-clock budget, relative to `diag = 1.00`, averaged
> over the seven models: `diag` 1.00, `kfac` 0.93, `tkfac` 0.91, **`ekfac` 0.82, `tekfac` 0.82**
> (`adam`/`adamw` 1.14). The group boundary is the same one the sign test finds. This is lot 7's
> per-step cost measurement (`ekfac`/`tekfac` ~1.8x their own fwd+bwd in extra work, `kfac`/`tkfac`
> ~1.0x) showing up as a *convergence* result for the first time, and it lines up with lot 5 from the
> other side: `ekfac`/`tekfac` hold the operational preconditioner **closest to the identity**
> (departure 1.3e-7 and 1.5e-7, against `kfac` 1.8e-5, `diag` 2.9e-5, `tkfac` 6.8e-5). They pay the
> most per step and deviate from the plain gradient the least. **State it carefully:** this is a
> result at *equal wall-clock*, at one shared untuned operating point. It does **not** show that
> EKFAC/TEKFAC approximate the curvature worse — the deficit is confounded with step count, and only
> an equal-step comparison would separate the two. That is the comparison `plan.md` §6.3 warns is
> rigged in the other direction.
>
> **What is robust across every seed: the Fisher modes beat both baselines on six models of seven,
> and lose totally on the seventh.** Paired against the better of `adam`/`adamw`, per seed:
> `vit_micro_cifar` **+12.6 to +14.4** points, `cct_2_3x2_cifar` **+10.1 to +11.1**,
> `cnn_gn_cifar_bn` **+4.9 to +5.9**, `cnn_gn_cifar` **+1.9 to +3.0**, `resnet20_cifar`
> **+1.2 to +2.3**, `mlp_ln_mnist` **+0.18 to +0.49** — winning on every seed in 34 of the 35
> (mode, seed) cells those six models hold. The seventh is `mnist_autoencoder`, where all five modes
> lose by **0.210** in final validation loss on **0 of 4** seeds, and by the *same* 0.210 for every
> mode. **So the stall is confirmed on four fresh seeds and is model-specific, not a property of the
> modes**: the five sit at `0.7072-0.7074 +- 0.0014` while `adam`/`adamw` reach `0.4976 +- 0.0207`,
> and the spread between the modes (`0.00024`) is **5.6x smaller than the seed-to-seed spread**.
> Lot 7's "all five converge to an indistinguishable final loss" is now measured over seeds — and it
> remains the wrong thing to celebrate, because the shared point is a bad one.
>
> **Two caveats on the numbers above.** The regime-B models carry **2** seeds, not 4, so their own
> per-model spreads rest on one degree of freedom (the pooled sign test does not — it pairs within
> cells). And seeds 1-4 all ran `--lr-schedule nominal` while **seed 0 is not one protocol**
> (`mlp_ln_mnist` ran `budget`, the other four pre-date `schedules.py`), so every figure in this
> block is computed on seeds 1-4 only and seed 0 is deliberately excluded.

> **E16 — a floor or three clips instead of the added `lambda`: code done and audited, runs not
> submitted**
> (`docs/reports/plan_floor_clip.md`; pre-registered in `plan_lambda_dominance.md` §E16). The two
> alternatives `fr/etude_clipping_vs_damping.md` set aside because both degenerate at the shipped
> `lambda` -- family A `max(s, lambda)`, family B a Sophia-type clip of the undamped step -- are now
> `AdaFisherMulti(rescale_form="floor"|"clip")` for `ekfac`/`tekfac`, to be tested at the E14
> operating point (`lambda` ~ 1e-11-1e-10) where they can differ from `s + lambda`. The clip absorbs
> the curvature's scale error, so it is the one candidate that could transfer between networks where
> E14 found no single `lambda` does, and it comes with three thresholds (`clip_threshold`):
> `"quantile"` (reset every step: a per-module normalisation, kept as the control), `"ema"` (the main
> arm: the step follows the momentum's size against its bias-corrected average over 1 000 steps) and
> `"fixed"` (a conditional clip: each module's median quantile over steps 1000-1999, frozen at step
> 2000). A six-agent audit before submission found and fixed, each measured: a **network-wide**
> threshold is incoherent here (the stored curvature's scale error differs between layers by up to
> four orders of magnitude -- a pooled calibration clipped a ViT head on 3 % of its coordinates and
> its final LayerNorm on 97 %), so every threshold is per module; the EMA seeded with its first step
> stayed biased for 12-18 % of a run, so it is bias-corrected; the clipped count could be off by one
> through rounding, so clipped is now defined by the tensor the threshold comes from; the per-module
> host synchronisation and per-step statistics would have overrun the job limits, so the quantile is
> a sort-and-gather and the statistics are lazy; and the decision script used to read an incomplete
> or invalid set of jobs as "equivalent". **Third amendment, the normalisation statistic corrected**
> (`plan_floor_clip.md` §11). `ekfac`/`tekfac` estimated a normalisation layer's `s*`/`Theta` from
> the gradient of the §4.6 surrogate `[z*delta, delta]`, not the layer's own `[delta*x_hat, delta]`.
> Measured at step 300, the scale column was 295-989x too small on `vit_micro_cifar` and 80-106x on
> `cct_2_3x2_cifar`, and the clip exposed it at `q <= 0.3` by throttling the shift steps 3-117x. The
> new knob `norm_exact_rescaling` (§3) uses the layer's own per-row gradient, projected into the same
> eigenbasis, which EKFAC's Lemma 1 makes the optimal diagonal there. **Every E16 arm runs with it,
> including add**, so E16 reruns E14's fix instead of pairing with E14's stored cells. The rerun is
> needed: at E14's `lambda` the corrected statistic raises the divisor `s + lambda` of the first
> LayerNorm's scale column to a median 2.45x `lambda` on the ViT and 2.21x on the CCT (step 2 000,
> measured), where the shipped one leaves it at 1.00x. Seed 0 carries two gates: the same add cell
> run twice, in two processes, must agree exactly; and on `cnn_gn_cifar`, which has no hooked
> normalisation layer, the knob must change nothing. The comparison with E14's stored cell is a
> diagnostic that does not vote. One limit stays recorded, not fixed: TF32 convolutions on the H100
> are noisier than the `1e-3 x rms` guard. **90 shard jobs + 30 merges**
> (`fisher_ref/slurm/e16_submit.sh MODEL SEED MODE`): each (network, seed, mode)'s 34 cells (36 at
> seed 0) split into 3 shards, each one process on one `h100_1g.10gb` slice (E14's own hardware),
> then a CPU merge that runs the seed-0 checks. Largest shard projected at 22/49/58 min for
> cnn/vit/cct, limits 0:35/1:15/1:30. About 62 h of 1g-slice time. All thirty come from ONE clean
> commit: **submitted 2026-09-21 from the clone `/home/blgr/new_adafisher_e16` at `7663d62`** (the
> cnn seed-0 `ekfac` test, 21543912-15, plus 21546022-21546138). That clone is not pulled or
> modified until they have all run: a pending job reads the code when it starts. The ResNets run
> from a separate checkout at the fourth amendment's commit (one commit per network is enough). **Fourth amendment (`plan_floor_clip.md` §12):** two
> ResNets are added as *exploratory* networks, which vote in none of the pre-registered rules and
> feed nothing to E17. `resnet20_cifar` gets the full design at 5 seeds (grid 1e-10..1e-12 around
> E14's 1e-11, ~46 h of 1g, projected). `resnet50_cifar` is first calibrated
> (`e16_resnet50_calibration.py`: an add scan at seed 0 over 1e-8..3e-13 plus clip timings, 12
> jobs), then gets a reduced design at 3 seeds on the grid the pre-registered rule derives from the
> scan. **Calibration done** (21546221-32): both modes peak at `lambda = 3e-12` (90.52 / 90.14 %
> validation, seed 0; a 31-35-point cliff at 3e-13), so the grid is [3e-11, 1e-11, 3e-12, 1e-12,
> 3e-13]; measured 2 486 s per add run and 3 320-3 735 s per clip run on a 1g slice; 5 shards per
> (seed, mode), largest 2.69 h, ~73 h of 1g in total (`plan_floor_clip.md` §12.1). A clip arm is "robust" on a ResNet if it loses to add in neither mode. The feasibility
> study's Sophia range was inverted:
> `win_rate` is the fraction *not* clipped, so the published target is a clipped fraction of 0.5-0.9.

> **E15 — one safety constant per layer: done, and adopted** (`plan_lambda_dominance.md`, "E15 —
> done"; jobs 21523753-62, all COMPLETED). Five seeds on `cnn_gn_cifar` and `vit_micro_cifar`, under
> the E protocol (batch 32, 15 epochs, step-size cap held per layer). All six reproduction cells hit
> E13's and E14's seed-0 accuracy to **0.00 points**. `damping="layer_relative"` (S1-b) beats the
> best single `lambda` in **all six** (network, mode) pairs, by +1.9 to +6.1 points, and in 30 of 30
> (pair, seed) cells. It also beats the network-wide relative `lambda` in all six, by +1.7 to +4.3,
> so the gain comes from treating layers separately, not from being relative. Against the default
> `lambda` it gains +8.2 to +14.8 points. Rule 3: **adopted**. Rule 4 as written: **not**
> tuning-free, because `kfac` wants `tau` = 0.3 to 3 while `ekfac`/`tekfac` want `tau = 0.1` on both
> networks. What S1-b does, read from the per-layer logs: it gives the head a constant hundreds of
> times above the single optimum, and the flattest layers one 2 to 20 times below it. A second
> session recomputed every number from the raw JSON with its own script, and all of them match.
> Limits: two networks of about 20 k parameters, at batch 32; nothing yet under the benchmark
> protocol, and no third network.

> **E17/E18 — ViT-S, the one network where the shipped Fisher arms lose to AdamW**
> (`plan_lambda_dominance.md` §E17, §E18). **E17 is pre-registered and amended, and not launchable
> yet.** It makes ViT-S (`vit_small_cifar`, then `vit_small_cifar100`) the held-out network for E15's
> and E16's verdicts. No experiment of the λ work has ever run it. The candidates are chosen by E15's
> and E16's own rules. Stage 1 asks whether the chosen value transfers: E protocol, five seeds, and an
> oracle sweep to locate ViT-S's own plateau. Stage 2 asks whether it closes the campaign-2 gap to
> AdamW: three seeds, one job per (dataset, seed). From E15, candidate C (S1-b) **does not qualify**,
> because rule 4 read literally fails; that reading is the user's decision. S1-b at `tau = 0.1` may
> run there only as an arm labelled exploratory. E16's candidates must run with
> `norm_exact_rescaling=True`, as E16 does. E17 is blocked on four things: E16's results;
> `vit_small_cifar` entries in the E15/E16 drivers; the `HParams` fields stage 2 needs; and a
> batch-32 calibration of ViT-S. **E18 is done** (jobs 21532555-60, all COMPLETED). It tests campaign 2's
> untested reading: at the shipped `lambda`, every Fisher arm is momentum SGD at
> `lr (1 - beta) / lambda`, which is 0.033 on the ViT benches. The new arm `sgdm`
> (`benchmarks/common/optimizers.py::LambdaLimitSGD`) is that limit exactly: `AdaFisherMulti`'s
> update with `F~ = lambda I`, where parameters no hooked module owns keep AdaFisher's own fallback.
> It is **bit-identical** to `AdaFisherMulti(gammas=(1.0, 0.0))` on 12 configurations, and equal to
> `torch.optim.SGD(momentum=beta)` at `lr (1 - beta) / (lambda (1 - beta^t))` to 1e-10 in fp64. For
> `diag` the reading is almost arithmetic. Its factors are min-max normalised, then averaged with
> (0.08, 0.008) from a start at 1, so `F~_D - lambda <= 7.56e-5` after warm-up, and its step is at
> least 0.973 times `sgdm`'s from step 200 on, at `lambda = 3e-3`. Each E18 job replays campaign 2's
> grouped job with `sgdm` added, then runs `sgdm` alone at `diag`'s exact 17 550 steps; seeds 0-2 on
> both datasets. **The reproduction gate passed:** at seed 0, `diag` is bit-identical to campaign 2
> over 17 550 of 17 550 steps, on both datasets. Outputs go to
> `benchmarks/outputs/controls/e18_sgdm/`, which `fisher_ref` does not read.
> **Result: the reading is refuted, in the unexpected direction.** At matched steps `sgdm` is
> **worse** than `diag`: test accuracy **−6.23 ± 0.53** points on CIFAR-10 and **−1.58 ± 0.46** on
> CIFAR-100 (3 seeds, paired; behind on all six cells). Against `adamw` it is −10.1 and −5.4 where
> `diag` is −4.0 and −4.2. So ViT-S's deficit is **not** that of momentum SGD at 0.033, which is
> further behind still. No Kronecker mode beats `sgdm` on both datasets (4.3-5.7 points ahead on
> CIFAR-10, tied on CIFAR-100), so rule 3 gives "curvature helps" nowhere. **Where the gap comes
> from:** after step 200 `diag`'s step is 0.973-1.00 times `sgdm`'s by arithmetic, so the gap is
> made in the first 200 steps, where `diag`'s step can fall to 0.279 times `sgdm`'s: an implicit
> warm-up coming from the identity seed. On CIFAR-10 `sgdm`'s loss spikes over its first 10 steps
> (to 3.70, against about 2.3 for `diag`), and the gap then grows until the end; on CIFAR-100 there
> is no spike and the gap stays small. That warm-up reading is **untested**; the control that
> settles it is `sgdm` with `diag`'s early step envelope, or a 200-step linear warm-up, at matched
> steps. The seed-0 budgeted arms reproduce campaign 2 to <= 0.04 (CIFAR-10) and <= 0.31 points
> (CIFAR-100). `plan_lambda_dominance.md`, "E18 — done".

> **E19 — E15's per-layer `λ` on CCT and the two ResNets: pre-registered, code done, submitted**
> (`plan_lambda_dominance.md` §E19). E16 found that the clip loses to the tuned single `λ` on exactly
> the three networks where S1-b, E15's winner, had never run: `cct_2_3x2_cifar`, `resnet20_cifar`,
> `resnet50_cifar`. E19 runs S1-b there (`ekfac`/`tekfac`; E15's eight `τ`, five on ResNet-50), plus
> `netadapt` on CCT and ResNet-20. It does **not** rerun E16's `add` grid or its clips: every E19
> cell pairs by seed with E16's stored cells, which share its initialisation, data order, corrected
> statistic (`norm_exact_rescaling=True`), 1g hardware, 4 workers and optimizer code (`src/` and
> `benchmarks/` unchanged since `7663d62`). Two checks license that pairing. **0a**: a seed-0
> `bridge` cell, E16's `add` cell rebuilt by E19's driver, must be bit-identical to E16's stored one;
> checked locally on CPU before submission for all six (network, mode) pairs, 16 steps with the `λ`
> log firing every 2 steps. **0b**: a `held` single-`λ` cell with S1-b's settings (`hold_cap`,
> `pos_embed` frozen, no decoupled decay) must tie with E16's `add` over every seed. Rules: S1-b −
> `add` and S1-b − `clipema` at 2 SE, summarised in E16 rule 3's words; whether `τ = 0.1` transfers
> (E16 rule 4's plateau); S1-b − `netadapt`. `wdctrl` measures CCT's decoupled decay under S1-b,
> which sizes the convention gap left in the ViT row of `e16_vs_lambda.md`. 392 runs, ~53 h of 1g
> slices, 64 shards and 26 merges (`fisher_ref/slurm/e19_submit.sh`), all from one clean clone,
> `/home/blgr/new_adafisher_e19`: do not move it until every job has run. Read with
> `fisher_ref/experiments/e19_decisions.py`, which needs E16's files beside E19's.

## Working language

All code, comments, docstrings, reports and documentation are written in **English**, to the standard
of a well-maintained academic repository. French output on explicit request only.

## Writing style — plain words, always

**Feynman's rule: if it cannot be said in simple words, it is not understood yet.** This applies to
every report, plan, docstring and chat answer in this project, not only to the ones written for an
outside reader.

- **Plain words over technical ones.** Write "the number the optimizer divides the step by", not
  "the preconditioner", unless that document has already said what a preconditioner is. Write "the
  list of curvature values, one per direction", not "the spectrum". Write "many directions have
  exactly zero curvature", not "the factor is rank-deficient".
- **One idea per sentence.** No stacked subordinate clauses. No chains of em-dashes carrying three
  separate thoughts. If a sentence needs two commas to stay upright, split it.
- **Define on first use, then reuse freely.** A term is allowed once it has been explained in that
  same document. A reader should never have to look elsewhere to parse a sentence.
- **Every number says what it is.** Units, and whether it was measured, computed on paper, or
  estimated. Never a bare figure whose provenance the reader has to guess.
- **No decorative hedging and no inflation.** "We measured X" or "this is not explained" — not
  "it would appear that X may plausibly obtain".
- **The reference examples in this repository** are `docs/reports/validation_noise_investigation.md`
  and `docs/reports/plan_lambda_dominance.md`. Match their register; the second one opens with a
  short version and a straight-through narrative before any detail, which is the shape to copy.

This governs prose. It does **not** rename anything in code: function names, variable names and the
papers' own notation (`A`, `B`, `Lambda`, `rvec`, K-FAC) stay exactly as they are. The rule is that
prose *around* those names explains them.

## Hard constraints

- **`reference_repos/` is read-only.** In particular `reference_repos/FisherAdapTune/`: it exists only
  to recover the original AdaFisher and to understand how it plugs into a training loop. Never edit
  it, never commit into it.
- **`reference_repos/FisherAdapTune/scripts/adafisher.py` is authoritative**, not
  `reference_repos/AdaFisher/` (the official repository has an EMA bug — `docs/reports/plan.md` §1.4).
- **Every claim attributed to a paper cites its section or equation.** The PDFs are in `papers/`. When
  in doubt about a formula below, **read the PDF** — the cheat sheet in this file is a condensed
  reminder, not a source.

## Layout

```
adafisher/
├── pyproject.toml                # package "adafisher-modes", src layout
├── .venv/                        # project-local venv (uv), not FisherAdapTune's conda env
├── papers/                       # 10 PDFs, primary sources (plan.md §0) — + resnet_1512.03385
│                               #   and vit_2010.11929, added lot 8 for the two CIFAR-10 nets
├── reference_repos/              # READ-ONLY
│   ├── FisherAdapTune/           # authoritative AdaFisher  →  scripts/adafisher.py
│   ├── AdaFisher/                # official repo (buggy), comparison only
│   └── EKFAC-pytorch/            # Thrandis — kfac.py, ekfac.py
├── src/adafisher_modes/
│   ├── optimizer.py              # AdaFisherMulti(Optimizer): hooks + Adam wrapper (lot 1: done;
│   │                              #   unchanged since — lots 2-4 needed no optimizer.py change)
│   ├── factors.py                # diagonal H_D/S_D extraction (lot 1: done); full A/B factors
│   │                              #   for Linear (lot 2), Conv2d groups=1/dilation=(1,1) (lot 4),
│   │                              #   and BatchNorm2d/LayerNorm normalized_shape 1-tuple (lot 5:
│   │                              #   done, see plan_lot2.md §0.1, plan_lot4.md §0.1-§0.4,
│   │                              #   plan_lot5.md §0.1-§0.4); the SUA channel-only Conv2d input
│   │                              #   factor (lot 6: done, plan_lot6.md §0.3, §1.1);
│   │                              #   normalized_norm_input/norm_exact_kfe_squares, a norm
│   │                              #   layer's x_hat and exact per-row KFE squares for
│   │                              #   norm_exact_rescaling (E16, plan_floor_clip.md §11)
│   ├── minmax.py                 # MinMaxNormalization + smart_detect_inf, ported (lot 1: done)
│   ├── ema.py                    # shared EMA update, ported (lot 1: done)
│   ├── approximations/
│   │   ├── base.py               # FisherApproximation ABC (lot 1: done, 3 corrections vs.
│   │   │                         #   plan.md §2.2 — see plan_lot1.md §0; unchanged since)
│   │   ├── diag.py               # DiagApproximation (lot 1: done)
│   │   ├── kfac.py ekfac.py      # KFACApproximation, EKFACApproximation; Linear (lot 2) +
│   │   │                         #   Conv2d, groups=1/dilation=(1,1) (lot 4) + BatchNorm2d/
│   │   │                         #   LayerNorm (lot 5) — kfac.py needed zero logic changes for
│   │   │                         #   either, see plan_lot4.md §0.5, plan_lot5.md §0.5; +
│   │   │                         #   conv_sua flag, precondition's per-offset branch (lot 6: done,
│   │   │                         #   plan_lot6.md §1.3-§1.4) — the first lot since lot 1 to touch
│   │   │                         #   every approximations/*.py file, see plan_lot6.md §0.5
│   │   ├── _rescale_utils.py     # rescale() + ClipRule: the division inside the eigenbasis for
│   │   │                         #   ekfac/tekfac, "add" | "floor" | "clip" with a "quantile" |
│   │   │                         #   "ema" | "fixed" threshold (E16, plan_floor_clip.md)
│   │   ├── _eigh_utils.py        # eigenbasis(): the conditioning ridge + CPU fallback shared by
│   │   │                         #   ekfac/tekfac (step 1 follow-up: cuSOLVER's eigh crashed on an
│   │   │                         #   exactly rank-deficient factor, killing two cluster runs)
│   │   ├── _kron_utils.py        # shared weight/bias augment/split for kfac/ekfac/tkfac/tekfac;
│   │   │                         #   Conv2d 4D-weight reshape added lot 4 (plan_lot2.md §0.2,
│   │   │                         #   plan_lot4.md §0.3); unchanged by lot 5 (1-D norm-layer weight
│   │   │                         #   already covered by that reshape, plan_lot5.md §0.5); +
│   │   │                         #   augment_conv2d_direction_sua/split_conv2d_direction_sua
│   │   │                         #   (lot 6: done, plan_lot6.md §0.4, §1.2)
│   │   ├── tkfac.py tekfac.py    # TKFACApproximation, TEKFACApproximation; Linear (lot 3) +
│   │   │                         #   Conv2d, groups=1/dilation=(1,1) (lot 4) + BatchNorm2d/
│   │   │                         #   LayerNorm (lot 5, zero logic changes) + conv_sua (lot 6: done,
│   │   │                         #   plan_lot6.md §1.5)
│   │   ├── _tkfac_utils.py       # shared (delta, Phi_raw, Psi_raw) numerators for tkfac/tekfac
│   │   │                         #   (lot 3: done, see plan_lot3.md §0.2; unchanged by lot 4 —
│   │   │                         #   already generic over any (h_bar, s) shape)
│   │   └── __init__.py           # MODES registry {name → factory}
│   └── config.py                 # not lot 1/2 — introduced when a benchmark needs YAML (plan_lot1.md §2)
├── tests/                        # lot 1: test_diag_bitexact.py, test_diag_eq4_semantics.py,
│                                  #   test_minmax_matches_official.py, conftest.py
│                                  # lot 2: test_full_factors_match_diag.py,
│                                  #   test_kfac_ekfac_precondition.py, test_frobenius_dominance.py
│                                  # lot 3: test_tkfac_tekfac_precondition.py,
│                                  #   test_ekfac_tekfac_equiv.py, test_frobenius_dominance.py extended
│                                  # lot 4: test_full_factors_match_diag.py,
│                                  #   test_frobenius_dominance.py, test_kfac_ekfac_precondition.py,
│                                  #   test_tkfac_tekfac_precondition.py all extended with a Conv2d
│                                  #   section; test_conv2d_optimizer_smoke.py (new)
│                                  # lot 5: test_full_factors_match_diag.py,
│                                  #   test_frobenius_dominance.py, test_kfac_ekfac_precondition.py,
│                                  #   test_tkfac_tekfac_precondition.py all extended with a
│                                  #   BatchNorm2d/LayerNorm section; test_norm_layers_optimizer_
│                                  #   smoke.py (new)
│                                  # lot 6: test_full_factors_match_diag.py,
│                                  #   test_frobenius_dominance.py, test_kfac_ekfac_precondition.py,
│                                  #   test_tkfac_tekfac_precondition.py all extended with a
│                                  #   SUA section; test_conv2d_sua_optimizer_smoke.py (new)
│                                  # lot 7: test_equal_wallclock_bench.py (new) — exercises the
│                                  #   dataset-agnostic time-budget training harness, not MNIST
│                                  # lot 8: test_cifar10_bench.py (new) — both models, the
│                                  #   parameter-pairing regression, fisher_batch_samples, both
│                                  #   weight-decay conventions, the eval-excluding budget
│                                  #   harness, Cutout, the 45k/5k split; all offline
│                                  # step 1: test_benchmark_models.py (new) — parametrized over
│                                  #   every benchmarks/models/<model>/ folder: exact parameter count,
│                                  #   hooked-module inventory + the explicit list of parameters
│                                  #   belonging to no hooked module, "no parameter left
│                                  #   un-updated", all 5 modes, the bench spec (<= 50 lines, no
│                                  #   control flow), and the checkpoint schedule; all offline
│                                  # step 1 follow-up: test_eigh_conditioning.py (new) — the ridge's
│                                  #   mathematical inertness, orthonormality on the exact
│                                  #   rank-deficient structure that killed two runs, the CPU
│                                  #   fallback; all offline
│                                  # cifar100 + imagenet: test_dataset_benches.py (new) — the
│                                  #   ImageFolder pipeline on a synthetic tree, both transform
│                                  #   regimes, the seeded split, output_group routing, and that
│                                  #   every bench has generated jobs in its own subdirectory
│                                  # S3: test_ema_seed_first.py (new) -- the identity-seed fix:
│                                  #   default inertness, the constant-input property, all 5 modes
│                                  # full-repo audit: test_optimizer_robustness.py (new) — two
│                                  #   parameter groups, LayerNorm(bias=False), a frozen bias, a
│                                  #   module first reached after step 0, two backwards over one
│                                  #   forward, a tied weight, precondition-vs-exp_avg aliasing;
│                                  #   test_eig_before_rescale.py (new) — the ekfac/tekfac
│                                  #   ordering knob; test_full_factors_match_diag.py extended
│                                  #   with the bias-free Conv2d scale quirk, three scope guards
│                                  #   and the raw-vs-normalised input-factor pin. All offline
│                                  # campaign 1 audit: test_lr_schedule.py (new) — NominalCosine
│                                  #   identical to torch's inside T_max and clamped outside it,
│                                  #   BudgetCosine's shape/monotonicity/floor, both through the
│                                  #   real loop; all offline
│                                  # authors' repro: test_official_adafisher_arm.py (new) — the
│                                  #   `official` arm is the authors' PUBLISHED optimizer, loaded
│                                  #   from their own file: the class, AdaFisherW under
│                                  #   decoupled_wd, the single-gamma translation and its two
│                                  #   refusals, and the MEASURED agreement with diag (min-max on)
│                                  #   on a Linear-only net and on all four hooked layer types
│                                  # E18: test_sgdm_arm.py (new) — the `sgdm` arm is bit-identical
│                                  #   to AdaFisherMulti with gammas=(1,0), i.e. F~ = lam*I (12
│                                  #   configurations, unhooked ViT parameters included), equals
│                                  #   torch SGD-momentum at lr(1-beta)/(lam(1-beta^t)) in fp64, and
│                                  #   pins the 7.56e-5 bound on diag's F~ - lam that E18 predicts from
├── benchmarks/                   # step 1 (plan_exp_step1.md): a package — one shared harness in
│   │                             #   common/, one folder per tested model. The five flat modules
│   │                             #   of lots 1/7/8 are gone; every line of them landed here.
│   │                             #   Post-step-1 reorganization: every model folder now sits one
│   │                             #   level deeper, under models/, so benchmarks/ itself only holds
│   │                             #   the shared harness, the SLURM generator, data/, outputs/ and
│   │                             #   archives/. discover_benchmarks() globs models/*/bench.py and
│   │                             #   imports benchmarks.models.<name>.bench; every CLI invocation,
│   │                             #   cross-folder import (e.g. vit_micro_cifar's ViTCIFAR from
│   │                             #   vit_small_cifar) and generated SLURM job follows the same
│   │                             #   benchmarks.models.<name> prefix. outputs/, data/ and slurm/
│   │                             #   did NOT move — only the 20 model source folders did.
│   ├── common/
│   │   ├── data.py               # mnist()/cifar10()/cifar100(), imagenet()/imagenet32() (an
│   │   │                         #   ImageFolder tree, staged by hand), Cutout,
│   │   │                         #   seeded_train_val_split
│   │   ├── loop.py               # train_under_budget() = lot 7's + lot 8's loops merged (D2),
│   │   │                         #   evaluate(), sync(), prepare_batch/metric_fn conventions;
│   │   │                         #   + the optional per-batch `lr_schedule` hook
│   │   ├── schedules.py          # NominalCosine (clamped past T_max) / BudgetCosine (anneals over
│   │   │                         #   the arm's own WCT budget) — the `--lr-schedule` protocol fix
│   │   ├── optimizers.py         # HParams, ARMS (5 modes + adam/adamw + reference + official
│   │   │                         #   + sgdm), build_optimizer(arm, model, hp). `reference` is
│   │   │                         #   FisherAdapTune's AdaFisher and `official` the PUBLISHED
│   │   │                         #   repository's, both loaded by file path and unmodified; the
│   │   │                         #   second is diag's like-for-like partner, since it min-maxes.
│   │   │                         #   `sgdm` (LambdaLimitSGD, E18) is AdaFisherMulti with F~ = lam*I:
│   │   │                         #   the curvature-free control, on the Fisher arms' lr/lam/beta
│   │   ├── records.py            # StepRecord/EpochRecord/ArmResult + csv/summary/plot/manifest
│   │   │                         #   writers, lot 8's column schema unchanged
│   │   ├── checkpoints.py        # --checkpoints 0,0.01,0.1,0.5,1 -> ckpt_<frac>.pt (D5); fractions are
│   │   │                         #   relative to the NOMINAL trajectory, shared across arms
│   │   └── runner.py             # Benchmark, build_parser(bench), main(bench),
│   │                             #   discover_benchmarks() — the folder list *is* the registry
│   ├── models/                    # one folder per tested model (moved here out of benchmarks/
│   │   │                         #   directly, see the note above); benchmarks.models.<name> is
│   │   │                         #   the import prefix and `python -m benchmarks.models.<name>.bench`
│   │   │                         #   the CLI invocation for every one of the 20 folders below.
│   │   ├── mnist_autoencoder/    # model.py bench.py  (migrated, lots 1+7)
│   │   ├── mlp_ln_mnist/         # model.py bench.py  (A1, new: 26 634 params)
│   │   ├── cnn_gn_cifar/         # model.py bench.py  (A2, new: 24 458; --norm gn|bn)
│   │   ├── vit_micro_cifar/      #          bench.py  (A3, new: 21 098; imports ViTCIFAR from
│   │   │                         #   vit_small_cifar/model.py — D1's one cross-folder import)
│   │   ├── resnet20_cifar/       # model.py bench.py  (B2, new: 269 722, option-A shortcuts)
│   │   ├── cct_2_3x2_cifar/      # model.py bench.py  (B1, new: 283 723, AdaFisher's own model)
│   │   ├── resnet50_cifar/       # model.py bench.py  (migrated verbatim, lot 8: 23 520 842)
│   │   ├── vit_small_cifar/      # model.py bench.py  (migrated verbatim, lot 8: 2 693 578)
│   │   ├── <arch>_cifar100/      # 6 folders: the six architectures above with num_classes=100.
│   │   │                         #   bench.py only — the model is imported from <arch>_cifar/
│   │   └── <arch>_imagenet/      # 6 folders, num_classes=1000. Four read ImageNet downsampled to
│   │                             #   32x32 (imagenet32) and reuse the CIFAR model unchanged;
│   │                             #   resnet50_imagenet has its OWN model.py (the paper's ImageNet
│   │                             #   7x7/s2+maxpool stem, 25 557 032) and vit_small_imagenet is
│   │                             #   ViTCIFAR configured as ViT-S/16 @224 (22 050 664)
│   ├── outputs/<group>/<model>/<arm>/   # results indexed by what they measure, not by lot (D6),
│   │                             #   grouped by dataset (Benchmark.output_group): mnist, cifar10,
│   │                             #   cifar100, imagenet. outputs/seeds/<model>/seed<n>/ keeps its
│   │                             #   own layout (its axis is the seed); outputs/lot8_cifar10/ is
│   │                             #   legacy and left untouched. fisher_ref/checkpoints.py reads
│   │                             #   BOTH the grouped and the flat layout
│   ├── slurm/                    # 187 generated sbatch jobs (20 models x (1 calibration + 7
│   │   ├── mnist/                #   per-arm) + 18 grouped + 1 lam sweep + 1 two-arm seeds job
│   │   ├── cifar10/              #   + 7 all-arm seed-completion jobs) + READMEs
│   │   ├── cifar100/             #   + generate_jobs.py, which enumerates model folders. One
│   │   └── imagenet/             #   subdirectory per dataset, matching outputs/. At the top
│   │                             #   level only: generate_jobs.py, train_v0_seeds.sh (two arms,
│   │                             #   spans two datasets), train_seeds_<model>.sh (all seven arms,
│   │                             #   from SEED_COMPLETION -- one per model, the completion of the
│   │                             #   seed axis) and the hand-written dispatch_remaining_arms.sh;
│   │                             #   imagenet/ also holds the hand-written stage_imagenet.sh
│   ├── authors_repro/            # the authors' OWN code, run as they run it (plan_authors_repro.md):
│   │                             #   slurm/run_authors_train.sh copies reference_repos/AdaFisher to
│   │                             #   $SLURM_TMPDIR, checks the copy (diff -r AND a file count),
│   │                             #   installs their pinned torch 2.3.0/torchvision 0.18.0/numpy
│   │                             #   1.26.4 and runs THEIR train.py on THEIR config, verbatim.
│   │                             #   asdl_stub/ is the one deviation: train.py imports asdl
│   │                             #   unconditionally for its Shampoo/K-FAC arms only
│   └── archives/                 # gitignored; superseded/duplicated result trees moved aside
│                                 #   between campaigns, each with its own README saying why
├── docs/reports/
│   ├── plan_exp_step1.md         # the Fisher-drift campaign's step 1: the benchmarks/ package
│   ├── plan_exp_draft_v0.md      # the campaign's original draft (French), kept verbatim
│   ├── plan_exp_draft.md         # THE campaign plan (v1): plan_exp_draft_v0 adapted to the
│   │                             #   benchmark models, runs, seeds and checkpoints this repo
│   │                             #   actually produces; its §0 lists every change from v0
│   ├── plan_exp_lot0.md          # lot-0 implementation plan: why v0's lot-0 exit criterion was
│   │                             #   unsatisfiable, the probe/registry/bridge design, the
│   │                             #   measured sample-independence tolerance
│   ├── plan_exp_lot1.md          # lot-1 implementation plan: the backward-hook pruning finding,
│   │                             #   the named_parameters() column layout, the curvlinops
│   │                             #   decision, and §4's smoke on the real A1 checkpoint
│   ├── campaign2_cifar100_imagenet.md # campaign 2's results: CIFAR-100, ImageNet32, ResNet-50 on
│   │                             #   ImageNet-1K, and the two large CIFAR-10 models re-run
│   ├── plan_authors_repro.md     # the authors' own code: the `official` arm on the stalling
│   │                             #   auto-encoder (done — the stall is not the port's) and their
│   │                             #   train.py on Table 2's CIFAR-10 ResNet18 rows (§6.2 pending)
│   └── archives/                 # the optimizer-implementation lots, moved aside once done. Still
│                                 #   the authoritative record of every design decision cited
│                                 #   throughout this file and in the code — plan.md (overall,
│                                 #   lots 1-8) and plan_lot1.md ... plan_lot8.md. Citations
│                                 #   elsewhere name them by bare filename: they live here.
├── fisher_ref/                   # the Fisher-drift campaign's own package (plan_exp_draft.md §7).
│   │                             #   A *reader*: trains nothing, changes nothing in benchmarks/
│   │                             #   or src/. Lots 0 and 1 (done) ship:
│   ├── conventions.py            #   fp64 policy, TF32 off + what was actually set, rvec/kron
│   │                             #   convention, reference_mode, assert_sample_independent
│   ├── probes.py                 #   fixed, augmentation-free, content-hashed probe sets on the
│   │                             #   run's own seeded train/val split
│   ├── registry.py               #   parameter block -> layer type, weight sharing read from one
│   │                             #   forward pass; partitions every parameter of every model
│   ├── checkpoints.py            #   benchmarks/outputs/ -> theta: both layouts (seed 0 and
│   │                             #   outputs/seeds/<model>/seed<n>/), rejects pre-fix payloads
│   ├── sources.py                #   lot 1: the backprop vectors — type-2 (closed-form softmax
│   │                             #   root), MC_K (with its K^{-1/2}), empirical; CE and MSE
│   ├── capture.py                #   lot 1: per-example (a, g) and the recomputed x_hat of a norm
│   │                             #   layer; per_sample_gradients per layer kind. Uses a *tensor*
│   │                             #   hook on the module output, not register_full_backward_hook
│   │                             #   — see "Things to watch" (plan_exp_lot1.md §0.1)
│   ├── reference/dense.py        #   lot 1: regime A — F = U^T U, E_hat, B_l in fp64, U streamed;
│   │                             #   columns in named_parameters() order (plan_exp_lot1.md §0.5).
│   │                             #   ONE P x P buffer on the device: addmm_ + symmetrize_ (§0.9)
│   ├── approx/, metrics/,        #   lot 2: the zoo (K-FAC, EKFAC, TKFAC, AF-raw, norm readings)
│   │   runners/p1_structural.py  #   and M1/M3/M5/M7/M8 + the P1 CLI. Lot 3 extended all three
│   │                             #   to weight sharing: reduce variants, approx/sharing.py (the
│   │                             #   B^exp decomposition), approx/embed.py (pos_embed), diag_py,
│   │                             #   the identity control (plan_exp_lot3.md)
│   ├── approx/adafisher_state.py #   lot 5: P2 -- AdaFisherMulti's live state as curvature blocks,
│   │                             #   in the layout P1 holds each block in. A normalisation layer's
│   │                             #   is INTERLEAVED there and BLOCKED here: norm_permutation()
│   ├── rewarm.py                 #   lot 5: frozen-theta re-warm from a checkpoint's weights;
│   │                             #   refuses < 10*TCov steps (0.08^k << lambda, not << 1)
│   ├── runners/p2_operational.py #   lot 5: the P1 runner with three more rungs injected, not a
│   │                             #   second runner -- one reference build, so P1 and P2 are paired
│   ├── folds.py                  #   lot 3: one traversal over K folds, sums offloaded to the host,
│   │                             #   halves assembled by addition -> per-layer intervals for free
│   ├── experiments/              #   measurement drivers, not tests; env-var constants, no CLI.
│   │                             #   rewarm_fidelity.py, identity_seed_residual.py,
│   │                             #   dense_reference_a1.py (lot 1 phase 5: F/E_hat/blocks, the
│   │                             #   source gap, the train/val gap, the §3.4 noise floor)
│   └── slurm/                    #   the campaign's ANALYSIS jobs — hand-written, separate from
│                                 #   benchmarks/slurm/ (read-only, and its generator enumerates
│                                 #   training benchmarks). Every job states its own memory.
├── requirements-cluster.txt      # lot 8: --no-index install list for the cluster's wheelhouse
└── CLAUDE.md
```

## Where the abstraction lives

`src/adafisher_modes/approximations/base.py`:

```python
class FisherApproximation(ABC):
    def update_factors(self, module, A, B, step: int) -> None: ...   # EMA (AdaFisher Eq. 3)
    def refresh(self, module, step: int) -> None: ...                # amortised inverses / eigenbases
    def precondition(self, module, direction: Tensor) -> Tensor: ... # applies F̃⁻¹
```

**`precondition` is applied to the bias-corrected first moment `m̂ = m/(1−β₁ᵗ)`, not the raw gradient.**
The original update `param.addcdiv_(exp_avg, F_tilde, -lr/bc)`
(`reference_repos/FisherAdapTune/scripts/adafisher.py:273`) is an element-wise division, hence
`θ ← θ − α·F̃_D⁻¹m̂` for a diagonal `F̃_D`. The four new modes are not diagonal in the parameter basis,
so the operator must be applied to `m̂`. The `diag` mode then reduces to `addcdiv_` exactly. In code,
`precondition` receives the raw momentum `m` and the optimizer folds `1/(1−β₁ᵗ)` into the step size,
as the reference does; for a linear operator that is the same thing. The one exception is
`rescale_form="clip"` (E16), which is not linear -- it compares the momentum with a threshold -- and
therefore receives `m̂` itself, with no further division (`consumes_bias_corrected_momentum`).
Preconditioning `g` before averaging would be a different algorithm (K-FAC + momentum) and would break
the `m^(t)` column of AdaFisher's Table 1.

## Selecting a mode

Constructor:

```python
AdaFisherMulti(model, lr=1e-3, beta=0.9, Lambda=1e-3, gammas=[0.92, 0.008], TCov=100,
               fisher_mode="ekfac",         # diag | kfac | ekfac | tkfac | tekfac
               minmax_normalization=True,   # diag mode only; True = faithful to Eq. (4)
               T_inv=100, T_eig=100, T_re=1,
               conv_sua=False,              # kfac/ekfac/tkfac/tekfac only; Conv2d SUA (lot 6)
               fisher_batch_samples=None,   # lot 8: estimate the factors from the first k examples
                                            #   of each batch only (memory); None = whole batch
               decoupled_weight_decay=False,# lot 8: True = the official AdaFisherW rule
               ema_seed_first=False,        # S3: start each running average from its first
                                            #   observation instead of the identity, so no residue
                                            #   of the identity survives (plan_lambda_dominance.md)
               eig_before_rescale=False,    # ekfac/tekfac only: rebuild the eigenbasis in the
                                            #   backward hook, BEFORE projecting the gradient into
                                            #   it, so s*/Theta is measured in the basis
                                            #   precondition then uses (EKFAC Lemma 1, both papers'
                                            #   Alg. 1). ValueError on the other three modes.
               damping="global",            # kfac/ekfac/tkfac/tekfac: "layer_relative" gives each
               damping_tau=None,            #   module lambda_l = tau * (mean eigenvalue of its own
                                            #   undamped stored curvature); "network_relative" one
                                            #   tau * (mean over all the network's directions).
                                            #   Fix S1 / E15 of plan_lambda_dominance.md
               hold_cap=False,              # multiply each preconditioned direction by the lambda
                                            #   inside it, so lr is the step-size cap whatever lambda
                                            #   is: hold_cap=True, lr=c  ==  lr=c*Lambda
               rescale_form="add",          # ekfac/tekfac only: divide by s+lambda ("add"), by
                                            #   max(s, lambda) ("floor"), or clip the undamped step
                                            #   ("clip", lambda unused). E16, plan_floor_clip.md
               clip_threshold="quantile",   # "clip" only, always per module: "quantile" = a fraction
               clip_fraction=None,          #   clip_fraction of the active coordinates clipped every
               clip_ema_horizon=None,       #   step; "ema" = the same, times the momentum's size
               clip_calibrate_at=None,      #   against its bias-corrected average (1000 steps);
               clip_calibration_window=None,#   "fixed" = the median of the quantile over the window
               clip_guard=1e-3,             #   before clip_calibrate_at, frozen (a conditional clip)
               norm_exact_rescaling=False)  # ekfac/tekfac only: estimate a BatchNorm2d/LayerNorm
                                            #   layer's s*/Theta from its own per-row gradient
                                            #   [delta*x_hat, delta], not from the §4.6 surrogate.
                                            #   Inert without a hooked norm layer. E16's third
                                            #   amendment, plan_floor_clip.md §11
```

YAML (style of `reference_repos/FisherAdapTune/crack_segmentation/config_segformer.yaml`):

```yaml
fisher_mode: ekfac
adafisher_tcov: 100
adafisher_gamma: [0.92, 0.008]
fisher_lambda: 1.0e-3
fisher_minmax: true    # diag mode only — true = paper-faithful (Eq. 4)
fisher_t_eig: 100      # ekfac / tekfac
fisher_t_inv: 100      # kfac / tkfac
fisher_t_re: 1         # tekfac, T_RE of Alg. 1
fisher_conv_sua: false # kfac/ekfac/tkfac/tekfac only — true = SUA's channel-only Conv2d input
                        #   factor (memory-tractable on ResNet/ViT-scale conv layers, lot 6)
fisher_batch_samples:  # lot 8, null = whole batch; an int caps the *curvature statistic* to the
                        #   first k examples (ekfac/tekfac's cached batch is 3.4 GB on ResNet-50
                        #   at batch 128 otherwise — plan_lot8.md §0.4)
fisher_decoupled_wd: false # lot 8, true = AdaFisherW (decoupled decay); use with AdamW baselines
fisher_ema_seed_first: false # S3, true = seed each running average from its first observation
fisher_eig_before_rescale: false # ekfac/tekfac only, true = measure the rescaling in the basis
                        #   precondition will use, not in the one about to be replaced
```

The mode is a **configuration field, not a separate optimizer**: no other part of the training code
knows which mode is active.

## Min-max normalisation — read this before touching `diag`

- **`diag` defaults to min-max ON**, faithful to Eq. (4) of the paper. It is applied to the
  *instantaneous* factors `H_D_i`, `S_D_i` **before** the EMA — the placement used by the official
  repository (`AdaFisher/optimizers/AdaFisher.py:412`, `:431`), not after accumulation.
- `minmax_normalization=False` reproduces `FisherAdapTune/scripts/adafisher.py` bit-exactly. This is
  the setting the non-regression test runs under.
- **The four new modes never apply min-max.** It would destroy TKFAC's trace preservation (Thm 4.1)
  and make `‖F − F̃‖_F` meaningless. Damping is `λ` only, as each paper prescribes.
- Reproduce the official semantics exactly, including the `smart_detect_inf` pre-pass (`+inf → 1`,
  `−inf → 0`) and the `epsilon = 1e-6` guard in the denominator. `_smart_detect_inf` already exists,
  unused, at `adafisher.py:21`.

## Every deviation from the original AdaFisher

**Why this section exists.** For a paper, or for any comparison against published AdaFisher numbers,
it must be possible to say exactly what this code does differently from the original, and whether the
difference is on by default. That is what this table is. It is the authoritative list; when
something changes, change it here too.

"Original" means two things that do not always agree, so both are named:
**P** = the AdaFisher paper (`papers/adafisher_2405.16397.pdf`, Algorithm 1, Eq. (3)-(4), Prop. 3.1-3.2);
**R** = the reference implementations (`reference_repos/FisherAdapTune/scripts/adafisher.py`, which is
authoritative here, and `reference_repos/AdaFisher/`, kept for comparison only).

### 1. What is reproduced exactly

These are the anchors. If any of them breaks, the port is broken.

| What | Against what | Test |
|---|---|---|
| `diag` with `minmax_normalization=False`: `F~` at every EMA update, and the parameter trajectory | **R** (FisherAdapTune), bit-exact | `test_diag_bitexact.py` |
| `diag` at its **default** (min-max ON): the parameter trajectory | **R** (official repo, the `official` arm), to `4.6e-7` relative over 40 steps on a `Linear`-only net and `3.6e-7` over 20 on all four hooked layer types; `2.58e-7` over a real 2 200-step run. Not bit-exact, by §2.1 alone | `test_official_adafisher_arm.py` |
| `min_max_normalization` and `smart_detect_inf`, including the `epsilon = 1e-6` guard and the `+inf -> 1` / `-inf -> 0` pre-pass | **R** (official repo), bit-exact | `test_minmax_matches_official.py` |
| Scale convention: mean over `batch x spatial` everywhere | **R** (AdaFisher's, *not* `EKFAC-pytorch`'s, which multiplies `grad_output` by the batch size) | the factor tests |

### 2. Differences that are ON by default

**These five change behaviour without being asked for.** Any comparison against a published number
has to account for them. Two of them (2.2 and 2.5) are bug fixes: the reference's own version of
each is wrong on a network or a parameter grouping this repository actually uses.

| # | Difference | Why | Size of the effect |
|---|---|---|---|
| 2.1 | The update is **unfused**: `precondition()` returns a direction, then `param.add_(direction, alpha=-lr/bc)`. **R** uses the fused `addcdiv_` (`adafisher.py:273`). | `addcdiv_` is an element-wise division, so it assumes the preconditioner is diagonal in the parameter basis. That is true only for `diag`. The four new modes are not diagonal there, so the operator has to be *applied*, not divided by. See `plan_lot1.md` §0.3. | A few ULPs per step on a `Linear` layer. Amplified to about `1e-4` over 6 steps once `BatchNorm2d`'s running statistics feed back. This is why `test_diag_bitexact` resyncs parameters after each step. |
| 2.2 | Parameters are paired to modules by **identity** (`id(param) -> module`), not by walking both lists positionally. **R** uses an index loop (`adafisher.py:275-307`). | **This is a bug fix, not a preference.** The index loop ran exactly `len(self.modules)` iterations and spent one on each *unpaired* parameter, so a network with `k` parameters outside the four hooked layer types silently lost its last `k` modules. Measured on `ViT-S/4`: `cls_token` and `pos_embed` cost the final `LayerNorm` **and the whole classification head**, which were never updated, with finite gradients and a still-falling loss. `plan_lot8.md` §0.3. | None on any network where the index loop was correct, which is asserted, not assumed (`test_pairing_matches_legacy_loop_on_reference_nets`). Total on any network with unpaired parameters. |
| 2.3 | `diag` applies min-max to the **instantaneous** factors, *before* the EMA. | This follows **R** (`AdaFisher.py:412`, `:431`) and contradicts **P** (Algorithm 1 lines 4-5, which averages first and normalises after). The two orderings are not cosmetic: normalising destroys scale, so "before" leaves the EMA's own mis-scaling in the result and "after" erases it. `audit_step.md` §4.4. | Decisive. In the shipped order `F~_D` lives in `[lambda, lambda + 7.6e-5]`, a 7.6% spread across a whole layer; in the paper's order it lives in `[lambda, 1 + lambda]`. Measured in `plan_lambda_dominance.md` Part 6, E1. Switchable with `minmax_after_average=True`. |
| 2.4 | `ekfac`/`tekfac` decompose through `approximations/_eigh_utils.py::eigenbasis`, which adds a relative ridge (`1e-6 * mean(diag(M))`) and falls back to the CPU solver. | **P** and **R** have no eigendecomposition at all, so there is nothing to deviate from; but this is code that runs and it should be on the list. Without it, cuSOLVER raises on an exactly rank-deficient factor, which killed two real cluster runs. | Mathematically inert on the returned basis: `M + cI` has the same eigenvectors as `M`, in the same order. |
| 2.5 | A module is stepped **once per `step()`**, through one `stepped` set shared by the whole walk over the parameter groups. **R** rebuilds its bookkeeping inside the `for group in self.param_groups` loop (`adafisher.py:281`). | **This is a bug fix, not a preference.** A module is preconditioned once, with its weight and bias together, so "already stepped" is a property of the module, not of the group a parameter happens to sit in. With the standard decay/no-decay split — weights in one group, biases in the other — every hooked module was reached from both groups and stepped twice. | Measured on a two-layer net with that split: every parameter moved **exactly 2.0000 times** as far from its initial value as under a single group. None on any model in this repository, all of which build one group — which is why this is a behaviour-preserving fix and not a knob. One limitation stays and is documented in `optimizer.py`: a module whose weight and bias end up in *different* groups is stepped with the hyperparameters of whichever group its first-encountered parameter is in. `tests/test_optimizer_robustness.py` |

### 3. Differences that are OFF by default

Every one of these reproduces today's exact behaviour when left alone. They exist so that an
experiment can turn one thing on at a time.

| Knob | What it does | Where it is justified |
|---|---|---|
| `fisher_mode` in `kfac`, `ekfac`, `tkfac`, `tekfac` | The whole point of the project: four alternative ways to build `v^(t)` from the same per-layer Fisher block. `diag` is AdaFisher's own. | `plan.md`, lots 2-6 |
| `conv_sua` | The SUA channel-only `Conv2d` input factor, `d_in = C_in[+1]` instead of `C_in*k_h*k_w[+1]`. | `plan_lot6.md` |
| `fisher_batch_samples` | Estimate the factors from the first `k` examples of each batch only. Changes the *estimator*, not the applied gradient. | `plan_lot8.md` §0.4 |
| `decoupled_weight_decay` | The official `AdaFisherW` rule, so ViT arms can be compared like-for-like with `AdamW`. | `plan_lot8.md` §0.6 |
| `gamma` | Overrides `gammas` with `(1-gamma, 1-gamma)`, which collapses the EMA to **P**'s Eq. (3) exactly. | `audit_step.md` §4.7-§4.8 |
| `minmax_after_average` | Moves the min-max to where Algorithm 1 puts it (see 2.3). | `audit_step.md` §4.4 |
| `ema_seed_first` | Starts each running average from its first observation instead of from the identity, so no residue of the identity is ever left in the state. Fix S3. | `plan_lambda_dominance.md` Part 4, `tests/test_ema_seed_first.py` |
| `eig_before_rescale` | `ekfac`/`tekfac` only. Rebuilds the eigenbasis **inside the backward hook**, before the gradient is projected into it, so the rescaling is measured in the basis `precondition` then uses; `refresh()` does not redo it that step. By default the basis is replaced *afterwards*, in `step()`, so `s*`/`Theta` describe a basis that no longer exists — EKFAC's Lemma 1 makes the optimal diagonal optimal for the `Q` it was measured in and no other, and Algorithm 1 of both papers, plus `EKFAC-pytorch/ekfac.py::step`, order it eigenbasis-then-rescaling. Passing it to the other three modes is a `ValueError`, not a silent no-op. **Measured** (24-16-6 net with a `LayerNorm`): the two orderings store an `s*`/`Theta` differing by **91%–141%** relative, but the *applied step* differs by only **3.0e-3** (60 steps, `TCov=10`) and **3.7e-5** (200 steps, `TCov=20`) at the shipped `Lambda=1e-3`, against **8.3** and **0.28** at `Lambda=1e-8`. So no trained model in this repository is affected — and **fix the ordering before acting on `plan_lambda_dominance.md`'s fix S1**, which lowers `Lambda`. | EKFAC Lemma 1 / Alg. 1, TEKFAC Alg. 1, `audit_full_repo.md` finding 3, `tests/test_eig_before_rescale.py` |
| `T_inv`, `T_eig`, `T_re` | Amortisation cadences for the inverse, the eigenbasis and the rescaling. | K-FAC §6.3, TEKFAC Alg. 1 |
| `damping`, `damping_tau` | The four Kronecker modes only. `"layer_relative"` replaces the one shared `Lambda` by `lambda_l = tau * c_l`, where `c_l` is the mean eigenvalue of module `l`'s own undamped stored curvature (`mean(s*)`, `mean(Theta)`, `(tr A/d_in)(tr B/d_out)`, `tr Phi_raw tr Psi_raw / (delta d_in d_out)`), recomputed at the start of every `step()`. `"network_relative"` gives every module one `tau * c_net`, the mean over all the network's directions. With `tekfac` and `tau = 1` this is TEKFAC's eq. (3.5) without its floor. `kfac`/`tkfac` bake the damping into inverses rebuilt every `T_inv` steps. `ValueError` on `diag`, whose min-max removes the scale a relative damping needs. **Measured in E15** (batch 32, 15 epochs, `cnn_gn_cifar` and `vit_micro_cifar`, five seeds, `hold_cap=True`): `layer_relative` beats the best single `lambda` in all six (network, mode) pairs by +1.9 to +6.1 points, and `network_relative` in all six too, so the gain is per-layer; +8.2 to +14.8 over the default `lambda`; `tau = 0.1` is in the plateau of `ekfac`/`tekfac` on both networks, `kfac` wants 0.3-3. | S1 and E15 of `plan_lambda_dominance.md`, `tests/test_relative_damping.py` |
| `hold_cap` | Multiplies each preconditioned direction by the damping inside it, so that `lr` is the step-size cap, the largest multiple of the momentum any direction can move by. `hold_cap=True, lr=c` is the update `lr=c*Lambda` gives under a global damping, which is how E7-E14 held the cap; under a relative damping it holds it in every layer at once. The plain-momentum fallback is scaled by the shared `Lambda`. Decoupled decay stays `1 - lr*wd`, so it no longer vanishes as `lambda` falls (rule 3 of `plan_lambda_dominance.md` Part 5). | E15, `tests/test_relative_damping.py` |
| `rescale_form`, `clip_threshold`, `clip_fraction`, `clip_guard`, `clip_ema_horizon`, `clip_calibrate_at`, `clip_calibration_window` | `ekfac`/`tekfac` only. How the projected direction is divided by `s*`/`Theta` inside the eigenbasis: `"add"` = `s + lambda` (default, bit-identical to before, checked against three earlier commits), `"floor"` = `max(s, lambda)` (family A), `"clip"` (family B, Sophia-type; `lambda` unused): an active coordinate (`\|M\| > guard = clip_guard * rms(M)`) moves by `sign(M) min(r / gamma, 1)`, `r = \|M\| / s`, an inactive one by `M / max(gamma s, guard)`. The threshold is always **per module**, because the stored curvature's scale error differs between layers by up to four orders of magnitude (`1/T` of shared layers, the norm-layer surrogate): `"quantile"` = the `max(floor(q n_a), 1)`-th largest `r` among the `n_a` active coordinates at every step, a per-module normalisation; `"ema"` = that times `mu_bar / mu`, `mu = rms(M)`, with `log mu_bar` Adam's bias-corrected running average of `log mu` over `clip_ema_horizon` steps; `"fixed"` = the lower median of the module's quantile over the `clip_calibration_window` steps before `clip_calibrate_at`, frozen then (continuous at the switch; a true conditional clip after). Clipped is defined by `r >= gamma` on the very tensor `gamma` is selected from, so the count is exact; the selection is a sort and a gather, with no host synchronisation; the statistics are computed only when read. Under `"clip"` the mode receives the bias-corrected momentum and the step is not divided again. `ValueError` on the other three modes, for an argument belonging to another threshold, and with `hold_cap` or a relative damping under `"clip"`. The P2 reader refuses both non-default forms. | E16, `plan_floor_clip.md`, `tests/test_rescale_form.py` |
| `norm_exact_rescaling` | `ekfac`/`tekfac` only. On a `BatchNorm2d`/`LayerNorm`, estimates `s*`/`Theta` from the layer's **own** per-row gradient `[delta*x_hat, delta]`, projected into the same eigenbasis, instead of the gradient of the §4.6 surrogate `[z*delta, delta]`. `x_hat` is recomputed in the forward hook with the statistics the layer used: per token for `LayerNorm`; for `BatchNorm2d`, the batch's in training and the running ones in evaluation. EKFAC's Lemma 1 makes this the optimal diagonal in whatever basis it is measured in. The factors and the eigenbasis are unchanged, so the §4.6 input factor still sets the basis. **Measured** at step 300: the scale-carrying eigen-column is **295-989x** the surrogate's on `vit_micro_cifar`, **80-106x** on `cct_2_3x2_cifar`; the shift column **0.96-1.03x**. At E14's `lambda` (step 2 000) it raises the first LayerNorm's divisor `s + lambda` from 1.00x to a median **2.45x** / **2.21x** `lambda`. Bit-identical to off on a network without a hooked norm layer. `ValueError` on the other three modes, and with `fisher_batch_samples` on a network with a `BatchNorm2d`, whose batch statistics cannot be recomputed from part of the batch. | E16 third amendment, `plan_floor_clip.md` §11, `tests/test_norm_exact_rescaling.py` |

### 4. Where the code and the paper disagree, and both reference repos take the code's side

**These are not this project's deviations.** They are properties of AdaFisher as published in code,
reproduced here on purpose. They matter for a paper because a reader will assume the paper's version.

| # | The paper says | Both reference repos do | Consequence, measured |
|---|---|---|---|
| 4.1 | Eq. (3): `gamma * old + (1-gamma) * new`, `gamma = 0.8` | `0.08 * old + 0.008 * new` (`gammas = [0.92, 0.008]`). The coefficients sum to 0.088, not 1. | The stored quantity settles at **1/115** of what it estimates, and **92%** of it is one minibatch. `audit_step.md` §4.2-§4.3 |
| 4.2 | Algorithm 1 recomputes the factors every step | `TCov = 100` | The curvature comes from one minibatch out of every hundred steps. `audit_step.md` §4.3 |
| 4.3 | The Fisher is an expectation over per-example gradients | The backward hook takes the gradient of the **batch-mean** loss | Each example's share of the curvature is divided by `batch^2`, i.e. **16 384** at batch 128. Confirmed directly: 56 measurements of the `1/batch^2` rule, all between 0.021 and 0.144 where it predicts 0.0625 (`plan_lambda_dominance.md` Part 6, E3). `EKFAC-pytorch` compensates for this; AdaFisher does not. |
| 4.4 | Prop. 3.1 for a normalisation layer: `S = sum_x s_x s_x^T` ("square-then-sum"), potentially full rank | `diag`'s `_h_batchnorm2d` / `_s_batchnorm2d` / `_h_layernorm` / `_s_layernorm` do "sum-then-square" | A different object, not a reduction-order difference. Left untouched in `diag`; the four new modes use Prop. 3.1 itself. `plan_lot5.md` §0.6 |
| 4.5 | — | `_h_conv2d` divides by `batch * S * P` **with a bias** and by `batch` alone **without one**, instead of `batch * S` in both cases | Two per-layer constant factors, harmless only because `diag`'s min-max erases per-layer scale. The new `Conv2d` code deliberately reproduces neither, so `compute_h_full`'s `Conv2d` diagonal is `P` times `compute_h_diag`'s with a bias and exactly `1/S` times it without one (measured: `1/25`, `1/16`, `1/9` on three shapes). Both are asserted as locked regressions. **Every convolution in every ResNet and CCT in this repository is `bias=False`**, so the `1/S` branch is the one that actually runs on the CNN benchmarks; until the full-repository audit only the `P` branch had a test. `plan_lot4.md` §0.2, `tests/test_full_factors_match_diag.py` |
| 4.6 | Prop. A.1's proof writes `H\|_nu = E[h h^T]` for the **normalised** activation `x_hat` — `y = gamma*x_hat + beta`, so `d/d gamma` pairs the output gradient with `x_hat`, not with the layer's input | The forward hook sees the input **before** normalisation, and that is what `augment_norm_input` pools — in `diag` and in all four Kronecker modes | **Not a reduction-order difference: a different activation.** Measured on `BatchNorm2d(8)` with a post-ReLU input and running statistics far from the batch statistics: `a_nu = A[0,0]` is **4.3991** from the raw input against **0.1296** from the train-mode `x_hat` (a factor of **33.9**), and the scale-shift coupling `A[0,1]` is **2.0377** against **4.3e-08**. The factor is **bit-identical in train and eval mode**, which is only possible because it never looks at the normalisation. For `diag` this is inherited from both reference repositories; for the four new modes it is this repository's own choice, and it was missing from this table until the full-repository audit. **Do not "fix" it by swapping in `x_hat`:** for `LayerNorm`, `x_hat` sums to zero across channels by construction, so `a_nu` would be exactly 0 — measured at **6.3e-15** on `LayerNorm(16)` — and the whole scale block would collapse to the damping term. Pinned, with these numbers, by `test_norm_input_factor_ignores_the_normalisation_itself` and `test_norm_input_factor_coupling_entry_is_far_from_the_normalised_value`. For `ekfac`/`tekfac`, what this factor also did to the **rescaling** `s*`/`Theta` is fixed by the opt-in `norm_exact_rescaling` (§3), which uses the true per-row gradient `[delta*x_hat, delta]` and leaves this factor, and so the eigenbasis, as they are. `audit_full_repo.md` finding 4 |

The joint effect of 4.1 and 4.3 is a factor of about **2 x 10^8** between the stored curvature and
the quantity it estimates. That is the single most important number for anyone comparing this code
with the paper, and it is why `Lambda = 1e-3` sits above **every** direction of **every** network
measured here. `plan_lambda_dominance.md` is the whole story.

### 5. Choices made inside the new modes, which have no AdaFisher counterpart

Relevant to a paper about the four new modes, not to a comparison with AdaFisher itself. Each is a
place where a reader of the source paper would expect something else.

| Choice | Instead of | Why |
|---|---|---|
| `ekfac`'s `s*` is an EMA of the **intra-batch** estimate | Either of the paper's two named variants ("from scratch every minibatch", or `-ra`'s squared batch-mean) | A deliberate third choice, so that all five modes share one EMA. `plan_lot2.md` §0.3 |
| A normalisation layer's input factor is a `2x2` matrix with a Frobenius-optimal scalar surrogate `a_nu` | Prop. 3.1's exact `H|_nu`, which is Hadamard- and not Kronecker-structured and does not fit the shared `kron(A,B)` machinery | Derived in `plan_lot5.md` §0.1-§0.2. The small `nu`-`beta` coupling term is **kept**, not forced to zero |
| SUA's input factor is the **centre offset** of every patch `extract_patches` already produces | `EKFAC-pytorch`'s own SUA, which pools the raw input independently | Only the centre-slice version stays row-aligned with the output factor's pooling when `stride != 1` or the padding is not "same". `plan_lot6.md` §0.3 |
| SUA treats the input factor as exactly block-diagonal across kernel offsets | The exact IAD+SH+SUA block, which has a rank-1 cross-offset coupling term | Matches `EKFAC-pytorch`'s own (also uncorrected) convention. A documented gap, not a bug. `plan_lot6.md` §0.1 |
| The bias direction under SUA is read back from the **centre** offset only | Any of the other `k_h*k_w - 1` offsets, which are expected to disagree | Inherited from `EKFAC-pytorch`; no theorem behind it. `plan_lot6.md` §0.4 |
| TEKFAC's trace-adaptive `lambda` (Eq. 3.5) and TKFAC's adaptive floor (Eq. 5.16) are **not implemented** | The papers' own conv-layer damping | A dimension-agnostic additive `lambda` is used instead. This is exactly what fix S1 of `plan_lambda_dominance.md` proposes to undo |
| `precondition()` is applied to the bias-corrected first moment `m_hat` (in code: the raw `m`, with `1/(1-beta^t)` folded into the step, except under `rescale_form="clip"`), not the raw gradient | Preconditioning `g` before averaging | Preconditioning first would be K-FAC + momentum, a different algorithm, and would break AdaFisher's own Table 1 |

### 6. Differences on the benchmark side

These do not touch the optimizer, but they change any number produced with it.

| Difference | Why |
|---|---|
| `NominalCosine` is **clamped** past `T_max`; torch's `CosineAnnealingLR` is periodic and climbs back up | Under a wall-clock budget a cheap arm overshoots `T_max`. Measured: `resnet20_cifar/adam` hit `lr = 0` at epoch 49, then trained 9 more epochs with the learning rate rising again, best accuracy decaying 88.82% -> 88.06%. `benchmarks/common/schedules.py` |
| `BudgetCosine` (`--lr-schedule budget`) anneals each arm over **its own** wall-clock budget | So every arm completes exactly one full cosine. The price is that such a run is no longer bit-reproducible |
| Checkpoint fractions are relative to the **nominal** trajectory, shared by every arm | So `ckpt_0.5` means the same amount of training in every arm. Using `max_epochs` put a budgeted arm's `ckpt_0.1` at 31% of its own run |
| The wall-clock-time protocol itself | AdaFisher's own (`adafisher_2405.16397.pdf` §5): one reference arm runs a fixed epoch count, every other arm gets its measured wall-clock time |

## Environment

A project-local venv, **not** FisherAdapTune's conda env (`fisheradaptune`) — that one exists only to
run FisherAdapTune's own crack-segmentation code, and is left untouched.

```bash
uv venv --python 3.11 .venv                        # already created; recreate if deleted
uv pip install -p .venv -e ".[dev,bench]"           # torch, torchvision, pytest, ruff, mypy
```

**The editable install's `sys.path` entry does not take effect in this sandboxed environment** —
`site.addpackage` silently no-ops on the generated `.pth` file for reasons not fully diagnosed (not a
project bug: a plain `PYTHONPATH=src` import works fine, see `docs/reports/plan_lot1.md` addendum in
the lot-1 completion notes). Two consequences, both already wired up:

- `pyproject.toml`'s `[tool.pytest.ini_options]` sets `pythonpath = ["src"]`, so `pytest` works
  without any extra flag.
- Each `benchmarks/models/<model>/bench.py` inserts both the repository root and `src` onto `sys.path`
  itself, so `python benchmarks/models/<model>/bench.py` works alongside `python -m
  benchmarks.models.<model>.bench`; running with `PYTHONPATH=src` explicitly also works.

If this turns out to be specific to this sandbox rather than the host machine in general, the editable
install may "just work" elsewhere — no need to route around it there too.

### Cluster sync — the real path (overrides `benchmarks/slurm/README.md`'s generic example)

`benchmarks/slurm/README.md` documents a generic
`<username>@fir.alliancecan.ca:~/projects/def-msh-ab/<username>/adafisher/` layout for any first-time
Alliance Canada setup. **The actual remote checkout for this project is elsewhere** — reachable via
the `rorqual` SSH config alias (host `rorqual3` at submission time, per `sacct`), at
`/home/blgr/new_adafisher/` (not `~/projects/def-msh-ab/blgr/adafisher/`, and the directory is named
`new_adafisher`, not `adafisher`). Use these paths, not the README's, when pulling results:

```bash
# SLURM logs (benchmarks/slurm/logs/*.out) — the fastest way to identify what a given JobID ran
rsync -av rorqual:/home/blgr/new_adafisher/benchmarks/slurm/logs/ \
      "benchmarks/slurm/logs/"

# results (csv/md/json/png only, no multi-GB checkpoints)
rsync -av --include='*/' --include='*.csv' --include='*.md' --include='*.json' \
      --include='*.png' --exclude='*' \
      rorqual:/home/blgr/new_adafisher/benchmarks/outputs/ \
      benchmarks/outputs/

# THE CHECKPOINTS ARE NOT IN THAT FILTER. Pull them explicitly, or fisher_ref's bridge finds a
# campaign's reports and none of its theta — which is how a 5-seed campaign silently reads as 1
# seed. This one line is what took available_seeds() from [0] to [0, 1, 2, 3, 4].
rsync -av --include='*/' --include='ckpt_*.pt' --exclude='*' \
      rorqual:/home/blgr/new_adafisher/benchmarks/outputs/seeds/ \
      benchmarks/outputs/seeds/

# checkpoints for one model only (large — pull selectively). Note the <group>: results are
# grouped by dataset (mnist / cifar10 / cifar100 / imagenet) on both sides.
rsync -av rorqual:/home/blgr/new_adafisher/benchmarks/outputs/<group>/<model>/ \
      benchmarks/outputs/<group>/<model>/

# check a specific JobID's name/state directly on the cluster instead of guessing from local logs
ssh rorqual "sacct -j <jobid> --format=JobID,JobName%30,Start,End,Elapsed,State,ExitCode"
```

Run these from the repository root on the laptop (the local checkout lives at
`/Users/baolgr/Documents/Projets/adafisher ` — note the trailing space in the directory name).

## Running the tests

```bash
.venv/bin/pytest tests/ -v                                    # everything: 1067 collected, 1026 passed and
                                                               #   41 skipped by default (40 gated on --runslow,
                                                               #   1 needing curvlinops). Measured 2026-09-21,
                                                               #   after E16's fourth amendment.
                                                               #   With --runslow, last measured 2026-09-20
                                                               #   before test_relative_damping.py: 862
                                                               #   passed, 1 skipped.
.venv/bin/pytest tests/test_diag_bitexact.py -v                # exit criteria 1 & 3 (bit-exactness)
.venv/bin/pytest tests/test_diag_eq4_semantics.py -v            # exit criterion 2 (Eq. 4 semantics)
.venv/bin/pytest tests/test_minmax_matches_official.py -v      # MinMaxNormalization vs. official repo
.venv/bin/pytest tests/test_full_factors_match_diag.py -v      # full A/B vs. diag's H_D/S_D (lots 2, 4-6)
.venv/bin/pytest tests/test_frobenius_dominance.py -v           # dominance/trace/orthogonality (lots 2-6)
.venv/bin/pytest tests/test_kfac_ekfac_precondition.py -v       # shapes, pi toggle, T_inv/T_eig cadence (lots 2, 4-6)
.venv/bin/pytest tests/test_tkfac_tekfac_precondition.py -v     # shapes, trace invariant, T_inv/T_eig/T_re cadence (lots 3-6)
.venv/bin/pytest tests/test_ekfac_tekfac_equiv.py -v            # EKFAC ↔ TEKFAC in the degenerate case
.venv/bin/pytest tests/test_conv2d_optimizer_smoke.py -v        # real hooks + step(), Conv2d net, 4 modes (lot 4)
.venv/bin/pytest tests/test_norm_layers_optimizer_smoke.py -v   # real hooks + step(), TinyMultiLayerNet, 4 modes,
                                                                 #   loss non-regression vs. diag (lot 5)
.venv/bin/pytest tests/test_conv2d_sua_optimizer_smoke.py -v    # real hooks + step(), Conv2d net, conv_sua=True,
                                                                 #   4 modes, + conv_sua inertness check (lot 6)
.venv/bin/pytest tests/test_equal_wallclock_bench.py -v         # time-budget harness: overshoot bound, max_steps
                                                                 #   guard, eigenbasis-mode overhead sanity (lot 7)
.venv/bin/pytest tests/test_cifar10_bench.py -v                 # models, the ViT pairing regression, fisher_batch_
                                                                 #   samples, both weight-decay rules, budget+eval (lot 8)
.venv/bin/pytest tests/test_eigh_conditioning.py -v              # the ekfac/tekfac eigh conditioning ridge:
                                                                 #   inertness, rank-deficient factors, CPU fallback
.venv/bin/pytest tests/test_ema_seed_first.py -v                  # the ema_seed_first knob: default
                                                                 #   inertness on a real trajectory, the
                                                                 #   exact constant-input property it
                                                                 #   exists for, all 5 modes; offline
.venv/bin/pytest tests/test_optimizer_robustness.py -v            # the awkward networks and loops: two
                                                                 #   parameter groups, a frozen bias, a module
                                                                 #   first reached after step 0, two backwards
                                                                 #   over one forward, a tied weight, and that
                                                                 #   precondition never aliases exp_avg
.venv/bin/pytest tests/test_relative_damping.py -v              # damping/damping_tau/hold_cap (E15): off
                                                                 #   and bit-identical by default; hold_cap ==
                                                                 #   lr*Lambda; a constant lambda_l != Lambda
                                                                 #   reproduces the single-lambda run at lambda_l;
                                                                 #   mean_curvature == dense mean eigenvalue;
                                                                 #   decoupled decay independent of lambda
.venv/bin/pytest tests/test_rescale_form.py -v                  # rescale_form (E16): "add" bit-identical by
                                                                 #   default; "floor" = max(s, lambda) and its
                                                                 #   f_tilde; the three per-module clip
                                                                 #   thresholds (exact count over the active
                                                                 #   coordinates, scale invariance, ema's bias
                                                                 #   correction, fixed's window median and its
                                                                 #   continuity); every CCT layer type
.venv/bin/pytest tests/test_e16_decisions.py -v                  # E16's decision script: an invalid or
                                                                 #   incomplete set of jobs is never read as a
                                                                 #   verdict; rule 5 qualifies rule 3
.venv/bin/pytest tests/test_e16_resnet50_calibration.py -v       # E16's ResNet-50 calibration: the grid rule
                                                                 #   on worked examples, the scan's coverage of
                                                                 #   E14's window, the cells' overrides
.venv/bin/pytest tests/test_norm_exact_rescaling.py -v           # norm_exact_rescaling (E16): x_hat is the
                                                                 #   layer's own normalised output, per-row
                                                                 #   gradients sum to the real grads, stored
                                                                 #   s*/Theta == autograd brute force (fp64);
                                                                 #   inert without norm layers; refusals
.venv/bin/pytest tests/test_e19_layer_damping_transfer.py -v     # E19: the bridge cell is built exactly as
                                                                 #   E16's add cell; every other cell has
                                                                 #   S1-b's settings; grids, seeds, shards;
                                                                 #   the decision script's rules 0b-4 and its
                                                                 #   refusals on synthetic runs
.venv/bin/pytest tests/test_sgdm_arm.py -v                        # the sgdm arm (E18): bit-identical to
                                                                 #   AdaFisherMulti with F~ = lam*I, torch
                                                                 #   SGD-momentum at the derived lr, the
                                                                 #   fallback for unhooked params, diag's bound
.venv/bin/pytest tests/test_eig_before_rescale.py -v              # the eig_before_rescale knob: default
                                                                 #   inertness, the basis s*/Theta is measured
                                                                 #   in vs the one precondition uses, and how
                                                                 #   far the difference reaches at two lambdas
.venv/bin/pytest tests/test_lr_schedule.py -v                    # the cosine under WCT: clamped nominal
                                                                 #   vs. budget-annealed, and both in the loop
.venv/bin/pytest tests/test_official_adafisher_arm.py -v          # the `official` arm: the authors' published
                                                                 #   optimizer, their class from their file, the
                                                                 #   single-gamma translation, and how closely
                                                                 #   diag reproduces it; all offline
.venv/bin/pytest tests/test_dataset_benches.py -v                 # CIFAR-100 + ImageNet-1K: the
                                                                 #   ImageFolder pipeline on a synthetic
                                                                 #   tree, both transform regimes, the
                                                                 #   seeded split, output_group routing,
                                                                 #   generated jobs per dataset; offline
.venv/bin/pytest tests/test_benchmark_models.py -v               # every model folder: parameter count,
                                                                 #   hooked-module inventory, "no parameter
                                                                 #   left un-updated", all 5 modes, the
                                                                 #   bench spec, checkpoints (step 1)
.venv/bin/pytest tests/test_fisher_ref_lot0.py -v                # the Fisher-drift campaign's lot 0
                                                                 #   (T0.1-T0.8): TF32/precision policy, the
                                                                 #   rvec/kron convention, reference mode and
                                                                 #   sample independence, probe sets, the
                                                                 #   layer-type registry over every model,
                                                                 #   the checkpoint bridge; all offline
.venv/bin/pytest tests/test_fisher_ref_lot1.py -v                # the campaign's lot 1 (T1, T2, T6): the three
                                                                 #   output-space roots, per-sample gradients vs.
                                                                 #   torch.func and vs. finite differences, the
                                                                 #   backward-hook pruning regression, the dense
                                                                 #   F/E_hat/B_l and their layout; all offline
.venv/bin/pytest tests/test_fisher_ref_lot5.py -v                # the campaign's lot 5 (T12): the state reader
                                                                 #   vs the optimizer's own applied preconditioner,
                                                                 #   all 5 modes x all 4 hooked layer types; diag vs
                                                                 #   the upstream optimizer; the norm-layer
                                                                 #   permutation (and that removing it fails); the
                                                                 #   re-warm's refusal below 10*TCov; conv_sua
                                                                 #   refused; k_lam inertness; the P1-py readings;
                                                                 #   TEKFAC's trace and dominance; all offline
.venv/bin/pytest tests/test_fisher_ref_lot3.py -v                # the campaign's lot 3: T5 (expand/reduce exact
                                                                 #   in their settings, and NOT in the other),
                                                                 #   T8-exp, T9-shared, T13-shared, B^exp vs brute
                                                                 #   force, pos_embed rows, the row checks, the fold
                                                                 #   engine, theorem orderings, diag_py; offline

# Every bench takes the same CLI; `python -m benchmarks.models.<model>.bench` and
# `python benchmarks/models/<model>/bench.py` are equivalent. The models are:
#   mnist_autoencoder  mlp_ln_mnist  cnn_gn_cifar  vit_micro_cifar
#   resnet20_cifar     cct_2_3x2_cifar  resnet50_cifar  vit_small_cifar
PYTHONPATH=src .venv/bin/python -m benchmarks.models.mnist_autoencoder.bench \
    --arms reference diag --epochs 5 --budget-mode epochs --no-minmax   # lot 1's non-regression demo
PYTHONPATH=src .venv/bin/python -m benchmarks.models.mnist_autoencoder.bench --epochs 20   # lot 7's WCT bench
# local smoke (seconds); the full 7-arm runs go to benchmarks/slurm/, see its README
PYTHONPATH=src .venv/bin/python -m benchmarks.models.vit_small_cifar.bench \
    --arms diag adam --epochs 2 --budget-mode epochs --train-subset 1024
PYTHONPATH=src .venv/bin/python -m benchmarks.models.resnet50_cifar.bench --epochs 50      # the real thing
# CIFAR-100 and ImageNet-1K: identical CLI, results under outputs/<dataset>/<model>/. CIFAR-100 is
# downloaded; ImageNet-1K must be staged by hand first (benchmarks/slurm/imagenet/README.md).
PYTHONPATH=src .venv/bin/python -m benchmarks.models.resnet20_cifar100.bench --epochs 50
PYTHONPATH=src .venv/bin/python -m benchmarks.models.cnn_gn_imagenet.bench \
    --arms diag adam --epochs 1 --budget-mode epochs --train-subset 2048 --no-allow-download
# trajectory checkpoints, the input steps 2-5 of plan_exp_draft.md consume; add
# --checkpoint-optimizer-state (opt-in, OFF by default) to also dump the optimizer's own state —
# its state_dict plus AdaFisherMulti's EMA'd Fisher factors keyed by module name (~1.8 GB for a
# full 6-model x 7-arm x 5-checkpoint campaign). Off means the payload is byte-for-byte what it
# was, which matters while cluster jobs run from $SLURM_SUBMIT_DIR.
PYTHONPATH=src .venv/bin/python -m benchmarks.models.cnn_gn_cifar.bench \
    --epochs 30 --checkpoints 0,0.01,0.1,0.5,1
# THE CAMPAIGN PROTOCOL: --lr-schedule budget, so every budgeted arm completes one full cosine
# instead of being cut off mid-anneal (or, if cheap, having its LR climb back up). Every generated
# SLURM job passes it; the CLI default is still `nominal` (now clamped). See "Things to watch".
PYTHONPATH=src .venv/bin/python -m benchmarks.models.cnn_gn_cifar.bench \
    --epochs 30 --budget-mode wct --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1
# the lam bracket on the stalling autoencoder bench (sweep_mnist_autoencoder_lam.sh on the cluster)
for L in 1e-5 1e-3 1e-1; do PYTHONPATH=src .venv/bin/python -m benchmarks.models.mnist_autoencoder.bench \
    --epochs 20 --budget-mode wct --lr-schedule budget --lam "$L" \
    --output-dir benchmarks/outputs/sweeps/mnist_autoencoder_lam/"$L"; done
```

`test_frobenius_dominance.py` now also covers TKFAC's trace preservation and TEKFAC's dominance
over TKFAC (lot 3), replays all four assertions on a toy `Conv2d` (lot 4) and a toy `BatchNorm2d`
(lot 5), and again at the SUA-consistent scale on that same toy `Conv2d` (lot 6), alongside the
K-FAC/EKFAC assertions below.

What each existing test guarantees:

| Test | Assertion | Source |
|---|---|---|
| `test_diag_bitexact::test_f_tilde_bitexact_across_steps` | `diag(minmax=False)` ≡ `AdaFisherBackbone._get_F_tilde`, at every EMA update, decoupled from the optimizer step's own op-order (model resynced after each step) | `adafisher.py:209-223` |
| `test_diag_bitexact::test_parameter_trajectory_close_on_linear_layer` | full trajectory within `1e-5` rtol on a `Linear`-only net (no BatchNorm feedback) | — |
| `test_diag_bitexact::test_parameter_trajectory_reasonable_on_full_net` | full trajectory within a loose `1e-2` rtol on `TinyMultiLayerNet` — loose on purpose, see below | — |
| `test_diag_eq4_semantics` | `diag(minmax=True)` matches Eq. (4) semantics, composed from two independently-validated pieces (not from the official repo's buggy EMA) | `AdaFisher.py:412`, `:431` |
| `test_minmax_matches_official` | `min_max_normalization` / `smart_detect_inf` bit-exact vs. the official repo, isolated from its EMA | `AdaFisher.py:13-45` |
| `test_full_factors_match_diag` | `diag(compute_h_full)` / `diag(compute_s_full)` match `compute_h_diag`/`compute_s_diag` (`allclose`); unsupported layers raise `NotImplementedError` | `plan_lot2.md` §0.1 |
| `test_frobenius_dominance::test_ekfac_dominates_kfac_in_frobenius_norm` | `‖F − F̃_EKFAC‖_F ≤ ‖F − F̃_KFAC‖_F` on a toy per-sample-exact `F` | EKFAC Thm 2/3, Appendix A.1 |
| `test_frobenius_dominance::test_eigenbases_are_orthogonal` | `Q_Aᵀ Q_A = I`, `Q_Bᵀ Q_B = I` | — |
| `test_frobenius_dominance::test_f_tilde_matches_precondition_for_{kfac,ekfac}` | the factored `precondition()` and the dense debug `f_tilde()` agree on a random direction | `plan_lot2.md` §0.5 |
| `test_kfac_ekfac_precondition` | output shapes (with/without bias), `pi` toggle, `T_inv`/`T_eig` amortisation cadence | `plan_lot2.md` §2.3 |
| `test_frobenius_dominance::test_tekfac_dominates_tkfac_in_frobenius_norm` | `‖F − F̃_TEKFAC‖_F ≤ ‖F − F̃_TKFAC‖_F` | TEKFAC Thm 3.1 |
| `test_frobenius_dominance::test_tkfac_trace_matches_exact_fisher` | `tr(F̃_TKFAC) = tr(F)` | TKFAC Thm 4.1 / Lemma 4.1 |
| `test_frobenius_dominance::test_f_tilde_matches_precondition_for_{tkfac,tekfac}` | the factored `precondition()` and the dense debug `f_tilde()` agree on a random direction | `plan_lot3.md` §0.6 |
| `test_tkfac_tekfac_precondition::test_trace_invariant_holds_under_real_gammas` | `tr(Phi_raw)=tr(Psi_raw)=delta` holds under the project's real `gammas`, not just the test-only `(1,1)` | `plan_lot3.md` §0.2 |
| `test_tkfac_tekfac_precondition` | shapes, `T_inv`/`T_eig`/`T_re` amortisation cadence | `plan_lot3.md` §2.1 |
| `test_ekfac_tekfac_equiv` | at `Φ=A, Ψ=B, δ=1`: TEKFAC ≡ EKFAC | TEKFAC Eq. 3.1-3.2 vs EKFAC §3.2 |
| `test_full_factors_match_diag::test_conv2d_pooled_patches_reconstruct_true_gradient` | pooled patches/gradients reconstruct the real `weight.grad`/`bias.grad` of a real `Conv2d` forward+backward exactly (not an approximation) | `plan_lot4.md` §0.7b |
| `test_full_factors_match_diag::test_conv2d_h_diag_scale_quirk_is_a_constant_factor` | `compute_h_full`'s `Conv2d` diagonal is `P`× the (unmodified) `compute_h_diag`'s — a verified, deliberate deviation, not a bug | `plan_lot4.md` §0.2 |
| `test_full_factors_match_diag::test_conv2d_{groups,dilation}_not_supported` | `groups≠1`/`dilation≠(1,1)` raise `NotImplementedError` | `plan_lot4.md` §0.4 |
| `test_frobenius_dominance::*_conv2d` | all four §6.1 assertions (EKFAC/KFAC, TEKFAC/TKFAC dominance; `tr(F̃_TKFAC)=tr(F)`; `Q` orthogonality; `f_tilde`/`precondition` consistency) replayed on a toy `Conv2d`, `F` built by an independent `torch.nn.functional.unfold`-based oracle | `plan_lot4.md` §0.7, §2.2 |
| `test_frobenius_dominance::test_unfold_oracle_matches_conv2d_identity` | the oracle's own patch layout matches `Conv2d.weight`'s, before trusting it | `plan_lot4.md` §0.7b |
| `test_{kfac_ekfac,tkfac_tekfac}_precondition::*_conv2d` | shapes, `T_inv`/`T_eig`/`T_re` cadence, trace invariant — `Conv2d` analogue of the `Linear` checks | `plan_lot4.md` §2.3 |
| `test_conv2d_optimizer_smoke` | a real `AdaFisherMulti` (hooks, `step()`) runs to completion, finite parameters, on a `Conv2d`+`Linear` net, all four modes | `plan_lot4.md` §2.4 |
| `test_full_factors_match_diag::test_full_{input,output}_factor_{batchnorm2d,layernorm}_*` | shapes, `A[1,1]==1` exactly, `A[0,0]` matches `augment_norm_input`'s own pooled statistic, symmetry | `plan_lot5.md` §0.2 |
| `test_full_factors_match_diag::test_layernorm_multi_dim_normalized_shape_not_supported` | a non-1-D `normalized_shape` raises `NotImplementedError` | `plan_lot5.md` §0.4 |
| `test_frobenius_dominance::*_norm` | all four §6.1 assertions replayed on a toy `BatchNorm2d(3)`, `F` built by an independent per-position pooling oracle (not calling `_pool_norm_layer`/`augment_norm_input`) | `plan_lot5.md` §0.1-§0.2, §2.2 |
| `test_{kfac_ekfac,tkfac_tekfac}_precondition::*_{batchnorm2d,layernorm}` | shapes, `T_inv`/`T_eig`/`T_re` cadence — normalisation-layer analogue of the `Linear`/`Conv2d` checks | `plan_lot5.md` §2.3 |
| `test_norm_layers_optimizer_smoke::test_optimizer_runs_on_full_net` | a real `AdaFisherMulti` (hooks, `step()`) runs to completion, finite parameters, on `TinyMultiLayerNet` (all four `SUPPORTED_MODULES`), all four modes | `plan_lot5.md` §2.4 |
| `test_norm_layers_optimizer_smoke::test_loss_non_regression_vs_diag` | `plan.md` §8's literal lot-5 exit criterion: from identical initial weights/minibatches, the new mode's loss does not diverge relative to `diag`'s | `plan.md` §8, lot 5 row |
| `test_full_factors_match_diag::test_augment_conv2d_input_sua_center_slice_equals_raw_pixel` | on `padding=(k-1)/2`, `stride=1`, the center offset of every patch is *exactly* the raw input pixel — the numeric version of the row-alignment derivation | `plan_lot6.md` §0.3 |
| `test_full_factors_match_diag::test_augment_conv2d_input_sua_row_alignment_with_output_grad` | row count matches `flatten_conv2d_output_grad`'s even when `H_in≠H_out` (`stride=2`, no padding) | `plan_lot6.md` §0.3 |
| `test_full_factors_match_diag::test_augment_conv2d_input_sua_matches_full_for_1x1_kernel` | SUA and the patch-based factor coincide exactly for a `1×1` kernel | `plan_lot6.md` §0.6 |
| `test_frobenius_dominance::*_conv2d_sua` | all four §6.1 assertions replayed at the SUA-consistent scale (`F_sua`, the center-pixel-pooled oracle) on the lot-4 toy `Conv2d` | `plan_lot6.md` §0.1-§0.4, §2.2 |
| `test_frobenius_dominance::test_f_tilde_matches_precondition_for_*_conv2d_sua` | `precondition()`'s per-kernel-offset application agrees with `f_tilde()`'s dense reconstruction independently at *every* offset; the returned bias matches only the center offset's own solve, and at least one other offset's solve is confirmed to disagree (non-vacuous) | `plan_lot6.md` §0.4 |
| `test_frobenius_dominance::test_conv2d_sua_matches_full_for_1x1_kernel` | `conv_sua=True`/`False` agree exactly on `precondition()`'s output for a `1×1` kernel, all four modes | `plan_lot6.md` §0.6 |
| `test_frobenius_dominance::test_sua_discards_offblock_frobenius_mass` | measured (not asserted): the fraction of the exact patch covariance's Frobenius mass in the cross-kernel-offset terms SUA discards — `kfac_conv_1602.01407.pdf` §5.1's "SUA loses information" finding, re-measured on this project's own toy conv | `plan_lot6.md` §0.7 |
| `test_frobenius_dominance::test_sua_memory_footprint_resnet18_scale` | `4609 → 513` input-factor width, `≈80.7×` byte reduction, on a ResNet-18-scale synthetic `Conv2d(512,512,3)` | `plan.md` §6.3/§7, `plan_lot6.md` exit criterion 2 |
| `test_{kfac_ekfac,tkfac_tekfac}_precondition::*_conv2d_sua` | shapes, `T_inv`/`T_eig`/`T_re` cadence, trace invariant, and that the cached factor/eigenbasis is `(C_in+1)×(C_in+1)` — not `(C_in·k_h·k_w+1)²` | `plan_lot6.md` §2.3 |
| `test_conv2d_sua_optimizer_smoke::test_optimizer_runs_on_conv_net_with_sua` | a real `AdaFisherMulti(conv_sua=True)` (hooks, `step()`) runs to completion, finite parameters, on a `Conv2d`+`Linear` net, all four modes | `plan_lot6.md` §2.4 |
| `test_conv2d_sua_optimizer_smoke::test_conv_sua_is_inert_without_any_conv2d_module` | `conv_sua=True` produces the exact same parameter trajectory as `conv_sua=False` on a `Conv2d`-free (pure `Linear`) network, all four modes | `plan_lot6.md` §2.4 |
| `test_equal_wallclock_bench::test_budget_is_respected_and_run_completes` | the dataset-agnostic time-budget harness runs to completion for all five modes on a tiny synthetic task; every loss finite; `elapsed_s` monotonic and overshoots the budget by at most that run's own largest single-batch duration, never more | `plan_lot7.md` §0.1, §1.3 |
| `test_equal_wallclock_bench::test_max_steps_guard_is_effective` | the `max_steps` defensive bound terminates a run independently of an effectively-unbounded time budget | `plan_lot7.md` §0.11 |
| `test_cifar10_bench::test_{resnet50,vit}_shapes_and_param_count` | 23.52 M / 2.69 M params, the hooked-module inventory, the `3x3`/no-max-pool CIFAR stem, Table 1's `MLP=4D` and `D/heads=64`, `qkv` is a hookable `nn.Linear` | `plan_lot8.md` §0.1, §0.2 |
| `test_cifar10_bench::test_every_parameter_is_updated` | after 2 real steps **no** parameter is bit-identical to its init, on ResNet-50 *and* ViT — the §0.3 regression (ViT used to leave `norm.{weight,bias}`, `head.{weight,bias}` untouched forever) | `plan_lot8.md` §0.3 |
| `test_cifar10_bench::test_pairing_matches_legacy_loop_on_reference_nets` | the new identity pairing selects exactly the `(module, weight, bias)` triples the removed index loop did, on the nets where that loop was correct | `plan_lot8.md` §0.3 |
| `test_cifar10_bench::test_fisher_batch_samples_*` | slices the example axis for 2-D/3-D/4-D inputs; bit-identical trajectory when the cap exceeds the batch (all five modes); bounds `ekfac`'s cached `h_bar` rows | `plan_lot8.md` §0.4 |
| `test_cifar10_bench::test_{de,}coupled_weight_decay_*` | `decoupled=True` differs from `wd=0` by exactly `-lr*wd*param`, the official `AdaFisherW._step` rule; the default coupled path still matches the reference `AdaFisher` at `wd=5e-4` | `plan_lot8.md` §0.6 |
| `test_cifar10_bench::test_budget_is_respected_and_eval_time_is_excluded` | lot 7's overshoot bound still holds, and a deliberately slow `eval_fn` is not charged to the budget | `plan_lot8.md` §0.7 |
| `test_cifar10_bench::test_{max_epochs_caps_a_cheap_arm,scheduler_steps_once_per_completed_epoch}` | the `--max-epoch-factor` cap; one scheduler step per *completed* epoch | `plan_lot8.md` §0.7 |
| `test_cifar10_bench::test_cutout_masks_one_square_region`, `::test_train_val_split_*` | Cutout masks one contiguous ≤16×16 region identically in every channel; the 45k/5k split is seeded, disjoint and exhaustive | `plan_lot8.md` §0.9 |
| `test_cifar10_bench::test_summary_and_csv_schema`, `::test_empty_arm_*` | the promised CSV/summary columns; an arm with zero steps does not break the report | `plan_lot8.md` §0.10 |
| `test_equal_wallclock_bench::test_eigenbasis_modes_pay_more_per_step_than_diag` | `ekfac`/`tekfac` (which project into and out of an eigenbasis) cost strictly more per step than `diag`, at equal step count — asserted; `kfac`/`tkfac` vs. `diag` reported for information only, since that comparison is not reliably ordered at toy scale (`diag`'s own multi-op min-max pipeline is comparably expensive to `kfac`/`tkfac`'s two cached-inverse matmuls) | `plan_lot7.md` §0.4, §5 point 5 |
| `test_optimizer_robustness::test_two_parameter_groups_step_each_module_once` | with the standard decay/no-decay split, the trajectory is `torch.equal` to the single-group one, all five modes — the fix for the double step (§2.5) | §2.5 |
| `test_optimizer_robustness::test_layernorm_without_bias_is_refused_when_the_optimizer_is_built` | `LayerNorm(d, bias=False)` raises `NotImplementedError` naming the module, in all five modes, instead of a shape error inside the mode | — |
| `test_optimizer_robustness::test_frozen_bias_on_a_trained_weight_is_refused_by_name` | a trainable weight with `bias.requires_grad = False` raises a message naming the module, instead of a bare `assert` (`diag`) or a matmul shape error (the other four) | — |
| `test_optimizer_robustness::test_module_first_reached_after_step_zero`, `::test_late_module_works_when_the_refresh_cadence_misses_its_first_step` | a module behind a conditional branch (stochastic depth) starts its running average on its first observation whatever the step, and gets an inverse/eigenbasis even when its first step is not a multiple of `T_inv`/`T_eig`. It used to raise `KeyError` in all five modes | — |
| `test_optimizer_robustness::test_two_backwards_over_one_forward_are_refused` | `ekfac`/`tkfac`/`tekfac` raise **one** clear error on the second backward, instead of a bare `KeyError` in two of them and a silent skip in the third. `::test_two_backwards_are_not_detected_by_the_two_cacheless_modes` pins the remaining gap: `diag` and `kfac` hold no per-example cache and still accept it silently | — |
| `test_optimizer_robustness::test_tied_weight_*` | a weight shared by two modules is owned by the **later** one, both modules still accumulate factors, and the shared weight is preconditioned once per sharing module. Pinned, not endorsed | — |
| `test_optimizer_robustness::test_precondition_returns_a_new_tensor_not_a_view_of_exp_avg` | the preconditioned direction never aliases the momentum buffer it was built from, all five modes, on a biased net (all four layer types) and a bias-free one — the case where aliasing is actually reachable | — |
| `test_eig_before_rescale::test_flag_off_is_bit_identical_to_not_passing_it` | the knob is inert by default, on a real trajectory, `ekfac` and `tekfac` | §3 |
| `test_eig_before_rescale::test_on_/test_off_the_rescaling_is_measured_in_...` | with the knob on, the basis `s*`/`Theta` was measured in is bit-identical to the one `precondition` uses; with it off, it is not. The second is the test that fires if the two orderings are ever swapped | EKFAC Lemma 1, TEKFAC Alg. 1 |
| `test_eig_before_rescale::test_the_difference_reaches_the_applied_step_only_once_lambda_is_lowered` | from one state, the ordering moves the applied step by `<1e-2` at `Lambda=1e-3` and `>100x` more at `Lambda=1e-8` | `audit_full_repo.md` finding 3 |
| `test_full_factors_match_diag::test_conv2d_h_diag_scale_quirk_without_bias_is_one_over_S` | the `bias=False` branch of §4.5, exactly `1/S` on three shapes — the branch every ResNet and CCT here actually runs | §4.5 |
| `test_full_factors_match_diag::test_conv2d_{non_zero_padding_mode,string_padding}_not_supported`, `::test_layernorm_without_bias_not_supported` | three layer configurations that used to be silently wrong (`padding_mode='reflect'`: 48% relative error) or to fail with an error naming neither the layer nor the option | — |
| `test_full_factors_match_diag::test_norm_input_factor_ignores_the_normalisation_itself`, `::test_norm_input_factor_coupling_entry_is_far_from_the_normalised_value` | §4.6 pinned with its numbers: the input factor is bit-identical in train and eval mode through real hooks while the output factor is not, and the `x_hat` alternative is degenerate (`6.3e-15`) | §4.6 |

**Known testing pitfalls.**
- **Two wall-clock tests fail now and then on a busy laptop, and that is not a regression.**
  `test_equal_wallclock_bench.py::test_budget_is_respected_and_run_completes` asserts that the timed
  phases cover at least 99 % of the elapsed clock, and `test_lr_schedule.py::
  test_loop_drives_budget_cosine_to_the_floor_by_the_budget` asserts a learning rate below 1e-6 at
  a measured time. Both depend on the machine's load. Measured on 2026-09-21: 1 to 3 failures in 31
  tests per run of those two files, **at the same rate on a clean export of `HEAD`** (`git archive`),
  and a full run with nothing else competing is green. Re-run them alone before suspecting a change.
- `addcdiv_` (the reference's fused update) and the unfused `div` + `add_(alpha=...)` this port uses
  (needed so `precondition` generalises beyond diagonal modes, `plan_lot1.md` §0.3) are **not**
  bit-identical in practice — empirically a few ULPs per step on a `Linear` layer, amplified to
  `O(1e-4)` over 6 steps by `BatchNorm2d`'s running-statistics feedback loop. This is why
  `test_f_tilde_bitexact_across_steps` resyncs the two models' parameters after every step (isolating
  `f_tilde` construction from update-rule op-order) rather than letting trajectories run free, and why
  the two trajectory tests use different, deliberately-scoped tolerances instead of one strict one.
- Dominance over the `diag` mode **is not a theorem** — `F̃_D` is not of the form `QDQᵀ` in the same
  basis, and min-max puts it on a different scale from `F`. It is *measured* after optimal scalar
  rescaling `c* = ⟨F,F̃⟩/‖F̃‖²_F`, and *reported*, never asserted. See `plan.md` §6.1.
- `torch.linalg.eigh` fixes neither the sign nor the ordering of eigenvectors: compare the **applied
  preconditioners**, never the bases.
- PyTorch code works in **`rvec`** convention (`d_out, d_in`), the literature in **`cvec`**. Hence
  `B⁻¹ G A⁻¹` in code against `A ⊗ B` in the papers. This is the central pitfall flagged by
  `papers/kfac_from_scratch_2507.05127.pdf` (Def. 1/2 and Def. 23); re-derived from scratch (not
  just cited) in `plan_lot2.md` §0.4, from the identity `vec_r(uvᵀ) = u ⊗ v`.
- `ekfac`'s `s*` is tracked as an EMA of the *intra-batch* estimate (Algorithm 1's own statistic),
  not either of the paper's two named variants ("from scratch every minibatch" or `-ra`'s squared
  batch-mean trick) — a deliberate third choice, justified in `plan_lot2.md` §0.3. Don't "fix" this
  to match `EKFAC-pytorch` literally; it is intentionally different, for a stated reason.
- The existing, **unmodified** diagonal `Conv2d` path (`_h_conv2d`) has a verified scale quirk —
  it divides by `batch*S*P` instead of `batch*S` — shared identically by both `AdaFisher` reference
  repositories, harmless only because `diag`'s default min-max normalisation erases per-layer
  constant scale factors. The lot-4 full-factor `Conv2d` code deliberately does **not** reproduce it
  (needed for TKFAC's `tr(F̃)=tr(F)` to hold on the real scale), so `compute_h_full`'s `Conv2d`
  diagonal is `P`× `compute_h_diag`'s by construction — asserted as a locked, constant-factor
  regression, not left untested. See `plan_lot4.md` §0.2.
- For `Conv2d`, `F` in the §6.1 toy tests means the **pooled-`(example, output-location)`** exact
  Fisher block, not a fully spatially-cross-correlated one — the literal generalisation of the
  `Linear`-case `F` under the same spatial-independence reading (TKFAC's own Assumption 4.1) the
  implementation itself relies on. Don't "strengthen" this into a fully-correlated oracle; that would
  test an unrelated, unproven property none of the four cited theorems make any claim about. See
  `plan_lot4.md` §0.7.
- Proposition 3.1's exact FIM for a normalisation layer is **Hadamard-, not Kronecker-, structured**
  (`FIM_ν = H|_ν ⊙ S`, `FIM_β = S` exactly) — a fact this file's own cheat sheet does not make
  explicit. The four new modes fit this into the shared `kron(A,B)` machinery via a
  Frobenius-optimal, `S`-independent scalar surrogate `a_ν` for the exact, full `H|_ν`
  (`A = [[a_ν, mean(z)],[mean(z), 1]]`, `z` = per-position channel-mean pre-activation) — not
  `plan.md` §5.2's "no invention" framing. The small `ν`-`β` coupling term `mean(z)` is **kept**,
  not forced to zero (Proposition A.1's own idealisation), for consistency with how `Linear`'s own
  bias column is already (also non-strictly) treated. Don't "fix" this to force exact block-diagonality
  — that would be a new, layer-type-specific special case, not a bug. See `plan_lot5.md` §0.1-§0.2.
- `diag.py`'s existing normalisation-layer formula (`_h_batchnorm2d`/`_s_batchnorm2d`/
  `_h_layernorm`/`_s_layernorm`) is a **further, pre-existing** deviation from Proposition 3.1
  ("sum-then-square" instead of the Proposition's "square-then-sum"), predating this project and
  left untouched. It is not compared against the new `compute_h_full`/`compute_s_full` for
  normalisation layers (unlike the `Linear`/`Conv2d` "diagonal of the full factor" checks) — the two
  paths do not estimate the same quantity even up to a reduction-order difference. See
  `plan_lot5.md` §0.6.
- SUA (`conv_sua=True`) is **IAD + SH + SUA only, not Theorem 4's IAD+SH+SUA+WD** — the output
  factor `B`/`Γ` stays full (channel-only, unchanged from lot 4) for every mode, SUA or not; "white
  derivatives" is deliberately not adopted. And even restricted to SUA alone, the exact
  IAD+SH+SUA Fisher block has a rank-1 cross-kernel-offset mean-coupling term
  (`M(j)M(j')`, nonzero for every `δ,δ'` pair, not just `δ=δ'`) that this implementation does
  **not** correct for — it treats the SUA input factor as exactly block-diagonal across kernel
  offsets, matching `EKFAC-pytorch::_precond_sua_ra`'s own (also uncorrected) convention. Don't
  "fix" this to add the Sherman-Morrison correction unasked; it is a documented, deliberate gap
  between the cited theorem and the tractable implementation, not a bug. See `plan_lot6.md` §0.1.
- The bias direction under SUA is broadcast into every kernel offset's slice before
  preconditioning, and the returned bias is read back from the **center** offset only — the other
  `k_h·k_w - 1` offsets' own solved bias value are *expected* to disagree (verified explicitly,
  not just assumed, in `test_frobenius_dominance.py`'s `_assert_precondition_matches_f_tilde_per_
  position`). There is no theorem behind this choice (inherited from `EKFAC-pytorch`); the center
  offset is used because it is the same reference offset the SUA input factor itself is built
  from, not because it is provably optimal. See `plan_lot6.md` §0.4.
- **A lot-8 smoke observation, not a finding.** In a 4-step ResNet-50 smoke run, `diag` (min-max ON,
  its paper-faithful default) took visibly larger early steps than the four Kronecker modes
  (train loss 17.6 vs ≈2.3 after 4 steps at `lr=1e-3`) — expected from Eq. (4)'s normalisation
  putting `F~_D` on `[0,1]+lambda`, so `m_hat/F~` can be ~`1/lambda` times the gradient early on.
  Four steps says nothing about 50 epochs under a cosine schedule; it is recorded so nobody
  re-discovers it and mistakes it for a bug. See `plan_lot8.md` §6.
- **`ekfac`/`tekfac` never call `linalg.eigh` on a raw factor.** Both go through
  `approximations/_eigh_utils.py::eigenbasis`, which adds a *relative* ridge
  (`1e-6 * mean(diag(M))`) before decomposing and falls back to the CPU solver with a
  `RuntimeWarning` if the device solver still raises. This is a **conditioning device, not
  damping**: `M + cI` shifts every eigenvalue by `c` and leaves every eigenvector and their
  ascending order unchanged, so the returned basis is mathematically identical and the
  "`lambda` only" rule above still holds — the applied scaling is still `s* + lambda` /
  `Theta + lambda`. Without it, cuSOLVER raises `_LinAlgError` ("ill-conditioned or has too many
  repeated eigenvalues") on an **exactly** rank-deficient factor, which killed two real cluster
  runs mid-training (`mlp_ln_mnist`, `resnet20_cifar`); `kfac`/`tkfac` were immune only because
  they damp *before* inverting. The measured instance: `mlp_ln_mnist`'s first `Linear` has
  `A` of shape `785x785`, rank 646, **136 exactly-zero eigenvalues**, `cond = 4e301` — MNIST's
  border pixels are identically zero in 130 of 784 coordinates. LAPACK on CPU solves the same
  matrices happily, so this is not reproducible on a CPU-only test runner; don't "simplify" the
  ridge away because the suite is green locally. See `_eigh_utils.py`'s docstring.
- **Checkpoint fractions are relative to the *nominal* trajectory (`--epochs`), not to
  `max_epochs`.** Under the WCT protocol every arm of a model shares one nominal length, so
  `ckpt_0.5` means the same amount of training in every arm and the dumps are comparable across
  them — which is the whole point of `plan_exp_draft.md` §7. Using `max_epochs` (which is
  `--max-epoch-factor` times longer) put a budgeted arm's `ckpt_0.1` at ~31% of its own run and
  made `ckpt_0.5` unreachable; measured on a real `cct_2_3x2_cifar` run. A budgeted arm need not
  reach the nominal length: `save_final` then pins the largest fraction to the arm's own end and
  marks that payload `scheduled=False`, and every payload carries `step`, `epoch` and the
  `total_steps` denominator so no fraction is ever ambiguous.
- **The cosine schedule is a protocol choice, and the default is not the campaign's.**
  `--lr-schedule` defaults to `nominal` (a shared `T_max = --epochs`, now **clamped** so the LR
  never rises past it); every generated *budgeted* SLURM job passes `--lr-schedule budget`, which
  anneals each arm over its own wall-clock budget. Use `budget` for anything comparing arms under
  the WCT protocol — with `nominal`, a cheap arm's LR climbs back up past `T_max` (torch's
  `CosineAnnealingLR` is periodic) and an expensive arm never reaches the floor, both measured on
  campaign 1 and quantified in `schedules.py`'s docstring. Two consequences worth remembering:
  the **reference arm always uses `nominal`** whichever mode is requested (it is unbudgeted — it is
  what *defines* the budget, so `elapsed / inf` would pin its LR at `base_lr`); and a `budget`-
  scheduled arm's LR depends on measured elapsed time, so it is **no longer bit-reproducible** —
  that is the price of the fix, and `nominal` remains available when determinism matters more.
  Don't "simplify" `NominalCosine` back to `CosineAnnealingLR`: the only difference is the clamp,
  and `tests/test_lr_schedule.py` asserts both the equality inside `T_max` and the divergence
  outside it.
- **The `mnist_autoencoder` stall is a real finding, not a broken run — and it is now confirmed on
  four fresh seeds.** All five modes land at `0.7072-0.7074 +- 0.0014` final validation loss while
  `adam`/`adamw` reach `0.4976 +- 0.0207`, losing on **0 of 4** seeds by the *same* `0.210`; the
  spread between the five modes is **5.6x smaller** than the spread between seeds. It is also the
  **only** model of the seven where the Fisher modes lose at all. All five Fisher modes
  freeze at MSE `~= 0.7085` within 2 epochs and stay there (parameters move 0.2-0.5% between 10%
  and 100% of training, having travelled `~1.0` from init) while `adam` descends to `0.4836`
  (travelling `25.8`). The architecture is 8 `Linear` layers with sigmoids throughout and a 30-dim
  bottleneck — exactly the saturating net the bench was chosen for — and `lam=1e-3` is the
  parameter setting the effective step. Before drawing any conclusion about the modes from this
  bench, read the status block's `lam` paragraph: a `{1e-5, 1e-3, 1e-1}` bracket has already been
  run and **`lam` is not the lever** — it moves the four Kronecker modes by under 1%, so don't
  re-run that sweep. And do not restate lot 7's "all five converge to an indistinguishable final
  loss" as evidence of convergence — lot 7 had no `adam` arm, so it could not see that the shared
  point is a bad one. **Two things are settled since, both measured** (`plan_authors_repro.md`):
  the authors' **published** optimizer, unmodified (the `official` arm), stalls at the same
  `0.70828` as `diag` — the two trajectories agree to `2.58e-7` relative over 2 200 steps, so this
  is **not** a porting defect; and **`lr` is not the lever either** — the authors' code ends within
  `7e-6` of `0.70828` at `lr` in `{1e-4, 1e-3, 1e-2, 1e-1}`, so don't run that bracket again.
  One nuance to keep: `0.70828` is the **network's** plateau, not AdaFisher's signature — `adam`
  lands on it too at `lr = 1e-2` and `1e-1`, and escapes only at `1e-3` and `1e-4`.
- SUA's input-factor construction (`augment_conv2d_input_sua`, center-slicing every patch
  `extract_patches` already produces) is deliberately **not** `EKFAC-pytorch`'s own SUA
  construction (pooling the raw, un-unfolded input independently over `(N, H_in, W_in)`) — the two
  coincide exactly for `stride=1`, `padding=(k-1)/2` (every `Conv2d` fixture in this codebase), but
  only this project's center-slice version stays row-aligned with the output factor's own pooling
  for `stride≠1`/non-"same" padding, which is what lets `ekfac`'s/`tekfac`'s existing intra-batch
  `s*`/`Θ` estimator be reused verbatim under SUA with zero `Conv2d`-specific branching. See
  `plan_lot6.md` §0.3, §4.2.

## Cheat sheet — the five rescalings and their bases

Notation: `A = E[h̄h̄ᵀ] ∈ R^{d_in×d_in}`, `B = E[δδᵀ] ∈ R^{d_out×d_out}`, `M` = direction to
precondition, shaped `(d_out, d_in)`, `λ` = damping.

| Mode | Basis | Rescaling | Applied form | Source |
|---|---|---|---|---|
| `diag` | parameter basis | `H'_D ⊗ S'_D + λ`, `H'_D`/`S'_D` = min-max of `diag(A)`/`diag(B)` | `M / (S'_D H'_Dᵀ + λ)` | AdaFisher Prop. 3.2, Eq. (4) |
| `kfac` | — (factored inversion) | `(A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹` | `B̃⁻¹ M Ã⁻¹`, `Ã=A+√(λπ)I`, `B̃=B+√(λ/π)I` | K-FAC §3.1 Eq. 2, §4.2; `π` §6.3/6.6 |
| `ekfac` | `Q_A ⊗ Q_B` (eigenvectors of `A`, `B`) | `s*_i = E[((Q_A⊗Q_B)ᵀ∇_θ)²_i]` — **optimal** diagonal in that basis | `Q_B [ (Q_Bᵀ M Q_A) / (s*+λ) ] Q_Aᵀ` | EKFAC §3.2, Lemma 1 (A.1), Thm 2/3 |
| `tkfac` | — (factored inversion) | `F = δ Φ ⊗ Ψ`, `δ=E[tr(Λ)tr(Γ)]`, `Φ=E[tr(Γ)Λ]/δ`, `Ψ=E[tr(Λ)Γ]/δ` (`tr Φ=tr Ψ=1`) | `(1/δ) Ψ̃⁻¹ M Φ̃⁻¹` | TKFAC Thm 4.1, Eq. (4.4)-(4.6), simplified (4.9) |
| `tekfac` | `Q_Φ ⊗ Q_Ψ` (eigenvectors of TKFAC's `Φ`, `Ψ`) | `Θ_ii = E[((Q_Φ⊗Q_Ψ)ᵀ∇_ω h)²_i]` — optimal diagonal in the TKFAC basis | `Q_Ψ [ (Q_Ψᵀ M Q_Φ) / (Θ+λ) ] Q_Φᵀ` | TEKFAC §3.1, Eq. (3.1)-(3.4), Thm 3.1 |

TEKFAC damping `λ`: fixed scalar for dense layers; `λ = max{tr(Θ_l), ϑ}/dim(Θ_l)` for conv layers
(Eq. 3.5), with `β = max_{l conv} max{tr(Θ_l),ϑ}/dim(Θ_l)` applied to a CNN's dense layers.

**Notation collision to watch:** TEKFAC calls `β₁`, `β₂` the EMA decays of `Θ` and of `(Φ,Ψ)`
(Eq. 3.6-3.8) — unrelated to Adam/AdaFisher's `β₁` (the momentum of `m^(t)`). In code: `beta_theta`,
`beta_factors`.

**Two EKFAC variants** (Alg. 1, §4): `EKFAC` estimates `s*` from **intra-batch** gradients;
`EKFAC-ra` maintains it as a running average of projected mini-batch gradients. Implementation
references: `EKFAC-pytorch/ekfac.py::_precond_intra` and `::_precond_ra`.

**SUA (`conv_sua=True`, lot 6)** is not a sixth mode — it is a `Conv2d`-only construction of `A`
for whichever of the four rescalings above is active, replacing the patch-based `A` (`d_in =
C_in·k_h·k_w[+1]`) by a channel-only one (`d_in = C_in[+1]`, `kfac_conv_1602.01407.pdf` p. 14).
`B` is unaffected; the rescaling formulas in the table above apply unchanged, just at the smaller
scale, and independently at each of the `k_h·k_w` kernel offsets (`plan_lot6.md` §0.4).

## Things to watch

- **The full factors do not exist in the original code.** `_ComputeHBarD` and `_ComputeSD` compute
  `einsum("ij,ij->j", X, X)` directly: **diagonals only**. `factors.py` must reintroduce full `A` and
  `B`. This is the main structural cost of the project.
- **Normalisation layers (lot 5: done).** For a normalisation layer, `A` is a `2×2` factor
  (`a_ν` a Frobenius-optimal scalar surrogate for Proposition 3.1's exact, full `H|_ν`; `a_β=1`
  exactly) and `B = S ∈ R^{C×C}` full-rank, so `F ≈ a_ν·S` for the `ν` block and `F = S` exactly for
  the `β` block, fitting the four new modes' shared `kron(A,B)` machinery — see `plan_lot5.md`
  §0.1-§0.2 for the Hadamard-vs-Kronecker derivation this required (not "no invention" as `plan.md`
  §1.5/§5.2's own cheat-sheet-level summary suggested). `S` follows **Prop. 3.1** (`Σ_x s_xs_xᵀ`,
  potentially full rank), **not** line 141 of the code (`outer(Σ_x s_x, ·)`, rank 1 → degenerate
  eigenbasis). The `diag` mode keeps the
  code's formula.
- **The dominant cost is not the eigendecomposition.** `O(d³)` amortises over `T_eig`; the per-step
  `O(d_in·d_out·(d_in+d_out))` projection does not. On the MNIST auto-encoder: ≈ 18 GFLOP/step of
  projection against ≈ 8.5 GFLOP/step for the model's own fwd+bwd, i.e. **≈ 2.1× overhead**. Any
  comparison budget must therefore be fixed in **wall-clock time**.
- **Memory, on conv layers (lot 6: done).** ResNet-18's largest factor has `d_in = 512·3·3+1 = 4609`,
  i.e. `A` = 85 MB in fp32 for a single layer (18 KB in `diag` mode). `conv_sua=True` brings
  `d_in → 513` (`≈80.7×` fewer bytes per factor), by pooling the *center* offset of every patch
  instead of the whole patch (`plan_lot6.md` §0.3) — adapted from, not copied from,
  `EKFAC-pytorch/ekfac.py::_precond_sua_ra`/`_to_kfe_sua` (that reference's own `_get_gathering_
  filter` is the *non*-SUA patch-gathering trick, already superseded here by lot 1's
  `extract_patches`; plan.md §7's table groups it with SUA prospectively, which this lot's own
  `plan_lot6.md` §0 corrects). ResNet/ViT-scale *training* is the separate, still-deferred CIFAR-10
  work (`plan.md` §6.3/§8's own "deferred, on request" row, distinct from lot 7's row — see
  `plan_lot6.md` §3 and `plan_lot7.md` §4 for that boundary); the equal-wall-clock bench it needs
  as a prerequisite is now done (lot 7).
- **Empirical Fisher, not the Fisher.** All modes use the true-label `grad_output`. This is AdaFisher's
  choice, EKFAC's (§4), and TKFAC/TEKFAC's. Limitations documented in
  `papers/empirical_fisher_limits_1905.12558.pdf`; test cases distinguishing type-I/II/empirical in
  `papers/kfac_from_scratch_2507.05127.pdf` (cheat sheet §6). Do not open this line of work unasked.
- **`fisher_ref/capture.py` deliberately does *not* use `register_full_backward_hook`.** It is the
  hook `adafisher_modes/optimizer.py` uses and the obvious one to copy, and copying it puts a
  **silent** hole in the campaign's references. That hook fires from the module's *input*-side node,
  so a module whose `grad_input` is not needed by the `inputs=` requested of `torch.autograd.grad`
  is pruned out of the backward graph and never fires. Measured on `Linear -> LayerNorm -> ReLU ->
  Linear` in fp64: `grad(out, params, grad_outputs=V)` fires all three modules when nothing requires
  grad (with a `UserWarning` — that fallback is what makes the bug look absent), and only the last
  two as soon as the input requires grad. On `mlp_ln_mnist` the missing module is the first
  `Linear`, **25 120 of 26 634 parameters**; the symptom is not a crash but zero columns in `U`,
  leaving `F` symmetric, PSD and wrong. `capture.py` takes `g` from a *tensor* hook on the module
  output (`output.register_hook`, checked to survive `ReLU(inplace=True)` on that output and a
  residual reuse of it) and the dense driver backpropagates with `inputs=[batch]`, never
  `inputs=params`. `test_backward_hook_pruning_regression` pins both halves. The two packages'
  hooks are not meant to converge — see `plan_exp_lot1.md` §0.1.
- **Parameters outside the four hooked module types (lot 8: fixed).** `optimizer.py`'s `step()` no
  longer walks parameters and modules by position and shape (the reference's `adafisher.py:275-307`
  bookkeeping, plus `_check_dim`, both removed): it uses an identity-keyed `(id(param) -> module)`
  map built in `_prepare_model`. The old loop ran exactly `len(self.modules)` iterations and spent
  one on each *unpaired* parameter, so a ViT's `cls_token`/`pos_embed` silently cost the last two
  modules — **the final `LayerNorm` and the whole classification head were never updated**, with
  finite gradients and a still-decreasing loss. Measured, then fixed, in `plan_lot8.md` §0.3; the
  equivalence to the old loop on every net where it *was* correct is asserted, not assumed
  (`test_pairing_matches_legacy_loop_on_reference_nets`). Don't "restore" the positional loop.
- **`ekfac`/`tekfac` hold every layer's cached input batch at once.** All forward hooks fire before
  any backward hook, so on a `TCov` step the `_cached_h_bar` of *every* module is live
  simultaneously: **3.37 GB** across ResNet-50 at batch 128 (1.51 GB even with `conv_sua=True`),
  against 272 MB for ViT-S/4 — measured in `plan_lot8.md` §0.4. `fisher_batch_samples=k` (lot 8)
  caps this by estimating the factors from the first `k` examples of the batch only; example-axis
  slicing is what keeps `h` and `delta` row-paired across two independently-firing hooks, which the
  intra-batch `s*`/`Theta` estimators require. It changes the *estimator*, not the applied gradient,
  is `None` (inert) by default, and must be reported alongside any curve produced with it.
- **`conv_sua=True` is not optional on ResNet-50.** Measured on the real architecture
  (`plan_lot8.md` §0.4-§0.5): it takes the widest input factor from `d_in = 4608` to `512`, total
  `A` storage from 498.5 to 96.1 MB, the per-refresh `eigh`/`inverse` from 495 to 122 GFLOP and the
  per-step projection from 288 to 107 GFLOP. Consequence for any CNN result: the four non-diagonal
  arms measure those modes **under SUA**, including `plan_lot6.md` §0.1's documented, uncorrected
  cross-offset gap — a ResNet result is not a statement about full-patch EKFAC.
- **The four `*_imagenet` benches at 32 px measure ImageNet32, not ImageNet-1K.** `cnn_gn`,
  `vit_micro`, `resnet20` and `cct_2_3x2` are 32x32-native architectures; they run on ImageNet
  downsampled to 32x32 (Chrabaszcz, Loshchilov & Hutter 2017, arXiv:1707.08819 §2 — the *whole*
  image squashed, aspect ratio not preserved, no crop), which is what lets them stay structurally
  identical to their CIFAR counterparts and therefore comparable with them arm by arm. It is a
  cited benchmark, not a shortcut, but it is **not** the 224 px task, and published ImageNet
  numbers are not a reference for it. Only `resnet50_imagenet` and `vit_small_imagenet` run at
  224 px. The pad-4-crop + flip this project applies on top of the squash is its own choice, not
  that paper's.
- **ImageNet-1K cannot be downloaded, and the pre-resized tree is not optional in practice.**
  ILSVRC-2012 needs an accepted image-net.org agreement; `allow_download` is structurally inert in
  `build_imagenet_loaders` and a missing tree is a `FileNotFoundError` naming the expected layout.
  `benchmarks/slurm/imagenet/stage_imagenet.sh` builds both trees (`imagenet/` and `imagenet32/`)
  from the two official tars. The 32 px benches *work* without `imagenet32/` — `imagenet_root`
  falls back and `Resize((32,32))` is in the pipeline either way, the two being numerically
  identical — but then every epoch decodes 1.28 M full-resolution JPEGs to feed a model whose
  forward pass takes microseconds. Build it.
- **Every bench is grouped by its dataset, and a run directory's name is a label, not an
  identifier.** `Benchmark.output_group` (`mnist`/`cifar10`/`cifar100`/`imagenet`) routes results
  to `outputs/<group>/<model>/` and jobs to `slurm/<group>/`; all twenty models are grouped, and
  the eight original result trees were migrated. Three things make that safe, and all three are
  asserted:
  * `fisher_ref/checkpoints.py::discover_runs` reads **both** layouts, grouped and flat, so a tree
    that has not moved still resolves. It descends only into a directory named after a group some
    bench actually *declares*, so `_calibration` and `outputs/lot8_cifar10/` cannot be mistaken
    for one.
  * The network is rebuilt from `manifest.json`'s `config["model"]` — written by `runner.main`
    since this change — not from the directory name, because `--output-dir` is free-form.
    `_LEGACY_BENCH_OF_DIRECTORY` covers the manifests written before that. The case that forced
    it: A2's BatchNorm variant lives in `outputs/cifar10/cnn_gn_cifar_bn/` with no
    `benchmarks/cnn_gn_cifar_bn/` folder, and it is exactly the run protocol P2 compares the
    GroupNorm one against — its 28 checkpoints were unloadable until then.
  * `RunRef.model` stays the **directory** label, so `cnn_gn_cifar` and `cnn_gn_cifar_bn` remain
    two rows and do not collapse into one.
  `outputs/seeds/<model>/seed<n>/` is deliberately left ungrouped: its axis is the seed.
- **The results rsync recipe does not include checkpoints.** Its filter is `*.csv *.md *.json
  *.png`, so a campaign's reports arrive and its `ckpt_*.pt` do not — and `discover_runs` then
  reports a 5-seed campaign as 1 seed, with no error anywhere. Measured: the extra-seed campaign
  had completed and `available_seeds()` still returned `[0]` until the checkpoints were pulled
  explicitly (the second rsync line under "Cluster sync" above). **217 runs / 840 checkpoints
  under `outputs/seeds/` is the current, correct state**, after the seed axis was completed to all
  seven arms; `available_seeds()` is `[0,1,2,3,4]` on the five regime-A models and `[0,1,2]` on the
  two regime-B ones, for every arm.
- **A new `benchmarks/models/<model>/` folder must be added to four registries, not one.** The folder list
  is the registry for *discovery*, but four tests exist precisely to fail when a folder appears
  without its contract: `tests/test_benchmark_models.py::EXPECTED`,
  `tests/test_fisher_ref_lot0.py::EXPECTED_LAYER_TYPES`,
  `tests/test_fisher_ref_lot1.py::EXPECTED_UNCOVERED`, and
  `benchmarks/slurm/generate_jobs.py::WALLTIME` (whose absence makes the generator raise `KeyError`
  *after* overwriting half the directory — `tests/test_dataset_benches.py` checks it up front).
- **`ViT-S/4` is an adaptation, not a paper variant.** `vit_2010.11929.pdf` Table 1 defines only
  Base/Large/Huge, all 224px/patch-16. `benchmarks/models/vit_small_cifar/model.py`'s 32x32 configuration
  (`patch 4`, `D=192`, `depth 6`, `heads 3`) keeps that table's `MLP = 4D` and `D/heads = 64` and
  cites §3.1's Eq. (1)-(4) for the *structure* only. Never attribute the configuration itself to the
  paper. And the paper's own caveat applies (§1, §3.1 "Inductive bias", §4.2): a from-scratch ViT on
  45k images lands well below a CIFAR CNN — the bench compares seven optimizers on one fixed
  architecture, not CIFAR-10 accuracy records.

## Benchmarks

| Bench | Model | Role |
|---|---|---|
| primary | 8-layer MNIST auto-encoder, `784-1000-500-250-30` + untied symmetric decoder | historical K-FAC / EKFAC bench (`ekfac_1806.03884.pdf` §4.1); small enough for all five modes |
| secondary (lot 8) | CIFAR-10 from scratch: **ResNet-50** (CIFAR stem, 23 520 842) and **ViT-S/4** (2 693 578), 7 arms — the five modes + `Adam` + `AdamW` | `plan.md` §8's formerly-deferred row. Code and SLURM jobs done (`benchmarks/{resnet50,vit_small}_cifar/`, `benchmarks/slurm/`); **run twice**: first pre-fix (archived), then re-run under the fixed protocol in campaign 2 (`docs/reports/campaign2_cifar100_imagenet.md` §6) |
| Fisher-drift campaign (step 1) | `mlp_ln_mnist` (A1, 26 634), `cnn_gn_cifar` (A2, 24 458), `vit_micro_cifar` (A3, 21 098), `resnet20_cifar` (B2, 269 722), `cct_2_3x2_cifar` (B1, 283 723) | `plan_exp_draft.md` §4's five new models, each on the shared harness with all 7 arms and `--checkpoints`. Implemented and smoke-tested; **no convergence run yet** |

| CIFAR-100 (new) | the six image-classification architectures with `num_classes=100`: `cnn_gn_cifar100` (30 308), `vit_micro_cifar100` (24 068), `resnet20_cifar100` (275 572), `cct_2_3x2_cifar100` (295 333), `resnet50_cifar100` (23 705 252), `vit_small_cifar100` (2 710 948) | same protocol, same hyperparameters, same 7 arms as CIFAR-10. **Run in campaign 2**, one seed (`docs/reports/campaign2_cifar100_imagenet.md`) |
| ImageNet-1K (new) | `resnet50_imagenet` (25 557 032) and `vit_small_imagenet` (22 050 664) at the native **224 px**; `cnn_gn_imagenet` (88 808), `vit_micro_imagenet` (53 768), `resnet20_imagenet` (334 072), `cct_2_3x2_imagenet` (411 433) on **downsampled ImageNet32** | AdaFisher's own ImageNet transforms; 7 arms each. Dataset staged; **run in campaign 2**, one seed — `docs/reports/campaign2_cifar100_imagenet.md` |

The workspace contains no pre-existing classification bench: `FisherAdapTune` only ships crack
segmentation (SAM2 / SegFormer) and a synthetic example.

## Code conventions

- On existing code: change only what was asked. No refactoring, no unsolicited cleanup.
- One iteration at a time, following the lot breakdown in `docs/reports/plan.md` §8.
- Scale convention: AdaFisher's everywhere (mean over `batch × spatial`), **not** `EKFAC-pytorch`'s
  (which multiplies `grad_output` by the batch size and divides by `num_locations`).
