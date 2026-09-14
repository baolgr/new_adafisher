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
> rather than only argued. `benchmarks/mnist_autoencoder.py::_make_optimizer` was generalized to
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

> **CIFAR-100 and ImageNet-1K: code done, datasets partially staged, no run yet.** The six
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
> is a **new, optional, empty-by-default** field routing both the results and the jobs into a
> one-level dataset subdirectory; the eight existing benches leave it empty and keep the flat
> `outputs/<model>/` layout, because that is the path `fisher_ref/checkpoints.py` discovers runs at.
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
> **Nothing has trained.** The `imagenet/` `--time` values are **ESTIMATED**, flagged as such in
> every header, which is why each of those six models also has a calibration job; and **ImageNet-1K
> itself is not staged** — it requires an accepted image-net.org agreement and 155 GB, so
> `benchmarks/slurm/imagenet/stage_imagenet.sh` builds the trees from the two official tars and
> `benchmarks/slurm/imagenet/README.md` is the file to read before submitting anything there.
> CIFAR-100 *is* downloaded, into `benchmarks/data/cifar-100-python/`, and the pipeline is
> verified end to end on it (`cnn_gn_cifar100`, 3 arms, 1 epoch: loss starts at `ln(100) = 4.605`,
> accuracy at chance, checkpoints written to `outputs/cifar100/<model>/<arm>/` with a `(100, 64)`
> head). Those smoke outputs were then deleted, so `outputs/{cifar100,imagenet}/` are empty.

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
> **The A1 run happened (job 21077038) and its headline is a problem with `N`, not with the code**
> (`plan_exp_lot1.md` §6). Peak device **6.10 GB** against 6.0 predicted, so the 10 GB decision is
> confirmed on the real thing. A full type-2 `F` at `N=4000`, `P=26 634` costs **11 s**; one
> `26 634^2` `eigvalsh` costs **1156 s** and does **not** thread (19.8 GFLOP/s at 1 thread, 18.6 at
> 8, 21.6 on the cluster's 16) — so budget these jobs from `(4/3)P^3/2e10` per spectrum and never
> ask for cores to speed one up. The job was cancelled at its 00:50:00 limit inside the *per-block*
> loop (the widest block is `25 120^2`, 83% of a full-`P` decomposition on its own) having produced
> everything else, and **wrote nothing**, since its only write was at the end — now incremental,
> cheapest-block-first, with `A1_BLOCK_SPECTRA=0` to skip. **The finding:** source gap
> `||E_hat - F||/||F|| = 0.760`, train/val `0.822`, and the §3.4 noise floor `0.872` on halves of
> `N/2 = 2000`, i.e. `sigma_N = 0.436` for one `N=4000` estimate and `0.617` for two independent
> ones under the null. Q1 sits at `1.74 sigma_N` and HF1 at `1.33x` its null: **at this `N` the
> campaign's two headline quantities are the same order as the estimator's own noise.** `N` was
> chosen in §2.2 for *rank* (`N(C-1) >= P`), which is far weaker than accuracy; `sigma_N = 0.1`
> needs `N ~ 76 000`, which at 11 s a build is ~3.5 min of GPU. **Set `N` from the noise floor, not
> from the rank condition, and report every gap with its floor.** Also measured: `rank(E_hat) =
> 3511/26 634` — bounded by `N` by construction, so a damped inverse of `E_hat` reads `lambda I` on
> 23 123 directions (Kunstner et al. arXiv:1905.12558, in this repository's own numbers); and
> `rank(F) = 18 564/26 634`, whose deficiency is roughly half explained by MNIST's 130 identically-
> zero pixels (the same fact that crashed cuSOLVER in campaign 1, `_eigh_utils.py`) and otherwise
> open.

## Working language

All code, comments, docstrings, reports and documentation are written in **English**, to the standard
of a well-maintained academic repository. French output on explicit request only.

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
adafisher /
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
│   │                              #   factor (lot 6: done, plan_lot6.md §0.3, §1.1)
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
│                                  #   every benchmarks/<model>/ folder: exact parameter count,
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
│                                  # campaign 1 audit: test_lr_schedule.py (new) — NominalCosine
│                                  #   identical to torch's inside T_max and clamped outside it,
│                                  #   BudgetCosine's shape/monotonicity/floor, both through the
│                                  #   real loop; all offline
├── benchmarks/                   # step 1 (plan_exp_step1.md): a package — one shared harness in
│   │                             #   common/, one folder per tested model. The five flat modules
│   │                             #   of lots 1/7/8 are gone; every line of them landed here.
│   ├── common/
│   │   ├── data.py               # mnist()/cifar10()/cifar100(), imagenet()/imagenet32() (an
│   │   │                         #   ImageFolder tree, staged by hand), Cutout,
│   │   │                         #   seeded_train_val_split
│   │   ├── loop.py               # train_under_budget() = lot 7's + lot 8's loops merged (D2),
│   │   │                         #   evaluate(), sync(), prepare_batch/metric_fn conventions;
│   │   │                         #   + the optional per-batch `lr_schedule` hook
│   │   ├── schedules.py          # NominalCosine (clamped past T_max) / BudgetCosine (anneals over
│   │   │                         #   the arm's own WCT budget) — the `--lr-schedule` protocol fix
│   │   ├── optimizers.py         # HParams, ARMS (5 modes + adam/adamw + reference),
│   │   │                         #   build_optimizer(arm, model, hp)
│   │   ├── records.py            # StepRecord/EpochRecord/ArmResult + csv/summary/plot/manifest
│   │   │                         #   writers, lot 8's column schema unchanged
│   │   ├── checkpoints.py        # --checkpoints 0,0.01,0.1,0.5,1 -> ckpt_<frac>.pt (D5); fractions are
│   │   │                         #   relative to the NOMINAL trajectory, shared across arms
│   │   └── runner.py             # Benchmark, build_parser(bench), main(bench),
│   │                             #   discover_benchmarks() — the folder list *is* the registry
│   ├── mnist_autoencoder/        # model.py bench.py  (migrated, lots 1+7)
│   ├── mlp_ln_mnist/             # model.py bench.py  (A1, new: 26 634 params)
│   ├── cnn_gn_cifar/             # model.py bench.py  (A2, new: 24 458; --norm gn|bn)
│   ├── vit_micro_cifar/          #          bench.py  (A3, new: 21 098; imports ViTCIFAR from
│   │                             #   vit_small_cifar/model.py — D1's one cross-folder import)
│   ├── resnet20_cifar/           # model.py bench.py  (B2, new: 269 722, option-A shortcuts)
│   ├── cct_2_3x2_cifar/          # model.py bench.py  (B1, new: 283 723, AdaFisher's own model)
│   ├── resnet50_cifar/           # model.py bench.py  (migrated verbatim, lot 8: 23 520 842)
│   ├── vit_small_cifar/          # model.py bench.py  (migrated verbatim, lot 8: 2 693 578)
│   ├── <arch>_cifar100/          # 6 folders: the six architectures above with num_classes=100.
│   │                             #   bench.py only — the model is imported from <arch>_cifar/
│   ├── <arch>_imagenet/          # 6 folders, num_classes=1000. Four read ImageNet downsampled to
│   │                             #   32x32 (imagenet32) and reuse the CIFAR model unchanged;
│   │                             #   resnet50_imagenet has its OWN model.py (the paper's ImageNet
│   │                             #   7x7/s2+maxpool stem, 25 557 032) and vit_small_imagenet is
│   │                             #   ViTCIFAR configured as ViT-S/16 @224 (22 050 664)
│   ├── outputs/<model>/<arm>/    # results indexed by what they measure, not by lot (D6);
│   │                             #   the old outputs/lot7_*, outputs/lot8_* are left untouched.
│   │                             #   outputs/{cifar100,imagenet}/<model>/<arm>/ for the new
│   │                             #   datasets (Benchmark.output_group); the 8 original models
│   │                             #   deliberately stay flat — fisher_ref/checkpoints.py reads that
│   ├── slurm/                    # 181 generated sbatch jobs (20 models x (1 calibration + 7
│   │   ├── cifar100/             #   per-arm) + 18 grouped + 1 lam sweep + 1 seeds job) + READMEs
│   │   └── imagenet/             #   + generate_jobs.py, which enumerates model folders. One
│   │                             #   subdirectory per dataset, matching outputs/; imagenet/ also
│   │                             #   holds the hand-written stage_imagenet.sh
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

**`precondition` receives the bias-corrected first moment `m̂ = m/(1−β₁ᵗ)`, not the raw gradient.**
The original update `param.addcdiv_(exp_avg, F_tilde, -lr/bc)`
(`reference_repos/FisherAdapTune/scripts/adafisher.py:273`) is an element-wise division, hence
`θ ← θ − α·F̃_D⁻¹m̂` for a diagonal `F̃_D`. The four new modes are not diagonal in the parameter basis,
so the operator must be applied to `m̂`. The `diag` mode then reduces to `addcdiv_` exactly.
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
               decoupled_weight_decay=False)# lot 8: True = the official AdaFisherW rule
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
- Each `benchmarks/<model>/bench.py` inserts both the repository root and `src` onto `sys.path`
  itself, so `python benchmarks/<model>/bench.py` works alongside `python -m
  benchmarks.<model>.bench`; running with `PYTHONPATH=src` explicitly also works.

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

# checkpoints for one model only (large — pull selectively)
rsync -av rorqual:/home/blgr/new_adafisher/benchmarks/outputs/<model>/ \
      benchmarks/outputs/<model>/

# check a specific JobID's name/state directly on the cluster instead of guessing from local logs
ssh rorqual "sacct -j <jobid> --format=JobID,JobName%30,Start,End,Elapsed,State,ExitCode"
```

Run these from the repository root on the laptop (the local checkout lives at
`/Users/baolgr/Documents/Projets/adafisher ` — note the trailing space in the directory name).

## Running the tests

```bash
.venv/bin/pytest tests/ -v                                    # everything (lots 1-8 + step 1 + campaign
                                                               #   lots 0-1: 341 tests, + 10 marked slow and
                                                               #   1 needing curvlinops, run with --runslow)
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
.venv/bin/pytest tests/test_lr_schedule.py -v                    # the cosine under WCT: clamped nominal
                                                                 #   vs. budget-annealed, and both in the loop
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

# Every bench takes the same CLI; `python -m benchmarks.<model>.bench` and
# `python benchmarks/<model>/bench.py` are equivalent. The models are:
#   mnist_autoencoder  mlp_ln_mnist  cnn_gn_cifar  vit_micro_cifar
#   resnet20_cifar     cct_2_3x2_cifar  resnet50_cifar  vit_small_cifar
PYTHONPATH=src .venv/bin/python -m benchmarks.mnist_autoencoder.bench \
    --arms reference diag --epochs 5 --budget-mode epochs --no-minmax   # lot 1's non-regression demo
PYTHONPATH=src .venv/bin/python -m benchmarks.mnist_autoencoder.bench --epochs 20   # lot 7's WCT bench
# local smoke (seconds); the full 7-arm runs go to benchmarks/slurm/, see its README
PYTHONPATH=src .venv/bin/python -m benchmarks.vit_small_cifar.bench \
    --arms diag adam --epochs 2 --budget-mode epochs --train-subset 1024
PYTHONPATH=src .venv/bin/python -m benchmarks.resnet50_cifar.bench --epochs 50      # the real thing
# CIFAR-100 and ImageNet-1K: identical CLI, results under outputs/<dataset>/<model>/. CIFAR-100 is
# downloaded; ImageNet-1K must be staged by hand first (benchmarks/slurm/imagenet/README.md).
PYTHONPATH=src .venv/bin/python -m benchmarks.resnet20_cifar100.bench --epochs 50
PYTHONPATH=src .venv/bin/python -m benchmarks.cnn_gn_imagenet.bench \
    --arms diag adam --epochs 1 --budget-mode epochs --train-subset 2048 --no-allow-download
# trajectory checkpoints, the input steps 2-5 of plan_exp_draft.md consume; add
# --checkpoint-optimizer-state (opt-in, OFF by default) to also dump the optimizer's own state —
# its state_dict plus AdaFisherMulti's EMA'd Fisher factors keyed by module name (~1.8 GB for a
# full 6-model x 7-arm x 5-checkpoint campaign). Off means the payload is byte-for-byte what it
# was, which matters while cluster jobs run from $SLURM_SUBMIT_DIR.
PYTHONPATH=src .venv/bin/python -m benchmarks.cnn_gn_cifar.bench \
    --epochs 30 --checkpoints 0,0.01,0.1,0.5,1
# THE CAMPAIGN PROTOCOL: --lr-schedule budget, so every budgeted arm completes one full cosine
# instead of being cut off mid-anneal (or, if cheap, having its LR climb back up). Every generated
# SLURM job passes it; the CLI default is still `nominal` (now clamped). See "Things to watch".
PYTHONPATH=src .venv/bin/python -m benchmarks.cnn_gn_cifar.bench \
    --epochs 30 --budget-mode wct --lr-schedule budget --checkpoints 0,0.01,0.1,0.5,1
# the lam bracket on the stalling autoencoder bench (sweep_mnist_autoencoder_lam.sh on the cluster)
for L in 1e-5 1e-3 1e-1; do PYTHONPATH=src .venv/bin/python -m benchmarks.mnist_autoencoder.bench \
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

**Known testing pitfalls.**
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
- **The `mnist_autoencoder` stall is a real finding, not a broken run.** All five Fisher modes
  freeze at MSE `~= 0.7085` within 2 epochs and stay there (parameters move 0.2-0.5% between 10%
  and 100% of training, having travelled `~1.0` from init) while `adam` descends to `0.4836`
  (travelling `25.8`). The architecture is 8 `Linear` layers with sigmoids throughout and a 30-dim
  bottleneck — exactly the saturating net the bench was chosen for — and `lam=1e-3` is the
  parameter setting the effective step. Before drawing any conclusion about the modes from this
  bench, read the status block's `lam` paragraph: a `{1e-5, 1e-3, 1e-1}` bracket has already been
  run and **`lam` is not the lever** — it moves the four Kronecker modes by under 1%, so don't
  re-run that sweep. And do not restate lot 7's "all five converge to an indistinguishable final
  loss" as evidence of convergence — lot 7 had no `adam` arm, so it could not see that the shared
  point is a bad one.
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
- **The CIFAR-100 and ImageNet checkpoints are invisible to `fisher_ref/checkpoints.py`.** The
  bridge walks `outputs/<model>/<arm>/` and was left unchanged: a dataset group directory is inert
  there — not mistaken for a model, no crash, no runs returned (asserted in
  `tests/test_dataset_benches.py`). The Fisher-drift campaign is defined over the eight original
  models (`plan_exp_draft.md` §4); pointing it at another dataset is a deliberate change to that
  bridge, not a side effect of adding a bench.
- **`Benchmark.output_group` must stay empty for the eight original models.** It routes results to
  `outputs/<group>/<model>/` and jobs to `slurm/<group>/`; `fisher_ref/checkpoints.py` discovers
  runs at `outputs/<model>/<arm>/` and several campaign reports cite that path, so moving MNIST and
  CIFAR-10 under `mnist/` + `cifar10/` would break the bridge for no gain. New datasets get
  subdirectories; the existing eight stay flat. `tests/test_dataset_benches.py` asserts both halves.
- **A new `benchmarks/<model>/` folder must be added to four registries, not one.** The folder list
  is the registry for *discovery*, but four tests exist precisely to fail when a folder appears
  without its contract: `tests/test_benchmark_models.py::EXPECTED`,
  `tests/test_fisher_ref_lot0.py::EXPECTED_LAYER_TYPES`,
  `tests/test_fisher_ref_lot1.py::EXPECTED_UNCOVERED`, and
  `benchmarks/slurm/generate_jobs.py::WALLTIME` (whose absence makes the generator raise `KeyError`
  *after* overwriting half the directory — `tests/test_dataset_benches.py` checks it up front).
- **`ViT-S/4` is an adaptation, not a paper variant.** `vit_2010.11929.pdf` Table 1 defines only
  Base/Large/Huge, all 224px/patch-16. `benchmarks/vit_small_cifar/model.py`'s 32x32 configuration
  (`patch 4`, `D=192`, `depth 6`, `heads 3`) keeps that table's `MLP = 4D` and `D/heads = 64` and
  cites §3.1's Eq. (1)-(4) for the *structure* only. Never attribute the configuration itself to the
  paper. And the paper's own caveat applies (§1, §3.1 "Inductive bias", §4.2): a from-scratch ViT on
  45k images lands well below a CIFAR CNN — the bench compares seven optimizers on one fixed
  architecture, not CIFAR-10 accuracy records.

## Benchmarks

| Bench | Model | Role |
|---|---|---|
| primary | 8-layer MNIST auto-encoder, `784-1000-500-250-30` + untied symmetric decoder | historical K-FAC / EKFAC bench (`ekfac_1806.03884.pdf` §4.1); small enough for all five modes |
| secondary (lot 8) | CIFAR-10 from scratch: **ResNet-50** (CIFAR stem, 23 520 842) and **ViT-S/4** (2 693 578), 7 arms — the five modes + `Adam` + `AdamW` | `plan.md` §8's formerly-deferred row. Code and SLURM jobs done (`benchmarks/{resnet50,vit_small}_cifar/`, `benchmarks/slurm/`); the runs themselves are **not done** — no lot-8 convergence number exists yet |
| Fisher-drift campaign (step 1) | `mlp_ln_mnist` (A1, 26 634), `cnn_gn_cifar` (A2, 24 458), `vit_micro_cifar` (A3, 21 098), `resnet20_cifar` (B2, 269 722), `cct_2_3x2_cifar` (B1, 283 723) | `plan_exp_draft.md` §4's five new models, each on the shared harness with all 7 arms and `--checkpoints`. Implemented and smoke-tested; **no convergence run yet** |

| CIFAR-100 (new) | the six image-classification architectures with `num_classes=100`: `cnn_gn_cifar100` (30 308), `vit_micro_cifar100` (24 068), `resnet20_cifar100` (275 572), `cct_2_3x2_cifar100` (295 333), `resnet50_cifar100` (23 705 252), `vit_small_cifar100` (2 710 948) | same protocol, same hyperparameters, same 7 arms as CIFAR-10. Implemented and tested offline; **no convergence run yet** |
| ImageNet-1K (new) | `resnet50_imagenet` (25 557 032) and `vit_small_imagenet` (22 050 664) at the native **224 px**; `cnn_gn_imagenet` (88 808), `vit_micro_imagenet` (53 768), `resnet20_imagenet` (334 072), `cct_2_3x2_imagenet` (411 433) on **downsampled ImageNet32** | AdaFisher's own ImageNet transforms; 7 arms each. Code and SLURM jobs done; **the dataset is not staged and nothing has run** — `--time` values are ESTIMATED |

The workspace contains no pre-existing classification bench: `FisherAdapTune` only ships crack
segmentation (SAM2 / SegFormer) and a synthetic example.

## Code conventions

- On existing code: change only what was asked. No refactoring, no unsolicited cleanup.
- One iteration at a time, following the lot breakdown in `docs/reports/plan.md` §8.
- Scale convention: AdaFisher's everywhere (mean over `batch × spatial`), **not** `EKFAC-pytorch`'s
  (which multiplies `grad_output` by the batch size and divides by `num_locations`).
