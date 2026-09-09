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
├── benchmarks/
│   ├── mnist_autoencoder.py      # lot 1: diag vs. reference AdaFisher, MNIST auto-encoder;
│   │                             #   lot 7 generalized _make_optimizer to all five modes
│   │                             #   (--optimizer kfac|ekfac|tkfac|tekfac|all), plan_lot7.md §1.1
│   ├── equal_wallclock_bench.py  # lot 7 (new): equal-wall-clock-budget convergence bench, all
│   │                             #   five modes, on the same MNIST auto-encoder — plan_lot7.md §1.2
│   ├── cifar10_classification.py # lot 8 (new): the 7-arm CIFAR-10 bench (5 modes + Adam + AdamW),
│   │                             #   the WCT budget protocol, per-model HPs — plan_lot8.md §1.4
│   ├── cifar10_models.py         # lot 8 (new): ResNet-50 (CIFAR stem) + ViT-S/4, from their own
│   │                             #   papers, hook-safe by construction — plan_lot8.md §0.1, §0.2
│   ├── cifar10_data.py           # lot 8 (new): 45k/5k split, crop+flip+Cutout — plan_lot8.md §0.9
│   └── slurm/                    # lot 8 (new): 16 generated sbatch jobs (2 calibration + 14 arms)
│                                 #   + README + generate_jobs.py — plan_lot8.md §0.11, §1.5
├── docs/reports/
│   ├── plan.md                   # overall design plan (lots 1-8)
│   ├── plan_lot1.md              # lot-1 implementation plan, with the 3 ABC corrections
│   ├── plan_lot2.md              # lot-2 implementation plan: full factors, s* estimator, rvec
│   │                             #   derivation, shared weight/bias plumbing
│   ├── plan_lot3.md              # lot-3 implementation plan: un-normalized-numerator EMA,
│   │                             #   damping derivation, TEKFAC's reuse of TKFAC's factors
│   ├── plan_lot4.md              # lot-4 implementation plan: Conv2d (KFC) for the four
│   │                             #   non-diagonal modes, the _h_conv2d scale-quirk finding,
│   │                             #   groups=1/dilation=(1,1) scope, the toy-conv oracle design
│   ├── plan_lot5.md              # lot-5 implementation plan: Proposition 3.1's Hadamard (not
│   │                             #   Kronecker) FIM structure for normalisation layers, the
│   │                             #   Frobenius-optimal scalar surrogate for H|_nu, why zero
│   │                             #   approximations/*.py changes were needed
│   ├── plan_lot6.md              # lot-6 implementation plan: the SUA approximation (IAD+SH+SUA,
│   │                             #   not Theorem 4's IAD+SH+SUA+WD), the center-slice-of-patch
│   │                             #   input-factor construction and why it row-aligns with the
│   │                             #   output factor for any stride/padding, the block-diagonal-
│   │                             #   across-kernel-offsets precondition application and its bias
│   │                             #   convention
│   ├── plan_lot7.md              # lot-7 implementation plan: equal-wall-clock-budget semantics
│                                 #   (checked per-batch, overshoot bound not undershoot — an
│                                 #   empirically-corrected design point, §5.5), the dataset-agnostic
│                                 #   harness enabling an offline test, the fwd+bwd/step timing split
│                                 #   giving §6.3's "≈2.1x" claim its first empirical measurement
│   └── plan_lot8.md              # lot-8 implementation plan: the two networks read off their own
│                                 #   papers and why neither is a torchvision/timm import, the
│                                 #   measured ViT parameter-pairing bug, the measured hook-memory
│                                 #   footprint at CIFAR scale, AdaFisher's WCT budget protocol,
│                                 #   the Alliance Canada SLURM path
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
- Standalone scripts (`benchmarks/mnist_autoencoder.py`) insert `src` onto `sys.path` themselves at
  the top of the file, or run with `PYTHONPATH=src` explicitly.

If this turns out to be specific to this sandbox rather than the host machine in general, the editable
install may "just work" elsewhere — no need to route around it there too.

## Running the tests

```bash
.venv/bin/pytest tests/ -v                                    # everything (lots 1-8: 167 tests)
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
PYTHONPATH=src .venv/bin/python benchmarks/mnist_autoencoder.py --optimizer both --epochs 5
PYTHONPATH=src .venv/bin/python benchmarks/mnist_autoencoder.py --optimizer all --epochs 5    # all 5 modes, epoch-fixed
PYTHONPATH=src .venv/bin/python benchmarks/equal_wallclock_bench.py --time-budget 60          # lot 7's §6.3 bench
# lot 8 — local smoke (seconds); the full 7-arm runs go to benchmarks/slurm/, see its README
PYTHONPATH=src .venv/bin/python benchmarks/cifar10_classification.py \
    --model vit_small --arms diag adam --epochs 2 --budget-mode epochs --train-subset 1024
PYTHONPATH=src .venv/bin/python benchmarks/cifar10_classification.py \
    --model resnet50 --arms diag kfac ekfac tkfac tekfac adam adamw --epochs 50   # the real thing
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
- **`ViT-S/4` is an adaptation, not a paper variant.** `vit_2010.11929.pdf` Table 1 defines only
  Base/Large/Huge, all 224px/patch-16. `benchmarks/cifar10_models.py`'s 32x32 configuration
  (`patch 4`, `D=192`, `depth 6`, `heads 3`) keeps that table's `MLP = 4D` and `D/heads = 64` and
  cites §3.1's Eq. (1)-(4) for the *structure* only. Never attribute the configuration itself to the
  paper. And the paper's own caveat applies (§1, §3.1 "Inductive bias", §4.2): a from-scratch ViT on
  45k images lands well below a CIFAR CNN — the bench compares seven optimizers on one fixed
  architecture, not CIFAR-10 accuracy records.

## Benchmarks

| Bench | Model | Role |
|---|---|---|
| primary | 8-layer MNIST auto-encoder, `784-1000-500-250-30` + untied symmetric decoder | historical K-FAC / EKFAC bench (`ekfac_1806.03884.pdf` §4.1); small enough for all five modes |
| secondary (lot 8) | CIFAR-10 from scratch: **ResNet-50** (CIFAR stem, 23.52 M) and **ViT-S/4** (2.69 M), 7 arms — the five modes + `Adam` + `AdamW` | `plan.md` §8's formerly-deferred row. Code and 16 SLURM jobs done (`benchmarks/cifar10_classification.py`, `benchmarks/slurm/`); the runs themselves are **not done** — no lot-8 convergence number exists yet |

The workspace contains no pre-existing classification bench: `FisherAdapTune` only ships crack
segmentation (SAM2 / SegFormer) and a synthetic example.

## Code conventions

- On existing code: change only what was asked. No refactoring, no unsolicited cleanup.
- One iteration at a time, following the lot breakdown in `docs/reports/plan.md` §8.
- Scale convention: AdaFisher's everywhere (mean over `batch × spatial`), **not** `EKFAC-pytorch`'s
  (which multiplies `grad_output` by the batch size and divides by `num_locations`).
