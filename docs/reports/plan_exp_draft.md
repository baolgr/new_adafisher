# Plan — Fisher reference bench: the drift of the approximations from the true and from the exact empirical Fisher, per layer type

*v1, September 2026. The adaptation of `plan_exp_draft_v0.md` to the models, runs and checkpoints
this repository actually produces (`plan_exp_step1.md`). `plan_exp_draft_v0.md` is kept verbatim as
the historical draft; where the two disagree, **this file is the one the code follows**, and §0
lists every disagreement.*

*It also supersedes the **simplified rewrite** that lived at the repository root as
`plan_exp_draft.md` (added by `6dd981c`, recoverable with `git show 6dd981c:plan_exp_draft.md`,
absent from the working tree since 2026-09-11). That file is where the campaign's **scope cuts**
were decided and where the section numbers quoted by today's code comments come from; §0.11-§0.13
carry both forward.*

Status conventions (as in `plan_exp_step1.md`): `[ESTABLISHED]` = checked against the primary
source; `[DERIVED]` = arithmetic done here from published numbers or from the built models;
`[ESTIMATE]` = order of magnitude, not measured, to be confirmed by the lot that needs it;
`[MEASURED]` = produced by a run in this repository, with the artifact named.

---

## 0. What v1 changes, and why (read this before §1)

v0 was written before `benchmarks/` existed as a package. It specified its own models, its own
training trajectories and its own checkpointing. All three now exist, are tested, and have been run
on the cluster; v1 therefore consumes them instead of rebuilding them.

| # | v0 said | v1 says | Why |
|---|---|---|---|
| 0.1 | A1-A4, B1-B5, C1-C2 defined in the plan | the **eight `benchmarks/<model>/` folders**, with their *measured* parameter counts (§1) | `plan_exp_step1.md` built five of them from their papers and migrated three; their counts, hooked-module inventories and "no parameter left un-updated" property are locked by `tests/test_benchmark_models.py` |
| 0.2 | "5 checkpoints per trajectory" as a thing to build | `--checkpoints 0,0.01,0.1,0.5,1` already writes `outputs/<model>/<arm>/ckpt_<frac>.pt`, and **campaign 1 has written them** | `benchmarks/common/checkpoints.py`; each payload carries `step`, `epoch`, `seed`, `total_steps` and a `scheduled` flag |
| 0.3 | trajectories: "AdamW (neutral reference) and AdaFisher" | the arms **`adamw`** and **`diag`**; seed 0 additionally has `kfac`, `ekfac`, `tkfac`, `tekfac`, `adam` | the WCT protocol runs all seven arms for seed 0; the extra-seed job (`slurm/train_v0_seeds.sh`) runs only those two, exactly §3.5's two trajectories |
| 0.4 | "3 training seeds per model, 5 in regime A" | seed 0 in `outputs/<model>/`, seeds 1-4 (regime A) and 1-2 (regime B) in `outputs/seeds/<model>/seed<n>/` | submitted, **in flight, not yet available**: every consumer must tolerate missing seeds and report the seeds it actually found, never assume 5 |
| 0.5 | P2 reads the *upstream* AdaFisher's internal state through an adapter, for `diag` only | P2 reads **`AdaFisherMulti`'s own state**, for all five modes | this repository's `diag` is bit-exact against `FisherAdapTune/scripts/adafisher.py` (`tests/test_diag_bitexact.py`), which is what the adapter existed to guarantee; the open question v0 §8.1 made P2 wait on is **settled** |
| 0.6 | checkpoints contain everything P2 needs | they contain **`model_state_dict` only** — no optimizer state | P2 must re-warm the EMA from θ; §3.2 derives how long that takes and what it costs in fidelity |
| 0.7 | lot 0 = conventions + probes + registry, exit criterion "T1-T6 pass" | lot 0 keeps its three deliverables, gains the **checkpoint bridge**, and its exit criterion becomes the new **T0.1-T0.8**; T1-T6 move to the lots that build their prerequisites (§8, §9) | T1 needs a dense `F`, T3-T5 need K-FAC, T6 needs per-sample gradients — none of which lot 0 delivers. v0's exit criterion was unsatisfiable as written |
| 0.8 | "1/4 of an H100" = `2g.20gb`, 20 GB | true, but the **training** jobs run on `1g.10gb`; regime A needs ≥ 20 GB and `mlp_ln_mnist` needs ~17 GB for `F` + `eigh` alone | `slurm/train_v0_seeds.sh` requests `h100_1g.10gb`. An analysis job that inherits that line silently cannot form `F` (§2.2) |
| 0.9 | one `parquet` per (model, checkpoint) | **CSV + a JSON metadata sidecar**, same writers' style as `benchmarks/common/records.py` | the repository has no `pandas`/`pyarrow` dependency and the cluster install is `--no-index`; `parquet` is an optional upgrade, not a prerequisite |
| 0.10 | A4 (GPT-micro), B3 (RoBERTa-LoRA), B4 (ViT-Tiny), B5 (ResNet-18), C1, C2 | A4/B3 **deferred** (`plan_exp_step1.md` §7 defers them explicitly); B4's role — "do the conclusions hold at a realistic width `d=192`?" — is served by the **existing `vit_small_cifar`**, which is `d=192`; B5/C1/C2 stay optional | no new benchmark folder is created by this campaign before its regime-A and regime-B maps are stable |

### 0.11 The scope cuts of the simplified rewrite are kept, as a priority order

The superseded root plan's §3 trimmed v0 deliberately, and v1 keeps those decisions — not by
deleting the material (it is the reference arithmetic, and §§2-6 below are where it lives) but by
marking what the campaign does **first**:

| v0 offers | v1's core | v1's option |
|---|---|---|
| 3 regimes | **A** and **B** | C (matrix-free) — deferred to the last lot, as it buys one model at the price of stochastic metrics and no type-2 K-FAC |
| 8 metrics | **M1, M5, M7, M8** (+ M3 in regime A, where a Cholesky is affordable) | M2, M4, M6 — they need Lanczos machinery to answer what M1 and M5 already answer |
| 3 sources | **type-2, empirical** | `MC_K`, `K in {1,4,16}` — it tests HF6 only |
| 8 hypotheses | **HF1, HF2, HF3, HF7** | HF4 is largely settled analytically here already (`plan_lot5.md` §0.1: a normalisation layer's FIM is Hadamard-, not Kronecker-structured); HF5 needs inter-layer blocks, i.e. regime A only; **HF8 becomes a reporting rule** (§3.4, §10.3), not a hypothesis |

### 0.12 "Reuse, do not rebuild" — and the one place it does not apply

The root plan's §2 states the rule: *any structure that already exists in `adafisher_modes` is
called, not reimplemented; the new code builds references, sources and metrics only*. v1 keeps it,
with one stated exception and a test that makes the exception cheap. `adafisher_modes` computes
EMA'd, damped, hook-driven factors at training precision and has no **source** parameter (it is
empirical by design — `CLAUDE.md`, "Empirical Fisher, not the Fisher"), while P1 needs the structure
alone, in fp64, from type-2 vectors. So P1 owns its structures, and **T13** pins each of them to
`adafisher_modes`' own at the degenerate setting (one update, identity EMA, empirical source,
`lambda = 0`) — which turns "our K-FAC" and "the optimizer's K-FAC" into one measured statement
rather than two implementations nobody compared. Everything else the repository already has —
hooks and per-layer factor extraction (`factors.py`), the five rescalings with their dense
`f_tilde()`, the toy exact-Fisher oracles in `tests/test_frobenius_dominance.py`, the whole
`benchmarks/` harness — is called, never rewritten.

### 0.13 Citation map, for the section numbers already in the code

Comments in `benchmarks/` and `tests/` cite `plan_exp_draft.md` by the **root file's** numbering.
They are not rewritten (that would be churn in files this campaign must not touch); this is the map:

| Cited as | Meant | In v1 |
|---|---|---|
| §4, "Models" | the model table | **§1** |
| §7, "Checkpoints" / "Protocols and invariants" | the checkpoint schedule, P1/P2, the invariants | **§3.5** (checkpoints), **§3.2** (protocols), **§8** (invariants) |
| §8, "Steps" | the step table; `plan_exp_step1.md` is its step 1 | **§9**, whose lots 0-6 continue after that step 1 (lot 0 is carved out of the root plan's step 2) |
| §5, "Metrics" | M1/M5/M7/M8 | **§5**, same ids, with §0.11's priority |

Nothing in `src/adafisher_modes/` and nothing in `benchmarks/` changes for lot 0. The campaign is a
**reader** of the benchmark artifacts, in its own top-level package `fisher_ref/`.

---

## 1. The models the campaign actually runs on

All counts `[MEASURED]`, asserted in `tests/test_benchmark_models.py`. `P_max` for regime A is
`sqrt(B/24)` (§2.2): **28 867** on 20 GB, 40 824 on 40 GB, and only **20 412** on the 10 GB MIG.

| Id | Folder | `P` | Regime | Layer types covered | Role |
|---|---|---|---|---|---|
| A1 | `mlp_ln_mnist` | 26 634 | **A** (`F` 5.68 GB, +`eigh` ≈ 17.0 GB) | `Linear` without sharing, `LayerNorm`, head | the control: the case K-FAC theory is cleanest on |
| A2 | `cnn_gn_cifar` | 24 458 | **A** (4.79 / 14.4 GB) | `Conv2d` (spatial sharing), `GroupNorm` (`--norm bn` for the BN variant), head | AdaFisher's own toy bench (App. B.2) but against an exact Fisher |
| A3 | `vit_micro_cifar` | 21 098 | **A** (3.56 / 10.7 GB) | patch-embed `Conv2d`, fused `qkv`, attention output, MLP, `LayerNorm`, `pos_embed`, head | a clean *reduce* setting (Eschenhagen et al.) |
| B0 | `mnist_autoencoder` | 2 837 314 | **B** | 8 `Linear`, no sharing, sigmoid throughout | the project's **primary** bench, and the one with the measured five-mode stall (§3.6) |
| B1 | `cct_2_3x2_cifar` | 283 723 | **B** | conv tokenizer, transformer, `LayerNorm`, sequence pooling (`attention_pool`), `pos_embed` | **AdaFisher's own model** (its Table 2) |
| B2 | `resnet20_cifar` | 269 722 | **B** | `Conv2d`, `BatchNorm2d` | the small ResNet of AdaFisher's CNN family |
| B4 | `vit_small_cifar` | 2 693 578 | **B**, streamed | transformer at a realistic width (`d=192`, 6 blocks, 3 heads) | checks that A3/B1's conclusions survive a realistic width |
| C | `resnet50_cifar` | 23 520 842 | **C** only | deep `Conv2d` + `BatchNorm2d` | optional; matrix-free only |

Not built, and deliberately not built by this campaign: **A4** `gpt_micro_char` (the *expand*
framework, a language-model head) and **B3** `roberta_lora` (the PEFT regime) — both deferred by
`plan_exp_step1.md` §7; **B5** ResNet-18, **C1** GPT-1/4-block, **C2** SegFormer-B0 — v0's own
options. A4 and B3 are the two that carry layer types no built model has (token embedding, causal
attention, LoRA adapters), so §9's last lot keeps them as the first candidates for a new folder.

Three structural facts about these models that the campaign must respect, all inherited:

1. **`GroupNorm` is not an `AdaFisherMulti` `SUPPORTED_MODULES` type**, so in A2 its parameters
   take the identity preconditioner. That is the point of the variant, not an oversight
   (`benchmarks/cnn_gn_cifar/model.py`'s docstring), and it gives the campaign a normalisation
   whose exact block is clean and whose operational counterpart is *absent* — a free control.
2. **Attention exposes a single fused `nn.Linear` named `qkv`** (A3, B1, B4), because
   `nn.MultiheadAttention` hides Q/K/V in a raw `Parameter` no hook sees (`plan_lot8.md` §0.2).
   v0 §2.5 wanted both the fused and the split variant; with the built models only the **fused**
   one exists, so "K-FAC's error on attention" is measured for the shared-input, one-block reading.
   The split variant is a §9 option, and would need a model change, i.e. a new folder.
3. **`pos_embed` (A3, B1) and `cls_token` (B4) are raw parameters**, owned by no hooked module.
   They are updated (the `plan_lot8.md` §0.3 regression test asserts it) but their curvature is the
   identity. `registry.py` must classify them explicitly rather than let them fall through.

---

## 2. Feasibility: the three computational regimes

### 2.1 Notation and the output factor

- `U ∈ R^{m×P}` stacks, for each probe `n` and each column `c` of a root `Λ_n^{1/2}`, the row
  `N^{-1/2} (Λ_n^{1/2} e_c)^T J_n`. Then `F = U^T U` exactly.
- Closed-form root for softmax, no decomposition needed: `S_n[:, c] = sqrt(p_c)(e_c − p_n)`
  satisfies `S_n S_n^T = diag(p_n) − p_n p_n^T` (the same choice as *KFAC from scratch*,
  arXiv:2507.05127, cheat sheet §6) `[ESTABLISHED]`. It gives `C` columns for a rank of `C−1`:
  +11 % rows at `C = 10`.
- Each row of `U` costs **one** backward pass of one output vector, and the `N` probes of a batch
  share it (independent samples: BN in eval mode, dropout off). Building `U` costs `C` backward
  passes per probe batch.
- `Ê` has the same format with `m = N` rows `N^{-1/2} ∇_θ ℓ_n`.
- For `mnist_autoencoder` the loss is **MSE**, not cross-entropy: `Λ_n = I` (`r = d_out = 784`
  columns per probe, so `m = 784 N`) and `F = GGN` holds by the canonical link (Martens
  arXiv:1412.1193 §9) `[ESTABLISHED]`. This is the one built model whose output factor is not the
  softmax root, and the one where `m` is large for a small `N`.

### 2.2 Regime A — dense

`F` formed by a SYRK `U^T U` in fp64. Memory `8P²`, about `3×` that with `eigh` (matrix,
eigenvectors, workspace), hence `P_max ≈ sqrt(B/24)` `[DERIVED]`. fp64 is not a luxury: Fisher
spectra span more than `10^8`, and in fp32 eigenvalues below `~10^{-7} λ_max` are not meaningful.

Choose `N` with `N(C−1) ≥ P` so that `F` can have full rank: `N = 4 000` for A1-A3 (`m = 40 000`
rows with the `C`-column root). Below that, every metric that goes through an inverse is dominated
by the damping inside `ker F` (§2.3). Cost of the SYRK: `m P² ≈ 2.8·10^13` fp64 flop for A1, i.e.
seconds on an H100 `[DERIVED]`.

**Hardware consequence, and a real trap.** `U` in fp32 is `4mP` = 4.26 GB for A1, and `F` in fp64
is 5.68 GB, so the SYRK must stream `U` in row blocks and only `F` stays resident. The `eigh` then
needs ~17 GB for A1. The training jobs request `h100_1g.10gb`; an analysis job that copies that
line **cannot form A1's or A2's `F` at all**. Analysis jobs request `2g.20gb` minimum, a full GPU
or a 40 GB A100 preferred, and ≥ 64 GB of host RAM to stage `U`.

### 2.3 Regime B — factored / dual, streamed per layer

`U` is never formed whole. Per layer `ℓ`:

1. the **per-layer Gram** `K_ℓ = U_ℓ U_ℓ^T ∈ R^{m×m}` (the layer's `Λ`-weighted NTK). Then
   `‖F‖_F² = ‖Σ_ℓ K_ℓ‖_F²`, `‖F_{ℓℓ'}‖_F² = ⟨K_ℓ, K_{ℓ'}⟩_F`, and the nonzero spectrum of `F` is
   that of `Σ_ℓ K_ℓ` (ViViT, arXiv:2106.02624) `[ESTABLISHED]`;
2. the **layer statistics** `(a_{n,t}, g_{n,c,t})` captured by hooks. Every structured approximation
   is built from them.

Two ways to get `K_ℓ` for a shared-weight linear layer (row `(n,c)`: `Σ_t g_{n,c,t} a_{n,t}^T`):
**materialise** `U_ℓ` (memory `4 m P_ℓ`) then SYRK, or **ghost** (no per-sample gradients),
`K_ℓ[(n,c),(n',c')] = Σ_{t,s} (a_{n,t}·a_{n',s})(g_{n,c,t}·g_{n',c',s})`, cost `≈ 2(mT)²(d_in+d_out)`
— the ghost norm of differential privacy (arXiv:2110.05679), chosen per layer as in mixed ghost
clipping (arXiv:2205.10683) `[ESTABLISHED]`.

`[DERIVED]` at `m = N(C−1)`, `C = 10`:

| Model | `N` (20 GB) | `m` | whole `U` fp32 | widest `U_ℓ` | Gram fp64 | whole `Ê` |
|---|---|---|---|---|---|---|
| `cct_2_3x2_cifar` | 1 000 | 9 000 | 10.2 GB | 2.7 GB | 0.65 GB | 1.1 GB |
| `resnet20_cifar` | 1 000 | 9 000 | 9.7 GB | 1.3 GB | 0.65 GB | 1.1 GB |
| `vit_small_cifar` | 1 000 | 9 000 | 97 GB (**streamed**) | 5.3 GB | 0.65 GB | 10.8 GB |
| `mnist_autoencoder` (`Λ = I`) | 32 | 25 088 | 285 GB (**streamed**) | 79 GB → **ghost** | 5.0 GB | 0.36 GB |

`mnist_autoencoder` is the awkward one and it is worth saying why: its output factor is
`Λ = I_{784}`, so a single probe contributes 784 rows. At `N = 32` the Gram is already
`25 088²`, and the widest layer (`784×1000`) cannot be materialised. Either take `N` small and
accept `m ≫ P_ℓ` (ghost is then also expensive, `T = 1` helps: no sharing), or subsample the output
coordinates with a random root `Λ^{1/2} Ω`, `Ω ∈ R^{784×k}` with `E[ΩΩ^T] = I` — an **unbiased**
sketch of `F`, whose variance is measured against the noise floor (§3.4) rather than assumed
negligible. Decide this in the lot that reaches B0, not before.

**The price of regime B is rank.** `F` has rank at most `m ≪ P`. Frobenius and direction metrics
stay exact; the metrics that go through a damped inverse mix two effects — the error inside
`im F`, and the curvature a full-rank approximation (K-FAC) **invents** inside `ker F`, where
`F_λ = λI`. Read them with a `λ` sweep, and against regime A, where `N(C−1) ≥ P` removes the
second effect.

### 2.4 Regime C — matrix-free

`curvlinops`' `GGNLinearOperator` / `EFLinearOperator` on the probes: exact `Fv` in one JVP + one
VJP whatever `C` is. Accessible: spectral norm of `F − cK` (Lanczos), traces (Hutch++/XTrace),
`ρ(K)` through conjugate gradients, spectra (SLQ). Not accessible cleanly: the KL (log-determinant
by SLQ is possible but stochastic; optional). `curvlinops` is **not** currently installed in
`.venv` and is **not** in `requirements-cluster.txt` — adding it is a lot-1 decision, and if the
cluster wheelhouse does not carry it, regime C's oracle role falls back to a hand-written
`Fv` (double backward), which T1 then has nothing independent to check against. Say so rather than
discover it.

### 2.5 Hardware and numerical traps (lot 0's business)

- **TF32 off**: `torch.backends.cuda.matmul.allow_tf32 = False` **and**
  `torch.backends.cudnn.allow_tf32 = False`. The second is `True` by default: on A100/H100
  convolutions then run in TF32, with a relative error around `10^{-3}` on per-sample gradients —
  enough to break every exactness test. Recent torch also exposes
  `torch.backends.{cuda.matmul,cudnn}.fp32_precision`; set whichever exists, and **record what was
  actually set** in the run metadata.
- **MIG**: one process, no NCCL, and (see §2.2) not the 10 GB instance.
- **BN**: in *train* mode a sample's output depends on the batch, so per-sample gradients, `Ê` and
  `F` are not defined. Every reference is computed in **eval** mode. The gap to train-mode factors
  (what AdaFisher actually sees) is measured separately, in P2. A2's `GroupNorm` default exists to
  have one model without the question at all.
- **Dropout off** for every reference. Measured on the built models: A3 `vit_micro_cifar` has
  `p = 0` everywhere (so eval mode changes nothing there), while **B1 `cct_2_3x2_cifar` and B4
  `vit_small_cifar` carry `p = 0.1`** — i.e. on those two, the training-time factors AdaFisher
  sees are estimated through live dropout, and the gap to the reference is a P2 quantity, not a
  detail.
- **`rvec` everywhere.** PyTorch flattens row-major, the literature writes `cvec`. The identity
  that fixes the convention is `rvec(B M A^T) = (B ⊗ A) rvec(M)`, so the repository's applied form
  `B̃^{-1} M Ã^{-1}` corresponds to the papers' `A ⊗ B` (`CLAUDE.md`, "Known testing pitfalls";
  `plan_lot2.md` §0.4). Lot 0 asserts it numerically rather than restating it.
- **Reused modules / tied weights**: a module called twice overwrites its hook. None of the eight
  built models does this (checked: every hooked module appears once in the forward). A4 would.

---

## 3. Experimental design

### 3.1 The factorial design: structure × source × reference

For each layer `ℓ` and each checkpoint, the error `d(S(σ), R)` for

- `S ∈ {full, block-diagonal, K-FAC-expand, K-FAC-reduce, EKFAC, TKFAC, AF-diag-Kronecker, exact
  diagonal, (normalisations: Hadamard, Prop. 3.1 as implemented)}`;
- `σ ∈ {type-2, MC_K with K ∈ {1, 4, 16}, empirical}`;
- `R ∈ {F, Ê}`.

| Cell | Isolates | Question |
|---|---|---|
| `d(Ê, F)` | **source** error alone, no structure | Q1 |
| `d(S(type-2), F)` | **structure** error alone, against the true Fisher | Q2 |
| `d(S(emp), Ê)` | **structure** error alone, against the empirical Fisher | Q2 |
| `d(S(emp), F)` | AdaFisher's / FisherAdapTune's actual path | Q3 |
| `d(S(MC_K), F)` | structure + Monte-Carlo noise | HF6 |
| the above, per layer type | everything, disaggregated | Q4 |

**Error decomposition (Q3).** Report, for each `S`, the total `d(S(emp), F)` against
`d(Ê, F) + d(S(emp), Ê)` and against `d(Ê, F) + d(S(type-2), F)`. For a true distance the triangle
inequality bounds the total by the first sum; the gap says whether the two errors **cancel**
(total ≪ sum) or **align** (total ≈ sum). For the KL, which is not a distance, report the three
terms only.

**Two paths to AdaFisher's diagonal (HF3).** `diag(A) ⊗ diag(G) = diag(A ⊗ G)`: AdaFisher's raw
estimator **is** the diagonal of K-FAC. Two paths lead from the exact block `B_ℓ` to it, and their
gap is entry-wise `E[a_j² g_i²] − E[a_j²]E[g_i²] = Cov(a_j², g_i²)` (no weight sharing) — the
independence bias **on the diagonal alone**, distinct from the loss of the off-diagonal.

**Shared-weight decomposition** (regime A, and regime B when `mT` stays reasonable):
`B_ℓ →(cross terms t ≠ t' dropped) B_ℓ^exp →(independence) K-FAC-expand`. The first arrow is the
**weight-sharing** error, the second the **independence** error. K-FAC-reduce takes another path
and is compared to `B_ℓ` directly.

### 3.2 Two protocols, and what the checkpoints do and do not contain

- **P1 — structural.** Same θ, same probes `D_N`, no EMA, no min-max, damping swept. Answers "what
  is the best approximation *possible* in each family".
- **P2 — operational.** The optimizer's internal state *as it serves the step*: after the EMA
  (`gammas = (0.92, 0.008)`), after min-max for `diag`, at `λ = 10^{-3}`, with BN in train mode,
  dropout live, at the training batch size — compared to the exact references at the **same** θ.

v1's P2 differs from v0's in two ways.

**(a) The state is this repository's, and covers all five modes.** v0 routed P2 through an adapter
onto the upstream AdaFisher because the port's fidelity was an open question. It is no longer:
`diag(minmax=False)` is bit-exact against `FisherAdapTune/scripts/adafisher.py`
(`tests/test_diag_bitexact.py`) and `diag(minmax=True)` matches Eq. (4)'s semantics
(`tests/test_diag_eq4_semantics.py`). So P2 reads `AdaFisherMulti`'s live per-module state directly
— which also makes P2 available for `kfac`/`ekfac`/`tkfac`/`tekfac`, not just `diag`. The upstream
repository stays read-only and is used only to re-confirm T12.

**(b) The checkpoints hold θ, not the optimizer.** `benchmarks/common/checkpoints.py` writes
`model_state_dict` and metadata; the EMA'd factors are gone when the job ends. P2 must therefore
**re-warm** the EMA from a checkpoint: load θ, build the optimizer, run `k·TCov` steps of the
model's own training configuration, and snapshot. What that costs is bounded, and the bound is the
reason it is acceptable: the update is `current ← (1 − γ₀)·current + γ₁·new` with
`γ₀ = 0.92`, i.e. the pre-existing state is multiplied by `0.08` at **every** `TCov = 100` steps, so
after `k` factor updates the pre-checkpoint history contributes `0.08^k` — `6.4·10^{-3}` at
`k = 2`, `5.1·10^{-4}` at `k = 3` `[DERIVED]`. The same arithmetic read as a *composition* is
blunter still: the relative weights of the successive factor updates are
`0.92 / 0.074 / 0.0059 / …`, so **92 % of the operational state is the factor of the single most
recent minibatch**. "The state that run had" is therefore a draw, not a property of its history,
and a re-warm draws from the same distribution.

**Measured — `cnn_gn_cifar`, 2026-09-11** (`fisher_ref/experiments/rewarm_fidelity.py`). A real run
of 2 000 steps (20 factor updates) defines `θ*` and the reference state `S_real`. From `θ*`, three
fresh optimizers re-warm on *different* batch orders: two with `θ` **frozen** (`lr = 0`) — their
mutual distance is the estimator's own **noise floor** — and one with `θ` **moving**, giving the
staleness term. Relative Frobenius distances, aggregated over the primary EMA'd state, and over the
*applied* preconditioner `F̃⁻¹m̂` on a fixed random direction (the object P2 actually compares to
`F`):

| mode | re-warm | state gap | floor | **precond. gap** | **floor** | gap/floor |
|---|---|---|---|---|---|---|
| `diag` | 300 / 1000 | 2.7e-1 / 3.0e-1 | 1.4e-1 / 1.5e-1 | 2.3e-3 / **1.8e-3** | 1.9e-3 / 1.4e-3 | 1.3x |
| `kfac` | 300 / 1000 | 2.3e-1 / 2.4e-1 | 6.0e-2 / 6.3e-2 | 1.2e-1 / **1.2e-4** | 8.1e-3 / 2.7e-5 | 4.5x |
| `ekfac` | 300 / 1000 | 1.1e+0 / 2.6e-1 | 4.2e-2 / 7.2e-2 | 8.7e-1 / **5.9e-6** | 1.9e-6 / 2.5e-6 | 2.4x |
| `tkfac` | 300 / 1000 | 6.9e+5 / 2.7e-1 | 1.8e-7 / 9.6e-2 | 7.0e-1 / **8.4e-4** | 1.5e-6 / 4.8e-4 | 1.8x |
| `tekfac` | 300 / 1000 | 7.3e+5 / 2.6e-1 | 1.8e-7 / 9.0e-2 | 8.7e-1 / **6.2e-6** | 1.5e-6 / 3.0e-6 | 2.1x |

Three findings, and one of them **corrects this section's own arithmetic**.

1. **The re-warm length is not set by `0.08^k ≪ 1`, but by `0.08^k ≪ λ`.** Every mode seeds its EMA
   with the **identity** at step 0 (`kfac.py:77` and its analogues), so a *fresh* optimizer's state
   carries `0.08^k·I` — which shifts every eigenvalue of the factor, i.e. acts as a **spurious extra
   damping** on top of `λ`. At `k = 3`, `0.08³ = 5.1·10⁻⁴` is half of `λ = 10⁻³`, and the applied
   preconditioner is 12-87 % wrong on the four Kronecker modes (`tkfac`/`tekfac`'s un-normalized
   numerators are off by six orders of magnitude). By `k = 10`, `0.08¹⁰ ≈ 10⁻¹¹ ≪ λ` and the gap has
   collapsed by three to six orders of magnitude. **P2 re-warms for at least `10·TCov` steps**, and
   for a `λ` sweep (§3.3) the smallest `λ` sets the length: `k ≳ log(λ/100)/log(0.08)`. The seed's
   own decay is measured separately in `fisher_ref/experiments/identity_seed_residual.py`.
2. **At a sufficient re-warm length every mode lands within 1.3-4.5× of its own noise floor**, and
   in absolute terms within `2·10⁻³` on the applied preconditioner. `ekfac`'s `s*` is the sharpest
   case: its re-warm gap (0.15) is *below* its own batch-draw floor (0.27) — indistinguishable from
   a second realisation of the real thing. This is what makes re-running the campaign to save the
   optimizer state **not worth it** (§0.2's flag exists for the *next* campaign, not for a re-run).
3. **The staleness term is not separately visible**: `θ`-moving vs `θ`-frozen differs by the same
   order as the gap itself (e.g. `kfac` 1.2e-4 vs 1.2e-4 at `k = 10`). The ≤ 8 % weight the EMA
   gives to older factors is simply too small for `θ`'s motion over 100-300 steps to show up.

Two caveats to carry with any P2 number. The preconditioner-level gaps are read at `λ = 10⁻³`, where
damping does much of the flattening; sensitivity belongs to the `λ` sweep. And the state-level gaps
stay at 2-4× the floor for every mode at every length — the raw factors *are* noisy (`diag`'s `_S`
has a floor of 0.60 by itself), which is the same point as §3.2's 92 %: what a re-run would preserve
is one draw of a high-variance estimator. Two caveats, to be stated with every P2 number: re-warming moves θ (so
snapshot the state and restore θ to the checkpoint before comparing — the optimizer state is what
is being sampled, not the parameters), and the re-warm batches are not the original run's (the
checkpoints carry the seed but not the loader position), so P2 samples *a* state of the estimator
at that θ, not *the* state that run had. A later campaign can remove the second caveat by dumping
optimizer state alongside θ. That switch now exists — **`--checkpoint-optimizer-state`**, opt-in and
**off by default**, adding the optimizer's `state_dict` plus `AdaFisherMulti`'s EMA'd Fisher factors
re-keyed by module name (the transient `_cached_*` input batches excluded), read back as
`LoadedCheckpoint.optimizer_state`. It is off because turning it on does not help the runs that
already exist: under the WCT protocol only the reference arm is bit-reproducible (the others' step
counts follow a *measured* budget), so re-running to collect the state would **replace** campaign 1
rather than augment it, for a state that is 92 % one minibatch's factor (§3.2's EMA weights). The
flag is there to be carried by the next campaign that would run anyway — the pending
`resnet50_cifar` / `vit_small_cifar` re-runs, or the `lr` bracket on `mnist_autoencoder` — at a cost
measured at **1.8 GB** for a full 6-model x 7-arm x 5-checkpoint campaign (1.0 GB keeping only the
EMA'd factors, since inverses and eigenbases are recomputable).

### 3.3 Damping and scale

No fixed `λ`: sweep `λ = α λ̄` with `λ̄ = tr(R)/P` and `α ∈ {10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1}`,
and plot the curves. Add AdaFisher's `λ = 10^{-3}` point, but **in P2 only**, since it applies to
factors normalised into `[0,1]`. For approximations that make no claim on scale (AdaFisher's
`diag`, Adam's free diagonal), also report the optimally rescaled version
`c* = ⟨R, K⟩_F / ‖K‖_F²`.

### 3.4 Noise floor, probe partition, seeds

- **The noise floor of `F` itself.** Split `D_N` into two halves (in regime B, two subsets of rows
  of `U`: free) and compute `d(F^{(1)}, F^{(2)})` over 20 random partitions → a 95 % interval.
  **Any difference between approximations below that floor is not interpretable** (HF8). Also plot
  `d(F_{N'}, F_N)` against `N' < N`.
- **Train vs val probes.** Two probe sets, from the *train* split and the *val* split of the same
  seeded partition the run itself used (`benchmarks/common/data.py::seeded_train_val_split`), so
  "train" means points that run actually trained on. HF1 lives here.
- **Seeds.** Seed 0, all 7 arms, for the six models of campaign 1 (`outputs/<model>/`);
  `resnet50_cifar` and `vit_small_cifar` are being **re-run** as grouped jobs (their campaign-1
  checkpoints are the mislabelled ones, §3.5 rule 3) and have nothing usable until then. Seeds 1-4 for
  `mlp_ln_mnist`, `cnn_gn_cifar`, `vit_micro_cifar` and seeds 1-2 for `cct_2_3x2_cifar`,
  `resnet20_cifar` (`outputs/seeds/<model>/seed<n>/`, arms `diag` and `adamw` only) — **submitted,
  in flight**. `mnist_autoencoder`, `vit_small_cifar` and `resnet50_cifar` have seed 0 only.
  Consequence for the code: the seed axis is *discovered*, never assumed; every result row carries
  the seed it came from, and a metric aggregated over seeds carries the count.
  One caveat inherited from the campaign: the seed job passes **no** `--lr-schedule`, i.e. the
  clamped `nominal` cosine, to match seed 0. Seed-0 runs of the *re-run* models (`resnet50_cifar`,
  `vit_small_cifar`) may end up on `budget`; check each run's `manifest.json` before pooling seeds,
  and never pool across schedules.

### 3.5 Checkpoints: what exists, and how to read it

```
benchmarks/outputs/<model>/<arm>/ckpt_{0,0.01,0.1,0.5,1}.pt        # seed 0, 7 arms
benchmarks/outputs/seeds/<model>/seed<n>/<arm>/ckpt_*.pt           # seeds 1-4 / 1-2, arms diag, adamw
benchmarks/outputs/.../manifest.json                                # config + per-arm summary + checkpoint paths
```

Each payload: `model_state_dict`, `fraction`, `step`, `epoch`, `seed`, `total_steps`, `scheduled`.
Three rules the campaign follows, all of them consequences of the WCT protocol:

1. **The fraction is relative to the *nominal* trajectory** (`--epochs × batches`), shared by every
   arm of a model. So `ckpt_0.5` is the same amount of *training* in every arm, which is what makes
   the arms comparable at a checkpoint. It is **not** the same amount of wall-clock time, and not
   the same position within an arm's own run.
2. **`scheduled=False` means "the arm's own end", not the nominal 100 %.** A budgeted arm that
   stopped short gets its largest fraction pinned there. Any figure that puts `ckpt_1` on a common
   x-axis must read `step / total_steps` from the payload instead of trusting the label.
3. **Every checkpoint produced before the campaign-1 denominator fix is mislabelled except the
   `diag` arms'** (`CLAUDE.md`'s status block). `resnet50_cifar` and `vit_small_cifar` were run
   entirely pre-fix and are being re-run. The bridge therefore refuses to load a checkpoint whose
   run predates the fix unless told to: the `manifest.json` is the discriminator (a post-fix
   payload carries `total_steps` and `scheduled`), and a payload missing those keys is rejected
   with a message naming the model and arm.

The campaign's θ axis is therefore `{0, 1 %, 10 %, 50 %, 100 %} × {diag, adamw} × seeds`, with
seed 0 additionally offering the other five arms — which buys a question v0 could not ask: *does
the drift at a given θ depend on which optimizer produced that θ?*

### 3.6 One target the campaign inherits: the `mnist_autoencoder` stall

On the project's primary bench all five Fisher modes freeze at MSE ≈ 0.7085 within 2 epochs while
`adam` descends to 0.4836; between 10 % and 100 % of training the parameters move 0.2-0.5 % against
147 % for `adam` `[MEASURED]`. `λ` has been swept over four orders of magnitude and **exonerated**
for the four Kronecker modes (`CLAUDE.md`'s status block: do not re-run that sweep). The two
untested candidates are the shared `lr = 1e-3` and *the geometry of `A ⊗ B` on this network*.

The second is exactly what this campaign measures, and the checkpoints needed already exist
(`outputs/mnist_autoencoder/{diag,kfac,...}/ckpt_*.pt`). The cheapest decisive measurement is M5's
`ρ` and the per-layer ratio `‖F̃^{-1} m̂‖ / ‖m̂‖` at the stall point, on B0, comparing the `diag`
trajectory's θ against the `adam` trajectory's θ at the same fraction. This is called out here
because it makes the campaign's first regime-B target a question the repository already needs
answered, rather than an exercise.

---

## 4. The approximation zoo, per layer type

*(unchanged from v0 §4 in substance; what follows restates it, with the layer types the built
models actually have.)*

Common notation. `a_{n,t}`: layer input at shared position `t` (token, spatial location), with a 1
appended for the bias. `g_{n,c,t}`: gradient backpropagated to the layer's pre-activation, for
column `c` of source `σ`:

- type-2: `c = 1..C`, output vector `S_n[:, c]`;
- `MC_K`: `c = 1..K`, vector `K^{-1/2}(p_n − e_{ỹ_{n,c}})` with `ỹ ∼ p_n`. The `K^{-1/2}` is the
  factor *KFAC from scratch* lists in its own errata; without it K-FAC-MC does not converge to the
  GGN;
- emp: `c = 1`, vector `p_n − e_{y_n}`.

Per-sample (per-column) layer gradient `G_{n,c} = Σ_t g_{n,c,t} a_{n,t}^T`; exact block
`B^σ_ℓ = (1/N) Σ_{n,c} rvec(G_{n,c}) rvec(G_{n,c})^T`, so `B^{type-2}_ℓ = F_{ℓℓ}` and
`B^{emp}_ℓ = Ê_{ℓℓ}`.

| Structure | Definition | Layers | Note |
|---|---|---|---|
| Full | `F` or `Ê` | all | the reference |
| Exact block-diagonal | `blkdiag_ℓ(B_ℓ)` | all | isolates the inter-layer information (HF5); two partitions: by `nn.Module` and by tensor (W / b) |
| K-FAC-expand | `A = (1/NT) Σ_{n,t} aa^T`, `G = (1/N) Σ_{n,c,t} gg^T` | Linear, Conv | exact in the *expand* setting for a deep linear network (arXiv:2311.00636 Prop. 1) `[ESTABLISHED]` |
| K-FAC-reduce | factors built on `Σ_t a_{n,t}` and `Σ_t g_{n,c,t}` | shared Linear, Conv | exact in the *reduce* setting with weighted-sum aggregation, e.g. mean pooling (Prop. 2) `[ESTABLISHED]`; **normalisation constants taken from `KFACLinearOperator`, not re-derived** |
| EKFAC | K-FAC's eigenbasis `Q_A ⊗ Q_G`, eigenvalues `s_ij = (1/N) Σ_{n,c} [(Q_G^T G_{n,c} Q_A)_ij]²` | Linear, Conv | preserves `tr(B_ℓ)` by construction (T9) |
| TKFAC | `K = δ Φ ⊗ Ψ` (arXiv:2011.10741 Eq. 4.3-4.9) `[ESTABLISHED]` | Linear, Conv | trace preserved **without sharing**; the *expand* adaptation for shared layers is a declared choice |
| AF-raw | `diag(A) ⊗ diag(G)`, `/|T|` for convs (App. A.3) | Linear, Conv | = K-FAC's diagonal = FisherAdapTune's `F̃_D` |
| AF-op | `H'_D ⊗ S'_D + λI` (Eq. 4), min-max and EMA included | all | **P2 only**, read off `AdaFisherMulti`'s state (§3.2) |
| Exact diagonal | `diag(B_ℓ)` | all | the control: the best a diagonal method can do |
| Free diagonal | Adam's second moment `v`, rescaled | all (P2) | arXiv:2507.18807's control, to be demanded of any diagonal method — and the `adam`/`adamw` arms give it for free |

**Normalisations** (`LayerNorm`, `GroupNorm`, `BatchNorm2d` in eval): parameters `γ, β ∈ R^C`,
normalised input `x̂_{n,t}`. Exact per-sample gradients `∇_γ = Σ_t g_t ⊙ x̂_t`, `∇_β = Σ_t g_t`. The
exact block is `2C × 2C` and is **always** computable, even in regime C.

| Structure | Definition |
|---|---|
| Exact joint | the `2C × 2C` block on `(γ, β)` |
| Exact separate | `γ` and `β` blocks with the cross terms dropped (what AdaFisher does) → measures HF4 |
| Hadamard (Prop. 3.1) | `γ`: `((1/|T|) Σ_t x̂x̂^T) ⊙ ((1/|T|) Σ_t gg^T)`, up to normalisation |
| Prop. 3.1 as implemented | the diagonal `[H_D]_c [S_D]_c` read from this repository's `diag.py` |

**A point to check in the code before any conclusion** — and this repository has already half
checked it. Prop. 3.1 speaks of "pre-normalized" activations while its proof says `h_{i-1}`
"contains normalized activations"; `∇_γ` involves `x̂` (normalised), and a `forward` hook on
`nn.BatchNorm2d`/`nn.LayerNorm` captures the input **before** normalisation. Two related, already
documented facts: `diag.py`'s normalisation formula is a "sum-then-square" deviation from the
Proposition's "square-then-sum" (`plan_lot5.md` §0.6), and the four Kronecker modes use a
Frobenius-optimal scalar surrogate for the exact `H|_ν` (`plan_lot5.md` §0.1-§0.2). The bench
measures all three against the exact `2C × 2C` block. **Read `factors.py` before asserting which
variable is being captured.**

**Embeddings** (`pos_embed` here, token embeddings in A4): one-hot input, `A` diagonal; the exact
block has nonzero rows only for positions present in the probes. For `pos_embed` the "input" is
constant, which is worth stating explicitly rather than special-casing silently.

**LoRA** (B3, not built): K-FAC per adapter; exact blocks; **A-B coupling** measured by M8.

**Head**: `Linear` without sharing, but `Λ(p)` depends on `a`: independence is false there by
construction, and it is a useful measurement point. Every built model has exactly one.

---

## 5. Metrics

`R ∈ {F, Ê}` is the reference, `K` the approximation, `X_λ = X + λI`.

| Id | Metric | What it measures | A | B | C |
|---|---|---|---|---|---|
| M1 | `e_F = ‖R−K‖_F/‖R‖_F`; `cos_F`; `e_F* = sqrt(1−cos_F²)` | entry-wise fidelity, dominated by the large eigenvalues | dense | `‖R‖_F² = ‖UU^T‖_F²`, `⟨R,K⟩ = tr(UKU^T)`, `‖K‖_F²` closed-form | Hutchinson |
| M2 | `e_2 = ‖R − c*K‖_2/‖R‖_2` | the worst direction | `eigvalsh` | Lanczos | Lanczos |
| M3 | `D_λ(K‖R)` = KL between the two preconditioner Gaussians (Stein loss) | **affine-invariant** gap between preconditioners | Cholesky | Woodbury (§5.1) | SLQ (optional) |
| M4 | `κ_λ` of `K_λ^{-1/2} R_λ K_λ^{-1/2}` | preconditioned CG iterations, `∝ sqrt(κ)` | generalised `eigh` | Lanczos | Lanczos |
| M5 | `ρ(K) = (g^T d)²/((d^T R_λ d)(g^T R_λ^{-1} g))`, `d = K_λ^{-1} g` | the fraction of the optimal quadratic decrease obtained along `d`; `ρ ∈ [0,1]` | exact | Woodbury | CG |
| M6 | dominant-eigenspace overlap `‖V_R^T V_K‖_F²/k`, `k ∈ {C, 10C}` | are the `O(C)` outlier directions captured? | exact | via the Gram | Lanczos |
| M7 | `σ₂/σ₁` and `Σ_{i>1}σ_i²/Σσ_i²` of the rearrangement `R(B_ℓ^exp)`; `‖B_ℓ − B_ℓ^exp‖_F/‖B_ℓ‖_F`; `‖diag B_ℓ − diag A ⊗ diag G‖/‖diag B_ℓ‖` | independence bias, the weight-sharing share, the diagonal bias | exact | randomised SVD | — |
| M8 | `c_{ℓℓ'} = ⟨K_ℓ, K_{ℓ'}⟩_F/(‖K_ℓ‖_F‖K_{ℓ'}‖_F)` | inter-block coupling: an uncentred CKA between layer tangent kernels | exact | per-layer Grams | — |

For M7, the rearrangement diagnostic has a precedent to cite: Koroko et al. (arXiv:2201.10285)
already approximate Fisher blocks by a Kronecker-product SVD on deep auto-encoders
`[ESTABLISHED, abstract]`. The bench's novelty is not the tool but the per-layer-type map on modern
architectures, against an exact reference, along a training trajectory.

### 5.1 Exact M3 and M5 in regime B

With `R = U^T U`, `G = UU^T`, `M = λI_m + G`:

- `log det R_λ = P log λ + log det M − m log λ` (Sylvester);
- `R_λ^{-1} = λ^{-1}(I − U^T M^{-1} U)`;
- `tr(K_λ R_λ^{-1}) = λ^{-1}[tr K + λP − tr(M^{-1}(UKU^T + λG))]`;
- `tr(K_λ^{-1} R_λ) = λ tr(K_λ^{-1}) + tr(U K_λ^{-1} U^T)`;
- `log det K_λ`: Kronecker, `Σ_{i,j} log(α_i γ_j + λ)`; diagonal, immediate; exact block-diagonal,
  `Σ_ℓ [P_ℓ log λ + log det(I + K_ℓ/λ)]`.

Self-consistency: for `K = R`, `tr(K_λ R_λ^{-1}) = P` and `D_λ = 0` (T7).

```python
import math, torch

def stein_kl_lowrank(U: torch.Tensor, K, lam: float) -> torch.Tensor:
    """KL( N(0, R_lam^{-1}) || N(0, K_lam^{-1}) ) with R = U^T U, U: (m, P), fp64.
    K must provide: apply_rows(U) -> U @ K (row-wise K u_i), trace(), logdet(lam)."""
    m, P = U.shape
    G = U @ U.T                                              # m x m Gram
    M = G + lam * torch.eye(m, dtype=U.dtype, device=U.device)
    L = torch.linalg.cholesky(M)
    UKUt = U @ K.apply_rows(U).T                             # m x m  = U K U^T
    X = torch.cholesky_solve(UKUt + lam * G, L)
    tr_KR = (K.trace() + lam * P - X.diagonal().sum()) / lam # tr(K_lam R_lam^{-1})
    logdet_R = P * math.log(lam) + 2 * torch.log(L.diagonal()).sum() - m * math.log(lam)
    return 0.5 * (tr_KR - P - K.logdet(lam) + logdet_R)
```

---

## 6. Position relative to existing work

*(unchanged from v0 §6.)* AdaFisher's own validation (App. B.2) is a toy model, a Monte-Carlo
"true Fisher" from NNGeometry, and a mean absolute error between **diagonals** (Fig. 12). The bench
replaces every element: analytic `Λ`, blocks and spectra instead of the diagonal, scale-invariant
metrics instead of a scale-dependent MAE, eight layer types, the paper's own model
(`cct_2_3x2_cifar`), and a noise floor. Kunstner, Balles & Hennig (arXiv:1905.12558) show the
`Ê`-vs-`F` gap without a per-layer map; Benzing (arXiv:2201.12250) judges exact updates at the
optimization level; Eschenhagen et al. (arXiv:2311.00636) prove expand/reduce exactness on linear
networks without measuring fidelity on nonlinear ones; Koroko et al. (arXiv:2201.10285) give KP-SVD
on auto-encoders; ViViT (arXiv:2106.02624) supplies regime B's tooling; Zhang et al.
(arXiv:2402.16788) and Ormaniec, Dangel & Singh (arXiv:2410.10986) motivate the per-layer-type
breakdown and the Q/K/V separation (which the built models do not currently allow, §1).

Claimed gap, **to confirm with a targeted search before writing**: no systematic per-layer-type map
of the K-FAC / EKFAC / TKFAC / AdaFisher-diagonal errors against **both** the analytic true Fisher
and the exact empirical Fisher, with a source/structure decomposition and a noise floor.

---

## 7. Software architecture

`fisher_ref/` is a **top-level package**, like `benchmarks/`, and imports from it. It does not live
under `src/` (which is the shipped `adafisher_modes` package) and it changes nothing in either.

```text
fisher_ref/
  conventions.py      # dtype policy, TF32 off, rvec, eval-mode context, metrics_version, run metadata
  probes.py           # versioned, hashed, augmentation-free probe sets from benchmarks.common.data
  registry.py         # module/parameter -> layer type, over any benchmarks/<model> network
  checkpoints.py      # the bridge to benchmarks/outputs/: discover runs, load theta at a fraction
  capture.py          # hooks: inputs a, x_hat for norms, output grads g per column c     [lot 1]
  sources.py          # backprop vectors: type-2 (closed-form root), MC_K (with K^{-1/2}), empirical [lot 1]
  reference/
    dense.py          # regime A: F, E_hat, B_exp, dense fp64                              [lot 1]
    factor.py         # regime B: per-layer U_l or ghost Grams, streamed                   [lot 4]
    matfree.py        # regime C: curvlinops operators (or a hand-written Fv)              [lot 6]
  approx/
    base.py           # the CurvatureBlock protocol                                        [lot 2]
    kfac.py ekfac.py tkfac.py diag.py blockdiag.py norm_layers.py                          [lot 2]
    adafisher_state.py # P2: reads AdaFisherMulti's live state, no re-implementation       [lot 5]
  metrics/            # frobenius, spectral, stein_kl, ngd, subspace, kron_diag, coupling, noise_floor [lot 2]
  runners/
    p1_structural.py  # fixed theta from a checkpoint, fixed probes, lambda sweep          [lot 2]
    p2_operational.py # EMA re-warm from a checkpoint, state snapshot                      [lot 5]
  experiments/        # measurement drivers, not tests: one script per protocol question, its
                      # answer recorded in this file. rewarm_fidelity.py (§3.2),
                      # identity_seed_residual.py (the 0.08^k seed acting as extra damping)
  outputs/<model>/<arm>/seed<n>/<fraction>/{metrics.csv,meta.json}
tests/test_fisher_ref_*.py
```

### 7.1 The common approximation interface

Every approximation exposes the same interface, so each metric is written once for all regimes:

```python
class CurvatureBlock(Protocol):
    P: int
    def matvec(self, v: Tensor) -> Tensor: ...            # K v
    def apply_rows(self, U: Tensor) -> Tensor: ...        # rows u_i -> K u_i
    def solve(self, v: Tensor, lam: float) -> Tensor: ...  # (K + lam I)^{-1} v
    def trace(self) -> Tensor: ...
    def fro2(self) -> Tensor: ...
    def logdet(self, lam: float) -> Tensor: ...
    def diag(self) -> Tensor: ...
    def to_dense(self) -> Tensor: ...                      # regime A only
```

Implementations: `Dense`, `LowRank(U)`, `Kron(A, G)`, `EKFAC(QA, QG, s)`, `ScaledKron` (TKFAC),
`Diag`, `HadamardNorm`, `BlockDiag([...])`.

### 7.2 Why P1 re-implements the structures instead of calling `adafisher_modes`

`src/adafisher_modes/` computes EMA'd, damped, hook-driven factors at training precision — the
operational object, which is exactly what P2 measures. P1 needs the *structure alone*: one batch of
probes, no EMA, no min-max, fp64, and a **source** parameter the optimizer does not have (type-2
and MC vectors never occur in training). So P1 owns its own implementations, and T-tests pin them
to the repository's at the degenerate setting (one update, EMA replaced by the identity, empirical
source, `λ = 0`). That cross-check is the point: it turns "our K-FAC" and "the optimizer's K-FAC"
into one measured statement instead of two implementations nobody compared.

---

## 8. Dependencies and invariants

| Dependency | Use | Status |
|---|---|---|
| `benchmarks/` (this repo) | models, the probe data pipeline, the trajectories, the checkpoints | **read-only for this campaign**; `fisher_ref` imports it, never edits it |
| `src/adafisher_modes/` (this repo) | P2's operational state; the degenerate-setting cross-checks | read-only; a campaign finding may later motivate a change, through its own lot |
| `reference_repos/AdaFisher`, `.../FisherAdapTune` | re-confirming T12, hyperparameters | **read-only**, as `CLAUDE.md` requires |
| `curvlinops` | regime C's oracle, and the K-FAC-reduce normalisation constants | **not installed**, not in `requirements-cluster.txt`; adding it is a lot-1 decision (§2.4) |
| `reference_repos/EKFAC-pytorch` | second EKFAC implementation (cross-check) | read-only |

**Invariants** (changing one invalidates earlier comparisons): the upstream code; the probe sets
(hashed); the precision policy (fp64 references, TF32 off); `rvec` and the `1/N` scale; the block
definition (by `nn.Module`, the by-tensor variant declared); and a `metrics_version` string written
into every result file.

**Experimental** (versioned, may change): the corrected normalisation estimator; TKFAC's *expand*
adaptation; the damping grid; the output-coordinate sketch for `mnist_autoencoder` (§2.3).

---

## 9. Implementation lots

These continue after **step 1** of the superseded root plan's §8 (`plan_exp_step1.md`: the
`benchmarks/` package and every model — done). Lot 0 is that plan's step 2 with its
non-curvature half carved out, for the reason `plan_exp_lot0.md` §0.1 gives.

| Lot | Content | Deliverable | Exit criterion |
|---|---|---|---|
| **0** | `conventions.py`, TF32 flags, `probes.py`, `registry.py`, **`checkpoints.py` (the bridge)** | `fisher_ref/` importable, green CI | **T0.1-T0.8** (§10.1) pass; the pre-existing test suite is untouched and still green |
| **1** | A1 end to end: `capture.py`, `sources.py`, `reference/dense.py`; `F`, `Ê`, `B_ℓ` in fp64 at one checkpoint | the first exact references | T1, T2, T6 pass; `curvlinops` decided (§2.4) |
| **2** | the zoo (`approx/`) + the metrics (`metrics/`) + `p1_structural.py`, on A1: 5 checkpoints × the available seeds, noise floor | the first figure set, `metrics.csv` | T3-T5, T7-T9 pass; the noise floor is plotted; HF3, HF4 decided on A1 |
| **3** | A2 (conv, GN and BN-eval), A3 (fused `qkv`, mean pooling); sharing/independence decomposition | the complete regime-A map | HF2 decided; every Q4 conclusion holds on ≥ 2 models |
| **4** | regime B: `factor.py`, per-layer Grams, ghost, Woodbury — **validated against A1-A3's dense first**, then B1, B2 and **B0 (`mnist_autoencoder`, §3.6)** | the regime-B map + the stall diagnosis | T10, T11 pass; §3.6's `ρ` / `‖F̃^{-1}m̂‖` measurement produced |
| **5** | P2: `adafisher_state.py`, the EMA re-warm (§3.2, **≥ 10·TCov steps**, `0.08^k ≪ λ`), on A2, B1, B2, all five modes | P1 vs P2 | T12 passes; the re-warm's fidelity re-measured on the lot's own models (§3.2 did it on A2); HF7 decided |
| **6** *(options)* | `vit_small_cifar` at realistic width; `resnet50_cifar` in regime C; then, only if the map calls for them, new benchmark folders for A4 (GPT-micro) and B3 (RoBERTa-LoRA) | extensions | decided after lot 5 |

The order is forced: regime A is regime B's oracle; P2 needs the zoo P1 builds; a new model folder
is a `benchmarks/` change and is the last thing this campaign should do.

---

## 10. Validation criteria

### 10.1 Lot 0's tests (new in v1)

| Test | Statement |
|---|---|
| T0.1 | `configure()` leaves both TF32 switches off (and whichever `fp32_precision` knob exists), and records what it set |
| T0.2 | the reference dtype is fp64 and `metrics_version` is present in the metadata a runner would write |
| T0.3 | the `rvec` convention: `rvec(uv^T) = u ⊗ v` and `rvec(B M A^T) = (B ⊗ A) rvec(M)`, numerically |
| T0.4 | the reference-mode context puts every module in eval and restores the previous mode; under it a model's output on a sub-batch equals the sub-batch of its output on the full batch (per-sample independence), and that check *fails* for a BN model in train mode |
| T0.5 | probe sets are deterministic (same digest twice), augmentation-free (bit-identical tensors twice, on a CIFAR **train** split), train/val-disjoint, and use the same seeded partition the training runs used |
| T0.6 | `registry.classify` partitions **every** parameter of **every** `benchmarks/<model>` network exactly once, with the expected per-model layer-type inventory, and marks as unhooked exactly the parameters `tests/test_benchmark_models.py` already lists |
| T0.7 | the checkpoint bridge discovers both output layouts (`outputs/<model>/<arm>` and `outputs/seeds/<model>/seed<n>/<arm>`), loads θ bit-identically into a fresh model, and exposes `step`/`epoch`/`seed`/`total_steps`/`scheduled` |
| T0.8 | the bridge **rejects** a pre-denominator-fix payload (no `total_steps`/`scheduled`) instead of silently mislabelling it, and reports a missing seed as "not found", never as an error |

### 10.2 Exactness tests (blocking), and the lot each now belongs to

| Test | Statement | Tolerance | Lot |
|---|---|---|---|
| T1 | dense `Fv` against `GGNLinearOperator` (or the double-backward `Fv`) on 20 random vectors | rel. ≤ `1e-10` (fp64) | 1 |
| T2 | type-2 `F` against a `K = 10^4` Monte-Carlo Fisher | `e_F = O(K^{-1/2})`, consistent over 5 seeds | 1 |
| T3 | *KFAC from scratch* Test 1 (`N=1`, no sharing): K-FAC-type-2 = GGN block, K-FAC-emp = EF block | ≤ `1e-12` | 2 |
| T4 | its Test 2 (deep linear, MSE): K-FAC-type-2 = GGN block, **K-FAC-emp ≠ EF** | ≤ `1e-12` / gap > `1e-3` | 2 |
| T5 | Eschenhagen Props. 1-2: expand exact in the expand setting, reduce exact in the reduce setting | ≤ `1e-10` | 2/3 |
| T6 | normalisation blocks and BN-eval: per-sample gradients against central finite differences | rel. ≤ `1e-6` | 1 |
| T7 | Woodbury KL (§5.1) against the dense KL on A1; `D_λ(R‖R) = 0` | ≤ `1e-8` | 2 |
| T8 | TKFAC: `tr(K) = tr(B_ℓ)` on unshared layers | ≤ `1e-10` | 2 |
| T9 | EKFAC: `tr(K) = tr(B_ℓ)`; cross-check against `EKFAC-pytorch` | ≤ `1e-10` / `1e-6` | 2 |
| T10 | regime B against regime A on A1-A3, every metric | rel. ≤ `1e-8` | 4 |
| T11 | ghost Gram against the materialised Gram on a layer where both fit | ≤ `1e-10` fp64 | 4 |
| T12 | P2's state reader against `AdaFisherMulti`'s own state, same batch and seed; and `diag` against the upstream optimizer | exact, or ≤ `1e-7` if reduction order differs | 5 |
| **T13** *(new)* | each P1 structure against `adafisher_modes`' own, at the degenerate setting (one update, identity EMA, empirical source, `λ=0`) | ≤ `1e-10` | 2 |

### 10.3 Reading rules

- A difference between two approximations is reported only if it exceeds `F`'s own 95 % noise floor
  (§3.4).
- Any "per layer type" conclusion must hold on at least one regime-A **and** one regime-B model.
- Any conclusion about an inverse (M3, M4) is shown across the whole `α` sweep, never at one `λ`.
- Any number aggregated over seeds carries the **number of seeds actually found** (§3.4); with the
  seed campaign in flight, "3 seeds" is a claim to check, not an assumption.

---

## 11. Decision rules: from result to research direction

| Observation | Consequence |
|---|---|
| structure error (type-2) ≫ source error, concentrated on shared-weight layers | prioritise rank-`r` Kronecker and a per-layer expand/reduce choice |
| source error dominant | prioritise iEF and the exact dual Fisher in PEFT |
| P2 ≫ P1 (HF7) | the weakness is the EMA / min-max / damping, not the structure: geodesic EMA, determinant gauge, directional forgetting |
| strong diagonal independence bias on normalisations (HF3-HF4) | a corrected normalisation estimator (Hadamard + the `γ`-`β` cross term): cheap, and directly opposable to Prop. 3.1 |
| `ρ(BD(F)) ≈ 1` (HF5 confirmed) | drop the inter-layer directions |
| differences below the noise floor at AdaFisher's own App. B.2 sample size | a methodological criticism: that validation is not falsifiable at that `N` |
| **B0: `ρ` collapses for the Kronecker modes at the stall point** | the `mnist_autoencoder` stall is a *geometry* failure, not a step-size one — and the campaign has answered a question the optimizer work is currently blocked on (§3.6) |

---

## 12. Risks

- **Rank deficiency in regime B**: inverse metrics dominated by `λ`. Mitigations: the sweep, regime
  A as judge, an image/kernel split for M3.
- **Precision**: accumulate `U` in fp32 but Grams and `UKU^T` in fp64; TF32 off (T1 detects it).
- **The 10 GB MIG** (§2.2): an analysis job that inherits the training job's `--gpus` line fails to
  allocate `F`. Every generated analysis job states its own memory requirement.
- **Checkpoint labelling** (§3.5 rule 3): pre-fix payloads are mislabelled; the bridge rejects them.
- **Seeds in flight**: the analysis must run, and report, on the seeds that exist.
- **API drift** (`curvlinops`, upstream AdaFisher): pinned versions, T12 replayed on each update.
- **Storage**: never write the Grams (`m²` per layer per checkpoint); write metrics only, except for
  one B1 checkpoint kept for post-hoc analysis.

---

## 13. Outputs and budget

**Format**: one CSV per `(model, arm, seed, fraction)` under
`fisher_ref/outputs/<model>/<arm>/seed<n>/<fraction>/metrics.csv`, with the columns
`layer, layer_type, structure, source, reference, lambda_alpha, metric, value, ci_low, ci_high, N,
probe_split, seed, protocol, metrics_version`, plus a `meta.json` sidecar carrying the probe
digest, the checkpoint payload's `step`/`epoch`/`total_steps`/`scheduled`, the torch version and the
TF32 state actually set. `parquet` is an optional later upgrade (§0.9).

**Figures**: F1 `ρ` map (structure × layer type), one per reference; F2 source/structure/interaction
decomposition bars; F3 curves along training, train vs val probes; F4 `σ₂/σ₁` per layer; F5 coupling
matrices `c_{ℓℓ'}`; F6 P1-vs-P2 rank scatter; F7 noise floor against `N`; F8 the normalisation panel.

**Budget** `[ESTIMATE, to be measured at lot 1]`, on a `2g.20gb` instance: regime A, minutes per
checkpoint (SYRK ≈ `2.8·10^13` fp64 flop, one `eigh` of `2.7·10^4`); B1/B2, < 30 min per checkpoint;
`vit_small_cifar` streamed, 1-2 h. Main campaign (A1-A3, B0-B2, 5 checkpoints, the seeds that
exist, P1 and P2): order 100-200 MIG-GPU-hours.

---

## References

As `plan_exp_draft_v0.md`'s reference list, unchanged (verified on arXiv on 2026-09-10), plus this
repository's own: `docs/reports/plan.md`, `plan_lot1.md`-`plan_lot8.md`, `plan_exp_step1.md`,
and `CLAUDE.md`'s status block for every `[MEASURED]` claim quoted above.
