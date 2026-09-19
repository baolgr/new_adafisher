# Lot 3 — regime A on shared weights: A2 (GroupNorm and BatchNorm-eval), A3, the sharing/independence decomposition, and HF2

*Implementation plan for lot 3 of `plan_exp_draft.md` §9. Continues `plan_exp_lot2.md`, whose zoo,
metrics and P1 runner this lot extends to the two regime-A models that have weight sharing.*

**Goal.** A1 has no weight sharing (`T = 1` on every layer), so everything lot 2 measured is silent
on the question K-FAC was generalised for: what happens when one weight is applied at many positions
(convolutions, token-wise linear layers). Lot 3 answers it on the two other regime-A models, where
the exact Fisher is still affordable:

* **A2** `cnn_gn_cifar` (24 458 parameters: 3 `Conv2d` at `T = 1024 / 256 / 64` positions, 3
  `GroupNorm`, a head) and its **BatchNorm variant** `cnn_gn_cifar_bn` (same count, BN in eval);
* **A3** `vit_micro_cifar` (21 098 parameters: a patch-embedding `Conv2d`, 8 token-wise `Linear` at
  `T = 64`, 5 `LayerNorm`, the raw `pos_embed`, a head).

**Exit criteria** (`plan_exp_draft.md` §9, §10.2): **T5** passes (expand exact in the expand
setting, reduce exact in the reduce setting — the half lot 2 left to this lot); **HF2 is decided**
by the rule pre-registered in §0.10; **every Q4 conclusion is checked on ≥ 2 models**, and the ones
that fail are reported as failing.

**Non-goals.** No regime B (lot 4), no P2 (lot 5), no new benchmark model, no seed other than 0 and
no arm other than `diag` (the θ axis is lot 2's: the five checkpoints of `diag`, seed 0). Nothing in
`benchmarks/` or `src/adafisher_modes/` changes.

---

## 0. Decisions, and the silent errors this lot must not inherit

Lot 2's code was written for shared layers but only ever *run* at `T = 1`, where most of what
follows is invisible by construction. Before any new measurement, each of the branches that
activate at `T > 1` was exercised on a toy model. Three of them are wrong, one is mislabelled, and
one lot-2 finding turns out to be a theorem. Each item below says what was measured, and what the
lot does about it.

### 0.1 Lot 2's K-FAC-reduce is `T²` too large — measured, not suspected

`approx/factors.py::augmented_input(mode="reduce")` **sums** the input over positions (and appends a
bias entry equal to `T`), and `output_grad(mode="reduce")` also sums. The factor product is then
`T²` times Eschenhagen et al.'s K-FAC-reduce, whose input factor is

    Â = 1/(N R²) Σ_n (Σ_r a_{n,r})(Σ_r a_{n,r})ᵀ          (arXiv:2311.00636 §3.3, Eq. 10)

i.e. built on the **mean** over the `R` shared positions, while the gradient factor keeps the
**sum**. Measured on a toy reduce setting (a shared `Linear` over `T = 5` positions, mean pooling,
a linear head, MSE, type-2): lot 2's `kfac` in `mode="reduce"` has relative Frobenius error
**24.000** and trace ratio **25.000** against the exact block — exactly `T² − 1` and `T²`. Its
docstring's claim that summing is "Eschenhagen et al.'s Prop. 2 normalisation" is wrong.

Invisible on A1 (`T = 1`), and silent everywhere else: a Kronecker structure off by a scalar still
has the right `cos_F`, so only `e_F`, `ρ` and the Stein KL would have been wrong — and wrong in a way
that systematically *favours expand*, which is exactly HF2's question. **Fix:** `reduce` averages the
input (bias entry `1`), sums the gradient. **T5** pins it (§2), including the negative half: expand
must *not* be exact in the reduce setting, nor reduce in the expand setting, which is what makes the
test able to catch a scale error at all.

### 0.2 EKFAC-reduce must project the *true* per-sample gradient

`accumulate_ekfac(mode="reduce")` formed its "per-sample gradient" from the *reduced* statistics,
`(Σ_t a_t)(Σ_t g_t)ᵀ`, which is not the gradient of anything unless the layer is in the reduce
setting. EKFAC's eigenvalues are `s* = E[((Q_G ⊗ Q_A)ᵀ ∇θ)²]` with `∇θ` the per-example gradient
(`ekfac_1806.03884.pdf` §3.2), whatever basis is used. With the fake gradient, `s` is not the
optimal diagonal in the reduce basis, and `tr(K) = tr(B)` (T9) fails. **Fix:** the second pass
always projects `G_{n,c} = Σ_t g_t āᵀ_t` built from the un-reduced, bias-augmented statistics; only
the basis depends on the mode. Expand's code path is unchanged bit-for-bit. EKFAC combined with
expand/reduce is **not** in Eschenhagen et al. (checked: no mention of an eigenvalue correction), so
`ekfac_reduce` is this campaign's construction, declared as such.

### 0.3 TKFAC on shared layers: a declared expand adaptation that preserves `tr(B^exp)`

TKFAC (`tkfac_2011.10741.pdf` Thm 4.1) preserves `tr(B)` only without sharing. Lot 2's accumulator,
at `T > 1`, used per-*example* traces `Σ_t ‖a_t‖²` and `Σ_t ‖g_t‖²`, whose product contains every
cross-position pair `‖a_t‖²‖g_s‖²`: its trace matches neither `tr(B)` nor `tr(B^exp)` (§0.5), and it
is not what `adafisher_modes` does. It was never run (A1 has `T = 1`).

**Declared choice:** TKFAC-expand flattens positions into the batch, one `(Λ, Γ) = (a aᵀ, g gᵀ)`
per `(n, c, t)` — exactly `adafisher_modes/approximations/_tkfac_utils.py::instantaneous_raw_factors`
applied to the flattened rows. Then `tr(K) = δ = (1/N) Σ_{n,c,t} ‖a_t‖²‖g_t‖² = tr(B^exp)` exactly,
the natural statement of trace preservation once cross-position terms are dropped (**T8-exp**).
TKFAC-reduce is the same estimator on `(ā, ĝ)`, and preserves the trace of the reduced block.
At `T = 1` both coincide with lot 2's, so A1's numbers are unaffected.

### 0.4 AdaFisher's diagonal on shared layers carries `1/|T|` on *both* factors

`adafisher_2405.16397.pdf` App. A.3 defines, for convolutions, `H_D = diag(⟦h̄⟧⟦h⟧ᵀ)/|T|` **and**
`S_D = diag(s sᵀ)/|T|`, and `diag.py`'s `_h_linear`/`_s_linear` do the same for a token-wise
`Linear` by flattening positions into the batch. K-FAC-expand (Eschenhagen et al. Eq. 7) divides
only the input factor by the positions. So on a shared layer AF-raw is **K-FAC-expand's diagonal
divided by `T`** — the same direction, a different scale (`cos_F` identical, `e_F` and `ρ` not).
`af_raw` is emitted at App. A.3's scale; `c*` (M1) removes it.

HF3's *diagonal bias* is then measured against K-FAC-expand's own diagonal (not against AF-raw, whose
scale would dominate it), and on a shared layer it splits into two genuinely different terms:
`diag B − diag B^exp` (sharing: cross-position covariances on the diagonal) and
`diag B^exp − diag A ⊗ diag G`, which is `Cov(a_j², g_i²)` over `(n, t)` pairs — lot 2's
single-term reading, now only the second half.

### 0.5 The sharing/independence decomposition, and what it costs

`plan_exp_draft.md` §3.1: `B →(cross-position terms dropped) B^exp →(independence) K-FAC-expand`, with

    B^exp = (1/N) Σ_{n,c,t} (g_t g_tᵀ) ⊗ (ā_t ā_tᵀ)        (rvec order, ā bias-augmented)

`B^exp` is a sum of `N·C·T` rank-one Kronecker terms, so its **rearrangement**
`𝓡(B^exp) = (1/N) Σ vec(g gᵀ) vec(ā āᵀ)ᵀ` (lot 2 §0.4) is the natural object: every quantity the
decomposition needs is a cheap contraction of it — `‖B^exp‖_F = ‖𝓡(B^exp)‖_F`,
`⟨B, B^exp⟩ = ⟨𝓡(B), 𝓡(B^exp)⟩`, `⟨B^exp, G ⊗ A⟩ = vec(G)ᵀ 𝓡(B^exp) vec(A)`, and its SVD gives
the best Kronecker fit of `B^exp`. Nothing of size `P × P` is formed beyond `B` itself.

Its cost is not negligible: `N·C·T·d_out²·d_in²` flops. Both `vec(g gᵀ)` and `vec(ā āᵀ)` are
symmetric, so accumulating in half-vectorised coordinates (off-diagonal entries weighted by `√2`,
an isometry on symmetric matrices) cuts it by 4, and the full `𝓡` is unpacked once on the host.
At `N = 45 000`, type-2 (`C = 10`) `[DERIVED]`:

| layer | `P` | `T` | full flops | half-vec flops | `𝓡` fp64 (full / half) |
|---|---:|---:|---:|---:|---:|
| A2 `features.0` | 448 | 1 024 | 9.3e13 | 2.5e13 | — |
| A2 `features.4` | 4 640 | 256 | 2.5e15 | 6.4e14 | 0.17 / 0.05 GB |
| A2 `features.8` | 18 496 | 64 | 9.9e15 | **2.5e15** | 2.74 / 0.70 GB |
| A3 all shared (10 layers) | ≤ 3 168 | 64 | 1.2e15 | 3.2e14 | ≤ 0.08 GB |

A2's decomposition is ~3.2e15 flops per checkpoint (type-2) — at lot 1's measured ≥ 2.65 TFLOP/s on
the 10 GB slice, ~20 min, the single largest item of the A2 jobs. The empirical source is 10× cheaper.

**A precision trap found while prototyping it.** `torch.where(mask, 1.0, math.sqrt(2.0))` returns a
**float32** tensor even when everything around it is fp64; the `√2` weights then carry a
`1e-8` relative error into every norm — above fp64 round-off by eight orders of magnitude, and far
below anything a smoke run would flag. Measured: `‖𝓡‖` off by `9.5e-9` relative, `0.0` once the
weights are built with an explicit `dtype`. The unpack-against-brute-force test (§2) is written to
catch exactly this.

For a normalisation layer `B^exp` is the dense `2C × 2C` sum of `v vᵀ`, `v = [g_t ⊙ x̂_t ; g_t]` —
trivially cheap, and it is what Proposition 3.1's Hadamard form approximates by independence.

### 0.6 A3's `pos_embed`: captured, not silently dropped

`pos_embed` (2 048 of A3's 21 098 parameters) belongs to no module, so `capture.coverage` reports it
uncovered and `build_dense_reference(modules=None)` refuses the model. **Lot 2's runner does not hit
that guard**: it passes `modules=list(capturable_modules)`, so on A3 it would have built a
*restricted* reference and labelled it `F` — 10 % of the parameters gone, `M8`'s coupling matrix
missing a row, with no error anywhere.

**Mechanism:** a raw parameter with a leading singleton dimension that broadcasts over the batch is
replaced, for the traversal only, by a batch-expanded copy (`torch.func.functional_call`); the
gradient w.r.t. row `n` of that copy is example `n`'s per-sample gradient. Prototyped on both
`vit_micro_cifar` (`pos_embed`) and `vit_small_cifar` (`cls_token`, `pos_embed`): the batch sum matches
autograd's parameter gradient exactly, and a single-example row matches a one-example backward to
`1e-18`/`1e-14`.

**The pruning trap again, in a new place** (`plan_exp_lot1.md` §0.1). The traversal backpropagates
with `inputs=[batch]`. The substituted leaf is not on the path to `batch`'s gradient, so the engine
prunes the edge and neither its `.grad` nor a hook on it is ever populated. It must be listed in
`inputs=` and read from `autograd.grad`'s return value.

**And a guard for every capture path, not just this one:** on the first micro-batch of every
reference build the runner checks the **sum rule** — `Σ_n` of the per-sample rows equals the
parameter gradient of `Σ_n ⟨v_n, f(x_n)⟩` computed by a plain backward, to `1e-10` relative. It
catches a wrong tap, a wrong layout, a missing position, a stride or padding error in the patch
extraction, and a BN in the wrong mode; it cannot catch a permutation *across* examples, which is
what `assert_sample_independent` is for. Cost: one extra backward per build.

### 0.7 Normalisation layers: one lot-2 statement holds only at `T = 1`, and one lot-2 finding is a theorem

1. **"Proposition 3.1 is exact on the `β` block"** (`plan_exp_lot2.md` §5.4) is true for A1's
   `LayerNorm` on a 2-D input and false for every normalisation of lot 3. With positions,
   `∂/∂β = Σ_t g_t`, so the exact `β` block is `(1/N) Σ_{n,c} (Σ_t g_t)(Σ_t g_t)ᵀ`, while the Hadamard
   form's `S` is the expand-style `(1/N) Σ_{n,c,t} g_t g_tᵀ`. The gap is the sharing term (§0.5).
2. **Lot 2's `as_implemented` reading is not `diag.py`'s formula.**
   `NormStats.as_implemented_diagonal()` returns `diag(H)·diag(S)` and `diag(S)` — which is
   precisely `diag(hadamard)`. `diag.py`'s `_h_batchnorm2d` / `_h_layernorm` instead sum the **raw,
   pre-normalisation** input over batch *and* positions before squaring, producing a **2-entry**
   `H_D` shared by all channels, and `_s_*` square the batch-summed gradient ("sum-then-square",
   `plan_lot5.md` §0.6). Lot 2 §6.3's reading (d) is therefore the diagonal of reading (b), not the
   shipped formula.
3. **Three of lot 2 §6.3's four orderings are theorems, not findings.** On any block `R`:
   `diag R` is the Frobenius-optimal diagonal, so `e_F(exact_diag) ≤ e_F(any diagonal)` — in
   particular ≤ `e_F(diag(hadamard))`; and `exact_separate` is the Frobenius-optimal matrix with zero
   `γ`-`β` cross blocks, a pattern containing both the Hadamard form's and every diagonal's, so
   `e_F(exact_separate) ≤ e_F(hadamard)` and `≤ e_F(exact_diag)`. Only **Hadamard vs exact diagonal**
   is an empirical comparison. "(d) is the least accurate of the four, even worse than (c)" was
   guaranteed before the run.

**What lot 3 does.** The lot-2 reading is emitted under its true name, `hadamard_diag`. The shipped
formula is added as `diag_py`, computed **by calling** `adafisher_modes.factors.compute_h_diag` /
`compute_s_diag` (§0.12's reuse rule) on training-size micro-batches with the empirical, mean-loss
gradient the optimizer sees, averaged over micro-batches — P1's convention (no EMA, no min-max),
pinned to `DiagApproximation(minmax=False, gammas=(1, 1), Lambda=0).f_tilde` by a test. Its scale is
arbitrary (mean-loss gradients, `1/B²` constants), so it is read through `cos_F`, `e_F*` and `c*`, and
`ρ` is computed on `c*·K`. `GroupNorm` is not hooked by the optimizer, whose treatment of it is the
identity: that is reported as the `identity` structure, added for every layer (the Frobenius-optimal
multiple of `I`; `ρ(identity)` is the plain-gradient step). The theorem orderings are asserted in the
tests, so a violation means a bug, not a finding. `plan_exp_lot2.md` §6.3 gets an erratum pointing
here; its A1 numbers stand, their interpretation does not.

### 0.8 Per-layer noise from folds, at no extra traversal

Lot 2 measured one noise floor, for the whole matrix, by rebuilding forty half-size references
(~49 min at A1's size), and left the per-layer version open (`plan_exp_lot2.md` §6.6). HF2 needs
more than that: an interval on the **difference** between two structures' errors on the same layer.

Every statistic of pass 1 and 2 is additive over probes. The runner therefore accumulates each one
separately on `K = 10` contiguous folds of the (already seeded-random) probe order, offloads each
fold's sums to the host at its boundary, and forms any half as a sum of five folds. One traversal
then yields the full-`N` result **and** `C(10,5)/2 = 126` balanced partitions, of which 20 are used
(`plan_exp_draft.md` §3.4). Each half is marginally a uniformly random half, so each split's
statistic has the right distribution; partitions share folds, which affects only the precision of
the interval's end-points. Declared approximations: EKFAC's `s` on a half is accumulated in the
**full-set** basis (basis noise not included), and `diag_py` has no interval.

Reported: the whole-matrix floor (lot 2's `noise_floor` rows, same definition), a **per-layer floor**
`d(B^(1), B^(2))`, and for every layer-level metric a `ci_low/ci_high` of `value ± 1.96·sd_half/√2`
(variance ∝ `1/N`), plus explicit **paired differences** for HF2. Host memory is the price: ten
`P × P` folds, 47.9 GB for A2, 35.6 GB for A3.

### 0.9 One traversal, several consumers — as lot 2 §0.2 intended

Lot 2 argued for a single traversal and implemented three (reference, factors, EKFAC). Lot 3 adds
two modes, the decomposition and the folds, so the loop is finally shared: **pass 1** feeds the dense
reference, the expand *and* reduce factors, and `B^exp`; **pass 2** feeds EKFAC's `s` for both bases.
`build_dense_reference` and `accumulate_factors` keep their signatures and behaviour — the lot-1 and
lot-2 suites must pass unmodified — and become thin wrappers over the same accumulators.

Two more costs that do not exist on A1 and would crash the job rather than mislead it, both fixed
before submission: **M5's gradient** was one forward over all `N` probes (fine for an MLP on MNIST,
~90 GB of fp64 activations for 45 000 CIFAR images through a CNN) — it is now micro-batched with
exact mean-weighting; and **M7's SVD** of a rearranged block was computed twice per layer (once for
`σ₂/σ₁`, once for the best Kronecker fit) — 4 096 × 83 521 on A2's last convolution, now once.

### 0.10 Pre-registered decision rules

Written before any lot-3 number exists, so the reading cannot be fitted to the data. All on the
type-2 source, seed 0, `diag` trajectory, the five checkpoints. A **cell** is `(model, layer,
checkpoint)`. CI95 is §0.8's fold interval.

**HF2 (ii) — "K-FAC-reduce beats K-FAC-expand in classification"** (`plan_exp_draft_v0.md`: refuted
if the expand/reduce gap is under the noise). Per cell on a shared layer (`T > 1`):
`Δ = e_F(kfac_reduce) − e_F(kfac)`; the cell votes **reduce** if CI95(Δ) < 0, **expand** if > 0,
**tie** otherwise. On one model: *confirmed* if ≥ 2/3 of its cells vote reduce, *refuted* if ≥ 2/3
vote expand or tie, *mixed* otherwise (then reported per layer type). **Decided** when A2 (GN) and A3
agree; A2-BN is a robustness check. `ρ` is reported alongside across the whole `α` sweep but does not
vote (§10.3: no conclusion about an inverse at a single `λ`).

**HF2 (i) — "K-FAC's structure error is larger on shared layers than on unshared Linear".** Per
model and checkpoint: the median `e_F(kfac)` over shared layers against the unshared `Linear` layers
of the same model (the head, for A2 and A3), and against A1's three unshared `Linear` (lot 2's
outputs, a different dataset: descriptive only). *Confirmed* on a model if the shared median exceeds
the head's `e_F` by more than their CIs at ≥ 4 of 5 checkpoints **and** every shared layer's
`sharing_share` exceeds its per-layer noise floor; *refuted* if the shared median is below at ≥ 4 of
5; *mixed* otherwise.

**A Q4 conclusion "holds on ≥ 2 models"** when its rule below gives the same verdict on A1 and on at
least one of A2 (GN) / A3, and does not give the opposite verdict on the other:

| id | claim (lot 2, corrected by §0.7) | rule per model |
|---|---|---|
| Q4.1 (HF3) | AF's diagonal shortcut costs something off MNIST's first layer | `diagonal_bias` (K-FAC-expand's diagonal vs `diag B`) ≥ 0.10 on every non-input layer at ≥ 4/5 checkpoints (v0's threshold; refuted if ≤ 0.05) |
| Q4.2 (HF4) | the `γ`-`β` cross terms carry a non-negligible share of a norm block | `cross_term_share_total` ≥ 0.10 on every norm layer at ≥ 4/5 checkpoints |
| Q4.3 | Hadamard form vs exact diagonal (the only empirical HF4 ordering) | the sign of `e_F(hadamard) − e_F(exact_diag)` agrees on every norm layer at ≥ 4/5 checkpoints |
| Q4.4 | every structure's `ρ` falls between the first and last checkpoint | `ρ(ckpt 1) < ρ(ckpt 0)` for every structure and layer at `α ∈ {1e-3, 1e-1}` |
| Q4.5 | inter-layer coupling is large everywhere | every off-diagonal `M8` entry ≥ 0.5 at ≥ 4/5 checkpoints |
| Q4.6 | EKFAC ≤ K-FAC and best-Kronecker ≤ every Kronecker structure | theorems (§2): asserted, not voted |

### 0.11 The runs

Three models, each `diag` seed 0 at fractions `{0, 0.01, 0.1, 0.5, 1}`, sources type-2 and
empirical, `N = 45 000` (the whole train split of the run's own seeded partition; `N(C−1) = 405 000
≫ P`), folds on type-2, decomposition on both sources, `α ∈ {1e-4, …, 1}`.

| job | model dir | `P` | device peak | host peak | per checkpoint |
|---|---|---:|---:|---:|---|
| A2-GN | `cifar10/cnn_gn_cifar` | 24 458 | ~7.5 GB | ~70 GB | pass 1 + decomposition ≈ 20-25 min, pass 2 + empirical ≈ 5 min, CPU metrics ≈ 10 min `[ESTIMATE]` |
| A2-BN | `cifar10/cnn_gn_cifar_bn` | 24 458 | same | same | same |
| A3 | `cifar10/vit_micro_cifar` | 21 098 | ~5 GB | ~55 GB | ≈ 10-15 min `[ESTIMATE]` |

The estimates are replaced by a **calibration job** (one checkpoint at `N = 4 096` on the GPU)
before the full jobs are sized: lot 2's first cluster job died after 36 s on a device crossing the
CPU-only laptop could not have exposed (`plan_exp_lot2.md` §5.8), and a 20-minute smoke is cheap
insurance for three multi-hour jobs.

### 0.12 What must not change

* The 579 pre-existing tests pass **unmodified** (41 skipped, as today).
* On A1 (`T = 1`) every structure's value is unchanged up to reduction order; the renamed
  `as_implemented → hadamard_diag` is the only schema difference.
* `METRICS_VERSION` becomes `fisher_ref/0.2`: shared-layer semantics changed (§0.1-§0.4) and a
  structure was renamed (§0.7), which is exactly what the invariant exists to record.
* The cluster checkout is updated by copying `fisher_ref/` (and the new tests and docs) only. The
  local working tree's uncommitted `benchmarks/models/` move is **not** deployed; everything this lot
  adds reaches benchmarks only through `discover_benchmarks()` and `bench.build_model`, which exist
  under both layouts.

---

## 1. Modules

| file | change |
|---|---|
| `capture.py` | `raw_parameter_names(model)`; `iter_probe_columns(raw_parameters=...)` substitutes batch-expanded leaves, lists them in `inputs=`, exposes `step.raw_grads`; unchanged code path when empty (§0.6) |
| `reference/dense.py` | `DenseAccumulator` (the row-filling loop, now shared) with `check_sum_rule`; `build_dense_reference(raw_parameters=, check_sum_rule=)`; `ParamLayout.block_slice` accepts a raw parameter name |
| `approx/factors.py` | reduce = mean input / summed gradient (§0.1); TKFAC per `(n, c, t)` (§0.3); EKFAC projects the true per-sample gradient (§0.2); `LayerFactors.merge`, `to()` over every tensor field, `af_raw` at App. A.3 scale |
| `approx/sharing.py` *(new)* | `SharingSums` (half-vectorised `𝓡(B^exp)` for linear/conv, dense `2C × 2C` for norms), unpack, `decomposition(...)` rows (§0.5) |
| `approx/norm_layers.py` | `hadamard_diag` naming, `diag_py_reading(...)` calling `compute_h_diag`/`compute_s_diag` (§0.7) |
| `approx/embed.py` *(new)* | `pos_embed` structures: `position_blockdiag` (cross-position terms dropped — this block's `B^exp`) and `kfac_onehot` (`A = I_T/T`) |
| `approx/base.py`, `kfac.py`, `ekfac.py` | `inner_rearranged(𝓡)`, so M1 against five structures rearranges a block once, not five times (2.7 GB per copy on A2's last layer) |
| `metrics/ngd.py` | `probe_gradient(batch_size=)`, micro-batched, exact mean weighting (§0.9) |
| `metrics/kron_diag.py` | one SVD serving `σ₂/σ₁`, the tail mass and the best Kronecker fit |
| `folds.py` *(new)* | fold assignment, the two passes over folds with host offload, half assembly, partitions, CI helpers (§0.8, §0.9) |
| `runners/p1_structural.py` | the lot-3 structures, the decomposition, raw parameters, folds, `diag_py`, `identity`; lot-2 flags unchanged; new flags off by default except where §0.12 allows |
| `slurm/p1_calibrate_lot3.sh`, `slurm/p1_structural_{a2_gn,a2_bn,a3}.sh` *(new)* | hand-written, each stating its own memory (`fisher_ref/slurm/README.md`) |
| `conventions.py` | `METRICS_VERSION = "fisher_ref/0.2"` |

## 2. Tests — `tests/test_fisher_ref_lot3.py`

| test | statement | tolerance |
|---|---|---|
| **T5-expand** | expand setting (shared `Linear`, per-position output, MSE, deep linear): K-FAC-expand = the GGN block; K-FAC-reduce is not | ≤ `1e-10` / gap > `1e-3` |
| **T5-reduce** | reduce setting (shared `Linear`, mean pooling, deep linear, MSE): K-FAC-reduce = the GGN block; K-FAC-expand is not | ≤ `1e-10` / gap > `1e-3` |
| **T5-conv** | the same reduce statement on a `Conv2d` followed by global average pooling | ≤ `1e-10` |
| **T8-exp** | TKFAC-expand: `tr(K) = tr(B^exp)` on a shared layer; TKFAC-reduce: `tr(K) = tr(B^red)`; both reduce to T8 at `T = 1` | ≤ `1e-10` |
| **T9-shared** | EKFAC-expand and EKFAC-reduce both preserve `tr(B)` on a shared layer (fails under the §0.2 defect) | ≤ `1e-10` |
| **T13-shared** | on a `Conv2d` and a token-wise `Linear`: campaign `A` = `compute_h_full`; campaign `G` = `T ×` `compute_s_full` fed the same `g`; TKFAC-expand's numerators = `instantaneous_raw_factors` on the flattened rows, up to the declared constants | ≤ `1e-10` |
| B^exp brute force | half-vectorised accumulation, unpacked, equals `(1/N) Σ kron(g gᵀ, ā āᵀ)` built densely; `‖𝓡‖ = ‖B^exp‖`; `⟨B, B^exp⟩` and `⟨B^exp, G⊗A⟩` by contraction equal the dense ones | ≤ `1e-12` rel. |
| decomposition at `T = 1` | `B^exp = B` and `sharing_share = 0` on an unshared layer | ≤ `1e-12` |
| raw parameters | `pos_embed` (and `cls_token`) per-sample rows equal `torch.func.vmap(grad)`; `build_dense_reference` over **all** parameters of a small `ViTCIFAR` passes the sum rule; the sum rule **raises** on a corrupted row | ≤ `1e-12` |
| folds | `Σ_k` fold sums = the single-fold accumulation (dense, factors, `s`, `B^exp`); a half built from folds equals a direct build on those probes | ≤ `1e-12` rel. |
| theorem orderings | on real norm, conv and linear blocks: exact_separate ≤ hadamard, exact_separate ≤ exact_diag ≤ hadamard_diag; EKFAC ≤ K-FAC (both modes); best_kron ≤ every Kronecker structure | exact inequalities (+ `1e-12` slack) |
| `diag_py` | one micro-batch equals `DiagApproximation(minmax=False, gammas=(1,1), Lambda=0).f_tilde`, reordered to `named_parameters()`; on `BatchNorm2d` and a 3-D `LayerNorm` | ≤ `1e-12` |
| lot-2 regressions | `hadamard_diag == diag(hadamard)`; at `T = 1` the new TKFAC equals lot 2's | ≤ `1e-12` |
| micro-batched gradient | `probe_gradient(batch_size=b)` equals the one-shot gradient | ≤ `1e-12` |
| runner | on tiny shared models (conv + GN, ViT with `pos_embed`): schema, the new structures present, `ci_low/ci_high` filled when folds are on, reduce rows absent at `T = 1` | — |

## 3. Order of work

1. Tests of §0.1-§0.3 written first, against the current code, and seen failing for the stated reason.
2. `factors.py` fixes; T5, T8-exp, T9-shared, T13-shared green; lot-2 suite unmodified and green.
3. Raw parameters in `capture.py` + `DenseAccumulator` + sum rule; lot-1 suite unmodified and green.
4. `approx/sharing.py`, `approx/embed.py`, `diag_py`, `inner_rearranged`, one-SVD M7, micro-batched M5.
5. `folds.py` and the runner; local smoke on the real A2-GN, A2-BN and A3 checkpoints at small `N`.
6. `ruff`/`mypy`; cluster sync of `fisher_ref/`; calibration job; sizing; the three jobs.
7. §5 findings during the work; §6 when the jobs have written their `metrics.csv`.

## 4. Exit criteria

1. The 579 pre-existing tests pass, unmodified (609 in total with lot 3's 30).
2. T5, T8-exp, T9-shared, T13-shared and every §2 test pass.
3. `ruff` and `mypy` clean on `fisher_ref/`.
4. `metrics.csv` + `meta.json` written for A2-GN, A2-BN and A3 at five checkpoints, with fold intervals.
5. HF2 decided by §0.10's rule; the Q4 table filled, failures included.
6. `CLAUDE.md`, `plan_exp_draft.md` §9 and `plan_exp_lot2.md` §6.3 (erratum) updated.

**Status (2026-09-18).** All six met. 1: 609 tests passing when the lot closed — the 579 that
existed, unmodified, plus its 30, 41 skipped as before (the suite has grown since with work outside
this lot). 2: T5, T8-exp, T9-shared, T13-shared and every §2 test pass. 3: `ruff` and `mypy` clean
on everything lot 3 touches; the remaining findings are in `experiments/` drivers this lot does not
modify. 4: `metrics.csv` + `meta.json` written for the three models at five checkpoints each, with
fold intervals. 5: §6. 6: done.

## 5. Findings during implementation

### 5.1 The three lot-2 branches failed their tests exactly as predicted, and pass after the fixes

Step 1 of §3 wrote §2's tests against lot 2's code first. Five failed, each for its stated reason:

| test | on lot 2's code | after |
|---|---|---|
| T5-reduce, shared `Linear`, `T = 5` | relative error **24.000**, trace ratio 25 | < `1e-10` |
| T5-reduce, `Conv2d` stride 2 on 5×5, `T = 9` | relative error **80.000** | < `1e-10` |
| T8-exp (TKFAC-expand vs `tr(B^exp)`) | fails | `rtol 1e-12` |
| T8-red (TKFAC-reduce vs `tr(B^red)`) | fails | `rtol 1e-12` |
| T9-shared, reduce basis | `tr(K) ≠ tr(B)` | `rtol 1e-12` |

`24 = 5² − 1` and `80 = 9² − 1`: the `T²` factor of §0.1, on both layer types. T5-expand and
T9-shared in the expand basis passed on the old code, as they should — the defects were confined to
the branches A1 never exercised. The 112 lot-1/lot-2 tests passed unmodified after the fixes.

### 5.2 T13-shared: the optimizer's Kronecker operators are `T` times smaller than K-FAC-expand's

Fed identical per-example gradients on a `Conv2d` (`T = 9`) and a token-wise `Linear`, the campaign's
`A` equals `compute_h_full` to `1e-10`, `G = T · compute_s_full`, TKFAC's `Φ` and `Ψ` agree and
`δ = T · δ_optimizer`. So on every shared layer the operator `adafisher_modes` builds is K-FAC-expand
(arXiv:2311.00636 Eq. 7) divided by `T`, so a fixed `λ` weighs `T` times more against it than it
would against K-FAC-expand: 64 on A3's tokens and A2's last convolution, 1 024 on A2's first. This is pinned as a
declared constant, not corrected: it is a property of the optimizer, and its consequence belongs to
P2 (lot 5), which compares the operational state at the optimizer's own scale.

### 5.3 Silent traps met while building, each now guarded

1. **A float32 `√2` in an fp64 computation** (§0.5): `torch.where(mask, 1.0, math.sqrt(2.0))` —
   `1e-8` relative error on every norm of `B^exp`. Caught by building the brute-force comparison
   first; the weights now carry an explicit `dtype`.
2. **`.to("cpu")` is not a copy on the host.** Offloading a fold with it, then zeroing the device
   buffer for the next fold, zeroes the stored fold too — on a CPU run only. Every local test would
   have assembled zeros while the cluster assembled data. `DenseAccumulator.offload` copies
   explicitly, and a test pins it.
3. **A second forward inside a traversal overwrites the capture.** The row checks recompute
   gradients with the same modules; without `Capture.paused()` their tensor hooks would have replaced
   `g` with a *different* gradient that the next consumer of the same step reads as the probe's.
4. **A forward hook that returns a value replaces the module's output.** Met in a test helper written
   as a `lambda` returning a tuple (the network then crashed on `relu(tuple)`); `capture.py`'s own
   hooks return `None`. Noted because a hook returning a *tensor* would not crash.
5. **Lot 2's runner cannot load A2-BN at all**: it resolved the bench by `--model`, and
   `cnn_gn_cifar_bn` is a directory label with no `benchmarks/` folder. The bench now comes from the
   run's manifest (`RunRef.bench_name`), as `checkpoints.py` already documented.
6. **M5's gradient in one forward** would have needed ~90 GB of fp64 activations for 45 000 CIFAR
   images; micro-batched now, equal to the one-shot gradient to `1e-12`.
7. **Unequal folds.** With `N` not a multiple of `folds × batch`, fold sizes differed (72/48 in the
   first smoke) and a partition's two halves had different sample sizes; `FoldPlan` now refuses it.
8. **`diag_py` at fewer probes than one training batch** has no definition (the statistic is
   batch-size dependent); it is skipped and the skip recorded, never computed at another batch size.

### 5.4 The row checks on the real checkpoints

Local smoke runs (CPU, `N = 240`, restricted module sets) on `cnn_gn_cifar/diag/ckpt_0.5`,
`cnn_gn_cifar_bn/diag/ckpt_1` and `vit_micro_cifar/diag/ckpt_0.1` (the last including `pos_embed`):
sum rule `1.0-3.2e-15`, single example `≤ 4.6e-16`, on both sources — GroupNorm, BatchNorm in eval
with the checkpoint's running statistics, stride-4 patch embedding, LayerNorm over tokens and the raw
positional parameter all reconstruct autograd's gradient to round-off. No theorem ordering was
violated. The smoke's metric values are **not** results: at `N = 240` everything sits under the
noise floor (lot 2 §5.6's lesson), and they are not quoted.

### 5.5 Sizing, measured on the GPU smoke rather than extrapolated

Job **21276016** (`p1_lot3_smoke.sh`, `h100_1g.10gb`, `N = 2 000`, two folds, one checkpoint per
model) **COMPLETED in 10:21**, host peak **36.1 GB**, no error and no invariant violation. Every
device path lot 3 added ran: the per-fold host offloads, the eigenbases moved back for pass 2, the
raw-parameter leaves, the row checks' extra backwards, the `diag.py` reading hooked on the GPU, and
the resolution of `cnn_gn_cifar_bn` to its bench through the manifest.

| stage | A2-GN | A2-BN | A3 | scales with `N`? |
|---|---:|---:|---:|---|
| pass 1, type-2 (dense + both K-FAC modes + `B^exp`) | 107.6 s | 107.6 s | 21.2 s | yes |
| pass 2 + assembly | 2.2 s | 2.2 s | ~1 s | yes |
| per-block metrics (host) | 44.1 s | 44.5 s | 34.5 s | **no** |
| fold intervals, per partition (host) | 24.1 s | 23.7 s | 11.5 s | **no** |
| pass 1, empirical | 11.8 s | 11.8 s | ~2 s | yes |

At `N = 45 000` (×22.5 on the linear parts) and 20 partitions: **~60 min per checkpoint for A2**
(~5 h for five) and **~17 min for A3** (~1.5 h). The jobs request `09:00:00` / `04:00:00`, and
`128G` / `96G` — the host memory is set by `P` and the fold count, not by the probes: ten `P × P`
folds are 47.9 GB for A2 and 35.6 GB for A3, which the smoke's 36.1 GB at two folds confirms.

The row checks on the GPU, at `N = 2 000`, on all three models and both sources: sum rule
`2.0e-16`-`5.7e-16`, single example `2.4e-16`-`5.3e-16`.

## 6. Results

Three jobs, all `COMPLETED` on 2026-09-17 with no error, no retry and **zero** `invariant_violation`
rows: **21276896** (A3, `01:02:33`, host peak 54.3 GB), **21276894** (A2-GN, `04:30:41`, 80.7 GB) and
**21276895** (A2-BN, `04:20:50`, 80.7 GB) — each at `N = 45 000` probes (the whole train split of the
run's own seeded partition), ten folds, 20 partitions, five checkpoints, both sources. Raw numbers
under `fisher_ref/outputs/<model>/diag/seed0/<fraction>/`. Every verdict below is produced by
`fisher_ref/experiments/lot3_decisions.py` applying §0.10's **pre-registered** rules; it chooses no
threshold of its own.

### 6.1 What can be trusted here, and why it differs from lot 2

Lot 2 had to reject its own smoke numbers because everything sat under the noise. At `N = 45 000`
the situation is reversed: the whole-matrix split-half distance is **0.005-0.039** depending on the
model and checkpoint (A1's, at `N = 55 000`, was 0.343), per-layer floors run from 0.002 to 0.18, and
the differences below are ten to a hundred times larger. Every layer-level number in the CSVs carries
a fold interval; every difference quoted as decisive has a CI excluding zero.

Both in-job checks passed everywhere, on every model, source and checkpoint: per-sample rows against
autograd (`≤ 5.3e-16` for the sum rule and the single-example rule) and the theorem orderings.

### 6.2 HF2 is decided: **confirmed, on all three models**

**(ii) K-FAC-reduce beats K-FAC-expand.** Per §0.10, a cell is `(shared layer, checkpoint)` and votes
on the fold CI of `Δe_F = e_F(reduce) − e_F(expand)`:

| model | cells | reduce | expand | tie | reduce-win range | expand wins |
|---|---:|---:|---:|---:|---|---|
| A2-GN `cnn_gn_cifar` | 15 | **14** | 1 | 0 | `−0.713 … −0.152` | `features.0` at 1 %: `+0.036` |
| A2-BN `cnn_gn_cifar_bn` | 15 | **15** | 0 | 0 | `−0.719 … −0.113` | — |
| A3 `vit_micro_cifar` | 45 | **40** | 5 | 0 | `−0.935 … −0.082` | `patch_embed.proj` at θ=0 (`+0.779`); four others at 1-10 % (`+0.019 … +0.179`) |

Every layer's own verdict is "reduce" on all three models, so §0.10's requirement (A2-GN and A3
agree) is met, with A2-BN as a third, independent confirmation. The asymmetry matters: reduce's wins
are large, expand's are small and concentrated in the earliest checkpoints — except the ViT's patch
embedding at initialisation, the one layer whose "positions" are disjoint image patches rather than a
sequence the network later pools.

**(i) K-FAC's structure error is larger on shared layers than on unshared ones.** Confirmed on all
three, at every checkpoint, with the mechanism required by the rule (each shared layer's
`sharing_share` above its own floor):

| model | median `e_F(K-FAC)` over shared layers | the model's head (unshared) |
|---|---|---|
| A2-GN | 0.951 → 0.971 (θ=0 → θ=1) | 0.011 → 0.252 |
| A2-BN | 0.975 → 0.975 | 0.005 → 0.359 |
| A3 | 0.954 → 0.971 | 0.033 → 0.637 |

**And the decomposition says why.** The median `sharing_share` — what dropping the cross-position
terms alone costs — is **0.971 on all three models** (range 0.78-0.995), while the independence step
on top of it costs 0.45-0.60. So on a shared layer, K-FAC's error is **first of all a weight-sharing
error**, not the independence assumption the literature usually discusses: the object K-FAC-expand
factorises (`B^exp`) has already lost ~97 % of the exact block.

**The Frobenius winner is not the step winner** — and this is the caveat that must travel with HF2.
Across the five dampings and every shared layer, `ρ(reduce) < ρ(expand)` in **70/75** cells on A2-GN,
**62/75** on A2-BN and **162/225** on A3: reduce is better in norm and usually *worse* for the update
it would produce. On A3's `qkv` at `α = 1e-3`, θ=1: `e_F` 0.726 (reduce) vs 0.971 (expand), `ρ` 0.143
vs 0.453. §0.10 votes on `e_F` and reports `ρ` across the whole sweep precisely because the two
disagree; anyone choosing between the two variants **for an optimizer** should read `ρ`, not `e_F`.

A related observation, visible at initialisation on every model: the exact block is nearly Kronecker
(`best_kron` = 0.047 on A3's `qkv`, 0.217-0.367 on A2's convolutions) while K-FAC-expand sits at
0.95-0.97. The Kronecker *structure* is not what fails there — K-FAC's own choice of factors is.

### 6.3 The Q4 conclusions, checked on ≥ 2 models

| id | claim | A1 (lot 2) | A2-GN | A2-BN | A3 | holds on ≥ 2? |
|---|---|---|---|---|---|---|
| Q4.1 | AdaFisher's diagonal shortcut costs something off the input layer | confirmed | confirmed | confirmed | confirmed | **yes** |
| Q4.2 | the `γ`-`β` cross terms are not negligible | confirmed | confirmed | confirmed | confirmed | **yes** |
| Q4.3 | Hadamard form vs exact diagonal, in `e_F` | Hadamard better | exact diag. better | exact diag. better | exact diag. better | **no — and §6.4 explains it** |
| Q4.4 | every structure's `ρ` falls along training | refuted (95 % of pairs) | refuted (52 %) | refuted (55 %) | refuted (71 %) | **no, as an "every"** |
| Q4.5 | inter-layer coupling ≥ 0.5 everywhere | confirmed (min 0.63) | refuted (min 0.44) | confirmed (min 0.49-0.67) | refuted (min 0.28) | **no** |

Q4.1's magnitudes are worth quoting: the diagonal bias at θ=1 is 0.86-0.90 on A2's convolutions and
0.83-0.97 on A3's token-wise layers, against 0.25-0.41 on the heads — an order of magnitude past
v0's 10 % threshold. Q4.2's cross-term share is 0.65-0.70 on A2's normalisations and 0.39-0.62 on
A3's.

Q4.4 is the interesting failure. Lot 2 §6.4's headline — "every approximation gets markedly worse
along training" — is a strong **tendency**, not a law: it holds for 95 % of (layer, structure) pairs
on A1 but only 52-71 % on the shared-weight models. The claim to keep is the tendency with its rate,
not the universal.

Q4.5 fails in an informative direction: the smallest inter-layer coupling is 0.63-0.73 on A1's MLP
but 0.28-0.50 on the ViT and 0.44-0.58 on the GroupNorm CNN. Layers are *less* entangled in the
architectures with weight sharing, so HF5's "the block-diagonal loses nothing" is more defensible
there than on an MLP — the opposite of what an MLP-only study would have suggested.

### 6.4 Proposition 3.1 under weight sharing: right direction, wrong size by 20-38×

Q4.3's reversal is not a disagreement about shape, and the measurement says so exactly. The optimal
rescaling `c*` of the Hadamard reading is **0.95 on A1** (no sharing) but **27.8 / 20.4 / 37.9** on
A2-GN / A2-BN / A3. Direction-wise (`cos_F`, which removes scale) the ranking is the same on all
four models — `exact_separate` (0.72-0.89) > `hadamard` (0.43-0.81) > `hadamard_diag` (0.34-0.36) ≈
`exact_diag` (0.20-0.29) — so **Proposition 3.1's form points the right way and is 20-38× too small**
once a weight is applied at many positions. The reason is structural and was predicted by §0.7: the
Hadamard form is built from expand-style statistics, i.e. it approximates `B^exp`, while the exact
`(γ, β)` block carries the cross-position sums the layer actually produces.

Two consequences. First, a corrected normalisation estimator — the campaign's own decision rule in
`plan_exp_draft.md` §11 — should carry a sharing-aware scale, not just the `γ`-`β` cross term.
Second, **the reading the optimizer ships is the weakest of the five, now measured rather than
argued**: `diag_py` (the faithful `diag.py` formula, §0.7) has `c*` of order `10^6` — its scale is
meaningless, as expected of a "sum-then-square" batch statistic — and even by direction it is last in
13/15 cells on A2-BN and 15/25 on A3, the remaining cells being won by `hadamard_diag`, the other
diagonal. This is the claim lot 2 §6.3 made about an object that was not the shipped formula; it now
holds for the shipped formula, and is a measurement rather than a theorem.

### 6.5 What is open after this lot

* **One seed, one arm.** Everything above is `diag`, seed 0. The `θ`-dependence question
  (`plan_exp_draft.md` §3.5) — does the drift at a given `θ` depend on which optimizer produced it —
  needs the `adamw` arm's checkpoints, which exist and cost one more job per model.
* **`ρ` and `e_F` disagree systematically** on the expand/reduce question. Which one predicts
  optimizer behaviour is a P2/lot-5 question, and this lot supplies the P1 half of the comparison.
* **The `B^exp` decomposition is measured for type-2 and empirical but only on layers up to
  `P ≤ 8 192` for its *interval*** (`--noise-decomposition-max-p`), so A2's last convolution has a
  point estimate without one.
* **Erratum (lot 5, `plan_exp_lot5.md` §6.6): `ekfac_reduce`'s inverse-based metrics are not
  reproducible.** The reduce output factor `G_reduce = (1/N) Σ_n (Σ_t g_{n,t})(Σ_t g_{n,t})ᵀ` is
  **exactly rank-deficient** on any layer followed by a normalisation, because that normalisation's
  backward makes the gradient sum to zero within each group and `reduce` sums over exactly the
  positions the constraint applies to. Measured on A2: `rank(G_reduce) = channels − num_groups`,
  **8 of 16** on `features.0` and 24 of 32 on `features.4`. Inside the resulting degenerate null
  space the eigenvectors are numerically arbitrary. K-FAC does not notice — its eigenvalue is
  constant there, so any rotation leaves `1/(λ_G λ_A + λ)` alone — but **EKFAC does**, because it
  assigns a data-estimated `s` that is not constant on that subspace. Measured: a `3.2e-16` relative
  perturbation of `G_reduce` rotates the null-space eigenvectors by **0.53**, moves `kfac_reduce`'s
  applied step by `1.2e-12` and **`ekfac_reduce`'s by `0.144`**.
  **§6.2's HF2 verdict is unaffected** — §0.10 votes on type-2 `e_F`, which involves no inverse, and
  lot 5 reproduced every type-2 row of this lot to `3.47e-13`. But **no `ρ`, `cos_ngd` or Stein-KL
  number for `ekfac_reduce` should be quoted to better than ~10 %**, and §6.2's "`ρ(reduce)` <
  `ρ(expand)` in 70/75, 62/75, 162/225 cells" should be read as a statement about `kfac_reduce`
  (whose `ρ` is stable to `1e-12`) rather than about the EKFAC pair.

* Regime B (lot 4) inherits the machinery: `folds.py` and `approx/sharing.py` are written against
  the `CurvatureBlock` protocol, not against the dense reference.
