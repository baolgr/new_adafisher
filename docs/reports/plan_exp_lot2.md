# Lot 2 — the approximation zoo, the metrics, and P1 on A1

*Implementation plan for lot 2 of `plan_exp_draft.md` §9. Continues `plan_exp_lot1.md`, which
delivered the exact references this lot is judged against.*

**Goal.** Turn "we have `F`, `Ê` and `B_ℓ`" into "we know how far each structured approximation is
from them, and by how much more than the noise". Three deliverables, in the plan's own words:
the zoo (`approx/`), the metrics (`metrics/`), and `p1_structural.py` run on A1 over the five
checkpoints × the available seeds, with the noise floor.

**Exit criteria** (`plan_exp_draft.md` §9, §10.2): **T3, T4, T7, T8, T9, T13** pass (T5 is shared
with lot 3, which builds the shared-weight models); the noise floor is plotted; **HF3** and **HF4**
are decided on A1.

**Non-goal.** No regime B (lot 4), no P2 / `AdaFisherMulti` state (lot 5), no new benchmark model.
`fisher_ref/` stays a reader: nothing in `benchmarks/` or `src/adafisher_modes/` changes.

---

## 0. Decisions taken here

### 0.1 Lot 1's open defect is this lot's first commit, not a later chore

`plan_exp_lot1.md` §6.3 left HF1 unresolved for an instrumentation reason, not a scientific one:
the three held-out gaps were reported against **two different denominators** (`‖F_train‖` for
train/val and train/test, `‖F_val‖` for val/test) and only the train-side norms were stored. A
`‖F_val‖` 1.6× `‖F_train‖` would reconcile `train/test = 1.44 × train/val` with
`val/test = 1.02 ×` its null, but a 5 000-probe estimate's Frobenius norm should be inflated by
~12 %, not 60 %.

So `DenseReference` gains `trace` and `fro` in its `metadata()`, every reference built by a runner
records them, and **every gap is reported with the norms of both operands**, so a reader can
re-normalise. This is a precondition for M1 anyway — `e_F` and `cos_F` need `‖R‖_F` and `‖K‖_F`
separately, not just their difference — so it is not a detour.

### 0.2 One traversal of the probes, several consumers

The zoo's factors (`A`, `G`, EKFAC's `s`, TKFAC's `Φ`, `Ψ`) are built from exactly the `(a, g)`
`capture.py` already produces for the references. Lot 1's `build_dense_reference` owns that loop —
micro-batch, forward, root, one backward per column — and discards each capture immediately.

Lot 2 extracts the loop into `capture.iter_probe_columns(...)`, a generator yielding the live
`Capture` once per `(micro-batch, column)`, and makes `build_dense_reference` its first consumer.
Two reasons, the second decisive:

1. a second copy of that loop is a second place for the `inputs=[batch]` / tensor-hook subtleties of
   `plan_exp_lot1.md` §0.1 to be got wrong;
2. **the reference and the approximations must see the same probes and the same Monte-Carlo draws.**
   With two passes they would agree only if every RNG were threaded identically; with one traversal
   it is true by construction, and `‖R − K‖` stops carrying a sampling difference between `R` and
   `K` that no metric could distinguish from structure error.

The 48 lot-1 tests are the safety net: they must pass **unmodified** after the extraction.

### 0.3 `P1` re-implements the structures, and T13 is what makes that safe

`plan_exp_draft.md` §0.12 and §7.2 already settle this: `adafisher_modes` computes EMA'd, damped,
hook-driven factors at training precision and has no *source* parameter, while P1 needs the
structure alone, in fp64, from type-2 vectors. So `approx/` owns its own K-FAC, EKFAC and TKFAC.

**T13 is the price and the point**: each of them is pinned to `adafisher_modes`' own at the
degenerate setting (one update, EMA replaced by the identity, empirical source, `λ = 0`), to
`1e-10`. That turns "our K-FAC" and "the optimizer's K-FAC" into one measured statement instead of
two implementations nobody compared. If a T13 fails, the *campaign's* implementation is what gets
questioned first — the optimizer's is pinned by 238 pre-existing tests.

### 0.4 The rearrangement operator is one primitive, shared by M1 and M7

For a block `B` of size `(d_out·d_in)²` in `rvec` order, the rearrangement `𝓡(B)` of shape
`(d_out², d_in²)` satisfies `⟨B, G ⊗ A⟩ = vec(G)ᵀ 𝓡(B) vec(A)` and `‖B − G ⊗ A‖_F =
‖𝓡(B) − vec(G)vec(A)ᵀ‖_F`. So:

* **M7** is literally the singular spectrum of `𝓡(B)` — `σ₂/σ₁` is the independence bias, and the
  best rank-1 Kronecker fit is its top singular pair (Van Loan–Pitsianis, the tool
  Koroko et al. arXiv:2201.10285 already used on auto-encoders);
* **M1** against any Kronecker structure needs `⟨R, K⟩`, which is the same bilinear form.

Building it once, with a blocked implementation, is therefore not a convenience: it is what stops
M1 from densifying `G ⊗ A` (5.05 GB for A1's first layer) to take an inner product. `𝓡(B)` is a
permutation-reshape of `B`, so it costs one pass and no extra asymptotic memory when done in row
blocks — the same discipline as `symmetrize_` and `relative` in lot 1 (§0.9 there).

### 0.5 Which metrics, and what "regime A" buys

`plan_exp_draft.md` §0.11 fixes the priority: **M1, M5, M7, M8 core, M3 where a Cholesky is
affordable, M2/M4/M6 optional**. Lot 2 implements the five core ones and leaves M2/M4/M6 out — they
need Lanczos machinery to answer what M1 and M5 already answer.

One thing regime A buys that the plan does not spell out: at `P = 26 634` a **Cholesky of `R_λ` is
5.68 GB and ~2 min**, far cheaper than the `eigvalsh` lot 1 measured at 1 280 s, because
`potrf` is `P³/3` against `syevd`'s `(4/3)P³` *and* it threads. So M3 and M5 are affordable on the
full matrix, and the `λ` sweep (§3.3, five values of `α`) costs one factorisation each.

### 0.6 Damping is swept, never fixed

`plan_exp_draft.md` §3.3: no fixed `λ`. Every metric that goes through an inverse is computed at
`λ = α·λ̄` with `λ̄ = tr(R)/P` and `α ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}`, and reported as a curve.
`metrics.csv` carries `lambda_alpha` as a column precisely so no single-`λ` number can be quoted
(§10.3). For A1, `λ̄ = tr(F)/P = 16.68/26634 = 6.26e-4`.

This matters more on A1 than the plan anticipated: `plan_exp_lot1.md` §6.4 measured `F`'s kernel at
**4 805 dimensions** and the head block's at exactly `d_in + 1`, so *every* inverse metric reads
`λI` on a fifth of the space whatever `α` is. The `α` sweep is what separates "the approximation is
wrong" from "the damping is doing the work".

### 0.7 HF3 and HF4 are decided from the exact blocks, not from a fit

Both exit criteria are read off objects lot 1 already produces, which is why they belong here:

* **HF3** (`AdaFisher`'s diagonal): `diag(A) ⊗ diag(G) = diag(A ⊗ G)`, so AdaFisher's raw estimator
  *is* K-FAC's diagonal. Two paths lead from `B_ℓ` to it, and §3.1 gives the gap entry-wise as
  `E[a_j² g_i²] − E[a_j²]E[g_i²] = Cov(a_j², g_i²)` when there is no sharing. A1 has no sharing, so
  the covariance is computable exactly and HF3 is a *measurement*, not an estimate.
* **HF4** (normalisations): the exact `2C × 2C` block on `(γ, β)` against (i) the same with the
  cross terms dropped, (ii) the Hadamard form of Proposition 3.1, (iii) `diag.py`'s own
  "sum-then-square" diagonal. A1 has two `LayerNorm(32)`, i.e. two `64 × 64` blocks — trivially
  affordable, and `plan_exp_lot1.md` §6.4 already reports them full rank.

Note what A1 can and cannot decide: it has **no weight sharing**, so HF2 (the expand/reduce
question) and T5 are lot 3's, on A2/A3.

---

## 1. Modules

### 1.1 `fisher_ref/approx/base.py`

`plan_exp_draft.md` §7.1's protocol, verbatim, plus the two things every implementation needs to
answer M1 without densifying:

```python
class CurvatureBlock(Protocol):
    P: int
    def matvec(self, v) -> Tensor              # K v
    def apply_rows(self, U) -> Tensor          # rows u_i -> K u_i
    def solve(self, v, lam) -> Tensor          # (K + lam I)^{-1} v
    def trace(self) -> Tensor
    def fro2(self) -> Tensor                   # ||K||_F^2, closed form
    def logdet(self, lam: float) -> Tensor
    def diag(self) -> Tensor
    def to_dense(self) -> Tensor               # regime A only
    def inner_dense(self, R: Tensor) -> Tensor # <R, K>, without densifying K
```

`inner_dense` is the addition. For `Kron` it is §0.4's bilinear form; for `Diag` it is
`R.diagonal() @ self.values`; for `Dense` it is the plain inner product. Without it, M1 against a
Kronecker structure has to materialise `G ⊗ A`.

Implementations, in `approx/`: `Dense`, `Diag`, `Kron(A, G)`, `EKFAC(QA, QG, s)`,
`ScaledKron(delta, Phi, Psi)` (TKFAC), `BlockDiag([...])`, `HadamardNorm(H, S)`.

### 1.2 `fisher_ref/approx/factors.py` — building the zoo from one traversal

```python
@dataclass
class LayerFactors:            # accumulated over the whole probe set, per layer
    name: str; kind: str; n_rows: int
    A: Tensor                  # (d_in[+1])^2   input second moment, K-FAC-expand normalisation
    G: Tensor                  # d_out^2        output second moment
    gram: Tensor               # the exact block B_l, when regime A can afford it
    tkfac: TkfacStats          # delta, Phi_raw, Psi_raw  (plan_exp_draft.md §4, arXiv:2011.10741)
    norm: Optional[NormStats]  # x_hat/g second moments for HF4's Hadamard form

def accumulate_factors(model, inputs, targets, *, source, ...) -> Dict[str, LayerFactors]
def kfac(factors)  / ekfac(factors, ...) / tkfac(factors) / af_raw(factors) / exact_diag(block)
```

EKFAC's `s_ij = (1/N) Σ_{n,c} [(Q_Gᵀ G_{n,c} Q_A)_ij]²` needs a **second pass** over the probes
once `Q_A`, `Q_G` are known — that is inherent to EKFAC, not a design choice, and it is why
`iter_probe_columns` (§0.2) is a generator rather than a one-shot builder.

### 1.3 `fisher_ref/metrics/`

One module per metric family, each taking `(R: Tensor | LowRank, K: CurvatureBlock, lam)`:

| module | metric | notes |
|---|---|---|
| `frobenius.py` | **M1** `e_F`, `cos_F`, `e_F*`, and the optimal rescaling `c* = ⟨R,K⟩/‖K‖²` | uses `inner_dense`; `c*` is required for the scale-free structures (§3.3) |
| `stein.py` | **M3** `D_λ(K‖R)` | dense Cholesky in regime A; §5.1's Woodbury form is lot 4's |
| `ngd.py` | **M5** `ρ(K)` | `d = K_λ^{-1} g`, `g` the true gradient at that `θ` on the probes |
| `kron_diag.py` | **M7** rearrangement `σ₂/σ₁`, tail mass, sharing share, diagonal bias | §0.4's shared primitive lives here |
| `coupling.py` | **M8** `c_{ℓℓ'}` | from the dense blocks in regime A |
| `noise_floor.py` | §3.4: 20 random partitions → 95 % interval; `d(F_{N'}, F_N)` against `N'` | replaces lot 1's single split |

### 1.4 `fisher_ref/runners/p1_structural.py`

The first real runner: a CLI (unlike `experiments/`), because it sweeps
`(model, arm, seed, fraction) × structure × source × λ`. Writes `plan_exp_draft.md` §13's format —
`metrics.csv` with the promised columns plus `meta.json` — under
`fisher_ref/outputs/<model>/<arm>/seed<n>/<fraction>/`.

---

## 2. Tests — `tests/test_fisher_ref_lot2.py`

| Test | Statement |
|---|---|
| **T3** | *KFAC from scratch* Test 1 (`N = 1`, no sharing): K-FAC-type-2 = the GGN block, K-FAC-emp = the EF block, ≤ `1e-12` |
| **T4** | its Test 2 (deep linear, MSE): K-FAC-type-2 = the GGN block ≤ `1e-12`, **and K-FAC-emp ≠ EF** by > `1e-3` |
| **T7** | `D_λ(R‖R) = 0` and the dense Stein KL against a direct `logdet`/`trace` evaluation, ≤ `1e-8` |
| **T8** | TKFAC: `tr(K) = tr(B_ℓ)` on an unshared layer, ≤ `1e-10` |
| **T9** | EKFAC: `tr(K) = tr(B_ℓ)` ≤ `1e-10`, and a cross-check against `reference_repos/EKFAC-pytorch` ≤ `1e-6` |
| **T13** | `Kron`, `EKFAC`, `ScaledKron` against `adafisher_modes`' own factors at the degenerate setting, ≤ `1e-10` |
| — | the rearrangement identity `⟨B, G⊗A⟩ = vec(G)ᵀ𝓡(B)vec(A)` and `‖B − G⊗A‖_F = ‖𝓡(B) − vec(G)vec(A)ᵀ‖_F` |
| — | `CurvatureBlock` conformance: for every implementation, `to_dense()` agrees with `matvec`, `trace`, `fro2`, `diag`, `inner_dense`, `solve` |
| — | `iter_probe_columns` refactor: `build_dense_reference` unchanged bit-for-bit (the lot-1 suite, unmodified) |
| — | M1/M5/M7/M8 against dense brute force on a tiny model |
| — | the noise floor's 20-partition interval contains lot 1's single-split point estimate |

---

## 3. Order of work

1. §0.1's denominator fix + `trace`/`fro` in every reference's metadata.
2. `capture.iter_probe_columns` extraction; lot-1 suite green **unmodified**.
3. `approx/base.py` + `Dense`, `Diag`, `Kron`, `BlockDiag`; conformance tests; **T3**, **T4**.
4. `approx/factors.py` accumulation; `EKFAC`, `ScaledKron`, `HadamardNorm`; **T8**, **T9**, **T13**.
5. `metrics/` M1, M7 (+ the rearrangement primitive), M5, M3, M8, noise floor; **T7**.
6. `runners/p1_structural.py`; the A1 run; HF3 and HF4 decided; the noise-floor figure.

Steps 1-5 are laptop-verifiable on tiny models. Step 6 is the cluster, sized from
`plan_exp_lot1.md` §6.1: the references are ~150 s each, a Cholesky ~2 min, and the whole
`(5 fractions × available seeds × 2 sources)` grid is dominated by the reference builds, not the
metrics.

## 4. Exit criteria

1. The 530 pre-existing tests pass, unmodified.
2. T3, T4, T7, T8, T9, T13 pass.
3. `ruff` and `mypy` clean on `fisher_ref/`.
4. `metrics.csv` + `meta.json` written for A1, with the noise-floor interval and the `λ` sweep.
5. HF3 and HF4 decided on A1, with their numbers recorded in §6 here.
6. `CLAUDE.md` and `plan_exp_draft.md` §9 updated.

## 5. Findings

### 5.1 `plan_exp_lot1.md` §6.3 is resolved: the held-out Fisher is *bigger*, and the tension was a denominator

Step 1 was worth doing first. With every reference's own `‖·‖_F` recorded, the inconsistency lot 1
could not diagnose — `train/test = 1.44 × train/val` while `val/test` sat at `1.02 ×` its null —
has a one-line cause. Measured on A1 at `θ = 0.5`, the three references restricted to
`{features.1, features.4, head}` at `n = 5 000` each:

| reference | `n` | `‖·‖_F` | vs `‖F_train‖` |
|---|---:|---:|---:|
| `F_train` | 5 000 | 0.6026 | 1.00 |
| `F_val` | 5 000 | 0.7827 | **1.30** |
| `F_test` | 5 000 | 0.9564 | **1.59** |

Lot 1 guessed that "a `‖F_val‖` about 1.6× `‖F_train‖` would reconcile all three" and rejected it
because a 5 000-probe estimate's norm should be inflated by only ~12 % by noise. **The guess was
right and the rejection was wrong**: the inflation is not noise, it is signal. At 50 % of its
trajectory the model is *more confident* on the points it trained on, so `Λ_n = diag(p) − p pᵀ` is
smaller there, and with it the whole Fisher. The held-out Fisher is genuinely **larger**, and
`F_test` — the set furthest from the training distribution — is the largest of the three.

Read against a denominator that does not privilege one operand (the geometric mean of the two
norms), the three gaps become coherent:

| gap | `/‖F_train‖` (lot 1's reading) | `/sqrt(‖a‖‖b‖)` |
|---|---:|---:|
| train/val | 0.4651 | **0.4081** |
| val/test | 0.4199 *(this one was `/‖F_val‖`)* | **0.3799** |
| train/test | 0.7066 | **0.5609** |

`val/test < train/val < train/test` — val and test are closer to each other than either is to
train, and test is furthest, which is exactly what the norm table predicts. The 1.44 factor was two
of the three gaps being divided by the smallest of the three norms.

**Two consequences for the campaign, both larger than the bug.**

1. **HF1 has a scale component and a direction component, and they must be separated.** A
   train/val difference that is mostly "the held-out Fisher is 30 % bigger" is a different claim
   from "the held-out Fisher points elsewhere". This is precisely what M1 distinguishes —
   `cos_F` and the optimally rescaled `e_F* = sqrt(1 − cos_F²)` against the raw `e_F` — so **HF1 is
   settled by M1, not by the Frobenius gaps lot 1 reported**, and §0.5's choice to implement M1
   first is now load-bearing rather than conventional.
2. **Any gap between references computed on different probe sets must carry both norms.** The
   `gap()` helper now returns `abs`, `fro_a`, `fro_b` and three ratios, and `metrics.csv` will do
   the same. A bare ratio is not a reportable quantity when the two operands have different scales.

Caveat: measured on the restricted three-module block set (`P = 458`), not the full `P = 26 634`.
The mechanism is identified and the instrumentation fixed; the full-scale numbers come with the
lot-2 cluster run.

### 5.2 The `rvec([W | b])` permutation is real, and T3 is what found it

`plan_exp_lot1.md` §0.5 chose the `named_parameters()` column layout and deferred the permutation
into `rvec([W | b])` to "the lot that has a K-FAC to compare against", predicting it would be a
small index vector. It is — and **T3 failed on its first run because of it**, with the diagnostic
signature of a layout bug rather than a mathematical one: the two matrices held *the same values in
a different order*.

A reference stacks `rvec(W)` then `b`, two contiguous parameters. A Kronecker product `G ⊗ A` with
a bias-augmented `A` interleaves the bias as the last **column of every output row**. So
`ParamLayout.augmented_permutation(module)` maps `i·(d_in+1) + j ↦ i·d_in + j` for `j < d_in` and
`i·(d_in+1) + d_in ↦ d_out·d_in + i`, and `to_augmented(block, perm)` applies it to both axes. It
refuses a normalisation layer outright: a `(γ, β)` block is Hadamard-structured and has no
`[W | b]` reading at all (`plan_lot5.md` §0.1).

Worth recording because the failure mode is silent in the other direction: had T3 been written
against a *symmetric* fixture, or with a loose tolerance, the permutation would have been missed
and every Kronecker metric of this campaign would have been comparing shuffled matrices.

### 5.3 T3 and T4 pass, and they say different things

* **T3** (`N = 1`, no sharing): K-FAC is **exact**, to `1e-12`, for the type-2 *and* the empirical
  source. There is nothing for the independence assumption to be wrong about — the single
  per-sample gradient is one outer product `g aᵀ`, so `B = (g gᵀ) ⊗ (a aᵀ)` identically.
* **T4** (deep linear, MSE, `N = 16`): K-FAC-type-2 still reproduces the GGN block to `1e-12`,
  while K-FAC-**empirical** does **not** reproduce the EF block — the gap exceeds `1e-3`, as the
  test demands. With `Λ = (2/D)I` the type-2 columns span the output space uniformly and the
  per-column outer products factorise; the empirical source has one data-dependent vector per
  sample and `E[(g aᵀ)(g aᵀ)ᵀ]` does not split.

Together they bound what a K-FAC result can mean: the structure is exact in a degenerate corner and
provably not exact as soon as the source stops being type-2, which is the setting `AdaFisherMulti`
actually runs in.

### 5.4 Step 4: T8, T9, T13 pass, and two bugs the identities caught

The zoo needed **one** new block class, not four. TKFAC is `δ·Φ ⊗ Ψ`, which in `rvec` order is a
`Kron` with the scalar folded into the output factor — and folding it there rather than carrying it
alongside is what keeps `solve` and `logdet` exact, since `δK + λI` is not Kronecker but
`kron(δΨ, Φ)` still is. A normalisation layer's four readings are `Dense`/`BlockDiag`/`Diag`. Only
EKFAC, which carries an eigenbasis, is genuinely new. Fewer representations, fewer places for the
`rvec` convention to go wrong — and both bugs below were exactly that kind of error.

**The trace identities are not decoration; they are what failed.**

* **T9 passed first time** — `tr(K_EKFAC) = tr(B_ℓ)` to `1e-10`. It cannot fail for a data reason:
  `Σ_ij s_ij` is the mean of `‖Q_Gᵀ G Q_A‖_F²` and an orthogonal change of basis preserves the
  Frobenius norm, so a failure would mean the bases are not orthonormal (asserted separately).
* **T8 failed by a factor of exactly 3** on a 3-class problem — i.e. by `C`. TKFAC's `δ` was being
  divided by `N·C` because the accumulator incremented its count once per `(probe, column)` step,
  while the reference normalises `B_ℓ` by `N` and *sums* over the `C` columns. The gap being
  precisely the class count is what made it a two-minute diagnosis rather than a search.
* **EKFAC's conformance test failed on a transposition**: the second contraction took `Q_G[i, o]`
  where it needed `Q_G[o, i]`. The eigenvectors are the *columns*, so the operator was being applied
  in a mirrored basis — `trace`, `fro2` and `diag` were all still correct, because they never touch
  the basis orientation. Only `matvec` against `to_dense()` sees it. This is the `CLAUDE.md` pitfall
  "compare the applied preconditioners, never the bases", in its constructive form.

**T13 passes**: this campaign's `A` and `G` agree with `adafisher_modes`' own `compute_h_full` /
`compute_s_full` to `1e-10` at the degenerate setting (one update, no EMA, empirical source, no
damping). The plan's §0.12 worry — "two implementations nobody compared" — is now one measured
statement. It works because A1 is unshared (`T = 1`): at `T > 1` the two normalisations differ by
`T` by construction, which is HF2 and lot 3's business, not a discrepancy.

**HF4 is instrumented and its four readings all exist** on a real `LayerNorm`: the exact `2C × 2C`
block, the same with the `γ`-`β` cross terms dropped, Proposition 3.1's Hadamard form, and
`diag.py`'s as-implemented diagonal. One thing already falls out analytically and is now asserted:
**Prop. 3.1 is exact on the `β` block**, since `∂/∂β = Σ_t g_t` makes that block `S` itself. So HF4
is entirely a statement about `γ` and about the cross terms — which narrows what the A1 run has to
measure.

### 5.5 Step 5: the metrics, T7, and two numerical facts worth carrying

`metrics/` implements §0.5's five: M1 (`frobenius`), M3 (`stein`), M5 (`ngd`), M7 (`kron_diag`),
M8 (`coupling`), plus §3.4's noise floor as an **interval** rather than lot 1's single split.
**T7 passes**: `D_λ(R‖R) = 0` to `1e-8` — the half that catches a sign or a transposition, since
every term is individually large and they must cancel — and the Cholesky form agrees with the dense
expression.

Two things the tests turned up that are properties of the *quantities*, not of the code, and that
change how results must be read:

1. **`e_F*` is floored at `√ε ≈ 1.5e-8` in fp64.** It is `sqrt(1 − cos_F²)`, which loses half its
   significant digits as `cos_F → 1` — and `cos_F → 1` is exactly the regime §5.1 made
   load-bearing ("the right shape, the wrong size"). Writing it as
   `sqrt(‖R‖² − ⟨R,K⟩²/‖K‖²)/‖R‖` cancels identically, so the floor is intrinsic. **A reported
   `e_F*` below `1e-7` means "indistinguishable from a pure rescaling", not a measurement.**
2. **Damping breaks the scale-invariance of the direction.** `(cR + λI)^{-1} g` is
   `(R + (λ/c)I)^{-1} g`: a structure that is right *up to a scalar* still takes a different step
   once damped, and `ρ < 1` for it. Measured on the fixture: `cos = 0.9987` at `c = 7`, and exactly
   `1` at `λ = 0`. This is the concrete reason `plan_exp_draft.md` §3.3 sweeps `λ` rather than
   fixing it, and it means `c*` (M1) and `ρ` (M5) genuinely answer different questions — a structure
   can be perfect for M1 after rescaling and still lose on M5.

The noise floor now returns `(median, 95 % interval, σ_N, null_two_independent)` over 20 random
partitions, with the scaling derived in the module rather than in passing: a split of halves
measures `2σ_N`, so two independent `N`-probe references must clear `d_split/√2` before their gap
means anything. `convergence_curve` supplies the `d(F_{N'}, F_N)` against `N'` that would say
whether lot 1's sub-`N^{-1/2}` decay (§6.2 there) is a systematic floor or a heavier tail.

## 6. The A1 P1 result (pending)

*Empty until step 6 runs.*
