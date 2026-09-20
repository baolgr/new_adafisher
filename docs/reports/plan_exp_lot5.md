# Lot 5 — P2, the operational protocol: what the optimizer's own preconditioner is worth

*Implementation plan for lot 5 of `plan_exp_draft.md` §9. Continues `plan_exp_lot3.md`, whose zoo,
metrics, fold engine and P1 runner this lot reuses unchanged and extends with one new protocol.*

**Goal.** P1 asked *"how good can this family of approximations possibly be?"*. P2 asks *"how good
is the thing the optimizer actually divides by?"*. The object is `AdaFisherMulti`'s live per-module
state: after the running average, after the min-max renormalisation (`diag` only), at the
optimizer's own damping, estimated in `float32` from real training batches with normalisation
layers in train mode and dropout live. It is compared to the same exact references P1 uses, at the
**same** weights and on the **same** probe examples, so the two protocols differ in exactly one
thing: which object is being judged.

**Exit criteria** (`plan_exp_draft.md` §9, §10.2): **T12** passes (the state reader reproduces the
optimizer's own applied preconditioner, and `diag` reproduces the upstream optimizer's); the
re-warm's fidelity is **re-measured on this lot's own models**, not inherited from §3.2's single
measurement on A2; **HF7 is decided** by the rule pre-registered in §0.9 below.

**Non-goals.** No regime B (lot 4), no new benchmark model, no change to
`src/adafisher_modes/` — `fisher_ref/` stays a reader. Nothing in `benchmarks/` changes.

---

## 0. Decisions taken here, and the traps this lot must not walk into

### 0.0 Which models, and why not the two the plan names

`plan_exp_draft.md` §9 assigns lot 5 to "**A2, B1, B2**". B1 (`cct_2_3x2_cifar`, 283 723
parameters) and B2 (`resnet20_cifar`, 269 722) are **regime-B** models: their whole-model `F` is
582 GB in fp64, so every reference this campaign builds for them has to come from lot 4's factored
regime, which does not exist yet. P2 cannot run there before lot 4 does, and neither can the P1
half of the comparison HF7 is about — so a P2 number on B1 or B2 today would have nothing to be
compared against.

Lot 5 therefore runs on **every regime-A model the campaign has**, which is a superset of the plan's
regime-A entry and exactly the set on which P1 has already been measured:

| | run directory | `P` | layer types | P1 measured by |
|---|---|---:|---|---|
| **A1** | `mnist/mlp_ln_mnist` | 26 634 | 3 `Linear`, 2 `LayerNorm`; **no weight sharing** | lot 2 |
| **A2-GN** | `cifar10/cnn_gn_cifar` | 24 458 | 3 `Conv2d`, 3 `GroupNorm` (**not hooked**), a head | lot 3 |
| **A2-BN** | `cifar10/cnn_gn_cifar_bn` | 24 458 | 3 `Conv2d`, 3 `BatchNorm2d`, a head | lot 3 |
| **A3** | `cifar10/vit_micro_cifar` | 21 098 | patch-embedding `Conv2d`, 8 token-wise `Linear`, 5 `LayerNorm`, the raw `pos_embed` (**not hooked**), a head | lot 3 |

B1 and B2 stay on lot 5's list and are run as a **follow-up job once lot 4 lands**; this is recorded
in §4 as a deliberately unmet part of the plan's own wording, not as an oversight.

### 0.1 The operational operator is exactly `kron(B̃, Ã)` — verified before anything was written

Every metric in `fisher_ref/metrics/` takes a `BlockOps`. So P2's first question is: what `BlockOps`
*is* the optimizer's preconditioner? The answer is short, and it was checked numerically before a
line of lot 5 was written rather than read off the source.

For one hooked module, `AdaFisherMulti` folds the weight and bias directions into one
`(d_out, d_in_aug)` matrix `M` (`_kron_utils.augment_direction`) and returns `F̃⁻¹` applied to it.
Written in this repository's `rvec` convention (output factor outer, input factor inner —
`conventions.kron_rvec`), the operator is:

| mode | the operator `F̃` | where its state lives |
|---|---|---|
| `diag` | `Diag(f_tilde.flatten())`, `f_tilde = kron(H, S)ᵀ + λ`, shape `(d_out, d_in_aug)` | `_H`, `_S` |
| `kfac` | `kron(B̃, Ã)`, `Ã = A + π√λ I`, `B̃ = B + (√λ/π) I` | `_A`, `_B` |
| `ekfac` | `(Q_B ⊗ Q_A) diag(s* + λ) (Q_B ⊗ Q_A)ᵀ` | `_Q_A`, `_Q_B`, `_s_star` |
| `tkfac` | `δ · kron(Ψ̃, Φ̃)` | `_delta`, `_Phi_raw`, `_Psi_raw` |
| `tekfac` | `(Q_Ψ ⊗ Q_Φ) diag(Θ + λ) (Q_Ψ ⊗ Q_Φ)ᵀ` | `_Q_Phi`, `_Q_Psi`, `_Theta` |

**Measured** on a four-layer fp64 network covering all four hooked module types (`Conv2d`,
`BatchNorm2d`, `LayerNorm`, `Linear`), three real optimizer steps per mode: solving `F̃ x = rvec(M)`
densely reproduces `approx.precondition(module, M_w, M_b)` to a relative error between **0.0 and
2.2e-15** in all 20 (mode, layer) combinations. So no new block class is needed — `Kron`, `EKFAC`
and `Diag` from the existing zoo represent all five modes exactly, and the rule "compare the applied
preconditioners, never the bases" (`CLAUDE.md`) is satisfied by construction because the applied
preconditioner is what the test compares.

Two consequences that are easy to get wrong and are therefore pinned by T12:

* `d_in_aug` for a **normalisation layer is 2**, not `C`: `factors.augment_norm_input` returns a
  `(T, 2)` matrix, so `Ã` is `2 × 2` and `B̃` is `C × C`, and `M` is `(C, 2)` with column 0 the
  scale `γ` and column 1 the shift `β`.
* a `Conv2d` weight is flattened `(C_out, C_in·k_h·k_w)` before the bias column is appended, which
  is the same flattening `Conv2d.weight` itself uses and the same one the campaign's reference
  block is permuted into.

### 0.2 The layout trap, in a new place: a normalisation block is interleaved

`plan_exp_lot2.md` §5.2 records a bug whose whole signature was "the same values in a different
order", found only because T3 was written against a non-symmetric fixture. Lot 5 meets the same trap
one layer type further along, and it is the single most dangerous point in this lot.

For a `Linear` or a `Conv2d`, P1 converts the reference block from `named_parameters()` order
(`rvec(W)` then `b`) into bias-augmented `rvec([W | b])` order with
`ParamLayout.augmented_permutation`. The operational operator is **natively** in that order, so
there is nothing to do.

For a normalisation layer there is. `ParamLayout.augmented_permutation` **refuses** such a module
outright (a `(γ, β)` block is Hadamard-structured and has no `[W | b]` reading, `plan_lot5.md`
§0.1), so P1 keeps the reference block in `named_parameters()` order: all `C` entries of `γ`, then
all `C` entries of `β`. The operational operator's `rvec` order over `(C, 2)` is the **interleaved**
one, `(γ_0, β_0, γ_1, β_1, …)`. The map between them is

    named index  j·C + c   ←→   rvec index  c·2 + j          (j = 0 for γ, 1 for β)

and a P2 norm block that skips it is symmetric, positive definite and silently wrong, exactly like
lot 2's. Lot 5 therefore materialises a normalisation layer's operator densely (`2C × 2C`, at most
`128 × 128` on these four models), permutes both axes, and wraps it in `Dense`. T12 checks the
*applied* result, not the matrix: for random `γ` and `β` directions, `K.solve(cat([γ_dir, β_dir]))`
must equal `cat(optimizer.precondition(...))`.

### 0.3 The damping is already inside the operator — `metrics` gains one optional argument, not a wrapper

`metrics.rho(R, K, g, lam)` computes `d = K.solve(g, lam)` and judges it against `R + λI`. That
signature assumes `K` is an undamped structure to which the sweep's `λ` is added. P2's operator is
not: its damping is already inside it, and — for `kfac` and `tkfac` — it is **factored** Tikhonov
(`A + π√λ I` on one side, `B + √λ/π I` on the other), which is not `K + λI` for any `λ`.

Two ways to handle this, and the first is wrong:

* a wrapper class whose `solve` ignores `λ` — but then `stein_kl` would add `λI` to `K.to_dense()`
  for its trace term while `K.logdet(λ)` ignored it, i.e. the two halves of M3 would describe
  different matrices;
* **an optional `k_lam` on the two metrics that go through an inverse.** `rho(R, K, g, lam,
  k_lam=None)` and `stein_kl(R, K, lam, k_lam=None)` default to `k_lam = lam`, which is today's
  behaviour byte for byte; P2 passes `k_lam = 0.0`. The reference stays damped at the sweep's `λ`
  (it is the yardstick, and `plan_exp_lot1.md` §6.4 measured `F`'s kernel at 4 805 of A1's 26 634
  directions, so it has to be), and the operator is applied exactly as the optimizer applies it.

This is also the honest reading of `plan_exp_draft.md` §3.3's rule: *"Add AdaFisher's `λ = 10⁻³`
point, but in P2 only"*. In P2 the optimizer's own `λ` is not one point of a sweep — it is part of
the object being measured.

### 0.4 Four rungs, not two: the gap has to be attributed, not just measured

HF7's consequence in `plan_exp_draft.md` §11 is *"the weakness is the EMA / min-max / damping, not
the structure"*. A bare P1-versus-P2 comparison cannot say **which** of those, because it changes
all of them at once — and it changes two more the plan does not name: the optimizer's factor
*formulas* are not the campaign's (they use the gradient of the **batch-mean** loss, so every
example's share of the curvature is divided by `batch²` — `CLAUDE.md` §4.3; and a normalisation
layer's input factor is a `2 x 2` surrogate, `plan_lot5.md` §0.2), and the operational statistics
come from **augmented** training batches with normalisation layers in **train** mode, while the
references come from clean images in eval mode.

Lot 5 therefore measures a ladder with two middle rungs, at one extra hooked pass over the probes:

| rung | `protocol` | what it is | what the step from the rung above costs |
|---|---|---|---|
| **P1** | `P1` | the structure alone: factors from one fp64 pass over the probes, per-example type-2 or empirical vectors, no running average, no min-max, `λ` swept | — |
| **P1-py** | `P1-py` | the **optimizer's own formulas** — `compute_h_diag`/`compute_s_diag` and `compute_h_full`/`compute_s_full`, *called* not re-implemented — on the same probes, in fp64, in eval mode, with no running average, no min-max and no damping | the **estimator formula**: the batch-mean gradient's `1/batch²`, the `2 x 2` normalisation surrogate, the `Conv2d` scale quirk |
| **P2-raw** | `P2-raw` | those same formulas, but averaged with `gammas = (0.92, 0.008)` over the re-warm, min-maxed for `diag`, in fp32, from augmented training batches in train mode — **without** the optimizer's damping, judged at the same swept `λ` as P1 | the **running average, the min-max, the augmentation and train mode, fp32** |
| **P2** | `P2` | the operator the optimizer inverts, its own damping folded in | the **damping** |

Structure names: `diag_py`, `diag_py_factors` and `kron_py` for P1-py, `p2_raw_<mode>` and
`p2_<mode>` for the last two.
One CSV per checkpoint holds every rung, on the same probes at the same weights, so every comparison
is paired by construction.

`diag_py` already exists — lot 3 built it for normalisation layers only (`plan_exp_lot3.md` §0.7).
Lot 5 gives `diag_py_reading` an optional `types` argument (default: today's two normalisation
types) so P2 can ask for **every hooked layer type**, and adds `kron_py_reading` beside it, which
returns `kron(compute_s_full(s, m), compute_h_full(h, m))` per module — the un-averaged, undamped
version of exactly what `kfac` builds. Both are read through `cos_F`, `e_F_star` and `c*`, never
through a raw `e_F`, because their overall scale is arbitrary (mean-loss gradients, `1/batch²`) —
the same caveat lot 3 already attached to `diag_py`.

**Why this rung is load-bearing and not a refinement.** Without it there is no like-for-like P1
counterpart of `p2_diag` (the campaign's `af_raw` is a *different object*: the diagonal of the
fp64, per-example, type-2 Kronecker factors), and on a **normalisation layer** there is no P1
counterpart of the four Kronecker modes at all — P1 builds no `kron` there, because its own `A` for
a norm layer is the `(C+1) x (C+1)` second moment of the normalised input, which does not even have
the right size for a `2C`-parameter block. `kron_py` is the only object in this campaign that reads
a normalisation layer the way the optimizer does.

### 0.4b Two diagnostics at one checkpoint per model

Two things the three-rung ladder cannot say, each measured at **one** fraction per model (the last
one), in the spirit of `--noise-at 1`:

* **The operational operator has its own sampling noise, and the fold intervals do not contain
  it.** A fold interval carries the *reference's* noise, because the operator does not depend on the
  probes. But the operator is a draw: `plan_exp_draft.md` §3.2 measured that **92 % of it is one
  minibatch's factor**. So a second re-warm on a different batch order, `p2rep_<mode>`, gives the
  operator's own draw-to-draw spread — without it, "mode A ranks above mode B" could be a batch
  draw. This is the single most important missing error bar on the P2 side.
* **The augmentation and train-mode term.** A re-warm fed the **probe tensors themselves** in eval
  mode — clean images, no crop, no flip, no Cutout, no batch statistics — with everything else
  unchanged (`p2clean_<mode>`) separates "the data the optimizer sees" from "the running average".

Both are off at every other fraction. Each emits both rungs (`p2rep_<mode>`/`p2rep_raw_<mode>`,
`p2clean_<mode>`/`p2clean_raw_<mode>`), so they cost ten extra re-warms and twenty extra structures
on one checkpoint per model.

### 0.4c The rung the ladder would otherwise mislabel: the running average's *sample size*

With `gammas = (0.92, 0.008)` the normalised weight on the factor update `j` steps back is
`0.92 · 0.08^j`, so `Σ w_j² = 0.852` and the **effective sample size is 1.17 factor updates** —
about **150 examples** at batch 128, against 45 000-50 000 probes on the P1 side `[DERIVED]`. On
A1's `features.0` that means a `785 × 785` input factor estimated from at most ~150 samples, i.e. of
rank ≤ 150.

The plan states the input fact — "92 % of the state is one minibatch" (§0.4b) — but the consequence
has to be drawn, because without it the `P1-py → P2-raw` rung would be labelled "the running
average, the min-max, the augmentation and train mode", when its dominant content is plain
**sampling variance**. §11's consequence for HF7 ("the weakness is the EMA / min-max / damping, not
the structure") would then be mis-stated: it would be the *sample size the EMA implies*, which is a
different research direction.

So the runner emits a **sample-size control**, `kron_py_1batch`: the same P1-py formula, the same
eval-mode forward, on **one** micro-batch of `bench.batch_size` examples instead of all the probes.
The gap `kron_py → kron_py_1batch` is "45 000 → 128 examples" alone; what is left of
`kron_py_1batch → p2_raw_kfac` is the running average, the min-max, the augmentation, train mode and
fp32. It costs one more call to machinery that already micro-batches, and runs at the same fractions
as the other two diagnostics.

### 0.5 P1 has no `tekfac`, and HF7 needs five modes — so lot 5 adds it

P1's zoo has a counterpart for four of the five optimizer modes (`af_raw` ↔ `diag`, `kfac`, `ekfac`,
`tkfac`) and **none for `tekfac`**. Lot 3 did not notice; HF7 is a statement about the ranking of
the five, so the gap has to be closed.

It costs almost nothing, because TEKFAC is EKFAC in a different orthonormal basis. `folds.eigenbases`
already returns `{basis: {layer: (Q_A, Q_G)}}` and `folds.second_pass` projects the per-sample
gradients into **every** basis it is handed. So lot 5 adds one basis, `tkfac_expand`, whose
`(Q_Φ, Q_Ψ)` are the eigenvectors of TKFAC's own `Φ_raw`, `Ψ_raw` — the same matrices, since
dividing a symmetric matrix by the positive scalar `δ` leaves its eigenvectors alone. The resulting
structure is `EKFAC(QA=Q_Φ, QG=Q_Ψ, s=Θ)` with `Θ` the exact second moment of the projected
per-sample gradients.

Two properties follow and are asserted, not assumed:

* `tr(K_tekfac) = tr(B_ℓ)` — an orthogonal change of basis preserves the Frobenius norm, so this is
  T9's argument in the new basis;
* **`e_F(tekfac) ≤ e_F(tkfac)` is a theorem**, not a finding (TEKFAC Thm 3.1: the optimal diagonal
  in a fixed basis beats any other diagonal in that basis, and TKFAC's own eigenvalues `δ λ_Φ λ_Ψ`
  are one such other diagonal). It joins `p1_structural.THEOREMS`, so a violation is reported as a
  bug.

The new basis is **off by default** (`eigenbases(..., tkfac_basis=False)`), so lot 3's code path and
its published numbers are untouched.

### 0.6 The re-warm: frozen weights, a separate model, and a refusal below 10·TCov

The checkpoints hold weights only, so P2 has to rebuild the optimizer's memory by running it
(`plan_exp_draft.md` §3.2). Four decisions, each with its reason.

1. **The weights are frozen during the re-warm (`lr = 0`).** §3.2 measured the staleness term — the
   effect of the weights moving while the memory warms — as *the same order as the estimator's own
   noise*, i.e. invisible. Freezing them is therefore free, and it buys something real: every factor
   is then estimated at exactly the checkpoint's `θ`, which is the `θ` the reference is built at.
   The "snapshot the state and restore `θ`" caveat §3.2 attaches to every P2 number disappears.
   `--rewarm-lr` exposes the moving variant; the fidelity experiment (§0.8) measures both.
2. **The re-warm runs on its own copy of the network, and the reference on a second copy loaded
   fresh from the checkpoint.** With normalisation layers in train mode — which P2 requires — a
   `BatchNorm2d`'s running mean and variance are rewritten on every forward, so after 1 000 steps
   the buffers of a re-warmed A2-BN would have nothing to do with the checkpoint's. Restoring them
   afterwards would work; not sharing the object cannot be got wrong. The parameters are asserted
   bit-identical after the re-warm, as the direct check that `lr = 0` did what it says.
3. **At least `10·TCov` steps, and the runner refuses fewer.** This is §3.2's corrected arithmetic:
   the length is set by `0.08^k ≪ λ`, not by `0.08^k ≪ 1`, because every mode seeds its running
   average with the identity and a short re-warm leaves `0.08^k · I` behind as a *spurious extra
   damping*. At `k = 3` that residue is `5.1e-4`, half of `λ = 10⁻³`, and the applied preconditioner
   was measured 12-87 % wrong. `--rewarm-steps` below `10 · TCov` raises, with that sentence in the
   message, unless `--allow-short-rewarm` is passed.
3b. **The length rule binds on the *damped* operator, and P2-raw is the undamped one.**
   `0.08^k ≪ λ` is the condition for `F̃`, where `λ` dominates anyway. **P2-raw has `λ` removed**
   and is judged at the sweep's `λ`, whose smallest point is `α = 10⁻⁴ · tr(R)/P`; the binding
   comparison for the *state* is `0.08^k` against the accumulated factor's own smallest eigenvalue,
   not against `λ`. `adafisher_state.py` already records `min_s_star` and `min_theta` per block, so
   the ratio is reported in `meta.json` per layer rather than assumed, and §0.8's confirmation rule
   is applied to the undamped operator as well as to the applied one. `k = 10` is the *damped*
   rule's answer; if the undamped ratio says otherwise on some model, that model's P2-raw rows carry
   the caveat.

4. **The re-warm runs in `float32`, the training dtype, not in the campaign's fp64.** P2 is defined
   as the object the optimizer holds; that object is fp32. The snapshot is cast to fp64 once, at the
   boundary, for the metrics. One declared deviation comes with it: `conventions.configure()` turns
   TF32 off, which the training runs did not do, so the re-warm's convolutions are computed more
   accurately than the real run's by a relative `~5e-4` `[ESTIMATE, from TF32's 10 explicit mantissa
   bits; not measured]`. That is three orders of magnitude below the
   estimator's own spread, whose dominant term is that **92 % of the state is one minibatch's
   factor** (§3.2), and it is recorded in the metadata rather than corrected.

### 0.7 What the optimizer does not precondition is a result, not a gap in the table

`AdaFisherMulti` hooks `Linear`, `Conv2d`, `BatchNorm2d` and `LayerNorm`. Everything else takes the
fallback branch of `step()`: plain momentum SGD, i.e. `F̃ = I` exactly. On this lot's four models
that is `cnn_gn_cifar`'s three `GroupNorm` layers (224 of 24 458 parameters) and
`vit_micro_cifar`'s `pos_embed` (2 048 of 21 098, i.e. **9.7 %**).

Those blocks get one structure, `p2_identity` (`Diag(ones)`), not five identical copies of it, and
they are **excluded from the ranking cells** of §0.9 — there is no ranking of five modes where all
five are the identity. They are reported separately, with two numbers per model that are worth
having on their own: the share of parameters and the share of `tr(F)` that the optimizer leaves
unpreconditioned.

Note `ρ` is invariant under a positive rescaling of the direction, so `ρ(p2_identity)` equals
`ρ(identity)`, P1's own Frobenius-optimal multiple of the identity — the plain-gradient step. Its
value is the natural zero of the whole comparison: a mode whose `ρ` is below it is worse than not
preconditioning at all.

### 0.8 The re-warm fidelity has been measured on one model; this lot re-measures it on four

`plan_exp_draft.md` §3.2's table is A2 only, and it is an exit criterion of this lot that it be
re-measured "on the lot's own models". `fisher_ref/experiments/rewarm_fidelity_lot5.py` does it in a
shape that starts from a **real checkpoint** instead of a 2 000-step training run:

from the checkpoint's `θ*`, with the weights frozen, one long warm of `30 · TCov` steps on batch
order A gives the converged state `S_ref`; two re-warms of `k · TCov` steps on batch orders B and C
give `S_B` and `S_C`. Then `‖S_B − S_C‖` is the estimator's **own noise floor** (two honest draws of
the same quantity) and `‖S_ref − S_B‖` is the re-warm gap. A fourth run with the weights moving
gives the staleness term. Everything is reported on the **applied preconditioner** — `F̃⁻¹ m̂` on a
fixed random direction, the object P2 actually compares to `F` — as well as on the raw state, and
swept over `k ∈ {3, 10, 20}`.

The pre-registered reading, so it cannot be fitted afterwards: `k = 10` is **confirmed sufficient**
on a model if, for all five modes, the re-warm gap on the applied preconditioner is within `5×` its
own noise floor and does not improve by more than `2×` going from `k = 10` to `k = 20`. If it is not
confirmed on some model, that model's P2 numbers are re-run at the length that is, and the first
ones are reported as superseded.

**What this experiment can and cannot establish, stated plainly.** There is no fixed state to
converge *to*: 92 % of the operational state is the last minibatch's factor (§3.2). So
`‖S_ref(k=30) − S_B(k=10)‖` and the floor `‖S_B − S_C‖` are two measurements of the *same*
batch-draw noise, which is why `plan_exp_draft.md` §3.2's own table already shows `ekfac`'s "gap"
(0.15) sitting *below* its "floor" (0.27). This is therefore a sound check that **the identity seed
has decayed** and that the estimator has reached its own noise level — it is not a measurement of
fidelity to a state the run once had, and no such state is recoverable. The lot's exit criterion is
worded as "the re-warm's fidelity re-measured"; what is actually delivered is this, and §6 says so.

### 0.9 Pre-registered decision rules for HF7

Written before any lot-5 number exists, in the spirit of `plan_exp_lot3.md` §0.10.
`fisher_ref/experiments/lot5_decisions.py` applies them and chooses no threshold of its own —
**every rule below is in that file**, because a pre-registration the runner ignores is not one.

**HF7 as stated** (`plan_exp_draft_v0.md` §72): *the ranking of the approximations under real
operating conditions is **not** the ranking under idealised ones.* The hypothesis predicts
**disagreement**; "refuted" means the rankings agree.

#### The trap this rule is built around

Both `e_F_star` and `ρ` are invariant under `K → cK` — `ρ` because it is invariant under `d → cd`,
and the P2 operator is applied with `k_lam = 0`. And §5.4 **measured** `cond(F̃)` between **1.00 and
1.11**. If the operational preconditioner is proportional to the identity, all five P2 values
collapse onto one and their order is a batch draw. A rank correlation against noise is ≈ 0 — which
a naive rule reads as "the rankings differ".

Simulated over 20 000 draws of a random ranking of five, the median `τ` over 15 cells is `≤ 0.2` in
**98.7 %** of them. So the naive rule would announce **HF7 confirmed with no evidence at all**, and
the informative claim would be the one that needs none. Two gates stand in front of the verdict, and
both use data the runner already produces (`p2rep_*`, a second re-warm on an independent batch
order):

* **the degeneracy gate.** In a cell, let `spread` be the range of the modes' `e_F_star` and `noise`
  the median over modes of `|e_F_star(p2rep_m) − e_F_star(p2_m)|` — the operator's own draw-to-draw
  spread. If `spread ≤ noise` there is no ranking to speak of: the cell is **degenerate**, is
  counted, and never votes. A model with more than half its cells degenerate is reported
  **"HF7 not decidable — the operational preconditioner has no measurable preferred direction"**,
  which is a finding in its own right and is *not* HF7 confirmed.
* **the reproducibility ceiling.** `τ(p2, p2rep)` bounds what `τ(P1, P2)` can mean. A P2 ranking
  that does not reproduce across two draws of the same estimator cannot be said to differ from
  anything.

The replica runs at one fraction per model, so `noise` is measured there and **transferred** to the
other fractions of the same `(model, arm, seed, layer)`. That is a declared approximation, not a
silent one.

**Measured on the corrected rule** (the same simulation, 2 000 draws per scenario, and pinned by
`tests/test_fisher_ref_lot5.py`):

| the P2 side is… | naive rule: CONFIRMED | corrected rule: CONFIRMED |
|---|---:|---:|
| proportional to the identity (what §5.4 measured) | **98.7 %** | **0.1 %** (86 % "not decidable") |
| a real, reproducible ranking that differs from P1 | — | **85 %** |
| a real ranking that agrees with P1 | — | 0 % (**100 % refuted**) |

#### The pairs, and the two classes of cell

The P1 side of a pair must be the same *object* as the P2 side, minus the operational compromises:

| verdict | pairs | layer kinds | rung |
|---|---|---|---|
| **primary** | `kfac`, `ekfac`, `tkfac`, `tekfac` | `linear` / `conv` | **P1** — fp64, per-example, like-for-like |
| five-mode variant | the above **+ `diag_py_factors` ↔ `p2_diag`** | `linear` / `conv` | mixed: `diag`'s partner sits at P1-py |
| secondary | `diag_py_factors ↔ p2_diag`, `kron_py ↔ p2_kfac` | `norm` | P1-py |

The primary's `linear`/`conv` blocks are **99.52 %** of A1's parameters, **99.08 %** of A2's and
**88.78 %** of A3's `[DERIVED, recomputed from the built models]`. A3's remainder is its five
`LayerNorm` (320 parameters) *and* `pos_embed` (2 048); an earlier draft of this section said
"90.3 %", which is everything except `pos_embed` — it silently counted the LayerNorms it also
excluded in the same sentence.

`diag` is **not** in the primary. Its P1-rung structure `af_raw` is a *different estimator* — the
diagonal of the campaign's own fp64 type-2 factors, where `diag.py` uses the gradient of the
batch-mean loss — which is exactly the confusion §0.14 item 1 records. Ranking it there would put
one of five items on the wrong object, and moving one item two places changes `τ` by up to 0.4 on a
scale whose thresholds are 0.2 and 0.6. It is ranked in the five-mode variant instead, against the
P1-py reading of its own formula, and both readings are reported.

On a `norm` layer only two pairs exist, and `τ_b` over two items is exactly `±1`, so a median of it
partitions at 50 % — a coin flip dressed as a rank statistic. The secondary verdict is therefore a
**sign-agreement rate with a binomial interval**, not a `τ`.

#### The reference, and the single statistic

* **`Ê`, not `F`, is the reference the verdict is read against.** The operational state, `kron_py`
  and `diag_py_*` are all built from the **empirical, true-label, batch-mean** gradient. Against `F`
  the `P1 → P1-py` step would carry the whole type-2 → empirical *source* change on top of the
  formula change it is meant to isolate — and lot 1 measured that source gap at `‖Ê − F‖_F/‖F‖_F =
  1.42` on A1's head and LayerNorm blocks, which is not a correction term. Type-2 is reported as a
  declared secondary. Both sources therefore carry fold intervals (`--fold-sources type2 empirical`).
* **`e_F_star` is the single voting statistic.** `ρ` is reported across the whole damping sweep and
  does **not** vote: the P2 operator has exactly one damping by construction (its own), so a `ρ`
  verdict would be a conclusion about an inverse at a single `λ`, which `plan_exp_draft.md` §10.3
  forbids. Making it vote would also give the hypothesis six shots and the null none, since the five
  `α` are not five independent tests — only the *reference's* damping moves between them.

#### The verdict

* **A cell** is `(model, arm, seed, layer, checkpoint, source)`, restricted to layers the optimizer
  hooks (§0.7). A cell with any missing or `NaN` value is dropped **and counted**; an all-tied P2
  side makes `τ_b` itself `NaN`, which is dropped rather than fed to a median (`statistics.median`
  sorts, so one `NaN` silently returns the wrong answer).
* **Per model**, on the admissible cells: *HF7 confirmed* if the median `τ(P1, P2) ≤ 0.2` **and** the
  median reproducibility ceiling `τ(p2, p2rep) ≥ 0.6`; *HF7 refuted* if the median `τ ≥ 0.6`;
  *mixed* otherwise; *not decidable* if more than half the cells are degenerate, or if the ranking
  differs but does not reproduce.
* **One limitation of the θ axis, declared rather than discovered later.** All four jobs read
  `--arm diag --seed 0`, so **all five modes' operators are read at weights the `diag` optimizer
  produced** — an off-trajectory state for the four Kronecker modes, which is not quite "real
  operating conditions" as HF7 words it. It is the right choice for a *paired* P1/P2 comparison (P1's
  own θ axis is `diag`'s), and the alternative confounds the comparison with a change of θ. The
  `adamw` arm's checkpoints exist and give a cheap, optimizer-neutral second θ — that is
  `plan_exp_draft.md` §3.5's own open question ("does the drift at a given θ depend on which
  optimizer produced it?"), and it is one more job per model, not a change to any code.
* **Decided** when **A2-GN and A3 agree** — A2-BN is a *robustness check*, not a third model: it is
  the same `cnn_gn_cifar` with `--norm bn`, same data, same hyperparameters, so "two models agree"
  could otherwise be met by one architecture twice. This restores `plan_exp_lot3.md` §0.10's own
  wording, which the first draft of this section dropped.
* **`plan_exp_draft.md` §10.3's regime-B clause cannot be met by this lot**: "any per-layer-type
  conclusion must hold on at least one regime-A **and** one regime-B model", and §0.0 defers B1/B2
  to lot 4. Every per-layer-type reading here is therefore **provisional**, and §4 records that as
  an exit criterion this lot does not meet rather than as one it forgot.

#### The magnitude, and the ladder

* **Magnitude** (§11's "P2 ≫ P1"): per cell, `Δ = median over pairs of e_F_star(P2) − e_F_star(P1)`.
  The operational compromises are said to **dominate the choice of mode** when the median `Δ`
  exceeds **both** that layer's own noise floor (`layer_noise_floor`, the reference's split-half
  distance) **and** the spread `max − min` of the P1 structures' `e_F_star` in the same cell. Both
  clauses, not one: the second is the falsifiable form of "it is the smoothing and the
  renormalisation, not the structure", and the first is §10.3's reading rule.
* **The ladder** (§0.4), on the two pairs that have all four rungs (`diag`, `kfac`): the three gaps
  as **shares of the sum of their absolute values**. The dominant rung is named only when every gap
  is non-negative *and* holds at least two thirds of the total — a later rung can fit better after
  optimal rescaling, and an argmax over mixed signs would name a winner among quantities that do not
  add up to anything. Cells with a negative gap are counted and reported.
* **Paired differences, not overlapping intervals.** All the P2 structures are judged against the
  *same* reference halves, so their fold intervals are strongly positively correlated and comparing
  two marginal intervals is over-conservative. The interval on the **difference** is the right test,
  and it is the machinery lot 3 already built for HF2 (`p1_structural.PAIRS`), extended here with
  the five `(P2, P1)` pairs and the five `(p2rep, p2)` ones.

### 0.10 One traversal, several consumers — again, and why `p1_structural.py` is extended rather than copied

Lot 2 §0.2 argued for one traversal of the probes and implemented three; lot 3 finally made it one.
P2 needs the same references, on the same probes, at the same `θ`, so copying `run_source` into a
second runner would reintroduce exactly the divergence those two lots spent effort removing — and
worse, it would make "the P1 number" and "the P2 number" come from two builds nobody compared.

So `p1_structural.py` gains **one optional argument**, threaded through `run_fraction` →
`run_source` → `analyse_block` / `half_values` → `structures_of`:

```python
@dataclass(frozen=True)
class ExtraStructures:
    blocks: Mapping[str, Mapping[str, BlockOps]]   # layer -> {structure -> operator}
    self_damped: FrozenSet[str]                    # names whose damping is already inside them
    protocol: Mapping[str, str]                    # row label, "P2" / "P2-raw"
```

`None` (the default) is byte-for-byte today's behaviour, and the 609 pre-existing tests are the
check. `p2_operational.py` is then a thin command-line front end: it builds the snapshots and calls
the same `run_fraction`.

Two properties fall out for free and are worth naming, because each would otherwise be a separate
piece of work:

* the **fold intervals cover P2 too**. The operational operator does not depend on the probes, so
  evaluating the *same fixed* operator against each half's reference gives an interval that carries
  the reference's sampling noise alone — which is exactly the right interval for "is this gap real?",
  and the same declared approximation lot 3 §0.8 already makes for EKFAC's basis.
* the cross-checks P1 already runs — the per-sample row checks against autograd, the theorem
  orderings — run on the P2 job too, unchanged.

### 0.11 A trap this lot inherits and must not spring: `conv_sua`

Under `conv_sua=True` the operational operator is **not** one `kron(B̃, Ã)` over the full
`(C_out, C_in·k_h·k_w + 1)` direction: it is the same small `(C_in + 1)`-wide operator applied
independently at each of the `k_h·k_w` kernel offsets, with the bias read back from the centre
offset only (`plan_lot6.md` §0.4). Representing that as a single `Kron` would be wrong, and wrong in
the silent way — right shape, wrong matrix.

None of this lot's four models uses it (all four have `conv_sua: false` in their manifests; the
flag exists for ResNet-50-scale layers). The state reader therefore **raises** `NotImplementedError`
on `conv_sua=True`, naming the reason, rather than producing a plausible wrong answer. A test pins
the refusal.

### 0.12 What must not change

* The **628** pre-existing tests pass **unmodified** (the count at the time lot 5 started; lot 3
  closed at 609 and the suite has grown since with work outside both lots).
* `p1_structural.py` with no `ExtraStructures` and `eigenbases` with `tkfac_basis=False` reproduce
  lot 3's behaviour exactly; lot 3's published `metrics.csv` files are not touched or regenerated.
* `METRICS_VERSION` becomes `fisher_ref/0.3`: the zoo gained a structure (`tekfac`) and the schema
  gained two protocol values, which is what the invariant exists to record.
* Results are written under `fisher_ref/outputs/p2/…`, a separate root, so nothing overwrites lot 2's
  or lot 3's.

---

### 0.13 M3 is not computed for the new structures, and the skip is written down

`analyse_block` computes M3 (the Stein / KL gap) for every structure under `--stein-max-p`. Each
one costs a dense `K` plus a Cholesky solve with a `P x P` right-hand side — `2 P³` flops. Lot 5
roughly triples the structure count on a `linear` or `conv` block (8 becomes 24 on a shared layer),
which would triple the job, and M3 is **not** in §0.9's verdict: `plan_exp_draft.md` §0.11 already
ranks it below M1, M5, M7 and M8, and HF7 is decided on `e_F_star` and `ρ`.

So M3 runs on the P1 rung's structures exactly as it does today, and the new ones get a recorded
`stein_kl_skipped_protocol` row instead of a silent absence — `--stein-extras-max-p` (default `0`,
i.e. skip) turns it on for small blocks if it is ever wanted. M1 and M5 run on **everything**: M1 is
a contraction against the rearrangement the block already holds, and M5's only expensive part (one
Cholesky of `R_λ` per `λ`) is shared by every structure.

### 0.14 What the naive re-read of this plan changed, before any code was written

The first draft of this file was re-read adversarially against the code it proposes to touch. Six
things were wrong or missing, and each changed the design rather than a sentence:

1. **There was no like-for-like P1 counterpart of `p2_diag`.** The draft paired it with `af_raw`,
   which is a different object — the diagonal of the campaign's fp64, per-example, type-2 Kronecker
   factors, where `diag.py` uses the gradient of the batch-mean loss. Fixed by the **P1-py rung**
   (§0.4), which calls the optimizer's own formulas on the probes.
2. **P1 has no Kronecker structure at all on a normalisation layer**, so three of the five pairs
   would have had nothing to compare against there: the campaign's own `A` for a norm layer is the
   `(C+1) x (C+1)` second moment of the *normalised* input, which does not have the size of a `2C`
   block. `kron_py` (§0.4) is the fix; §0.9 now splits the verdict into a primary one on
   `linear`/`conv` and a declared secondary one on normalisation layers.
3. **The P2 operator's own sampling noise was nowhere.** Fold intervals carry the reference's noise;
   the operator is a draw of a quantity that is 92 % one minibatch. `p2rep_<mode>` (§0.4b) measures
   it, and §0.9 now forbids reporting a ranking difference smaller than it.
4. **The augmentation and train-mode term was silently folded into "the EMA".** `p2clean_<mode>`
   (§0.4b) separates it.
5. **M3 would have tripled the job for a metric that does not enter the verdict** (§0.13).
6. **The plan had no sizing table**, which is the one thing `fisher_ref/slurm/README.md` makes a
   rule — "every job justifies its own memory against the `P` of the model it runs on" (§3.5).

### 0.15 The two failure modes A1 is the test case for

A1 is in this lot partly because it is the model on which two known defects would bite:

* its first `Linear` has an input factor of shape `785 x 785` and **rank 646**, with 136 exactly-zero
  eigenvalues, because 130 of MNIST's 784 pixels are identically zero. That is the matrix on which
  cuSOLVER's `eigh` raised `_LinAlgError` and killed two campaign-1 training runs; the re-warm of
  `ekfac` and `tekfac` on A1 runs straight into it, and `approximations/_eigh_utils.py::eigenbasis`
  is the fix that has to hold;
* the same block is `25 120` of `26 634` parameters and carries **63 % of `tr(F)`**, so a job that
  skips M5 on it (which lot 2's first A1 job did) answers the wrong question. `--rho-max-p 30000`
  keeps it in, at one Cholesky of `25 120²` per `λ` — **measured at 30 s** in
  `p1_features0_a1.sh`'s own sizing, i.e. 300 s per checkpoint over five `λ` and two sources.


## 1. Modules

| file | change |
|---|---|
| `fisher_ref/approx/adafisher_state.py` *(new)* | `snapshot(model, optimizer)` → `{module_name: OperationalBlocks}`: the five modes read into `Kron` / `EKFAC` / `Diag` / `Dense` (§0.1), in the layout P1 uses for that block kind (§0.2), in fp64 on the host; the damped operator and the undamped one (§0.4); `conv_sua` refusal (§0.11); `norm_permutation(C)` |
| `fisher_ref/rewarm.py` *(new)* | `rewarm(bench, theta, mode, hp, steps, ...)`: a fresh model + `AdaFisherMulti`, the arm's own loader and hyperparameters from the run's manifest, `lr = 0` by default, `model.train()`, the exact step order `common/loop.py` uses; refuses `steps < 10·TCov`; asserts the parameters unchanged when frozen |
| `fisher_ref/metrics/ngd.py` | `rho(..., k_lam=None)` (§0.3) |
| `fisher_ref/metrics/stein.py` | `stein_kl(..., k_lam=None)` (§0.3) |
| `fisher_ref/folds.py` | `eigenbases(..., tkfac_basis=False)` → the `tkfac_expand` basis (§0.5) |
| `fisher_ref/approx/factors.py` | `LayerFactors.tkfac_eigenbases()` — the eigenvectors of `Φ_raw`, `Ψ_raw` |
| `fisher_ref/approx/norm_layers.py` | `diag_py_reading(..., types=...)` (default: today's two normalisation types); `kron_py_reading(...)` beside it — the P1-py rung (§0.4), both **calling** `adafisher_modes.factors` |
| `fisher_ref/runners/p1_structural.py` | `ExtraStructures`; the optional argument threaded through `run_fraction`/`run_source`/`analyse_block`/`half_values`/`structures_of`; `tekfac` in the zoo and in `THEOREMS`; per-structure `protocol` on the rows |
| `fisher_ref/runners/p2_operational.py` *(new)* | the CLI: re-warm the five modes at each checkpoint, snapshot, call `run_fraction` with the extras, write `metrics.csv` + `meta.json` under `fisher_ref/outputs/p2/` |
| `fisher_ref/experiments/rewarm_fidelity_lot5.py` *(new)* | §0.8's four-model re-measurement |
| `fisher_ref/experiments/lot5_decisions.py` *(new)* | §0.9's pre-registered rules applied to the CSVs |
| `fisher_ref/conventions.py` | `METRICS_VERSION = "fisher_ref/0.3"` |
| `fisher_ref/slurm/p2_operational_{a1,a2_gn,a2_bn,a3}.sh`, `p2_smoke.sh`, `rewarm_fidelity_lot5.sh` *(new)* | hand-written, each stating its own memory against its `P` (`fisher_ref/slurm/README.md`) |

## 2. Tests — `tests/test_fisher_ref_lot5.py`

| test | statement | tolerance |
|---|---|---|
| **T12-applied** | for all five modes and all four hooked layer types, `K.solve(v, 0)` equals `approx.precondition` applied to the same direction, in the layout P1 uses for that block kind | ≤ `1e-10` (fp64) |
| **T12-upstream** | `diag`'s snapshot reproduces `FisherAdapTune`'s own `_get_F_tilde`, same batches and seed, reordered into the campaign's layout | ≤ `1e-7` |
| **T12-norm-layout** | the interleaved→blocked permutation: a `γ`-only direction must come back as a `γ`-only result, and the test **fails** when the permutation is removed | exact |
| re-warm freezes `θ` | after `k·TCov` steps at `lr = 0`, every parameter is bit-identical to the checkpoint's; the `BatchNorm2d` buffers are **not** (which is why the reference uses a second model) | `torch.equal` |
| re-warm refusal | `steps < 10·TCov` raises, naming `0.08^k ≪ λ`; `--allow-short-rewarm` lets it through | — |
| `conv_sua` refusal | `snapshot` raises `NotImplementedError` on a `conv_sua=True` optimizer with a `Conv2d` | — |
| coverage | every module the optimizer hooks appears in the snapshot; every capturable module it does not hook gets `p2_identity`; a raw parameter gets `p2_identity` | — |
| P2-raw vs P2 | at `λ → 0` the two coincide for `ekfac`/`tekfac`/`diag` (additive damping) and **do not** for `kfac`/`tkfac` (factored Tikhonov) — the negative half is what makes the test able to catch a mis-wiring | ≤ `1e-8` / gap > `1e-6` |
| `k_lam` inertness | `rho`/`stein_kl` with `k_lam=None` are bit-identical to the pre-lot-5 functions on a random fixture | `torch.equal` |
| **T9-tekfac** | the new basis preserves the trace: `tr(K_tekfac) = tr(B_ℓ)` on a shared and an unshared layer | ≤ `1e-10` |
| tekfac dominance | `e_F(tekfac) ≤ e_F(tkfac)` on real blocks (TEKFAC Thm 3.1, §0.5) | exact + `1e-12` |
| `eigenbases` default | `tkfac_basis=False` returns exactly lot 3's keys, and the lot-3 suite passes unmodified | — |
| `ExtraStructures` inertness | `structures_of`/`analyse_block`/`half_values` with `extra=None` return exactly what they returned before | `torch.equal` on the row values |
| `diag_py` inertness | `diag_py_reading` with the default `types` returns exactly lot 3's result, on `BatchNorm2d` and a 3-D `LayerNorm` | `torch.equal` |
| `kron_py` layout | `kron_py_reading`'s block equals `kron(compute_s_full, compute_h_full)` reordered into the reference's layout, on all four hooked types; and on a `Linear` it agrees with the campaign's own `kfac` built from the *same* empirical micro-batch statistics, up to the declared `1/batch²` scale | ≤ `1e-10` after `c*` |
| M3 skip is recorded | a `stein_kl_skipped_protocol` row exists for every skipped extra structure, so the absence is in the data | — |
| runner | on a tiny model: the schema, `protocol` values `P1`/`P1-py`/`P2-raw`/`P2` all present, `ci_low`/`ci_high` filled for P2 rows when folds are on, `p2_identity` on an unhooked block, `p2rep_*`/`p2clean_*` only at the requested fraction | — |

## 3. Order of work

1. §0.1's layout probe, as a test, on all five modes and all four layer types — written first and
   seen to pass, since it is the premise everything else rests on.
2. `adafisher_state.py` + `rewarm.py`; T12 and the re-warm tests green.
3. `k_lam` on `rho`/`stein_kl`; the inertness tests green; the pre-existing suite unmodified.
4. `tekfac` in the zoo (`folds.eigenbases`, `factors`, `structures_of`, `THEOREMS`); T9-tekfac and
   the dominance test green.
5. `ExtraStructures` threaded through `p1_structural.py`; inertness tests green; the full
   pre-existing suite green, unmodified.
6. `p2_operational.py`; a local smoke on the real A2-GN and A3 checkpoints at a small probe count
   and a short re-warm.
7. `ruff`, `mypy`; **a naive agentic audit of the whole implementation** before anything is
   submitted; cluster sync; the smoke job; then the four P2 jobs and the fidelity job.
8. §5 during the work; §6 when the jobs have written their CSVs.

## 3.5 Sizing, and the jobs

Every figure below is either **measured** on lot 3's own runs or derived from them; none is a fresh
guess. Lot 3 measured, per checkpoint: A2 ~60 min (job 21276894, `04:30:41` for five checkpoints),
A3 ~12 min (job 21276896, `01:02:33`). Lot 5 adds, per checkpoint: five re-warms of `10·TCov = 1 000`
steps each (**~5 min**, derived from `cnn_gn_cifar/diag`'s own measured 10 530 steps in 46.8 s, times
a factor 5 for the slower modes and the MIG slice); one extra hooked pass over the probes for the
P1-py rung (**~1 min**, the same shape as lot 3's `diag_py` pass); three times the structures in M1
and in the fold intervals, whose 24 s per partition becomes ~60 s (**+12 min per checkpoint at 20
partitions**); and M5 for 24 structures instead of 8, which is a few vector solves on top of a
Cholesky that is already paid. M3 on the new structures is skipped (§0.13).

| job | run directory | `P` | probes | batch | folds | partitions | `--mem` | `--time` | per checkpoint |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `p2_smoke.sh` | all four, one checkpoint each | — | 2 000 | 250 | **4** | 3 | 64G | `03:00:00` | exercises every device path; its metric values are **not** results |
| `p2_operational_a3.sh` | `cifar10/vit_micro_cifar` | 21 098 | 45 000 | 250 | 10 | 20 | 96G | `09:00:00` | ~45 min |
| `p2_operational_a2_gn.sh` | `cifar10/cnn_gn_cifar` | 24 458 | 45 000 | 250 | 10 | 20 | 128G | `16:00:00` | ~100 min |
| `p2_operational_a2_bn.sh` | `cifar10/cnn_gn_cifar_bn` | 24 458 | 45 000 | 250 | 10 | 20 | 128G | `16:00:00` | ~100 min |
| `p2_operational_a1.sh` | `mnist/mlp_ln_mnist` | 26 634 | 50 000 | 250 | 8 | 20 | 128G | `12:00:00` | ~55 min `[ESTIMATE]` |
| `rewarm_fidelity_lot5.sh` | all four | — | — | — | — | — | 32G | `04:00:00` | §0.8's re-measurement |

Three constraints fixed those numbers rather than taste:

* **`FoldPlan` refuses unequal folds**, so `probes` must divide `folds × batch`. A2 and A3 keep lot
  3's own `45 000 / 250 / 10`, **which is deliberate**: the recomputed P1 rows can then be checked
  against lot 3's published ones, which is a free regression test on the whole reuse (§4, criterion
  8). A1 cannot keep lot 2's 55 000 with eight folds (`55 000 % 2 000 ≠ 0`), so it uses 50 000; its
  P1 rows are therefore **not** comparable to lot 2's, and that is stated rather than assumed.
* **The host memory is set by `P` and the fold count, not by the probes** (`fisher_ref/slurm/
  README.md`). Ten folds are 47.9 GB for A2 and 35.6 GB for A3; A1's eight are 45.4 GB, and its
  `features.0` adds two transient 5.05 GB buffers (the block and its rearrangement) per half.
* **The device holds one `P × P` fp64 accumulator** (5.68 GB at A1's `P`), which is why every job
  asks for the same `h100_1g.10gb` slice the training jobs use, and nothing larger.

Fractions are ordered `1,0,0.5,0.1,0.01` so a timeout leaves the two endpoints written, and every
source is written to disk as soon as it is done — lot 1's own lesson, where a job produced every
number and saved none.


## 4. Exit criteria

1. The pre-existing tests pass, **unmodified**. *(Met: **674 passing**, 41 skipped — the 628 that
   existed plus lot 5's **46** (`test_fisher_ref_lot5.py`, parametrised over the five modes); no
   pre-existing test file touched.)*
2. T12 passes, in all three of its forms; every §2 test passes. *(Met.)*
3. `ruff` and `mypy` clean on everything lot 5 touches. *(Met; the remaining `ruff`/`mypy` findings
   are in `experiments/` drivers this lot does not modify, as lot 3 also recorded.)*
4. `metrics.csv` + `meta.json` written for A1, A2-GN, A2-BN and A3, at five checkpoints, with all
   four protocol rungs and fold intervals. *(Met: 181 574 rows, 0 invariant violations, 0 non-finite
   values, row checks `≤ 1.04e-15` — §6.1.)*
5. The re-warm's fidelity re-measured on all four models by §0.8's rule, and `k = 10` either
   confirmed or replaced by a measured length. *(Met: **confirmed on all four**, gap/floor 0.37-1.82
   at `k = 10` — §6.2.)*
6. HF7 decided by §0.9's pre-registered rule; the ladder attribution reported. *(Met, and the rule's
   own verdict is reported **with the campaign's §10.3 floor applied on top**, which changes the
   reading — §6.3. The ladder is §6.5.)*
7. `CLAUDE.md` and `plan_exp_draft.md` §9 updated. **B1 and B2 are recorded as deferred to lot 4**,
   with §0.0's reason, rather than quietly dropped.
8. On A2-GN, A2-BN and A3 — which re-use lot 3's own `45 000 / 250 / 10` probe configuration — the
   **recomputed P1 rows agree with lot 3's published ones**. This is a free regression test on the
   whole reuse: if injecting the extra structures had perturbed the P1 path, or if the `tekfac`
   basis had changed the other two bases' sums, it would show here. Tolerance `1e-12` relative,
   loosened only for values whose own magnitude is below `1e-10`. *(Met on the type-2 source: worst
   `3.47e-13` over 33 065 rows, zero above `1e-10`. The empirical source has one documented
   exception, `ekfac_reduce`'s inverse metrics, traced to an exact rank deficiency — §6.1, §6.6.)*
9. The naive agentic audit of §3 step 7 is run and its findings are either fixed or recorded as
   accepted, **before** any job is submitted. *(Met — see §5.7.)*

**Two criteria this lot does not meet, recorded rather than quietly dropped.**

* `plan_exp_draft.md` §10.3: *"any per-layer-type conclusion must hold on at least one regime-A
  **and** one regime-B model"*. §0.0 defers B1/B2 to lot 4, so **every per-layer-type reading in
  lot 5 is provisional** until it is replayed there.
* The lot's own wording "the re-warm's **fidelity** re-measured" (§4.5) is stronger than what §0.8
  can deliver: with 92 % of the state being the last minibatch there is no fixed state to be
  faithful to, and what the experiment establishes is that the identity seed has decayed and the
  estimator has reached its own noise level. §0.8 now says so.

## 5. Findings during implementation

### 5.1 The premise held, measured before anything was built

§0.1's table was checked numerically first, on a four-layer fp64 network covering all four hooked
module types, three real optimizer steps per mode: solving `F̃ x = rvec(M)` densely reproduces
`approx.precondition(module, M_w, M_b)` to a relative error between **0.0 and 2.2e-15** in all
twenty (mode, layer) combinations. So no new block class was needed — `Kron`, `EKFAC` and `Diag`
represent all five modes exactly. T12 is that check, promoted to a test.

### 5.2 `diag_py` averages the product; the optimizer averages the factors

The first version of `kron_py` was written to satisfy "its diagonal is `diag_py`'s reading". It
does not, and the test that asserted it failed for a real reason rather than a tolerance:
`diag_py_reading` (lot 3's) accumulates `outer(S_D, H_D)` per micro-batch and averages the
**product**, while the optimizer keeps one running average per **factor** and multiplies at the
end. The two differ by the covariance of `H_D` and `S_D` across micro-batches.

Neither is wrong, but only one is the EMA-free counterpart of `p2_diag`. So lot 3's function is left
untouched — its A1, A2 and A3 rows stay exactly what they were — and `diag_py_factor_reading` is
added beside it. **Both** are emitted (`diag_py`, `diag_py_factors`), which turns the choice into a
measured number in the CSV instead of an argument made here. On one micro-batch they agree exactly,
which is what the new test pins.

### 5.3 `c* = 1.6e4` is `CLAUDE.md` §4.3's `1/batch²`, measured for the first time on a real block

On the real `cnn_gn_cifar/diag/ckpt_0.5`, the P1-py readings sit at `c* = 1.599e4` (`kron_py`) and
`1.619e4` (`diag_py_factors`) against the exact Fisher block of the head. The training batch is 128,
so `batch² = 16 384`. That is the factor `CLAUDE.md` §4.3 derives from the optimizer taking the
gradient of the **batch-mean** loss, confirmed end to end on a trained network rather than on paper.

It has a direct consequence the first draft of this plan missed: **`ρ` on a scale-free reading has
to be evaluated on `c*·K`**, or the sweep's `λ` swamps the operator and `ρ` reports the
plain-gradient step whatever the structure's shape. Lot 3 already did this for `diag_py`, but its
implementation only handled a `Diag`; `kron_py` is a `Kron` and a `Dense`. `p1_structural.rescaled`
now covers every block kind, and the P2 structures are deliberately **excluded** from it: their
scale is the object under measurement, not a nuisance.

### 5.4 On one layer of one model, the operational preconditioner is numerically a multiple of the identity

Measured on `cnn_gn_cifar/diag/ckpt_0.5`'s head, at the real re-warm length of `10 · TCov = 1 000`
steps, `cond(F̃)` — the ratio of the largest to the smallest eigenvalue of the operator the
optimizer inverts:

| mode | `k = 1` | `k = 3` | `k = 10` |
|---|---:|---:|---:|
| `kfac` | 2.98 | 3.51 | **1.02** |
| `ekfac` | 1.00 | 1.00 | **1.00** |
| `tkfac` | 1.00 | 1.00 | **1.11** |
| `tekfac` | 1.00 | 1.00 | **1.00** |
| `diag` | 1.16 | 1.07 | **1.05** |

A condition number of 1 means `F̃ = c·I`: the preconditioner has no preferred direction at all, and
the step it produces is the plain gradient's, rescaled. The mechanism is the one
`plan_lambda_dominance.md` measured from the other side — the joint effect of the EMA coefficients
summing to 0.088 and of the batch-mean gradient's `1/batch²` puts the stored curvature about
`2 × 10⁸` below the quantity it estimates, so `λ = 10⁻³` sits above every direction of it. The same
smoke gives `c*(p2_ekfac) = 2.7e-2` on that block; **if** `F̃ ≈ λI`, which the condition numbers say
it is, then `c* = λ̄/λ` and `λ` is **37 ×** the mean eigenvalue of the exact Fisher there
`[DERIVED from the measured c*, under that approximation]`.

Two cautions, and they are why this is a finding recorded during implementation rather than a
result: it is **one layer, one checkpoint, one model**, and the numbers come from a smoke run, not
from a job. What it does establish is that the machinery measures the object it is supposed to, and
that `ρ(P2) ≈ ρ(identity)` — which will appear in the real runs — is a measurement rather than a
bug. The `k = 1` and `k = 3` columns are also a direct sighting of §0.6's identity seed: `kfac`'s
operator is *more* anisotropic at a short re-warm than at a correct one, because `0.08^k · I` is
then comparable to the accumulated factor.

### 5.4b The vintage guard, and why the obvious version of it is wrong

`precondition` applies the inverse cached at the last `refresh`; the snapshot builds the operator
from the current factors. With `TCov == T_inv` those are the same vintage — but "is" and "should be"
are different claims, so the snapshot measures it and **refuses** rather than reporting an operator
the optimizer does not apply.

The obvious check, `‖Ã A_inv − I‖`, is the wrong one, and it was written first: it conflates a stale
inverse with an ill-conditioned factor, and this campaign has both — A1's first input factor is
`785 × 785` of rank **646**, with 136 exactly-zero eigenvalues. It duly tripped on a fixture with
`λ = 0`, for a non-reason. Comparing the cached inverse against a **freshly computed inverse of the
same matrix** cancels the conditioning by construction: same input, same call, so a same-vintage
pair agrees to round-off however singular the factor is, while a different vintage is `O(1)` away.

Measured at the real re-warm length (`10 · TCov = 1 000` steps) on real checkpoints, all residuals
against a `10⁻²` bound:

| model | `kfac` `A` / `B` | `tkfac` `Φ` / `Ψ` | `ekfac` | `δ` vintage |
|---|---|---|---|---|
| `cnn_gn_cifar` | `5.2e-7` / `4.0e-8` | `2.5e-7` / `8.2e-8` | no cached inverse | `0.0` |
| `mlp_ln_mnist` | `1.7e-7` / `7.2e-8` | `1.3e-6` / `1.5e-7` | no cached inverse | `0.0` |

What is left is the precision gap — the optimizer inverts in fp32, the check compares against fp64 —
and it is four to five orders under the bound. **A1's `ekfac` re-warm ran clean**, which was not
guaranteed: it is exactly the rank-deficient factor on which cuSOLVER's `eigh` raised
`_LinAlgError` and killed two campaign-1 training runs, so this is `_eigh_utils.py::eigenbasis`
holding on the case it was written for (§0.15).

### 5.5 The inertness claim, checked directly rather than argued

§0.12 claims that with no extras the P1 path is what it was. The 657-test suite is the wide check;
the direct one is a paired run on the **real** `cnn_gn_cifar/diag/ckpt_0.5`, three modules, 256
probes, both runners on the same probes: **340 of 340 P1 rows are bit-identical** (`worst relative
difference 0.000e+00`), and the only structure the P2 run has that the P1 run does not is `tekfac`,
which it enables on purpose. That is exit criterion 8 in miniature, and it is what makes reusing
`run_source` safe rather than merely convenient.

### 5.5b Local verification before submitting anything

Four runs on **real** checkpoints, not fixtures, each covering a path a fixture cannot:

| run | what it exercised | result |
|---|---|---|
| `cnn_gn_cifar`, 3 modules, both variants, 5 modes | the whole pipeline end to end | 1 111 rows; row checks `4.4e-15` / `4.4e-16` |
| the same, pure P1 | the inertness claim | **340/340 P1 rows bit-identical** (§5.5) |
| `vit_micro_cifar`, 2 fractions, 2 sources, folds + intervals | the multi-checkpoint loop, the variant gating, both sources | replica present at `1.0` and absent at `0.5`, as asked; P2 rows carry fold intervals |
| `vit_micro_cifar`, **unrestricted** | the one path a restricted run cannot reach | all 16 parameter blocks, and `pos_embed` — 2 048 of 21 098 parameters, belonging to no module — correctly gets `p2_identity` and is named in `meta.json` as unhooked |

The last one matters because "the optimizer preconditions nothing here" is a **result about the
model** (§0.7), and the failure mode is silence: a block that quietly never receives an operator
would simply be missing from the table.

### 5.6 Traps met while building, each now guarded

1. **The re-warm's model must be its own.** With normalisation layers in train mode — which P2
   requires — a `BatchNorm2d` rewrites its running statistics on every forward, so after 1 000 steps
   a shared network would carry buffers that have nothing to do with the checkpoint's. Two models,
   and an assertion that the frozen re-warm left every parameter bit-identical.
2. **Overriding only `solve` would have split M3 in half.** A wrapper whose `solve` ignored `λ`
   would still have had `stein_kl` add `λI` to `K.to_dense()` for its trace term while
   `K.logdet(λ)` did not — the two halves of the expression describing different matrices. The
   `k_lam` argument keeps them the same object (§0.3).
3. **A test that compared the campaign's reading with the optimizer's failed at 0.1 relative**,
   because the reading runs under `reference_mode` (eval) and the freshly-built network was in train
   mode: a `BatchNorm2d` normalising by batch statistics is a different function, and so is
   everything downstream of it. That difference is precisely what the P1-py → P2-raw rung measures;
   inside a test it just made the two sides incomparable.
4. **A 20-step smoke measured the identity seed, not the optimizer.** With `TCov = 100`, twenty
   steps are *one* factor update, so the state is `0.08·I + 0.008·new` and every mode's `ρ` came out
   equal to the plain gradient's. The runner refuses such a length by default for exactly this
   reason; the smoke had to pass `--allow-short-rewarm` to reach it.
5. **The smoke job asked for more partitions than exist.** `folds.partitions(folds, count)`
   refuses `count` above `C(folds, folds/2)/2`, and a two-fold smoke has **one** balanced
   partition, not the three the job asked for — it would have died on a `ValueError` several
   minutes in, after paying for the re-warms and the reference build. Found locally, before
   submission, by checking the arithmetic rather than by running it: `--folds 4` gives exactly 3.
   The four real jobs are fine (`C(10,5)/2 = 126` and, for A1, `C(8,4)/2 = 35`, against 20 asked),
   and the probe count divides `folds × batch` in every one of them — `FoldPlan` refuses unequal
   folds, and an unequal pair would silently mix two sample sizes inside one interval.


### 5.7 The naive agentic audit, and the P0 it found

§3 step 7 makes an adversarial audit a gate in front of submission, and it earned its place. Two
independent auditors were run on the finished implementation — one on the code (layouts, silent-wrong
hazards, inertness, SLURM flags), one on the protocol (is the comparison apples-to-apples, are the
rules falsifiable, is anything the campaign requires missing). Every finding below was **fixed before
any job was submitted**.

**The P0, and it would have invalidated the lot's headline.** §0.9's first version read "median `τ`
near 0" as "the rankings differ", i.e. as HF7 confirmed. But `e_F_star` and `ρ` are both invariant
under `K → cK`, and §5.4 had *already measured* that the operational operator is proportional to the
identity — so the P2 ranking is a batch draw, `τ ≈ 0`, and the rule would have announced **HF7
confirmed with no evidence at all**, in 98.7 % of simulated draws of pure noise. The informative
claim was the one that needed no evidence. §0.9 is rewritten around two gates (a degeneracy gate and
a reproducibility ceiling, both built from the `p2rep_*` replica the runner already produces), and
the corrected rule confirms under noise in **0.1 %** of draws while still detecting a real
disagreement in 85 % and a real agreement in 100 % — pinned by three tests.

The other findings, in the order they mattered:

| # | finding | fix |
|---|---|---|
| P0-2 | five of §0.9's rules were **prose only**; `lot5_decisions.py` implemented none of them, and a `NaN` `τ` (which an all-tied P2 side produces) silently corrupts `statistics.median` | the file is rewritten; `median()` drops `NaN` first, and the dropped cells are counted |
| P0-3 | the `diag` pair used `af_raw`, the object §0.14 item 1 records as *wrong* for exactly this comparison | the primary ranks the four Kronecker modes at the P1 rung; `diag` moves to a five-mode variant against `diag_py_factors` |
| P1-4 | the **source** was an unnamed confound larger than the terms the ladder names: against `F`, the `P1 → P1-py` step carries the whole type-2 → empirical change (lot 1 measured that gap at 1.42 relative) | `Ê` is the primary reference; `--fold-sources type2 empirical` so both carry intervals |
| P1-5 | the running average's **effective sample size is ~150 examples**, so the `P1-py → P2-raw` rung is mostly sampling variance, not "the EMA and the min-max" | §0.4c's `kron_py_1batch` control separates them |
| P1-6 | "decided when two models agree" could be met by **one architecture twice** (A2-GN and A2-BN are the same net) | lot 3's own wording restored: A2-GN and A3 decide, A2-BN is a robustness check |
| P1-7 | the normalisation verdict was a `τ` over **two** items, which is always `±1` and partitions at 50 % | a sign-agreement rate with a binomial interval |
| P1-8 | confirm was an OR over six correlated statistics, refute an AND — two shots for the hypothesis, none for the null | `e_F_star` is the single voting statistic; `ρ` is reported across the sweep |
| P1-9 | the tie test compared **marginal** intervals where the paired difference is available and correct | the five `(P2, P1)` and five `(p2rep, p2)` pairs joined `p1_structural.PAIRS` |
| P1-10 | the decision script's cell key dropped `arm` and `seed`, so a second seed would silently overwrite the first | both are in the key |
| P1-11 | `half_values` omitted the `c*` rescaling `analyse_block` applies, so a scale-free reading's interval was computed on the degenerate object | applied in both |
| P2-13 | the replica used `seed + 1`, which replays the main run's own RNG stream shifted by one step — not an independent draw | `+10 000` |
| P2-14 | five numbers stated as measured or derived that were not: A3's share was **88.78 %**, not 90.3 % (the wrong figure silently counted the LayerNorms the same sentence excluded); the TF32 gap is an estimate; §5.4's heading generalised one layer | each corrected and tagged |
| P2-16 | three tests §2 promised did not exist, while exit criterion 2 claimed they passed | written, including an end-to-end runner test on a real checkpoint |
| — | the smoke job asked for 3 balanced partitions of 2 folds, of which **1** exists | `--folds 4`; found before submission, see §5.6 item 5 |

**The most useful thing the code audit produced was not a bug.** It rebuilt the normalisation-layer
check from scratch — a per-example `dγ`/`dβ` Gram against the campaign's reference block — and
confirmed both halves independently: the reference block really is blocked (`1.0e-16` and `5.9e-17`
against `‖B‖ ≈ 0.4`, where the interleaved reading is `0.40` / `0.36` off, i.e. ~99 % wrong), and
the snapshot's applied operator matches `precondition` to `0.0 … 1.8e-15` for all five modes on all
four hooked layer types.

Then it measured the negative control, and that is the part to carry: **undoing the permutation
moves the applied result by only `2.1e-3 … 3.0e-2` relative** — because §5.4's own finding is that
the operational operator is nearly `c·I`, and a permutation of `c·I` is `c·I`. So a forgotten
permutation would be **smaller than a fold interval on `e_F`**: invisible in the CSV, invisible in
every figure, and visible only to the test that compares the applied preconditioners directly.
`test_t12_norm_layout_permutation_is_not_vacuous` is therefore not a formality; on this lot's data it
is the *only* thing that can catch the bug §0.2 exists for.

**One bug the audit did not find, and the fix for it is a test.** Running `lot5_decisions.py` on
real output for the first time produced an **empty report** — no error, no warning, nothing. Two
vocabulary mismatches between the writer and the reader: the rule compared the CSV's `source`
column against `"E_hat"`, which lives in the `reference` column (`source` is `type2` / `empirical`);
and it restricted the primary verdict to `("linear", "conv")` while `registry.LAYER_TYPES` calls an
unshared `Linear` `linear`, a token-wise one `linear_shared` and the output module `head` — so on
`cnn_gn_cifar` it would have dropped the head, and on `vit_micro_cifar` the eight token-wise layers
as well. Both are now `PRIMARY_KINDS` / `PRIMARY_SOURCE` constants, and
`test_decision_vocabularies_match_what_the_runner_writes` checks them against the registry's own
tuple and the runner's own sources, because the failure mode is silence.

**Two findings were accepted rather than fixed**, and are now in the plan as declared limitations:
`plan_exp_draft.md` §10.3's regime-B clause cannot be met until lot 4 (§4), and §0.8's experiment
establishes that the identity seed has decayed, not "fidelity" to a state that no longer exists
(§0.8). A third is recorded as a caveat on any P2 `ekfac`-vs-`kfac` reading: the operational EKFAC
pairs an `s*` accumulated in the *previous* refresh's basis with the *current* basis, faithfully,
because that is what the shipped optimizer does — so it is **not** "the optimal diagonal in the basis
it is applied in", and a P2 ranking of it is not a test of EKFAC's own theorem
(`adafisher_state.py`'s docstring).

The audits also reported nothing in ten categories, of which the two worth naming are the
normalisation-layer permutation (§0.2 — the lot's self-declared most dangerous point) and the
`k_lam` design (§0.3).

## 6. Results

Six jobs, all **COMPLETED** with exit code 0 on 2026-09-18/19: **21388028** (re-warm fidelity,
`00:28:16`), **21388029** (smoke, `00:42:25`), **21388030** (A3, `01:58:21`), **21388031** (A2-GN,
`06:58:35`), **21388032** (A2-BN, `07:25:11`), **21388033** (A1, `02:04:48`). Raw numbers under
`fisher_ref/outputs/p2/<model>/diag/seed0/<fraction>/`; the verdicts come from
`experiments/lot5_decisions.py` applying §0.9's pre-registered rules, and the fidelity verdict from
`experiments/rewarm_fidelity_lot5.py` applying §0.8's.

### 6.1 What can be trusted here

**The runs.** 181 574 rows across 4 models × 5 checkpoints × 2 sources. **Zero**
`invariant_violation` rows, **zero** non-finite values. The per-sample row checks against autograd —
the sum rule and the single-example rule — are `≤ 1.04e-15` on every model and both sources. Host
peaks came in at 77.9-81.6 GB against the 80.7 GB lot 3 measured and this plan predicted; every job
finished well inside its `--time`.

**Exit criterion 8, the free regression test on the whole reuse.** On A2-GN, A2-BN and A3, which
re-use lot 3's own `45 000 / 250 / 10` configuration, the recomputed P1 rows are compared with lot
3's published ones: on the **type-2** source the worst relative difference over 33 065 rows is
`3.47e-13`, with **zero** rows above `1e-10`. HF7's own five P1 structures agree to `≤ 5.5e-10`, and
the verdict statistic `e_F_star` to `≤ 1.5e-13`. Injecting the P2 rungs and the TEKFAC basis
perturbs the P1 path by nothing.

**The one exception, and it is a finding rather than a defect — see §6.6.** On the *empirical*
source, 2 % of rows differ by more than `1e-10`, concentrated almost entirely in one structure,
`ekfac_reduce`, and only in its inverse-based metrics (`rho`, `cos_ngd`, `stein_kl`), up to `0.376`.
It does not touch lot 3's HF2 verdict (read on type-2 `e_F`, which involves no inverse) nor any
lot-5 statistic.

### 6.2 Exit criterion 5: the re-warm, confirmed on all four models

§0.8's pre-registered rule returns **confirmed** for `cnn_gn_cifar`, `cnn_gn_cifar_bn`,
`vit_micro_cifar` and `mlp_ln_mnist`. At `k = 10` the re-warm gap on the applied preconditioner is
within **0.37-1.82×** its own batch-draw noise floor on all 20 (model, mode) pairs, and going to
`k = 20` changes it by less than a factor 2 everywhere. The staleness term is the same order as the
floor at every model and mode — §3.2's third finding, previously measured on A2 alone.

`k = 3` is where §0.6's arithmetic becomes visible, and it is more violent than §3.2's A2-only table
showed:

| mode | gap / floor at `k = 3` | applied preconditioner wrong by |
|---|---|---|
| `diag` | 1.36 - 2.51 | 0.1-0.3 % |
| `kfac` | 2.24 - 16.2 | 7.9-13 % |
| `ekfac` / `tkfac` / `tekfac` | **4.0×10⁵ - 8.0×10⁵** | **52-87 %** |

The ratio explodes not because the gap grows but because the *floor collapses*: at `k = 3` the state
is dominated by the deterministic identity seed, which two replicas share exactly, so their mutual
distance falls to `~1e-6` while the gap to the converged state stays at `0.5-0.87`. That is the
spurious-extra-damping mechanism of §0.6.3, measured.

### 6.3 HF7: the rule says confirmed — and the honest reading is different, and stronger

§0.9's rule returns **CONFIRMED (the rankings differ)** on all four models, on the primary
(four Kronecker modes at the P1 rung) and on the five-mode variant:

| model | median `τ(P1, P2)` | reproducibility ceiling `τ(P2, P2rep)` | degenerate | dropped |
|---|---:|---:|---:|---:|
| `cnn_gn_cifar` | **−0.67** | +1.00 | 0 | 0 |
| `cnn_gn_cifar_bn` | −0.33 | +0.83 | 0 | 0 |
| `mlp_ln_mnist` | −0.33 | +1.00 | 0 | 0 |
| `vit_micro_cifar` | −0.33 | +0.67 | 0 | 0 |

The degeneracy gate that §0.9 built the whole rule around had a real chance to fire and did not, and
the ranking reproduces across two independent re-warms. By the letter of the pre-registration, HF7
is confirmed, and the rankings are not merely different but *anti-correlated*.

**And that is not the result.** Apply the campaign's own reading rule instead —
`plan_exp_draft.md` §10.3: *"a difference between two approximations is reported only if it exceeds
`F`'s own 95 % noise floor"*:

| model | spread between the five modes, **P1** | spread, **P2** | per-layer noise floor | P1 / floor | P2 / floor |
|---|---:|---:|---:|---:|---:|
| `mlp_ln_mnist` | 0.396 | **0.0003** | 0.167 | 2.4 | **0.0016** |
| `cnn_gn_cifar` | 0.348 | **0.0000** | 0.026 | 13.2 | **0.0001** |
| `cnn_gn_cifar_bn` | 0.453 | **0.0000** | 0.029 | 15.7 | **0.0002** |
| `vit_micro_cifar` | 0.492 | **0.0000** | 0.034 | 14.7 | **0.0000** |

At the structural rung the choice of mode is 2-16 × the noise floor, i.e. reportable. At the
operational rung it is **100-1000 × below** it. So there is no P2 ranking to compare, and **HF7 as
worded is ill-posed at the operational point**: it asks which of two orderings is which, when one of
them does not exist.

**This is a correction to §0.9's own gate, not to the data.** The gate compared the P2 spread with
the *operator's replica noise*, which turns out to be `1e-8`-`6e-6` — so a spread of `3e-4` cleared
it by 33-145 ×. That yardstick is too permissive: it asks "is this difference bigger than the
estimator's own jitter", when the campaign's standing rule asks "is it bigger than the reference's
sampling noise". The second is the binding one, and lot 5's verdict is reported under it.

### 6.4 What is actually there: the operational preconditioner is the identity

| measurement | result |
|---|---|
| `cond(F̃)` at `θ = 1`, the job's own 1 000-step re-warm | **1.0000 - 1.1035** `[MEASURED on 25 (mode, block) pairs: all five modes × `cnn_gn_cifar`'s four hooked blocks, plus `mlp_ln_mnist`'s five]` |
| `λ` as a share of `F̃`'s mean eigenvalue, same 25 pairs | **98.9 - 100.0 %** |
| `\|cos_F(p2_mode) − cos_F(identity)\|`, 2 550 cells | median `7e-7` - `8e-5`, p95 `≤ 9e-3`, max `1.9e-2` |
| `\|ρ(p2_mode) − ρ(identity)\| / ρ(identity)`, 7 750 cells | median `1.4e-6` - `2.0e-4`, p95 `≤ 1.2e-2`, max `0.116` |

Per mode, the `cond(F̃)` ranges are `ekfac` and `tekfac` **1.0000-1.0013**, `kfac` 1.0003-1.0206,
`diag` 1.0001-1.0568, **`tkfac` 1.0010-1.1035** — the same ordering §6.6(b) finds from `ρ`, measured
independently. The last two rows are the ones that carry the claim, because they come from **the
jobs' own output**, over every hooked block of every model at every checkpoint, both sources and all
five dampings. `ρ` is invariant under
`K → cK`, so `F̃ = c·I` implies `ρ(F̃) = ρ(identity)` *exactly*; measured, the two agree to a median
of one part in `10⁴` to `10⁶`, and the whole distribution's 95th percentile is `≤ 1.2 %`.

In plain words: **the preconditioner the optimizer divides by has no preferred direction. The step it
produces is the plain gradient's, rescaled.** For comparison, the same structures in their P1 form
deliver **0.58-0.88** of the ideal quadratic decrease where the operational ones deliver
**0.18-0.42**, which is exactly what doing nothing delivers.

This is `plan_lambda_dominance.md` seen from the other end. That report measured the *state*: the EMA
coefficients sum to 0.088 and the backward hook takes the gradient of the batch-mean loss, so the
stored curvature sits about `2×10⁸` below the quantity it estimates. Lot 5 measures the
*consequence*: a fixed `λ = 10⁻³` then accounts for essentially all of `F̃`.

### 6.5 Where the fidelity is lost — and it is not the averaging

The four-rung ladder of §0.4, with §0.4c's sample-size control splitting the estimator rung.
Median step in `e_F_star` over every hooked linear/conv block, empirical source, at `θ = 1`:

| model | formula | sample size (45 000 → 128) | EMA + min-max + train mode + fp32 | damping |
|---|---:|---:|---:|---:|
| `cnn_gn_cifar` | 0.000 | 0.001 | 0.105 | **0.232** |
| `cnn_gn_cifar_bn` | 0.000 | 0.013 | 0.020 | **0.256** |
| `mlp_ln_mnist` | 0.000 | **0.275** | −0.035 | 0.060 |
| `vit_micro_cifar` | −0.000 | 0.010 | 0.030 | **0.126** |

Three readings, and the third is the one that matters.

1. **The optimizer's own factor formulas cost nothing.** `kron_py` equals the campaign's `kfac` to
   three decimals everywhere. The caveat is that `e_F_star` is scale-free by construction, and the
   two known formula differences — `CLAUDE.md` §4.3's `1/batch²` and lot 3 §5.2's `1/T` — are pure
   rescalings. So what this establishes is that they add no *directional* error, which was not
   obvious for the `2 × 2` normalisation surrogate or the `Conv2d` scale quirk.
2. **The effective sample size costs a great deal, but only where weight sharing does not rescue
   it.** The running average's effective sample size is `1.17` factor updates — about **150
   examples** (§0.4c). On A1, whose first `Linear` has a `785 × 785` input factor, dropping from
   45 000 probes to 128 costs **0.275**; on the CIFAR convolutions it costs `0.001-0.013`, because
   128 examples are 128 × 1024 *patches*. Weight sharing is what makes the operational estimator
   viable at all.
3. **The damping is what destroys the structure, and the averaging is not.** Measuring the distance
   to the identity *before* and *after* the optimizer's own `λ`:

| model | `\|cos_F − cos_F(identity)\|`, **P2-raw** (no `λ`) | **P2** (with `λ`) | ratio |
|---|---:|---:|---:|
| `mlp_ln_mnist` | 0.068 - 0.253 | `1e-6` - `5e-4` | 139× - 177 000× |
| `cnn_gn_cifar` | 0.010 - 0.430 | `2.5e-8` - `4.1e-5` | 8 700× - 2.6 M× |
| `cnn_gn_cifar_bn` | 0.022 - 0.369 | `2.6e-8` - `1.0e-4` | 221× - 5.8 M× |
| `vit_micro_cifar` | 0.068 - 0.345 | `1.9e-8` - `1.1e-5` | 17 000× - 3.6 M× |

   After the EMA, the min-max, train mode, the augmentation, fp32 and the 150-example sample, the
   operator still carries `0.01`-`0.43` of structure. Adding `λ` takes it to `10⁻⁸`-`10⁻⁴`. **The
   smoothing is not the problem. The damping is.** §11's decision rule lumps "the EMA / min-max /
   damping" together; this separates them, and only the last one is guilty.

**§0.4b's second diagnostic closes the estimator rung.** `p2clean_raw_*` re-warms on the campaign's
own clean probes in eval mode — no crop, no flip, no Cutout, no batch statistics — with everything
else unchanged. Against `p2_raw_*` at `θ = 1`, median over the five modes:

| model | augmentation + train mode, in `e_F_star` |
|---|---:|
| `cnn_gn_cifar` | **+0.031** |
| `vit_micro_cifar` | +0.009 |
| `mlp_ln_mnist` | +0.007 |
| `cnn_gn_cifar_bn` | +0.004 |

Always in the expected direction — data the reference was not built on fits it worse — and always
**small**: at most 3 % of `e_F_star`, against the damping's 0.06-0.26. So of the "EMA + min-max +
train mode + augmentation + fp32" column above, roughly a third is the data on `cnn_gn_cifar` and
much less elsewhere; the rest is the smoothing itself. Worth one caution: A2-GN and A2-BN see the
*same* images and the same augmentation, and differ by 8× here, so this number is not a clean
measurement of "augmentation" alone — it also carries whatever the two normalisations do differently
between train and eval mode, and at this magnitude the two cannot be separated.

### 6.6 Two findings that belong to other lots

**(a) `ekfac_reduce`'s inverse metrics are not reproducible, for a structural reason.** Chasing
§6.1's exception led to a property of the reduce construction nobody had looked at. The reduce-mode
output factor is `G_reduce = (1/N) Σ_n (Σ_t g_{n,t})(Σ_t g_{n,t})ᵀ`, and on any layer followed by a
normalisation it is **exactly rank-deficient**: a normalisation's backward makes the gradient sum to
zero within each group, and `reduce` sums over exactly the positions that constraint applies to.
Measured on `cnn_gn_cifar`: `rank(G_reduce) = channels − num_groups`, **8 of 16** on `features.0` and
24 of 32 on `features.4`, with the predicted number of *exactly* equal (zero) eigenvalues in each
case.

Inside that degenerate null space the eigenvectors are numerically arbitrary. K-FAC does not care —
the eigenvalue is constant there, so `1/(λ_G λ_A + λ)` is too, and any rotation leaves the operator
alone. **EKFAC does care**, because it assigns a *data-estimated* `s` that is not constant on that
subspace. Measured directly: a `3.2e-16` relative perturbation of `G_reduce` rotates the null-space
eigenvectors by **0.53**, leaves the others at `2.9e-15`, moves `kfac_reduce`'s applied step by
`1.2e-12` and **`ekfac_reduce`'s by `0.144`**.

Consequence, and it is lot 3's to carry: `e_F` and `cos_F` of `ekfac_reduce` are fine, so **HF2's
verdict stands** (§0.10 there votes on type-2 `e_F`); but any `ρ`, `cos_ngd` or Stein-KL number for
`ekfac_reduce` is reproducible only to `~10 %`, and should not be quoted more precisely than that.

**(b) The only mode that survives its own damping is the one whose `λ` is relative.** Ranking the
five by how far the operational preconditioner departs from doing nothing —
median `|ρ(p2) − ρ(identity)|/ρ(identity)` over every block, model, source and damping:

| mode | departure | its damping |
|---|---:|---|
| `ekfac` | `1.3e-7` | `s* + λ` — absolute |
| `tekfac` | `1.5e-7` | `Θ + λ` — absolute |
| `kfac` | `1.8e-5` | `A + π√λ I`, `B + (√λ/π) I` — absolute |
| `diag` | `2.9e-5` | `H ⊗ S + λ`, factors min-maxed into `[0,1]` |
| **`tkfac`** | **`6.8e-5`** | `Φ + √(λ/δ) I` with `tr Φ = tr Ψ = 1` — **relative to `δ = tr(F)`** |

`tkfac` is 500 × further from the identity than `ekfac`, its departure *grows* along training
(`7e-6` at `θ = 0` to `1.6e-4` at `θ = 1`), and the single largest departure in all 7 750 cells is
`tkfac` on A1's `features.0` at initialisation: **+11.6 %** more of the ideal decrease than the
plain gradient. It is the only mode whose damping is normalised by the curvature's own scale.

That is quantitative support for fix **S1** of `plan_lambda_dominance.md` (make `λ` dimension-aware),
arriving from a different direction: not "the stored state is mis-scaled" but "the one mode that
already scales its `λ` by the state is the one whose preconditioner still does something".

### 6.7 What this lot does **not** establish

* **One arm, one seed.** Everything is `diag`, seed 0. All five modes' operators are read at weights
  the `diag` optimizer produced — an off-trajectory state for the four Kronecker modes. The `adamw`
  checkpoints exist and are one job per model (§0.9).
* **Regime A only.** `plan_exp_draft.md` §10.3 requires a per-layer-type conclusion to hold on a
  regime-A *and* a regime-B model; B1/B2 wait on lot 4, so every per-layer-type reading above is
  provisional (§4).
* **`λ` is not swept on the P2 side**, by construction: the operator carries the optimizer's own
  single `λ`. Whether these conclusions survive a different `λ` is the obvious next experiment, and
  `plan_lambda_dominance.md`'s own `lam` bracket on `mnist_autoencoder` says the four Kronecker modes
  move by under 1 % over four orders of magnitude — which, read together with §6.4, now has a
  mechanism: `λ` cannot matter to the *loss* through a preconditioner that is `λ·I` whatever `λ` is.
* **The `mnist_autoencoder` stall is not addressed here.** It is B0, a regime-B model (lot 4). But
  §6.4 supplies the hypothesis its §3.6 measurement should test first.

### 6.8 Relation to the two investigations run in parallel

`plan_lambda_dominance.md` and `validation_noise_investigation.md` reached §6.4's conclusion first,
by a different route, and lot 5 must be read against them rather than as a discovery.

**The same fact, measured on three different objects, and the numbers cross-check.**

| where | object measured | result |
|---|---|---|
| `plan_lambda_dominance.md` E1 | the **stored curvature** against `λ`, direction by direction | `λ` larger in **105 of 105** cases, every direction |
| `validation_noise_investigation.md` Step 7 | the same, 7 networks × 5 modes | curvature `< 3 %` of `λ` in 99 % of directions; the largest anywhere is **7 % (`diag`)**, **~0.1 %** (Kronecker modes) |
| **lot 5 §6.4** | the **assembled operator `F̃`**, and the **step it produces** | `cond(F̃) ∈ [1.0000, 1.1035]`; per mode `diag` **5.7 %**, `ekfac`/`tekfac` **0.13 %**; `ρ(P2) = ρ(identity)` to `10⁻⁴`-`10⁻⁶` |

Step 7's "7 % for `diag`, ~0.1 % for the Kronecker modes" and §6.4's per-mode condition numbers are
the same two numbers, obtained from the state and from the operator independently. That agreement is
worth more than either measurement alone.

**What lot 5 adds.** Those two studies measure the optimizer against *itself* — curvature against
its own `λ`. Lot 5 measures it against the **exact Fisher on the same probes**, which is what turns
"the division does not adapt" into a price: the operational preconditioner delivers `0.18`-`0.42` of
the ideal quadratic decrease where the same structure computed properly delivers `0.58`-`0.88`. It
also separates §11's three suspects (§6.5: the averaging leaves `0.01`-`0.43` of structure, `λ`
removes it), and supplies the positive control for fix **S1** that both reports ask for (§6.6b:
`tkfac`, the one mode whose `λ` is already relative, is the one still measurably anisotropic —
confirmed twice, by `ρ` and by `cond(F̃)`).

**Where lot 5 is the weaker of the three, and this bounds how §6.4 may be quoted.**
`validation_noise_investigation.md` Steps 9-16 tested the "it is just momentum" reading against
**real training**, and it does not generalise: on `mlp_ln_mnist` a curvature-free stand-in is
indistinguishable from `kfac` (the gap equals the seed noise), but on `cnn_gn_cifar` and
`resnet20_cifar` it moves final accuracy by **8-33 ×** that noise; across 30 (network, mode) pairs,
11 clean passes, 12 a few percent off, 2 clear failures. Its Step 9 explains why, and the
explanation applies word for word to lot 5: **Step 7 measured a size, not a shape** — and so does
§6.4. An operator that is `(1 + ε)·I` with `ε ≈ 10⁻⁴` is indistinguishable from the identity *in one
step at fixed θ*, which is exactly what `cos_F` and `ρ` see, and can still move a 10 000-step
trajectory, because the departure is systematic rather than random and it compounds.

**So §6.4 says "the per-step preconditioner is numerically the identity". It does not say "the five
modes are interchangeable in training", and the parallel work has already shown that to be false on
convolutional networks.**

**One thing lot 5's data was checked against and does *not* explain.**
`validation_noise_investigation.md` Step 16 leaves open why `cnn_gn_cifar`/`kfac` is the exception,
with a correlational lead: it is the only mode whose applied step is anisotropic in the first ~200
steps. Lot 5 has per-mode, per-checkpoint departures from the identity, including at `θ = 0`, so the
lead is directly testable here. It fails: `kfac`'s departure on `cnn_gn_cifar` is `1.1e-4` at
`θ = 0` and `1.6e-5` at `θ = 0.01`, **smaller** than on `mlp_ln_mnist` (`1.9e-3`, `9.4e-4`) — the
network where the stand-in reproduces `kfac` exactly. The ordering is backwards.

The reason is a genuine blind spot rather than a contradiction: their early-training regime is a
*fresh* optimizer whose estimator has not converged, dominated by the start-up transient their Step 8
identified — and §0.6 **deliberately excludes exactly that regime**, refusing any re-warm shorter
than `10·TCov` in order to remove the transient. Lot 5 measures "initial weights, converged
estimator"; their exception lives at "initial weights, un-converged estimator". The two are
different objects, and nothing here speaks to theirs.
