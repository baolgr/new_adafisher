# Lot 1 — the exact references: capture, sources, dense `F` / `Ê` / `B_ℓ`

*Implementation plan and completion record for lot 1 of `plan_exp_draft.md` §9. In the repository's
own numbering this is lot 11; it is named after the experimental plan, like `plan_exp_lot0.md`.*

**Goal.** The first curvature matrices of the campaign: given a model, a set of probes and a
*source* of backprop vectors, produce the exact `F = U^T U`, the exact empirical `Ê`, and the exact
per-layer blocks `B_ℓ`, in fp64, in regime A.

**Non-goal.** No approximation (lot 2's `approx/`), no metric (lot 2's `metrics/`), no runner, no
figure. Nothing in `src/adafisher_modes/` and nothing in `benchmarks/` changes — `fisher_ref/` stays
the reader lot 0 made it.

**Status: done.** Code and tests (§1-§3), the real-checkpoint smoke (§4), and the A1 cluster run at
`N = 55 000` (§5, §6 — job 21082966, `COMPLETED` in `01:06:33`). Q1 is measured and interpretable;
HF1 is measured and **not** settled, for a reason §6.3 states precisely.

---

## 0. Decisions taken here, and the two things that were nearly wrong

### 0.1 `register_full_backward_hook` is the wrong hook, and the failure is silent

The obvious design is to copy `AdaFisherMulti`'s pattern (`optimizer.py:111-119`): a forward hook
for the input, a `register_full_backward_hook` for the output gradient. It has a hole.
`register_full_backward_hook` fires from the module's **input**-side node, so a module whose
`grad_input` is not needed by the `inputs=` requested of `torch.autograd.grad` is pruned out of the
backward graph and its hook never fires. Measured on a `Linear → LayerNorm → ReLU → Linear` in
fp64:

| call | `x.requires_grad` | module hooks that fired |
|---|---|---|
| `autograd.grad(out, params, grad_outputs=V)` | `False` | `0, 1, 3` (plus a `UserWarning`) |
| `autograd.grad(out, params, grad_outputs=V)` | **`True`** | **`1, 3` — module 0 missing, silently** |
| `autograd.grad(out, [x], grad_outputs=V)` | `True` | `0, 1, 3` |
| `autograd.backward(out, V)` | either | `0, 1, 3` |

The no-grad row is a fallback: with nothing requiring grad, torch fires from the output side and
says so in a warning. That fallback is what makes the naive design *look* correct. On
`mlp_ln_mnist` the module that disappears is the first `Linear`, **25 120 of 26 634 parameters**
(94 % of `P`) — and the symptom is not a crash: those columns of `U` are simply zero, `F` stays
symmetric and PSD, and every exactness test still passes on the layers that did fire.

Two consequences, both implemented:

1. `capture.py` takes `g` from a **tensor** hook registered on the module's output from inside the
   forward hook (`output.register_hook`). It is exactly `grad_output[0]`, fires unconditionally,
   and was checked to survive an in-place `ReLU(inplace=True)` applied to that output and a
   residual reuse of it — the two cases a ResNet and a ViT respectively make routine.
2. The dense driver backpropagates with `inputs=[batch]`, never `inputs=params`. That is also
   strictly cheaper: it skips every weight-gradient GEMM, which the driver does not use (the
   per-sample gradients are reconstructed from `(a, g)`).

`tests/test_fisher_ref_lot1.py::test_backward_hook_pruning_regression` pins **both** halves: that
the pruning still happens (so the finding does not rot into a superstition after a torch upgrade),
and that `Capture` is not subject to it.

### 0.2 `m = N(C−1)` is a rank, not a row count

`plan_exp_draft.md` §2.1 says the type-2 softmax root gives "`C` columns for a rank of `C−1`:
+11 % rows at `C = 10`", but §2.2 and §2.3's memory tables are written at `m = N(C−1)`. Both are
right about different things and they must not be read as one number: `U` has `N·C` **rows**, of
rank `N(C−1)`, because `S_n^T sqrt(p_n) = 0`. `DenseReference.n_rows` is `N·C` and says so in its
docstring; the null vector is asserted numerically rather than restated.

### 0.3 fp64 end to end, and `CAPTURE_DTYPE` is not lot 1's

`conventions.CAPTURE_DTYPE = float32` is regime B's allowance (`plan_exp_draft.md` §12:
"accumulate `U` in fp32 but Grams and `UKU^T` in fp64"). Lot 1 is regime A and T1's tolerance is
`1e-10`, which fp32 cannot reach, so the builder's `dtype` argument defaults to `REFERENCE_DTYPE`
and the model is cast to it (on a deep copy, only when a cast is actually needed — a reader must
not mutate the caller's model). The gap is **measured, not assumed**: on the tiny classifier,
building the same `F` with the capture in fp32 gives a relative Frobenius gap of **`1.1e-7`**. So
§12's advice is safe for Frobenius-level metrics and hopeless for an exactness test, which is
exactly the distinction to keep. Reported by `test_fp32_capture_gap_is_measured`, not asserted.

### 0.4 `curvlinops`: no — but T1 still has an independent oracle

`plan_exp_draft.md` §2.4 makes this a lot-1 exit criterion, and warns that if the cluster wheelhouse
does not carry it, "regime C's oracle role falls back to a hand-written `Fv` (double backward),
which T1 then has nothing independent to check against". **The warning does not apply**, so the
decision costs nothing: `curvlinops` is not added.

- Verified absent from `.venv`, from `pyproject.toml` and from `requirements-cluster.txt`, which
  installs `--no-index` against the Alliance wheelhouse.
- T1's oracle is genuinely independent without it: `Jv` by `torch.func.jvp` (forward mode, torch
  2.14), then `Lambda_n u = diag(p)u − p(p^T u)` in closed form, then one VJP. It never forms a
  root of `Lambda`, never calls `sources.py`, and gets its per-example structure from `torch.func`
  rather than from hooks. Measured agreement with the dense build: **`3.8e-16`** worst relative
  gap over 20 random vectors, against the `1e-10` T1 asks for.
- It lives **in the test file**, not in `fisher_ref/`. `plan_exp_draft.md` §7 assigns
  `reference/matfree.py` to lot 6; shipping the oracle as API now would invite sharing code with
  `dense.py`, and sharing code is precisely what would make T1 vacuous.
- `test_curvlinops_cross_check` is an `importorskip` third opinion for anyone who installs it.

### 0.5 The column layout is `named_parameters()` order, not `rvec([W | b])`

`adafisher_modes` works in the bias-augmented per-module layout `rvec([W | b])`, which is the one
the Kronecker formulas are written in. `U`'s columns are **not** in that layout. Three reasons, in
increasing order of weight:

1. a parameter-space vector — a gradient, a `θ`, a random direction — needs no conversion, so the
   T1 oracle compares like with like and a wrong permutation cannot hide;
2. `weight` is registered before `bias` in every relevant module, so a module's parameters are
   contiguous and `B_ℓ` is a **slice** of `F` rather than a second construction (asserted on all
   eight built models by `test_block_slices_are_contiguous_on_every_model`);
3. decisively, `F` must span **every** parameter, including the ones that have no `[W | b]` slot at
   all — `pos_embed`, `cls_token`, a `GroupNorm` affine. A per-module augmented layout cannot
   represent them without inventing slots.

Lot 2 adds the small index permutation into `rvec([W | b])` when it has a K-FAC to compare against;
it is not built now.

### 0.6 `x̂` is recomputed, and `augment_norm_input` is not reused

`∇_γ = Σ_t g_t ⊙ x̂_t` needs the **normalised** activation, while a forward hook on
`nn.LayerNorm` / `nn.BatchNorm2d` / `nn.GroupNorm` sees the input *before* normalisation —
`plan_exp_draft.md` §4 flags this as "a point to check in the code before any conclusion". Checked:
under `reference_mode` (eval) all three are closed-form from the captured input plus the module's
own `eps` and buffers, and the result is exact. Measured against autograd on all four shapes:

| layer | `(g ⊙ x̂).sum(t)` vs `weight.grad` |
|---|---|
| `LayerNorm`, 2-D input | `4.4e-16` |
| `LayerNorm`, 3-D input | `8.9e-16` |
| `BatchNorm2d`, eval | `1.8e-15` |
| `GroupNorm` | `8.9e-16` |

`factors.augment_norm_input` is **not** reused for this: it returns the `(T, 2)` Frobenius-optimal
surrogate `[z_x, 1]` of `plan_lot5.md` §0.2, built on the *pre*-normalisation input — the right
object for the four Kronecker modes and the wrong one for an exact reference. The only helper this
lot imports from `factors.py` is `extract_patches`.

A `BatchNorm*` in **train** mode is refused rather than approximated (`x̂` would depend on the rest
of the batch, so there are no per-sample gradients to stack at all), as is one with
`track_running_stats=False`.

### 0.7 An uncovered parameter is an error, not a zero column

A parameter belonging to no capturable module would contribute all-zero columns to `U`, leaving `F`
symmetric, PSD and quietly wrong. `build_dense_reference` therefore refuses, naming the offenders,
unless the caller restricts the build with `modules=[...]`. Measured coverage over the eight built
models: the uncovered set is **exactly** the raw `nn.Parameter`s — `pos_embed` (A3, B1),
`cls_token` + `pos_embed` (B4) — and nothing else. In particular `cnn_gn_cifar`'s six `GroupNorm`
parameters **are** covered, though `registry` reports them `hooked=False`: the campaign measures
curvature, not what the optimizer preconditions (`plan_exp_lot0.md` §0.6). Locked by
`test_coverage_lock`, with the "a new model folder with no entry fails the suite" companion lot 0
introduced.

### 0.8 `modules=` exists because `B_ℓ` and a laptop need the same mechanism

The builder takes an optional list of modules restricting the columns. It is not a convenience: at
A1's `P = 26 634` a full `F` is 5.68 GB, so the only way to exercise the real model on the real θ
outside the cluster is to restrict it — and the restricted build is *by definition* the block
`B_ℓ`, which `test_block_is_a_slice_of_f` asserts against the slice of the unrestricted `F`. One
mechanism, two uses, no second code path.

### 0.9 `P_max = sqrt(B/24)` is a property of the implementation, not of the mathematics

`plan_exp_draft.md` §2.2 budgets regime A at `8P²` for the matrix and "about `3×` that with `eigh`",
hence `P_max ≈ sqrt(B/24)`, and §12 makes "an analysis job that inherits the training job's `--gpus`
line cannot allocate `F`" a named risk. Sizing the A1 job showed where the factor 3 comes from, and
that two thirds of it are avoidable:

| expression | extra `P × P` buffers |
|---|---|
| `matrix += rows.T @ rows` | 1 (the product, before the add) |
| `matrix = 0.5 * (matrix + matrix.T)` | 2 (the sum, then the scaling) |
| `matrix.addmm_(rows.T, rows)` | **0** |
| `symmetrize_(matrix)`, block-wise in place | **0** (one `block × block`) |

Both naive spellings were in the lot's first draft, so the "17 GB for A1" figure was about to be
reproduced by the code. Measured on CPU at `P = 12 030` (`P²` fp64 = 1.158 GB): the builder now
peaks at **`1.27 × P²`** end to end, while the single expression `0.5 * (M + M.T)` alone adds
**`+1.16 × P²`** on top of an already-built matrix. At A1's `P = 26 634` that is the difference
between **~6 GB** and **~17 GB**, i.e. between fitting the 10 GB MIG slice every training job uses
and not fitting it.

Consequence for the campaign, and for `plan_exp_draft.md` §12's risk: the rule is not "analysis jobs
need a big GPU". It is **hold one `P × P` on the device and put everything needing two on the
host** — which is what `fisher_ref/slurm/dense_reference_a1.sh` does, and why it asks for
`h100_1g.10gb:1` and `--mem=64G` rather than a 40 GB slice. The same host-side trap is closed in
the experiment script's `relative()`, which accumulates `‖A − B‖_F` over row blocks instead of
materialising `A − B` at the moment four references are live.

---

## 1. Modules

### 1.1 `fisher_ref/sources.py`

```python
SOURCES = ("type2", "mc", "empirical")
LOSSES  = ("cross_entropy", "mse")

@dataclass(frozen=True)
class OutputRoot:
    columns: Tensor          # (N, R, d_out), unscaled
    source: str; loss: str
    def column(self, index) -> Tensor        # (N, d_out), one backward's seed

def output_root(outputs, targets, *, source, loss="cross_entropy", k=1, generator=None) -> OutputRoot
def loss_kind(bench) -> str                  # read off bench.loss_fn, no second table
```

| source | cross-entropy | `R` |
|---|---|---|
| `type2` | `S_n[:, c] = sqrt(p_c)(e_c − p_n)`, `S S^T = diag(p) − p pᵀ` (`kfac_from_scratch_2507.05127.pdf`, cheat sheet §6) | `C` |
| `mc` | `K^{-1/2}(p_n − e_{ỹ_c})`, `ỹ ∼ p_n`; the `K^{-1/2}` is that paper's own errata item | `K` |
| `empirical` | `p_n − e_{y_n} = ∂ℓ_n/∂f_n` | `1` |

For MSE, `Λ_n = (2/D)·I` — the per-sample loss consistent with `nn.MSELoss`'s mean is
`ℓ_n = (1/D)Σ_d(f_d − y_d)²`, so the loss's own `1/D` sits **inside** `Λ`, and `F = GGN` by the
canonical link (Martens arXiv:1412.1193 §9). The `type2` root is then `sqrt(2/D)·e_c`, returned as
an expanded view rather than a materialised `(N, D, D)` tensor. `mc` + `mse` raises: the MC source
exists to test HF6 on the categorical models (`plan_exp_draft.md` §0.11).

Two conventions the module fixes, each pinned by a test: `ℓ_n` is the **per-sample** loss, never the
batch mean (so `bench.loss_fn`, a `reduction="mean"` criterion, is deliberately never called), and
the columns are **unscaled** — `plan_exp_draft.md` §2.1's `N^{-1/2}` is the driver's business, since
only the driver knows the total probe count.

### 1.2 `fisher_ref/capture.py`

```python
KINDS = ("linear", "conv", "norm")

@dataclass class CapturedLayer: name, module, kind, a (N,T,d_in), g (N,T,d_out)
class Capture:                       # context manager; every handle removed on exit
    def reset(self); def layer(name) -> CapturedLayer; def __iter__

def layer_kind(module) -> str | None
def capturable_modules(model) -> dict[str, nn.Module]
def coverage(model) -> tuple[list[str], list[str]]
def normalized_input(module, x) -> Tensor                  # x_hat, closed form
def pool_input(module, kind, x) / pool_output_grad(module, kind, g) -> Tensor
def per_sample_gradients(layer) -> dict[str, Tensor]       # {"weight": (N,*), "bias": (N,*)}
```

One forward fills `a`; every backward overwrites `g`, so the driver re-uses a single forward across
the root's columns (`retain_graph` on all but the last) — `plan_exp_draft.md` §2.1's cost model,
"each row of `U` costs one backward pass of one output vector, and the `N` probes of a batch share
it".

`per_sample_gradients` is one einsum `Σ_t g_{n,t} a_{n,t}^T` for `linear`/`conv` and an
**element-wise** `Σ_t g_t ⊙ x̂_t` / `Σ_t g_t` for `norm` — the same statement as `plan_lot5.md`
§0.1's "Hadamard-, not Kronecker-structured", here in its exact form rather than as a surrogate.
`a` carries **no** bias column: in the §0.5 layout the bias gradient is `g.sum(t)` in its own slice.

Two guards: a module called twice in one forward raises (its per-sample gradient would be a sum over
calls this package does not form — none of the eight built models does it, and the guard makes that
a checked property), and the forward hook returns early when `torch.is_grad_enabled()` is false, so
a `no_grad` probe forward — `assert_sample_independent`'s, or `registry`'s — captures nothing.

### 1.3 `fisher_ref/reference/dense.py`

```python
@dataclass(frozen=True) class ParamLayout: names, slices, shapes, P
    @classmethod def of(model, params=None); def block_slice(module_name) -> slice

@dataclass class DenseReference: matrix, layout, source, loss, n_probes, n_columns, probe_digest
    def block(module_name); def matvec(v); def trace(); def fro2(); def diag(); def metadata()

def build_dense_reference(model, inputs, targets, *, source, loss="cross_entropy", k=1,
                          batch_size=256, dtype=REFERENCE_DTYPE, device="cpu", generator=None,
                          modules=None, probe_digest=None, check_independence=True)
```

Per probe micro-batch: one forward under `reference_mode` + `Capture`; the root from `sources`; then
one `autograd.grad(outputs, [batch], grad_outputs=root.column(c))` per column, each filling a
`(N_mb, P)` row block whose `rows^T rows` is accumulated into `F`. `U` is never materialised (54 MB
per row block at `P = 26 634`, against 4.26 GB for the whole thing). The `1/N` is applied once at
the end rather than as an `N^{-1/2}` on every row, and the result is symmetrised — a Gram is
symmetric, so this only removes round-off.

`check_independence` runs `conventions.assert_sample_independent` once, on the first micro-batch:
lot 0 built the check and lot 1 is its first consumer, so "the reference is defined at all"
(`plan_exp_draft.md` §2.5) is a checked property rather than a comment. The driver also verifies,
on every column, that the capture filled **every** column of `U` — the §0.1 failure mode, caught
even if the hook design is later changed.

---

## 2. Tests — `tests/test_fisher_ref_lot1.py`

44 tests (2 of them `slow`, 1 skipped unless `curvlinops` is installed), all offline, CPU, fp64.

| Test | Statement |
|---|---|
| `test_type2_root_reconstructs_lambda_and_has_rank_c_minus_one` | `S S^T = diag(p) − p pᵀ`; `S^T sqrt(p) = 0` (§0.2) |
| `test_mc_root_needs_its_k_inverse_sqrt_to_converge_to_lambda` | with the factor, `< 0.02` from `Λ` at `K = 2·10⁴`; without it, `> 1` |
| `test_empirical_root_is_the_per_sample_loss_gradient` | `p − e_y` and `(2/D)(f − y)` against autograd, per sample |
| `test_mse_type2_root_is_the_canonical_link_lambda` | `Λ = (2/D) I` |
| `test_loss_kind_is_read_off_the_bench`, `test_mc_source_is_refused_for_mse` | the CE/MSE dispatch and its declared gap |
| `test_capture_reconstructs_parameter_gradients` | `Σ_n` per-sample gradients `==` autograd's, on `Linear` (unshared **and** shared over positions), `Conv2d`, `BatchNorm2d`-eval, `LayerNorm`, `GroupNorm` |
| `test_capture_matches_vmap_per_sample_gradients` | per-sample, not just summed, against `torch.func.vmap(grad(functional_call))` — the sum can be right while individual rows of `U` are wrong |
| **`test_backward_hook_pruning_regression`** | §0.1, both halves: the pruning still happens, and `Capture` is immune |
| `test_capture_rejects_a_reused_module`, `test_capture_rejects_batchnorm_in_train_mode` | the two guards |
| `test_coverage_lock` (+ the "every benchmark" companion) | the covered/uncovered lists per model, `uncovered ⊆ registry.unhooked_parameters` |
| `test_uncovered_parameters_raise_instead_of_zero_columns` | §0.7, on `vit_micro_cifar`'s `pos_embed` |
| **`test_t1_dense_fv_matches_matrix_free`** | T1: 20 random vectors, rel ≤ `1e-10`; measured `3.8e-16` |
| `test_t1_empirical_matches_the_per_sample_gradient_gram` | `Ê` against a `torch.func` per-sample-gradient Gram of the real loss |
| **`test_t2_type2_matches_monte_carlo`** | T2: `O(K^{-1/2})` over 5 seeds and `K ∈ {10², 10³, 10⁴}` |
| `test_type2_head_block_kernel_is_the_logit_shift_subspace` | §4's finding, locked (see below) |
| **`test_t6_norm_gradients_match_finite_differences`** | T6: central FD on `γ`/`β`, rel ≤ `1e-6`, on `LayerNorm`, `BatchNorm2d`-eval, `GroupNorm` |
| `test_t6_raw_input_fails_finite_differences` | non-vacuity: the *pre*-normalisation input fails the same check by `> 1e-2` |
| `test_block_is_a_slice_of_f`, `test_block_slices_are_contiguous_on_every_model` | §0.5, §0.8 |
| `test_streaming_batch_size_is_inert` | `batch_size ∈ {3, 5, 12}` agree to `1e-14` |
| `test_fp32_capture_gap_is_measured` | §0.3's number, reported not asserted |
| `test_reference_metadata_carries_the_invariants` | `metrics_version`, the precision block, the probe digest |
| `test_curvlinops_cross_check` | `importorskip`, §0.4 |
| `test_regime_a_model_end_to_end` | A1 built the way the bench builds it, restricted to the head and the two `LayerNorm`s |

**The T6 trap worth naming.** The finite difference must differentiate `⟨s_n.detach(), f(θ, x_n)⟩`
with the backprop vector **held fixed**. Differentiating a `v` recomputed from `p(θ)` adds a
`∂v/∂θ` term, and the resulting ~`1e-2` mismatch looks exactly like an `x̂` bug.

**T2's cost, for whoever wonders why it takes 4 s.** The MC source needs one backward pass per
sample, so `5 seeds × (10² + 10³ + 10⁴)` is 55 500 backward passes. Measured means over 5 seeds:
`0.091` at `K = 10²`, `0.025` at `10³`, `0.0066` at `10⁴` — ratios `3.6` and `3.8` against the
`sqrt(10) = 3.16` of a clean `K^{-1/2}`. The test asserts a bracket around that rate, not the point
estimate: five seeds do not pin a variance to two digits.

---

## 3. Exit criteria

1. The 297 pre-existing tests pass, unmodified — **done** (341 passing, 11 skipped, up from 297/8).
2. T1, T2, T6 and the supporting tests pass — **done** (44 new tests).
3. `ruff check` and `mypy` clean on `fisher_ref/` — **done**.
4. The `curvlinops` decision recorded here (§0.4) and in `plan_exp_draft.md` §2.4/§8 — **done**.
5. `CLAUDE.md` updated — **done**.
6. The end-to-end smoke on a real checkpoint, recorded in §4 — **done**.

---

## 4. Smoke run (exit criterion 6) — done

`fisher_ref/` end to end on the real A1 artifacts: `benchmarks/outputs/mlp_ln_mnist/diag/ckpt_0.5`
and the staged MNIST, columns restricted to the head and the two `LayerNorm`s (`P = 458`, seconds on
the laptop) so the full `P = 26 634` / 5.68 GB reference stays the cluster's business.

```text
precision: TF32 off = True | metrics_version: fisher_ref/0.1
theta: mlp_ln_mnist/diag ckpt 0.5  step=4290 epoch=9 progress=0.500
probes: 256 train / 256 val, digests 4069efafb5ab / 78ef8370ef16
coverage: 10 parameters covered, uncovered=[]

F      P=458  rows=2560  cols=10  0.02s   tr=2.5882e+00
E_hat  P=458  rows=256   cols=1   0.00s   tr=2.0261e+00
spectrum of F: lambda_max=4.312e-01  lambda_min=-1.25e-17  rank(1e-12)=393/458
source gap ||E_hat - F||_F / ||F||_F = 1.4159

  B[features.1] ( 64,  64)  tr=1.2516e+00  share of tr(F)=0.484
  B[features.4] ( 64,  64)  tr=2.6996e-01  share of tr(F)=0.104
  B[head      ] (330, 330)  tr=1.0666e+00  share of tr(F)=0.412
```

Three things worth recording from it.

- **The type-2 `F` is rank-deficient however many probes are used, and the deficiency is entirely in
  the head.** Per block: `features.1` and `features.4` are full rank (64/64), the head is **297 of
  330** — a deficiency of exactly `33 = d_in + 1`. The reason is structural, not numerical: adding a
  constant to every logit leaves the softmax unchanged, so every direction `W += 1_C v^T`,
  `b += c·1_C` is in `ker F`. Verified rather than argued — such a direction gives
  `‖B d‖ / (‖B‖‖d‖) = 9.1e-18` against `9.1e-2` for a random direction, and the same holds for
  `Ê`. **Consequence for lots 2 and 4:** every metric that goes through a damped inverse reads
  `λI` on a `(d_in + 1)`-dimensional subspace of any head block, at any `N`. That is not the
  rank-deficiency of `plan_exp_draft.md` §2.3 (which comes from `m < P` and goes away in regime A);
  it does not go away. `test_type2_head_block_kernel_is_the_logit_shift_subspace` locks it.
- **The source gap is not a correction term.** `‖Ê − F‖_F / ‖F‖_F = 1.42` on this block set: at
  50 % of an A1 trajectory the empirical Fisher is further from the true Fisher than the true Fisher
  is from zero. Q1 is not a rounding question, at least here.
- The `1/N` and the layout survive the real path: `F` is PSD to `-1.3e-17`, and the three blocks'
  traces sum to `tr(F)` by construction.

## 5. Phase 5 — the A1 run: done

Two files, both new, neither generated:

- **`fisher_ref/experiments/dense_reference_a1.py`**, in `experiments/`' own style — one question,
  no CLI, constants at the top (environment variables with defaults, so the sbatch script can set
  them), a printed fixed-width table, the result pasted back into its docstring.
- **`fisher_ref/slurm/dense_reference_a1.sh`** (+ a README for the directory), because
  `benchmarks/slurm/` is read-only for this campaign and its generator enumerates *training*
  benchmarks.

**The shape it asks for, and why it is not the 20-40 GB §2.2 assumed.** `h100_1g.10gb:1`,
`--mem=64G`, `--cpus-per-task=4`. §0.9 is the reason: with one `P × P` buffer on the device the
build peaks at ~6 GB of the 10 GB slice, and everything that needs two at once — the two references
side by side, the `eigvalsh` — is host-side. So the job uses the *same* slice as every campaign-1
training job: no scheduling risk, cheapest billing. `--cpus-per-task=4` because the long pole is
not the GPU but two host-side `26 634²` fp64 `eigvalsh` calls, and **`eigvalsh` does not thread**
(§6.1).

Two runs were needed, and §6 reports the second:

| job | `N` | `--time` | outcome |
|---|---:|---|---|
| 21077038 | 4 000 | 00:50:00 | **CANCELLED** at the limit, inside the per-block loop, having written nothing (§6.6) |
| 21082966 | 55 000 | 01:30:00 | **COMPLETED**, `01:06:33`, `MaxRSS 25.3 GB` |

---

## 6. The A1 reference (job 21082966, 2026-09-15)

`h100_1g.10gb`, **`N = 55 000`** train probes — the whole MNIST training pool — with the two
out-of-sample sets capped at 5 000 by the val split. `mlp_ln_mnist/diag/ckpt_0.5`, `P = 26 634`,
`550 000` rows of `U`. Summary and spectra in
`fisher_ref/outputs/mlp_ln_mnist/diag/seed0/0.5/`.

### 6.1 Cost, and the memory decision confirmed

```
type2  train  N=55 000  rows=550 000   147.4 s      empirical  N=55 000   15.0 s
type2  val    n= 5 000  rows= 50 000    13.5 s      type2 test  n= 5 000  13.5 s
type2  halves n=27 500  rows=275 000    73.6 s (x2)
eigvalsh(F) 1279.8 s    eigvalsh(E_hat) 1286.5 s    peak device 6.06 GB    MaxRSS 25.3 GB
```

**§0.9's decision holds at full scale: 6.06 GB peak on a 10 GB slice**, against the ~6.0 GB
predicted and the ~17 GB `plan_exp_draft.md` §2.2 budgets. That factor-3 was the three-buffer
implementation, not the matrix.

**The references are cheap and the spectra are not, by a factor 9.** A full type-2 `F` over the
*entire* MNIST training set costs **147 s**; one `26 634²` `eigvalsh` costs **1280 s**, and does not
thread (19.8 GFLOP/s at 1 thread, 18.6 at 8, 21.6 on 16 cluster cores). Build time is linear in `N`
as expected — 11.2 s at `N = 4 000`, 147.4 s at 55 000, a ratio of 13.2 against 13.75 in `N`. So
**`N` is nearly free and the spectrum is the budget**: `--mem` can drop to 32G (MaxRSS 25.3 GB) and
`--time` is `(4/3)P³ / 2·10¹⁰` per decomposition plus a few minutes.

### 6.2 The gaps, and the floor they must clear

| quantity | `N = 4 000` (job 21077038) | **`N = 55 000`** |
|---|---:|---:|
| noise floor, halves of `N/2` | 0.8721 | **0.2986** |
| ⇒ `σ_N`, error of one `N`-probe estimate | 0.436 | **0.1493** |
| source gap `‖Ê − F‖/‖F‖` (Q1) | 0.7597 | **0.3751** |
| train/val gap (HF1) | 0.8215 | **0.7121** |
| val/test gap | — | **0.7111** |
| train/test gap | — | **1.0222** |

**Q1 is now interpretable, and it was not before.** The source gap sits at **2.51 σ_N** (against
1.74 at `N = 4 000`): the empirical Fisher differs from the true Fisher by two and a half times the
estimator's own noise, at 50 % of an A1 trajectory. Note how much of the `N = 4 000` value was
noise — 0.76 → 0.38 — which is exactly why §6.2 of the first run refused to read it.

**But the `N^{-1/2}` extrapolation that recommendation rested on is measured to be optimistic.**
`σ_N` fell from 0.436 to 0.1493, a factor **2.92**, where `√(55000/4000) = 3.71` was expected. The
floor is still falling, but more slowly than `N^{-1/2}`, so the "`σ_N = 0.1` needs `N ≈ 76 000`"
arithmetic in the first run's §6.2 was wrong in the safe direction — 55 000 probes, the whole pool,
gives 0.149 and MNIST has nothing more to give. `d(F_{N'}, F_N)` against `N'` (§3.4) is what would
say whether the residual is a genuine systematic floor or a heavier-tailed variance.

**HF1 is still not settled**, and it is the out-of-sample side that blocks it: the val and test sets
are 5 000 probes each, where `σ ≈ 0.50`. The train/val gap of 0.712 sits at roughly 1.4× its null.

### 6.3 The val/test check: interchangeable — and a tension the instrumentation cannot resolve

§6.5's falsification check, run for the first time. Both sets at the same `n = 5 000`, so the null
for two independent estimates of the same matrix is `√2 · σ_5000 ≈ 0.70`:

* **`‖F_test − F_val‖ / ‖F_val‖ = 0.7111`, i.e. `1.02 ×` that null.** Val and test are
  statistically **indistinguishable**. On this evidence the distributional worry does not
  materialise, and pooling them into a 15 000-probe out-of-sample set is justified — which is what
  §6.5's table priced at an HF1 null of 0.25 instead of 0.41.

* **But `train/test = 1.0222` is `1.44 ×` `train/val = 0.7121`**, and that does not fit. If val and
  test were interchangeable, both should stand at the same distance from the train reference.

The tension is **not resolvable from what was recorded**, and that is an instrumentation failure,
not a finding: the three gaps use two different denominators (`‖F_train‖` for two of them,
`‖F_val‖` for the third) and the script stores `‖·‖_F` only for the train-side `F` and `Ê`. A
`‖F_val‖` about 1.6× `‖F_train‖` would reconcile all three, but a 5 000-probe estimate's norm
should be inflated by only ~12 %, not 60 %.

**So HF1 stays open, and lot 2's first job on it is to close this.** One line in the builder —
record `trace` and `‖·‖_F` for *every* reference built — plus reporting all three gaps against a
common denominator. Until then, do **not** quote the val/test result as licence to pool.

### 6.4 Spectra and the per-layer blocks

| | `λ_max` | `rank(1e-12)` | `κ` |
|---|---:|---:|---:|
| `F` (type-2) | `9.2124e-01` | **21 829 / 26 634** | `9.99e+11` |
| `Ê` (empirical) | `7.8299e-01` | **18 342 / 26 634** | `9.98e+11` |
| | `tr(F) = 16.68` | `tr(Ê) = 11.95` | |

At `N = 55 000 > P`, `Ê` is no longer rank-limited by the probe count (it was 3 511 at `N = 4 000`,
bounded by `N`), so its remaining deficiency of **8 292** is structural. It is *more* rank-deficient
than `F` — 2.5× the deficiency — which is a sharper version of Kunstner, Balles & Hennig's point
(arXiv:1905.12558) than the first run could show.

The per-block table, the row the first run never reached:

| block | `P` | trace share | rank | deficiency |
|---|---:|---:|---:|---:|
| `features.0` `Linear(784→32)` | 25 120 | **0.627** | 20 541 | **4 579** |
| `features.3` `Linear(32→32)` | 1 056 | 0.208 | 1 022 | 34 |
| `features.1` `LayerNorm(32)` | 64 | 0.083 | 64 | 0 |
| `head` `Linear(32→10)` | 330 | 0.066 | 297 | **33** |
| `features.4` `LayerNorm(32)` | 64 | 0.017 | 64 | 0 |

Every deficiency is structural, and two were predicted before the run:

* **the head's is exactly `33 = d_in + 1`** — §4's logit-shift kernel, confirmed at full scale on
  the real model: adding a constant to every logit leaves the softmax unchanged, so
  `W += 1_C v^T, b += c·1_C` is annihilated for every `v`;
* **`features.0`'s 4 579 ≈ 32 × 143** — MNIST's dead pixels, one killed direction per output unit
  per dead input coordinate. `_eigh_utils.py` records that layer's input factor as rank 646/785
  with 136 exactly-zero eigenvalues (130 identically-zero pixels); 143 is that plus the
  near-zero border, so the agreement is good;
* `features.3`'s 34 is the same mechanism through the preceding `LayerNorm`, whose output is
  mean-zero and unit-variance per sample, i.e. two constraints on its 32 dimensions;
* the block deficiencies sum to 4 646 against `F`'s own 4 805, the 159 difference being the
  inter-block directions.

`features.0` also carries **63 %** of `tr(F)`: A1's curvature is overwhelmingly in its first layer,
which is worth knowing before reading any per-layer-type conclusion drawn on this model.

### 6.5 A third probe split, and why it is a *check* before it is a sample size

§6.2's floor is dominated by the **out-of-sample side**: the HF1 null is
`sqrt(σ_train² + σ_held²)`, so with the train side at 55 000 (`σ = 0.149`) and `val` stuck at 5 000
(`σ ≈ 0.50`) the null stays near 0.52. `val` is the binding constraint and it cannot be widened:
`val_size = 5000` is what the *training* runs partitioned on, so changing it would retroactively
redefine which images those weights saw — the same "it would replace campaign 1 rather than augment
it" argument as `--checkpoint-optimizer-state`.

What *is* available is the dataset's own test set — 10 000 further MNIST images that
`build_probe_set` never touched (lot 0 used `train=True` unconditionally), and which the training
harness loads only to report a final number. Two facts decide whether it may be used:

* **Epistemically, `val` and `test` are in the same position here.** Checked in the harness: the
  validation split drives **nothing** — no early stopping, no best-checkpoint selection, no
  `ReduceLROnPlateau`; `best_val_acc` appears only in the report writer. And the hyperparameters
  are fixed a priori from the AdaFisher paper rather than tuned. Neither split influenced the
  weights. (If that ever changes, `val` must go back to being its own category.)
* **Distributionally, they may not be.** `val` is a random slice of the training set; MNIST's test
  set is a separate collection, with disjoint writers in the original NIST split.

So `split="test"` is added to `fisher_ref/probes.py` — a reader change, nothing in `benchmarks/` —
but the two are **not pooled**. The job builds `F_val` and `F_test` at the *same* `n` and reports
their gap; §6.3 is that measurement, and it says "interchangeable" while a second gap says
otherwise, so the pooling below stays **pending**.

| out-of-sample side | `n` | `σ` | HF1 null with train at 55 000 |
|---|---:|---:|---:|
| `val` (today) | 5 000 | ~0.50 | ~0.52 |
| `test` | 10 000 | ~0.35 | ~0.38 |
| `val + test`, **only once §6.3 is resolved** | 15 000 | ~0.29 | ~0.32 |

### 6.6 What the two runs changed in the code

1. **The summary is written incrementally** (`checkpoint_summary()` after every stage). Job
   21077038 computed all three gaps and both spectra and saved **none** of them, because the single
   write was at the end — the same failure `benchmarks/common/runner.py` already fixed for a
   grouped training job ("rewrite the report after every completed arm").
2. **The per-block loop runs cheapest block first**, and `A1_BLOCK_SPECTRA=0` skips it. `features.0`
   is `25 120²`, 83 % of a full-`P` decomposition on its own — it is what the first run's time limit
   hit, and the four cheap blocks behind it were lost for nothing.
3. `--cpus-per-task` 16 → 4, `--time` 00:50:00 → 01:30:00, both on the measured basis (§6.1).
4. **The seed is filtered explicitly.** `discover_runs(model, arm)` matched five runs once the
   extra-seed campaign landed (seed 0 under `outputs/<group>/<model>/`, seeds 1-4 under
   `outputs/seeds/`), and the script took `runs[0]` — correct only by the accident that
   `"mnist" < "seeds"`. It now filters on `A1_SEED` and refuses to guess.
5. **`fisher_ref/slurm/logs/*.out` and `fisher_ref/outputs/**/*.pt` are gitignored**, as their
   `benchmarks/` counterparts already were.

### 6.7 Still open

* **HF1**, blocked on §6.3's denominator problem, then on whether val and test may be pooled.
* **The noise floor as an interval**: §3.4 asks for 20 random partitions and a 95 % band; one split
  gives a point estimate, and §6.2's sub-`N^{-1/2}` decay is the kind of thing a band would settle.
* **`Ê`'s structural 8 292-dimensional kernel** — measured here, unexplained.
