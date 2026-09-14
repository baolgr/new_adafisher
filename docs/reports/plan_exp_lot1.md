# Lot 1 — the exact references: capture, sources, dense `F` / `Ê` / `B_ℓ`

*Implementation plan and completion record for lot 1 of `plan_exp_draft.md` §9. In the repository's
own numbering this is lot 11; it is named after the experimental plan, like `plan_exp_lot0.md`.*

**Goal.** The first curvature matrices of the campaign: given a model, a set of probes and a
*source* of backprop vectors, produce the exact `F = U^T U`, the exact empirical `Ê`, and the exact
per-layer blocks `B_ℓ`, in fp64, in regime A.

**Non-goal.** No approximation (lot 2's `approx/`), no metric (lot 2's `metrics/`), no runner, no
figure. Nothing in `src/adafisher_modes/` and nothing in `benchmarks/` changes — `fisher_ref/` stays
the reader lot 0 made it.

**Status: phases 1-4 done** (code, tests, the real-checkpoint smoke of §4). The full A1 run at
`N = 4000` is §5, deliberately left to the cluster; §6 is therefore empty, as `plan_lot8.md`'s was.

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

## 5. Phase 5 — the A1 run: written, not yet submitted

Two files, both new, neither generated:

- **`fisher_ref/experiments/dense_reference_a1.py`**, in `experiments/`' own style — one question,
  no CLI, constants at the top (here environment variables with defaults, so the sbatch script can
  set them), a printed fixed-width table, the result to be pasted back into its docstring. It
  builds, at `N = 4000` type-2 probes (`N(C−1) = 36 000 ≥ P = 26 634`, so `F` is not rank-limited by
  the probe count): `F` and `Ê` on the train probes and their spectra; `F` on the val probes, for
  HF1; and a two-way split of the train probes, for §3.4's **noise floor** — the quantity that says
  whether any later difference is interpretable at all. Per-block trace shares and ranks come free
  from the slices. Only `reference_summary.json` and `spectrum.pt` are written; `F` is 5.68 GB and
  recomputable in under a minute of GPU time (`plan_exp_draft.md` §12).
- **`fisher_ref/slurm/dense_reference_a1.sh`** (+ a README for the directory), because
  `benchmarks/slurm/` is read-only for this campaign and its generator enumerates *training*
  benchmarks.

**The shape it asks for, and why it is not the 20-40 GB §2.2 assumed.** `h100_1g.10gb:1`,
`--mem=64G`, `--cpus-per-task=16`. §0.9 is the reason: with one `P × P` buffer on the device the
build peaks at ~6 GB of the 10 GB slice, and everything that needs two at once — the two references
side by side, the `eigvalsh` — is host-side, where memory is cheap. So the job uses the *same*
slice as every campaign-1 training job, which also means no scheduling risk and the cheapest
billing. `--cpus-per-task=16` because the long pole is not the GPU at all: it is two host-side
`26 634²` fp64 `eigvalsh` calls at `≈2.5·10¹³` flop each. `--time=00:50:00` is an **estimate**, and
labelled as one — this job has never run, which is exactly the situation `benchmarks/slurm/
README.md`'s "every `--time` is measured" rule exists to end; the first COMPLETED run replaces it.

Prerequisites verified present on `rorqual:/home/blgr/new_adafisher`: `dataset/MNIST` and
`benchmarks/outputs/mlp_ln_mnist/diag/ckpt_0.5.pt`. The script was dry-run locally end to end on
that same real checkpoint, restricted to three modules (`A1_MODULES`), so only the scale is
untested.

Its numbers become §6 below and turn `plan_exp_draft.md` §13's
`[ESTIMATE, to be measured at lot 1]` into `[MEASURED]`.

## 6. The A1 reference — first run (job 21077038, 2026-09-14)

`h100_1g.10gb`, `N = 4000`, `mlp_ln_mnist/diag/ckpt_0.5`, `P = 26 634`. The job was **CANCELLED at
its 00:50:00 limit inside the per-block loop**, having produced everything else; the per-block
ranks are the only missing row and the re-run at `--time=01:30:00` fills them. Nothing was written
to disk, because the single write was at the end — fixed (§6.4).

### 6.1 What it cost, and the memory decision

```
type2      P=26634  rows=40000   11.2s   5.67 GB   peak device 6.10 GB
empirical  P=26634  rows=4000     1.2s   5.67 GB   peak device 6.10 GB
type2 val  P=26634  rows=40000   10.9s             peak device 6.10 GB
type2 half P=26634  rows=20000    5.5s  (x2)       peak device 6.11 GB
eigvalsh(F) 1155.6s     eigvalsh(E_hat) 1155.7s
```

**§0.9's decision is confirmed on the real thing: peak device 6.10 GB against the ~6.0 GB
predicted, on a 10 GB slice.** `plan_exp_draft.md` §2.2's "an analysis job that inherits the
training job's `--gpus` line cannot form A1's `F` at all" is now measured to be false — it was a
property of the three-buffer implementation, not of the matrix.

**The references are cheap; the spectra are not.** A full type-2 `F` at `N = 4000` takes **11 s**,
while one `26 634²` `eigvalsh` takes **1156 s** — and that is not a misconfiguration: torch's
`eigvalsh` does not thread (measured 19.8 GFLOP/s at 1 thread, 18.6 at 8 on a laptop; the cluster
ran at 21.6 GFLOP/s on 16 cores). `--cpus-per-task` was therefore dropped from 16 to 4, and the
job's `--time` is set by `(4/3)P³ / 2·10¹⁰` seconds per decomposition.

### 6.2 The headline: at `N = 4000` this reference is noise-dominated

| quantity | measured |
|---|---|
| source gap `‖Ê − F‖_F / ‖F‖_F` (Q1) | **0.7597** |
| train/val gap `‖F_val − F_train‖_F / ‖F_train‖_F` (HF1) | **0.8215** |
| noise floor `‖F^(1) − F^(2)‖_F / ‖F^(2)‖_F`, halves of `N/2 = 2000` (§3.4) | **0.8721** |

The floor is not directly comparable to the other two and must be rescaled. `F^(1) − F^(2)` is the
difference of two *independent* `N/2` estimates, so it is `√2·σ_{N/2} = 2·σ_N`, giving

* `σ_N ≈ **0.436**` — the error of a single `N = 4000` estimate against `F_∞`;
* `√2·σ_N ≈ **0.617**` — what two independent `N = 4000` estimates differ by **under the null**.

So the source gap sits at `1.74 σ_N` and the train/val gap at `1.33 ×` its own null. **Both headline
quantities of the campaign are the same order as the estimator's own noise at this `N`.** §3.4's
reading rule ("any difference below the floor is not interpretable") bites on the very first run,
and it bites on Q1 and HF1 themselves, not on some downstream comparison of approximations.

The fix is cheap and the run itself shows why: `N` was chosen in §2.2 for **rank**
(`N(C−1) = 36 000 ≥ P`), which is a far weaker requirement than accuracy. Since `σ_N ∝ N^{-1/2}`,
reaching `σ_N = 0.1` needs `N ≈ 76 000` — and a build is 11 s, so that is ~3.5 min of GPU per
reference, not a new regime. **The campaign's `N` should be set by the noise floor, not by the rank
condition**, and every gap must be reported with its floor.

Two caveats to carry with these numbers. This is **one** split, where §3.4 asks for 20 random
partitions and a 95 % interval — that is lot 2's `metrics/noise_floor.py`, and the point estimate
above may move. And a *relative Frobenius* error of 0.44 does not mean the leading structure is
noise: the Frobenius norm weights the bulk, so the top eigenspace may well be far better determined
than this single number suggests (M6 is the metric that would say).

### 6.3 Spectra

| | `λ_max` | `λ_min` | `rank(1e-12)` |
|---|---|---|---|
| `F` (type-2) | `9.1613e-01` | `-2.61e-15` | **18 564 / 26 634** |
| `Ê` (empirical) | `1.1169e+00` | `-5.53e-15` | **3 511 / 26 634** |

* **`Ê` is rank-limited by construction**: it is a sum of `N = 4000` outer products, so its rank
  cannot exceed 4 000 — here 3 511. Any metric through a damped inverse of `Ê` therefore reads `λI`
  on **23 123 of 26 634 directions**. That is Kunstner, Balles & Hennig's (arXiv:1905.12558) point
  in this repository's own numbers, and it is a reason to prefer `F` as the reference wherever the
  campaign has a choice.
* **`F`'s deficiency of 8 070 is larger than the head's 33** (§4's logit-shift kernel). Part of it
  is already documented elsewhere in this repository: `mlp_ln_mnist`'s first `Linear` has an input
  factor of rank 646/785 with 136 exactly-zero eigenvalues, because 130 of MNIST's 784 pixels are
  identically zero (`_eigh_utils.py`'s docstring — the fact that crashed cuSOLVER in campaign 1).
  A dead input coordinate kills one column of that layer's block per output unit, i.e. `32 × 136`
  ≈ 4 352 directions, plus the head's 33. That accounts for roughly half; the rest is **not
  explained here** and is a question for lot 2, not a claim.

### 6.4 A third probe split, and why it is a *check* before it is a sample size

§6.2's floor is dominated by the **out-of-sample side**: the HF1 null is
`sqrt(σ_train² + σ_held²)`, so with the train side raised to 55 000 (`σ = 0.118`) and `val` stuck at
5 000 (`σ = 0.390`) the null only falls from 0.617 to **0.407**. `val` is the binding constraint,
and it cannot be widened: `val_size = 5000` is what the *training* runs partitioned on, so changing
it would retroactively redefine which images those weights saw — the same "it would replace
campaign 1 rather than augment it" argument as `--checkpoint-optimizer-state`.

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
but the two are **not pooled**. The A1 job now builds `F_val` and `F_test` at the *same* `n` and
reports `‖F_test − F_val‖ / ‖F_val‖`. If that sits inside their combined floor, they are
interchangeable and a later lot may pool them, taking the HF1 null from 0.407 to **0.254** (a
factor 1.6, and 2.4 against the present 0.617). If it does not, pooling would have turned a
distribution shift into what reads as HF1 signal — and the check is what says which, at the cost of
one extra 11 s build.

| out-of-sample side | `n` | `σ` | HF1 null with train at 55 000 |
|---|---:|---:|---:|
| `val` (today) | 5 000 | 0.390 | 0.407 |
| `test` | 10 000 | 0.276 | 0.301 |
| `val + test`, **only if the check passes** | 15 000 | 0.225 | 0.254 |

These `σ` extrapolate one measured point (`σ_4000 = 0.436`) as `N^{-1/2}`; §3.4's `d(F_{N'}, F_N)`
curve is what would verify the exponent.

### 6.5 What the run changed in the code

1. **The summary is now written incrementally** (`checkpoint_summary()` after every stage). The run
   computed all three gaps and both spectra and saved **none** of them, because the single write
   was at the end — the same failure `benchmarks/common/runner.py` already fixed for a grouped
   training job ("rewrite the report after every completed arm").
2. **The per-block loop runs cheapest block first**, and `A1_BLOCK_SPECTRA=0` skips it. On A1 the
   widest block is `25 120²`, i.e. 83 % of a full-`P` decomposition on its own — it is what the
   time limit actually hit, and the four cheap blocks behind it were lost for nothing.
3. `--cpus-per-task` 16 → 4 (§6.1), `--time` 00:50:00 → 01:30:00 on the measured basis.
