# Lot 5 Implementation Plan — Normalisation layers (`BatchNorm2d`, `LayerNorm`) for
# `kfac`/`ekfac`/`tkfac`/`tekfac`

**Status:** draft, awaiting implementation in this same session. No lot-5 code written yet.
**Parent plan:** `docs/reports/plan.md` §8, lot 5 row: *"Normalisation layers (option (c), §5.2) —
loss non-regression vs. `diag` on a small CNN."* This document refines that one-line entry to
implementation-ready precision, the way `plan_lot2.md`/`plan_lot3.md`/`plan_lot4.md` did for
lots 2-4. It does not revisit lots 1-4 (closed, 73 tests passing) or lots 6-7 (SUA, equal-wall-clock
bench).

**Source note.** Every equation/proposition number below was re-read directly from
`docs/papers/adafisher_2405.16397.pdf` for this lot: Proposition 3.1 (statement, p. 4), its proof —
restated as Proposition A.1 (Appendix A.2, p. 21) — and Appendix A.3 ("Computation of KFs", p. 25),
which confirms normalisation layers are handled by Proposition 3.1 directly, not by the generic
Proposition 3.2 diagonal recipe used for `Linear`/`Conv2d`. `plan.md` §1.5/§5.2's cheat-sheet-level
summary ("`H|_ν` collapses to a scalar... all matrix structure is carried by `S`") is confirmed
correct as an *operational* description of the code, but the exact route from Proposition 3.1's
literal statement to that description is not given anywhere in `plan.md` — deriving it precisely is
this lot's main content (§0.1-§0.2).

Exit criteria (`plan.md` §8, lot 5 row, unpacked against §5.2/§6.1, the same way prior lots
unpacked theirs):

1. `compute_h_full`/`compute_s_full` (hence `kfac`/`ekfac`/`tkfac`/`tekfac`) support `BatchNorm2d`
   and `LayerNorm` with a 1-tuple `normalized_shape` — the only case exercised anywhere in this
   codebase (`conftest.py`'s `TinyMultiLayerNet` uses `nn.LayerNorm(8)` on a `(N, 8)` tensor).
2. On a toy `BatchNorm2d(3)` where the pooled exact Fisher block `F` is computable by an
   *independent* oracle (not reusing this lot's own extraction code):
   - `‖F − F̃_EKFAC‖_F ≤ ‖F − F̃_KFAC‖_F` (EKFAC Thm 2/3) — §6.1 assertion 1, replayed.
   - `‖F − F̃_TEKFAC‖_F ≤ ‖F − F̃_TKFAC‖_F` (TEKFAC Thm 3.1) — §6.1 assertion 2, replayed.
   - `tr(F̃_TKFAC) = tr(F)` (TKFAC Thm 4.1/Lemma 4.1) — §6.1 assertion 3, replayed.
   - `Q_A`, `Q_B`, `Q_Φ`, `Q_Ψ` orthogonal — §6.1 assertion 4, replayed.
3. A `LayerNorm` with a non-1-tuple `normalized_shape` raises a typed `NotImplementedError`, not a
   silently wrong factor.
4. No regression: the full lots 1-4 suite (73 tests) still passes unmodified.
5. A real `AdaFisherMulti` (hooks, not direct calls) runs a few `.step()`s on
   `conftest.py`'s `TinyMultiLayerNet` (the one net in this codebase already exercising all four
   `SUPPORTED_MODULES`, including both `BatchNorm2d` and `LayerNorm`) without error, for all four
   modes, with finite parameters that have moved — the lot-4 smoke-test pattern, extended with the
   literal exit-criterion wording from `plan.md` §8: a loss-non-regression check against `diag` from
   identical initial weights (§2.4).

---

## 0. Design decisions this lot must resolve before code exists

### 0.1 What Proposition 3.1 actually specifies, re-derived from its own proof

Proposition 3.1 (`adafisher_2405.16397.pdf` p. 4, restated with proof as Proposition A.1, p. 21)
states, for a normalisation layer with scale/shift parameters `(ν_i, β_i) ∈ R^{C_i}`:

```
H_{i-1}|_{ν_i} = (1/|T_i|) Σ_{x∈T_i} h_{i-1,x} h_{i-1,x}^T ∈ R^{C_i×C_i}    (exact, full-rank matrix)
H_{i-1}|_{β_i} = 11^T ∈ R^{C_i×C_i}                                        (exact, fixed constant)
S_i            = (1/|T_i|) Σ_{x∈T_i} s_{i,x} s_{i,x}^T ∈ R^{C_i×C_i}       (exact, full-rank matrix)
```

`h_{i-1}, s_i ∈ R^{C_i×|T_i|}` are the pre-normalisation activations and the gradient w.r.t. the
layer's own output; `T_i` is the set of dimensions normalisation statistics are computed over
(batch × spatial for `BatchNorm2d`; batch for the `(N, C)`-shaped `LayerNorm` usage this codebase
exercises). This `S_i` is exactly the "square-then-sum" full matrix `plan.md` §5.2 already asks the
four new modes to build (as opposed to the code's own diagonal path, which does "sum-then-square" —
`plan.md` §5.2's own flagged deviation, unaffected by this lot).

**The combination rule is Hadamard, not Kronecker — the fact `plan.md`'s cheat sheet elides.** The
proof (p. 21-22) derives, for `∇_{ν_i}J(θ) = Σ_x h_{i-1,x} ⊙ s_{i,x} ∈ R^{C_i}` (element-wise, since
`ν_i` performs no cross-channel mixing — `h_i = ν_i ⊙ h_{i-1} + β_i`):

```
E[∇_{ν_i}J ∇_{ν_i}J^T] ≈ E[Σ_x (h_{i-1,x}h_{i-1,x}^T) ⊙ (s_{i,x}s_{i,x}^T)]     (K-FAC independence
                        = H_{i-1}|_{ν_i} ⊙ S_i                                  assumption, p. 21)
```

(the paper's own "⊗" symbol here denotes the **element-wise (Hadamard) product** of two `C_i×C_i`
matrices, not the Kronecker product — the only reading under which the algebra is dimensionally
consistent, since `∇_{ν_i}J` is a `C_i`-vector, not a `C_i²`-vector). Symmetrically,
`FIM_{β_i} = H_{i-1}|_{β_i} ⊙ S_i = 11^T ⊙ S_i = S_i` exactly (Hadamard product with the all-ones
matrix is the identity operation). The proof's closing line, *"Cross-terms between `ν_i` and `β_i`
are excluded under the diagonal block assumption,"* means the full `2C_i×2C_i` FIM over `(ν_i,β_i)`
jointly is **block-diagonal**: `diag(H_{i-1}|_{ν_i}⊙S_i, S_i)`, no `ν`-`β` coupling.

This is a fundamentally different algebraic structure from `Linear`/`Conv2d`'s `F_l = A⊗B` (genuine
Kronecker product, because a weight matrix *does* mix inputs into outputs via matrix multiplication,
`kfac_1503.05671.pdf` §3.1 Eq. 2). Fitting it into this codebase's existing `kron(A,B)`-shaped
`FisherApproximation` machinery (§0.2) therefore needs one genuine, cited design decision — not
"no invention" as `plan.md` §5.2 characterises it, but a small, principled one, spelled out here so
it is not silently smuggled into `factors.py`.

### 0.2 Fitting the Hadamard structure into the existing Kronecker machinery

The whole point of §2.1 of `plan.md` (a single shared ABC across all five modes) is that
`kfac.py`/`ekfac.py`/`tkfac.py`/`tekfac.py` only ever consume `compute_h_full`/`compute_s_full`'s
output through the generic `(A, B) ↦ kron(B̃,Ã)`-style machinery (`plan_lot2.md` §0.4's `rvec`
identity `M ↦ B⁻¹MA⁻¹`). Reusing that machinery unmodified for a **fourth** layer type (after
`Linear`, and `Conv2d` in lot 4) is what `plan.md` §5.2 calls "no invention" — but it is only
possible if the normalisation layer's *exact* Hadamard-structured FIM (§0.1) is first replaced by
something of the `A⊗B` shape. The bridge is the same Hadamard-identity already used above:

> **If `A` is taken to be *diagonal*, `diag(a_ν, a_β)`, then `kron(diag(a_ν,a_β), S)` reduces
> exactly to `block_diag(a_ν·S, a_β·S)`** (direct expansion: `kron(D,S)` for diagonal `D` places
> `D_{ii}·S` on the `i`-th diagonal block and zero elsewhere). Setting `a_β := 1` reproduces
> `FIM_{β_i} = 1·S_i = S_i` **exactly** (matching §0.1's exact result, no approximation at all), and
> setting `a_ν` to some scalar reproduces `FIM_{ν_i} ≈ a_ν·S_i` in place of the exact
> `H_{i-1}|_{ν_i}⊙S_i` — the one genuine approximation this lot introduces.

`a_ν` is chosen as the **Frobenius-optimal, `S`-independent** scalar multiple of the all-ones matrix
`J` approximating the exact `H_{i-1}|_{ν_i}`:

```
a_ν := argmin_a ‖H_{i-1}|_{ν_i} − a·J‖_F² = ⟨H_{i-1}|_{ν_i}, J⟩_F / ‖J‖_F²
     = (1/C_i²) Σ_{c,c'} H_{i-1}|_{ν_i}[c,c'] = (1/C_i²)(1/|T_i|) Σ_x (Σ_c h_{i-1,c,x})²
     = (1/|T_i|) Σ_x z_x²,   z_x := (1/C_i) Σ_c h_{i-1,c,x}   (per-position channel-mean pre-activation)
```

`S`-independence is deliberate (not merely convenient): minimising `‖(H_{i-1}|_{ν_i}−aJ)⊙S_i‖_F`
directly would make `a` depend on `S_i` (a weighted least-squares problem with weights `S_i[c,c']²`),
which would destroy the entire premise of `A` and `B` being independent, reusable factors —
`refresh()`'s inverses/eigenbases for `A` and `B` are computed and cached completely separately in
every existing mode, and a `S`-coupled `a_ν` would break that structurally. Minimising the
`J`-projection error alone, ignoring `S_i` entirely, is the natural analogue of choosing a basis- or
data-independent structural simplification, and is the *only* choice consistent with keeping `A` a
genuine, standalone `2×2` factor.

**Why the off-diagonal is *not* forced to zero.** Building `A` via the *same* "augment with a
constant-1 column, then reduce `X.t()@X/N`" pattern every other layer type already uses (§1.1) gives

```
A = h_bar^T h_bar / |T_i|,   h_bar := [z_x, 1]_{x∈T_i}  (a (|T_i|, 2) matrix)
  = [[ mean_x(z_x²),  mean_x(z_x) ],
     [ mean_x(z_x),   1           ]]
```

— `A[0,0] = a_ν` and `A[1,1] = 1 = a_β` land *exactly* on the values derived above (no extra
invention needed to get those two entries right), but `A[0,1] = mean_x(z_x)` is generally **nonzero**,
unlike Proposition A.1's own idealised "cross-terms excluded" assumption. Two ways to remove it were
considered and rejected:

1. *Special-case `compute_h_full`'s normalisation branch to hard-code `A = diag(a_ν, 1)`* while
   `augment_input` (needed by `ekfac`/`tkfac`/`tekfac` for the raw per-sample batch, §0.3) still
   naturally reduces to the non-zero-off-diagonal `A` above. This was rejected: `compute_h_full` is
   *defined*, for every other layer type, as `augment_*_input(...).t() @ augment_*_input(...) / N`
   (`_h_full_linear`, `_h_full_conv2d`) — `kfac.py`'s own `_A[module]` (built via `compute_h_full`)
   would then silently disagree with `ekfac.py`'s own `_A[module]` (built via `h_bar.t()@h_bar/N` on
   `augment_input`'s output) for the *same* layer and the *same* batch, breaking `CLAUDE.md`'s
   opening invariant that "all five modes share the same exact per-layer Fisher block ... and differ
   only in what they do with it."
2. *Centre `z_x` before augmenting* (`z̃_x := z_x − mean_x(z_x)`, giving `A[0,1] ≡ 0` exactly, at the
   cost of `A[0,0]` becoming `Var(z)` instead of `mean(z²)`). This was rejected for consistency:
   `augment_linear_input`/`augment_conv2d_input` never centre their own activations before appending
   the bias column either — `Linear`'s existing `A[0:d_in, d_in] = mean(h)` is a real, uncentred,
   generally-nonzero empirical correlation between the activation and the bias direction, and no
   test or design decision in lots 1-4 treats that as a defect. Centring only the normalisation-layer
   branch would be a new, layer-type-specific special case introduced purely to chase a closer match
   to Proposition A.1's idealisation, at the cost of inconsistency with every other layer type's own
   bias treatment.

**Resolved:** use the plain, uncentred `h_bar = [z_x, 1]` construction, accept the resulting small
extra `ν`-`β` coupling term as a documented, honest generalisation of Proposition A.1's block-diagonal
idealisation (this codebase estimates a correlation the paper assumes to be exactly zero, rather than
discarding one it assumes to be nonzero) — consistent with, not a departure from, how `Linear`'s own
bias column is already treated. This term vanishes automatically whenever the layer's channel-mean
pre-activation happens to be zero-mean over `T_i` (a common, though not universal, regime).

### 0.3 Pooling convention: `T_i`, channel dim, per layer type — reusing the code's own axis choices

`T_i` and "channel" must be read identically to how the *existing*, untouched diagonal path already
reads them for each layer type (a structural fact about the layer, not a modelling choice this lot is
free to redefine):

- **`BatchNorm2d`.** `h`/`s` shaped `(N, C, H, W)`; channel = dim 1 (matches `_h_batchnorm2d`'s own
  `tsum(h, dim=(0,2,3))`). `T_i = N·H·W`. Pool via `h.permute(0,2,3,1).reshape(-1, C)`.
- **`LayerNorm`.** The existing ported code has a latent asymmetry here, noted for the record but not
  fixed (`CLAUDE.md`: change only what's requested): `_h_layernorm`'s `dim_to_reduce` excludes dim 1,
  while `_s_layernorm` excludes the *last* dim — these coincide only when `h.ndim == 2`, which is the
  only case this codebase's own `TinyMultiLayerNet` (`nn.LayerNorm(8)` on `(N, 8)`) or any existing
  test ever exercises. This lot does not need to resolve that latent 3D-input ambiguity: it scopes
  `LayerNorm` support to `normalized_shape` a 1-tuple `(C,)` (§0.4's guard), the textbook-correct
  reading (channel = the trailing `len(normalized_shape)` dims, PyTorch's own semantics,
  `weight`/`bias` both shaped `normalized_shape`) that also happens to be the only configuration
  exercised anywhere in this codebase. Channel = the last dim; `T_i` = product of all other dims.
  Pool via `h.reshape(-1, C)` (channel already last, no permute needed — the same pattern
  `_h_linear`/`_s_linear` already use).

Both pooling operations are applied *identically* to `h` (forward hook, pre-normalisation input) and
`s` (backward hook, gradient w.r.t. the layer's own output, same shape as `h` by construction) — so
row `x` of the pooled `h` and row `x` of the pooled `s` refer to the same `(batch, spatial)` position
in both cases, preserving the row-pairing `ekfac`/`tkfac`/`tekfac`'s intra-batch `s*`/`Θ` estimators
already rely on for `Linear`/`Conv2d` (`plan_lot2.md` §0.3, `plan_lot4.md` §0.1).

### 0.4 Scope restriction: `LayerNorm` with `normalized_shape` a 1-tuple, explicit and typed

Mirroring lot 4's `groups`/`dilation` guards (`plan_lot4.md` §0.4) rather than silently mishandling a
multi-dimensional `normalized_shape` (e.g. a transformer applying `LayerNorm` over the last two axes
of a `(N, L, C)` tensor, which this codebase's own `_h_layernorm`/`_s_layernorm` axis convention was
never designed for either, per §0.3):

```python
def _check_layernorm_supported(layer: LayerNorm) -> None:
    if len(layer.normalized_shape) != 1:
        raise NotImplementedError(
            f"Full Kronecker factors for LayerNorm only support a 1-D normalized_shape so far "
            f"(lot 5 scope); got normalized_shape={tuple(layer.normalized_shape)}."
        )
```

`BatchNorm2d` needs no analogous guard: its channel axis (dim 1) and `T_i` (dims 0,2,3) are fixed by
the module's own definition, with no configurable analogue of `normalized_shape` to go wrong.
`affine=False` (`BatchNorm2d`) / `elementwise_affine=False` (`LayerNorm`) — `weight`/`bias` both
`None` — are **not newly guarded against**: this is an inherited gap already present in
`compute_h_diag`/`compute_s_diag` and in `optimizer.py`'s own module/parameter pairing since lot 1
(neither special-cases a normalisation layer with no learnable affine parameters), not a new one
introduced here.

### 0.5 Consequence: zero changes needed to any `approximations/*.py` file — cleaner than lot 4

Re-examining every consumer against §0.1-§0.4, exactly the way `plan_lot4.md` §0.5 did for `Conv2d`:

- **`_kron_utils.py`.** `augment_direction`'s lot-4 branch already handles any `weight_direction`
  with `ndim != 2` by reshaping to `(weight_direction.size(0), -1)` — for a normalisation layer's
  1-D `(C,)` weight, this reshapes to `(C, 1)`, exactly the shape needed to `cat` with
  `bias_direction.unsqueeze(1)` into a `(C, 2)` joint direction. `split_direction` is already
  shape-agnostic. **Zero changes.**
- **`kfac.py`.** `update_input_factor`/`update_output_factor` already dispatch through
  `compute_h_full`/`compute_s_full`; `precondition` already goes through `_kron_utils`. **Zero
  changes** — the same outcome `plan_lot4.md` §0.5 already recorded for `Conv2d`.
- **`ekfac.py`, `tkfac.py`, `tekfac.py`.** All three already call the *dispatching* `augment_input`/
  `flatten_output_grad` (introduced in lot 4 precisely so a fourth layer type would not need new
  import/call swaps in these three files, `plan_lot4.md` §1.3). `_tkfac_utils.py`'s
  `instantaneous_raw_factors` is already generic over any `(h_bar, s)` shape (it only ever computes
  `‖·‖²` row norms and outer-product-weighted sums, independent of what the columns *mean*).
  **Zero changes.**
- **`base.py`, `optimizer.py`, `diag.py`, `approximations/__init__.py`.** Unaffected, as in every
  prior lot; `BatchNorm2d`/`LayerNorm` are already in `SUPPORTED_MODULES` and already hooked since
  lot 1 (needed for `diag`'s own support of these types).

This lot's entire code change is therefore confined to `factors.py`: new pooling/augmentation
functions plus new branches in the four existing dispatchers (`compute_h_full`, `compute_s_full`,
`augment_input`, `flatten_output_grad`). This is one step cleaner than lot 4, which still needed a
one-line change to `_kron_utils.py` for `Conv2d`'s 4-D weight — here, that change already exists and
covers this lot's 1-D weight case too, by coincidence of `reshape(size(0), -1)` handling both.

### 0.6 `diag.py` is untouched

`plan.md` §5.2 is explicit: *"the four new modes follow Prop. 3.1; the `diag` mode keeps the code's
formula."* `compute_h_diag`/`compute_s_diag`'s existing `_h_batchnorm2d`/`_s_batchnorm2d`/
`_h_layernorm`/`_s_layernorm` (lot 1, bit-exact port of `adafisher.py`) are not modified, not
refactored to share code with this lot's new functions, and not compared for `allclose` against the
new `compute_h_full`/`compute_s_full` (unlike `Linear`'s lot-2 check and `Conv2d`'s lot-4 "constant
factor `P`" check) — the two paths compute genuinely different statistics by design (§0.1: the
existing code's "sum-then-square" `H_D`/"sum-then-square" `S_D` for normalisation layers is *itself*
already a further, undocumented deviation from Proposition 3.1's literal "square-then-sum" formulas,
predating this project; `CLAUDE.md`'s own §5.2 flags this precisely). There is no meaningful "diagonal
of the full factor" identity to assert here the way there was for `Linear`/`Conv2d`, because the two
paths do not even agree on what quantity `H_D`/`A[0,0]` estimates.

---

## 1. File-by-file design

```
src/adafisher_modes/
└── factors.py    # + _pool_batchnorm2d, _check_layernorm_supported, _pool_layernorm (shared pooling)
                   # + augment_norm_input, _h_full_norm          (compute_h_full's new branch)
                   # + flatten_norm_output_grad, _s_full_norm    (compute_s_full's new branch)
                   # + compute_h_full/compute_s_full/augment_input/flatten_output_grad: new
                   #   BatchNorm2d/LayerNorm branches (replacing their NotImplementedError paths)
```

No other file changes (§0.5, §0.6). `docs/reports/plan.md` is unchanged (overall plan, not a per-lot
log); `CLAUDE.md`'s status header, file-layout comment and "Running the tests" table are updated at
the end, once the exit tests pass, exactly as after lots 1-4.

### 1.1 `factors.py` additions

```python
def _pool_batchnorm2d(x: Tensor) -> Tensor:
    """(N, C, H, W) -> (N*H*W, C): move the channel axis (dim 1, BatchNorm2d's own convention,
    _h_batchnorm2d/_s_batchnorm2d) last and flatten batch+spatial onto one axis."""
    return x.permute(0, 2, 3, 1).reshape(-1, x.size(1))


def _check_layernorm_supported(layer: LayerNorm) -> None:
    if len(layer.normalized_shape) != 1:
        raise NotImplementedError(
            f"Full Kronecker factors for LayerNorm only support a 1-D normalized_shape so far "
            f"(lot 5 scope); got normalized_shape={tuple(layer.normalized_shape)}."
        )


def _pool_layernorm(x: Tensor, layer: LayerNorm) -> Tensor:
    """(..., C) -> (T, C): channel is already the last axis for a 1-D normalized_shape (PyTorch's
    own convention), so this is a plain flatten, mirroring _h_linear/_s_linear."""
    _check_layernorm_supported(layer)
    return x.reshape(-1, x.size(-1))


def _pool_norm_layer(x: Tensor, layer: Module) -> Tensor:
    """Dispatching sibling: (T, C) pooled view shared by the H and S sides of a normalisation
    layer's full-factor construction (docs/reports/plan_lot5.md §0.3)."""
    if isinstance(layer, BatchNorm2d):
        return _pool_batchnorm2d(x)
    if isinstance(layer, LayerNorm):
        return _pool_layernorm(x, layer)
    raise NotImplementedError(
        f"_pool_norm_layer only supports BatchNorm2d and LayerNorm; got {type(layer)}"
    )


def augment_norm_input(h: Tensor, layer: Module) -> Tensor:
    """(T, 2) matrix [z_x, 1], the normalisation-layer analogue of augment_linear_input/
    augment_conv2d_input. z_x is the per-position channel-mean pre-activation -- the Frobenius-
    optimal, S-independent scalar surrogate (docs/reports/plan_lot5.md §0.2) for Proposition 3.1's
    exact H_{i-1}|_{nu_i} (a full C x C matrix, Hadamard- not Kronecker-combined with S there), fit
    into this codebase's shared kron(A,B) machinery. Column 1 (the constant) reproduces
    H_{i-1}|_{beta_i}=11^T's exact contribution (=S) once run through that same machinery.
    """
    pooled = _pool_norm_layer(h, layer)
    z = pooled.mean(dim=1)
    return torch.stack([z, torch.ones_like(z)], dim=1)


def _h_full_norm(h: Tensor, layer: Module) -> Tensor:
    h_bar = augment_norm_input(h, layer)
    return h_bar.t() @ h_bar / h_bar.size(0)


def flatten_norm_output_grad(s: Tensor, layer: Module) -> Tensor:
    """(T, C) pooled gradient batch -- literally Proposition 3.1's S_i once reduced (_s_full_norm),
    and the raw per-sample batch ekfac/tkfac/tekfac need for their intra-batch estimators."""
    return _pool_norm_layer(s, layer)


def _s_full_norm(s: Tensor, layer: Module) -> Tensor:
    s_pool = flatten_norm_output_grad(s, layer)
    return s_pool.t() @ s_pool / s_pool.size(0)
```

`compute_h_full`, `compute_s_full`, `augment_input`, `flatten_output_grad` each gain one new branch:

```python
def compute_h_full(h: Tensor, layer: Module) -> Tensor:
    if isinstance(layer, Linear):
        return _h_full_linear(h, layer)
    if isinstance(layer, Conv2d):
        return _h_full_conv2d(h, layer)
    if isinstance(layer, (BatchNorm2d, LayerNorm)):
        return _h_full_norm(h, layer)
    raise NotImplementedError(
        f"compute_h_full only supports Linear, Conv2d, BatchNorm2d and LayerNorm; got {type(layer)}"
    )

# compute_s_full, augment_input, flatten_output_grad: same pattern, dispatching to
# _s_full_norm / augment_norm_input / flatten_norm_output_grad respectively.
```

The module docstring's lot-4-era `"lot 4 scope"` `NotImplementedError` messages are updated to drop
`BatchNorm2d`/`LayerNorm` from the unsupported list (they now only exclude genuinely unsupported
inputs: `groups≠1`/`dilation≠(1,1)` `Conv2d`, `LayerNorm` with a non-1-D `normalized_shape`, and any
other `nn.Module` subtype).

---

## 2. Tests

### 2.1 `test_full_factors_match_diag.py` — replace the "raises `NotImplementedError`" block, add
    shape/exactness checks

The existing `test_full_factors_raise_on_unsupported_layers` (parametrized over `BatchNorm2d` and
`LayerNorm`, both currently expected to raise) is **removed** — both are now supported — and replaced
with:

- **Shapes.** `compute_h_full(h, bn)` is `(2, 2)` for `bn = nn.BatchNorm2d(3)`,
  `h = randn(4, 3, 5, 5)`; `compute_s_full(s, bn)` is `(3, 3)`. Likewise
  `compute_h_full(h, ln)` is `(2, 2)`, `compute_s_full(s, ln)` is `(8, 8)` for
  `ln = nn.LayerNorm(8)`, `h = randn(4, 8)`.
- **`A[1,1] == 1.0` exactly** (§0.2's `a_β` — not merely `allclose` to a tolerance, since it is a
  literal `mean_x(1²)` with no floating-point accumulation beyond a sum of ones), for both layer
  types.
- **`A[0,0] == mean(z²)`, checked against `augment_norm_input`'s own pooled `z`** directly (a
  reduction-order variant of the same statistic — `allclose`, mirroring lot 2/4's own
  diagonal-vs-full sanity checks).
- **Symmetry.** `A`, `B` both symmetric for both layer types (`A[0,1]==A[1,0]` since it is a
  Gram matrix; `B` symmetric since `S_i` is a Gram matrix, per Proposition 3.1).
- **Scope guard.** `nn.LayerNorm((4, 5))` (a 2-tuple `normalized_shape`) raises `NotImplementedError`
  from `compute_h_full`, `compute_s_full`, and `augment_input`.
- **Regression.** `Conv2d` `groups≠1`/`dilation≠(1,1)` guards (lot 4) still raise; the `Linear`/
  `Conv2d` shape/symmetry/reconstruction tests are unmodified.

### 2.2 `test_frobenius_dominance.py` — new "Lot 5: normalisation layers" section

Mirrors lot 4's `Conv2d` section structure exactly (`plan_lot4.md` §2.2). New module-level constants:

```python
NORM_C, NORM_HW, NORM_N = 3, 4, 6     # BatchNorm2d(3), (N, 3, 4, 4) -> T = N*H*W = 96
NORM_D_IN_AUG = 2
```

`_exact_fisher_block_norm(h, s, layer)`: an **independent** oracle (re-derives the pooling locally,
does not call `_pool_norm_layer`/`augment_norm_input`/`flatten_norm_output_grad`):

```python
def _exact_fisher_block_norm(h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # h, s: (N, C, H, W), BatchNorm2d's own axis convention re-derived independently.
    pooled_h = h.permute(0, 2, 3, 1).reshape(-1, NORM_C)
    pooled_s = s.permute(0, 2, 3, 1).reshape(-1, NORM_C)
    z = pooled_h.mean(dim=1)
    h_bar = torch.stack([z, torch.ones_like(z)], dim=1)
    dense_dim = NORM_C * NORM_D_IN_AUG
    F_exact = torch.zeros(dense_dim, dense_dim)
    for x in range(h_bar.size(0)):
        F_exact += torch.kron(torch.outer(pooled_s[x], pooled_s[x]), torch.outer(h_bar[x], h_bar[x]))
    return F_exact / h_bar.size(0)
```

This is, by construction (§0.2), exactly the quantity `compute_h_full`/`compute_s_full` +
`kfac`/`ekfac`/`tkfac`/`tekfac`'s existing machinery are designed to approximate for a normalisation
layer — the literal per-position pooled-sample analogue of `Linear`/`Conv2d`'s own `F`, with `h_bar`
built from the channel-mean `z_x` instead of a full feature/patch vector. `_build_norm_kfac_and_ekfac`/
`_build_norm_tkfac_and_tekfac` mirror `_build_conv_kfac_and_ekfac`/`_build_conv_tkfac_and_tekfac`
exactly (`gammas=(1.0,1.0)`, bootstrap-then-redrive for `s*`/`Θ`), driven with raw
`(N, C, H, W)`-shaped `h`, `s` through a real `nn.BatchNorm2d(3)` as `module`.

Eight tests, direct `_norm`-suffixed mirrors of the `Linear`/`Conv2d` ones:
`test_ekfac_dominates_kfac_in_frobenius_norm_norm`, `test_eigenbases_are_orthogonal_norm`,
`test_f_tilde_matches_precondition_for_kfac_norm`, `test_f_tilde_matches_precondition_for_ekfac_norm`,
`test_tkfac_trace_matches_exact_fisher_norm` (`Lambda=1e-12`, same reasoning as the `Linear`/`Conv2d`
versions), `test_tekfac_dominates_tkfac_in_frobenius_norm_norm`,
`test_f_tilde_matches_precondition_for_tkfac_norm`, `test_f_tilde_matches_precondition_for_tekfac_norm`.

### 2.3 `test_kfac_ekfac_precondition.py`, `test_tkfac_tekfac_precondition.py` — extend

For each of the four modes, on `nn.BatchNorm2d(3)` (`h, s = randn(4, 3, 5, 5)` twice) and
`nn.LayerNorm(8)` (`h, s = randn(16, 8)` twice):

- **Shape.** `precondition(layer, weight_direction, bias_direction)` returns a `(C,)`/`(C,)` pair
  (both layer types always have a bias-shaped parameter here — `affine=True`/
  `elementwise_affine=True`, PyTorch's own defaults, §0.4).
- **Cadence.** `T_inv`/`T_eig`/`T_re` gate refresh identically to the existing `Linear`/`Conv2d`
  cadence tests.
- **Regression.** The full pre-lot-5 body of both files is re-run unmodified.

### 2.4 `test_norm_layers_optimizer_smoke.py` — new

The practical claim this lot exists to make, and the literal wording of `plan.md`'s own exit
criterion ("loss non-regression vs. `diag` on a small CNN"), mirroring
`test_conv2d_optimizer_smoke.py`'s structure (`plan_lot4.md` §2.4) but reusing `conftest.py`'s
existing `TinyMultiLayerNet` (the one net already covering all four `SUPPORTED_MODULES`, including
both `BatchNorm2d` and `LayerNorm` — no new network needed, unlike lot 4's purpose-built
`TinyConvNet`, which deliberately excluded normalisation layers because they were unsupported then).

```python
@pytest.mark.parametrize("mode", ["kfac", "ekfac", "tkfac", "tekfac"])
def test_optimizer_runs_on_full_net(mode: str) -> None:
    seed_all(0)
    model = TinyMultiLayerNet()
    initial = {name: p.clone() for name, p in model.named_parameters()}
    opt = AdaFisherMulti(model, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, **_MODE_KWARGS[mode])

    for _ in range(6):
        x = torch.randn(6, 2, 5, 5)
        opt.zero_grad()
        loss = model(x).pow(2).sum()
        loss.backward()
        opt.step()

    moved = False
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"{name} has non-finite entries after {mode} training"
        if not torch.equal(p, initial[name]):
            moved = True
    assert moved, f"no parameter moved under {mode} — preconditioning was a silent no-op"
```

Plus the literal exit-criterion check, run separately (own seed, own model instance) against `diag`:

```python
@pytest.mark.parametrize("mode", ["kfac", "ekfac", "tkfac", "tekfac"])
def test_loss_non_regression_vs_diag(mode: str) -> None:
    """plan.md §8's own lot-5 exit test, verbatim: with identical initial weights and identical
    minibatches, a few steps of the new mode must not diverge relative to diag -- a sanity floor,
    not a claim that the new modes beat diag (plan.md §6.1: dominance over diag is measured, not
    asserted, and that reasoning applies here too -- F~_D is not of the form the four new modes
    share, and comparing final losses is not a Frobenius-norm statement in the first place).
    """
    seed_all(1)
    model_new = TinyMultiLayerNet()
    model_diag = TinyMultiLayerNet()
    model_diag.load_state_dict(model_new.state_dict())

    opt_new = AdaFisherMulti(model_new, lr=1e-3, fisher_mode=mode, TCov=1, Lambda=1e-2, **_MODE_KWARGS[mode])
    opt_diag = AdaFisherMulti(model_diag, lr=1e-3, fisher_mode="diag", TCov=1, Lambda=1e-2)

    batches = [torch.randn(6, 2, 5, 5) for _ in range(6)]
    loss_new = loss_diag = None
    for x in batches:
        for model, opt in ((model_new, opt_new), (model_diag, opt_diag)):
            model.zero_grad()
            loss = model(x).pow(2).sum()
            loss.backward()
            opt.step()
        loss_new, loss_diag = model_new(batches[-1]).pow(2).sum(), model_diag(batches[-1]).pow(2).sum()

    assert torch.isfinite(loss_new)
    assert loss_new <= 10 * loss_diag + 1.0
```

(`_MODE_KWARGS` reuses `test_conv2d_optimizer_smoke.py`'s own dict, redefined locally to keep this
file self-contained per the existing per-test-file convention.) The `10×` factor is a generous
non-divergence floor, not a tight quantitative claim — consistent with `plan.md` §6.1's own
established position that comparisons against `diag` are reported, not asserted as a tight bound;
here a *pass/fail* floor is still appropriate (unlike §6.1's Frobenius measurements) because the
literal exit criterion in `plan.md` §8 is phrased as "non-regression", i.e. a floor, not a
measurement to log.

---

## 3. Explicitly out of scope for lot 5

- The SUA approximation and the equal-wall-clock convergence bench — lots 6-7 (`plan.md` §6.3, §7,
  §8), unaffected by anything in this lot.
- `affine=False` (`BatchNorm2d`) / `elementwise_affine=False` (`LayerNorm`) — §0.4, an inherited gap
  from lot 1's own module/parameter pairing, not introduced or newly exposed here.
- `LayerNorm` with a `normalized_shape` of rank `≠ 1` — §0.4, a typed, explicit guard, matching the
  project's established `groups`/`dilation` pattern (`plan_lot4.md` §0.4) rather than silently
  reusing the wrong axis convention.
- Enforcing Proposition A.1's exact "cross-terms between `ν_i` and `β_i` excluded" independence —
  §0.2, a deliberate, cited non-strict reading, kept consistent with how `Linear`/`Conv2d`'s own bias
  column is already (also non-strictly) treated.
- Any change to `optimizer.py`, `approximations/base.py`, `diag.py`, `kfac.py`, `ekfac.py`,
  `tkfac.py`, `tekfac.py`, `_kron_utils.py`, `_tkfac_utils.py`, `approximations/__init__.py`, or
  lots 1-4's existing tests, beyond what §2 adds alongside them.
- Benchmark integration (`benchmarks/mnist_autoencoder.py` has no normalisation layer and is not
  extended to one here) — lot 7's equal-wall-clock comparison.

---

## 4. Points worth flagging explicitly (not blocking, per this session's operating mode)

1. §0.5 — this lot needs **zero** changes to any file under `approximations/` (not even
   `_kron_utils.py`, which lot 4 still had to touch once for `Conv2d`'s 4-D weight): the lot-4
   `augment_direction` reshape and the lot-4 dispatching `augment_input`/`flatten_output_grad`
   functions already generalise far enough to cover a fourth layer type's `1×2`-shaped joint
   direction with no further change. This is the third consecutive lot recording the shared-ABC
   architecture (`plan.md` §2.2) paying for itself, and the strongest instance yet.
2. §0.1-§0.2 — Proposition 3.1's exact FIM for a normalisation layer is **Hadamard-**, not
   Kronecker-, structured (`FIM_ν = H|_ν ⊙ S`, `FIM_β = S` exactly), a fact `plan.md`'s own
   cheat-sheet-level summary ("`A` is a scalar... no invention") does not make explicit. Fitting this
   into the shared `kron(A,B)` machinery is a genuine, principled design decision (a
   Frobenius-optimal, `S`-independent scalar surrogate for the `ν`-block's exact `H|_ν`), not the
   "no invention" `plan.md` §5.2 suggested — recorded here precisely so the derivation, not just the
   conclusion, is auditable.
3. §0.2 — the resulting small, non-zero `ν`-`β` coupling term (`A[0,1] = mean_x(z_x)`) is a
   deliberate choice to stay *consistent* with how `Linear`/`Conv2d`'s own bias column is already
   (non-strictly) treated, rather than a special-cased correction chasing exact fidelity to
   Proposition A.1's own stated idealisation. Flagged explicitly as a documented, cited departure,
   not a silent approximation error.
4. §0.6 — `diag.py`'s existing normalisation-layer formula (already known, per `CLAUDE.md`, to
   itself deviate from Proposition 3.1 via "sum-then-square" instead of "square-then-sum") is left
   completely untouched, and this lot's new `compute_h_full`/`compute_s_full` are *not* checked for
   any `allclose` relationship against it — unlike every previous lot's `Linear`/`Conv2d` sanity
   check — because the two paths do not estimate the same quantity even up to a reduction-order
   difference; asserting a relationship between them would test something that is not true by
   design.
