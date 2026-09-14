# Lot 4 Implementation Plan — `Conv2d` (KFC) for `kfac`/`ekfac`/`tkfac`/`tekfac`

**Status:** draft, awaiting implementation in this same session. No lot-4 code written yet.
**Parent plan:** `docs/reports/plan.md` §8, lot 4 row: *"`Conv2d` (KFC) — §6.1 replayed on a toy `3×3`,
2-channel conv."* This document refines that one-line entry to implementation-ready precision, the
way `plan_lot2.md`/`plan_lot3.md` did for lots 2/3. It does not revisit lots 1-3 (closed, 43 tests
passing) or lots 5-7 (normalisation layers, SUA, equal-wall-clock bench).

Exit criteria (`plan.md` §8, lot 4 row, unpacked against §6.1, the same way lot 2/3 unpacked theirs):

1. `compute_h_full`/`compute_s_full` (hence `kfac`/`ekfac`/`tkfac`/`tekfac`) support `Conv2d` with
   `groups=1`, `dilation=(1,1)` — the KFC construction (Grosse & Martens 2016, `plan.md` §1.5/§4.2).
2. On a toy `Conv2d(2, 3, kernel_size=3, padding=1)` where the per-(example, output-location)
   pooled exact Fisher block `F` is computable by an *independent* oracle (not reusing this lot's
   own extraction code):
   - `‖F − F̃_EKFAC‖_F ≤ ‖F − F̃_KFAC‖_F` (EKFAC Thm 2/3) — §6.1 assertion 1, replayed.
   - `‖F − F̃_TEKFAC‖_F ≤ ‖F − F̃_TKFAC‖_F` (TEKFAC Thm 3.1) — §6.1 assertion 2, replayed.
   - `tr(F̃_TKFAC) = tr(F)` (TKFAC Thm 4.1/Lemma 4.1) — §6.1 assertion 3, replayed.
   - `Q_A`, `Q_B`, `Q_Φ`, `Q_Ψ` orthogonal — §6.1 assertion 4, replayed.
3. `groups≠1` and `dilation≠(1,1)` raise a typed `NotImplementedError`, not a silently wrong factor.
4. No regression: the full lots 1-3 suite (43 tests) still passes unmodified.
5. A real `AdaFisherMulti` (hooks, not direct calls) runs a few `.step()`s on a small Conv2d+Linear
   network without error, for all four modes — the practical point of this lot, not itself one of
   the four numbered `plan.md` §6.1 assertions but necessary to call Conv2d support "done" (§0.7).

---

## 0. Design decisions this lot must resolve before code exists

### 0.1 The pooled-`(batch, output-location)` view *is* KFC — already half-built in `factors.py`

`plan.md` §1.5 already established, for the **diagonal** path, that AdaFisher's own `Conv2d`
handling (`_extract_patches` → flatten `(batch, spatial)` onto axis 0 → reduce) "is exactly the KFC
construction of Grosse & Martens 2016." Concretely: `extract_patches` (`factors.py:32-52`, a verbatim
port of `_extract_patches`, `adafisher.py:33-50`) already turns a `Conv2d` input `(N, C_in, H, W)`
into patches `(N, groups, S, P)` with `S = H_out·W_out` output locations and
`P = (C_in/groups)·k_h·k_w`; `_h_conv2d`/`compute_h_diag`'s `Conv2d` branch already reshapes this to
`(N·S, P)` and treats every `(example, location)` pair as one i.i.d. sample before reducing.

**This lot's entire job is to stop reducing to a diagonal at that point and instead form the full
outer-product mean** — exactly the diagonal-to-full generalisation `plan_lot2.md` §0.1 already did
for `Linear` (`_h_linear`'s `einsum("ij,ij->j", h_bar, h_bar)` → `_h_full_linear`'s
`h_bar.t() @ h_bar`), applied to the *same* pooled `(N·S, P[+1])` matrix `extract_patches` already
builds. No new patch-extraction logic is needed; `extract_patches` is reused unmodified.

**The independence assumption this treatment relies on has a name and a citation.** Treating distinct
output locations as extra i.i.d. samples — i.e. discarding the cross-location terms `E[a_t a_{t'}^T]`,
`t≠t'`, in the true per-layer Fisher block — is precisely **TKFAC's Assumption 4.1**
(`tkfac_2011.10741.pdf` §4.3, cited already in `plan.md` §4.4: *"products of activations and
pre-activation derivatives are uncorrelated at any two distinct spatial locations"*). It is not a new
assumption introduced by this lot; it is the general form of the same pooling the diagonal path
already performs, made explicit because §0.7 below needs to cite it precisely when defining the exit
test's oracle `F`.

**Consequence: this generalises to all four modes with no per-mode invention.** `kfac`/`ekfac`'s
independence assumption (`A⊗B` in place of the true block) and TKFAC/TEKFAC's trace-restricted
factorisation are both defined generically over *any* paired sample set `{(a_i, δ_i)}` — neither
Lemma 1's proof (`ekfac_1806.03884.pdf` Appendix A.1) nor Theorem 4.1's proof
(`tkfac_2011.10741.pdf`, the partial-trace argument) refers to where the samples came from. Once
`Conv2d`'s `(example, location)` pairs are exposed as an `(N_eff, d_in_aug)`/`(N_eff, d_out)` sample
matrix in the same shape `Linear`'s batch already has, `_tkfac_utils.instantaneous_raw_factors`/
`bootstrap_raw_factors` (lot 3, generic over `(h_bar, s)`) need **zero changes**, and neither does
`kfac.py` (§0.5).

### 0.2 A pre-existing scale quirk in the *diagonal* `Conv2d` path — found, verified, deliberately not
    reproduced

Empirically verifying `plan.md` §5.3's "mean over batch × spatial" convention against the ported
`_h_conv2d` (`factors.py:55-63`) surfaces a real discrepancy, present identically in **both**
`reference_repos/FisherAdapTune/scripts/adafisher.py:66-76` and
`reference_repos/AdaFisher/optimizers/AdaFisher.py:143-155` (so it predates this project, inherited
from the paper authors' own code, not a FisherAdapTune-specific bug):

```python
h = _extract_patches(h, layer.kernel_size, layer.stride, layer.padding, layer.groups)  # (N, groups, S, P)
spatial_size = h.size(2) * h.size(3)     # = S * P, NOT S  (h.size(2)=S, h.size(3)=P after extract_patches)
h = h.reshape(-1, h.size(-1))
...
return einsum("ij,ij->j", h_bar, h_bar) / (batch_size * spatial_size)   # divides by N*S*P, not N*S
```

Verified directly (`Conv2d(2, 3, kernel_size=3, bias=True)`, `x` = `randn(4, 2, 5, 5)`): building `A`
the mathematically-correct way (`h_bar.t() @ h_bar / (batch_size * S)` with the *real* `S = 9`) and
comparing its diagonal to `compute_h_diag`'s output gives a **constant ratio of exactly `18` per
entry** — `18 = P = C_in·k_h·k_w`, i.e. `compute_h_diag`'s `Conv2d` branch is scaled down by a factor
of `P` relative to a true "mean over batch × spatial" statistic. (`_s_conv2d`, by contrast, computes
`spatial_size = s.size(2) * s.size(3)` **before** any reshape — `s`'s original `(N, C_out, H_out,
W_out)` shape — so its `spatial_size` is genuinely `S`; there is no analogous quirk on the `S`/output
side.)

This is harmless for `diag(minmax_normalization=True)` (the paper-faithful default,
`plan.md` §5.1): min-max normalisation rescales each layer's `H_D` to `[0,1]` independently, so any
per-layer constant multiplicative factor is erased before it ever reaches `F̃_D`. It is *not* harmless
for `kfac`/`ekfac`/`tkfac`/`tekfac`: none of the four new modes apply min-max
(`CLAUDE.md`, "the four new modes never apply min-max"), damping is `λ` only, and TKFAC's whole
`tr(F̃)=tr(F)` guarantee (§6.1 assertion 3) requires the *actual* per-layer Fisher scale, not one
arbitrarily shrunk by `P`. **Decision: `compute_h_full`'s `Conv2d` branch uses the correct
`spatial_size = S` (computed the same way `_s_conv2d` already correctly does it, i.e. from the
patches tensor's own spatial axis before any reshape collapses it against `P`), not `_h_conv2d`'s
scale.** `_h_conv2d`/`compute_h_diag` are **not** touched — `CLAUDE.md`'s "change only what's
requested" applies to lot-1 code with passing bit-exactness tests tied to its exact (quirky) form —
so `torch.diagonal(compute_h_full(h, conv_layer))` and `compute_h_diag(h, conv_layer)` will *not* be
`allclose` for `Conv2d` (unlike the `Linear` case in lot 2, where they do match): they differ by the
constant factor `P`, and the lot-4 test asserts that constant explicitly (§2.1) rather than leaving
the discrepancy unverified.

### 0.3 Weight/bias plumbing: one small, already-generic extension to `_kron_utils.py`

A `Conv2d`'s weight has shape `(C_out, C_in, k_h, k_w)`, not `(d_out, d_in)`. `split_direction`
already handles this with **no change**: `M[:, :-1].reshape(weight_shape)` reshapes a
`(C_out, C_in·k_h·k_w)` slice back into `(C_out, C_in, k_h, k_w)` correctly, *because* the patch
dimension `P`'s internal layout — `(C_in/groups, k_h, k_w)`, established by `extract_patches`'s own
`permute(0,1,3,4,2,5,6)` (`factors.py:49`) — is exactly `Conv2d.weight`'s own dimension order after
`C_out`. This was verified directly (§0.7's reconstruction check re-derives `weight.grad` itself from
pooled patches/gradients and matches to float precision, which would fail immediately under any
ordering mismatch).

`augment_direction` **does** need one line: it currently assumes `weight_direction` is already 2-D.

```python
def augment_direction(weight_direction: Tensor, bias_direction: Optional[Tensor]) -> Tensor:
    W = weight_direction if weight_direction.ndim == 2 else weight_direction.reshape(weight_direction.size(0), -1)
    if bias_direction is None:
        return W
    return cat([W, bias_direction.unsqueeze(1)], dim=1)
```

This is the **only** change to `_kron_utils.py`, and it is the **only** change `kfac.py` and
`tkfac.py` need at all (§0.5) — both call `augment_direction`/`split_direction` and nothing else
shape-specific in `precondition()`.

### 0.4 Scope restriction: `groups=1`, `dilation=(1,1)`, explicit and typed

**`groups`.** KFC's derivation assumes every input channel can, in principle, influence every output
channel through one shared `A⊗B` block. A grouped convolution's weight has structural zeros between
groups (`weight[:, c, :, :]` for an input channel `c` outside a given output channel's group is not a
parameter at all), so pooling all groups' patches into one `A` and all output channels into one `B`
would silently mix statistics across channels that share no actual weight — a materially different,
block-diagonal-per-group derivation would be required (not covered by any of the four cited papers'
main-text formulas `plan.md` §4 sources). Depthwise convolutions (`groups=C_in`) are the extreme,
practically-relevant case of this. **Out of scope for lot 4**, exactly the kind of explicit,
typed-error boundary the project already draws elsewhere (`compute_h_full` raising for `BatchNorm2d`/
`LayerNorm`, `plan_lot2.md` §0.1) rather than silently computing something wrong.

**`dilation`.** `extract_patches`/`_extract_patches` (lot 1, ported bit-exact, ungated) has no
`dilation` parameter at all — a dilated `Conv2d`'s receptive field is not what `unfold` extracts,
so the diagonal path has silently been wrong for `dilation≠(1,1)` since lot 1 (inherited, not
introduced by this lot — `TinyMultiLayerNet`'s and every existing test's `Conv2d` all use the
default `dilation=(1,1)`, so this was never exercised). This lot does not fix the diagonal path
(out of scope, `CLAUDE.md`), but the **new** `Conv2d` code added here explicitly guards against it
rather than silently inheriting the same latent gap into four more modes.

```python
def _check_conv2d_supported(layer: Conv2d) -> None:
    if layer.groups != 1:
        raise NotImplementedError(
            f"Full Kronecker factors for Conv2d only support groups=1 so far (lot 4 scope); "
            f"got groups={layer.groups}. A grouped/depthwise layer needs a block-diagonal "
            f"per-group treatment, not a single global A (x) B."
        )
    if layer.dilation != (1, 1):
        raise NotImplementedError(
            f"Full Kronecker factors for Conv2d only support dilation=(1,1) so far (lot 4 scope); "
            f"got dilation={layer.dilation}. extract_patches has no dilation parameter."
        )
```

### 0.5 What each of the four modes actually needs to change — and why `kfac.py` needs nothing

Re-examining each mode's source against §0.1-§0.3:

- **`kfac.py`.** `update_input_factor`/`update_output_factor` already call the *dispatching*
  `compute_h_full`/`compute_s_full` (`plan_lot2.md` §1.3) — adding a `Conv2d` branch inside those two
  functions is enough; `kfac.py` itself is not touched except its module docstring (scope claim).
  `precondition()` calls `augment_direction`/`split_direction`, both `Conv2d`-ready per §0.3. **Zero
  logic changes** — the clean generalisation `plan_lot2.md` §0.2's shared-ABC argument promised.
- **`ekfac.py`.** Needs the *raw* augmented batch (not just its `A` reduction) to cache for the `s*`
  intra-batch estimator (`plan_lot2.md` §0.3) — it does not go through `compute_h_full` for that
  reason. Its own `A_i = h_bar.t() @ h_bar / h_bar.size(0)` line is already shape-agnostic (works
  identically whether `h_bar` came from `Linear` or pooled `Conv2d` patches); the **only** change is
  what builds `h_bar` and `s_flat` in the first place — swap the direct call to (`Linear`-only)
  `augment_linear_input` for a new dispatching `augment_input` (§1.1), and the inline
  `s.reshape(-1, s.shape[-1]) if s.ndim > 2 else s` for a new dispatching `flatten_output_grad`
  (§1.1) — two one-line swaps, no other change.
- **`tkfac.py`/`tekfac.py`.** Same two swaps (`augment_linear_input`→`augment_input`,
  inline `s.reshape(...)`→`flatten_output_grad`) in `update_input_factor`/`update_output_factor`.
  `_tkfac_utils.instantaneous_raw_factors`/`bootstrap_raw_factors` are untouched (§0.1) — they
  already operate on whatever `(h_bar, s)` matrices they are handed. `precondition()` in both files
  uses `augment_direction`/`split_direction`, `Conv2d`-ready per §0.3, unchanged.

This confirms `plan_lot2.md` §0.2's own bet (a single shared ABC, a single shared `_kron_utils.py`)
pays off a second time: adding a fourth supported layer type to four modes costs one new dispatcher
pair in `factors.py`, one line in `_kron_utils.py`, and a one-line import/call swap in three files —
not a rewrite of any mode's core algorithm.

### 0.6 Conv-specific *adaptive* damping (TKFAC's `ν`-floor, TEKFAC's `β`-rescaling) — deliberately
    deferred, not silently dropped

`plan.md` §4.4/§4.5 (from `CLAUDE.md`'s own cheat sheet) already flag that both trace-restricted
papers define a **convolution-specific** damping refinement beyond their shared-with-`Linear` `λ`:

- TKFAC: *"Adaptive conv damping: §5, Eq. (5.16), Alg. 1"* (`tkfac_2011.10741.pdf`) — a `ν`-floor
  mechanism for the conv case, distinct from the plain `λ` this project's `tkfac.py` already applies
  (`plan_lot3.md` §0.3's `(1/δ)Ψ̃⁻¹MΦ̃⁻¹` derivation, which is dimension-agnostic and does not itself
  distinguish `Linear` from `Conv2d`).
- TEKFAC Eq. (3.5): `λ = max{tr(Θ_l), ϑ}/dim(Θ_l)` for conv layers specifically, with a derived `β`
  factor applied to a CNN's *dense* layers "to keep pace with convolution layers" — again layered on
  top of, not replacing, the plain-`λ` mechanism `tekfac.py` already implements generically.

**Decision: lot 4 reuses each mode's existing, already-implemented damping scheme unchanged for
`Conv2d`** (`kfac`'s factored `π` Tikhonov, `ekfac`'s additive `λ` on `s*`, `tkfac`'s
`√(λ/δ)`-scaled identity, `tekfac`'s additive `λ` on `Θ`) — all four are dimension-agnostic algebra
that already runs correctly on `Conv2d`-shaped factors once §0.1-§0.4 are in place. The *adaptive*,
conv-specific refinements above are **not implemented in this lot**: none of §6.1's cited theorems
(Lemma 1's optimal-diagonal argument, Thm 2/3, Thm 4.1's trace identity, Thm 3.1) depend on which
positive-definite damping scheme is layered on top of the undamped decomposition — damping is added
after the approximation that those theorems characterise, not part of it. This is the same category
of deliberate, cited simplification `plan_lot3.md` §0.4 already made for `T_fim` ("introducing it now,
unused by any existing mode, would be exactly the kind of speculative parameter `CLAUDE.md`'s
conventions ask to avoid") — here, the adaptive conv damping has no bearing on this lot's exit
criteria, and speculatively implementing it now (untested against a real CNN training run, which is
lot 7's job) would be exactly that kind of premature addition.

### 0.7 The toy-conv oracle for the exit test — what "`F`" must mean, and why an *independent*
    construction is required

§6.1's Linear-layer tests build `F` by literal per-sample accumulation over the batch that
`update_input_factor`/`update_output_factor` themselves consume — no independent re-implementation
was needed there because a `Linear` layer has no spatial structure for the test to get subtly wrong.
`Conv2d` does, so two things must be gotten right and are treated separately:

**(a) What quantity is `F`?** Per §0.1, the four modes approximate the **pooled-sample** Fisher block
`F := E_{(n,t)}[(δ_{n,t}δ_{n,t}^T) ⊗ (a_{n,t}a_{n,t}^T)]`, the expectation over `(example, output
location)` pairs treated as i.i.d. — *not* the true per-layer block, which would also carry
`E[a_t a_{t'}^T]`, `t≠t'`, cross-location terms. This is not a weaker or informal choice: it is the
literal pooled-sample generalisation of §6.1's own `Linear` recipe, under the same spatial
independence assumption (TKFAC's Assumption 4.1, §0.1) the implementation itself relies on to be
correct in the first place. Building a "true" `F` that includes cross-location terms and comparing
the four modes against *that* would not be "§6.1 replayed" — it would silently test a different,
unrelated approximation (the spatial-pooling assumption itself) that none of the four cited theorems
make any claim about, and for which nothing in this project's four source papers proves a dominance
relationship. `plan.md` §4.4's own citation of TKFAC's Theorem 4.3 "under Assumption 4.1" is exactly
this same scoping, one level up (that theorem's own conv-case guarantee is *also* only stated relative
to the pooled/decorrelated block, not the fully-cross-correlated one).

**(b) Is the implementation's own patch/gradient pooling *correct*, independent of (a)?** This is
where an independent oracle matters, and where a real bug (wrong patch layout, wrong `(example,
location)` pairing between `a` and `δ`) would actually be caught. Verified directly, using a real
`Conv2d(2, 3, kernel_size=3, bias=True)` forward/backward (not a toy — the strongest available check,
since it has a known-correct target):

```python
out = layer(x); loss = (out * torch.randn_like(out)).sum()
grad_out = torch.autograd.grad(loss, out, retain_graph=True)[0]     # (N, C_out, H_out, W_out)
loss.backward()                                                     # -> layer.weight.grad (ground truth)

patches = extract_patches(x, layer.kernel_size, layer.stride, layer.padding, layer.groups)
h_bar = cat([patches.reshape(-1, P), ones_column], dim=1)           # (N*S, P+1)  -- factors.py's own pooling
s_pool = grad_out.transpose(1, 2).transpose(2, 3).reshape(-1, C_out)  # (N*S, C_out) -- ditto, output side

recon = s_pool.t() @ h_bar                                          # (C_out, P+1)
# recon[:, :-1].reshape(weight.shape) == layer.weight.grad   (max abs diff 9.5e-6, float32)
# recon[:, -1]                        == layer.bias.grad     (max abs diff 1.9e-6, float32)
```

This is not a coincidence to be taken on faith: `Σ_{(n,t)} δ_{n,t} a_{n,t}^T` **is**, by the chain
rule, exactly how a convolution's weight gradient is computed (a sum over every output position's
contribution), so if `factors.py`'s pooled `h_bar` and `s_pool` are row-aligned correctly (row `k` of
each coming from the *same* `(example, location)` pair) and use the *same* patch layout the real
`Conv2d.weight` uses, this reconstruction is an exact identity, not an approximation — a wrong
`permute`/`transpose` order, or a mismatched pairing between the two pooled tensors, breaks it
immediately and visibly (not by some subtle Frobenius-norm degradation). This is the sharpest test
available and is included directly in the test suite (§2.1), ahead of and independent from the
Frobenius-dominance/trace tests, which only make sense once this pairing is known to be right.

Given (b) is checked this way (via real autograd, no toy needed), (a)'s toy dominance/trace tests
(§2.2) draw `a`, `s` **synthetically** — i.i.d. Gaussian, exactly `plan_lot2.md`/`plan_lot3.md`'s own
Linear-case recipe — but exercise the **real** `extract_patches`-based pooling by feeding raw
`Conv2d`-shaped tensors (`(N, C_in, H, W)` inputs, `(N, C_out, H_out, W_out)` synthetic gradients)
directly into `update_input_factor`/`update_output_factor`, and cross-check against an oracle `F`
built independently using `torch.nn.functional.unfold` (a separate, PyTorch-native primitive from
this project's own `extract_patches`) for the patch side, and the same transpose/reshape sequence
(re-derived independently in the test file, not imported from `factors.py`) for the gradient side.
`F.unfold`'s patch-channel layout matching `Conv2d.weight`'s own is standard, documented PyTorch
behaviour (`torch.nn.functional.unfold` docs) and is itself cross-checked in the test via the
elementary identity `conv2d(x, W, b) == fold(W.reshape(C_out,-1) @ unfold(x)) + b`
(verified directly, `max abs diff ≈ 1e-4` at `float32`) before being trusted as an oracle.

---

## 1. File-by-file design

```
src/adafisher_modes/
├── factors.py                       # + Conv2d branches: compute_h_full, compute_s_full;
│                                     #   + augment_conv2d_input, augment_input (dispatcher);
│                                     #   + flatten_conv2d_output_grad, flatten_output_grad (dispatcher);
│                                     #   + _check_conv2d_supported
└── approximations/
    ├── _kron_utils.py               # augment_direction: handle weight_direction.ndim > 2
    ├── kfac.py                      # docstring only — no logic change (§0.5)
    ├── ekfac.py                     # augment_linear_input -> augment_input;
    │                                 #   inline s.reshape(...) -> flatten_output_grad
    ├── tkfac.py, tekfac.py          # same two swaps as ekfac.py
    └── __init__.py                  # unchanged (registry already complete since lot 3)
tests/
├── test_full_factors_match_diag.py  # EXTEND — §2.1: Conv2d shapes, gradient-reconstruction
│                                     #   identity, the P-factor quirk regression, groups/dilation
│                                     #   guards, BatchNorm2d/LayerNorm still NotImplementedError
├── test_frobenius_dominance.py      # EXTEND — §2.2: toy-conv replay of §6.1 assertions 1-4
├── test_kfac_ekfac_precondition.py  # EXTEND — §2.3: Conv2d shape/cadence checks
├── test_tkfac_tekfac_precondition.py# EXTEND — §2.3: Conv2d shape/cadence checks
└── test_conv2d_optimizer_smoke.py   # NEW — §2.4: real hooks, real AdaFisherMulti.step(), 4 modes
```

`optimizer.py` and `approximations/base.py` are **unchanged** — `Conv2d` is already in
`SUPPORTED_MODULES` and already hooked since lot 1 (needed for `diag` mode's own `Conv2d` support);
`_check_dim` and the `step()` loop's module/parameter pairing are shape-agnostic already. This is the
same "the ABC/optimizer scaffolding needs nothing" outcome lots 2 and 3 both recorded.
`docs/reports/plan.md` is unchanged (overall plan, not a per-lot log); `CLAUDE.md`'s status header,
file-layout comment and "Running the tests" table are updated at the end, once the exit tests pass,
exactly as after lots 1-3.

### 1.1 `factors.py` additions

```python
def _check_conv2d_supported(layer: Conv2d) -> None:
    if layer.groups != 1:
        raise NotImplementedError(
            f"Full Kronecker factors for Conv2d only support groups=1 so far (lot 4 scope); "
            f"got groups={layer.groups}."
        )
    if layer.dilation != (1, 1):
        raise NotImplementedError(
            f"Full Kronecker factors for Conv2d only support dilation=(1,1) so far (lot 4 scope); "
            f"got dilation={layer.dilation}."
        )


def augment_conv2d_input(h: Tensor, layer: Conv2d) -> Tensor:
    """Pool (batch, output-location) patches onto one axis and, iff `layer` has a bias, append a
    ones column: the Conv2d analogue of `augment_linear_input`, reusing `extract_patches` unmodified
    (lot 1). See plan_lot4.md S0.1 for the i.i.d.-pooled-sample reading this relies on."""
    _check_conv2d_supported(layer)
    patches = extract_patches(h, layer.kernel_size, layer.stride, layer.padding, layer.groups)
    patches = patches.reshape(-1, patches.size(-1))          # (N*S, P): groups=1, so dim 1 is trivial
    if layer.bias is not None:
        return cat([patches, patches.new_ones(patches.size(0), 1)], 1)
    return patches


def augment_input(h: Tensor, layer: Module) -> Tensor:
    """Dispatching sibling of augment_linear_input/augment_conv2d_input (plan_lot4.md S0.5): used
    directly by ekfac/tkfac/tekfac, which need the raw augmented batch, not just compute_h_full's
    reduction (plan_lot2.md S0.3)."""
    if isinstance(layer, Linear):
        return augment_linear_input(h, layer)
    if isinstance(layer, Conv2d):
        return augment_conv2d_input(h, layer)
    raise NotImplementedError(
        f"augment_input only supports Linear and Conv2d so far (lot 4 scope); got {type(layer)}"
    )


def _h_full_conv2d(h: Tensor, layer: Conv2d) -> Tensor:
    h_bar = augment_conv2d_input(h, layer)
    return h_bar.t() @ h_bar / h_bar.size(0)                  # mean over batch * real spatial (S0.2)


def compute_h_full(h: Tensor, layer: Module) -> Tensor:
    if isinstance(layer, Linear):
        return _h_full_linear(h, layer)
    if isinstance(layer, Conv2d):
        return _h_full_conv2d(h, layer)
    raise NotImplementedError(
        f"compute_h_full only supports Linear and Conv2d so far (lot 4 scope); got {type(layer)}"
    )


def flatten_conv2d_output_grad(s: Tensor, layer: Conv2d) -> Tensor:
    """Pool (batch, output-location) gradients onto one axis -- same transpose/reshape sequence as
    _s_conv2d (factors.py, unmodified), minus its diagonal reduction. No S0.2-style quirk here:
    _s_conv2d's spatial_size was already computed correctly (before any reshape)."""
    return s.transpose(1, 2).transpose(2, 3).reshape(-1, s.size(1))


def flatten_output_grad(s: Tensor, layer: Module) -> Tensor:
    """Dispatching sibling, S side. Linear branch is the same inline reshape ekfac/tkfac/tekfac
    already each wrote (extracted here now that a Conv2d branch must exist alongside it anyway)."""
    if isinstance(layer, Linear):
        return s.reshape(-1, s.shape[-1]) if s.ndim > 2 else s
    if isinstance(layer, Conv2d):
        return flatten_conv2d_output_grad(s, layer)
    raise NotImplementedError(
        f"flatten_output_grad only supports Linear and Conv2d so far (lot 4 scope); got {type(layer)}"
    )


def _s_full_conv2d(s: Tensor, layer: Conv2d) -> Tensor:
    s_pool = flatten_conv2d_output_grad(s, layer)
    return s_pool.t() @ s_pool / s_pool.size(0)


def compute_s_full(s: Tensor, layer: Module) -> Tensor:
    if isinstance(layer, Linear):
        return _s_full_linear(s, layer)
    if isinstance(layer, Conv2d):
        return _s_full_conv2d(s, layer)
    raise NotImplementedError(
        f"compute_s_full only supports Linear and Conv2d so far (lot 4 scope); got {type(layer)}"
    )
```

`_check_conv2d_supported` is called from `augment_conv2d_input` only — both `compute_h_full`'s
`Conv2d` branch and `augment_input`'s `Conv2d` branch route through it (§0.5), and `compute_s_full`'s
`Conv2d` branch does not need it (the `S`/output factor has no per-group structure to violate; the
guard belongs on the input/patch side only).

### 1.2 `approximations/_kron_utils.py` — one-line change

```python
def augment_direction(weight_direction: Tensor, bias_direction: Optional[Tensor]) -> Tensor:
    W = weight_direction if weight_direction.ndim == 2 else weight_direction.reshape(weight_direction.size(0), -1)
    if bias_direction is None:
        return W
    return cat([W, bias_direction.unsqueeze(1)], dim=1)
```

`split_direction` is unchanged (already shape-agnostic, §0.3).

### 1.3 `approximations/ekfac.py`, `tkfac.py`, `tekfac.py` — two import/call swaps each

```python
# before (Linear only):
from adafisher_modes.factors import augment_linear_input          # (+ compute_s_full, ekfac only)
...
h_bar = augment_linear_input(h, module)                            # ekfac's update_input_factor
self._cached_h_bar[module] = augment_linear_input(h, module)       # tkfac/tekfac's update_input_factor
...
s_flat = s.reshape(-1, s.shape[-1]) if s.ndim > 2 else s            # all three, in update_output_factor

# after:
from adafisher_modes.factors import augment_input, flatten_output_grad   # (+ compute_s_full, ekfac only)
...
h_bar = augment_input(h, module)
self._cached_h_bar[module] = augment_input(h, module)
...
s_flat = flatten_output_grad(s, module)
```

No other line in any of the three files changes: `A_i = h_bar.t() @ h_bar / h_bar.size(0)` (ekfac),
`instantaneous_raw_factors(h_bar, s_flat)` (tkfac/tekfac), the `s*`/`Θ` intra-batch estimators, the
bootstrap calls, `refresh()`, `f_tilde()`, `precondition()` are all already shape-agnostic over
`d_in_aug`/`d_out` (§0.1, §0.5).

### 1.4 `approximations/kfac.py` — docstring only

Module docstring's `"lot 2 scope (Linear only ...)"` becomes `"Linear and Conv2d with groups=1,
dilation=(1,1) -- see docs/reports/plan_lot4.md"`. No code change (§0.5).

---

## 2. Tests

### 2.1 `test_full_factors_match_diag.py` — extend

New toy layer/input, module-level constants alongside the existing `Linear(7,5)` ones:

```python
CONV_LAYER = lambda bias=True: nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=bias)
CONV_BATCH, CONV_H, CONV_W = 6, 5, 5     # matches conftest.py's tiny_batch shape (6, 2, 5, 5)
```

- **Shapes.** `compute_h_full(x, conv)` is `(19, 19)` (`P+1 = 2*3*3+1`), `compute_s_full(s, conv)` is
  `(3, 3)`, for `x = randn(6, 2, 5, 5)`, `s = randn(6, 3, 5, 5)` (padding=1 keeps spatial size).
  `bias=False` drops the shape to `(18, 18)`.
- **Gradient-reconstruction identity (§0.7b) — the primary correctness anchor.** Real forward +
  backward through `CONV_LAYER()`, reconstruct `weight.grad`/`bias.grad` from
  `flatten_conv2d_output_grad(grad_out, conv).t() @ augment_conv2d_input(x, conv)`, assert
  `allclose(..., atol=1e-4, rtol=1e-4)` against the autograd-computed ground truth, for both
  `bias=True` and `bias=False`.
- **The `_h_conv2d` `P`-factor quirk, locked as a regression (§0.2).**
  `torch.diagonal(compute_h_full(x, conv)) / compute_h_diag(x, conv)` is `allclose` to a constant
  tensor equal to `18.0` (`P`, not `P+1`) — **not** `allclose(compute_h_full.diagonal(),
  compute_h_diag(...))` directly, unlike lot 2's `Linear` assertion, with a comment citing §0.2.
  `torch.diagonal(compute_s_full(s, conv))` **is** `allclose` to `compute_s_diag(s, conv)` (no quirk
  on the `S` side) — asserted directly, mirroring lot 2's `Linear` check.
- **Scope guards.** `Conv2d(4, 4, kernel_size=3, groups=2)` and
  `Conv2d(2, 2, kernel_size=3, dilation=2)` both raise `NotImplementedError` from `compute_h_full`
  and from `augment_input`.
- **Regression.** `BatchNorm2d`/`LayerNorm` still raise `NotImplementedError` from `compute_h_full`/
  `compute_s_full` (lot 2's existing assertion, now read as "not yet — lot 5").

### 2.2 `test_frobenius_dominance.py` — extend, "Lot 4: Conv2d (KFC)" section

New module-level constants, mirroring the existing `D_IN, D_OUT, N` block:

```python
C_IN, C_OUT, KERNEL, PAD, HW, N_CONV = 2, 3, 3, 1, 5, 8    # -> S=25 locations, N_eff=200 >> D_IN_AUG_CONV=19
D_IN_AUG_CONV = C_IN * KERNEL * KERNEL + 1                 # 19
```

`_exact_fisher_block_conv2d(x, s, layer)`: an **independent** oracle (does not call
`extract_patches`/`augment_conv2d_input`/`flatten_conv2d_output_grad`) —

```python
def _exact_fisher_block_conv2d(x, s, layer):
    patches = F.unfold(x, kernel_size=layer.kernel_size, padding=layer.padding)   # (N, P, S) -- torch built-in
    patches = patches.transpose(1, 2).reshape(-1, patches.size(1))                # (N*S, P)
    a = torch.cat([patches, patches.new_ones(patches.size(0), 1)], dim=1)         # (N*S, P+1)
    d = s.transpose(1, 2).transpose(2, 3).reshape(-1, s.size(1))                  # (N*S, C_out)
    F_exact = torch.zeros(C_OUT * D_IN_AUG_CONV, C_OUT * D_IN_AUG_CONV)
    for n in range(a.size(0)):
        F_exact += torch.kron(torch.outer(d[n], d[n]), torch.outer(a[n], a[n]))
    return F_exact / a.size(0)
```

with a one-time self-check (asserted once, not per-call) that `F.unfold`'s layout matches
`Conv2d.weight`'s: `conv2d(x, W, b)` reconstructed from `W.reshape(C_out,-1) @ unfold(x) + b` matches
`layer(x)` directly (§0.7b).

`_build_conv_kfac_and_ekfac(x, s, layer, Lambda)` / `_build_conv_tkfac_and_tekfac(...)`: identical
structure to `_build_kfac_and_ekfac`/`_build_tkfac_and_tekfac` (`gammas=(1.0, 1.0)`, the same
bootstrap-then-redrive sequence for `s*`/`Θ`), except `update_input_factor`/`update_output_factor`
are called with the **raw** `x`/`s` (shape `(N_CONV, C_IN, HW, HW)`/`(N_CONV, C_OUT, HW, HW)`), not
pre-flattened — the real `augment_conv2d_input`/`flatten_conv2d_output_grad` do the pooling.

Six tests, each a direct `_conv2d`-suffixed mirror of an existing `Linear` test, replaying the same
assertion against `F_exact = _exact_fisher_block_conv2d(...)`:
`test_ekfac_dominates_kfac_in_frobenius_norm_conv2d`,
`test_eigenbases_are_orthogonal_conv2d`,
`test_f_tilde_matches_precondition_for_kfac_conv2d`,
`test_f_tilde_matches_precondition_for_ekfac_conv2d`,
`test_tkfac_trace_matches_exact_fisher_conv2d` (`Lambda=1e-12`, same reasoning as the `Linear` version),
`test_tekfac_dominates_tkfac_in_frobenius_norm_conv2d`,
`test_f_tilde_matches_precondition_for_tkfac_conv2d`,
`test_f_tilde_matches_precondition_for_tekfac_conv2d`.

(`test_measured_rescaled_error_report`-style informational logging is *not* replayed for conv — it is
explicitly non-normative in the `Linear` case already, `plan.md` §6.1, and skipping it for conv is not
a loss of coverage.)

### 2.3 `test_kfac_ekfac_precondition.py`, `test_tkfac_tekfac_precondition.py` — extend

For each of the four modes, on `Conv2d(2, 3, kernel_size=3, padding=1, bias={True,False})`:

- **Shape.** `precondition(conv, weight_direction, bias_direction)` returns a tensor shaped
  `(3, 2, 3, 3)` (`bias=False`) or a `(3,2,3,3)`/`(3,)` pair (`bias=True`) — not a flattened
  `(C_out, P)` matrix.
- **Cadence.** `T_inv`/`T_eig`/`T_re` gate refresh identically to the existing `Linear` cadence tests
  (direct `update_input_factor`/`update_output_factor`/`refresh` calls in a loop, `TCov` bypassed —
  same pattern, conv-shaped tensors).
- **Regression.** The full pre-lot-4 body of both files (their existing `Linear` assertions) is
  re-run unmodified.

### 2.4 `test_conv2d_optimizer_smoke.py` — new

Not one of `plan.md` §6.1's four numbered assertions, but the practical claim this lot exists to make
("Conv2d actually works through the real optimizer, not just through direct calls to
`update_input_factor`") is untested by §2.1-§2.3 alone — none of them exercise hook
registration/firing or the `step()` loop's module/parameter pairing for a non-`diag` mode on a
`Conv2d` layer. A small, local `TinyConvNet` (Conv2d → ReLU → Conv2d → Flatten → Linear — **no**
`BatchNorm2d`/`LayerNorm`, since those remain `NotImplementedError` for the four new modes until
lot 5, and `conftest.py`'s existing `TinyMultiLayerNet` fixture has both):

```python
class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * 5 * 5, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(x.flatten(1))
```

For `mode in ("kfac", "ekfac", "tkfac", "tekfac")`: build `AdaFisherMulti(TinyConvNet(), fisher_mode=
mode, TCov=1, T_inv=2, T_eig=2, T_re=1)`, run 6 steps of `forward → cross_entropy → backward →
step()` on `randn(4, 2, 5, 5)` / random integer targets, assert every parameter is finite
(`torch.isfinite(p).all()`) after the loop and that at least one parameter moved from its initial
value by more than a numerical-noise threshold (a real sanity check that preconditioning did
something, not a silent no-op).

---

## 3. Explicitly out of scope for lot 4

- Normalisation layers (`BatchNorm2d`, `LayerNorm`) for `kfac`/`ekfac`/`tkfac`/`tekfac` — lot 5
  (`plan.md` §5.2's "option (c)" full-`S` treatment), unaffected by anything in this lot.
- `groups≠1` (grouped/depthwise `Conv2d`) — §0.4, would need a block-diagonal-per-group derivation.
- `dilation≠(1,1)` — §0.4, `extract_patches` itself has no dilation support (inherited gap, not
  introduced here; not fixed for the diagonal path either).
- TKFAC's `ν`-floor conv damping (Eq. 5.16) and TEKFAC's trace-adaptive conv `λ`/CNN-wide `β`
  rescaling (Eq. 3.5) — §0.6, layered on top of, not part of, this lot's cited theorems.
- The SUA approximation (`EKFAC-pytorch/ekfac.py::_precond_sua_ra`) for memory-tractable
  `ResNet`/`ViT`-scale conv factors — lot 6 (`plan.md` §6.3, §7's "Memory" callout).
- Any change to `optimizer.py`, `approximations/base.py`, `diag.py`, `_tkfac_utils.py`, or lots 1-3's
  existing tests, beyond what §2 adds alongside them.
- Benchmark integration (`benchmarks/mnist_autoencoder.py` has no `Conv2d` layer and is not extended
  to one here) — lot 7's equal-wall-clock comparison.

---

## 4. Points worth flagging explicitly (not blocking, per this session's operating mode)

1. §0.2 — a real, verified (`ratio ≡ 18.0000` across all 19 diagonal entries) scale discrepancy in
   the *existing, untouched* diagonal `Conv2d` path (`_h_conv2d`), shared identically by both
   `AdaFisher` reference repositories, harmless only because `diag` mode's default min-max
   normalisation erases per-layer constant scale factors. The new full-factor code deliberately does
   **not** reproduce it (§0.2's cited reasoning: TKFAC's `tr(F̃)=tr(F)` guarantee needs the real
   scale), so `compute_h_full`'s `Conv2d` diagonal will not match `compute_h_diag`'s — asserted as a
   known, constant-factor (`=P`) discrepancy rather than left as an untested surprise (§2.1).
2. §0.5 — `kfac.py` needs **zero** logic changes to gain `Conv2d` support; only `factors.py` (the
   dispatchers) and `_kron_utils.py` (one line) change underneath it. This is the second consecutive
   lot (after lot 2/3's own "no change to `optimizer.py`/`base.py`") confirming the shared-ABC
   architecture's payoff, worth recording as evidence the design decision in `plan.md` §2.2 was right.
3. §0.6 — the conv-specific *adaptive* damping refinements each paper defines (TKFAC Eq. 5.16, TEKFAC
   Eq. 3.5) are consciously deferred, not silently skipped: none of this lot's cited exit-criteria
   theorems depend on the damping scheme, and speculatively building an untested refinement ahead of
   a real conv training benchmark (lot 7) would be exactly the premature-parameter pattern
   `plan_lot3.md` §0.4 already declined once for `T_fim`.
4. §0.7 — the toy-conv oracle intentionally targets the **pooled-sample** Fisher block (TKFAC
   Assumption 4.1), not a fully spatially-cross-correlated one; this is not a weaker substitute test
   but the literal, correctly-scoped generalisation of what §6.1's `Linear` tests already check,
   argued explicitly to avoid the mistake of "strengthening" the test into checking a different,
   unproven property.
