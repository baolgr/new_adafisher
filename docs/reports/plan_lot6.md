# Lot 6 Implementation Plan — the SUA approximation for `Conv2d`, `kfac`/`ekfac`/`tkfac`/`tekfac`

**Status:** draft, awaiting implementation in this same session. No lot-6 code written yet.
**Parent plan:** `docs/reports/plan.md` §8, lot 6 row: *"SUA approximation for conv layers — §6.1
replayed; memory footprint measured on ResNet-18."* Also §6.3's "Memory — the real blocker on
convolutional nets" callout: `A ∈ R^{4609²}` (85 MB, fp32) for a single ResNet-18 conv factor under
lot 4's patch-based `kfac`/`ekfac`/`tkfac`/`tekfac`, vs. 18 KB for `diag`'s `H_D` vector — the gap this
lot exists to close. This document refines that one-line entry to implementation-ready precision, the
way `plan_lot2.md`/`plan_lot3.md`/`plan_lot4.md`/`plan_lot5.md` did for lots 2-5. It does not revisit
lots 1-5 (closed, 101 tests passing) or lot 7 (equal-wall-clock bench).

**Source note.** The primary source is `docs/papers/kfac_conv_1602.01407.pdf` (Grosse & Martens,
2016, "A Kronecker-factored approximate Fisher matrix for convolution layers"), re-read directly for
this lot (page/section numbers below are from that PDF, extracted with `pdftotext -layout`):
Definitions of IAD/SH/SUD (p. 9, "Here are the approximations we will make..."), Theorem 1 (p. 10),
the "spatially uncorrelated activations (SUA)" definition (p. 14, in the PRONG-comparison
subsection of §4.2), and Theorem 4 (p. 14, combining IAD+SH+SUA+WD), together with the paper's own
empirical caveat in §5.1 (p. 15-16) that SUA "appears to lose a lot of information" relative to the
patch-based KFC this project's lot 4 already implements. The secondary source is
`reference_repos/EKFAC-pytorch/ekfac.py`'s `sua=True` branch (`_precond_sua_ra`, `_precond_intra_sua`,
`_to_kfe_sua`, `_compute_kfe`'s `if group['layer_type']=='Conv2d' and not self.sua` guard,
`_get_gathering_filter`) — adapted, not copied, exactly as `plan.md` §7 prescribes; §0.4 below records
where this lot's design deliberately improves on that reference's own row-alignment handling, and
where it deliberately keeps the reference's own (non-theorem-backed) bias convention.

Exit criteria (`plan.md` §8, lot 6 row, unpacked against §6.1/§6.3, the same way prior lots unpacked
theirs):

1. `kfac`/`ekfac`/`tkfac`/`tekfac` gain a `conv_sua: bool = False` constructor flag. With
   `conv_sua=False` (the default), behaviour is bit-for-bit identical to lots 2-5 — a pure additive
   capability, not a rewrite of the existing patch-based `Conv2d` path.
2. With `conv_sua=True`, the input factor for a `Conv2d` module drops from
   `(C_in·k_h·k_w[+1])²` entries to `(C_in[+1])²` — the "4609 → 512" reduction `plan.md` §6.3 and §7
   cite — measured on a ResNet-18-scale synthetic layer (`Conv2d(512, 512, 3, padding=1)`).
3. On a toy `Conv2d` where the *SUA-consistent* exact Fisher block `F_sua` is computable by an
   independent oracle (not reusing this lot's own extraction code):
   - `‖F_sua − F̃_EKFAC-SUA‖_F ≤ ‖F_sua − F̃_KFAC-SUA‖_F` (EKFAC Thm 2/3) — §6.1 assertion 1, replayed.
   - `‖F_sua − F̃_TEKFAC-SUA‖_F ≤ ‖F_sua − F̃_TKFAC-SUA‖_F` (TEKFAC Thm 3.1) — §6.1 assertion 2, replayed.
   - `tr(F̃_TKFAC-SUA) = tr(F_sua)` (TKFAC Thm 4.1/Lemma 4.1) — §6.1 assertion 3, replayed.
   - `Q_A`, `Q_B`, `Q_Φ`, `Q_Ψ` orthogonal — §6.1 assertion 4, replayed.
   - `precondition()`'s per-kernel-position factored application agrees with `f_tilde()`'s dense
     reconstruction, independently at *every* kernel offset, plus the bias convention (§0.5) — the
     SUA analogue of lots 2-4's own `f_tilde`/`precondition` consistency check.
4. `groups≠1`/`dilation≠(1,1)` still raise the same typed `NotImplementedError` under `conv_sua=True`
   as under `conv_sua=False` (reusing lot 4's existing guard, not a new one).
5. No regression: the full lots 1-5 suite (101 tests) still passes unmodified.
6. A real `AdaFisherMulti(fisher_mode=mode, conv_sua=True, ...)` (hooks, not direct calls) runs a few
   `.step()`s on a `Conv2d`+`Linear` network without error, for all four modes, with finite parameters
   that have moved — the lot-4 smoke-test pattern (`test_conv2d_optimizer_smoke.py`), extended with
   `conv_sua=True`.
7. `plan.md` §5.1's own empirical finding ("SUA loses a lot of information") is *measured*, not
   asserted, on this project's own toy setup — consistent with §6.1's established "measured, not
   asserted" treatment of anything not backed by a theorem (the `diag`-comparison precedent).

---

## 0. Design decisions this lot must resolve before code exists

### 0.1 What SUA is, precisely, and what this lot borrows from it — and what it does not

Grosse & Martens's KFC (already this project's lot 4) rests on three approximations (p. 9):

- **IAD** (independent activations and derivatives): `{a_{j,t}} ⊥ {δs_{i,t'}}`.
- **SH** (spatial homogeneity): first-order statistics of activations are location-independent
  (`E[a_{j,t}] = M(j)`); second-order statistics of activations *and* pre-activation derivatives
  depend only on the offset `t'-t` (`E[a_{j,t}a_{j',t'}] = Ω(j,j',t'-t)`,
  `E[δs_{i,t}δs_{i',t'}] = Γ(i,i',t'-t)`).
- **SUD** (spatially uncorrelated derivatives): `Γ(i,i',δ)=0` for `δ≠0`.

Combined (Theorem 1, p. 10), these give the *exact* KFC factorisation this project's lot 4 already
implements: `F̂_ℓ = Ω_{ℓ-1} ⊗ Γ_ℓ`, with `Ω_{ℓ-1}` a `(C_in·|Δ|+1)×(C_in·|Δ|+1)` matrix built from
patches — the `A` factor `factors.py::_h_full_conv2d`/`augment_conv2d_input` already compute
(estimated directly, per Theorem 2 p. 11, by the patch matrix's own Gram product — no boundary-`β`
formula needed, since the empirical patches already encode padding correctly).

The paper's own further reduction, in its comparison against PRONG (Desjardins et al., 2015, p. 14),
adds a *fourth* approximation:

> **SUA (spatially uncorrelated activations).** The activations at any two distinct spatial
> locations are uncorrelated: `Cov(a_{j,t}, a_{j',t'}) = 0` for `t≠t'`. Combined with SH, write
> `Cov(a_{j,t}, a_{j',t}) = Σ(j,j')`.

and pairs it with a *fifth*, **WD** ("white derivatives": `Γ(i,i',δ) ∝ 1_{i=i'}1_{δ=0}`) to get
Theorem 4's fully-decoupled approximation (`E[Dw_{i,j,δ}Dw_{i',j',δ'}] = β(δ,δ')·Ω̃(j,j',δ'-δ)·1_{i=i'}`,
`Ω̃(j,j',δ) = Σ(j,j')1_{δ=0} + M(j)M(j')`) — the idealised-PRONG target.

**This lot adopts SUA only, not WD.** `Γ` (this project's `B`/output factor) is left exactly as lot
4 built it — a full, channel-only `C_out×C_out` matrix, not the white/scalar-times-identity `Γ` WD
would force. Reasons:

1. `plan.md`'s own goal for this lot (§6.3, §7) is specifically the *input*-factor memory blow-up
   (`4609² = 85 MB`); the output factor is already `C_out×C_out` (≤ a few thousand entries even for
   ResNet-18's widest layer) and was never the stated problem.
2. WD is a much stronger, empirically weaker assumption for this codebase's purpose: it collapses
   *all* structure in `Γ`, whereas AdaFisher/K-FAC's whole premise is exploiting `Γ`'s real
   cross-channel curvature. Nothing in `plan.md` asks for it, and `CLAUDE.md`'s empirical-Fisher
   convention (output gradients carry real, non-white cross-channel signal) argues against it.
3. Every other mode in this codebase (`diag`, the non-SUA `kfac`/`ekfac`/`tkfac`/`tekfac`) keeps `Γ`
   full; dropping it only for `conv_sua=True` would make the four modes' `B` factor mean different
   things depending on a flag, which is a bigger behavioural surprise than this lot should introduce.

So the precise approximation implemented here is **IAD + SH + SUA** (not Theorem 4's IAD+SH+SUA+WD):

```
E[Dw_{i,j,δ} Dw_{i',j',δ'}] = β(δ,δ') · [ Σ(j,j') 1_{δ=δ'} + M(j)M(j') ] · Γ(i,i',0)
```

Two consequences, both honestly *not* what a literal reading of this formula would require, and both
recorded here rather than silently assumed:

- The `M(j)M(j')` term is **nonzero for every `δ,δ'` pair**, not just `δ=δ'` — SUA zeroes the
  *centred* covariance, not the raw second moment. A mathematically exact treatment would therefore
  make `Ω_SUA` a *sum* of two Kronecker terms, `β·(Σ⊗I_{|Δ|}) + β·(M⊗𝟙_{|Δ|})(M⊗𝟙_{|Δ|})ᵀ` (the
  second term rank 1, since `M` is a vector — invertible in closed form via Sherman-Morrison). This
  lot does **not** implement that correction. Instead — §0.4 below — it treats `Ω_SUA` as exactly
  block-diagonal across `δ` (dropping the rank-1 cross-`δ` mean term), matching
  `EKFAC-pytorch::_precond_sua_ra`'s own convention (broadcast the *bias* companion channel, not this
  weight-weight mean term, across positions; see below). This is a real, further approximation beyond
  what IAD+SH+SUA alone would produce exactly — flagged here rather than left implicit, and consistent
  with how this project has always drawn a hard line between what a theorem gives verbatim and what a
  tractable implementation additionally simplifies (lot 5's `a_ν` surrogate is the precedent).
- The boundary weight `β(δ,δ')` is, as in lot 4, not computed via its combinatorial formula (Eq. 24);
  it is estimated empirically from real (correctly zero-padded) data, which handles boundary effects
  automatically — see §0.3.

### 0.2 Why the output factor needs zero changes

`Γ`/`B` is already `factors.py::_s_full_conv2d`/`flatten_conv2d_output_grad`: a plain
`(batch·H_out·W_out, C_out)` pooling with **no patch extraction anywhere on the output side** — SUA
only concerns the *input* factor. `compute_s_full`, `flatten_output_grad`, and every approximation
class's `update_output_factor` are therefore untouched by this lot.

### 0.3 The new input factor: pool by channel, not by patch — and *why the center-of-patch slice, not
independent raw-pixel pooling*, is the right way to build it

`EKFAC-pytorch`'s own SUA branch (`_compute_kfe`, `if group['layer_type']=='Conv2d': if not
self.sua: ... x = F.conv2d(..., gathering_filter, ...)`, i.e. skip the patch-gathering when
`sua=True`) estimates `Σ` by pooling the **raw, un-unfolded input** `x` over `(N, H_in, W_in)`
directly: `xxt = x.permute(1,0,2,3).reshape(C_in,-1) @ ... / (N·H_in·W_in)`.

This project builds `Σ` differently, and better: **slice the *center* offset out of every patch this
project's existing `extract_patches` already produces**, instead of pooling the raw input
independently. Concretely, `augment_conv2d_input_sua` calls the same `extract_patches` lot 1/4
already use, reshapes each patch's last dimension from `C_in·k_h·k_w` back to `(C_in, k_h, k_w)`, and
keeps only the `(⌊k_h/2⌋, ⌊k_w/2⌋)` slice:

```python
patches = extract_patches(h, layer.kernel_size, layer.stride, layer.padding, layer.groups)
patches = patches.reshape(-1, patches.size(-1))              # (N*S, C_in*k_h*k_w)
center = patches.view(-1, layer.in_channels, kh, kw)[:, :, kh // 2, kw // 2]   # (N*S, C_in)
```

Why this is not just "simpler" but **more correct** for this codebase's own conventions:

- **Row alignment.** `extract_patches` produces exactly `S = H_out·W_out` patches per example, in
  `(H_out, W_out)`-order — the *same* grid and the *same* row order (`N` outer, `H_out`, `W_out` inner)
  that `flatten_conv2d_output_grad` already uses for `Γ`'s pooling (verified by direct index-order
  inspection of both functions, §1.1). `EKFAC-pytorch`'s "pool the raw input independently" approach
  pools over `(N, H_in, W_in)` — a *different* grid size from `(N, H_out, W_out)` whenever
  `H_in≠H_out` (any conv with `padding` not equal to `(k-1)/2`, or `stride>1`). This project's every
  other Conv2d branch (lot 4's `A`, `B`, and the `ekfac`/`tekfac` "intra-batch" statistics that need
  `h_bar` and `s_flat` to be *row-paired* per `(example, output-location)`) already depends on this
  `(N,H_out,W_out)` alignment. Center-slicing a patch preserves it automatically, for *any*
  `stride`/`padding` (not just the common "same" case) — the naive independent-pooling approach
  would silently misalign `ekfac`'s/`tekfac`'s intra-batch `s*`/`Θ` estimator (§0.4) whenever
  `H_in≠H_out`, which is why `EKFAC-pytorch` only exposes `_precond_sua_ra` (a running-average
  estimator that never needs row-pairing) for its intra-batch analogue, and implements a separate,
  more complex, real-convolution-based routine (`_precond_intra_sua`/`grad_wrt_kernel`) instead of
  simple row-pairing.
- **Validity under SH.** SH says the *statistics* of `a_{j,t}` are location-independent, so sampling
  from any one fixed spatial reference is a valid empirical estimator of `Σ(j,j')`, `M(j)` — the
  paper's own Theorem 2 already licenses estimating population statistics from *any* consistent
  pooling of the real, padded data. Center-slicing is exactly that: one particular (very convenient)
  choice of reference offset.
- **Coincidence with the reference in the common case, worth recording.** For `stride=(1,1)` and
  `padding=((k_h-1)/2, (k_w-1)/2)` (odd kernel, "same" padding — the overwhelmingly common case in
  practice, and the case used by every `Conv2d` in this codebase's own tests/fixtures:
  `TinyMultiLayerNet.conv`, `TinyConvNet.conv1/conv2`, all `kernel_size=3, padding=1`), the padded
  input's centre pixel for output location `t` is *exactly* `x[t]` itself (no padding-zero ever
  contributes — `pad` adds exactly `(k-1)/2` on each side, so the receptive field's centre always maps
  onto a real, unpadded pixel, verified index-for-index in §1.1's test). In that regime this lot's
  `Σ` is **numerically identical** to `EKFAC-pytorch`'s own raw-pixel pooling — this lot's approach is
  a strict generalisation that happens to coincide with the reference exactly where the reference is
  well-behaved, and stays correct (row-aligned) where the reference's approach would not even apply.

`augment_conv2d_input_sua` reuses `_check_conv2d_supported` (lot 4, unchanged) — `groups≠1`/
`dilation≠(1,1)` raise the same typed `NotImplementedError` as the non-SUA path, since SUA inherits
`extract_patches`'s own scope restriction rather than defining a new one.

### 0.4 Applying the preconditioner: block-diagonal across kernel offsets, not a single flat matmul

Lots 2-5's `precondition()` bodies all share one shape: flatten the weight (+ bias) direction into
one `(C_out, d_in_aug)` matrix (`_kron_utils.augment_direction`) and apply a *single* `B⁻¹ (·) A⁻¹`-
shaped operator to it. Under SUA, `A` (now `(C_in[+1])×(C_in[+1])`) is **smaller** than the weight's
own flattened width (`C_in·k_h·k_w[+1]`) — the same flat matmul is not even shape-compatible anymore.

The exact-under-IAD+SH+SUA structure (§0.1, dropping the rank-1 cross-`δ` mean term per that
section's flagged simplification) is **block-diagonal across the `k_h·k_w` kernel offsets**: apply
the *same* small `B⁻¹ (·) A⁻¹`-shaped operator independently to each of the `k_h·k_w` slices
`M[:, :, k_h_i, k_w_i]` of the real `(C_out, C_in, k_h, k_w)` weight direction. Concretely (new
`_kron_utils.augment_conv2d_direction_sua`/`split_conv2d_direction_sua`, §1.2):

```
M_sua = permute/reshape(weight_direction)            # (P, C_out, C_in),  P = k_h*k_w
direction = B⁻¹ @ M_sua @ A⁻¹                        # torch's batched-matmul broadcasting: B⁻¹, A⁻¹
                                                       # are 2-D, M_sua is (P, C_out, C_in) — broadcast
                                                       # over P automatically, no explicit loop needed
weight_out = permute/reshape(direction)               # back to (C_out, C_in, k_h, k_w)
```

**Bias.** The bias parameter is a single, position-independent `(C_out,)` vector — it has no `δ`
index of its own. Following `EKFAC-pytorch::_precond_sua_ra`'s own convention (there is no
theorem-backed alternative; §0.1 already flagged this whole block-diagonal step as a further
approximation beyond the cited theorems): the bias direction is **broadcast** into every one of the
`P` position-slices as an extra, `δ`-independent companion column (`M_sua[:, :, C_in] = bias`,
identical at every position — mirroring how `augment_conv2d_input_sua` appends a `δ`-independent ones
column to `Σ`), so every position-slice becomes `(C_out, C_in+1)`. After preconditioning, this leaves
`P` different candidate values for "the preconditioned bias gradient" (one per position, since the
input companion channel's contribution mixes with each position's own — genuinely different — weight
content). **This lot picks the center position's answer**, `(⌊k_h/2⌋, ⌊k_w/2⌋)`, exactly
`EKFAC-pytorch`'s own choice (`gb = g_nat[:, -1, s[2]//2, s[3]//2]`) — and, worth noting, now for a
*principled* reason this project's own design supplies rather than an arbitrary pick: it is the same
reference offset §0.3 already uses to build `Σ` itself, so "the center position's bias estimate" is
the estimate computed in the same coordinate frame as the statistics that produced the preconditioner
applied to it. Internal consistency, not a new invention.

### 0.5 What changes, file by file — and the running "zero-change" tally

Given §0.2-§0.4, the change footprint is:

| File | Change |
|---|---|
| `factors.py` | + `augment_conv2d_input_sua`, `_h_full_conv2d_sua`; `compute_h_full`/`augment_input` gain an optional `sua: bool = False` parameter, consulted **only** on their `Conv2d` branch |
| `approximations/_kron_utils.py` | + `augment_conv2d_direction_sua`, `split_conv2d_direction_sua` |
| `approximations/kfac.py` | + `conv_sua` constructor flag; `update_input_factor` passes `sua=self.conv_sua`; `precondition` branches on `isinstance(module, Conv2d) and self.conv_sua` |
| `approximations/ekfac.py` | same three changes as `kfac.py` |
| `approximations/tkfac.py` | same three changes |
| `approximations/tekfac.py` | same three changes |
| `optimizer.py`, `approximations/__init__.py`, `approximations/base.py`, `diag.py`, `_tkfac_utils.py`, `ema.py`, `minmax.py` | **unchanged** |

The last row continues a running theme this project's own `CLAUDE.md` already tracks (lot 4: "no
change to `optimizer.py`/`base.py`"; lot 5: "zero changes to any `approximations/*.py` file"). Lot 6
is the first lot since lot 1 that *does* touch every `approximations/*.py` file — necessarily, since
SUA changes the *shape* of the direction application, not just which numbers flow through the
existing shape (unlike lot 5, where the normalisation-layer `A` stayed a plain 2×2 matrix matching
the existing 2-column direction). This is recorded explicitly rather than left as a silent departure
from the pattern the status banner otherwise celebrates. `refresh()` and `f_tilde()` need **no**
change in any of the four files: both already operate purely on `self._A[module]`/`self._B[module]`
(or `Phi_raw`/`Psi_raw`/`Q_Phi`/`Q_Psi`/`Theta`), agnostic to whether those came from patches or from
`Σ_sua` — only their *size* changes. `update_output_factor` also needs no change in any file, per
§0.2.

### 0.6 The `k_h=k_w=1` degeneracy — a strong, cheap internal-consistency check

When the kernel is `1×1`, `extract_patches`'s patches already *are* channel-only (`C_in·1·1 =
C_in`), so `augment_conv2d_input_sua` and `augment_conv2d_input` compute the exact same tensor, and
`P=1` makes the "which position for bias" question vacuous. `conv_sua=True` and `conv_sua=False`
must therefore agree exactly on a `1×1` `Conv2d` — a cheap, high-confidence regression test with a
clear mathematical justification (§2.1), not just a smoke check.

### 0.7 Measuring, not asserting, the paper's own "SUA loses information" finding

`plan.md` §6.1's methodology already distinguishes theorem-backed assertions from measured-only
comparisons (the `diag`-vs-four-modes rescaled-Frobenius report). This lot's SUA-vs-full-KFC
comparison is the second instance of that pattern: Grosse & Martens's own §5.1 (p. 15-16) reports
that SUA "loses a lot of information" empirically; the honest way to carry that into this project's
own toy setup is to **measure** it, not to assert a threshold that would be arbitrary and could make
the test suite flaky against a different random seed.

The clean, size-consistent measurement (§2.2): take the *exact*, independently-computed, patch-based
input covariance `Ω_full` (lot 4's own oracle, restricted to its weight-weight block, size
`(C_in·k_h·k_w)²`) and its exact **block-diagonal-across-`δ`-offset projection** (zero every
`(c,δ),(c',δ')` entry with `δ≠δ'` — precisely the structure §0.4's block-diagonal approximation
targets). Report `‖Ω_full − Ω_block-diagonal‖_F / ‖Ω_full‖_F`: the fraction of the true input
covariance's Frobenius mass that lives in the cross-position correlations SUA discards. This needs no
embedding of the small `Σ_sua` operator into the big patch space, stays at one consistent scale, and
is a direct, independently-computed measurement of exactly the effect the paper describes — printed,
plus a non-degeneracy sanity assertion (`> 0`, i.e. this toy random input actually has *some*
cross-position correlation, so the measurement is not vacuous) rather than a threshold assertion on
the fraction's value.

---

## 1. File-by-file design

### 1.1 `factors.py` additions

```python
def augment_conv2d_input_sua(h: Tensor, layer: Conv2d) -> Tensor:
    """Channel-only analogue of augment_conv2d_input (SUA, kfac_conv_1602.01407.pdf p. 14): pools
    the *center* offset of every receptive-field patch instead of the whole patch, dropping the
    input factor from (C_in*k_h*k_w[+1])^2 to (C_in[+1])^2 (docs/reports/plan_lot6.md §0.3). Row-
    aligned with flatten_conv2d_output_grad's (N*S, C_out) pooling by construction, since both are
    built from extract_patches's own (H_out, W_out) grid.
    """
    _check_conv2d_supported(layer)
    kh, kw = layer.kernel_size
    patches = extract_patches(h, layer.kernel_size, layer.stride, layer.padding, layer.groups)
    patches = patches.reshape(-1, patches.size(-1))  # (N*S, C_in*kh*kw)
    center = patches.view(-1, layer.in_channels, kh, kw)[:, :, kh // 2, kw // 2]  # (N*S, C_in)
    if layer.bias is not None:
        return cat([center, center.new_ones(center.size(0), 1)], 1)
    return center


def _h_full_conv2d_sua(h: Tensor, layer: Conv2d) -> Tensor:
    h_bar = augment_conv2d_input_sua(h, layer)
    return h_bar.t() @ h_bar / h_bar.size(0)
```

`compute_h_full` and `augment_input` each gain one optional parameter, consulted only on the `Conv2d`
branch (every other branch is untouched — `sua` is meaningless for `Linear`/`BatchNorm2d`/
`LayerNorm`, so it is silently ignored there rather than rejected, matching how `pi`/`T_inv`/`T_eig`
already are meaningless-but-harmless for layer types those knobs don't apply to):

```python
def compute_h_full(h: Tensor, layer: Module, sua: bool = False) -> Tensor:
    if isinstance(layer, Linear):
        return _h_full_linear(h, layer)
    if isinstance(layer, Conv2d):
        return _h_full_conv2d_sua(h, layer) if sua else _h_full_conv2d(h, layer)
    if isinstance(layer, (BatchNorm2d, LayerNorm)):
        return _h_full_norm(h, layer)
    raise NotImplementedError(...)


def augment_input(h: Tensor, layer: Module, sua: bool = False) -> Tensor:
    if isinstance(layer, Linear):
        return augment_linear_input(h, layer)
    if isinstance(layer, Conv2d):
        return augment_conv2d_input_sua(h, layer) if sua else augment_conv2d_input(h, layer)
    if isinstance(layer, (BatchNorm2d, LayerNorm)):
        return augment_norm_input(h, layer)
    raise NotImplementedError(...)
```

`compute_s_full`, `flatten_output_grad`, `SUPPORTED_MODULES` — unchanged (§0.2).

### 1.2 `approximations/_kron_utils.py` additions

```python
def augment_conv2d_direction_sua(
    weight_direction: Tensor, bias_direction: Optional[Tensor]
) -> Tensor:
    """(C_out, C_in, k_h, k_w) [+ bias, broadcast identically into every position] ->
    (k_h*k_w, C_out, C_in[+1]) -- one independent (C_out, C_in_aug) slice per kernel offset,
    ordered (k_h, k_w) row-major to match split_conv2d_direction_sua's inverse reshape
    (docs/reports/plan_lot6.md §0.4).
    """
    c_out, c_in, kh, kw = weight_direction.shape
    M = weight_direction.permute(2, 3, 0, 1).reshape(kh * kw, c_out, c_in)
    if bias_direction is None:
        return M
    b = bias_direction.reshape(1, c_out, 1).expand(kh * kw, c_out, 1)
    return cat([M, b], dim=2)


def split_conv2d_direction_sua(
    M: Tensor, weight_shape: Size, bias_shape: Optional[Size]
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Inverse of augment_conv2d_direction_sua. The bias estimate is taken from the *center*
    kernel offset (plan_lot6.md §0.4) -- the same reference offset augment_conv2d_input_sua uses
    to build Sigma, not an arbitrary choice.
    """
    c_out, c_in, kh, kw = weight_shape
    W = M[:, :, :c_in].reshape(kh, kw, c_out, c_in).permute(2, 3, 0, 1)
    if bias_shape is None:
        return W
    center = (kh // 2) * kw + (kw // 2)
    bias = M[center, :, -1].reshape(bias_shape)
    return W, bias
```

### 1.3 `approximations/kfac.py`

```python
def __init__(self, Lambda=1e-3, gammas=(0.92, 0.008), T_inv=100, pi=True, conv_sua=False):
    ...
    self.conv_sua = conv_sua

def update_input_factor(self, module, h, step):
    A_i = compute_h_full(h, module, sua=self.conv_sua)
    ...  # unchanged otherwise

def precondition(self, module, weight_direction, bias_direction):
    if self.conv_sua and isinstance(module, Conv2d):
        M = augment_conv2d_direction_sua(weight_direction, bias_direction)
        direction = self._B_inv[module] @ M @ self._A_inv[module]
        bias_shape = None if bias_direction is None else bias_direction.shape
        return split_conv2d_direction_sua(direction, weight_direction.shape, bias_shape)
    M = augment_direction(weight_direction, bias_direction)
    direction = self._B_inv[module] @ M @ self._A_inv[module]
    bias_shape = None if bias_direction is None else bias_direction.shape
    return split_direction(direction, weight_direction.shape, bias_shape)
```

`update_output_factor`, `_pi`, `_damped_factors`, `refresh`, `f_tilde` — unchanged (§0.2, §0.5). New
import: `from torch.nn import Conv2d, Module`.

### 1.4 `approximations/ekfac.py`

Same three changes as `kfac.py`'s shape: `conv_sua` flag; `update_input_factor`'s
`h_bar = augment_input(h, module, sua=self.conv_sua)`; `precondition`'s branch, using
`Q_A`/`Q_B`/`self._s_star[module]` in place of `A_inv`/`B_inv`:

```python
def precondition(self, module, weight_direction, bias_direction):
    Q_A, Q_B = self._Q_A[module], self._Q_B[module]
    if self.conv_sua and isinstance(module, Conv2d):
        M = augment_conv2d_direction_sua(weight_direction, bias_direction)
        M_kfe = (Q_B.t() @ M @ Q_A) / (self._s_star[module] + self.Lambda)
        direction = Q_B @ M_kfe @ Q_A.t()
        bias_shape = None if bias_direction is None else bias_direction.shape
        return split_conv2d_direction_sua(direction, weight_direction.shape, bias_shape)
    M = augment_direction(weight_direction, bias_direction)
    M_kfe = (Q_B.t() @ M @ Q_A) / (self._s_star[module] + self.Lambda)
    direction = Q_B @ M_kfe @ Q_A.t()
    bias_shape = None if bias_direction is None else bias_direction.shape
    return split_direction(direction, weight_direction.shape, bias_shape)
```

`update_output_factor`'s intra-batch `s*` estimator is untouched: `h_bar` (cached from
`update_input_factor`) is already `(N·S, C_in[+1])`-shaped and row-aligned with `s_flat` regardless of
`conv_sua` (§0.3), so `h_kfe = h_bar @ self._Q_A[module]` and the `g2` outer product work unchanged —
`Q_A` is simply smaller.

### 1.5 `approximations/tkfac.py`, `approximations/tekfac.py`

Same pattern: `conv_sua` flag; `update_input_factor` caches
`augment_input(h, module, sua=self.conv_sua)`; `precondition` branches exactly as `ekfac.py`'s, using
`Psi_inv`/`Phi_inv`/`delta_at_refresh` (`tkfac`) or `Q_Psi`/`Q_Phi`/`Theta` (`tekfac`) in place of
`Q_B`/`Q_A`/`s_star`. `instantaneous_raw_factors`/`bootstrap_raw_factors` in `_tkfac_utils.py` need no
change (already shape-agnostic, confirmed by lot 4's own docstring note reused verbatim here).

---

## 2. Tests

### 2.1 `test_full_factors_match_diag.py` — new "Lot 6: SUA" section

- `test_augment_conv2d_input_sua_shape_with_without_bias` — output width `C_in+1` / `C_in`.
- `test_augment_conv2d_input_sua_center_slice_equals_raw_pixel` — on the "same"-padding,
  `stride=1`, odd-kernel toy config (`CONV_PAD=(CONV_K-1)//2`), asserts
  `augment_conv2d_input_sua(h, layer)[:, :-1]` (the non-bias columns) equals `h` itself, permuted to
  `(N*H*W, C_in)` the same way `flatten_conv2d_output_grad` would — the concrete, numeric version of
  §0.3's derivation, not just an argued claim.
  `test_augment_conv2d_input_sua_row_alignment_with_output_grad` — same fixture, asserts
  `augment_conv2d_input_sua(h, layer).size(0) == flatten_conv2d_output_grad(s, layer).size(0)` for `h`,
  `s` with mismatched spatial size relative to each other being impossible by construction (both
  derived from the *same* `H_out, W_out`) — regression guard for §0.3's row-alignment claim under a
  **non**-"same" padding/stride config too (e.g. `stride=2`, `padding=0`), where `H_in≠H_out`.
- `test_augment_conv2d_input_sua_matches_full_for_1x1_kernel` — §0.6's degeneracy: for
  `kernel_size=1`, `augment_conv2d_input_sua(h, layer)` and `augment_conv2d_input(h, layer)` are
  `torch.equal`.
- `test_augment_conv2d_input_sua_groups_dilation_not_supported` — reuses `_check_conv2d_supported`;
  same typed `NotImplementedError` as the non-SUA path.

### 2.2 `test_frobenius_dominance.py` — new "Lot 6: SUA" section

Reuses `CONV_C_IN=2, CONV_C_OUT=3, CONV_K=3, CONV_PAD=1, CONV_HW=5, CONV_N=8` (lot 4's own constants —
`padding=(k-1)/2`, `stride=1`, so §0.3's center-slice-equals-raw-pixel identity holds exactly, letting
the oracle be written by simple direct pixel pooling without reimplementing `extract_patches`'s
padding logic). `CONV_SUA_D_IN_AUG = CONV_C_IN + 1` (`= 3`, vs. lot 4's `CONV_D_IN_AUG = 19`).

```python
def _exact_fisher_block_conv2d_sua(x: Tensor, s: Tensor) -> Tensor:
    """Independent oracle: pools raw x, s directly over (N, H, W) -- valid here specifically
    because padding=(k-1)/2, stride=1 makes this identical to center-slicing every patch
    (plan_lot6.md §0.3), verified separately in §2.1's own numeric test.
    """
    a = x.permute(0, 2, 3, 1).reshape(-1, CONV_C_IN)
    a = torch.cat([a, a.new_ones(a.size(0), 1)], dim=1)
    d = s.permute(0, 2, 3, 1).reshape(-1, CONV_C_OUT)
    F_exact = torch.zeros(CONV_C_OUT * CONV_SUA_D_IN_AUG, CONV_C_OUT * CONV_SUA_D_IN_AUG)
    for n in range(a.size(0)):
        F_exact += torch.kron(torch.outer(d[n], d[n]), torch.outer(a[n], a[n]))
    return F_exact / a.size(0)
```

- `test_ekfac_sua_dominates_kfac_sua_in_frobenius_norm`, `test_tekfac_sua_dominates_tkfac_sua_...`,
  `test_tkfac_sua_trace_matches_exact_fisher`, `test_eigenbases_are_orthogonal_conv2d_sua` — the four
  §6.1 assertions, built via `KFACApproximation(conv_sua=True)` etc. driven directly
  (`update_input_factor`/`update_output_factor`/`refresh`, exactly like the lot 2-5 builders), and
  `f_tilde(layer)` compared against `_exact_fisher_block_conv2d_sua` — both now the *same*,
  `CONV_C_OUT*CONV_SUA_D_IN_AUG`-sized dense matrix (no embedding needed; §0.7).
- `test_f_tilde_matches_precondition_for_kfac_conv2d_sua` (+ `ekfac`/`tkfac`/`tekfac` variants) — the
  per-position consistency check §0.4/exit-criterion 3 calls for: for a random
  `(CONV_C_OUT, CONV_C_IN, CONV_K, CONV_K)` weight direction and `(CONV_C_OUT,)` bias direction, loop
  over all `CONV_K*CONV_K` positions and assert `precondition()`'s per-position weight slice matches
  `torch.linalg.solve(f_tilde(layer), M_p.flatten())` for **that position's** `M_p = [weight[:, :,
  kh_i, kw_i] | bias]`; separately assert the returned bias matches the solve's bias column
  **at the center position only** (§0.4's convention, not every position — the other `P-1` positions'
  solved bias columns are expected to *disagree*, which is a fact about the approximation, not a bug).
- `test_conv2d_sua_matches_full_for_1x1_kernel` — §0.6: build both a `conv_sua=True` and a
  `conv_sua=False` `KFACApproximation`/`EKFACApproximation`/`TKFACApproximation`/`TEKFACApproximation`
  on the *same* `Conv2d(C_in, C_out, kernel_size=1, bias=True)` and the *same* `(h, s)`, and assert
  `precondition()` returns `torch.allclose` weight/bias directions for both.
- `test_sua_discards_offblock_frobenius_mass` (§0.7, informational): build `Ω_full` (weight-weight
  block only, via `torch.nn.functional.unfold`, independent of `extract_patches`) and its
  block-diagonal-across-position projection; print
  `‖Ω_full − Ω_block-diagonal‖_F / ‖Ω_full‖_F`; assert only `> 1e-6` (non-degeneracy: this toy input
  really does have cross-position correlation for SUA to discard).
- `test_sua_memory_footprint_resnet18_scale` — `Conv2d(512, 512, 3, padding=1, bias=True)`; asserts
  `augment_conv2d_input_sua(h, layer).size(1) == 513` vs.
  `augment_conv2d_input(h, layer).size(1) == 4609` (exact, matching `plan.md` §6.3/§7's cited figures
  to the entry), and reports (prints) the implied per-factor byte reduction
  `(4609/513)**2 ≈ 80.7×`.

### 2.3 `test_kfac_ekfac_precondition.py`, `test_tkfac_tekfac_precondition.py` — new "Lot 6: SUA" section

Mirrors the existing `Conv2d` sections (`test_precondition_shapes_with_bias_conv2d`,
`test_kfac_inverse_cadence_respects_t_inv_conv2d`, etc.) with `conv_sua=True` passed to the
constructor: output shapes (`(C_out, C_in, k_h, k_w)` / `(C_out,)`, unchanged from the non-SUA case —
`precondition`'s *output* shape is identical regardless of `conv_sua`, only its internal factor sizes
differ), and the `T_inv`/`T_eig`/`T_re` amortisation cadence (unaffected by `conv_sua`, since
`refresh()` is untouched — these tests mainly guard against a future edit accidentally coupling the
two).

### 2.4 `test_conv2d_sua_optimizer_smoke.py` — new

Real `AdaFisherMulti(model, fisher_mode=mode, conv_sua=True, TCov=1, ...)`, hooks + `step()`, on
`TinyConvNet` (imported from `test_conv2d_optimizer_smoke.py` — same net, no duplication), for all
four modes: finite parameters, at least one parameter moved — the lot-4 pattern, extended with
`conv_sua=True`. Plus one test confirming the flag is properly scoped: `conv_sua=True` on a
`Conv2d`-free network (a plain two-`Linear`-layer net) produces the exact same parameter trajectory as
`conv_sua=False`, for all four modes, from identical seeds — demonstrating the flag is inert wherever
there is no `Conv2d` to apply it to (§1.1: `sua` is silently ignored on `Linear`'s branch).

---

## 3. Explicitly out of scope for lot 6

- The rank-1 cross-`δ` mean-coupling correction to `Ω_SUA` that an exact IAD+SH+SUA derivation would
  include (§0.1) — flagged, not implemented; matches `EKFAC-pytorch`'s own (also uncorrected)
  convention.
- White derivatives (WD) / Theorem 4's fully-decoupled Fisher — deliberately not adopted (§0.1);
  `Γ` stays full for every mode, SUA or not.
- `groups≠1` / `dilation≠(1,1)` — unchanged scope restriction from lot 4, reused verbatim (§0.3).
- Per-layer `conv_sua` (some `Conv2d` modules SUA, others patch-based, within the same optimizer
  instance) — `conv_sua` is one flag per `FisherApproximation` instance, applied uniformly to every
  `Conv2d` module it manages, matching both `EKFAC-pytorch`'s own `sua` flag (instance-wide) and this
  project's existing single-mode-per-optimizer design (`plan.md` §2.3: "the mode is a configuration
  field, not a separate optimizer").
- `BatchNorm2d`/`LayerNorm` — untouched; SUA is a `Conv2d`-only concept (there is no patch dimension
  to collapse for a normalisation layer, whose lot-5 input factor is already the smallest possible,
  `2×2`).
- Benchmark integration (`benchmarks/mnist_autoencoder.py` has no `Conv2d` layer) — lot 7's
  equal-wall-clock comparison, and/or the deferred CIFAR-10/ResNet work `plan.md` §6.3 gates on this
  lot.
- Actually instantiating `torchvision.models.resnet18` for the memory-footprint test — a synthetic
  `Conv2d(512, 512, 3, padding=1)` (ResNet-18's largest conv factor's exact dimensions, per `plan.md`
  §6.3) suffices and avoids adding a hard `torchvision` test dependency (it is currently an optional
  `bench`-only extra, `pyproject.toml`).

---

## 4. Points worth flagging explicitly (not blocking, per this session's operating mode)

1. §0.5's tally is the first time since lot 1 that every `approximations/*.py` file changes in the
   same lot — worth recording as the boundary case of the shared-ABC architecture's "usually free"
   extensibility: SUA changes the *shape* of the direction application (patch-sized vs.
   channel-sized), which `_kron_utils.py`'s existing flat-matmul contract cannot express, unlike lot
   5's normalisation layers (which stayed inside that contract by construction).
2. §0.3's center-slice construction is a genuine, if modest, improvement on `EKFAC-pytorch`'s own
   SUA implementation, not just an adaptation of it: it is provably row-aligned with the output-side
   pooling for *any* `stride`/`padding` (not only "same" padding, `stride=1`), which is exactly what
   lets this lot reuse `ekfac`'s/`tekfac`'s existing intra-batch `s*`/`Θ` estimator verbatim, with no
   `Conv2d`-SUA-specific branch inside `update_output_factor` at all. Worth keeping if a future lot
   ever needs `stride>1` `Conv2d` support broadly (currently untested here, but not excluded by any
   guard either — `_check_conv2d_supported` only restricts `groups`/`dilation`).
3. §0.4's bias convention (pick the center position) is inherited from `EKFAC-pytorch` because there
   is no theorem to derive a better one from — this project's own contribution is only the
   *justification* (same reference offset as `Σ`'s own construction), not a different formula. If this
   ever needs revisiting, the honest fix is deriving the Sherman-Morrison correction from §0.1's
   flagged rank-1 term (which *would* couple all `P` positions' bias estimates into one consistent
   value) rather than picking a different single position.
