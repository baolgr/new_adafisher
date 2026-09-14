# Lot 3 Implementation Plan — `tkfac.py`, `tekfac.py` (`Linear` only)

**Status:** draft, awaiting implementation in this same session. No lot-3 code written yet.
**Parent plan:** `docs/reports/plan.md` §8, lot 3. This document refines lot 3 to
implementation-ready precision, the way `plan_lot2.md` did for lot 2. It does not revisit lots 1-2
(closed, lot 2 exit criteria met) or lots 4-7.

**Source note.** `CLAUDE.md` points at `papers/`; in this checkout the PDFs actually live at
`docs/papers/` (`tkfac_2011.10741.pdf`, `tekfac_2011.13609.pdf`). Every equation/theorem number
below was re-read from those two PDFs directly for this lot (pages 1-18 of the TKFAC paper, pages
1-10 of the TEKFAC paper) — not taken on faith from `CLAUDE.md`'s cheat sheet, which is itself a
condensed reminder per that file's own "Hard constraints" section. Two corrections/refinements
against the cheat sheet's shorthand fall out of that re-read and are recorded in §0.3 and §0.5.

Exit criteria (`plan.md` §8, lot 3 row, unpacked against §6.1/§6.2):

1. `tr(F̃_TKFAC) = tr(F)` (TKFAC Thm 4.1 / Lemma 4.1), on the same toy per-sample-exact `F` lot 2
   already validates against.
2. `‖F − F̃_TEKFAC‖_F ≤ ‖F − F̃_TKFAC‖_F` (TEKFAC Thm 3.1).
3. EKFAC ↔ TEKFAC consistency in the degenerate case `Φ:=A, Ψ:=B, δ:=1` (`plan.md` §6.2).
4. No regression: the full lot-1/lot-2 suite still passes unmodified.

---

## 0. Design decisions this lot must resolve before code exists

### 0.1 What TKFAC actually asks the codebase to compute (Eq. 4.2, 4.9)

TKFAC's Theorem 4.1 (verified, `tkfac_2011.10741.pdf` p. 8, Eq. 4.2-4.9) decomposes the per-layer
Fisher block as

```
F_l = E[Λ_{l-1} ⊗ Γ_l],   Λ_{l-1} = a_{l-1}a_{l-1}^T,   Γ_l = g_l g_l^T
F_l ≈ δ_l Φ_l ⊗ Ψ_l  (Eq. 4.3), with the "simplified formulas" of Eq. 4.9 (paper's own choice,
"in the rest of this paper we all use the simplified formulas as (4.9)"):

    δ_l = E[tr(Λ_{l-1})tr(Γ_l)],  Φ_l = E[tr(Γ_l)Λ_{l-1}]/δ_l,  Ψ_l = E[tr(Λ_{l-1})Γ_l]/δ_l
```

In this project's notation, `Λ_{l-1} = h̄h̄^T` (bias-augmented input, same `h̄` as `kfac`/`ekfac`)
and `Γ_l = δδ^T` (output gradient — unfortunate symbol clash with the paper's own damping `λ`
and this project's EMA `δ`; code below never uses the bare name `delta` for the gradient). Per
example `n`, `tr(Λ_n) = ‖h̄_n‖²` and `tr(Γ_n) = ‖s_n‖²` are scalars, so `δ_l`, `Φ_l`, `Ψ_l` are all
expectations over the **same** per-example pairing `(h̄_n, s_n)` — exactly the same kind of
per-example correlation `ekfac`'s `s*` already needed (`plan_lot2.md` §0.3), solved the same way:
cache `h̄` at the forward hook, consume it together with `s` at the backward hook (forward always
precedes backward for one `loss.backward()` call).

### 0.2 EMA design: accumulate the *unnormalized* numerators, divide only at read time

`Φ_l`, `Ψ_l` are **ratios** of two expectations, not expectations of a ratio. This project's shared
`update_running_avg` (`ema.py`) implements `current ← (1−γ₀)·current + γ₁·new`, whose coefficients
do **not** sum to 1 (`plan.md` §1.4 — a deliberate, authoritative property of the ported formula,
not a bug to fix). This has a consequence specific to TKFAC that lot 2 never had to face: if the
*already-normalized* instantaneous estimate `Φ_i := (E_batch[tr(Γ)Λ]) / (E_batch[tr(Λ)tr(Γ)])`
(individually satisfying `tr(Φ_i) = 1` for every batch) were EMA'd directly, the accumulated
`tr(Φ)` would drift to `(1−γ₀)+γ₁ ≠ 1`, silently breaking Theorem 4.1's own trace-preservation
property — a correctness bug that would only show up as a slightly-wrong number, not a crash.

**Resolved:** maintain three EMA'd state tensors per module — `_delta` (scalar), `_Phi_raw`
(`d_in_aug × d_in_aug`), `_Psi_raw` (`d_out × d_out`) — accumulating the **unnormalized**
per-batch numerators

```
delta_i  = mean_n[ tr(Λ_n)·tr(Γ_n) ] = mean_n[ ‖h̄_n‖²·‖s_n‖² ]
Phi_raw_i = mean_n[ tr(Γ_n)·Λ_n ]    = h̄^T diag(‖s_n‖²) h̄ / N
Psi_raw_i = mean_n[ tr(Λ_n)·Γ_n ]    = s^T diag(‖h̄_n‖²) s / N
```

with the **same** `update_running_avg(·, ·, self.gammas)` call, at the **same** step, for all three
— and only divide by `_delta` when `Φ_l`/`Ψ_l` are actually needed (`refresh`, `f_tilde`). Since
`tr(Phi_raw_i) = delta_i` and `tr(Psi_raw_i) = delta_i` **exactly**, for *every* batch (immediate
from the trace identity `tr(A⊗B)=tr(A)tr(B)` collapsed to the scalar case), and since `tr(·)` and
`update_running_avg` are both linear operations applied with *identical* coefficients to all three
tensors, this gives

```
tr(self._Phi_raw[module]) == self._delta[module]   (exactly, up to float-add associativity)
tr(self._Psi_raw[module]) == self._delta[module]
```

**at every step, for any `gammas`, unconditionally** — not just at the `gammas=(1,1)` special case
lot 2's dominance tests use. This is what makes `tr(Φ_l)=tr(Ψ_l)=1` (hence `tr(F̃_TKFAC)=tr(F))`
hold up to floating-point rounding only, never to EMA-scheme drift. `tests/test_tkfac_tekfac_precondition.py`
asserts this identity directly, under the project's real `gammas=(0.92, 0.008)`, as a regression
guard independent of the brute-force oracle test.

**Bootstrap, consistent with the invariant above.** `kfac`/`ekfac` bootstrap `A=B=eye(d)` at
`step==0` (so `F_bootstrap = kron(eye,eye) = I`, an inert preconditioner). The analogous TKFAC
bootstrap that (a) also gives `F̃_bootstrap = I` and (b) satisfies the trace invariant above from
the very first step is

```
_delta[module]   = d_in_aug · d_out
_Phi_raw[module] = d_out · eye(d_in_aug)
_Psi_raw[module] = d_in_aug · eye(d_out)
```

(`tr(Phi_raw_0) = d_out·d_in_aug = delta_0` ✓, `tr(Psi_raw_0) = d_in_aug·d_out = delta_0` ✓;
`Φ_0 = eye/d_in_aug`, `Ψ_0 = eye/d_out`, `δ_0·Φ_0⊗Ψ_0 = I` ✓). Unlike `kfac`/`ekfac`, where `A` and
`B` bootstrap independently in their own hook (each only needs its own dimension), TKFAC needs
**both** `d_in_aug` and `d_out` simultaneously, which are only jointly known at the backward hook
(after the forward-cached `h̄` is retrieved) — so, architecturally, **all** of TKFAC's state is
created in `update_output_factor`, not split across both hooks the way `kfac`'s `_A`/`_B` are.
`update_input_factor` does nothing but cache `h̄`.

Because this numerator/denominator split and its bootstrap are needed identically by `tekfac.py`
(§0.4), they are factored into a new private module `approximations/_tkfac_utils.py` —
the same "small shared module, ≥2 consumers" justification `_kron_utils.py` already established in
`plan_lot2.md` §0.2, applied one level up (shared *statistics*, not shared *shape plumbing*).

### 0.3 Damping (Eq. 5.14-5.15), re-derived to match the `(1/δ)·Ψ̃⁻¹MΦ̃⁻¹` form of `CLAUDE.md`'s cheat sheet

The paper's own normal-damping technique for fully-connected layers (`tkfac_2011.10741.pdf` p. 16,
Eq. 5.14-5.15) adds `λI` to the **undamped** `F_l = δ_lΦ_l⊗Ψ_l`, then reproduces Martens & Grosse's
factored-Tikhonov trick (K-FAC §6.3) directly on the *scaled* factors:

```
Φ̂_l = √δ_l·Φ_l + √λ·I,      Ψ̂_l = √δ_l·Ψ_l + √λ·I         (Eq. 5.15)
update: ω ← ω − α(Φ̂_l⁻¹ ⊗ Ψ̂_l⁻¹)∇_{ω_l}h                   (unnumbered display, p. 17, before Alg. 1)
```

`CLAUDE.md`'s cheat sheet instead states `precondition(M) = (1/δ)·Ψ̃⁻¹MΦ̃⁻¹` without specifying
what damping term `Φ̃`/`Ψ̃` add — this lot must pin that down, and the two forms are shown here to
be **exactly equal**, not just "in the same spirit":

```
Φ̂_l = √δ_l·Φ_l + √λ·I = √δ_l·(Φ_l + √(λ/δ_l)·I) = √δ_l · Φ̃_l,   Φ̃_l := Φ_l + √(λ/δ_l)·I
Ψ̂_l = √δ_l · Ψ̃_l,                                                Ψ̃_l := Ψ_l + √(λ/δ_l)·I
⇒ Φ̂_l⁻¹ = (1/√δ_l)·Φ̃_l⁻¹,  Ψ̂_l⁻¹ = (1/√δ_l)·Ψ̃_l⁻¹
⇒ Ψ̂_l⁻¹MΦ̂_l⁻¹ = (1/δ_l)·Ψ̃_l⁻¹MΦ̃_l⁻¹
```

which is `CLAUDE.md`'s form, now with the damping term made explicit: **both** `Φ̃` and `Ψ̃` get
`+√(λ/δ)·I` (a single scalar shared between the two factors — TKFAC has no K-FAC-style asymmetric
`π` split; the paper's own Eq. 5.15 splits the damping evenly by construction, via the shared `√λ`
term applied identically to both `Φ̂` and `Ψ̂`). This derivation is the lot-3 analogue of
`plan_lot2.md` §0.4's `rvec` derivation: re-derived from the paper's own equations rather than
assumed correct because the cheat sheet already wrote it that way.

`δ_l` itself is never floored/epsilon-guarded beyond `λ` — consistent with `CLAUDE.md`'s "damping
is `λ` only, as each paper prescribes" for the four new modes. `δ_l = E[‖h̄‖²‖g‖²] ≥ 0` and is
generically bounded away from 0 for any non-degenerate layer (would require an entire batch of
all-zero activations or all-zero gradients to vanish); the paper's own `ν`-floor (Eq. 5.16) is an
explicitly **convolutional-only** device (`plan.md` §4.4: "Convolutional case: ... Adaptive conv
damping: §5, Eq. (5.16)"), out of scope for this `Linear`-only lot exactly like lot 2's `kfac`/`ekfac`
left `Conv2d` to lot 4.

**A fourth cached quantity, not just two.** `kfac`'s `refresh()` freezes `_A_inv`/`_B_inv` every
`T_inv` steps; `precondition()` only ever reads those two frozen tensors. TKFAC's `precondition`
additionally needs `1/δ_l` — and it must be **the same vintage of `δ_l`** used to build the
currently-frozen `Φ̃⁻¹`/`Ψ̃⁻¹` (mixing a live, EMA-drifted `δ` with stale inverses would silently
apply an inconsistent operator). So `refresh()` also freezes `_delta_at_refresh[module]`, read only
by `precondition`; `f_tilde()` (a debug accessor, like every other mode's) instead reads the *live*
`_delta`/`_Phi_raw`/`_Psi_raw`, mirroring `kfac.f_tilde()`'s own "reflects EMA drift since the last
`refresh()`" contract (`plan_lot2.md` §0.5) — the two are documented to agree only right after a
fresh `refresh()` call, exactly as lot 2 already documents for `kfac`/`ekfac`.

### 0.4 TEKFAC reuses TKFAC's `(δ, Φ, Ψ)` machinery unchanged; only the correction is new

TEKFAC (`tekfac_2011.13609.pdf` §2.4/§3.1, Eq. 2.9-2.10 reproduced verbatim from TKFAC, confirmed
by direct comparison against `tkfac_2011.10741.pdf` Eq. 4.9 — identical formula, `σ` renamed `δ`
back in the TKFAC paper's own notation and kept as `δ` here) computes eigenbases directly from the
**undivided** `Φ_l`, `Ψ_l`:

```
F_l ≈ δ_lΦ_l⊗Ψ_l = δ_l(Q_Φ Λ_Φ Q_Φ^T)⊗(Q_Ψ Λ_Ψ Q_Ψ^T) = δ_l(Q_Φ⊗Q_Ψ)(Λ_Φ⊗Λ_Ψ)(Q_Φ⊗Q_Ψ)^T   (Eq. 3.1)
```

then, exactly EKFAC's own Lemma 1 argument applied to the orthogonal basis `Q_Φ⊗Q_Ψ` instead of
`Q_A⊗Q_B`, replaces the (still inexact) rescaling `δ_l(Λ_Φ⊗Λ_Ψ)` with the **directly estimated**
optimal diagonal

```
Θ_ii = E[((Q_Φ⊗Q_Ψ)^T ∇_ω h)_i²]     (Eq. 3.2)     F_l ≈ (Q_Φ⊗Q_Ψ)Θ_l(Q_Φ⊗Q_Ψ)^T   (Eq. 3.3)
```

**note the absence of a separate `δ` multiplying `Θ`** — `Θ` is estimated directly from the raw
per-example gradient in the `(Q_Φ, Q_Ψ)` eigenbasis, so it already has the correct scale built in
(this is the point of EKFAC-style correction: it replaces a *derived* rescaling by a *measured*
one). Theorem 3.1 (`‖F−F̃_TEKFAC‖_F ≤ ‖F−F̃_TKFAC‖_F`) is proved by the identical argument as
EKFAC's Theorem 2/3 (Lemma 1 applied to a fixed orthogonal basis), just with `σΛ_Φ⊗Λ_Ψ` playing the
role of K-FAC's suboptimal diagonal `S_A⊗S_B`.

**Consequence for this codebase:** `eigh` needs eigenvectors of `Φ_l`/`Ψ_l`, i.e. of
`self._Phi_raw[module]`/`self._Psi_raw[module]` **or** of those divided by `δ_l` — the two share
the same eigenvectors (dividing a symmetric matrix by a positive scalar rescales its eigenvalues,
never its eigenvectors), so `refresh()` can `eigh` the raw, undivided numerators directly, skipping
a division. `Θ`'s per-example estimator is then **structurally identical** to `ekfac`'s `s*`
estimator (`plan_lot2.md` §0.3's own "third choice": intra-batch, EMA-tracked with this project's
`update_running_avg`), with `(Q_A,Q_B)→(Q_Φ,Q_Ψ)`. The state TEKFAC needs is therefore exactly
TKFAC's three tensors (`_delta`, `_Phi_raw`, `_Psi_raw`, updated via the **same**
`_tkfac_utils.instantaneous_raw_factors`/`bootstrap_raw_factors` helpers §0.2 introduces) plus
`_Q_Phi`, `_Q_Psi`, `_Theta`, `_cached_h_bar` — a direct structural mirror of `ekfac.py`.

**Decoupled cadences (Algorithm 1, `tekfac_2011.13609.pdf` p. 8): `T_FIM`, `T_EIG`, `T_RE`.**
TKFAC's own Algorithm 1 (`tkfac_2011.10741.pdf` p. 17) *also* lists `T_FIM`/`T_INV` as two
independently-configurable cadences, distinct from what this codebase calls `TCov` (the optimizer's
own hook-firing gate, `plan.md` §1.1). Lot 2 already made a deliberate, precedent-setting choice
here for `kfac`/`ekfac`: **no separate `T_fim`-like parameter** — the raw-factor EMA update
(`_A`/`_B`) runs on *every* hook fire (i.e., already gated externally at `TCov` cadence by
`optimizer.py`), and only the amortised step (`T_inv`/`T_eig`) is a mode-owned cadence. This lot
keeps that choice for **both** `tkfac` (`T_inv` only) and `tekfac`'s `(δ, Φ, Ψ)` triple (no
`T_fim`), for consistency across all five modes and because introducing it now, unused by any
existing mode, would be exactly the kind of speculative parameter `CLAUDE.md`'s conventions ask to
avoid. **`T_RE` is kept**, however, because it is not redundant with anything already gated: it
decouples `Θ`'s own EMA-update cadence from both `TCov` (which already gates whether
`update_output_factor` runs at all) and `T_eig` (which gates the eigenbasis). Concretely,
`update_output_factor` only folds a new `Θ` estimate in when `step % T_re == 0` **and** an
eigenbasis already exists — the second condition being `ekfac`'s own existing bootstrap gate,
`plan_lot2.md` §0.3. `CLAUDE.md`'s own constructor illustration already shows `T_re=1` as a
first-class parameter (and does not show any `T_fim`), which is consistent with this decision.

**`beta_theta`/`beta_factors` (notation collision flagged in `CLAUDE.md`).** TEKFAC's own Eq. 3.6-3.8
use a proper convex-combination EMA (`Θ⁽ᵏ⁺¹⁾ ← β₁Θ_new + (1−β₁)Θ⁽ᵏ⁾`, coefficients summing to 1) —
a **different** convention from this project's `gammas=[1−γ₀,γ₁]` (deliberately **not** summing to
1, `plan.md` §1.4, and shared verbatim by every other mode so far). Per this project's own
established precedent (`plan_lot2.md` §0.3: "keeps every mode's factor-smoothing mechanism
identical" is itself the deliberate design goal, stated as more important than matching a paper's
exact smoothing formula verbatim), **`beta_theta` and `beta_factors` are each a `gammas`-shaped
2-tuple fed into this project's own `update_running_avg`** — not the paper's literal β/`1−β` form —
one pair for `Θ` (`T_re`-gated), one for `(δ,Φ,Ψ)` (ungated beyond `TCov`, like `kfac`/`ekfac`'s
`A`/`B`). Both default to the constructor's `gammas` if not overridden, so a caller who does not
care about decoupling gets the same single-`gammas` behaviour as every other mode.

### 0.5 A necessary correction to `CLAUDE.md`'s cheat-sheet formula for `tekfac`'s applied form

`CLAUDE.md`'s table row for `tekfac` reads `Q_Ψ[(Q_Ψ^T M Q_Φ)/(Θ+λ)]Q_Φ^T` — this is exactly right
and is what this lot implements (structurally identical to `ekfac.py`'s already-implemented,
already-tested `precondition`, §0.4). Flagged here only to confirm explicitly, after actually
re-reading Eq. 3.9 (`ω^{(k+1)} ← ω^{(k)} − η(Q_Φ⊗Q_Ψ)^{(k+1)}[(Θ_l+λI)^{(k+1)}]^{-1}(Q_Φ^T⊗Q_Ψ^T)^{(k+1)}∇_{ω_l}h^{(k+1)}`,
cvec convention), that translating it into this codebase's row-major (`rvec`) convention via the
same identity `plan_lot2.md` §0.4 derived for `ekfac` (`Q` input-side outer↔inner swaps between
`cvec`/`rvec`) reproduces the cheat sheet's row exactly — no correction needed here, unlike §0.3
above where the *damping* form needed an explicit derivation to confirm equivalence.

### 0.6 `f_tilde()`, dense debug accessors — same contract as lot 2

```python
# tkfac:   f_tilde(module) = delta * kron(Psi_tilde, Phi_tilde)         # live EMA state, §0.2/§0.3
# tekfac:  f_tilde(module) = kron(Q_Psi, Q_Phi) @ diag((Theta+Lambda).flatten()) @ kron(Q_Psi, Q_Phi).T
```

Both mirror `kfac.f_tilde`/`ekfac.f_tilde` exactly (`plan_lot2.md` §0.5): dense, `O(d²)` to build,
test-only, reflecting live EMA drift for `tkfac` and the live eigenbasis/`Θ` for `tekfac`.

---

## 1. File-by-file design

```
src/adafisher_modes/
└── approximations/
    ├── _tkfac_utils.py               # NEW — instantaneous_raw_factors / bootstrap_raw_factors (§0.2)
    ├── tkfac.py                      # NEW — TKFACApproximation
    ├── tekfac.py                     # NEW — TEKFACApproximation
    └── __init__.py                   # MODES gains "tkfac", "tekfac"
src/adafisher_modes/__init__.py       # exports TKFACApproximation, TEKFACApproximation
tests/
├── test_tkfac_tekfac_precondition.py # NEW — shapes, trace invariant, T_inv/T_eig/T_re cadence
├── test_frobenius_dominance.py       # EXTENDED — exit criteria 1, 2 (already anticipated in CLAUDE.md)
└── test_ekfac_tekfac_equiv.py        # NEW — exit criterion 3 (§6.2)
```

`optimizer.py`, `approximations/base.py`, `_kron_utils.py`, `factors.py`, `diag.py`, `kfac.py`,
`ekfac.py` are **unchanged**. `augment_linear_input`/`augment_direction`/`split_direction` are
reused as-is (§0.1, §0.6).

### 1.1 `approximations/_tkfac_utils.py`

```python
def instantaneous_raw_factors(h_bar: Tensor, s: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """(delta_i, Phi_raw_i, Psi_raw_i): the un-normalized numerators of TKFAC Eq. (4.9) /
    TEKFAC Eq. (2.10), from one batch's bias-augmented input h_bar and output gradient s, paired
    per-example. See plan_lot3.md SS0.2 for why these stay un-normalized until read.
    """
    norm_a = (h_bar ** 2).sum(dim=1)   # tr(Lambda_n) = ||h_bar_n||^2
    norm_g = (s ** 2).sum(dim=1)       # tr(Gamma_n)  = ||s_n||^2
    n = h_bar.size(0)
    delta_i = (norm_a * norm_g).mean()
    phi_raw_i = (h_bar * norm_g.unsqueeze(1)).t() @ h_bar / n
    psi_raw_i = (s * norm_a.unsqueeze(1)).t() @ s / n
    return delta_i, phi_raw_i, psi_raw_i


def bootstrap_raw_factors(d_in: int, d_out: int, dtype, device) -> Tuple[Tensor, Tensor, Tensor]:
    """delta=d_in*d_out, Phi_raw=d_out*I, Psi_raw=d_in*I -- the bootstrap satisfying the trace
    invariant of SS0.2 from step 0, and reducing to F~=I (an inert preconditioner), matching
    kfac/ekfac's own eye/ones bootstraps.
    """
    delta_0 = torch.tensor(float(d_in * d_out), dtype=dtype, device=device)
    phi_raw_0 = d_out * torch.eye(d_in, dtype=dtype, device=device)
    psi_raw_0 = d_in * torch.eye(d_out, dtype=dtype, device=device)
    return delta_0, phi_raw_0, psi_raw_0
```

### 1.2 `approximations/tkfac.py` — `TKFACApproximation`

```python
class TKFACApproximation(FisherApproximation):
    def __init__(self, Lambda=1e-3, gammas=(0.92, 0.008), T_inv=100):
        ...
        self._delta, self._Phi_raw, self._Psi_raw = {}, {}, {}
        self._Phi_inv, self._Psi_inv, self._delta_at_refresh = {}, {}, {}
        self._cached_h_bar = {}

    def update_input_factor(self, module, h, step):
        self._cached_h_bar[module] = augment_linear_input(h, module)

    def update_output_factor(self, module, s, step):
        h_bar = self._cached_h_bar.pop(module)
        s_flat = s.reshape(-1, s.shape[-1]) if s.ndim > 2 else s
        delta_i, phi_raw_i, psi_raw_i = instantaneous_raw_factors(h_bar, s_flat)
        if step == 0:
            self._delta[module], self._Phi_raw[module], self._Psi_raw[module] = \
                bootstrap_raw_factors(h_bar.size(1), s_flat.size(1), h_bar.dtype, h_bar.device)
        update_running_avg(delta_i, self._delta[module], self.gammas)
        update_running_avg(phi_raw_i, self._Phi_raw[module], self.gammas)
        update_running_avg(psi_raw_i, self._Psi_raw[module], self.gammas)

    def _damped_factors(self, module):
        delta = self._delta[module]
        Phi, Psi = self._Phi_raw[module] / delta, self._Psi_raw[module] / delta
        damp = (self.Lambda / delta).sqrt()                      # SS0.3
        Phi_tilde = Phi + damp * eye(Phi.size(0), ...)
        Psi_tilde = Psi + damp * eye(Psi.size(0), ...)
        return delta, Phi_tilde, Psi_tilde

    def refresh(self, module, step):
        if step % self.T_inv != 0:
            return
        delta, Phi_tilde, Psi_tilde = self._damped_factors(module)
        self._delta_at_refresh[module] = delta
        self._Phi_inv[module], self._Psi_inv[module] = Phi_tilde.inverse(), Psi_tilde.inverse()

    def f_tilde(self, module):
        delta, Phi_tilde, Psi_tilde = self._damped_factors(module)
        return delta * kron(Psi_tilde, Phi_tilde)                 # B outer, A inner (SS0.1/plan_lot2 SS0.4)

    def precondition(self, module, weight_direction, bias_direction):
        M = augment_direction(weight_direction, bias_direction)
        direction = (self._Psi_inv[module] @ M @ self._Phi_inv[module]) / self._delta_at_refresh[module]
        return split_direction(direction, weight_direction.shape,
                                None if bias_direction is None else bias_direction.shape)
```

### 1.3 `approximations/tekfac.py` — `TEKFACApproximation`

```python
class TEKFACApproximation(FisherApproximation):
    def __init__(self, Lambda=1e-3, gammas=(0.92, 0.008),
                 beta_factors=None, beta_theta=None, T_eig=100, T_re=1):
        self.beta_factors = tuple(beta_factors) if beta_factors is not None else tuple(gammas)
        self.beta_theta = tuple(beta_theta) if beta_theta is not None else tuple(gammas)
        ...
        self._delta, self._Phi_raw, self._Psi_raw = {}, {}, {}   # SS0.4 -- same triple as TKFAC
        self._Q_Phi, self._Q_Psi, self._Theta = {}, {}, {}
        self._cached_h_bar = {}

    def update_input_factor(self, module, h, step):
        self._cached_h_bar[module] = augment_linear_input(h, module)

    def update_output_factor(self, module, s, step):
        h_bar = self._cached_h_bar.pop(module)
        s_flat = s.reshape(-1, s.shape[-1]) if s.ndim > 2 else s
        delta_i, phi_raw_i, psi_raw_i = instantaneous_raw_factors(h_bar, s_flat)
        if step == 0:
            self._delta[module], self._Phi_raw[module], self._Psi_raw[module] = \
                bootstrap_raw_factors(h_bar.size(1), s_flat.size(1), h_bar.dtype, h_bar.device)
        update_running_avg(delta_i, self._delta[module], self.beta_factors)
        update_running_avg(phi_raw_i, self._Phi_raw[module], self.beta_factors)
        update_running_avg(psi_raw_i, self._Psi_raw[module], self.beta_factors)

        if module in self._Q_Phi and step % self.T_re == 0:
            h_kfe, s_kfe = h_bar @ self._Q_Phi[module], s_flat @ self._Q_Psi[module]
            theta_i = (s_kfe.t() ** 2) @ (h_kfe ** 2) / h_bar.size(0)     # Eq. 3.2, intra-batch
            update_running_avg(theta_i, self._Theta[module], self.beta_theta)

    def refresh(self, module, step):
        if step % self.T_eig != 0:
            return
        # Eigenvectors of Phi_raw/Psi_raw == eigenvectors of Phi_raw/delta, Psi_raw/delta (SS0.4):
        # dividing a symmetric matrix by a positive scalar does not move its eigenvectors.
        _, Q_Phi = linalg.eigh(self._Phi_raw[module])
        _, Q_Psi = linalg.eigh(self._Psi_raw[module])
        if module not in self._Q_Phi:
            self._Theta[module] = Q_Psi.new_ones(Q_Psi.size(0), Q_Phi.size(0))   # bootstrap, SS0.4
        self._Q_Phi[module], self._Q_Psi[module] = Q_Phi, Q_Psi

    def f_tilde(self, module):
        Q_Phi, Q_Psi = self._Q_Phi[module], self._Q_Psi[module]
        scale = (self._Theta[module] + self.Lambda).flatten()
        Q = kron(Q_Psi, Q_Phi)
        return Q @ diag(scale) @ Q.t()

    def precondition(self, module, weight_direction, bias_direction):
        M = augment_direction(weight_direction, bias_direction)
        Q_Phi, Q_Psi = self._Q_Phi[module], self._Q_Psi[module]
        M_kfe = (Q_Psi.t() @ M @ Q_Phi) / (self._Theta[module] + self.Lambda)
        direction = Q_Psi @ M_kfe @ Q_Phi.t()
        return split_direction(direction, weight_direction.shape,
                                None if bias_direction is None else bias_direction.shape)
```

### 1.4 `approximations/__init__.py`

```python
MODES: Dict[str, Callable[..., FisherApproximation]] = {
    "diag": DiagApproximation, "kfac": KFACApproximation, "ekfac": EKFACApproximation,
    "tkfac": TKFACApproximation, "tekfac": TEKFACApproximation,
}
```

No change to `optimizer.py`: `T_inv`/`T_eig`/`T_re`/`beta_factors`/`beta_theta` already flow through
unmodified via the existing `**mode_kwargs` pass-through.

---

## 2. Tests

### 2.1 `test_tkfac_tekfac_precondition.py` — unit-level, no oracle

Mirrors `test_kfac_ekfac_precondition.py`'s structure:

- **Shapes**, with and without bias, for both `tkfac` and `tekfac`.
- **Trace invariant (§0.2), the main new regression guard this lot adds:** drive `tkfac` for several
  steps with the project's real `gammas=(0.92, 0.008)` (not the test-only `(1,1)`) and random
  `(h, s)` batches; assert `torch.allclose(self._Phi_raw[layer].trace(), self._delta[layer])` and
  likewise for `_Psi_raw`, after **every** step — this is what would break silently if a future
  edit normalized `Φ`/`Ψ` before accumulating (§0.2's exact failure mode).
- **`T_inv`/`T_eig`/`T_re` cadence**, each following lot 2's established pattern: cache the
  amortised quantity at a refresh step, drive several more non-refresh steps, assert
  `torch.equal` (unchanged) across those, then `not torch.equal` at the next refresh step. `T_re`'s
  test additionally checks that `_Theta` does **not** move at all before an eigenbasis exists
  (mirrors `ekfac`'s own bootstrap-gating, already implicitly covered by `plan_lot2.md`'s tests for
  `ekfac`, checked explicitly here for `tekfac` too since `T_re` adds a second gate).
- **`f_tilde` / `precondition` internal consistency**, exactly `test_f_tilde_matches_precondition_for_kfac/ekfac`'s
  pattern, for both `tkfac` and `tekfac`.

### 2.2 `test_frobenius_dominance.py` — extended, exit criteria 1 and 2

Reuses the existing file's `D_IN, D_OUT, N` constants and its `_exact_fisher_block` helper
unchanged (same toy dimension and exact-`F` oracle already validated for `kfac`/`ekfac`). Adds:

```python
def _build_tkfac_and_tekfac(h, s, Lambda=1e-8):
    layer = nn.Linear(D_IN, D_OUT, bias=True)

    tkfac = TKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_inv=1)
    tkfac.update_input_factor(layer, h, step=0); tkfac.update_output_factor(layer, s, step=0)
    tkfac.refresh(layer, step=0)

    tekfac = TEKFACApproximation(Lambda=Lambda, gammas=(1.0, 1.0), T_eig=1, T_re=1)
    tekfac.update_input_factor(layer, h, step=0); tekfac.update_output_factor(layer, s, step=0)
    tekfac.refresh(layer, step=0)                                    # Q_Phi, Q_Psi exact; Theta bootstrapped
    tekfac.update_input_factor(layer, h, step=1); tekfac.update_output_factor(layer, s, step=1)  # Theta <- exact

    return layer, tkfac, tekfac
```

- `test_tkfac_trace_matches_exact_fisher`: `torch.allclose(tkfac.f_tilde(layer).trace(), F_exact.trace(), atol=1e-4)`
  — exit criterion 1 (TKFAC Thm 4.1 / Lemma 4.1).
- `test_tekfac_dominates_tkfac_in_frobenius_norm`: `‖F_exact − F̃_tekfac‖_F ≤ ‖F_exact − F̃_tkfac‖_F + 1e-4`
  — exit criterion 2 (TEKFAC Thm 3.1), same tolerance convention as lot 2's KFAC/EKFAC test.
- `test_f_tilde_matches_precondition_for_{tkfac,tekfac}`: same `torch.linalg.solve` internal-consistency
  check as lot 2.
- `test_measured_rescaled_error_report`: extended to also print `e(F̃)` for `tkfac`/`tekfac`
  (informational only, `plan.md` §6.1/§9 answer 2 — measured, never asserted).

### 2.3 `test_ekfac_tekfac_equiv.py` — exit criterion 3, `plan.md` §6.2

Forces the degenerate substitution `Φ:=A, Ψ:=B, δ:=1` directly on TEKFAC's internal state (bypassing
its normal `(δ,Φ,Ψ)` estimation), so that `Q_Φ=Q_A`, `Q_Ψ=Q_B` and `Θ`'s estimator becomes
*identical in form* to `ekfac`'s `s*` estimator (§0.4):

```python
def test_ekfac_tekfac_agree_in_degenerate_case():
    seed_all(0)
    h, s = torch.randn(N, D_IN), torch.randn(N, D_OUT)
    layer = nn.Linear(D_IN, D_OUT, bias=True)

    ekfac = EKFACApproximation(Lambda=1e-3, gammas=(1.0, 1.0), T_eig=1)
    ekfac.update_input_factor(layer, h, 0); ekfac.update_output_factor(layer, s, 0)
    ekfac.refresh(layer, 0)
    ekfac.update_input_factor(layer, h, 1); ekfac.update_output_factor(layer, s, 1)

    tekfac = TEKFACApproximation(Lambda=1e-3, gammas=(1.0, 1.0), T_eig=1, T_re=1)
    # Bypass the (delta, Phi, Psi) estimation entirely: Phi := A, Psi := B (delta implicitly 1,
    # since precondition/f_tilde never multiply Theta by delta -- SS0.4).
    tekfac._Phi_raw[layer], tekfac._Psi_raw[layer] = ekfac._A[layer].clone(), ekfac._B[layer].clone()
    tekfac.refresh(layer, 0)     # Q_Phi = Q_A, Q_Psi = Q_B (same matrices, up to eigh's own sign/order)
    tekfac._cached_h_bar[layer] = ekfac._cached_h_bar_replay  # same (h_bar) used by ekfac at step 1
    tekfac.update_output_factor(layer, s, 1)   # Theta <- same intra-batch estimate as ekfac's s*

    weight_direction, bias_direction = torch.randn(D_OUT, D_IN), torch.randn(D_OUT)
    w_e, b_e = ekfac.precondition(layer, weight_direction, bias_direction)
    w_t, b_t = tekfac.precondition(layer, weight_direction, bias_direction)
    assert torch.allclose(w_e, w_t, rtol=1e-5) and torch.allclose(b_e, b_t, rtol=1e-5)
```

(The sketch above re-derives `h_bar` locally via `augment_linear_input(h, layer)` rather than
reaching into `ekfac`'s private cache, which is already popped by the time `update_output_factor`
returns — the actual test file computes it directly instead of the illustrative
`_cached_h_bar_replay` placeholder above.) This is the most discriminating test of the lot: it
simultaneously validates TEKFAC's eigenbasis construction, its `Θ` estimator, and its `rvec`
application, against the already-validated `ekfac.py` as oracle — precisely because, in this
degenerate substitution, the two implementations are mathematically required to coincide exactly
(`plan.md` §6.2), while normally being built from entirely different per-layer statistics.
`torch.linalg.eigh`'s sign/ordering ambiguity (`plan.md`'s own listed testing pitfall) is a
non-issue here because both `ekfac` and `tekfac` `eigh` the exact same matrices (`A`/`Phi_raw` and
`B`/`Psi_raw` are literally the same tensor objects), so any sign/order choice `eigh` makes is
identical between the two calls — the comparison is on **applied preconditioners**, per that same
listed pitfall, not on the bases themselves.

---

## 3. Explicitly out of scope for lot 3

- `Conv2d` for `tkfac`/`tekfac` (Theorem 4.3, Assumption 4.1, the `ν`-floor/`β`-rescaling damping of
  Eq. 5.16) — lot 4, same boundary lot 2 drew for `kfac`/`ekfac`.
- Normalisation layers for any of the four non-diagonal modes — lot 5.
- Any change to `optimizer.py`, `approximations/base.py`, `diag.py`, `kfac.py`, `ekfac.py`,
  `_kron_utils.py`, `factors.py`, or lots 1-2's tests.
- Benchmark integration (`benchmarks/mnist_autoencoder.py` comparing `tkfac`/`tekfac`) — lot 7's
  equal-wall-clock comparison.
- A separate `T_fim`-style cadence for `tkfac`'s or `tekfac`'s raw-factor EMA (§0.4) — a deliberate,
  documented choice to keep all five modes' raw-factor cadence identical (≡ `TCov`), not an
  oversight.

---

## 4. Points worth flagging explicitly (not blocking, per this session's operating mode)

1. §0.2 — accumulating **unnormalized** numerators (`_Phi_raw`, `_Psi_raw`, `_delta`) rather than
   the paper's own `Φ_l`/`Ψ_l` directly is a deliberate implementation choice, not a transcription
   of Eq. 4.9 as written. It is mathematically required (not just convenient) for `tr(F̃_TKFAC)=tr(F)`
   to hold exactly under this project's own non-summing-to-1 `gammas` EMA — normalizing before
   accumulating would silently break Theorem 4.1's headline property.
2. §0.3 — the `(1/δ)Ψ̃⁻¹MΦ̃⁻¹` form is derived, not assumed, from the paper's own Eq. 5.15; the
   derivation pins down the previously-unspecified damping term as `+√(λ/δ)·I` on both factors.
3. §0.4 — `beta_theta`/`beta_factors` reuse this project's `gammas`-shaped EMA rather than the
   paper's literal convex-combination `β`/`1−β` form, for the same reason `plan_lot2.md` §0.3 gave
   for `ekfac`'s `s*`: one consistent smoothing mechanism across all five modes.
4. `T_fim` is deliberately not introduced for either mode (§0.4), extending lot 2's own precedent
   for `kfac`/`ekfac` rather than following TKFAC/TEKFAC's own Algorithm 1 pseudocode literally.
