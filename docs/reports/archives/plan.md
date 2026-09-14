# Implementation Plan — Interchangeable Fisher Approximation Modes for AdaFisher

**Status:** design decisions resolved (2026-09-07). Awaiting go-ahead before writing implementation code.
**Scope:** add K-FAC, EKFAC, TKFAC and TEKFAC as alternative, interchangeable ways of building
AdaFisher's second moment `v^(t)`, behind a single configuration flag.

---

## 0. Primary sources

### `papers/` — 8 PDFs, each verified by extracting the title from page 1

| File | arXiv | Verified title |
|---|---|---|
| `adafisher_2405.16397.pdf` | 2405.16397v3 | *AdaFisher: Adaptive Second Order Optimization via Fisher Information*, ICLR 2025 |
| `kfac_1503.05671.pdf` | 1503.05671v7 | Martens & Grosse, *Optimizing Neural Networks with Kronecker-factored Approximate Curvature* |
| `kfac_conv_1602.01407.pdf` | 1602.01407v2 | Grosse & Martens, *A Kronecker-factored approximate Fisher matrix for convolution layers* (KFC) |
| `kfac_from_scratch_2507.05127.pdf` | 2507.05127v1 | Dangel, Mucsányi, Weber, Eschenhagen, *KFAC From Scratch* |
| `ekfac_1806.03884.pdf` | 1806.03884v2 | George et al., *Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis* |
| `tkfac_2011.10741.pdf` | 2011.10741v1 | Gao et al., *A Trace-restricted Kronecker-Factored Approximation to Natural Gradient* |
| `tekfac_2011.13609.pdf` | 2011.13609v1 | Gao et al., *Eigenvalue-corrected Natural Gradient Based on a New Approximation* (TEKFAC) |
| `empirical_fisher_limits_1905.12558.pdf` | **1905.12558**v3 | Kunstner, Balles, Hennig, *Limitations of the Empirical Fisher Approximation…* |

The Kunstner arXiv ID was not supplied; it was **verified** against `arxiv.org/abs/1905.12558`
(exact title returned), not guessed. All seven supplied IDs were correct.

### `reference_repos/` — read-only

| Directory | Role |
|---|---|
| `FisherAdapTune/` | **authoritative AdaFisher implementation** (`scripts/adafisher.py`). Do not modify. |
| `AdaFisher/` | official `AtlasAnalyticsLab/AdaFisher` repository. Contains an EMA bug (§1.4) — comparison only, never a source of truth. |
| `EKFAC-pytorch/` | `Thrandis/EKFAC-pytorch` — `kfac.py`, `ekfac.py`. Tested implementation reference. |

---

## 1. Graft point in the existing code

Everything below refers to
[reference_repos/FisherAdapTune/scripts/adafisher.py](reference_repos/FisherAdapTune/scripts/adafisher.py).

### 1.1 Where `H` and `S` are computed and EMA-updated (AdaFisher Eq. 3)

| Element | Location |
|---|---|
| Hook registration | `AdaFisherBackbone._prepare_model`, [lines 202-207](reference_repos/FisherAdapTune/scripts/adafisher.py#L202-L207) — `register_forward_hook` + `register_full_backward_hook` on `Linear`, `Conv2d`, `BatchNorm2d`, `LayerNorm` (`SUPPORTED_MODULES`, line 158). |
| Activation factor `H̄` | `_ComputeHBarD`, [lines 52-111](reference_repos/FisherAdapTune/scripts/adafisher.py#L52-L111); called from `_save_input`, [line 190](reference_repos/FisherAdapTune/scripts/adafisher.py#L188-L193). |
| Gradient factor `S` | `_ComputeSD`, [lines 113-155](reference_repos/FisherAdapTune/scripts/adafisher.py#L113-L155); called from `_save_grad_output`, [line 197](reference_repos/FisherAdapTune/scripts/adafisher.py#L195-L200). |
| EMA (Eq. 3) | `_update_running_avg`, [lines 28-30](reference_repos/FisherAdapTune/scripts/adafisher.py#L28-L30): `current *= 1-γ₀; current += γ₁·new`, with `gammas = [0.92, 0.008]` (line 172). |
| Cadence | gated by `self.steps % self.TCov == 0` (lines 189 and 196), `TCov=100` by default. |

**Critical fact 1 — the full factors are never formed.** `_ComputeHBarD` and `_ComputeSD` compute
`einsum("ij,ij->j", X, X)` directly, i.e. **only the diagonal** of `E[h̄h̄ᵀ]` and `E[δδᵀ]`.
`self.H_bar_D[module]` and `self.S_D[module]` are **vectors**, not matrices. Consequently the four new
modes cannot simply hook in downstream of the existing EMA: the full factors
`A = E[h̄h̄ᵀ] ∈ R^{d_in×d_in}` and `B = E[δδᵀ] ∈ R^{d_out×d_out}` must be reintroduced. This sits
*upstream* of the assumed graft point and is the main structural cost of the project.

**Critical fact 2 — min-max normalisation is absent from this implementation.** Eq. (4) of the paper
defines `F̃_D ≜ H'_D ⊗ S'_D + λI` where `H'_D`, `S'_D` are the min-max normalisations of `H_D`, `S_D`.
The official repository applies `MinMaxNormalization` to the *instantaneous* factor **before** the EMA
(`AdaFisher/optimizers/AdaFisher.py:412` and `:431`); `FisherAdapTune/scripts/adafisher.py` does not —
`_smart_detect_inf` (line 21) is defined but never called, and there is no `MinMaxNormalization`
equivalent. **Resolved in §5.1:** the `diag` mode is paper-faithful (min-max on) by default, with an
opt-out that reproduces the FisherAdapTune code bit-exactly.

### 1.2 Where the diagonals are combined and `λI` added (Eq. 4) — the block to replace

`AdaFisherBackbone._get_F_tilde`, [lines 209-223](reference_repos/FisherAdapTune/scripts/adafisher.py#L209-L223):

```python
F_tilde = kron(self.H_bar_D[module].unsqueeze(1),
               self.S_D[module].unsqueeze(0)).t() + self.Lambda   # lines 215-217
if module.bias is not None:
    F_tilde = [F_tilde[:, :-1], F_tilde[:, -1:]]                  # weight / bias split
    F_tilde[0] = F_tilde[0].view(*module.weight.grad.data.size())
    F_tilde[1] = F_tilde[1].view(*module.bias.grad.data.size())
    return F_tilde
return F_tilde.reshape(module.weight.grad.data.size())
```

This is indeed the single graft point: `kron(...) + Lambda` (lines 215-217) is the only place where the
Fisher approximation is assembled. The weight/bias split (lines 218-222) and the `reshape` (line 223)
are shape plumbing and must be preserved verbatim.

### 1.3 Where the result enters the Adam-style update (`v^(t)` of Table 1)

`AdaFisher._step`, [lines 258-273](reference_repos/FisherAdapTune/scripts/adafisher.py#L258-L273), and
`AdaFisher.step`, [lines 275-307](reference_repos/FisherAdapTune/scripts/adafisher.py#L275-L307):

```python
exp_avg.mul_(beta).add_(grad, alpha=1 - beta)          # m^(t), line 271
step_size = hparams["lr"] / bias_correction            # 1/(1-β₁^t), line 272
param.addcdiv_(exp_avg, F_tilde, value=-step_size)     # θ ← θ − α m̂ / F̃, line 273
```

This matches Table 1 of the paper (`m^(t) = (1−β₁)Σβ₁^{t−i}g_i / (1−β₁^t)`, `v^(t) = F̃_D`, update
`θ⁺ = θ − α m/√v` **without the square root** for AdaFisher — §3.3: "AdaFisher omits the square root
and the traditional EMA applied over the second moment since the FIM naturally incorporates an EMA of
its KFs").

**Key design consequence.** `addcdiv_` is an **element-wise** division: it assumes the preconditioner
is diagonal *in the parameter basis*. True for `diag`, false by construction for K-FAC, EKFAC, TKFAC
and TEKFAC, whose entire point is to be non-diagonal in that basis. `addcdiv_` therefore cannot be
kept as is — see §2.2.

### 1.4 Divergence between the official repository and FisherAdapTune

For the record, since `reference_repos/AdaFisher/` remains readable: its signature is
`gamma: float = 0.8` (`AdaFisher.py:362`), validated by `if not 0.0 <= gamma < 1.0`, while its own
docstring example advertises `gamma=[0.92, 0.008]` (`AdaFisher.py:567`). A scalar passed to
`update_running_avg` yields `current *= 0.08; current += 0.008·new` — coefficients that do not sum to
1, so the running average decays towards 0. **The FisherAdapTune version is authoritative** (user
instruction). It is numerically identical on that point (`1−0.92 = 0.08`, `γ₁ = 0.008`) but drops the
min-max normalisation.

### 1.5 Current treatment of `Conv2d` and normalisation layers (Prop. 3.1)

**`Conv2d`** ([lines 66-76](reference_repos/FisherAdapTune/scripts/adafisher.py#L66-L76) and
[127-131](reference_repos/FisherAdapTune/scripts/adafisher.py#L127-L131)): `_extract_patches` unfolds
the input into patches (line 71), flattens `(batch × spatial locations)` onto axis 0, concatenates a
column of ones for the bias, then averages over `batch × spatial_size`. This is exactly the KFC
construction of Grosse & Martens 2016. The corresponding full factors would be
`A ∈ R^{(C_in·k_h·k_w + 1)²}` and `B ∈ R^{C_out²}`.

**Normalisation layers** (`_batchnorm2d` [lines 91-98](reference_repos/FisherAdapTune/scripts/adafisher.py#L91-L98),
`_layernorm` [lines 101-111](reference_repos/FisherAdapTune/scripts/adafisher.py#L101-L111)) — reading
the code precisely:

```python
sum_h = tsum(h, dim=(0,2,3)).unsqueeze(1) / spatial_size**2   # (C,1)
h_bar = cat([sum_h, ones(C,1)], 1)                            # (C,2)
return einsum("ij,ij->j", h_bar, h_bar) / batch_size**2       # (2,)
```

`einsum("ij,ij->j")` sums over `i` = the channels: the result is a **2-element vector**,
`[Σ_c (Σ_T h_c)², C]`. Combined with `S_D ∈ R^C`, `kron` yields a `(2, C)` matrix whose column 0 feeds
`ν` and column 1 feeds `β` (split at line 219). This is the operational reading of Prop. 3.1: `H|_ν`
collapses to a **scalar**, `H|_β = 11ᵀ` collapses to the scalar `C`, and all matrix structure is
carried by `S ∈ R^{C×C}`.

That reading is what makes §5.2 possible: for a normalisation layer the Fisher block has the form
`F = a · S` with `a` scalar and `S` of size `C×C`, so the four new modes are **dimensionally
applicable** with no invention — `A` is simply a `1×1` factor.

---

## 2. Design

### 2.1 Where the new code lives (**resolved: `src/adafisher_modes/`**)

`reference_repos/FisherAdapTune/` is read-only, so the new code is a self-contained package at the
workspace root:

```
adafisher /
├── papers/                     # primary sources (§0)
├── reference_repos/            # read-only
├── src/adafisher_modes/
│   ├── __init__.py
│   ├── optimizer.py            # AdaFisherMulti(Optimizer): hooks + Adam wrapper
│   ├── factors.py              # per-layer-type extraction (h̄, δ) → FULL factors (A, B)
│   ├── approximations/
│   │   ├── base.py             # FisherApproximation ABC
│   │   ├── diag.py             # mode "diag"  — AdaFisher Eq. (4), reference
│   │   ├── kfac.py             # mode "kfac"
│   │   ├── ekfac.py            # mode "ekfac"
│   │   ├── tkfac.py            # mode "tkfac"
│   │   ├── tekfac.py           # mode "tekfac"
│   │   └── __init__.py         # MODES registry {name → class}
│   └── config.py               # dataclass + YAML loading
├── tests/
├── benchmarks/
├── docs/reports/plan.md
└── CLAUDE.md
```

`optimizer.py` derives from `AdaFisherBackbone`/`AdaFisher`: hook structure, `step` loop, and the
module↔parameter pairing of `_check_dim` (lines 225-230) and of the loop (lines 280-306), which is
brittle and will not be rewritten.

### 2.2 Common interface: `FisherApproximation`

A single abstract base class rather than four separate optimizers, for three reasons:

1. **The varying surface is tiny.** Hooks, EMA (Eq. 3), per-layer-type extraction (Prop. 3.1), Adam
   wrapper (`β₁`, bias correction, `weight_decay`), module↔parameter pairing — all strictly shared
   across the five modes. Only the construction of `v^(t)` changes. Duplicating ~250 lines of plumbing
   five times to vary ~40 lines of formulas guarantees silent drift between modes and makes the §6.3
   comparisons uninterpretable.
2. **Equal-budget comparison requires an identical code path.** Criterion (iii) compares convergence
   speed. If two modes differ by anything other than the preconditioning formula, the measurement is
   meaningless.
3. **The dominance tests (§6.1) are interface tests.** `‖F − F̃_EKFAC‖_F ≤ ‖F − F̃_KFAC‖_F` is checked
   by instantiating two approximations over the *same* `(A, B)`, which is only possible if `(A, B)`
   come from shared code.

```python
class FisherApproximation(ABC):
    """Builds AdaFisher's second moment v^(t) for a given module."""

    @abstractmethod
    def update_factors(self, module, A, B, step: int) -> None:
        """EMA of the raw factors (AdaFisher Eq. 3 / TEKFAC Eq. 3.7-3.8)."""

    @abstractmethod
    def refresh(self, module, step: int) -> None:
        """Amortised re-estimation: inverses (K-FAC/TKFAC), eigenbases (EKFAC/TEKFAC)."""

    @abstractmethod
    def precondition(self, module, direction: Tensor) -> Tensor:
        """Apply F̃⁻¹ to `direction`, shaped like the gradient (d_out, d_in+bias)."""
```

**Deliberate deviation from the suggested `precondition(grad) -> grad` signature.** The argument is
not the raw gradient but the **bias-corrected first moment** `m̂^(t) = m^(t)/(1−β₁^t)`. Rationale: the
current AdaFisher update is `θ ← θ − α · m̂ / F̃_D` (line 273), i.e. `θ ← θ − α · F̃_D⁻¹ m̂` since
`F̃_D` is diagonal. Generalising correctly to a non-diagonal `F̃` therefore requires applying the
operator to `m̂`, not to `g`. This gives the property we want:

> for the `diag` mode, `precondition(m̂) = m̂ / (H_D ⊗ S_D + λ)` reproduces `addcdiv_` (line 273) exactly.

Preconditioning `g` and then averaging (the opposite order) would be a different algorithm — classic
K-FAC with momentum — and would break the `m^(t)` column of Table 1. On the optimizer side,
`param.addcdiv_(exp_avg, F_tilde, value=-step_size)` becomes:

```python
param.add_(approx.precondition(module, exp_avg / bias_correction), alpha=-lr)
```

a one-line rewrite, valid for all five modes.

### 2.3 Mode selection

A configuration field, **not a new optimizer**:

```python
AdaFisherMulti(model, lr=1e-3, beta=0.9, Lambda=1e-3, gammas=[0.92, 0.008], TCov=100,
               fisher_mode="diag",       # diag | kfac | ekfac | tkfac | tekfac
               minmax_normalization=True,   # diag mode only; see §5.1
               T_inv=100, T_eig=100, T_re=1)
```

YAML, in the style of `reference_repos/FisherAdapTune/crack_segmentation/config_segformer.yaml`:

```yaml
fisher_mode: ekfac
adafisher_tcov: 100
adafisher_gamma: [0.92, 0.008]
fisher_lambda: 1.0e-3
fisher_minmax: true     # diag mode only — true = paper-faithful (Eq. 4)
fisher_t_eig: 100       # ekfac / tekfac
fisher_t_inv: 100       # kfac / tkfac
fisher_t_re: 1          # tekfac, T_RE of Alg. 1
```

`fisher_mode` maps into a `MODES: dict[str, type[FisherApproximation]]` registry in
`approximations/__init__.py`. No other point of the training code knows about the mode.

---

## 3. Unchanged vs. experimental

### 3.1 Unchanged across all five modes

| Element | Source | Reference location |
|---|---|---|
| Forward/backward hooks, `SUPPORTED_MODULES` | — | `adafisher.py:202-207` |
| Per-layer-type `(h̄, δ)` extraction | AdaFisher §A.3, Prop. 3.1 | `adafisher.py:52-155` |
| Factor EMA, `TCov` cadence | AdaFisher Eq. (3) | `adafisher.py:28-30`, `:189`, `:196` |
| First moment `m^(t)`, `β₁`, bias correction | AdaFisher Table 1 | `adafisher.py:269-272` |
| No square root on `v^(t)` | AdaFisher §3.3 | `adafisher.py:273` |
| Weight/bias split and `reshape` | — | `adafisher.py:218-223` |
| `weight_decay`, `AdaFisherW` | AdamW | `adafisher.py:267-268` |

### 3.2 Experimental: only the construction of `v^(t)`

| Mode | Per-layer state | Re-estimation | Cost per step |
|---|---|---|---|
| `diag` (ref.) | `H_D ∈ R^{d_in}`, `S_D ∈ R^{d_out}` | — | `O(d_in·d_out)` |
| `kfac` | `A ∈ R^{d_in²}`, `B ∈ R^{d_out²}`, `A⁻¹`, `B⁻¹` | `T_inv`: `O(d³)` | `O(d_in d_out (d_in+d_out))` |
| `ekfac` | `Q_A`, `Q_B` (orthogonal), `s* ∈ R^{d_out×d_in}` | `T_eig`: `O(d³)` | same + `O(d_in d_out)` |
| `tkfac` | `δ_l ∈ R`, `Φ_l ∈ R^{d_in²}`, `Ψ_l ∈ R^{d_out²}`, inverses | `T_inv`: `O(d³)` | same as `kfac` |
| `tekfac` | `δ_l`, `Φ_l`, `Ψ_l`, `Q_Φ`, `Q_Ψ`, `Θ_l ∈ R^{d_out×d_in}` | decoupled `T_FIM`, `T_EIG`, `T_RE` (Alg. 1, `tekfac_2011.13609.pdf` p. 8) | same as `ekfac` |

---

## 4. The five modes: sourced formulas

Shared notation, fixed by `kfac_from_scratch_2507.05127.pdf` (Def. 22-23): `A = R·Σ_n x_n x_nᵀ` (input
factor, `x` = `h̄` augmented with the bias), `B = (1/N)·Σ_n Σ_c g_{n,c} g_{n,c}ᵀ` (grad-output factor),
and `KFAC(C(vec W)) = A ⊗ B` in **`cvec`** convention, `B ⊗ A` in **`rvec`** convention. **PyTorch code
is `rvec`** — the central pitfall flagged by that tutorial (Def. 1/2, §3), and the reason
`EKFAC-pytorch/kfac.py::_precond` writes `B⁻¹ G A⁻¹` with `G` of shape `(d_out, d_in)`.

### 4.1 `diag` — AdaFisher, reference

`F̃_D = H'_D ⊗ S'_D + λI` (Prop. 3.2, Eq. 4). `v^(t) = F̃_D`, `θ⁺ = θ − α m̂/F̃_D` (Table 1).
`H'_D`, `S'_D` are the min-max normalisations of `H_D = diag(A)` and `S_D = diag(B)`, applied to the
**instantaneous** factor before the EMA (§5.1).

### 4.2 `kfac`

Exact per-layer Fisher block `F_l = E[h̄_{l−1}h̄_{l−1}ᵀ ⊗ δ_lδ_lᵀ]`. K-FAC applies the **independence
assumption** (`kfac_1503.05671.pdf` §3.1, Eq. 2):

> `E[ā⁽¹⁾ā⁽²⁾g⁽¹⁾g⁽²⁾] ≈ E[ā⁽¹⁾ā⁽²⁾]·E[g⁽¹⁾g⁽²⁾]`, i.e. "statistical independence between products
> `ā⁽¹⁾ā⁽²⁾` of unit activities and products `g⁽¹⁾g⁽²⁾` of unit input derivatives".

Hence `F̆ = diag(Ā_{0,0} ⊗ G_{1,1}, …)` (§4.2), the inversion identity `(A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹` (§4.2),
and `(A ⊗ B)vec(X) = vec(BXAᵀ)` (§4.2). Factored Tikhonov damping `π`: §6.3/6.6 of the same paper,
implemented in `EKFAC-pytorch/kfac.py::_inv_covs` (option `pi=True`).

`precondition(M) = B̃⁻¹ M Ã⁻¹` with `Ã = A + √(λπ)I`, `B̃ = B + √(λ/π)I`.

Convolutional case: KFC, `kfac_conv_1602.01407.pdf` — spatial independence assumptions, `A` built on
unfolded patches (identical to `_extract_patches`, `adafisher.py:33-50`), `B` averaged over locations
(`num_locations` in `EKFAC-pytorch/kfac.py::_compute_covs`).

### 4.3 `ekfac`

`G_EKFAC = (U_A ⊗ U_B) S* (U_A ⊗ U_B)ᵀ` with `S*_ii = E[((U_A ⊗ U_B)ᵀ ∇_θ)²_i]`
(`ekfac_1806.03884.pdf` §3.2).

- **Lemma 1** (Appendix A.1): for a given orthogonal `Q`, `D_ii = E[(Qᵀ∇_θ)²_i]` minimises
  `‖G − QDQᵀ‖_F` among diagonal matrices. Proof: invariance of the Frobenius norm under orthogonal
  multiplication, then cancellation of the diagonal terms.
- **Theorem 2**: the case `Q = U_A ⊗ U_B`, legitimate because the Kronecker product of two orthogonal
  matrices is orthogonal.
- **Theorem 3**: `‖G − G_EKFAC‖_F ≤ ‖G − G_KFAC‖_F`, an immediate corollary, since K-FAC uses the
  constrained `D = S_A ⊗ S_B`.

**Algorithm 1** and its two variants (§4): `EKFAC` re-estimates `s*` as the second moment of
**intra-batch** projected gradients; `EKFAC-ra` maintains it as a running average of projected
mini-batch gradients. Code correspondence: `EKFAC-pytorch/ekfac.py::_precond_intra` vs `::_precond_ra`.

`precondition(M)`: `M̃ = Q_Bᵀ M Q_A`; `M̃ ← M̃/(s* + ε)`; return `Q_B M̃ Q_Aᵀ`.

### 4.4 `tkfac`

`F_l = E[Λ_{l−1} ⊗ Γ_l]` with `Λ = aaᵀ`, `Γ = ggᵀ` (`tkfac_2011.10741.pdf` §4.1, Eq. 4.2). Impose
`F_l = δ_l Φ_l ⊗ Ψ_l` (Eq. 4.3). **Theorem 4.1** (Eq. 4.4-4.6):

```
δ_l = E[tr(Λ)tr(Γ)] / (tr(Φ)tr(Ψ))
Φ_l = tr(Φ)·E[tr(Γ)Λ] / E[tr(Λ)tr(Γ)]
Ψ_l = tr(Ψ)·E[tr(Λ)Γ] / E[tr(Λ)tr(Γ)]
```

with the `tr(Φ)=tr(Ψ)=1` convention adopted by the authors (Eq. 4.9, "In the rest of this paper, we
all use the simplified formulas as (4.9)"):

```
δ_l = E[tr(Λ)tr(Γ)],   Φ_l = E[tr(Γ)Λ]/δ_l,   Ψ_l = E[tr(Λ)Γ]/δ_l
```

The proof rests on the **partial trace operator** `(PTr(A))_ij = tr(A_ij)` (§3, after Filipiak et al.
2018) and on **Lemma 4.1**: `PTr(E[Λ ⊗ Γ]) = E[tr(Γ)Λ]`. Trace preservation is the point:
`tr(F_TKFAC) = tr(F)` by construction, which K-FAC does not guarantee.

Convolutional case: **Theorem 4.3** (Eq. 4.12-4.13) under **Assumption 4.1** (products of activations
and pre-activation derivatives are uncorrelated at any two distinct spatial locations). Adaptive conv
damping: §5, Eq. (5.16), Alg. 1.

`precondition(M) = (1/δ_l)·Ψ̃⁻¹ M Φ̃⁻¹`.

### 4.5 `tekfac`

TKFAC eigenbasis: `F_l ≈ σ_l(Q_Φ ⊗ Q_Ψ)(Λ_Φ ⊗ Λ_Ψ)(Q_Φ ⊗ Q_Ψ)ᵀ` (`tekfac_2011.13609.pdf` §3.1,
Eq. 3.1), then rescaling correction — **exactly EKFAC's lemma applied to the TKFAC basis**:

```
(Θ_l)_ii = E[((Q_Φ ⊗ Q_Ψ)ᵀ ∇_ω h)²_i]        (Eq. 3.2)
F_l ≈ (Q_Φ ⊗ Q_Ψ) Θ_l (Q_Φ ⊗ Q_Ψ)ᵀ           (Eq. 3.3)
```

**Theorem 3.1**: `‖F − F_TEKFAC‖_F ≤ ‖F − F_TKFAC‖_F`, same optimality argument as EKFAC's Theorem 2,
applied to `Q_Φ ⊗ Q_Ψ`.

Damping (Eq. 3.4-3.5):

```
F_l ≈ (Q_Φ ⊗ Q_Ψ)(Θ_l + λI)(Q_Φ ⊗ Q_Ψ)ᵀ                        (3.4)
λ = max{tr(Θ_l), ϑ} / dim(Θ_l)      for convolutional layers    (3.5)
```

`λ` is a fixed scalar for dense layers ("same damping technique as EKFAC for FNNs") and
trace-adaptive for conv layers, with a factor `β = max_{l conv} max{tr(Θ_l), ϑ}/dim(Θ_l)` applied to
the dense layers of a CNN, "to keep pace with convolution layers".

Decoupled EMAs (Eq. 3.6-3.8): `Θ` with `β₁`, `Φ`/`Ψ` with `β₂`. Decoupled frequencies `T_FIM`,
`T_EIG`, `T_RE` (Alg. 1). Final update Eq. 3.9, using `(A ⊗ U)vec(X) = vec(UᵀXA)`.

**Notation collision.** The TEKFAC paper calls `β₁`, `β₂` the EMA decays of `Θ` and of `(Φ,Ψ)` —
unrelated to Adam/AdaFisher's `β₁` (the momentum of `m^(t)`). In code: `beta_theta` and
`beta_factors`.

---

## 5. Design decisions — **resolved**

### 5.1 Min-max normalisation — *resolved: paper-faithful by default, switchable off*

- **`diag` mode:** min-max **on by default**, faithful to Eq. (4) of the paper. Applied to the
  *instantaneous* factor `H_D_i`, `S_D_i` **before** the EMA — the placement used by the official
  repository (`AdaFisher/optimizers/AdaFisher.py:412`, `:431`), not after accumulation.
- **Opt-out:** `minmax_normalization=False` reproduces `FisherAdapTune/scripts/adafisher.py`
  bit-exactly. This is the setting under which the non-regression test of §6.1 runs.
- **The four new modes never apply min-max.** It would destroy TKFAC's trace preservation (Thm 4.1)
  and make `‖F − F̃‖_F` meaningless. Damping is `λ` only, as prescribed by each paper.

Implementation note: `MinMaxNormalization` must reproduce the official semantics, including the
`smart_detect_inf` pre-pass (`+inf → 1`, `−inf → 0`) and the `epsilon = 1e-6` guard in the denominator.
`_smart_detect_inf` already exists unused at `adafisher.py:21` and can be reused verbatim.

### 5.2 Normalisation layers — *resolved: option (c), generalised*

Per the §1.5 reading, for a normalisation layer `A` is a **scalar** and `B = S ∈ R^{C×C}`. The four
modes therefore apply with no invention: `F = a·S`, eigenbasis `Q_S`, well-defined EKFAC/TEKFAC
diagonal correction. Normalisation layers are **not excluded** from any mode — AdaFisher explicitly
attributes a generalisation gain to their treatment (§Ablation: "normalization layers, as detailed in
Proposition 3.1, significantly enhances the generalization"), and excluding them would bias the §6.3
comparison.

**Caveat to implement carefully.** Building a *full* `S` for these layers requires following
**Prop. 3.1**, not line 141 of the code. `_ComputeSD._batchnorm2d` computes
`einsum("i,i->i", Σ_T s, Σ_T s)` — *sum then square*; the corresponding full version
`outer(Σ_T s, Σ_T s)` is **rank 1**, which yields a degenerate eigenbasis (a single informative
eigenvector). Prop. 3.1 writes `S_i = (1/|T_i|)Σ_{x∈T_i} s_{i,x}s_{i,x}ᵀ` — *square then sum* — which
is potentially full rank. **The four new modes follow Prop. 3.1; the `diag` mode keeps the code's
formula** and stays bit-exact. Fallback if the measured behavioural gap becomes a problem: reuse
AdaFisher's diagonal treatment for normalisation layers regardless of mode.

### 5.3 Scale conventions — *resolved: AdaFisher convention everywhere*

`EKFAC-pytorch` multiplies `grad_output` by the batch size in its hook
(`ekfac.py::_save_grad_output`) and divides by `num_locations`, whereas AdaFisher averages over
`batch × spatial_size`. These differ by a known scalar factor, absorbable into `λ` and the step size.
**Adopt the AdaFisher convention throughout** (mean over `batch × spatial`), so that `diag` stays
bit-exact and all five modes share the same `λ` scale. Verified explicitly in test (i).

### 5.4 Empirical Fisher vs. type-II/MC Fisher — *resolved: stay with the empirical Fisher*

All modes derived from the true-label `grad_output` compute the **empirical Fisher**, not the Fisher.
`kfac_from_scratch_2507.05127.pdf` (cheatsheet §6) provides the test cases that distinguish them, and
`empirical_fisher_limits_1905.12558.pdf` documents the limitations. **Stay on the empirical Fisher** —
this is what AdaFisher does, what EKFAC does (§4: "For all our experiments KFAC and EKFAC approximate
the empirical Fisher G"), and what TKFAC/TEKFAC do. Do not open this line of work; it is recorded in
`CLAUDE.md` as a known limitation.

---

## 6. Empirical validation criteria

### 6.1 Non-regression / Frobenius dominance

Bench: a toy MLP in very small dimension (`d_in ≤ 8`, `d_out ≤ 6`, batch of 64) where
`F_l = E[h̄h̄ᵀ ⊗ δδᵀ] ∈ R^{(d_in·d_out)²}` is computable **exactly** by per-sample accumulation.

Assertions **guaranteed by theorems** (hard failures):

| Assertion | Source |
|---|---|
| `‖F − F̃_EKFAC‖_F ≤ ‖F − F̃_KFAC‖_F` | EKFAC Thm 2/3, Appendix A.1 |
| `‖F − F̃_TEKFAC‖_F ≤ ‖F − F̃_TKFAC‖_F` | TEKFAC Thm 3.1 |
| `tr(F̃_TKFAC) = tr(F)` (within `1e-10`) | TKFAC Thm 4.1 / Lemma 4.1 |
| `Q_A`, `Q_B`, `Q_Φ`, `Q_Ψ` orthogonal (`QᵀQ = I`) | — |
| `diag` mode with `minmax_normalization=False` bit-exact vs. `AdaFisherBackbone._get_F_tilde` | §1.2 |
| `diag` mode with `minmax_normalization=True` matches Eq. (4) semantics of the official repo | `AdaFisher.py:412`, `:431`, `:514` |

**Comparison against the `diag` mode — measured, not asserted.** The original brief asked that the
four modes also dominate `diag`. This is **not** a theorem: EKFAC's Thm 2/3 and TEKFAC's Thm 3.1
compare approximations sharing the *same orthogonal basis*, and AdaFisher's `F̃_D` does not have that
form; nothing in the AdaFisher paper establishes a Frobenius bound relative to `F`. Moreover, min-max
normalisation puts `F̃_D` on a different scale from `F`, so a raw `‖F − F̃_D‖_F` is arbitrary.

**Resolved:** measure this comparison after **optimal scalar rescaling**,

```
e(F̃) = min_{c>0} ‖F − c·F̃‖_F ,     closed form  c* = ⟨F, F̃⟩ / ‖F̃‖²_F
```

and report `e(F̃)` for all five modes as a **measurement** in the test report, not as a pass/fail
assertion. The expectation is that the four modes dominate; a failure would not be a bug. The
rescaled metric is scale-invariant, so it is also the right way to compare `diag` with and without
min-max.

### 6.2 EKFAC ↔ TEKFAC consistency

Degenerate test: forcing `tr(Φ)=tr(Ψ)=1` then substituting `Φ := A`, `Ψ := B`, `δ := 1` gives
`Q_Φ = Q_A` and `Q_Ψ = Q_B`, so `Θ` (Eq. 3.2) and `s*` (EKFAC §3.2) are **the same quantity**.
Assertion: `allclose(precondition_TEKFAC(M), precondition_EKFAC(M), rtol=1e-6)` at identical `λ`. This
is the most discriminating test in the project: it simultaneously validates both projections, both
rescalings, and agreement on the `rvec` convention.

Precaution: `torch.linalg.eigh` fixes neither the sign nor the ordering of eigenvectors. The test must
compare the **applied preconditioners**, never the bases themselves.

### 6.3 Equal-budget convergence

Primary bench: **8-layer MNIST auto-encoder**, encoder `784-1000-500-250-30` with sigmoid activations
plus an untied symmetric decoder (`ekfac_1806.03884.pdf` §4.1) — the historical bench shared by K-FAC,
EKFAC and Desjardins et al., and small enough for all five modes to fit in memory.

**Lot 8** (was: *deferred, on request*; requested and planned in `plan_lot8.md`): CIFAR-10
classification on ResNet-50 / ViT-S/4, trained from scratch, with **Adam and AdamW** as additional
baselines — seven arms per model, compared under AdaFisher's own wall-clock-time protocol
(`adafisher_2405.16397.pdf` §5). Its two prerequisites named below (SUA, lot 6; the equal-wall-clock
harness, lot 7) are in place.

Metric: training loss as a function of (a) epoch and (b) wall-clock time — both, since the overhead is
the subject.

**Overhead figures, MNIST auto-encoder** (layers, bias included: `785×1000`, `1001×500`, `501×250`,
`251×30`, `31×250`, `251×500`, `501×1000`, `1001×784`):

| Item | Cost |
|---|---|
| Model fwd+bwd, batch 500 (`≈6·B·P`, `P ≈ 2.84e6`) | `≈ 8.5` GFLOP/step |
| `diag` mode, building `v^(t)` | `≈ 2.8` MFLOP/step — **negligible** |
| KFE projection `Q_Bᵀ M Q_A` + inverse, all non-`diag` modes (`Σ 4·d_in·d_out·(d_in+d_out)`) | `≈ 18` GFLOP/step — **≈ 2.1× fwd+bwd** |
| Full eigendecomposition / inversion (`Σ d³`) | `≈ 5.5` GFLOP per sweep |
| … amortised at `T_eig = 50` | `≈ 0.11` GFLOP/step — **negligible next to the projection** |

Takeaway: **the dominant cost is not the `O(d³)` eigendecomposition but the `O(d²)` per-step
projection**, which cannot be amortised. This is consistent with the ~2× wall-clock overhead reported
in EKFAC §4. The equal-budget requirement of criterion (iii) must therefore be fixed in **wall-clock
time**, otherwise the comparison is rigged in favour of the expensive modes.

**Memory — the real blocker on convolutional nets.** ResNet-18's largest conv factor has
`d_in = 512·3·3+1 = 4609`, i.e. `A ∈ R^{4609²}` = **85 MB in fp32 for a single layer** (vs. 18 KB for
the `diag` mode's `H_D` vector, a `4609×` factor). Across `layer3`+`layer4` this exceeds **0.4 GB** of
factors, before eigenbases double it. A single `eigh` at `d = 4609` costs `≈ 1e11` FLOP.
**Practical consequence: ResNet/ViT work will require the SUA approximation**
(`EKFAC-pytorch/ekfac.py::_precond_sua_ra`, `_to_kfe_sua`, `_get_gathering_filter`), which replaces the
patch factor `(C_in·k²)` by a channel factor `C_in`, bringing `4609 → 512`. Planned as lot 6, before
the CIFAR-10 work — confirmed on ResNet-50 in `plan_lot8.md` §0.4/§0.5 (`d_in` 4608 → 512, total `A`
storage 498.5 → 96.1 MB, per-step projection 288 → 107 GFLOP).

---

## 7. Dependencies on upstream repositories

| To reimplement (paper formulas) | To adapt from `EKFAC-pytorch` (no direct copy) | To reuse from `FisherAdapTune` |
|---|---|---|
| TKFAC Thm 4.1 / Eq. 4.9 (`δ, Φ, Ψ`), Thm 4.3 conv | hook structure `_save_input` / `_save_grad_output` (`kfac.py`) | per-layer-type `(h̄, δ)` extraction, `adafisher.py:52-155` |
| TEKFAC Eq. 3.2-3.5 + Alg. 1 (`Θ`, trace-adaptive damping, `T_FIM/T_EIG/T_RE`) | `_compute_kfe` (`ekfac.py`): `eigh` of `xxt`/`ggt` → `kfe_x`, `kfe_gy` | `_extract_patches`, `adafisher.py:33-50` |
| Normalisation-layer treatment, Prop. 3.1 (covered by none of the four papers) | `_precond_ra` / `_precond_intra` (`ekfac.py`): EKFAC Alg. 1's two variants | `step` loop and module↔parameter pairing, `adafisher.py:275-307` |
| K-FAC factored `π` damping (§6.3/6.6) | `_inv_covs` (`kfac.py`): `√(επ)` regularisation and inversion | Adam wrapper `_step`, `adafisher.py:258-273` |
| Min-max normalisation, Eq. (4) (§5.1) | `_precond_sua`, `_to_kfe_sua`, `_get_gathering_filter`: SUA conv case (lot 6) | `_check_dim`, `adafisher.py:225-230`; `_smart_detect_inf`, `adafisher.py:21` |

Known limitations of `EKFAC-pytorch`, not to be reproduced: hooks only `Linear`/`Conv2d` (no
normalisation layers); overwrites `weight.grad` instead of producing a `v^(t)` for an Adam wrapper;
different scale convention (§5.3); `ekfac.py` rejects `alpha != 1` outside `ra` mode.

---

## 8. Proposed breakdown (one iteration at a time)

| Lot | Content | Exit test |
|---|---|---|
| 1 | `optimizer.py` + `factors.py` + `base.py` + `diag.py` (incl. min-max toggle) | `diag(minmax=False)` bit-exact vs. `AdaFisherBackbone._get_F_tilde`; `diag(minmax=True)` matches Eq. (4); MNIST auto-encoder reproduces the AdaFisher curve |
| 2 | `kfac.py`, `ekfac.py` (`Linear` only) | §6.1 assertions 1, 4, 5 |
| 3 | `tkfac.py`, `tekfac.py` (`Linear` only) | §6.1 assertions 2, 3 + §6.2 consistency test |
| 4 | `Conv2d` (KFC) | §6.1 replayed on a toy `3×3`, 2-channel conv |
| 5 | Normalisation layers (option (c), §5.2) | loss non-regression vs. `diag` on a small CNN |
| 6 | SUA approximation for conv layers | §6.1 replayed; memory footprint measured on ResNet-18 |
| 7 | Convergence bench at equal wall-clock budget, 5 modes | §6.3 |
| 8 | CIFAR-10 from-scratch training, ResNet-50 / ViT-S/4, Adam + AdamW baselines (`plan_lot8.md`) | §6.3's WCT protocol, 7 arms × 2 models; a ViT-exposed `optimizer.py` pairing bug fixed (`plan_lot8.md` §0.3) |

---

## 9. Resolved questions

| # | Question | Decision |
|---|---|---|
| 1 | Normalisation layers in the four new modes | **Option (c)** — generalised: `A` scalar, `S` full per Prop. 3.1 (§5.2) |
| 2 | Dominance over `diag`: assertion or measurement? | **Measurement**, after optimal scalar rescaling `c* = ⟨F,F̃⟩/‖F̃‖²` (§6.1) |
| 3 | Min-max in the `diag` mode | **Faithful to Eq. (4) by default**, switchable off via `minmax_normalization=False` (§5.1) |
| 4 | Primary bench | **MNIST auto-encoder** (lots 1, 7). CIFAR-10 ResNet-50/ViT-S/4 from-scratch + Adam/AdamW baselines: requested, lot 8 (§6.3, `plan_lot8.md`) |
| 5 | Package location | **`src/adafisher_modes/`** (§2.1) |

**Working language:** all code, comments, docstrings, reports and documentation are written in
English, to the standard of a well-maintained academic repository. French output on explicit request
only.
