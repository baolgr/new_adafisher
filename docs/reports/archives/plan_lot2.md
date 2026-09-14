# Lot 2 Implementation Plan — `kfac.py`, `ekfac.py` (`Linear` only)

**Status:** draft, awaiting implementation in this same session. No lot-2 code written yet.
**Parent plan:** `docs/reports/plan.md` §8, lot 2. This document refines lot 2 to
implementation-ready precision, the way `plan_lot1.md` did for lot 1. It does not revisit lot 1
(closed, 10/10 tests passing) or lots 3-7.

Exit criteria (`plan.md` §8, lot 2 row, unpacked against §6.1):

1. `‖F − F̃_EKFAC‖_F ≤ ‖F − F̃_KFAC‖_F` (EKFAC Thm 2/3, Appendix A.1) on a toy `Linear` layer where
   `F` is computable exactly by per-sample accumulation.
2. `Q_A`, `Q_B` orthogonal (`QᵀQ = I`).
3. No regression: the full lot-1 suite (10 tests) still passes unmodified.

---

## 0. Five design decisions this lot must resolve before code exists

`plan.md` §4.2/§4.3 and the `CLAUDE.md` cheat sheet already fix the *mathematics* of `kfac` and
`ekfac`. What they do not fix is how that mathematics meets `FisherApproximation`'s actual
hook-driven, `TCov`/`T_inv`/`T_eig`-cadenced architecture (the same category of gap `plan_lot1.md`
§0 closed for `diag`). Five points, resolved here, all consistent with `plan_lot1.md`'s corrections
(the ABC itself is *not* touched — its `update_input_factor`/`update_output_factor`/`refresh`/
`precondition` signature already generalises cleanly to non-diagonal modes).

### 0.1 Full factors are built from an augmented batch tensor, not from `factors.py`'s diagonals

`compute_h_diag`/`compute_s_diag` (lot 1) reduce straight to a diagonal via `einsum`; there is no
intermediate full matrix to reuse. Two new functions are added to `factors.py`, exactly the
extension point `plan_lot1.md` §1 pre-registered:

```python
def compute_h_full(h: Tensor, layer: Module) -> Tensor: ...   # A = E[h_bar h_bar^T], Linear only
def compute_s_full(s: Tensor, layer: Module) -> Tensor: ...   # B = E[delta delta^T],  Linear only
```

Both dispatch on `isinstance(layer, Linear)` and raise `NotImplementedError` otherwise (`Conv2d`,
`BatchNorm2d`, `LayerNorm` are lots 4-5, per `plan.md` §8 — silently falling back to some default
would hide a real gap, so an explicit, typed error is the honest failure mode, mirroring
`compute_h_diag`'s own `raise NotImplementedError` branch for unsupported types).

`compute_h_full` reuses a new, also-public `augment_linear_input(h, layer) -> Tensor` — flattens
`h` to `(N, d_in)` and appends a ones column iff `layer.bias is not None`, i.e. the bias-augmented
`h̄` of `plan.md`'s header formula. This is the same augmentation `_h_linear` already performs
inline (`adafisher.py:61-68`, ported at `factors.py::_h_linear`); it is pulled out to a named
function here (not duplicated) because `ekfac.py` needs the raw augmented **batch**, not just its
reduction — see §0.3.

**Scale convention** (`plan.md` §5.3): mean over batch, `A = h̄ᵀh̄ / N`, `B = sᵀs / N` — matching
`diag`'s `/batch_size` and *not* `EKFAC-pytorch`'s `grad_output * batch_size` convention
(`ekfac.py::_save_grad_output`). Verified indirectly: `torch.diagonal(compute_h_full(h, layer))`
and `compute_h_diag(h, layer)` are the same real number (not bit-identical — matmul- vs.
einsum-order — but `allclose`), checked in `tests/test_full_factors_match_diag.py` (§3.1).

### 0.2 Weight/bias joint handling: a small shared module, not a `diag.py`-style inline

`diag.py` inlines its own `F[:, :-1]` / `F[:, -1:]` split. For `kfac`/`ekfac` the identical pattern
recurs twice in this lot and will recur twice more in lot 3 (`tkfac`, `tekfac`, per `plan.md` §2.1's
file layout) — four consumers of the same eight lines is past the point where duplication is a
reasonable default (`plan.md` §2.2's own argument for one shared ABC applies here at smaller scale).
A new **private** module, `approximations/_kron_utils.py`:

```python
def augment_direction(weight_direction: Tensor, bias_direction: Optional[Tensor]) -> Tensor: ...
def split_direction(M: Tensor, weight_shape, bias_shape: Optional[torch.Size]): ...
```

`diag.py` is **not** refactored to use it — `CLAUDE.md`'s "change only what's requested, no
unsolicited cleanup" applies to lot-1 code that already has passing tests tied to its exact form.

### 0.3 `ekfac`'s `s*`: intra-batch (Algorithm 1), tracked by this project's own EMA — not either
    paper variant verbatim

`ekfac_1806.03884.pdf` §4 names two variants: **EKFAC** re-estimates `s*_i = E[((U_A⊗U_B)ᵀ∇θ)_i²]`
from scratch every minibatch, using **per-example** gradients in the KFE basis (Algorithm 1); **
EKFAC-ra** instead tracks a running average of `bs·(batch-mean gradient in KFE)²`
(`EKFAC-pytorch/ekfac.py::_precond_ra`, line 103), avoiding the need for per-example gradients at
the cost of a cruder estimator.

AdaFisher's hooks already hand `FisherApproximation` the **raw batch** tensors (`input[0].data`,
`grad_output[0].data`), not a single averaged gradient — exactly what per-example EKFAC needs,
since a `Linear` layer's per-example gradient is `δ_n h̄_nᵀ`, directly reconstructable from the
batch's rows without any per-example-gradient autograd machinery. Concretely:

- `update_input_factor` caches the augmented batch `h̄` (`self._cached_h_bar[module]`) in addition
  to folding it into `A`'s EMA.
- `update_output_factor` (same step, same batch — forward always precedes backward for one
  `loss.backward()` call, so this pairing is safe under the standard training loop, exactly as
  `EKFAC-pytorch`'s own `state[mod]['x']` caching relies on) retrieves `h̄`, and if an eigenbasis
  `(Q_A, Q_B)` already exists from a previous `refresh`, projects both `h̄` and `s` into it and forms
  the intra-batch estimate `g²_{i,j} = (1/N)Σ_n (s_kfe_{n,i})² (h_kfe_{n,j})²` — the diagonal
  entries of `E[((Q_A⊗Q_B)ᵀ∇θ)²]` restricted to this batch, i.e. literally Algorithm 1's
  `COMPUTE SCALINGS` step.
- This `g²` is then folded into `self._s_star[module]` via the **same** `update_running_avg` (Eq. 3
  gammas) every other factor in this codebase uses, rather than Algorithm 1's "from scratch" reset
  or `-ra`'s batch-mean-squared trick.

This is a deliberate **third choice**, distinct from both named variants: it keeps the intra-batch
statistic Algorithm 1 defines (does not need the `-ra` approximation, since real per-example data is
available), while keeping every mode's factor-smoothing mechanism identical (`CLAUDE.md`: "Every-
thing else — hooks, factor EMA, β₁ momentum ... is identical across the five modes"). Re-deriving
`-ra`'s own smoothing would introduce a second, inconsistent EMA scheme for no benefit here. Flagged
explicitly since it is a genuine design choice, not a transcription of either paper variant.

**Bootstrap.** `s*` cannot be estimated before an eigenbasis exists. `refresh()` initialises
`self._s_star[module] = ones_like(...)` the first time it computes `(Q_A, Q_B)` for a module —
the direct generalisation of `diag`'s own `H.new_ones(...)` bootstrap (`plan_lot1.md` §2.2) from a
diagonal to a full `(d_out, d_in_aug)` tensor. `precondition` never needs to special-case a missing
`s*`: by construction (§0.5 below) `refresh` always runs, and thus `s*` always exists, before the
first `precondition` call for any module (identical reasoning to why `diag`'s `_H`/`_S` are always
populated in time — hooks and `refresh`/`precondition` are strictly ordered within one `step()`).

### 0.4 The `rvec` convention, derived precisely (not just cited)

`plan.md` §4's preamble states PyTorch works in `rvec` (`B⊗A` applied as `M ↦ B⁻¹MA⁻¹`) while the
papers use `cvec` (`A⊗B`), citing `kfac_from_scratch_2507.05127.pdf` Def. 1/2/23 as "the central
pitfall". Re-derived here from scratch, independent of that PDF, using only how `torch.flatten()`
(row-major) relates to the standard *column-major* Kronecker-vec identity `(P⊗Q)vec_c(X)=vec_c(QXPᵀ)`
(this identity itself is what `kfac_1503.05671.pdf` §4.2 uses, unnumbered display equation after
eq. 6, `U_i = G_{i,i}⁻¹ V_i Ā_{i−1,i−1}⁻¹`):

- A per-example weight gradient is `ΔW_n = δ_n h̄_nᵀ`, shape `(d_out, d_in_aug)`.
- `vec_r(ΔW_n)` (row-major flatten, what `.flatten()` gives) satisfies `vec_r(ΔW_n) = δ_n ⊗ h̄_n`
  (direct index check: `vec_r(uvᵀ)_{i·|v|+j} = u_i v_j = (u⊗v)_{i·|v|+j}`).
- Hence the exact per-example Fisher contribution, in this flattening, is
  `vec_r(ΔW_n)vec_r(ΔW_n)ᵀ = (δ_nδ_nᵀ) ⊗ (h̄_nh̄_nᵀ) = B_n ⊗ A_n` — **`B` outer, `A` inner**, matching
  `plan.md` §4's "`B⊗A` in `rvec`" claim exactly, and fixing the ordering this lot's `f_tilde()`
  debug accessors (§0.5) must use to be comparable to a brute-force per-example `F`.
- Using `vec_r(X) = vec_c(Xᵀ)` and the column-major identity above: for `M` of shape
  `(d_out, d_in_aug)`, `(B⊗A)vec_r(M) = (B⊗A)vec_c(Mᵀ) = vec_c(A Mᵀ Bᵀ) = vec_c(AMᵀB)` (`B`
  symmetric) `= vec_r((AMᵀB)ᵀ) = vec_r(BᵀMAᵀ) = vec_r(BMA)` (`A` symmetric). So the **forward**
  operator is `M ↦ BMA`; its inverse — the natural-gradient direction — is `M ↦ B⁻¹MA⁻¹`, confirming
  `CLAUDE.md`'s cheat sheet (`B̃⁻¹ M Ã⁻¹`) from first principles rather than by citation alone.

**Consequence for correctness, independent of this derivation being "the" literature convention:**
what actually matters for `precondition` and `f_tilde` to agree, and for the dominance theorem to
hold empirically, is that `A` (built from `h̄`) and `B` (built from `δ`) are used *consistently* the
same way in the brute-force test oracle, in `f_tilde()`, and in `precondition()`. A global
inner/outer swap would not break the K-FAC/EKFAC identities (Lemma 1 is convention-agnostic), so
`tests/test_frobenius_dominance.py` (§3.2) is the actual authority; §0.4 is recorded for the required
per-paper-equation citation discipline (`CLAUDE.md`, "Hard constraints"), not as an unverified
assumption.

### 0.5 `f_tilde()` debug accessors, dense reconstruction — needed by the dominance test

`plan_lot1.md` §2.2 already anticipated this: "lot 2/3's Frobenius-dominance tests will need every
mode ... to expose 'the combined `F̃` for this module' in one comparable form." `precondition()`
never forms a dense `(d_out·d_in_aug)²` matrix (the entire point of the factored/eigenbasis form is
avoiding that), but a debug/test-only `f_tilde(module) -> Tensor` is added to both `kfac.py` and
`ekfac.py`, exactly mirroring `diag.py`'s existing accessor:

```python
# kfac:   f_tilde(module) = kron(B_tilde, A_tilde)                       # dense, O(d^2) to build
# ekfac:  f_tilde(module) = kron(Q_B, Q_A) @ diag((s* + Lambda).flatten()) @ kron(Q_B, Q_A).T
```

Both reconstruct from the *current* `_A`/`_B` (kfac) or `_Q_A`/`_Q_B`/`_s_star` (ekfac) — i.e. they
reflect any EMA drift since the last `refresh()`, exactly like `diag.f_tilde()` reflects the current
`_H`/`_S`. The dominance test calls `refresh()` immediately before `f_tilde()` so the two coincide;
this is documented in both docstrings as the intended usage, not assumed silently.

---

## 1. File-by-file design

```
src/adafisher_modes/
├── factors.py                       # + augment_linear_input, compute_h_full, compute_s_full
└── approximations/
    ├── _kron_utils.py               # NEW — augment_direction / split_direction (§0.2)
    ├── kfac.py                      # NEW — KFACApproximation
    ├── ekfac.py                     # NEW — EKFACApproximation
    └── __init__.py                  # MODES gains "kfac", "ekfac"
src/adafisher_modes/__init__.py      # exports KFACApproximation, EKFACApproximation
tests/
├── test_full_factors_match_diag.py  # NEW — §3.1
├── test_kfac_ekfac_precondition.py  # NEW — §3.3 (shape/consistency/orthogonality)
└── test_frobenius_dominance.py      # NEW — §3.2 (exit criteria 1, 2)
```

`optimizer.py` and `approximations/base.py` are **unchanged** — the entire point of lot 1's ABC
(`plan_lot1.md` §0) was for this to be true. `docs/reports/plan.md` is unchanged (it is the
overall plan, not a per-lot log); `CLAUDE.md`'s status header and file-layout comment are updated
at the end, once the exit tests pass, exactly as they were after lot 1.

### 1.1 `factors.py` additions

```python
def augment_linear_input(h: Tensor, layer: Linear) -> Tensor:
    if h.ndim > 2:
        h = h.reshape(-1, h.shape[-1])
    if layer.bias is not None:
        return cat([h, h.new_ones(h.size(0), 1)], 1)
    return h

def _h_full_linear(h: Tensor, layer: Linear) -> Tensor:
    h_bar = augment_linear_input(h, layer)
    return h_bar.t() @ h_bar / h_bar.size(0)

def compute_h_full(h: Tensor, layer: Module) -> Tensor:
    if isinstance(layer, Linear):
        return _h_full_linear(h, layer)
    raise NotImplementedError(f"compute_h_full only supports Linear so far (lot 2 scope); got {type(layer)}")

def _s_full_linear(s: Tensor, layer: Linear) -> Tensor:
    if s.ndim > 2:
        s = s.reshape(-1, s.shape[-1])
    return s.t() @ s / s.size(0)

def compute_s_full(s: Tensor, layer: Module) -> Tensor:
    if isinstance(layer, Linear):
        return _s_full_linear(s, layer)
    raise NotImplementedError(f"compute_s_full only supports Linear so far (lot 2 scope); got {type(layer)}")
```

### 1.2 `approximations/_kron_utils.py`

```python
def augment_direction(weight_direction: Tensor, bias_direction: Optional[Tensor]) -> Tensor:
    if bias_direction is None:
        return weight_direction
    return cat([weight_direction, bias_direction.unsqueeze(1)], dim=1)

def split_direction(M: Tensor, weight_shape, bias_shape):
    if bias_shape is None:
        return M.reshape(weight_shape)
    return M[:, :-1].reshape(weight_shape), M[:, -1:].reshape(bias_shape)
```

### 1.3 `approximations/kfac.py` — `KFACApproximation`

State: `_A`, `_B` (EMA'd full factors, bootstrap `eye(d)` at `step == 0` — the full-matrix analogue
of `diag`'s `ones` vector bootstrap, since `diag(ones) = I`), `_A_inv`, `_B_inv` (refreshed every
`T_inv` steps).

```python
class KFACApproximation(FisherApproximation):
    def __init__(self, Lambda=1e-3, gammas=(0.92, 0.008), T_inv=100, pi=True): ...

    def update_input_factor(self, module, h, step):
        A_i = compute_h_full(h, module)
        if step == 0: self._A[module] = eye(A_i.size(0), ...)
        update_running_avg(A_i, self._A[module], self.gammas)

    def update_output_factor(self, module, s, step):   # symmetric, self._B

    def _pi(self, A, B):
        if not self.pi: return A.new_tensor(1.0)
        return ((A.trace() / A.size(0)) / (B.trace() / B.size(0))).sqrt()   # kfac_1503.05671 S6.3

    def _damped_factors(self, module):
        A, B = self._A[module], self._B[module]
        pi = self._pi(A, B)
        s = self.Lambda ** 0.5
        return A + (pi * s) * eye(...), B + (s / pi) * eye(...)

    def refresh(self, module, step):
        if step % self.T_inv != 0: return
        A_tilde, B_tilde = self._damped_factors(module)
        self._A_inv[module], self._B_inv[module] = A_tilde.inverse(), B_tilde.inverse()

    def f_tilde(self, module):
        A_tilde, B_tilde = self._damped_factors(module)
        return kron(B_tilde, A_tilde)                   # S0.4 — B outer, A inner

    def precondition(self, module, weight_direction, bias_direction):
        M = augment_direction(weight_direction, bias_direction)
        direction = self._B_inv[module] @ M @ self._A_inv[module]
        return split_direction(direction, weight_direction.shape,
                                None if bias_direction is None else bias_direction.shape)
```

`π` (§0.4/`kfac_1503.05671.pdf` §6.3) defaults on, matching the paper's recommendation ("one of the
best and most robust choices"); off (`pi=False`) reduces to `π=1`, i.e. plain, non-factored
Tikhonov damping split evenly between `A` and `B`. Exposed as a constructor kwarg like `diag`'s
`minmax_normalization` — mode-specific, not part of the shared ABC.

### 1.4 `approximations/ekfac.py` — `EKFACApproximation`

State: `_A`, `_B` (identical to `kfac`, same bootstrap), `_Q_A`, `_Q_B` (eigenvectors, refreshed
every `T_eig` steps), `_s_star` (EMA'd per §0.3), `_cached_h_bar` (transient, popped every call).

```python
class EKFACApproximation(FisherApproximation):
    def __init__(self, Lambda=1e-3, gammas=(0.92, 0.008), T_eig=100): ...

    def update_input_factor(self, module, h, step):
        h_bar = augment_linear_input(h, module)
        A_i = h_bar.t() @ h_bar / h_bar.size(0)
        if step == 0: self._A[module] = eye(A_i.size(0), ...)
        update_running_avg(A_i, self._A[module], self.gammas)
        self._cached_h_bar[module] = h_bar

    def update_output_factor(self, module, s, step):
        B_i = compute_s_full(s, module)
        if step == 0: self._B[module] = eye(B_i.size(0), ...)
        update_running_avg(B_i, self._B[module], self.gammas)
        h_bar = self._cached_h_bar.pop(module, None)
        if module in self._Q_A and h_bar is not None:
            s_flat = s.reshape(-1, s.shape[-1]) if s.ndim > 2 else s
            h_kfe, s_kfe = h_bar @ self._Q_A[module], s_flat @ self._Q_B[module]
            g2 = (s_kfe.t() ** 2) @ (h_kfe ** 2) / h_bar.size(0)     # Algorithm 1, intra-batch
            update_running_avg(g2, self._s_star[module], self.gammas)

    def refresh(self, module, step):
        if step % self.T_eig != 0: return
        _, Q_A = linalg.eigh(self._A[module])
        _, Q_B = linalg.eigh(self._B[module])
        if module not in self._Q_A:
            self._s_star[module] = Q_B.new_ones(Q_B.size(0), Q_A.size(0))   # bootstrap, S0.3
        self._Q_A[module], self._Q_B[module] = Q_A, Q_B

    def f_tilde(self, module):
        Q_A, Q_B = self._Q_A[module], self._Q_B[module]
        scale = (self._s_star[module] + self.Lambda).flatten()
        Q = kron(Q_B, Q_A)
        return Q @ diag(scale) @ Q.t()

    def precondition(self, module, weight_direction, bias_direction):
        M = augment_direction(weight_direction, bias_direction)
        Q_A, Q_B = self._Q_A[module], self._Q_B[module]
        M_kfe = (Q_B.t() @ M @ Q_A) / (self._s_star[module] + self.Lambda)
        direction = Q_B @ M_kfe @ Q_A.t()
        return split_direction(direction, weight_direction.shape,
                                None if bias_direction is None else bias_direction.shape)
```

### 1.5 `approximations/__init__.py`

```python
MODES: Dict[str, Callable[..., FisherApproximation]] = {
    "diag": DiagApproximation,
    "kfac": KFACApproximation,
    "ekfac": EKFACApproximation,
}
```

No change to `optimizer.py`: `T_inv`/`T_eig`/`pi` already flow through unmodified via the existing
`**mode_kwargs` pass-through (`plan_lot1.md` §2.4), exactly as designed.

---

## 2. Tests

### 2.1 `test_full_factors_match_diag.py`

Sanity link between the new full factors and the already-validated diagonal ones: on a random
`Linear(7, 5, bias=True)` and a random batch, `torch.diagonal(compute_h_full(h, layer))` is
`allclose` (not `equal` — different reduction order, §0.1) to `compute_h_diag(h, layer)`, and
likewise for `compute_s_full`/`compute_s_diag`. Also checks the `bias=False` branch (no augmentation
column) and the `Conv2d`/`BatchNorm2d`/`LayerNorm` branches raise `NotImplementedError`.

### 2.2 `test_frobenius_dominance.py` — exit criteria 1 and 2

Per `plan.md` §6.1: a toy dimension (`d_in=7`, `d_out=5`, `N=64`) where the exact per-layer block
`F = E_n[ (δ_nδ_nᵀ) ⊗ (h̄_nh̄_nᵀ) ]` (§0.4 ordering) is built by explicit per-sample accumulation —
no real network or autograd needed: `h`, `s` are directly sampled i.i.d. Gaussian batches, exactly
the shape `update_input_factor`/`update_output_factor` consume; a throwaway `nn.Linear(7, 5)` is
used purely as `module` (for hook-dict keys and its `.bias is not None` check) and is never
forwarded.

```python
def exact_fisher_block(h, s, has_bias) -> Tensor:
    N = h.size(0)
    F = torch.zeros(...)
    for n in range(N):
        h_n = torch.cat([h[n], h.new_ones(1)]) if has_bias else h[n]
        F += torch.kron(torch.outer(s[n], s[n]), torch.outer(h_n, h_n))
    return F / N
```

Critically, `s*` must be estimated from the **same** `(h, s)` samples that define `F` (the theorem
is about a single, closed data-generating process — an `s*` estimated from a *different* batch than
`A`/`B`/`F` would not be testing Lemma 1/Theorem 2 at all). Driving sequence, with
`gammas=(1.0, 1.0)` (makes `update_running_avg` an exact assignment, `current ← new`, regardless of
the `eye`/`ones` bootstrap — the cleanest way to obtain *pure*, EMA-artifact-free factors from the
real, stateful classes without introducing a parallel pure-math API):

```python
kfac.update_input_factor(layer, h, 0); kfac.update_output_factor(layer, s, 0); kfac.refresh(layer, 0)
F_kfac = kfac.f_tilde(layer)

ekfac.update_input_factor(layer, h, 0); ekfac.update_output_factor(layer, s, 0)   # no eigenbasis yet
ekfac.refresh(layer, 0)                                                          # Q_A, Q_B from exact A, B; s* bootstrapped
ekfac.update_input_factor(layer, h, 1); ekfac.update_output_factor(layer, s, 1)  # SAME (h, s) -> s* := exact intra-batch estimate
F_ekfac = ekfac.f_tilde(layer)
```

`Lambda` is set very small (`1e-8`, not `0`) purely as numerical safety margin — with `N=64 ≫
d_in_aug=8, d_out=5` the raw `A`, `B` are full-rank almost surely, so no real damping is needed for
the theorem to hold; `1e-8` avoids relying on exact-singularity-never-happens luck without
perturbing the comparison.

Assertions:

- `torch.linalg.matrix_norm(F_exact - F_ekfac) <= torch.linalg.matrix_norm(F_exact - F_kfac)` —
  exit criterion 1 (EKFAC Thm 2/3).
- `torch.allclose(Q_A.T @ Q_A, eye(d_in_aug), atol=1e-5)` and likewise for `Q_B` — exit criterion 2.
- Internal consistency, independent of the §0.4 convention being exactly right: for a random
  direction `M`, `precondition(...)` reshaped/flattened equals `torch.linalg.solve(f_tilde(...),
  M.flatten())` for **both** `kfac` and `ekfac` — this is what would actually break if `f_tilde`'s
  dense reconstruction and `precondition`'s factored application disagreed on ordering, and is
  checked before trusting the dominance comparison above.
- Informational only (`plan.md` §6.1, "measured, not asserted"): print/log
  `e(F̃) = min_c‖F − cF̃‖_F` for `diag`, `kfac`, `ekfac` via the closed form
  `c* = ⟨F, F̃⟩/‖F̃‖_F²`, using `diag`'s existing `f_tilde()` reshaped to the same dense
  `(d_out·d_in_aug)²` comparison via `torch.diagflat`. No assertion on this value, per the already-
  resolved §9 answer 2 (measurement, not a pass/fail gate).

### 2.3 `test_kfac_ekfac_precondition.py`

Unit-level checks independent of the brute-force oracle:

- **Shape:** for a `Linear` with and without bias, `precondition` returns a tensor / 2-tuple of the
  correct shapes.
- **`π` toggling:** `pi=False` reduces `_pi` to exactly `1.0` (`torch.equal`), and changes the
  damping split (`A_tilde`/`B_tilde` differ from the `pi=True` run) without changing `A_tilde ⊗
  B_tilde`'s trace-normalized ballpark by more than the expected amount — a smoke test, not a
  precision one.
- **`T_inv`/`T_eig` cadence:** with `T_inv=3` (`T_eig=3`), `_A_inv`/`_Q_A` change only at steps
  `0, 3, 6, ...` even though `_A` itself (fed every step here, `TCov=1` in this test) keeps moving —
  checked by asserting `torch.equal` on the cached inverse/eigenbasis across non-refresh steps and
  `not torch.equal` (generically, with a fixed seed known to differ) across a refresh step.
- **Regression:** the full lot-1 suite (`tests/test_diag_*.py`, `tests/test_minmax_*.py`) is
  re-run unmodified as part of this lot's exit check (`plan.md` §8, "no regression").

---

## 3. Explicitly out of scope for lot 2

- `Conv2d` for `kfac`/`ekfac` (`compute_h_full`/`compute_s_full` raise `NotImplementedError`) — lot 4.
- `tkfac`, `tekfac` — lot 3, though `_kron_utils.py` is deliberately written so they can reuse it
  unchanged (§0.2).
- `EKFAC-pytorch`'s SUA approximation, `-ra` variant, `constraint_norm` rescaling — none of these
  are part of the cited theorems this lot's exit criteria check, and `plan.md` §7 already flags SUA
  as lot 6.
- Any change to `optimizer.py`, `approximations/base.py`, `diag.py`, or lot 1's tests.
- Benchmark integration (`benchmarks/mnist_autoencoder.py` comparing `kfac`/`ekfac` against `diag`)
  — that is lot 7's equal-wall-clock comparison, not a lot-2 exit criterion.

---

## 4. Points worth flagging explicitly (not blocking, per this session's operating mode)

1. §0.3 — the `s*` estimator is a deliberate third option (intra-batch statistic, EMA-tracked),
   not a transcription of either named paper variant. Mathematically justified (it *is* Algorithm
   1's `COMPUTE SCALINGS` step, just smoothed across `TCov`-separated batches the same way every
   other factor in this codebase already is) and it is what makes `ekfac` implementable at all
   inside AdaFisher's existing hook architecture without either discarding cross-batch smoothing or
   introducing a second, inconsistent smoothing rule.
2. §0.2 — a new private shared module (`_kron_utils.py`) for weight/bias plumbing, justified by four
   forthcoming consumers (this lot's two, plus lot 3's two), without touching `diag.py`.
3. `pi=True` default for `kfac` — matches the K-FAC paper's own stated recommendation, not an
   arbitrary choice.
