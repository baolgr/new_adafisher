# Lot 1 Implementation Plan — `optimizer.py` + `factors.py` + `base.py` + `diag.py`

**Status:** draft, awaiting validation. No implementation code written yet.
**Parent plan:** `docs/reports/plan.md` §8, lot 1. This document only refines lot 1 to
implementation-ready precision; it does not revisit lots 2-7.

Exit criteria (from `plan.md` §8, unchanged):

1. `diag(minmax_normalization=False)` bit-exact vs. `AdaFisherBackbone._get_F_tilde`.
2. `diag(minmax_normalization=True)` matches the Eq. (4) semantics of the official repository.
3. The MNIST auto-encoder reproduces the AdaFisher training curve.

---

## 0. Three corrections to the already-approved ABC (`plan.md` §2.2)

Writing the ABC to implementation precision surfaced three points where the signature sketched in
`plan.md` §2.2 does not survive contact with the actual hook/state architecture of
`reference_repos/FisherAdapTune/scripts/adafisher.py`. None of these change the resolved design
decisions of §5/§9 (min-max default, normalisation-layer handling, etc.) — they only fix the method
signatures of `FisherApproximation` before lot 2 builds on them. Flagging them explicitly rather than
adjusting silently, since lot 2/3 depend on this interface.

### 0.1 `update_factors(module, A, B, step)` → two hook-driven calls

The plan assumed one call receiving both factors together. But `H_bar_D` and `S_D` are populated by
**two independent hooks** that fire at different times — forward ([`_save_input`, adafisher.py:188-193](reference_repos/FisherAdapTune/scripts/adafisher.py#L188-L193))
and backward ([`_save_grad_output`, adafisher.py:195-200](reference_repos/FisherAdapTune/scripts/adafisher.py#L195-L200)).
There is no moment where both raw factors are simultaneously available to hand to a single method.
Resolved signature:

```python
def update_input_factor(self, module: Module, h: Tensor, step: int) -> None: ...   # forward hook
def update_output_factor(self, module: Module, s: Tensor, step: int) -> None: ...  # backward hook
```

### 0.2 `precondition(module, direction)` → whole-module call, not per-parameter

`_get_F_tilde` ([adafisher.py:209-223](reference_repos/FisherAdapTune/scripts/adafisher.py#L209-L223))
builds `kron(H, S) + Lambda` **once** per module and slices it into a weight part and a bias part.
If `precondition` is called once per parameter (as the singular-`direction` signature implies), a
non-diagonal mode would redo its most expensive step — the eigenbasis/inverse projection quantified
at ≈18 GFLOP/step in `plan.md` §6.3 — **twice** per module that has a bias, silently doubling the
already-dominant cost identified in that section. Resolved signature:

```python
def precondition(
    self, module: Module, weight_direction: Tensor, bias_direction: Optional[Tensor]
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Returns the preconditioned weight direction (and bias direction, if `module.bias`
    is not None), computing any shared, expensive state (inverse, eigenbasis, projection)
    exactly once per call."""
```

Mirrors the existing `_get_F_tilde` convention (single tensor vs. `[weight, bias]` pair,
[lines 218-223](reference_repos/FisherAdapTune/scripts/adafisher.py#L218-L223)) — no new convention
invented, just made explicit as part of the ABC contract.

### 0.3 `precondition` receives the raw `exp_avg`, not the bias-corrected `m̂`

`plan.md` §2.2 proposed passing `m̂ = exp_avg/(1-β₁ᵗ)` into `precondition`. Since a linear
preconditioner commutes with a later scalar multiply (`c·F̃⁻¹M = F̃⁻¹(cM)`), this is mathematically
equivalent to the original — but it is *not* the same sequence of floating-point operations as
`addcdiv_` ([adafisher.py:273](reference_repos/FisherAdapTune/scripts/adafisher.py#L273)), which folds
`1/(1-β₁ᵗ)` into a single Python-scalar `step_size` multiplying the whole quotient, never dividing
`exp_avg` by the bias correction as a separate tensor op. Passing the *raw* `exp_avg` and folding bias
correction into the optimizer's outer `alpha` keeps the new code as close as possible to the original's
op sequence, which matters for the trajectory-level check in test 2 below (§3.2). Resolved: `optimizer.py`
computes `param.add_(direction, alpha=-lr/bias_correction)`, exactly mirroring
`step_size = lr/bias_correction` at [adafisher.py:272](reference_repos/FisherAdapTune/scripts/adafisher.py#L272).

None of §0.1-0.3 touches per-mode math (K-FAC/EKFAC/TKFAC/TEKFAC formulas in `plan.md` §4 are
unaffected); they only fix how the shared ABC talks to the optimizer loop.

---

## 1. Scope decision: `factors.py` is diagonal-only in lot 1

`plan.md` §2.1 describes `factors.py` as producing "FULL factors (A, B)". For lot 1, where `diag` is
the only consumer, building full `A ∈ R^{d_in×d_in}`, `B ∈ R^{d_out×d_out}` and then slicing their
diagonal would be:

- **wasteful** — `diag` mode exists specifically to be the `O(d)` reference mode (`plan.md` §3.2 table);
  computing `O(d²)` factors just to discard the off-diagonal defeats that;
- **not bit-exact** — `diag(A) via einsum('ij,ij->j', X, X)` (the original's method) and
  `torch.diagonal(X.T @ X)` are the same real number but not necessarily the same floating-point
  value, since matmul-based and einsum-based reductions can accumulate in different order. Exit
  criterion 1 requires `torch.equal`, not `torch.allclose`.

**Decision:** `factors.py` in lot 1 contains only a renamed, line-for-line port of `_ComputeHBarD`
and `_ComputeSD` ([adafisher.py:52-155](reference_repos/FisherAdapTune/scripts/adafisher.py#L52-L155)),
covering `Linear`, `Conv2d`, `BatchNorm2d`, `LayerNorm`:

```python
def compute_h_diag(h: Tensor, layer: Module) -> Tensor: ...   # dispatches to _linear/_conv2d/_batchnorm2d/_layernorm
def compute_s_diag(s: Tensor, layer: Module) -> Tensor: ...
```

`_extract_patches` ([adafisher.py:33-50](reference_repos/FisherAdapTune/scripts/adafisher.py#L33-L50))
is ported verbatim alongside them. When lot 2 needs full `Linear`/`Conv2d` factors, the natural
extension point is a sibling `compute_h_full` / `compute_s_full` pair reusing `_extract_patches` — not
a rewrite of this module. Recorded here so the lot-2 change isn't a surprise refactor.

---

## 2. File-by-file design

```
adafisher /
├── pyproject.toml                        # new — see §5
├── src/adafisher_modes/
│   ├── __init__.py                       # exports AdaFisherMulti
│   ├── factors.py                        # §1 — diagonal extraction, ported from adafisher.py:33-155
│   ├── minmax.py                         # §2.3 — MinMaxNormalization, ported from AdaFisher.py:13-45
│   ├── optimizer.py                      # §2.4 — AdaFisherMulti(Optimizer)
│   └── approximations/
│       ├── __init__.py                   # MODES = {"diag": DiagApproximation}
│       ├── base.py                       # FisherApproximation ABC, §0-corrected
│       └── diag.py                       # §2.5 — DiagApproximation
├── tests/
│   ├── conftest.py                       # shared tiny-model fixture, seeding helpers
│   ├── test_minmax_matches_official.py
│   ├── test_diag_bitexact.py
│   └── test_diag_eq4_semantics.py
└── benchmarks/
    └── mnist_autoencoder.py              # §4
```

`config.py` is **not** part of lot 1 — `plan.md` §8's lot-1 row does not list it, and nothing in this
lot's exit criteria consumes YAML. It will be introduced when a benchmark first needs a config-driven
CLI (likely lot 7). Building it now would be exactly the kind of unused abstraction the project's
conventions ask to avoid.

### 2.1 `approximations/base.py`

```python
class FisherApproximation(ABC):
    """Builds AdaFisher's second moment v^(t) for one module, from raw per-hook factors."""

    @abstractmethod
    def update_input_factor(self, module: Module, h: Tensor, step: int) -> None: ...

    @abstractmethod
    def update_output_factor(self, module: Module, s: Tensor, step: int) -> None: ...

    @abstractmethod
    def refresh(self, module: Module, step: int) -> None:
        """Amortised re-estimation (inverses, eigenbases). No-op for diag."""

    @abstractmethod
    def precondition(
        self, module: Module, weight_direction: Tensor, bias_direction: Optional[Tensor]
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]: ...
```

No `register()` method: per-module state is lazily initialised inside `update_input_factor` /
`update_output_factor` on `step == 0`, exactly mirroring the original's
`if self.steps == 0: self.H_bar_D[module] = H_i.new(...).fill_(1)` pattern
([adafisher.py:191-192](reference_repos/FisherAdapTune/scripts/adafisher.py#L191-L192)). Adding a
separate registration hook would duplicate that logic for no lot-1 benefit.

### 2.2 `approximations/diag.py` — `DiagApproximation`

```python
class DiagApproximation(FisherApproximation):
    def __init__(self, Lambda: float = 1e-3, gammas: Sequence[float] = (0.92, 0.008),
                 minmax_normalization: bool = True, epsilon: float = 1e-6):
        self.Lambda, self.gammas = Lambda, gammas
        self.minmax_normalization, self.epsilon = minmax_normalization, epsilon
        self._H: Dict[Module, Tensor] = {}
        self._S: Dict[Module, Tensor] = {}

    def update_input_factor(self, module, h, step):
        H_i = compute_h_diag(h, module)                              # factors.py
        if self.minmax_normalization:
            H_i = min_max_normalization(H_i, self.epsilon)           # minmax.py
        if step == 0:
            self._H[module] = H_i.new_ones(H_i.size(0))
        _update_running_avg(H_i, self._H[module], self.gammas)       # ported EMA, adafisher.py:28-30

    def update_output_factor(self, module, s, step):
        ...  # symmetric, self._S

    def refresh(self, module, step):
        pass  # nothing to amortise

    def f_tilde(self, module) -> Tensor:
        """Raw (unsplit) F̃_D = kron(H, S) + λ. Exposed as a debug/test hook — also the
        quantity the Frobenius-dominance tests of lot 2/3 will need to compare against F."""
        return kron(self._H[module].unsqueeze(1), self._S[module].unsqueeze(0)).t() + self.Lambda

    def precondition(self, module, weight_direction, bias_direction):
        F = self.f_tilde(module)
        if module.bias is not None:
            Fw, Fb = F[:, :-1].reshape(weight_direction.shape), F[:, -1:].reshape(bias_direction.shape)
            return weight_direction / Fw, bias_direction / Fb
        return weight_direction / F.reshape(weight_direction.shape)
```

`f_tilde()` is the one addition beyond strict lot-1 necessity. It is cheap (a debug accessor, not new
control flow), it is exactly the `_get_F_tilde` computation the bit-exactness test already needs to
call out, and lot 2/3's Frobenius-dominance tests (`plan.md` §6.1) will need every mode — `diag`
included — to expose "the combined `F̃` for this module" in one comparable form. Adding it now avoids
re-deriving the same accessor under time pressure in lot 2.

`_update_running_avg` is ported verbatim from
[adafisher.py:28-30](reference_repos/FisherAdapTune/scripts/adafisher.py#L28-L30) (the two-parameter
`gammas` formula — **not** the official repository's single-scalar, buggy `gamma`, per `plan.md` §1.4).

### 2.3 `minmax.py`

Verbatim port of `_smart_detect_inf` ([adafisher.py:21-26](reference_repos/FisherAdapTune/scripts/adafisher.py#L21-L26),
already unused dead code at that location) and of the official repository's `MinMaxNormalization`
(`reference_repos/AdaFisher/optimizers/AdaFisher.py:13-45`):

```python
def smart_detect_inf(tensor: Tensor) -> Tensor:
    result = tensor.clone()
    result[tensor == inf] = 1.0
    result[tensor == -inf] = 0.0
    return result

def min_max_normalization(tensor: Tensor, epsilon: float = 1e-6) -> Tensor:
    tensor = smart_detect_inf(tensor)                 # already a clone — safe to mutate in place below
    min_t, max_t = tensor.min(), tensor.max()
    return tensor.add_(-min_t).div_(max_t - min_t + epsilon)
```

Applied to the **instantaneous** per-step factor, **before** the EMA — the placement used by the
official repository at
[`AdaFisher.py:412`](reference_repos/AdaFisher/optimizers/AdaFisher.py) /
[`:431`](reference_repos/AdaFisher/optimizers/AdaFisher.py)
(`update_running_avg(MinMaxNormalization(H_D_i), self.H_D[module], self.gamma)`), not to the
accumulated running average.

### 2.4 `optimizer.py` — `AdaFisherMulti`

Ports `AdaFisherBackbone` + `AdaFisher` ([adafisher.py:157-307](reference_repos/FisherAdapTune/scripts/adafisher.py#L157-L307))
almost verbatim: hook registration ([`_prepare_model`](reference_repos/FisherAdapTune/scripts/adafisher.py#L202-L207)),
`_check_dim` ([lines 225-230](reference_repos/FisherAdapTune/scripts/adafisher.py#L225-L230)), and the
index-bookkeeping `step()` loop ([lines 275-307](reference_repos/FisherAdapTune/scripts/adafisher.py#L275-L307),
already flagged in `plan.md` §2.1 as brittle and *not* to be rewritten) are kept structurally
unchanged. Three deltas, all consequences of §0:

- `_save_input`/`_save_grad_output` call `self.approx.update_input_factor(module, input[0].data, self.steps)`
  / `update_output_factor(...)` instead of computing `_compute_H`/EMA inline.
- The per-parameter `_step` (elementwise `addcdiv_`) is replaced by a per-**module** step that groups
  weight (and bias) together, calls `self.approx.precondition(module, weight_exp_avg, bias_exp_avg)`
  once, and applies `param.add_(direction, alpha=-lr/bias_correction)` to each. `state["step"]` is
  tracked once per module rather than once per parameter — safe, since weight and bias of the same
  module are always updated in the same call and therefore always share the same step count; this is
  a simplification of bookkeeping, not a behavioural change.
- The `F_tilde = ones_like(...)` fallback for a `_check_dim`-mismatched parameter
  ([line 282](reference_repos/FisherAdapTune/scripts/adafisher.py#L282)) is preserved at the optimizer
  level (plain SGD-with-momentum step for that parameter), independent of `fisher_mode` — no
  `FisherApproximation` needs to know about it.

```python
class AdaFisherMulti(Optimizer):
    SUPPORTED_MODULES = ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm")

    def __init__(self, model, lr=1e-3, beta=0.9, Lambda=1e-3, gammas=(0.92, 0.008), TCov=100,
                 weight_decay=0.0, dist_training=False,
                 fisher_mode: str = "diag", minmax_normalization: bool = True, **mode_kwargs):
        ...
        self.approx = MODES[fisher_mode](Lambda=Lambda, gammas=gammas,
                                          minmax_normalization=minmax_normalization, **mode_kwargs)
```

`minmax_normalization` is accepted by `DiagApproximation` only; passing it while `fisher_mode != "diag"`
will raise `TypeError` from the constructor call, which is an acceptable, honest failure mode for lot 1
(only `diag` exists) — revisit once lot 2 adds modes for which this argument is meaningless, per
`CLAUDE.md`'s "diag mode only" note.

### 2.5 `approximations/__init__.py`

```python
MODES: Dict[str, Type[FisherApproximation]] = {"diag": DiagApproximation}
```

One entry for lot 1; lots 2/3 add `"kfac"`, `"ekfac"`, `"tkfac"`, `"tekfac"` here without touching
`optimizer.py`.

---

## 3. Tests

### 3.1 `test_minmax_matches_official.py`

`min_max_normalization` vs. `reference_repos.AdaFisher.optimizers.AdaFisher.MinMaxNormalization`
(imported directly — reading from `reference_repos/` for comparison is exactly its stated purpose;
nothing there is modified). This isolates Eq. (4)'s normalisation semantics from the official
repository's buggy EMA (`plan.md` §1.4), so the oracle here is trustworthy. Cases: several random
tensors with a fixed seed; one tensor containing `+inf`/`-inf` entries to exercise `smart_detect_inf`;
one constant tensor (`max == min`, exercises the `epsilon` guard). Assertion: `torch.equal`.

### 3.2 `test_diag_bitexact.py`

A tiny synthetic net covering all four supported layer types (e.g. `Conv2d(2,3,3) → BatchNorm2d(3) →
Flatten → Linear(27,8) → LayerNorm(8) → Linear(8,4)`), fixed seed, random batch of 6. Run `N=3·TCov`
steps (`TCov=2`) through **both**:

- `reference_repos.FisherAdapTune.scripts.adafisher.AdaFisher` (imported directly), and
- `AdaFisherMulti(fisher_mode="diag", minmax_normalization=False)`.

Assertions:

- **`torch.equal(approx.f_tilde(module), original._get_F_tilde(module)` reassembled into one
  tensor)`** at every step where the factors update — this is exit criterion 1, stated precisely.
- `torch.allclose(param_new, param_original, rtol=1e-6, atol=1e-8)` for every parameter after every
  step — a full-trajectory check, deliberately **not** `torch.equal`. Rationale: §0.3 keeps the op
  order as close as possible to `addcdiv_`, but whether PyTorch's fused `addcdiv_` kernel is bit-
  identical to the unfused `div` + `add_(alpha=...)` used here is an empirical question, not a
  guarantee from the docs. If it turns out to be bit-identical in practice, tighten this assertion
  to `torch.equal` once observed — do not assume it upfront.

### 3.3 `test_diag_eq4_semantics.py`

Composes the two validated pieces instead of trusting the official repository's buggy optimizer as an
oracle: at `step == 0` on the same tiny net, check that
`DiagApproximation(minmax_normalization=True)`'s internal `_H[module]` after one update equals
`_update_running_avg(min_max_normalization(H_raw), ones_like(H_raw), gammas)`, where `H_raw` is the
**exact** instantaneous diagonal already validated bit-exact against FisherAdapTune in §3.2 (obtained
here via `compute_h_diag` directly, no optimizer needed). This validates exit criterion 2 without
depending on the official repository's EMA at all.

---

## 4. Benchmark: `benchmarks/mnist_autoencoder.py`

The 8-layer auto-encoder of `ekfac_1806.03884.pdf` §4.1 (already specified in `plan.md` §6.3):
encoder `784-1000-500-250-30`, sigmoid activations, symmetric decoder with **untied** weights (8
`Linear` layers total, no biasless layers). `torchvision.datasets.MNIST` — confirmed reachable from
this environment (`ossci-datasets.s3.amazonaws.com` and the `storage.googleapis.com` mirror both
resolve). Lot-1 scope: plain `argparse` CLI (no YAML — see §2), trains with
`reference_repos.FisherAdapTune.scripts.adafisher.AdaFisher` and with
`AdaFisherMulti(fisher_mode="diag")` (both `minmax_normalization` settings selectable), logs
per-epoch reconstruction loss (MSE) to stdout/CSV. Success is qualitative for lot 1: with
`minmax_normalization=False` the two curves should be visually superimposed (a direct consequence of
§3.2's bit-exactness, at the trajectory-tolerance already established there); with `True`, the curve
should differ from `False` but train stably. No comparison to K-FAC/EKFAC/TKFAC/TEKFAC yet — that is
lot 7's equal-wall-clock comparison (`plan.md` §6.3, §8).

---

## 5. Environment and tooling

- Runtime: conda env `fisheradaptune` (`~/anaconda3/envs/fisheradaptune`) already has
  `torch 2.12.1`, `torchvision 0.27.1`, `numpy 2.4.6`. It is **missing `pytest`** — proposal: `pip
  install pytest` into that same env (additive, reversible, does not touch anything the FisherAdapTune
  project itself depends on).
- New root-level `pyproject.toml`, `src` layout, package name `adafisher-modes`
  (`src/adafisher_modes`), `requires-python = ">=3.10"` (matches the conda env; the plan's target
  audience is this repository, not a wider distribution), dependencies `torch>=2.0`, dev extra
  `pytest`. Styled after `reference_repos/FisherAdapTune/pyproject.toml` (`[tool.ruff]`,
  `[tool.mypy]` sections reused with the same conventions) but with its own `[project]` metadata —
  not copied wholesale, since license/author fields must not misattribute this new package to
  FisherAdapTune's authors.

---

## 6. Explicitly out of scope for lot 1

- Full (non-diagonal) factor computation, `kfac.py`, `ekfac.py`, `tkfac.py`, `tekfac.py` — lots 2-3.
- `Conv2d`/normalisation full-factor handling beyond the diagonal path already in AdaFisher — lots 4-5.
- `config.py` / YAML-driven runs — introduced when a benchmark first needs it.
- Equal-wall-clock, multi-mode comparison — lot 7.
- CIFAR-10 / ResNet / ViT / Adam baseline — deferred on request (`plan.md` §9, row 4).

---

## 7. Points requiring explicit go-ahead

1. §0.1-0.3 — the three ABC signature corrections. They do not change lot 1's behaviour or exit
   criteria, but they are the interface lot 2 will build on, so worth confirming before code exists
   that depends on them.
2. §5 — `pip install pytest` into the existing `fisheradaptune` conda env.
3. §2.2 — adding `DiagApproximation.f_tilde()` now, ahead of strict lot-1 necessity, because lot 2/3's
   already-approved dominance tests will need every mode to expose it.
