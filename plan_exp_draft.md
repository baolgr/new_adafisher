# Experimental plan — Fisher reference bench: how far the approximations drift, by layer type

*Simplified rewrite of the long draft, which is preserved verbatim at
`docs/reports/plan_exp_draft_v0.md`. Every parameter count, memory budget and derivation quoted
below comes from it; nothing new is asserted here. Status tags are kept from that draft:
`[ESTABLISHED]` = checked against the primary source, `[DERIVED]` = arithmetic done from published
numbers or known architectures, `[ESTIMATE]` = order of magnitude, unmeasured.*

---

## 1. The question

At a fixed `theta` and on a fixed, versioned probe set `D_N = {x_n}`, build **two exact references**:

- **true Fisher** `F = (1/N) sum_n J_n^T Lambda_n J_n`, with `J_n = dz_n/dtheta` and `Lambda_n`
  computed **analytically**, never sampled. Softmax: `Lambda_n = diag(p_n) - p_n p_n^T`, with the
  closed-form root `S_n[:,c] = sqrt(p_c) (e_c - p_n)`
  (`docs/papers/kfac_from_scratch_2507.05127.pdf`, cheat sheet §6) `[ESTABLISHED]`.
- **exact empirical Fisher** `E_hat = (1/N) sum_n grad(l_n) grad(l_n)^T`, true labels, no structure
  imposed.

Then measure `d(K, R)` for every approximation `K` in the zoo and both references
`R in {F, E_hat}`, **per layer type**, and split the drift into its two independent causes:

| Cell | Isolates | Question |
|---|---|---|
| `d(E_hat, F)` | the **source** alone (no structure) | Q1 |
| `d(S(type-2), F)` and `d(S(emp), E_hat)` | the **structure** alone | Q2 |
| `d(S(emp), F)` | AdaFisher's actual path — both errors at once | Q3 |
| the above, per layer type | Linear, weight-shared Linear, Conv, normalisation, head | Q4 |
| `d(state of the real optimizer, F)` | EMA, min-max, damping, BN in train mode | Q5 |

For a genuine distance the triangle inequality bounds Q3 by Q1 + Q2; the gap between the two says
whether the errors **cancel** or **align**. That decomposition is the point of the bench.

---

## 2. What the repository already provides, and what is missing

**Already there — reuse, do not rebuild:**

| Piece | Where |
|---|---|
| Hooks + per-layer factor extraction (`Linear`, `Conv2d`, `BatchNorm2d`, `LayerNorm`, incl. SUA) | `src/adafisher_modes/factors.py` |
| The five rescalings `diag / kfac / ekfac / tkfac / tekfac`, each with a dense debug `f_tilde()` | `src/adafisher_modes/approximations/` |
| Toy **exact** Fisher oracles: per-sample-exact block, `unfold`-based conv oracle, per-position norm oracle | `tests/test_frobenius_dominance.py` |
| Frobenius / trace / eigenbasis metrics on those toy blocks | same file |
| Wall-clock-budget training harness, CIFAR-10 pipeline, SLURM jobs | `benchmarks/` |

**Missing, and therefore the actual work:**

1. exact `F` and `E_hat` at *model* scale, not toy scale;
2. scale-invariant metrics (the repo only measures Frobenius on toy blocks);
3. the **type-2 source** — the repo is empirical-only by design (`CLAUDE.md`, "Empirical Fisher, not
   the Fisher"), so type-2 backpropagation is new code, kept strictly outside `adafisher_modes`;
4. a model set covering every layer type at a size where an exact reference is affordable;
5. checkpointed trajectories, to evaluate at `t in {0, 1%, 10%, 50%, 100%}`.

**Rule.** Any structure that already exists in `adafisher_modes` is called, not reimplemented. The
new code builds *references*, *sources* and *metrics* only.

---

## 3. What this plan drops from the long draft, and why

| Dropped | Reason |
|---|---|
| Regime C (matrix-free operators only) | buys one model (GPT-1, 4 blocks) at the price of stochastic metrics and no type-2 K-FAC; deferred to step 6 |
| 12 models -> 5 core + the 3 already in the repo | the draft's own §0 says "the core is A1-A4 + B1-B4"; four of those already cover every layer type, the rest is coverage of *papers*, not of *layer types* |
| 8 metrics -> 4 (+1 optional) | M2/M4/M6 need Lanczos machinery to answer what M1/M5 already answer |
| 3 sources -> 2 (type-2, empirical) | Monte-Carlo `K in {1,4,16}` only tests HF6, a side question; kept as an option |
| 8 falsifiable hypotheses -> 4 | HF4 is largely settled analytically in this repo already (`plan_lot5.md` §0.1: a normalisation layer's FIM is Hadamard-, not Kronecker-structured); HF5 needs inter-layer blocks, i.e. regime A only; HF8 becomes a *reporting rule*, not a hypothesis |
| A 30-file `fisher_ref/` tree | 6 modules suffice once `adafisher_modes` is reused |
| Per-metric budget tables, hardware sizing, risk register | moved to the step that needs them; `plan_exp_draft_v0.md` keeps the arithmetic |

---

## 4. Models

`P` = parameter count `[DERIVED]`. Regimes as in `plan_exp_draft_v0.md` §2: **A** = `F` formed
densely in fp64 (`P <= 2.9e4` on a 20 GB MIG slice), **B** = `F` kept as per-layer factors / Grams,
exact but of rank `<= m = N(C-1)`.

| Id | Model | `P` | Regime | Layer types covered | Status |
|---|---|---|---|---|---|
| A1 | MLP 784-32-32-10 + LayerNorm (MNIST) | 26 634 | A | Linear (no sharing), LN | new |
| A2 | CNN 3x conv (16-32-64) + GroupNorm / BN-eval (CIFAR-10) | 24 458 | A | Conv (spatial sharing), normalisation | new |
| A3 | ViT-micro, `d=32`, 2 blocks, mean pooling (CIFAR-10) | 21 162 | A | patch-embed, QKV, attention out, MLP, LN, head | new |
| B1 | **CCT-2/3x2** (CIFAR-10) | 283 723 | B, `N ~ 900` | conv tokenizer, transformer, LN, sequence pooling | new — AdaFisher's own model (Table 2) |
| B2 | ResNet-20 (CIFAR-10) | 269 722 | B, `N ~ 900` | Conv, BN | new |
| — | MNIST auto-encoder `784-1000-500-250-30` | 2 837 314 | — | Linear only | **exists** (lots 1, 7) |
| — | ResNet-50, CIFAR stem | 23 519 178 | — | Conv, BN, Linear | **exists** (lot 8) |
| — | ViT-S/4 | 2 685 898 | — | Conv, LN, Linear (`qkv`) | **exists** (lot 8) |

The three existing models stay **training benches only**: their `P` puts them outside regimes A and
B, so they measure optimizer *behaviour* (lots 7-8's wall-clock protocol), never approximation
*fidelity*. The MNIST auto-encoder is MSE, so its true Fisher has `Lambda = I` and `m = 784 N` —
exact, but regime C; not used as a reference.

**Deferred to step 6, on request:** GPT-micro char-level (A4, the *expand* framework), RoBERTa-base
+ LoRA `r=8` (B3 — the PEFT regime, the one closest to the internship's own topic; deferred only
because it pulls in `transformers` + `peft`), ViT-Tiny/4 (B4), ResNet-18 with ghost Grams (B5).

---

## 5. Metrics

`R` = reference, `K` = approximation, `X_lam = X + lam I`.

| Id | Metric | What it measures |
|---|---|---|
| M1 | `e_F = ||R-K||_F/||R||_F`; `cos_F = <R,K>_F/(||R||_F ||K||_F)`; `e*_F = sqrt(1-cos_F^2)` | entry-wise fidelity, dominated by the top eigenvalues |
| M5 | `rho(K) = (g^T d)^2 / ((d^T R_lam d)(g^T R_lam^-1 g))`, `d = K_lam^-1 g` | fraction of the optimal quadratic decrease obtained along `d`; `rho in [0,1]`, the natural score for a **preconditioner** |
| M7 | `sigma_2/sigma_1` of the rearrangement of the exact block; `||diag B - diag A (x) diag G||/||diag B||`; `||B - B_exp||/||B||` | independence bias, weight-sharing bias, and the two split apart |
| M8 | `c_ll' = ||F_ll'||_F^2 / (||F_ll||_F ||F_l'l'||_F)` | inter-block coupling (an uncentered CKA between layer tangent kernels) |
| M3 *(optional, regime A)* | Stein KL `D_lam(K||R)` | affine-invariant gap between preconditioners |

Two rules apply to every metric:

- **Never a single damping.** Sweep `lam = alpha * tr(R)/P`, `alpha in {1e-4, 1e-3, 1e-2, 1e-1, 1}`,
  and plot the curve. For approximations that do not claim to respect scale (`diag` with min-max,
  Adam's `v`), also report the optimally-rescaled version `c* = <R,K>_F/||K||_F^2` — the convention
  the repo already uses for `diag` (`plan.md` §6.1).
- **Noise floor.** Split the probes in half, over 20 random partitions, and report the 95 % interval
  of `d(F^(1), F^(2))`. **A difference between two approximations smaller than that floor is not
  reported.** This replaces the draft's HF8 with a reporting rule.

---

## 6. Falsifiable hypotheses (written before any run)

- **HF1** — On *train* probes, `d(E_hat, F)` measured by `rho` degrades along training; on *val*
  probes, much less. *Refuted if* the two curves coincide within the noise floor.
- **HF2** — Structure error (type-2 source) is larger on weight-shared layers (QKV, conv) than on
  plain `Linear`, and K-FAC-**reduce** beats K-FAC-**expand** in classification (Eschenhagen et al.,
  arXiv:2311.00636, Props. 1-2). *Refuted if* the expand/reduce gap is under the floor.
- **HF3** — AdaFisher's independence assumption `E[a_j^2 g_i^2] ~ E[a_j^2] E[g_i^2]` has >= 10 %
  relative error on normalisation and attention layers. Note `diag(A) (x) diag(G) = diag(A (x) G)`:
  AdaFisher's raw estimator **is** the diagonal of K-FAC, so the two paths `B -> A(x)G -> diag` and
  `B -> diag(B)` differ by exactly `Cov(a_j^2, g_i^2)` entry-wise (absent weight sharing).
- **HF7** — The P1 ranking is not the P2 ranking: EMA and min-max, not structure, dominate
  AdaFisher's real error (Kendall rank correlation, per layer type).

---

## 7. Protocols and invariants

- **P1 — structural.** Fixed `theta`, fixed probes, no EMA, no min-max, `lam` swept. Answers "what
  is the best each family *can* do".
- **P2 — operational.** Run the real `AdaFisherMulti` (**unchanged**) and, at each checkpoint,
  extract the state as it enters the step — factors after EMA, min-max, `lam = 1e-3`, BN in train
  mode, dropout active, batch 256 — and compare it to the exact reference at the *same* `theta`.
  Min-max makes the preconditioner scale-free, so P2 uses the scale-invariant metrics only.
- **Checkpoints** `t in {0, 1%, 10%, 50%, 100%}` of the steps, on two trajectories: `adamw`
  (neutral) and `diag` (AdaFisher's own). 3 seeds.
- **Invariants** — changing any of these invalidates earlier comparisons: fp64 for every reference;
  TF32 off (**both** `torch.backends.cuda.matmul.allow_tf32` *and*
  `torch.backends.cudnn.allow_tf32`, the second defaults to `True`); references computed in **eval**
  mode (BN running statistics, no dropout) so samples are independent and `F` is defined at all; the
  `rvec` convention and `1/N` scaling everywhere; hashed, versioned probe sets; `reference_repos/`
  read-only; a `metrics_version` string in every result file.

---

## 8. Steps

| Step | Content | Exit criterion |
|---|---|---|
| **1** | **Benchmark unification + every model implemented** — `docs/reports/plan_exp_step1.md` | every model trains and dumps checkpoints; the 167 existing tests still pass |
| 2 | `src/fisher_ref/`: probes, type-2 source, dense `F`/`E_hat`, metrics M1/M5/M7/M8, noise floor | `d(R,R) = 0`; `Fv` matches `curvlinops.GGNLinearOperator` to `1e-10` in fp64; K-FAC-type-2 reproduces the repo's own toy oracles |
| 3 | Regime A map: A1-A3 x {type-2, emp} x the zoo x `lam` sweep x 5 checkpoints x 5 seeds | HF2 and HF3 decided on regime A, above the noise floor |
| 4 | Regime B: per-layer Grams (materialised or ghost), Woodbury identities; validated on A1-A3 against the dense answer, then run on B1, B2 | regime B agrees with regime A to `1e-8` on A1-A3 |
| 5 | P2 on A2, B1, B2 | HF7 decided |
| 6 | On request: LoRA/RoBERTa (B3), GPT-micro (A4), ViT-Tiny (B4), ResNet-18 ghost (B5), regime C | — |

Order rationale: regime A is the oracle that validates regime B; P2 needs the trajectories step 1
produces; the extensions only make sense once the core map is stable.

---

## 9. Software

```text
src/fisher_ref/            # new, small; structures come from adafisher_modes
  conventions.py           # dtype policy, TF32 off, rvec, 1/N, metrics_version
  probes.py                # probe sets (train/val), hashed and versioned, no augmentation
  sources.py               # backprop vectors: type-2 (closed-form root), empirical, MC_K (option)
  reference_dense.py       # regime A: F, E_hat, exact per-layer blocks, dense fp64
  reference_factor.py      # regime B: per-layer U_l / Gram K_l, Woodbury for M5
  metrics.py               # M1, M5, M7, M8, noise floor, lambda sweep
benchmarks/fisher_drift.py # the runner: (model, checkpoint, protocol) -> parquet rows
```

One protocol, so each metric is written once and works in both regimes:

```python
class CurvatureBlock(Protocol):
    P: int
    def matvec(self, v: Tensor) -> Tensor: ...              # K v
    def apply_rows(self, U: Tensor) -> Tensor: ...          # rows u_i -> K u_i
    def solve(self, v: Tensor, lam: float) -> Tensor: ...   # (K + lam I)^{-1} v
    def trace(self) -> Tensor: ...
    def fro2(self) -> Tensor: ...
    def diag(self) -> Tensor: ...
    def to_dense(self) -> Tensor: ...                       # regime A only
```

Implementations: `Dense`, `LowRank(U)`, `Kron(A, G)`, `EKFAC(Q_A, Q_G, s)`, `ScaledKron` (TKFAC),
`Diag`, `BlockDiag`, plus one adapter wrapping an `adafisher_modes` approximation, so the five
existing modes enter the zoo without being rewritten.

Output: one `parquet` per (model, checkpoint), columns `layer, layer_type, structure, source,
reference, lambda_alpha, metric, value, ci_low, ci_high, N, seed, protocol, metrics_version`.

---

## 10. Risks

- **Rank deficiency in regime B** — metrics that invert are dominated by `lam` in the kernel of `F`.
  Parry: sweep `alpha`, and use regime A (where `N(C-1) >= P`) as the judge.
- **Precision** — accumulate `U` in fp32 but Grams and `U K U^T` in fp64; TF32 off (the `Fv` oracle
  test detects it).
- **Tied weights / re-used modules** — a module called twice overwrites its hook; untie or
  accumulate explicitly. Relevant to A3/A4 and to any LM head tied to its embeddings.
- **BN train vs. eval** — the reference is eval; the gap to what AdaFisher actually sees is a P2
  measurement, not a bug.
- **`curvlinops` API drift** — pin the version, re-run the oracle test on every upgrade.

---

## References

The full list is in `docs/reports/plan_exp_draft_v0.md`. The ones this plan depends on: Martens
arXiv:1412.1193 §9 (`F = GGN` under a canonical link); Martens & Grosse arXiv:1503.05671; Grosse &
Martens arXiv:1602.01407; George et al. arXiv:1806.03884; Gao et al. arXiv:2011.10741; Eschenhagen
et al. arXiv:2311.00636 (expand/reduce); Dangel et al. arXiv:2507.05127 (*KFAC from scratch* — the
closed-form softmax root, the `rvec` pitfall); Dangel et al. arXiv:2501.19183 (`curvlinops`);
Dangel, Tatzel & Hennig arXiv:2106.02624 (ViViT, regime B's tooling); Kunstner, Balles & Hennig
arXiv:1905.12558 (empirical vs. true Fisher); Martins Gomes et al. arXiv:2405.16397 (AdaFisher —
Prop. 3.1, Eq. 4, App. A.3, App. B.2, Table 2).
