# E20 — pre-registered: does one safety constant per layer survive a network nobody tuned it on?

*A pre-registration, written on 22 September 2026, before any run it describes. Plain language
throughout. It extends `plan_lambda_dominance.md`, whose words it reuses: the **safety constant** `λ`
is the number added to every curvature value before the optimizer divides by it, **S1-b** is E15's
per-layer version of it, and the **step-size cap** is the largest multiple of the momentum any
direction can move by. Anything below that changes after the first E20 run is a deviation, and will
be recorded as one, in a dated section at the end of this file.*

---

## 0. Why this exists, and what it is not

**The question.** E15 found that giving each layer its own constant, `λ_l = τ × (mean stored
curvature of layer l)`, beats the best single `λ` on the two networks it ran (`cnn_gn_cifar`,
`vit_micro_cifar`), in all six (network, mode) pairs. For `ekfac` and `tekfac`, `τ = 0.1` lay inside
every plateau. But `τ = 0.1` was chosen on those two networks. Does it still work, unchanged, on a
network it was never tuned on? And does the per-layer form still beat a single `λ` there?

**The network is ViT-S** (`vit_small_cifar`, 2 693 578 parameters). It is where the shipped Fisher
arms lose to AdamW, in campaign 2 and on all three datasets. It is a transformer 128 times larger than
`vit_micro_cifar`, built from the same code with the same hyperparameters. And no experiment has yet
run it with any `λ` or `τ` other than the shipped ones.

**How this relates to E17.** E17 (in `plan_lambda_dominance.md`) is the held-out test of E15's and
E16's verdicts on ViT-S. Its amendment of 21 September, by the user's decision, reads its candidate
C's entry condition literally. That excludes S1-b from E17's vote and allows it only as an
exploratory arm. **E20 does not change that amendment.** It is a separate, explicit test of S1-b on
ViT-S, decided on 22 September by the user, after E15's and E16's results were known, and before any
ViT-S run of S1-b. What was known when it was written is stated here so that nobody has to infer it:

- E15's results, including that `τ = 0.1` is in the plateau of all four `ekfac`/`tekfac` pairs.
- E16's results, and so the fact that S1-b is the only candidate E17 would otherwise have tested
  besides the single `λ`.
- Nothing about ViT-S at any setting other than the shipped one.

**The one ordering constraint.** E17 forbids any non-shipped ViT-S run before its own transferred
values are fixed, so that they cannot be influenced. Those values are already determined by E16's
decision script (no clip, no floor, the added `λ` at 5.4772e-11) and by the amendment (C
exploratory). But they are fixed only once E17's dated addendum is committed. **No E20 job except the
calibration may be submitted before that addendum is committed.** The calibration runs the shipped
setting only, which E17 allows.

---

## 1. What is transferred, fixed now

Two values, both chosen by E15's own rule 1 (best five-seed mean of the final validation accuracy)
on `cnn_gn_cifar` and `vit_micro_cifar`, and nothing chosen on ViT-S:

| rule | value sent to ViT-S | where it comes from |
|---|---|---|
| S1-b | **`τ = 0.1`** | the only value inside all four `ekfac`/`tekfac` plateaus of E15 (E15, Result 1) |
| single `λ` | **`λ = 1e-10`** | the geometric mean of E15's four validation-selected single values (1e-10, 3e-11, 1e-10, 1e-10), which is 7.40e-11, rounded to the nearest point of E15's half-decade grid |

The single `λ` is transferred from E15 and not from E16 (whose value is 5.4772e-11), so that the two
rules compared here come from the same experiment, the same protocol and the same selection.

---

## 2. Stage 1: the E-series protocol on ViT-S

Everything E15 fixed, so that `τ` means on ViT-S what it meant where it was chosen: batch 32,
15 epochs, the clamped cosine, the shipped estimator, `eig_before_rescale=True`, 4 data-loader
workers, seeds 0-4 sharing initialisation and data order across every cell of a seed. The Fisher
cells are built by **E15's own `run_cell`**, called unmodified, with E15's held configuration:

- the step-size cap held per layer (`hold_cap=True`, `lr = cap = base_lr / base_λ = 1/3`);
- the parameters no hooked module owns frozen: `cls_token` and `pos_embed`, 12 672 of 2 693 578;
- the decoupled weight decay at the benchmark's own rate, `base_lr × wd = 1e-5` per step at the top
  of the cosine, whatever `λ` is (rule 3 of `plan_lambda_dominance.md`, Part 5).

**Cells, per seed.**

| arm | what varies | values | runs per mode |
|---|---|---|---|
| reference | nothing: the shipped `λ = 3e-3` | — | 1 |
| single `λ` | one `λ` for the network | {1e-9, 3e-10, **1e-10**, 3e-11, 1e-11}, E15's `vit_micro_cifar` grid | 5 |
| S1-b | `λ_l = τ · c̄_l` per layer | {1, 0.3, **0.1**, 0.03, 0.01, 0.003, 0.001, 3e-4}, E15's grid | 8 |
| AdamW | its learning rate | {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}, once per seed | — (6 per seed) |

The transferred values, in bold, are cells of the grids. The rest of each grid is the **oracle**: it
measures how far the transferred value sits from ViT-S's own best, and it cannot change what was
transferred. Modes: `ekfac` and `tekfac`. `kfac` is not run: E15's `τ` for it was different, and it
is not the mode anyone would ship.

**AdamW is tuned on ViT-S, on purpose.** Two E16 sessions report that at batch 32 AdamW's best
learning rate on `vit_micro_cifar` sits at the top of a `baseline_lr × {1/3 … 30}` grid (58.0 %,
still rising). The benchmark's `baseline_lr = 1e-4` is set for batch 128. A baseline stopped too low
would flatter every Fisher arm, so AdamW gets a grid three times wider at the top, and it is *tuned
on ViT-S itself*, while S1-b is not. That is conservative against S1-b. AdamW is configured as the
benchmark runs it, and as E16's baselines ran it: nothing frozen, decoupled weight decay at
`wd = 0.01` applied by AdamW itself.

**A bridge, at seed 0.** Before anything on ViT-S, each seed-0 job re-runs one E15 cell on
`vit_micro_cifar`: S1-b at `τ = 0.1`, in its own mode. It must reproduce E15's stored test accuracy
(60.55 % for `ekfac`, 60.09 % for `tekfac`) and its distance travelled, bit for bit. The code has
changed since E15 ran (E16's options, E18's arm). A mismatch stops the job, and no E20 result is read
until it is explained.

**Cost.** Not measured: a 15-epoch ViT-S run at batch 32 has never been timed. E17 estimates 6 to 10
minutes from campaign 2's batch-128 time. A calibration job runs one shipped reference cell first and
sets the time limits. At 8 minutes a run: 14 runs per (seed, mode), about 1 h 50; 6 AdamW runs per
seed, about 50 minutes; 15 jobs, about 23 GPU-hours.

### 2.1 Stage-1 rules, fixed now

All comparisons are paired by seed on **test** accuracy. "Win", "tie" and "loss" use E15 rule 2's
criterion: the mean paired difference is above 2 standard errors, within them, or below −2. Any value
*chosen* on ViT-S is chosen on the five-seed mean of the final **validation** accuracy (E15 rule 1).

Per mode (`ekfac`, `tekfac`):

1. **Does it help?** S1-b at `τ = 0.1` against the reference arm.
2. **Does `τ` transfer?** The plateau is E13's: every value on ViT-S's S1-b grid whose five-seed mean
   test accuracy lies within one standard error of the best. `τ = 0.1` **transfers** if it lies inside
   it. If the best value sits at an edge of the grid, the grid is extended by two values on that side,
   once. If it is still at the edge, the verdict is **unresolved**.
3. **Does per-layer still beat a single `λ`, with nothing tuned on ViT-S?** S1-b at `τ = 0.1` against
   the single `λ` at `1e-10`. **This is E20's primary claim.**
4. **The same, with both tuned on ViT-S.** S1-b and the single `λ`, each at its validation-selected
   value. This is E15 rule 2, replicated on a third network.
5. **Against AdamW.** S1-b at `τ = 0.1` (transferred) against AdamW at its validation-selected
   learning rate (tuned on ViT-S). Reported with that asymmetry stated. If AdamW's selected learning
   rate is at an edge of its grid, the grid is extended by two values on that side, once.

**Verdicts.**

- **Per-layer damping confirmed on a held-out network** if rule 3 is a win for both modes. **Refuted**
  if it is a tie or a loss for both. Otherwise **mixed**.
- **`τ = 0.1` transfers to ViT-S** if rule 2 says "transfers" for both modes. It **fails to transfer**
  if it lies outside the plateau for both. Otherwise **mixed**.
- Rules 1, 4 and 5 are reported per mode. Rule 5 is the one that says whether ViT-S's loss to AdamW
  survives at a working operating point.

**Written down so it can fail.** From E15, where S1-b beat the single `λ` by 6.0 points and the
default by 14.5 to 14.8 on `vit_micro_cifar`: rule 1 helps and rule 3 wins, in both modes. And, from
E15's per-layer logs, S1-b's constant on the classification head sits hundreds of times above the
single `λ`, as on `vit_micro_cifar`. These are expectations, not rules. They are checked, not voted
on.

---

## 3. Stage 2: the campaign protocol, conditional

Run only if stage 1's "`τ = 0.1` transfers" is not "fails to transfer". It uses E17's stage 2
unchanged: campaign 2's protocol and flags (batch 128, 50 nominal epochs, the wall-clock budget with
`diag` as reference, the budget cosine, `--max-epoch-factor 3`), both datasets (`vit_small_cifar`,
`vit_small_cifar100`), seeds 0-2, each (dataset, seed) one job with its own `diag`, `adamw`, shipped
`ekfac`/`tekfac`, and S1-b. `τ` is free of scale and is used unchanged at batch 128. The cap, the
frozen parameters and the decay rate are as in stage 1. E17's stage-2 rules apply: S1-b **closes the
gap** if its final test accuracy is at least AdamW's on all three seeds, **narrows** it if its gap is
smaller than the shipped arm's on all three seeds, and otherwise **does not close** it.

Stage 2 needs harness code that does not exist yet: `HParams` fields carrying `damping`,
`damping_tau`, `hold_cap` and the frozen parameters into `benchmarks/`, inert and bit-identical when
unset. It is written only if stage 2 is triggered, and pre-registered before it runs.

---

## 4. What E20 cannot settle

- One more network. "Transfers to ViT-S" means one more network, not every network.
- The AdamW comparison is at the E protocol (batch 32, 15 epochs). The campaign's own protocol is
  stage 2's business.
- `kfac` is not tested here.

---

## 5. Deviations and amendments

### Amendment 1, 23 September 2026 — a network-wide relative arm, and a second reading of `τ`

**When this was written.** Stage 1's fifteen jobs (21645885-99) had finished on the cluster. **No
E20 output had been opened**, by this session or any other: this text and the new jobs are committed
before the first read, and that order is the only evidence available. What triggered the amendment
is E19's results, measured on three other networks (`plan_lambda_dominance.md`, "E19 — done"), not
anything from ViT-S.

**What E19 found that bears on E20.** Three things.
1. `τ = 0.1` is no longer a value that transfers. The `τ` E19 selects is 0.3 on CCT and ResNet-20 and
   0.1 on ResNet-50, and over the ten `ekfac`/`tekfac` pairs now measured no single `τ` lies inside
   every plateau.
2. S1-b loses to a single `λ` tuned per network on CCT and ResNet-50, and gains a little on
   ResNet-20. So E15's "S1-b wins everywhere" does not survive three new networks.
3. On CCT — the only transformer among them — the **network-wide** relative `λ` beats the per-layer
   one by about one point and ties the tuned single `λ`. E15's rule 5 said the opposite on its two
   networks.

**What changes.**
- **A new arm, `netadapt`**: one `λ(t) = τ × (mean curvature of the whole network)`
  (`damping="network_relative"`), on E15's eight `τ`, both modes, five seeds, with S1-b's settings
  otherwise. It is run as ten new jobs; stage 1's fifteen files are not touched.
- **A second reading of S1-b, at `τ = 0.3`**, reported beside the primary one.

**What does not change.** The transferred value stays `τ = 0.1`, and rule 3 at that value stays
E20's primary claim. Rules 1 to 5 are unchanged, and so are their verdicts. Stage 1's existing cells
are not re-run.

**The new rules, fixed now.** Same conventions as section 2.1: paired by seed on test accuracy,
win/tie/loss at 2 standard errors, any value chosen on ViT-S chosen on the five-seed mean of the
final validation accuracy.

6. **Network-wide against per-layer.** `netadapt` at its selected `τ` against S1-b at its selected
   `τ`. This is E19's rule 5 on ViT-S. It says which of the two relative rules is the better one
   here.
7. **Network-wide against a tuned single `λ`.** `netadapt` at its selected `τ` against the single
   `λ` at its selected value. Both are tuned on ViT-S, so this asks whether a relative rule is worth
   anything once the constant is tuned.
8. **Reported, not voted on:** S1-b at `τ = 0.3` against the single `λ` at the transferred 1e-10,
   paired. `τ = 0.3` is what E19 selected on its two smaller networks. It is a second reading of the
   same question rule 3 asks, at the value a reader of E19 would have transferred instead. Also
   reported: `netadapt` at `τ = 0.03`, the value E19 selected on CCT, against the same single `λ`.

**Cost.** Ten jobs, 8 cells each, about 70 minutes per job at the 8.73 minutes per run the
calibration measured, so about 12 GPU-hours.
