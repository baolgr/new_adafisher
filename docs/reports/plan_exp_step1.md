# Step 1 — one benchmark harness, one folder per model

*Implementation plan for step 1 of `plan_exp_draft.md` §8. In the repository's own numbering this is
lot 9; it is named after the experimental plan rather than the lot sequence because it serves the
Fisher-drift campaign, not the optimizer implementation.*

**Goal.** Every model of `plan_exp_draft.md` §4 trains from one shared, small, readable harness, in
its own folder, and can dump checkpoints. Nothing about the Fisher references, sources or metrics
(steps 2-5) is built here.

**Non-goal.** No change to `src/adafisher_modes/`. No new optimizer behaviour. No re-tuning of any
hyperparameter.

---

## 0. Where the current code stands, and what is wrong with it

| Fact today | Consequence |
|---|---|
| Three training loops: `mnist_autoencoder._train_one`, `equal_wallclock_bench.train_under_time_budget`, `cifar10_classification.train_classifier_under_budget` | the third is a strict superset of the second (adds `eval_fn`, `scheduler`, epoch records); the first is a plain epoch loop. Three places to fix any bug |
| Two optimizer factories: `mnist_autoencoder._make_optimizer` (7 names incl. `reference`), `cifar10_classification.make_optimizer` (7 arms incl. `adam`/`adamw`) | the five Fisher modes are configured twice, differently |
| `cifar10_classification` imports `StepRecord`, `_median`, `_sync` from `equal_wallclock_bench`, and both do `sys.path.insert(benchmarks)` | flat sibling modules importing each other's private names |
| Per-model hyperparameters live in `MODEL_DEFAULTS` inside the runner | adding a model means editing the runner |
| Flat `cifar10_*.py` namespace, one `cifar10_models.py` for all architectures | five more networks make that file unreadable |
| Output paths named after lots (`outputs/lot7_equal_wallclock`, `outputs/lot8_cifar10/<model>`) | results are indexed by *when* they were produced, not by *what* they measure |

None of this is wrong per lot; it is what incremental lots produce. Step 1 pays it off once.

---

## 1. Decisions

- **D1 — one folder per tested model.** A folder holds `model.py` **only if it introduces a new
  architecture**; otherwise its `bench.py` imports the architecture from the folder that does (the
  ViT family: `vit_micro_cifar/bench.py` imports `ViT` from `vit_small_cifar/model.py`). No
  architecture is ever duplicated, and no architecture lives in `common/`.
- **D2 — one training loop**, `common/loop.py::train_under_budget`, built as the *merge* of lot 7's
  and lot 8's, with their semantics preserved exactly: budget checked before each batch **after** it
  is fetched (so a run overshoots by at most one batch's processing time, `plan_lot7.md` §0.1);
  `eval_fn`'s duration subtracted from the clock (`plan_lot8.md` §0.7); one scheduler step per
  *completed* epoch; `max_steps` defensive bound. Lot 7's "re-iterate the loader forever" behaviour
  is `max_epochs=10**9`, its default.
- **D3 — one runner.** `common/runner.py::main(bench)` owns the arm loop, the WCT protocol, seeding,
  reporting and the CLI. A model's `bench.py` is a `Benchmark(...)` literal plus
  `if __name__ == "__main__": main(BENCH)` — target **≤ 50 lines**, no control flow.
- **D4 — hyperparameters live with the model**, as an `HParams(...)` literal in its `bench.py`, each
  field carrying the citation it already carries in `MODEL_DEFAULTS`. `MODEL_DEFAULTS` disappears.
- **D5 — checkpointing is added now** (`common/checkpoints.py`, ~30 lines): `--checkpoints
  0,0.01,0.1,0.5,1` writes `outputs/<model>/<arm>/ckpt_<frac>.pt` with model state, step, epoch and
  the RNG seed. Off by default; steps 2-5 of the experimental plan cannot start without it.
- **D6 — outputs indexed by model**: `benchmarks/outputs/<model_id>/<arm>/`. Existing
  `outputs/lot7_*`, `outputs/lot8_*` directories are left untouched — they are results already
  produced, not code.
- **D7 — tests keep their assertions**, only their imports move; one new parametrized test file
  covers every model.
- **D8 — `benchmarks/` becomes a package** (`__init__.py` in it and in each subfolder), imported as
  `benchmarks.common.loop`. Each `bench.py` keeps the existing one-line `sys.path` insert so
  `python benchmarks/<model>/bench.py` still works alongside `python -m benchmarks.<model>.bench`.

---

## 2. Target layout

```text
benchmarks/
├── common/
│   ├── data.py           # mnist(), cifar10(), cifar100(); Cutout, seeded_train_val_split
│   ├── loop.py           # train_under_budget(), evaluate(), sync()
│   ├── optimizers.py     # HParams, ARMS, build_optimizer(arm, model, hp)
│   ├── records.py        # StepRecord, EpochRecord, ArmResult, csv/summary/plot writers
│   ├── checkpoints.py    # checkpoint schedule + save
│   └── runner.py         # Benchmark, build_parser(), main(bench)
├── mnist_autoencoder/    # model.py  bench.py     (migrated, lots 1+7)
├── mlp_ln_mnist/         # model.py  bench.py     (A1, new)
├── cnn_gn_cifar/         # model.py  bench.py     (A2, new)
├── vit_micro_cifar/      #           bench.py     (A3, new — imports ViT from vit_small_cifar)
├── resnet20_cifar/       # model.py  bench.py     (B2, new)
├── cct_2_3x2_cifar/      # model.py  bench.py     (B1, new)
├── resnet50_cifar/       # model.py  bench.py     (migrated, lot 8)
├── vit_small_cifar/      # model.py  bench.py     (migrated, lot 8)
├── outputs/<model_id>/<arm>/
└── slurm/                # regenerated: jobs enumerate model folders
```

Deleted after migration: `mnist_autoencoder.py`, `equal_wallclock_bench.py`,
`cifar10_classification.py`, `cifar10_models.py`, `cifar10_data.py` — every line of them lands in
`common/` or in a model folder.

---

## 3. The shared API (the whole of it)

```python
# common/optimizers.py
ARMS = ("diag", "kfac", "ekfac", "tkfac", "tekfac", "adam", "adamw", "reference")

@dataclass
class HParams:
    lr: float = 1e-3
    baseline_lr: float = 1e-3        # adam / adamw (Table 9 tunes them separately)
    weight_decay: float = 0.0
    lam: float = 1e-3
    beta: float = 0.9
    gammas: tuple[float, float] = (0.92, 0.008)
    tcov: int = 100
    t_inv: int = 100                 # kfac, tkfac
    t_eig: int = 100                 # ekfac, tekfac
    t_re: int = 1                    # tekfac
    minmax: bool = True              # diag only
    conv_sua: bool = False           # the four Kronecker modes, Conv2d only
    fisher_batch_samples: int | None = None
    decoupled_wd: bool = False       # True = AdaFisherW / AdamW convention

def build_optimizer(arm: str, model: nn.Module, hp: HParams): ...
```

```python
# common/loop.py
def train_under_budget(model, optimizer, loader, loss_fn, *, budget_s,
                       max_epochs=10**9, device=CPU, scheduler=None, eval_fn=None,
                       on_step=None, max_steps=10**9, prepare_batch=default_prepare_batch,
                       log_fn=print) -> tuple[list[StepRecord], list[EpochRecord]]: ...

def evaluate(model, loader, loss_fn, device, *, prepare_batch=..., metric_fn=None) -> tuple[float, float]: ...
```

`on_step(step, epoch, model)` is the single extension point; `common/checkpoints.py` is its only
user today.

```python
# common/runner.py
@dataclass(frozen=True)
class Benchmark:
    name: str                              # = folder name = output directory
    build_model: Callable[[], nn.Module]
    build_data: Callable[..., tuple[DataLoader, DataLoader | None, DataLoader | None]]
    loss_fn: Callable[[Tensor, Tensor], Tensor]
    hparams: HParams
    prepare_batch: Callable = default_prepare_batch   # (batch) -> (inputs, targets)
    metric_fn: Callable | None = None                  # None => reconstruction task, no accuracy
    epochs: int = 50
    batch_size: int = 128
    arms: tuple[str, ...] = ("diag", "kfac", "ekfac", "tkfac", "tekfac", "adam", "adamw")

def main(bench: Benchmark, argv=None) -> None: ...
```

The two task shapes are covered by two fields and nothing else: an auto-encoder passes
`prepare_batch=lambda b: (b[0].flatten(1), b[0].flatten(1))` and `metric_fn=None`; a classifier
passes the defaults and `metric_fn=top1`.

Every `HParams` field gets a CLI override of the same name, generated by one loop over
`fields(HParams)` — not fifteen hand-written `add_argument` calls.

---

## 4. The models

| Folder | Source | `P` (asserted in the test) | Dataset | New/migrated |
|---|---|---|---|---|
| `mnist_autoencoder` | `ekfac_1806.03884.pdf` §4.1 | 2 837 314 | MNIST | migrated |
| `mlp_ln_mnist` | A1: `784-32-32-10`, `LayerNorm(32)` after each hidden `Linear` | 26 634 `[DERIVED, arithmetic checks out exactly]` | MNIST | new |
| `cnn_gn_cifar` | A2: `conv 3x3` 3→16→32→64 with bias, `GroupNorm` after each, global average pool, `Linear(64,10)` | 24 458 `[DERIVED, exact]` | CIFAR-10 | new |
| `vit_micro_cifar` | A3: patch 4, `d=32`, 2 blocks, 2 heads, `MLP=2D`, mean pooling | ≈ 21 162 `[DERIVED]` — the exact value is fixed by the config and locked by the test once built | CIFAR-10 | new |
| `resnet20_cifar` | `resnet_1512.03385.pdf` §4.2 (`6n+2`, `n=3`) | 269 722 (paper: "0.27M") | CIFAR-10 | new |
| `cct_2_3x2_cifar` | Hassani et al. arXiv:2104.05704 (`cct_2_3x2_32`: 2 layers, 2 heads, `mlp_ratio=1`, `d=128`, 2-conv `3x3` tokenizer, learned position, sequence pooling) | 283 723 (paper: "0.28M") | CIFAR-10 | new |
| `resnet50_cifar` | `resnet_1512.03385.pdf` §4.1/§4.2 | 23 519 178 | CIFAR-10 | migrated |
| `vit_small_cifar` | `vit_2010.11929.pdf` §3.1 Eq. (1)-(4), 32px adaptation | 2 685 898 | CIFAR-10 | migrated |

Two constraints, inherited from `plan_lot8.md` §0.1-§0.2, apply to **every** new `model.py` and are
the reason none of them is a `torchvision`/`timm` import:

1. **Hook safety** — `register_full_backward_hook` rejects a module whose output is later mutated in
   place: residual adds are `out = out + identity`, never `+=`, and every activation is
   `inplace=False`.
2. **Every learnable projection lives in a hooked module type** (`Conv2d`, `BatchNorm2d`, `Linear`,
   `LayerNorm`). `nn.MultiheadAttention` hides Q/K/V in a raw `Parameter` that no hook sees, so
   attention exposes a single `nn.Linear` named `qkv`. CCT's sequence pooling and the ViT class
   token are raw parameters by construction — they are *allowed*, and the per-model test asserts
   they are nonetheless updated (the `plan_lot8.md` §0.3 regression).

For A2, `GroupNorm` is **not** one of `SUPPORTED_MODULES`, so its parameters take the identity
preconditioner. That is intentional (the draft picked GN precisely to have a normalisation whose
exact block is clean), and it must be stated in the model's docstring, not discovered later. A
`--norm bn` switch builds the `BatchNorm2d` variant used by P2.

---

## 5. Migration map

| From | To | Change |
|---|---|---|
| `cifar10_data.py` | `common/data.py` | `build_dataloaders` → `cifar10(...)`; add `mnist(...)` (55k/5k split, same seeded splitter) and `cifar100(...)` (same pipeline, `CIFAR100`, its own mean/std) |
| `equal_wallclock_bench.{StepRecord,_median,_sync,write_csv,write_summary,write_plots}` | `common/records.py`, `common/loop.py` | private names lose the underscore; writers gain the epoch/accuracy columns lot 8 added |
| `equal_wallclock_bench.{train_under_time_budget,run_all_modes,main}` | `common/loop.py`, `common/runner.py` | merged into D2's single loop; `run_all_modes` is the runner's arm loop; the MNIST driver becomes `mnist_autoencoder/bench.py --budget-mode wct` |
| `cifar10_classification.{train_classifier_under_budget,evaluate}` | `common/loop.py` | gains `prepare_batch` and `on_step` |
| `cifar10_classification.{make_optimizer,MODEL_DEFAULTS,FISHER_ARMS,...}` | `common/optimizers.py` + each `bench.py` | `MODEL_DEFAULTS[m]` becomes model `m`'s `HParams` literal, citations carried over |
| `cifar10_classification.{write_*,ArmResult,run_arm,build_parser,main}` | `common/records.py`, `common/runner.py` | unchanged behaviour; output dir becomes `outputs/<name>/` |
| `mnist_autoencoder.{AutoEncoder,_mnist_loader,_make_optimizer,_train_one,main}` | `mnist_autoencoder/model.py`, `common/*` | the `reference` arm (FisherAdapTune's `AdaFisher`, loaded from its file path) moves to `common/optimizers.py` and stays available to any bench |
| `cifar10_models.{Bottleneck,ResNetCIFAR,build_resnet50_cifar}` | `resnet50_cifar/model.py` | verbatim |
| `cifar10_models.{ViTCIFAR,...,build_vit_small_cifar}` | `vit_small_cifar/model.py` | verbatim; `ViT` gains `dim/depth/heads/patch/mlp_ratio` arguments so A3 is a configuration, not a copy |
| `cifar10_models.MODELS` | deleted | the folder list is the registry; `slurm/generate_jobs.py` enumerates folders |

---

## 6. Test impact

The 167 existing tests must still pass, with **no assertion weakened**. Only imports move:

- `tests/test_equal_wallclock_bench.py` — `from equal_wallclock_bench import
  train_under_time_budget` → `from benchmarks.common.loop import train_under_budget`; call sites gain
  `budget_s=` and unpack `steps, _`. Its three tests (overshoot bound, `max_steps` guard,
  eigenbasis-mode overhead) are unchanged.
- `tests/test_cifar10_bench.py` — the six names from `cifar10_classification` split between
  `benchmarks.common.loop` and `benchmarks.common.records`; `Cutout` / `seeded_train_val_split` come
  from `benchmarks.common.data`; the two builders from their model folders; `MODELS` is replaced by
  the new registry in the parametrized test below.
- `sys.path.insert(REPO_ROOT / "benchmarks")` → `sys.path.insert(REPO_ROOT)` in both files.

**New: `tests/test_benchmark_models.py`**, parametrized over every model folder — the one test that
makes adding a model safe:

| Test | Assertion |
|---|---|
| `test_model_builds_with_expected_parameter_count` | `sum(p.numel())` equals §4's number exactly |
| `test_hooked_module_inventory` | the count of `Conv2d`/`BatchNorm2d`/`Linear`/`LayerNorm` modules, and the list of parameters that belong to **no** hooked module (must be an explicitly expected list: `cls_token`, `pos_embed`, GroupNorm, sequence pooling) |
| `test_every_parameter_is_updated` | after 2 real `AdaFisherMulti` steps on a synthetic batch, no parameter is bit-identical to its init — `plan_lot8.md` §0.3's regression, generalized. Catches unhooked parameters, in-place mutation breaking the backward hook, and pairing bugs |
| `test_all_modes_run` | 2 steps in each of the five modes; every parameter finite (`resnet50_cifar` marked slow) |
| `test_bench_spec_is_wellformed` | each `bench.py` exposes `BENCH`, its `name` equals its folder name, and `build_data` accepts the runner's keyword arguments |

Everything runs on synthetic tensors — no MNIST, no CIFAR download — the discipline of
`plan_lot7.md` §0.8.

Also to regenerate, not hand-edit: `benchmarks/slurm/*.sh` via `generate_jobs.py`, whose `MODELS`
table becomes the folder list and whose command line becomes `python -m benchmarks.<model>.bench`.

---

## 7. Sub-iterations

One at a time, each ending green.

**1.1 — harness + migration, no new network.**
Create `common/`, migrate the three existing benches into their folders, delete the five flat
modules, repoint the two test files, regenerate the SLURM jobs.
*Exit:* 167 tests pass; `benchmarks/mnist_autoencoder/bench.py --arms reference diag --epochs 1`
and `benchmarks/resnet50_cifar/bench.py --arms diag adam --epochs 1 --train-subset 512
--budget-mode epochs` both run; the CSV/summary schema is byte-identical in its columns to lot 8's.

**1.2 — the three regime-A networks** (`mlp_ln_mnist`, `cnn_gn_cifar`, `vit_micro_cifar`) +
`common/checkpoints.py`.
*Exit:* `test_benchmark_models.py` passes for them, parameter counts exact; each trains 1 epoch;
`--checkpoints 0,0.01,0.1,0.5,1` writes 5 files whose `state_dict` reloads into a fresh model.

**1.3 — the two regime-B networks** (`resnet20_cifar`, `cct_2_3x2_cifar`).
*Exit:* same tests; parameter counts match the published 0.27 M / 0.28 M; CCT's sequence-pooling
parameter is confirmed updated.

**Deferred to step 6** (`plan_exp_draft.md` §4): `gpt_micro_char` (A4) — a sequence task needs a
`prepare_batch` returning `(tokens[:, :-1], tokens[:, 1:])` and a text dataset, which the harness
supports but which is not needed before step 6; `roberta_lora` (B3), `vit_tiny_cifar` (B4).

---

## 8. Step-1 exit criteria

1. The 167 pre-existing tests pass unchanged in substance, plus the new per-model tests.
2. Every model in §4 trains for at least one epoch through the *same* `main(BENCH)`, with all five
   Fisher modes and both baselines available.
3. Each `bench.py` is ≤ 50 lines and contains no control flow — the "add a model" cost is one
   folder, one `model.py`, one `Benchmark` literal.
4. `--checkpoints` produces reloadable checkpoints at `{0, 1%, 10%, 50%, 100%}` — the input steps 2-5
   consume.
5. `ruff` and `mypy` clean on `benchmarks/`; `CLAUDE.md`'s Layout and "Running the tests" sections
   updated to the new tree.

---

## 9. Risks

- **Silent behaviour change in the merged loop.** The overshoot bound, the excluded eval time and
  the one-step-per-completed-epoch scheduler are each covered by an existing test; do the merge
  *first* and run those tests before touching anything else.
- **`reference` arm.** `mnist_autoencoder`'s comparison against `FisherAdapTune/scripts/adafisher.py`
  is the lot-1 non-regression demo; it must survive the move (loading by file path, not by import).
- **Parameter-count drift.** A3's count is the one number in §4 not cross-checked against a
  published value; fix the configuration in 1.2, then lock the count in the test.
- **Cross-folder import (D1's ViT exception).** If it starts to spread beyond the ViT family, the
  rule has failed and the architecture belongs in its own folder with an explicit config argument —
  do not let `common/` accumulate networks.
