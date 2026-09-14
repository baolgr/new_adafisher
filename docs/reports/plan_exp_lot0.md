# Lot 0 — conventions, probes, registry, and the bridge to the benchmark checkpoints

*Implementation plan for lot 0 of `plan_exp_draft.md` §9. In the repository's own numbering this is
lot 10; it is named after the experimental plan, like `plan_exp_step1.md`.*

**Goal.** `fisher_ref/` exists, is importable, and can answer four questions with no curvature code
written yet: *what precision and conventions are we in* (`conventions.py`), *which inputs do we
measure at* (`probes.py`), *what kind of layer is each parameter block* (`registry.py`), and *which
θ are available and how do I load one* (`checkpoints.py`).

**Non-goal.** No `F`, no `Ê`, no approximation, no metric. Nothing in `src/adafisher_modes/` and
nothing in `benchmarks/` changes — lot 0 is a pure reader.

---

## 0. Decisions taken here, and the two things v0 got wrong

### 0.1 v0's lot-0 exit criterion was unsatisfiable

`plan_exp_draft_v0.md` §9 gives lot 0 the deliverables *"`conventions.py`, TF32 flags, probes,
`registry.py`; tests T1-T6"* with the exit criterion *"T1-T6 pass in fp64"*. But T1 compares a dense
`F` against `GGNLinearOperator`, T3-T5 compare K-FAC variants, and T6 checks per-sample gradients of
normalisation layers — none of those objects is a lot-0 deliverable. The exit criterion tested lots
1 and 2, not lot 0.

v1 therefore gives lot 0 its own tests, **T0.1-T0.8** (`plan_exp_draft.md` §10.1), and reassigns
T1-T6 to the lots that build their prerequisites (§10.2's new "Lot" column). This is a correction to
the plan, not a weakening: the eight new tests are the ones that can actually fail at lot 0, and
three of them (T0.4, T0.5, T0.6) check properties the later lots silently assume.

The superseded simplified plan (the repository root's own `plan_exp_draft.md`, `git show
6dd981c:plan_exp_draft.md`) had already merged these deliverables into its **step 2**, alongside the
dense `F`/`E_hat` and the metrics. Lot 0 is that step's non-curvature half, split off so it can end
green on its own: conventions, probes, registry and the checkpoint bridge are what every later lot
reads, and none of them needs a single curvature matrix to be testable.

### 0.2 `fisher_ref/` is a top-level package, not `src/fisher_ref/`

`src/` holds the one shipped, installable package (`adafisher_modes`, `pyproject.toml`'s
`packages.find` include list). `benchmarks/` is already a top-level, non-installed package that
imports `adafisher_modes`. `fisher_ref/` is the same kind of object one level further out — it
imports `benchmarks` *and* `adafisher_modes` — so it sits next to `benchmarks/`, and `pyproject.toml`
is not touched at all. Tests reach it the way `tests/test_benchmark_models.py` already reaches
`benchmarks`: `sys.path.insert(0, REPO_ROOT)`.

### 0.3 The probe sets cannot go through `build_loaders`

`benchmarks/common/data.py::build_loaders` applies the **train** transform to the train split, and
for CIFAR that transform is `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` + optional
`Cutout`. A probe set must be fixed and augmentation-free (`plan_exp_draft.md` §1.1), so probes are
built from the **eval** transform on both splits, while reusing the *same* seeded partition
(`seeded_train_val_split`) so that "train probe" means a point that run actually trained on.

The dataset choice is not re-declared: every `bench.build_data` is a
`functools.partial(build_loaders, SPEC)`, so the spec is read off the partial
(`dataset_spec_of(bench)`), with a clear error if a future bench builds its data some other way.
The alternative — a second model→dataset table in `fisher_ref` — is exactly the kind of drift
`plan_exp_step1.md` D4 removed, so it is not reintroduced.

Probes are selected as the **first `n` indices of the seeded permutation's split**, not by a fresh
random draw: deterministic, no RNG consumed, and a prefix relationship holds (the first 256 probes
of a 1 000-probe set are the 256-probe set), which is what makes the `N' < N` noise-floor curve of
§3.4 free.

### 0.4 A probe set is hashed over its *content*, not over its recipe

The digest covers the version string, model, dataset, split, `n`, seed, the index list **and the
tensor bytes**. Hashing only the recipe would not notice a torchvision transform change or a
different copy of the dataset on the cluster — which is the failure the invariant exists to catch
(`plan_exp_draft.md` §8).

### 0.5 `registry.py` needs a forward pass, and says so

Whether a `Linear` is *shared* (`linear_shared`) cannot be read off the module: `nn.Linear` is the
same object in `mlp_ln_mnist` (input `(N, 784)`) and in `vit_micro_cifar`'s `qkv` (input
`(N, T, 32)`). The distinction is the input rank, and it decides which K-FAC (expand/reduce) even
applies. So `classify(model, example_input)` runs **one** forward pass under the reference-mode
context and records, per module, the input rank and the number of shared positions `T`. Without an
example input the function still returns a classification, with `positions=None` and no `head`
detection, and says so in the returned records rather than guessing.

`head` is assigned to the module whose output **is** (identity, not shape) the model's output.
For `mnist_autoencoder` that is the last decoder `Linear`; calling it the "head" is a slight abuse,
kept because §4's per-layer-type tables use `head` to mean "the layer the loss sees first", which
is exactly what it is there too.

### 0.6 Unhooked parameters are classified, not skipped

`pos_embed` (A3, B1), `cls_token` (B4) and every `GroupNorm` parameter (A2) belong to no
`AdaFisherMulti` `SUPPORTED_MODULES` instance and therefore receive the identity preconditioner. The
registry marks them `hooked=False` and still gives them a layer type (`embed`, `norm`), because the
campaign measures *curvature*, not *what the optimizer happens to precondition* — the `GroupNorm`
blocks in particular are a free control (`plan_exp_draft.md` §1, point 1). T0.6 asserts that the
registry's unhooked list is **exactly** the list `tests/test_benchmark_models.py` already locks, so
the two views of the same models cannot drift apart.

### 0.7 The bridge refuses mislabelled checkpoints

Every checkpoint written before the campaign-1 denominator fix has a fraction relative to
`max_epochs` rather than to the nominal trajectory, i.e. it is mislabelled on every arm but `diag`
(`CLAUDE.md`'s status block). A post-fix payload is recognisable: it carries `total_steps` and
`scheduled`, which the pre-fix writer did not emit. `load_theta(..., strict=True)` (the default)
rejects a payload missing either key, naming the model and arm; `strict=False` loads it and marks
the record `mislabelled=True` so a caller that knowingly wants a `diag` arm's old dump can have it
without the rest of the campaign silently inheriting a wrong x-axis.

### 0.8 What "TF32 off" has to mean in 2026

Two switches, and a third spelling. `torch.backends.cuda.matmul.allow_tf32` and
`torch.backends.cudnn.allow_tf32` are the historical pair (the second defaults to **`True`**, which
is the trap). Recent torch adds `torch.backends.{cuda.matmul,cudnn}.fp32_precision`, whose "ieee"
value is the same intent expressed differently, and on some versions writing the old attribute emits
a deprecation warning. `configure()` therefore sets whichever exists, guarded, and **returns what it
actually set** — which goes into every result file's metadata. A test that only asserted
`allow_tf32 is False` would pass on a build where that attribute no longer drives anything; T0.1
asserts the returned record and the live state agree.

`torch.use_deterministic_algorithms(True)` is deliberately **not** set by default: it turns a
missing deterministic kernel into a hard error at an arbitrary later point, which would break
callers for a property lot 0 does not need (fp64 references are reproducible enough without it). It
is an opt-in argument.

### 0.9 The independence check's tolerance is relative, and the margin is measured

`assert_sample_independent` compares `f(x)[:k]` with `f(x[:k])`. An absolute `1e-6` looked fine and
**failed on the first real model**: `cnn_gn_cifar` reloaded from its 50 % checkpoint, in eval mode,
with `GroupNorm` (which cannot mix examples at all), differs by `6.3e-6` between a batch of 256 and
a batch of 4 — different batch sizes take different BLAS kernels, so fp32 reduction order alone
moves the last significant digits.

The fix is a *relative* comparison with a dtype-keyed default (`INDEPENDENCE_RTOL`), and the
defaults are measured rather than guessed. On `cnn_gn_cifar` and `resnet20_cifar`, batch 256 against
batch 4:

| Model | mode | dtype | gap (relative) |
|---|---|---|---|
| `cnn_gn_cifar` | eval | fp32 | `9.1e-7` |
| `resnet20_cifar` | eval | fp32 | `5.4e-7` |
| `resnet20_cifar` | **train** | fp32 | **`3.6e-2`** |
| both | eval | fp64 | `~1e-15` (absolute) |

Five orders of magnitude separate "reduction order" from "BatchNorm is mixing the batch", so
`1e-5` (fp32) and `1e-10` (fp64) sit comfortably in between and the check keeps all of its
discriminating power. Do not tighten these back to an absolute tolerance: the first real model
already disproved it.

---

## 1. Modules

### 1.1 `fisher_ref/conventions.py`

```python
METRICS_VERSION = "fisher_ref/0.1"
REFERENCE_DTYPE = torch.float64      # F, E_hat, Grams, every metric
CAPTURE_DTYPE   = torch.float32      # U and the layer statistics, before the fp64 reduction

@dataclass(frozen=True)
class PrecisionState:                # what configure() actually set, for the metadata
    matmul_allow_tf32, cudnn_allow_tf32: bool | None
    matmul_fp32_precision, cudnn_fp32_precision: str | None
    cudnn_deterministic, cudnn_benchmark: bool | None
    deterministic_algorithms: bool

def configure(*, deterministic_algorithms: bool = False) -> PrecisionState
def precision_state() -> PrecisionState                 # read-only snapshot
def run_metadata(extra: Mapping | None = None) -> dict  # version, torch, precision, git sha
@contextmanager
def reference_mode(model)                               # eval everywhere, restored on exit
def assert_sample_independent(model, inputs, *, k, atol) # BN-in-train detector
def rvec(M) / unrvec(v, shape) / kron_rvec(output_factor, input_factor)
```

`kron_rvec(B, A) = torch.kron(B, A)` is the whole convention: `rvec(B M A^T) = (B ⊗ A) rvec(M)`, so
the papers' `A ⊗ B` (input factor first, `cvec`) is this repository's `B ⊗ A`. One function, one
docstring, one numeric test — instead of the re-derivation being repeated in each lot.

### 1.2 `fisher_ref/probes.py`

```python
PROBE_VERSION = "probes/1"

@dataclass(frozen=True)
class ProbeSet:
    model, dataset, split, seed, n, indices, inputs, targets, digest, version
    def as_model_batch(self, bench) -> tuple[Tensor, Tensor]   # applies bench.prepare_batch
    def head(self, n) -> ProbeSet                              # the prefix property of §0.3
    def save(path) / load(path)

def dataset_spec_of(bench) -> DatasetSpec
def build_probe_set(bench, *, split, n, seed=0, data_root=..., allow_download=False,
                    spec=None) -> ProbeSet
def build_probe_pair(bench, *, n, seed=0, ...) -> tuple[ProbeSet, ProbeSet]   # train, val
```

`spec=` exists so the tests can inject a synthetic dataset and stay offline
(`plan_lot7.md` §0.8's discipline); the real-data path is exercised by one test guarded on
`benchmarks/data/` being staged.

### 1.3 `fisher_ref/registry.py`

```python
LAYER_TYPES = ("linear", "linear_shared", "conv", "norm", "embed",
               "lora_a", "lora_b", "head", "other")

@dataclass(frozen=True)
class LayerInfo:
    name, module_type, layer_type, params, n_params, hooked,
    input_ndim, positions, is_output

def classify(model, example_input=None) -> list[LayerInfo]
def by_type(infos) -> dict[str, list[LayerInfo]]
def unhooked_parameters(infos) -> list[str]
def assert_partitions(model, infos)      # every parameter covered exactly once
```

Rules, in order: `lora_a`/`lora_b` by parameter name; `nn.Embedding` → `embed`; any `*Norm` →
`norm`; `Conv*d` → `conv`; `Linear` → `head` if its output is the model output, else
`linear_shared` if its input rank > 2, else `linear`; a raw `Parameter` on a container →
`embed` if its name contains `pos_embed`/`cls_token`/`embed`, else `other`.

### 1.4 `fisher_ref/checkpoints.py`

```python
@dataclass(frozen=True) class RunRef:  model, arm, seed, directory, manifest, checkpoints: dict[float, Path]
@dataclass(frozen=True) class LoadedCheckpoint: model, run, fraction, step, epoch, total_steps, scheduled, mislabelled, path

def discover_runs(outputs_root=..., *, model=None, arm=None, seed=None) -> list[RunRef]
def available_seeds(outputs_root, model) -> list[int]
def load_theta(run, fraction, *, strict=True, device="cpu") -> LoadedCheckpoint
```

Both layouts, one scanner: `outputs/<model>/<arm>/` (seed from the manifest, default 0) and
`outputs/seeds/<model>/seed<n>/<arm>/` (seed from the directory, cross-checked against the
manifest). `_calibration` and any directory whose name is not in `benchmarks.common.optimizers.ARMS`
are skipped. The model is rebuilt through its own `bench.build_model`, with the `model_choices`
keywords (A2's `--norm`) read from the run's `manifest.json`, so a `bn` run does not silently reload
into a `gn` network.

---

## 2. Tests — `tests/test_fisher_ref_lot0.py`

One file, the eight T0 tests of `plan_exp_draft.md` §10.1, all offline except one guarded on staged
datasets. Nothing downloads; nothing needs a GPU; nothing reads `benchmarks/outputs/`, which is
gitignored and may be absent (the checkpoint tests build their own tree with the **real**
`CheckpointWriter`, so they test the real payload format rather than a copy of it).

| Test | What would break it |
|---|---|
| T0.1 | a torch upgrade that renames the TF32 knobs; `cudnn.allow_tf32` silently back to `True` |
| T0.2 | a result file written without `metrics_version` |
| T0.3 | someone "simplifying" `kron_rvec` to `kron(A, B)` |
| T0.4 | a reference computed with BN in train mode (per-sample gradients then undefined) |
| T0.5 | probes built through `build_loaders`, i.e. with augmentation; a split that is not the run's |
| T0.6 | a new model folder whose parameters the registry does not cover; a GroupNorm silently counted as hooked |
| T0.7 | either output layout being unreadable; a `norm=bn` run reloaded into a `gn` model |
| T0.8 | a pre-fix checkpoint silently accepted, with fractions off by `--max-epoch-factor` |

---

## 3. Exit criteria

1. The 261 pre-existing tests still pass, unmodified — lot 0 adds files, changes none.
2. T0.1-T0.8 pass.
3. `ruff` and `mypy` clean on `fisher_ref/`.
4. `CLAUDE.md`'s Layout and "Running the tests" sections mention `fisher_ref/` and the new test file.
5. One end-to-end smoke, run by hand and recorded in §4: build the train and val probe sets of one
   regime-A model from the staged datasets, classify its layers, and load one real checkpoint.

---

## 4. Smoke run (exit criterion 5) — done

`fisher_ref/` run end to end on the real artifacts, on the staged CIFAR-10 and one real campaign-1
checkpoint (`benchmarks/outputs/cnn_gn_cifar/diag/`):

```text
precision: TF32 off | metrics_version: fisher_ref/0.1
probes: 256 train / 256 val, cifar10, digest 9daab9ba63ba / 26fdff092a51, disjoint=True
runs discovered: 42; seeds for cnn_gn_cifar: [0]
  ckpt 0.0   step      0 epoch   0 progress 0.000 scheduled=True
  ckpt 0.01  step    105 epoch   0 progress 0.010 scheduled=True
  ckpt 0.1   step   1053 epoch   2 progress 0.100 scheduled=True
  ckpt 0.5   step   5265 epoch  14 progress 0.500 scheduled=True
  ckpt 1.0   step  10530 epoch  29 progress 1.000 scheduled=True
theta at 50%: loss=1.0644 on the train probes
layers={'conv': 3, 'head': 1, 'norm': 3}; unhooked=6 GroupNorm parameters
    features.0   Conv2d     conv   P=448    T=1024 hooked=True
    features.1   GroupNorm  norm   P=32     T=1024 hooked=False
    features.4   Conv2d     conv   P=4640   T=256  hooked=True
    features.5   GroupNorm  norm   P=64     T=256  hooked=False
    features.8   Conv2d     conv   P=18496  T=64   hooked=True
    features.9   GroupNorm  norm   P=128    T=64   hooked=False
    head         Linear     head   P=650    T=1    hooked=True
```

Three things worth recording from it:

- **42 runs** are discoverable today: 6 models x 7 arms, seed 0, all post-fix (`mislabelled=False`
  on every one). `resnet50_cifar` and `vit_small_cifar` contribute none, as §3.4 of the plan says;
  the extra seeds contribute none *yet* — `available_seeds(model=...)` returns `[0]` everywhere,
  which is the honest answer while the job is in flight, not an error.
- The five checkpoints of an unbudgeted reference arm land exactly on their nominal fractions
  (`progress` = the label), which is what makes them comparable across arms.
- `loss = 1.0644` on 256 held-out-of-augmentation train probes, against the `1.0484` train loss the
  run's own manifest records for its last epoch — consistent, and a first sanity check that the
  reloaded θ is the trained one and not a fresh initialisation.

## 5. Completion

Delivered: `fisher_ref/{__init__,conventions,probes,registry,checkpoints}.py` and
`tests/test_fisher_ref_lot0.py` (27 tests, 2 of them `slow`). The suite goes from **261 to 288
passing**, with no pre-existing test modified and no file outside `fisher_ref/`, `tests/` and
`docs/reports/` touched. `ruff check` and `mypy` are clean on `fisher_ref/`.

Two things this lot found, both measured rather than assumed: §0.9's tolerance (an absolute `1e-6`
is wrong for fp32 forward passes, and the real margin is five orders of magnitude wide), and the
fact that every checkpoint currently on disk is post-fix, so `strict=True` costs nothing today —
its value is that a re-appearing archive cannot silently re-enter the campaign.
