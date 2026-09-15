"""Lot 0 of the Fisher-drift campaign (``docs/reports/plan_exp_draft.md`` §10.1, T0.1-T0.8;
``plan_exp_lot0.md`` §2).

`fisher_ref/` is a **reader**: it trains nothing and changes nothing in ``benchmarks/`` or
``src/adafisher_modes/``. These tests check the four things every later lot inherits — the
precision and vectorisation conventions, the probe sets, the layer-type registry, and the bridge to
the trajectory checkpoints.

Everything is offline. The probe tests use a synthetic ``DatasetSpec`` (no MNIST, no CIFAR) except
for one test guarded on the datasets being staged under ``benchmarks/data/``; the checkpoint tests
build their own output tree with the **real** ``CheckpointWriter``, so they test the payload format
the harness actually writes rather than a copy of it, and never read ``benchmarks/outputs/`` (which
is gitignored and may be absent).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from conftest import seed_all

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from test_benchmark_models import EXPECTED as MODEL_CONTRACT  # noqa: E402

from benchmarks.common.checkpoints import CheckpointWriter  # noqa: E402
from benchmarks.common.data import DatasetSpec, seeded_train_val_split  # noqa: E402
from benchmarks.common.runner import Benchmark, discover_benchmarks  # noqa: E402
from fisher_ref import conventions, probes, registry  # noqa: E402
from fisher_ref.checkpoints import available_seeds, discover_runs, load_theta  # noqa: E402

BENCHES: Dict[str, Benchmark] = discover_benchmarks()
SLOW = {"resnet50_cifar", "resnet50_cifar100", "resnet50_imagenet", "vit_small_imagenet"}


# ----------------------------------------------------------------------------------------------
# T0.1 / T0.2 — precision policy and metadata
# ----------------------------------------------------------------------------------------------


def test_t0_1_configure_turns_tf32_off_and_reports_what_it_set() -> None:
    """The cuDNN switch defaults to True, and the knob has three spellings across torch versions;
    asserting one attribute is not evidence (``plan_exp_lot0.md`` §0.8), so the returned record and
    the live state must agree and both must say "no TF32".
    """
    state = conventions.configure()
    assert state.tf32_is_off(), state
    assert state == conventions.precision_state(), "configure() reported a state it did not set"
    assert state.matmul_allow_tf32 is not True and state.cudnn_allow_tf32 is not True
    assert state.deterministic_algorithms is False  # opt-in only, see §0.8


def test_t0_2_metadata_carries_the_invariants() -> None:
    meta = conventions.run_metadata({"protocol": "P1"})
    assert meta["metrics_version"] == conventions.METRICS_VERSION
    assert meta["reference_dtype"] == "torch.float64"
    assert conventions.REFERENCE_DTYPE is torch.float64
    assert meta["precision"]["cudnn_allow_tf32"] is not True
    assert meta["protocol"] == "P1"


# ----------------------------------------------------------------------------------------------
# T0.3 — the rvec convention
# ----------------------------------------------------------------------------------------------


def test_t0_3_rvec_kronecker_convention() -> None:
    """``rvec(u v^T) = u (x) v`` and ``rvec(B M A^T) = (B (x) A) rvec(M)``, so the papers' ``A (x)
    B`` is ``kron(B, A)`` here. A "simplification" to ``kron(A, B)`` fails the second assertion.
    """
    seed_all(0)
    d_out, d_in = 4, 3
    u, v = torch.randn(d_out, dtype=torch.float64), torch.randn(d_in, dtype=torch.float64)
    assert torch.allclose(conventions.rvec(torch.outer(u, v)), torch.kron(u, v))

    M = torch.randn(d_out, d_in, dtype=torch.float64)
    A = torch.randn(d_in, d_in, dtype=torch.float64)
    B = torch.randn(d_out, d_out, dtype=torch.float64)
    applied = conventions.rvec(B @ M @ A.T)
    assert torch.allclose(conventions.kron_rvec(B, A) @ conventions.rvec(M), applied, atol=1e-12)
    assert not torch.allclose(torch.kron(A, B) @ conventions.rvec(M), applied)
    assert torch.equal(conventions.unrvec(conventions.rvec(M), M.shape), M)


# ----------------------------------------------------------------------------------------------
# T0.4 — reference mode and per-sample independence
# ----------------------------------------------------------------------------------------------


class _BatchNormNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 3, 3, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Linear(3 * 4 * 4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.bn(self.conv(x))).flatten(1))


def test_t0_4_reference_mode_gives_sample_independence_and_restores_modes() -> None:
    """With BN in train mode a sample's output depends on the rest of the batch, so per-sample
    gradients — and therefore ``F`` and ``E_hat`` — are not defined (``plan_exp_draft.md`` §2.5).
    The check must *fail* there and pass under ``reference_mode``, and the previous modes must come
    back.
    """
    seed_all(0)
    model = _BatchNormNet()
    inputs = torch.randn(8, 2, 4, 4)

    model.train()
    with pytest.raises(AssertionError, match="not sample-independent"):
        conventions.assert_sample_independent(model, inputs, k=2)

    with conventions.reference_mode(model):
        assert not any(m.training for m in model.modules())
        conventions.assert_sample_independent(model, inputs, k=2)
    assert all(m.training for m in model.modules()), "reference_mode did not restore train mode"

    model.eval()
    with conventions.reference_mode(model):
        pass
    assert not any(m.training for m in model.modules())


def test_t0_4_independence_tolerance_is_relative_and_dtype_aware() -> None:
    """The check compares *relatively*, with a dtype-dependent default: in fp32 two batch sizes
    take different BLAS kernels, so a sample-independent network still differs in the last digits
    (measured: 5e-7 relative), while BN-in-train differs by ~1e-2 — the margin the default sits in
    (``conventions.INDEPENDENCE_RTOL``). A too-tight tolerance must be the caller's choice.
    """
    seed_all(0)
    model = _BatchNormNet().eval()
    inputs = torch.randn(16, 2, 4, 4)
    conventions.assert_sample_independent(model, inputs, k=2)
    conventions.assert_sample_independent(model.double(), inputs.double(), k=2)

    assert conventions.INDEPENDENCE_RTOL[torch.float64] < conventions.INDEPENDENCE_RTOL[
        torch.float32]
    model.float().train()
    with pytest.raises(AssertionError, match=r"max\(\|f\|, 1\)"):
        conventions.assert_sample_independent(model, inputs, k=2)


# ----------------------------------------------------------------------------------------------
# T0.5 — probe sets
# ----------------------------------------------------------------------------------------------


class _SyntheticDataset:
    """A torchvision-shaped dataset: ``(HWC uint8 array, int label)``, deterministic in its index.

    Deterministic content is what makes "augmentation-free" testable: two builds of the same probe
    set may differ only if a random transform was applied.
    """

    def __init__(self, root: str, train: bool = True, download: bool = False,
                 transform=None, n: int = 20) -> None:
        self.transform = transform
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Tuple[object, int]:
        rng = np.random.RandomState(index)
        image = rng.randint(0, 256, size=(8, 8, 3), dtype=np.uint8)
        return (self.transform(image) if self.transform else image), index % 10


SYNTHETIC_SPEC = DatasetSpec("synthetic", _SyntheticDataset, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25),
                             val_size=4, augment=True)


def _synthetic_probes(split: str, n: int, seed: int = 0) -> probes.ProbeSet:
    return probes.build_probe_set(BENCHES["cnn_gn_cifar"], split=split, n=n, seed=seed,
                                 data_root="unused", spec=SYNTHETIC_SPEC)


def test_t0_5_probe_sets_are_deterministic_disjoint_and_augmentation_free() -> None:
    train, val = _synthetic_probes("train", 6), _synthetic_probes("val", 4)
    again = _synthetic_probes("train", 6)

    # Deterministic, content-hashed: the spec declares augment=True, so a probe built through the
    # *train* transform would differ between the two builds (plan_exp_lot0.md §0.3).
    assert train.digest == again.digest
    assert torch.equal(train.inputs, again.inputs)
    assert train.digest != val.digest

    # The run's own partition, and its prefix property.
    expected_train, expected_val = seeded_train_val_split(len(_SyntheticDataset("")), 4, 0)
    assert list(train.indices) == expected_train[:6]
    assert list(val.indices) == expected_val[:4]
    assert not set(train.indices) & set(val.indices)
    assert train.head(3).digest == _synthetic_probes("train", 3).digest

    assert train.metadata()["probe_version"] == probes.PROBE_VERSION
    with pytest.raises(ValueError):
        _synthetic_probes("val", 5)  # only 4 val examples exist
    with pytest.raises(ValueError):
        _synthetic_probes("nonexistent", 1)


def test_t0_5_test_probes_are_a_third_out_of_sample_source() -> None:
    """``split="test"`` reads the dataset's own held-out set instead of partitioning the training
    one — added at lot 1 because ``val`` holds 5 000 images and the HF1 null gap is dominated by
    the out-of-sample side (``plan_exp_lot1.md`` §6.4). It is a *separate* split, never silently
    pooled with ``val``: the digest distinguishes them, and whether they may be pooled is a
    measured question (``probes.build_probe_set``'s docstring).
    """
    test = _synthetic_probes("test", 6)
    assert test.split == "test" and len(test) == 6
    assert list(test.indices) == list(range(6))          # natural order, hence head()'s prefix
    assert test.head(3).digest == _synthetic_probes("test", 3).digest
    assert test.digest != _synthetic_probes("train", 6).digest
    assert torch.equal(test.inputs, _synthetic_probes("test", 6).inputs)

    # The synthetic dataset ignores train=/False, so this asserts the *plumbing*, not disjointness:
    # on a real dataset the test set is a different file entirely.
    assert probes.build_probe_set(BENCHES["cnn_gn_cifar"], split="test", n=2,
                                  spec=SYNTHETIC_SPEC).split == "test"


def test_t0_5_probe_sets_round_trip_through_disk(tmp_path: Path) -> None:
    train = _synthetic_probes("train", 5)
    reloaded = probes.ProbeSet.load(train.save(tmp_path / "train.pt"))
    assert reloaded.digest == train.digest and torch.equal(reloaded.inputs, train.inputs)

    payload = torch.load(tmp_path / "train.pt", weights_only=True)
    payload["inputs"] = payload["inputs"] + 1.0
    torch.save(payload, tmp_path / "tampered.pt")
    with pytest.raises(ValueError, match="fails its own digest"):
        probes.ProbeSet.load(tmp_path / "tampered.pt")


@pytest.mark.skipif(not (REPO_ROOT / "benchmarks/data").is_dir(),
                    reason="datasets not staged under benchmarks/data/")
def test_t0_5_real_probe_sets_match_the_training_split() -> None:
    """The one test that touches a real dataset (never downloading it): the probes of a CIFAR
    *train* split are bit-identical across two builds, i.e. the eval transform was used.
    """
    bench = BENCHES["cnn_gn_cifar"]
    train, val = probes.build_probe_pair(bench, n=4, seed=0)
    assert probes.dataset_spec_of(bench).name == "cifar10"
    assert train.inputs.shape == (4, 3, 32, 32) and train.dataset == "cifar10"
    assert torch.equal(train.inputs, probes.build_probe_set(bench, split="train", n=4).inputs)
    assert not set(train.indices) & set(val.indices)
    inputs, targets = train.as_model_batch(bench)
    assert inputs.shape == (4, 3, 32, 32) and targets.shape == (4,)

    # The real test set is a different file, so it is genuinely 10 000 further held-out images —
    # the point of the lot-1 addition. Its probes differ from the val ones at the same size.
    test = probes.build_probe_set(bench, split="test", n=4)
    assert test.inputs.shape == (4, 3, 32, 32) and test.split == "test"
    assert not torch.equal(test.inputs, val.inputs)

    autoencoder = BENCHES["mnist_autoencoder"]
    ae_probes = probes.build_probe_set(autoencoder, split="train", n=3)
    ae_inputs, ae_targets = ae_probes.as_model_batch(autoencoder)
    assert ae_inputs.shape == (3, 784) and torch.equal(ae_inputs, ae_targets)


# ----------------------------------------------------------------------------------------------
# T0.6 — the layer-type registry
# ----------------------------------------------------------------------------------------------


EXPECTED_LAYER_TYPES: Dict[str, Dict[str, int]] = {
    "mnist_autoencoder": {"linear": 7, "head": 1},
    "mlp_ln_mnist": {"linear": 2, "norm": 2, "head": 1},
    "cnn_gn_cifar": {"conv": 3, "norm": 3, "head": 1},
    "vit_micro_cifar": {"conv": 1, "linear_shared": 8, "norm": 5, "embed": 1, "head": 1},
    "resnet20_cifar": {"conv": 19, "norm": 19, "head": 1},
    "cct_2_3x2_cifar": {"conv": 2, "linear_shared": 9, "norm": 5, "embed": 1, "head": 1},
    "vit_small_cifar": {"conv": 1, "linear_shared": 24, "norm": 13, "embed": 2, "head": 1},
    "resnet50_cifar": {"conv": 53, "norm": 53, "head": 1},
    # CIFAR-100 and ImageNet32: the same architectures with a wider head, so the layer-type
    # inventory is by construction identical to the CIFAR-10 row above it. Only the two ImageNet
    # 224 px benches are genuinely different networks — ResNet-50's ImageNet stem keeps the 53/53
    # count (a 7x7 conv is still one conv), and ViT-S/16 is depth 12 against ViT-S/4's depth 6,
    # hence 48 shared Linears and 25 norms rather than 24 and 13.
    "cnn_gn_cifar100": {"conv": 3, "norm": 3, "head": 1},
    "vit_micro_cifar100": {"conv": 1, "linear_shared": 8, "norm": 5, "embed": 1, "head": 1},
    "resnet20_cifar100": {"conv": 19, "norm": 19, "head": 1},
    "cct_2_3x2_cifar100": {"conv": 2, "linear_shared": 9, "norm": 5, "embed": 1, "head": 1},
    "vit_small_cifar100": {"conv": 1, "linear_shared": 24, "norm": 13, "embed": 2, "head": 1},
    "resnet50_cifar100": {"conv": 53, "norm": 53, "head": 1},
    "cnn_gn_imagenet": {"conv": 3, "norm": 3, "head": 1},
    "vit_micro_imagenet": {"conv": 1, "linear_shared": 8, "norm": 5, "embed": 1, "head": 1},
    "resnet20_imagenet": {"conv": 19, "norm": 19, "head": 1},
    "cct_2_3x2_imagenet": {"conv": 2, "linear_shared": 9, "norm": 5, "embed": 1, "head": 1},
    "resnet50_imagenet": {"conv": 53, "norm": 53, "head": 1},
    "vit_small_imagenet": {"conv": 1, "linear_shared": 48, "norm": 25, "embed": 2, "head": 1},
}


def _model_ids() -> List[object]:
    return [pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n
            for n in sorted(EXPECTED_LAYER_TYPES)]


def _classify(name: str) -> Tuple[nn.Module, List[registry.LayerInfo]]:
    bench = BENCHES[name]
    seed_all(0)
    model = bench.build_model(**{k: v[0] for k, v in bench.model_choices.items()})
    shape = MODEL_CONTRACT[name].input_shape
    inputs, _ = bench.prepare_batch((torch.randn(2, *shape), torch.zeros(2).long()))
    return model, registry.classify(model, inputs)


def test_t0_6_every_model_folder_has_an_expected_inventory() -> None:
    assert set(EXPECTED_LAYER_TYPES) == set(BENCHES), (
        "a benchmarks/<model>/bench.py exists with no expected layer-type inventory here"
    )


@pytest.mark.parametrize("name", _model_ids())
def test_t0_6_registry_partitions_and_classifies(name: str) -> None:
    model, infos = _classify(name)
    registry.assert_partitions(model, infos)
    assert registry.type_counts(infos) == dict(sorted(EXPECTED_LAYER_TYPES[name].items()))
    assert sum(info.n_params for info in infos) == MODEL_CONTRACT[name].params
    assert all(info.layer_type in registry.LAYER_TYPES for info in infos)
    assert sum(info.is_output for info in infos) == 1, "exactly one module produces the output"


@pytest.mark.parametrize("name", _model_ids())
def test_t0_6_unhooked_list_matches_the_benchmark_contract(name: str) -> None:
    """The registry's ``hooked=False`` set must be exactly the list
    ``tests/test_benchmark_models.py`` already locks — the two views of the same models cannot
    drift apart (``plan_exp_lot0.md`` §0.6).
    """
    _model, infos = _classify(name)
    assert registry.unhooked_parameters(infos) == MODEL_CONTRACT[name].unhooked


def test_t0_6_weight_sharing_is_read_from_a_forward_pass() -> None:
    """``nn.Linear`` is the same class shared (``qkv``, input rank 3) and unshared (an MLP, rank 2);
    the distinction decides whether K-FAC-expand/reduce even applies, so it is measured, not
    guessed — and without an example input it is honestly reported as unknown.
    """
    _model, mlp = _classify("mlp_ln_mnist")
    _model2, vit = _classify("vit_micro_cifar")
    by_name = {info.name: info for info in vit}

    assert all(info.positions == 1 and info.shared is False
               for info in mlp if info.layer_type in ("linear", "head"))
    qkv = by_name["blocks.0.attn.qkv"]
    assert qkv.layer_type == "linear_shared" and qkv.positions == 64 and qkv.shared is True
    assert by_name["patch_embed.proj"].layer_type == "conv"
    assert by_name["pos_embed"].layer_type == "embed" and not by_name["pos_embed"].hooked

    bench = BENCHES["vit_micro_cifar"]
    blind = registry.classify(bench.build_model())
    blind_qkv = {info.name: info for info in blind}["blocks.0.attn.qkv"]
    assert blind_qkv.positions is None and blind_qkv.shared is None
    assert blind_qkv.layer_type == "linear"  # no forward pass: reported unshared, not guessed


# ----------------------------------------------------------------------------------------------
# T0.7 / T0.8 — the checkpoint bridge
# ----------------------------------------------------------------------------------------------


def _write_run(root: Path, model: str, arm: str, *, seed: int, fractions=(0.0, 0.5, 1.0),
               model_kwargs: Dict[str, str] | None = None,
               record_model: bool = False) -> nn.Module:
    """One arm's output directory, written with the real ``CheckpointWriter``.

    ``record_model`` writes ``config["model"]`` the way ``runner.main`` now does; leaving it off
    reproduces the manifests of every run already on disk, which is the fallback path.
    """
    bench = BENCHES[model]
    seed_all(seed)
    net = bench.build_model(**(model_kwargs or {}))
    writer = CheckpointWriter(output_dir=root / arm, fractions=fractions, total_steps=100,
                              seed=seed)
    for fraction in fractions:
        writer.save(fraction, int(fraction * 100), int(fraction * 10), net,
                    scheduled=fraction != 1.0)
    config: Dict[str, Any] = {"seed": seed, "epochs": 5, **(model_kwargs or {})}
    if record_model:
        config["model"] = model
    manifest = {"config": config,
                "arms": {arm: {"steps": 100, "checkpoints": dict(writer.written)}}}
    (root / "manifest.json").write_text(__import__("json").dumps(manifest))
    return net


def test_t0_7_bridge_discovers_both_layouts_and_reloads_theta(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    # Layout 1: outputs/<model>/<arm>/, seed from the manifest. A `bn` run, to prove the
    # architecture variant is honoured: a `gn` network would reject the BatchNorm state_dict.
    seed0 = _write_run(outputs / "cnn_gn_cifar", "cnn_gn_cifar", "kfac", seed=0,
                       model_kwargs={"norm": "bn"})
    # Layout 2: outputs/seeds/<model>/seed<n>/<arm>/, seed from the directory name.
    _write_run(outputs / "seeds" / "mlp_ln_mnist" / "seed3", "mlp_ln_mnist", "diag", seed=3)
    (outputs / "cnn_gn_cifar" / "_calibration").mkdir()  # must be skipped, not treated as an arm

    runs = discover_runs(outputs)
    assert {(r.model, r.arm, r.seed) for r in runs} == {
        ("cnn_gn_cifar", "kfac", 0), ("mlp_ln_mnist", "diag", 3)
    }
    assert available_seeds(outputs, model="mlp_ln_mnist") == [3]
    assert discover_runs(outputs, model="nowhere") == [] and available_seeds(outputs) == [0, 3]

    run = next(r for r in runs if r.model == "cnn_gn_cifar")
    assert run.fractions == (0.0, 0.5, 1.0) and run.model_kwargs() == {"norm": "bn"}
    assert run.summary()["steps"] == 100

    loaded = load_theta(run, 0.5)
    assert loaded.step == 50 and loaded.total_steps == 100 and loaded.progress == 0.5
    assert loaded.scheduled is True and loaded.mislabelled is False
    for (name, restored), (_, original) in zip(loaded.model.named_parameters(),
                                               seed0.named_parameters()):
        assert torch.equal(restored, original), name
    assert load_theta(run, 1.0).scheduled is False  # pinned to the arm's own end
    assert load_theta(run, 0.0, build_model=False).step == 0
    with pytest.raises(KeyError):
        load_theta(run, 0.1)


def test_t0_7_bridge_descends_into_a_declared_dataset_group(tmp_path: Path) -> None:
    """``outputs/<group>/<model>/<arm>/`` is discovered alongside the flat layout — otherwise a
    bench gaining an ``output_group`` would silently make its whole result tree invisible.

    Only a *declared* group name is descended into. A legacy directory that merely looks like one
    (``outputs/lot8_cifar10/``) stays a model directory, so nothing is invented.
    """
    outputs = tmp_path / "outputs"
    _write_run(outputs / "cifar10" / "cnn_gn_cifar", "cnn_gn_cifar", "diag", seed=0,
               record_model=True)
    _write_run(outputs / "mnist" / "mlp_ln_mnist", "mlp_ln_mnist", "adamw", seed=0,
               record_model=True)
    _write_run(outputs / "resnet20_cifar", "resnet20_cifar", "diag", seed=0)  # still flat
    _write_run(outputs / "lot8_cifar10" / "cnn_gn_cifar", "cnn_gn_cifar", "diag", seed=0)

    found = {(r.model, r.arm) for r in discover_runs(outputs)}
    assert found == {("cnn_gn_cifar", "diag"), ("mlp_ln_mnist", "adamw"),
                     ("resnet20_cifar", "diag")}
    assert discover_runs(outputs, model="cnn_gn_cifar")[0].bench_name == "cnn_gn_cifar"
    # A model appears once, not twice: a directory holding arms is never also treated as a group.
    assert len(discover_runs(outputs, model="cnn_gn_cifar")) == 1


def test_t0_7_bench_is_resolved_from_the_manifest_not_the_directory_name(tmp_path: Path) -> None:
    """``--output-dir`` is free-form, so a run directory's name is a label, not an identifier.

    The case that made this concrete: A2's BatchNorm variant is written to
    ``outputs/cnn_gn_cifar_bn/`` by ``--norm bn``, and there is no ``benchmarks/cnn_gn_cifar_bn/``
    folder — so resolving the bench by directory name left its 28 checkpoints unloadable, and that
    is exactly the run protocol P2 compares the GroupNorm one against. ``runner.main`` now records
    ``config["model"]``; :data:`_LEGACY_BENCH_OF_DIRECTORY` covers the trees written before it did.
    """
    outputs = tmp_path / "outputs"
    recorded = _write_run(outputs / "some_ad_hoc_dir", "cnn_gn_cifar", "diag", seed=0,
                          model_kwargs={"norm": "bn"}, record_model=True)
    legacy = _write_run(outputs / "cnn_gn_cifar_bn", "cnn_gn_cifar", "diag", seed=0,
                        model_kwargs={"norm": "bn"})  # no config["model"]: the on-disk shape
    _write_run(outputs / "unknown_dir", "cnn_gn_cifar", "diag", seed=0)

    runs = {r.model: r for r in discover_runs(outputs)}
    # `model` stays the directory label — two runs of one bench must not collapse into one row.
    assert set(runs) == {"some_ad_hoc_dir", "cnn_gn_cifar_bn", "unknown_dir"}
    assert runs["some_ad_hoc_dir"].bench_name == "cnn_gn_cifar"      # from the manifest
    assert runs["cnn_gn_cifar_bn"].bench_name == "cnn_gn_cifar"      # from the legacy map
    assert runs["unknown_dir"].bench_name == "unknown_dir"           # no evidence: unchanged

    for key, original in (("some_ad_hoc_dir", recorded), ("cnn_gn_cifar_bn", legacy)):
        loaded = load_theta(runs[key], 0.5)
        assert runs[key].model_kwargs() == {"norm": "bn"}
        # The variant is honoured, not merely tolerated: a `gn` rebuild would carry GroupNorm.
        assert any(isinstance(m, nn.BatchNorm2d) for m in loaded.model.modules()), key
        assert not any(isinstance(m, nn.GroupNorm) for m in loaded.model.modules()), key
        for (name, restored), (_, before) in zip(loaded.model.named_parameters(),
                                                 original.named_parameters()):
            assert torch.equal(restored, before), f"{key}/{name}"

    # An unresolvable directory fails loudly, and the message says what to do about it.
    with pytest.raises(KeyError, match="_LEGACY_BENCH_OF_DIRECTORY"):
        load_theta(runs["unknown_dir"], 0.5)
    assert load_theta(runs["unknown_dir"], 0.5, build_model=False).step == 50  # metadata still ok


def test_t0_7_the_runner_records_the_bench_it_ran(tmp_path: Path) -> None:
    """The other half of the fix: without ``config["model"]`` in what ``runner.main`` writes, the
    bridge is back to guessing from a directory name.
    """
    import json

    from benchmarks.common.records import write_manifest

    path = tmp_path / "manifest.json"
    write_manifest([], {"model": "cnn_gn_cifar", "output_group": "", "norm": "bn"}, path)
    config = json.loads(path.read_text())["config"]
    assert config["model"] == "cnn_gn_cifar" and config["norm"] == "bn"
    assert "output_group" in config

    source = (REPO_ROOT / "benchmarks" / "common" / "runner.py").read_text()
    assert '"model": bench.name' in source, "runner.main must record the bench it ran"


def test_t0_8_bridge_rejects_a_pre_fix_checkpoint(tmp_path: Path) -> None:
    """A payload without ``total_steps``/``scheduled`` predates the denominator fix: its fractions
    are ``--max-epoch-factor`` times too long on every arm but ``diag``. Loading it silently would
    put the whole campaign on a wrong x-axis (``plan_exp_draft.md`` §3.5, rule 3).
    """
    outputs = tmp_path / "outputs"
    arm_dir = outputs / "mlp_ln_mnist" / "adam"
    arm_dir.mkdir(parents=True)
    seed_all(0)
    net = BENCHES["mlp_ln_mnist"].build_model()
    torch.save({"model_state_dict": net.state_dict(), "fraction": 0.5, "step": 7, "epoch": 1,
                "seed": 0}, arm_dir / "ckpt_0.5.pt")

    run = discover_runs(outputs)[0]
    with pytest.raises(ValueError, match="predates the checkpoint-denominator fix"):
        load_theta(run, 0.5)

    lenient = load_theta(run, 0.5, strict=False)
    assert lenient.mislabelled is True and lenient.total_steps is None
    assert lenient.progress is None  # no denominator: no honest x-axis
    assert lenient.metadata()["mislabelled"] is True


def test_t0_8_missing_runs_are_absent_not_an_error(tmp_path: Path) -> None:
    """The extra-seed campaign is in flight: a consumer reports the seeds it found, and an empty
    or absent outputs tree is a normal answer (``plan_exp_draft.md`` §3.4).
    """
    assert discover_runs(tmp_path / "does_not_exist") == []
    (tmp_path / "empty").mkdir()
    assert discover_runs(tmp_path / "empty") == [] and available_seeds(tmp_path / "empty") == []
