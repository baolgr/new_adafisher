"""CIFAR-100 and ImageNet-1K: the data pipeline and the twelve benches built on it.

``tests/test_benchmark_models.py`` already covers every ``benchmarks/<model>/`` folder's
architecture — parameter count, hooked-module inventory, "no parameter left un-updated", all five
Fisher modes — and picked the new folders up the moment they existed. What it does *not* cover is
the half that is new here: a dataset that is an ``ImageFolder`` tree rather than a torchvision
archive, two input resolutions, and the ``output_group`` that keeps three datasets' results and
jobs apart.

Everything is offline. ImageNet is exercised on a synthetic 2-class tree written to ``tmp_path``,
which is enough to pin the parts that can silently go wrong — the train/val/test wiring, the split,
the class-mapping agreement between the two trees, and the two transform regimes — without the
155 GB. CIFAR-100 needs no fixture at all: it shares ``build_loaders`` with CIFAR-10 bit for bit.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.data import (  # noqa: E402
    CIFAR100_SPEC,
    IMAGENET_SPEC,
    build_imagenet_loaders,
    build_imagenet_transforms,
    imagenet_root,
)
from benchmarks.common.runner import BENCHMARKS_DIR, discover_benchmarks  # noqa: E402

BENCHES = discover_benchmarks()
CIFAR100_MODELS = sorted(n for n in BENCHES if n.endswith("cifar100"))
IMAGENET_MODELS = sorted(n for n in BENCHES if n.endswith("imagenet"))
NATIVE_IMAGENET = ("resnet50_imagenet", "vit_small_imagenet")  # the two 224 px benches


# ----------------------------------------------------------------------------------------------
# A synthetic ImageNet tree
# ----------------------------------------------------------------------------------------------


def make_tree(root: Path, n_train_per_class: int = 6, n_val_per_class: int = 2,
              size: tuple[int, int] = (80, 60)) -> Path:
    """``<root>/imagenet/{train,val}/<wnid>/*.JPEG``, two classes, deliberately non-square images
    (80x60) so an aspect-ratio bug in either transform regime shows up as a shape error.
    """
    for split, count in (("train", n_train_per_class), ("val", n_val_per_class)):
        for wnid in ("n01440764", "n01443537"):
            directory = root / "imagenet" / split / wnid
            directory.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                Image.new("RGB", size, color=(i * 7 % 256, 128, 255 - i)).save(
                    directory / f"{wnid}_{i}.JPEG"
                )
    return root


@pytest.fixture(scope="module")
def imagenet_tree(tmp_path_factory) -> Path:
    return make_tree(tmp_path_factory.mktemp("data"))


# ----------------------------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("img_size", [224, 32])
def test_both_transform_regimes_emit_the_declared_resolution(img_size: int) -> None:
    train_tf, eval_tf = build_imagenet_transforms(IMAGENET_SPEC, img_size, cutout=True,
                                                  cutout_length=16)
    image = Image.new("RGB", (80, 60), color=(10, 20, 30))
    for transform in (train_tf, eval_tf):
        out = transform(image)
        assert out.shape == (3, img_size, img_size), f"{img_size}px: {tuple(out.shape)}"
        assert out.dtype == torch.float32 and torch.isfinite(out).all()


def test_downsampled_regime_squashes_the_whole_image_not_a_crop() -> None:
    """Chrabaszcz et al. 2017 §2 downsample the *whole* image; the 224 px regime crops instead.
    A vertical split of colour is preserved by a squash and may be cropped away by the other.
    """
    image = Image.new("RGB", (64, 64))
    image.paste((255, 0, 0), (0, 0, 64, 32))  # top half red, bottom half black
    train_tf, eval_tf = build_imagenet_transforms(IMAGENET_SPEC, 32, cutout=False,
                                                  cutout_length=16)
    out = eval_tf(image)  # eval: squash only, no crop and no flip
    assert out[0, :16, :].mean() > out[0, 16:, :].mean(), "the top half must stay the red one"


def test_pre_resized_tree_is_numerically_a_no_op() -> None:
    """Why ``stage_imagenet.sh resize`` is safe: ``Resize((s, s))`` applied to an image that is
    already ``s x s`` returns it unchanged, so the pre-resized tree and the on-the-fly pipeline
    produce identical tensors — the staging script only moves the cost off the training loop.
    """
    from torchvision import transforms

    full = Image.new("RGB", (80, 60))
    full.paste((200, 30, 60), (10, 10, 50, 40))
    squash = transforms.Resize((32, 32))
    once, twice = squash(full), squash(squash(full))
    assert torch.equal(transforms.ToTensor()(once), transforms.ToTensor()(twice))


# ----------------------------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------------------------


def small_spec(val_size: int = 4):
    """``IMAGENET_SPEC`` with a val split the fixture tree can actually supply."""
    return replace(IMAGENET_SPEC, val_size=val_size)


@pytest.mark.parametrize("img_size", [224, 32])
def test_loaders_wire_train_val_test(imagenet_tree: Path, img_size: int) -> None:
    train, val, test = build_imagenet_loaders(
        small_spec(), imagenet_tree, img_size=img_size, batch_size=2, num_workers=0,
    )
    # 2 classes x 6 train images = 12, minus the 4-image val split.
    assert len(train.dataset) == 8 and len(val.dataset) == 4
    assert len(test.dataset) == 4  # 2 classes x 2 official-val images
    inputs, targets = next(iter(train))
    assert inputs.shape == (2, 3, img_size, img_size)
    assert targets.dtype == torch.int64 and int(targets.max()) < 2


def test_split_is_seeded_disjoint_and_exhaustive(imagenet_tree: Path) -> None:
    train, val, _ = build_imagenet_loaders(small_spec(), imagenet_tree, img_size=32, seed=7,
                                           num_workers=0)
    train_idx, val_idx = set(train.dataset.indices), set(val.dataset.indices)
    assert not (train_idx & val_idx)
    assert train_idx | val_idx == set(range(12))
    again, _, _ = build_imagenet_loaders(small_spec(), imagenet_tree, img_size=32, seed=7,
                                         num_workers=0)
    assert again.dataset.indices == train.dataset.indices
    other, _, _ = build_imagenet_loaders(small_spec(), imagenet_tree, img_size=32, seed=8,
                                         num_workers=0)
    assert other.dataset.indices != train.dataset.indices


def test_val_stream_is_not_augmented(imagenet_tree: Path) -> None:
    """The validation split comes from the *training* directory, so the one thing that must not
    leak across is the train transform (random crop, flip, Cutout) — two passes must agree.
    """
    _, val, _ = build_imagenet_loaders(small_spec(), imagenet_tree, img_size=32, num_workers=0)
    first = torch.cat([x for x, _ in val])
    second = torch.cat([x for x, _ in val])
    assert torch.equal(first, second)


def test_missing_tree_names_the_layout(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="train/n01440764"):
        build_imagenet_loaders(small_spec(), tmp_path, img_size=32, num_workers=0)


def test_class_mapping_disagreement_is_an_error(tmp_path: Path) -> None:
    """A val tree whose directory names differ from the train tree's would silently train against
    permuted labels. ILSVRC's own val tar is flat, so this is the realistic staging mistake.
    """
    make_tree(tmp_path)
    wnid = tmp_path / "imagenet" / "val" / "n01443537"
    wnid.rename(wnid.with_name("n99999999"))
    with pytest.raises(ValueError, match="class -> index mapping"):
        build_imagenet_loaders(small_spec(), tmp_path, img_size=32, num_workers=0)


def test_pre_resized_tree_is_preferred_only_for_the_downsampled_benches(tmp_path: Path) -> None:
    make_tree(tmp_path)
    (tmp_path / "imagenet32" / "train").mkdir(parents=True)
    assert imagenet_root(tmp_path, 32).name == "imagenet32"
    assert imagenet_root(tmp_path, 224).name == "imagenet"
    # Absent, the 32 px benches fall back to the full-resolution tree rather than failing.
    assert imagenet_root(tmp_path / "nowhere", 32).name == "imagenet"


# ----------------------------------------------------------------------------------------------
# The benches
# ----------------------------------------------------------------------------------------------


def test_the_expected_twelve_benches_exist() -> None:
    architectures = {"cnn_gn", "vit_micro", "resnet20", "cct_2_3x2", "resnet50", "vit_small"}
    assert {n[: -len("_cifar100")] for n in CIFAR100_MODELS} == architectures
    assert {n[: -len("_imagenet")] for n in IMAGENET_MODELS} == architectures


@pytest.mark.parametrize("name", CIFAR100_MODELS + IMAGENET_MODELS)
def test_output_group_routes_results_into_a_dataset_subdirectory(name: str) -> None:
    bench = BENCHES[name]
    expected = "cifar100" if name.endswith("cifar100") else "imagenet"
    assert bench.output_group == expected
    assert bench.output_root(BENCHMARKS_DIR / "outputs") == (
        BENCHMARKS_DIR / "outputs" / expected / name
    )


#: Every bench is grouped by the dataset it trains on, so ``outputs/`` and ``slurm/`` have one
#: subdirectory per dataset and none of the twenty models sits at the top level.
EXPECTED_GROUPS = {
    "mnist_autoencoder": "mnist", "mlp_ln_mnist": "mnist",
    "cnn_gn_cifar": "cifar10", "vit_micro_cifar": "cifar10", "resnet20_cifar": "cifar10",
    "cct_2_3x2_cifar": "cifar10", "resnet50_cifar": "cifar10", "vit_small_cifar": "cifar10",
}


def test_every_bench_declares_the_group_of_its_dataset() -> None:
    """A bench's ``output_group`` must match the dataset its ``build_data`` is bound to — that is
    the whole invariant behind ``outputs/<group>/<model>/`` and ``slurm/<group>/``.
    """
    for name, bench in BENCHES.items():
        assert bench.output_group, f"{name} declares no output_group"
        assert bench.output_group == bench.build_data.args[0].name, name
    for name, group in EXPECTED_GROUPS.items():
        assert BENCHES[name].output_group == group


def test_the_checkpoint_bridge_follows_the_grouped_layout() -> None:
    """``fisher_ref/checkpoints.py`` resolves runs at ``outputs/<group>/<model>/<arm>/`` as well as
    the flat layout, so grouping the eight original models does not hide their result trees
    (``tests/test_fisher_ref_lot0.py`` pins the discovery itself).
    """
    from fisher_ref.checkpoints import _output_groups

    assert set(EXPECTED_GROUPS.values()) | {"cifar100", "imagenet"} == set(_output_groups())


@pytest.mark.parametrize("name", CIFAR100_MODELS + IMAGENET_MODELS)
def test_head_width_matches_the_dataset(name: str) -> None:
    bench = BENCHES[name]
    expected = 100 if name.endswith("cifar100") else 1000
    model = bench.build_model(**{k: v[0] for k, v in bench.model_choices.items()})
    size = 224 if name in NATIVE_IMAGENET else 32
    with torch.no_grad():
        assert model(torch.randn(2, 3, size, size)).shape == (2, expected)


@pytest.mark.parametrize("name", CIFAR100_MODELS + IMAGENET_MODELS)
def test_bench_points_at_the_right_dataset_and_resolution(name: str) -> None:
    bench = BENCHES[name]
    spec = bench.build_data.args[0]  # generate_jobs.dataset_of reads exactly this
    if name.endswith("cifar100"):
        assert spec is CIFAR100_SPEC
        return
    assert spec is IMAGENET_SPEC
    expected = 224 if name in NATIVE_IMAGENET else 32
    assert bench.build_data.keywords["img_size"] == expected


@pytest.mark.parametrize("name", ["cnn_gn_imagenet", "vit_micro_imagenet"])
def test_a_bench_trains_end_to_end_on_the_synthetic_tree(name: str, imagenet_tree: Path) -> None:
    """The one check that the bench, the loader and ``AdaFisherMulti`` fit together: two real
    ``ekfac`` steps on real batches out of ``build_imagenet_loaders``, at the bench's own
    resolution and head width.
    """
    from adafisher_modes import AdaFisherMulti

    bench = BENCHES[name]
    torch.manual_seed(0)
    model = bench.build_model(**{k: v[0] for k, v in bench.model_choices.items()})
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = AdaFisherMulti(model, lr=1e-3, TCov=1, T_eig=1, fisher_mode="ekfac")
    train, _, _ = build_imagenet_loaders(
        small_spec(), imagenet_tree, img_size=bench.build_data.keywords["img_size"],
        batch_size=4, num_workers=0,
    )
    for batch in list(train)[:2]:
        inputs, targets = bench.prepare_batch(batch)
        optimizer.zero_grad()
        bench.loss_fn(model(inputs), targets).backward()
        optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    unchanged = [n for n, p in model.named_parameters() if torch.equal(p.detach(), before[n])]
    assert unchanged == [], f"{name}: {unchanged}"


# ----------------------------------------------------------------------------------------------
# SLURM
# ----------------------------------------------------------------------------------------------


def test_every_bench_has_generated_jobs_in_its_own_subdirectory() -> None:
    """The generator enumerates model folders, so a new bench gets jobs for free — but only if it
    has a ``WALLTIME`` row. Without one ``generate_jobs.py`` raises ``KeyError`` mid-run, after
    having already overwritten half the directory.
    """
    sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "slurm"))
    import generate_jobs

    slurm = REPO_ROOT / "benchmarks" / "slurm"
    for name, bench in BENCHES.items():
        assert name in generate_jobs.WALLTIME, f"{name} has no --time row in generate_jobs.py"
        directory = slurm / bench.output_group if bench.output_group else slurm
        for arm in bench.arms:
            assert (directory / f"train_{name}_{arm}.sh").is_file()
        assert (directory / f"calibrate_{name}.sh").is_file()


def test_generated_jobs_write_into_the_dataset_subdirectory() -> None:
    for name in CIFAR100_MODELS + IMAGENET_MODELS:
        group = BENCHES[name].output_group
        text = (REPO_ROOT / "benchmarks" / "slurm" / group / f"train_{name}_diag.sh").read_text()
        assert f"benchmarks/outputs/{group}/{name}/diag" in text
        assert f"python -m benchmarks.{name}.bench" in text
        if group == "imagenet":
            assert "IMAGENET_ARCHIVE" in text, "an ImageNet job must stage to $SLURM_TMPDIR"


# ----------------------------------------------------------------------------------------------
# The Fisher-drift campaign's bridge
# ----------------------------------------------------------------------------------------------


def test_grouped_and_flat_result_trees_both_resolve(tmp_path: Path) -> None:
    """The migration is safe in both directions: a result tree already written flat keeps being
    found after its bench gains an ``output_group``, and one written under the group is found too.
    Nothing has to move on the same day the benches are grouped.
    """
    from fisher_ref import checkpoints as bridge

    payload = {"model_state_dict": {}, "step": 0, "epoch": 0, "seed": 0, "fraction": 0.0,
               "total_steps": 10, "scheduled": True}
    for directory in (tmp_path / "cifar10" / "resnet20_cifar" / "diag",   # grouped
                      tmp_path / "vit_micro_cifar" / "diag"):             # still flat
        directory.mkdir(parents=True)
        torch.save(payload, directory / "ckpt_0.pt")

    assert {(r.model, r.arm) for r in bridge.discover_runs(tmp_path)} == {
        ("resnet20_cifar", "diag"), ("vit_micro_cifar", "diag")
    }
