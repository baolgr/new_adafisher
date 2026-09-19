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
from benchmarks.common.optimizers import ARMS  # noqa: E402
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


def _gradient_image(width: int = 80, height: int = 60) -> Image.Image:
    """A smooth image, which is where JPEG's own error is easiest to see and to bound."""
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x * 3 % 256, y * 4 % 256, (x + y) % 256)
    return image


def test_pre_resized_tree_has_the_same_geometry_as_the_on_the_fly_pipeline() -> None:
    """What ``imagenet_root``'s fallback rests on: at ``img_size = 32`` this project's *own* eval
    transform applied to a full-resolution image equals the same transform applied to an image
    already squashed to 32x32. ``Resize((32, 32))`` on a 32x32 input is the identity, so a bench
    reading the pre-resized tree and one reading the full tree see the same geometry.

    This says nothing about pixels; see the test below, which is the half that used to be claimed
    and not checked.
    """
    from torchvision import transforms

    full = _gradient_image()
    _train_tf, eval_tf = build_imagenet_transforms(IMAGENET_SPEC, 32, cutout=False,
                                                   cutout_length=16)
    pre_resized = transforms.Resize((32, 32))(full)
    assert pre_resized.size == (32, 32)
    assert torch.equal(eval_tf(full), eval_tf(pre_resized))
    # The idempotence this rests on, stated on its own.
    squash = transforms.Resize((32, 32))
    assert torch.equal(transforms.ToTensor()(squash(full)),
                       transforms.ToTensor()(squash(squash(full))))


def test_the_staged_tree_is_not_pixel_identical_because_it_is_re_encoded() -> None:
    """``stage_imagenet.sh resize`` writes JPEGs, so the staged tree is *not* bit-equivalent to
    the on-the-fly pipeline — only geometrically equivalent. What it buys is cost: 1.28 M 32x32
    decodes per epoch instead of 1.28 M full-resolution ones.

    Measured here on a smooth gradient, at the quality 95 the staging script uses: 63.5% of the
    pixels move, by 0.84/255 on average and 5/255 at worst. Bounded rather than asserted equal,
    which is what ``benchmarks/common/data.py::imagenet_root`` already documents.
    """
    import io

    from torchvision import transforms

    pre_resized = transforms.Resize((32, 32))(_gradient_image())
    buffer = io.BytesIO()
    pre_resized.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    staged = Image.open(buffer).convert("RGB")

    to_tensor = transforms.ToTensor()
    original, reloaded = to_tensor(pre_resized), to_tensor(staged)
    delta = (original - reloaded).abs()
    assert not torch.equal(original, reloaded), "a JPEG round trip is not lossless"
    assert float((delta > 0).float().mean()) > 0.2, "the fixture no longer shows the re-encoding"
    assert float(delta.max()) * 255 < 16, "but the error stays within a few levels out of 255"
    assert float(delta.mean()) * 255 < 2


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
    """A val tree whose directory names differ from the train tree's would silently evaluate
    against permuted labels. This is the *renamed*-directory case: the tree has class folders, but
    one of them carries a WordNet id the train tree does not know.

    The flat-tar case the staging README warns about is a different failure and is covered by the
    test below — it never reaches this check.
    """
    make_tree(tmp_path)
    wnid = tmp_path / "imagenet" / "val" / "n01443537"
    wnid.rename(wnid.with_name("n99999999"))
    with pytest.raises(ValueError, match="class -> index mapping"):
        build_imagenet_loaders(small_spec(), tmp_path, img_size=32, num_workers=0)


def test_a_flat_val_tree_fails_before_the_class_mapping_check(tmp_path: Path) -> None:
    """ILSVRC's own ``ILSVRC2012_img_val.tar`` expands **flat**: 50 000 JPEGs with no class
    directories at all. ``stage_imagenet.sh`` runs the standard ``valprep`` step to sort them, and
    forgetting it is the realistic staging mistake.

    What happens then is not this project's ``ValueError`` — ``ImageFolder`` never gets as far as
    building a class map. It raises ``FileNotFoundError`` naming the directory, which is a clear
    enough message to act on; the point of pinning it is that a reader looking for the flat-tar
    case knows which error to expect and does not assume the ``ValueError`` covers it.
    """
    make_tree(tmp_path)
    val = tmp_path / "imagenet" / "val"
    for class_dir in list(val.iterdir()):
        for image in class_dir.iterdir():
            image.rename(val / image.name)
        class_dir.rmdir()
    assert all(p.is_file() for p in val.iterdir()), "the fixture must be a genuinely flat tree"

    with pytest.raises(FileNotFoundError, match="[Cc]ould.?n.t find any class folder"):
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


def _generate_jobs():
    sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "slurm"))
    import generate_jobs

    return generate_jobs


def test_every_bench_has_generated_jobs_in_its_own_subdirectory() -> None:
    """The generator enumerates model folders, so a new bench gets jobs for free — but only if it
    has a ``WALLTIME`` row. Without one ``generate_jobs.py`` raises ``KeyError`` mid-run, after
    having already overwritten half the directory.
    """
    generate_jobs = _generate_jobs()

    slurm = REPO_ROOT / "benchmarks" / "slurm"
    for name, bench in BENCHES.items():
        assert name in generate_jobs.WALLTIME, f"{name} has no --time row in generate_jobs.py"
        directory = slurm / bench.output_group if bench.output_group else slurm
        for arm in bench.arms:
            assert (directory / f"train_{name}_{arm}.sh").is_file()
        assert (directory / f"calibrate_{name}.sh").is_file()


def test_every_generate_jobs_table_names_only_real_benches() -> None:
    """The reverse direction of the check above, for every table in the generator.

    ``WALLTIME`` was checked one way only — every bench has a row — so a row left behind by a
    deleted model went unnoticed. The same applies to ``GROUPED``, ``SWEEPS`` and ``V0_SEEDS``,
    whose stale rows would generate jobs for a folder that no longer exists.
    """
    generate_jobs = _generate_jobs()

    assert set(generate_jobs.WALLTIME) == set(BENCHES), (
        "WALLTIME and the model folders must agree in both directions; the difference is "
        f"{set(generate_jobs.WALLTIME) ^ set(BENCHES)}"
    )
    for table in ("GROUPED", "RESOURCES", "SWEEPS", "V0_SEEDS"):
        stale = set(getattr(generate_jobs, table)) - set(BENCHES)
        assert not stale, f"{table} has rows for models that no longer exist: {sorted(stale)}"
    assert set(generate_jobs.V0_SEED_ARMS) <= set(ARMS)
    assert generate_jobs.REFERENCE_ARM in ARMS


def test_every_imagenet_bench_declares_its_own_resources() -> None:
    """``RESOURCES`` is unchecked in either direction, and its *default* is the 10 GB MIG slice
    with 8 CPUs — which is silently wrong for ImageNet: 1.28 M JPEG decodes per epoch need cores,
    and ResNet-50 or ViT-S at 224 px need a whole H100. A new ``*_imagenet`` folder with no row
    would fall back to the small slice and simply run out of memory hours into the queue.
    """
    generate_jobs = _generate_jobs()

    for name in IMAGENET_MODELS:
        assert name in generate_jobs.RESOURCES, f"{name} has no RESOURCES row"
        gpus, cpus, mem = generate_jobs.RESOURCES[name]
        assert cpus > generate_jobs.DEFAULT_RESOURCES[1], f"{name}: {cpus} cpus is the default"
        assert gpus == ("h100:1" if name in NATIVE_IMAGENET else "h100_1g.10gb:1"), name
        assert mem.endswith("G") and int(mem[:-1]) >= 64, f"{name}: {mem}"
    # No non-ImageNet bench overrides the default profile: they all share the MIG slice.
    assert set(generate_jobs.RESOURCES) == set(IMAGENET_MODELS)
    # --num-workers follows --cpus-per-task, which is the whole point of raising it.
    for name in IMAGENET_MODELS:
        text = (REPO_ROOT / "benchmarks" / "slurm" / "imagenet"
                / f"train_{name}_diag.sh").read_text()
        cpus = generate_jobs.RESOURCES[name][1]
        assert f"--cpus-per-task={cpus}" in text and f"--num-workers {cpus}" in text, name


def test_generated_jobs_write_into_the_dataset_subdirectory() -> None:
    for name in CIFAR100_MODELS + IMAGENET_MODELS:
        group = BENCHES[name].output_group
        text = (REPO_ROOT / "benchmarks" / "slurm" / group / f"train_{name}_diag.sh").read_text()
        assert f"benchmarks/outputs/{group}/{name}/diag" in text
        assert f"python -m benchmarks.models.{name}.bench" in text
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
