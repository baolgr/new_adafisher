"""The single data pipeline: MNIST, CIFAR-10, CIFAR-100 and ImageNet-1K.

Five ready-made builders, each returning ``(train_loader, val_loader, test_loader)`` from one
call: ``mnist``, ``cifar10``, ``cifar100``, ``imagenet`` (native 224 px) and ``imagenet32``
(ImageNet-1K downsampled to 32x32). A bench names one of them as its ``build_data``.

Protocol, from the papers rather than from habit
------------------------------------------------

- **A seeded train/val split** of the official training set — 45k/5k on CIFAR, 55k/5k on MNIST.
  The official test set is held out entirely and evaluated once, at the end of a run. This is the
  ResNet paper's own CIFAR protocol (``papers/resnet_1512.03385.pdf`` §4.2, p. 7: the schedule is
  "determined on a 45k/5k train/val split"). ``seeded_train_val_split`` draws one permutation from
  its own generator, so the split depends on the run's ``--seed`` and on nothing else.
- **CIFAR train augmentation**: "4 pixels are padded on each side, and a 32x32 crop is randomly
  sampled from the padded image or its horizontal flip" (§4.2, verbatim), then per-channel
  normalization, then optionally :class:`Cutout` (DeVries & Taylor, 2017) with one 16 px hole.
  Cutout is not in the ResNet paper; it is set in every shipped AdaFisher config
  (``reference_repos/AdaFisher/Image_Classification/configs/*.yaml``: ``cutout: True, n_holes: 1,
  cutout_length: 16``), hence on by default and switchable with ``--no-cutout``.
- **CIFAR eval**: normalization only — §4.2's "we only evaluate the single view of the original
  32x32 image". The validation loader uses this eval transform, never the train augmentation.
- **MNIST**: no augmentation at all, train or eval, the historical auto-encoder protocol of
  ``papers/ekfac_1806.03884.pdf`` §4.1. ``cutout`` is therefore structurally inert for MNIST
  rather than silently ignored.
- **CIFAR-100** shares CIFAR-10's protocol exactly — same 32x32 3-channel images, same split, same
  augmentation. Only the normalization constants and the label cardinality differ.
- **ImageNet-1K** is the one dataset that is not a torchvision archive: it is an ``ImageFolder``
  tree (``<root>/imagenet/{train,val}/<wnid>/*.JPEG``) that has to be staged by hand, because
  downloading it requires an accepted image-net.org agreement. Its 224 px transforms are
  AdaFisher's own (``reference_repos/AdaFisher/Image_Classification/src/utils/data.py:157-194``:
  ``RandomResizedCrop(224)`` + flip + ``Normalize(0.485/0.456/0.406, 0.229/0.224/0.225)`` +
  Cutout on train, ``Resize(256)`` + ``CenterCrop(224)`` on eval). The official 50k validation set
  plays the role the CIFAR test set plays — held out, evaluated once — and this project's own
  seeded split carves a *validation* stream out of the 1.28 M training images.
  ``build_imagenet_loaders`` documents the downsampled ``img_size=32`` variant and why it exists.

Downloading is opt-in (``allow_download``): compute nodes on the cluster this project runs on have
no internet, and torchvision's ``download=True`` there fails as an opaque network timeout instead
of a clear "dataset not staged" error.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder


class Cutout:
    """Zero out ``n_holes`` square regions of side ``length``, centered uniformly at random
    (DeVries & Taylor, 2017, "Improved regularization of convolutional neural networks with
    cutout"; cited by AdaFisher's own configs). Applied *after* normalization, so the masked value
    0 is the per-channel mean of the normalized distribution.
    """

    def __init__(self, n_holes: int = 1, length: int = 16) -> None:
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: Tensor) -> Tensor:
        h, w = img.size(1), img.size(2)
        mask = img.new_ones(h, w)
        for _ in range(self.n_holes):
            y = int(torch.randint(h, (1,)).item())
            x = int(torch.randint(w, (1,)).item())
            y0, y1 = max(0, y - self.length // 2), min(h, y + self.length // 2)
            x0, x1 = max(0, x - self.length // 2), min(w, x + self.length // 2)
            mask[y0:y1, x0:x1] = 0.0
        return img * mask.unsqueeze(0)


def seeded_train_val_split(n_total: int, n_val: int, seed: int) -> Tuple[List[int], List[int]]:
    """Reproducible split of ``range(n_total)`` into ``(train, val)`` index lists."""
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    return perm[: n_total - n_val], perm[n_total - n_val :]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_cls: Any
    mean: Sequence[float]
    std: Sequence[float]
    val_size: int
    augment: bool  # random crop + horizontal flip + (optional) Cutout on the train split


MNIST_SPEC = DatasetSpec("mnist", MNIST, (0.1307,), (0.3081,), val_size=5_000, augment=False)
CIFAR10_SPEC = DatasetSpec(
    "cifar10", CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 5_000, True
)
CIFAR100_SPEC = DatasetSpec(
    "cifar100", CIFAR100, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 5_000, True
)


def build_transforms(spec: DatasetSpec, cutout: bool, cutout_length: int, n_holes: int = 1):
    normalize = transforms.Normalize(spec.mean, spec.std)
    eval_ops = [transforms.ToTensor(), normalize]
    if not spec.augment:
        return transforms.Compose(eval_ops), transforms.Compose(eval_ops)
    train_ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    if cutout:
        train_ops.append(Cutout(n_holes=n_holes, length=cutout_length))
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def build_loaders(
    spec: DatasetSpec,
    data_root: str | Path,
    *,
    batch_size: int = 128,
    seed: int = 0,
    num_workers: int = 4,
    cutout: bool = True,
    cutout_length: int = 16,
    allow_download: bool = True,
    train_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """``(train_loader, val_loader, test_loader)``. ``train_subset`` truncates the *training* split
    only (smoke runs); validation and test stay full so a smoke run's metrics still mean what they
    say. ``cutout`` has no effect on a non-augmented spec (MNIST).
    """
    root = str(Path(data_root).expanduser())
    train_tf, eval_tf = build_transforms(spec, cutout, cutout_length)

    train_full = spec.dataset_cls(root=root, train=True, download=allow_download,
                                  transform=train_tf)
    val_full = spec.dataset_cls(root=root, train=True, download=False, transform=eval_tf)
    test_set = spec.dataset_cls(root=root, train=False, download=allow_download,
                                transform=eval_tf)

    train_idx, val_idx = seeded_train_val_split(len(train_full), spec.val_size, seed)
    if train_subset is not None:
        train_idx = train_idx[:train_subset]

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True,
                              drop_last=True, **common)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=batch_size, shuffle=False,
                            **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------------------------------
# ImageNet-1K
# ----------------------------------------------------------------------------------------------

#: ``ImageFolder``, not a torchvision archive: ILSVRC-2012 cannot be fetched without an accepted
#: image-net.org agreement, so ``allow_download`` is structurally inert here (a missing tree is a
#: ``FileNotFoundError`` naming the expected layout, never a silent download attempt).
#: ``val_size`` carves the *validation* stream out of the 1 281 167 training images — 25 000, i.e.
#: 25 per class on average, uniformly at random rather than class-stratified, which is the same
#: unstratified ``randperm`` the CIFAR/MNIST specs already use (``seeded_train_val_split``). The
#: official 50 000-image validation set is never trained or model-selected on: it is this
#: pipeline's *test* loader, evaluated once at the end of a run, exactly as the CIFAR test set is.
IMAGENET_SPEC = DatasetSpec(
    "imagenet", ImageFolder, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
    val_size=25_000, augment=True,
)


def build_imagenet_transforms(
    spec: DatasetSpec, img_size: int, cutout: bool, cutout_length: int, n_holes: int = 1
):
    """Two regimes, selected by ``img_size``, because this repository's models are two families.

    ``img_size >= 64`` — the **native** recipe, verbatim from AdaFisher's own pipeline
    (``reference_repos/AdaFisher/Image_Classification/src/utils/data.py:157-194``, itself the
    standard ILSVRC recipe of ``papers/resnet_1512.03385.pdf`` §3.4: "a 224x224 crop is randomly
    sampled from an image or its horizontal flip"): ``RandomResizedCrop(img_size)`` + flip on
    train, ``Resize(img_size * 256/224)`` + ``CenterCrop(img_size)`` on eval. Cutout is appended
    when requested — AdaFisher appends it on ImageNet too, at the same ``cutout_length: 16`` its
    CIFAR configs use (``configs/AdaFisherCNN.yaml``), so the length is not rescaled with the
    resolution here either.

    ``img_size < 64`` — **downsampled ImageNet**, the construction of Chrabaszcz, Loshchilov &
    Hutter 2017 (arXiv:1707.08819) §2: the *whole* image is resized to ``img_size x img_size``
    (aspect ratio not preserved, no crop), which is what makes a 32x32-native CIFAR architecture
    applicable to ImageNet-1K unmodified. This project then reuses its own CIFAR train
    augmentation on top (pad ``img_size/8`` + random crop + flip + optional Cutout), so that a
    ``<model>_imagenet`` bench differs from its ``<model>_cifar`` counterpart in exactly two
    things: the data and the width of the classification head. The pad-crop-flip choice is this
    project's, not that paper's.
    """
    normalize = transforms.Normalize(spec.mean, spec.std)
    if img_size >= 64:
        train_ops = [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
        eval_ops = [
            transforms.Resize(round(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        squash = transforms.Resize((img_size, img_size))
        train_ops = [
            squash,
            transforms.RandomCrop(img_size, padding=img_size // 8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
        eval_ops = [squash, transforms.ToTensor(), normalize]
    if cutout:
        train_ops.append(Cutout(n_holes=n_holes, length=cutout_length))
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def imagenet_root(data_root: str | Path, img_size: int = 224) -> Path:
    """``<data_root>/imagenet``, or ``<data_root>/imagenet<img_size>`` when a pre-resized tree of
    that resolution has been staged.

    The two apply the same geometry: ``transforms.Resize((s, s))`` on an image that is already
    ``s x s`` is the identity, and ``stage_imagenet.sh resize`` builds the tree with that very
    transform. They are *not* pixel-identical, because the staged tree is re-encoded as JPEG:
    measured on a synthetic smooth image, a quality-95 round trip moves about 61% of the pixels,
    by 0.8/255 on average and 5/255 at worst. What the staged tree buys is cost — 1.28 M 32x32
    JPEG decodes per epoch instead of 1.28 M full-resolution ones, for a model whose forward pass
    is a few hundred microseconds.
    """
    root = Path(data_root).expanduser()
    if img_size < 64 and (root / f"imagenet{img_size}" / "train").is_dir():
        return root / f"imagenet{img_size}"
    return root / "imagenet"


STAGING_HINT = """ImageNet-1K is not downloadable by torchvision (ILSVRC-2012 requires an accepted
image-net.org agreement). Expected layout, one directory per WordNet id:

    {root}/train/n01440764/*.JPEG   (1 281 167 images, 1000 classes)
    {root}/val/n01440764/*.JPEG     (50 000 images, the official validation set)

`benchmarks/slurm/imagenet/stage_imagenet.sh` builds exactly this tree from the two official
tars; the generated jobs beside it stage the result into $SLURM_TMPDIR."""


def build_imagenet_loaders(
    spec: DatasetSpec,
    data_root: str | Path,
    *,
    img_size: int = 224,
    batch_size: int = 128,
    seed: int = 0,
    num_workers: int = 4,
    cutout: bool = True,
    cutout_length: int = 16,
    allow_download: bool = True,
    train_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """``(train_loader, val_loader, test_loader)``, same contract as ``build_loaders``.

    ``allow_download`` is accepted and ignored — the runner passes it to every bench uniformly, and
    there is nothing to download (see ``STAGING_HINT``). A missing tree raises immediately.
    """
    root = imagenet_root(data_root, img_size)
    train_dir, test_dir = root / "train", root / "val"
    for directory in (train_dir, test_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} does not exist.\n\n" + STAGING_HINT.format(root=root)
            )

    train_tf, eval_tf = build_imagenet_transforms(spec, img_size, cutout, cutout_length)
    train_full = ImageFolder(str(train_dir), transform=train_tf)
    # The *same* scan, re-used with the eval transform. A second ``ImageFolder(train_dir, ...)``
    # would re-walk 1.28 M files for nothing; a shallow copy shares ``samples``/``targets`` and
    # owns its own ``transform`` attribute, which is the only thing that differs.
    val_full = copy.copy(train_full)
    val_full.transform = eval_tf
    test_set = ImageFolder(str(test_dir), transform=eval_tf)
    if train_full.class_to_idx != test_set.class_to_idx:
        raise ValueError(
            f"{train_dir} and {test_dir} disagree on the class -> index mapping; the validation "
            "tree must use the same WordNet-id directory names as the training tree"
        )

    train_idx, val_idx = seeded_train_val_split(len(train_full), spec.val_size, seed)
    if train_subset is not None:
        train_idx = train_idx[:train_subset]

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    # ``persistent_workers`` on the *training* loader only. Restarting 16 workers per epoch is pure
    # overhead when each has to re-fork a 1.28 M-entry ``samples`` list, but keeping three sets of
    # them alive at once would triple the copy-on-write cost of that same list for two loaders that
    # each run once an epoch — and that cost is real, since CPython's refcounting writes to the
    # pages it reads.
    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size, shuffle=True,
                              drop_last=True, persistent_workers=num_workers > 0, **common)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=batch_size, shuffle=False,
                            **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


mnist = partial(build_loaders, MNIST_SPEC)
cifar10 = partial(build_loaders, CIFAR10_SPEC)
cifar100 = partial(build_loaders, CIFAR100_SPEC)
#: ImageNet-1K at its native 224 px — for the two architectures actually designed for it
#: (``resnet50_imagenet``, ``vit_small_imagenet``).
imagenet = partial(build_imagenet_loaders, IMAGENET_SPEC, img_size=224)
#: ImageNet-1K downsampled to 32x32 (Chrabaszcz et al. 2017 §2) — for the four 32x32-native
#: CIFAR architectures, which are then structurally unchanged apart from a 1000-way head. Without
#: it, ``cct_2_3x2`` alone would emit a 3136-token sequence at 224 px and its attention map would
#: not fit any GPU at a usable batch size.
imagenet32 = partial(build_imagenet_loaders, IMAGENET_SPEC, img_size=32)
