"""CIFAR-10 pipeline for lot 8's classification bench (docs/reports/plan_lot8.md §0.9).

Protocol, from the papers rather than from habit:

- **45k/5k train/val split** of the official 50k train set, seeded and reproducible; the official
  10k test set is held out entirely and evaluated once, at the end of a run. This is the ResNet
  paper's own CIFAR-10 protocol (``papers/resnet_1512.03385.pdf`` §4.2, p. 7: the schedule is
  "determined on a 45k/5k train/val split").
- **Train augmentation**: "4 pixels are padded on each side, and a 32x32 crop is randomly sampled
  from the padded image or its horizontal flip" (§4.2, verbatim), then per-channel normalization,
  then optionally **Cutout** (DeVries & Taylor, 2017) with 1 hole of 16 px — not in the ResNet
  paper, but set in every shipped AdaFisher config
  (``reference_repos/AdaFisher/Image_Classification/configs/*.yaml``: ``cutout: True, n_holes: 1,
  cutout_length: 16``), hence on by default and switchable (plan_lot8.md §0.9).
- **Eval**: normalization only — §4.2's "we only evaluate the single view of the original 32x32
  image".

``download`` is opt-in (``allow_download``): Alliance Canada's Rorqual compute nodes have no
internet, and torchvision's ``download=True`` there fails as an opaque network timeout instead of a
clear "dataset not staged" error (plan_lot8.md §0.9, §0.11).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


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


def build_transforms(cutout: bool = True, cutout_length: int = 16, n_holes: int = 1):
    train_ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    if cutout:
        train_ops.append(Cutout(n_holes=n_holes, length=cutout_length))
    eval_ops = [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


def build_dataloaders(
    data_root: str | Path,
    batch_size: int = 128,
    *,
    seed: int = 0,
    num_workers: int = 4,
    val_size: int = 5000,
    cutout: bool = True,
    cutout_length: int = 16,
    allow_download: bool = True,
    train_subset: int | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """``(train_loader, val_loader, test_loader)``. ``train_subset`` truncates the *training* split
    only (smoke runs, plan_lot8.md §3 step 1); validation and test stay full so a smoke run's
    metrics still mean what they say.
    """
    root = str(Path(data_root).expanduser())
    train_tf, eval_tf = build_transforms(cutout=cutout, cutout_length=cutout_length)

    train_full = CIFAR10(root=root, train=True, download=allow_download, transform=train_tf)
    val_full = CIFAR10(root=root, train=True, download=False, transform=eval_tf)
    test_set = CIFAR10(root=root, train=False, download=allow_download, transform=eval_tf)

    train_idx, val_idx = seeded_train_val_split(len(train_full), val_size, seed)
    if train_subset is not None:
        train_idx = train_idx[:train_subset]

    pin_memory = torch.cuda.is_available()
    common = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(
        Subset(train_full, train_idx), batch_size=batch_size, shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(
        Subset(val_full, val_idx), batch_size=batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader
