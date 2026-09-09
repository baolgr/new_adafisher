"""8-layer MNIST auto-encoder bench (docs/reports/plan.md §6.3, lot 1 scope).

Encoder 784-1000-500-250-30 with sigmoid activations, symmetric decoder with untied weights
(``ekfac_1806.03884.pdf`` §4.1) — the historical K-FAC / EKFAC bench. Lot 1 only exercises
``fisher_mode="diag"``, against the reference ``AdaFisher`` optimizer
(``reference_repos/FisherAdapTune/scripts/adafisher.py``), to give a qualitative check of exit
criterion 3: with ``--minmax-normalization=false`` the two training curves should be visually
superimposed (a direct consequence of the bit-exact ``f_tilde`` construction already checked by
``tests/test_diag_bitexact.py``); with ``--minmax-normalization=true`` the curve should differ but
still train stably.

No YAML config in lot 1 (nothing here needs it yet, docs/reports/plan_lot1.md §2/§6); a plain
argparse CLI is enough.

Lot 7 (docs/reports/plan_lot7.md §1.1) generalized ``_make_optimizer`` to build any of the five
``AdaFisherMulti`` modes (``--optimizer kfac|ekfac|tkfac|tekfac|all``), for a quick, epoch-fixed
sanity comparison across modes; the equal-*wall-clock*-budget comparison itself (§6.3's actual exit
criterion) is ``benchmarks/equal_wallclock_bench.py``, which imports ``AutoEncoder``,
``_mnist_loader`` and ``_make_optimizer`` from this file rather than duplicating them.

Usage (from the repository root, with the project venv):
    PYTHONPATH=src .venv/bin/python benchmarks/mnist_autoencoder.py --optimizer both --epochs 5
    PYTHONPATH=src .venv/bin/python benchmarks/mnist_autoencoder.py --optimizer all --epochs 5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adafisher_modes import AdaFisherMulti  # noqa: E402


def _load_reference_adafisher():
    path = REPO_ROOT / "reference_repos/FisherAdapTune/scripts/adafisher.py"
    spec = importlib.util.spec_from_file_location("_ref_fisheradaptune_adafisher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENCODER_DIMS = [784, 1000, 500, 250, 30]


class AutoEncoder(nn.Module):
    """8 Linear layers, sigmoid activations, untied symmetric decoder."""

    def __init__(self, dims: List[int] = ENCODER_DIMS) -> None:
        super().__init__()
        encoder_layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        decoder_dims = list(reversed(dims))
        decoder_layers = [nn.Linear(decoder_dims[i], decoder_dims[i + 1]) for i in range(len(decoder_dims) - 1)]
        self.layers = nn.ModuleList(encoder_layers + decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = torch.sigmoid(layer(x))
        return self.layers[-1](x)  # linear output layer, reconstructs raw pixel intensities


def _mnist_loader(batch_size: int, train: bool) -> DataLoader:
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root=str(REPO_ROOT / "benchmarks/data"), train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)


# Per-mode extra kwargs (lot 7, docs/reports/plan_lot7.md §1.1): T_inv for kfac/tkfac, T_eig for
# ekfac/tekfac, T_re for tekfac only — mirrors tests/test_conv2d_optimizer_smoke.py's
# ``_MODE_KWARGS`` convention. Read from the CLI so the defaults (100/100/1, CLAUDE.md's "Selecting
# a mode") are explicit and overridable, without changing --optimizer reference|diag|both's
# existing behaviour (these flags are inert unless one of these four modes is selected).
_NEW_MODE_KWARGS = {
    "kfac": lambda a: {"T_inv": a.t_inv},
    "ekfac": lambda a: {"T_eig": a.t_eig},
    "tkfac": lambda a: {"T_inv": a.t_inv},
    "tekfac": lambda a: {"T_eig": a.t_eig, "T_re": a.t_re},
}


def _make_optimizer(name: str, model: nn.Module, args: argparse.Namespace, reference_module):
    if name == "reference":
        return reference_module.AdaFisher(
            model, lr=args.lr, beta=args.beta, Lambda=args.lam, TCov=args.tcov,
            gammas=list(args.gammas),
        )
    if name == "diag":
        return AdaFisherMulti(
            model, lr=args.lr, beta=args.beta, Lambda=args.lam, TCov=args.tcov,
            gammas=list(args.gammas), fisher_mode="diag",
            minmax_normalization=args.minmax_normalization,
        )
    if name in _NEW_MODE_KWARGS:
        return AdaFisherMulti(
            model, lr=args.lr, beta=args.beta, Lambda=args.lam, TCov=args.tcov,
            gammas=list(args.gammas), fisher_mode=name, **_NEW_MODE_KWARGS[name](args),
        )
    raise ValueError(name)


def _train_one(name: str, args: argparse.Namespace, reference_module) -> List[float]:
    torch.manual_seed(args.seed)
    model = AutoEncoder()
    optimizer = _make_optimizer(name, model, args, reference_module)
    loss_fn = nn.MSELoss()
    loader = _mnist_loader(args.batch_size, train=True)

    epoch_losses = []
    for epoch in range(args.epochs):
        total_loss, n_batches = 0.0, 0
        t0 = time.time()
        for images, _ in loader:
            x = images.view(images.size(0), -1)
            optimizer.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        mean_loss = total_loss / n_batches
        epoch_losses.append(mean_loss)
        print(f"[{name}] epoch {epoch + 1}/{args.epochs}  loss={mean_loss:.6f}  ({time.time() - t0:.1f}s)")
    return epoch_losses


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimizer",
        choices=["reference", "diag", "kfac", "ekfac", "tkfac", "tekfac", "both", "all"],
        default="both",
        help='"both" = reference + diag (lot 1); "all" = the five AdaFisherMulti modes, '
        "excluding reference (docs/reports/plan_lot7.md §0.10).",
    )
    parser.add_argument("--minmax-normalization", dest="minmax_normalization", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--lam", type=float, default=1e-3)
    parser.add_argument("--tcov", type=int, default=100)
    parser.add_argument("--gammas", type=float, nargs=2, default=(0.92, 0.008))
    parser.add_argument("--t-inv", dest="t_inv", type=int, default=100)
    parser.add_argument("--t-eig", dest="t_eig", type=int, default=100)
    parser.add_argument("--t-re", dest="t_re", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    reference_module = _load_reference_adafisher()
    if args.optimizer == "both":
        names = ["reference", "diag"]
    elif args.optimizer == "all":
        names = ["diag", "kfac", "ekfac", "tkfac", "tekfac"]
    else:
        names = [args.optimizer]

    results = {name: _train_one(name, args, reference_module) for name in names}

    if len(results) > 1:
        print("\nepoch  " + "  ".join(f"{name:>12s}" for name in results))
        for epoch in range(args.epochs):
            row = "  ".join(f"{results[name][epoch]:12.6f}" for name in results)
            print(f"{epoch + 1:5d}  {row}")


if __name__ == "__main__":
    main()
