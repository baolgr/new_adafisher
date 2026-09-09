"""Shared fixtures for the lot-1 test suite.

Reference implementations are loaded directly from their file path (bypassing the containing
package's ``__init__.py``, which pulls in unrelated modules such as ``trainer.py`` or
``AdaHessian.py`` with their own extra dependencies) so that only the exact reference class/function
under comparison is imported.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
FISHERADAPTUNE_ADAFISHER = REPO_ROOT / "reference_repos/FisherAdapTune/scripts/adafisher.py"
OFFICIAL_ADAFISHER = REPO_ROOT / "reference_repos/AdaFisher/optimizers/AdaFisher.py"


def _load_module_from_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def fisheradaptune_adafisher() -> ModuleType:
    """``reference_repos/FisherAdapTune/scripts/adafisher.py``, loaded standalone."""
    return _load_module_from_path("_ref_fisheradaptune_adafisher", FISHERADAPTUNE_ADAFISHER)


@pytest.fixture(scope="session")
def official_adafisher() -> ModuleType:
    """``reference_repos/AdaFisher/optimizers/AdaFisher.py``, loaded standalone.

    Its EMA has a known bug (docs/reports/plan.md §1.4) — only use this fixture for the parts that
    do not depend on the EMA, e.g. ``MinMaxNormalization`` / ``smart_detect_inf`` in isolation.
    """
    return _load_module_from_path("_ref_official_adafisher", OFFICIAL_ADAFISHER)


def seed_all(seed: int = 0) -> None:
    torch.manual_seed(seed)


class TinyMultiLayerNet(nn.Module):
    """Covers all four ``SUPPORTED_MODULES`` in one forward pass: Conv2d, BatchNorm2d, Linear,
    LayerNorm — the minimal net exercising every branch of ``factors.py``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 5 * 5, 8, bias=True)
        self.ln = nn.LayerNorm(8)
        self.fc2 = nn.Linear(8, 4, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(self.bn(self.conv(x)))
        x = self.flatten(x)
        x = torch.sigmoid(self.ln(self.fc1(x)))
        return self.fc2(x)


@pytest.fixture
def tiny_model_pair():
    """Two structurally- and weight-identical ``TinyMultiLayerNet`` instances, for side-by-side
    optimizer comparisons.
    """
    seed_all(0)
    model_a = TinyMultiLayerNet()
    model_b = TinyMultiLayerNet()
    model_b.load_state_dict(model_a.state_dict())
    return model_a, model_b


@pytest.fixture
def tiny_batch():
    seed_all(1)
    return torch.randn(6, 2, 5, 5)
