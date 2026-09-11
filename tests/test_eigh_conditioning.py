"""``approximations/_eigh_utils.py``: the conditioning ridge that stopped ``ekfac``/``tekfac`` from
dying on an exactly rank-deficient factor.

Two real cluster runs died mid-training at ``ekfac.py``'s ``linalg.eigh``
(``benchmarks/slurm/logs/{mlp_ln_mnist,resnet20_cifar}_all-*.out``, ``_LinAlgError``: *"the input
matrix is ill-conditioned or has too many repeated eigenvalues"*). The measured cause, on
``mlp_ln_mnist``'s first ``Linear``: MNIST's border pixels are identically zero across the whole
dataset (130 of 784), so ``A`` is ``785 x 785`` of rank 646 with 136 exactly-zero eigenvalues.

**These tests cannot reproduce the crash itself** — it is cuSOLVER-specific, and LAPACK on CPU
solves the same matrices happily (verified by replaying the failing arm locally). What they *do*
pin down is the fix's contract: the ridge is mathematically inert on the returned basis, the basis
stays orthonormal on the exact structure that killed the runs, and the CPU fallback works when the
device solver raises.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from adafisher_modes import AdaFisherMulti
from adafisher_modes.approximations._eigh_utils import EIGH_RIDGE, eigenbasis
from conftest import seed_all


def _rank_deficient_factor(d: int = 64, n_zero: int = 20) -> torch.Tensor:
    """A bias-augmented second-moment matrix whose first ``n_zero`` input coordinates are
    identically zero — the exact structure of MNIST's border pixels.
    """
    seed_all(0)
    x = torch.randn(512, d)
    x[:, :n_zero] = 0.0
    h_bar = torch.cat([x, x.new_ones(x.size(0), 1)], dim=1)
    return h_bar.t() @ h_bar / h_bar.size(0)


def test_the_factor_under_test_really_is_rank_deficient() -> None:
    """Guard the premise before asserting anything about the fix."""
    A = _rank_deficient_factor()
    assert A.shape == (65, 65)
    assert int(torch.linalg.matrix_rank(A.double())) == 45  # 64 - 20 zero columns + 1 bias
    eigenvalues = torch.linalg.eigvalsh(A.double())
    assert int((eigenvalues.abs() < 1e-10).sum()) >= 20  # repeated zero eigenvalues


def test_eigenbasis_is_orthonormal_on_a_rank_deficient_factor() -> None:
    Q = eigenbasis(_rank_deficient_factor())
    assert torch.allclose(Q.t() @ Q, torch.eye(Q.size(0)), atol=1e-5)


def test_eigenbasis_diagonalises_the_raw_factor_not_just_the_conditioned_one() -> None:
    """The ridge's inertness, asserted rather than argued: ``M + cI`` shares every eigenvector with
    ``M``, so the returned basis must diagonalise ``M`` itself. This is what lets the fix claim it
    changes no EKFAC/TEKFAC semantics — only the solver's path.
    """
    A = _rank_deficient_factor()
    Q = eigenbasis(A)
    projected = Q.t() @ A @ Q
    off_diagonal = projected - torch.diag(projected.diagonal())
    assert off_diagonal.abs().max() < 1e-4 * A.diagonal().mean()


def test_eigenbasis_recovers_the_same_spectrum_as_plain_eigh() -> None:
    """On a well-conditioned factor, ``Q^T M Q``'s diagonal must be ``eigh(M)``'s eigenvalues —
    the ridge shifts the *conditioned* matrix's spectrum, never the raw factor's.
    """
    seed_all(1)
    X = torch.randn(96, 32)
    M = X.t() @ X / 96 + torch.eye(32)
    expected = torch.linalg.eigvalsh(M)
    got = (eigenbasis(M).t() @ M @ eigenbasis(M)).diagonal()
    assert torch.allclose(got, expected, atol=1e-4)


def test_ridge_is_relative_to_the_factor_scale() -> None:
    """A fixed absolute ridge would swamp a small-scale factor and be invisible on a large one."""
    M = _rank_deficient_factor()
    for scale in (1e-6, 1.0, 1e6):
        Q = eigenbasis(M * scale)
        assert torch.allclose(Q.t() @ Q, torch.eye(Q.size(0)), atol=1e-4), scale
    assert EIGH_RIDGE > 0.0


def test_eigenbasis_falls_back_to_cpu_when_the_device_solver_fails(monkeypatch) -> None:
    """The second line of defence: if the ridge is not enough, a slow refresh beats a dead job.
    ``linalg.eigh`` is made to raise exactly once, as cuSOLVER does.
    """
    from adafisher_modes.approximations import _eigh_utils

    real_eigh = torch.linalg.eigh
    calls = {"n": 0}

    def flaky(tensor):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.linalg.LinAlgError("linalg.eigh: The algorithm failed to converge")
        return real_eigh(tensor)

    monkeypatch.setattr(_eigh_utils.linalg, "eigh", flaky)
    A = _rank_deficient_factor()
    with pytest.warns(RuntimeWarning, match="retrying on CPU"):
        Q = eigenbasis(A)
    assert calls["n"] == 2
    assert torch.allclose(Q.t() @ Q, torch.eye(Q.size(0)), atol=1e-5)


class _ZeroBorderNet(nn.Module):
    """``mlp_ln_mnist`` in miniature: a first ``Linear`` fed coordinates that are always zero."""

    def __init__(self, d_in: int = 40, hidden: int = 8, num_classes: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.act = nn.ReLU(inplace=False)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.act(self.ln(self.fc1(x))))


@pytest.mark.parametrize("mode,kwargs", [("ekfac", {"T_eig": 1}), ("tekfac", {"T_eig": 1, "T_re": 1})])
def test_eigenbasis_modes_run_on_an_exactly_rank_deficient_input(mode: str, kwargs: dict) -> None:
    """End-to-end, through the real hooks and ``refresh`` cadence, on the input structure that
    killed the two cluster jobs. On CPU this passed before the fix too (LAPACK copes); it is here to
    document the scenario and to keep it exercised on a CUDA runner.
    """
    seed_all(0)
    model = _ZeroBorderNet()
    optimizer = AdaFisherMulti(model, lr=1e-2, TCov=1, fisher_mode=mode, **kwargs)
    x = torch.randn(16, 40)
    x[:, :15] = 0.0  # always-zero coordinates, as MNIST's border
    y = torch.randint(0, 3, (16,))
    for _ in range(4):
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
