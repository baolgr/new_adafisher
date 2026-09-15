"""Lot 2 of the Fisher-drift campaign: the approximation zoo and the metrics
(``docs/reports/plan_exp_lot2.md`` §2; ``plan_exp_draft.md`` §9's lot 2, §10.2's T3-T13).

Offline, CPU, fp64 throughout, as lot 1. The structures are compared against dense brute force on
tiny models, and against the exact blocks lot 1 already knows how to build.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.nn as nn
from conftest import seed_all
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fisher_ref import capture  # noqa: E402
from fisher_ref.approx import (  # noqa: E402
    BlockDiag,
    Dense,
    Diag,
    Kron,
    accumulate_ekfac,
    accumulate_factors,
    af_raw_from_factors,
    block_diagonal_of,
    cross_term_share,
    exact_separate,
    optimal_scale,
    rearrange,
)
from fisher_ref.reference import ParamLayout, build_dense_reference, to_augmented  # noqa: E402

DTYPE = torch.float64


# ----------------------------------------------------------------------------------------------
# The rearrangement, shared by M1 and M7 (plan_exp_lot2.md §0.4)
# ----------------------------------------------------------------------------------------------


def test_rearrangement_identities() -> None:
    """``<B, G(x)A> = vec(G)^T R(B) vec(A)`` and ``||B - G(x)A||_F = ||R(B) - vec(G)vec(A)^T||_F``.

    The second is what makes M7's ``sigma_2/sigma_1`` *the* independence bias: the best rank-1
    Kronecker fit of a block is the top singular pair of its rearrangement (Van Loan-Pitsianis).
    """
    seed_all(0)
    d_out, d_in = 3, 4
    B = torch.randn(d_out * d_in, d_out * d_in, dtype=DTYPE)
    A = torch.randn(d_in, d_in, dtype=DTYPE)
    G = torch.randn(d_out, d_out, dtype=DTYPE)

    rearranged = rearrange(B, d_out, d_in)
    assert rearranged.shape == (d_out * d_out, d_in * d_in)
    assert torch.allclose((B * torch.kron(G, A)).sum(),
                          G.reshape(-1) @ rearranged @ A.reshape(-1), atol=1e-12)
    assert torch.allclose(torch.linalg.matrix_norm(B - torch.kron(G, A)),
                          torch.linalg.matrix_norm(rearranged - torch.outer(G.reshape(-1),
                                                                           A.reshape(-1))),
                          atol=1e-12)


def test_rearrangement_top_singular_pair_is_the_best_kronecker_fit() -> None:
    """A block that *is* a Kronecker product has a rank-1 rearrangement: `sigma_2/sigma_1 = 0`."""
    seed_all(1)
    d_out, d_in = 3, 4
    A = torch.randn(d_in, d_in, dtype=DTYPE)
    G = torch.randn(d_out, d_out, dtype=DTYPE)
    singular = torch.linalg.svdvals(rearrange(torch.kron(G, A), d_out, d_in))
    assert float(singular[1] / singular[0]) < 1e-14


# ----------------------------------------------------------------------------------------------
# CurvatureBlock conformance
# ----------------------------------------------------------------------------------------------


def _blocks() -> list:
    seed_all(2)
    d_out, d_in = 3, 4
    A = torch.randn(d_in, d_in, dtype=DTYPE)
    A = A @ A.T + torch.eye(d_in, dtype=DTYPE)
    G = torch.randn(d_out, d_out, dtype=DTYPE)
    G = G @ G.T + torch.eye(d_out, dtype=DTYPE)
    dense = torch.randn(6, 6, dtype=DTYPE)
    dense = dense @ dense.T
    values = torch.rand(7, dtype=DTYPE) + 0.5
    kron = Kron(A=A, G=G)
    inner = Dense(dense[:3, :3])
    return [
        pytest.param(Dense(dense), id="Dense"),
        pytest.param(Diag(values), id="Diag"),
        pytest.param(kron, id="Kron"),
        pytest.param(BlockDiag([("a", slice(0, 3), inner),
                                ("b", slice(3, 6), Dense(dense[3:, 3:]))], P=6), id="BlockDiag"),
    ]


@pytest.mark.parametrize("block", _blocks())
def test_curvature_block_conformance(block) -> None:
    """Every accessor agrees with the dense matrix the block claims to represent — including the
    closed forms that exist precisely so the dense matrix is never built at scale.
    """
    seed_all(3)
    dense = block.to_dense()
    assert dense.shape == (block.P, block.P)

    v = torch.randn(block.P, dtype=DTYPE)
    assert torch.allclose(block.matvec(v), dense @ v, atol=1e-10)

    X = torch.randn(block.P, 3, dtype=DTYPE)
    assert torch.allclose(block.matmat(X), dense @ X, atol=1e-10)
    assert torch.allclose(block.apply_rows(X.T), X.T @ dense, atol=1e-10)

    assert torch.allclose(block.diag(), dense.diagonal(), atol=1e-10)
    assert torch.allclose(block.trace(), dense.diagonal().sum(), atol=1e-10)
    assert torch.allclose(block.fro2(), (dense * dense).sum(), atol=1e-10)

    R = torch.randn(block.P, block.P, dtype=DTYPE)
    R = R @ R.T
    assert torch.allclose(block.inner_dense(R), (R * dense).sum(), atol=1e-9)
    # The generic column sweep of BlockOps must agree with whatever closed form was substituted.
    from fisher_ref.approx.base import BlockOps  # noqa: PLC0415
    assert torch.allclose(BlockOps.inner_dense(block, R), (R * dense).sum(), atol=1e-9)

    lam = 0.7
    damped = dense + lam * torch.eye(block.P, dtype=DTYPE)
    assert torch.allclose(block.solve(v, lam), torch.linalg.solve(damped, v), atol=1e-9)
    assert torch.allclose(block.logdet(lam), torch.logdet(damped), atol=1e-9)


def test_kron_never_materialises_for_its_closed_forms() -> None:
    """`trace`, `fro2`, `diag`, `inner_dense` and `solve` on a `Kron` are closed forms; only
    `to_dense` builds the product. At A1's first layer that product is 5.05 GB.
    """
    seed_all(4)
    A = torch.rand(785, 785, dtype=DTYPE)  # A1's widest input factor, bias-augmented
    A = A @ A.T / 785
    G = torch.rand(32, 32, dtype=DTYPE)
    G = G @ G.T / 32
    kron = Kron(A=A, G=G)
    assert kron.P == 25120
    assert torch.allclose(kron.trace(), A.diagonal().sum() * G.diagonal().sum(), atol=1e-10)
    assert torch.allclose(kron.fro2(), (A * A).sum() * (G * G).sum(), atol=1e-8)
    assert kron.diag().numel() == 25120
    assert kron.eigenvalues().shape == (32, 785)


# ----------------------------------------------------------------------------------------------
# The named_parameters() <-> rvec([W | b]) permutation (plan_exp_lot1.md §0.5)
# ----------------------------------------------------------------------------------------------


def test_augmented_permutation_round_trips_a_known_layout() -> None:
    """The permutation is a relabelling, not a change of content: it must be a bijection, place the
    bias as the last column of every output row, and be the identity without a bias.
    """
    seed_all(13)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3, bias=False)).to(DTYPE)
    layout = ParamLayout.of(model)

    perm = layout.augmented_permutation("0")
    d_out, d_in = 5, 4
    assert perm.numel() == d_out * (d_in + 1)
    assert sorted(perm.tolist()) == list(range(d_out * (d_in + 1)))          # a bijection
    for row in range(d_out):
        assert perm[row * (d_in + 1) + d_in].item() == d_out * d_in + row    # the bias column
        assert perm[row * (d_in + 1)].item() == row * d_in                   # W's row start

    assert torch.equal(layout.augmented_permutation("2"), torch.arange(3 * 5))  # no bias


def test_augmented_permutation_refuses_a_normalisation_layer() -> None:
    """A ``(gamma, beta)`` block is Hadamard-structured and has no ``[W | b]`` reading at all
    (``plan_lot5.md`` §0.1), so the permutation must refuse rather than produce nonsense.
    """
    seed_all(14)
    model = nn.Sequential(nn.Linear(4, 5), nn.LayerNorm(5)).to(DTYPE)
    layout = ParamLayout.of(model)
    with pytest.raises(NotImplementedError, match="Hadamard"):
        layout.augmented_permutation("1")


# ----------------------------------------------------------------------------------------------
# T3 / T4 — KFAC from scratch's own two tests (plan_exp_draft.md §10.2)
# ----------------------------------------------------------------------------------------------


def _layer_factors(model: nn.Module, inputs: Tensor, targets: Tensor, source: str,
                   loss: str = "cross_entropy") -> dict:
    """``A``, ``G`` and the exact block, from one traversal — the minimal accumulator T3/T4 need."""
    modules = capture.capturable_modules(model)
    stats: dict = {}
    n_probes = int(inputs.shape[0])
    for step in capture.iter_probe_columns(model, inputs, targets, source=source, loss=loss,
                                           modules=modules, batch_size=n_probes, dtype=DTYPE):
        for layer in step.capturer:
            a = layer.a.reshape(-1, layer.a.shape[-1])
            g = layer.g.reshape(-1, layer.g.shape[-1])
            if layer.module.bias is not None:
                a = torch.cat([a, a.new_ones(a.shape[0], 1)], dim=1)
            entry = stats.setdefault(layer.name, {"A": 0.0, "G": 0.0, "T": layer.positions,
                                                  "seen_a": False})
            if not entry["seen_a"]:         # the input does not depend on the column
                entry["A"] = entry["A"] + a.T @ a
            entry["G"] = entry["G"] + g.T @ g
        for layer in step.capturer:
            stats[layer.name]["seen_a"] = True
    for entry in stats.values():
        entry["A"] = entry["A"] / (n_probes * entry["T"])
        entry["G"] = entry["G"] / n_probes
    return stats


def test_t3_kfac_is_exact_for_a_single_sample_without_sharing() -> None:
    """T3 — *KFAC from scratch* Test 1: with ``N = 1`` and no weight sharing, K-FAC is **exact**.

    The per-sample gradient is the single outer product ``g a^T``, so
    ``B = rvec(g a^T) rvec(g a^T)^T = (g g^T) (x) (a a^T) = G (x) A`` with the ``N = 1``
    normalisation — there is nothing for the independence assumption to be wrong about. Holds for
    the type-2 source against the GGN block and for the empirical source against the EF block.
    """
    seed_all(5)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3)).to(DTYPE)
    inputs = torch.randn(1, 4, dtype=DTYPE)
    targets = torch.randint(0, 3, (1,))

    for source in ("type2", "empirical"):
        reference = build_dense_reference(model, inputs, targets, source=source, batch_size=1)
        factors = _layer_factors(model, inputs, targets, source)
        for name, entry in factors.items():
            # The reference is in named_parameters() order, the Kronecker product in rvec([W | b]);
            # same entries, different order (plan_exp_lot1.md §0.5, and how this test first failed).
            perm = reference.layout.augmented_permutation(name)
            exact = to_augmented(reference.block(name), perm)
            kron = Kron(A=entry["A"], G=entry["G"])
            assert torch.allclose(kron.to_dense(), exact, atol=1e-12), f"{source}/{name}"


def test_t4_kfac_type2_is_the_ggn_but_kfac_empirical_is_not_the_ef() -> None:
    """T4 — its Test 2: on a **deep linear** network with MSE, K-FAC-type-2 reproduces the GGN block
    exactly, while K-FAC-**empirical** does *not* reproduce the EF block.

    With `Lambda = (2/D) I` constant, the type-2 backprop vectors span the output space uniformly
    and the per-column outer products factorise; the empirical source has a single, data-dependent
    vector per sample, so ``E[g a^T (g a^T)^T]`` does not split — which is the whole point of the
    test. The gap is what HF-level readings of an empirical K-FAC are measuring.
    """
    seed_all(6)
    model = nn.Sequential(nn.Linear(4, 6, bias=False), nn.Linear(6, 3, bias=False)).to(DTYPE)
    inputs = torch.randn(16, 4, dtype=DTYPE)
    targets = torch.randn(16, 3, dtype=DTYPE)

    exact_type2 = build_dense_reference(model, inputs, targets, source="type2", loss="mse",
                                        batch_size=16)
    kfac_type2 = _layer_factors(model, inputs, targets, "type2", loss="mse")
    for name, entry in kfac_type2.items():
        perm = exact_type2.layout.augmented_permutation(name)
        block = to_augmented(exact_type2.block(name), perm)
        kron = Kron(A=entry["A"], G=entry["G"])
        assert torch.allclose(kron.to_dense(), block, atol=1e-12), name

    exact_emp = build_dense_reference(model, inputs, targets, source="empirical", loss="mse",
                                      batch_size=16)
    kfac_emp = _layer_factors(model, inputs, targets, "empirical", loss="mse")
    gaps = []
    for name, entry in kfac_emp.items():
        perm = exact_emp.layout.augmented_permutation(name)
        block = to_augmented(exact_emp.block(name), perm)
        kron = Kron(A=entry["A"], G=entry["G"])
        gaps.append(float(torch.linalg.matrix_norm(kron.to_dense() - block)
                          / torch.linalg.matrix_norm(block)))
    assert max(gaps) > 1e-3, f"K-FAC-empirical should NOT equal the EF block; gaps {gaps}"


# ----------------------------------------------------------------------------------------------
# HF3 — AdaFisher's raw estimator is K-FAC's diagonal
# ----------------------------------------------------------------------------------------------


def test_hf3_af_raw_is_exactly_the_kfac_diagonal() -> None:
    """``diag(A) (x) diag(G) = diag(A (x) G)``: AdaFisher's raw estimator **is** K-FAC's diagonal
    (``plan_exp_draft.md`` §3.1). So HF3's question is not "is it a different structure" but "how
    far is that diagonal from the exact one", which is the pair of paths §3.1 describes.
    """
    seed_all(7)
    A = torch.rand(5, 5, dtype=DTYPE)
    A = A @ A.T
    G = torch.rand(3, 3, dtype=DTYPE)
    G = G @ G.T
    kron = Kron(A=A, G=G)
    assert torch.allclose(af_raw_from_factors(A, G).values, kron.diag(), atol=1e-12)
    assert torch.allclose(kron.diag(), kron.to_dense().diagonal(), atol=1e-12)


def test_hf3_diagonal_bias_is_the_covariance_of_squares() -> None:
    """Without weight sharing the entry-wise gap between the exact diagonal and
    ``diag(A) (x) diag(G)`` is ``Cov(a_j^2, g_i^2)`` — a *measurement*, not an estimate
    (``plan_exp_draft.md`` §3.1).
    """
    seed_all(8)
    n, d_in, d_out = 200, 4, 3
    a = torch.randn(n, d_in, dtype=DTYPE)
    g = torch.randn(n, d_out, dtype=DTYPE) * a[:, :1]      # deliberately correlated
    exact_diag = ((g * g).unsqueeze(2) * (a * a).unsqueeze(1)).mean(0)      # E[g_i^2 a_j^2]
    A = a.T @ a / n
    G = g.T @ g / n
    factorised = torch.outer(G.diagonal(), A.diagonal())                    # E[g_i^2] E[a_j^2]
    covariance = ((g * g).unsqueeze(2) * (a * a).unsqueeze(1)).mean(0) - factorised
    assert torch.allclose(exact_diag - factorised, covariance, atol=1e-12)
    assert float(covariance.abs().max()) > 1e-2, "the fixture should have a real correlation"


# ----------------------------------------------------------------------------------------------
# The block-diagonal control, and the optimal rescaling
# ----------------------------------------------------------------------------------------------


def test_block_diagonal_control_matches_the_reference_blocks() -> None:
    seed_all(9)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3)).to(DTYPE)
    inputs = torch.randn(12, 4, dtype=DTYPE)
    targets = torch.randint(0, 3, (12,))
    reference = build_dense_reference(model, inputs, targets, source="type2", batch_size=6)
    layout = reference.layout
    ranges = [(name, layout.block_slice(name)) for name in ("0", "2")]

    control = block_diagonal_of(reference.matrix, ranges)
    assert torch.allclose(control.named()["0"].to_dense(), reference.block("0"), atol=1e-12)
    dense = control.to_dense()
    for name, columns in ranges:
        assert torch.allclose(dense[columns, columns], reference.block(name), atol=1e-12)
    assert torch.allclose(control.trace(), sum(reference.block(n).diagonal().sum()
                                               for n, _ in ranges), atol=1e-10)


def test_optimal_scale_minimises_the_frobenius_gap() -> None:
    """``c* = <R,K>/||K||^2`` is the minimiser of ``||R - cK||_F`` — needed for any structure that
    makes no claim on scale, and, since ``plan_exp_lot2.md`` §5.1, whenever two references built on
    different probe sets are compared at all.
    """
    seed_all(10)
    R = torch.randn(6, 6, dtype=DTYPE)
    R = R @ R.T
    K = Dense(torch.diag(torch.rand(6, dtype=DTYPE) + 0.1))
    c = float(optimal_scale(R, K))
    dense = K.to_dense()
    best = float(torch.linalg.matrix_norm(R - c * dense))
    for other in (c * 0.9, c * 1.1, c + 0.05, c - 0.05):
        assert best <= float(torch.linalg.matrix_norm(R - other * dense)) + 1e-12


# ----------------------------------------------------------------------------------------------
# The traversal extraction (plan_exp_lot2.md §0.2)
# ----------------------------------------------------------------------------------------------


def test_iter_probe_columns_visits_every_example_and_column_once() -> None:
    seed_all(11)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3)).to(DTYPE)
    inputs = torch.randn(10, 4, dtype=DTYPE)
    targets = torch.randint(0, 3, (10,))
    modules = capture.capturable_modules(model)

    seen = []
    for step in capture.iter_probe_columns(model, inputs, targets, source="type2",
                                           modules=modules, batch_size=4, dtype=DTYPE):
        seen.append((step.start, step.stop, step.column))
        assert step.capturer.layer("0").a.shape[0] == step.n_examples
    assert len(seen) == 3 * 3                                   # 3 micro-batches x C = 3 columns
    assert sorted({(s, e) for s, e, _ in seen}) == [(0, 4), (4, 8), (8, 10)]
    assert sorted({c for _, _, c in seen}) == [0, 1, 2]


def test_iter_probe_columns_is_reproducible_across_two_passes() -> None:
    """The reason the traversal is a generator: EKFAC needs a second pass, and the two must see the
    same probes and the same draws, or ``||R - K||`` would carry a sampling difference
    (``plan_exp_lot2.md`` §0.2).
    """
    seed_all(12)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3)).to(DTYPE)
    inputs = torch.randn(8, 4, dtype=DTYPE)
    targets = torch.randint(0, 3, (8,))
    modules = capture.capturable_modules(model)

    def collect() -> Tuple[Tensor, ...]:
        return tuple(step.capturer.layer("2").g.clone()
                     for step in capture.iter_probe_columns(
                         model, inputs, targets, source="type2", modules=modules,
                         batch_size=4, dtype=DTYPE))

    first, second = collect(), collect()
    assert len(first) == len(second)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


# ----------------------------------------------------------------------------------------------
# T8 / T9 / T13 — the estimators built from the shared traversal
# ----------------------------------------------------------------------------------------------


def _unshared_net() -> Tuple[nn.Module, Tensor, Tensor]:
    seed_all(15)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3)).to(DTYPE)
    return model, torch.randn(20, 4, dtype=DTYPE), torch.randint(0, 3, (20,))


def _zoo(model: nn.Module, inputs: Tensor, targets: Tensor, source: str = "type2"):
    modules = capture.capturable_modules(model)
    factors = accumulate_factors(model, inputs, targets, source=source, modules=modules,
                                 batch_size=int(inputs.shape[0]), dtype=DTYPE)
    ekfacs = accumulate_ekfac(model, inputs, targets, source=source, modules=modules,
                              factors=factors, batch_size=int(inputs.shape[0]), dtype=DTYPE)
    reference = build_dense_reference(model, inputs, targets, source=source,
                                      batch_size=int(inputs.shape[0]))
    return factors, ekfacs, reference


def test_t8_tkfac_preserves_the_trace_of_the_exact_block() -> None:
    """T8 — ``tr(K_TKFAC) = tr(B_l)``, to ``1e-10``, on an unshared layer.

    An identity of the estimator, not a property of the data: ``tr(K) = delta`` by unit-trace
    normalisation, and ``delta = E[tr(Lambda) tr(Gamma)] = E[||a||^2 ||g||^2] = tr(B_l)``.
    """
    model, inputs, targets = _unshared_net()
    factors, _, reference = _zoo(model, inputs, targets)
    for name, entry in factors.items():
        perm = reference.layout.augmented_permutation(name)
        exact = to_augmented(reference.block(name), perm)
        tkfac = entry.tkfac()
        assert torch.allclose(tkfac.trace(), exact.diagonal().sum(), rtol=0, atol=1e-10), name


def test_t9_ekfac_preserves_the_trace_and_its_bases_are_orthonormal() -> None:
    """T9 — ``tr(K_EKFAC) = tr(B_l)`` to ``1e-10``.

    ``sum_ij s_ij`` is the mean of ``||Q_G^T G Q_A||_F^2``, and an orthogonal change of basis leaves
    the Frobenius norm alone. So if this fails, the eigenbases are not orthonormal — which is
    asserted separately, because that is the only way the identity can break.
    """
    model, inputs, targets = _unshared_net()
    factors, ekfacs, reference = _zoo(model, inputs, targets)
    assert ekfacs, "the second sweep produced no EKFAC"
    for name, block in ekfacs.items():
        perm = reference.layout.augmented_permutation(name)
        exact = to_augmented(reference.block(name), perm)
        assert torch.allclose(block.trace(), exact.diagonal().sum(), rtol=0, atol=1e-10), name
        for basis in (block.QA, block.QG):
            identity = torch.eye(basis.shape[0], dtype=DTYPE)
            assert torch.allclose(basis.T @ basis, identity, atol=1e-10), name


def test_ekfac_dominates_kfac_in_frobenius_norm() -> None:
    """EKFAC's whole claim: the optimal diagonal *in K-FAC's basis* cannot be worse than K-FAC's own
    (EKFAC Thm 2/3). Measured against the exact block, as `test_frobenius_dominance.py` does for the
    optimizer's own implementations.
    """
    model, inputs, targets = _unshared_net()
    factors, ekfacs, reference = _zoo(model, inputs, targets)
    for name, block in ekfacs.items():
        perm = reference.layout.augmented_permutation(name)
        exact = to_augmented(reference.block(name), perm)
        kfac_gap = float(torch.linalg.matrix_norm(factors[name].kfac().to_dense() - exact))
        ekfac_gap = float(torch.linalg.matrix_norm(block.to_dense() - exact))
        assert ekfac_gap <= kfac_gap + 1e-12, f"{name}: EKFAC {ekfac_gap} > KFAC {kfac_gap}"


@pytest.mark.parametrize("name", ["0", "2"])
def test_ekfac_conformance(name: str) -> None:
    """The protocol, on a real EKFAC rather than a synthetic one."""
    model, inputs, targets = _unshared_net()
    _, ekfacs, _ = _zoo(model, inputs, targets)
    block = ekfacs[name]
    dense = block.to_dense()
    seed_all(16)
    v = torch.randn(block.P, dtype=DTYPE)
    assert torch.allclose(block.matvec(v), dense @ v, atol=1e-10)
    assert torch.allclose(block.diag(), dense.diagonal(), atol=1e-10)
    assert torch.allclose(block.trace(), dense.diagonal().sum(), atol=1e-10)
    assert torch.allclose(block.fro2(), (dense * dense).sum(), atol=1e-10)
    R = torch.randn(block.P, block.P, dtype=DTYPE)
    R = R @ R.T
    assert torch.allclose(block.inner_dense(R), (R * dense).sum(), atol=1e-9)
    lam = 0.5
    damped = dense + lam * torch.eye(block.P, dtype=DTYPE)
    assert torch.allclose(block.solve(v, lam), torch.linalg.solve(damped, v), atol=1e-9)
    assert torch.allclose(block.logdet(lam), torch.logdet(damped), atol=1e-9)


def test_t13_factors_match_adafisher_modes_at_the_degenerate_setting() -> None:
    """T13 — this campaign's K-FAC factors against ``adafisher_modes``' own, to ``1e-10``.

    The degenerate setting of ``plan_exp_draft.md`` §10.2: one update, no EMA, the **empirical**
    source (one column, so ``C = 1``), no damping. ``compute_h_full``/``compute_s_full`` are the
    optimizer's own formulas, pinned by 238 pre-existing tests — so this turns "our K-FAC" and "the
    optimizer's K-FAC" into one measured statement instead of two implementations nobody compared
    (``plan_exp_draft.md`` §0.12).
    """
    from adafisher_modes.factors import compute_h_full, compute_s_full  # noqa: PLC0415

    model, inputs, targets = _unshared_net()
    modules = capture.capturable_modules(model)
    factors = accumulate_factors(model, inputs, targets, source="empirical", modules=modules,
                                 batch_size=int(inputs.shape[0]), dtype=DTYPE)

    captured: dict = {}
    for step in capture.iter_probe_columns(model, inputs, targets, source="empirical",
                                           modules=modules, batch_size=int(inputs.shape[0]),
                                           dtype=DTYPE):
        for layer in step.capturer:
            captured[layer.name] = (layer.a.reshape(-1, layer.a.shape[-1]),
                                    layer.g.reshape(-1, layer.g.shape[-1]), layer.module)

    for name, (a, g, module) in captured.items():
        theirs_a = compute_h_full(a, module)
        theirs_g = compute_s_full(g, module)
        assert torch.allclose(factors[name].A, theirs_a, rtol=0, atol=1e-10), f"A of {name}"
        assert torch.allclose(factors[name].G, theirs_g, rtol=0, atol=1e-10), f"G of {name}"


# ----------------------------------------------------------------------------------------------
# HF4 — the normalisation block's four readings
# ----------------------------------------------------------------------------------------------


def test_hf4_normalisation_readings_are_all_computable_and_differ() -> None:
    """The exact ``2C x 2C`` block, the same with the cross terms dropped, Prop. 3.1's Hadamard
    form, and ``diag.py``'s as-implemented diagonal — all four on one real ``LayerNorm``.

    HF4 is the gap between the first two; the other two are what the optimizer actually uses.
    """
    seed_all(17)
    model = nn.Sequential(nn.Linear(4, 6), nn.LayerNorm(6), nn.Tanh(), nn.Linear(6, 3)).to(DTYPE)
    inputs = torch.randn(24, 4, dtype=DTYPE)
    targets = torch.randint(0, 3, (24,))
    modules = capture.capturable_modules(model)
    factors = accumulate_factors(model, inputs, targets, source="type2", modules=modules,
                                 batch_size=24, dtype=DTYPE)
    reference = build_dense_reference(model, inputs, targets, source="type2", batch_size=24)

    exact = reference.block("1")                       # (gamma, beta) stacked: 2C x 2C
    channels = 6
    assert exact.shape == (2 * channels, 2 * channels)

    separate = exact_separate(exact, channels)
    assert torch.allclose(separate.to_dense()[:channels, :channels], exact[:channels, :channels],
                          atol=1e-12)
    assert float(separate.to_dense()[:channels, channels:].abs().max()) == 0.0

    total_share, diagonal_share = cross_term_share(exact, channels)
    assert 0.0 < total_share < 1.0 and diagonal_share > 0.0

    stats = factors["1"].norm_stats()
    hadamard = stats.hadamard()
    assert hadamard.P == 2 * channels
    # beta's exact block is S itself (d/d beta = sum_t g), so Prop. 3.1 is exact there.
    assert torch.allclose(hadamard.to_dense()[channels:, channels:], exact[channels:, channels:],
                          atol=1e-10)

    as_implemented = stats.as_implemented_diagonal()
    assert as_implemented.P == 2 * channels
