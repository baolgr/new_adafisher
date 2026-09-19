"""Lot 3 of the Fisher-drift campaign: weight sharing in regime A
(``docs/reports/plan_exp_lot3.md`` §2; ``plan_exp_draft.md`` §9's lot 3, §10.2's T5).

Offline, CPU, fp64 throughout, as lots 1 and 2. Every test here exercises a branch that only
activates at ``T > 1`` positions — which is why none of them could fail on A1, and why three of the
branches they cover were wrong before this lot (``plan_exp_lot3.md`` §0.1-§0.3).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch
import torch.nn as nn
from conftest import seed_all
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fisher_ref import capture  # noqa: E402
from fisher_ref.approx import Kron, accumulate_ekfac, accumulate_factors  # noqa: E402
from fisher_ref.reference import build_dense_reference, to_augmented  # noqa: E402

DTYPE = torch.float64


# ----------------------------------------------------------------------------------------------
# Toy settings of Eschenhagen et al. (arXiv:2311.00636 §3.2-§3.3): deep linear networks, MSE
# ----------------------------------------------------------------------------------------------


class _ExpandLinear(nn.Module):
    """Expand setting: a shared ``Linear`` over ``T`` tokens, a per-token linear head, one loss term
    per ``(n, t)`` (the output is flattened, so the MSE root has one column per token-and-class).
    """

    def __init__(self, d_in: int = 3, d_hidden: int = 4, d_out: int = 2) -> None:
        super().__init__()
        self.shared = nn.Linear(d_in, d_hidden)
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.shared(x)).flatten(1)


class _ReduceLinear(nn.Module):
    """Reduce setting: a shared ``Linear`` over ``T`` tokens, mean pooling, a linear head."""

    def __init__(self, d_in: int = 3, d_hidden: int = 4, d_out: int = 2) -> None:
        super().__init__()
        self.shared = nn.Linear(d_in, d_hidden)
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.shared(x).mean(dim=1))


class _ExpandConv(nn.Module):
    """Expand setting for a convolution: a 1x1 convolution after it is a per-position linear map."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Conv2d(2, 3, kernel_size=3, padding=1)
        self.out = nn.Conv2d(3, 2, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.shared(x)).flatten(1)


class _ReduceConv(nn.Module):
    """Reduce setting for a convolution: global average pooling, then a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Conv2d(2, 3, kernel_size=3, padding=1, stride=2)
        self.out = nn.Linear(3, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.shared(x).mean(dim=(2, 3)))


def _setting(kind: str) -> Tuple[nn.Module, Tensor, Tensor]:
    seed_all(40)
    n = 7
    if kind == "expand_linear":
        model, inputs = _ExpandLinear(), torch.randn(n, 5, 3, dtype=DTYPE)
    elif kind == "reduce_linear":
        model, inputs = _ReduceLinear(), torch.randn(n, 5, 3, dtype=DTYPE)
    elif kind == "expand_conv":
        model, inputs = _ExpandConv(), torch.randn(n, 2, 4, 4, dtype=DTYPE)
    else:
        model, inputs = _ReduceConv(), torch.randn(n, 2, 5, 5, dtype=DTYPE)
    model = model.to(DTYPE)
    with torch.no_grad():
        targets = torch.randn_like(model(inputs))
    return model, inputs, targets


def _exact_block(model: nn.Module, inputs: Tensor, targets: Tensor, source: str = "type2",
                 loss: str = "mse", name: str = "shared") -> Tensor:
    reference = build_dense_reference(model, inputs, targets, source=source, loss=loss,
                                      batch_size=int(inputs.shape[0]))
    return to_augmented(reference.block(name), reference.layout.augmented_permutation(name))


def _factors(model: nn.Module, inputs: Tensor, targets: Tensor, mode: str, source: str = "type2",
             loss: str = "mse") -> Dict[str, object]:
    return accumulate_factors(model, inputs, targets, source=source, loss=loss,
                              modules=capture.capturable_modules(model), mode=mode,
                              batch_size=int(inputs.shape[0]), dtype=DTYPE)


def _rel(a: Tensor, b: Tensor) -> float:
    return float(torch.linalg.matrix_norm(a - b) / torch.linalg.matrix_norm(b))


# ----------------------------------------------------------------------------------------------
# T5 — expand exact in the expand setting, reduce exact in the reduce setting (§0.1)
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["expand_linear", "expand_conv"])
def test_t5_kfac_expand_is_exact_in_the_expand_setting_and_reduce_is_not(kind: str) -> None:
    """T5, Prop. 1 of arXiv:2311.00636 (§3.2): K-FAC-expand is exact for a deep linear network with a
    Gaussian likelihood in the expand setting. The negative half is what gives the test teeth:
    K-FAC-reduce must *not* be exact here, or a scale error in either could pass unnoticed.
    """
    model, inputs, targets = _setting(kind)
    exact = _exact_block(model, inputs, targets)
    expand = _factors(model, inputs, targets, "expand")["shared"]
    reduce = _factors(model, inputs, targets, "reduce")["shared"]
    assert _rel(Kron(A=expand.A, G=expand.G).to_dense(), exact) < 1e-10
    assert _rel(Kron(A=reduce.A, G=reduce.G).to_dense(), exact) > 1e-3


@pytest.mark.parametrize("kind", ["reduce_linear", "reduce_conv"])
def test_t5_kfac_reduce_is_exact_in_the_reduce_setting_and_expand_is_not(kind: str) -> None:
    """T5, Prop. 2 of arXiv:2311.00636 (§3.3, Eq. 10): K-FAC-reduce — the input factor built on the
    **mean** over positions, the gradient factor on the **sum** — is exact in the reduce setting.

    Lot 2's reduce branch summed both, so it was exactly ``T^2`` times this block: relative error
    ``T^2 - 1`` (24.000 at ``T = 5``, measured; ``plan_exp_lot3.md`` §0.1). This test is what fails
    on that defect.
    """
    model, inputs, targets = _setting(kind)
    exact = _exact_block(model, inputs, targets)
    expand = _factors(model, inputs, targets, "expand")["shared"]
    reduce = _factors(model, inputs, targets, "reduce")["shared"]
    assert _rel(Kron(A=reduce.A, G=reduce.G).to_dense(), exact) < 1e-10
    assert _rel(Kron(A=expand.A, G=expand.G).to_dense(), exact) > 1e-3


# ----------------------------------------------------------------------------------------------
# T8-exp / T9-shared — the trace identities on a shared layer (§0.2, §0.3)
# ----------------------------------------------------------------------------------------------


class _SharedClassifier(nn.Module):
    """A nonlinear shared layer (neither setting is exact): conv, tanh, pooling, head."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Conv2d(2, 3, kernel_size=3, padding=1)
        self.act = nn.Tanh()
        self.out = nn.Linear(3, 4)

    def forward(self, x: Tensor) -> Tensor:
        return self.out(self.act(self.shared(x)).mean(dim=(2, 3)))


def _shared_classifier() -> Tuple[nn.Module, Tensor, Tensor]:
    seed_all(41)
    model = _SharedClassifier().to(DTYPE)
    return model, torch.randn(9, 2, 4, 4, dtype=DTYPE), torch.randint(0, 4, (9,))


def _captured_rows(model: nn.Module, inputs: Tensor, targets: Tensor, name: str,
                   source: str = "type2", loss: str = "cross_entropy"):
    """Every ``(n, c)`` column's bias-augmented ``a`` and ``g`` for one layer, ``(N*C, T, d)``."""
    a_rows, g_rows = [], []
    modules = capture.capturable_modules(model)
    for step in capture.iter_probe_columns(model, inputs, targets, source=source, loss=loss,
                                           modules=modules, batch_size=int(inputs.shape[0]),
                                           dtype=DTYPE):
        layer = step.capturer.layer(name)
        a = torch.cat([layer.a, layer.a.new_ones(*layer.a.shape[:2], 1)], dim=2)
        a_rows.append(a.clone())
        g_rows.append(layer.g.clone())
    return torch.cat(a_rows), torch.cat(g_rows)


def test_t8_exp_tkfac_expand_preserves_the_trace_of_b_exp() -> None:
    """T8-exp: TKFAC-expand treats every ``(n, c, t)`` as one sample, as ``adafisher_modes`` does,
    so ``tr(K) = (1/N) sum ||a_t||^2 ||g_t||^2 = tr(B^exp)`` — trace preservation once the
    cross-position terms are dropped (``plan_exp_lot3.md`` §0.3). Lot 2's per-example summed traces
    fail this: their product carries every cross-position pair.
    """
    model, inputs, targets = _shared_classifier()
    n = int(inputs.shape[0])
    a, g = _captured_rows(model, inputs, targets, "shared")
    trace_b_exp = ((a * a).sum(-1) * (g * g).sum(-1)).sum() / n
    entry = accumulate_factors(model, inputs, targets, source="type2",
                               modules=capture.capturable_modules(model), batch_size=n,
                               dtype=DTYPE)["shared"]
    assert torch.allclose(entry.tkfac().trace(), trace_b_exp, rtol=1e-12, atol=0)


def test_t8_red_tkfac_reduce_preserves_the_trace_of_the_reduced_block() -> None:
    """The same estimator on ``(a_bar, g_hat) = (mean_t a, sum_t g)``: ``tr(K) = tr(B^red)``."""
    model, inputs, targets = _shared_classifier()
    n = int(inputs.shape[0])
    a, g = _captured_rows(model, inputs, targets, "shared")
    a_bar, g_hat = a.mean(dim=1), g.sum(dim=1)
    trace_b_red = ((a_bar * a_bar).sum(-1) * (g_hat * g_hat).sum(-1)).sum() / n
    entry = accumulate_factors(model, inputs, targets, source="type2", mode="reduce",
                               modules=capture.capturable_modules(model), batch_size=n,
                               dtype=DTYPE)["shared"]
    assert torch.allclose(entry.tkfac().trace(), trace_b_red, rtol=1e-12, atol=0)


@pytest.mark.parametrize("mode", ["expand", "reduce"])
def test_t9_shared_ekfac_preserves_the_trace_of_the_exact_block(mode: str) -> None:
    """T9 on a shared layer, both bases: ``s`` is the second moment of the **true** per-sample
    gradient projected into the basis, so ``sum s = tr(B)`` whatever the basis. Lot 2's reduce
    branch projected ``(sum_t a)(sum_t g)^T`` instead — not a gradient of anything — and fails
    (``plan_exp_lot3.md`` §0.2).
    """
    model, inputs, targets = _shared_classifier()
    n = int(inputs.shape[0])
    modules = capture.capturable_modules(model)
    factors = accumulate_factors(model, inputs, targets, source="type2", modules=modules,
                                 mode=mode, batch_size=n, dtype=DTYPE)
    ekfacs = accumulate_ekfac(model, inputs, targets, source="type2", modules=modules,
                              factors=factors, mode=mode, batch_size=n, dtype=DTYPE)
    exact = _exact_block(model, inputs, targets, loss="cross_entropy")
    assert torch.allclose(ekfacs["shared"].trace(), exact.diagonal().sum(), rtol=1e-12, atol=0)
    kfac_gap = _rel(factors["shared"].kfac().to_dense(), exact)
    ekfac_gap = _rel(ekfacs["shared"].to_dense(), exact)
    assert ekfac_gap <= kfac_gap + 1e-12, f"EKFAC {ekfac_gap} > K-FAC {kfac_gap} ({mode})"


# ----------------------------------------------------------------------------------------------
# T13-shared — this campaign's shared-layer factors against adafisher_modes' own (§0.3, §0.4)
# ----------------------------------------------------------------------------------------------


class _TokenNet(nn.Module):
    """A token-wise ``Linear`` (3-D input) and a ``Conv2d``, both shared, feeding a classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=3, padding=1, stride=2)
        self.token = nn.Linear(3, 5)
        self.head = nn.Linear(5, 4)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.conv(x).flatten(2).transpose(1, 2)            # (N, T, 3)
        return self.head(torch.tanh(self.token(tokens)).mean(dim=1))


def test_t13_shared_factors_match_adafisher_modes_up_to_the_declared_constants() -> None:
    """T13 at ``T > 1``, at the degenerate setting (empirical source, one update, no EMA, no damping).

    Fed the **same** per-example gradients, the optimizer's formulas average over ``(n, t)`` rows on
    both factors, while K-FAC-expand (arXiv:2311.00636 Eq. 7) divides only the input factor by the
    positions. So ``A`` agrees exactly and ``G_campaign = T * B_optimizer``; TKFAC's ``Phi``, ``Psi``
    agree exactly and ``delta_campaign = T * delta_optimizer``. Both operators are therefore ``T``
    times the optimizer's — a declared constant, pinned here, not a discrepancy
    (``plan_exp_lot3.md`` §0.3-§0.4).
    """
    from adafisher_modes.approximations._tkfac_utils import (  # noqa: PLC0415
        instantaneous_raw_factors,
    )
    from adafisher_modes.factors import (  # noqa: PLC0415
        augment_input,
        compute_h_full,
        compute_s_full,
        flatten_output_grad,
    )

    seed_all(42)
    model = _TokenNet().to(DTYPE)
    inputs = torch.randn(6, 2, 6, 6, dtype=DTYPE)
    targets = torch.randint(0, 4, (6,))
    n = int(inputs.shape[0])

    raw_inputs: Dict[str, Tensor] = {}
    raw_grads: Dict[str, Tensor] = {}
    handles = []
    for name in ("conv", "token"):
        module = getattr(model, name)

        def hook(mod, args, out, _name=name):
            raw_inputs[_name] = args[0].detach()
            out.register_hook(lambda grad, __name=_name: raw_grads.__setitem__(__name, grad))

        handles.append(module.register_forward_hook(hook))
    loss = nn.functional.cross_entropy(model(inputs), targets, reduction="sum")
    loss.backward()
    for handle in handles:
        handle.remove()

    factors = accumulate_factors(model, inputs, targets, source="empirical",
                                 modules=capture.capturable_modules(model), batch_size=n,
                                 dtype=DTYPE)
    for name in ("conv", "token"):
        module, entry = getattr(model, name), factors[name]
        positions = entry.positions
        assert positions > 1
        assert torch.allclose(entry.A, compute_h_full(raw_inputs[name], module),
                              rtol=0, atol=1e-10), name
        assert torch.allclose(entry.G, positions * compute_s_full(raw_grads[name], module),
                              rtol=0, atol=1e-10), name
        delta_i, phi_i, psi_i = instantaneous_raw_factors(
            augment_input(raw_inputs[name], module), flatten_output_grad(raw_grads[name], module))
        stats_delta = entry.tkfac_delta / entry.tkfac_count
        assert torch.allclose(stats_delta, positions * delta_i, rtol=1e-12, atol=0), name
        tk = entry.tkfac()
        assert torch.allclose(tk.A, phi_i / delta_i, rtol=0, atol=1e-10), name
        assert torch.allclose(tk.G / stats_delta, psi_i / delta_i, rtol=0, atol=1e-10), name


# ----------------------------------------------------------------------------------------------
# Raw parameters and the row checks (§0.6)
# ----------------------------------------------------------------------------------------------


def _tiny_vit(pool: str) -> nn.Module:
    from benchmarks.common.runner import discover_benchmarks  # noqa: PLC0415

    bench = discover_benchmarks()["vit_micro_cifar"]
    vit_cls = bench.build_model.func
    seed_all(43)
    model = vit_cls(img_size=8, patch_size=4, embed_dim=8, depth=1, num_heads=2, mlp_ratio=2.0,
                    pool=pool, drop=0.0, attn_drop=0.0, num_classes=3)
    with torch.no_grad():  # zero-initialised raw parameters would make the check trivial
        for name, parameter in model.named_parameters():
            if "." not in name:
                parameter.normal_()
    return model.to(DTYPE)


@pytest.mark.parametrize("pool", ["mean", "cls"])
def test_raw_parameter_rows_match_torch_func_per_sample_gradients(pool: str) -> None:
    """``pos_embed`` (and ``cls_token``) per-sample rows against ``vmap(grad)``, and a whole-model
    reference whose rows pass both autograd checks. Before lot 3 this model's reference could only
    be built restricted — 10 % of A3's parameters missing, and still labelled ``F``.
    """
    from torch.func import functional_call, grad, vmap  # noqa: PLC0415

    from fisher_ref.reference import DenseAccumulator, ParamLayout  # noqa: PLC0415

    model = _tiny_vit(pool)
    raw = capture.raw_parameter_names(model)
    assert raw == (["cls_token", "pos_embed"] if pool == "cls" else ["pos_embed"])
    seed_all(44)
    inputs = torch.randn(5, 3, 8, 8, dtype=DTYPE)
    targets = torch.randint(0, 3, (5,))

    reference = build_dense_reference(model, inputs, targets, source="empirical", batch_size=5,
                                      raw_parameters=raw, check_rows=True)
    assert reference.P == sum(p.numel() for p in model.parameters())

    params = {name: p.detach() for name, p in model.named_parameters()}

    def sample_loss(theta, x, y):
        logits = functional_call(model, theta, (x.unsqueeze(0),))
        return nn.functional.cross_entropy(logits, y.unsqueeze(0), reduction="sum")

    with capture.reference_mode(model):
        per_sample = vmap(grad(sample_loss), in_dims=(None, 0, 0))(params, inputs, targets)
    step = next(capture.iter_probe_columns(model, inputs, targets, source="empirical",
                                           modules=capture.capturable_modules(model),
                                           batch_size=5, dtype=DTYPE, raw_parameters=raw))
    accumulator = DenseAccumulator(ParamLayout.of(model), dtype=DTYPE, device="cpu")
    accumulator.consume(step)
    for name in raw:
        expected = per_sample[name].reshape(5, -1)
        got = step.raw_grads[name].reshape(5, -1)
        assert torch.allclose(got, expected, rtol=0, atol=1e-12), name


def test_row_checks_catch_a_scaled_column_and_an_exchanged_example() -> None:
    """Both checks must be able to fail. A rescaled parameter slice breaks the sum rule; two
    exchanged examples preserve the sum exactly and are caught only by the single-example check —
    which is why there are two (``plan_exp_lot3.md`` §0.6).
    """
    from fisher_ref.reference import (  # noqa: PLC0415
        DenseAccumulator,
        ParamLayout,
        RowCheckError,
        check_rows_against_autograd,
    )

    model = _tiny_vit("mean")
    raw = capture.raw_parameter_names(model)
    seed_all(45)
    inputs = torch.randn(4, 3, 8, 8, dtype=DTYPE)
    targets = torch.randint(0, 3, (4,))
    layout = ParamLayout.of(model)
    step = next(capture.iter_probe_columns(model, inputs, targets, source="type2",
                                           modules=capture.capturable_modules(model),
                                           batch_size=4, dtype=DTYPE, raw_parameters=raw))
    accumulator = DenseAccumulator(layout, dtype=DTYPE, device="cpu")
    accumulator.consume(step, model=model, check=True)       # the genuine rows pass
    assert accumulator.check_report is not None
    assert max(accumulator.check_report.values()) < 1e-10
    rows = accumulator._rows.clone()

    scaled = rows.clone()
    scaled[:, layout.slices["pos_embed"]] *= 1.001
    with pytest.raises(RowCheckError, match="sum rule"):
        check_rows_against_autograd(model, step, scaled, layout)

    exchanged = rows[[1, 0, 2, 3]]
    assert torch.allclose(exchanged.sum(0), rows.sum(0), rtol=0, atol=1e-14)
    with pytest.raises(RowCheckError, match="single example"):
        check_rows_against_autograd(model, step, exchanged, layout)


def test_offload_returns_a_copy_even_on_the_host() -> None:
    """On a CPU run ``.to("cpu")`` is the same tensor, so a fold stored that way would be zeroed by the
    buffer reset that follows — all-zero folds locally, correct ones on the GPU."""
    from fisher_ref.reference import DenseAccumulator, ParamLayout  # noqa: PLC0415

    model = nn.Linear(2, 2).to(DTYPE)
    accumulator = DenseAccumulator(ParamLayout.of(model), dtype=DTYPE, device="cpu")
    accumulator.matrix += 1.0
    stored = accumulator.offload()
    assert float(stored.sum()) == 36.0 and float(accumulator.matrix.abs().sum()) == 0.0


# ----------------------------------------------------------------------------------------------
# The sharing decomposition (§0.5)
# ----------------------------------------------------------------------------------------------


def test_b_exp_half_vectorised_accumulation_matches_brute_force() -> None:
    """``R(B^exp)`` accumulated in half-vectorised coordinates and unpacked equals
    ``(1/N) sum kron(g g^T, a a^T)`` built densely, and every contraction the decomposition uses
    agrees with its dense counterpart. Written to fail on the float32-``sqrt(2)`` trap of §0.5.
    """
    from fisher_ref.approx.base import rearrange  # noqa: PLC0415
    from fisher_ref.approx.sharing import (  # noqa: PLC0415
        linear_decomposition,
        sharing_sums_for,
    )

    model, inputs, targets = _shared_classifier()
    n = int(inputs.shape[0])
    a, g = _captured_rows(model, inputs, targets, "shared")
    d_in, d_out = a.shape[-1], g.shape[-1]
    dense = torch.einsum("rti,rtk,rtj,rtl->ikjl", g, a, g, a).reshape(
        d_out * d_in, d_out * d_in) / n

    sums = None
    modules = capture.capturable_modules(model)
    for step in capture.iter_probe_columns(model, inputs, targets, source="type2", modules=modules,
                                           batch_size=4, dtype=DTYPE):
        layer = step.capturer.layer("shared")
        sums = sums or sharing_sums_for(layer)
        sums.consume(layer, first_column=(step.column == 0), chunk_bytes=2048)
    assert sums is not None and sums.n_probes == n
    rearranged = sums.rearranged()
    assert torch.allclose(rearranged, rearrange(dense, d_out, d_in), rtol=1e-12, atol=1e-15)
    assert abs(float(rearranged.norm() / torch.linalg.matrix_norm(dense)) - 1.0) < 1e-13

    exact = _exact_block(model, inputs, targets, loss="cross_entropy")
    kfac = accumulate_factors(model, inputs, targets, source="type2", modules=modules,
                              batch_size=n, dtype=DTYPE)["shared"].kfac()
    rows = {row["metric"]: row["value"] for row in linear_decomposition(
        rearrange(exact, d_out, d_in), rearranged, kfac, exact.diagonal(), d_out, d_in)}
    assert abs(rows["sharing_share"] - _rel(dense, exact)) < 1e-8
    assert abs(rows["independence_share_exp"] - _rel(kfac.to_dense(), dense)) < 1e-8
    expected_diag = float((exact.diagonal() - dense.diagonal()).norm() / exact.diagonal().norm())
    assert abs(rows["diag_sharing"] - expected_diag) < 1e-12
    singular = torch.linalg.svdvals(rearrange(dense, d_out, d_in))
    assert abs(rows["bexp_sigma2_over_sigma1"] - float(singular[1] / singular[0])) < 1e-8
    # The triangle inequality through B^exp bounds K-FAC-expand's own error.
    assert _rel(kfac.to_dense(), exact) <= rows["sharing_share"] + rows["independence_share_total"] + 1e-12


def test_b_exp_is_the_exact_block_without_sharing() -> None:
    """At ``T = 1`` there are no cross-position terms: ``B^exp = B`` identically."""
    from fisher_ref.approx.sharing import sharing_sums_for  # noqa: PLC0415

    seed_all(46)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3)).to(DTYPE)
    inputs, targets = torch.randn(8, 4, dtype=DTYPE), torch.randint(0, 3, (8,))
    sums = None
    for step in capture.iter_probe_columns(model, inputs, targets, source="type2",
                                           modules=capture.capturable_modules(model),
                                           batch_size=8, dtype=DTYPE):
        layer = step.capturer.layer("0")
        sums = sums or sharing_sums_for(layer)
        sums.consume(layer, first_column=(step.column == 0))
    assert sums is not None
    exact = _exact_block(model, inputs, targets, loss="cross_entropy", name="0")
    from fisher_ref.approx.base import rearrange  # noqa: PLC0415
    assert torch.allclose(sums.rearranged(), rearrange(exact, 5, 5), rtol=0, atol=1e-13)


def test_norm_b_exp_and_the_beta_block_under_sharing() -> None:
    """On a ``GroupNorm`` over feature maps: ``B^exp`` is ``(1/N) sum v v^T`` per position, and
    Proposition 3.1's ``beta = S`` equals ``B^exp``'s ``beta`` block — **not** the exact one, which
    carries the cross-position sum ``(sum_t g_t)(sum_t g_t)^T`` (lot 2 §5.4 holds at ``T = 1`` only).
    """
    from fisher_ref.approx.sharing import sharing_sums_for  # noqa: PLC0415

    seed_all(47)
    model = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.GroupNorm(2, 4), nn.Tanh(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)).to(DTYPE)
    inputs, targets = torch.randn(6, 2, 4, 4, dtype=DTYPE), torch.randint(0, 3, (6,))
    modules = capture.capturable_modules(model)
    sums = None
    for step in capture.iter_probe_columns(model, inputs, targets, source="type2", modules=modules,
                                           batch_size=6, dtype=DTYPE):
        layer = step.capturer.layer("1")
        sums = sums or sharing_sums_for(layer)
        sums.consume(layer, first_column=(step.column == 0))
    assert sums is not None
    reference = build_dense_reference(model, inputs, targets, source="type2", batch_size=6)
    exact = reference.block("1")
    bexp = sums.dense()
    factors = accumulate_factors(model, inputs, targets, source="type2", modules=modules,
                                 batch_size=6, dtype=DTYPE)
    hadamard = factors["1"].norm_stats().hadamard().to_dense()
    assert torch.allclose(hadamard[4:, 4:], bexp[4:, 4:], rtol=1e-12, atol=1e-15)
    assert _rel(hadamard[4:, 4:], exact[4:, 4:]) > 1e-3


# ----------------------------------------------------------------------------------------------
# Normalisation readings (§0.7): the renamed lot-2 reading, diag.py's own, and the theorems
# ----------------------------------------------------------------------------------------------


def test_lot2_as_implemented_is_the_diagonal_of_the_hadamard_form() -> None:
    seed_all(48)
    from fisher_ref.approx import NormStats  # noqa: PLC0415

    H = torch.randn(5, 5, dtype=DTYPE)
    S = torch.randn(5, 5, dtype=DTYPE)
    stats = NormStats(H=H @ H.T, S=S @ S.T, C=5)
    assert torch.equal(stats.as_implemented_diagonal().values, stats.hadamard_diagonal().values)
    assert torch.allclose(stats.hadamard_diagonal().values, stats.hadamard().to_dense().diagonal(),
                          rtol=0, atol=1e-14)


@pytest.mark.parametrize("norm", ["batchnorm", "layernorm3d"])
def test_diag_py_reading_is_diag_approximation_f_tilde(norm: str) -> None:
    """One micro-batch of :func:`diag_py_reading` equals ``DiagApproximation(minmax=False,
    gammas=(1, 1), Lambda=0).f_tilde`` fed the optimizer's own ``(h, s)`` — the shipped formula,
    called rather than re-derived (``plan_exp_lot3.md`` §0.7)."""
    from adafisher_modes.approximations.diag import DiagApproximation  # noqa: PLC0415

    from fisher_ref.approx.norm_layers import diag_py_reading  # noqa: PLC0415

    seed_all(49)
    if norm == "batchnorm":
        model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1), nn.BatchNorm2d(3), nn.ReLU(),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4)).to(DTYPE)
        with torch.no_grad():
            model[1].running_mean.normal_()
            model[1].running_var.uniform_(0.5, 2.0)
        inputs = torch.randn(8, 2, 4, 4, dtype=DTYPE)
    else:
        class Tokens(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(3, 5)
                self.norm = nn.LayerNorm(5)
                self.head = nn.Linear(5, 4)

            def forward(self, x: Tensor) -> Tensor:
                return self.head(self.norm(self.embed(x)).mean(dim=1))

        model = Tokens().to(DTYPE)
        inputs = torch.randn(8, 6, 3, dtype=DTYPE)
    targets = torch.randint(0, 4, (8,))
    name = "1" if norm == "batchnorm" else "norm"
    module = dict(model.named_modules())[name]
    loss_fn = nn.CrossEntropyLoss()

    reading = diag_py_reading(model, inputs, targets, {name: module}, loss_fn=loss_fn,
                              batch_size=8, dtype=DTYPE)[name].values

    captured: Dict[str, Tensor] = {}

    def hook(mod: nn.Module, args: Tuple, out: Tensor) -> None:  # must return None, or it
        captured["h"] = args[0].detach()                          # replaces the module output
        out.register_hook(lambda g: captured.__setitem__("s", g))

    handle = module.register_forward_hook(hook)
    model.eval()
    loss_fn(model(inputs), targets).backward()
    handle.remove()
    approx = DiagApproximation(Lambda=0.0, gammas=(1.0, 1.0), minmax_normalization=False)
    approx.update_input_factor(module, captured["h"], step=0)
    approx.update_output_factor(module, captured["s"], step=0)
    f_tilde = approx.f_tilde(module)                                     # (C, 2)
    expected = torch.cat([f_tilde[:, 0], f_tilde[:, 1]])
    assert torch.allclose(reading, expected, rtol=1e-12, atol=0)


def test_theorem_orderings_hold_on_real_blocks() -> None:
    """The orderings §0.7 shows are theorems: a violation is a bug, never a finding.

    Norm block: ``exact_separate <= hadamard``, ``exact_separate <= exact_diag <= hadamard_diag``.
    Shared layer, both modes: ``EKFAC <= K-FAC``; ``best_kron <= `` every Kronecker structure.
    """
    from fisher_ref.approx import Diag, exact_separate  # noqa: PLC0415
    from fisher_ref.approx.base import rearrange  # noqa: PLC0415
    from fisher_ref.metrics import frobenius, kronecker_analysis  # noqa: PLC0415

    seed_all(50)
    model = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.GroupNorm(2, 4), nn.Tanh(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)).to(DTYPE)
    inputs, targets = torch.randn(10, 2, 4, 4, dtype=DTYPE), torch.randint(0, 3, (10,))
    modules = capture.capturable_modules(model)
    reference = build_dense_reference(model, inputs, targets, source="type2", batch_size=10)

    norm_block = reference.block("1")
    stats = accumulate_factors(model, inputs, targets, source="type2", modules=modules,
                               batch_size=10, dtype=DTYPE)["1"].norm_stats()
    e = {name: frobenius(norm_block, block).e_F for name, block in {
        "separate": exact_separate(norm_block, 4), "hadamard": stats.hadamard(),
        "exact_diag": Diag(norm_block.diagonal().clone()),
        "hadamard_diag": stats.hadamard_diagonal()}.items()}
    assert e["separate"] <= e["hadamard"] + 1e-12
    assert e["separate"] <= e["exact_diag"] + 1e-12
    assert e["exact_diag"] <= e["hadamard_diag"] + 1e-12

    exact = to_augmented(reference.block("0"), reference.layout.augmented_permutation("0"))
    report, A_best, G_best = kronecker_analysis(rearrange(exact, 4, 19), 4, 19)
    best = frobenius(exact, Kron(A=A_best, G=G_best)).e_F
    for mode in ("expand", "reduce"):
        factors = accumulate_factors(model, inputs, targets, source="type2", modules=modules,
                                     mode=mode, batch_size=10, dtype=DTYPE)
        ekfacs = accumulate_ekfac(model, inputs, targets, source="type2", modules=modules,
                                  factors=factors, mode=mode, batch_size=10, dtype=DTYPE)
        kfac_e = frobenius(exact, factors["0"].kfac()).e_F
        assert frobenius(exact, ekfacs["0"]).e_F <= kfac_e + 1e-12, mode
        assert best <= kfac_e + 1e-12 and best <= frobenius(exact, factors["0"].tkfac()).e_F + 1e-12


def test_kronecker_analysis_matches_the_svd() -> None:
    """One Gram eigendecomposition gives the same M7 numbers and the same best fit as lot 2's two
    SVDs, on both orientations of the rearrangement."""
    from fisher_ref.approx.base import rearrange  # noqa: PLC0415
    from fisher_ref.metrics import (  # noqa: PLC0415
        best_kronecker_fit,
        kronecker_analysis,
        kronecker_structure,
    )

    seed_all(51)
    for d_out, d_in in ((3, 5), (5, 3)):
        B = torch.randn(d_out * d_in, d_out * d_in, dtype=DTYPE)
        B = B @ B.T
        old = kronecker_structure(B, d_out, d_in)
        new, A, G = kronecker_analysis(rearrange(B, d_out, d_in), d_out, d_in)
        assert abs(new.sigma_ratio - old.sigma_ratio) < 1e-10
        assert abs(new.tail_mass - old.tail_mass) < 1e-10
        A_old, G_old = best_kronecker_fit(B, d_out, d_in)
        assert torch.allclose(torch.kron(G, A), torch.kron(G_old, A_old), rtol=0, atol=1e-10)


def test_inner_rearranged_agrees_with_inner_dense() -> None:
    from fisher_ref.approx.base import rearrange  # noqa: PLC0415

    model, inputs, targets = _shared_classifier()
    modules = capture.capturable_modules(model)
    factors = accumulate_factors(model, inputs, targets, source="type2", modules=modules,
                                 batch_size=9, dtype=DTYPE)
    ekfac = accumulate_ekfac(model, inputs, targets, source="type2", modules=modules,
                             factors=factors, batch_size=9, dtype=DTYPE)["shared"]
    exact = _exact_block(model, inputs, targets, loss="cross_entropy")
    rearranged = rearrange(exact, 3, 19)
    for block in (factors["shared"].kfac(), ekfac):
        assert torch.allclose(block.inner_rearranged(rearranged), block.inner_dense(exact),
                              rtol=1e-12, atol=0)


def test_micro_batched_probe_gradient_is_exact() -> None:
    from fisher_ref.metrics import probe_gradient  # noqa: PLC0415

    model, inputs, targets = _shared_classifier()
    loss_fn = nn.CrossEntropyLoss()
    whole = probe_gradient(model, inputs, targets, loss_fn)
    batched = probe_gradient(model, inputs, targets, loss_fn, batch_size=4)
    assert torch.allclose(batched, whole, rtol=1e-12, atol=1e-15)


# ----------------------------------------------------------------------------------------------
# pos_embed structures (§0.6)
# ----------------------------------------------------------------------------------------------


def test_pos_embed_structures() -> None:
    """``kfac_onehot`` is ``kron(I_T, mean_t B_tt)`` in the ``(t, d)`` layout, it shares every
    position block's trace with the block-diagonal, and ``position_blockdiag`` keeps exactly the
    per-position blocks."""
    from fisher_ref.approx.embed import (  # noqa: PLC0415
        kfac_onehot,
        position_blockdiag,
        position_coupling_share,
    )

    model = _tiny_vit("mean")
    seed_all(52)
    inputs, targets = torch.randn(6, 3, 8, 8, dtype=DTYPE), torch.randint(0, 3, (6,))
    reference = build_dense_reference(model, inputs, targets, source="type2", batch_size=6,
                                      raw_parameters=["pos_embed"])
    block = reference.block("pos_embed")
    positions, width = 4, 8
    blockdiag = position_blockdiag(block, positions, width).to_dense()
    for t in range(positions):
        span = slice(t * width, (t + 1) * width)
        assert torch.equal(blockdiag[span, span], block[span, span])
    onehot = kfac_onehot(block, positions, width)
    mean_block = sum(block[t * width:(t + 1) * width, t * width:(t + 1) * width]
                     for t in range(positions)) / positions
    assert torch.allclose(onehot.to_dense(), torch.kron(torch.eye(positions, dtype=DTYPE),
                                                        mean_block), rtol=0, atol=1e-14)
    assert torch.allclose(onehot.trace(), block.diagonal().sum(), rtol=1e-12, atol=0)
    share, _ = position_coupling_share(block, positions, width)
    assert abs(share - _rel(blockdiag, block)) < 1e-10


# ----------------------------------------------------------------------------------------------
# Folds (§0.8)
# ----------------------------------------------------------------------------------------------


def test_layer_factors_merge_equals_a_single_accumulation() -> None:
    from fisher_ref.approx import LayerFactors  # noqa: PLC0415

    model, inputs, targets = _shared_classifier()
    modules = capture.capturable_modules(model)
    whole = accumulate_factors(model, inputs, targets, source="type2", modules=modules,
                               batch_size=9, dtype=DTYPE)["shared"]
    parts = [accumulate_factors(model, inputs[s], targets[s], source="type2", modules=modules,
                                batch_size=9, dtype=DTYPE)["shared"]
             for s in (slice(0, 4), slice(4, 9))]
    merged = LayerFactors.merge(parts)
    assert merged.n_probes == 9
    for attr in ("A", "G"):
        assert torch.allclose(getattr(merged, attr), getattr(whole, attr), rtol=1e-13, atol=0)
    assert torch.allclose(merged.tkfac().to_dense(), whole.tkfac().to_dense(), rtol=1e-12, atol=0)


def test_fold_engine_reassembles_every_statistic_exactly() -> None:
    """One traversal over folds, reassembled, equals the direct builders — the dense reference,
    both K-FAC modes, EKFAC in both bases, ``B^exp`` — and a half assembled from folds equals a direct
    build on exactly those probes (``plan_exp_lot3.md`` §0.8)."""
    from fisher_ref import folds as fold_engine  # noqa: PLC0415
    from fisher_ref.approx.sharing import sharing_sums_for  # noqa: PLC0415
    from fisher_ref.reference import ParamLayout  # noqa: PLC0415

    seed_all(53)
    model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1), nn.GroupNorm(1, 3), nn.Tanh(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4)).to(DTYPE)
    inputs, targets = torch.randn(24, 2, 4, 4, dtype=DTYPE), torch.randint(0, 4, (24,))
    modules = capture.capturable_modules(model)
    layout = ParamLayout.of(model)
    plan = fold_engine.FoldPlan(n_probes=24, batch_size=3, folds=4)
    assert plan.sizes() == [6, 6, 6, 6]

    folds, report = fold_engine.first_pass(model, inputs, targets, source="type2", modules=modules,
                                           layout=layout, plan=plan, dtype=DTYPE)
    assert report is not None and max(report.values()) < 1e-10
    full_factors = fold_engine.assemble(folds, range(4)).factors
    bases = fold_engine.eigenbases(full_factors)
    fold_engine.second_pass(model, inputs, targets, folds, source="type2", modules=modules,
                            bases=bases, plan=plan, dtype=DTYPE)
    full = fold_engine.assemble(folds, range(4), bases=bases)

    reference = build_dense_reference(model, inputs, targets, source="type2", batch_size=3)
    assert torch.allclose(full.matrix, reference.matrix, rtol=1e-12, atol=1e-15)
    for mode in ("expand", "reduce"):
        direct = accumulate_factors(model, inputs, targets, source="type2", modules=modules,
                                    mode=mode, batch_size=3, dtype=DTYPE)
        ek = accumulate_ekfac(model, inputs, targets, source="type2", modules=modules,
                              factors=direct, mode=mode, batch_size=3, dtype=DTYPE)
        assert torch.allclose(full.factors[mode]["0"].kfac().to_dense(),
                              direct["0"].kfac().to_dense(), rtol=1e-12, atol=1e-15)
        assert torch.allclose(full.factors[mode]["0"].tkfac().to_dense(),
                              direct["0"].tkfac().to_dense(), rtol=1e-11, atol=1e-15)
        assert torch.allclose(full.ekfac[mode]["0"].to_dense(), ek["0"].to_dense(),
                              rtol=1e-11, atol=1e-15)
    assert "0" not in full.factors["reduce"] or full.factors["reduce"]["0"].positions > 1
    assert "5" not in full.factors["reduce"], "no reduce statistics at T = 1"
    assert set(full.sharing) == {"0", "1"}, "B^exp only where there are positions"

    half = fold_engine.assemble(folds, [0, 2])
    probes = torch.cat([torch.arange(0, 6), torch.arange(12, 18)])
    direct_half = build_dense_reference(model, inputs[probes], targets[probes], source="type2",
                                        batch_size=3)
    assert torch.allclose(half.matrix, direct_half.matrix, rtol=1e-12, atol=1e-15)
    sums = None
    for step in capture.iter_probe_columns(model, inputs[probes], targets[probes], source="type2",
                                           modules=modules, batch_size=3, dtype=DTYPE):
        layer = step.capturer.layer("0")
        sums = sums or sharing_sums_for(layer)
        sums.consume(layer, first_column=(step.column == 0))
    assert sums is not None
    assert torch.allclose(half.sharing["0"].rearranged(), sums.rearranged(), rtol=1e-12, atol=1e-15)


def test_partitions_are_distinct_and_balanced() -> None:
    from fisher_ref.folds import interval, partitions  # noqa: PLC0415

    splits = partitions(10, 20, seed=0)
    assert len(splits) == 20
    keys = {tuple(min(a, b)) for a, b in splits}
    assert len(keys) == 20
    for first, second in splits:
        assert sorted(first + second) == list(range(10)) and len(first) == 5
    with pytest.raises(ValueError):
        partitions(4, 4)
    from fisher_ref.folds import FoldPlan  # noqa: PLC0415
    with pytest.raises(ValueError, match="divisible"):
        FoldPlan(n_probes=240, batch_size=24, folds=4)
    assert FoldPlan(n_probes=240, batch_size=30, folds=4).sizes() == [60] * 4
    low, high, sd = interval(1.0, [0.9, 1.1, 1.0, 1.0])
    assert low < 1.0 < high and abs((high - low) / 2 - 1.96 * sd / 2 ** 0.5) < 1e-12


# ----------------------------------------------------------------------------------------------
# The runner, end to end (§1)
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bench_name, modules, raw", [
    ("cnn_gn_cifar", ["features.0", "features.1", "head"], ["auto"]),
    ("vit_micro_cifar", ["blocks.0.norm1", "blocks.0.attn.proj", "head"], ["pos_embed"]),
])
def test_p1_runner_lot3_structures_and_intervals(tmp_path: Path, bench_name: str, modules: list,
                                                 raw: list) -> None:
    """On a synthetic checkpoint tree of a real shared-weight model: the lot-3 structures appear
    exactly where they apply (reduce and ``B^exp`` only at ``T > 1``; ``pos_embed``'s own structures
    only when it is requested), fold intervals fill ``ci_low``/``ci_high``, HF2's paired
    differences and the per-layer floors are written, and no theorem ordering is violated."""
    import csv
    import json

    from benchmarks.common.checkpoints import CheckpointWriter  # noqa: PLC0415
    from benchmarks.common.runner import discover_benchmarks  # noqa: PLC0415
    from fisher_ref import conventions  # noqa: PLC0415
    from fisher_ref.runners import p1_structural  # noqa: PLC0415

    data_root = REPO_ROOT / "benchmarks" / "data"
    if not (data_root / "cifar-10-batches-py").is_dir():
        pytest.skip("CIFAR-10 is not staged under benchmarks/data")
    bench = discover_benchmarks()[bench_name]
    seed_all(54)
    model = bench.build_model()
    arm_dir = tmp_path / "outputs" / "cifar10" / bench_name / "diag"
    writer = CheckpointWriter(output_dir=arm_dir, fractions=(0.5,), total_steps=100, seed=0)
    writer.save(0.5, 50, 1, model, scheduled=True)
    (tmp_path / "outputs" / "cifar10" / bench_name / "manifest.json").write_text(
        json.dumps({"config": {"seed": 0, "model": bench_name}, "arms": {"diag": {}}}))

    out = tmp_path / "results"
    p1_structural.main([
        "--model", bench_name, "--arm", "diag", "--fractions", "0.5", "--probes", "24",
        "--batch", "6", "--modules", *modules, "--raw-parameters", *raw, "--folds", "2",
        "--partitions", "1", "--noise-partitions", "0", "--alphas", "1e-3,1",
        "--outputs-root", str(tmp_path / "outputs"), "--out-dir", str(out),
        "--data-root", str(data_root), "--device", "cpu",
    ])
    written = out / bench_name / "diag" / "seed0" / "0.5"
    rows = list(csv.DictReader((written / "metrics.csv").open()))
    assert rows and list(rows[0]) == list(p1_structural.COLUMNS)
    assert not [row for row in rows if row["metric"] == "invariant_violation"]

    def structures_of(layer: str) -> set:
        return {row["structure"] for row in rows if row["layer"] == layer}

    shared = modules[1] if bench_name == "vit_micro_cifar" else "features.0"
    assert {"kfac", "kfac_reduce", "ekfac_reduce", "tkfac_reduce", "b_exp", "identity"} <= \
        structures_of(shared)
    assert not {"kfac_reduce", "b_exp"} & structures_of("head"), "no reduce/B^exp at T = 1"
    norm = modules[0] if bench_name == "vit_micro_cifar" else "features.1"
    assert {"hadamard", "exact_separate", "hadamard_diag", "b_exp"} <= structures_of(norm)
    # 24 probes < one training batch of 128: diag_py is skipped and the skip recorded, never
    # computed at a different batch size (the LayerNorm reading itself is test_diag_py_*'s job).
    assert "diag_py" not in structures_of(norm)
    if bench_name == "vit_micro_cifar":
        assert {"position_blockdiag", "kfac_onehot"} <= structures_of("pos_embed")
    else:
        assert "pos_embed" not in {row["layer"] for row in rows}

    typed = [row for row in rows if row["source"] == "type2" and row["metric"] == "e_F"
             and row["layer"] == shared]
    assert typed and all(row["ci_low"] not in ("", None) for row in typed)
    assert any(row["metric"] == "delta_e_F" and row["structure"] == "kfac_reduce-kfac"
               for row in rows)
    assert any(row["metric"] == "layer_noise_floor" and row["layer"] == shared for row in rows)
    assert not [row for row in rows if row["source"] == "empirical" and row["ci_low"]]

    meta = json.loads((written / "meta.json").read_text())
    assert meta["metrics_version"] == conventions.METRICS_VERSION
    checks = meta["references"]["0.5/type2"]["row_checks"]
    assert max(checks.values()) < 1e-10
    assert meta["diag_py_skipped"] == {"0.5": 24}


def test_tkfac_at_one_position_is_lot2s_estimator() -> None:
    """The §0.3 change is confined to ``T > 1``: without sharing, the per-``(n, c, t)`` flattening
    and lot 2's per-example traces are the same sum, so A1's numbers are untouched."""
    seed_all(55)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3)).to(DTYPE)
    inputs, targets = torch.randn(12, 4, dtype=DTYPE), torch.randint(0, 3, (12,))
    modules = capture.capturable_modules(model)
    entry = accumulate_factors(model, inputs, targets, source="type2", modules=modules,
                               batch_size=12, dtype=DTYPE)["0"]

    delta = torch.zeros((), dtype=DTYPE)
    phi = psi = None
    for step in capture.iter_probe_columns(model, inputs, targets, source="type2",
                                           modules=modules, batch_size=12, dtype=DTYPE):
        layer = step.capturer.layer("0")
        a = torch.cat([layer.a, layer.a.new_ones(*layer.a.shape[:2], 1)], dim=2)
        g = layer.g
        # lot 2's formula: traces summed over positions per example, then multiplied
        trace_a, trace_g = (a * a).sum(dim=(1, 2)), (g * g).sum(dim=(1, 2))
        delta = delta + (trace_a * trace_g).sum()
        wa = torch.einsum("n,ntd,nte->de", trace_g, a, a)
        wg = torch.einsum("n,ntd,nte->de", trace_a, g, g)
        phi = wa if phi is None else phi + wa
        psi = wg if psi is None else psi + wg
    assert torch.allclose(entry.tkfac_delta, delta, rtol=1e-12, atol=0)
    assert torch.allclose(entry.tkfac_phi, phi, rtol=1e-12, atol=0)
    assert torch.allclose(entry.tkfac_psi, psi, rtol=1e-12, atol=0)


# ----------------------------------------------------------------------------------------------
# The fold engine's two probe passes: what makes them describe the same probes
# ----------------------------------------------------------------------------------------------


def _mc_fold_setup(folds_count: int = 2):
    """A tiny classifier plus the fold plan and layout the two passes below share."""
    from fisher_ref.reference import ParamLayout  # noqa: PLC0415

    seed_all(56)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 6)).to(DTYPE)
    inputs, targets = torch.randn(16, 4, dtype=DTYPE), torch.randint(0, 6, (16,))
    modules = capture.capturable_modules(model)
    layout = ParamLayout.of(model)
    from fisher_ref import folds as fold_engine  # noqa: PLC0415
    plan = fold_engine.FoldPlan(n_probes=16, batch_size=8, folds=folds_count)
    return model, inputs, targets, modules, layout, plan


def _run_both_passes(source: str, k: int = 1, mc_seed: int = 0):
    """One full two-pass traversal, returned as ``(assembled, layout)``."""
    from fisher_ref import folds as fold_engine  # noqa: PLC0415

    model, inputs, targets, modules, layout, plan = _mc_fold_setup()
    parts, _ = fold_engine.first_pass(model, inputs, targets, source=source, modules=modules,
                                      layout=layout, plan=plan, dtype=DTYPE, k=k, mc_seed=mc_seed)
    bases = fold_engine.eigenbases(fold_engine.assemble(parts, range(plan.folds)).factors)
    fold_engine.second_pass(model, inputs, targets, parts, source=source, modules=modules,
                            bases=bases, plan=plan, dtype=DTYPE, k=k, mc_seed=mc_seed)
    return fold_engine.assemble(parts, range(plan.folds), bases=bases), layout, parts


@pytest.mark.parametrize("source,k", [("type2", 1), ("mc", 1), ("mc", 4)])
def test_the_two_probe_passes_see_the_same_backprop_vectors(source: str, k: int) -> None:
    """EKFAC's ``sum_ij s_ij = tr(B_l)`` across the fold engine's **two** traversals.

    Pass 1 builds the dense reference and K-FAC's factors; pass 2 projects the per-sample gradients
    into the resulting eigenbasis to get EKFAC's rescaling ``s``. Both sums of squares are the same
    per-sample gradients seen twice, and an orthogonal change of basis preserves the Frobenius norm,
    so ``sum_ij s_ij`` must equal the trace of the exact block **exactly** -- as long as the two
    passes are looking at the same backprop vectors.

    ``type2`` is the control: it is a closed form of the predicted distribution and is therefore
    paired whatever the random state. ``mc`` draws its labels, so this identity is precisely the
    test of whether the two passes re-seeded from the same number. Unpaired, ``s`` is measured on a
    second draw of the labels and the identity breaks by a few percent -- and with it EKFAC's
    Frobenius dominance over K-FAC, which is a theorem only for the gradients ``s`` was measured on.
    """
    full, layout, _ = _run_both_passes(source, k=k)
    assert full.ekfac["expand"], "the second pass produced no EKFAC"
    for name, block in full.ekfac["expand"].items():
        columns = layout.block_slice(name)
        exact = to_augmented(full.matrix[columns, columns], layout.augmented_permutation(name))
        assert torch.allclose(block.trace(), exact.diagonal().sum(), rtol=0, atol=1e-12), (
            f"{name}: sum(s) = {float(block.trace()):.12g} against tr(B) = "
            f"{float(exact.diagonal().sum()):.12g}")


def test_monte_carlo_draws_are_reproducible_and_mc_seed_moves_them() -> None:
    """The pairing is a *fixed* draw, not a suppressed one: same ``mc_seed`` reproduces the
    reference bit for bit, a different one gives a different draw (or the knob does nothing)."""
    again, _, _ = _run_both_passes("mc", mc_seed=0)
    once, _, _ = _run_both_passes("mc", mc_seed=0)
    other, _, _ = _run_both_passes("mc", mc_seed=1)
    assert torch.equal(once.matrix, again.matrix), "same mc_seed must give the same draw"
    assert not torch.equal(once.matrix, other.matrix), "mc_seed does not select the draw"


def test_mc_sample_count_reaches_the_traversal() -> None:
    """``k`` was unreachable from the fold engine, pinning the Monte-Carlo source at one sample.

    It is checked on the column count rather than on a value, because that is what ``k`` controls:
    the root has ``k`` columns per probe, each carrying the ``k^-1/2`` that keeps the estimator
    unbiased (``fisher_ref.sources``).
    """
    for k in (1, 3):
        _, _, parts = _run_both_passes("mc", k=k)
        assert [part.n_columns for part in parts] == [k] * len(parts), f"k={k}"


@pytest.mark.parametrize("source,expected", [("type2", 6), ("empirical", 1)])
def test_first_pass_records_the_root_column_count(source: str, expected: int) -> None:
    """Every fold carries how many root columns one probe contributed.

    It is the rank budget a reference's metadata quotes as ``n_rows = n_probes * n_columns``, and
    the accumulator that knows it does not outlive the traversal. Six for the type-2 softmax root
    of a six-class head (rank five per probe: ``sqrt(p)`` is in its kernel), one for the empirical
    source's single gradient.
    """
    from fisher_ref import folds as fold_engine  # noqa: PLC0415

    model, inputs, targets, modules, layout, plan = _mc_fold_setup()
    parts, _ = fold_engine.first_pass(model, inputs, targets, source=source, modules=modules,
                                      layout=layout, plan=plan, dtype=DTYPE)
    assert [part.n_columns for part in parts] == [expected] * plan.folds
    assert sum(part.n_probes for part in parts) * expected == 16 * expected


# ----------------------------------------------------------------------------------------------
# The pre-registered decision rules (fisher_ref/experiments/lot3_decisions.py)
# ----------------------------------------------------------------------------------------------


def _decision_rows(*, shared_ci_low: str, floor: str) -> Dict[float, list]:
    """One checkpoint's worth of HF2 rows: one shared convolution and one head.

    Built so the HF2 (i) clause is decided by exactly the two numbers under test — the shared
    layers' lower bound against the head's upper bound, and each shared layer's sharing share
    against its own noise floor.
    """
    def row(layer, layer_type, structure, metric, value, ci_low="", ci_high=""):
        return {"layer": layer, "layer_type": layer_type, "source": "type2",
                "structure": structure, "metric": metric, "value": value,
                "ci_low": ci_low, "ci_high": ci_high, "lambda_alpha": ""}

    return {0.5: [
        row("conv1", "conv", "kfac", "e_F", "0.7", ci_low=shared_ci_low, ci_high="0.8"),
        row("head", "head", "kfac", "e_F", "0.2", ci_low="0.1", ci_high="0.3"),
        row("conv1", "conv", "reference", "layer_noise_floor", floor),
        row("conv1", "conv", "b_exp", "sharing_share", "0.3"),
    ]}


def test_a_measured_zero_bound_is_read_as_zero_not_skipped() -> None:
    """A lower bound of exactly ``0.0`` must be used, not stepped over.

    ``_float(ci_low) or _float(value) or 0.0`` falls through on any falsy value, and ``0.0`` is
    falsy — so a bound measured *at* zero was silently replaced by the point value, which is a
    different number and, here, a larger one. The rule compares the shared layers' lower bound
    against the head's upper bound, so substituting the point value can turn "the intervals
    overlap" into "the shared layers are strictly worse" and confirm HF2 on a checkpoint that does
    not support it.
    """
    from fisher_ref.experiments.lot3_decisions import hf2_shared_vs_unshared  # noqa: PLC0415

    out = hf2_shared_vs_unshared("toy", _decision_rows(shared_ci_low="0.0", floor="0.1"))
    # low = 0.0 is not above the head's high = 0.3, and high = 0.8 is not below its low = 0.1.
    assert out["supports"] == 0 and out["against"] == 0
    assert out["verdict"] == "mixed"


def test_a_measured_zero_noise_floor_is_read_as_zero_not_as_one() -> None:
    """A per-layer noise floor of exactly ``0.0`` must stay ``0.0``, not become ``1.0``.

    ``floors.get(layer) or 1.0`` exists so a *missing* floor fails the mechanism clause. A floor
    measured at zero is not missing: it means every share clears it. Read as ``1.0`` it means
    almost none does, which is the opposite reading and flips the clause -- and with it the
    verdict, since the clause is an ``and`` on the supporting branch.
    """
    from fisher_ref.experiments.lot3_decisions import hf2_shared_vs_unshared  # noqa: PLC0415

    out = hf2_shared_vs_unshared("toy", _decision_rows(shared_ci_low="0.5", floor="0.0"))
    # low = 0.5 > head high = 0.3, and the share 0.3 clears a floor of 0.0.
    assert out["supports"] == 1 and out["verdict"] == "confirmed"


def test_a_missing_bound_or_floor_still_fails_the_rule() -> None:
    """The conservative half of the same rule, which the fix must not loosen: an **absent** value
    falls back exactly as before -- a missing bound to the point value, a missing floor to 1.0."""
    from fisher_ref.experiments.lot3_decisions import hf2_shared_vs_unshared  # noqa: PLC0415

    # No ci_low on the shared layer: the point value 0.7 stands in, and 0.7 > 0.3 supports HF2.
    assert hf2_shared_vs_unshared("toy", _decision_rows(shared_ci_low="", floor="0.1"))[
        "supports"] == 1
    # No floor row at all: the clause must fail, so nothing is counted as support.
    data = _decision_rows(shared_ci_low="0.5", floor="0.1")
    data[0.5] = [r for r in data[0.5] if r["metric"] != "layer_noise_floor"]
    assert hf2_shared_vs_unshared("toy", data)["supports"] == 0
