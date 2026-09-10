"""Lot 8 (docs/reports/plan_lot8.md §2): the CIFAR-10 classification bench's models, its optimizer
changes, and its budget harness — all exercised **offline**, on synthetic data.

No test here downloads CIFAR-10, trains a real network to convergence, or touches the network: the
same discipline as ``tests/test_equal_wallclock_bench.py`` (``plan_lot7.md`` §0.8). The two
model-construction tests do instantiate the real ResNet-50/ViT (a fraction of a second each, no
data), because their parameter counts and hooked-module inventories are exactly what §0.1/§0.2 of
the plan claim.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from time import perf_counter, sleep
from typing import List, Optional, Tuple

import pytest
import torch
import torch.nn as nn
from conftest import TinyMultiLayerNet, seed_all
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adafisher_modes import AdaFisherMulti  # noqa: E402

from benchmarks.common.data import Cutout, seeded_train_val_split  # noqa: E402
from benchmarks.common.loop import train_under_budget  # noqa: E402
from benchmarks.common.records import (  # noqa: E402
    ArmResult,
    EpochRecord,
    StepRecord,
    write_epoch_csv,
    write_step_csv,
    write_summary,
)
from benchmarks.resnet50_cifar.model import build_resnet50_cifar  # noqa: E402
from benchmarks.vit_small_cifar.model import build_vit_small_cifar  # noqa: E402

# The two lot-8 nets, kept as a local table: the repository-wide registry is now the folder list
# (plan_exp_step1.md §5), exercised by tests/test_benchmark_models.py.
MODELS = {"resnet50": build_resnet50_cifar, "vit_small": build_vit_small_cifar}

_MODE_KWARGS = {
    "diag": {},
    "kfac": {"T_inv": 1},
    "ekfac": {"T_eig": 1},
    "tkfac": {"T_inv": 1},
    "tekfac": {"T_eig": 1, "T_re": 1},
}


def _count_types(model: nn.Module) -> dict:
    counts: dict = {}
    for module in model.modules():
        name = type(module).__name__
        if name in ("Linear", "Conv2d", "BatchNorm2d", "LayerNorm"):
            counts[name] = counts.get(name, 0) + 1
    return counts


# ----------------------------------------------------------------------------------------------
# Models (plan_lot8.md §0.1, §0.2)
# ----------------------------------------------------------------------------------------------


def test_resnet50_shapes_and_param_count() -> None:
    seed_all(0)
    model = build_resnet50_cifar(num_classes=10)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)

    n_params = sum(p.numel() for p in model.parameters())
    assert 23.4e6 < n_params < 23.6e6, n_params  # 23.52 M, plan_lot8.md §0.1
    assert _count_types(model) == {"Conv2d": 53, "BatchNorm2d": 53, "Linear": 1}

    # resnet_1512.03385.pdf §4.2's CIFAR stem: 3x3, stride 1, and no max-pool anywhere.
    assert model.conv1.kernel_size == (3, 3) and model.conv1.stride == (1, 1)
    assert not any(isinstance(m, nn.MaxPool2d) for m in model.modules())
    # Bottleneck expansion 4 (§4.1): layer4's output width is 512*4.
    assert model.fc.in_features == 2048


def test_vit_shapes_and_param_count() -> None:
    seed_all(0)
    model = build_vit_small_cifar(num_classes=10)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)

    n_params = sum(p.numel() for p in model.parameters())
    assert 2.6e6 < n_params < 2.8e6, n_params  # 2.69 M, plan_lot8.md §0.2
    assert _count_types(model) == {"Conv2d": 1, "LayerNorm": 13, "Linear": 25}

    # 32/4 = 8 -> 64 patches, plus the [class] token of Eq. (1) = 65 positions.
    assert model.patch_embed.num_patches == 64
    assert model.pos_embed.shape == (1, 65, 192)
    # Table 1's invariants, kept: MLP = 4D, D/heads = 64 (plan_lot8.md §0.2).
    block = model.blocks[0]
    assert block.mlp.fc1.out_features == 4 * 192
    assert block.attn.head_dim == 64
    # The projection must be a hookable nn.Linear, never nn.MultiheadAttention's raw Parameter.
    assert isinstance(block.attn.qkv, nn.Linear)
    assert not any(isinstance(m, nn.MultiheadAttention) for m in model.modules())


# ----------------------------------------------------------------------------------------------
# The §0.3 pairing regression
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_every_parameter_is_updated(model_name: str) -> None:
    """plan_lot8.md §0.3: with the reference's index-bookkeeping loop, ViT-S/4 left
    ``norm.weight``, ``norm.bias``, ``head.weight`` and ``head.bias`` bit-identical to their
    initial values forever — two unpaired ``Parameter``s (``cls_token``, ``pos_embed``) each cost
    one module its turn in a loop that ran exactly ``len(self.modules)`` times.
    """
    seed_all(0)
    model = MODELS[model_name](10)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = AdaFisherMulti(model, lr=1e-2, TCov=1, fisher_mode="diag")
    loss_fn = nn.CrossEntropyLoss()
    x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))

    for _ in range(2):
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()

    unchanged = [n for n, p in model.named_parameters() if torch.equal(p.detach(), before[n])]
    assert unchanged == [], f"{model_name}: {len(unchanged)} parameter(s) never updated: {unchanged}"
    assert all(torch.isfinite(p).all() for p in model.parameters())


def _legacy_selection(optimizer: AdaFisherMulti) -> List[Tuple[int, int, Optional[int]]]:
    """Replay the *removed* index-bookkeeping loop (adafisher.py:275-307) on this optimizer's
    parameters, returning the ``(id(module), id(weight), id(bias))`` triples it would have stepped.
    Used only to prove the new identity pairing reproduces it where it was correct.
    """
    params = optimizer.param_groups[0]["params"]
    modules = optimizer.modules
    selection: List[Tuple[int, int, Optional[int]]] = []
    idx_param = idx_module = buffer_count = 0
    for _ in range(len(modules)):
        if params[idx_param].grad is None:
            idx_param += 1
            if params[idx_param].ndim > 1:
                idx_module += 1
            else:
                buffer_count += 1
            if buffer_count == 2:
                idx_module += 1
                buffer_count = 0
            continue
        m = modules[idx_module]
        p = params[idx_param]
        matches = p.data.size() == m.weight.data.size() or (
            m.bias is not None and p.data.size() == m.bias.data.size()
        )
        if matches:
            has_bias = m.bias is not None
            selection.append(
                (id(m), id(params[idx_param]), id(params[idx_param + 1]) if has_bias else None)
            )
            idx_param += 2 if has_bias else 1
            idx_module += 1
        else:
            idx_param += 1
    return selection


class _TinyConvNet(nn.Module):
    """A Conv2d+BatchNorm2d+Linear net in the shape of lots 4/6's own smoke nets."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 4 * 4, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.relu(self.bn(self.conv(x))).flatten(1))


@pytest.mark.parametrize(
    "factory,batch",
    [
        (TinyMultiLayerNet, torch.randn(6, 2, 5, 5)),
        (_TinyConvNet, torch.randn(6, 3, 4, 4)),
    ],
)
def test_pairing_matches_legacy_loop_on_reference_nets(factory, batch) -> None:
    """plan_lot8.md §0.3: on every net where the old loop was correct, the new identity pairing
    must select exactly the same ``(module, weight, bias)`` triples, in the same order — which is
    why all 145 pre-lot-8 tests, bit-exactness included, keep passing.
    """
    seed_all(0)
    model = factory()
    optimizer = AdaFisherMulti(model, lr=1e-3, TCov=1, fisher_mode="diag")
    model(batch).sum().backward()

    stepped: List[Tuple[int, int, Optional[int]]] = []
    original = optimizer._step_module

    def spy(hparams, module, weight_param, bias_param):
        stepped.append((id(module), id(weight_param), None if bias_param is None else id(bias_param)))
        return original(hparams, module, weight_param, bias_param)

    optimizer._step_module = spy  # type: ignore[method-assign]
    expected = _legacy_selection(optimizer)
    optimizer.step()
    assert stepped == expected


# ----------------------------------------------------------------------------------------------
# fisher_batch_samples (plan_lot8.md §0.4)
# ----------------------------------------------------------------------------------------------


def test_fisher_batch_samples_slices_the_example_axis() -> None:
    seed_all(0)
    model = nn.Linear(4, 3)
    optimizer = AdaFisherMulti(model, fisher_batch_samples=2)
    for shape in [(8, 4), (8, 5, 4), (8, 3, 5, 5)]:  # Linear 2-D/3-D, Conv2d/BatchNorm2d 4-D
        sliced = optimizer._fisher_slice(torch.randn(*shape))
        assert sliced.shape == (2,) + tuple(shape[1:])
    # Inert when the batch is already smaller than the cap, and when the knob is unset.
    assert optimizer._fisher_slice(torch.randn(1, 4)).shape == (1, 4)
    assert AdaFisherMulti(nn.Linear(4, 3))._fisher_slice(torch.randn(8, 4)).shape == (8, 4)


@pytest.mark.parametrize("mode", list(_MODE_KWARGS))
def test_fisher_batch_samples_is_inert_when_larger_than_batch(mode: str) -> None:
    """Default (``None``) and a cap above the batch size must give bit-identical trajectories —
    the guarantee that this knob costs nothing when unused (plan_lot8.md §0.4, point 3).
    """
    trajectories = []
    for cap in (None, 1000):
        seed_all(0)
        model = TinyMultiLayerNet()
        optimizer = AdaFisherMulti(
            model, lr=1e-3, TCov=1, fisher_mode=mode, fisher_batch_samples=cap, **_MODE_KWARGS[mode]
        )
        seed_all(1)
        x = torch.randn(6, 2, 5, 5)
        for _ in range(3):
            optimizer.zero_grad()
            model(x).pow(2).mean().backward()
            optimizer.step()
        trajectories.append([p.detach().clone() for p in model.parameters()])
    for a, b in zip(*trajectories):
        assert torch.equal(a, b)


def test_fisher_batch_samples_bounds_the_cached_batch() -> None:
    """``ekfac`` caches the augmented input batch for its intra-batch ``s*`` estimator
    (plan_lot2.md §0.3); the cap must bound that cache's row count, which is the memory this knob
    exists to control (plan_lot8.md §0.4's 3.4 GB → 378 MB).
    """
    seed_all(0)
    model = nn.Sequential(nn.Linear(4, 3))
    optimizer = AdaFisherMulti(model, lr=1e-3, TCov=1, fisher_mode="ekfac", fisher_batch_samples=2)
    linear = model[0]
    seen: List[int] = []
    # Registered after the optimizer's own forward hook, so it fires after the cache is filled.
    linear.register_forward_hook(
        lambda mod, inp, out: seen.append(optimizer.approx._cached_h_bar[mod].size(0))
    )
    model(torch.randn(8, 4)).sum().backward()
    assert seen == [2]


# ----------------------------------------------------------------------------------------------
# Weight decay conventions (plan_lot8.md §0.6)
# ----------------------------------------------------------------------------------------------


def test_decoupled_weight_decay_matches_official_rule() -> None:
    """``AdaFisherW._step`` (reference_repos/AdaFisher/optimizers/AdaFisher.py:685):
    ``param -= lr * (exp_avg / bc / F_tilde + weight_decay * param)``. Comparing against the same
    optimizer at ``weight_decay=0`` isolates the decay term exactly: the preconditioned direction
    is identical (decoupled decay never enters the gradient, hence never enters ``exp_avg`` nor any
    factor), so the difference must be exactly ``-lr * wd * param_before``.
    """
    lr, wd = 0.5, 0.1
    finals = []
    for weight_decay, decoupled in [(wd, True), (0.0, False)]:
        seed_all(0)
        model = TinyMultiLayerNet()
        optimizer = AdaFisherMulti(
            model, lr=lr, TCov=1, fisher_mode="diag", weight_decay=weight_decay,
            decoupled_weight_decay=decoupled,
        )
        seed_all(1)
        before = [p.detach().clone() for p in model.parameters()]
        optimizer.zero_grad()
        model(torch.randn(6, 2, 5, 5)).pow(2).mean().backward()
        optimizer.step()
        finals.append((before, [p.detach().clone() for p in model.parameters()]))

    (before_decoupled, after_decoupled), (_, after_plain) = finals
    for p0, p_dec, p_plain in zip(before_decoupled, after_decoupled, after_plain):
        assert torch.allclose(p_dec, p_plain - lr * wd * p0, atol=1e-7)


def test_coupled_weight_decay_matches_reference_adafisher(fisheradaptune_adafisher) -> None:
    """The default (``decoupled_weight_decay=False``) coupled path is unchanged by lot 8: one step
    at a non-zero weight decay still matches the authoritative reference optimizer
    (``FisherAdapTune/scripts/adafisher.py``), the same comparison ``test_diag_bitexact`` makes at
    ``weight_decay=0``.
    """
    seed_all(0)
    model_a, model_b = TinyMultiLayerNet(), TinyMultiLayerNet()
    model_b.load_state_dict(model_a.state_dict())
    opt_a = AdaFisherMulti(
        model_a, lr=1e-2, TCov=1, fisher_mode="diag", minmax_normalization=False, weight_decay=5e-4
    )
    opt_b = fisheradaptune_adafisher.AdaFisher(model_b, lr=1e-2, TCov=1, weight_decay=5e-4)

    seed_all(1)
    x = torch.randn(6, 2, 5, 5)
    for opt, model in ((opt_a, model_a), (opt_b, model_b)):
        opt.zero_grad()
        model(x).pow(2).mean().backward()
        opt.step()
    for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.allclose(p_a, p_b, rtol=1e-6, atol=1e-8)


# ----------------------------------------------------------------------------------------------
# Budget harness (plan_lot8.md §0.7)
# ----------------------------------------------------------------------------------------------


def _synthetic_task(n: int = 64, batch_size: int = 8):
    seed_all(0)
    x, y = torch.randn(n, 6), torch.randint(0, 3, (n,))
    return (
        nn.Sequential(nn.Linear(6, 5), nn.Sigmoid(), nn.Linear(5, 3)),
        DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True),
    )


def test_budget_is_respected_and_eval_time_is_excluded() -> None:
    """The eval pass must not be charged to the budget (plan_lot8.md §0.7), and lot 7's overshoot
    bound must still hold: total elapsed exceeds the budget by at most one batch's own processing
    duration (plan_lot7.md §0.1).
    """
    seed_all(0)
    model, loader = _synthetic_task()
    optimizer = AdaFisherMulti(model, lr=1e-3, TCov=1, fisher_mode="diag")
    eval_calls = {"n": 0}

    def slow_eval() -> Tuple[float, float]:
        eval_calls["n"] += 1
        sleep(0.05)  # far larger than a step at this scale
        return 0.5, 0.25

    budget = 0.3
    t0 = perf_counter()
    steps, epochs = train_under_budget(
        model, optimizer, loader, nn.CrossEntropyLoss(),
        budget_s=budget, max_epochs=1000, eval_fn=slow_eval, log_fn=lambda _: None,
    )
    wall = perf_counter() - t0

    assert steps and epochs
    assert all(math.isfinite(r.loss) for r in steps)
    assert eval_calls["n"] == len(epochs)
    max_step = max(r.fwd_bwd_s + r.step_s for r in steps)
    assert steps[-1].elapsed_s <= budget + max_step + 1e-3
    # The excluded eval time really is excluded: real wall-clock exceeds the accounted budget by
    # roughly one 0.05s eval per epoch.
    assert wall >= steps[-1].elapsed_s + 0.05 * len(epochs) - 1e-2
    assert [e.epoch for e in epochs] == list(range(len(epochs)))
    assert all(e.val_acc == 0.25 for e in epochs)


def test_max_epochs_caps_a_cheap_arm() -> None:
    """plan_lot8.md §0.7 step 3: ``--max-epoch-factor`` bounds how far a cheap arm can run when the
    reference arm's budget turns out generous.
    """
    seed_all(0)
    model, loader = _synthetic_task()
    optimizer = AdaFisherMulti(model, lr=1e-3, TCov=1, fisher_mode="diag")
    steps, epochs = train_under_budget(
        model, optimizer, loader, nn.CrossEntropyLoss(),
        budget_s=1e6, max_epochs=3, log_fn=lambda _: None,
    )
    assert len(epochs) == 3
    assert len(steps) == 3 * len(loader)


def test_scheduler_steps_once_per_completed_epoch() -> None:
    seed_all(0)
    model, loader = _synthetic_task()
    optimizer = AdaFisherMulti(model, lr=1.0, TCov=1, fisher_mode="diag")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    _, epochs = train_under_budget(
        model, optimizer, loader, nn.CrossEntropyLoss(),
        budget_s=1e6, max_epochs=3, scheduler=scheduler, log_fn=lambda _: None,
    )
    assert [round(e.lr, 6) for e in epochs] == [0.5, 0.25, 0.125]


# ----------------------------------------------------------------------------------------------
# Data pipeline (plan_lot8.md §0.9) and reporting (§0.10)
# ----------------------------------------------------------------------------------------------


def test_cutout_masks_one_square_region() -> None:
    seed_all(0)
    img = torch.ones(3, 32, 32)
    out = Cutout(n_holes=1, length=16)(img)
    zeroed = (out[0] == 0)
    assert zeroed.any(), "Cutout masked nothing"
    assert zeroed.sum().item() <= 16 * 16
    rows, cols = zeroed.any(dim=1).nonzero().flatten(), zeroed.any(dim=0).nonzero().flatten()
    # One axis-aligned, contiguous rectangle.
    assert torch.equal(rows, torch.arange(int(rows[0]), int(rows[-1]) + 1))
    assert torch.equal(cols, torch.arange(int(cols[0]), int(cols[-1]) + 1))
    # Identical mask across channels, and untouched pixels are untouched.
    assert torch.equal(out[0] == 0, out[1] == 0) and torch.equal(out[0] == 0, out[2] == 0)
    assert torch.equal(out[~zeroed.unsqueeze(0).expand(3, -1, -1)], img[~zeroed.unsqueeze(0).expand(3, -1, -1)])


def test_train_val_split_is_seeded_disjoint_and_sized() -> None:
    """resnet_1512.03385.pdf §4.2's 45k/5k split, reproducible from a seed."""
    train, val = seeded_train_val_split(50_000, 5_000, seed=0)
    assert (len(train), len(val)) == (45_000, 5_000)
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(50_000))
    assert (train, val) == seeded_train_val_split(50_000, 5_000, seed=0)
    assert train != seeded_train_val_split(50_000, 5_000, seed=1)[0]


def _fake_result(arm: str) -> ArmResult:
    steps = [
        StepRecord(step=i, epoch=i / 4, elapsed_s=0.1 * i, loss=1.0 / (i + 1),
                   fwd_bwd_s=0.01, step_s=0.02)
        for i in range(8)
    ]
    epochs = [
        EpochRecord(epoch=e, steps=4 * (e + 1), elapsed_s=0.4 * (e + 1), train_loss=0.5 / (e + 1),
                    val_loss=0.6, val_acc=0.3 + 0.1 * e, lr=1e-3)
        for e in range(2)
    ]
    return ArmResult(arm, steps, epochs, test_acc=0.42, test_loss=0.7, total_s=0.8)


def test_summary_and_csv_schema(tmp_path: Path) -> None:
    results = [_fake_result("diag"), _fake_result("adamw")]
    write_step_csv(results, tmp_path / "records.csv")
    write_epoch_csv(results, tmp_path / "epochs.csv")
    summary = write_summary(results, tmp_path / "summary.md", skip_first=0)

    step_lines = (tmp_path / "records.csv").read_text().strip().splitlines()
    assert step_lines[0] == "arm,step,epoch,elapsed_s,loss,fwd_bwd_s,step_s"
    assert len(step_lines) == 1 + 2 * 8
    epoch_lines = (tmp_path / "epochs.csv").read_text().strip().splitlines()
    assert epoch_lines[0] == "arm,epoch,steps,elapsed_s,train_loss,val_loss,val_acc,lr"
    assert len(epoch_lines) == 1 + 2 * 2

    for column in ("best val acc", "test acc", "step/fwd+bwd", "total wall-clock"):
        assert column in summary
    assert "| diag |" in summary and "| adamw |" in summary
    assert "2.00x" in summary  # step_s / fwd_bwd_s = 0.02 / 0.01
    assert "40.00%" in summary  # best val acc = 0.3 + 0.1
    assert "42.00%" in summary  # test acc


def test_empty_arm_does_not_crash_the_report(tmp_path: Path) -> None:
    """A budget too small for a single batch yields no records; the report must still be written
    (the same degenerate case lot 7's ``write_summary`` already handles)."""
    empty = ArmResult("tekfac", [], [], test_acc=float("nan"), test_loss=float("nan"), total_s=0.0)
    summary = write_summary([empty], tmp_path / "summary.md")
    assert "| tekfac | 0 |" in summary
