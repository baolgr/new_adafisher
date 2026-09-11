"""Step 1 (``docs/reports/plan_exp_step1.md`` §6): the one test that makes adding a model safe.

Parametrized over every ``benchmarks/<model>/bench.py`` the folder registry discovers, so a new
model folder is covered the moment it exists. Everything runs on synthetic tensors — no MNIST, no
CIFAR download — the discipline of ``plan_lot7.md`` §0.8.

The expected parameter counts, hooked-module inventories and unhooked-parameter lists below are
the *contract*: ``plan_exp_step1.md`` §4's table, with three values corrected against the built
models (each noted at its entry). An unhooked parameter is one belonging to no
``AdaFisherMulti`` ``SUPPORTED_MODULES`` instance; it receives the identity preconditioner
(momentum-SGD fallback), so the list must be explicit rather than discovered.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from conftest import seed_all

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adafisher_modes import AdaFisherMulti  # noqa: E402
from adafisher_modes.optimizer import SUPPORTED_MODULES  # noqa: E402

from benchmarks.common.optimizers import ARMS  # noqa: E402
from benchmarks.common.runner import Benchmark, discover_benchmarks  # noqa: E402

BENCHES: Dict[str, Benchmark] = discover_benchmarks()

_MODE_KWARGS = {
    "diag": {},
    "kfac": {"T_inv": 1},
    "ekfac": {"T_eig": 1},
    "tkfac": {"T_inv": 1},
    "tekfac": {"T_eig": 1, "T_re": 1},
}


class Expected:
    """``(parameters, hooked-module counts, unhooked parameter names, synthetic input shape)``."""

    def __init__(self, params: int, counts: Dict[str, int], unhooked: List[str],
                 input_shape: Tuple[int, ...]) -> None:
        self.params, self.counts, self.unhooked, self.input_shape = (
            params, counts, unhooked, input_shape
        )


EXPECTED: Dict[str, Expected] = {
    # plan_exp_step1.md §4, "migrated" rows.
    "mnist_autoencoder": Expected(2_837_314, {"Linear": 8}, [], (1, 28, 28)),
    "resnet50_cifar": Expected(
        # §4 transcribes 23 519 178; the model — migrated verbatim from lot 8's cifar10_models.py,
        # not modified here — has 23 520 842. plan_lot8.md's own "23.52 M" is the correct rounding.
        23_520_842, {"Conv2d": 53, "BatchNorm2d": 53, "Linear": 1}, [], (3, 32, 32),
    ),
    "vit_small_cifar": Expected(
        # §4 transcribes 2 685 898; the verbatim-migrated model has 2 693 578 ("2.69 M" either way).
        2_693_578, {"Conv2d": 1, "LayerNorm": 13, "Linear": 25}, ["cls_token", "pos_embed"],
        (3, 32, 32),
    ),
    # plan_exp_step1.md §4, "new" rows — A1, A2, A3, B2, B1.
    "mlp_ln_mnist": Expected(26_634, {"Linear": 3, "LayerNorm": 2}, [], (1, 28, 28)),
    "cnn_gn_cifar": Expected(
        24_458, {"Conv2d": 3, "Linear": 1},
        # GroupNorm is deliberately not a SUPPORTED_MODULES type (see the model's docstring).
        ["features.1.weight", "features.1.bias", "features.5.weight", "features.5.bias",
         "features.9.weight", "features.9.bias"],
        (3, 32, 32),
    ),
    "vit_micro_cifar": Expected(
        # §4 estimates "≈ 21 162 [DERIVED] — the exact value is fixed by the config and locked by
        # the test once built". Mean pooling removes the cls_token (32) and its position row (32).
        21_098, {"Conv2d": 1, "LayerNorm": 5, "Linear": 9}, ["pos_embed"], (3, 32, 32),
    ),
    "resnet20_cifar": Expected(
        269_722, {"Conv2d": 19, "BatchNorm2d": 19, "Linear": 1}, [], (3, 32, 32),
    ),
    "cct_2_3x2_cifar": Expected(
        283_723, {"Conv2d": 2, "LayerNorm": 5, "Linear": 10}, ["pos_embed"], (3, 32, 32),
    ),
}

SLOW = {"resnet50_cifar"}
MODEL_IDS = sorted(BENCHES)


def _build(name: str) -> nn.Module:
    bench = BENCHES[name]
    return bench.build_model(**{k: v[0] for k, v in bench.model_choices.items()})


def _hooked_inventory(model: nn.Module) -> Tuple[Dict[str, int], List[str]]:
    """``(count per hooked module type, names of parameters owned by no hooked module)`` — the
    same ownership rule ``AdaFisherMulti._prepare_model`` applies.
    """
    counts: Dict[str, int] = {}
    owned: set = set()
    for module in model.modules():
        type_name = type(module).__name__
        if type_name not in SUPPORTED_MODULES or getattr(module, "weight", None) is None:
            continue
        counts[type_name] = counts.get(type_name, 0) + 1
        owned.add(id(module.weight))
        if module.bias is not None:
            owned.add(id(module.bias))
    return counts, [n for n, p in model.named_parameters() if id(p) not in owned]


def _synthetic_batch(name: str, batch_size: int = 4):
    seed_all(1)
    shape = EXPECTED[name].input_shape
    return torch.randn(batch_size, *shape), torch.randint(0, 10, (batch_size,))


def _one_arm_steps(
    name: str, mode: str, n_steps: int = 2
) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    bench = BENCHES[name]
    seed_all(0)
    model = _build(name)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = AdaFisherMulti(model, lr=1e-2, TCov=1, fisher_mode=mode, **_MODE_KWARGS[mode])
    batch = _synthetic_batch(name)
    for _ in range(n_steps):
        inputs, targets = bench.prepare_batch(batch)
        optimizer.zero_grad()
        bench.loss_fn(model(inputs), targets).backward()
        optimizer.step()
    return model, before


def _model_params():
    """``resnet50_cifar`` is marked slow (``plan_exp_step1.md`` §6): 23.5 M parameters times five
    modes is minutes, not seconds. Run it with ``pytest --runslow``.
    """
    return [pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n for n in MODEL_IDS]


# ----------------------------------------------------------------------------------------------
# Architecture contract
# ----------------------------------------------------------------------------------------------


def test_every_model_folder_is_covered() -> None:
    assert set(BENCHES) == set(EXPECTED), (
        "a benchmarks/<model>/bench.py exists with no entry in EXPECTED (or the reverse); "
        "adding a model means adding its row here"
    )


@pytest.mark.parametrize("name", MODEL_IDS)
def test_model_builds_with_expected_parameter_count(name: str) -> None:
    seed_all(0)
    model = _build(name)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED[name].params, f"{name}: {n_params} parameters"
    inputs, _ = _synthetic_batch(name)
    inputs, _targets = BENCHES[name].prepare_batch((inputs, torch.zeros(inputs.size(0)).long()))
    assert model(inputs).shape[0] == inputs.shape[0]


@pytest.mark.parametrize("name", MODEL_IDS)
def test_hooked_module_inventory(name: str) -> None:
    seed_all(0)
    counts, unhooked = _hooked_inventory(_build(name))
    assert counts == EXPECTED[name].counts
    assert unhooked == EXPECTED[name].unhooked


# ----------------------------------------------------------------------------------------------
# The plan_lot8.md §0.3 regression, generalized to every model
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", _model_params())
def test_every_parameter_is_updated(name: str) -> None:
    """After 2 real ``AdaFisherMulti`` steps no parameter may be bit-identical to its init. This
    catches unhooked parameters silently skipped by the step loop (``plan_lot8.md`` §0.3), in-place
    mutation breaking a full backward hook, and pairing bugs — including CCT's sequence pooling
    and the ViT class token.
    """
    model, before = _one_arm_steps(name, "diag")
    unchanged = [n for n, p in model.named_parameters() if torch.equal(p.detach(), before[n])]
    assert unchanged == [], f"{name}: {len(unchanged)} parameter(s) never updated: {unchanged}"
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize("name", _model_params())
@pytest.mark.parametrize("mode", list(_MODE_KWARGS))
def test_all_modes_run(name: str, mode: str) -> None:
    model, _ = _one_arm_steps(name, mode)
    assert all(torch.isfinite(p).all() for p in model.parameters()), f"{name}/{mode}"


# ----------------------------------------------------------------------------------------------
# Bench specification (plan_exp_step1.md D3)
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODEL_IDS)
def test_bench_spec_is_wellformed(name: str) -> None:
    bench = BENCHES[name]
    assert bench.name == name, "Benchmark.name must equal its folder name (it is the output dir)"
    assert bench.arms and set(bench.arms) <= set(ARMS)
    assert bench.epochs > 0 and bench.batch_size > 0
    # The runner calls build_data with exactly these keywords (runner.run_arm).
    inspect.signature(bench.build_data).bind(
        "some/data/root", batch_size=8, seed=0, num_workers=0, cutout=True,
        allow_download=False, train_subset=16,
    )
    # ... and build_model with the keywords the bench declares, and nothing else.
    inspect.signature(bench.build_model).bind(**{k: v[0] for k, v in bench.model_choices.items()})


def test_bench_py_files_are_declarative() -> None:
    """``plan_exp_step1.md`` D3 / exit criterion 3: a ``bench.py`` is a ``Benchmark`` literal plus
    ``main(BENCH)`` — at most 50 lines, and no control flow beyond the ``__main__`` guard.
    """
    for name in MODEL_IDS:
        path = REPO_ROOT / "benchmarks" / name / "bench.py"
        lines = path.read_text().splitlines()
        assert len(lines) <= 50, f"{name}/bench.py is {len(lines)} lines"
        body = [ln for ln in lines if not ln.lstrip().startswith("#")]
        for keyword in ("for ", "while ", "def ", "class ", "elif ", "try:"):
            assert not any(ln.lstrip().startswith(keyword) for ln in body), (
                f"{name}/bench.py contains control flow ({keyword.strip()!r})"
            )
        assert sum(ln.startswith("if ") for ln in body) == 1  # the __main__ guard only


# ----------------------------------------------------------------------------------------------
# Checkpointing (plan_exp_step1.md D5, exit criterion 4)
# ----------------------------------------------------------------------------------------------


def test_checkpoints_are_written_on_schedule_and_reload(tmp_path: Path) -> None:
    """``{0, 1%, 10%, 50%, 100%}`` of a run's steps, each reloading into a *fresh* model — the
    input steps 2-5 of ``plan_exp_draft.md`` consume.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from benchmarks.common.checkpoints import CheckpointWriter, parse_fractions
    from benchmarks.common.loop import train_under_budget

    fractions = parse_fractions("0,0.01,0.1,0.5,1")
    assert fractions == (0.0, 0.01, 0.1, 0.5, 1.0)

    seed_all(0)
    model = _build("mlp_ln_mnist")
    loader = DataLoader(
        TensorDataset(torch.randn(80, 1, 28, 28), torch.randint(0, 10, (80,))), batch_size=8
    )
    total_steps = 5 * len(loader)
    writer = CheckpointWriter(tmp_path, fractions, total_steps=total_steps, seed=0)
    optimizer = AdaFisherMulti(model, lr=1e-2, TCov=1, fisher_mode="diag")
    steps, _ = train_under_budget(
        model, optimizer, loader, nn.CrossEntropyLoss(), budget_s=1e6, max_epochs=5,
        on_step=writer, log_fn=lambda _: None,
    )
    assert len(steps) == total_steps
    writer.save_final(len(steps), 4, model)

    written = sorted((tmp_path / f"ckpt_{label}.pt").name for label in writer.written)
    assert written == ["ckpt_0.01.pt", "ckpt_0.1.pt", "ckpt_0.5.pt", "ckpt_0.pt", "ckpt_1.pt"]
    fresh = _build("mlp_ln_mnist")
    for fraction in fractions:
        payload = torch.load(tmp_path / f"ckpt_{fraction:g}.pt", weights_only=False)
        assert payload["step"] == round(fraction * total_steps)
        assert payload["seed"] == 0 and payload["fraction"] == fraction
        # Self-describing payload: the denominator, and whether it landed on schedule.
        assert payload["total_steps"] == total_steps
        assert payload["scheduled"] is True
        fresh.load_state_dict(payload["model_state_dict"])  # reloads into a fresh model
    # t=0 is the initialization and t=1 the final state — they must differ.
    initial = torch.load(tmp_path / "ckpt_0.pt", weights_only=False)["model_state_dict"]
    final = torch.load(tmp_path / "ckpt_1.pt", weights_only=False)["model_state_dict"]
    assert not torch.equal(initial["head.weight"], final["head.weight"])


def test_save_final_pins_the_last_fraction_under_a_wall_clock_budget(tmp_path: Path) -> None:
    """Under ``--budget-mode wct`` the final step count is unknown in advance, so the schedule's
    largest fraction cannot fire from ``on_step``; ``save_final`` pins it to wherever the run
    actually ended (``common/checkpoints.py``).
    """
    from benchmarks.common.checkpoints import CheckpointWriter

    seed_all(0)
    model = _build("mlp_ln_mnist")
    writer = CheckpointWriter(tmp_path, (0.0, 1.0), total_steps=10_000, seed=3)
    writer(0, 0, model)
    assert sorted(writer.written) == ["0"]
    writer.save_final(37, 2, model)
    assert sorted(writer.written) == ["0", "1"]
    payload = torch.load(tmp_path / "ckpt_1.pt", weights_only=False)
    assert (payload["step"], payload["epoch"], payload["seed"]) == (37, 2, 3)
    # Pinned to the arm's own end, not to the nominal 100% point — the consumer must be able to
    # tell the two apart.
    assert payload["scheduled"] is False and payload["total_steps"] == 10_000


def test_checkpoint_fractions_share_one_denominator_across_arms(tmp_path: Path) -> None:
    """The whole runner, offline, on a synthetic task: under ``--budget-mode wct`` the reference arm
    and a budgeted arm must schedule their checkpoints against the **same** nominal denominator, so
    ``ckpt_0.5`` means the same amount of training in both.

    Regression: the denominator used to be ``max_epochs * batches``, i.e. ``--max-epoch-factor``
    times too long for every budgeted arm — putting its ``ckpt_0.1`` at ~31% of its own run and
    making ``ckpt_0.5`` unreachable. Measured on a real ``cct_2_3x2_cifar`` run before the fix.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from benchmarks.common.optimizers import HParams
    from benchmarks.common.runner import main as run_main

    epochs, n, batch = 4, 64, 8

    def build_data(data_root, *, batch_size=8, seed=0, num_workers=0, cutout=True,
                   allow_download=False, train_subset=None):
        seed_all(0)
        dataset = TensorDataset(torch.randn(n, 6), torch.randint(0, 3, (n,)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return loader, None, None

    bench = Benchmark(
        name="synthetic",
        build_model=lambda: nn.Sequential(nn.Linear(6, 5), nn.Sigmoid(), nn.Linear(5, 3)),
        build_data=build_data,
        loss_fn=nn.CrossEntropyLoss(),
        hparams=HParams(lr=1e-2, tcov=1),
        epochs=epochs,
        batch_size=batch,
        arms=("diag", "adam"),
    )
    run_main(bench, argv=["--output-dir", str(tmp_path), "--checkpoints", "0,0.5,1",
                          "--budget-mode", "wct", "--device", "cpu", "--no-plot",
                          "--num-workers", "0"])

    nominal = epochs * (n // batch)  # 4 epochs x 8 batches = 32 steps, identical for both arms
    for arm in ("diag", "adam"):
        for fraction in (0.0, 0.5, 1.0):
            payload = torch.load(tmp_path / arm / f"ckpt_{fraction:g}.pt", weights_only=False)
            assert payload["total_steps"] == nominal, (arm, fraction)
            if payload["scheduled"]:
                assert payload["step"] == round(fraction * nominal), (arm, fraction)
    # The comparability that matters: both arms' 50% point is the same step count.
    diag_half = torch.load(tmp_path / "diag" / "ckpt_0.5.pt", weights_only=False)
    adam_half = torch.load(tmp_path / "adam" / "ckpt_0.5.pt", weights_only=False)
    assert diag_half["step"] == adam_half["step"] == nominal // 2
