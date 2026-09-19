"""Step 1 (``docs/reports/plan_exp_step1.md`` §6): the one test that makes adding a model safe.

Parametrized over every ``benchmarks/models/<model>/bench.py`` the folder registry discovers, so a new
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
    """``(parameters, hooked-module counts, unhooked parameter names, synthetic input shape,
    output width)``.

    ``out_features`` is the width of what the model emits — the number of classes for a
    classifier, the flattened image for the auto-encoder. Keyword-only and without a default, so a
    new model folder cannot be added without stating it.
    """

    def __init__(self, params: int, counts: Dict[str, int], unhooked: List[str],
                 input_shape: Tuple[int, ...], *, out_features: int) -> None:
        self.params, self.counts, self.unhooked, self.input_shape = (
            params, counts, unhooked, input_shape
        )
        self.out_features = out_features


EXPECTED: Dict[str, Expected] = {
    # plan_exp_step1.md §4, "migrated" rows.
    "mnist_autoencoder": Expected(2_837_314, {"Linear": 8}, [], (1, 28, 28), out_features=784),
    "resnet50_cifar": Expected(
        # §4 transcribes 23 519 178; the model — migrated verbatim from lot 8's cifar10_models.py,
        # not modified here — has 23 520 842. plan_lot8.md's own "23.52 M" is the correct rounding.
        23_520_842, {"Conv2d": 53, "BatchNorm2d": 53, "Linear": 1}, [], (3, 32, 32),
        out_features=10,
    ),
    "vit_small_cifar": Expected(
        # §4 transcribes 2 685 898; the verbatim-migrated model has 2 693 578 ("2.69 M" either way).
        2_693_578, {"Conv2d": 1, "LayerNorm": 13, "Linear": 25}, ["cls_token", "pos_embed"],
        (3, 32, 32),
        out_features=10,
    ),
    # plan_exp_step1.md §4, "new" rows — A1, A2, A3, B2, B1.
    "mlp_ln_mnist": Expected(26_634, {"Linear": 3, "LayerNorm": 2}, [], (1, 28, 28),
                             out_features=10),
    "cnn_gn_cifar": Expected(
        24_458, {"Conv2d": 3, "Linear": 1},
        # GroupNorm is deliberately not a SUPPORTED_MODULES type (see the model's docstring).
        ["features.1.weight", "features.1.bias", "features.5.weight", "features.5.bias",
         "features.9.weight", "features.9.bias"],
        (3, 32, 32),
        out_features=10,
    ),
    "vit_micro_cifar": Expected(
        # §4 estimates "≈ 21 162 [DERIVED] — the exact value is fixed by the config and locked by
        # the test once built". Mean pooling removes the cls_token (32) and its position row (32).
        21_098, {"Conv2d": 1, "LayerNorm": 5, "Linear": 9}, ["pos_embed"], (3, 32, 32),
        out_features=10,
    ),
    "resnet20_cifar": Expected(
        269_722, {"Conv2d": 19, "BatchNorm2d": 19, "Linear": 1}, [], (3, 32, 32),
        out_features=10,
    ),
    "cct_2_3x2_cifar": Expected(
        283_723, {"Conv2d": 2, "LayerNorm": 5, "Linear": 10}, ["pos_embed"], (3, 32, 32),
        out_features=10,
    ),
    # ------------------------------------------------------------------------------------------
    # CIFAR-100: the six CIFAR-10 architectures, unchanged apart from a 100-way head. The hooked
    # inventories and unhooked lists are therefore identical to their CIFAR-10 rows above, and the
    # parameter deltas are exactly ``90 * (embed_dim + 1)`` for the head — the cheapest possible
    # check that nothing else moved with the dataset.
    # ------------------------------------------------------------------------------------------
    "cnn_gn_cifar100": Expected(
        30_308, {"Conv2d": 3, "Linear": 1},
        ["features.1.weight", "features.1.bias", "features.5.weight", "features.5.bias",
         "features.9.weight", "features.9.bias"],
        (3, 32, 32),
        out_features=100,
    ),
    "vit_micro_cifar100": Expected(
        24_068, {"Conv2d": 1, "LayerNorm": 5, "Linear": 9}, ["pos_embed"], (3, 32, 32),
        out_features=100,
    ),
    "resnet20_cifar100": Expected(
        275_572, {"Conv2d": 19, "BatchNorm2d": 19, "Linear": 1}, [], (3, 32, 32),
        out_features=100,
    ),
    "cct_2_3x2_cifar100": Expected(
        295_333, {"Conv2d": 2, "LayerNorm": 5, "Linear": 10}, ["pos_embed"], (3, 32, 32),
        out_features=100,
    ),
    "resnet50_cifar100": Expected(
        23_705_252, {"Conv2d": 53, "BatchNorm2d": 53, "Linear": 1}, [], (3, 32, 32),
        out_features=100,
    ),
    "vit_small_cifar100": Expected(
        2_710_948, {"Conv2d": 1, "LayerNorm": 13, "Linear": 25}, ["cls_token", "pos_embed"],
        (3, 32, 32),
        out_features=100,
    ),
    # ------------------------------------------------------------------------------------------
    # ImageNet-1K. Two resolutions, and ``input_shape`` is what distinguishes them: the four
    # 32x32-native architectures run on downsampled ImageNet (``imagenet32``) and are again
    # structurally unchanged, while ResNet-50 and ViT-S run at the native 224 px in their own
    # ImageNet configurations — the 7x7/stride-2 + max-pool stem of resnet_1512.03385.pdf Table 1
    # (25 557 032 parameters, the standard ResNet-50 count) and patch-16 / D=384 / depth 12
    # (22 050 664). ``vit_small_imagenet``'s 25 LayerNorms and 49 Linears against
    # ``vit_small_cifar``'s 13 and 25 are depth 12 against depth 6.
    # ------------------------------------------------------------------------------------------
    "cnn_gn_imagenet": Expected(
        88_808, {"Conv2d": 3, "Linear": 1},
        ["features.1.weight", "features.1.bias", "features.5.weight", "features.5.bias",
         "features.9.weight", "features.9.bias"],
        (3, 32, 32),
        out_features=1000,
    ),
    "vit_micro_imagenet": Expected(
        53_768, {"Conv2d": 1, "LayerNorm": 5, "Linear": 9}, ["pos_embed"], (3, 32, 32),
        out_features=1000,
    ),
    "resnet20_imagenet": Expected(
        334_072, {"Conv2d": 19, "BatchNorm2d": 19, "Linear": 1}, [], (3, 32, 32),
        out_features=1000,
    ),
    "cct_2_3x2_imagenet": Expected(
        411_433, {"Conv2d": 2, "LayerNorm": 5, "Linear": 10}, ["pos_embed"], (3, 32, 32),
        out_features=1000,
    ),
    "resnet50_imagenet": Expected(
        25_557_032, {"Conv2d": 53, "BatchNorm2d": 53, "Linear": 1}, [], (3, 224, 224),
        out_features=1000,
    ),
    "vit_small_imagenet": Expected(
        22_050_664, {"Conv2d": 1, "LayerNorm": 25, "Linear": 49}, ["cls_token", "pos_embed"],
        (3, 224, 224),
        out_features=1000,
    ),
}

#: Marked slow (``plan_exp_step1.md`` §6): 20 M+ parameters times five modes, and for the two
#: ImageNet-224 rows a 224x224 forward on top. Run them with ``pytest --runslow``.
SLOW = {"resnet50_cifar", "resnet50_cifar100", "resnet50_imagenet", "vit_small_imagenet"}
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
        "a benchmarks/models/<model>/bench.py exists with no entry in EXPECTED (or the reverse); "
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
    # The batch axis AND the head width. ``shape[0] == inputs.shape[0]`` alone compares the batch
    # axis with itself and cannot fail, so six models had no output-shape assertion at all.
    with torch.no_grad():
        assert model(inputs).shape == (inputs.shape[0], EXPECTED[name].out_features), name


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
    """Every mode on every model: finite *and* moving.

    This is 100 of this file's items and it used to assert finiteness only — which a mode that
    silently updated nothing would pass, since the initial weights are finite too. The "no
    parameter left un-updated" check above ran with ``fisher_mode="diag"`` alone, so the four
    Kronecker modes were never checked for it on any model. They are now: measured across all 20
    model folders x 5 modes, zero parameters stay bit-identical to their initial value after two
    real steps.
    """
    model, before = _one_arm_steps(name, mode)
    assert all(torch.isfinite(p).all() for p in model.parameters()), f"{name}/{mode}"
    unchanged = [n for n, p in model.named_parameters() if torch.equal(p.detach(), before[n])]
    assert unchanged == [], (
        f"{name}/{mode}: {len(unchanged)} parameter(s) never updated: {unchanged[:8]}"
    )


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
        path = REPO_ROOT / "benchmarks" / "models" / name / "bench.py"
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


# ----------------------------------------------------------------------------------------------
# --checkpoint-optimizer-state (opt-in; plan_exp_draft.md §3.2, the P2 re-warm alternative)
# ----------------------------------------------------------------------------------------------


DEFAULT_PAYLOAD_KEYS = {"model_state_dict", "fraction", "step", "epoch", "seed", "total_steps",
                        "scheduled"}


def _write_one(tmp_path: Path, arm: str, *, save_state: bool, model_name: str = "mlp_ln_mnist"):
    """One checkpoint of one arm, through the real writer, after 2 real steps."""
    from benchmarks.common.checkpoints import CheckpointWriter
    from benchmarks.common.optimizers import HParams, build_optimizer

    bench = BENCHES[model_name]
    seed_all(0)
    model = _build(model_name)
    optimizer = build_optimizer(arm, model, HParams(tcov=1, t_inv=1, t_eig=1))
    batch = _synthetic_batch(model_name)
    for _ in range(2):
        inputs, targets = bench.prepare_batch(batch)
        optimizer.zero_grad()
        bench.loss_fn(model(inputs), targets).backward()
        optimizer.step()
    writer = CheckpointWriter(tmp_path / arm, (1.0,), total_steps=2, seed=0,
                              optimizer=optimizer, save_optimizer_state=save_state)
    writer.save(1.0, 2, 0, model)
    return model, optimizer, torch.load(tmp_path / arm / "ckpt_1.pt", weights_only=False)


def test_checkpoint_optimizer_state_is_off_by_default(tmp_path: Path) -> None:
    """The default payload must be exactly what it was before the flag existed: a run in flight
    picks up this code from ``$SLURM_SUBMIT_DIR``, so "off" has to mean *byte-for-byte unchanged*.
    """
    _model, _opt, payload = _write_one(tmp_path, "ekfac", save_state=False)
    assert set(payload) == DEFAULT_PAYLOAD_KEYS
    from benchmarks.common.checkpoints import CheckpointWriter
    writer = CheckpointWriter(tmp_path / "plain", (1.0,), total_steps=2, seed=0)
    assert writer.optimizer is None and writer.save_optimizer_state is False


@pytest.mark.parametrize("arm", ["diag", "kfac", "ekfac", "tkfac", "tekfac"])
def test_checkpoint_optimizer_state_round_trips_by_module_name(tmp_path: Path, arm: str) -> None:
    """The live Fisher state is keyed by ``nn.Module`` objects, which do not survive a reload, so
    it is re-keyed by module name — and every hooked module must be present, for every mode.
    """
    model, optimizer, payload = _write_one(tmp_path, arm, save_state=True)
    state = payload["optimizer_state"]
    assert state["optimizer_class"] == "AdaFisherMulti"
    assert state["optimizer_steps"] == 2
    assert "exp_avg" in next(iter(state["optimizer_state_dict"]["state"].values()))

    hooked = {name for name, module in model.named_modules() if module in optimizer.modules}
    assert hooked, "the test model must have hooked modules"
    fisher = state["fisher_state"]
    assert fisher, f"{arm}: no Fisher factor was dumped"
    for family, by_name in fisher.items():
        assert not family.startswith("_cached"), f"{family} is a batch, not state"
        assert set(by_name) <= hooked
        assert all(torch.is_tensor(t) for t in by_name.values())
    # Every hooked module appears in at least one family, and the values are the live ones.
    assert set().union(*fisher.values()) == hooked
    live = vars(optimizer.approx)
    for family, by_name in fisher.items():
        for name, tensor in by_name.items():
            module = dict(model.named_modules())[name]
            assert torch.equal(tensor, live[family][module].cpu())


def test_checkpoint_optimizer_state_for_the_baselines(tmp_path: Path) -> None:
    """``adam``/``adamw`` have no ``approx``: they contribute their ``state_dict`` alone, with no
    branch per arm in the writer (it is duck-typed).
    """
    _model, _opt, payload = _write_one(tmp_path, "adam", save_state=True)
    state = payload["optimizer_state"]
    assert state["optimizer_class"] == "Adam" and "fisher_state" not in state
    assert "exp_avg_sq" in next(iter(state["optimizer_state_dict"]["state"].values()))


def test_checkpoint_optimizer_state_excludes_the_transient_batch_cache(tmp_path: Path) -> None:
    """``ekfac``/``tekfac`` cache every layer's input batch between the forward and the backward
    hook (3.37 GB across ResNet-50 at batch 128, ``plan_lot8.md`` §0.4). It is a batch, not state.

    Between two completed steps the cache happens to be empty — the backward hook ``pop``s it — so
    the filter is checked on an entry injected deliberately, not on that timing.
    """
    from benchmarks.common.checkpoints import TRANSIENT_PREFIXES, optimizer_state_payload

    model, optimizer, payload = _write_one(tmp_path, "ekfac", save_state=True)
    assert not any(k.startswith(TRANSIENT_PREFIXES) for k in payload["optimizer_state"]
                   ["fisher_state"])

    hooked = optimizer.modules[0]
    optimizer.approx._cached_h_bar[hooked] = torch.zeros(4096, 4096)  # one live cached batch
    assert any(v for k, v in vars(optimizer.approx).items() if k.startswith(TRANSIENT_PREFIXES))
    fisher = optimizer_state_payload(optimizer, model)["fisher_state"]
    assert not any(k.startswith(TRANSIENT_PREFIXES) for k in fisher)


def test_checkpoint_optimizer_state_reaches_the_fisher_ref_bridge(tmp_path: Path) -> None:
    """The campaign's own reader exposes it, and still reads a checkpoint written without it."""
    import json

    from fisher_ref.checkpoints import discover_runs, load_theta

    outputs = tmp_path / "outputs" / "mlp_ln_mnist"
    _write_one(outputs, "tekfac", save_state=True)
    _write_one(outputs, "diag", save_state=False)
    (outputs / "manifest.json").write_text(json.dumps({"config": {"seed": 0}}))

    loaded = {run.arm: load_theta(run, 1.0) for run in discover_runs(tmp_path / "outputs")}
    assert loaded["diag"].optimizer_state is None
    assert loaded["diag"].metadata()["has_optimizer_state"] is False
    assert loaded["tekfac"].optimizer_state is not None
    assert loaded["tekfac"].metadata()["has_optimizer_state"] is True
    assert set(loaded["tekfac"].optimizer_state["fisher_state"]["_Theta"]) == {
        "features.0", "features.1", "features.3", "features.4", "head"
    }
