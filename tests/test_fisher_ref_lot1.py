"""Lot 1 of the Fisher-drift campaign: the exact references (``docs/reports/plan_exp_draft.md``
§9's lot 1, exit criterion "T1, T2, T6 pass"; ``plan_exp_lot1.md`` §2).

``fisher_ref`` stays a **reader**: nothing in ``benchmarks/`` or ``src/adafisher_modes/`` changes.
Everything here is offline, CPU and fp64 — the exactness tolerances (``1e-10`` on T1) need fp64 end
to end, which is why lot 1 ignores ``conventions.CAPTURE_DTYPE`` and takes its storage dtype as an
argument instead (``plan_exp_lot1.md`` §0.3).

The T1 oracle is deliberately written here rather than shipped in ``fisher_ref``: it shares **no**
code with the root-based build (no ``sources.output_root``, no ``capture``), which is the only
property that makes the comparison worth anything. ``reference/matfree.py`` is lot 6's.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from conftest import TinyMultiLayerNet, seed_all
from torch import Tensor
from torch.func import functional_call, grad, jvp, vjp, vmap

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from test_benchmark_models import EXPECTED as MODEL_CONTRACT  # noqa: E402

from benchmarks.common.runner import Benchmark, discover_benchmarks  # noqa: E402
from fisher_ref import capture, conventions, registry, sources  # noqa: E402
from fisher_ref.reference import ParamLayout, build_dense_reference, symmetrize_  # noqa: E402

BENCHES: Dict[str, Benchmark] = discover_benchmarks()
SLOW = {"resnet50_cifar", "resnet50_cifar100", "resnet50_imagenet", "vit_small_imagenet"}
DTYPE = torch.float64

#: Parameters belonging to no capturable module, per model — the raw ``nn.Parameter`` of the
#: transformer families. A ``GroupNorm`` affine is **not** here even though ``AdaFisherMulti`` does
#: not hook it: the campaign measures curvature, not what the optimizer preconditions
#: (``plan_exp_lot0.md`` §0.6).
EXPECTED_UNCOVERED: Dict[str, Tuple[str, ...]] = {
    "mnist_autoencoder": (),
    "mlp_ln_mnist": (),
    "cnn_gn_cifar": (),
    "vit_micro_cifar": ("pos_embed",),
    "resnet20_cifar": (),
    "cct_2_3x2_cifar": ("pos_embed",),
    "vit_small_cifar": ("cls_token", "pos_embed"),
    "resnet50_cifar": (),
    # CIFAR-100 / ImageNet: same architectures, so the same raw Parameters (or none).
    "cnn_gn_cifar100": (),
    "vit_micro_cifar100": ("pos_embed",),
    "resnet20_cifar100": (),
    "cct_2_3x2_cifar100": ("pos_embed",),
    "vit_small_cifar100": ("cls_token", "pos_embed"),
    "resnet50_cifar100": (),
    "cnn_gn_imagenet": (),
    "vit_micro_imagenet": ("pos_embed",),
    "resnet20_imagenet": (),
    "cct_2_3x2_imagenet": ("pos_embed",),
    "resnet50_imagenet": (),
    "vit_small_imagenet": ("cls_token", "pos_embed"),
}


def _model_ids() -> List[object]:
    return [pytest.param(name, marks=pytest.mark.slow) if name in SLOW else name
            for name in sorted(EXPECTED_UNCOVERED)]


def _tiny_classifier() -> nn.Module:
    seed_all(0)
    return nn.Sequential(nn.Linear(6, 5), nn.LayerNorm(5), nn.ReLU(), nn.Linear(5, 4)).to(DTYPE)


def _tiny_batch(n: int = 12, classes: int = 4) -> Tuple[Tensor, Tensor]:
    seed_all(3)
    return torch.randn(n, 6, dtype=DTYPE), torch.randint(0, classes, (n,))


# ----------------------------------------------------------------------------------------------
# sources.py — the output-space roots
# ----------------------------------------------------------------------------------------------


def test_type2_root_reconstructs_lambda_and_has_rank_c_minus_one() -> None:
    """``S S^T = diag(p) - p p^T`` exactly, and ``sqrt(p)`` is the null vector: ``C`` columns for a
    rank of ``C - 1``, which is why ``plan_exp_draft.md`` §2.2-§2.3's ``m = N(C-1)`` is a **rank**
    and ``U`` has ``N*C`` rows (``plan_exp_lot1.md`` §0.2).
    """
    seed_all(0)
    logits = torch.randn(7, 5, dtype=DTYPE)
    probabilities = torch.softmax(logits, dim=-1)
    root = sources.output_root(logits, torch.zeros(7, dtype=torch.long), source="type2")

    assert root.n_columns == 5
    for n in range(7):
        columns = root.columns[n]                       # (C, C): row c is the column S[:, c]
        lambda_exact = torch.diag(probabilities[n]) - torch.outer(probabilities[n],
                                                                  probabilities[n])
        assert torch.allclose(columns.T @ columns, lambda_exact, atol=1e-14)
        assert torch.allclose(columns.T @ probabilities[n].sqrt(),
                              torch.zeros(5, dtype=DTYPE), atol=1e-14)


def test_mc_root_needs_its_k_inverse_sqrt_to_converge_to_lambda() -> None:
    """Without the ``K^{-1/2}`` of *KFAC from scratch*'s own errata, the MC source does not converge
    to ``Lambda`` at all (it is off by a factor ``K``).
    """
    seed_all(0)
    logits = torch.randn(1, 4, dtype=DTYPE)
    probabilities = torch.softmax(logits, dim=-1)[0]
    lambda_exact = torch.diag(probabilities) - torch.outer(probabilities, probabilities)

    generator = torch.Generator().manual_seed(0)
    root = sources.output_root(logits, torch.zeros(1, dtype=torch.long), source="mc", k=20_000,
                               generator=generator)
    estimate = root.columns[0].T @ root.columns[0]
    assert torch.linalg.matrix_norm(estimate - lambda_exact) < 0.02
    unscaled = estimate * 20_000
    assert torch.linalg.matrix_norm(unscaled - lambda_exact) > 1.0


def test_empirical_root_is_the_per_sample_loss_gradient() -> None:
    """``p - e_y`` for cross entropy and ``(2/D)(f - y)`` for MSE: the per-sample loss, never the
    batch mean, and the loss's own ``1/D`` sits inside ``Lambda`` (``sources.py``'s convention 1).
    """
    seed_all(0)
    logits = torch.randn(5, 4, dtype=DTYPE, requires_grad=True)
    targets = torch.randint(0, 4, (5,))
    for n in range(5):
        loss = F.cross_entropy(logits[n:n + 1], targets[n:n + 1])
        (gradient,) = torch.autograd.grad(loss, logits, retain_graph=True)
        root = sources.output_root(logits.detach(), targets, source="empirical")
        assert torch.allclose(root.columns[n, 0], gradient[n], atol=1e-14)

    outputs = torch.randn(5, 7, dtype=DTYPE, requires_grad=True)
    reconstruction = torch.randn(5, 7, dtype=DTYPE)
    for n in range(5):
        loss = F.mse_loss(outputs[n:n + 1], reconstruction[n:n + 1])
        (gradient,) = torch.autograd.grad(loss, outputs, retain_graph=True)
        root = sources.output_root(outputs.detach(), reconstruction, source="empirical", loss="mse")
        assert torch.allclose(root.columns[n, 0], gradient[n], atol=1e-14)


def test_mse_type2_root_is_the_canonical_link_lambda() -> None:
    """``Lambda = (2/D) I`` for ``nn.MSELoss``'s per-sample loss, so ``F = GGN`` (Martens
    arXiv:1412.1193 §9) — the one built model whose output factor is not the softmax root.
    """
    seed_all(0)
    outputs = torch.randn(3, 6, dtype=DTYPE)
    root = sources.output_root(outputs, outputs, source="type2", loss="mse")
    assert root.n_columns == 6
    columns = root.columns[0]
    assert torch.allclose(columns.T @ columns, (2.0 / 6.0) * torch.eye(6, dtype=DTYPE), atol=1e-14)


def test_loss_kind_is_read_off_the_bench() -> None:
    assert sources.loss_kind(BENCHES["mlp_ln_mnist"]) == "cross_entropy"
    assert sources.loss_kind(BENCHES["mnist_autoencoder"]) == "mse"


def test_mc_source_is_refused_for_mse() -> None:
    with pytest.raises(NotImplementedError, match="categorical"):
        sources.output_root(torch.randn(2, 3, dtype=DTYPE), torch.randn(2, 3, dtype=DTYPE),
                            source="mc", loss="mse", k=4)


# ----------------------------------------------------------------------------------------------
# capture.py — per-sample gradients
# ----------------------------------------------------------------------------------------------


class _SharedLinearNet(nn.Module):
    """A ``Linear`` applied at several token positions, the ``qkv`` situation of A3/B1/B4."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3)
        self.head = nn.Linear(3, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(torch.tanh(self.proj(x)).mean(dim=1))


class _GroupNormNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3, padding=1)
        self.norm = nn.GroupNorm(2, 4)
        self.head = nn.Linear(4 * 5 * 5, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(torch.tanh(self.norm(self.conv(x))).flatten(1))


def _norm_net_cases() -> List[Tuple[str, nn.Module, Tensor]]:
    seed_all(0)
    tiny = TinyMultiLayerNet().to(DTYPE)
    tiny.bn.running_mean.normal_()
    tiny.bn.running_var.uniform_(0.5, 1.5)
    return [
        ("tiny_all_kinds", tiny, torch.randn(6, 2, 5, 5, dtype=DTYPE)),
        ("shared_linear", _SharedLinearNet().to(DTYPE), torch.randn(5, 7, 4, dtype=DTYPE)),
        ("group_norm", _GroupNormNet().to(DTYPE), torch.randn(4, 2, 5, 5, dtype=DTYPE)),
    ]


@pytest.mark.parametrize("label,model,inputs", _norm_net_cases(), ids=lambda case: case)
def test_capture_reconstructs_parameter_gradients(label: str, model: nn.Module,
                                                  inputs: Tensor) -> None:
    """``sum_n`` of the per-sample gradients is autograd's own gradient, for every layer kind:
    ``Linear`` (unshared and shared over positions), ``Conv2d``, ``BatchNorm2d``-eval, ``LayerNorm``
    and ``GroupNorm``.
    """
    seed_all(1)
    with conventions.reference_mode(model), capture.capture(model) as capturer:
        outputs = model(inputs)
        cotangent = torch.randn_like(outputs)
        torch.autograd.backward(outputs, cotangent)
        reconstructed = {
            f"{layer.name}.{p_name}": gradient
            for layer in capturer
            for p_name, gradient in capture.per_sample_gradients(layer).items()
        }

    named = dict(model.named_parameters())
    assert set(reconstructed) == set(named), "a parameter was left without a per-sample gradient"
    for name, per_sample in reconstructed.items():
        assert per_sample.shape[1:] == named[name].shape
        assert torch.allclose(per_sample.sum(0), named[name].grad, atol=1e-11, rtol=0), name


def _vmap_per_sample_gradients(model: nn.Module, inputs: Tensor,
                               cotangent: Tensor) -> Dict[str, Tensor]:
    """Per-sample gradients of ``<v_n, f(x_n)>`` through ``torch.func`` — an oracle that knows
    nothing about hooks, layer kinds or ``x_hat``.
    """
    parameters = {name: value.detach() for name, value in model.named_parameters()}
    buffers = {name: value.detach() for name, value in model.named_buffers()}

    def scalar(params: Dict[str, Tensor], one_input: Tensor, one_cotangent: Tensor) -> Tensor:
        output = functional_call(model, (params, buffers), (one_input.unsqueeze(0),))
        return (output.squeeze(0) * one_cotangent).sum()

    return vmap(grad(scalar), in_dims=(None, 0, 0))(parameters, inputs, cotangent)


@pytest.mark.parametrize("label,model,inputs", _norm_net_cases(), ids=lambda case: case)
def test_capture_matches_vmap_per_sample_gradients(label: str, model: nn.Module,
                                                   inputs: Tensor) -> None:
    """Per-sample agreement, not just the sum — the sum can be right while individual rows of ``U``
    are wrong, and it is the rows that build ``F``.
    """
    seed_all(2)
    with conventions.reference_mode(model), capture.capture(model) as capturer:
        outputs = model(inputs)
        cotangent = torch.randn_like(outputs)
        torch.autograd.backward(outputs, cotangent)
        reconstructed = {
            f"{layer.name}.{p_name}": gradient
            for layer in capturer
            for p_name, gradient in capture.per_sample_gradients(layer).items()
        }
    # Outside the capture: the oracle runs its own forwards, which the reuse guard would refuse.
    with conventions.reference_mode(model):
        oracle = _vmap_per_sample_gradients(model, inputs, cotangent)

    for name, per_sample in reconstructed.items():
        assert torch.allclose(per_sample, oracle[name], atol=1e-11, rtol=0), name


def test_backward_hook_pruning_regression() -> None:
    """``plan_exp_lot1.md`` §0.1, locked: ``register_full_backward_hook`` fires from the module's
    *input*-side node, so with ``inputs=params`` and an input that requires grad the **first**
    module is pruned out and its hook silently never fires. On ``mlp_ln_mnist`` that module holds
    25 120 of 26 634 parameters. ``Capture``'s tensor hook does not have the failure mode.
    """
    model = _tiny_classifier()
    fired: List[int] = []
    handles = [module.register_full_backward_hook(lambda mod, gi, go, i=index: fired.append(i))
               for index, module in enumerate(model) if list(module.parameters(recurse=False))]
    try:
        inputs = torch.randn(5, 6, dtype=DTYPE, requires_grad=True)
        outputs = model(inputs)
        cotangent = torch.randn_like(outputs)
        torch.autograd.grad(outputs, list(model.parameters()), grad_outputs=cotangent)
    finally:
        for handle in handles:
            handle.remove()
    assert 0 not in fired, "the pruning this design avoids no longer happens — revisit capture.py"
    assert 3 in fired

    with conventions.reference_mode(model), capture.capture(model) as capturer:
        inputs = torch.randn(5, 6, dtype=DTYPE, requires_grad=True)
        outputs = model(inputs)
        torch.autograd.grad(outputs, [inputs], grad_outputs=torch.randn_like(outputs))
        captured = {layer.name for layer in capturer}
    assert captured == {"0", "1", "3"}


def test_capture_rejects_a_reused_module() -> None:
    """A module called twice overwrites its own statistic and its per-sample gradient is a sum over
    calls this package does not form. None of the eight built models does it; the guard makes that
    a checked property rather than a comment.
    """
    seed_all(0)
    shared = nn.Linear(4, 4).to(DTYPE)
    model = nn.Sequential(shared, shared)
    with conventions.reference_mode(model), capture.capture(model) as _:
        with pytest.raises(RuntimeError, match="called 2 times"):
            model(torch.randn(3, 4, dtype=DTYPE, requires_grad=True))


def test_capture_rejects_batchnorm_in_train_mode() -> None:
    """With BN in train mode ``x_hat`` depends on the rest of the batch, so there are no per-sample
    gradients to stack (``plan_exp_draft.md`` §2.5). Refused, not approximated.
    """
    seed_all(0)
    module = nn.BatchNorm2d(3).to(DTYPE)
    module.train()
    with pytest.raises(RuntimeError, match="train mode"):
        capture.normalized_input(module, torch.randn(4, 3, 2, 2, dtype=DTYPE))


@pytest.mark.parametrize("name", _model_ids())
def test_coverage_lock(name: str) -> None:
    """Which parameters of each built model the capture covers. A ``GroupNorm`` affine is covered
    here while ``registry`` reports it ``hooked=False``: the campaign measures curvature, not what
    ``AdaFisherMulti`` preconditions. The uncovered set is exactly the raw ``nn.Parameter``s.
    """
    bench = BENCHES[name]
    seed_all(0)
    model = bench.build_model(**{key: value[0] for key, value in bench.model_choices.items()})
    covered, uncovered = capture.coverage(model)

    assert tuple(uncovered) == EXPECTED_UNCOVERED[name]
    assert len(covered) + len(uncovered) == len(list(model.named_parameters()))
    infos = registry.classify(model)
    registry.assert_partitions(model, infos)
    assert set(uncovered) <= set(registry.unhooked_parameters(infos))


def test_coverage_lock_covers_every_benchmark() -> None:
    """A new ``benchmarks/<model>/`` folder with no entry here fails the suite, as in lot 0."""
    assert set(EXPECTED_UNCOVERED) == set(BENCHES)


def test_uncovered_parameters_raise_instead_of_zero_columns() -> None:
    """``vit_micro_cifar``'s ``pos_embed`` belongs to no capturable module. Its columns of ``U``
    would be identically zero, leaving ``F`` symmetric, PSD and wrong — so the builder refuses.
    """
    bench = BENCHES["vit_micro_cifar"]
    seed_all(0)
    model = bench.build_model()
    inputs, targets = bench.prepare_batch((torch.randn(2, 3, 32, 32), torch.zeros(2).long()))
    with pytest.raises(NotImplementedError, match="pos_embed"):
        build_dense_reference(model, inputs, targets, source="empirical", batch_size=2)


# ----------------------------------------------------------------------------------------------
# T1 — the dense reference against an independent matrix-free Fv
# ----------------------------------------------------------------------------------------------


def _matrix_free_fisher_matvec(model: nn.Module, inputs: Tensor, vector: Tensor) -> Tensor:
    """``F v = (1/N) sum_n J_n^T Lambda_n J_n v``, by one forward-mode JVP, the closed-form
    ``Lambda_n u = diag(p) u - p (p^T u)``, and one VJP.

    Independent of the build under test by construction: it never forms a root of ``Lambda``, never
    touches ``fisher_ref.sources``, and gets its per-example structure from ``torch.func`` instead
    of from hooks.
    """
    names = [name for name, _ in model.named_parameters()]
    shapes = [value.shape for _, value in model.named_parameters()]
    numels = [value.numel() for _, value in model.named_parameters()]
    parameters = {name: value.detach() for name, value in model.named_parameters()}
    buffers = {name: value.detach() for name, value in model.named_buffers()}

    def forward(params: Dict[str, Tensor]) -> Tensor:
        return functional_call(model, (params, buffers), (inputs,))

    tangent, offset = {}, 0
    for name, shape, numel in zip(names, shapes, numels):
        tangent[name] = vector[offset:offset + numel].view(shape)
        offset += numel

    outputs, jacobian_vector = jvp(forward, (parameters,), (tangent,))
    probabilities = torch.softmax(outputs, dim=-1)
    weighted = (probabilities * jacobian_vector
                - probabilities * (probabilities * jacobian_vector).sum(-1, keepdim=True))
    _, vjp_fn = vjp(forward, parameters)
    cotangents = vjp_fn(weighted)[0]
    return torch.cat([cotangents[name].reshape(-1) for name in names]) / inputs.shape[0]


def test_t1_dense_fv_matches_matrix_free() -> None:
    """T1 (``plan_exp_draft.md`` §10.2): the dense ``F v`` against the matrix-free one, 20 random
    vectors, relative ``1e-10`` in fp64.
    """
    model = _tiny_classifier()
    inputs, targets = _tiny_batch()
    reference = build_dense_reference(model, inputs, targets, source="type2", batch_size=5)

    seed_all(7)
    worst = 0.0
    for _ in range(20):
        vector = torch.randn(reference.P, dtype=DTYPE)
        oracle = _matrix_free_fisher_matvec(model, inputs, vector)
        worst = max(worst, float((reference.matvec(vector) - oracle).norm() / oracle.norm()))
    assert worst <= 1e-10, f"worst relative gap {worst:.2e}"


def test_t1_empirical_matches_the_per_sample_gradient_gram() -> None:
    """``E_hat = (1/N) sum_n grad l_n grad l_n^T``, assembled from ``torch.func`` per-sample
    gradients of the real loss — a second, independent route to the same matrix.
    """
    model = _tiny_classifier()
    inputs, targets = _tiny_batch()
    reference = build_dense_reference(model, inputs, targets, source="empirical", batch_size=5)

    parameters = {name: value.detach() for name, value in model.named_parameters()}
    names = list(parameters)

    def per_sample_loss(params: Dict[str, Tensor], one_input: Tensor, one_target: Tensor) -> Tensor:
        output = functional_call(model, params, (one_input.unsqueeze(0),))
        return F.cross_entropy(output, one_target.unsqueeze(0))

    gradients = vmap(grad(per_sample_loss), in_dims=(None, 0, 0))(parameters, inputs, targets)
    rows = torch.cat([gradients[name].reshape(inputs.shape[0], -1) for name in names], dim=1)
    expected = rows.T @ rows / inputs.shape[0]
    assert torch.allclose(reference.matrix, expected, atol=1e-12, rtol=0)


# ----------------------------------------------------------------------------------------------
# T2 — type-2 against Monte-Carlo
# ----------------------------------------------------------------------------------------------


def test_type2_head_block_kernel_is_the_logit_shift_subspace() -> None:
    """A structural property nothing in the code encodes, so it checks the whole pipeline at once
    (root, capture, layout): adding a constant to every logit leaves the softmax unchanged, so the
    head block of the **type-2** Fisher annihilates ``W += 1_C v^T, b += c 1_C`` for every ``v``,
    i.e. it has a kernel of dimension exactly ``d_in + 1`` however many probes are used.

    Measured on the real A1 checkpoint (``plan_exp_lot1.md`` §4): deficiency 33 = 32 + 1 on a
    ``330 x 330`` block built from 256 probes, and ``||B d|| / (||B|| ||d||) = 9e-18`` on such a
    direction against ``9e-2`` on a random one. Later lots must read every inverse metric on a head
    block against this kernel, not against ``lambda`` alone.
    """
    seed_all(0)
    classes, features = 4, 5
    model = nn.Sequential(nn.Linear(6, features), nn.Tanh(), nn.Linear(features, classes)).to(DTYPE)
    inputs, targets = _tiny_batch(n=16, classes=classes)
    reference = build_dense_reference(model, inputs, targets, source="type2", batch_size=8,
                                      modules=["2"])
    block = reference.block("2")

    eigenvalues = torch.linalg.eigvalsh(block)
    deficiency = block.shape[0] - int((eigenvalues > 1e-12 * eigenvalues.max()).sum())
    assert deficiency == features + 1

    direction = torch.cat([torch.ones(classes, 1, dtype=DTYPE).mm(torch.randn(1, features,
                                                                              dtype=DTYPE)).reshape(-1),
                           torch.ones(classes, dtype=DTYPE)])
    scale = float(torch.linalg.matrix_norm(block) * direction.norm())
    assert float((block @ direction).norm()) / scale < 1e-14

    empirical = build_dense_reference(model, inputs, targets, source="empirical", batch_size=8,
                                      modules=["2"])
    assert float((empirical.block("2") @ direction).norm()) / scale < 1e-14


def test_t2_type2_matches_monte_carlo() -> None:
    """T2: ``e_F(F_type2, F_MC(K)) = O(K^{-1/2})``, consistent over 5 seeds.

    Measured here (5 seeds, ``N = 32``): ``0.091`` at ``K = 1e2``, ``0.025`` at ``1e3``, ``0.0066``
    at ``1e4`` — ratios ``3.6`` and ``3.8`` against the ``sqrt(10) = 3.16`` of a clean
    ``K^{-1/2}``. The assertion is a bracket around that rate, not the point estimate, because five
    seeds do not pin a variance to two digits. Runs in ~4 s: one backward pass per Monte-Carlo
    sample, 55 500 of them.
    """
    seed_all(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.LayerNorm(5), nn.ReLU(), nn.Linear(5, 4)).to(DTYPE)
    inputs, targets = _tiny_batch(n=32)
    exact = build_dense_reference(model, inputs, targets, source="type2", batch_size=32)
    norm = float(torch.linalg.matrix_norm(exact.matrix))

    means = []
    for samples in (100, 1_000, 10_000):
        errors = []
        for seed in range(5):
            generator = torch.Generator().manual_seed(seed)
            estimate = build_dense_reference(model, inputs, targets, source="mc", k=samples,
                                             batch_size=32, generator=generator)
            errors.append(float(torch.linalg.matrix_norm(estimate.matrix - exact.matrix)) / norm)
        means.append(sum(errors) / len(errors))

    assert means[2] < means[1] < means[0], f"no decay in K: {means}"
    assert means[2] < 0.02, f"K = 1e4 should be within 2% of the type-2 Fisher, got {means[2]:.4f}"
    for coarse, fine in zip(means, means[1:]):
        assert 2.2 <= coarse / fine <= 4.5, f"decay rate {coarse / fine:.2f} is not O(K^-1/2)"


# ----------------------------------------------------------------------------------------------
# T6 — normalisation layers against central finite differences
# ----------------------------------------------------------------------------------------------


def _finite_difference_per_sample_gradients(model: nn.Module, inputs: Tensor, cotangent: Tensor,
                                            parameter: Tensor, step: float = 1e-5) -> Tensor:
    """Central differences of ``phi_n(theta) = <v_n, f(theta, x_n)>`` over every coordinate of
    ``parameter``, returning ``(N, *parameter.shape)``.

    ``cotangent`` is **fixed**: differentiating a ``v`` recomputed from ``p(theta)`` would add a
    ``d v / d theta`` term and the mismatch would look like an ``x_hat`` bug.
    """
    flat = parameter.detach().reshape(-1).clone()
    columns = []
    with torch.no_grad():
        for index in range(flat.numel()):
            original = flat[index].item()
            flat[index] = original + step
            parameter.data.copy_(flat.view(parameter.shape))
            plus = (model(inputs) * cotangent).flatten(1).sum(1)
            flat[index] = original - step
            parameter.data.copy_(flat.view(parameter.shape))
            minus = (model(inputs) * cotangent).flatten(1).sum(1)
            flat[index] = original
            parameter.data.copy_(flat.view(parameter.shape))
            columns.append((plus - minus) / (2 * step))
    return torch.stack(columns, dim=1).reshape(inputs.shape[0], *parameter.shape)


def _norm_layer_cases() -> List[Tuple[str, nn.Module, Tensor, str]]:
    seed_all(0)
    tiny = TinyMultiLayerNet().to(DTYPE)
    tiny.bn.running_mean.normal_()
    tiny.bn.running_var.uniform_(0.5, 1.5)
    return [
        ("layernorm", tiny, torch.randn(4, 2, 5, 5, dtype=DTYPE), "ln"),
        ("batchnorm2d_eval", tiny, torch.randn(4, 2, 5, 5, dtype=DTYPE), "bn"),
        ("groupnorm", _GroupNormNet().to(DTYPE), torch.randn(3, 2, 5, 5, dtype=DTYPE), "norm"),
    ]


@pytest.mark.parametrize("label,model,inputs,layer_name", _norm_layer_cases(),
                         ids=lambda case: case)
def test_t6_norm_gradients_match_finite_differences(label: str, model: nn.Module, inputs: Tensor,
                                                    layer_name: str) -> None:
    """T6: the per-sample gradients of a normalisation layer's ``gamma`` and ``beta`` against
    central finite differences, relative ``1e-6``. This is the test that ``x_hat`` — recomputed in
    closed form, since a forward hook sees the *pre*-normalised input — is the right variable.
    """
    seed_all(4)
    with conventions.reference_mode(model), capture.capture(model) as capturer:
        outputs = model(inputs)
        cotangent = torch.randn_like(outputs)
        torch.autograd.backward(outputs, cotangent)
        analytic = capture.per_sample_gradients(capturer.layer(layer_name))

    module = dict(model.named_modules())[layer_name]
    with conventions.reference_mode(model):
        for p_name, per_sample in analytic.items():
            numeric = _finite_difference_per_sample_gradients(
                model, inputs, cotangent, getattr(module, p_name)
            )
            gap = float((per_sample - numeric).norm() / numeric.norm())
            assert gap <= 1e-6, f"{layer_name}.{p_name}: relative gap {gap:.2e}"


def test_t6_raw_input_fails_finite_differences() -> None:
    """Non-vacuity: the same contraction on the *pre*-normalisation input — what a forward hook
    actually sees — does not reproduce the finite differences. Without this, T6 would pass on a
    capture that never normalised anything.
    """
    seed_all(4)
    model = TinyMultiLayerNet().to(DTYPE)
    inputs = torch.randn(4, 2, 5, 5, dtype=DTYPE)
    raw: Dict[str, Tensor] = {}
    handle = model.ln.register_forward_hook(
        lambda module, args, output: raw.__setitem__("ln", args[0].detach())
    )
    try:
        with conventions.reference_mode(model), capture.capture(model) as capturer:
            outputs = model(inputs)
            cotangent = torch.randn_like(outputs)
            torch.autograd.backward(outputs, cotangent)
            layer = capturer.layer("ln")
            analytic = capture.per_sample_gradients(layer)["weight"]
            pooled_raw = raw["ln"].reshape(inputs.shape[0], -1, model.ln.normalized_shape[0])
            wrong = (layer.g * pooled_raw).sum(dim=1)
    finally:
        handle.remove()

    with conventions.reference_mode(model):
        numeric = _finite_difference_per_sample_gradients(model, inputs, cotangent,
                                                          model.ln.weight)
    assert float((analytic - numeric).norm() / numeric.norm()) <= 1e-6
    assert float((wrong - numeric).norm() / numeric.norm()) > 1e-2


# ----------------------------------------------------------------------------------------------
# reference/dense.py — layout, blocks, streaming, precision
# ----------------------------------------------------------------------------------------------


def test_block_is_a_slice_of_f() -> None:
    """``B_l`` is a submatrix of ``F``, not a second construction — and it equals the reference
    built over that module alone, which is what makes a restricted build a valid ``B_l``.
    """
    model = _tiny_classifier()
    inputs, targets = _tiny_batch()
    reference = build_dense_reference(model, inputs, targets, source="type2", batch_size=6)
    columns = reference.layout.block_slice("0")
    assert torch.equal(reference.block("0"), reference.matrix[columns, columns])

    restricted = build_dense_reference(model, inputs, targets, source="type2", batch_size=6,
                                       modules=["0"])
    assert torch.allclose(restricted.matrix, reference.block("0"), atol=1e-12, rtol=0)


@pytest.mark.parametrize("name", _model_ids())
def test_block_slices_are_contiguous_on_every_model(name: str) -> None:
    """The ``named_parameters()`` column layout only gives ``B_l`` as a slice because ``weight`` is
    registered before ``bias`` in every relevant module. Asserted, not assumed.
    """
    bench = BENCHES[name]
    seed_all(0)
    model = bench.build_model(**{key: value[0] for key, value in bench.model_choices.items()})
    layout = ParamLayout.of(model)
    assert layout.P == sum(p.numel() for p in model.parameters())
    for module_name in capture.capturable_modules(model):
        columns = layout.block_slice(module_name)
        owned = sum(p.numel() for p in dict(model.named_modules())[module_name]
                    .parameters(recurse=False))
        assert columns.stop - columns.start == owned


@pytest.mark.parametrize("block", [1, 3, 7, 64])
def test_symmetrize_is_the_naive_expression_without_its_memory(block: int) -> None:
    """``symmetrize_`` must equal ``0.5 * (M + M.T)`` exactly, at every block size including ones
    that do not divide ``P``. It exists because the naive spelling allocates two further ``P x P``
    tensors — 11.4 GB of them at A1's ``P = 26 634``, which is what would put a dense reference
    over the 10 GB MIG slice (``plan_exp_lot1.md`` §0.9).
    """
    seed_all(0)
    matrix = torch.randn(17, 17, dtype=DTYPE)
    expected = 0.5 * (matrix + matrix.T)
    assert torch.equal(symmetrize_(matrix.clone(), block=block), expected)


def test_streaming_batch_size_is_inert() -> None:
    """``U`` is never materialised; the micro-batch size is an implementation detail and must not
    move the result beyond fp64 round-off.
    """
    model = _tiny_classifier()
    inputs, targets = _tiny_batch()
    references = [build_dense_reference(model, inputs, targets, source="type2", batch_size=size)
                  for size in (3, 5, 12)]
    for other in references[1:]:
        assert torch.allclose(references[0].matrix, other.matrix, atol=1e-14, rtol=0)


def test_fp32_capture_gap_is_measured(capsys: pytest.CaptureFixture) -> None:
    """Reported, not asserted. ``plan_exp_draft.md`` §12 allows accumulating ``U`` in fp32 with the
    Grams in fp64; this is the number that advice rests on, and the reason lot 1 itself stays fp64
    (T1's ``1e-10`` does not survive fp32).
    """
    model = _tiny_classifier()
    inputs, targets = _tiny_batch()
    exact = build_dense_reference(model, inputs, targets, source="type2", batch_size=6)
    single = build_dense_reference(model, inputs, targets, source="type2", batch_size=6,
                                   dtype=torch.float32)
    gap = float((single.matrix.double() - exact.matrix).norm()
                / torch.linalg.matrix_norm(exact.matrix))
    with capsys.disabled():
        print(f"\n  [measured] fp32 vs fp64 capture: relative Frobenius gap {gap:.2e}")
    assert gap < 1e-4  # a sanity bound only: the point is the printed number


def test_reference_metadata_carries_the_invariants() -> None:
    """The precision block records the *live* state: a reference is only TF32-free because the
    driver called ``configure()``, which a library must not do behind the caller's back.
    """
    conventions.configure()
    model = _tiny_classifier()
    inputs, targets = _tiny_batch()
    reference = build_dense_reference(model, inputs, targets, source="type2", batch_size=6,
                                      probe_digest="deadbeef")
    meta = reference.metadata()
    assert meta["metrics_version"] == conventions.METRICS_VERSION
    assert meta["source"] == "type2" and meta["regime"] == "A"
    assert meta["n_rows"] == reference.n_probes * reference.n_columns
    assert meta["probe_digest"] == "deadbeef"
    assert meta["precision"]["cudnn_allow_tf32"] is not True


def test_curvlinops_cross_check() -> None:
    """Optional. ``curvlinops`` is not a dependency of this project (``plan_exp_lot1.md`` §0.4): the
    matrix-free oracle above is what T1 rests on. If someone installs it locally, this is the third
    opinion.
    """
    curvlinops = pytest.importorskip("curvlinops")
    model = _tiny_classifier()
    inputs, targets = _tiny_batch()
    reference = build_dense_reference(model, inputs, targets, source="type2", batch_size=12)
    operator = curvlinops.GGNLinearOperator(
        model, nn.CrossEntropyLoss(), list(model.parameters()), [(inputs, targets)]
    )
    seed_all(9)
    vector = torch.randn(reference.P, dtype=DTYPE)
    expected = torch.as_tensor(operator @ vector.numpy(), dtype=DTYPE)
    assert torch.allclose(reference.matvec(vector), expected, atol=1e-8, rtol=1e-8)


# ----------------------------------------------------------------------------------------------
# A real benchmark model, end to end
# ----------------------------------------------------------------------------------------------


def test_regime_a_model_end_to_end() -> None:
    """A1 (``mlp_ln_mnist``) on a synthetic batch, restricted to the head and the two LayerNorms:
    the full path — build the model the way the bench does, prepare a batch, build ``F`` and
    ``E_hat``, read a block — at a size that fits on a laptop. The unrestricted ``P = 26 634``
    reference is 5.68 GB and belongs to the cluster run (``plan_exp_lot1.md`` §5).
    """
    bench = BENCHES["mlp_ln_mnist"]
    seed_all(0)
    model = bench.build_model()
    shape = MODEL_CONTRACT["mlp_ln_mnist"].input_shape
    inputs, targets = bench.prepare_batch(
        (torch.randn(16, *shape), torch.randint(0, 10, (16,)))
    )
    modules = ["features.1", "features.4", "head"]

    fisher = build_dense_reference(model, inputs, targets, source="type2", batch_size=8,
                                   modules=modules)
    empirical = build_dense_reference(model, inputs, targets, source="empirical", batch_size=8,
                                      modules=modules)
    assert fisher.P == empirical.P == 32 + 32 + 32 + 32 + 10 * 32 + 10
    assert fisher.n_columns == 10 and fisher.n_rows == 160
    assert empirical.n_columns == 1
    assert float(torch.linalg.eigvalsh(fisher.matrix).min()) > -1e-12
    assert fisher.block("head").shape == (10 * 32 + 10, 10 * 32 + 10)
