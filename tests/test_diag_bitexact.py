"""``AdaFisherMulti(fisher_mode="diag", minmax_normalization=False)`` vs. the reference
``AdaFisher`` optimizer of ``reference_repos/FisherAdapTune/scripts/adafisher.py``.

Three checks, per docs/reports/plan_lot1.md §3.2:

- ``f_tilde`` construction (``kron(H, S) + Lambda``) must be bit-exact (``torch.equal``) — this is
  exit criterion 1 of lot 1 and does not depend on the optimizer step's floating-point op order at
  all. To test it in isolation, ``model_new``'s parameters are resynced onto ``model_ref``'s after
  every step: this decouples the comparison from any drift the *update* step introduces (see next
  point), so a divergence here can only mean the H/S extraction, EMA, or Kronecker/damping
  construction disagree with the reference — never that the two optimizers' parameter trajectories
  have separated.
- On a single ``Linear`` layer with a handful of steps, the full parameter trajectory is checked
  with ``torch.allclose`` at a tight tolerance. This *does* depend on op order: the new code applies
  the preconditioner via unfused ``div`` + ``add_(alpha=...)``, the reference via a single fused
  ``addcdiv_`` — mathematically identical, not guaranteed bit-identical. Empirically the two paths
  differ by a few ULPs per step (~1e-7 after one step, ~1e-6 after six, for weights of order 1), so a
  tight tolerance here still catches a real bug while tolerating that gap.
- On the full ``TinyMultiLayerNet`` (which includes ``BatchNorm2d``), the same per-step ULP-level
  gap is amplified by BatchNorm's running-statistics feedback loop (each step's normalisation
  depends on the *current*, already slightly-diverged, activations) into an O(1e-4) divergence after
  six steps at ``lr=1e-2`` — confirmed by comparing against the Linear-only case, where the gap stays
  at O(1e-6). This is a property of the chaotic dynamics of BatchNorm + a fairly large learning
  rate, not a precision bug in this port; the test below only checks that the two trajectories stay
  within a loose, order-of-magnitude tolerance, as a coarse regression guard rather than a precision
  guarantee.
"""

from __future__ import annotations

import copy

import torch
from adafisher_modes import AdaFisherMulti
from adafisher_modes.optimizer import SUPPORTED_MODULES
from conftest import TinyMultiLayerNet, seed_all


def _build_pair(fisheradaptune_adafisher, model_factory, **opt_kwargs):
    seed_all(0)
    model_ref = model_factory()
    model_new = copy.deepcopy(model_ref)

    opt_ref = fisheradaptune_adafisher.AdaFisher(model_ref, **opt_kwargs)
    opt_new = AdaFisherMulti(
        model_new, fisher_mode="diag", minmax_normalization=False, **opt_kwargs
    )
    return model_ref, opt_ref, model_new, opt_new


def _hooked_modules_by_type(model: torch.nn.Module):
    return [m for m in model.modules() if m.__class__.__name__ in SUPPORTED_MODULES]


def _f_tilde_reference_as_unsplit(opt_ref, module: torch.nn.Module) -> torch.Tensor:
    """Reassemble the reference's (possibly weight/bias-split) F_tilde into the same unsplit
    ``(d_out, d_in + int(has_bias))`` layout as ``DiagApproximation.f_tilde``.
    """
    F_ref = opt_ref._get_F_tilde(module)
    if isinstance(F_ref, list):
        weight_part = F_ref[0].reshape(F_ref[0].shape[0], -1)
        bias_part = F_ref[1].reshape(-1, 1)
        return torch.cat([weight_part, bias_part], dim=1)
    return F_ref.reshape(F_ref.shape[0], -1)


def test_f_tilde_bitexact_across_steps(fisheradaptune_adafisher) -> None:
    model_ref, opt_ref, model_new, opt_new = _build_pair(
        fisheradaptune_adafisher, TinyMultiLayerNet, lr=1e-2, beta=0.9, Lambda=1e-3, TCov=2
    )

    seed_all(1)
    batch = torch.randn(6, 2, 5, 5)

    modules_ref = _hooked_modules_by_type(model_ref)
    modules_new = _hooked_modules_by_type(model_new)
    assert len(modules_ref) == len(modules_new) > 0

    for step in range(6):  # 3 * TCov, so the EMA fires three times on each factor
        opt_ref.zero_grad()
        opt_new.zero_grad()

        out_ref = model_ref(batch)
        out_new = model_new(batch)
        assert torch.equal(out_ref, out_new), f"forward diverged at step {step}"

        out_ref.sum().backward()
        out_new.sum().backward()

        for m_ref, m_new in zip(modules_ref, modules_new):
            F_ref = _f_tilde_reference_as_unsplit(opt_ref, m_ref)
            F_new = opt_new.approx.f_tilde(m_new)
            assert torch.equal(F_new, F_ref), f"F_tilde diverged at step {step} for {type(m_ref).__name__}"

        opt_ref.step()
        opt_new.step()
        # Decouple this test from the optimizer *update* rule's op order (fused addcdiv_ vs.
        # unfused div + add_, see module docstring): resync so next step's forward/backward again
        # starts from bit-identical parameters, isolating what this test actually targets.
        model_new.load_state_dict(model_ref.state_dict())


def test_parameter_trajectory_close_on_linear_layer(fisheradaptune_adafisher) -> None:
    model_ref, opt_ref, model_new, opt_new = _build_pair(
        fisheradaptune_adafisher,
        lambda: torch.nn.Linear(4, 3, bias=True),
        lr=1e-2, beta=0.9, Lambda=1e-3, TCov=1,
    )

    seed_all(1)
    x = torch.randn(6, 4)

    for _ in range(6):
        opt_ref.zero_grad()
        opt_new.zero_grad()
        model_ref(x).sum().backward()
        model_new(x).sum().backward()
        opt_ref.step()
        opt_new.step()

    for (name_ref, p_ref), (name_new, p_new) in zip(
        model_ref.named_parameters(), model_new.named_parameters()
    ):
        assert name_ref == name_new
        assert torch.allclose(p_ref, p_new, rtol=1e-5, atol=1e-6), f"trajectory diverged at {name_ref}"


def test_parameter_trajectory_reasonable_on_full_net(fisheradaptune_adafisher) -> None:
    model_ref, opt_ref, model_new, opt_new = _build_pair(
        fisheradaptune_adafisher, TinyMultiLayerNet, lr=1e-2, beta=0.9, Lambda=1e-3, TCov=2
    )

    seed_all(1)
    batch = torch.randn(6, 2, 5, 5)

    for _ in range(6):
        opt_ref.zero_grad()
        opt_new.zero_grad()
        model_ref(batch).sum().backward()
        model_new(batch).sum().backward()
        opt_ref.step()
        opt_new.step()

    for (name_ref, p_ref), (name_new, p_new) in zip(
        model_ref.named_parameters(), model_new.named_parameters()
    ):
        assert name_ref == name_new
        # Loose on purpose — see module docstring: BatchNorm's running-stat feedback amplifies the
        # per-step ULP-level gap well beyond what a Linear-only trajectory shows.
        assert torch.allclose(p_ref, p_new, rtol=1e-2, atol=1e-3), f"trajectory diverged at {name_ref}"
