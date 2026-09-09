"""``DiagApproximation(minmax_normalization=True)`` matches Eq. (4) semantics.

Cannot use the official repository's ``AdaFisher`` optimizer as a bit-exact oracle here: its EMA is
buggy (docs/reports/plan.md §1.4), so trusting it end-to-end would validate against a broken
reference. Instead this test composes two pieces already validated independently elsewhere:

- the raw (non-normalised) instantaneous diagonal, exact per ``test_diag_bitexact.py`` (bit-exact
  against FisherAdapTune, whose extraction functions are what ``factors.py`` ports);
- ``min_max_normalization``, exact per ``test_minmax_matches_official.py`` (bit-exact against the
  official repository's ``MinMaxNormalization``, in isolation from its EMA).

and checks that ``DiagApproximation``'s internal state, with ``minmax_normalization=True``, equals
applying them in the order Eq. (4) and the official repository specify: normalise the instantaneous
factor, *then* EMA it (``AdaFisher.py:412,431`` — normalisation is applied to ``H_D_i``/``S_D_i``
before ``update_running_avg``, not to the accumulator). See docs/reports/plan_lot1.md §3.3.
"""

from __future__ import annotations

import torch
from adafisher_modes.approximations.diag import DiagApproximation
from adafisher_modes.ema import update_running_avg
from adafisher_modes.factors import compute_h_diag, compute_s_diag
from adafisher_modes.minmax import min_max_normalization
from conftest import TinyMultiLayerNet, seed_all


def _capture_raw_factors(model: TinyMultiLayerNet, named, x: torch.Tensor):
    """Run one forward+backward pass and return, per module, the exact raw tensors the hooks
    would see (``input[0].data`` / ``grad_output[0].data``).
    """
    captured_inputs, captured_grad_outputs = {}, {}

    def make_fwd_hook(name):
        def hook(module, inp, out):
            captured_inputs[name] = inp[0].detach().clone()

        return hook

    def make_bwd_hook(name):
        def hook(module, grad_input, grad_output):
            captured_grad_outputs[name] = grad_output[0].detach().clone()

        return hook

    handles = []
    for name, module in named:
        handles.append(module.register_forward_hook(make_fwd_hook(name)))
        handles.append(module.register_full_backward_hook(make_bwd_hook(name)))

    model(x).sum().backward()
    for h in handles:
        h.remove()
    return captured_inputs, captured_grad_outputs


def test_minmax_applied_before_ema_at_step_zero_and_one() -> None:
    seed_all(2)
    model = TinyMultiLayerNet()
    named = [("conv", model.conv), ("bn", model.bn), ("fc1", model.fc1), ("ln", model.ln), ("fc2", model.fc2)]

    approx = DiagApproximation(Lambda=1e-3, gammas=(0.92, 0.008), minmax_normalization=True)
    expected_H = {name: None for name, _ in named}
    expected_S = {name: None for name, _ in named}

    for step, seed in enumerate((10, 11)):  # step 0 exercises the ones-init branch, step 1 the EMA blend
        seed_all(seed)
        x = torch.randn(6, 2, 5, 5)
        captured_inputs, captured_grad_outputs = _capture_raw_factors(model, named, x)

        for name, module in named:
            approx.update_input_factor(module, captured_inputs[name], step=step)
            approx.update_output_factor(module, captured_grad_outputs[name], step=step)

            H_raw = compute_h_diag(captured_inputs[name], module)
            S_raw = compute_s_diag(captured_grad_outputs[name], module)

            if step == 0:
                expected_H[name] = H_raw.new_ones(H_raw.size(0))
                expected_S[name] = S_raw.new_ones(S_raw.size(0))
            update_running_avg(min_max_normalization(H_raw), expected_H[name], approx.gammas)
            update_running_avg(min_max_normalization(S_raw), expected_S[name], approx.gammas)

            assert torch.equal(approx._H[module], expected_H[name]), f"H mismatch for {name} at step {step}"
            assert torch.equal(approx._S[module], expected_S[name]), f"S mismatch for {name} at step {step}"


def test_minmax_true_differs_from_minmax_false() -> None:
    """Sanity check that the ``minmax_normalization`` flag actually changes the accumulated
    state (not a semantics check by itself — see the two tests above for that).
    """
    seed_all(3)
    fc1 = TinyMultiLayerNet().fc1  # Linear(75, 8, bias=True)

    approx_on = DiagApproximation(minmax_normalization=True)
    approx_off = DiagApproximation(minmax_normalization=False)

    raw_input = torch.randn(6, 75)
    approx_on.update_input_factor(fc1, raw_input, step=0)
    approx_off.update_input_factor(fc1, raw_input, step=0)
    assert not torch.equal(approx_on._H[fc1], approx_off._H[fc1])
