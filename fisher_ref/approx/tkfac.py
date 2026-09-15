"""TKFAC: the trace-preserving Kronecker factorisation (``plan_exp_draft.md`` §4;
arXiv:2011.10741 Eq. 4.3-4.9).

``K = delta * Phi (x) Psi`` with, in the paper's notation (``Lambda`` the per-sample input factor
``a a^T``, ``Gamma`` the output one ``g g^T``):

    delta = E[tr(Lambda) tr(Gamma)],   Phi = E[tr(Gamma) Lambda] / delta,
    Psi   = E[tr(Lambda) Gamma] / delta,   so tr(Phi) = tr(Psi) = 1.

**It needs no new block class.** ``delta * Phi (x) Psi`` in this repository's ``rvec`` convention is
``kron(delta * Psi, Phi)``, i.e. a :class:`~fisher_ref.approx.kfac.Kron` with the scalar folded into
one factor — and folding it there rather than carrying it separately is what keeps ``solve`` and
``logdet`` exact, since ``delta K + lam I`` is not a Kronecker product but ``kron(delta*Psi, Phi)``
still is.

**Why the trace is preserved exactly** (T8): ``tr(K) = delta * tr(Phi) * tr(Psi) = delta``, and
``tr(B_l) = E[||rvec(g a^T)||^2] = E[||a||^2 ||g||^2] = E[tr(Lambda) tr(Gamma)] = delta``. It is an
identity of the estimator, not a property of the data — which is why `plan_lot3.md` §0.2 insists the
EMA be kept on the **un-normalised numerators**: normalising first and multiplying back loses it.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from .kfac import Kron


@dataclass(frozen=True)
class TkfacStats:
    """The un-normalised accumulators, summed over probes and columns.

    ``delta`` is their common denominator; keeping the numerators raw is what makes ``tr(K)``
    exactly ``tr(B_l)`` rather than exactly-up-to-round-off.
    """

    delta: Tensor        # sum over (n, c) of tr(Lambda) tr(Gamma)
    phi_raw: Tensor      # sum over (n, c) of tr(Gamma) * Lambda      (d_in, d_in)
    psi_raw: Tensor      # sum over (n, c) of tr(Lambda) * Gamma      (d_out, d_out)
    count: int

    def build(self) -> Kron:
        """``delta * Phi (x) Psi`` as a :class:`Kron`, with ``delta`` folded into the output factor.

        ``Phi = phi_raw / delta`` and ``Psi = psi_raw / delta`` both have unit trace, so
        ``delta * Phi (x) Psi = (phi_raw / delta) (x) (psi_raw / delta) * delta``; the mean over
        ``count`` cancels in ``Phi`` and ``Psi`` and survives once in ``delta``.
        """
        delta_mean = self.delta / self.count
        phi = self.phi_raw / self.delta
        psi = self.psi_raw / self.delta
        return Kron(A=phi, G=psi * delta_mean)


__all__ = ["TkfacStats"]
