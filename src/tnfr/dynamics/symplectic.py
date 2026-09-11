"""Symplectic integration for the mechanical conjugate-pair representation.

EPI stores ``[q, q_dot]`` and the second half of DeltaNFR stores force, so
``dq/dt = q_dot`` and ``d(q_dot)/dt = nu_f * force``. This is the inertial
mapping in ``physics.classical_mechanics``, distinct from the bare overdamped
nodal equation. In particular, zero inverse inertia does not stop an existing
velocity. The ordinary nodal integrator implements the zero-capacity rule.

The stated symplectic orders require an autonomous separable Hamiltonian,
constant ``nu_f`` during the step, and a position-only conservative force.
Symplecticity alone does not certify bounded trajectories or grammar U2.
"""

from __future__ import annotations

import math
from typing import Callable

from ..constants import DNFR_PRIMARY, EPI_PRIMARY, VF_PRIMARY
from ..mathematics.unified_numerical import np
from ..types import TNFRNode


def _finite_step(dt: float) -> float:
    """Allow reversible signed steps, but reject nonfinite time intervals."""
    value = float(dt)
    if not math.isfinite(value):
        raise ValueError("Symplectic timestep must be finite")
    return value


class TNFRSymplecticIntegrator:
    """Synchronized mechanical steppers sharing the velocity-Verlet kernel."""

    @staticmethod
    def velocity_verlet(
        node: TNFRNode, dt: float, force_evaluator: Callable[[TNFRNode], np.ndarray]
    ) -> None:
        """Apply one second-order kick-drift-kick step to ``[q, q_dot]``.

        The stored DeltaNFR must contain the force at the initial position.
        ``force_evaluator`` returns a full pressure vector at the updated
        position; its second half drives acceleration. Forces depending on
        velocity or time do not have the stated separable-Hamiltonian guarantee.
        A zero step leaves the node untouched and does not evaluate the force.
        """
        dt = _finite_step(dt)
        if dt == 0.0:
            return

        epi = node[EPI_PRIMARY]
        nu_f = node[VF_PRIMARY]
        dnfr = node[DNFR_PRIMARY]
        n = len(epi) // 2
        q, q_dot = epi[:n], epi[n:]

        q_dot_half = q_dot + 0.5 * nu_f * dnfr[n:] * dt
        q_new = q + q_dot_half * dt
        node[EPI_PRIMARY] = np.concatenate([q_new, q_dot_half])
        dnfr_new = force_evaluator(node)
        node[DNFR_PRIMARY] = dnfr_new
        q_dot_new = q_dot_half + 0.5 * nu_f * dnfr_new[n:] * dt
        node[EPI_PRIMARY] = np.concatenate([q_new, q_dot_new])

    @staticmethod
    def leapfrog(
        node: TNFRNode, dt: float, force_evaluator: Callable[[TNFRNode], np.ndarray]
    ) -> None:
        """Apply synchronized kick-drift-kick leapfrog (velocity Verlet)."""
        TNFRSymplecticIntegrator.velocity_verlet(node, dt, force_evaluator)

    @staticmethod
    def yoshida_4th_order(
        node: TNFRNode, dt: float, force_evaluator: Callable[[TNFRNode], np.ndarray]
    ) -> None:
        """Compose three Verlet steps into a fourth-order symmetric step.

        ``S(w1*h) S(w0*h) S(w1*h)`` has ``2*w1+w0=1`` and
        ``2*w1**3+w0**3=0``. These identities preserve the requested interval
        and cancel the leading cubic error of the symmetric second-order map.
        The middle substep is negative; the force must permit reversible stages.
        """
        dt = _finite_step(dt)
        if dt == 0.0:
            return
        root_two = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - root_two)
        w0 = -root_two * w1
        for coefficient in (w1, w0, w1):
            TNFRSymplecticIntegrator.velocity_verlet(node, coefficient * dt, force_evaluator)
