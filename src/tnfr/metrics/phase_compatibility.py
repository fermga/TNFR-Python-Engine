"""Configured phase-compatibility scores using the shared circular difference.

The score 1 - abs(wrap(theta_b - theta_a))/pi interpolates between alignment
and antiphase. It is a declared bounded affinity, not a derived interference
law or a substitute for the operators' live U3 phase limit and state contracts.
The local network readout delegates to the shared phase-order diagnostic.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .._coherence_validation import validate_structural_coherence
from .._exact_time import finite_represented_real
from ..utils.numeric import angle_diff

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

__all__ = [
    "compute_phase_coupling_strength",
    "is_phase_compatible",
    "compute_network_phase_alignment",
]


def compute_phase_coupling_strength(
    theta_a: float,
    theta_b: float,
) -> float:
    """Return the configured linear affinity of two finite real phases.

    Both phases are admitted before conversion through the represented-real
    owner. The existing signed circular-difference kernel handles wrapping and
    rejects an unrepresentable subtraction. Equal phases score one, separation
    pi/2 scores one half, and antiphase scores zero. This score alone does not
    authorize an operator or determine subsequent phase evolution.
    """
    first = finite_represented_real(theta_a, "theta_a")[0]
    second = finite_represented_real(theta_b, "theta_b")[0]
    phase_diff = abs(angle_diff(second, first))
    return 1.0 - (phase_diff / math.pi)


def is_phase_compatible(
    theta_a: float,
    theta_b: float,
    threshold: float = 0.5,
) -> bool:
    """Compare the phase affinity with a supplied closed-unit-interval threshold.

    The default 0.5 admits wrapped separation at most pi/2; in general the
    selected limit is pi*(1-threshold), with equality admitted. This is a score
    policy, not the engine's independent configured U3 admission mechanism.
    Invalid phases or thresholds raise instead of becoming a compatibility
    decision.
    """
    minimum = validate_structural_coherence(threshold, name="threshold")
    coupling = compute_phase_coupling_strength(theta_a, theta_b)
    return coupling >= minimum


def compute_network_phase_alignment(
    G: TNFRGraph,
    node: NodeId,
    radius: int = 1,
) -> float:
    """Delegate center-inclusive phase order to its shared observation owner.

    This unweighted graph-ball magnitude is distinct from pairwise affinity and
    live coupling admission. Radius, stored-phase and empty-resultant semantics
    are those of metrics.phase_coherence.compute_phase_alignment.
    """
    from .phase_coherence import compute_phase_alignment

    return compute_phase_alignment(G, node, radius=radius)
