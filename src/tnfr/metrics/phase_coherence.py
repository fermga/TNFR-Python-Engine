"""Read-only local and global phase-order diagnostics.

The Kuramoto magnitude measures cancellation of represented unit phasors; a
small value need not mean disorder (a regular alternating pattern can cancel).
It is distinct from structural coherence C(t), a pairwise U3 admission test,
and any theorem about operator action or persistence. In particular IL does
not promise to increase this readout.

Phase materialization and phasor reduction reuse the engine's trigonometric
and Kuramoto owners. A vanishing resultant has a defined magnitude of zero;
these functions neither request nor supply a mean direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .local_coherence import _radius_nodes
from .trig_cache import compute_theta_trig

if TYPE_CHECKING:
    from ..types import TNFRGraph

__all__ = [
    "compute_phase_alignment",
    "compute_global_phase_coherence",
]


def _phase_order_on_nodes(G: TNFRGraph, nodes: tuple[Any, ...]) -> float:
    """Reduce the selected phases without skipping invalid stored evidence."""
    # Import locally: gamma itself imports metrics during package startup.
    from ..gamma import _kuramoto_from_trig

    trig = compute_theta_trig((node, G.nodes[node]) for node in nodes)
    if len(trig.order) <= 1:
        # Preserve the historical empty/singleton observation conventions, but
        # validate the singleton phase before returning the trivial magnitude.
        return 1.0
    magnitude, _ = _kuramoto_from_trig(trig)
    return min(1.0, magnitude)


def compute_phase_alignment(G: TNFRGraph, node: Any, radius: int = 1) -> float:
    """Return phase-order magnitude on the center-inclusive graph ball.

    Radius must be a nonnegative integer and the center must exist. Directed
    graphs use outgoing support paths; edge weights do not change membership
    and parallel edges do not repeat a node. Only selected phases are consumed.
    The shared phase reader uses the authoritative alias and a zero convention
    when every phase alias is absent; malformed stored values raise.

    Radius zero and an admitted isolated node return 1.0. A two-node ball can
    have any order magnitude between zero and one, depending on phase separation.
    A vanishing resultant is a valid zero magnitude, not an available direction.
    The final upper clamp removes floating overshoot of the unit bound; it is
    neither a coupling threshold nor a stability policy. No graph data is written.
    """
    return _phase_order_on_nodes(G, _radius_nodes(G, node, radius))


def compute_global_phase_coherence(G: TNFRGraph) -> float:
    """Return phase-order magnitude across all graph nodes.

    This is the same observation as local alignment with the entire node set.
    The historical empty-graph convention is 1.0, not measured synchronization
    evidence. A singleton is also 1.0 after phase validation. Missing phase
    aliases use the shared zero convention; invalid present values raise rather
    than being dropped or replaced by another alias. No graph data is written.
    """
    return _phase_order_on_nodes(G, tuple(G.nodes()))
