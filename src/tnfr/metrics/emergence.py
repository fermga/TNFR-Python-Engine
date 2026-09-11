"""Emergence metrics for T'HOL structural metabolism.

Provides quantitative measures of complexity emergence, bifurcation dynamics,
and metabolic efficiency in self-organizing systems.
"""

from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI
from ..glyph_history import current_operator_step

__all__ = [
    "compute_structural_complexity",
    "compute_bifurcation_rate",
    "compute_metabolic_efficiency",
    "compute_emergence_index",
]


def compute_structural_complexity(G: TNFRGraph, node: NodeId) -> int:
    """Measure structural complexity by counting nested sub-EPIs.

    Structural complexity reflects the number of bifurcations that have
    occurred, indicating the degree of self-organized internal structure.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier

    Returns
    -------
    int
        Number of sub-EPIs generated through T'HOL bifurcations

    Notes
    -----
    Higher complexity indicates more sophisticated internal organization
    but may also indicate higher maintenance costs (higher νf required).

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.operators.definitions import SelfOrganization
    >>> from tnfr.metrics.emergence import compute_structural_complexity
    >>> G, node = create_nfr("system", epi=0.5, vf=1.0)
    >>> # Initialize history for bifurcation
    >>> G.nodes[node]["epi_history"] = [0.3, 0.4, 0.6]  # Accelerating
    >>> SelfOrganization()(G, node, tau=0.05)  # Low threshold
    >>> complexity = compute_structural_complexity(G, node)
    >>> complexity  # doctest: +SKIP
    1
    """
    sub_epis = G.nodes[node].get("sub_epis", [])
    return len(sub_epis)


def compute_bifurcation_rate(G: TNFRGraph, node: NodeId, window: int = 10) -> float:
    """Calculate frequency of bifurcations in recent history.

    Bifurcation rate indicates how actively the node is generating new
    structural complexity through T'HOL operations.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier
    window : int
        Time window for rate calculation (in operator steps, default 10)

    Returns
    -------
    float
        Bifurcations per step in the window (0.0 to 1.0 typical)

    Notes
    -----
    High bifurcation rate (> 0.5) may indicate:
    - Active adaptation to changing environment
    - High structural instability
    - Rich exploratory dynamics

    Low rate (< 0.1) may indicate:
    - Stable structural regime
    - Low adaptive pressure
    - Insufficient ΔNFR for bifurcation

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.metrics.emergence import compute_bifurcation_rate
    >>> G, node = create_nfr("evolving", epi=0.6, vf=1.0)
    >>> # Simulate several bifurcations
    >>> G.nodes[node]["sub_epis"] = [
    ...     {"timestamp": 5}, {"timestamp": 8}, {"timestamp": 12}
    ... ]
    >>> rate = compute_bifurcation_rate(G, node, window=10)
    >>> rate  # 2 bifurcations in last 10 steps
    0.2
    """
    if isinstance(window, bool) or not isinstance(window, Integral) or window <= 0:
        raise ValueError("window must be a positive integer")
    window = int(window)

    sub_epis = G.nodes[node].get("sub_epis", [])
    if not sub_epis:
        return 0.0

    node_data = G.nodes[node]
    timestamps: list[int] = []
    for record in sub_epis:
        raw_timestamp = record.get("timestamp", 0)
        if (
            isinstance(raw_timestamp, bool)
            or not isinstance(raw_timestamp, Integral)
            or raw_timestamp < 0
        ):
            raise ValueError("sub-EPI timestamps must be nonnegative integer steps")
        timestamps.append(int(raw_timestamp))

    # The explicit counter survives bounded glyph-history eviction.  The
    # timestamp maximum keeps legacy records readable when no counter exists.
    current_time = max(current_operator_step(node_data), max(timestamps, default=0))
    recent_count = sum(
        current_time - window < timestamp <= current_time
        for timestamp in timestamps
    )
    return recent_count / float(window)


def compute_metabolic_efficiency(G: TNFRGraph, node: NodeId) -> float:
    """Calculate EPI gain per T'HOL application (metabolic efficiency).

    Metabolic efficiency is a retrospective ratio of signed EPI change to
    recorded T'HOL applications under one fixed observation protocol.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier

    Returns
    -------
    float
        Average EPI increase per T'HOL application
        Returns 0.0 if no T'HOL applications recorded

    Notes
    -----
    This is a signed net EPI change divided by the number of recorded THOL
    applications. It is a retrospective heuristic and does not attribute the
    change causally to THOL when other operators occur in the same interval.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.metrics.emergence import compute_metabolic_efficiency
    >>> G, node = create_nfr("productive", epi=0.5, vf=1.0)
    >>> # Record initial EPI
    >>> G.nodes[node]["epi_initial"] = 0.3
    >>> # Simulate T'HOL applications
    >>> G.nodes[node]["glyph_history"] = ["THOL", "THOL", "IL"]
    >>> # Current EPI increased to 0.5
    >>> efficiency = compute_metabolic_efficiency(G, node)
    >>> efficiency  # (0.5 - 0.3) / 2 = 0.1
    0.1
    """
    from ..types import Glyph

    # Count T'HOL applications
    glyph_history = G.nodes[node].get("glyph_history", [])
    thol_count = sum(1 for g in glyph_history if g == "THOL" or g == Glyph.THOL.value)

    if thol_count == 0:
        return 0.0

    # Calculate EPI delta
    current_epi = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
    initial_epi = float(G.nodes[node].get("epi_initial", current_epi))

    epi_gain = current_epi - initial_epi

    return epi_gain / float(thol_count)


def compute_emergence_index(G: TNFRGraph, node: NodeId) -> float:
    """Composite metric combining complexity, rate, and efficiency.

    Emergence index combines recorded structural complexity, bifurcation rate,
    and retrospective EPI efficiency under one fixed observation protocol.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier

    Returns
    -------
    float
        Nonnegative, unbounded heuristic geometric mean:
        cbrt(complexity * rate * max(efficiency, 0)).

    Notes
    -----
    This index balances three factors:
    - Complexity: how much structure has emerged
    - Rate: how actively new structure forms
    - Efficiency: how productive each reorganization is

    The three factors have different units and are not normalized, so the
    result is suitable only for comparisons made with the same sampling and
    history protocol. A zero or negative net EPI efficiency yields zero.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.metrics.emergence import compute_emergence_index
    >>> G, node = create_nfr("emergent", epi=0.7, vf=1.0)
    >>> # Setup for high emergence
    >>> G.nodes[node]["epi_initial"] = 0.3
    >>> G.nodes[node]["glyph_history"] = ["THOL", "THOL", "IL"]
    >>> G.nodes[node]["sub_epis"] = [{"timestamp": 1}, {"timestamp": 2}]
    >>> index = compute_emergence_index(G, node)
    >>> index  # doctest: +SKIP
    0.430886...
    """
    complexity = float(compute_structural_complexity(G, node))
    rate = compute_bifurcation_rate(G, node)
    efficiency = compute_metabolic_efficiency(G, node)

    # A negative net EPI change is a loss, not a complex-valued emergence
    # magnitude. Exact zeros remain zero rather than being lifted by epsilon.
    product = complexity * rate * max(efficiency, 0.0)
    return float(product ** (1.0 / 3.0)) if product > 0.0 else 0.0
