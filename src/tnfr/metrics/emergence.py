"""Retrospective THOL record heuristics, not autonomous emergence certificates."""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import NodeId, TNFRGraph

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI
from ..glyph_history import current_operator_step
from ..types import require_finite_real_scalar_epi

__all__ = [
    "compute_structural_complexity",
    "compute_bifurcation_rate",
    "compute_metabolic_efficiency",
    "compute_emergence_index",
]


def compute_structural_complexity(G: TNFRGraph, node: NodeId) -> int:
    """Measure structural complexity by counting nested sub-EPIs.

    This counts retained sub-EPI descriptors, not independent degrees of
    freedom, live child nodes or a mathematical bifurcation of an evolution law.

    Parameters
    ----------
    G : TNFRGraph
        Graph containing the node
    node : NodeId
        Node identifier

    Returns
    -------
    int
        Number of retained sub-EPI descriptors

    Notes
    -----
    The count alone establishes neither organization quality nor a capacity
    requirement. Descriptor deletion changes this retrospective count.

    Examples
    --------
    >>> from tnfr.structural import create_nfr
    >>> from tnfr.metrics.emergence import compute_structural_complexity
    >>> G, node = create_nfr("system", epi=0.5, vf=1.0)
    >>> # Supplied record fixture; no formation or bifurcation is executed.
    >>> G.nodes[node]["sub_epis"] = [{"timestamp": 1}]
    >>> complexity = compute_structural_complexity(G, node)
    >>> complexity
    1
    """
    sub_epis = G.nodes[node].get("sub_epis", [])
    return len(sub_epis)


def compute_bifurcation_rate(G: TNFRGraph, node: NodeId, window: int = 10) -> float:
    """Count recent sub-EPI records per operator step in a fixed window.

    The compatibility name refers to recorded THOL descriptors; this reading
    does not detect a bifurcation of a continuous dynamical system.

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
    This uses operator indices, not elapsed physical time. It counts retained
    descriptors in a fixed-width window, including during startup; it is not
    an adaptive intensity or a stability diagnostic. Multiple records at one
    step can yield a value above one. Future records are rejected when an
    explicit current operator counter is present.

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
    >>> rate  # 3 records in the interval (2, 12]
    0.3
    """
    return float(_bifurcation_ratio(G, node, window))


def _bifurcation_ratio(G: TNFRGraph, node: NodeId, window: int) -> Fraction:
    """Keep the represented record count exact for downstream composition."""
    if isinstance(window, bool) or not isinstance(window, Integral) or window <= 0:
        raise ValueError("window must be a positive integer")
    window = int(window)

    sub_epis = G.nodes[node].get("sub_epis", [])
    if not sub_epis:
        return Fraction(0)

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

    # Only legacy records without a counter may infer an endpoint from records.
    if "_operator_step" in node_data:
        counter = node_data["_operator_step"]
        if (
            isinstance(counter, bool)
            or not isinstance(counter, Integral)
            or counter < 0
        ):
            raise ValueError("operator counter must be a nonnegative integer step")
        current_time = max(int(counter), current_operator_step(node_data))
        if max(timestamps) > current_time:
            raise ValueError("sub-EPI timestamp exceeds the current operator step")
    else:
        current_time = max(current_operator_step(node_data), max(timestamps))
    recent_count = sum(
        current_time - window < timestamp <= current_time for timestamp in timestamps
    )
    return Fraction(recent_count, window)


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
    The caller must align ``epi_initial`` with the retained glyph window;
    history eviction does not preserve a lifetime THOL denominator. Unsupported
    scalar EPI or a nonrepresentable result raises rather than reporting zero.

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
    ratio = _metabolic_ratio(G, node)
    try:
        result = float(ratio)
    except OverflowError as exc:
        raise ValueError("metabolic efficiency exceeds finite float range") from exc
    return result


def _metabolic_ratio(G: TNFRGraph, node: NodeId) -> Fraction:
    """Read one signed chart and avoid overflow before division by the count."""
    from ..types import Glyph

    # Count T'HOL applications
    glyph_history = G.nodes[node].get("glyph_history", [])
    thol_count = sum(1 for g in glyph_history if g == "THOL" or g == Glyph.THOL.value)

    if thol_count == 0:
        return Fraction(0)

    # Calculate EPI delta
    current_epi = get_attr(
        G.nodes[node],
        ALIAS_EPI,
        0.0,
        strict=True,
        conv=require_finite_real_scalar_epi,
    )
    initial_epi = require_finite_real_scalar_epi(
        G.nodes[node].get("epi_initial", current_epi), "initial EPI"
    )
    return (Fraction(current_epi) - Fraction(initial_epi)) / thol_count


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
    - Complexity: number of retained sub-EPI descriptors
    - Rate: recent descriptor count per operator step
    - Efficiency: signed EPI change per retained THOL application

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
    complexity = compute_structural_complexity(G, node)
    rate = _bifurcation_ratio(G, node, 10)
    efficiency = _metabolic_ratio(G, node)

    # A negative net EPI change is a loss, not a complex-valued emergence
    # magnitude. Exact zeros remain zero rather than being lifted by epsilon.
    product = complexity * rate * max(efficiency, 0)
    if product == 0:
        return 0.0
    # Normalize before the approximate cube root: the final result can be
    # representable even when the product overflows or rounds to float zero.
    exponent = (product.numerator.bit_length() - product.denominator.bit_length()) // 3
    scaled = product / Fraction(2) ** (3 * exponent)
    return math.ldexp(float(scaled) ** (1.0 / 3.0), exponent)
