"""Exact detached transport budgets on materialized symmetric conductance.

These read-outs reuse the engine's weighted EPI adjacency and unweighted
support-neighbor conventions. They neither evolve a graph nor authenticate
an execution. Stored pressure may include phase, operator writes or stale
values; it is never silently identified with pure EPI diffusion.
"""

from dataclasses import dataclass
from fractions import Fraction

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..alias import get_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_VF
from ._conductance import read_conductance
from ._cycle_algebra import Vector, dot, ordered_vector
from .structural_diffusion import structural_field

__all__ = [
    "SupportTransportSnapshot", "SupportTransportReset", "SupportTransportEuler",
    "observe_support_transport", "observe_support_transport_reset",
    "observe_support_transport_euler",
]


@dataclass(frozen=True)
class SupportTransportSnapshot:
    """Exact reference to finite represented state; no causal proof seal."""

    nodes: tuple
    conductance: tuple
    support_neighbors: tuple
    epi: Vector
    capacity: Vector
    stored_pressure: Vector
    epi_gradient: Vector
    capacity_gradient: Vector
    topology_gradient: Vector
    dirichlet_gradient: Vector
    rate: Vector
    dirichlet_energy: Fraction
    energy_rate: Fraction


def _laplacian(conductance, values):
    result = [Fraction(0) for _ in values]
    for i, j, weight in conductance:
        result[i] += weight * (values[i] - values[j])
    return tuple(result)


def _energy(conductance, values):
    return sum(
        (weight * (values[i] - values[j])**2 for i, j, weight in conductance),
        Fraction(0),
    ) / 4


def _from_data(nodes, conductance, support_neighbors, epi, capacity, pressure):
    nodes = tuple(nodes)
    size = len(nodes)
    if len(set(nodes)) != size:
        raise ValueError("node order must contain distinct nodes")
    x, nu, p = (
        ordered_vector(values, label)
        for values, label in ((epi, "epi"), (capacity, "capacity"),
                              (pressure, "stored_pressure"))
    )
    if any(len(values) != size for values in (x, nu, p)):
        raise ValueError("state vectors must match the node order")
    if any(value < 0 for value in nu):
        raise ValueError("capacity must be nonnegative")
    support = tuple(tuple(row) for row in support_neighbors)
    if len(support) != size:
        raise ValueError("support rows must match the node order")
    for row in support:
        if len(set(row)) != len(row) or any(
            type(j) is not int or not 0 <= j < size for j in row
        ):
            raise ValueError("support must contain distinct valid neighbor indices")
    entries = {}
    for i, j, raw_weight in conductance:
        if (type(i) is not int or type(j) is not int
                or not 0 <= i < size or not 0 <= j < size or j not in support[i]):
            raise ValueError("conductance indices must belong to the support")
        weight = exact_or_represented_real(raw_weight, "conductance")
        if weight <= 0 or (i, j) in entries:
            raise ValueError("conductance entries must be unique and positive")
        entries[i, j] = weight
    if any(entries.get((j, i)) != weight for (i, j), weight in entries.items()):
        raise ValueError("transport energy requires symmetric conductance")
    edges = tuple((i, j, weight) for (i, j), weight in sorted(entries.items()))
    strengths = [Fraction(0) for _ in nodes]
    for i, _, weight in edges:
        strengths[i] += weight
    bx = _laplacian(edges, x)
    epi_gradient = tuple(-value / d if d else Fraction(0)
                         for value, d in zip(bx, strengths))
    degree = tuple(len(row) for row in support)

    def support_gradient(values):
        return tuple(
            sum((values[j] - values[i] for j in row), Fraction(0)) / len(row)
            if row else Fraction(0)
            for i, row in enumerate(support)
        )

    rate = tuple(v * pressure_i for v, pressure_i in zip(nu, p))
    return SupportTransportSnapshot(
        nodes, edges, support, x, nu, p, epi_gradient,
        support_gradient(nu), support_gradient(degree), bx, rate,
        _energy(edges, x), dot(bx, rate),
    )


def observe_support_transport(G) -> SupportTransportSnapshot:
    """Read weighted EPI and unweighted capacity/topology gradients exactly.

    Effective conductances (including parallel aggregation) and scalar state
    are materialized by shared binary64 readers, then treated as exact real
    coefficients. Loops count once per row; zero-weight edges remain in the
    support channels but carry no EPI flux. Isolates have zero gradients.
    No pressure refresh or graph/cache write occurs. The nonlinear canonical
    phase channel is deliberately outside this exact rational decomposition.
    ``rate`` is the exact product of represented capacity and stored pressure,
    not an assertion that the latter was refreshed or that a step was taken.
    """
    adjacency = read_conductance(G, symmetric=True)
    nodes = tuple(adjacency.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    support = tuple(tuple(sorted(index[j] for j in G.neighbors(node))) for node in nodes)

    def state(alias):
        return tuple(
            finite_represented_real(
                get_attr(G.nodes[node], alias, 0.0, conv=lambda v: v, strict=True),
                alias[0],
            )[1]
            for node in nodes
        )

    edges = tuple(
        (int(i), int(j), Fraction(float(weight)))
        for i, j, weight in zip(adjacency.source, adjacency.target, adjacency.weight)
    )
    return _from_data(
        nodes, edges, support, tuple(structural_field(G, list(nodes))),
        state(ALIAS_VF), state(ALIAS_DNFR),
    )


def _rebuild(value):
    if type(value) is not SupportTransportSnapshot:
        raise TypeError("state must be a SupportTransportSnapshot")
    # Detached public fields are data, not provenance. Recompute all caches.
    return _from_data(value.nodes, value.conductance, value.support_neighbors,
                      value.epi, value.capacity, value.stored_pressure)


@dataclass(frozen=True)
class SupportTransportReset:
    """Same-EPI energy change from an observed conductance change."""

    before: SupportTransportSnapshot
    after: SupportTransportSnapshot
    energy_change: Fraction
    edge_energy_change: Fraction
    identity_residual: Fraction


def observe_support_transport_reset(before, after) -> SupportTransportReset:
    """Account for added, removed or reweighted edges on one aligned node set.

    Birth itself is outside this identity: capture the first snapshot after
    birth, before attachment. Capacity and pressure may differ; this energy
    identity does not attribute their changes to the edge operation alone.
    """
    before, after = _rebuild(before), _rebuild(after)
    if before.nodes != after.nodes or before.epi != after.epi:
        raise ValueError("a support reset requires identical node order and EPI")
    weights = {(i, j): -weight for i, j, weight in before.conductance}
    for i, j, weight in after.conductance:
        weights[i, j] = weights.get((i, j), Fraction(0)) + weight
    edge_change = _energy(
        tuple((i, j, weight) for (i, j), weight in weights.items()), before.epi,
    )
    change = after.dirichlet_energy - before.dirichlet_energy
    return SupportTransportReset(before, after, change, edge_change, change - edge_change)


@dataclass(frozen=True)
class SupportTransportEuler:
    """Exact energy accounting for an observed endpoint and held nodal rate."""

    before: SupportTransportSnapshot
    after: SupportTransportSnapshot
    dt: Fraction
    expected_epi: Vector
    state_defect: Vector
    drift_term: Fraction
    quadratic_term: Fraction
    defect_term: Fraction
    energy_change: Fraction
    identity_residual: Fraction


def observe_support_transport_euler(before, after, dt) -> SupportTransportEuler:
    """Decompose endpoint energy on fixed conductance and capacity.

    With r=nu*p_before, the exact Euler reference is y=x+h*r. The observed
    defect delta=x_after-y contributes (B*y) dot delta + delta^T B delta/2.
    This residual can include rounding, clipping or a different evolution;
    the algebra alone never certifies the caller's solver or refresh history.
    """
    before, after = _rebuild(before), _rebuild(after)
    if (before.nodes != after.nodes or before.conductance != after.conductance
            or before.capacity != after.capacity):
        raise ValueError("Euler budget requires fixed node order, conductance and capacity")
    h = exact_or_represented_real(dt, "dt")
    if h < 0:
        raise ValueError("dt must be nonnegative")
    expected = tuple(x + h * r for x, r in zip(before.epi, before.rate))
    defect = tuple(x - y for x, y in zip(after.epi, expected))
    drift = h * before.energy_rate
    quadratic = h**2 * _energy(before.conductance, before.rate)
    defect_term = (dot(_laplacian(before.conductance, expected), defect)
                   + _energy(before.conductance, defect))
    change = after.dirichlet_energy - before.dirichlet_energy
    return SupportTransportEuler(
        before, after, h, expected, defect, drift, quadratic, defect_term, change,
        change - drift - quadratic - defect_term,
    )
