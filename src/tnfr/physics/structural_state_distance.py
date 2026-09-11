r"""A metric on fixed-topology TNFR graph states modulo relabeling.

This module gives a restricted exact result for the structural-state geometry
problem.  A state assigns five scalar coordinates to every node: ``EPI``,
``nu_f``, circular phase, ``DeltaNFR`` and ``dEPI``, together with effective
conductance and structural-length coordinates per edge.  After the caller
supplies one positive scale for each channel, their scaled Euclidean product
is a metric on a labelled graph.  The phase factor uses the geodesic distance
on the circle with period ``2*pi``.  A missing edge ``weight`` means unit
conductance.  Explicit ``length`` is the structural path coordinate; absent
``length``, the legacy ``weight`` fallback is used.

For two isomorphic graphs, the distance returned here is the minimum product
distance over every topology- and label-preserving graph isomorphism.  This is
an exact metric on the corresponding isomorphism classes:

* non-negativity and symmetry follow from the product metric and inverse
  isomorphisms;
* distance zero holds exactly when a state-preserving isomorphism exists; and
* the triangle inequality follows by composing minimizing isomorphisms.

The minimum exists because a finite graph has finitely many isomorphisms.
Enumeration can nevertheless be factorial in the number of nodes.  Graphs in
different topology/label classes are rejected because this construction does
not supply a cross-topology distance.  Multigraphs are also rejected: an edge
multiplicity metric would require an additional declared matching contract.
If several isomorphisms attain the same minimum distance with different
channel decompositions, the certificate uses the lexicographically least
``(EPI, frequency, phase, pressure, EPI-rate, conductance, length)`` component
vector.
This order-independent selector changes no metric distance.  A mapping is
published only when exactly one minimizer realizes the selected vector.

The result is read-only.  It reports the residual of the nodal equation
``dEPI = nu_f * DeltaNFR`` at both snapshots but does not silently project an
inconsistent snapshot onto that constraint.  It also does not turn the lossy
coherence scalar ``C=1/(1+|DeltaNFR|+|dEPI|)`` into a complete state geometry.
Extending the construction across changing support, nested EPI identity or
operator histories remains open.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import networkx as nx
from networkx.algorithms import isomorphism as iso

from ..constants.aliases import (
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from ._edge_semantics import effective_edge_length
from ._helpers import finite_real_scalar, wrap_angle

__all__ = [
    "StructuralChannelScales",
    "StructuralStateDistanceCertificate",
    "circular_phase_distance",
    "fixed_topology_structural_state_distance",
]


_PHASE_PERIOD = 2.0 * math.pi
_CHANNEL_NAMES = (
    "EPI",
    "frequency",
    "phase",
    "pressure",
    "EPI rate",
    "edge conductance",
    "edge length",
)


@dataclass(frozen=True, slots=True)
class StructuralChannelScales:
    """Positive reference scales used to make every channel dimensionless.

    No non-phase scale is inferred by the implementation.  A caller comparing
    dimensionless normalized states may explicitly pass ``1.0``; physical
    comparisons should instead pass scales with the corresponding EPI,
    ``Hz_str``, pressure, EPI-rate, edge-conductance and edge-length units.
    ``phase=pi`` is a natural explicit choice because ``pi`` is the exact
    maximum circular phase separation, but it is deliberately not an implicit
    default.  For compatibility with the original six-scale API, omitted
    ``edge_length`` reuses ``edge_conductance``; callers using dimensional edge
    data should declare it explicitly.
    """

    epi: float
    frequency: float
    phase: float
    pressure: float
    epi_rate: float
    edge_conductance: float
    edge_length: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("epi", self.epi),
            ("frequency", self.frequency),
            ("phase", self.phase),
            ("pressure", self.pressure),
            ("epi_rate", self.epi_rate),
            ("edge_conductance", self.edge_conductance),
            (
                "edge_length",
                self.edge_conductance
                if self.edge_length is None
                else self.edge_length,
            ),
        ):
            try:
                parsed = finite_real_scalar(value, f"{name} scale")
            except ValueError as exc:
                raise ValueError(
                    f"{name} scale must be a finite positive real"
                ) from exc
            if parsed <= 0.0:
                raise ValueError(f"{name} scale must be a finite positive real")
        if self.edge_length is None:
            object.__setattr__(self, "edge_length", float(self.edge_conductance))


@dataclass(frozen=True, slots=True)
class StructuralStateDistanceCertificate:
    """Exact quotient-distance result for one fixed topology/label class.

    ``minimizer_count`` exposes automorphism ambiguity at the metric minimum.
    When equal-distance minimizers have different channel decompositions, the
    reported components are the lexicographically least component vector.  If
    several minimizers also share that vector, no arbitrary node
    correspondence is exposed: ``minimizing_mapping`` is empty and
    ``minimizing_mapping_unique`` is false.
    """

    distance: float
    epi_component: float
    frequency_component: float
    phase_component: float
    pressure_component: float
    epi_rate_component: float
    edge_conductance_component: float
    edge_length_component: float
    node_count: int
    admissible_isomorphism_count: int
    minimizer_count: int
    minimizing_mapping: tuple[tuple[Any, Any], ...]
    minimizing_mapping_unique: bool
    node_label_attributes: tuple[str, ...]
    edge_label_attributes: tuple[str, ...]
    left_nodal_equation_residual_linf: float
    right_nodal_equation_residual_linf: float
    phase_period: float
    exact_metric_on_isomorphism_classes: bool
    scope: str


def circular_phase_distance(left: float, right: float) -> float:
    """Return the geodesic separation of two finite phases modulo ``2*pi``."""
    a = _finite_real(left, "left phase")
    b = _finite_real(right, "right phase")
    # Reduce before subtracting so two individually finite, very large phase
    # representatives cannot overflow in ``a-b``.  The shared TNFR wrapper
    # maps the bounded signed difference to the canonical [-pi, pi) interval.
    return abs(wrap_angle(wrap_angle(a) - wrap_angle(b)))


def _finite_real(value: Any, name: str) -> float:
    return finite_real_scalar(value, name)


def _attribute_names(value: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of attribute names")
    try:
        names = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of attribute names") from exc
    if any(not isinstance(item, str) or not item for item in names):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} must not contain duplicate names")
    return names


def _required_channel(
    data: Mapping[str, Any],
    aliases: Sequence[str],
    *,
    side: str,
    node: Any,
    channel: str,
) -> float:
    """Read the first canonical alias without inventing a missing value."""
    for attribute in aliases:
        if attribute in data:
            value = _finite_real(
                data[attribute], f"{side} node {node!r} {channel}"
            )
            if channel == "frequency" and value < 0.0:
                raise ValueError(
                    f"{side} node {node!r} frequency must be nonnegative"
                )
            return value
    raise ValueError(f"{side} node {node!r} is missing required {channel}")


def _extract_state(graph: nx.Graph, *, side: str) -> dict[Any, tuple[float, ...]]:
    state: dict[Any, tuple[float, ...]] = {}
    for node, data in graph.nodes(data=True):
        state[node] = (
            _required_channel(data, ALIAS_EPI, side=side, node=node, channel="EPI"),
            _required_channel(
                data, ALIAS_VF, side=side, node=node, channel="frequency"
            ),
            _required_channel(
                data, ALIAS_THETA, side=side, node=node, channel="phase"
            ),
            _required_channel(
                data, ALIAS_DNFR, side=side, node=node, channel="pressure"
            ),
            _required_channel(
                data, ALIAS_DEPI, side=side, node=node, channel="EPI rate"
            ),
        )
    return state


def _edge_key(graph: nx.Graph, source: Any, target: Any) -> Any:
    """Return a direction-aware key without requiring comparable node ids."""
    if graph.is_directed():
        return source, target
    return frozenset((source, target))


def _extract_edge_state(
    graph: nx.Graph, *, side: str
) -> tuple[dict[Any, float], dict[Any, float]]:
    """Read finite nonnegative conductance and structural path length."""
    conductance: dict[Any, float] = {}
    length: dict[Any, float] = {}
    for source, target, data in graph.edges(data=True):
        weight = _finite_real(
            data.get("weight", 1.0),
            f"{side} edge ({source!r}, {target!r}) conductance",
        )
        if weight < 0.0:
            raise ValueError(
                f"{side} edge ({source!r}, {target!r}) conductance "
                "must be nonnegative"
            )
        try:
            edge_length = effective_edge_length(data)
        except ValueError as exc:
            raise ValueError(
                f"{side} edge ({source!r}, {target!r}) structural length "
                "must be a finite nonnegative real"
            ) from exc
        key = _edge_key(graph, source, target)
        conductance[key] = weight
        length[key] = edge_length
    return conductance, length


def _scaled_difference(
    left: float, right: float, scale: float, *, channel: str
) -> float:
    """Return ``(left-right)/scale`` or reject unrepresentable arithmetic."""
    if left == right:
        return 0.0
    difference = left - right
    if math.isfinite(difference):
        result = difference / scale
        if math.isfinite(result) and result != 0.0:
            return result

    # The direct subtraction or division lost floating-point range.  Exact
    # binary-rational arithmetic recovers every case whose final ratio is
    # representable and turns genuine overflow/underflow into an explicit
    # scope error rather than a false non-isomorphism or zero distance.
    from fractions import Fraction

    exact_difference = Fraction.from_float(left) - Fraction.from_float(right)
    exact = exact_difference / Fraction.from_float(scale)
    try:
        result = float(exact)
    except OverflowError as exc:
        raise ValueError(
            f"{channel} scaled difference exceeds finite floating-point range"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(
            f"{channel} scaled difference exceeds finite floating-point range"
        )
    if result == 0.0 and exact:
        raise ValueError(
            f"{channel} scaled difference is below nonzero floating-point range"
        )
    return result


def _finite_hypot(values: Sequence[float], *, channel: str) -> float:
    # A canonical magnitude order removes graph insertion order from the
    # floating-point reduction while preserving the Euclidean norm.
    result = math.hypot(*sorted(abs(value) for value in values))
    if not math.isfinite(result):
        raise ValueError(f"{channel} distance exceeds finite floating-point range")
    return result


def _label_match(attributes: tuple[str, ...]):
    """Return an exact matcher that distinguishes missing attributes."""
    missing = object()

    def match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        for attribute in attributes:
            left_value = left.get(attribute, missing)
            right_value = right.get(attribute, missing)
            if left_value is missing or right_value is missing:
                if left_value is not right_value:
                    return False
                continue
            try:
                equal = left_value == right_value
                if not isinstance(equal, bool):
                    equal = bool(equal)
            except (TypeError, ValueError):
                return False
            if not equal:
                return False
        return True

    return match


def _graph_matcher(
    left: nx.Graph,
    right: nx.Graph,
    *,
    node_labels: tuple[str, ...],
    edge_labels: tuple[str, ...],
):
    matcher_class = iso.DiGraphMatcher if left.is_directed() else iso.GraphMatcher
    kwargs: dict[str, Any] = {}
    if node_labels:
        kwargs["node_match"] = _label_match(node_labels)
    if edge_labels:
        kwargs["edge_match"] = _label_match(edge_labels)
    return matcher_class(left, right, **kwargs)


def _nodal_residual_linf(state: Mapping[Any, tuple[float, ...]]) -> float:
    if not state:
        return 0.0
    residuals: list[float] = []
    for _, frequency, _, pressure, epi_rate in state.values():
        product = frequency * pressure
        if not math.isfinite(product) or (
            frequency != 0.0 and pressure != 0.0 and product == 0.0
        ):
            raise ValueError(
                "nodal-equation product exceeds finite nonzero floating-point range"
            )
        residuals.append(
            abs(
                _scaled_difference(
                    epi_rate,
                    product,
                    1.0,
                    channel="nodal-equation residual",
                )
            )
        )
    return max(residuals)


def fixed_topology_structural_state_distance(
    left: nx.Graph,
    right: nx.Graph,
    *,
    scales: StructuralChannelScales,
    node_label_attributes: Sequence[str] = (),
    edge_label_attributes: Sequence[str] = (),
) -> StructuralStateDistanceCertificate:
    r"""Minimize a declared product metric over all graph isomorphisms.

    Bare adjacency, edge direction and self-loops are always part of the
    admissibility test.  Effective edge conductance (the finite nonnegative
    ``weight``, with unit default) and effective structural path length are
    metric channels.  Explicit ``length`` defines the latter; ``weight`` is its
    compatibility fallback.
    ``node_label_attributes`` and
    ``edge_label_attributes`` add exact categorical constraints.  A persistent
    node-identity label can therefore distinguish a genuine support change
    from a relabeling.  Structural channels themselves remain costs rather than
    hard labels.

    Raises
    ------
    TypeError
        If either input is not a NetworkX graph or ``scales`` has the wrong
        type.
    ValueError
        If the graph kinds differ, a multigraph is supplied, required state is
        missing/invalid, or no label-preserving graph isomorphism exists.
    """
    if not isinstance(left, nx.Graph) or not isinstance(right, nx.Graph):
        raise TypeError("left and right must be NetworkX graph instances")
    if not isinstance(scales, StructuralChannelScales):
        raise TypeError("scales must be a StructuralChannelScales instance")
    if left.is_multigraph() or right.is_multigraph():
        raise ValueError(
            "multigraph state distance is outside this fixed-support scope"
        )
    if left.is_directed() != right.is_directed():
        raise ValueError(
            "directed and undirected graphs are different topology classes"
        )

    node_labels = _attribute_names(node_label_attributes, "node_label_attributes")
    edge_labels = _attribute_names(edge_label_attributes, "edge_label_attributes")
    left_state = _extract_state(left, side="left")
    right_state = _extract_state(right, side="right")
    left_conductance, left_length = _extract_edge_state(left, side="left")
    right_conductance, right_length = _extract_edge_state(right, side="right")

    matcher = _graph_matcher(
        left,
        right,
        node_labels=node_labels,
        edge_labels=edge_labels,
    )
    nodes = tuple(left.nodes())
    assert scales.edge_length is not None
    scale_values = (
        float(scales.epi),
        float(scales.frequency),
        float(scales.phase),
        float(scales.pressure),
        float(scales.epi_rate),
        float(scales.edge_conductance),
        float(scales.edge_length),
    )
    best_distance = math.inf
    best_components: tuple[float, ...] | None = None
    best_mapping: dict[Any, Any] | None = None
    minimizer_count = 0
    component_minimizer_count = 0
    count = 0

    for mapping in matcher.isomorphisms_iter():
        count += 1
        component_values: list[list[float]] = [[] for _ in range(7)]
        for node in nodes:
            left_values = left_state[node]
            right_values = right_state[mapping[node]]
            for index in (0, 1, 3, 4):
                component_values[index].append(
                    _scaled_difference(
                        left_values[index],
                        right_values[index],
                        scale_values[index],
                        channel=_CHANNEL_NAMES[index],
                    )
                )
            component_values[2].append(
                _scaled_difference(
                    circular_phase_distance(left_values[2], right_values[2]),
                    0.0,
                    scale_values[2],
                    channel="phase",
                )
            )
        for source, target in left.edges():
            left_weight = left_conductance[_edge_key(left, source, target)]
            right_weight = right_conductance[
                _edge_key(right, mapping[source], mapping[target])
            ]
            component_values[5].append(
                _scaled_difference(
                    left_weight,
                    right_weight,
                    scale_values[5],
                    channel="edge conductance",
                )
            )
            left_edge_length = left_length[_edge_key(left, source, target)]
            right_edge_length = right_length[
                _edge_key(right, mapping[source], mapping[target])
            ]
            component_values[6].append(
                _scaled_difference(
                    left_edge_length,
                    right_edge_length,
                    scale_values[6],
                    channel="edge length",
                )
            )
        components = tuple(
            _finite_hypot(values, channel=channel)
            for values, channel in zip(component_values, _CHANNEL_NAMES)
        )
        distance = _finite_hypot(components, channel="structural-state")
        if distance < best_distance:
            best_distance = distance
            best_components = components
            best_mapping = dict(mapping)
            minimizer_count = 1
            component_minimizer_count = 1
        elif distance == best_distance:
            minimizer_count += 1
            if best_components is None or components < best_components:
                best_components = components
                best_mapping = dict(mapping)
                component_minimizer_count = 1
            elif components == best_components:
                component_minimizer_count += 1

    if count == 0 or best_components is None or best_mapping is None:
        raise ValueError(
            "graphs are not isomorphic under the declared topology and label contract"
        )

    mapping_unique = component_minimizer_count == 1
    public_mapping = (
        tuple((node, best_mapping[node]) for node in nodes)
        if mapping_unique
        else ()
    )

    return StructuralStateDistanceCertificate(
        distance=best_distance,
        epi_component=best_components[0],
        frequency_component=best_components[1],
        phase_component=best_components[2],
        pressure_component=best_components[3],
        epi_rate_component=best_components[4],
        edge_conductance_component=best_components[5],
        edge_length_component=best_components[6],
        node_count=len(nodes),
        admissible_isomorphism_count=count,
        minimizer_count=minimizer_count,
        minimizing_mapping=public_mapping,
        minimizing_mapping_unique=mapping_unique,
        node_label_attributes=node_labels,
        edge_label_attributes=edge_labels,
        left_nodal_equation_residual_linf=_nodal_residual_linf(left_state),
        right_nodal_equation_residual_linf=_nodal_residual_linf(right_state),
        phase_period=_PHASE_PERIOD,
        exact_metric_on_isomorphism_classes=True,
        scope=(
            "EXACT for finite simple graphs within one declared topology/label "
            "class; all isomorphisms are enumerated (factorial worst case), "
            "equal-distance component ties use a lexicographic rule, and an "
            "unresolved mapping is suppressed; cross-topology, multigraph, "
            "nested-EPI and history geometry OPEN"
        ),
    )
