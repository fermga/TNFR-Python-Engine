"""Immutable pre-EN reads and the shared Reception scalar kernel.

Reception source discovery and neighbour integration must describe one
temporal state. ``ReceptionReadSnapshot`` materializes both observations
before EN writes EPI or its semantic kind. Direct execution and the
two-phase network stage consume the same value object, so later graph writes
cannot alter the neighbour mean or source telemetry reported for that call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import Any

from ..alias import get_attr
from ..constants.aliases import ALIAS_EPI, ALIAS_EPI_KIND
from ..types import Glyph
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_proof_signature,
)
from ._epi_domain import require_real_scalar_epi
from ._neighbor_epi_kernel import (
    neighbor_epi_blend_value,
    neighbor_epi_represented_affine_row,
    neighbor_epi_unweighted_mean,
    reception_proposed_epi_kind,
)
from ._reception_compatibility import (
    ReceptionCompatibilityDecision,
    _reception_compatibility_payload_stamp,
    reception_compatibility_matches_graph,
)

RECEPTION_PRE_STATE_BOUNDARY = "pre_en_snapshot"
RECEPTION_PRESSURE_OBSERVATION_BOUNDARY = "operator_metrics_collection"
RECEPTION_NO_SOURCES_WARNING_PATTERN = (
    r".*no emission sources detected under the configured telemetry policy.*"
)
_RECEPTION_READ_SNAPSHOT_VERSION = "reception_read_snapshot_v3"
_MISSING_NEIGHBOR_EPI = object()


def _raw_alias(
    graph: Any,
    node: Any,
    aliases: tuple[str, ...],
    default: Any,
) -> Any:
    return get_attr(
        graph.nodes[node],
        aliases,
        default,
        strict=True,
        conv=lambda value: value,
    )


def _node_kind(graph: Any, node: Any) -> str:
    return str(_raw_alias(graph, node, ALIAS_EPI_KIND, ""))


def reception_no_sources_warning(node: Any) -> str:
    """Return the shared empty-source telemetry warning for EN."""

    return (
        f"EN: node {node!r} has no emission sources detected under the "
        "configured telemetry policy."
    )


def reception_input_neighbors(graph: Any, node: Any) -> tuple[Any, ...]:
    """Return EN inputs in causal order for undirected and directed support.

    An arc ``source -> receiver`` carries incoming Reception content, so a
    directed receiver reads predecessors. Undirected support keeps the ordinary
    NetworkX neighbour order.
    """

    directed_probe = getattr(graph, "is_directed", None)
    if directed_probe is None:
        return tuple(graph.neighbors(node))
    if not callable(directed_probe):
        raise TypeError("Reception graph is_directed attribute must be callable")
    directed = bool(directed_probe())
    if directed:
        predecessors = getattr(graph, "predecessors", None)
        if not callable(predecessors):
            raise TypeError("Directed Reception graphs must expose predecessors")
        return tuple(predecessors(node))
    return tuple(graph.neighbors(node))


def _read_explicit_reception_sources(
    graph: Any, node: Any, source_nodes: Any
) -> tuple[tuple[Any, ...], tuple[float, ...]]:
    """Admit ordered source identities and materialize their explicit forms."""

    if type(source_nodes) is not tuple:
        raise TypeError("Reception source_nodes must be a tuple or None")
    if not source_nodes:
        raise ValueError("Reception source_nodes must be nonempty")
    available = set(reception_input_neighbors(graph, node))
    seen: set[Any] = set()
    values: list[float] = []
    for source in source_nodes:
        if source in seen:
            raise ValueError("Reception source_nodes must not repeat a source")
        if source not in available:
            raise ValueError("Reception sources must be realized incoming neighbors")
        seen.add(source)
        raw = _raw_alias(graph, source, ALIAS_EPI, _MISSING_NEIGHBOR_EPI)
        if raw is _MISSING_NEIGHBOR_EPI:
            raise ValueError("An explicit Reception source must have explicit EPI")
        values.append(
            require_real_scalar_epi(
                raw, operator="Reception", label=f"selected source EPI for {source!r}"
            )
        )
    return source_nodes, tuple(values)


def validate_reception_source_nodes(
    graph: Any, node: Any, source_nodes: Any
) -> tuple[Any, ...] | None:
    """Admit a supplied ordered source set, or preserve the legacy None request.

    Explicit sources are unique, nonempty realized incoming neighbors with
    present scalar EPI. Their supplied order is causal for semantic-kind ties.
    This is neither phase/capacity compatibility nor diagnostic source ranking.
    No source coordinate is consumed when the request is None; callers retain
    their existing legacy read and missing-EPI policy in that case.
    """

    if source_nodes is None:
        return None
    return _read_explicit_reception_sources(graph, node, source_nodes)[0]


def _reception_read_snapshot_stamp(
    snapshot: "ReceptionReadSnapshot",
) -> tuple[Any, ...]:
    """Seal EN values while binding graph owners only by object identity."""

    return (
        _RECEPTION_READ_SNAPSHOT_VERSION,
        id(object.__getattribute__(snapshot, "_read_graph_owner")),
        id(object.__getattribute__(snapshot, "_metric_consumer_graph_owner")),
        _reception_read_payload_stamp(snapshot),
        snapshot._graph_identity,
        snapshot._metric_consumer_graph_identity,
    )


def _reception_read_payload_stamp(
    snapshot: "ReceptionReadSnapshot",
) -> tuple[Any, ...]:
    """Sign the materialized EN read without traversing graph owners."""

    return structural_proof_signature(
        (
            snapshot.node,
            snapshot.target_epi,
            snapshot.target_epi_kind,
            snapshot.neighbors,
            snapshot.neighbor_epi_values,
            snapshot.neighbor_dominant_values,
            snapshot.neighbor_epi_kinds,
            snapshot.neighbor_epi_mean,
            snapshot.source_tracking_enabled,
            snapshot.source_max_distance,
            snapshot.reception_sources,
            snapshot.source_nodes,
            snapshot.source_selection_origin,
            (
                _reception_compatibility_payload_stamp(snapshot.compatibility_decision)
                if snapshot.compatibility_decision is not None
                else None
            ),
            snapshot.read_boundary,
        )
    )


@dataclass(frozen=True, slots=True)
class ReceptionReadSnapshot:
    """Fully materialized EN read set from one pre-write graph state."""

    node: Any
    target_epi: float
    target_epi_kind: str
    neighbors: tuple[Any, ...]
    neighbor_epi_values: tuple[float, ...]
    neighbor_dominant_values: tuple[float, ...]
    neighbor_epi_kinds: tuple[str, ...]
    neighbor_epi_mean: float
    source_tracking_enabled: bool
    source_max_distance: int | None
    reception_sources: tuple[tuple[Any, float, float], ...] | None
    _read_graph_owner: Any = field(repr=False, compare=False)
    _metric_consumer_graph_owner: Any = field(repr=False, compare=False)
    _graph_identity: int = field(repr=False, compare=False)
    _metric_consumer_graph_identity: int = field(repr=False, compare=False)
    read_boundary: str = RECEPTION_PRE_STATE_BOUNDARY
    source_nodes: tuple[Any, ...] | None = None
    compatibility_decision: ReceptionCompatibilityDecision | None = None
    _proof_stamp: tuple[Any, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Reject incomplete or contradictory materialized reads."""

        tuple_fields = (
            self.neighbors,
            self.neighbor_epi_values,
            self.neighbor_dominant_values,
            self.neighbor_epi_kinds,
        )
        if any(type(value) is not tuple for value in tuple_fields):
            raise TypeError("Reception snapshot neighbor fields must be tuples")
        lengths = tuple(len(value) for value in tuple_fields)
        if len(set(lengths)) != 1:
            raise ValueError("Reception snapshot neighbor fields must align")
        scalar_values = (
            self.target_epi,
            self.neighbor_epi_mean,
            *self.neighbor_epi_values,
            *self.neighbor_dominant_values,
        )
        if any(
            type(value) is not float or not math.isfinite(value)
            for value in scalar_values
        ):
            raise ValueError("Reception snapshot EPI fields must be finite floats")
        if type(self.target_epi_kind) is not str or any(
            type(value) is not str for value in self.neighbor_epi_kinds
        ):
            raise TypeError("Reception snapshot EPI kinds must be strings")
        decision = self.compatibility_decision
        if decision is not None:
            if (
                type(decision) is not ReceptionCompatibilityDecision
                or not decision._proof_fields_are_intact()
                or not decision.selected_sources
                or not proof_stamps_are_identical(
                    structural_proof_signature(
                        (decision.node, decision.selected_sources)
                    ),
                    structural_proof_signature((self.node, self.source_nodes)),
                )
            ):
                raise ValueError(
                    "Reception compatibility decision does not match its read"
                )
        if self.source_nodes is not None:
            if type(self.source_nodes) is not tuple or not self.source_nodes:
                raise ValueError(
                    "Reception explicit snapshot sources must be nonempty tuples"
                )
            if len(set(self.source_nodes)) != len(self.source_nodes):
                raise ValueError("Reception explicit snapshot sources must be unique")
            if not proof_stamps_are_identical(
                structural_proof_signature(self.source_nodes),
                structural_proof_signature(self.neighbors),
            ):
                raise ValueError(
                    "Reception snapshot must retain the requested source order"
                )
            if self.neighbor_dominant_values != self.neighbor_epi_values:
                raise ValueError(
                    "Explicit Reception sources cannot use missing-EPI fallback"
                )
        expected_mean = (
            neighbor_epi_unweighted_mean(self.neighbor_epi_values)
            if self.neighbors
            else self.target_epi
        )
        if self.neighbor_epi_mean != expected_mean:
            raise ValueError("Reception snapshot neighbor mean is inconsistent")
        if type(self.source_tracking_enabled) is not bool:
            raise TypeError("Reception source-tracking flag must be a bool")
        if self.source_tracking_enabled:
            if (
                type(self.source_max_distance) is not int
                or self.source_max_distance < 0
            ):
                raise ValueError("Reception source distance must be nonnegative")
        elif self.source_max_distance is not None:
            raise ValueError("Disabled source tracking cannot retain a distance")
        if type(self._graph_identity) is not int or self._graph_identity <= 0:
            raise ValueError("Reception snapshot graph identity is invalid")
        if (
            type(self._metric_consumer_graph_identity) is not int
            or self._metric_consumer_graph_identity <= 0
        ):
            raise ValueError("Reception metrics consumer identity is invalid")
        if self._graph_identity != id(self._read_graph_owner):
            raise ValueError("Reception read graph owner identity changed")
        if self._metric_consumer_graph_identity != id(
            self._metric_consumer_graph_owner
        ):
            raise ValueError("Reception metrics graph owner identity changed")
        sources = self.reception_sources
        if self.source_tracking_enabled != (sources is not None):
            raise ValueError("Reception source evidence contradicts its flag")
        if sources is not None:
            if type(sources) is not tuple:
                raise TypeError("Reception sources must be a tuple or None")
            for source in sources:
                if type(source) is not tuple or len(source) != 3:
                    raise TypeError("Reception source records must be triples")
                compatibility = source[1]
                activity = source[2]
                if (
                    type(compatibility) is not float
                    or not math.isfinite(compatibility)
                    or not 0.0 <= compatibility <= 1.0
                    or type(activity) is not float
                    or not math.isfinite(activity)
                    or activity < 0.0
                ):
                    raise ValueError("Reception source scores are invalid")
        if self.read_boundary != RECEPTION_PRE_STATE_BOUNDARY:
            raise ValueError("Reception snapshot read boundary changed")

    @property
    def source_selection_origin(self) -> str:
        """Distinguish the actual read request from optional source telemetry."""

        if self.compatibility_decision is not None:
            return "compatibility_policy"
        return (
            "legacy_all_neighbors" if self.source_nodes is None else "explicit_request"
        )

    def _proof_fields_are_intact(self) -> bool:
        """Whether values and both graph-owner bindings retain their seal."""

        try:
            self.__post_init__()
            return proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                _reception_read_snapshot_stamp(self),
            )
        except BaseException:
            return False


def capture_reception_read_snapshot(
    graph: Any,
    node: Any,
    *,
    track_sources: bool,
    max_distance: Any = 2,
    source_nodes: tuple[Any, ...] | None = None,
    compatibility_decision: ReceptionCompatibilityDecision | None = None,
    _metric_consumer_graph_owner: Any = None,
) -> ReceptionReadSnapshot:
    """Capture actual EN inputs and optional source diagnostics before writing.

    An explicit source tuple changes the causal mean and kind selection while
    retaining the existing arithmetic. It does not assert a compatibility law.
    None preserves the legacy neighborhood and missing-form fallbacks.
    """

    if type(track_sources) is not bool:
        raise TypeError("Reception source-tracking flag must be a bool")
    if track_sources and (
        isinstance(max_distance, bool)
        or not isinstance(max_distance, Integral)
        or max_distance < 0
    ):
        raise ValueError("Reception max_distance must be nonnegative integer")

    if compatibility_decision is not None:
        if not reception_compatibility_matches_graph(graph, compatibility_decision):
            raise RuntimeError("Reception compatibility decision is stale or invalid")
        if not proof_stamps_are_identical(
            structural_proof_signature(
                (compatibility_decision.node, compatibility_decision.selected_sources)
            ),
            structural_proof_signature((node, source_nodes)),
        ):
            raise ValueError("Reception compatibility decision source request changed")
    raw_target = _raw_alias(graph, node, ALIAS_EPI, 0.0)
    target_epi = require_real_scalar_epi(
        raw_target,
        operator="Reception",
        label="target EPI state",
    )
    values: list[float] = []
    dominant_values: list[float] = []
    kinds: list[str] = []
    has_explicit_neighbor_epi = False
    if source_nodes is None:
        candidates = reception_input_neighbors(graph, node)
        for neighbor in candidates:
            raw_neighbor = _raw_alias(
                graph,
                neighbor,
                ALIAS_EPI,
                _MISSING_NEIGHBOR_EPI,
            )
            if raw_neighbor is _MISSING_NEIGHBOR_EPI:
                values.append(target_epi)
                dominant_values.append(0.0)
            else:
                scalar = require_real_scalar_epi(
                    raw_neighbor,
                    operator="Reception",
                    label=f"neighbor EPI for {neighbor!r}",
                )
                values.append(scalar)
                dominant_values.append(scalar)
                has_explicit_neighbor_epi = True
            kinds.append(_node_kind(graph, neighbor))
    else:
        candidates, selected_values = _read_explicit_reception_sources(
            graph, node, source_nodes
        )
        values.extend(selected_values)
        dominant_values.extend(selected_values)
        kinds.extend(_node_kind(graph, source) for source in candidates)
        has_explicit_neighbor_epi = True

    if candidates and has_explicit_neighbor_epi:
        neighbors = candidates
        neighbor_values = tuple(values)
        dominant = tuple(dominant_values)
        neighbor_kinds = tuple(kinds)
        neighbor_mean = neighbor_epi_unweighted_mean(neighbor_values)
    else:
        neighbors = ()
        neighbor_values = ()
        dominant = ()
        neighbor_kinds = ()
        neighbor_mean = target_epi

    sources = None
    source_max_distance = None
    if track_sources:
        from .network_analysis.source_detection import detect_emission_sources

        source_max_distance = int(max_distance)
        sources = tuple(
            detect_emission_sources(
                graph,
                node,
                max_distance=source_max_distance,
            )
        )

    consumer_owner = (
        graph if _metric_consumer_graph_owner is None else _metric_consumer_graph_owner
    )
    candidate = ReceptionReadSnapshot(
        node=node,
        target_epi=target_epi,
        target_epi_kind=_node_kind(graph, node),
        neighbors=neighbors,
        neighbor_epi_values=neighbor_values,
        neighbor_dominant_values=dominant,
        neighbor_epi_kinds=neighbor_kinds,
        neighbor_epi_mean=neighbor_mean,
        source_tracking_enabled=track_sources,
        source_max_distance=source_max_distance,
        reception_sources=sources,
        source_nodes=source_nodes,
        compatibility_decision=compatibility_decision,
        _read_graph_owner=graph,
        _metric_consumer_graph_owner=consumer_owner,
        _graph_identity=id(graph),
        _metric_consumer_graph_identity=id(consumer_owner),
    )
    if consumer_owner is not graph:
        consumer_snapshot = capture_reception_read_snapshot(
            consumer_owner,
            node,
            track_sources=track_sources,
            max_distance=max_distance,
            source_nodes=source_nodes,
            compatibility_decision=compatibility_decision,
        )
        if not proof_stamps_are_identical(
            _reception_read_payload_stamp(candidate),
            _reception_read_payload_stamp(consumer_snapshot),
        ):
            raise ValueError("Reception metrics consumer does not match the read graph")
    return replace(
        candidate,
        _proof_stamp=_reception_read_snapshot_stamp(candidate),
    )


def apply_reception_read_snapshot(
    node: Any,
    mix: float,
    snapshot: ReceptionReadSnapshot,
) -> tuple[float, str]:
    """Apply EN from a previously materialized read without rereading peers."""

    if type(snapshot) is not ReceptionReadSnapshot:
        raise TypeError("Reception requires a canonical pre-state snapshot")
    if not snapshot._proof_fields_are_intact():
        raise ValueError("Reception snapshot proof fields are not intact")
    if (
        not hasattr(node, "G")
        or node.G is not snapshot._read_graph_owner
        or id(node.G) != snapshot._graph_identity
    ):
        raise ValueError("Reception snapshot belongs to a different graph")
    if not hasattr(node, "n") or not proof_stamps_are_identical(
        structural_proof_signature(node.n),
        structural_proof_signature(snapshot.node),
    ):
        raise ValueError("Reception snapshot belongs to a different target")
    if not reception_read_snapshot_matches_graph(node.G, snapshot):
        raise RuntimeError("prepared Reception state is stale")
    default_kind = Glyph.EN.value
    if not snapshot.neighbors:
        node.epi_kind = default_kind
        return snapshot.target_epi, default_kind

    from . import _finite_operator_scalar, _validated_epi_assignment_value

    proposed = _finite_operator_scalar(
        neighbor_epi_blend_value(
            snapshot.target_epi,
            snapshot.neighbor_epi_mean,
            mix,
        ),
        "EN EPI proposal",
    )
    final_kind = reception_proposed_epi_kind(
        snapshot.target_epi_kind,
        zip(
            snapshot.neighbor_dominant_values,
            snapshot.neighbor_epi_kinds,
            strict=True,
        ),
        unclipped_target_epi=proposed,
        fallback_kind=default_kind,
    )
    bounded_epi = _validated_epi_assignment_value(node, proposed)
    epi_before = node.EPI
    kind_before = node.epi_kind
    node.EPI = bounded_epi
    try:
        node.epi_kind = final_kind
    except BaseException:
        try:
            node.EPI = epi_before
            node.epi_kind = kind_before
        except BaseException:
            pass
        raise
    return snapshot.neighbor_epi_mean, final_kind


def reception_read_snapshot_matches_graph(
    graph: Any,
    snapshot: ReceptionReadSnapshot,
) -> bool:
    """Whether a prepared EN record still equals the graph's current read set."""

    if (
        type(snapshot) is not ReceptionReadSnapshot
        or not snapshot._proof_fields_are_intact()
        or graph is not snapshot._read_graph_owner
        or id(graph) != snapshot._graph_identity
    ):
        return False
    try:
        snapshot.__post_init__()
        current = capture_reception_read_snapshot(
            graph,
            snapshot.node,
            track_sources=snapshot.source_tracking_enabled,
            max_distance=(
                snapshot.source_max_distance
                if snapshot.source_max_distance is not None
                else 0
            ),
            _metric_consumer_graph_owner=(snapshot._metric_consumer_graph_owner),
            source_nodes=snapshot.source_nodes,
            compatibility_decision=snapshot.compatibility_decision,
        )
    except Exception:
        return False
    fields = (
        "node",
        "target_epi",
        "target_epi_kind",
        "neighbors",
        "neighbor_epi_values",
        "neighbor_dominant_values",
        "neighbor_epi_kinds",
        "neighbor_epi_mean",
        "source_tracking_enabled",
        "source_max_distance",
        "reception_sources",
        "source_nodes",
        "compatibility_decision",
        "source_selection_origin",
        "read_boundary",
        "_graph_identity",
        "_metric_consumer_graph_identity",
    )
    return proof_stamps_are_identical(
        structural_proof_signature(
            tuple(object.__getattribute__(snapshot, name) for name in fields)
        ),
        structural_proof_signature(
            tuple(object.__getattribute__(current, name) for name in fields)
        ),
    )


__all__ = [
    "RECEPTION_NO_SOURCES_WARNING_PATTERN",
    "RECEPTION_PRE_STATE_BOUNDARY",
    "RECEPTION_PRESSURE_OBSERVATION_BOUNDARY",
    "ReceptionReadSnapshot",
    "apply_reception_read_snapshot",
    "capture_reception_read_snapshot",
    "reception_read_snapshot_matches_graph",
    "reception_no_sources_warning",
    "reception_input_neighbors",
    "validate_reception_source_nodes",
    "reception_blend_value",
    "reception_represented_affine_row",
    "reception_unweighted_mean",
]


reception_unweighted_mean = neighbor_epi_unweighted_mean
reception_blend_value = neighbor_epi_blend_value
reception_represented_affine_row = neighbor_epi_represented_affine_row
