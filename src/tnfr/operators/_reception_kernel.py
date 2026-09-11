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

RECEPTION_PRE_STATE_BOUNDARY = "pre_en_snapshot"
RECEPTION_PRESSURE_OBSERVATION_BOUNDARY = "operator_metrics_collection"
RECEPTION_NO_SOURCES_WARNING_PATTERN = (
    r".*no emission sources detected under the configured telemetry policy.*"
)
_RECEPTION_READ_SNAPSHOT_VERSION = "reception_read_snapshot_v1"
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
    _metric_consumer_graph_owner: Any = None,
) -> ReceptionReadSnapshot:
    """Capture all EN neighbour, kind and source inputs before any EN write."""

    if type(track_sources) is not bool:
        raise TypeError("Reception source-tracking flag must be a bool")
    if track_sources and (
        isinstance(max_distance, bool)
        or not isinstance(max_distance, Integral)
        or max_distance < 0
    ):
        raise ValueError("Reception max_distance must be nonnegative integer")

    raw_target = _raw_alias(graph, node, ALIAS_EPI, 0.0)
    target_epi = require_real_scalar_epi(
        raw_target,
        operator="Reception",
        label="target EPI state",
    )
    candidates = reception_input_neighbors(graph, node)
    values: list[float] = []
    dominant_values: list[float] = []
    kinds: list[str] = []
    has_explicit_neighbor_epi = False
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
        graph
        if _metric_consumer_graph_owner is None
        else _metric_consumer_graph_owner
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
        )
        if not proof_stamps_are_identical(
            _reception_read_payload_stamp(candidate),
            _reception_read_payload_stamp(consumer_snapshot),
        ):
            raise ValueError(
                "Reception metrics consumer does not match the read graph"
            )
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
            _metric_consumer_graph_owner=(
                snapshot._metric_consumer_graph_owner
            ),
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
    "reception_blend_value",
    "reception_represented_affine_row",
    "reception_unweighted_mean",
]


reception_unweighted_mean = neighbor_epi_unweighted_mean
reception_blend_value = neighbor_epi_blend_value
reception_represented_affine_row = neighbor_epi_represented_affine_row
