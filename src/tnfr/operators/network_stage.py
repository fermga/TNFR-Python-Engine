"""Atomic two-phase network stages for neighbour-coupled operators.

Reception (EN) and Resonance (RA) read neighbouring nodal state.  Applying
their public node operators one target at a time therefore implements a
Gauss--Seidel sweep: later targets observe earlier writes.  This module owns
the bounded network-level repair for those two operators:

``complete snapshot -> preflight all -> propose all -> validate all -> commit``.

Every proposal is derived from the same detached stage-start graph and stored
in a frozen value object.  The live graph is changed only after every target
has passed grammar, U3, factor and scalar-domain validation.  A complete outer
snapshot keeps state, topology, histories, caches, monitor bookkeeping,
telemetry and the pressure refresh inside one rollback boundary.

Other canonical operators retain their established operator-major
Gauss--Seidel semantics until their cross-target write sets have explicit
merge rules.
"""

from __future__ import annotations

import math
import threading
import warnings
from collections import deque
from collections.abc import (
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
)
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    FunctionType,
    MappingProxyType,
    MethodType,
)
from typing import Any, Literal
from weakref import WeakValueDictionary

import networkx as nx

from .. import glyph_history
from ..alias import get_attr, set_attr_str
from ..constants.aliases import (
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SOURCE_GLYPH,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from ..rng import validate_graph_seed
from ..types import Glyph
from ..utils import angle_diff
from ._argument_validation import require_list_sink
from ._epi_domain import require_real_scalar_epi
from ._neighbor_epi_kernel import (
    neighbor_epi_blend_value,
    neighbor_epi_unweighted_mean,
)
from ._phase_gate import U3PhaseGateError, resolve_u3_phase_neighbors
from ._resonance_identity import (
    RA_RUNTIME_AMPLIFICATION_TRIGGER,
    normalize_resonance_epi_kind,
    resonance_identity_failures,
    resonance_neighbor_circular_mean,
    resonance_proposed_epi_kind,
    validate_resonance_runtime_factors,
)
from .factor_contracts import resolve_runtime_operator_factors


TWO_PHASE_JACOBI = "two_phase_jacobi"
OPERATOR_MAJOR_GAUSS_SEIDEL = "operator_major_gauss_seidel"
STAGE_SCHEDULE_KEY = "_last_network_stage_schedule"

_RUNTIME_GRAPH_KEYS = frozenset(
    {"integrity_monitor", "_node_cache", "_node_cache_weak", "_creating_node"}
)
_REFERENCE_PRESERVING_RUNTIME_MAPPINGS = frozenset(
    {"_node_cache", "_node_cache_weak"}
)
_NETWORKX_GRAPH_INTERNAL_ATTRIBUTES = frozenset(
    {
        "graph",
        "_node",
        "_adj",
        "_succ",
        "_pred",
        "adj",
        "nodes",
        "degree",
        "edges",
        "in_edges",
        "out_edges",
        "in_degree",
        "out_degree",
        "__networkx_cache__",
        "__networkx_backend__",
        "_last_operator_applied",
    }
)
_IMMUTABLE_CALLABLE_TYPES = (
    FunctionType,
    BuiltinFunctionType,
    MethodType,
    BuiltinMethodType,
    type,
)
_LOCK_TYPES = (type(threading.Lock()), type(threading.RLock()))


@dataclass(frozen=True, slots=True)
class _RuntimeGraphValue:
    """Reference-preserving snapshot for graph runtime objects."""

    key: Any
    value: Any
    container_kind: str | None
    container_state: Any
    object_state: Mapping[str, Any] | None
    slot_state: tuple[tuple[str, bool, Any], ...]


def _runtime_slot_names(value: Any) -> tuple[str, ...]:
    """Return concrete slot attribute names across an object's MRO."""

    names: list[str] = []
    seen: set[str] = set()
    for cls in type(value).__mro__:
        raw_slots = vars(cls).get("__slots__", ())
        slots = (raw_slots,) if isinstance(raw_slots, str) else tuple(raw_slots)
        for raw_name in slots:
            if raw_name in {"__dict__", "__weakref__"}:
                continue
            name = raw_name
            if raw_name.startswith("__") and not raw_name.endswith("__"):
                name = f"_{cls.__name__.lstrip('_')}{raw_name}"
            if name not in seen:
                names.append(name)
                seen.add(name)
    return tuple(names)


def _known_immutable_runtime_value(value: Any) -> bool:
    """Recognize values whose identity can safely span a rollback boundary."""

    if value is None or isinstance(
        value, (bool, int, float, complex, str, bytes, range, Enum)
    ):
        return True
    if np is not None and isinstance(value, np.generic):
        return True
    if isinstance(value, tuple):
        return all(_known_immutable_runtime_value(item) for item in value)
    if isinstance(value, frozenset):
        return all(_known_immutable_runtime_value(item) for item in value)
    return isinstance(value, _IMMUTABLE_CALLABLE_TYPES) or type(value) is object


def _seed_runtime_lock_memo(
    value: Any, memo: dict[int, Any], seen: set[int] | None = None
) -> None:
    """Keep synchronization primitives by identity during state deepcopy."""

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, _LOCK_TYPES):
        memo[identity] = value
        return
    if isinstance(value, MappingProxyType):
        memo[identity] = value
        return
    if identity in memo:
        return
    if _known_immutable_runtime_value(value):
        return
    if np is not None and isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            for item in value.flat:
                _seed_runtime_lock_memo(item, memo, seen)
        return
    if isinstance(value, Mapping):
        for key, item in tuple(value.items()):
            _seed_runtime_lock_memo(key, memo, seen)
            _seed_runtime_lock_memo(item, memo, seen)
    elif isinstance(value, (tuple, list, set, frozenset, deque)):
        for item in tuple(value):
            _seed_runtime_lock_memo(item, memo, seen)
    if hasattr(value, "__dict__"):
        _seed_runtime_lock_memo(vars(value), memo, seen)
    for name in _runtime_slot_names(value):
        if hasattr(value, name):
            _seed_runtime_lock_memo(getattr(value, name), memo, seen)


def _prepare_runtime_copy_memo(
    key: Any, value: Any, memo: dict[int, Any]
) -> bool:
    """Seed identities that are resources rather than rollback state."""

    is_cache_manager = False
    if key == "_tnfr_cache_manager":
        from ..utils.cache import CacheManager

        if isinstance(value, CacheManager):
            is_cache_manager = True
            for layer in getattr(value, "_layers", ()):
                memo[id(layer)] = layer
            for name in ("_storage_layer", "_graph_owner"):
                resource = getattr(value, name, None)
                if resource is not None:
                    memo[id(resource)] = resource
    _seed_runtime_lock_memo(vars(value) if hasattr(value, "__dict__") else value, memo)
    if is_cache_manager:
        storage = value._storage
        memo[id(storage)] = storage
    return is_cache_manager


def _capture_runtime_value(
    key: Any,
    value: Any,
    memo: dict[int, Any],
    *,
    preserve_mapping_items: bool = False,
) -> _RuntimeGraphValue:
    """Capture one identity-bearing runtime value or reject it before writes."""

    is_cache_manager = _prepare_runtime_copy_memo(key, value, memo)
    container_kind: str | None = None
    container_state: Any = None
    if is_cache_manager:
        container_kind = "cache_manager"
        container_state = deepcopy(tuple(value._storage.items()), memo)
    elif isinstance(value, MutableMapping):
        raw_items = tuple(value.items())
        if preserve_mapping_items:
            container_kind = "mapping_reference"
            container_state = raw_items
        else:
            container_kind = "mapping"
            container_state = deepcopy(raw_items, memo)
    elif isinstance(value, deque):
        container_kind = "deque"
        container_state = (deepcopy(tuple(value), memo), value.maxlen)
    elif np is not None and isinstance(value, np.ndarray):
        container_kind = "ndarray"
        container_state = np.array(value, copy=True, subok=True)
    elif isinstance(value, MutableSequence):
        container_kind = "sequence"
        container_state = deepcopy(tuple(value), memo)
    elif isinstance(value, MutableSet):
        container_kind = "set"
        container_state = deepcopy(tuple(value), memo)

    object_state = None
    if hasattr(value, "__dict__") and not isinstance(
        value, _IMMUTABLE_CALLABLE_TYPES
    ):
        try:
            object_state = deepcopy(vars(value), memo)
        except Exception as exc:
            raise TNFRValueError(
                f"runtime object {key!r} cannot be snapshotted atomically"
            ) from exc

    slot_state: list[tuple[str, bool, Any]] = []
    if not isinstance(value, _IMMUTABLE_CALLABLE_TYPES):
        try:
            for name in _runtime_slot_names(value):
                present = hasattr(value, name)
                slot_state.append(
                    (
                        name,
                        present,
                        deepcopy(getattr(value, name), memo) if present else None,
                    )
                )
        except Exception as exc:
            raise TNFRValueError(
                f"runtime object {key!r} cannot be snapshotted atomically"
            ) from exc

    if (
        container_kind is None
        and object_state is None
        and not slot_state
        and not _known_immutable_runtime_value(value)
    ):
        raise TNFRValueError(
            f"runtime object {key!r} has unsupported mutable state; "
            "atomic rollback cannot be guaranteed"
        )

    return _RuntimeGraphValue(
        key=key,
        value=value,
        container_kind=container_kind,
        container_state=container_state,
        object_state=object_state,
        slot_state=tuple(slot_state),
    )


def _restore_runtime_value(
    snapshot: _RuntimeGraphValue, memo: dict[int, Any]
) -> None:
    """Restore one previously captured runtime value without replacing it."""

    value = snapshot.value
    _prepare_runtime_copy_memo(snapshot.key, value, memo)
    state = snapshot.container_state
    if snapshot.container_kind == "mapping":
        value.clear()
        value.update(deepcopy(state, memo))
    elif snapshot.container_kind == "mapping_reference":
        value.clear()
        value.update(state)
    elif snapshot.container_kind == "deque":
        items, maxlen = state
        if value.maxlen != maxlen:
            raise TNFRValueError("runtime deque maxlen changed during transaction")
        value.clear()
        value.extend(deepcopy(items, memo))
    elif snapshot.container_kind == "ndarray":
        if value.shape != state.shape or value.dtype != state.dtype:
            raise TNFRValueError(
                "runtime ndarray shape or dtype changed during transaction"
            )
        np.copyto(value, state, casting="no")
    elif snapshot.container_kind == "sequence":
        value.clear()
        value.extend(deepcopy(state, memo))
    elif snapshot.container_kind == "set":
        value.clear()
        value.update(deepcopy(state, memo))
    elif snapshot.container_kind == "cache_manager":
        value._storage.clear()
        value._storage.update(deepcopy(state, memo))

    if snapshot.object_state is not None:
        vars(value).clear()
        vars(value).update(deepcopy(dict(snapshot.object_state), memo))
    for name, present, prior in snapshot.slot_state:
        if present:
            setattr(value, name, deepcopy(prior, memo))
        elif hasattr(value, name):
            delattr(value, name)


def _restore_mapping_order(
    mapping: MutableMapping[Any, Any],
    order: Sequence[Any],
    *,
    label: str,
) -> None:
    """Restore the exact insertion order without replacing the mapping object."""

    if len(mapping) != len(order) or any(key not in mapping for key in order):
        raise TNFRValueError(
            f"cannot restore {label}: mapping membership changed unexpectedly"
        )
    ordered_items = tuple((key, mapping[key]) for key in order)
    mapping.clear()
    mapping.update(ordered_items)


class GraphTransactionSnapshot:
    """Capture and restore the complete mutable surface of a NetworkX graph."""

    def __init__(self, graph: Any) -> None:
        self._nodes = tuple(graph.nodes)
        self._node_data = {
            node: deepcopy(dict(graph.nodes[node])) for node in self._nodes
        }
        self._directed = bool(graph.is_directed())
        self._multigraph = bool(graph.is_multigraph())
        self._adjacency_order = {
            node: tuple(graph.adj[node]) for node in self._nodes
        }
        self._predecessor_order = (
            {node: tuple(graph.pred[node]) for node in self._nodes}
            if self._directed
            else {}
        )
        self._adjacency_key_order = (
            {
                (node, neighbor): tuple(graph.adj[node][neighbor])
                for node in self._nodes
                for neighbor in self._adjacency_order[node]
            }
            if self._multigraph
            else None
        )
        self._predecessor_key_order = (
            {
                (node, neighbor): tuple(graph.pred[node][neighbor])
                for node in self._nodes
                for neighbor in self._predecessor_order[node]
            }
            if self._multigraph and self._directed
            else {}
        )
        if self._multigraph:
            self._edges = tuple(
                (left, right, key, deepcopy(dict(data)))
                for left, right, key, data in graph.edges(keys=True, data=True)
            )
        else:
            self._edges = tuple(
                (left, right, deepcopy(dict(data)))
                for left, right, data in graph.edges(data=True)
            )

        ordinary: list[tuple[Any, Any]] = []
        runtime: list[_RuntimeGraphValue] = []
        runtime_items = tuple(
            (key, value)
            for key, value in graph.graph.items()
            if key in _RUNTIME_GRAPH_KEYS or "cache" in str(key).lower()
        )
        graph_attribute_items = tuple(
            (key, value)
            for key, value in vars(graph).items()
            if key not in _NETWORKX_GRAPH_INTERNAL_ATTRIBUTES
        )
        runtime_memo = {id(graph): graph}
        runtime_memo.update({id(value): value for _key, value in runtime_items})
        runtime_memo.update(
            {id(value): value for _key, value in graph_attribute_items}
        )
        for key, value in graph.graph.items():
            is_runtime = key in _RUNTIME_GRAPH_KEYS or "cache" in str(key).lower()
            if not is_runtime:
                ordinary.append((key, deepcopy(value, runtime_memo)))
                continue
            runtime.append(
                _capture_runtime_value(
                    key,
                    value,
                    runtime_memo,
                    preserve_mapping_items=(
                        key in _REFERENCE_PRESERVING_RUNTIME_MAPPINGS
                        or isinstance(value, WeakValueDictionary)
                    ),
                )
            )
        self._graph_data_order = tuple(graph.graph)
        self._ordinary_graph_data = tuple(ordinary)
        self._runtime_graph_data = tuple(runtime)
        self._graph_attribute_names = frozenset(
            key for key, _value in graph_attribute_items
        )
        self._graph_attributes = tuple(
            _capture_runtime_value(key, value, runtime_memo)
            for key, value in graph_attribute_items
        )
        self._had_last_operator = hasattr(graph, "_last_operator_applied")
        self._last_operator = getattr(graph, "_last_operator_applied", None)

    def restore(self, graph: Any) -> None:
        """Restore topology, attributes, caches and monitor bookkeeping."""

        if self._multigraph:
            graph.remove_edges_from(tuple(graph.edges(keys=True)))
        else:
            graph.remove_edges_from(tuple(graph.edges))

        original_nodes = frozenset(self._nodes)
        graph.remove_nodes_from(
            node for node in tuple(graph.nodes) if node not in original_nodes
        )
        for node in self._nodes:
            if node not in graph:
                graph.add_node(node)
            data = graph.nodes[node]
            data.clear()
            data.update(deepcopy(self._node_data[node]))

        if self._multigraph:
            for left, right, key, data in self._edges:
                graph.add_edge(left, right, key=key, **deepcopy(data))
        else:
            for left, right, data in self._edges:
                graph.add_edge(left, right, **deepcopy(data))

        _restore_mapping_order(graph._node, self._nodes, label="node order")
        if self._directed:
            _restore_mapping_order(graph._succ, self._nodes, label="successor node order")
            _restore_mapping_order(graph._pred, self._nodes, label="predecessor node order")
            for node in self._nodes:
                _restore_mapping_order(
                    graph._succ[node],
                    self._adjacency_order[node],
                    label=f"successor order for node {node!r}",
                )
                _restore_mapping_order(
                    graph._pred[node],
                    self._predecessor_order[node],
                    label=f"predecessor order for node {node!r}",
                )
        else:
            _restore_mapping_order(graph._adj, self._nodes, label="adjacency node order")
            for node in self._nodes:
                _restore_mapping_order(
                    graph._adj[node],
                    self._adjacency_order[node],
                    label=f"neighbor order for node {node!r}",
                )

        if self._multigraph:
            for (node, neighbor), order in self._adjacency_key_order.items():
                adjacency = graph._succ if self._directed else graph._adj
                _restore_mapping_order(
                    adjacency[node][neighbor],
                    order,
                    label=f"edge-key order for ({node!r}, {neighbor!r})",
                )
            if self._directed:
                for (node, neighbor), order in self._predecessor_key_order.items():
                    _restore_mapping_order(
                        graph._pred[node][neighbor],
                        order,
                        label=(
                            "predecessor edge-key order for "
                            f"({node!r}, {neighbor!r})"
                        ),
                    )

        runtime_memo = {id(graph): graph}
        runtime_memo.update(
            {id(snapshot.value): snapshot.value for snapshot in self._runtime_graph_data}
        )
        runtime_memo.update(
            {id(snapshot.value): snapshot.value for snapshot in self._graph_attributes}
        )
        graph.graph.clear()
        graph.graph.update(deepcopy(dict(self._ordinary_graph_data), runtime_memo))
        for snapshot in self._runtime_graph_data:
            _restore_runtime_value(snapshot, runtime_memo)
            graph.graph[snapshot.key] = snapshot.value
        _restore_mapping_order(
            graph.graph, self._graph_data_order, label="graph-attribute order"
        )

        current_custom_attributes = tuple(
            key
            for key in vars(graph)
            if key not in _NETWORKX_GRAPH_INTERNAL_ATTRIBUTES
        )
        for key in current_custom_attributes:
            if key not in self._graph_attribute_names:
                delattr(graph, key)
        for snapshot in self._graph_attributes:
            _restore_runtime_value(snapshot, runtime_memo)
            setattr(graph, snapshot.key, snapshot.value)

        if self._had_last_operator:
            graph._last_operator_applied = self._last_operator
        elif hasattr(graph, "_last_operator_applied"):
            delattr(graph, "_last_operator_applied")


@dataclass(frozen=True, slots=True)
class NeighborStageProposal:
    """Immutable target proposal derived from one stage-start snapshot."""

    node: Any
    glyph: Glyph
    epi_before: float
    epi_before_payload: Any
    epi_after: float
    write_epi: bool
    epi_kind_before: str
    epi_kind_after: str
    vf_before: float
    vf_after: float
    write_vf: bool
    theta_before: float
    theta_after: float
    write_theta: bool
    neighbor_epi_mean: float
    neighbors: tuple[Any, ...]
    reception_sources: tuple[tuple[Any, float, float], ...] | None = None


@dataclass(frozen=True, slots=True)
class NetworkStageResult:
    """Completed network stage and its executable scheduling semantics."""

    operator: str
    glyph: str
    schedule: Literal["two_phase_jacobi", "operator_major_gauss_seidel"]
    nodes_processed: int


def _discard_pending_monitor(graph: Any) -> None:
    monitor = graph.graph.get("integrity_monitor")
    discard = getattr(monitor, "discard_pending_operator", None)
    if callable(discard):
        try:
            discard()
        except Exception:
            pass


def _detached_stage_graph(graph: Any) -> Any:
    """Return a complete detached logical graph for immutable stage reads."""

    if graph.is_directed():
        snapshot = nx.MultiDiGraph() if graph.is_multigraph() else nx.DiGraph()
    else:
        snapshot = nx.MultiGraph() if graph.is_multigraph() else nx.Graph()
    for key, value in graph.graph.items():
        if key in _RUNTIME_GRAPH_KEYS or "cache" in str(key).lower():
            continue
        snapshot.graph[key] = deepcopy(value)
    # The monitor is never invoked while proposals are built, but its public
    # shape remains part of common operator argument preflight.
    monitor = graph.graph.get("integrity_monitor")
    if monitor is not None:
        snapshot.graph["integrity_monitor"] = monitor

    snapshot.add_nodes_from(
        (node, deepcopy(dict(data))) for node, data in graph.nodes(data=True)
    )
    if graph.is_multigraph():
        snapshot.add_edges_from(
            (left, right, key, deepcopy(dict(data)))
            for left, right, key, data in graph.edges(keys=True, data=True)
        )
    else:
        snapshot.add_edges_from(
            (left, right, deepcopy(dict(data)))
            for left, right, data in graph.edges(data=True)
        )
    return snapshot


def _raw_alias(
    graph: Any, node: Any, aliases: tuple[str, ...], default: Any
) -> Any:
    return get_attr(
        graph.nodes[node],
        aliases,
        default,
        strict=True,
        conv=lambda value: value,
    )


def _raw_node_kind(graph: Any, node: Any) -> Any:
    return get_attr(
        graph.nodes[node],
        ALIAS_EPI_KIND,
        "",
        strict=True,
        conv=lambda value: value,
    )


def _node_kind(graph: Any, node: Any) -> str:
    return str(_raw_node_kind(graph, node))


def _finite_scalar(value: Any, *, operator: str, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TNFRValueError(
            f"{operator} {label} must be a finite real scalar",
            context={"operator": operator, "field": label, "value": repr(value)},
        )
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"{operator} {label} must be representable as a finite scalar",
            context={"operator": operator, "field": label, "value": repr(value)},
        ) from exc
    if not math.isfinite(resolved):
        raise TNFRValueError(
            f"{operator} {label} must remain finite",
            context={"operator": operator, "field": label, "value": repr(value)},
        )
    return resolved


def _bounded_epi(snapshot: Any, node: Any, proposed: float) -> float:
    """Use the canonical operator boundary without changing live state."""

    from ..node import NodeNX
    from . import _validated_epi_assignment_value

    adapter = NodeNX.from_graph(snapshot, node)
    return float(_validated_epi_assignment_value(adapter, proposed))


def _propose_reception(
    snapshot: Any,
    node: Any,
    factors: Mapping[str, Any],
    *,
    track_sources: bool,
    max_distance: Any,
) -> NeighborStageProposal:
    raw_epi = _raw_alias(snapshot, node, ALIAS_EPI, 0.0)
    epi_before = require_real_scalar_epi(
        raw_epi, operator="Reception", label="target EPI state"
    )
    candidate_neighbors = tuple(snapshot.neighbors(node))
    neighbor_values: list[float] = []
    dominant_values: list[float] = []
    has_explicit_neighbor_epi = False
    for neighbor in candidate_neighbors:
        raw_neighbor_epi = _raw_alias(snapshot, neighbor, ALIAS_EPI, None)
        if raw_neighbor_epi is None:
            # Match get_neighbor_epi: a missing neighbour EPI contributes the
            # target value to the mean once another explicit source exists,
            # while its NodeNX identity fallback has zero magnitude.
            neighbor_values.append(epi_before)
            dominant_values.append(0.0)
            continue
        scalar = require_real_scalar_epi(
            raw_neighbor_epi,
            operator="Reception",
            label=f"neighbor EPI for {neighbor!r}",
        )
        neighbor_values.append(scalar)
        dominant_values.append(scalar)
        has_explicit_neighbor_epi = True

    current_kind = _node_kind(snapshot, node)
    if candidate_neighbors and has_explicit_neighbor_epi:
        neighbors = candidate_neighbors
        epi_bar = neighbor_epi_unweighted_mean(neighbor_values)
        proposed = neighbor_epi_blend_value(
            epi_before, epi_bar, float(factors["EN_mix"])
        )
        epi_after = _bounded_epi(snapshot, node, proposed)

        dominant_kind: str | None = None
        best_abs = 0.0
        for neighbor, value in zip(neighbors, dominant_values, strict=True):
            magnitude = abs(value)
            if magnitude > best_abs:
                best_abs = magnitude
                dominant_kind = _node_kind(snapshot, neighbor)
        if not dominant_kind:
            dominant_kind = Glyph.EN.value
            best_abs = 0.0
        final_kind = dominant_kind if best_abs > abs(epi_after) else current_kind
        final_kind = str(final_kind or Glyph.EN.value)
        write_epi = True
    else:
        neighbors = ()
        epi_bar = epi_before
        epi_after = epi_before
        final_kind = Glyph.EN.value
        write_epi = False

    sources = None
    if track_sources:
        from .network_analysis.source_detection import detect_emission_sources

        sources = tuple(
            detect_emission_sources(snapshot, node, max_distance=max_distance)
        )

    return NeighborStageProposal(
        node=node,
        glyph=Glyph.EN,
        epi_before=epi_before,
        epi_before_payload=deepcopy(raw_epi),
        epi_after=epi_after,
        write_epi=write_epi,
        epi_kind_before=current_kind,
        epi_kind_after=final_kind,
        # These channels are deliberately absent from the EN read set.
        vf_before=0.0,
        vf_after=0.0,
        write_vf=False,
        theta_before=0.0,
        theta_after=0.0,
        write_theta=False,
        neighbor_epi_mean=epi_bar,
        neighbors=neighbors,
        reception_sources=sources,
    )


def _propose_resonance(
    snapshot: Any,
    node: Any,
    factors: Mapping[str, Any],
) -> NeighborStageProposal:
    diff = float(factors["RA_epi_diff"])
    vf_boost = float(factors["RA_vf_amplification"])
    phase_coupling = float(factors["RA_phase_coupling"])
    invalid_factors = validate_resonance_runtime_factors(
        diff, vf_boost, phase_coupling
    )
    if invalid_factors:
        raise TNFRValueError(
            "Resonance factor gate rejected the proposed propagation: "
            + "; ".join(invalid_factors),
            context={
                "operator": "Resonance",
                "failed_conditions": invalid_factors,
            },
        )

    def phase(candidate: Any) -> Any:
        return _raw_alias(snapshot, candidate, ALIAS_THETA, None)

    try:
        selection = resolve_u3_phase_neighbors(
            snapshot.graph,
            phase(node),
            snapshot.neighbors(node),
            phase_getter=phase,
            operator_code="RA",
        )
    except U3PhaseGateError as exc:
        raise TNFRValueError(
            f"Resonance phase gate rejected the operation: {exc}",
            context={
                "operator": "Resonance",
                "failed_condition": exc.failed_condition,
            },
        ) from exc

    raw_epi = _raw_alias(snapshot, node, ALIAS_EPI, 0.0)
    epi_before = require_real_scalar_epi(
        raw_epi, operator="Resonance", label="target EPI state"
    )
    kind_before = normalize_resonance_epi_kind(_raw_node_kind(snapshot, node))
    neighbors = selection.neighbors
    neighbor_value_kinds: list[tuple[float, str]] = []
    for neighbor in neighbors:
        neighbor_value_kinds.append(
            (
                require_real_scalar_epi(
                    _raw_alias(snapshot, neighbor, ALIAS_EPI, 0.0),
                    operator="Resonance",
                    label=f"neighbor EPI for {neighbor!r}",
                ),
                normalize_resonance_epi_kind(
                    _raw_node_kind(snapshot, neighbor)
                ),
            )
        )

    epi_bar = neighbor_epi_unweighted_mean(
        value for value, _kind in neighbor_value_kinds
    )
    proposed_epi = _bounded_epi(
        snapshot,
        node,
        neighbor_epi_blend_value(epi_before, epi_bar, diff),
    )

    theta_before = selection.target_phase
    theta_after = theta_before
    neighbor_phase_mean, phase_mean_defined = resonance_neighbor_circular_mean(
        selection.phases
    )
    if phase_mean_defined and neighbor_phase_mean is not None:
        theta_after = (
            theta_before
            + phase_coupling * angle_diff(neighbor_phase_mean, theta_before)
        ) % (2.0 * math.pi)
    theta_after = _finite_scalar(
        theta_after, operator="Resonance", label="phase proposal"
    )

    proposed_kind = resonance_proposed_epi_kind(
        kind_before,
        neighbor_value_kinds,
        proposed_epi,
        fallback_kind=Glyph.RA.value,
    )
    identity_failures = resonance_identity_failures(
        epi_before, proposed_epi, kind_before, proposed_kind
    )
    if identity_failures:
        raise TNFRValueError(
            "Resonance identity gate rejected the proposed propagation: "
            + ", ".join(identity_failures),
            context={
                "operator": "Resonance",
                "epi_before": epi_before,
                "epi_proposed": proposed_epi,
                "epi_kind_before": kind_before,
                "epi_kind_proposed": proposed_kind,
                "failed_conditions": identity_failures,
            },
        )

    vf_before = _finite_scalar(
        _raw_alias(snapshot, node, ALIAS_VF, 0.0),
        operator="Resonance",
        label="nu_f state",
    )
    amplification_active = abs(epi_bar) > RA_RUNTIME_AMPLIFICATION_TRIGGER
    vf_after = (
        vf_before * (1.0 + vf_boost) if amplification_active else vf_before
    )
    vf_after = _finite_scalar(
        vf_after, operator="Resonance", label="capacity proposal"
    )
    if vf_after < vf_before:
        raise TNFRValueError(
            "Resonance capacity proposal must be nondecreasing",
            context={
                "operator": "Resonance",
                "vf_before": vf_before,
                "vf_proposed": vf_after,
            },
        )

    return NeighborStageProposal(
        node=node,
        glyph=Glyph.RA,
        epi_before=epi_before,
        epi_before_payload=deepcopy(raw_epi),
        epi_after=proposed_epi,
        write_epi=True,
        epi_kind_before=kind_before,
        epi_kind_after=proposed_kind,
        vf_before=vf_before,
        vf_after=vf_after,
        write_vf=amplification_active,
        theta_before=theta_before,
        theta_after=theta_after,
        write_theta=True,
        neighbor_epi_mean=epi_bar,
        neighbors=neighbors,
    )


def _validate_proposals(
    proposals: Sequence[NeighborStageProposal], targets: tuple[Any, ...], glyph: Glyph
) -> None:
    """Validate the complete immutable proposal set before its first write."""

    if len(proposals) != len(targets):
        raise RuntimeError("Network stage proposal cardinality changed")
    if tuple(proposal.node for proposal in proposals) != targets:
        raise RuntimeError("Network stage proposal target order changed")
    if any(proposal.glyph is not glyph for proposal in proposals):
        raise RuntimeError("Network stage proposal glyph changed")
    for proposal in proposals:
        values = [("EPI", proposal.epi_after)]
        if glyph is Glyph.RA:
            values.extend(
                (("nu_f", proposal.vf_after), ("phase", proposal.theta_after))
            )
        for label, value in values:
            if not math.isfinite(float(value)):
                raise TNFRValueError(
                    f"{glyph.value} {label} proposal must be finite",
                    context={"node": proposal.node, "field": label},
                )
        if glyph is Glyph.RA:
            failures = resonance_identity_failures(
                proposal.epi_before,
                proposal.epi_after,
                proposal.epi_kind_before,
                proposal.epi_kind_after,
            )
            # The complete identity was already checked while the proposal
            # was built; any failure here indicates internal corruption.
            if failures:
                raise RuntimeError("Resonance proposal identity changed")


def _commit_structural_proposals(
    graph: Any, proposals: Sequence[NeighborStageProposal]
) -> None:
    """Commit already validated target-local channels through NodeNX setters."""

    from ..node import NodeNX
    from . import _set_epi_with_boundary_check

    for proposal in proposals:
        node = NodeNX.from_graph(graph, proposal.node)
        if proposal.write_epi:
            _set_epi_with_boundary_check(
                node, proposal.epi_after, apply_clip=False
            )
        node.epi_kind = proposal.epi_kind_after
        if proposal.write_vf:
            node.vf = proposal.vf_after
        if proposal.write_theta:
            node.theta = proposal.theta_after
        if proposal.reception_sources is not None:
            graph.nodes[proposal.node]["_reception_sources"] = list(
                proposal.reception_sources
            )


def _append_ra_telemetry(
    graph: Any,
    snapshot: Any,
    proposals: Sequence[NeighborStageProposal],
) -> None:
    collect_metrics = bool(graph.graph.get("COLLECT_RA_METRICS", False))
    track_coherence = bool(graph.graph.get("TRACK_NETWORK_COHERENCE", False))

    c_before = None
    if track_coherence:
        try:
            from ..metrics.common import compute_coherence

            c_before = compute_coherence(snapshot)
        except ImportError:
            pass

    if collect_metrics:
        sink = graph.graph.setdefault("ra_metrics", [])
        for proposal in proposals:
            vf_after = float(get_attr(graph.nodes[proposal.node], ALIAS_VF, 0.0))
            theta_after = float(
                get_attr(graph.nodes[proposal.node], ALIAS_THETA, 0.0)
            )
            sink.append(
                {
                    "operator": "RA",
                    "epi_propagated": proposal.neighbor_epi_mean,
                    "vf_amplification": (
                        vf_after / proposal.vf_before
                        if proposal.vf_before > 0
                        else 1.0
                    ),
                    "neighbors_influenced": len(proposal.neighbors),
                    "identity_preserved": True,
                    "epi_before": proposal.epi_before_payload,
                    "epi_after": float(
                        get_attr(graph.nodes[proposal.node], ALIAS_EPI, 0.0)
                    ),
                    "vf_before": proposal.vf_before,
                    "vf_after": vf_after,
                    "phase_before": proposal.theta_before,
                    "phase_after": theta_after,
                    "phase_alignment_strengthened": bool(
                        proposal.theta_after != proposal.theta_before
                    ),
                }
            )

    if track_coherence and c_before is not None:
        try:
            from ..metrics.common import compute_coherence

            c_after = compute_coherence(graph)
            sink = graph.graph.setdefault("_ra_c_tracking", [])
            for proposal in proposals:
                sink.append(
                    {
                        "node": proposal.node,
                        "c_before": c_before,
                        "c_after": c_after,
                        "c_delta": c_after - c_before,
                    }
                )
        except ImportError:
            pass


def _record_histories_and_patterns(
    graph: Any,
    proposals: Sequence[NeighborStageProposal],
    *,
    window: int,
) -> None:
    from .grammar_application import _recognize_applied_patterns

    for proposal in proposals:
        storage = graph.nodes[proposal.node]
        glyph_history.push_glyph(storage, proposal.glyph.value, window)
        set_attr_str(storage, ALIAS_SOURCE_GLYPH, proposal.glyph.value)
    for proposal in proposals:
        _recognize_applied_patterns(graph, proposal.node)


def _run_postcommit_checks(
    graph: Any,
    snapshot: Any,
    operator: Any,
    proposals: Sequence[NeighborStageProposal],
    states_before: Mapping[Any, dict[str, Any]],
    execution_kwargs: Mapping[str, Any],
) -> None:
    monitor = graph.graph.get("integrity_monitor")
    validate_equation = bool(
        execution_kwargs.get("validate_nodal_equation", False)
    ) or bool(graph.graph.get("VALIDATE_NODAL_EQUATION", False))
    collect_metrics = bool(execution_kwargs.get("collect_metrics", False)) or bool(
        graph.graph.get("COLLECT_OPERATOR_METRICS", False)
    )

    for proposal in proposals:
        node = proposal.node
        if monitor is not None:
            monitor.before_operator(snapshot, node)
            monitor.after_operator(graph, node, operator.name)

        if validate_equation:
            from .nodal_equation import validate_nodal_equation

            validate_nodal_equation(
                graph,
                node,
                epi_before=states_before[node]["epi"],
                epi_after=float(get_attr(graph.nodes[node], ALIAS_EPI, 0.0)),
                dt=float(execution_kwargs.get("dt", 1.0)),
                operator_name=operator.name,
                strict=graph.graph.get("NODAL_EQUATION_STRICT", False),
            )

        if collect_metrics:
            graph.graph.setdefault("operator_metrics", []).append(
                operator._collect_metrics(graph, node, states_before[node])
            )


def _record_schedule(
    graph: Any,
    *,
    operator: Any,
    schedule: Literal["two_phase_jacobi", "operator_major_gauss_seidel"],
    count: int,
) -> None:
    graph.graph[STAGE_SCHEDULE_KEY] = {
        "operator": operator.name,
        "glyph": operator.glyph.value,
        "schedule": schedule,
        "nodes_processed": count,
    }


def record_gauss_seidel_stage(graph: Any, operator: Any, count: int) -> None:
    """Label an established non-EN/RA stage without changing its semantics."""

    _record_schedule(
        graph,
        operator=operator,
        schedule=OPERATOR_MAJOR_GAUSS_SEIDEL,
        count=count,
    )


def execute_neighbor_stage(
    graph: Any,
    operator: Any,
    targets: Sequence[Any],
    *,
    sequence_context: Any = None,
    compute_delta_nfr: Any = None,
    transaction_snapshot: GraphTransactionSnapshot | None = None,
    **execution_kwargs: Any,
) -> NetworkStageResult:
    """Execute one atomic EN/RA stage from a shared immutable snapshot."""

    if operator.glyph not in (Glyph.EN, Glyph.RA):
        raise ValueError("Two-phase network stages currently support EN and RA only")

    from . import _validated_execution_window
    from .grammar_application import enforce_canonical_grammar
    from .grammar_debt import require_replayable_history
    from .grammar_types import glyph_function_name

    targets_tuple = tuple(targets)
    seen_targets: set[Any] = set()
    for node in targets_tuple:
        if node in seen_targets:
            raise TNFRValueError(
                "network stage targets must be unique",
                context={"duplicate_target": repr(node)},
            )
        seen_targets.add(node)
    transaction = transaction_snapshot or GraphTransactionSnapshot(graph)
    try:
        if not targets_tuple:
            if callable(compute_delta_nfr):
                compute_delta_nfr(graph)
            _record_schedule(
                graph,
                operator=operator,
                schedule=TWO_PHASE_JACOBI,
                count=0,
            )
            return NetworkStageResult(
                operator=operator.name,
                glyph=operator.glyph.value,
                schedule=TWO_PHASE_JACOBI,
                nodes_processed=0,
            )
        window = _validated_execution_window(
            graph, execution_kwargs.get("window")
        )
        execution_kwargs = dict(execution_kwargs)
        execution_kwargs["window"] = window
        validate_graph_seed(graph)
        snapshot = _detached_stage_graph(graph)

        for node in targets_tuple:
            if node not in snapshot:
                raise KeyError(node)
            require_replayable_history(snapshot.nodes[node].get("glyph_history"))

        selections: list[Any] = []
        states_before: dict[Any, dict[str, Any]] = {}
        needs_state_before = bool(
            execution_kwargs.get("validate_nodal_equation", False)
        ) or bool(snapshot.graph.get("VALIDATE_NODAL_EQUATION", False)) or bool(
            execution_kwargs.get("collect_metrics", False)
        ) or bool(snapshot.graph.get("COLLECT_OPERATOR_METRICS", False))
        for node in targets_tuple:
            operator._validate_hard_invariants(snapshot, node)
            operator._validate_application_preconditions(
                snapshot, node, sequence_context=sequence_context, **execution_kwargs
            )
            selected = enforce_canonical_grammar(
                snapshot, node, operator.glyph, sequence_context
            )
            selections.append(selected)
            if needs_state_before:
                states_before[node] = operator._capture_state(snapshot, node)

        exact_stage = all(
            glyph_function_name(selected) == operator.name
            for selected in selections
        )
        if not exact_stage:
            # Standalone grammar selection may legally replace the request.
            # Preserve that compatibility under a complete transaction and
            # label the resulting heterogeneous public-operator sweep honestly.
            for node in targets_tuple:
                graph._last_operator_applied = operator.name
                operator(
                    graph,
                    node,
                    sequence_context=sequence_context,
                    **execution_kwargs,
                )
            if callable(compute_delta_nfr):
                compute_delta_nfr(graph)
            record_gauss_seidel_stage(graph, operator, len(targets_tuple))
            return NetworkStageResult(
                operator=operator.name,
                glyph=operator.glyph.value,
                schedule=OPERATOR_MAJOR_GAUSS_SEIDEL,
                nodes_processed=len(targets_tuple),
            )

        factors = resolve_runtime_operator_factors(
            snapshot.graph.get("GLYPH_FACTORS"), operator.glyph, snapshot.graph
        )
        require_list_sink(
            snapshot.graph,
            "recognized_coherence_patterns",
            operator=operator.name,
        )
        if operator.glyph is Glyph.RA:
            if bool(snapshot.graph.get("COLLECT_RA_METRICS", False)):
                require_list_sink(snapshot.graph, "ra_metrics", operator=operator.name)
            if bool(snapshot.graph.get("TRACK_NETWORK_COHERENCE", False)):
                require_list_sink(
                    snapshot.graph,
                    "_ra_c_tracking",
                    operator=operator.name,
                )

        if operator.glyph is Glyph.EN:
            track_sources = bool(execution_kwargs.get("track_sources", True))
            max_distance = execution_kwargs.get("max_distance", 2)
            proposals = tuple(
                _propose_reception(
                    snapshot,
                    node,
                    factors,
                    track_sources=track_sources,
                    max_distance=max_distance,
                )
                for node in targets_tuple
            )
        else:
            proposals = tuple(
                _propose_resonance(snapshot, node, factors)
                for node in targets_tuple
            )

        _validate_proposals(proposals, targets_tuple, operator.glyph)
        graph._last_operator_applied = operator.name
        _commit_structural_proposals(graph, proposals)
        if operator.glyph is Glyph.RA:
            _append_ra_telemetry(graph, snapshot, proposals)
        _record_histories_and_patterns(graph, proposals, window=window)
        _run_postcommit_checks(
            graph,
            snapshot,
            operator,
            proposals,
            states_before,
            execution_kwargs,
        )
        if callable(compute_delta_nfr):
            compute_delta_nfr(graph)
        _record_schedule(
            graph,
            operator=operator,
            schedule=TWO_PHASE_JACOBI,
            count=len(targets_tuple),
        )

        # Reception's user-facing absence warning belongs to the accepted
        # transaction.  Warning-as-error policies therefore still roll back.
        if operator.glyph is Glyph.EN and bool(
            execution_kwargs.get("track_sources", True)
        ):
            for proposal in proposals:
                if not proposal.reception_sources:
                    warnings.warn(
                        f"EN: node {proposal.node} has no sources; "
                        "external coherence not integrated.",
                        stacklevel=3,
                    )

        return NetworkStageResult(
            operator=operator.name,
            glyph=operator.glyph.value,
            schedule=TWO_PHASE_JACOBI,
            nodes_processed=len(targets_tuple),
        )
    except BaseException:
        _discard_pending_monitor(graph)
        transaction.restore(graph)
        raise


__all__ = [
    "GraphTransactionSnapshot",
    "NetworkStageResult",
    "OPERATOR_MAJOR_GAUSS_SEIDEL",
    "STAGE_SCHEDULE_KEY",
    "TWO_PHASE_JACOBI",
    "execute_neighbor_stage",
    "record_gauss_seidel_stage",
]
