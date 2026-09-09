"""Atomic network stages for canonical operators.

Reception (EN), Coherence (IL), Coupling (UM), Dissonance (OZ) and
Resonance (RA) read neighbouring nodal state. Applying their public node
operators one target at a time therefore implements a Gauss--Seidel sweep:
later targets observe earlier writes. This module owns the bounded network-level
snapshot repair for all promoted stages:

``complete snapshot -> preflight all -> propose all -> validate all -> commit``.

All thirteen canonical glyph stages use this transaction shape with
target-bound immutable proposals and explicit structural and lifecycle merges.
Every proposal is derived from the same detached stage-start graph and stored
in a frozen value object. The live graph is changed only after every target has
passed grammar, factor and scalar-domain validation. THOL additionally
materializes and validates its complete child-support and hierarchy merge on a
second detached graph. Recursivity merges its node-level advisory once per
telemetry step and leaves structural channels unchanged; explicit delayed EPI
mixing remains the separate apply_network_remesh operation. A complete outer
snapshot keeps state, topology, histories, caches, monitor bookkeeping,
telemetry and the pressure refresh inside one rollback boundary.
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from fractions import Fraction
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
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SOURCE_GLYPH,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
from ..rng import resolve_graph_seed, validate_graph_seed
from ..types import Glyph
from ..utils import angle_diff
from ._argument_validation import require_list_sink
from ._epi_domain import require_real_scalar_epi
from ._neighbor_epi_kernel import (
    neighbor_epi_blend_value,
    neighbor_epi_unweighted_mean,
    reception_proposed_epi_kind,
)
from ._phase_gate import U3PhaseGateError, resolve_u3_phase_neighbors
from ._recursivity_stage_kernel import RecursivityAdvisoryProposal
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
POINTWISE_TWO_PHASE_GLYPHS = frozenset(
    {Glyph.AL, Glyph.IL, Glyph.SHA, Glyph.VAL, Glyph.NUL, Glyph.ZHIR, Glyph.NAV}
)
POINTWISE_EPI_JUMP_GLYPHS = frozenset(
    {Glyph.AL, Glyph.SHA, Glyph.VAL, Glyph.NUL, Glyph.ZHIR, Glyph.NAV}
)
STAGE_SCHEDULE_KEY = "_last_network_stage_schedule"
STAGE_CONTRACT_KEY = "_last_network_stage_contract"

_RUNTIME_GRAPH_KEYS = frozenset(
    {
        "integrator",
        "integrity_monitor",
        "_node_cache",
        "_node_cache_weak",
        "_creating_node",
        "_epi_hist",
    }
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


@dataclass(frozen=True, slots=True)
class _RuntimeNDArrayState:
    """Logical ndarray state needed for identity-preserving restoration."""

    values: Any
    writeable: bool


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


def _seed_runtime_tuple_member_memo(
    value: Any,
    memo: dict[int, Any],
    seen: set[int] | None = None,
) -> None:
    """Preserve mutable tuple-member aliases captured by their own snapshots."""

    if not isinstance(value, tuple):
        return
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    for item in value:
        if _known_immutable_runtime_value(item):
            continue
        memo.setdefault(id(item), item)
        _seed_runtime_tuple_member_memo(item, memo, seen)


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
        container_state = _RuntimeNDArrayState(
            values=np.array(value, copy=True, subok=True),
            writeable=bool(value.flags.writeable),
        )
    elif isinstance(value, MutableSequence):
        container_kind = "sequence"
        container_state = deepcopy(tuple(value), memo)
    elif isinstance(value, MutableSet):
        container_kind = "set"
        container_state = deepcopy(tuple(value), memo)
    elif isinstance(value, tuple):
        member_snapshots: list[tuple[int, _RuntimeGraphValue]] = []
        for index, item in enumerate(value):
            if _known_immutable_runtime_value(item):
                continue
            member_snapshots.append(
                (
                    index,
                    _capture_runtime_value(
                        f"{key!s}[{index}]",
                        item,
                        memo,
                    ),
                )
            )
        container_kind = "tuple_members"
        container_state = tuple(member_snapshots)

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


def _runtime_ndarray_replacement(state: _RuntimeNDArrayState) -> Any:
    """Build a detached ndarray fallback with the captured logical state."""

    replacement = np.array(
        state.values,
        copy=True,
        order="K",
        subok=True,
    )
    replacement.flags.writeable = state.writeable
    return replacement


def _restore_runtime_ndarray(
    value: Any,
    state: _RuntimeNDArrayState,
) -> Any:
    """Restore ndarray shape, dtype, values and mutability when feasible."""

    expected = state.values
    try:
        if not value.flags.writeable:
            value.flags.writeable = True

        if value.dtype != expected.dtype:
            try:
                value.dtype = expected.dtype
            except Exception:
                if value.dtype.hasobject or expected.dtype.hasobject:
                    return _runtime_ndarray_replacement(state)
                value.dtype = np.uint8
                value.resize((expected.nbytes,), refcheck=False)
                value.dtype = expected.dtype

        if value.shape != expected.shape:
            try:
                value.shape = expected.shape
            except Exception:
                value.resize(expected.shape, refcheck=False)

        if value.shape != expected.shape or value.dtype != expected.dtype:
            return _runtime_ndarray_replacement(state)
        np.copyto(value, expected, casting="no")
        value.flags.writeable = state.writeable
    except Exception:
        return _runtime_ndarray_replacement(state)
    return value


def _rebuild_runtime_tuple(value: tuple[Any, ...], members: list[Any]) -> Any:
    """Reconstruct an immutable tuple container after a member fallback."""

    if type(value) is tuple:
        return tuple(members)
    try:
        if hasattr(value, "_fields"):
            return type(value)(*members)
        return type(value)(members)
    except Exception as exc:
        raise TNFRValueError(
            f"runtime tuple {type(value).__qualname__} cannot be reconstructed"
        ) from exc


def _restore_runtime_value(
    snapshot: _RuntimeGraphValue, memo: dict[int, Any]
) -> Any:
    """Restore a runtime value, reconstructing it only when required."""

    value = memo.get(id(snapshot.value), snapshot.value)
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
        value = _restore_runtime_ndarray(value, state)
    elif snapshot.container_kind == "sequence":
        value.clear()
        value.extend(deepcopy(state, memo))
    elif snapshot.container_kind == "set":
        value.clear()
        value.update(deepcopy(state, memo))
    elif snapshot.container_kind == "cache_manager":
        value._storage.clear()
        value._storage.update(deepcopy(state, memo))
    elif snapshot.container_kind == "tuple_members":
        members = list(value)
        member_replaced = False
        for index, member_snapshot in state:
            restored_member = _restore_runtime_value(member_snapshot, memo)
            if restored_member is not member_snapshot.value:
                members[index] = restored_member
                member_replaced = True
        if member_replaced:
            value = _rebuild_runtime_tuple(value, members)

    if snapshot.object_state is not None:
        vars(value).clear()
        vars(value).update(deepcopy(dict(snapshot.object_state), memo))
    for name, present, prior in snapshot.slot_state:
        if present:
            setattr(value, name, deepcopy(prior, memo))
        elif hasattr(value, name):
            delattr(value, name)

    memo[id(snapshot.value)] = value
    return value

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
    """Capture and restore graph-owned state and runtime references.

    Mutable runtime containers retain identity when they permit in-place
    restoration. If an ndarray refuses the required shape or dtype repair,
    graph-visible aliases are rebound to a detached captured value; references
    held only by external code remain outside the graph transaction.
    """

    def __init__(self, graph: Any) -> None:
        self._nodes = tuple(graph.nodes)
        self._directed = bool(graph.is_directed())
        self._multigraph = bool(graph.is_multigraph())
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
        for _key, value in runtime_items:
            _seed_runtime_tuple_member_memo(value, runtime_memo)
        for _key, value in graph_attribute_items:
            _seed_runtime_tuple_member_memo(value, runtime_memo)
        for value in graph.graph.values():
            _seed_runtime_lock_memo(value, runtime_memo)
        for node in self._nodes:
            _seed_runtime_lock_memo(graph.nodes[node], runtime_memo)
        for edge in (
            graph.edges(keys=True, data=True)
            if self._multigraph
            else graph.edges(data=True)
        ):
            _seed_runtime_lock_memo(edge[-1], runtime_memo)

        self._node_data = {
            node: deepcopy(dict(graph.nodes[node]), runtime_memo)
            for node in self._nodes
        }
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
                (left, right, key, deepcopy(dict(data), runtime_memo))
                for left, right, key, data in graph.edges(keys=True, data=True)
            )
        else:
            self._edges = tuple(
                (left, right, deepcopy(dict(data), runtime_memo))
                for left, right, data in graph.edges(data=True)
            )

        ordinary: list[tuple[Any, Any]] = []
        runtime: list[_RuntimeGraphValue] = []
        for key, value in graph.graph.items():
            is_runtime = key in _RUNTIME_GRAPH_KEYS or "cache" in str(key).lower()
            if not is_runtime:
                # Synchronization primitives are external resources, not
                # serializable graph state. Preserve their identity while
                # copying the surrounding ordinary metadata so unrelated
                # user locks do not make canonical stages unexecutable.
                _seed_runtime_lock_memo(value, runtime_memo)
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

        runtime_memo = {id(graph): graph}
        runtime_memo.update(
            {
                id(snapshot.value): snapshot.value
                for snapshot in self._runtime_graph_data
            }
        )
        runtime_memo.update(
            {
                id(snapshot.value): snapshot.value
                for snapshot in self._graph_attributes
            }
        )
        for value in self._node_data.values():
            _seed_runtime_lock_memo(value, runtime_memo)
        for edge in self._edges:
            _seed_runtime_lock_memo(edge[-1], runtime_memo)
        for _key, value in self._ordinary_graph_data:
            _seed_runtime_lock_memo(value, runtime_memo)

        restored_runtime_graph_data = tuple(
            (snapshot, _restore_runtime_value(snapshot, runtime_memo))
            for snapshot in self._runtime_graph_data
        )
        restored_graph_attributes = tuple(
            (snapshot, _restore_runtime_value(snapshot, runtime_memo))
            for snapshot in self._graph_attributes
        )

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
            data.update(deepcopy(self._node_data[node], runtime_memo))

        if self._multigraph:
            for left, right, key, data in self._edges:
                graph.add_edge(
                    left,
                    right,
                    key=key,
                    **deepcopy(data, runtime_memo),
                )
        else:
            for left, right, data in self._edges:
                graph.add_edge(left, right, **deepcopy(data, runtime_memo))

        _restore_mapping_order(graph._node, self._nodes, label="node order")
        if self._directed:
            _restore_mapping_order(
                graph._succ, self._nodes, label="successor node order"
            )
            _restore_mapping_order(
                graph._pred, self._nodes, label="predecessor node order"
            )
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
            _restore_mapping_order(
                graph._adj, self._nodes, label="adjacency node order"
            )
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

        graph.graph.clear()
        graph.graph.update(deepcopy(dict(self._ordinary_graph_data), runtime_memo))
        for snapshot, restored_value in restored_runtime_graph_data:
            graph.graph[snapshot.key] = restored_value
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
        for snapshot, restored_value in restored_graph_attributes:
            setattr(graph, snapshot.key, restored_value)

        if self._had_last_operator:
            graph._last_operator_applied = self._last_operator
        elif hasattr(graph, "_last_operator_applied"):
            delattr(graph, "_last_operator_applied")

    def restore_after_failure(
        self,
        graph: Any,
        failure: BaseException,
    ) -> bool:
        """Restore without replacing the primary transaction failure."""

        try:
            self.restore(graph)
        except BaseException as rollback_failure:
            note = (
                "TNFR graph rollback also failed: "
                f"{type(rollback_failure).__qualname__}: {rollback_failure}"
            )
            add_note = getattr(failure, "add_note", None)
            if callable(add_note):
                add_note(note)
            try:
                setattr(failure, "_tnfr_rollback_failure", rollback_failure)
            except Exception:
                pass
            return False
        return True


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
class PointwiseStageProposal:
    """Target-bound immutable payload for one pointwise operator."""

    node: Any
    glyph: Glyph
    payload: Any


@dataclass(frozen=True, slots=True)
class RecursivityStageProposal:
    """Target-bound view of one shared immutable REMESH advisory."""

    node: Any
    glyph: Glyph
    advisory: RecursivityAdvisoryProposal


@dataclass(frozen=True, slots=True)
class NetworkStageResult:
    """Completed network stage and its executable scheduling semantics.

    The pointwise or neighbor EPI jump certificate is populated only when the
    caller requests the opt-in EPI audit. Evidence is computed from the
    executor's detached stage-start graph and frozen proposals before any
    live structural commit; unsupported theorem domains report an explicit
    abstention reason.
    """

    operator: str
    glyph: str
    schedule: Literal["two_phase_jacobi", "operator_major_gauss_seidel"]
    nodes_processed: int
    pointwise_epi_jump_certificate: Any | None = field(
        default=None, repr=False, compare=False
    )
    neighbor_epi_jump_certificate: Any | None = field(
        default=None, repr=False, compare=False
    )
    epi_jump_certificate_abstention_reason: str | None = None

    def __post_init__(self) -> None:
        """Reject contradictory evidence payloads at their source."""

        certificates = (
            self.pointwise_epi_jump_certificate,
            self.neighbor_epi_jump_certificate,
        )
        if sum(certificate is not None for certificate in certificates) > 1:
            raise ValueError(
                "NetworkStageResult cannot contain multiple EPI jump certificates"
            )
        if (
            any(certificate is not None for certificate in certificates)
            and self.epi_jump_certificate_abstention_reason is not None
        ):
            raise ValueError(
                "A certified stage cannot also declare certificate abstention"
            )
        if (
            self.epi_jump_certificate_abstention_reason is not None
            and type(self.epi_jump_certificate_abstention_reason) is not str
        ):
            raise TypeError(
                "epi_jump_certificate_abstention_reason must be a string or None"
            )

    @property
    def epi_jump_certificate(self) -> Any | None:
        """Return the available operator-family EPI jump certificate."""

        if self.pointwise_epi_jump_certificate is not None:
            return self.pointwise_epi_jump_certificate
        return self.neighbor_epi_jump_certificate

    @property
    def epi_jump_certificate_kind(self) -> str | None:
        """Return ``pointwise`` or ``neighbor`` for the available certificate."""

        if self.pointwise_epi_jump_certificate is not None:
            return "pointwise"
        if self.neighbor_epi_jump_certificate is not None:
            return "neighbor"
        return None


@dataclass(frozen=True, slots=True)
class _TwoPhasePreflight:
    """Shared validated inputs for one immutable all-target stage."""

    snapshot: Any
    window: int
    execution_kwargs: dict[str, Any]
    states_before: Mapping[Any, dict[str, Any]]
    precondition_warnings: Mapping[
        Any, tuple[tuple[str, type[Warning]], ...]
    ]
    exact_stage: bool


def _discard_pending_monitor(graph: Any) -> None:
    monitor = graph.graph.get("integrity_monitor")
    discard = getattr(monitor, "discard_pending_operator", None)
    if callable(discard):
        try:
            discard()
        except Exception:
            pass


def _unique_stage_targets(targets: Sequence[Any]) -> tuple[Any, ...]:
    """Materialize a stage target order and reject duplicate identities."""

    resolved = tuple(targets)
    seen: set[Any] = set()
    for node in resolved:
        if node in seen:
            raise TNFRValueError(
                "network stage targets must be unique",
                context={"duplicate_target": repr(node)},
            )
        seen.add(node)
    return resolved


def _detached_stage_graph(graph: Any) -> Any:
    """Return a complete detached logical graph for immutable stage reads."""

    if graph.is_directed():
        snapshot = nx.MultiDiGraph() if graph.is_multigraph() else nx.DiGraph()
    else:
        snapshot = nx.MultiGraph() if graph.is_multigraph() else nx.Graph()
    copy_memo: dict[int, Any] = {
        id(graph): snapshot,
        id(graph.graph): snapshot.graph,
    }
    for key, value in graph.graph.items():
        if key in _RUNTIME_GRAPH_KEYS or "cache" in str(key).lower():
            copy_memo[id(value)] = value
        _seed_runtime_lock_memo(value, copy_memo)
    for _node, data in graph.nodes(data=True):
        _seed_runtime_lock_memo(data, copy_memo)
    for edge in (
        graph.edges(keys=True, data=True)
        if graph.is_multigraph()
        else graph.edges(data=True)
    ):
        _seed_runtime_lock_memo(edge[-1], copy_memo)
    for key, value in graph.graph.items():
        if key in _RUNTIME_GRAPH_KEYS or "cache" in str(key).lower():
            continue
        snapshot.graph[key] = deepcopy(value, copy_memo)
    # The monitor is never invoked while proposals are built, but its public
    # shape remains part of common operator argument preflight.
    monitor = graph.graph.get("integrity_monitor")
    if monitor is not None:
        snapshot.graph["integrity_monitor"] = monitor

    snapshot.add_nodes_from(
        (node, deepcopy(dict(data), copy_memo))
        for node, data in graph.nodes(data=True)
    )
    if graph.is_multigraph():
        snapshot.add_edges_from(
            (left, right, key, deepcopy(dict(data), copy_memo))
            for left, right, key, data in graph.edges(keys=True, data=True)
        )
    else:
        snapshot.add_edges_from(
            (left, right, deepcopy(dict(data), copy_memo))
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

        final_kind = reception_proposed_epi_kind(
            current_kind,
            (
                (value, _node_kind(snapshot, neighbor))
                for neighbor, value in zip(
                    neighbors, dominant_values, strict=True
                )
            ),
            unclipped_target_epi=proposed,
            fallback_kind=Glyph.EN.value,
        )
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


def _propose_pointwise(
    snapshot: Any,
    node: Any,
    operator: Any,
    glyph: Glyph,
    factors: Mapping[str, Any],
    *,
    timestamp: str | None,
    tau: Any = None,
    transition_now: datetime | None = None,
    resolved_seed: int | None = None,
    node_offset: int | None = None,
    execution_kwargs: Mapping[str, Any] | None = None,
    coherence_global_before: Any = None,
) -> PointwiseStageProposal:
    """Build one target-local proposal from the shared stage snapshot."""

    if glyph is Glyph.AL:
        from .al_sha_stage_proposals import propose_emission_stage

        if timestamp is None:
            raise RuntimeError("Emission stage proposal lacks a stage timestamp")
        payload = propose_emission_stage(
            snapshot, node, factors, timestamp=timestamp
        )
    elif glyph is Glyph.SHA:
        from .al_sha_stage_proposals import propose_silence_stage

        if timestamp is None:
            raise RuntimeError("Silence stage proposal lacks a stage timestamp")
        payload = propose_silence_stage(
            snapshot, node, factors, timestamp=timestamp
        )
    elif glyph is Glyph.IL:
        from ._coherence_stage_kernel import propose_coherence_stage
        from .preconditions.coherence import coherence_precondition_warnings

        kwargs = dict(execution_kwargs or {})
        warnings_enabled = bool(
            kwargs.get("validate_preconditions", True)
        ) and bool(
            snapshot.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False)
        )
        payload = propose_coherence_stage(
            snapshot,
            node,
            factors["IL_dnfr_factor"],
            radius=kwargs.get("coherence_radius", 1),
            phase_locking_coefficient=kwargs.get(
                "phase_locking_coefficient", 0.3
            ),
            global_before=coherence_global_before,
            precondition_warnings=(
                coherence_precondition_warnings(snapshot, node)
                if warnings_enabled
                else ()
            ),
        )
    elif glyph in (Glyph.VAL, Glyph.NUL):
        from ..constants import DEFAULTS
        from ._scale_operator_kernel import propose_scale_operator

        edge_aware = bool(
            snapshot.graph.get(
                "EDGE_AWARE_ENABLED",
                DEFAULTS.get("EDGE_AWARE_ENABLED", True),
            )
        )
        epi_before = None
        if edge_aware:
            epi_before = require_real_scalar_epi(
                _raw_alias(snapshot, node, ALIAS_EPI, 0.0),
                operator=(
                    "Expansion" if glyph is Glyph.VAL else "Contraction"
                ),
                label="target EPI state",
            )
        factor_key = "VAL_scale" if glyph is Glyph.VAL else "NUL_scale"
        payload = propose_scale_operator(
            glyph=glyph,
            factor=factors[factor_key],
            vf_before=_raw_alias(snapshot, node, ALIAS_VF, 0.0),
            dnfr_before=(
                _raw_alias(snapshot, node, ALIAS_DNFR, 0.0)
                if glyph is Glyph.NUL
                else None
            ),
            configured_densification_factor=(
                factors.get("NUL_densification_factor")
                if glyph is Glyph.NUL
                else None
            ),
            edge_aware_enabled=edge_aware,
            epi_before=epi_before,
            epi_min=snapshot.graph.get(
                "EPI_MIN", DEFAULTS.get("EPI_MIN", -1.0)
            ),
            epi_max=snapshot.graph.get(
                "EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0)
            ),
            epsilon=snapshot.graph.get(
                "EDGE_AWARE_EPSILON",
                DEFAULTS.get("EDGE_AWARE_EPSILON", 1e-12),
            ),
            clip_mode=str(snapshot.graph.get("CLIP_MODE", "hard")),
        )
    elif glyph is Glyph.ZHIR:
        from ._mutation_stage_kernel import propose_mutation_network_stage

        payload = propose_mutation_network_stage(
            snapshot,
            node,
            factors,
            tau=tau,
        )
    elif glyph is Glyph.NAV:
        if transition_now is None:
            raise RuntimeError("Transition stage proposal lacks a stage time")
        payload = operator._build_network_stage_proposal(
            snapshot,
            node,
            now=transition_now,
            resolved_seed=resolved_seed,
            node_offset=node_offset,
            **dict(execution_kwargs or {}),
        )
    else:
        raise ValueError(
            f"No pointwise stage proposal is registered for {glyph.value}"
        )
    return PointwiseStageProposal(node=node, glyph=glyph, payload=payload)


def _validate_pointwise_proposals(
    proposals: Sequence[PointwiseStageProposal],
    targets: tuple[Any, ...],
    glyph: Glyph,
) -> None:
    """Validate target binding and payload identity before the first write."""

    if len(proposals) != len(targets):
        raise RuntimeError("Pointwise stage proposal cardinality changed")
    if tuple(proposal.node for proposal in proposals) != targets:
        raise RuntimeError("Pointwise stage proposal target order changed")
    for proposal in proposals:
        if proposal.glyph is not glyph:
            raise RuntimeError("Pointwise stage proposal glyph changed")
        if getattr(proposal.payload, "glyph", None) is not glyph:
            raise RuntimeError("Pointwise stage payload glyph changed")
        payload_node = getattr(proposal.payload, "node", proposal.node)
        if payload_node != proposal.node:
            raise RuntimeError("Pointwise stage payload target changed")


def _validate_pointwise_epi_jump_certificate(
    snapshot: Any,
    proposals: tuple[PointwiseStageProposal, ...],
    glyph: Glyph,
    certificate: Any,
    *,
    fixed_support_declared: bool,
) -> None:
    """Bind one proof-stamped pointwise certificate to frozen proposals."""

    from ..operators.operator_contracts import contract_for
    from ..physics.pointwise_stage_stability import (
        PointwiseEPIJumpRealizationCertificate,
    )

    if type(certificate) is not PointwiseEPIJumpRealizationCertificate:
        raise RuntimeError(
            "Pointwise EPI certificate builder returned an unexpected type"
        )
    if not certificate._proof_fields_are_intact():
        raise RuntimeError("Pointwise EPI certificate proof fields are not intact")

    nodes = tuple(snapshot.nodes)
    targets = tuple(proposal.node for proposal in proposals)
    contract = contract_for(glyph.value)
    if (
        certificate.operator_name != contract.english_name
        or certificate.glyph != glyph.value
        or tuple(certificate.nodes) != nodes
        or tuple(certificate.target_nodes) != targets
        or certificate.stage_schedule != TWO_PHASE_JACOBI
        or certificate.fixed_support_declared is not fixed_support_declared
    ):
        raise RuntimeError(
            "Pointwise EPI certificate identity or schedule diverged from "
            "the frozen stage"
        )

    expected_before = tuple(
        require_real_scalar_epi(
            _raw_alias(snapshot, node, ALIAS_EPI, 0.0),
            operator=contract.english_name,
            label=f"node {node!r} EPI snapshot",
        )
        for node in nodes
    )
    expected_after = list(expected_before)
    node_indices = {node: index for index, node in enumerate(nodes)}
    for proposal in proposals:
        payload = proposal.payload
        if proposal.glyph is Glyph.AL:
            expected_after[node_indices[proposal.node]] = float(payload.epi_after)
        elif proposal.glyph in (Glyph.VAL, Glyph.NUL) and payload.write_epi:
            if payload.epi_after is None:
                raise RuntimeError(
                    "Pointwise EPI-writing proposal lost its endpoint"
                )
            expected_after[node_indices[proposal.node]] = float(payload.epi_after)

    if not np.array_equal(
        np.asarray(certificate.state_before, dtype=float),
        np.asarray(expected_before, dtype=float),
    ) or not np.array_equal(
        np.asarray(certificate.runtime_proposed_state_after, dtype=float),
        np.asarray(expected_after, dtype=float),
    ):
        raise RuntimeError(
            "Pointwise EPI certificate endpoints diverged from frozen proposals"
        )

    resolved_factors = resolve_runtime_operator_factors(
        snapshot.graph.get("GLYPH_FACTORS"), glyph, snapshot.graph
    )
    expected_linear = np.eye(len(nodes), dtype=float)
    expected_offset = np.zeros(len(nodes), dtype=float)
    for proposal in proposals:
        index = node_indices[proposal.node]
        payload = proposal.payload
        if proposal.glyph is Glyph.AL:
            expected_offset[index] = float(resolved_factors["AL_boost"])
        elif proposal.glyph in (Glyph.VAL, Glyph.NUL) and payload.write_epi:
            expected_linear[index, index] = float(payload.requested_scale)

    exact_linear = tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in expected_linear
    )
    exact_offset = tuple(
        Fraction.from_float(float(value)) for value in expected_offset
    )
    if (
        not np.array_equal(
            np.asarray(certificate.represented_linear_map, dtype=float),
            expected_linear,
        )
        or not np.array_equal(
            np.asarray(certificate.represented_offset, dtype=float),
            expected_offset,
        )
        or certificate.exact_represented_linear_map != exact_linear
        or certificate.exact_represented_offset != exact_offset
    ):
        raise RuntimeError(
            "Pointwise EPI certificate affine map diverged from frozen proposals"
        )


def _commit_pointwise_lifecycle_before_structure(
    graph: Any, proposals: Sequence[PointwiseStageProposal]
) -> None:
    """Merge target metadata and warnings whose direct order precedes writes."""

    from .al_sha_stage_proposals import (
        commit_emission_lifecycle,
        commit_silence_lifecycle,
    )

    for proposal in proposals:
        if proposal.glyph is Glyph.AL:
            commit_emission_lifecycle(graph, proposal.payload)
        elif proposal.glyph is Glyph.SHA:
            commit_silence_lifecycle(graph, proposal.payload)


def _commit_pointwise_structure(
    graph: Any, proposals: Sequence[PointwiseStageProposal]
) -> None:
    """Commit disjoint primary channels from already validated proposals."""

    from ..node import NodeNX
    from . import _set_epi_with_boundary_check
    from .al_sha_stage_proposals import (
        commit_emission_structure,
        commit_silence_structure,
    )

    for proposal in proposals:
        payload = proposal.payload
        if proposal.glyph is Glyph.AL:
            commit_emission_structure(graph, payload)
            continue
        if proposal.glyph is Glyph.SHA:
            commit_silence_structure(graph, payload)
            continue
        if proposal.glyph is Glyph.ZHIR:
            from ._mutation_stage_kernel import commit_mutation_structure

            commit_mutation_structure(graph, payload)
            continue
        if proposal.glyph is Glyph.NAV:
            from .transition import commit_transition_network_structure

            commit_transition_network_structure(graph, payload)
            continue

        node = NodeNX.from_graph(graph, proposal.node)
        if proposal.glyph is Glyph.IL:
            node.dnfr = payload.dnfr_after
            node.theta = payload.phase.theta_after
            continue
        node.vf = payload.vf_after
        if proposal.glyph is Glyph.NUL:
            if payload.dnfr_after is None:
                raise RuntimeError("Contraction proposal lacks DeltaNFR output")
            node.dnfr = payload.dnfr_after
        if payload.write_epi:
            if payload.epi_after is None:
                raise RuntimeError(
                    f"{proposal.glyph.value} proposal lacks required EPI output"
                )
            _set_epi_with_boundary_check(
                node, payload.epi_after, apply_clip=False
            )


def _merge_coherence_stage_telemetry(
    graph: Any,
    proposals: Sequence[PointwiseStageProposal],
) -> None:
    """Merge committed IL telemetry in requested target order."""

    coherence_proposals = tuple(
        proposal
        for proposal in proposals
        if proposal.glyph is Glyph.IL
    )
    if not coherence_proposals:
        return

    from ._coherence_stage_kernel import (
        capture_coherence_globals,
        coherence_phase_event,
        coherence_reduction_event,
        coherence_tracking_event,
    )

    global_after = capture_coherence_globals(graph)
    for proposal in coherence_proposals:
        payload = proposal.payload
        phase_event = coherence_phase_event(payload)
        if phase_event is not None:
            graph.graph.setdefault("IL_phase_locking", []).append(phase_event)
        graph.graph.setdefault("IL_dnfr_reductions", []).append(
            coherence_reduction_event(payload)
        )
        graph.graph.setdefault("IL_coherence_tracking", []).append(
            coherence_tracking_event(
                graph,
                payload,
                global_after=global_after,
                scope="stage",
            )
        )


def _merge_pointwise_audit_streams(
    graph: Any, proposals: Sequence[PointwiseStageProposal]
) -> None:
    """Append proposal-bound scale telemetry in requested target order."""

    from ._scale_operator_kernel import (
        edge_aware_intervention_event,
        nul_densification_event,
    )

    for proposal in proposals:
        payload = proposal.payload
        if proposal.glyph is Glyph.NUL:
            graph.graph.setdefault("nul_densification_log", []).append(
                nul_densification_event(payload, proposal.node)
            )
        if proposal.glyph in (Glyph.VAL, Glyph.NUL) and (
            payload.edge_aware_adapted
        ):
            graph.graph.setdefault("edge_aware_interventions", []).append(
                edge_aware_intervention_event(payload, proposal.node)
            )


def _merge_pointwise_posthistory_streams(
    graph: Any, proposals: Sequence[PointwiseStageProposal]
) -> None:
    """Append audit events whose operator-step timestamp follows history."""

    from ._mutation_stage_kernel import commit_mutation_lifecycle

    for proposal in proposals:
        if proposal.glyph is Glyph.ZHIR:
            commit_mutation_lifecycle(graph, proposal.payload)
        elif proposal.glyph is Glyph.NAV:
            from .transition import commit_transition_network_lifecycle

            commit_transition_network_lifecycle(graph, proposal.payload)


def _emit_pointwise_precommit_warnings(
    graph: Any,
    proposals: Sequence[PointwiseStageProposal],
) -> None:
    """Preserve NAV's established proposal-bound precommit warnings."""

    from .transition import emit_transition_network_warnings

    for proposal in proposals:
        if proposal.glyph is Glyph.NAV:
            emit_transition_network_warnings(graph, proposal.payload)


def _emit_coherence_commit_warnings(
    proposals: Sequence[PointwiseStageProposal],
) -> None:
    """Emit IL warnings after every fallible stage effect has succeeded."""

    for proposal in proposals:
        if proposal.glyph is Glyph.IL:
            for message in proposal.payload.precondition_warnings:
                warnings.warn(message, UserWarning, stacklevel=3)


def _verify_pointwise_postconditions(
    graph: Any,
    operator: Any,
    proposals: Sequence[PointwiseStageProposal],
    execution_kwargs: Mapping[str, Any],
) -> None:
    """Run configured operator postconditions inside the stage transaction."""

    if operator.glyph is not Glyph.ZHIR:
        return
    validate = bool(
        execution_kwargs.get("validate_postconditions", False)
    ) or bool(graph.graph.get("VALIDATE_OPERATOR_POSTCONDITIONS", False))
    if not validate:
        return
    for proposal in proposals:
        payload = proposal.payload
        operator._verify_postconditions(
            graph,
            proposal.node,
            {
                "theta": payload.phase.theta_before,
                "epi_kind": payload.epi_kind_before,
            },
        )


def _emit_pointwise_commit_logs(
    proposals: Sequence[PointwiseStageProposal],
) -> None:
    """Publish accepted proposal context without reopening the transaction.

    Logging is an irreversible external side effect. A custom failing handler
    must therefore neither roll back an already accepted graph nor prevent the
    remaining accepted targets from publishing their context.
    """

    from ._mutation_stage_kernel import emit_mutation_lifecycle_log

    for proposal in proposals:
        if proposal.glyph is Glyph.ZHIR:
            try:
                emit_mutation_lifecycle_log(proposal.payload)
            except Exception:
                # Structural state, history, diagnostics, checks, pressure and
                # the stage contract have already committed. Logging remains
                # best-effort so an external handler cannot contradict them.
                continue


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


def _certify_neighbor_epi_jump_from_proposals(
    snapshot: Any,
    proposals: tuple[NeighborStageProposal, ...],
    glyph: Glyph,
    factors: Mapping[str, Any],
) -> tuple[Any | None, str | None]:
    """Certify one EN/RA stage and bind it to its frozen proposals.

    The general all-target certificate rebuilds one read-only structural step
    from the same detached stage-start graph.  This adapter accepts that result
    only when every runtime-facing field agrees exactly with the proposals the
    executor is about to commit.  Mathematical domain rejection is an
    abstention; disagreement between two supposedly shared kernels is an
    internal error and must abort the surrounding transaction.
    """

    from ..physics.network_stage_stability import (
        AllTargetNeighborStageCertificate,
        _validate_bridge_stage_certificate,
        certify_all_target_neighbor_stage,
    )
    from ..physics._conductance import read_conductance
    from ..physics._neighbor_epi_realization import (
        exact_binary64_matrix,
        represented_neighbor_blend_map,
    )

    nodes = tuple(snapshot.nodes())
    targets = tuple(proposal.node for proposal in proposals)
    if not targets:
        return None, "neighbor_certificate_requires_nonempty_targets"
    if targets != nodes:
        return None, "neighbor_certificate_requires_complete_snapshot_node_order"

    # These are valid executor inputs but explicit exclusions of the symmetric
    # positive-conductance theorem used by this optional certificate.  Detect
    # them before entering the proof builder so an implementation exception
    # cannot be mislabeled as mathematical abstention.
    if len(nodes) < 2:
        return None, "neighbor_certificate_domain_rejected:at_least_two_nodes"
    if snapshot.is_directed():
        return None, "neighbor_certificate_domain_rejected:undirected_support"
    if any(not proposal.neighbors for proposal in proposals):
        return None, "neighbor_certificate_domain_rejected:nonempty_neighbor_sets"
    if any(
        not any(alias in snapshot.nodes[node] for alias in ALIAS_EPI)
        for node in nodes
    ):
        return None, "neighbor_certificate_domain_rejected:explicit_epi_state"

    try:
        conductance = read_conductance(snapshot, list(nodes), symmetric=True)
        strength = conductance.strength
    except ValueError as exc:
        return (
            None,
            "neighbor_certificate_domain_rejected:conductance:"
            f"{exc}",
        )
    if np.any(strength <= 0.0):
        return None, "neighbor_certificate_domain_rejected:positive_row_strength"
    adjacency = conductance.dense()
    reached = {0}
    frontier = [0]
    while frontier:
        source = frontier.pop()
        for target in np.flatnonzero(adjacency[source] > 0.0):
            target = int(target)
            if target != source and target not in reached:
                reached.add(target)
                frontier.append(target)
    if len(reached) != len(nodes):
        return None, "neighbor_certificate_domain_rejected:connected_conductance"

    capacities: list[float] = []
    for node in nodes:
        raw_capacity = _raw_alias(snapshot, node, ALIAS_VF, None)
        if isinstance(raw_capacity, (bool, str, bytes, bytearray, complex)):
            return None, "neighbor_certificate_domain_rejected:positive_capacity"
        try:
            capacity = float(raw_capacity)
        except (TypeError, ValueError, OverflowError):
            return None, "neighbor_certificate_domain_rejected:positive_capacity"
        if not math.isfinite(capacity) or capacity <= 0.0:
            return None, "neighbor_certificate_domain_rejected:positive_capacity"
        capacities.append(capacity)
    with np.errstate(
        over="ignore", divide="ignore", invalid="ignore", under="ignore"
    ):
        metric = strength / np.asarray(capacities, dtype=float)
    if not np.all(np.isfinite(metric)) or np.any(metric <= 0.0):
        return None, "neighbor_certificate_domain_rejected:finite_positive_metric"

    kwargs: dict[str, Any] = {
        "fixed_support_declared": True,
        "repetitions": 1,
    }
    if glyph is Glyph.EN:
        kwargs["mix_factor"] = factors["EN_mix"]
    else:
        kwargs.update(
            {
                "fixed_phase_neighbor_sets_declared": False,
                "mix_factor": factors["RA_epi_diff"],
                "vf_amplification_factor": factors[
                    "RA_vf_amplification"
                ],
                "phase_coupling_factor": factors["RA_phase_coupling"],
            }
        )
    certificate = certify_all_target_neighbor_stage(
        snapshot,
        glyph,
        **kwargs,
    )

    if type(certificate) is not AllTargetNeighborStageCertificate:
        raise RuntimeError(
            "Neighbor-stage certificate builder returned an unexpected type"
        )
    if (
        certificate.operator_name
        != ("Reception" if glyph is Glyph.EN else "Resonance")
        or certificate.glyph != glyph.value
        or tuple(certificate.nodes) != nodes
        or certificate.repetitions_requested != 1
        or certificate.repetitions_observed != 1
        or certificate.repetitions_completed != 1
        or not certificate.all_stages_admissible
        or len(certificate.steps) != 1
    ):
        raise RuntimeError(
            "Neighbor-stage certificate identity or cardinality diverged"
        )

    step = certificate.steps[0]
    resolved_mix = float(
        factors["EN_mix" if glyph is Glyph.EN else "RA_epi_diff"]
    )
    exact_resolved_mix = Fraction.from_float(resolved_mix)
    if (
        step.mix_factor != resolved_mix
        or step.exact_mix_factor != exact_resolved_mix
    ):
        raise RuntimeError(
            "Neighbor-stage certificate diverged from resolved runtime factors"
        )

    node_indices = {node: index for index, node in enumerate(nodes)}
    expected_rows = []
    for index, proposal in enumerate(proposals):
        expected_local_map = represented_neighbor_blend_map(
            len(nodes),
            index,
            tuple(node_indices[neighbor] for neighbor in proposal.neighbors),
            resolved_mix,
        )
        expected_rows.append(expected_local_map[index])
    expected_stage_map = np.asarray(expected_rows, dtype=float)
    if (
        not np.array_equal(
            np.asarray(step.represented_stage_map, dtype=float),
            expected_stage_map,
        )
        or step.exact_represented_stage_map
        != exact_binary64_matrix(expected_stage_map)
    ):
        raise RuntimeError(
            "Neighbor-stage certificate map diverged from frozen stage proposals"
        )

    if (
        step.index != 0
        or tuple(step.nodes) != nodes
        or not step.runtime_stage_admissible
        or step.atomic_rejection_nodes
        or len(step.local_certificates) != len(proposals)
        or tuple(step.runtime_neighbor_sets)
        != tuple(proposal.neighbors for proposal in proposals)
        or not np.array_equal(
            np.asarray(step.state_before, dtype=float),
            np.asarray(
                tuple(proposal.epi_before for proposal in proposals),
                dtype=float,
            ),
        )
        or not np.array_equal(
            np.asarray(step.runtime_accepted_state_after, dtype=float),
            np.asarray(
                tuple(proposal.epi_after for proposal in proposals),
                dtype=float,
            ),
        )
    ):
        raise RuntimeError(
            "Neighbor-stage certificate diverged from frozen stage proposals"
        )

    for index, (proposal, local) in enumerate(
        zip(proposals, step.local_certificates, strict=True)
    ):
        common_matches = bool(
            tuple(local.nodes) == nodes
            and local.target == proposal.node
            and local.target_index == index
            and tuple(local.runtime_neighbors) == proposal.neighbors
            and np.array_equal(
                np.asarray(local.state_before, dtype=float),
                np.asarray(step.state_before, dtype=float),
            )
            and float(local.unweighted_runtime_neighbor_mean)
            == proposal.neighbor_epi_mean
            and float(local.runtime_target_value) == proposal.epi_after
            and local.epi_kind_before == proposal.epi_kind_before
            and local.epi_kind_after == proposal.epi_kind_after
            and float(local.mix_factor) == resolved_mix
            and local.exact_mix_factor == exact_resolved_mix
        )
        if glyph is Glyph.EN:
            channel_matches = bool(
                proposal.write_epi
                and not proposal.write_vf
                and not proposal.write_theta
            )
        else:
            resolved_vf_amplification = float(
                factors["RA_vf_amplification"]
            )
            resolved_phase_coupling = float(factors["RA_phase_coupling"])
            channel_matches = bool(
                proposal.write_epi
                and proposal.write_theta
                and float(local.vf_amplification_factor)
                == resolved_vf_amplification
                and float(local.phase_coupling_factor)
                == resolved_phase_coupling
                and float(local.frequency_before[index])
                == proposal.vf_before
                and float(local.frequency_after[index])
                == proposal.vf_after
                and bool(local.frequency_amplification_active)
                is proposal.write_vf
                and float(local.phase_before) == proposal.theta_before
                and float(local.phase_after) == proposal.theta_after
            )
        if not common_matches or not channel_matches:
            raise RuntimeError(
                "Neighbor-stage local certificate diverged from its frozen proposal"
            )
    validated = _validate_bridge_stage_certificate(certificate)
    if validated.step is not step:
        raise RuntimeError("Neighbor-stage certificate validation lost its step")
    return certificate, None


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
    proposals: Sequence[NeighborStageProposal | PointwiseStageProposal],
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
    proposals: Sequence[NeighborStageProposal | PointwiseStageProposal],
    states_before: Mapping[Any, dict[str, Any]],
    execution_kwargs: Mapping[str, Any],
) -> None:
    snapshot._last_operator_applied = operator.name
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
    # Local import keeps the declarative registry independent of this runtime
    # module while making it the source of live stage diagnostics.
    from .stage_contracts import stage_schedule_metadata

    graph.graph[STAGE_SCHEDULE_KEY] = {
        "operator": operator.name,
        "glyph": operator.glyph.value,
        "schedule": schedule,
        "nodes_processed": count,
    }
    graph.graph[STAGE_CONTRACT_KEY] = stage_schedule_metadata(
        operator,
        observed_schedule=schedule,
    )


def record_gauss_seidel_stage(graph: Any, operator: Any, count: int) -> None:
    """Label an explicit transactional sequential stage without changing it."""

    _record_schedule(
        graph,
        operator=operator,
        schedule=OPERATOR_MAJOR_GAUSS_SEIDEL,
        count=count,
    )


def _execute_gauss_seidel_targets(
    graph: Any,
    operator: Any,
    targets: tuple[Any, ...],
    *,
    sequence_context: Any,
    execution_kwargs: Mapping[str, Any],
) -> None:
    """Run the established ordered target loop inside an outer transaction."""

    for node in targets:
        graph._last_operator_applied = operator.name
        if sequence_context is None:
            operator(graph, node, **execution_kwargs)
        else:
            operator(
                graph,
                node,
                sequence_context=sequence_context,
                **execution_kwargs,
            )


def _preflight_two_phase_stage(
    graph: Any,
    operator: Any,
    targets: tuple[Any, ...],
    *,
    sequence_context: Any,
    execution_kwargs: Mapping[str, Any],
    transition_now: datetime | None = None,
) -> _TwoPhasePreflight:
    """Validate all targets against one detached stage-start graph."""

    from . import _validated_execution_window
    from .grammar_application import enforce_canonical_grammar
    from .grammar_debt import require_replayable_history
    from .grammar_types import glyph_function_name

    window = _validated_execution_window(
        graph, execution_kwargs.get("window")
    )
    resolved_kwargs = dict(execution_kwargs)
    resolved_kwargs["window"] = window
    validate_graph_seed(graph)
    snapshot = _detached_stage_graph(graph)

    for node in targets:
        if node not in snapshot:
            raise KeyError(node)
        require_replayable_history(snapshot.nodes[node].get("glyph_history"))

    selections: list[Any] = []
    states_before: dict[Any, dict[str, Any]] = {}
    precondition_warnings: dict[
        Any, tuple[tuple[str, type[Warning]], ...]
    ] = {}
    needs_state_before = bool(
        resolved_kwargs.get("validate_nodal_equation", False)
    ) or bool(snapshot.graph.get("VALIDATE_NODAL_EQUATION", False)) or bool(
        resolved_kwargs.get("collect_metrics", False)
    ) or bool(snapshot.graph.get("COLLECT_OPERATOR_METRICS", False))
    for node in targets:
        operator._validate_hard_invariants(snapshot, node)
        validation_kwargs = dict(resolved_kwargs)
        if operator.glyph is Glyph.NAV:
            validation_kwargs["_stage_now"] = transition_now
        if operator.glyph is Glyph.IL:
            validation_kwargs["_defer_il_precondition_warnings"] = True
        if operator.glyph is Glyph.OZ:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                operator._validate_application_preconditions(
                    snapshot,
                    node,
                    sequence_context=sequence_context,
                    **validation_kwargs,
                )
            precondition_warnings[node] = tuple(
                (str(item.message), item.category) for item in caught
            )
        else:
            operator._validate_application_preconditions(
                snapshot,
                node,
                sequence_context=sequence_context,
                **validation_kwargs,
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
    if exact_stage and operator.glyph is Glyph.IL:
        from .coherence import Coherence, _CANONICAL_COHERENCE_EXECUTE

        exact_stage = (
            type(operator) is Coherence
            and type(operator)._execute is _CANONICAL_COHERENCE_EXECUTE
        )
    if exact_stage and operator.glyph is Glyph.OZ:
        from .dissonance import Dissonance, _CANONICAL_DISSONANCE_EXECUTE

        exact_stage = (
            type(operator) is Dissonance
            and type(operator)._execute is _CANONICAL_DISSONANCE_EXECUTE
        )
    if exact_stage and operator.glyph is Glyph.THOL:
        from .self_organization import (
            SelfOrganization,
            _CANONICAL_SELF_ORGANIZATION_EXECUTE,
        )

        exact_stage = (
            type(operator) is SelfOrganization
            and type(operator)._execute is _CANONICAL_SELF_ORGANIZATION_EXECUTE
        )
    return _TwoPhasePreflight(
        snapshot=snapshot,
        window=window,
        execution_kwargs=resolved_kwargs,
        states_before=states_before,
        precondition_warnings=precondition_warnings,
        exact_stage=exact_stage,
    )


def execute_operator_major_stage(
    graph: Any,
    operator: Any,
    targets: Sequence[Any],
    *,
    sequence_context: Any = None,
    compute_delta_nfr: Any = None,
    transaction_snapshot: GraphTransactionSnapshot | None = None,
    **execution_kwargs: Any,
) -> NetworkStageResult:
    """Execute one transactional operator-major Gauss--Seidel stage.

    This boundary deliberately preserves the established sequential read/write
    semantics: later targets may observe earlier target effects.  It adds only
    stage-level failure atomicity.  A late operator, telemetry, monitor, or
    pressure-refresh failure restores the complete stage-start graph.

    It is therefore not an all-target proposal rule and does not establish
    target-order invariance or relabeling equivariance.
    """

    if operator.glyph in (Glyph.EN, Glyph.RA):
        raise ValueError(
            "Reception and Resonance require the two-phase neighbour stage"
        )

    targets_tuple = _unique_stage_targets(targets)

    transaction = transaction_snapshot or GraphTransactionSnapshot(graph)
    try:
        _execute_gauss_seidel_targets(
            graph,
            operator,
            targets_tuple,
            sequence_context=sequence_context,
            execution_kwargs=execution_kwargs,
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
    except BaseException as failure:
        _discard_pending_monitor(graph)
        transaction.restore_after_failure(graph, failure)
        raise


def _commit_dissonance_target(graph: Any, proposal: Any) -> None:
    """Commit one frozen local OZ pressure and RNG transition."""

    from copy import deepcopy as _copy

    from ..alias import set_dnfr

    progress_key = "_rng_jitter_progress"
    storage = graph.nodes[proposal.node]
    has_progress = progress_key in storage
    progress = storage.get(progress_key)
    if (
        has_progress != proposal.had_jitter_progress
        or progress != proposal.jitter_progress_before
    ):
        raise RuntimeError(
            f"stale OZ jitter progress for target {proposal.node!r}"
        )
    if proposal.precondition_context is not None:
        storage["_oz_precondition_context"] = _copy(
            proposal.precondition_context
        )
    if (
        proposal.has_jitter_progress_after != proposal.had_jitter_progress
        or proposal.jitter_progress_after != proposal.jitter_progress_before
    ):
        if proposal.has_jitter_progress_after:
            storage[progress_key] = _copy(proposal.jitter_progress_after)
        else:
            storage.pop(progress_key, None)
    if proposal.writes_local_pressure:
        set_dnfr(graph, proposal.node, proposal.dnfr_local_after)


def _commit_dissonance_reduction(graph: Any, stage: Any) -> None:
    """Commit final pressures and ordered propagation telemetry."""

    from ..alias import set_dnfr

    propagation_key = "_oz_propagation"
    for update in stage.pressure_updates:
        if update.write_pressure:
            set_dnfr(graph, update.node, update.dnfr_after)

    for contribution in stage.contributions:
        storage = graph.nodes[contribution.neighbor]
        if propagation_key not in storage:
            storage[propagation_key] = []
        storage[propagation_key].append(contribution.event())

    if stage.graph_events:
        sink = graph.graph.setdefault("_oz_propagation_events", [])
        for event in stage.graph_events:
            sink.append(event.as_record())


def execute_dissonance_stage(
    graph: Any,
    operator: Any,
    targets: Sequence[Any],
    *,
    sequence_context: Any = None,
    compute_delta_nfr: Any = None,
    transaction_snapshot: GraphTransactionSnapshot | None = None,
    **execution_kwargs: Any,
) -> NetworkStageResult:
    """Execute one atomic all-target OZ stage from an immutable snapshot.

    Each target's local OZ proposal and outgoing propagation read the same
    stage-start graph. For every touched node, the final pressure is its local
    proposal when targeted (otherwise its snapshot pressure) plus all incoming
    propagation magnitudes, reduced with ``math.fsum`` in snapshot-node rank.
    This makes committed pressure and node-local RNG progress target-order
    invariant before the opaque pressure refresh. Histories, warnings, metrics,
    monitor calls and propagation event lists retain requested target order.

    The OZ magnitude postcondition applies to each local proposal. Incoming
    positive pressure can partially cancel a negative local value, so the final
    signed accumulation is reported separately and is not reclassified as the
    local destabilizing action.
    """

    if operator.glyph is not Glyph.OZ:
        raise ValueError("Dissonance stage supports OZ only")

    targets_tuple = _unique_stage_targets(targets)
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

        preflight = _preflight_two_phase_stage(
            graph,
            operator,
            targets_tuple,
            sequence_context=sequence_context,
            execution_kwargs=execution_kwargs,
        )
        snapshot = preflight.snapshot
        window = preflight.window
        execution_kwargs = preflight.execution_kwargs
        states_before = preflight.states_before
        if not preflight.exact_stage:

            _execute_gauss_seidel_targets(
                graph,
                operator,
                targets_tuple,
                sequence_context=sequence_context,
                execution_kwargs=execution_kwargs,
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

        resolve_runtime_operator_factors(
            snapshot.graph.get("GLYPH_FACTORS"),
            operator.glyph,
            snapshot.graph,
        )
        require_list_sink(
            snapshot.graph,
            "recognized_coherence_patterns",
            operator=operator.name,
        )

        from .dissonance import _resolve_dissonance_propagation

        propagate = _resolve_dissonance_propagation(snapshot, execution_kwargs)
        configured_seed: int | None = None
        resolved_seed: int | None = None
        if bool(snapshot.graph.get("OZ_NOISE_MODE", False)):
            sigma = _finite_scalar(
                snapshot.graph.get("OZ_SIGMA", 0.1),
                operator=operator.name,
                label="noise sigma",
            )
            if sigma > 0.0:
                configured_seed = validate_graph_seed(snapshot)
                resolved_seed = resolve_graph_seed(snapshot)

        from ._dissonance_stage_kernel import propose_dissonance_stage

        preconditions_validated = bool(
            execution_kwargs.get("validate_preconditions", True)
        ) and bool(
            snapshot.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False)
        )
        stage = propose_dissonance_stage(
            snapshot,
            targets_tuple,
            propagate=propagate,
            propagation_mode=execution_kwargs.get(
                "propagation_mode", "phase_weighted"
            ),
            preconditions_validated=preconditions_validated,
        )
        if stage.targets != targets_tuple:
            raise RuntimeError("Dissonance stage target order changed")
        if stage.graph_events:
            require_list_sink(
                snapshot.graph,
                "_oz_propagation_events",
                operator=operator.name,
            )

        if resolved_seed is not None:
            if validate_graph_seed(graph) != configured_seed:
                raise RuntimeError(
                    "stale OZ random seed: configuration changed before commit"
                )
            if configured_seed is None:
                graph.graph["RANDOM_SEED"] = resolved_seed

        graph._last_operator_applied = operator.name
        # Preserve the direct lifecycle boundary: each local pressure and its
        # history are visible to that target's monitor/metrics before network
        # propagation is reduced and published.
        for proposal in stage.target_proposals:
            _commit_dissonance_target(graph, proposal)
            _record_histories_and_patterns(graph, (proposal,), window=window)
            _run_postcommit_checks(
                graph,
                snapshot,
                operator,
                (proposal,),
                states_before,
                execution_kwargs,
            )

        _commit_dissonance_reduction(graph, stage)
        if callable(compute_delta_nfr):
            compute_delta_nfr(graph)
        _record_schedule(
            graph,
            operator=operator,
            schedule=TWO_PHASE_JACOBI,
            count=len(targets_tuple),
        )
        result = NetworkStageResult(
            operator=operator.name,
            glyph=operator.glyph.value,
            schedule=TWO_PHASE_JACOBI,
            nodes_processed=len(targets_tuple),
        )
        for node in targets_tuple:
            for message, category in preflight.precondition_warnings.get(
                node, ()
            ):
                warnings.warn(message, category, stacklevel=3)
        return result
    except BaseException as failure:
        _discard_pending_monitor(graph)
        transaction.restore_after_failure(graph, failure)
        raise


def _commit_coupling_structure(graph: Any, stage: Any) -> None:
    """Commit a validated UM merge in deterministic snapshot-rank order."""

    from ..alias import set_dnfr, set_theta, set_vf
    from ..node import add_edge

    for update in stage.node_updates:
        if update.theta_after is not None:
            set_theta(graph, update.node, update.theta_after)
        if update.vf_after is not None:
            set_vf(graph, update.node, update.vf_after)
        if update.dnfr_after is not None:
            set_dnfr(graph, update.node, update.dnfr_after)
    for edge in stage.edges:
        add_edge(graph, edge.left, edge.right, edge.weight)


def execute_coupling_stage(
    graph: Any,
    operator: Any,
    targets: Sequence[Any],
    *,
    sequence_context: Any = None,
    compute_delta_nfr: Any = None,
    transaction_snapshot: GraphTransactionSnapshot | None = None,
    **execution_kwargs: Any,
) -> NetworkStageResult:
    """Execute one atomic all-target UM stage from an immutable snapshot.

    The overlapping phase reducer is an explicit engine policy: shortest-arc
    displacements are averaged in snapshot-rank order. It is not a TNFR
    theorem. Structural phase, optional target capacity and pressure, and new
    edge support are target-order invariant before the opaque pressure refresh.
    Ordered histories, metrics, and monitor streams retain target order.
    """

    if operator.glyph is not Glyph.UM:
        raise ValueError("Coupling stage supports UM only")

    targets_tuple = _unique_stage_targets(targets)
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

        preflight = _preflight_two_phase_stage(
            graph,
            operator,
            targets_tuple,
            sequence_context=sequence_context,
            execution_kwargs=execution_kwargs,
        )
        snapshot = preflight.snapshot
        window = preflight.window
        execution_kwargs = preflight.execution_kwargs
        states_before = preflight.states_before

        if not preflight.exact_stage:

            _execute_gauss_seidel_targets(
                graph,
                operator,
                targets_tuple,
                sequence_context=sequence_context,
                execution_kwargs=execution_kwargs,
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
            snapshot.graph.get("GLYPH_FACTORS"),
            operator.glyph,
            snapshot.graph,
        )
        require_list_sink(
            snapshot.graph,
            "recognized_coherence_patterns",
            operator=operator.name,
        )

        functional_links = bool(
            snapshot.graph.get("UM_FUNCTIONAL_LINKS", True)
        )
        configured_seed = validate_graph_seed(snapshot)
        resolved_seed: int | None = None
        node_offsets: Mapping[Any, int] = {}
        if functional_links:
            resolved_seed = resolve_graph_seed(snapshot)
            from ..utils.cache import ensure_node_offset_map

            node_offsets = dict(ensure_node_offset_map(snapshot))

        from ._coupling_stage_kernel import propose_coupling_stage

        stage = propose_coupling_stage(
            snapshot,
            targets_tuple,
            factors,
            resolved_seed=resolved_seed,
            node_offsets=node_offsets,
        )
        if stage.targets != targets_tuple:
            raise RuntimeError("Coupling stage target order changed")

        if resolved_seed is not None:
            if validate_graph_seed(graph) != configured_seed:
                raise RuntimeError(
                    "stale UM random seed: configuration changed before commit"
                )
            if configured_seed is None:
                graph.graph["RANDOM_SEED"] = resolved_seed

        graph._last_operator_applied = operator.name
        _commit_coupling_structure(graph, stage)
        _record_histories_and_patterns(
            graph, stage.target_proposals, window=window
        )
        _run_postcommit_checks(
            graph,
            snapshot,
            operator,
            stage.target_proposals,
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
        return NetworkStageResult(
            operator=operator.name,
            glyph=operator.glyph.value,
            schedule=TWO_PHASE_JACOBI,
            nodes_processed=len(targets_tuple),
        )
    except BaseException as failure:
        _discard_pending_monitor(graph)
        transaction.restore_after_failure(graph, failure)
        raise


def execute_self_organization_stage(
    graph: Any,
    operator: Any,
    targets: Sequence[Any],
    *,
    sequence_context: Any = None,
    compute_delta_nfr: Any = None,
    transaction_snapshot: GraphTransactionSnapshot | None = None,
    **execution_kwargs: Any,
) -> NetworkStageResult:
    """Execute one atomic all-target THOL stage from an immutable snapshot.

    Every target is planned against the same detached stage-start graph. Child
    identifiers are allocated in snapshot-node rank, and the complete merged
    support, sub-EPI records and hierarchy are materialized and validated on a
    second detached graph before the first live write. Nodal channels and
    structural support then commit in snapshot rank, while histories,
    diagnostics, monitor calls and metrics retain requested target order.

    A subclass override or grammar replacement uses the transactional
    operator-major Gauss--Seidel fallback.
    """

    if operator.glyph is not Glyph.THOL:
        raise ValueError("Self-organization stage supports THOL only")

    targets_tuple = _unique_stage_targets(targets)
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

        preflight = _preflight_two_phase_stage(
            graph,
            operator,
            targets_tuple,
            sequence_context=sequence_context,
            execution_kwargs=execution_kwargs,
        )
        snapshot = preflight.snapshot
        window = preflight.window
        execution_kwargs = preflight.execution_kwargs
        states_before = preflight.states_before
        if not preflight.exact_stage:

            _execute_gauss_seidel_targets(
                graph,
                operator,
                targets_tuple,
                sequence_context=sequence_context,
                execution_kwargs=execution_kwargs,
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

        require_list_sink(
            snapshot.graph,
            "recognized_coherence_patterns",
            operator=operator.name,
        )
        from .self_organization import _merge_stage_execution_proposals

        raw = tuple(
            (node, operator._prepare_execution(snapshot, node, execution_kwargs))
            for node in targets_tuple
        )
        merged = _merge_stage_execution_proposals(snapshot, raw)
        proposals = tuple(
            PointwiseStageProposal(node=node, glyph=Glyph.THOL, payload=payload)
            for node, payload in merged
        )
        if tuple(proposal.node for proposal in proposals) != targets_tuple:
            raise RuntimeError("THOL stage proposal target order changed")
        if any(proposal.glyph is not Glyph.THOL for proposal in proposals):
            raise RuntimeError("THOL stage proposal glyph changed")

        rank = {node: index for index, node in enumerate(snapshot.nodes)}
        structural = tuple(
            sorted(proposals, key=lambda proposal: rank[proposal.node])
        )
        validation_candidate = _detached_stage_graph(snapshot)
        operator._validate_merged_stage_support(
            validation_candidate,
            tuple(
                (proposal.node, proposal.payload) for proposal in structural
            ),
        )

        graph._last_operator_applied = operator.name
        for proposal in structural:
            operator._commit_primary_channels(
                graph, proposal.node, proposal.payload
            )
        for proposal in structural:
            operator._commit_support_and_hierarchy(
                graph, proposal.node, proposal.payload
            )

        for proposal in proposals:
            _record_histories_and_patterns(graph, (proposal,), window=window)
            operator._commit_lifecycle_proposal(
                graph,
                proposal.node,
                proposal.payload,
                emit_depth_warning=False,
            )
            _run_postcommit_checks(
                graph,
                snapshot,
                operator,
                (proposal,),
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
        result = NetworkStageResult(
            operator=operator.name,
            glyph=operator.glyph.value,
            schedule=TWO_PHASE_JACOBI,
            nodes_processed=len(targets_tuple),
        )
        # Depth warnings are the final transactional effect, so a late monitor,
        # metric or pressure-refresh failure cannot publish warnings for an
        # aborted stage.
        for proposal in proposals:
            operator._emit_depth_limit_warning(
                proposal.node, proposal.payload
            )
        return result
    except BaseException as failure:
        _discard_pending_monitor(graph)
        transaction.restore_after_failure(graph, failure)
        raise


def execute_pointwise_stage(
    graph: Any,
    operator: Any,
    targets: Sequence[Any],
    *,
    sequence_context: Any = None,
    compute_delta_nfr: Any = None,
    transaction_snapshot: GraphTransactionSnapshot | None = None,
    epi_jump_fixed_support_declared: bool | None = None,
    _allow_epi_jump_certificate_abstention: bool = False,
    **execution_kwargs: Any,
) -> NetworkStageResult:
    """Execute an atomic all-target stage for a proven pointwise kernel.

    AL, IL, SHA, VAL, NUL, ZHIR and NAV have snapshot-merge rules. Their
    structural and lifecycle proposals are built from one detached stage-start
    graph, fully validated, and then merged. Ordered warnings, audit records,
    histories, metrics and monitor callbacks retain the requested target
    order. If live grammar selects any replacement glyph, the complete stage
    falls back to the established transactional Gauss--Seidel path.

    Supplying ``epi_jump_fixed_support_declared`` as an actual Boolean requests
    the conditional EPI realization/gain certificate. The certificate consumes
    this execution's own frozen proposals before commit and is returned on the
    ``NetworkStageResult``. It is unavailable for IL, empty stages, or a grammar
    replacement because none of those paths supplies the supported exact
    pointwise EPI proposal boundary.
    """

    if operator.glyph not in POINTWISE_TWO_PHASE_GLYPHS:
        raise ValueError(
            "Snapshot two-phase stages support AL, IL, SHA, VAL, NUL, ZHIR and NAV only"
        )
    if type(_allow_epi_jump_certificate_abstention) is not bool:
        raise TypeError(
            "_allow_epi_jump_certificate_abstention must be a bool"
        )
    if (
        epi_jump_fixed_support_declared is not None
        and type(epi_jump_fixed_support_declared) is not bool
    ):
        raise TypeError("epi_jump_fixed_support_declared must be a bool or None")
    if (
        epi_jump_fixed_support_declared is not None
        and operator.glyph not in POINTWISE_EPI_JUMP_GLYPHS
    ):
        raise ValueError(
            "The pointwise EPI jump certificate supports AL, SHA, VAL, NUL, "
            "ZHIR and NAV"
        )

    targets_tuple = _unique_stage_targets(targets)
    if (
        epi_jump_fixed_support_declared is not None
        and not targets_tuple
        and not _allow_epi_jump_certificate_abstention
    ):
        raise ValueError(
            "The pointwise EPI jump certificate requires at least one target"
        )

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
                epi_jump_certificate_abstention_reason=(
                    "pointwise_certificate_requires_nonempty_targets"
                    if epi_jump_fixed_support_declared is not None
                    else None
                ),
            )

        transition_now = (
            datetime.now(timezone.utc) if operator.glyph is Glyph.NAV else None
        )
        preflight = _preflight_two_phase_stage(
            graph,
            operator,
            targets_tuple,
            sequence_context=sequence_context,
            execution_kwargs=execution_kwargs,
            transition_now=transition_now,
        )
        snapshot = preflight.snapshot
        window = preflight.window
        execution_kwargs = preflight.execution_kwargs
        states_before = preflight.states_before
        if not preflight.exact_stage:
            if (
                epi_jump_fixed_support_declared is not None
                and not _allow_epi_jump_certificate_abstention
            ):
                raise TNFRValueError(
                    "Pointwise EPI certification requires the requested canonical "
                    "glyph to retain its two-phase proposal path.",
                    context={
                        "operator": operator.name,
                        "glyph": operator.glyph.value,
                        "reason": "grammar_replacement_or_noncanonical_operator",
                    },
                )
            _execute_gauss_seidel_targets(
                graph,
                operator,
                targets_tuple,
                sequence_context=sequence_context,
                execution_kwargs=execution_kwargs,
            )
            if callable(compute_delta_nfr):
                compute_delta_nfr(graph)
            record_gauss_seidel_stage(graph, operator, len(targets_tuple))
            return NetworkStageResult(
                operator=operator.name,
                glyph=operator.glyph.value,
                schedule=OPERATOR_MAJOR_GAUSS_SEIDEL,
                nodes_processed=len(targets_tuple),
                epi_jump_certificate_abstention_reason=(
                    "pointwise_certificate_requires_canonical_two_phase_stage"
                    if epi_jump_fixed_support_declared is not None
                    else None
                ),
            )

        factors = resolve_runtime_operator_factors(
            snapshot.graph.get("GLYPH_FACTORS"),
            operator.glyph,
            snapshot.graph,
        )
        require_list_sink(
            snapshot.graph,
            "recognized_coherence_patterns",
            operator=operator.name,
        )
        stage_now = transition_now or datetime.now(timezone.utc)
        timestamp = (
            stage_now.isoformat()
            if operator.glyph in (Glyph.AL, Glyph.SHA)
            else None
        )
        resolved_seed: int | None = None
        configured_seed: int | None = None
        node_offsets: Mapping[Any, int] = {}
        if (
            operator.glyph is Glyph.NAV
            and bool(snapshot.graph.get("NAV_RANDOM", True))
            and float(factors["NAV_jitter"]) > 0.0
        ):
            configured_seed = validate_graph_seed(snapshot)
            resolved_seed = resolve_graph_seed(snapshot)
            from ..utils.cache import ensure_node_offset_map

            node_offsets = dict(ensure_node_offset_map(snapshot))
        coherence_global_before = None
        if operator.glyph is Glyph.IL:
            from ._coherence_stage_kernel import capture_coherence_globals

            coherence_global_before = capture_coherence_globals(snapshot)
        proposals = tuple(
            _propose_pointwise(
                snapshot,
                node,
                operator,
                operator.glyph,
                factors,
                timestamp=timestamp,
                tau=execution_kwargs.get("tau"),
                transition_now=stage_now,
                resolved_seed=resolved_seed,
                node_offset=node_offsets.get(node),
                execution_kwargs=execution_kwargs,
                coherence_global_before=coherence_global_before,
            )
            for node in targets_tuple
        )
        _validate_pointwise_proposals(
            proposals, targets_tuple, operator.glyph
        )
        if operator.glyph is Glyph.NUL:
            require_list_sink(
                snapshot.graph,
                "nul_densification_log",
                operator=operator.name,
            )
        if any(
            proposal.glyph in (Glyph.VAL, Glyph.NUL)
            and proposal.payload.edge_aware_adapted
            for proposal in proposals
        ):
            require_list_sink(
                snapshot.graph,
                "edge_aware_interventions",
                operator=operator.name,
            )

        pointwise_certificate = None
        if epi_jump_fixed_support_declared is not None:
            from ..physics.pointwise_stage_stability import (
                certify_pointwise_epi_jump_realization,
            )

            pointwise_certificate = certify_pointwise_epi_jump_realization(
                snapshot,
                proposals,
                fixed_support_declared=epi_jump_fixed_support_declared,
                stage_schedule=TWO_PHASE_JACOBI,
                stage_timestamp=timestamp,
                zhir_tau=execution_kwargs.get("tau"),
                nav_transition_now=stage_now,
                nav_resolved_seed=resolved_seed,
                nav_node_offsets=node_offsets,
                nav_execution_kwargs=execution_kwargs,
            )
            _validate_pointwise_epi_jump_certificate(
                snapshot,
                proposals,
                operator.glyph,
                pointwise_certificate,
                fixed_support_declared=epi_jump_fixed_support_declared,
            )

        _emit_pointwise_precommit_warnings(graph, proposals)
        if resolved_seed is not None:
            if validate_graph_seed(graph) != configured_seed:
                raise RuntimeError(
                    "stale NAV random seed: configuration changed before commit"
                )
            if configured_seed is None:
                graph.graph["RANDOM_SEED"] = resolved_seed
        graph._last_operator_applied = operator.name
        _commit_pointwise_lifecycle_before_structure(graph, proposals)
        _commit_pointwise_structure(graph, proposals)
        _merge_coherence_stage_telemetry(graph, proposals)
        _merge_pointwise_audit_streams(graph, proposals)
        _record_histories_and_patterns(graph, proposals, window=window)
        _merge_pointwise_posthistory_streams(graph, proposals)
        _run_postcommit_checks(
            graph,
            snapshot,
            operator,
            proposals,
            states_before,
            execution_kwargs,
        )
        _verify_pointwise_postconditions(
            graph,
            operator,
            proposals,
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
        result = NetworkStageResult(
            operator=operator.name,
            glyph=operator.glyph.value,
            schedule=TWO_PHASE_JACOBI,
            nodes_processed=len(targets_tuple),
            pointwise_epi_jump_certificate=pointwise_certificate,
        )
        # IL precondition warnings are the final transactional effect. A
        # warning promoted to an exception therefore restores the stage, and
        # no later failure can publish a warning for an aborted IL commit.
        _emit_coherence_commit_warnings(proposals)
    except BaseException as failure:
        _discard_pending_monitor(graph)
        transaction.restore_after_failure(graph, failure)
        raise
    _emit_pointwise_commit_logs(proposals)
    return result


def _validate_recursivity_proposals(
    proposals: Sequence[RecursivityStageProposal],
    targets: tuple[Any, ...],
) -> None:
    """Validate target binding and the shared advisory before live writes."""

    if len(proposals) != len(targets):
        raise RuntimeError("Recursivity stage proposal cardinality changed")
    if tuple(proposal.node for proposal in proposals) != targets:
        raise RuntimeError("Recursivity stage proposal target order changed")
    if any(proposal.glyph is not Glyph.REMESH for proposal in proposals):
        raise RuntimeError("Recursivity stage proposal glyph changed")
    if proposals and any(
        proposal.advisory != proposals[0].advisory
        for proposal in proposals[1:]
    ):
        raise RuntimeError("Recursivity stage advisory diverged across targets")


def execute_recursivity_stage(
    graph: Any,
    operator: Any,
    targets: Sequence[Any],
    *,
    sequence_context: Any = None,
    compute_delta_nfr: Any = None,
    transaction_snapshot: GraphTransactionSnapshot | None = None,
    **execution_kwargs: Any,
) -> NetworkStageResult:
    """Execute the canonical REMESH advisory as one atomic snapshot stage.

    The node-level Recursivity glyph is intentionally advisory-only. Every
    target is preflighted against one detached graph, while one graph-level
    proposal is shared and committed at most once for the telemetry step.
    Histories, metrics and monitor calls retain requested target order.
    Structural EPI, nu_f, phase, DeltaNFR and topology remain unchanged before
    the opaque pressure-refresh callback. Explicit delayed EPI mixing remains
    available only through apply_network_remesh.

    A Recursivity subclass, overridden execution hook or grammar replacement
    retains the complete transactional operator-major compatibility path.
    """

    if operator.glyph is not Glyph.REMESH:
        raise ValueError("Recursivity stage supports REMESH only")

    targets_tuple = _unique_stage_targets(targets)
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

        preflight = _preflight_two_phase_stage(
            graph,
            operator,
            targets_tuple,
            sequence_context=sequence_context,
            execution_kwargs=execution_kwargs,
        )
        snapshot = preflight.snapshot
        window = preflight.window
        execution_kwargs = preflight.execution_kwargs
        states_before = preflight.states_before

        from .recursivity import (
            Recursivity,
            _CANONICAL_RECURSIVITY_EXECUTE,
        )

        exact_stage = (
            preflight.exact_stage
            and type(operator) is Recursivity
            and type(operator)._execute is _CANONICAL_RECURSIVITY_EXECUTE
        )
        if not exact_stage:
            _execute_gauss_seidel_targets(
                graph,
                operator,
                targets_tuple,
                sequence_context=sequence_context,
                execution_kwargs=execution_kwargs,
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

        resolve_runtime_operator_factors(
            snapshot.graph.get("GLYPH_FACTORS"),
            operator.glyph,
            snapshot.graph,
        )
        require_list_sink(
            snapshot.graph,
            "recognized_coherence_patterns",
            operator=operator.name,
        )

        from ._recursivity_stage_kernel import (
            commit_recursivity_advisory,
            propose_recursivity_advisory,
        )

        advisory = propose_recursivity_advisory(snapshot)
        proposals = tuple(
            RecursivityStageProposal(
                node=node,
                glyph=Glyph.REMESH,
                advisory=advisory,
            )
            for node in targets_tuple
        )
        _validate_recursivity_proposals(proposals, targets_tuple)

        # Exercise the exact advisory merge on a second detached graph before
        # the first live write. This validates history conversion and its
        # append sink without altering the monitor's stage-start snapshot.
        validation_graph = _detached_stage_graph(snapshot)
        commit_recursivity_advisory(validation_graph, advisory)

        graph._last_operator_applied = operator.name
        commit_recursivity_advisory(graph, advisory)
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
        return NetworkStageResult(
            operator=operator.name,
            glyph=operator.glyph.value,
            schedule=TWO_PHASE_JACOBI,
            nodes_processed=len(targets_tuple),
        )
    except BaseException as failure:
        _discard_pending_monitor(graph)
        transaction.restore_after_failure(graph, failure)
        raise


def execute_neighbor_stage(
    graph: Any,
    operator: Any,
    targets: Sequence[Any],
    *,
    sequence_context: Any = None,
    compute_delta_nfr: Any = None,
    transaction_snapshot: GraphTransactionSnapshot | None = None,
    include_epi_jump_certificate: bool = False,
    **execution_kwargs: Any,
) -> NetworkStageResult:
    """Execute one atomic EN/RA stage from a shared immutable snapshot.

    When requested, the optional EPI certificate is created before commit from
    this stage's detached snapshot and accepted only after exact comparison
    with every frozen proposal. Unsupported mathematical domains abstain while
    preserving ordinary stage execution.
    """

    if operator.glyph not in (Glyph.EN, Glyph.RA):
        raise ValueError("Two-phase network stages currently support EN and RA only")
    if type(include_epi_jump_certificate) is not bool:
        raise TypeError("include_epi_jump_certificate must be a bool")

    targets_tuple = _unique_stage_targets(targets)
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
                epi_jump_certificate_abstention_reason=(
                    "neighbor_certificate_requires_nonempty_targets"
                    if include_epi_jump_certificate
                    else None
                ),
            )
        preflight = _preflight_two_phase_stage(
            graph,
            operator,
            targets_tuple,
            sequence_context=sequence_context,
            execution_kwargs=execution_kwargs,
        )
        snapshot = preflight.snapshot
        window = preflight.window
        execution_kwargs = preflight.execution_kwargs
        states_before = preflight.states_before
        if not preflight.exact_stage:
            # Standalone grammar selection may legally replace the request.
            # Preserve that compatibility under a complete transaction and
            # label the resulting heterogeneous public-operator sweep honestly.
            _execute_gauss_seidel_targets(
                graph,
                operator,
                targets_tuple,
                sequence_context=sequence_context,
                execution_kwargs=execution_kwargs,
            )
            if callable(compute_delta_nfr):
                compute_delta_nfr(graph)
            record_gauss_seidel_stage(graph, operator, len(targets_tuple))
            return NetworkStageResult(
                operator=operator.name,
                glyph=operator.glyph.value,
                schedule=OPERATOR_MAJOR_GAUSS_SEIDEL,
                nodes_processed=len(targets_tuple),
                epi_jump_certificate_abstention_reason=(
                    "neighbor_certificate_requires_canonical_two_phase_stage"
                    if include_epi_jump_certificate
                    else None
                ),
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
        neighbor_certificate = None
        certificate_abstention_reason = None
        if include_epi_jump_certificate:
            (
                neighbor_certificate,
                certificate_abstention_reason,
            ) = _certify_neighbor_epi_jump_from_proposals(
                snapshot,
                proposals,
                operator.glyph,
                factors,
            )
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
            neighbor_epi_jump_certificate=neighbor_certificate,
            epi_jump_certificate_abstention_reason=(
                certificate_abstention_reason
            ),
        )
    except BaseException as failure:
        _discard_pending_monitor(graph)
        transaction.restore_after_failure(graph, failure)
        raise


__all__ = [
    "GraphTransactionSnapshot",
    "NetworkStageResult",
    "OPERATOR_MAJOR_GAUSS_SEIDEL",
    "STAGE_CONTRACT_KEY",
    "STAGE_SCHEDULE_KEY",
    "TWO_PHASE_JACOBI",
    "execute_coupling_stage",
    "execute_dissonance_stage",
    "execute_neighbor_stage",
    "execute_operator_major_stage",
    "execute_pointwise_stage",
    "execute_recursivity_stage",
    "execute_self_organization_stage",
    "POINTWISE_EPI_JUMP_GLYPHS",
    "POINTWISE_TWO_PHASE_GLYPHS",
    "record_gauss_seidel_stage",
]
