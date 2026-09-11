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

import copyreg
import logging
import math
import threading
import warnings
from collections import OrderedDict, defaultdict, deque
from collections.abc import (
    Mapping,
    MutableMapping,
    Sequence,
)
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
from functools import partial
from gc import get_referents
from inspect import getattr_static
from numbers import Real
from random import Random
from re import Pattern
from types import (
    BuiltinFunctionType,
    BuiltinMethodType,
    CellType,
    FunctionType,
    GetSetDescriptorType,
    MemberDescriptorType,
    MappingProxyType,
    MethodType,
    ModuleType,
)
from typing import Any, Literal
from weakref import ReferenceType, WeakValueDictionary
from zoneinfo import ZoneInfo

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
from ..mathematics.epi import BEPIElement
from ..mathematics.unified_numerical import np
from ..physics.mutation_trigger import (
    MutationTriggerCertificate,
    MutationTriggerEvidence,
)
from ..rng import resolve_graph_seed, validate_graph_seed
from ..types import Glyph
from ..utils import CallbackSpec, angle_diff
from ..utils.cache import NodeCache
from ..utils._structural_signature import (
    proof_stamps_are_identical,
    structural_object_state_signature,
    structural_proof_signature,
)
from ._argument_validation import require_list_sink
from ._epi_domain import require_real_scalar_epi
from ._neighbor_epi_kernel import (
    neighbor_epi_blend_value,
    neighbor_epi_unweighted_mean,
    reception_proposed_epi_kind,
)
from ._phase_gate import U3PhaseGateError, resolve_u3_phase_neighbors
from ._reception_kernel import (
    RECEPTION_PRE_STATE_BOUNDARY,
    ReceptionReadSnapshot,
    _reception_read_payload_stamp,
    capture_reception_read_snapshot,
    reception_no_sources_warning,
)
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
        "compute_delta_nfr",
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
        "succ",
        "pred",
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
_NETWORKX_STRUCTURAL_STORAGE_ATTRIBUTES = frozenset(
    {"graph", "_node", "_adj", "_succ", "_pred", "_last_operator_applied"}
)
_IMMUTABLE_CALLABLE_TYPES = (
    BuiltinFunctionType,
    MethodType,
    BuiltinMethodType,
    type,
)
_LOCK_TYPES = (type(threading.Lock()), type(threading.RLock()))
_EXTERNAL_RUNTIME_RESOURCE_TYPES = (
    *_LOCK_TYPES,
    logging.Logger,
    logging.RootLogger,
)
_IMMUTABLE_DESCRIPTOR_TYPES = (GetSetDescriptorType, MemberDescriptorType)
_NETWORKX_FACTORY_ATTRIBUTES = (
    "graph_attr_dict_factory",
    "node_dict_factory",
    "node_attr_dict_factory",
    "adjlist_outer_dict_factory",
    "adjlist_inner_dict_factory",
    "edge_key_dict_factory",
    "edge_attr_dict_factory",
)
_TYPE_DICTIONARY_DESCRIPTOR = type.__dict__["__dict__"]
_TYPE_MRO_DESCRIPTOR = type.__dict__["__mro__"]
_TYPE_FLAGS_DESCRIPTOR = type.__dict__["__flags__"]
_TYPE_NAME_DESCRIPTORS = {
    name: type.__dict__[name]
    for name in ("__module__", "__qualname__", "__name__")
}
_FUNCTION_TYPE_PARAMETERS_DESCRIPTOR = FunctionType.__dict__.get(
    "__type_params__"
)
_DEQUE_MAXLEN_DESCRIPTOR = deque.__dict__["maxlen"]
_DEFAULTDICT_FACTORY_DESCRIPTOR = defaultdict.__dict__["default_factory"]
_OBJECT_HASH_DESCRIPTOR = object.__dict__["__hash__"]
_OBJECT_EQUAL_DESCRIPTOR = object.__dict__["__eq__"]
_BUILTIN_METHOD_RECEIVER_DESCRIPTOR = BuiltinMethodType.__dict__["__self__"]
_PY_TPFLAGS_HEAPTYPE = 1 << 9
_MISSING_RUNTIME_BINDING = object()
_RUNTIME_DEEPCOPY_PROTOCOL_NAMES = frozenset(
    {
        "__deepcopy__",
        "__delattr__",
        "__getattr__",
        "__getattribute__",
        "__getnewargs__",
        "__getnewargs_ex__",
        "__getstate__",
        "__new__",
        "__reduce__",
        "__reduce_ex__",
        "__setattr__",
        "__setstate__",
    }
)
_RUNTIME_ATTRIBUTE_PROTOCOL_NAMES = frozenset(
    {"__delattr__", "__getattr__", "__getattribute__", "__setattr__"}
)
_SAFE_DEEPCOPY_PROTOCOL_OWNERS = frozenset(
    owner
    for kind in (
        tuple,
        list,
        dict,
        set,
        frozenset,
        deque,
        defaultdict,
        OrderedDict,
        WeakValueDictionary,
        partial,
        Random,
    )
    for owner in kind.__mro__
)
_CALLBACK_SPEC_PROTOCOL_BINDINGS = tuple(
    (name, CallbackSpec.__dict__[name])
    for name in ("__new__", "__getnewargs__")
)
_NODE_CACHE_PROTOCOL_BINDINGS = (
    ("__reduce__", NodeCache.__dict__["__reduce__"]),
)
_BEPI_ELEMENT_PROTOCOL_BINDINGS = tuple(
    (name, BEPIElement.__dict__[name])
    for name in ("__delattr__", "__getstate__", "__setattr__", "__setstate__")
)
_ATOMIC_IMMUTABLE_RUNTIME_TYPES = (
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    range,
    Fraction,
    date,
    datetime,
    time,
    timedelta,
    timezone,
    Decimal,
    Pattern,
    ZoneInfo,
    ReferenceType,
)


def _numpy_atomic_scalar_runtime_types() -> frozenset[type[Any]]:
    """Return exact NumPy value-scalar types with no mutable referent."""

    if np is None:
        return frozenset()
    scalar_types: set[type[Any]] = set()
    for candidate in getattr(np, "sctypeDict", {}).values():
        if not isinstance(candidate, type):
            continue
        try:
            safe = (
                issubclass(candidate, np.generic)
                and not issubclass(candidate, np.void)
                and not np.dtype(candidate).hasobject
            )
        except (TypeError, ValueError):
            safe = False
        if safe:
            scalar_types.add(candidate)
    return frozenset(scalar_types)


_NUMPY_SCALAR_RUNTIME_TYPES = _numpy_atomic_scalar_runtime_types()
_NUMPY_ARRAY_CAPTURE_HOOKS = frozenset(
    {
        "__array__",
        "__array_finalize__",
        "__array_function__",
        "__array_ufunc__",
        "__getattr__",
        "__getattribute__",
        "__getnewargs__",
        "__getnewargs_ex__",
        "__getstate__",
        "__new__",
        "__reduce__",
        "__reduce_ex__",
        "__setstate__",
    }
)


def _runtime_object_array_item_is_immutable(value: Any) -> bool:
    """Recognize object-array members with no mutable reachable state."""

    if _atomic_immutable_runtime_value(value) or type(value) is object:
        return True
    if type(value) is tuple:
        return all(
            _runtime_object_array_item_is_immutable(item)
            for item in tuple.__iter__(value)
        )
    if type(value) is frozenset:
        return all(
            _runtime_object_array_item_is_immutable(item)
            for item in frozenset.__iter__(value)
        )
    return False


def _runtime_ndarray_dtype_and_flat(value: Any) -> tuple[Any, Any]:
    """Read ndarray storage after rejecting subclass-controlled hooks."""

    for owner in _runtime_class_mro(type(value)):
        if owner is np.ndarray:
            break
        namespace = _runtime_class_namespace(owner)
        declared = tuple(
            name for name in _NUMPY_ARRAY_CAPTURE_HOOKS if name in namespace
        )
        if declared:
            raise TNFRValueError(
                "NumPy ndarray subclasses with custom array/copy hooks cannot "
                "be snapshotted observationally"
            )

    dtype = type(np.ndarray.dtype).__get__(
        np.ndarray.dtype,
        value,
        type(value),
    )
    flat = type(np.ndarray.flat).__get__(
        np.ndarray.flat,
        value,
        type(value),
    )
    return dtype, flat


def _validate_runtime_ndarray_references(
    value: Any,
    *,
    captured_reference_ids: frozenset[int] = frozenset(),
) -> None:
    """Reject mutable referents hidden behind an object-dtype ndarray.

    ``numpy.array(..., copy=True)`` and ``numpy.copyto`` copy object-array
    pointers rather than the reachable Python state.  Treating such an array
    as a complete rollback root would therefore leave mutations to a list,
    mapping, callable, or arbitrary instance alive after restoration.  Exact
    immutable atoms and recursively immutable built-in tuples/frozensets are
    safe because retaining their identities is the complete logical state.

    Read the dtype and elements through the base ndarray descriptors so an
    ndarray subclass cannot run an attribute override during this preflight.
    """

    dtype, flat = _runtime_ndarray_dtype_and_flat(value)
    if not dtype.hasobject:
        return
    if any(
        not _runtime_object_array_item_is_immutable(item)
        and id(item) not in captured_reference_ids
        for item in flat
    ):
        raise TNFRValueError(
            "NumPy object array contains mutable references and cannot be "
            "snapshotted atomically"
        )


def _preflight_runtime_epi_history(
    runtime_items: tuple[tuple[Any, Any], ...],
) -> tuple[tuple[int, Any], ...]:
    """Validate canonical history storage and expose object-array referents.

    The transaction must be established before REMESH reads configuration or
    plans a map. This preflight therefore gives the known ``_epi_hist`` channel
    its domain error before a generic opaque-state error. Mutable referents are
    returned as explicit rollback roots; no other object array receives this
    exception.
    """

    referents: list[tuple[int, Any]] = []
    for key, history in runtime_items:
        if type(key) is not str or key != "_epi_hist" or history is None:
            continue
        owners = _runtime_class_mro(type(history))
        if (
            tuple in owners
            or list in owners
            or deque in owners
            or type(history) is range
        ):
            continue
        if np is not None and np.ndarray in owners:
            dtype, flat = _runtime_ndarray_dtype_and_flat(history)
            shape = type(np.ndarray.shape).__get__(
                np.ndarray.shape,
                history,
                type(history),
            )
            if type(shape) is not tuple or len(shape) != 1:
                raise TNFRValueError(
                    "_epi_hist NumPy storage must be one-dimensional"
                )
            if dtype.hasobject:
                referents.extend(
                    (id(history), item)
                    for item in flat
                    if id(item) != id(history)
                    and not _runtime_object_array_item_is_immutable(item)
                )
            continue
        raise TNFRValueError("_epi_hist must be a replayable indexed history")
    return tuple(referents)


def _is_runtime_graph_key(key: Any) -> bool:
    """Classify runtime keys without invoking user string or hash methods."""

    return type(key) is str and (
        key in _RUNTIME_GRAPH_KEYS or "cache" in key.lower()
    )


@dataclass(frozen=True, slots=True)
class _RuntimeSlotDescriptor:
    """One owner-qualified built-in slot descriptor."""

    owner: type[Any]
    declared_name: str
    storage_name: str
    descriptor: Any


@dataclass(frozen=True, slots=True)
class _RuntimeSlotState:
    """Captured state for one concrete slot, including hidden MRO storage."""

    slot: _RuntimeSlotDescriptor
    present: bool
    value: Any


@dataclass(frozen=True, slots=True)
class _RuntimeGraphValue:
    """Reference-preserving snapshot for graph runtime objects."""

    key: Any
    value: Any
    value_type: type[Any]
    container_kind: str | None
    container_state: Any
    object_namespace: dict[Any, Any] | None
    object_state: Mapping[str, Any] | None
    slot_state: tuple[_RuntimeSlotState, ...]


@dataclass(frozen=True, slots=True)
class _RuntimeNDArrayState:
    """Logical ndarray state needed for identity-preserving restoration."""

    values: Any
    writeable: bool


@dataclass(frozen=True, slots=True)
class _RuntimeClosureCellState:
    """Original binding of one closure cell."""

    cell: CellType
    present: bool
    value: Any


@dataclass(frozen=True, slots=True)
class _RuntimeFunctionState:
    """Mutable bindings owned by one configured Python function."""

    function: FunctionType
    namespace: dict[Any, Any]
    namespace_items: tuple[tuple[Any, Any], ...]
    annotations: dict[str, Any]
    code: Any
    defaults: tuple[Any, ...] | None
    documentation: str | None
    keyword_defaults: dict[str, Any] | None
    module: str | None
    name: str
    qualified_name: str
    type_parameters: Any
    closure: tuple[CellType, ...] | None
    closure_cells: tuple[_RuntimeClosureCellState, ...]


@dataclass(frozen=True, slots=True)
class _NetworkXRuntimeLayout:
    """NetworkX storage materialized without calling graph virtual APIs."""

    directed: bool
    multigraph: bool
    graph_mapping: MutableMapping[Any, Any]
    node_outer: MutableMapping[Any, Any]
    adjacency_outer: MutableMapping[Any, Any]
    predecessor_outer: MutableMapping[Any, Any] | None
    node_data: tuple[tuple[Any, MutableMapping[Any, Any]], ...]
    adjacency_inner: tuple[tuple[Any, MutableMapping[Any, Any]], ...]
    predecessor_inner: tuple[tuple[Any, MutableMapping[Any, Any]], ...]
    adjacency_edge_keys: tuple[tuple[Any, Any, MutableMapping[Any, Any]], ...]
    predecessor_edge_keys: tuple[
        tuple[Any, Any, MutableMapping[Any, Any]], ...
    ]
    edges: tuple[Any, ...]


def _runtime_descriptor_get(descriptor: Any, value: Any, owner: type[Any]) -> Any:
    """Read state through a known built-in descriptor implementation."""

    return type(descriptor).__get__(descriptor, value, owner)


def _runtime_class_mro(kind: type[Any]) -> tuple[type[Any], ...]:
    """Read a class MRO without invoking a custom metaclass hook."""

    owners = _runtime_descriptor_get(_TYPE_MRO_DESCRIPTOR, kind, type(kind))
    if type(owners) is not tuple:
        raise TNFRValueError("runtime class MRO must be a tuple")
    return owners


def _runtime_derives_from(value: Any, *bases: type[Any]) -> bool:
    """Test concrete inheritance without consulting ``value.__class__``."""

    owners = _runtime_class_mro(type(value))
    return any(base in owners for base in bases)


def _runtime_type_is_heap_allocated(kind: type[Any]) -> bool:
    """Return the interpreter heap-type flag without metaclass dispatch."""

    flags = _runtime_descriptor_get(_TYPE_FLAGS_DESCRIPTOR, kind, type(kind))
    return type(flags) is int and bool(flags & _PY_TPFLAGS_HEAPTYPE)


def _runtime_type_has_unmodeled_c_state(kind: type[Any]) -> bool:
    """Detect a C-defined storage layer outside the modeled built-in types."""

    for owner in _runtime_class_mro(kind):
        if owner is object:
            continue
        if not _runtime_type_is_heap_allocated(owner):
            return True
        constructor = _runtime_class_namespace(owner).get("__new__")
        if type(constructor) is BuiltinFunctionType:
            return True
    return False


def _runtime_class_namespace(kind: type[Any]) -> MappingProxyType:
    """Read a class namespace without invoking a custom metaclass hook."""

    namespace = _runtime_descriptor_get(
        _TYPE_DICTIONARY_DESCRIPTOR,
        kind,
        type(kind),
    )
    if type(namespace) is not MappingProxyType:
        raise TNFRValueError("runtime class namespace must be a mapping proxy")
    return namespace


def _runtime_class_text(kind: type[Any], name: str) -> str:
    """Read stable class text through the corresponding type descriptor."""

    value = _runtime_descriptor_get(_TYPE_NAME_DESCRIPTORS[name], kind, type(kind))
    if type(value) is not str:
        raise TNFRValueError(f"runtime class {name} must be a string")
    return value


def _runtime_special_method(kind: type[Any], name: str) -> Any:
    """Resolve one special method without invoking metaclass user code."""

    for owner in _runtime_class_mro(kind):
        namespace = _runtime_class_namespace(owner)
        if name in namespace:
            return namespace[name]
    return None


def _runtime_deque_maxlen(value: deque[Any]) -> int | None:
    """Read deque capacity without invoking a subclass attribute hook."""

    return _runtime_descriptor_get(_DEQUE_MAXLEN_DESCRIPTOR, value, type(value))


def _runtime_default_factory(value: defaultdict[Any, Any]) -> Any:
    """Read defaultdict factory without invoking an attribute override."""

    return _runtime_descriptor_get(
        _DEFAULTDICT_FACTORY_DESCRIPTOR,
        value,
        type(value),
    )


def _runtime_instance_namespace_descriptor(value: Any) -> Any | None:
    """Find the built-in descriptor owning an instance namespace."""

    # ``isinstance(value, type)`` may consult a hostile virtual ``__class__``.
    # Reading the concrete type's MRO through the built-in descriptor is
    # observational and also handles custom metaclasses.
    if type in _runtime_class_mro(type(value)):
        return None
    for owner in _runtime_class_mro(type(value)):
        descriptor = _runtime_class_namespace(owner).get("__dict__")
        if descriptor is None:
            continue
        if type(descriptor) is not GetSetDescriptorType:
            raise TNFRValueError(
                "runtime instance namespace descriptor is not a built-in getset"
            )
        return descriptor
    return None


def _runtime_instance_namespace(value: Any) -> dict[Any, Any] | None:
    """Read an instance dictionary without invoking ``__getattribute__``."""

    descriptor = _runtime_instance_namespace_descriptor(value)
    if descriptor is None:
        return None
    namespace = _runtime_descriptor_get(descriptor, value, type(value))
    if type(namespace) is not dict:
        raise TNFRValueError("runtime instance namespace must be a built-in dict")
    return namespace


def _runtime_slot_descriptors(value: Any) -> tuple[_RuntimeSlotDescriptor, ...]:
    """Return every concrete slot descriptor across an object's MRO.

    The owner is retained because Python permits a subclass to redeclare an
    inherited slot name.  Those declarations allocate distinct storage even
    though ordinary ``getattr`` exposes only the most-derived descriptor.
    """

    descriptors: list[_RuntimeSlotDescriptor] = []
    for cls in _runtime_class_mro(type(value)):
        namespace = _runtime_class_namespace(cls)
        raw_slots = namespace.get("__slots__", ())
        slots = (raw_slots,) if type(raw_slots) is str else tuple(raw_slots)
        for raw_name in slots:
            if raw_name in {"__dict__", "__weakref__"}:
                continue
            name = raw_name
            if raw_name.startswith("__") and not raw_name.endswith("__"):
                class_name = _runtime_class_text(cls, "__name__")
                name = f"_{class_name.lstrip('_')}{raw_name}"
            descriptor = namespace.get(name)
            if type(descriptor) is not MemberDescriptorType:
                raise TNFRValueError(
                    "runtime slot declaration has no built-in member descriptor"
                )
            descriptors.append(
                _RuntimeSlotDescriptor(
                    owner=cls,
                    declared_name=raw_name,
                    storage_name=name,
                    descriptor=descriptor,
                )
            )
    return tuple(descriptors)


def _read_runtime_slot(value: Any, slot: _RuntimeSlotDescriptor) -> Any:
    """Read one slot through its declaring owner's built-in descriptor."""

    return type(slot.descriptor).__get__(slot.descriptor, value, type(value))


def _write_runtime_slot(
    value: Any,
    slot: _RuntimeSlotDescriptor,
    state: Any,
) -> None:
    """Write one slot without resolving a same-named derived descriptor."""

    type(slot.descriptor).__set__(slot.descriptor, value, state)


def _delete_runtime_slot(value: Any, slot: _RuntimeSlotDescriptor) -> None:
    """Delete one slot without resolving a same-named derived descriptor."""

    type(slot.descriptor).__delete__(slot.descriptor, value)


def _runtime_slot_is_present(value: Any, slot: _RuntimeSlotDescriptor) -> bool:
    try:
        _read_runtime_slot(value, slot)
    except AttributeError:
        return False
    return True


def _validate_runtime_deepcopy_protocol(
    value: Any,
    *,
    manual_capture: bool = False,
) -> None:
    """Reject hooks that could execute if ``deepcopy`` reached ``value``."""

    kind = type(value)
    if any(
        registered is kind
        for registered in dict.__iter__(copyreg.dispatch_table)
    ):
        raise TNFRValueError(
            "runtime metadata with a registered copyreg reducer cannot be "
            "deep-copied observationally"
        )
    namespace = _runtime_instance_namespace(value)
    if namespace is not None:
        instance_hooks = tuple(
            name
            for name in _RUNTIME_DEEPCOPY_PROTOCOL_NAMES
            if name in namespace
        )
        if instance_hooks:
            raise TNFRValueError(
                "runtime metadata with instance copy-protocol hooks cannot "
                "be deep-copied observationally"
            )
    trusted_bindings = None
    trusted_name = None
    if kind is CallbackSpec:
        trusted_bindings = _CALLBACK_SPEC_PROTOCOL_BINDINGS
        trusted_name = "CallbackSpec"
    elif kind is NodeCache:
        trusted_bindings = _NODE_CACHE_PROTOCOL_BINDINGS
        trusted_name = "NodeCache"
    elif kind is BEPIElement:
        trusted_bindings = _BEPI_ELEMENT_PROTOCOL_BINDINGS
        trusted_name = "BEPIElement"
    if trusted_bindings is not None:
        trusted_namespace = _runtime_class_namespace(kind)
        declared = frozenset(
            name
            for name in _RUNTIME_DEEPCOPY_PROTOCOL_NAMES
            if name in trusted_namespace
        )
        expected = frozenset(
            name for name, _binding in trusted_bindings
        )
        bindings_are_canonical = (
            declared == expected
            and all(
                trusted_namespace[name] is binding
                for name, binding in trusted_bindings
            )
        )
        if not bindings_are_canonical:
            raise TNFRValueError(
                f"runtime {trusted_name} has modified copy-protocol hooks and "
                "cannot be snapshotted observationally"
            )
        return
    for owner in _runtime_class_mro(kind):
        if any(
            owner is safe_owner
            for safe_owner in _SAFE_DEEPCOPY_PROTOCOL_OWNERS
        ) or (
            np is not None and owner is np.ndarray
        ):
            continue
        if not _runtime_type_is_heap_allocated(owner):
            continue
        owner_namespace = _runtime_class_namespace(owner)
        declared = tuple(
            name
            for name in _RUNTIME_DEEPCOPY_PROTOCOL_NAMES
            if name in owner_namespace
            and not (
                manual_capture and name in _RUNTIME_ATTRIBUTE_PROTOCOL_NAMES
            )
        )
        if not declared:
            continue
        if "__deepcopy__" in declared:
            raise TNFRValueError(
                "runtime metadata with a custom __deepcopy__ hook cannot be "
                "snapshotted observationally or restored atomically; runtime "
                "object cannot be snapshotted atomically"
            )
        opaque_detail = (
            "; opaque interpreter state cannot be restored atomically"
            if _runtime_type_has_unmodeled_c_state(kind)
            else ""
        )
        raise TNFRValueError(
            "runtime metadata with custom copy-protocol or attribute hooks "
            f"cannot be deep-copied observationally{opaque_detail}"
        )


def _runtime_owner_qualified_slot_state(value: Any) -> tuple[Any, ...]:
    """Materialize every slot with an owner-qualified identity token."""

    state: list[Any] = []
    for slot in _runtime_slot_descriptors(value):
        key = (
            _runtime_class_text(slot.owner, "__module__"),
            _runtime_class_text(slot.owner, "__qualname__"),
            id(slot.owner),
            slot.declared_name,
            slot.storage_name,
        )
        try:
            slot_value = _read_runtime_slot(value, slot)
        except AttributeError:
            state.append((key, False, None))
        else:
            state.append((key, True, slot_value))
    return tuple(state)


def _graph_factory_items(graph: Any) -> tuple[tuple[str, Any], ...]:
    """Return stable raw NetworkX factory bindings that influence graph state."""

    items: list[tuple[str, Any]] = []
    for name in _NETWORKX_FACTORY_ATTRIBUTES:
        try:
            value = getattr_static(graph, name)
        except AttributeError:
            continue
        if type(value) in (staticmethod, classmethod):
            value = value.__func__
        elif type(value) is MemberDescriptorType:
            try:
                value = type(value).__get__(value, graph, type(graph))
            except AttributeError:
                continue
        items.append((name, value))
    return tuple(items)


def _graph_factory_state_signature(graph: Any) -> tuple[Any, ...]:
    """Sign factory identities and their instance-owned mutable state."""

    return tuple(
        (name, structural_object_state_signature(value))
        for name, value in _graph_factory_items(graph)
    )


def _networkx_internal_mapping_items(
    graph: Any,
    *,
    layout: _NetworkXRuntimeLayout | None = None,
) -> tuple[tuple[Any, Any], ...]:
    """Return mappings created by NetworkX's structural dict factories."""

    if layout is None:
        layout = _networkx_runtime_layout(graph)
    items: list[tuple[Any, Any]] = [
        (("node-outer",), layout.node_outer),
        (("adjacency-outer",), layout.adjacency_outer),
    ]
    items.extend(
        (("adjacency-inner", node), value)
        for node, value in layout.adjacency_inner
    )
    if layout.directed:
        items.append((("predecessor-outer",), layout.predecessor_outer))
        items.extend(
            (("predecessor-inner", node), value)
            for node, value in layout.predecessor_inner
        )
    if layout.multigraph:
        items.extend(
            (("edge-key-adjacency", node, neighbor), value)
            for node, neighbor, value in layout.adjacency_edge_keys
        )
        if layout.directed:
            items.extend(
                (("edge-key-predecessor", node, neighbor), value)
                for node, neighbor, value in layout.predecessor_edge_keys
            )
    return tuple(items)


def _networkx_internal_mapping_state_signature(graph: Any) -> tuple[Any, ...]:
    """Sign identity and own state of every structural mapping factory output."""

    return tuple(
        (label, structural_object_state_signature(value))
        for label, value in _networkx_internal_mapping_items(graph)
    )


def _graph_transaction_protected_values(
    graph: Any,
    *,
    excluded: tuple[Any, ...] = (),
    _layout: _NetworkXRuntimeLayout | None = None,
) -> tuple[Any, ...]:
    """Return graph-owned identities already covered by the transaction.

    Callback-state traversal treats these objects as opaque references.  The
    configured callback itself can be excluded by callers that need to inspect
    its owned state while keeping aliases back into the live graph opaque.
    """

    layout = _networkx_runtime_layout(graph) if _layout is None else _layout
    namespace = _runtime_instance_namespace(graph)
    if namespace is None:
        raise TNFRValueError("graph instance has no restorable namespace")
    graph_mapping = layout.graph_mapping
    nodes = tuple(node for node, _data in layout.node_data)
    raw_edges = layout.edges
    values: list[Any] = [graph, graph_mapping, *namespace.values(), *nodes]
    for slot in _runtime_slot_descriptors(graph):
        try:
            values.append(_read_runtime_slot(graph, slot))
        except AttributeError:
            continue
    values.extend(value for _name, value in _graph_factory_items(graph))
    values.extend(
        value
        for _label, value in _networkx_internal_mapping_items(
            graph,
            layout=layout,
        )
    )
    values.extend(value for _node, value in layout.node_data)
    if layout.multigraph:
        values.extend(key for _left, _right, key, _data in raw_edges)
    values.extend(edge[-1] for edge in raw_edges)
    values.extend(
        value
        for key, value in _runtime_mapping_items(graph_mapping)
        if _is_runtime_graph_key(key)
    )
    excluded_ids = {id(value) for value in excluded}
    unique: dict[int, Any] = {}
    for value in values:
        identity = id(value)
        if identity not in excluded_ids:
            unique.setdefault(identity, value)
    return tuple(unique.values())


def _atomic_immutable_runtime_value(value: Any) -> bool:
    """Recognize exact immutable atoms with no hidden mutable referent."""

    kind = type(value)
    if value is None or kind in _ATOMIC_IMMUTABLE_RUNTIME_TYPES:
        if kind in (datetime, time):
            tzinfo = value.tzinfo
            return tzinfo is None or type(tzinfo) in (timezone, ZoneInfo)
        return True
    if kind in _NUMPY_SCALAR_RUNTIME_TYPES:
        return True
    if np is not None and np.dtype in _runtime_class_mro(kind):
        return True
    return False


def _known_immutable_runtime_value(value: Any) -> bool:
    """Recognize values whose identity can safely span a rollback boundary."""

    if _atomic_immutable_runtime_value(value):
        return True
    if type(value) is tuple:
        return all(
            _known_immutable_runtime_value(item)
            for item in tuple.__iter__(value)
        )
    if type(value) is frozenset:
        return all(
            _known_immutable_runtime_value(item)
            for item in frozenset.__iter__(value)
        )
    kind = type(value)
    return (
        kind in _IMMUTABLE_CALLABLE_TYPES
        or kind in _IMMUTABLE_DESCRIPTOR_TYPES
        or kind is object
        or type in _runtime_class_mro(kind)
    )


def _runtime_value_has_stable_rollback_identity(value: Any) -> bool:
    """Recognize immutable values and exact external-resource aggregates."""

    if _known_immutable_runtime_value(value):
        return True
    if type(value) in _EXTERNAL_RUNTIME_RESOURCE_TYPES:
        return True
    if type(value) is tuple:
        return all(
            _runtime_value_has_stable_rollback_identity(item)
            for item in tuple.__iter__(value)
        )
    if type(value) is frozenset:
        return all(
            _runtime_value_has_stable_rollback_identity(item)
            for item in frozenset.__iter__(value)
        )
    return False


def _mapping_proxy_member_is_immutable(
    value: Any,
    seen: set[int],
) -> bool:
    """Recognize a proxy-exposed value with no mutable reachable state."""

    if _atomic_immutable_runtime_value(value) or type(value) is object:
        return True
    identity = id(value)
    if identity in seen:
        return True
    seen.add(identity)
    if type(value) is tuple:
        return all(
            _mapping_proxy_member_is_immutable(item, seen)
            for item in tuple.__iter__(value)
        )
    if type(value) is frozenset:
        return all(
            _mapping_proxy_member_is_immutable(item, seen)
            for item in frozenset.__iter__(value)
        )
    if type(value) is MappingProxyType:
        try:
            _validate_runtime_mapping_proxy_references(value, seen=seen)
        except TNFRValueError:
            return False
        return True
    return False


def _validate_runtime_mapping_proxy_references(
    value: MappingProxyType,
    *,
    seen: set[int] | None = None,
) -> None:
    """Accept only exact-dict proxies exposing transitively immutable values."""

    referents = get_referents(value)
    if len(referents) != 1 or type(referents[0]) is not dict:
        raise TNFRValueError(
            "MappingProxyType graph metadata must wrap an exact built-in dict"
        )
    visited = set() if seen is None else seen
    mapping = referents[0]
    for key, item in dict.items(mapping):
        if not _mapping_proxy_member_is_immutable(
            key,
            visited,
        ) or not _mapping_proxy_member_is_immutable(item, visited):
            raise TNFRValueError(
                "MappingProxyType graph metadata exposes mutable referents and "
                "cannot be snapshotted atomically"
            )


def _runtime_identity_key_has_owned_state(key: Any) -> bool:
    """Return whether a structural key can carry mutable instance state."""

    return (
        _runtime_instance_namespace_descriptor(key) is not None
        or bool(_runtime_slot_descriptors(key))
    )


def _validate_runtime_identity_key(key: Any, *, label: str) -> None:
    """Reject mutable structural keys whose hash can drift during a stage.

    NetworkX stores node identifiers and multigraph edge keys directly in
    dictionaries.  A mutable key is rollback-safe only when equality and hash
    retain object-identity semantics while its owned namespace and slots are
    restored in place.
    """

    if _known_immutable_runtime_value(key):
        return
    if (
        _runtime_special_method(type(key), "__hash__")
        is _OBJECT_HASH_DESCRIPTOR
        and _runtime_special_method(type(key), "__eq__")
        is _OBJECT_EQUAL_DESCRIPTOR
    ):
        return
    raise TNFRValueError(
        f"mutable {label} must use object-identity hash and equality for "
        "atomic rollback"
    )


def _runtime_mapping_items(value: MutableMapping[Any, Any]) -> tuple[Any, ...]:
    """Materialize a supported mapping without invoking subclass overrides."""

    owners = _runtime_class_mro(type(value))
    if OrderedDict in owners:
        return tuple(OrderedDict.items(value))
    if dict in owners:
        return tuple(dict.items(value))
    if WeakValueDictionary in owners:
        return tuple(WeakValueDictionary.items(value))
    raise TNFRValueError(
        "runtime mapping implementation cannot be restored without user code"
    )


def _replace_runtime_mapping(
    value: MutableMapping[Any, Any],
    items: Any,
) -> None:
    """Replace mapping entries through a known base implementation."""

    owners = _runtime_class_mro(type(value))
    if OrderedDict in owners:
        OrderedDict.clear(value)
        for key, item in items:
            OrderedDict.__setitem__(value, key, item)
        return
    if dict in owners:
        dict.clear(value)
        for key, item in items:
            dict.__setitem__(value, key, item)
        return
    if WeakValueDictionary in owners:
        WeakValueDictionary.clear(value)
        for key, item in items:
            WeakValueDictionary.__setitem__(value, key, item)
        return
    raise TNFRValueError(
        "runtime mapping implementation cannot be restored without user code"
    )


def _set_runtime_mapping_item(
    value: MutableMapping[Any, Any],
    key: Any,
    item: Any,
) -> None:
    """Set one entry through a known mapping base implementation."""

    owners = _runtime_class_mro(type(value))
    if OrderedDict in owners:
        OrderedDict.__setitem__(value, key, item)
        return
    if dict in owners:
        dict.__setitem__(value, key, item)
        return
    if WeakValueDictionary in owners:
        WeakValueDictionary.__setitem__(value, key, item)
        return
    raise TNFRValueError(
        "runtime mapping implementation cannot be restored without user code"
    )


def _runtime_mapping_known_value(
    value: MutableMapping[Any, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Read one known string key without invoking mapping overrides."""

    for candidate, item in _runtime_mapping_items(value):
        if type(candidate) is str and candidate == key:
            return item
    return default


def _runtime_stored_attribute(value: Any, name: str) -> Any:
    """Read an instance-stored value without invoking attribute overrides."""

    for owner in _runtime_class_mro(type(value)):
        descriptor = _runtime_class_namespace(owner).get(name)
        if type(descriptor) in _IMMUTABLE_DESCRIPTOR_TYPES:
            return _runtime_descriptor_get(descriptor, value, type(value))
    namespace = _runtime_instance_namespace(value)
    if namespace is not None and name in namespace:
        return dict.__getitem__(namespace, name)
    raise TNFRValueError(f"runtime object has no stored {name!r} attribute")


def _write_runtime_stored_attribute(value: Any, name: str, state: Any) -> None:
    """Write instance storage without invoking ``__setattr__`` overrides."""

    for owner in _runtime_class_mro(type(value)):
        descriptor = _runtime_class_namespace(owner).get(name)
        if type(descriptor) in _IMMUTABLE_DESCRIPTOR_TYPES:
            type(descriptor).__set__(descriptor, value, state)
            return
    namespace = _runtime_instance_namespace(value)
    if namespace is None:
        raise TNFRValueError(f"runtime object cannot store {name!r}")
    dict.__setitem__(namespace, name, state)


def _delete_runtime_stored_attribute(value: Any, name: str) -> None:
    """Delete instance storage without invoking ``__delattr__`` overrides."""

    for owner in _runtime_class_mro(type(value)):
        descriptor = _runtime_class_namespace(owner).get(name)
        if type(descriptor) in _IMMUTABLE_DESCRIPTOR_TYPES:
            try:
                type(descriptor).__delete__(descriptor, value)
            except AttributeError:
                pass
            return
    namespace = _runtime_instance_namespace(value)
    if namespace is not None:
        dict.pop(namespace, name, None)


def _runtime_mapping_identity_value(
    items: tuple[tuple[Any, Any], ...],
    key: Any,
    *,
    label: str,
) -> Any:
    """Resolve a structural mapping entry strictly by key identity."""

    for candidate, value in items:
        if candidate is key:
            return value
    raise TNFRValueError(f"{label} is inconsistent with node identity support")


def _require_runtime_mapping(value: Any, *, label: str) -> MutableMapping[Any, Any]:
    """Validate one factory-produced mapping before transaction writes."""

    if not _runtime_derives_from(
        value,
        dict,
        OrderedDict,
        WeakValueDictionary,
    ):
        raise TNFRValueError(f"{label} is not a mutable mapping")
    _runtime_mapping_items(value)
    return value


def _graph_kind_flags(graph: Any) -> tuple[bool, bool]:
    """Return ``(directed, multigraph)`` from the class MRO without dispatch."""

    mro = _runtime_class_mro(type(graph))
    return (
        nx.DiGraph in mro or nx.MultiDiGraph in mro,
        nx.MultiGraph in mro or nx.MultiDiGraph in mro,
    )


def _networkx_runtime_layout(graph: Any) -> _NetworkXRuntimeLayout:
    """Read NetworkX topology through stored mappings and base primitives."""

    directed, multigraph = _graph_kind_flags(graph)
    graph_mapping = _require_runtime_mapping(
        _runtime_stored_attribute(graph, "graph"),
        label="graph attribute storage",
    )
    node_outer = _require_runtime_mapping(
        _runtime_stored_attribute(graph, "_node"),
        label="node storage",
    )
    adjacency_outer = _require_runtime_mapping(
        _runtime_stored_attribute(graph, "_adj"),
        label="adjacency storage",
    )
    predecessor_outer = (
        _require_runtime_mapping(
            _runtime_stored_attribute(graph, "_pred"),
            label="predecessor storage",
        )
        if directed
        else None
    )
    node_data = tuple(
        (
            node,
            _require_runtime_mapping(data, label="node attribute storage"),
        )
        for node, data in _runtime_mapping_items(node_outer)
    )
    nodes = tuple(node for node, _data in node_data)
    adjacency_outer_items = _runtime_mapping_items(adjacency_outer)
    adjacency_inner = tuple(
        (
            node,
            _require_runtime_mapping(
                _runtime_mapping_identity_value(
                    adjacency_outer_items,
                    node,
                    label="adjacency storage",
                ),
                label="inner adjacency storage",
            ),
        )
        for node in nodes
    )
    predecessor_inner: tuple[tuple[Any, MutableMapping[Any, Any]], ...]
    if predecessor_outer is None:
        predecessor_inner = ()
    else:
        predecessor_outer_items = _runtime_mapping_items(predecessor_outer)
        predecessor_inner = tuple(
            (
                node,
                _require_runtime_mapping(
                    _runtime_mapping_identity_value(
                        predecessor_outer_items,
                        node,
                        label="predecessor storage",
                    ),
                    label="inner predecessor storage",
                ),
            )
            for node in nodes
        )

    adjacency_edge_keys: list[
        tuple[Any, Any, MutableMapping[Any, Any]]
    ] = []
    predecessor_edge_keys: list[
        tuple[Any, Any, MutableMapping[Any, Any]]
    ] = []
    edges: list[Any] = []
    seen_undirected: set[tuple[int, int]] = set()
    for node, neighbors in adjacency_inner:
        for neighbor, edge_storage in _runtime_mapping_items(neighbors):
            if multigraph:
                edge_key_mapping = _require_runtime_mapping(
                    edge_storage,
                    label="edge-key storage",
                )
                adjacency_edge_keys.append((node, neighbor, edge_key_mapping))
            edge_token = (min(id(node), id(neighbor)), max(id(node), id(neighbor)))
            if not directed and edge_token in seen_undirected:
                continue
            if not directed:
                seen_undirected.add(edge_token)
            if multigraph:
                for key, data in _runtime_mapping_items(edge_key_mapping):
                    edges.append(
                        (
                            node,
                            neighbor,
                            key,
                            _require_runtime_mapping(
                                data,
                                label="edge attribute storage",
                            ),
                        )
                    )
            else:
                edges.append(
                    (
                        node,
                        neighbor,
                        _require_runtime_mapping(
                            edge_storage,
                            label="edge attribute storage",
                        ),
                    )
                )
    if multigraph:
        for node, neighbors in predecessor_inner:
            for neighbor, edge_storage in _runtime_mapping_items(neighbors):
                predecessor_edge_keys.append(
                    (
                        node,
                        neighbor,
                        _require_runtime_mapping(
                            edge_storage,
                            label="predecessor edge-key storage",
                        ),
                    )
                )
    return _NetworkXRuntimeLayout(
        directed=directed,
        multigraph=multigraph,
        graph_mapping=graph_mapping,
        node_outer=node_outer,
        adjacency_outer=adjacency_outer,
        predecessor_outer=predecessor_outer,
        node_data=node_data,
        adjacency_inner=adjacency_inner,
        predecessor_inner=predecessor_inner,
        adjacency_edge_keys=tuple(adjacency_edge_keys),
        predecessor_edge_keys=tuple(predecessor_edge_keys),
        edges=tuple(edges),
    )


def _seed_runtime_tuple_member_memo(
    value: Any,
    memo: dict[int, Any],
    seen: set[int] | None = None,
) -> None:
    """Preserve mutable tuple-member aliases captured by their own snapshots."""

    if not _runtime_derives_from(value, tuple):
        return
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    for item in tuple.__iter__(value):
        if _known_immutable_runtime_value(item):
            continue
        memo.setdefault(id(item), item)
        _seed_runtime_tuple_member_memo(item, memo, seen)


def _seed_runtime_resource_memo(
    value: Any,
    memo: dict[int, Any],
    seen: set[int] | None = None,
    *,
    preserve_bound_methods: bool = True,
) -> None:
    """Keep external runtime resources by identity during state deepcopy.

    Transaction snapshots retain Python bound-method identity because
    callable-owned state is captured separately. Detached SDK data copies set
    ``preserve_bound_methods=False`` so deepcopy can rebind ordinary stored
    Python methods to copied receivers while this walk still discovers their
    nested external resources. Built-in methods keep Python's atomic deepcopy
    behavior.
    """

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    kind = type(value)
    owners = _runtime_class_mro(kind)
    if kind in _EXTERNAL_RUNTIME_RESOURCE_TYPES:
        memo[identity] = value
        return
    if kind is MappingProxyType:
        _validate_runtime_mapping_proxy_references(value)
        memo[identity] = value
        return
    if kind is CallbackSpec:
        # CallbackSpec is the engine's exact immutable callback carrier. Keep
        # it by identity so deepcopy never invokes even its trusted generated
        # NamedTuple reconstruction methods. Callable discovery separately
        # traverses its members and snapshots state owned by ``func``.
        _validate_runtime_deepcopy_protocol(value, manual_capture=True)
        memo[identity] = value
        for item in tuple.__iter__(value):
            _seed_runtime_resource_memo(
                item,
                memo,
                seen,
                preserve_bound_methods=preserve_bound_methods,
            )
        return
    if not preserve_bound_methods and kind is MethodType:
        receiver = value.__self__
        if receiver is not None and type(receiver) is not ModuleType:
            _seed_runtime_resource_memo(
                receiver,
                memo,
                seen,
                preserve_bound_methods=False,
            )
        return
    if _atomic_immutable_runtime_value(value):
        # Some C-backed immutable values (notably datetime/date/timedelta)
        # deepcopy to a value-equal replacement.  Retain the original atom so
        # graph-visible alias identity is unchanged by rollback.
        memo[identity] = value
        return
    if np is not None and np.ndarray in owners:
        if identity in memo:
            return
        _validate_runtime_ndarray_references(
            value,
            captured_reference_ids=frozenset(memo),
        )
        return
    traversed_container = False
    if any(
        base in owners
        for base in (dict, OrderedDict, defaultdict, WeakValueDictionary)
    ):
        traversed_container = True
        try:
            mapping_items = _runtime_mapping_items(value)
        except TNFRValueError:
            namespace = _runtime_instance_namespace(value)
            if namespace is None:
                raise
            _seed_runtime_resource_memo(
                namespace,
                memo,
                seen,
                preserve_bound_methods=preserve_bound_methods,
            )
            return
        for key, item in mapping_items:
            _seed_runtime_resource_memo(
                key,
                memo,
                seen,
                preserve_bound_methods=preserve_bound_methods,
            )
            _seed_runtime_resource_memo(
                item,
                memo,
                seen,
                preserve_bound_methods=preserve_bound_methods,
            )
    elif tuple in owners:
        traversed_container = True
        for item in tuple.__iter__(value):
            _seed_runtime_resource_memo(
                item,
                memo,
                seen,
                preserve_bound_methods=preserve_bound_methods,
            )
    elif list in owners:
        traversed_container = True
        for item in list.__iter__(value):
            _seed_runtime_resource_memo(
                item,
                memo,
                seen,
                preserve_bound_methods=preserve_bound_methods,
            )
    elif set in owners:
        traversed_container = True
        for item in set.__iter__(value):
            _seed_runtime_resource_memo(
                item,
                memo,
                seen,
                preserve_bound_methods=preserve_bound_methods,
            )
    elif frozenset in owners:
        traversed_container = True
        for item in frozenset.__iter__(value):
            _seed_runtime_resource_memo(
                item,
                memo,
                seen,
                preserve_bound_methods=preserve_bound_methods,
            )
    elif deque in owners:
        traversed_container = True
        for item in deque.__iter__(value):
            _seed_runtime_resource_memo(
                item,
                memo,
                seen,
                preserve_bound_methods=preserve_bound_methods,
            )
    if identity in memo:
        return
    if _known_immutable_runtime_value(value) and not (
        not preserve_bound_methods and kind in (tuple, frozenset)
    ):
        memo[identity] = value
        return
    if (
        preserve_bound_methods
        and traversed_container
        and kind in (tuple, frozenset)
        and _runtime_value_has_stable_rollback_identity(value)
    ):
        memo[identity] = value
        return
    namespace = _runtime_instance_namespace(value)
    if namespace is not None:
        _seed_runtime_resource_memo(
            namespace,
            memo,
            seen,
            preserve_bound_methods=preserve_bound_methods,
        )
    for slot in _runtime_slot_descriptors(value):
        try:
            slot_value = _read_runtime_slot(value, slot)
        except AttributeError:
            continue
        _seed_runtime_resource_memo(
            slot_value,
            memo,
            seen,
            preserve_bound_methods=preserve_bound_methods,
        )


def _validate_runtime_deepcopy_value(
    value: Any,
    *,
    opaque_ids: set[int],
    seen: set[int] | None = None,
) -> None:
    """Reject user copy hooks before any transaction-owned deep copy runs."""

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen or identity in opaque_ids:
        return
    seen.add(identity)
    if type(value) in _EXTERNAL_RUNTIME_RESOURCE_TYPES:
        return
    if _atomic_immutable_runtime_value(value):
        return
    # Validate protocol declarations before any ``isinstance`` operation.
    # CPython may consult an instance's virtual ``__class__`` attribute for
    # ``isinstance``; a hostile ``__getattribute__`` must therefore be rejected
    # while all inspection still uses raw type/namespace descriptors.
    _validate_runtime_deepcopy_protocol(value)
    if np is not None and isinstance(value, np.void):
        raise TNFRValueError(
            "NumPy void/record metadata cannot be snapshotted atomically"
        )
    if isinstance(value, BuiltinMethodType):
        try:
            receiver = _runtime_descriptor_get(
                _BUILTIN_METHOD_RECEIVER_DESCRIPTOR,
                value,
                BuiltinMethodType,
            )
        except BaseException as exc:
            raise TNFRValueError(
                "built-in callable receiver cannot be inspected safely"
            ) from exc
        # Module-owned built-ins do not own the ambient module namespace.
        if receiver is not None and not isinstance(receiver, ModuleType):
            _validate_runtime_deepcopy_value(
                receiver,
                opaque_ids=opaque_ids,
                seen=seen,
            )
        return
    if (
        _known_immutable_runtime_value(value)
    ):
        return
    if isinstance(value, MappingProxyType):
        _validate_runtime_mapping_proxy_references(value)
        return
    if np is not None and isinstance(value, np.ndarray):
        _validate_runtime_ndarray_references(value)
        return
    if isinstance(value, Mapping):
        if not isinstance(value, MutableMapping):
            raise TNFRValueError(
                "runtime mapping implementation cannot be copied without "
                "user code"
            )
        for key, item in _runtime_mapping_items(value):
            _validate_runtime_identity_key(
                key,
                label="nested mapping key",
            )
            _validate_runtime_deepcopy_value(
                key,
                opaque_ids=opaque_ids,
                seen=seen,
            )
            _validate_runtime_deepcopy_value(
                item,
                opaque_ids=opaque_ids,
                seen=seen,
            )
        if isinstance(value, defaultdict):
            _validate_runtime_deepcopy_value(
                _runtime_default_factory(value),
                opaque_ids=opaque_ids,
                seen=seen,
            )
        return
    supported_container = True
    if isinstance(value, tuple):
        members = tuple.__iter__(value)
    elif isinstance(value, list):
        members = list.__iter__(value)
    elif isinstance(value, set):
        members = set.__iter__(value)
    elif isinstance(value, frozenset):
        members = frozenset.__iter__(value)
    elif isinstance(value, deque):
        members = deque.__iter__(value)
    else:
        supported_container = False
        members = ()
    for member in members:
        if isinstance(value, (set, frozenset)):
            _validate_runtime_identity_key(
                member,
                label="nested set member",
            )
        _validate_runtime_deepcopy_value(
            member,
            opaque_ids=opaque_ids,
            seen=seen,
        )
    namespace = _runtime_instance_namespace(value)
    if namespace is not None:
        for item in namespace.values():
            _validate_runtime_deepcopy_value(
                item,
                opaque_ids=opaque_ids,
                seen=seen,
            )
    slots = _runtime_slot_descriptors(value)
    for slot in slots:
        try:
            item = _read_runtime_slot(value, slot)
        except AttributeError:
            continue
        _validate_runtime_deepcopy_value(
            item,
            opaque_ids=opaque_ids,
            seen=seen,
        )
    if (
        not supported_container
        and type(value) is not Random
        and _runtime_type_has_unmodeled_c_state(type(value))
    ):
        raise TNFRValueError(
            "runtime metadata contains opaque interpreter state that cannot "
            "be restored atomically"
        )


def _capture_runtime_deepcopy(value: Any, memo: dict[int, Any]) -> Any:
    """Deep-copy captured state only after a side-effect-free preflight."""

    _seed_runtime_resource_memo(value, memo)
    _validate_runtime_deepcopy_value(value, opaque_ids=set(memo))
    return deepcopy(value, memo)


def _prepare_runtime_copy_memo(
    key: Any, value: Any, memo: dict[int, Any]
) -> bool:
    """Seed identities that are resources rather than rollback state."""

    is_cache_manager = False
    if key == "_tnfr_cache_manager":
        from ..utils.cache import CacheManager

        if _runtime_derives_from(value, CacheManager):
            is_cache_manager = True
            for layer in getattr(value, "_layers", ()):
                memo[id(layer)] = layer
            for name in ("_storage_layer", "_graph_owner"):
                resource = getattr(value, name, None)
                if resource is not None:
                    memo[id(resource)] = resource
    namespace = _runtime_instance_namespace(value)
    _seed_runtime_resource_memo(namespace if namespace is not None else value, memo)
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
    allow_captured_object_array_references: bool = False,
) -> _RuntimeGraphValue:
    """Capture one identity-bearing runtime value or reject it before writes."""

    if type(value) in _EXTERNAL_RUNTIME_RESOURCE_TYPES:
        memo[id(value)] = value
        return _RuntimeGraphValue(
            key=key,
            value=value,
            value_type=type(value),
            container_kind=None,
            container_state=None,
            object_namespace=None,
            object_state=None,
            slot_state=(),
        )
    owners = _runtime_class_mro(type(value))
    if np is not None and np.ndarray in owners:
        _validate_runtime_ndarray_references(
            value,
            captured_reference_ids=(
                frozenset(memo)
                if allow_captured_object_array_references
                else frozenset()
            ),
        )
    is_cache_manager = _prepare_runtime_copy_memo(key, value, memo)
    container_kind: str | None = None
    container_state: Any = None
    if is_cache_manager:
        container_kind = "cache_manager"
        container_state = _capture_runtime_deepcopy(
            _runtime_mapping_items(value._storage),
            memo,
        )
    elif defaultdict in owners:
        raw_items = _runtime_mapping_items(value)
        if preserve_mapping_items:
            container_kind = "defaultdict_reference"
            container_state = (raw_items, _runtime_default_factory(value))
        else:
            _seed_runtime_resource_memo(raw_items, memo)
            container_kind = "defaultdict"
            container_state = (
                _capture_runtime_deepcopy(raw_items, memo),
                _capture_runtime_deepcopy(
                    _runtime_default_factory(value),
                    memo,
                ),
            )
    elif any(
        base in owners for base in (dict, OrderedDict, WeakValueDictionary)
    ):
        raw_items = _runtime_mapping_items(value)
        if preserve_mapping_items:
            container_kind = "mapping_reference"
            container_state = raw_items
        else:
            container_kind = "mapping"
            # Attribute mappings are memoized by identity before capture so
            # aliases can be rebuilt exactly.  Traverse their detached items
            # explicitly: otherwise the memo short-circuit hides ordinary
            # locks stored as node or edge metadata from ``deepcopy``.
            _seed_runtime_resource_memo(raw_items, memo)
            container_state = _capture_runtime_deepcopy(raw_items, memo)
    elif deque in owners:
        container_kind = "deque"
        container_state = (
            _capture_runtime_deepcopy(tuple(deque.__iter__(value)), memo),
            _runtime_deque_maxlen(value),
        )
    elif Random in owners:
        if type(value) is not Random:
            raise TNFRValueError(
                "random generator subclasses cannot be snapshotted without "
                "executing user code"
            )
        container_kind = "random"
        container_state = Random.getstate(value)
    elif np is not None and np.ndarray in owners:
        container_kind = "ndarray"
        container_state = _RuntimeNDArrayState(
            values=np.array(value, copy=True, subok=True),
            writeable=bool(value.flags.writeable),
        )
    elif list in owners:
        container_kind = "sequence"
        container_state = _capture_runtime_deepcopy(
            tuple(list.__iter__(value)),
            memo,
        )
    elif set in owners:
        container_kind = "set"
        container_state = _capture_runtime_deepcopy(
            tuple(set.__iter__(value)),
            memo,
        )
    elif tuple in owners:
        member_snapshots: list[tuple[int, _RuntimeGraphValue]] = []
        for index, item in enumerate(tuple.__iter__(value)):
            if _runtime_value_has_stable_rollback_identity(item):
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

    object_namespace, object_state, slot_state = _capture_runtime_instance_state(
        key,
        value,
        memo,
    )

    if (
        container_kind is None
        and object_state is None
        and not slot_state
        and not _known_immutable_runtime_value(value)
    ):
        if _runtime_type_has_unmodeled_c_state(type(value)):
            raise TNFRValueError(
                "runtime metadata contains opaque interpreter state that "
                "cannot be restored atomically"
            )
        raise TNFRValueError(
            f"runtime object {key!r} has unsupported mutable state; "
            "atomic rollback cannot be guaranteed"
        )

    return _RuntimeGraphValue(
        key=key,
        value=value,
        value_type=type(value),
        container_kind=container_kind,
        container_state=container_state,
        object_namespace=object_namespace,
        object_state=object_state,
        slot_state=slot_state,
    )


def _capture_runtime_instance_state(
    key: Any,
    value: Any,
    memo: dict[int, Any],
    *,
    include_callable_state: bool = False,
) -> tuple[
    dict[Any, Any] | None,
    Mapping[str, Any] | None,
    tuple[_RuntimeSlotState, ...],
]:
    """Capture only an object's namespace and concrete owner-qualified slots."""

    kind = type(value)
    if type in _runtime_class_mro(kind) or (
        kind in _IMMUTABLE_CALLABLE_TYPES and not include_callable_state
    ):
        return None, None, ()
    object_namespace = None
    object_state = None
    slot_state: list[_RuntimeSlotState] = []
    try:
        namespace = _runtime_instance_namespace(value)
        if namespace is not None:
            object_namespace = namespace
            _seed_runtime_resource_memo(namespace, memo)
            object_state = _capture_runtime_deepcopy(namespace, memo)
        for slot in _runtime_slot_descriptors(value):
            try:
                slot_value = _read_runtime_slot(value, slot)
            except AttributeError:
                present = False
                slot_value = None
            else:
                present = True
                _seed_runtime_resource_memo(slot_value, memo)
            slot_state.append(
                _RuntimeSlotState(
                    slot=slot,
                    present=present,
                    value=(
                        _capture_runtime_deepcopy(slot_value, memo)
                        if present
                        else None
                    ),
                )
            )
    except Exception as exc:
        raise TNFRValueError(
            f"runtime object {key!r} cannot be snapshotted atomically"
        ) from exc
    return object_namespace, object_state, tuple(slot_state)


def _capture_runtime_owned_state(
    key: Any,
    value: Any,
    memo: dict[int, Any],
) -> _RuntimeGraphValue:
    """Capture identity and instance-owned state, excluding container entries."""

    object_namespace, object_state, slot_state = _capture_runtime_instance_state(
        key,
        value,
        memo,
        include_callable_state=True,
    )
    return _RuntimeGraphValue(
        key=key,
        value=value,
        value_type=type(value),
        container_kind=None,
        container_state=None,
        object_namespace=object_namespace,
        object_state=object_state,
        slot_state=slot_state,
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
        return tuple.__new__(type(value), tuple(members))
    except Exception as exc:
        raise TNFRValueError(
            f"runtime tuple {type(value).__qualname__} cannot be reconstructed"
        ) from exc


def _restore_runtime_object_state(
    snapshot: _RuntimeGraphValue,
    value: Any,
    memo: dict[int, Any],
) -> None:
    """Restore only instance-owned namespace and owner-qualified slot state."""

    if type(value) is not snapshot.value_type:
        try:
            object.__setattr__(value, "__class__", snapshot.value_type)
        except (AttributeError, TypeError) as exc:
            raise TNFRValueError(
                "runtime object type changed and cannot be restored atomically"
            ) from exc
    if snapshot.object_state is not None:
        expected_namespace = snapshot.object_namespace
        namespace = _runtime_instance_namespace(value)
        if namespace is None or expected_namespace is None:
            raise TNFRValueError("runtime object namespace disappeared")
        if namespace is not expected_namespace:
            descriptor = _runtime_instance_namespace_descriptor(value)
            if descriptor is None:
                raise TNFRValueError("runtime object namespace disappeared")
            try:
                type(descriptor).__set__(descriptor, value, expected_namespace)
            except (AttributeError, TypeError) as exc:
                raise TNFRValueError(
                    "runtime object namespace changed and cannot be restored atomically"
                ) from exc
            namespace = expected_namespace
        dict.clear(namespace)
        dict.update(namespace, deepcopy(dict(snapshot.object_state), memo))
    for slot_state in snapshot.slot_state:
        if slot_state.present:
            _write_runtime_slot(
                value,
                slot_state.slot,
                deepcopy(slot_state.value, memo),
            )
        elif _runtime_slot_is_present(value, slot_state.slot):
            _delete_runtime_slot(value, slot_state.slot)


def _restore_runtime_value(
    snapshot: _RuntimeGraphValue, memo: dict[int, Any]
) -> Any:
    """Restore a runtime value, reconstructing it only when required."""

    value = memo.get(id(snapshot.value), snapshot.value)
    _prepare_runtime_copy_memo(snapshot.key, value, memo)
    state = snapshot.container_state
    if snapshot.container_kind == "mapping":
        _replace_runtime_mapping(value, deepcopy(state, memo))
    elif snapshot.container_kind == "mapping_reference":
        _replace_runtime_mapping(value, state)
    elif snapshot.container_kind == "defaultdict":
        items, default_factory = state
        _replace_runtime_mapping(value, deepcopy(items, memo))
        type(_DEFAULTDICT_FACTORY_DESCRIPTOR).__set__(
            _DEFAULTDICT_FACTORY_DESCRIPTOR,
            value,
            deepcopy(default_factory, memo),
        )
    elif snapshot.container_kind == "defaultdict_reference":
        items, default_factory = state
        _replace_runtime_mapping(value, items)
        type(_DEFAULTDICT_FACTORY_DESCRIPTOR).__set__(
            _DEFAULTDICT_FACTORY_DESCRIPTOR,
            value,
            default_factory,
        )
    elif snapshot.container_kind == "deque":
        items, maxlen = state
        if _runtime_deque_maxlen(value) != maxlen:
            raise TNFRValueError("runtime deque maxlen changed during transaction")
        deque.clear(value)
        deque.extend(value, deepcopy(items, memo))
    elif snapshot.container_kind == "ndarray":
        value = _restore_runtime_ndarray(value, state)
    elif snapshot.container_kind == "random":
        Random.setstate(value, state)
    elif snapshot.container_kind == "sequence":
        list.clear(value)
        list.extend(value, deepcopy(state, memo))
    elif snapshot.container_kind == "set":
        set.clear(value)
        set.update(value, deepcopy(state, memo))
    elif snapshot.container_kind == "cache_manager":
        _replace_runtime_mapping(value._storage, deepcopy(state, memo))
    elif snapshot.container_kind == "tuple_members":
        members = list(tuple.__iter__(value))
        member_replaced = False
        for index, member_snapshot in state:
            restored_member = _restore_runtime_value(member_snapshot, memo)
            if restored_member is not member_snapshot.value:
                members[index] = restored_member
                member_replaced = True
        if member_replaced:
            value = _rebuild_runtime_tuple(value, members)

    _restore_runtime_object_state(snapshot, value, memo)

    memo[id(snapshot.value)] = value
    return value

def _restore_mapping_order(
    mapping: MutableMapping[Any, Any],
    order: Sequence[Any],
    *,
    label: str,
) -> None:
    """Restore the exact insertion order without replacing the mapping object."""

    current_items = _runtime_mapping_items(mapping)
    if len(current_items) != len(order):
        raise TNFRValueError(
            f"cannot restore {label}: mapping membership changed unexpectedly"
        )
    current = dict(current_items)
    if any(key not in current for key in order):
        raise TNFRValueError(
            f"cannot restore {label}: mapping membership changed unexpectedly"
        )
    ordered_items = tuple((key, current[key]) for key in order)
    _replace_runtime_mapping(mapping, ordered_items)


def _seed_runtime_snapshot_resource_memo(
    snapshot: _RuntimeGraphValue,
    memo: dict[int, Any],
) -> None:
    """Seed external resources held by captured state."""

    _seed_runtime_resource_memo(snapshot.container_state, memo)
    if snapshot.object_state is not None:
        _seed_runtime_resource_memo(snapshot.object_state, memo)
    for slot_state in snapshot.slot_state:
        if slot_state.present:
            _seed_runtime_resource_memo(slot_state.value, memo)


def _runtime_callable_state_items(
    runtime_items: tuple[tuple[Any, Any], ...],
    *,
    protected_values: tuple[Any, ...],
) -> tuple[
    tuple[tuple[Any, Any], ...],
    tuple[tuple[Any, FunctionType], ...],
]:
    """Discover mutable state transitively owned by configured callables.

    Bound methods own state through their receiver.  ``partial`` additionally
    exposes a read-only function/argument binding and a mutable keyword mapping.
    All mutable objects reachable through those components are snapshotted as
    identity-bearing roots.  Graph structures already covered by the enclosing
    transaction are opaque references, which preserves aliases such as
    ``worker.graph is graph`` without recursively copying the live graph.
    """

    protected = {id(value) for value in protected_values}
    discovered: dict[int, tuple[Any, Any]] = {}
    functions: dict[int, tuple[Any, FunctionType]] = {}
    seen: set[int] = set()

    def add(value: Any, path: tuple[Any, ...]) -> None:
        identity = id(value)
        if identity not in protected:
            discovered.setdefault(identity, (path, value))

    def walk_namespace(
        namespace: dict[Any, Any],
        path: tuple[Any, ...],
    ) -> None:
        for index, (key, item) in enumerate(dict.items(namespace)):
            _validate_runtime_identity_key(
                key,
                label="runtime namespace key",
            )
            walk(key, (*path, "attribute-key", index))
            walk(item, (*path, "attribute", index))

    def walk(value: Any, path: tuple[Any, ...], *, expand: bool = False) -> None:
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if identity in protected and not expand:
            return
        kind = type(value)
        owners = _runtime_class_mro(kind)
        if kind in _EXTERNAL_RUNTIME_RESOURCE_TYPES:
            return
        if kind is MappingProxyType:
            _validate_runtime_mapping_proxy_references(value)
            return
        if kind in _IMMUTABLE_DESCRIPTOR_TYPES:
            return
        if _atomic_immutable_runtime_value(value) or type in owners:
            return
        if np is not None and np.ndarray in owners:
            add(value, path)
            _validate_runtime_ndarray_references(value)
            return
        if np is not None and np.void in owners:
            raise TNFRValueError(
                "NumPy void/record callback state cannot be snapshotted "
                "observationally"
            )
        mapping_kind = any(
            base in owners
            for base in (dict, OrderedDict, defaultdict, WeakValueDictionary)
        )
        container_kind = any(
            base in owners
            for base in (tuple, list, set, frozenset, deque)
        )
        callable_kind = (
            kind in (MethodType, BuiltinMethodType, FunctionType)
            or partial in owners
            or _runtime_special_method(kind, "__call__") is not None
        )
        supported_runtime_kind = bool(
            mapping_kind
            or container_kind
            or Random in owners
            or callable_kind
            or (np is not None and np.generic in owners)
        )
        if (
            not supported_runtime_kind
            and _runtime_type_has_unmodeled_c_state(kind)
        ):
            if _runtime_special_method(kind, "__deepcopy__") is not None:
                raise TNFRValueError(
                    "runtime metadata with a custom __deepcopy__ hook cannot "
                    "be snapshotted observationally or restored atomically; "
                    "runtime object cannot be snapshotted atomically"
                )
            raise TNFRValueError(
                "runtime metadata has unsupported mutable state in opaque "
                "interpreter state that cannot be restored atomically"
            )
        _validate_runtime_deepcopy_protocol(value, manual_capture=True)
        if kind is MethodType:
            add(value, path)
            walk(value.__func__, (*path, "function"))
            walk(value.__self__, (*path, "receiver"))
            return
        if kind is BuiltinMethodType:
            receiver = _runtime_descriptor_get(
                _BUILTIN_METHOD_RECEIVER_DESCRIPTOR,
                value,
                BuiltinMethodType,
            )
            if receiver is not None and type(receiver) is not ModuleType:
                add(value, path)
                walk(receiver, (*path, "receiver"))
            return
        if partial in owners:
            if kind is not partial:
                raise TNFRValueError(
                    "partial subclasses cannot be inspected without user code"
                )
            add(value, path)
            walk(value.func, (*path, "function"))
            for index, item in enumerate(value.args):
                walk(item, (*path, "argument", index))
            walk(value.keywords, (*path, "keywords"))
            namespace = _runtime_instance_namespace(value)
            if namespace is not None:
                walk_namespace(namespace, path)
            return
        if kind is FunctionType:
            functions.setdefault(identity, (path, value))
            add(value, path)
            namespace = _runtime_instance_namespace(value)
            if namespace is not None:
                walk_namespace(namespace, path)
            defaults = value.__defaults__
            if defaults is not None:
                for index, item in enumerate(defaults):
                    walk(item, (*path, "default", index))
            keyword_defaults = value.__kwdefaults__
            if keyword_defaults is not None:
                walk(keyword_defaults, (*path, "keyword-defaults"))
            walk(value.__annotations__, (*path, "annotations"))
            if _FUNCTION_TYPE_PARAMETERS_DESCRIPTOR is not None:
                type_parameters = _runtime_descriptor_get(
                    _FUNCTION_TYPE_PARAMETERS_DESCRIPTOR,
                    value,
                    FunctionType,
                )
                walk(type_parameters, (*path, "type-parameters"))
            closure = value.__closure__
            if closure is not None:
                for index, cell in enumerate(closure):
                    try:
                        cell_value = cell.cell_contents
                    except ValueError:
                        continue
                    walk(cell_value, (*path, "closure", index))
            return
        if Random in owners:
            if kind is not Random:
                raise TNFRValueError(
                    "random generator subclasses cannot be inspected without "
                    "executing user code"
                )
            add(value, path)
            namespace = _runtime_instance_namespace(value)
            if namespace is not None:
                walk_namespace(namespace, path)
            return
        if mapping_kind:
            add(value, path)
            for index, (key, item) in enumerate(_runtime_mapping_items(value)):
                _validate_runtime_identity_key(
                    key,
                    label="nested mapping key",
                )
                walk(key, (*path, "mapping-key", index))
                walk(item, (*path, "mapping-value", index))
            if defaultdict in owners:
                walk(
                    _runtime_default_factory(value),
                    (*path, "default-factory"),
                )
            return
        if tuple in owners:
            for index, item in enumerate(tuple.__iter__(value)):
                walk(item, (*path, "immutable-item", index))
            return
        if frozenset in owners:
            for index, item in enumerate(frozenset.__iter__(value)):
                _validate_runtime_identity_key(
                    item,
                    label="nested frozenset member",
                )
                walk(item, (*path, "immutable-item", index))
            return
        if list in owners:
            add(value, path)
            for index, item in enumerate(list.__iter__(value)):
                walk(item, (*path, "container-item", index))
            return
        if set in owners:
            add(value, path)
            for index, item in enumerate(set.__iter__(value)):
                _validate_runtime_identity_key(
                    item,
                    label="nested set member",
                )
                walk(item, (*path, "container-item", index))
            return
        if deque in owners:
            add(value, path)
            for index, item in enumerate(deque.__iter__(value)):
                walk(item, (*path, "container-item", index))
            return

        add(value, path)
        namespace = _runtime_instance_namespace(value)
        if namespace is not None:
            walk_namespace(namespace, path)
        for slot in _runtime_slot_descriptors(value):
            try:
                slot_value = _read_runtime_slot(value, slot)
            except AttributeError:
                continue
            walk(
                slot_value,
                (
                    *path,
                    "slot",
                    _runtime_class_text(slot.owner, "__module__"),
                    _runtime_class_text(slot.owner, "__qualname__"),
                    slot.declared_name,
                ),
            )

    for key, value in runtime_items:
        is_callback_registry = bool(
            type(key) is tuple
            and key
            and key[0] == "graph-callback-registry"
        )
        kind = type(value)
        owners = _runtime_class_mro(kind)
        is_callable = (
            kind in (MethodType, BuiltinMethodType, FunctionType)
            or partial in owners
            or (
                type not in owners
                and _runtime_special_method(kind, "__call__") is not None
            )
        )
        if is_callable or is_callback_registry:
            root_kind = (
                "graph-callback-registry"
                if is_callback_registry
                else "runtime-callable"
            )
            walk(value, (root_kind, key), expand=True)
    return tuple(discovered.values()), tuple(functions.values())


def _discover_runtime_manual_state_items(
    search_roots: tuple[tuple[Any, Any], ...],
    *,
    protected_values: tuple[Any, ...],
) -> tuple[tuple[Any, Any], ...]:
    """Find custom state that ordinary metadata must not pass to deepcopy.

    Built-in containers remain owned by the enclosing graph snapshot. Custom
    instances and callables are promoted to manually captured runtime roots so
    deepcopy sees only their identity and never dispatches copy, reconstruction,
    or attribute hooks. This avoids duplicate ownership of graph containers
    while sealing custom state transitively.
    """

    protected = {id(value) for value in protected_values}
    discovered: dict[int, tuple[Any, Any]] = {}
    seen: set[int] = set()

    def walk(value: Any, path: tuple[Any, ...], *, root: bool = False) -> None:
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if identity in protected and not root:
            return
        kind = type(value)
        owners = _runtime_class_mro(kind)
        if _atomic_immutable_runtime_value(value):
            return
        if (
            kind in _EXTERNAL_RUNTIME_RESOURCE_TYPES
            or kind in _IMMUTABLE_DESCRIPTOR_TYPES
            or type in owners
        ):
            return
        if kind is MappingProxyType:
            _validate_runtime_mapping_proxy_references(value)
            return
        if np is not None and np.ndarray in owners:
            _validate_runtime_ndarray_references(value)
            return
        if np is not None and np.void in owners:
            raise TNFRValueError(
                "NumPy void/record metadata cannot be snapshotted atomically"
            )
        if kind is BuiltinMethodType:
            receiver = _runtime_descriptor_get(
                _BUILTIN_METHOD_RECEIVER_DESCRIPTOR,
                value,
                BuiltinMethodType,
            )
            if receiver is None or type(receiver) is ModuleType:
                return
            _validate_runtime_deepcopy_protocol(value, manual_capture=True)
            discovered.setdefault(
                identity,
                (
                    (
                        "graph-callback-registry",
                        "graph-owned-state",
                        path,
                    ),
                    value,
                ),
            )
            return
        is_callable = (
            kind in (MethodType, FunctionType)
            or partial in owners
            or _runtime_special_method(kind, "__call__") is not None
        )
        if is_callable:
            _validate_runtime_deepcopy_protocol(value, manual_capture=True)
            discovered.setdefault(
                identity,
                (
                    (
                        "graph-callback-registry",
                        "graph-owned-state",
                        path,
                    ),
                    value,
                ),
            )
            return
        if any(
            base in owners
            for base in (dict, OrderedDict, defaultdict, WeakValueDictionary)
        ):
            if kind not in (dict, OrderedDict, defaultdict, WeakValueDictionary):
                _validate_runtime_deepcopy_protocol(
                    value,
                    manual_capture=True,
                )
                discovered.setdefault(
                    identity,
                    (
                        (
                            "graph-callback-registry",
                            "graph-owned-state",
                            path,
                        ),
                        value,
                    ),
                )
                return
            _validate_runtime_deepcopy_protocol(value)
            for index, (key, item) in enumerate(_runtime_mapping_items(value)):
                walk(key, (*path, "mapping-key", index))
                walk(item, (*path, "mapping-value", index))
            if defaultdict in owners:
                walk(
                    _runtime_default_factory(value),
                    (*path, "default-factory"),
                )
            return
        _validate_runtime_deepcopy_protocol(value, manual_capture=True)
        builtin_container_bases = (tuple, list, set, frozenset, deque)
        if (
            any(base in owners for base in builtin_container_bases)
            and kind not in builtin_container_bases
        ):
            discovered.setdefault(
                identity,
                (
                    (
                        "graph-callback-registry",
                        "graph-owned-state",
                        path,
                    ),
                    value,
                ),
            )
            return
        if tuple in owners:
            members = tuple.__iter__(value)
        elif list in owners:
            members = list.__iter__(value)
        elif set in owners:
            members = set.__iter__(value)
        elif frozenset in owners:
            members = frozenset.__iter__(value)
        elif deque in owners:
            members = deque.__iter__(value)
        else:
            discovered.setdefault(
                identity,
                (
                    (
                        "graph-callback-registry",
                        "graph-owned-state",
                        path,
                    ),
                    value,
                ),
            )
            return
        for index, item in enumerate(members):
            walk(item, (*path, "container-item", index))

    for path, value in search_roots:
        walk(value, ("graph-owned-state", path), root=True)
    return tuple(discovered.values())


def _capture_runtime_function_state(
    function: FunctionType,
) -> _RuntimeFunctionState:
    """Capture defaults and closure bindings without copying opaque resources."""

    namespace = _runtime_instance_namespace(function)
    if namespace is None:
        raise TNFRValueError("runtime function namespace is unavailable")
    closure = function.__closure__
    cells: list[_RuntimeClosureCellState] = []
    if closure is not None:
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:
                cells.append(
                    _RuntimeClosureCellState(cell=cell, present=False, value=None)
                )
            else:
                cells.append(
                    _RuntimeClosureCellState(cell=cell, present=True, value=value)
                )
    return _RuntimeFunctionState(
        function=function,
        namespace=namespace,
        namespace_items=tuple(dict.items(namespace)),
        annotations=function.__annotations__,
        code=function.__code__,
        defaults=function.__defaults__,
        documentation=function.__doc__,
        keyword_defaults=function.__kwdefaults__,
        module=function.__module__,
        name=function.__name__,
        qualified_name=function.__qualname__,
        type_parameters=(
            _MISSING_RUNTIME_BINDING
            if _FUNCTION_TYPE_PARAMETERS_DESCRIPTOR is None
            else _runtime_descriptor_get(
                _FUNCTION_TYPE_PARAMETERS_DESCRIPTOR,
                function,
                FunctionType,
            )
        ),
        closure=closure,
        closure_cells=tuple(cells),
    )


def _restore_runtime_function_state(snapshot: _RuntimeFunctionState) -> None:
    """Restore one function's mutable bindings and closure cell contents."""

    function = snapshot.function
    namespace = _runtime_instance_namespace(function)
    if namespace is not snapshot.namespace:
        raise TNFRValueError("runtime function namespace identity changed")
    _replace_runtime_mapping(namespace, snapshot.namespace_items)
    function.__annotations__ = snapshot.annotations
    function.__code__ = snapshot.code
    function.__defaults__ = snapshot.defaults
    function.__doc__ = snapshot.documentation
    function.__kwdefaults__ = snapshot.keyword_defaults
    function.__module__ = snapshot.module
    function.__name__ = snapshot.name
    function.__qualname__ = snapshot.qualified_name
    if snapshot.type_parameters is not _MISSING_RUNTIME_BINDING:
        type(_FUNCTION_TYPE_PARAMETERS_DESCRIPTOR).__set__(
            _FUNCTION_TYPE_PARAMETERS_DESCRIPTOR,
            function,
            snapshot.type_parameters,
        )
    current_closure = function.__closure__
    if (
        (current_closure is None) is not (snapshot.closure is None)
        or current_closure is not None
        and snapshot.closure is not None
        and (
            len(current_closure) != len(snapshot.closure)
            or any(
                current is not expected
                for current, expected in zip(
                    current_closure,
                    snapshot.closure,
                    strict=True,
                )
            )
        )
    ):
        raise TNFRValueError("runtime function closure identity changed")
    for cell_state in snapshot.closure_cells:
        if cell_state.present:
            cell_state.cell.cell_contents = cell_state.value
        else:
            try:
                del cell_state.cell.cell_contents
            except ValueError:
                pass


class GraphTransactionSnapshot:
    """Capture and restore graph-owned state and runtime references.

    Mutable runtime containers retain identity when they permit in-place
    restoration. If an ndarray refuses the required shape or dtype repair,
    graph-visible aliases are rebound to a detached captured value; references
    held only by external code remain outside the graph transaction.
    """

    def __init__(self, graph: Any) -> None:
        # Retain the concrete owner so a snapshot can never be supplied to a
        # different, merely value-compatible graph. The strong reference also
        # prevents process-local identity reuse for the snapshot lifetime.
        self._graph = graph
        self._graph_type = type(graph)
        layout = _networkx_runtime_layout(graph)
        self._nodes = tuple(node for node, _data in layout.node_data)
        self._directed = layout.directed
        self._multigraph = layout.multigraph
        graph_namespace = _runtime_instance_namespace(graph)
        if graph_namespace is None:
            raise TNFRValueError("graph instance has no restorable namespace")
        self._graph_namespace = graph_namespace
        graph_mapping = layout.graph_mapping
        node_outer_mapping = layout.node_outer
        adjacency_outer_mapping = layout.adjacency_outer
        predecessor_outer_mapping = layout.predecessor_outer
        graph_mapping_items = _runtime_mapping_items(graph_mapping)
        runtime_items = tuple(
            (key, value)
            for key, value in graph_mapping_items
            if _is_runtime_graph_key(key)
        )
        epi_history_object_referents = _preflight_runtime_epi_history(
            runtime_items
        )
        epi_history_object_array_ids = frozenset(
            array_id for array_id, _item in epi_history_object_referents
        )
        graph_attribute_items = tuple(
            (key, value)
            for key, value in graph_namespace.items()
            if key not in _NETWORKX_STRUCTURAL_STORAGE_ATTRIBUTES
        )
        graph_slot_descriptors = tuple(
            slot
            for slot in _runtime_slot_descriptors(graph)
            if slot.storage_name not in _NETWORKX_STRUCTURAL_STORAGE_ATTRIBUTES
        )
        graph_slot_items_list: list[tuple[_RuntimeSlotDescriptor, Any]] = []
        for slot in graph_slot_descriptors:
            try:
                value = _read_runtime_slot(graph, slot)
            except AttributeError:
                continue
            graph_slot_items_list.append((slot, value))
        graph_slot_items = tuple(graph_slot_items_list)
        graph_factory_items = _graph_factory_items(graph)
        node_mapping_items = layout.node_data
        raw_edges = layout.edges
        identity_key_items: list[tuple[tuple[str, int], Any]] = [
            (("node-identity", index), node)
            for index, node in enumerate(self._nodes)
        ]
        if self._multigraph:
            identity_key_items.extend(
                (("edge-key-identity", index), key)
                for index, (_left, _right, key, _data) in enumerate(raw_edges)
            )
        identity_key_items.extend(
            (("graph-attribute-key", index), key)
            for index, (key, _value) in enumerate(graph_mapping_items)
        )
        metadata_mappings = (
            *(data for _node, data in node_mapping_items),
            *(edge[-1] for edge in raw_edges),
        )
        metadata_key_index = 0
        for mapping in metadata_mappings:
            for key, _value in _runtime_mapping_items(mapping):
                identity_key_items.append(
                    (("attribute-key", metadata_key_index), key)
                )
                metadata_key_index += 1
        unique_identity_keys: dict[int, tuple[tuple[str, int], Any]] = {}
        for label, key in identity_key_items:
            _validate_runtime_identity_key(key, label=label[0])
            if _runtime_identity_key_has_owned_state(key):
                unique_identity_keys.setdefault(id(key), (label, key))
        internal_mapping_items = _networkx_internal_mapping_items(
            graph,
            layout=layout,
        )
        for _label, mapping in internal_mapping_items:
            _runtime_mapping_items(mapping)
        protected_values = _graph_transaction_protected_values(
            graph,
            _layout=layout,
        )
        callable_search_roots: list[tuple[Any, Any]] = [
            (("graph-metadata", index), value)
            for index, (_key, value) in enumerate(graph_mapping_items)
            if id(value) not in epi_history_object_array_ids
        ]
        callable_search_roots.extend(
            (("epi-history-object", index), value)
            for index, (_array_id, value) in enumerate(
                epi_history_object_referents
            )
        )
        callable_search_roots.extend(
            (("graph-attribute", index), value)
            for index, (_key, value) in enumerate(graph_attribute_items)
            if _key not in _NETWORKX_GRAPH_INTERNAL_ATTRIBUTES
        )
        callable_search_roots.extend(
            (("graph-slot", index), value)
            for index, (_slot, value) in enumerate(graph_slot_items)
            if _slot.storage_name not in _NETWORKX_GRAPH_INTERNAL_ATTRIBUTES
        )
        callable_search_roots.extend(
            (("graph-factory", index), value)
            for index, (_name, value) in enumerate(graph_factory_items)
        )
        callable_search_roots.extend(
            (("node-metadata", node_index, value_index), value)
            for node_index, (_node, data) in enumerate(node_mapping_items)
            for value_index, (_key, value) in enumerate(
                _runtime_mapping_items(data)
            )
        )
        callable_search_roots.extend(
            (("edge-metadata", edge_index, value_index), value)
            for edge_index, edge in enumerate(raw_edges)
            for value_index, (_key, value) in enumerate(
                _runtime_mapping_items(edge[-1])
            )
        )
        reachable_manual_state_items = _discover_runtime_manual_state_items(
            tuple(callable_search_roots),
            protected_values=protected_values,
        )
        configured_callable_items = (
            *runtime_items,
            *(
                (("graph-callback-registry", index), value)
                for index, (key, value) in enumerate(graph_mapping_items)
                if type(key) is str and key == "callbacks"
            ),
            *(
                (("graph-factory", name), value)
                for name, value in graph_factory_items
            ),
            *reachable_manual_state_items,
        )
        callable_state_items, callable_functions = _runtime_callable_state_items(
            configured_callable_items,
            protected_values=protected_values,
        )
        self._node_outer_mapping = node_outer_mapping
        self._adjacency_outer_mapping = adjacency_outer_mapping
        self._predecessor_outer_mapping = (
            predecessor_outer_mapping if self._directed else None
        )
        self._adjacency_inner_mappings = layout.adjacency_inner
        self._predecessor_inner_mappings = layout.predecessor_inner
        self._adjacency_edge_key_mappings = layout.adjacency_edge_keys
        self._predecessor_edge_key_mappings = layout.predecessor_edge_keys
        runtime_memo = {id(graph): graph}
        runtime_memo.update({id(value): value for _key, value in runtime_items})
        runtime_memo.update(
            {id(value): value for _key, value in graph_attribute_items}
        )
        runtime_memo.update(
            {id(value): value for _slot, value in graph_slot_items}
        )
        runtime_memo.update(
            {id(value): value for _name, value in graph_factory_items}
        )
        runtime_memo.update(
            {id(value): value for _node, value in node_mapping_items}
        )
        runtime_memo.update(
            {id(edge[-1]): edge[-1] for edge in raw_edges}
        )
        runtime_memo.update(
            {id(value): value for _label, value in internal_mapping_items}
        )
        # Structural keys are identities owned by the graph topology. Keep
        # every one of them opaque to detached metadata copies, including
        # stateless keys: copying such a key would silently change support and
        # could invoke an arbitrary user ``__deepcopy__`` hook. Keys with
        # mutable owned state are captured separately below for in-place
        # restoration.
        runtime_memo.update(
            {id(value): value for _label, value in identity_key_items}
        )
        runtime_memo.update(
            {id(value): value for _label, value in unique_identity_keys.values()}
        )
        runtime_memo.update(
            {id(value): value for _path, value in callable_state_items}
        )
        opaque_copy_ids = set(runtime_memo)
        for key, value in graph_mapping_items:
            if not _is_runtime_graph_key(key):
                _validate_runtime_deepcopy_value(
                    value,
                    opaque_ids=opaque_copy_ids,
                )
        for _node, data in node_mapping_items:
            for _key, value in _runtime_mapping_items(data):
                _validate_runtime_deepcopy_value(
                    value,
                    opaque_ids=opaque_copy_ids,
                )
        for edge in raw_edges:
            for _key, value in _runtime_mapping_items(edge[-1]):
                _validate_runtime_deepcopy_value(
                    value,
                    opaque_ids=opaque_copy_ids,
                )
        graph_value_candidates: list[tuple[Any, Any]] = []
        seen_graph_values: set[int] = set()
        for label, value in callable_search_roots:
            identity = id(value)
            if identity in seen_graph_values or identity in runtime_memo:
                continue
            seen_graph_values.add(identity)
            if (
                _runtime_value_has_stable_rollback_identity(value)
                or type(value) in _EXTERNAL_RUNTIME_RESOURCE_TYPES
                or type(value) is MappingProxyType
            ):
                continue
            graph_value_candidates.append((label, value))
        # Preseed every direct graph-owned identity before capturing any one
        # root. This preserves cross-root aliases and prevents an early root
        # from recursively copying later roots that are captured in place.
        runtime_memo.update(
            {id(value): value for _label, value in graph_value_candidates}
        )
        graph_value_states = [
            _capture_runtime_value(
                ("graph-owned-value", label),
                value,
                runtime_memo,
            )
            for label, value in graph_value_candidates
        ]
        # Detached mapping copies retain each direct graph-owned object.
        # Rollback repairs its state in place and then restores all graph,
        # node, or edge bindings through this shared memo identity.
        self._graph_value_states = tuple(graph_value_states)
        self._graph_mapping = _capture_runtime_value(
            "graph.graph",
            graph_mapping,
            runtime_memo,
            preserve_mapping_items=True,
        )
        self._identity_key_states = tuple(
            _capture_runtime_owned_state(label, value, runtime_memo)
            for label, value in unique_identity_keys.values()
        )
        for _key, value in runtime_items:
            _seed_runtime_tuple_member_memo(value, runtime_memo)
        for _key, value in graph_attribute_items:
            _seed_runtime_tuple_member_memo(value, runtime_memo)
        for _slot, value in graph_slot_items:
            _seed_runtime_tuple_member_memo(value, runtime_memo)
        for _name, value in graph_factory_items:
            _seed_runtime_tuple_member_memo(value, runtime_memo)
        for _key, value in graph_mapping_items:
            _seed_runtime_resource_memo(value, runtime_memo)
        for _node, data in node_mapping_items:
            _seed_runtime_resource_memo(data, runtime_memo)
        for edge in raw_edges:
            _seed_runtime_resource_memo(edge[-1], runtime_memo)

        self._node_data = tuple(
            (
                node,
                _capture_runtime_value(
                    ("node-attribute-mapping", node),
                    data,
                    runtime_memo,
                ),
            )
            for node, data in node_mapping_items
        )
        self._adjacency_order = {
            node: tuple(
                neighbor
                for neighbor, _value in _runtime_mapping_items(mapping)
            )
            for node, mapping in layout.adjacency_inner
        }
        self._predecessor_order = {
            node: tuple(
                neighbor
                for neighbor, _value in _runtime_mapping_items(mapping)
            )
            for node, mapping in layout.predecessor_inner
        }
        self._adjacency_key_order = (
            {
                (node, neighbor): tuple(
                    key for key, _value in _runtime_mapping_items(mapping)
                )
                for node, neighbor, mapping in layout.adjacency_edge_keys
            }
            if self._multigraph
            else None
        )
        self._predecessor_key_order = {
            (node, neighbor): tuple(
                key for key, _value in _runtime_mapping_items(mapping)
            )
            for node, neighbor, mapping in layout.predecessor_edge_keys
        }
        if self._multigraph:
            self._edges = tuple(
                (
                    left,
                    right,
                    key,
                    _capture_runtime_value(
                        ("edge-attribute-mapping", left, right, key),
                        data,
                        runtime_memo,
                    ),
                )
                for left, right, key, data in raw_edges
            )
        else:
            self._edges = tuple(
                (
                    left,
                    right,
                    _capture_runtime_value(
                        ("edge-attribute-mapping", left, right),
                        data,
                        runtime_memo,
                    ),
                )
                for left, right, data in raw_edges
            )

        ordinary: list[tuple[Any, Any]] = []
        runtime: list[_RuntimeGraphValue] = []
        for key, value in graph_mapping_items:
            is_runtime = _is_runtime_graph_key(key)
            if not is_runtime:
                # Locks and standard loggers are external resources, not
                # serializable graph state. Preserve their identity while
                # copying surrounding metadata so they do not make canonical
                # stages unexecutable.
                _seed_runtime_resource_memo(value, runtime_memo)
                ordinary.append((key, deepcopy(value, runtime_memo)))
                continue
            runtime.append(
                _capture_runtime_value(
                    key,
                    value,
                    runtime_memo,
                    preserve_mapping_items=(
                        key in _REFERENCE_PRESERVING_RUNTIME_MAPPINGS
                        or _runtime_derives_from(value, WeakValueDictionary)
                    ),
                    allow_captured_object_array_references=(
                        id(value) in epi_history_object_array_ids
                    ),
                )
            )
        self._graph_data_order = tuple(key for key, _value in graph_mapping_items)
        self._ordinary_graph_data = tuple(ordinary)
        self._runtime_graph_data = tuple(runtime)
        self._runtime_callable_states = tuple(
            _capture_runtime_value(path, value, runtime_memo)
            for path, value in callable_state_items
        )
        self._runtime_function_states = tuple(
            _capture_runtime_function_state(function)
            for _path, function in callable_functions
        )
        self._graph_attribute_names = frozenset(
            key for key, _value in graph_attribute_items
        )
        self._graph_attributes = tuple(
            _capture_runtime_value(key, value, runtime_memo)
            for key, value in graph_attribute_items
        )
        self._graph_slot_descriptors = graph_slot_descriptors
        self._graph_slot_value_descriptor_ids = frozenset(
            id(slot.descriptor) for slot, _value in graph_slot_items
        )
        self._graph_slots = tuple(
            (
                slot,
                _capture_runtime_value(
                    (
                        "graph-slot",
                        _runtime_class_text(slot.owner, "__module__"),
                        _runtime_class_text(slot.owner, "__qualname__"),
                        slot.declared_name,
                    ),
                    value,
                    runtime_memo,
                ),
            )
            for slot, value in graph_slot_items
        )
        self._graph_factories = tuple(
            (
                name,
                _capture_runtime_owned_state(
                    ("graph-factory", name),
                    value,
                    runtime_memo,
                ),
            )
            for name, value in graph_factory_items
        )
        unique_internal_mappings: dict[int, tuple[Any, Any]] = {}
        for label, value in internal_mapping_items:
            unique_internal_mappings.setdefault(id(value), (label, value))
        self._networkx_mapping_states = tuple(
            _capture_runtime_owned_state(
                ("networkx-mapping", label),
                value,
                runtime_memo,
            )
            for label, value in unique_internal_mappings.values()
        )
        try:
            self._last_operator = _runtime_stored_attribute(
                graph,
                "_last_operator_applied",
            )
        except TNFRValueError:
            self._had_last_operator = False
            self._last_operator = None
        else:
            self._had_last_operator = True

    def _restore_networkx_structural_mappings(
        self,
        graph: Any,
        restored_node_data: tuple[tuple[Any, MutableMapping[Any, Any]], ...],
        restored_edges: tuple[Any, ...],
        runtime_memo: dict[int, Any],
    ) -> None:
        """Rebind factory-produced mappings with identity and own state intact."""

        def clear_once(values: Sequence[MutableMapping[Any, Any]]) -> None:
            seen: set[int] = set()
            for value in values:
                identity = id(value)
                if identity in seen:
                    continue
                seen.add(identity)
                _replace_runtime_mapping(value, ())

        adjacency_inner = dict(self._adjacency_inner_mappings)
        predecessor_inner = dict(self._predecessor_inner_mappings)
        adjacency_edge_keys = {
            (node, neighbor): value
            for node, neighbor, value in self._adjacency_edge_key_mappings
        }
        predecessor_edge_keys = {
            (node, neighbor): value
            for node, neighbor, value in self._predecessor_edge_key_mappings
        }

        clear_once(tuple(adjacency_inner.values()))
        clear_once(tuple(predecessor_inner.values()))
        clear_once(tuple(adjacency_edge_keys.values()))
        clear_once(tuple(predecessor_edge_keys.values()))

        if self._multigraph:
            edge_data: dict[tuple[Any, Any, Any], MutableMapping[Any, Any]] = {}
            for left, right, key, data in restored_edges:
                edge_data[(left, right, key)] = data
                if not self._directed:
                    edge_data[(right, left, key)] = data
            for (node, neighbor), edge_keys in adjacency_edge_keys.items():
                for key in self._adjacency_key_order[(node, neighbor)]:
                    _set_runtime_mapping_item(
                        edge_keys,
                        key,
                        edge_data[(node, neighbor, key)],
                    )
            for (node, neighbor), edge_keys in predecessor_edge_keys.items():
                for key in self._predecessor_key_order[(node, neighbor)]:
                    _set_runtime_mapping_item(
                        edge_keys,
                        key,
                        edge_data[(neighbor, node, key)],
                    )
            for node, neighbors in self._adjacency_order.items():
                for neighbor in neighbors:
                    _set_runtime_mapping_item(
                        adjacency_inner[node],
                        neighbor,
                        adjacency_edge_keys[(node, neighbor)],
                    )
            for node, neighbors in self._predecessor_order.items():
                for neighbor in neighbors:
                    _set_runtime_mapping_item(
                        predecessor_inner[node],
                        neighbor,
                        predecessor_edge_keys[(node, neighbor)],
                    )
        else:
            edge_data = {}
            for left, right, data in restored_edges:
                edge_data[(left, right)] = data
                if not self._directed:
                    edge_data[(right, left)] = data
            for node, neighbors in self._adjacency_order.items():
                for neighbor in neighbors:
                    _set_runtime_mapping_item(
                        adjacency_inner[node],
                        neighbor,
                        edge_data[(node, neighbor)],
                    )
            for node, neighbors in self._predecessor_order.items():
                for neighbor in neighbors:
                    _set_runtime_mapping_item(
                        predecessor_inner[node],
                        neighbor,
                        edge_data[(neighbor, node)],
                    )

        _replace_runtime_mapping(self._node_outer_mapping, restored_node_data)
        _replace_runtime_mapping(
            self._adjacency_outer_mapping,
            self._adjacency_inner_mappings,
        )
        if self._directed:
            _replace_runtime_mapping(
                self._predecessor_outer_mapping,
                self._predecessor_inner_mappings,
            )

        _write_runtime_stored_attribute(
            graph,
            "_node",
            self._node_outer_mapping,
        )
        _write_runtime_stored_attribute(
            graph,
            "_adj",
            self._adjacency_outer_mapping,
        )
        if self._directed:
            _write_runtime_stored_attribute(
                graph,
                "_succ",
                self._adjacency_outer_mapping,
            )
            _write_runtime_stored_attribute(
                graph,
                "_pred",
                self._predecessor_outer_mapping,
            )

        for snapshot in self._networkx_mapping_states:
            _restore_runtime_object_state(
                snapshot,
                snapshot.value,
                runtime_memo,
            )

    def restore(self, graph: Any) -> None:
        """Restore topology, attributes, caches and monitor bookkeeping."""

        if graph is not self._graph:
            raise TNFRValueError(
                "graph transaction snapshot belongs to a different graph"
            )
        if type(graph) is not self._graph_type:
            try:
                object.__setattr__(graph, "__class__", self._graph_type)
            except (AttributeError, TypeError) as exc:
                raise TNFRValueError(
                    "graph type changed and cannot be restored atomically"
                ) from exc
        current_graph_namespace = _runtime_instance_namespace(graph)
        if current_graph_namespace is not self._graph_namespace:
            descriptor = _runtime_instance_namespace_descriptor(graph)
            if descriptor is None:
                raise TNFRValueError("graph instance namespace disappeared")
            try:
                type(descriptor).__set__(
                    descriptor,
                    graph,
                    self._graph_namespace,
                )
            except (AttributeError, TypeError) as exc:
                raise TNFRValueError(
                    "graph namespace changed and cannot be restored atomically"
                ) from exc
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
        runtime_memo.update(
            {
                id(snapshot.value): snapshot.value
                for _slot, snapshot in self._graph_slots
            }
        )
        runtime_memo.update(
            {
                id(snapshot.value): snapshot.value
                for _name, snapshot in self._graph_factories
            }
        )
        runtime_memo.update(
            {
                id(snapshot.value): snapshot.value
                for _node, snapshot in self._node_data
            }
        )
        runtime_memo.update(
            {id(edge[-1].value): edge[-1].value for edge in self._edges}
        )
        runtime_memo.update(
            {
                id(snapshot.value): snapshot.value
                for snapshot in self._networkx_mapping_states
            }
        )
        runtime_memo.update(
            {
                id(snapshot.value): snapshot.value
                for snapshot in self._identity_key_states
            }
        )
        runtime_memo.update(
            {
                id(snapshot.value): snapshot.value
                for snapshot in self._runtime_callable_states
            }
        )
        runtime_memo.update(
            {
                id(snapshot.value): snapshot.value
                for snapshot in self._graph_value_states
            }
        )
        runtime_memo[id(self._graph_mapping.value)] = self._graph_mapping.value
        for _node, snapshot in self._node_data:
            _seed_runtime_snapshot_resource_memo(snapshot, runtime_memo)
        for edge in self._edges:
            _seed_runtime_snapshot_resource_memo(edge[-1], runtime_memo)
        for _key, value in self._ordinary_graph_data:
            _seed_runtime_resource_memo(value, runtime_memo)
        _seed_runtime_snapshot_resource_memo(self._graph_mapping, runtime_memo)
        for snapshot in self._runtime_graph_data:
            _seed_runtime_snapshot_resource_memo(snapshot, runtime_memo)
        for snapshot in self._runtime_callable_states:
            _seed_runtime_snapshot_resource_memo(snapshot, runtime_memo)
        for snapshot in self._graph_value_states:
            _seed_runtime_snapshot_resource_memo(snapshot, runtime_memo)
        for snapshot in self._graph_attributes:
            _seed_runtime_snapshot_resource_memo(snapshot, runtime_memo)
        for _slot, snapshot in self._graph_slots:
            _seed_runtime_snapshot_resource_memo(snapshot, runtime_memo)
        for _name, snapshot in self._graph_factories:
            _seed_runtime_snapshot_resource_memo(snapshot, runtime_memo)
        for snapshot in self._networkx_mapping_states:
            _seed_runtime_snapshot_resource_memo(snapshot, runtime_memo)
        for snapshot in self._identity_key_states:
            _seed_runtime_snapshot_resource_memo(snapshot, runtime_memo)

        # Structural identifiers must recover their original class and owned
        # state before any dictionary keyed by them is queried or rebuilt.
        for snapshot in self._identity_key_states:
            _restore_runtime_value(snapshot, runtime_memo)

        restored_runtime_graph_data = tuple(
            (snapshot, _restore_runtime_value(snapshot, runtime_memo))
            for snapshot in self._runtime_graph_data
        )
        for snapshot in self._runtime_function_states:
            _restore_runtime_function_state(snapshot)
        for snapshot in self._runtime_callable_states:
            _restore_runtime_value(snapshot, runtime_memo)
        for snapshot in self._graph_value_states:
            _restore_runtime_value(snapshot, runtime_memo)
        restored_graph_attributes = tuple(
            (snapshot, _restore_runtime_value(snapshot, runtime_memo))
            for snapshot in self._graph_attributes
        )
        restored_graph_slots = tuple(
            (slot, snapshot, _restore_runtime_value(snapshot, runtime_memo))
            for slot, snapshot in self._graph_slots
        )
        for _name, snapshot in self._graph_factories:
            _restore_runtime_value(snapshot, runtime_memo)
        restored_node_data = tuple(
            (node, _restore_runtime_value(snapshot, runtime_memo))
            for node, snapshot in self._node_data
        )
        if self._multigraph:
            restored_edges = tuple(
                (
                    left,
                    right,
                    key,
                    _restore_runtime_value(snapshot, runtime_memo),
                )
                for left, right, key, snapshot in self._edges
            )
        else:
            restored_edges = tuple(
                (
                    left,
                    right,
                    _restore_runtime_value(snapshot, runtime_memo),
                )
                for left, right, snapshot in self._edges
        )
        restored_graph_mapping = _restore_runtime_value(
            self._graph_mapping,
            runtime_memo,
        )

        current_custom_attributes = tuple(
            key
            for key in (_runtime_instance_namespace(graph) or {})
            if key not in _NETWORKX_STRUCTURAL_STORAGE_ATTRIBUTES
        )
        for key in current_custom_attributes:
            if key not in self._graph_attribute_names:
                _delete_runtime_stored_attribute(graph, key)
        for snapshot, restored_value in restored_graph_attributes:
            _write_runtime_stored_attribute(
                graph,
                snapshot.key,
                restored_value,
            )
        for slot in self._graph_slot_descriptors:
            if (
                id(slot.descriptor)
                not in self._graph_slot_value_descriptor_ids
                and _runtime_slot_is_present(graph, slot)
            ):
                _delete_runtime_slot(graph, slot)
        for slot, _snapshot, restored_value in restored_graph_slots:
            _write_runtime_slot(graph, slot, restored_value)

        self._restore_networkx_structural_mappings(
            graph,
            restored_node_data,
            restored_edges,
            runtime_memo,
        )

        _write_runtime_stored_attribute(graph, "graph", restored_graph_mapping)
        _replace_runtime_mapping(
            restored_graph_mapping,
            deepcopy(self._ordinary_graph_data, runtime_memo),
        )
        for snapshot, restored_value in restored_runtime_graph_data:
            _set_runtime_mapping_item(
                restored_graph_mapping,
                snapshot.key,
                restored_value,
            )
        _restore_mapping_order(
            restored_graph_mapping,
            self._graph_data_order,
            label="graph-attribute order",
        )
        _restore_runtime_object_state(
            self._graph_mapping,
            restored_graph_mapping,
            runtime_memo,
        )
        for _name, snapshot in self._graph_factories:
            _restore_runtime_value(snapshot, runtime_memo)

        if self._had_last_operator:
            _write_runtime_stored_attribute(
                graph,
                "_last_operator_applied",
                self._last_operator,
            )
        else:
            _delete_runtime_stored_attribute(graph, "_last_operator_applied")

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


def _select_graph_transaction(
    graph: Any,
    transaction_snapshot: GraphTransactionSnapshot | None,
) -> GraphTransactionSnapshot:
    """Return a fresh or owner-bound exact transaction snapshot."""

    if transaction_snapshot is None:
        return GraphTransactionSnapshot(graph)
    if type(transaction_snapshot) is not GraphTransactionSnapshot:
        raise TNFRValueError(
            "transaction_snapshot must be an exact GraphTransactionSnapshot"
        )
    if object.__getattribute__(transaction_snapshot, "_graph") is not graph:
        raise TNFRValueError(
            "graph transaction snapshot belongs to a different graph"
        )
    return transaction_snapshot


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
    reception_read_snapshot: ReceptionReadSnapshot | None = None


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


_RECEPTION_STAGE_OBSERVATION_VERSION = "reception_stage_observation_v1"
_RECEPTION_STAGE_POST_STATE_BOUNDARY = "completed_en_stage_before_result"


def _same_reception_value(left: Any, right: Any) -> bool:
    """Compare EN evidence without invoking node-identifier equality."""

    return proof_stamps_are_identical(
        structural_proof_signature(left),
        structural_proof_signature(right),
    )


def _reception_stage_observation_stamp(
    observation: "ReceptionStageObservation",
) -> tuple[Any, ...]:
    """Seal one public EN read independently of the detached graph."""

    return (
        _RECEPTION_STAGE_OBSERVATION_VERSION,
        structural_proof_signature(
            (
                observation.target_index,
                observation.node,
                observation.target_epi_before,
                observation.target_epi_after,
                observation.target_epi_kind_before,
                observation.target_epi_kind_after,
                observation.neighbors,
                observation.neighbor_epi_values,
                observation.neighbor_dominant_values,
                observation.neighbor_epi_kinds,
                observation.neighbor_epi_mean,
                observation.source_tracking_enabled,
                observation.source_max_distance,
                observation.reception_sources,
                observation.reception_sources_present_after,
                observation.reception_sources_after,
                observation.read_boundary,
                observation.post_state_boundary,
                observation.glyph,
                object.__getattribute__(
                    observation,
                    "auxiliary_stability_certified",
                ),
                object.__getattribute__(
                    observation,
                    "_read_payload_stamp",
                ),
                object.__getattribute__(
                    observation,
                    "_post_state_payload_stamp",
                ),
            )
        ),
    )


def _reception_observation_read_snapshot(
    observation: "ReceptionStageObservation",
) -> ReceptionReadSnapshot:
    """Reconstruct the published read payload without graph ownership."""

    validation_owner = object()
    return ReceptionReadSnapshot(
        node=observation.node,
        target_epi=observation.target_epi_before,
        target_epi_kind=observation.target_epi_kind_before,
        neighbors=observation.neighbors,
        neighbor_epi_values=observation.neighbor_epi_values,
        neighbor_dominant_values=observation.neighbor_dominant_values,
        neighbor_epi_kinds=observation.neighbor_epi_kinds,
        neighbor_epi_mean=observation.neighbor_epi_mean,
        source_tracking_enabled=observation.source_tracking_enabled,
        source_max_distance=observation.source_max_distance,
        reception_sources=observation.reception_sources,
        _read_graph_owner=validation_owner,
        _metric_consumer_graph_owner=validation_owner,
        _graph_identity=id(validation_owner),
        _metric_consumer_graph_identity=id(validation_owner),
        read_boundary=observation.read_boundary,
    )


def _reception_observation_post_state_stamp(
    observation: "ReceptionStageObservation",
) -> tuple[Any, ...]:
    """Sign executor-observed EN state after the complete stage."""

    return structural_proof_signature(
        (
            observation.target_index,
            observation.node,
            observation.target_epi_after,
            observation.target_epi_kind_after,
            observation.reception_sources_present_after,
            observation.reception_sources_after,
            observation.post_state_boundary,
        )
    )


def _validate_reception_stage_observation(
    observation: "ReceptionStageObservation",
) -> None:
    """Validate the value-domain and explicit scope of one EN observation."""

    if type(observation.target_index) is not int or observation.target_index < 0:
        raise ValueError("Reception observation index must be nonnegative")
    _reception_observation_read_snapshot(observation)
    if (
        type(observation.target_epi_after) is not float
        or not math.isfinite(observation.target_epi_after)
        or type(observation.target_epi_kind_after) is not str
    ):
        raise ValueError("Reception observation post-state is invalid")
    if observation.glyph is not Glyph.EN:
        raise ValueError("Reception observation glyph must be EN")
    if type(observation.reception_sources_present_after) is not bool:
        raise TypeError("Reception source-presence observation must be a bool")
    after_sources = observation.reception_sources_after
    if observation.source_tracking_enabled:
        if not observation.reception_sources_present_after or after_sources is None:
            raise ValueError("Reception tracked source state is missing")
    elif after_sources is not None:
        raise ValueError("Disabled source tracking cannot claim source contents")
    if after_sources is not None:
        if type(after_sources) is not tuple:
            raise TypeError("Reception post-state sources must be a tuple")
        for source in after_sources:
            if type(source) is not tuple or len(source) != 3:
                raise TypeError("Reception post-state source records are invalid")
            if (
                type(source[1]) is not float
                or not math.isfinite(source[1])
                or not 0.0 <= source[1] <= 1.0
                or type(source[2]) is not float
                or not math.isfinite(source[2])
                or source[2] < 0.0
            ):
                raise ValueError("Reception post-state source scores are invalid")
    if (
        observation.source_tracking_enabled
        and not _same_reception_value(
            after_sources,
            observation.reception_sources,
        )
    ):
        raise ValueError("Reception tracked sources did not survive the stage")
    if (
        observation.post_state_boundary
        != _RECEPTION_STAGE_POST_STATE_BOUNDARY
    ):
        raise ValueError("Reception post-state boundary changed")
    if object.__getattribute__(
        observation,
        "auxiliary_stability_certified",
    ) is not False:
        raise ValueError("Reception observations cannot claim stability")


@dataclass(frozen=True, slots=True)
class ReceptionStageObservation:
    """Sealed per-target EN read captured at one two-phase stage boundary."""

    target_index: int
    node: Any
    target_epi_before: float
    target_epi_after: float
    target_epi_kind_before: str
    target_epi_kind_after: str
    neighbors: tuple[Any, ...]
    neighbor_epi_values: tuple[float, ...]
    neighbor_dominant_values: tuple[float, ...]
    neighbor_epi_kinds: tuple[str, ...]
    neighbor_epi_mean: float
    source_tracking_enabled: bool
    source_max_distance: int | None
    reception_sources: tuple[tuple[Any, float, float], ...] | None
    reception_sources_present_after: bool
    reception_sources_after: tuple[tuple[Any, float, float], ...] | None
    read_boundary: str = field(
        default=RECEPTION_PRE_STATE_BOUNDARY,
        init=False,
    )
    post_state_boundary: str = field(
        default=_RECEPTION_STAGE_POST_STATE_BOUNDARY,
        init=False,
    )
    glyph: Glyph = field(default=Glyph.EN, init=False)
    auxiliary_stability_certified: bool = field(default=False, init=False)
    _read_payload_stamp: tuple[Any, ...] = field(
        default=(),
        init=False,
        repr=False,
        compare=False,
    )
    _post_state_payload_stamp: tuple[Any, ...] = field(
        default=(),
        init=False,
        repr=False,
        compare=False,
    )
    _proof_stamp: tuple[Any, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def __getattribute__(self, name: str) -> Any:
        if name == "auxiliary_stability_certified":
            return False
        return object.__getattribute__(self, name)

    def __post_init__(self) -> None:
        _validate_reception_stage_observation(self)

    def _proof_fields_are_intact(self) -> bool:
        """Whether the captured EN read and its false scope claim are intact."""

        try:
            read_snapshot = _reception_observation_read_snapshot(self)
            if not proof_stamps_are_identical(
                object.__getattribute__(self, "_read_payload_stamp"),
                _reception_read_payload_stamp(read_snapshot),
            ):
                return False
            if not proof_stamps_are_identical(
                object.__getattribute__(self, "_post_state_payload_stamp"),
                _reception_observation_post_state_stamp(self),
            ):
                return False
            if not proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                _reception_stage_observation_stamp(self),
            ):
                return False
            _validate_reception_stage_observation(self)
        except BaseException:
            return False
        return True


def _observe_reception_proposal(
    graph: Any,
    proposal: NeighborStageProposal,
    *,
    target_index: int,
) -> ReceptionStageObservation:
    """Detach one sealed public record from a frozen internal EN proposal."""

    read = proposal.reception_read_snapshot
    if proposal.glyph is not Glyph.EN or type(read) is not ReceptionReadSnapshot:
        raise TypeError("Reception observation requires an EN proposal")
    if not read._proof_fields_are_intact():
        raise RuntimeError("Reception proposal read proof fields changed")
    committed_epi = require_real_scalar_epi(
        _raw_alias(graph, proposal.node, ALIAS_EPI, 0.0),
        operator="Reception",
        label="committed target EPI state",
    )
    committed_kind = _node_kind(graph, proposal.node)
    if (
        committed_epi != proposal.epi_after
        or committed_kind != proposal.epi_kind_after
    ):
        raise RuntimeError("Reception commit diverged from its frozen proposal")
    storage = graph.nodes[proposal.node]
    sources_present_after = "_reception_sources" in storage
    committed_sources = None
    if read.source_tracking_enabled and sources_present_after:
        raw_sources = storage["_reception_sources"]
        if type(raw_sources) is not list:
            raise RuntimeError("Reception source metadata has an invalid type")
        try:
            committed_sources = tuple(tuple(source) for source in raw_sources)
        except TypeError as exc:
            raise RuntimeError("Reception source metadata is not iterable") from exc
    if read.source_tracking_enabled:
        if not sources_present_after:
            raise RuntimeError("Reception did not commit its source list")
        if not _same_reception_value(
            committed_sources,
            read.reception_sources,
        ):
            raise RuntimeError("Reception source commit diverged from its snapshot")
    candidate = ReceptionStageObservation(
        target_index=target_index,
        node=proposal.node,
        target_epi_before=proposal.epi_before,
        target_epi_after=committed_epi,
        target_epi_kind_before=proposal.epi_kind_before,
        target_epi_kind_after=committed_kind,
        neighbors=read.neighbors,
        neighbor_epi_values=read.neighbor_epi_values,
        neighbor_dominant_values=read.neighbor_dominant_values,
        neighbor_epi_kinds=read.neighbor_epi_kinds,
        neighbor_epi_mean=read.neighbor_epi_mean,
        source_tracking_enabled=read.source_tracking_enabled,
        source_max_distance=read.source_max_distance,
        reception_sources=read.reception_sources,
        reception_sources_present_after=sources_present_after,
        reception_sources_after=committed_sources,
    )
    object.__setattr__(
        candidate,
        "_read_payload_stamp",
        _reception_read_payload_stamp(read),
    )
    object.__setattr__(
        candidate,
        "_post_state_payload_stamp",
        _reception_observation_post_state_stamp(candidate),
    )
    object.__setattr__(
        candidate,
        "_proof_stamp",
        _reception_stage_observation_stamp(candidate),
    )
    return candidate


_MUTATION_DECISION_PROOF_VERSION = "mutation_stage_decision_observation_v1"


def _mutation_decision_stamp(
    observation: "MutationStageDecisionObservation",
) -> tuple[Any, ...]:
    """Seal one accepted ZHIR decision independently of live graph metadata."""

    return (
        _MUTATION_DECISION_PROOF_VERSION,
        structural_proof_signature(
            (
                observation.target_index,
                observation.node,
                observation.trigger_certificate,
                observation.minimum_nu_f,
                observation.theta_before,
                observation.theta_after,
                observation.theta_shift,
                observation.fixed_mode,
                observation.regime_changed,
                observation.regime_before,
                observation.regime_after,
                observation.structural_acceleration,
                observation.acceleration_magnitude,
                observation.tau,
                observation.bifurcation_potential,
                observation.destabilizer_operator,
                observation.destabilizer_distance,
                observation.recent_history,
                observation.epi_kind_before,
                observation.operator_step,
                observation.glyph,
            )
        ),
    )


def _validate_mutation_trigger_certificate(
    certificate: MutationTriggerCertificate,
) -> None:
    """Reject internally contradictory trigger evidence in a public record."""

    required_floats = (
        certificate.current_epi,
        certificate.nu_f,
        certificate.delta_nfr,
        certificate.xi,
        certificate.predicted_depi_dt,
    )
    optional_floats = (
        certificate.observed_depi_dt,
        certificate.rate_gap,
    )
    if any(
        type(value) is not float or not math.isfinite(value)
        for value in required_floats
    ):
        raise ValueError("Mutation trigger certificate has invalid finite fields")
    if any(
        value is not None
        and (type(value) is not float or not math.isfinite(value))
        for value in optional_floats
    ):
        raise ValueError("Mutation trigger certificate has invalid optional rates")
    boolean_fields = (
        certificate.predicted_crossed,
        certificate.evidence_available,
        certificate.evidence_valid,
        certificate.capacity_active,
        certificate.threshold_gate_satisfied,
        certificate.physical_time_resolved,
    )
    if any(type(value) is not bool for value in boolean_fields):
        raise TypeError("Mutation trigger certificate has non-Boolean flags")
    if certificate.observed_crossed is not None and type(
        certificate.observed_crossed
    ) is not bool:
        raise TypeError("Mutation trigger observed_crossed must be bool or None")
    if certificate.current_endpoint_matches_state is not None and type(
        certificate.current_endpoint_matches_state
    ) is not bool:
        raise TypeError("Mutation trigger endpoint flag must be bool or None")
    for value in (certificate.source, certificate.time_basis, certificate.reason):
        if value is not None and type(value) is not str:
            raise TypeError("Mutation trigger certificate labels must be strings")

    evidence = certificate.evidence
    if type(evidence) is not MutationTriggerEvidence:
        raise TypeError("Mutation trigger certificate requires canonical evidence")
    evidence_required_floats = (evidence.previous_epi, evidence.current_epi)
    evidence_optional_floats = (
        evidence.previous_time,
        evidence.current_time,
        evidence.sample_interval,
        evidence.observed_depi_dt,
    )
    if any(
        type(value) is not float or not math.isfinite(value)
        for value in evidence_required_floats
    ) or any(
        value is not None
        and (type(value) is not float or not math.isfinite(value))
        for value in evidence_optional_floats
    ):
        raise ValueError("Mutation trigger evidence has invalid finite fields")
    if (
        type(evidence.source) is not str
        or type(evidence.time_basis) is not str
        or type(evidence.physical_time_resolved) is not bool
        or type(evidence.is_valid) is not bool
        or (
            evidence.current_endpoint_matches_state is not None
            and type(evidence.current_endpoint_matches_state) is not bool
        )
        or (evidence.reason is not None and type(evidence.reason) is not str)
    ):
        raise TypeError("Mutation trigger evidence metadata is invalid")

    predicted = certificate.nu_f * certificate.delta_nfr
    if (
        certificate.xi < 0.0
        or certificate.nu_f <= 0.0
        or certificate.predicted_depi_dt != predicted
        or certificate.predicted_crossed != (predicted > certificate.xi)
        or certificate.capacity_active != (certificate.nu_f > 0.0)
        or not certificate.evidence_available
        or not certificate.evidence_valid
        or not certificate.threshold_gate_satisfied
        or certificate.observed_crossed is not True
        or certificate.observed_depi_dt is None
        or not certificate.observed_depi_dt > certificate.xi
        or not evidence.is_valid
        or evidence.observed_depi_dt != certificate.observed_depi_dt
        or evidence.source != certificate.source
        or evidence.time_basis != certificate.time_basis
        or evidence.physical_time_resolved
        != certificate.physical_time_resolved
        or evidence.current_endpoint_matches_state
        != certificate.current_endpoint_matches_state
        or certificate.reason is not None
        or evidence.reason is not None
    ):
        raise ValueError("Mutation trigger certificate is internally inconsistent")
    if certificate.physical_time_resolved:
        if (
            evidence.previous_time is None
            or evidence.current_time is None
            or evidence.sample_interval is None
            or evidence.sample_interval <= 0.0
            or evidence.current_endpoint_matches_state is not True
            or evidence.current_epi != certificate.current_epi
        ):
            raise ValueError("Physical Mutation trigger evidence is inconsistent")
        expected_gap = certificate.observed_depi_dt - certificate.predicted_depi_dt
        if math.isfinite(expected_gap) and certificate.rate_gap != expected_gap:
            raise ValueError("Mutation trigger rate gap is inconsistent")
    elif certificate.rate_gap is not None:
        raise ValueError("Legacy Mutation trigger evidence cannot declare a rate gap")


def _validate_mutation_decision_fields(
    observation: "MutationStageDecisionObservation",
) -> None:
    """Validate the complete value-domain contract of one ZHIR observation."""

    if (
        type(observation.target_index) is not int
        or observation.target_index < 0
    ):
        raise ValueError("Mutation observation target_index must be nonnegative")
    if type(observation.trigger_certificate) is not MutationTriggerCertificate:
        raise TypeError(
            "Mutation observation requires a MutationTriggerCertificate"
        )
    certificate = observation.trigger_certificate
    _validate_mutation_trigger_certificate(certificate)

    finite_fields = (
        (observation.minimum_nu_f, "minimum_nu_f"),
        (observation.theta_before, "theta_before"),
        (observation.theta_after, "theta_after"),
        (observation.theta_shift, "theta_shift"),
        (observation.structural_acceleration, "structural_acceleration"),
        (observation.acceleration_magnitude, "acceleration_magnitude"),
        (observation.tau, "tau"),
    )
    for value, label in finite_fields:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError(
                f"Mutation observation {label} must be a finite float"
            )
    if observation.minimum_nu_f < 0.0:
        raise ValueError("Mutation observation minimum_nu_f must be nonnegative")
    if certificate.nu_f < observation.minimum_nu_f:
        raise ValueError("Mutation observation capacity is below its minimum")
    if observation.acceleration_magnitude != abs(
        observation.structural_acceleration
    ):
        raise ValueError("Mutation observation acceleration magnitude is inconsistent")
    if observation.tau < 0.0:
        raise ValueError("Mutation observation tau must be nonnegative")
    if type(observation.bifurcation_potential) is not bool or (
        observation.bifurcation_potential
        != (observation.acceleration_magnitude > observation.tau)
    ):
        raise ValueError("Mutation observation bifurcation decision is inconsistent")
    if type(observation.fixed_mode) is not bool:
        raise TypeError("Mutation observation fixed_mode must be a bool")
    if observation.fixed_mode:
        if any(
            value is not None
            for value in (
                observation.regime_changed,
                observation.regime_before,
                observation.regime_after,
            )
        ):
            raise ValueError("Fixed Mutation observations cannot declare regimes")
    elif (
        type(observation.regime_changed) is not bool
        or type(observation.regime_before) is not int
        or type(observation.regime_after) is not int
    ):
        raise ValueError("Dynamic Mutation observations require regime decisions")
    else:
        expected_before = int(observation.theta_before // (math.pi / 2.0))
        expected_after = int(observation.theta_after // (math.pi / 2.0))
        if (
            observation.regime_before != expected_before
            or observation.regime_after != expected_after
            or observation.regime_changed
            != (observation.regime_before != observation.regime_after)
            or not 0 <= observation.regime_before <= 3
            or not 0 <= observation.regime_after <= 3
        ):
            raise ValueError("Mutation observation regime decision is inconsistent")
    expected_theta_after = (
        observation.theta_before + (observation.theta_shift % math.tau)
    ) % math.tau
    if (
        not 0.0 <= observation.theta_before < math.tau
        or not 0.0 <= observation.theta_after < math.tau
        or observation.theta_after != expected_theta_after
    ):
        raise ValueError("Mutation observation phase decision is inconsistent")
    if type(observation.recent_history) is not tuple or any(
        type(item) is not str for item in observation.recent_history
    ):
        raise TypeError("Mutation observation recent_history must be a string tuple")
    if observation.destabilizer_operator is not None and type(
        observation.destabilizer_operator
    ) is not str:
        raise TypeError("Mutation observation destabilizer_operator is invalid")
    if observation.destabilizer_distance is not None and (
        type(observation.destabilizer_distance) is not int
        or observation.destabilizer_distance < 0
    ):
        raise ValueError("Mutation observation destabilizer_distance is invalid")
    if observation.epi_kind_before is not None and type(
        observation.epi_kind_before
    ) is not str:
        raise TypeError("Mutation observation epi_kind_before is invalid")
    if (
        type(observation.operator_step) is not int
        or observation.operator_step < 0
    ):
        raise ValueError("Mutation observation operator_step must be nonnegative")
    if observation.glyph is not Glyph.ZHIR:
        raise ValueError("Mutation observation glyph must be ZHIR")


@dataclass(frozen=True, slots=True)
class MutationStageDecisionObservation:
    """Value-sealed accepted ZHIR decision from one stage-start snapshot."""

    target_index: int
    node: Any
    trigger_certificate: MutationTriggerCertificate
    minimum_nu_f: float
    theta_before: float
    theta_after: float
    theta_shift: float
    fixed_mode: bool
    regime_changed: bool | None
    regime_before: int | None
    regime_after: int | None
    structural_acceleration: float
    acceleration_magnitude: float
    tau: float
    bifurcation_potential: bool
    destabilizer_operator: str | None
    destabilizer_distance: int | None
    recent_history: tuple[str, ...]
    epi_kind_before: str | None
    operator_step: int
    glyph: Glyph = field(default=Glyph.ZHIR, init=False)
    _proof_stamp: tuple[Any, ...] = field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _validate_mutation_decision_fields(self)

    def _proof_fields_are_intact(self) -> bool:
        """Whether no decisive field or nested trigger evidence was altered."""

        try:
            if not proof_stamps_are_identical(
                object.__getattribute__(self, "_proof_stamp"),
                _mutation_decision_stamp(self),
            ):
                return False
            _validate_mutation_decision_fields(self)
        except BaseException:
            return False
        return True


def _observe_mutation_proposal(
    proposal: PointwiseStageProposal,
    *,
    target_index: int,
) -> MutationStageDecisionObservation:
    """Detach one public observation from an internal frozen ZHIR proposal."""

    from ._mutation_stage_kernel import MutationNetworkStageProposal

    payload = proposal.payload
    if (
        proposal.glyph is not Glyph.ZHIR
        or type(payload) is not MutationNetworkStageProposal
    ):
        raise TypeError("Mutation decision observation requires a ZHIR proposal")
    phase = payload.phase
    gate = payload.runtime_gate
    candidate = MutationStageDecisionObservation(
        target_index=target_index,
        node=proposal.node,
        trigger_certificate=deepcopy(gate.threshold.certificate),
        minimum_nu_f=float(gate.minimum_nu_f),
        theta_before=float(phase.theta_before),
        theta_after=float(phase.theta_after),
        theta_shift=float(phase.theta_shift),
        fixed_mode=phase.fixed_mode,
        regime_changed=phase.regime_changed,
        regime_before=phase.regime_before,
        regime_after=phase.regime_after,
        structural_acceleration=float(payload.structural_acceleration),
        acceleration_magnitude=float(payload.acceleration_magnitude),
        tau=float(payload.tau),
        bifurcation_potential=payload.bifurcation_potential,
        destabilizer_operator=payload.destabilizer_operator,
        destabilizer_distance=payload.destabilizer_distance,
        recent_history=tuple(payload.recent_history),
        epi_kind_before=payload.epi_kind_before,
        operator_step=payload.operator_step,
    )
    return replace(
        candidate,
        _proof_stamp=_mutation_decision_stamp(candidate),
    )


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
    mutation_decision_observations: tuple[
        MutationStageDecisionObservation, ...
    ] = ()
    reception_observations: tuple[ReceptionStageObservation, ...] = ()

    def __post_init__(self) -> None:
        """Reject contradictory evidence payloads at their source."""

        observations = self.mutation_decision_observations
        if type(observations) is not tuple:
            raise TypeError("mutation_decision_observations must be a tuple")
        is_two_phase_mutation = bool(
            self.glyph == Glyph.ZHIR.value and self.schedule == TWO_PHASE_JACOBI
        )
        if is_two_phase_mutation:
            if len(observations) != self.nodes_processed:
                raise ValueError(
                    "Two-phase Mutation results require one decision observation "
                    "per processed target"
                )
            for index, observation in enumerate(observations):
                if type(observation) is not MutationStageDecisionObservation:
                    raise TypeError(
                        "mutation_decision_observations contain an invalid record"
                    )
                if observation.target_index != index:
                    raise ValueError(
                        "Mutation decision observation target order changed"
                    )
                if observation.glyph is not Glyph.ZHIR:
                    raise ValueError("Mutation decision observation glyph changed")
                if not observation._proof_fields_are_intact():
                    raise ValueError(
                        "Mutation decision observation proof fields are not intact"
                    )
        elif observations:
            raise ValueError(
                "Only two-phase Mutation results may carry decision observations"
            )

        reception_observations = self.reception_observations
        if type(reception_observations) is not tuple:
            raise TypeError("reception_observations must be a tuple")
        is_two_phase_reception = bool(
            self.glyph == Glyph.EN.value and self.schedule == TWO_PHASE_JACOBI
        )
        if is_two_phase_reception and reception_observations:
            if len(reception_observations) != self.nodes_processed:
                raise ValueError(
                    "Reception evidence requires one observation per target"
                )
            for index, observation in enumerate(reception_observations):
                if (
                    type(observation) is not ReceptionStageObservation
                    or observation.target_index != index
                    or observation.glyph is not Glyph.EN
                    or not observation._proof_fields_are_intact()
                ):
                    raise ValueError(
                        "Reception stage observations are not intact or ordered"
                    )
        elif reception_observations:
            raise ValueError(
                "Only two-phase Reception results may carry EN observations"
            )

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
    monitor = _runtime_mapping_known_value(graph.graph, "integrity_monitor")
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


def _restore_detached_neighbor_order(
    snapshot: Any,
    layout: _NetworkXRuntimeLayout,
) -> None:
    """Make the detached graph retain every live adjacency iteration order."""

    def reorder(target: MutableMapping[Any, Any], source: Any) -> None:
        target_by_identity = {
            id(key): (key, value)
            for key, value in _runtime_mapping_items(target)
        }
        ordered = tuple(
            target_by_identity[id(key)]
            for key, _value in _runtime_mapping_items(source)
        )
        if len(ordered) != len(target_by_identity):
            raise RuntimeError("detached stage adjacency does not match live graph")
        target.clear()
        target.update(ordered)

    adjacency = _runtime_stored_attribute(snapshot, "_adj")
    for node, source in layout.adjacency_inner:
        reorder(adjacency[node], source)
    if layout.directed:
        predecessors = _runtime_stored_attribute(snapshot, "_pred")
        for node, source in layout.predecessor_inner:
            reorder(predecessors[node], source)


def _detached_stage_graph(graph: Any) -> Any:
    """Return a complete detached logical graph for immutable stage reads."""

    layout = _networkx_runtime_layout(graph)
    if layout.directed:
        snapshot = nx.MultiDiGraph() if layout.multigraph else nx.DiGraph()
    else:
        snapshot = nx.MultiGraph() if layout.multigraph else nx.Graph()
    copy_memo: dict[int, Any] = {
        id(graph): snapshot,
        id(layout.graph_mapping): snapshot.graph,
    }
    graph_mapping_items = _runtime_mapping_items(layout.graph_mapping)
    detached_search_roots: list[tuple[Any, Any]] = [
        (("graph-metadata", index), value)
        for index, (key, value) in enumerate(graph_mapping_items)
        if not _is_runtime_graph_key(key)
    ]
    detached_search_roots.extend(
        (("node-metadata", node_index, value_index), value)
        for node_index, (_node, data) in enumerate(layout.node_data)
        for value_index, (_key, value) in enumerate(_runtime_mapping_items(data))
    )
    detached_search_roots.extend(
        (("edge-metadata", edge_index, value_index), value)
        for edge_index, edge in enumerate(layout.edges)
        for value_index, (_key, value) in enumerate(
            _runtime_mapping_items(edge[-1])
        )
    )
    manual_state_items = _discover_runtime_manual_state_items(
        tuple(detached_search_roots),
        protected_values=_graph_transaction_protected_values(
            graph,
            _layout=layout,
        ),
    )
    copy_memo.update({id(value): value for _path, value in manual_state_items})
    for key, value in graph_mapping_items:
        if _is_runtime_graph_key(key):
            copy_memo[id(value)] = value
        _seed_runtime_resource_memo(value, copy_memo)
    for _node, data in layout.node_data:
        _seed_runtime_resource_memo(data, copy_memo)
    for edge in layout.edges:
        _seed_runtime_resource_memo(edge[-1], copy_memo)
    for key, value in graph_mapping_items:
        if _is_runtime_graph_key(key):
            continue
        snapshot.graph[key] = deepcopy(value, copy_memo)
    # The monitor is never invoked while proposals are built, but its public
    # shape remains part of common operator argument preflight.
    monitor = _runtime_mapping_known_value(
        layout.graph_mapping,
        "integrity_monitor",
    )
    if monitor is not None:
        snapshot.graph["integrity_monitor"] = monitor

    snapshot.add_nodes_from(
        (node, deepcopy(dict(data), copy_memo))
        for node, data in layout.node_data
    )
    if layout.multigraph:
        snapshot.add_edges_from(
            (left, right, key, deepcopy(dict(data), copy_memo))
            for left, right, key, data in layout.edges
        )
    else:
        snapshot.add_edges_from(
            (left, right, deepcopy(dict(data), copy_memo))
            for left, right, data in layout.edges
        )
    _restore_detached_neighbor_order(snapshot, layout)
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
    metric_consumer_graph: Any,
) -> NeighborStageProposal:
    raw_epi = _raw_alias(snapshot, node, ALIAS_EPI, 0.0)
    read_snapshot = capture_reception_read_snapshot(
        snapshot,
        node,
        track_sources=track_sources,
        max_distance=max_distance,
        _metric_consumer_graph_owner=metric_consumer_graph,
    )
    epi_before = read_snapshot.target_epi
    current_kind = read_snapshot.target_epi_kind
    neighbors = read_snapshot.neighbors
    epi_bar = read_snapshot.neighbor_epi_mean
    if neighbors:
        proposed = neighbor_epi_blend_value(
            epi_before, epi_bar, float(factors["EN_mix"])
        )
        epi_after = _bounded_epi(snapshot, node, proposed)

        final_kind = reception_proposed_epi_kind(
            current_kind,
            zip(
                read_snapshot.neighbor_dominant_values,
                read_snapshot.neighbor_epi_kinds,
                strict=True,
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
        reception_sources=read_snapshot.reception_sources,
        reception_read_snapshot=read_snapshot,
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
        if payload_node is not proposal.node:
            try:
                target_matches = bool(payload_node == proposal.node)
            except BaseException:
                target_matches = False
            if not target_matches:
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
        read_snapshot = proposal.reception_read_snapshot
        if glyph is Glyph.EN:
            if type(read_snapshot) is not ReceptionReadSnapshot:
                raise RuntimeError("Reception proposal lost its pre-state snapshot")
            if not read_snapshot._proof_fields_are_intact():
                raise RuntimeError("Reception proposal read proof fields changed")
            if (
                not _same_reception_value(read_snapshot.node, proposal.node)
                or read_snapshot.target_epi != proposal.epi_before
                or read_snapshot.target_epi_kind
                != proposal.epi_kind_before
                or not _same_reception_value(
                    read_snapshot.neighbors,
                    proposal.neighbors,
                )
                or read_snapshot.neighbor_epi_mean
                != proposal.neighbor_epi_mean
                or not _same_reception_value(
                    read_snapshot.reception_sources,
                    proposal.reception_sources,
                )
            ):
                raise RuntimeError("Reception proposal read snapshot changed")
        elif read_snapshot is not None:
            raise RuntimeError("Only Reception proposals may carry EN reads")
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
            metric_state = states_before[node]
            if (
                isinstance(proposal, NeighborStageProposal)
                and proposal.glyph is Glyph.EN
            ):
                from .definitions_base import _PREPARED_OPERATOR_STATE_KEY

                metric_state = dict(metric_state)
                metric_state[_PREPARED_OPERATOR_STATE_KEY] = (
                    proposal.reception_read_snapshot
                )
            graph.graph.setdefault("operator_metrics", []).append(
                operator._collect_metrics(graph, node, metric_state)
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

    transaction = _select_graph_transaction(graph, transaction_snapshot)
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
    transaction = _select_graph_transaction(graph, transaction_snapshot)
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
    transaction = _select_graph_transaction(graph, transaction_snapshot)
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
    transaction = _select_graph_transaction(graph, transaction_snapshot)
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

    transaction = _select_graph_transaction(graph, transaction_snapshot)
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
        mutation_decision_observations = (
            tuple(
                _observe_mutation_proposal(
                    proposal,
                    target_index=index,
                )
                for index, proposal in enumerate(proposals)
            )
            if operator.glyph is Glyph.ZHIR
            else ()
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
            mutation_decision_observations=(
                mutation_decision_observations
            ),
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
    transaction = _select_graph_transaction(graph, transaction_snapshot)
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
    transaction = _select_graph_transaction(graph, transaction_snapshot)
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
                    metric_consumer_graph=graph,
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

        reception_observations = (
            tuple(
                _observe_reception_proposal(
                    graph,
                    proposal,
                    target_index=index,
                )
                for index, proposal in enumerate(proposals)
            )
            if operator.glyph is Glyph.EN
            else ()
        )

        result = NetworkStageResult(
            operator=operator.name,
            glyph=operator.glyph.value,
            schedule=TWO_PHASE_JACOBI,
            nodes_processed=len(targets_tuple),
            neighbor_epi_jump_certificate=neighbor_certificate,
            epi_jump_certificate_abstention_reason=(
                certificate_abstention_reason
            ),
            reception_observations=reception_observations,
        )

        # Emit accepted-transaction telemetry last. Invalid final evidence
        # cannot leak a warning, while warning-as-error still rolls back.
        if operator.glyph is Glyph.EN and bool(
            execution_kwargs.get("track_sources", True)
        ):
            for proposal in proposals:
                if not proposal.reception_sources:
                    warnings.warn(
                        reception_no_sources_warning(proposal.node),
                        stacklevel=3,
                    )
        return result
    except BaseException as failure:
        _discard_pending_monitor(graph)
        transaction.restore_after_failure(graph, failure)
        raise


__all__ = [
    "GraphTransactionSnapshot",
    "MutationStageDecisionObservation",
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
    "ReceptionStageObservation",
    "record_gauss_seidel_stage",
]
