"""Rebuild exact support evidence and detached views of recorded nodal state.

Complete archival records are checked against the existing physics owners:
derived fields are recomputed, then every saved field must agree. This admits
internal arithmetic consistency, not execution provenance. In particular a
captured phase gradient remains supplied evidence; no phase kernel is rerun.
Raw nodal records and exact coefficient records have separate numeric schemas.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction

import networkx as nx

from .._exact_time import exact_or_represented_real, finite_represented_real
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from ..physics.forced_support import ForcedSupportBalance, derive_forced_support_balance
from ..physics.forcing_realization import (
    NonEpiForcingObservation,
    decompose_non_epi_forcing,
)
from ..physics.support_transport import SupportTransportSnapshot, _from_data
from ..types import require_finite_real_scalar_epi

__all__ = [
    "read_recorded_support_snapshot",
    "read_recorded_forcing",
    "read_recorded_forced_reference",
    "bind_recorded_nodal_state",
    "graph_from_recorded_nodal_state",
]


def _mapping(value, label):
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _array(value, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an ordered array")
    return tuple(value)


def _exact(value, label):
    # Recorded exact coefficients use Fraction's integer or p/q spelling.
    # Do not interpret decimal text or floats as an exact archival field.
    if type(value) is str:
        if re.fullmatch(r"-?[0-9]+(?:/[1-9][0-9]*)?", value) is None:
            raise ValueError(f"{label} must contain integer/rational text")
        value = Fraction(value)
    if type(value) not in (int, Fraction):
        raise ValueError(f"{label} must contain exact integer/rational coordinates")
    return exact_or_represented_real(value, label)


def _vector(value, label):
    return tuple(_exact(item, label) for item in _array(value, label))


def _record_tree(value):
    """Normalize native asdict records to their canonical JSON representation."""
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("record field names must be strings")
        return {key: _record_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record_tree(item) for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        finite_represented_real(value, "record number")
        return value
    raise ValueError("record contains an unsupported serialized value")


def _equal_record(expected, actual, label):
    # JSON comparison also distinguishes logical true/false from numeric 1/0.
    def encoded(value):
        return json.dumps(_record_tree(value), sort_keys=True, allow_nan=False)

    if encoded(expected) != encoded(actual):
        raise ValueError(f"retained {label} differs from its reconstructed evidence")


def _node(value):
    if isinstance(value, (tuple, list)):
        return tuple(_node(item) for item in value)
    if type(value) in (int, str):
        return value
    if type(value) is float:
        finite_represented_real(value, "node identifier")
        return value
    raise ValueError("node identifiers must be finite JSON scalars or nested arrays")


def _nodes(value):
    nodes = tuple(_node(item) for item in _array(value, "nodes"))
    if len(set(nodes)) != len(nodes):
        raise ValueError("node order must contain distinct nodes")
    return nodes


def read_recorded_support_snapshot(payload: Mapping) -> SupportTransportSnapshot:
    """Rebuild primitive transport data and verify the complete saved snapshot.

    Accept an ``asdict`` mapping or its canonical JSON representation, whose
    exact fields use integer/rational strings. All derived caches are checked
    against their reconstruction; missing, extra or conflicting fields fail.
    Serialized array-valued node identifiers are restored as immutable tuples.
    """
    raw = _mapping(payload, "support snapshot")
    edges = []
    for entry in _array(raw["conductance"], "conductance"):
        i, j, weight = _array(entry, "conductance entry")
        edges.append((i, j, _exact(weight, "conductance")))
    result = _from_data(
        _nodes(raw["nodes"]),
        tuple(edges),
        tuple(
            _array(row, "support row")
            for row in _array(raw["support_neighbors"], "support_neighbors")
        ),
        _vector(raw["epi"], "epi"),
        _vector(raw["capacity"], "capacity"),
        _vector(raw["stored_pressure"], "stored_pressure"),
    )
    _equal_record(asdict(result), raw, "support snapshot")
    return result


def read_recorded_forced_reference(payload: Mapping) -> ForcedSupportBalance:
    """Re-derive the held-source reference and check every recorded field."""
    raw = _mapping(payload, "forced reference")
    result = derive_forced_support_balance(
        read_recorded_support_snapshot(raw["source"]),
        epi_weight=_exact(raw["epi_weight"], "epi_weight"),
        forcing=_vector(raw["forcing"], "forcing"),
    )
    _equal_record(asdict(result), raw, "forced reference")
    return result


def read_recorded_forcing(payload: Mapping) -> tuple:
    """Check a complete ``{observation, components}`` forcing capture.

    Return ``(snapshot, observation, components)``. The captured phase gradient
    is an input coefficient, not proof that it came from the stated phase,
    a particular kernel branch or a causal execution. Exact decomposition,
    full-kernel assembly defect and stored-pressure residual are all checked.
    """
    raw = _mapping(payload, "forcing capture")
    row = _mapping(raw["observation"], "forcing observation")
    snapshot = read_recorded_support_snapshot(row["snapshot"])
    names = (
        "phase",
        "forcing",
        "phase_gradient",
        "full_kernel_pressure",
        "kernel_pressure_defect",
        "stored_pressure_residual",
    )
    values = {name: _vector(row[name], name) for name in names}
    if any(len(value) != len(snapshot.nodes) for value in values.values()):
        raise ValueError("captured vectors must match the complete support")
    weights = []
    for entry in _array(row["normalized_weights"], "normalized_weights"):
        name, value = _array(entry, "channel weight")
        weights.append((name, _exact(value, "channel weight")))
    observation = NonEpiForcingObservation(
        snapshot=snapshot,
        epi_weight=_exact(row["epi_weight"], "epi_weight"),
        normalized_weights=tuple(weights),
        **values,
    )
    components = decompose_non_epi_forcing(observation)
    _equal_record(
        {"observation": asdict(observation), "components": components},
        raw,
        "complete forcing capture",
    )
    model = tuple(
        observation.epi_weight * gradient + force
        for gradient, force in zip(
            snapshot.epi_gradient, observation.forcing, strict=True
        )
    )
    _equal_record(
        tuple(p - m for p, m in zip(observation.full_kernel_pressure, model)),
        row["kernel_pressure_defect"],
        "kernel pressure defect",
    )
    _equal_record(
        tuple(
            p - fresh
            for p, fresh in zip(
                snapshot.stored_pressure, observation.full_kernel_pressure
            )
        ),
        row["stored_pressure_residual"],
        "stored pressure residual",
    )
    return snapshot, observation, components


def _raw_number(value, label):
    if type(value) not in (int, float):
        raise ValueError(f"{label} must retain finite real JSON numbers")
    finite_represented_real(value, label)
    return value


def _nodal_values(payload):
    raw = _mapping(payload, "nodal state")
    nodes = _nodes(raw["nodes"])
    values = {}
    for name in ("epi", "capacity", "phase", "pressure"):
        vector = tuple(_raw_number(item, name) for item in _array(raw[name], name))
        if len(vector) != len(nodes):
            raise ValueError("raw state vectors must match the node order")
        values[name] = vector
    for value in values["epi"]:
        require_finite_real_scalar_epi(value)
    if any(value < 0 for value in values["capacity"]):
        raise ValueError("capacity must be nonnegative")
    time = _raw_number(raw["time"], "time")
    return nodes, values, time


def bind_recorded_nodal_state(state: Mapping, capture: Mapping, snapshot) -> None:
    """Bind raw recorded triad/pressure to a complete detached forcing capture.

    All raw channels, including phase, require finite JSON numbers; textual
    coercion and boolean zero/one are forbidden. This compares nodal values
    without constructing a graph, authenticating history or checking edges.
    """
    nodes, values, _ = _nodal_values(state)
    if type(snapshot) is not SupportTransportSnapshot:
        raise TypeError("snapshot must be a SupportTransportSnapshot")
    supplied = read_recorded_support_snapshot(asdict(snapshot))
    captured, observation, _ = read_recorded_forcing(capture)
    if supplied != captured or nodes != supplied.nodes:
        raise ValueError("live snapshot node/state differs from forcing capture")
    for field, key in (
        ("epi", "epi"),
        ("capacity", "capacity"),
        ("stored_pressure", "pressure"),
    ):
        _equal_record(
            getattr(supplied, field),
            tuple(Fraction(value) for value in values[key]),
            f"live snapshot {field}",
        )
    _equal_record(
        observation.phase,
        tuple(Fraction(value) for value in values["phase"]),
        "live captured phase",
    )


def graph_from_recorded_nodal_state(payload: Mapping) -> nx.Graph:
    """Build a detached simple undirected view of the raw ``_state`` schema.

    Retain node order, recorded edge insertion order/attributes, raw signed
    EPI, capacity, phase, pressure and displayed time. Required inputs are
    never defaulted. No history, cache, operator, pressure refresh or runtime
    configuration is reconstructed. An absent edge attribute stays absent;
    subsequent observers retain their own explicit adjacency conventions.
    """
    nodes, values, time = _nodal_values(payload)
    known = set(nodes)
    edges, seen = [], set()
    for entry in _array(payload["edges"], "edges"):
        left, right, attributes = _array(entry, "edge")
        left, right = _node(left), _node(right)
        if left not in known or right not in known:
            raise ValueError("edge endpoints must belong to the node order")
        key = frozenset((left, right))
        if left == right or key in seen:
            raise ValueError("edges must be distinct undirected pairs without loops")
        seen.add(key)
        attributes = _mapping(attributes, "edge attributes")
        if "weight" in attributes:
            weight = _raw_number(attributes["weight"], "edge weight")
            if weight < 0:
                raise ValueError("edge weight must be nonnegative")
        edges.append((left, right, deepcopy(dict(attributes))))
    graph = nx.Graph()
    graph.graph["_t"] = time
    aliases = (
        ("epi", ALIAS_EPI),
        ("capacity", ALIAS_VF),
        ("phase", ALIAS_THETA),
        ("pressure", ALIAS_DNFR),
    )
    for index, node in enumerate(nodes):
        graph.add_node(
            node, **{alias[0]: values[name][index] for name, alias in aliases}
        )
    graph.add_edges_from(edges)
    return graph
