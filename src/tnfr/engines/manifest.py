"""Shared graph payload and telemetry contract for engine manifests.

Version 1 supports finite JSON state, scalar node/edge IDs, and all four
NetworkX graph kinds. It preserves attributes, including JSON operator
history, but is not a checkpoint for Python callbacks, RNG objects, arrays,
or other runtime state. Unsupported values raise instead of being stringified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

import networkx as nx

from ..mathematics.unified_numerical import np
from ..utils.io import safe_write

GRAPH_SCHEMA = "tnfr-graph-json-v1"


def _finite_json_state(
    value: Any,
    path: str,
    *,
    extended_sequences: bool,
) -> Any:
    """Implement the manifest's strict and extended finite-JSON policies."""

    if extended_sequences and isinstance(value, np.generic):
        return _finite_json_state(
            value.item(), path, extended_sequences=extended_sequences
        )
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if math.isfinite(value):
            return value
        raise ValueError(f"{path} requires finite JSON state; non-finite number")
    if isinstance(value, list) or (extended_sequences and isinstance(value, tuple)):
        return [
            _finite_json_state(
                item,
                f"{path}[{index}]",
                extended_sequences=extended_sequences,
            )
            for index, item in enumerate(value)
        ]
    mapping_type = Mapping if extended_sequences else dict
    if isinstance(value, mapping_type):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{path} requires finite JSON state; non-string key")
        return {
            key: _finite_json_state(
                item, f"{path}.{key}", extended_sequences=extended_sequences
            )
            for key, item in value.items()
        }
    raise ValueError(
        f"{path} requires finite JSON state; unsupported {type(value).__name__}"
    )


def finite_json_state(value: Any, path: str = "state") -> Any:
    """Return a finite JSON-native copy without changing value semantics.

    Tuples become JSON arrays and NumPy scalars become their Python scalar
    equivalents. Booleans remain booleans, ``None`` remains JSON null, and
    nested string-keyed mappings retain their structure. Non-finite numbers
    and unsupported runtime objects are rejected with their value path.
    """

    return _finite_json_state(value, path, extended_sequences=True)


def _json_state(value: Any, path: str = "state") -> Any:
    """Validate persistent graph state under the narrower v1 graph schema."""

    return _finite_json_state(value, path, extended_sequences=False)


def _scalar_id(value: Any) -> Any:
    if value is None or type(value) not in (str, bool, int, float):
        raise ValueError("manifest node and edge IDs must be non-null JSON scalars")
    return _json_state(value, "identifier")


def encode_graph(graph: nx.Graph) -> dict[str, Any]:
    """Capture supported graph state without mutating or relabelling it."""
    nodes = [
        {"id": _scalar_id(node), "attributes": _json_state(dict(data), "node.attributes")}
        for node, data in graph.nodes(data=True)
    ]
    edges = []
    source_edges = (
        graph.edges(keys=True, data=True) if graph.is_multigraph()
        else ((u, v, None, data) for u, v, data in graph.edges(data=True))
    )
    for source, target, key, data in source_edges:
        entry = {
            "source": _scalar_id(source), "target": _scalar_id(target),
            "attributes": _json_state(dict(data), "edge.attributes"),
        }
        if graph.is_multigraph():
            entry["key"] = _scalar_id(key)
        edges.append(entry)
    return {
        "schema": GRAPH_SCHEMA, "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "attributes": _json_state(dict(graph.graph), "graph.attributes"),
        "nodes": nodes, "edges": edges,
    }


def decode_graph(payload: dict[str, Any]) -> nx.Graph:
    """Load the declared graph schema, rejecting unknown or lossy state."""
    if payload.get("schema") != GRAPH_SCHEMA:
        raise ValueError("Unsupported manifest graph schema")
    if type(payload.get("directed")) is not bool or type(payload.get("multigraph")) is not bool:
        raise ValueError("Manifest graph flags must be booleans")
    graph_type = (
        nx.MultiDiGraph if payload["directed"] else nx.MultiGraph
    ) if payload["multigraph"] else (nx.DiGraph if payload["directed"] else nx.Graph)
    graph = graph_type()
    graph.graph.update(_json_state(payload["attributes"], "graph.attributes"))
    for record in payload["nodes"]:
        node = _scalar_id(record["id"])
        if node in graph:
            raise ValueError("Duplicate node ID in manifest graph")
        graph.add_node(node)
        graph.nodes[node].update(_json_state(record["attributes"], "node.attributes"))
    for record in payload["edges"]:
        source, target = _scalar_id(record["source"]), _scalar_id(record["target"])
        if source not in graph or target not in graph:
            raise ValueError("Manifest edge references an undeclared node")
        attributes = _json_state(record["attributes"], "edge.attributes")
        if graph.is_multigraph():
            key = _scalar_id(record["key"])
            if graph.has_edge(source, target, key):
                raise ValueError("Duplicate edge key in manifest graph")
            graph.add_edge(source, target, key=key)
            graph[source][target][key].update(attributes)
        else:
            if graph.has_edge(source, target):
                raise ValueError("Duplicate edge in manifest graph")
            graph.add_edge(source, target)
            graph[source][target].update(attributes)
    return graph


def collect_manifest_telemetry(graph: nx.Graph) -> dict[str, Any]:
    """Read canonical C, mean Si, and potential; record unavailable metrics."""
    from ..metrics.common import compute_coherence
    from ..metrics.sense_index import compute_Si
    from ..physics.canonical import compute_structural_potential

    # Metric caches belong to this readout copy, not the serialized source.
    readout = graph.copy()
    telemetry: dict[str, Any] = {}
    errors = {}
    computations = {
        "coherence": lambda: float(compute_coherence(readout)),
        "sense_index": lambda: fmean(compute_Si(readout, inplace=False).values()) if readout else 0.0,
        "structural_potential_range": lambda: (
            [min(values.values()), max(values.values())]
            if (values := compute_structural_potential(readout)) else None
        ),
    }
    for name, compute in computations.items():
        try:
            telemetry[name] = _json_state(compute(), name)
        except Exception as exc:
            telemetry[name] = None
            errors[name] = str(exc)
    if errors:
        telemetry["errors"] = errors
    return telemetry


def write_manifest_bundle(
    output_dir: Path, manifest_name: str, summary_name: str,
    manifest: dict[str, Any], summary: dict[str, Any],
    partitions: Iterable[tuple[str, nx.Graph, dict[str, Any]]],
) -> dict[str, Path]:
    """Write graph payloads and a compatible entries index beside the summary."""
    output_dir = Path(output_dir)
    entries = []
    files = []
    for index, (partition_id, graph, telemetry) in enumerate(partitions):
        filename = f"{Path(manifest_name).stem}_partition_{index}.json"
        files.append((filename, {
            "partition_id": partition_id, "graph": encode_graph(graph),
            "partition": {"telemetry": telemetry},
        }))
        entries.append({
            "partition_id": partition_id, "relative_path": filename,
            "size": len(graph), "telemetry": telemetry,
        })
    manifest = {**manifest, "graph_schema": GRAPH_SCHEMA, "entries": entries}
    files.extend([(manifest_name, manifest), (summary_name, summary)])
    # Validate every payload before publishing any file in this bundle.
    encoded = [(name, json.dumps(data, indent=2, allow_nan=False)) for name, data in files]
    for name, data in encoded:
        safe_write(output_dir / name, lambda handle, text=data: handle.write(text))
    return {
        "manifest_absolute": (output_dir / manifest_name).resolve(),
        "summary_absolute": (output_dir / summary_name).resolve(),
    }
