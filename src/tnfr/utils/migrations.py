"""Utilities for migrating legacy payloads to the English-only contract."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, Hashable, Iterable, cast

from ..types import GraphLike

__all__ = ("migrate_legacy_phase_attributes", "migrate_legacy_remesh_cooldown")


LEGACY_REMESH_COOLDOWN_KEY = "REMESH_COOLDOWN_VENTANA"
REMESH_COOLDOWN_KEY = "REMESH_COOLDOWN_WINDOW"


def _iter_node_payloads(
    obj: GraphLike | Mapping[Hashable, MutableMapping[str, Any]]
) -> Iterable[MutableMapping[str, Any]]:
    nodes = getattr(obj, "nodes", None)
    if nodes is None:
        if isinstance(obj, Mapping):
            return (cast(MutableMapping[str, Any], data) for data in obj.values())
        raise TypeError("Object does not expose node payloads")
    return (
        cast(MutableMapping[str, Any], data)
        for _, data in nodes(data=True)
        if isinstance(data, MutableMapping)
    )


def migrate_legacy_phase_attributes(
    obj: GraphLike | Mapping[Hashable, MutableMapping[str, Any]]
) -> int:
    """Replace legacy ``"fase"``/``"θ"`` node keys with ``"theta"``/``"phase"``.

    Parameters
    ----------
    obj:
        ``networkx``-style graph or mapping exposing node payloads.

    Returns
    -------
    int
        Number of node payloads updated.
    """

    updated = 0
    for data in _iter_node_payloads(obj):
        if "fase" in data:
            legacy_value = data.pop("fase")
        elif "θ" in data:
            legacy_value = data.pop("θ")
        else:
            continue
        try:
            coerced = float(legacy_value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("Legacy phase value is not numeric") from exc
        if "theta" not in data:
            data["theta"] = coerced
        else:
            data["theta"] = float(data["theta"])
        data["phase"] = float(data["theta"])
        updated += 1
    return updated


def _get_graph_mapping(obj: GraphLike | MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    if isinstance(obj, MutableMapping):
        return obj
    graph_attr = getattr(obj, "graph", None)
    if not isinstance(graph_attr, MutableMapping):
        raise TypeError("Object does not expose a mutable graph mapping")
    return cast(MutableMapping[str, Any], graph_attr)


def migrate_legacy_remesh_cooldown(
    obj: GraphLike | MutableMapping[str, Any]
) -> int:
    """Remove ``REMESH_COOLDOWN_VENTANA`` and promote the English key."""

    graph_data = _get_graph_mapping(obj)
    if LEGACY_REMESH_COOLDOWN_KEY not in graph_data:
        return 0

    legacy_value = graph_data.pop(LEGACY_REMESH_COOLDOWN_KEY)
    try:
        canonical_value = int(legacy_value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("Legacy remesh cooldown must be numeric") from exc

    if REMESH_COOLDOWN_KEY not in graph_data:
        graph_data[REMESH_COOLDOWN_KEY] = canonical_value

    return 1
