"""Utilities for migrating legacy payloads to the English-only contract."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any, Hashable, Iterable, cast

from ..types import GraphLike

__all__ = ("migrate_legacy_phase_attributes",)


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
