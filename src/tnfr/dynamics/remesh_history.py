"""Canonical rolling EPI-history snapshots for delayed REMESH.

The runtime samples one whole-network EPI state immediately before its
optional delayed REMESH jump. Keeping this small operation here gives the
ordinary step runtime and explicit event/REMESH cycles the same indexing
convention: a delay tau is read later at history[-(tau + 1)].
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .._remesh_contract import remesh_history_maxlen
from ..constants import DEFAULTS
from ..errors import TNFRValueError
from ..types import TNFRGraph
from .aliases import ALIAS_EPI

__all__ = ["RemeshEPIHistoryAppend", "append_remesh_epi_history_snapshot"]


@dataclass(frozen=True, slots=True)
class RemeshEPIHistoryAppend:
    """Observable result of one canonical pre-REMESH history sample."""

    history_length_before: int
    history_length_after: int
    history_maxlen: int
    oldest_snapshot_evicted: bool
    snapshot_items: tuple[tuple[Any, Any], ...]


def append_remesh_epi_history_snapshot(
    graph: TNFRGraph,
) -> RemeshEPIHistoryAppend:
    """Append the current full-support EPI snapshot to canonical REMESH memory.

    Existing noncanonical containers are materialized into the same bounded
    deque used by tnfr.dynamics.runtime.step. The retained snapshot is the
    pre-jump state; this helper neither executes REMESH nor records its
    post-jump output.
    """

    from ..operators.network_stage import (
        _networkx_runtime_layout,
        _runtime_mapping_items,
    )

    layout = _networkx_runtime_layout(graph)
    graph_items = _runtime_mapping_items(layout.graph_mapping)

    def graph_value(key: str, default: Any) -> Any:
        return next(
            (
                value
                for candidate, value in graph_items
                if type(candidate) is str and candidate == key
            ),
            default,
        )

    maxlen = remesh_history_maxlen(
        graph_value("REMESH_TAU_GLOBAL", DEFAULTS["REMESH_TAU_GLOBAL"]),
        graph_value("REMESH_TAU_LOCAL", DEFAULTS["REMESH_TAU_LOCAL"]),
    )
    snapshot_items: list[tuple[Any, Any]] = []
    for node, data in layout.node_data:
        items = _runtime_mapping_items(data)
        value = next(
            (
                stored
                for alias in ALIAS_EPI
                for candidate, stored in items
                if type(candidate) is str and candidate == alias
            ),
            0.0,
        )
        snapshot_items.append((node, value))
    return _append_remesh_epi_history_snapshot_from_materialized(
        graph,
        history_maxlen=maxlen,
        snapshot_items=tuple(snapshot_items),
    )


def _append_remesh_epi_history_snapshot_from_materialized(
    graph: TNFRGraph,
    *,
    history_maxlen: int,
    snapshot_items: tuple[tuple[Any, Any], ...],
) -> RemeshEPIHistoryAppend:
    """Append one bridge-owned snapshot already bound to ordered support."""

    from ..operators._delayed_remesh_kernel import _materialize_indexed_history
    from ..operators.network_stage import (
        _networkx_runtime_layout,
        _runtime_mapping_items,
        _set_runtime_mapping_item,
    )

    if type(history_maxlen) is not int or history_maxlen <= 0:
        raise TNFRValueError("history_maxlen must be a positive int")
    if type(snapshot_items) is not tuple:
        raise TNFRValueError("snapshot_items must be an ordered tuple")
    layout = _networkx_runtime_layout(graph)
    if len(snapshot_items) != len(layout.node_data) or any(
        observed_node is not expected_node
        for (observed_node, _value), (expected_node, _data) in zip(
            snapshot_items,
            layout.node_data,
            strict=True,
        )
    ):
        raise TNFRValueError(
            "snapshot_items must match the frozen ordered node support"
        )
    graph_items = _runtime_mapping_items(layout.graph_mapping)
    raw_history = next(
        (
            value
            for key, value in graph_items
            if type(key) is str and key == "_epi_hist"
        ),
        None,
    )
    if raw_history is None:
        retained = ()
    else:
        retained = _materialize_indexed_history(raw_history)

    if (
        type(raw_history) is not deque
        or raw_history.maxlen != history_maxlen
    ):
        raw_history = deque(
            retained[-history_maxlen:],
            maxlen=history_maxlen,
        )
        _set_runtime_mapping_item(layout.graph_mapping, "_epi_hist", raw_history)

    length_before = deque.__len__(raw_history)
    deque.append(raw_history, dict(snapshot_items))
    length_after = deque.__len__(raw_history)
    expected_length_after = min(length_before + 1, history_maxlen)
    if length_after != expected_length_after:
        raise TNFRValueError(
            "canonical _epi_hist append did not add exactly one snapshot"
        )
    return RemeshEPIHistoryAppend(
        history_length_before=length_before,
        history_length_after=length_after,
        history_maxlen=history_maxlen,
        oldest_snapshot_evicted=(
            length_before == history_maxlen
            and length_after == history_maxlen
        ),
        snapshot_items=snapshot_items,
    )
