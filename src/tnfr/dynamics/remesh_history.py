"""Canonical rolling EPI-history snapshots for delayed REMESH.

The runtime samples one whole-network EPI state immediately before its
optional delayed REMESH jump. Keeping this small operation here gives the
ordinary step runtime and explicit event/REMESH cycles the same indexing
convention: a delay tau is read later at history[-(tau + 1)].
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._remesh_contract import remesh_history_maxlen
from ..alias import get_attr
from ..constants import get_param
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

    maxlen = remesh_history_maxlen(
        get_param(graph, "REMESH_TAU_GLOBAL"),
        get_param(graph, "REMESH_TAU_LOCAL"),
    )
    raw_history = graph.graph.get("_epi_hist")
    if raw_history is None:
        retained = ()
    elif isinstance(raw_history, (str, bytes, bytearray, Mapping)):
        raise TNFRValueError(
            "_epi_hist must be a replayable indexed history"
        )
    elif not hasattr(raw_history, "__len__") or not hasattr(
        raw_history,
        "__getitem__",
    ):
        raise TNFRValueError(
            "_epi_hist must be a replayable indexed history"
        )
    else:
        try:
            retained = tuple(raw_history)
        except (OverflowError, TypeError) as exc:
            raise TNFRValueError(
                "_epi_hist must be a replayable indexed history"
            ) from exc

    if type(raw_history) is not deque or raw_history.maxlen != maxlen:
        raw_history = deque(retained[-maxlen:], maxlen=maxlen)
        graph.graph["_epi_hist"] = raw_history

    length_before = len(raw_history)
    snapshot_items = tuple(
        (node, get_attr(data, ALIAS_EPI, 0.0))
        for node, data in graph.nodes(data=True)
    )
    raw_history.append(dict(snapshot_items))
    length_after = len(raw_history)
    expected_length_after = min(length_before + 1, maxlen)
    if length_after != expected_length_after:
        raise TNFRValueError(
            "canonical _epi_hist append did not add exactly one snapshot"
        )
    return RemeshEPIHistoryAppend(
        history_length_before=length_before,
        history_length_after=length_after,
        history_maxlen=maxlen,
        oldest_snapshot_evicted=(
            length_before == maxlen and length_after == maxlen
        ),
        snapshot_items=snapshot_items,
    )
