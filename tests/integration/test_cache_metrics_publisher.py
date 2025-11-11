"""Integration coverage for cache telemetry publishers."""

from __future__ import annotations

import logging

import networkx as nx
import pytest

from tnfr.utils.callbacks import CallbackEvent, callback_manager
from tnfr.constants import inject_defaults
from tnfr.telemetry import publish_graph_cache_metrics
from tnfr.utils import edge_version_cache


@pytest.mark.integration
def test_publish_graph_cache_metrics_emits_edge_cache_snapshots(
    caplog: pytest.LogCaptureFixture,
) -> None:
    G = nx.Graph()
    inject_defaults(G)
    events: list[dict[str, object]] = []

    def recorder(graph, ctx):
        events.append(dict(ctx))

    callback_manager.register_callback(
        G,
        CallbackEvent.CACHE_METRICS,
        recorder,
        name="capture_cache_metrics",
    )

    build_calls = {"count": 0}

    def builder():
        build_calls["count"] += 1
        return {"payload": build_calls["count"]}

    first = edge_version_cache(G, "alpha", builder)
    second = edge_version_cache(G, "alpha", builder)

    assert first == second == {"payload": 1}
    assert build_calls["count"] == 1, "Second access should hit the cache"

    with caplog.at_level(logging.INFO, logger="tnfr.telemetry.cache"):
        publish_graph_cache_metrics(G)

    assert events, "Cache metrics callbacks must be invoked"
    matching = [ctx for ctx in events if ctx.get("cache") == "_edge_version_state"]
    assert matching, "edge_version_cache metrics should be published"

    payload = matching[-1]["metrics"]
    assert isinstance(payload, dict)
    assert payload["hits"] >= 1
    assert payload["misses"] >= 1
    assert payload["hit_ratio"] is not None
    assert payload["avg_latency"] is not None and payload["avg_latency"] >= 0.0

    assert any("_edge_version_state" in record.message for record in caplog.records)
