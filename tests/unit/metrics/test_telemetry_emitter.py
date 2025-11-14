from __future__ import annotations

import json
import networkx as nx

from tnfr.metrics.telemetry import TelemetryEmitter


def _graph():
    G = nx.erdos_renyi_graph(8, 0.25)
    for n in G.nodes:
        G.nodes[n]["phase"] = 0.1 * n
        G.nodes[n]["delta_nfr"] = 0.01 * (n + 1)
    return G


def test_telemetry_emitter_basic_flow(tmp_path):
    path = tmp_path / "telemetry.jsonl"
    G = _graph()
    with TelemetryEmitter(str(path)) as emitter:
        emitter.record(
            G,
            operator="start",
            extra={"nodes": G.number_of_nodes()},
        )
        emitter.record(G, operator="step", extra={"note": "test"})
        emitter.flush()
    contents = path.read_text(encoding="utf-8").splitlines()
    assert len(contents) >= 2
    assert any('"operator": "start"' in ln for ln in contents)


def test_telemetry_emitter_utc_timestamps(tmp_path):
    """Ensure timestamps are timezone-aware UTC (not deprecated utcnow)."""
    from datetime import datetime

    path = tmp_path / "telemetry_utc.jsonl"
    G = _graph()
    with TelemetryEmitter(str(path)) as emitter:
        emitter.record(G, operator="test_utc")
    contents = path.read_text(encoding="utf-8").splitlines()
    assert len(contents) >= 1
    event = json.loads(contents[0])
    # Check t_iso is parseable and has timezone info
    ts = datetime.fromisoformat(event["t_iso"])
    assert ts.tzinfo is not None, "Timestamp should be timezone-aware"
