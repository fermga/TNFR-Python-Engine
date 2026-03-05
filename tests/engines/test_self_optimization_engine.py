"""Tests for TNFR self-optimization dry-run support."""

import json
from pathlib import Path

import networkx as nx
import pytest

from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine


def _read_signature(signature_path: Path) -> str:
    text = signature_path.read_text(encoding="utf-8").strip()
    # Signature lines follow the conventional "hash  filename" format.
    return text.split()[0]


def test_optimize_automatically_dry_run_creates_snapshot(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_edge(1, 2)

    engine = TNFRSelfOptimizingEngine()

    result = engine.optimize_automatically(
        graph,
        "diagnostic",
        dry_run=True,
        seed=42,
        node="alpha beta",
        operator_sequence=["AL", "UM", "IL", "SHA"],
        output_dir=tmp_path,
    )

    assert result["dry_run"] is True
    assert result["learning_updated"] is False
    assert result["telemetry_snapshots"] is not None

    snapshot_path = Path(result["snapshot_path"])
    assert snapshot_path.exists()

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["operation_type"] == "diagnostic"
    assert payload["metadata"]["seed"] == "42"
    assert payload["metadata"]["node"].startswith("alpha")
    assert payload["validation"]["passed"] is True

    signature_path = snapshot_path.with_suffix(snapshot_path.suffix + ".sha256")
    assert signature_path.exists()
    assert result["signature"] == _read_signature(signature_path)


def test_optimize_automatically_dry_run_requires_valid_sequence(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_node(1)

    engine = TNFRSelfOptimizingEngine()

    with pytest.raises(ValueError):
        engine.optimize_automatically(
            graph,
            "diagnostic",
            dry_run=True,
            operator_sequence=["UM"],  # coupling without generator violates grammar
            output_dir=tmp_path,
        )
