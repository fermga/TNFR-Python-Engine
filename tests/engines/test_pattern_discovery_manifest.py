"""Tests for TNFREmergentPatternEngine manifest export and self-optimization integration."""

import json
import tempfile
from pathlib import Path

import networkx as nx
import pytest

from tnfr.engines.pattern_discovery import TNFREmergentPatternEngine


@pytest.fixture
def sample_network():
    """Create a small TNFR network for testing."""
    G = nx.karate_club_graph()
    for node in G.nodes():
        G.nodes[node]["vf"] = 1.0 + 0.1 * node
        G.nodes[node]["phase"] = 0.0
        G.nodes[node]["EPI"] = 1.0
    return G


def test_export_pattern_manifest_creates_files(sample_network):
    """Test that export_pattern_manifest creates manifest and summary files."""
    engine = TNFREmergentPatternEngine()
    discovery_result = engine.discover_all_patterns(sample_network)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result = engine.export_pattern_manifest(
            G=sample_network,
            discovery_result=discovery_result,
            output_dir=output_dir,
            partition_id="test_partition_001",
        )

        # Verify return structure
        assert "manifest_absolute" in result
        assert "summary_absolute" in result
        assert result["manifest_absolute"].exists()
        assert result["summary_absolute"].exists()


def test_manifest_structure_is_valid(sample_network):
    """Test that exported manifest has correct structure."""
    engine = TNFREmergentPatternEngine()
    discovery_result = engine.discover_all_patterns(sample_network)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result = engine.export_pattern_manifest(
            G=sample_network,
            discovery_result=discovery_result,
            output_dir=output_dir,
            partition_id="test_partition_002",
        )

        # Load and validate manifest
        with open(result["manifest_absolute"]) as f:
            manifest = json.load(f)

        # Check required fields
        assert manifest["operation_type"] == "pattern_discovery"
        assert manifest["partition_id"] == "test_partition_002"
        assert "timestamp" in manifest
        assert "network_metadata" in manifest
        assert "telemetry" in manifest
        assert "discovered_patterns" in manifest
        assert "discovery_statistics" in manifest

        # Check network metadata
        assert manifest["network_metadata"]["node_count"] == len(sample_network.nodes())
        assert manifest["network_metadata"]["edge_count"] == len(sample_network.edges())

        # Check patterns are serialized
        assert isinstance(manifest["discovered_patterns"], list)


def test_summary_structure_is_valid(sample_network):
    """Test that exported summary has correct structure."""
    engine = TNFREmergentPatternEngine()
    discovery_result = engine.discover_all_patterns(sample_network)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result = engine.export_pattern_manifest(
            G=sample_network,
            discovery_result=discovery_result,
            output_dir=output_dir,
            partition_id="test_partition_003",
        )

        # Load and validate summary
        with open(result["summary_absolute"]) as f:
            summary = json.load(f)

        # Check required fields
        assert summary["operation_type"] == "pattern_discovery"
        assert summary["partition_id"] == "test_partition_003"
        assert "pattern_count" in summary
        assert "coherence" in summary
        assert "sense_index" in summary
        assert "compression_potential" in summary
        assert "predictive_accuracy" in summary


def test_sdk_wrapper_accepts_pattern_discovery_manifests(sample_network, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "factorization-lab"))
    """Test that SDK wrapper can process pattern discovery manifests."""
    engine = TNFREmergentPatternEngine()
    discovery_result = engine.discover_all_patterns(sample_network)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result = engine.export_pattern_manifest(
            G=sample_network,
            discovery_result=discovery_result,
            output_dir=output_dir,
            partition_id="test_partition_004",
        )

        # Import SDK function
        from tnfr.sdk import run_pattern_discovery_optimization

        opt_result = run_pattern_discovery_optimization(
            manifest_path=result["manifest_absolute"],
            manifest_summary_path=result["summary_absolute"],
            base_name="test_opt",
            output_root=output_dir / "optimization",
        )
        assert opt_result is not None
        runner = opt_result["runner_summary"]
        assert runner["success_count"] == 1
        assert runner["failure_count"] == 0
        assert opt_result["promotable"] == {}
        for entry in runner["partition_results"]:
            assert entry["telemetry_deltas"]["delta_c"] == 0.0



def test_manifest_telemetry_includes_coherence(sample_network):
    """Test that telemetry includes coherence and sense_index."""
    engine = TNFREmergentPatternEngine()
    discovery_result = engine.discover_all_patterns(sample_network)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result = engine.export_pattern_manifest(
            G=sample_network,
            discovery_result=discovery_result,
            output_dir=output_dir,
            partition_id="test_partition_005",
        )

        with open(result["manifest_absolute"]) as f:
            manifest = json.load(f)

        # Check telemetry fields
        assert "telemetry" in manifest
        telemetry = manifest["telemetry"]

        # Coherence and sense_index should be present (may be None if physics unavailable)
        assert "coherence" in telemetry
        assert "sense_index" in telemetry
        assert isinstance(telemetry["coherence"], float)
        assert isinstance(telemetry["sense_index"], float)
