"""Tests for tetrad snapshot collection with telemetry density modes.

Validates:
- Snapshot structure and completeness
- Telemetry density modes (low/medium/high)
- Physics invariance (read-only, no operator changes)
- Sample interval computation
"""

import pytest
import networkx as nx
import numpy as np

from tnfr.config import set_telemetry_density
from tnfr.metrics.tetrad import (
    collect_tetrad_snapshot,
    get_tetrad_sample_interval,
)


@pytest.fixture
def reset_config():
    """Reset config to defaults after each test."""
    yield
    set_telemetry_density("low")


@pytest.fixture
def simple_tnfr_graph():
    """Create a simple TNFR graph with required attributes."""
    G = nx.watts_strogatz_graph(20, 4, 0.3, seed=42)
    
    # Add TNFR attributes
    for i in G.nodes():
        G.nodes[i]["ΔNFR"] = np.random.uniform(-1, 1)
        G.nodes[i]["theta"] = np.random.uniform(0, 2 * np.pi)
        G.nodes[i]["νf"] = 1.0
    
    return G


def test_snapshot_basic_structure(simple_tnfr_graph, reset_config):
    """Snapshot contains all required tetrad fields."""
    set_telemetry_density("low")
    
    snapshot = collect_tetrad_snapshot(simple_tnfr_graph)
    
    # Required top-level keys
    assert "phi_s" in snapshot
    assert "phase_grad" in snapshot
    assert "phase_curv" in snapshot
    assert "xi_c" in snapshot
    assert "metadata" in snapshot
    
    # Metadata
    assert snapshot["metadata"]["telemetry_density"] == "low"
    assert snapshot["metadata"]["node_count"] == 20


def test_snapshot_low_density_statistics(simple_tnfr_graph, reset_config):
    """Low density: mean, max, min, std only."""
    set_telemetry_density("low")
    
    snapshot = collect_tetrad_snapshot(simple_tnfr_graph)
    
    # Check phi_s statistics
    phi_s_stats = snapshot["phi_s"]
    assert "mean" in phi_s_stats
    assert "max" in phi_s_stats
    assert "min" in phi_s_stats
    assert "std" in phi_s_stats
    
    # Should NOT have percentiles
    assert "p25" not in phi_s_stats
    assert "p50" not in phi_s_stats
    assert "p75" not in phi_s_stats
    assert "histogram" not in phi_s_stats


def test_snapshot_medium_density_statistics(
    simple_tnfr_graph, reset_config
):
    """Medium density: adds quartiles (p25, p50, p75)."""
    set_telemetry_density("medium")
    
    snapshot = collect_tetrad_snapshot(simple_tnfr_graph)
    
    # Check phase_grad statistics
    grad_stats = snapshot["phase_grad"]
    assert "mean" in grad_stats
    assert "max" in grad_stats
    assert "min" in grad_stats
    assert "std" in grad_stats
    assert "p25" in grad_stats
    assert "p50" in grad_stats
    assert "p75" in grad_stats
    
    # Should NOT have tail percentiles or histograms
    assert "p10" not in grad_stats
    assert "p90" not in grad_stats
    assert "p99" not in grad_stats
    assert "histogram" not in grad_stats


def test_snapshot_high_density_statistics(simple_tnfr_graph, reset_config):
    """High density: adds tail percentiles (p10, p90, p99) + histograms."""
    set_telemetry_density("high")
    
    snapshot = collect_tetrad_snapshot(simple_tnfr_graph)
    
    # Check phase_curv statistics
    curv_stats = snapshot["phase_curv"]
    assert "mean" in curv_stats
    assert "max" in curv_stats
    assert "min" in curv_stats
    assert "std" in curv_stats
    assert "p25" in curv_stats
    assert "p50" in curv_stats
    assert "p75" in curv_stats
    assert "p10" in curv_stats
    assert "p90" in curv_stats
    assert "p99" in curv_stats
    
    # Should have histogram
    assert "histogram" in curv_stats
    assert "counts" in curv_stats["histogram"]
    assert "edges" in curv_stats["histogram"]
    assert len(curv_stats["histogram"]["counts"]) == 20  # 20 bins


def test_snapshot_histogram_override(simple_tnfr_graph, reset_config):
    """Histogram inclusion can be overridden explicitly."""
    set_telemetry_density("low")
    
    # Force histograms even in low density
    snapshot = collect_tetrad_snapshot(
        simple_tnfr_graph, include_histograms=True
    )
    
    phi_s_stats = snapshot["phi_s"]
    assert "histogram" in phi_s_stats
    
    # Force no histograms even in high density
    set_telemetry_density("high")
    snapshot = collect_tetrad_snapshot(
        simple_tnfr_graph, include_histograms=False
    )
    
    phi_s_stats = snapshot["phi_s"]
    assert "histogram" not in phi_s_stats


def test_snapshot_physics_invariance(simple_tnfr_graph, reset_config):
    """Snapshot collection does NOT modify graph attributes."""
    set_telemetry_density("high")
    
    # Store original attributes
    original_dnfr = {
        i: simple_tnfr_graph.nodes[i]["ΔNFR"]
        for i in simple_tnfr_graph.nodes()
    }
    original_theta = {
        i: simple_tnfr_graph.nodes[i]["theta"]
        for i in simple_tnfr_graph.nodes()
    }
    
    # Collect snapshot
    snapshot = collect_tetrad_snapshot(simple_tnfr_graph)
    
    # Verify attributes unchanged
    for i in simple_tnfr_graph.nodes():
        assert (
            simple_tnfr_graph.nodes[i]["ΔNFR"] == original_dnfr[i]
        ), f"ΔNFR modified for node {i}"
        assert (
            simple_tnfr_graph.nodes[i]["theta"] == original_theta[i]
        ), f"theta modified for node {i}"
    
    # Verify snapshot contains data
    assert snapshot["phi_s"]["mean"] is not None
    assert snapshot["phase_grad"]["mean"] is not None


def test_snapshot_density_consistency(simple_tnfr_graph, reset_config):
    """Same graph, different densities: values consistent, detail varies."""
    # Collect snapshots at all densities
    set_telemetry_density("low")
    snap_low = collect_tetrad_snapshot(simple_tnfr_graph)
    
    set_telemetry_density("medium")
    snap_medium = collect_tetrad_snapshot(simple_tnfr_graph)
    
    set_telemetry_density("high")
    snap_high = collect_tetrad_snapshot(simple_tnfr_graph)
    
    # Core statistics should be identical (same graph)
    for field in ["phi_s", "phase_grad", "phase_curv"]:
        assert np.isclose(
            snap_low[field]["mean"], snap_medium[field]["mean"], rtol=1e-10
        )
        assert np.isclose(
            snap_low[field]["mean"], snap_high[field]["mean"], rtol=1e-10
        )
        assert np.isclose(
            snap_low[field]["max"], snap_medium[field]["max"], rtol=1e-10
        )
        assert np.isclose(
            snap_low[field]["max"], snap_high[field]["max"], rtol=1e-10
        )
    
    # Detail increases with density
    assert "p25" not in snap_low["phi_s"]
    assert "p25" in snap_medium["phi_s"]
    assert "p25" in snap_high["phi_s"]
    
    assert "p90" not in snap_medium["phase_grad"]
    assert "p90" in snap_high["phase_grad"]


def test_sample_interval_computation(reset_config):
    """Sample interval scales with telemetry density."""
    base_dt = 0.1
    
    set_telemetry_density("low")
    assert get_tetrad_sample_interval(base_dt) == 10.0 * base_dt
    
    set_telemetry_density("medium")
    assert get_tetrad_sample_interval(base_dt) == 5.0 * base_dt
    
    set_telemetry_density("high")
    assert get_tetrad_sample_interval(base_dt) == 1.0 * base_dt


def test_snapshot_empty_graph(reset_config):
    """Snapshot handles empty graph gracefully."""
    G = nx.Graph()
    
    snapshot = collect_tetrad_snapshot(G)
    
    assert snapshot["phi_s"]["mean"] is None
    assert snapshot["phase_grad"]["mean"] is None
    assert snapshot["phase_curv"]["mean"] is None
    assert snapshot["xi_c"] is None
    assert snapshot["metadata"]["node_count"] == 0


def test_snapshot_isolated_nodes(reset_config):
    """Snapshot handles graphs with isolated nodes."""
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])  # No edges
    
    for i in G.nodes():
        G.nodes[i]["ΔNFR"] = 0.5
        G.nodes[i]["theta"] = 0.0
        G.nodes[i]["νf"] = 1.0
    
    snapshot = collect_tetrad_snapshot(G)
    
    # Should still compute statistics
    assert snapshot["phi_s"]["mean"] is not None
    # Isolated nodes have zero phase gradient
    assert snapshot["phase_grad"]["mean"] == 0.0
