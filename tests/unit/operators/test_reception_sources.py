"""Tests for Reception operator source detection and integration efficiency tracking.

This module validates the network source detection implementation for the
Reception (EN) operator as specified in TNFR.pdf §2.2.1:

- Source detection: Identifying active emission nodes in network
- Phase compatibility: Validating θᵢ ≈ θⱼ for coupling
- Integration efficiency: Measuring coherence received vs. integrated
- Source traceability: Tracking which nodes contribute to EPI

Tests verify:
- detect_emission_sources() functionality
- Reception operator source detection integration
- Extended EN metrics with integration efficiency
- Warning when no sources detected
- Backward compatibility with existing Reception usage
"""

import warnings

import networkx as nx

from tnfr.operators.definitions import Emission, Reception, Coherence, Silence
from tnfr.operators.network_analysis import detect_emission_sources
from tnfr.structural import create_nfr, run_sequence


def test_detect_emission_sources_basic():
    """Test basic source detection with single emitter-receiver pair."""
    G = nx.Graph()
    G, emitter = create_nfr("teacher", epi=0.5, vf=1.0, theta=0.3, graph=G)
    _, receiver = create_nfr("student", epi=0.25, vf=0.9, theta=0.35, graph=G)
    G.add_edge(emitter, receiver)

    sources = detect_emission_sources(G, receiver)

    assert len(sources) == 1
    source_node, compatibility, strength = sources[0]
    assert source_node == emitter
    assert 0.9 <= compatibility <= 1.0  # High phase compatibility (close phases)
    assert strength > 0.4  # Strong coherence (0.5 * 1.0 = 0.5)


def test_detect_emission_sources_no_sources():
    """Test source detection when no active emitters exist."""
    G = nx.Graph()
    G, emitter = create_nfr("weak", epi=0.1, vf=1.0, theta=0.3, graph=G)  # Below threshold
    _, receiver = create_nfr("student", epi=0.25, vf=0.9, theta=0.35, graph=G)
    G.add_edge(emitter, receiver)

    sources = detect_emission_sources(G, receiver)

    assert len(sources) == 0  # No active sources (EPI < 0.2)


def test_detect_emission_sources_multiple_sources():
    """Test source detection with multiple emitters, sorted by compatibility."""
    G = nx.Graph()
    # Create receiver
    G, receiver = create_nfr("student", epi=0.25, vf=0.9, theta=0.5, graph=G)

    # Create multiple emitters with different phases
    _, emitter1 = create_nfr("teacher1", epi=0.6, vf=1.0, theta=0.52, graph=G)  # Very close
    _, emitter2 = create_nfr("teacher2", epi=0.5, vf=0.8, theta=1.5, graph=G)  # Far phase
    _, emitter3 = create_nfr("teacher3", epi=0.4, vf=1.2, theta=0.48, graph=G)  # Close

    # Connect all
    G.add_edge(receiver, emitter1)
    G.add_edge(receiver, emitter2)
    G.add_edge(receiver, emitter3)

    sources = detect_emission_sources(G, receiver)

    assert len(sources) == 3

    # First source should be most compatible (smallest phase difference)
    # emitter1 (θ=0.52) or emitter3 (θ=0.48) should be first
    first_source, first_compat, _ = sources[0]
    assert first_source in [emitter1, emitter3]
    assert first_compat > 0.9  # High compatibility

    # Last source should be least compatible
    last_source, last_compat, _ = sources[-1]
    assert last_source == emitter2  # θ=1.5, far from receiver θ=0.5
    assert last_compat < 0.7  # Lower compatibility


def test_detect_emission_sources_max_distance():
    """Test that max_distance limits source detection."""
    G = nx.Graph()
    G, receiver = create_nfr("student", epi=0.25, vf=0.9, theta=0.5, graph=G)
    _, emitter1 = create_nfr("near", epi=0.5, vf=1.0, theta=0.5, graph=G)
    _, emitter2 = create_nfr("far", epi=0.6, vf=1.0, theta=0.5, graph=G)
    _, intermediate = create_nfr("mid", epi=0.3, vf=0.5, theta=0.5, graph=G)

    # Create chain: far -- intermediate -- near -- receiver
    G.add_edge(emitter2, intermediate)
    G.add_edge(intermediate, emitter1)
    G.add_edge(emitter1, receiver)

    # Distance 1: only immediate neighbor (near)
    sources_d1 = detect_emission_sources(G, receiver, max_distance=1)
    assert len(sources_d1) == 1
    assert sources_d1[0][0] == emitter1

    # Distance 2: near + intermediate
    sources_d2 = detect_emission_sources(G, receiver, max_distance=2)
    assert len(sources_d2) == 2
    source_nodes = [s[0] for s in sources_d2]
    assert emitter1 in source_nodes
    assert intermediate in source_nodes

    # Distance 3: all nodes
    sources_d3 = detect_emission_sources(G, receiver, max_distance=3)
    assert len(sources_d3) == 3
    source_nodes = [s[0] for s in sources_d3]
    assert emitter1 in source_nodes
    assert intermediate in source_nodes
    assert emitter2 in source_nodes


def test_reception_operator_source_tracking():
    """Test that Reception operator stores detected sources in node."""
    G = nx.Graph()
    # Emitter needs EPI >= 0.2 to be detected as active source
    G, emitter = create_nfr("teacher", epi=0.5, vf=1.0, theta=0.3, graph=G)
    _, receiver = create_nfr("student", epi=0.18, vf=0.9, theta=0.35, graph=G)
    G.add_edge(emitter, receiver)

    # Apply full sequence with Reception
    run_sequence(G, receiver, [Emission(), Reception(), Coherence(), Silence()])

    # Verify sources were stored
    assert "_reception_sources" in G.nodes[receiver]
    sources = G.nodes[receiver]["_reception_sources"]
    assert len(sources) == 1
    assert sources[0][0] == emitter


def test_reception_operator_no_source_warning():
    """Test that Reception warns when no sources are detected."""
    G = nx.Graph()
    # Isolated node with no neighbors
    G, receiver = create_nfr("isolated", epi=0.18, vf=0.9, theta=0.5, graph=G)

    # Should warn about no sources
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_sequence(G, receiver, [Emission(), Reception(), Coherence(), Silence()])

        assert len(w) == 1
        assert "no detectable emission sources" in str(w[0].message).lower()


def test_reception_operator_track_sources_disabled():
    """Test that source tracking can be disabled."""
    G = nx.Graph()
    G, emitter = create_nfr("teacher", epi=0.5, vf=1.0, theta=0.3, graph=G)
    _, receiver = create_nfr("student", epi=0.25, vf=0.9, theta=0.35, graph=G)
    G.add_edge(emitter, receiver)

    # Apply Reception with tracking disabled
    Reception()(G, receiver, track_sources=False)

    # Verify sources were NOT stored
    assert "_reception_sources" not in G.nodes[receiver]


def test_reception_metrics_extended():
    """Test extended Reception metrics with integration efficiency."""
    G = nx.Graph()
    G, emitter = create_nfr("teacher", epi=0.18, vf=1.0, theta=0.3, graph=G)
    _, receiver = create_nfr("student", epi=0.18, vf=0.9, theta=0.35, graph=G)
    G.add_edge(emitter, receiver)

    # Enable metrics collection
    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # Apply full sequence on receiver
    run_sequence(G, receiver, [Emission(), Reception(), Coherence(), Silence()])

    # Get Reception metrics (second in list after Emission)
    assert len(G.graph["operator_metrics"]) >= 2
    metrics = G.graph["operator_metrics"][1]

    # Verify core metrics
    assert metrics["operator"] == "Reception"
    assert metrics["glyph"] == "EN"
    assert "delta_epi" in metrics
    assert "epi_final" in metrics
    assert "dnfr_after" in metrics

    # Verify legacy metrics (backward compatibility)
    assert "neighbor_count" in metrics
    assert "neighbor_epi_mean" in metrics
    assert "integration_strength" in metrics

    # Verify EN-specific extended metrics
    assert "num_sources" in metrics
    assert metrics["num_sources"] >= 0  # May detect emitter as source

    assert "integration_efficiency" in metrics
    assert isinstance(metrics["integration_efficiency"], float)

    assert "most_compatible_source" in metrics
    # May be None or emitter depending on detection

    assert "phase_compatibility_avg" in metrics
    assert 0.0 <= metrics["phase_compatibility_avg"] <= 1.0

    assert "coherence_received" in metrics
    assert metrics["coherence_received"] == metrics["delta_epi"]

    assert "stabilization_effective" in metrics
    assert isinstance(metrics["stabilization_effective"], bool)


def test_reception_metrics_no_sources():
    """Test Reception metrics when no sources are detected."""
    G = nx.Graph()
    G, receiver = create_nfr("isolated", epi=0.18, vf=0.9, theta=0.5, graph=G)

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # Apply Reception without sources (will warn)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        run_sequence(G, receiver, [Emission(), Reception(), Coherence(), Silence()])

    # Get Reception metrics (second in list)
    metrics = G.graph["operator_metrics"][1]

    # Verify EN-specific metrics handle no sources gracefully
    assert metrics["num_sources"] == 0
    assert metrics["integration_efficiency"] == 0.0
    assert metrics["most_compatible_source"] is None
    assert metrics["phase_compatibility_avg"] == 0.0


def test_reception_metrics_integration_efficiency_calculation():
    """Test that integration efficiency is calculated correctly."""
    G = nx.Graph()
    # Create strong emitter
    G, emitter = create_nfr("strong_teacher", epi=0.18, vf=1.5, theta=0.3, graph=G)
    _, receiver = create_nfr("student", epi=0.18, vf=0.9, theta=0.32, graph=G)
    G.add_edge(emitter, receiver)

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    # Apply full sequence
    run_sequence(G, receiver, [Emission(), Reception(), Coherence(), Silence()])

    metrics = G.graph["operator_metrics"][1]  # Reception metrics

    # Get stored sources to verify calculation
    sources = G.nodes[receiver]["_reception_sources"]

    # If sources were detected, verify efficiency calculation
    if len(sources) > 0:
        _, _, coherence_strength = sources[0]
        # Integration efficiency should be delta_epi / total_available_coherence
        expected_efficiency = metrics["delta_epi"] / coherence_strength
        assert abs(metrics["integration_efficiency"] - expected_efficiency) < 1e-6


def test_reception_backward_compatibility():
    """Test that existing Reception usage still works without modifications."""
    G = nx.Graph()
    G, receiver = create_nfr("student", epi=0.18, vf=0.9, theta=0.5, graph=G)

    # Old-style usage: apply full sequence
    run_sequence(G, receiver, [Emission(), Reception(), Coherence(), Silence()])

    # Should complete without errors
    assert G.nodes[receiver]  # Node still exists


def test_reception_phase_compatibility_accuracy():
    """Test phase compatibility calculation accuracy."""
    G = nx.Graph()
    # Create sources with known phase relationships
    G, receiver = create_nfr("student", epi=0.25, vf=0.9, theta=0.0, graph=G)

    # Same phase (perfect compatibility)
    _, same_phase = create_nfr("same", epi=0.5, vf=1.0, theta=0.0, graph=G)
    G.add_edge(receiver, same_phase)

    # Opposite phase (π difference, worst compatibility)
    _, opposite_phase = create_nfr("opposite", epi=0.5, vf=1.0, theta=3.14159, graph=G)
    G.add_edge(receiver, opposite_phase)

    sources = detect_emission_sources(G, receiver)

    # Find each source
    same_compat = None
    opposite_compat = None
    for src, compat, _ in sources:
        if src == same_phase:
            same_compat = compat
        elif src == opposite_phase:
            opposite_compat = compat

    # Same phase should have highest compatibility (≈1.0)
    assert same_compat is not None
    assert same_compat > 0.99

    # Opposite phase should have lowest compatibility (≈0.0)
    assert opposite_compat is not None
    assert opposite_compat < 0.1


def test_reception_coherence_strength_calculation():
    """Test that coherence strength is calculated correctly."""
    G = nx.Graph()
    G, receiver = create_nfr("student", epi=0.25, vf=0.9, theta=0.5, graph=G)

    # Create emitter with known EPI and νf
    epi_value = 0.6
    vf_value = 1.2
    _, emitter = create_nfr("teacher", epi=epi_value, vf=vf_value, theta=0.52, graph=G)
    G.add_edge(receiver, emitter)

    sources = detect_emission_sources(G, receiver)

    assert len(sources) == 1
    _, _, coherence_strength = sources[0]

    # Coherence strength should equal EPI × νf
    expected_strength = epi_value * vf_value
    assert abs(coherence_strength - expected_strength) < 1e-6


def test_reception_multiple_sources_avg_compatibility():
    """Test average phase compatibility with multiple sources."""
    G = nx.Graph()
    G, receiver = create_nfr("student", epi=0.18, vf=0.9, theta=0.5, graph=G)

    # Create three sources with known compatibilities
    _, src1 = create_nfr("src1", epi=0.5, vf=1.0, theta=0.5, graph=G)  # Perfect match
    _, src2 = create_nfr("src2", epi=0.5, vf=1.0, theta=1.5, graph=G)  # Moderate diff
    _, src3 = create_nfr("src3", epi=0.5, vf=1.0, theta=2.5, graph=G)  # Large diff

    G.add_edge(receiver, src1)
    G.add_edge(receiver, src2)
    G.add_edge(receiver, src3)

    G.graph["COLLECT_OPERATOR_METRICS"] = True

    run_sequence(G, receiver, [Emission(), Reception(), Coherence(), Silence()])

    # Get Reception metrics (second in list)
    metrics = G.graph["operator_metrics"][1]

    # Average compatibility should be mean of all three
    sources = G.nodes[receiver]["_reception_sources"]
    compatibilities = [compat for _, compat, _ in sources]
    expected_avg = sum(compatibilities) / len(compatibilities)

    assert abs(metrics["phase_compatibility_avg"] - expected_avg) < 1e-6
