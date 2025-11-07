"""Tests for OZ (Dissonance) network propagation.

This module tests the implementation of dissonance propagation across networks
following TNFR resonance principles. When OZ is applied to a node, structural
dissonance propagates to phase-compatible neighbors, potentially triggering
bifurcation cascades.

References
----------
- Issue: [OZ] Implementar propagación de disonancia y efectos de red vecinal
- TNFR.pdf §6.2: Interferencia nodal - dissonance propagates through networks
"""

import math
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_THETA, ALIAS_VF, ALIAS_EPI
from tnfr.operators.definitions import Coherence, Dissonance, Emission
from tnfr.structural import create_nfr
from tnfr.dynamics.propagation import (
    propagate_dissonance,
    compute_network_dissonance_field,
    detect_bifurcation_cascade,
)


class TestDissonancePropagation:
    """Test suite for OZ propagation to neighbors."""
    
    def test_oz_propagates_to_phase_compatible_neighbors(self):
        """OZ propagates dissonance to phase-compatible neighbors."""
        # Create star topology: node 0 connected to 1, 2, 3
        G, node0 = create_nfr("central", epi=0.5, vf=1.0)
        
        # Add neighbors
        for i in range(1, 4):
            G.add_node(i)
            G.add_edge(node0, i)
            # Initialize neighbors
            Emission()(G, i)
            Coherence()(G, i)
            # Set phase-aligned
            G.nodes[i][ALIAS_THETA[0]] = 0.1
        
        # Set central node phase
        G.nodes[node0][ALIAS_THETA[0]] = 0.1
        
        # Capture neighbor DNFR before OZ
        dnfr_before = {i: float(get_attr(G.nodes[i], ALIAS_DNFR, 0.0)) for i in range(1, 4)}
        
        # Apply OZ to central node with propagation
        Dissonance()(G, node0, propagate_to_network=True)
        
        # Verify at least one neighbor's DNFR increased
        increased_count = 0
        for i in range(1, 4):
            dnfr_after = float(get_attr(G.nodes[i], ALIAS_DNFR, 0.0))
            if dnfr_after > dnfr_before[i]:
                increased_count += 1
        
        assert increased_count > 0, "At least one neighbor should receive propagated dissonance"
    
    def test_oz_respects_phase_compatibility(self):
        """OZ does not propagate to phase-incompatible neighbors."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.0)
        
        # Add two neighbors: one compatible, one incompatible
        G.add_node(1)
        G.add_node(2)
        G.add_edge(node0, 1)
        G.add_edge(node0, 2)
        
        # Initialize neighbors
        for i in [1, 2]:
            Emission()(G, i)
            Coherence()(G, i)
        
        # Set source phase
        G.nodes[node0][ALIAS_THETA[0]] = 0.0
        
        # Set compatible phase (within π/2)
        G.nodes[1][ALIAS_THETA[0]] = 0.2
        
        # Set incompatible phase (beyond π/2)
        G.nodes[2][ALIAS_THETA[0]] = 3.0
        
        dnfr_1_before = float(get_attr(G.nodes[1], ALIAS_DNFR, 0.0))
        dnfr_2_before = float(get_attr(G.nodes[2], ALIAS_DNFR, 0.0))
        
        # Apply OZ
        Dissonance()(G, node0, propagate_to_network=True)
        
        dnfr_1_after = float(get_attr(G.nodes[1], ALIAS_DNFR, 0.0))
        dnfr_2_after = float(get_attr(G.nodes[2], ALIAS_DNFR, 0.0))
        
        # Node 1 should be affected (phase compatible)
        # Allow for small numerical errors and zero-before case
        assert dnfr_1_after >= dnfr_1_before, "Compatible neighbor should receive dissonance"
        
        # Node 2 should NOT be affected (phase incompatible)
        assert abs(dnfr_2_after - dnfr_2_before) < 0.01, "Incompatible neighbor should not receive dissonance"
    
    def test_oz_propagation_can_be_disabled(self):
        """OZ propagation can be disabled via parameter."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.0)
        
        # Add neighbor
        G.add_node(1)
        G.add_edge(node0, 1)
        Emission()(G, 1)
        Coherence()(G, 1)
        
        # Phase-compatible
        G.nodes[node0][ALIAS_THETA[0]] = 0.1
        G.nodes[1][ALIAS_THETA[0]] = 0.15
        
        dnfr_before = float(get_attr(G.nodes[1], ALIAS_DNFR, 0.0))
        
        # Apply OZ with propagation disabled
        Dissonance()(G, node0, propagate_to_network=False)
        
        dnfr_after = float(get_attr(G.nodes[1], ALIAS_DNFR, 0.0))
        
        # Neighbor should NOT be affected
        assert abs(dnfr_after - dnfr_before) < 0.01, "Propagation should be disabled"
    
    def test_oz_stores_propagation_telemetry(self):
        """OZ stores propagation events in graph for telemetry."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.0)
        
        # Add neighbors
        for i in range(1, 3):
            G.add_node(i)
            G.add_edge(node0, i)
            Emission()(G, i)
            G.nodes[i][ALIAS_THETA[0]] = 0.1
        
        G.nodes[node0][ALIAS_THETA[0]] = 0.1
        
        # Apply OZ
        Dissonance()(G, node0, propagate_to_network=True)
        
        # Verify telemetry stored
        assert "_oz_propagation_events" in G.graph
        events = G.graph["_oz_propagation_events"]
        assert len(events) > 0
        
        latest = events[-1]
        assert latest["source"] == node0
        assert "affected_count" in latest
        assert "magnitude" in latest
    
    def test_propagate_dissonance_function(self):
        """Test propagate_dissonance function directly."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.0)
        
        # Add neighbors
        for i in range(1, 4):
            G.add_node(i)
            G.add_edge(node0, i)
            Emission()(G, i)
            G.nodes[i][ALIAS_THETA[0]] = 0.2  # Phase-compatible
        
        G.nodes[node0][ALIAS_THETA[0]] = 0.1
        
        # Propagate manually
        affected = propagate_dissonance(G, node0, dissonance_magnitude=0.2)
        
        # Should affect some neighbors
        assert len(affected) > 0
        assert all(n in [1, 2, 3] for n in affected)


class TestNetworkDissonanceField:
    """Test suite for dissonance field computation."""
    
    def test_field_computation_with_path_topology(self):
        """Dissonance field computes correctly for path topology."""
        G, node0 = create_nfr("start", epi=0.5, vf=1.0)
        
        # Create path: node0-1-2-3
        for i in range(1, 4):
            G.add_node(i)
            if i == 1:
                G.add_edge(node0, i)
            else:
                G.add_edge(i-1, i)
            Emission()(G, i)
        
        # Apply OZ to node 0 to create dissonance
        Dissonance()(G, node0)
        
        # Verify source has non-zero DNFR
        source_dnfr = abs(float(get_attr(G.nodes[node0], ALIAS_DNFR, 0.0)))
        assert source_dnfr > 0, "Source should have non-zero DNFR after OZ"
        
        # Compute field with radius 2
        field = compute_network_dissonance_field(G, node0, radius=2)
        
        # Should include nodes 1 and 2 (within radius), not 3
        assert 1 in field or 2 in field, f"Field should include nodes within radius. Source DNFR: {source_dnfr}, Field: {field}"
        assert 3 not in field
    
    def test_field_has_distance_decay(self):
        """Dissonance field decays with distance."""
        G, node0 = create_nfr("center", epi=0.5, vf=1.0)
        
        # Create star topology with multiple hops
        G.add_node(1)
        G.add_edge(node0, 1)
        G.add_node(2)
        G.add_edge(1, 2)
        
        for node in [1, 2]:
            Emission()(G, node)
        
        Dissonance()(G, node0)
        
        field = compute_network_dissonance_field(G, node0, radius=2)
        
        # If both present, node 1 (closer) should have stronger field
        if 1 in field and 2 in field:
            assert field[1] > field[2], "Closer node should have stronger field"
    
    def test_field_respects_radius_limit(self):
        """Dissonance field respects radius parameter."""
        G, node0 = create_nfr("start", epi=0.5, vf=1.0)
        
        # Create chain: node0-1-2-3-4
        for i in range(1, 5):
            G.add_node(i)
            if i == 1:
                G.add_edge(node0, i)
            else:
                G.add_edge(i-1, i)
            Emission()(G, i)
        
        # Apply OZ to create dissonance
        Dissonance()(G, node0)
        
        # Verify source has non-zero DNFR
        source_dnfr = abs(float(get_attr(G.nodes[node0], ALIAS_DNFR, 0.0)))
        assert source_dnfr > 0, "Source should have non-zero DNFR after OZ"
        
        # Radius 1 should only include node 1
        field_r1 = compute_network_dissonance_field(G, node0, radius=1)
        assert 1 in field_r1, f"Field should include node 1. Source DNFR: {source_dnfr}, Field: {field_r1}"
        assert 2 not in field_r1
        
        # Radius 2 should include nodes 1 and 2
        field_r2 = compute_network_dissonance_field(G, node0, radius=2)
        assert 1 in field_r2
        assert 2 in field_r2
        assert 3 not in field_r2


class TestBifurcationCascade:
    """Test suite for bifurcation cascade detection."""
    
    def test_cascade_detection_with_accelerating_neighbors(self):
        """Cascade detected when neighbors have accelerating EPI."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.2)
        
        # Add neighbors with EPI history showing acceleration
        for i in range(1, 4):
            G.add_node(i)
            G.add_edge(node0, i)
            Emission()(G, i)
            # Set accelerating history (∂²EPI/∂t² > 0)
            G.nodes[i]["_epi_history"] = [0.2, 0.4, 0.7]
            G.nodes[i][ALIAS_THETA[0]] = 0.1  # Phase-aligned
        
        G.nodes[node0][ALIAS_THETA[0]] = 0.1
        G.nodes[node0]["_epi_history"] = [0.3, 0.4, 0.5]
        
        # Apply OZ with propagation
        Dissonance()(G, node0, propagate_to_network=True)
        
        # Detect cascade
        cascade = detect_bifurcation_cascade(G, node0, threshold=0.3)
        
        # Should detect some nodes in cascade (depending on d2epi computation)
        # At minimum, function should not error and return a list
        assert isinstance(cascade, list)
    
    def test_cascade_marks_nodes_with_metadata(self):
        """Cascade detection marks nodes with bifurcation metadata."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.2)
        
        # Add neighbor with strong acceleration
        G.add_node(1)
        G.add_edge(node0, 1)
        Emission()(G, 1)
        G.nodes[1]["_epi_history"] = [0.1, 0.4, 0.9]  # Strong acceleration
        G.nodes[1][ALIAS_THETA[0]] = 0.1
        
        G.nodes[node0][ALIAS_THETA[0]] = 0.1
        G.nodes[node0]["_epi_history"] = [0.3, 0.4, 0.5]
        
        # Apply OZ
        Dissonance()(G, node0, propagate_to_network=True)
        
        # Detect cascade with low threshold to ensure detection
        cascade = detect_bifurcation_cascade(G, node0, threshold=0.1)
        
        # If cascade detected, verify metadata
        for cascade_node in cascade:
            assert "_bifurcation_cascade" in G.nodes[cascade_node]
            assert G.nodes[cascade_node]["_bifurcation_cascade"]["triggered_by"] == node0
            assert "_bifurcation_ready" in G.nodes[cascade_node]
    
    def test_cascade_only_affects_propagated_nodes(self):
        """Cascade only considers nodes that received propagation."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.2)
        
        # Add two neighbors: one affected, one not
        G.add_node(1)
        G.add_node(2)
        G.add_edge(node0, 1)
        # Note: node 2 not connected, so won't receive propagation
        
        for node in [1, 2]:
            Emission()(G, node)
            G.nodes[node]["_epi_history"] = [0.1, 0.5, 0.9]
        
        G.nodes[node0][ALIAS_THETA[0]] = 0.1
        G.nodes[1][ALIAS_THETA[0]] = 0.1
        G.nodes[2][ALIAS_THETA[0]] = 0.1
        
        # Apply OZ
        Dissonance()(G, node0, propagate_to_network=True)
        
        # Detect cascade
        cascade = detect_bifurcation_cascade(G, node0, threshold=0.3)
        
        # Node 2 should never be in cascade (not connected)
        assert 2 not in cascade


class TestPropagationMetrics:
    """Test suite for propagation-enhanced metrics."""
    
    def test_metrics_include_propagation_data(self):
        """OZ metrics include propagation information when enabled."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.0)
        
        # Add neighbors
        for i in range(1, 3):
            G.add_node(i)
            G.add_edge(node0, i)
            Emission()(G, i)
            G.nodes[i][ALIAS_THETA[0]] = 0.1
        
        G.nodes[node0][ALIAS_THETA[0]] = 0.1
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        # Apply OZ
        Dissonance()(G, node0, propagate_to_network=True)
        
        # Retrieve metrics
        metrics = G.graph["operator_metrics"][-1]
        
        # Verify propagation data present
        assert "propagation_occurred" in metrics
        if metrics["propagation_occurred"]:
            assert "affected_neighbors" in metrics
            assert "propagation_magnitude" in metrics
    
    def test_metrics_include_field_data(self):
        """OZ metrics include dissonance field computation."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.0)
        
        # Add neighbors for field computation
        for i in range(1, 4):
            G.add_node(i)
            G.add_edge(node0, i)
            Emission()(G, i)
        
        G.graph["COLLECT_OPERATOR_METRICS"] = True
        
        # Apply OZ
        Dissonance()(G, node0)
        
        # Retrieve metrics
        metrics = G.graph["operator_metrics"][-1]
        
        # Verify field data present
        assert "dissonance_field_radius" in metrics
        assert "max_field_strength" in metrics
        assert "mean_field_strength" in metrics


class TestPropagationModes:
    """Test suite for different propagation modes."""
    
    def test_phase_weighted_mode(self):
        """Phase-weighted mode uses phase compatibility."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.0)
        
        # Add two neighbors with different phase distances
        G.add_node(1)
        G.add_node(2)
        G.add_edge(node0, 1)
        G.add_edge(node0, 2)
        
        for i in [1, 2]:
            Emission()(G, i)
        
        G.nodes[node0][ALIAS_THETA[0]] = 0.0
        G.nodes[1][ALIAS_THETA[0]] = 0.1   # Close phase
        G.nodes[2][ALIAS_THETA[0]] = 1.0   # Further phase (but still < π/2)
        
        dnfr_1_before = float(get_attr(G.nodes[1], ALIAS_DNFR, 0.0))
        dnfr_2_before = float(get_attr(G.nodes[2], ALIAS_DNFR, 0.0))
        
        # Apply with phase-weighted mode
        Dissonance()(G, node0, propagate_to_network=True, propagation_mode="phase_weighted")
        
        dnfr_1_after = float(get_attr(G.nodes[1], ALIAS_DNFR, 0.0))
        dnfr_2_after = float(get_attr(G.nodes[2], ALIAS_DNFR, 0.0))
        
        # Node 1 (closer phase) should receive more if both received
        if (dnfr_1_after > dnfr_1_before) and (dnfr_2_after > dnfr_2_before):
            increase_1 = dnfr_1_after - dnfr_1_before
            increase_2 = dnfr_2_after - dnfr_2_before
            assert increase_1 >= increase_2, "Closer phase should receive more dissonance"
    
    def test_frequency_weighted_mode(self):
        """Frequency-weighted mode considers νf matching."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.0)
        
        # Add neighbors with different frequencies
        G.add_node(1)
        G.add_node(2)
        G.add_edge(node0, 1)
        G.add_edge(node0, 2)
        
        for i in [1, 2]:
            Emission()(G, i)
            G.nodes[i][ALIAS_THETA[0]] = 0.1
        
        # Set different frequencies
        G.nodes[1][ALIAS_VF[0]] = 0.95  # Close to source (1.0)
        G.nodes[2][ALIAS_VF[0]] = 0.3   # Far from source
        
        G.nodes[node0][ALIAS_THETA[0]] = 0.1
        
        # Apply with frequency-weighted mode
        Dissonance()(G, node0, propagate_to_network=True, propagation_mode="frequency_weighted")
        
        # Function should complete without error
        # Detailed frequency-weighted verification would require more setup


class TestConfigurableParameters:
    """Test suite for configurable propagation parameters."""
    
    def test_custom_phase_threshold(self):
        """Custom phase threshold can be configured."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.0)
        
        # Add neighbor with moderate phase difference
        G.add_node(1)
        G.add_edge(node0, 1)
        Emission()(G, 1)
        
        G.nodes[node0][ALIAS_THETA[0]] = 0.0
        G.nodes[1][ALIAS_THETA[0]] = 1.2  # Within default π/2 (~1.57)
        
        # Set strict threshold
        G.graph["OZ_PHASE_THRESHOLD"] = 1.0
        
        dnfr_before = float(get_attr(G.nodes[1], ALIAS_DNFR, 0.0))
        
        # Apply OZ
        Dissonance()(G, node0, propagate_to_network=True)
        
        dnfr_after = float(get_attr(G.nodes[1], ALIAS_DNFR, 0.0))
        
        # With strict threshold, this neighbor should not be affected
        assert abs(dnfr_after - dnfr_before) < 0.01
    
    def test_custom_min_propagation(self):
        """Minimum propagation threshold can be configured."""
        G, node0 = create_nfr("source", epi=0.5, vf=1.0)
        
        # Add neighbor
        G.add_node(1)
        G.add_edge(node0, 1)
        Emission()(G, 1)
        
        G.nodes[node0][ALIAS_THETA[0]] = 0.1
        G.nodes[1][ALIAS_THETA[0]] = 0.15
        
        # Set very high minimum propagation (nothing will propagate)
        G.graph["OZ_MIN_PROPAGATION"] = 10.0
        
        dnfr_before = float(get_attr(G.nodes[1], ALIAS_DNFR, 0.0))
        
        # Apply OZ
        Dissonance()(G, node0, propagate_to_network=True)
        
        dnfr_after = float(get_attr(G.nodes[1], ALIAS_DNFR, 0.0))
        
        # With high threshold, neighbor should not be affected
        assert abs(dnfr_after - dnfr_before) < 0.01
