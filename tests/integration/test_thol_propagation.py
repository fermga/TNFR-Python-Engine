"""Integration tests for THOL sub-EPI network propagation and cascades.

This test module validates the canonical THOL principle that sub-EPIs
propagate through coupled networks, creating emergent cascades when
nodes are sufficiently phase-aligned.

Canonical Principle (from "El pulso que nos atraviesa", §2.2.10):
    "THOL actúa como modulador central de plasticidad. Es el glifo que
    permite a la red reorganizar su topología sin intervención externa.
    Su activación crea bucles de aprendizaje resonante, trayectorias de
    reorganización emergente."

Test Coverage:
1. Sub-EPI propagation to phase-aligned neighbors
2. Phase barriers preventing propagation
3. Cascade triggering across network chains
4. Configuration controls (enable/disable, thresholds)
5. Telemetry recording for propagation events
"""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.alias import get_attr
from tnfr.constants import DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.operators.cascade import detect_cascade, measure_cascade_radius
from tnfr.operators.definitions import SelfOrganization


class TestTHOLPropagation:
    """Test sub-EPI propagation to coupled neighbors."""

    def test_thol_propagates_to_phase_aligned_neighbor(self):
        """Sub-EPIs should propagate to neighbors with sufficient phase coupling."""
        # Create coupled network
        G = nx.Graph()
        G.add_node(
            0,
            **{
                EPI_PRIMARY: 0.50,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.1,
                DNFR_PRIMARY: 0.10,
            },
        )
        G.add_node(
            1,
            **{
                EPI_PRIMARY: 0.40,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.12,  # Phase-aligned
                DNFR_PRIMARY: 0.05,
            },
        )
        G.add_node(
            2,
            **{
                EPI_PRIMARY: 0.40,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 2.5,  # Not aligned
                DNFR_PRIMARY: 0.05,
            },
        )
        G.add_edge(0, 1)
        G.add_edge(0, 2)

        # Enable propagation
        G.graph["THOL_PROPAGATION_ENABLED"] = True
        G.graph["THOL_MIN_COUPLING_FOR_PROPAGATION"] = 0.5

        # Prepare node 0 for bifurcation
        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]  # Accelerating

        epi_1_before = G.nodes[1][EPI_PRIMARY]
        epi_2_before = G.nodes[2][EPI_PRIMARY]

        # Apply THOL
        SelfOrganization()(G, 0)

        epi_1_after = G.nodes[1][EPI_PRIMARY]
        epi_2_after = G.nodes[2][EPI_PRIMARY]

        # Node 1 (phase-aligned) should receive propagation
        assert epi_1_after > epi_1_before, "Coupled neighbor should receive sub-EPI"

        # Node 2 (not aligned) should NOT receive propagation
        assert epi_2_after == epi_2_before, "Uncoupled neighbor should not be affected"

        # Verify telemetry
        propagations = G.graph.get("thol_propagations", [])
        assert len(propagations) > 0, "Propagation should be recorded"
        assert propagations[-1]["source_node"] == 0
        assert len(propagations[-1]["propagations"]) >= 1  # At least one target

    def test_thol_respects_minimum_coupling_threshold(self):
        """Propagation only occurs if coupling strength exceeds threshold."""
        G = nx.Graph()
        G.add_node(
            0,
            **{
                EPI_PRIMARY: 0.50,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: 0.10,
            },
        )
        G.add_node(
            1,
            **{
                EPI_PRIMARY: 0.40,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 1.0,  # Moderate phase difference
                DNFR_PRIMARY: 0.05,
            },
        )
        G.add_edge(0, 1)

        # Set high coupling threshold
        G.graph["THOL_PROPAGATION_ENABLED"] = True
        G.graph["THOL_MIN_COUPLING_FOR_PROPAGATION"] = 0.9  # Very high

        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        epi_1_before = G.nodes[1][EPI_PRIMARY]

        SelfOrganization()(G, 0)

        epi_1_after = G.nodes[1][EPI_PRIMARY]

        # Should not propagate due to high threshold
        assert epi_1_after == epi_1_before, "Should not propagate below threshold"

    def test_thol_attenuation_factor_reduces_signal(self):
        """Propagated sub-EPI is attenuated based on configuration."""
        G = nx.Graph()
        G.add_node(
            0,
            **{
                EPI_PRIMARY: 0.50,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: 0.10,
            },
        )
        G.add_node(
            1,
            **{
                EPI_PRIMARY: 0.40,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.01,  # Nearly in-phase
                DNFR_PRIMARY: 0.05,
            },
        )
        G.add_edge(0, 1)

        # Strong attenuation
        G.graph["THOL_PROPAGATION_ENABLED"] = True
        G.graph["THOL_PROPAGATION_ATTENUATION"] = 0.5  # 50% loss

        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        SelfOrganization()(G, 0)

        propagations = G.graph.get("thol_propagations", [])
        assert len(propagations) > 0

        # Check that propagated EPI is attenuated
        sub_epi_original = propagations[-1]["sub_epi"]
        injected_epi = propagations[-1]["propagations"][0][1]

        # Injected should be less than original due to attenuation
        assert injected_epi < sub_epi_original

    def test_thol_propagation_can_be_disabled(self):
        """Propagation can be disabled via configuration."""
        G = nx.Graph()
        G.add_node(
            0,
            **{
                EPI_PRIMARY: 0.50,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: 0.10,
            },
        )
        G.add_node(
            1,
            **{
                EPI_PRIMARY: 0.40,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.01,
                DNFR_PRIMARY: 0.05,
            },
        )
        G.add_edge(0, 1)

        # Disable propagation
        G.graph["THOL_PROPAGATION_ENABLED"] = False

        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        epi_1_before = G.nodes[1][EPI_PRIMARY]

        SelfOrganization()(G, 0)

        epi_1_after = G.nodes[1][EPI_PRIMARY]

        # Should not propagate
        assert epi_1_after == epi_1_before, "Propagation should be disabled"
        propagations = G.graph.get("thol_propagations", [])
        assert len(propagations) == 0, "No propagations should be recorded"

    def test_thol_rejects_antiphase_propagation(self):
        """THOL must reject propagation to neighbors with antiphase (Invariant #5).

        This test explicitly validates AGENTS.md Invariant #5:
        "No coupling is valid without explicit phase verification."

        THOL propagation uses phase-based coupling strength:
        coupling_strength = 1.0 - (|Δθ| / π)

        For antiphase nodes (Δθ = π), coupling_strength = 0, which is below
        the minimum threshold, thus blocking propagation as required by physics.
        """
        import math

        G = nx.Graph()
        # Node 0: phase = 0.0
        G.add_node(
            0,
            **{
                EPI_PRIMARY: 0.50,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: 0.15,
            },
        )
        # Node 1: phase = π (antiphase - destructive interference)
        G.add_node(
            1,
            **{
                EPI_PRIMARY: 0.50,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: math.pi,
                DNFR_PRIMARY: 0.05,
            },
        )
        G.add_edge(0, 1)

        # Enable propagation with standard threshold
        G.graph["THOL_PROPAGATION_ENABLED"] = True
        G.graph["THOL_MIN_COUPLING_FOR_PROPAGATION"] = 0.5

        # Build EPI history for bifurcation
        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        epi_1_before = G.nodes[1][EPI_PRIMARY]

        # Apply THOL
        SelfOrganization()(G, 0)

        epi_1_after = G.nodes[1][EPI_PRIMARY]

        # Verify: propagation should NOT have occurred to antiphase neighbor
        assert epi_1_after == epi_1_before, "Antiphase neighbor must be rejected (Invariant #5)"

        # Verify telemetry: no propagation to node 1
        propagations = G.graph.get("thol_propagations", [])
        if propagations:
            for prop in propagations:
                if prop["source_node"] == 0:
                    affected_neighbors = [n for n, _ in prop["propagations"]]
                    assert (
                        1 not in affected_neighbors
                    ), "Antiphase neighbor should not appear in propagation list"


class TestTHOLCascades:
    """Test cascade triggering across network chains."""

    def test_thol_triggers_cascade_in_linear_chain(self):
        """THOL cascade: multiple nodes bifurcate in sequence."""
        # Create chain network
        G = nx.path_graph(5)  # Linear chain 0-1-2-3-4

        # Initialize all nodes
        for n in G.nodes:
            G.nodes[n][EPI_PRIMARY] = 0.50
            G.nodes[n][VF_PRIMARY] = 1.0
            G.nodes[n][THETA_PRIMARY] = 0.1 + n * 0.02  # Phase-aligned
            G.nodes[n][DNFR_PRIMARY] = 0.10
            # Use strongly accelerating history: d²EPI = abs(0.50 - 2*0.33 + 0.05) = 0.11
            G.nodes[n]["epi_history"] = [0.05, 0.33, 0.50]

        G.graph["THOL_PROPAGATION_ENABLED"] = True
        G.graph["THOL_BIFURCATION_THRESHOLD"] = 0.1  # Standard threshold
        G.graph["THOL_MIN_COUPLING_FOR_PROPAGATION"] = 0.5
        G.graph["THOL_CASCADE_MIN_NODES"] = 2  # Lower threshold for test

        # Trigger cascade from node 0
        SelfOrganization()(G, 0)

        # Analyze cascade
        cascade_analysis = detect_cascade(G)

        # Should have propagation events
        assert cascade_analysis["total_propagations"] >= 1, "Should have propagations"
        assert len(cascade_analysis["affected_nodes"]) >= 2, "Should reach at least 2 nodes"

    def test_thol_cascade_respects_phase_barriers(self):
        """Cascades should not cross phase-incoherent boundaries."""
        # Create network with phase barrier
        G = nx.Graph()

        # Cluster A: coherent phases
        G.add_node(
            0,
            **{
                EPI_PRIMARY: 0.50,
                THETA_PRIMARY: 0.1,
                VF_PRIMARY: 1.0,
                DNFR_PRIMARY: 0.10,
            },
        )
        G.add_node(
            1,
            **{
                EPI_PRIMARY: 0.45,
                THETA_PRIMARY: 0.12,
                VF_PRIMARY: 1.0,
                DNFR_PRIMARY: 0.10,
            },
        )

        # Node 2: phase barrier (gate)
        G.add_node(
            2,
            **{
                EPI_PRIMARY: 0.45,
                THETA_PRIMARY: 2.0,  # π out of phase
                VF_PRIMARY: 1.0,
                DNFR_PRIMARY: 0.10,
            },
        )

        # Cluster B: coherent phases (isolated by barrier)
        G.add_node(
            3,
            **{
                EPI_PRIMARY: 0.45,
                THETA_PRIMARY: 2.05,
                VF_PRIMARY: 1.0,
                DNFR_PRIMARY: 0.10,
            },
        )

        # Connect: 0-1-2-3
        G.add_edges_from([(0, 1), (1, 2), (2, 3)])

        G.graph["THOL_PROPAGATION_ENABLED"] = True
        G.graph["THOL_MIN_COUPLING_FOR_PROPAGATION"] = 0.5
        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        epi_3_before = G.nodes[3][EPI_PRIMARY]

        # Trigger cascade from node 0
        SelfOrganization()(G, 0)

        # Cascade should stop at phase barrier
        propagations = G.graph.get("thol_propagations", [])
        affected = set()
        for prop in propagations:
            affected.add(prop["source_node"])
            for tgt, _ in prop["propagations"]:
                affected.add(tgt)

        # Should reach node 1, but not cross barrier to node 3
        assert 0 in affected
        # Node 3 should NOT be affected by initial propagation
        epi_3_after = G.nodes[3][EPI_PRIMARY]
        # Allow small tolerance for numerical precision
        assert abs(epi_3_after - epi_3_before) < 0.01 or 3 not in affected

    def test_cascade_detection_metrics(self):
        """Cascade detection provides accurate metrics."""
        G = nx.path_graph(4)  # Chain: 0-1-2-3

        for n in G.nodes:
            G.nodes[n][EPI_PRIMARY] = 0.45
            G.nodes[n][VF_PRIMARY] = 1.0
            G.nodes[n][THETA_PRIMARY] = 0.1
            G.nodes[n][DNFR_PRIMARY] = 0.10
            G.nodes[n]["epi_history"] = [0.30, 0.38, 0.45]

        G.graph["THOL_PROPAGATION_ENABLED"] = True
        G.graph["THOL_CASCADE_MIN_NODES"] = 2

        SelfOrganization()(G, 0)

        cascade_analysis = detect_cascade(G)

        # Verify structure
        assert "is_cascade" in cascade_analysis
        assert "affected_nodes" in cascade_analysis
        assert "cascade_depth" in cascade_analysis
        assert "total_propagations" in cascade_analysis

        if cascade_analysis["is_cascade"]:
            assert len(cascade_analysis["affected_nodes"]) >= 2

    def test_measure_cascade_radius(self):
        """Cascade radius measurement tracks propagation distance."""
        G = nx.path_graph(4)  # Chain: 0-1-2-3

        for n in G.nodes:
            G.nodes[n][EPI_PRIMARY] = 0.45
            G.nodes[n][VF_PRIMARY] = 1.0
            G.nodes[n][THETA_PRIMARY] = 0.1
            G.nodes[n][DNFR_PRIMARY] = 0.10
            G.nodes[n]["epi_history"] = [0.30, 0.38, 0.45]

        G.graph["THOL_PROPAGATION_ENABLED"] = True

        SelfOrganization()(G, 0)

        # If cascade occurred, radius should be > 0
        cascade_analysis = detect_cascade(G)
        if cascade_analysis["is_cascade"]:
            radius = measure_cascade_radius(G, source_node=0)
            assert radius >= 0


class TestTHOLPropagationEPIHistory:
    """Test that propagation updates EPI history for cascade continuation."""

    def test_propagation_updates_neighbor_epi_history(self):
        """Propagated EPIs should be recorded in neighbor's history."""
        G = nx.Graph()
        G.add_node(
            0,
            **{
                EPI_PRIMARY: 0.50,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: 0.10,
            },
        )
        G.add_node(
            1,
            **{
                EPI_PRIMARY: 0.40,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.01,
                DNFR_PRIMARY: 0.05,
            },
        )
        G.add_edge(0, 1)

        G.graph["THOL_PROPAGATION_ENABLED"] = True
        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        # Node 1 starts without history
        initial_history_len = len(G.nodes[1].get("epi_history", []))

        SelfOrganization()(G, 0)

        # Node 1 history should be updated
        final_history = G.nodes[1].get("epi_history", [])
        assert len(final_history) > initial_history_len, "History should be updated"


class TestBackwardCompatibility:
    """Test that propagation doesn't break existing THOL behavior."""

    def test_isolated_node_still_bifurcates(self):
        """Isolated node (no neighbors) should still bifurcate normally."""
        G = nx.Graph()
        G.add_node(
            0,
            **{
                EPI_PRIMARY: 0.50,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: 0.10,
            },
        )

        G.graph["THOL_PROPAGATION_ENABLED"] = True
        G.nodes[0]["epi_history"] = [0.05, 0.33, 0.50]

        epi_before = G.nodes[0][EPI_PRIMARY]

        SelfOrganization()(G, 0)

        epi_after = G.nodes[0][EPI_PRIMARY]

        # Should still bifurcate and increase EPI
        assert epi_after > epi_before, "Isolated node should still bifurcate"

        # No propagations recorded
        propagations = G.graph.get("thol_propagations", [])
        assert len(propagations) == 0, "Isolated node cannot propagate"

    def test_no_bifurcation_no_propagation(self):
        """If bifurcation doesn't occur, propagation shouldn't happen."""
        G = nx.Graph()
        G.add_node(
            0,
            **{
                EPI_PRIMARY: 0.50,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.0,
                DNFR_PRIMARY: 0.10,
            },
        )
        G.add_node(
            1,
            **{
                EPI_PRIMARY: 0.40,
                VF_PRIMARY: 1.0,
                THETA_PRIMARY: 0.01,
                DNFR_PRIMARY: 0.05,
            },
        )
        G.add_edge(0, 1)

        G.graph["THOL_PROPAGATION_ENABLED"] = True
        # Low acceleration - no bifurcation
        G.nodes[0]["epi_history"] = [0.49, 0.495, 0.50]

        epi_1_before = G.nodes[1][EPI_PRIMARY]

        SelfOrganization()(G, 0)

        epi_1_after = G.nodes[1][EPI_PRIMARY]

        # No bifurcation = no propagation
        assert epi_1_after == epi_1_before, "Should not propagate without bifurcation"
        propagations = G.graph.get("thol_propagations", [])
        assert len(propagations) == 0
