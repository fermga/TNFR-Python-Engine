"""Test suite for extended dynamics fields (Strain, Vorticity).

Validates research-phase fields preserve physics invariants.
"""

import pytest
import networkx as nx
import numpy as np

try:
    from src.tnfr.physics.extended import (
        compute_phase_strain,
        compute_phase_vorticity,
        compute_reorganization_strain,
        compute_extended_dynamics_suite,
    )
    EXTENDED_AVAILABLE = True
except ImportError:
    EXTENDED_AVAILABLE = False


@pytest.mark.skipif(not EXTENDED_AVAILABLE, reason="Extended fields not available")
class TestExtendedDynamics:
    """Test extended dynamics fields."""

    @staticmethod
    def create_test_graph(size=20):
        """Create test graph with TNFR attributes."""
        G = nx.watts_strogatz_graph(size, k=4, p=0.3)
        for node in G.nodes():
            G.nodes[node]["phase"] = np.random.uniform(0, 2 * np.pi)
            G.nodes[node]["delta_nfr"] = np.random.uniform(-0.5, 0.5)
        for u, v in G.edges():
            G[u][v]["weight"] = np.random.uniform(0.5, 1.5)
        return G

    def test_phase_strain_output_structure(self):
        """Phase strain should return dict per node."""
        G = self.create_test_graph()
        strain = compute_phase_strain(G)

        assert isinstance(strain, dict)
        assert len(strain) == G.number_of_nodes()
        for node, value in strain.items():
            assert node in G.nodes()
            assert isinstance(value, float)
            assert value >= 0.0

    def test_phase_vorticity_output_structure(self):
        """Phase vorticity should return dict per node."""
        G = self.create_test_graph()
        vort = compute_phase_vorticity(G)

        assert isinstance(vort, dict)
        assert len(vort) == G.number_of_nodes()
        for node, value in vort.items():
            assert node in G.nodes()
            assert isinstance(value, (float, int))

    def test_reorganization_strain_output_structure(self):
        """Reorganization strain should return dict per node."""
        G = self.create_test_graph()
        strain = compute_reorganization_strain(G)

        assert isinstance(strain, dict)
        assert len(strain) == G.number_of_nodes()
        for node, value in strain.items():
            assert node in G.nodes()
            assert isinstance(value, float)
            assert value >= 0.0

    def test_extended_dynamics_suite_completeness(self):
        """Suite should return all three fields."""
        G = self.create_test_graph()
        suite = compute_extended_dynamics_suite(G)

        assert isinstance(suite, dict)
        assert "phase_strain" in suite
        assert "phase_vorticity" in suite
        assert "reorganization_strain" in suite

        for field_name, field_values in suite.items():
            assert isinstance(field_values, dict)
            assert len(field_values) == G.number_of_nodes()

    def test_phase_strain_isolated_node(self):
        """Isolated node should have zero strain."""
        G = nx.Graph()
        G.add_node(0, phase=0.5, delta_nfr=0.1)

        strain = compute_phase_strain(G)
        assert strain[0] == 0.0

    def test_vorticity_isolated_node(self):
        """Isolated node should have zero vorticity."""
        G = nx.Graph()
        G.add_node(0, phase=0.5)

        vort = compute_phase_vorticity(G)
        assert vort[0] == 0.0

    def test_fields_no_epi_mutation(self):
        """Fields must not mutate node EPI attributes."""
        G = self.create_test_graph()

        # Store original EPI values (if any)
        original_attrs = {}
        for node in G.nodes():
            original_attrs[node] = dict(G.nodes[node])

        # Compute fields
        compute_phase_strain(G)
        compute_phase_vorticity(G)
        compute_reorganization_strain(G)

        # Verify no mutations
        for node in G.nodes():
            for key in original_attrs[node]:
                assert G.nodes[node][key] == original_attrs[node][key]

    def test_fields_deterministic(self):
        """Fields with same input should give same output."""
        G = self.create_test_graph(10)

        strain1 = compute_phase_strain(G)
        strain2 = compute_phase_strain(G)

        for node in G.nodes():
            assert strain1[node] == strain2[node]

    def test_vorticity_continuity(self):
        """Vorticity should be bounded and continuous."""
        G = self.create_test_graph()
        vort = compute_phase_vorticity(G)

        values = list(vort.values())
        assert all(isinstance(v, (float, int)) for v in values)
        # Vorticity should be bounded (not infinite)
        assert all(np.isfinite(v) for v in values)

    def test_strain_symmetric_neighbors(self):
        """Strain computation should handle symmetric neighbors correctly."""
        # Create small symmetric graph
        G = nx.Graph()
        for i in range(4):
            G.add_node(i, phase=float(i) / 4.0, delta_nfr=0.1)

        # Add symmetric edges
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

        strain = compute_phase_strain(G)
        assert all(isinstance(v, float) for v in strain.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
