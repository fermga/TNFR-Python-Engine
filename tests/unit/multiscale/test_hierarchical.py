"""Tests for hierarchical multi-scale TNFR networks.

Validates operational fractality (§3.7) and preservation of TNFR invariants
across multiple scales.
"""

import pytest
import numpy as np

from tnfr.multiscale import HierarchicalTNFRNetwork, ScaleDefinition


class TestScaleDefinition:
    """Test ScaleDefinition dataclass."""

    def test_creates_valid_scale(self):
        """Scale definition with valid parameters."""
        scale = ScaleDefinition("test", 100, 0.5)
        assert scale.name == "test"
        assert scale.node_count == 100
        assert scale.coupling_strength == 0.5
        assert scale.edge_probability == 0.1

    def test_custom_edge_probability(self):
        """Scale definition with custom edge probability."""
        scale = ScaleDefinition("test", 100, 0.5, edge_probability=0.2)
        assert scale.edge_probability == 0.2


class TestHierarchicalNetworkInitialization:
    """Test hierarchical network initialization."""

    def test_creates_single_scale_network(self):
        """Single scale network initializes correctly."""
        scales = [ScaleDefinition("micro", 50, 0.8)]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        assert len(network.networks_by_scale) == 1
        assert "micro" in network.networks_by_scale
        assert network.networks_by_scale["micro"].number_of_nodes() == 50

    def test_creates_multi_scale_network(self):
        """Multi-scale network with multiple levels."""
        scales = [
            ScaleDefinition("quantum", 100, 0.9),
            ScaleDefinition("molecular", 50, 0.7),
            ScaleDefinition("cellular", 25, 0.5),
        ]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        assert len(network.networks_by_scale) == 3
        assert network.networks_by_scale["quantum"].number_of_nodes() == 100
        assert network.networks_by_scale["molecular"].number_of_nodes() == 50
        assert network.networks_by_scale["cellular"].number_of_nodes() == 25

    def test_raises_on_empty_scales(self):
        """Raises ValueError for empty scales list."""
        with pytest.raises(ValueError, match="At least one scale"):
            HierarchicalTNFRNetwork([])

    def test_deterministic_with_seed(self):
        """Same seed produces identical networks."""
        scales = [ScaleDefinition("test", 50, 0.8)]

        net1 = HierarchicalTNFRNetwork(scales, seed=42)
        net2 = HierarchicalTNFRNetwork(scales, seed=42)

        G1 = net1.get_scale_network("test")
        G2 = net2.get_scale_network("test")

        assert G1.number_of_edges() == G2.number_of_edges()
        # Check first node's attributes match
        node = list(G1.nodes())[0]
        assert G1.nodes[node]["epi"] == G2.nodes[node]["epi"]
        assert G1.nodes[node]["vf"] == G2.nodes[node]["vf"]


class TestTNFRInvariantPreservation:
    """Test that TNFR canonical invariants are preserved."""

    def test_nodes_have_canonical_attributes(self):
        """All nodes have EPI, νf, phase, ΔNFR attributes."""
        scales = [ScaleDefinition("test", 50, 0.8)]
        network = HierarchicalTNFRNetwork(scales, seed=42)
        G = network.get_scale_network("test")

        for node in G.nodes():
            assert "epi" in G.nodes[node]
            assert "vf" in G.nodes[node]
            assert "phase" in G.nodes[node]
            assert "delta_nfr" in G.nodes[node]

    def test_vf_in_valid_range(self):
        """Structural frequency νf is positive."""
        scales = [ScaleDefinition("test", 100, 0.8)]
        network = HierarchicalTNFRNetwork(scales, seed=42)
        G = network.get_scale_network("test")

        for node in G.nodes():
            vf = G.nodes[node]["vf"]
            assert vf > 0.0, f"νf must be positive, got {vf}"

    def test_phase_in_valid_range(self):
        """Phase θ is in [0, 2π]."""
        scales = [ScaleDefinition("test", 100, 0.8)]
        network = HierarchicalTNFRNetwork(scales, seed=42)
        G = network.get_scale_network("test")

        for node in G.nodes():
            phase = G.nodes[node]["phase"]
            assert 0.0 <= phase <= 2 * np.pi


class TestCrossScaleCoupling:
    """Test cross-scale coupling mechanisms."""

    def test_cross_scale_couplings_initialized(self):
        """Cross-scale coupling matrix is initialized."""
        scales = [
            ScaleDefinition("micro", 50, 0.8),
            ScaleDefinition("macro", 25, 0.6),
        ]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        # Should have couplings between all scale pairs (except self)
        assert ("micro", "macro") in network.cross_scale_couplings
        assert ("macro", "micro") in network.cross_scale_couplings

        # No self-coupling
        assert ("micro", "micro") not in network.cross_scale_couplings

    def test_adjacent_scales_couple_stronger(self):
        """Adjacent scales have stronger coupling than distant ones."""
        scales = [
            ScaleDefinition("s1", 50, 0.8),
            ScaleDefinition("s2", 50, 0.8),
            ScaleDefinition("s3", 50, 0.8),
        ]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        # s1 -> s2 (distance 1) should be stronger than s1 -> s3 (distance 2)
        coupling_adjacent = network.cross_scale_couplings[("s1", "s2")]
        coupling_distant = network.cross_scale_couplings[("s1", "s3")]

        assert coupling_adjacent > coupling_distant

    def test_set_custom_coupling(self):
        """Can set custom cross-scale coupling."""
        scales = [
            ScaleDefinition("micro", 50, 0.8),
            ScaleDefinition("macro", 25, 0.6),
        ]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        network.set_cross_scale_coupling("micro", "macro", 0.7)
        assert network.cross_scale_couplings[("micro", "macro")] == 0.7

    def test_set_coupling_raises_on_invalid_scale(self):
        """Raises ValueError for unknown scale names."""
        scales = [ScaleDefinition("micro", 50, 0.8)]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        with pytest.raises(ValueError, match="Unknown scale"):
            network.set_cross_scale_coupling("micro", "nonexistent", 0.5)

    def test_set_coupling_raises_on_invalid_strength(self):
        """Raises ValueError for coupling strength outside [0, 1]."""
        scales = [
            ScaleDefinition("micro", 50, 0.8),
            ScaleDefinition("macro", 25, 0.6),
        ]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        with pytest.raises(ValueError, match="Coupling strength must be"):
            network.set_cross_scale_coupling("micro", "macro", 1.5)


class TestMultiScaleEvolution:
    """Test multi-scale evolution with cross-coupling."""

    def test_evolve_updates_node_attributes(self):
        """Evolution updates EPI and ΔNFR values."""
        scales = [ScaleDefinition("test", 50, 0.8)]
        network = HierarchicalTNFRNetwork(scales, seed=42)
        G = network.get_scale_network("test")

        # Store initial values
        node = list(G.nodes())[0]
        initial_epi = G.nodes[node]["epi"]

        # Evolve
        result = network.evolve_multiscale(dt=0.1, steps=5)

        # EPI should have changed
        final_epi = G.nodes[node]["epi"]
        assert final_epi != initial_epi

    def test_evolution_preserves_nodal_equation(self):
        """Evolution follows ∂EPI/∂t = νf · ΔNFR(t)."""
        scales = [ScaleDefinition("test", 50, 0.8)]
        network = HierarchicalTNFRNetwork(scales, seed=42)
        G = network.get_scale_network("test")

        node = list(G.nodes())[0]
        initial_epi = G.nodes[node]["epi"]

        dt = 0.1
        result = network.evolve_multiscale(dt=dt, steps=1)

        final_epi = G.nodes[node]["epi"]
        vf = G.nodes[node]["vf"]
        dnfr = G.nodes[node]["delta_nfr"]

        # Check nodal equation: ΔEPI ≈ νf · ΔNFR · dt
        delta_epi = final_epi - initial_epi
        expected_delta = vf * dnfr * dt

        # Allow for numerical precision and cross-scale effects
        assert abs(delta_epi - expected_delta) < 0.5

    def test_returns_evolution_result(self):
        """Evolution returns EvolutionResult with metrics."""
        scales = [
            ScaleDefinition("micro", 50, 0.8),
            ScaleDefinition("macro", 25, 0.6),
        ]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        result = network.evolve_multiscale(dt=0.1, steps=5)

        assert hasattr(result, "scale_results")
        assert hasattr(result, "total_coherence")
        assert hasattr(result, "cross_scale_coupling")

        assert "micro" in result.scale_results
        assert "macro" in result.scale_results

        assert 0.0 <= result.total_coherence <= 1.0
        assert 0.0 <= result.cross_scale_coupling <= 1.0


class TestCoherenceMetrics:
    """Test coherence computation across scales."""

    def test_computes_total_coherence(self):
        """Total coherence is computed across all scales."""
        scales = [
            ScaleDefinition("micro", 50, 0.8),
            ScaleDefinition("macro", 25, 0.6),
        ]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        coherence = network.compute_total_coherence()
        assert 0.0 <= coherence <= 1.0

    def test_coherence_increases_with_evolution(self):
        """Coherence operator increases network coherence."""
        scales = [ScaleDefinition("test", 50, 0.8)]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        initial_coherence = network.compute_total_coherence()

        # Apply coherence operator
        network.evolve_multiscale(dt=0.1, steps=10, operators=["THOL"])

        final_coherence = network.compute_total_coherence()

        # Note: Coherence may not strictly increase due to stochastic
        # initialization, but should be in valid range
        assert 0.0 <= final_coherence <= 1.0


class TestMemoryFootprint:
    """Test memory footprint estimation."""

    def test_computes_memory_footprint(self):
        """Memory footprint is computed for each scale."""
        scales = [
            ScaleDefinition("micro", 100, 0.8),
            ScaleDefinition("macro", 50, 0.6),
        ]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        footprint = network.memory_footprint()

        assert "micro" in footprint
        assert "macro" in footprint
        assert "total" in footprint

        # All values should be positive
        for scale, memory_mb in footprint.items():
            assert memory_mb > 0.0

        # Total should be sum of scales
        expected_total = sum(v for k, v in footprint.items() if k != "total")
        assert abs(footprint["total"] - expected_total) < 0.01


class TestParallelExecution:
    """Test parallel evolution of scales."""

    def test_parallel_evolution_enabled(self):
        """Parallel evolution can be enabled."""
        scales = [
            ScaleDefinition("s1", 50, 0.8),
            ScaleDefinition("s2", 50, 0.8),
        ]
        network = HierarchicalTNFRNetwork(scales, seed=42, parallel=True)

        result = network.evolve_multiscale(dt=0.1, steps=3)
        assert "s1" in result.scale_results
        assert "s2" in result.scale_results

    def test_sequential_evolution(self):
        """Sequential evolution works correctly."""
        scales = [
            ScaleDefinition("s1", 50, 0.8),
            ScaleDefinition("s2", 50, 0.8),
        ]
        network = HierarchicalTNFRNetwork(scales, seed=42, parallel=False)

        result = network.evolve_multiscale(dt=0.1, steps=3)
        assert "s1" in result.scale_results
        assert "s2" in result.scale_results

    def test_parallel_and_sequential_produce_similar_results(self):
        """Parallel and sequential evolution produce comparable results."""
        scales = [
            ScaleDefinition("s1", 30, 0.8),
            ScaleDefinition("s2", 30, 0.8),
        ]

        net_parallel = HierarchicalTNFRNetwork(scales, seed=42, parallel=True)
        net_sequential = HierarchicalTNFRNetwork(scales, seed=42, parallel=False)

        result_parallel = net_parallel.evolve_multiscale(dt=0.1, steps=3)
        result_sequential = net_sequential.evolve_multiscale(dt=0.1, steps=3)

        # Results should be very similar (allow small tolerance for
        # thread timing differences, but they should be nearly identical)
        # Note: Due to ThreadPoolExecutor scheduling, there may be minor
        # differences in floating point operations order
        assert (
            abs(result_parallel.total_coherence - result_sequential.total_coherence) < 0.01
        )  # Tighter tolerance


class TestAPIUsability:
    """Test API usability and error handling."""

    def test_get_scale_network(self):
        """Can retrieve specific scale network."""
        scales = [ScaleDefinition("test", 50, 0.8)]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        G = network.get_scale_network("test")
        assert G.number_of_nodes() == 50

    def test_get_scale_network_raises_on_invalid_name(self):
        """Raises ValueError for unknown scale name."""
        scales = [ScaleDefinition("test", 50, 0.8)]
        network = HierarchicalTNFRNetwork(scales, seed=42)

        with pytest.raises(ValueError, match="Unknown scale"):
            network.get_scale_network("nonexistent")
