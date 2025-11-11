"""Tests for the simplified TNFR SDK fluent API."""

import pytest
import networkx as nx

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from tnfr.sdk import TNFRNetwork, NetworkConfig, NetworkResults


class TestNetworkConfig:
    """Test NetworkConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NetworkConfig()
        assert config.random_seed is None
        assert config.validate_invariants is True
        assert config.auto_stabilization is True
        assert config.default_vf_range == (0.1, 1.0)
        assert config.default_epi_range == (0.1, 0.9)

    def test_custom_config(self):
        """Test custom configuration."""
        config = NetworkConfig(
            random_seed=42, validate_invariants=False, default_vf_range=(0.5, 2.0)
        )
        assert config.random_seed == 42
        assert config.validate_invariants is False
        assert config.default_vf_range == (0.5, 2.0)


class TestNetworkResults:
    """Test NetworkResults dataclass."""

    def test_summary(self):
        """Test results summary generation."""
        results = NetworkResults(
            coherence=0.75,
            sense_indices={"n1": 0.8, "n2": 0.6},
            delta_nfr={"n1": 0.1, "n2": 0.2},
            graph=nx.Graph(),
            avg_vf=1.0,
            avg_phase=0.5,
        )
        summary = results.summary()
        assert "0.750" in summary  # Coherence
        assert "0.700" in summary  # Avg Si
        assert "0.150" in summary  # Avg ΔNFR

    def test_to_dict(self):
        """Test conversion to dictionary."""
        results = NetworkResults(
            coherence=0.8,
            sense_indices={"n1": 0.9},
            delta_nfr={"n1": 0.05},
            graph=nx.Graph(),
        )
        data = results.to_dict()
        assert data["coherence"] == 0.8
        assert data["sense_indices"] == {"n1": 0.9}
        assert "summary_stats" in data
        assert data["summary_stats"]["node_count"] == 1


class TestTNFRNetwork:
    """Test TNFRNetwork fluent API."""

    def test_initialization(self):
        """Test network initialization."""
        network = TNFRNetwork("test_network")
        assert network.name == "test_network"
        assert network._config is not None
        assert network._graph is None

    def test_add_nodes(self):
        """Test adding nodes to network."""
        network = TNFRNetwork()
        network.add_nodes(5)

        assert network._graph is not None
        assert network._graph.number_of_nodes() == 5

        # Check that nodes have TNFR properties
        for node in network._graph.nodes():
            node_data = network._graph.nodes[node]
            assert "νf" in node_data  # structural frequency (Greek nu)
            assert "theta" in node_data  # phase
            assert "EPI" in node_data  # EPI

    def test_fluent_chaining(self):
        """Test method chaining returns self."""
        network = TNFRNetwork()
        result = network.add_nodes(3)
        assert result is network

        result = network.connect_nodes(0.5, "random")
        assert result is network

    def test_connect_nodes_random(self):
        """Test random network connectivity."""
        network = TNFRNetwork()
        network.add_nodes(10)
        network.connect_nodes(0.5, "random")

        # Should have some edges
        assert network._graph.number_of_edges() > 0

    def test_connect_nodes_ring(self):
        """Test ring network topology."""
        network = TNFRNetwork()
        network.add_nodes(5)
        network.connect_nodes(connection_pattern="ring")

        # Ring has exactly n edges
        assert network._graph.number_of_edges() == 5

        # Each node should have degree 2
        for node in network._graph.nodes():
            assert network._graph.degree(node) == 2

    def test_connect_nodes_small_world(self):
        """Test small-world network topology."""
        network = TNFRNetwork()
        network.add_nodes(20)
        network.connect_nodes(0.1, "small_world")

        # Should create a connected network
        assert network._graph.number_of_edges() > 0

    def test_apply_sequence_predefined(self):
        """Test applying predefined operator sequence."""
        network = TNFRNetwork()
        network.add_nodes(5)
        network.connect_nodes(0.3, "random")
        network.apply_sequence("basic_activation", repeat=1)

        # Should complete without errors
        assert network._graph is not None

    def test_apply_sequence_custom(self):
        """Test applying custom operator sequence."""
        network = TNFRNetwork()
        network.add_nodes(3)
        network.connect_nodes(0.5, "ring")

        # Custom valid sequence
        network.apply_sequence(
            ["emission", "reception", "coherence", "resonance", "silence"]
        )

        assert network._graph is not None

    def test_apply_sequence_invalid_name(self):
        """Test error on invalid sequence name."""
        network = TNFRNetwork()
        network.add_nodes(3)

        with pytest.raises(ValueError, match="Unknown sequence"):
            network.apply_sequence("invalid_sequence_name")

    def test_measure(self):
        """Test measuring network metrics."""
        network = TNFRNetwork()
        network.add_nodes(5)
        network.connect_nodes(0.3, "random")
        network.apply_sequence("basic_activation")

        results = network.measure()

        assert isinstance(results, NetworkResults)
        assert isinstance(results.coherence, float)
        assert len(results.sense_indices) == 5
        assert len(results.delta_nfr) == 5
        assert results.avg_vf is not None
        assert results.avg_phase is not None

    def test_full_workflow(self):
        """Test complete workflow from creation to measurement."""
        results = (
            TNFRNetwork("workflow_test")
            .add_nodes(8)
            .connect_nodes(0.4, "random")
            .apply_sequence("network_sync", repeat=2)
            .measure()
        )

        assert results.coherence >= 0.0
        assert results.coherence <= 1.0
        assert len(results.sense_indices) == 8

    def test_graph_property(self):
        """Test accessing underlying graph."""
        network = TNFRNetwork()
        network.add_nodes(3)

        graph = network.graph
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 3

    def test_graph_property_before_creation(self):
        """Test error when accessing graph before creation."""
        network = TNFRNetwork()

        with pytest.raises(ValueError, match="No network created"):
            _ = network.graph

    def test_operations_without_nodes(self):
        """Test error when operating on empty network."""
        network = TNFRNetwork()

        with pytest.raises(ValueError, match="No nodes"):
            network.connect_nodes(0.5)

        with pytest.raises(ValueError, match="No nodes"):
            network.apply_sequence("basic_activation")

        with pytest.raises(ValueError, match="No network created"):
            network.measure()
