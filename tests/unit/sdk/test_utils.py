"""Tests for TNFR SDK utility functions."""

import pytest
import json
from pathlib import Path

from tnfr.sdk import (
    TNFRNetwork,
    TNFRExperimentBuilder,
    compare_networks,
    compute_network_statistics,
    export_to_json,
    import_from_json,
    format_comparison_table,
    suggest_sequence_for_goal,
)


class TestUtilityFunctions:
    """Test SDK utility functions."""

    def test_compare_networks(self):
        """Test network comparison function."""
        # Create two simple networks
        results1 = (
            TNFRNetwork("net1")
            .add_nodes(10, random_seed=42)
            .connect_nodes(0.3, "random")
            .apply_sequence("basic_activation")
            .measure()
        )

        results2 = (
            TNFRNetwork("net2")
            .add_nodes(10, random_seed=43)
            .connect_nodes(0.5, "ring")
            .apply_sequence("network_sync")
            .measure()
        )

        networks = {"net1": results1, "net2": results2}
        comparison = compare_networks(networks)

        assert "net1" in comparison
        assert "net2" in comparison
        assert "coherence" in comparison["net1"]
        assert "avg_si" in comparison["net1"]
        assert isinstance(comparison["net1"]["coherence"], float)

    def test_compute_network_statistics(self):
        """Test statistics computation."""
        results = (
            TNFRNetwork("test")
            .add_nodes(10, random_seed=42)
            .connect_nodes(0.3)
            .apply_sequence("basic_activation")
            .measure()
        )

        stats = compute_network_statistics(results)

        assert "coherence" in stats
        assert "node_count" in stats
        assert "avg_si" in stats
        assert "min_si" in stats
        assert "max_si" in stats
        assert "std_si" in stats
        assert stats["node_count"] == 10

    def test_export_import_json(self, tmp_path):
        """Test JSON export and import."""
        network = (
            TNFRNetwork("test")
            .add_nodes(5, random_seed=42)
            .connect_nodes(0.4)
            .apply_sequence("basic_activation")
            .measure()
        )

        # Export
        json_path = tmp_path / "network.json"
        export_to_json(network, json_path)

        assert json_path.exists()

        # Import
        data = import_from_json(json_path)

        assert "name" in data
        assert data["name"] == "test"
        assert "metadata" in data
        assert data["metadata"]["nodes"] == 5

    def test_format_comparison_table(self):
        """Test comparison table formatting."""
        comparison = {
            "network1": {"coherence": 0.75, "avg_si": 0.6},
            "network2": {"coherence": 0.82, "avg_si": 0.7},
        }

        table = format_comparison_table(comparison)

        assert "Network" in table
        assert "coherence" in table
        assert "network1" in table
        assert "network2" in table
        assert "0.750" in table or "0.75" in table

    def test_suggest_sequence_for_goal(self):
        """Test sequence suggestion."""
        seq, desc = suggest_sequence_for_goal("stabilize")
        assert seq == "stabilization"
        assert "coherent" in desc.lower() or "recursive" in desc.lower()

        seq, desc = suggest_sequence_for_goal("synchronize")
        assert seq == "network_sync"
        assert "coupling" in desc.lower() or "resonance" in desc.lower()

        seq, desc = suggest_sequence_for_goal("explore")
        assert seq == "exploration"

        # Unknown goal should return default
        seq, desc = suggest_sequence_for_goal("unknown_goal")
        assert seq == "basic_activation"


class TestConvenienceMethods:
    """Test convenience methods added to TNFRNetwork."""

    def test_get_node_count(self):
        """Test node count getter."""
        network = TNFRNetwork().add_nodes(15)
        assert network.get_node_count() == 15

    def test_get_edge_count(self):
        """Test edge count getter."""
        network = TNFRNetwork().add_nodes(5).connect_nodes(pattern="ring")
        assert network.get_edge_count() == 5

    def test_get_average_degree(self):
        """Test average degree calculation."""
        network = TNFRNetwork().add_nodes(4).connect_nodes(pattern="ring")
        # Ring has degree 2 for all nodes
        assert network.get_average_degree() == 2.0

    def test_get_density(self):
        """Test network density calculation."""
        network = TNFRNetwork().add_nodes(4)
        # Complete graph of 4 nodes
        for i in range(4):
            for j in range(i + 1, 4):
                network._graph.add_edge(f"node_{i}", f"node_{j}")

        # Complete graph has density 1.0
        assert network.get_density() == 1.0

    def test_clone(self):
        """Test network cloning."""
        original = TNFRNetwork("original").add_nodes(5, random_seed=42).connect_nodes(0.3)

        cloned = original.clone()

        assert cloned.name == "original_copy"
        assert cloned.get_node_count() == 5
        assert cloned.get_edge_count() == original.get_edge_count()

        # Verify it's a deep copy
        original.add_nodes(2)
        assert original.get_node_count() == 7
        assert cloned.get_node_count() == 5

    def test_reset(self):
        """Test network reset."""
        network = TNFRNetwork().add_nodes(10).connect_nodes(0.3)
        assert network.get_node_count() == 10

        network.reset()

        with pytest.raises(ValueError, match="No network created"):
            network.get_node_count()

    def test_export_to_dict(self):
        """Test dictionary export."""
        network = (
            TNFRNetwork("test")
            .add_nodes(8, random_seed=42)
            .connect_nodes(0.4)
            .apply_sequence("basic_activation")
            .measure()
        )

        data = network.export_to_dict()

        assert data["name"] == "test"
        assert "metadata" in data
        assert data["metadata"]["nodes"] == 8
        assert "metrics" in data
        assert "config" in data
        assert data["config"]["random_seed"] == 42

    def test_convenience_methods_without_network(self):
        """Test error handling when network not created."""
        network = TNFRNetwork()

        with pytest.raises(ValueError, match="No network created"):
            network.get_node_count()

        with pytest.raises(ValueError, match="No network created"):
            network.get_edge_count()

        with pytest.raises(ValueError, match="No network created"):
            network.get_average_degree()

        with pytest.raises(ValueError, match="No network created"):
            network.get_density()

        with pytest.raises(ValueError, match="No network created"):
            network.clone()

        with pytest.raises(ValueError, match="No network created"):
            network.export_to_dict()


class TestIntegrationWithUtils:
    """Test integration of utilities with SDK."""

    def test_full_workflow_with_utils(self, tmp_path):
        """Test complete workflow using utilities."""
        # Create and compare topologies
        comparison_results = TNFRExperimentBuilder.compare_topologies(
            node_count=15, steps=3, random_seed=42
        )

        # Compare networks
        comparison = compare_networks(comparison_results)
        assert len(comparison) == 3

        # Format table
        table = format_comparison_table(comparison)
        assert "random" in table
        assert "ring" in table
        assert "small_world" in table

        # Export one network
        network = TNFRNetwork("workflow_test")
        network.add_nodes(10, random_seed=42)
        network.connect_nodes(0.3)
        network.apply_sequence("basic_activation")
        network.measure()

        json_path = tmp_path / "workflow.json"
        export_to_json(network, json_path)

        # Import and verify
        loaded_data = import_from_json(json_path)
        assert loaded_data["name"] == "workflow_test"
        assert loaded_data["metadata"]["nodes"] == 10
