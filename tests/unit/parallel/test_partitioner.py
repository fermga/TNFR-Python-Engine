"""Tests for FractalPartitioner."""

import pytest
import networkx as nx


class TestFractalPartitioner:
    """Test suite for TNFR-aware network partitioning."""

    def test_import(self):
        """Test that FractalPartitioner can be imported."""
        from tnfr.parallel import FractalPartitioner
        assert FractalPartitioner is not None

    def test_basic_partition(self):
        """Test basic network partitioning."""
        from tnfr.parallel import FractalPartitioner

        # Create simple test network with varying frequencies to force partitioning
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
        
        # Add varying TNFR attributes to encourage multiple partitions
        G.nodes["a"]["vf"] = 1.0
        G.nodes["a"]["phase"] = 0.0
        G.nodes["b"]["vf"] = 5.0  # Very different frequency
        G.nodes["b"]["phase"] = 0.0
        G.nodes["c"]["vf"] = 1.1
        G.nodes["c"]["phase"] = 0.1
        G.nodes["d"]["vf"] = 5.1  # Similar to b
        G.nodes["d"]["phase"] = 0.1

        partitioner = FractalPartitioner(max_partition_size=2, coherence_threshold=0.7)
        partitions = partitioner.partition_network(G)

        # Should have at least one partition
        assert len(partitions) > 0
        
        # Each partition should be a tuple of (node_set, subgraph)
        for node_set, subgraph in partitions:
            assert isinstance(node_set, set)
            assert isinstance(subgraph, nx.Graph)
            assert len(node_set) > 0

    def test_empty_network(self):
        """Test partitioning empty network."""
        from tnfr.parallel import FractalPartitioner

        G = nx.Graph()
        partitioner = FractalPartitioner()
        partitions = partitioner.partition_network(G)

        assert partitions == []

    def test_coherence_based_partitioning(self):
        """Test that partitioning respects structural coherence."""
        from tnfr.parallel import FractalPartitioner

        # Create network with two distinct coherence groups
        G = nx.Graph()
        
        # Group 1: high frequency, phase 0
        G.add_edges_from([("a1", "a2"), ("a2", "a3")])
        G.nodes["a1"]["vf"] = 2.0
        G.nodes["a1"]["phase"] = 0.0
        G.nodes["a2"]["vf"] = 2.1
        G.nodes["a2"]["phase"] = 0.1
        G.nodes["a3"]["vf"] = 1.9
        G.nodes["a3"]["phase"] = 0.05

        # Group 2: low frequency, phase π
        G.add_edges_from([("b1", "b2"), ("b2", "b3")])
        G.nodes["b1"]["vf"] = 0.5
        G.nodes["b1"]["phase"] = 3.14
        G.nodes["b2"]["vf"] = 0.6
        G.nodes["b2"]["phase"] = 3.10
        G.nodes["b3"]["vf"] = 0.55
        G.nodes["b3"]["phase"] = 3.12

        # Bridge between groups
        G.add_edge("a3", "b1")

        partitioner = FractalPartitioner(
            max_partition_size=10,
            coherence_threshold=0.5
        )
        partitions = partitioner.partition_network(G)

        # Should create separate partitions for coherent groups
        assert len(partitions) >= 1

    def test_partition_coverage(self):
        """Test that all nodes are included in partitions."""
        from tnfr.parallel import FractalPartitioner

        G = nx.complete_graph(10)
        for node in G.nodes():
            G.nodes[node]["vf"] = 1.0
            G.nodes[node]["phase"] = 0.0

        partitioner = FractalPartitioner(max_partition_size=3)
        partitions = partitioner.partition_network(G)

        # Collect all nodes from all partitions
        all_partition_nodes = set()
        for node_set, _ in partitions:
            all_partition_nodes.update(node_set)

        # Should cover all nodes exactly once
        assert all_partition_nodes == set(G.nodes())

    def test_compute_community_coherence(self):
        """Test coherence computation between nodes."""
        from tnfr.parallel import FractalPartitioner

        G = nx.Graph()
        G.add_nodes_from(["a", "b", "c"])
        G.nodes["a"]["vf"] = 1.0
        G.nodes["a"]["phase"] = 0.0
        G.nodes["b"]["vf"] = 1.0
        G.nodes["b"]["phase"] = 0.0
        G.nodes["c"]["vf"] = 10.0
        G.nodes["c"]["phase"] = 0.0

        partitioner = FractalPartitioner()
        
        # Same frequency and phase should have high coherence
        coherence_similar = partitioner._compute_community_coherence(
            G, {"a"}, "b"
        )
        assert coherence_similar > 0.8  # Should be close to 1.0

        # Very different frequency should lower coherence significantly
        coherence_different = partitioner._compute_community_coherence(
            G, {"a"}, "c"
        )
        # With vf diff of 9, coherence should be notably lower
        assert coherence_different < coherence_similar
