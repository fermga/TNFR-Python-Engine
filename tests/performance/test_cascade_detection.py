"""Performance benchmarks for cascade detection in large networks.

This module tests the scalability of detect_cascade() for networks
with >1000 nodes, targeting <100ms detection time for 10k node networks.

References:
    - Issue: [THOL][Performance] Optimizar detect_cascade() para escalabilidad
    - Module: src/tnfr/operators/cascade.py
"""

import time

import networkx as nx
import pytest

from tnfr.operators.cascade import detect_cascade
from tnfr.structural import create_nfr


def create_test_network_with_cascade(n_nodes, cascade_length=None):
    """Create test network with simulated THOL cascade propagations.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network
    cascade_length : int, optional
        Number of propagation events to simulate.
        Defaults to n_nodes // 10 for realistic cascade density.

    Returns
    -------
    TNFRGraph
        Network with thol_propagations populated
    """
    if cascade_length is None:
        cascade_length = max(10, n_nodes // 10)

    # Create base network
    G = nx.Graph()

    # Add nodes with TNFR attributes
    for i in range(n_nodes):
        G.add_node(i, epi=0.50, vf=1.0, theta=0.1 + i * 0.001)

    # Add edges for coupling
    # Create small-world topology for realistic network
    if n_nodes < 100:
        # Complete graph for small networks
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                G.add_edge(i, j)
    else:
        # Watts-Strogatz small-world for large networks
        # k=6 means each node connects to 6 neighbors
        k = min(6, n_nodes - 1)
        G = nx.watts_strogatz_graph(n=n_nodes, k=k, p=0.1)

        # Add TNFR attributes to regenerated graph
        for i in range(n_nodes):
            G.nodes[i]["epi"] = 0.50
            G.nodes[i]["vf"] = 1.0
            G.nodes[i]["theta"] = 0.1 + i * 0.001

    # Simulate cascade propagations
    # Each propagation event affects 2-5 neighbors
    propagations = []
    for event_idx in range(cascade_length):
        source_node = event_idx % n_nodes

        # Get neighbors for this source
        neighbors = list(G.neighbors(source_node))
        if not neighbors:
            continue

        # Propagate to 2-5 neighbors (or all if fewer exist)
        import random

        random.seed(42 + event_idx)  # Deterministic for reproducibility
        n_targets = min(random.randint(2, 5), len(neighbors))
        targets = random.sample(neighbors, n_targets)

        # Create propagation record
        propagations.append(
            {
                "source_node": source_node,
                "propagations": [(t, 0.10) for t in targets],
                "timestamp": 10 + event_idx,
            }
        )

    G.graph["thol_propagations"] = propagations
    G.graph["THOL_CASCADE_MIN_NODES"] = 3

    return G


class TestCascadeDetectionScaling:
    """Test cascade detection performance vs network size."""

    @pytest.mark.parametrize("n_nodes", [100, 500, 1000, 5000])
    def test_cascade_detection_time(self, n_nodes, benchmark):
        """Measure cascade detection time for various network sizes.

        Target: <100ms for 10,000 nodes (tested separately due to time).
        Expected scaling: Should be sub-linear with incremental cache.
        """
        G = create_test_network_with_cascade(n_nodes)

        # Benchmark the detection
        result = benchmark(detect_cascade, G)

        # Verify correctness
        assert "is_cascade" in result
        assert "affected_nodes" in result
        assert "cascade_depth" in result
        assert "total_propagations" in result

        # Performance assertion (generous bounds for CI variability)
        # After optimization, these should be much faster
        if n_nodes <= 1000:
            # Small networks should be fast even without optimization
            assert (
                benchmark.stats.median < 0.1
            ), f"Detection too slow for n={n_nodes}: {benchmark.stats.median:.3f}s"
        elif n_nodes <= 5000:
            # Mid-size networks: current implementation may struggle
            # After optimization, should be <50ms
            assert (
                benchmark.stats.median < 0.5
            ), f"Detection too slow for n={n_nodes}: {benchmark.stats.median:.3f}s"

    def test_cascade_detection_10k_nodes(self, benchmark):
        """Test detection on large 10k node network.

        This is the target scenario from the issue.
        Current implementation may be slow; after optimization should be <100ms.
        """
        n_nodes = 10000
        G = create_test_network_with_cascade(n_nodes)

        result = benchmark(detect_cascade, G)

        # Verify correctness
        assert result["is_cascade"] is True
        assert len(result["affected_nodes"]) >= 3

        # Performance target from issue: <100ms = 0.1s
        # Allow 1s for now (will improve with optimization)
        assert (
            benchmark.stats.median < 1.0
        ), f"Detection too slow for 10k nodes: {benchmark.stats.median:.3f}s"

    def test_multiple_detections_same_network(self):
        """Test repeated detections on same network (cache benefit).

        With incremental cache, subsequent detections should be O(1).
        """
        n_nodes = 5000
        G = create_test_network_with_cascade(n_nodes)

        # First detection (may build cache)
        start = time.time()
        result1 = detect_cascade(G)
        first_time = time.time() - start

        # Second detection (should use cache)
        start = time.time()
        result2 = detect_cascade(G)
        second_time = time.time() - start

        # Results should be identical
        assert result1["is_cascade"] == result2["is_cascade"]
        assert len(result1["affected_nodes"]) == len(result2["affected_nodes"])

        # Note: Without cache, times will be similar
        # With cache, second should be much faster
        print(f"First: {first_time:.3f}s, Second: {second_time:.3f}s")


class TestCascadeDetectionCorrectness:
    """Test that optimization preserves correctness."""

    def test_no_cascade_empty_propagations(self):
        """Empty propagations should report no cascade."""
        G = nx.Graph()
        G.add_node(0, epi=0.50, vf=1.0, theta=0.1)
        G.graph["thol_propagations"] = []

        result = detect_cascade(G)

        assert result["is_cascade"] is False
        assert len(result["affected_nodes"]) == 0
        assert result["cascade_depth"] == 0
        assert result["total_propagations"] == 0

    def test_small_cascade_below_threshold(self):
        """Cascade affecting <3 nodes should not be detected."""
        G = nx.Graph()
        G.add_node(0, epi=0.50, vf=1.0, theta=0.1)
        G.add_node(1, epi=0.50, vf=1.0, theta=0.1)
        G.add_edge(0, 1)

        # Propagation affecting only 2 nodes
        G.graph["thol_propagations"] = [
            {
                "source_node": 0,
                "propagations": [(1, 0.10)],
                "timestamp": 10,
            }
        ]
        G.graph["THOL_CASCADE_MIN_NODES"] = 3

        result = detect_cascade(G)

        assert result["is_cascade"] is False
        assert len(result["affected_nodes"]) == 2  # Source + target

    def test_cascade_above_threshold(self):
        """Cascade affecting ≥3 nodes should be detected."""
        G = create_test_network_with_cascade(n_nodes=10, cascade_length=5)

        result = detect_cascade(G)

        # With 5 propagation events in 10-node network, should reach threshold
        assert result["is_cascade"] is True
        assert len(result["affected_nodes"]) >= 3
        assert result["cascade_depth"] > 0
        assert result["total_propagations"] > 0

    def test_affected_nodes_counted_once(self):
        """Each node should be counted only once even if affected multiple times."""
        G = nx.Graph()
        for i in range(5):
            G.add_node(i, epi=0.50, vf=1.0, theta=0.1)

        # Multiple propagations to same nodes
        G.graph["thol_propagations"] = [
            {
                "source_node": 0,
                "propagations": [(1, 0.10), (2, 0.09)],
                "timestamp": 10,
            },
            {
                "source_node": 1,
                "propagations": [(2, 0.08), (3, 0.07)],  # Node 2 affected again
                "timestamp": 11,
            },
        ]
        G.graph["THOL_CASCADE_MIN_NODES"] = 3

        result = detect_cascade(G)

        # Should count nodes 0, 1, 2, 3 = 4 unique nodes
        assert len(result["affected_nodes"]) == 4
        assert result["affected_nodes"] == {0, 1, 2, 3}


class TestCascadeDetectionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_node_isolated(self):
        """Isolated node should report no cascade."""
        G = nx.Graph()
        G.add_node(0, epi=0.50, vf=1.0, theta=0.1)
        G.graph["thol_propagations"] = []

        result = detect_cascade(G)

        assert result["is_cascade"] is False

    def test_very_large_cascade(self):
        """Very large cascade should be handled correctly."""
        n_nodes = 1000
        # Dense cascade: propagations = n_nodes (one per node)
        G = create_test_network_with_cascade(n_nodes, cascade_length=n_nodes)

        result = detect_cascade(G)

        assert result["is_cascade"] is True
        # Should affect significant portion of network
        assert len(result["affected_nodes"]) > n_nodes // 2

    def test_custom_cascade_threshold(self):
        """Custom cascade threshold should be respected."""
        G = create_test_network_with_cascade(n_nodes=20, cascade_length=3)

        # Set high threshold
        G.graph["THOL_CASCADE_MIN_NODES"] = 100

        result = detect_cascade(G)

        # Should not detect cascade with high threshold
        assert result["is_cascade"] is False

        # Lower threshold
        G.graph["THOL_CASCADE_MIN_NODES"] = 5
        result = detect_cascade(G)

        # Should detect with lower threshold
        assert result["is_cascade"] is True


if __name__ == "__main__":
    # Quick manual test
    print("Testing cascade detection performance...\n")

    for n in [100, 500, 1000, 5000, 10000]:
        G = create_test_network_with_cascade(n)

        start = time.time()
        result = detect_cascade(G)
        elapsed = time.time() - start

        print(
            f"n={n:5d}: {elapsed:.3f}s - "
            f"cascade={result['is_cascade']}, "
            f"affected={len(result['affected_nodes']):4d}, "
            f"depth={result['cascade_depth']:3d}"
        )
