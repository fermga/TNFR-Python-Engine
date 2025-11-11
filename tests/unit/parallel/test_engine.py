"""Tests for TNFRParallelEngine."""

import pytest
import networkx as nx


class TestTNFRParallelEngine:
    """Test suite for parallel execution engine."""

    def test_import(self):
        """Test that TNFRParallelEngine can be imported."""
        from tnfr.parallel import TNFRParallelEngine

        assert TNFRParallelEngine is not None

    def test_engine_initialization(self):
        """Test engine initialization with different configurations."""
        from tnfr.parallel import TNFRParallelEngine

        # Default initialization
        engine = TNFRParallelEngine()
        assert engine.max_workers > 0
        assert engine.execution_mode in ["threads", "processes"]

        # Custom workers
        engine = TNFRParallelEngine(max_workers=2)
        assert engine.max_workers == 2

        # Custom execution mode
        engine = TNFRParallelEngine(execution_mode="processes")
        assert engine.execution_mode == "processes"

    def test_recommend_workers(self):
        """Test worker recommendation logic."""
        from tnfr.parallel import TNFRParallelEngine

        engine = TNFRParallelEngine(max_workers=8)

        # Small graph - should recommend serial
        workers = engine.recommend_workers(graph_size=30)
        assert workers == 1

        # Medium graph - should recommend limited parallelism
        workers = engine.recommend_workers(graph_size=200)
        assert 1 < workers <= engine.max_workers

        # Large graph - should use full parallelism
        workers = engine.recommend_workers(graph_size=1000)
        assert workers == engine.max_workers

    def test_compute_delta_nfr_parallel(self):
        """Test parallel ΔNFR computation."""
        from tnfr.parallel import TNFRParallelEngine

        # Create test network
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])

        # Initialize with TNFR attributes
        for node in G.nodes():
            G.nodes[node]["vf"] = 1.0
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["epi"] = 0.5
            G.nodes[node]["delta_nfr"] = 0.0

        engine = TNFRParallelEngine(max_workers=2)

        # Should execute without error
        result = engine.compute_delta_nfr_parallel(G)

        # Should return dict with all nodes
        assert isinstance(result, dict)
        assert set(result.keys()) == set(G.nodes())

        # All values should be floats
        for value in result.values():
            assert isinstance(value, float)

    def test_compute_si_parallel(self):
        """Test parallel Si computation."""
        from tnfr.parallel import TNFRParallelEngine

        # Create test network
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c")])

        # Initialize with TNFR attributes
        for node in G.nodes():
            G.nodes[node]["nu_f"] = 1.0  # Si uses "nu_f" alias
            G.nodes[node]["phase"] = 0.0
            G.nodes[node]["delta_nfr"] = 0.0

        engine = TNFRParallelEngine(max_workers=2)

        # Should execute without error
        result = engine.compute_si_parallel(G)

        # Should return dict with all nodes
        assert isinstance(result, dict)
        assert set(result.keys()) == set(G.nodes())

        # All values should be floats
        for value in result.values():
            assert isinstance(value, float)

    def test_integration_with_existing_api(self):
        """Test that engine integrates with existing TNFR dynamics."""
        from tnfr.parallel import TNFRParallelEngine
        from tnfr.structural import create_nfr

        # Use existing TNFR infrastructure
        G, node = create_nfr("test", epi=0.5, vf=1.0)

        # Add more nodes
        G.add_node("n2", epi=0.6, vf=1.1, phase=0.1, delta_nfr=0.0)
        G.add_node("n3", epi=0.4, vf=0.9, phase=0.2, delta_nfr=0.0)
        G.add_edge(node, "n2")
        G.add_edge("n2", "n3")

        engine = TNFRParallelEngine(max_workers=2)

        # Should work with existing graph structure
        result = engine.compute_delta_nfr_parallel(G)
        assert len(result) == 3
