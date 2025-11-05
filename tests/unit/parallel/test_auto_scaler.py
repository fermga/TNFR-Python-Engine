"""Tests for TNFRAutoScaler."""

import pytest


class TestTNFRAutoScaler:
    """Test suite for auto-scaling recommendations."""

    def test_import(self):
        """Test that TNFRAutoScaler can be imported."""
        from tnfr.parallel import TNFRAutoScaler
        assert TNFRAutoScaler is not None

    def test_scaler_initialization(self):
        """Test scaler initialization."""
        from tnfr.parallel import TNFRAutoScaler

        scaler = TNFRAutoScaler()
        assert scaler.performance_history == {}
        assert scaler.optimal_configs == {}

    def test_recommend_sequential(self):
        """Test recommendation for small networks."""
        from tnfr.parallel import TNFRAutoScaler

        scaler = TNFRAutoScaler()
        strategy = scaler.recommend_execution_strategy(
            graph_size=50,
            available_memory_gb=8.0,
            has_gpu=False
        )

        assert strategy["backend"] == "sequential"
        assert strategy["workers"] == 1
        assert "explanation" in strategy
        assert "estimated_time_minutes" in strategy
        assert "estimated_memory_gb" in strategy

    def test_recommend_multiprocessing(self):
        """Test recommendation for medium networks."""
        from tnfr.parallel import TNFRAutoScaler

        scaler = TNFRAutoScaler()
        strategy = scaler.recommend_execution_strategy(
            graph_size=500,
            available_memory_gb=8.0,
            has_gpu=False
        )

        assert strategy["backend"] == "multiprocessing"
        assert strategy["workers"] > 1
        assert "explanation" in strategy

    def test_recommend_gpu(self):
        """Test recommendation for large networks with GPU."""
        from tnfr.parallel import TNFRAutoScaler

        scaler = TNFRAutoScaler()
        strategy = scaler.recommend_execution_strategy(
            graph_size=5000,
            available_memory_gb=16.0,
            has_gpu=True
        )

        assert strategy["backend"] == "gpu"
        assert "gpu_engine" in strategy
        assert strategy["gpu_engine"] == "jax"

    def test_recommend_distributed(self):
        """Test recommendation for massive networks."""
        from tnfr.parallel import TNFRAutoScaler

        scaler = TNFRAutoScaler()
        strategy = scaler.recommend_execution_strategy(
            graph_size=50000,
            available_memory_gb=8.0,
            has_gpu=False
        )

        assert strategy["backend"] == "distributed"
        assert strategy["workers"] > 1
        assert "chunk_size" in strategy

    def test_memory_warning(self):
        """Test memory warning for large networks."""
        from tnfr.parallel import TNFRAutoScaler

        scaler = TNFRAutoScaler()
        strategy = scaler.recommend_execution_strategy(
            graph_size=1000000,  # Much larger to trigger warning
            available_memory_gb=0.1,  # Very limited memory
            has_gpu=False
        )

        # Should generate warning for truly large networks
        assert "warning" in strategy
        assert "recommendation" in strategy

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        from tnfr.parallel import TNFRAutoScaler

        scaler = TNFRAutoScaler()
        
        # Sequential should have lowest memory
        mem_seq = scaler._estimate_memory_usage(1000, "sequential")
        
        # GPU should have highest (needs CPU + GPU copies)
        mem_gpu = scaler._estimate_memory_usage(1000, "gpu")
        
        assert mem_seq > 0
        assert mem_gpu > mem_seq

    def test_time_estimation(self):
        """Test execution time estimation."""
        from tnfr.parallel import TNFRAutoScaler

        scaler = TNFRAutoScaler()
        
        # Sequential should be slowest
        time_seq = scaler._estimate_execution_time(1000, "sequential")
        
        # GPU should be fastest
        time_gpu = scaler._estimate_execution_time(1000, "gpu")
        
        assert time_seq > 0
        assert time_gpu < time_seq

    def test_optimization_suggestions(self):
        """Test optimization suggestion generation."""
        from tnfr.parallel import TNFRAutoScaler

        scaler = TNFRAutoScaler()
        
        # Test with poor performance metrics
        poor_metrics = {
            "parallelization_efficiency": 0.2,
            "memory_efficiency": 0.05,
            "operations_per_second": 50
        }
        
        suggestions = scaler.get_optimization_suggestions(poor_metrics)
        assert len(suggestions) >= 3  # Should have multiple suggestions
        
        # Test with good performance metrics
        good_metrics = {
            "parallelization_efficiency": 0.9,
            "memory_efficiency": 0.8,
            "operations_per_second": 1000
        }
        
        suggestions = scaler.get_optimization_suggestions(good_metrics)
        assert any("optimal" in s.lower() for s in suggestions)
