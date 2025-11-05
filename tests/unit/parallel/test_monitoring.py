"""Tests for ParallelExecutionMonitor."""

import pytest
import time


class TestParallelExecutionMonitor:
    """Test suite for execution monitoring."""

    def test_import(self):
        """Test that monitor classes can be imported."""
        from tnfr.parallel.monitoring import (
            ParallelExecutionMonitor,
            PerformanceMetrics
        )
        assert ParallelExecutionMonitor is not None
        assert PerformanceMetrics is not None

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        from tnfr.parallel import ParallelExecutionMonitor

        monitor = ParallelExecutionMonitor()
        assert monitor._metrics_history == []
        assert monitor._current_metrics is None

    def test_start_stop_monitoring(self):
        """Test basic monitoring workflow."""
        from tnfr.parallel import ParallelExecutionMonitor

        monitor = ParallelExecutionMonitor()
        
        # Start monitoring
        monitor.start_monitoring(expected_nodes=100, workers=2)
        assert monitor._current_metrics is not None
        assert monitor._current_metrics["expected_nodes"] == 100
        assert monitor._current_metrics["workers"] == 2
        
        # Simulate some work
        time.sleep(0.1)
        
        # Stop monitoring
        metrics = monitor.stop_monitoring(
            final_coherence=0.85,
            initial_coherence=0.75
        )
        
        assert metrics.nodes_processed == 100
        assert metrics.workers_used == 2
        assert metrics.duration_seconds > 0
        assert abs(metrics.coherence_improvement - 0.10) < 0.001  # Float comparison

    def test_metrics_attributes(self):
        """Test that metrics contain all expected attributes."""
        from tnfr.parallel import ParallelExecutionMonitor

        monitor = ParallelExecutionMonitor()
        monitor.start_monitoring(expected_nodes=50, workers=1)
        time.sleep(0.05)
        metrics = monitor.stop_monitoring(
            final_coherence=0.9,
            initial_coherence=0.8
        )

        # Check all expected attributes exist
        assert hasattr(metrics, "start_time")
        assert hasattr(metrics, "end_time")
        assert hasattr(metrics, "duration_seconds")
        assert hasattr(metrics, "peak_memory_mb")
        assert hasattr(metrics, "avg_cpu_percent")
        assert hasattr(metrics, "workers_used")
        assert hasattr(metrics, "nodes_processed")
        assert hasattr(metrics, "operations_per_second")
        assert hasattr(metrics, "coherence_improvement")
        assert hasattr(metrics, "parallelization_efficiency")
        assert hasattr(metrics, "memory_efficiency")

    def test_operations_per_second(self):
        """Test throughput calculation."""
        from tnfr.parallel import ParallelExecutionMonitor

        monitor = ParallelExecutionMonitor()
        monitor.start_monitoring(expected_nodes=1000, workers=4)
        time.sleep(0.1)  # Simulate work
        metrics = monitor.stop_monitoring(
            final_coherence=0.9,
            initial_coherence=0.8
        )

        # Should have reasonable throughput
        assert metrics.operations_per_second > 0
        # 1000 nodes in ~0.1s should be >5000 ops/s
        assert metrics.operations_per_second > 5000

    def test_metrics_history(self):
        """Test that history is maintained."""
        from tnfr.parallel import ParallelExecutionMonitor

        monitor = ParallelExecutionMonitor()
        
        # Run multiple monitoring sessions
        for i in range(3):
            monitor.start_monitoring(expected_nodes=100 * (i + 1), workers=2)
            time.sleep(0.01)
            monitor.stop_monitoring(
                final_coherence=0.8 + i * 0.05,
                initial_coherence=0.7
            )
        
        # Should have all three in history
        history = monitor.history
        assert len(history) == 3
        
        # Check that history is properly ordered
        for i, metrics in enumerate(history):
            assert metrics.nodes_processed == 100 * (i + 1)

    def test_optimization_suggestions(self):
        """Test optimization suggestion generation."""
        from tnfr.parallel import ParallelExecutionMonitor

        monitor = ParallelExecutionMonitor()
        
        # No history initially - should return empty history message
        suggestions = monitor.get_optimization_suggestions()
        assert len(suggestions) == 1
        # Check for expected behavior rather than specific message text
        assert len(monitor.history) == 0
        
        # Add some execution
        monitor.start_monitoring(expected_nodes=100, workers=2)
        time.sleep(0.05)
        metrics = monitor.stop_monitoring(
            final_coherence=0.9,
            initial_coherence=0.8
        )
        
        # Should now have suggestions based on performance
        suggestions = monitor.get_optimization_suggestions()
        assert len(suggestions) > 0

    def test_stop_without_start_raises_error(self):
        """Test that stopping without starting raises error."""
        from tnfr.parallel import ParallelExecutionMonitor

        monitor = ParallelExecutionMonitor()
        
        with pytest.raises(RuntimeError, match="not started"):
            monitor.stop_monitoring(
                final_coherence=0.9,
                initial_coherence=0.8
            )

    def test_performance_metrics_dataclass(self):
        """Test PerformanceMetrics dataclass."""
        from tnfr.parallel.monitoring import PerformanceMetrics

        metrics = PerformanceMetrics(
            start_time=1000.0,
            end_time=1010.0,
            duration_seconds=10.0,
            peak_memory_mb=500.0,
            avg_cpu_percent=75.0,
            workers_used=4,
            nodes_processed=1000,
            operations_per_second=100.0,
            coherence_improvement=0.15,
            parallelization_efficiency=0.85,
            memory_efficiency=2.0
        )

        assert metrics.duration_seconds == 10.0
        assert metrics.workers_used == 4
        assert metrics.coherence_improvement == 0.15
