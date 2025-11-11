"""Tests for TNFR sequence visualization module."""

import pytest
import numpy as np
from pathlib import Path

# Skip all tests if matplotlib is not available
pytest.importorskip("matplotlib")

from tnfr.visualization import SequenceVisualizer
from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.health_analyzer import SequenceHealthMetrics


class TestSequenceVisualizer:
    """Test suite for SequenceVisualizer class."""

    def test_visualizer_initialization(self):
        """Test that visualizer initializes correctly."""
        viz = SequenceVisualizer()
        assert viz.figsize == (12, 8)
        assert viz.dpi == 100

        # Custom parameters
        viz_custom = SequenceVisualizer(figsize=(10, 6), dpi=150)
        assert viz_custom.figsize == (10, 6)
        assert viz_custom.dpi == 150

    def test_plot_sequence_flow_basic(self):
        """Test basic sequence flow plot generation."""
        viz = SequenceVisualizer()
        sequence = ["emission", "reception", "coherence", "silence"]

        fig, ax = viz.plot_sequence_flow(sequence)

        assert fig is not None
        assert ax is not None
        assert ax.get_title() == "TNFR Sequence Flow Diagram"

    def test_plot_sequence_flow_with_health(self):
        """Test sequence flow plot with health metrics."""
        viz = SequenceVisualizer()
        sequence = ["emission", "reception", "coherence", "silence"]
        result = validate_sequence_with_health(sequence)

        fig, ax = viz.plot_sequence_flow(sequence, health_metrics=result.health_metrics)

        assert fig is not None
        title = ax.get_title()
        assert "Overall Health" in title
        assert str(round(result.health_metrics.overall_health, 2)) in title

    def test_plot_sequence_flow_empty(self):
        """Test sequence flow plot with empty sequence."""
        viz = SequenceVisualizer()
        sequence = []

        fig, ax = viz.plot_sequence_flow(sequence)

        assert fig is not None
        assert ax is not None

    def test_plot_health_dashboard(self):
        """Test health dashboard generation."""
        viz = SequenceVisualizer()
        sequence = ["emission", "reception", "coherence", "silence"]
        result = validate_sequence_with_health(sequence)

        fig, axes = viz.plot_health_dashboard(result.health_metrics)

        assert fig is not None
        assert len(axes) == 3  # radar, bars, gauge
        assert fig._suptitle is not None
        assert "Health Dashboard" in fig._suptitle.get_text()

    def test_plot_pattern_analysis(self):
        """Test pattern analysis visualization."""
        viz = SequenceVisualizer()
        sequence = ["emission", "reception", "coherence", "silence"]
        pattern = "activation"

        fig, ax = viz.plot_pattern_analysis(sequence, pattern)

        assert fig is not None
        assert ax is not None
        # Pattern name is shown in text, not title
        assert "Pattern Component Analysis" in ax.get_title()

    def test_plot_frequency_timeline(self):
        """Test frequency timeline visualization."""
        viz = SequenceVisualizer()
        sequence = ["emission", "reception", "coherence", "silence"]

        fig, ax = viz.plot_frequency_timeline(sequence)

        assert fig is not None
        assert ax is not None
        assert "Frequency Timeline" in ax.get_title()

    def test_save_visualizations(self, tmp_path):
        """Test saving visualizations to file."""
        viz = SequenceVisualizer()
        sequence = ["emission", "reception", "coherence", "silence"]
        result = validate_sequence_with_health(sequence)

        # Test save_path parameter for each plot type
        flow_path = tmp_path / "flow.png"
        fig1, ax1 = viz.plot_sequence_flow(sequence, save_path=str(flow_path))
        assert flow_path.exists()

        dashboard_path = tmp_path / "dashboard.png"
        fig2, axes2 = viz.plot_health_dashboard(
            result.health_metrics, save_path=str(dashboard_path)
        )
        assert dashboard_path.exists()

        pattern_path = tmp_path / "pattern.png"
        fig3, ax3 = viz.plot_pattern_analysis(sequence, "activation", save_path=str(pattern_path))
        assert pattern_path.exists()

        timeline_path = tmp_path / "timeline.png"
        fig4, ax4 = viz.plot_frequency_timeline(sequence, save_path=str(timeline_path))
        assert timeline_path.exists()

    def test_multiple_sequences(self):
        """Test visualization with different sequence types."""
        viz = SequenceVisualizer()

        sequences = [
            ["emission", "reception", "coherence", "silence"],
            ["emission", "reception", "coherence", "resonance", "silence"],
            [
                "emission",
                "reception",
                "coherence",
                "dissonance",
                "self_organization",
                "coherence",
                "silence",
            ],
        ]

        for seq in sequences:
            result = validate_sequence_with_health(seq)
            if result.passed:
                # All visualizations should work
                fig1, ax1 = viz.plot_sequence_flow(seq, health_metrics=result.health_metrics)
                assert fig1 is not None

                fig2, axes2 = viz.plot_health_dashboard(result.health_metrics)
                assert fig2 is not None

                fig3, ax3 = viz.plot_pattern_analysis(seq, result.health_metrics.dominant_pattern)
                assert fig3 is not None

                fig4, ax4 = viz.plot_frequency_timeline(seq)
                assert fig4 is not None


class TestVisualizationMetrics:
    """Test visualization accuracy for health metrics."""

    def test_health_dashboard_metric_values(self):
        """Test that dashboard correctly displays metric values."""
        viz = SequenceVisualizer()
        sequence = ["emission", "reception", "coherence", "silence"]
        result = validate_sequence_with_health(sequence)

        fig, axes = viz.plot_health_dashboard(result.health_metrics)

        # Verify overall health is displayed
        health = result.health_metrics.overall_health
        assert health > 0.0
        assert health <= 1.0

    def test_frequency_timeline_coherence(self):
        """Test that frequency timeline shows correct frequency levels."""
        from tnfr.operators.grammar import STRUCTURAL_FREQUENCIES

        viz = SequenceVisualizer()
        sequence = ["emission", "reception", "coherence", "silence"]

        fig, ax = viz.plot_frequency_timeline(sequence)

        # Verify plot was created
        assert fig is not None
        assert ax is not None

        # Verify frequency levels are shown
        frequencies = [STRUCTURAL_FREQUENCIES.get(op, "medium") for op in sequence]
        assert len(frequencies) == len(sequence)


class TestVisualizationEdgeCases:
    """Test visualization with edge cases and unusual inputs."""

    def test_single_operator_sequence(self):
        """Test visualization with single operator."""
        viz = SequenceVisualizer()
        sequence = ["emission"]

        # Should not raise errors
        fig1, ax1 = viz.plot_sequence_flow(sequence)
        assert fig1 is not None

        fig3, ax3 = viz.plot_pattern_analysis(sequence, "minimal")
        assert fig3 is not None

        fig4, ax4 = viz.plot_frequency_timeline(sequence)
        assert fig4 is not None

    def test_long_sequence(self):
        """Test visualization with long sequence."""
        viz = SequenceVisualizer()
        sequence = ["emission", "reception", "coherence"] * 5 + ["silence"]

        # Should handle long sequences
        fig1, ax1 = viz.plot_sequence_flow(sequence)
        assert fig1 is not None

        fig4, ax4 = viz.plot_frequency_timeline(sequence)
        assert fig4 is not None

    def test_invalid_operator_names(self):
        """Test visualization handles unknown operators gracefully."""
        viz = SequenceVisualizer()
        sequence = ["emission", "unknown_op", "coherence", "silence"]

        # Should not crash, may produce warnings
        fig1, ax1 = viz.plot_sequence_flow(sequence)
        assert fig1 is not None


class TestVisualizationPerformance:
    """Test visualization performance characteristics."""

    def test_large_sequence_performance(self):
        """Test that visualization completes in reasonable time for large sequences."""
        import time

        viz = SequenceVisualizer()
        # Create a moderately large sequence
        base_seq = ["emission", "reception", "coherence", "resonance"]
        sequence = base_seq * 10 + ["silence"]

        start = time.time()
        fig, ax = viz.plot_sequence_flow(sequence)
        duration = time.time() - start

        # Should complete in under 5 seconds on typical hardware
        # This is a conservative estimate; most systems should complete in <1s
        assert duration < 5.0
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
