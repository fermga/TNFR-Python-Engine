"""Public pattern-plot rendering remains independent of physics constants."""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib import pyplot as plt

from tnfr.visualization.sequence_plotter import SequenceVisualizer


def test_nonempty_pattern_analysis_renders_and_saves_operator_labels(tmp_path):
    sequence = ["emission", "coherence", "silence"]
    original = list(sequence)
    destination = tmp_path / "pattern.png"
    existing_figures = set(plt.get_fignums())
    try:
        figure, axes = SequenceVisualizer().plot_pattern_analysis(
            sequence, "activation", save_path=str(destination)
        )
        figure.canvas.draw()
        assert axes.figure is figure
        assert len(axes.patches) == len(sequence)
        assert set(sequence).issubset({label.get_text() for label in axes.texts})
        assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert sequence == original
    finally:
        for figure_number in set(plt.get_fignums()) - existing_figures:
            plt.close(figure_number)
