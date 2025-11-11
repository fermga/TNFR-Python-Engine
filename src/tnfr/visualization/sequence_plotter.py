"""Advanced sequence visualizer for TNFR operator sequences.

This module implements comprehensive visualization tools for structural operator sequences,
including flow diagrams, health dashboards, pattern analysis, and frequency timelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from ..operators.health_analyzer import SequenceHealthMetrics

from ..config.operator_names import (
    COHERENCE,
    COUPLING,
    DISSONANCE,
    EMISSION,
    MUTATION,
    RECURSIVITY,
    RESONANCE,
    SELF_ORGANIZATION,
    SILENCE,
    TRANSITION,
    canonical_operator_name,
    operator_display_name,
)
from ..validation.compatibility import CompatibilityLevel, get_compatibility_level

__all__ = ["SequenceVisualizer"]


# Color mapping for compatibility levels
COMPATIBILITY_COLORS = {
    CompatibilityLevel.EXCELLENT: "#2ecc71",  # Green
    CompatibilityLevel.GOOD: "#3498db",  # Blue
    CompatibilityLevel.CAUTION: "#f39c12",  # Orange
    CompatibilityLevel.AVOID: "#e74c3c",  # Red
}

# Color mapping for frequency levels
FREQUENCY_COLORS = {
    "high": "#e74c3c",  # Red - high energy
    "medium": "#3498db",  # Blue - moderate
    "zero": "#95a5a6",  # Gray - paused
}

# Operator category colors for pattern analysis
OPERATOR_CATEGORY_COLORS = {
    "initiator": "#9b59b6",  # Purple
    "stabilizer": "#2ecc71",  # Green
    "transformer": "#e67e22",  # Orange
    "amplifier": "#e74c3c",  # Red
    "organizer": "#1abc9c",  # Teal
}


def _get_operator_category(operator: str) -> str:
    """Determine the structural category of an operator."""
    if operator == EMISSION:
        return "initiator"
    elif operator in {COHERENCE, SILENCE}:
        return "stabilizer"
    elif operator in {DISSONANCE, MUTATION, TRANSITION}:
        return "transformer"
    elif operator in {RESONANCE, COUPLING}:
        return "amplifier"
    elif operator in {SELF_ORGANIZATION, RECURSIVITY}:
        return "organizer"
    else:
        return "stabilizer"  # Default for other operators


class SequenceVisualizer:
    """Advanced visualizer for TNFR operator sequences.

    Provides multiple visualization types:
    - Sequence flow diagrams with transition compatibility coloring
    - Health metrics dashboards with radar charts
    - Pattern analysis with component highlighting
    - Frequency timelines showing structural evolution

    Examples
    --------
    >>> from tnfr.visualization import SequenceVisualizer
    >>> from tnfr.operators.grammar import validate_sequence_with_health
    >>>
    >>> sequence = ["emission", "reception", "coherence", "silence"]
    >>> result = validate_sequence_with_health(sequence)
    >>>
    >>> visualizer = SequenceVisualizer()
    >>> fig, ax = visualizer.plot_sequence_flow(sequence, result.health_metrics)
    """

    def __init__(self, figsize: Tuple[float, float] = (12, 8), dpi: int = 100):
        """Initialize the sequence visualizer.

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Default figure size for plots, by default (12, 8)
        dpi : int, optional
            Default DPI for plots, by default 100
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_sequence_flow(
        self,
        sequence: List[str],
        health_metrics: Optional[SequenceHealthMetrics] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot sequence flow diagram with compatibility-colored transitions.

        Creates a flow diagram showing operators as nodes with arrows representing
        transitions. Arrow colors indicate compatibility level (green=excellent,
        blue=good, orange=caution, red=avoid).

        Parameters
        ----------
        sequence : List[str]
            Sequence of operator names (canonical form)
        health_metrics : SequenceHealthMetrics, optional
            Health metrics to display alongside the flow
        save_path : str, optional
            Path to save the figure

        Returns
        -------
        Tuple[Figure, Axes]
            The matplotlib figure and axes objects

        Examples
        --------
        >>> visualizer = SequenceVisualizer()
        >>> sequence = ["emission", "coherence", "resonance", "silence"]
        >>> fig, ax = visualizer.plot_sequence_flow(sequence)
        >>> fig.savefig("flow.png")
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if not sequence:
            ax.text(0.5, 0.5, "Empty sequence", ha="center", va="center", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return fig, ax

        # Normalize operator names
        normalized = [canonical_operator_name(op) or op for op in sequence]

        # Calculate positions for operators
        n_ops = len(normalized)
        positions = {}

        if n_ops == 1:
            positions[0] = (0.5, 0.5)
        else:
            # Arrange in a flowing pattern
            for i, op in enumerate(normalized):
                x = 0.15 + (i / (n_ops - 1)) * 0.7
                # Add slight vertical variation for visual interest
                y = 0.5 + 0.1 * np.sin(i * np.pi / 3)
                positions[i] = (x, y)

        # Draw transitions with compatibility coloring
        for i in range(len(normalized) - 1):
            curr_op = normalized[i]
            next_op = normalized[i + 1]

            # Get compatibility level
            compat = get_compatibility_level(curr_op, next_op)
            color = COMPATIBILITY_COLORS.get(compat, "#95a5a6")

            # Draw arrow
            start = positions[i]
            end = positions[i + 1]

            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=2.5,
                    connectionstyle="arc3,rad=0.1",
                ),
            )

        # Draw operator nodes
        for i, op in enumerate(normalized):
            pos = positions[i]

            # Get operator category for coloring
            category = _get_operator_category(op)
            node_color = OPERATOR_CATEGORY_COLORS.get(category, "#95a5a6")

            # Note: Frequency-based styling removed (R5 constraint eliminated)
            # All operators now use standard border width
            border_width = 2

            # Draw node
            circle = plt.Circle(pos, 0.04, color=node_color, ec="black", lw=border_width, zorder=10)
            ax.add_patch(circle)

            # Add operator label
            display_name = operator_display_name(op) or op
            ax.text(
                pos[0],
                pos[1] - 0.08,
                display_name,
                ha="center",
                va="top",
                fontsize=10,
                weight="bold",
            )

        # Add title
        title = "TNFR Sequence Flow Diagram"
        if health_metrics:
            title += f"\nOverall Health: {health_metrics.overall_health:.2f}"
        ax.set_title(title, fontsize=14, weight="bold", pad=20)

        # Add legend
        legend_elements = [
            mpatches.Patch(
                color=COMPATIBILITY_COLORS[CompatibilityLevel.EXCELLENT],
                label="Excellent transition",
            ),
            mpatches.Patch(
                color=COMPATIBILITY_COLORS[CompatibilityLevel.GOOD],
                label="Good transition",
            ),
            mpatches.Patch(
                color=COMPATIBILITY_COLORS[CompatibilityLevel.CAUTION],
                label="Caution transition",
            ),
            mpatches.Patch(
                color=COMPATIBILITY_COLORS[CompatibilityLevel.AVOID],
                label="Avoid transition",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        # Add health metrics sidebar if provided
        if health_metrics:
            metrics_text = (
                f"Coherence: {health_metrics.coherence_index:.2f}\n"
                f"Balance: {health_metrics.balance_score:.2f}\n"
                f"Sustainability: {health_metrics.sustainability_index:.2f}\n"
                f"Pattern: {health_metrics.dominant_pattern}"
            )
            ax.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig, ax

    def plot_health_dashboard(
        self,
        health_metrics: SequenceHealthMetrics,
        save_path: Optional[str] = None,
    ) -> Tuple[Figure, np.ndarray]:
        """Plot comprehensive health metrics dashboard with radar chart.

        Creates a multi-panel dashboard showing:
        - Radar chart with all health metrics
        - Bar chart comparing metrics to benchmarks
        - Overall health gauge

        Parameters
        ----------
        health_metrics : SequenceHealthMetrics
            Health metrics to visualize
        save_path : str, optional
            Path to save the figure

        Returns
        -------
        Tuple[Figure, np.ndarray]
            The matplotlib figure and array of axes objects

        Examples
        --------
        >>> from tnfr.operators.grammar import validate_sequence_with_health
        >>> result = validate_sequence_with_health(["emission", "coherence"])
        >>> visualizer = SequenceVisualizer()
        >>> fig, axes = visualizer.plot_health_dashboard(result.health_metrics)
        """
        fig = plt.figure(figsize=(14, 10), dpi=self.dpi)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Create subplots
        ax_radar = fig.add_subplot(gs[0, 0], projection="polar")
        ax_bars = fig.add_subplot(gs[0, 1])
        ax_gauge = fig.add_subplot(gs[1, :])

        # --- Radar Chart ---
        metrics_labels = [
            "Coherence",
            "Balance",
            "Sustainability",
            "Efficiency",
            "Frequency",
            "Completeness",
            "Smoothness",
        ]
        metrics_values = [
            health_metrics.coherence_index,
            health_metrics.balance_score,
            health_metrics.sustainability_index,
            health_metrics.complexity_efficiency,
            health_metrics.frequency_harmony,
            health_metrics.pattern_completeness,
            health_metrics.transition_smoothness,
        ]

        # Number of variables
        num_vars = len(metrics_labels)

        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        metrics_values_plot = metrics_values + [metrics_values[0]]
        angles += angles[:1]

        # Plot radar chart
        ax_radar.plot(angles, metrics_values_plot, "o-", linewidth=2, color="#3498db")
        ax_radar.fill(angles, metrics_values_plot, alpha=0.25, color="#3498db")
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics_labels, size=9)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_radar.set_title("Health Metrics Radar", size=12, weight="bold", pad=20)
        ax_radar.grid(True)

        # --- Bar Chart ---
        # Define benchmark values for ideal sequences
        # These represent canonical TNFR targets for well-formed sequences
        BENCHMARK_COHERENCE = 0.7
        BENCHMARK_BALANCE = 0.6
        BENCHMARK_SUSTAINABILITY = 0.7
        BENCHMARK_EFFICIENCY = 0.6
        BENCHMARK_FREQUENCY = 0.8
        BENCHMARK_COMPLETENESS = 0.7
        BENCHMARK_SMOOTHNESS = 0.9

        benchmarks = [
            BENCHMARK_COHERENCE,
            BENCHMARK_BALANCE,
            BENCHMARK_SUSTAINABILITY,
            BENCHMARK_EFFICIENCY,
            BENCHMARK_FREQUENCY,
            BENCHMARK_COMPLETENESS,
            BENCHMARK_SMOOTHNESS,
        ]
        x_pos = np.arange(num_vars)
        width = 0.35

        bars1 = ax_bars.bar(
            x_pos - width / 2, metrics_values, width, label="Current", color="#3498db"
        )
        bars2 = ax_bars.bar(
            x_pos + width / 2,
            benchmarks,
            width,
            label="Benchmark",
            color="#95a5a6",
            alpha=0.6,
        )

        ax_bars.set_ylabel("Score", fontsize=10)
        ax_bars.set_title("Metrics vs Benchmarks", fontsize=12, weight="bold")
        ax_bars.set_xticks(x_pos)
        ax_bars.set_xticklabels(
            [label[:4] for label in metrics_labels], rotation=45, ha="right", fontsize=8
        )
        ax_bars.legend(fontsize=9)
        ax_bars.set_ylim(0, 1.1)
        ax_bars.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax_bars.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        # --- Overall Health Gauge ---
        overall = health_metrics.overall_health

        # Determine color based on health
        if overall >= 0.8:
            gauge_color = "#2ecc71"  # Excellent
            status = "EXCELLENT"
        elif overall >= 0.6:
            gauge_color = "#3498db"  # Good
            status = "GOOD"
        elif overall >= 0.4:
            gauge_color = "#f39c12"  # Fair
            status = "FAIR"
        else:
            gauge_color = "#e74c3c"  # Poor
            status = "NEEDS IMPROVEMENT"

        # Draw gauge background
        ax_gauge.barh(0, 1, height=0.3, color="#ecf0f1", left=0)
        # Draw gauge fill
        ax_gauge.barh(0, overall, height=0.3, color=gauge_color, left=0)

        # Add markers
        for i in range(0, 11):
            val = i / 10
            ax_gauge.axvline(val, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

        ax_gauge.set_xlim(0, 1)
        ax_gauge.set_ylim(-0.5, 0.5)
        ax_gauge.set_yticks([])
        ax_gauge.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_gauge.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])

        # Add overall health value and status
        ax_gauge.text(
            0.5,
            0.7,
            f"Overall Health: {overall:.3f}",
            ha="center",
            va="center",
            fontsize=16,
            weight="bold",
            transform=ax_gauge.transAxes,
        )
        ax_gauge.text(
            0.5,
            0.3,
            status,
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
            color=gauge_color,
            transform=ax_gauge.transAxes,
        )

        # Add metadata
        metadata_text = (
            f"Sequence Length: {health_metrics.sequence_length}\n"
            f"Dominant Pattern: {health_metrics.dominant_pattern}\n"
            f"Recommendations: {len(health_metrics.recommendations)}"
        )
        ax_gauge.text(
            0.02,
            -0.4,
            metadata_text,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax_gauge.set_title("Overall Structural Health", fontsize=14, weight="bold", pad=20)
        ax_gauge.spines["top"].set_visible(False)
        ax_gauge.spines["right"].set_visible(False)
        ax_gauge.spines["left"].set_visible(False)

        fig.suptitle("TNFR Sequence Health Dashboard", fontsize=16, weight="bold", y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig, np.array([ax_radar, ax_bars, ax_gauge])

    def plot_pattern_analysis(
        self,
        sequence: List[str],
        pattern: str,
        save_path: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot pattern analysis with component highlighting.

        Visualizes the detected pattern within the sequence, highlighting
        key components and their structural roles.

        Parameters
        ----------
        sequence : List[str]
            Sequence of operator names
        pattern : str
            Detected pattern name (e.g., "activation", "therapeutic")
        save_path : str, optional
            Path to save the figure

        Returns
        -------
        Tuple[Figure, Axes]
            The matplotlib figure and axes objects
        """
        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.dpi)

        if not sequence:
            ax.text(0.5, 0.5, "Empty sequence", ha="center", va="center", fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return fig, ax

        normalized = [canonical_operator_name(op) or op for op in sequence]
        n_ops = len(normalized)

        # Create horizontal layout
        x_positions = np.linspace(0.1, 0.9, n_ops)
        y_base = 0.5

        # Draw operators with category-based coloring
        for i, op in enumerate(normalized):
            category = _get_operator_category(op)
            color = OPERATOR_CATEGORY_COLORS.get(category, "#95a5a6")

            # Draw operator box
            box = mpatches.FancyBboxPatch(
                (x_positions[i] - 0.03, y_base - 0.08),
                0.06,
                0.16,
                boxstyle="round,pad=0.01",
                facecolor=color,
                edgecolor="black",
                linewidth=2,
                alpha=0.7,
            )
            ax.add_patch(box)

            # Add operator name
            display_name = operator_display_name(op) or op
            ax.text(
                x_positions[i],
                y_base,
                display_name,
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="white",
            )

            # Add category label below
            ax.text(
                x_positions[i],
                y_base - 0.15,
                category,
                ha="center",
                va="top",
                fontsize=7,
                style="italic",
            )

        # Draw connecting arrows
        for i in range(n_ops - 1):
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.03, y_base),
                xytext=(x_positions[i] + 0.03, y_base),
                arrowprops=dict(arrowstyle="->", lw=2, color="#34495e"),
            )

        # Add pattern name and description
        ax.text(
            0.5,
            0.85,
            f"Detected Pattern: {pattern.upper()}",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
            transform=ax.transAxes,
        )

        # Add legend for categories
        legend_elements = [
            mpatches.Patch(color=OPERATOR_CATEGORY_COLORS["initiator"], label="Initiator"),
            mpatches.Patch(color=OPERATOR_CATEGORY_COLORS["stabilizer"], label="Stabilizer"),
            mpatches.Patch(color=OPERATOR_CATEGORY_COLORS["transformer"], label="Transformer"),
            mpatches.Patch(color=OPERATOR_CATEGORY_COLORS["amplifier"], label="Amplifier"),
            mpatches.Patch(color=OPERATOR_CATEGORY_COLORS["organizer"], label="Organizer"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9, ncol=5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("TNFR Pattern Component Analysis", fontsize=14, weight="bold", pad=20)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig, ax

    def plot_operator_sequence(
        self,
        sequence: List[str],
        save_path: Optional[str] = None,
    ) -> Tuple[Figure, Axes]:
        """Plot simple timeline of operators through the sequence.

        Shows operator progression through the sequence with category-based coloring.
        Note: Frequency validation (R5) has been removed from TNFR grammar as it
        was not a fundamental physical constraint.

        Parameters
        ----------
        sequence : List[str]
            Sequence of operator names
        save_path : str, optional
            Path to save the figure

        Returns
        -------
        Tuple[Figure, Axes]
            The matplotlib figure and axes objects
        """
        fig, ax = plt.subplots(figsize=(14, 6), dpi=self.dpi)

        if not sequence:
            ax.text(0.5, 0.5, "Empty sequence", ha="center", va="center", fontsize=14)
            return fig, ax

        normalized = [canonical_operator_name(op) or op for op in sequence]

        # Map operators to categories for consistent visual grouping
        categories = [_get_operator_category(op) for op in normalized]
        category_values = {
            "generator": 3,
            "stabilizer": 2,
            "transformer": 3,
            "connector": 2,
            "closure": 1,
        }
        y_values = [category_values.get(cat, 2) for cat in categories]

        # Plot operator line
        x_pos = np.arange(len(normalized))
        ax.plot(
            x_pos,
            y_values,
            marker="o",
            markersize=12,
            linewidth=2.5,
            color="#3498db",
            label="Operator flow",
            zorder=2,
        )

        # Annotate operators with category colors
        for i, (op, cat) in enumerate(zip(normalized, categories)):
            display_name = operator_display_name(op) or op
            y_offset = 0.2 if i % 2 == 0 else -0.2

            cat_color = OPERATOR_CATEGORY_COLORS.get(cat, "#95a5a6")
            ax.annotate(
                display_name,
                xy=(x_pos[i], y_values[i]),
                xytext=(x_pos[i], y_values[i] + y_offset),
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor=cat_color,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1.5,
                ),
                zorder=3,
            )

        # Styling
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["Closure", "Moderate", "Intensive"], fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Step {i+1}" for i in range(len(normalized))], fontsize=9)
        ax.set_ylabel("Operator Intensity", fontsize=12, weight="bold")
        ax.set_xlabel("Sequence Position", fontsize=12, weight="bold")
        ax.set_title("TNFR Operator Sequence Timeline", fontsize=14, weight="bold", pad=20)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(0.5, 3.5)

        # Add category legend
        legend_elements = [
            mpatches.Patch(color=OPERATOR_CATEGORY_COLORS["generator"], label="Generator"),
            mpatches.Patch(color=OPERATOR_CATEGORY_COLORS["stabilizer"], label="Stabilizer"),
            mpatches.Patch(color=OPERATOR_CATEGORY_COLORS["transformer"], label="Transformer"),
            mpatches.Patch(color=OPERATOR_CATEGORY_COLORS["connector"], label="Connector"),
            mpatches.Patch(color=OPERATOR_CATEGORY_COLORS["closure"], label="Closure"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9, ncol=2)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        return fig, ax
