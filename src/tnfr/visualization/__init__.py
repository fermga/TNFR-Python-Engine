"""Visualization tools for TNFR operator sequences and structural analysis.

This module provides advanced visualization capabilities for:
- Sequence flow diagrams with compatibility-colored transitions
- Health metrics dashboards with radar charts and gauges
- Pattern analysis with component highlighting
- Frequency timelines showing structural evolution
- Cascade propagation and temporal dynamics
- Hierarchical bifurcation structures (NEW)

Requires matplotlib for plotting (optional). Install with::

    pip install tnfr[viz]

Hierarchy visualization (ASCII) has no external dependencies.

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
>>> fig.savefig("sequence_flow.png")

>>> # Cascade visualization
>>> from tnfr.visualization import plot_cascade_propagation, plot_cascade_timeline
>>> fig = plot_cascade_propagation(G)
>>> fig.savefig("cascade_propagation.png")

>>> # Hierarchy visualization (no matplotlib required)
>>> from tnfr.visualization import print_bifurcation_hierarchy
>>> print_bifurcation_hierarchy(G, node)
"""

# Always available (no dependencies)
from .hierarchy import (
    print_bifurcation_hierarchy,
    get_hierarchy_info,
)

_import_error: ImportError | None = None

try:
    from .sequence_plotter import SequenceVisualizer
    from .cascade_viz import (
        plot_cascade_propagation,
        plot_cascade_timeline,
        plot_cascade_metrics_summary,
    )

    __all__ = [
        "SequenceVisualizer",
        "plot_cascade_propagation",
        "plot_cascade_timeline",
        "plot_cascade_metrics_summary",
        "print_bifurcation_hierarchy",
        "get_hierarchy_info",
    ]
except ImportError as _import_err:
    _import_error = _import_err
    from typing import Any as _Any

    def _missing_viz_dependency(*args: _Any, **kwargs: _Any) -> None:
        missing_deps = []
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            missing_deps.append("matplotlib")

        if missing_deps:
            deps_str = " and ".join(missing_deps)
            raise ImportError(
                f"Visualization functions require {deps_str}. "
                "Install with: pip install tnfr[viz]"
            ) from _import_error
        else:
            raise ImportError(
                "Visualization functions are not available. " "Install with: pip install tnfr[viz]"
            ) from _import_error

    SequenceVisualizer = _missing_viz_dependency  # type: ignore[assignment]
    plot_cascade_propagation = _missing_viz_dependency  # type: ignore[assignment]
    plot_cascade_timeline = _missing_viz_dependency  # type: ignore[assignment]
    plot_cascade_metrics_summary = _missing_viz_dependency  # type: ignore[assignment]

    __all__ = [
        "SequenceVisualizer",
        "plot_cascade_propagation",
        "plot_cascade_timeline",
        "plot_cascade_metrics_summary",
        "print_bifurcation_hierarchy",
        "get_hierarchy_info",
    ]
