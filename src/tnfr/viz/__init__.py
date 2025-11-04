"""Visualization helpers for TNFR telemetry.

This module requires optional dependencies (numpy, matplotlib). Install with::

    pip install tnfr[viz]

or::

    pip install numpy matplotlib
"""

try:
    from .matplotlib import plot_coherence_matrix, plot_phase_sync, plot_spectrum_path

    __all__ = [
        "plot_coherence_matrix",
        "plot_phase_sync",
        "plot_spectrum_path",
    ]
except ImportError as _import_err:
    # matplotlib or numpy not available - provide informative stubs
    from typing import Any as _Any

    def _missing_viz_dependency(*args: _Any, **kwargs: _Any) -> None:
        raise ImportError(
            "Visualization functions require numpy and matplotlib. "
            "Install them with: pip install tnfr[viz]"
        ) from _import_err

    plot_coherence_matrix = _missing_viz_dependency  # type: ignore[assignment]
    plot_phase_sync = _missing_viz_dependency  # type: ignore[assignment]
    plot_spectrum_path = _missing_viz_dependency  # type: ignore[assignment]

    __all__ = [
        "plot_coherence_matrix",
        "plot_phase_sync",
        "plot_spectrum_path",
    ]
