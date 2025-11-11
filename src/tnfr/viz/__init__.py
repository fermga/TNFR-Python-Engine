"""Visualization helpers for TNFR telemetry.

This module requires optional dependencies (numpy, matplotlib). Install with::

    pip install tnfr[viz]

or::

    pip install numpy matplotlib
"""

_import_error: ImportError | None = None

try:
    from .matplotlib import plot_coherence_matrix, plot_phase_sync, plot_spectrum_path

    __all__ = [
        "plot_coherence_matrix",
        "plot_phase_sync",
        "plot_spectrum_path",
    ]
except ImportError as _import_err:
    # matplotlib or numpy not available - provide informative stubs
    _import_error = _import_err
    from typing import Any as _Any

    def _missing_viz_dependency(*args: _Any, **kwargs: _Any) -> None:
        # Provide more specific error message based on what's missing
        missing_deps = []
        try:
            import numpy  # noqa: F401
        except ImportError:
            missing_deps.append("numpy")
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
            # Some other import error
            raise ImportError(
                "Visualization functions are not available. " "Install with: pip install tnfr[viz]"
            ) from _import_error

    plot_coherence_matrix = _missing_viz_dependency  # type: ignore[assignment]
    plot_phase_sync = _missing_viz_dependency  # type: ignore[assignment]
    plot_spectrum_path = _missing_viz_dependency  # type: ignore[assignment]

    __all__ = [
        "plot_coherence_matrix",
        "plot_phase_sync",
        "plot_spectrum_path",
    ]
