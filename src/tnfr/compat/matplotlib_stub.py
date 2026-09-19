"""Lightweight stub for matplotlib when it's not installed.

This stub provides minimal type compatibility for matplotlib when it's not
installed, allowing type checking to succeed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["pyplot", "axes", "figure"]


class _NotInstalledError(RuntimeError):
    """Raised when trying to use matplotlib operations without matplotlib installed."""

    def __init__(self, operation: str = "matplotlib operation") -> None:
        super().__init__(
            f"Cannot perform {operation}: matplotlib is not installed. "
            "Install it with: pip install tnfr[viz] or pip install matplotlib"
        )


class Axes:
    """Stub for matplotlib.axes.Axes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise _NotInstalledError("Axes creation")


class Figure:
    """Stub for matplotlib.figure.Figure."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise _NotInstalledError("Figure creation")


class _PyPlotStub:
    """Stub for matplotlib.pyplot module."""

    @staticmethod
    def subplots(*args: Any, **kwargs: Any) -> tuple[Figure, Axes]:
        raise _NotInstalledError("pyplot.subplots")

    @staticmethod
    def savefig(*args: Any, **kwargs: Any) -> None:
        raise _NotInstalledError("pyplot.savefig")

    @staticmethod
    def show(*args: Any, **kwargs: Any) -> None:
        raise _NotInstalledError("pyplot.show")

    @staticmethod
    def figure(*args: Any, **kwargs: Any) -> Figure:
        raise _NotInstalledError("pyplot.figure")


class _AxesStub:
    """Stub for matplotlib.axes module."""

    Axes = Axes


class _FigureStub:
    """Stub for matplotlib.figure module."""

    Figure = Figure


# Module-level stubs
pyplot = _PyPlotStub()
axes = _AxesStub()
figure = _FigureStub()
