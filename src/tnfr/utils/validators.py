"""Compatibility layer for :mod:`tnfr.validation.graph`.

The canonical graph validation APIs now live under :mod:`tnfr.validation`.
This module preserves backwards compatibility for callers still importing
through :mod:`tnfr.utils`, while emitting a :class:`DeprecationWarning` to
encourage migration.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..validation.graph import GRAPH_VALIDATORS
from ..validation.graph import run_validators as _run_validators
from ..validation.graph import validate_window as _validate_window

__all__ = ("validate_window", "run_validators", "GRAPH_VALIDATORS")

_WARNING = (
    "tnfr.utils.validators is deprecated; import from tnfr.validation instead"
)


def _warn() -> None:
    warnings.warn(_WARNING, DeprecationWarning, stacklevel=3)


def validate_window(window: int, *, positive: bool = False) -> int:
    """Delegate to :func:`tnfr.validation.validate_window` with a warning."""

    _warn()
    return _validate_window(window, positive=positive)


def run_validators(graph: Any) -> None:
    """Delegate to :func:`tnfr.validation.run_validators` with a warning."""

    _warn()
    _run_validators(graph)
