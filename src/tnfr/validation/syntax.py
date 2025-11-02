"""Deprecated entry point for sequence validation.

Projects should import :func:`tnfr.operators.grammar.validate_sequence` directly;
this shim remains only for backwards compatibility and will be removed in a
future release.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import warnings

from . import ValidationOutcome
from ..operators import grammar as _canonical
from ..operators.grammar import validate_sequence as _validate_sequence

__all__ = ("validate_sequence",)

warnings.warn(
    "'tnfr.validation.syntax' is deprecated; use 'tnfr.operators.grammar' "
    "for sequence validation.",
    DeprecationWarning,
    stacklevel=2,
)


def validate_sequence(
    names: Iterable[str] | object = _canonical._MISSING, **kwargs: Any
) -> ValidationOutcome[tuple[str, ...]]:
    """Proxy to :func:`tnfr.operators.grammar.validate_sequence`."""

    warnings.warn(
        "'tnfr.validation.syntax.validate_sequence' is deprecated; "
        "use 'tnfr.operators.grammar.validate_sequence' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _validate_sequence(names, **kwargs)

