"""Deprecated compatibility shim for the historical mathematics validators.

Import :mod:`tnfr.validation` (or :mod:`tnfr.validation.spectral`) directly
instead of relying on this module.  It will be removed in a future release
after the ecosystem migrates to the unified validation namespace.
"""

from __future__ import annotations

import warnings
from typing import Any

__all__ = ("NFRValidator",)


def __getattr__(name: str) -> Any:
    if name == "NFRValidator":
        warnings.warn(
            "tnfr.mathematics.validators is deprecated; import from "
            "'tnfr.validation' (preferred) or 'tnfr.validation.spectral' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from ..validation.spectral import NFRValidator as _NFRValidator

        return _NFRValidator
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(__all__))

