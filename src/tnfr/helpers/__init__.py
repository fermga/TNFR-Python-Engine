"""Compatibility layer for :mod:`tnfr.helpers`.

This package is deprecated in favour of :mod:`tnfr.utils`. Importing it emits a
``DeprecationWarning`` while continuing to expose the utilities provided by
``tnfr.utils`` for backwards compatibility.
"""

from __future__ import annotations

import warnings
from typing import Any

from .. import utils as _utils

__all__ = tuple(_utils.__all__) + ("__getattr__",)  # type: ignore[attr-defined]

warnings.warn(
    "'tnfr.helpers' is deprecated and will be removed in a future release; "
    "import from 'tnfr.utils' instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str) -> Any:  # pragma: no cover - simple delegation
    return getattr(_utils, name)


def __dir__() -> list[str]:  # pragma: no cover - simple reflection
    return sorted(__all__)
