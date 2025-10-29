from __future__ import annotations

from typing import Any

from .. import utils as _utils

__all__ = tuple(_utils.__all__) + ("__getattr__",)  # type: ignore[attr-defined]

def __getattr__(name: str) -> Any: ...
