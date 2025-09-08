from __future__ import annotations

import json
from typing import Any

from .import_utils import optional_import

__all__ = ["fast_dumps"]

_orjson = optional_import("orjson")


def fast_dumps(obj: Any, *, sort_keys: bool = False) -> bytes:
    """Serialize ``obj`` to JSON using ``orjson`` when available.

    Returns ``bytes`` for compatibility with :func:`hashlib` and other
    consumers.  When ``orjson`` is missing the standard :mod:`json` module is
    used with compact separators to reduce overhead.
    """
    if _orjson is not None:
        option = _orjson.OPT_SORT_KEYS if sort_keys else 0
        return _orjson.dumps(obj, option=option)
    return json.dumps(
        obj, sort_keys=sort_keys, separators=(",", ":")
    ).encode("utf-8")
