
from __future__ import annotations

"""JSON serialization helpers.

This module lazily imports :mod:`orjson` on first use of :func:`json_dumps`.
"""

import json
import warnings
import threading
from typing import Any, Callable

from .import_utils import optional_import

__all__ = ["json_dumps"]

_orjson: Any | None = None
_orjson_loaded = False
_ignored_param_warned = False
_warn_lock = threading.Lock()


def json_dumps(
    obj: Any,
    *,
    sort_keys: bool = False,
    default: Callable[[Any], Any] | None = None,
    ensure_ascii: bool = True,
    separators: tuple[str, str] = (",", ":"),
    cls: type[json.JSONEncoder] | None = None,
    to_bytes: bool = True,
    **kwargs: Any,
) -> bytes | str:
    """Serialize ``obj`` to JSON using ``orjson`` when available.

    When :mod:`orjson` is used, the ``ensure_ascii``, ``separators``, ``cls``
    and any additional keyword arguments are ignored because they are not
    supported by :func:`orjson.dumps`. A warning is emitted only the first time
    such ignored parameters are detected.
    """
    global _orjson, _orjson_loaded
    if not _orjson_loaded:
        _orjson = optional_import("orjson")
        _orjson_loaded = True
    if _orjson is not None:
        if (
            ensure_ascii is not True
            or separators != (",", ":")
            or cls is not None
            or kwargs
        ):
            global _ignored_param_warned
            with _warn_lock:
                if not _ignored_param_warned:
                    warnings.warn(
                        "'ensure_ascii', 'separators', 'cls' and extra kwargs are ignored when using orjson",
                        UserWarning,
                        stacklevel=2,
                    )
                    _ignored_param_warned = True
        option = _orjson.OPT_SORT_KEYS if sort_keys else 0
        data = _orjson.dumps(obj, option=option, default=default)
        return data if to_bytes else data.decode("utf-8")
    result = json.dumps(
        obj,
        sort_keys=sort_keys,
        ensure_ascii=ensure_ascii,
        separators=separators,
        cls=cls,
        default=default,
        **kwargs,
    )
    return result if not to_bytes else result.encode("utf-8")
