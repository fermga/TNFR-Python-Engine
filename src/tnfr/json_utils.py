
from __future__ import annotations

import json
import warnings
import threading
from typing import Any, Callable

from .import_utils import optional_import

__all__ = ["json_dumps"]

_orjson = optional_import("orjson")
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

    When :mod:`orjson` is used, the ``ensure_ascii`` and ``separators`` options
    are ignored because they are not supported by ``orjson.dumps``. A warning is
    emitted only the first time such ignored parameters are detected.
    """
    if _orjson is not None:
        if ensure_ascii is not True or separators != (",", ":"):
            global _ignored_param_warned
            with _warn_lock:
                if not _ignored_param_warned:
                    warnings.warn(
                        "'ensure_ascii' and 'separators' are ignored when using orjson",
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
