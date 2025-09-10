"""JSON serialization helpers.

This module lazily imports :mod:`orjson` on first use of :func:`json_dumps`.
"""

from __future__ import annotations

import json
import warnings
import threading
from typing import Any, Callable, overload, Literal

from dataclasses import dataclass
from functools import lru_cache
from .import_utils import optional_import

__all__ = ("json_dumps", "json_dumps_str")

_ignored_param_warned = False
_warn_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load_orjson() -> Any | None:
    """Lazily import :mod:`orjson` once."""
    return optional_import("orjson")


@dataclass
class JsonDumpsParams:
    sort_keys: bool
    default: Callable[[Any], Any] | None
    ensure_ascii: bool
    separators: tuple[str, str]
    cls: type[json.JSONEncoder] | None
    to_bytes: bool


def _json_dumps_orjson(
    orjson: Any,
    obj: Any,
    params: JsonDumpsParams,
    **kwargs: Any,
) -> bytes | str:
    """Serialize using :mod:`orjson` and warn about unsupported parameters."""
    if (
        params.ensure_ascii is not True
        or params.separators != (",", ":")
        or params.cls is not None
        or kwargs
    ):
        global _ignored_param_warned
        with _warn_lock:
            if not _ignored_param_warned:
                warnings.warn(
                    "'ensure_ascii', 'separators', 'cls' and extra kwargs are "
                    "ignored when using orjson",
                    UserWarning,
                    stacklevel=3,
                )
                _ignored_param_warned = True
    option = orjson.OPT_SORT_KEYS if params.sort_keys else 0
    data = orjson.dumps(obj, option=option, default=params.default)
    return data if params.to_bytes else data.decode("utf-8")


def _json_dumps_std(
    obj: Any,
    params: JsonDumpsParams,
    **kwargs: Any,
) -> bytes | str:
    """Serialize using the standard library :func:`json.dumps`."""
    result = json.dumps(
        obj,
        sort_keys=params.sort_keys,
        ensure_ascii=params.ensure_ascii,
        separators=params.separators,
        cls=params.cls,
        default=params.default,
        **kwargs,
    )
    return result if not params.to_bytes else result.encode("utf-8")


@overload
def json_dumps(
    obj: Any,
    *,
    sort_keys: bool = ...,
    default: Callable[[Any], Any] | None = ...,
    ensure_ascii: bool = ...,
    separators: tuple[str, str] = ...,
    cls: type[json.JSONEncoder] | None = ...,
    to_bytes: Literal[True] = ...,
    **kwargs: Any,
) -> bytes:
    ...


@overload
def json_dumps(
    obj: Any,
    *,
    sort_keys: bool = ...,
    default: Callable[[Any], Any] | None = ...,
    ensure_ascii: bool = ...,
    separators: tuple[str, str] = ...,
    cls: type[json.JSONEncoder] | None = ...,
    to_bytes: Literal[False],
    **kwargs: Any,
) -> str:
    ...


def json_dumps(
    obj: Any,
    *,
    sort_keys: bool = False,
    default: Callable[[Any], Any] | None = None,
    ensure_ascii: bool = True,
    separators: tuple[str, str] = (",", ":"),
    cls: type[json.JSONEncoder] | None = None,
    to_bytes: bool = False,
    **kwargs: Any,
) -> bytes | str:
    """Serialize ``obj`` to JSON using ``orjson`` when available.

    Returns a ``str`` by default. Pass ``to_bytes=True`` to obtain a ``bytes``
    result. When :mod:`orjson` is used, the ``ensure_ascii``, ``separators``,
    ``cls`` and any additional keyword arguments are ignored because they are
    not supported by :func:`orjson.dumps`. A warning is emitted only the first
    time such ignored parameters are detected.
    """
    params = JsonDumpsParams(
        sort_keys=sort_keys,
        default=default,
        ensure_ascii=ensure_ascii,
        separators=separators,
        cls=cls,
        to_bytes=to_bytes,
    )
    orjson = _load_orjson()
    if orjson is not None:
        return _json_dumps_orjson(orjson, obj, params, **kwargs)
    return _json_dumps_std(obj, params, **kwargs)


def json_dumps_str(obj: Any, **kwargs: Any) -> str:
    """``json_dumps`` wrapper that always returns ``str``."""
    return json_dumps(obj, to_bytes=False, **kwargs)
