"""JSON helpers with optional :mod:`orjson` support.

This module lazily imports :mod:`orjson` on first use of :func:`json_dumps`.
The fast serializer is brought in through
``tnfr.import_utils.cached_import``; its cache and failure registry can be
reset using ``cached_import.cache_clear()`` and
:func:`tnfr.import_utils.prune_failed_imports`.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .import_utils import cached_import
from .logging_utils import get_logger
from .logging_utils import warn_once

_ORJSON_PARAMS_MSG = (
    "'ensure_ascii', 'separators', 'cls' and extra kwargs are ignored when using orjson: %s"
)

# Track combinations of parameters for which a warning has already been emitted.
logger = get_logger(__name__)
_warn_orjson_params_once = warn_once(logger, _ORJSON_PARAMS_MSG)


def _format_ignored_params(combo: frozenset[str]) -> str:
    """Return a stable representation for ignored parameter combinations."""
    return "{" + ", ".join(map(repr, sorted(combo))) + "}"


JsonDumpsParams = Mapping[str, Any]


_DEFAULT_PARAMS_DICT: dict[str, Any] = {
    "sort_keys": False,
    "default": None,
    "ensure_ascii": True,
    "separators": (",", ":"),
    "cls": None,
    "to_bytes": False,
}


DEFAULT_PARAMS: Mapping[str, Any] = MappingProxyType(_DEFAULT_PARAMS_DICT)


_ORJSON_PARAM_CHECKS: tuple[tuple[Callable[[JsonDumpsParams], bool], str], ...] = (
    (lambda p: p["ensure_ascii"] is not True, "ensure_ascii"),
    (lambda p: p["separators"] != (",", ":"), "separators"),
    (lambda p: p["cls"] is not None, "cls"),
)


def _json_dumps_orjson(
    orjson: Any,
    obj: Any,
    params: JsonDumpsParams,
    **kwargs: Any,
) -> bytes | str:
    """Serialize using :mod:`orjson` and warn about unsupported parameters."""
    ignored = {name for check, name in _ORJSON_PARAM_CHECKS if check(params)}
    if kwargs:
        ignored.update(kwargs)
    if ignored:
        combo = frozenset(ignored)
        _warn_orjson_params_once({combo: _format_ignored_params(combo)})

    option = orjson.OPT_SORT_KEYS if params["sort_keys"] else 0
    data = orjson.dumps(obj, option=option, default=params["default"])
    return data if params["to_bytes"] else data.decode("utf-8")


def _json_dumps_std(
    obj: Any,
    params: JsonDumpsParams,
    **kwargs: Any,
) -> bytes | str:
    """Serialize using the standard library :func:`json.dumps`."""
    result = json.dumps(
        obj,
        sort_keys=params["sort_keys"],
        ensure_ascii=params["ensure_ascii"],
        separators=params["separators"],
        cls=params["cls"],
        default=params["default"],
        **kwargs,
    )
    return result if not params["to_bytes"] else result.encode("utf-8")


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
    not supported by :func:`orjson.dumps`. A warning is emitted when such
    ignored parameters are detected and, by default, is shown only once per
    process.
    """
    if not isinstance(sort_keys, bool):
        raise TypeError("sort_keys must be a boolean")
    if default is not None and not callable(default):
        raise TypeError("default must be callable when provided")
    if not isinstance(ensure_ascii, bool):
        raise TypeError("ensure_ascii must be a boolean")
    if not isinstance(separators, tuple) or len(separators) != 2:
        raise TypeError("separators must be a tuple of two strings")
    if not all(isinstance(part, str) for part in separators):
        raise TypeError("separators must be a tuple of two strings")
    if cls is not None:
        if not isinstance(cls, type) or not issubclass(cls, json.JSONEncoder):
            raise TypeError("cls must be a subclass of json.JSONEncoder")
    if not isinstance(to_bytes, bool):
        raise TypeError("to_bytes must be a boolean")

    if (
        sort_keys is False
        and default is None
        and ensure_ascii is True
        and separators == (",", ":")
        and cls is None
        and to_bytes is False
    ):
        params: JsonDumpsParams = DEFAULT_PARAMS
    else:
        params = {
            "sort_keys": sort_keys,
            "default": default,
            "ensure_ascii": ensure_ascii,
            "separators": separators,
            "cls": cls,
            "to_bytes": to_bytes,
        }
    orjson = cached_import("orjson", emit="log")
    if orjson is not None:
        return _json_dumps_orjson(orjson, obj, params, **kwargs)
    return _json_dumps_std(obj, params, **kwargs)
