"""Serialisation helpers for TNFR, including JSON convenience wrappers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable

from .init import cached_import, get_logger, warn_once

__all__ = (
    "JsonDumpsParams",
    "DEFAULT_PARAMS",
    "json_dumps",
    "clear_orjson_param_warnings",
)

_ORJSON_PARAMS_MSG = (
    "'ensure_ascii', 'separators', 'cls' and extra kwargs are ignored when using orjson: %s"
)

logger = get_logger(__name__)

_warn_ignored_params_once = warn_once(logger, _ORJSON_PARAMS_MSG)


def clear_orjson_param_warnings() -> None:
    """Reset cached warnings for ignored :mod:`orjson` parameters."""

    _warn_ignored_params_once.clear()


def _format_ignored_params(combo: frozenset[str]) -> str:
    """Return a stable representation for ignored parameter combinations."""

    return "{" + ", ".join(map(repr, sorted(combo))) + "}"


@dataclass(frozen=True)
class JsonDumpsParams:
    """Container describing the parameters used by :func:`json_dumps`."""

    sort_keys: bool = False
    default: Callable[[Any], Any] | None = None
    ensure_ascii: bool = True
    separators: tuple[str, str] = (",", ":")
    cls: type[json.JSONEncoder] | None = None
    to_bytes: bool = False


DEFAULT_PARAMS = JsonDumpsParams()


def _collect_ignored_params(
    params: JsonDumpsParams, extra_kwargs: dict[str, Any]
) -> frozenset[str]:
    """Return a stable set of parameters ignored by :mod:`orjson`."""

    ignored: set[str] = set()
    if params.ensure_ascii is not True:
        ignored.add("ensure_ascii")
    if params.separators != (",", ":"):
        ignored.add("separators")
    if params.cls is not None:
        ignored.add("cls")
    if extra_kwargs:
        ignored.update(extra_kwargs.keys())
    return frozenset(ignored)


def _json_dumps_orjson(
    orjson: Any,
    obj: Any,
    params: JsonDumpsParams,
    **kwargs: Any,
) -> bytes | str:
    """Serialize using :mod:`orjson` and warn about unsupported parameters."""

    ignored = _collect_ignored_params(params, kwargs)
    if ignored:
        _warn_ignored_params_once(ignored, _format_ignored_params(ignored))

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
    """Serialize ``obj`` to JSON using ``orjson`` when available."""

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
        params = DEFAULT_PARAMS
    else:
        params = JsonDumpsParams(
            sort_keys=sort_keys,
            default=default,
            ensure_ascii=ensure_ascii,
            separators=separators,
            cls=cls,
            to_bytes=to_bytes,
        )
    orjson = cached_import("orjson", emit="log")
    if orjson is not None:
        return _json_dumps_orjson(orjson, obj, params, **kwargs)
    return _json_dumps_std(obj, params, **kwargs)
