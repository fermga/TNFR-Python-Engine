"""JSON helpers with optional :mod:`orjson` support.

This module lazily imports :mod:`orjson` on first use of :func:`json_dumps`.
The fast serializer is brought in through
``tnfr.import_utils.cached_import``; its cache and failure registry can be
reset using ``cached_import.cache_clear()`` and
:func:`tnfr.import_utils.prune_failed_imports` or the local
:func:`clear_orjson_cache` helper.
"""

from __future__ import annotations

import json

from typing import Any, Callable, Literal, overload

from dataclasses import dataclass
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


def clear_orjson_cache() -> None:
    """Clear cached :mod:`orjson` module and warning state."""
    _warn_orjson_params_once.clear()
    cache_clear = getattr(cached_import, "cache_clear", None)
    if cache_clear:
        cache_clear()


@dataclass(slots=True, frozen=True)
class JsonDumpsParams:
    sort_keys: bool
    default: Callable[[Any], Any] | None
    ensure_ascii: bool
    separators: tuple[str, str]
    cls: type[json.JSONEncoder] | None
    to_bytes: bool


DEFAULT_PARAMS = JsonDumpsParams(
    sort_keys=False,
    default=None,
    ensure_ascii=True,
    separators=(",", ":"),
    cls=None,
    to_bytes=False,
)


def _make_params(
    *,
    sort_keys: bool = False,
    default: Callable[[Any], Any] | None = None,
    ensure_ascii: bool = True,
    separators: tuple[str, str] = (",", ":"),
    cls: type[json.JSONEncoder] | None = None,
    to_bytes: bool = False,
) -> JsonDumpsParams:
    """Return a :class:`JsonDumpsParams` applying defaults.

    The function reuses ``DEFAULT_PARAMS`` when all options match the
    canonical defaults so downstream serializers can rely on identity
    comparisons to avoid extra allocations.
    """
    if (
        sort_keys is False
        and default is None
        and ensure_ascii is True
        and separators == (",", ":")
        and cls is None
        and to_bytes is False
    ):
        return DEFAULT_PARAMS
    return JsonDumpsParams(
        sort_keys=sort_keys,
        default=default,
        ensure_ascii=ensure_ascii,
        separators=separators,
        cls=cls,
        to_bytes=to_bytes,
    )


def _json_dumps_orjson(
    orjson: Any,
    obj: Any,
    params: JsonDumpsParams,
    **kwargs: Any,
) -> bytes | str:
    """Serialize using :mod:`orjson` and warn about unsupported parameters."""
    checks = [
        (params.ensure_ascii is not True, "ensure_ascii"),
        (params.separators != (",", ":"), "separators"),
        (params.cls is not None, "cls"),
    ]
    ignored = {name for cond, name in checks if cond}
    if kwargs:
        ignored.update(kwargs)
    if ignored:
        combo = frozenset(ignored)
        _warn_orjson_params_once({combo: _format_ignored_params(combo)})

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
) -> bytes: ...


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
) -> str: ...


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
    params = _make_params(
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
