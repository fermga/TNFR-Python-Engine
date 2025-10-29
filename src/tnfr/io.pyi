from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__: tuple[str, ...]


class JsonDumpsParams:
    sort_keys: bool
    default: Callable[[Any], Any] | None
    ensure_ascii: bool
    separators: tuple[str, str]
    cls: type[json.JSONEncoder] | None
    to_bytes: bool


DEFAULT_PARAMS: JsonDumpsParams


def clear_orjson_param_warnings() -> None: ...


def _json_dumps_orjson(
    orjson: Any,
    obj: Any,
    params: JsonDumpsParams,
    **kwargs: Any,
) -> bytes | str: ...


def _json_dumps_std(
    obj: Any,
    params: JsonDumpsParams,
    **kwargs: Any,
) -> bytes | str: ...


def json_dumps(
    obj: Any,
    *,
    sort_keys: bool = ...,
    default: Callable[[Any], Any] | None = ...,
    ensure_ascii: bool = ...,
    separators: tuple[str, str] = ...,
    cls: type[json.JSONEncoder] | None = ...,
    to_bytes: bool = ...,
    **kwargs: Any,
) -> bytes | str: ...


class StructuredFileError(Exception):
    path: Path


TOMLDecodeError: type[BaseException]
YAMLError: type[BaseException]


def read_structured_file(path: Path) -> Any: ...


def safe_write(
    path: str | Path,
    write: Callable[[Any], Any],
    *,
    mode: str = ...,
    encoding: str | None = ...,
    atomic: bool = ...,
    sync: bool | None = ...,
    **open_kwargs: Any,
) -> None: ...
