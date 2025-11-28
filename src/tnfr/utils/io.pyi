from __future__ import annotations

import json
from _typeshed import Incomplete
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "JsonDumpsParams",
    "DEFAULT_PARAMS",
    "clear_orjson_param_warnings",
    "json_dumps",
    "read_structured_file",
    "safe_write",
    "StructuredFileError",
    "TOMLDecodeError",
    "YAMLError",
]

def clear_orjson_param_warnings() -> None: ...
@dataclass(frozen=True)
class JsonDumpsParams:
    sort_keys: bool = ...
    default: Callable[[Any], Any] | None = ...
    ensure_ascii: bool = ...
    separators: tuple[str, str] = ...
    cls: type[json.JSONEncoder] | None = ...
    to_bytes: bool = ...

DEFAULT_PARAMS: Incomplete

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
) -> bytes | str: ...

class _LazyBool:
    def __init__(self, value: Any) -> None: ...
    def __bool__(self) -> bool: ...

TOMLDecodeError: Incomplete
YAMLError: Incomplete

class StructuredFileError(Exception):
    path: Incomplete
    def __init__(self, path: Path, original: Exception) -> None: ...

def read_structured_file(path: Path) -> Any: ...
def safe_write(
    path: str | Path,
    write: Callable[[Any], Any],
    *,
    mode: str = "w",
    encoding: str | None = "utf-8",
    atomic: bool = True,
    sync: bool | None = None,
    **open_kwargs: Any,
) -> None: ...
