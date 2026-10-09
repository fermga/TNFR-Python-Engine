from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from _typeshed import Incomplete

__all__ = [
    "JsonDumpsParams",
    "DEFAULT_PARAMS",
    "json_dumps",
    "json_loads",
    "read_structured_file",
    "safe_write",
    "StructuredFileError",
    "TOMLDecodeError",
    "YAMLError",
]

@dataclass(frozen=True)
class JsonDumpsParams:
    sort_keys: bool = ...
    default: Callable[[Any], Any] | None = ...
    ensure_ascii: bool = ...
    separators: tuple[str, str] = ...
    cls: type[json.JSONEncoder] | None = ...
    to_bytes: bool = ...

DEFAULT_PARAMS: Incomplete

def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> None: ...
def json_loads(text: str | bytes | bytearray) -> Any: ...
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

def read_structured_file(
    path: Path | str,
    *,
    base_dir: Path | str | None = None,
    allowed_extensions: tuple[str, ...] | None = (".json", ".yaml", ".yml", ".toml"),
) -> Any: ...
def safe_write(
    path: str | Path,
    write: Callable[[Any], Any],
    *,
    mode: str = "w",
    encoding: str | None = "utf-8",
    atomic: bool = True,
    sync: bool | None = None,
    base_dir: str | Path | None = None,
    **open_kwargs: Any,
) -> None: ...
