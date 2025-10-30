"""Backward compatible wrapper for structured IO helpers."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from .utils import io as _io

JsonDumpsParams = _io.JsonDumpsParams
DEFAULT_PARAMS = _io.DEFAULT_PARAMS
clear_orjson_param_warnings = _io.clear_orjson_param_warnings
json_dumps = _io.json_dumps
read_structured_file = _io.read_structured_file
safe_write = _io.safe_write
StructuredFileError = _io.StructuredFileError
TOMLDecodeError = _io.TOMLDecodeError
YAMLError = _io.YAMLError


def __getattr__(name: str) -> Any:  # pragma: no cover - thin delegation
    return getattr(_io, name)


def __dir__() -> list[str]:  # pragma: no cover - thin delegation
    return sorted(set(globals()) | set(dir(_io)))


def _get_underlying_module() -> ModuleType:  # pragma: no cover - convenience
    """Return the underlying implementation module for advanced uses."""

    return _io


__all__ = (
    "JsonDumpsParams",
    "DEFAULT_PARAMS",
    "clear_orjson_param_warnings",
    "json_dumps",
    "read_structured_file",
    "safe_write",
    "StructuredFileError",
    "TOMLDecodeError",
    "YAMLError",
)

