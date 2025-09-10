"""Structured file I/O utilities."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable
from functools import lru_cache

from .import_utils import optional_import
from .logging_utils import get_logger


@lru_cache(maxsize=None)
def _missing_dependency_error(dep: str) -> type[Exception]:
    """Return a fallback :class:`Exception` when ``dep`` is unavailable."""

    class _MissingDependencyError(Exception):
        pass

    _MissingDependencyError.__doc__ = f"Fallback error used when {dep} is missing."
    return _MissingDependencyError


tomllib = optional_import("tomllib") or optional_import("tomli")
if tomllib is not None:
    TOMLDecodeError = getattr(tomllib, "TOMLDecodeError", Exception)
    has_toml = True
else:  # pragma: no cover - depende de tomllib/tomli
    has_toml = False
    TOMLDecodeError = _missing_dependency_error("tomllib/tomli")


yaml = optional_import("yaml")
if yaml is not None:
    YAMLError = getattr(yaml, "YAMLError", Exception)
else:  # pragma: no cover - depende de pyyaml
    YAMLError = _missing_dependency_error("pyyaml")


def _missing_dependency(name: str) -> Callable[[str], Any]:
    def _raise(_: str) -> Any:
        raise ImportError(f"{name} is not installed")

    return _raise


def _parse_yaml(text: str) -> Any:
    """Parse YAML ``text`` using ``safe_load`` if available."""
    return getattr(yaml, "safe_load", _missing_dependency("pyyaml"))(text)


def _parse_toml(text: str) -> Any:
    """Parse TOML ``text`` using ``tomllib`` or ``tomli``."""
    return getattr(tomllib, "loads", _missing_dependency("tomllib/tomli"))(text)


PARSERS = {
    ".json": json.loads,
    ".yaml": _parse_yaml,
    ".yml": _parse_yaml,
    ".toml": _parse_toml,
}


def _get_parser(suffix: str) -> Callable[[str], Any]:
    try:
        return PARSERS[suffix]
    except KeyError as exc:
        raise ValueError(f"Unsupported suffix: {suffix}") from exc


ERROR_MESSAGES = {
    OSError: "Could not read {path}: {e}",
    UnicodeDecodeError: "Encoding error while reading {path}: {e}",
    json.JSONDecodeError: "Error parsing JSON file at {path}: {e}",
    YAMLError: "Error parsing YAML file at {path}: {e}",
    ImportError: "Missing dependency parsing {path}: {e}",
}
if has_toml:
    ERROR_MESSAGES[TOMLDecodeError] = "Error parsing TOML file at {path}: {e}"


def _format_structured_file_error(path: Path, e: Exception) -> str:
    for exc, msg in ERROR_MESSAGES.items():
        if isinstance(e, exc):
            return msg.format(path=path, e=e)
    return f"Error parsing {path}: {e}"


class StructuredFileError(Exception):
    """Error while reading or parsing a structured file."""

    def __init__(self, path: Path, original: Exception):
        super().__init__(_format_structured_file_error(path, original))
        self.path = path


def read_structured_file(path: Path) -> Any:
    """Read a JSON, YAML or TOML file and return parsed data."""
    suffix = path.suffix.lower()
    try:
        parser = _get_parser(suffix)
    except ValueError as e:
        raise StructuredFileError(path, e) from e
    try:
        text = path.read_text(encoding="utf-8")
        return parser(text)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        YAMLError,
        TOMLDecodeError,
        ImportError,
    ) as e:
        raise StructuredFileError(path, e) from e


logger = get_logger(__name__)


def safe_write(
    path: str | Path,
    write: Callable[[Any], Any],
    *,
    mode: str = "w",
    encoding: str | None = "utf-8",
    **open_kwargs: Any,
) -> None:
    """Write to ``path`` ensuring parent directory exists and handle errors.

    Parameters
    ----------
    path:
        Destination file path.
    write:
        Callback receiving the opened file object and performing the actual
        write.
    mode:
        File mode passed to :func:`open`. Text modes (default) use UTF-8
        encoding unless ``encoding`` is ``None``. When a binary mode is used
        (``'b'`` in ``mode``) no encoding parameter is supplied so
        ``write`` may write bytes.
    encoding:
        Encoding for text modes. Ignored for binary modes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    open_params = dict(mode=mode, **open_kwargs)
    if "b" not in mode and encoding is not None:
        open_params["encoding"] = encoding
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, delete=False, **open_params
        ) as tmp:
            tmp_path = Path(tmp.name)
            write(tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
        try:
            os.replace(tmp_path, path)
        except OSError as e:
            logger.error(
                "Atomic replace failed for %s -> %s: %s", tmp_path, path, e
            )
            raise
    except (OSError, ValueError, TypeError) as e:
        raise type(e)(f"Failed to write file {path}: {e}") from e
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


__all__ = (
    "read_structured_file",
    "safe_write",
    "StructuredFileError",
    "TOMLDecodeError",
    "YAMLError",
)
