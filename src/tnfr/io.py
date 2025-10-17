"""Structured file I/O utilities.

Optional parsers such as ``tomllib``/``tomli`` and ``pyyaml`` are loaded via
the :func:`tnfr.utils.cached_import` helper. Their import results and
failure states are cached and can be cleared with
``cached_import.cache_clear()`` and :func:`tnfr.utils.prune_failed_imports`
when needed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable
from functools import partial

from .utils import cached_import, get_logger


def _raise_import_error(name: str, *_: Any, **__: Any) -> Any:
    raise ImportError(f"{name} is not installed")


_MISSING_TOML_ERROR = type(
    "MissingTOMLDependencyError",
    (Exception,),
    {"__doc__": "Fallback error used when tomllib/tomli is missing."},
)

_MISSING_YAML_ERROR = type(
    "MissingPyYAMLDependencyError",
    (Exception,),
    {"__doc__": "Fallback error used when pyyaml is missing."},
)


tomllib = cached_import(
    "tomllib",
    emit="log",
    fallback=cached_import("tomli", emit="log"),
)
has_toml = tomllib is not None


TOMLDecodeError = cached_import(
    "tomllib",
    "TOMLDecodeError",
    emit="log",
    fallback=cached_import(
        "tomli",
        "TOMLDecodeError",
        emit="log",
        fallback=_MISSING_TOML_ERROR,
    ),
)


_TOML_LOADS: Callable[[str], Any] = cached_import(
    "tomllib",
    "loads",
    emit="log",
    fallback=cached_import(
        "tomli",
        "loads",
        emit="log",
        fallback=partial(_raise_import_error, "tomllib/tomli"),
    ),
)


yaml = cached_import("yaml", emit="log")


YAMLError = cached_import(
    "yaml",
    "YAMLError",
    emit="log",
    fallback=_MISSING_YAML_ERROR,
)


_YAML_SAFE_LOAD: Callable[[str], Any] = cached_import(
    "yaml",
    "safe_load",
    emit="log",
    fallback=partial(_raise_import_error, "pyyaml"),
)


def _parse_yaml(text: str) -> Any:
    """Parse YAML ``text`` using ``safe_load`` if available."""
    return _YAML_SAFE_LOAD(text)


def _parse_toml(text: str) -> Any:
    """Parse TOML ``text`` using ``tomllib`` or ``tomli``."""
    return _TOML_LOADS(text)


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
    atomic: bool = True,
    sync: bool | None = None,
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
    atomic:
        When ``True`` (default) writes to a temporary file and atomically
        replaces the destination after flushing to disk. When ``False``
        writes directly to ``path`` without any atomicity guarantee.
    sync:
        When ``True`` flushes and fsyncs the file descriptor after writing.
        ``None`` uses ``atomic`` to determine syncing behaviour.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    open_params = dict(mode=mode, **open_kwargs)
    if "b" not in mode and encoding is not None:
        open_params["encoding"] = encoding
    if sync is None:
        sync = atomic
    tmp_path: Path | None = None
    try:
        if atomic:
            tmp_fd = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
            tmp_path = Path(tmp_fd.name)
            tmp_fd.close()
            with open(tmp_path, **open_params) as fd:
                write(fd)
                if sync:
                    fd.flush()
                    os.fsync(fd.fileno())
            try:
                os.replace(tmp_path, path)
            except OSError as e:
                logger.error(
                    "Atomic replace failed for %s -> %s: %s", tmp_path, path, e
                )
                raise
        else:
            with open(path, **open_params) as fd:
                write(fd)
                if sync:
                    fd.flush()
                    os.fsync(fd.fileno())
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
