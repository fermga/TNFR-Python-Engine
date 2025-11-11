"""Structured file and JSON utilities shared across the TNFR engine."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

from .init import LazyImportProxy, cached_import, get_logger, warn_once

logger = get_logger(__name__)

_ORJSON_PARAMS_MSG = (
    "'ensure_ascii', 'separators', 'cls' and extra kwargs are ignored when using orjson: %s"
)

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


def _resolve_lazy(value: Any) -> Any:
    if isinstance(value, LazyImportProxy):
        return value.resolve()
    return value


class _LazyBool:
    __slots__ = ("_value",)

    def __init__(self, value: Any) -> None:
        self._value = value

    def __bool__(self) -> bool:
        return _resolve_lazy(self._value) is not None


_TOMLI_MODULE = cached_import("tomli", emit="log", lazy=True)
tomllib = cached_import(
    "tomllib",
    emit="log",
    lazy=True,
    fallback=_TOMLI_MODULE,
)
has_toml = _LazyBool(tomllib)

_TOMLI_TOML_ERROR = cached_import(
    "tomli",
    "TOMLDecodeError",
    emit="log",
    lazy=True,
    fallback=_MISSING_TOML_ERROR,
)
TOMLDecodeError = cached_import(
    "tomllib",
    "TOMLDecodeError",
    emit="log",
    lazy=True,
    fallback=_TOMLI_TOML_ERROR,
)

_TOMLI_LOADS = cached_import(
    "tomli",
    "loads",
    emit="log",
    lazy=True,
    fallback=partial(_raise_import_error, "tomllib/tomli"),
)
_TOML_LOADS: Callable[[str], Any] = cached_import(
    "tomllib",
    "loads",
    emit="log",
    lazy=True,
    fallback=_TOMLI_LOADS,
)

yaml = cached_import("yaml", emit="log", lazy=True)

YAMLError = cached_import(
    "yaml",
    "YAMLError",
    emit="log",
    lazy=True,
    fallback=_MISSING_YAML_ERROR,
)

_YAML_SAFE_LOAD: Callable[[str], Any] = cached_import(
    "yaml",
    "safe_load",
    emit="log",
    lazy=True,
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


_BASE_ERROR_MESSAGES: dict[type[BaseException], str] = {
    OSError: "Could not read {path}: {e}",
    UnicodeDecodeError: "Encoding error while reading {path}: {e}",
    json.JSONDecodeError: "Error parsing JSON file at {path}: {e}",
    ImportError: "Missing dependency parsing {path}: {e}",
}


def _resolve_exception_type(candidate: Any) -> type[BaseException] | None:
    resolved = _resolve_lazy(candidate)
    if isinstance(resolved, type) and issubclass(resolved, BaseException):
        return resolved
    return None


_OPTIONAL_ERROR_MESSAGE_FACTORIES: tuple[
    tuple[Callable[[], type[BaseException] | None], str],
    ...,
] = (
    (
        lambda: _resolve_exception_type(YAMLError),
        "Error parsing YAML file at {path}: {e}",
    ),
    (
        lambda: _resolve_exception_type(TOMLDecodeError),
        "Error parsing TOML file at {path}: {e}",
    ),
)

_BASE_STRUCTURED_EXCEPTIONS = (
    OSError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    ImportError,
)


def _iter_optional_exceptions() -> list[type[BaseException]]:
    errors: list[type[BaseException]] = []
    for resolver, _ in _OPTIONAL_ERROR_MESSAGE_FACTORIES:
        exc_type = resolver()
        if exc_type is not None:
            errors.append(exc_type)
    return errors


def _is_structured_error(exc: Exception) -> bool:
    if isinstance(exc, _BASE_STRUCTURED_EXCEPTIONS):
        return True
    for optional_exc in _iter_optional_exceptions():
        if isinstance(exc, optional_exc):
            return True
    return False


def _format_structured_file_error(path: Path, e: Exception) -> str:
    for exc, msg in _BASE_ERROR_MESSAGES.items():
        if isinstance(e, exc):
            return msg.format(path=path, e=e)

    for resolver, msg in _OPTIONAL_ERROR_MESSAGE_FACTORIES:
        exc_type = resolver()
        if exc_type is not None and isinstance(e, exc_type):
            return msg.format(path=path, e=e)

    return f"Error parsing {path}: {e}"


class StructuredFileError(Exception):
    """Error while reading or parsing a structured file."""

    def __init__(self, path: Path, original: Exception) -> None:
        super().__init__(_format_structured_file_error(path, original))
        self.path = path


def read_structured_file(
    path: Path | str,
    *,
    base_dir: Path | str | None = None,
    allowed_extensions: tuple[str, ...] | None = (".json", ".yaml", ".yml", ".toml"),
) -> Any:
    """Read a JSON, YAML or TOML file and return parsed data.

    This function includes path traversal protection. When ``base_dir`` is
    provided, the resolved path must stay within that directory.

    Parameters
    ----------
    path : Path | str
        Path to the structured file to read.
    base_dir : Path | str | None, optional
        Base directory to restrict file access. If provided, the resolved
        path must stay within this directory (prevents path traversal).
    allowed_extensions : tuple[str, ...] | None, optional
        Tuple of allowed file extensions. Default is JSON, YAML, and TOML.
        Pass None to allow any extension (not recommended for user input).

    Returns
    -------
    Any
        Parsed data from the file.

    Raises
    ------
    StructuredFileError
        If the file cannot be read or parsed.
    ValueError
        If the path is invalid or contains unsafe patterns.
    PathTraversalError
        If path traversal is detected.

    Examples
    --------
    >>> from pathlib import Path
    >>> # Read config file with path validation
    >>> data = read_structured_file("config.json")  # doctest: +SKIP

    >>> # Read with base directory restriction
    >>> data = read_structured_file(
    ...     "settings.yaml",
    ...     base_dir="/home/user/configs"
    ... )  # doctest: +SKIP
    """
    # Import here to avoid circular dependency
    from ..security import resolve_safe_path, validate_file_path, PathTraversalError

    # Validate and resolve the path
    try:
        if base_dir is not None:
            # Resolve path within base directory (prevents traversal)
            validated_path = resolve_safe_path(
                path,
                base_dir,
                must_exist=True,
                allowed_extensions=allowed_extensions,
            )
        else:
            # Validate path without base directory restriction
            path_obj = Path(path) if not isinstance(path, Path) else path
            validated_path = validate_file_path(
                path_obj,
                allow_absolute=True,
                allowed_extensions=allowed_extensions,
            ).resolve()

            # Check existence
            if not validated_path.exists():
                raise FileNotFoundError(f"File not found: {validated_path}")
    except (ValueError, PathTraversalError) as e:
        raise StructuredFileError(Path(path), e) from e
    except FileNotFoundError as e:
        raise StructuredFileError(Path(path), e) from e

    suffix = validated_path.suffix.lower()
    try:
        parser = _get_parser(suffix)
    except ValueError as e:
        raise StructuredFileError(validated_path, e) from e
    try:
        text = validated_path.read_text(encoding="utf-8")
        return parser(text)
    except Exception as e:
        if _is_structured_error(e):
            raise StructuredFileError(validated_path, e) from e
        raise


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
) -> None:
    """Write to ``path`` ensuring parent directory exists and handle errors.

    This function includes path traversal protection. When ``base_dir`` is
    provided, the resolved path must stay within that directory.

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
    base_dir:
        Optional base directory to restrict file writes. If provided, the
        resolved path must stay within this directory (prevents path traversal).

    Raises
    ------
    ValueError
        If the path is invalid or contains unsafe patterns.
    PathTraversalError
        If path traversal is detected when base_dir is provided.
    """
    # Import here to avoid circular dependency
    from ..security import resolve_safe_path, validate_file_path, PathTraversalError

    # Validate and resolve the path
    try:
        if base_dir is not None:
            # Resolve path within base directory (prevents traversal)
            validated_path = resolve_safe_path(
                path,
                base_dir,
                must_exist=False,
            )
        else:
            # Validate path without base directory restriction
            path_obj = Path(path) if not isinstance(path, Path) else path
            validated_path = validate_file_path(
                path_obj,
                allow_absolute=True,
            ).resolve()
    except (ValueError, PathTraversalError) as e:
        raise type(e)(f"Invalid path {path!r}: {e}") from e

    path = validated_path
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
                logger.error("Atomic replace failed for %s -> %s: %s", tmp_path, path, e)
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
