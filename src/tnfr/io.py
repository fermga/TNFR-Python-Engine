"""Structured file I/O utilities."""

from __future__ import annotations
from typing import Any, Callable
import json
from pathlib import Path
from functools import lru_cache

from .import_utils import optional_import
import tempfile


tomllib = optional_import("tomllib") or optional_import("tomli")
if tomllib is not None:
    TOMLDecodeError = getattr(tomllib, "TOMLDecodeError", Exception)
    has_toml = True
else:  # pragma: no cover - depende de tomllib/tomli
    has_toml = False

    class TOMLDecodeError(Exception):
        pass

yaml = optional_import("yaml")
if yaml is not None:
    YAMLError = getattr(yaml, "YAMLError", Exception)
else:  # pragma: no cover - depende de pyyaml

    class YAMLError(Exception):
        pass


def _missing_dependency(name: str) -> Callable[[str], Any]:
    def _raise(_: str) -> Any:
        raise ImportError(f"{name} no está instalado")

    return _raise

PARSERS = {
    ".json": json.loads,
    ".yaml": lambda text: getattr(
        yaml, "safe_load", _missing_dependency("pyyaml")
    )(text),
    ".yml": lambda text: getattr(
        yaml, "safe_load", _missing_dependency("pyyaml")
    )(text),
    ".toml": lambda text: getattr(
        tomllib, "loads", _missing_dependency("tomllib/tomli")
    )(text),
}


@lru_cache(maxsize=None)
def _get_parser(suffix: str) -> Callable[[str], Any]:
    try:
        return PARSERS[suffix]
    except KeyError:
        raise ValueError(f"Unsupported suffix: {suffix}")


def _format_structured_file_error(path: Path, e: Exception) -> str:
    if isinstance(e, OSError):
        return f"No se pudo leer {path}: {e}"
    if isinstance(e, UnicodeDecodeError):
        return f"Error de codificación al leer {path}: {e}"
    if isinstance(e, json.JSONDecodeError):
        return f"Error al parsear archivo JSON en {path}: {e}"
    if isinstance(e, YAMLError):
        return f"Error al parsear archivo YAML en {path}: {e}"
    if has_toml and isinstance(e, TOMLDecodeError):
        return f"Error al parsear archivo TOML en {path}: {e}"
    if isinstance(e, ImportError):
        return f"Dependencia faltante al parsear {path}: {e}"
    return f"Error al parsear {path}: {e}"


class StructuredFileError(Exception):
    """Error while reading or parsing a structured file."""

    def __init__(self, path: Path, original: Exception):
        super().__init__(_format_structured_file_error(path, original))
        self.path = path


def read_structured_file(path: Path) -> Any:
    """Read a JSON, YAML or TOML file and return parsed data."""
    suffix = path.suffix.lower()
    parser = _get_parser(suffix)
    try:
        text = path.read_text(encoding="utf-8")
        return parser(text)
    except Exception as e:
        raise StructuredFileError(path, e) from e


def safe_write(
    path: str | Path,
    write: Callable[[Any], Any],
    *,
    mode: str = "w",
    encoding: str | None = "utf-8",
    **open_kwargs: Any,
) -> None:
    """Write to ``path`` ensuring parent directory exists and handle errors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    open_params = dict(mode=mode, **open_kwargs)
    if encoding is not None:
        open_params["encoding"] = encoding
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        with open(tmp_path, **open_params) as f:
            write(f)
        tmp_path.replace(path)
    except OSError as e:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        raise OSError(f"Failed to write file {path}: {e}") from e


__all__ = ["read_structured_file", "safe_write", "StructuredFileError"]

