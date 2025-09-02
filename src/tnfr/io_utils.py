"""Utilities for reading and writing structured files."""
from __future__ import annotations

from typing import Any, Callable, Dict
import json
from json import JSONDecodeError
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
    from yaml import YAMLError  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

    class YAMLError(Exception):  # type: ignore
        pass


__all__ = ["read_structured_file", "ensure_parent"]


def _parse_json(text: str) -> Any:
    """Parse ``text`` as JSON."""
    return json.loads(text)


def _parse_yaml(text: str) -> Any:
    """Parse ``text`` as YAML."""
    if not yaml:  # pragma: no cover - optional dependency
        raise RuntimeError("pyyaml no está instalado")
    return yaml.safe_load(text)


PARSERS: Dict[str, Callable[[str], Any]] = {
    ".json": _parse_json,
    ".yaml": _parse_yaml,
    ".yml": _parse_yaml,
}


def read_structured_file(path: Path) -> Any:
    """Read a JSON or YAML file and return the parsed data."""
    suffix = path.suffix.lower()
    if suffix not in PARSERS:
        raise ValueError(f"Extensión de archivo no soportada: {suffix}")
    parser = PARSERS[suffix]
    if not path.is_file():
        raise ValueError(f"El archivo no existe: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except PermissionError as e:
        raise ValueError(f"Permiso denegado al leer {path}: {e}") from e
    except FileNotFoundError as e:  # pragma: no cover - unlikely race
        raise ValueError(f"El archivo no existe: {path}") from e

    try:
        return parser(text)
    except JSONDecodeError as e:
        raise ValueError(f"Error al parsear archivo JSON en {path}: {e}") from e
    except YAMLError as e:
        raise ValueError(f"Error al parsear archivo YAML en {path}: {e}") from e
    except RuntimeError as e:
        raise ValueError(f"Dependencia faltante al parsear {path}: {e}") from e


def ensure_parent(path: str | Path) -> None:
    """Create the parent directory for ``path`` if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

