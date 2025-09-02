"""Configuration utilities."""
from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
from .helpers import read_structured_file

from .constants import inject_defaults


def load_config(path: Path) -> Dict[str, Any]:
    """Read a JSON/YAML file and return a ``dict`` with parameters."""
    data = read_structured_file(path)
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain an object")
    return data


def apply_config(G, path: Path) -> None:
    """Inject parameters from ``path`` into ``G.graph``.

    Reuses :func:`inject_defaults` to keep canonical default semantics.
    """
    cfg = load_config(path)
    inject_defaults(G, cfg, override=True)
