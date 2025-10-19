"""Core configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..constants import inject_defaults
from ..io import read_structured_file

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    import networkx as nx

__all__ = ("load_config", "apply_config")


def load_config(path: str | Path) -> Mapping[str, Any]:
    """Read a JSON/YAML file and return a mapping with parameters."""

    path_obj = path if isinstance(path, Path) else Path(path)
    data = read_structured_file(path_obj)
    if not isinstance(data, Mapping):
        raise ValueError("Configuration file must contain an object")
    return data


def apply_config(G: "nx.Graph", path: str | Path) -> None:
    """Inject parameters from ``path`` into ``G.graph``.

    Reuses :func:`tnfr.constants.inject_defaults` to keep canonical default
    semantics.
    """

    cfg = load_config(path)
    inject_defaults(G, cfg, override=True)
