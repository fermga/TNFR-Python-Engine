"""Configuration utilities."""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from pathlib import Path
from .io import read_structured_file

from .constants import inject_defaults

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    import networkx as nx


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON/YAML file and return a ``dict`` with parameters."""
    data = read_structured_file(Path(path))
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain an object")
    return data


def apply_config(G: nx.Graph, path: str | Path) -> None:
    """Inject parameters from ``path`` into ``G.graph``.

    Reuses :func:`inject_defaults` to keep canonical default semantics.
    """
    path = Path(path)
    cfg = load_config(path)
    inject_defaults(G, cfg, override=True)
