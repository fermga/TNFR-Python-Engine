"""Core configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils import read_structured_file

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    import networkx as nx

__all__ = ("load_config", "apply_config")


def load_config(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> Mapping[str, Any]:
    """Read a JSON/YAML file and return a mapping with parameters.

    Parameters
    ----------
    path : str | Path
        Path to the configuration file.
    base_dir : str | Path | None, optional
        Base directory to restrict config file access. If provided, the
        resolved path must stay within this directory (prevents path traversal).

    Returns
    -------
    Mapping[str, Any]
        Configuration parameters as a mapping.

    Raises
    ------
    ValueError
        If the configuration file is invalid or contains unsafe patterns.
    PathTraversalError
        If path traversal is detected when base_dir is provided.
    StructuredFileError
        If the file cannot be read or parsed.
    """
    path_obj = path if isinstance(path, Path) else Path(path)
    data = read_structured_file(path_obj, base_dir=base_dir)
    if not isinstance(data, Mapping):
        raise ValueError("Configuration file must contain an object")
    return data


def apply_config(
    G: "nx.Graph",
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> None:
    """Inject parameters from ``path`` into ``G.graph``.

    Uses inject_defaults from this module to keep canonical default
    semantics.

    Parameters
    ----------
    G : nx.Graph
        The graph to configure.
    path : str | Path
        Path to the configuration file.
    base_dir : str | Path | None, optional
        Base directory to restrict config file access.
    """
    # Import inject_defaults locally to avoid circular import
    from . import inject_defaults

    cfg = load_config(path, base_dir=base_dir)
    inject_defaults(G, cfg, override=True)
