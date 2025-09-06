"""Minimal public API for :mod:`tnfr`.

This package only re-exports a handful of high level helpers.  Most
functionality lives in submodules that should be imported directly, for
example :mod:`tnfr.metrics`, :mod:`tnfr.observers` or
:mod:`tnfr.program`.  Recommended entry points are:

- ``step`` and ``run`` in :mod:`tnfr.dynamics`
- ``preparar_red`` in :mod:`tnfr.ontosim`
- ``create_nfr`` in :mod:`tnfr.structural`
- ``NodeState`` in :mod:`tnfr.types`
"""

from __future__ import annotations

try:  # pragma: no cover
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("tnfr")
except PackageNotFoundError:  # pragma: no cover
    try:
        import tomllib
        from pathlib import Path

        with (Path(__file__).resolve().parents[2] / "pyproject.toml").open("rb") as f:
            __version__ = tomllib.load(f)["project"]["version"]
    except (OSError, KeyError, ValueError):  # pragma: no cover
        __version__ = "0+unknown"

# Minimal public API re-exports
from .dynamics import step, run
from .ontosim import preparar_red
from .structural import create_nfr
from .types import NodeState
from .trace import CallbackSpec  # re-exported for tests

__all__ = [
    "__version__",
    "step",
    "run",
    "preparar_red",
    "create_nfr",
    "NodeState",
]

