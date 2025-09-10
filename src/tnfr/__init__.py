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

from .dynamics import step, run
from .ontosim import preparar_red
from .structural import create_nfr
from .types import NodeState
from .operators import apply_topological_remesh

# re-exported for tests
from .trace import CallbackSpec  # noqa: F401
from .import_utils import optional_import

_metadata = optional_import("importlib.metadata")
if _metadata is None:  # pragma: no cover
    _metadata = optional_import("importlib_metadata")

if _metadata is not None:  # pragma: no cover
    version = _metadata.version  # type: ignore[attr-defined]
    PackageNotFoundError = _metadata.PackageNotFoundError  # type: ignore[attr-defined]
else:  # pragma: no cover
    class PackageNotFoundError(Exception):
        pass

    def version(_: str) -> str:
        raise PackageNotFoundError

try:
    __version__ = version("tnfr")
except PackageNotFoundError:  # pragma: no cover
    tomllib = optional_import("tomllib")
    if tomllib is not None:
        from pathlib import Path

        try:
            with (Path(__file__).resolve().parents[2] / "pyproject.toml").open(
                "rb",
            ) as f:
                __version__ = tomllib.load(f)["project"]["version"]
        except (OSError, KeyError, ValueError):  # pragma: no cover
            __version__ = "0+unknown"
    else:  # pragma: no cover
        __version__ = "0+unknown"

__all__ = [
    "__version__",
    "step",
    "run",
    "preparar_red",
    "create_nfr",
    "NodeState",
    "apply_topological_remesh",
    "CallbackSpec",
]
