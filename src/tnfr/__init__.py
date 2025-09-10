"""Minimal public API for :mod:`tnfr`.

This package only re-exports a handful of high level helpers.  Most
functionality lives in submodules that should be imported directly, for
example :mod:`tnfr.metrics`, :mod:`tnfr.observers` or
:mod:`tnfr.program`.  Recommended entry points are:

- ``step`` and ``run`` in :mod:`tnfr.dynamics`
- ``preparar_red`` in :mod:`tnfr.ontosim`
- ``create_nfr`` and ``run_sequence`` in :mod:`tnfr.structural`
- ``NodeState`` in :mod:`tnfr.types`
"""

from __future__ import annotations

from .ontosim import preparar_red
from .types import NodeState


def _missing_dependency(name: str, exc: Exception):
    def _stub(*args, **kwargs):
        raise ImportError(
            f"{name} is unavailable because dependencies are missing. "
            f"Original error: {exc}. Install required packages such as "
            "'networkx' or grammar modules."
        ) from exc

    return _stub


try:  # pragma: no cover - exercised in import tests
    from .dynamics import step, run
except Exception as exc:  # pragma: no cover - no missing deps in CI
    step = _missing_dependency("step", exc)
    run = _missing_dependency("run", exc)


_HAS_RUN_SEQUENCE = False
try:  # pragma: no cover - exercised in import tests
    from .structural import create_nfr, run_sequence
except Exception as exc:  # pragma: no cover - no missing deps in CI
    create_nfr = _missing_dependency("create_nfr", exc)
    run_sequence = _missing_dependency("run_sequence", exc)
else:
    _HAS_RUN_SEQUENCE = True


_HAS_APPLY_TOPOLOGICAL_REMESH = False
try:  # pragma: no cover - exercised in import tests
    from .operators import apply_topological_remesh
except Exception as exc:  # pragma: no cover - no missing deps in CI
    apply_topological_remesh = _missing_dependency("apply_topological_remesh", exc)
else:
    _HAS_APPLY_TOPOLOGICAL_REMESH = True

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
    "CallbackSpec",
]

if _HAS_RUN_SEQUENCE:
    __all__.append("run_sequence")
if _HAS_APPLY_TOPOLOGICAL_REMESH:
    __all__.append("apply_topological_remesh")
