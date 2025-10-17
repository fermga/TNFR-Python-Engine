"""Minimal public API for :mod:`tnfr`.

This package only re-exports a handful of high level helpers.  Most
functionality lives in submodules that should be imported directly, for
example :mod:`tnfr.metrics`, :mod:`tnfr.observers` or the DSL utilities
in :mod:`tnfr.tokens`, :mod:`tnfr.flatten` and :mod:`tnfr.execution`.
Recommended entry points are:

- ``step`` and ``run`` in :mod:`tnfr.dynamics`
- ``preparar_red`` in :mod:`tnfr.ontosim`
- ``create_nfr`` and ``run_sequence`` in :mod:`tnfr.structural`
- ``cached_import`` and ``prune_failed_imports`` in :mod:`tnfr.utils` for
  optional dependencies
"""

from __future__ import annotations

from importlib import import_module

from .ontosim import preparar_red


def _missing_dependency(name: str, exc: ImportError):
    def _stub(*args, **kwargs):
        raise ImportError(
            f"{name} is unavailable because required dependencies could not be imported. "
            f"Original error ({exc.__class__.__name__}): {exc}. "
            "Install the missing packages (e.g. 'networkx' or grammar modules)."
        ) from exc

    return _stub


try:  # pragma: no cover - exercised in import tests
    from .dynamics import step, run
except ImportError as exc:  # pragma: no cover - no missing deps in CI
    step = _missing_dependency("step", exc)
    run = _missing_dependency("run", exc)


_HAS_RUN_SEQUENCE = False
try:  # pragma: no cover - exercised in import tests
    from .structural import create_nfr, run_sequence
except ImportError as exc:  # pragma: no cover - no missing deps in CI
    create_nfr = _missing_dependency("create_nfr", exc)
    run_sequence = _missing_dependency("run_sequence", exc)
else:
    _HAS_RUN_SEQUENCE = True


try:  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore[import-not-found]


try:
    __version__ = version("tnfr")
except PackageNotFoundError:  # pragma: no cover
    from pathlib import Path

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    __version__ = "0+unknown"
    for module_name in ("tomllib", "tomli"):
        try:
            parser = import_module(module_name)
        except ModuleNotFoundError:
            continue

        load = getattr(parser, "load", None)
        if load is None:
            continue
        try:
            with pyproject_path.open("rb") as f:
                data = load(f)
        except (OSError, AttributeError, TypeError, ValueError):
            continue
        try:
            project = data["project"]
            project_version = project["version"]
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(project_version, str):
            __version__ = project_version
            break

__all__ = [
    "__version__",
    "step",
    "run",
    "preparar_red",
    "create_nfr",
]

if _HAS_RUN_SEQUENCE:
    __all__.append("run_sequence")
