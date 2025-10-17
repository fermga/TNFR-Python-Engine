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

from pathlib import Path
from typing import Any

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


try:  # pragma: no cover - Python 3.11+
    import tomllib as _tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    try:
        import tomli as _tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
        _tomllib = None  # type: ignore[assignment]

if _tomllib is not None:  # pragma: no cover - trivial branch
    _TOML_DECODE_ERRORS = (getattr(_tomllib, "TOMLDecodeError", ValueError),)
else:  # pragma: no cover - optional dependency missing
    _TOML_DECODE_ERRORS = (ValueError,)


def _read_pyproject_version() -> str | None:
    """Return the project version declared in :file:`pyproject.toml`.

    The file is parsed using :mod:`tomllib` (or :mod:`tomli` as a fallback for
    Python versions prior to 3.11).  ``None`` is returned when the file cannot
    be read or parsed, or when the version field is absent.
    """

    if _tomllib is None:
        return None

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as stream:
            data: dict[str, Any] = _tomllib.load(stream)
    except OSError:
        return None
    except _TOML_DECODE_ERRORS:  # type: ignore[misc]
        return None

    project_data = data.get("project")
    if not isinstance(project_data, dict):
        return None

    version = project_data.get("version")
    if isinstance(version, str):
        return version
    return None


try:
    __version__ = version("tnfr")
except PackageNotFoundError:  # pragma: no cover
    _fallback_version = _read_pyproject_version()
    __version__ = _fallback_version if _fallback_version is not None else "0+unknown"

__all__ = [
    "__version__",
    "step",
    "run",
    "preparar_red",
    "create_nfr",
]

if _HAS_RUN_SEQUENCE:
    __all__.append("run_sequence")
