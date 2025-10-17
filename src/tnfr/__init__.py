"""Minimal public API for :mod:`tnfr`.

This package only re-exports a handful of high level helpers.  Most
functionality lives in submodules that should be imported directly, for
example :mod:`tnfr.metrics`, :mod:`tnfr.observers` or the DSL utilities
in :mod:`tnfr.tokens`, :mod:`tnfr.flatten` and :mod:`tnfr.execution`.

Exported helpers and their dependencies
---------------------------------------
The :data:`EXPORT_DEPENDENCIES` mapping enumerates which internal
submodules and third-party packages are required to load each helper.
The imports are grouped as follows:

``step`` / ``run``
    Provided by :mod:`tnfr.dynamics`.  These helpers rely exclusively on
    internal TNFR modules (``tnfr.operators``, ``tnfr.metrics`` and
    friends) and do not require additional third-party packages.

``preparar_red``
    Defined in :mod:`tnfr.ontosim`.  It configures graphs generated via
    :mod:`networkx`, but the helper itself only depends on TNFR modules
    such as :mod:`tnfr.constants`, :mod:`tnfr.dynamics` and
    :mod:`tnfr.utils`.

``create_nfr`` / ``run_sequence``
    Re-exported from :mod:`tnfr.structural`.  Both helpers require the
    ``networkx`` package in addition to TNFR structural utilities
    (``tnfr.structural``, ``tnfr.validation`` and operator registries).

``cached_import`` and ``prune_failed_imports`` remain available from
``tnfr.utils`` for optional dependency management.
"""

from __future__ import annotations

import warnings
from importlib import import_module, metadata
from importlib.metadata import PackageNotFoundError
from typing import Any

from .ontosim import preparar_red


EXPORT_DEPENDENCIES: dict[str, dict[str, tuple[str, ...]]] = {
    "step": {
        "submodules": ("tnfr.dynamics",),
        "third_party": (),
    },
    "run": {
        "submodules": ("tnfr.dynamics",),
        "third_party": (),
    },
    "preparar_red": {
        "submodules": ("tnfr.ontosim",),
        "third_party": (),
    },
    "create_nfr": {
        "submodules": ("tnfr.structural",),
        "third_party": ("networkx",),
    },
    "run_sequence": {
        "submodules": ("tnfr.structural",),
        "third_party": ("networkx",),
    },
}


try:  # pragma: no cover - exercised in version resolution tests
    __version__ = metadata.version("tnfr")
except PackageNotFoundError:  # pragma: no cover - fallback tested explicitly
    from ._version import __version__ as _fallback_version

    __version__ = _fallback_version


def _is_internal_import_error(exc: ImportError) -> bool:
    missing_name = getattr(exc, "name", None) or ""
    if missing_name.startswith("tnfr"):
        return True
    missing_path = getattr(exc, "path", None) or ""
    if missing_path:
        normalized = missing_path.replace("\\", "/")
        if "/tnfr/" in normalized or normalized.endswith("/tnfr"):
            return True
    return False


def _missing_dependency(name: str, exc: ImportError, *, module: str | None = None):
    missing_name = getattr(exc, "name", None)

    def _stub(*args: Any, **kwargs: Any):
        raise ImportError(
            f"{name} is unavailable because required dependencies could not be imported. "
            f"Original error ({exc.__class__.__name__}): {exc}. "
            "Install the missing packages (e.g. 'networkx' or grammar modules)."
        ) from exc

    _stub.__tnfr_missing_dependency__ = {
        "export": name,
        "module": module,
        "missing": missing_name,
    }
    return _stub


_MISSING_EXPORTS: dict[str, dict[str, Any]] = {}


def _assign_exports(module: str, names: tuple[str, ...]) -> bool:
    try:  # pragma: no cover - exercised in import tests
        mod = import_module(f".{module}", __name__)
    except ImportError as exc:  # pragma: no cover - no missing deps in CI
        if _is_internal_import_error(exc):
            raise
        for export_name in names:
            stub = _missing_dependency(export_name, exc, module=module)
            globals()[export_name] = stub
            _MISSING_EXPORTS[export_name] = getattr(
                stub, "__tnfr_missing_dependency__", {}
            )
        return False
    else:
        for export_name in names:
            globals()[export_name] = getattr(mod, export_name)
        return True


_assign_exports("dynamics", ("step", "run"))


_HAS_RUN_SEQUENCE = _assign_exports("structural", ("create_nfr", "run_sequence"))


def _emit_missing_dependency_warning() -> None:
    if not _MISSING_EXPORTS:
        return
    details = ", ".join(
        f"{name} (missing: {info.get('missing') or 'unknown'})"
        for name, info in sorted(_MISSING_EXPORTS.items())
    )
    warnings.warn(
        "TNFR helpers disabled because dependencies are missing: " + details,
        ImportWarning,
        stacklevel=2,
    )


_emit_missing_dependency_warning()


__all__ = [
    "__version__",
    "step",
    "run",
    "preparar_red",
    "create_nfr",
]

if _HAS_RUN_SEQUENCE:
    __all__.append("run_sequence")
