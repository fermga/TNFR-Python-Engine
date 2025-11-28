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
    Provided by :mod:`tnfr.dynamics`.  These helpers rely on the
    machinery defined within the :mod:`tnfr.dynamics` package (operator
    orchestration, validation hooks and metrics integration) and require
    the ``networkx`` package for graph handling.

``prepare_network``
    Defined in :mod:`tnfr.ontosim`.  Besides :mod:`tnfr.ontosim`
    itself, the helper imports :mod:`tnfr.callback_utils`,
    :mod:`tnfr.constants`, :mod:`tnfr.dynamics`, :mod:`tnfr.glyph_history`,
    :mod:`tnfr.initialization` and :mod:`tnfr.utils` to assemble the
    graph preparation pipeline.  It also requires ``networkx`` at import
    time.

``create_nfr`` / ``run_sequence``
    Re-exported from :mod:`tnfr.structural`.  They depend on
    :mod:`tnfr.structural`, :mod:`tnfr.constants`, :mod:`tnfr.dynamics`,
    :mod:`tnfr.operators.definitions`, :mod:`tnfr.operators.registry` and
    :mod:`tnfr.validation`, and additionally require the ``networkx``
    package.

``cached_import`` and ``prune_failed_imports`` remain available from
``tnfr.utils`` for optional dependency management.
"""

from __future__ import annotations

import warnings
from importlib import import_module, metadata
from importlib.metadata import PackageNotFoundError
from typing import Any, Callable, NoReturn

EXPORT_DEPENDENCIES: dict[str, dict[str, tuple[str, ...]]] = {
    "step": {
        "submodules": ("tnfr.dynamics",),
        "third_party": ("networkx",),
    },
    "run": {
        "submodules": ("tnfr.dynamics",),
        "third_party": ("networkx",),
    },
    "prepare_network": {
        "submodules": (
            "tnfr.ontosim",
            "tnfr.callback_utils",
            "tnfr.constants",
            "tnfr.dynamics",
            "tnfr.glyph_history",
            "tnfr.initialization",
            "tnfr.utils",
        ),
        "third_party": ("networkx",),
    },
    "create_nfr": {
        "submodules": (
            "tnfr.structural",
            "tnfr.constants",
            "tnfr.dynamics",
            "tnfr.operators.definitions",
            "tnfr.operators.registry",
            "tnfr.validation",
        ),
        "third_party": ("networkx",),
    },
    "create_math_nfr": {
        "submodules": (
            "tnfr.structural",
            "tnfr.constants",
            "tnfr.dynamics",
            "tnfr.operators.definitions",
            "tnfr.operators.registry",
            "tnfr.validation",
            "tnfr.mathematics",
        ),
        "third_party": (
            "networkx",
            "numpy",
        ),
    },
    "run_sequence": {
        "submodules": (
            "tnfr.structural",
            "tnfr.constants",
            "tnfr.dynamics",
            "tnfr.operators.definitions",
            "tnfr.operators.registry",
            "tnfr.validation",
        ),
        "third_party": ("networkx",),
    },
    "get_hz_bridge": {
        "submodules": (
            "tnfr.units",
            "tnfr.constants",
        ),
        "third_party": ("networkx",),
    },
    "hz_str_to_hz": {
        "submodules": (
            "tnfr.units",
            "tnfr.constants",
        ),
        "third_party": ("networkx",),
    },
    "hz_to_hz_str": {
        "submodules": (
            "tnfr.units",
            "tnfr.constants",
        ),
        "third_party": ("networkx",),
    },
    # Math module symbolic analysis functions
    "get_nodal_equation": {
        "submodules": ("tnfr.math",),
        "third_party": ("sympy",),
    },
    "solve_nodal_equation_constant_params": {
        "submodules": ("tnfr.math",),
        "third_party": ("sympy",),
    },
    "integrated_evolution_symbolic": {
        "submodules": ("tnfr.math",),
        "third_party": ("sympy",),
    },
    "check_convergence_exponential": {
        "submodules": ("tnfr.math",),
        "third_party": ("sympy",),
    },
    "compute_second_derivative_symbolic": {
        "submodules": ("tnfr.math",),
        "third_party": ("sympy",),
    },
    "evaluate_bifurcation_risk": {
        "submodules": ("tnfr.math",),
        "third_party": ("sympy",),
    },
    "latex_export": {
        "submodules": ("tnfr.math",),
        "third_party": ("sympy",),
    },
    "pretty_print": {
        "submodules": ("tnfr.math",),
        "third_party": ("sympy",),
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

    module_name = getattr(exc, "module", None) or ""
    if module_name.startswith("tnfr"):
        return True

    missing_path = getattr(exc, "path", None) or ""
    if missing_path:
        normalized = missing_path.replace("\\", "/")
        if "/tnfr/" in normalized or normalized.endswith("/tnfr"):
            return True

    message = str(exc)
    lowered = message.lower()
    mentions_base_package = "module 'tnfr'" in lowered or 'module "tnfr"' in lowered
    if ("tnfr." in message or mentions_base_package) and (
        "circular import" in lowered or "partially initialized module" in lowered
    ):
        return True

    return False


def _missing_dependency(
    name: str, exc: ImportError, *, module: str | None = None
) -> Callable[..., NoReturn]:
    missing_name = getattr(exc, "name", None)

    def _stub(*args: Any, **kwargs: Any) -> NoReturn:
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


class ExportDependencyError(RuntimeError):
    """Raised when the export dependency manifest is inconsistent."""


def _validate_export_dependencies() -> None:
    """Ensure exported helpers and their manifest entries stay in sync."""

    if "__all__" not in globals():
        # Defensive guard for unusual import orders (should never trigger).
        return

    issues: list[str] = []
    manifest = EXPORT_DEPENDENCIES
    export_names = [name for name in __all__ if name != "__version__"]
    manifest_names = set(manifest)

    for export_name in export_names:
        if export_name not in manifest:
            issues.append(
                f"helper '{export_name}' is exported via __all__ but missing from EXPORT_DEPENDENCIES"
            )
            continue

        entry = manifest[export_name]
        if not isinstance(entry, dict):
            issues.append(
                f"helper '{export_name}' has a malformed manifest entry (expected mapping, got {type(entry)!r})"
            )
            continue

        for key in ("submodules", "third_party"):
            value = entry.get(key)
            if not value:
                issues.append(
                    f"helper '{export_name}' is missing '{key}' dependencies in EXPORT_DEPENDENCIES"
                )

    missing_exports = manifest_names.difference(export_names).difference(_MISSING_EXPORTS)
    for manifest_only in sorted(missing_exports):
        entry = manifest[manifest_only]
        if not isinstance(entry, dict):
            issues.append(
                f"helper '{manifest_only}' has a malformed manifest entry (expected mapping, got {type(entry)!r})"
            )
            continue

        for key in ("submodules", "third_party"):
            value = entry.get(key)
            if not value:
                issues.append(
                    f"helper '{manifest_only}' is missing '{key}' dependencies in EXPORT_DEPENDENCIES"
                )

        issues.append(
            f"helper '{manifest_only}' is listed in EXPORT_DEPENDENCIES but not exported via __all__"
        )

    if issues:
        raise ExportDependencyError(
            "Invalid TNFR export dependency manifest:\n- " + "\n- ".join(issues)
        )


def _assign_exports(module: str, names: tuple[str, ...]) -> bool:
    try:  # pragma: no cover - exercised in import tests
        mod = import_module(f".{module}", __name__)
    except ImportError as exc:  # pragma: no cover - no missing deps in CI
        if _is_internal_import_error(exc):
            raise
        for export_name in names:
            stub = _missing_dependency(export_name, exc, module=module)
            globals()[export_name] = stub
            _MISSING_EXPORTS[export_name] = getattr(stub, "__tnfr_missing_dependency__", {})
        return False
    else:
        for export_name in names:
            globals()[export_name] = getattr(mod, export_name)
        return True


def __getattr__(name: str) -> Any:
    """Lazy load SDK components and handle missing dependencies."""
    # SDK exports - lazy loaded to avoid circular dependencies
    if name == "TNFRNetwork":
        try:
            from .sdk.fluent import TNFRNetwork

            globals()[name] = TNFRNetwork
            return TNFRNetwork
        except ImportError as exc:
            if _is_internal_import_error(exc):
                raise
            stub = _missing_dependency(name, exc, module="sdk.fluent")
            globals()[name] = stub
            return stub
    elif name == "TNFRTemplates":
        try:
            from .sdk.templates import TNFRTemplates

            globals()[name] = TNFRTemplates
            return TNFRTemplates
        except ImportError as exc:
            if _is_internal_import_error(exc):
                raise
            stub = _missing_dependency(name, exc, module="sdk.templates")
            globals()[name] = stub
            return stub
    elif name == "TNFRExperimentBuilder":
        try:
            from .sdk.builders import TNFRExperimentBuilder

            globals()[name] = TNFRExperimentBuilder
            return TNFRExperimentBuilder
        except ImportError as exc:
            if _is_internal_import_error(exc):
                raise
            stub = _missing_dependency(name, exc, module="sdk.builders")
            globals()[name] = stub
            return stub

    raise AttributeError(f"module 'tnfr' has no attribute '{name}'")


_assign_exports("dynamics", ("step", "run"))

_HAS_PREPARE_NETWORK = _assign_exports("ontosim", ("prepare_network",))

_HAS_STRUCTURAL_EXPORTS = _assign_exports(
    "structural", ("create_nfr", "run_sequence", "create_math_nfr")
)

_assign_exports("units", ("get_hz_bridge", "hz_str_to_hz", "hz_to_hz_str"))

# Export math module for symbolic analysis
_assign_exports(
    "math",
    (
        "get_nodal_equation",
        "solve_nodal_equation_constant_params",
        "integrated_evolution_symbolic",
        "check_convergence_exponential",
        "compute_second_derivative_symbolic",
        "evaluate_bifurcation_risk",
        "latex_export",
        "pretty_print",
    ),
)


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
    "prepare_network",
    "create_nfr",
    "get_hz_bridge",
    "hz_str_to_hz",
    "hz_to_hz_str",
    # Math module symbolic analysis
    "get_nodal_equation",
    "solve_nodal_equation_constant_params",
    "integrated_evolution_symbolic",
    "check_convergence_exponential",
    "compute_second_derivative_symbolic",
    "evaluate_bifurcation_risk",
    "latex_export",
    "pretty_print",
    # SDK exports (lazily loaded)
    "TNFRNetwork",
    "TNFRTemplates",
    "TNFRExperimentBuilder",
]

if _HAS_STRUCTURAL_EXPORTS:
    __all__.extend(["run_sequence", "create_math_nfr"])

# Add SDK exports to dependency manifest for validation
EXPORT_DEPENDENCIES["TNFRNetwork"] = {
    "submodules": (
        "tnfr.sdk.fluent",
        "tnfr.structural",
        "tnfr.metrics",
        "tnfr.validation",
    ),
    "third_party": ("networkx",),
}
EXPORT_DEPENDENCIES["TNFRTemplates"] = {
    "submodules": ("tnfr.sdk.templates", "tnfr.sdk.fluent"),
    "third_party": ("networkx",),
}
EXPORT_DEPENDENCIES["TNFRExperimentBuilder"] = {
    "submodules": ("tnfr.sdk.builders", "tnfr.sdk.fluent"),
    "third_party": ("networkx",),
}

_validate_export_dependencies()
