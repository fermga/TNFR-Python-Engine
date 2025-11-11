from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn

from .dynamics import run, step
from .ontosim import prepare_network
from .structural import create_nfr, run_sequence

EXPORT_DEPENDENCIES: dict[str, dict[str, tuple[str, ...]]]
"""Manifest describing required submodules and third-party packages."""

_MISSING_EXPORTS: dict[str, dict[str, Any]]

__version__: str
__all__: list[str]

class ExportDependencyError(RuntimeError):
    """Raised when the export dependency manifest is inconsistent."""

def _is_internal_import_error(exc: ImportError) -> bool: ...
def _missing_dependency(
    name: str,
    exc: ImportError,
    *,
    module: str | None = ...,
) -> Callable[..., NoReturn]: ...
def _validate_export_dependencies() -> None: ...
def _assign_exports(module: str, names: tuple[str, ...]) -> bool: ...
def _emit_missing_dependency_warning() -> None: ...

_HAS_PREPARE_NETWORK: bool
_HAS_RUN_SEQUENCE: bool
