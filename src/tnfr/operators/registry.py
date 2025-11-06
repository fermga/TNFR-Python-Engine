"""Registry mapping operator names to their classes."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

from ..config.operator_names import canonical_operator_name

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .definitions import Operator

OPERATORS: dict[str, type["Operator"]] = {}


def register_operator(cls: type["Operator"]) -> type["Operator"]:
    """Register ``cls`` under its declared ``name`` in :data:`OPERATORS`."""

    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"Operator {cls.__name__} must declare a non-empty 'name' attribute"
        )

    existing = OPERATORS.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"Operator '{name}' is already registered")

    OPERATORS[name] = cls
    return cls


def get_operator_class(name: str) -> type["Operator"]:
    """Return the operator class registered for ``name`` or its canonical alias."""

    try:
        return OPERATORS[name]
    except KeyError:
        canonical = canonical_operator_name(name)
        if canonical == name:
            raise
        try:
            return OPERATORS[canonical]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(name) from exc


def discover_operators() -> None:
    """Import all operator submodules so their decorators run."""

    package = importlib.import_module("tnfr.operators")
    package_path = getattr(package, "__path__", None)
    if not package_path:
        return

    if getattr(package, "_operators_discovered", False):  # pragma: no cover - cache
        return

    prefix = f"{package.__name__}."
    for module_info in pkgutil.walk_packages(package_path, prefix):
        if module_info.name == f"{prefix}registry":
            continue
        importlib.import_module(module_info.name)

    setattr(package, "_operators_discovered", True)


__all__ = (
    "OPERATORS",
    "register_operator",
    "discover_operators",
    "get_operator_class",
)
