"""Registry mapping operator names to their classes."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .definitions import Operador


OPERADORES: dict[str, type["Operador"]] = {}


def register_operator(cls: type["Operador"]) -> type["Operador"]:
    """Register ``cls`` under its declared ``name`` in :data:`OPERADORES`."""

    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"El operador {cls.__name__} debe declarar un atributo 'name' no vacío"
        )

    existing = OPERADORES.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"El operador '{name}' ya está registrado")

    OPERADORES[name] = cls
    return cls


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


__all__ = ("OPERADORES", "register_operator", "discover_operators")
