"""Utilidades para importaciones opcionales."""

from __future__ import annotations

import importlib
import warnings
import logging
from functools import lru_cache
from typing import Any

__all__ = ["optional_import", "get_numpy"]


logger = logging.getLogger(__name__)


def optional_import(name: str, fallback: Any | None = None) -> Any | None:
    """Importa ``name`` devolviendo ``fallback`` si falla.

    ``name`` puede apuntar a un módulo, submódulo o atributo. Si la
    importación o el acceso al atributo fallan se emite una advertencia y se
    devuelve ``fallback``.

    Parameters
    ----------
    name:
        Ruta completa del módulo, submódulo o atributo a importar.
    fallback:
        Valor a devolver cuando la importación falla. Por defecto ``None``.

    Returns
    -------
    Any | None
        Objeto importado o ``fallback`` si ocurre un error.

    Notes
    -----
    Se devuelve ``fallback`` cuando el módulo no está disponible o el
    atributo solicitado no existe. En ambos casos se emite una advertencia.
    """

    try:
        return importlib.import_module(name)
    except ImportError:
        if "." in name:
            module_name, attr = name.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                return getattr(module, attr)
            except (ImportError, AttributeError):
                pass
        warnings.warn(
            f"No se pudo importar '{name}'; usando valor de fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fallback


@lru_cache(maxsize=1)
def get_numpy(*, warn: bool = False) -> Any | None:
    """Devuelve el módulo :mod:`numpy` o ``None`` si no está disponible.

    Parameters
    ----------
    warn:
        Si es ``True`` se registra una advertencia cuando la importación
        falla. En caso contrario se registra a nivel ``DEBUG``.
    """

    module = optional_import("numpy")
    if module is None:
        log = logger.warning if warn else logger.debug
        log(
            "Fallo al importar numpy, se continuará con el modo no vectorizado"
        )
    return module
