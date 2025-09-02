"""Utilities for value conversion."""
from __future__ import annotations

from typing import Any, Callable, TypeVar
import logging

T = TypeVar("T")

logger = logging.getLogger(__name__)

__all__ = ["_convert_value"]


def _convert_value(
    value: Any,
    conv: Callable[[Any], T],
    *,
    strict: bool = False,
    key: str | None = None,
    log_level: int | None = None,
) -> tuple[bool, T | None]:
    """Intenta convertir ``value`` usando ``conv`` manejando errores.

    ``log_level`` controla el nivel de logging cuando la conversión falla en
    modo laxo. Por defecto se usa ``logging.ERROR`` si ``strict`` es ``True`` y
    ``logging.DEBUG`` en caso contrario.
    """
    try:
        return True, conv(value)
    except (ValueError, TypeError) as exc:
        level = log_level if log_level is not None else (
            logging.ERROR if strict else logging.DEBUG
        )
        if key is not None:
            logger.log(level, "No se pudo convertir el valor para %r: %s", key, exc)
        else:
            logger.log(level, "No se pudo convertir el valor: %s", exc)
        if strict:
            raise
        return False, None
