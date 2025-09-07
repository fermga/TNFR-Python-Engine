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
    """Attempt to convert ``value`` using ``conv`` with error handling.

    ``log_level`` controls the logging level when conversion fails in lax
    mode. Defaults to ``logging.DEBUG``. If ``strict`` is ``True`` the
    exception is raised and no log is emitted.
    """
    try:
        return True, conv(value)
    except (ValueError, TypeError) as exc:
        if strict:
            raise
        level = log_level if log_level is not None else logging.DEBUG
        if key is not None:
            logger.log(level, "No se pudo convertir el valor para %r: %s", key, exc)
        else:
            logger.log(level, "No se pudo convertir el valor: %s", exc)
        return False, None
