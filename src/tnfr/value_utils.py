"""Conversion helpers with logging for value normalisation.

Wraps conversion callables to standardise error handling and logging.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar
import logging
from .logging import get_module_logger

T = TypeVar("T")

logger = get_module_logger(__name__)

__all__ = ("convert_value",)


def convert_value(
    value: Any,
    conv: Callable[[Any], T],
    *,
    strict: bool = False,
    key: str | None = None,
    log_level: int | None = None,
) -> tuple[bool, T | None]:
    """Attempt to convert a value and report failures.

    Parameters
    ----------
    value : Any
        Input value to convert.
    conv : Callable[[Any], T]
        Callable performing the conversion.
    strict : bool, optional
        Raise exceptions directly instead of logging them. Defaults to ``False``.
    key : str, optional
        Name associated with the value for logging context.
    log_level : int, optional
        Logging level used when reporting failures. Defaults to
        ``logging.DEBUG``.

    Returns
    -------
    tuple[bool, T | None]
        ``(True, result)`` on success or ``(False, None)`` when conversion
        fails.
    """
    try:
        return True, conv(value)
    except (ValueError, TypeError) as exc:
        if strict:
            raise
        level = log_level if log_level is not None else logging.DEBUG
        if key is not None:
            logger.log(level, "Could not convert value for %r: %s", key, exc)
        else:
            logger.log(level, "Could not convert value: %s", exc)
        return False, None
