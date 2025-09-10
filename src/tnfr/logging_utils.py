"""Logging utilities for TNFR.

Provides a helper to obtain module-specific loggers with a
consistent configuration across the project.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Hashable, Mapping

__all__ = ("get_logger", "warn_once")


_LOGGING_CONFIGURED = False


def _configure_root() -> None:
    """Configure the root logger if it has no handlers."""

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    root = logging.getLogger()
    if not root.handlers:
        kwargs = {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
        if root.level == logging.NOTSET:
            kwargs["level"] = logging.INFO
        logging.basicConfig(**kwargs)

    _LOGGING_CONFIGURED = True


_configure_root()


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger."""

    return logging.getLogger(name)



def warn_once(
    logger: logging.Logger,
    msg: str,
    *,
    maxsize: int = 1024,
) -> callable:
    """Return a function that logs ``msg`` once per key.

    The returned callable accepts a mapping of keys to values. Keys are
    tracked in an LRU set limited to ``maxsize`` entries. When called,
    new keys trigger a warning with their associated values while
    repeated keys are ignored. The callable exposes ``clear()`` to reset
    the tracked keys, useful for tests.
    """
    _seen: OrderedDict[Hashable, None] = OrderedDict()

    def _log(mapping: Mapping[Hashable, Any]) -> None:
        new: dict[Hashable, Any] = {}
        for k, v in mapping.items():
            if k in _seen:
                _seen.move_to_end(k)
            else:
                _seen[k] = None
                new[k] = v
                if len(_seen) > maxsize:
                    _seen.popitem(last=False)
        if new:
            logger.warning(msg, new)

    _log.clear = _seen.clear  # type: ignore[attr-defined]
    return _log

