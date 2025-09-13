"""Shared logging configuration helpers for TNFR modules.

Centralises root logger configuration so that both lightweight and full
logging helpers apply the same defaults without duplicating code.
"""

from __future__ import annotations

import logging

__all__ = ["_configure_root"]

_LOGGING_CONFIGURED = False


def _configure_root() -> None:
    """Ensure the root logger has handlers and a default format."""

    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    root = logging.getLogger()
    if not root.handlers:
        kwargs = {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
        if root.level == logging.NOTSET:
            kwargs["level"] = logging.INFO
        logging.basicConfig(**kwargs)

    _LOGGING_CONFIGURED = True

