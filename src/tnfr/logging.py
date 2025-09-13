"""Simplified logging helpers for TNFR modules.

Provides :func:`get_module_logger` which mirrors the default configuration
used by :mod:`tnfr.logging_utils` without importing the heavier module.
This keeps compatibility with existing logging setups while avoiding
additional dependencies when only a plain logger is required.
"""

from __future__ import annotations

import logging

__all__ = ["get_module_logger"]

_LOGGING_CONFIGURED = False


def _configure_root() -> None:
    """Ensure the root logger has handlers and a default format.

    The configuration matches :func:`tnfr.logging_utils.get_logger` so modules
    can safely switch to this lighter helper without affecting existing
    behaviour.
    """

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


def get_module_logger(name: str) -> logging.Logger:
    """Return a module-specific logger configured with TNFR defaults."""

    _configure_root()
    return logging.getLogger(name)
