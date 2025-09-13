"""Simplified logging helpers for TNFR modules.

Provides :func:`get_module_logger` which mirrors the default configuration
used by :mod:`tnfr.logging_utils` without importing the heavier module.
This keeps compatibility with existing logging setups while avoiding
additional dependencies when only a plain logger is required.
"""

from __future__ import annotations

import logging

from .logging_base import _configure_root

__all__ = ["get_module_logger"]


def get_module_logger(name: str) -> logging.Logger:
    """Return a module-specific logger configured with TNFR defaults."""

    _configure_root()
    return logging.getLogger(name)
