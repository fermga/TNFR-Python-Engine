"""Compatibility wrapper for deprecated tnfr.logging module."""
from __future__ import annotations

from .logging_utils import get_logger as get_module_logger

__all__ = ["get_module_logger"]
