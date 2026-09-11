"""Shared scalar parsing for graph and environment configuration."""

from __future__ import annotations

from typing import Any

_TRUE_VALUES = frozenset({"1", "true", "on", "yes", "y", "t", "enable", "enabled"})
_FALSE_VALUES = frozenset({"0", "false", "off", "no", "n", "f", "disable", "disabled"})


def parse_bool(value: Any) -> bool:
    """Parse explicit boolean strings; preserve normal non-string conversion.

    Unknown strings raise instead of silently enabling a flag through Python's
    nonempty-string truthiness. Whitespace and letter case are ignored.
    """
    if not isinstance(value, str):
        return bool(value)
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError("Expected an explicit true/false configuration value")
