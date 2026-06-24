"""Runtime helpers for structural operator glyphs decoupled from validation internals.

This module provides utilities for working with glyphs (structural symbols like
AL, EN, IL, etc.) that represent the application of structural operators to nodes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ("last_glyph",)


def last_glyph(nd: Mapping[str, Any]) -> str | None:
    """Return the most recent glyph for node or ``None``."""

    hist = nd.get("glyph_history")
    return hist[-1] if hist else None
