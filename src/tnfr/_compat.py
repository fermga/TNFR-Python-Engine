"""Compatibility helpers for bridging typing features across Python versions."""

from __future__ import annotations

try:  # pragma: no cover - exercised implicitly by importers
    from typing import TypeAlias  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - Python < 3.10
    from typing_extensions import TypeAlias  # type: ignore[assignment]

__all__ = ["TypeAlias"]
