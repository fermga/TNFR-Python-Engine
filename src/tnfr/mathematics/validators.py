"""Backward-compatible re-export of spectral validators."""

from __future__ import annotations

__all__ = ("NFRValidator",)


def __getattr__(name: str) -> object:
    if name == "NFRValidator":
        from ..validation.spectral import NFRValidator as _NFRValidator

        return _NFRValidator
    raise AttributeError(name)

