"""Compatibility shim removed; direct users to :mod:`tnfr.utils.cache`."""

from __future__ import annotations

raise ImportError(
    "The 'tnfr.cache' module has been removed. Import cache helpers from 'tnfr.utils.cache'."
)
