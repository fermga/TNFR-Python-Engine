"""Compatibility shim removed; direct users to :mod:`tnfr.utils.io`."""

from __future__ import annotations

raise ImportError(
    "The 'tnfr.io' module has been removed. Import IO helpers from 'tnfr.utils.io'."
)
