"""Configuration package for TNFR.

This package groups helpers and canonical defaults that orchestrate how
configuration payloads interact with the engine.  The public API mirrors the
previous module level functions so downstream importers remain stable.
"""

from __future__ import annotations

from .feature_flags import context_flags, get_flags
from .init import apply_config, load_config

__all__ = ("load_config", "apply_config", "get_flags", "context_flags")
