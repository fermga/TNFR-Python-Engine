"""Global extension registry singleton.

Provides a global registry instance for TNFR extensions that can be accessed
throughout the application.

Examples
--------
>>> from tnfr.extensions import get_global_registry
>>> registry = get_global_registry()
>>> registry.register_extension(MyExtension())
"""

from __future__ import annotations

from .base import ExtensionRegistry

__all__ = ["get_global_registry"]


_global_registry: ExtensionRegistry | None = None


def get_global_registry() -> ExtensionRegistry:
    """Get or create the global extension registry.
    
    Returns
    -------
    ExtensionRegistry
        Global registry singleton instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ExtensionRegistry()
    return _global_registry
