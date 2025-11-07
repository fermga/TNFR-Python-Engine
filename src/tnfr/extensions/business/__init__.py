"""Business domain extension for TNFR.

Provides business process patterns, organizational change sequences,
and sales/marketing patterns.

Examples
--------
>>> from tnfr.extensions.business import BusinessExtension
>>> from tnfr.extensions import get_global_registry
>>> 
>>> registry = get_global_registry()
>>> registry.register_extension(BusinessExtension())
>>> patterns = registry.get_domain_patterns("business")
"""

from __future__ import annotations

from .extension import BusinessExtension

__all__ = ["BusinessExtension"]
