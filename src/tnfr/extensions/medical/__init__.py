"""Medical domain extension for TNFR.

Provides healthcare and therapeutic patterns, specialized health metrics,
and medical-domain visualizations.

Examples
--------
>>> from tnfr.extensions.medical import MedicalExtension
>>> from tnfr.extensions import get_global_registry
>>> 
>>> registry = get_global_registry()
>>> registry.register_extension(MedicalExtension())
>>> patterns = registry.get_domain_patterns("medical")
"""

from __future__ import annotations

from .extension import MedicalExtension

__all__ = ["MedicalExtension"]
