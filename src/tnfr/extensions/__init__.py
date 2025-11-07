"""TNFR Extension System - Enable domain-specific extensions.

This module provides the infrastructure for extending TNFR with domain-specific
patterns, health analyzers, visualizers, and cookbook recipes.

Examples
--------
>>> from tnfr.extensions import ExtensionRegistry, TNFRExtension
>>> registry = ExtensionRegistry()
>>> registry.register_extension(MedicalExtension())
>>> patterns = registry.get_domain_patterns("medical")
"""

from __future__ import annotations

from .base import TNFRExtension, ExtensionRegistry, PatternDefinition
from .registry import get_global_registry

__all__ = [
    "TNFRExtension",
    "ExtensionRegistry",
    "PatternDefinition",
    "get_global_registry",
]
