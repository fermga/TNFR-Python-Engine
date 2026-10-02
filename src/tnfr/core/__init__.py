"""Core architectural components for TNFR engine.

This package contains the foundational interfaces and dependency injection
infrastructure that enable separation of responsibilities across the TNFR
engine layers.

Public API
----------
OperatorRegistry, ValidationService, DynamicsEngine, TelemetryCollector
    Protocol interfaces defining contracts for each architectural layer.
TNFRContainer
    Dependency injection container for configuring engine components.
"""

from __future__ import annotations

from .container import TNFRContainer
from .interfaces import (
    DynamicsEngine,
    OperatorRegistry,
    TelemetryCollector,
    TraceContext,
    ValidationService,
)

__all__ = (
    "OperatorRegistry",
    "ValidationService",
    "DynamicsEngine",
    "TelemetryCollector",
    "TraceContext",
    "TNFRContainer",
)
