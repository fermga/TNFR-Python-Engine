"""Service layer for TNFR orchestration.

This package provides the service layer that coordinates execution of TNFR
operator sequences while maintaining clean separation of responsibilities
across validation, execution, dynamics, and telemetry concerns.

Public API
----------
TNFROrchestrator
    Main orchestration service coordinating sequence execution.
"""

from __future__ import annotations

from .orchestrator import TNFROrchestrator

__all__ = ("TNFROrchestrator",)
