"""Emergent integration engine wrapper.

This module re-exports the canonical implementation from
``tnfr.dynamics.emergent_integration_engine`` so that legacy imports from the
``tnfr.engines.integration`` namespace remain valid without duplicating code.
"""

from ...dynamics.emergent_integration_engine import (
    IntegrationOpportunity,
    IntegrationPattern,
    IntegrationResult,
    TNFREmergentIntegrationEngine,
    get_emergent_integration_engine,
    discover_and_apply_integrations,
)

__all__ = [
    "IntegrationOpportunity",
    "IntegrationPattern",
    "IntegrationResult",
    "TNFREmergentIntegrationEngine",
    "get_emergent_integration_engine",
    "discover_and_apply_integrations",
]
