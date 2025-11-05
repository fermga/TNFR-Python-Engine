"""Simplified SDK for non-expert TNFR users.

This module provides a high-level, user-friendly API for creating and
simulating TNFR networks without requiring deep knowledge of the underlying
theory. The SDK maintains full theoretical fidelity while hiding complexity
through fluent interfaces, pre-configured templates, and domain-specific
patterns.

Public API
----------
TNFRNetwork
    Fluent API for creating and evolving TNFR networks with method chaining.
TNFRTemplates
    Pre-configured templates for common domain-specific use cases.
TNFRExperimentBuilder
    Builder pattern for standard TNFR experiment workflows.
NetworkResults
    Structured results container for TNFR metrics and graph state.
"""

from __future__ import annotations

__all__ = [
    "TNFRNetwork",
    "NetworkConfig",
    "NetworkResults",
    "TNFRTemplates",
    "TNFRExperimentBuilder",
]

# Lazy imports to avoid circular dependencies and optional dependency issues
def __getattr__(name: str):
    """Lazy load SDK components."""
    if name == "TNFRNetwork" or name == "NetworkConfig" or name == "NetworkResults":
        from .fluent import TNFRNetwork, NetworkConfig, NetworkResults
        if name == "TNFRNetwork":
            return TNFRNetwork
        elif name == "NetworkConfig":
            return NetworkConfig
        else:
            return NetworkResults
    elif name == "TNFRTemplates":
        from .templates import TNFRTemplates
        return TNFRTemplates
    elif name == "TNFRExperimentBuilder":
        from .builders import TNFRExperimentBuilder
        return TNFRExperimentBuilder
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
