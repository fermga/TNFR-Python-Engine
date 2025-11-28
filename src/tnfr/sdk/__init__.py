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
NetworkConfig
    Configuration dataclass for network settings.

Utilities
---------
compare_networks
    Compare metrics across multiple networks.
compute_network_statistics
    Compute extended statistics for a network.
export_to_json
    Export network data to JSON file.
import_from_json
    Import network data from JSON file.
format_comparison_table
    Format network comparison as readable table.
suggest_sequence_for_goal
    Suggest operator sequence for a specific goal.
"""

from __future__ import annotations

__all__ = [
    "TNFRNetwork",
    "NetworkConfig",
    "NetworkResults",
    "TNFRTemplates",
    "TNFRExperimentBuilder",
    "TNFRAdaptiveSystem",
    # Utilities
    "compare_networks",
    "compute_network_statistics",
    "export_to_json",
    "import_from_json",
    "format_comparison_table",
    "suggest_sequence_for_goal",
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
    elif name == "TNFRAdaptiveSystem":
        from .adaptive_system import TNFRAdaptiveSystem

        return TNFRAdaptiveSystem
    elif name in [
        "compare_networks",
        "compute_network_statistics",
        "export_to_json",
        "import_from_json",
        "format_comparison_table",
        "suggest_sequence_for_goal",
    ]:
        from .utils import (
            compare_networks,
            compute_network_statistics,
            export_to_json,
            import_from_json,
            format_comparison_table,
            suggest_sequence_for_goal,
        )

        mapping = {
            "compare_networks": compare_networks,
            "compute_network_statistics": compute_network_statistics,
            "export_to_json": export_to_json,
            "import_from_json": import_from_json,
            "format_comparison_table": format_comparison_table,
            "suggest_sequence_for_goal": suggest_sequence_for_goal,
        }
        return mapping[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
