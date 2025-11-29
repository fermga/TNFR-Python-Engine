"""**TNFR SDK** - Simple & Powerful API for TNFR Networks ⭐ **OPTIMIZED**

The TNFR SDK provides an intuitive, production-ready interface for creating,
evolving, and analyzing Resonant Fractal Networks with complete theoretical
fidelity. Designed for both newcomers and experts.

**CORE PHILOSOPHY**: Maximum power, minimum complexity.

**QUICK START**:
```python
from tnfr.sdk import TNFR

# One-line network creation and evolution
results = TNFR.create(10).ring().evolve(5).results()
print(f'Coherence: {results.coherence:.3f}')
```

**PRIMARY API** ⭐ **SIMPLIFIED**:
----------
**TNFR**
    Static factory for instant network creation with method chaining.
**Network**
    Core network class with essential TNFR operations.
**Results**
    Lightweight results container with key metrics.

**ADVANCED FEATURES**:
---------
**auto_optimize()**
    One-line self-optimization using unified field analysis.
**template(name)**
    Pre-configured networks for common use cases.
**compare(*networks)**
    Multi-network analysis and comparison.
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
