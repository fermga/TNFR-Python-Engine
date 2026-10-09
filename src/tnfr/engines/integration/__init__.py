"""TNFR Integration Engines

Configured computational integration and reuse opportunities.
Recommendations concern implementations and measured evidence; they do not
derive hierarchical coupling or physical cross-scale dynamics.

Main Classes:
- TNFREmergentIntegrationEngine: Computational integration recommendations

Usage:
```python
from tnfr.engines.integration import TNFREmergentIntegrationEngine
engine = TNFREmergentIntegrationEngine()
opportunities = engine.discover_integration_opportunities(network)
```
"""

try:
    from .emergent_integration import TNFREmergentIntegrationEngine

    __all__ = ["TNFREmergentIntegrationEngine"]
except ImportError:
    __all__ = []
