"""TNFR Integration Engines

Emergent integration and multi-scale analysis tools.
Handles hierarchical coupling and cross-scale information flow.

Main Classes:
- TNFREmergentIntegrationEngine: Multi-scale emergent integration

Usage:
```python
from tnfr.engines.integration import TNFREmergentIntegrationEngine
engine = TNFREmergentIntegrationEngine()
result = engine.integrate_scales(network)
```
"""

try:
    from .emergent_integration import TNFREmergentIntegrationEngine

    __all__ = ["TNFREmergentIntegrationEngine"]
except ImportError:
    __all__ = []
