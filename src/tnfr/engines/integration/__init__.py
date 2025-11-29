"""TNFR Integration Engines

Emergent integration and multi-scale analysis tools.
Handles hierarchical coupling and cross-scale information flow.

Main Classes:
- EmergentIntegrationEngine: Multi-scale emergent integration

Usage:
```python
from tnfr.engines.integration import EmergentIntegrationEngine
integration_engine = EmergentIntegrationEngine()
result = integration_engine.integrate_scales(network)
```
"""

try:
    from .emergent_integration import EmergentIntegrationEngine
    __all__ = ["EmergentIntegrationEngine"]
except ImportError:
    __all__ = []
