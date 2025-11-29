"""TNFR Self-Optimization Engine

Automatic network optimization using TNFR operators and physics.
Based on nodal equation: ∂EPI/∂t = νf · ΔNFR(t)

Main Classes:
- TNFRSelfOptimizingEngine: Core optimization engine

Usage:
```python
from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine
engine = TNFRSelfOptimizingEngine(network)
success, metrics = engine.step(node_id)
```
"""

from .engine import TNFRSelfOptimizingEngine

__all__ = ["TNFRSelfOptimizingEngine"]
