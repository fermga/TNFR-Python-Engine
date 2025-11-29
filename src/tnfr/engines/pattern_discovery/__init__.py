"""TNFR Pattern Discovery Engines

Mathematical pattern detection and emergence analysis tools.
Detects patterns in operator sequences and emergent mathematical structures.

Main Classes:
- TNFREmergentPatternEngine: Mathematical pattern discovery
- UnifiedPatternDetector: Operator sequence pattern detection

Usage:
```python
from tnfr.engines.pattern_discovery import TNFREmergentPatternEngine
pattern_engine = TNFREmergentPatternEngine()
patterns = pattern_engine.discover_patterns(network)
```
"""

try:
    from .mathematical_patterns import TNFREmergentPatternEngine
    __all__ = ["TNFREmergentPatternEngine"]
except ImportError:
    __all__ = []

try:
    from .operator_patterns import UnifiedPatternDetector
    __all__.append("UnifiedPatternDetector")
except ImportError:
    pass
