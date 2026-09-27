"""TNFR Pattern Discovery Engines

Configured graph and operator-sequence pattern diagnostics.
Reported patterns retain each detector's assumptions and observation scope;
they do not establish autonomous formation or physical emergence.

Main Classes:
- TNFREmergentPatternEngine: Mathematical pattern discovery
- UnifiedPatternDetector: Operator sequence pattern detection

Usage:
```python
from tnfr.engines.pattern_discovery import TNFREmergentPatternEngine
pattern_engine = TNFREmergentPatternEngine()
patterns = pattern_engine.discover_all_patterns(network)
```
"""

try:
    from .mathematical_patterns import TNFREmergentPatternEngine

    __all__ = ["TNFREmergentPatternEngine"]
except ImportError:
    __all__ = []

try:
    from ...operators.pattern_detection import UnifiedPatternDetector

    __all__.append("UnifiedPatternDetector")
except ImportError:
    pass
