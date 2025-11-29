"""
TNFR Engines Hub - Centralized Mathematical & Optimization Tools

This module centralizes all TNFR engines for easier discovery and maintenance.
Each engine implements specific mathematical capabilities based on TNFR physics.

Engines Available
-----------------
1. **Self-Optimization**: Automatic network optimization using TNFR operators
2. **Pattern Discovery**: Mathematical pattern detection and emergence analysis  
3. **Computation**: High-performance computing backends (GPU, FFT, etc.)
4. **Integration**: Emergent integration and multi-scale analysis

Usage
-----
```python
# Direct engine imports
from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine
from tnfr.engines.pattern_discovery import TNFREmergentPatternEngine
from tnfr.engines.computation import GPUEngine, FFTEngine
from tnfr.engines.integration import EmergentIntegrationEngine

# Or via SDK (recommended)
from tnfr.sdk import TNFR
net = TNFR.create(50).auto_optimize()  # Uses self-optimization engine
```

Architecture
------------
Each engine follows TNFR principles:
- Based on nodal equation: ∂EPI/∂t = νf · ΔNFR(t)
- Respects unified grammar (U1-U6)
- Maintains canonical invariants
- Provides measurable improvements

Documentation
-------------
- Main theory: AGENTS.md (Single Source of Truth)
- Grammar rules: UNIFIED_GRAMMAR_RULES.md
- Individual engine docs in each subdirectory
"""

# Import all engines for convenient access
try:
    from .self_optimization.engine import TNFRSelfOptimizingEngine
    __all__ = ["TNFRSelfOptimizingEngine"]
except ImportError:
    __all__ = []

try:
    from .pattern_discovery.mathematical_patterns import TNFREmergentPatternEngine
    __all__.append("TNFREmergentPatternEngine")
except ImportError:
    pass

try:
    from .computation.gpu_engine import GPUEngine
    from .computation.fft_engine import FFTEngine
    __all__.extend(["GPUEngine", "FFTEngine"])
except ImportError:
    pass

try:
    from .integration.emergent_integration import EmergentIntegrationEngine
    __all__.append("EmergentIntegrationEngine")
except ImportError:
    pass

# Version info
__version__ = "0.0.1"
__engines_hub_version__ = "1.0.0"