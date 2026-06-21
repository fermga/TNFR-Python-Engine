# TNFR Engines Hub

Centralized location for all TNFR mathematical and optimization engines.

## 🏗️ **Engine Categories**

### 1. **Self-Optimization** (`self_optimization/`)
- **TNFRSelfOptimizingEngine**: Automatic network optimization
- **Physics**: Based on ∂EPI/∂t = νf · ΔNFR(t) nodal equation
- **Capabilities**: Node targeting, multi-operator optimization, real-time metrics

### 2. **Pattern Discovery** (`pattern_discovery/`)
- **TNFREmergentPatternEngine**: Mathematical pattern detection  
- **UnifiedPatternDetector**: Operator sequence analysis
- **Discoveries**: Eigenmodes, spectral cascades, fractal scaling, symmetry breaking

### 3. **Computation** (`computation/`)
- **GPUEngine**: GPU-accelerated TNFR computations
- **FFTEngine**: Fast Fourier Transform processing
- **Performance**: High-throughput parallel processing

### 4. **Integration** (`integration/`)
- **EmergentIntegrationEngine**: Multi-scale analysis
- **Capabilities**: Hierarchical coupling, cross-scale information flow

## 🚀 **Quick Usage**

### Direct Engine Access
```python
from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine
from tnfr.engines.pattern_discovery import TNFREmergentPatternEngine

# Self-optimization
engine = TNFRSelfOptimizingEngine(network)
success, metrics = engine.step(node_id)

# Pattern discovery
pattern_engine = TNFREmergentPatternEngine()
patterns = pattern_engine.discover_patterns(network)
```

### Via SDK (Recommended)
```python
from tnfr.sdk import TNFR

# Integrated auto-optimization
net = TNFR.create(50).random(0.3).auto_optimize()

# Fluent API chains
result = TNFR.create(100).evolve(10).auto_optimize().results()
```

## 📚 **Documentation**

- **Theory**: [AGENTS.md](../../../AGENTS.md) - Single Source of Truth
- **Grammar**: [UNIFIED_GRAMMAR_RULES.md](../../../theory/UNIFIED_GRAMMAR_RULES.md) 
- **Examples**: [examples/](../../../examples/) - Sequential tutorials
- **Individual engines**: See README.md in each subdirectory

## 🎯 **Engine Selection Guide**

| **Use Case** | **Engine** | **Import** |
|--------------|------------|------------|
| Auto-optimize network | Self-Optimization | `from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine` |
| Discover math patterns | Pattern Discovery | `from tnfr.engines.pattern_discovery import TNFREmergentPatternEngine` |
| GPU acceleration | Computation | `from tnfr.engines.computation import GPUEngine` |
| Multi-scale analysis | Integration | `from tnfr.engines.integration import EmergentIntegrationEngine` |
| Simple workflows | SDK | `from tnfr.sdk import TNFR` |

## 🧮 **TNFR Physics Foundation**

All engines implement TNFR principles:
- **Nodal equation**: ∂EPI/∂t = νf · ΔNFR(t)
- **Structural-field tetrad**: Φ_s, |∇φ|, K_φ, ξ_C (associated with φ,γ,π,e notationally; only π genuine)
- **Unified grammar**: U1-U6 operator constraints
- **Canonical invariants**: 6 fundamental preservation laws
- **Structural field tetrad**: Complete multi-scale characterization

**Status**: Production ready, fully integrated with TNFR v0.0.1