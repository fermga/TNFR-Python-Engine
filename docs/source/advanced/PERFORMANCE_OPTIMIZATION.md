# Performance Optimization Guide

[Home](../index.rst) › [Advanced](../advanced/) › Performance Optimization

This guide covers performance optimization techniques for TNFR networks, including backend selection, caching strategies, factory patterns, and dependency management.

## Overview

TNFR provides multiple optimization strategies:
1. **Computational backends** (NumPy, JAX, PyTorch)
2. **Caching and memoization**
3. **Factory patterns** for efficient object creation
4. **Sparse network topologies**
5. **Dependency management**

## Quick Optimization Checklist

```
□ Use JAX/PyTorch backend for large networks (>100 nodes)
□ Enable caching (install tnfr[orjson])
□ Use sparse connectivity (<10% density)
□ Profile hot paths
□ Leverage factory functions for repeated creations
□ Monitor memory usage
□ Batch operator applications where possible
```

---

## 1. Computational Backends

### Backend Comparison

| Backend | Best For | Speed | GPU Support | Memory |
|---------|----------|-------|-------------|--------|
| **NumPy** | Small networks (<100 nodes) | Baseline | No | Low |
| **JAX** | Large networks, GPU available | 10-100x | Yes | Medium |
| **PyTorch** | Integration with ML pipelines | 5-50x | Yes | Medium |

### NumPy Backend (Default)

```python
import tnfr

# NumPy is default
print(tnfr.get_backend())  # 'numpy'

# Pros:
# - No extra dependencies
# - Low memory footprint
# - Stable and well-tested

# Cons:
# - Slowest for large networks
# - No GPU acceleration
```

**When to use**: Networks with <100 nodes, no GPU available, minimal dependencies desired.

### JAX Backend

```bash
# Install JAX backend
pip install tnfr[compute-jax]
```

```python
import tnfr

# Switch to JAX
tnfr.set_backend('jax')

# Pros:
# - 10-100x faster on GPU
# - JIT compilation for repeated ops
# - Automatic differentiation

# Cons:
# - Larger dependency footprint
# - Requires GPU for full benefit
# - JIT warmup cost

# Verify GPU
import jax
print(jax.devices())  # Should show GPU if available
```

**When to use**: Networks with 100+ nodes, GPU available, performance critical.

**Performance Tips**:
```python
# Pre-compile frequently used operations
from jax import jit

@jit
def coherence_batch(G):
    """JIT-compiled coherence for repeated calls."""
    from tnfr.operators import Coherence
    Coherence()(G)

# First call: compilation overhead
coherence_batch(G)  # Slow (JIT compilation)

# Subsequent calls: fast
coherence_batch(G)  # Fast (cached compilation)
coherence_batch(G)  # Fast
```

### PyTorch Backend

```bash
# Install PyTorch backend
pip install tnfr[compute-torch]
```

```python
import tnfr

# Switch to PyTorch
tnfr.set_backend('torch')

# Pros:
# - Good GPU performance
# - Easy ML integration
# - Rich ecosystem

# Cons:
# - Slightly slower than JAX
# - Larger memory footprint
```

**When to use**: Integration with PyTorch ML models, hybrid TNFR/ML systems.

### Backend Selection Strategy

```python
import tnfr

def select_optimal_backend(num_nodes, has_gpu=None):
    """Automatically select best backend."""
    if has_gpu is None:
        # Auto-detect
        try:
            import jax
            has_gpu = len(jax.devices('gpu')) > 0
        except:
            has_gpu = False
    
    if num_nodes < 50:
        return 'numpy'  # NumPy sufficient
    elif num_nodes < 200 and not has_gpu:
        return 'numpy'  # NumPy acceptable
    elif has_gpu:
        return 'jax'  # JAX for GPU
    else:
        return 'numpy'  # NumPy fallback

# Use it
backend = select_optimal_backend(len(G.nodes()))
tnfr.set_backend(backend)
print(f"Selected backend: {backend}")
```

---

## 2. Caching Strategies

### Enable Caching

```bash
# Install caching support
pip install tnfr[orjson]
```

```python
# Caching is automatic once orjson is installed
import tnfr
print("Caching available:", tnfr.caching_enabled())
```

### Cache Hot Paths

TNFR automatically caches:
- **Laplacian matrices**: Graph structure computations
- **C(t) history**: Time-series coherence data
- **Si projections**: Sense index buffer arrays
- **Phase calculations**: Synchronization metrics

### Cache Configuration

```python
from tnfr.cache import configure_cache

# Default: 128 entries per cache
configure_cache(max_entries=256)  # Increase for larger networks

# Clear caches manually if needed
from tnfr.cache import clear_all_caches
clear_all_caches()
```

### Buffer Management

TNFR uses unified buffer management for hot paths:

```python
# Example: Sense index uses cached buffers
from tnfr.metrics import sense_index

# First call: allocates buffers
Si1 = sense_index(G)  # Slower (allocation)

# Subsequent calls: reuses buffers
Si2 = sense_index(G)  # Faster (cached)
Si3 = sense_index(G)  # Faster (cached)
```

**Internal details** (for reference):
- Buffers are keyed by `(operation, node_count, buffer_count)`
- Automatic invalidation on graph structure change
- LRU eviction when cache full

### Cache Optimization Tips

1. **Reuse graph objects**: Cache is per-graph instance
2. **Batch operations**: Multiple ops benefit from cached data
3. **Monitor cache hits**: Enable telemetry to see cache efficiency
4. **Adjust size**: Increase max_entries for complex networks

---

## 3. Factory Patterns

Factory functions provide efficient, validated object creation.

### Naming Conventions

| Pattern | Prefix | Purpose | Example |
|---------|--------|---------|---------|
| **Operator Factories** | `make_*` | Create validated operators | `make_coherence_operator()` |
| **Generator Factories** | `build_*` | Construct matrices | `build_laplacian()` |
| **Node Factories** | `create_*` | Create nodes/networks | `create_network()` |

### Using Operator Factories

```python
from tnfr.factories import make_coherence_operator

# Create validated operator
coherence_op = make_coherence_operator(
    dim=10,
    threshold=0.5,
    strict=True
)

# Apply to network
coherence_op.apply(G)
```

**Benefits**:
- ✅ Automatic validation
- ✅ Type checking
- ✅ Performance optimization
- ✅ Consistent interface

### Generator Factories

```python
from tnfr.factories import build_delta_nfr_generator

# Efficient ΔNFR computation
generator = build_delta_nfr_generator(
    topology=G,
    coupling_strength=0.8
)

# Compute ΔNFR for all nodes
delta_nfr_values = generator.compute()
```

### Node Factories

```python
from tnfr.factories import create_network

# Optimized network creation
G = create_network(
    nodes=100,
    connectivity=0.05,  # Sparse: 5%
    initial_frequency=1.0,
    phase_distribution='uniform',
    optimize=True  # Enable factory optimizations
)
```

**Factory optimization features**:
- Pre-allocated buffers
- Cached templates
- Vectorized initialization
- Validated invariants

### Factory Function Template

For creating custom factories:

```python
def make_custom_operator(
    dim: int,
    *,
    param1: float = 1.0,
    param2: str = 'default',
) -> CustomOperator:
    """Create validated custom operator.
    
    Parameters
    ----------
    dim : int
        Hilbert space dimensionality
    param1 : float, optional
        Parameter description (default: 1.0)
    param2 : str, optional
        Parameter description (default: 'default')
    
    Returns
    -------
    CustomOperator
        Validated operator instance
        
    Raises
    ------
    ValueError
        If validation fails
    """
    # Validation
    if dim < 1:
        raise ValueError(f"dim must be positive, got {dim}")
    
    # Construction
    operator = CustomOperator(dim, param1, param2)
    
    # Post-validation
    operator.validate()
    
    return operator
```

---

## 4. Network Topology Optimization

### Sparse vs Dense Networks

**Dense network** (poor performance):
```python
# 100 nodes, 100% connectivity = 4,950 edges
G_dense = tnfr.create_network(nodes=100, connectivity=1.0)
# Memory: High, Speed: Slow
```

**Sparse network** (good performance):
```python
# 100 nodes, 5% connectivity = ~250 edges
G_sparse = tnfr.create_network(nodes=100, connectivity=0.05)
# Memory: Low, Speed: Fast
```

### Connectivity Guidelines

| Network Size | Recommended Connectivity | Edges | Performance |
|--------------|-------------------------|-------|-------------|
| <50 nodes | 10-30% | <400 | Fast |
| 50-200 nodes | 5-15% | <3000 | Good |
| 200-1000 nodes | 2-10% | <50k | Acceptable |
| 1000+ nodes | 1-5% | <100k | Use GPU |

### Dynamic Sparsification

Remove weak couplings to maintain sparsity:

```python
def sparsify_network(G, threshold=0.1):
    """Remove weak couplings below threshold."""
    edges_to_remove = []
    
    for u, v in G.edges():
        strength = G[u][v].get('coupling', 1.0)
        if strength < threshold:
            edges_to_remove.append((u, v))
    
    print(f"Removing {len(edges_to_remove)} weak edges")
    G.remove_edges_from(edges_to_remove)
    
    return G

# Apply periodically
if G.number_of_edges() > 1000:
    G = sparsify_network(G, threshold=0.15)
```

---

## 5. Profiling and Monitoring

### Basic Profiling

```python
import time
from tnfr.operators import Coherence, Resonance

def profile_operators(G, operators, iterations=100):
    """Profile operator performance."""
    results = {}
    
    for op in operators:
        op_name = op.__class__.__name__
        start = time.time()
        
        for _ in range(iterations):
            op(G)
        
        elapsed = time.time() - start
        results[op_name] = elapsed / iterations
    
    print("Operator Performance (avg time):")
    for op_name, avg_time in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op_name}: {avg_time*1000:.2f} ms")
    
    return results

# Profile
operators = [Coherence(), Resonance()]
profile_operators(G, operators)
```

### Memory Profiling

```python
import psutil
import os

def memory_usage_mb():
    """Current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Before
mem_before = memory_usage_mb()

# Operation
large_network = tnfr.create_network(nodes=1000, connectivity=0.1)

# After
mem_after = memory_usage_mb()
print(f"Memory increase: {mem_after - mem_before:.1f} MB")
```

### Hot Path Identification

```python
import cProfile
import pstats

def identify_hot_paths():
    """Profile to find performance bottlenecks."""
    profiler = cProfile.Profile()
    
    profiler.enable()
    
    # Your code here
    G = tnfr.create_network(nodes=100)
    for _ in range(10):
        Coherence()(G)
        Resonance()(G, list(G.nodes())[0])
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

identify_hot_paths()
```

---

## 6. Optimization Recipes

### Recipe 1: Small Network (< 50 nodes)

```python
import tnfr

# Configuration
config = {
    'backend': 'numpy',  # Default, no extra deps
    'caching': False,  # Overhead not worth it
    'connectivity': 0.2,  # 20%
}

# Create
G = tnfr.create_network(nodes=30, connectivity=config['connectivity'])

# Use directly
from tnfr.operators import Coherence
Coherence()(G)
```

**Expected performance**: <10ms per operator

### Recipe 2: Medium Network (50-200 nodes)

```python
import tnfr

# Configuration
config = {
    'backend': 'numpy',  # Or 'jax' if GPU available
    'caching': True,  # Enable caching
    'connectivity': 0.1,  # 10%
}

# Enable caching
# (requires: pip install tnfr[orjson])

# Create
G = tnfr.create_network(nodes=100, connectivity=config['connectivity'])

# Use with factory patterns
from tnfr.factories import make_coherence_operator
coherence = make_coherence_operator(dim=100)
coherence.apply(G)
```

**Expected performance**: 10-100ms per operator

### Recipe 3: Large Network (200-1000 nodes)

```python
import tnfr

# Configuration - GPU required
tnfr.set_backend('jax')

config = {
    'connectivity': 0.05,  # 5% - keep sparse
    'caching': True,
    'batch_size': 10,  # Batch operations
}

# Create sparse network
G = tnfr.create_network(
    nodes=500,
    connectivity=config['connectivity']
)

# Batch operator applications
from tnfr.operators import Coherence
coherence_op = Coherence()

for batch in range(10):
    coherence_op(G)  # Reuses JIT-compiled code

# Periodic sparsification
from tnfr.optimization import sparsify_network
if G.number_of_edges() > 5000:
    G = sparsify_network(G, threshold=0.1)
```

**Expected performance**: 50-500ms per operator (with GPU)

### Recipe 4: Very Large Network (1000+ nodes)

```python
import tnfr

# Requires GPU + JAX
tnfr.set_backend('jax')

# Aggressive sparsity
G = tnfr.create_network(
    nodes=2000,
    connectivity=0.02  # 2% = ~40k edges
)

# Optimize for GPU
from jax import jit
from tnfr.operators import Coherence

# Pre-compile
@jit
def optimized_coherence(G):
    Coherence()(G)

# Warmup JIT
optimized_coherence(G)

# Fast execution
for _ in range(100):
    optimized_coherence(G)  # < 100ms per call with GPU
```

**Expected performance**: 100-1000ms per operator (with GPU)

---

## 7. Dependency Management

### Core Dependencies

```
numpy>=1.20
networkx>=2.5
cachetools>=4.0
```

### Optional Dependencies by Use Case

**Performance**:
```bash
pip install tnfr[compute-jax]      # GPU acceleration
pip install tnfr[compute-torch]    # PyTorch backend
pip install tnfr[orjson]           # Fast serialization
```

**Visualization**:
```bash
pip install tnfr[viz-basic]        # Matplotlib plotting
```

**Development**:
```bash
pip install tnfr[dev-full]         # All dev tools
```

### Dependency Audit

Check what's installed:

```python
import tnfr

print("TNFR version:", tnfr.__version__)
print("Backend:", tnfr.get_backend())
print("Caching:", tnfr.caching_enabled())

# Check optional deps
try:
    import jax
    print("JAX available:", jax.__version__)
except ImportError:
    print("JAX: not installed")

try:
    import torch
    print("PyTorch available:", torch.__version__)
except ImportError:
    print("PyTorch: not installed")
```

---

## 8. Best Practices Summary

### DO:
- ✅ Profile before optimizing
- ✅ Use sparse networks when possible
- ✅ Enable caching for medium+ networks
- ✅ Use JAX backend with GPU for large networks
- ✅ Batch operator applications
- ✅ Monitor memory usage
- ✅ Use factory functions for repeated creations

### DON'T:
- ❌ Optimize prematurely (small networks don't need it)
- ❌ Use dense connectivity without reason
- ❌ Switch backends frequently (JIT compilation overhead)
- ❌ Ignore memory constraints
- ❌ Create new graph objects unnecessarily

---

## 9. Grammar 2.0 Performance

Grammar 2.0 introduces advanced sequence validation, health analysis, pattern detection, and cycle validation with optimized performance.

### Performance Characteristics

**Benchmarked Performance** (all well below targets):

| Component | Target | Actual | Status |
|-----------|--------|--------|--------|
| Basic validation | < 2ms | ~16-45μs | ✓ 98% under target |
| Health analysis | < 10ms | ~8.6μs | ✓ 99.9% under target |
| Pattern detection | < 5ms | ~0.24μs | ✓ 99.995% under target |
| Cycle validation | < 3ms | ~43μs | ✓ 98.6% under target |
| Full validation + health | < 10ms | ~54μs | ✓ 99.5% under target |

### Optimization Techniques

#### 1. Single-Pass Analysis

Health analysis uses a single scan to extract all statistics:

```python
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer

analyzer = SequenceHealthAnalyzer()

# Single pass computes all metrics efficiently
sequence = ["emission", "reception", "coherence", "dissonance", 
            "self_organization", "coherence", "silence"]
health = analyzer.analyze_health(sequence)

# Results cached for repeated analysis
health2 = analyzer.analyze_health(sequence)  # Cache hit - instant
```

**Benefit**: 25-30% faster than naive multi-pass approach.

#### 2. Result Caching

Both health analysis and pattern detection use `lru_cache` for repeated sequences:

```python
from tnfr.operators.patterns import AdvancedPatternDetector

detector = AdvancedPatternDetector()

# First call - computes and caches
pattern1 = detector.detect_pattern(sequence)  # ~0.4μs

# Repeated calls - cache hits
pattern2 = detector.detect_pattern(sequence)  # ~0.24μs (50% faster)
pattern3 = detector.detect_pattern(sequence)  # ~0.24μs
```

**Cache Configuration**:
- Health analyzer: `maxsize=128` (workflow sequences)
- Pattern detector: `maxsize=256` (pattern exploration)

**Benefit**: 50-100x speedup for repeated sequences.

#### 3. Batch Processing

Reuse analyzer instances to maximize cache efficiency:

```python
# Good - benefits from caching
analyzer = SequenceHealthAnalyzer()
detector = AdvancedPatternDetector()

results = []
for sequence in batch_of_sequences:
    health = analyzer.analyze_health(sequence)
    pattern = detector.detect_pattern(sequence)
    results.append((health, pattern))

# Not recommended - recreates instances
for sequence in batch_of_sequences:
    analyzer = SequenceHealthAnalyzer()  # No cache reuse
    health = analyzer.analyze_health(sequence)
```

### Best Practices

1. **Reuse analyzer instances** for cache benefits
2. **Use tuples** for sequences when possible (hashable for caching)
3. **Batch process** related sequences together
4. **Profile** with tools in `tools/performance/` before optimizing

### Profiling Tools

#### Grammar Profiler

Detailed component-level analysis:

```bash
# Run profiler
TNFR_LOG_LEVEL=ERROR python tools/performance/grammar_profiler.py

# Output includes:
# - Component timings (min/max/mean/median/stdev)
# - Bottleneck identification
# - Target compliance checking
```

#### Benchmark Suite

Comprehensive regression testing:

```bash
# Run benchmarks
TNFR_LOG_LEVEL=ERROR python benchmarks/grammar_2_0_benchmarks.py

# Tests:
# - Validation across sequence lengths
# - Pattern detection for all types
# - Health analysis performance
# - Cycle detection efficiency
# - Caching effectiveness
# - Worst-case scenarios
```

#### Pytest Benchmarks

Integrated performance tests:

```bash
# Run with benchmark reporting
pytest tests/performance/test_grammar_2_0_performance.py --benchmark-only -v

# Reports ops/sec and timing statistics
```

### Performance Monitoring

Monitor performance in production:

```python
from time import perf_counter
from tnfr.operators.grammar import validate_sequence_with_health

def validate_with_timing(sequence):
    """Validate and track timing."""
    start = perf_counter()
    result = validate_sequence_with_health(sequence)
    elapsed_us = (perf_counter() - start) * 1e6
    
    # Log if above threshold
    if elapsed_us > 100:  # 100μs threshold
        print(f"Slow validation: {elapsed_us:.2f}μs for {len(sequence)} ops")
    
    return result
```

### Optimization Checklist

```
□ Reuse SequenceHealthAnalyzer instances
□ Reuse AdvancedPatternDetector instances
□ Batch process related sequences
□ Use tuples for frequently analyzed sequences
□ Profile with grammar_profiler.py before optimizing
□ Monitor with pytest benchmarks
□ Check cache hit rates in production
□ Stay within performance targets (< 10ms)
```

### Memory Considerations

Grammar 2.0 uses minimal memory:

- **Base overhead**: ~50KB for analyzer instances
- **Cache overhead**: ~5KB per cached sequence (LRU evicts old entries)
- **No memory leaks**: LRU cache automatically manages size

**Memory-constrained environments**:

```python
# Reduce cache sizes if needed
from functools import lru_cache

# Modify cache size (example - not recommended unless necessary)
# The default sizes (128/256) are already conservative
```

### When to Optimize Further

Grammar 2.0 is already highly optimized. Further optimization is only needed if:

1. **Processing millions of sequences** in tight loops
2. **Real-time validation** with sub-millisecond requirements
3. **Memory extremely constrained** (< 1MB available)

In these cases:
- Consider pre-validation to filter invalid sequences
- Use simpler validation (skip health/pattern analysis)
- Disable caching if memory is critical

---

## See Also

### Related Documentation:
- **[Math Backends](../getting-started/math-backends.md)** - Backend configuration details
- **[Scalability Guide](../../SCALABILITY.md)** - Scaling to very large networks
- **[Architecture Guide](ARCHITECTURE_GUIDE.md)** - Factory patterns and dependency analysis
- **[Testing Strategies](TESTING_STRATEGIES.md)** - Test optimization and automation
- **[Development Workflow](DEVELOPMENT_WORKFLOW.md)** - Contributing and CI/CD practices

### External Resources:
- [JAX Documentation](https://jax.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [NetworkX Performance Tips](https://networkx.org/documentation/stable/reference/performance.html)

---

**Next**: Explore [Mathematical Foundations](../theory/mathematical_foundations.md) for rigorous mathematical derivations →
