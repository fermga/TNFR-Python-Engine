# Migration Guide: Leveraging TNFR Optimization Infrastructure

## Overview

This guide helps you migrate existing TNFR code to take advantage of the optimization infrastructure. The TNFR engine already includes comprehensive optimization capabilities - this guide shows you how to use them.

## Quick Win Checklist

If you're using TNFR and want immediate performance improvements:

- [ ] **Switch to vectorized backend** (default, but verify)
- [ ] **Configure graph for your topology** (sparse vs dense)
- [ ] **Enable profiling** to identify bottlenecks
- [ ] **Consider GPU backends** for large graphs (>10k nodes)
- [ ] **Tune cache parameters** for repeated computations

## Migration Scenarios

### Scenario 1: Basic Usage (No Code Changes Needed)

**Before:**
```python
from tnfr.dynamics.dnfr import default_compute_delta_nfr
import networkx as nx

G = nx.erdos_renyi_graph(1000, 0.1)
default_compute_delta_nfr(G)
```

**Status:** ✅ Already optimized!

The default implementation automatically uses NumPy vectorization when available. No changes needed.

**To verify optimization is active:**
```python
profile = {}
default_compute_delta_nfr(G, profile=profile)
print(f"Execution path: {profile.get('dnfr_path')}")  # Should show "vectorized"
```

### Scenario 2: Explicit Backend Selection

**Before (if you were using custom implementations):**
```python
# Custom ΔNFR computation
def my_compute_dnfr(G):
    for node in G.nodes():
        # Manual computation...
        pass
```

**After (using backends):**
```python
from tnfr.backends import get_backend

backend = get_backend("numpy")  # or "jax", "torch"
backend.compute_delta_nfr(G)
```

**Benefits:**
- ✅ Automatic vectorization
- ✅ Intelligent caching
- ✅ Sparse/dense strategy selection
- ✅ No manual loop writing

### Scenario 3: Large-Scale Simulations

**Before:**
```python
import networkx as nx
from tnfr.dynamics.dnfr import default_compute_delta_nfr

# Large graph
G = nx.barabasi_albert_graph(10000, 5)

# This works but might be slow
for step in range(100):
    default_compute_delta_nfr(G)
    update_phases(G)
```

**After (optimized for repeated computations):**
```python
from tnfr.backends import get_backend

# Use JAX backend with JIT compilation
backend = get_backend("jax")

# First call compiles (slow)
backend.compute_delta_nfr(G)

# Subsequent calls reuse compiled code (fast!)
for step in range(100):
    backend.compute_delta_nfr(G)  # ~10-50x faster
    update_phases(G)
```

**Benefits:**
- ✅ JIT compilation amortizes over multiple runs
- ✅ GPU acceleration (if available)
- ✅ Minimal memory overhead

### Scenario 4: Memory-Constrained Environments

**Before (might run out of memory):**
```python
G = nx.erdos_renyi_graph(50000, 0.1)
default_compute_delta_nfr(G)  # Could exhaust memory
```

**After (tuned for memory efficiency):**
```python
from tnfr.backends import get_backend

# Configure for low memory usage
G.graph["DNFR_CHUNK_SIZE"] = 1000  # Process in chunks
backend = get_backend("numpy")

# Limit cache size
backend.compute_delta_nfr(G, cache_size=2)

# For coherence matrix, use sparse storage
G.graph["COHERENCE"] = {
    "store_mode": "sparse",
    "threshold": 0.2,  # Only store edges above threshold
}
```

**Benefits:**
- ✅ Reduced peak memory usage
- ✅ Chunked processing
- ✅ Sparse matrix storage

### Scenario 5: Dense Graphs

**Before (suboptimal for dense graphs):**
```python
# Dense graph
G = nx.erdos_renyi_graph(1000, 0.5)  # 50% density
default_compute_delta_nfr(G)
```

**After (optimized for dense topology):**
```python
# Force dense matrix strategy
G.graph["dnfr_force_dense"] = True
backend = get_backend("numpy")
backend.compute_delta_nfr(G)

# For coherence matrix
G.graph["COHERENCE"] = {
    "store_mode": "dense",  # Use dense matrix
}
```

**Benefits:**
- ✅ Matrix multiplication instead of edge iteration
- ✅ Better cache locality
- ✅ ~2x speedup for dense graphs

### Scenario 6: Parallel Si Computation

**Before (sequential):**
```python
from tnfr.metrics.sense_index import compute_Si

G = nx.barabasi_albert_graph(5000, 5)
si_values = compute_Si(G, inplace=False)  # Sequential
```

**After (parallel):**
```python
# Configure parallelization
G.graph["SI_N_JOBS"] = 4  # Use 4 worker processes

si_values = compute_Si(G, inplace=False, n_jobs=4)
```

**Or use vectorized backend:**
```python
backend = get_backend("numpy")  # Automatic vectorization
si_values = backend.compute_si(G, inplace=False)
```

**Benefits:**
- ✅ Parallel computation for Python fallback
- ✅ Vectorized computation with NumPy
- ✅ ~2-15x speedup depending on graph size

## Configuration Migration

### ΔNFR Weights

**Before (if you were modifying code):**
```python
# Hard-coded weights in custom function
def my_dnfr_compute(G):
    phase_weight = 0.4
    epi_weight = 0.3
    # ...
```

**After (declarative configuration):**
```python
G.graph["DNFR_WEIGHTS"] = {
    "phase": 0.4,
    "epi": 0.3,
    "vf": 0.2,
    "topo": 0.1,
}

# Weights are automatically applied
backend.compute_delta_nfr(G)
```

### Si Weights

**Before:**
```python
# Manual Si calculation with custom weights
alpha, beta, gamma = 0.4, 0.3, 0.3
# ... manual computation
```

**After:**
```python
G.graph["SI_WEIGHTS"] = {
    "alpha": 0.4,  # νf weight
    "beta": 0.3,   # Phase alignment
    "gamma": 0.3,  # ΔNFR attenuation
}

backend.compute_si(G)
```

### Coherence Matrix

**Before:**
```python
# Manual similarity computation
def compute_coherence_matrix(G):
    n = len(G.nodes())
    W = [[0.0] * n for _ in range(n)]
    # ... nested loops
    return W
```

**After:**
```python
from tnfr.metrics.coherence import coherence_matrix

# Configure once
G.graph["COHERENCE"] = {
    "weights": {
        "phase": 0.25,
        "epi": 0.25,
        "vf": 0.25,
        "si": 0.25,
    },
    "scope": "neighbors",  # or "all"
    "store_mode": "sparse",
    "threshold": 0.1,
}

# Compute (automatically vectorized)
nodes, W = coherence_matrix(G)
```

## Performance Optimization Workflow

### Step 1: Establish Baseline

```python
import time

# Measure current performance
start = time.perf_counter()
default_compute_delta_nfr(G)
baseline_time = time.perf_counter() - start

print(f"Baseline: {baseline_time:.3f}s")
```

### Step 2: Enable Profiling

```python
profile = {}
default_compute_delta_nfr(G, profile=profile)

# Identify bottlenecks
for stage, duration in sorted(profile.items()):
    if stage != "dnfr_path":
        print(f"{stage}: {duration:.3f}s")
```

### Step 3: Apply Optimizations

```python
# Try different strategies
strategies = [
    ("default", {}),
    ("force_dense", {"dnfr_force_dense": True}),
    ("chunked", {"DNFR_CHUNK_SIZE": 500}),
]

for name, config in strategies:
    G_test = G.copy()
    G_test.graph.update(config)
    
    start = time.perf_counter()
    backend.compute_delta_nfr(G_test)
    elapsed = time.perf_counter() - start
    
    print(f"{name}: {elapsed:.3f}s")
```

### Step 4: Select Best Backend

```python
from tnfr.backends import available_backends

backends_to_test = available_backends().keys()

for backend_name in backends_to_test:
    try:
        backend = get_backend(backend_name)
        G_test = G.copy()
        
        start = time.perf_counter()
        backend.compute_delta_nfr(G_test)
        elapsed = time.perf_counter() - start
        
        print(f"{backend_name}: {elapsed:.3f}s")
    except Exception as e:
        print(f"{backend_name}: Not available ({e})")
```

## Common Issues and Solutions

### Issue 1: "NumPy not found" Warning

**Problem:**
```
UserWarning: NumPy not available, using Python fallback
```

**Solution:**
NumPy is a core dependency and should be installed. Verify:
```bash
pip install numpy
python -c "import numpy; print(numpy.__version__)"
```

### Issue 2: Slow First Run with JAX

**Problem:**
First call to JAX backend is very slow.

**Solution:**
This is expected - JAX compiles on first run:
```python
backend = get_backend("jax")

# First run: slow (compilation)
backend.compute_delta_nfr(G)  # ~2s

# Subsequent runs: fast
backend.compute_delta_nfr(G)  # ~0.1s
```

### Issue 3: Out of Memory

**Problem:**
`MemoryError` on large graphs.

**Solution:**
```python
# Reduce cache size
backend.compute_delta_nfr(G, cache_size=1)

# Use chunked processing
G.graph["DNFR_CHUNK_SIZE"] = 500
G.graph["SI_CHUNK_SIZE"] = 500

# Use sparse storage for coherence
G.graph["COHERENCE"]["store_mode"] = "sparse"
G.graph["COHERENCE"]["threshold"] = 0.2
```

### Issue 4: Results Don't Match

**Problem:**
Different backends give slightly different results.

**Solution:**
All backends maintain TNFR semantic fidelity, but floating-point differences are expected:
```python
import numpy as np

# Results should match within tolerance
backend1 = get_backend("numpy")
backend2 = get_backend("jax")

G1 = G.copy()
G2 = G.copy()

backend1.compute_delta_nfr(G1)
backend2.compute_delta_nfr(G2)

dnfr1 = [G1.nodes[n].get("delta_nfr", 0) for n in G1.nodes()]
dnfr2 = [G2.nodes[n].get("delta_nfr", 0) for n in G2.nodes()]

assert np.allclose(dnfr1, dnfr2, rtol=1e-6)  # Within tolerance
```

## Environment-Specific Recommendations

### Development/Testing
```python
# Use NumPy backend (default)
# - Stable
# - Predictable
# - Good error messages
set_backend("numpy")
```

### Production (CPU-only)
```python
# Use OptimizedNumPy backend
# - Best CPU performance
# - No extra dependencies
set_backend("optimized")
```

### Production (GPU available)
```python
# Use JAX backend
# - Best overall performance
# - GPU acceleration
# - JIT compilation
set_backend("jax")
```

### Memory-Constrained
```python
# Configure for low memory
G.graph["DNFR_CHUNK_SIZE"] = 100
G.graph["SI_CHUNK_SIZE"] = 100
backend = get_backend("numpy")
backend.compute_delta_nfr(G, cache_size=1)
```

## Verification Checklist

After migration, verify:

- [ ] **Correctness**: Results match baseline (within tolerance)
- [ ] **Performance**: Speedup achieved vs baseline
- [ ] **Memory**: Peak memory usage acceptable
- [ ] **Determinism**: Results reproducible with same seed
- [ ] **Profiling**: Bottlenecks identified and addressed

## Example: Complete Migration

**Before (old code):**
```python
import networkx as nx

# Create graph
G = nx.barabasi_albert_graph(1000, 5)

# Initialize manually
for node in G.nodes():
    G.nodes[node]["phase"] = random.random() * 2 * math.pi
    G.nodes[node]["nu_f"] = random.random()
    G.nodes[node]["epi"] = random.random()

# Custom ΔNFR computation
for node in G.nodes():
    neighbors = list(G.neighbors(node))
    if neighbors:
        # Manual phase mean
        # Manual gradient computation
        # ...
        pass

# Manual Si computation
# ...
```

**After (optimized):**
```python
import networkx as nx
from tnfr.backends import get_backend
from tnfr.initialization import init_node_attrs

# Create and initialize graph
G = nx.barabasi_albert_graph(1000, 5)
init_node_attrs(G)

# Configure
G.graph["DNFR_WEIGHTS"] = {"phase": 0.4, "epi": 0.3, "vf": 0.2, "topo": 0.1}
G.graph["SI_WEIGHTS"] = {"alpha": 0.4, "beta": 0.3, "gamma": 0.3}

# Use optimized backend
backend = get_backend("optimized")

# Compute (vectorized, cached, parallelized automatically)
backend.compute_delta_nfr(G)
backend.compute_si(G)
```

**Benefits:**
- ✅ 10-15x faster
- ✅ Less code to maintain
- ✅ Automatic optimization
- ✅ Better error handling
- ✅ Profiling built-in
- ✅ GPU-ready

## Next Steps

1. **Read** [OPTIMIZATION_GUIDE.md](./OPTIMIZATION_GUIDE.md) for detailed backend information
2. **Run** [examples/optimization_quickstart.py](../examples/optimization_quickstart.py) for practical examples
3. **Benchmark** your specific workload with [examples/backend_performance_comparison.py](../examples/backend_performance_comparison.py)
4. **Configure** your graphs for optimal performance
5. **Monitor** with profiling to identify remaining bottlenecks

## Support

- **Documentation**: [OPTIMIZATION_GUIDE.md](./OPTIMIZATION_GUIDE.md)
- **Examples**: [examples/optimization_quickstart.py](../examples/optimization_quickstart.py), [examples/backend_performance_comparison.py](../examples/backend_performance_comparison.py)
- **Backend API**: [src/tnfr/backends](../src/tnfr/backends/)
- **Issues**: Report problems or questions on GitHub

## Summary

The TNFR engine already includes comprehensive optimization infrastructure. Migration is straightforward:

1. ✅ **Default code is already optimized** (NumPy vectorization)
2. ✅ **Switch backends for more speedup** (JAX, Torch)
3. ✅ **Configure for your topology** (sparse/dense, chunked)
4. ✅ **Profile to verify improvements**

No major code changes required - the infrastructure is already there!
