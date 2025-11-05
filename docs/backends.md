# TNFR Backend System

The TNFR engine provides a flexible backend system for high-performance computation of ΔNFR and Si metrics. This document explains how to use and extend the backend system.

## Overview

The backend system allows you to choose different numerical libraries for TNFR computations while maintaining semantic fidelity to the canonical nodal equation:

```
∂EPI/∂t = νf · ΔNFR(t)
```

All backends preserve TNFR structural invariants:
1. ΔNFR semantics (sign/magnitude modulate reorganization)
2. Phase verification (explicit synchrony checking)
3. Operator closure (valid TNFR state transformations)
4. Determinism (reproducible with fixed seeds)
5. Si stability (correlates with network coherence)

## Available Backends

### NumPy Backend (Default, Stable)

**Status**: Production-ready, thoroughly tested

The NumPy backend provides vectorized implementations using `numpy` arrays and operations:

- **Vectorization**: Neighbor accumulation via `np.bincount`, matrix operations
- **Performance**: 1.3-1.6x faster than pure Python fallback
- **Memory**: Strategic buffer caching to minimize allocations
- **Scalability**: Efficient up to 10,000+ nodes
- **Strategy**: Automatic sparse/dense selection based on graph density

**Requirements**: `numpy>=1.24`

**Characteristics**:
- `supports_gpu`: False (CPU-only)
- `supports_jit`: False (no JIT compilation)

### JAX Backend (Experimental)

**Status**: Experimental - API may change

The JAX backend provides foundation for JIT-compiled, GPU-accelerated computations:

- **JIT Compilation**: Future support for `@jax.jit` optimizations
- **GPU Acceleration**: Can utilize CUDA/ROCm when available
- **Autodiff**: Automatic differentiation for sensitivity analysis
- **XLA**: Compiler optimizations via XLA backend

**Requirements**: `jax>=0.4`, `jaxlib`

**Characteristics**:
- `supports_gpu`: True
- `supports_jit`: True

**Current Implementation**: Delegates to NumPy backend while providing interface for future JIT implementations.

### PyTorch Backend (Experimental)

**Status**: Experimental - API may change

The PyTorch backend provides foundation for GPU-accelerated tensor operations:

- **GPU Support**: CUDA/ROCm acceleration when available
- **Tensor Operations**: Optimized torch tensor operations
- **Mixed Precision**: Future FP16/BF16 support for memory efficiency
- **PyG Integration**: Potential integration with PyTorch Geometric

**Requirements**: `torch>=2.1`

**Characteristics**:
- `supports_gpu`: True  
- `supports_jit`: False (TorchScript not yet integrated)

**Current Implementation**: Delegates to NumPy backend while providing interface for future GPU implementations.

## Usage

### Basic Usage

```python
from tnfr.backends import get_backend
import networkx as nx

# Create a graph
G = nx.erdos_renyi_graph(100, 0.2)
for node in G.nodes():
    G.nodes[node]['phase'] = 0.0
    G.nodes[node]['nu_f'] = 1.0
    G.nodes[node]['epi'] = 0.5

# Get backend and compute
backend = get_backend("numpy")
backend.compute_delta_nfr(G)
si_values = backend.compute_si(G, inplace=False)
```

### Backend Selection

The backend selection follows this precedence order:

1. **Explicit name** in `get_backend(name)`
2. **`set_backend(name)`** call
3. **`TNFR_BACKEND` environment variable**
4. **Default**: "numpy"

#### Example: Explicit Selection

```python
from tnfr.backends import get_backend

# Always use NumPy backend
backend = get_backend("numpy")
```

#### Example: Environment Variable

```bash
export TNFR_BACKEND=numpy
python your_simulation.py
```

```python
from tnfr.backends import get_backend

# Uses backend from TNFR_BACKEND env var
backend = get_backend()
```

#### Example: Set Default

```python
from tnfr.backends import set_backend, get_backend

# Set default for all subsequent get_backend() calls
set_backend("numpy")

backend = get_backend()  # Returns NumPy backend
assert backend.name == "numpy"
```

### Profiling

Backends support profiling to analyze performance:

```python
from tnfr.backends import get_backend
import networkx as nx

G = nx.erdos_renyi_graph(100, 0.3)
# ... initialize nodes ...

backend = get_backend("numpy")

# ΔNFR profiling
dnfr_profile = {}
backend.compute_delta_nfr(G, profile=dnfr_profile)

print(f"Path: {dnfr_profile['dnfr_path']}")  # "vectorized" or "fallback"
print(f"Cache rebuild: {dnfr_profile['dnfr_cache_rebuild']:.4f}s")
print(f"Neighbor accumulation: {dnfr_profile['dnfr_neighbor_accumulation']:.4f}s")

# Si profiling
si_profile = {}
backend.compute_si(G, inplace=True, profile=si_profile)

print(f"Path: {si_profile['path']}")  # "vectorized" or "fallback"
print(f"Cache rebuild: {si_profile['cache_rebuild']:.4f}s")
print(f"Phase mean computation: {si_profile['neighbor_phase_mean_bulk']:.4f}s")
```

### Checking Backend Capabilities

```python
from tnfr.backends import get_backend, available_backends

# List all registered backends
backends = available_backends()
print("Available:", list(backends.keys()))

# Check specific backend capabilities
backend = get_backend("numpy")
print(f"Name: {backend.name}")
print(f"Supports GPU: {backend.supports_gpu}")
print(f"Supports JIT: {backend.supports_jit}")
```

## Performance Characteristics

### NumPy Backend Performance

Based on benchmarks with Erdős-Rényi graphs (p=0.2):

| Nodes | ΔNFR (ms) | Si (ms) | Total (ms) |
|-------|-----------|---------|------------|
| 50    | 1.3       | 1.5     | 2.8        |
| 100   | 2.2       | 2.6     | 4.7        |
| 200   | 4.4       | 5.3     | 9.7        |

**Speedup vs. Python fallback**: 1.3-1.6x

### Optimization Strategies

The NumPy backend automatically selects the optimal accumulation strategy:

**Sparse Path** (Density ≤ 0.25):
- Edge-based accumulation using `np.bincount`
- Efficient for graphs with few edges
- Memory: O(edges)

**Dense Path** (Density > 0.25 or forced):
- Adjacency matrix multiplication
- Efficient for dense graphs
- Memory: O(nodes²)

Force dense mode:
```python
G.graph["dnfr_force_dense"] = True
backend.compute_delta_nfr(G)
```

## API Reference

### TNFRBackend (Abstract Base)

Base class for all backend implementations.

**Properties**:
- `name: str` - Backend identifier
- `supports_gpu: bool` - GPU acceleration capability
- `supports_jit: bool` - JIT compilation support

**Methods**:

#### `compute_delta_nfr(graph, *, cache_size=1, n_jobs=None, profile=None)`

Compute ΔNFR for all nodes.

**Parameters**:
- `graph`: TNFRGraph with node attributes (phase, EPI, νf)
- `cache_size`: Max cached configurations (None = unlimited)
- `n_jobs`: Parallel workers (for fallback path)
- `profile`: Dict to collect timing metrics

**Effects**: Writes ΔNFR to `graph.nodes[n]['ΔNFR']`

#### `compute_si(graph, *, inplace=True, n_jobs=None, chunk_size=None, profile=None)`

Compute sense index for all nodes.

**Parameters**:
- `graph`: TNFRGraph with node attributes (νf, ΔNFR, phase)
- `inplace`: Write Si to graph nodes
- `n_jobs`: Parallel workers (for fallback path)
- `chunk_size`: Batch size for chunked processing
- `profile`: Dict to collect timing metrics

**Returns**: Dict or ndarray mapping nodes to Si values

### Backend Management

#### `get_backend(name=None) -> TNFRBackend`

Get backend instance by name.

**Parameters**:
- `name`: Backend name (None = use default resolution)

**Returns**: Backend instance

**Raises**:
- `ValueError`: Unknown backend
- `RuntimeError`: Backend initialization failed

#### `set_backend(name: str) -> None`

Set default backend for subsequent operations.

**Parameters**:
- `name`: Backend name to set as default

**Raises**:
- `ValueError`: Unknown backend name

#### `available_backends() -> Mapping[str, type[TNFRBackend]]`

Get registered backend classes.

**Returns**: Dict mapping backend names to classes

## Extending the Backend System

### Creating a Custom Backend

```python
from tnfr.backends import TNFRBackend, register_backend
from typing import Any, MutableMapping

class CustomBackend(TNFRBackend):
    """Custom TNFR backend implementation."""
    
    @property
    def name(self) -> str:
        return "custom"
    
    @property
    def supports_gpu(self) -> bool:
        return False
    
    @property
    def supports_jit(self) -> bool:
        return False
    
    def compute_delta_nfr(
        self,
        graph,
        *,
        cache_size=1,
        n_jobs=None,
        profile=None,
    ):
        """Your custom ΔNFR implementation."""
        # Must preserve TNFR structural invariants
        # Must write results to graph.nodes[n]['ΔNFR']
        pass
    
    def compute_si(
        self,
        graph,
        *,
        inplace=True,
        n_jobs=None,
        chunk_size=None,
        profile=None,
    ):
        """Your custom Si implementation."""
        # Must preserve TNFR structural invariants
        # Must return dict or array of Si values
        pass

# Register your backend
register_backend("custom", CustomBackend)

# Use it
from tnfr.backends import get_backend
backend = get_backend("custom")
```

### Structural Invariants (CRITICAL)

All backend implementations **MUST** preserve these TNFR invariants:

1. **ΔNFR semantics**: Sign and magnitude must modulate reorganization rate correctly
2. **Phase verification**: Coupling requires explicit phase synchrony check
3. **Operator closure**: All transformations must map to valid TNFR states
4. **Determinism**: Computations must be reproducible with fixed graph topology
5. **Si stability**: Sense index must correlate with network coherence

**Failure to preserve these invariants breaks TNFR semantic fidelity.**

## Troubleshooting

### Backend Not Available

```python
from tnfr.backends import get_backend

try:
    backend = get_backend("jax")
except RuntimeError as e:
    print(f"Backend unavailable: {e}")
    # Fall back to NumPy
    backend = get_backend("numpy")
```

### Performance Issues

1. **Check vectorization is active**:
   ```python
   profile = {}
   backend.compute_delta_nfr(G, profile=profile)
   assert profile["dnfr_path"] == "vectorized"
   ```

2. **Try dense mode for dense graphs**:
   ```python
   G.graph["dnfr_force_dense"] = True
   ```

3. **Adjust chunk size for Si**:
   ```python
   backend.compute_si(G, chunk_size=100)
   ```

### Memory Issues

For large graphs (>10,000 nodes), consider:

1. **Disable caching**:
   ```python
   backend.compute_delta_nfr(G, cache_size=0)
   ```

2. **Use chunked Si computation**:
   ```python
   backend.compute_si(G, chunk_size=500)
   ```

3. **Monitor memory usage**:
   ```python
   import psutil
   process = psutil.Process()
   
   backend.compute_delta_nfr(G)
   memory_mb = process.memory_info().rss / 1024 / 1024
   print(f"Memory: {memory_mb:.1f} MB")
   ```

## Future Roadmap

### Planned Enhancements

1. **JAX Backend**:
   - JIT-compiled ΔNFR computation with `@jax.jit`
   - GPU acceleration via JAX device placement
   - Vectorized operations using `jax.numpy`
   - Target: 10-50x speedup on GPU

2. **PyTorch Backend**:
   - GPU-accelerated tensor operations
   - Sparse tensor support for large graphs
   - Mixed precision (FP16/BF16) for memory efficiency
   - Integration with PyTorch Geometric

3. **Optimization**:
   - Fused kernels for phase dispersion + Si
   - Graph topology caching across iterations
   - SIMD optimizations for NumPy backend

### Contributing

To contribute backend improvements:

1. Preserve all TNFR structural invariants
2. Add comprehensive tests validating semantics
3. Benchmark against NumPy baseline
4. Document performance characteristics
5. Follow AGENTS.md guidelines

## References

- TNFR.pdf - Canonical TNFR theory (in repository root)
- AGENTS.md - Development guidelines (in repository root)
- `tnfr.dynamics.dnfr` module - ΔNFR implementation
- `tnfr.metrics.sense_index` module - Si implementation
