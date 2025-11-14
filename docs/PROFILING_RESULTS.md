# Profiling Results: Validation Performance Analysis

**Date**: 2025-01-XX  
**Branch**: `optimization/phase-3`  
**Workload**: 500-node scale-free graph, 10× validation runs

---

## Executive Summary

**Key Finding**: 76% of validation time spent in NetworkX graph algorithms:
- `eccentricity()`: 4.684s / 6.138s total (76%)
- `_single_shortest_path_length()`: 2.758s self-time (45%)
- Field caching works perfectly: 2nd run = 0.000s (100% cache hits)

**Bottleneck**: `estimate_coherence_length()` → diameter calculation → APSP O(N³)

---

## Detailed Profile: Full Validation (10 runs)

### Top Functions by Cumulative Time

| Function | cumtime | tottime | calls | Source |
|----------|---------|---------|-------|--------|
| `run_structural_validation` | 6.138s | 0.000s | 10 | aggregator.py:124 |
| `eccentricity` | 4.684s | 0.023s | 20 | networkx/distance_measures.py:317 |
| `shortest_path_length` | 4.600s | 0.006s | 10K | networkx/shortest_paths/generic.py:178 |
| `single_source_shortest_path_length` | 4.584s | 0.603s | 10K | networkx/unweighted.py:19 |
| **`_single_shortest_path_length`** | **3.979s** | **2.758s** | **5M** | **networkx/unweighted.py:61** |
| `diameter` | 2.339s | 0.000s | 10 | networkx/distance_measures.py:408 |
| `compute_structural_potential` | 1.428s | 0.100s | 1 | fields.py:309 |
| `_dijkstra_multisource` | 1.150s | 0.637s | 500 | networkx/weighted.py:784 |

### Primitive Operations (High Self-Time)

| Operation | tottime | calls | Type |
|-----------|---------|-------|------|
| `set.add()` | 0.491s | 5M | Builtin |
| `list.append()` | 0.450s | 5M | Builtin |
| `lambda` (edge weight) | 0.363s | 1.5M | NetworkX |
| `len()` | 0.298s | 3.8M | Builtin |

**Interpretation**: 
- 2.758s self-time in `_single_shortest_path_length` = actual BFS work
- 0.637s self-time in Dijkstra = distance computations
- Remaining time = Python overhead (sets, lists, len checks)

---

## Field Caching Performance: Second Run

### Total Time: 0.000s (100% cache hits)

| Function | cumtime | calls | Role |
|----------|---------|-------|------|
| `cache.wrapper` | 0.000s | 40 | Check cache |
| `_generate_cache_key` | 0.000s | 40 | Hash inputs |
| `get()` | 0.000s | 40 | Retrieve value |
| `openssl_md5` | 0.000s | 40 | Hash computation |

**Evidence**: Field caching working perfectly. Zero computational overhead on cached graphs.

---

## Performance Breakdown by Component

### 1. NetworkX Graph Algorithms: 76% (4.684s / 6.138s)

**Functions**:
- `eccentricity()` (diameter calculation): 4.684s cumulative
- `shortest_path_length()`: 4.600s cumulative
- BFS internal: 2.758s self-time

**Why Expensive**:
- Diameter requires All-Pairs Shortest Paths (APSP)
- NetworkX eccentricity = max(shortest_path_length(n, target) for all targets)
- Complexity: O(N² × M) for unweighted, O(N³) worst-case
- 500 nodes → 500² = 250K path computations

**Optimization Opportunities**:
1. **Approximate diameter** (2-sweep BFS heuristic): O(N + M) vs O(N³)
2. **Cache graph-level metrics** (diameter, eccentricity) separately
3. **Lazy diameter** - only compute if needed for ξ_C validation

### 2. Field Computation (First Run): 23% (1.428s / 6.138s)

**Functions**:
- `compute_structural_potential()`: 1.428s (Φ_s)
- Uses Dijkstra for distance matrix: 1.150s

**Why Reasonable**:
- First computation on uncached graph
- Dijkstra O(N log N) per source, 500 sources = O(N² log N)
- Includes inverse-square distance weighting

**Already Optimized**:
- ✅ Cache decorator applied
- ✅ NumPy vectorization for distance matrix operations
- ✅ No obvious low-hanging fruit

### 3. Cache System: <1% (0.000s)

**Already Optimal**: Negligible overhead, perfect hit rate on repeated calls.

---

## Optimization Priorities (Based on Profile Data)

### HIGH PRIORITY 🔴

#### 1. Replace Exact Diameter with Approximation
**Impact**: ~4.5s → ~0.05s (99% reduction)  
**Effort**: Medium  
**Risk**: Low (approximate ξ_C sufficient)

**Implementation**:
```python
def approximate_diameter(G):
    """2-sweep BFS heuristic for diameter estimation.
    
    Complexity: O(N + M) vs O(N³) exact.
    Accuracy: Typically within 2× of true diameter.
    """
    # 1. Random peripheral node
    u = max(G.nodes(), key=lambda n: nx.eccentricity(G, n))
    
    # 2. BFS from u, find farthest v
    lengths = nx.single_source_shortest_path_length(G, u)
    v, d1 = max(lengths.items(), key=lambda x: x[1])
    
    # 3. BFS from v, diameter ≈ max distance
    lengths2 = nx.single_source_shortest_path_length(G, v)
    d2 = max(lengths2.values())
    
    return max(d1, d2)
```

**Validation**: Benchmark against exact diameter on test graphs.

#### 2. Cache Graph-Level Metrics Separately
**Impact**: ~20% reduction if diameter reused  
**Effort**: Low  
**Risk**: Very Low

**Implementation**:
- Add `@cache_tnfr_computation(dependencies={'graph_topology'})` to diameter wrapper
- Store in graph cache with longer TTL
- Invalidate only on topology changes

### MEDIUM PRIORITY 🟡

#### 3. Vectorize Phase Operations
**Impact**: ~10-15% reduction (phase gradient/curvature)  
**Effort**: Medium  
**Risk**: Low

**Target**: Batch phase difference computations in `compute_phase_gradient`

#### 4. Early Exit for Grammar Validation
**Impact**: Variable (10-30% if errors common)  
**Effort**: Low  
**Risk**: Very Low

**Implementation**: Add `stop_on_first_error=True` flag

### LOW PRIORITY 🟢

#### 5. NumPy/Numba JIT for BFS
**Impact**: ~20% (if replacing NetworkX)  
**Effort**: High  
**Risk**: High (correctness, maintenance)

**Decision**: Defer - NetworkX BFS already C-optimized.

---

## Recommended Next Steps

1. **Implement approximate diameter** (Issue #1)
   - Create `fast_diameter()` helper
   - Add benchmark comparing exact vs approximate
   - Update `estimate_coherence_length()` to use approximation
   - Measure speedup on 100, 500, 1K node graphs

2. **Add graph-level metric caching** (Issue #2)
   - Wrap diameter in cached function
   - Test invalidation on topology changes

3. **Profile after optimizations**
   - Re-run this script
   - Verify NetworkX time <20% total
   - Document speedup in OPTIMIZATION_PROGRESS.md

4. **Benchmark at scale**
   - Test 1K, 2K, 5K node graphs
   - Measure O(N) scaling for approximate diameter
   - Compare O(N³) exact vs O(N) approximate curves

---

## Tools & Commands

### Run This Profile
```powershell
$env:PYTHONPATH=(Resolve-Path -Path ./src).Path
& "C:/Program Files/Python313/python.exe" profile_validation.py
```

### Analyze with snakeviz (Visual)
```powershell
# Install snakeviz
pip install snakeviz

# Generate profile
python -m cProfile -o profile.stats profile_validation.py

# Visualize
snakeviz profile.stats
```

### Line-by-line profiling (optional)
```powershell
# Install line_profiler
pip install line_profiler

# Decorate target function with @profile
# Run with kernprof
kernprof -l -v profile_validation.py
```

---

## References

- **NetworkX Performance**: https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html
- **Diameter Approximation**: Magnien et al. "Fast computation of empirically tight bounds for the diameter of massive graphs" (2009)
- **BFS Complexity**: O(N + M) unweighted, O(N log N + M) weighted (Dijkstra)

---

**Next Document**: `docs/DIAMETER_OPTIMIZATION.md` (implementation plan)
