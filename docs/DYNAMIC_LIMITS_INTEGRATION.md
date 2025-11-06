# Integration Guide: Dynamic Canonical Limits

This guide explains how to integrate dynamic canonical limits into the existing TNFR codebase.

## Overview

Dynamic limits have been implemented but are not yet integrated into the main validation and configuration system. This guide provides recipes for integration at different levels.

## Quick Start: Using Dynamic Limits

### Basic Usage

```python
import networkx as nx
from tnfr.dynamics import compute_dynamic_limits

# Your TNFR network
G = nx.Graph()
# ... add nodes with νf, theta, EPI, Si, ΔNFR, dEPI_dt ...

# Compute dynamic limits
limits = compute_dynamic_limits(G)

print(f"EPI limit: {limits.epi_max_effective}")
print(f"νf limit: {limits.vf_max_effective} Hz_str")
print(f"Network coherence: {limits.coherence}")
```

### With Custom Configuration

```python
from tnfr.dynamics import compute_dynamic_limits, DynamicLimitsConfig

# Conservative configuration
conservative = DynamicLimitsConfig(
    alpha=0.3,
    beta=0.2,
    max_expansion_factor=2.0
)

limits = compute_dynamic_limits(G, conservative)
```

### Configuration Presets

```python
# Conservative: Small expansion, suitable for unstable systems
CONSERVATIVE_CONFIG = DynamicLimitsConfig(
    base_epi_max=1.0,
    base_vf_max=10.0,
    alpha=0.3,
    beta=0.2,
    max_expansion_factor=2.0,
    enabled=True
)

# Balanced: Default, good for most cases
BALANCED_CONFIG = DynamicLimitsConfig(
    base_epi_max=1.0,
    base_vf_max=10.0,
    alpha=0.5,
    beta=0.3,
    max_expansion_factor=3.0,
    enabled=True
)

# Exploratory: Large expansion, for highly stable systems
EXPLORATORY_CONFIG = DynamicLimitsConfig(
    base_epi_max=1.0,
    base_vf_max=10.0,
    alpha=0.8,
    beta=0.5,
    max_expansion_factor=5.0,
    enabled=True
)
```

## Integration Points

### 1. Configuration System Integration

Add to `src/tnfr/config/tnfr_config.py`:

```python
from ..dynamics.dynamic_limits import (
    compute_dynamic_limits,
    DynamicLimitsConfig,
)

class TNFRConfig:
    def __init__(
        self,
        defaults: Mapping[str, TNFRConfigValue] | None = None,
        validate_invariants: bool = True,
        use_dynamic_limits: bool = False,  # NEW
        dynamic_limits_config: DynamicLimitsConfig | None = None,  # NEW
    ) -> None:
        self._defaults = defaults or {}
        self._validate_invariants = validate_invariants
        self._use_dynamic_limits = use_dynamic_limits
        self._dynamic_limits_config = dynamic_limits_config
    
    def get_effective_limits(
        self, 
        G: TNFRGraph | None = None
    ) -> tuple[float, float, float, float]:
        """Get effective EPI and νf limits (static or dynamic)."""
        if not self._use_dynamic_limits or G is None:
            # Return static limits
            return (
                self._defaults.get("EPI_MIN", -1.0),
                self._defaults.get("EPI_MAX", 1.0),
                self._defaults.get("VF_MIN", 0.0),
                self._defaults.get("VF_MAX", 10.0),
            )
        
        # Compute dynamic limits
        limits = compute_dynamic_limits(G, self._dynamic_limits_config)
        return (
            -limits.epi_max_effective,  # Symmetric for EPI_MIN
            limits.epi_max_effective,
            0.0,  # VF_MIN always 0
            limits.vf_max_effective,
        )
```

### 2. Validation Integration

Modify `validate_vf_bounds()` and `validate_epi_bounds()`:

```python
def validate_vf_bounds(
    self,
    vf_min: float | None = None,
    vf_max: float | None = None,
    vf: float | None = None,
    G: TNFRGraph | None = None,  # NEW: pass graph for dynamic limits
) -> bool:
    """Validate νf bounds (with optional dynamic limits)."""
    
    if vf_max is None:
        if self._use_dynamic_limits and G is not None:
            limits = compute_dynamic_limits(G, self._dynamic_limits_config)
            vf_max = limits.vf_max_effective
        else:
            vf_max = float(self._defaults.get("VF_MAX", 10.0))
    
    # ... rest of validation logic
```

### 3. Operator Integration

Update operators to check dynamic limits before applying:

```python
# In src/tnfr/operators/emission.py (example)

def apply(self, G: TNFRGraph, node: NodeId, **kwargs) -> None:
    """Apply Emission operator with dynamic limit awareness."""
    
    # Compute current dynamic limits
    from ..dynamics import compute_dynamic_limits
    limits = compute_dynamic_limits(G)
    
    # Get node data
    nd = G.nodes[node]
    current_vf = get_attr(nd, ALIAS_VF, 1.0)
    
    # Boost νf respecting dynamic limit
    boost_factor = self.config.get("AL_boost", 0.05)
    new_vf = current_vf * (1 + boost_factor)
    
    # Clamp to dynamic limit
    new_vf = min(new_vf, limits.vf_max_effective)
    
    set_attr(nd, ALIAS_VF, new_vf)
```

### 4. Adaptation Module Integration

Modify `src/tnfr/dynamics/adaptation.py`:

```python
def adapt_vf_by_coherence(
    G: TNFRGraph,
    tau: int | None = None,
    mu: float | None = None,
    n_jobs: int | None = None,
    use_dynamic_limits: bool = True,  # NEW
) -> None:
    """Adapt νf with dynamic limit awareness."""
    
    # ... existing setup code ...
    
    # Get limits (static or dynamic)
    if use_dynamic_limits:
        from .dynamic_limits import compute_dynamic_limits
        limits = compute_dynamic_limits(G)
        vf_min = 0.0
        vf_max = limits.vf_max_effective
    else:
        vf_min = float(get_graph_param(G, "VF_MIN"))
        vf_max = float(get_graph_param(G, "VF_MAX"))
    
    # ... rest of adaptation logic using vf_min, vf_max ...
```

### 5. Runtime Integration

Update `src/tnfr/dynamics/runtime.py`:

```python
def _update_nodes(
    G: TNFRGraph,
    dt: float,
    t: float,
    use_dynamic_limits: bool = False,  # NEW
) -> None:
    """Update nodes with dynamic limit clamping."""
    
    # ... existing update logic ...
    
    # Apply clamping
    if use_dynamic_limits:
        from .dynamic_limits import compute_dynamic_limits
        limits = compute_dynamic_limits(G)
        epi_max = limits.epi_max_effective
        vf_max = limits.vf_max_effective
    else:
        epi_max = get_graph_param(G, "EPI_MAX", 1.0)
        vf_max = get_graph_param(G, "VF_MAX", 10.0)
    
    for node in G.nodes:
        nd = G.nodes[node]
        
        # Clamp EPI
        epi = get_attr(nd, ALIAS_EPI, 0.0)
        epi = max(-epi_max, min(epi, epi_max))
        set_attr(nd, ALIAS_EPI, epi)
        
        # Clamp νf
        vf = get_attr(nd, ALIAS_VF, 1.0)
        vf = max(0.0, min(vf, vf_max))
        set_attr(nd, ALIAS_VF, vf)
```

## Migration Strategy

### Phase 1: Opt-in (Recommended for initial release)

Add dynamic limits as an **optional feature**:

```python
# In graph setup
G.graph["use_dynamic_limits"] = True  # Opt-in
G.graph["dynamic_limits_config"] = {
    "alpha": 0.5,
    "beta": 0.3,
    "max_expansion_factor": 3.0,
}
```

### Phase 2: Validation and Testing

1. Run existing test suite with dynamic limits enabled
2. Add comparison tests (static vs dynamic)
3. Validate that TNFR invariants are preserved
4. Monitor numerical stability

### Phase 3: Default Enable (Future)

Once validated, make dynamic limits the **default**:

```python
# In CoreDefaults
USE_DYNAMIC_LIMITS: bool = True  # Changed from False
DYNAMIC_LIMITS_ALPHA: float = 0.5
DYNAMIC_LIMITS_BETA: float = 0.3
DYNAMIC_LIMITS_MAX_EXPANSION: float = 3.0
```

## Performance Considerations

### Computation Cost

Dynamic limits computation requires:
- 1x `compute_coherence()` call
- 1x average over Si values
- 1x `kuramoto_order()` call

**Cost**: O(N) where N = number of nodes

**Optimization**: Cache limits for multiple operations:

```python
# Compute once per step
limits = compute_dynamic_limits(G)
G.graph["_cached_dynamic_limits"] = limits

# Use cached value in operators
limits = G.graph.get("_cached_dynamic_limits")
if limits is None:
    limits = compute_dynamic_limits(G)
```

### When to Recompute

Recompute limits when:
- Network topology changes (nodes added/removed)
- After significant state evolution (e.g., every N steps)
- Before/after operator sequences
- On explicit request

## Testing Dynamic Limits

### Unit Tests

```python
def test_dynamic_limits_integration():
    """Test that dynamic limits integrate with existing code."""
    G = create_test_network()
    
    # Compute static limits
    static_epi_max = 1.0
    static_vf_max = 10.0
    
    # Compute dynamic limits
    limits = compute_dynamic_limits(G)
    
    # Dynamic should adapt based on coherence
    if limits.coherence > 0.7:
        assert limits.epi_max_effective > static_epi_max
        assert limits.vf_max_effective > static_vf_max
    elif limits.coherence < 0.4:
        assert limits.epi_max_effective <= static_epi_max * 1.2
```

### Integration Tests

```python
def test_operators_respect_dynamic_limits():
    """Test that operators respect dynamic limits."""
    G = create_coherent_network()
    limits = compute_dynamic_limits(G)
    
    # Apply operators
    apply_glyph(G, node, "AL")  # Emission
    
    # Check that νf doesn't exceed dynamic limit
    vf = G.nodes[node]["νf"]
    assert vf <= limits.vf_max_effective
```

## Troubleshooting

### Issue: Limits too restrictive

**Solution**: Increase α or β coefficients

```python
config = DynamicLimitsConfig(alpha=0.8, beta=0.5)
```

### Issue: Limits too permissive

**Solution**: Decrease max_expansion_factor

```python
config = DynamicLimitsConfig(max_expansion_factor=2.0)
```

### Issue: Numerical instability

**Solution**: Use conservative preset or lower max_expansion_factor

```python
config = DynamicLimitsConfig(
    alpha=0.3,
    beta=0.2,
    max_expansion_factor=1.5,
)
```

### Issue: Performance concerns

**Solution**: Cache limits and recompute less frequently

```python
# Recompute only every 10 steps
if step % 10 == 0:
    G.graph["_cached_limits"] = compute_dynamic_limits(G)

limits = G.graph["_cached_limits"]
```

## Best Practices

1. **Start conservative**: Use lower α, β values initially
2. **Monitor coherence**: Track C(t), Si, R_kuramoto over time
3. **Validate invariants**: Ensure TNFR invariants preserved
4. **Cache when possible**: Avoid redundant computations
5. **Test thoroughly**: Compare static vs dynamic behavior
6. **Document configuration**: Explain why specific α, β chosen

## Future Enhancements

Potential extensions to dynamic limits:

1. **Per-node limits**: Individual limits based on local coherence
2. **Temporal smoothing**: Add inertia to limit changes
3. **ΔNFR limits**: Apply similar approach to reorganization gradients
4. **Adaptive coefficients**: Learn α, β from network evolution
5. **Multi-scale limits**: Different limits at different hierarchical levels

## References

- Implementation: `src/tnfr/dynamics/dynamic_limits.py`
- Tests: `tests/unit/dynamics/test_dynamic_limits.py`
- Example: `examples/dynamic_limits_demo.py`
- Research: `docs/DYNAMIC_LIMITS_RESEARCH.md`
- Issue: fermga/TNFR-Python-Engine#2624
