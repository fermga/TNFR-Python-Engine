# Phase 2: Cache Consolidation - Migration Guide

**Date**: 2025-11-05  
**Status**: Complete  
**Deprecation Timeline**: 6 months  

---

## Overview

Phase 2 consolidates all TNFR hierarchical caching functionality from `tnfr.caching/` into `tnfr.utils.cache`, with `tnfr.cache` as the canonical public API.

**Key Changes**:
- Single source of truth for all caching functionality
- Fixes §3.8 (Controlled Determinism) violations
- 89% reduction in `caching/` package complexity
- Full backward compatibility with deprecation warnings

---

## Quick Migration

### Old Code (Deprecated)
```python
from tnfr.caching import (
    TNFRHierarchicalCache,
    CacheLevel,
    CacheEntry,
    cache_tnfr_computation,
    invalidate_function_cache,
    GraphChangeTracker,
    track_node_property_update,
    PersistentTNFRCache,
)
```

### New Code (Recommended)
```python
from tnfr.cache import (
    TNFRHierarchicalCache,
    CacheLevel,
    CacheEntry,
    cache_tnfr_computation,
    invalidate_function_cache,
    GraphChangeTracker,
    track_node_property_update,
    PersistentTNFRCache,
)
```

### Direct Import (Advanced)
```python
# For internal code that needs direct access
from tnfr.utils.cache import (
    TNFRHierarchicalCache,
    CacheLevel,
    # ... etc
)
```

---

## Module-by-Module Migration

### 1. Hierarchical Cache

**Old**:
```python
from tnfr.caching import TNFRHierarchicalCache, CacheLevel, CacheEntry
```

**New**:
```python
from tnfr.cache import TNFRHierarchicalCache, CacheLevel, CacheEntry
```

**No API changes** - All methods and parameters remain identical.

---

### 2. Cache Decorators

**Old**:
```python
from tnfr.caching import cache_tnfr_computation, invalidate_function_cache
```

**New**:
```python
from tnfr.cache import cache_tnfr_computation, invalidate_function_cache
```

**No API changes** - Decorator syntax and behavior unchanged.

---

### 3. Invalidation Tracking

**Old**:
```python
from tnfr.caching import GraphChangeTracker, track_node_property_update
```

**New**:
```python
from tnfr.cache import GraphChangeTracker, track_node_property_update
```

**No API changes** - Tracking hooks work identically.

---

### 4. Persistent Cache

**Old**:
```python
from tnfr.caching import PersistentTNFRCache
```

**New**:
```python
from tnfr.cache import PersistentTNFRCache
```

**No API changes** - Persistence behavior unchanged.

---

## What's Changed

### File Structure

**Before**:
```
src/tnfr/
├── cache.py (180 lines, aggregator)
├── utils/
│   └── cache.py (2,839 lines, core)
└── caching/
    ├── __init__.py (79 lines)
    ├── hierarchical_cache.py (618 lines)
    ├── decorators.py (219 lines)
    ├── invalidation.py (214 lines)
    └── persistence.py (267 lines)
```

**After**:
```
src/tnfr/
├── cache.py (180 lines, canonical public API)
├── utils/
│   └── cache.py (4,130 lines, unified implementation)
└── caching/ (compatibility shims, deprecated)
    ├── __init__.py (105 lines, shim + warnings)
    ├── hierarchical_cache.py (32 lines, shim)
    ├── decorators.py (39 lines, shim)
    ├── invalidation.py (27 lines, shim)
    └── persistence.py (26 lines, shim)
```

### Code Reduction

- **Before**: 1,397 lines of implementation in `caching/`
- **After**: 229 lines of shims in `caching/`
- **Reduction**: 89% (1,168 lines consolidated)

---

## Backward Compatibility

All old imports **continue to work** with deprecation warnings:

```python
>>> from tnfr.caching import TNFRHierarchicalCache
<stdin>:1: DeprecationWarning: The 'tnfr.caching' package is deprecated and will be 
removed in a future version. Please use 'tnfr.cache' instead. All functionality is 
available through tnfr.cache with identical APIs. See migration guide in documentation.
```

**Timeline**:
- **Now - 6 months**: Deprecation warnings issued
- **After 6 months**: `tnfr.caching/` package removed

---

## Benefits

### 1. Fixed §3.8 Controlled Determinism

**Before**: Multiple cache implementations could produce inconsistent results.

**After**: Single canonical cache ensures deterministic behavior across all operations.

### 2. Simplified Architecture

**Before**: 7 import paths for cache functionality
```python
from tnfr.cache import TNFRHierarchicalCache
from tnfr.caching import TNFRHierarchicalCache
from tnfr.caching.hierarchical_cache import TNFRHierarchicalCache
from tnfr.utils.cache import CacheManager
# ... and 3 more variations
```

**After**: 2 canonical paths
```python
from tnfr.cache import TNFRHierarchicalCache  # Public API
from tnfr.utils.cache import TNFRHierarchicalCache  # Direct access
```

### 3. Easier Maintenance

- Single source of truth for all caching logic
- No duplicate code to keep in sync
- Simpler testing and debugging

---

## Testing

All 60 existing cache tests pass with 100% backward compatibility:

```bash
$ pytest tests/unit/caching/ -q
............................................................             [100%]
60 passed, 4 warnings in 0.11s
```

Warnings are expected deprecation notices for old imports.

---

## Troubleshooting

### Q: My code imports from `tnfr.caching` and I see warnings. Do I need to change it immediately?

**A**: No. Your code will continue to work for the next 6 months. The warnings are just to notify you of the upcoming change. You can migrate at your convenience.

### Q: What if I ignore the deprecation warnings?

**A**: Your code will break in 6 months when `tnfr.caching/` is removed. We recommend migrating within the next few months.

### Q: Are there any API changes?

**A**: No. All APIs remain 100% identical. Only the import paths change.

### Q: How do I suppress the deprecation warnings during migration?

**A**: 
```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tnfr.caching')
```

But we recommend addressing them instead.

---

## Support

For questions or issues during migration:

1. Check this guide first
2. Review the examples in `examples/` directory
3. Open an issue on GitHub if you encounter problems

---

## Summary

**Action Required**: Update import statements from `tnfr.caching` to `tnfr.cache`  
**Timeline**: 6 months  
**Difficulty**: Low (simple find-and-replace)  
**Risk**: None (100% backward compatible)  
**Benefit**: Improved architecture, fixed TNFR invariant violations
