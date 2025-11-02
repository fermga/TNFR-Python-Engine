# Utility Module Migration Guide

## Overview

This guide documents the canonical locations for utility functions in the TNFR engine. As of the current version, all generic helper functions are centralized in `tnfr.utils`.

## Current Structure (Canonical)

All utilities are properly organized with no redundancy:

```python
# Numeric helpers
from tnfr.utils import clamp, clamp01, angle_diff, kahan_sum_nd

# Cache infrastructure
from tnfr.utils import CacheManager, cached_node_list, edge_version_cache

# Data normalization
from tnfr.utils import normalize_weights, convert_value, ensure_collection

# IO and parsing
from tnfr.utils import json_dumps, read_structured_file, safe_write

# Graph utilities
from tnfr.utils import get_graph, node_set_checksum

# Import and logging
from tnfr.utils import cached_import, get_logger
```

## Removed Modules

### `tnfr.cache` (Removed)

**Status**: Legacy compatibility shim that raises `ImportError`

**Migration**:
```python
# OLD (raises ImportError)
from tnfr.cache import CacheManager

# NEW (canonical)
from tnfr.utils import CacheManager
from tnfr.utils.cache import CacheManager  # or more explicit
```

## Non-Redundant Modules

These modules are **not redundant** and remain in their current locations:

### `tnfr.callback_utils` (Stays at root)

**Why**: Used by ~10 core modules throughout the codebase. Root location is appropriate for cross-cutting functionality.

```python
# CORRECT - callback utilities at root level
from tnfr.callback_utils import CallbackEvent, callback_manager
```

### `tnfr.cli.utils` (Stays in CLI package)

**Why**: CLI-specific utilities (argument specs). Only used within CLI module, properly scoped.

```python
# CORRECT - CLI utilities in CLI package
from tnfr.cli.utils import spec, _parse_cli_variants
```

## Validation Modules (Not Duplicates)

The validation modules are **not redundant** - each serves a distinct structural purpose:

```python
# CORRECT - these are NOT duplicates
from tnfr.validation.rules import normalized_dnfr  # context-aware wrapper
from tnfr.metrics.common import normalize_dnfr      # generic normalization

# normalized_dnfr provides context-aware defaults
# normalize_dnfr is the low-level primitive
# This is proper architectural layering
```

### Validation Module Purposes

- **`validation/rules.py`** - Grammar validation rules
- **`validation/spectral.py`** - Spectral validation
- **`validation/graph.py`** - Graph structure validation
- **`validation/compatibility.py`** - Structural compatibility checks
- **`validation/window.py`** - Window parameter validation
- **`validation/soft_filters.py`** - Soft filter utilities

## Import Patterns

### Recommended: Import from top-level utils

```python
# Most common - import specific functions
from tnfr.utils import clamp, json_dumps, normalize_weights

# When you need many functions
from tnfr.utils import (
    clamp, clamp01,
    CacheManager, cached_node_list,
    json_dumps, read_structured_file,
)
```

### Alternative: Import submodules

```python
# When working extensively with one category
from tnfr.utils import numeric, cache, data

result = numeric.clamp(value, 0, 1)
manager = cache.CacheManager()
weights = data.normalize_weights(raw_weights, fields)
```

### For Specific Categories

```python
# Numeric operations
from tnfr.utils.numeric import clamp, angle_diff

# Cache infrastructure
from tnfr.utils.cache import CacheManager, EdgeCacheManager

# Data normalization
from tnfr.utils.data import normalize_weights, convert_value

# IO operations
from tnfr.utils.io import json_dumps, read_structured_file
```

## Stability Guarantees

All public functions exported from `tnfr.utils` (listed in `__all__`) constitute the **stable utility API**:

- ✅ Function signatures are stable
- ✅ Behavior is deterministic and documented
- ✅ Thread-safety guarantees are explicit
- ✅ Breaking changes will follow semantic versioning

Functions prefixed with `_` are **internal implementation details** and may change without notice.

## Testing Your Migration

After updating imports, verify with:

```bash
# Run utility tests
python -m pytest tests/unit/structural/test_*_helpers.py
python -m pytest tests/unit/structural/test_cache_*.py
python -m pytest tests/unit/structural/test_json_utils.py

# Run full test suite
python -m pytest tests/unit/structural/
```

## Questions?

See comprehensive documentation in:
- `docs/utils_reference.md` - Complete API reference
- `src/tnfr/utils/__init__.py` - Module docstring with examples

## Summary

**Key Points**:
1. ✅ All generic utilities are in `tnfr.utils` (already consolidated)
2. ✅ No redundant helper modules exist
3. ✅ `callback_utils` and `cli.utils` are appropriately located
4. ✅ Validation modules serve distinct purposes (not duplicates)
5. ✅ Legacy `tnfr.cache` correctly raises ImportError

No code changes needed - the consolidation is complete!
