# Configuration MutableMapping Interface Implementation

## Summary

Updated the TNFR configuration system to properly implement and document the `MutableMapping` interface for configuration dictionaries, enabling safe runtime mutations while maintaining type safety.

## Changes Made

### 1. Type Annotations

#### `src/tnfr/types.py`
- **Changed**: `TNFRConfigValue` type alias from `Mapping[str, "TNFRConfigValue"]` to `MutableMapping[str, "TNFRConfigValue"]`
- **Rationale**: Configuration dictionaries need to support mutations like `__setitem__`, `.update()`, and `.pop()` for runtime adjustments
- **Impact**: Type checkers now correctly recognize that config dicts support full dict operations

#### `src/tnfr/types.pyi`
- **Changed**: Synced stub file to match `types.py` with `MutableMapping`
- **Rationale**: Ensures type checking consistency across the codebase

### 2. Enhanced Documentation

#### `src/tnfr/constants/__init__.py`
Enhanced docstrings for core configuration functions:

**`get_param()`**:
- Added comprehensive parameter/return/raises documentation
- Documented the MutableMapping interface support
- Provided usage examples showing `.get()`, `__setitem__`, and `.update()` operations

**`inject_defaults()`**:
- Documented deep-copy behavior for configuration isolation
- Explained override parameter behavior
- Clarified parameter types and defaults

**`merge_overrides()`**:
- Documented deep-copy behavior for safe mutations
- Added parameter/raises documentation
- Provided usage examples

### 3. Comprehensive Test Suite

#### `tests/unit/config/test_mutablemapping_interface.py`
Created 17 new tests validating the MutableMapping interface:

1. **Basic Interface Tests**:
   - `test_get_param_returns_mutable_dict_for_nested_configs`: Verifies MutableMapping instance check
   - `test_config_dict_supports_get_method`: Tests `.get()` with defaults
   - `test_config_dict_supports_setitem`: Tests `[key] = value` mutations
   - `test_config_dict_supports_update`: Tests `.update()` batch mutations
   - `test_config_dict_supports_del`: Tests `del` key removal
   - `test_config_dict_supports_pop`: Tests `.pop()` with defaults
   - `test_config_dict_supports_setdefault`: Tests `.setdefault()` behavior

2. **Isolation Tests**:
   - `test_config_mutations_isolated_per_graph`: Ensures mutations don't affect other graphs or DEFAULTS
   - `test_nested_dict_mutations`: Verifies nested dictionary mutations work correctly

3. **Type Safety Tests**:
   - `test_scalar_config_values_remain_immutable`: Confirms scalar values work correctly
   - `test_all_dict_configs_are_mutable_mappings`: Parametrized test for all dict configs

## TNFR Compliance

All changes preserve TNFR structural coherence principles:

1. **Configuration Isolation**: Each graph maintains its own mutable configuration instance through deep-copy, preventing unintended coupling between structures
2. **Canonical Defaults**: `DEFAULTS` remains immutable (MappingProxyType), preserving the canonical parameter set
3. **Structural Traceability**: Configuration mutations are local to each graph, enabling independent structural evolution
4. **Type Safety**: MutableMapping type annotations maintain compile-time structural validation while enabling runtime flexibility

## Validation Results

### Tests Passing
- ✅ All 17 new MutableMapping interface tests pass
- ✅ All existing diagnosis_parallel tests pass (9 tests)
- ✅ All existing config tests pass (27 tests total)
- ✅ All metrics tests pass (198 tests)

### Pre-existing Failures
- ❌ `tests/unit/dynamics/test_canon.py::test_validate_canon_clamps` - Pre-existing BEPI conversion issue (documented in PRE_EXISTING_FAILURES.md)

## Usage Examples

### Safe Configuration Mutation
```python
from tnfr.constants import get_param, inject_defaults
import networkx as nx

G = nx.Graph()
inject_defaults(G)

# Get mutable config
diagnosis_cfg = get_param(G, "DIAGNOSIS")

# Mutate safely - only affects this graph
diagnosis_cfg["compute_symmetry"] = False
diagnosis_cfg.update({"window": 32})

# DEFAULTS remains unchanged
assert DEFAULTS["DIAGNOSIS"]["compute_symmetry"] is True
```

### Type-Safe Access
```python
from collections.abc import MutableMapping

config = get_param(G, "DIAGNOSIS")

# Type checker recognizes MutableMapping operations
if isinstance(config, MutableMapping):
    config["custom_key"] = "custom_value"
    value = config.get("history_key", "default")
```

## Implementation Notes

### Deep Copy Behavior
Configuration dictionaries are deep-copied when injected into graphs to ensure:
- Each graph has independent configuration
- Mutations don't affect other graphs
- DEFAULTS remains immutable

### Immutable DEFAULTS
`DEFAULTS` remains a `MappingProxyType` (read-only) to preserve canonical values. Only individual graph configurations (retrieved via `get_param`) are mutable.

### Backward Compatibility
All changes are backward compatible:
- Existing code using config dicts continues to work
- Type annotations are more permissive (MutableMapping includes all Mapping operations)
- No API surface changes

## Related Files

- Problem statement addressed all 4 points:
  1. ✅ Inspected `src/tnfr/config/` and `src/tnfr/constants/` configuration materialization
  2. ✅ Adjusted annotations to implement `MutableMapping[str, TNFRConfigValue]`
  3. ✅ Reviewed `get_param`/`DEFAULTS` to return compatible types with documented interface
  4. ✅ Adapted tests in `tests/unit/metrics/test_diagnosis_parallel.py` and created comprehensive new tests

## Security Considerations

No security implications:
- Configuration mutations are scoped to individual graphs
- No exposure of sensitive data
- Type safety maintained through proper annotations
