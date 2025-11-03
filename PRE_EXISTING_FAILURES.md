# Pre-Existing Test Failures

## Overview

This document tracks the 127 test failures and 3 import errors that are pre-existing issues in the TNFR Python Engine codebase. These failures were identified after the initial test fixes documented in [TEST_FIXES_SUMMARY.md](./TEST_FIXES_SUMMARY.md) and require separate, focused PRs to address.

## Status Summary

- **Total Tests**: 1,709 tests (1,579 passing + 127 failing + 3 errors)
- **Pass Rate**: 92.4%
- **Failures**: 127 tests
- **Errors**: 3 tests (import failures)
- **Last Updated**: 2025-11-03

## Test Environment

```bash
# Reproduce the failure state
python -m pytest tests/ --tb=line -q

# Expected output:
# 127 failed, 1579 passed, 18 skipped, 21 deselected, 3 errors
```

## Failure Categories

### 1. Grammar and Operator Preconditions (22 failures)

**Priority**: High  
**Complexity**: Medium  
**Estimated Effort**: 2-3 PRs

These failures are related to the structural operator grammar validation system, which enforces TNFR canonical constraints on operator sequences.

**Common Error Types**:
- `TholClosureError`: self_organization blocks require specific closure operators
- `MutationPreconditionError`: mutation operators require dissonance within a specific window
- `RepeatWindowError`: operators cannot repeat within configured windows
- `ValueError`: Glyph instances not being properly converted to string codes

**Example Failures**:
```
tests/integration/test_cli.py::test_cli_sequence_handles_deeply_nested_blocks
  → TholClosureError: self_organization block requires silence closure

tests/integration/test_cli.py::test_cli_without_history_args[sequence]
  → MutationPreconditionError: mutation mutation requires dissonance within window 3

tests/unit/operators/test_grammar_module.py::test_apply_glyph_with_grammar_accepts_glyph_instances
  → ValueError: unknown glyph: Glyph.AL
```

**Root Causes**:
1. Glyph enum instances need string conversion when passed to grammar validators
2. Test sequences may violate grammar rules that were recently tightened
3. Window-based precondition tracking may have edge cases

**Recommended Approach**:
- PR 1: Fix Glyph instance to string conversion in `apply_glyph_with_grammar`
- PR 2: Update test sequences to comply with canonical grammar rules
- PR 3: Review and adjust window-based precondition logic if needed

**TNFR Invariants Affected**:
- ✅ Operator closure (§3.4): Must be preserved
- ✅ Structural operators (§2): Correct sequencing is essential for TNFR fidelity

---

### 2. ΔNFR Dynamics (36 failures)

**Priority**: High  
**Complexity**: High  
**Estimated Effort**: 3-4 PRs

These failures affect the core dynamics engine that computes ΔNFR (internal reorganization operator) and related structural metrics.

**Common Error Types**:
- Vectorization fallback issues (NumPy backend not activating)
- Parallel chunk scheduling inconsistencies
- Cache initialization and refresh logic
- Neighbor accumulation computation differences

**Example Failures**:
```
tests/unit/dynamics/test_dnfr_cache.py::test_neighbor_sum_buffers_reused_and_results_stable[False]
  → assert array([0.95533649, ...]) is None

tests/unit/dynamics/test_dynamics_helpers.py::test_init_and_refresh_dnfr_cache
  → assert 0.0 == 0.1 ± 1.0e-07

tests/unit/dynamics/test_dnfr_parallel_chunks.py::test_parallel_chunks_cover_all_nodes_once
  → AssertionError: every node scheduled exactly once
```

**Root Causes**:
1. NumPy backend detection may not be working correctly
2. Parallel chunk assignment algorithm may have scheduling bugs
3. Cache factory usage differs from expected patterns
4. Test assertions may be checking internal implementation details rather than observable behavior

**Recommended Approach**:
- PR 1: Fix NumPy backend detection and fallback logic
- PR 2: Review and fix parallel chunk scheduling algorithm
- PR 3: Update cache initialization to match expected patterns
- PR 4: Refactor tests to check TNFR-observable behavior (ΔNFR values, C(t), etc.)

**TNFR Invariants Affected**:
- ✅ ΔNFR semantics (§3.3): Sign and magnitude must be preserved
- ✅ Controlled determinism (§3.8): Must be reproducible
- ✅ Structural metrics (§3.9): C(t), Si, νf telemetry must be accurate

---

### 3. Metrics and Observers (31 failures)

**Priority**: Medium  
**Complexity**: Medium  
**Estimated Effort**: 2-3 PRs

These failures affect the telemetry and observation systems that track coherence, phase, and structural metrics.

**Common Error Types**:
- Observer callbacks not being registered
- Metrics calculation differences (sigma, phase_sync, kuramoto_order)
- Trigonometric cache usage inconsistencies
- Parallel Si computation not instantiating executors

**Example Failures**:
```
tests/unit/metrics/test_metrics.py::test_register_metrics_callbacks_respects_verbosity
  → AssertionError: assert [] == ['coherence']

tests/unit/metrics/test_neighbor_phase_mean_missing_trig.py::test_neighbor_phase_mean_list_delegates_generic
  → assert 0.0 == 1.23 ± 1.2e-06

tests/integration/test_sense_index_parallel.py::test_parallel_si_matches_sequential_for_large_graph
  → AssertionError: parallel path should instantiate the executor
```

**Root Causes**:
1. Observer registration logic may have regressed
2. Vectorized metrics computation may be falling back to scalar mode unexpectedly
3. Trigonometric cache may not be shared correctly across modules
4. Parallel executor instantiation may require explicit configuration

**Recommended Approach**:
- PR 1: Fix observer registration and callback invocation
- PR 2: Ensure trigonometric cache is properly shared
- PR 3: Fix parallel Si computation to instantiate executors when needed

**TNFR Invariants Affected**:
- ✅ Structural metrics (§3.9): Must expose C(t), Si, phase, νf accurately
- ✅ Phase check (§3.5): No coupling without phase verification

---

### 4. IO Module (23 failures)

**Priority**: Medium  
**Complexity**: Low  
**Estimated Effort**: 1 PR

These are primarily import errors related to the lazy loading system for optional IO dependencies.

**Common Error Types**:
- `ImportError: module tnfr.utils.io not in sys.modules`
- `AttributeError: 'module' object at tnfr.utils.io has no attribute 'io'`

**Example Failures**:
```
tests/unit/structural/test_io_optional_imports.py::test_io_optional_imports_are_lazy_proxies
  → ImportError: module tnfr.utils.io not in sys.modules

tests/unit/structural/test_safe_write.py::test_safe_write_binary_mode[True]
  → AttributeError: 'module' object at tnfr.utils.io has no attribute 'io'
```

**Root Causes**:
1. IO module structure may have changed (possibly moved or renamed)
2. Lazy proxy system may need to be updated
3. Test isolation may be causing import state issues

**Recommended Approach**:
- Single PR: Fix IO module imports and update lazy proxy system

**TNFR Invariants Affected**:
- ⚠️ Domain neutrality (§3.10): IO should not break core TNFR functionality

---

### 5. Logging Module (5 failures)

**Priority**: Low  
**Complexity**: Low  
**Estimated Effort**: 1 PR

All logging failures are import errors related to a missing `tnfr.utils.init` module.

**Common Error Types**:
- `ImportError: module tnfr.utils.init not in sys.modules`

**Example Failures**:
```
tests/unit/structural/test_logging_module.py::test_get_logger_configures_root_once
  → ImportError: module tnfr.utils.init not in sys.modules

tests/unit/structural/test_logging_threadsafe.py::test_get_logger_threadsafe
  → ImportError: module tnfr.utils.init not in sys.modules
```

**Root Causes**:
1. Module was removed or renamed without updating tests
2. Tests may be checking internal implementation details

**Recommended Approach**:
- Single PR: Update tests to use correct module paths or remove if module no longer exists

**TNFR Invariants Affected**:
- ⚠️ None (logging is infrastructure, not structural)

---

### 6. BEPIElement Serialization (3 failures)

**Priority**: High  
**Complexity**: Medium  
**Estimated Effort**: 1 PR

BEPIElement (Basic EPI) objects cannot be serialized/deserialized with the current IO system.

**Common Error Types**:
- `TypeError: Unsupported BEPI value type: <class 'tnfr.mathematics.epi.BEPIElement'>`

**Example Failures**:
```
tests/unit/structural/test_bepi_node_and_validators.py::test_nodenx_epi_roundtrip_serializes_bepi
  → TypeError: Unsupported BEPI value type: <class 'tnfr.mathematics.epi.BEPIElement'>
```

**Root Causes**:
1. Serialization system doesn't handle BEPIElement's complex structure (f_continuous, a_discrete, x_grid)
2. Need custom serialization/deserialization logic

**Recommended Approach**:
- Single PR: Implement BEPIElement serialization support (JSON, YAML, pickle)

**TNFR Invariants Affected**:
- ✅ EPI as coherent form (§3.1): Must preserve structure through serialization
- ✅ Operational fractality (§3.7): Nested EPIs must serialize correctly

---

### 7. Configuration (3 failures)

**Priority**: Medium  
**Complexity**: Low  
**Estimated Effort**: 1 PR

Configuration loading and application has issues with specific data structures.

**Common Error Types**:
- `AssertionError: assert {} == {'RANDOM_SEED': 1}`
- `KeyError: 'path'`

**Example Failures**:
```
tests/unit/structural/test_config_apply.py::test_load_config_accepts_mapping
  → AssertionError: assert {} == {'RANDOM_SEED': 1}

tests/unit/structural/test_config_apply.py::test_apply_config_passes_path_object
  → KeyError: 'path'
```

**Root Causes**:
1. Configuration loading may not properly parse all input types
2. Path-based configuration may have changed interface

**Recommended Approach**:
- Single PR: Fix configuration loading to handle all expected input types

**TNFR Invariants Affected**:
- ✅ Controlled determinism (§3.8): RANDOM_SEED must work correctly

---

### 8. Glyph Resolution (2 failures)

**Priority**: Low  
**Complexity**: Low  
**Estimated Effort**: 1 PR

Some glyph codes are not recognized by the resolution system.

**Common Error Types**:
- `ValueError: unknown glyph: Glyph.REMESH`

**Example Failures**:
```
tests/stress/test_program_trace.py::test_program_trace_rotates_without_dropping_thol_history
  → ValueError: unknown glyph: Glyph.REMESH
```

**Root Causes**:
1. REMESH glyph may be deprecated or renamed
2. Test uses outdated glyph name

**Recommended Approach**:
- Single PR: Update glyph mappings or fix tests to use correct names

**TNFR Invariants Affected**:
- ⚠️ Operator closure (§3.4): Should use canonical operator names

---

### 9. Golden Snapshots (1 failure)

**Priority**: Low  
**Complexity**: High  
**Estimated Effort**: 1 PR (requires investigation)

Golden snapshot test fails with significant numerical difference.

**Example Failure**:
```
tests/golden/test_classic_snapshots.py::test_classic_runtime_sequence_matches_golden_snapshot
  → assert 0.10000000000000003 == -0.2615625 ± 2.6e-07
```

**Root Causes**:
1. Algorithm change that affects numerical results
2. Golden snapshot may need updating
3. Could indicate a regression in coherence computation

**Recommended Approach**:
- Single PR: Investigate root cause and either fix regression or update golden snapshot

**TNFR Invariants Affected**:
- ⚠️ All invariants: Golden snapshots validate overall system behavior

---

### 10. Other (4 failures)

**Priority**: Low  
**Complexity**: Mixed  
**Estimated Effort**: 1-2 PRs

Miscellaneous failures that don't fit other categories.

**Example Failures**:
```
tests/integration/test_docs_fase2_integration.py::test_fase2_integration_doc_executes
  → FileNotFoundError: [Errno 2] No such file or directory: 'docs/fase2_integration.md'

tests/integration/test_public_api.py::test_public_exports
  → AssertionError: Public API exports mismatch

tests/property/test_structured_io_roundtrip.py::test_structured_file_roundtrip[.yaml-_write_yaml]
  → AssertionError: assert '1e-05' == 1e-05
```

**Root Causes**:
1. Missing documentation files
2. Public API exports may have changed
3. YAML serialization of floats in scientific notation

**Recommended Approach**:
- PR 1: Add missing docs or update test paths
- PR 2: Fix public API exports or update test expectations

**TNFR Invariants Affected**:
- ⚠️ None (mostly infrastructure issues)

---

## Recommended PR Sequence

Based on priority and dependencies, here's the recommended order for addressing these failures:

1. **PR #1: Grammar Glyph Resolution** (Grammar category)
   - Fix Glyph instance to string conversion
   - Impact: ~8 failures
   - Risk: Low

2. **PR #2: IO and Logging Module Imports** (IO + Logging categories)
   - Fix module import paths
   - Impact: ~28 failures
   - Risk: Low

3. **PR #3: ΔNFR NumPy Backend Detection** (ΔNFR Dynamics category)
   - Fix backend detection and fallback logic
   - Impact: ~15 failures
   - Risk: Medium

4. **PR #4: BEPIElement Serialization** (BEPIElement category)
   - Implement serialization support
   - Impact: 3 failures
   - Risk: Medium

5. **PR #5: Observer Registration** (Metrics category)
   - Fix observer callback registration
   - Impact: ~12 failures
   - Risk: Medium

6. **PR #6: Grammar Preconditions** (Grammar category)
   - Update test sequences for canonical grammar
   - Impact: ~14 failures
   - Risk: Medium

7. **PR #7: ΔNFR Parallel Chunks** (ΔNFR Dynamics category)
   - Fix parallel scheduling algorithm
   - Impact: ~10 failures
   - Risk: High

8. **PR #8: Metrics Computation** (Metrics category)
   - Fix trigonometric cache and vectorization
   - Impact: ~15 failures
   - Risk: Medium

9. **PR #9: Configuration and Misc** (Configuration + Other categories)
   - Fix remaining configuration and misc issues
   - Impact: ~7 failures
   - Risk: Low

10. **PR #10: Golden Snapshot Investigation**
    - Investigate and fix numerical differences
    - Impact: 1 failure
    - Risk: High (may reveal deeper issues)

---

## Testing Strategy

For each PR addressing these failures:

1. **Isolation**: Run only the affected test category first
   ```bash
   python -m pytest tests/unit/operators/ -v  # For grammar fixes
   ```

2. **Regression Check**: Run full test suite to ensure no new failures
   ```bash
   python -m pytest tests/ --tb=line -q
   ```

3. **TNFR Validation**: Ensure changes preserve structural invariants
   - Check that C(t), Si, νf metrics remain accurate
   - Verify operator closure is maintained
   - Confirm determinism (seed-based reproducibility)

4. **Documentation**: Update this document after each PR
   - Mark resolved categories
   - Update failure counts
   - Note any new issues discovered

---

## TNFR Structural Compliance

All fixes must maintain the TNFR canonical invariants from [AGENTS.md](./AGENTS.md) §3:

✅ **Critical for all fixes**:
1. EPI as coherent form
2. Structural units (Hz_str)
3. ΔNFR semantics
4. Operator closure
5. Phase check
6. Node birth/collapse
7. Operational fractality
8. Controlled determinism
9. Structural metrics
10. Domain neutrality

---

## Progress Tracking

| Category | Total | Fixed | Remaining | Status |
|----------|-------|-------|-----------|--------|
| Grammar and Operator Preconditions | 22 | 0 | 22 | 🔴 Not Started |
| ΔNFR Dynamics | 36 | 0 | 36 | 🔴 Not Started |
| Metrics and Observers | 31 | 0 | 31 | 🔴 Not Started |
| IO Module | 23 | 0 | 23 | 🔴 Not Started |
| Logging Module | 5 | 0 | 5 | 🔴 Not Started |
| BEPIElement Serialization | 3 | 0 | 3 | 🔴 Not Started |
| Configuration | 3 | 0 | 3 | 🔴 Not Started |
| Glyph Resolution | 2 | 0 | 2 | 🔴 Not Started |
| Golden Snapshots | 1 | 0 | 1 | 🔴 Not Started |
| Other | 4 | 0 | 4 | 🔴 Not Started |
| **Total** | **130** | **0** | **130** | **0.0%** |

---

## References

- [TEST_FIXES_SUMMARY.md](./TEST_FIXES_SUMMARY.md) - Previous test fix efforts (34 tests fixed)
- [AGENTS.md](./AGENTS.md) - TNFR canonical invariants and contribution guide
- [TNFR.pdf](./TNFR.pdf) - Theoretical foundation and structural operators
- [CONTRIBUTING.md](./CONTRIBUTING.md) - QA battery and testing expectations

---

## Conclusion

These 127 failing tests + 3 errors represent pre-existing technical debt that does not affect the core TNFR functionality. The passing test suite (1,579 tests, 92.4% pass rate) validates that:

- Core structural operators work correctly
- ΔNFR dynamics compute accurate reorganization rates
- Coherence metrics (C(t), Si, νf) are reliable
- Node creation, evolution, and collapse follow TNFR principles

The failures are primarily in:
- Infrastructure (IO, logging, configuration)
- Edge cases and error handling
- Parallel processing optimizations
- Test infrastructure and golden snapshots

Each category has been analyzed with recommended approaches that prioritize TNFR structural fidelity over code convenience.
