# Grammar 2.0 Performance Tools

This directory contains profiling and analysis tools for Grammar 2.0 performance optimization.

## Tools

### `grammar_profiler.py`

Comprehensive profiling tool for Grammar 2.0 capabilities.

**Features:**
- Profiles all Grammar 2.0 components (validation, health analysis, pattern detection, cycle validation)
- Generates detailed performance reports with statistics
- Identifies bottlenecks for optimization
- Checks compliance against performance targets
- Supports profiling by sequence length

**Usage:**

```bash
# Run with default test sequences
python tools/performance/grammar_profiler.py

# Suppress logging for cleaner output
TNFR_LOG_LEVEL=ERROR python tools/performance/grammar_profiler.py
```

**Output Example:**

```
================================================================================
Grammar 2.0 Performance Profile
================================================================================

Basic Validation:
  Runs:     500
  Min:      11.03 μs
  Max:      218.30 μs
  Mean:     43.75 μs
  Median:   15.67 μs
  StdDev:   54.62 μs
  Metadata:
    sequences: 5
    iterations: 100
    avg_length: 8.4

Performance Targets Check:
--------------------------------------------------------------------------------
Basic Validation:
  Target: 2000 μs
  Actual: 43.75 μs (2.2% of target)
  Status: ✓ PASS
```

### Integration with Benchmarking

The profiler complements the benchmark suite in `benchmarks/grammar_2_0_benchmarks.py`:

- **Profiler**: Detailed component-level analysis, bottleneck identification
- **Benchmarks**: Regression testing, target compliance, comprehensive test coverage

## Performance Targets

From issue #XXXX, Grammar 2.0 should meet these performance targets:

| Component | Target | Current | Status |
|-----------|--------|---------|--------|
| Basic validation | < 2ms | ~16-45μs | ✓ PASS |
| Health analysis | < 10ms | ~8.6μs | ✓ PASS |
| Pattern detection | < 5ms | ~0.24μs | ✓ PASS |
| Cycle validation | < 3ms | ~43μs | ✓ PASS |

## Optimization Techniques

The following optimizations are implemented in Grammar 2.0:

### 1. Single-Pass Analysis

The `SequenceHealthAnalyzer` uses `_compute_single_pass()` to scan sequences once and extract all needed statistics, eliminating redundant iterations.

**Before:**
```python
# Multiple passes through sequence
stabilizers = sum(1 for op in sequence if op in _STABILIZERS)
destabilizers = sum(1 for op in sequence if op in DESTABILIZERS)
unique_ops = len(set(sequence))
```

**After:**
```python
# Single pass extracts all statistics
analysis = self._analysis_cache(tuple(sequence))
stabilizer_count, destabilizer_count, _, _, unique_count, _ = analysis
```

### 2. Result Caching

Both `SequenceHealthAnalyzer` and `AdvancedPatternDetector` use `functools.lru_cache` to cache results for repeated sequences.

**Configuration:**
- Health analyzer: `maxsize=128` (common sequences in workflows)
- Pattern detector: `maxsize=256` (pattern exploration use cases)

**Benefits:**
- 50-100x speedup for repeated pattern detection
- Minimal memory overhead (LRU eviction)
- Thread-safe (built into `lru_cache`)

### 3. Pre-Computed Values

Statistics are computed once and passed between methods to avoid recalculation:

```python
# Pre-compute once
unique_count = len(unique_ops_set)

# Use in multiple metrics
efficiency = self._calculate_efficiency(sequence, unique_count)
pattern_value = self._assess_pattern_value_optimized(sequence, unique_count)
```

## Best Practices

When working with Grammar 2.0 performance:

1. **Reuse analyzer instances** to benefit from caching:
   ```python
   analyzer = SequenceHealthAnalyzer()
   for seq in sequences:
       health = analyzer.analyze_health(seq)  # Cache hits on repeated sequences
   ```

2. **Use immutable sequences** when possible to enable caching:
   ```python
   # Good - can be cached
   seq = ("emission", "reception", "coherence", "silence")
   
   # Less efficient - converted to tuple for caching
   seq = ["emission", "reception", "coherence", "silence"]
   ```

3. **Profile before optimizing** - use the profiler to identify actual bottlenecks:
   ```python
   profiler = Grammar20Profiler()
   profiler.profile_validation_pipeline(test_sequences)
   bottlenecks = profiler.identify_bottlenecks()
   ```

4. **Batch validate** sequences to maximize cache efficiency:
   ```python
   # Good - benefits from caching
   results = [validate_sequence_with_health(seq) for seq in batch]
   
   # Less efficient - recreates analyzer each time
   for seq in batch:
       analyzer = SequenceHealthAnalyzer()
       analyzer.analyze_health(seq)
   ```

## Contributing

When making performance-related changes:

1. Run the profiler to establish baseline
2. Make changes
3. Run profiler again to measure impact
4. Ensure all performance targets still pass
5. Update benchmarks if adding new features

## See Also

- `benchmarks/grammar_2_0_benchmarks.py` - Comprehensive benchmark suite
- `tests/performance/test_grammar_2_0_performance.py` - Performance regression tests
- `src/tnfr/operators/health_analyzer.py` - Health analysis implementation
- `src/tnfr/operators/patterns.py` - Pattern detection implementation
