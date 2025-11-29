# TNFR Mathematical Purity Era Test Suite

This document describes the completely rewritten test suite for TNFR after achieving **497+ canonical constants** and **pure mathematical framework** status. All tests now validate mathematical purity, canonical parameter usage, and fundamental TNFR physics.

## Test Organization (Mathematical Purity Era)

- **mathematical_purity/** - TIER 1: Canonical constants and zero empirical fitting validation
- **core_physics/** - TIER 2: Nodal equation, structural fields, and fundamental physics
- **grammar_operators/** - TIER 3: Unified grammar U1-U6 and 13 canonical operators  
- **integration/** - TIER 4: End-to-end pure TNFR engine workflows
- **validation/** - TIER 5: Performance and regression with canonical constants
- **archive/pre_purification/** - Archived tests from before mathematical purity achievement

## Quantitative Validation Suite

The **validation/** directory contains comprehensive quantitative validation tests for TNFR n-body dynamics, implementing experiments from `docs/source/theory/09_classical_mechanics_numerical_validation.md`.

See [validation/README.md](validation/README.md) for details on:
- Running validation tests and example scripts
- Acceptance criteria and parameters
- Generated visualization outputs
- Reproducibility with fixed seeds

Quick start:
```bash
# Run validation tests
pytest tests/validation/ -v -s

# Generate visualizations
python examples/nbody_quantitative_validation.py
```

## Backend selection

Pass `--math-backend=<name>` to `pytest` (or set `TNFR_TEST_MATH_BACKEND`) to exercise the suite under a specific numerical adapter. The flag sets `TNFR_MATH_BACKEND` before modules import and clears cached backend instances, ensuring that optional dependencies such as JAX or PyTorch are only used when explicitly requested.

```bash
PYTHONPATH=src pytest tests/mathematics --math-backend=torch
```

When the requested backend is unavailable, `pytest` will skip the backend-specific tests while the remainder of the suite continues to run under NumPy.

## Debugging and diagnostic tests

- **test_logging_module.py** – verifies that the root logger is configured once and sets a default level.
- **test_logging_threadsafe.py** – ensures thread-safe initialization of the logging system across concurrent calls.
- **test_warn_failure_emit.py** – checks warning and logging behavior for failed module imports.
- **test_trace.py** – confirms that trace utilities register callbacks and capture metadata.
- **test_diagnosis_state.py** – validates mapping of coherence metrics into diagnostic states.

Tests that previously covered warning cache pruning and warn-once limits were removed to reduce maintenance overhead.

## Performance regression tests

- **performance/** – covers the NumPy-accelerated ΔNFR pipeline, alias caches, and trigonometric metrics to guard against performance regressions. The tests are marked `slow` and are excluded by default via the `addopts` setting in `pyproject.toml`.

  ```bash
  pytest -m slow tests/performance
  ```

  The suite requires NumPy for the vectorized branches; `pytest` will skip the affected checks automatically when the dependency is missing.

Benchmark helpers remain available behind the `benchmarks` marker. To run them explicitly use:

```bash
pytest benchmarks --benchmark-only
```

This command complements the default `--benchmark-skip` flag defined in `pyproject.toml`.
