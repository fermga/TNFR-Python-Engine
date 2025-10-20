# Test suite overview

This document describes the purpose of tests in the suite that monitor internal debugging, diagnostic features, and performance regressions.

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
