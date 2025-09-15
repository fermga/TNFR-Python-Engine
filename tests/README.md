# Test suite overview

This document describes the purpose of tests in the suite that monitor internal debugging and diagnostic features.

## Debugging and diagnostic tests

- **test_logging_module.py** – verifies that the root logger is configured once and sets a default level.
- **test_logging_threadsafe.py** – ensures thread-safe initialization of the logging system across concurrent calls.
- **test_warn_failure_emit.py** – checks warning and logging behavior for failed module imports.
- **test_trace.py** – confirms that trace utilities register callbacks and capture metadata.
- **test_diagnosis_state.py** – validates mapping of coherence metrics into diagnostic states.

Tests that previously covered warning cache pruning and warn-once limits were removed to reduce maintenance overhead.
