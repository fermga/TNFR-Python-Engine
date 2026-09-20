# TNFR Test Suite

[TESTING.md](../TESTING.md) owns installation, selection, optional backends,
commands and validation reporting. [AGENTS.md](../AGENTS.md) and affected source
contracts define the behavior to verify.

This directory contains top-level tests and subject-specific subdirectories.
The routine engine selection is owned by `testpaths` in
[pyproject.toml](../pyproject.toml); `python -m pytest tests` explicitly selects
the full retained inventory. Specialized research controls are run when their
models change, rather than on every engine/API edit.
[conftest.py](conftest.py) owns shared pytest configuration and fixtures;
[utils.py](utils.py) supplies additional helpers; [data/](data/) stores fixtures.
Keep test-specific assumptions explicit. Research producers and retained
evidence have separate provenance requirements in
[benchmarks/README.md](../benchmarks/README.md).
