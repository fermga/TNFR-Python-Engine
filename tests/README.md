# TNFR Test Suite

[TESTING.md](../TESTING.md) owns installation, selection, optional backends,
commands and validation reporting. [AGENTS.md](../AGENTS.md) and affected source
contracts define the behavior to verify.

This directory contains top-level tests and subject-specific subdirectories.
[conftest.py](conftest.py) owns shared pytest configuration and fixtures;
[utils.py](utils.py) supplies additional helpers; [data/](data/) stores fixtures.
Keep test-specific assumptions explicit. Research producers and retained
evidence have separate provenance requirements in
[benchmarks/README.md](../benchmarks/README.md).
