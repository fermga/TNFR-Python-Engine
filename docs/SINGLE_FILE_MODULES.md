# Single-File Module Guide

These public modules remain single Python files. Their documentation is
centralized here so the source tree does not contain same-named, non-package
directories that resemble importable packages.

| Module | Responsibility |
| --- | --- |
| `tnfr.flatten` | Flattening helpers for nested structural data |
| `tnfr.gamma` | Gamma-related numerical helpers retained by the public API |
| `tnfr.glyph_history` | Operator-history representation and access |
| `tnfr.glyph_runtime` | Runtime glyph execution support |
| `tnfr.immutable` | Immutable structural-data helpers |
| `tnfr.initialization` | Network and node initialization helpers |
| `tnfr.io` | Public input/output facade |
| `tnfr.node` | Nodal data structures and lifecycle helpers |
| `tnfr.observers` | Runtime observer interfaces |
| `tnfr.structural` | NFR creation and canonical sequence execution |

The canonical physics and invariants are defined in
[AGENTS.md](../AGENTS.md). Operator composition is specified in
[Unified Grammar Rules](../theory/UNIFIED_GRAMMAR_RULES.md). Public functions
and compatibility status are documented by their module docstrings and type
stubs.
