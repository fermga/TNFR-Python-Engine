# TNFR Backends — Canonical Module Hub (Single Source of Truth)

Centralized entry point for numerical backend integration. English-only; links consolidate authoritative references and avoid duplication.

- Canonical physics and invariants: `AGENTS.md`
- Unified grammar (U1–U6): `UNIFIED_GRAMMAR_RULES.md`
- Core equation: ∂EPI/∂t = νf · ΔNFR (`TNFR.pdf` §1–2)
- Computational mathematics hub: `src/tnfr/mathematics/README.md`

## Scope
- Backend selection and API shims
- Array creation, linear algebra, eigendecomposition, sparse ops
- Compatibility with NumPy/JAX/PyTorch where supported

## Guarantees
- Stable, explicit backend selection
- Type-annotated interfaces
- No theory duplication; defers to central hubs

## Usage
```python
from tnfr.backends import as_array, eigh
```

## No redundancy policy
This README links to canonical sources and module files; it intentionally avoids restating theory covered centrally.
