# TNFR Dynamics — Canonical Module Hub (Single Source of Truth)

Centralized entry point for nodal equation integration and dynamical evolution. English-only; links consolidate authoritative references and avoid duplication.

- Core equation: ∂EPI/∂t = νf · ΔNFR (see `TNFR.pdf` §1–2)
- Canonical physics and invariants: `AGENTS.md`
- Unified grammar derivations (U1–U6): `UNIFIED_GRAMMAR_RULES.md`
- Computational mathematics hub: `src/tnfr/mathematics/README.md`
- Structural fields (telemetry): `src/tnfr/physics/README.md`

## Scope
- Nodal integration and solvers: `integrators.py`, `steppers.py` (if present), `nodal_equation.py`
- Stability and boundedness checks (U2): helper routines in this package
- Hooks for operator application within integration steps

## Guarantees
- Integrators respect grammar constraints: generators at start, closure at end (U1)
- Convergence monitored via ∫ νf·ΔNFR dt with stabilizers where required (U2)
- Bifurcation control: triggers `{OZ, ZHIR}` must have handlers `{THOL, IL}` (U4)

## Usage
```python
from tnfr.dynamics.nodal_equation import step
EPI_next = step(G, dt=1.0)  # Advances EPI by applying ∂EPI/∂t = νf · ΔNFR
```

## Tests
- See `tests/` for latency invariance under Silence, convergence checks, and propagation behavior.

## No redundancy policy
This README links out to canonical documents and implementation files; it deliberately avoids restating theory already covered centrally.
