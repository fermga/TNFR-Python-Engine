# TNFR Operators — Canonical Module Hub (Single Source of Truth)

This README is the centralized entry point for the Operators module. It consolidates the essential references and avoids redundant documentation. All content is in English.

- Canonical physics: `AGENTS.md` (invariants, operator taxonomy) and `UNIFIED_GRAMMAR_RULES.md` (U1–U6)
- Core equation: ∂EPI/∂t = νf · ΔNFR (see `TNFR.pdf` §1–2)
- Computational mathematics hub: `src/tnfr/mathematics/README.md`
- Structural fields (telemetry): `src/tnfr/physics/README.md`

## Scope
- Operator definitions and contracts: `definitions.py`
- Grammar validation and composition: `grammar.py`, `unified_grammar.py`
- Preconditions/Postconditions: `preconditions/`, `postconditions/`
- Registry and canonical patterns: `registry.py`, `canonical_patterns.py`

## Guarantees
- All EPI changes occur only via operators (Invariant #1)
- Resonant coupling requires phase verification (U3) before any `UM`/`RA`
- Destabilizers {OZ, ZHIR, VAL} must be followed by stabilizers {IL, THOL} (U2/U4)
- Operator sequences must start with a generator and end with closure (U1)

## Usage
```python
from tnfr.operators.definitions import Coherence, Dissonance, Resonance
from tnfr.operators.grammar import validate_resonant_coupling, apply_sequence

seq = [Coherence(), Dissonance(intensity=0.2), Coherence()]
apply_sequence(G, node, seq)  # Preserves U1–U4 contracts
```

## Tests
- See `tests/` for coherence monotonicity, bifurcation handlers, and propagation.

## No redundancy policy
This module README links to canonical sources and implementation files. It intentionally avoids duplicating theoretical content already covered in `AGENTS.md`, `UNIFIED_GRAMMAR_RULES.md`, and the mathematics/physics READMEs.
