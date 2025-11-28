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

### Auto-Registration & Decorator

Operators are auto-registered via reflective discovery (`discover_operators()`)
and may declare explicit intent using the `@structural_operator` decorator:

```python
from tnfr.operators.definitions_base import Operator
from tnfr.operators.registry import structural_operator
from tnfr.types import Glyph

@structural_operator
class MyOperator(Operator):
    name = "my_operator"
    glyph = Glyph.AL  # choose canonical glyph or leave None for non-glyph ops

    def __call__(self, G, node, **kw):
        super().__call__(G, node, **kw)
        # structural transformation via grammar already applied
```

Idempotent: multiple imports or hot-reloads never duplicate registry entries.

### Metaclass Alternative (OperatorMetaAuto)

All operator subclasses also auto-register via the optional metaclass
`OperatorMetaAuto` attached to the base `Operator`. This is physics-neutral:
it does not alter glyph semantics. You can opt out by setting
`__register__ = False` on a subclass (rare; mainly for abstract helpers).

Decorator + metaclass together remain safe; duplicate registrations are
silently ignored.

### Telemetry & Cache Stats

Operator registry telemetry is available via:

```python
from tnfr.operators.registry import get_operator_cache_stats
stats = get_operator_cache_stats()
print(stats)
# {'registrations': 13, 'soft_invalidations': 2, 'hard_invalidations': 0,
#  'last_invalidation_ts': 1731552000.123, 'current_count': 13}
```

Grammar-level LRU cache statistics (for memoized validation helpers) via:

```python
from tnfr.operators.grammar import get_grammar_cache_stats
gstats = get_grammar_cache_stats()
```

### CLI Helper (scripts/tnfr_ops_reload.py)

Interactive development reload:

```bash
python scripts/tnfr_ops_reload.py          # soft reload
python scripts/tnfr_ops_reload.py --hard   # hard reload (drops registry)
python scripts/tnfr_ops_reload.py --stats  # show stats only
```

Output includes registry before/after, invalidation mode, operator names,
and grammar cache metrics in JSON form.

### Cache Invalidation (Hot Reload)

Use `invalidate_operator_cache()` for development hot-reload after editing
operator source files:

```python
from tnfr.operators.registry import invalidate_operator_cache, OPERATORS

stats = invalidate_operator_cache(hard=False)  # soft: preserves existing mapping
print(stats)  # {'count': 13, 'cleared': 0}

# Hard reset (drops existing registry first):
invalidate_operator_cache(hard=True)
```

Soft invalidation clears the discovery flag and re-runs reflection.
Hard invalidation also empties `OPERATORS` before re-population.

Telemetry fields updated:

- `soft_invalidations` / `hard_invalidations`
- `last_invalidation_ts` (ISO available in CLI helper)
- `registrations` (unique successful registrations)

### Registry Integrity Tests

`tests/test_operator_registry_consistency.py` enforces:

- Every glyph maps to a registered operator name
- No duplicate class objects bound to multiple names
- Name→class mapping mirrors `OPERATORS`


### Design Rationale

- Reflection avoids manual lists in grammar (`_CORE_OPERATORS` removed)
- Decorator communicates structural intent close to definition
- Invalidation integrates with existing cache philosophy (targeted, physics-safe)

## Tests

- See `tests/` for coherence monotonicity, bifurcation handlers, and propagation.

## No redundancy policy

This module README links to canonical sources and implementation files. It intentionally avoids duplicating theoretical content already covered in `AGENTS.md`, `UNIFIED_GRAMMAR_RULES.md`, and the mathematics/physics READMEs.
