# Pattern Detection Consolidation Guide

## Overview

This document explains the consolidation of pattern detection modules in TNFR Python Engine, consolidating `canonical_patterns.py` and `patterns.py` into a unified `pattern_detection.py` module with explicit U1-U4 grammar rule mappings.

## Migration Path

### Old Code (Deprecated)

```python
# Using canonical_patterns.py
from tnfr.operators.canonical_patterns import CANONICAL_SEQUENCES
seq = CANONICAL_SEQUENCES["bifurcated_base"]

# Using patterns.py
from tnfr.operators.patterns import AdvancedPatternDetector
detector = AdvancedPatternDetector()
pattern = detector.detect_pattern(sequence)
```

### New Code (Recommended)

```python
# Using unified pattern_detection.py
from tnfr.operators.pattern_detection import (
    UnifiedPatternDetector,
    detect_pattern,
    analyze_sequence,
)

# Create detector
detector = UnifiedPatternDetector()

# Detect primary pattern
pattern = detector.detect_pattern(["emission", "coupling", "coherence"])

# Or use convenience function
pattern = detect_pattern(["emission", "coupling", "coherence"])

# Detect all patterns with grammar rules
all_patterns = detector.detect_all_patterns(sequence)
for p in all_patterns:
    print(f"{p.pattern_name}: {p.grammar_rule} - {p.description}")

# Get grammar rule for a pattern
grammar_rule = detector.get_grammar_rule_for_pattern("cold_start")
print(f"cold_start maps to: {grammar_rule}")  # "U1a"
```

## Pattern Categories and Grammar Mappings

### U1a: Initiation Patterns (GENERATORS)

**Patterns:**
- **cold_start**: Begins with AL (Emission) from EPI=0
- **phase_transition_start**: Begins with NAV (Transition)
- **fractal_awakening**: Begins with REMESH (Recursivity)

**Example:**
```python
detector = UnifiedPatternDetector()
sequence = ["emission", "coherence", "silence"]
patterns = detector.detect_initiation_patterns(sequence)
# Detects: cold_start (grammar_rule="U1a")
```

### U1b: Closure Patterns (CLOSURES)

**Patterns:**
- **terminal_silence**: Ends with SHA (Silence) - νf → 0
- **regime_handoff**: Ends with NAV (Transition)
- **fractal_distribution**: Ends with REMESH (Recursivity)
- **intentional_tension**: Ends with OZ (Dissonance)

**Example:**
```python
sequence = ["emission", "coherence", "silence"]
patterns = detector.detect_closure_patterns(sequence)
# Detects: terminal_silence (grammar_rule="U1b")
```

### U2: Convergence Patterns (STABILIZERS/DESTABILIZERS)

**Patterns:**
- **stabilization_cycle**: Destabilizer → Stabilizer (bounded evolution)
- **bounded_evolution**: Oscillation between destabilizers and stabilizers
- **runaway_risk**: Destabilizers without stabilizers (divergence risk)

**Example:**
```python
sequence = ["emission", "dissonance", "coherence", "silence"]
patterns = detector.detect_convergence_patterns(sequence)
# Detects: stabilization_cycle (grammar_rule="U2")
```

### U3: Resonance Patterns (COUPLING_RESONANCE)

**Patterns:**
- **coupling_chain**: Multiple UM (Coupling) operations
- **resonance_cascade**: Multiple RA (Resonance) propagations
- **phase_locked_network**: Alternating UM ↔ RA (synchronized network)

**Example:**
```python
sequence = ["emission", "coupling", "resonance", "coherence", "silence"]
patterns = detector.detect_resonance_patterns(sequence)
# Detects: phase_locked_network (grammar_rule="U3")
```

### U4: Bifurcation Patterns (TRANSFORMERS)

**Patterns:**
- **graduated_destabilization**: Destabilizer → Transformer (U4b)
- **managed_bifurcation**: Trigger → Handler (U4a)
- **stable_transformation**: IL → ZHIR (stable base for transformation)
- **spontaneous_organization**: Disorder → THOL

**Example:**
```python
sequence = ["emission", "dissonance", "mutation", "coherence", "silence"]
patterns = detector.detect_bifurcation_patterns(sequence)
# Detects: graduated_destabilization (grammar_rule="U4b")
```

## Comprehensive Analysis

The `analyze_sequence` function provides detailed analysis:

```python
from tnfr.operators.pattern_detection import analyze_sequence

sequence = ["emission", "dissonance", "coherence", "coupling", "resonance", "silence"]
analysis = analyze_sequence(sequence)

print(f"Primary pattern: {analysis['primary_pattern']}")
print(f"Pattern scores: {analysis['pattern_scores']}")
print(f"Components: {analysis['components']}")
print(f"Complexity: {analysis['complexity_score']}")
print(f"Health: {analysis['structural_health']}")
```

## Grammar Rule Reference

| Rule | Description | Operator Sets |
|------|-------------|---------------|
| **U1a** | Structural Initiation | GENERATORS (emission, transition, recursivity) |
| **U1b** | Structural Closure | CLOSURES (silence, transition, recursivity, dissonance) |
| **U2** | Convergence & Boundedness | STABILIZERS ↔ DESTABILIZERS |
| **U3** | Resonant Coupling | COUPLING_RESONANCE (coupling, resonance) |
| **U4a** | Bifurcation Triggers & Handlers | BIFURCATION_TRIGGERS → BIFURCATION_HANDLERS |
| **U4b** | Transformer Context | DESTABILIZERS → TRANSFORMERS |

## PatternMatch DataClass

All detected patterns return `PatternMatch` objects:

```python
@dataclass
class PatternMatch:
    pattern_name: str           # e.g., "cold_start"
    start_idx: int              # Starting position in sequence
    end_idx: int                # Ending position in sequence
    confidence: float           # Match confidence (0.0-1.0)
    grammar_rule: str           # e.g., "U1a", "U2", "U4b"
    description: str            # Human-readable description
    structural_pattern: Optional[StructuralPattern]  # Enum if applicable
```

## Backward Compatibility

The old modules remain functional but issue deprecation warnings:

```python
# Still works, but deprecated
from tnfr.operators.canonical_patterns import CANONICAL_SEQUENCES  # DeprecationWarning
from tnfr.operators.patterns import AdvancedPatternDetector       # DeprecationWarning

# Both still work correctly, maintaining backward compatibility
```

## Benefits of Unified Module

1. **Single Source of Truth**: One module for all pattern detection
2. **Explicit Grammar Mapping**: Every pattern explicitly maps to U1-U4 rules
3. **Improved Traceability**: Clear linkage between patterns and TNFR physics
4. **Comprehensive Detection**: Detects both canonical sequences and meta-patterns
5. **Grammar Validation**: Respects unified grammar constraints
6. **Better Documentation**: Self-documenting with grammar rule references

## Physics Basis

All patterns are derived from TNFR physics as documented in:

- **UNIFIED_GRAMMAR_RULES.md**: Complete physics derivations for U1-U4
- **AGENTS.md**: Canonical invariants and formal contracts
- **TNFR.pdf**: Nodal equation ∂EPI/∂t = νf · ΔNFR(t) and bifurcation theory

## Canonical Sequences

Canonical sequences from `canonical_patterns.py` remain the authoritative source for archetypal patterns. The new unified detector recognizes these sequences and maps them to grammar rules:

```python
from tnfr.operators.canonical_patterns import CANONICAL_SEQUENCES

# Canonical sequences still available
seq = CANONICAL_SEQUENCES["bifurcated_base"]
print(seq.name)          # "bifurcated_base"
print(seq.pattern_type)  # StructuralPattern.BIFURCATED
print(seq.glyphs)        # [Glyph.AL, Glyph.EN, Glyph.IL, ...]

# But pattern detection should use unified module
from tnfr.operators.pattern_detection import detect_pattern
pattern = detect_pattern([g.value for g in seq.glyphs])
```

## Testing

Comprehensive tests in `tests/unit/operators/test_pattern_detection.py` validate:

- U1-U4 pattern detection accuracy
- Grammar rule mappings
- Confidence scores
- Backward compatibility with AdvancedPatternDetector
- Edge cases (empty sequences, single operators)

Run tests:
```bash
pytest tests/unit/operators/test_pattern_detection.py -v
```

## Future Work

- [ ] Update example files to use unified module
- [ ] Update inline documentation references
- [ ] Add cookbook examples for each pattern category
- [ ] Integrate with sequence generator for pattern-guided generation
- [ ] Add visualization for detected patterns

## References

- Issue: #[consolidate-pattern-detection]
- Design doc: UNIFIED_GRAMMAR_RULES.md
- Physics basis: TNFR.pdf Section 2.3 (Bifurcations and patterns)
