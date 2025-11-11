# TNFR Structural Patterns Reference

## Overview

This reference catalogs all **18 structural patterns** in Grammar 2.0. Each pattern represents a specific structural signature that can be detected in operator sequences.

**Pattern Categories**:
- **5 Fundamental**: Basic structural types (LINEAR, HIERARCHICAL, FRACTAL, CYCLIC, BIFURCATED)
- **5 Domain-Specific**: Application-specialized patterns (THERAPEUTIC, EDUCATIONAL, ORGANIZATIONAL, CREATIVE, REGENERATIVE)
- **5 Compositional**: Building blocks (BOOTSTRAP, EXPLORE, STABILIZE, RESONATE, COMPRESS)
- **3 Complexity**: Meta-patterns (COMPLEX, MINIMAL, UNKNOWN)

---

## Pattern Properties

### Coherence Weights

Patterns are scored using **coherence-weighted matching**. Weights reflect structural depth (emergence, self-organization, phase transitions):

| Level | Weight | Patterns |
|-------|--------|----------|
| **Level 3** (Emergent) | 3.0 | THERAPEUTIC, EDUCATIONAL, ORGANIZATIONAL, CREATIVE, REGENERATIVE |
| **Level 2** (Transformational) | 2.0 | HIERARCHICAL, BIFURCATED, FRACTAL, CYCLIC |
| **Level 1** (Compositional) | 1.0 | BOOTSTRAP, EXPLORE, STABILIZE, RESONATE, COMPRESS |
| **Level 0.5** (Simple) | 0.5 | LINEAR, MINIMAL |
| **Level 1.5** (Moderate) | 1.5 | COMPLEX |
| **Level 0.1** (Fallback) | 0.1 | UNKNOWN |

**Rationale**: Emergent patterns (with self-organization, phase transitions) have fundamentally higher structural complexity than simple compositional patterns.

---

## Fundamental Patterns

### 1. LINEAR

**Category**: Fundamental  
**Coherence Weight**: 0.5 (simple)  
**Structural Depth**: Low

#### Description
Simple progression without transformation. No dissonance, mutation, or self-organization. Straightforward flow from start to end.

#### Signature
- Maximum 5 operators
- Excludes: DISSONANCE, MUTATION, SELF_ORGANIZATION
- Linear flow without branching or emergence

#### Canonical Sequences
```python
["emission", "reception", "coherence"]
["emission", "coherence", "silence"]
["transition", "emission", "coherence"]
```

#### Health Profile
- **Coherence**: 0.75-0.85 (good flow)
- **Balance**: 0.70-0.80 (stable, no forces)
- **Sustainability**: 0.60-0.75 (moderate)
- **Overall**: 0.70-0.80

#### Use Cases
- Quick activations
- Simple initialization
- Basic state transitions
- Testing/prototyping

---

### 2. HIERARCHICAL

**Category**: Fundamental  
**Coherence Weight**: 2.0 (transformational)  
**Structural Depth**: Medium-High

#### Description
Self-organization creates nested structure. Emergence of sub-structures within larger structure. Implies hierarchical organization.

#### Signature
- **Requires**: SELF_ORGANIZATION (THOL)
- Creates nested EPIs
- Hierarchical emergence

#### Canonical Sequences
```python
["dissonance", "self_organization", "coherence"]
["emission", "dissonance", "self_organization", "resonance"]
["reception", "emission", "coherence", "dissonance", "self_organization"]
```

#### Health Profile
- **Coherence**: 0.75-0.90 (emergent pattern)
- **Balance**: 0.75-0.85 (destabilizer + stabilizer)
- **Sustainability**: 0.70-0.85 (self-organizing)
- **Overall**: 0.75-0.88

#### Use Cases
- Organizational structure formation
- System architecture emergence
- Team self-organization
- Nested system development

---

### 3. FRACTAL

**Category**: Fundamental  
**Coherence Weight**: 2.0 (transformational)  
**Structural Depth**: Medium-High

#### Description
Recursive structure across scales. Self-similar patterns at different levels. Requires transition and coupling/recursivity.

#### Signature
- **Requires**: TRANSITION (NAV)
- **Requires one of**: COUPLING (UM) or RECURSIVITY (REMESH)
- Scale-invariant patterns

#### Canonical Sequences
```python
["transition", "coupling", "transition", "recursivity"]
["transition", "emission", "coupling", "transition"]
["recursivity", "transition", "coupling", "recursivity"]
```

#### Health Profile
- **Coherence**: 0.70-0.85 (multi-scale)
- **Balance**: 0.65-0.80 (varied)
- **Sustainability**: 0.75-0.90 (recursive renewal)
- **Overall**: 0.70-0.85

#### Use Cases
- Multi-scale systems
- Recursive processes
- Fractal architectures
- Self-similar dynamics

---

### 4. CYCLIC

**Category**: Fundamental  
**Coherence Weight**: 2.0 (transformational)  
**Structural Depth**: Medium-High

#### Description
Regenerative loops with multiple state transitions. Implies cyclic reorganization and renewal.

#### Signature
- **Minimum**: 2 TRANSITION operators
- Multiple state changes
- Loop structure

#### Canonical Sequences
```python
["transition", "emission", "coherence", "transition", "resonance"]
["emission", "transition", "coherence", "transition", "emission"]
["transition", "coupling", "transition", "coupling"]
```

#### Health Profile
- **Coherence**: 0.70-0.85 (cyclic flow)
- **Balance**: 0.70-0.85 (balanced cycle)
- **Sustainability**: 0.80-0.95 (regenerative)
- **Overall**: 0.75-0.88

#### Use Cases
- Continuous processes
- Feedback loops
- Cyclic workflows
- Iterative refinement

---

### 5. BIFURCATED

**Category**: Fundamental  
**Coherence Weight**: 2.0 (transformational)  
**Structural Depth**: Medium-High

#### Description
Branching through phase transition. Possibility space branches at critical point.

#### Signature
- **Requires adjacent pair**: 
  - DISSONANCE → MUTATION, or
  - DISSONANCE → CONTRACTION
- Phase transition branching

#### Canonical Sequences
```python
["coherence", "dissonance", "mutation", "coherence"]
["emission", "dissonance", "contraction", "silence"]
["reception", "coherence", "dissonance", "mutation"]
```

#### Health Profile
- **Coherence**: 0.70-0.85 (branching structure)
- **Balance**: 0.75-0.85 (controlled forces)
- **Sustainability**: 0.65-0.80 (transition dependent)
- **Overall**: 0.70-0.83

#### Use Cases
- Decision points
- Phase transitions
- Branching processes
- Critical state changes

---

## Domain-Specific Patterns

### 6. THERAPEUTIC

**Category**: Domain-Specific  
**Coherence Weight**: 3.0 (emergent)  
**Structural Depth**: High

#### Description
Complete healing cycle with controlled crisis resolution. Full therapeutic process from reception through crisis to integration.

#### Signature
**Canonical sequence**: `RECEPTION → EMISSION → COHERENCE → DISSONANCE → SELF_ORGANIZATION → COHERENCE`

#### Operators
1. **RECEPTION**: Receive current state, acknowledge trauma/issue
2. **EMISSION**: Activate therapeutic process
3. **COHERENCE**: Stabilize preparation phase
4. **DISSONANCE**: Confront challenge/trauma (controlled crisis)
5. **SELF_ORGANIZATION**: Reorganize identity (emergent healing)
6. **COHERENCE**: Integrate new structure

#### Full Example
```python
therapeutic = [
    "reception",           # EN: Acknowledge current state
    "emission",            # AL: Activate healing
    "coherence",           # IL: Prepare safely
    "dissonance",          # OZ: Face trauma
    "self_organization",   # THOL: Identity reorganization
    "coherence"            # IL: Integrate new self
]
```

#### Health Profile
- **Coherence**: 0.90-0.95 (complete cycle)
- **Balance**: 0.85-0.90 (balanced forces)
- **Sustainability**: 0.90-0.95 (fully resolved)
- **Completeness**: 0.95-1.00 (full pattern)
- **Overall**: 0.88-0.92

#### Applications
- **Psychotherapy**: Crisis → integration → new coherence
- **Trauma healing**: Activation → confrontation → reorganization
- **Emotional regulation**: Distress → processing → stabilization
- **Personal transformation**: Old self → crisis → new self

#### Variations
```python
# Extended with resonance
[
    "reception", "emission", "coherence",
    "dissonance", "self_organization", "coherence",
    "resonance"  # Propagate new coherence
]

# With preparation
[
    "silence",  # Preparation space
    "reception", "emission", "coherence",
    "dissonance", "self_organization", "coherence"
]
```

---

### 7. EDUCATIONAL

**Category**: Domain-Specific  
**Coherence Weight**: 3.0 (emergent)  
**Structural Depth**: High

#### Description
Transformative learning with phase shift. Complete learning process from reception through expansion to paradigm mutation.

#### Signature
**Canonical sequence**: `RECEPTION → EMISSION → COHERENCE → EXPANSION → DISSONANCE → MUTATION`

#### Operators
1. **RECEPTION**: Receive information, initial exposure
2. **EMISSION**: Activate learning process
3. **COHERENCE**: Initial integration, surface understanding
4. **EXPANSION**: Explore implications, broaden understanding
5. **DISSONANCE**: Challenge existing models, cognitive conflict
6. **MUTATION**: Paradigm shift, mental model transformation

#### Full Example
```python
educational = [
    "reception",      # EN: Receive new concept
    "emission",       # AL: Engage with material
    "coherence",      # IL: Initial understanding
    "expansion",      # VAL: Explore implications
    "dissonance",     # OZ: Challenge beliefs
    "mutation"        # ZHIR: Paradigm shift
]
```

#### Health Profile
- **Coherence**: 0.85-0.92 (transformative)
- **Balance**: 0.80-0.88 (growth forces)
- **Sustainability**: 0.75-0.85 (needs integration)
- **Completeness**: 0.90-0.95 (full learning)
- **Overall**: 0.82-0.90

#### Applications
- **Conceptual breakthrough**: Understanding → challenge → paradigm shift
- **Skill acquisition**: Practice → expansion → mastery transition
- **Knowledge integration**: Facts → exploration → systemic understanding
- **Deep learning**: Surface → depth → transformation

#### Variations
```python
# With final integration
[
    "reception", "emission", "coherence",
    "expansion", "dissonance", "mutation",
    "coherence"  # Integrate new paradigm
]

# Collaborative learning
[
    "reception", "emission", "coupling",  # Social learning
    "coherence", "expansion", "dissonance", "mutation"
]
```

---

### 8. ORGANIZATIONAL

**Category**: Domain-Specific  
**Coherence Weight**: 3.0 (emergent)  
**Structural Depth**: High

#### Description
Institutional evolution with emergent reorganization. Complete organizational change process with self-organizing dynamics.

#### Signature
**Canonical sequence**: `TRANSITION → EMISSION → RECEPTION → COUPLING → RESONANCE → DISSONANCE → SELF_ORGANIZATION`

#### Operators
1. **TRANSITION**: Initiate organizational change
2. **EMISSION**: Leadership vision activation
3. **RECEPTION**: Stakeholder input gathering
4. **COUPLING**: Team/department coordination
5. **RESONANCE**: Cultural amplification
6. **DISSONANCE**: Challenge old structure, productive conflict
7. **SELF_ORGANIZATION**: Emergent new organization

#### Full Example
```python
organizational = [
    "transition",          # NAV: Initiate change
    "emission",            # AL: Leadership vision
    "reception",           # EN: Gather input
    "coupling",            # UM: Coordinate units
    "resonance",           # RA: Amplify culture
    "dissonance",          # OZ: Confront old patterns
    "self_organization"    # THOL: Emerge new org
]
```

#### Health Profile
- **Coherence**: 0.88-0.94 (complete cycle)
- **Balance**: 0.82-0.90 (balanced transformation)
- **Sustainability**: 0.85-0.92 (self-sustaining)
- **Completeness**: 0.92-0.98 (full process)
- **Overall**: 0.87-0.93

#### Applications
- **Cultural transformation**: Old structure → tension → emergent culture
- **Strategic pivots**: Current state → crisis → new configuration
- **Team evolution**: Group → conflict → self-organized collaboration
- **System change**: Hierarchy → disruption → network structure

#### Variations
```python
# With final stabilization
[
    "transition", "emission", "reception",
    "coupling", "resonance", "dissonance",
    "self_organization", "coherence"  # Stabilize new org
]

# Rapid transformation
[
    "dissonance", "transition", "emission",
    "coupling", "self_organization"  # Compressed cycle
]
```

---

### 9. CREATIVE

**Category**: Domain-Specific  
**Coherence Weight**: 3.0 (emergent)  
**Structural Depth**: High

#### Description
Artistic emergence through self-organization. Creative process from void/pause through exploration to emergent form.

#### Signature
**Canonical sequence**: `SILENCE → EMISSION → EXPANSION → DISSONANCE → MUTATION → SELF_ORGANIZATION`

#### Operators
1. **SILENCE**: Void, pause, preparation space
2. **EMISSION**: Inspiration, creative spark
3. **EXPANSION**: Exploration, experimentation
4. **DISSONANCE**: Challenge, creative tension
5. **MUTATION**: Breakthrough, radical shift
6. **SELF_ORGANIZATION**: Emergent artistic form

#### Full Example
```python
creative = [
    "silence",             # SHA: Creative void
    "emission",            # AL: Inspiration strikes
    "expansion",           # VAL: Explore possibilities
    "dissonance",          # OZ: Creative tension
    "mutation",            # ZHIR: Breakthrough
    "self_organization"    # THOL: Form emerges
]
```

#### Health Profile
- **Coherence**: 0.85-0.92 (emergent creation)
- **Balance**: 0.75-0.85 (creative tension)
- **Sustainability**: 0.80-0.90 (self-generating)
- **Completeness**: 0.88-0.95 (full cycle)
- **Overall**: 0.82-0.90

#### Applications
- **Artistic creation**: Void → inspiration → experimentation → work
- **Innovation**: Constraint → ideation → synthesis → product
- **Design process**: Research → prototyping → iteration → solution
- **Creative writing**: Blank page → idea → exploration → story

#### Variations
```python
# With final refinement
[
    "silence", "emission", "expansion",
    "dissonance", "mutation", "self_organization",
    "coherence", "silence"  # Refine and rest
]

# Collaborative creativity
[
    "silence", "emission", "coupling",  # Shared creation
    "expansion", "dissonance", "mutation",
    "self_organization"
]
```

---

### 10. REGENERATIVE

**Category**: Domain-Specific  
**Coherence Weight**: 3.0 (emergent)  
**Structural Depth**: High

#### Description
Self-sustaining cycle with autonomous renewal. Complete regenerative loop that maintains itself over time.

#### Signature
**Canonical sequence**: `COHERENCE → RESONANCE → EXPANSION → SILENCE → TRANSITION → EMISSION → RECEPTION → COUPLING → COHERENCE`

- **Minimum length**: 9 operators
- **Requires**: Regenerators (TRANSITION, RECURSIVITY, or SILENCE)
- **Requires**: Balanced stabilizers before/after regenerators
- **Requires**: R5 cycle validation passes

#### Operators (Full Cycle)
1. **COHERENCE**: Initial stable state
2. **RESONANCE**: Amplification phase
3. **EXPANSION**: Growth, elaboration
4. **SILENCE**: Rest, integration (regenerator)
5. **TRANSITION**: State change, renewal (regenerator)
6. **EMISSION**: Reactivation
7. **RECEPTION**: Input, feedback
8. **COUPLING**: Network synchronization
9. **COHERENCE**: Return to stable state (cycle complete)

#### Full Example
```python
regenerative = [
    "coherence",      # IL: Stable operation
    "resonance",      # RA: Amplify success
    "expansion",      # VAL: Grow
    "silence",        # SHA: Rest/integrate (regenerator)
    "transition",     # NAV: Renew state (regenerator)
    "emission",       # AL: Reactivate
    "reception",      # EN: New input
    "coupling",       # UM: Sync network
    "coherence"       # IL: Stable again (cycle)
]
```

#### Health Profile
- **Coherence**: 0.82-0.90 (cyclic)
- **Balance**: 0.85-0.92 (well-balanced)
- **Sustainability**: 0.90-0.98 (self-sustaining)
- **Completeness**: 0.92-1.00 (full cycle)
- **Overall**: 0.85-0.93

#### Applications
- **Ecosystem dynamics**: Growth → dormancy → renewal → growth
- **Institutional sustainability**: Operation → reflection → adaptation → operation
- **Personal practices**: Activity → rest → integration → activity
- **Living systems**: Metabolism → sleep → regeneration → metabolism

#### Variations
```python
# RECURSIVE regeneration
[
    "coherence", "resonance", "recursivity",  # Fractal renewal
    "transition", "emission", "coupling", "coherence"
]

# TRANSFORMATIVE regeneration
[
    "coherence", "resonance", "expansion",
    "transition", "transition",  # Multiple state changes
    "emission", "coupling", "coherence"
]
```

---

## Compositional Patterns

### 11. BOOTSTRAP

**Category**: Compositional  
**Coherence Weight**: 1.0  
**Structural Depth**: Low-Medium

#### Description
Rapid initialization sequence. Quick setup of new node or subsystem.

#### Signature
**Canonical**: `EMISSION → COUPLING → COHERENCE`  
**Max length**: 5 operators

#### Example
```python
["emission", "coupling", "coherence"]
["emission", "reception", "coupling", "coherence"]
```

#### Use Cases
- System initialization
- Node activation
- Quick setup
- Subsystem launch

---

### 12. EXPLORE

**Category**: Compositional  
**Coherence Weight**: 1.0  
**Structural Depth**: Low-Medium

#### Description
Controlled exploration sequence. Try new configuration with resolution.

#### Signature
**Canonical**: `DISSONANCE → MUTATION → COHERENCE`

#### Example
```python
["dissonance", "mutation", "coherence"]
["emission", "dissonance", "mutation", "coherence"]
```

#### Use Cases
- Exploration
- Testing alternatives
- Controlled experimentation
- Search processes

---

### 13. STABILIZE

**Category**: Compositional  
**Coherence Weight**: 1.0  
**Structural Depth**: Low-Medium

#### Description
Consolidation ending. Properly terminate and stabilize.

#### Signature
**Endings**: `COHERENCE → SILENCE` or `COHERENCE → RESONANCE`

#### Example
```python
["emission", "reception", "coherence", "silence"]
["emission", "coherence", "resonance"]
```

#### Use Cases
- Proper endings
- Consolidation
- Stabilization
- Rest periods

---

### 14. RESONATE

**Category**: Compositional  
**Coherence Weight**: 1.0  
**Structural Depth**: Low-Medium

#### Description
Amplification through coupling. Strengthen patterns via network.

#### Signature
**Canonical**: `RESONANCE → COUPLING → RESONANCE`

#### Example
```python
["resonance", "coupling", "resonance"]
["emission", "resonance", "coupling", "resonance"]
```

#### Use Cases
- Amplification
- Network propagation
- Pattern strengthening
- Viral spreading

---

### 15. COMPRESS

**Category**: Compositional  
**Coherence Weight**: 1.0  
**Structural Depth**: Low-Medium

#### Description
Simplification sequence. Reduce and stabilize.

#### Signature
**Canonical**: `CONTRACTION → COHERENCE → SILENCE`

#### Example
```python
["contraction", "coherence", "silence"]
["expansion", "contraction", "coherence", "silence"]
```

#### Use Cases
- Simplification
- Compression
- Reduction
- Minimization

---

## Complexity Patterns

### 16. COMPLEX

**Category**: Complexity  
**Coherence Weight**: 1.5  
**Structural Depth**: Variable

#### Description
Multiple patterns combined. Long sequence with diverse high-coherence pattern matches.

#### Signature
- Length > 8 operators
- Contains ≥3 pattern matches
- Multiple coherence levels

#### Example
```python
[
    "emission", "coupling", "coherence",        # BOOTSTRAP
    "dissonance", "mutation", "coherence",      # EXPLORE
    "resonance", "silence"                      # STABILIZE
]
```

---

### 17. MINIMAL

**Category**: Complexity  
**Coherence Weight**: 0.5  
**Structural Depth**: Very Low

#### Description
Single or very few operators. Simplest possible sequence.

#### Signature
- Length ≤ 1

#### Example
```python
["emission"]
["coherence"]
```

---

### 18. UNKNOWN

**Category**: Complexity  
**Coherence Weight**: 0.1  
**Structural Depth**: None

#### Description
Unclassified sequence. No clear pattern match.

#### Signature
- No other pattern matches
- Fallback classification

---

## Pattern Detection API

```python
from tnfr.operators.patterns import AdvancedPatternDetector
from tnfr.operators.grammar import validate_sequence_with_health

# Direct detection
detector = AdvancedPatternDetector()
pattern = detector.detect_pattern(sequence)
print(pattern.value)  # Pattern name

# Via validation
result = validate_sequence_with_health(sequence)
print(result.metadata['detected_pattern'])
```

---

## Resources

- **[GLYPH_SEQUENCES_GUIDE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/GLYPH_SEQUENCES_GUIDE.md)**: Complete guide
- **[HEALTH_METRICS_GUIDE.md](HEALTH_METRICS_GUIDE.md)**: Health metrics deep dive
- **[examples/domain_applications/](https://github.com/fermga/TNFR-Python-Engine/tree/main/examples/domain_applications)**: Pattern examples

---

*Last updated: 2025-11-07*  
*Grammar version: 2.0*
