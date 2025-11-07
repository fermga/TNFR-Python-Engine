# TNFR Operator Sequences Guide - Grammar 2.0

## Introduction

This guide documents **canonical operator sequences** for the TNFR paradigm - proven patterns of structural operators that produce coherent reorganization and controlled transformation. **Grammar 2.0** introduces advanced capabilities for sequence validation, health metrics, and pattern detection.

> **📖 Visual Companion**: See [docs/source/api/OPERATORS_VISUAL_GUIDE.md](docs/source/api/OPERATORS_VISUAL_GUIDE.md) for comprehensive visual documentation with ASCII diagrams, detailed explanations, and interactive examples for all 13 operators.
>
> **🔄 Migrating from Grammar 1.0?** See [MIGRATION_GUIDE_2.0.md](docs/MIGRATION_GUIDE_2.0.md) for breaking changes and upgrade path.

---

## What's New in Grammar 2.0

### 🎯 Major Enhancements

1. **Structural Frequencies (R5)**: Each operator has a defined structural frequency (νf) in Hz_str units
   - **High**: Initiation, amplification, concentration (EMISSION, DISSONANCE, RESONANCE, MUTATION, CONTRACTION)
   - **Medium**: Capture, stabilization, coupling, organization (RECEPTION, COHERENCE, COUPLING, EXPANSION, SELF_ORGANIZATION, TRANSITION, RECURSIVITY)
   - **Zero**: Suspended reorganization (SILENCE)

2. **Graduated Compatibility**: Three-level compatibility system
   - ✓ **COMPATIBLE**: Recommended transitions (green light)
   - ⚠ **CAUTION**: Context-specific transitions (warnings issued)
   - ✗ **INCOMPATIBLE**: Invalid transitions (validation fails)

3. **18 Structural Patterns**: Comprehensive pattern detection system
   - **Fundamental**: LINEAR, HIERARCHICAL, FRACTAL, CYCLIC, BIFURCATED
   - **Domain-Specific**: THERAPEUTIC, EDUCATIONAL, ORGANIZATIONAL, CREATIVE, REGENERATIVE
   - **Compositional**: BOOTSTRAP, EXPLORE, STABILIZE, RESONATE, COMPRESS
   - **Complexity**: COMPLEX, MINIMAL, UNKNOWN

4. **Health Metrics System**: Quantitative sequence assessment (0.0-1.0 scale)
   - **Coherence Index**: Global flow quality
   - **Balance Score**: Stabilizer/destabilizer equilibrium
   - **Sustainability Index**: Long-term maintenance capacity
   - **Complexity Efficiency**: Value-to-complexity ratio
   - **Frequency Harmony**: Structural frequency transition smoothness
   - **Pattern Completeness**: Cycle completion measure
   - **Transition Smoothness**: Valid transition ratio

5. **Regenerative Cycles (R5)**: Self-sustaining sequence validation
   - Minimum cycle length enforcement (5 operators)
   - Required regenerator operators (TRANSITION, RECURSIVITY, SILENCE)
   - Balanced stabilizers before/after regenerators
   - Structural health threshold (>0.6)

### 🔧 Breaking Changes

- **SELF_ORGANIZATION validation**: Now requires destabilizer (DISSONANCE, MUTATION, CONTRACTION) within 3-operator window
- **Pattern classification**: More specific patterns may change existing classifications
- **Validation warnings**: CAUTION-level transitions generate non-blocking warnings

### 🚀 Quick Start with Grammar 2.0

```python
from tnfr.operators.grammar import validate_sequence_with_health

# Validate a sequence with health metrics
result = validate_sequence_with_health([
    "emission",      # High frequency: initiate
    "reception",     # Medium: capture
    "coherence",     # Medium: stabilize
    "silence"        # Zero: pause
])

print(f"Valid: {result.passed}")
print(f"Pattern: {result.metadata['detected_pattern']}")
print(f"Health: {result.health_metrics.overall_health:.2f}")
print(f"Coherence: {result.health_metrics.coherence_index:.2f}")
print(f"Balance: {result.health_metrics.balance_score:.2f}")

# Check recommendations
if result.health_metrics.recommendations:
    print("Recommendations:")
    for rec in result.health_metrics.recommendations:
        print(f"  - {rec}")
```

---

## Grammar Fundamentals

### Composition Principles

1. **Operational Closure**: Valid sequences preserve TNFR system closure
2. **Structural Coherence**: Transitions maintain C(t) > threshold
3. **ΔNFR Balance**: Sequences balance creation/reduction of ΔNFR
4. **Phase Preservation**: θ maintains structural continuity
5. **Frequency Harmony**: Structural frequency transitions respect νf harmonics (R5)

### Validation Rules (R1-R5)

#### R1: Sequence Structure
- Must start with valid initiator: EMISSION, RECEPTION, TRANSITION, SILENCE, COHERENCE, SELF_ORGANIZATION
- Must end with valid terminator: COHERENCE, SILENCE, RESONANCE, COUPLING, SELF_ORGANIZATION, RECURSIVITY, CONTRACTION, MUTATION

#### R2: Operator Compatibility
- Consecutive operators must be compatible (no ✗ INCOMPATIBLE transitions)
- CAUTION transitions (○) generate warnings but don't block validation

#### R3: SELF_ORGANIZATION Preconditions
- SELF_ORGANIZATION requires destabilizer within 3-operator window before it
- Destabilizers: DISSONANCE, MUTATION, CONTRACTION

#### R4: Sequence Patterns
- No consecutive identical operators (except TRANSITION, RECURSIVITY)
- No forbidden patterns (e.g., SILENCE → DISSONANCE)

#### R5: Advanced Rules (New in 2.0)
- **Frequency Harmony**: Zero → High transitions must pass through Medium
- **Regenerative Cycles**: Cycles require regenerators and balanced stabilizers

### Sequence Notation

- `→` Direct transition | `|` Alternatives | `()` Optional | `[]` Repeatable | `*` Any repetition

---

## Structural Frequencies (νf)

Each operator has an intrinsic **structural frequency** (νf) measured in **Hz_str** (structural hertz) - the rate at which it reorganizes node structure. Understanding frequencies is crucial for creating harmonic sequences.

### Frequency Categories

| Category | Hz_str | Operators | Structural Effect |
|----------|--------|-----------|-------------------|
| **High** | νf > 0.8 | EMISSION (AL), DISSONANCE (OZ), RESONANCE (RA), MUTATION (ZHIR), CONTRACTION (NUL) | Initiate, amplify, concentrate, pivot structure |
| **Medium** | 0.3 < νf ≤ 0.8 | RECEPTION (EN), COHERENCE (IL), COUPLING (UM), EXPANSION (VAL), SELF_ORGANIZATION (THOL), TRANSITION (NAV), RECURSIVITY (REMESH) | Capture, stabilize, couple, organize, hand-off |
| **Zero** | νf ≈ 0 | SILENCE (SHA) | Suspend reorganization while preserving form |

### Frequency Transition Rules (R5)

Valid structural frequency transitions preserve coherence:

- ✅ **High ↔ Medium**: Bidirectional energy exchange (natural)
- ✅ **Medium ↔ Zero**: Stabilization can pause, pause can resume
- ✅ **High ↔ High**: High-energy operators can chain directly
- ⚠️ **Zero → High**: **INVALID** - must pass through Medium first

```python
# Valid frequency transitions
["silence", "coherence", "emission"]     # Zero → Medium → High ✅
["emission", "dissonance", "resonance"]  # High → High → High ✅
["reception", "coherence", "silence"]    # Medium → Medium → Zero ✅

# Invalid frequency transition (generates warning)
["silence", "emission"]                  # Zero → High ⚠️
# Fix: Add medium operator
["silence", "transition", "emission"]    # Zero → Medium → High ✅
```

### Structural Frequency Example

```python
from tnfr.operators.grammar import validate_frequency_transition, STRUCTURAL_FREQUENCIES

# Check frequency compatibility
prev_op = "silence"     # Zero frequency
next_op = "emission"    # High frequency

is_valid, message = validate_frequency_transition(prev_op, next_op)
print(f"Valid: {is_valid}")  # False
print(f"Message: {message}")  # Explains the frequency mismatch

# Query operator frequencies
for op in ["emission", "coherence", "silence"]:
    freq = STRUCTURAL_FREQUENCIES[op]
    print(f"{op}: {freq} Hz_str")
```

**Key Insight**: Frequency transitions aren't arbitrary - they reflect the energetic continuity of structural reorganization. A zero-frequency pause (SILENCE) cannot instantly jump to high-energy initiation (EMISSION) without intermediate stabilization.

---

## Fundamental Sequences

### 1. Basic Activation: EMISSION → COHERENCE

**Frequency Profile**: High → Medium (✅ natural transition)

**Context**: Initiation and immediate stabilization of latent node

**Applications**:
- Meditation: Practice initiation → coherence establishment
- Therapy: Therapeutic space activation → frame stabilization
- Learning: Attention activation → sustained focus

**Structural Effects**:
- EPI: 0.2 → 0.5 → 0.52 (activation and stabilization)
- ΔNFR: +0.15 → +0.03 (initial peak, then reduction)
- C(t): +0.2 (global coherence increase)
- νf: High → Medium (smooth energy transition)

**Health Metrics** (Grammar 2.0):
- Overall health: ~0.75 (good)
- Pattern: MINIMAL or BOOTSTRAP (if extended with COUPLING)
- Balance: Excellent (stabilizer follows initiator)

```python
from tnfr.structural import create_nfr, run_sequence
from tnfr.operators.definitions import Emission, Coherence
from tnfr.operators.grammar import validate_sequence_with_health

# Create and execute sequence
G, node = create_nfr("meditation_start", epi=0.2, vf=0.85)
run_sequence(G, node, [Emission(), Coherence()])

# Validate with health metrics
result = validate_sequence_with_health(["emission", "coherence"])
print(f"Health: {result.health_metrics.overall_health:.2f}")
print(f"Pattern: {result.metadata['detected_pattern']}")
```

---

### 2. Stabilized Reception: RECEPTION → COHERENCE

**Frequency Profile**: Medium → Medium (✅ stable transition)

**Context**: Integration and consolidation of external information

**Applications**:
- Biofeedback: Signal received → integrated into physiology
- Education: Concept received → integrated into mental model
- Communication: Message received → understood and accepted

**Structural Effects**:
- EPI: +0.1 (integration), then stabilization
- ΔNFR: Reduced through successful integration
- Network coupling: Strengthened with emitting source

```python
G, student = create_nfr("learning_reception", epi=0.30, vf=0.95)
run_sequence(G, student, [Reception(), Coherence()])
# Result: Information integrated and stabilized in memory
```

---

### 3. Coupled Propagation: UM → RA

**Context**: Synchronization followed by network resonance

**Applications**:
- Cardiac coherence: Heart-brain → whole body
- Collective insight: Synchronized pair → full team
- Social movement: Aligned core → broader community

**Structural Effects**:
- θ: Phase convergence between nodes
- Propagation: EPI extends through network
- Global C(t): Significant increase

```python
G, community = create_nfr("social_network", vf=1.10, theta=0.40)
run_sequence(G, community, [Coupling(), Resonance()])
# Result: Coherence propagates through coupled network
```

---

## Intermediate Sequences

### 4. Transformation Cycle: AL → NAV → IL

**Context**: Activation with controlled transition before stabilization

**Applications**:
- Organizational change: Start → transition → new stability
- Personal transformation: Decision → process → integration
- Innovation: Idea → development → product

```python
G, org = create_nfr("company_transform", epi=0.35, vf=0.90, theta=0.25)
run_sequence(G, org, [Emission(), Transition(), Coherence()])
# Result: Organizational change completed and stabilized
```

---

### 5. Creative Resolution: OZ → IL

**Context**: Generative dissonance followed by emergent coherence

**Applications**:
- Therapy: Emotional crisis → transformative integration
- Science: Experimental anomaly → new paradigm
- Art: Creative chaos → coherent form

```python
G, patient = create_nfr("therapeutic_crisis", epi=0.45, theta=0.15)
run_sequence(G, patient, [Dissonance(), Coherence()])
# Result: Dissonance resolved into new personal coherence
```

---

### 6. Emergent Self-Organization: OZ → THOL

**Context**: Dissonance catalyzes autonomous reorganization

**Applications**:
- Complex systems: Perturbation → self-organization
- Deep learning: Confusion → emergent insight
- Ecosystems: Disturbance → new configuration

```python
G, ecosystem = create_nfr("complex_system", epi=0.55, vf=1.05)
run_sequence(G, ecosystem, [Dissonance(), SelfOrganization()])
# Result: New organization emerges from perturbation
```

---

## Advanced Sequences

### 7. Complete Reorganization Cycle: AL → NAV → IL → OZ → THOL → RA → UM

**Context**: Integral deep structural transformation process

**Applications**:
- Complete personal development
- Transformative organizational innovation
- Complex system evolution
- Multi-level deep healing

**Phases**:

1. **AL (Emission)**: Process initiation, node activation
2. **NAV (Transition)**: Movement toward new regime
3. **IL (Coherence)**: Transient form stabilization
4. **OZ (Dissonance)**: Creative challenge, exploration
5. **THOL (Self-organization)**: New structure emergence
6. **RA (Resonance)**: New coherence propagation
7. **UM (Coupling)**: Final synchronization

**Net Effects**:
- EPI: Qualitative transformation (new regime)
- C(t): Significant post-reorganization increase
- Si (Sense Index): Substantial improvement
- Network: Emergent new topology

```python
from tnfr.operators.definitions import (
    Emission, Transition, Coherence, Dissonance,
    SelfOrganization, Resonance, Coupling
)

G, person = create_nfr("deep_transformation", epi=0.20, vf=0.80, theta=0.30)

sequence = [
    Emission(),          # Conscious process initiation
    Transition(),        # Change preparation
    Coherence(),         # Preparatory stabilization
    Dissonance(),        # Shadow/trauma confrontation
    SelfOrganization(),  # New identity emergence
    Resonance(),         # Life-area propagation
    Coupling()           # Social/family integration
]

run_sequence(G, person, sequence)
# Result: Deep and sustainable personal transformation
```

---

### 8. Propagative Resonance Sequence: AL → RA → EN → IL

**Context**: Emission that propagates and is received with stabilization

**Applications**:
- Effective teaching: Teacher emits → propagates → students receive → integrate
- Organizational communication: Leadership communicates → propagates → team integrates
- Cultural transmission: Tradition emitted → propagated → receiving generation

```python
G_class, teaching = create_nfr("classroom_teaching", epi=0.25, vf=1.00)
run_sequence(G_class, teaching, [
    Emission(),    # Teacher presents concept
    Resonance(),   # Concept resonates in classroom
    Reception(),   # Students actively receive
    Coherence()    # Understanding consolidates
])
# Result: Effective and consolidated learning
```

---

### 9. Controlled Mutation: IL → ZHIR → IL

**Context**: Phase change stabilized before and after

**Applications**:
- Personal paradigm shift
- Organizational pivot
- System phase transition

```python
G, company = create_nfr("strategic_pivot", epi=0.60, theta=0.25)
run_sequence(G, company, [
    Coherence(),   # Stabilize current position
    Mutation(),    # Execute pivot (phase change)
    Coherence()    # Stabilize new direction
])
# Result: Controlled strategic transformation
```

---

## Structural Patterns (Grammar 2.0)

Grammar 2.0 introduces a comprehensive **18-pattern typology** for detecting and classifying operator sequences. Each pattern has a specific structural signature and coherence weight reflecting its complexity.

### Pattern Classification System

Patterns are detected using **coherence-weighted scoring**: each pattern is evaluated independently, then weighted by its structural depth (emergence, self-organization, phase transitions). This ensures that emergent patterns are appropriately recognized over simple compositional patterns.

### Fundamental Patterns (5)

#### 1. LINEAR
**Signature**: Simple progression without transformation  
**Characteristics**:
- No DISSONANCE, MUTATION, or SELF_ORGANIZATION
- Maximum 5 operators
- Coherence weight: 0.5 (simple)

**Example**:
```python
["emission", "reception", "coherence"]  # Basic activation
```

#### 2. HIERARCHICAL
**Signature**: Self-organization creates nested structure  
**Characteristics**:
- Contains SELF_ORGANIZATION
- Implies emergent sub-structures
- Coherence weight: 2.0 (structural transformation)

**Example**:
```python
["dissonance", "self_organization", "coherence"]  # Emergence from instability
```

#### 3. FRACTAL
**Signature**: Recursive structure across scales  
**Characteristics**:
- Requires TRANSITION
- Contains COUPLING or RECURSIVITY
- Implies scale-invariant patterns
- Coherence weight: 2.0 (recursive)

**Example**:
```python
["transition", "coupling", "transition", "recursivity"]  # Multi-scale coordination
```

#### 4. CYCLIC
**Signature**: Regenerative loops  
**Characteristics**:
- Multiple TRANSITION operators (≥2)
- Implies cyclic reorganization
- Coherence weight: 2.0 (regenerative)

**Example**:
```python
["transition", "emission", "coherence", "transition", "resonance"]  # Cycle
```

#### 5. BIFURCATED
**Signature**: Branching through phase transition  
**Characteristics**:
- Contains DISSONANCE → MUTATION or DISSONANCE → CONTRACTION pairs
- Implies possibility space branching
- Coherence weight: 2.0 (phase transition)

**Example**:
```python
["coherence", "dissonance", "mutation", "coherence"]  # Controlled branching
```

---

### Domain-Specific Patterns (5)

These patterns reflect specialized application domains with full structural cycles.

#### 6. THERAPEUTIC
**Signature**: Complete healing cycle with controlled crisis resolution  
**Characteristics**:
- Sequence: RECEPTION → EMISSION → COHERENCE → DISSONANCE → SELF_ORGANIZATION → COHERENCE
- Coherence weight: 3.0 (emergent self-organizing)

**Example**:
```python
from examples.domain_applications.therapeutic_patterns import get_therapeutic_sequence
sequence = get_therapeutic_sequence()
# ["reception", "emission", "coherence", "dissonance", "self_organization", "coherence"]
```

**Applications**:
- Psychotherapy: Crisis → integration → new coherence
- Trauma healing: Activation → confrontation → reorganization
- Emotional regulation: Distress → processing → stabilization

#### 7. EDUCATIONAL
**Signature**: Transformative learning with phase shift  
**Characteristics**:
- Sequence: RECEPTION → EMISSION → COHERENCE → EXPANSION → DISSONANCE → MUTATION
- Coherence weight: 3.0 (transformative)

**Example**:
```python
from examples.domain_applications.educational_patterns import get_conceptual_breakthrough_sequence
sequence = get_conceptual_breakthrough_sequence()
# Includes expansion of understanding and mutation of mental model
```

**Applications**:
- Conceptual breakthrough: Understanding → challenge → paradigm shift
- Skill acquisition: Practice → mastery transition
- Knowledge integration: Facts → systemic understanding

#### 8. ORGANIZATIONAL
**Signature**: Institutional evolution with emergent reorganization  
**Characteristics**:
- Sequence: TRANSITION → EMISSION → RECEPTION → COUPLING → RESONANCE → DISSONANCE → SELF_ORGANIZATION
- Coherence weight: 3.0 (organizational emergence)

**Example**:
```python
from examples.domain_applications.organizational_patterns import get_cultural_transformation_sequence
sequence = get_cultural_transformation_sequence()
# Institutional change with self-organizing dynamics
```

**Applications**:
- Cultural transformation: Old structure → tension → emergent culture
- Strategic pivots: Current state → crisis → new configuration
- Team evolution: Group → conflict → self-organized collaboration

#### 9. CREATIVE
**Signature**: Artistic emergence through self-organization  
**Characteristics**:
- Sequence: SILENCE → EMISSION → EXPANSION → DISSONANCE → MUTATION → SELF_ORGANIZATION
- Coherence weight: 3.0 (creative emergence)

**Example**:
```python
from examples.domain_applications.creative_patterns import get_creative_process_sequence
sequence = get_creative_process_sequence()
# Pause → inspiration → exploration → breakthrough → form
```

**Applications**:
- Artistic creation: Void → inspiration → experimentation → work
- Innovation: Constraint → ideation → synthesis → product
- Design process: Research → prototyping → iteration → solution

#### 10. REGENERATIVE
**Signature**: Self-sustaining cycle with autonomous renewal  
**Characteristics**:
- Long sequence (≥9 operators) with balanced cycle
- Contains regenerators (TRANSITION, RECURSIVITY, SILENCE)
- Coherence weight: 3.0 (self-sustaining)

**Example**:
```python
["coherence", "resonance", "expansion", "silence", "transition", 
 "emission", "reception", "coupling", "coherence"]
# Complete regenerative cycle
```

**Applications**:
- Ecosystem dynamics: Growth → dormancy → renewal → growth
- Institutional sustainability: Operation → reflection → adaptation → operation
- Personal practices: Activity → rest → integration → activity

---

### Compositional Patterns (5)

Building blocks for larger sequences.

#### 11. BOOTSTRAP
**Signature**: Rapid initialization  
**Characteristics**:
- Sequence: EMISSION → COUPLING → COHERENCE
- Maximum 5 operators
- Coherence weight: 1.0 (compositional)

**Example**:
```python
["emission", "coupling", "coherence"]  # Start new node or subsystem
```

#### 12. EXPLORE
**Signature**: Controlled exploration  
**Characteristics**:
- Sequence: DISSONANCE → MUTATION → COHERENCE
- Coherence weight: 1.0 (compositional)

**Example**:
```python
["dissonance", "mutation", "coherence"]  # Try new configuration
```

#### 13. STABILIZE
**Signature**: Consolidation ending  
**Characteristics**:
- Ends with COHERENCE → SILENCE or COHERENCE → RESONANCE
- Coherence weight: 1.0 (compositional)

**Example**:
```python
["emission", "reception", "coherence", "silence"]  # Consolidate and pause
```

#### 14. RESONATE
**Signature**: Amplification through coupling  
**Characteristics**:
- Sequence: RESONANCE → COUPLING → RESONANCE
- Coherence weight: 1.0 (compositional)

**Example**:
```python
["resonance", "coupling", "resonance"]  # Amplify through network
```

#### 15. COMPRESS
**Signature**: Simplification sequence  
**Characteristics**:
- Sequence: CONTRACTION → COHERENCE → SILENCE
- Coherence weight: 1.0 (compositional)

**Example**:
```python
["contraction", "coherence", "silence"]  # Simplify and stabilize
```

---

### Complexity Patterns (3)

#### 16. COMPLEX
**Signature**: Multiple patterns combined  
**Characteristics**:
- Long sequence (>8 operators)
- Contains ≥3 pattern matches with diverse coherence levels
- Coherence weight: 1.5 (moderate complexity)

**Example**:
```python
# A sequence showing BOOTSTRAP + EXPLORE + STABILIZE
["emission", "coupling", "coherence",     # BOOTSTRAP
 "dissonance", "mutation", "coherence",   # EXPLORE
 "resonance", "silence"]                  # STABILIZE
```

#### 17. MINIMAL
**Signature**: Single or very few operators  
**Characteristics**:
- Length ≤ 1
- Coherence weight: 0.5 (simple)

**Example**:
```python
["emission"]  # Single operator
```

#### 18. UNKNOWN
**Signature**: Unclassified sequences  
**Characteristics**:
- No clear pattern match
- Fallback classification
- Coherence weight: 0.1 (lowest)

---

### Pattern Detection Example

```python
from tnfr.operators.grammar import validate_sequence_with_health
from tnfr.operators.patterns import AdvancedPatternDetector

# Therapeutic sequence
therapeutic = [
    "reception", "emission", "coherence",
    "dissonance", "self_organization", "coherence"
]

result = validate_sequence_with_health(therapeutic)
print(f"Pattern: {result.metadata['detected_pattern']}")  # THERAPEUTIC
print(f"Health: {result.health_metrics.overall_health:.2f}")

# Direct pattern detection
detector = AdvancedPatternDetector()
pattern = detector.detect_pattern(therapeutic)
print(f"Detected: {pattern.value}")  # therapeutic
```

---

## Anti-patterns (Sequences to Avoid)

### ❌ SHA → OZ (Silence followed by Dissonance)

**Problem**: Contradicts silence purpose (preservation)

**Why it fails**: SHA reduces νf to preserve EPI, but OZ immediately increases ΔNFR, creating reorganization pressure that violates SHA intention.

**Correct alternative**: SHA → NAV → OZ (controlled transition before challenge)

---

### ❌ OZ → OZ (Consecutive Dissonance)

**Problem**: Excessive instability without resolution

**Why it fails**: Cumulative ΔNFR without reduction → structural collapse, not creative reorganization.

**Correct alternative**: OZ → IL → OZ (resolve between dissonances)

---

### ❌ SHA → SHA (Redundant Silence)

**Problem**: No structural purpose

**Why it fails**: Second SHA adds no effect if νf already ≈ 0.

**Correct alternative**: SHA → AL (reactivation) or SHA → NAV (transition)

---

### ❌ AL → SHA (Activation immediately silenced)

**Problem**: Contradicts activation purpose

**Why it fails**: Activating a node to immediately silence it is structurally inefficient and contradictory.

**Correct alternative**: AL → IL (activate and stabilize) or AL alone

---

## Operator Compatibility Matrix

| Post \ Pre | AL | EN | IL | OZ | UM | RA | SHA | VAL | NUL | THOL | ZHIR | NAV | REMESH |
|-----------|----|----|----|----|----|----|-----|-----|-----|------|------|-----|--------|
| **AL**    | ○  | ✓  | ✓  | ✓  | ○  | ✓  | ✓   | ○   | ○   | ○    | ○    | ✓   | ○      |
| **EN**    | ✓  | ○  | ✓  | ○  | ✓  | ✓  | ○   | ○   | ○   | ✓    | ○    | ✓   | ○      |
| **IL**    | ✓  | ✓  | ○  | ✓  | ✓  | ✓  | ✓   | ✓   | ✓   | ✓    | ✓    | ✓   | ✓      |
| **OZ**    | ○  | ○  | ✓  | ✗  | ○  | ○  | ✗   | ○   | ○   | ✓    | ✓    | ✓   | ○      |
| **UM**    | ✓  | ✓  | ✓  | ○  | ○  | ✓  | ○   | ○   | ○   | ✓    | ○    | ✓   | ○      |
| **RA**    | ○  | ✓  | ✓  | ○  | ✓  | ○  | ○   | ○   | ○   | ✓    | ○    | ✓   | ✓      |
| **SHA**   | ✓  | ○  | ✓  | ✗  | ○  | ○  | ✗   | ○   | ○   | ○    | ○    | ✓   | ○      |
| **VAL**   | ○  | ○  | ✓  | ✓  | ○  | ✓  | ○   | ○   | ✗   | ✓    | ○    | ✓   | ✓      |
| **NUL**   | ○  | ○  | ✓  | ○  | ○  | ○  | ○   | ✗   | ○   | ○    | ○    | ✓   | ○      |
| **THOL**  | ○  | ○  | ✓  | ○  | ✓  | ✓  | ○   | ○   | ○   | ○    | ✓    | ✓   | ✓      |
| **ZHIR**  | ○  | ○  | ✓  | ○  | ○  | ○  | ○   | ○   | ○   | ○    | ○    | ✓   | ○      |
| **NAV**   | ✓  | ✓  | ✓  | ✓  | ✓  | ✓  | ○   | ✓   | ✓   | ✓    | ✓    | ○   | ✓      |
| **REMESH**| ○  | ○  | ✓  | ○  | ○  | ✓  | ○   | ✓   | ○   | ✓    | ○    | ✓   | ○      |

**Legend**: ✓ Highly compatible (recommended) | ○ Compatible (context-specific) | ✗ Incompatible (avoid)

---

## Structural Health Metrics (Grammar 2.0)

Grammar 2.0 introduces **quantitative health assessment** for operator sequences. All metrics range from 0.0 (poor) to 1.0 (excellent), providing objective measures of sequence quality.

### Seven Health Dimensions

#### 1. Coherence Index (0.0-1.0)
**Measures**: Global sequential flow quality  
**Factors**:
- Valid transition ratio
- Recognizable pattern structure
- Proper structural closure (ending)

**Interpretation**:
- **>0.7**: Excellent flow, clear pattern
- **0.4-0.7**: Moderate flow, recognizable structure
- **<0.4**: Poor flow, unclear structure

**Example**:
```python
# High coherence: clear activation pattern
["emission", "reception", "coherence", "silence"]  # ~0.85

# Low coherence: random transitions
["expansion", "contraction", "mutation"]  # ~0.35
```

#### 2. Balance Score (0.0-1.0)
**Measures**: Equilibrium between stabilizers and destabilizers  
**Factors**:
- Ratio of stabilizers (COHERENCE, SELF_ORGANIZATION, SILENCE, RESONANCE)
- Ratio of destabilizers (DISSONANCE, MUTATION, CONTRACTION)
- Ideal: balanced structural forces

**Interpretation**:
- **>0.8**: Excellent balance
- **0.5-0.8**: Good balance
- **<0.5**: Imbalanced (too many destabilizers or too passive)

**Example**:
```python
# Balanced
["dissonance", "coherence", "mutation", "coherence"]  # ~0.8

# Imbalanced: too much destabilization
["dissonance", "mutation", "contraction"]  # ~0.3
```

#### 3. Sustainability Index (0.0-1.0)
**Measures**: Capacity for long-term maintenance  
**Factors**:
- Final stabilization present
- Dissonance resolved
- Regenerative elements (TRANSITION, RECURSIVITY, SILENCE)

**Interpretation**:
- **>0.7**: Highly sustainable
- **0.4-0.7**: Moderately sustainable
- **<0.4**: Unsustainable, unstable ending

**Example**:
```python
# Sustainable: ends with stabilizer
["emission", "dissonance", "coherence", "silence"]  # ~0.85

# Unsustainable: ends with destabilizer
["emission", "coherence", "dissonance"]  # ~0.25
```

#### 4. Complexity Efficiency (0.0-1.0)
**Measures**: Value-to-complexity ratio  
**Factors**:
- Sequence length vs structural value
- Redundancy penalty
- Pattern clarity bonus

**Interpretation**:
- **>0.7**: Efficient, minimal redundancy
- **0.4-0.7**: Acceptable efficiency
- **<0.4**: Inefficient, unnecessarily complex

**Example**:
```python
# Efficient: concise pattern
["emission", "coherence"]  # ~0.9

# Inefficient: redundant operators
["emission", "emission", "coherence", "coherence", "silence", "silence"]  # ~0.3
```

#### 5. Frequency Harmony (0.0-1.0)
**Measures**: Structural frequency transition smoothness (R5)  
**Factors**:
- Valid frequency transitions (High ↔ Medium, Medium ↔ Zero)
- Frequency gradient continuity
- Harmonic flow

**Interpretation**:
- **>0.8**: Excellent frequency harmony
- **0.5-0.8**: Good harmony, some minor issues
- **<0.5**: Poor harmony, incoherent jumps

**Example**:
```python
# Harmonic: Zero → Medium → High
["silence", "coherence", "emission"]  # ~0.95

# Inharmonic: Zero → High
["silence", "emission"]  # ~0.4 (warning issued)
```

#### 6. Pattern Completeness (0.0-1.0)
**Measures**: How complete the detected pattern is  
**Factors**:
- Full cycle vs partial cycle
- Required elements present
- Pattern-specific completeness criteria

**Interpretation**:
- **>0.8**: Complete pattern, full cycle
- **0.5-0.8**: Partial pattern, recognizable
- **<0.5**: Incomplete, fragmented

**Example**:
```python
# Complete therapeutic pattern
therapeutic_full = [
    "reception", "emission", "coherence",
    "dissonance", "self_organization", "coherence"
]  # ~0.95

# Partial therapeutic pattern
therapeutic_partial = [
    "reception", "emission", "coherence"
]  # ~0.5
```

#### 7. Transition Smoothness (0.0-1.0)
**Measures**: Quality of operator transitions  
**Factors**:
- Ratio of valid transitions to total transitions
- COMPATIBLE vs CAUTION vs INCOMPATIBLE
- Flow continuity

**Interpretation**:
- **>0.8**: Excellent transitions, all compatible
- **0.5-0.8**: Good transitions, some CAUTION
- **<0.5**: Poor transitions, incompatibilities

**Example**:
```python
# Smooth: all compatible
["emission", "reception", "coherence"]  # ~0.95

# Rough: has CAUTION transition
["emission", "silence"]  # ~0.6 (CAUTION)
```

---

### Overall Health Score

The **overall_health** metric is a weighted average of all dimensions:

```python
overall_health = (
    coherence_index * 0.20 +
    balance_score * 0.20 +
    sustainability_index * 0.20 +
    complexity_efficiency * 0.15 +
    frequency_harmony * 0.10 +
    pattern_completeness * 0.10 +
    transition_smoothness * 0.05
)
```

**Interpretation**:
- **>0.80**: Excellent sequence quality
- **0.65-0.80**: Good sequence quality
- **0.50-0.65**: Acceptable, room for improvement
- **<0.50**: Poor quality, needs revision

---

### Health Analysis Example

```python
from tnfr.operators.grammar import validate_sequence_with_health

# Analyze a therapeutic sequence
sequence = [
    "reception", "emission", "coherence",
    "dissonance", "self_organization", "coherence"
]

result = validate_sequence_with_health(sequence)
health = result.health_metrics

print(f"Overall Health: {health.overall_health:.2f}")
print(f"Coherence: {health.coherence_index:.2f}")
print(f"Balance: {health.balance_score:.2f}")
print(f"Sustainability: {health.sustainability_index:.2f}")
print(f"Efficiency: {health.complexity_efficiency:.2f}")
print(f"Frequency: {health.frequency_harmony:.2f}")
print(f"Completeness: {health.pattern_completeness:.2f}")
print(f"Smoothness: {health.transition_smoothness:.2f}")
print(f"\nPattern: {health.dominant_pattern}")

# Check recommendations
if health.recommendations:
    print("\nRecommendations:")
    for rec in health.recommendations:
        print(f"  - {rec}")
```

**Sample Output**:
```
Overall Health: 0.88
Coherence: 0.92
Balance: 0.85
Sustainability: 0.90
Efficiency: 0.88
Frequency: 0.95
Completeness: 0.95
Smoothness: 0.90

Pattern: therapeutic

Recommendations:
  []  # No recommendations - excellent sequence!
```

---

### Health Optimization Strategies

#### Strategy 1: Improve Balance
**Problem**: Too many destabilizers without resolution  
**Solution**: Add stabilizers after destabilizing operators

```python
# Before (balance ~0.4)
["dissonance", "mutation", "contraction"]

# After (balance ~0.8)
["dissonance", "coherence", "mutation", "coherence"]
```

#### Strategy 2: Improve Sustainability
**Problem**: Sequence ends with destabilizer  
**Solution**: Add final stabilizer

```python
# Before (sustainability ~0.3)
["emission", "coherence", "dissonance"]

# After (sustainability ~0.85)
["emission", "coherence", "dissonance", "coherence", "silence"]
```

#### Strategy 3: Improve Frequency Harmony
**Problem**: Zero → High jump  
**Solution**: Add Medium frequency bridge

```python
# Before (frequency ~0.4)
["silence", "emission"]

# After (frequency ~0.95)
["silence", "transition", "emission"]
```

#### Strategy 4: Improve Completeness
**Problem**: Partial pattern  
**Solution**: Complete the structural cycle

```python
# Before (completeness ~0.5)
["reception", "emission", "coherence"]

# After (completeness ~0.95)
["reception", "emission", "coherence", 
 "dissonance", "self_organization", "coherence"]  # Full therapeutic
```

---

### Recommendations System

The health analyzer provides **actionable recommendations** when issues are detected:

| Health Issue | Recommendation Example |
|--------------|----------------------|
| Low balance | "Add stabilizer after destabilizer at position X" |
| Low sustainability | "End sequence with stabilizer (coherence, silence, or resonance)" |
| Low frequency | "Insert medium-frequency operator between silence and emission" |
| Low completeness | "Complete pattern by adding [missing operators]" |
| Low efficiency | "Remove redundant operator at position X" |

**Example**:
```python
sequence = ["dissonance", "mutation", "dissonance"]  # Imbalanced
result = validate_sequence_with_health(sequence)

print(result.health_metrics.recommendations)
# ["Add stabilizer between destabilizers for better balance",
#  "End with stabilizer for sustainability"]
```

---

## Regenerative Cycles (R5)

Grammar 2.0 introduces **regenerative cycle validation** - ensuring that self-sustaining sequences meet structural requirements for autonomous renewal.

### What is a Regenerative Cycle?

A regenerative cycle is a sequence that:
1. **Sustains itself**: Has balanced structural forces
2. **Renews autonomously**: Contains regenerator operators
3. **Maintains coherence**: Achieves minimum health threshold
4. **Completes structure**: Forms a recognizable cyclic pattern

### R5 Validation Rules

#### Rule 1: Minimum Length
- Cycles must have ≥5 operators
- Maximum: 13 operators (all canonical operators once)

#### Rule 2: Regenerator Requirement
At least one **regenerator operator** must be present:
- **TRANSITION (NAV)**: Controlled state transitions
- **RECURSIVITY (REMESH)**: Fractal renewal across scales
- **SILENCE (SHA)**: Pause for integration before renewal

#### Rule 3: Balanced Stabilizers
- Must have stabilizers **before** regenerator (preparation)
- Must have stabilizers **after** regenerator (consolidation)
- Balance score: `stabilizers_before / stabilizers_after` should be ≈1.0

#### Rule 4: Minimum Health
- Overall health score must be >0.6
- Ensures structural viability of the cycle

### Regenerator Operators

#### TRANSITION (NAV)
**Function**: Controlled hand-off between states  
**Frequency**: Medium (νf ~0.5)  
**Use in cycles**: State transition points

```python
["coherence", "transition", "emission", "coherence"]  # State change
```

#### RECURSIVITY (REMESH)
**Function**: Fractal echo across scales  
**Frequency**: Medium (νf ~0.5)  
**Use in cycles**: Multi-scale renewal

```python
["transition", "recursivity", "coupling"]  # Fractal regeneration
```

#### SILENCE (SHA)
**Function**: Pause for integration  
**Frequency**: Zero (νf ≈0)  
**Use in cycles**: Rest before renewal

```python
["resonance", "silence", "transition", "emission"]  # Rest → renew
```

### Regenerative Cycle Example

```python
from tnfr.operators.grammar import validate_sequence_with_health

# Complete regenerative cycle
regenerative = [
    "coherence",      # Stabilizer before
    "resonance",      # Amplification
    "expansion",      # Growth
    "silence",        # ⭐ REGENERATOR: Pause for integration
    "transition",     # ⭐ REGENERATOR: State transition
    "emission",       # Renewal
    "reception",      # Input
    "coupling",       # Network coordination
    "coherence"       # Stabilizer after
]

result = validate_sequence_with_health(regenerative)

print(f"Pattern: {result.metadata['detected_pattern']}")  # REGENERATIVE
print(f"Health: {result.health_metrics.overall_health:.2f}")
print(f"Sustainability: {result.health_metrics.sustainability_index:.2f}")

# Check R5 validation
assert result.passed, "R5 validation failed"
assert result.metadata['detected_pattern'] == 'regenerative'
```

### Cycle Types

Based on dominant regenerator:

| Cycle Type | Dominant Regenerator | Characteristic |
|------------|---------------------|----------------|
| **TRANSFORMATIVE** | TRANSITION | Phase transitions, state changes |
| **RECURSIVE** | RECURSIVITY | Fractal renewal, multi-scale |
| **MEDITATIVE** | SILENCE | Paused renewal, integration |

### Invalid Regenerative Cycles

```python
# ❌ Too short (< 5 operators)
["coherence", "silence", "emission"]  # Only 3 operators

# ❌ No regenerator
["emission", "coherence", "resonance", "coupling", "coherence"]

# ❌ Unbalanced (no stabilizer before regenerator)
["dissonance", "transition", "emission", "coherence"]

# ❌ Low health (<0.6)
["dissonance", "silence", "mutation", "contraction", "emission"]  # Health ~0.35
```

### R5 Validation Example

```python
from tnfr.operators.cycle_detection import CycleDetector

detector = CycleDetector()

# Find regenerator position
sequence = ["coherence", "resonance", "silence", "transition", "emission", "coherence"]
regenerator_index = sequence.index("transition")  # Position 3

# Analyze cycle
analysis = detector.analyze_potential_cycle(sequence, regenerator_index)

print(f"Valid: {analysis.is_valid_regenerative}")
print(f"Type: {analysis.cycle_type.value}")
print(f"Health: {analysis.health_score:.2f}")
print(f"Balance: {analysis.balance_score:.2f}")
print(f"Stabilizers before: {analysis.stabilizer_count_before}")
print(f"Stabilizers after: {analysis.stabilizer_count_after}")
```

---

## Multi-Domain Examples

### Biomedical Domain

#### Cardiac Coherence Training Protocol
```python
G, heart = create_nfr("cardiac_training", epi=0.25, vf=0.85)

# Phase 1: Activation with conscious breathing
run_sequence(G, heart, [Emission()])

# Phase 2: Heart rhythm stabilization
run_sequence(G, heart, [Coherence()])

# Phase 3: Nervous system propagation
run_sequence(G, heart, [Resonance()])

# Phase 4: Heart-brain coupling
run_sequence(G, heart, [Coupling()])

# Phase 5: Final coherence stabilization
run_sequence(G, heart, [Coherence()])

# Result: Sustainable cardiac coherence state
# Benefits: Stress reduction, mental clarity, autonomic balance
```

---

### Cognitive/Educational Domain

#### Deep Learning Process
```python
G, learner = create_nfr("deep_learning", epi=0.20, vf=0.90)

# Phase 1: Attention activation
run_sequence(G, learner, [Emission()])

# Phase 2: Information reception
run_sequence(G, learner, [Reception()])

# Phase 3: Initial integration
run_sequence(G, learner, [Coherence()])

# Phase 4: Challenge with difficult problem
run_sequence(G, learner, [Dissonance()])

# Phase 5: Insight and reorganization
run_sequence(G, learner, [SelfOrganization()])

# Phase 6: Memory consolidation
run_sequence(G, learner, [Coherence(), Silence()])

# Result: Deep and lasting understanding
```

---

### Social/Organizational Domain

#### Team Cultural Transformation
```python
G, team = create_nfr("team_culture", epi=0.40, vf=1.00, theta=0.35)

# Phase 1: Dialogue activation
run_sequence(G, team, [Emission()])

# Phase 2: Mutual listening
run_sequence(G, team, [Reception()])

# Phase 3: Initial alignment
run_sequence(G, team, [Coupling()])

# Phase 4: Conflict exploration
run_sequence(G, team, [Dissonance()])

# Phase 5: Self-organization into new dynamics
run_sequence(G, team, [SelfOrganization()])

# Phase 6: New norms propagation
run_sequence(G, team, [Resonance()])

# Phase 7: Cultural consolidation
run_sequence(G, team, [Coherence()])

# Result: Transformed and sustainable team culture
```

---

## Sequence Validation

### Validity Criteria

A glyph sequence is valid if it meets:

1. **Preserves TNFR Closure**: Does not violate fundamental invariants
2. **Maintains C(t) > 0**: Global coherence never collapses
3. **ΔNFR Balance**: Reorganization peaks are resolved
4. **θ Continuity**: Phase maintains continuous trajectory
5. **Structural Purpose**: Each operator has clear function

### Sequence Quality Metrics

- **Efficiency**: Minimum operators for objective
- **Robustness**: Tolerance to parameter variations
- **Reproducibility**: Consistent results across executions
- **Scalability**: Works across different network sizes

---

## Advanced Usage

### Domain-Specific Applications

Grammar 2.0 pattern system enables domain-specific sequence design. See detailed guides:

- **[Therapeutic Applications](examples/domain_applications/README_THERAPEUTIC.md)**: Healing, trauma resolution, emotional regulation
- **[Educational Applications](examples/domain_applications/README_EDUCATIONAL.md)**: Learning, skill acquisition, knowledge integration
- **[Organizational Applications](examples/domain_applications/README_ORGANIZATIONAL.md)**: Cultural transformation, strategic pivots, team evolution
- **[Creative Applications](examples/domain_applications/README_CREATIVE.md)**: Artistic creation, innovation, design processes

### Pattern Construction Principles

#### Principle 1: Start with Purpose
Define the **structural outcome** before choosing operators:
- What coherence needs to emerge?
- What transformation is required?
- What stability is the goal?

#### Principle 2: Balance Forces
Every destabilizer should have a stabilizer:
- DISSONANCE → COHERENCE
- MUTATION → COHERENCE
- CONTRACTION → EXPANSION (when appropriate)

#### Principle 3: Respect Frequencies
Follow R5 frequency transitions:
- Zero → Medium → High (never Zero → High)
- Use Medium operators as bridges
- End with appropriate frequency for context

#### Principle 4: Complete Patterns
Don't leave cycles unfinished:
- If you start disruption, stabilize it
- If you activate, consolidate it
- If you expand, integrate it

#### Principle 5: Measure Health
Always validate sequences with health metrics:
```python
result = validate_sequence_with_health(your_sequence)
assert result.health_metrics.overall_health > 0.65, "Improve sequence quality"
```

### Health Optimization Guidelines

#### For Low Coherence (<0.6)
- Simplify sequence structure
- Remove random transitions
- Use recognized patterns as templates

#### For Low Balance (<0.5)
- Add stabilizers after destabilizers
- Avoid consecutive destabilizers
- Use COHERENCE more frequently

#### For Low Sustainability (<0.5)
- End with stabilizer (COHERENCE, SILENCE, or RESONANCE)
- Resolve all DISSONANCE before ending
- Add regenerative elements if long sequence

#### For Low Frequency Harmony (<0.6)
- Check for Zero → High jumps
- Insert Medium bridges (TRANSITION, COHERENCE)
- Review operator frequency categories

### Troubleshooting Common Issues

#### Issue: Validation fails with "THOL without destabilizer"
**Problem**: SELF_ORGANIZATION requires destabilizer within 3-operator window  
**Solution**: Add DISSONANCE, MUTATION, or CONTRACTION before SELF_ORGANIZATION

```python
# ❌ Fails
["emission", "reception", "self_organization"]

# ✅ Passes
["emission", "dissonance", "self_organization"]
```

#### Issue: Warning "Zero → High frequency jump"
**Problem**: SILENCE → EMISSION/DISSONANCE/RESONANCE/MUTATION/CONTRACTION  
**Solution**: Insert Medium operator bridge

```python
# ⚠️ Warning
["coherence", "silence", "emission"]

# ✅ No warning
["coherence", "silence", "transition", "emission"]
```

#### Issue: Low health score (<0.5)
**Problem**: Multiple structural issues  
**Solution**: Check recommendations and apply fixes

```python
result = validate_sequence_with_health(sequence)
for rec in result.health_metrics.recommendations:
    print(f"Fix: {rec}")
```

#### Issue: Pattern detected as UNKNOWN
**Problem**: Sequence doesn't match any recognized pattern  
**Solution**: Either:
1. Simplify to match a known pattern
2. Ensure structural coherence
3. Consider if COMPLEX is more appropriate

---

## API Reference

### Core Validation Functions

#### `validate_sequence_with_health(names)`

Enhanced validation with health metrics (Grammar 2.0).

**Parameters**:
- `names` (Iterable[str]): Sequence of operator names

**Returns**:
- `SequenceValidationResult`: Validation result with health metrics

**Example**:
```python
from tnfr.operators.grammar import validate_sequence_with_health

result = validate_sequence_with_health(["emission", "coherence", "silence"])
assert result.passed
print(result.health_metrics.overall_health)  # 0.78
```

#### `validate_sequence(names)`

Standard validation without health metrics.

**Parameters**:
- `names` (Iterable[str]): Sequence of operator names

**Returns**:
- `SequenceValidationResult`: Basic validation result

**Example**:
```python
from tnfr.operators.grammar import validate_sequence

result = validate_sequence(["emission", "coherence"])
print(result.passed)  # True
print(result.message)  # Success message
```

#### `validate_frequency_transition(prev_operator, next_operator)`

Check structural frequency compatibility (R5).

**Parameters**:
- `prev_operator` (str): Previous operator name
- `next_operator` (str): Next operator name

**Returns**:
- `tuple[bool, str]`: (is_valid, message)

**Example**:
```python
from tnfr.operators.grammar import validate_frequency_transition

is_valid, msg = validate_frequency_transition("silence", "emission")
print(is_valid)  # False
print(msg)  # "Incoherent frequency transition: ..."
```

### Pattern Detection

#### `AdvancedPatternDetector.detect_pattern(sequence)`

Detect structural pattern using coherence-weighted scoring.

**Parameters**:
- `sequence` (Sequence[str]): Operator names

**Returns**:
- `StructuralPattern`: Detected pattern enum

**Example**:
```python
from tnfr.operators.patterns import AdvancedPatternDetector

detector = AdvancedPatternDetector()
pattern = detector.detect_pattern([
    "reception", "emission", "coherence",
    "dissonance", "self_organization", "coherence"
])
print(pattern.value)  # "therapeutic"
```

### Health Analysis

#### `SequenceHealthAnalyzer.analyze_health(sequence)`

Perform comprehensive health analysis.

**Parameters**:
- `sequence` (List[str]): Operator names

**Returns**:
- `SequenceHealthMetrics`: Complete health metrics

**Example**:
```python
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer

analyzer = SequenceHealthAnalyzer()
health = analyzer.analyze_health(["emission", "coherence", "silence"])

print(f"Coherence: {health.coherence_index:.2f}")
print(f"Balance: {health.balance_score:.2f}")
print(f"Overall: {health.overall_health:.2f}")
```

### Cycle Detection

#### `CycleDetector.analyze_potential_cycle(sequence, regenerator_index)`

Analyze regenerative cycle validity (R5).

**Parameters**:
- `sequence` (Sequence[str]): Operator names
- `regenerator_index` (int): Position of regenerator operator

**Returns**:
- `CycleAnalysis`: Cycle validation results

**Example**:
```python
from tnfr.operators.cycle_detection import CycleDetector

detector = CycleDetector()
sequence = ["coherence", "silence", "transition", "emission", "coherence"]
analysis = detector.analyze_potential_cycle(sequence, 2)  # "transition" at index 2

print(f"Valid: {analysis.is_valid_regenerative}")
print(f"Type: {analysis.cycle_type.value}")
print(f"Health: {analysis.health_score:.2f}")
```

### Data Structures

#### `SequenceHealthMetrics`

Dataclass containing all health metrics.

**Attributes**:
- `coherence_index` (float): Flow quality (0.0-1.0)
- `balance_score` (float): Stabilizer/destabilizer equilibrium (0.0-1.0)
- `sustainability_index` (float): Long-term maintenance capacity (0.0-1.0)
- `complexity_efficiency` (float): Value-to-complexity ratio (0.0-1.0)
- `frequency_harmony` (float): νf transition smoothness (0.0-1.0)
- `pattern_completeness` (float): Cycle completion (0.0-1.0)
- `transition_smoothness` (float): Valid transition ratio (0.0-1.0)
- `overall_health` (float): Composite health index (0.0-1.0)
- `sequence_length` (int): Number of operators
- `dominant_pattern` (str): Detected pattern name
- `recommendations` (List[str]): Improvement suggestions

#### `StructuralPattern`

Enum of 18 pattern types.

**Values**:
- Fundamental: `LINEAR`, `HIERARCHICAL`, `FRACTAL`, `CYCLIC`, `BIFURCATED`
- Domain: `THERAPEUTIC`, `EDUCATIONAL`, `ORGANIZATIONAL`, `CREATIVE`, `REGENERATIVE`
- Compositional: `BOOTSTRAP`, `EXPLORE`, `STABILIZE`, `RESONATE`, `COMPRESS`
- Complexity: `COMPLEX`, `MINIMAL`, `UNKNOWN`

#### `SequenceValidationResult`

Validation result container.

**Attributes**:
- `passed` (bool): Validation success
- `message` (str): Result message
- `metadata` (dict): Additional info (detected_pattern, etc.)
- `health_metrics` (SequenceHealthMetrics | None): Health data (if requested)
- `tokens` (list): Original tokens
- `canonical_tokens` (list): Canonicalized tokens
- `error` (Exception | None): Error if validation failed

### Constants

#### `STRUCTURAL_FREQUENCIES`

Dictionary mapping operator names to frequency categories:
```python
{
    "emission": "high",
    "reception": "medium",
    "coherence": "medium",
    "dissonance": "high",
    "coupling": "medium",
    "resonance": "high",
    "silence": "zero",
    "expansion": "medium",
    "contraction": "high",
    "self_organization": "medium",
    "mutation": "high",
    "transition": "medium",
    "recursivity": "medium",
}
```

#### `FREQUENCY_TRANSITIONS`

Valid frequency transitions:
```python
{
    "high": {"high", "medium"},
    "medium": {"high", "medium", "zero"},
    "zero": {"medium"},  # Cannot jump to high
}
```

#### `REGENERATORS`

Operators enabling regenerative cycles:
```python
["transition", "recursivity", "silence"]
```

---

## Additional Resources

### TNFR References

- **[TNFR.pdf](TNFR.pdf)**: Fundamental paradigm document with theoretical foundations
- **[AGENTS.md](AGENTS.md)**: Guide for AI agents working with TNFR (includes invariants)
- **[GLOSSARY.md](GLOSSARY.md)**: Complete reference of TNFR terms, variables, and operators
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture and orchestration layers

### Documentation

- **[docs/source/api/operators.md](docs/source/api/operators.md)**: Technical operator reference
- **[docs/source/api/OPERATORS_VISUAL_GUIDE.md](docs/source/api/OPERATORS_VISUAL_GUIDE.md)**: Visual diagrams and examples
- **[MIGRATION_GUIDE_2.0.md](docs/MIGRATION_GUIDE_2.0.md)**: Upgrading from Grammar 1.0
- **[docs/HEALTH_METRICS_GUIDE.md](docs/HEALTH_METRICS_GUIDE.md)**: Deep dive into health metrics
- **[docs/PATTERN_REFERENCE.md](docs/PATTERN_REFERENCE.md)**: Complete pattern catalog
- **[docs/DOMAIN_APPLICATIONS.md](docs/DOMAIN_APPLICATIONS.md)**: Domain-specific usage guides

### Implementation

- **[src/tnfr/operators/grammar.py](src/tnfr/operators/grammar.py)**: Canonical implementation
- **[src/tnfr/operators/patterns.py](src/tnfr/operators/patterns.py)**: Pattern detection
- **[src/tnfr/operators/health_analyzer.py](src/tnfr/operators/health_analyzer.py)**: Health metrics
- **[src/tnfr/operators/cycle_detection.py](src/tnfr/operators/cycle_detection.py)**: R5 validation
- **[src/tnfr/operators/definitions.py](src/tnfr/operators/definitions.py)**: Operator classes

### Examples

- **[examples/domain_applications/](examples/domain_applications/)**: Complete domain examples
- **[examples/hello_world.py](examples/hello_world.py)**: Simplest example
- **[tests/examples/](tests/examples/)**: Validated example tests

---

## Validation Examples

---

## Validation Examples

### Example 1: Basic Validation

```python
from tnfr.operators.grammar import validate_sequence_with_health

# Simple activation sequence
sequence = ["emission", "coherence"]
result = validate_sequence_with_health(sequence)

print(f"Valid: {result.passed}")
print(f"Pattern: {result.metadata['detected_pattern']}")
print(f"Health: {result.health_metrics.overall_health:.2f}")

# Output:
# Valid: True
# Pattern: minimal
# Health: 0.75
```

### Example 2: Therapeutic Sequence

```python
# Complete therapeutic cycle
therapeutic = [
    "reception",           # EN: Receive current state
    "emission",            # AL: Activate process
    "coherence",           # IL: Stabilize preparation
    "dissonance",          # OZ: Confront challenge/trauma
    "self_organization",   # THOL: Reorganize (emergence)
    "coherence"            # IL: Integrate new structure
]

result = validate_sequence_with_health(therapeutic)

print(f"Pattern: {result.metadata['detected_pattern']}")  # therapeutic
print(f"Health: {result.health_metrics.overall_health:.2f}")  # ~0.88
print(f"Coherence: {result.health_metrics.coherence_index:.2f}")
print(f"Balance: {result.health_metrics.balance_score:.2f}")
print(f"Sustainability: {result.health_metrics.sustainability_index:.2f}")
```

### Example 3: Regenerative Cycle

```python
# Self-sustaining regenerative cycle
regenerative = [
    "coherence",      # IL: Initial stability
    "resonance",      # RA: Amplification
    "expansion",      # VAL: Growth
    "silence",        # SHA: Rest (regenerator)
    "transition",     # NAV: State change (regenerator)
    "emission",       # AL: Renewal
    "reception",      # EN: Input
    "coupling",       # UM: Network sync
    "coherence"       # IL: Final stability
]

result = validate_sequence_with_health(regenerative)

print(f"Pattern: {result.metadata['detected_pattern']}")  # regenerative
print(f"Health: {result.health_metrics.overall_health:.2f}")  # ~0.85
print(f"Sustainability: {result.health_metrics.sustainability_index:.2f}")  # ~0.90

# R5 cycle validation passed
assert result.passed
assert "regenerative" in result.metadata['detected_pattern']
```

### Example 4: Frequency Validation

```python
from tnfr.operators.grammar import validate_frequency_transition

# Check individual transitions
transitions = [
    ("silence", "emission"),      # Zero → High (invalid)
    ("silence", "coherence"),     # Zero → Medium (valid)
    ("coherence", "emission"),    # Medium → High (valid)
    ("emission", "dissonance"),   # High → High (valid)
]

for prev, next in transitions:
    is_valid, msg = validate_frequency_transition(prev, next)
    status = "✅" if is_valid else "⚠️"
    print(f"{status} {prev} → {next}: {msg if not is_valid else 'OK'}")

# Output:
# ⚠️ silence → emission: Incoherent frequency transition: ...
# ✅ silence → coherence: OK
# ✅ coherence → emission: OK
# ✅ emission → dissonance: OK
```

### Example 5: Health Optimization

```python
# Start with low-health sequence
poor_sequence = ["dissonance", "mutation", "contraction"]
result_poor = validate_sequence_with_health(poor_sequence)

print(f"Initial Health: {result_poor.health_metrics.overall_health:.2f}")  # ~0.35
print(f"Balance: {result_poor.health_metrics.balance_score:.2f}")  # ~0.25
print(f"Sustainability: {result_poor.health_metrics.sustainability_index:.2f}")  # ~0.20

# Apply recommendations
print("\nRecommendations:")
for rec in result_poor.health_metrics.recommendations:
    print(f"  - {rec}")

# Improve sequence based on recommendations
improved_sequence = [
    "coherence",      # Add initial stabilizer
    "dissonance",
    "coherence",      # Stabilize after destabilizer
    "mutation",
    "coherence",      # Stabilize after destabilizer
    "silence"         # End with stabilizer
]

result_improved = validate_sequence_with_health(improved_sequence)

print(f"\nImproved Health: {result_improved.health_metrics.overall_health:.2f}")  # ~0.78
print(f"Balance: {result_improved.health_metrics.balance_score:.2f}")  # ~0.82
print(f"Sustainability: {result_improved.health_metrics.sustainability_index:.2f}")  # ~0.85
```

### Example 6: Pattern Detection

```python
from tnfr.operators.patterns import AdvancedPatternDetector

detector = AdvancedPatternDetector()

# Test different sequences
sequences = {
    "bootstrap": ["emission", "coupling", "coherence"],
    "explore": ["dissonance", "mutation", "coherence"],
    "therapeutic": [
        "reception", "emission", "coherence",
        "dissonance", "self_organization", "coherence"
    ],
    "complex": [
        "emission", "coupling", "coherence",  # BOOTSTRAP
        "dissonance", "mutation", "coherence",  # EXPLORE
        "resonance", "silence"  # STABILIZE
    ],
}

for name, sequence in sequences.items():
    pattern = detector.detect_pattern(sequence)
    result = validate_sequence_with_health(sequence)
    print(f"{name:15s}: {pattern.value:15s} (health: {result.health_metrics.overall_health:.2f})")

# Output:
# bootstrap      : bootstrap       (health: 0.82)
# explore        : explore         (health: 0.75)
# therapeutic    : therapeutic     (health: 0.88)
# complex        : complex         (health: 0.80)
```

### Example 7: Custom Sequence Builder

```python
from tnfr.operators.grammar import validate_sequence_with_health

def build_sequence_with_validation(operators, min_health=0.65):
    """Build and validate sequence with minimum health requirement."""
    result = validate_sequence_with_health(operators)
    
    if not result.passed:
        raise ValueError(f"Sequence validation failed: {result.message}")
    
    health = result.health_metrics.overall_health
    if health < min_health:
        raise ValueError(
            f"Health {health:.2f} below threshold {min_health}. "
            f"Recommendations: {result.health_metrics.recommendations}"
        )
    
    return result

# Use it
try:
    sequence = ["emission", "dissonance", "mutation"]
    result = build_sequence_with_validation(sequence, min_health=0.70)
except ValueError as e:
    print(f"Error: {e}")
    # Fix sequence based on error
    sequence = ["emission", "dissonance", "coherence", "mutation", "coherence"]
    result = build_sequence_with_validation(sequence, min_health=0.70)
    print(f"Fixed sequence health: {result.health_metrics.overall_health:.2f}")
```

---

## Contributions

To propose new canonical sequences:

1. **Document application context**: Describe the real-world use case
2. **Provide functional code examples**: Include complete, runnable code
3. **Include expected metrics**: Document EPI, ΔNFR, C(t), νf effects
4. **Validate across domains**: Test in at least 3 different application areas
5. **Include failure cases**: Document related anti-patterns
6. **Measure health**: Ensure overall_health >0.65
7. **Add tests**: Include validation tests in `tests/examples/`

**Submission Template**:
```python
"""
New Pattern: [PATTERN_NAME]

Domain: [PRIMARY DOMAIN]
Context: [WHAT PROBLEM IT SOLVES]
"""

def get_[pattern_name]_sequence():
    """[DESCRIPTION]
    
    Expected Health Metrics:
    - Overall: >0.75
    - Coherence: >0.80
    - Balance: >0.75
    - Sustainability: >0.80
    
    Applications:
    - [USE CASE 1]
    - [USE CASE 2]
    - [USE CASE 3]
    """
    return [
        "operator1",  # Comment: structural effect
        "operator2",  # Comment: structural effect
        # ...
    ]

# Validation test
def test_[pattern_name]_sequence():
    sequence = get_[pattern_name]_sequence()
    result = validate_sequence_with_health(sequence)
    
    assert result.passed
    assert result.health_metrics.overall_health > 0.75
    assert result.metadata['detected_pattern'] == '[expected_pattern]'
```

---

## Conclusion

**Grammar 2.0** transforms TNFR operator sequences from validated patterns into **quantitatively assessed structural processes**. The additions of structural frequencies, health metrics, pattern detection, and regenerative cycle validation provide:

### What You Gain

✨ **Structural Frequencies (R5)**: Understand and control reorganization energy flow  
📊 **Health Metrics**: Quantitative assessment of sequence quality (0.0-1.0)  
🎯 **18 Patterns**: Comprehensive typology from LINEAR to REGENERATIVE  
🔄 **Regenerative Cycles**: Self-sustaining sequences with validated renewal  
⚖️ **Graduated Compatibility**: Three-level guidance (Compatible/Caution/Incompatible)  
💡 **Recommendations**: Actionable suggestions for improving sequences  
🏗️ **Domain Specificity**: Specialized patterns for therapy, education, organizations, creativity

### Mastering These Sequences Enables

- **Therapeutic Design**: Evidence-based healing interventions with measurable health
- **Educational Excellence**: Learning sequences optimized for transformation
- **Organizational Evolution**: Cultural change with structural guarantees
- **Creative Innovation**: Artistic processes grounded in coherence principles
- **System Modeling**: Complex adaptive systems with TNFR precision
- **Application Development**: Software that truly **reorganizes**, not just represents

### Fundamental Reminder

In TNFR, sequences don't **describe** processes - they **are** the processes. Operator execution doesn't **model** reality - it **participates** in it through structural coupling.

**Grammar 2.0 makes this participation measurable, optimizable, and reproducible.**

### Next Steps

1. **Start Simple**: Begin with fundamental sequences (EMISSION → COHERENCE)
2. **Measure Always**: Use `validate_sequence_with_health()` for all sequences
3. **Optimize Health**: Target overall_health >0.65, ideal >0.80
4. **Learn Patterns**: Study the 18 structural patterns
5. **Apply Domains**: Try domain-specific examples in `examples/domain_applications/`
6. **Build Cycles**: Experiment with regenerative sequences (R5)
7. **Contribute**: Share your validated sequences with the community

**Welcome to Grammar 2.0 - where structural coherence becomes computational reality.**

---

*Last updated: 2025-11-07*  
*Version: 2.0.1 (Grammar 2.0 Complete)*  
*License: MIT (aligned with TNFR-Python-Engine)*
