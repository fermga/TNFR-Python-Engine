# TNFR Health Metrics Guide

## Overview

Grammar 2.0 introduces **Structural Health Metrics** - a quantitative assessment system for evaluating TNFR operator sequence quality. All metrics range from 0.0 (poor) to 1.0 (excellent), providing objective measures aligned with TNFR canonical principles.

**Purpose**: Enable data-driven optimization of operator sequences through measurable quality indicators.

---

## Quick Start

```python
from tnfr.operators.grammar import validate_sequence_with_health

# Analyze any sequence
sequence = ["emission", "reception", "coherence", "silence"]
result = validate_sequence_with_health(sequence)

# Access health metrics
health = result.health_metrics

print(f"Overall Health: {health.overall_health:.2f}")  # Composite score
print(f"Coherence: {health.coherence_index:.2f}")      # Flow quality
print(f"Balance: {health.balance_score:.2f}")          # Force equilibrium
print(f"Sustainability: {health.sustainability_index:.2f}")  # Long-term viability

# Get improvement recommendations
for rec in health.recommendations:
    print(f"  - {rec}")
```

---

## The Seven Health Dimensions

### 1. Coherence Index

**Measures**: Global sequential flow quality  
**Range**: 0.0 (chaotic) to 1.0 (perfect flow)  
**Weight in overall**: 20%

#### What It Measures

- **Valid Transitions**: Ratio of compatible operator pairs
- **Pattern Structure**: Whether sequence forms recognizable pattern
- **Structural Closure**: Proper ending with valid terminator

#### Scoring Formula

```
coherence_index = (transition_quality + pattern_clarity + structural_closure) / 3.0
```

Where:
- `transition_quality`: Valid transitions / total transitions
- `pattern_clarity`: Pattern match score (0.0-1.0)
- `structural_closure`: 1.0 if ends with valid terminator, else 0.0

#### Interpretation

| Score | Quality | Meaning |
|-------|---------|---------|
| 0.85-1.00 | Excellent | Clear pattern, all transitions valid, proper ending |
| 0.70-0.84 | Good | Recognizable structure, mostly valid transitions |
| 0.50-0.69 | Fair | Some structure, mix of valid/CAUTION transitions |
| 0.30-0.49 | Poor | Unclear structure, many problematic transitions |
| 0.00-0.29 | Very Poor | No clear pattern, mostly invalid transitions |

#### Examples

```python
# High coherence (0.92)
["emission", "reception", "coherence", "silence"]
# - All transitions COMPATIBLE
# - Clear STABILIZE pattern
# - Ends with valid terminator (SILENCE)

# Medium coherence (0.65)
["emission", "silence", "transition"]
# - Has CAUTION transition (EMISSION → SILENCE)
# - Partial pattern recognition
# - Ends properly but flow disrupted

# Low coherence (0.35)
["mutation", "expansion", "contraction"]
# - Random transitions
# - No recognizable pattern
# - Doesn't end with preferred stabilizer
```

#### Improving Coherence

1. **Use recognized patterns**: BOOTSTRAP, THERAPEUTIC, STABILIZE, etc.
2. **Avoid random transitions**: Follow compatibility matrix
3. **End properly**: Use COHERENCE, SILENCE, or RESONANCE
4. **Keep it simple**: Shorter sequences often have better coherence

---

### 2. Balance Score

**Measures**: Equilibrium between stabilizing and destabilizing forces  
**Range**: 0.0 (imbalanced) to 1.0 (perfectly balanced)  
**Weight in overall**: 20%

#### What It Measures

- **Stabilizer Ratio**: Proportion of COHERENCE, SELF_ORGANIZATION, SILENCE, RESONANCE
- **Destabilizer Ratio**: Proportion of DISSONANCE, MUTATION, CONTRACTION
- **Equilibrium**: How well these forces counterbalance each other

#### Scoring Formula

```
stabilizer_ratio = count(stabilizers) / len(sequence)
destabilizer_ratio = count(destabilizers) / len(sequence)

if stabilizer_ratio == 0 and destabilizer_ratio == 0:
    balance = 0.5  # Neutral (no forces)
elif destabilizer_ratio == 0:
    balance = min(1.0, 0.5 + stabilizer_ratio * 0.5)  # All stabilizers
elif stabilizer_ratio == 0:
    balance = max(0.0, 0.5 - destabilizer_ratio * 0.5)  # All destabilizers
else:
    ratio = min(stabilizer_ratio, destabilizer_ratio) / max(stabilizer_ratio, destabilizer_ratio)
    balance = 0.5 + (ratio * 0.5)  # Balance measure
```

**Ideal**: Approximately equal numbers of stabilizers and destabilizers (balance ≈ 1.0).

#### Interpretation

| Score | Balance | Meaning |
|-------|---------|---------|
| 0.85-1.00 | Excellent | Well-balanced forces |
| 0.70-0.84 | Good | Slightly favors one side |
| 0.50-0.69 | Fair | Moderate imbalance |
| 0.30-0.49 | Poor | Heavy imbalance |
| 0.00-0.29 | Very Poor | Extreme imbalance, unstable |

#### Examples

```python
# High balance (0.88)
["dissonance", "coherence", "mutation", "self_organization", "coherence"]
# - 3 stabilizers (coherence, self_organization, coherence)
# - 2 destabilizers (dissonance, mutation)
# - Good equilibrium

# Medium balance (0.58)
["emission", "dissonance", "mutation", "contraction", "coherence"]
# - 1 stabilizer (coherence)
# - 3 destabilizers (dissonance, mutation, contraction)
# - Skewed toward destabilization

# Low balance (0.25)
["dissonance", "mutation", "contraction"]
# - 0 stabilizers
# - 3 destabilizers
# - No counterbalancing forces
```

#### Improving Balance

1. **Add stabilizers after destabilizers**: DISSONANCE → COHERENCE
2. **Avoid consecutive destabilizers**: Use COHERENCE between them
3. **End with stabilizer**: Final COHERENCE or SILENCE
4. **Use SELF_ORGANIZATION**: Counts as stabilizer while enabling emergence

---

### 3. Sustainability Index

**Measures**: Capacity for long-term structural maintenance  
**Range**: 0.0 (unsustainable) to 1.0 (highly sustainable)  
**Weight in overall**: 20%

#### What It Measures

- **Final Stabilization**: Ends with stabilizer operator
- **Resolved Dissonance**: All DISSONANCE followed by resolution
- **Regenerative Elements**: Contains TRANSITION, RECURSIVITY, or SILENCE
- **Closure Quality**: Proper termination structure

#### Scoring Formula

```
final_stabilizer_score = 1.0 if ends_with_stabilizer else 0.0
resolved_dissonance_score = resolved_dissonances / total_dissonances
regenerative_score = 1.0 if has_regenerators else 0.5

sustainability = (
    final_stabilizer_score * 0.4 +
    resolved_dissonance_score * 0.4 +
    regenerative_score * 0.2
)
```

#### Interpretation

| Score | Sustainability | Meaning |
|-------|---------------|---------|
| 0.85-1.00 | Excellent | Self-sustaining, stable long-term |
| 0.70-0.84 | Good | Mostly stable, minor sustainability issues |
| 0.50-0.69 | Fair | Some sustainability, needs maintenance |
| 0.30-0.49 | Poor | Low sustainability, likely to degrade |
| 0.00-0.29 | Very Poor | Unsustainable, will collapse |

#### Examples

```python
# High sustainability (0.92)
["emission", "dissonance", "coherence", "transition", "silence"]
# - Ends with SILENCE (stabilizer)
# - DISSONANCE resolved with COHERENCE
# - Contains regenerators (TRANSITION, SILENCE)

# Medium sustainability (0.55)
["emission", "reception", "coherence", "dissonance"]
# - Ends with DISSONANCE (destabilizer) - POOR
# - No final stabilization
# - Has some stable elements earlier

# Low sustainability (0.20)
["dissonance", "mutation", "contraction"]
# - Ends with CONTRACTION (destabilizer)
# - No resolution
# - No regenerative elements
```

#### Improving Sustainability

1. **End with stabilizer**: Add COHERENCE, SILENCE, or RESONANCE at end
2. **Resolve dissonance**: Every DISSONANCE should have COHERENCE after
3. **Add regenerators**: Include TRANSITION, RECURSIVITY, or SILENCE
4. **Use full cycles**: Complete patterns (THERAPEUTIC, REGENERATIVE)

---

### 4. Complexity Efficiency

**Measures**: Value-to-complexity ratio (inverse of redundancy)  
**Range**: 0.0 (inefficient) to 1.0 (maximally efficient)  
**Weight in overall**: 15%

#### What It Measures

- **Sequence Length**: Penalty for overly long sequences
- **Redundancy**: Penalty for repeated operators
- **Structural Value**: Bonus for complete patterns
- **Compactness**: Ratio of value to length

#### Scoring Formula

```
base_efficiency = 1.0 / (1.0 + (length / 10.0))  # Length penalty

redundancy_penalty = repeated_operators * 0.1
pattern_bonus = 0.2 if pattern_detected else 0.0

efficiency = max(0.0, min(1.0, 
    base_efficiency - redundancy_penalty + pattern_bonus
))
```

#### Interpretation

| Score | Efficiency | Meaning |
|-------|-----------|---------|
| 0.85-1.00 | Excellent | Concise, no redundancy, high value |
| 0.70-0.84 | Good | Reasonably efficient, minor redundancy |
| 0.50-0.69 | Fair | Some inefficiency, longer than needed |
| 0.30-0.49 | Poor | Inefficient, redundant, bloated |
| 0.00-0.29 | Very Poor | Extremely inefficient, very redundant |

#### Examples

```python
# High efficiency (0.92)
["emission", "coherence"]
# - Very short (2 operators)
# - No redundancy
# - Clear pattern (MINIMAL)

# Medium efficiency (0.65)
["emission", "reception", "coherence", "resonance", "coupling", "coherence"]
# - Moderate length (6 operators)
# - One repeated operator (coherence)
# - Recognizable pattern

# Low efficiency (0.35)
["emission", "emission", "coherence", "coherence", "silence", "silence", "emission"]
# - Long (7 operators)
# - High redundancy (3 operators repeated)
# - Unclear value
```

#### Improving Efficiency

1. **Remove redundancy**: Avoid repeating operators unnecessarily
2. **Keep it concise**: Use minimum operators for goal
3. **Use patterns**: Recognized patterns get bonus
4. **Avoid bloat**: Every operator should serve a purpose

---

### 5. Frequency Harmony

**Measures**: Structural frequency transition smoothness (R5)  
**Range**: 0.0 (inharmonic) to 1.0 (perfectly harmonic)  
**Weight in overall**: 10%

#### What It Measures

- **Valid Transitions**: High ↔ Medium, Medium ↔ Zero
- **Invalid Jumps**: Zero → High (without Medium bridge)
- **Frequency Gradient**: Smoothness of νf changes

#### Scoring Formula

```
valid_freq_transitions = 0
total_transitions = len(sequence) - 1

for i in range(total_transitions):
    is_valid, _ = validate_frequency_transition(seq[i], seq[i+1])
    if is_valid:
        valid_freq_transitions += 1

frequency_harmony = valid_freq_transitions / total_transitions if total_transitions > 0 else 1.0
```

#### Interpretation

| Score | Harmony | Meaning |
|-------|---------|---------|
| 0.90-1.00 | Excellent | All transitions harmonic |
| 0.75-0.89 | Good | Mostly harmonic, 1-2 minor issues |
| 0.50-0.74 | Fair | Some inharmonic transitions |
| 0.30-0.49 | Poor | Many inharmonic transitions |
| 0.00-0.29 | Very Poor | Chaotic frequency changes |

#### Frequency Categories

- **High** (νf > 0.8): EMISSION, DISSONANCE, RESONANCE, MUTATION, CONTRACTION
- **Medium** (0.3 < νf ≤ 0.8): RECEPTION, COHERENCE, COUPLING, EXPANSION, SELF_ORGANIZATION, TRANSITION, RECURSIVITY
- **Zero** (νf ≈ 0): SILENCE

#### Valid Transitions

- ✅ High → High
- ✅ High → Medium
- ✅ Medium → High
- ✅ Medium → Medium
- ✅ Medium → Zero
- ✅ Zero → Medium
- ❌ Zero → High (must go through Medium)

#### Examples

```python
# High harmony (0.95)
["silence", "transition", "emission", "coherence"]
# - Zero → Medium (valid)
# - Medium → High (valid)
# - High → Medium (valid)

# Medium harmony (0.67)
["emission", "silence", "coherence"]
# - High → Zero (valid)
# - Zero → Medium (valid)
# But better with bridge: emission → coherence → silence

# Low harmony (0.40)
["silence", "emission", "silence", "mutation"]
# - Zero → High (INVALID)
# - High → Zero (valid)
# - Zero → High (INVALID)
```

#### Improving Frequency Harmony

1. **Avoid Zero → High**: Insert Medium bridge (TRANSITION, COHERENCE)
2. **Use gradual transitions**: High → Medium → Zero
3. **Check operator frequencies**: Use STRUCTURAL_FREQUENCIES dict
4. **Fix warnings**: Address all R5 frequency warnings

---

### 6. Pattern Completeness

**Measures**: How complete the detected pattern is  
**Range**: 0.0 (incomplete/fragmented) to 1.0 (complete cycle)  
**Weight in overall**: 10%

#### What It Measures

- **Full Cycle**: Whether pattern includes all required elements
- **Required Operators**: Presence of pattern-specific operators
- **Structural Closure**: Pattern reaches natural conclusion

#### Scoring Logic

Different patterns have different completeness criteria:

**Domain Patterns** (THERAPEUTIC, EDUCATIONAL, etc.):
- Complete: All required operators present
- Partial: Missing some but structure recognizable

**Compositional Patterns** (BOOTSTRAP, EXPLORE, etc.):
- Complete: Core structure present
- Partial: Incomplete structure

**Fundamental Patterns** (LINEAR, CYCLIC, etc.):
- Complete: Full cycle or expected length
- Partial: Fragment of expected pattern

#### Interpretation

| Score | Completeness | Meaning |
|-------|--------------|---------|
| 0.90-1.00 | Excellent | Complete pattern, full cycle |
| 0.75-0.89 | Good | Nearly complete, minor omissions |
| 0.50-0.74 | Fair | Partial pattern, recognizable |
| 0.30-0.49 | Poor | Fragment, incomplete |
| 0.00-0.29 | Very Poor | Barely recognizable pattern |

#### Examples

```python
# High completeness (0.95) - THERAPEUTIC
[
    "reception", "emission", "coherence",
    "dissonance", "self_organization", "coherence"
]
# - All THERAPEUTIC elements present
# - Full healing cycle

# Medium completeness (0.60) - Partial THERAPEUTIC
["reception", "emission", "coherence"]
# - Start of THERAPEUTIC
# - Missing crisis resolution (dissonance, self_organization)

# Low completeness (0.30) - Fragment
["dissonance", "mutation"]
# - Part of EXPLORE pattern
# - Missing stabilizing coherence at end
```

#### Improving Completeness

1. **Complete cycles**: Add missing pattern elements
2. **Use templates**: Follow documented pattern structures
3. **Add resolution**: Ensure patterns conclude properly
4. **Check requirements**: Verify all required operators present

---

### 7. Transition Smoothness

**Measures**: Quality of operator-to-operator transitions  
**Range**: 0.0 (rough) to 1.0 (perfectly smooth)  
**Weight in overall**: 5%

#### What It Measures

- **COMPATIBLE Transitions**: ✅ Highly compatible pairs
- **CAUTION Transitions**: ⚠️ Context-specific pairs
- **INCOMPATIBLE Transitions**: ❌ Invalid pairs

#### Scoring Formula

```
compatible_count = sum(1 for trans in transitions if trans == COMPATIBLE)
caution_count = sum(1 for trans in transitions if trans == CAUTION)
total_transitions = len(sequence) - 1

smoothness = (compatible_count + 0.5 * caution_count) / total_transitions if total_transitions > 0 else 1.0
```

#### Interpretation

| Score | Smoothness | Meaning |
|-------|-----------|---------|
| 0.90-1.00 | Excellent | All transitions COMPATIBLE |
| 0.75-0.89 | Good | Mostly COMPATIBLE, some CAUTION |
| 0.50-0.74 | Fair | Mix of COMPATIBLE and CAUTION |
| 0.30-0.49 | Poor | Many CAUTION transitions |
| 0.00-0.29 | Very Poor | Many INCOMPATIBLE transitions (shouldn't validate) |

#### Examples

```python
# High smoothness (0.95)
["emission", "reception", "coherence"]
# - EMISSION → RECEPTION: COMPATIBLE
# - RECEPTION → COHERENCE: COMPATIBLE

# Medium smoothness (0.75)
["emission", "silence", "transition"]
# - EMISSION → SILENCE: CAUTION
# - SILENCE → TRANSITION: COMPATIBLE

# Low smoothness (0.33)
["emission", "silence", "coherence"]
# - EMISSION → SILENCE: CAUTION
# - SILENCE → COHERENCE: COMPATIBLE
# Only 1/2 fully compatible
```

#### Improving Smoothness

1. **Use COMPATIBLE transitions**: Check compatibility matrix
2. **Avoid CAUTION pairs**: Replace with smoother alternatives
3. **Fix INCOMPATIBLE**: Should never occur in valid sequence
4. **Follow patterns**: Recognized patterns have smooth transitions

---

## Overall Health Score

### Computation

The overall health score is a **weighted average** of all seven dimensions:

```python
overall_health = (
    coherence_index       * 0.20 +  # 20% - Flow quality
    balance_score         * 0.20 +  # 20% - Force equilibrium
    sustainability_index  * 0.20 +  # 20% - Long-term viability
    complexity_efficiency * 0.15 +  # 15% - Value/complexity ratio
    frequency_harmony     * 0.10 +  # 10% - R5 smoothness
    pattern_completeness  * 0.10 +  # 10% - Cycle completion
    transition_smoothness * 0.05    #  5% - Transition quality
)
```

### Weights Rationale

**Primary metrics (20% each)**:
- **Coherence**: Flow quality is fundamental
- **Balance**: Force equilibrium is essential for stability
- **Sustainability**: Long-term viability matters most

**Secondary metrics (10-15% each)**:
- **Efficiency**: Important but not critical
- **Frequency**: R5 harmony enhances quality
- **Completeness**: Pattern completion is valuable

**Tertiary metrics (5%)**:
- **Smoothness**: Important but covered by coherence

### Interpretation

| Score | Quality | Recommendation |
|-------|---------|----------------|
| 0.90-1.00 | Exceptional | Optimal sequence, no changes needed |
| 0.80-0.89 | Excellent | High quality, minor optimizations possible |
| 0.70-0.79 | Good | Solid sequence, consider improvements |
| 0.60-0.69 | Fair | Acceptable but should optimize |
| 0.50-0.59 | Mediocre | Significant improvements needed |
| 0.00-0.49 | Poor | Redesign recommended |

---

## Recommendations System

### How Recommendations Work

The health analyzer automatically generates **actionable recommendations** when specific issues are detected:

```python
result = validate_sequence_with_health(sequence)
for rec in result.health_metrics.recommendations:
    print(f"  - {rec}")
```

### Recommendation Types

#### 1. Balance Recommendations

**Trigger**: `balance_score < 0.6`

**Examples**:
- "Add stabilizer after destabilizer at position 2"
- "Sequence too destabilizer-heavy, add COHERENCE"
- "Balance destabilizers with stabilizers"

#### 2. Sustainability Recommendations

**Trigger**: `sustainability_index < 0.6`

**Examples**:
- "End sequence with stabilizer (coherence, silence, or resonance)"
- "Resolve DISSONANCE at position 3 with COHERENCE"
- "Add regenerative element (transition, recursivity, or silence)"

#### 3. Frequency Recommendations

**Trigger**: `frequency_harmony < 0.7`

**Examples**:
- "Insert medium-frequency operator between silence and emission"
- "Fix Zero → High frequency jump at position 2"
- "Use TRANSITION or COHERENCE as frequency bridge"

#### 4. Completeness Recommendations

**Trigger**: `pattern_completeness < 0.7`

**Examples**:
- "Complete THERAPEUTIC pattern by adding: dissonance, self_organization"
- "Add final stabilizer to complete cycle"
- "EXPLORE pattern incomplete, add final COHERENCE"

#### 5. Efficiency Recommendations

**Trigger**: `complexity_efficiency < 0.6`

**Examples**:
- "Remove redundant operator at position 4"
- "Simplify sequence, consider using recognized pattern"
- "Replace operators 3-5 with single TRANSITION"

---

## Usage Patterns

### Pattern 1: Quality Gate

Use health metrics as a quality gate in production:

```python
def require_minimum_health(sequence, min_health=0.70):
    """Enforce minimum health threshold."""
    result = validate_sequence_with_health(sequence)
    
    if not result.passed:
        raise ValueError(f"Sequence invalid: {result.message}")
    
    health = result.health_metrics.overall_health
    if health < min_health:
        raise ValueError(
            f"Health {health:.2f} below threshold {min_health}\n"
            f"Recommendations:\n" + 
            "\n".join(f"  - {r}" for r in result.health_metrics.recommendations)
        )
    
    return result

# Use it
try:
    result = require_minimum_health(my_sequence)
    print(f"✅ Sequence passed with health {result.health_metrics.overall_health:.2f}")
except ValueError as e:
    print(f"❌ {e}")
```

### Pattern 2: Iterative Optimization

Optimize sequence based on health feedback:

```python
def optimize_sequence(initial_sequence, target_health=0.80, max_iterations=5):
    """Iteratively improve sequence based on recommendations."""
    sequence = initial_sequence.copy()
    
    for iteration in range(max_iterations):
        result = validate_sequence_with_health(sequence)
        health = result.health_metrics.overall_health
        
        print(f"Iteration {iteration}: Health = {health:.2f}")
        
        if health >= target_health:
            print("✅ Target health reached!")
            return sequence, result
        
        if not result.health_metrics.recommendations:
            print("⚠️ No more recommendations")
            break
        
        # Apply first recommendation (simplified)
        rec = result.health_metrics.recommendations[0]
        print(f"  Applying: {rec}")
        
        # TODO: Parse and apply recommendation
        # This would require recommendation parsing logic
        
    return sequence, result
```

### Pattern 3: Comparative Analysis

Compare multiple sequences:

```python
def compare_sequences(sequences, names=None):
    """Compare health metrics across sequences."""
    if names is None:
        names = [f"Sequence {i+1}" for i in range(len(sequences))]
    
    results = []
    for name, seq in zip(names, sequences):
        result = validate_sequence_with_health(seq)
        if result.passed:
            health = result.health_metrics
            results.append({
                'name': name,
                'overall': health.overall_health,
                'coherence': health.coherence_index,
                'balance': health.balance_score,
                'sustainability': health.sustainability_index,
            })
    
    # Sort by overall health
    results.sort(key=lambda x: x['overall'], reverse=True)
    
    # Display
    print(f"{'Sequence':<20} {'Overall':>8} {'Coherence':>10} {'Balance':>8} {'Sustain':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<20} {r['overall']:>8.2f} {r['coherence']:>10.2f} {r['balance']:>8.2f} {r['sustain']:>8.2f}")

# Use it
sequences = {
    "Therapeutic": ["reception", "emission", "coherence", "dissonance", "self_organization", "coherence"],
    "Bootstrap": ["emission", "coupling", "coherence"],
    "Explore": ["dissonance", "mutation", "coherence"],
}
compare_sequences(list(sequences.values()), list(sequences.keys()))
```

### Pattern 4: Health-Based Sequence Selection

Choose best sequence from options:

```python
def select_best_sequence(candidates, weights=None):
    """Select sequence with highest weighted health score."""
    if weights is None:
        weights = {'overall': 1.0}  # Default to overall health
    
    best_score = -1
    best_sequence = None
    best_result = None
    
    for seq in candidates:
        result = validate_sequence_with_health(seq)
        if not result.passed:
            continue
        
        health = result.health_metrics
        score = sum(
            getattr(health, metric) * weight
            for metric, weight in weights.items()
        )
        
        if score > best_score:
            best_score = score
            best_sequence = seq
            best_result = result
    
    return best_sequence, best_result

# Use it
candidates = [
    ["emission", "coherence"],
    ["emission", "reception", "coherence", "silence"],
    ["reception", "emission", "coherence", "dissonance", "self_organization", "coherence"],
]

# Prioritize sustainability
weights = {
    'overall_health': 0.5,
    'sustainability_index': 0.3,
    'balance_score': 0.2,
}

best, result = select_best_sequence(candidates, weights)
print(f"Best sequence: {best}")
print(f"Health: {result.health_metrics.overall_health:.2f}")
```

---

## Advanced Topics

### Custom Health Thresholds

Different applications may require different health thresholds:

```python
# High-stakes therapeutic applications
THERAPEUTIC_MIN_HEALTH = 0.85
THERAPEUTIC_MIN_SUSTAINABILITY = 0.90

# Experimental/exploratory sequences
EXPERIMENTAL_MIN_HEALTH = 0.60

# Production/stable sequences
PRODUCTION_MIN_HEALTH = 0.75
PRODUCTION_MIN_BALANCE = 0.70
```

### Health Trends Over Time

Track health improvements across iterations:

```python
def track_health_evolution(sequence_versions):
    """Track health metrics across sequence iterations."""
    import matplotlib.pyplot as plt
    
    metrics = {
        'overall': [],
        'coherence': [],
        'balance': [],
        'sustainability': [],
    }
    
    for seq in sequence_versions:
        result = validate_sequence_with_health(seq)
        if result.passed:
            h = result.health_metrics
            metrics['overall'].append(h.overall_health)
            metrics['coherence'].append(h.coherence_index)
            metrics['balance'].append(h.balance_score)
            metrics['sustainability'].append(h.sustainability_index)
    
    # Plot evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, values in metrics.items():
        ax.plot(range(len(values)), values, marker='o', label=name.title())
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')
    ax.set_title('Health Metrics Evolution')
    ax.legend()
    ax.grid(True)
    plt.show()
```

---

## FAQ

### Q: What's a "good" overall health score?

**A**: Depends on context:
- **Therapeutic/Production**: >0.80 (excellent quality)
- **General Use**: >0.70 (good quality)
- **Experimental**: >0.60 (acceptable)
- **Below 0.60**: Needs improvement

### Q: Why is my health score low despite valid sequence?

**A**: Validation checks correctness; health measures quality. A sequence can be:
- Valid but imbalanced (low balance score)
- Valid but inefficient (low efficiency score)
- Valid but unsustainable (low sustainability score)

Use recommendations to improve.

### Q: Can I weight metrics differently?

**A**: Yes! You can manually compute custom overall scores:

```python
result = validate_sequence_with_health(sequence)
h = result.health_metrics

# Custom weights
custom_score = (
    h.coherence_index * 0.3 +
    h.sustainability_index * 0.4 +
    h.balance_score * 0.3
)
```

### Q: How do recommendations work?

**A**: The analyzer detects specific issues (low balance, missing stabilizer, frequency jumps) and generates targeted suggestions. Recommendations are rule-based and actionable.

### Q: Should I always aim for 1.0 health?

**A**: Not necessarily. Context matters:
- Some applications prioritize sustainability over efficiency
- Exploratory sequences may intentionally have lower balance
- Short sequences may have lower completeness (expected)

Aim for "good enough" for your use case.

---

## Resources

- **[GLYPH_SEQUENCES_GUIDE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/GLYPH_SEQUENCES_GUIDE.md)**: Complete Grammar 2.0 guide
- **[MIGRATION_GUIDE_2.0.md](MIGRATION_GUIDE_2.0.md)**: Upgrading from 1.0
- **[PATTERN_REFERENCE.md](PATTERN_REFERENCE.md)**: Pattern catalog
- **[src/tnfr/operators/health_analyzer.py](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/operators/health_analyzer.py)**: Implementation

---

*Last updated: 2025-11-07*  
*Grammar version: 2.0*
