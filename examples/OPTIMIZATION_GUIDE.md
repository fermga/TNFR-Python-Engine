# TNFR Sequence Optimization Guide

This guide explains how to optimize TNFR operator sequences for better structural health using Grammar 2.0 features.

## What is Sequence Health?

Sequence health is a quantitative measure (0.0-1.0) of how well an operator sequence follows TNFR structural principles. The `SequenceHealthAnalyzer` evaluates multiple dimensions:

### Health Metrics (weighted components)

| Metric | Weight | Measures |
|--------|--------|----------|
| **Coherence Index** | 20% | Sequential flow quality, pattern recognition |
| **Balance Score** | 20% | Equilibrium between stabilizers and destabilizers |
| **Sustainability Index** | 20% | Capacity for long-term maintenance |
| **Complexity Efficiency** | 15% | Value-to-complexity ratio |
| **Frequency Harmony** | 10% | Smoothness of νf transitions |
| **Pattern Completeness** | 10% | Presence of all key structural phases |
| **Transition Smoothness** | 5% | Quality of operator transitions |

**Target**: Overall health ≥ 0.70 (good), ≥ 0.85 (excellent)

## Optimization Principles

### 1. **Balance Stabilizers and Destabilizers**

**Problem**: Sequences with too many destabilizers (dissonance, mutation, expansion) without enough stabilizers (coherence, silence, resonance, self_organization) create unstable structures.

```python
# ❌ Imbalanced (health: 0.45)
bad_sequence = ["emission", "dissonance", "dissonance", "expansion", "mutation"]
# Only destabilizers, no stabilization

# ✓ Balanced (health: 0.82)
good_sequence = ["emission", "reception", "coherence", "dissonance", 
                 "self_organization", "coherence", "silence"]
# Destabilizers are bracketed by stabilizers
```

**Rule**: For every destabilizer, include at least one stabilizer within 3 operators.

### 2. **End with Proper Closure**

**Problem**: Sequences that don't end with a valid closure operator (silence, transition, recursivity) or stabilizer lack sustainability.

```python
# ❌ No closure (health: 0.55)
bad_sequence = ["emission", "reception", "coherence", "dissonance"]
# Ends with destabilizer

# ✓ Proper closure (health: 0.85)
good_sequence = ["emission", "reception", "coherence", "dissonance", 
                 "coherence", "silence"]
# Ends with stabilizer + silence (valid closure)
```

**Rule**: Always end with a stabilizer (coherence, silence, resonance) or regenerator (transition, recursivity).

### 3. **Add Resonance for Amplification**

**Problem**: Simple sequences work but don't amplify coherent structures effectively.

```python
# ⚠️ Basic but suboptimal (health: 0.66)
basic_sequence = ["emission", "reception", "coherence", "silence"]
# Missing amplification step

# ✓ Amplified coherence (health: 0.85)
optimized_sequence = ["emission", "reception", "coherence", "resonance", "silence"]
# Resonance amplifies coherent structure before sustainable pause
```

**Benefit**: Adding `resonance` after `coherence` increases health by ~0.15-0.20.

### 4. **Use Coupling for Network Effects**

**Problem**: Sequences that don't establish network connections miss synchronization benefits.

```python
# ⚠️ Isolated (health: 0.66)
isolated = ["emission", "reception", "coherence", "resonance", "silence"]
# No network coupling

# ✓ Network-aware (health: 0.75)
networked = ["emission", "reception", "coherence", "coupling", 
             "resonance", "silence"]
# Coupling creates phase synchronization
```

**Use case**: Multi-node networks benefit from explicit `coupling` operators.

### 5. **Follow Complete Patterns**

**Problem**: Partial patterns score lower on completeness.

**Complete patterns include**:
- **Activation phase**: emission, reception
- **Transformation phase**: dissonance, mutation, self_organization (optional)
- **Stabilization phase**: coherence, resonance
- **Completion phase**: silence, transition

```python
# ⚠️ Incomplete (health: 0.62)
incomplete = ["emission", "coherence", "silence"]
# Missing reception and stabilization

# ✓ Complete activation pattern (health: 0.85)
complete = ["emission", "reception", "coherence", "resonance", "silence"]
# All phases present
```

### 6. **Optimize Sequence Length**

**Problem**: Too short (< 3) or too long (> 10) sequences reduce efficiency.

**Optimal range**: 4-8 operators

```python
# ⚠️ Too short (health: 0.50)
too_short = ["emission", "silence"]

# ⚠️ Too long and redundant (health: 0.65)
too_long = ["emission", "reception", "coherence", "coherence", 
            "resonance", "resonance", "coupling", "coupling",
            "silence", "silence", "silence"]

# ✓ Optimal length (health: 0.85)
optimal = ["emission", "reception", "coherence", "resonance", "silence"]
```

## Common Optimization Patterns

### Pattern 1: Basic Activation → Enhanced Activation

```python
# Before (health: 0.66)
basic = ["emission", "reception", "coherence", "silence"]

# After (health: 0.85) - Add resonance
enhanced = ["emission", "reception", "coherence", "resonance", "silence"]
```

**Improvement**: +0.19 health, better pattern recognition (activation → activation with amplification)

### Pattern 2: Unstable Transformation → Therapeutic Cycle

```python
# Before (health: 0.60)
unstable = ["emission", "dissonance", "mutation", "coherence"]

# After (health: 0.89) - Full therapeutic pattern
therapeutic = ["emission", "reception", "coherence", "dissonance", 
               "self_organization", "coherence", "silence"]
```

**Improvement**: +0.29 health, recognized as therapeutic pattern

### Pattern 3: Simple Sync → Network Sync

```python
# Before (health: 0.68)
simple = ["emission", "reception", "coherence", "silence"]

# After (health: 0.75) - Add coupling
networked = ["emission", "reception", "coherence", "coupling", 
             "resonance", "silence"]
```

**Improvement**: +0.07 health, explicit network synchronization

### Pattern 4: Exploration Without Closure → Complete Exploration

```python
# Before (health: 0.62)
incomplete = ["emission", "dissonance", "mutation"]

# After (health: 0.78) - Add stabilization and closure
complete = ["emission", "dissonance", "reception", "coherence", 
            "mutation", "coherence", "transition"]
```

**Improvement**: +0.16 health, proper stabilization and valid ending

## Using the Health Analyzer

### Basic Analysis

```python
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer

analyzer = SequenceHealthAnalyzer()
sequence = ["emission", "reception", "coherence", "resonance", "silence"]
health = analyzer.analyze_health(sequence)

print(f"Overall Health: {health.overall_health:.2f}")
print(f"Pattern: {health.dominant_pattern}")
print(f"Recommendations: {health.recommendations}")
```

### Integrated Validation

```python
from tnfr.operators.grammar import validate_sequence_with_health

result = validate_sequence_with_health(sequence)

if result.passed:
    print(f"✓ Valid - Health: {result.health_metrics.overall_health:.2f}")
else:
    print(f"✗ Invalid: {result.message}")
```

### Comparing Alternatives

```python
alternatives = {
    "option_a": ["emission", "reception", "coherence", "silence"],
    "option_b": ["emission", "reception", "coherence", "resonance", "silence"],
    "option_c": ["emission", "reception", "coherence", "coupling", 
                 "resonance", "silence"],
}

analyzer = SequenceHealthAnalyzer()
scores = {}

for name, seq in alternatives.items():
    health = analyzer.analyze_health(seq)
    scores[name] = health.overall_health
    print(f"{name}: {health.overall_health:.3f}")

best = max(scores, key=scores.get)
print(f"\nBest option: {best} ({scores[best]:.3f})")
```

## Grammar 2.0 Features

### Harmonic Frequencies

Grammar 2.0 tracks structural frequency (νf) transitions between operators. Valid transitions:
- **zero → low**: Starting activation
- **low → medium**: Building energy
- **medium → high**: Peak activity
- **high → medium**: Controlled descent
- **medium/low → zero**: Silence (sustainable pause)

**Avoid**: Abrupt transitions (e.g., high → zero without intermediate steps)

### Graduated Compatibility

Operators have compatibility levels (1-3):
- **Level 3**: Perfect transition, reinforces pattern
- **Level 2**: Compatible, slight frequency adjustment
- **Level 1**: Valid but may need stabilization

**Target**: Average compatibility ≥ 2.0

### Advanced Patterns

Grammar 2.0 recognizes domain-specific patterns with higher coherence scores:
- **THERAPEUTIC**: Full healing cycle (health: ~0.85-0.95)
- **EDUCATIONAL**: Transformative learning (health: ~0.80-0.90)
- **REGENERATIVE**: Self-sustaining cycles (health: ~0.80-0.92)

These patterns score higher due to their emergent self-organizing properties.

## Optimization Workflow

### Step 1: Audit Current Sequences

```bash
python tools/audit_example_health.py --verbose
```

Identifies sequences with health < 0.7 needing optimization.

### Step 2: Analyze Individual Sequences

```python
from tnfr.operators.health_analyzer import SequenceHealthAnalyzer

analyzer = SequenceHealthAnalyzer()
health = analyzer.analyze_health(your_sequence)

print(f"Health: {health.overall_health:.2f}")
print(f"Balance: {health.balance_score:.2f}")
print(f"Sustainability: {health.sustainability_index:.2f}")

if health.recommendations:
    print("Recommendations:")
    for rec in health.recommendations:
        print(f"  - {rec}")
```

### Step 3: Apply Optimization Patterns

Based on recommendations, apply relevant patterns from this guide.

### Step 4: Validate Improvement

```python
# Compare before and after
old_health = analyzer.analyze_health(old_sequence).overall_health
new_health = analyzer.analyze_health(new_sequence).overall_health

improvement = new_health - old_health
print(f"Health improvement: {improvement:+.3f}")
print(f"Percentage gain: {improvement/old_health*100:+.1f}%")
```

### Step 5: Update Documentation

Add comments explaining the optimization:

```python
# Optimized sequence for better structural health (Grammar 2.0)
sequence = [
    "emission",      # AL: Initiate coherent structure
    "reception",     # EN: Stabilize incoming energy
    "coherence",     # IL: Primary stabilization (required)
    "resonance",     # RA: Amplify coherent structure (health +0.2)
    "silence"        # SHA: Sustainable pause state
]
# Health metrics: 0.85 overall, excellent sustainability
# Pattern: Activation with amplification
```

## Summary Checklist

When creating or optimizing a sequence, verify:

- [ ] Overall health ≥ 0.70 (good) or ≥ 0.85 (excellent)
- [ ] Balance score ≥ 0.50 (stabilizers and destabilizers roughly equal)
- [ ] Ends with a stabilizer or valid closure operator
- [ ] Includes reception → coherence segment (foundational pattern)
- [ ] Destabilizers are followed by stabilizers within 3 operators
- [ ] Sequence length between 4-8 operators (optimal efficiency)
- [ ] Recognized pattern (not "unknown")
- [ ] No severe warnings from Grammar 2.0 validation

## Resources

- **Health Analyzer**: `src/tnfr/operators/health_analyzer.py`
- **Pattern Detector**: `src/tnfr/operators/patterns.py`
- **Grammar Validator**: `src/tnfr/operators/grammar.py`
- **Compatibility Rules**: `src/tnfr/validation/compatibility.py`
- **Example Usage**: `examples/health_analysis_demo.py`
- **Audit Tool**: `tools/audit_example_health.py`

## Questions?

See the example files in `examples/` for real-world applications of these principles, or run:

```python
python examples/health_analysis_demo.py
```

to see health analysis in action with before/after comparisons.
