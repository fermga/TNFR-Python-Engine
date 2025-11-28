# TNFR Grammar 2.0 Troubleshooting Guide

## Overview

This guide provides solutions to common issues when working with TNFR operator sequences in Grammar 2.0.

---

## Validation Errors

### Error: "SELF_ORGANIZATION requires destabilizer within 3-operator window"

**Problem**: SELF_ORGANIZATION (THOL) operator appears without a recent destabilizer.

**Why**: Self-organization emerges from instability, not from stable states (R3 rule).

**Solution**: Add DISSONANCE, MUTATION, or CONTRACTION within 3 operators before SELF_ORGANIZATION.

```python
# ❌ Fails
["emission", "reception", "self_organization"]

# ✅ Fixed - destabilizer adjacent
["emission", "dissonance", "self_organization"]

# ✅ Fixed - destabilizer within window
["dissonance", "emission", "reception", "self_organization"]

# ✅ Fixed - mutation works too
["emission", "mutation", "reception", "self_organization"]
```

---

### Error: "Invalid start operator"

**Problem**: Sequence starts with invalid operator.

**Valid starters**: EMISSION, RECEPTION, TRANSITION, SILENCE, COHERENCE, SELF_ORGANIZATION

**Solution**: Start with valid initiator.

```python
# ❌ Fails - CONTRACTION not valid starter
["contraction", "coherence"]

# ✅ Fixed
["emission", "contraction", "coherence"]
```

---

### Error: "Invalid end operator"

**Problem**: Sequence ends with invalid operator.

**Valid enders**: COHERENCE, SILENCE, RESONANCE, COUPLING, SELF_ORGANIZATION, RECURSIVITY, CONTRACTION, MUTATION

**Solution**: End with valid terminator.

```python
# ❌ Fails - EXPANSION not valid ender
["emission", "expansion"]

# ✅ Fixed
["emission", "expansion", "coherence"]
```

---

### Error: "Incompatible transition"

**Problem**: Two consecutive operators are incompatible (✗ in compatibility matrix).

**Forbidden transitions**:
- DISSONANCE → DISSONANCE
- DISSONANCE → SILENCE
- SILENCE → DISSONANCE
- SILENCE → SILENCE
- EXPANSION → CONTRACTION
- Others (see compatibility matrix)

**Solution**: Insert bridging operator or reorder sequence.

```python
# ❌ Fails - SILENCE → DISSONANCE incompatible
["emission", "silence", "dissonance"]

# ✅ Fixed - add bridge
["emission", "silence", "transition", "dissonance"]

# ✅ Fixed - reorder
["emission", "dissonance", "silence"]
```

---

## Validation Warnings

### Warning: "Zero → High frequency transition"

**Problem**: Jumping from zero frequency (SILENCE) to high frequency (EMISSION, DISSONANCE, RESONANCE, MUTATION, CONTRACTION) without medium bridge.

**Why**: R5 frequency harmony requires gradual transitions.

**Solution**: Insert medium-frequency operator between SILENCE and high-frequency operator.

```python
# ⚠️ Warning
["coherence", "silence", "emission"]

# ✅ Fixed - medium bridge
["coherence", "silence", "transition", "emission"]
# or
["coherence", "silence", "coherence", "emission"]
```

**Medium-frequency operators** (bridges):
- TRANSITION, COHERENCE, RECEPTION, COUPLING, EXPANSION, SELF_ORGANIZATION, RECURSIVITY

---

### Warning: "CAUTION transition detected"

**Problem**: Transition is context-specific (○ in compatibility matrix).

**Impact**: Non-blocking warning, sequence still valid.

**Solution**: Either:
1. Accept the warning (sequence is still valid)
2. Replace with fully compatible (✓) transition

```python
# ⚠️ CAUTION warning
["emission", "silence"]  # High → Zero is CAUTION

# ✅ Fully compatible alternative
["emission", "coherence", "silence"]  # High → Medium → Zero
```

---

## Health Issues

### Issue: Low Overall Health (<0.60)

**Symptoms**: Sequence validates but health score is low.

**Solution**: Check individual metrics and apply targeted fixes.

```python
result = validate_sequence_with_health(sequence)
health = result.health_metrics

print(f"Overall: {health.overall_health:.2f}")
print(f"Coherence: {health.coherence_index:.2f}")
print(f"Balance: {health.balance_score:.2f}")
print(f"Sustainability: {health.sustainability_index:.2f}")

# Check recommendations
for rec in health.recommendations:
    print(f"Fix: {rec}")
```

---

### Issue: Low Balance Score (<0.50)

**Symptom**: Too many destabilizers without stabilizers, or vice versa.

**Stabilizers**: COHERENCE, SELF_ORGANIZATION, SILENCE, RESONANCE  
**Destabilizers**: DISSONANCE, MUTATION, CONTRACTION

**Solution**: Balance forces.

```python
# ❌ Imbalanced - all destabilizers
["dissonance", "mutation", "contraction"]  # Balance: 0.25

# ✅ Balanced
["dissonance", "coherence", "mutation", "coherence"]  # Balance: 0.80

# ✅ Balanced with emergence
["dissonance", "self_organization", "coherence"]  # Balance: 0.75
```

**Rule of thumb**: Every destabilizer should have a stabilizer nearby.

---

### Issue: Low Sustainability (<0.50)

**Symptoms**:
- Ends with destabilizer
- Unresolved DISSONANCE
- No regenerative elements

**Solution**:

1. **End with stabilizer**:
```python
# ❌ Poor sustainability
["emission", "dissonance"]  # Ends with destabilizer

# ✅ Good sustainability
["emission", "dissonance", "coherence"]  # Ends with stabilizer
```

2. **Resolve dissonance**:
```python
# ❌ Unresolved
["emission", "dissonance", "emission"]

# ✅ Resolved
["emission", "dissonance", "coherence", "emission"]
```

3. **Add regenerative elements**:
```python
# ❌ No regeneration
["emission", "coherence"]

# ✅ With regeneration
["emission", "coherence", "silence"]  # SILENCE is regenerator
```

---

### Issue: Low Complexity Efficiency (<0.50)

**Symptoms**:
- Sequence too long
- Repeated operators
- No clear pattern

**Solution**:

1. **Remove redundancy**:
```python
# ❌ Redundant
["emission", "emission", "coherence", "coherence"]

# ✅ Efficient
["emission", "coherence"]
```

2. **Simplify**:
```python
# ❌ Unnecessarily complex
["emission", "reception", "emission", "coherence", "resonance", "coherence"]

# ✅ Simpler
["emission", "reception", "coherence"]
```

3. **Use patterns**:
```python
# ❌ No pattern
["emission", "expansion", "contraction", "silence"]

# ✅ Recognized pattern
["emission", "coupling", "coherence"]  # BOOTSTRAP
```

---

### Issue: Low Frequency Harmony (<0.60)

**Symptom**: Invalid frequency transitions (Zero → High).

**Solution**: Fix R5 violations.

```python
# ❌ Low harmony (0.40)
["silence", "emission", "silence", "dissonance"]
# - Zero → High (INVALID)
# - Zero → High (INVALID)

# ✅ Good harmony (0.95)
["silence", "transition", "emission", "coherence", "silence"]
# - Zero → Medium (valid)
# - Medium → High (valid)
# - High → Medium (valid)
# - Medium → Zero (valid)
```

---

## Pattern Detection Issues

### Issue: Pattern Detected as UNKNOWN

**Problem**: Sequence doesn't match any recognized pattern.

**Causes**:
- Random operator order
- No clear structure
- Mixed incompatible elements

**Solutions**:

1. **Use template patterns**:
```python
# ❌ UNKNOWN
["expansion", "mutation", "reception"]

# ✅ BOOTSTRAP
["emission", "coupling", "coherence"]
```

2. **Follow domain patterns**:
```python
# ✅ THERAPEUTIC
[
    "reception", "emission", "coherence",
    "dissonance", "self_organization", "coherence"
]
```

3. **Check documentation**: See [PATTERN_REFERENCE.md](PATTERN_REFERENCE.md)

---

### Issue: Pattern Changed from Grammar 1.0

**Problem**: Sequence classified differently in 2.0 vs 1.0.

**Why**: Grammar 2.0 has more specific patterns (18 vs ~6).

**Solution**: Update code to use pattern categories, not specific names.

```python
# ❌ Fragile - depends on exact name
if pattern == 'activation':
    handle_activation()

# ✅ Robust - uses categories
SIMPLE_PATTERNS = ['minimal', 'linear', 'bootstrap', 'stabilize']
DOMAIN_PATTERNS = ['therapeutic', 'educational', 'organizational', 'creative']

if pattern in SIMPLE_PATTERNS:
    handle_simple()
elif pattern in DOMAIN_PATTERNS:
    handle_domain_specific()
```

---

## API Usage Issues

### Issue: validate_sequence_with_health() returns None for health_metrics

**Problem**: Health metrics only computed for valid sequences.

**Solution**: Check validation first.

```python
result = validate_sequence_with_health(sequence)

if not result.passed:
    print(f"Validation failed: {result.message}")
    print(f"Fix the sequence first")
else:
    # Now health_metrics is available
    health = result.health_metrics
    print(f"Health: {health.overall_health:.2f}")
```

---

### Issue: Frequency validation seems wrong

**Problem**: Confusion about operator frequencies.

**Solution**: Check STRUCTURAL_FREQUENCIES dict.

```python
from tnfr.operators.grammar import STRUCTURAL_FREQUENCIES

# Query operator frequencies
for op in ["emission", "coherence", "silence"]:
    freq = STRUCTURAL_FREQUENCIES[op]
    print(f"{op}: {freq}")

# Output:
# emission: high
# coherence: medium
# silence: zero
```

**Frequency categories**:
- **High**: EMISSION, DISSONANCE, RESONANCE, MUTATION, CONTRACTION
- **Medium**: RECEPTION, COHERENCE, COUPLING, EXPANSION, SELF_ORGANIZATION, TRANSITION, RECURSIVITY
- **Zero**: SILENCE

---

## Regenerative Cycle Issues

### Issue: R5 regenerative cycle validation fails

**Common causes**:

1. **Too short** (< 5 operators):
```python
# ❌ Too short
["coherence", "silence", "emission"]  # Only 3

# ✅ Minimum length
["coherence", "silence", "transition", "emission", "coherence"]  # 5
```

2. **No regenerator**:
```python
# ❌ No regenerator
["emission", "coherence", "resonance", "coupling", "coherence"]

# ✅ Has regenerator
["emission", "coherence", "silence", "transition", "emission"]
# SILENCE and TRANSITION are regenerators
```

3. **Unbalanced stabilizers**:
```python
# ❌ No stabilizer before regenerator
["dissonance", "transition", "emission"]

# ✅ Balanced
["coherence", "resonance", "silence", "transition", "emission", "coherence"]
# - Stabilizers before: coherence, resonance
# - Stabilizers after: coherence
```

4. **Low health** (<0.6):
```python
# ❌ Low health
["dissonance", "silence", "mutation", "contraction"]  # ~0.35

# ✅ Good health
["coherence", "resonance", "silence", "transition", "emission", "coherence"]  # ~0.85
```

**Regenerator operators**: TRANSITION (NAV), RECURSIVITY (REMESH), SILENCE (SHA)

---

## Migration from Grammar 1.0

### Issue: Code breaks after upgrading to 2.0

**Most common cause**: SELF_ORGANIZATION without destabilizer.

**Solution**: Add destabilizers.

```python
# Old 1.0 code (now breaks)
sequence = ["emission", "reception", "self_organization"]

# Fixed for 2.0
sequence = ["emission", "dissonance", "self_organization"]
# or
sequence = ["dissonance", "emission", "reception", "self_organization"]
```

See [MIGRATION_GUIDE_2.0.md](MIGRATION_GUIDE_2.0.md) for complete migration guide.

---

## Performance Issues

### Issue: Validation is slow

**Cause**: Health metrics computation adds overhead.

**Solution**: Use `validate_sequence()` if health not needed.

```python
# Fast - no health metrics
from tnfr.operators.grammar import validate_sequence
result = validate_sequence(sequence)

# Slower - includes health metrics
from tnfr.operators.grammar import validate_sequence_with_health
result = validate_sequence_with_health(sequence)
```

---

## Best Practices

### ✅ Do's

1. **Always validate sequences**:
```python
result = validate_sequence_with_health(sequence)
assert result.passed, f"Invalid: {result.message}"
```

2. **Check health for production sequences**:
```python
if result.health_metrics.overall_health < 0.70:
    log_warning(f"Low health: {result.health_metrics.overall_health:.2f}")
```

3. **Use recommendations**:
```python
for rec in result.health_metrics.recommendations:
    log_info(f"Suggestion: {rec}")
```

4. **Follow patterns**:
```python
# Use documented patterns
therapeutic = [
    "reception", "emission", "coherence",
    "dissonance", "self_organization", "coherence"
]
```

5. **Balance forces**:
```python
# Every destabilizer should have a stabilizer
["dissonance", "coherence", "mutation", "coherence"]
```

### ❌ Don'ts

1. **Don't ignore validation errors**:
```python
# ❌ Bad
result = validate_sequence(sequence)
# Assume it passed...

# ✅ Good
result = validate_sequence(sequence)
if not result.passed:
    raise ValueError(result.message)
```

2. **Don't use SELF_ORGANIZATION without destabilizer**:
```python
# ❌ Bad
["emission", "self_organization"]

# ✅ Good
["emission", "dissonance", "self_organization"]
```

3. **Don't jump Zero → High**:
```python
# ❌ Bad
["silence", "emission"]

# ✅ Good
["silence", "transition", "emission"]
```

4. **Don't end with destabilizers**:
```python
# ❌ Bad
["emission", "coherence", "dissonance"]

# ✅ Good
["emission", "coherence", "dissonance", "coherence"]
```

5. **Don't ignore health metrics**:
```python
# ❌ Bad
result = validate_sequence_with_health(sequence)
# Ignore health...

# ✅ Good
if result.health_metrics.overall_health < 0.65:
    improve_sequence()
```

---

## Quick Reference

### Validation Checklist

- [ ] Starts with valid initiator
- [ ] Ends with valid terminator
- [ ] All transitions compatible (no ✗)
- [ ] SELF_ORGANIZATION has destabilizer within 3 ops
- [ ] No forbidden patterns
- [ ] Frequency transitions valid (no Zero → High)

### Health Optimization Checklist

- [ ] Balance score >0.65 (stabilizers ≈ destabilizers)
- [ ] Sustainability >0.65 (ends with stabilizer)
- [ ] Coherence >0.70 (clear pattern)
- [ ] Frequency harmony >0.70 (no R5 violations)
- [ ] Overall health >0.65 (target: >0.80)

### Common Fixes

| Problem | Fix |
|---------|-----|
| Low balance | Add stabilizers after destabilizers |
| Low sustainability | End with COHERENCE/SILENCE/RESONANCE |
| R5 warning | Insert Medium operator between Zero and High |
| THOL error | Add DISSONANCE/MUTATION/CONTRACTION before THOL |
| Low efficiency | Remove redundancy, use patterns |
| UNKNOWN pattern | Follow documented pattern templates |

---

## Resources

- **[GLYPH_SEQUENCES_GUIDE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/GLYPH_SEQUENCES_GUIDE.md)**: Complete Grammar 2.0 guide
- **[MIGRATION_GUIDE_2.0.md](MIGRATION_GUIDE_2.0.md)**: Upgrading from 1.0
- **[HEALTH_METRICS_GUIDE.md](HEALTH_METRICS_GUIDE.md)**: Health metrics deep dive
- **[PATTERN_REFERENCE.md](PATTERN_REFERENCE.md)**: Pattern catalog
- **[examples/domain_applications/](https://github.com/fermga/TNFR-Python-Engine/tree/main/examples/domain_applications)**: Working examples

---

## Getting Help

If your issue isn't covered here:

1. Check the [complete guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/GLYPH_SEQUENCES_GUIDE.md)
2. Review [pattern reference](PATTERN_REFERENCE.md)
3. Study [domain examples](https://github.com/fermga/TNFR-Python-Engine/tree/main/examples/domain_applications)
4. Open a GitHub issue with:
   - Sequence that causes issue
   - Error message or unexpected behavior
   - Expected vs actual result

---

*Last updated: 2025-11-07*  
*Grammar version: 2.0*
