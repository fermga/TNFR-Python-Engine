# Migration Guide: TNFR Grammar 1.0 → 2.0

## Overview

Grammar 2.0 is a **backward-compatible enhancement** to TNFR operator sequence validation. All previously valid sequences remain valid, with additional warnings, health metrics, and pattern detection capabilities.

**TL;DR**: 
- ✅ All valid 1.0 sequences remain valid in 2.0
- ⚠️ New warnings may appear (non-breaking)
- 🎯 Enhanced pattern classification (backward compatible)
- 📊 Optional health metrics available

---

## Breaking Changes

While Grammar 2.0 maintains backward compatibility, there are a few areas where behavior has changed:

### 1. SELF_ORGANIZATION (THOL) Validation

**What Changed**: SELF_ORGANIZATION now requires a destabilizer within 3-operator window before it.

**Why**: Self-organization emerges from instability, not from stable states. This change enforces canonical TNFR principles.

#### Before (Grammar 1.0)
```python
# This was valid in 1.0
["emission", "reception", "self_organization"]  # ✅ Passed
```

#### After (Grammar 2.0)
```python
# Now requires destabilizer
["emission", "reception", "self_organization"]  # ❌ Fails

# Fixed versions:
["emission", "dissonance", "self_organization"]  # ✅ Passes
["dissonance", "reception", "self_organization"]  # ✅ Passes
["emission", "mutation", "reception", "self_organization"]  # ✅ Passes (within 3-op window)
```

**Migration Action**: Review all sequences using SELF_ORGANIZATION and ensure they have DISSONANCE, MUTATION, or CONTRACTION within 3 operators before.

---

### 2. Pattern Detection Changes

**What Changed**: Pattern detection is more specific and uses coherence-weighted scoring.

**Why**: Better recognition of domain-specific patterns and structural depth.

#### Before (Grammar 1.0)
```python
# This might be classified as "activation"
["reception", "emission", "coherence"]
```

#### After (Grammar 2.0)
```python
# Now classified more specifically
["reception", "emission", "coherence"]  # MINIMAL or LINEAR

# To get THERAPEUTIC pattern:
["reception", "emission", "coherence", 
 "dissonance", "self_organization", "coherence"]  # THERAPEUTIC
```

**Migration Action**: If your code relies on specific pattern classifications, review and update pattern matching logic.

---

### 3. Validation Warnings

**What Changed**: CAUTION-level transitions now generate warnings instead of silently passing.

**Why**: Provides feedback on potentially problematic transitions while maintaining flexibility.

#### Before (Grammar 1.0)
```python
result = validate_sequence(["emission", "silence"])
# Passes silently, no warnings
```

#### After (Grammar 2.0)
```python
result = validate_sequence(["emission", "silence"])
# Still passes, but may include warnings in metadata
print(result.metadata.get('warnings', []))
# ['CAUTION transition: emission → silence']
```

**Migration Action**: Check for warnings in validation results if you want to handle them.

---

## New Capabilities

### 1. Structural Frequencies (R5)

**New Feature**: Each operator has a defined structural frequency (νf) in Hz_str units.

**Usage**:
```python
from tnfr.operators.grammar import STRUCTURAL_FREQUENCIES, validate_frequency_transition

# Check operator frequency
freq = STRUCTURAL_FREQUENCIES["emission"]  # "high"

# Validate frequency transitions
is_valid, msg = validate_frequency_transition("silence", "emission")
print(is_valid)  # False - Zero → High invalid
print(msg)  # Explains frequency mismatch
```

**Migration Benefit**: Better understanding of energy flow in sequences.

---

### 2. Health Metrics System

**New Feature**: Quantitative assessment of sequence quality (0.0-1.0).

**Usage**:
```python
from tnfr.operators.grammar import validate_sequence_with_health

# Old way (still works)
result = validate_sequence(["emission", "coherence"])

# New way (with health metrics)
result = validate_sequence_with_health(["emission", "coherence"])
print(result.health_metrics.overall_health)  # 0.75
print(result.health_metrics.coherence_index)  # 0.80
print(result.health_metrics.balance_score)  # 0.85
```

**Migration Benefit**: Objective quality assessment and optimization guidance.

---

### 3. 18 Structural Patterns

**New Feature**: Comprehensive pattern typology (was ~6 patterns, now 18).

**Usage**:
```python
from tnfr.operators.patterns import AdvancedPatternDetector

detector = AdvancedPatternDetector()
pattern = detector.detect_pattern([
    "reception", "emission", "coherence",
    "dissonance", "self_organization", "coherence"
])
print(pattern.value)  # "therapeutic"
```

**Pattern Categories**:
- **Fundamental**: LINEAR, HIERARCHICAL, FRACTAL, CYCLIC, BIFURCATED
- **Domain**: THERAPEUTIC, EDUCATIONAL, ORGANIZATIONAL, CREATIVE, REGENERATIVE
- **Compositional**: BOOTSTRAP, EXPLORE, STABILIZE, RESONATE, COMPRESS
- **Complexity**: COMPLEX, MINIMAL, UNKNOWN

**Migration Benefit**: Better pattern recognition and classification.

---

### 4. Regenerative Cycles (R5)

**New Feature**: Validation of self-sustaining sequences.

**Usage**:
```python
from tnfr.operators.cycle_detection import CycleDetector

detector = CycleDetector()
sequence = ["coherence", "silence", "transition", "emission", "coherence"]
regenerator_index = 2  # "transition" at index 2

analysis = detector.analyze_potential_cycle(sequence, regenerator_index)
print(analysis.is_valid_regenerative)  # True/False
print(analysis.cycle_type.value)  # "transformative"
print(analysis.health_score)  # 0.0-1.0
```

**Migration Benefit**: Design and validate self-sustaining processes.

---

### 5. Graduated Compatibility

**New Feature**: Three-level compatibility system.

**Levels**:
- ✅ **COMPATIBLE**: Recommended transitions (highest compatibility)
- ⚠️ **CAUTION**: Context-specific transitions (generates warnings)
- ❌ **INCOMPATIBLE**: Invalid transitions (validation fails)

**Usage**:
```python
# CAUTION transitions now generate warnings
result = validate_sequence_with_health(["silence", "emission"])
# Still passes, but:
print(result.metadata.get('warnings'))  # Frequency warning
```

**Migration Benefit**: More nuanced validation feedback.

---

## Migration Steps

### Step 1: Update Validation Calls

**Old Code**:
```python
from tnfr.operators.grammar import validate_sequence

result = validate_sequence(operators)
if not result.passed:
    handle_error(result.message)
```

**New Code** (with health metrics):
```python
from tnfr.operators.grammar import validate_sequence_with_health

result = validate_sequence_with_health(operators)
if not result.passed:
    handle_error(result.message)
else:
    # Optionally check health
    if result.health_metrics.overall_health < 0.65:
        log_warning(f"Low health: {result.health_metrics.overall_health:.2f}")
        log_recommendations(result.health_metrics.recommendations)
```

**Note**: You can continue using `validate_sequence()` if you don't need health metrics.

---

### Step 2: Review SELF_ORGANIZATION Usage

**Find all SELF_ORGANIZATION sequences**:
```python
# Search your codebase for sequences containing "self_organization" or THOL
```

**Check for destabilizers**:
```python
def has_nearby_destabilizer(sequence, thol_index):
    """Check if THOL has destabilizer within 3-operator window."""
    destabilizers = {"dissonance", "mutation", "contraction"}
    window_start = max(0, thol_index - 3)
    window = sequence[window_start:thol_index]
    return any(op in destabilizers for op in window)

# Example
sequence = ["emission", "reception", "self_organization"]
thol_index = sequence.index("self_organization")
if not has_nearby_destabilizer(sequence, thol_index):
    print("⚠️ NEEDS UPDATE: Add destabilizer before THOL")
```

**Fix patterns**:
```python
# Before
["emission", "reception", "self_organization"]

# After - add destabilizer
["emission", "dissonance", "self_organization"]
# or
["dissonance", "emission", "reception", "self_organization"]
```

---

### Step 3: Optionally Adopt Health Metrics

**Benefits of health metrics**:
- Objective quality measurement
- Optimization guidance via recommendations
- Early detection of structural issues

**Integration example**:
```python
def validate_with_quality_check(sequence, min_health=0.65):
    """Validate sequence with minimum health requirement."""
    result = validate_sequence_with_health(sequence)
    
    if not result.passed:
        return False, f"Validation failed: {result.message}"
    
    health = result.health_metrics.overall_health
    if health < min_health:
        recommendations = "\n".join(result.health_metrics.recommendations)
        return False, f"Health {health:.2f} below {min_health}.\n{recommendations}"
    
    return True, f"Valid with health {health:.2f}"

# Use it
is_valid, message = validate_with_quality_check(my_sequence)
```

---

### Step 4: Update Pattern Matching

**If you relied on specific pattern names**:
```python
# Old code
result = validate_sequence(sequence)
if result.metadata.get('pattern') == 'activation':
    handle_activation()

# New code - patterns are more specific
result = validate_sequence_with_health(sequence)
pattern = result.metadata.get('detected_pattern')
if pattern in ['minimal', 'bootstrap', 'linear']:
    handle_simple_activation()
elif pattern in ['therapeutic', 'educational']:
    handle_domain_activation()
```

**Migration tip**: Use pattern categories rather than specific names for more robust code.

---

### Step 5: Test Your Sequences

**Run comprehensive tests**:
```python
import pytest
from tnfr.operators.grammar import validate_sequence_with_health

def test_all_sequences_valid_in_2_0():
    """Ensure all 1.0 sequences remain valid in 2.0."""
    sequences = [
        # Your existing sequences
        ["emission", "coherence"],
        ["reception", "coupling", "resonance"],
        # ... etc
    ]
    
    for seq in sequences:
        result = validate_sequence_with_health(seq)
        assert result.passed, f"Sequence failed: {seq}\nError: {result.message}"

def test_health_above_threshold():
    """Optionally test health scores."""
    sequences = [
        # Your production sequences
    ]
    
    for seq in sequences:
        result = validate_sequence_with_health(seq)
        if result.passed:
            health = result.health_metrics.overall_health
            if health < 0.65:
                print(f"⚠️ Low health {health:.2f} for: {seq}")
                print(f"   Recommendations: {result.health_metrics.recommendations}")
```

---

## Compatibility Guarantees

### What's Guaranteed

✅ **All valid 1.0 sequences remain valid in 2.0**  
- Exception: SELF_ORGANIZATION without destabilizer (now correctly invalid)

✅ **API stability**  
- All 1.0 functions still work
- New functions are additions, not replacements

✅ **Validation behavior**  
- Valid sequences still pass
- Invalid sequences still fail
- New warnings are informational only

### What's Not Guaranteed

⚠️ **Pattern classification may change**  
- Sequences may be classified differently (more specific patterns)
- Code depending on exact pattern names needs review

⚠️ **Warning messages may appear**  
- CAUTION transitions generate warnings
- Frequency harmony issues generate warnings
- Warnings don't block validation

⚠️ **Health scores are new**  
- No backward compatibility for non-existent metrics
- Health scores are optional enhancements

---

## Examples: Before & After

### Example 1: Simple Activation

**Grammar 1.0**:
```python
result = validate_sequence(["emission", "coherence"])
print(result.passed)  # True
# No additional information
```

**Grammar 2.0**:
```python
result = validate_sequence_with_health(["emission", "coherence"])
print(result.passed)  # True
print(result.health_metrics.overall_health)  # 0.75
print(result.metadata['detected_pattern'])  # 'minimal'
print(result.health_metrics.recommendations)  # []
```

---

### Example 2: Therapeutic Sequence

**Grammar 1.0**:
```python
therapeutic = [
    "emission", "reception", "self_organization", "coherence"  # Missing destabilizer!
]
result = validate_sequence(therapeutic)
print(result.passed)  # True (incorrectly allowed)
```

**Grammar 2.0**:
```python
# Old sequence now correctly fails
therapeutic_old = [
    "emission", "reception", "self_organization", "coherence"
]
result = validate_sequence_with_health(therapeutic_old)
print(result.passed)  # False (R3 violation)

# Fixed version
therapeutic_new = [
    "reception", "emission", "coherence",
    "dissonance", "self_organization", "coherence"
]
result = validate_sequence_with_health(therapeutic_new)
print(result.passed)  # True
print(result.metadata['detected_pattern'])  # 'therapeutic'
print(result.health_metrics.overall_health)  # 0.88
```

---

### Example 3: Frequency Harmony

**Grammar 1.0**:
```python
# No frequency validation
sequence = ["silence", "emission"]
result = validate_sequence(sequence)
print(result.passed)  # True
# No feedback about frequency jump
```

**Grammar 2.0**:
```python
# Frequency validation with warnings
sequence = ["silence", "emission"]
result = validate_sequence_with_health(sequence)
print(result.passed)  # True (still passes)
print(result.metadata.get('warnings'))  # Warning about Zero→High
print(result.health_metrics.frequency_harmony)  # ~0.4 (low)

# Better version
sequence_fixed = ["silence", "transition", "emission"]
result = validate_sequence_with_health(sequence_fixed)
print(result.health_metrics.frequency_harmony)  # ~0.95 (excellent)
```

---

## Troubleshooting

### Issue: "SELF_ORGANIZATION requires destabilizer"

**Problem**: Sequence has SELF_ORGANIZATION without recent destabilizer.

**Solution**: Add DISSONANCE, MUTATION, or CONTRACTION within 3 operators before SELF_ORGANIZATION.

```python
# Before
["emission", "reception", "self_organization"]

# After (option 1)
["emission", "dissonance", "self_organization"]

# After (option 2)
["dissonance", "emission", "reception", "self_organization"]
```

---

### Issue: "Pattern changed from 1.0"

**Problem**: Sequence now classified as different pattern.

**Solution**: Use pattern categories, not specific names:

```python
# Fragile - depends on exact name
if pattern == 'activation':
    ...

# Robust - uses category
if pattern in ['minimal', 'bootstrap', 'linear', 'stabilize']:
    handle_simple_patterns()
elif pattern in ['therapeutic', 'educational', 'organizational']:
    handle_domain_patterns()
```

---

### Issue: "Frequency harmony warnings"

**Problem**: Zero → High frequency jumps generate warnings.

**Solution**: Insert Medium-frequency operator bridge:

```python
# Generates warning
["silence", "emission"]

# Fixed
["silence", "transition", "emission"]  # Zero → Medium → High
# or
["silence", "coherence", "emission"]   # Zero → Medium → High
```

---

### Issue: "Health score lower than expected"

**Problem**: Sequence valid but health score <0.65.

**Solution**: Check recommendations and optimize:

```python
result = validate_sequence_with_health(sequence)
if result.health_metrics.overall_health < 0.65:
    print("Recommendations:")
    for rec in result.health_metrics.recommendations:
        print(f"  - {rec}")
    
    # Common fixes:
    # 1. Add stabilizer after destabilizer
    # 2. End with stabilizer (coherence/silence/resonance)
    # 3. Balance stabilizers and destabilizers
    # 4. Fix frequency transitions
```

---

## Quick Reference

### API Changes

| 1.0 Function | 2.0 Function | Change |
|--------------|--------------|--------|
| `validate_sequence()` | `validate_sequence()` | **No change** - still available |
| N/A | `validate_sequence_with_health()` | **New** - enhanced validation |
| N/A | `validate_frequency_transition()` | **New** - R5 validation |
| Pattern in metadata | `AdvancedPatternDetector` | **Enhanced** - 18 patterns |

### Validation Changes

| Aspect | 1.0 | 2.0 | Action |
|--------|-----|-----|--------|
| SELF_ORGANIZATION | Allowed without destabilizer | Requires destabilizer | Review sequences |
| Patterns | ~6 basic patterns | 18 detailed patterns | Update pattern matching |
| Warnings | Silent on CAUTION | Warnings generated | Handle warnings if needed |
| Health metrics | Not available | 7 dimensions | Optionally adopt |
| Frequency | Not validated | R5 validation | Fix Zero→High jumps |

---

## Resources

- **[GLYPH_SEQUENCES_GUIDE.md](../GLYPH_SEQUENCES_GUIDE.md)**: Complete Grammar 2.0 documentation
- **[docs/HEALTH_METRICS_GUIDE.md](HEALTH_METRICS_GUIDE.md)**: Deep dive into health metrics
- **[docs/PATTERN_REFERENCE.md](PATTERN_REFERENCE.md)**: Complete pattern catalog
- **[examples/domain_applications/](../examples/domain_applications/)**: Updated examples

---

## Summary

Grammar 2.0 is a **significant enhancement** that:
- ✅ Maintains backward compatibility (with THOL fix)
- 📊 Adds quantitative quality assessment
- 🎯 Provides more specific pattern detection
- 🔄 Enables regenerative cycle validation
- ⚡ Validates structural frequency harmony

**Migration is straightforward**:
1. Review SELF_ORGANIZATION sequences (add destabilizers)
2. Optionally adopt `validate_sequence_with_health()`
3. Update pattern matching if using specific names
4. Test your sequences
5. Enjoy the new capabilities!

**Questions?** See [GLYPH_SEQUENCES_GUIDE.md](../GLYPH_SEQUENCES_GUIDE.md) or open an issue.

---

*Last updated: 2025-11-07*  
*Grammar version: 2.0*
