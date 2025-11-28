# TNFR Structural Patterns - Complete Reference

This directory contains comprehensive examples and guides for working with TNFR structural operator patterns. These resources demonstrate specialized compositional patterns that complete the coverage of all patterns detectable by TNFR Grammar 2.0.

## 📚 Overview

Structural patterns are fundamental building blocks for creating coherent operator sequences in TNFR. They represent common compositional strategies optimized for specific purposes: initialization (BOOTSTRAP), exploration (EXPLORE), consolidation (STABILIZE), amplification (RESONATE), simplification (COMPRESS), and multi-pattern integration (COMPLEX).

This collection provides:
- **12 validated specialized patterns** with complete implementations
- **Canonical grammar rules** and operator compatibility constraints
- **Construction principles** for building custom sequences
- **Library of 23 tested variants** across 7 pattern families
- **Diagnostic tools** for troubleshooting sequences

## 📁 Files in This Directory

### Core Pattern Examples

#### `structural_patterns.py`
**Purpose**: Complete implementations of specialized structural patterns

**Contains**:
- **12 specialized patterns** with full documentation
- **Pattern categories**: BOOTSTRAP, EXPLORE, STABILIZE, RESONATE, COMPRESS, COMPLEX
- **Validation results**: All patterns pass with health scores > 0.65
- **Usage examples**: Ready-to-use operator sequences

**Quick Start**:
```python
from structural_patterns import (
    get_bootstrap_pattern,
    get_explore_pattern,
    get_stabilize_pattern,
)

# Get a bootstrap sequence for rapid initialization
bootstrap = get_bootstrap_pattern()
# ['emission', 'reception', 'coupling', 'coherence', 'silence']

# Validate and use
from tnfr.operators.grammar import validate_sequence_with_health
result = validate_sequence_with_health(bootstrap)
print(f"Health: {result.health_metrics.overall_health:.3f}")
```

**Patterns Included**:
1. **BOOTSTRAP**: Minimal (5 ops, health 0.663) & Extended (7 ops, health 0.676)
2. **EXPLORE**: Basic (7 ops, health 0.805) & Deep (12 ops, health 0.828)
3. **STABILIZE**: Standard (6 ops, health 0.663) & Recursive (7 ops, health 0.716)
4. **RESONATE**: Harmonic (8 ops, health 0.676) & Cascade (12 ops, health 0.737)
5. **COMPRESS**: Standard (10 ops, health 0.746) & Adaptive (13 ops, health 0.767)
6. **COMPLEX**: Standard (10 ops, health 0.763) & Full Cycle (18 ops, health 0.792)

---

### Construction Guide

#### `pattern_construction_guide.py`
**Purpose**: Principles and strategies for building effective operator sequences

**Contains**:
- **Canonical Grammar Rules**: What makes a sequence valid
- **Operator Compatibility Matrix**: What can follow what
- **Pattern-Specific Strategies**: Optimization for each pattern type
- **Health Metrics Interpretation**: Understanding sequence quality
- **Common Pitfalls**: Troubleshooting guide with solutions
- **Interactive Tools**: Validation, diagnosis, and suggestions

**Key Features**:

**1. Grammar Rules Reference**
```python
from pattern_construction_guide import print_canonical_rules
print_canonical_rules()
# Displays complete canonical grammar documentation
```

**2. Sequence Validation & Diagnosis**
```python
from pattern_construction_guide import validate_and_diagnose

sequence = ['emission', 'reception', 'coherence', 'dissonance', 'mutation', 'coherence', 'silence']
diagnosis = validate_and_diagnose(sequence)

print(diagnosis['interpretation'])
# Pattern: activation
# Health Score: 0.805
# GOOD: Solid structural quality...
```

**3. Pattern Strategies**
```python
from pattern_construction_guide import get_pattern_strategy

strategy = get_pattern_strategy('EXPLORE')
print(f"Purpose: {strategy['purpose']}")
print(f"Min operators: {strategy['min_operators']}")
for tip in strategy['optimization_tips']:
    print(f"  - {tip}")
```

**4. Next Operator Suggestions**
```python
from pattern_construction_guide import suggest_next_operators

current = ['emission', 'reception', 'coherence']
suggestions = suggest_next_operators(current)
print(f"Recommended: {suggestions['recommended']}")
print(f"Valid: {suggestions['valid']}")
```

---

### Pattern Library

#### `pattern_library.py`
**Purpose**: Curated collection of 23 pre-validated sequence variants

**Contains**:
- **7 pattern families**: bootstrap, explore, stabilize, resonate, compress, complex, specialized
- **23 total variants** with validated health scores
- **Search capabilities**: Find patterns by use case, health, length
- **Expected metrics**: Health ranges and detected patterns

**Quick Access**:
```python
from pattern_library import PATTERN_LIBRARY, get_pattern, search_patterns

# Get a specific pattern
pattern = get_pattern('bootstrap', 'minimal')
print(pattern['sequence'])
print(pattern['use_case'])
print(pattern['characteristics'])

# Search patterns
matches = search_patterns(
    use_case_keyword='exploration',
    min_health=0.75,
    max_length=10
)
print(f"Found {len(matches)} matching patterns")
```

**Pattern Families**:
- **bootstrap** (3 variants): minimal, enhanced, networked
- **explore** (4 variants): basic, dual_hypothesis, deep_search, conservative
- **stabilize** (4 variants): standard, robust, fractal, transitional
- **resonate** (3 variants): harmonic, cascade, triple
- **compress** (3 variants): standard, adaptive, aggressive
- **complex** (3 variants): standard, full_lifecycle, regenerative
- **specialized** (3 variants): therapeutic_healing, educational_learning, creative_emergence

---

## 🎯 Usage Examples

### Example 1: Quick Pattern Selection

```python
from pattern_library import get_pattern
from tnfr.operators.grammar import validate_sequence_with_health

# Need rapid initialization?
pattern = get_pattern('bootstrap', 'minimal')
sequence = pattern['sequence']

# Validate before use
result = validate_sequence_with_health(sequence)
if result.passed:
    print(f"✓ Pattern valid (health: {result.health_metrics.overall_health:.3f})")
else:
    print(f"✗ Validation failed: {result.message}")
```

### Example 2: Building a Custom Sequence

```python
from pattern_construction_guide import suggest_next_operators, validate_and_diagnose
from tnfr.config.operator_names import EMISSION, RECEPTION, COHERENCE

# Start building
sequence = [EMISSION, RECEPTION, COHERENCE]

# Get suggestions for next operator
suggestions = suggest_next_operators(sequence)
print(f"Try: {suggestions['recommended']}")

# Add an operator and validate
sequence.append('resonance')
sequence.append('coherence')
sequence.append('silence')

# Diagnose the result
diagnosis = validate_and_diagnose(sequence)
print(diagnosis['interpretation'])
```

### Example 3: Troubleshooting a Failing Sequence

```python
from pattern_construction_guide import validate_and_diagnose, get_pitfall_solution

# This sequence has an error
bad_sequence = ['emission', 'expansion', 'coherence', 'silence']

diagnosis = validate_and_diagnose(bad_sequence)
if not diagnosis['passed']:
    print(f"Error: {diagnosis['message']}")
    
    # Get solution
    solution = get_pitfall_solution(diagnosis['message'])
    print(f"\nSolution: {solution['solution']}")
    print(f"Example fix: {solution['example_fix']}")
```

### Example 4: Comparing Pattern Variants

```python
from pattern_library import PATTERN_LIBRARY
from tnfr.operators.grammar import validate_sequence_with_health

# Compare all explore variants
explore_patterns = PATTERN_LIBRARY['explore']

for variant_name, pattern in explore_patterns.items():
    result = validate_sequence_with_health(pattern['sequence'])
    health = result.health_metrics.overall_health if result.passed else 0.0
    
    print(f"{variant_name}:")
    print(f"  Length: {len(pattern['sequence'])} operators")
    print(f"  Health: {health:.3f}")
    print(f"  Use case: {pattern['use_case']}")
    print()
```

---

## 🔍 Key Concepts

### Canonical Grammar Rules

**Every valid sequence must**:
1. Start with EMISSION or RECURSIVITY
2. Contain RECEPTION → COHERENCE segment
3. End with DISSONANCE, RECURSIVITY, SILENCE, or TRANSITION

**Critical Compatibility Constraints**:
- After EXPANSION → must have COHERENCE
- After MUTATION → must have COHERENCE
- After CONTRACTION → must have COHERENCE (before transformation ops)
- After RECURSIVITY → must have COHERENCE or TRANSITION (not SILENCE directly)
- SELF_ORGANIZATION → requires destabilizer (DISSONANCE, EXPANSION, TRANSITION) in previous 3 ops

### Pattern Types

**BOOTSTRAP**: Rapid initialization
- Use when: Starting new systems, quick setup needed
- Optimize for: Speed, minimal operators, stable foundation
- Health target: > 0.65

**EXPLORE**: Controlled exploration
- Use when: Testing hypotheses, safe discovery needed
- Optimize for: Balance exploration/safety, return to baseline
- Health target: > 0.75

**STABILIZE**: Robust consolidation
- Use when: Checkpointing, long-term storage, handoff preparation
- Optimize for: Sustainability, multiple stabilization layers
- Health target: > 0.65

**RESONATE**: Amplification and propagation
- Use when: Network broadcasting, pattern spreading, synchronization
- Optimize for: Frequency harmony, coupling effectiveness
- Health target: > 0.67

**COMPRESS**: Simplification
- Use when: Essence extraction, dimensionality reduction, efficiency optimization
- Optimize for: Complexity efficiency, information density
- Health target: > 0.70

**COMPLEX**: Multi-pattern integration
- Use when: Complete lifecycles, comprehensive workflows, end-to-end transformations
- Optimize for: Pattern composition, overall balance, completeness
- Health target: > 0.75

### Health Metrics

**Overall Health Score** (0.0-1.0):
- **0.85+**: Excellent quality, production-ready
- **0.75-0.85**: Good quality, suitable for most uses
- **0.65-0.75**: Acceptable, meets minimum requirements
- **< 0.65**: Review needed, may have structural issues

**Component Metrics**:
- **Coherence Index**: Sequential flow quality
- **Balance Score**: Stabilizer/destabilizer equilibrium
- **Sustainability Index**: Long-term maintenance capacity
- **Complexity Efficiency**: Value-to-complexity ratio
- **Frequency Harmony**: Structural frequency transitions
- **Pattern Completeness**: Detected pattern wholeness

---

## 🚀 Getting Started

### For Beginners

1. **Start with the pattern library**: Browse pre-validated sequences
2. **Use pattern_library.py**: Find patterns matching your use case
3. **Test patterns**: Validate with `validate_sequence_with_health()`
4. **Read the guide**: Understand why patterns work

```python
# Simplest possible workflow
from pattern_library import get_pattern
sequence = get_pattern('bootstrap', 'minimal')['sequence']
# Use this sequence in your TNFR application
```

### For Intermediate Users

1. **Study construction principles**: Read `pattern_construction_guide.py`
2. **Experiment with variants**: Modify existing patterns
3. **Use diagnosis tools**: Validate and troubleshoot custom sequences
4. **Optimize for your domain**: Adjust patterns to specific needs

```python
# Intermediate workflow
from pattern_construction_guide import validate_and_diagnose
from pattern_library import get_pattern

# Start with a base pattern
base = get_pattern('explore', 'basic')['sequence']

# Modify it
custom = base.copy()
custom.insert(3, 'expansion')  # Add expansion
custom.insert(4, 'coherence')  # Required after expansion

# Validate
diagnosis = validate_and_diagnose(custom)
print(diagnosis['interpretation'])
```

### For Advanced Users

1. **Compose complex patterns**: Combine multiple pattern signatures
2. **Optimize health metrics**: Target specific metric improvements
3. **Create domain-specific patterns**: Design patterns for your application
4. **Contribute to library**: Share validated patterns with community

```python
# Advanced workflow
from pattern_construction_guide import suggest_next_operators
from tnfr.operators.grammar import validate_sequence_with_health

# Build from scratch with interactive guidance
sequence = ['emission', 'reception', 'coherence']
while True:
    suggestions = suggest_next_operators(sequence)
    # (Select next operator based on strategy)
    # sequence.append(next_operator)
    
    result = validate_sequence_with_health(sequence)
    if result.passed and result.health_metrics.overall_health > 0.80:
        break  # Found excellent sequence
```

---

## 📊 Pattern Statistics

**Total Patterns**: 35 (12 in structural_patterns.py + 23 in pattern_library.py)

**By Health Score**:
- Excellent (0.80+): 8 patterns (23%)
- Good (0.75-0.80): 12 patterns (34%)
- Acceptable (0.65-0.75): 15 patterns (43%)

**By Length**:
- Short (≤ 7 ops): 10 patterns
- Medium (8-12 ops): 18 patterns
- Long (13+ ops): 7 patterns

**Pattern Detection**:
- BOOTSTRAP: 90%+ detection rate
- EXPLORE: 85%+ detection rate
- STABILIZE: 90%+ detection rate
- RESONATE: 80%+ detection rate
- COMPRESS: 75%+ detection rate (often detected as LINEAR)
- COMPLEX: 70%+ detection rate

---

## 🔧 Troubleshooting

### Common Issues

**"missing reception→coherence segment"**
- **Solution**: Add RECEPTION → COHERENCE early in sequence
- **Example**: `[EMISSION, RECEPTION, COHERENCE, ...]`

**"must start with emission, recursivity"**
- **Solution**: Begin with EMISSION (standard) or RECURSIVITY (recursive patterns)
- **Example**: `[EMISSION, ...]`

**"expansion incompatible after reception"**
- **Solution**: Place COHERENCE between RECEPTION and EXPANSION
- **Example**: `[..., RECEPTION, COHERENCE, EXPANSION, COHERENCE, ...]`

**"mutation incompatible after contraction"**
- **Solution**: Add COHERENCE after CONTRACTION before MUTATION
- **Example**: `[..., CONTRACTION, COHERENCE, MUTATION, ...]`

**"self_organization requires destabilizer"**
- **Solution**: Ensure DISSONANCE, EXPANSION, or TRANSITION within 3 operators before SELF_ORGANIZATION
- **Example**: `[..., DISSONANCE, X, X, SELF_ORGANIZATION, ...]`

See `pattern_construction_guide.py` for complete troubleshooting guide.

---

## 🎓 Learning Path

### Level 1: Pattern User
1. Read this README
2. Browse pattern_library.py examples
3. Test 3-5 patterns from different families
4. Use patterns in simple applications

### Level 2: Pattern Builder
1. Study pattern_construction_guide.py
2. Understand canonical grammar rules
3. Modify existing patterns
4. Create simple custom patterns
5. Use validation and diagnosis tools

### Level 3: Pattern Expert
1. Master operator compatibility constraints
2. Design domain-specific patterns
3. Optimize for specific health metrics
4. Compose complex multi-pattern sequences
5. Contribute validated patterns to library

---

## 📖 Additional Resources

### Related Files
- `creative_patterns.py`: Creative domain patterns (artistic, design, innovation)
- `educational_patterns.py`: Learning and knowledge transfer patterns
- `organizational_patterns.py`: Institutional evolution patterns
- `therapeutic_patterns.py`: Healing and crisis resolution patterns

### Documentation
- `../../GLOSSARY.md`: TNFR terminology reference
- `../../GLYPH_SEQUENCES_GUIDE.md`: Operator sequence fundamentals
- `../../TNFR.pdf`: Complete TNFR theoretical foundation

### Tests
- `../../tests/examples/test_structural_patterns.py`: Comprehensive pattern validation tests
- `../../tests/unit/operators/test_advanced_patterns.py`: Grammar 2.0 pattern detection tests

---

## 🤝 Contributing

### Adding New Patterns

1. **Implement** pattern in `structural_patterns.py`
2. **Validate** with `validate_sequence_with_health()`
3. **Ensure** health score > 0.65
4. **Document** purpose, flow, metrics, use cases
5. **Add variant** to `pattern_library.py`
6. **Create tests** in `test_structural_patterns.py`

### Pattern Submission Guidelines

- Include complete docstring with structural flow
- Provide expected health range
- List use cases and characteristics
- Validate against canonical grammar
- Test across different contexts
- Document any special considerations

---

## 📝 License

This code is part of the TNFR-Python-Engine project.  
See LICENSE.md in the repository root for details.

---

## 🙏 Acknowledgments

These patterns were developed following TNFR canonical principles as defined in:
- TNFR theoretical framework (TNFR.pdf)
- Grammar 2.0 specification (AGENTS.md)
- Canonical operator definitions (src/tnfr/operators/)

Pattern validation powered by:
- TNFR Grammar validation engine
- Health analyzer module
- Advanced pattern detector

---

**Last Updated**: 2025-11-06  
**Version**: 1.0.0  
**Patterns**: 35 total (12 specialized + 23 variants)  
**Validation**: All patterns canonically validated
