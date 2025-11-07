# TNFR Sequence Generator

Context-guided sequence generation tools for creating optimal TNFR operator sequences.

## Overview

The sequence generator provides intelligent tools to construct TNFR operator sequences based on:
- **Domain and objective** (therapeutic, educational, organizational, creative)
- **Target structural patterns** (BOOTSTRAP, THERAPEUTIC, STABILIZE, etc.)
- **Sequence improvement** with explanatory recommendations
- **Flexible constraints** (health, length, pattern requirements)

All generated sequences respect TNFR canonical principles:
- ✅ Operator closure (only canonical operators)
- ✅ Phase coherence (compatible transitions)
- ✅ Structural health (balanced forces, proper closure)
- ✅ Operational fractality (composable patterns)

## Quick Start

### Python API

```python
from tnfr.tools import ContextualSequenceGenerator

# Initialize generator
generator = ContextualSequenceGenerator(seed=42)

# Generate for domain and objective
result = generator.generate_for_context(
    domain="therapeutic",
    objective="crisis_intervention",
    min_health=0.75
)
print(result.sequence)
# ['emission', 'reception', 'coherence', 'resonance', 'silence']

# Generate for specific pattern
result = generator.generate_for_pattern(
    target_pattern="BOOTSTRAP",
    min_health=0.70
)

# Improve existing sequence
current = ["emission", "coherence", "silence"]
improved, recommendations = generator.improve_sequence(
    current,
    target_health=0.80
)
```

### Command Line Interface

```bash
# List available domains
tnfr-generate --list-domains

# List objectives for a domain
tnfr-generate --list-objectives therapeutic

# Generate for context
tnfr-generate --domain therapeutic --objective crisis_intervention --min-health 0.75

# Generate for pattern
tnfr-generate --pattern BOOTSTRAP --max-length 5

# Improve sequence
tnfr-generate --improve "emission,coherence,silence" --target-health 0.80

# JSON output
tnfr-generate --domain educational --objective skill_development --format json
```

## Domain Templates

### Therapeutic Domain
- **crisis_intervention**: Rapid stabilization for immediate crisis response
- **process_therapy**: Complete transformative therapeutic cycle
- **healing_cycle**: Gradual healing and integration process
- **trauma_processing**: Safe trauma processing with containment

### Educational Domain
- **concept_introduction**: Introduce new concepts with exploration
- **skill_development**: Progressive skill building with challenge
- **knowledge_integration**: Connect and integrate multiple concepts
- **transformative_learning**: Deep learning with paradigm shift

### Organizational Domain
- **change_management**: Organizational transformation process
- **team_building**: Build cohesive team dynamics
- **crisis_response**: Organizational crisis management
- **innovation_cycle**: Foster organizational innovation

### Creative Domain
- **artistic_process**: Creative work from conception to completion
- **design_thinking**: Design process from empathy to prototype
- **innovation**: Innovation through creative destruction
- **collaborative_creation**: Group creative process with emergent outcomes

## Features

### Context-Based Generation

Generate sequences optimized for specific domains and objectives:

```python
result = generator.generate_for_context(
    domain="educational",
    objective="skill_development",
    max_length=8,
    min_health=0.75
)
```

### Pattern-Targeted Generation

Generate sequences that maximize probability of matching target patterns:

```python
result = generator.generate_for_pattern(
    target_pattern="THERAPEUTIC",
    min_health=0.75
)
```

Supported patterns:
- **BOOTSTRAP**: System initialization
- **THERAPEUTIC**: Healing cycle with controlled crisis
- **EDUCATIONAL**: Transformative learning
- **ORGANIZATIONAL**: Institutional evolution
- **CREATIVE**: Artistic emergence
- **STABILIZE**: Consolidation and closure
- **EXPLORE**: Controlled exploration
- **RESONATE**: Amplification and propagation

### Sequence Improvement

Improve existing sequences with detailed recommendations:

```python
current = ["emission", "coherence", "silence"]
improved, recommendations = generator.improve_sequence(
    current,
    target_health=0.80
)

# Example recommendations:
# - "Overall health improved by 0.15"
# - "Added reception after emission: improves completeness (+0.25)"
# - "Balance improved by 0.10"
```

### Constraint Handling

All generation methods support flexible constraints:

```python
result = generator.generate_for_context(
    domain="therapeutic",
    objective="process_therapy",
    max_length=6,        # Maximum sequence length
    min_health=0.80,     # Minimum health score
)
```

### Deterministic Generation

Use seeds for reproducible results:

```python
gen1 = ContextualSequenceGenerator(seed=42)
gen2 = ContextualSequenceGenerator(seed=42)

# Both produce identical results
result1 = gen1.generate_for_context("therapeutic", "crisis_intervention")
result2 = gen2.generate_for_context("therapeutic", "crisis_intervention")
```

## Generation Metrics

Each generated sequence includes comprehensive metrics:

```python
result = generator.generate_for_context(...)

result.sequence           # List of operator names
result.health_score       # Overall health (0.0-1.0)
result.detected_pattern   # Primary structural pattern
result.domain             # Source domain (if applicable)
result.objective          # Source objective (if applicable)
result.method             # Generation method used
result.recommendations    # Improvement suggestions
result.metadata           # Additional generation metadata
```

## Examples

See the included examples:
- `examples/sequence_generator_demo.py` - Comprehensive demonstration
- `tests/tools/test_sequence_generator.py` - Usage examples in tests

## CLI Options

```
Generation Modes:
  --domain DOMAIN              Application domain
  --objective OBJECTIVE        Specific objective within domain
  --pattern PATTERN            Target structural pattern
  --improve SEQUENCE           Comma-separated sequence to improve

Constraints:
  --max-length N               Maximum sequence length (default: 10)
  --min-health SCORE          Minimum health score 0.0-1.0 (default: 0.70)
  --target-health SCORE       Target health for improvement

Output Options:
  --format {compact,detailed,json}  Output format (default: compact)
  --seed N                     Random seed for deterministic generation
  --quiet                      Only output the sequence

Listing Options:
  --list-domains               List all available domains
  --list-objectives DOMAIN     List objectives for a domain
```

## Test Coverage

The generator includes comprehensive tests covering:
- ✅ Domain templates (10 tests)
- ✅ Context-based generation (11 tests)
- ✅ Pattern-targeted generation (6 tests)
- ✅ Sequence improvement (5 tests)
- ✅ Health constraints (3 tests)
- ✅ Determinism (2 tests)
- ✅ Length constraints (2 tests)

**Total: 41 tests, 100% passing**

## Integration

The generator integrates with existing TNFR modules:
- **SequenceHealthAnalyzer**: Health metrics and recommendations
- **AdvancedPatternDetector**: Pattern detection and scoring
- **GRADUATED_COMPATIBILITY**: Transition validation
- **Domain Examples**: Template sequences

## API Reference

### ContextualSequenceGenerator

Main class for sequence generation.

**Methods:**
- `generate_for_context(domain, objective, max_length, min_health, required_pattern)` → GenerationResult
- `generate_for_pattern(target_pattern, max_length, min_health)` → GenerationResult
- `improve_sequence(current, target_health, max_length)` → tuple[list[str], list[str]]

### Domain Templates

**Functions:**
- `list_domains()` → list[str]
- `list_objectives(domain)` → list[str]
- `get_template(domain, objective)` → list[str]

**Constant:**
- `DOMAIN_TEMPLATES`: dict[str, dict[str, dict]]

### GenerationResult

Result dataclass with fields:
- `sequence`: list[str] - Generated operator sequence
- `health_score`: float - Overall health (0.0-1.0)
- `detected_pattern`: str - Primary structural pattern
- `domain`: str | None - Source domain
- `objective`: str | None - Source objective
- `method`: str - Generation method
- `recommendations`: list[str] - Improvement suggestions
- `metadata`: dict - Additional metadata

## License

MIT License - Part of TNFR Python Engine
