# TNFR Interactive Validator - User Guide

## Overview

The TNFR Interactive Validator is a user-friendly command-line tool that helps you work with TNFR (Resonant Fractal Nature Theory) operator sequences without requiring programming knowledge. It provides an intuitive interface for validating, generating, optimizing, and exploring sequences through simple menus and prompts.

## Quick Start

### Installation

The validator is included with the TNFR package:

```bash
pip install tnfr
```

### Launching the Interactive Validator

Simply run:

```bash
tnfr-validate --interactive
```

Or use the short form:

```bash
tnfr-validate -i
```

You'll be greeted with:

```
┌──────────────────────────────────────────────────────────┐
│          TNFR Interactive Sequence Validator             │
│               Grammar 2.0 - Full Capabilities            │
└──────────────────────────────────────────────────────────┘
```

## Main Features

The validator offers five main modes:

1. **[v] Validate** - Check if a sequence is valid and see its health metrics
2. **[g] Generate** - Create new sequences based on patterns or domains
3. **[o] Optimize** - Improve existing sequences
4. **[e] Explore** - Learn about patterns and domains
5. **[h] Help** - Get detailed documentation

## Mode Guides

### Validation Mode

**Purpose**: Validate a sequence and see detailed health metrics.

**Example Session**:

```
Main Menu:
Select option: v

VALIDATE SEQUENCE
────────────────────────────────────────────────────────────
Enter operators separated by spaces or commas.
Example: emission reception coherence silence

Sequence: emission reception coherence silence

✓ VALID SEQUENCE

┌─ Health Metrics ─────────────────────────────────────────┐
│ Overall Health:      ██████░░░░ 0.65 ⚠ (Moderate)
│ Coherence Index:     █████████░ 0.97
│ Balance Score:       ░░░░░░░░░░ 0.00
│ Sustainability:      ███████░░░ 0.80
│ Pattern Detected:    ACTIVATION
│ Sequence Length:     4
└──────────────────────────────────────────────────────────┘
```

**Input Formats**:
- Space-separated: `emission reception coherence`
- Comma-separated: `emission,reception,coherence`
- Mixed: `emission, reception coherence`

**Health Metrics Explained**:

| Metric | Range | Meaning |
|--------|-------|---------|
| Overall Health | 0.0-1.0 | Composite quality score |
| Coherence Index | 0.0-1.0 | How well operators flow together |
| Balance Score | 0.0-1.0 | Equilibrium between stability/change |
| Sustainability | 0.0-1.0 | Long-term maintenance capacity |

**Health Status Icons**:
- ✓ (Excellent): 0.8-1.0
- ⚠ (Moderate): 0.6-0.79
- ✗ (Poor): Below 0.6

### Generation Mode

**Purpose**: Create new sequences based on your needs.

#### Option 1: By Domain and Objective

Generate sequences tailored to specific application contexts.

**Example**:

```
GENERATE SEQUENCE
────────────────────────────────────────────────────────────

Generation mode:
  [d] By domain and objective
  [p] By structural pattern
  [b] Back to main menu

Select mode: d

Available domains:
  1. therapeutic
  2. educational
  3. organizational
  4. creative

Select domain (number): 1

Objectives for 'therapeutic':
  1. crisis_intervention
  2. trauma_integration
  3. emotional_regulation
  4. behavioral_change

Select objective (number, or 0 for any): 1

Generating sequence...

✓ GENERATED SEQUENCE

Sequence:  emission → reception → coherence → transition → silence
Health:    0.85 ✓
Pattern:   REGENERATIVE
Domain:    therapeutic
Objective: crisis_intervention

💡 Recommendations:
  1. Sequence achieves high health (0.85)
  2. Strong sustainability for crisis contexts
```

#### Option 2: By Structural Pattern

Generate sequences matching known TNFR patterns.

**Example**:

```
Select mode: p

Common structural patterns:
  1. BOOTSTRAP
  2. THERAPEUTIC
  3. STABILIZE
  4. REGENERATIVE
  5. EXPLORATION
  6. TRANSFORMATIVE
  7. COUPLING
  8. SIMPLE

Select pattern (number): 4

Generating REGENERATIVE sequence...

✓ GENERATED SEQUENCE

Sequence:  emission → reception → coherence → recursivity → silence
Health:    0.82 ✓
Pattern:   REGENERATIVE
```

### Optimization Mode

**Purpose**: Improve existing sequences that have low health scores.

**Example**:

```
OPTIMIZE SEQUENCE
────────────────────────────────────────────────────────────
Enter the sequence you want to improve.

Current sequence: emission coherence silence

Target health score (0.0-1.0, or Enter for default): 0.8

Optimizing...

✓ OPTIMIZATION COMPLETE

Original:  emission → coherence → silence
  Health:  0.62 ⚠

Improved:  emission → reception → coherence → resonance → silence
  Health:  0.83 ✓
  Delta:   +0.21 ✓

Changes made:
  1. Added reception for better information flow
  2. Added resonance to improve propagation
  3. Overall health improved by 0.21
```

### Exploration Mode

**Purpose**: Learn about TNFR domains, objectives, and patterns.

**Options**:

```
EXPLORE
────────────────────────────────────────────────────────────

  [d] List all domains
  [o] List objectives for a domain
  [p] Learn about structural patterns
  [b] Back to main menu
```

#### List Domains

Shows all available application domains:

```
Select option: d

Available Domains:
  • therapeutic
  • educational
  • organizational
  • creative
```

#### List Objectives

Shows objectives for a specific domain:

```
Select option: o

Domain name: therapeutic

Objectives for 'therapeutic':
  • crisis_intervention
  • trauma_integration
  • emotional_regulation
  • behavioral_change
  • resilience_building
```

#### Learn Patterns

Explains structural patterns:

```
Select option: p

Structural Patterns:

  BOOTSTRAP       - Initialize new nodes/systems
  THERAPEUTIC     - Healing and stabilization
  STABILIZE       - Maintain coherent structure
  REGENERATIVE    - Self-renewal and growth
  EXPLORATION     - Discovery with dissonance
  TRANSFORMATIVE  - Phase transitions
  COUPLING        - Network formation
  SIMPLE          - Minimal effective sequences
```

### Help Mode

**Purpose**: Get detailed information about TNFR operators and metrics.

```
HELP & DOCUMENTATION
════════════════════════════════════════════════════════════

TNFR (Resonant Fractal Nature Theory) Operators:

  emission      - Initiate resonant pattern (AL)
  reception     - Receive and integrate patterns (EN)
  coherence     - Stabilize structure (IL)
  dissonance    - Introduce controlled instability (OZ)
  coupling      - Create structural links (UM)
  resonance     - Amplify and propagate (RA)
  silence       - Freeze evolution temporarily (SHA)
  expansion     - Increase complexity (VAL)
  contraction   - Reduce complexity (NUL)
  self_organization - Spontaneous pattern formation (THOL)
  mutation      - Phase transformation (ZHIR)
  transition    - Movement between states (NAV)
  recursivity   - Nested operations (REMESH)

Health Metrics:

  Overall Health    - Composite quality score (0.0-1.0)
  Coherence Index   - Sequential flow quality
  Balance Score     - Stability/instability equilibrium
  Sustainability    - Long-term maintenance capacity
```

## Advanced Usage

### Deterministic Generation

For reproducible results, use a seed:

```bash
tnfr-validate -i --seed 42
```

This ensures the same sequences are generated each time with the same inputs.

### Keyboard Shortcuts

- **Ctrl+C**: Return to main menu (doesn't exit)
- **Ctrl+D**: Exit the validator
- **Enter**: Use default values when prompted

## Common Workflows

### Workflow 1: Validating an Existing Sequence

1. Launch validator: `tnfr-validate -i`
2. Select **[v]** for Validate
3. Enter your sequence: `emission reception coherence silence`
4. Review health metrics
5. If health is low, try optimization mode

### Workflow 2: Creating a New Sequence

1. Launch validator
2. Select **[g]** for Generate
3. Choose domain (e.g., therapeutic) and objective
4. Review generated sequence
5. If needed, optimize further

### Workflow 3: Learning TNFR

1. Launch validator
2. Select **[h]** for Help to see operators
3. Select **[e]** for Explore to learn patterns
4. Try validating simple sequences to see metrics
5. Experiment with generation to see good examples

## Understanding TNFR Operators

### Starting Operators (Begin a sequence)
- **emission**: Initiates a new resonant pattern
- **reception**: Begins by receiving information

### Stabilizing Operators (End or stabilize)
- **coherence**: Consolidates structure
- **silence**: Freezes evolution temporarily
- **self_organization**: Forms stable patterns

### Transformation Operators (Middle of sequence)
- **dissonance**: Introduces controlled instability
- **mutation**: Causes phase transitions
- **transition**: Moves between states

### Connection Operators (Link structures)
- **coupling**: Creates links between nodes
- **resonance**: Amplifies and propagates patterns
- **recursivity**: Enables nested operations

## Troubleshooting

### Invalid Sequence Errors

**Problem**: Sequence is marked as invalid.

**Common Causes**:
- Starting with a non-starter (e.g., `silence emission`)
- Not ending with a stabilizer
- Invalid operator names (typos)

**Solutions**:
1. Check operator spelling
2. Ensure sequence starts with `emission` or `reception`
3. End with `coherence`, `silence`, or `self_organization`

### Low Health Scores

**Problem**: Sequence validates but has low health.

**Common Causes**:
- Unbalanced stabilizers/destabilizers
- Missing regenerative elements
- Too short or too long

**Solutions**:
1. Use optimization mode to improve
2. Add `resonance` or `recursivity` for sustainability
3. Balance `dissonance` with stabilizers

### Can't Find Right Domain/Objective

**Problem**: Not sure which domain fits your use case.

**Solution**:
1. Use Explore mode to list all options
2. Try generating with multiple domains
3. Use pattern-based generation instead

## Tips for Best Results

1. **Start Simple**: Begin with 3-4 operators, then expand
2. **Balance Forces**: Mix stabilizers with transformers
3. **End Stable**: Always finish with a stabilizing operator
4. **Learn Patterns**: Study generated sequences to understand patterns
5. **Iterate**: Use optimization to refine sequences
6. **Check Health**: Aim for Overall Health > 0.7

## Example Use Cases

### Crisis Response Sequence

```
Domain: therapeutic
Objective: crisis_intervention
Result: emission → reception → coherence → resonance → silence
Health: 0.85 ✓
```

### Learning Activation

```
Domain: educational
Objective: knowledge_acquisition
Result: emission → reception → coupling → transition → coherence
Health: 0.78 ✓
```

### Organizational Change

```
Domain: organizational
Objective: change_implementation
Result: emission → dissonance → mutation → coherence → silence
Health: 0.72 ✓
```

## Getting Help

- **In-app Help**: Press **[h]** in the main menu
- **GitHub**: https://github.com/fermga/TNFR-Python-Engine
- **Documentation**: Run `tnfr-validate --help` for CLI options

## Appendix: Command-Line Reference

```bash
# Interactive mode (main feature)
tnfr-validate --interactive
tnfr-validate -i

# With deterministic seed
tnfr-validate -i --seed 42

# Graph validation (legacy mode)
tnfr-validate graph.graphml

# Show help
tnfr-validate --help
```

## Appendix: Health Score Interpretation

| Score | Status | Interpretation | Action |
|-------|--------|----------------|--------|
| 0.9-1.0 | Exceptional | Nearly perfect sequence | Ready to use |
| 0.8-0.89 | Excellent | High-quality sequence | Minor tweaks possible |
| 0.7-0.79 | Good | Solid, usable sequence | Consider optimization |
| 0.6-0.69 | Moderate | Functional but improvable | Optimization recommended |
| 0.5-0.59 | Fair | Needs improvement | Optimize or regenerate |
| Below 0.5 | Poor | Significant issues | Regenerate or major rework |

---

**Note**: This tool implements TNFR Grammar 2.0 with full structural validation and health analysis capabilities.
