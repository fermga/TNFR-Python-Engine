# TNFR Presets

This directory contains reproducible YAML preset templates for TNFR simulations.
Each preset demonstrates specific structural dynamics patterns using the canonical
TNFR operators.

## Available Presets

### resonant_bootstrap.yaml
Demonstrates resonant initialization and network stabilization. This is the
fundamental sequence showing how coherent structures emerge through the
application of core structural operators.

**Operators**: AL, EN, IL, RA, VAL, UM, SHA

### contained_mutation.yaml
Explores controlled phase transitions (mutations) within a structured framework.
Uses THOL (recursive blocks) to contain dissonant operators within a coherent
structure, demonstrating mutation control.

**Operators**: AL, EN, OZ, ZHIR, IL, RA, SHA

### coupling_exploration.yaml
Investigates network coupling dynamics with recursive navigation. Shows how
different coupling configurations can be explored through structural operators.

**Operators**: AL, EN, IL, VAL, UM, OZ, NAV, RA, SHA

### fractal_expand.yaml
Demonstrates operational fractality through recursive expansion. Shows how
structures can grow while maintaining their operational identity through
nested THOL blocks.

**Operators**: THOL, VAL, UM, NUL, RA

### fractal_contract.yaml
Demonstrates operational fractality through recursive contraction. Shows how
structures can consolidate while preserving their structural identity.

**Operators**: THOL, NUL, UM, SHA, RA

## YAML Format

Each preset file follows this structure:

```yaml
metadata:
  name: preset_name
  description: >
    Multi-line description of what this preset demonstrates
  operators:
    - LIST_OF_OPERATORS

topology:
  type: ring | complete | erdos
  nodes: 24
  seed: 1

dynamics:
  dt: 0.1
  integrator: euler | rk4
  steps: 100

sequence:
  - OPERATOR_NAME
  - WAIT: N
  - THOL:
      body:
        - OPERATOR_LIST
      repeat: N
      close: OPERATOR_NAME
```

## Using Presets

### From CLI
```bash
# Run a preset by name (loads from config/presets.py)
tnfr run --preset resonant_bootstrap

# Load and run a YAML preset file
tnfr run --sequence-file presets/resonant_bootstrap.yaml

# Override preset parameters
tnfr run --preset resonant_bootstrap --nodes 48 --steps 200
```

### From Python
```python
from tnfr.config.presets import get_preset
from tnfr.execution import play
import networkx as nx

# Get preset tokens
program = get_preset("resonant_bootstrap")

# Create and execute
G = nx.cycle_graph(24)
play(G, program)
```

## Structural Operators Reference

| Operator | Canonical Name | Function |
|----------|---------------|----------|
| AL | Emission | Initialize/emit structure |
| EN | Reception | Enable coupling receptivity |
| IL | Coupling | Establish connections |
| RA | Resonance | Propagate coherence |
| VAL | Validation | Check structural integrity |
| UM | Self-organization | Optimize internal structure |
| OZ | Dissonance | Introduce controlled instability |
| ZHIR | Mutation | Phase transition |
| NAV | Navigation/Transition | Explore state space |
| NUL | Expansion/Contraction | Scale transformation |
| SHA | Silence | Stabilize/freeze evolution |
| THOL | Recursivity | Nested structural block |

## Creating Custom Presets

1. Copy an existing preset as a template
2. Modify the `sequence` section with your operator sequence
3. Adjust `topology` and `dynamics` parameters as needed
4. Document the structural intent in `metadata.description`
5. List all operators used in `metadata.operators`
6. Test with: `tnfr run --sequence-file your_preset.yaml`

## TNFR Invariants

All presets must respect TNFR canonical invariants:

1. **EPI coherence**: Structure changes only via structural operators
2. **Operator closure**: All operators map to valid TNFR states
3. **Phase verification**: Couplings require explicit phase synchrony
4. **Operational fractality**: THOL blocks preserve functional identity
5. **Structural units**: Frequencies expressed in Hz_str

See `AGENTS.md` for complete canonical invariants.
