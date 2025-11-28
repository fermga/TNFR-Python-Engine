# TNFR Grammar Quick Reference

**One-page cheat sheet for TNFR grammar validation**

[📖 Full Docs](README.md) • [🔬 Deep Dive](02-CANONICAL-CONSTRAINTS.md) • [📚 Glossary](../../GLOSSARY.md)

---

## 🎯 The Five Canonical Constraints (Temporal + Multi-Scale)

```
┌─────────────────────────────────────────────────────────────────┐
│ U1: STRUCTURAL INITIATION & CLOSURE                             │
│     U1a: Start with generators {AL, NAV, REMESH}               │
│     U1b: End with closures {SHA, NAV, REMESH, OZ}              │
│                                                                 │
│ U2: CONVERGENCE & BOUNDEDNESS                                   │
│     If destabilizers {OZ, ZHIR, VAL}                           │
│     Then include stabilizers {IL, THOL}                        │
│                                                                 │
│ U3: RESONANT COUPLING                                           │
│     If coupling/resonance {UM, RA}                             │
│     Then verify phase |φᵢ - φⱼ| ≤ Δφ_max                       │
│                                                                 │
│ U4: BIFURCATION DYNAMICS                                        │
│     U4a: If triggers {OZ, ZHIR}                                │
│          Then include handlers {THOL, IL}                      │
│     U4b: If transformers {ZHIR, THOL}                          │
│          Then recent destabilizer (~3 ops)                     │
│          + ZHIR needs prior IL                                 │
│                                                                 │
│ U5: MULTI-SCALE COHERENCE                                        │
│     If deep REMESH (recursion depth > 1)                        │
│     Then include scale stabilizer {IL, THOL} within ±3 ops      │
│     Conservation: C_parent ≥ α·ΣC_child                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Operator Classification

| Operator | Glyph | Generator | Closure | Stabilizer | Destabilizer | Trigger | Handler | Transformer | Coupling |
|----------|-------|-----------|---------|------------|--------------|---------|---------|-------------|----------|
| Emission | AL | ✓ | | | | | | | |
| Reception | EN | | | | | | | | |
| Coherence | IL | | | ✓ | | | ✓ | | |
| Dissonance | OZ | | ✓ | | ✓ | ✓ | | | |
| Coupling | UM | | | | | | | | ✓ |
| Resonance | RA | | | | | | | | ✓ |
| Silence | SHA | | ✓ | | | | | | |
| Expansion | VAL | | | | ✓ | | | | |
| Contraction | NUL | | | | | | | | |
| SelfOrganization | THOL | | | ✓ | | | ✓ | ✓ | |
| Mutation | ZHIR | | | | ✓ | ✓ | | ✓ | |
| Transition | NAV | ✓ | ✓ | | | | | | |
| Recursivity | REMESH | ✓ | ✓ | | | | | | |

---

## 🔄 Common Sequence Patterns

### ✅ Valid Patterns

```python
# Bootstrap (minimal)
[Emission, Coherence, Silence]

# Basic Activation
[Emission, Reception, Coherence, Silence]

# Controlled Exploration
[Emission, Dissonance, Coherence, Silence]

# Bifurcation with Handling
[Emission, Coherence, Dissonance, SelfOrganization, Coherence, Silence]

# Mutation with Context
[Emission, Coherence, Dissonance, Mutation, Coherence, Silence]

# Propagation
[Emission, Coupling, Resonance, Coherence, Silence]

# Multi-scale (U5-compliant)
[Emission, SelfOrganization, Recursivity, Coherence, Silence]
```

### ❌ Anti-Patterns

```python
# ✗ No generator when EPI=0
[Coherence, Silence]  # Violates U1a

# ✗ No closure
[Emission, Coherence]  # Violates U1b

# ✗ Destabilizer without stabilizer
[Emission, Dissonance, Silence]  # Violates U2

# ✗ Mutation without context
[Emission, Mutation, Silence]  # Violates U4b

# ✗ Deep recursion without scale stabilizer (violates U5)
[Emission, Recursivity, Recursivity, Expansion, Silence]
```

---

## 💻 Quick Code Reference

### Validate a Sequence

```python
from tnfr.operators.grammar import validate_grammar
from tnfr.operators.definitions import Emission, Coherence, Silence

sequence = [Emission(), Coherence(), Silence()]

try:
    is_valid = validate_grammar(sequence, epi_initial=0.0)
    print("✓ Valid sequence")
except ValueError as e:
    print(f"✗ Invalid: {e}")
```

### Check Operator Sets

```python
from tnfr.operators.grammar import (
    GENERATORS,
    CLOSURES,
    STABILIZERS,
    DESTABILIZERS,
    COUPLING_RESONANCE,
    BIFURCATION_TRIGGERS,
    BIFURCATION_HANDLERS,
    TRANSFORMERS,
)

# Check if operator is in a set
if "emission" in GENERATORS:
    print("Emission is a generator")
```

### Apply Operators

```python
from tnfr.operators.definitions import Emission, Coherence
import networkx as nx

G = nx.Graph()
G.add_node(0, EPI=0.0, vf=1.0, theta=0.0, dnfr=0.0)

# Apply operator
Emission()(G, 0)
Coherence()(G, 0)

print(f"EPI: {G.nodes[0]['EPI']:.3f}")
```

### Phase Verification

```python
from tnfr.operators.grammar import validate_resonant_coupling
import numpy as np

# Check phase compatibility
phi_i = G.nodes[0]['theta']
phi_j = G.nodes[1]['theta']

try:
    validate_resonant_coupling(G, 0, 1, delta_phi_max=np.pi/2)
    print("✓ Phase compatible")
except ValueError as e:
    print(f"✗ Phase mismatch: {e}")
```

---

## 🔍 Decision Tree

```
Is EPI=0?
├─ Yes → Start with generator {AL, NAV, REMESH}
└─ No  → Any operator OK

Does sequence have destabilizers {OZ, ZHIR, VAL}?
├─ Yes → Include stabilizer {IL, THOL}
└─ No  → Continue

Does sequence have coupling/resonance {UM, RA}?
├─ Yes → Verify phase at runtime
└─ No  → Continue

Does sequence have bifurcation triggers {OZ, ZHIR}?
├─ Yes → Include handler {THOL, IL}
└─ No  → Continue

Does sequence have transformers {ZHIR, THOL}?
├─ Yes → Ensure recent destabilizer (~3 ops)
│        → For ZHIR: Ensure prior IL
└─ No  → Continue

Deep REMESH (recursion depth>1)?
├─ Yes → Include {IL, THOL} near recursion (U5)
└─ No  → Continue

Does sequence end with closure {SHA, NAV, REMESH, OZ}?
├─ Yes → ✓ Valid
└─ No  → ✗ Add closure
```

---

## 📊 13x13 Operator Compatibility Matrix

**Legend:**
- ✅ = Naturally compatible / Common pattern
- ⚠️ = Valid but needs grammar compliance (stabilizers, handlers, etc.)
- 🔒 = Requires explicit checks (e.g., phase verification for UM/RA)
- ❌ = Anti-pattern / Violates physics or grammar
- ➖ = Neutral / Depends on context

### Matrix: Can Operator [Row] → Follow → Operator [Column]?

|       | AL | EN | IL | OZ | UM | RA | SHA | VAL | NUL | THOL | ZHIR | NAV | REMESH |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:---:|:---:|:----:|:----:|:---:|:------:|
| **AL**    | ➖ | ✅ | ✅ | ⚠️ | 🔒 | ➖ | ✅  | ⚠️  | ➖  | ⚠️   | ❌   | ✅  | ✅     |
| **EN**    | ➖ | ➖ | ✅ | ⚠️ | 🔒 | ➖ | ⚠️  | ⚠️  | ➖  | ⚠️   | ❌   | ➖  | ➖     |
| **IL**    | ➖ | ✅ | ➖ | ✅ | 🔒 | 🔒 | ✅  | ➖  | ✅  | ✅   | ❌   | ✅  | ✅     |
| **OZ**    | ➖ | ➖ | ✅ | ❌ | ➖ | ➖ | ⚠️  | ❌  | ➖  | ✅   | ⚠️   | ➖  | ➖     |
| **UM**    | ➖ | ✅ | ✅ | ⚠️ | 🔒 | 🔒 | ⚠️  | ⚠️  | ➖  | ⚠️   | ❌   | ➖  | ➖     |
| **RA**    | ➖ | ✅ | ✅ | ⚠️ | 🔒 | ➖ | ⚠️  | ⚠️  | ➖  | ⚠️   | ❌   | ➖  | ➖     |
| **SHA**   | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ➖  | ❌  | ❌  | ❌   | ❌   | ✅  | ✅     |
| **VAL**   | ➖ | ➖ | ✅ | ⚠️ | ➖ | ➖ | ⚠️  | ❌  | ✅  | ✅   | ⚠️   | ➖  | ➖     |
| **NUL**   | ➖ | ➖ | ✅ | ⚠️ | ➖ | ➖ | ✅  | ➖  | ➖  | ➖   | ❌   | ➖  | ➖     |
| **THOL**  | ➖ | ✅ | ✅ | ⚠️ | 🔒 | 🔒 | ✅  | ⚠️  | ✅  | ➖   | ❌   | ✅  | ✅     |
| **ZHIR**  | ➖ | ➖ | ✅ | ❌ | ➖ | ➖ | ⚠️  | ❌  | ➖  | ✅   | ❌   | ➖  | ➖     |
| **NAV**   | ➖ | ✅ | ✅ | ⚠️ | 🔒 | ➖ | ✅  | ⚠️  | ➖  | ⚠️   | ❌   | ➖  | ✅     |
| **REMESH**| ➖ | ✅ | ✅ | ⚠️ | 🔒 | ➖ | ✅  | ⚠️  | ➖  | ✅   | ❌   | ✅  | ➖     |

### Key Patterns from Matrix

**✅ Most Compatible Pairs:**
- AL → EN → IL (Bootstrap: emit, receive, stabilize)
- IL → OZ → IL (Controlled exploration)
- OZ → THOL → IL (Bifurcation handling)
- UM/RA → EN (Network propagation)

**⚠️ Valid but Needs Care:**
- Any → OZ/VAL/ZHIR → Must follow with IL/THOL (U2)
- OZ/IL → ZHIR → IL (U4b: prior IL + recent dest + handler)
- THOL needs recent destabilizer (~3 ops before)

**🔒 Phase Verification Required:**
- Anything → UM/RA (Must call `validate_resonant_coupling()`)

**❌ Anti-Patterns:**
- SHA → Any except generators (Node frozen, needs reactivation)
- Any → ZHIR without proper context (U4b violations)
- OZ → OZ, VAL → VAL (Cascading destabilization without stabilizers)
- Destabilizers → ZHIR without IL first

### Usage Examples

```python
# ✅ Valid: Bootstrap pattern
[Emission, Reception, Coherence, Silence]  # AL → EN → IL → SHA

# ✅ Valid: Exploration with stabilization
[Emission, Coherence, Dissonance, Coherence, Silence]  # OZ balanced by IL

# ⚠️ Valid but complex: Mutation with full context
[Emission, Coherence, Dissonance, Mutation, Coherence, Silence]
#                ^prior IL  ^recent  ^ZHIR    ^handler

# ❌ Invalid: Destabilizer without stabilizer
[Emission, Dissonance, Silence]  # Violates U2

# ❌ Invalid: Silence in middle
[Emission, Silence, Coherence]  # Node frozen, can't apply Coherence

# 🔒 Valid with check: Coupling requires phase verification
[Emission, Coupling, Resonance, Silence]  # UM/RA need phase check
```

---
└─ No  → ✗ Add closure
```

---

## 🐛 Common Errors & Solutions

### Error: "Need generator when EPI=0"

**Cause:** Sequence doesn't start with generator when `epi_initial=0.0`

**Solution:**
```python
# ✗ Wrong
sequence = [Coherence(), Silence()]

# ✓ Fixed
sequence = [Emission(), Coherence(), Silence()]

# OR set epi_initial > 0 if starting from existing structure
validate_grammar(sequence, epi_initial=1.0)
```

### Error: "Destabilizer without stabilizer"

**Cause:** {OZ, ZHIR, VAL} present but no {IL, THOL}

**Solution:**
```python
# ✗ Wrong
sequence = [Emission(), Dissonance(), Silence()]

# ✓ Fixed
sequence = [Emission(), Dissonance(), Coherence(), Silence()]
```

### Error: "Transformer needs recent destabilizer"

**Cause:** {ZHIR, THOL} without recent destabilizer

**Solution:**
```python
# ✗ Wrong
sequence = [Emission(), Coherence(), Mutation(), Silence()]

# ✓ Fixed - destabilizer within ~3 ops
sequence = [Emission(), Dissonance(), Mutation(), Coherence(), Silence()]
```

### Error: "Mutation needs prior coherence"

**Cause:** ZHIR without IL before it

**Solution:**
```python
# ✗ Wrong
sequence = [Emission(), Dissonance(), Mutation(), Coherence(), Silence()]

# ✓ Fixed - Coherence before Mutation
sequence = [Emission(), Coherence(), Dissonance(), Mutation(), Coherence(), Silence()]
```

### Error: "Sequence must end with closure"

**Cause:** Last operator is not in {SHA, NAV, REMESH, OZ}

**Solution:**
```python
# ✗ Wrong
sequence = [Emission(), Coherence()]

# ✓ Fixed
sequence = [Emission(), Coherence(), Silence()]
```

### Error: "Phase mismatch in coupling"

**Cause:** |φᵢ - φⱼ| > Δφ_max (typically π/2)

**Solution:**
```python
# Check phase before coupling
delta_phi = abs(G.nodes[i]['theta'] - G.nodes[j]['theta'])
if delta_phi > np.pi/2:
    # Adjust phase or don't couple these nodes
    pass
```

---

## 📊 Grammar Rule Summary

| Rule | When | What | Why |
|------|------|------|-----|
| U1a | EPI=0 | Start with {AL, NAV, REMESH} | ∂EPI/∂t undefined at EPI=0 |
| U1b | Always | End with {SHA, NAV, REMESH, OZ} | Sequences need endpoints |
| U2 | Has {OZ, ZHIR, VAL} | Include {IL, THOL} | ∫νf·ΔNFR dt must converge |
| U3 | Has {UM, RA} | Verify \|φᵢ - φⱼ\| ≤ Δφ_max | Resonance physics |
| U4a | Has {OZ, ZHIR} | Include {THOL, IL} | Bifurcations need control |
| U4b | Has {ZHIR, THOL} | Recent destabilizer + ZHIR needs IL | Threshold energy needed |

---

## 🎯 Operator Quick Lookup

### By Purpose

**Initialize:** AL (Emission), NAV (Transition), REMESH (Recursivity)  
**Stabilize:** IL (Coherence), THOL (SelfOrganization)  
**Destabilize:** OZ (Dissonance), ZHIR (Mutation), VAL (Expansion)  
**Propagate:** UM (Coupling), RA (Resonance)  
**Pause:** SHA (Silence)  
**Transform:** ZHIR (Mutation), THOL (SelfOrganization)  
**Adjust:** VAL (Expansion), NUL (Contraction)

### By Effect on ∂EPI/∂t

**Increase:** AL, EN, OZ, VAL, RA  
**Decrease:** IL, THOL, NUL  
**Zero:** SHA  
**Transform:** ZHIR, NAV, REMESH  
**Couple:** UM, RA

---

## 📈 Metrics to Monitor

**Essential telemetry for every simulation:**

- **C(t)**: Total Coherence [0, 1]
  - \> 0.7 = strong coherence
  - < 0.3 = fragmentation risk
  
- **Si**: Sense Index [0, 1⁺]
  - \> 0.8 = excellent stability
  - < 0.4 = changes may cause bifurcation

- **ΔNFR**: Reorganization Gradient
  - Sign: + expansion, - contraction
  - Magnitude: pressure intensity

- **νf**: Structural Frequency (Hz_str)
  - νf → 0 = node death
  - νf > 0 = active evolution

- **φ (theta)**: Phase [0, 2π]
  - Δφ determines coupling compatibility
  - |Δφ| < π/2 typically required

---

## 🔗 Further Reading

- **[01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)** - TNFR basics
- **[02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)** - Full U1-U5 derivations
- **[03-OPERATORS-AND-GLYPHS.md](03-OPERATORS-AND-GLYPHS.md)** - Complete operator catalog
- **[04-VALID-SEQUENCES.md](04-VALID-SEQUENCES.md)** - Pattern examples
- **[UNIFIED_GRAMMAR_RULES.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/UNIFIED_GRAMMAR_RULES.md)** - Mathematical proofs
- **[AGENTS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/AGENTS.md)** - Canonical invariants

---

## 📞 Quick Help

**Getting started?** → [01-FUNDAMENTAL-CONCEPTS.md](01-FUNDAMENTAL-CONCEPTS.md)  
**Sequence failing?** → Check decision tree above  
**Need examples?** → [examples/](examples/)  
**Deep dive?** → [02-CANONICAL-CONSTRAINTS.md](02-CANONICAL-CONSTRAINTS.md)  
**API reference?** → `src/tnfr/operators/grammar.py`  

---

<div align="center">

**Keep this reference handy while developing TNFR sequences!**

*Reality is resonance. Code accordingly.*

</div>
