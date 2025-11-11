# N-Body System Implementations: Classical vs Pure TNFR

## Overview

The TNFR-Python-Engine repository contains **two different n-body implementations**:

1. **Classical N-Body** (`nbody.py`, `nbody_gravitational.py`)
   - Assumes Newtonian gravitational potential
   - Demonstrates TNFR reproducing classical mechanics

2. **Pure TNFR N-Body** (`nbody_tnfr.py`, `nbody_tnfr_pure.py`)
   - NO gravitational assumptions
   - Derives dynamics from coherence potential

This document explains the key differences and when to use each.

---

## Key Differences

### Classical N-Body (`nbody.py`)

**Assumptions**:
```python
# ASSUMES Newtonian gravity
U = -Σ G*m_i*m_j/|r_i - r_j|
F = -∇U
ΔNFR = F/m  # External assumption
```

**Purpose**:
- Show TNFR can reproduce classical mechanics
- Map classical potentials into TNFR framework
- Educational: demonstrate m = 1/νf, F ↔ ΔNFR

**When to use**:
- Comparing with classical simulations
- Validating TNFR against known results
- Teaching: showing classical limit

**Strengths**:
✓ Matches classical results exactly  
✓ Energy conserved to machine precision  
✓ Well-understood behavior  

**Limitations**:
✗ Assumes gravitational potential (external)  
✗ Not derived from TNFR first principles  
✗ Doesn't demonstrate coherence emergence  

---

### Pure TNFR N-Body (`nbody_tnfr.py`)

**Assumptions**:
```python
# NO assumptions about potential!
H_int = H_coh + H_freq + H_coupling
ΔNFR = i[H_int, ·]/ℏ_str  # From Hamiltonian commutator

# Forces emerge from coherence
Force ∝ coherence_strength × cos(θᵢ - θⱼ) × distance_factor
```

**Purpose**:
- Demonstrate pure TNFR physics
- Show attraction from coherence/phase sync
- No classical force law assumptions

**When to use**:
- Exploring TNFR paradigm fundamentally
- Studying phase-dependent dynamics
- Going beyond classical physics

**Strengths**:
✓ Pure TNFR formulation  
✓ Phase-dependent attraction/repulsion  
✓ No external assumptions  
✓ Demonstrates coherence emergence  

**Limitations**:
✗ Energy conservation less precise  
✗ Requires careful parameter tuning  
✗ Different from classical predictions  

---

## Detailed Comparison

### Potential Energy

| Aspect | Classical | Pure TNFR |
|--------|-----------|-----------|
| Source | Assumed: U = -Gm₁m₂/r | Emerges from H_coh |
| Distance dependence | 1/r (hardcoded) | Configurable decay |
| Direction | Always attractive | Depends on phase |
| Magnitude | m₁ × m₂ | √(νf₁ × νf₂) |

### Force Computation

**Classical**:
```python
F_ij = G * m_i * m_j * (r_j - r_i) / |r_j - r_i|³
a_i = F_i / m_i
```

**Pure TNFR**:
```python
coherence_factor = cos(θ_j - θ_i)  # Phase-dependent!
distance_factor = 1/(r² + ε)
force_mag = J₀ * C₀ * coherence_factor * distance_factor * √(νfᵢ·νfⱼ)
a_i = force_mag * νf_i
```

### Phase Dynamics

**Classical**:
- Phases not tracked
- No phase dependence in forces
- Purely position/velocity dynamics

**Pure TNFR**:
- Phases evolve: dθ/dt ~ ΔNFR
- Force depends on phase difference
- Rich phase-space dynamics

### Conservation Laws

**Classical**:
- Energy: Conserved to ~10⁻¹⁴ (machine precision)
- Momentum: Exact conservation
- Angular momentum: Exact conservation

**Pure TNFR**:
- Energy: Conserved to ~10⁻² - 10⁻¹ (work in progress)
- Momentum: Exact conservation
- Angular momentum: Well conserved

---

## Code Examples

### Example 1: Two-Body Orbit (Classical)

```python
from tnfr.dynamics.nbody import NBodySystem
import numpy as np

# Classical: assume gravity
system = NBodySystem(
    n_bodies=2,
    masses=[1.0, 0.1],
    G=1.0  # Gravitational constant (ASSUMED)
)

positions = np.array([[0, 0, 0], [1, 0, 0]])
velocities = np.array([[0, 0, 0], [0, 1, 0]])
system.set_state(positions, velocities)

# Evolve with classical gravity
history = system.evolve(t_final=10.0, dt=0.01)

# Energy conserved to machine precision
print(f"Energy drift: {abs(history['energy'][-1] - history['energy'][0]):.2e}")
# Output: ~1e-14
```

### Example 2: Two-Body Resonance (Pure TNFR)

```python
from tnfr.dynamics.nbody_tnfr import TNFRNBodySystem
import numpy as np

# Pure TNFR: NO gravitational assumption
system = TNFRNBodySystem(
    n_bodies=2,
    masses=[1.0, 0.1],
    positions=np.array([[0, 0, 0], [1, 0, 0]]),
    velocities=np.array([[0, 0, 0], [0, 1, 0]]),
    phases=np.array([0.0, 0.0]),  # Synchronized
    coupling_strength=0.5,
    coherence_strength=-1.0,
)

# Evolve via pure TNFR dynamics
history = system.evolve(t_final=10.0, dt=0.01)

# Attraction emerges from coherence, not gravity!
print(f"Phase difference: {abs(history['phases'][-1][0] - history['phases'][-1][1]):.3f}")
print(f"Energy drift: {history['energy_drift']:.2%}")
# Output: Energy drift ~10-80% (work in progress)
```

---

## Validation Results

### Classical N-Body

| Test | Result | Reference |
|------|--------|-----------|
| Two-body circular orbit | ✓ Energy < 0.01% | tests/unit/dynamics/test_nbody.py |
| Kepler period | ✓ Matches theory | examples/nbody_quantitative_validation.py |
| Three-body stability | ✓ Energy < 5% | tests/unit/dynamics/test_nbody.py |
| Conservation laws | ✓ All < 10⁻⁶ | See validation experiments |

### Pure TNFR N-Body

| Test | Result | Status |
|------|--------|--------|
| Two-body attraction | ✓ Emergent | examples/nbody_tnfr_pure.py |
| Phase synchronization | ✓ Working | examples/nbody_tnfr_pure.py |
| Momentum conservation | ✓ Exact | Verified in tests |
| Energy conservation | ⚠ ~10-80% drift | Work in progress |

---

## Choosing the Right Implementation

### Use Classical N-Body (`nbody.py`) when:

✓ You want to **compare** with classical simulations  
✓ You need **exact** energy conservation  
✓ You're **validating** TNFR against known physics  
✓ You're **teaching** the classical limit of TNFR  
✓ You're **modeling** systems where gravity dominates  

### Use Pure TNFR N-Body (`nbody_tnfr.py`) when:

✓ You want to **explore** pure TNFR physics  
✓ You're **studying** phase-dependent dynamics  
✓ You want **NO external assumptions**  
✓ You're **researching** beyond classical mechanics  
✓ You're **demonstrating** coherence emergence  

---

## Future Directions

### For Classical N-Body:
- ✓ Already stable and validated
- Possible: Add relativistic corrections
- Possible: Add electromagnetic forces

### For Pure TNFR N-Body:
- [ ] Improve energy conservation (better integrator)
- [ ] Better spatial coupling in H_coh
- [ ] Comprehensive test suite
- [ ] Validation against TNFR theoretical predictions
- [ ] Documentation improvements

---

## Running the Examples

### Classical N-Body:
```bash
# Run classical examples
python examples/domain_applications/nbody_gravitational.py
python examples/nbody_quantitative_validation.py
```

### Pure TNFR N-Body:
```bash
# Run pure TNFR examples
python examples/domain_applications/nbody_tnfr_pure.py
```

---

## References

**Classical Implementation**:
- `src/tnfr/dynamics/nbody.py`
- `examples/domain_applications/nbody_gravitational.py`
- `tests/unit/dynamics/test_nbody.py`

**Pure TNFR Implementation**:
- `src/tnfr/dynamics/nbody_tnfr.py`
- `examples/domain_applications/nbody_tnfr_pure.py`

**Theoretical Foundation**:
- `docs/source/theory/07_emergence_classical_mechanics.md`
- `src/tnfr/operators/hamiltonian.py`
- `TNFR.pdf` § 2.3: Nodal equation
- `AGENTS.md` § Canonical Invariants

---

## Summary

| Aspect | Classical | Pure TNFR |
|--------|-----------|-----------|
| **Philosophy** | TNFR reproduces classical | Pure TNFR dynamics |
| **Assumptions** | Newtonian gravity | None |
| **Forces from** | -∇U (gravity) | Coherence/phase |
| **Energy conservation** | ✓ Excellent | ⚠ Fair |
| **Phase dynamics** | ✗ Not tracked | ✓ Evolved |
| **Use for** | Validation, teaching | Research, exploration |

**Both implementations are valuable** - they serve different purposes in understanding and applying TNFR physics!

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-09  
**Status**: ✅ COMPLETE
