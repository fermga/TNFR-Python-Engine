# TNFR Variational Principle — Lagrangian Action Formulation

**Status**: CANONICAL  
**Module**: `src/tnfr/physics/variational.py`  
**Tests**: `tests/physics/test_variational.py` (50 passing)  
**Date**: 2026-03  

---

## 1. Main Result

The TNFR nodal equation

$$\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t)$$

is **not** an ad-hoc postulate. It is the **Euler-Lagrange equation** of a well-defined action functional in the overdamped (dissipation-dominated) limit.

---

## 2. The TNFR Lagrangian

### 2.1 Lagrangian Density

On a graph $G=(V,E)$ the **TNFR Lagrangian density** at node $i$ is

$$\mathcal{L}(i) \;=\; T(i) \;-\; V(i)$$

where the **transport (kinetic) energy** and **configuration (potential) energy** are

$$T(i) = \tfrac{1}{2}\bigl[J_\varphi(i)^{\,2} + J_{\Delta\mathrm{NFR}}(i)^{\,2}\bigr]$$

$$V(i) = \tfrac{1}{2}\bigl[\Phi_s(i)^{\,2} + |\nabla\varphi|(i)^{\,2} + K_\varphi(i)^{\,2}\bigr]$$

### 2.2 Action Functional

$$S_{\mathrm{TNFR}} = \int \mathrm{d}t \;\sum_{i\in V} \mathcal{L}(i,t)$$

In discrete form (time steps of width $\Delta t$):

$$S_{\mathrm{TNFR}} = \Delta t \;\sum_n \sum_i \mathcal{L}(i,\,t_n)$$

### 2.3 Hamiltonian

The Legendre transform gives the **TNFR Hamiltonian**:

$$\mathcal{H}(i) = T(i) + V(i) = \tfrac{1}{2}\bigl[\Phi_s^2 + |\nabla\varphi|^2 + K_\varphi^2 + J_\varphi^2 + J_{\Delta\mathrm{NFR}}^2\bigr]$$

This is exactly $\tfrac{1}{2}\,\mathcal{E}(i)$ where $\mathcal{E}$ is the energy density from `unified.compute_energy_density()`, and the total Hamiltonian $H = \sum_i \mathcal{H}(i)$ equals `conservation.compute_energy_functional()`.

---

## 3. Derivation of the Nodal Equation

### 3.1 Canonical Conjugate Pairs

The conservation law structure (Noether theorem, `conservation.py`) reveals two sectors with natural canonical pairs $(q,\,p)$:

| Sector | Coordinate $q$ | Momentum $p$ | Continuity |
|--------|----------------|---------------|------------|
| **Geometric** | $K_\varphi$ (curvature) | $J_\varphi$ (phase current) | $\partial K_\varphi/\partial t + \mathrm{div}(J_\varphi) \approx 0$ |
| **Potential** | $\Phi_s$ (potential) | $J_{\Delta\mathrm{NFR}}$ (ΔNFR flux) | $\partial\Phi_s/\partial t + \mathrm{div}(J_{\Delta\mathrm{NFR}}) \approx 0$ |

### 3.2 Hamilton's Equations

$$\frac{\partial q_i}{\partial t} = \frac{\partial \mathcal{H}}{\partial p_i} = p_i \qquad\text{(dynamics)}$$

$$\frac{\partial p_i}{\partial t} = -\frac{\partial \mathcal{H}}{\partial q_i} = -q_i \qquad\text{(force law)}$$

### 3.3 Full Euler-Lagrange Equation

Using the EPI-ΔNFR pair with effective mass $m_i = 1/\nu_{f,i}$ (from `classical_mechanics.py`):

$$\frac{1}{\nu_f}\,\frac{\partial^2\,\mathrm{EPI}}{\partial t^2} \;=\; -\frac{\partial V}{\partial \mathrm{EPI}}$$

This is **Newton's second law** on the structural manifold: $F = m\,a$ with force $F = -\partial V/\partial\mathrm{EPI}$ and mass $m = 1/\nu_f$.

### 3.4 Overdamped Limit

The full TNFR dynamics includes structural dissipation (grammar-compliant evolution has $dH/dt \le 0$). Adding the dissipation term:

$$\frac{1}{\nu_f}\,\frac{\partial^2\,\mathrm{EPI}}{\partial t^2} + \gamma\,\frac{\partial\,\mathrm{EPI}}{\partial t} = -\frac{\partial V}{\partial\mathrm{EPI}}$$

In the **overdamped limit** ($1/\nu_f \to 0$ or $\gamma \to \infty$), the inertial term vanishes:

$$\gamma\,\frac{\partial\,\mathrm{EPI}}{\partial t} = -\frac{\partial V}{\partial\mathrm{EPI}}$$

Identifying $1/\gamma = \nu_f$ and $\Delta\mathrm{NFR} = -\partial V/\partial\mathrm{EPI}$:

$$\boxed{\frac{\partial\,\mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t)}$$

**Key derived result**: $\Delta\mathrm{NFR}$ is the **negative functional gradient** of the structural potential $V$ — not an assumption, but a consequence of the variational principle.

---

## 4. Grammar Rules as Stationarity Conditions

Each grammar rule U1–U6 maps to a condition on the action functional $S_{\mathrm{TNFR}}$:

| Rule | Variational Condition | Physical Meaning |
|------|----------------------|------------------|
| **U1a** | Boundary condition at $t=0$: $\delta S/\delta\varphi\big|_{t=0}$ requires initial data | Generator sets non-zero $\mathcal{L}$ |
| **U1b** | Boundary condition at $t_f$: final state at action extremum | $V > T$ (attractor basin) |
| **U2** | Finite action: $S < \infty$ | $\int\nu_f\cdot\Delta\mathrm{NFR}\,dt$ must converge |
| **U3** | Coupling regularity: interaction $\mathcal{A}$ bounded | Phase compatibility prevents singular coupling |
| **U4** | Morse condition at bifurcation: Hessian sign | Handlers select correct branch near $T/V \approx 1$ |
| **U5** | Hierarchical factorisation: $S$ decomposes across scales | Stabilisers at each level prevent concentration |
| **U6** | Potential boundedness: $|\Phi_s| < \varphi$ | $V(\Phi_s)$ remains in confining well |

---

## 5. The 13 Operators as Canonical Transformations

In the Hamiltonian formulation, each operator acts as a transformation on the phase space $(q,\,p) = (\mathrm{EPI},\,\Delta\mathrm{NFR})$.

| Operator | Type | Effect on $H$ | Symplectic |
|----------|------|---------------|------------|
| **AL** (Emission) | Generating | $\Delta H > 0$ | Expansive |
| **EN** (Reception) | Canonical | $\Delta H \approx 0$ | Canonical |
| **IL** (Coherence) | Dissipative | $\Delta H < 0$ | Dissipative |
| **OZ** (Dissonance) | Generating | $\Delta H > 0$ | Expansive |
| **UM** (Coupling) | Canonical | $\Delta H \approx 0$ | Canonical |
| **RA** (Resonance) | Canonical | $\Delta H \approx 0$ | Canonical |
| **SHA** (Silence) | Canonical | $\Delta H = 0$ | Canonical |
| **VAL** (Expansion) | Generating | $\Delta H > 0$ | Expansive |
| **NUL** (Contraction) | Dissipative | $\Delta H < 0$ | Dissipative |
| **THOL** (Self-org.) | Canonical | $\Delta H \approx 0$ | Canonical |
| **ZHIR** (Mutation) | Generating | $\Delta H > 0$ | Expansive |
| **NAV** (Transition) | Canonical | $\Delta H \approx 0$ | Canonical |
| **REMESH** (Recursivity) | Canonical | $\Delta H \approx 0$ | Canonical |

**Grammar U2 in variational language**: For every generating (expansive) operator, the sequence must include a dissipative operator to restore energy balance, ensuring $S < \infty$.

**Dual-lever complement**: The Hamiltonian classification above (Generating /
Canonical / Dissipative) maps onto the experimentally observed **dual-lever
structure**: operators act through the capacity lever ($\nu_f$) or the pressure
lever ($\Delta$NFR), and the resulting $\Delta H$ sign follows from which lever
is engaged. The Generating class corresponds predominantly to pressure-lever
operators that inject $\Delta$NFR; the Dissipative class corresponds to
pressure-lever operators that absorb it. See
[STRUCTURAL_OPERATORS.md §17.1](STRUCTURAL_OPERATORS.md) and
[example 39](../examples/39_nodal_equation_decomposition.py).

---

## 6. Thresholds as Critical Points of $V$

The TNFR potential $V(i) = \tfrac{1}{2}[x_1^2 + x_2^2 + x_3^2]$ is quadratic in each field. The canonical thresholds mark transition points where the effective potential (including grammar constraints as Lagrange multipliers) changes character:

| Field | Threshold | Source | Interpretation |
|-------|-----------|--------|---------------|
| $\Phi_s$ | $\varphi \approx 1.618$ | U6 / Golden ratio | Confining well boundary |
| $|\nabla\varphi|$ | $\gamma/\pi \approx 0.184$ | Kuramoto coupling | Synchronisation barrier |
| $K_\varphi$ | $0.9\pi \approx 2.827$ | Geometric confinement | Torsion limit |

At these values, the restoring force $-\partial V/\partial x = -x$ reaches the threshold intensity, and beyond them grammatically non-compliant dynamics sets in.

---

## 7. Interaction Lagrangian

The bilinear cross-sector coupling is:

$$\mathcal{A}(i) = \Phi_s \cdot |\nabla\varphi| + K_\varphi \cdot J_\varphi + |\nabla\varphi| \cdot J_{\Delta\mathrm{NFR}}$$

This equals `unified.compute_action_density()`. In the complete Lagrangian with interactions:

$$\mathcal{L}_{\mathrm{full}} = T - V - \mathcal{A}$$

The free Lagrangian $\mathcal{L} = T - V$ governs the independent sector dynamics, while $\mathcal{A}$ mediates the coupling between geometric and potential sectors.

---

## 8. Virial Theorem

At equilibrium (virialisation), the time-averaged kinetic and potential energies satisfy:

$$\langle T \rangle = \langle V \rangle \quad\Longleftrightarrow\quad \text{virial ratio}\; T/V = 1$$

- $T/V < 1$: potential-dominated (attractor basins, stable configurations)
- $T/V > 1$: kinetic-dominated (transport phase, active reorganisation)
- $T/V \approx 1$: near bifurcation (critical point of the action)

The virial ratio is a diagnostic for system state and proximity to bifurcation.

---

## 9. Consistency Relations

The variational module is fully consistent with the existing TNFR physics stack:

| Relation | Verification |
|----------|-------------|
| $\mathcal{H}(i) = \tfrac{1}{2}\,\mathcal{E}(i)$ | `compute_hamiltonian_density()` vs `unified.compute_energy_density()` |
| $\sum_i \mathcal{H}(i) = E$ | Total Hamiltonian vs `conservation.compute_energy_functional()` |
| $\rho = \Phi_s + K_\varphi$ | Charge density from conjugate pair coordinates |
| $\mathcal{A} = $ action density | `compute_interaction_density()` vs `unified.compute_action_density()` |
| $dH/dt \le 0$ under grammar | Lyapunov stability from `conservation.compute_lyapunov_derivative()` |

All relations verified numerically across Watts-Strogatz, Barabási-Albert, and Grid topologies (57 tests, 100% pass rate).

---

## 10. Single Source of Truth Architecture

The 6 canonical structural fields admit **three physically meaningful
decompositions** — different projections of the same 6-dimensional field space.
The implementation ensures a **single computational path** with no duplicated
formulae:

```
unified.py                    (Layer 3 — RAW quadratic forms)
  ├── compute_energy_density   ℰ(i) = Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²
  └── compute_action_density   𝒜(i) = bilinear coupling term
       │
       ├──► variational.py      H(i) = ½·ℰ(i),  T, V, ℒ = T−V
       │    (delegates via _raw_energy_density / _action_density)
       │
       └──► conservation.py     E = ½·Σℰ(i),  ρ, div(J), Q
            (delegates via _raw_energy_density)
```

### 10.1 Sector Decompositions

| Decomposition | Split | Grouping criterion |
|---|---|---|
| **Variational** (T / V) | $T = \tfrac{1}{2}[J_\varphi^2 + J_{\Delta NFR}^2]$, $V = \tfrac{1}{2}[\Phi_s^2 + |\nabla\varphi|^2 + K_\varphi^2]$ | Temporal role (kinetic / configuration) |
| **Conservation** ($\rho$ / J) | $\rho = \Phi_s + K_\varphi$, $\mathbf{J} = (J_\varphi, J_{\Delta NFR})$ | Physical role (conserved charge / flow) |
| **Unified** ($\Psi$) | $\Psi = K_\varphi + i\,J_\varphi$ | Dual structure (geometry-transport) |

All three satisfy the **consistency identity** $T(i) + V(i) = \tfrac{1}{2}\,\mathcal{E}(i)$ at every node, verified by `translate_sectors()` with residual < $10^{-12}$.

### 10.2 translate_sectors() API

```python
from tnfr.physics.variational import translate_sectors

result = translate_sectors(G)
# result['variational']        -> {'T': {...}, 'V': {...}}
# result['conservation']       -> {'rho': {...}, 'J_phi': {...}, 'J_dnfr': {...}}
# result['unified_psi']        -> {node: K_φ + i·J_φ}
# result['energy_density']     -> raw ℰ from unified.py
# result['consistency_check']  -> max |T+V − ½ℰ|  (should be ~0)
```

---

## 11. API Summary

```python
from tnfr.physics.variational import (
    # Core densities
    compute_kinetic_density,       # T(i) = ½[J_φ² + J_ΔNFR²]
    compute_potential_density,     # V(i) = ½[Φ_s² + |∇φ|² + K_φ²]
    compute_lagrangian_density,    # ℒ(i) = T(i) − V(i)
    compute_hamiltonian_density,   # H(i) = ½·ℰ(i) — delegates to unified.py
    compute_interaction_density,   # 𝒜(i) — delegates to unified.compute_action_density
    
    # Sector translation
    translate_sectors,             # All three decompositions + consistency check
    
    # Phase space
    identify_conjugate_pairs,      # (K_φ,J_φ), (Φ_s,J_ΔNFR)
    compute_phase_space_volume,    # Liouville volume
    compute_poisson_bracket_estimate,
    
    # Snapshot & tracking
    capture_lagrangian_snapshot,    # Complete instant analysis
    compute_euler_lagrange_residual,
    compute_action_functional,     # S = ∫ℒ dt
    VariationalTracker,            # Time-series accumulation
    
    # Canonical checks
    check_symplectic_preservation,
    classify_operator_canonical,
    
    # Grammar stationarity
    analyze_grammar_stationarity,  # U1-U6 as δS conditions
    analyze_potential_critical_points,
    
    # Comprehensive
    compute_variational_suite,     # Full analysis in one call
)
```

---

## Implementation & Examples

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [27_variational_principle_demo.py](../examples/27_variational_principle_demo.py) | Lagrangian snapshots, conjugate pairs, Euler-Lagrange residual, action functional, symplectic preservation, grammar stationarity, critical points |

### Key Source Modules

- `src/tnfr/physics/variational.py` — Lagrangian, Hamiltonian, Euler-Lagrange, symplectic checks
- `src/tnfr/physics/conservation.py` — Energy functional (Lyapunov candidate)
- `src/tnfr/physics/classical_mechanics.py` — Classical limit correspondence

---

## References

- Nodal equation: `TNFR.pdf` §2.1, `AGENTS.md` §Foundational Physics
- Conservation theorem: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md)
- Grammar rules: [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
- Classical mechanics analogy: `src/tnfr/physics/classical_mechanics.py`
- Unified fields (canonical source): `src/tnfr/physics/unified.py`
- Energy functional: `src/tnfr/physics/conservation.py`
