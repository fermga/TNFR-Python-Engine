# Physical Regime Correspondences

**Status**: Technical reference
**Version**: 0.0.3.2
**Date**: March 2026

---

## 1. Scope

This document derives five physical regimes as limiting cases of the nodal equation $\partial\mathrm{EPI}/\partial t = \nu_f \, \Delta\mathrm{NFR}(t)$. Each regime is specified by measurable conditions on the structural field tetrad ($\Phi_s$, $|\nabla\phi|$, $K_\phi$, $\xi_C$), the observable mappings between TNFR quantities and classical/quantum variables, and the validation artifacts linking theory to reproducible simulations.

### Verification Status

| Regime | Implementation | External reference | Test coverage | Status |
|--------|---------------|-------------------|---------------|---------|
| Classical mechanics | `classical_mechanics.py` | Kepler orbit (analytical) | `test_classical_mechanics.py` | **Verified** |
| Inertial | `classical_mechanics.py` | Two-train analytical | Embedded in example | **Verified** |
| Quantum | `quantum_mechanics.py` | Particle-in-box ($E_n = (\pi n/L)^2$) | None | Partial |
| Uncertainty/interference | Example only | Fourier bound | None | Partial |
| Thermodynamics | Example only | Newton cooling law | None | Demonstration |


---

## 2. Classical Mechanics (Low-Dissonance Limit)

### 2.1 Regime Conditions

$$
|\nabla\phi| \to 0, \qquad \nu_f = \mathrm{const}, \qquad C(t) \approx 1
$$

Under these constraints the nodal equation reduces to Newton's second law.

### 2.2 Observable Mapping

| Classical quantity | Symbol | TNFR quantity | Access |
|-------------------|--------|---------------|--------|
| Position | $q$ | Spatial component of EPI | `ClassicalMechanicsMapper.position` |
| Velocity | $\dot{q}$ | Flow component | Same accessor |
| Mass | $m$ | $1/\nu_f$ | Telemetry |
| Force | $F$ | $\Delta\mathrm{NFR}$ | Structural units |
| Potential | $V$ | $\Phi_s$ | `compute_structural_potential()` |
| Action | $S$ | Phase accumulation | Diagnostics |

### 2.3 Derivation

Starting from the nodal equation:

$$
\frac{\partial\mathrm{EPI}}{\partial t} = \nu_f \, \Delta\mathrm{NFR}
\quad \Rightarrow \quad
\frac{dv}{dt} = \nu_f \, \Delta\mathrm{NFR}_{\mathrm{force}}
$$

Substituting $\nu_f = 1/m$ yields $F = ma$. The correspondence is exact under the regime conditions; departures introduce corrections proportional to $|\nabla\phi|$.

### 2.4 Force Interpretation

| Classical force | TNFR mechanism | Telemetry observable |
|----------------|----------------|---------------------|
| Gravity | Phase-coherence gradient | $-\nabla\Phi_s$ along trajectories |
| Friction | Coherence stabilizer (IL operator) | Reduction of high-frequency $\Delta\mathrm{NFR}$ |
| Harmonic restoring | Phase-gradient confinement | $|\nabla\phi|$ deviation generating restoring pressure |

### 2.5 Integration Scheme

Symplectic integrators (Verlet/Yoshida 4th-order) in `src/tnfr/dynamics/symplectic.py` preserve structural invariants analogously to Liouville's theorem.

Workflow:
1. Select integrator order; record in run metadata.
2. Verify $|\nabla\phi|$ and $K_\phi$ remain within canonical thresholds during integration.
3. Export telemetry ($C(t)$, $\Phi_s$, $|\nabla\phi|$) alongside classical observables ($q$, $p$, energy).
4. Compare against analytic references; flag deviations beyond tolerance.

### 2.6 Validation

**Kepler benchmark** (`examples/12_classical_mechanics_demo.py`): single node in coherence-gradient potential approximating an ellipse with eccentricity $e \approx 0.5$. Artifacts include trajectory overlays, phase-space loops, and conservation plots (energy/angular momentum drift target $< 10^{-4}$).

---

## 3. Inertial Regime (Zero Structural Pressure)

### 3.1 Regime Conditions

$$
\Delta\mathrm{NFR} = 0 \quad \Rightarrow \quad \frac{\partial\mathrm{EPI}}{\partial t} = 0 \text{ (co-moving frame)}
$$

Practical checklist:
- Exclude destabilizers (no OZ/VAL) from operator schedules.
- Confirm $|\nabla\phi| < 10^{-4}$ and $K_\phi \approx 0$ over the interval.
- Record initial phase current $J_\phi$ as the momentum analog; verify $C(t) > 0.99$.

### 3.2 Constant-Velocity Motion

With zero $\Delta\mathrm{NFR}$, the structural state is frozen and nodes translate uniformly. This is the TNFR analog of Newton's first law: no reorganization pressure implies no change in the structural trajectory.

### 3.3 Validation

**Two-train benchmark** (`examples/15_train_crossing_demo.py`):

| Parameter | Train A | Train B |
|-----------|---------|---------|
| Initial position | $x = 0$ km | $x = 600$ km |
| Velocity | $+300$ km/h | $-250$ km/h |
| Operators | [AL, IL, SHA] | [AL, IL, SHA] |

Analytical prediction:

$$
t_c = \frac{600}{300 + 250} \approx 1.0909 \text{ h}, \quad x_c = 300 \cdot t_c \approx 327.27 \text{ km}
$$

Numerical runs match within integration tolerance ($< 10^{-3}$); larger deviations indicate unintended structural forces.

---

## 4. Quantum Regime (High-Dissonance Limit)

### 4.1 Regime Conditions

High phase gradient ($|\nabla\phi| \sim \pi$), boundary reflections, or proximity to phase singularities (vortices). The classical approximation breaks down; discrete resonant modes emerge.

### 4.2 Observable Mapping

| Quantum quantity | Symbol | TNFR analogue | Notes |
|-----------------|--------|---------------|-------|
| Wavefunction | $\psi$ | Complex field $\Psi = K_\phi + iJ_\phi$ | Curvature + current components |
| Energy | $E$ | Structural frequency $\nu_f$ | Domain-specific proportionality |
| Potential | $V(x)$ | $\Phi_s(x)$ | Identical boundary conditions |
| Quantum number | $n$ | Winding number $w$ | $\oint \nabla\phi = 2\pi w$ |
| Collapse | — | Decoherence via IL/SHA | Grammar U2 enforcement |

### 4.3 Quantization Mechanism

1. **Evolution**: Nodes follow $\partial\mathrm{EPI}/\partial t = \nu_f \, \Delta\mathrm{NFR}$.
2. **Boundary feedback**: Reflections inside finite domains superimpose outgoing and incoming phase waves.
3. **Interference**: Coherent modes form when accumulated phase matches $2\pi w$ ($w \in \mathbb{Z}$); otherwise $|\nabla\phi|$ spikes and coherence degrades.
4. **Selection**: Stabilizers drive the system toward minimal $\Delta\mathrm{NFR}$, retaining only resonant modes.

Quantized spectra arise without additional axioms. Superposition of EPI states is the default behavior of linear wave dynamics; "collapse" is the decoherence process where environmental coupling selects eigenstates (grammar rule U2).

### 4.4 Validation

**One-dimensional cavity benchmark** (`examples/13_quantum_mechanics_demo.py`): define cavity length $L$, initialize with random $\nu_f$ and phase profile ($C(t_0) > 0.6$), integrate until $\nu_f$ converges. Expected outcome: discrete $\nu_n$ proportional to $n^2$ (linear dispersion) or $n$ (other media).

---

## 5. Structural Uncertainty and Interference

### 5.1 Uncertainty Relation

For wave packets occupying time window $\Delta t_{\mathrm{EPI}}$ with structural frequency spread $\Delta\nu_f$:

$$
\Delta t_{\mathrm{EPI}} \cdot \Delta\nu_f \geq K \tag{1}
$$

where $K$ depends on the analysis window (Gaussian packets yield $K \approx 0.16$). This emerges from the Fourier relationship between form (EPI) and frequency ($\nu_f$): localizing a pattern in structural space increases its frequency spread, and conversely.

### 5.2 Two-Path Interference

The double-slit experiment maps to two emission nodes executing [AL, RA] into a propagation medium. Receiving nodes integrate $\Psi = K_\phi + iJ_\phi$ and report per-pixel telemetry:

- **Constructive bands**: $\Delta\phi \approx 0$ → increased $C(t)$ and $Si$.
- **Destructive bands**: $\Delta\phi \approx \pi$ → elevated $|\nabla\phi|$ and reduced $C(t)$.

Interference is described entirely through phase-coupled dynamics. No wave/particle duality narrative is invoked; only structural metrics.

### 5.3 Validation

`examples/14_uncertainty_and_interference.py` generates: scatter of $\sigma_t$ vs. $\sigma_f$ with constant-product reference line, 2D detector intensity maps, and fringe-spacing line profiles.

> **Note**: The example demonstrates qualitative uncertainty behavior (constant $\sigma_t \cdot \sigma_f$ product) but does not validate the product against a specific analytical bound. The double-slit section is incomplete. No dedicated test module exists.

---

## 6. Thermodynamic Laws from Phase Dynamics

> **Note**: This section describes a demonstration-level correspondence. The example `17_thermodynamics_demo.py` implements a Kuramoto model where "temperature" is defined as $|\nabla\phi|$, but does not extract a time constant or validate quantitatively against Newton's law of cooling. No dedicated test module exists. Promotion to verified status requires: (a) extraction of decay constant $k$ from simulation, (b) quantitative comparison with the analytical form $T(t) = T_{\mathrm{env}} + (T_0 - T_{\mathrm{env}})e^{-kt}$, and (c) a dedicated test file.

### 6.1 Observable Mapping

| Thermodynamic quantity | TNFR interpretation | Symbol |
|----------------------|---------------------|--------|
| Heat ($Q$) | Incoherent phase noise | $\sigma_\phi$ |
| Temperature ($T$) | Local phase-gradient variance | $\mathrm{Var}(|\nabla\phi|)$ |
| Entropy ($S$) | Structural decoherence | $S \propto 1/C(t)$ |
| Equilibrium | Phase synchronization | $\Delta\phi \to 0$ |

### 6.2 Zeroth Law — Resonant Transitivity

If node A and node C satisfy $|\phi_A - \phi_C| \leq \Delta\phi_{\max}$, and node B satisfies the same relation with C, then $|\phi_A - \phi_B|$ automatically respects the bound. Coupling operators (UM, RA) enforce this check before activation per grammar rule U3.

### 6.3 First Law — Structural Balance

Structural current is conserved up to explicit operator work:

$$
\Delta E_{\mathrm{total}} = \Delta E_{\mathrm{coherent}} + \Delta E_{\mathrm{incoherent}}
$$

In closed experiments, the sum over $J_\phi$ remains constant. This accounts for how resonance-preserving work (coherent) and decohering work (incoherent) partition the same nodal update.

### 6.4 Second Law — Passive Desynchronization

Without stabilizers (IL, THOL), random perturbations drive the network toward larger $|\nabla\phi|$ and reduced coherence. Operators lacking closure steps predictably lose $C(t)$, producing the familiar time arrow without additional hypotheses.

### 6.5 Validation

**Coffee-cup cooling benchmark**: 2D lattice with a high-$\nu_f$, random-phase central patch (sample) surrounded by an aligned-phase, low-$\nu_f$ outer ring (bath). Dynamics follow the coupled-oscillator form:

$$
\frac{d\phi_i}{dt} = \omega_i + \frac{K}{N}\sum_{j \in \mathcal{N}(i)} \sin(\phi_j - \phi_i)
$$

The phase-gradient mismatch decays exponentially:

$$
T(t) = T_{\mathrm{env}} + (T_0 - T_{\mathrm{env}}) \, e^{-kt}
$$

recovering Newton's law of cooling from nodal dynamics.

---

## 7. Regime Transition Summary

The five regimes form a coherent hierarchy indexed by the degree of structural dissonance:

| Regime | $\Delta\mathrm{NFR}$ | $|\nabla\phi|$ | Primary telemetry | Governing reduction |
|--------|---------------------|----------------|-------------------|-------------------|
| Inertial | $= 0$ | $< 10^{-4}$ | $J_\phi$ (momentum) | Constant velocity |
| Classical mechanics | Low | $\to 0$ | $\Phi_s$, $J_\phi$ | $F = ma$ |
| Thermodynamic | Distributed | Moderate | $\mathrm{Var}(|\nabla\phi|)$, $C(t)$ | Cooling laws |
| Quantum | High | $\sim \pi$ | $\Psi$, winding number | Discrete eigenvalues |
| Uncertainty | High + localized | Broadband | $\sigma_t \sigma_f$ product | Fourier bound |

All regimes emerge from the same nodal equation without additional postulates.

---

## 8. Implementation Reference

| Component | Location |
|-----------|----------|
| Classical mechanics mapper | `src/tnfr/physics/classical_mechanics.py` |
| Quantum mechanics module | `src/tnfr/physics/quantum_mechanics.py` |
| Symplectic integrators | `src/tnfr/dynamics/symplectic.py` |
| Structural field computation | `src/tnfr/physics/fields.py` |
| Kepler benchmark | `examples/12_classical_mechanics_demo.py` |
| Quantum cavity benchmark | `examples/13_quantum_mechanics_demo.py` |
| Uncertainty/interference | `examples/14_uncertainty_and_interference.py` |
| Two-train kinematics | `examples/15_train_crossing_demo.py` |

---

## Implementation & Examples

### TNFR ↔ Classical Mechanics Dictionary

> Originally `docs/TNFR_CLASSICAL_MAPPING.md`. Consolidated here as the canonical mapping reference.

**Scope**: Applies to low-dissonance, high-coherence regimes where TNFR reproduces Newtonian behavior (see `src/tnfr/dynamics/nbody.py`).

| TNFR Quantity | Definition (TNFR) | Classical Analog | Notes |
|---------------|-------------------|------------------|-------|
| **EPI** | Coherent form (spatial + kinematic state) | Generalized coordinates $q$, velocities $\dot{q}$ | Structural Triad |
| **νf** | Reorganization rate (Hz_str) | Inertial mass via $m = 1/\nu_f$ | High νf → low inertia |
| **ΔNFR** | Structural pressure | Generalized force $F = -\nabla U$ | Hamiltonian-derived |
| **Φ_s** | Inverse-square ΔNFR accumulation | Potential energy $U(q)$ | U6 confinement ↔ potential wells |
| **\|∇φ\|** | Local desynchronization | Stress/strain rate, tidal gradients | Shear force analog |
| **K_φ** | Phase torsion | Curvature-induced forces (centripetal) | Geometric confinement |
| **ξ_C** | Correlation decay scale | Interaction range / mean free path | Large ξ_C → long-range forces |
| **Ψ = K_φ + i·J_φ** | Complex geometric field | Complexified action density | Hamilton-Jacobi analog |
| **Operator sequences** | Canonical transformations | Work/impulse protocols | Grammar U1-U6 ↔ mechanical admissibility |

**Structural Triad ↔ Phase Space**:
- Form (EPI) → Canonical coordinates $(q, p)$
- Frequency (νf) → Mass/inertia $m$
- Phase (φ/θ) → Canonical phase (action-angle coordinates)

**Field Tetrad ↔ Energetics**:
- Φ_s → potential wells; $\Delta\Phi_s < \varphi$ mirrors bounded energy basins
- |∇φ| → velocity potential gradient (fluid mechanics analog)
- K_φ → curvature-induced forces (centripetal/Coriolis terms)
- ξ_C → interaction range; large ξ_C reproduces gravity/electromagnetism, small ξ_C mimics contact forces

### Executable Demonstrations

| Example | Concept from this document |
|---------|---------------------------|
| [11_classical_limit_comparison.py](../examples/11_classical_limit_comparison.py) | TNFR vs classical N-body comparison |
| [12_classical_mechanics_demo.py](../examples/12_classical_mechanics_demo.py) | Keplerian orbits from symplectic integrator |
| [13_quantum_mechanics_demo.py](../examples/13_quantum_mechanics_demo.py) | Emergent quantization from resonant standing waves |
| [14_uncertainty_and_interference.py](../examples/14_uncertainty_and_interference.py) | Structural uncertainty (ΔForm·Δνf ≥ K), double slit |
| [15_train_crossing_demo.py](../examples/15_train_crossing_demo.py) | Free-particle classical kinematics |

### Key Source Modules

- `src/tnfr/physics/classical_mechanics.py` — Classical limit (Keplerian orbits, Newton's laws)
- `src/tnfr/physics/quantum_mechanics.py` — Quantum regime (quantization, superposition)

---

## 9. References

- [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) — Universal Tetrahedral Correspondence
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) — U1–U6 derivations
- [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) — Conservation laws
- [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) — Lagrangian formulation
- [GLOSSARY.md](GLOSSARY.md) — Operational definitions
