# TNFR ↔ Classical Mechanics Mapping

**Status**: Research-to-canonical pipeline (documentation phase)  
**Purpose**: Provide an explicit dictionary between TNFR structural quantities and the objects of classical mechanics (positions, momenta, energies) to guide experiments, proofs, and SDK usage.  
**Scope**: Applies to low-dissonance, high-coherence regimes where TNFR reproduces Newtonian behavior (see `src/tnfr/dynamics/nbody.py`).

---

## 1. High-Level Correspondence

Classical mechanics emerges from TNFR when the nodal equation `∂EPI/∂t = νf · ΔNFR` is evaluated on slowly varying, phase-synchronized EPIs. The table below summarizes the mapping.

| TNFR Quantity | Definition (TNFR) | Classical Analog | Notes & References |
|---------------|-------------------|------------------|--------------------|
| **EPI** (Primary Information Structure) | Coherent form storing spatial + kinematic state. | Generalized coordinates `q` and velocities `\dot{q}`. | See `AGENTS.md` (Structural Triad) and `src/tnfr/dynamics/nbody.py` docstring. |
| **νf** (Structural frequency) | Reorganization rate (Hz_str). | Inertial mass via `m = 1/νf`. | High νf → low inertia; matches remarks in `nbody.py`. |
| **ΔNFR** (Nodal gradient) | Structural pressure from Hamiltonian or assumed potential. | Generalized force `F = -∇U`. | Pure TNFR: `ΔNFR = i[H_int, ·]/ℏ_str` (`nbody_tnfr.py`). |
| **Φ_s** (Structural potential) | Inverse-square accumulation of ΔNFR; monitors confinement. | Potential energy landscape `U(q)`. | Grammar rule U6 treats Φ_s minima like gravitational wells (`AGENTS.md`). |
| **\|∇φ\|** (Phase gradient) | Local desynchronization measure. | Stress/strain rate or tidal gradients. | Bounded analog to shear forces in continuum mechanics. |
| **K_φ** (Phase curvature) | Geometric torsion of phase field. | Curvature-induced forces (e.g., centripetal). | References: `docs/STRUCTURAL_FIELDS_TETRAD.md`. |
| **ξ_C** (Coherence length) | Correlation decay scale. | Interaction range / mean free path. | Large ξ_C → long-range forces like gravity. |
| **Ψ = K_φ + i·J_φ** | Complex geometric field coupling curvature and current. | Complexified action density (Hamilton-Jacobi). | Emerging invariant from unified field audit. |
| **Operator sequences** | Canonical transformations (AL, IL, UM, …). | Work/impulse protocols (impulses, damping, coupling). | Grammar U1-U6 enforces mechanical admissibility. |

---

## 2. Derivation Highlights

### 2.1 Nodal Equation ↔ Newton's Second Law

Starting from the nodal equation:

```text
∂EPI/∂t = νf · ΔNFR
```

- Write EPI components explicitly: `EPI = (q, \dot{q})` (stored structurally).
- Substitute `m = 1/νf` to obtain `m · ∂²q/∂t² = F`, with `F ≡ ΔNFR`.
- Therefore Newton's law is the low-dissonance limit of TNFR where ΔNFR arises from a classical potential.

### 2.2 Structural Triad ↔ Phase Space

| TNFR Triad | Classical Object | Commentary |
|------------|------------------|------------|
| Form (EPI) | Canonical coordinates `(q, p)` | EPI stores both spatial configuration and flow information. |
| Frequency (νf) | Mass/inertia `m` | Structural inertia resists reorganization exactly like mass resists acceleration. |
| Phase (φ/θ) | Canonical phase (angle variable) | Phase locking enforces resonance (analogous to action-angle coordinates). |

### 2.3 Field Tetrad ↔ Energetics

- **Φ_s** corresponds to potential wells; maintaining `ΔΦ_s < φ` (golden ratio) mirrors confining a system inside a bounded potential energy basin.
- **|∇φ|** measures local stress similarly to the gradient of velocity potential in fluid mechanics.
- **K_φ** detects torsion, which maps to curvature-induced forces (think centripetal/Coriolis terms) in classical dynamics.
- **ξ_C** sets the interaction range. Long ξ_C reproduces long-range classical forces (gravity, electromagnetism), short ξ_C mimics contact forces.

These analogies follow the derivations in `docs/STRUCTURAL_FIELDS_TETRAD.md` and the universality analysis in `docs/TNFR_FORCES_EMERGENCE.md`.

---

## 3. Validation Pathway

1. **Analytical**
   - Start from the TNFR Hamiltonian `H_int = H_coh + H_freq + H_coupling` (`src/tnfr/operators/hamiltonian.py`).
   - Evaluate the commutator to obtain `ΔNFR = i[H_int, ·]/ℏ_str`.
   - Show that in phase-synchronized regimes this reduces to `-∇U` with `U` identified as the coherence potential.

2. **Numerical**
   - Use `src/tnfr/dynamics/nbody.py` (classical assumption) and `src/tnfr/dynamics/nbody_tnfr.py` (pure TNFR) on identical initial conditions.
   - Regime: nearly synchronized phases, large ξ_C, small |∇φ|.
   - Metrics: compare trajectories, total energy drift, momentum conservation, and Φ_s telemetry.

3. **Telemetry Mapping**
   - Log `Φ_s`, `|∇φ|`, `K_φ`, `ξ_C` alongside classical quantities (U, |F|, curvature, interaction range).
   - Demonstrate monotonic relationships to cement the mapping.

---

## 4. Reference Experiment (`examples/11_classical_limit_comparison.py`)

To make the correspondence tangible, run the paired simulation script:

```bash
python examples/11_classical_limit_comparison.py --t-final 15.0 --dt 0.01
```

**What it does**:

- Evolves the classical solver (`NBodySystem`) and the pure TNFR solver (`TNFRNBodySystem`) with identical initial conditions (two-body low-dissonance orbit).
- Annotates both graphs with ΔNFR telemetry (accelerations for the classical case, Hamiltonian-derived ΔNFR for TNFR) before calling `compute_structural_telemetry()`.
- Prints per-system diagnostics: energy drift %, potential energy, and statistical summaries of Φ_s, |∇φ|, |K_φ|, ξ_C.

**How to interpret**:

- Energy drift values should match within numerical tolerances, confirming that the TNFR Hamiltonian reproduces Newtonian conservation laws in this regime.
- Φ_s mean tracks the classical potential energy (up to scaling), while |∇φ| mirrors classical force magnitudes; ξ_C remaining high indicates the long-range coupling characteristic of gravity.
- Deviations appear only when phases desynchronize (raising |∇φ|) or when coherence length shrinks, signaling departure from the classical limit.

Researchers can extend the script with additional logging/plots or port it into a notebook to capture trajectory overlays.

---

## 5. Next Steps

1. **Derivation Expansion**: Extend §2 with full Lagrangian/Hamiltonian calculus (reference `docs/source/theory/08_classical_mechanics_euler_lagrange.md`).
2. **Notebook Demonstration**: Create a Jupyter notebook comparing `nbody.py` vs `nbody_tnfr.py`, highlighting convergence in low-dissonance conditions.
3. **Operator-Level Mapping**: Document how specific operator sequences (e.g., `UM → RA → IL`) emulate classical impulses, damping, and coupling.
4. **Publish Telemetry Plots**: Add plots showing Φ_s tracking classical potential energy for canonical examples (Kepler orbit, harmonic oscillator).

This document becomes the canonical entry point for researchers extending TNFR into classical mechanics, ensuring that every analogue is traceable back to the nodal equation and structural field tetrad.
