# Classical Mechanics Correspondence Memo

**Status**: Technical reference  
**Version**: 0.3.0 (November 30, 2025)  
**Owner**: `theory/CLASSICAL_MECHANICS_CORRESPONDENCE.md`

---

## 1. Scope

Document the precise conditions under which TNFR nodal dynamics reduce to classical constant-mass mechanics, list implementation hooks, and summarize validation artifacts. The memo confines itself to reproducible limits and telemetry.

---

## 2. Mapping Conditions

Low-dissonance regime:

\[
|\nabla \phi| \rightarrow 0, \qquad \nu_f = \text{constant}, \qquad C(t) \approx 1.
\]

Under these constraints the nodal equation becomes equivalent to Newton’s equations. The working dictionary is summarized below.

| Classical item | Symbol | TNFR quantity | Notes |
| --- | --- | --- | --- |
| Position | \(q\) | Spatial component of EPI | Extract via `ClassicalMechanicsMapper.position`. |
| Velocity | \(\dot q\) | Flow component | Same accessor. |
| Mass | \(m\) | \(1/\nu_f\) | Stored in telemetry for audits. |
| Force | \(F\) | \(\Delta \text{NFR}\) | Must be reported in structural units. |
| Potential | \(V\) | \(\Phi_s\) | Map gradients directly. |
| Action | \(S\) | Phase accumulation | Optional for diagnostics. |

Derivation sketch:

\[
\frac{\partial \text{EPI}}{\partial t} = \nu_f \Delta \text{NFR} \Rightarrow \frac{dv}{dt} = \nu_f \Delta \text{NFR}_{\text{force}}.
\]

Substituting \(\nu_f = 1/m\) yields \(F = m a\). Provide derivation references to `AGENTS.md` (Foundational Physics) when citing this result.

---

## 3. Structural Interpretation of Standard Forces

| Classical label | TNFR mechanism | Observable |
| --- | --- | --- |
| Gravity | Phase-coherence gradient | Monitor \(-\nabla \Phi_s\) along trajectories. |
| Friction | Coherence stabilizer (IL) | Track reduction in high-frequency \(\Delta \text{NFR}\). |
| Harmonic restoring | Phase-gradient confinement | Measure \(\lvert \nabla \phi \rvert\) deviation and resulting pressure. |

Each entry should be backed by telemetry traces before publication; speculative language has been removed.

---

## 4. Symplectic Structure and Workflow

- `src/tnfr/dynamics/symplectic.py` hosts Verlet/Yoshida schemes used for constant-mass integrations.
- `src/tnfr/physics/classical_mechanics.py` maps TNFR states into \((q, p)\) tuples and exposes Poisson-bracket utilities.
- `tests/test_classical_mechanics.py` checks conservation of structural invariants and regression outputs.

Workflow:

1. Select the integrator order (default 4th-order Yoshida) and record it in run metadata.  
2. Ensure low-dissonance conditions by verifying \(|\nabla \phi|\) and \(K_\phi\) remain within canonical thresholds during integration.  
3. After each run, export telemetry (`C(t)`, \(\Phi_s\), \(|\nabla \phi|\)) alongside classical observables (`q`, `p`, energy) to `results/classical_demo/run_<seed>.csv`.  
4. Compare outputs with analytic references; flag deviations beyond tolerances in the regression report.

The integrator preserves coherence analogously to Liouville’s theorem; cite test logs rather than informal statements.

---

## 5. Example Experiment (Kepler Benchmark)

`examples/12_classical_mechanics_demo.py` configures a single node in a coherence-gradient potential to approximate an ellipse with eccentricity \(e \approx 0.5\).

Artifacts:

- `results/classical_demo/01_trajectory.png` – orbital path overlay.  
- `results/classical_demo/02_phase_space.png` – \((q, p)\) loop confirming bounded motion.  
- `results/classical_demo/03_conservation.png` – relative drift of total energy/ang. momentum (target < \(10^{-4}\)).

Metadata (seed, \(\Delta t\), integrator order) lives in `results/classical_demo/run_<seed>.json`. Runs should be rerun whenever integrator changes occur.

---

## 6. Outstanding Work

1. Extend regression coverage to include oscillators and free-fall trajectories with analytic comparisons.  
2. Publish a unit-conversion note mapping structural units to SI when benchmarking against laboratory data.  
3. Automate telemetry capture (`C(t)`, \(\Phi_s\), \(|\nabla \phi|\)) for every example run to document compliance with the low-dissonance assumption.

