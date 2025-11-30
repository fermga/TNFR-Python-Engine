# Celestial Mechanics Phenomenology Memo

**Status**: Technical reference  
**Version**: 0.2.0 (November 30, 2025)  
**Owner**: `theory/CELESTIAL_MECHANICS_PHENOMENOLOGY.md`

---

## 1. Scope

Define reproducible experiments for modeling observed orbital behavior via TNFR observables (\(\nu_f, \Phi_s, |\nabla \phi|, K_\phi\)). Emphasize telemetry, operator schedules, and validation criteria instead of qualitative narratives.

---

## 2. Phase-Gradient Model

- Central node phase field:
	\[
	\phi(r, \theta, t) = \omega_0 t - k r + m \theta.
	\]
- Secondary nodes minimize \(|\Delta \phi|\) under the nodal equation, yielding effective attraction.  
- Stability requirements: \(|\nabla \phi| < 0.2904\), coupling via `UM`/`RA` must satisfy U3, and `OZ` events signal resonance loss.

Telemetry: log phase gradients, coherence, and applied operators for every run.

---

## 3. Orbit Construction Workflow

1. Define 2D domain (projection of visible sky).  
2. Initialize vortex field with specified \(\omega_0\) and radial decay parameters (store in config file).  
3. Integrate node trajectories using phase-gradient feedback only.  
4. Record \(C(t)\), \(\Phi_s\), \(K_\phi\), and trajectory geometry.  
5. Compare resulting limit cycles with observational ephemerides; report residuals (phase, distance).

---

## 4. Interpretation Guidelines

- Identify regions where \(|\Delta \phi|\) remains constant to classify circular/elliptical paths.  
- Deviations show up as shifts in \(K_\phi\) and reductions in coherence; document thresholds triggering correction operators.  
- Kepler-like relations emerge from synchronization rather than imposed force laws; support conclusions with telemetry plots.

---

## 5. Next Actions

1. Add automated comparison against JPL ephemerides in `benchmarks/`.  
2. Publish reference datasets (phase fields, trajectories, telemetry) under `results/celestial_mechanics/`.  
3. Study sensitivity to \(\omega_0\) perturbations and document tolerance bands.
