# Nodal Equation to Macroscopic Systems Tutorial

**Status**: Technical reference  
**Version**: 0.2.0 (November 30, 2025)  
**Owner**: `theory/TUTORIAL_FROM_NODAL_EQUATION_TO_COSMOS.md`

---

## 1. Scope

Provide a reproducible roadmap showing how the nodal equation

\[
\frac{\partial \text{EPI}}{\partial t} = \nu_f \Delta \text{NFR}
\]

produces macroscopic transport equations across microscopic, biological, planetary, and neural regimes. The tutorial points to calculators, benchmarks, and telemetry requirements rather than offering speculative narratives.

---

## 2. Baseline Derivation

1. **Decomposition** – Split \(\Delta \text{NFR}\) into diffusive (stabilizing) and solenoidal (transport) components.  
2. **Averaging** – Apply spatial/temporal averaging to obtain coarse-grained PDEs used in `docs/TNFR_MATHEMATICS_REFERENCE.md`.  
3. **Operator Mapping** – Associate TNFR operators with PDE source terms (e.g., `AL` as generation, `IL` as damping) and document grammar requirements.  
4. **Telemetry Projection** – Express the resulting fields in terms of \(\Phi_s\), \(|\nabla \phi|\), \(K_\phi\), and \(\xi_C\) so experiments expose consistent metrics.

Mathematical steps should be paired with executable notebooks (`notebooks/nodal_to_macro_*.ipynb`) to maintain traceability.

---

## 3. Regime Templates

| Regime | Key assumption | Governing reduction | Reference artifacts |
| --- | --- | --- | --- |
| Microscopic (atomic) | Stationary limit \(\partial_t \text{EPI} \approx 0\) | Eigenvalue problems on bounded domains leading to discrete \(\nu_f\) spectra | `examples/38_tnfr_master_class.py`, `results/micro_modes/*.png` |
| Biological (flux capture) | Operator loop `[AL, VAL, OZ, THOL, IL]` with U2 enforcement | Nonlinear transport with competition constraints | `examples/08_emergent_phenomena.py`, `results/biology_flux/*.csv` |
| Planetary (vortex mechanics) | Carrier-modulator decomposition of \(\phi\) fields | Coupled oscillator models compared against ephemerides | `examples/22_planetary_mandalas.py`, `benchmarks/universality_clusters.py` |
| Neural (coherence) | High-dissonance regime with synchronization thresholds | Kuramoto-style reductions plus telemetry of \(C(t)\) and \(\nu_f\) distributions | `examples/19_neuroscience_demo.py`, `results/neural_coherence/*.json` |

Each template lists the files required to regenerate figures and the telemetry that must accompany publications (seeds, integration steps, boundary data).

---

## 4. Implementation Assets

- **Symbolic derivations**: `notebooks/nodal_to_macro.ipynb` (links to `sympy` outputs).  
- **Core libraries**: `src/tnfr/mathematics/operators.py`, `src/tnfr/dynamics/symplectic.py`.  
- **Benchmarks**: `benchmarks/phase_curvature_investigation.py`, `benchmarks/benchmark_optimization_tracks.py`.  
- **Tests**: `tests/test_classical_mechanics.py`, `tests/test_operator_sequences.py`, `tests/test_quantum_examples.py`.

All intermediate data should be stored under `results/tutorial_nodal_to_macro/` with metadata (seed, lattice spacing, operator schedule).

---

## 5. Validation Checklist

1. **Symbolic** – Confirm that reduction steps match notebook outputs and that assumptions (e.g., \(|\nabla \phi|\) bounds) are documented.  
2. **Numerical** – Reproduce example scripts covering each regime and compare against recorded telemetry.  
3. **Regression** – Ensure changes are caught by monotonicity/bifurcation tests in `tests/`.  
4. **Telemetry** – Verify each dataset exposes \(C(t)\), \(\Phi_s\), \(|\nabla \phi|\), and \(K_\phi\).

---

## 6. Open Actions

1. Publish short reference notebooks showing each reduction step with inline explanations.  
2. Add CI jobs that run representative scripts for every regime with fixed seeds.  
3. Document limitations (e.g., sensitivity to lattice regularity, boundary reflections) so contributors can prioritize extensions.  
4. Create a troubleshooting appendix describing common failure modes (unbounded \(\Delta \text{NFR}\), missing stabilizers, telemetry gaps).

---

## 7. Contact

For questions or contributions, open issues referencing this file and include links to the supporting notebooks, benchmarks, or telemetry artifacts.
