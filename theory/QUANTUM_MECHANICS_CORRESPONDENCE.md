# Quantum Correspondence Memo

**Status**: Technical reference  
**Version**: 0.3.0 (November 30, 2025)  
**Owner**: `theory/QUANTUM_MECHANICS_CORRESPONDENCE.md`

---

## 1. Scope

Summarize how the high-dissonance regime of TNFR reproduces standard quantum-mechanical behavior, document the mapping of observables, and link to reproducible simulations. Serve as the reference for `examples/13_quantum_mechanics_demo.py` and related experiments.

---

## 2. Correspondence Table

| Quantum quantity | Symbol | TNFR analogue | Notes |
| --- | --- | --- | --- |
| Wavefunction | \(\psi\) | Complex structural field \(\Psi = K_\phi + i J_\phi\) | Store both curvature and current components. |
| Energy | \(E\) | Structural frequency \(\nu_f\) | Proportional via domain-specific constant. |
| Potential | \(V(x)\) | Structural potential \(\Phi_s(x)\) | Use identical boundary conditions. |
| Quantum number | \(n\) | Winding number \(w\) | Enforced via \(\oint \nabla \phi = 2\pi w\). |
| Collapse | - | Decoherence via stabilizers (`IL`, `SHA`) | Implementation follows grammar rule U2. |

---

## 3. Quantization Mechanism

1. **Evolution** - Nodes follow \(\partial \text{EPI} / \partial t = \nu_f \Delta \text{NFR}\).  
2. **Boundary feedback** - Reflections inside finite domains superimpose outgoing and incoming phase waves.  
3. **Interference outcome** - Coherent modes occur when accumulated phase matches integer multiples of \(2\pi\); otherwise \(|\nabla \phi|\) spikes and coherence degrades.  
4. **Selection** - Stabilizers drive the system toward minimal \(\Delta \text{NFR}\), leaving only resonant modes. Quantized spectra arise without adding axioms.

Telemetry requirements: log \(C(t)\), \(\nu_f(t)\), winding number estimates, and structural potential profiles for each run in `results/quantum_demo/run_<seed>.csv`.

---

## 4. One-Dimensional Cavity Benchmark

Experiment steps implemented in `examples/13_quantum_mechanics_demo.py`:

1. Define cavity length \(L\) and boundary operators enforcing reflective conditions.  
2. Initialize node with random \(\nu_f\) and phase profile while keeping \(C(t_0)\) above 0.6.  
3. Integrate using the high-dissonance solver until \(\nu_f\) converges or a timeout occurs.  
4. Record converged frequencies \(\nu_n\), standing-wave shapes, and convergence rates.

Expected outcome: discrete \(\nu_n\) proportional to \(n^2\) (for linear dispersion) or \(n\) (for other media). Deviations should be flagged with the seed and configuration file for review.

---

## 5. Artifacts

- `results/quantum_demo/01_quantization_levels.png` - histogram of observed \(\nu_n\).  
- `results/quantum_demo/02_convergence.png` - time series of coherence and \(\nu_f\) per run.  
- `results/quantum_demo/03_wavefunctions.png` - reconstructed \(\Psi\) amplitudes for the first few modes.  
- `results/quantum_demo/run_<seed>.csv` - full telemetry with \(C(t)\), \(\nu_f\), \(|\nabla \phi|\), and winding numbers.

Ensure files include metadata (seed, cavity length, integration step, operator schedule).

---

## 6. Outstanding Work

1. Add uncertainty quantification by rerunning the benchmark over randomized boundary conditions and reporting \(\nu_n\) variance.  
2. Extend mapping examples to multi-node entanglement analogues (phase-locked pairs) with explicit telemetry.  
3. Integrate these runs into CI so quantization regressions fail when discretization artifacts appear.

