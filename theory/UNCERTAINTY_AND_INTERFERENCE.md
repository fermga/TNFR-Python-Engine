# Structural Uncertainty & Interference Memo

**Status**: Technical reference  
**Version**: 0.3.0 (November 30, 2025)  
**Owner**: `theory/UNCERTAINTY_AND_INTERFERENCE.md`

---

## 1. Scope

Define how TNFR captures time–frequency bounds and interference phenomena using structural metrics. This memo documents the governing equations, workflow for `examples/14_uncertainty_and_interference.py`, required telemetry (\(\sigma_t\), \(\sigma_f\), \(C(t)\), \(|\nabla \phi|\), \(K_\phi\)), and artifact locations under `results/quantum_demo_2/`.

---

## 2. Experiment Inputs and Workflow

1. Configure packet envelopes (Gaussian default) with width parameter `sigma_t` saved in `configs/uncertainty/*.yaml`.  
2. Initialize emission nodes with `[AL, RA]` sequences; record seeds, operator order, and lattice resolution in `results/quantum_demo_2/run_<seed>.json`.  
3. For interference studies, instantiate two emission channels propagating through a shared medium while detectors sample the complex field \(\Psi = K_\phi + i J_\phi\).  
4. Execute `python examples/14_uncertainty_and_interference.py --config <file>` to generate plots plus telemetry tables.

---

## 3. Structural Uncertainty Relation

Packets occupying a time window \(\Delta t_{\text{EPI}}\) with structural frequency spread \(\Delta \nu_f\) obey

\[
\Delta t_{\text{EPI}} \cdot \Delta \nu_f \ge K,
\]

where \(K\) depends on the analysis window (Gaussian packets yield \(K \approx 0.16\)). Interpretation:

- Tight temporal localization increases spectral width and structural pressure variance.  
- Stable frequency estimation requires longer observation windows, trading temporal resolution for spectral stability.

Telemetry checklist:

1. Log packet envelope width \(\sigma_t\) and store alongside configuration hashes.  
2. Compute \(\sigma_f\) via FFT of \(\nu_f(t)\) with documented windowing parameters.  
3. Report \(\sigma_t \sigma_f\) (mean ± uncertainty) in `results/quantum_demo_2/uncertainty_metrics.csv`.  
4. Track \(C(t)\), \(|\nabla \phi|\), and \(K_\phi\) to ensure canonical thresholds (<0.2904, <2.8274) remain satisfied.

---

## 4. Two-Path Interference Model

The double-slit experiment maps to two emission nodes running `[AL, RA]` into a propagation medium. Receiving nodes integrate \(\Psi\) and log per-pixel telemetry:

- Constructive bands: \(\Delta \phi \approx 0\) → increased coherence \(C(t)\) and sense index \(Si\).  
- Destructive bands: \(\Delta \phi \approx \pi\) → elevated \(|\nabla \phi|\) and reduced \(C(t)\).

Detector outputs include coherence, phase difference, accumulated intensity, and structural fields; interference is described entirely through phase-coupled dynamics without wave/particle narratives.

---

## 5. Reproduction Script & Artifacts

`examples/14_uncertainty_and_interference.py` generates:

- `results/quantum_demo_2/01_uncertainty_principle.png` — scatter of \(\sigma_t\) vs. \(\sigma_f\) with constant-product reference line.  
- `results/quantum_demo_2/02_interference_pattern.png` — 2D detector intensity map.  
- `results/quantum_demo_2/03_interference_profile.png` — line-out showing fringe spacing.  
- `results/quantum_demo_2/run_<seed>.csv` — tabulated \(\sigma_t\), \(\sigma_f\), \(\sigma_t \sigma_f\), coherence, \(|\nabla \phi|\), and detector metrics.

All artifacts must carry metadata (seed, grid size, operator schedules) for reproducibility.

---

## 6. Outstanding Work

1. Extend uncertainty experiments to spatial domains (\(\Delta x\) vs. \(\Delta k\)) and compare against analytical bounds.  
2. Automate fringe-visibility calculation plus regression tests tied to reference seeds.  
3. Perform sensitivity sweeps over detector spacing and medium parameters, logging resultant \(C(t)\) and \(|\nabla \phi|\) trends.

