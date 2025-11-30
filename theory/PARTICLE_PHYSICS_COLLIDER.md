# Particle Collision Modeling Memo

**Status**: Analytical memo  
**Version**: 0.3.0 (November 30, 2025)  
**Owner**: `theory/PARTICLE_PHYSICS_COLLIDER.md`

---

## 1. Scope

This memo outlines how to model high-energy collisions within the TNFR framework using reproducible operator sequences and telemetry. Subatomic particles are represented as coherent EPIs; collisions are treated as controlled bifurcations. The focus is documenting the modeling pipeline and validation steps, not asserting new physics.

---

## 2. Node Specification

| Quantity | Description | Notes |
| --- | --- | --- |
| \(\nu_f\) | Structural frequency of the node (maps to energy). | Extracted from accelerator settings or simulation inputs. |
| \(\mathbf{J}_\phi\) | Phase current (momentum proxy). | Track both magnitude and direction. |
| \(C(t)\) | Coherence during acceleration and collision. | Required for verifying stability before impact. |
| Operator schedule | `[AL, RA, IL, ...]` used to prepare nodes. | Must obey U1–U6; log exact order. |

Parameters should be stored in `results/collider_demo/config_<run>.yml` so headless runs can reproduce the same setup.

---

## 3. Collision Pipeline

1. **Preparation** – Initialize two nodes (e.g., proton analogues) with target \(\nu_f\) and \(\mathbf{J}_\phi\). Apply stabilizers to ensure \(C(t)\) exceeds the threshold before acceleration.  
2. **Acceleration** – Increase \(\mathbf{J}_\phi\) using scripted operators or external forces while logging energy input.  
3. **Overlap phase** – When positions converge, record phase differences and structural pressure \(\Delta \text{NFR}\).  
4. **Bifurcation handling** – If \(\Delta \text{NFR}\) exceeds tolerance, invoke handler sequences (`THOL`, `IL`) to form child EPIs.  
5. **Propagation** – Track the child nodes (“jets”) and collect telemetry per node (\(\nu_f\), \(\mathbf{J}_\phi\), \(C(t)\)).

All steps should include deterministic seeds and metadata (time step, integration method) for auditing.

---

## 4. Conservation & Telemetry Checks

Collisions must satisfy conservation relationships within numerical tolerance:

\[
\sum_i \nu_f^{(i)}(t_0^-) \approx \sum_j \nu_f^{(j)}(t_0^+), \qquad \sum_i \mathbf{J}_\phi^{(i)}(t_0^-) = \sum_j \mathbf{J}_\phi^{(j)}(t_0^+).
\]

Automated tests should compute relative errors and fail runs that exceed configured thresholds. Additional checks include:

- Coherence budgets before/after impact.  
- Spatial distribution of \(\Phi_s\) to confirm detector-region balance.  
- Compliance with grammar rule U4 (destabilizer/handler pairing).

---

## 5. Artifacts

`examples/16_particle_collider_demo.py` produces reference datasets:

- `results/collider_demo/01_collision_event.png` – 2D event display showing incoming nodes and resulting jets.  
- `results/collider_demo/02_energy_spectrum.png` – Histogram of child \(\nu_f\) values.  
- `results/collider_demo/run_<seed>.csv` – Telemetry table (time, \(\nu_f\), \(\mathbf{J}_\phi\), \(C(t)\)).

Files must include headers documenting units, integrator details, and operator sequences. Store SHA manifests in `results/collider_demo/manifest.json`.

---

## 6. Outstanding Work

1. Integrate the collision benchmark into CI with deterministic seeds.  
2. Add uncertainty estimates by running ensembles with slightly varied initial phases and reporting variance in \(\nu_f\) and \(\mathbf{J}_\phi\).  
3. Compare TNFR-derived spectra with published collider data (where available) to quantify predictive value; document residuals and uncertainties.  
4. Record numerical stability limits (time-step restrictions, lattice resolution) to guide future experiments.
