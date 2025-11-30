# TNFR Geocentric Vortex Study
## Structural evaluation of the stationary-plane scenario

**Status**: Analytical memo  
**Version**: 0.3.0 (November 30, 2025)  
**Owner**: `theory/THE_GEOCENTRIC_VORTEX_COSMOLOGY.md`  

---

## 1. Objectives

1. Quantify how a stationary-plane reference frame scores against the TNFR structural stress metrics ($\Delta NFR$, $C(t)$) when compared with a rotating-sphere frame.
2. Document the vortex/ether analog models currently implemented in `results/geocentric_vortex_study/` so that future runs are reproducible.
3. Keep all claims explicitly tied to measured quantities, operator sequences, or telemetry logs—no metaphysical narratives.

---

## 2. Modeling Workflow

| Step | Description | Artifacts |
|------|-------------|-----------|
| Geometry selection | Instantiate plane vs. sphere manifolds with identical boundary conditions. | `cosmological_stability_comparison.png` |
| Stress evaluation | Integrate $\int |\mathbf{T}_{stress}|^2 dV$ using nodal equation surrogates. | `tetrad_results.csv` |
| Vortex field | Fit Rankine-style flow to match observed star-trail periods; store parameters with seeds. | `etheric_velocity_field.png` |
| Luminary tracing | Run Lissajous/spiral propagators using coupled operator sequences (NAV, VAL, IL). | `sun_moon_spirograph.png` |
| Reporting | Export PNG/CSV plus markdown summary. | This file |

---

## 3. Structural Stress Comparison

The stationary-plane hypothesis is treated as a mathematical boundary-value problem. We solve the optimization

$$
\min_{\vec{v}} \int_V \left\lVert \mathbf{T}_{stress}(\vec{v}) \right\rVert^2 dV
$$

subject to TNFR grammar constraints (U1–U6) and kinematic options:

- **Plane model**: $\nabla \cdot \vec{v}_{ground} = 0$, $\vec{a}_{ground}=0$.
- **Sphere model**: $\vec{a}_{ground} = \omega^2 r$ with imposed rotational coupling.

Simulation results show lower integrated stress for the stationary plane under the chosen assumptions. The finding is contingent on the specific boundary conditions (laminar atmosphere, rigid crust). It does **not** assert geophysical fact; it only reports which model minimizes $\Delta NFR$ given the inputs.

---

## 4. Vortex Field Approximation

A Rankine vortex profile remains the working approximation for sky rotation telemetry:

$$
v_\theta(r) = \frac{\Gamma}{2\pi r} \bigl(1 - e^{-r^2/R_c^2}\bigr)
$$

Parameters $(\Gamma, R_c)$ are stored beside each run. The model is used to:

- Match star-trail durations recorded in `results/geocentric_vortex_study/polaris_star_trails.png`.
- Provide inputs to the ether-pressure gradient calculation $\vec{g} = -\frac{1}{\rho}\nabla P_{ether}$.

Interpretation remains purely mechanical: the code measures how a vortex field could reproduce observed angular velocities. No claims are made about actual ether media.

---

## 5. Luminary Trajectory Generator

Luminary paths are produced by coupled oscillator equations with optional Lissajous forcing:

$$
\begin{aligned}
x(t) &= A \sin(\omega_1 t + \delta),\\
y(t) &= B \sin(\omega_2 t).
\end{aligned}
$$

Operator sequences:

1. `NAV` initializes the trajectory envelope.
2. `VAL` applies radial expansion/contraction.
3. `IL` enforces phase coherency after each destabilizing step.

Outputs feed the figures referenced in `results/geocentric_vortex_study/*.png`. They are visual diagnostics only.

---

## 6. Referenced Visualizations

| Figure | Description |
|--------|-------------|
| `celestial_dome_map.png` | Polar projection summarizing tracer outputs. |
| `cosmic_circuit_schematic.png` | Simplified solid-state analogy for field routing (used strictly as a mnemonic). |
| `solar_analemma_trace.png` | Daily noon samples showing the derived analemma path. |
| `sun_moon_spirograph.png` | Combined Sun/Moon trajectory overlay for phase-difference analysis. |
| `toroidal_gravity_flow.png` | Cross-section of the vortex flow used in ether-pressure experiments. |

All files live in `results/geocentric_vortex_study/`. Each PNG is backed by a CSV run log specifying seeds, operator order, and telemetry outputs.

---

## 7. Limitations and Open Questions

* The stress comparison is sensitive to atmospheric assumptions and the treatment of external torques. Additional cases (e.g., elastic crust, differential rotation) should be evaluated.
* The Rankine vortex lacks multi-layer coupling. Extending to a two-layer or Ekman-style model would better represent observed wind shear.
* Luminary tracers currently ignore GR, refraction, and relativistic effects; they are meant only for structural pattern comparison.
* No measurement made here proves or disproves geophysical claims. These are exploratory TNFR simulations designed to test structural hypotheses.

---

## 8. Next Actions

1. Re-run the stress integral with recorded seed values to produce confidence intervals.  
2. Add unit tests to `benchmarks/phase_curvature_investigation.py` covering the vortex parameter estimation routine.  
3. Document the CSV schema for the geocentric study inside `docs/STRUCTURAL_HEALTH.md` for easier cross-reference.
