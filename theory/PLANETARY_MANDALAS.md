# Planetary Mandalas Memo

**Status**: Technical reference (orbital visualization)  
**Version**: 0.2.0 (November 30, 2025)  
**Owner**: `theory/PLANETARY_MANDALAS.md`

---

## 1. Scope

Document a reproducible workflow for constructing geocentric resonance patterns (“mandalas”) from TNFR nodal simulations. All claims must derive from telemetry: coherence `C(t)`, phase gradients `|∇φ|`, curvature `K_φ`, and recorded frequency ratios. Narrative descriptions of symbolic meaning are out of scope.

Reference implementation: `examples/22_planetary_mandalas.py` → outputs PNG assets under `results/geocentric_vortex_study/`.

---

## 2. Inputs and Configuration

- **Orbital data**: Adopt synodic periods and semi-major axes from JPL Horizons (store CSV snapshot in `data/planetary_periods.csv`).
- **Simulation grid**: 2D geocentric plane with Earth fixed at origin; each planet modeled as a secondary node with angular velocity `ω_p = 2π / T_p`.
- **Time resolution**: Default `DT = 0.01` (structural years); adjust to capture high-frequency planets without aliasing.
- **Run duration**: Integrate over the least common multiple of Earth and planet periods (e.g., Venus 8 yr, Mars 16 yr) to close loops.
- **Operators**: Use `UM`/`RA` to maintain coupling during integration; apply `IL` when coherence drops below 0.8; forbid `OZ` unless explicitly testing bifurcation.

Store configuration files (`yaml`) per target planet under `configs/mandalas/` so that runs remain reproducible.

---

## 3. Simulation Workflow

1. Load orbital parameters and generate node objects (class `GeocentricPlanet`).
2. Integrate nodal equation for both Earth and target planet; compute geocentric vector via difference (epicycle analogy).  
    `x_geo = x_p - x_e`, `y_geo = y_p - y_e`.
3. Sample trajectories at uniform time steps, store arrays in `results/geocentric_vortex_study/<planet>_path.npy`.
4. Plot trajectories with consistent axes, add coherence annotations, and export PNG/SVG along with JSON metadata capturing telemetry and rational frequency fits.
5. Record summary table (planet, duration, ratio, residual RMS, max |∇φ|, mean K_φ) in `results/geocentric_vortex_study/summary.csv`.

Example execution:

```pwsh
python examples/22_planetary_mandalas.py
```

---

## 4. Telemetry and Validation

- **Frequency ratios**: Fit rational approximations `n:m` to angular velocity ratios; report residual < 1e-3 where loops appear closed.
- **Field thresholds**: Ensure `|∇φ| < 0.2904` and `|K_φ| < 2.8274` throughout the run. Flag frames violating these limits.
- **Coherence**: Track `C(t)`; mandala plots are only published when coherence stays above 0.75. Attach `results/geocentric_vortex_study/<planet>_telemetry.parquet` for audit.
- **Observation comparison**: Overlay simulated polar coordinates with ephemeris samples (monthly) and compute phase drift, amplitude error. Report metrics in validation log.

---

## 5. Interpretation Guidelines

- Closed figures indicate near-rational ratios of structural frequencies, not metaphysical alignment. Document the ratio, error bars, and telemetry evidence.
- Distortions or cusp-like features correlate with coherence drops or curvature spikes; note operator interventions used to stabilize the run.
- Highlight deviations between TNFR integration and ephemeris data, especially secular trends not captured by circular models.

---

## 6. Outstanding Work

1. Extend model to include perturbations (e.g., Jupiter influence on Mars) and document resulting changes in `C(t)` and `|∇φ|`.
2. Publish raw telemetry files and plotting notebooks under `notebooks/planetary_mandalas.ipynb` with nbconvert HTML exports.
3. Automate rational fit reporting and regression tests comparing to latest JPL ephemerides.
