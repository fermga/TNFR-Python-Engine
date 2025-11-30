# Geocentric Vortex Evidence Memo

**Status**: Observational data reference  
**Version**: 0.3.0 (November 30, 2025)  
**Owner**: `theory/GEOCENTRIC_VORTEX_EVIDENCE.md`

---

## 1. Scope

Provide measurement protocols and analysis checkpoints for datasets often cited when discussing geocentric renderings (star trails, solar motion, horizon tests, hydraulic surveys). Every claim must point to raw data plus quantified uncertainties. Interpretations stay within the TNFR field framework: coherence `C(t)`, phase gradients, curvature, and telemetry from the nodal simulations that accompany each observation.

All artifacts live under `results/geocentric_vortex_evidence/` with metadata tables describing instrument settings, calibration steps, and links to reproducible notebooks.

---

## 2. Observational Data Streams

### 2.1 Star-Trail Imaging

- Capture long-exposure sequences centered on the celestial pole using calibrated equatorial mounts.  
- Record exposure time, sensor geometry, temperature, and pointing accuracy in `star_trails_metadata.json`.
- Compare extracted trail curvature and drift against TNFR nodal simulations that encode precession, nutation, and parallax. Output residual plots and ensure reported uncertainties exceed instrument noise floors.

### 2.2 Seasonal Solar Motion

- Collect daily solar altitude/azimuth pairs via theodolite or horizon-tracking software. Store values in `solar_motion.parquet` with timestamps (UTC) and site-specific metadata (lat/long, elevation).  
- Fit both heliocentric baselines and TNFR field reconstructions; publish residual RMS, phase offsets, and field thresholds (|∇φ|, K_φ) encountered in the simulation.

### 2.3 Water-Level Surveys

- Document leveling campaigns specifying instrument model, calibration certificate, baseline length, and applied geoid correction.  
- Present the difference between planar leveling and geoid-constrained leveling for each baseline; include uncertainty propagation from instrument specs.  
- Where structural arguments are made, connect them to TNFR coherence readings rather than rhetorical statements about curvature.

### 2.4 Horizon/LoS Tests

- For optical visibility studies, log lens focal length, aperture, sensor size, refraction estimates, and weather data.  
- Compute theoretical disappearance angles for multiple curvature hypotheses and compare to measured target elevations.  
- Provide raw frames whenever possible along with calibration targets to rule out sensor artifacts.

---

## 3. Data Management and Telemetry

- **File structure**: separate `/raw`, `/processed`, and `/analysis` folders per dataset. Include SHA256 manifests for integrity.
- **Telemetry linkage**: each observation referencing TNFR simulations must cite the exact operator log (e.g., `results/geocentric_vortex_evidence/logs/<run_id>.json`) showing applied sequences and resulting `C(t)` curves.
- **Uncertainty budgets**: maintain spreadsheets (or notebooks) detailing measurement, model, and environmental error contributions. Avoid point estimates without uncertainty ranges.

---

## 4. Analysis Workflow

1. Acquire raw measurements with complete calibration metadata.  
2. Run paired TNFR simulations (e.g., star-trail field models) using identical timestamps and site parameters.  
3. Extract structural fields (`Φ_s`, `|∇φ|`, `K_φ`, `ξ_C`) across the region of interest; log whenever thresholds from `docs/STRUCTURAL_FIELDS_TETRAD.md` are approached.  
4. Compute residuals between observation and model predictions; summarize results in reproducible tables with columns `[dataset, metric, residual, uncertainty, operator_sequence]`.  
5. Publish plots, residual histograms, and telemetry overlays via notebooks exported to HTML (`nbconvert`) into `results/reports/geocentric_vortex_evidence/`.

---

## 5. Interpretation Guardrails

- Observational agreement or disagreement should be framed in terms of quantified residuals, not categorical declarations.  
- Never extrapolate beyond the measurement's spatial/temporal window; note assumptions (e.g., ignoring atmospheric dispersion) directly in captions.  
- Highlight limitations: instrument drift, atmospheric seeing, refraction models, or TNFR simulation simplifications.  
- When referencing external sources (engineering drawings, survey docs), cite exact document titles and include scans when licensing allows.

---

## 6. Outstanding Work

1. Build `notebooks/geocentric_vortex_validation.ipynb` to centralize star-trail and solar-motion comparisons with automated report generation.  
2. Add standardized telemetry templates (`*.schema.json`) so every dataset publishes identical field metrics.  
3. Integrate atmospheric refraction modeling into horizon tests and document its effect on residual uncertainty.
