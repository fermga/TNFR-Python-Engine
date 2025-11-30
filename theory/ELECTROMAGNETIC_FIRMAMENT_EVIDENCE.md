# Electromagnetic Firmament Evidence Memo

**Status**: Measurement and modeling reference  
**Version**: 0.3.0 (November 30, 2025)  
**Owner**: `theory/ELECTROMAGNETIC_FIRMAMENT_EVIDENCE.md`

---

## 1. Scope

Collect electromagnetic measurements relevant to TNFR celestial analyses (optical scattering, ionospheric circuits, tidal coupling) and tie each dataset to reproducible instrumentation logs. Interpretations must reference structural fields (Φ_s, |∇φ|, K_φ, ξ_C) or operator sequences extracted from paired simulations. All raw files live under `results/electromagnetic_firmament/` with manifest hashes and metadata sheets.

---

## 2. Measurement Inventory

| Phenomenon | Instrumentation | Key Metrics | Repository Artifacts |
| --- | --- | --- | --- |
| Crepuscular rays | Stereo photogrammetry rigs + radiance meters | Convergence angle vs. cloud altitude; scattering phase function | `results/electromagnetic_firmament/crepuscular/*.csv`, `*.png` |
| Stellar scintillation | High-speed photometer with chromatic filter wheel | Intensity variance spectra, coherence time | `results/.../scintillation/telemetry.parquet` |
| Tidal phase offsets | Tide gauges co-located with GNSS receivers | Phase lead/lag relative to Moon/Sun in minutes | `results/.../tides/phase_analysis.csv` |
| Vertical electric field | Field mills + balloon sondes | Mean Ez, diurnal variance, correlation with Φ_s gradients | `results/.../ez_profiles.nc` |

Only measurements with calibration logs and uncertainty budgets remain in scope. Anecdotal content has been removed.

---

## 3. Modeling Guidelines

- Model luminary-driven excitations as field oscillators with explicit \(\nu_f\) and curvature terms; avoid metaphorical analogies.  
- Translate hypotheses (e.g., “firmament conductivity”) into solvable PDEs coupled to TNFR operators (`AL`, `UM`, `IL` sequences).  
- Provide direct links from narrative text to raw data and analysis notebooks; no claim stands without referencing telemetry plus scripts.

---

## 4. Experiment Templates

### 4.1 Moonlight Temperature Differential

1. Deploy dual thermistor probes with NIST-traceable calibration sheets.  
2. Alternate exposures between moonlit and shaded conditions while logging wind speed, humidity, and sky clarity.  
3. Compare results with radiative-transfer simulations and TNFR field reconstructions; publish uncertainty envelopes (instrument ± environmental).

### 4.2 Ocean–Ionosphere Circuit

Represent the cavity as a capacitor of plate separation `h` and area `A` with vertical field

\[
E_z = \frac{\sigma}{\epsilon_0}.
\]

Link measured Ez and charge density σ to structural potential Φ_s retrieved from TNFR simulations. Store model inputs/outputs in `results/electromagnetic_firmament/circuit_model/*` for audit.

---

## 5. Telemetry and Validation

- Capture `C(t)`, Φ_s, |∇φ|, and K_φ values whenever simulations are run to contextualize observations.  
- Maintain `validation_log.md` alongside datasets summarizing residuals between measurement and model, plus operator sequences used.  
- Require nbconvert-generated HTML reports for every major update so reviewers can reproduce plots without executing notebooks.

---

## 6. Outstanding Work

1. Complete stereo photogrammetry campaign for crepuscular rays; publish depth reconstructions and scattering fits.  
2. Integrate GNSS-tied tide gauge data with TNFR tidal field solvers to quantify phase offsets and coherence trends.  
3. Expand vertical electric-field measurements to include balloon profiles, relating altitude-dependent Φ_s gradients to ξ_C estimates.
