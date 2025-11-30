# Vortex Celestial Mechanics Memo

**Status**: Analytical reference  
**Version**: 0.3.0 (November 30, 2025)  
**Owner**: `theory/VORTEX_CELESTIAL_MECHANICS.md`

---

This memo documents requirements for evaluating vortex-style celestial coordinate models without making cosmological claims. The objective is to test whether a vortex parameterization explains observed geocentric trajectories better than standard orbital descriptions.

## 1. Scope

- Define the mathematical form of the carrier-modulator decomposition used to generate Lissajous-style trajectories.
- Provide inputs for both heliocentric (Keplerian) and vortex parameterizations so that comparisons share identical ephemeris data.
- Identify telemetry needed to determine whether one model offers measurable improvements (e.g., lower residuals between simulation and observation).

## 2. Data and Models

- Source daily heliocentric positions from JPL DE-series ephemerides; convert to geocentric using standard transforms.
- Express the vortex hypothesis as a set of coupled phase oscillators with carrier period $T_c$ (sidereal frame) and modulator period $T_m$ (planet-specific drift).
- Record all parameters (frequencies, amplitudes, phase offsets) in a JSON or YAML artifact so the comparison is reproducible.

## 3. Evaluation Procedure

- For each planet, integrate both models over a multi-year window and compute geocentric trajectories projected on the ecliptic plane.
- Measure deviation from observed right ascension/declination pairs using RMS error and spectral leakage metrics.
- Document whether vortex-specific parameters (e.g., inferred carrier amplitude) remain stable across windows; unstable parameters indicate overfitting.

## 4. Outstanding Work

1. Automate the comparison in a benchmark (preferably under `benchmarks/`) so the pipeline can run in CI.  
2. Publish plots and residual tables in `results/` with metadata describing data sources and processing scripts.  
3. Document open technical questions (e.g., how to constrain modulator amplitudes) so future contributors can extend the study systematically.
