# Geocentric Dynamics Memo

**Status**: Technical reference  
**Version**: 0.2.0 (November 30, 2025)  
**Owner**: `theory/GEOCENTRIC_DYNAMICS.md`

---

## 1. Scope

Model apparent planetary motion from Earth using TNFR phase-field representations with reproducible simulation steps and telemetry. This memo specifies inputs, integration workflow, and validation criteria.

---

## 2. Phase-Field Formulation

Parameterize the sky plane in polar coordinates \((r, \theta)\) and define:

1. Background rotation \(\phi_{bg}(\theta, t) = \omega_{bg} t\).  
2. Solar driver \(\phi_{\odot}(t) = \omega_{\odot} t + \phi_0\).  
3. Planetary evolution:
	\[
	\frac{d}{dt} \text{EPI}_p = \nu_f^p \Delta \text{NFR}(\phi_{bg}, \phi_{\odot}).
	\]

Retrograde loops appear when \(\dot{\phi}_{\odot} - \dot{\phi}_{bg}\) changes sign locally.

Telemetry requirements: track \(C(t)\), \(\Phi_s\), \(|\nabla \phi|\), applied operators, and ephemeris references.

---

## 3. Simulation Workflow

1. Load \(\omega_{bg}\) and \(\omega_{\odot}\) from astronomical datasets.  
2. Configure operator sequences satisfying U1–U6; record seeds and schedules.  
3. Integrate planetary nodes across the desired time window.  
4. Produce trajectory plots (e.g., spirograph patterns) and store data under `results/geocentric_dynamics/`.  
5. Document assumptions (projection, coordinate transforms, observation site).

---

## 4. Validation

- Compare simulated retrograde intervals with JPL ephemerides; report phase/timing residuals.  
- Run statistical checks to ensure deviations stay within configured tolerances.  
- Include raw comparison tables (CSV) for peer review.

---

## 5. Outstanding Tasks

1. Automate ephemeris ingestion and comparison in `benchmarks/`.  
2. Add sensitivity analysis for \(\omega_{bg}\) perturbations.  
3. Document limitations (atmospheric refraction, observation geometry) affecting telemetry.
