# U6+: Structural Field Suite — Tetrad (telemetry) + Flux (dynamics)

Status: ✅ CANONICAL (Tetrad: read-only telemetry) • ✅ CANONICAL (Flux: dynamics variables)
Promoted: 2025-11-12
Scope: Extends U6 (Structural Potential Confinement) into a multi-field safety suite and clarifies the Hexad taxonomy

---

## Summary

We formalize the Structural Field Hexad as two complementary groups with distinct roles:

- Telemetry Tetrad (read-only, U6 umbrella):
  - Φ_s: Global structural potential (field-theoretic aggregate of ΔNFR)
  - |∇φ|: Local phase gradient (desynchronization/stress)
  - K_φ: Phase curvature (geometric confinement/torsion)
  - ξ_C: Coherence length (spatial correlation scale, criticality)

- Flux Pair (dynamics variables, used by the extended system):
  - J_φ: Phase current (transport in ∂θ/∂t)
  - ∇·J_ΔNFR: Reorganization flux divergence (conservation in ∂ΔNFR/∂t)

Conclusion: No new grammar rules (U7/U8) are required. U1–U5 remain the only prescriptive constraints; U6 is the umbrella for read-only structural-field telemetry (the tetrad). The flux pair is canonical in dynamics but not a U6 telemetry check.


---

## Why no U7/U8?

- Extended dynamics adds transport and conservation (J_φ, ∇·J_ΔNFR), but does not alter the fundamental nodal equation: ∂EPI/∂t = νf·ΔNFR(t).
- Operator-level preconditions are already covered:
  - U3 (Resonant coupling): Enforces phase verification for any coupling/resonance that generates J_φ
  - U2/U4 (Boundedness/Bifurcation): Contain gradients that drive fluxes (J_ΔNFR), ensuring integral convergence and controlled transitions
  - U5 (Multi-scale coherence): Stabilizes nested structures where field stresses can amplify across scales
- Therefore, no new prescriptive constraints emerge from flux fields; only additional telemetry is warranted.

---

## Canonical Structural Fields and Safety Criteria (U6 telemetry)

All four telemetry fields are CANONICAL and should be exported for every run. They act as read-only safety signals, analogous to U6’s Φ_s criterion.

1) Structural Potential (Φ_s) — Global potential
- Definition: Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)^α  (α=2)
- Evidence: corr(ΔΦ_s, ΔC) = −0.822; R² ≈ 0.68; CV ≈ 0.1% across 5 topologies
- Safety: ΔΦ_s < 2.0 (escape threshold)
- Role: Global confinement (already specified by U6)

2) Phase Gradient (|∇φ|) — Local stress
- Definition: |∇φ|(i) = mean_{j∈N(i)} |θ_i − θ_j|
- Evidence: corr(Δ|∇φ|, Δmax_ΔNFR) = +0.6554; superior predictor of peak stress
- Safety: |∇φ| < 0.38 for stable operation
- Role: Early warning for desynchronization-induced fragmentation

3) Phase Curvature (K_φ) — Geometric confinement
- Definition: K_φ(i) = φ_i − (1/deg(i)) Σ_{j∈N(i)} φ_j
- Evidence: New threshold |K_φ| ≥ 3.0 flags confinement/fault zones with 100% accuracy in aggressive tests; multiscale decay var(K_φ) ~ 1/r^α with α ≈ 2.76
- Safety: Local |K_φ| < 3.0; multiscale safety via k_phi_multiscale_safety(G, α_hint≈2.76)
- Role: Identifies torsion pockets and mutation-prone loci invisible to Φ_s or |∇φ|

4) Coherence Length (ξ_C) — Spatial correlation scale
- Definition: Extracted from spatial autocorrelation c_i = 1/(1+|ΔNFR_i|) with C(r) ~ exp(−r/ξ_C)
- Evidence: 1,170 measurements; critical point I_c ≈ 2.015; ξ_C diverges at criticality; topology-dependent critical exponents ν ≈ 0.61–0.95
- Safety: Monitor regimes — ξ_C > system_diameter (critical), ξ_C > 3×mean_distance (watch), ξ_C < mean_distance (stable)
- Role: Detects phase transitions and long-range reorganization onset

---

## Integration with Grammar

- Prescriptive rules (U1–U5): Unchanged
  - U1: Initiation & closure
  - U2: Convergence & boundedness
  - U3: Resonant coupling (phase verification)
  - U4: Bifurcation dynamics (handlers and context)
  - U5: Multi-scale coherence

- Descriptive telemetry (U6 umbrella):
  - U6a: Φ_s confinement (ΔΦ_s < 2.0)
  - U6b: |∇φ| desynchronization safety (|∇φ| < 0.38)
  - U6c: K_φ curvature safety (|K_φ| < 3.0; optional multiscale check)
  - U6d: ξ_C criticality monitoring (thresholds by regime)

This keeps the grammar minimal and physics-first: one family (U6x) of read-only safety checks instead of new prescriptive rules.

---

## Implementation Notes

- Compute telemetry fields from tnfr.physics.fields:
  - `compute_structural_potential(G)`
  - `compute_phase_gradient(G)`
  - `compute_phase_curvature(G)`, `k_phi_multiscale_safety(G)`
  - `estimate_coherence_length(G)`
- Export all four in telemetry alongside C(t), Si, νf, ΔNFR, and phase
- Validation should warn on threshold crossings; prescriptive failures still originate from U1–U5

- Flux fields (dynamics) are computed via the centralized APIs used by integrators:
  - compute_phase_current(G) → J_φ (transport term in ∂θ/∂t)
  - compute_dnfr_flux(G) and divergence operators → ∇·J_ΔNFR (conservation in ∂ΔNFR/∂t)

---

## Relation to Extended Dynamics (J_φ, ∇·J_ΔNFR)

- J_φ emerges from coupling/resonance (U3) and modulates ∂θ/∂t via transport; its risks are captured by |∇φ| and K_φ telemetry.
- ∇·J_ΔNFR drives ∂ΔNFR/∂t via conservation; risks are bounded by U2 (stabilizers) and surfaced by Φ_s and ξ_C telemetry.
- Therefore, extended dynamics amplifies the value of field telemetry but does not necessitate new operator constraints.

---

## Cross-reference

- Canonical invariants and grammar: `UNIFIED_GRAMMAR_RULES.md`
- Canonical status and thresholds: `AGENTS.md` (Structural Fields section)
- Extended dynamics equations: `src/tnfr/dynamics/canonical.py` (compute_extended_nodal_system)

---

## Try it: script + notebook

Two runnable artifacts demonstrate the Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) and the U6 sequential ΔΦ_s validator:

- Script: `examples/particle_atlas.py`
  - Generates an HTML atlas at `examples/output/particle_atlas.html`.
  - Computes and displays the four telemetry fields and summarizes PASS/WARN status using centralized thresholds from `tnfr.telemetry.constants`.
  - Runs a short sequential step and prints the canonical U6 ΔΦ_s check (PASS/FAIL) using the validator.

- Script: `examples/periodic_table_atlas.py`
  - Early demo that iterates Z=1..10 using radial element-like graphs and writes `examples/output/periodic_table_atlas.html`.
  - Summarizes ξ_C, mean|∇φ|, mean|K_φ|, locality fraction, and ΔΦ_s PASS/FAIL per element (synthetic [AL, RA, IL]-like step).

- Script: `examples/atom_atlas.py`
  - Hydrogen-like radial topology (nucleus + ring) with the same Tetrad telemetry and ΔΦ_s validation.
  - Writes `examples/output/atom_atlas.html` for a compact, single-element view.

- Notebook: `notebooks/TNFR_Particle_Atlas_U6_Sequential.ipynb`
  - Interactive version of the atlas with the same telemetry and thresholds.
  - Includes a U6 sequential ΔΦ_s validation cell and an HTML parity-export cell that mirrors the script output.
  - Adds a mini “Flux Pair” demo (read-only proxies for J_φ and ∇·J_ΔNFR) with pre/post telemetry snapshots.

Notes
- Both paths use shared helpers in `src/tnfr/examples_utils` (seeded WS builder and a minimal synthetic [AL, RA, IL]-like activation step) to ensure reproducibility and parity.
- Centralized thresholds/messages live in `src/tnfr/telemetry/constants.py` and are reflected uniformly in both the script and the notebook.
- Focused tests cover ΔΦ_s PASS/FAIL and telemetry warnings: see `tests/examples/test_u6_sequential_demo.py` and `tests/unit/operators/test_telemetry_warnings_extended.py`.

### Run it from VS Code tasks

Use the pre-wired tasks (Command Palette → “Run Task”) to execute scripts or export notebooks:

- Run Atom Atlas (script)
- Run Periodic Table Atlas (script)
- Export Particle Atlas U6 Sequential (HTML - classic)
- Export Periodic Table Atlas (HTML - classic)

Outputs:
- Script HTML/CSV/JSONL under `examples/output/`
- Notebook HTML under `results/reports/`

See also: Periodic Table Atlas walkthrough in `docs/examples/PERIODIC_TABLE_ATLAS.md`.

---

## Bottom Line

- No U7/U8 needed. Extended dynamics is fully governed by U1–U5; safety remains under U6.
- Promote routine monitoring of the Structural Field Tetrad to achieve early warning and richer diagnostics without altering the canonical operator grammar.

---

## Implications for Emergent "Fundamental Particles"

In TNFR, particle-like entities are structural quanta: localized, persistent coherence sustained by resonant coupling and bounded reorganization. The unified 3‑equation system does not change the emergence pathway; it clarifies it and adds richer telemetry.

A locus behaves as a structural quantum when, in sustained operation:

- Boundedness: ∫ νf·ΔNFR dτ converges and ΔNFR → 0 (U2)
- Phase confinement: |∇φ| is small, implying vanishing phase current and dθ/dt ≈ 0 (U3)
- Flux equilibrium: ∇·J_ΔNFR ≈ 0 with relaxation, so dΔNFR/dt ≈ 0
- Field criteria (read‑only): low ΔΦ_s; |∇φ| < 0.38; |K_φ| < 3.0 (multiscale‑safe); ξ_C below alert/critical regimes

Birth/sustain/decay sequences are compositions of the canonical 13 operators (e.g., [AL, UM/RA, IL] for birth; IL/THOL for sustain; OZ/VAL without stabilizers for decay). No new operators are introduced by the unified system.

For a complete treatment, see `docs/PARTICLE_EMERGENCE_WITH_UNIFIED_SYSTEM.md`.
