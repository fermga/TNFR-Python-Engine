# Structural Field Hexad: Telemetry Tetrad + Flux Pair

Purpose: Clarify taxonomy and canonical status of the six fields used in TNFR extended dynamics and safety telemetry.

- Telemetry Tetrad (read-only, under U6)
  - Φ_s: Structural potential (global)
  - |∇φ|: Phase gradient (local stress)
  - K_φ: Phase curvature (geometric confinement)
  - ξ_C: Coherence length (spatial correlation / criticality)

- Flux Pair (dynamics variables)
  - J_φ: Phase current (transport in ∂θ/∂t)
  - ∇·J_ΔNFR: Reorganization flux divergence (conservation in ∂ΔNFR/∂t)

Canonical status:
- Tetrad: CANONICAL telemetry under U6; read-only safety checks; no prescriptive constraints.
- Flux Pair: CANONICAL as dynamic variables within the extended 3-equation system; governed by U1–U5; not U6 checks.

No new grammar rules are introduced by adding flux dynamics. U1–U5 remain the prescriptive operator constraints; U6 remains an umbrella for read-only safety telemetry (now routinely including all four telemetry fields).

References:
- `AGENTS.md` — Structural Fields (CANONICAL statuses)
- `UNIFIED_GRAMMAR_RULES.md` — U1–U6 specification
- `docs/grammar/U6_STRUCTURAL_FIELD_TETRAD.md` — safety criteria and integration notes
- `src/tnfr/dynamics/canonical.py` — extended nodal system (compute_extended_nodal_system)
