# Augmented Nodal Equation (Unified Form)

Status: Canonical formulation (derives from TNFR nodal equation + transport/conservation)
Last Updated: 2025-11-12

---

## Purpose

Provide a single, unified vector formulation for the extended TNFR dynamics that couples the classical nodal equation with phase transport and ΔNFR conservation. This settles whether we “need three equations” or can unify them: the minimal coherent unification is a vector-valued nodal equation on an augmented state.

---

## Unified Vector Form

Let the augmented state be S(t) = [EPI(t), θ(t), ΔNFR(t)]^T. Then

∂S/∂t = 𝓛_TNFR[S; G] =
[
  νf · ΔNFR,
  f_phase(νf, ΔNFR, J_φ; κ),
  f_conserve(∇·J_ΔNFR)
]^T

with the canonical components:
- Classical nodal (unchanged): ∂EPI/∂t = νf · ΔNFR
- Phase transport: ∂θ/∂t = α·νf·sin(π·ΔNFR) + β·ΔNFR + γ·κ·J_φ
- ΔNFR conservation: ∂ΔNFR/∂t = -∇·J_ΔNFR - λ·|∇·J_ΔNFR|·sign(∇·J_ΔNFR)

Here κ is local coupling strength; J_φ and ∇·J_ΔNFR are computed by centralized physics routines (compute_phase_current, compute_dnfr_flux and divergence operators).

---

## Why Vector (not Single Scalar) is Minimal

- Physical dimensions differ (EPI, θ, ΔNFR). Collapsing into a single scalar would either:
  - destroy unit consistency, or
  - hide essential structure in ad-hoc embeddings.
- Causality is triangular:
  - ΔNFR drives EPI (primary nodal equation)
  - θ evolves from νf, ΔNFR, and J_φ (transport)
  - ΔNFR evolves by flux conservation (∇·J)
  This structure is lost in a single-scalar collapse.
- Invariants mapping:
  - U1–U5 remain prescriptive operator constraints
  - U6 remains read-only telemetry (Φ_s, |∇φ|, K_φ, ξ_C)
  A vector form preserves a clean separation between prescriptive rules and telemetry.

Conclusion: The unified vector equation is the minimal faithful representation. The classical nodal equation remains the first component and the source of canonicity.

---

## Operator and Grammar Compatibility

- Operators map unchanged; no new operators required.
- Grammar:
  - U1–U5: prescriptive (sequences, stabilization, phase checks, hierarchy)
  - U6: read-only safety suite (Φ_s, |∇φ|, K_φ, ξ_C) — complements, does not constrain
- Flux variables (J_φ, ∇·J_ΔNFR) emerge from compositions (UM, RA, OZ, VAL, IL) and are parameterized measurements, not new operator primitives.

---

## Code Reference

- compute_extended_nodal_system in `src/tnfr/dynamics/canonical.py` implements the three components coherently for each node.
- Integrators call centralized field computations from `src/tnfr/physics/extended_canonical_fields.py`.

---

## Tests

See `tests/dynamics/test_extended_nodal_system.py` for unit tests that validate:
- Classical limit (∂EPI/∂t = νf·ΔNFR; ∂ΔNFR/∂t = 0 when ∇·J = 0)
- Monotonicity of ∂θ/∂t with J_φ (for κ > 0)
- Sign convention in ΔNFR conservation (± divergence)

---

## Summary

- We can and should unify via a vector-valued nodal equation on S = [EPI, θ, ΔNFR].
- The scalar nodal equation remains fundamental; transport and conservation are auxiliary but canonical.
- No new grammar rules (U7/U8) are required.
