# The Structural-Field Tetrad as the Minimal Basis

**Status**: Canonical — the tetrad of *fields* is the DERIVED minimal basis; only
π is a genuine structural scale
**Foundation**: the nodal equation ∂EPI/∂t = νf·ΔNFR(t)
**Prerequisites**: [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md), [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §4

---

## 1. Statement

The state of any coherent system on a graph is read through four structural
fields — the **structural-field tetrad**. These four fields are the four orders
of the discrete derivative tower over the graph, and they are the *minimal
complete* structural read-out of a scalar phase field coupled to a scalar source.
The minimality is **derived** (see [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md)).

| Field | Symbol | Tower order | Genuine structural scale |
|-------|--------|-------------|--------------------------|
| Structural potential | Φ_s | 0th — global aggregation | π-derived confinement (π/4 per-node, π/2 drift) |
| Phase gradient | \|∇φ\| | 1st — local derivative | π (phase-wrap bound) |
| Phase curvature | K_φ | 2nd — discrete Laplacian | π (phase-wrap bound); K_φ = L_rw·φ |
| Coherence length | ξ_C | non-local — correlation | Spectral gap, ξ_C ∝ 1/√λ₂ |

The one genuine structural constant is **π**: both phase derivatives are wrapped
angles, so |∇φ| ≤ π and |K_φ| ≤ π for any configuration. The Φ_s confinement
bound is π-derived (per-node π/4 ≈ 0.785, drift π/2 ≈ 1.571 — quarter / half
phase-wrap), and the coherence length is set by the spectral gap λ₂
(ξ_C ∝ 1/√λ₂). φ, γ, e are **not** structural scales and no longer appear in the
engine; everything other than π is derived from the nodal dynamics or is a free
operational parameter.

---

## 2. The one structural scale: π

Within TNFR only **π** carries a genuine structural role — it is the phase-wrap
bound of the phase sector. Both |∇φ| and K_φ are means of wrapped angles, so each
is ≤ π; π is the period of e^{ix} (angular closure), and that geometric closure is
exactly what bounds the two phase derivatives.

φ, γ, e are **not** structural scales of the tetrad fields and no longer appear in
the engine: every parameter other than π is either derived from the nodal dynamics
/ spectral gap or is a free operational parameter. In particular the coherence
length is set by the spectral gap (ξ_C ∝ 1/√λ₂) and the Φ_s confinement bound is
π-derived (quarter / half phase-wrap).

---

## 3. The field scales

- **π — genuine (geometric, exact).** |∇φ| ≤ π and |K_φ| ≤ π for any
  configuration; |K_φ| < 0.9·π ≈ 2.827 is the operational safety bound. π is the
  one constant that scales the whole phase sector.
- **ξ_C — set by the spectral gap.** The correlation length is set by the Fiedler
  value: ξ_C ∝ 1/√λ₂. The structural content is the spectral gap.
- **Φ_s — π-derived.** The confinement bounds are tied to the π phase scale:
  per-node |Φ_s| < π/4 ≈ 0.785 (quarter phase-wrap) and drift ΔΦ_s < π/2 ≈ 1.571
  (half phase-wrap).
- **|∇φ| onset — heuristic.** The synchronization onset is a measured ≈ 0.29 and
  σ-dependent; a fixed ≈ 0.18 level is retained only as a heuristic early-warning
  level, not a derived bound.

The field computations themselves (`compute_structural_potential`,
`compute_phase_gradient`, `compute_phase_curvature`, `estimate_coherence_length`)
read these scales directly from the graph and the nodal equation.

---

## 4. References

- Minimality of the tetrad: [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md)
- Field definitions and scales: [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §4, [STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)
- Implementation: `src/tnfr/physics/fields.py`, `src/tnfr/physics/canonical.py`, `src/tnfr/constants/canonical.py`
