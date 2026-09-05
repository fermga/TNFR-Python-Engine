# The Structural-Field Tetrad as the Canonical Diagnostic Basis

**Status**: Canonical diagnostic read-out; stronger minimal-completeness claims
remain open
**Foundation**: the nodal equation ∂EPI/∂t = νf·ΔNFR(t)
**Prerequisites**: [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md), [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §4

---

## 1. Statement

The state of a TNFR graph is diagnosed through four structural fields — the
**structural-field tetrad**. They cover aggregation, local first and second
phase differences, and non-local correlation. This order-based organization is
derived. Independence, sufficiency for a declared diagnostic target, and
complete arbitrary-state reconstruction require additional hypotheses; see
[MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md).

| Field | Symbol | Tower order | Genuine structural scale |
|-------|--------|-------------|--------------------------|
| Structural potential | Φ_s | 0th — global aggregation | Graph- and pressure-dependent; π/4 and π/2 are selected warning policies |
| Phase gradient | \|∇φ\| | 1st — local derivative | π (phase-wrap bound) |
| Phase curvature | K_φ | 2nd — local curvature | π (exact wrapped bound); L_rw agreement is a scoped linearization |
| Coherence length | ξ_C | non-local — correlation | Spectral gap, ξ_C ∝ 1/√λ₂ |

The one exact phase-sector scale is **π**: both phase derivatives use wrapped
angles, so |∇φ| ≤ π and |K_φ| ≤ π. The π/4 potential magnitude and π/2 drift
values are selected policies rather than consequences of phase wrapping.
The spectral coherence-length estimate scales as ξ_C ∝ 1/√λ₂ under its stated
graph hypotheses. Other parameters must be labelled as derived under explicit
hypotheses or operational.

---

## 2. The one structural scale: π

Within TNFR only **π** carries a genuine structural role — it is the phase-wrap
bound of the phase sector. Both |∇φ| and K_φ are means of wrapped angles, so each
is ≤ π; π is the period of e^{ix} (angular closure), and that geometric closure is
exactly what bounds the two phase derivatives.

φ, γ, e are not additional exact scales of the tetrad fields. The
coherence-length estimate is spectral; the Φ_s thresholds are operational
warning policies.

---

## 3. The field scales

- **π — genuine (geometric, exact).** |∇φ| ≤ π and |K_φ| ≤ π for any
  configuration; |K_φ| < 0.9·π ≈ 2.827 is an operational warning margin. π is the
  one constant that scales the whole phase sector.
- **ξ_C — set by the spectral gap.** The correlation length is set by the Fiedler
  value: ξ_C ∝ 1/√λ₂. The structural content is the spectral gap.
- **Φ_s — graph- and pressure-dependent.** The per-node π/4 and drift π/2
  values are selected warning policies. A general bound must include the graph
  kernel and a pressure bound.
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
