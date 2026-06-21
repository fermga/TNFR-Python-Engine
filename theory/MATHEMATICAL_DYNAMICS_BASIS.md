# The Structural-Field Tetrad as the Minimal Basis

**Status**: Canonical — the tetrad of *fields* is the DERIVED minimal basis (the
constants are notational labels, only π is a genuine structural scale)
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
| Structural potential | Φ_s | 0th — global aggregation | Empirical confinement (no closed form) |
| Phase gradient | \|∇φ\| | 1st — local derivative | π (phase-wrap bound) |
| Phase curvature | K_φ | 2nd — discrete Laplacian | π (phase-wrap bound); K_φ = L_rw·φ |
| Coherence length | ξ_C | non-local — correlation | Spectral gap, ξ_C ∝ 1/√λ₂ |

The one genuine structural constant is **π**: both phase derivatives are wrapped
angles, so |∇φ| ≤ π and |K_φ| ≤ π for any configuration. The Φ_s confinement
bound is empirical, and the coherence length is set by the spectral gap λ₂
(ξ_C ∝ 1/√λ₂). The constants φ, γ, e are **notational labels**, not the
structural scales of their fields.

---

## 2. The mathematical character of the notational constants

The engine writes some parameters as combinations of four mathematical constants
(φ, γ, π, e) for notational consistency (an anti-magic-number convention). Each
constant represents a distinct, well-known class of mathematical dynamics. This
is a property of the *constants themselves*; it does **not** make them the
structural scales of the tetrad fields.

| Constant | Class of dynamics | Defining property |
|----------|-------------------|-------------------|
| φ (golden ratio) | Self-similar proportion | Fixed point of x = 1 + 1/x; the "most irrational" number |
| γ (Euler–Mascheroni) | Discrete accumulation | Gap between Σ 1/k and ∫ dx/x |
| π (Archimedes) | Circular geometry | Period of e^{ix}; **the phase-wrap scale** |
| e (Napier) | Exponential growth/decay | Eigenfunction of d/dx (rate = state) |

As *mathematical* objects these four classes are mutually irreducible. But within
TNFR only **π** carries a genuine structural role (the phase-wrap bound of the
phase sector); the combinations built from φ, γ, e are heuristic or empirical
telemetry values, not derivations from the nodal equation.

---

## 3. Genuine vs notational scales

- **π — genuine (geometric, exact).** |∇φ| ≤ π and |K_φ| ≤ π for any
  configuration; |K_φ| < 0.9·π ≈ 2.827 is the operational safety bound. π is the
  one constant that scales the whole phase sector.
- **ξ_C — genuine (spectral gap).** The correlation length is set by the Fiedler
  value: ξ_C ∝ 1/√λ₂. Exponential decay has base e tautologically; the structural
  content is the spectral gap, not e.
- **Φ_s — empirical.** ΔΦ_s < 1.618 and per-node |Φ_s| < 0.7711 are validated
  across topologies, with no closed form in (φ, γ, π, e).
- **|∇φ| onset — heuristic.** The synchronization onset is a measured ≈ 0.29 and
  σ-dependent; the value γ/π ≈ 0.1837 is retained only as a heuristic
  early-warning level, not a derived bound.

The field computations themselves (`compute_structural_potential`,
`compute_phase_gradient`, `compute_phase_curvature`, `estimate_coherence_length`)
read these scales directly from the graph and the nodal equation — they do not
use the φ/γ/e combinations. Those combinations appear only as downstream
telemetry thresholds.

---

## 4. References

- Minimality of the tetrad: [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md)
- Field definitions and scales: [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §4, [STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)
- Implementation: `src/tnfr/physics/fields.py`, `src/tnfr/physics/canonical.py`, `src/tnfr/constants/canonical.py`
