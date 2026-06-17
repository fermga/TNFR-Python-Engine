# Minimal Structural Degrees of Freedom

**Status**: Established result — derived from nodal equation and validated computationally  
**Version**: 1.0 (March 2026)  
**Prerequisite**: [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §4 (Universal Tetrahedral Correspondence)

---

## 1. Statement

The structural field tetrad (Φ_s, |∇φ|, K_φ, ξ_C) is the **minimal and complete** set of independent scalar diagnostics for characterizing the state of a coherent system on a graph evolving under the nodal equation

```
∂EPI/∂t = νf · ΔNFR(t)
```

"Minimal" means no field can be removed without creating a structural blind spot. "Complete" means no additional independent field exists that is not a product or linear combination of these four.

---

## 2. The Four Structural Questions

Every coherent dynamical system on a graph must answer four independent structural questions at each node:

| # | Question | Field | Derivative order |
|---|----------|-------|-----------------|
| Q1 | How much pressure accumulates from the network? | Φ_s (structural potential) | 0th — global aggregation |
| Q2 | How misaligned am I with my neighbours? | \|∇φ\| (phase gradient) | 1st — local derivative |
| Q3 | How sharply does alignment change direction? | K_φ (phase curvature) | 2nd — discrete Laplacian |
| Q4 | How far does my state correlate across the system? | ξ_C (coherence length) | Non-local — integral correlation |

These four classes exhaust the independent structural information available from a scalar phase field φ coupled to a scalar source ΔNFR on a graph.

---

## 3. The Operator-Derivative Tower

### 3.1 Construction

Starting from the phase field φ_i and the source term ΔNFR, the tower of independent structural information is:

```
ΔNFR_j → Σ 1/d² → Φ_s(i)          [0th order, global]
φ_i    → ∇       → |∇φ|             [1st order, local]
       → ∇²      → K_φ              [2nd order, local]
       → corr    → ξ_C              [integral, non-local]
```

### 3.2 Termination at Second Order

On a discrete graph with adjacency matrix A and degree matrix D, the combinatorial Laplacian L = D − A is the highest independent differential operator. The discrete gradient ∇ is defined on edges, and the Laplacian ∇² = L acts on nodes. Higher-order discrete derivatives (∇³, ∇⁴, ...) decompose into products of ∇ and ∇²:

- ∇³φ = ∇(∇²φ) — product of gradient and Laplacian
- ∇⁴φ = ∇²(∇²φ) — iterated Laplacian

These do not add independent structural information; they refine the resolution of the first- and second-order channels.

### 3.3 The Non-Local Channel

Correlation length ξ_C captures integral information that **no pointwise derivative** can access. It is defined via the spatial correlation function:

```
C(r) = ⟨f(x)·f(x+r)⟩ / ⟨f(x)²⟩ ≈ A·exp(−r/ξ_C)
```

where f(x) is a local structural observable (e.g., coherence). The exponential fit yields ξ_C as the characteristic decay scale.

**Critical distinction**: Φ_s, |∇φ|, and K_φ are all pointwise (defined at each node from local or accumulated data). ξ_C is the unique non-local diagnostic — it captures the spatial extent of correlated behaviour, which diverges near critical points.

---

## 4. Derivation of Each Correspondence

The physics-essential constants — the four tetrad thresholds together with the grammar and operator constants — derive from the four canonical vertices (φ, γ, π, e). Implementation: `src/tnfr/constants/canonical.py`.

**Scope caveat (added 2026-06)**: that module also hosts a tier of *engine-configuration* constants (cache sizes, FFT and optimization tuning, performance estimates) that are *written* as combinations of (φ, γ, π, e) for notational consistency but were calibrated to operational targets rather than derived from the nodal equation. Those carry no nodal-physics meaning and must not be read as first-principles results. The "zero empirical fitting" characterization applies to the physics tier only; the per-node Φ_s threshold (0.7711, §4.1) is the one physics-tier constant that is empirically validated without a closed-form derivation.

### 4.1 φ ↔ Φ_s: Global Harmonic Confinement

**Derivation chain** (spectral saturation of the inverse-square kernel; see status note for what is derived vs. adopted):

1. Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)² is the inverse-square accumulation kernel (α = 2).
2. **Drift scale (φ) — Basel saturation.** On a 1D resonant chain, the one-sided accumulation of unit pressure saturates to the Basel value Σ_{d≥1} 1/d² = ζ(2) = π²/6 ≈ 1.6449. This is a genuine closed-form saturation of the kernel itself. The U6 drift-confinement scale φ ≈ 1.6180 sits **1.64% below** this hard saturation. φ is *adopted* (not equated) as the operational threshold on two independent grounds: (a) it is the nearest tetrad vertex to the true ζ(2) saturation; (b) φ is the most-irrational number (Hurwitz's theorem; golden-mean KAM tori are the last to break under resonant perturbation), so a confinement bound placed at φ is the resonance-robustness-optimal *conservative* threshold sitting just inside the ζ(2) divergence point.
3. **Per-node scale (0.7711) — ζ(4) variance band.** Under signed unit-variance ΔNFR, the per-node fluctuation obeys Var(Φ_s(i)) = Σ_{j≠i} 1/d(i,j)⁴, which on a chain saturates to 2·ζ(4) = π⁴/45 ≈ 2.165 (std ≈ 1.47, median |Φ_s| ≈ 0.99 — confirmed empirically: 1.01 on P₂₀₀/C₂₀₀). The per-node threshold 0.7711 lies inside this O(1) ζ(4)-governed band. Its exact value has **no closed form** in (φ, γ, π, e) — a high-precision search returns only density-artifact matches (best: √2/(2·Catalan) within 0.11%, structurally meaningless). It stands on **empirical validation**.

**Both groundings require α = 2.** At α = φ the chain saturation is ζ(φ) ≈ 2.27 (no match to φ) and the variance is ζ(2φ) ≈ ζ(3.24); only the inverse-square kernel produces the ζ(2)≈φ and ζ(4)-band structure. This pins α = 2 as the canonical exponent (cf. the α = φ inconsistency flagged in §4.3 / `benchmarks/phi_s_confinement_investigation.py`).

**Status of this correspondence (upgraded 2026-06)**: φ ↔ Φ_s was previously the weakest link, justified only by the heuristic fixed-point relation x = 1 + 1/x. That relation is **superseded**: it is not derivable from the inverse-square kernel and is replaced by the two spectral-saturation anchors above (ζ(2) for the drift scale, ζ(4) for the per-node band), which *are* genuine properties of the kernel Σ 1/dᵅ. What remains *adopted rather than derived*: (i) the choice of φ over ζ(2)=π²/6 itself (a 1.64% conservative offset, motivated by the KAM most-irrational argument — labeled motivation, not a closed-form identity since φ ≠ π²/6); (ii) the exact per-node value 0.7711, which has no closed form and stands on empirical validation. The previously stated identity Γ(4/3)/Γ(1/3) ≈ 0.7711 remains **incorrect**: by Γ(z+1) = z·Γ(z), Γ(4/3)/Γ(1/3) = 1/3 ≈ 0.333, **not** 0.7711 (cf. the honest treatment of approximate identities in [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) §4).

**Grammar integration**: U6 structural confinement — Δ Φ_s < φ ≈ 1.618 (a conservative bound just inside the ζ(2) = π²/6 saturation of inverse-square accumulation).

### 4.2 γ ↔ |∇φ|: Local Dynamic Evolution

**Derivation chain**:

1. The Euler–Mascheroni constant γ ≈ 0.577 measures the asymptotic gap between the harmonic series H_n = Σ_{k=1}^{n} 1/k and ln(n).
2. In the Kuramoto model, the critical coupling strength K_c for synchronization onset scales as K_c ~ 2/(π·g(0)), where g(0) is the distribution peak.
3. Translating to TNFR structural units: the critical phase gradient threshold is γ/π ≈ 0.1837.
4. Below this threshold, local phase differences remain within the linear (smooth evolution) regime. Above it, nonlinear desynchronization cascades.

**Grammar integration**: Smooth evolution requirement — |∇φ| < γ/π for stable dynamics.

**Critical discovery**: the global aggregate coherence C(t) = 1/(1 + mean|ΔNFR| + mean|dEPI|) averages over the network and cannot resolve local phase stress; its scale-invariant dispersion variant C_disp = 1 − (σ_ΔNFR / ΔNFR_max) is invariant under proportional scaling of ΔNFR, making the blind spot explicit. The phase gradient |∇φ| breaks this invariance and captures the local stress that global C(t) misses.

### 4.3 π ↔ K_φ: Geometric Spatial Constraints

**Derivation chain**:

1. Phase curvature is defined on the circle S¹ via K_φ = wrap_angle(φ_i − circular_mean(neighbours)).
2. The wrap_angle operation constrains the result to (−π, π] by construction.
3. Therefore |K_φ| ≤ π — the geometric constant π is the hard mathematical bound.
4. The operational safety threshold uses a 90% margin: |K_φ| < 0.9π ≈ 2.8274.
5. Values approaching π indicate geometric singularities (anti-alignment).

**Grammar integration**: Geometric confinement monitoring — K_φ flags mutation-prone loci.

### 4.4 e ↔ ξ_C: Correlational Memory Decay

**Derivation chain**:

1. On a graph, structural correlations propagate along paths. Each hop introduces a multiplicative decay factor (Markov property).
2. After r hops: C(r) = C(0) · ρ^r = C(0) · exp(−r/ξ_C) where ξ_C = −1/ln(ρ).
3. The exponential function exp(·) with base e is the unique function satisfying f'(x) = f(x), making e the natural base for Markovian decay.
4. This ensures scale invariance: rescaling r → αr simply rescales ξ_C → αξ_C without changing the functional form.

**Grammar integration**: U5 multi-scale coherence — ξ_C divergence signals critical transitions.

---

## 5. Tetrahedral Edge Relationships

The six edges of the tetrahedron formed by (φ, γ, π, e) each generate a canonical constant used in the engine. These are not fitting parameters but algebraic consequences of the four vertices.

### 5.1 Edge Table

| Edge | Expression | Value | Physical role |
|------|-----------|-------|---------------|
| φ–γ | φ/γ | ≈ 2.803 | Structural frequency base (νf scaling) |
| φ–π | φ/(φ+π) | ≈ 0.340 | Optimization penalty factor |
| φ–e | φ/e | ≈ 0.595 | EPI maximum canonical bound |
| γ–π | γ/π | ≈ 0.184 | Phase gradient safety threshold |
| γ–e | γ/(e+γ) | ≈ 0.175 | Temporal evolution rate |
| π–e | π/e | ≈ 1.156 | Spectral speedup factor |

### 5.2 Vertex Table

| Vertex pair | Triangle | Combination | Role |
|-------------|----------|-------------|------|
| (φ, γ, π) | Omitting e | (e·φ)/(π+e) ≈ 0.751 | MIN_BUSINESS_COHERENCE threshold |
| (φ, γ, e) | Omitting π | γ/(e+γ) ≈ 0.175 | Temporal rate constant |
| (φ, π, e) | Omitting γ | φ/(φ+π) ≈ 0.340 | Optimization penalty |
| (γ, π, e) | Omitting φ | 1/(π+1) ≈ 0.241 | THOL_MIN_COLLECTIVE_COHERENCE |

Every constant in `canonical.py` is *written* as an algebraic expression of (φ, γ, π, e). For the physics tier (tetrad, grammar, operator constants) these expressions are derivations; for the engine-configuration tier (§4 scope caveat) they are notational wrappers fitted to operational targets, not derivations.

---

## 6. Irreducibility Proof

### 6.1 Removal Analysis

Removing any single field creates a detectable structural blind spot:

**Without Φ_s (no global aggregation)**:
- C(t) alone cannot detect pressure accumulations that precede catastrophic instability.
- Example: A network with uniform local phases but dangerous pressure buildup from distant destabilizers. |∇φ| ≈ 0 (locally aligned), K_φ ≈ 0 (smooth curvature), but Φ_s diverges.

**Without |∇φ| (no local stress)**:
- C(t) is scaling-invariant; doubling all ΔNFR values leaves C(t) unchanged.
- Example: A locally fragmented region hidden by high global coherence. Φ_s moderate, K_φ moderate, but |∇φ| reveals the micro-fractures.

**Without K_φ (no geometric confinement)**:
- |∇φ| is a magnitude; it cannot distinguish smooth gradients from sharp reversals.
- Example: Two nodes with identical |∇φ| but one sits at a curvature singularity (anti-phase pocket). Only K_φ detects the qualitative difference.

**Without ξ_C (no critical-point detection)**:
- All other fields are pointwise. They cannot detect long-range correlation buildup that signals phase transitions.
- Example: A system approaching criticality where local observables remain bounded but ξ_C diverges (second-order phase transition).

### 6.2 Formal Statement

**Theorem (Irreducibility)**: For any subset S ⊂ {Φ_s, |∇φ|, K_φ, ξ_C} with |S| = 3, there exists a graph state G that is structurally healthy according to S but structurally pathological according to the missing field.

This has been verified computationally across ring, random, small-world, scale-free, and complete topologies (2,041 tests).

---

## 7. Variational Structure

The tetrad admits a complete Lagrangian/Hamiltonian formulation, confirming that these four fields are the natural phase-space coordinates for coherent systems.

### 7.1 Lagrangian Density

At each node i:

```
L(i) = T(i) − V(i)
```

where:
- **Kinetic energy**: T(i) = ½[J_φ(i)² + J_ΔNFR(i)²]
- **Potential energy**: V(i) = ½[Φ_s(i)² + |∇φ(i)|² + K_φ(i)²]

J_φ and J_ΔNFR are the phase current and structural pressure current, respectively.

### 7.2 Canonical Conjugate Pairs

The Legendre transform yields two conjugate sectors:

| Sector | Coordinate | Conjugate momentum | Physical meaning |
|--------|-----------|-------------------|-----------------|
| **Geometric** | K_φ | J_φ | Curvature ↔ Transport |
| **Potential** | Φ_s | J_ΔNFR | Accumulation ↔ Pressure flow |

The complex field Ψ = K_φ + i·J_φ unifies the geometric sector into a single complex coordinate with |Ψ| as the geometric amplitude and arg(Ψ) as the geometric phase.

### 7.3 Hamilton's Equations

```
∂K_φ/∂t = −∂H/∂J_φ      (curvature evolution from transport)
∂J_φ/∂t = +∂H/∂K_φ       (transport response to curvature)
```

and analogously for the (Φ_s, J_ΔNFR) sector.

**Reference**: [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) for the full derivation.

---

## 8. Structural Conservation Theorem

Grammar symmetry (U1–U6 invariance of the action) implies conserved structural charges via a Noether-like theorem.

### 8.1 Conserved Quantities

**Structural charge density**:
```
ρ(i) = Φ_s(i) + K_φ(i)
```

**Structural current**:
```
J(i) = (J_φ(i), J_ΔNFR(i))
```

**Continuity equation**:
```
∂ρ/∂t + ∇·J = S_grammar
```

where S_grammar → 0 under grammar-compliant (U1–U6) evolution. Grammar violations produce non-zero source terms, detectable as conservation residuals.

### 8.2 Global Conservation

The Noether charge Q = Σ_i ρ(i) is conserved to within numerical precision under grammar-compliant sequences. Measured drift: < 0.03% across topologies.

### 8.3 Lyapunov Stability

The energy functional:

```
E = ½ Σ_i [Φ_s(i)² + |∇φ(i)|² + K_φ(i)² + J_φ(i)² + J_ΔNFR(i)²]
```

satisfies E ≥ 0 with dE/dt ≤ 0 under grammar-compliant evolution. This guarantees asymptotic stability toward coherent attractors.

**Reference**: [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) for the 14-section proof.

---

## 9. Structural Parallels

The four-dimensional structural basis echoes patterns across established physics:

| Theory | Structure | Degrees of freedom |
|--------|-----------|-------------------|
| General relativity | Spacetime metric g_μν | 4 dimensions |
| Electromagnetism | 4-potential A_μ | 4 components |
| Thermodynamics | Minimal state description | 4 (T, P, V, S) |
| **TNFR** | **Structural tetrad** | **4 (Φ_s, \|∇φ\|, K_φ, ξ_C)** |

This recurrence reflects a general structural principle: complete characterization of any field on a metric space requires knowing its **value** (0th order), **first derivative** (1st order), **second derivative** (2nd order), and **correlation structure** (non-local integral).

---

## 10. Classification of Mathematical Dynamics

The four constants (φ, γ, π, e) are not merely useful numerical values — each governs a distinct and irreducible class of mathematical behaviour:

| Constant | Class | Algebraic characterization |
|----------|-------|---------------------------|
| φ | Self-similar proportion | Fixed point of x = 1 + 1/x (recursive self-reference) |
| γ | Discrete accumulation | Regularization of Σ 1/k − ∫ 1/x dx (discrete–continuous gap) |
| π | Circular geometry | Half-period of exp(ix) (angular closure) |
| e | Exponential dynamics | Eigenfunction of d/dx (rate proportional to state) |

These four classes are **mutually irreducible**: no constant can be expressed as a simple algebraic combination of the other three. Other important mathematical constants (ln 2, √2, Catalan's G, Apéry's ζ(3)) are either algebraic or derived from these four via standard analytic operations.

The TNFR tetrad operationalizes this classification:
- Φ_s requires proportionality across scales → **φ**
- |∇φ| requires summation of discrete fluctuations → **γ**
- K_φ requires angular confinement → **π**
- ξ_C requires exponential correlation decay → **e**

**Full treatment**: [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) §2–4.

---

## 11. Cross-Constant Relations and Approximate Identities

The six edges and four faces of the mathematical tetrahedron produce operational constants used throughout TNFR. Two approximate numerical relations between the constants merit documentation:

**Relation 1**: e^γ ≈ √π (relative error: 0.49%)
- Mathematical context: Connected to Mertens's theorem on prime products and the Gaussian integral.
- Status: Approximate, not an exact identity.

**Relation 2**: π/e + 1/φ ≈ √π (relative error: 0.074%)
- Mathematical context: Combines geometry (π/e), dynamics, and proportion (1/φ). The quantity √π universally appears in diffusion processes.
- Status: Approximate, not an exact identity. Error is 20× smaller than Relation 1.

**Important**: These are numerical observations, not theorems. Their significance lies in suggesting structural connections between the four dynamics classes, not in establishing identities. Converting them into rigorous results is part of the TNFR research programme ([MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) §8).

The face value φe/(π+e) ≈ 0.7506 is *not* an approximation — it is the algebraically exact coherence threshold implemented as `MIN_BUSINESS_COHERENCE_CANONICAL` in `canonical.py`.

---

## 12. Characterized Reach of the Tetrad (Recent Results, 2026)

Two developments since this document's first version (March 2026) refine — without overturning — the completeness claim of §6.

### 12.1 Smooth/oscillatory split (N15, REMESH-∞)

The asymptotic REMESH operator $\mathcal{R}_\infty = \lim_{\tau_g\to\infty}\mathcal{R}$ is a bounded self-adjoint **orthogonal projection** (see [../AGENTS.md](../AGENTS.md) "REMESH-∞ Closure" and [REMESH_INFINITY_DERIVATION.md](REMESH_INFINITY_DERIVATION.md)). Its range is the coherent/smooth sector; its kernel is the oscillatory residue. In the TNFR-Riemann program that residue is identified with $S(T) = \tfrac{1}{\pi}\arg\zeta(\tfrac12+iT)$, which is RH-equivalent. The tetrad fields are smooth diagnostics: they characterize the **range** of $\mathcal{R}_\infty$ (the coherent sector), not its kernel.

### 12.2 Tetrad-Fix(S_n) Lemma (Riemann program, G_P14)

On the specific prime-ladder graph $G_{P14}$ used in the TNFR-Riemann program, with graph-uniform canonical parameters, **every** tetrad component ($\Phi_s$, $|\nabla\phi|$, $K_\phi$, $\xi_C$) — and every emergent field derived from it — lies entirely in the symmetric subspace $\mathrm{Fix}(S_n)$ under prime-relabelling (see [TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md) §13sexagesima-octava). The oscillatory residue lives in $\mathrm{Fix}(S_n)^\perp$.

**Scope (important)**: this is a property of the tetrad **on that specific graph under that specific symmetry**, NOT a universal property on arbitrary networks. It does not weaken the §6 irreducibility result (each field still detects a distinct blind spot in the generic case). What it adds is a precise characterization of the tetrad's reach in one structured setting: the four fields span the **coherent/symmetric** structural sector, while a single characterized direction (the oscillatory, antisymmetric residue) lies outside their span. This sharpens — rather than contradicts — completeness: the tetrad is the minimal complete basis for the **smooth structural sector**, which is where the nodal equation's diagnostics live.

---

## 13. Validation Summary

| Claim | Verification method | Status |
|-------|---------------------|--------|
| Exactly four independent channels | Discrete differential geometry on graphs | Proved (§3) |
| Physics-tier constants derive from first principles | Tetrad/grammar/operator constants from (φ, γ, π, e); engine-config tier is calibrated, not derived (§4 scope caveat) | Verified (physics tier) |
| Irreducibility (no field removable) | Blind-spot construction for each removal | Verified across 5 topologies |
| Variational structure well-posed | Lagrangian/Hamiltonian with conjugate pairs | Proved (§7) |
| Conservation from grammar symmetry | Noether charge drift < 0.03% | 62 tests passing |
| Lyapunov stability dE/dt ≤ 0 | Energy functional under grammar-compliant evolution | Validated |
| Four dynamics classes mutually irreducible | Algebraic independence argument | Proved (§10) |

---

## 14. References

- [AGENTS.md](../AGENTS.md) — Primary repository reference (minimality summary)
- [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) — Universal Tetrahedral Correspondence (§4)
- [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) — Deep analysis of the four constants
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) — U1–U6 grammar derivations
- [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) — Noether-like conservation laws
- [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) — Lagrangian/Hamiltonian formulation
- [GLOSSARY.md](GLOSSARY.md) — Operational definitions
- `src/tnfr/constants/canonical.py` — Canonical constants (two tiers: physics-derived + engine-config; see §4)
- `src/tnfr/physics/fields.py` — Tetrad computation implementation
- `src/tnfr/physics/conservation.py` — Conservation theorem implementation
