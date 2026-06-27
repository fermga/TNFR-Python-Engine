# Minimal Structural Degrees of Freedom

**Status**: Established result — derived from nodal equation and validated computationally  
**Version**: 1.0 (March 2026)  
**Prerequisite**: [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) §4 (the structural-field tetrad)

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

## 4. The Field Scales

Each field has a characteristic scale, read directly from the graph and the nodal equation. Only **π** is a genuine structural constant (it bounds the phase sector); the Φ_s bound is π-derived (per-node π/4, drift π/2), ξ_C is set by the spectral gap, and the ≈ 0.18 level for |∇φ| is a heuristic early-warning, not a derived bound. Implementation: `src/tnfr/constants/canonical.py`.

**Scope caveat**: that module also hosts a tier of *engine-configuration* constants (cache sizes, FFT and optimization tuning, performance estimates) calibrated to operational targets rather than derived from the nodal equation. Those carry no nodal-physics meaning and must not be read as first-principles results. Only π is a genuine structural scale; every other parameter is derived from the nodal dynamics / spectral gap or is a free operational parameter.

### 4.1 Φ_s — π-derived confinement

**Bound** (tied to the one structural scale π; the inverse-square kernel sets the O(1) fluctuation band):

1. Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)² is the inverse-square accumulation kernel (α = 2).
2. **Drift bound — π/2 (half phase-wrap).** The confinement bound ΔΦ_s < π/2 ≈ 1.571 ties the drift to the one genuine structural scale, π. It is an O(1) bound, consistent with the kernel's saturation: the one-sided accumulation of unit pressure on a 1D resonant chain saturates to the Basel value Σ_{d≥1} 1/d² = ζ(2) = π²/6 ≈ 1.6449, also O(1).
3. **Per-node bound — π/4 (quarter phase-wrap).** The per-node bound |Φ_s| < π/4 ≈ 0.785 is likewise π-derived. Empirically the per-node fluctuation sits in an O(1) band: under signed unit-variance ΔNFR, Var(Φ_s(i)) = Σ_{j≠i} 1/d(i,j)⁴ saturates on a chain to 2·ζ(4) = π⁴/45 ≈ 2.165 (std ≈ 1.47, median |Φ_s| ≈ 0.99 — confirmed empirically: 1.01 on P₂₀₀/C₂₀₀).

**Both anchors require α = 2.** At α ≠ 2 the chain saturation and variance lose the ζ(2)/ζ(4) structure; only the inverse-square kernel produces the O(1) band. This pins α = 2 as the canonical exponent (cf. `benchmarks/phi_s_confinement_investigation.py`).

**Status (2026)**: the earlier empirical 0.7711 (per-node) and golden-ratio φ ≈ 1.618 (drift) framing is **superseded** by the π-derived bounds π/4 and π/2, tying the confinement bound to the one genuine structural scale. The earlier x = 1 + 1/x fixed-point and Γ(4/3)/Γ(1/3) rationales were already incorrect (Γ(4/3)/Γ(1/3) = 1/3, not 0.7711) and are dropped.

**Grammar integration**: U6 structural confinement — Δ Φ_s < π/2 ≈ 1.571 (half phase-wrap, tied to the one structural scale π).

### 4.2 |∇φ| — phase-wrap bound

**Scale**:

1. |∇φ|(i) is the mean wrapped phase difference to neighbours; being a mean of WRAPPED angles, its genuine bound is **|∇φ| ≤ π** — the same phase-wrap bound as K_φ.
2. The synchronization onset is a measured ≈ 0.29 and σ-dependent (a dynamical transition, not a constant). The level γ/π ≈ 0.1837 is retained only as a heuristic early-warning, **not** a derived threshold.

**Critical discovery**: the global aggregate coherence C(t) = 1/(1 + mean|ΔNFR| + mean|dEPI|) averages over the network and cannot resolve local phase stress; its scale-invariant dispersion variant C_disp = 1 − (σ_ΔNFR / ΔNFR_max) is invariant under proportional scaling of ΔNFR, making the blind spot explicit. The phase gradient |∇φ| breaks this invariance and captures the local stress that global C(t) misses.

### 4.3 K_φ — geometric phase-wrap (π, genuine)

**Derivation chain**:

1. Phase curvature is defined on the circle S¹ via K_φ = wrap_angle(φ_i − circular_mean(neighbours)).
2. The wrap_angle operation constrains the result to (−π, π] by construction.
3. Therefore |K_φ| ≤ π — the geometric constant π is the hard mathematical bound.
4. The operational safety threshold uses a 90% margin: |K_φ| < 0.9π ≈ 2.8274.
5. Values approaching π indicate geometric singularities (anti-alignment).

**Grammar integration**: Geometric confinement monitoring — K_φ flags mutation-prone loci.

### 4.4 ξ_C — spectral gap

**Scale**:

1. On a graph, structural correlations propagate along paths, decaying per hop (Markov property): C(r) = C(0)·exp(−r/ξ_C). Exponential decay has base e tautologically.
2. The genuine structural scale is the **spectral gap** (Fiedler value λ₂): ξ_C ∝ 1/√λ₂. Rescaling r → αr rescales ξ_C → αξ_C without changing the functional form.

**Grammar integration**: U5 multi-scale coherence — ξ_C divergence signals critical transitions.

---

## 5. Operational parameters

Beyond the one structural scale π, the engine uses operational parameters (operator gains, numerical clamps, dt, coupling rates, telemetry thresholds). Except for the π phase-wrap bounds, the π-derived Φ_s bound, and ξ_C ∝ 1/√λ₂, these are free operational values, not derivations from the nodal equation.

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

## 10. The one structural scale

Only **π** is a genuine structural scale in TNFR — the phase-wrap bound of the phase sector (both |∇φ| and K_φ are means of wrapped angles, so each is ≤ π; π is the half-period of exp(ix), the angular closure that bounds them). The four tetrad fields are the four orders of the discrete derivative tower (§3), **not** four constants:

- **Φ_s** (0th order, global aggregation) — π-derived confinement bound (per-node π/4, drift π/2).
- **|∇φ|** (1st order, local) — π phase-wrap bound; the ≈ 0.18 onset level is heuristic.
- **K_φ** (2nd order, local) — π phase-wrap bound; K_φ = L_rw·φ.
- **ξ_C** (non-local correlation) — set by the spectral gap, ξ_C ∝ 1/√λ₂.

φ, γ, e are **not** structural scales and no longer appear in the engine; everything other than π is derived from the nodal dynamics / spectral gap or is a free operational parameter. **Full treatment**: [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md).

---

## 11. Operational parameters and the engine-configuration tier

Beyond the one structural scale π, every numeric parameter in the engine is either derived from the nodal dynamics / spectral gap or is a **free operational parameter**: operator gain magnitudes (the theory fixes each operator's channel and sign via its contract, not its magnitude), numerical clamps, dt, coupling rates, and the engine-configuration tier (cache, FFT, optimization, performance). The operational business-health cut `MIN_BUSINESS_COHERENCE` (≈ 0.75) is one such operational parameter (the canonical strong-coherence gate is the emergent π/(π+1) ≈ 0.7585). The authoritative current values live in [`src/tnfr/constants/canonical.py`](../src/tnfr/constants/canonical.py); only π is a genuine structural scale.

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
| Only π is a genuine structural scale | π phase-wrap (\|∇φ\|, K_φ); ξ_C ∝ 1/√λ₂; π-derived Φ_s bound; other params operational | Verified |
| Irreducibility (no field removable) | Blind-spot construction for each removal | Verified across 5 topologies |
| Variational structure well-posed | Lagrangian/Hamiltonian with conjugate pairs | Proved (§7) |
| Conservation from grammar symmetry | Noether charge drift < 0.03% | 62 tests passing |
| Lyapunov stability dE/dt ≤ 0 | Energy functional under grammar-compliant evolution | Validated |

---

## 14. References

- [AGENTS.md](../AGENTS.md) — Primary repository reference (minimality summary)
- [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md) — the structural-field tetrad (§4)
- [MATHEMATICAL_DYNAMICS_BASIS.md](MATHEMATICAL_DYNAMICS_BASIS.md) — The structural-field tetrad and the one structural scale (π)
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) — U1–U6 grammar derivations
- [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md) — Noether-like conservation laws
- [TNFR_VARIATIONAL_PRINCIPLE.md](TNFR_VARIATIONAL_PRINCIPLE.md) — Lagrangian/Hamiltonian formulation
- [GLOSSARY.md](GLOSSARY.md) — Operational definitions
- `src/tnfr/constants/canonical.py` — Canonical constants (only π is a genuine structural scale; the rest are derived or operational)
- `src/tnfr/physics/fields.py` — Tetrad computation implementation
- `src/tnfr/physics/conservation.py` — Conservation theorem implementation
