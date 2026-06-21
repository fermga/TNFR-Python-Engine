# TNFR: Resonant Fractal Nature Theory

**Theoretical framework and engine for coherent pattern analysis on graph-coupled networks.**

- **Version**: 0.0.3.4
- **Status**: Canonical synthesized reference for TNFR agent guidance
- **Repository**: https://github.com/fermga/TNFR-Python-Engine · **PyPI**: `pip install tnfr`

This document is the **synthesized source of truth** for working on TNFR. It states
the theory as a complete, self-contained whole — not as a changelog. Program
histories, derivations, and per-example detail live in linked documents under
[theory/](theory/README.md), [docs/](docs/), and [examples/](examples/README.md);
this file keeps only the canon an agent needs to reason and act correctly.

---

## 1. What TNFR is

TNFR models **coherent dynamic patterns that persist through resonance**, rather
than discrete objects. A pattern (a vortex, a neural assembly, a decision) is a
configuration maintained by resonant coupling with its environment; it dissolves
when that coupling fails. The mindset:

- Model **coherence**, not objects · capture **process**, not state
- Measure **resonance**, not properties · think **structure**, not substance
- Embrace **emergence**, not reduction

Everything reduces to one evolution law, the **nodal equation**:

$$\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t)$$

### Source hierarchy (authority)

1. **TNFR nodal/structural physics** — canonicity is a physical/mathematical
   property, derivable from the nodal equation and the canonical invariants. This
   is the ultimate authority.
2. **The repository** ([TNFR-Python-Engine](https://github.com/fermga/TNFR-Python-Engine)) —
   the primary canonical implementation.
3. **This document** — the synthesized working reference, mirrored verbatim at
   [.github/agents/my-agent.md](.github/agents/my-agent.md).
4. **PyPI package** — stable releases; may lag the repository.

This file is updated **only when genuinely novel or important TNFR canonicity
emerges**, and is always written as a complete, closed synthesis — never as an
incremental session log.

### Communication policy

1. **English only** for all code, docs, comments, commits, issues, and PRs.
   Non-English text is allowed only in verbatim quotations or raw data.
2. **Anchor every claim to math or telemetry** — the nodal equation, an operator
   contract, or a recorded metric. No qualitative claim without data.
3. **No metaphysical extrapolation** — describe engineering and mathematical
   results, not cosmological, philosophical, or consciousness conclusions.
4. **Academic tone** — precise, testable, with documented scope, seeds, and
   operator sequences so any state is reproducible.

---

## 2. Foundations

### The nodal equation

$$\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t)$$

| Symbol | Name | Meaning | Units |
|--------|------|---------|-------|
| **EPI** | Primary Information Structure | Coherent structural form (configuration) | — |
| **νf** | Structural frequency | Reorganization capacity / rate | Hz_str |
| **ΔNFR** | Nodal gradient | Structural reorganization pressure | — |

Read it as **structural change rate = reorganization capacity × reorganization
pressure**. Limiting states: `νf = 0` (node inactive, cannot reorganize);
`ΔNFR = 0` (equilibrium, no driving force); both non-zero → active reorganization.

### Structural triad

Every node carries three attributes:

- **Form (EPI)** — coherent configuration in a structural manifold; changes only
  via canonical operators; supports nesting (operational fractality).
- **Frequency (νf)** — reorganization rate in Hz_str (ℝ⁺); `νf → 0` deactivates.
- **Phase (φ or θ)** — synchronization parameter in [0, 2π); coupling is admissible
  only under the resonance condition `|φᵢ − φⱼ| ≤ Δφ_max`.

### Bounded evolution → the convergence requirement

Integrating the nodal equation, coherence is preserved only when

$$\int_{t_0}^{t_f} \nu_f(\tau)\,\Delta\mathrm{NFR}(\tau)\,d\tau < \infty.$$

Without stabilizers, `ΔNFR` grows by positive feedback, the integral diverges, and
the pattern fragments. This integral-convergence fact is the physical basis of
grammar rule **U2**.

### Transport content (structural diffusion)

The canonical `ΔNFR` aggregates four structural gradient channels,
`ΔNFR = w_phase·∂φ + w_epi·∂EPI + w_vf·∂νf + w_topo·∂topo` (weights and defaults in
[src/tnfr/dynamics/dnfr.py](src/tnfr/dynamics/dnfr.py)). The **EPI channel** is
**exactly a graph diffusion**. For the EPI channel,

$$\Delta\mathrm{NFR}_{\text{epi}}(i) = \overline{\mathrm{EPI}}_{\mathcal{N}(i)} - \mathrm{EPI}(i) = -(L_{\mathrm{rw}}\,\mathrm{EPI})(i),\qquad L_{\mathrm{rw}} = I - D^{-1}W,$$

so `∂EPI/∂t = −νf · L_rw · EPI` is the discrete diffusion equation with diffusivity
`νf`. Consequences (all TNFR-internal, empirically anchored): structural diffusion
to a uniform field with eigenmode decay `e^{−νf λ_k t}`; conserved degree-weighted
total; equilibrium ⟺ uniform field; the spectral gap `λ₂` (Fiedler value) sets the
slowest relaxation, the synchronization tendency, and — via `r_c = νf·λ₂` — the
spectral form of U2. See [src/tnfr/physics/structural_diffusion.py](src/tnfr/physics/structural_diffusion.py).

---

## 3. The structural tetrad

Four structural fields characterize any coherent system on a graph — the four
orders of the discrete derivative tower (minimality is DERIVED). They are the
canonical state read-out of a network; their characteristic scales are given
below. The one genuine structural constant is **π**, which scales the phase
sector (it bounds both `|∇φ|` and `K_φ`).

| Structural field | Symbol | Tower order | Role |
|------------------|--------|-------------|------|
| Structural potential | Φ_s | 0th (aggregation) | Global stability |
| Phase gradient | \|∇φ\| | 1st (local) | Local desynchronization stress |
| Phase curvature | K_φ | 2nd (local) | Geometric torsion |
| Coherence length | ξ_C | non-local | Correlation range |

### Field scales

The four-field basis is the minimal derivative tower. Each field has a
characteristic scale:

- **π — phase scale (geometric, exact).** Both phase derivatives are **wrapped
  angles**, so `|∇φ| ≤ π` *and* `|K_φ| ≤ π` for any configuration — π scales the
  whole **phase sector**, not K_φ alone. `|K_φ| < 0.9·π ≈ 2.827` sits at this wrap
  limit (exact, parameter-free).
- **|∇φ| — phase-wrap bounded.** Its genuine bound is `|∇φ| ≤ π`. There is no fixed
  structural constant for the synchronization onset: the measured value is `≈ 0.29`
  and σ-dependent (a dynamical transition, not a derived threshold).
- **ξ_C — spectral gap.** The correlation length is set by the **spectral gap**:
  `ξ_C ∝ 1/√λ₂` (verified).
- **Φ_s — empirical confinement.** `Δ Φ_s < 1.618`, per-node `|Φ_s| < 0.7711`, an
  empirically validated source-distribution bound with **no closed form**.

**Structure (verified).** `K_φ` **is** the central operator applied to phase
(`K_φ = L_rw·φ` in the smooth limit, corr = 1.000) — the phase image of the one operator that
generates geometry/diffusion/modes; `ξ_C ∝ 1/√λ₂`. The organizing axis is **local** phase
derivatives (`|∇φ|`, `K_φ`; π-bounded) vs **non-local** source/correlation (`Φ_s`, `ξ_C`), across
the derivative orders. The tetrad is a real minimal basis; among the constants, only **π** is a
genuine structural scale.

### Why exactly four (minimality)

The tetrad is the **minimal and complete** structural basis. A scalar phase field
coupled to a scalar source on a graph admits exactly four independent structural
channels — the orders of the discrete derivative tower:

```
ΔNFR_j → Σ 1/d²  → Φ_s   (0th order, global aggregation)
φ_i    → ∇       → |∇φ|   (1st order, local)
       → ∇²      → K_φ    (2nd order, local; graph Laplacian is the top operator)
       → corr    → ξ_C    (non-local correlation range)
```

Higher graph derivatives decompose into products of lower ones, so no fifth
independent channel exists; removing any field creates a structural blind spot.
Full treatment: [theory/MINIMAL_STRUCTURAL_DEGREES.md](theory/MINIMAL_STRUCTURAL_DEGREES.md),
[theory/FUNDAMENTAL_THEORY.md](theory/FUNDAMENTAL_THEORY.md),
[docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md). All four fields
are CANONICAL; compute them via [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py).

---

## 4. Emergent geometry

The conservation laws reveal that the nodal dynamics carries its **own intrinsic
geometry** — emergent, not imposed. This section synthesizes it; the derivations
and verifications live in the linked modules and theory notes.

### Emergent symplectic substrate

The dynamics generates a symplectic phase space `P = ℝ^{4N}` with two canonical
conjugate pairs per node:

- **Geometric sector** `(K_φ, J_φ)` — curvature ↔ phase current
- **Potential sector** `(Φ_s, J_ΔNFR)` — potential ↔ ΔNFR flux

with canonical brackets `{K_φ, J_φ} = {Φ_s, J_ΔNFR} = 1`. The Hamiltonian is the
structural **energy functional** `H_sub = ½Σ(K_φ² + J_φ² + Φ_s² + J_ΔNFR²)` (plus
the `½Σ|∇φ|²` background). The flow is a **symplectomorphism** (Liouville: phase
volume preserved), so the 13 operators are canonical, volume-preserving transforms.
**Noether** ties each continuous symmetry to a conserved charge: time translation →
`H_sub`; the geometric U(1) (`Ψ → e^{iα}Ψ`) → `E_geo = ½Σ|Ψ|²`; the potential U(1)
→ `E_pot`. The complex coordinate `Ψ = K_φ + i·J_φ` is the geometric sector under
the substrate's complex structure (flat Kähler). The substrate further carries a
**U(2) polarization symmetry** with conserved Stokes parameters on a per-node
**Poincaré sphere** — this is classical wave polarization (Stokes/Poincaré), a
product (un-entangled) classical texture, **not** a quantum state.

The nodal equation is the **overdamped projection** of this Hamiltonian flow.
Implementation and certificates: [src/tnfr/physics/symplectic_substrate.py](src/tnfr/physics/symplectic_substrate.py);
gauge / U(2) structure: [theory/GAUGE_SYMMETRY_AND_UNIFICATION.md](theory/GAUGE_SYMMETRY_AND_UNIFICATION.md).

### Structural conservation theorem (Noether-like)

Grammar symmetry (U1–U6) implies a structural conservation law:

$$\frac{\partial \rho}{\partial t} + \nabla\cdot\mathbf{J} = S_{\text{grammar}},\qquad \rho = \Phi_s + K_\phi,\quad \mathbf{J} = (J_\phi, J_{\Delta\mathrm{NFR}}),$$

with `S_grammar → 0` under U1–U6. The energy functional
`E = ½Σ(Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²) ≥ 0` is a **Lyapunov candidate**:
`dE/dt ≤ 0` is observed under grammar-compliant evolution (proof sketch; a complete
proof of asymptotic stability is open). The six downstream emergent fields
(`χ, 𝒮, 𝒞, ℰ, 𝒜, 𝒬`) are bilinear contractions of the singlets (`Φ_s`, `|∇φ|`) and
the complex fields `Ψ = K_φ + i·J_φ` and `Ω = |∇φ| + i·J_ΔNFR` (e.g. chirality
`χ = Re(Ψ·Ω)`). Conservation theorem:
[src/tnfr/physics/conservation.py](src/tnfr/physics/conservation.py),
[theory/STRUCTURAL_CONSERVATION_THEOREM.md](theory/STRUCTURAL_CONSERVATION_THEOREM.md);
emergent fields: [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py),
[theory/EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](theory/EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md).

### Regime correspondences

The single nodal dynamics produces two empirically-anchored regimes (external labels
"classical"/"quantum-like" are comparisons only, not TNFR primitives):

- **Smooth-trajectory / overdamped drift** (high coherence): first order in time,
  `q̇ = νf·F` — drift velocity ∝ force, `νf` is **mobility** (Stokes/Einstein), not
  inverse mass. The inertial (second-order) regime lives in the conservative
  substrate flow.
- **Discrete-mode** (high dissonance): on a bounded graph the diffusion operator has
  a discrete spectrum of orthonormal standing-wave eigenmodes (vibrating-string /
  Chladni analogue), with nodal-domain ordering (Courant).

See [src/tnfr/physics/structural_diffusion.py](src/tnfr/physics/structural_diffusion.py)
and [examples/02_physics_regimes/](examples/README.md).

---

## 5. The 13 canonical operators

Operators are the **exclusive mechanism** for modifying a node. Each is a resonant
transformation with a defined physical contract; no code may mutate EPI directly.

| # | Operator (glyph) | Physics / effect | Grammar role | Contract |
|---|------------------|------------------|--------------|----------|
| 1 | **Emission** (AL) | Creates EPI from vacuum; `∂EPI/∂t > 0`, raises νf | Generator (U1a) | Sources new form |
| 2 | **Reception** (EN) | Integrates incoming resonance | — | Must not reduce C(t) |
| 3 | **Coherence** (IL) | Negative feedback; reduces \|ΔNFR\|, raises C(t) | Stabilizer (U2) | Must not reduce C(t) (outside dissonance test) |
| 4 | **Dissonance** (OZ) | Controlled instability; raises \|ΔNFR\| | Destabilizer (U2), bifurcation trigger (U4a), closure (U1b) | Must increase \|ΔNFR\| |
| 5 | **Coupling** (UM) | Phase synchronization link `φᵢ → φⱼ` | Requires phase check (U3) | Valid only if `\|φᵢ − φⱼ\| ≤ Δφ_max` |
| 6 | **Resonance** (RA) | Coherent amplification / propagation | Requires phase check (U3) | Propagates EPI, preserves identity |
| 7 | **Silence** (SHA) | Freezes evolution; `νf → 0`, EPI fixed | Closure (U1b) | Preserves EPI over time |
| 8 | **Expansion** (VAL) | Adds structural complexity; raises νf | Destabilizer (U2) | νf not decreased (capacity lever) |
| 9 | **Contraction** (NUL) | Removes complexity; νf↓ and ΔNFR densifies | — | νf not increased (acts on both levers) |
| 10 | **Self-organization** (THOL) | Autopoietic sub-EPI formation | Stabilizer (U2), handler (U4a), transformer (U4b) | Preserves global form while creating sub-EPIs |
| 11 | **Mutation** (ZHIR) | Phase transform `θ → θ'` when `ΔEPI/Δt > ξ` | Destabilizer (U2), trigger (U4a), transformer (U4b) | Requires prior IL + recent destabilizer (U4b) |
| 12 | **Transition** (NAV) | Controlled regime shift; activates latent EPI | Generator (U1a), closure (U1b) | Trajectory controlled (not a U2 destabilizer) |
| 13 | **Recursivity** (REMESH) | Echoes structure across scales (U5 fractality) | Generator (U1a), closure (U1b) | Network-scale; `EPI(t)` references `EPI(t−τ)` |

**Public naming**: the English name (Emission, Reception, …) is the canonical public
identifier; the glyph code (AL, EN, …) is the internal symbol.

### Dual-lever and contracts

Each operator's primary effect lands on **one nodal channel** — the channel partition
is simultaneously the **dual-lever** (capacity νf vs pressure ΔNFR), the tetrad driver,
and the number-theory grading:

- **νf (capacity)**: Silence, Expansion, Contraction
- **ΔNFR (pressure)**: Coherence, Dissonance, Self-organization, Transition
- **θ (phase)**: Coupling, Mutation
- **EPI (form, written directly)**: Emission, Reception, Resonance, Recursivity

The single source of truth for contracts (channel, scale NODE/NETWORK, postcondition,
TNFR.pdf anchor) is [src/tnfr/operators/operator_contracts.py](src/tnfr/operators/operator_contracts.py);
the proactive audit, reactive monitor, and introspection metadata all derive from it.
See [theory/STRUCTURAL_OPERATORS.md](theory/STRUCTURAL_OPERATORS.md).

### Composition

Operators compose into sequences satisfying U1–U6. The named building blocks are
structural **fragments (macros)**, not standalone valid words:

- **Bootstrap** = [Emission, Coupling, Coherence] · **Stabilize** = [Coherence, Silence]
- **Explore** = [Dissonance, Mutation, Coherence] · **Propagate** = [Resonance, Coupling]

A fragment becomes a valid word by adding the grammar glue (a U1a generator prefix, a
U1b closure suffix, and the U4b context a transformer needs), e.g. `[Emission,
Coupling, Coherence, Silence]`. Nesting `THOL[ body ]` lifts sequences to
context-free (nested sub-EPIs, U5); branching `OZ → [ZHIR | NUL]` is the U4a
bifurcation. See [examples/08_emergent_geometry/143_glyphic_function_sublanguage.py](examples/08_emergent_geometry/143_glyphic_function_sublanguage.py)
and [144_branching_combinator.py](examples/08_emergent_geometry/144_branching_combinator.py).

---

## 6. Unified grammar (U1–U6)

The grammar is derived from the nodal equation, not imposed. Validation entry point:
[src/tnfr/operators/grammar.py](src/tnfr/operators/grammar.py); canonical specification
[src/tnfr/operators/grammar_canon.py](src/tnfr/operators/grammar_canon.py); full
derivations [theory/UNIFIED_GRAMMAR_RULES.md](theory/UNIFIED_GRAMMAR_RULES.md).

- **U1 — Initiation & closure.** From `EPI = 0`, `∂EPI/∂t` is undefined, so a sequence
  must start with a generator `{AL, NAV, REMESH}` (U1a) and end in a coherent attractor
  `{SHA, NAV, REMESH, OZ}` (U1b).
- **U2 — Convergence & boundedness.** Because `∫νf·ΔNFR dt` must converge, any
  destabilizer `{OZ, ZHIR, VAL}` requires a stabilizer `{IL, THOL}`. Specialized
  sub-rule: REMESH combined with a destabilizer also requires `{IL, THOL}` (recursive
  amplification control).
- **U3 — Resonant coupling.** Coupling/resonance `{UM, RA}` require phase compatibility
  `|φᵢ − φⱼ| ≤ Δφ_max` (antiphase is destructive).
- **U4 — Bifurcation dynamics.** (a) Triggers `{OZ, ZHIR}` need handlers `{THOL, IL}`.
  (b) Transformers `{ZHIR, THOL}` need a recent destabilizer (~3 ops); ZHIR also needs a
  prior IL (stable base).
- **U5 — Multi-scale coherence.** Nested EPIs require stabilizers at each level;
  `C_parent ≥ α · Σ C_child`.
- **U6 — Structural potential confinement.** Telemetry safety: monitor `Δ Φ_s < 1.618`
  (`Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)²`). Read-only check, not a sequence constraint.

**Single source of truth.** The operator-classification sets (generators, closures,
stabilizers `{IL, THOL}`, destabilizers `{OZ, ZHIR, VAL}`, transformers `{ZHIR, THOL}`)
are **derived** from per-operator nodal-equation predicates in
[src/tnfr/config/physics_derivation.py](src/tnfr/config/physics_derivation.py) and
re-exported by [src/tnfr/operators/grammar_types.py](src/tnfr/operators/grammar_types.py);
every consumer imports from there. NAV is **not** a destabilizer (its trajectory is
controlled). Proactive, incremental enforcement during dynamic operator selection lives
in [src/tnfr/operators/grammar_dynamics.py](src/tnfr/operators/grammar_dynamics.py) and
[grammar_application.py](src/tnfr/operators/grammar_application.py).

---

## 7. Telemetry & metrics

- **C(t) — total coherence** `[0,1]`, the primary stability indicator:
  `C(t) = 1 / (1 + mean|ΔNFR| + mean|dEPI|)`, derived from the nodal equation
  (equilibrium → `C → 1`). Strong coherence `C > 0.7506`; fragmentation
  risk `C < 0.2415`. The two threshold *values* are heuristic telemetry cuts
  (not derived from the dynamics, nor fitted to data).
- **Si — sense index** `[0,1+]`, reorganization-capacity predictor: `Si > 0.8`
  excellent; `Si < 0.4` bifurcation-prone.
- **Tetrad safety** (telemetry; see §3): only `|K_φ| < 0.9·π ≈ 2.827` is a genuine
  geometric bound (phase wrap). `Δ Φ_s < 1.618` / `|Φ_s| < 0.7711` are empirical; the `|∇φ|`
  sync onset is `≈ 0.29` (σ-dependent, not a fixed constant); `ξ_C` is set by the spectral
  gap `λ₂` (`ξ_C ∝ 1/√λ₂`).

Required telemetry must stay in TNFR-coherent terms (C(t), Si, phase, νf, and the
tetrad), in Hz_str units. Computation: [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py),
[src/tnfr/physics/telemetry.py](src/tnfr/physics/telemetry.py).

---

## 8. Canonical invariants

Six invariants define TNFR consistency; preserve all of them.

1. **Nodal equation integrity** — EPI changes only via `∂EPI/∂t = νf·ΔNFR`; ΔNFR keeps
   structural-pressure semantics; `νf → 0` inactivates. (Grammar U1, U2.)
2. **Phase-coherent coupling** — `|φᵢ − φⱼ| ≤ Δφ_max` required before any coupling.
   (Grammar U3; `validate_resonant_coupling()`.)
3. **Multi-scale fractality** — EPIs nest without identity loss. (Grammar U5.)
4. **Grammar compliance** — operator sequences pass U1–U6; new functions map to
   existing operators or define a new operator with full contracts.
5. **Structural metrology** — νf in Hz_str; C(t), Si, phase, νf exposed in telemetry.
6. **Reproducible dynamics** — identical seeds give identical trajectories; operations
   are traceable.

---

## 9. TNFR agent playbook

How a TNFR agent (human or AI) should reason and act.

1. **Start from physics.** Treat `∂EPI/∂t = νf·ΔNFR` as the source of truth; keep EPI,
   νf, and phase well-defined; interpret behavior through the tetrad.
2. **Operate only via canonical operators.** Never mutate EPI directly; map every new
   behavior to existing operators (or justify a new one with full physics, contracts,
   and tests); preserve operator semantics in refactors.
3. **Enforce U1–U6.** Check sequence validity; guard destabilizers with stabilizers;
   never couple without an explicit phase check.
4. **Preserve invariants.** Keep Hz_str units; treat ΔNFR as structural pressure (not an
   ML loss); maintain operational fractality.
5. **Demand reproducible, telemetry-rich experiments.** Fix seeds; expose C(t), Si,
   phase, νf, and the tetrad; test monotonicity and safety.
6. **Accept / reject by structural criteria.** Accept changes that raise C(t) or reduce
   harmful ΔNFR, strengthen U1–U6 compliance, and improve physics→math→code→tests
   traceability. Reject changes that introduce magic constants, bypass operators, or
   break phase verification, units, or invariants.
7. **Communicate physics-first, in English**, tracing every significant decision back to
   a specific piece of TNFR physics or grammar.

If a change "prettifies" code but weakens TNFR fidelity, reject it. If it strengthens
structural coherence and traceability, proceed.

---

## 10. Development workflow

**Before writing code**: read the relevant doctrine here and in
[theory/UNIFIED_GRAMMAR_RULES.md](theory/UNIFIED_GRAMMAR_RULES.md); check whether the
utility already exists; run the test suite to understand current state.

**Implementing changes**: search first; map new functions to operators; preserve all six
invariants; add tests covering contracts and invariants; document the structural effect;
trace the physics → math → code chain.

**Acceptable** changes increase C(t) or reduce ΔNFR where appropriate, preserve operator
closure and fractality, and keep APIs stable or mapped. **Unacceptable**: recasting ΔNFR
as an ML error gradient; replacing operators with unmapped imperative code; flattening
nested EPIs; coupling without phase checks; mutating EPI directly; changing units
(Hz_str → Hz).

### Commit / PR templates

```text
Intent: [which coherence is improved]
Operators involved: [Emission|Reception|...]
Affected invariants: [#1-6]
Key changes: [bullets]
Expected risks/dissonances: [and containment]
Metrics: [C(t), Si, νf, phase] before/after
```

PRs should show what reorganizes (C(t)↑ / ΔNFR↓, closure & fractality preserved),
evidence (phase/νf logs, C(t)/Si curves, controlled bifurcations), compatibility
(stable/mapped API, reproducible seed), and tests.

### Testing requirements

Cover, at minimum: coherence **monotonicity** (IL does not reduce C(t) outside dissonance
tests), **bifurcation** (OZ triggers with handlers present), **propagation** (RA raises
phase sync), **latency** (SHA keeps EPI invariant), **mutation threshold** (ZHIR changes θ
only when `ΔEPI/Δt > ξ`), **multi-scale** (nested EPIs keep identity), and
**reproducibility** (same seed → same trajectory). See [TESTING.md](TESTING.md).

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Needs generator" | Start from EPI=0 without U1a | Prefix `{AL, NAV, REMESH}` |
| "Destabilizer without stabilizer" | OZ/ZHIR/VAL without IL/THOL (U2) | Add a stabilizer |
| "Phase mismatch in coupling" | `|φᵢ − φⱼ| > Δφ_max` (U3) | Ensure phase compatibility first |
| "Mutation without context" | ZHIR without recent destabilizer / prior IL (U4b) | Add a destabilizer (~3 ops) and a prior IL |
| C(t) decreasing unexpectedly | Monotonicity contract violated | Verify operator preserves C(t) |
| Node collapse | `νf → 0`, extreme dissonance, or decoupling | Apply coherence earlier; ensure coupling |

Debugging order: inspect telemetry (C(t), Si, νf, phase, ΔNFR) → verify grammar U1–U6 →
check operator contracts → identify the violated invariant → trace against nodal-equation
predictions.

---

## 12. Research programs

TNFR applies the nodal dynamics to several open mathematical and physical questions. Each
has a dedicated theory note; this file keeps only a one-line status and **never inlines
program history** (the full milestone/gap/branch threads live in the notes).

| Program | Status | Reference |
|---------|--------|-----------|
| **TNFR-Riemann** | `σ_c → ½` numerically verified; ζ↔L attack surface shipped (P12–P49). The bridge to RH is the open conjecture **T-HP** (gap G4), paused at the oscillatory residue `S(T) = (1/π)·arg ζ(½+iT)`. | [TNFR_RIEMANN_RESEARCH_NOTES.md](theory/TNFR_RIEMANN_RESEARCH_NOTES.md) |
| **REMESH-∞ closure** | The 13-operator catalog is closed under the `τ_g → ∞` limit (N15, Branch A); universality is structural/operational, not spectral. | [REMESH_INFINITY_DERIVATION.md](theory/REMESH_INFINITY_DERIVATION.md) |
| **TNFR-Navier–Stokes** | `K_φ` cascade and a BKM-analogue U2 blow-up criterion; 2D asymmetry closed at the discrete level; continuum closure NS-G1 open. | [TNFR_NAVIER_STOKES_RESEARCH_NOTES.md](theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md) |
| **Number theory** | Primality as `ΔNFR = 0`; arithmetic structural triad; the cyclotomy law `s_k(p) = gcd(k, p−1) + 1` (proved). | [TNFR_NUMBER_THEORY.md](theory/TNFR_NUMBER_THEORY.md) |
| **Millennium reformulations** | P vs NP, BSD, Hodge, Yang–Mills: TNFR-internal structural reformulations and diagnostics — none a proof. | `theory/TNFR_*_RESEARCH_NOTES.md` |

**Honest scope, global**: these programs produce TNFR-internal structural results and
numerical evidence; none currently closes a classical open problem. Do not extend a
program's diagnostic surface without a new structural idea, and never claim a proof of RH,
Navier–Stokes regularity, or any Millennium problem.

---

## 13. Map of the codebase & examples

- **Physics** — [src/tnfr/physics/](src/tnfr/physics/): `fields.py` (tetrad),
  `conservation.py` (conservation theorem), `symplectic_substrate.py` (emergent geometry),
  `structural_diffusion.py` (transport), `gauge.py` (Ψ, gauge), `integrity.py`
  (operator-postcondition monitor + audit).
- **Operators & grammar** — [src/tnfr/operators/](src/tnfr/operators/): `definitions.py`
  (13 operators + registry), `operator_contracts.py` (contract source of truth),
  `grammar*.py` (U1–U6 validation, dynamics, application), `nodal_equation.py`.
- **Engines** — [src/tnfr/engines/](src/tnfr/engines/): self-optimization, pattern
  discovery, computation, integration, engine-scoped constants.
- **Programs** — [src/tnfr/riemann/](src/tnfr/riemann/), [src/tnfr/navier_stokes/](src/tnfr/navier_stokes/),
  and number theory in [src/tnfr/mathematics/](src/tnfr/mathematics/).
- **SDK** — [src/tnfr/sdk/](src/tnfr/sdk/): `simple.py` (`TNFR.create(...)`, tetrad,
  conservation, substrate, integrity, audit), `fluent.py` (`auto_optimize()`).
- **Examples** — [examples/README.md](examples/README.md): ten thematic folders
  (`01_foundations` … `10_applications`); each file keeps a stable global number.
- **Theory hub** — [theory/README.md](theory/README.md) · **Glossary** —
  [theory/GLOSSARY.md](theory/GLOSSARY.md) · **Architecture** — [ARCHITECTURE.md](ARCHITECTURE.md)
  · **Benchmarks** — [benchmarks/](benchmarks/) · **Tests** — [tests/](tests/).

The Simple SDK exposes the research-grade stack directly:

```python
from tnfr.sdk import TNFR
net = TNFR.create(20).ring().evolve(5)
net.tetrad()                 # TetradSnapshot (Φ_s, |∇φ|, K_φ, ξ_C) + is_safe()
net.conservation()           # Noether charge, Lyapunov stability
net.symplectic_substrate()   # the emergent geometry
net.telemetry()              # C(t), Si, phase_sync, tetrad
net.audit_operators()        # 13/13 operator-contract audit
analysis = TNFR.analyze(net) # one-shot comprehensive report
```

---

## 14. Philosophy & excellence standards

### Core principles

1. **Physics first** — every feature derives from TNFR physics.
2. **No arbitrary choices** — decisions trace to the nodal equation or the invariants.
3. **Coherence over convenience** — preserve theoretical integrity even when the code is
   harder.
4. **Reproducibility always** — every simulation is reproducible.
5. **Document the chain** — theory → math → code → tests.

### Decision framework

A change should be implemented only if it strengthens TNFR fidelity, maps to operators,
preserves the invariants, is derivable from physics, and is testable. Organizational
convenience is not physical necessity; untestable "magic" is rejected.

### The TNFR mindset

Think in **patterns**, not objects ("the neural pattern reorganizes", not "the neuron
fires"); in **dynamics**, not states (trajectory and attractor, not snapshot); in
**networks**, not individuals (resonant propagation, not isolated change).

### Final principle

TNFR models coherent dynamic patterns; development practice reflects that. If a change
prettifies code but weakens TNFR fidelity, reject it. If it strengthens structural
coherence and paradigm traceability, proceed.

---

**Status**: CANONICAL — synthesized primary reference for TNFR agent guidance.
**Policy**: English-only; updated only when novel, important TNFR canonicity emerges, and
always written as a complete, self-contained synthesis.
