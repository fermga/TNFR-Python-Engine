# TNFR: Resonant Fractal Nature Theory

**Theoretical framework and engine for coherent pattern analysis on graph-coupled networks.**

- **Version**: 0.0.3.5
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

- **Form (EPI)** — coherent configuration in a structural manifold; named
  transformations use canonical operators, while declared solver steps use the
  shared nodal integrator with explicit pressure and provenance; supports
  nesting (operational fractality).
- **Frequency (νf)** — reorganization rate in Hz_str (ℝ⁺); `νf → 0` deactivates.
- **Phase (φ or θ)** — synchronization parameter in [0, 2π); coupling is admissible
  only when the circular separation
  `δ(φᵢ,φⱼ) = |wrap(φᵢ − φⱼ)| ≤ Δφ_max`.

The graph engine's real scalar EPI chart is represented either directly or by
the uniform-real `BEPIElement` embedding. `real_scalar_epi` recovers the signed
coordinate and `scalarize_epi` applies the same rule to live and serialized
values. `abs(BEPIElement)` remains a nonnegative magnitude; genuinely nonuniform
or complex elements have no signed one-dimensional representative and are
rejected by canonical glyphs that require a real scalar EPI coordinate,
scalar-only diffusion and affine certificates.

### The fractal-resonant node (NFR)

The node carrying the triad is a **Nodo Fractal Resonante (NFR)** — canonically, a
**region of structural coherence coupled to a network** (TNFR.pdf §1.4.1). The triad
(EPI, νf, φ) *defines* it; four properties characterize it:

- **Multiscalar (fractal)** — an NFR can nest other NFRs (operational fractality; THOL
  sub-EPIs, REMESH, grammar U5). A single node is a micro-NFR; a coherent region is a
  macro-NFR.
- **Autopoietic (self-generated)** — emerges by local reorganization, no external
  support (Emission from vacuum; THOL).
- **Relational** — exists only by coupling (U3). **Temporal** — persists while it
  reorganizes its coherence.

Its **nodal topology** is **radial** (one central nucleus), **annular** (passive center,
peripheral ring) or **multinodal** (several centers), read from the emergent
structural-potential geometry by [classify_nodal_topology](src/tnfr/physics/fields.py)
using its calibrated canonical inverse-square kernel (`alpha = 2`), and surfaced
as a whole-NFR read-out by `Network.nfr()` ([src/tnfr/sdk/simple.py](src/tnfr/sdk/simple.py)).

The pressure equilibrium `ΔNFR = 0` is the **zero-pressure fixed-point set** —
the state where reorganization pressure vanishes (`C = 1` when `dEPI = 0`).
Stationarity alone is weaker: zero capacity freezes EPI even under nonzero pressure.
For the pure EPI channel on a fixed connected symmetric graph with positive capacity,
equilibrium is a uniform field; disconnected components can have different constants.
In that restricted model the uniform field is an attractor. This result does
not establish attractors for the full multichannel dynamics or for static
arithmetic and chemical pressure fields. The shared numeric predicate is
[is_structural_equilibrium](src/tnfr/metrics/common.py), and the per-node coherence map
`structural_coherence` (`C = 1/(1+|ΔNFR|+|dEPI|)`) is the **single kernel** every domain
reads — graph nodes, arithmetic nodes (primes), chemical nodes (noble gases) — with only
the `ΔNFR` realisation domain-specific.

### Accumulated evolution and the convergence policy

Integrating the nodal equation gives the accumulated structural change:

$$\mathrm{EPI}(t)-\mathrm{EPI}(t_0)=\int_{t_0}^{t}\nu_f(\tau)\,\Delta\mathrm{NFR}(\tau)\,d\tau.$$

Finite-horizon integrability, bounded trajectories, convergence as `t → ∞`, and
absolute integrability are distinct conditions. Absolute integrability is sufficient
for a finite limiting EPI; bounded or vanishing pressure alone is insufficient.
Positive feedback can destabilize a trajectory, while diffusion can relax without
named stabilizers. Grammar **U2** is the engine's stabilization and debt policy,
motivated by controlling this accumulated change; it is not a universal convergence
theorem for arbitrary pressure laws. See
[theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md](theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).

### Transport content (structural diffusion)

The canonical `ΔNFR` aggregates four structural gradient channels,
`ΔNFR = w_phase·∂φ + w_epi·∂EPI + w_vf·∂νf + w_topo·∂topo` (weights and defaults in
[src/tnfr/dynamics/dnfr.py](src/tnfr/dynamics/dnfr.py)). The **EPI channel** is
**exactly a graph diffusion**. For the EPI channel,

$$\Delta\mathrm{NFR}_{\text{epi}}(i) = \overline{\mathrm{EPI}}_{\mathcal{N}(i)} - \mathrm{EPI}(i) = -(L_{\mathrm{rw}}\,\mathrm{EPI})(i),\qquad L_{\mathrm{rw}} = I - D^{-1}W,$$

so `∂EPI/∂t = −diag(νf) · L_rw · EPI` describes the isolated channel, with zero
Laplacian rows at isolates. For fixed capacities and fixed symmetric nonnegative
weights with a common
positive capacity, modes decay as `e^{−νf λ_k t}` and the degree-weighted total
is conserved. Positive heterogeneous capacities instead conserve weights `d_i/νf_i`;
their rates are eigenvalues of `diag(νf)L_rw`. On a connected homogeneous graph,
`νf·λ₂` is the slowest nonuniform decay rate. The reaction threshold `r_c=νf·λ₂`
concerns nonuniform modes; a positive reaction already grows the uniform mode.
The executable fixed-capacity certificate rationalizes its materialized
binary64 generator: consensus-subspace invariance, the stronger uniform
fixed-point identity and preservation of the displayed weighted mean are three
separate facts. The bounded time-varying theorem instead treats the effective
binary64 conductances materialized by the shared reader and the declared bounds
as exact real coefficients, derives its Laplacian and rate rationally, and keeps
ordinary binary64 spectral diagnostics separate. A proven positive exact rate
may lack an operational float representation after underflow. Future-schedule
and numerical-integration verification remain open.
These scoped identities do not certify U2 for arbitrary operator sequences.
The exact Dirichlet balance below uses this same adjacency convention. See
[src/tnfr/physics/structural_diffusion.py](src/tnfr/physics/structural_diffusion.py).

**Phase scale boundary.** In a selected open-semicircle chart, the pairwise
phase-pressure realization `-(1/π) diag(νf)L_rw q` inherits the reversible EPI
quotient identities. The actual canonical phase channel uses the argument of an
unweighted neighbor-phasor sum (including zero-weight support edges) and is
nonlinear. It closes on a lifted
block-constant subspace only under equitable support profiles, no intra-fiber
edges, equal active macro-neighbor multiplicities, block-constant capacity and
nonzero phasor resultants. A `K3,3` same-macro-state counterexample refutes
global projected autonomy. Branch crossing, changing support and finite-time
phase closure remain open. See
[src/tnfr/physics/phase_quotient.py](src/tnfr/physics/phase_quotient.py).

---

## 3. The structural tetrad

Four structural fields form the canonical diagnostic read-out of a network.
They organize aggregation, local phase derivatives and non-local correlation.
Their selection does not establish complete reconstruction of the graph state or
its evolution. **π** is the exact phase-wrap scale bounding `|∇φ|` and `|K_φ|`.

| Structural field | Symbol | Tower order | Role |
|------------------|--------|-------------|------|
| Structural potential | Φ_s | 0th (aggregation) | Global stability |
| Phase gradient | \|∇φ\| | 1st (local) | Local desynchronization stress |
| Phase curvature | K_φ | 2nd (local) | Geometric torsion |
| Coherence length | ξ_C | non-local | Correlation range |

### Field scales

Each field has a characteristic scale or configured interpretation:

- **π — phase scale (geometric, exact).** Both phase derivatives are **wrapped
  angles**, so `|∇φ| ≤ π` *and* `|K_φ| ≤ π` for any configuration — π scales the
  whole **phase sector**. `|K_φ| < 0.9·π ≈ 2.827` is a selected safety margin
  inside that exact bound; the factor 0.9 is a policy choice.
- **|∇φ| — phase-wrap bounded.** Its genuine bound is `|∇φ| ≤ π`. There is no fixed
  structural constant for the synchronization onset: the measured value is `≈ 0.29`
  and σ-dependent (a dynamical transition, not a derived threshold).
- **ξ_C — correlation length.** The correlation fit is state-dependent; the
  spectral scale `1/√λ₂` supplies a model comparison and fallback, not an identity
  for every fitted field or graph.
- **Φ_s — confinement policy.** `Δ Φ_s < π/2 ≈ 1.571` is the U6 drift policy;
  per-node `|Φ_s| < π/4 ≈ 0.785` is a separate magnitude policy. Both are selected
  π-scaled thresholds. Phase wrapping alone does not bound the source aggregation:
  `||Φ_s||∞ ≤ ||B_G||∞ ||ΔNFR||∞`, where `(B_G)_ij=1/d(i,j)²` off the diagonal
  and zero for unreachable pairs. Explicit edge `length` defines `d`; absent
  `length`, the transport `weight` remains a compatibility fallback. A complete
  graph with unit pressure gives `Φ_s=n−1`.

**Structure and scope.** Circular phase curvature linearizes to a Laplacian action
in a small-spread regime with matching neighborhood weights. The wrapped nonlinear
field is not globally the linear EPI diffusion operator. Local phase derivatives
and non-local source/correlation diagnostics provide complementary information.

### Why these four (scope and minimality boundary)

The four canonical channels organize the following read-outs:

```
ΔNFR_j → Σ 1/d²  → Φ_s   (0th order, global aggregation)
φ_i    → ∇       → |∇φ|   (1st order, local)
       → ∇²      → K_φ    (2nd order, local; Laplacian after phase linearization)
       → corr    → ξ_C    (non-local correlation range)
```

Higher Laplacian powers belong to the algebra generated by the Laplacian, but this
does not make their observations reconstructible from lossy tetrad summaries.
For four distinct eigenvalues, `I,L,L²,L³` are linearly independent. Uniformly
rescaling capacity can preserve the tetrad while changing the nodal evolution rate.
A minimal complete state basis therefore remains unproved; the tetrad remains the
required diagnostic interface.
Full treatment: [theory/MINIMAL_STRUCTURAL_DEGREES.md](theory/MINIMAL_STRUCTURAL_DEGREES.md),
[theory/FUNDAMENTAL_THEORY.md](theory/FUNDAMENTAL_THEORY.md),
[docs/STRUCTURAL_FIELDS_TETRAD.md](docs/STRUCTURAL_FIELDS_TETRAD.md). All four fields
are CANONICAL; compute them via [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py).

---

## 4. Emergent geometry

Graph fields support geometric read-outs and specified auxiliary dynamical models.
Their algebraic identities, model flows and correspondence with engine trajectories
must be stated separately. The implementations expose the corresponding scope.

### Auxiliary symplectic substrate

The substrate model uses an ambient symplectic space `P = ℝ^{4N}` with two canonical
conjugate pairs per node, initialized from extracted graph fields:

- **Geometric sector** `(K_φ, J_φ)` — curvature ↔ phase current
- **Potential sector** `(Φ_s, J_ΔNFR)` — potential ↔ ΔNFR flux

with canonical brackets `{K_φ, J_φ} = {Φ_s, J_ΔNFR} = 1`. The Hamiltonian is the
structural **energy functional** `H_sub = ½Σ(K_φ² + J_φ² + Φ_s² + J_ΔNFR²)` (plus
the held-fixed `½Σ|∇φ|²` background). Its exact harmonic flow is a
**symplectomorphism**, preserving phase volume. This does not certify the 13 engine
operators: each induced map would need its own symplecticity check, and extracted
graph fields need not fill or remain on the ambient space under that flow.
**Noether** ties each continuous symmetry to a conserved charge: time translation →
`H_sub`; the geometric U(1) (`Ψ → e^{iα}Ψ`) → `E_geo = ½Σ|Ψ|²`; the potential U(1)
→ `E_pot`. The complex coordinate `Ψ = K_φ + i·J_φ` is the geometric sector under
the substrate's complex structure (flat Kähler). The substrate further carries a
**U(2) polarization symmetry** with conserved Stokes parameters on a per-node
**Poincaré sphere** — this is classical wave polarization (Stokes/Poincaré), a
product (un-entangled) classical texture, **not** a quantum state.

The full nodal equation has not been derived as an overdamped projection of this
isotropic Hamiltonian. A restricted exact bridge exists for EPI-only diffusion
on fixed symmetric nonnegative conductance: with `B=D−W`,
`E_D=½ EPIᵀ B EPI`, and `M=diag(νf_i/d_i)` (zero at isolates),
`dEPI/dt=−M∇E_D` and `dE_D/dt=−∇E_Dᵀ M∇E_D≤0`.
`compute_diffusion_energy` reads this balance without evolving the graph.
This Dirichlet energy differs from the tetrad potential; zero capacity yields
degenerate mobility. A separate damped graph wave with stiffness `L_rw` has a
slow diffusion limit, while the isotropic substrate has identity stiffness.
See [theory/TNFR_VARIATIONAL_PRINCIPLE.md](theory/TNFR_VARIATIONAL_PRINCIPLE.md).
Implementation and certificates: [src/tnfr/physics/symplectic_substrate.py](src/tnfr/physics/symplectic_substrate.py);
gauge / U(2) structure: [theory/GAUGE_SYMMETRY_AND_UNIFICATION.md](theory/GAUGE_SYMMETRY_AND_UNIFICATION.md).

### Structural conservation theorem (Noether-like)

The structural conservation diagnostics measure the balance

$$\frac{\partial \rho}{\partial t} + \nabla\cdot\mathbf{J} = S_{\text{grammar}},\qquad \rho = \Phi_s + K_\phi,\quad \mathbf{J} = (J_\phi, J_{\Delta\mathrm{NFR}}),$$

where the residual `S_grammar` must be measured along the actual trajectory;
valid operator labels alone do not prove it vanishes. The energy functional
`E = ½Σ(Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²) ≥ 0` is a **Lyapunov candidate**:
its non-increase requires trajectory evidence or a model-specific proof. A
complete general proof of asymptotic stability remains open. The six derived fields
(`χ, 𝒮, 𝒞, ℰ, 𝒜, 𝒬`) are bilinear contractions of the singlets (`Φ_s`, `|∇φ|`) and
the complex fields `Ψ = K_φ + i·J_φ` and `Ω = |∇φ| + i·J_ΔNFR` (e.g. chirality
`χ = Re(Ψ·Ω)`). Conservation theorem:
[src/tnfr/physics/conservation.py](src/tnfr/physics/conservation.py),
[theory/STRUCTURAL_CONSERVATION_THEOREM.md](theory/STRUCTURAL_CONSERVATION_THEOREM.md);
emergent fields: [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py),
[theory/EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md](theory/EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md).

### Regime correspondences

The nodal law and graph-wave diagnostics admit the following scoped comparisons
(external labels "classical"/"quantum-like" are not TNFR primitives):

- **Smooth-trajectory / overdamped drift**: first order in time,
  `q̇ = νf·F` — drift velocity ∝ force, `νf` is **mobility** (Stokes/Einstein), not
  inverse mass. The inertial (second-order) regime lives in the conservative
  substrate flow.
- **Discrete-mode**: finite graphs give a finite spectrum independently of dissonance.
  Symmetric normalized diffusion supplies an orthonormal basis; graph-wave
  frequencies are `√λ_k`. Nodal-domain upper bounds do not imply monotonic counts
  across every degenerate eigenbasis. Directed transport needs separate analysis.

The graph-wave model has a **sustained vibration — the pulse**, read at two scales: the
**collective** network rhythm (resonances `ω_k = √λ_k`, the fundamental, the dominant beat
`ω_j − ω_k`, vibration energy; `compute_emergent_pulse`, SDK `net.rhythm()`) and the **per-NFR**
pulse — every NFR a phase oscillator pulsing at its own `νf` with phase `φ`, coupled by
**resonance** (`local_phase_sync` per NFR, the Kuramoto order `R`, gate `Δφ_max = π/2`;
`compute_nodal_pulse`, SDK `net.resonance()`). These report modal rhythm and phase
synchronization separately; a high Kuramoto order does not by itself establish
an engine trajectory following the auxiliary conservative wave.

See [src/tnfr/physics/structural_diffusion.py](src/tnfr/physics/structural_diffusion.py)
and [examples/02_physics_regimes/](examples/README.md).

---

## 5. The 13 canonical operators

Within the engine, registered canonical operators are the **semantic interface**
for named node and network transformations. Declared numerical solvers form a
separate evolution path: they may advance EPI only through the shared nodal
integrator from explicit `nu_f` and `DeltaNFR`, with provenance or a residual;
ad hoc state assignment is forbidden. The current registry contains 13
transformations with defined physical contracts. Whether their composition,
nesting and branching generate every mathematically admissible TNFR
transformation remains open under research line S10.

| # | Operator (glyph) | Physics / effect | Grammar role | Contract |
|---|------------------|------------------|--------------|----------|
| 1 | **Emission** (AL) | Sources EPI from vacuum at pre-existing basal νf; leaves νf unchanged | Generator (U1a) | Sources new form without writing capacity, pressure or phase |
| 2 | **Reception** (EN) | Integrates incoming resonance | — | Must not reduce C(t) |
| 3 | **Coherence** (IL) | Negative feedback; reduces \|ΔNFR\|, raises C(t) | Stabilizer (U2) | Must not reduce C(t) (outside dissonance test) |
| 4 | **Dissonance** (OZ) | Controlled instability; raises \|ΔNFR\| | Destabilizer (U2), bifurcation trigger (U4a), closure (U1b) | Must increase \|ΔNFR\| |
| 5 | **Coupling** (UM) | Phase synchronization link `φᵢ → φⱼ` | Requires phase check (U3) | Valid only if `\|wrap(φᵢ − φⱼ)\| ≤ Δφ_max` |
| 6 | **Resonance** (RA) | Coherent amplification / propagation | Requires phase check (U3) | Blends scalar EPI over U3-compatible neighbours while preserving sign/kind identity |
| 7 | **Silence** (SHA) | Freezes evolution; `νf → 0`, EPI fixed | Closure (U1b) | Preserves EPI over time |
| 8 | **Expansion** (VAL) | Adds structural complexity; raises νf | Destabilizer (U2) | νf not decreased (capacity lever) |
| 9 | **Contraction** (NUL) | Removes complexity; νf↓ and ΔNFR densifies | — | νf not increased (acts on both levers) |
| 10 | **Self-organization** (THOL) | Autopoietic sub-EPI formation | Stabilizer (U2), handler (U4a), transformer (U4b) | Preserves global form while creating sub-EPIs |
| 11 | **Mutation** (ZHIR) | Phase transform `θ → θ'` when `ΔEPI/Δt > ξ` | Destabilizer (U2), trigger (U4a), transformer (U4b) | Requires prior IL + recent destabilizer (U4b) |
| 12 | **Transition** (NAV) | Controlled regime shift; activates latent EPI | Generator (U1a), closure (U1b) | Trajectory controlled (not a U2 destabilizer) |
| 13 | **Recursivity** (REMESH) | Echoes structure across scales (U5 fractality) | Generator (U1a), closure (U1b) | Network-scale; `EPI(t)` references `EPI(t−τ)` |

**Public naming**: the lowercase English token (`emission`, `reception`, …) is
the canonical executable identifier. Title-case names (Emission, Reception, …)
are public display/class names; glyph codes (AL, EN, …) are internal symbols.

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

**Mutation evidence boundary.** The nodal product `νf·ΔNFR` is the instantaneous
model prediction; it is not by itself an observed ZHIR trigger. Direct Mutation
requires active capacity and a finite signed two-sample secant strictly above
`ZHIR_THRESHOLD_XI`. Timestamped `epi_time_history` is authoritative, uses its
physical interval, and must end at the current EPI state. Legacy `epi_history` and
`_epi_history` retain an explicit unit-operator-step interpretation and make no
physical-time claim. Missing, invalid, stale, equal-threshold or contracting evidence
rejects direct execution; autonomous selection instead applies IL and records the
abstention. The SDK preflights every target before a word containing ZHIR. Structural
acceleration is a separate three-sample diagnostic, centralized by
`compute_d2epi_dt2`; it does not replace the two-sample gate or U4b context. The
`νf = 0.5` bifurcation-router threshold only proposes a branch and is not a Mutation
admission condition. `ZHIR_BIFURCATION_MODE` supports detection only; the legacy
`variant_creation` value is rejected before any write. ZHIR remains a theta-only
transformation with bifurcation detection; topology and sub-EPI creation belong to
THOL. Pure certificates and `Network.nodal_state()` report prediction and evidence
without certifying U4b execution readiness.

### Operator events and physical time

Canonical operator events are represented as zero-duration hybrid jumps. A
finite schedule with `m` events therefore declares exactly `m + 1` nodal-flow
intervals: before, between and after the jumps. Materialized binary64 durations
have authoritative exact-rational values and offsets; absolute float timestamps
are display values and are never subtracted to recover duration or inserted as
finite secants into `epi_time_history`. Coincident events retain their explicit
event-index order in `hybrid_event_log`. For fixed connected symmetric
positive-capacity pure-EPI diffusion, the exact quotient-rate certificate and
rational log/exp enclosures provide a sufficient duration for a requested
disagreement-energy fraction. These schedule and duration objects are read-only
and do not turn U2/U4 into an adaptive policy.

`execute_operator_event_schedule` binds a valid finite schedule to the
configured nodal integrator and shared all-target stage dispatcher. It freezes
the initial targets, requires the live binary64 clock at every boundary, rejects
collapsed or nonadditive positive intervals, and commits graph-owned flow, jump,
history, cache and event-log state in one transaction. Flow boundaries supply
timestamped EPI evidence; a same-time EPI jump restarts that history and remains
a zero-duration event. ZHIR therefore requires a positive representable
immediately preceding flow. With `include_flow_certificates=True`, detached
snapshots around each actual positive flow produce an
`ExecutedNodalFlowInterval`. Its evidence keeps the exact rational
held-pressure nodal identity and pure-EPI quotient theorem separate from the
binary64 Euler replay. Trusted binary64 held-pressure identification requires
the exact built-in `DefaultIntegrator`, Euler with one substep, live Gamma type
`none`, inactive clipping, disabled extended dynamics, stable support,
unchanged capacity and pressure, and a matching replay. Exact rational pure-EPI
affine promotion additionally requires fixed symmetric nonnegative conductance,
positive row strengths and capacity, the stored pressure `-L_rw EPI` and the
exact nodal identity. Stale pressure can pass the trusted binary64 runtime level
while the pure-EPI affine map and quotient theorem abstain.
Solver accuracy, refinement equivalence, mixed glyph/REMESH gain, adaptive
U2/U4 and future or repeated schedule stability remain unproved.

`execute_event_remesh_cycle` composes one finite schedule with the separately
invoked delayed map under an outer graph transaction. It appends exactly one
full-support pre-REMESH `_epi_hist` sample through the ordinary runtime helper,
so delay `tau` remains `history[-(tau + 1)]`; no post-jump delayed-history
sample is stored. An applied jump separately records its same-time right
endpoint in authoritative `epi_time_history`. The schedule must preserve ordered
node support and incoming delayed history; edges may change there. The endpoint
clock, committed event log, phase, pressure-hook identity and deterministic
REMESH controls remain frozen. The EPI-only map treats ON_REMESH callbacks as
observers and preserves the post-schedule topology and all non-EPI channels.
One frozen positive diagonal metric measures cycle-level weighted EPI
observations and feeds delayed-map evidence; legacy metadata retains unweighted
means. Exact values remain authoritative when an optional float display is
`None`. Weighted-consensus drift, capacity and pressure refresh remain separate.
The optional refresh runs only after an applied map. This certifies graph-owned
one-cycle execution, not solver accuracy, mixed gain or repeated stability with
evolving history. External effects remain outside rollback. See
[src/tnfr/operators/event_timing.py](src/tnfr/operators/event_timing.py),
[src/tnfr/operators/event_runtime.py](src/tnfr/operators/event_runtime.py),
[src/tnfr/operators/event_remesh_runtime.py](src/tnfr/operators/event_remesh_runtime.py)
and [src/tnfr/physics/event_duration.py](src/tnfr/physics/event_duration.py).

### Composition

Operators compose into sequences satisfying U1–U6. The named building blocks are
structural **fragments (macros)**, not standalone valid words:

- **Bootstrap** = [Emission, Coupling, Coherence] · **Stabilize** = [Coherence, Silence]
- **Explore** = [Dissonance, Mutation, Coherence] · **Propagate** = [Resonance, Coupling]

A fragment becomes a valid word by adding the grammar glue (a U1a generator prefix, a
U1b closure suffix, and the U4b context a transformer needs), e.g. `[Emission,
Coupling, Coherence, Silence]`. Nesting `THOL[ body ]` lifts sequences to
context-free (nested sub-EPIs, U5); branching `OZ → [ZHIR | NUL]` is the U4a
bifurcation. SDK words and supported GPU blocks preserve operator order. When
grammar keeps the requested glyph for every target, all thirteen stages
read one immutable snapshot and commit atomically (two-phase Jacobi):
neighbour-reading EN/IL/RA, pointwise AL/SHA/VAL/NUL/ZHIR/NAV, overlapping
phase/topology stage UM, overlapping-pressure stage OZ, and child-support and
hierarchy stage THOL, and advisory stage REMESH. IL contracts
target `DeltaNFR` magnitude and
locks target phase from the shared snapshot; its canonical stage/global and
radius-local structural `C(t)` telemetry is distinct from retained legacy
pressure-dispersion fields. Supported GPU AL/RA blocks reuse the same paths.
Their committed primary structural channels are target-order invariant before
the opaque pressure-refresh callback; AL/SHA also share one stage timestamp.
NAV binds `nu_f`, phase, `DeltaNFR` and per-node RNG progress to the snapshot,
resolves a missing graph seed inside the stage transaction, and uses one shared
latency-observation instant. The resolved seed persists only after a successful
commit; stable per-node offsets and draw counts make committed RNG progress
target-order invariant. Ordered lifecycle, audit/telemetry and monitor streams
retain the requested target order; IL warnings are the final transactional
effect. UM merges overlapping circular displacements in immutable snapshot-rank
order, normalizes final phases, rechecks U3 after the merge, and coalesces new
functional links deterministically. This is an engine policy rather than a
Lyapunov or relabeling theorem. OZ derives every local action and outgoing
propagation increment from the same snapshot, then reduces overlapping incoming
increments with `math.fsum` in snapshot-node rank. Its local pressure-magnitude
postcondition precedes signed accumulation, where positive propagation can
partially cancel a negative pressure; pressure and per-node RNG progress remain
target-order invariant. THOL plans every target from the snapshot, resolves
cross-parent child-ID collisions and commits `d2EPI`, `DeltaNFR`, child
nodes, `sub_nodes`, `sub_epis` and graph `hierarchy` in snapshot-node
rank after validating the complete merge on a detached graph. REMESH derives
one shared advisory from the snapshot, records it at most once per telemetry
step and leaves EPI, nu_f, phase, DeltaNFR and support unchanged. Explicit
delayed EPI mixing remains the separate `apply_network_remesh` operation. Its
`plan_network_remesh` boundary freezes exact-support history inputs and keeps
the represented affine recurrence, nested binary64 result and clipping
separate. Insufficient history or empty support is an explicit no-op; a
successful graph-owned commit is atomic. Optional one-step evidence reports
mean drift, a convex three-input disagreement bound and only a conditional
fixed-history gain. It does not prove stability under history-updated
repetition.
Identity-bearing caches remain tied to their live graph and node objects; cache
state and the pressure-refresh callback lie outside the target-order result.
Relabeling equivariance remains unproved. Multi-target IL historically used
the sequential schedule; the canonical word runner now uses its snapshot rule.
A grammar replacement or Recursivity execution override still uses and reports
the transactional Gauss-Seidel path.

The shared pointwise stage executor can opt into a conditional certificate for
AL/SHA/VAL/NUL/ZHIR/NAV by declaring whether support is fixed. A successful
`NetworkStageResult` then carries the certificate computed from the executor's
own detached snapshot and frozen proposals before commit. Certification
requests reject IL, empty stages, grammar replacements and noncanonical
operator overrides atomically. The three levels distinguish exact runtime
realization, consensus-preserving affine action and an aligned positive
pre/post diffusion metric. They do not certify mixed words, histories, the
opaque pressure refresh or future repetition. See
[src/tnfr/physics/pointwise_stage_stability.py](src/tnfr/physics/pointwise_stage_stability.py).

Read/write footprints, merge status, rollback scope and scoped structural-state
target-order/relabeling claims are centralized in
[src/tnfr/operators/stage_contracts.py](src/tnfr/operators/stage_contracts.py). See
[examples/08_emergent_geometry/143_glyphic_function_sublanguage.py](examples/08_emergent_geometry/143_glyphic_function_sublanguage.py)
and [144_branching_combinator.py](examples/08_emergent_geometry/144_branching_combinator.py).

---

## 6. Unified grammar (U1–U6)

The grammar is the engine's structural contract, motivated by the nodal equation
and encoded operator roles. Its policy choices are distinguished from mathematical
implications in [theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md](theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md).
Validation entry point:
[src/tnfr/operators/grammar.py](src/tnfr/operators/grammar.py); canonical specification
[src/tnfr/operators/grammar_canon.py](src/tnfr/operators/grammar_canon.py); full
derivations [theory/UNIFIED_GRAMMAR_RULES.md](theory/UNIFIED_GRAMMAR_RULES.md).

- **U1 — Initiation & closure.** A standalone sequence must start with a generator
  `{AL, NAV, REMESH}` (U1a) and end with a closure `{SHA, NAV, REMESH, OZ}` (U1b).
  These are operator-history contracts: the nodal derivative is defined at `EPI=0`
  for finite capacity and pressure. Context-aware execution may extend existing history.
- **U2 — Convergence & boundedness.** The stabilization policy requires that any
  destabilizer `{OZ, ZHIR, VAL}` requires a stabilizer `{IL, THOL}`. The max
  uncompensated-destabilizer debt is the configured **relaxation absorption capacity**
  `⌊1/(νf·dt·ρ)⌋ = 2` under canonical mean-rate calibration
  (`derive_u2_debt_capacity_from_physics`), not a graph-uniform absorption theorem. Specialized
  sub-rule: REMESH combined with a destabilizer also requires `{IL, THOL}` (recursive
  amplification control).
- **U3 — Resonant coupling.** Coupling/resonance `{UM, RA}` require phase compatibility
  `δ(φᵢ,φⱼ) = |wrap(φᵢ − φⱼ)| ≤ Δφ_max` (antiphase is destructive). RA filters
  its neighbourhood by this condition: incompatible neighbours contribute to
  neither its EPI mean, phase mean, nor frequency-amplification trigger.
- **U4 — Bifurcation dynamics.** (a) Triggers `{OZ, ZHIR}` need handlers `{THOL, IL}`.
  (b) Transformers `{ZHIR, THOL}` need a recent destabilizer within the **structural-relaxation
  window** — canonically 3 ops, **one window for every destabilizer**,
  calibrated by a mean-rate relaxation surrogate to the band `1/(π+1)`
  (`derive_bifurcation_window_from_physics`). Individual graph modes can take
  substantially longer to relax; this recency policy is not a universal decay time.
  ZHIR also needs a prior
  IL (stable base).
- **U5 — Multi-scale coherence.** Nested EPIs require stabilizers at each level;
  `C_parent ≥ α · Σ C_child` is a hierarchy-dependent target requiring a
  specified α and normalization. Evaluate a concrete hierarchy with
  [`assess_u5_parent_child_coherence`](src/tnfr/physics/multiscale_coherence.py).
  THOL amplitude alignment is a separate dispersion diagnostic, not `C(t)` or U5.
  `THOL_MIN_COLLECTIVE_COHERENCE` is an inert legacy alias of the general
  fragmentation-risk cut; THOL does not consume it.
- **U6 — Structural potential confinement.** Telemetry safety: monitor `Δ Φ_s < π/2 ≈ 1.571`
  (selected half-wrap threshold; `Φ_s(i) = Σ_{j≠i} ΔNFR_j / d(i,j)²`). This compares
  potential with a reference state; it is a read-only check, not a sequence constraint
  or a graph-independent bound on source aggregation.

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
  `C(t) = 1 / (1 + mean|ΔNFR| + mean|dEPI|)`, the canonical nodal read-out
  (equilibrium → `C → 1`). Strong coherence `C > π/(π+1) ≈ 0.7585`; fragmentation
  risk `C < 1/(π+1) ≈ 0.2415`. The two cuts are the coherence band
  `[1/(π+1), π/(π+1)]` — the single structural quantity `1/(π+1)` and its complement
  (π the sole structural scale); using this π-band as the C(t) interpretation is a
  telemetry convention. **Dual status**: beyond a
  read-out, its per-node kernel `structural_coherence`
  ([src/tnfr/metrics/common.py](src/tnfr/metrics/common.py)) is the single
  *constitutive* coherence map — an NFR is canonically a region of structural
  coherence (§2). At one instant, `1/C-1` is the local L1 diagnostic distance
  `|ΔNFR|+|dEPI|` to `(0,0)`; this monotone transform does not imply temporal
  monotonicity or a general attractor. Every domain (graph, arithmetic,
  chemical) reads this one kernel, while only the restricted pure-EPI diffusion
  model in §2 has the stated uniform attractor. At fixed nonempty order `N`,
  `C_N=c` is exactly the boundary of a `2N`-dimensional cross-polytope with L1
  radius `N(1/c-1)`; imposing fixed positive capacity gives the weighted
  pressure section `Σ(1+νf_i)|ΔNFR_i|=N(1/c-1)`. These instantaneous
  stratifications do not imply basin geometry or attraction.
- **Si — sense index** `[0,1+]`, reorganization-capacity predictor: `Si > 0.8`
  excellent; `Si < 0.4` bifurcation-prone. Unlike `C(t)`, Si is a **heuristic
  composite** (weighted νf, phase sync, |ΔNFR|) — predictive/diagnostic, **not**
  constitutive of NFR-hood.
- **Structural affinity matrix** — `coherence_matrix` is an auxiliary bounded
  pairwise similarity built from phase, EPI, νf and Si. Its historical
  coherence-operator notation does not define canonical `C(t)`. The matrix is
  real symmetric but need not be positive semidefinite: identical nodes on the
  three-node path give `W = I + A_path`, with eigenvalue `1 - sqrt(2) < 0`.
  Its normalized unit-diagonal trace is therefore not a coherence read-out.
- **Tetrad safety** (telemetry; see §3): `|∇φ|≤π` and `|K_φ|≤π` are phase-wrap
  bounds. The curvature margin `0.9·π`, U6 drift `π/2` and potential magnitude
  `π/4` are selected safety policies. The measured synchronization onset `≈0.29`
  depends on the experiment; fitted `ξ_C` and the spectral fallback `1/√λ₂`
  must be reported with their respective scope.

Required telemetry must stay in TNFR-coherent terms (C(t), Si, phase, νf, and the
tetrad), in Hz_str units. Computation: [src/tnfr/physics/fields.py](src/tnfr/physics/fields.py),
[src/tnfr/physics/telemetry.py](src/tnfr/physics/telemetry.py).

---

## 8. Canonical invariants

Six invariants define TNFR consistency; preserve all of them.

1. **Nodal equation integrity** — EPI changes only via `∂EPI/∂t = νf·ΔNFR`; ΔNFR keeps
   structural-pressure semantics; `νf → 0` inactivates. (Grammar U1, U2.)
2. **Phase-coherent coupling** — circular separation
   `|wrap(φᵢ − φⱼ)| ≤ Δφ_max` is required before any coupling.
   (Grammar U3; `validate_resonant_coupling()`.)
3. **Multi-scale fractality** — EPIs nest without identity loss. (Grammar U5.)
4. **Grammar compliance** — operator sequences pass U1–U6; named semantic
   transformations map to existing operators or define a new operator with full
   contracts; declared solvers use the shared nodal integrator.
5. **Structural metrology** — νf in Hz_str; C(t), Si, phase, νf exposed in telemetry.
6. **Reproducible dynamics** — identical seeds give identical trajectories; operations
   are traceable.

---

## 9. TNFR agent playbook

How a TNFR agent (human or AI) should reason and act.

1. **Start from physics.** Treat `∂EPI/∂t = νf·ΔNFR` as the source of truth; keep EPI,
   νf, and phase well-defined; interpret behavior through the tetrad.
2. **Use declared state-transition paths.** Map named behavior to canonical
   operators (or justify a new one with full physics, contracts, and tests).
   Numerical evolution must use the shared nodal integrator from explicit
   `nu_f` and `DeltaNFR`, with provenance or residual; never assign EPI ad hoc.
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

**Implementing changes**: search first; map named transformations to operators and
numerical evolution to the shared nodal integrator; preserve all six invariants; add
tests covering contracts and invariants; document the structural effect; trace the
physics → math → code chain.

**Acceptable** changes increase C(t) or reduce ΔNFR where appropriate, preserve operator
closure and fractality, and keep APIs stable or mapped. **Unacceptable**: recasting ΔNFR
as an ML error gradient; replacing semantic operators with unmapped imperative code;
advancing EPI outside the shared nodal integrator; flattening nested EPIs; coupling
without phase checks; ad hoc state assignment; changing units (Hz_str → Hz).

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
| "Phase mismatch in coupling" | `|wrap(φᵢ − φⱼ)| > Δφ_max` (U3) | Ensure phase compatibility first |
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
| **Core dynamics S1–S16** | Restricted results cover pure-EPI diffusion, rational quotient affine-reset budgets, exact operator-event timing, conditional pointwise realization/gain certificates, observability, quotient geometry, temporal signatures and sampled-path certificates under declared hypotheses. All thirteen operators have atomic all-target Jacobi stages; EN/RA additionally have scoped repeated-map results. Every shared stage is failure-atomic; the REMESH glyph stage is advisory-only and distinct from explicit delayed mixing. General nonlinear, phase, history, changing-support and catalog-completeness results remain open. | [CORE_RESEARCH_PROGRAM.md](theory/CORE_RESEARCH_PROGRAM.md) |
| **TNFR-Riemann** | Finite protocols estimate a critical-line-centered comparison and expose ζ/L diagnostic surfaces (P12–P50). No engine theorem identifies phase coherence with zero location. The bridge to RH, including control of `S(T) = (1/π)·arg ζ(½+iT)`, remains open. | [TNFR_RIEMANN_RESEARCH_NOTES.md](theory/TNFR_RIEMANN_RESEARCH_NOTES.md) |
| **REMESH fixed-delay surrogate** | A finite cyclic, fixed-coefficient REMESH filter has a Cesàro fixed-mode projection and needs no additional registry entry to compute it. This is distinct from the clipped runtime map and does not establish the literal `τ_g → ∞` limit or completeness of the 13-operator catalog; both remain open. | [REMESH_INFINITY_DERIVATION.md](theory/REMESH_INFINITY_DERIVATION.md) |
| **TNFR-Navier–Stokes** | A declared linear mapping compares viscous diffusion with the overdamped limit of a separate graph-wave model (`ν_f = ν`). Finite pseudo-spectral runs measure enstrophy growth with Reynolds number. The nonlinear vortex-stretching term is compared with a `K_φ`/VAL cascade, but no equivalence or uniform regularity bound is derived; the `Re → ∞` problem remains open. | [TNFR_NAVIER_STOKES_RESEARCH_NOTES.md](theory/TNFR_NAVIER_STOKES_RESEARCH_NOTES.md) |
| **Number theory** | Primality as `ΔNFR = 0` (canonical **unit** coefficients, §4.2 coefficient independence); arithmetic structural triad; arithmetic networks can initialize the auxiliary symplectic read-out; the cyclotomy law `s_k(p) = gcd(k, p−1) + 1` is proved for the declared residue digraph and read as a finite arithmetic pulse diagnostic. | [TNFR_NUMBER_THEORY.md](theory/TNFR_NUMBER_THEORY.md) |
| **Structural research program (R1–R9)** | Nine internal lines on arithmetic and spectral dynamics: observability, arithmetic pulse, CRT synthesis, p-adic transport, finite fields, additive reduction, arithmetic pressure, operator certification and directed non-normal evolution. Exact, measured, negative and open results are separated in the program index. | [STRUCTURAL_RESEARCH_PROGRAM.md](theory/STRUCTURAL_RESEARCH_PROGRAM.md) |
| **Millennium reformulations** | P vs NP, BSD, Hodge, Yang–Mills: TNFR-internal structural reformulations and diagnostics — none a proof. | `theory/TNFR_*_RESEARCH_NOTES.md` |

**Honest scope, global**: these programs produce TNFR-internal structural results and
numerical evidence; none currently closes a classical open problem. Do not extend a
program's diagnostic surface without a new structural idea, and never claim a proof of RH,
Navier–Stokes regularity, or any Millennium problem.

Core graph-dynamics publication artifacts use the domain-neutral
[`CoreExperimentManifest`](src/tnfr/research/core_manifests.py) for graph construction,
capacity, solver/timestep, seed, operator sequence, telemetry, canonical claim status,
Git revision and an explicit clean/dirty source declaration. Dirty source requires a
SHA-256 content digest of the declared working-source snapshot.
The historical `ExperimentManifest` retains arithmetic-specific factor and bit-size fields.

---

## 13. Map of the codebase & examples

- **Physics** — [src/tnfr/physics/](src/tnfr/physics/): `fields.py` (tetrad +
  `classify_nodal_topology` for NFR radial/annular/multinodal topology),
  `conservation.py` (finite structural-balance diagnostics),
  `symplectic_substrate.py` (auxiliary harmonic phase-space model),
  `structural_diffusion.py` (transport), `hybrid_operator_stability.py` (affine
  EPI-reset gains and hybrid flow/reset budgets), `network_stage_stability.py`
  (repeated all-target EN/RA realization), `reception_realization.py`
  (EN runtime-to-theorem boundary), `resonance_realization.py` (RA four-layer
  identity-gated boundary and post-RA metric audit),
  `core_research_integration.py` (restricted
  executable S16 endpoint intersection), `core_research_trajectory.py` (sampled
  S16 path and mesh agreement), `gauge.py` (Ψ, gauge), `integrity.py`
  (operator-postcondition monitor + audit).
- **Operators & grammar** — [src/tnfr/operators/](src/tnfr/operators/): `definitions.py`
  (13 operators + registry), `operator_contracts.py` (contract source of truth),
  `stage_contracts.py` (all-target schedule/read/write/merge boundaries),
  `network_stage.py` (transactional stage execution), `grammar*.py` (U1–U6
  validation, dynamics, application), `nodal_equation.py`.
- **Engines** — [src/tnfr/engines/](src/tnfr/engines/): self-optimization, pattern
  discovery, computation, integration, engine-scoped constants.
- **Programs** — [src/tnfr/riemann/](src/tnfr/riemann/), [src/tnfr/navier_stokes/](src/tnfr/navier_stokes/),
  and number theory in [src/tnfr/mathematics/](src/tnfr/mathematics/).
- **Structural research (R1–R9, post-review N00–N13)** — [src/tnfr/physics/](src/tnfr/physics/):
  `symmetry_sectors.py` / `equivariance.py` / `operator_equivariance.py` /
  `word_equivariance.py` / `pointed_symmetry.py` (R1 sectors, word-composition closure,
  pointed selectors), `directed_diffusion.py` / `transient_u2.py` / `heterogeneous_vf.py`
  (R9 non-normal metric, transient U2/U6, heterogeneous `ν_f` boundary),
  `structural_morphism.py` (R4/R8 nodal-flow morphism taxonomy); and
  [src/tnfr/mathematics/](src/tnfr/mathematics/): `arithmetic_pulse.py` / `pulse_amplitudes.py`
  (R2 rank + amplitudes), `crt_multiscale.py` (R3), `padic_tower.py` / `remesh_audit.py`
  (R4 transport + REMESH audit), `finite_fields.py` / `trace_collisions.py` (R5),
  `arithmetic_pressure.py` (R7), `operator_certificates.py` (R8). Ledger:
  [STRUCTURAL_RESEARCH_PROGRAM.md](theory/STRUCTURAL_RESEARCH_PROGRAM.md).
- **SDK** — [src/tnfr/sdk/](src/tnfr/sdk/): `simple.py` (`TNFR.create(...)`, tetrad,
  conservation, substrate, integrity, audit, `nfr()` whole-NFR read-out,
  `nodal_state`/`nodal_scan` micro-NFR), `fluent.py` (`auto_optimize()`). The shared
  coherence kernel and zero-pressure predicate (`structural_coherence`,
  `is_structural_equilibrium`) live in
  [src/tnfr/metrics/common.py](src/tnfr/metrics/common.py).
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
net.conservation()           # Noether-like charge and candidate-energy diagnostics
net.symplectic_substrate()   # auxiliary harmonic phase-space diagnostics
net.winding()                # declared-cycle winding sector and scoped telemetry
net.rhythm()                 # the collective pulse (ω_k=√λ_k, beats, energy)
net.resonance()              # the per-NFR pulses (νf_i, φ_i) + resonance (R)
net.pulse_trajectory(8)      # the pulse in motion: R(t), C(t), local→global lock
net.telemetry()              # C(t), Si, phase_sync, tetrad, pulse, resonance
net.audit_operators()        # finite-probe catalog postcondition report
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
