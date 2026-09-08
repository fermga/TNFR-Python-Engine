# TNFR Glossary

**Purpose**: Operational quick reference for the Resonant Fractal Nature Theory (TNFR)
**Status**: Canonical reference, aligned with the current engine and TNFR.pdf
**Version**: 0.0.3.5 (June 2026)
**Authority**: [AGENTS.md](../AGENTS.md) is the single source of truth; this glossary mirrors it API-first

**Scope**: API-focused definitions for developers implementing TNFR networks — the
nodal equation, the structural triad and the **fractal-resonant node (NFR)**, the
structural-field tetrad, the 13 operators and the unified grammar (U1–U6). For full
derivations see [AGENTS.md](../AGENTS.md), [FUNDAMENTAL_THEORY.md](FUNDAMENTAL_THEORY.md)
and [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md).

---

## Core Variables

### Fractal-Resonant Node (NFR)

**What:** **Nodo Fractal Resonante** — a *region of structural coherence coupled to a
network* (TNFR.pdf §1.4.1), the fundamental entity of TNFR. The structural triad
(EPI, νf, φ) **defines** it; it is read out as a whole by `Network.nfr()`.
**Properties:** **multiscalar** (an NFR can nest other NFRs — operational fractality),
**autopoietic** (emerges by local reorganization, no external support), **relational**
(exists only by coupling) and **temporal** (persists while it reorganizes its coherence).
**Nodal topology:** **radial** (one central nucleus), **annular** (passive center,
peripheral ring) or **multinodal** (several centers), classified from the
structural-potential geometry by `classify_nodal_topology(G)`.
**Equilibrium:** `ΔNFR = 0` is a fixed-point condition, not an NFR.  For the
restricted pure-EPI model on a fixed connected symmetric graph with positive
capacity, diffusion converges to a uniform field and `C → 1`. Other pressure
realizations do not inherit that restoring-attractor result. Scale-relative: a
single node is a micro-NFR and a coherent region is a macro-NFR.
**API:** `tnfr.physics.fields.classify_nodal_topology`, `tnfr.sdk.simple.Network.nfr`,
`tnfr.structural.create_nfr`
**Theory:** [AGENTS.md §2 (The fractal-resonant node)](../AGENTS.md), TNFR.pdf §1.4.1

### Primary Information Structure (EPI)

**Code:** `G.nodes[n]['EPI']`, `ALIAS_EPI`
**Symbol:** \(\text{EPI}\) or \(E\)
**What:** Coherent structural form of a node
**Space:** \(B_{\text{EPI}}\) (Banach space)
**Rules:** Modified only via structural operators, never directly
**Scalar representation:** Raw real EPI and the uniform-real `BEPIElement`
embedding represent the same signed graph coordinate. `real_scalar_epi(value)`
returns that coordinate or `None`; `scalarize_epi(value)` preserves it and uses
maximum-component magnitude only for genuinely nonuniform or complex elements.
`abs(BEPIElement)` is always the nonnegative magnitude. Glyphs that require a
real scalar EPI coordinate, scalar-only diffusion and affine certificates reject
richer elements rather than silently discarding their component structure.
**API:** `tnfr.structural` operators; `tnfr.types.real_scalar_epi`,
`tnfr.types.scalarize_epi`
**Math:** [FUNDAMENTAL_THEORY.md §2.2 (Structural Triad — Banach space B_EPI)](FUNDAMENTAL_THEORY.md)

### Structural Frequency (νf)

**Code:** `G.nodes[n]['vf']`, `ALIAS_VF`
**Symbol:** \(\nu_f\)
**Units:** Hz_str (structural hertz)
**Range:** \(\mathbb{R}^+\) (positive reals; node collapse when \(\nu_f \to 0\))
**What:** Rate of structural reorganization
**API:** `adapt_vf_after_structural_stability()`; the historical
`adapt_vf_by_coherence()` name remains a compatibility alias; operators
**Math:** [FUNDAMENTAL_THEORY.md §2 (Governing Dynamics)](FUNDAMENTAL_THEORY.md)

### Internal Reorganization Operator (ΔNFR)

**Code:** `G.nodes[n]['dnfr']`, `ALIAS_DNFR`
**Symbol:** \(\Delta\text{NFR}\)
**What:** Structural reorganization **pressure** — the gradient driving evolution.
**Four gradient channels:** \(\Delta\text{NFR} = w_\phi\,\partial\phi + w_E\,\partial\text{EPI} + w_{\nu}\,\partial\nu_f + w_\tau\,\partial\text{topo}\) (phase desync, EPI gradient, νf gradient, topology). Operational default weights (tunable, free parameters) `DNFR_WEIGHTS = {phase ≈ 0.737, epi ≈ 0.155, vf ≈ 0.090, topo = 0.0}` in [config/defaults_core.py](../src/tnfr/config/defaults_core.py); normalized and applied by `_configure_dnfr_weights` in [dynamics/dnfr.py](../src/tnfr/dynamics/dnfr.py).
**EPI channel = graph diffusion (KEY):** the EPI channel is exactly the
random-walk graph Laplacian,
\(\Delta\text{NFR}_\text{epi}= -L_\text{rw}\,\text{EPI}\), hence
\(\dot{\text{EPI}}=-\operatorname{diag}(\nu_i)L_\text{rw}\,\text{EPI}\).
On a fixed symmetric nonnegative graph with homogeneous positive capacity
\(\nu_i=\nu_f\), modes decay as \(e^{-\nu_f\lambda_k t}\) and
\(\sum_i d_i\,\text{EPI}_i\) is conserved. With fixed heterogeneous
positive capacities, the conserved quantity is
\(\sum_i(d_i/\nu_i)\,\text{EPI}_i\), and the rates are the eigenvalues
of \(\operatorname{diag}(\nu_i)L_\text{rw}\), rather than
\(\nu_f\lambda_k\). In both cases, pressure equilibrium is constant on
each connected component; it is one uniform field only when the graph is
connected. Isolates have zero Laplacian rows.
**Sign:** positive = expansion, negative = contraction.
**Compute:** `default_compute_delta_nfr` hook, automatic in `step()`.
**Math:** [FUNDAMENTAL_THEORY.md §2.1](FUNDAMENTAL_THEORY.md), [AGENTS.md §2 (Transport content)](../AGENTS.md), [src/tnfr/physics/structural_diffusion.py](../src/tnfr/physics/structural_diffusion.py)

### Phase (φ, θ)

**Code:** `G.nodes[n]['theta']`, `collect_theta_attr()`
**Symbol:** \(\theta\) or \(\phi\)
**Range:** \([0, 2\pi)\) or \([-\pi, \pi)\) radians
**What:** Network synchrony parameter (relative timing)
**Phase difference:** \(\Delta\theta = \theta_i - \theta_j\)
**API:** Phase adaptation in dynamics
**Math:** [FUNDAMENTAL_THEORY.md §2.2 (Structural Triad — phase)](FUNDAMENTAL_THEORY.md)

### Total Coherence (C(t))

**Code:** `compute_coherence(G)` → float ∈ [0,1]; per-node kernel `structural_coherence(dnfr, depi)`
**Symbol:** \(C(t)\)
**Formula:** \(C(t) = 1/(1 + \overline{|\Delta\text{NFR}|} + \overline{|d\text{EPI}|})\) (canonical; derived from the nodal equation — equilibrium \(\Delta\text{NFR}\to 0 \wedge d\text{EPI}\to 0 \Rightarrow C\to 1\))
**Range:** \([0, 1]\) where 1 = perfect coherence, 0 = total fragmentation
**What:** Global stability measure (recorded in `history['C_steps']`). **Dual status:** beyond a telemetry read-out, its per-node kernel `structural_coherence` (= \(1/(1+|\Delta\text{NFR}|+|d\text{EPI}|)\)) is the constitutive coherence map for graph dynamics and is reused by the arithmetic triad. The assumption-explicit shell model reuses only the scalar zero-pressure predicate; this does not identify the domains' state spaces or dynamics. An NFR *is* a region of structural coherence, so \(C\) measures the coherence that **defines** NFR-hood. `compute_coherence` delegates to this kernel.
**Thresholds:** strong \(C > \pi/(\pi+1) \approx 0.7585\); fragmentation risk \(C < 1/(\pi+1) \approx 0.2415\) (the coherence band; π the sole structural scale).
**Code:** [src/tnfr/metrics/common.py](../src/tnfr/metrics/common.py) (`compute_coherence`, `structural_coherence`)
**Math:** [FUNDAMENTAL_THEORY.md §5.1](FUNDAMENTAL_THEORY.md), [AGENTS.md §7](../AGENTS.md)

### Structural Equilibrium (ΔNFR = 0)

**Code:** `is_structural_equilibrium(dnfr, depi=0, *, eps_dnfr, eps_depi)` → bool
**What:** The shared numeric **fixed-point predicate**: \(|\Delta\text{NFR}| \le\)
`eps_dnfr` and \(|d\text{EPI}| \le\) `eps_depi` (default
`EPS_DNFR_STABLE = 1e-3`). Graph, arithmetic and chemical models may reuse this
predicate for their own pressure fields; this does not make their dynamics,
basins or attractors identical. The tolerance is declared per domain (1e-12
for exact integer arithmetic).
**Code:** [src/tnfr/metrics/common.py](../src/tnfr/metrics/common.py) (`is_structural_equilibrium`)
**Theory:** [AGENTS.md §2, §7](../AGENTS.md)

### Structural affinity matrix (historical Ĉ interface)

**Code:** `coherence_matrix(G)` → `(nodes, W)`
**Symbol:** \(W=(w_{ij})\); older APIs call it \(\hat C\)
**What:** Auxiliary bounded pairwise similarity built from phase, EPI,
\(\nu_f\) and Si. It is distinct from constitutive coherence \(C(t)\).
**Properties:** The implementation produces a real symmetric matrix with
entries in \([0,1]\), including a symmetric support mask for directed inputs.
It is not positive semidefinite in general. For three identical nodes on a
path under neighbour scope, \(W=I+A_{P_3}\) has eigenvalues
\(1,1-\sqrt2,1+\sqrt2\). Its normalized trace is therefore not a definition
of \(C(t)\); with the default unit diagonal it is identically one.
**Math:** [src/tnfr/metrics/coherence.py](../src/tnfr/metrics/coherence.py)
(`coherence_matrix`)

### Sense Index (Si)

**Code:** `G.nodes[n]['Si']`, `ALIAS_SI`, `compute_Si_node()`
**Symbol:** \(\text{Si}\) (global) or \(S_i\) (node i)
**Formula:** \(\text{Si} = \alpha \cdot \nu_{f,\text{norm}} + \beta \cdot (1 - \text{disp}_\theta) + \gamma \cdot (1 - |\Delta\text{NFR}|_{\text{norm}})\)
**Range:** \([0, 1^+]\) typically, higher = more stable reorganization
**What:** Reorganization-capacity predictor. Unlike `C(t)`, Si is a **heuristic composite** (weighted νf, phase sync, \(|\Delta\text{NFR}|\)) — predictive/diagnostic, **not** constitutive of NFR-hood. `Si > 0.8` excellent; `Si < 0.4` bifurcation-prone.
**Weights:** operational defaults \(\alpha \approx 0.737\), \(\beta \approx 0.155\), \(\gamma_w \approx 0.114\) (`SI_WEIGHTS` in `config/defaults_core.py`; free parameters, sum \(\approx 1\))
**Math:** [Mathematical Foundations - Metrics](MATHEMATICAL_DYNAMICS_BASIS.md)

### Phase Gradient (|∇φ|) - CANONICAL

**Code:** `compute_phase_gradient(G)` → Dict[NodeId, float]
**Symbol:** \(|\nabla\phi|(i)\)
**Formula:** \(|\nabla\phi|(i) = \text{mean}_{j \in N(i)} |\theta_i - \theta_j|\) (circular mean)
**What:** Local phase desynchronization / stress proxy field
**Status:** **CANONICAL** (Nov 2025)
**Physics:** Locates phase stress that the global scalar C(t) does not spatially resolve
**Threshold:** Kinematic bound |∇φ| ≤ π (phase wrap — same as K_φ); π/16 ≈ 0.196 is the selected early-warning policy, not a derived bound (measured sync-onset ≈ 0.29, σ-dependent)
**API:** `tnfr.physics.fields.compute_phase_gradient()`
**Usage:** Stress detection, local instability prediction
**Documentation:** [docs/STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)

### Phase Curvature (K_φ) - CANONICAL

**Code:** `compute_phase_curvature(G)` → Dict[NodeId, float]
**Symbol:** \(K_\phi(i)\)
**Formula:** \(K_\phi = \text{wrap\_angle}(\phi_i - \text{circular\_mean}(\text{neighbors}))\)
**What:** Phase torsion and geometric confinement field
**Status:** **CANONICAL** (Nov 2025)
**Physics:** Flags mutation-prone loci via geometric constraints
**Threshold:** Classical bound |K_φ| < 2.8274 (90% of π theoretical maximum)
**API:** `tnfr.physics.fields.compute_phase_curvature()`
**Usage:** Geometric confinement monitoring, bifurcation prediction
**Documentation:** [docs/STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)

### Coherence Length (ξ_C) - CANONICAL

**Code:** `estimate_coherence_length(G)` → float
**Symbol:** \(\xi_C\)
**Formula:** Spatial correlation function \(C(r) = A \exp(-r/\xi_C)\)
**What:** Spatial correlation scale of local coherence
**Status:** **CANONICAL** (Nov 2025)
**Physics:** Critical phenomena and finite-size scaling analysis
**Thresholds:**
- Critical: ξ_C > 1.0 × diameter (finite-size scaling dominates)
- Watch: ξ_C > π ≈ 3.14 × mean_distance (RG scaling)
- Stable: ξ_C < mean_distance (bulk behavior)
**API:** `tnfr.physics.fields.estimate_coherence_length()`
**Usage:** Critical point detection, correlation analysis
**Documentation:** [docs/STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)

---

### Structural Potential (Φ_s) - CANONICAL

**Code:** `compute_structural_potential(G, alpha=2.0)` → Dict[NodeId, float]
**Symbol:** \(\Phi_s(i)\)
**Formula:** \(\Phi_s(i) = \sum_{j \neq i} \frac{\Delta\text{NFR}_j}{d(i,j)^\alpha}\) where \(\alpha = 2\)
**What:** Global structural potential field from ΔNFR distribution
**Status:** **CANONICAL** (Nov 2025)
**Validation:** Structural-potential and tetrad tests across finite graph topologies
**Physics:** Distance-weighted pressure aggregation
**Grammar:** U6 monitors potential drift against the selected π/2 safety policy
**API:** `tnfr.physics.fields.compute_structural_potential()`
**Threshold:** |Φ_s| < π/4 ≈ 0.785 is the selected per-node warning policy
**Documentation:** [docs/STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)
- [src/tnfr/physics/fields.py](../src/tnfr/physics/fields.py) - Implementation

**Interpretation:** Φ_s depends on both pressure and graph geometry. The π/4
per-node and π/2 drift values are selected monitoring policies, not universal
bounds. Crossing either value records a warning and does not by itself prove
fragmentation or grammar failure.

---

## The Nodal Equation

**The fundamental equation of TNFR** governs structural evolution:

\[
\frac{\partial \text{EPI}}{\partial t} = \nu_f \cdot \Delta\text{NFR}(t)
\]

**Where:**
- \(\frac{\partial \text{EPI}}{\partial t}\): Rate of change of structure
- \(\nu_f\): Structural frequency (reorganization rate) in Hz_str
- \(\Delta\text{NFR}(t)\): Reorganization gradient (driving pressure)

**Interpretation:**
- Structure changes **only when** both \(\nu_f > 0\) (capacity) and \(\Delta\text{NFR} \neq 0\) (pressure) exist
- Rate of change is **proportional** to both frequency and gradient
- When \(\nu_f \to 0\), evolution freezes (node collapse)
- When \(\Delta\text{NFR} = 0\), structure reaches equilibrium

**Implementation:** See `src/tnfr/dynamics/` for numerical integration
**Theory:** [Nodal equation](FUNDAMENTAL_THEORY.md) §2

---

## Structural Diffusion & Spectral Parameters

For a fixed symmetric nonnegative conductance graph, the EPI channel of
ΔNFR is a graph diffusion. The homogeneous and heterogeneous capacity models
have different invariants and relaxation generators; the table marks that
scope explicitly.

| Factor | Symbol | Meaning | API |
| --- | --- | --- | --- |
| Diffusivity | νf | reorganization rate / mobility (Hz_str) | `structural_diffusivity(G)` |
| Spectral gap (Fiedler value) | λ₂ | smallest positive Laplacian eigenvalue on a connected graph; with homogeneous capacity it sets the slowest nonconstant decay | `relaxation_spectrum(G)` |
| Homogeneous relaxation rate | r_k = νf·λ_k | decay rate of mode k for fixed common capacity (amplitude ∝ e^{−r_k t}) | `relaxation_spectrum(G)` |
| Linear instability threshold | r_c = νf·λ₂ | in the declared homogeneous growth-diffusion model $\sigma_k=r-\nu_f\lambda_k$, the first nonuniform mode changes sign here; this is not U2 itself or a fragmentation theorem | `instability_threshold(G)` |
| Coherence length | ξ_C | state-dependent correlation fit; `1/√λ₂` is a connected-graph spectral comparison/fallback | `estimate_coherence_length(G)` |
| Fiedler partition | — | spectral two-way partition/proxy; it need not be the globally optimal cut | `fiedler_partition(G)` |
| Structural rank | — | number of distinct relaxation frequencies | `structural_frequency_rank(G)` |
| Homogeneous invariant | Σ d_i·EPI_i | conserved for fixed common positive capacity | `degree_weighted_total(G)` |
| Heterogeneous invariant | Σ (d_i/ν_i)·EPI_i | conserved for fixed positive node capacities; not returned by `degree_weighted_total` | theorem-level read-out |
| Pulse resonance | ω_k = √λ_k | the conservative-face standing-wave frequencies — the rhythm the substrate plays (collective) | `compute_emergent_pulse(G)` / `net.rhythm()` |
| Per-NFR pulse | (νf_i, φ_i) | each NFR a phase oscillator; resonance (`local_phase_sync`, Kuramoto `R`) couples them into the collective rhythm | `compute_nodal_pulse(G)` / `net.resonance()` |
| Relaxation window | min{n : qⁿ < 1/(π+1)} | scalar-surrogate calibration of the **U4b / repeat-avoidance policy** = 3 operator positions; not a graph-modal solver-step bound | `derive_bifurcation_window_from_physics()` |
| Debt capacity | ⌊1/(1−q)⌋ | scalar-surrogate calibration used by the selected U2 debt policy (=2); not a graph-modal capacity theorem | `derive_u2_debt_capacity_from_physics()` |

- **Pulse and policy calibration.** The auxiliary substrate has frequencies
  $\omega_k=\sqrt{\lambda_k}$. Separately, the U4b window (=3 positions) and U2
  debt (=2) use a scalar relaxation surrogate
  `q=1−νf·dt·ρ` with the selected defaults. They count operator positions/debt,
  not physical modal time. The graph-specific Euler diagnostic must be used for
  solver-step relaxation.
- The zero eigenspace of `L_rw` consists of fields that are constant on
  each connected component; a connected graph has one uniform zero mode. Under
  fixed symmetric homogeneous pure-EPI diffusion,
  **Σ d_i·EPI_i is invariant** (`degree_weighted_total`). Fixed positive
  heterogeneous capacity instead preserves $\sum_i(d_i/\nu_i)\,\text{EPI}_i$.
  The quantity
  $Q=\sum(\Phi_s+K_\phi)$ is a separate Noether-like diagnostic whose drift must
  be measured; grammar labels do not generally conserve it.
- Under fixed symmetric pure-EPI diffusion with positive capacities,
  equilibrium is componentwise constant. It is a single uniform field on a
  connected graph. This statement does not establish uniformity for general
  multichannel operator dynamics; differentiated pure-EPI nodal topology lives
  off-equilibrium.
- **API:** [src/tnfr/physics/structural_diffusion.py](../src/tnfr/physics/structural_diffusion.py), [src/tnfr/physics/conservation.py](../src/tnfr/physics/conservation.py).

---

## Key Parameters & Factors

The quantities that govern TNFR dynamics, with their canonical status.

| Parameter | Symbol | Default / value | Role | Status |
| --- | --- | --- | --- | --- |
| Structural frequency | νf | ℝ⁺ (Hz_str) | reorganization capacity = diffusivity/**mobility**; νf→0 inactivates | state |
| Reorganization pressure | ΔNFR | ℝ | drive (4 channels); ΔNFR=0 = equilibrium | state |
| Phase | φ, θ | [0, 2π) | synchronization | state |
| Phase-coupling tolerance | Δφ_max | π/2 ≈ 1.5708 rad (90°) | U3 admissible coupling \|φᵢ−φⱼ\| ≤ Δφ_max | canonical policy |
| Mutation threshold | ξ | `ZHIR_THRESHOLD_XI = 0.1` | Non-disableable ZHIR admission gate: a valid observed signed secant, timestamped in physical time or interpreted as a legacy unit-operator-step difference, must satisfy `observed dEPI/dt > ξ` | operational calibration |
| Equilibrium tolerance | eps_dnfr / eps_depi | EPS_DNFR_STABLE = 1e-3 | `is_structural_equilibrium` cut (1e-12 for exact arithmetic) | numerical scale |
| Spectral gap | λ₂ | graph-dependent | slowest homogeneous-diffusion relaxation; supplies the `1/√λ₂` comparison/fallback for ξ_C; r_c = νf·λ₂ in the stated linear instability model | structural |
| Phase scale | π | exact | the **one genuine structural constant**: bounds \|∇φ\| and \|K_φ\| | genuine |
| Non-structural parameters | — | free / derived | operator gains, clamps, dt, coupling rates — derived from the dynamics or free operational parameters | operational |

**Only π is a genuine structural constant** (the phase-wrap bound). φ, γ, e are **not**
structural scales and no longer appear in the engine. Other values are derived under
stated hypotheses (for example, the `1/√λ₂` spectral comparison for ξ_C), selected canonical policies (for
example, the π/2 ΔΦ_s warning threshold), or free operational parameters (for
example, the π/16 ≈ 0.196 |∇φ| early-warning policy).

---

## Structural Operators

The 13 canonical operators are the **only way** to modify nodes in TNFR. They're not arbitrary functions—they're **resonant transformations** with rigorous physics.

For complete specifications with physics derivations, contracts, and usage examples, see **[AGENTS.md § The 13 Canonical Operators](../AGENTS.md#5-the-13-canonical-operators)**.

### Quick Reference

| Symbol | Name | Physics | Grammar Sets | When to Use |
|--------|------|---------|-------------|-------------|
| **AL** | Emission | Creates EPI from vacuum via resonant emission | Generator (U1a) | Starting new patterns, initializing from EPI=0 |
| **EN** | Reception | Captures and integrates incoming resonance | - | Information gathering, listening phase |
| **IL** | Coherence | Stabilizes form through negative feedback | Stabilizer (U2) | After changes, consolidation |
| **OZ** | Dissonance | Introduces controlled instability | Destabilizer (U2), Bifurcation trigger (U4a), Closure (U1b) | Breaking local optima, exploration |
| **UM** | Coupling | Creates structural links via phase synchronization | Requires phase verification (U3) | Network formation, connecting nodes |
| **RA** | Resonance | Amplifies and propagates patterns coherently | Requires phase verification (U3) | Pattern reinforcement, spreading coherence |
| **SHA** | Silence | Freezes evolution temporarily (νf → 0) | Closure (U1b) | Observation windows, pause for synchronization |
| **VAL** | Expansion | Increases structural complexity (dim ↑) | Destabilizer (U2) | Adding degrees of freedom |
| **NUL** | Contraction | Reduces structural complexity (dim ↓) | - | Simplification, dimensionality reduction |
| **THOL** | Self-organization | Spontaneous autopoietic pattern formation | Stabilizer (U2), Handler (U4a), Transformer (U4b) | Emergent organization, fractal structuring |
| **ZHIR** | Mutation | Phase transformation at threshold | Bifurcation trigger (U4a), Transformer (U4b) | Qualitative state changes |
| **NAV** | Transition | Regime shift, activates latent EPI | Generator (U1a), Closure (U1b) | Switching between attractor states |
| **REMESH** | Recursivity | Echoes structure across scales | Generator (U1a), Closure (U1b) | Multi-scale operations, memory |

### Operator Composition

Operators combine into **sequences** that implement complex behaviors:

- **Bootstrap** = [Emission, Coupling, Coherence]
- **Stabilize** = [Coherence, Silence]
- **Explore** = [Dissonance, Mutation, Coherence]
- **Propagate** = [Resonance, Coupling]

**Critical**: All sequences must satisfy unified grammar (U1-U6).

For ZHIR, the nodal-equation quantity `predicted_depi_dt = nu_f * DeltaNFR` is
an **instantaneous prediction**. `predicted_crossed` compares that prediction
strictly with `xi`; the SDK's legacy `near_bifurcation` name is an alias of this
prediction. Neither value is the observed Mutation gate.

The **observed signed secant** comes from the last two EPI samples. Preferred
`epi_time_history` records `(time, EPI)` pairs and uses
`(EPI[k] - EPI[k-1]) / (t[k] - t[k-1])`. The interval must be finite and
strictly positive, and the final sample must be a fresh representation of the
current EPI endpoint. Supplied physical history is authoritative, so invalid or
stale records do not fall back to legacy evidence. Compatibility histories
`epi_history` and `_epi_history` instead use a unit **operator-step** interval;
they are marked `physical_time_resolved=False` and make no physical-time or
endpoint-freshness claim.

Direct ZHIR requires valid evidence with `observed_depi_dt > xi`, finite active
capacity (`nu_f > 0`), and any explicitly configured `ZHIR_MIN_VF`. Equality,
contraction, invalid or missing evidence, non-increasing physical time, and a
stale physical endpoint reject it. A dynamic selector that cannot support a
proposed ZHIR substitutes IL before ordinary grammar enforcement and records
the requested and applied glyphs with the reason in `mutation_abstentions`.
The SDK whole-word runner checks every target node before the word's first
operator and rejects physical evidence that an earlier EPI-channel operator in
the word would stale before ZHIR. Invalid or unavailable evidence is represented
by the tri-state `observed_crossed=None`; `rate_gap = observed - predicted` is
available only when valid physical timestamps make the two rates comparable.

`compute_d2epi_dt2` is a separate three-sample **structural-acceleration**
diagnostic: physical histories use adjacent timestamped secants, while legacy
histories use a unit-step second difference. It does not supply the two-sample
Mutation gate. The default
`ZHIR_BIFURCATION_VF_THRESHOLD = 0.5` controls whether a branch selector
proposes ZHIR; it is not the ZHIR threshold or an admission precondition.
Finally, `MutationTriggerCertificate` and SDK `nodal_state()` report evidence
and predictions only. They do not evaluate the prior-IL/recent-destabilizer
context and do not certify U4b execution readiness.

`epi_kind` stores structural EPI identity. `source_glyph` stores the most recent
successful operator as provenance (with ordered glyph history as the fuller
record). A successful ZHIR preserves `epi_kind` and records `ZHIR` in
`source_glyph`; the two attributes are not aliases.

**API:**
- `tnfr.structural.<OperatorName>()` - Individual operators
- `run_sequence(G, node, ops)` - Execute operator sequences
- `validate_sequence(ops)` - Check grammar compliance

**Grammar:** See [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) for complete rules
**Detailed Specs:** See [AGENTS.md § The 13 Canonical Operators](../AGENTS.md#5-the-13-canonical-operators)
**Math:** [Mathematical Foundations](MATHEMATICAL_DYNAMICS_BASIS.md)

---

## Canonical Invariants (Optimized Set)

From [AGENTS.md](../AGENTS.md) - Optimized from 10 to 6 invariants based on mathematical derivation:

1. **Nodal Equation Integrity**: EPI evolution only via ∂EPI/∂t = νf · ΔNFR(t)
2. **Phase-Coherent Coupling**: |wrap(φᵢ - φⱼ)| ≤ Δφ_max required for resonant operations
3. **Multi-Scale Fractality**: Operational fractality and nested EPIs maintained
4. **Grammar Compliance**: All operator sequences must satisfy U1-U6 validation
5. **Structural Metrology**: Units consistency (νf in Hz_str) and telemetry exposure
6. **Reproducible Dynamics**: Deterministic evolution with seed-based control

---

## Quick Reference Tables

### Variable Summary

| Symbol | Mathematical | Code Attribute | Units | Range | Type |
|--------|--------------|----------------|-------|-------|------|
| \(\text{EPI}\) | Primary Information Structure | `'EPI'` | dimensionless | \(B_{\text{EPI}}\) | Coherent form |
| \(\nu_f\) | Structural frequency | `'vf'` | Hz_str | \(\mathbb{R}^+\) | Reorganization rate |
| \(\Delta\text{NFR}\) | Reorganization operator | `'dnfr'` | dimensionless | \(\mathbb{R}\) | Evolution gradient |
| \(\theta\), \(\phi\) | Phase angle | `'theta'` | radians | \([0, 2\pi)\) | Network synchrony |
| \(C(t)\) | Total coherence | `compute_coherence()` | dimensionless | \([0, 1]\) | Global stability |
| \(\text{Si}\) | Sense Index | `'Si'` | dimensionless | \([0, 1^+]\) | Reorganization stability |

### Common API Patterns

```python
# Access node attributes
epi = G.nodes[node_id]['EPI']
vf = G.nodes[node_id]['vf']
theta = G.nodes[node_id]['theta']

# Compute metrics
C_t = compute_coherence(G)
nodes, W = coherence_matrix(G)
Si = compute_Si_node(G, node_id)

# Apply operators
from tnfr.structural import Emission, Coherence, Resonance
run_sequence(G, node_id, [Emission(), Coherence(), Resonance()])

# Evolution step
from tnfr.dynamics import step
step(G, use_Si=True, apply_glyphs=True)

# Canonical fixed point (per-node kernel + equilibrium predicate)
from tnfr.metrics.common import structural_coherence, is_structural_equilibrium
C_node = structural_coherence(G.nodes[node_id]['dnfr'])
at_equilibrium = is_structural_equilibrium(G.nodes[node_id]['dnfr'])

# Whole-NFR read-out (region) + nodal topology (radial/annular/multinodal)
from tnfr.sdk import TNFR
net = TNFR.create(20).ring().evolve(5)
nfr = net.nfr()  # geometry, telemetry availability, equilibrium fraction, xi_C
```

---

## Telemetry & Traces

Expose in telemetry:
- `C(t)` - Total coherence
- `νf` per node - Structural frequency
- `phase` per node - Synchrony state
- `Si` per node/network - Sense index
- `ΔNFR` per node - Reorganization gradient
- Operator history - Applied transformations
- Events - Birth, bifurcation, collapse

**API:** `tnfr.utils.callback_manager`, history tracking in `G.graph['_hist']`

---

## Domain Neutrality & the Two-Layer Ontology

TNFR is **domain-neutral**: the structural operators apply to graph-coupled networks
without a built-in application domain. The nodal equation gives the fixed-point
condition `ΔNFR = 0` when capacity is positive. Domain models can share the
numeric kernel `structural_coherence` and predicate `is_structural_equilibrium`
while using different state spaces, pressures and dynamics. Around that condition the read-outs span a
**spectrum of emergence**, contrasted as two layers:

- **Closed-loop phase read-out:** the winding `W ∈ ℤ` is a topological invariant of a
  declared single-valued phase loop. The implementation reports neutral, unit-winding
  and higher-winding sectors; these are structural classes, not particle species
  (`tnfr.physics.emergent_particles`). A prepared loop supplies the winding, so this
  read-out does not show that nodal dynamics creates it.
- **Symbolic / informational domains — numbers, chemistry:** a structural prime is
  `ΔNFR_arith = 0` and a noble gas model has `ΔNFR_chem = 0`. These share the abstract
  zero-pressure predicate while using different state spaces and evolution laws. The
  per-node arithmetic/chemical ΔNFR **consumes** its domain data (divisibility τ/σ/ω; the
  aufbau order) — the informational shadow of the structural grammar, not a direct
  *topological* emergence.

**Refinement — emergence is a spectrum, not a clean binary.** The symbolic layer is not
uniformly "consuming". Per [TNFR_NUMBER_THEORY.md §9.5](TNFR_NUMBER_THEORY.md), primality
has three sectors: **A** (arithmetic `ΔNFR = 0`, what the SDK `primes()`/`primality()`
expose) is an exact but *circular* re-expression that consumes Ω/τ/σ; **B** (spectral — the
Paley/residue Fiedler gap, input only `x² mod n`) is a **non-circular spectral diagnostic
under its stated graph family** (primes-OUT); **C** (representation-theoretic
irreducibility) is *refuted*. Sector B is partial (the real spectrum reaches only
`n ≡ 1 (mod 4)`) and never lives in the per-node substrate. Its remaining
`Fix(G)^⊥` projection obstruction is analogous to, but not identified with, the paused
TNFR-Riemann obstruction. The two-layer split is the SDK-level contrast; the trichotomy
is the precise account.

The graph and arithmetic paths reuse `structural_coherence`; the shell model reuses
only `is_structural_equilibrium` on its independently defined distance. Each domain
realizes its own ΔNFR and state space.

**Guideline:** avoid domain-specific hard-coding in the core engine; be honest about the
evidence type (declared-cycle topological / non-circular spectral / divisibility-consuming) per
read-out.

---

## Reproducibility

All simulations must be:
1. **Seeded:** Explicit RNG seeds
2. **Traceable:** Log operators, parameters, states
3. **Deterministic:** Same seed → same trajectory

**Tools:** RNG scaffolding, structural history, telemetry caches

---

## Unified Grammar Terms

### Unified Grammar

The consolidated TNFR grammar system (**U1-U6**) that replaces the old C1-C3 and RC1-RC4 systems.

**Source of Truth:** [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)
**Quick Reference:** [AGENTS.md § Unified Grammar (U1-U6)](../AGENTS.md#6-unified-grammar-u1u6)
**Implementation:** `src/tnfr/operators/grammar.py`

The implemented canonical grammar consists of six rules, U1-U6. This is the
current engine contract; it is not a theorem that no future specification could
ever require another rule.

**Six Canonical Constraints:**

| Rule | Requirement | Scope |
|------|-------------|-------|
| **U1** | Start with {AL, NAV, REMESH}; end with {SHA, NAV, REMESH, OZ} | Initialization and finite-word boundary contract |
| **U2** | Destabilizers {OZ, ZHIR, VAL} require stabilizers {IL, THOL} within the debt policy | Calibrated finite-word coverage policy |
| **U3** | {UM, RA} require wrapped phase compatibility | Runtime state precondition |
| **U4** | Triggers require handlers; transformers require recent destabilizer context | Bifurcation composition contract |
| **U5** | Nested EPI depth requires scale-local stabilization | Multi-scale identity contract |
| **U6** | Monitor ΔΦ_s against π/2 | Read-only selected safety policy |

**Not Part of Grammar** (telemetry/dynamics, NOT rules):
- **Structural Field Hexad**: Tetrad (Φ_s, |∇φ|, K_φ, ξ_C) + Flux Pair (J_φ, ∇·J_ΔNFR)
- **"Proposed U7"**: Historical research direction (Temporal Ordering) - NOT canonical, NOT implemented

**See Also:**
- [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) - Current grammar specification and stated physical motivations
- [AGENTS.md § Unified Grammar](../AGENTS.md#6-unified-grammar-u1u6) - Quick reference
- [PHYSICS_VERIFICATION.md](../docs/grammar/PHYSICS_VERIFICATION.md) - Grammar verification scope
- [STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md) - Structural-field interface, limits, and U6 validation details
- [src/tnfr/physics/fields.py](../src/tnfr/physics/fields.py) - Φ_s implementation

---

### Generator Operator

Operator that can create EPI from null/dormant states.

**Set:** GENERATORS = {emission, transition, recursivity}

**Physics:** Only these operators can initialize when EPI=0

**Grammar Rule:** U1a (STRUCTURAL INITIATION)

**See:** UNIFIED_GRAMMAR_RULES.md § U1a

---

### Closure Operator

Operator allowed as the terminal token of a U1b-valid word.

**Set:** CLOSURES = {silence, transition, recursivity, dissonance}

**Physics:** U1b syntactic closure of the declared sequence. The set includes
Dissonance, so membership does not by itself prove that the terminal state is a
coherent dynamical attractor.

**Grammar Rule:** U1b (STRUCTURAL CLOSURE)

**See:** UNIFIED_GRAMMAR_RULES.md § U1b

---

### Stabilizer Operator

Operator assigned the U2 role that compensates destabilizer debt.

**Set:** STABILIZERS = {coherence, self_organization}

**Physics:** Coherence and Self-organization have local stabilizing contracts,
and U2 requires their presence in the relevant operator context. The role does
not by itself prove convergence of `∫nu_f*DeltaNFR dt` or monotonic decrease of
a common Lyapunov functional.

**Grammar Rule:** U2 (CONVERGENCE & BOUNDEDNESS)

**See:** UNIFIED_GRAMMAR_RULES.md § U2

---

### Destabilizer Operator

Operator that incurs U2 destabilizer debt through its canonical role.

**Set:** DESTABILIZERS = {dissonance, mutation, expansion}

**Physics:** An uncompensated occurrence violates the U2 sequence policy.
Divergence is a dynamical claim that additionally requires an implemented map,
state functional and elapsed-time model.

**Grammar Rule:** U2 (CONVERGENCE & BOUNDEDNESS)

**See:** UNIFIED_GRAMMAR_RULES.md § U2

---

### Coupling/Resonance Operator

Operators that require phase verification for valid coupling.

**Set:** COUPLING_RESONANCE = {coupling, resonance}

**Physics:** Resonance requires |wrap(φᵢ - φⱼ)| ≤ Δφ_max

**Grammar Rule:** U3 (RESONANT COUPLING)

**See:** UNIFIED_GRAMMAR_RULES.md § U3

---

## The Structural-Field Tetrad

**Theory:** The four structural fields form the canonical diagnostic read-out.
Only π supplies an exact phase-wrap scale. Minimal complete state reconstruction
from the tetrad remains open.

### The one structural scale

Only **π** is a genuine structural scale: it bounds the wrapped phase sector.
The coherence-length estimate is spectral. Φ_s safety values and the 0.9π
curvature margin are selected policies; other parameters must be labelled as
derived under stated hypotheses or operational.

### Structural Fields and their bounds

1. **Φ_s** (0th order): selected warnings ΔΦ_s < π/2 and |Φ_s| < π/4
2. **|∇φ|** (1st order): bound |∇φ| ≤ π (phase wrap); π/16 ≈ 0.196 is the selected early-warning policy
3. **K_φ** (2nd order): exact wrapped bound π; 0.9π is a warning margin; agreement with L_rw·φ requires small phase spread and matching conventions
4. **ξ_C** (correlation): state-dependent fit; `1/√λ₂` is its connected-graph spectral comparison/fallback (not base e)

**Documentation:** [Structural-field tetrad](FUNDAMENTAL_THEORY.md)

---

### Flux Fields & the Emergent Symplectic Substrate

The tetrad has two **conjugate flux fields** that embed selected static fields in the
auxiliary symplectic substrate. They are the *currents* paired with those fields;
adding them does not establish complete state observability.

- **Phase current J_φ** — geometric, phase-driven transport; conjugate to curvature K_φ.
  Compute: `compute_phase_current(G)`.
- **ΔNFR flux J_ΔNFR** — potential-driven reorganization transport; conjugate to the
  potential Φ_s. Compute: `compute_dnfr_flux(G)`.
- **API:** `tnfr.physics.extended` (`compute_phase_current`, `compute_dnfr_flux`).

**Auxiliary symplectic substrate.** A graph-field snapshot initializes an ambient
symplectic phase space \\(\\mathbb{R}^{4N}\\) with two assigned coordinate pairs per node — **geometric**
\\((K_\\phi, J_\\phi)\\) and **potential** \\((\\Phi_s, J_{\\Delta\\text{NFR}})\\) — with brackets
\\(\\{K_\\phi, J_\\phi\\} = \\{\\Phi_s, J_{\\Delta\\text{NFR}}\\} = 1\\) and Hamiltonian
\\(H_\\text{sub} = \\tfrac{1}{2}\\sum(K_\\phi^2 + J_\\phi^2 + \\Phi_s^2 + J_{\\Delta\\text{NFR}}^2)\\).
The model's exact harmonic flow is a **symplectomorphism** (Liouville: phase volume
preserved). This does not certify any engine operator as symplectic. The complex coordinate
\\(\\Psi = K_\\phi + i\\,J_\\phi\\) carries a **U(1)** gauge symmetry; the substrate further
carries a **U(2)** polarization symmetry (per-node Poincaré sphere, classical Stokes
texture — **not** a quantum state). A separate damped graph wave has a restricted
pure-EPI diffusion limit; the full nodal equation is not derived from this isotropic flow.

**API:** `tnfr.physics.symplectic_substrate`, `Network.symplectic_substrate()`.
**Documentation:** [AGENTS.md §4 (Emergent geometry)](../AGENTS.md), [src/tnfr/physics/symplectic_substrate.py](../src/tnfr/physics/symplectic_substrate.py)

---

### Bifurcation Trigger

Operators that may trigger phase transitions.

**Set:** BIFURCATION_TRIGGERS = {dissonance, mutation}

**Physics:** ZHIR (Mutation) is the canonical bifurcation operator. Its
non-disableable gate requires a valid observed signed secant with the strict
crossing `observed dEPI/dt > ξ`: either fresh timestamped physical evidence or
legacy unit-operator-step evidence. This observed gate is distinct from the
instantaneous prediction `nu_f * DeltaNFR`, the three-sample structural
acceleration, and the `0.5` capacity threshold used only to propose a selector
branch. The default `ξ = 0.1` is an operational calibration. U4b context remains
a separate grammar requirement.

**Grammar Rule:** U4a (requires handlers)

**See:** UNIFIED_GRAMMAR_RULES.md § U4a

---

### Bifurcation Handler

Operators that manage structural reorganization during bifurcations.

**Set:** BIFURCATION_HANDLERS = {self_organization, coherence}

**Physics:** Provide stability during phase transitions

**Grammar Rule:** U4a (BIFURCATION DYNAMICS)

**See:** UNIFIED_GRAMMAR_RULES.md § U4a

---

### Transformer Operator

Operators that perform threshold-crossing structural phase transitions.

**Set:** TRANSFORMERS = {mutation, self_organization}

**Physics:** Require recent destabilizer context; ZHIR separately requires a
prior IL and valid observed temporal evidence

**Grammar Rule:** U4b (requires context + prior IL for ZHIR)

**See:** UNIFIED_GRAMMAR_RULES.md § U4b

---

## Related Documentation

### Core References (Essential)
- **[AGENTS.md](../AGENTS.md)** ⭐ - Single source of truth for TNFR agent guidance, invariants, and philosophy
- **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** ⭐ - Current U1-U6 grammar specification and scope
- **[Mathematical Foundations](MATHEMATICAL_DYNAMICS_BASIS.md)** ⭐ - Mathematical models, restricted proofs and spectral context

### Theory & Physics
- [TNFR.pdf](TNFR.pdf) - Original theoretical companion (paradigm, nodal equation, foundational physics)
- [PHYSICS_VERIFICATION.md](../docs/grammar/PHYSICS_VERIFICATION.md) - Grammar physics verification and scope
- [STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md) - Structural fields validation (Φ_s, phase gradients)


### Implementation & API
- [ARCHITECTURE.md](../ARCHITECTURE.md) - System design and architecture patterns
- [Foundations](../AGENTS.md) - Runtime/API guide
- [API Overview](../AGENTS.md) - Package architecture
- [Structural Operators](STRUCTURAL_OPERATORS.md) - Operator implementation details
- [Examples](../examples/README.md) - Runnable scenarios across domains

### Grammar & Migration
- [docs/grammar/](../docs/grammar/) - Grammar documentation directory (U6, fundamental concepts, etc.)

### Testing & Development
- [TESTING.md](../TESTING.md) - Test conventions and invariant verification
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Detailed contribution guidelines
- [REPRODUCIBILITY.md](../AGENTS.md) - Determinism requirements

### Cross-References and Documentation Hub

**Primary Sources:**
- **[AGENTS.md](../AGENTS.md)** - Single source of truth for TNFR theory
- **[UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md)** - Current U1-U6 grammar specification
- **[Structural Fields and the Tetrad](FUNDAMENTAL_THEORY.md)** - Mathematical foundations

**Implementation References:**
- **[src/tnfr/physics/fields.py](../src/tnfr/physics/fields.py)** - Unified Structural Field Tetrad (Canonical)
- **[src/tnfr/dynamics/self_optimizing_engine.py](../src/tnfr/dynamics/self_optimizing_engine.py)** - Self-optimization & auto-optimization
- **[docs/STRUCTURAL_FIELDS_TETRAD.md](../docs/STRUCTURAL_FIELDS_TETRAD.md)** - Technical field specifications
- **[docs/grammar/PHYSICS_VERIFICATION.md](../docs/grammar/PHYSICS_VERIFICATION.md)** - Grammar physics verification

**Development Resources:**
- **[src/tnfr/sdk/](../src/tnfr/sdk/)** - Simplified & Fluent API
- **[examples/](../examples/)** - Complete tutorial suite
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - System design patterns

---

## Assumption-explicit shell-model correspondence

**Technical approach:** a constructed Fibonacci-sphere graph supplies a reproducible
numerical comparison with low-lying angular multiplicities `2l+1`. Occupation capacities
`2(2l+1)`, the Madelung `(n+l, n)` filling order and the selected duet/octet closures are
declared shell-model inputs; they are not derived from the graph spectrum or the nodal
equation. The domain field `ΔNFR_chem(Z)` is defined as the count distance to that one
centralized closure set, so `ΔNFR_chem(Z) = 0` exactly when `Z` belongs to the declared
set. Reuse of `is_structural_equilibrium` is a scalar-predicate analogy and does not imply
a chemical relaxation law, force or common state space with arithmetic. See
[src/tnfr/physics/emergent_chemistry.py](../src/tnfr/physics/emergent_chemistry.py)
and the SDK `TNFR.element(Z)` / `TNFR.magic_numbers()`.

### Graph-field probe signatures

**Code:** `tnfr.physics.signatures`
**What:** A diagnostic tuple computed from the supplied graph and, by default,
its response to the declared detached probe word
`[Emission, Coherence, Silence]`.
**Metrics:** fitted ξ_C; mean |∇φ|; mean and maximum |K_φ|; and the maximum
and mean absolute nodewise Φ_s drift caused by that probe.
**API:** `compute_element_signature(G)`, `compute_au_like_signature(G)`
(the element-oriented names are compatibility APIs).
**Scope:** This is neither a spectral-coherence metric nor a chemical-element
classifier. It does not imply autonomous restoring dynamics. With
`apply_synthetic_step=False`, the reported zero drift means that no probe was
run; it is not stability evidence. In that mode, `signature_class="stable"`
reflects only the phase gates because the absent drift check is treated as
satisfied.

### Au-like compatibility label

**Symbol:** Au (from Latin *aurum*)
**What:** Historical boolean label over the same graph-field probe response.
**Criteria:** medium/extended fitted ξ_C or the legacy `n > 50` size branch;
mean |∇φ| < π/2; maximum |K_φ| < 0.95π; and Φ_s probe drift below the U6
policy threshold.
**Detection:** `compute_au_like_signature()["is_au_like"]`
**Scope:** No atomic identity, metallic property, chemical stability, or
optimality follows from this label.

### Coupling analogy

**TNFR read-out:** Phase-compatible graph coupling with U3 verification: |wrap(φᵢ - φⱼ)| ≤ Δφ_max
**API:** Coupling operators with phase compatibility check
**Scope:** The implementation supplies no chemical bond-energy or bond-strength model

### Operator-sequence analogy

**TNFR:** Validated operator sequences can model abstract structural reorganization.
**Grammar:** Any example must satisfy the applicable U1–U6 rules; no fixed word is assigned to a chemical reaction.
**API:** Sequence validation via `grammar.py`
**Scope:** The shell-model module implements no reaction kinetics or transition-state model.

### Geometry boundary

TNFR can measure graph topology, phase gradients and structural pressure on a supplied
network. The shell-model correspondence does not infer molecular geometry, VSEPR,
hybridization or stable molecular configurations from those quantities.

**Implementation:** the shell-model correspondence is implemented in
[src/tnfr/physics/emergent_chemistry.py](../src/tnfr/physics/emergent_chemistry.py).
The independent graph-field probe API is implemented in
[src/tnfr/physics/signatures.py](../src/tnfr/physics/signatures.py).

## Self-Optimizing Engine

**Self-Optimization:** The TNFR engine includes self-optimization capabilities using unified field telemetry.

### Core Components

**TNFRSelfOptimizingEngine:** `src/tnfr/dynamics/self_optimizing_engine.py`
**Purpose:** Closes feedback loop via unified field monitoring
**Monitors:** Complex Geometric Field (Ψ), Chirality (χ), Symmetry Breaking (𝒮), Coherence Coupling (𝒞)
**Detects:** configured diagnostic conditions using energy density ℰ and the historical `topological_charge` read-out 𝒬
**Usage:** `engine = TNFRSelfOptimizingEngine(G); success, metrics = engine.step(node_id)`

### Auto-Optimization API

**Fluent Integration:** `TNFRNetwork(G).focus(node).auto_optimize().execute()`
**Field Analysis:** `analyze_optimization_potential(G)` - Mathematical structure analysis
**Strategy Recommendations:** `recommend_field_optimization_strategy(G)` - Optimization strategies
**Automatic Execution:** `auto_optimize_field_computation(G)` - Self-optimizing computation

## Unified Field Framework (Nov 2025)

**Mathematical framework:** Complex field relationships and finite balance diagnostics.

### Complex Geometric Field (Ψ)

**Definition:** Ψ = K_φ + i·J_φ (unifies geometry + transport)
**Finite evidence:** r(K_φ, J_φ) = -0.854 to -0.997 in the cited historical protocol; not a universal correlation law
**API:** `compute_complex_geometric_field(G)`
**Usage:** Unified geometry-transport analysis

### Emergent Fields

**Chirality (χ):** `χ = |∇φ|·K_φ - J_φ·J_ΔNFR` - signed handedness diagnostic
**Symmetry Breaking (𝒮):** candidate transition indicator requiring a declared finite-size protocol
**Coherence Coupling (𝒞):** Multi-scale connector field
**API:** `compute_emergent_fields(G)`

### Tensor diagnostics

**Energy Density (ℰ):** `ℰ = Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²`
**Historical `topological_charge` name (𝒬):** `𝒬 = |∇φ|·J_φ - K_φ·J_ΔNFR`; no quantization or grammar-wide conservation theorem
**Balance diagnostic:** ∂ρ/∂t + ∇·𝐉 = S for the supplied trajectory and discretization
**API:** `compute_tensor_invariants(G)`

**Unified Telemetry:** `compute_unified_telemetry(G)` - combined diagnostic
suite: tetrad/coherence read-outs plus an auxiliary spectral-pulse read-out.
It is not a complete observer of arbitrary TNFR state.

---

## Operator-Tetrad Synergies (Experimental, March 2026)

Finite diagnostics connecting canonical operators to the structural-field tetrad.
Reference: [STRUCTURAL_OPERATORS.md §17](STRUCTURAL_OPERATORS.md), examples 37-39.

### Dual-Lever Structure (channel partition)

**What:** Each operator's primary effect lands on exactly **one nodal channel** — the
partition is simultaneously the dual-lever (capacity νf vs pressure ΔNFR), the tetrad
driver and the number-theory grading (AGENTS.md §5):
- **νf (capacity):** Silence (SHA), Expansion (VAL), Contraction (NUL).
- **ΔNFR (pressure):** Coherence (IL), Dissonance (OZ), Self-organization (THOL), Transition (NAV).
- **θ (phase):** Coupling (UM), Mutation (ZHIR).
- **EPI (written directly):** Emission (AL), Reception (EN), Resonance (RA), Recursivity (REMESH).

**Source of truth:** [src/tnfr/operators/operator_contracts.py](../src/tnfr/operators/operator_contracts.py).
**Evidence:** `examples/02_physics_regimes/39_nodal_equation_decomposition.py`.

### Sampled operator-tetrad response matrix

**What:** The example tabulates relative tetrad changes for one graph, target,
seed and runtime history. It records requested and actual glyphs because grammar
fallbacks can make different requests execute the same operator. It does not
establish universal uniqueness or tetrad completeness.
**Evidence:** `examples/02_physics_regimes/37_operator_tetrad_synergy.py`.

### IL/OZ finite comparison

**What:** A historical single-node protocol reported equal magnitudes after
runtime substitutions. The current example exposes actual glyph history; this
sample cannot establish an IL/OZ symmetry or explain U2.

### Linear Response of Phi_s

**What:** For the fixed graph and perturbation protocol in example 39, Phi_s is
linear in DNFR and the sampled Pearson correlation has |r| = 1.000.
**Contrast:** xi_C is a nonlinear, state- and fit-dependent statistic; the
reported crossover near DNFR 0.3 belongs only to that protocol.
**Evidence:** `examples/02_physics_regimes/39_nodal_equation_decomposition.py`.

### Diagnostic dependency chain

**What:** `Operator -> updated nodal/graph state -> tetrad -> derived diagnostics`.
The maps are deterministic for a fully specified state but are not generally
injective and need not determine future dynamics.

### Grammar-Energy Landscape

**What:** The cumulative policy multiplier `Pi < 1` is a nominal contractivity
score, not a sufficient condition for descent of the five-field energy.
Experimentally, one fixture has `Pi = 1.288` while its sampled net energy change
is `-9.59`.
**Interpretation:** The score does not determine the observed energy sign.
Grammar compliance supplies no Lyapunov guarantee without specified dynamics
and a proof.
**Evidence:** `examples/02_physics_regimes/38_grammar_energy_landscape.py`.

---

## Structural Conservation Theorem

**Main Result:** The module evaluates a Noether-like structural balance on supplied finite trajectories; grammar validity alone does not imply its residual vanishes.

**Charge Density:** ρ = Φ_s + K_φ (potential + geometric sectors)
**Current:** 𝐉 = (J_φ, J_ΔNFR) (transport channels)
**Balance:** ∂ρ/∂t + ∇·𝐉 = S for the chosen fields and discretization; the measured source/residual must be reported
**Two Sectors:** Potential (Φ_s ↔ J_ΔNFR) and Geometric (K_φ ↔ J_φ), coupled through Ψ = K_φ + i·J_φ
**Lyapunov:** E = ½Σ(Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²) ≥ 0 is a candidate; each trajectory must establish the sign of dE/dt, while a theorem exists only for the restricted diffusion model
**Validation:** finite tests report their seeds, discretization and measured balance residuals; no fixed count or universal drift bound is part of the theorem
**API:** `tnfr.physics.conservation` — Noether charge Q, energy functional E, Ward identities, spectral decomposition
**Documentation:** [theory/STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md)

---

## Integrity Monitor

**What:** Optional reactive audit of declared postconditions after operator execution.
**API:** `tnfr.physics.integrity`
**Coverage:** Distinguishes exact postconditions, tolerance-based checks,
balance alerts and metadata-only cases; it does not prove a grammar-wide
Lyapunov theorem.
**Usage:** Available to operator-application paths that explicitly enable or invoke the audit.
**Documentation:** [src/tnfr/physics/integrity.py](../src/tnfr/physics/integrity.py)

---

## Grammar-Aware Dynamics

**What:** Incremental checks for the sequence rules that can be decided from the available history and state.
**API:** `tnfr.operators.grammar_dynamics.GrammarAwareDynamics`
**Checks:** U1a initiation, U2 destabilizer/stabilizer debt tracking, U3 phase compatibility for UM/RA, U4a/U4b bifurcation context
**Boundary:** U5 needs nested-state information and U6 is a separate before/after telemetry observation.
**Documentation:** [src/tnfr/operators/grammar_dynamics.py](../src/tnfr/operators/grammar_dynamics.py)

---

## Grammar Application

**What:** Contextual operator application with the checks available at that call site.
**API:** `tnfr.operators.grammar_application.apply_glyph_with_grammar()`
**Pipeline:** Grammar check → operator application → postcondition verification (integrity monitor)
**Boundary:** It does not infer unavailable U3/U5 state or turn the read-only U6 observation into a sequence proof.
**Documentation:** [src/tnfr/operators/grammar_application.py](../src/tnfr/operators/grammar_application.py)

---

## Per-Node Φ_s Threshold

**Value:** PHI_S_VON_KOCH_THRESHOLD = π/4 ≈ 0.785
**What:** Per-node safety threshold for structural potential |Φ_s|.
**Status:** Selected warning policy. Writing the value as a fraction of π does not derive a graph-independent potential bound.
**Usage:** |Φ_s(i)| ≥ π/4 records a per-node potential warning.
**API:** `tnfr.constants.canonical.PHI_S_VON_KOCH_THRESHOLD`
**Relation to U6:** π/4 is a per-node magnitude warning; π/2 is the separate drift-monitor policy.

---

## Regime Correspondences

**Theory:** The single nodal dynamics produces two empirically-anchored regimes. The
external labels "classical"/"quantum-like" are comparisons only, not TNFR primitives.

### Smooth-Trajectory / Overdamped-Drift Regime (High Coherence)
**Condition:** C(t) → 1, |∇φ| → 0
**Correspondence:** first order in time, `q̇ = νf·F` — drift velocity ∝ force, so **νf is
mobility** (Stokes/Einstein), **not** inverse mass; `F = ΔNFR` (force ↔ structural
pressure). The inertial (second-order) regime lives in the conservative symplectic
substrate, not here.
**API:** `tnfr.physics.structural_diffusion`, `tnfr.physics.classical_mechanics`

### Discrete-Mode Regime (High Dissonance)
**Condition:** |∇φ| ~ π, near phase singularities
**Emergent:** on a bounded graph the diffusion operator has a discrete spectrum of
orthonormal standing-wave eigenmodes (vibrating-string / Chladni analogue), with
nodal-domain ordering (Courant); uncertainty (Fourier ΔEPI·Δνf ≥ K), superposition.
**API:** `tnfr.physics.structural_diffusion`, `tnfr.physics.quantum_mechanics`

---

## TNFR-Riemann Program

**What:** A structural attack surface relating discrete TNFR operators to the Riemann
zeta function. **Not a proof of RH** — the bridge is the open conjecture **T-HP** (gap G4).
**Core Operator:** H^(k)(σ) = L_k + V_σ where L_k = graph Laplacian, V_σ = diagonal potential
**Numerical result:** the critical parameter σ_c^(k) → 1/2 as k → ∞, verified across
topologies (`σ_c^(k) = 1/2 + O(log⁻¹ k)`).
**Honest scope:** TNFR-internal structural results and numerical evidence; no classical
open problem is closed.
**Implementation:** `src/tnfr/riemann/`
**Documentation:** [theory/TNFR_RIEMANN_RESEARCH_NOTES.md](TNFR_RIEMANN_RESEARCH_NOTES.md)

---

## Key Canonical Thresholds

Quick reference for canonical threshold values from `src/tnfr/constants/canonical.py`:

| Threshold | Value | Derivation | Usage |
|-----------|-------|------------|-------|
| PHI_S_VON_KOCH_THRESHOLD | π/4 ≈ 0.785 | Selected policy | Per-node Φ_s warning |
| PHASE_GRADIENT_THRESHOLD | π/16 ≈ 0.196 | Selected early-warning policy (not derived; bound is π) | \|∇φ\| stability |
| K_PHI_CANONICAL_THRESHOLD | 0.9×π ≈ 2.8274 | Selected 90% margin inside the exact wrap-angle bound π | K_φ fault zone detection |
| U6 drift monitor | π/2 ≈ 1.571 | Selected policy | ΔΦ_s drift warning |
| MIN_BUSINESS_COHERENCE | ≈ 0.75 | Operational (free parameter) | Business-health cut (the canonical strong-coherence gate is the emergent π/(π+1) ≈ 0.7585) |
| THOL_MIN_COLLECTIVE_COHERENCE | 1/(π+1) ≈ 0.2415 | Geometric series bound | Fragmentation risk threshold |

---

## Contributing Guidelines

When adding new functionality:

1. **Verify theoretical foundation**: Align with [AGENTS.md](../AGENTS.md) physics
2. **Preserve canonical invariants**: Follow optimized 6-invariant set
3. **Use established terminology**: Reference this glossary for consistency
4. **Map to canonical operators**: All functions must correspond to 13 canonical operators
5. **Validate grammar compliance**: Ensure U1-U6 satisfaction
6. **Maintain English-only policy**: All documentation in English for canonical terminology
7. **Write comprehensive tests**: Cover invariants and operator contracts

**Development Workflow:**
1. Read [AGENTS.md](../AGENTS.md) completely - **SINGLE SOURCE OF TRUTH**
2. Study [UNIFIED_GRAMMAR_RULES.md](UNIFIED_GRAMMAR_RULES.md) for physics foundations
3. Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines
4. Test with [TESTING.md](../TESTING.md) requirements

**Version**: 0.0.3.5 (June 2026)
**Status**: Canonical operational reference, aligned with the current engine, AGENTS.md and TNFR.pdf
**Language**: English only (canonical documentation policy)
