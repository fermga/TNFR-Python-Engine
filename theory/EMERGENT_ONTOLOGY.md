# Emergent Ontology from the Nodal Equation

**Status**: WORKING DRAFT — EXPLORATORY (not canonical)
**Date**: 2026-09-06 (audited against the current implementation and research certificates)
**Prerequisite**: [AGENTS.md](../AGENTS.md), [STRUCTURAL_CONSERVATION_THEOREM.md](STRUCTURAL_CONSERVATION_THEOREM.md), [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md)

---

## 0. Honest scope (read first)

This document catalogs the structures that **emerge** from the single nodal equation

$$\frac{\partial \mathrm{EPI}}{\partial t} = \nu_f \cdot \Delta\mathrm{NFR}(t)$$

and organizes them around the canonical dynamics and several explicitly
separate auxiliary comparison models (§2). Every entry carries exactly one
label:

| Label | Meaning |
|-------|---------|
| **POSITED** | An axiom/primitive of TNFR (assumed, not derived). |
| **DERIVED** | An exact structural identity, derivable from the nodal equation, with a proof or repository anchor. |
| **ANALOGY** | A structural resemblance used for intuition — **not** a derivation. |
| **OPEN CONJECTURE** | A research target that is **not** established; the document states what a derivation would require. |

**What this document claims.** The nodal equation in **Hz_str** organizes a
family of graph-state observables and several explicitly separate comparison
models. The EPI-only channel is exactly graph diffusion under its stated
hypotheses. The graph wave, isotropic harmonic substrate, stochastic models,
reaction normal forms and arithmetic encodings are auxiliary or domain-specific
models; they are not all trajectories of one proved dynamical law. The document
uses them to catalogue scoped mathematical correspondences and finite
measurements. Its broader "structural history" and physical-language readings
are **ANALOGY/POSIT**, not deductions of physical cosmology, relativity,
thermodynamics, electromagnetism or particle physics.

**What this document does NOT claim.** It does **not** derive the Standard Model (particle masses,
spins, the full quantum-number spectrum) or quantum mechanics (a complex Hilbert space, the Born
rule, genuine entanglement), and it is **not** "a theory of everything". The emergent substrate is
**classical** (a symplectic flow with classical wave polarization; §5.3). The reach to genuine
particles and quantum phenomena is collected in §9 as **OPEN CONJECTURE**. The emergent structural
history (§2.5) is TNFR's **own** — a structural "cosmology" by **analogy of scale**: the network's
emergent macro-history (a genesis §7.5, an emergent time with an arrow, structure formation,
composite assembly §7.4 / §9.1, a causal horizon), each stage carrying its own DERIVED/ANALOGY
label. It is a structural re-expression, not offered as a model of the physical universe's measured
cosmology (general relativity and observational cosmology are empirically ironclad and out of scope).

**Empirical caveat (from the validation record).** This unification is **structural/descriptive**,
not a source of novel empirical predictions. Pre-registered tests
(`benchmarks/u2_destabilization_irreversibility.py`, the 2026 grid/U2 studies) found **no** case
where a distinctive TNFR construct out-predicts standard methods on an established problem. What the
empirical **arm** *has* shown is that the canonical magnitudes carry real-data **structure** — the
local phase tetrad and the two-face diagnosis on real coupled-oscillator data (`ξ_C` competitive on
real EEG, [STRUCTURAL_INTERFACE_THEORY.md](../docs/STRUCTURAL_INTERFACE_THEORY.md); the
signal→canonical-magnitude confrontation pipeline,
[example 159](../examples/10_applications/159_empirical_confrontation_pipeline.py)) — a *falsifiable*
value, but still a **tie** with strong baselines, not an out-prediction. The value here is *one
vocabulary for many structures* (and a falsifiable instrument), not *better forecasts*.

**Scope axis (read second).** This document catalogs exact graph read-outs, results for
restricted channels of the nodal dynamics, and explicitly separate comparison models
(geometry, thermodynamics, relativistic structure, and gauge coordinates). That is **one
axis** of the emergent ontology. The
orthogonal **cross-domain** axis — how the *same* fixed point `ΔNFR = 0` is read out across
domains as a **spectrum of emergence** (declared-cycle winding as a direct graph
read-out; the number-theory
spectral sector *genuinely* but partially; the arithmetic `ΔNFR` a *circular* re-expression
that consumes divisibility; chemistry *mixed*) — is the three-sector trichotomy of
[TNFR_NUMBER_THEORY.md §9.5](TNFR_NUMBER_THEORY.md) and the two-layer ontology of
[GLOSSARY.md](GLOSSARY.md). Together: one fixed point, many read-outs.

For numbers, that cross-domain axis is now assembled into an explicit **ontological
position ladder** ([TNFR_NUMBER_THEORY.md §9.12](TNFR_NUMBER_THEORY.md), example
[155](../examples/08_emergent_geometry/155_ontological_position_of_numbers.py)): a number is a
**cardinal** (represented by selected multiplicity and simplex constructions,
not identified universally with spectral or spatial dimension; §3.2), carries
graph-product encodings of **+, ×**,
has its **primality** and **factorization type** (`Ω, τ` → the `ΔNFR` triad) read off the residue
spectrum (Sector B), and only the prime **identities** and the continuous `arg ζ` phase remain at
the wall. The arithmetic `ΔNFR` coefficients are themselves canonically **unity** — only `π` is a
genuine structural scale, and the [§4.2](TNFR_NUMBER_THEORY.md) coefficient-independence theorem
shows that every positive weighting has the same prime zero set. Unit weights
are the canonical parameter-free convention (not forced uniquely by that theorem), so Sector A's "circular re-expression" is the
*consumed* read-out of a fixed point whose *emergent* read-out (Sector B) genuinely derives the
arithmetic **up to that wall**. The wall is located on the **non-self-adjoint directed residue operator**
([§10.5](TNFR_NUMBER_THEORY.md), `benchmarks/residue_phase_vs_riemann.py`) — a non-symmetric circulant
(hence *normal*), diagonalized by the `Z/n` characters: it carries arithmetic in the **phase**
(`√p` Gauss sums = Fourier coefficients of the residue set), structurally distinct from the `ζ` zeros — the obstruction is
**sharpened and relocated, not dissolved**. This cross-domain refinement is a structural read-out
catalog (one fixed point, many emergence sectors); it closes no open problem.

**The organizing synthesis** compares several uses of a declared graph operator
at three levels (*form → scaling readout → dynamics*) and tracks their different
observability limits. It is a taxonomy, not proof that every domain shares one
operator or one obstruction; see **§2.4**.

---

## 1. First principles

**The nodal law and its primitives.** The canonical engine starts from the nodal equation and its
multichannel gradient `ΔNFR = w_phase·∂φ + w_epi·∂EPI + w_vf·∂νf + w_topo·∂topo`
([dnfr.py](../src/tnfr/dynamics/dnfr.py)). Later sections also introduce
explicit auxiliary models and analogies that do not derive from this equation
alone. The primitives are **POSITED** — the bedrock, in
Hz_str, *prior to* any physical magnitude (a temperature, a frequency in Hz, an energy in joules
are their manifestations at scale, not the reverse):

| Primitive | Symbol | Role |
|-----------|--------|------|
| Primary information structure | EPI | coherent form on a node |
| Structural frequency | νf (Hz_str) | reorganization rate — *prior to physical time* |
| Nodal gradient | ΔNFR | reorganization pressure |
| Phase | φ (θ) | synchronization coordinate |
| Coupling network | G (graph) | the relational substrate (connectivity only; its *geometry* is derived, §3) |

**The emergent-first rule.** *Never import an external physical magnitude before it appears as an
emergent of the nodal dynamics.* Derive the magnitude from `∂EPI/∂t = νf·ΔNFR` first; only then
recognize which empirical phenomenon instantiates it. Two consequences the rest respects:
**coherence `C` is a TNFR primitive** (`C = 1/(1 + mean|ΔNFR| + mean|dEPI|)`, proximity to
`ΔNFR→0`), not a relabeled order parameter or variance; and **νf is a structural rate** (Hz_str),
with physical time and frequency themselves *emergents* (§4.2).

**Structural level vs manifestation (the directionality POSIT).** A shared mathematical form
(e.g. both being a diffusion equation) is direction-neutral. TNFR adds an interpretive **posit**:
the nodal dynamics is the *structural substrate* (Hz_str), and the physical law is its
*manifestation at a scale* — "the same structural law manifests, at the thermal scale, as heat".
This is **not** provable from the form-sharing alone. Its **testable hook** is
fractal recurrence (operational fractality, grammar U5 / REMESH): an explicit
coarse map must first define which observables, relaxation rates and grammar
constraints are compared. Self-similar recurrence is then a falsifiable research
question, not an assumed identity. Throughout:
the shared **form** is **DERIVED**; the **structural-priority** reading is **POSITED**; the
document never asserts that TNFR *is* thermodynamics or relativity — only that the one law
**manifests as** them at scale.

**The grammar is the engine's generative syntax.** Operators are the exclusive
mechanism that changes EPI, and U1–U6 constrain accepted operator sequences and
telemetry. This makes grammar upstream of operator-driven engine trajectories;
it does not make every auxiliary model below a theorem of grammar. U3 is an
operator admissibility contract, U2 and U4 are calibrated sequence policies,
and U6 is a read-only warning policy. Their mathematical promotions remain scoped:

| Grammar rule | Coherence / existence condition | Emergent it enables |
|---|---|---|
| **U1** initiation & closure | start from the vacuum, end in a declared closure mode | the vacuum→structure boundary (§7.1a) |
| **U2** stabilization & debt | destabilizers require stabilizer coverage within the configured debt policy | admissible operator histories; no general convergence theorem |
| **U3** resonant coupling | phase compatibility `\|φᵢ−φⱼ\|≤Δφ_max` | synchronization (§6.1); coupling & EM (§7.2) |
| **U4** bifurcation | triggers need handlers | transitions / criticality (§6.2) |
| **U5** multi-scale coherence | nested EPIs keep identity | composites (§7.4); fractal recurrence (§1) |
| **U6** potential confinement | monitor `ΔΦ_s < π/2` from a declared reference | finite-trajectory potential alert (§3.3, §6.2) |

Read this way, the information capacity of §8 is a formal-language diagnostic of the grammar:
the bits-per-operator of the syntax that makes coherent emergence possible at all.

---

## 2. Canonical dynamics and auxiliary comparison models

The nodal equation is first order. The repository also defines conservative
comparison models with related graph-field inputs. Their useful correspondences
must not be mistaken for two limits of one generally derived flow.

| | **Diffusive face** (overdamped) | **Conservative face** (inertial / wave) |
|---|---|---|
| Order in time | 1st: `∂EPI/∂t = νf·ΔNFR` | 2nd: auxiliary graph-wave or harmonic-substrate flow |
| Character | dissipative in the fixed EPI-channel model | reversible, oscillatory auxiliary models |
| Comparison | graph heat flow and restricted Dirichlet balance (§4) | graph-wave propagation and harmonic-substrate geometry (§5) |
| Causal cone | **none** (infinite signal speed) | **a finite-speed light cone** |
| Charges / defects | annihilate (dissipative) | orbit (Hamiltonian, integrable) |

An exact overdamped bridge exists for a separately defined damped graph wave
with stiffness `L_rw`. The isotropic harmonic substrate has identity stiffness
and is not thereby a lift of the nodal engine. The phase,
reaction and driven examples below therefore remain explicitly scoped models.

### 2.1 What threads the two towers — recurring structural pivots

Beyond the two-face split, a few quantities **recur** across the otherwise-separate emergents,
tying them into one structure (the synergies a first pass can miss):

- **Related operators, scoped readouts.** `L_rw`/`L_sym` supports exact
  diffusion identities and supplies spectra used by metric, heat-kernel and
  graph-wave diagnostics. Similarity, normalization and symmetry hypotheses
  must be stated for each use.
- **Form, scaling and dynamics are distinct objects.** A coupling operator, a
  fitted spectral dimension, a simplex grade and the values `sqrt(lambda_k)`
  can be compared, but none is generally identical to the others. Finite
  heat-trace diagnostics retain separate Hausdorff and spectral dimensions.
- **A recurring spectral scale, `λ₂`.** Under fixed homogeneous diffusion it
  sets the slowest nonuniform decay rate; separate graph-wave and reaction
  models reuse it under their own assumptions. This recurrence is not one
  universal engine clock.
- **One diagnostic interface.** The four fields (§3.3) organize readouts
  of the higher emergents — `|∇φ|` measures local synchronization stress
  (§6.1; the selected `π/16` warning is distinct from the measured,
  σ-dependent onset near `0.29`), `Φ_s` is the source aggregation monitored by U6,
  `K_φ` enters the historical bilinear Q snapshot (§7.1), while integer phase
  winding requires an explicitly declared cycle; `ξ_C` records a finite-network
  correlation scale. A divergence claim requires a declared control parameter and a
  reproducible finite-size limit (§6.2).
- **One partition, the four channels.** The dual-lever channels sort the emergents: `νf` (capacity)
  → time, transport, fluctuations, wave speed; `ΔNFR` (pressure) → criticality, potential; `phase`
  → synchronization, EM, optics; `EPI` (form) → diffusion, modes, composites.

### 2.2 Higher-order capabilities — what the layers produce together

Some emergents appear only at the **intersection** of others — capabilities no single layer shows:

- **Fault-tolerant memory** (winding model × thermodynamics × information). A
  declared cycle's integer winding `W` (§7.1b) is unchanged under continuous
  deformations that retain the cycle, nonzero phase support and a positive wrap-branch
  margin. In the selected noise protocol,
  a stored `W=2` is **retained with probability ≈1 below a noise threshold** (`σ ≲ 0.3`) and lost
  above it, even while the coupling continually restores the field: a noise-margined,
  **error-corrected memory** (the classical analog of topological storage). → robust information
  storage.
- **Candidate scale-free fluctuations** (thermodynamics × criticality). The auxiliary
  reaction-diffusion normal form in §6.2 has an analytic critical-slowing limit. The canonical
  finite-graph telemetry currently records only sampled susceptibility and `ξ_C`; it has not
  established divergence, `1/f` noise, avalanches or a graph-independent universality class.
  Those remain targets for the finite-size protocol.
- **Scale-reduction diagnostics** (geometry × multi-scale grammar). Static
  Kron reduction preserves selected effective resistances, while observer-aware
  reduction diagnostics show that arbitrary transient closure generally needs
  memory and nonlinear observers can be lost. A static reduction is not REMESH execution, U5
  certification or renormalization-group covariance.
- **A confinement mass gap** (conservative face × bounded structure). On a bounded structure the
  lowest wave mode has a *non-zero* frequency `ω_min = c√λ₂` — a **dispersion gap** that turns the
  massless low-`k` continuum (§5.2) into **gapped, massive-like** modes (the same gap that
  discretizes the matter stage, §7.1a, and binds composites, §7.4). → the structural form of an
  effective mass from confinement (not a derived particle mass).

### 2.3 A directness taxonomy — structural level and declared symmetry action

Sections 2.1–2.2 motivate a comparative taxonomy for the examples in this
repository: a declared-cycle winding is computed directly from a phase field, a
spectral statistic is computed after diagonalizing a selected graph operator, and
an arithmetic `ΔNFR` can be circular when its input already contains the
divisibility property under study. These are different observer pipelines. Their
ordering is a documentation convention, not a theorem fixed by representation
theory. An `Aut(G)` decomposition such as `Fix(G) ⊕ Fix(G)^⊥` applies only after
the acted-on vector space and representation have been specified
([ex 123](../examples/08_emergent_geometry/123_symmetry_sector_decomposition.py)).

| Level / read-out | `Aut(G)` sector | Directness |
|---|---|---|
| **occupant** — declared-cycle winding `W` | scalar invariant under the tested transported relabeling | **DIRECT GRAPH READ-OUT** |
| **stage** — spectral rank `ρ` | invariant of the selected matrix spectrum | **INDIRECT** — requires a spectral transform |
| **process** — `ΔNFR(Ω,τ,σ)` | — (consumes its input) | **CIRCULAR** (arithmetic) |

**Finite evidence**
([ex 156](../examples/08_emergent_geometry/156_emergence_directness_law.py)).
The recorded `C12` test preserves `|W|` when both the phase field and declared
cycle are relabeled. This scalar invariance does not place `W` itself in a vector
subspace called `Fix(G)`. An equivariant per-node observer is orbit-constant on
an invariant input to a vertex-transitive graph; arbitrary perturbed inputs need
not be. The arithmetic example is circular precisely when the discriminator
consumes the divisibility data it reports. These observations support the
taxonomy only for the declared finite experiments.

**Distinct algebraic comparisons.** Several benchmarks use matrices related to
the same selected adjacency `A`, but the actions and state spaces must remain
separate:

| Construction | Finite identity | What it establishes |
|---|---|---|
| graph automorphism `P` | `PA=AP` | equivariance under that permutation; orbit-constant output needs invariant input |
| bipartite sublattice involution `Γ` | `Γ A Γ=-A` | adjacency eigenvalues pair as `λ` and `-λ`; it does not construct integer additive inverses |
| pointwise phase negation | `φ → -φ`, hence `W → -W` | orientation reversal of a declared-loop phase map; a separate `Z₂` action, not physical charge conjugation |
| non-symmetric normal circulant | `A!=A^T`, `[A,A^T]=0` | Fourier diagonalization and, for the selected residue matrices, a measured Gauss-sum phase |
| Cartesian/tensor graph product | selected spectra add/multiply | spectral composition identities, without unique factorization ([composition_arithmetic.py](../benchmarks/composition_arithmetic.py)) |

The chiral benchmark makes the negative identification explicit
([chiral_involution.py](../benchmarks/chiral_involution.py)): sublattice
anticommutation acts on adjacency eigenvectors, while phase negation acts on a
loop field. Both square to the identity and reverse a signed read-out, but they
are not the same transformation. The scalar identities `λ+(-λ)=0`
and `W+(-W)=0` do not describe a shared additive operation or a simulated
annihilation process.

**Scope.** The table mixes standard finite identities, measurements on named
examples and an observer taxonomy. It is a comparison, not a derived
cross-domain law, and it closes no open mathematical problem.

### 2.4 Shared interfaces across domains — **the synthesis**

The pivots above assemble a comparative ontology: several domains can be
represented using graph operators, spectra, symmetry sectors and the shared
coherence kernel. The operator, state space, pressure realization and observer
must nevertheless be declared per domain. Similar obstruction patterns do not
make them one mathematical object, and scalar invariants are not state-vector
sectors.

| Domain | Form (the coupling) | Dimension (grade / `d_s`) | Dynamics (the spectrum) | Domain-specific limitation |
|--------|--------------------|---------------------------|-------------------------|------------------------------|
| **Physics** | the graph / field | spatial `d_s` (§3.2) | the tetrad, the pulse `ω_k=√λ_k`, thermodynamics, gauge (§3–§9) | the non-spectral residue — no genuine particle / quantum closure (§9, OPEN) |
| **Networks** | the network | `d_s`, the metric `R_eff` (§3.1) | transport, relaxation, synchronization (§4, §6) | **isospectral graphs** — Kac: the shape is not heard from the spectrum |
| **Number theory** | the residue Cayley net | selected cardinal/multiplicity and simplex-grade readouts ([NT §9.12](TNFR_NUMBER_THEORY.md)) | prime = `ΔNFR=0`; the cyclotomy rank `s_k(p)=gcd(k,p−1)+1` = the **arithmetic pulse** ([NT §9.13](TNFR_NUMBER_THEORY.md)) | prime identities and the `arg ζ` phase remain unresolved |
| **Music** | the resonator's shape | the dimension sets the regime (§5.5) | pitch `ω_k=√λ_k`, chord, timbre; consonance = phase; **1D harmonic, 2D+ inharmonic** | **Kac again** — you cannot hear the shape of the drum: the type, not the identity |

Thus the ontology offers **shared interfaces and comparisons across domains**.
Numbers, dimensions, graph geometries and musical descriptions remain distinct
objects even where one declared spectrum supplies related readouts. Likewise,
prime identity, inverse-spectral ambiguity, the zeta residue and the
particle/quantum frontier are separate open problems; `Fix(G)^⊥` is a useful
comparison only where the relevant representation action is explicitly
defined.

**Limits of the cross-domain comparison.** Earlier versions described the
Riemann and Navier–Stokes programs as having “the same wall.” That identification
is unsupported. Bounds for fluctuations of the zeta argument and energy or
higher-derivative estimates for fluid solutions concern different state spaces,
operators and limiting questions. The Riemann hypothesis concerns zero
locations; the three-dimensional Navier–Stokes problem concerns global regularity.
Neither is equivalent to a `Fix(G)^⊥` obstruction or to the other problem.

The function
[`structural_coherence`](../src/tnfr/metrics/common.py) computes the same algebraic
formula `C = 1/(1+|ΔNFR|+|dEPI|)` wherever a caller supplies those two
quantities. Reusing that function does not prove that independently defined
domain pressures, equilibria or trajectories are mathematically equivalent.
Cross-domain uses must specify how `ΔNFR` and `dEPI` are realized and validate
the resulting observer in that domain.

### 2.5 A structural-history analogy (the temporal spine)

Section 2.4 compares interfaces across domains. This section arranges selected
engine and auxiliary-model results as a **structural-history analogy**. They do
not form one demonstrated trajectory from a common initial state, so the
ordering is expository rather than a derived cosmology.

1. **Initialization and generation.** `EPI = 0` is a valid state and grammar U1
   requires a generator for a standalone history. Emission and coherence can be
   measured along that engine history. The unit-winding state used in the
   Kibble-like comparison is constructed separately; no implemented bifurcation
   has been shown to generate it (§7.5).
2. **A relaxation arrow.** In fixed homogeneous EPI diffusion,
   `τ = 1/(νf·λ₂)` is the slow-mode timescale and the Dirichlet energy falls
   monotonically (§4.4). Other capacity and topology regimes use different
   bounds; this supplies a structural-time diagnostic, not an identity with
   physical time.
3. **Structure formation (coarsening).** From an inhomogeneous early field the coherent domains
   **merge** over structural time — the coherent scale grows, an emergent structure-formation
   history.
4. **A growing causal horizon.** On the conservative face a perturbation spreads at a finite
   emergent speed, so the causally-connected region grows ~linearly — an emergent expanding causal
   horizon (§5.1), the nearest TNFR-native analogue of an "expansion" (on a *fixed* emergent metric).
5. **A regime-dependent fate.** On a fixed symmetric graph with positive
   capacities, the passive diffusive face relaxes to a componentwise-constant
   field, which is uniform when the graph is connected
   (§4.3). A separately declared continuously driven normal form can instead sustain
   structure (§6.3); its operator labels do not establish U2 compliance.

**The remaining sections provide scoped readings, not stages of one proven history.** §3
describes graph geometry; §4 restricted diffusive results; §5 auxiliary conservative
models; §6 synchronization diagnostics and separate normal forms; §7 and §9 winding
and particle-physics comparison programs. The static synthesis (§2.4) and temporal
scenario (§2.5) are two organizing axes of this taxonomy. Their juxtaposition does
not make every listed construction a trajectory of one dynamics
([emergent_structural_cosmology.py](../benchmarks/emergent_structural_cosmology.py),
[emergent_structural_genesis.py](../benchmarks/emergent_structural_genesis.py)).

> **Honest boundary — ANALOGY of scale.** This is the network's *own* emergent macro-history: the
> heat-semigroup H-theorem (§4.4), curvature/Ising coarsening, and a lattice light cone (§5.1),
> re-read as one structural history. "Cosmology" here is an **analogy of scale** — a structural
> re-expression of the abstract network's large-scale evolution, carrying the same DERIVED/ANALOGY
> labels as the rest of the document.

---

## 3. Graph-derived geometric read-outs

The canonical EPI channel propagates with `L_rw = I − D⁻¹W`
(`ΔNFR_epi = −L_rw·EPI`, exact). For a symmetric nonnegative conductance graph
without isolates,
`L_sym = D^(1/2) L_rw D^(−1/2)`: `L_sym` is a symmetric isospectral
representative of `L_rw`, not the same nodal operator. Their eigenvalues agree,
while their eigenvectors are related by degree scaling; the matrices coincide
on regular connected components. The auxiliary graph-wave model must declare
which representative supplies its stiffness. The isotropic harmonic substrate
uses identity stiffness. Connectivity is primitive; every metric or spectral
geometry below states which graph operator generates it.

### 3.1 Emergent metric — distance is derived, not imposed — **DERIVED**

For a symmetric conductance graph, the combinatorial Laplacian `B=D-W`
defines the **effective resistance**
`R_eff(i,j) = B⁺_ii + B⁺_jj − 2B⁺_ij` (a true metric; symmetric, non-negative, triangle
inequality — [effective_resistance](../src/tnfr/physics/structural_diffusion.py), ex.124). It is
**not** the hop count: it sees *all* parallel paths (transport difficulty). Measured — a ring's
antipodal nodes give `R_eff = 50` vs `hops = 100` (two parallel paths halve it); a tree's two
leaves give `R_eff = hops = 12` (a unique path, no shortcut). Distance is **derived**; it
coincides with the imposed hop count only when the path is unique.

### 3.2 Dimension readouts — **MIXED: exact definitions and finite measurements**

There are several distinct dimension-like readouts. They can be compared on a
declared graph family but are not generally one quantity.

**(a) The ambient spectral dimension `d_s`** — the dimension an *arbitrary* network *carries*.
From the heat-trace return probability `p(t)=Z(t)/n ~ t^{−d_s/2}` (`Z=Σ e^{−λ_k t}`) it emerges
from the spectrum: measured `d_s ≈ 1.01` (ring), `2.22` (2D torus), `3.36` (3D torus) — toward
1/2/3 with finite-size bias (ex.134). It is *measured from the dynamics, not declared* — but for a
generic graph it is a **free input** (a THOL tree gives `≈ 1.6`, resonant coupling tunes it,
[emergent_base_dimension.py](../benchmarks/emergent_base_dimension.py)): no bare network singles
out `d = 3`.

**(b) The simplex grade and representation multiplicity.** A
maximally-coupled cluster of `k+1` mutually-resonant NFRs is the 1-skeleton `K_{k+1}` of the
`k`-simplex; its Laplacian multiplicity `k` is the standard-irrep dim of `S_{k+1}` = the emergent
cardinal readout (§0), while `k` is separately the simplex grade. These equal
integers in that construction but remain different mathematical objects
([emergent_simplex_dimension.py](../benchmarks/emergent_simplex_dimension.py)). The canonical AL + U3 dynamics
can build the configured simplex sequence
([emergent_dimension_dynamics.py](../benchmarks/emergent_dimension_dynamics.py)).

**Self-similar comparison.** A corner-glued simplex gasket has exact similarity
dimension `d_H=log(m)/log(2)` and theoretical spectral dimension
`d_s=2log(m)/log(m+2)`. They are different. No continuum convergence, THOL
execution or identity with the
`U(2)` substrate sector count is claimed.

**The shell read-out (the atom).** A *multi-shell* coherent form (a THOL nest) inherits the grade
as its **shell degeneracy**: every shell has degeneracy = the simplex grade = the emergent
dimension ([emergent_atomic_shells.py](../benchmarks/emergent_atomic_shells.py), exact), so the
*atom's* shell structure is a read-out of the form's dimension — not of an imported spatial ball.
The cumulative shell closures **co-occur** with the `U(grade)` isotropic-oscillator magic numbers
(grade 2 → the 2D quantum-dot tower `2,6,12,20`, matching the substrate's own locked `U(2)`; grade
3 → the 3D-oscillator / nuclear `2,8,20,40`).

> **Honest boundary.** The shell degeneracy, selected simplex grade and
> representation multiplicity coincide numerically in the configured
> construction; this does not identify them with Hausdorff, spectral or
> physical dimension. The `U(grade)` magic-number tower is a **co-occurrence** (the Sierpinski localized
> modes take the largest closures), **not** a clean emergence. This reaches only the
> **independent-particle** skeleton. The full **chemical** periodic table
> (`2,10,18,36,54,86` = SO(4,2)/Madelung) needs the two-body screening correction — which is **the
> same `Fix(G)^⊥` wall** (§7.1) that traps the prime fine structure and `S(T)`: the *atom* and the
> *integer* share one structure (reachable **cardinals** ⊕ an unreachable **fine-structure
> residue**). The fixed point `ΔNFR = 0` is shared at the **predicate** level only — a prime (an
> irreducible *generator*) and a noble gas (a saturated *closure*) are *both* zero-pressure fixed
> points of their respective `ΔNFR` encodings, but land on *different* integers (noble-gas `Z` are
> composite): one predicate, many read-outs, **not** one number.

### 3.3 The tetrad — the canonical diagnostic basis — **CANONICAL / reconstruction open**

Four structural fields `(Φ_s, |∇φ|, K_φ, ξ_C)` form the canonical diagnostic basis (the discrete
derivative tower; [MINIMAL_STRUCTURAL_DEGREES.md](MINIMAL_STRUCTURAL_DEGREES.md)). Universal
minimality and complete state reconstruction remain open. Only **π** is a genuine structural scale;
the field bounds are of two different kinds:

- **Kinematic (geometric) — exact.** Both `|∇φ|` and `K_φ` are **wrapped angles**
  (`|∇φ| = mean|wrap Δφ| ∈ [0,π]`, `K_φ = wrap(φ − circmean) ∈ (−π,π]`), so `|∇φ| ≤ π` and
  `|K_φ| ≤ π` hold for *any* configuration, parameter-independently. The
  `|K_φ| < 0.9π ≈ 2.83` value is a selected warning margin inside that exact
  geometric bound.
- **The other thresholds are not structural constants.** The selected `|∇φ|`
  early-warning policy (`π/16 ≈ 0.196`) sits below the wrap bound, while a fair
  test finds `|∇φ|` at the synchronization onset is
  **≈ 0.29 and varies with the disorder `σ`** — not a fixed constant. The `Φ_s` values
  (per-node `π/4 ≈ 0.785`, drift `π/2 ≈ 1.571`) are selected warning policies; and the
  fitted coherence length is state- and estimator-dependent. `1/√λ₂` is a
  spectral comparison or fallback, not a universal identity for fitted `ξ_C`.

> **Honest boundary.** Only **π** is a genuine structural scale (the phase-wrap bound of the phase
> sector). The other field scales are not structural constants: the `|∇φ|` onset is a σ-dependent
> dynamical transition (≈ 0.29), the `Φ_s` values are selected policies, and the
> spectral value `1/√λ₂` is a comparison/fallback for the state-dependent `ξ_C`
> estimator. The tetrad is the
> canonical diagnostic basis; complete state reconstruction remains open.

**Genuine relationships (verified).** A fresh study found the real structure
behind the four fields: `K_φ` agrees with `L_rw·φ` in the smooth,
consistent-branch limit with matching conventions (**corr = 1.000** on the
recorded protocol). The fitted `ξ_C` can be compared with the spectral fallback
`1/√λ₂`, but no universal equality is established. The organizing axis is **local**
phase derivatives (`|∇φ|`, `K_φ`, both π-bounded) vs **non-local** source/correlation (`Φ_s`, `ξ_C`),
across the derivative orders — *not* four separate constants.

> **Honest boundary.** §3 is the **intrinsic geometry of the diffusion operator** — standard
> spectral graph theory (Kirchhoff 1847; commute time, Chandra et al. 1996) and the spectral
> dimension of anomalous diffusion. It is the metric/dimension the substrate *carries*; it is
> **not** a derivation of curved physical spacetime.

---

## 4. The diffusive channel — thermodynamic correspondences

On a fixed graph, the isolated EPI channel is exactly graph diffusion. The
thermodynamic labels below are correspondences to standard diffusion and
stochastic models, not a derivation of all thermodynamics.

### 4.1 The EPI channel is a diffusion equation (the form of heat flow) — **DERIVED (exact)**

$$\Delta\mathrm{NFR}_{\text{epi}}(i) = \overline{\mathrm{EPI}}_{\mathcal N(i)} - \mathrm{EPI}(i) = -(L_{\mathrm{rw}}\,\mathrm{EPI})(i),\qquad \frac{\partial \mathrm{EPI}}{\partial t} = -\nu_f\,L_{\mathrm{rw}}\,\mathrm{EPI}.$$

The EPI channel of the nodal equation is, *exactly*, a discrete **diffusion equation** with
diffusivity `νf` ([structural_diffusion.py](../src/tnfr/physics/structural_diffusion.py)) — the same
mathematical form as the heat equation `∂T/∂t = D∇²T`. **What is shared is the equation, not the
quantity:** `EPI` is a structural configuration, **not** heat or temperature, and **nothing thermal
is computed** (no temperature, no energy, no joules) — only the spreading of `EPI`. The one
diffusion law *manifests*, at the thermal scale, as heat flow (the form is DERIVED; the
structural-priority reading is POSITED, §1.1). The same caveat governs every `=` heading below and
every "manifests as" in this document: it equates **equations / structures**, never the underlying
substances.

### 4.2 Structural relaxation clocks — **DERIVED in scoped diffusion models**

On a fixed graph with homogeneous constant `νf`, isolated eigenmodes decay as
`e^{−νf λ_k t}` and the slowest nonstationary mode is set by `νf·λ₂`.
With heterogeneous fixed capacity the relevant rate is the first positive
generalized eigenvalue of `(D-W, diag(d_i/ν_i))`; time-varying capacity and
switching topology require their own conditional common-function and
exact-common-metric bounds. Declared affine EPI resets can be composed only
after their gain is certified in that same metric. See
[the diffusion stability theorem](TNFR_DIFFUSION_STABILITY_THEOREM.md). These
are dynamical relaxation rates in structural time, not a proof that physical
time itself emerges.

### 4.3 Coherence and equilibrium — **DERIVED**

`C = 1/(1 + mean|ΔNFR| + mean|dEPI|)` is the parameter-free proximity to equilibrium.
Because both terms are nonnegative, `C→1` exactly when both
`mean|ΔNFR|→0` and `mean|dEPI|→0`. Pressure decay alone implies this only in a
scope where the nodal relation `dEPI=nu_f*ΔNFR` holds and `nu_f` remains
bounded. On a fixed symmetric nonnegative graph with positive capacities,
the diffusion equilibria are exactly the fields that are constant on each
connected component. A connected graph therefore has one uniform consensus
field; a disconnected graph may have a different constant on each component.
At these pressure equilibria `ΔNFR=0`, so the source-aggregated field `Φ_s`
vanishes. Zero-capacity nodes and other pressure channels require separate
stationarity conditions.

### 4.4 The arrow of time = the structural H-theorem — **DERIVED (proven)**

The Dirichlet energy `F = ½Σ A_ij(EPI_i − EPI_j)²` is **monotonically non-increasing** under the
fixed symmetric diffusion flow. Its decay is bounded by an exponential envelope
set by the appropriate spectral gap; equality with one exponential occurs only
for an isolated eigenmode. This is a proven Lyapunov functional of the heat semigroup
(conservation theorem §8.6, [ex.135](../examples/08_emergent_geometry/135_arrow_of_time_h_theorem.py)).
→ the thermodynamic arrow of time / entropy increase.

### 4.5 Conservation — **DERIVED**

For fixed homogeneous positive capacity, the degree-weighted total
`Σ_i d_i·EPI_i` is conserved. For fixed heterogeneous positive capacity, the
invariant is `Σ_i(d_i/ν_i)·EPI_i`; `Σ_i d_i·EPI_i` is not generally conserved.
These statements apply componentwise on disconnected graphs, with isolates
stationary under their zero Laplacian rows. The proposed charge
`Q = Σ(Φ_s + K_φ)` is a diagnostic of the
auxiliary conservation construction; general grammar evolution does not conserve it.

### 4.6 Comparison of distinct relaxation diagnostics — **MIXED**

Several models expose relaxation-like quantities, but only fixed homogeneous
EPI diffusion has the clock `νf·λ₂`:

| Phenomenon | Functional | Anchor |
|-----------|-----------|--------|
| Diffusion (heat) | Dirichlet energy `F` | §8.6 (proven) |
| Structural-field candidate | `E = ½Σ(Φ_s² + \|∇φ\|² + K_φ² + …)` | no general decay theorem |
| Symbolic-sequence diagnostic | Parry / Markov model | [ex.150](../examples/08_emergent_geometry/150_emergent_grammatical_pattern_parry.py) |

No identity currently equates these clocks. Comparing them requires a declared
state map and time parametrization.

> **Honest boundary.** The diffusive face has **no causal cone** (the heat kernel has infinite
> support — a perturbation reaches every node instantly, arrival time `∝ k²`, front `~√t`). It is
> thermodynamic, **not** relativistic. The cone lives in the conservative face (§5).

### 4.7 Thermal fluctuations — Brownian motion and the fluctuation-dissipation theorem — **DERIVED**

Adding thermal noise to the diffusive face (the overdamped dynamics, whose mobility is `νf` —
AGENTS.md's Stokes/Einstein mobility) gives a Langevin process `∂u/∂t = −νf·L u + ξ`
(`⟨ξ_i ξ_j⟩ = 2νf T δ_ij`) whose equilibrium reproduces the primary thermal observables:

- **Equipartition / fluctuation-dissipation.** Each nonzero mode carries energy
  `T/2`: `λ_k·⟨u_k²⟩ = T`. Kernel modes have no restoring force and are
  excluded; there is one such mode per connected component.
  Slow modes fluctuate more
  (`⟨u_k²⟩ = T/λ_k`), exactly compensating their weak restoring force — the fluctuation–
  dissipation balance.
- **Einstein relation.** In the declared Langevin comparison, the short-time
  diffusion coefficient is `D = μ·T` with `μ = νf`.
- **Mobility vs temperature.** Doubling `νf` rescales the relaxation time but leaves the
  equilibrium fluctuations (set by `T` alone) unchanged — the separation of mobility (kinetic)
  from temperature (equilibrium) that *is* the Einstein content.

Maps to observables: **Brownian motion**, **thermal fluctuations**, the **fluctuation-dissipation
theorem**, **Johnson–Nyquist noise**.

> **Honest boundary.** This is the standard overdamped Langevin /
> Ornstein–Uhlenbeck process on a selected diffusion operator. The repository
> currently provides no maintained benchmark anchoring numerical equipartition
> or Einstein-ratio estimates in this section.

### 4.8 Emergent transport — Ohm's law and conductivity — **DERIVED**

The combinatorial Laplacian `L = D − A` is the **conductance matrix** (Kirchhoff): the structural
operator is a resistor network, and the effective resistance `R_eff` (§3.1) *is* Ohm's law.

- **Ohm's law.** Injecting a current `I` between two nodes (`L V = I`) gives a voltage drop
  `V = I·R_eff` (measured exactly) — `V = IR`.
- **Series / parallel.** A chain of `n` unit resistors has `R = n` (series, `R ∝ length`); a ring's
  two parallel paths give `R = ℓ/2` (parallel composition) — the circuit composition laws.
- **Resistivity.** The busbar-to-busbar resistance of a `d`-dimensional block scales as
  `R ∝ L^{2−d}` (measured exponents `+1.10` (1D), `+0.10` (2D), `−0.87` (3D) vs predicted
  `+1, 0, −1`): resistance **grows** in 1D, is a **constant sheet resistance** in 2D, and
  **shrinks** in 3D — an intensive bulk **conductivity** emerges.
- **Nernst–Einstein.** Conductivity tracks the diffusion coefficient `D = νf·T` (§4.7): the *same*
  mobility `νf` sets transport and fluctuations.

Maps to observables: **Ohm's law**, **resistance**, **conductivity / resistivity**, **circuits**.

> **Honest boundary.** This is **Kirchhoff resistor-network theory** (1847) — the graph Laplacian
> as the conductance matrix, the effective resistance as Ohm's law — re-expressed on the canonical
> structural operator. Standard, nothing new derived.

---

## 5. Auxiliary conservative models and relativistic analogies

The graph-wave model is second order and wave-like. It is separate from the
isotropic harmonic substrate. Perturb one node and measure the **arrival time**
`t_arr(k)` at distance `k` (spectral propagation on a chain via `L_sym`):

### 5.1 A causal light cone — **DERIVED (nuanced)**

| Regime | Dynamics | Arrival law (measured) | Causal structure |
|--------|----------|------------------------|------------------|
| Diffusive | 1st order `∂u/∂t = −νf L u` | `t_arr ∝ k²` (R²=**0.9997**) | none — infinite speed |
| **Wave** | 2nd order `∂²u/∂t² = −c² L u` | `t_arr ∝ k` (R²=**0.9999**), `v≈0.755` | **a light cone** |

A finite propagation cone is measured in this graph-wave model. It is not the
second-order flow of the isotropic substrate in §5.3 and does not establish a
causal law for the full nodal engine.

### 5.2 Approximate Lorentz invariance at low energy — **DERIVED (nuanced)**

The wave dispersion `ω(k) = c·√λ(k)` carries an **approximate relativistic symmetry at long
wavelength**:

- **Linearity (1D).** A ring is regular, so `L_sym = L_rw` for the selected
  normalization. From their common ring spectrum (`λ = 1 − cos q`), `ω(q)` is
  **linear at low `q`**: a fit `ω = v·q` gives `v = 0.696`, **R² = 0.9998** (a "massless"
  relativistic dispersion); it bends sub-linearly at the zone boundary (`ω(π)/[v·π] = 0.65`).
- **Isotropy (2D).** On a **square** lattice, `ω(k)` along the axis vs the diagonal has ratio
  **1.001 at `|k|=0.2`** (a *round* light cone — emergent rotational invariance), rising to
  **1.27 at the zone boundary** (where the square lattice finally shows).

So a **round, linear, relativistic light cone emerges at low energy**, broken at the lattice scale.

### 5.3 The auxiliary symplectic geometry — **EXACT FOR THE SPECIFIED MODEL**

Embedding graph-field snapshots in independent ambient coordinates specifies a symplectic
phase space (pairs `(K_φ,J_φ)`, `(Φ_s,J_ΔNFR)`), a harmonic Hamiltonian, its Noether charges,
and a **U(2) polarization** with conserved
Stokes parameters on a per-node Poincaré sphere
([symplectic_substrate.py](../src/tnfr/physics/symplectic_substrate.py)). These are exact
properties of the declared ambient model and a structural correspondence with classical
Hamiltonian mechanics and wave polarization. They do not show that extracted graph states fill
the ambient space or that engine operators preserve its symplectic form. Its global oscillator
rotations generate conserved model charges; they are distinct from the node-dependent pure-gauge
coordinate rephasing of §7.2.

> **Honest boundary.** The finite speed is the **lattice wave speed** (set by `νf` and the graph
> spectrum), only **approximately** Lorentz-invariant at low energy — a finite causal cone, **not**
> exact special relativity or curved spacetime (this is the standard emergent relativistic symmetry
> of lattice field theories: Dirac cones, critical points). The Stokes/Poincaré structure is
> **classical**, un-entangled polarization — **not** a quantum state (this matters for §9).

### 5.4 Emergent optics — refraction and the refractive index — **DERIVED (skeleton)**

A region of different wave speed (different local `νf` / stiffness) is an emergent **refractive
medium**. Driving a monochromatic wave across an interface (`c_1 = 1.0`, `c_2 = 0.6`):

- **Refractive index / wavelength change (clean).** At normal incidence the wave **slows and
  shortens** in the denser region: measured `λ_2/λ_1 = 0.598` vs the predicted `c_2/c_1 = 0.600`
  — an emergent **refractive index** `n = c_1/c_2 = 1.67` (frequency conserved, wavelength set by
  the local speed).
- **Refraction (Snell's direction).** At oblique incidence the beam **bends toward the normal**
  (`θ_2 < θ_1` for every angle), following **Snell's law** `sin θ_1 / sin θ_2 = c_1/c_2` — set by
  tangential-wavevector conservation at the interface (exact for this equation; the quick
  numerical angle extraction is approximate, but the wavelength ratio above pins the index cleanly).

Maps to observables: **refraction**, the **refractive index**, **Snell's law**, **lenses**.

> **Honest boundary.** This is the standard **variable-coefficient wave equation** (classical
> optics / acoustics) on the conservative face — the refractive index *is* the wave-speed ratio.
> It is the optical face of the conservative regime, **not** a derivation of Maxwell's equations
> or QED.

---

### 5.5 The pulse — the network rhythm and the per-NFR pulses — **DERIVED (within TNFR)**

The conservative face is, most simply, a **sustained vibration**: every structural eigenmode
oscillates at `ω_k = √λ_k`, so the substrate *keeps a rhythm*. This rhythm has **two scales**, both
read closed-form from the spectrum and the node state — no time integration.

- **The collective pulse (the network rhythm).** The leading resonances `ω_k = √λ_k`, the
  fundamental (the slowest non-uniform resonance), the dominant **beat** (the slowest
  `ω_j − ω_k`), the self-similar (fractal) spectral multiplicity, and the vibration energy
  `½Σλ_k`. This is the rhythm the *whole network* plays
  ([`compute_emergent_pulse`](../src/tnfr/physics/structural_diffusion.py); SDK `net.rhythm()`).
- **The per-NFR pulse (the bricks).** Every NFR is itself a phase oscillator — the single-node
  reduction of the nodal equation `∂EPIᵢ/∂t = νfᵢ·ΔNFRᵢ` — pulsing at its **own** structural
  frequency `νfᵢ` with phase `φᵢ`. *Resonance* couples those pulses: the **local** phase synchrony
  per NFR (`local_phase_sync`), the **collective** Kuramoto order `R` (`kuramoto_R_psi`), and the
  U3 admissibility gate `Δφ_max = π/2`. The collective pulse **emerges** as the per-NFR pulses lock
  (`R → 1`) ([`compute_nodal_pulse`](../src/tnfr/physics/structural_diffusion.py); SDK
  `net.resonance()`).

The equilibrium and pulse quantities are separate read-outs. `ΔNFR = 0`
defines a pressure-equilibrium snapshot, while the graph-wave spectrum defines
an auxiliary conservative rhythm. A damped graph-wave model can be chosen whose
slow limit approaches diffusion, but the repository has not shown that the
conservative pulse generates engine equilibria or that every engine trajectory
follows that auxiliary flow.

**The fractal-pulse hypothesis.** On a selected self-similar graph, the spectrum
of a declared Laplacian may form bands. Pure EPI modes then relax according to
that diffusion generator. Extending the same rate law to wrapped phase requires
the small-spread linearization and matching phase weights; outside that regime
it is not valid. Fine-to-coarse phase locking is therefore an auxiliary finite
protocol, not a consequence of U5 alone
([`emergent_fractal_pulse.py`](../benchmarks/emergent_fractal_pulse.py)).

**The arithmetic face.** The same pulse read on the *arithmetic* NFR — the residue
Cayley network `Cay(ℤ/n, R_k)` — has a tone-count equal to the **proved cyclotomy
law** `s_k(p) = gcd(k, p−1) + 1`: a prime is its **most degenerate chord** (the
silent mode + two tones of multiplicity `(p−1)/2`), and composites split it
multiplicatively into the factorization type
([TNFR_NUMBER_THEORY.md §9.13](TNFR_NUMBER_THEORY.md)).

**The music of the NFR — music as a lens on structural frequency.** Music is used
here as an *epistemic lens*, not as audio: the frequencies are **structural**
(`νf`, `ω_k = √λ_k`, in `Hz_str`), and the point is to read what happens
*structurally*, not to make sound. Through that lens the pulse is a whole musical
structure — pitch (`ω_k`), chord (the distinct tones), timbre (the eigenvalue
multiplicities), beats (`ω_j − ω_k`), and the standing **nodes** (the `ΔNFR = 0`
NFRs, §3). **The dynamical regime follows the dimension** (§3.2): a **1D coherent
thread** (a chain) is *harmonic* — its structural-frequency ratios are the just
consonances (octave `2:1`, fifth `3:2`, fourth `4:3`, measured to <0.5 %), a
**pitched** regime, so the **harmonic series itself is emergent**; a **2D+** form (a
membrane / Chladni plate) is *inharmonic*, an **unpitched** regime where consonance
is no longer a frequency ratio but **phase coherence** (the U3 gate `Δφ_max = π/2`,
`R = cos(Δφ/2)`); the `0D` clique is one rigid tone (a bell). Only the **tempered
scale** (equal temperament) is an imposed human convention. Distinct primes are the
**independent voices** of the polyphony (the Euler product, decoupled ladders). And
the lens also has an inverse-observer limitation: isospectral graphs share the
same modal frequencies, and the selected `rho(pq)` statistic can coincide for
distinct semiprimes. Those are two separate non-injectivity examples. They show
that the selected read-outs do not always recover identity; they do not define a
single `Fix(G)^⊥` obstruction
([`emergent_musical_nfr.py`](../benchmarks/emergent_musical_nfr.py)).

> **Honest boundary.** This is the standard standing-wave spectrum (`ω_k = √λ_k`), beat
> interference, and Kuramoto phase-locking on the conservative face — re-read in TNFR terms (each
> NFR a phase oscillator, resonance the coupling). It surfaces existing canon (`νf` the per-NFR
> frequency, `local_phase_sync` / `kuramoto_R_psi` the resonance); the fractal cascade is the
> banded-spectrum synchronization of a self-similar graph, re-read as operational fractality (U5)
> in time; the *musical* reading is the same **structural-frequency** spectrum used as a lens
> (not audio) — the harmonic series and the just consonances (octave/fifth/fourth) **are**
> emergent on a 1D coherent thread, while only **equal temperament** and the chosen scale are
> imposed conventions; a 2D+ form is inharmonic (unpitched), where consonance is the U3 phase
> gate and the Kac wall is the inverse spectral problem. It derives no new physics.

---

## 6. Emergent collective phenomena

### 6.1 A Kuramoto-type phase channel — **SCOPED CORRESPONDENCE**

When the selected phase interaction is `h(Δφ)=sin(Δφ)`, the auxiliary phase
equation has Kuramoto form. This permits controlled comparisons with standard
coupled-oscillator systems. It does not identify every synchronization process,
nor does it make their domain-specific dynamics trajectories of the EPI nodal
equation.

### 6.2 Criticality in an auxiliary Landau graph model — **DERIVED IN MODEL / OPEN FOR TNFR OPERATORS**

Consider the separately declared reaction-diffusion normal form
`∂u/∂t = r·u − νf·L u − u³`, with the conserved mean projected out. Its
linearized Fiedler-mode threshold is `r_c = νf·λ₂`. This identity belongs to
that auxiliary model: the grammar rule U2 constrains operator words and does
not define the continuous reaction coefficient `r`. Equating the two without
an operator-to-generator derivation was an error. Within the normal form:

- **Order-parameter onset.** The steady amplitude `m(r)` is **zero below `r_c`** (disordered /
  uniform) and **rises continuously above it** (ordered) — a 2nd-order transition.
- **Critical slowing down.** Linearization gives the model relaxation time
  `τ = 1/|r − r_c|`; this diverges in the analytic infinite-time model as
  `r → r_c`. Finite sampled trajectories do not prove a thermodynamic limit.
- **Emergent pattern.** Above `r_c` the field condenses into the **Fiedler eigenvector** — the
  longest-wavelength spatial mode — a Turing-like pattern.

These are standard continuous-transition behaviors of the selected Landau
normal form. They supply a controlled comparison protocol for TNFR telemetry,
not a derivation that canonical operator dynamics has the same transition.

> **Boundary.** The exponent `β = 1/2` follows from cubic saturation in this
> mean-field normal form. The engine's `phase_transition` utility fits an
> effective time-series slope and does not compare it with this value. The
> balanced finite-size diagnostic reports slopes against node count without
> promoting them to thermodynamic exponents. An U2-to-criticality map remains
> open until operator events are related to a continuous generator.

### 6.3 Sustained structures far from equilibrium — the driven regime — **AUXILIARY MODEL**

The diffusive channel above relaxes; the conservative comparison models
oscillate. A separate continuously driven normal form can model a sustained
far-from-equilibrium pattern. With a
continuous drive carrying the U2 balance (a destabilizer plus a stabilizer), the driven nodal
dynamics `dz/dt = (μ+iω₀)z − |z|²z + K·∇²z` produces a **self-sustained coherent structure**:

- **Existence (a Hopf onset).** For `μ<0` the field relaxes to the **dead** state; for `μ>0` a
  **sustained** structure appears (`|z|≈√μ`, permanently active), **coherent in a window** (`R` up
  to 0.84 near onset) that gives way to **turbulence** at large drive (`R→0`).
- **Saturation.** The displayed cubic term `−|z|²z` is itself the stabilizer and
  bounds the amplitude near `√μ`. Blow-up belongs to a different equation in
  which this saturation is removed; it is not produced merely by setting
  `μ>0` in the displayed model. Calling this a continuous analogue of U2 is an
  interpretation, not a grammar theorem.
- **The slogan, measured.** In the coherent window the collective coherence is `R=0.84` **with**
  coupling and **collapses to `R=0.09`** when the coupling is cut: *maintained by resonance,
  dissolving when coupling fails*.

This is the **first regime here that is not relaxation** — the driven, dissipative-structure
counterpart of the two *passive* faces (§2).

> **Honest boundary.** This is the Stuart–Landau / complex Ginzburg–Landau model (the Hopf normal
> form; Prigogine dissipative structures, Kuramoto) — known physics. What is new *here* is the
> regime itself (beyond the equilibrium derivations) and its TNFR reading — the sustained structure
> as the grammar's U2 balance made dynamic. It does **not** yet yield a prediction the standard
> dissipative-structure framework lacks.

---

## 7. Matter-comparison models — winding classes and auxiliary dynamics

This section separates exact graph identities from comparison models. A winding
class is a property of a declared phase map under explicit regularity conditions.
Calling it charge, matter or a particle is an analogy; interaction equations must
be identified independently.

### 7.1 The three-level structure: stage → occupant → process — **DERIVED facts**

- **(a) The stage — a discrete standing-mode spectrum.** On a finite network
  `L_sym` has a discrete spectrum. Its kernel is spanned by componentwise-constant
  modes and is one-dimensional only for a connected graph. On a connected 1D box,
  the low modes follow the particle-in-a-box law `λ_k ∝ k²` (measured log-log
  slope **1.997**), ordered by Courant nodal-domain bounds, with homogeneous-
  capacity lifetimes `1/(νf·λ_k)`. It says **where** an excitation may sit.
- **(b) The occupant coordinate — an integer winding class.** On a declared
  oriented cycle, a non-ambiguous node-phase map has winding `W ∈ ℤ`
  ([emergent_particles.py](../src/tnfr/physics/emergent_particles.py)). The
  preferred API is `WindingSector` / `classify_winding_sector`; the legacy
  particle-named aliases remain for compatibility. The implementation labels
  only zero, unit and multi-winding classes; it does not
  infer bosonic/fermionic statistics, matter/antimatter or a particle species.
  The sign reverses with cycle orientation. Preservation across evolution is
  conditional on retaining the cycle and avoiding zeros and the wrap branch.
  Supporting energy and historical-Q means are explicitly whole-graph snapshot
  telemetry, including when the declared cycle is a proper subgraph.
  [Example 133](../examples/08_emergent_geometry/133_psi_topological_defects.py)
  separately shows that raw `arg(Ψ)` face winding is a static coordinate degree
  and changes under node-dependent U(1) rephasing.
- **(c) The process — auxiliary defect interaction.** Attraction, translation,
  orbiting and annihilation claims in this repository belong to separately
  posited point-vortex or Coulomb-gas comparison equations (§7.3). They have not
  been derived from the canonical coupling operator or the nodal equation.
- **(d) Ring mode/winding coincidence in a selected construction.** On a ring,
  the complex Fourier vector `exp(2πikj/n)` is an exact eigenvector with
  eigenvalue `λ_k = 1 − cos(2πk/n)` under the declared normalization and has
  winding `k` within the branch-resolved range. This is a joint property of the
  chosen vector; it does not identify a generic extracted TNFR `Ψ` field, a
  particle charge, or quantum wave-particle duality. A
  real narrow-band ripple built from a few low-`k` modes stays at winding `0` for every amplitude
  tested (up to 20 rad) — a generic wave excitation of the vacuum is topologically trivial; only
  the specific complex mode carries nonzero winding. This is the classical
  topology of maps `S¹→S¹` evaluated on a selected graph eigenvector
  ([emergent_wave_particle_correspondence.py](../benchmarks/emergent_wave_particle_correspondence.py)).
  The genuinely quantum content of wave-particle duality — a probability amplitude, the Born rule,
  single-particle interference statistics, a physical `ħ` — is the classical-substrate frontier of
  §9.2 (OPEN).

### 7.2 Auxiliary local-U(1) coordinate model — **DERIVED (pure-gauge identity) / ANALOGY (electromagnetism)**

[gauge.py](../src/tnfr/physics/gauge.py) defines a local rephasing coordinate for
the diagnostic complex field `Ψ`. Its derived edge connection
`A_ij = arg Ψ_j − arg Ψ_i` is the exact one-form `d(arg Ψ)`: oriented cycle sums
therefore telescope to zero, up to floating-point closure residuals, and the
covariant graph Laplacian is unitarily equivalent to its ordinary weighted graph
Laplacian. This proves covariance of an auxiliary coordinate description; it does
not supply an independent gauge degree of freedom, non-zero curvature, magnetic
flux, a gauge vortex, or gauge-field dynamics. Electromagnetic language is an
**ANALOGY** unless an independent edge field and its dynamics are derived from the
nodal equation and canonical operators.

The complex covariance statement applies on the non-zero support of `Ψ`. At a
zero, `arg Ψ` is undefined and the implementation fixes it to zero as a
deterministic convention; the covariant-difference magnitude remains the
convention-independent endpoint-amplitude contrast. This coordinate identity is
not a symmetry theorem for nodal trajectories or canonical operators.

### 7.3 Auxiliary point-vortex comparison — **ANALOGY**

When winding defects are inserted into the separately posited classical
point-vortex equations, two opposite circulations translate as a pair, two like
circulations orbit, and the three-vortex model carries the familiar invariants
`(H, |P|², L)` to integrator precision. Those equations provide a controlled
comparison model; the repository has not derived them as a trajectory of the
nodal equation or as a composition of the 13 canonical operators.

### 7.4 Composite matter — atoms, bonds, and bands — **DERIVED (skeleton)**

Localized **wells** in the structural operator (`H = L_sym − U·P_well`, a region that holds
coherence more strongly) bind discrete states out of the continuum, reproducing the tight-binding
hierarchy of bound matter (derived fresh, not assumed):

- **One well = an atom** — a discrete **bound state** splits off below the band (`E = −0.41` at
  `U=1`) with a **localized orbital** (participation ratio `≈1.9` of 81 nodes).
- **Two wells = a molecule** — the state splits into **bonding** (lower) and **antibonding**
  (higher); the splitting **grows as the wells approach** (`0.0002 → 0.27` as `d: 10 → 2`), the
  **covalent bond**, with the bonding level *below* the single-atom level — the molecule is
  **bound (stable)**.
- **Many wells = a band** — `N` wells give `N` levels broadening into a **band** (width saturating
  `≈0.084`), the tight-binding origin of solid-state bands.

Maps to observables: **atomic orbitals**, the **covalent bond** (molecular orbitals), and **band
structure**.

> **Honest boundary.** §7 combines exact finite-graph identities with explicitly
> auxiliary classical models: discrete spectra and integer winding are graph
> results; the point-vortex, Coulomb-gas, local-U(1), and tight-binding pieces are
> comparison constructions under their own declared equations. Their coexistence
> in one vocabulary does not derive one model from another or derive any real
> particle's measured mass, charge, or spin, QED, or QFT. A genuine quantum
> particle (second quantization) needs ingredients the classical substrate lacks
> (§9).

### 7.5 A structural-genesis scenario — **MIXED CONSTRUCTION / Kibble analogy**

The benchmark juxtaposes engine states with a separately constructed winding
ring ([emergent_structural_genesis.py](../benchmarks/emergent_structural_genesis.py)):

1. **The vacuum.** `EPI = 0`; the nodal equation `∂EPI/∂t = νf·ΔNFR` remains mathematically
 defined for finite capacity and pressure, while grammar **U1a** requires a generator `{AL, NAV, REMESH}`
   to open any sequence (§1). "Something rather than nothing" is a structural necessity here, not a
   spontaneous event.
2. **Emission — the first form.** `AL` sources `EPI` from the vacuum (`∂EPI/∂t > 0`, `νf` activates):
   measured, `EPI` rises monotonically from `0` (`0 → 0.09 → … → 0.50`) — form where there was none.
3. **A coherent reference state.** The coherence flow can raise the recorded
   coherence; a separately declared zero-winding ring supplies the comparison.
4. **A unit-winding construction.** The benchmark creates `winding_ring(..., 1)`
   directly. It does not demonstrate that a validated destabilizer/stabilizer
   word generates this class from the preceding state.

> **Honest boundary.** Emission and coherence are engine operations; the final
> winding state is constructed rather than dynamically derived. Its resemblance
> to Kibble defect formation is an analogy and supplies no particle genesis.

---

## 8. Emergent information and computation

The **grammar** (U1–U6) is the generative syntax of canonical operator histories.
Auxiliary models above do not thereby become grammar-generated. Finite counts of
accepted flat sequences give a reproducible combinatorial diagnostic.

### 8.1 Flat-sequence prefix growth — **MEASURED**

The valid operator sequences form a **language** over the 13-operator alphabet. Counting them (the
canonical `validate_grammar`, derived fresh):

| `n` | 1 | 2 | 3 | 4 | 5 | 6 |
|-----|---|---|---|---|---|---|
| valid sequences `N(n)` | 2 | 9 | 84 | 852 | 9378 | 109920 |

The finite-prefix growth diagnostic `log₂(N_n/N_{n−1})` takes the recorded
values `2.17 → 3.22 → 3.34 → 3.46 → 3.55`, below the unconstrained
`log₂13 = 3.70` on these lengths. These six counts alone do not prove an
asymptotic rate. The current default-depth history projection is represented by
a 320-state reachable automaton and a 52-state minimal complete DFA, whose
numerical adjacency radius is `10.9560791442`. Runtime U3 and reference-dependent
U6 are external state checks; full U5 nesting is context-free and requires a
stack-like model.

### 8.2 Operator-frequency hierarchy in finite prefixes — **MEASURED**

At length five, the exact finite enumeration has `NAV` and `REMESH` at about
17.95% each and `ZHIR` at about 0.09%. Boundary roles and ZHIR's lifetime-IL plus
recent-destabilizer preconditions help interpret this sample; it is not an
asymptotic operator-frequency hierarchy.

Maps to observables: **formal languages** (the Chomsky hierarchy), the **Shannon channel capacity**
(the noiseless-channel coding theorem), and **symbolic dynamics** (subshift entropy).

> **Honest boundary.** Exact automata and monoid results apply to the constructed
> flat, default-depth symbolic/history projection. The selected Parry chain is a
> maximum-entropy Markov policy on its dominant component, not a TNFR physical
> equilibrium or H-theorem. Full U5 nesting is context-free, and U3/U6 require
> runtime state outside this symbolic automaton.

---

## 9. The particle / quantum frontier — forms derived, values & QM open

The reach from the §7 graph constructions toward physical particles and quantum
mechanics. Section 9.1 records exact properties of selected abstract graph
models; identifying those properties with a particle catalog, generations or
masses remains analogy or open conjecture. Sections 9.2–9.3 remain open.

### 9.1 Particle-like graph patterns — **DERIVED IN DECLARED MODELS / PHYSICAL IDENTIFICATION OPEN**

The repository constructs graph patterns labelled by structural read-outs.
Exact winding, spectral and symmetry statements about
those constructions remain useful mathematics. They do not establish that a
physical particle is an NFR, that the constructions reproduce the Standard
Model catalog, or that graph eigenvalues represent measured masses.

| Layer | Question | Status |
|-------|----------|--------|
| **1** | graph defect | **DERIVED IN MODEL** — integer winding `W ∈ ℤ` under the stated ring assumptions |
| **2** | selected graph-pattern catalog | **MODEL-DEPENDENT** — follows from chosen graphs and energies |
| **3 · form** | symmetry motifs | **DERIVED IN MODEL** — selected `Z₃` and nested-graph constructions |
| **3 · scale** | structural frequency | **OPEN** — the nodal equation supplies no universal `νf` fixed point |
| **3 · physical values** | masses and ratios | **OPEN / UNMAPPED** |

**Layer 2 — the selected graph catalog is structured (DERIVED IN MODEL).**
(a) The declared-cycle winding classes lie in **ℤ** when the cycle and branch
regularity hypotheses of §7.1b hold. This class does not specify quantum statistics.
(b) The energy of a charge-`W` structure scales as `E ∝ W²` (exact on the ring: the excitation
energy above vacuum is the integer squares `1,4,9,16,25`), so `|W|=2` costs more than two `|W|=1`
and splits in the selected 2D energy model (a `|W|=2` core has higher
self-energy than two separated `|W|=1`). Thus `|W|=1` is the unit winding in
that model; the physical-charge interpretation is not derived.
([emergent_particle_catalog.py](../benchmarks/emergent_particle_catalog.py),
[emergent_mass_charge_spectrum.py](../benchmarks/emergent_mass_charge_spectrum.py)).
(c) A localised coherent core — the maximally-coupled simplex `K_{g+1}` (§3.2) — binds only
**finitely** many internal states, so the generation tower is **truncated**: the **generation count
= the simplex grade = the cardinal** = the standard-irrep dimension of `S_{g+1}`, so graph theory
(spectral counting) and number theory (the cardinal, §3.2) are *one* count. **Grade 3** is the grade
whose THOL/Sierpinski nesting dimension `log 4 / log 2 = 2` matches the substrate's own **U(2)
fibre** (§3.2) — a self-consistency selection (the U(2)↔2 convergence is a *noted* convergence,
§3.2, not a derived identity).
([emergent_generation_count.py](../benchmarks/emergent_generation_count.py)).
(d) The construction carries a form-channel mode index `n` with
`ω_n = √λ_n`, separately from phase winding `W`. This supplies a two-index
graph-pattern label `(W,n)`. Calling it a same-charge mass tower would require
the physical map that is currently absent.
([emergent_internal_quantum_numbers.py](../benchmarks/emergent_internal_quantum_numbers.py)).

**Layer 3 · form — a selected three-dimensional symmetry model.**
The standard representation of `S₄` is three-dimensional, and a selected
`C₃` action has cube-root eigenvalues at 120° on a circle. This is an exact
representation-theory statement. Its resemblance to a phase
parameterization of the empirical Koide relation is an analogy; it does not
derive three physical generations or lepton masses. The nested-graph
benchmark repeats the selected triplet motif across its construction, which
is a property of that construction rather than evidence for physical family
bands.
([emergent_generation_phase_circle.py](../benchmarks/emergent_generation_phase_circle.py),
[emergent_resonant_pattern_tower.py](../benchmarks/emergent_resonant_pattern_tower.py)).

**Layer 3 · scale — no universal `νf` has been derived.**
The nodal equation specifies `dEPI/dt = νf·ΔNFR`; it does not by itself give an
evolution equation forcing `νf` to one graph-independent value. Canonical
operators can raise, lower or silence structural capacity. Consequently,
`νf·sqrt(λ_n)` is a conditional modal scale for a declared state and graph,
not a universal mass scale or fixed-point theorem.

> **Honest boundary — the values are open.** What is **not** derived is the numerical **ratios**.
> The real lepton masses (`1 : 207 : 3477`) satisfy Koide with a *specific* circle: amplitude `√2`
> — equivalently the √-mass vector at **exactly 45°** to the democratic axis `(1,1,1)`, the maximal
> equal-split — and a phase `δ`. **Tested:** *no* natural TNFR breaking of the tetrahedron selects
> the 45°/`√2` condition; every natural breaking sits near the democratic axis (`Q ≈ 1/3`,
> near-degenerate), the **opposite** extreme from the leptons' 45° (`Q = 2/3`)
> ([emergent_generation_phase_circle.py](../benchmarks/emergent_generation_phase_circle.py) M4).
> The mass-to-eigenvalue identification is not fixed by TNFR, so a free choice
> can fit many numerical patterns. Claiming that measured masses emerge would
> therefore be unsupported. This graph construction does not derive Koide's
> empirical `2/3` relation or the observed fermion mass hierarchy; both remain
> outside the established TNFR results.

**Synthesis.** The selected graph models have exact winding, mode and symmetry
properties. Their interpretation as particle charge, generations or mass is
an analogy and no universal `νf` scale is derived. The numerical values and the
physical map remain open; the associated benchmarks are exploratory model
studies rather than evidence for a particle theory.

### 9.2 Quantum mechanics — **DERIVED (the boundary) / OPEN CONJECTURE (the reach)**

*Conjecture.* Quantum phenomena would emerge from the substrate. *Status:* the emergent geometry is
**classical** (§5.3) — a symplectic flow with classical, un-entangled Stokes/Poincaré
polarization. This boundary is now a **measured, decisive** one, not just an assertion: a local,
realistic hidden-variable model built directly on the substrate's own polarization angle (the
equatorial circle of the Poincaré sphere, §5.3) gives a CHSH value `|S| = 2.0000` (2×10⁶-sample
measurement) — *exactly* the classical (Bell) bound, not the quantum (Tsirelson) bound
`2√2 ≈ 2.828`
([emergent_bell_inequality_bound.py](../benchmarks/emergent_bell_inequality_bound.py)). By **Bell's
theorem** (1964; confirmed by loophole-free experiments), this is not a gap further work could
close: *any* substrate that is both **local** (§5.1's finite causal cone) and **classical** (§5.3's
un-entangled polarization) is bounded this way — TNFR's own already-derived properties fix the
boundary. A complex Hilbert space, the Born rule, genuine (Bell-violating) entanglement, and a
physical `ħ` (a dimensionful empirical constant with no `Hz_str`-native partner to calibrate against
without importing one, §1) are the reach beyond that boundary — ingredients TNFR does not currently
have.

### 9.3 A descriptive note — **ANALOGY**

TNFR's slogan — "coherent patterns maintained by resonance, dissolving when coupling fails" —
*describes* vortices, neural assemblies, and convection cells with one vocabulary, but does **not**
derive their continuum dynamics (e.g. the 3D Navier–Stokes closure — Clay — stays **open**; the
program reads it as the nonlinear `K_φ` cascade whose uniform-in-`Re` moment-ladder
closure is the wall, localised — not closed — by the emergent-geometry coherence attractor,
[TNFR_NAVIER_STOKES_RESEARCH_NOTES.md](TNFR_NAVIER_STOKES_RESEARCH_NOTES.md)). A descriptive
unification, not a derived identity.

### 9.4 What would move the frontier to derivation

- **§9.1 (values):** a *structural* principle that fixes the generation **ratios** — concretely, one
  that selects the Koide `√2` / 45° (maximal equal-split) circle and the phase `δ`, which no natural
  tetrahedron breaking reaches (§9.1). The **forms** are in hand; the **numbers** are not.
- **§9.2:** genuine entanglement (Bell-violating correlations, `|S| > 2`) — measured to sit at the
  classical bound (`|S| = 2.0000`, §9.2) while the substrate stays local (§5.1) and classical
  (§5.3); moving this would mean abandoning one of those two already-derived properties.
- **§7.2:** an independent edge connection with non-zero curvature and dynamics
  derived from the nodal equation. The implemented connection is pure gauge, and
  the separate point-vortex comparison does not provide a photon or a TNFR
  electromagnetic field.
- **All:** a *predictive* test the standard framework does **not** make (the bar the empirical
  record repeatedly set, and did not clear).

---

## 10. Bottom line

From the nodal equation, its exact fixed-graph diffusion restriction, and
separately declared auxiliary models, this document organizes a catalogue of
exact identities, finite measurements, analogies and open conjectures:

- a **structural-history analogy** (§2.5) — a genesis from
  the vacuum, an emergent time with an arrow, structure formation by coarsening, a growing causal
  horizon, and a regime-dependent fate — the **temporal spine** the rest of the towers read;
- several **geometric readouts** (a metric, distinct dimension notions, the tetrad);
- a **diffusive (thermodynamic) face** — heat, an emergent clock (time), coherence, an arrow of
  time, conservation (all on **one relaxation clock**), thermal fluctuations (Einstein /
  fluctuation-dissipation), and Ohmic transport (conductivity);
- a **conservative (relativistic) face** — a causal light cone, an approximate low-energy Lorentz
  invariance, an auxiliary symplectic model, and optics (refraction, the refractive index,
  Snell's law);
- **synchronization** diagnostics and an auxiliary Landau model with a
  continuous transition; mapping that model to U2 operator dynamics remains
  open;
- a **matter-comparison sector** — a discrete mode lattice and integer winding,
  alongside separately declared point-vortex, pure-gauge U(1), and tight-binding
  constructions; their physical identification and dynamical coupling are open;
- a **particle-like graph-model arc** (§9.1) with exact properties inside
  declared constructions, while physical identification, masses and a
  universal frequency scale remain open;
- an **information layer** — the grammar as a regular formal language with a finite Shannon channel
  capacity.

This is a proposed **structural organization** of related mathematical forms,
not evidence that all listed phenomena are one mechanism. It is not, on
current evidence, a derivation of the Standard Model, quantum mechanics,
relativity or cosmology, nor a source of predictions standard physics does not
already make. Those claims remain outside the established scope.

---

**Status**: WORKING DRAFT — EXPLORATORY. Promote an entry to a canonical theory note only after its
derivation is complete and status-checked.
