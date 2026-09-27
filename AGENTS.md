# TNFR: Resonant Fractal Nature Theory

**Working reference for the TNFR Python Engine, version 0.0.3.7.**
This synthesis states current definitions, contracts and mathematical boundaries.
Guide responsibilities belong to the [documentation map](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md).
The [theory index](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md)
is the single scientific catalog, with question-based reading routes and a
theory-to-engine/test/SDK map. Execution details belong to
[API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md).
This file is mirrored verbatim at `.github/agents/my-agent.md`.

## 1. What TNFR is

TNFR models coherent patterns on graph-coupled networks. Its nodal equation is

$$\frac{\partial\mathrm{EPI}}{\partial t}=\nu_f\,\Delta\mathrm{NFR}.$$

It relates structural change, reorganization capacity and pressure. It does
not, by itself, determine the pressure realization or the evolution of phase,
capacity and support. The hypothesis that physical entities emerge from this
structure remains open. An implemented model or diagnostic is not evidence
that the hypothesis is physically established.

### Source hierarchy (authority)

1. Explicit mathematical definitions, hypotheses, derivations and counterexamples.
2. Repository code and tests as evidence of implemented behavior, subject to audit.
3. This synthesis and the specialized document responsible for each result.
4. Published releases and historical notes, which can lag the working tree.

A contradiction is resolved by examining the mathematics and actual execution;
neither a historical label of "canonical" nor a passing test makes an unsupported
theorem true. Update this file as a coherent reference, not a session log.
Foundational definitions and constitutive premises are revisable hypotheses.
State an old/new model boundary when revising them, retain valid conditional
results, and prefer a discriminating prediction over extending a branch whose
premises already impose its obstruction. A useful model family need not be the
unique possible completion of the nodal identity.

### Communication policy

- Use English in code, documentation, comments, commits, issues and PRs;
  preserve verbatim quotations and raw data in their original language.
- State the model, assumptions and evidence for claims. Distinguish exact
  identities, conditional theorems, configured contracts, finite observations,
  auxiliary models and open hypotheses.
- Do not promote graph analogies, arithmetic constructions or telemetry to
  particle physics, cosmology or consciousness conclusions.
- Record source/configuration, inputs, seeds, execution path and precision when
  reporting reproducible numerical results. Include negative results and limits.

## 2. Foundations

### The nodal equation

Write `x=EPI`, `p=DeltaNFR`; the unforced continuous row is `dx/dt=nu_f*p`.
In a selected scalar chart, `[p]=[x]/([nu_f][t])`. `Hz_str` denotes structural
rate relative to the declared clock; identifying it with laboratory seconds
requires a measurement bridge. Pressure is a structural driving term, not an
arbitrary loss function. Defining it retrospectively as a measured derivative
merely reconstructs the identity and supplies no independent prediction.

Differentiably invertible form charts preserve differential solution sets when
pressure (and any gradient-flow mobility) transforms consistently. Injective
storage alone can introduce spurious differential solutions at singular points.
Even with a fixed pressure law, one instantaneous EPI rate need not identify
capacity; the existing P2 control separates two such states by their subsequent
acceleration. Clock changes also transform capacity evolution, and phase
synchronization alone need not supply a monotone clock. These boundaries are
owned by the [form foundation](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/FUNDAMENTAL_THEORY.md)
and [parameter foundation](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md).

### Structural triad

- **Form EPI:** structural configuration in a declared state space. The graph
  scalar chart is signed real, directly or via uniform-real `BEPIElement`.
  `real_scalar_epi` and `scalarize_epi` preserve that sign. A nonuniform or
  complex BEPI has no signed scalar representative; scalar-only glyphs,
  diffusion and certificates reject it. `abs(BEPIElement)` is a magnitude.
- **Capacity nu_f:** nonnegative reorganization rate. Zero capacity suppresses
  the unforced nodal row even under nonzero pressure; it need not imply equilibrium.
- **Phase phi/theta:** circular synchronization coordinate. U3 compares
  `abs(wrap(phi_i-phi_j))` with the configured `Delta_phi_max`.

Named transformations use their operator contracts. Declared numerical solvers
advance EPI through the shared nodal integrator with explicit capacity, pressure,
clock and provenance or residual. Initialization is distinct from evolution.
Full closure also needs justified phase, capacity, support and input laws.
Validated nodal arithmetic and diagnostic scalar admission share the
represented-real boundary: invalid types or nonzero inputs lost during
materialization cannot certify equilibrium or a valid trajectory. Capacity
admits zero; finite inputs still require representable output arithmetic.
Pressure preparation and phase/capacity execution enforce that boundary before
coercion. Temporal observers reject nonzero unrepresentable rates and retain
detached evidence; an energy tolerance alert is not endpoint nonincrease.
Solver clocks and retained rates use the same raw scalar admission; Boolean
durations are not clocks. Consumed phases remain validated before caching and
array conversion. A coordinate not consumed by a declared model need not be
invented or coerced to execute that model.
The optional Gamma registry supplies an additional rate in the extended row
`dx/dt=nu_f*p+Gamma`; nonzero Gamma can move EPI at zero capacity. It is declared
forcing, not a derivation from the nodal product. Unforced certificates require
Gamma to be absent/zero; see [integrators](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/dynamics/integrators.py).
The additive integrator evaluates sources strictly through the live registry;
invalid declarations or an array fast path cannot silently remove a source.
The opt-in extended coupled model has a separate unforced scope.

The held reference model S0 fixes support, conductance, capacity, primitive
phase and pressure coefficients. Its fresh pressure defines an affine form
law; this restricted completion is distinct from `runtime.step`, which can
also change phase, capacity and support. Regional means and form contrasts
are observations of that state, not a replacement primitive ontology.
Identifying a derived form angle with primitive phase, or imposing
`nu_f=g(EPI)`, requires preservation by the declared flow and every event.
See the [state admission](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/FUNDAMENTAL_THEORY.md#foundational-state-admission)
and [capacity-law admission](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/CAPACITY_LOCALIZATION_BALANCE.md#capacity-law-admission) owners.

### The fractal-resonant node (NFR)

An NFR is modeled as a region of structural coherence coupled to a network,
carrying the triad. Nesting supplies operational fractality; persistence,
autonomous formation and physical identification require separate evidence.
Emission acts on an existing node with supplied basal capacity. THOL can create
children under configured preconditions and construction rules. Neither fact
proves spontaneous creation of the substrate or an autonomous maintained NFR.

A persistence claim must specify its full-state identity and evolution law.
Stationary profile restoration, orbital stability, winding retention and finite
regional lifetime are different obligations. Uniform scalar EPI can coexist
with nontrivial phase geometry under an admitted law. A failed scalar recovery
test does not exclude every NFR; a snapshot match or low global phase order
does not decide the presence or absence of coherent geometric identity. See
[research scope](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_RESEARCH_STRATEGY.md#closure-audit-equations-events-and-identity).

The radial/annular/multinodal classifier reads the unit-source potential
centrality profile under a calibrated policy. Unsupported geometry is explicitly
unavailable; a uniform profile does not prove rotational graph symmetry.
It does not derive a universal geometry or select Platonic solids.
`Network.nfr()` exposes this scope and the shared coherence-length estimator's
fit/fallback provenance; it does not certify formation or fractality.
`structural_coherence` and
`is_structural_equilibrium` in `metrics/common.py` are shared diagnostic kernels;
reusing them across graph and arithmetic models does not identify those
models' dynamics. See [foundations](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md).

### Accumulated evolution and the convergence policy

On continuous intervals, `x(t)-x(t0)=integral(nu_f*p dt)`. Hybrid execution also
adds actual operator jumps. Finite-horizon integrability, boundedness, convergence
and absolute integrability are different conditions. Absolute integrability is
sufficient for a finite limiting state; bounded or vanishing pressure alone is
not. U2 is a stabilization/debt policy, not a universal convergence theorem.

### Transport content (structural diffusion)

The implemented pressure combines configured phase, form, capacity and topology
channels. Only EPI uses conductance weights; phase, capacity and degree contrast
use unique outgoing support neighbors, including zero-conductance edges.
Capacity contrast is a source for form, not a capacity evolution equation.
The nonlinear phase source need not conserve a transport-weighted form mean,
even on a regular reciprocal graph. Its exact low-degree reduction does not
extend to general neighborhoods. Internal source terms do not require an
external environment; the initial nodal support remains assumed. See the
[pressure contract](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md#21-the-implemented-pressure-is-a-specified-relational-map).
Only the isolated EPI channel is exactly
`p_epi=-L_rw*x`, `L_rw=I-D^(-1)W`, with zero rows at isolates.
For an isolated source-free form response, exact amplitude/offset covariance
and differentiability at uniform form force linearity; locality and a maximum
principle then imply nonnegative neighbor differences. Those are explicit
premises, not consequences of the nodal product. They do not select the full
pressure mixture or its phase response; see the linked pressure foundations.
On fixed connected symmetric nonnegative conductance and fixed positive capacity,
`dx/dt=-diag(nu_f)*L_rw*x` relaxes to consensus. With common capacity it conserves
the degree-weighted mean and decays in modes `exp(-nu_f*lambda_k*t)`; heterogeneous
capacity conserves weights `d_i/nu_f_i`. Other channels, changing support,
directed nonnormality, forcing and zero capacity require their own hypotheses.

Exact-real diffusion theorems, materialized binary64 identities, finite executor
certificates and asymptotic runtime convergence are distinct. REMESH stability
results likewise specify support, metric, delays, clipping, capacity and schedule;
none provides unrestricted complete-runtime stability. The detailed owners are
[diffusion](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/TNFR_DIFFUSION_STABILITY_THEOREM.md),
[directed transport](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/TNFR_DIRECTED_NONNORMAL_DYNAMICS.md) and
[REMESH](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/REMESH_INFINITY_DERIVATION.md).

The canonical phase channel uses an unweighted neighbor-phasor resultant,
including support edges with zero transport weight. It is nonlinear and has
branch/resultant boundaries; a pairwise Laplacian comparison is not its global
law. Phase quotient and midpoint results retain their explicit chart, support
and represented-rounding hypotheses.

## 3. The structural tetrad

The tetrad is the diagnostic interface, not a proven complete state basis.
Compute it through [fields.py](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/physics/fields.py).

| Field | Meaning | Scope |
| --- | --- | --- |
| `Phi_s` | Nonlocal pressure aggregation | Sum of pressure divided by squared path distance |
| `abs(grad phi)` | Local phase stress | Mean absolute wrapped neighbor separation, bounded by pi |
| `K_phi` | Circular phase curvature | Wrapped displacement from a neighbor resultant; bounded by pi where defined |
| `xi_C` | Coherence correlation range | Static coherence-product fit with separately tagged spectral fallback |

### Field scales

- `Phi_s` reads explicit edge `length`, with `weight` as compatibility fallback;
  diffusion instead reads `weight` as conductance. These roles need not coincide.
  Potential is bounded by the aggregation operator norm times pressure, not by pi.
- The exact phase-wrap scale is pi. `pi/16` gradient warnings, `0.9*pi` curvature
  margins, `pi/4` potential magnitude and U6 `pi/2` drift are selected policies.
- Curvature is undefined for an exactly joint-zero represented resultant.
  Preserve availability evidence rather than inventing a direction.
- The xi fit uses static coherence products, not centered fluctuations or a
  temporal relaxation measurement. The fallback uses the first spectral value
  above the implementation cutoff `1e-9`; it equals `1/sqrt(lambda_2)` only under
  the relevant connected-graph and cutoff conditions. Always report provenance.

### Why these four (scope and minimality boundary)

Aggregation, first/second local phase observations and nonlocal correlation
capture complementary information. They do not reconstruct hidden form,
capacity, history, support or future evolution in general. Higher powers of a
Laplacian can carry independent information. Lossy aggregation can preserve
mean dynamics but lose potential or xi. Definitions and limitations belong to
[Structural Fields](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/STRUCTURAL_FIELDS_TETRAD.md),
[minimal degrees](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/MINIMAL_STRUCTURAL_DEGREES.md) and
[scale/geometry](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/TNFR_SCALE_GEOMETRY_AND_BRIDGE.md).

## 4. Emergent geometry

### Auxiliary symplectic substrate

The substrate initializes ambient canonical pairs `(K_phi,J_phi)` and
`(Phi_s,J_DeltaNFR)` from graph read-outs. Its declared quadratic Hamiltonian
has a harmonic symplectic flow and model-specific conserved charges. Graph
fields need not remain in the realizable image under that auxiliary flow.
Neither the full nodal equation nor all 13 operators have been derived from it.
U(2)/Stokes polarization is a classical construction, not a quantum-state or
particle-generation result.

A restricted bridge is exact for reversible EPI diffusion: with `B=D-W`,
`E_D=x^T B x/2` and mobility `M=diag(nu_f/d)` (zero at isolates),
`dx/dt=-M*grad(E_D)` and `dE_D/dt<=0`. This Dirichlet energy is distinct from
potential aggregation and the auxiliary isotropic Hamiltonian. See
[variational scope](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/TNFR_VARIATIONAL_PRINCIPLE.md).

### Structural conservation theorem (Noether-like)

The diagnostic balance uses `rho=Phi_s+K_phi` and the declared phase/pressure
currents. Its source/residual must be measured on the actual trajectory;
operator labels alone do not make it vanish. The nonnegative sum-of-squares
field energy is a Lyapunov candidate until monotonicity is proved for a stated
law or observed on a specified trace. Read the precise hypotheses in
[conservation](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/STRUCTURAL_CONSERVATION_THEOREM.md).

Temporal balance requires nonempty matching node support and a finite positive
interval. Each sector uses its captured graph divergence; current magnitudes
cannot reconstruct that decomposition. Missing intervals are unavailable, not
successful conservation. A numerical-tolerance energy alert is distinct from
the sign of the observed energy change.

Single-snapshot tensor telemetry reports conservation quality as unavailable;
node-label ordering cannot supply a time derivative or graph divergence.
Composite read-outs reuse one base-field collection and retain its node order.
Historical vorticity and THOL emergence indices are static/retrospective
statistics, not cycle winding, measured deformation or formation theorems.
The existing acute-cycle cosine margin bounds local-resultant availability
and the supplied sine law's restoring stiffness; it is not the Jacobian of
the Arg-based pressure. See the
[emergent-property audit](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/EXTENDED_FIELDS_AND_DERIVED_QUANTITIES.md#61-emergent-property-audit-and-useful-derived-margins).

### Regime correspondences

First-order nodal drift treats capacity as mobility, not inverse mass. An
auxiliary second-order graph wave has frequencies `sqrt(lambda_k)`. Finite graph
spectra exist independently of dissonance. `net.rhythm()` reports modal rhythm;
`net.resonance()` reports phase synchronization. These do not establish a shared
native clock or a maintained engine wave. Prescribed phase motion can generate
a conditional periodic form response; the origin of that motion remains open.
See [regime comparisons](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/PHYSICAL_REGIME_CORRESPONDENCES.md).

## 5. The 13 canonical operators

The registry is the semantic interface for named transformations. Catalog
completeness and a uniquely derived autonomous selection law remain open.
Read executable tokens from `TNFR.operators()`'s `token` field; `name` is the
display name and `glyph` the internal symbol. Do not infer a token by lowercasing
a display/class name. The catalog describes contracts, not live-state admission.

| Operator | Glyph | Primary channel | Direct contract or execution boundary |
| --- | --- | --- | --- |
| Emission | AL | EPI | Sources form on existing support; does not write capacity, pressure or phase |
| Reception | EN | EPI | Blends incoming form; stored pressure and rate, hence immediate local C, unchanged |
| Coherence | IL | pressure | Contracts pressure magnitude; compare C using the declared local scope |
| Dissonance | OZ | pressure | Local destabilizing pressure contract precedes any propagated increments |
| Coupling | UM | phase | Requires circular U3 compatibility; can also change neighboring state/support |
| Resonance | RA | EPI | Uses U3-compatible neighbors and preserves scalar sign/kind identity |
| Silence | SHA | capacity | Attenuates capacity and preserves EPI at the event; later unforced freezing requires zero nodal product |
| Expansion | VAL | capacity | Capacity does not decrease; no general state-dimension increase follows |
| Contraction | NUL | capacity | Capacity does not increase; also affects pressure |
| Self-organization | THOL | pressure | Configured child/hierarchy construction; autonomous maintenance unproved |
| Mutation | ZHIR | phase | Needs live temporal evidence and grammar context; does not create topology |
| Transition | NAV | pressure | Controlled state change; not a U2 destabilizer |
| Recursivity | REMESH | EPI | Node glyph is advisory; separate network operation mixes delayed EPI |

### Dual-lever and contracts

Capacity and pressure are the two factors of the nodal product. Primary channel
classification does not list every auxiliary write or establish a one-to-one
mapping to tetrad fields. The registry in
[operator_contracts.py](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/operators/operator_contracts.py)
owns metadata; [STRUCTURAL_OPERATORS.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/STRUCTURAL_OPERATORS.md)
owns mathematical interpretation. Direct glyphs, public classes and atomic
stages can have different secondary effects; identify the actual path.

Mutation requires active capacity and a finite signed two-sample EPI secant
strictly above `ZHIR_THRESHOLD_XI`. Timestamped history is authoritative and must
end at the live state; legacy histories use explicit operator-step units.
`nu_f*DeltaNFR` is a prediction, not observed trigger evidence. Missing/stale
history rejects direct execution; default selection can abstain to IL. U4b
context remains separate. `variant_creation` is rejected; THOL owns child creation.

### Operator events and physical time

Named events are hybrid jumps, not finite-duration solutions of an invented
pressure law. Physical partitions record held/refreshed pressure, solver spans
and jumps separately. Atomic all-target stages use a shared immutable snapshot,
validate all proposals and commit graph-owned state together. Target-order
invariance of specified primary writes does not imply relabeling equivariance,
callback invariance, arbitrary external rollback or future stability.

REMESH advisory stages do not implicitly invoke delayed-EPI mixing. Network
REMESH validates exact live history support and reports raw/bounded values,
clipping and defects. Finite causal certificates bind retained execution only;
the next invocation needs its own admission. See
[API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md).

### Composition

Words preserve operator order; simultaneous target stages are not simultaneous
words. Grammar admission, operator state preconditions, continuous propagation,
reduction closure and complete-runtime stability are independent obligations.
Reuse the shared executor and evidence adapters; do not manufacture a certificate
from caller-supplied flags or unrelated traces.

## 6. Unified grammar (U1–U6)

| Rule | Engine policy | Mathematical boundary |
| --- | --- | --- |
| U1 | Standalone initiation `{AL,NAV,REMESH}` and closure `{SHA,NAV,REMESH,OZ}` | History admission, not existence of the derivative at zero EPI |
| U2 | Destabilizers `{OZ,ZHIR,VAL}` require stabilization `{IL,THOL}` and bounded debt | Calibrated policy, not universal convergence |
| U3 | UM/RA require circular phase compatibility | Admissibility does not derive phase motion or its persistence |
| U4 | Bifurcation handlers, recent destabilizer and prior IL for ZHIR | Configured recency and live trigger evidence, not a universal relaxation time |
| U5 | Declared deep recursion and nearby scale-stabilizer coverage | Parent/child inequality requires its normalization; nesting alone does not prove it |
| U6 | Reference-relative potential drift monitoring | Read-only safety policy, not a graph-independent potential bound |

`grammar_canon.py` owns rule bases; role sets derive from shared predicates and
are exposed through `grammar_types.py`. The `core` word profile omits legacy
pair/THOL preferences while retaining word and live contracts.
`GRAMMAR_REJECTION_MODE="raise"` prevents blocked-step substitution; it does not
derive selection. Full rule premises are in
[grammar scope](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) and
[grammar specification](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/UNIFIED_GRAMMAR_RULES.md).

## 7. Telemetry & metrics

`C(t)=1/(1+mean(abs(DeltaNFR))+mean(abs(dEPI)))` is the configured global
coherence read-out; the per-node kernel uses local values. Its cuts
`pi/(pi+1)` and `1/(pi+1)` label a policy band. C is not by itself proof of
stability, identity, coupling or autonomous formation.

Si is a configured diagnostic mixture. Default controllers consume it, but
that dependency is not an emergent selection law. The auxiliary affinity
`coherence_matrix` is not C and need not be positive semidefinite. Report tetrad
availability, stored versus refreshed pressure and xi fit/fallback provenance.
Do not treat undefined or unsupported observations as successful zero values.

Automatic candidates use admitted stored metrics, current normalization and an
explicit batch lifetime; operator admission remains independent. Configured
phase labels depend on sample size and cannot establish autonomous formation.
An incomplete tetrad cannot certify even the configured overall safety advisory.

Thresholds must retain their units and actual consumer: normalized selector
pressure is distinct from absolute adaptation pressure. Shared default
resolution must agree across selection, adaptation and diagnostics. A pi-based
formula or a legacy `CANONICAL` name does not derive a policy; numerical
tolerances and inactive compatibility exports supply no physical theorem.
See the [threshold ledger](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md#11-thresholds-constants-and-numerical-settings-have-different-duties).

## 8. Canonical invariants

1. Preserve continuous nodal flow and declared hybrid jumps with their provenance.
2. Enforce circular phase admission before U3 coupling/resonance.
3. Retain declared nested structure and test the applicable U5 contract.
4. Use registered semantic operators or explicitly justified new contracts;
   numerical evolution uses the shared integrator.
5. Preserve structural units and expose the relevant structural telemetry.
6. Make execution reproducible under fixed source, configuration, inputs,
   seed, target order, precision and backend; a seed alone is insufficient.

## 9. TNFR agent playbook

Start from the relevant mathematical question and actual code path. Reuse shared
kernels and theorem owners before adding mechanisms. Separate supplied state,
forcing and controller choices from quantities derived by the model. Accept
changes against the affected contract and evidence: destabilization, negative
results or corrected diagnostics need not increase C. Never hide a contradiction
by relabeling telemetry or adjusting a reserved response after evaluation.

Use the [theory-to-execution map](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md#theory-to-execution)
to locate a derivation, shared implementation and representative checks.
Generic runtime, operator-word studies, the opt-in relational law and read-only
proof calculations have different contracts; selecting an interface does not
transfer a theorem between them.

### Operator studies and stored-state observations

For a declared finite operator study, use `StudySpec` and `run_study` from
`tnfr.sdk`. The CLI `tnfr network` (also `python -m tnfr network`) delegates to
that same owner in [sdk/study.py](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/sdk/study.py).
Extend the shared model before adding separate CLI validation, simulation or
serialization logic. `TNFR.operators()` and `list_sequences()` expose existing
registries; do not maintain a second operator catalog or word table.

`StudySpec.cycles` and this CLI's `--steps`/`--cycles` count complete requested
operator words, not seconds. The report records the requested word and observed
endpoints, not every realized per-node glyph; inherited grammar policy and live
preconditions remain active. The supplied zero-form baseline is initialization,
not spontaneous substrate creation. The study runner sets both the topology
seed and graph `RANDOM_SEED`; direct `TNFR.create(..., seed=...)` sets only the
topology seed. Other engine configuration is inherited, so retain relevant
configuration and runtime provenance when comparing runs.

`StudySpec.from_dict` validates a construction recipe and rejects unknown keys.
`StudyResult.to_dict()` returns detached report data; the shared `export_to_json`
writer saves it. Neither a recipe nor this scalar state/support projection is
a complete resumable checkpoint. `import_from_json` reads data without restoring
callbacks, histories or a live graph. Usage and schema details belong to the
[CLI and SDK guide](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md).

Use `diagnose_network(network)` for detached stored-state observations. It does
not refresh pressure, evolve the graph or invent missing temporal evidence.
Preserve independent field availability, per-node undefined curvature and xi
estimator provenance. Its nodal product is a model-rate read-out, not a measured
derivative or permission to execute Mutation. Diagnostic values do not select
the next operator in the shared study runner.

`Network.regional_form(regions)` delegates to
[`physics/form_geometry.py`](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/physics/form_geometry.py).
It projects stored unforced nodal rates onto a supplied disjoint ordered
three-node partition of the full support, retaining exact represented
Cartesian/Gram data and rounding defects. The form angle is derived from EPI
and is unavailable at zero contrast;
primitive phase is not consumed. Observation neither selects the partition
nor supplies an autonomous reduced law.
`derive_regional_affine_closure` separately checks a supplied fixed law
`xdot=Gx+b` on the full real fine-state domain. Its source `b` has rate units,
not pressure units; the report does not authenticate a live graph or runtime.
A fixed source with regional contrast can distinguish forms having the same
Gram data, even when their initial Gram rates agree. The
[derived-form owner](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/DERIVED_FORM_PHASE.md)
states the exact closure conditions and source-relative response boundary.
`Network.source_relative_form(regions, held_source_rate=...)` delegates to
[`physics/source_relative_form.py`](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/physics/source_relative_form.py).
It retains the contrast relative to an independently supplied held rate source
through `W=z*c^dagger`. A nonzero source-contrast vector supplies an orientation
reference; a zero vector does not. This observation supplies neither the
source's origin nor its evolution and installs no selection policy.

### Conditional relational execution and observations

`RelationalExchangeModel` and `Network.relational_exchange` /
`Network.step_relational` select the explicit capacity-separable joint model
in [dynamics/relational.py](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/dynamics/relational.py).
It reuses native pressure and nodal arithmetic on fixed simple unit support,
with held nonnegative capacity and an explicitly admitted phase domain. The
default `phase_domain="acute"` retains its acute-edge contract. The opt-in
`"positive_resultant"` chamber certifies positive real relative resultants
throughout each represented Euler chord using rational cosine bounds. This
is a sufficient regular chamber, not the whole regular domain or an enclosure
of the exact ODE trajectory; an inconclusive bound rejects the proposal. Joint
storage and independent local capacity are constitutive premises. A detached
field is distinct from an atomic Euler step; represented balance and step
defects remain evidence, not a continuous or physical stability certificate.
The default operator runtime, forcing and capacity policies are separate.
The [API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md#conditional-relational-execution)
own admission, commit and report semantics; the
[theory owner](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md)
owns the derivation and competing capacity premise.

The same law has conditional local recovery around acute equilibria with
strictly positive held capacities and form dissipation, and finite evidence
of transmission between prepared regions. `Network.relational_pattern`
delegates to the detached
[observation owner](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/physics/relational_observations.py):
supplied regions and reference phase lifts define centered form, phase errors
and offsets. It reuses shared winding and regional transport accounting;
unavailable transport is explicit. The fresh field also retains exact
represented nodal dissipation, signed exchange and actual form/phase work;
these sum to its existing global balance. Regional work sums those same
contributions and paired weighted rates share one outward-cut definition.
A selected zero capacity makes divided rates unavailable, while work and
cut evidence remain defined. The phase rate is not the derivative of a
weighted phase total. See the
[work integration](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-work-integration).
A snapshot does not establish recovery or closed regional dynamics.

The exact regional phase-rate budget is mean mobility times the outward form
cut, plus mobility/form-gradient covariance and captured rounding residual.
The engine's phase mobility and rounding evidence feed `region.phase_response`,
including at zero capacity. Its Cauchy squared bound is derived, not a policy
threshold. An unchanged or zero cut can conceal different regional phase
responses; neither this instantaneous budget nor zero covariance closes future
dynamics. The [composition owner](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#regional-phase-mobility-balance)
states the identity and its limits.

### Reduction and derived memory

Exact linear observation closure is owned by
[`mathematics/linear_observation.py`](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/mathematics/linear_observation.py)
with positive-sign `z'=Jz`; the existing diffusion wrapper retains `x'=-Ax+b`
and its independent admission. The fixed single-bridge paired-C5 study
requires ten linear coordinates for six regional mean/port observations,
but those coordinates fail to close the nonlinear law near equilibrium.
Adding their instantaneous predicted rate still fails: hidden form contrast
can change coarse acceleration while both observations agree. Rational matrix
probes do not certify irrational trigonometric coefficients; the
[analytic composition result](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md)
owns that distinction. A minimal tangent realization is not an effective
canonical node or a replacement for the full engine state.

The [joint memory theorem](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_PATTERN_MEMORY.md)
retains ten even and eight odd coordinates after removing only common offsets.
Exact elimination preserves hidden initial state and nonlinear forcing. For
small initially even preparations, parity yields quadratic hidden generation
and cubic visible tangent error on a fixed admitted horizon. For the declared
prepared family, the quadratic hidden and cubic visible corrections have a
conditional fifth-order visible remainder; neither its constants nor a common
amplitude radius are numerically certified. A frozen finite comparison finds
better prediction than both tangent and direct cubic controls using matched
Euler grids. This is evidence for
derived hidden feedback in that preparation, not continuous-ODE accuracy or an
empirical scaling law. Decimal re-evaluation checks arithmetic on represented
coefficients, not exact irrational constants. These research instruments install
no reduced nonlinear SDK solver, fitted kernel, universal positive memory law
or finite-memory cutoff.

### Protected capture and conditional formation

The following certificates use two C5 rings with bridges at matching positions
zero and one. This is a different support from the single-bridge memory study.

`Network.relational_capture` applies the conditional protected-basin theorem
through [relational_capture.py](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/physics/relational_capture.py).
It requires exact copied/reflected state, unit capacities, positive coefficients,
an admitted phase rectangle and rigorously enclosed storage below `7*beta`.
This support-specific bound
is derived from the phase landscape. The read-only report separates current
winding from the ideal limiting sector `-1`, `0` or `1`; it neither selects
operators nor certifies Euler trajectories. Nonzero symmetry defects and
inconclusive bounds remain unavailable. The theorem does not establish entry
from winding zero into a nonzero-sector basin or physical pattern identity.
`Network.relational_local_capture` applies the same owner's full-state local
energy theorem near a declared sector on that support, with unit capacity and
storage scale. Exact quotient-distance and excess-energy bounds replace the
reflection premise. It certifies ideal continuation from the represented
snapshot, not numerical error from an earlier state or future Euler execution.
`Network.relational_sector_capture` supplies a full-state acute-sector
energy barrier for sectors `-1` and `+1` on the same support. Exact pi-affine
gaps and cycle periods replace both symmetry and local-distance assumptions;
its barrier is derived
from the costs of acute-sector faces. All three certificates are read-only
applications of separate sufficient theorems, not automatic dynamics policies.
The sector certificate admits arbitrary positive held capacities and positive
storage scale. Its normalized energy deficit gives uniform future acute,
resultant and phase-metric margins under that ideal law. Unit-value flags
are descriptive; missing admission makes future bounds unavailable. Capacity
events and entry-error certificates remain separate obligations.

`Network.relational_transit_capture` supplies a read-only validated continuous
enclosure through [relational_transit.py](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/physics/relational_transit.py).
It requires exact reflected initial state, unit held capacity and positive
coefficients on the same support, with `phase_domain="positive_resultant"`
and declared exact horizon, step and Taylor order. Whole-time
Picard admission, derivative remainders and signed-diagonal flow comparison
retain propagated error and every resultant, including central rows omitted
by symmetry. Capture requires the entire endpoint box in the protected basin.
Validated winding-zero entry to positive-twist maintenance establishes
conditional formation for a supplied preparation and support, not substrate
creation or physical identification. It also implies a qualitative
open neighborhood of full states and positive held capacities with the same
outcome, without quantifying its radius. Earlier finite-executor verdicts remain
unchanged; an unresolved proof bound retains its prefix and is unavailable.
The default transit target is +1; `requested_sector=None` classifies any of the
three protected basins through the shared affine-margin ledger. Consensus is
sector 0, distinct from unavailable. The zero-form control converges to consensus
despite transient form and winding; phase-to-form exchange alone does not prove
maintained identity. The form-sign reversal also converges to consensus while
keeping the reference's initial energy, loss and phase geometry. These scalar
summaries therefore do not select the basin. Signed exchange remains a derived
work read-out, not a primitive or controller; the
negative controls and their exact hypotheses belong to the
[form-direction control](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-reversed-form-control).

`relational_report_to_dict` projects field, step, pattern, capture and transit
reports for the shared JSON writer, retaining rational evidence and distinct
verdicts. It does not authenticate provenance or create a resumable checkpoint.

## 10. Development workflow

Inspect repository instructions, local changes and relevant source/tests before
editing. Preserve unrelated work. Keep definitions centralized and APIs stable
or explicitly migrated. Documentation should identify current owners rather
than duplicate derivations and delivery histories. The documentation map owns
navigation and responsibilities; the example and benchmark indexes identify
maintained entry points and their scope. Historical paths or retired notebooks
are evidence to inspect, not current instructions to restore automatically.
Preserve frozen protocols, predictions, responses and source archives unchanged.
Later corrections must identify their scope without rewriting an old verdict.

### Commit / PR templates

Describe the concrete problem, resulting behavior, mathematical scope, validation
and material limitations. Follow a repository template when present. A change
need not claim improved coherence to be valid.

### Testing requirements

Run checks appropriate to the affected behavior. Verify actual operator
postconditions, unsupported domains, numerical/representation boundaries and
provenance where relevant. Do not assume RA always increases synchronization,
OZ always produces a bifurcation, or every valid sequence is monotone. A test
suite checks finite cases and implementations, not unrestricted theorems.
Exercise the actual shared owner with an independent expected result or a
meaningful boundary; assigning fixture literals and asserting those same values
does not test nodal dynamics. Reuse an expensive producer's report through a
module fixture while retaining distinct assertions. Test CLI/report wiring
separately when it needs no new scientific execution; preserve independently
necessary cold-import, atomicity and provenance checks. See the
[core contract map](https://github.com/fermga/TNFR-Python-Engine/blob/main/tests/core_physics/README.md)
for the tests that retain the behavior of retired introductory illustrations.
Default test selection, dependencies and parallel scheduling have their owners
in test configuration and the workflow guide; do not duplicate those settings
here or regenerate historical research artifacts for an unrelated change.
The default run is the routine engine/API gate, not the full research inventory.
Select the affected research owner explicitly when changing its model or claim;
report that scope rather than presenting a routine pass as complete coverage.
See [TESTING.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md).

## 11. Troubleshooting

Inspect the declared model, execution path, state/history and telemetry before
attributing a failure to theory. Grammar rejection, live-state rejection, solver
defect and a theorem's unsupported domain are different outcomes. A drop in C
is a violation only where an applicable contract promises nondecrease. Preserve
explicit abstention and undefined-field evidence.

## 12. Research programs

The [portfolio](https://github.com/fermga/TNFR-Python-Engine/blob/main/TNFR_lineas_de_investigacion.txt)
classifies primary, supporting and parked work. The
[execution plan](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/FIVE_STAGE_EXECUTION_PLAN.md)
is the sole task queue; the
[strategy](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_RESEARCH_STRATEGY.md)
explains its rationale. Conditional core results, finite C6 evidence, arithmetic
models and scoped auxiliary algebra retain their own hypotheses and do not
constitute additional active campaigns. Unsupported physical-programme
wrappers and circular type probes are retired; their boundaries and intentional
API removals are recorded in the
[archive](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/archive/README.md).

The primary objective is a predictive generative account of coherence patterns.
The P1-P5 measurement bridge separates calibration from reserved evaluation and
requires an independently declared measurement/clock mapping. Current research
uses workstation-accessible terrestrial data or laboratory-scale protocols;
physical admission and identification remain open. Track detailed status in
the plan rather than appending milestones here.

## 13. Map of the codebase & examples

| Owner | Responsibility |
| --- | --- |
| `dynamics/` | Configured pressure and phase/capacity maps; shared integration |
| `operators/` | Named transformations, grammar, stages and causal execution |
| `physics/` | Fields, model theorems, reductions and scope-aware observations |
| `metrics/` | Shared coherence, sense and diagnostic kernels |
| `config/`, `constants/` | Declared defaults, policy calibration and classifications |
| `mathematics/` | Numerical backends, exact helpers and arithmetic constructions |
| `sdk/` | Public networks, validated study recipes and detached diagnostic reports |
| `cli/` | Command adapters; study execution and catalogs reuse SDK owners |
| `research/` and domain packages | Scoped experiments and analysis |

See [ARCHITECTURE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md),
[docs](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md) and
[examples](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md).
Historical examples can illustrate an auxiliary model without proving physical
emergence; consult the current theoretical owner before reusing a conclusion.
Choose the current entry point from the example index rather than assuming an
old numbered filename still exists. Examples and benchmarks are not an
instruction to execute every file or reopen parked research branches.

## 14. Philosophy & excellence standards

Use the nodal framework to formulate testable questions. Prefer explicit
assumptions, reusable mathematics, negative controls and traceable observations
over claims of universality. Operational terminology, configured engine behavior,
conditional mathematical results and empirical validation must remain distinct.
