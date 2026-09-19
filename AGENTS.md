# TNFR: Resonant Fractal Nature Theory

**Working reference for the TNFR Python Engine, version 0.0.3.6.**
This synthesis states current definitions, contracts and mathematical boundaries.
Guide responsibilities belong to the [documentation map](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md).
Detailed derivations belong to the [theory index](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md);
execution details belong to [API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md).
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

### Communication policy

- Use English in code, documentation, comments, commits, issues and PRs;
  preserve verbatim quotations and raw data in their original language.
  The educational manuscript in `manual/` retains Spanish by explicit user
  instruction; its technical claims follow the same current scope.
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
The optional Gamma registry supplies an additional rate in the extended row
`dx/dt=nu_f*p+Gamma`; nonzero Gamma can move EPI at zero capacity. It is declared
forcing, not a derivation from the nodal product. Unforced certificates require
Gamma to be absent/zero; see [integrators](https://github.com/fermga/TNFR-Python-Engine/blob/main/src/tnfr/dynamics/integrators.py).

### The fractal-resonant node (NFR)

An NFR is modeled as a region of structural coherence coupled to a network,
carrying the triad. Nesting supplies operational fractality; persistence,
autonomous formation and physical identification require separate evidence.
Emission acts on an existing node with supplied basal capacity. THOL can create
children under configured preconditions and construction rules. Neither fact
proves spontaneous creation of the substrate or an autonomous maintained NFR.

The radial/annular/multinodal classifier is a calibrated read-out of potential
geometry. It does not derive a universal geometry or select Platonic solids.
`Network.nfr()` exposes this read-out. `structural_coherence` and
`is_structural_equilibrium` in `metrics/common.py` are shared diagnostic kernels;
reusing them across arithmetic, graph and chemical models does not identify
those models' dynamics. See [foundations](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md).

### Accumulated evolution and the convergence policy

On continuous intervals, `x(t)-x(t0)=integral(nu_f*p dt)`. Hybrid execution also
adds actual operator jumps. Finite-horizon integrability, boundedness, convergence
and absolute integrability are different conditions. Absolute integrability is
sufficient for a finite limiting state; bounded or vanishing pressure alone is
not. U2 is a stabilization/debt policy, not a universal convergence theorem.

### Transport content (structural diffusion)

The implemented pressure combines configured phase, form, capacity and topology
channels. Only the isolated EPI channel is exactly
`p_epi=-L_rw*x`, `L_rw=I-D^(-1)W`, with zero rows at isolates.
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
Lowercase English names are execution tokens, title-case names are display/class
names, and glyphs are internal symbols.

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

## 10. Development workflow

Inspect repository instructions, local changes and relevant source/tests before
editing. Preserve unrelated work. Keep definitions centralized and APIs stable
or explicitly migrated. Documentation should identify current owners rather
than duplicate derivations and delivery histories.

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
models and Millennium notes retain their own scope and do not constitute
additional active campaigns. Historical instructions are preserved in the
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
| `dynamics/` | Pressure realization, phase/capacity evolution and shared integration |
| `operators/` | Named transformations, grammar, stages and causal execution |
| `physics/` | Fields, model theorems, reductions and scope-aware observations |
| `metrics/` | Shared coherence, sense and diagnostic kernels |
| `config/`, `constants/` | Declared defaults, policy calibration and classifications |
| `mathematics/` | Numerical backends, exact helpers and arithmetic constructions |
| `sdk/` | Public network interfaces |
| `research/` and domain packages | Scoped experiments and analysis |

See [ARCHITECTURE.md](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md),
[docs](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md) and
[examples](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md).
Historical examples can illustrate an auxiliary model without proving physical
emergence; consult the current theoretical owner before reusing a conclusion.

## 14. Philosophy & excellence standards

Use the nodal framework to formulate testable questions. Prefer explicit
assumptions, reusable mathematics, negative controls and traceable observations
over claims of universality. Operational terminology, configured engine behavior,
conditional mathematical results and empirical validation must remain distinct.
