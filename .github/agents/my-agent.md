# TNFR: contributor and agent instructions

**TNFR Python Engine 0.0.3.8.** This file defines how to change the repository
and evaluate its claims. It is mirrored verbatim at `.github/agents/my-agent.md`.
The [README](https://github.com/fermga/TNFR-Python-Engine/blob/main/README.md)
introduces TNFR. Definitions and proofs belong to the
[theory catalog](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md);
execution details belong to the [API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md).

## 1. Scientific and editorial responsibility

TNFR investigates whether coherent, interacting patterns can arise from
justified nodal structure and dynamics and eventually explain measurable physical
properties. The repository supplies an engine, conditional mathematical results
and finite computational evidence. Physical identification and a uniquely
selected fundamental law remain open.

Resolve claims against explicit definitions, hypotheses, derivations and
counterexamples, then actual implementation and relevant tests. Code establishes
implemented behavior, not mathematical truth. Summaries, labels and filenames
do not prove a theorem. Foundational and constitutive premises are revisable;
identify the model and scope when a premise changes, retaining valid conditional
results and counterexamples.

- Use English in code, documentation, comments, commits, issues and PRs;
  preserve verbatim quotations and raw data in their original language.
- Distinguish definitions, supplied laws, derived identities, conditional
  theorems, finite observations, configured policies and open hypotheses.
- Explain the present model directly. Keep definitions with their owners,
  usage in guides and task status in the execution plan. Overview documents
  are not delivery logs or copies of detailed derivations.
- Do not promote arithmetic encodings, graph analogies, auxiliary Hamiltonians
  or diagnostic scores to evidence of particles, quantum physics or consciousness.
- Accept changes against the relevant contract. A correction or negative result
  need not increase coherence or support a preferred interpretation.

## 2. Foundations

Write `x=EPI` and `p=DeltaNFR`. The unforced continuous nodal row is

$$\frac{\partial\mathrm{EPI}}{\partial t}=\nu_f\,\Delta\mathrm{NFR}.$$

A complete model additionally specifies pressure and every consumed phase,
capacity, support, clock, input and event law. Holding a quantity fixed is an
explicit premise. Do not invent or coerce a coordinate the model does not consume.

- **Form:** the scalar engine accepts signed real EPI and uniform-real
  `BEPIElement`. Use shared signed-scalar admission. A complex/nonuniform BEPI
  and a magnitude projection cannot substitute for a signed form coordinate.
- **Capacity:** nonnegative. Zero freezes the unforced form row despite nonzero
  pressure; it does not establish full equilibrium. Identifying capacity with
  inverse mass requires a separate model and bridge.
- **Phase:** circular. Use wrapped differences and shared U3 admission. A
  form-derived regional angle is distinct from primitive phase unless the
  complete law preserves their proposed identification.
- **Pressure:** an independently specified driving term. Reconstructing
  `p=x_dot/nu_f` from the evaluated response is not prediction and fails at
  zero capacity. The nodal identity does not uniquely select the pressure mixture.
- **Clock:** declare it and preserve `[p]=[x]/([nu_f][t])`. Structural rates
  require a measurement bridge to laboratory units. Clock changes transform
  all evolution rows; synchronization alone need not define a monotone clock.
- **Support and identity:** distinguish supplied state, initialization, admitted
  events, pattern formation, maintenance and physical identification. Specify
  an NFR's identity and evolution. Nesting or configured child creation does
  not derive the substrate's origin.

Read the [form foundation](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/FUNDAMENTAL_THEORY.md),
[parameter foundation](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md)
and selected law's owner before changing definitions. Regular chart changes,
exact symmetries and lossy observations require different proofs.

## 3. Execution and operator contracts

Choose the actual execution path before editing or applying a result:

- **Operator studies:** `StudySpec` and `run_study` own the shared `tnfr network`
  workflow. Extend the shared owner before adding CLI-specific execution or
  validation. Cycles count complete words, not seconds. The runner sets topology
  and execution seeds; `TNFR.create(..., seed=...)` sets the topology seed.
- **Continuous execution:** reuse `dynamics/integrators.py` with explicit
  pressure, capacity, clock and rate/residual evidence. Held and refreshed
  pressure are different models. The live Gamma registry supplies a declared
  source in `dx/dt=nu_f*p+Gamma`; no branch may silently drop it. Unforced
  certificates require Gamma absent or zero.
- **Relational execution:** `RelationalExchangeModel` and
  `Network.step_relational` select `dynamics/relational.py`, with admitted
  support, held capacity and phase domain. Its storage/capacity premises are
  explicit. The general operator runtime remains a separate model.
- **Observations and proofs:** regional reports, hypothetical attachments and
  capture/transit checks do not install laws, add live edges or select events.
  Apply every theorem hypothesis. An Euler step, a continuous theorem and a
  validated enclosure provide different evidence.

The 13 named operators use `operators/operator_contracts.py`. Read executable
`token` values from `TNFR.operators()` and words from `list_sequences()`;
do not create parallel catalogs or infer tokens from display names. Preserve
word order and live preconditions. Direct glyphs, public classes and atomic
stages have distinct secondary effects.

Grammar bases belong to `operators/grammar_canon.py`; role sets derive through
shared predicates and `grammar_types.py`. U1-U6 are admission/monitoring
contracts, not universal stability or autonomous-selection theorems.
`GRAMMAR_REJECTION_MODE="raise"` prevents blocked-step substitution. Read the
[grammar](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/UNIFIED_GRAMMAR_RULES.md)
and [operator](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/STRUCTURAL_OPERATORS.md)
owners before changing semantics.

Mutation requires active capacity, a live finite signed two-sample EPI secant
strictly above its configured threshold, and independent grammar context.
Timestamped history ends at the live state; step-indexed history declares
operator-step units. A nodal-product prediction cannot replace observed trigger
evidence. Missing/stale evidence rejects execution or yields explicit abstention.

Named events are hybrid jumps, not invented finite-duration pressure flows.
Atomic stages propose from one immutable snapshot, validate all targets and
commit graph-owned state together; arbitrary callback rollback and future
stability do not follow. Capture optional diagnostics only when requested.
Network REMESH has a separate history/mixing contract; its node advisory does
not execute delayed mixing. See [event contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/contracts/OPERATOR_EVENTS.md)
and [relational contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/contracts/RELATIONAL_DYNAMICS.md).

## 4. Numerical and observation integrity

Reuse admission before float conversion, caching or array construction.
`_exact_time.py` owns represented-real admission, `types.py` signed scalar EPI,
and `alias.py` attribute reads/writes. Invalid authoritative values cannot fall
through to another alias. Reject nonfinite inputs and nonzero values lost during
materialization. Booleans are not physical scalars or clocks; finite inputs
still require representable output arithmetic. Reuse shared Boolean parsing
and parameter resolution rather than implementing competing defaults.

Pressure channels have different support semantics. EPI uses conductance
weights; phase, capacity and degree contrast use unique outgoing neighbors,
including zero-conductance edges. Capacity contrast drives form, not capacity
evolution. The nonlinear neighbor-resultant phase channel is not globally a
pairwise Laplacian. Pure-EPI diffusion results retain their fixed/reversible/
positive-capacity hypotheses. Read the [pressure contract](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md#21-the-implemented-pressure-is-a-specified-relational-map).

Compute fields through `physics/fields.py` and coherence reductions through
`metrics/common.py`. Preserve the following boundaries:

- The tetrad is diagnostic, not a complete state or future predictor. Field
  geometry uses edge `length` with `weight` fallback; diffusion uses conductance.
- Curvature can be undefined at a zero resultant. Coherence-length fit and
  spectral fallback have distinct provenance. Retain independent availability.
- Stored pressure, refreshed fields, predicted rates and measured temporal
  evidence are different. A snapshot or operator label cannot prove conservation.
- Missing observations/baselines remain explicitly unavailable, not zero or
  passing safety flags. Invalid thresholds cannot certify safety; preserve
  the consumer's rejection contract. A tolerance alert is not exact equality,
  equilibrium or endpoint nonincrease.
- `C`, `Si`, classifiers and thresholds are diagnostics or policies. Their use
  by a controller does not derive that controller. Pi-based formulas do not
  establish universal physical constants.

Use `diagnose_network` for detached stored-state observations. Other readers
may maintain rebuildable caches; follow their declared mutation scope.
[Structural fields](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/STRUCTURAL_FIELDS_TETRAD.md)
and [API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md)
own definitions, causal reporting and numerical limits.

JSON readers share `utils.io.json_loads`; SDK exports use the shared atomic
writer. Preserve duplicate-key rejection, finite-number admission and explicit
availability for projected values. A recipe/export is neither a complete
checkpoint nor provenance authentication.

## 5. Research and evidence discipline

The [execution plan](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
is the sole task queue. The [portfolio](https://github.com/fermga/TNFR-Python-Engine/blob/main/TNFR_lineas_de_investigacion.txt)
classifies branches; the [strategy](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_RESEARCH_STRATEGY.md)
explains their rationale. An existing example or theorem does not create an
active campaign. Use the plan's F1-F4 admission method:

1. Define sufficient state, domains, units, observations and discarded information.
2. Declare all laws, separating assumptions, derived restrictions and remaining
   freedom. Supplied forcing or a controller is not an inferred cause.
3. Check joint consistency, symmetry, boundaries, balances and events on the
   actual model. Combine results only when their hypotheses are compatible.
4. Make a discriminating prospective prediction, or prove an equivalence or
   obstruction. Freeze preparation, law, clock, observation, horizon and
   numerical budget before evaluating the reserved response.

Use the [theory-to-execution map](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md#theory-to-execution)
to reuse derivations, implementations and tests. Tangent reduction need not
close a nonlinear law. Hidden-state elimination can introduce memory and
retains hidden initial state and forcing. Formation, recovery, winding,
lifetime and maintenance have separate obligations; an admissible connection
need not have a derived occurrence law.
Account for support-event storage jumps separately from continuous loss.
Loss is not a reusable event reserve without a declared reservoir and law.
Passivity is an additional event premise and does not select occurrence.

Integrate justified reusable mechanisms into shared engine owners and consuming
interfaces. Auxiliary constructions and arithmetic applications keep their own
premises. A finite comparison does not select a unique microscopic ontology or
transfer a proof to another runtime.

Physical evaluation follows P1-P5: independent observation/preparation and clock
models, separate calibration and reserved evaluation, suitable alternatives,
then partial-observation, memory and maintenance tests under the same protocol.
A collective measurement map is allowed; EPI need not be a sensor reading.
Use workstation-accessible terrestrial data or laboratory-scale protocols,
without extraterrestrial observations or large facilities.

Record source/configuration, support/node order, inputs, seeds, precision,
backend, pressure refresh, events and residuals. A seed alone is insufficient.
Preserve frozen protocols, source archives, responses and verdicts. Corrections
get separately identified evidence; do not rewrite an evaluated prediction or
regenerate a frozen producer for an unrelated change.

## 6. Documentation and implementation owners

| Need | Owner |
| --- | --- |
| Introduction and first use | [README](https://github.com/fermga/TNFR-Python-Engine/blob/main/README.md) |
| Definitions, proofs and theory/code/test map | [Theory catalog](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md) |
| Concept classification | [Glossary](https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/GLOSSARY.md); change definition/proof owners before their concept summaries |
| Package boundaries and shared execution | [Architecture](https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md) |
| Guide responsibilities and website navigation | [Documentation map](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md) |
| Public usage and execution contracts | [CLI/SDK](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md), [API contracts](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md), [regional workflows](https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/guides/REGIONAL_AND_RELATIONAL.md) |
| Development and checks | [Contributing](https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md), [Testing](https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md), [Workflows](https://github.com/fermga/TNFR-Python-Engine/blob/main/.github/WORKFLOWS.md) |
| Runnable illustrations and instruments | [Examples](https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md), [benchmarks](https://github.com/fermga/TNFR-Python-Engine/blob/main/benchmarks/README.md) |

Catalogs generate their checked website menus; the operator registry generates
its contract table. Glossary checks validate declarations/references, not
scientific truth. Do not create duplicate catalogs, rule tables or glossaries.

## 7. Development and verification workflow

Inspect instructions, local changes, relevant source and tests before editing.
Preserve unrelated work. Start from the mathematical question and actual
consumer. Prefer shared kernels; retain useful public adapters. Keep APIs
stable or document and test an explicit migration.

Update the implementation and its responsible contract/theory document together.
Keep this file focused on working rules and its mirror byte-identical. After
changing a catalog or operator registry, refresh generated views with
`python scripts/check_documentation.py --write-generated`. Edit maintained
inputs rather than `build/docs-source/` or `site/`.

- Check actual postconditions, invalid domains, representation boundaries,
  atomicity and provenance. Do not assume every valid sequence increases `C`,
  Resonance always synchronizes, or Dissonance always creates a bifurcation.
- Exercise the shared owner with independent expected results or meaningful
  counterexamples. Asserting fixture literals against themselves is not evidence.
- Reuse expensive producer reports through module fixtures. Test CLI/report
  wiring separately when no scientific execution is needed; retain necessary
  cold-import, isolation and provenance controls.
- Run the affected contracts and select a research owner explicitly when its
  model or claim changes. The routine engine/API gate is not full research
  coverage. Test selection and dependencies belong to the testing guide.
- For documentation, verify the mirror, generated views, links, executable
  examples and strict site build. Report checks and material limitations.

Distinguish grammar rejection, live-state rejection, solver defects, unavailable
observations and unsupported theorem domains. A coherence decrease violates
only a contract that promises otherwise. Commits and PRs explain the concrete
problem, resulting behavior, mathematical scope and validation, following the
repository template where present.

## 8. Canonical invariants

These six invariant identifiers retain this order for registry references.
Detailed admission and verification belong to the linked contracts.

1. Preserve continuous nodal flow and declared hybrid jumps with their provenance.
2. Enforce circular phase admission before U3 coupling/resonance.
3. Retain declared nested structure and test the applicable U5 contract.
4. Use registered semantic operators or explicitly justified new contracts;
   numerical evolution uses the shared integrator.
5. Preserve structural units and expose the relevant structural telemetry.
6. Make execution reproducible under fixed source, configuration, inputs,
   seed, target order, precision and backend; a seed alone is insufficient.
