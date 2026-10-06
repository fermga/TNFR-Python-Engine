# TNFR: contributor and agent instructions

**TNFR Python Engine 0.0.3.8.** Rules for changing the repository and evaluating
its claims. Keep `.github/agents/my-agent.md` byte-identical to this file.
The [README] introduces TNFR; the [owner map](#6-documentation-and-implementation-owners)
routes definitions, contracts and implementation work.

## 1. Scientific and editorial responsibility

TNFR investigates whether justified nodal dynamics can produce coherent,
interacting patterns and eventually explain measurable physical properties.
The repository provides an engine, conditional mathematics and finite evidence.
Physical identification and a uniquely selected fundamental law remain open.

Evaluate claims against definitions, hypotheses, derivations and counterexamples,
then implementation and relevant tests. Code establishes implemented behavior,
not mathematical truth; labels, summaries and filenames prove nothing.
Foundational and constitutive premises remain revisable: identify the affected
model and scope, preserving valid conditional results and counterexamples.

- Use English for code, documentation, comments, commits, issues and PRs;
  preserve verbatim quotations and raw data in their original language.
- Distinguish definitions, supplied laws, derived identities, conditional
  theorems, finite observations, policies and open hypotheses.
- Explain the present model directly. Definitions belong with their owners,
  usage in guides and task status in the execution plan; overviews are neither
  delivery logs nor copies of derivations.
- Arithmetic encodings, graph analogies, auxiliary Hamiltonians and diagnostic
  scores are not evidence of particles, quantum physics or consciousness.
- Accept changes against the relevant contract. Corrections and negative
  results need not increase coherence or support a preferred interpretation.

## 2. Foundations

Write `x=EPI`, `p=DeltaNFR`. The unforced continuous nodal row is

$$\frac{\partial\mathrm{EPI}}{\partial t}=\nu_f\,\Delta\mathrm{NFR}.$$

A complete model specifies pressure and every consumed phase, capacity,
support, clock, input and event law. Holding a quantity fixed is an explicit
premise. Do not invent or coerce coordinates the model does not consume.

- **Form:** signed real EPI and uniform-real `BEPIElement` use shared
  signed-scalar admission. Complex/nonuniform BEPI or magnitude projection
  cannot substitute for signed form.
- **Capacity:** nonnegative. Zero freezes the unforced form row even at nonzero
  pressure, not the complete state. Inverse-mass identification needs its own
  model and measurement bridge.
- **Phase:** circular; use wrapped differences and shared U3 admission.
  A form-derived regional angle is not primitive phase unless the complete
  law preserves that identification.
- **Pressure:** an independently specified driver. Reconstructing `p=x_dot/nu_f`
  from the evaluated response is circular and fails at zero capacity. The
  nodal identity does not uniquely select a pressure mixture.
- **Clock:** declare it and retain `[p]=[x]/([nu_f][t])`. Transform every
  evolution row under a clock change. Structural rates need a laboratory-unit
  bridge; synchronization need not define a monotone clock.
- **Support and identity:** distinguish supplied state, initialization, admitted
  events, formation, maintenance and physical identification. Define an NFR's
  identity and evolution; nesting or configured child creation does not derive
  the substrate's origin.

Read the [form foundation], [parameter foundation] and selected law's owner
before changing definitions. Regular chart changes, exact symmetries and lossy
observations require different proofs.

## 3. Execution and operator contracts

Choose the execution path before applying a result or changing its consumer:

| Path | Shared owner and boundary |
| --- | --- |
| Operator studies | `StudySpec` / `run_study` own `tnfr network`; extend them before CLI-specific execution or validation. Cycles count complete words, not seconds. The runner sets topology and execution seeds; `TNFR.create(..., seed=...)` sets only the topology seed. |
| Continuous flow | Reuse `dynamics/integrators.py` with pressure, capacity, clock and rate/residual evidence. Held and refreshed pressure are different models. Retain the live Gamma registry source in `dx/dt=nu_f*p+Gamma`; unforced certificates require Gamma absent or zero. |
| Native relational flow | `RelationalExchangeModel` / `Network.step_relational` select `dynamics/relational.py`, with admitted support, held capacity, phase domain and storage premises. This differs from the general operator runtime and the [separate normalized-sine comparison]. |
| Detached observations/proofs | Regional reports, hypothetical attachments and capture/transit checks install no laws, edges or event selectors. Apply every hypothesis; Euler steps, continuous theorems and validated enclosures provide different evidence. |

The 13 operators use `operators/operator_contracts.py`. Read executable `token`
values from `TNFR.operators()` and words from `list_sequences()`, not display
names or parallel catalogs. Preserve word order and live preconditions;
direct glyphs, public classes and atomic stages have different secondary effects.

Grammar bases belong to `operators/grammar_canon.py`; role sets derive through
shared predicates and `grammar_types.py`. U1-U6 are admission/monitoring
contracts, not universal stability or autonomous-selection theorems.
`GRAMMAR_REJECTION_MODE="raise"` prevents blocked-step substitution.
Read the [grammar] and [operator] owners before changing semantics.

Mutation requires active capacity, a live finite signed two-sample EPI secant
strictly above its configured threshold, and independent grammar context.
Timestamped history ends at the live state; step-indexed history declares
operator-step units. Nodal-product predictions cannot replace observed trigger
evidence. Missing/stale evidence rejects execution or yields explicit abstention.

Named events are hybrid jumps, not invented finite-duration pressure flows.
Atomic stages propose from one immutable snapshot, validate every target and
commit graph-owned state together; callback rollback and future stability do
not follow. Capture optional diagnostics only when requested. Network REMESH
has its own history/mixing contract; its node advisory does not execute delayed
mixing. Use the [event contracts] and [relational contracts].

## 4. Numerical and observation integrity

Admit values before float conversion, caching or array construction:
`_exact_time.py` owns represented-real admission, `types.py` signed scalar EPI,
and `alias.py` attribute access. Invalid authoritative values cannot fall
through to another alias. Reject nonfinite inputs, Boolean physical scalars or
clocks, and nonzero values lost during materialization. Finite inputs still
require representable output arithmetic. Reuse shared Boolean parsing and
parameter resolution instead of competing defaults.

Pressure support differs by channel: EPI uses conductance weights; phase,
capacity and degree contrast use unique outgoing neighbors, including
zero-conductance edges. Capacity contrast drives form, not capacity evolution.
Neighbor-resultant phase pressure is not globally a pairwise Laplacian.
Pure-EPI diffusion retains its fixed/reversible/positive-capacity hypotheses;
see the [pressure contract].

Use `physics/fields.py` for fields, `metrics/common.py` for coherence reductions
and `diagnose_network` for detached stored-state observations. Other readers
may maintain rebuildable caches only within their declared mutation scope.
The [structural field guide] and [API contracts] own these boundaries:

- The tetrad is diagnostic, not complete state or a future predictor. Geometry
  uses edge `length` with `weight` fallback; diffusion uses conductance.
- Curvature may be undefined at zero resultant. Coherence-length fit and
  spectral fallback retain separate provenance and availability.
- Stored pressure, refreshed fields, predicted rates and measured temporal
  evidence differ; snapshots and operator labels cannot prove conservation.
- Missing observations/baselines remain unavailable, not zero or passing flags.
  Invalid thresholds cannot certify safety. Preserve rejection contracts;
  tolerance alerts are not exact equality, equilibrium or endpoint nonincrease.
- `C`, `Si`, classifiers and thresholds are diagnostics/policies. Using them
  in a controller does not derive that controller; pi-based formulas do not
  establish universal physical constants.

JSON readers use `utils.io.json_loads`; SDK exports use the shared atomic
writer. Retain duplicate-key rejection, finite-number admission and explicit
projected-value availability. Recipes/exports are neither complete checkpoints
nor provenance authentication.

A report-consuming calculation must re-admit primitive state, law and evidence
and rebuild consumed derived fields. Cached bounds/verdicts cannot replace
premises. Compute with normalized values before retaining an equivalent source
association: Python equality is neither scalar admission nor authentication.
Distinguish circular observations from continuous phase lifts; declare the lifts
used by collective means. A lift-dependent storage remainder need not be
nonnegative. Preserve proved cycle/mean correlations: overlapping marginal
state budgets do not prove entry.

## 5. Research and evidence discipline

The [execution plan] is the sole queue; the [portfolio] classifies branches and
[strategy] explains their rationale. Existing examples/theorems do not activate
campaigns. On closing a gate, replace its queue status and link the result
owner. Preserve unresolved dependencies and frozen evidence without per-turn
logs or duplicate result inventories. Follow F1-F4:

1. Define sufficient state, domains, units, observations and discarded information.
2. Declare all laws, separating assumptions, derived restrictions and remaining
   freedom. Supplied forcing/controllers are not inferred causes.
3. Check joint consistency, symmetry, boundaries, balances and events on the
   actual model.
4. Make a discriminating prospective prediction, or prove an equivalence or
   obstruction. Freeze preparation, law, clock, observation, horizon and
   numerical budget before evaluating the reserved response.

Reuse the [theory-to-execution map]. Combine results only after matching complete
law, support, capacity, clock, preparation and retained coordinates. Methods can
transfer between models; their coefficients and verdicts do not automatically
transfer. Tangent reduction need not close a nonlinear law. Eliminating hidden
state can introduce memory and retains hidden initialization and forcing.
Formation, recovery, winding, lifetime and maintenance have separate obligations:
response peaks, pulses, recurrence, invariant families and finite retention do
not establish formation or autonomous scale selection. An admissible connection
or passive event does not select its occurrence.

Account for support-event storage jumps separately from continuous loss. Loss
is not an event reserve without a declared reservoir and law; event passivity
is an additional premise. Integrate justified reusable mechanisms in shared
engine owners and consumers. Auxiliary constructions and arithmetic applications
retain their own premises; finite comparisons do not select a unique ontology.

Physical evaluation follows P1-P5: independent observation/preparation and clock
models, separate calibration/reserved evaluation, alternatives, then partial
observation, memory and maintenance under the same protocol. Collective
measurement maps are allowed; EPI need not equal a sensor reading. Use
workstation-accessible terrestrial data or laboratory-scale protocols, without
extraterrestrial observations or large facilities.

Record source/configuration, support/node order, inputs, seeds, precision,
backend, pressure refresh, events and residuals; a seed alone is insufficient.
Preserve frozen protocols, source archives, responses and verdicts. Corrections
need separate evidence: do not rewrite evaluated predictions or regenerate
frozen producers for unrelated changes.

## 6. Documentation and implementation owners

| Need | Owner |
| --- | --- |
| Introduction / first use | [README] |
| Definitions, proofs, theory/code/test map | [Theory catalog] |
| Concept classification | [Glossary]; change definition/proof owners before summaries |
| Package boundaries / shared execution | [Architecture] |
| Guide responsibilities / website navigation | [Documentation map] |
| Usage / execution contracts | [CLI/SDK], [API contracts], [regional workflows] |
| Development / checks | [Contributing], [Testing], [Workflows] |
| Illustrations / instruments | [Examples], [benchmarks] |

Catalogs generate website menus; operator contracts and grammar roles generate
their checked tables. Glossary checks validate declarations/references, not
scientific truth.
Do not create duplicate catalogs, rule tables or glossaries.

Maintained theory/docs owners have a 4,000-physical-line editorial limit checked
by `scripts/check_documentation.py`. Split by model/responsibility; update
canonical links and preserve published aliases. Archive inactive chronology
with its evidence; retain useful conditional results and do not create a second
next-task queue.
Retire superseded drivers, duplicate plans and unused compatibility layers only
after checking imports, public consumers and evidence dependencies. For unchanged
committed files, record recovery revision/blob and replacement rather than
another historical copy. Preserve uncommitted work, unique proofs,
counterexamples, frozen protocols and evaluated responses.

## 7. Development and verification workflow

Inspect instructions, local changes, relevant source and tests before editing;
preserve unrelated work. Start from the mathematical question and consumer.
Prefer shared kernels, retain useful adapters and keep APIs stable or document
and test an explicit migration. Update implementation and its contract/theory
owner together. Keep this file focused on working rules.

After catalog/registry edits, refresh generated views with
`python scripts/check_documentation.py --write-generated`. Edit maintained
inputs, not `build/docs-source/` or `site/`.

- Check postconditions, invalid domains, representation boundaries, atomicity
  and provenance. Valid words need not increase `C`; Resonance need not
  synchronize, and Dissonance need not create a bifurcation.
- Exercise shared owners with independent expected results or meaningful
  counterexamples; fixture literals asserted against themselves are not evidence.
- Reuse expensive reports through module fixtures. Test CLI/report wiring
  separately when scientific execution is unnecessary; retain cold-import,
  isolation and provenance controls.
- Run affected contracts and explicitly select research owners whose models or
  claims changed. The routine engine/API gate is not full research coverage;
  [Testing] owns selection and dependencies.
- For docs, verify mirror, generated views, links, executable examples and strict
  site build. Report checks and material limitations.

Distinguish grammar rejection, live-state rejection, solver defects, unavailable
observations and unsupported theorem domains. Coherence decrease violates only
contracts promising otherwise. Commits/PRs state the problem, resulting behavior,
mathematical scope and validation, using the repository template where present.

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

[README]: https://github.com/fermga/TNFR-Python-Engine/blob/main/README.md
[Theory catalog]: https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md
[API contracts]: https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/API_CONTRACTS.md
[form foundation]: https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/FUNDAMENTAL_THEORY.md
[parameter foundation]: https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md
[grammar]: https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/UNIFIED_GRAMMAR_RULES.md
[operator]: https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/STRUCTURAL_OPERATORS.md
[event contracts]: https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/contracts/OPERATOR_EVENTS.md
[relational contracts]: https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/contracts/RELATIONAL_DYNAMICS.md
[separate normalized-sine comparison]: https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#detached-normalized-sine-complete-law-comparison
[pressure contract]: https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_PARAMETER_FOUNDATIONS.md#21-the-implemented-pressure-is-a-specified-relational-map
[structural field guide]: https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/STRUCTURAL_FIELDS_TETRAD.md
[execution plan]: https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate
[portfolio]: https://github.com/fermga/TNFR-Python-Engine/blob/main/TNFR_lineas_de_investigacion.txt
[strategy]: https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/NODAL_RESEARCH_STRATEGY.md
[theory-to-execution map]: https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/README.md#theory-to-execution
[Glossary]: https://github.com/fermga/TNFR-Python-Engine/blob/main/theory/GLOSSARY.md
[Architecture]: https://github.com/fermga/TNFR-Python-Engine/blob/main/ARCHITECTURE.md
[Documentation map]: https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/README.md
[CLI/SDK]: https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/CLI_AND_SDK.md
[regional workflows]: https://github.com/fermga/TNFR-Python-Engine/blob/main/docs/guides/REGIONAL_AND_RELATIONAL.md
[Contributing]: https://github.com/fermga/TNFR-Python-Engine/blob/main/CONTRIBUTING.md
[Testing]: https://github.com/fermga/TNFR-Python-Engine/blob/main/TESTING.md
[Workflows]: https://github.com/fermga/TNFR-Python-Engine/blob/main/.github/WORKFLOWS.md
[Examples]: https://github.com/fermga/TNFR-Python-Engine/blob/main/examples/README.md
[benchmarks]: https://github.com/fermga/TNFR-Python-Engine/blob/main/benchmarks/README.md
