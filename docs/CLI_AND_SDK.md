# CLI and SDK usage

Use the SDK to prepare an organized state, select its complete law, evolve or
assess it, and retain the evidence needed for the next question. A `Network`
holds graph-owned state; the object's identity, admissible actions and
observation map belong to the selected model and its contracts.

The native object route below combines public execution and recovery readers.
Other collective constructions use scoped mathematical owners. Requested
operator words have a shared CLI/Python runner through `StudySpec` and
`run_study`. Choose the interface that implements the intended law and action;
an API name or shared report format does not make those routes interchangeable.

## Install and discover

```bash
python -m pip install -e .
tnfr --help
python -m tnfr --help
```

Run these commands at the repository root to use the source documented here.
`python -m pip install tnfr` instead installs the published release; inspect
that installation's help for its available interfaces. The examples below
need the core package only. Plotting and optional numerical backends have
separate extras listed in the [root README](../README.md#installation).

## Work with an organized object

Start with the [maintained introductory construction](../examples/01_foundations/01_hello_world.py).
It prepares a winding-one cycle with nonuniform signed form and keeps its
complete state in `Network`. `RelationalExchangeModel` declares the native
joint form/phase law. The
[report-reading guide](guides/relational/RELATIONAL_CAPTURE_AND_MEMORY.md#read-an-organized-objects-recovery-report)
explains the returned records:

- `Network.relational_cycle_capture` assesses sufficient conditions for ideal
  recovery from the supplied state and declared cycle.
- `Network.step_relational` advances the live state through one admitted
  finite step and reports the numerical storage-balance defect.
- Reassessing capture at the endpoint applies the theorem to that snapshot;
  retaining the full state and law supports a subsequent response study.

Winding labels this object's recovery class. It does not replace the internal
form, phase, capacity or support needed to predict an interaction. The example
supplies the initial organization; its prior formation is a separate claim.
The [regional workflow index](guides/REGIONAL_AND_RELATIONAL.md) routes other
preparations, collective state descriptions and observation interfaces.

For operator actions, retain the actual targets and the history/context consumed
by their preconditions. Applying an operator to a constituent is not
automatically an action on the organization as a whole. Use the
[collective action criterion](../theory/STRUCTURAL_OPERATORS.md#collective-grammar-descent)
before interpreting a requested word as collective evolution. The registered
word example below exercises its own prepared graph and execution path.

<a id="create-evolve-and-diagnose"></a>
## Create, evolve and diagnose an operator study

```python
from tnfr.sdk import TNFR, diagnose_network

network = TNFR.create(6, seed=42).ring()
network.evolve(steps=1, sequence="basic_activation")
diagnostics = diagnose_network(network)
print(diagnostics)
```

This supplies a six-node ring and one registered operator word. Initial EPI is
zero, capacity is one, and phase is zero. A uniform initial state can retain
uniform diagnostics; this preparation does not demonstrate spontaneous pattern
formation. On this direct API, `seed` configures stochastic topology builders;
runtime operators resolve the separate graph `RANDOM_SEED` setting. The
`run_study` interface below supplies the same declared seed to both owners.

`steps=1` executes one complete word, not one second of physical time. Word
admission and live node/edge preconditions still apply. In particular, a word
requiring resonance with a neighbor can fail on an isolated node in a random
graph. A topology seed makes the random construction repeatable within its
runtime; it does not make an inadmissible preparation valid.

`Network.evolve(..., record=True)` records diagnostics after each completed
word within the same shared invocation as unrecorded execution. All cycles
retain the initial target order and whole-request preflight, including when a
stage creates new support. A failed partial word adds no sample. Earlier
completed stages remain applied, and a post-word diagnostic failure does not
roll back that word; recording adds no transaction beyond the stage contract.

Construction and cloning of a fluent `TNFRNetwork` build an owned canonical
`NetworkConfig` from its five declared fields, re-admitting flags, seed and
initialization ranges. Ranges become owned numeric endpoint tuples.
Subclass copy hooks and undeclared extra fields do not define the engine's
recipe. Later changes to the external declaration cannot alter that recipe
or the network's random stream. A supplied seed for NumPy-backed SDK
initialization must be an integer in `[0, 2**32)`; `None` retains unseeded
initialization. Other topology seed consumers retain their declared domain.

`evolve_grammar_aware` is a separate sequential direct-glyph policy. It resolves
the complete candidate list before selection, uses the supplied order to choose
the first incrementally admitted glyph and abstains when none is admitted.
Live operator failures propagate; filtering does not guarantee state admission,
coherence growth or a complete valid word. Earlier successful operations remain
applied if a later operation fails; this path has no whole-stage rollback.
New support is visited on the next pass. Missing grammar support raises rather
than substituting a different word.

`diagnose_network` observes a detached graph copy. It reads the stored pressure;
it does not refresh pressure, evolve the network, infer a phase law, or supply
missing temporal observations. A nodal product `nu_f * DeltaNFR` is a model-rate
read-out, not a measured finite-time change.
Its state projection admits nonnegative capacity, including zero at nonzero
pressure. An invalid authoritative alias remains unavailable with an error;
it cannot become a valid observation by falling through to another alias.

Invalid or unavailable diagnostics retain explicit error/availability data
instead of inventing a number. Circular curvature can be unavailable at
individual nodes. Coherence-length provenance distinguishes the static
product-fit length from the separate spectral fallback. See the
[tetrad owner](STRUCTURAL_FIELDS_TETRAD.md) for definitions and units. Diagnostic
flags are observations or policies, not authorization to execute an operator
or a guarantee of future stability.

Request additional observations only for the question being studied:

```python
diagnostics = diagnose_network(
    network,
    include_coherence_profile=True,
    include_phase_evidence=True,
    winding_cycles=(ordered_cycle,),
)
```

Here `ordered_cycle` is a supplied sequence of graph node IDs in traversal
order, with the closing edge present. The optional `observations` section
retains profile fit evidence, represented phase resultants and the declared
winding cycles with independent availability. Unsupported cycles remain
explicit. The default call keeps its existing schema and computes no optional
channels. State projection already retains signed EPI, pressure, phase and
nonnegative capacity. Choose observations using the
[reconstruction requirements](../theory/MINIMAL_STRUCTURAL_DEGREES.md), and
retain each report's scope and estimator provenance when comparing results.

Both SDK interfaces share the circular-mean availability policy: a vanishing
finite resultant has no mean direction, while an invalid authoritative phase
raises instead of falling through to another alias. The global mean's numerical
tolerance is distinct from the tetrad curvature's exact represented-resultant
criterion. Signed pressure means and population spreads reuse stable shared
reductions, so an overflowing intermediate sum cannot turn a finite constant
sample into infinite dispersion. Density follows the graph's directedness;
loops and parallel edges can give density above one.

`compare_networks` and `compute_network_statistics` re-admit consumed report
values before calculation: coherence lies in `[0, 1]`, average capacity is
nonnegative, and supplied numerical observations must be finite represented
reals. Invalid values raise; absent optional averages remain omitted.
Comparisons inspect only the selected columns. Extrema and reductions use the
admitted representation while signed pressure retains its sign.

Fluent `measure()` computes its metrics on one detached graph snapshot. Core
metric failures propagate; optional unified-field failures retain explicit
`unified_fields_available` and `unified_fields_error` metadata. Exported scalar
maps are detached from the result. Comparison tables display unavailable
measurements as unavailable and include columns present in any supplied row.

Fluent `save(path)` delegates to `export_to_json`, refreshing the current
measurements once and returning the network for chaining. It writes metadata
and a measurement report, not a graph checkpoint. Validation and encoding
precede atomic destination replacement.

`StructuralObservation` detaches payloads on construction and export, validates
provenance and optional tolerance, and retains Python value types. The graph
adapters preserve opaque node-label identity while copying field containers.
Nested payloads are not recursively frozen and the envelope is not itself a
JSON encoder. `StudyResult` instead enforces its string-key JSON schema and
rejects invalid numerical values before constructing a retained report.

## Regional and relational workflows

<a id="observe-regional-form-and-its-nodal-response"></a>
<a id="retain-orientation-relative-to-a-held-source"></a>
<a id="execute-the-conditional-relational-model"></a>
<a id="observe-a-prepared-relational-pattern"></a>
<a id="compare-a-supplied-connection"></a>
<a id="check-a-protected-relational-basin"></a>
<a id="validate-continuous-transit-to-a-protected-basin"></a>

The [regional and relational SDK guide](guides/REGIONAL_AND_RELATIONAL.md)
contains the Python preparations for form observations, supplied sources,
conditional joint evolution, pattern reports, bridge/joint-reset comparisons
and capture certificates. Its
[coefficient-response workflow](guides/relational/OBSERVATION_AND_INFORMATION.md#bound-a-prepared-coefficient-response)
also covers module-level jet/sample uncertainty, exact export and read-only
acquisition audits. Those uncertainty and acquisition readers are module-level
functions, separate from the native `Network` methods and CLI operator studies.
Detailed admission and report fields belong to the
[relational contracts](contracts/RELATIONAL_DYNAMICS.md); their theorems remain
in the [theory catalog](../theory/README.md).

The same guide covers [hidden bridge memory](guides/relational/SINE_RESPONSE_AND_MEMORY.md#retain-sine-bridge-memory),
[exact collective families](guides/relational/SINE_REGIONAL_DYNAMICS.md#phase-offset-partition),
[finite maintenance with uncertainty](guides/relational/SINE_REGIONAL_DYNAMICS.md#moving-pattern-window)
and [actual collective-pulse feedback](guides/relational/SINE_REGIONAL_DYNAMICS.md#collective-pulse-balance).
These are detached module-level readers with different support and evidence
requirements. `relational_report_to_dict` exports their scopes, exact fractions,
intervals and unavailable fields through the shared SDK envelope; exporting a
report neither advances a network nor admits it as input to another theorem.

For acquired winding classes, recovery, joined interfaces and complete-state
readouts, follow the [formed-object workflow](guides/relational/SINE_PATTERNS.md#formed-object-sdk-workflow).
It uses the existing mathematical owners and the same exact report exporter,
with a zero-horizon storage example that performs no evolution. These APIs
retain their supplied source and complete law; they do not implicitly read
`Network.G` or change the native relational execution model.

## Run and export the same study from either interface

The Python declaration is explicit and serializable:

```python
from tnfr.sdk import StudySpec, export_to_json, run_study

spec = StudySpec(
    nodes=6,
    topology="ring",
    seed=42,
    sequence="basic_activation",
    cycles=1,
    name="ring-study",
)
result = run_study(spec)
export_to_json(spec.to_dict(), "study.json")
export_to_json(result, "report.json")
```

The equivalent CLI preparation is:

```bash
tnfr network --nodes 6 --topology ring --seed 42 --sequence basic_activation --steps 1 --name ring-study --export-spec study.json --output report.json
```

The CLI maps `--steps` to the declaration's `cycles`. To run an already exported
declaration:

```bash
python -m tnfr network --spec study.json --output replay-report.json
```

`--spec` is exclusive with preparation options such as `--nodes`, `--seed` and
`--steps`. Edit or construct the declaration before running it; an additional
command-line value does not silently override a recorded input.

Python reads the same file through the strict declaration constructor:

```python
from tnfr.sdk import StudySpec, import_from_json, run_study

spec = StudySpec.from_dict(import_from_json("study.json"))
replayed = run_study(spec)
report = replayed.to_dict()
```

`StudySpec` is immutable and validates its declared inputs. Its fields are
`nodes`, `topology`, `seed`, `sequence`, `cycles`, `probability` and `name`.
Supported topologies are `ring`, `path`, `star`, `complete` and `random`;
`probability` configures random edges. `sequence` names a registered word.
Each call creates its own prepared network rather than continuing another run.
SDK topology probabilities admit the original nonboolean real in `[0, 1]`
before binary64 conversion; a nonzero source cannot become zero. Coupling
sweeps validate both endpoints and integer counts before array or network
construction, using the same topology owner.

The result records the declaration, runtime provenance, realized initial and
final triad/support, and final diagnostics. The registered word is captured
before dispatch, so later registry edits cannot change its reported declaration.
The validated word rejects blocked steps without substituting operators; its
declaration is not a per-target execution trace or an occurrence law.
`to_dict()` returns detached data;
`export_to_json` uses the shared JSON writer. Without `--output`, the CLI writes
JSON to standard output and sends messages to standard error. An output path
receives the report instead. Retain the declaration together with the report
when comparing studies.

JSON export rejects nonfinite numerical payloads before replacing the
destination. Represent unavailable observations with their availability record
and `null`, not a nonstandard `NaN` or `Infinity` token.
The generic exporter also rejects recursively colliding encoded object keys
(for example integer `1` and string `"1"`) before replacing the destination.
Non-colliding key conversions retain the existing JSON encoder's behavior.
`Results.to_dict()` checks tetrad node labels before converting them to
strings, so this conversion cannot silently merge distinct nodes either.
`import_from_json`, also used for CLI recipe input, rejects duplicate decoded
keys at every depth, nonfinite numbers and nonzero fractional literals that
underflow to binary64 zero. Integer literals remain Python integers; ordinary
fractional literals retain binary64 rounding. Decoding neither authenticates
report provenance nor replaces `StudySpec` field validation.
The shared decoder is `tnfr.utils.io.json_loads`; JSON configuration files use
the same admission policy before injection. YAML/TOML keep their parser
semantics and still require validation by the consuming model.

For stored metric histories, `metrics.export_metrics` supports CSV and JSON
with the same admission and availability rules. Missing samples are empty CSV
cells or JSON `null`; clocks come only from their retained stream. It never
replaces absent measurements with zero or invents timestamps. See the
[stored-observation contract](API_CONTRACTS.md#stored-observations-and-validation)
for invalid-value rejection and the scope of atomic output.

Decimal scalar flags share a CLI parser that rejects nonfinite values and
nonzero literals lost to binary64 zero before command execution. Exact zero
and representable subnormals remain valid; each consumer still checks its
range, sign and model domain. Programmatic namespaces receive the same
scalar, count, probability and Boolean admission at their consumers, before
coercion or spectral array construction. Text flags use shared Boolean parsing;
Python truthiness cannot turn a supplied `"false"` into an enabled policy.

| Report key | Content |
| --- | --- |
| `spec` | Validated declaration, including the seed and requested cycles |
| `execution` | Package/runtime versions, precision mode, registered word and completed cycles |
| `initial_state` | Indexed scalar triad/pressure observations and support before execution |
| `final` | Final state projection, metrics, nodal observations and tetrad with availability/provenance |

Other engine configuration is inherited from the running process. The report
does not capture every effective setting; retain relevant external configuration
with the declaration. Scalar state rows use indices in graph iteration order;
display labels are not serialized node identities, and edge attributes are not
included in this projection.

The executable [reproducible study example](../examples/01_foundations/reproducible_study.py)
writes a declaration, reads it back, runs it and exports the result:

```bash
python examples/01_foundations/reproducible_study.py --output-dir output/study
```

## Discover registered words and operators

```bash
tnfr sequences
tnfr sequences basic_activation
tnfr operators
tnfr operators emission
```

These commands expose existing catalogs rather than a second grammar. They
accept `--output` for JSON export. Python accesses the same named words with
`list_sequences()` or `list_sequences("basic_activation")` from `tnfr.sdk`.
The [operator contracts](API_CONTRACTS.md) own operator metadata, while
[grammar scope](../theory/DIAGNOSTIC_AND_GRAMMAR_SCOPE.md) distinguishes admission
policies from mathematical stability results.

Use English operator names when explaining a word. The SDK already returns
these names alongside compact glyph notation; pass ordered executable tokens
when requesting an explanation:

```python
from tnfr.sdk import TNFR

report = TNFR.explain_sequence(["emission", "coherence", "silence"])
print(" → ".join(report["operators"]))  # Emission → Coherence → Silence
print(report["notation"])               # AL → IL → SHA (compact form)
print(report["compatibility_profile"])  # core
print(report["valid"])                  # True

legacy = TNFR.explain_sequence(
    ["emission", "silence"], context={"compatibility_profile": "legacy"}
)
print(legacy["valid"])                  # False: historical adjacency preference
```

The default `core` profile uses the shared word contracts. Explicit `legacy`
adds historical adjacency and THOL-terminal preferences; it does not restore
an alternate causal validator. Both profiles still require live admission
when executing. `context={"initial_epi_nonzero": True}` can describe prior
initialization to a detached word validator; it cannot supply live state or
Mutation evidence. The report leaves manuscript syntax and trajectory checks
unassessed. Arrows are display notation: pass `["AL", "SHA"]`, not
`["AL → SHA"]`, apostrophe spellings or `"THOL[...]"` to this flat-word API.
See the [profile migration contract](API_CONTRACTS.md#word-profile-migration).

## Reproducibility and scope

`Network.nodal_scan()` returns supplied prediction/readout records. Their
`mean_local_coherence` averages admitted local values in [0,1]; it does not
replace invalid negative values by magnitudes. Total coherence still aggregates
the supplied pressure/rate channels independently. Logical verdicts accept
booleans or unavailable `None`, with separate `active_unavailable_count`,
`equilibrium_unavailable_count` and `bifurcation_unavailable_count` totals.
Truthy strings/numbers and contradictory prediction aliases reject at report
consumption. Export keeps the existing string-keyed node mapping, but raises
when distinct labels such as `1` and `"1"` collide instead of dropping a node.
These checks do not authenticate a manually constructed report as a live state.

`Network.conservation()` uses strictly increasing retained observation times.
Its balance and candidate-energy secant use the same represented interval;
missing intervals remain unavailable. `candidate_energy_nonincreasing` reads
the admitted secant's sign, which agrees with the captured endpoint ordering;
`candidate_energy_within_numerical_tolerance` retains the separate numerical
alert. A small positive energy change is still an increase even when it passes
that alert. Nonzero unrepresentable temporal rates reject instead of reporting
perfect balance. Failed combined observations do not append partial evidence;
returned tracker snapshots and reports are detached from retained data.
Neither observation establishes general dynamical stability.
Raw scalar and field-map getters require their implementation to be available;
otherwise they raise `ImportError`. Use the structured report's availability
metadata when a caller needs an explicit unavailable result.

`Network.nfr()` is a stored-state observation, not a certificate that an NFR
has formed. Its radial/annular/multinodal labels classify the unit-source
potential centrality profile under a configured policy; a uniform profile
does not establish a literal ring or rotational symmetry. Empty or unsupported
geometry returns `topology="unavailable"`, `topology_available=False` and an
explicit `topology_status`. Consumers must handle that availability state.
An all-zero centrality profile caused by numeric underflow reports
`centrality_below_represented_range`; it cannot establish annular geometry.

Its `coherence_length` uses the shared tetrad estimator, with explicit
`coherence_length_available` and `coherence_length_provenance`. A fitted length has structural
distance units; the spectral fallback is a separate dimensionless scale.
Neither is a fractal dimension. Pressure observations remain available even
when rate/capacity information is missing. Partial or invalid stored rates do
not become equilibrium evidence; only absent rate telemetry permits the
explicitly labelled unforced nodal-product prediction. It does not infer Gamma,
refresh pressure, establish full-state equilibrium or measure persistence.

`depi_dt_status` records rate availability. The nodal-product fallback reuses
the canonical derivative; if two nonzero factors produce a rounded zero, it
reports `nodal_product_underflow` rather than certifying observed stationarity.
This observation rule does not alter runtime product rounding. Tetrad summaries
retain unavailable fields, and their overall safety advisory requires matching
nonempty local field support; an empty snapshot cannot pass it.

`Network.phase()` exposes the classifier's actual imbalance ratios, node count,
and coherence-length availability/provenance. Its phase labels are
configured, size-sensitive diagnostics, not autonomous events or a biological
claim. See the [phase classification scope](../theory/STRUCTURAL_STABILITY_AND_DYNAMICS.md#22-phase-classification).

| Retained item | What it establishes | What it does not establish |
| --- | --- | --- |
| Declaration and seed | Supplied preparation, word and requested cycle count | Autonomous selection of those inputs |
| Initial/final triad and support | Observed endpoints of that finite invocation | Every intermediate state or future behavior |
| Runtime provenance | Context for comparing executions | Bit-identical results across arbitrary versions, platforms or numerical backends |
| Diagnostic values and availability | Read-outs of stored state with estimator scope | Complete reconstruction or a new constitutive law |

A study report is not a complete resumable checkpoint. The declaration can be
executed again from its supplied initial state; the report does not restore
all callbacks, caches, operator history or external state. This distinction also
applies to older `export_to_json` payloads and `import_from_json`, which reads
JSON data without reconstructing a live execution.

For experimental interpretation, the
[research plan](../theory/research/FIVE_STAGE_EXECUTION_PLAN.md) and
[measurement protocol](../theory/research/PASSIVE_TRANSPORT_PROTOCOL.md) remain
the owners of hypothesis selection, calibration and reserved evaluation.
Exporting a study does not by itself satisfy those scientific admission gates.

## Existing advanced execution routes

`tnfr run`, `sequence`, `math.run`, `epi.validate` and `metrics` retain their
specialized execution/configuration paths. Their `--help` output owns the
available flags. They are not aliases for `network`, and their history formats
are not `StudySpec` declarations. Prefer `network` for the shared SDK/CLI study
workflow and use a specialized route when its actual configuration is needed.

### Scoped sequence programs

`tnfr sequence` uses the minimal `canonical_example` preset, `AL → IL → SHA`,
when no explicit preset or sequence file is supplied. It is a different program
from the SDK's registered `basic_activation` word. Select one input source;
`--preset` and `--sequence-file` cannot be combined. A JSON sequence file can
contain `["AL", "IL", "SHA", {"WAIT": 2}]`, directly or under a `sequence` key.
Use `tnfr sequence --sequence-file program.json` to execute it with the route's
configured graph and native runtime steps.

For explicit nesting, inspect the existing typed program representation before
execution:

```python
from tnfr.execution import block, compile_sequence, target
from tnfr.tokens import OpTag
from tnfr.types import Glyph

program = [
    target([0]),
    block(target([1]), Glyph.IL, repeat=2, close=Glyph.SHA),
    Glyph.SHA,
]
operations = compile_sequence(program)
print([payload for tag, payload in operations
       if tag in {OpTag.THOL, OpTag.GLYPH}])
# ['THOL', 'IL', 'SHA', 'IL', 'SHA', 'SHA']
```

This example compiles only. Its THOL opening still needs admitted live history
and state; compilation does not supply them. The complete `operations` list
also retains `BLOCK_ENTER`/`BLOCK_EXIT` metadata. Each repeated body inherits
target 0, selects target 1 locally, and restores target 0 on exit. The final
SHA therefore targets 0. Brackets do not implicitly select newly created
children or execute parallel branches.

Closure is explicit: omitted `close` adds nothing; use `close=Glyph.NUL` to
retain the old implicit NUL. Every supplied closing glyph is emitted and must
pass its actual admission. WAIT and repeat counts are positive non-Boolean
integers; values such as `2.0`, `"2"` or `False` reject. Ordered groups are
retained, cyclic groups and unordered sets reject, and unsupported mapping
fields are not silently ignored.

`play(graph, program, step_fn=...)` validates the complete program syntax before
execution, then dispatches native glyphs and the supplied step function.
Requested glyphs reject grammar failures instead of being replaced. Accepted
earlier operations are not rolled back after a later live failure. The step
function runs after every glyph; WAIT adds further calls. The default native
step may itself select operators and change phase/capacity, so WAIT is not an
operator-free pause or a duration in seconds. Use the
[program contract](API_CONTRACTS.md#program-compilation-contract) for compiled
metadata and migration details, and the
[event contract](contracts/OPERATOR_EVENTS.md) for separately declared hybrid
events and flow intervals. Source prosody or manuscript nesting adds no
unimplemented clock, child law or selector to either route.

For execution provenance, inspect `graph.graph["history"]["program_trace"]`.
Explicit glyph rows name their ordered targets and
`selection_origin="supplied_program"`; `step_origin` identifies the separate
step function. The row exists before stepping, so a failed callback retains
the successful supplied glyph with `step_status="failed"` and its call counts.
`event_time` precedes the step; `t` is the admitted post-step clock on success.
An invalid callback clock leaves the last admitted `t` with
`clock_sample_available=False`. WAIT reports calls without claiming a glyph.
The trace is bounded and cannot replace a complete checkpoint or independent
evidence of which events a controller selected internally.

### Diagnostics and runtime controls

`epi.validate` reports stored affinity, capacity and all-edge phase diagnostics.
Its compatibility flag `--check-coherence` checks the sign of stored `W_mean`
affinity with `--tolerance`; it does not measure canonical C preservation.
Capacity must be finite and nonnegative, and U3 uses its configured hard gate;
the affinity tolerance relaxes neither requirement. `--no-check-coherence`,
`--no-check-frequency` and `--no-check-phase` disable individual checks.
Malformed inputs fail, missing observations remain unavailable, and selecting
no checks or obtaining no observed checks returns a nonzero exit status.

On `run`, explicit `--stop-early-window` or `--stop-early-fraction` options enable
the stopping policy. The runtime requires a Boolean `enabled`, a positive
integer window and a finite real fraction in `[0, 1]`; inactive window/fraction
fields are not consumed. The policy is fixed for the invocation. Stopping
requires new observations and a complete consecutive window of valid recorded
stability fractions. Invalid/missing samples break that window; older retained
telemetry alone cannot stop a new invocation. The built-in metric producer
tracks a sample revision, including with bounded histories. Custom growing
series can signal new samples by increasing their length. An uninstrumented
fixed-size custom buffer supplies no freshness evidence. Only the required
tail is inspected, rather than rescanning the full history after each step.
This remains a configured finite-observation stopping rule, not a proof of
convergence or full-state equilibrium.

`HISTORY_MAXLEN` bounds each retained metric series, not the number of metric
names. Resizing a bound preserves the newest samples; disabling it restores
growing lists. Explicit least-used-key removal remains a separate operation.
Runtime callbacks and candidate sampling share a zero-based execution ordinal,
independent of retained metrics and physical time. Both callback boundaries see
the same index. `current_step_idx` reads that active index during execution and
the next index between calls. The first tracked invocation starts at zero;
old metrics do not reconstruct unobserved runtime history. Admission failure
reserves no index, while an admitted call that later fails consumes its index
because partial state changes may remain. Same-graph recursive calls reject.
Standalone graphs without runtime markers retain their documented history-index
fallbacks; those are not lifetime execution counts. Physical-time observations
continue to use the separately declared clock.

REMESH cooldown uses the runtime ordinal after an epoch has been established;
standalone calls keep their stable-sample-count basis. The stored basis prevents
subtracting those different quantities during migration. The first eligible
successful operation establishes the new basis, while the physical-time
cooldown remains independent. Limiting history does not remove metric names or
prevent bounded REMESH transactions, and detached stage validation cannot append
advisory events into the live history.

The auxiliary `dynamics.computational_hub.batch_execute_computations` dispatcher
accepts a positive integer `max_parallel` and limits that batch's outstanding
requests through the shared hub. Results retain request order, including
adapter failures. A 300-second collection timeout raises `TimeoutError` before
further submissions; already submitted work continues without cancellation or
rollback. This limit does not govern unrelated hub clients. Execution durations
use one monotonic clock.

The [example index](../examples/README.md) classifies the other demonstrations.
The SDK's fluent builders, auxiliary physics adapters and optimizer policies
have their own scopes; this guide does not promote them into autonomous nodal
laws.

## Auxiliary arithmetic command

`tnfr-is-prime` uses the declared integer arithmetic-pressure model, separately
from network evolution. Basic execution reuses the shared divisor and factor
functions. `--optimized` reuses one sieve/result-cache owner; `--batch` evaluates
sorted distinct inputs. `--cached` and `--no-optimize` select the basic route.
`--benchmark N` requires `N >= 100` and prints the mathematics benchmark report;
its timings are machine/input dependent. Automatic arithmetic execution uses
NumPy and requires no GPU runtime.

The historical `tnfr.tools.tnfr_is_prime_cli_optimized` module delegates to this
same command. Its `benchmark_basic` compatibility call returns the shared
benchmark schema (`total_numbers_tested`, `total_time_ms`, `cache_statistics`,
and related fields), replacing its former basic-versus-cached timing schema.
