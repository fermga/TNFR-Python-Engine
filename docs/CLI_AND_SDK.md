# CLI and SDK usage

For operator studies, the CLI and Python SDK share the same declared study
runner. Use a `Network` for direct manipulation, or a `StudySpec` to retain a
reusable preparation and operator word. The SDK also exposes the separate
[conditional relational model](#execute-the-conditional-relational-model).
Each route delegates to its engine execution and observation owners; the
interface adds no physical law.

## Install and discover

```bash
python -m pip install tnfr
tnfr --help
python -m tnfr --help
```

For the checked-out repository, install it with `python -m pip install -e .`.
The commands below need the core package only. Plotting and optional numerical
backends have separate extras listed in the [root README](../README.md#installation).
Use the installed command's help to check its available options.

## Create, evolve and diagnose

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

Invalid or unavailable diagnostics retain explicit error/availability data
instead of inventing a number. Circular curvature can be unavailable at
individual nodes. Coherence-length provenance distinguishes the static
product-fit length from the separate spectral fallback. See the
[tetrad owner](STRUCTURAL_FIELDS_TETRAD.md) for definitions and units. Diagnostic
flags are observations or policies, not authorization to execute an operator
or a guarantee of future stability.

Both SDK interfaces share the circular-mean availability policy: a vanishing
finite resultant has no mean direction, while an invalid authoritative phase
raises instead of falling through to another alias. The global mean's numerical
tolerance is distinct from the tetrad curvature's exact represented-resultant
criterion. Signed pressure means and population spreads reuse stable shared
reductions, so an overflowing intermediate sum cannot turn a finite constant
sample into infinite dispersion. Density follows the graph's directedness;
loops and parallel edges can give density above one.

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

## Observe regional form and its nodal response

`Network.regional_form(regions)` delegates to the shared
[`observe_regional_form`](../src/tnfr/physics/form_geometry.py) owner. Supply
an ordered partition into three-node regions; the order fixes each region's
reporting frame. The partition is an input, not a detected NFR or a selected
topology. For example:

```python
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.sdk import TNFR

network = TNFR.create(6, seed=42).ring()
default_compute_delta_nfr(network.G)
regional = network.regional_form(((0, 1, 2), (3, 4, 5)))
print(regional.regions[0].intensity, regional.regions[0].phase_estimate)
```

The example explicitly evaluates configured pressure before observation;
newly created nodes need not already carry a stored pressure value.
The detached immutable report retains regional means, two Cartesian contrast
coordinates, squared amplitudes and the complete cross-region Gram data,
together with their instantaneous rates. These rates project the shared
unforced nodal product `nu_f * stored_DeltaNFR`. They do not refresh pressure,
measure an elapsed interval or include Gamma, operator jumps or clipping.
Stored primitive phase is neither read nor changed. The zero-form preparation
above has zero contrast and no defined form angle.

Exact rational coordinates retain the represented scalar inputs and rate
arithmetic; square roots and polar read-outs are separately labeled numerical
estimates. An unavailable angle at zero contrast is distinct from a zero
angular rate. Cartesian and Gram observations remain defined there. Missing
consumed state and invalid scalar/partition inputs reject explicitly. EPI must
already be a real scalar or a materialized uniform-real `BEPIElement`;
serialized form containers are not implicitly decoded by this observer.
`contrast_a` and `contrast_b` represent
`z=contrast_a/sqrt(2)+i*contrast_b/sqrt(6)`, while `gram_imag_numerator`
represents `sqrt(12)*Im(zz^dagger)`. The rate matrices use the same scaling.
`nodal_rate_rounding_defect` retains rounded rate minus the exact product of
the represented capacity and pressure. `estimate_status` distinguishes zero
amplitude from an unrepresentable numerical estimate; the exact quantities
remain available even when a polar estimate overflows or underflows.

This observation is available on any graph with an admitted supplied
partition. An autonomous law for the reduced observations requires the separate
[interface and capacity hypotheses](../theory/nodal/DERIVED_FORM_PHASE.md#collective-interaction-closure-and-relational-state).
The report does not establish those hypotheses or select the next operator.
It is an exact-data Python report containing `Fraction` values, not the
JSON-only `StudyResult` schema or a resumable checkpoint.

For a separately declared fixed affine law, use
`tnfr.physics.derive_regional_affine_closure(nodes, regions, generator=G, source=b)`.
It checks whether the regional means and complete Gram matrix have an autonomous
law for all real fine forms. Supply `G` and `b` independently; the function
does not infer the current dynamics from a graph or from a measured derivative.
For held pressure `p=e*G_W*x+F`, these inputs are `G=diag(nu)*e*G_W` and
`b=diag(nu)*F`, so `source` is a form rate, not a pressure.

The immutable report retains exact block-circulant and source-contrast defects.
`all_state_closed` is true exactly when both vanish. For admitted laws it
supplies the affine mean law and complex contrast generator through separate
rational real and `imag_over_sqrt3` matrices. Those generator fields are `None`
on failed admission; failed reduction does not invalidate the fine model.
Rational model coefficients stay exact, while other real coefficients are
materialized as binary64. This differs from the stored-state observer's
binary64 rate arithmetic. The
[affine-source theorem](../theory/nodal/DERIVED_FORM_PHASE.md#held-affine-source-closure)
owns the proof and boundaries, including zero contrast and the need to keep
the source fixed when comparing equal observations. Neither a detached model
report nor one observed state certifies the live runtime's future closure.
On failed admission, retained `mean_generator` and `mean_source` are projected
coefficients; an omitted contrast contribution can still drive the means.
Their presence alone does not establish a complete autonomous mean law.

### Retain orientation relative to a held source

`Network.source_relative_form(regions, held_source_rate=b)` delegates to
[`observe_source_relative_form`](../src/tnfr/physics/source_relative_form.py).
Supply the independent affine **rate** source in the network's node order;
for a pressure source F, this is `b=nu*F`. The interface never reconstructs b
from the response or treats the full stored nodal rate as that source.

```python
# Continue the six-node preparation above; b is a declared model input.
b = (1, -1, 0, 0, 0, 0)
relative = network.source_relative_form(((0, 1, 2), (3, 4, 5)), held_source_rate=b)
print(relative.relative_real, relative.contrast_reconstructible)
```

The immutable report embeds the shared `form` observation and retains source
means and contrasts c. Its full matrix is `W=z*c^dagger`, represented by
`relative_real` and `relative_imag_numerator=sqrt(12)*Im(W)`. Rate fields use
the same scaling, holding c fixed. A changing source needs the additional
`z*c_dot^dagger` term, which this interface does not supply. Node order comes
from `relative.form.nodes`. Exact rational source coefficients remain exact;
other real source coefficients follow the affine-law admission above.

For a known nonzero contrast-source vector, W retains all form contrasts,
including regions whose local source contrast is zero. Diagonal products alone
do not do this. A source constant within every region has c=0, leaving W=0
and `contrast_reconstructible=False`; no direction is invented. This flag
describes information content, not a claim that the input source is physically
justified, that primitive phase/capacity are reconstructible, or that a future
law closes. The [source-relative theorem](../theory/nodal/DERIVED_FORM_PHASE.md#source-relative-engine-integration)
owns the identities and scope. These exact Python reports share neither the
JSON study schema nor a new CLI execution mode; the CLI still uses the single
operator study runner.

## Execute the conditional relational model

```python
from tnfr.sdk import Network, RelationalExchangeModel
import networkx as nx

graph = nx.path_graph(3)
for node, form, capacity in ((0, 0.1, 1.0), (1, 0.0, 0.0), (2, -0.1, 2.0)):
    graph.nodes[node].update(EPI=form, theta=0.0, nu_f=capacity)
network = Network(graph)
model = RelationalExchangeModel(storage_scale=1.0)

field = network.relational_exchange(model)  # detached, no graph mutation
step = network.step_relational(model, dt=0.01)  # one joint Euler step
```

Both calls delegate to [one engine owner](../src/tnfr/dynamics/relational.py).
The model explicitly selects the EPI/phase pressure coefficients and joint
storage scale. Its phase law follows from the conditional
[capacity-separability theorem](../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#capacity-separable-exchange);
neither that capacity premise nor the storage scale is physically selected.
The default operator-word `evolve` route and `StudySpec` keep their existing
semantics. This API supplies a separate explicit continuous-model step.

The execution domain is a connected simple undirected unit-conductance graph
with at least two nodes, signed real scalar form and nonnegative held capacity.
The default `phase_domain="acute"` requires strictly acute represented edge
gaps and keeps the Euler segment on its initial relative-phase lift; a full
turn cannot hide a crossed chart boundary. Uniform-real BEPI is the supported
scalar embedding; richer form, extra forcing and a simultaneous
extended-dynamics declaration reject.

An explicit alternative admits a sufficient regular chamber of the same law:

```python
wider_model = RelationalExchangeModel(
    storage_scale=1.0, phase_domain="positive_resultant"
)
wider_field = network.relational_exchange(wider_model)
lower_bounds = wider_field.resultant_real_lower_bounds  # exact fractions
```

This mode requires a certified positive real part of every relative neighbor
phasor sum. It can admit individual nonacute edges, while remaining narrower
than the full regular phase domain. Rational cosine enclosures consume the
exact represented raw phase angles. Failure to obtain a positive lower bound
can mean that the finite enclosure is inconclusive, not that a singularity has
been proved. The field's scope records the selected domain and enclosure
method; native pressure, metric and rates retain their numerical residuals.

Both initial and proposed endpoint states must be admitted. A step with
`wider_model` additionally retains `segment_resultant_real_lower_bounds`:
initial bounds minus the sum of absolute relative phase increments. Their
strict positivity certifies the whole straight chord between represented
endpoints. It does not certify the exact ODE trajectory or its numerical
error. These optional field/step margins are `None` in default acute mode.
The [execution contract](API_CONTRACTS.md#conditional-relational-execution)
owns the exact admission formulas and limitations.

Each step uses one initial state for both rates, then validates the complete
endpoint before updating the graph. Capacity and support are held. There is
no operator selection, clipping, automatic capacity adaptation or artificial
history. Ambient pressure weights and legacy clipping rails do not define
this explicitly selected model. The report retains its actual coefficients,
state order, phase geometry, native pressure and numerical balance defects.
Its time coordinate is structural; `dt` is not an operator-word count or an
independently calibrated laboratory second.

Continuous storage nonincrease does not imply Euler-step nonincrease. A
successful report can retain a positive energy increment, rather than hide
it behind a tolerance or claim stability. Failed proposal admission leaves
the live graph unchanged. Repeated calls are separate committed steps; they
are not a transaction over an entire user loop.

The [local-recovery theorem](../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-recovery)
requires positive held capacities, positive form dissipation and a prepared
acute equilibrium; it is stronger than successful step admission. The
[paired-region control](../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-region-interaction)
tests transmitted deformation through that same engine law. Both retain
supplied support/phase geometry and leave autonomous formation open.

### Observe a prepared relational pattern

Continuing with the `network` and `model` above, supply a reference phase lift
and ordered regions explicitly:

```python
from tnfr.sdk import export_to_json, relational_report_to_dict

pattern = network.relational_pattern(
    model,
    reference_phase={node: 0.0 for node in graph},
    regions=(tuple(graph),),
)
region = pattern.regions[0]
form_deformation = region.form_norm_squared
phase_deformation = region.phase_norm_squared
signed_exchange = region.work.exchange  # Positive: phase storage toward form.
boundary = region.boundary
if boundary.weighted_rate_unavailable_reason is None:
    phase_rate = boundary.phase_weighted_rate
    phase_rounding_residual = boundary.phase_rate_residual
response = region.phase_response  # Unweighted rates, also at zero capacity.
mean_phase_rate = response.mean_rate
mobility_form_contribution = response.covariance_rate
covariance_squared_bound = response.covariance_rate_squared_bound
export_to_json(relational_report_to_dict(pattern), "relational-pattern.json")
```

This reads one fresh detached field without advancing the graph. Regions and
the reference are inputs, not discovered patterns. The immutable report keeps
the field, reference phases, region means, phase-error offsets and centered
coordinates. It reports form and phase norms separately; it does not assign a
physical unit conversion or certify equilibrium, recovery or a closed reduced
law. Reference errors use the supplied real lifts without automatic wrapping.
Retain a consistent phase frame when comparing successive observations.
The supplied model selects the same phase chamber as direct evaluation;
pattern observation does not broaden that admission.

Supply `cycles=((node_a, node_b, node_c, ...), ...)` for explicitly ordered
support cycles when winding is relevant. Winding and regional transport reuse
their existing owners. Regions may overlap and need not cover the graph.
Transport accounting is unavailable for any zero capacity in the full graph,
zero EPI weight or a whole-support region, with an explicit reason; the
example above therefore still supplies form/phase observations without a
regional transport claim. Where available, transport retains independent
phase forcing and pressure-split defects.

`pattern.field.work` exposes nodal dissipation, signed exchange, actual
form/phase work and separate arithmetic residuals. `region.work` sums these
contributions; overlapping regions must not be summed as a partition.
`region.boundary` pairs form and phase rates using the same outward form cut.
It also supports the full-graph region above and zero form-diffusion weight.
These rates require positive capacity within that region, regardless of a
zero capacity elsewhere. When unavailable, the reason and `None` rates are
explicit while cut and work remain available. The phase quantity is an
instantaneous weighted rate, not a conserved weighted phase total. Reports
observe the selected law and do not drive it.

`region.phase_response` gives the ordinary sum and mean of the captured phase
rates, including at zero capacity. It separates the shared cut contribution
from mobility/form-gradient covariance and retains the rounding residual.
The derived squared bound limits that covariance contribution; it does not
set a new stability threshold. A region with zero cut can still change its
mean phase because its nodes respond differently to their form gradients.
These are instantaneous model predictions, not measured derivatives or a
closed law for a new effective node. The
[composition theorem](../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#regional-phase-mobility-balance)
owns the mathematical interpretation.

The same `relational_report_to_dict` helper projects `field` and `step` reports
for `export_to_json`, retaining fractions as numerator/denominator records in
the versioned `tnfr.relational-report.v1` envelope. The saved report is a
detached observation, not a restart file or authenticated source record. See
the [observation contract](API_CONTRACTS.md#prepared-relational-pattern-observations)
for scope. The CLI's operator-study route continues to use `StudySpec` and its
own report schema.

### Check a protected relational basin

The conditional capture theorem supplies a separate read-only check for two
exactly copied/reflected five-node rings with matching adjacent-port bridges:

```python
import networkx as nx
from tnfr.sdk import Network, RelationalExchangeModel

graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
graph.add_edges_from(((0, 5), (1, 6)))
phase = (9 / 4, -9 / 4, -9 / 8, 0.0, 9 / 8)
for node in graph:
    graph.nodes[node].update(EPI=0.0, theta=phase[node % 5], nu_f=1.0)
model = RelationalExchangeModel(1.0, phase_domain="positive_resultant")
capture = Network(graph).relational_capture(
    model, cycles=(tuple(range(5)), tuple(range(5, 10)))
)
assert capture.admitted
assert capture.target_sector == 1
energy_interval = capture.storage_bounds  # exact rational enclosure
```

This initializes a supplied nonacute phase geometry; it does not evolve or
discover the support. The certificate checks exact copy/reflection, unit
capacities, positive coefficients, a proved phase rectangle and rigorously
bounded joint storage below `7*beta`. It then applies the continuous-law
capture theorem, without simulating a trajectory. Positive twist, consensus
and negative twist have separate sufficient rectangles under the same law.

`target_sector` describes the ideal limiting state, while `winding` describes
the current snapshot. They can differ: the central basin includes winding-one
states that converge to consensus. An unavailable certificate retains reasons
and sets no target; small symmetry errors are never projected away. It does
not guarantee a chosen Euler step or certify spontaneous formation from
winding zero. Export it with the same `relational_report_to_dict` helper.
The [capture contract](API_CONTRACTS.md#conditional-relational-capture) owns
all premises, report fields and limits.

For an actual endpoint near a declared target, exact reflection is unnecessary:

```python
local = Network(graph).relational_local_capture(
    model, cycles=(tuple(range(5)), tuple(range(5, 10))), target_sector=1
)
local_evidence = relational_report_to_dict(local)
```

This applies a different, smaller full-state local basin with unit storage
scale and capacities. It retains exact distance and excess-energy enclosures
and can return unavailable for the nonacute preparation above. An admitted
endpoint has a proved ideal continuation under the supplied law. It does not
certify numerical integration error from an earlier preparation.

`Network.relational_sector_capture(model, cycles=..., target_sector=1)` uses
a larger full-state acute-sector energy basin on the same support. It checks
all wrapped-edge inequalities and exact cycle periods against a derived
energy barrier, without symmetry projection or a small-distance gate.
Only sectors `+1` and `-1` are supported by this theorem. Its report separates
current state evidence, declared target and fully admitted ideal continuation;
use the same exact exporter. It supplies no numerical path-error certificate.
Capacities may be heterogeneous but must stay positive and held; beta may be
any admitted positive storage scale. `future_phase_metric_lower_bounds` and
the corresponding acute/resultant bounds are available only after full
admission. These are derived continuation guarantees, not commands to change
capacity or select operators. The [maintained example](../examples/08_emergent_geometry/180_relational_exchange.py)
compares the three theorem scopes on supplied asymmetric snapshots.

### Validate continuous transit to a protected basin

`Network.relational_transit_capture` can attempt a rigorous connection from
a supplied initial state to a protected basin. It computes bounds
for the ideal continuous ODE without advancing the stored network. Supply the
same exactly copied/reflected two-ring support, held unit capacities and
positive coefficients used by `relational_capture`, and select
`phase_domain="positive_resultant"`. Small symmetry defects are not repaired.
The initial snapshot may be outside the sufficient protected basin.

Using the graph and model prepared above, the call and exact export are:

```python
from fractions import Fraction
from tnfr.sdk import export_to_json, relational_report_to_dict

transit = Network(graph).relational_transit_capture(
    model=model,
    cycles=(tuple(range(5)), tuple(range(5, 10))),
    horizon=Fraction(1, 4),
    time_step=Fraction(1, 32),
    order=12,
    requested_sector=None,
)
print(transit.status, transit.validated_horizon, transit.target_sector)
export_to_json(relational_report_to_dict(transit), "relational-transit.json")
```

This illustrates the interface without asserting a successful transit for a
particular preparation or horizon. Times must be positive exact `Fraction`
or integer durations; floats and Booleans are rejected. They measure elapsed
structural time from the supplied state. Taylor order is a numerical policy
from 4 through 16, with default 12. The last proof interval is shortened to
the requested horizon; the routine does not retry with changed settings or
extend the declared time. At most 4096 declared proof steps are admitted.
`requested_sector=None` permits any certified positive, consensus or negative
basin. Omit the argument to retain the default positive-sector requirement,
or pass integer `1`, `0` or `-1` to require that specific limiting sector.
Booleans and floats are rejected. This argument changes admission criteria,
not the supplied dynamics; it is not a prediction of the resulting sector.

Every accepted interval must enclose the whole exact ODE segment in a regular
Picard tube. Interval Taylor remainders bound local truncation, and a Metzler
comparison propagates uncertainty from earlier intervals. Admission then
requires the whole final enclosure to lie in a matching protected rectangle
with storage strictly below `7*beta`. Checking only the endpoint center or a
sampled trajectory would not meet this contract.

Inspect `transit.admitted` and `transit.target_sector` for complete capture
admission. A certified consensus target is `0`; test admission explicitly
rather than interpreting the target as a Boolean. The report retains all
three `candidate_rectangle_margin_bounds`, the matching `rectangle_kind`
and selected `rectangle_margin_bounds`. The compatibility field
`positive_rectangle_margins` always describes the positive candidate.
Geometric margins alone do not imply complete admission, and unavailable
reports retain `target_sector=None`.

`transit.initial` retains the separate initial-state capture report;
its verdict need not equal the transit verdict. `initial_winding_zero` is
separate evidence needed when interpreting an admitted transit as entry from
winding zero. If a step cannot be validated, `steps`, `validated_horizon` and
`endpoint` retain the accepted prefix, while `failed_tube` and
`unavailable_reasons` describe the unresolved calculation. Completing the
horizon without passing the final sufficient gate also returns unavailable.
The analytic `atan(u)/u` enclosure handles zero through its existing local
series and can also handle intervals outside that series domain when they
exclude zero. This is a numerical enclosure extension of the same law, not
an added phase mechanism or an additional evolution step; unresolved bounds
remain unavailable.

The calculation supplies conditional ODE evidence, not a live engine
trajectory or a guarantee for future Euler steps. Its result does not revise
an original frozen experimental verdict. See the
[API contract](API_CONTRACTS.md#validated-conditional-relational-transit) and
[derivation](../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-validated-transit)
for the exact scope and retained evidence.

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

The result records the declaration, runtime provenance, realized initial and
final triad/support, and final diagnostics. `to_dict()` returns detached data;
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
`import_from_json`, also used for CLI recipe input, rejects duplicate decoded
keys at every depth, nonfinite numbers and nonzero fractional literals that
underflow to binary64 zero. Integer literals remain Python integers; ordinary
fractional literals retain binary64 rounding. Decoding neither authenticates
report provenance nor replaces `StudySpec` field validation.

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

`Network.nfr()` is a stored-state observation, not a certificate that an NFR
has formed. Its radial/annular/multinodal labels classify the unit-source
potential centrality profile under a configured policy; a uniform profile
does not establish a literal ring or rotational symmetry. Empty or unsupported
geometry returns `topology="unavailable"`, `topology_available=False` and an
explicit `topology_status`. Consumers must handle that availability state.
An all-zero centrality profile caused by numeric underflow reports
`centrality_below_represented_range`; it cannot establish annular geometry.

Its `coherence_length` now uses the shared tetrad estimator, with
`coherence_length_available` and `coherence_length_provenance`, instead of the
old untagged topology-only spectral proxy. A fitted length has structural
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
and coherence-length availability/provenance. Its historical phase labels are
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
same command. Its `benchmark_basic` compatibility call now returns the shared
benchmark schema (`total_numbers_tested`, `total_time_ms`, `cache_statistics`,
and related fields), replacing its former basic-versus-cached timing schema.
