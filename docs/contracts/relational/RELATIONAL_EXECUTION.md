# Native relational execution and observations

Argument-based field and finite-step execution, native tangents, prepared patterns and hypothetical support/state events.

Part of [Relational dynamics contract index](../RELATIONAL_DYNAMICS.md). Section links remain stable; hypotheses and model changes remain local to each result.

## Conditional relational execution

The opt-in owner [dynamics/relational.py](../../../src/tnfr/dynamics/relational.py)
implements `RelationalExchangeModel`, `evaluate_relational_exchange` and
`step_relational_exchange`. `Network.relational_exchange(model)` and
`Network.step_relational(model, dt=..., t=...)` are thin SDK delegates.
The [theory owner](../../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#capacity-separable-exchange)
states the joint-storage, independent-capacity and zero-phase-activity premises.
No runtime flag implicitly substitutes this model for operator execution.

The [resonance contract](SINE_RESPONSE_AND_MEMORY.md#sine-resonance) separately assesses the normalized-sine
comparison law. It does not interpret this native executor's spectrum or the
configured RA operator as that law's frequency response.

The [uniform-locality classification](../../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#primitive-locality-phase-clock)
fixes the same relative phase evolution under explicit graph-family and storage
premises, without assuming individual phase freezing. The only remaining term
is a state-independent common angular rate. This executor uses its zero-rate
representative and retains both rows frozen at zero capacity. Whole-system
rest or same-model capacity homogeneity are sufficient reference conditions;
clock-unit conversion alone is not. No supplied common clock is inferred.
Its ideal rates consume own capacity, incident primitive form/phase and fixed
model coefficients; total neighbor degrees and remote states are not row
inputs. This does not bypass whole-graph validation: an invalid remote node
still rejects evaluation. Global storage aggregation and backend-dependent
floating arithmetic remain separate from mathematical locality. The result
justifies the existing relative dynamics within that class; it adds no mode,
automatic law selector or new public API.

The [joint-storage classification](../../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#joint-storage-locality-classification)
also selects this executor's quadratic/cosine storage from a larger periodic
edge-cost class when e,w>0, uniform locality, capacity homogeneity and its exact
continuous loss are imposed across the full regular graph family. It does not
derive that loss from passivity, fix beta or extend the executor's phase domain.
Mixed candidate costs are rejected by mathematical work identities, not by
silently evaluating them with this executor's already selected phase row.

The [passive-loss comparison](../../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-passive-loss-completion)
adds `rho*N*g` only in its declared mathematical comparison law. It retains
the same equilibria and native form row but has additional phase loss and a
different response. `RelationalExchangeModel` exposes no `rho` option: this
executor and its work reports implement the exact-loss reference. Shared field
and tangent readers supply detached baseline geometry for the comparison;
their output is not an execution or certificate of the alternative. Existing
finite capture/recovery records retain their original law and bounds.
The [nonlinear passive countermodel](../../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-nonlinear-passive-completion)
likewise introduces no executor parameter. Its unchanged equilibrium tangent
and newly proved common local energy sublevel do not authenticate a reference
capture report for the alternative or transfer a finite transit enclosure.
Its comparison work residual can have either sign; `continuous_loss` in the
reference report remains the exact-loss reference quantity.

`storage_scale` is a required finite positive beta. Nonnegative EPI weight
and positive phase weight use the native coefficient normalization once;
`model.effective_weights` retains that normalized pair. Native staging and
detached sine capture revalidate authoritative stored coefficients through the
shared relational owner before field arithmetic or writes. Revalidation rejects
Boolean, nonfinite and out-of-domain values without normalizing again. Exact
rational proof coefficients retain their values; native outputs also require
binary64 representability. Reconstructing a model or report does not authenticate
its provenance or exempt its fields from admission. Graph-level pressure
mixes, custom pressure hooks, clipping rails, histories and automatic policies
are outside this selected model, and their configuration is preserved.
Active/unknown Gamma and a simultaneous independent-pressure extension reject
without invoking their callbacks; an overridden registry entry for `none`
cannot hide forcing.

The [phase-storage classification](../../../theory/TNFR_VARIATIONAL_PRINCIPLE.md#native-phase-storage-classification)
keeps the existing unit cosine-cost normalization. A mathematical choice
`U_phi=kappa*V_phi` with coefficient `beta_U` is represented by the single
`storage_scale=beta_U*kappa`; the API does not add a redundant cost multiplier.
Changing that product changes storage and phase rates while leaving the
specified form-pressure row unchanged. The positive metric at zero phase
pressure uses the existing continuous sinc limit, not an inferred ratio of
two zeros. None of these contracts identifies storage with physical energy.

The executor admits connected, simple, loopless, undirected
support with at least two nodes and unit conductance. Each node must supply
finite signed real scalar EPI, primitive phase and nonnegative capacity.
Uniform-real BEPI, materialized or in its canonical serialized mapping, is
admitted through the shared signed-scalar validator and, like the scalar nodal
solver, commits as its signed scalar. Richer, complex or malformed form rejects.
Raw Boolean/text values and nonzero inputs lost during materialization reject.
This admission does not include isolated-node evolution or a continuous ramp
from zero to unit conductance. Generic pressure has different channel support:
a zero-weight graph edge can still contribute phase, capacity and topology.
The [relation foundation](../../../theory/nodal/RELATION_FOUNDATIONS.md#zero-relation-boundary)
separates those cases from true absence; its weighted comparisons are not a
new execution mode of this API.
`model.phase_domain` selects one of three admission paths for the same joint law:

- `"acute"` is the default. Every represented edge gap in the shared signed
  `atan2(sin(delta),cos(delta))` chart must be strictly acute; its positive
  metric check is numerical, not an exact transcendental certificate.
  The same chart admits the whole Euler segment. Reduction modulo binary64
  `2*pi` is not used: at large raw lifts it can disagree substantially with
  the trigonometric phase used by the law. Raw phases remain in the report.
- `"positive_resultant"` requires a certified positive real part of every
  relative neighbor resultant `z_i = sum_j exp(i*(theta_j-theta_i))`. The
  [shared rational enclosure](../../../src/tnfr/mathematics/_phase_resultant_chamber.py)
  computes lower bounds `L_i <= Re(z_i)` from exact represented raw radians,
  using the existing mathematical-pi enclosure and a bounded cosine series.
  All `L_i` must be positive; the materialized resultant real part and metric
  must also pass their positivity checks. Individual obtuse or antipodal edges
  can be admitted when their full neighborhood meets these conditions.
- `"regular"` admits relative resultants in the principal-argument domain
  $\mathbb{C}\setminus(-\infty,0]$. The same enclosure owner computes exact rational
  rectangles `([C_lower,C_upper],[S_lower,S_upper])` for each mathematical
  cosine/sine sum at the exact represented raw phases. The quantity
  `m_i=max(0,C_lower,S_lower,-S_upper)` lower-bounds distance to the excluded
  nonpositive-real ray. Every `m_i` must be strictly positive. Negative real
  parts are allowed when the imaginary part is separated from zero. Zero
  resultants and negative-real branch-cut resultants remain excluded.

The second path is a sufficient right-half-plane chamber, not the entire
regular slit-plane domain. The third path targets that larger domain but its
finite-work certificates remain sufficient tests: an unresolved rectangle or
nonpositive lower margin need not prove an actual singularity or branch cut.
Work limits and outward rounding are numerical policies, not physical
thresholds. The options keep the same specified continuous pressure and phase
laws and do not alter default runtime dispatch. A regular snapshot or admitted
proposal does not establish all-future regularity, and it does not transfer
acute stability or winding-protection theorems to a nonacute state.

Regular execution also checks its materialized cosine/sine sums against any
certified component signs and requires a represented argument strictly between
`-pi` and `pi`. An unresolved represented branch, nonpositive metric or lost
nonzero derived value rejects evaluation. With `a=atan2(S,C)`, the regular
metric is evaluated as `H=pi*S/a` when `S` is nonzero, with `H=pi*C` at
`S=a=0,C>0`. This uses the captured sine sum directly rather than evaluating
`sin(a)` again near the branch cut. It is the same ideal metric
`pi*|z|*sinc(a)`, with a numerical realization suitable for this domain.

All three relational modes use the private shared pressure adapter in
[the pressure owner](../../../src/tnfr/dynamics/dnfr.py). It combines `g=a/pi`
from each mode's captured relative sums with the existing stable EPI
neighbor reduction and configured two-channel weights. It checks live node,
phase and support order, finite sources in `(-1,1)` and representable pressure
before writing its detached state. It does not reconstruct the phase source
by subtracting a separately rounded global neighbor angle. Field
`pressure_path="relative_resultant_canonical"` identifies this realization;
a committed relational step records the same value in `_DNFR_META.hook` and
`_dnfr_hook_name`. The ordinary default, fused and phase-only pressure paths
keep their existing contracts. A private supplied-source call does not itself
authenticate the caller's derivation of the sources. Reusing one captured
source removes a second global-angle reconstruction from the relational
field; pressure and phase work still retain their actual rounding residuals.
This numerical correction does not re-evaluate or reinterpret frozen response
artifacts produced through earlier arithmetic paths.

Evaluation uses a detached preparation and actual native pressure, with no
live graph mutation. Its immutable `RelationalExchangeField` retains node
order, edges, state, source, metric, both rates and execution path. Storage,
continuous loss, actual work, balance residual and pressure/product defects
use exact fractions of represented values. Tiny exact derived storage is
retained even when it has no nonzero binary64 display. `phase_storage` is the
unscaled `V_phi`; `storage` includes its beta factor.
For `"positive_resultant"`, `resultant_real_lower_bounds` retains the exact
rational `L_i` values in field node order. For `"regular"`, `resultant_bounds`
retains the component rectangles and
`resultant_regular_margin_lower_bounds` retains the corresponding `m_i`.
Unused domain evidence remains `None`; all three are `None` in acute mode.
These explicit enclosures differ from exact arithmetic on rounded storage
or work values: they certify mathematical trigonometric sums and domain
separation, not errors in the separately computed native pressure, phase
metric or rates. Measured pressure-split, rate and work residuals are retained
in every domain. The model and `scope` retain the selected chamber and
enclosure-method provenance. [Regular pressure controls](../../../tests/test_relational_regular_pressure.py)
cover captured-source coherence, branch preservation and prewrite rejection.

### Conditional and fast-mediator observations

The [nonlinear fast-mediator limit](../../../theory/nodal/RELATIONAL_MEDIATOR_DYNAMICS.md#fast-mediator-reduction)
has a separate continuous approximation contract. Its instantaneous reduced
field is observed by evaluating a detached full graph with the mediator at
the declared local midpoint, then selecting the visible rows. This does not
make midpoint replacement an exact finite-capacity step or a supported
ten-node direct-edge execution mode. The proof retains an initial transient,
a regular neighborhood and a fixed finite horizon; it supplies no fixed-step
Euler error guarantee as mediator capacity grows. Existing field/work reports
suffice for its static controls; no reduced solver or infinite capacity is
admitted by this API.

The [two-intermediary and series result](../../../theory/nodal/RELATIONAL_MEDIATOR_DYNAMICS.md#two-mediator-composition)
uses the same detached evaluation after the declared path reconstruction.
Keep its selected phase lift and original port degrees. Replacing a reduced
segment by an ordinary unit edge, or choosing the principal endpoint phase
difference without retaining the path's phase information, can change the
response. Stationary composition is separate from the error bounds for a
simultaneous fast limit; no sequential projection is a finite-capacity step.

The [three-port reduction](../../../theory/nodal/RELATIONAL_MEDIATOR_DYNAMICS.md#three-port-collective-interaction)
uses a detached star reconstruction: mediator form is the mean of its three
port forms, and mediator phase is the argument of their collective resultant.
Admit the full reconstructed graph, including every acute fine edge, before
selecting visible rows. A zero resultant does not define a mediator phase;
nonzero resultant alone does not certify the acute domain. Retain original
port degrees and internal neighbor resultants. The inherited storage and
field generally require all three ports together; independent pair interfaces
cannot replace them on an open neighborhood. The separate local fast bound
retains the moving circular mean and hidden initial state. Reconstruction is
an observation preparation, not an instantaneous reset of a live mediator.

### Uniform-form tangent observation

`evaluate_relational_uniform_tangent(graph, model=...)` and the thin SDK
delegate `Network.relational_uniform_tangent(model)` reuse detached field
admission. They require exactly uniform represented EPI, with no equilibrium
tolerance. Nonzero phase pressure and form rates remain visible in the retained
`field`; uniform form alone does not establish a critical phase geometry.

`RelationalUniformTangent.generator` follows `(all form, all phase)` in field
node order. It differentiates the declared smooth law with held support,
capacities and model coefficients. The phase-source derivative uses the Arg
formula with the captured relative resultants. Since `Bx=0`, derivatives of
the phase metric contribute no phase-to-phase block at this state. This is a
materialized ideal derivative, not the derivative of floating-point rounding
or a certified spectral enclosure. Nonfinite or lost nonzero coefficients
are rejected using shared admission.

`phase_source_jacobian`, exact represented `common_offset_residuals` and the
derived `phase_source_row_sum_residuals` retain numerical defects rather than
projecting them to zero. The shared `relational_report_to_dict` exporter admits
this report and validates its field's node labels. It exports stored dataclass
fields; the derived row-sum property can be recomputed from the stored Jacobian.
No eigenmode classification, event selection, source, time step or live graph
mutation is part of this observer. Its use in a
[collective pulse study](../../../theory/nodal/RELATIONAL_RETURN_PATH_GEOMETRY.md#shared-collective-pulse)
keeps local modal evidence distinct from a maintained nonlinear oscillation.

At an admitted uniform-form state, multiplying `generator` by the concatenated
`field.form_rate + field.phase_rate` evaluates the ideal law's local second
time derivative, with the same materialization limits. In the
[regular source/receiver audit](../../../theory/nodal/RELATIONAL_NATIVE_FORMATION.md#regular-seeded-reachability-audit),
this distinguishes zero initial phase velocity from nonzero phase acceleration.
Independent rational bounds certify that preparation's derivative signs; the
matrix product alone does not bound a Taylor remainder or a future response.

<a id="phase-consensus-tangent-observation"></a>
### Phase-consensus tangent observation

`evaluate_relational_consensus_tangent(graph, model=...)` and
`Network.relational_consensus_tangent(model)` admit exactly equal captured
raw phases with arbitrary signed form. `RelationalConsensusTangent` retains
the same field, generator, phase-source Jacobian and represented offset
residuals as the uniform-form reader. Both use one shared derivative builder
after their own exact admission. The older reader still requires uniform EPI.

At phase consensus, the first phase derivative of the reciprocal metric
vanishes even when `L*x` is nonzero. With `K=diag(nu_i/d_i)` and the
combinatorial Laplacian `L`, the ideal blocks are
`(-e*K*L, -(w/pi)*K*L; (w/(beta*pi))*K*L, 0)`. The
[derivation and full-state comparison](../../../theory/nodal/RELATIONAL_RESPONSE_IDENTIFICATION.md#native-consensus-full-state-tangent)
retain arbitrary positive capacity for spectral claims. The reader itself
also admits zero capacities, under the same native field contract.

The actual source can have nonzero form and phase rates. This is its local
Jacobian, not a claim that it is an equilibrium or that the nonlinear flow
keeps a constant generator as phases separate. Entries are materialized
derivatives with the existing finite-arithmetic checks, not outward spectral
or trajectory enclosures. There is no eigenvalue classifier, time step,
formation verdict, forcing installation or graph mutation. Exact export
preserves source order and residuals through the shared SDK report writer.

## Native relational stepping and pattern reports

The following step and graph observers consume the native Arg field from
`dynamics/relational.py`, including its selected phase domain.

### Finite relational steps

A step requires positive `dt` and a finite clock advance. It uses explicit
`t`, or graph `_t` with initial default zero. Both Euler rows use the same
initial snapshot and the shared nodal/Euler arithmetic. In the default mode,
the entire proposed phase segment must remain in its initial acute relative
lift. In the positive-resultant mode, let `h_i` be the exact rational difference
between the represented phase endpoint and its initial represented value.
The sufficient whole-segment condition is
`L_i - sum_j abs(h_j-h_i) > 0`, using the unit Lipschitz bound for cosine.
`segment_resultant_real_lower_bounds` retains these exact margins in node order
only in positive-resultant mode.

Regular mode uses the same exact increment differences and requires
`m_i - sum_j abs(h_j-h_i) > 0`. The complex exponential and distance to a
closed set both have unit Lipschitz bounds, so these margins separate every
resultant along the straight phase chord from the excluded ray.
`segment_resultant_regular_margin_lower_bounds` retains these exact margins
only in regular mode. Unused segment evidence is `None`; both fields are
`None` in acute mode. These sufficient tests can reject a chord that is in
fact regular. They certify the straight represented-endpoint proposal,
including endpoint-rounding effects, not the continuous ODE solution or its
numerical error. Endpoint state and pressure are separately refreshed and
admitted before commit. Endpoint admission alone cannot skip a chart boundary.
No finite solver defect is silently converted into a source.

All admission, arithmetic, endpoint checks and graph-owned cache hooks are
staged before commit to ordinary NetworkX attribute dictionaries. Rejection
leaves live state and metadata unchanged. Success writes form, phase, fresh
endpoint pressure and model rate, advances `_t`, updates pressure/trigonometric
cache metadata, removes stale second-derivative aliases and retains the
immutable report at `_relational_exchange`. Existing operator histories are
not extended; they may consequently be stale for a history-consuming operator.
This is neither a concurrent-access transaction nor an entire-loop rollback.

`RelationalExchangeStep` retains `before`, `after`, `epi_update_defect`,
`phase_update_defect`, `clock_defect` and actual `energy_change`.
`energy_step_defect = energy_change - dt * before.storage_rate` measures the
departure from the initial represented work. It is not a bound on ODE error.
An energy increase is reported honestly; successful admission is not a
monotonicity certificate. Repeat steps explicitly and inspect their evidence.
No physical clock, autonomous pattern formation or universal-law selection is
inferred. [Routine controls](../../../tests/test_relational_exchange_execution.py)
exercise this owner and SDK delegation; [usage](../../guides/relational/RELATIONAL_EXECUTION.md#execute-the-conditional-relational-model)
provides a minimal preparation.

The [local-recovery theorem](../../../theory/nodal/RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-local-recovery)
requires strictly positive held capacities, positive EPI dissipation and a
strictly acute phase equilibrium. It concerns the continuous law near a
prepared geometry, modulo common form and phase offsets. The executor also
admits zero capacity and zero EPI weight, where that recovery result need not
hold. Neither admission nor a small instantaneous work residual certifies
recovery for an arbitrary Euler step or a complete runtime with other events.

The [paired-region result](../../../theory/nodal/RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-region-interaction)
uses the same law and shared regional support accounting. Its bridge and phase
sectors are prepared. Transmitted deformation does not derive a new interaction
force, autonomous formation or a closed evolution for regional means/storage.

<a id="relational-pattern-observation"></a>

### Prepared relational pattern observations

[`observe_relational_pattern`](../../../src/tnfr/physics/relational_observations.py)
and `Network.relational_pattern(model, *, reference_phase, regions, cycles=())`
evaluate one fresh detached relational field and reuse it for the supplied
observations. They do not mutate the graph, select regions, infer a reference
equilibrium or certify a temporal recovery result. The report's `field`
retains the same model, node order, native pressure and numerical defects as
the execution owner.

`reference_phase` maps every graph node to its explicitly supplied real phase
lift; the report retains it in field node order. Each ordered region retains
its nodes, exact represented form mean, phase-error mean, centered form and
centered phase error. Regions must be nonempty and contain distinct existing
nodes; different regions may overlap and need not partition the graph.
Differences use the supplied lifts without independent
wrapping or an inferred history. Common reference-frame offsets are distinct
from internal deformation; arbitrary per-node `2*pi` changes can alter these
lifted observations. Their separate squared norms use the declared form/phase
coordinates and are not combined into a physical distance.

Supplied oriented `cycles` delegate to the shared winding observer. Regional
`transport` delegates to `observe_regional_support_balance`, using the phase
source `w*g` independently of the form rate and retaining native pressure
split defects. Any zero capacity in the full graph, zero EPI weight or a
whole-support region makes transport unavailable, checked in that order and
reported through `transport_unavailable_reason`;
other available pattern observations remain retained. Neither the weighted
form accounting nor a winding integer closes future regional dynamics.

The fresh field's `work` is a `RelationalWorkBalance` aligned with `field.nodes`.
Its rational `form_gradient` retains the exact represented `q=Bx`, before the
older floating `field.form_gradient` is rounded. It separates nonnegative
`dissipation`, signed `exchange` (positive toward form storage), `form_work`,
`phase_work`, `form_residual`, `phase_residual` and their `balance_residual`.
The existing global loss, storage rate and balance residual sum these same
contributions. They are gradient-work contributions to total storage, not
derivatives of an independently assigned nodal energy.

Each `region.work` sums those contributions over its supplied nodes. Only a
partition sums to the global quantities; overlapping regions double count
shared nodes. `region.boundary` uses a shared `RegionalSupportCut`, projected
from admitted `transport` or obtained through `observe_regional_support_cut`.
That cut observer revalidates captured primitives and permits full support,
zero capacities and zero EPI weight without dividing by any of them. Existing
`RegionalSupportBalance.cut` is a read-only projection of its retained data,
not a separate provenance check. All cut indices refer to the full node order.

Writing its outward current as `Q_R`, the boundary report retains:

| Report quantity | Exact arithmetic on the represented field |
| --- | --- |
| `form_weighted_rate` | `sum_R (d_i/nu_i)*field.form_rate_i` |
| `form_boundary_rate` | `-e*Q_R` |
| `form_source_rate` | `w*sum_R d_i*g_i` |
| `form_pressure_defect_rate` | `sum_R d_i*pressure_split_residual_i` |
| `form_rounding_defect_rate` | `sum_R (d_i/nu_i)*nodal_rate_rounding_defect_i` |
| `form_identity_residual` | Actual weighted form rate minus the four terms above; exact zero |
| `phase_weighted_rate` | `sum_R (H_i/nu_i)*field.phase_rate_i` |
| `phase_boundary_rate` | `(w/beta)*Q_R` |
| `phase_rate_residual` | Actual weighted phase rate minus its boundary term; retained, not assigned zero |

Here `d_i` is the full-graph degree and `H_i` the captured phase metric.
The phase rate is not the derivative of a weighted phase total because `H`
changes. Unlike legacy `transport`, these balances admit `e=0`, full-support
regions (empty cut), and zero capacity outside the region. A zero capacity
inside it sets `weighted_rate_unavailable_reason="zero_capacity_in_region"`;
both weighted rates, form rounding/identity residuals and phase rate residual
are then `None`. Nodal/regional work, cut and undivided model terms remain
available. No division through zero or inference of equilibrium is performed.
The [derivation and scope](../../../theory/nodal/RELATIONAL_RECOVERY_AND_INTERACTION.md#relational-work-integration)
distinguish these represented balances from exact-real storage identities.

The fresh field additionally retains exact rational `phase_mobility`,
`a_i=nu_i/H_i`, and `phase_rate_rounding_defect`, the represented phase rate
minus `(w/beta)*a_i*q_i`. These use the captured positive metric and exact
`work.form_gradient`; the older floating gradient can lose information.

`region.phase_response` is a `RegionalPhaseResponse` computed from this same
field and cut. For `n` selected nodes and `k=w/beta`, it retains:

| Report quantity | Exact arithmetic on the represented field |
| --- | --- |
| `mean_mobility`, `mean_form_gradient` | Arithmetic regional means of `a` and `q` |
| `mobility_variance`, `form_gradient_variance`, `mobility_gradient_covariance` | Population variances and covariance, with denominator `n` |
| `mean_mobility_boundary_rate` | `k*mean_mobility*Q_R` |
| `covariance_rate` | `k*n*Cov_R(a,q)` |
| `covariance_rate_squared_bound` | `k^2*n^2*Var_R(a)*Var_R(q)` |
| `model_total_rate` | `k*sum_R a_i*q_i`; equals the two contributions above |
| `rounding_residual` | Sum of captured `phase_rate_rounding_defect` |
| `total_rate`, `mean_rate` | Sum and arithmetic mean of actual captured phase rates |
| `identity_residual` | Actual total minus boundary, covariance and rounding terms; exact zero |

The squared bound bounds `covariance_rate**2`, without a tolerance or square
root. This unweighted rate admits zero capacity, a singleton, full support
and zero EPI weight under the field's existing admission. It is a model-rate
observation, not a measured time derivative. A zero cut need not imply zero
response. Covariance includes boundary mobility variation on a proper region;
it need not arise solely from internal edges or from phase geometry when
capacity is heterogeneous. Partitioned totals add, but their mean-mobility
boundary terms need not cancel. See the
[nonlinear response theorem](../../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#regional-phase-mobility-balance).

The actual engine always supplies field `work`, `phase_mobility` and
`phase_rate_rounding_defect`, and regional `work`, `boundary` and
`phase_response`. Optional `None` defaults preserve earlier manually
constructed report records. These additional observations alter no evolution
law and certify no autonomous regional closure or monotone clock.
The [state-and-rate counterexample](../../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#state-rate-predictivity)
shows why: a ten-coordinate coarse projection and its predicted rate can agree
while coarse acceleration differs. This does not identify the full reports:
their retained centered form vectors distinguish the omitted internal
contrast. No acceleration observer or reduced evolution law is implied by
the existing snapshot interface.
The separate [joint memory theorem](../../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md)
uses those full coordinates to justify a prepared-even approximation with
conditional finite-horizon error bounds. Its neighborhood and derivative
constants are not part of an executable admission certificate; the SDK still
executes and reports the full nonlinear state.

<a id="relational-report-export"></a>

`tnfr.sdk.relational_report_to_dict(report)` projects native relational reports
and explicitly registered owner-managed reports. The
[shared dispatcher](../../../src/tnfr/sdk/relational_reports.py) owns the accepted
types; unsupported values reject. Each report's subsection states its scientific
scope. Having a `to_dict()` method alone does not register an arbitrary object.
Known-type dispatch resolves only the relevant registered owner, rather than
importing unrelated research report owners. It checks the actual registered
class identity and retains support for subclasses inheriting that registered
base; a matching class name or module string alone is insufficient. This is
dispatch admission, not re-admission of the scientific fields.
The detached `SineClassPortReadout` and `SineClassStorageReadout` reports use
the same export path as the existing formation, maintenance and composition
reports, with no new live-network execution wrapper.
Its detached JSON-compatible envelope has
`schema="tnfr.relational-report.v1"`, `report_type` and recursively projected
`report` fields. Exact fractions use
`{"numerator": ..., "denominator": ...}` records; tuples become ordered arrays,
including tuple node labels. JSON scalar labels are supported; opaque objects
are rejected, including labels in nested supports, regions and cuts, rather
than replaced with `repr` or serialized as dataclass contents. Save the mapping with
`export_to_json(relational_report_to_dict(report), path)`.

This projection is not the `StudyResult` schema, a restorable checkpoint, a
source fingerprint or authentication of a caller-supplied report. Node-type
reconstruction is not promised. Export does not recompute theorem eligibility,
validate acquisition or regenerate a producer; report-consuming scientific
calculations retain their own primitive and evidence reconstruction contracts.
It supplies no new CLI execution mode.
[Usage](../../guides/relational/RELATIONAL_EXECUTION.md#observe-a-prepared-relational-pattern) belongs to the
shared SDK guide; the [formed-object workflow](../../guides/relational/SINE_PATTERNS.md#formed-object-sdk-workflow)
includes an exact zero-horizon export example.

<a id="relational-attachment-observation"></a>
### Supplied relational attachment observation

The shared attachment observer in
[relational observations](../../../src/tnfr/physics/relational_observations.py)
and the SDK delegate compare two disjoint, separately admitted connected
components with one supplied ordered unit bridge. The model must select the
same phase domain for both components and the joined field: `"acute"`,
`"positive_resultant"` or `"regular"`. Every field is admitted independently;
admitting two components does not establish admission of their joined support.
Component forcing, invalid primitive data, nonunit edges, overlapping labels
or invalid cross-component endpoints reject the comparison.

The report retains both complete component fields, the fresh joined field and
the two before/after port cards. Each card retains primitive form/phase/capacity,
degree, exact represented form gradient, relative cosine/sine resultant,
pressure, metric and both rates. The resultant is captured from the sums
already used by the engine; no second pressure implementation is introduced.
The new field observation has a compatibility default of None for older
manually constructed reports; current native evaluations always populate it.

Rational pressure, metric and rate differences follow the joined node order
and subtract the matching component row. Storage changes subtract both
component totals; phase storage remains the unscaled cosine cost, while total
storage includes the model's beta. Shared support-reset accounting supplies
the form-energy identity, and the cut is outward from the full left component.
Zero capacity retains the native frozen-row semantics.

The derived property `continuous_loss_change` subtracts the component loss
rates from the joined rate. It is not event work: a rate change cannot fund an
instantaneous storage jump. `represented_zero_supply_passive` tests whether
the captured `storage_change` is nonpositive. For the ideal state-preserving
unit-edge addition, the required supply is
`(x_a-x_b)**2/2 + beta*(1-cos(theta_b-theta_a))`; the EPI dissipation coefficient
does not multiply this storage increment.

`attachment.assess_supply(supplied_work)` checks the **additional** event-passivity
premise `storage_change <= supplied_work` against the captured represented
storage. Work is required explicitly and has the same structural-storage units.
It may be signed: positive supplies storage and negative extracts it. Exact
rational inputs remain exact, including values too small for binary64; other
real inputs use shared represented-real admission. Booleans, nonfinite values
and non-real inputs reject. No graph is read, refreshed or modified.

The frozen `RelationalAttachmentSupplyAssessment` retains `required_supply`
(the existing storage change), `supplied_work`, `supply_margin` and
`represented_balance_satisfied`, with explicit scope. The common relational
exporter accepts this assessment and retains exact fractions; attachment exports
also include the two derived properties. The caller's work is not authenticated,
and arithmetic on a publicly constructed report is not proof of its provenance.
A represented balance is not an exact-real trigonometric passivity certificate.
Even a zero-cost permitted attachment can change rates; neither that balance nor
the continuous loss selects an occurrence time or requires an event.

Both live graphs remain untouched. The disconnected union is used only by
the support-transport observer, never by the connected relational evaluator.
This is a hypothetical support comparison, not an executed event, a selector,
a recovery certificate or a closed coarse-state model. Exact arithmetic on
captured binary64 values is not a bound on ideal trigonometric evaluation.
The [interface derivation](../../../theory/nodal/RELATIONAL_SUPPORT_EVENTS.md#one-bridge-interface-admission)
owns the result; [SDK usage](../../guides/relational/RELATIONAL_EXECUTION.md#compare-a-supplied-connection)
owns the call and report field names. The common relational exporter also
accepts this report, validating labels in its nested component/joined fields,
bridge and port cards and retaining detached exact differences.

<a id="relational-relocation-observation"></a>
### Supplied relational bridge relocation

`observe_relational_relocation(graph, *, model, remove_bridge, add_bridge)`
in the same [observation owner](../../../src/tnfr/physics/relational_observations.py)
and `Network.relational_relocation(model, *, remove_bridge, add_bridge)` compare
one supplied support exchange without changing primitive form, phase or held
capacity. The original graph must pass the native connected, simple, unit-edge,
unforced admission in the selected phase domain. Removing the supplied existing
edge must leave exactly two connected components, each containing at least two
nodes. The first old endpoint identifies the first component. The new ordered endpoints must
cross those components in the same order, and the new edge must be absent from
the original support. No-op replacements, internal new edges and removal of a
non-bridge or leaf bridge reject. The final connected field passes the same
selected phase-domain admission independently. Nonnegative capacities,
including zero, retain their normal execution meaning.

The frozen `RelationalRelocationObservation` retains:

- `before` and `after`: both fresh complete fields in the original node order.
- `components`: the two ordered node partitions after removal, **not** component
  fields; no relational evaluation runs on the disconnected intermediate graph.
- `remove_bridge`, `add_bridge` and `ports`: supplied edges and before/after
  cards for their unique endpoints in field order. The existing
  `RelationalAttachmentPort` card is reused, including for a shared endpoint.
- Exact represented pressure, metric and rate differences, form/phase/total
  storage changes, and the shared form-only `transport_reset`.
- `cut_before` and `cut_after`, both directed outward from the first component.

All internal edges and the primitive state are retained. This preserves existing
internal cycles and their instantaneous phase data; it does not establish their
future winding, attraction or physical identity. The generic graph observer is
distinct from the conditional two-C5 recovery argument in the
[relocation theorem](../../../theory/nodal/RELATIONAL_SUPPORT_EVENTS.md#identity-preserving-bridge-relocation).

The attachment and relocation reports share `assess_supply(supplied_work)` and
`represented_zero_supply_passive`. `continuous_loss_change` is the new minus
old loss rate, which supplies no instantaneous event work. The reused
`RelationalAttachmentSupplyAssessment.required_supply` is a **signed lower
bound on net supplied work**, namely the captured `storage_change`; it can be
negative when relocation releases storage. Declared extraction is admitted by
the budget exactly when `supplied_work >= storage_change`. This comparison
does not authenticate a supply, select a support event or derive its time.

The common exporter retains both fields, both cuts, the node partitions and
exact changes with the same label checks and rational representation. All
live attributes, histories and support remain unchanged. Floating phase
evaluation supplies represented evidence, not an ideal trigonometric enclosure
or an executable recovery certificate. See
[usage](../../guides/relational/RELATIONAL_EXECUTION.md#compare-a-supplied-bridge-relocation).

<a id="relational-reset-observation"></a>
### Supplied joint state and support reset

`observe_relational_reset(before_graph, after_graph, *, storage_scale)` and
`Network.relational_reset(after, *, storage_scale)` compare supplied endpoints
without mutating either graph. They admit the same nonempty ordered nodes,
simple undirected loopless support, symmetric nonnegative conductances and
explicit finite phase at every node. Shared transport admission validates
signed scalar form, nonnegative capacity and stored pressure. It retains its
zero defaults when capacity or stored pressure is absent; these defaults are
not evidence of explicit or measured zero values. Disconnected
support, isolates and zero/nonunit weights are allowed **for observation**;
neither endpoint is thereby admitted to the relational evolution law.

The positive finite `storage_scale` is beta in `S=E_D+beta*V`. Form storage
uses conductance, while phase storage uses every support edge, including zero
conductance, consistently with the native phase neighborhood. The shared
half-sine evaluation avoids cancellation in small phase costs. Its signed
atan2 chart matches the acute native field, including large raw lifts;
binary64 `2*pi` is not treated as an exact trigonometric period. Raw phase
subtraction must still materialize finitely before wrapping. Reports use exact
fractions of represented primitives, not ideal trigonometric enclosures.

`RelationalResetObservation` retains:

- `before`, `after`: detached transport snapshots, including capacity and stored
  pressure with the inherited defaults above, and actual conductance;
  `phase_before/after`, `edges_before/after`.
- `form_state_change`, `phase_state_change`: changes on the old support.
- `form_support_change`, `phase_support_change`: changes of support at the
  new nodal state. `transport_reset.before` is this algebraic intermediate,
  not another observed event.
- `form_storage_change`, `phase_storage_change`, `phase_storage_before/after`,
  `storage_before/after`, `storage_change` and `identity_residual`. Phase terms
  are unscaled `V`; total storage applies beta. The exact decomposition is
  `Delta S = Delta E_state + Delta E_support + beta*(Delta V_state + Delta V_support)`.

The same budget mixin as attachment/relocation provides
`represented_zero_supply_passive` and `assess_supply(supplied_work)`. The latter
retains the signed `required_supply=storage_change` without clipping. These
compare a declared work budget; they do not infer a reservoir, authenticate
an actual event or assign an activation time. A sequence's net storage decline
does not prove each constituent event was separately passive. Birth, deletion
or relabeling of nodes lies outside this same-node interface.

`relational_report_to_dict` retains these fields and exact fractions through
the existing detached exporter; unsupported opaque node labels reject. See
the [joint reset guide](../../guides/relational/RELATIONAL_EXECUTION.md#compare-a-joint-state-and-support-reset),
[theory](../../../theory/nodal/RELATIONAL_SUPPORT_EVENTS.md#nodal-reorganization-and-contact)
and [actual operator controls](../../../tests/physics/test_coupling_attachment_budget.py).
