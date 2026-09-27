# Relational dynamics API contracts

This chapter owns the admission, execution and report contracts of the selected
relational model. The linked source modules own their respective APIs;
mathematical hypotheses and proofs remain with the linked theory owners.
See the [common contract hub](../API_CONTRACTS.md) for shared scalar/solver
boundaries and the [regional and relational guide](../guides/REGIONAL_AND_RELATIONAL.md)
for executable SDK examples.

## Conditional relational execution

The opt-in owner [dynamics/relational.py](../../src/tnfr/dynamics/relational.py)
implements `RelationalExchangeModel`, `evaluate_relational_exchange` and
`step_relational_exchange`. `Network.relational_exchange(model)` and
`Network.step_relational(model, dt=..., t=...)` are thin SDK delegates.
The [theory owner](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#capacity-separable-exchange)
states the joint-storage, independent-capacity and zero-phase-activity premises.
No runtime flag implicitly substitutes this model for operator execution.

`storage_scale` is a required finite positive beta. Nonnegative EPI weight
and positive phase weight use the native coefficient normalization once;
`model.effective_weights` retains that normalized pair. Graph-level pressure
mixes, custom pressure hooks, clipping rails, histories and automatic policies
are outside this selected model, and their configuration is preserved.
Active/unknown Gamma and a simultaneous independent-pressure extension reject
without invoking their callbacks; an overridden registry entry for `none`
cannot hide forcing.

The executor admits connected, simple, loopless, undirected
support with at least two nodes and unit conductance. Each node must supply
finite signed real scalar EPI, primitive phase and nonnegative capacity.
Materialized uniform-real BEPI is admitted and, like the scalar nodal solver,
commits as its signed scalar; richer or unmaterialized serialized form rejects.
Raw Boolean/text values and nonzero inputs lost during materialization reject.
`model.phase_domain` selects one of two admission paths for the same joint law:

- `"acute"` is the default and preserves the existing numerical path. Every
  represented wrapped edge gap must be strictly acute; its positive metric
  check is numerical, not an exact transcendental certificate.
- `"positive_resultant"` requires a certified positive real part of every
  relative neighbor resultant `z_i = sum_j exp(i*(theta_j-theta_i))`. The
  [shared rational enclosure](../../src/tnfr/mathematics/_phase_resultant_chamber.py)
  computes lower bounds `L_i <= Re(z_i)` from exact represented raw radians,
  using the existing mathematical-pi enclosure and a bounded cosine series.
  All `L_i` must be positive; the materialized resultant real part and metric
  must also pass their positivity checks. Individual obtuse or antipodal edges
  can be admitted when their full neighborhood meets these conditions.

The second path is a sufficient right-half-plane chamber, not the entire
regular slit-plane domain. A nonpositive enclosure can be inconclusive; it
does not prove a singular or nonpositive actual resultant. Work limits and
outward rounding are numerical policies, not physical thresholds. Neither
option changes the pressure law, supplies a new evolution law or implicitly
alters default runtime dispatch.

Evaluation uses a detached preparation and actual native pressure, with no
live graph mutation. Its immutable `RelationalExchangeField` retains node
order, edges, state, source, metric, both rates and execution path. Storage,
continuous loss, actual work, balance residual and pressure/product defects
use exact fractions of represented values. Tiny exact derived storage is
retained even when it has no nonzero binary64 display. `phase_storage` is the
unscaled `V_phi`; `storage` includes its beta factor.
For `"positive_resultant"`, `resultant_real_lower_bounds` retains the exact
rational `L_i` values in field node order. It is `None` for `"acute"`.
These explicit enclosures differ from exact arithmetic on rounded storage
or work values: they certify mathematical cosine sums, not errors in the
separately computed native pressure, phase metric or rates. The model and
`scope` retain the selected chamber and enclosure-method provenance.

A step requires positive `dt` and a finite clock advance. It uses explicit
`t`, or graph `_t` with initial default zero. Both Euler rows use the same
initial snapshot and the shared nodal/Euler arithmetic. In the default mode,
the entire proposed phase segment must remain in its initial acute relative
lift. In the positive-resultant mode, let `h_i` be the exact rational difference
between the represented phase endpoint and its initial represented value.
The sufficient whole-segment condition is
`L_i - sum_j abs(h_j-h_i) > 0`, using the unit Lipschitz bound for cosine.
`segment_resultant_real_lower_bounds` retains these exact margins in node order
and is `None` in acute mode. This certifies the straight represented-endpoint
proposal chord, including endpoint-rounding effects; it is not an enclosure
of the continuous ODE solution or a bound on solver error. Endpoint state and
pressure are separately refreshed and admitted before commit. Endpoint
admission alone cannot skip a chart boundary. No finite solver defect is
silently converted into a source.

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
inferred. [Routine controls](../../tests/test_relational_exchange_execution.py)
exercise this owner and SDK delegation; [usage](../guides/REGIONAL_AND_RELATIONAL.md#execute-the-conditional-relational-model)
provides a minimal preparation.

The [local-recovery theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-local-recovery)
requires strictly positive held capacities, positive EPI dissipation and a
strictly acute phase equilibrium. It concerns the continuous law near a
prepared geometry, modulo common form and phase offsets. The executor also
admits zero capacity and zero EPI weight, where that recovery result need not
hold. Neither admission nor a small instantaneous work residual certifies
recovery for an arbitrary Euler step or a complete runtime with other events.

The [paired-region result](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-region-interaction)
uses the same law and shared regional support accounting. Its bridge and phase
sectors are prepared. Transmitted deformation does not derive a new interaction
force, autonomous formation or a closed evolution for regional means/storage.

### Exact linear observations of a supplied generator

`tnfr.mathematics.linear_observation.derive_linear_observation(J, O, *,
max_rank_calls=4096)` returns an immutable minimal row-space realization for
`z'=Jz`, `y=Oz`. It admits square finite exact/represented real `J` and ordered
nonempty output rows, including dependent and zero rows. Rational values
remain exact; floating values define their represented rational model.
Its `observation`, `right_inverse`, `reduced_generator` and `output_map`
verify `CT=I`, `CJ=GC`, `O=DC`, with explicit rank progression and a strict
rank-call budget. A zero output has no state; budget exhaustion returns no
partial certificate. This is not graph, stability, nonlinear-law or exact
transcendental admission. The old affine diffusion API retains its independent
model checks and negative generator sign; see the
[composition scope](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md).

<a id="relational-pattern-observation"></a>

### Prepared relational pattern observations

[`observe_relational_pattern`](../../src/tnfr/physics/relational_observations.py)
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
The [derivation and scope](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-work-integration)
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
[nonlinear response theorem](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#regional-phase-mobility-balance).

The actual engine always supplies field `work`, `phase_mobility` and
`phase_rate_rounding_defect`, and regional `work`, `boundary` and
`phase_response`. Optional `None` defaults preserve earlier manually
constructed report records. These additional observations alter no evolution
law and certify no autonomous regional closure or monotone clock.
The [state-and-rate counterexample](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#state-rate-predictivity)
shows why: a ten-coordinate coarse projection and its predicted rate can agree
while coarse acceleration differs. This does not identify the full reports:
their retained centered form vectors distinguish the omitted internal
contrast. No acceleration observer or reduced evolution law is implied by
the existing snapshot interface.
The separate [joint memory theorem](../../theory/nodal/RELATIONAL_PATTERN_MEMORY.md)
uses those full coordinates to justify a prepared-even approximation with
conditional finite-horizon error bounds. Its neighborhood and derivative
constants are not part of an executable admission certificate; the SDK still
executes and reports the full nonlinear state.

`tnfr.sdk.relational_report_to_dict(report)` accepts only a
`RelationalExchangeField`, `RelationalExchangeStep`,
`RelationalPatternObservation`, `RelationalAttachmentObservation`,
`RelationalRelocationObservation`, `RelationalAttachmentSupplyAssessment`, or one of
the scoped capture/transit certificate types documented below.
Its detached JSON-compatible envelope has
`schema="tnfr.relational-report.v1"`, `report_type` and recursively projected
`report` fields. Exact fractions use
`{"numerator": ..., "denominator": ...}` records; tuples become ordered arrays,
including tuple node labels. JSON scalar labels are supported; opaque objects
are rejected rather than replaced with `repr`. Save the mapping with
`export_to_json(relational_report_to_dict(report), path)`.

This projection is not the `StudyResult` schema, a restorable checkpoint, a
source fingerprint or authentication of a caller-supplied report. Node-type
reconstruction is not promised. It supplies no new CLI execution mode.
[Usage](../guides/REGIONAL_AND_RELATIONAL.md#observe-a-prepared-relational-pattern) belongs to the
shared SDK guide.

<a id="relational-attachment-observation"></a>
### Supplied relational attachment observation

The shared attachment observer in
[relational observations](../../src/tnfr/physics/relational_observations.py)
and the SDK delegate compare two disjoint, separately admitted connected
components with one supplied ordered unit bridge. The model must select the
acute phase domain, and the joined support is admitted independently.
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
The [interface derivation](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#one-bridge-interface-admission)
owns the result; [SDK usage](../guides/REGIONAL_AND_RELATIONAL.md#compare-a-supplied-connection)
owns the call and report field names. The common relational exporter also
accepts this report, validating labels in its nested component/joined fields,
bridge and port cards and retaining detached exact differences.

<a id="relational-relocation-observation"></a>
### Supplied relational bridge relocation

`observe_relational_relocation(graph, *, model, remove_bridge, add_bridge)`
in the same [observation owner](../../src/tnfr/physics/relational_observations.py)
and `Network.relational_relocation(model, *, remove_bridge, add_bridge)` compare
one supplied support exchange without changing primitive form, phase or held
capacity. The original graph must pass the native connected, simple, unit-edge,
unforced acute-model admission. Removing the supplied existing edge must leave
exactly two connected components, each containing at least two nodes. The first
old endpoint identifies the first component. The new ordered endpoints must
cross those components in the same order, and the new edge must be absent from
the original support. No-op replacements, internal new edges and removal of a
non-bridge or leaf bridge reject. The final connected field passes native acute
admission independently. Nonnegative capacities, including zero, retain their
normal execution meaning.

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
[relocation theorem](../../theory/nodal/RELATIONAL_PATTERN_COMPOSITION.md#identity-preserving-bridge-relocation).

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
[usage](../guides/REGIONAL_AND_RELATIONAL.md#compare-a-supplied-bridge-relocation).

### Conditional relational capture

[`certify_relational_capture`](../../src/tnfr/physics/relational_capture.py) and
`Network.relational_capture(model, *, cycles)` apply the
[protected-capture theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-protected-capture)
to one fresh detached field. They add no solver, projection or controller.
The existing model/graph/phase-domain admission runs first. Supply exactly
two ordered five-node cycles covering the full graph; its only other edges
must join matching positions zero and one. Unsupported topology or malformed
inputs raise. An admitted engine snapshot need not satisfy this theorem.

Each ring must have exactly identical form and phase coordinates
`x=m+(A,-A,-B,0,B)` and `theta=c+(a,-a,-b,0,b)`, with held unit capacities
and positive EPI, phase and storage coefficients. Fractions of the captured
values retain both offsets and every copy/reflection defect. Nonzero defects
are not rounded away, even when small. The theorem's exact symmetry is a
sufficient premise; its failure is not proof that recovery is impossible.

Mathematical-pi enclosures test three disjoint open rectangles in `(a,b)`:

| Rectangle | Phase coordinates | Ideal limiting phase sector |
| --- | --- | --- |
| Positive twist | `(2*pi/3,pi) x (0,pi/2)` | `+1` |
| Consensus | `(-2*pi/3,2*pi/3) x (-pi/2,pi/2)` | `0` |
| Negative twist | `(-pi,-2*pi/3) x (-pi/2,0)` | `-1` |

The same rational cosine owner encloses full-support phase storage at the
exact represented raw radians. `storage_bounds` adds exact form storage and
the declared beta factor; its upper bound must be strictly below `7*beta`.
This threshold is derived for this support, symmetry and storage model, not a
universal physical constant or a configurable acceptance tolerance. The
field's floating `phase_storage` is retained separately and is not used as
an exact energy enclosure. Chosen real phase lifts are not silently replaced.

`RelationalCaptureCertificate` retains all candidate rectangle margins, the
selected `rectangle_kind`, exact symmetry defects, coefficient/capacity
admission, storage bounds and enclosure provenance. It sets `target_sector`
only when every premise passes; otherwise `status="unavailable"`,
`target_sector=None` and explicit `unavailable_reasons` are retained.
Its `admitted` property concerns the ideal continuous conditional law from
this exact represented initial state. Current winding is computed separately
by the shared observer and need not match the predicted limiting sector.

The result does not certify future Euler steps, authenticate a caller-created
dataclass, establish an error bound for a numerical trajectory, or demonstrate
formation from winding zero. Exact capture implies an open full-state basin
qualitatively, but no asymmetry tolerance is computed or admitted here.
The shared `relational_report_to_dict` exporter preserves the exact evidence.
[Usage](../guides/REGIONAL_AND_RELATIONAL.md#check-a-protected-relational-basin) and
[routine controls](../../tests/test_relational_capture.py) exercise the same owner.

`certify_relational_local_capture(graph, *, model, cycles, target_sector=1)`
and `Network.relational_local_capture` apply the existing full-state local
energy theorem to the same supplied support. They require unit capacity and
storage scale, positive coefficients and a declared sector in `{-1,0,1}`.
They do not require reflection symmetry. The ideal reference in each ring is
`sector*pi*(4/5,-4/5,-2/5,0,2/5)`; common form and phase offsets are removed
exactly, without changing individual phase lifts. Rational pi and cosine
enclosures must prove squared quotient distance below `9/800` and excess
storage above the target below `1/100000`. These conservative constants come
from the local energy barrier, not fitting the evaluated response.

`RelationalLocalCaptureCertificate` retains the declared target, affine-pi
phase errors, distance/energy bounds and individual admission reasons. Its
target is available only when all premises hold. A negative lower bound on
the enclosed excess is allowed: interval uncertainty can straddle zero,
while the admitted local theorem supplies nonnegativity of the exact excess.
This certifies ideal continuation from the precise represented snapshot.
Applying it to a numerical endpoint supplies no error enclosure connecting
that endpoint to the original continuous initial-value problem. Both capture
reports use the same exact exporter; neither controls the dynamics.

`certify_relational_sector_capture(graph, *, model, cycles, target_sector=1)`
and `Network.relational_sector_capture` apply the
[full-state acute-sector theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-acute-sector-capture).
On the same support they require strictly positive held capacities, positive
storage scale and coefficients, strictly acute gaps and both ring periods equal to the declared
`+1` or `-1` sector. Exact affine-pi inequalities validate integer full-turn
candidates for each raw edge difference. The three independent cycle periods
include the four-edge bridge cycle, which must have period zero. Floating
remainders and numerical winding telemetry do not establish these premises.

`RelationalSectorCaptureCertificate` retains these gap margins and periods,
full energy bounds and a certified interval for the sufficient barrier
`beta*[10-5*cos(2*pi/5)-4*cos(3*pi/8)]`. The energy upper bound must be strictly below
the barrier lower bound. Exact reflection and a small Euclidean distance to a
target are unnecessary. The limiting target is circular, modulo a common
offset and fixed nodewise full-turn representatives. This is a sufficient
barrier, not a claimed sharp bound or a numerical tolerance. Unresolved
premises remain unavailable with evidence. As with the other certificates,
admission concerns ideal continuation from the captured state; applying a
new theorem after an experiment does not revise its frozen decision criteria.

`geometric_barrier_bounds` encloses the unscaled geometric cost and
`capture_barrier_bounds` includes beta. The retained `unit_capacity` and
`unit_storage_scale` flags are descriptive; `positive_capacity` supplies
the capacity admission. Zero capacity is not admitted by this recovery theorem.
For full admission, `normalized_energy_margin_lower_bound=eta_lower>0`
supplies three conditional all-future bounds: acute edge margin `13*eta_lower`,
nodewise resultant real part `26*degree*eta_lower/pi_upper`, and phase metric
`26*degree*eta_lower`. The resultants/metrics follow `field.nodes`. All three
future-bound fields are `None` when any premise fails, even if energy alone
passes. They describe the ideal continuation, not Euler error or a solver step.
The [derivation](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-sector-consolidation)
connects those margins to the same Jensen barrier without adding a threshold policy.

| Certificate | Distinguishing sufficient premises | Scope |
| --- | --- | --- |
| `relational_capture` | Exact reflected state, unit capacity, positive beta, strict rectangle and energy below `7*beta` | Can admit nonacute states and consensus |
| `relational_local_capture` | Unit capacity/beta, small full-state quotient distance and excess energy | No reflection; declared target `-1`, `0` or `1` |
| `relational_sector_capture` | Positive held capacities/beta, exact acute sector and scaled barrier | No reflection or local-radius gate; targets `-1` or `1`; quantitative regularity |

These independent sufficient theorems share one capture owner and exact
exporter. No ordering of their verdicts or automatic policy is implied.

### Validated conditional relational transit

[`certify_relational_transit_capture`](../../src/tnfr/physics/relational_transit.py)
and `Network.relational_transit_capture(*, model, cycles, horizon, time_step,
order=12, requested_sector=1)` perform a read-only proof computation for the
ideal continuous law.
The [validated-transit derivation](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-validated-transit)
owns the reduction and enclosure argument. This API does not advance the
graph, execute a numerical engine step, select operators or alter the supplied
preparation. A represented Euler chord is not used as an exact ODE tube.

The initial state must have the same exact copied/reflected two-ring support
and coordinates as `relational_capture`, with held unit capacities, positive
coefficients and `model.phase_domain="positive_resultant"`. Existing graph,
scalar and cycle admission remains active. Reflection defects are never
projected away. The initial snapshot need not already pass a protected-basin
rectangle or storage test: its complete detached `RelationalCaptureCertificate`
is retained as `initial`, independently of the transit verdict.

`horizon` and `time_step` must be strictly positive exact `Fraction` or integer
values; floats and Booleans are rejected. These times are relative structural
durations from the supplied state, with no interpretation of a stored graph
clock or laboratory units. `order` must be an integer from 4 through 16.
The work policy admits at most 4096 declared proof steps. The last interval is
shortened to reach the exact horizon; there is no automatic retry, preparation
change, adaptive step selection or horizon extension.

Interval coordinates are `(q,r,a,b)`, where `q=3*A-B` and `r=2*B-A`. Every
accepted `TransitStep` retains a strict whole-time Picard inclusion, positive
relative-resultant bounds for all distinct node rows including the central
row, and a rational endpoint enclosure. Centered interval Taylor expansion
bounds analytic truncation on that whole tube. A Metzler matrix comparison
propagates the entering coordinate uncertainty using enclosed Jacobian bounds;
128-bit outward dyadic arithmetic retains rounding uncertainty. Whole-time
regularity, local truncation and propagated error are separate obligations.
The analytic `atan(u)/u` enclosure retains its zero-safe series for constant
intervals inside `[-1/2,1/2]`. Outside that interval, Taylor evaluation requires
the constant interval to exclude zero and encloses the same function through
`atan(u)` and interval division. An interval outside the series domain that
also contains zero is unavailable. This extends the proof arithmetic's
admitted domain without changing the relational pressure law, phase chamber
or evolution equation. It introduces no additional evolution. Frozen source
and result bundles retain their original provenance.

`RelationalTransitCertificate.admitted` requires the entire requested horizon
to be validated and the entire final box to lie in a selected protected
rectangle, with joint-storage upper bound strictly below `7*beta`. The
`requested_sector` policy accepts integer `1`, `0`, `-1` or `None`; Booleans,
floats and other values are rejected. Its default `1` requires the positive
rectangle `(2*pi/3,pi) x (0,pi/2)`. Zero requests the consensus rectangle
`(-2*pi/3,2*pi/3) x (-pi/2,pi/2)`, and `-1` requests the reflection of the
positive rectangle. `None` accepts any one of these disjoint rectangles.
The point and interval capture owners share their affine margin definitions.

This sufficient endpoint gate invokes the existing protected-basin theorem.
Only full admission sets `target_sector` to the limiting sector actually
certified: consensus is the valid integer `0`, not an unavailable result.
`requested_sector` is a caller's admission policy; it neither changes the
trajectory nor establishes its current winding or limiting target. Endpoint
centers, sampled storage or floating winding alone cannot pass the gate.
`initial_winding_zero` separately
certifies a sufficient zero-winding condition on the original lifted phase
state; it is not a gate for a general transit certificate. A claim of entry
from winding zero requires both this flag and complete transit admission.

An unresolved proof step stops the calculation and retains the accepted
`steps`, `validated_horizon`, their final `endpoint`, the attempted
`failed_tube` and explicit `unavailable_reasons`. A fully validated horizon
can also return unavailable when the whole endpoint fails the sufficient
capture gate. Unavailable means the requested proof was not obtained, not that
the ideal trajectory fails to recover. No limiting target is assigned then.
The report preserves `initial_box`, requested clock/order policy,
`endpoint_storage` and enclosure-method scope. All three candidates' interval
margins appear in `candidate_rectangle_margin_bounds`, in sector order
`1,0,-1`. A matching strict rectangle supplies `rectangle_kind` and
`rectangle_margin_bounds`, even if another obligation leaves the overall
report unavailable; without a match they are `None` and empty respectively.
`positive_rectangle_margins` remains the positive candidate's margins for
compatibility, including when consensus or negative capture is requested.
These geometric observations do not override horizon, storage or initial-law
admission failures, which always leave `target_sector=None`.

`relational_report_to_dict` exports the full nested evidence with exact
fraction endpoints and validates node/cycle labels through `initial`.
Projection is detached evidence, not a restart format or provenance
authentication. Successful admission concerns this conditional ideal ODE;
it supplies no guarantee for subsequent finite engine execution and leaves
every original frozen experimental verdict unchanged. See the
[SDK usage](../guides/REGIONAL_AND_RELATIONAL.md#validate-continuous-transit-to-a-protected-basin).

### Exact cycle resultant sectors

[`derive_cycle_resultant_sector`](../../src/tnfr/physics/phase_resultant_sectors.py)
accepts a supplied pure cycle, its full oriented `cycle_nodes` order and an
exact full-support `phase_turns` mapping. Turns mean angles divided by
mathematical `2*pi`; inputs must be rational numbers, excluding Booleans and
floats. Live graph phases, capacity and conductance weights are not consumed.
The shared topology budget applies; at least five nodes are required.

The detached report separates ordinary `support_winding`, the windings of
derived skip-two cycles, `zero_resultant_nodes` and
`negative_real_resultant_nodes`. Antipodal edges make the corresponding
winding unavailable. `resultant_sector_available` only requires nonzero
neighbor resultants; `regular_phase_chart` also excludes the local negative-real
Arg branch. Neither flag admits a runtime step. The auxiliary cycles add no
coupling edges and are not another state variable. Exact classifications of
declared turns do not certify rounded radian measurements or a whole path.
The [formation-domain theorem](../../theory/nodal/RELATIONAL_EXCHANGE_ADMISSION.md#relational-formation-domain)
owns the conditional invariant and its degree-two scope; extra neighbors can
invalidate that obstruction. This exact-turn observer does not select or
extend either of the executor's separately declared phase chambers.
