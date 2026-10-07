# Sine pattern geometry, formation and recovery

Relative patterns and origins, composition, equilibrium and symmetry, prepared formation, recovery/capture, controlled slow references and budget exclusions.

Part of [Regional and relational SDK workflow index](../REGIONAL_AND_RELATIONAL.md). Section links remain stable; hypotheses and model changes remain local to each result.

<a id="sine-formation-response"></a>
### Check formation and its inherited receiver signature together

The fixed doubled-C5 comparison starts with zero nominal phases and a declared
integer form profile. Two preparations allocate the same nominal internal
storage to different pairs. This example checks acquisition and subsequent
geometric retention, together with a finite absolute receiver reading under
the same supplied positive-loss law:

```python
from fractions import Fraction as Q
from tnfr.physics.relational_sine_formation_response import (
    assess_sine_formation_response,
)

report = assess_sine_formation_response(
    scaled_time=100,
    form_error_bound=Q(1, 10**10),
    phase_error_bound=Q(1, 10**10),
    readout_error_bound=Q(1, 10**10),
    radius=Q(1, 8),
)
assert report.initial_zero_winding_certified
assert all(report.formation_certified_by_preparation)
assert report.recorded_difference_bounds.lo > Q(1, 10**8)
assert report.status == "certified_formation_response"
evidence = report.to_dict()
```

The time is `tau=e*t` with `e=1023/1024`; it is neither operator cycles nor
laboratory seconds. The initial error boxes include every fine coordinate
and the form origin. The response is the actual mean form of receiver pair 1
at that time, including readout uncertainty. It is not an integral of just one
pressure channel. No trajectory is numerically executed or reset after entry.

Inspect the separate radius, acute-geometry, storage and response margins when
changing a budget. `unavailable` means these sufficient estimates do not
certify the requested joint claim. Geometric identity is retained after entry;
the finite receiver contrast need not persist indefinitely. The preparation,
support and loss remain supplied. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-formation-response)
and [frozen protocol and proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-formation-response).

### Describe a whole pattern relative to a moving node

For a complete supplied sine-law state, remove unknown common origins while
retaining independent errors in its geometry:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

graph = nx.path_graph(3)
for node, form, phase in ((0, 0, 0), (1, 1, 0.5), (2, 2, 0)):
    graph.nodes[node].update(EPI=form, theta=phase, nu_f=1)
graph.graph["GAMMA"] = {"type": "none"}
error = Fraction(1, 1024)
pattern = bound_relational_sine_pattern(
    graph,
    reference_node=0,
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    form_error_bounds=(error, error, error),
    phase_error_bounds=(error, error, error),
)
assert pattern.relative_form_bounds[0].width == 0
assert pattern.relative_form_rate_bounds[0].contains(0)
assert pattern.reference_form_rate_bounds.lo > 0
evidence = pattern.to_dict()
```

The zero reference-relative coordinate is an identity, although the actual
reference is moving. The supplied radii concern synchronous residual errors
after a shared offset, in consistently lifted phases; this example is not
measured data. Every intermediary and capacity remains present.

For a prospective full-state enclosure, call `pattern.forecast` with explicit
`observation_time`, `end_time`, `time_step` and `order`.
Inspect `admitted` and `full_forecast.validated_end_time`. Use the returned
forecast bounds for the propagated family; the source's static storage bound
does not cover every point of the solver's larger outer box.
See the [contract](../../contracts/relational/SINE_PATTERNS.md#relative-sine-patterns-and-moving-references).
This does not infer a missing node or certify recovery. In particular, the
absolute-rate inverse below cannot consume moving-reference rates unchanged.

### Certify recovery of an uncertain cycle pattern

This example supplies a deformed five-node cycle and nonzero residual errors.
The target remains the exact mathematical twist; the nominal phases need not
be an equilibrium:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

graph = nx.cycle_graph(5)
forms = (1, -1, 0, 0, 0)
for node in graph:
    graph.nodes[node].update(
        EPI=Fraction(forms[node], 1024), theta=Fraction(5*node, 4), nu_f=1,
    )
graph.graph["GAMMA"] = {"type": "none"}
error = Fraction(1, 4096)
pattern = bound_relational_sine_pattern(
    graph,
    reference_node=0,
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    form_error_bounds=(error,)*5,
    phase_error_bounds=(error,)*5,
)
recovery = pattern.certify_cycle_recovery(
    cycle=tuple(range(5)), winding=1, radius=Fraction(1, 16),
)
assert recovery.admitted
assert recovery.norm_margin > 0 and recovery.energy_margin > 0
recovery_evidence = recovery.to_dict()
```

The proof covers every state consistent with those residual bounds, including
arbitrary common origins. It uses no trajectory and claims no laboratory
measurement. An explicit analytic check of this same dyadic observation class
is given by the [proof owner](../../../theory/nodal/SINE_PATTERN_RECOVERY.md#sine-cycle-recovery).

The same method on a `SineRelativeForecast` assesses its full endpoint box
at the actual validated time, preserving the solver's original status.
Inspect `hypothesis_failures` and `unresolved_conditions` if admission is
unavailable. This sufficient test does not diagnose instability.
The [contract](../../contracts/relational/SINE_PATTERNS.md#whole-set-sine-cycle-recovery)
requires the **complete** support to be the cycle: do not remove environmental
connections to make a subregion pass.

### Protect cycle identity during conservative motion

The same barrier used for dissipative recovery can instead certify that a
conservative cycle stays near its prepared phase geometry. The family-level
recurrence theorem then applies without asserting a period or an individual
return. Here the support is the **whole** isolated five-node ring:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.physics.relational_sine_recovery import assess_sine_cycle_identity
from tnfr.sdk import relational_report_to_dict

graph = nx.cycle_graph(5)
forms = (1, -1, 0, 0, 0)
for node in graph:
    graph.nodes[node].update(
        EPI=Fraction(forms[node], 1024), theta=Fraction(5*node, 4), nu_f=1,
    )
model = RelationalExchangeModel(
    1, epi_weight=0, phase_weight=1, phase_domain="regular",
)
error = Fraction(1, 4096)
pattern = bound_relational_sine_pattern(
    graph, reference_node=0, reference_model=model,
    form_error_bounds=(error,)*5, phase_error_bounds=(error,)*5,
)
identity = assess_sine_cycle_identity(
    pattern, cycle=tuple(range(5)), winding=1, radius=Fraction(1, 16),
    excess_ceiling=Fraction(1, 2560), form_mean_bounds=(-1, 1),
)
assert identity.family_admitted
assert identity.source_set_trapping_certified
assert identity.relative_source_family_membership == "certified_inside"
assert identity.family_almost_everywhere_recurrence_certified
assert identity.individual_recurrence_status == "unavailable_for_chosen_state"
identity_evidence = relational_report_to_dict(identity)
```

All states consistent with these relative error bounds retain winding one.
The declared mean interval is a separate restriction defining a finite-volume
family: the relative observation cannot measure an absolute common form
origin. Almost every member of that family returns near its starting full
state, but the displayed flags do not certify recurrence of this particular
preparation. They also do not imply that the ring returns to its exact target
or that it formed autonomously. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-conservative-identity)
for independent family, trapping and membership verdicts.

### Check a formation preparation before evolving it

This declared family starts with a twisted donor and a uniform receiver.
A form pulse at the intermediary and unequal receiver capacities produce
an initial directed deformation, but that does not guarantee a new twist:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

model = RelationalExchangeModel(1, phase_domain="regular")
preparation = dict(
    model=model, amplitude=2, capacity_contrast=Fraction(1, 2),
)
initial_checks = assess_sine_mediated_formation(**preparation)
assert initial_checks.initial_form_storage == 4
assert initial_checks.target_budget_status == "passed"
assert initial_checks.entry_budget_status == "passed"
assert initial_checks.phase_action_bound.necessary_entry_time_lower_bound > 240
barrier = initial_checks.maintained_target_obstruction
assert barrier.maintained_target_excluded
assert barrier.target_margin_bounds.lo > Fraction(1, 4)
assert initial_checks.status == "excluded"
assert not initial_checks.early_loss_exclusion_certified
timed = assess_sine_mediated_formation(
    **preparation, exclusion_time=Fraction(1, 5),
)
assert timed.status == "excluded"
assert timed.phase_action_bound.entry_through_horizon_excluded
assert timed.early_loss_exclusion_certified
formation_evidence = timed.to_dict()

# A different exchange/loss balance inside the proved domain.
changed_balance = assess_sine_mediated_formation(
    model=RelationalExchangeModel(
        1, epi_weight=4, phase_weight=5, phase_domain="regular",
    ),
    amplitude=2, capacity_contrast=Fraction(1, 2),
)
changed_barrier = changed_balance.maintained_target_obstruction
assert 1 < changed_barrier.exchange_to_loss_ratio < Fraction(3, 2)
assert changed_barrier.maintained_target_excluded
assert changed_barrier.target_margin_bounds == barrier.target_margin_bounds

# A separate endpoint: the donor unwinds and only the receiver stays twisted.
transfer = timed.receiver_transfer()
assert transfer.source is timed
assert transfer.status == "passes_necessary_conditions"
assert transfer.target_storage_margin == 4
assert transfer.target_functional_margin == Fraction(16, 5)
assert transfer.combined_phase_action_cost == Fraction(29, 25)
assert transfer.compatible_common_phase_turn_offset == Fraction(-11, 190)
assert not transfer.early_loss_exclusion_certified
assert transfer.receiver_first_barrier_phase_storage == Fraction(7, 2)
assert transfer.necessary_receiver_running_work_strict_lower_bound == Fraction(7, 2)
assert transfer.donor_barrier_first_required
assert transfer.simultaneous_barrier_passage_excluded
transfer_evidence = transfer.to_dict()
```

The initial storage checks pass, but the independent mixed-functional
certificate excludes convergence to the maintained two-twist target without
a supplied horizon. A necessary entry time above 240 is conditional on joint
acute entry; it predicts neither entry nor capture. The timed report separately
proves that dissipation exhausts the joint-entry budget before that entry can
occur. Both reports use analytic bounds without evolving a trajectory.

The [whole-class proof](../../../theory/research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-maintained-target-obstruction)
excludes the maintained target for all five donor forms and intermediary form
with initial form storage at most four, under the fixed capacity and storage
premises and the sufficient ratio domain `0<w/e<3/2`. The example calls are
applications of that theorem. This interval is not a sharp physical threshold.
The global
certificate alone does not exclude every transient acute-region visit or
receiver-only winding after losing the donor's identity. Leaving the admitted
ratio domain or changing storage scale or contrast makes this certificate
unavailable; that is not positive formation. Common raw-weight scaling is
removed by ideal normalization; rounded inputs still use their represented
effective ratio. Neither operation implements a new execution clock.

The [contract](../../contracts/relational/SINE_PATTERNS.md#sine-formation-eligibility-and-timed-exclusion)
documents profile admission, the separate timed bounds and export fields.
Use `profile="explicit"` with five ordered `donor_epi` values to retain internal
donor differences; `amplitude` still supplies intermediary form. No support
event, new loss law or native runtime change is involved.

`receiver_transfer()` keeps this same preparation but asks about a different
final identity. Its passed necessary checks do not prove that the receiver
will acquire the pattern. The original two-pattern target remains excluded.
The [transfer contract](../../contracts/relational/SINE_PATTERNS.md#receiver-identity-transfer-with-donor-unwinding)
keeps both winding changes, their combined phase-action cost and the new loss
allowance explicit. The reported common phase offset is one compatible lift,
not a predicted final absolute phase. To certify recovery of an actual later
observation, pass `transfer.target_phase_turns` to the existing full-pattern
recovery reader; a positive initial budget does not replace that observation.

The passage flags describe necessary geometry and work **if transfer occurs**.
At form budget at most `4`, the donor must cross its own potential barrier
before the receiver reaches its first one. This is not a statement about
when either winding changes. The receiver must receive accumulated signed
port work exceeding `7/2`, with an additional loss cost for a crossing within
a given time. These are prospective requirements: the initial report does
not evaluate that accumulated work, and a positive instantaneous work rate
does not satisfy them. A false ordering flag leaves the order unknown.

### Distinguish source geometry from a formation prediction

The existing preparation report can retain its communicating and silent
components while bounding the tangent receiver response:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

source = assess_sine_mediated_formation(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    amplitude=1, capacity_contrast=Fraction(1, 2),
    profile="explicit", donor_epi=(0, 1, 0, 0, -1),
)
excitation = source.receiver_excitation()
assert excitation.available
assert excitation.even_form_storage == 1
assert excitation.odd_form_storage == 2
assert excitation.tangent_receiver_phase_cost_upper_bound == Fraction(3, 4)
assert excitation.nonlinear_correction_status == "positive_bound"
assert excitation.full_phase_budget_status == "available"
assert excitation.actual_full_phase_storage_upper_bound == Fraction(139, 20)
assert excitation.donor_potential_barrier_first_required
assert excitation.necessary_donor_phase_release_lower_bound > 0
assert tuple(a + b for a, b in zip(
    excitation.even_initial_epi, excitation.odd_initial_epi,
)) == source.initial_epi
excitation_evidence = excitation.to_dict()
```

This example evaluates no trajectory. The tangent ceiling and correction
threshold describe the response needed for nonlinear barrier passage. The
separate full-phase ceiling applies to the actual law at total form budget
at most `4`: receiver passage requires an earlier donor barrier crossing.
Neither bound proves that passage occurs or identifies the donor's potential
decrease as useful receiver work.
Reversing only the odd source retains its energy and tangent receiver response
but changes nonlinear receiver work. Keep both components when studying the
actual law. The [contract](../../contracts/relational/SINE_PATTERNS.md#source-geometry-and-the-necessary-nonlinear-receiver-correction)
defines unavailable bounds and the [proof](../../../theory/research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-source-receiver-excitation)
states the exact distinction; neither selects this example for a new response.

### Certify receiver consensus without selecting the donor endpoint

A regional nonlinear certificate applies to every six-coordinate preparation
whose total initial form storage is at most `(5-sqrt(5))/2`. It can establish
receiver consensus when the separate donor-recovery checks are inconclusive:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

source = assess_sine_mediated_formation(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    amplitude=1, capacity_contrast=Fraction(1, 2),
)
assert source.initial_form_storage == 1
assert not source.donor_well_retention().relative_donor_pattern_convergence_certified
assert not source.donor_dissipative_capture().relative_donor_pattern_convergence_certified
localization = source.receiver_localization()
assert localization.status == "certified"
assert localization.relative_receiver_consensus_certified
assert localization.receiver_nonflat_equilibria_excluded
transfer = source.receiver_transfer()
assert transfer.receiver_localization == localization
assert "weighted_receiver_localization" in transfer.exclusion_reasons
assert transfer.status == "excluded"
localization_evidence = localization.to_dict()
```

This evaluates no trajectory. The proof weights regional potentials and
retains their full nonlinear coupling. It selects the receiver's relative
limit, not a donor limit or a time to recovery. Transient receiver excursions
remain possible; no all-time barrier verdict follows. A `not_certified` result
above the threshold leaves acquisition unresolved. The
[contract](../../contracts/relational/SINE_PATTERNS.md#nonlinear-receiver-localization-from-regional-storage)
and [proof](../../../theory/research/archive/receiver/SINE_RECEIVER_FORMATION_BOUNDS.md#sine-weighted-receiver-exclusion)
own the exact admission and scope.

### Certify recovery of the prepared donor

The same preparation reader can now select the donor-only relative limit
for a proved interval of form storage. This example is nonsilent and has
enough total storage to pass the bare phase barrier, but the full coupled
dynamics still returns to the original donor pattern:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

source = assess_sine_mediated_formation(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    amplitude=Fraction(7, 30), capacity_contrast=Fraction(1, 2),
)
retention = source.donor_well_retention()
assert retention.initial_form_storage == Fraction(49, 900)
assert retention.relative_donor_pattern_convergence_certified
assert retention.receiver_only_targets_excluded
assert retention.exact_retention_polynomial_margin > 0
transfer = source.receiver_transfer()
assert transfer.status == "excluded"
assert transfer.donor_well_retention.relative_donor_pattern_convergence_certified
retention_evidence = retention.to_dict()
```

The certificate applies to all five donor forms and the intermediary form,
not only this localized example. It combines the exact equilibrium geometry
with the existing decreasing auxiliary function. Its threshold is
`F_c=(25*sqrt(5)-55)/16`, approximately `0.056356`, and admission compares
`(16*F+55)**2 <= 3125` exactly. Neither this proof coefficient nor its threshold
is a new physical constant.

This proves the final relative donor identity and excludes either maintained
receiver-only twist. It gives no arrival time or guarantee that winding never
changes during the transient. Above the threshold the certificate does not
decide the outcome; a connected static path is insufficient. The
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-donor-well-retention)
retains the law premises, exact admission and distinct transfer bounds.

### Prove dissipative recovery above the initial barrier

A preparation can exceed the static donor-retention bound and still recover
the donor. This certificate uses the full law's short-time dissipation and
phase displacement to prove entry into the donor component:

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_formation import assess_sine_mediated_formation

source = assess_sine_mediated_formation(
    model=RelationalExchangeModel(1, phase_domain="regular"),
    amplitude=Fraction(1, 4), capacity_contrast=Fraction(1, 2),
)
assert source.initial_form_storage == Fraction(1, 16)
assert source.donor_well_retention().status == "not_certified"
capture = source.donor_dissipative_capture()
assert capture.horizon == Fraction(1, 4)
assert capture.initial_dissipative_norm_squared == Fraction(1, 6)
assert capture.donor_component_entry_certified
assert capture.relative_donor_pattern_convergence_certified
transfer = source.receiver_transfer()
assert transfer.status == "excluded"
assert transfer.donor_dissipative_capture.receiver_only_targets_excluded
capture_evidence = capture.to_dict()
```

The horizon certifies membership in a region that guarantees eventual donor
recovery; it is not the time at which recovery finishes. No trajectory is
numerically evolved. The bound uses both form storage and its full-support
gradient, so equal storage alone does not determine its verdict. A failed
sufficient check leaves the outcome open. This certificate has the original
half-weight law scope; the static donor-well result retains its wider
coefficient-ratio domain. See the
[capture contract](../../contracts/relational/SINE_PATTERNS.md#sine-donor-dissipative-capture)
for exact admission and retained evidence.

### Compose established component Hessians through a bridge tree

Use this algebra only after deriving each component's relative Hessian inertia
and the bridge signs. Here two C5 components have independently established
positive four-dimensional relative Hessians; a singleton mediates their union:

```python
from tnfr.physics.phase_cycle_geometry import compose_bridge_tree_hessian_inertia

components = ((4, 0, 0), (4, 0, 0), (0, 0, 0))
aligned = compose_bridge_tree_hessian_inertia(
    components, ((0, 2, 1), (2, 1, 1)),
)
assert aligned.total_nodes == 11
assert aligned.relative_inertia == (10, 0, 0)
opposed = compose_bridge_tree_hessian_inertia(
    components, ((0, 2, -1), (2, 1, 1)),
)
assert opposed.relative_inertia == (9, 1, 0)
composition_evidence = aligned.to_dict()
```

This does not reconstruct or check a live phase configuration. The specialized
C5 catalog below supplies those premises from its exact critical states and
delegates to the same algebra. Under the full positive-loss sine theorem,
an aligned bridge preserves positive restoring geometry and an antipodal
bridge adds an unstable direction. Extra connections that close a cycle
require another compatibility argument; neither stability nor the input
inertia triples establish formation. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#exact-geometric-inertia-and-full-law-stability)
and [proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-bridge-tree-composition).

### Reject incompatible cycle periods before seeking an equilibrium

Cycles sharing paths cannot choose their phase periods independently. The
shared exact reader tests one combined-cycle constraint, including cancellation
on shared edges. This example has two fundamental five-edge cycles; each
period individually meets its acute length bound, but their joint assignment
fails on the four-edge cycle formed by their difference.

```python
import networkx as nx
from tnfr.physics.phase_cycle_geometry import (
    assess_acute_cycle_periods,
    derive_phase_cycle_geometry,
)

graph = nx.Graph()
graph.add_nodes_from(range(6))
graph.add_edges_from(((0, 1), (0, 2), (2, 3), (1, 4), (3, 4), (1, 5), (3, 5)))
geometry = derive_phase_cycle_geometry(graph)
assert tuple(map(len, geometry.fundamental_cycles)) == (5, 5)

single = assess_acute_cycle_periods(
    geometry, cycle_periods=(1, -1), cycle_combination=(1, 0)
)
assert single.status == "necessary_bound_passed"
joint = assess_acute_cycle_periods(
    geometry, cycle_periods=(1, -1), cycle_combination=(1, -1)
)
assert joint.combined_period == 2
assert joint.strict_period_bound == 1
assert joint.obstruction_certified
payload = joint.to_dict()
```

An obstruction rules out every strictly acute state with those periods,
not only one proposed phase assignment. Passing a witness leaves existence
unresolved. Even a reconstructed acute phase state can fail the separate
nodal sine-balance condition. The
[proof and controls](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-cycle-sector-compatibility)
distinguish these obligations. This readout creates no connections and does
not establish that a compatible pattern forms.

### Certify sector capture without supplying its final pattern

A whole-sector storage barrier can guarantee convergence without giving the
reader equilibrium coordinates. This source has nonuniform form and a phase
profile with nonzero cycle period. Its future is assessed under the explicitly
supplied complete sine law; the initial state is not declared stationary.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

graph = nx.cycle_graph(5)
for node in graph:
    graph.nodes[node].update(
        EPI=Fraction(1, 1024) if node == 0 else 0,
        theta=Fraction(5 * node, 4),
        nu_f=2 if node == 1 else 1,
    )
source = bound_relational_sine_pattern(
    graph,
    reference_node=0,
    reference_model=RelationalExchangeModel(1, phase_domain="regular"),
    form_error_bounds=(Fraction(1, 65536),) * 5,
    phase_error_bounds=(Fraction(1, 65536),) * 5,
)
# Canonical edges: (0,1), (0,4), (1,2), (2,3), (3,4).
# Subtract a full turn from the 0->4 phase difference.
capture = source.certify_sector_capture(edge_turn_offsets=(0, -1, 0, 0, 0))
assert capture.cycle_periods == (1,)
assert capture.admitted
assert capture.energy_margin > 0
assert len(capture.boundary_face_lower_bounds) == 10
assert capture.weighted_form_mean is None  # the common origin is unobserved
payload = capture.to_dict()
```

The certificate covers all signed faces of the acute sector and the entire
declared uncertainty set. A positive verdict proves existence, uniqueness and
convergence there without a target solve. An unavailable verdict may reflect
a conservative bound even when an equilibrium exists. This does not generate
the initial winding or support, and a supplied event needs separate state and
storage-jump admission. See the [contract](../../contracts/relational/SINE_PATTERNS.md#target-free-acute-sector-capture)
and [proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-target-free-sector-capture).

### Join two relative patterns with explicit origins

Internal relative coordinates do not specify how two components are aligned.
Declare the right-minus-left common form and phase origins, a common structural
observation time and one bridge. This static assessment rebuilds the full
support and hands the joint uncertainty family to the existing capture owner.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.sdk import relational_report_to_dict

model = RelationalExchangeModel(1, phase_domain="regular")

def component(labels, capacities):
    graph = nx.path_graph(labels)
    for node, capacity in zip(labels, capacities):
        graph.nodes[node].update(EPI=0, theta=0, nu_f=capacity)
    graph.graph["GAMMA"] = {"type": "none"}
    return bound_relational_sine_pattern(
        graph, reference_node=labels[0], reference_model=model,
        form_error_bounds=(0, 0), phase_error_bounds=(0, 0),
    )

left = component((0, 1), (1, 2))
right = component((2, 3), (3, 4))
contact = left.compose_with(
    right, bridge=(1, 2), observation_time=2,
    form_origin_difference=Fraction(1, 4),
    phase_origin_difference=Fraction(1, 4),
    edge_turn_offsets=(0, 0, 0),  # canonical edges: (0,1), (1,2), (2,3)
    work_allowance=Fraction(1, 8),
)
assert contact.status == "available"
assert contact.budget_status == "within_allowance"
assert contact.capture.admitted
assert contact.joined.degrees == (1, 2, 2, 1)
assert contact.capture.weighted_form_mean is None
payload = relational_report_to_dict(contact)
```

These offsets describe the common additive origins in each source family;
they are not measurements of the reference-node gap when residuals are present.
Omitting either offset leaves composition unavailable. A supplied time declares
synchrony; the source reports do not authenticate it. Capture, bridge-work
allowance and event occurrence are separate questions. This call neither adds
a live edge nor runs a trajectory. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-relative-frame-composition)
and [derivation](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-relative-frame-composition).

### Certify acquisition from initially equal phases

The same sine law can transfer a supplied form profile into nonzero phase
winding. This example declares the profile, coefficients and horizon before
evaluating the analytic certificate. It uses no trajectory fit or supplied
equilibrium coordinates.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import certify_sine_prepared_entry

graph = nx.cycle_graph(5)
for node in graph:
    graph.nodes[node].update(EPI=4092 * (node - 2), theta=0, nu_f=1)
graph.graph["GAMMA"] = {"type": "none"}
model = RelationalExchangeModel(
    1, epi_weight=Fraction(1023, 1024),
    phase_weight=Fraction(1, 1024), phase_domain="regular",
)
source = bound_relational_sine_exchange(graph, reference_model=model)
entry = certify_sine_prepared_entry(
    source, scaled_time=100, edge_turn_offsets=(0, -1, 0, 0, 0),
)
assert entry.admitted
assert entry.initial_cycle_periods == (0,)
assert entry.capture.cycle_periods == (1,)
assert entry.horizon == Fraction(102400, 1023)
assert entry.initial_form_storage == 167444640
assert entry.capture.energy_margin > Fraction(1, 100)
payload = entry.to_dict()
```

`endpoint_form_bounds` and `endpoint_phase_bounds` enclose all nodes at the
declared horizon. Tighter correlated edge bounds feed the shared capture
kernel; the full Cartesian node box is only an outer projection. The initial
phase winding is absent, but the nonuniform form information and large initial
storage are supplied. The result proves formation and subsequent retention
under this law; it does not select that preparation or identify matter.
Uniform initial form with equal phases instead stays stationary. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#analytic-prepared-entry-into-a-captured-sector)
and [proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-prepared-sector-entry).

The same graph, model and horizon also admit a full preparation family. Keep
the residual radii fixed before evaluating the following extension:

```python
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern

preparations = bound_relational_sine_pattern(
    graph, reference_node=0, reference_model=model,
    form_error_bounds=(Fraction(1, 16),) * 5,
    phase_error_bounds=(Fraction(1, 65536),) * 5,
)
family = preparations.certify_prepared_entry(
    scaled_time=100, edge_turn_offsets=(0, -1, 0, 0, 0),
)
assert family.admitted and family.initial_zero_winding_certified
assert family.initial_cycle_periods == (0,)
assert family.capture.cycle_periods == (1,)
assert family.capture.energy_margin > Fraction(12, 1000)
assert family.endpoint_form_bounds is None
assert family.weighted_form_mean is None
assert len(family.centered_endpoint_phase_bounds) == 5
```

Every member starts with zero winding and acquires the captured unit winding.
The relative source leaves common origins unknown: use the centered endpoint
bounds, not invented absolute values. Each member keeps its own conserved
means. Initial form and phase storage are interval bounds, so the nominal
budget does not describe the entire family. With uniform nominal form and
the same small errors, the initial zero-sector capture test instead proves
convergence to consensus; that uncertain control is not stationary.

### Join two analytically acquired patterns

Keep the original preparations and analytic endpoint sets. This fixed example
joins two copies at their first nodes after the same declared duration:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_pattern import bound_relational_sine_pattern
from tnfr.sdk import relational_report_to_dict

model = RelationalExchangeModel(
    1, epi_weight=Fraction(1023, 1024), phase_weight=Fraction(1, 1024),
    phase_domain="regular",
)

def prepared_component(start):
    graph = nx.cycle_graph(range(start, start + 5))
    for j, node in enumerate(graph):
        graph.nodes[node].update(EPI=4092 * (j - 2), theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    source = bound_relational_sine_pattern(
        graph, reference_node=start, reference_model=model,
        form_error_bounds=(0,) * 5, phase_error_bounds=(0,) * 5,
    )
    return source.certify_prepared_entry(
        scaled_time=100, edge_turn_offsets=(0, -1, 0, 0, 0),
    )

joined = prepared_component(0).compose_with(
    prepared_component(5), bridge=(0, 5),
    left_initial_time=0, right_initial_time=0,
    form_origin_difference=0, phase_origin_difference=0,
    edge_turn_offsets=(0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0),
)
assert joined.observation_time == Fraction(102400, 1023)
assert joined.acquisition_and_capture_certified
assert joined.capture.cycle_periods == (1, 1)
assert joined.capture.energy_margin > Fraction(9, 1000)
assert joined.budget_status == "not_supplied"
payload = relational_report_to_dict(joined)
```

The call rebuilds each analytic prefix and derives the common endpoint time.
Its source retains both preparations; the endpoint is not converted to an
independent observation box. `status` describes frame availability, while
`acquisition_and_capture_certified` requires both acquisitions and joint
maintenance. No event budget or occurrence follows. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-prepared-composition) and
[proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-prepared-composition).

### Certify an entire phase-flat budget family

Keep the preceding `1023:1` coefficient ratio, but bound the original form
storage by `B=160`. This declares every signed form within that ceiling and
exactly equal initial phases, with arbitrary common origins. The graph
supplies only support; no synthetic nodal state is needed.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.phase_cycle_geometry import derive_phase_cycle_geometry
from tnfr.physics.relational_sine_budget import certify_sine_budget_consensus
from tnfr.sdk import relational_report_to_dict

geometry = derive_phase_cycle_geometry(nx.cycle_graph(5))
model = RelationalExchangeModel(
    1, epi_weight=1023, phase_weight=1, phase_domain="regular",
)
budget = certify_sine_budget_consensus(
    geometry, reference_model=model, capacity=(1,) * 5,
    form_storage_budget=160,
)
assert budget.status == "certified"
assert budget.consensus_certified and budget.zero_winding_certified
assert budget.acute_trapping_certified
assert max(budget.phase_edge_upper_bounds) < Fraction(1, 50)
payload = relational_report_to_dict(budget)
```

This certifies consensus and zero winding for the entire family at all future
times. The earlier acquisition source costs `160*1023^2` and lies outside this
family. A zero budget instead makes the whole family stationary. In general,
the theorem protects the natural `(-pi,pi)` edge strip; strict acuteness is a
separate, stronger flag. Its `eta<=1` condition is a sufficient proof premise,
not an instability threshold. When a comparison is unavailable, inspect
`unresolved`: candidate bounds remain visible, but certified trajectory
bounds are `None`.

Under the feedback premise, `nonzero_winding_budget_lower_bound` records a
necessary cost growing as `(w/e)^-2`; exceeding it does not prove formation.
The [contract](../../contracts/relational/SINE_PATTERNS.md#sine-budget-consensus) and
[proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-budget-consensus)
give the family and admission boundaries. The exact `payload` can be saved
with the SDK's `export_to_json`.

### Distinguish equal budgets by preparation symmetry

The acquisition example above uses initial storage `160*1023^2`. The
following reflection-fixed preparation has exactly the same budget under
the same law, but its symmetry excludes nonzero winding on the declared
cycle whenever principal increments are defined.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.sdk import relational_report_to_dict

graph = nx.cycle_graph(5)
for node, value in enumerate((8, 4, -8, -8, 4)):
    graph.nodes[node].update(EPI=1023 * value, theta=0, nu_f=1)
graph.graph["GAMMA"] = {"type": "none"}
model = RelationalExchangeModel(
    1, epi_weight=Fraction(1023, 1024),
    phase_weight=Fraction(1, 1024), phase_domain="regular",
)
source = bound_relational_sine_exchange(graph, reference_model=model)
symmetry = source.assess_cycle_symmetry(
    permutation_indices=(0, 4, 3, 2, 1), cycle=(0, 1, 2, 3, 4),
)
assert symmetry.initial_form_storage == 160 * 1023**2
assert symmetry.status == "certified"
assert symmetry.trajectory_symmetry_certified
assert symmetry.zero_winding_when_nonantipodal
assert symmetry.nonzero_winding_limit_excluded
payload = relational_report_to_dict(symmetry)
```

Supply an exact comparison and a permutation in its complete node order.
Inspect the symmetry and winding flags separately: the moving source is
neither certified stationary nor guaranteed to avoid antipodal edges.
Relative uncertainty reports are unsupported by this reader.

The [positive acquisition example](#certify-acquisition-from-initially-equal-phases)
already supplies the opposite outcome at this budget. Scalar resources alone
therefore do not determine the acquired geometry. See the
[contract](../../contracts/relational/SINE_PATTERNS.md#sine-cycle-symmetry) and
[equal-budget proof](../../../theory/nodal/SINE_PATTERN_DYNAMICS.md#sine-equal-budget-preparation)
for the full-state symmetry and orientation conditions.

### Classify long-time endpoints without selecting a basin

For two C5 rings connected through one intermediary, the complete sine law
has a finite exact set of relative equilibria. The following source has
arbitrary finite form and phase; the reader does not require a prepared twist:

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange

graph = nx.Graph()
graph.add_nodes_from(range(11))
cycles = (tuple(range(5)), tuple(range(5, 10)))
for cycle in cycles:
    nx.add_cycle(graph, cycle)
graph.add_edges_from(((0, 10), (5, 10)))
for node in graph:
    graph.nodes[node].update(EPI=node / 10, theta=node / 7, nu_f=1)
graph.nodes[6]["nu_f"] = 1.5
graph.nodes[9]["nu_f"] = 0.5

source = bound_relational_sine_exchange(
    graph, reference_model=RelationalExchangeModel(1, phase_domain="regular"),
)
result = source.asymptotic_equilibria(cycles=cycles)
assert result.admitted
assert result.single_relative_equilibrium_convergence_certified
assert result.full_lifted_state_convergence_certified
assert result.selected_equilibrium_status == "unavailable_no_basin_selection"

catalog = result.critical_set
assert len(catalog.cycle_edge_turn_options) == 30
assert catalog.relative_state_count == 3600
# Pick one symbolic catalog member, not the predicted endpoint of this source.
member = catalog.reconstruct(
    cycle_choices=(0, 0), bridge_turns=(0, Fraction(1, 2)),
)
assert not any(member.symbolic_sine_coefficients)
endpoint_evidence = result.to_dict()

# Classify a declared target under the source's law, not the source's basin.
twist = catalog.cycle_edge_turn_options.index((Fraction(1, 5),) * 5)
stable = result.classify_equilibrium(
    cycle_choices=(twist, twist), bridge_turns=(0, 0),
)
assert stable.local_exponential_attraction_certified
assert stable.relative_stable_modes == 20
assert stable.relative_unstable_modes == 0
opposed = result.classify_equilibrium(
    cycle_choices=(twist, twist), bridge_turns=(0, Fraction(1, 2)),
)
assert opposed.nonlinear_instability_certified
assert opposed.relative_unstable_modes == 1
assert catalog.phase_hessian_index_counts[0] == 9
assert sum(catalog.phase_hessian_index_counts) == 3600
stability_evidence = stable.to_dict()
```

Each cycle choice indexes its declared oriented ring. The two bridge choices
follow `catalog.bridge_edge_indices` in the stored full geometry; inspect
those indices rather than assuming label sorting or cycle order. Exact turns
in the reconstructed member represent multiples of mathematical `2*pi`.
They are distinct from the captured source's represented radian values.

The convergence theorem does not select the member, its terminal phase turns
or a numerical rate. The separate stability reader classifies each declared
target: nine are locally exponentially attracting modulo common origins;
the other 3,591 are unstable. This does not place the observed source in a
target's basin or supply a certified neighborhood radius. Adding a half-turn
on one bridge in this example creates one unstable direction in the complete
form-phase dynamics, even though both ring geometries remain individually
acute. The catalog keeps all nonacute branches. It stores 30 local
options and independent bridge choices, avoiding a list of 3,600 full states.
Zero loss or any zero capacity makes the stronger convergence verdict
unavailable. The [contract](../../contracts/relational/SINE_PATTERNS.md#sine-asymptotic-equilibria)
separates that admission from geometric classification and from native Arg
dynamics. Both evidence dictionaries can be saved using the SDK's
`export_to_json`.

### Declare the full-state receiver barrier experiment

The source-specific [receiver barrier protocol](../../contracts/relational/SINE_PATTERNS.md#frozen-receiver-barrier-exclusion)
combines a whole-time initial enclosure with a total-energy tail. Preparation
does not run the trajectory:

```python
from tnfr.research.relational_receiver_barrier import prepare_receiver_barrier

protocol = prepare_receiver_barrier()
assert len(protocol["nodes"]) == 11
assert protocol["maximum_steps"] == 256
```

The [benchmark](../../../benchmarks/relational_receiver_barrier.py) archives the
protocol with `--prepare`; the same command without that flag performs its
one fixed-budget evaluation. Use a fresh output path for a separately declared
regression, never overwrite the original response. A partial enclosure is
unavailable; a failed upper-bound comparison is unresolved. Neither establishes
receiver formation. The execution plan owns the retained scientific verdict.
