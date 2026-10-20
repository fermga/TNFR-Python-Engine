# Conservative sine regional and collective dynamics

Regional storage and currents, conservative winding acquisition and retention, compatible collective families, actual contact feedback and forecast-derived organization.

Part of [Regional and relational SDK workflow index](../REGIONAL_AND_RELATIONAL.md). Section links remain stable; hypotheses and model changes remain local to each result.

<a id="conservative-regional-winding"></a>

### Certify finite winding from conservative environmental exchange

This standalone example retains two cycles, an intermediary and a supplied
return edge. All phases start equal and the receiver starts with zero form;
only two environmental nodes carry a supplied form contrast.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import certify_sine_conservative_winding_entry
from tnfr.sdk import relational_report_to_dict

conservative_graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
conservative_graph.add_edges_from(((0, 10), (10, 5), (1, 6)))
conservative_graph.graph["GAMMA"] = {"type": "none"}
for node in conservative_graph:
    conservative_graph.nodes[node].update(
        EPI=24 if node == 10 else -24 if node == 1 else 0,
        theta=0, nu_f=1,
    )
conservative_source = bound_relational_sine_exchange(
    conservative_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
regional_entry = certify_sine_conservative_winding_entry(
    conservative_source, cycle=(5, 6, 7, 8, 9),
    scaled_window=(Fraction(1, 4), Fraction(1, 3)),
    edge_turn_offsets=(1, 0, 0, 0, 0),
)
assert regional_entry.acquisition_certified
assert regional_entry.initial_winding == 0
assert regional_entry.certified_winding == -1
assert regional_entry.branch_margin_lower_bound > 0
regional_storage = conservative_source.regional_storage_balance(region=(5, 6, 7, 8, 9))
assert regional_storage.regional_form_storage == 0
assert regional_storage.complement_form_storage == 864
assert regional_storage.boundary_form_storage == 576
entry_export = relational_report_to_dict(regional_entry)
storage_export = relational_report_to_dict(regional_storage)
```

The acquisition interval uses `tau=t/pi`; the snapshot storage ledger uses
original structural `t` for work rates. The certificate encloses the complete
nonlinear phase evolution throughout the interval without running a trajectory.
Its acquired winding is **nonacute**, so it supplies neither equilibrium nor
indefinite maintenance. The nonzero environmental form and connections are
preparation premises. The
[proof and controls](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#conservative-regional-winding-entry)
and [entry contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-conservative-winding-entry)
state that scope. The
[storage contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-regional-storage-balance)
distinguishes instantaneous power from integrated receiver work.

`python -m benchmarks.conservative_regional_winding` records a fresh analytic
certificate and source archive under `artifacts/research/`. It preserves
existing records and runs no trajectory or parameter search.

<a id="conservative-source-geometry"></a>

### Inspect a distributed source and certify acute entry

This standalone preparation changes the supplied support and environmental
form, retaining the conservative law. Each receiver node has one private
environmental leaf. The shared map shows which initial phase directions the
environment can supply; the entry certificate then bounds the complete
nonlinear evolution, not a linearized trajectory.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_entry import (
    analyze_sine_conservative_source_geometry,
    certify_sine_conservative_winding_entry,
)
from tnfr.sdk import relational_report_to_dict

distributed_graph = nx.cycle_graph(5)
distributed_graph.add_edges_from((i, 5 + i) for i in range(5))
distributed_graph.graph["GAMMA"] = {"type": "none"}
for node in distributed_graph:
    distributed_graph.nodes[node].update(
        EPI=0 if node < 5 else -30 * (node - 7), theta=0, nu_f=1,
    )
distributed_source = bound_relational_sine_exchange(
    distributed_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
source_geometry = analyze_sine_conservative_source_geometry(
    distributed_source, receiver=(0, 1, 2, 3, 4),
)
assert source_geometry.relative_source_rank == 4
acute_entry = certify_sine_conservative_winding_entry(
    distributed_source, cycle=(0, 1, 2, 3, 4),
    scaled_window=(Fraction(1, 8), Fraction(2, 15)),
    edge_turn_offsets=(0, 0, 0, 0, -1),
)
assert acute_entry.acute_acquisition_certified
assert acute_entry.certified_winding == 1
assert acute_entry.acute_margin_lower_bound > Fraction(1, 5)
geometry_payload = relational_report_to_dict(source_geometry)
```

The [analytic control](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#sine-conservative-source-geometry)
compares this preparation with a full-state reflection at equal storage,
storage partitions and conserved means. The latter cannot acquire acute
nonzero winding. The positive interval lasts `pi/120` in original structural
time; the source and contacts remain supplied, and this witness later exits
the acute sector. It is not a capture or maintenance certificate.

To create a separately named analytic record and source archive, use the same
instrument with the distributed declaration; existing files are protected:

```console
python -m benchmarks.conservative_regional_winding --declaration docs/assets/conservative_source_geometry/declaration.json --output artifacts/research/conservative_source_geometry/certificate-v1.json
```

<a id="conservative-handoff"></a>

### Distinguish acquired geometry from maintained relative motion

Continue the distributed-source example with one algebraic member of the
declared velocity-ramp class. This tests a general inequality; it does not
rerun or retune the earlier frozen response.

```python
from tnfr.physics.relational_sine_entry import assess_sine_conservative_handoff

handoff_graph = distributed_graph.copy()
for node in handoff_graph:
    handoff_graph.nodes[node]["EPI"] = 0 if node < 5 else -192 * (node - 7)
handoff_source = bound_relational_sine_exchange(
    handoff_graph, reference_model=distributed_source.reference_model,
)
handoff = assess_sine_conservative_handoff(handoff_source, cycle=range(5))
assert handoff.omega == 64
assert handoff.entry_acute_certified and handoff.entry_subbarrier_certified
assert handoff.forced_exit_certified
assert handoff.handoff_obstruction_certified
assert handoff.unavoidable_boundary_work_lower_bound > Fraction(1, 100)
handoff_payload = relational_report_to_dict(handoff)
```

The region genuinely acquires acute winding with internal storage below its
acute-face barrier. Nevertheless the retained environment imposes relative
phase velocities that force it out again. The positive work bound concerns
the first exit; it is not a prediction of net work at a later endpoint.
This does not rule out maintenance for another source. Compatible rhythms
mean control of actual relative phase evolution; they do not require all
phases to coincide or identify held capacity with angular frequency. See the
[proof](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#sine-conservative-handoff-obstruction)
and [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-conservative-handoff).

<a id="robust-conservative-passage"></a>

### Retain a brief acquired identity despite full-state preparation errors

Continue the preceding algebraic `handoff_source`, whose nominal form ramp
has `omega=64`. The existing entry reader can admit a nonzero error at every
form and phase coordinate and expose a full-state box at the entry time:

```python
from fractions import Fraction
from tnfr.physics.relational_sine_entry import certify_sine_conservative_winding_entry

passage = certify_sine_conservative_winding_entry(
    handoff_source,
    cycle=range(5),
    scaled_window=(Fraction(11, 512), Fraction(3, 128)),
    edge_turn_offsets=(0, 0, 0, 0, -1),
    source_error_bound=Fraction(1, 4096),
)
assert passage.source_initial_zero_winding_certified
assert passage.acute_acquisition_certified
assert passage.entry_box_contains_source_flow
assert passage.entry_box_acute_retention_certified
assert passage.scaled_retention_duration == Fraction(1, 512)
assert len(passage.entry_form_bounds) == len(passage.entry_phase_bounds) == 10
assert passage.entry_form_radius == Fraction(89, 4096)
assert passage.entry_phase_radius == Fraction(751, 1048576)
assert passage.acute_margin_lower_bound > Fraction(1, 16)
assert passage.entry_box_acute_margin_lower_bound > Fraction(1, 16)
passage_evidence = relational_report_to_dict(passage)
```

All initial states in the declared source box begin with zero winding.
Their actual endpoints lie inside the reported entry box, and **every**
state in that independent box retains acute winding `+1` for the next
`1/512` of scaled time. The environment continues to evolve. The box centers
are bounding references, not replacements for actual trajectories.

This is a robust short passage, consistent with the separate forced-exit
result. Its duration is not extended retention or an entry into the later
moving-family example. `initial_total_storage` describes the nominal center;
`initial_storage_bounds` instead covers the initial source uncertainty. That
source budget does not cover every independent point of the later box.
The [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-conservative-winding-entry)
and [proof](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#robust-conservative-passage)
keep those distinctions explicit. No trajectory is run by this example.

<a id="phase-offset-partition"></a>

### Describe a region's compatible internal and collective motion

This standalone example uses a supplied C5 receiver and one private leaf
per receiver node. It asks whether fixed internal phase offsets can coexist
with moving collective form and phase under the same complete conservative
law. The captured zero state supplies the law and support; it is not the
twisted family and is not evolved into it by this calculation.

```python
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_partition import assess_sine_phase_offset_partition

partition_graph = nx.cycle_graph(5)
partition_graph.add_edges_from((i, 5 + i) for i in range(5))
partition_graph.graph["GAMMA"] = {"type": "none"}
for node in partition_graph:
    partition_graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
partition_source = bound_relational_sine_exchange(
    partition_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
offset_turns = tuple(Fraction(i, 5) for i in range(5)) * 2
partition = assess_sine_phase_offset_partition(
    partition_source,
    blocks=(tuple(range(5)), tuple(range(5, 10))),
    phase_offset_turns=offset_turns,
)
assert partition.status == "certified"
assert partition.invariance_certified
collective_state = partition.evaluate(
    collective_form=(1, 0),
    collective_phase_turns=(0, Fraction(1, 8)),
)
assert collective_state.block_phase_rates == (Fraction(1, 3), Fraction(-1))
print(collective_state.block_form_rates)
```

The receiver's constituents retain their phase differences while the two
blocks have their own compensated common motion. These angular rates use
`tau=t/pi`; the declared phase inputs are turns, and held capacity is not
an angular frequency. The returned form rates enclose the existing sine
response at this independently supplied state. No time step is executed.

Along the exact family the receiver's internal storage stays constant and
its net internal regional power is zero, although collective coordinates
and contact dynamics remain active. This is a compatible pattern family,
not autonomous formation, attraction or a robustness certificate. A
phase-flat source outside the family cannot enter it exactly at finite time
under the unchanged smooth law; a finite-width handoff needs its own proof.
For another partition, `unavailable` means the equality checks did not
decide the criterion, whereas `excluded` proves at least one required
condition fails. See the
[contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-phase-offset-partition)
and [theory](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-phase-offset-partition).

<a id="weighted-phase-offset-partition"></a>

### Unequal capacities with an exact collective law

This standalone example uses a four-node star coupled to one external node.
The star's center and leaves have different degrees and capacities; their
ratios `nu_i/d_i` agree. Form exchange and phase exchange retain different
coupling coefficients. The calculation tests the complete family and then
evaluates one supplied collective state, without evolving the graph.

```python
from fractions import Fraction
import networkx as nx
from tnfr.constants.canonical import SHA_VF_FACTOR
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.physics.relational_sine_weighted_partition import (
    assess_sine_weighted_capacity_reset,
    assess_sine_weighted_phase_offset_partition,
)
from tnfr.sdk import relational_report_to_dict

graph = nx.Graph()
graph.add_nodes_from(range(5))
graph.add_weighted_edges_from((0, node, 3) for node in (1, 2, 3))
graph.add_weighted_edges_from((node, 4, 2) for node in range(4))
graph.graph["GAMMA"] = {"type": "none"}
capacities = (4, 2, 2, 2, 8)
for node, capacity in enumerate(capacities):
    graph.nodes[node].update(EPI=0, theta=0, nu_f=capacity)
source = bound_relational_sine_support_exchange(
    graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
weighted_partition = assess_sine_weighted_phase_offset_partition(
    source,
    blocks=((0, 1, 2, 3), (4,)),
    phase_offset_turns=(0, 0, 0, 0, 0),
)
assert weighted_partition.invariance_certified
state = weighted_partition.evaluate((Fraction(1, 4), 0), (0, Fraction(1, 8)))

# Supply represented primary-map outputs, without executing a grammatical event.
whole_block = tuple(float(v) * SHA_VF_FACTOR for v in capacities[:4]) + (8.0,)
one_member = (4.0 * SHA_VF_FACTOR, 2.0, 2.0, 2.0, 8.0)
preserved = assess_sine_weighted_capacity_reset(
    weighted_partition, capacity_after=whole_block,
)
broken = assess_sine_weighted_capacity_reset(
    weighted_partition, capacity_after=one_member,
)
assert preserved.preservation_certified
assert not broken.preservation_certified
exported = relational_report_to_dict(preserved)
print(state.block_form_rates)
```

Every rate here uses the original structural time `t`; phase inputs are
exact turns and phase rates are radian derivatives. The common-block reset
preserves the collective family while changing its relative motion. The
singleton reset leaves the current forms and phases in that family but
breaks the subsequent all-state closure condition. The chosen powers-of-two
capacities make the represented products share an exact ratio. Arbitrary
capacities require a fresh check even with one common configured factor.

The graph is a law/support anchor, not an acquired organization. No operator
history or occurrence has been supplied. A real SHA stage also needs its
fine targets, live admission, grammar and lifecycle context. Exact family
invariance does not certify attraction or resistance to perturbations.
See the [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-weighted-phase-offset-partition)
and [proof](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-weighted-phase-offset-partition).

<a id="weighted-star-transverse-band"></a>

### Bound infinitesimal internal deviations on a libration band

This standalone calculation studies the preceding weighted-star law as a
supplied mathematical family. It does not inspect or evolve the example's
graph. All three internal spatial modes are covered over the declared scale
and libration-amplitude band, using outward scalar bounds rather than a
trajectory or parameter scan.

```python
from fractions import Fraction
from tnfr.constants.canonical import SHA_VF_FACTOR
from tnfr.physics.relational_sine_weighted_transverse import (
    assess_sine_weighted_star_transverse_band,
)
from tnfr.sdk import relational_report_to_dict

transverse_band = assess_sine_weighted_star_transverse_band(
    capacity_scale_bounds=(Fraction(23, 25), 1),
    amplitude_bound=Fraction(1, 4),
)
assert transverse_band.linear_transverse_stability_certified
assert [(mode.laplacian_eigenvalue, mode.multiplicity)
        for mode in transverse_band.modes] == [(1, 2), (4, 1)]
assert all(mode.strict_trace_margin > 0 for mode in transverse_band.modes)

# Inspect a represented factor; this does not execute an operator.
sha_ratio = Fraction.from_float(float(SHA_VF_FACTOR))
assert Fraction(23, 25) <= sha_ratio <= 1
transverse_payload = relational_report_to_dict(transverse_band)
```

The default SHA factor lies inside this band, and its products with the
star's powers-of-two capacities preserve an exact common ratio. Each held
capacity law in the band has its own normalized clock and transverse
coefficients. The conclusion bounds infinitesimal deviations from the
collective family; it does not bound arbitrary finite deviations or the
phase difference between neighboring collective trajectories. At zero
amplitude, stability follows from constant oscillators; reported period
bounds include the limit as amplitude tends to zero. No spontaneous
formation or physical correspondence follows. See the
[contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-weighted-star-transverse-band)
and [proof](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-weighted-star-transverse-band).

<a id="weighted-star-state"></a>

### Retain the internal state of a collective NFR candidate

The four-node group and its fifth boundary node continue to obey the same
complete weighted law. This standalone example compares two supplied fine
states with the same collective means, complex phase moment and full
storage, but different moment derivatives. It evaluates instantaneous rows;
it runs no trajectory and executes no operator.

```python
from fractions import Fraction
from tnfr.physics.relational_sine_weighted_state import (
    assess_sine_weighted_star_state,
    reconstruct_sine_weighted_star_state,
)
from tnfr.sdk import relational_report_to_dict

internal_size = Fraction(1, 8)
phase_size = Fraction(1, 32)  # exact turns; the reader retains this lift
state_plus = assess_sine_weighted_star_state(
    form=(0, internal_size, -internal_size, 0, 0),
    phase_turns=(0, phase_size, -phase_size, 0, 0),
)
state_minus = assess_sine_weighted_star_state(
    form=(0, -internal_size, internal_size, 0, 0),
    phase_turns=state_plus.phase_turns,
)
assert state_plus.collective_form == state_minus.collective_form
assert state_plus.collective_phase_turns == state_minus.collective_phase_turns
assert state_plus.moment_real_expression == state_minus.moment_real_expression
assert state_plus.moment_imag_expression == state_minus.moment_imag_expression
assert state_plus.form_storage == state_minus.form_storage
assert state_plus.phase_storage_expression == state_minus.phase_storage_expression
assert state_plus.moment_real_rate_bounds.hi < 0
assert state_minus.moment_real_rate_bounds.lo > 0

reconstructed = reconstruct_sine_weighted_star_state(
    capacity_scale=state_plus.capacity_scale,
    collective_form=state_plus.collective_form,
    collective_phase_turns=state_plus.collective_phase_turns,
    internal_form=state_plus.internal_form,
    internal_phase_turns=state_plus.internal_phase_turns,
)
assert reconstructed.form == state_plus.form
assert reconstructed.phase_turns == state_plus.phase_turns
assert reconstructed.exact_row_reconstruction_certified
state_payload = relational_report_to_dict(reconstructed)
```

The difference is the association between internal form and phase. Retaining
the internal coordinates reconstructs both full rows exactly. This is a
sufficient continuous description, with ten independent coordinates including
the dynamic boundary, rather than a proof that a smaller autonomous NFR
description is impossible. The marked center and leaves remain associated.

Phase means use the supplied lifts and all phase rates are radians per
original structural time. Changing individual phase representatives requires
transporting the internal moment and collective gap together. Full storage
belongs to the declared fine model; its split relative to the coherent family can
have a negative remainder. The continuous report supplies neither measured
histories nor grammar context and cannot select or admit an operator. See
the [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-weighted-star-state)
and [complete state study](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-weighted-star-state-sufficiency).

<a id="weighted-star-retention"></a>

### Certify a finite-amplitude internal identity

Continue the supplied `state_plus` above. Its members need not have identical
forms or phases. This certificate uses the full model's conserved storage
to keep center-leaf form contrasts below `1/2` and circular phase gaps below
`pi/2`, with the fifth node continuing to evolve.

```python
from tnfr.physics.relational_sine_weighted_state import (
    assess_sine_weighted_star_retention,
)

retained_identity = assess_sine_weighted_star_retention(state_plus)
assert retained_identity.source_identity_admitted
assert retained_identity.storage_margin > 0
assert retained_identity.all_time_identity_retention_certified
retention_payload = relational_report_to_dict(retained_identity)
```

The claim holds for this declared identity under the unchanged conservative
law, without integrating a trajectory. It does not say that the state returns
to perfect synchrony, follows a particular orbit or formed spontaneously.
An `unavailable` result for another state means this sufficient storage
test did not certify maintenance; it does not prove that the group breaks
apart. The [proof](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-weighted-star-identity-retention)
also covers an explicit nonzero preparation neighborhood around the earlier
coherent band. See the
[contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-weighted-star-retention)
for thresholds, uncertainty scope and capacity-reset boundaries.

<a id="weighted-star-contact"></a>

### Connect two maintained organizations with accounted support work

This standalone example supplies two complete stars and one boundary-to-boundary
edge. It evaluates the actual joined law and its maintenance certificate;
it does not run a trajectory or execute the Coupling operator.

```python
from fractions import Fraction
from tnfr.physics.relational_sine_weighted_state import (
    assess_sine_weighted_star_state,
)
from tnfr.physics.relational_sine_weighted_contact import (
    assess_sine_weighted_star_contact,
)
from tnfr.sdk import relational_report_to_dict

contact_forms = (Fraction(1, 16),) * 4 + (0,)
contact_left = assess_sine_weighted_star_state(
    form=contact_forms, phase_turns=(0,) * 5,
)
contact_right = assess_sine_weighted_star_state(
    form=contact_forms, phase_turns=(Fraction(1, 16),) * 5,
)
star_contact = assess_sine_weighted_star_contact(contact_left, contact_right)
assert star_contact.source_individual_retention_certified == (True, True)
assert star_contact.support_work_bounds.lo > 0
assert star_contact.component_storage_rate_bounds[0].hi < 0
assert star_contact.component_storage_rate_bounds[1].lo > 0
assert star_contact.all_time_joint_identity_retention_certified
contact_payload = relational_report_to_dict(star_contact)
```

The two components initially exchange stored structure while both declared
identities remain certified for all time under the joined law. Initial
exchange signs are not a prediction of their signs for all later times.
The bridge adds positive storage; it is not a free or autonomously selected
connection. Both boundary nodes have five neighbors, so their rates
must be read from `star_contact`, not from the isolated inputs.

The joint storage budget is an additional obligation. With higher joined
storage, both isolated certificates can pass while the joint certificate
is unavailable; that result does not establish breakup. The
[proof](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-weighted-star-maintenance-composition)
also admits finite perturbations in every one of the twenty coordinates.
See the [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-weighted-star-contact)
for complete-law, export and execution boundaries.

<a id="weighted-star-canonical-contact"></a>

### Execute a canonical word and assess its actual successor

This standalone example prepares two weighted stars, then applies a
validated `emission,coupling,silence` word only to the first boundary.
The phases below are exact dyadic radians. Source preparation and event
invocation are supplied; the example does not acquire the stars from an
earlier trajectory. UTC lifecycle timestamps come from the actual runtime.

```python
import math
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.operators.definitions import Emission, Coupling, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.physics.relational_sine_star_retention import (
    assess_sine_weighted_star_support_retention,
)
from tnfr.sdk import relational_report_to_dict

word_graph = nx.Graph()
word_graph.add_nodes_from(range(10))
star_edges = ((0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4))
star_weights = (3, 3, 3, 2, 2, 2, 2)
for offset in (0, 5):
    for (i, j), weight in zip(star_edges, star_weights):
        word_graph.add_edge(offset + i, offset + j, weight=weight, length=1)
    for local, capacity in enumerate((4, 2, 2, 2, 8)):
        word_graph.nodes[offset + local].update(
            EPI=1/4 + (1/16 if local < 4 else 0),
            theta=0.0 if offset == 0 else 201/512,
            nu_f=float(capacity), delta_nfr=0.0, Si=0.5, glyph_history=[],
        )
word_graph.graph.update(
    _t=7, RANDOM_SEED=17, GAMMA={"type": "none"},
    GRAMMAR_REJECTION_MODE="raise", VALIDATE_OPERATOR_PRECONDITIONS=True,
    EPI_LATENT_MAX=1, VF_BASAL_THRESHOLD=1/8, SHA_MIN_VF=1/8,
    UM_MIN_EPI=1/8, UM_MIN_VF=1/8, MIN_NETWORK_DEGREE_COUPLING=1,
    DELTA_PHI_MAX=math.pi/2, UM_MAX_PHASE_DIFF=math.pi/2,
    UM_BIDIRECTIONAL=True, UM_SYNC_VF=True,
    UM_STABILIZE_DNFR=True, UM_FUNCTIONAL_LINKS=True,
    UM_COMPAT_THRESHOLD=3/4, UM_CANDIDATE_COUNT=0,
    UM_CANDIDATE_MODE="proximity", GLYPH_HYSTERESIS_WINDOW=8,
    EPI_MIN=-1, EPI_MAX=1, CLIP_MODE="hard",
    GLYPH_FACTORS={
        "AL_boost": 1/1024, "UM_theta_push": 1/1024,
        "UM_vf_sync": 1/8, "UM_dnfr_reduction": 1/8,
        "SHA_vf_factor": 1/2,
    },
)
word_operators = (Emission(), Coupling(), Silence())
word_context = {"compatibility_profile": "core"}
assert validate_sequence(
    [operator.name for operator in word_operators], context=word_context,
).passed
word_sequence = ValidatedSequence(word_operators, context=word_context)
for index, operator in enumerate(word_operators):
    execute_network_operator_stage(
        word_graph, operator, (4,), sequence_context=word_sequence.step(index),
    )

successor = bound_relational_sine_support_exchange(
    word_graph,
    reference_model=RelationalExchangeModel(1, epi_weight=0, phase_domain="regular"),
)
word_retention = assess_sine_weighted_star_support_retention(
    successor, stars=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),
)
assert word_graph.number_of_edges() == 19
assert word_graph.nodes[4]["nu_f"] == 117/32
assert tuple(word_graph.nodes[4]["glyph_history"]) == ("AL", "UM", "SHA")
assert word_retention.all_time_identity_retention_certified
word_retention_payload = relational_report_to_dict(word_retention)
```

UM selects five actual cross edges and their affinity-derived weights.
The complete word changes the target capacity, leaving the original star
subgraphs intact. The certificate uses all actual edges and capacities;
the right star's perfect internal synchrony need not persist even though
its declared identity does. Positive capacity after SHA still permits
continuous form evolution.

The detached `successor` recomputes pressure for the declared sine law;
it does not install that pressure or a flow law in `word_graph`. Stored
operator pressure, event history and measured temporal samples are
different inputs. No continuous integration occurs here. See the
[prospective study and proof](../../../theory/nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md)
and [successor contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-weighted-star-support-retention).

<a id="weighted-star-finite-response"></a>

### Bound the internal response after a finite duration

Continue the preceding canonical-word example. The complete successor law
has a nonzero right center-leaf derivative at the initial state. A finite
response requires control of the subsequent nonlinear evolution as well:

```python
from fractions import Fraction

finite_response = successor.finite_form_response_bounds(duration=Fraction(1, 128))
right_center = finite_response.source.nodes.index(5)
right_leaf = finite_response.source.nodes.index(6)
finite_contrast = (
    finite_response.endpoint_form_bounds[right_center]
    - finite_response.endpoint_form_bounds[right_leaf]
)
assert finite_contrast.hi < -Fraction(1, 32768)
assert word_retention.all_time_identity_retention_certified
finite_response_payload = relational_report_to_dict(finite_response)
```

The duration is elapsed original structural time after the word. This
calculation bounds the full nonlinear law without integrating a trajectory
or modifying `word_graph`. The reusable reader derives a whole-window form
radius from the global speed bound, then reuses the complete acceleration
envelope. Its endpoint intervals include the resulting nonlinear remainder.

The [matched comparison](../../../theory/nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md#weighted-star-finite-response-protocol)
changes only the word's functional-link flag. The disconnected right star
retains exactly zero center-leaf contrast by its invariant-family theorem;
zero initial derivatives alone would not prove that. Both alternatives
retain their original identities. This is a conditional structural response,
not a measured laboratory signal or a newly formed NFR. See the
[finite-response contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#sine-finite-form-response).

<a id="weighted-star-collective-state"></a>

### Retain the contact's internal motion in an exact collective state

Continue the canonical-contact and finite-response examples. The left four
members can share one form and phase. On the right, the center must remain
separate from the three leaves. The following five-block family retains
both dynamical boundaries and the right center-leaf contrast:

```python
from tnfr.physics.relational_sine_weighted_partition import (
    assess_sine_weighted_phase_offset_partition,
)

contact_partition = assess_sine_weighted_phase_offset_partition(
    successor,
    blocks=((0, 1, 2, 3), (4,), (5,), (6, 7, 8), (9,)),
    phase_offset_turns=(0,) * 10,
)
assert contact_partition.invariance_certified
contact_state = contact_partition.observe_radian_state()
assert contact_state.source_membership_certified
assert contact_state.full_row_equality_certified
assert contact_state.collective_phase_radians == (
    0, 0, Fraction(201, 512), Fraction(201, 512), Fraction(201, 512),
)
assert contact_state.partition.block_mean_weights == (
    4, Fraction(32, 13), Fraction(5, 4), Fraction(9, 2), Fraction(5, 8),
)
assert contact_state.phase_rate_residuals == (0,) * 10
assert contact_state.full_source == successor
assert contact_state.full_source == finite_response.source
contact_state_payload = relational_report_to_dict(contact_state)
```

The ten collective coordinates describe this invariant family exactly;
lifting them reconstructs its twenty fine form/phase coordinates and both
complete rate rows. The preceding finite-response certificate therefore
keeps the same source and observable. This statement does not discard errors
outside the family or prove that they decay.

Use `contact_partition.evaluate_radians(forms, phases)` to examine a supplied
member with five real forms and five real radian lifts. The observer above
instead verifies actual source membership without averaging. Both require
exact zero offsets; neither converts the source phases to rational turns.

The reduced law keeps separate cross-edge multiplicities and conductance
sums. A simple five-node graph with one unweighted phase edge per connection
would lose that information. This mathematical description also installs
no collective operator, event selector or new NFR identity. See the
[proof](../../../theory/nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md#weighted-star-collective-state-result)
and [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-weighted-radian-state).

<a id="weighted-star-off-family-robustness"></a>

### Bound errors outside the exact contact family

Continue the preceding examples. Allow independent errors in every form and
declared radian lift, with the same errors supplied to the attached and
disconnected branches. Keep each branch's actual post-word law fixed:

```python
epsilon = Fraction(1, 2**20)
contact_errors = successor.initial_state_perturbation_bounds(
    duration=Fraction(1, 128),
    form_error_bounds=(epsilon,) * 10,
    phase_error_bounds=(epsilon,) * 10,
)
assert contact_errors.status == "enclosed"
assert contact_errors.form_error_supremum_upper_bound < 21 * epsilon / 20
assert contact_errors.phase_error_supremum_upper_bound < 11 * epsilon / 10

# The matched-word theorem gives the same primary node states without links.
# These detached component graphs do not execute another word or a trajectory.
control_sources = tuple(
    bound_relational_sine_support_exchange(
        word_graph.subgraph(nodes).copy(), reference_model=successor.reference_model,
    )
    for nodes in ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))
)
control_errors = tuple(
    source.initial_state_perturbation_bounds(
        duration=Fraction(1, 128),
        form_error_bounds=(epsilon,) * 5,
        phase_error_bounds=(epsilon,) * 5,
    )
    for source in control_sources
)
assert all(report.status == "enclosed" for report in control_errors)
assert control_sources[1].nodes == (5, 6, 7, 8, 9)
attached_errors = contact_errors.whole_window_form_error_bounds
right_control_errors = control_errors[1].whole_window_form_error_bounds
robust_contrast_upper = (
    finite_contrast.hi + attached_errors[5] + attached_errors[6]
    + right_control_errors[0] + right_control_errors[1]
)
assert robust_contrast_upper < -Fraction(1, 32768)
assert contact_errors.initial_storage_upper_bound < Fraction(35, 64)
assert all(
    report.initial_storage_upper_bound < Fraction(35, 64)
    for report in control_errors
)
contact_errors_payload = relational_report_to_dict(contact_errors)
```

The strict inequality holds for the whole declared error box, not just a
chosen perturbed sample. The unperturbed control is exactly null by its
family theorem; the perturbed control has its own nonzero error allowance.
The attached reference is the exact five-block solution, while its perturbed
neighbors retain all twenty coordinates.

The same protected old edges, positive capacities, initial identity margins
and complete storage bounds also retain both marked identities. This is
finite tracking of a prepared neighborhood, without a claim that errors
decay or that UM produces the entire neighborhood. Changing the input to the
word could change its support, weights and capacities and needs separate
admission. See the [proof](../../../theory/nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md#weighted-star-off-family-robustness-result)
and [error-envelope contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#sine-initial-state-perturbation).

<a id="weighted-star-preparation-robustness"></a>

### Continue with uncertainty in the produced edge weights

Continue the canonical-contact examples. A preparation error before UM can
change its output weights even when the same five links remain admitted.
The preparation theorem supplies the following state and conductance budgets
under its explicit arithmetic premises. This example evaluates that
continuation box; it does not execute a perturbed word or verify an actual
endpoint's membership in the box.

```python
preparation_epsilon = Fraction(1, 2**24)
inserted_edges = {frozenset((4, node)) for node in range(5, 10)}
prepared_contact_bounds = successor.state_and_conductance_perturbation_bounds(
    duration=Fraction(1, 128),
    form_error_bounds=(2 * preparation_epsilon,) * 10,
    phase_error_bounds=(2 * preparation_epsilon,) * 10,
    conductance_error_bounds=tuple(
        4 * preparation_epsilon if frozenset(edge) in inserted_edges else 0
        for edge in successor.edges
    ),
)
assert prepared_contact_bounds.status == "enclosed"
assert prepared_contact_bounds.max_phase_law_forcing_upper_bound < preparation_epsilon
assert (
    prepared_contact_bounds.form_error_supremum_upper_bound
    < 21 * preparation_epsilon / 10
)
prepared_control_bounds = tuple(
    source.initial_state_perturbation_bounds(
        duration=Fraction(1, 128),
        form_error_bounds=(2 * preparation_epsilon,) * 5,
        phase_error_bounds=(2 * preparation_epsilon,) * 5,
    )
    for source in control_sources
)
prepared_contrast_upper = finite_contrast.hi + 2 * (
    prepared_contact_bounds.form_error_supremum_upper_bound
    + prepared_control_bounds[1].form_error_supremum_upper_bound
)
assert prepared_contrast_upper < -Fraction(1, 32768)
assert prepared_contact_bounds.initial_storage_upper_bound < Fraction(35, 64)
assert all(
    report.initial_storage_upper_bound < Fraction(35, 64)
    for report in prepared_control_bounds
)
prepared_contact_payload = relational_report_to_dict(prepared_contact_bounds)
```

Actual use also requires checking support, capacities, every weight and both
coordinate budgets against the captured endpoint. Keep raw phases and an
explicit integer lift chart: an output near a full turn is compared using
`raw_phase - 2*pi*k`, with outward pi intervals. A subtraction using rounded
`math.tau` would change the exact continuous state. These checks are separate
from the box reader and are exercised by the frozen static word controls.

The uniform implementation statement retains a declared numerical allowance;
finite comparisons do not prove every backend's arithmetic accuracy. See the
[preparation theorem and controls](../../../theory/nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md#weighted-star-preparation-robustness-result)
and [state/conductance contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#sine-state-conductance-perturbation).

<a id="weighted-star-relative-realization"></a>

### Find the internal state required by relative regional motion

Continue the canonical-contact example. The following reader studies the
held successor law around its separately declared uniform equilibrium family.
It retains the captured source, without asserting that its finite post-word
state has reached that equilibrium:

```python
from tnfr.physics.relational_sine_consensus_observation import (
    assess_sine_consensus_observation,
)

contact_linear = assess_sine_consensus_observation(
    successor, regions=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),
    max_rank_calls=128,
)
assert contact_linear.source == successor
assert not contact_linear.source_in_consensus_family
assert contact_linear.clock == "t"
assert not contact_linear.two_coordinate_closure_certified
assert contact_linear.linear_observation.full_state_dimension == 20
assert contact_linear.linear_observation.output_rank == 2
assert contact_linear.linear_observation.dimension == 8
assert contact_linear.linear_observation.extra_coordinates == 6
contact_linear_payload = relational_report_to_dict(contact_linear)
```

The outputs are the left-minus-right form and phase means, weighted with
each region's actual `d/nu_f`. Six further linear coordinates are required
to predict those two outputs for arbitrary fine infinitesimal preparations.
The exact selected-coordinate dynamics uses
`s_dot=(exchange_coefficient/pi)*linear_observation.reduced_generator*s`.
The common factor remains in the original clock; the stored rational matrix
alone is not the original-time generator.

The matched disconnected law instead preserves its own two regional means.
Its degree weights differ, so it must be rebuilt before comparison. This
eight-coordinate result concerns the complete equilibrium tangent, not an
exact nonlinear projection of arbitrary fine states. Eliminating retained
internal coordinates requires their initial contribution and causal memory;
setting them to zero would restrict the preparation. See the
[proof and frozen controls](../../../theory/nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md#weighted-star-relative-realization-result)
and [contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#sine-consensus-observation).

<a id="weighted-star-nonlinear-validity"></a>

### Bound the nonlinear error of the collective predictor

Keep the nominal held law, but explicitly prepare a new near-consensus
source. The earlier finite post-word source is outside this box; this
replacement is a detached preparation, not its subsequent trajectory or
an operator reset. Three right leaves carry a zero-mean internal phase
pattern that the entire eight-state tangent observation cannot see:

```python
from dataclasses import replace

epsilon = Fraction(1, 16)
hidden_amplitude = epsilon / 4
hidden_source = replace(
    successor,
    epi=(Fraction(0),) * 10,
    phase=(Fraction(0),) * 6 + (
        hidden_amplitude, hidden_amplitude, -2 * hidden_amplitude, Fraction(0),
    ),
)
hidden_linear = assess_sine_consensus_observation(
    hidden_source, regions=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)),
    max_rank_calls=128,
)
nonlinear_bounds = hidden_linear.nonlinear_response_bounds(
    duration=Fraction(1, 16), consensus_form=0, consensus_phase=0,
    form_amplitude_bounds=(epsilon,) * 10,
    phase_amplitude_bounds=(epsilon,) * 10,
)
assert nonlinear_bounds.status == "enclosed"
assert nonlinear_bounds.initial_observation_state == (Fraction(0),) * 8
assert nonlinear_bounds.initial_outputs == (Fraction(0),) * 2
assert nonlinear_bounds.output_error_upper_bounds[0] < 5 * epsilon**3
assert nonlinear_bounds.output_error_upper_bounds[1] < 3 * epsilon**3
nonlinear_payload = relational_report_to_dict(nonlinear_bounds)
```

These are whole-window bounds for all twenty independently prepared fine
coordinates in the declared box, against the tangent solution with the
same complete initial state. The selected linear coordinates alone still
do not close the nonlinear law: reversing the hidden phase pattern keeps
them, both means and full storage unchanged, but reverses a nonzero initial
relative form rate. That derivative statement is not a measured finite-time
response. The [proof](../../../theory/nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md#weighted-star-nonlinear-validity-result)
also establishes both original identity barriers throughout this box;
neither result establishes formation of a common identity. See the
[contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#sine-consensus-nonlinear-response).
Before using these reports in a formation study, apply the
[source and mechanism admission](../../../theory/nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md#weighted-star-formation-admission).
It identifies which preparation each certificate covers and what separate
entry and event evidence the proposed common identity would require.

<a id="weighted-star-permanent-registration"></a>

### Admit the recurrence boundary before a permanent formation claim

Return to the actual canonical `successor`, keeping its complete held
weighted law. The following family admits its captured storage and
own-law form mean:

```python
contact_recurrence = successor.recurrence_bounds(
    energy_ceiling=Fraction(3, 4), form_mean_bounds=(-1, 1),
)
assert contact_recurrence.snapshot_family_membership == "inside"
assert contact_recurrence.divergence == 0
assert contact_recurrence.positive_conductance_weight_floor > 0
assert contact_recurrence.almost_everywhere_recurrence_certified
assert contact_recurrence.individual_recurrence_status == "unavailable_for_chosen_state"
recurrence_payload = relational_report_to_dict(contact_recurrence)
```

The family theorem is about almost every state in ambient real-form and
circular-phase volume. It does not establish recurrence of this particular
successor or a measure on an exact energy surface. The report itself also
evaluates no registration target. The
[separate proof](../../../theory/nodal/SINE_WEIGHTED_STAR_CANONICAL_CONTACT.md#weighted-star-permanent-registration-result)
declares such a closed target, excludes the source with a uniform margin,
and applies recurrence to its positive-width fine-state neighborhoods.
This obstructs robust permanent capture under the unchanged conservative
flow. Finite common-registration episodes require their own entry,
lifetime and subsequent-response evidence. See the
[contract](../../contracts/relational/SINE_COMPARISON_AND_INFERENCE.md#sine-support-recurrence).

For finite entry, use the
[separate registration comparison](../../../theory/nodal/SINE_WEIGHTED_STAR_FINITE_IDENTITY.md#weighted-star-finite-registration-result).
The disconnected control's conserved component means exclude the fixed
circular target for all time. The joined law's source storage and two means
permit a target configuration, but a configuration path is not a trajectory.
No registration observer or formation verdict is supplied by the recurrence
report. A later episode must carry the actual reached full state into its
residence and response calculation; a tangent comparison tests the role of
nonlinear terms, not a universal prerequisite for an NFR.
The [admitted short-arc formula](../../../theory/nodal/SINE_WEIGHTED_STAR_FINITE_IDENTITY.md#weighted-star-registration-short-arc)
also replaces global circle minimization by the full arithmetic-centered
phase variance on the retained common lifts. Rewrapping each node separately
does not preserve that raw-variance formula.

<a id="moving-pattern-window"></a>

### Retain compatible motion with errors in every fine node

Continue the [phase-offset partition example](#phase-offset-partition).
Here the reference has nonzero collective exchange and zero conserved
weighted origins. Independent errors affect all receiver and environmental
forms and phases. The reader proves a finite window around this supplied
reference; it does not evolve the earlier zero-state source into the family.

```python
from tnfr.physics.relational_sine_partition import assess_sine_moving_pattern_window

moving_reference = partition.evaluate(
    collective_form=(Fraction(1, 20), Fraction(-3, 20)),
    collective_phase_turns=(Fraction(-2, 5), Fraction(-2, 5)),
)
window_parameters = dict(
    form_error_bounds=(Fraction(1, 1000),) * 10,
    phase_error_bounds=(Fraction(1, 1000),) * 10,
    scaled_horizon=5,
    receiver_phase_radius=Fraction(1, 10),
    contact_phase_radius=Fraction(1, 10),
    reference_contact_bound=Fraction(1, 4),
)
moving_window = assess_sine_moving_pattern_window(
    moving_reference, **window_parameters,
)
assert moving_window.whole_window_retention_certified
assert moving_window.phase_flat_acquisition_status == "not_excluded"
assert moving_window.retention_margin_lower_bound > 0

zero_pulse_reference = partition.evaluate(
    collective_form=(0, 0),
    collective_phase_turns=(Fraction(-2, 5), Fraction(-2, 5)),
)
zero_pulse_window = assess_sine_moving_pattern_window(
    zero_pulse_reference, **window_parameters,
)
assert zero_pulse_window.whole_window_retention_certified
assert zero_pulse_window.phase_flat_acquisition_status == "excluded"
assert zero_pulse_window.initial_full_storage_bounds.hi < Fraction(7, 2)
```

The horizon `5` is in `tau`, or `5pi` in the original structural clock.
Phase uncertainty and all three phase bounds are radians, whereas the
reference phase coordinates above are turns. The certificate retains acute
receiver winding, internal form/phase storage and full contact error bounds
while the environment moves. It also keeps uncertainty in the conserved
origins; relative edge errors alone would discard those coordinates.

Both examples establish maintenance of a prepared neighborhood. The
zero-pulse neighborhood has too little total storage to have been acquired
from phase-flat preparation under this unchanged law. The moving example
avoids that particular obstruction; `not_excluded` does not prove formation.
Its small full-storage interval also excludes a handoff from the earlier
large-energy ramp sources. Their environmental state cannot be discarded
to make the budgets agree.

The same estimate holds for the preceding finite window, so the central
state is not the first appearance of the identity. An acquisition claim
needs an earlier identity-absent preparation, compatible conserved storage
and independent evidence that its actual full state reaches this
neighborhood. See the [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-moving-pattern-window)
and [theory](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-moving-pattern-window).

<a id="collective-pulse-balance"></a>

### Observe transfer between collective contact motion and internal structure

This standalone example starts with flat phase and zero receiver winding.
The supplied form dipole breaks the exact uniform contact-pulse family.
The reader examines its actual full-state response at the initial instant;
it does not substitute a prepared invariant pattern or run a trajectory.

```python
from dataclasses import replace
from fractions import Fraction
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_partition import observe_sine_collective_pulse

pulse_graph = nx.cycle_graph(5)
pulse_graph.add_edges_from((i, 5 + i) for i in range(5))
pulse_graph.graph["GAMMA"] = {"type": "none"}
u, epsilon = Fraction(119, 100), Fraction(1, 20)
for node in pulse_graph:
    form = u / 4 if node < 5 else -3 * u / 4
    if node == 0:
        form += epsilon
    elif node == 1:
        form -= epsilon
    pulse_graph.nodes[node].update(EPI=form, theta=0, nu_f=1)
pulse_anchor = bound_relational_sine_exchange(
    pulse_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
pulse_source = replace(
    pulse_anchor,
    epi=tuple(pulse_graph.nodes[node]["EPI"] for node in pulse_anchor.nodes),
)
pulse_balance = observe_sine_collective_pulse(
    pulse_source, cycle=range(5), contact_turn_offsets=(0,) * 5,
)
assert pulse_balance.form_gap == u
assert pulse_balance.flat_phase_jet_available
assert pulse_balance.first_three_energy_derivatives == (0, 0, 0)
assert pulse_balance.contact_rate_variance == Fraction(1, 180)
assert pulse_balance.contact_rate_third_central_moment == 0
assert pulse_balance.fourth_energy_derivative == 4 * u * u / 135
assert pulse_balance.full_storage_bounds.contains(Fraction(14201, 4000))
```

The detached replacement explicitly supplies the rational preparation in
the anchor's node order, avoiding graph capture's floating materialization.
The observer re-admits those primitives and recomputes all consumed rates
and storage; cached derived fields in the anchor are not used as evidence.

The positive fourth derivative shows the first nonzero transfer goes toward
the collective contact pulse. It does not predict later receiver organization.
The collective pulse is not separately conserved away from the exact family:
its feedback depends on the five actual contact-phase deviations. The
reported remainder `H-5h` is signed unless extra convexity is established.

The integer offsets declare continuous contact lifts; independently wrapping
each phase difference before averaging would change this observation. A
snapshot cannot recover a lost branch history. All reported derivatives use
`tau=t/pi`, and no numerical evaluation horizon follows from the local jet.

Although this source's total storage exceeds the necessary phase-flat path
barrier, it is below the sharper lower bound of the preceding moving-pattern
example's fixed error box. That particular handoff is excluded. Testing
finite acute identity and certifying a subsequent full-state maintenance
window remain separate tasks, each retaining the source's actual conserved
budget. See the [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-collective-pulse-transfer)
and [derivation](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-collective-pulse-transfer).

<a id="relative-phase-feedback"></a>

### Distinguish momentary phase compensation from maintained geometry

The same observer reports the internal and contact contributions to receiver
relative phase velocity, together with its acceleration and third derivative.
This exact rational control reuses `pulse_anchor` from the preceding example.
The environment cancels the receiver's initial relative velocity and
acceleration, but the full law produces a nonzero third derivative.

```python
compensated_forms = (1, -1, 0, 0, 0, 4, -4, 1, 0, -1)
compensated_source = replace(
    pulse_anchor,
    epi=tuple(Fraction(compensated_forms[node]) for node in pulse_anchor.nodes),
    phase=(Fraction(0),) * len(pulse_anchor.nodes),
)
compensated = observe_sine_collective_pulse(
    compensated_source, cycle=range(5), contact_turn_offsets=(0,) * 5,
)
assert compensated.receiver_relative_phase_rates == (0,) * 5
assert all(
    internal + contact == 0
    for internal, contact in zip(
        compensated.receiver_internal_phase_rates,
        compensated.receiver_contact_phase_rates,
    )
)
assert all(
    bound.contains(0)
    for bound in compensated.receiver_relative_phase_acceleration_bounds
)
expected_jerk = (Fraction(22, 9), -Fraction(22, 9), 1, 0, -1)
assert all(
    bound.contains(value)
    for bound, value in zip(
        compensated.receiver_relative_phase_jerk_bounds, expected_jerk,
    )
)
assert compensated.receiver_relative_phase_jerk_bounds[0].lo > 0
```

The [algebraic counterexample](../../../theory/nodal/SINE_COLLECTIVE_PHASE_DYNAMICS.md#sine-relative-phase-feedback)
proves the zero derivatives and nonzero cubic drift; interval overlap alone
would not prove either equality. This flat-phase example tests feedback, not
formation. The theorem also gives an exactly twisted symbolic counterpart,
without treating a floating approximation to pi as exact geometry.
SDK report export includes these fields. Neither this local
calculation nor exact rigidity classification certifies the extended
formation-to-retention window; that requires control over the whole interval.

<a id="conservative-source-admission"></a>

### Exclude unsuitable conservative sources before running a trajectory

These two readers assess different necessary conditions. The standalone
example keeps contact phases nonuniform and retains every private leaf.
Exact rational values are explicitly supplied after graph capture.

```python
from dataclasses import replace
from fractions import Fraction as Q
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_partition import assess_sine_contact_averaging
from tnfr.physics.relational_sine_regional import assess_sine_cycle_barrier
from tnfr.sdk import relational_report_to_dict

admission_graph = nx.cycle_graph(5)
admission_graph.add_edges_from((i, i + 5) for i in range(5))
admission_graph.graph["GAMMA"] = {"type": "none"}
for node in admission_graph:
    admission_graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
admission_anchor = bound_relational_sine_exchange(
    admission_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
contact_phases = tuple(Q(1, 4) + Q(sign, 10) for sign in (1, 1, -1, -1, 0))
low_source = replace(
    admission_anchor,
    epi=(Q(1, 4),) * 5 + (Q(-3, 4),) * 5,
    phase=(Q(0),) * 5 + contact_phases,
)
barrier = assess_sine_cycle_barrier(low_source, cycle=range(5))
assert barrier.full_storage_bounds.hi < Q(7, 2)
assert all(sector.acute_acquisition_excluded for sector in barrier.sectors)

rapid_source = replace(low_source, epi=(Q(5),) * 5 + (Q(-15),) * 5)
averaging = assess_sine_contact_averaging(
    rapid_source, cycle=range(5), scaled_horizon=1,
)
assert averaging.initial_full_storage_bounds.lo >= 1000
assert averaging.whole_window_acute_winding_excluded
admission_evidence = relational_report_to_dict(averaging)
```

The first exclusion follows from conservation for all times; the second
covers only `0 <= tau <= 1` (`0 <= t <= pi`). Increasing a uniform form gap
does not by itself create useful directed transfer. Neither reader evolves
the graph, chooses a preparation or rules out every possible coherent
pattern. Their [contracts](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-cycle-sector-barrier)
retain unavailable bounds and distinguish wider-sector protection from
strict acute retention.

<a id="reversible-preparation"></a>

### Connect an absent identity to a retained region

Use `assess_sine_reversible_preparation` when an admitted conservative
checkpoint already has an energy-speed retention window and a validated
complete-law forecast starts from its reversed forms. It consumes the
existing `SineForecast` and `SineExchangeComparison`; it does not run another
solver. Supply the ordered receiver cycle, exact positive source and target
radii, and scaled retention duration. The source center is constructed from
each endpoint, with its enclosure error retained in the forward-inclusion
bound. A reversed midpoint by itself does not certify a preparation.

The [shared instrument](../../../benchmarks/conservative_regional_organization.py)
supports this workflow through a declaration whose `evaluation_kind` is
`reversible_preparation`. Freeze the declaration and source archive with
`--prepare` before the single evaluation. The declaration owns the fixed
grid, numerical budget, target, source-radius rule and first-success rule;
the [research plan](../../../theory/research/FIVE_STAGE_EXECUTION_PLAN.md#current-g3-gate)
owns permission to schedule a new candidate. Reading a saved response does
not authorize rerunning or tuning it.

Inspect `outcome`, `horizon_complete`, `selected_step_index` and all `steps`.
Only `certified` establishes the conditional source-to-retention inclusion.
`unavailable` and `no_certificate_on_declared_grid` preserve different
coverage limits. These are mathematical preparation certificates, not
measurements or autonomous formation laws. The
[contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-reversible-preparation)
and [proof](../../../theory/nodal/SINE_CONSERVATIVE_PREPARATION.md#sine-reversible-preparation)
own the full hypotheses and interpretation.

<a id="energy-speed-retention"></a>

### Certify a whole retention window without a moving reference

This standalone example supplies a target with independent uncertainty in
all twenty form/phase coordinates. It preserves the complete conservative
C5/private-leaf law. The rational phases below are actual declared centers,
not an assertion that a rounded angle is the exact uniform twist.

```python
from dataclasses import replace
from fractions import Fraction as Q
import networkx as nx
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_comparison import bound_relational_sine_exchange
from tnfr.physics.relational_sine_regional import assess_sine_cycle_retention
from tnfr.sdk import relational_report_to_dict

target_graph = nx.cycle_graph(5)
target_graph.add_edges_from((i, i + 5) for i in range(5))
target_graph.graph["GAMMA"] = {"type": "none"}
for node in target_graph:
    target_graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
target_anchor = bound_relational_sine_exchange(
    target_graph,
    reference_model=RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular",
    ),
)
u, kappa = Q(7, 50), Q(1, 128)
q = tuple(kappa * value for value in (1, -1, 0, 0, 0))
lq = tuple(kappa * value for value in (3, -3, 1, 0, -1))
forms = tuple(u / 4 + value for value in q) + tuple(
    -3 * u / 4 + value + laplacian for value, laplacian in zip(q, lq)
)
phases = tuple((i - 2) * Q(1256637, 10**6) for i in range(5)) * 2
target = replace(
    target_anchor,
    epi=tuple(forms[node] for node in target_anchor.nodes),
    phase=tuple(phases[node] for node in target_anchor.nodes),
)
retention = assess_sine_cycle_retention(
    target, cycle=range(5), scaled_duration=1,
    source_error_bound=Q(1, 4096),
)
assert retention.initial_winding == 1
assert retention.full_storage_bounds.lo > Q(7, 2)
assert retention.whole_window_retention_certified
assert retention.retention_margin_lower_bound > Q(1, 100)
retention_evidence = relational_report_to_dict(retention)
```

Every state in this exact-radius target retains acute unit winding for
`-1 <= tau <= 1`. The environment remains live and the center has nonuniform
form; no rigid reference is substituted for its trajectory. This does not
show how an earlier zero-winding state reaches the target. Its full energy
avoids one necessary exclusion, while directed entry remains the research
obligation. The [contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-cycle-retention)
distinguishes whole-window guarantees from unavailable bounds.

<a id="conservative-regional-phase-transport"></a>

### Inspect propagation into the receiver interior

Continue the [eleven-node winding example](#conservative-regional-winding)
with its `conservative_source`. The contrast below observes the
phase separation between non-port nodes `7` and `9`. Source-node order is
explicit; the coefficients select an observation and do not alter the law.

```python
from tnfr.physics.relational_sine_entry import analyze_sine_conservative_phase_transport

interior_contrast = tuple(
    int(node == 9) - int(node == 7) for node in conservative_source.nodes
)
phase_transport = analyze_sine_conservative_phase_transport(
    conservative_source, contrasts=(interior_contrast,),
)
assert phase_transport.contrast_initial_velocity == (Fraction(0),)
assert phase_transport.contrast_initial_jerk == (Fraction(-80, 3),)
assert phase_transport.contrast_acceleration_bounds == (Fraction(3),)
assert phase_transport.contrast_quadratic_remainder_coefficients == (Fraction(3, 2),)
transport_export = relational_report_to_dict(phase_transport)
```

The interior starts without relative phase velocity but has a nonzero third
phase derivative. This is a local propagation mechanism. The separate global
bound gives `abs(theta_9-theta_7)<=3*tau^2/2` for this preparation; it does not
give a nonzero lower bound or prove later organization. The
[proof](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#conservative-regional-phase-transport)
explains why a fully acute unit winding requires `tau^2>pi/3` on this support.
The [regional work criterion](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#conservative-regional-work-retention)
adds the form-storage and future boundary-work obligations needed for its
stronger finite-retention conclusion. The early winding witness does not
already satisfy them. See the
[API contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-conservative-phase-transport).

<a id="conservative-regional-forecast"></a>

### Observe a continuous conservative regional forecast

This inexpensive stationary control illustrates the shared solver and observer.
It is independent of the reserved environmental formation experiment.

```python
from fractions import Fraction
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.physics.relational_sine_forecast import bound_sine_flow
from tnfr.physics.relational_sine_regional import assess_sine_regional_organization
from tnfr.sdk import relational_report_to_dict

cycle_neighbors = tuple(((i - 1) % 5, (i + 1) % 5) for i in range(5))
stationary_forecast = bound_sine_flow(
    (0,) * 10 + (1,), neighbors=cycle_neighbors,
    visible_capacity=(1,) * 4,
    model=RelationalExchangeModel(1, epi_weight=0, phase_domain="regular"),
    observation_time=0, end_time=Fraction(1, 16),
    time_step=Fraction(1, 16), order=2,
)
regional_observation = assess_sine_regional_organization(
    stationary_forecast, cycle_indices=(0, 1, 2, 3, 4),
    minimum_duration=Fraction(1, 32), acute_margin=Fraction(1, 16),
)
assert regional_observation.horizon_complete
assert regional_observation.outcome == "acute_winding_excluded_on_horizon"
assert regional_observation.endpoint_integrated_regional_work.contains(0)
regional_export = relational_report_to_dict(regional_observation)
```

The observer uses whole-time tubes, original structural `t` and the complete
network rates. The stationary state has zero winding throughout; this control
does not exclude organization from another preparation. A failed enclosure
remains unresolved rather than becoming a negative physical result. See the
[contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-regional-organization).

Reuse that same forecast to separate internal conversion from boundary inputs,
without another solver call:

```python
from tnfr.physics.relational_sine_regional import assess_sine_regional_channels

channel_history = assess_sine_regional_channels(
    stationary_forecast, cycle_indices=(0, 1, 2, 3, 4),
)
assert channel_history.horizon_complete
assert channel_history.initial_phase_flat_certified
assert channel_history.acute_entry_excluded_on_validated_prefix
assert channel_history.cumulative_internal_conversion.contains(0)
channel_export = relational_report_to_dict(channel_history)
```

For a nonstationary source, positive internal conversion transfers phase
storage into form. Boundary inputs to the separate stores differ from the
conjugate work components of the total regional ledger. The running `7/2`
barrier is necessary for acute unit-winding entry from flat receiver phase;
it is not sufficient entry or a ban on every nonacute winding. See the
[channel contract](../../contracts/relational/SINE_REGIONAL_DYNAMICS.md#sine-regional-channel-history).

The [frozen environmental control](../../../theory/nodal/SINE_REGIONAL_FORMATION.md#conservative-regional-organization-control)
has a supplied nonuniform environment and no prescribed future input. Its
preparation and evaluation are already closed; the
[benchmark lifecycle](../../../benchmarks/README.md#freeze-before-evaluating-a-reserved-response)
owns any separately declared new evaluation. To analyze its channel transfers under
current analysis code, retain the original response and use a separate output:

```sh
python -m benchmarks.conservative_regional_organization --analyze-channels docs/assets/conservative_regional_organization/response-v1.json --output artifacts/research/conservative_regional_organization/channel-analysis-v1.json
```

This reads and verifies saved evidence without replaying the forecast. It
records the original file hashes, a separate analysis source archive and
retrospective results. It does not alter the original prediction or verdict.
