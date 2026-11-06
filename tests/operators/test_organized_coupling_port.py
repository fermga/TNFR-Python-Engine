"""Conditional marked-port admission and detached represented controls.

The source theorem retains its complete observation history and fine errors.
Constructed numerical boundary states do not authenticate acquired endpoints.
Stored sense indices and glyph metadata are explicit additional completions.
"""

import math
from collections import Counter
from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.config import inject_defaults
from tnfr.config.operator_names import U2_DEBT_CAPACITY
from tnfr.constants.aliases import ALIAS_SI
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, arg, cos, pi_interval, sin
from tnfr.metrics import phase_compatibility
from tnfr.operators import _coupling_stage_kernel as coupling
from tnfr.operators import _phase_gate as gate
from tnfr.operators import network_stage, word_execution
from tnfr.operators._argument_validation import require_list_sink
from tnfr.operators.definitions import Coupling
from tnfr.operators.definitions_base import Operator
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.physics import _organized_external_interaction as external
from tnfr.physics.relational_sine_regional import _branches
from tnfr.rng import resolve_graph_seed, validate_graph_seed
from tnfr.types import Glyph
from tnfr.utils import angle_diff
from tnfr.utils.cache import ensure_node_offset_map

Q_SOURCE = Q(1, 10**70)
ERROR = Q(1, 10**10)
BUDGET = Q(1, 2**40)
FORM_BOUND = Q(2, 10**10)


@pytest.fixture(scope="module", autouse=True)
def no_committed_event_or_scientific_replay():
    def forbidden(*args, **kwargs):
        pytest.fail("marked-port controls cannot commit events or integrate a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (network_stage, "_commit_coupling_structure"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


def _wrap(value):
    _, principal, _, _ = _branches((I.coerce(value),))
    assert principal is not None, "a regular exact branch is required by this control"
    return principal[0]


def _gap(first, second):
    return _wrap(I(Q(first) - Q(second)))


@pytest.fixture
def arithmetic_trace(monkeypatch, record_property):
    trace = {name: [] for name in ("angles", "consensus", "weights", "writes")}
    consensus = coupling.compute_consensus_phase
    link_weight = coupling._link_weight

    def checked_angle(first, second):
        result = angle_diff(first, second)
        discrepancy = (I(Q(result)) - _gap(first, second)).abs_max
        assert discrepancy < BUDGET
        trace["angles"].append(discrepancy)
        return result

    def checked_consensus(phases):
        exact = tuple(I(Q(value)) for value in phases)
        real = sum((cos(value) for value in exact), I(0))
        imaginary = sum((sin(value) for value in exact), I(0))
        reference = arg(real, imaginary)
        result = consensus(phases)
        discrepancy = _wrap(I(Q(result)) - reference).abs_max
        assert discrepancy < BUDGET
        trace["consensus"].append((tuple(phases), reference, discrepancy))
        return result

    def checked_weight(candidate, phases):
        result = link_weight(candidate, phases)
        x, y = Q(candidate.epi_source), Q(candidate.epi_target)
        regularizer = Q(coupling._EPI_SIMILARITY_EPSILON)
        form_similarity = 1 - abs(x - y) / (abs(x) + abs(y) + regularizer)
        sense_similarity = 1 - abs(Q(candidate.si_source) - Q(candidate.si_target))
        # Keep mathematical circular distance and the actual represented pi
        # denominator. Floating normalized-score arithmetic is checked whole.
        reference = (
            (
                1
                - abs(_gap(phases[candidate.source], phases[candidate.target]))
                / Q(math.pi)
            )
            / 2
            + form_similarity / 4
            + sense_similarity / 4
        )
        discrepancy = (I(Q(result)) - reference).abs_max
        assert discrepancy < BUDGET
        trace["weights"].append((result, reference, discrepancy))
        return result

    for module in (coupling, gate, phase_compatibility):
        monkeypatch.setattr(module, "angle_diff", checked_angle)
    monkeypatch.setattr(coupling, "compute_consensus_phase", checked_consensus)
    monkeypatch.setattr(coupling, "_link_weight", checked_weight)
    yield trace
    for name, values in trace.items():
        errors = (
            values if name in {"angles", "writes"} else [item[-1] for item in values]
        )
        record_property(name + "_count", len(values))
        record_property(name + "_maximum_error", str(max(errors, default=Q(0))))
    record_property("prospective_discrepancy_allowance", str(BUDGET))


@pytest.fixture(scope="module")
def source():
    return external._assess_organized_external_interaction(
        form_error_bounds=(ERROR,) * 6, phase_error_bounds=(ERROR,) * 6
    )


@pytest.fixture(scope="module")
def geometry():
    return external._derive_organized_external_interaction()


def _state(graph):
    return deepcopy(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))
    )


def _constructed(geometry, origin=0, host_sign=1, donor_sign=1):
    graph = nx.Graph()
    graph.add_nodes_from(range(54))
    graph.add_edges_from(
        ((i, j) for i, j, _ in geometry.internal_edges + geometry.outer_edges),
        weight=1.0,
        length=1.0,
    )
    inject_defaults(graph)
    graph.graph.update(
        RANDOM_SEED=17,
        _node_sample=(18,),
        UM_CANDIDATE_COUNT=0,
        UM_SYNC_VF=False,
        UM_STABILIZE_DNFR=False,
        GRAMMAR_REJECTION_MODE="raise",
        COLLECT_OPERATOR_METRICS=False,
        VALIDATE_NODAL_EQUATION=False,
    )
    beta = 0.6 if origin == 0 else 2 * math.pi / 5 - 0.6
    for node in graph:
        child, slot = divmod(node, 9)
        phase = 2 * math.pi * (slot - 4) / 9
        phase += 4 * (child - 2) / math.pi if child < 5 else beta
        form = (
            host_sign * FORM_BOUND
            if child == 2
            else -host_sign * FORM_BOUND if child == 0 else Q(0)
        )
        if node in (45, 46):
            form = donor_sign * Q_SOURCE / 2 * (1 if node == 45 else -1)
        graph.nodes[node].update(EPI=float(form), theta=phase, nu_f=1.0, delta_nfr=0.0)
        graph.nodes[node][ALIAS_SI[0]] = 1.0
    graph.nodes[45].update(
        glyph_history=("AL",),
        _grammar_u2_debt=0,
        _grammar_prior_coherence=False,
        _operator_step=1,
    )
    assert graph.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False) is False
    assert graph.graph.get("UM_BIDIRECTIONAL", True) is True
    return graph


def _precommit(graph):
    """Mirror the actual stage driver's read boundary; commit remains blocked."""
    before = _state(graph)
    keys = set(graph.__dict__)
    try:
        preflight = network_stage._preflight_two_phase_stage(
            graph, Coupling(), (45,), sequence_context=None, execution_kwargs={}
        )
        assert preflight.exact_stage and preflight.snapshot is not graph
        snapshot_state = _state(preflight.snapshot)
        # The shared detached reader deliberately omits these injected runtime
        # cache keys; every other supplied field must survive unchanged.
        omitted = {"JITTER_CACHE_SIZE", "_node_set_checksum_cache", "_node_list_cache"}
        assert set(before[0]) - set(snapshot_state[0]) == omitted
        assert not set(snapshot_state[0]) - set(before[0])
        assert snapshot_state[0] == {
            key: value for key, value in before[0].items() if key not in omitted
        }
        assert snapshot_state[1:] == before[1:]
        snapshot = preflight.snapshot
        factors = resolve_runtime_operator_factors(
            snapshot.graph.get("GLYPH_FACTORS"), Glyph.UM, snapshot.graph
        )
        require_list_sink(
            snapshot.graph, "recognized_coherence_patterns", operator="coupling"
        )
        assert validate_graph_seed(snapshot) == 17
        seed = resolve_graph_seed(snapshot)
        offsets = dict(ensure_node_offset_map(snapshot))
        assert offsets == {node: node for node in range(54)}
        logical_before = network_stage._detached_stage_graph(snapshot)
        stage = coupling.propose_coupling_stage(
            snapshot, (45,), factors, resolved_seed=seed, node_offsets=offsets
        )
        assert nx.utils.graphs_equal(
            network_stage._detached_stage_graph(snapshot), logical_before
        )
        return preflight, stage, factors
    finally:
        assert _state(graph) == before and set(graph.__dict__) == keys


def _phases(graph, stage=None):
    phases = {node: graph.nodes[node]["theta"] for node in graph}
    if stage is not None:
        for update in stage.node_updates:
            assert update.vf_after is None and update.dnfr_after is None
            if update.theta_after is not None:
                phases[update.node] = update.theta_after
    return phases


def _old_cycle_identity(phases, geometry):
    for cycle in geometry.internal_cycles + geometry.outer_cycles:
        differences = tuple(
            Q(phases[b]) - Q(phases[a]) for a, b in zip(cycle, cycle[1:] + cycle[:1])
        )
        assert sum(differences) == 0
        offsets, principal, winding, margin = _branches(
            tuple(I(value) for value in differences)
        )
        assert principal is not None and winding == -sum(offsets) == 1 and margin > 0


def _check_writes(graph, stage, factors, trace):
    assert len(trace["consensus"]) == 1
    inputs, reference, _ = trace["consensus"][0]
    target = stage.target_proposals[0]
    assert target.compatible_neighbors == (46, 53)
    assert inputs == tuple(graph.nodes[node]["theta"] for node in (45, 46, 53))
    gain = Q(factors["UM_theta_push"])
    assert 0 < gain < Q(1, 4)
    assert {update.node for update in stage.node_updates} == {45, 46, 53}
    final = _phases(graph, stage)
    for node in (45, 46, 53):
        initial = I(Q(graph.nodes[node]["theta"]))
        ideal = initial + gain * _wrap(reference - initial)
        discrepancy = _wrap(I(Q(final[node])) - ideal).abs_max
        assert discrepancy < BUDGET
        trace["writes"].append(discrepancy)
    assert all(
        final[node] == graph.nodes[node]["theta"]
        for node in range(54)
        if node not in (45, 46, 53)
    )
    return final


@pytest.mark.parametrize("origin", (0, 1))
@pytest.mark.parametrize("host_sign", (-1, 1))
@pytest.mark.parametrize("donor_sign", (-1, 1))
def test_constructed_boundary_state_admits_one_default_port(
    geometry, arithmetic_trace, record_property, origin, host_sign, donor_sign
):
    graph = _constructed(geometry, origin, host_sign, donor_sign)
    assert len(graph) == 54 and graph.number_of_edges() == 99
    assert all(not graph.has_edge(i, j) for i, j, _ in geometry.contact_edges)
    assert sum(Q(graph.nodes[node]["EPI"]) for node in graph) == 0
    _old_cycle_identity(_phases(graph), geometry)
    preflight, stage, factors = _precommit(graph)
    assert preflight.states_before == {} and preflight.window == 7
    final = _check_writes(graph, stage, factors, arithmetic_trace)
    _old_cycle_identity(final, geometry)
    assert tuple((edge.left, edge.right) for edge in stage.edges) == ((18, 45),)
    target = stage.target_proposals[0]
    assert tuple(candidate.target for candidate in target.link_candidates) == (18,)
    assert abs(_gap(final[45], final[18])).hi < Q(7, 10)
    weight = stage.edges[0].weight
    assert Q(weight) > Q(5, 6) > Q(target.compatibility_threshold) > Q(3, 4)
    assert Q(weight) != Q(1, 100000)
    assert arithmetic_trace["weights"][0][0] == weight
    record_property("actual_conductance", weight.hex())
    record_property(
        "canonical_compatibility_threshold", target.compatibility_threshold.hex()
    )
    record_property("old_cycles_retained", 15)
    record_property("proposed_marked_edges", 1)


@pytest.mark.parametrize("origin", (0, 1))
def test_unequal_sense_completion_omits_the_same_phase_admitted_port(
    geometry, arithmetic_trace, record_property, origin
):
    graph = _constructed(geometry, origin)
    graph.nodes[45][ALIAS_SI[0]] = 0.0
    _, stage, factors = _precommit(graph)
    final = _check_writes(graph, stage, factors, arithmetic_trace)
    _old_cycle_identity(final, geometry)
    assert tuple(
        candidate.target for candidate in stage.target_proposals[0].link_candidates
    ) == (18,)
    assert abs(_gap(final[45], final[18])).hi < Q(7, 10)
    assert stage.edges == ()
    score = arithmetic_trace["weights"][0][0]
    assert Q(score) <= Q(3, 4) < Q(stage.target_proposals[0].compatibility_threshold)
    record_property("omitted_marked_edges", 1)
    record_property("actual_score", score.hex())


@pytest.mark.parametrize("origin", (0, 1))
def test_strict_activity_policy_rejects_small_donor_before_proposal(geometry, origin):
    graph = _constructed(geometry, origin)
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    with pytest.raises(OperatorPreconditionError, match="EPI too low"):
        _precommit(graph)


@pytest.mark.parametrize("obstruction", ("empty_history", "excess_debt"))
def test_primary_coordinates_do_not_supply_runtime_grammar_context(
    geometry, obstruction
):
    graph = _constructed(geometry)
    if obstruction == "empty_history":
        graph.nodes[45].update(EPI=0.0, glyph_history=(), _operator_step=0)
        rule = "U1a"
    else:
        graph.nodes[45]["_grammar_u2_debt"] = U2_DEBT_CAPACITY + 1
        rule = "U2"
    with pytest.raises(StructuralGrammarError) as caught:
        _precommit(graph)
    assert caught.value.rule == rule


@pytest.mark.parametrize(
    "node,field,bad",
    (
        (18, "EPI", True),
        (18, "EPI", complex(1, 1)),
        (18, "EPI", math.nan),
        (18, ALIAS_SI[0], True),
        (18, ALIAS_SI[0], -1.0),
        (45, ALIAS_SI[0], math.inf),
    ),
)
def test_malformed_consumed_port_primitives_do_not_produce_a_stage(
    geometry, node, field, bad
):
    graph = _constructed(geometry)
    graph.nodes[node][field] = bad
    with pytest.raises((TypeError, ValueError)):
        _precommit(graph)


def test_nominal_reflection_and_complete_frozen_word_cancel_at_host18(source, geometry):
    """Exact incidence/oddness, not sampled evolution or interval midpoints."""
    nominal = source.host_acquisition.primitive_entry
    assert nominal.source.epi == tuple(Q(4092 * (c - 2)) for c in range(5))
    assert nominal.source.phase == (Q(0),) * 5
    assert nominal.source.capacity == (Q(1),) * 5
    assert nominal.reference_model.phase_domain == "regular"
    edges = {frozenset(edge) for edge in nominal.geometry.edges}
    reflection = tuple(4 - c for c in range(5))
    assert {frozenset(reflection[c] for c in edge) for edge in edges} == edges
    assert all(
        nominal.source.epi[reflection[c]] == -nominal.source.epi[c] for c in range(5)
    )
    # The unit incidence commutes with reflection. Combining this with odd
    # sine and a sign change preserves both complete rows, hence X2=Phi2=0
    # for the nominal unforced law. Actual source residuals need not reflect.
    laplacian = [[0] * 5 for _ in range(5)]
    for edge in edges:
        a, b = tuple(edge)
        laplacian[a][a] += 1
        laplacian[b][b] += 1
        laplacian[a][b] -= 1
        laplacian[b][a] -= 1
    assert all(
        laplacian[reflection[i]][reflection[j]] == laplacian[i][j]
        for i in range(5)
        for j in range(5)
    )
    protocol = source.observation_protocol
    assert protocol.marked_child_order == tuple(range(5))
    assert protocol.window_command_multipliers == (0, 1, -1, 0)
    assert protocol.event_multipliers == (1, -2, 1)
    assert protocol.probe_amplitude == Q(1, 10)
    host_edges = tuple(
        (a, b)
        for a, b, _ in geometry.internal_edges + geometry.outer_edges
        if max(a, b) < 45
    )
    neighbors = tuple(b if a == 18 else a for a, b in host_edges if 18 in (a, b))
    assert set(neighbors) == {9, 19, 26, 27}
    # Symbolic angles are (coefficient of Phi3=-Phi1, carrier alpha, epsilon).
    # Every nonzero angle has the same multiplicity as its exact negative.
    angles = Counter()
    shifts = {9: Counter(), 27: Counter()}
    for marked in protocol.marked_child_order:
        for command in protocol.window_command_multipliers:
            for node in neighbors:
                child, slot = divmod(node, 9)
                outer = -1 if child == 1 else 1 if child == 3 else 0
                carrier = ((slot + 4) % 9) - 4
                shift = command * (int(child == marked) - int(2 == marked))
                angles[(outer, carrier, shift)] += 1
                if node in shifts:
                    shifts[node][shift] += 1
    assert all(
        count == angles[tuple(-value for value in angle)]
        for angle, count in angles.items()
    )
    assert shifts[9] == shifts[27] == Counter({0: 16, 1: 2, -1: 2})
    # Thus each outer current has coefficient 16+4*cos(epsilon), and their
    # opposite base angles cancel. Frozen diffusion also vanishes at X2=0.
    assert sum(-1 if node == 9 else 1 if node == 27 else 0 for node in neighbors) == 0


def test_full_row_lipschitz_and_observation_transport_give_uniform_form_bound(
    source, geometry, record_property
):
    assert all(source.source_admission_flags)
    assert source.relative_preparation_dimension == 96
    host = source.host_acquisition
    radius_x, radius_theta = 4 * Q_SOURCE, 7 * Q_SOURCE
    assert host.full_form_error_upper_bound < radius_x
    assert host.full_phase_error_upper_bound < radius_theta
    initial_form = (
        max(box.abs_max for box in host.primitive_entry.endpoint_form_bounds) + radius_x
    )
    assert initial_form < Q(7, 5000)
    horizon = source.observation_protocol.total_elapsed_scaled_time
    assert horizon == Q(1, 5000)
    assert source.observation_protocol.elapsed_window_scaled_time == Q(1, 100000)
    gamma = source.geometry.gamma_bounds.hi
    assert gamma < Q(1, 3000)
    assert initial_form + gamma * horizon < Q(1401, 10**6)
    # Every old host row has degree4: ||L||inf=8, so quarter diffusion
    # has Lipschitz constant2. Sine's unit Lipschitz bound gives 2*gamma
    # for the phase row of F. The supplied jumps change phase commands only.
    degrees = [0] * 45
    for left, right, weight in geometry.internal_edges + geometry.outer_edges:
        if left < 45 and right < 45:
            assert weight == 1
            degrees[left] += 1
            degrees[right] += 1
    assert degrees == [4] * 45
    assert max(2 * degree for degree in degrees) / 4 == 2
    g, b, t = Q(1, 3000), Q(1401, 10**6), horizon
    form_drift = 2 * b + g
    phase_drift = 2 * g * b
    row_error_at_zero = 2 * radius_x + 2 * g * radius_theta
    row_error_slope = 2 * form_drift + 2 * g * phase_drift
    endpoint = radius_x + row_error_at_zero * t + row_error_slope * t**2 / 2
    assert (
        endpoint
        == radius_x
        + (2 * radius_x + 2 * g * radius_theta) * t
        + (2 * b + g + 2 * g**2 * b) * t**2
    )
    assert endpoint < FORM_BOUND
    record_property("host18_uniform_form_upper_bound", str(FORM_BOUND))
    record_property("complete_observation_windows", 20)


@pytest.mark.parametrize("origin", (0, 1))
def test_uniform_source_phase_and_score_allowances_are_strict(source, origin):
    pi = pi_interval()
    host = source.host_acquisition
    residual = (
        host.full_phase_error_upper_bound + source.carried_phase_drift_upper_bound
    )
    host18 = host.primitive_entry.endpoint_phase_bounds[2] + I(-residual, residual)
    assert host18 == source.outer_one_host_phase_offset_bounds[2]
    assert host18.abs_max < Q(1, 40000)
    beta = I(Q(3, 5)) if origin == 0 else 2 * pi / 5 - Q(3, 5)
    origin_box = beta + I(-Q(1, 1000), Q(1, 1000))
    assert (origin_box - host18).abs_max + 2 * Q_SOURCE < Q(7, 10)
    alpha = 2 * pi / 9
    assert 3 * alpha.lo / 4 - 4 * Q_SOURCE > Q(1, 2)
    assert 5 * alpha.hi / 4 + 4 * Q_SOURCE < 1
    # Three donor phasors lie within q of their symmetric nominal inputs.
    # The nominal resultant is 1+2*cos(alpha)>2; perturbation size<=3q.
    # Its angular error is less than2q, without discarding the tiny q terms.
    assert (1 + 2 * cos(alpha)).lo > 2
    assert 3 * Q_SOURCE / (2 - 3 * Q_SOURCE) < 2 * Q_SOURCE
    regularizer = Q(coupling._EPI_SIMILARITY_EPSILON)
    assert regularizer >= Q(1, 10**9)
    form_similarity = regularizer / (regularizer + Q_SOURCE + FORM_BOUND)
    assert form_similarity > Q(4, 5)
    lower_score = Q(3, 4) - Q(7, 10) / (2 * pi.lo) + form_similarity / 4
    assert lower_score > Q(5, 6)
    # Exact ideal threshold pi/(pi+1). Represented controls retain the actual
    # resolved value separately; the unequal-Si score cannot exceed3/4.
    assert Q(3, 4) < (pi / (pi + 1)).lo <= (pi / (pi + 1)).hi < Q(4, 5) < Q(5, 6)
    assert Q_SOURCE < Q(1, 20)  # default strict UM_MIN_EPI=0.05
