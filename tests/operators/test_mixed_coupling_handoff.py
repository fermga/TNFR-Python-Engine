"""Acquired-source obstructions and conditional, uncommitted UM proposals.

The recovered source is a theorem about reached-state sets. The binary64
graphs below are explicitly constructed controls, not authenticated endpoints.
Their sense indices, stored pressure, selection policy and seed are supplied.
No operator event, trajectory, historical word or successor capture is run.
"""

import math
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.config import inject_defaults
from tnfr.constants.aliases import ALIAS_SI
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, arg, cos, pi_interval, sin
from tnfr.metrics import phase_compatibility
from tnfr.operators import _coupling_stage_kernel as coupling
from tnfr.operators import _phase_gate as phase_gate
from tnfr.operators import network_stage, word_execution
from tnfr.operators._argument_validation import require_list_sink
from tnfr.operators.definitions import Coupling
from tnfr.operators.definitions_base import Operator
from tnfr.operators.factor_contracts import resolve_runtime_operator_factors
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.preconditions import (
    OperatorPreconditionError,
    _coupling_activity_policy,
    validate_coupling,
)
from tnfr.physics import _sine_mixed_capture as source_owner
from tnfr.physics.relational_sine_regional import _branches
from tnfr.rng import resolve_graph_seed, validate_graph_seed
from tnfr.types import Glyph
from tnfr.utils import angle_diff
from tnfr.utils.cache import ensure_node_offset_map

ERROR = Q(1, 10**10)
EPSILON = Q(1, 10**70)
ETA = 20 * EPSILON
DELTA = Q(1, 1024)
REPRESENTATION_BUDGET = Q(1, 2**40)
FORM_UNIT = Q(1, 2**240)


@pytest.fixture(scope="module", autouse=True)
def no_event_or_source_replay():
    def forbidden(*args, **kwargs):
        pytest.fail("mixed-source controls cannot commit or replay a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for owner, name in (
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
            patch.setattr(owner, name, forbidden)
        yield


@pytest.fixture(scope="module")
def source():
    # Only geometry, original recoveries and isolated waiting are consumed.
    # Its supplied degree-three joined-capture verdict is not transferred.
    return source_owner._assess_sine_mixed_capture(
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        observation_error_bound=Q(1, 10**6),
    )


def _wrap(value):
    _, principal, _, _ = _branches((I.coerce(value),))
    assert principal is not None
    return principal[0]


def _gap(first, second):
    return _wrap(I(Q(first) - Q(second)))


def _state(graph):
    return deepcopy(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))
    )


def _constructed(geometry, *, split_sense=False):
    graph = nx.Graph()
    graph.add_nodes_from(geometry.nodes)
    graph.add_edges_from(
        ((a, b) for a, b, _ in geometry.internal_edges), weight=1.0, length=1.0
    )
    inject_defaults(graph)
    graph.graph.update(
        RANDOM_SEED=17,
        GRAMMAR_REJECTION_MODE="raise",
        COLLECT_OPERATOR_METRICS=False,
        VALIDATE_NODAL_EQUATION=False,
    )
    for node, turn in enumerate(geometry.target_phase_turns):
        ring, slot = divmod(node, 9)
        form = FORM_UNIT * (1 if slot < 8 else -8)
        origin = (-DELTA / 2, DELTA / 2)[ring]
        graph.nodes[node].update(
            EPI=float(form),
            theta=2 * math.pi * float(turn) + float(origin),
            nu_f=1.0,
            delta_nfr=0.0,
        )
        graph.nodes[node][ALIAS_SI[0]] = float(ring if split_sense else 0)
    return graph


@contextmanager
def _checked_arithmetic():
    """Independent outward arithmetic for these represented calls only."""
    trace = {key: [] for key in ("angles", "consensus", "weights", "writes")}
    original_consensus = coupling.compute_consensus_phase
    original_weight = coupling._link_weight

    def checked_angle(first, second):
        result = angle_diff(first, second)
        error = (I(Q(result)) - _gap(first, second)).abs_max
        assert error < REPRESENTATION_BUDGET
        trace["angles"].append(error)
        return result

    def checked_consensus(phases):
        values = tuple(I(Q(value)) for value in phases)
        real = sum((cos(value) for value in values), I(0))
        imaginary = sum((sin(value) for value in values), I(0))
        reference = arg(real, imaginary)
        result = original_consensus(phases)
        error = _wrap(I(Q(result)) - reference).abs_max
        assert error < REPRESENTATION_BUDGET
        trace["consensus"].append((tuple(phases), reference, error))
        return result

    def checked_weight(candidate, phases):
        result = original_weight(candidate, phases)
        x, y = Q(candidate.epi_source), Q(candidate.epi_target)
        epsilon = Q(coupling._EPI_SIMILARITY_EPSILON)
        form_similarity = 1 - abs(x - y) / (abs(x) + abs(y) + epsilon)
        sense_similarity = 1 - abs(Q(candidate.si_source) - Q(candidate.si_target))
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
        error = (I(Q(result)) - reference).abs_max
        assert error < REPRESENTATION_BUDGET
        trace["weights"].append((candidate, result, reference, error))
        return result

    with pytest.MonkeyPatch.context() as patch:
        for owner in (coupling, phase_gate, phase_compatibility):
            patch.setattr(owner, "angle_diff", checked_angle)
        patch.setattr(coupling, "compute_consensus_phase", checked_consensus)
        patch.setattr(coupling, "_link_weight", checked_weight)
        yield trace


def _preflight(graph, **kwargs):
    before = _state(graph)
    try:
        result = network_stage._preflight_two_phase_stage(
            graph,
            Coupling(),
            tuple(range(18)),
            sequence_context=None,
            execution_kwargs=kwargs,
        )
        assert result.exact_stage and result.snapshot is not graph
        snapshot = _state(result.snapshot)
        omitted = {"JITTER_CACHE_SIZE", "_node_set_checksum_cache", "_node_list_cache"}
        assert set(before[0]) - set(snapshot[0]) == omitted
        assert snapshot[0] == {
            key: value for key, value in before[0].items() if key not in omitted
        }
        assert snapshot[1:] == before[1:]
        return result
    finally:
        assert _state(graph) == before


def _proposal(graph):
    before = _state(graph)
    with _checked_arithmetic() as trace:
        snapshot = _preflight(graph).snapshot
        factors = resolve_runtime_operator_factors(
            snapshot.graph.get("GLYPH_FACTORS"), Glyph.UM, snapshot.graph
        )
        require_list_sink(
            snapshot.graph, "recognized_coherence_patterns", operator="UM"
        )
        assert validate_graph_seed(snapshot) == 17
        seed = resolve_graph_seed(snapshot)
        offsets = dict(ensure_node_offset_map(snapshot))
        assert offsets == {node: node for node in range(18)}
        logical_before = network_stage._detached_stage_graph(snapshot)
        stage = coupling.propose_coupling_stage(
            snapshot,
            tuple(range(18)),
            factors,
            resolved_seed=seed,
            node_offsets=offsets,
        )
        assert nx.utils.graphs_equal(
            network_stage._detached_stage_graph(snapshot), logical_before
        )
        _check_merged_writes(graph, stage, factors, trace)
    assert _state(graph) == before
    return graph, stage, factors, trace


def _phases(graph, stage):
    result = {node: graph.nodes[node]["theta"] for node in graph}
    for update in stage.node_updates:
        assert update.theta_after is not None
        result[update.node] = update.theta_after
    return result


def _check_merged_writes(graph, stage, factors, trace):
    assert len(trace["consensus"]) == 18
    gain = Q(factors["UM_theta_push"])
    assert 0 < gain < Q(1, 4)
    references = {}
    for proposal, (inputs, reference, _) in zip(
        stage.target_proposals, trace["consensus"], strict=True
    ):
        assert set(proposal.compatible_neighbors) == set(graph.neighbors(proposal.node))
        assert inputs == tuple(
            graph.nodes[node]["theta"]
            for node in (proposal.node, *proposal.compatible_neighbors)
        )
        references[proposal.node] = reference
    for update in stage.node_updates:
        node = update.node
        initial = I(Q(graph.nodes[node]["theta"]))
        contributors = (node, *graph.neighbors(node))
        ideal = (
            initial
            + gain
            * sum((_wrap(references[other] - initial) for other in contributors), I(0))
            / 3
        )
        error = _wrap(I(Q(update.theta_after)) - ideal).abs_max
        assert error < REPRESENTATION_BUDGET
        trace["writes"].append(error)


@pytest.fixture(scope="module")
def proposals(source):
    return tuple(
        _proposal(_constructed(source.geometry, split_sense=split))
        for split in (False, True)
    )


def _expected_edges(geometry, *, split_sense):
    result = set()
    for ring, step in ((0, 2), (1, 4)):
        for j in range(9):
            result.add(tuple(sorted((9 * ring + j, 9 * ring + (j + step) % 9))))
    if not split_sense:
        for a in range(9):
            for b in range(9, 18):
                turns = geometry.target_phase_turns[b] - geometry.target_phase_turns[a]
                turns = (turns + Q(1, 2)) % 1 - Q(1, 2)
                if abs(turns) <= Q(2, 9):
                    result.add((a, b))
    return result


def _nominal_work(*, split_sense):
    pi = pi_interval()
    first, second = cos(2 * pi / 9), cos(4 * pi / 9)
    chords = 9 * (2 - first - second)
    if split_sense:
        return chords
    return chords + 9 * (5 - cos(I(DELTA)) * (1 + 2 * first + 2 * second))


def _work(graph, stage):
    phases = _phases(graph, stage)
    old_before = sum(
        (
            1 - cos(I(Q(graph.nodes[b]["theta"]) - Q(graph.nodes[a]["theta"])))
            for a, b in graph.edges
        ),
        I(0),
    )
    old_after = sum(
        [1 - cos(I(Q(phases[b]) - Q(phases[a]))) for a, b in graph.edges], I(0)
    )
    new_potential = sum(
        (
            1 - cos(I(Q(phases[edge.right]) - Q(phases[edge.left])))
            for edge in stage.edges
        ),
        I(0),
    )
    new_kinetic = sum(
        Q(edge.weight)
        * (Q(graph.nodes[edge.left]["EPI"]) - Q(graph.nodes[edge.right]["EPI"])) ** 2
        / 2
        for edge in stage.edges
    )
    return old_after - old_before + new_potential + new_kinetic, new_kinetic


def test_fresh_actual_source_preserves_both_classes_and_isolated_waiting(source):
    assert source.source_certified and source.isolated_waiting_certified
    assert source.form_error_bound == source.phase_error_bound == ERROR
    assert source.recovered_radius == EPSILON and source.waiting_radius == ETA
    assert source.geometry.class_order == (1, 2)
    assert source.geometry.held_capacities == (Q(1),) * 18
    assert len(source.geometry.internal_edges) == 18
    assert all(weight == 1 for _, _, weight in source.geometry.internal_edges)
    for winding, recovery in zip((1, 2), source.recoveries, strict=True):
        assert recovery.target_winding == winding
        assert recovery.source_certified and recovery.recovery_certified
        assert recovery.formation_time == 100
        assert recovery.relaxation_duration == 10**13
        assert recovery.endpoint_radius == EPSILON
        assert recovery.endpoint_form_norm_squared_upper_bound < EPSILON**2
        assert recovery.endpoint_phase_norm_squared_upper_bound < EPSILON**2


def test_source_admission_obstructions_are_scoped_not_a_u3_failure(
    source, record_property
):
    assert source.source_certified and source.isolated_waiting_certified
    pi = pi_interval()
    initial_u3_margin = pi.lo / 18 - 2 * ETA
    assert initial_u3_margin > 0
    assert pi.lo / 2 - DELTA - 2 * ETA > 0
    minimum_form, minimum_capacity = _coupling_activity_policy({})
    assert minimum_form == 0.05 and minimum_capacity == 0.01
    assert Q(minimum_form) - ETA > 0 and 1 > Q(minimum_capacity)
    lower_profile = (1 - DELTA,) + (Q(5, 4) - DELTA,) * 8
    assert sum(weight > 1 for weight in lower_profile) == 8
    assert min(lower_profile[1:]) - 1 == Q(255, 1024) > 0
    record_property("actual_source_U3_margin_lower", str(initial_u3_margin))
    record_property("selected_default_activity_margin", str(Q(minimum_form) - ETA))
    record_property("new_link_upper_bound_obstruction", str(Q(255, 1024)))


def test_real_source_neighborhood_has_uniform_reset_score_and_work_margins(source):
    assert source.isolated_waiting_certified
    pi = pi_interval()
    alpha = 2 * pi / 9
    for angle in (alpha, 2 * alpha):
        assert (1 + 2 * cos(angle)).lo > 1
    assert 3 * ETA / (1 - 3 * ETA) < 6 * ETA
    gain = 1 / (pi + 1)
    assert gain.hi < Q(1, 4)
    assert (1 + 7 * gain.hi) * ETA < 3 * ETA
    assert pi.lo / 18 - 6 * ETA > 0
    assert 2 * pi.lo / 3 - DELTA - 2 * ETA > pi.hi / 2
    epsilon = Q(1, 10**9)
    lowest_score = 1 - (4 * pi.hi / 9 + DELTA + 6 * ETA) / (2 * pi.lo)
    lowest_score -= ETA / (2 * epsilon)
    threshold = pi / (pi + 1)
    assert Q(3, 4) < threshold.lo <= threshold.hi < lowest_score
    cosine_identity = cos(alpha) + cos(2 * alpha) - cos(pi / 9)
    assert cosine_identity.lo <= 0 <= cosine_identity.hi
    work_error = 522 * ETA + 126 * ETA**2
    assert work_error < 1
    assert _nominal_work(split_sense=False).lo > 27
    assert _nominal_work(split_sense=True).lo > 9
    assert _nominal_work(split_sense=False).lo - work_error > 26
    assert _nominal_work(split_sense=True).lo - work_error > 8


def test_constructed_primary_state_declares_completion_without_invented_history(source):
    geometry = source.geometry
    for split in (False, True):
        graph = _constructed(geometry, split_sense=split)
        assert tuple(graph) == tuple(range(18))
        assert nx.number_connected_components(graph) == 2
        assert graph.number_of_edges() == 18
        assert all(degree == 2 for _, degree in graph.degree)
        assert graph.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False) is False
        assert graph.graph.get("UM_CANDIDATE_COUNT", 0) == 0
        assert graph.graph.get("_node_sample") is None
        for key in (
            "UM_BIDIRECTIONAL",
            "UM_SYNC_VF",
            "UM_STABILIZE_DNFR",
            "UM_FUNCTIONAL_LINKS",
        ):
            assert graph.graph.get(key, True) is True
        for ring in range(2):
            forms = tuple(Q(graph.nodes[9 * ring + j]["EPI"]) for j in range(9))
            assert all(value != 0 for value in forms) and sum(forms) == 0
            assert sum(value**2 for value in forms) < ETA**2
        for node in graph:
            data = graph.nodes[node]
            assert data["nu_f"] == 1 and data["delta_nfr"] == 0
            assert data[ALIAS_SI[0]] == (node // 9 if split else 0)
            assert "glyph_history" not in data and "_operator_step" not in data


def test_common_sense_proposes_sixty_three_edges_and_degree_nine(
    source, proposals, record_property
):
    graph, stage, _, _ = proposals[0]
    actual = {(edge.left, edge.right) for edge in stage.edges}
    assert actual == _expected_edges(source.geometry, split_sense=False)
    assert len(actual) == 63
    assert sum(a < 9 <= b for a, b in actual) == 45
    assert all(not graph.has_edge(*edge) for edge in actual)
    degrees = Counter(node for edge in (*graph.edges, *actual) for node in edge)
    assert tuple(degrees[node] for node in range(18)) == (9,) * 18
    assert all(0 <= edge.weight <= 1 for edge in stage.edges)
    assert len(stage.target_proposals) == len(stage.node_updates) == 18
    assert all(
        len(proposal.link_candidates) == 7 for proposal in stage.target_proposals
    )
    record_property("common_sense_new_edges", len(actual))
    record_property("common_sense_total_edges", graph.number_of_edges() + len(actual))
    record_property("common_sense_uniform_degree", 9)


def test_split_sense_omits_all_cross_edges_but_adds_eighteen_chords(
    source, proposals, record_property
):
    graph, stage, _, trace = proposals[1]
    actual = {(edge.left, edge.right) for edge in stage.edges}
    assert actual == _expected_edges(source.geometry, split_sense=True)
    assert len(actual) == 18 and all(a // 9 == b // 9 for a, b in actual)
    degrees = Counter(node for edge in (*graph.edges, *actual) for node in edge)
    assert tuple(degrees[node] for node in range(18)) == (4,) * 18
    cross = [
        item for item in trace["weights"] if item[0].source // 9 != item[0].target // 9
    ]
    assert len(cross) == 90
    assert all(item[1] <= Q(3, 4) for item in cross)
    assert all(
        proposal.compatibility_threshold > Q(3, 4)
        for proposal in stage.target_proposals
    )
    record_property("split_sense_new_edges", len(actual))
    record_property("split_sense_total_edges", graph.number_of_edges() + len(actual))
    record_property("split_sense_uniform_degree", 4)


def test_identical_primary_state_does_not_select_the_same_completed_proposal(proposals):
    common_graph, common, common_factors, _ = proposals[0]
    split_graph, split, split_factors, _ = proposals[1]
    assert common_graph.graph == split_graph.graph
    assert tuple(common_graph.edges(data=True)) == tuple(split_graph.edges(data=True))
    for node in common_graph:
        for key in ("EPI", "theta", "nu_f", "delta_nfr"):
            assert common_graph.nodes[node][key] == split_graph.nodes[node][key]
    assert common_factors == split_factors
    assert common.node_updates == split.node_updates
    assert common.edges != split.edges
    assert tuple(common_graph.nodes) == tuple(split_graph.nodes)


def test_all_target_reset_retains_circular_windings_and_fixed_channels(
    source, proposals, record_property
):
    pi = pi_interval()
    for index, (graph, stage, _, trace) in enumerate(proposals):
        phases = _phases(graph, stage)
        for update in stage.node_updates:
            assert update.vf_after == 1 and update.dnfr_after == 0
            node = update.node
            nominal = 2 * pi * source.geometry.target_phase_turns[node]
            nominal += (-DELTA / 2, DELTA / 2)[node // 9]
            assert (
                _wrap(I(Q(graph.nodes[node]["theta"])) - nominal).abs_max
                < REPRESENTATION_BUDGET
            )
            assert _wrap(I(Q(phases[node])) - nominal).abs_max < REPRESENTATION_BUDGET
        for ring, winding in ((0, 1), (1, 2)):
            cycle = tuple(range(9 * ring, 9 * ring + 9))
            differences = tuple(
                I(Q(phases[b]) - Q(phases[a]))
                for a, b in zip(cycle, cycle[1:] + cycle[:1], strict=True)
            )
            offsets, principal, observed, margin = _branches(differences)
            assert principal is not None and observed == -sum(offsets) == winding
            assert margin > 0
        for label, values in trace.items():
            errors = (
                values
                if label in {"angles", "writes"}
                else [item[-1] for item in values]
            )
            assert errors and max(errors) < REPRESENTATION_BUDGET
            record_property(
                f"completion_{index}_{label}_maximum_error", str(max(errors))
            )
        record_property(f"completion_{index}_proposed_form_writes", 0)
    record_property("represented_discrepancy_allowance", str(REPRESENTATION_BUDGET))


def test_independent_full_support_work_includes_old_phase_reset_and_weighted_form(
    proposals, record_property
):
    for split, (graph, stage, _, _) in zip((False, True), proposals, strict=True):
        work, kinetic = _work(graph, stage)
        reference = _nominal_work(split_sense=split)
        assert kinetic >= 0
        assert kinetic <= 2 * len(stage.edges) * ETA**2
        # Each old-before, old-after and new edge has at most2B phase error.
        propagated = 2 * (36 + len(stage.edges)) * REPRESENTATION_BUDGET + kinetic
        assert (work - reference).abs_max < propagated
        assert work.lo > (8 if split else 26)
        phases = _phases(graph, stage)
        bare_minus_weighted_potential = sum(
            (1 - Q(edge.weight))
            * (1 - cos(I(Q(phases[edge.right]) - Q(phases[edge.left]))))
            for edge in stage.edges
        )
        assert bare_minus_weighted_potential.lo > 0
        record_property(
            "split_work_lower" if split else "common_work_lower", str(work.lo)
        )
        record_property(
            "split_work_upper" if split else "common_work_upper", str(work.hi)
        )
    # Absolute supplied work is not excess above the new graph's target.
    # Neither these sums nor unchanged windings certify subsequent capture.


def test_activity_policy_selection_is_distinct_from_the_default_preflight(source):
    graph = _constructed(source.geometry)
    assert _preflight(graph).exact_stage
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    with pytest.raises(OperatorPreconditionError, match="EPI too low"):
        _preflight(graph)
    assert _preflight(graph, validate_preconditions=False).exact_stage


def test_zero_form_empty_history_is_a_separate_grammar_obligation(source):
    graph = _constructed(source.geometry)
    graph.nodes[0]["EPI"] = 0.0
    assert "glyph_history" not in graph.nodes[0]
    with pytest.raises(
        StructuralGrammarError, match="EPI=0 with empty history requires a generator"
    ):
        _preflight(graph)
    graph.graph.update(VALIDATE_OPERATOR_PRECONDITIONS=True, UM_MIN_EPI=0)
    validate_coupling(graph, 0)
    with pytest.raises(
        StructuralGrammarError, match="EPI=0 with empty history requires a generator"
    ):
        _preflight(graph)


def test_hard_u3_cannot_be_disabled_or_widened_by_optional_activity_flags(source):
    graph = _constructed(source.geometry)
    graph.nodes[0]["theta"] += math.pi
    graph.graph.update(UM_STRICT_PHASE_CHECK=False, UM_MAX_PHASE_DIFF=math.pi)
    assert graph.graph.get("VALIDATE_OPERATOR_PRECONDITIONS", False) is False
    with pytest.raises(OperatorPreconditionError, match="U3"):
        _preflight(graph, validate_preconditions=False)


def test_malformed_auxiliary_completion_cannot_replace_missing_source_evidence(source):
    for key in (ALIAS_SI[0], "delta_nfr"):
        graph = _constructed(source.geometry)
        graph.nodes[0][key] = None
        before = _state(graph)
        with pytest.raises((TypeError, ValueError)):
            _proposal(graph)
        assert _state(graph) == before
