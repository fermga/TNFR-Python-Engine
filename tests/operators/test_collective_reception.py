"""Synthetic EN controls on the marked weighted contact, without trajectories.

These independent preparations exercise the actual grammar and atomic stage.
They are not replays of a retained research response or sampled family proofs.
"""

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction as Q
from math import pi

import networkx as nx
import pytest

from tnfr.alias import set_dnfr
from tnfr.constants.canonical import EN_MIX_FACTOR
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors import TNFRValueError
from tnfr.operators import network_stage
from tnfr.operators.definitions import Coherence, Reception, Silence
from tnfr.operators.grammar_debt import node_debt
from tnfr.operators.grammar_dynamics import (
    validate_candidate,
    validate_sequence_incremental,
)
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_types import StructuralGrammarError
from tnfr.operators.grammar_u6 import observe_structural_potential_confinement
from tnfr.operators.network_stage import execute_neighbor_stage
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.fields import compute_structural_potential
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.physics.relational_sine_symmetry import assess_sine_support_involution
from tnfr.types import require_finite_real_scalar_epi

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
PORTS = (0, 1)
PERMUTATION = (0, 1) + tuple(node ^ 1 for node in range(2, 20))
NOW = datetime(2030, 1, 2, 3, 4, 7, tzinfo=timezone.utc)


class _SuppliedClock:
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is not None else NOW.replace(tzinfo=None)


def _contact(*, reverse_edges=False):
    graph = nx.Graph()
    graph.graph.update(
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        GRAMMAR_REJECTION_MODE="raise",
        GLYPH_FACTORS={"SHA_vf_factor": 0.5},
        SHA_MIN_VF=0.125,
        EPI_LATENT_MAX=1.0,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        _t=7,
    )
    for node in range(20):
        history = ("AL", "UM", "SHA", "IL", "OZ")
        if node == 1:
            history = ("AL", "AL") + history
        form = 19 / 64 if node < 10 else 21 / 64
        if node in PORTS:
            form = (20 + node) / 64
        offset = 1 / 64 if node == 1 else 1 / 16 if node in (10, 11) else 0
        graph.add_node(
            node,
            EPI=form,
            EPI_kind=f"member-{node}",
            theta=2 * pi * (node % 10 // 2) / 5 + offset,
            nu_f=0.5 if node in PORTS else 1.0,
            delta_nfr=0.0,
            glyph_history=deque(history, maxlen=8),
            _grammar_u2_debt=1,
            _grammar_prior_coherence=True,
        )
    edges = [
        (i, j, 1.0)
        for i in graph
        for j in range(i + 1, 20)
        if i // 10 == j // 10 and (i % 10 // 2 - j % 10 // 2) % 5 in (1, 4)
    ]
    edges += [(0, 1, 7 / 8)]
    edges += [(i, j, (15 if i == 0 else 13) / 16) for i in PORTS for j in (10, 11)]
    for left, right, weight in reversed(edges) if reverse_edges else edges:
        graph.add_edge(left, right, weight=weight)
    _refresh(graph)
    return graph


def _fields(graph):
    return bound_relational_sine_support_exchange(graph, reference_model=MODEL)


def _refresh(graph):
    for node, pressure in zip(graph, _fields(graph).pressure, strict=True):
        set_dnfr(graph, node, float(pressure.midpoint))


def _forms(graph):
    return tuple(
        require_finite_real_scalar_epi(graph.nodes[node]["EPI"]) for node in graph
    )


def _en_stage(graph, targets=PORTS):
    sequence = ValidatedSequence(
        (Reception(), Coherence(), Silence()),
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    # All forms here lie below the source telemetry's activity cut. No emission
    # source is a permitted observation, not a substitute precondition for EN.
    with pytest.warns(UserWarning, match="no emission sources detected"):
        return execute_network_operator_stage(
            graph, Reception(), targets, sequence_context=sequence.step(0)
        )


@pytest.fixture(scope="module")
def complete_word():
    graph = _contact()
    before = deepcopy(graph)
    operators = (Reception(), Coherence(), Silence())
    sequence = ValidatedSequence(
        operators,
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )
    assert all(
        item.allowed
        for node in PORTS
        for item in validate_sequence_incremental(graph, node, ("EN", "IL", "SHA"))
    )
    stages, states, observations = [], [], []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        for index, operator in enumerate(operators):
            assert all(
                validate_candidate(graph, node, operator.glyph.value).allowed
                for node in PORTS
            )
            baseline = compute_structural_potential(graph)
            if index == 0:
                with pytest.warns(UserWarning, match="no emission sources detected"):
                    result = execute_network_operator_stage(
                        graph,
                        operator,
                        PORTS,
                        sequence_context=sequence.step(index),
                        compute_delta_nfr=_refresh,
                    )
            else:
                result = execute_network_operator_stage(
                    graph,
                    operator,
                    PORTS,
                    sequence_context=sequence.step(index),
                    compute_delta_nfr=_refresh,
                )
            stages.append(result)
            states.append(deepcopy(graph))
            observations.append(
                observe_structural_potential_confinement(
                    graph,
                    baseline,
                    compute_structural_potential(graph),
                    reference=f"synthetic_before_{operator.glyph.value}",
                )
            )
    return before, tuple(stages), tuple(states), tuple(observations)


def test_complete_word_keeps_live_grammar_and_closes_inherited_debt(complete_word):
    before, stages, states, observations = complete_word
    assert tuple(stage.glyph for stage in stages) == ("EN", "IL", "SHA")
    assert all(
        stage.schedule == "two_phase_jacobi" and stage.nodes_processed == 2
        for stage in stages
    )
    assert [
        tuple(node_debt(state.nodes[node]) for node in PORTS) for state in states
    ] == [(1, 1), (0, 0), (0, 0)]
    assert all(item.confined for item in observations)
    for state in states:
        assert state.graph["_t"] == before.graph["_t"] == 7
        assert tuple(state.edges(data=True)) == tuple(before.edges(data=True))
        reduction = assess_sine_support_involution(
            _fields(state), permutation_indices=PERMUTATION
        )
        assert reduction.source_trajectory_reduction_certified
        assert reduction.multiplicities == (1, 1) + (2,) * 9
        for left, right in state.edges:
            gap = (state.nodes[right]["theta"] - state.nodes[left]["theta"] + pi) % (
                2 * pi
            ) - pi
            assert abs(gap) < pi / 2
    for node in PORTS:
        assert tuple(states[-1].nodes[node]["glyph_history"])[-3:] == (
            "EN",
            "IL",
            "SHA",
        )
        assert states[-1].nodes[node]["latency_start_time"] == NOW.isoformat()
        assert states[-1].nodes[node]["nu_f"] == 0.25


def test_en_retains_bare_neighbor_multiplicity_kinds_and_empty_source_observation(
    complete_word,
):
    before, stages, states, _ = complete_word
    for observation in stages[0].reception_observations:
        node = observation.node
        neighbors = tuple(before.neighbors(node))
        values = tuple(Q(before.nodes[neighbor]["EPI"]) for neighbor in neighbors)
        mean = sum(values) / 7
        expected = (1 - Q(EN_MIX_FACTOR)) * Q(before.nodes[node]["EPI"]) + Q(
            EN_MIX_FACTOR
        ) * mean
        assert observation.neighbors == neighbors
        assert observation.neighbor_epi_values == tuple(
            float(value) for value in values
        )
        assert observation.neighbor_epi_mean == float(mean)
        assert _forms(states[0])[node] == pytest.approx(float(expected), abs=1e-16)
        assert observation.source_tracking_enabled
        assert (
            observation.reception_sources == observation.reception_sources_after == ()
        )
        assert observation.reception_sources_present_after
        assert observation.read_boundary == "pre_en_snapshot"
        assert not observation.auxiliary_stability_certified
        dominant = next(
            peer for peer in neighbors if before.nodes[peer]["EPI"] == max(values)
        )
        assert observation.target_epi_kind_after == before.nodes[dominant]["EPI_kind"]
        weights = tuple(Q(before[node][peer]["weight"]) for peer in neighbors)
        weighted_mean = sum(
            value * weight for value, weight in zip(values, weights, strict=True)
        ) / sum(weights)
        assert abs(mean - weighted_mean) > Q(1, 10000)
        representative_mean = (
            sum(Q(before.nodes[peer]["EPI"]) for peer in (1 - node, 2, 8, 10)) / 4
        )
        assert abs(mean - representative_mean) > Q(1, 10000)


def test_form_only_en_changes_reciprocal_phase_rate_but_not_fresh_pressure(
    complete_word,
):
    before, _, states, _ = complete_word
    after = states[0]
    old, new = _fields(before), _fields(after)
    assert old.phase == new.phase
    assert old.capacity == new.capacity
    assert old.pressure == new.pressure
    assert old.form_rates == new.form_rates
    assert old.phase_storage == new.phase_storage
    assert old.phase_rates != new.phase_rates
    assert tuple(before.nodes[n]["delta_nfr"] for n in before) == tuple(
        after.nodes[n]["delta_nfr"] for n in after
    )
    displacement = tuple(b - a for a, b in zip(old.epi, new.epi, strict=True))
    expected_gradient_change = tuple(
        sum(
            Q(before[node][peer]["weight"]) * (displacement[node] - displacement[peer])
            for peer in before.neighbors(node)
        )
        for node in before
    )
    assert (
        tuple(b - a for a, b in zip(old.form_gradient, new.form_gradient, strict=True))
        == expected_gradient_change
    )
    old_rate, new_rate = old.phase_rate_numerators(), new.phase_rate_numerators()
    assert tuple(b - a for a, b in zip(old_rate, new_rate, strict=True)) == tuple(
        Q(before.nodes[node]["nu_f"])
        * expected_gradient_change[node]
        / before.degree(node)
        for node in before
    )
    linear_work = sum(
        q * h for q, h in zip(old.form_gradient, displacement, strict=True)
    )
    quadratic_work = sum(
        Q(before[i][j]["weight"]) * (displacement[i] - displacement[j]) ** 2 / 2
        for i, j in before.edges
    )
    assert new.form_storage - old.form_storage == linear_work + quadratic_work
    isolated_quadratic = sum(
        sum(Q(before[node][peer]["weight"]) for peer in before.neighbors(node))
        * displacement[node] ** 2
        / 2
        for node in PORTS
    )
    cross_term = -Q(before[0][1]["weight"]) * displacement[0] * displacement[1]
    assert cross_term != 0
    assert (
        new.form_storage - old.form_storage - linear_work - isolated_quadratic
        == cross_term
    )


def test_joint_en_is_not_two_sequential_port_events():
    joint, sequential = _contact(), _contact()
    before = _forms(joint)
    _en_stage(joint)
    _en_stage(sequential, (0,))
    _en_stage(sequential, (1,))
    first_jump = Q(_forms(joint)[0]) - Q(before[0])
    observed_difference = _forms(sequential)[1] - _forms(joint)[1]
    assert observed_difference == pytest.approx(
        float(Q(EN_MIX_FACTOR) * first_jump / 7), abs=1e-16
    )
    assert abs(observed_difference) > 1e-6


def test_neighbor_order_is_retained_even_when_physical_twins_are_equal():
    forward, reverse = _contact(), _contact(reverse_edges=True)
    first, second = _en_stage(forward), _en_stage(reverse)
    assert _forms(forward) == _forms(reverse)
    assert (
        first.reception_observations[0].neighbors
        != second.reception_observations[0].neighbors
    )
    assert forward.nodes[0]["EPI_kind"] != reverse.nodes[0]["EPI_kind"]


def test_en_neutrality_does_not_pay_inherited_debt():
    graph = _contact()
    sequence = ValidatedSequence(
        (Reception(), Silence()), context={"initial_epi_nonzero": True}
    )
    with pytest.warns(UserWarning, match="no emission sources detected"):
        execute_network_operator_stage(
            graph, Reception(), PORTS, sequence_context=sequence.step(0)
        )
    execute_network_operator_stage(
        graph, Silence(), PORTS, sequence_context=sequence.step(1)
    )
    assert tuple(node_debt(graph.nodes[node]) for node in PORTS) == (1, 1)


def test_declared_initialized_word_cannot_override_live_u1_rejection():
    operators = (Reception(), Coherence(), Silence())
    with pytest.raises(TNFRValueError, match="U1a"):
        ValidatedSequence(operators)
    sequence = ValidatedSequence(operators, context={"initial_epi_nonzero": True})
    graph = _contact()
    graph.nodes[1].update(
        EPI=0.0, glyph_history=[], _grammar_u2_debt=0, _grammar_prior_coherence=False
    )
    verdict = validate_candidate(graph, 1, "EN")
    assert not verdict.allowed
    assert {item.rule for item in verdict.violations} == {"U1a"}
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(graph.graph)
    with pytest.raises(StructuralGrammarError, match="requires a generator"):
        execute_network_operator_stage(
            graph, Reception(), PORTS, sequence_context=sequence.step(0)
        )
    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph == before_graph


@pytest.mark.parametrize("channel,value", [("EPI", 0.9), ("delta_nfr", 0.15)])
def test_live_reception_rejection_rolls_back_every_target(channel, value):
    graph = _contact()
    graph.nodes[1][channel] = value
    before_nodes, before_graph = deepcopy(dict(graph.nodes(data=True))), deepcopy(
        graph.graph
    )
    with pytest.raises(TNFRValueError, match="EN precondition failed"):
        execute_network_operator_stage(graph, Reception(), PORTS)
    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph == before_graph


def test_phase_source_telemetry_is_distinct_from_direct_numeric_inputs():
    graph = _contact()
    graph.nodes[4]["EPI"] = 0.75
    before = deepcopy(graph)
    result = execute_neighbor_stage(graph, Reception(), (0,), track_sources=True)
    observation = result.reception_observations[0]
    assert observation.reception_sources == ((4, pytest.approx(1 / 5), 0.75),)
    assert 4 not in observation.neighbors
    expected_mean = (
        sum(Q(before.nodes[peer]["EPI"]) for peer in before.neighbors(0)) / 7
    )
    assert observation.neighbor_epi_mean == float(expected_mean)
    assert observation.reception_sources_after == observation.reception_sources


def test_weighted_stationary_ports_can_receive_positive_work():
    graph = _contact()
    for node in graph:
        graph.nodes[node]["EPI"] = float(
            Q(451, 1472)
            if node in PORTS
            else Q(21, 64) if node in (10, 11) else Q(19, 64)
        )
    for i, j in graph.edges:
        if i in PORTS and (j in PORTS or j in (10, 11)):
            graph[i][j]["weight"] = 7 / 8
    before = _fields(graph)
    _en_stage(graph)
    after = _fields(graph)
    expected_jump = Q(EN_MIX_FACTOR) / 1288
    for node in PORTS:
        assert abs(before.form_gradient[node]) < Q(1, 10**15)
        assert float(after.epi[node] - before.epi[node]) == pytest.approx(
            float(expected_jump), abs=1e-16
        )
    expected_work = Q(23, 4) * expected_jump**2
    assert after.form_storage > before.form_storage
    assert float(after.form_storage - before.form_storage) == pytest.approx(
        float(expected_work), abs=1e-18
    )


def test_uniform_weight_single_port_has_passive_ideal_work_with_explicit_rounding():
    graph = _contact()
    for left, right in graph.edges:
        graph[left][right]["weight"] = 1.0
    before = _fields(graph)
    q = sum(before.epi[0] - before.epi[peer] for peer in graph.neighbors(0))
    assert q != 0
    degree, mix = graph.degree(0), Q(EN_MIX_FACTOR)
    ideal_jump = -mix * q / degree
    ideal_work = -mix * (1 - mix / 2) * q**2 / degree
    assert ideal_work < 0
    _en_stage(graph, (0,))
    after = _fields(graph)
    actual_jump = after.epi[0] - before.epi[0]
    represented_defect = actual_jump - ideal_jump
    expected_defect_work = q * represented_defect + degree * (
        ideal_jump * represented_defect + represented_defect**2 / 2
    )
    assert after.form_storage - before.form_storage - ideal_work == expected_defect_work
    assert after.form_storage < before.form_storage
