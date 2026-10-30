"""ZHIR events on general sine states with manufactured reader evidence.

The supplied EPI histories exercise admission, not an observed trajectory of
the selected sine law. Only existing atomic events are executed; no retained
research producer or continuous campaign is evaluated.
"""

from copy import deepcopy
from fractions import Fraction as Q
from math import copysign, cos, pi, sin, tau

import networkx as nx
import pytest

from tests.operators.test_collective_reception import (
    PERMUTATION,
    PORTS,
    _contact,
    _fields,
    _refresh,
)
from tnfr.operators.definitions import Coherence, Dissonance, Mutation, Silence
from tnfr.operators.grammar_debt import node_debt, reset_grammar_state_from_history
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_u6 import observe_structural_potential_confinement
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.fields import compute_structural_potential
from tnfr.physics.relational_sine_symmetry import assess_sine_support_involution


def _word():
    operators = (
        Coherence(),
        Dissonance(),
        Coherence(),
        Mutation(),
        Coherence(),
        Silence(),
    )
    return operators, ValidatedSequence(
        operators,
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )


def _manufactured_evidence(graph, *, prefix=("IL", "OZ", "IL")):
    """Supply reader inputs without assigning any generating continuous law."""
    graph.graph.update(
        ZHIR_THRESHOLD_XI=0.1,
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        RANDOM_SEED=37,
    )
    for node in graph:
        data = graph.nodes[node]
        form = data["EPI"]
        data["epi_time_history"] = [(10.0, form - 0.25), (12.0, form)]
        data["glyph_history"] = list(prefix)
        reset_grammar_state_from_history(data)


def _general_contact():
    graph = _contact()
    for node in graph:
        # A short global phase lift, independent of the old winding source.
        graph.nodes[node]["theta"] = 0.25 + (node // 2) / 32
    _refresh(graph)
    _manufactured_evidence(graph)
    return graph


def _mutation(graph, targets=PORTS, *, refresh=True):
    operators, sequence = _word()
    raw_pressure = []

    def refresh_pressure(subject):
        raw_pressure.append(tuple(subject.nodes[node]["delta_nfr"] for node in subject))
        _refresh(subject)

    result = execute_network_operator_stage(
        graph,
        operators[3],
        targets,
        sequence_context=sequence.step(3),
        compute_delta_nfr=refresh_pressure if refresh else None,
    )
    return result, raw_pressure


def _phase_work(before, after):
    return sum(
        cos(before.nodes[j]["theta"] - before.nodes[i]["theta"])
        - cos(after.nodes[j]["theta"] - after.nodes[i]["theta"])
        for i, j in before.edges
    )


@pytest.fixture(scope="module")
def admitted_action():
    before = _general_contact()
    after = deepcopy(before)
    result, raw_pressure = _mutation(after)
    return before, after, result, raw_pressure[0], _fields(before), _fields(after)


def test_mutation_commits_associated_decisions_and_preserves_nonphase_channels(
    admitted_action,
):
    before, after, result, raw, initial, final = admitted_action
    assert result.glyph == "ZHIR"
    assert result.schedule == "two_phase_jacobi"
    assert tuple(after.edges(data=True)) == tuple(before.edges(data=True))
    assert initial.epi == final.epi
    assert initial.capacity == final.capacity
    assert initial.form_gradient == final.form_gradient
    assert initial.form_storage == final.form_storage
    assert raw == tuple(before.nodes[node]["delta_nfr"] for node in before)
    assert after.graph["_t"] == before.graph["_t"] == 7
    assert tuple(item.node for item in result.mutation_decision_observations) == PORTS
    for item in result.mutation_decision_observations:
        data = after.nodes[item.node]
        assert item._proof_fields_are_intact()
        assert item.trigger_certificate.observed_depi_dt == 0.125
        assert item.trigger_certificate.time_basis == "physical_time"
        assert item.trigger_certificate.current_endpoint_matches_state
        assert item.destabilizer_operator == "dissonance"
        assert item.recent_history[-3:] == ("coherence", "dissonance", "coherence")
        assert item.theta_shift == 0.25
        assert data["theta"] == before.nodes[item.node]["theta"] + 0.25
        assert data["EPI_kind"] == before.nodes[item.node]["EPI_kind"]
        assert node_debt(data) == 1
        assert not observe_structural_acceleration(before, item.node).available
        assert item.structural_acceleration == 0.0  # Legacy unavailable sentinel.
        assert not item.bifurcation_potential


def test_complete_weighted_storage_fresh_rows_and_lifted_means(admitted_action):
    before, after, _, _, initial, final = admitted_action
    expected_work = _phase_work(before, after)
    actual_work = float((final.storage - initial.storage).midpoint)
    assert actual_work == pytest.approx(expected_work, abs=2e-14)
    assert abs(expected_work) > 1e-3
    assert initial.phase_rate_numerators() == final.phase_rate_numerators()
    assert initial.pressure != final.pressure
    assert initial.form_rates != final.form_rates
    for index, node in enumerate(after):
        phase = after.nodes[node]["theta"]
        expected_pressure = sum(
            sin(after.nodes[neighbor]["theta"] - phase) for neighbor in after[node]
        ) / (pi * after.degree[node])
        assert float(final.pressure[index].midpoint) == pytest.approx(expected_pressure)
        assert float(final.form_rates[index].midpoint) == pytest.approx(
            after.nodes[node]["nu_f"] * expected_pressure
        )
    rho = tuple(
        Q(before.degree[node]) / Q(before.nodes[node]["nu_f"]) for node in before
    )
    mass = sum(rho)
    form_mean_jump = (
        sum(r * (y - x) for r, x, y in zip(rho, initial.epi, final.epi)) / mass
    )
    phase_mean_jump = (
        sum(r * (y - x) for r, x, y in zip(rho, initial.phase, final.phase)) / mass
    )
    assert form_mean_jump == 0
    assert phase_mean_jump == sum(rho[node] for node in PORTS) / (4 * mass)
    assert phase_mean_jump > 0


def test_adjacent_simultaneous_phase_work_is_not_sum_of_isolated_costs(admitted_action):
    before, after, *_ = admitted_action
    isolated_costs = []
    for target in PORTS:
        isolated = deepcopy(before)
        _mutation(isolated, (target,))
        isolated_costs.append(_phase_work(before, isolated))
    correction = _phase_work(before, after) - sum(isolated_costs)
    # The common shift leaves edge (0,1) unchanged. Each isolated action costs
    # 1-cos(1/4) on that edge, irrespective of its conductance 7/8.
    assert correction == pytest.approx(-2 * (1 - cos(0.25)), abs=2e-15)
    assert abs(correction) > 0.06


@pytest.mark.parametrize("neighbor_phase, expected_sign", [(0.1, 1), (0.25, -1)])
def test_default_mutation_has_both_storage_signs_above_the_nodal_trigger(
    neighbor_phase, expected_sign
):
    before = _general_contact()
    for node in before:
        before.nodes[node].update(theta=neighbor_phase, nu_f=4.0)
    before.nodes[0]["theta"] = 0.0
    _refresh(before)
    # Strict instantaneous margins imply a short true secant exists by local
    # smooth flow; the supplied execution records still remain manufactured.
    assert _fields(before).form_rates[0].lo > Q.from_float(0.1)
    after = deepcopy(before)
    result, _ = _mutation(after, (0,))
    assert result.mutation_decision_observations[0].theta_shift == 0.25
    work = _phase_work(before, after)
    expected = before.degree[0] * (cos(neighbor_phase) - cos(neighbor_phase - 0.25))
    assert work == pytest.approx(expected, abs=1e-15)
    assert work * expected_sign > 0


def test_signed_zero_and_fixed_branch_need_retained_execution_context():
    positive = _general_contact()
    for node in positive:
        positive.nodes[node]["theta"] = 0.25
    _refresh(positive)
    negative = deepcopy(positive)
    negative.nodes[0]["delta_nfr"] = -0.0
    assert positive.nodes[0]["delta_nfr"] == negative.nodes[0]["delta_nfr"]
    assert copysign(1, positive.nodes[0]["delta_nfr"]) == 1
    assert copysign(1, negative.nodes[0]["delta_nfr"]) == -1
    plus_result, _ = _mutation(positive, (0,), refresh=False)
    minus_result, _ = _mutation(negative, (0,), refresh=False)
    assert positive.nodes[0]["theta"] == 0.5
    assert negative.nodes[0]["theta"] == 0.0
    assert plus_result.mutation_decision_observations[0].theta_shift == 0.25
    assert minus_result.mutation_decision_observations[0].theta_shift == -0.25
    assert copysign(1, negative.nodes[0]["delta_nfr"]) == -1
    # Fixed branch ignores the pressure direction while retaining its gate.
    fixed = _general_contact()
    fixed.nodes[0]["delta_nfr"] = -0.0
    fixed.graph["GLYPH_FACTORS"]["ZHIR_theta_shift"] = 0.375
    result, _ = _mutation(fixed, (0,))
    decision = result.mutation_decision_observations[0]
    assert decision.fixed_mode
    assert decision.theta_shift == 0.375
    assert decision.regime_changed is None
    assert fixed.nodes[0]["theta"] == 0.625


def test_full_word_closes_live_debt_with_supplied_temporal_evidence():
    graph = _general_contact()
    _manufactured_evidence(graph, prefix=("AL",))
    source_forms = tuple(graph.nodes[node]["EPI"] for node in graph)
    operators, sequence = _word()
    debts = []
    observations = []
    for index, operator in enumerate(operators):
        baseline = compute_structural_potential(graph)
        result = execute_network_operator_stage(
            graph,
            operator,
            PORTS,
            sequence_context=sequence.step(index),
            compute_delta_nfr=_refresh,
        )
        assert result.glyph == operator.glyph.value
        debts.append(tuple(node_debt(graph.nodes[node]) for node in PORTS))
        observations.append(
            observe_structural_potential_confinement(
                graph,
                baseline,
                compute_structural_potential(graph),
                reference="independent_manufactured_word_control",
            )
        )
    assert debts == [(0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (0, 0)]
    assert all(item.confined for item in observations)
    assert tuple(graph.nodes[node]["EPI"] for node in graph) == source_forms
    assert all(graph.nodes[node]["latent"] for node in PORTS)
    for node in PORTS:
        assert tuple(graph.nodes[node]["glyph_history"])[-6:] == (
            "IL",
            "OZ",
            "IL",
            "ZHIR",
            "IL",
            "SHA",
        )
        assert graph.nodes[node]["epi_time_history"][-1][1] == graph.nodes[node]["EPI"]


def test_default_admitted_phase_jump_can_leave_an_acute_sector():
    graph = nx.star_graph(3)
    for node, phase in enumerate((0.0, 1.4, 1.4, -1.4)):
        graph.nodes[node].update(EPI=0.25, nu_f=1.0, theta=phase, delta_nfr=0.0)
    nx.set_edge_attributes(graph, 1.0, "weight")
    _refresh(graph)
    _manufactured_evidence(graph)
    assert all(
        abs(graph.nodes[j]["theta"] - graph.nodes[i]["theta"]) < pi / 2
        for i, j in graph.edges
    )
    assert graph.nodes[0]["delta_nfr"] > 0.1
    assert validate_candidate(graph, 0, "ZHIR").allowed
    _mutation(graph, (0,))
    gap = (graph.nodes[3]["theta"] - graph.nodes[0]["theta"] + pi) % tau - pi
    assert gap == pytest.approx(-1.65)
    assert abs(gap) > pi / 2


def test_pair_closed_targets_preserve_reduction_but_single_member_does_not():
    before = _general_contact()
    assert assess_sine_support_involution(
        _fields(before), permutation_indices=PERMUTATION
    ).source_trajectory_reduction_certified
    paired, singleton = deepcopy(before), deepcopy(before)
    paired.graph["GLYPH_FACTORS"]["ZHIR_theta_shift"] = 0.25
    singleton.graph["GLYPH_FACTORS"]["ZHIR_theta_shift"] = 0.25
    _mutation(paired, (2, 3))
    _mutation(singleton, (2,))
    paired_report = assess_sine_support_involution(
        _fields(paired), permutation_indices=PERMUTATION
    )
    single_report = assess_sine_support_involution(
        _fields(singleton), permutation_indices=PERMUTATION
    )
    assert paired_report.source_trajectory_reduction_certified
    assert paired_report.multiplicities == (1, 1) + (2,) * 9
    assert not single_report.source_trajectory_reduction_certified
    assert paired.nodes[2]["theta"] == paired.nodes[3]["theta"]
    assert singleton.nodes[2]["theta"] != singleton.nodes[3]["theta"]
    assert all(
        abs((paired.nodes[j]["theta"] - paired.nodes[i]["theta"] + pi) % tau - pi)
        < pi / 2
        for i, j in paired.edges
    )


@pytest.mark.parametrize("invalid", ["history", "grammar"])
def test_one_invalid_target_rejects_entire_phase_stage_atomically(invalid):
    graph = _general_contact()
    if invalid == "history":
        del graph.nodes[1]["epi_time_history"]
        reason = "at least two"
    else:
        graph.nodes[1]["glyph_history"] = ["IL", "SHA", "NAV"]
        reset_grammar_state_from_history(graph.nodes[1])
        reason = "U4b violation"
        assert not validate_candidate(graph, 1, "ZHIR").allowed
    before = deepcopy(
        (dict(graph.nodes(data=True)), tuple(graph.edges(data=True)), graph.graph)
    )
    with pytest.raises(OperatorPreconditionError, match=reason):
        _mutation(graph)
    assert (
        dict(graph.nodes(data=True)),
        tuple(graph.edges(data=True)),
        graph.graph,
    ) == before
