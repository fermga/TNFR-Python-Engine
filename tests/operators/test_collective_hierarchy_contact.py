"""Frozen synthetic THOL/UM composition with an explicit isolate completion.

Source and configuration were fixed before event evaluation. Three-sample
records are manufactured reader inputs, not observed sine trajectories.
Only canonical event stages and shared detached field readers are used.
"""

from fractions import Fraction as Q
from math import cos, pi, sin

import networkx as nx
import pytest

from tests.operators.test_collective_reception import MODEL, _SuppliedClock
from tnfr.alias import get_attr, set_dnfr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.constants.canonical import UM_COMPAT_THRESHOLD
from tnfr.errors import TNFRValueError
from tnfr.operators import network_stage
from tnfr.operators.definitions import Coupling, Dissonance, SelfOrganization, Silence
from tnfr.operators.grammar_debt import node_debt, reset_grammar_state_from_history
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_u6 import observe_structural_potential_confinement
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.fields import compute_structural_potential
from tnfr.physics.relational_sine_support import (
    bound_relational_sine_frozen_isolates,
    bound_relational_sine_support_exchange,
)
from tnfr.sdk._state import copy_graph_state

CHILD = "0_sub_0"


def _fields(graph):
    return bound_relational_sine_frozen_isolates(graph, reference_model=MODEL)


def _connected_fields(graph):
    return bound_relational_sine_support_exchange(graph, reference_model=MODEL)


def _refresh(graph):
    source = _fields(graph)
    for node, pressure in zip(source.nodes, source.pressure, strict=True):
        set_dnfr(graph, node, float(pressure.midpoint))


def _connected_refresh(graph):
    source = _connected_fields(graph)
    for node, pressure in zip(source.nodes, source.pressure, strict=True):
        set_dnfr(graph, node, float(pressure.midpoint))


def _source():
    graph = nx.Graph()
    graph.graph.update(
        _t=3.0,
        RANDOM_SEED=17,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        THOL_METABOLIC_ENABLED=True,
        EPI_MIN=0.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        EPI_LATENT_MAX=1.0,
        SHA_MIN_VF=1 / 8,
        DELTA_PHI_MAX=pi / 2,
        UM_MAX_PHASE_DIFF=pi / 2,
        UM_BIDIRECTIONAL=True,
        UM_SYNC_VF=True,
        UM_STABILIZE_DNFR=True,
        UM_FUNCTIONAL_LINKS=True,
        UM_CANDIDATE_COUNT=0,
        UM_CANDIDATE_MODE="proximity",
        GLYPH_FACTORS={
            "UM_theta_push": 1 / 8,
            "UM_vf_sync": 1 / 8,
            "UM_dnfr_reduction": 1 / 8,
            "SHA_vf_factor": 1 / 2,
        },
    )
    for node, form, phase in ((0, 0.5, 0.0), (1, 0.75, 0.125)):
        graph.add_node(
            node,
            EPI=form,
            nu_f=2.0,
            theta=phase,
            delta_nfr=0.0,
            Si=0.0,
            EPI_kind="supplied_parent" if node == 0 else "boundary",
            glyph_history=["AL", "IL"],
        )
        reset_grammar_state_from_history(graph.nodes[node])
    graph.add_edge(0, 1, weight=1.0)
    graph.nodes[0]["epi_time_history"] = [(0.0, 0.5), (1.0, 0.25), (3.0, 0.5)]
    _refresh(graph)
    return graph


def _word():
    operators = (Dissonance(), SelfOrganization(), Coupling(), Silence())
    return operators, ValidatedSequence(
        operators,
        context={"initial_epi_nonzero": True, "compatibility_profile": "core"},
    )


def _stage(graph, index, *, targets=(0,), refresh=_refresh):
    operators, sequence = _word()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        return execute_network_operator_stage(
            graph,
            operators[index],
            targets,
            sequence_context=sequence.step(index),
            compute_delta_nfr=refresh,
        )


@pytest.fixture(scope="module")
def complete_word():
    graph = _source()
    states = [copy_graph_state(graph)]
    stages = []
    operators, _ = _word()
    for index in range(4):
        assert validate_candidate(graph, 0, operators[index].glyph.value).allowed
        stages.append(_stage(graph, index))
        states.append(copy_graph_state(graph))
    return tuple(states), tuple(stages)


def _physical(graph, nodes):
    return tuple(
        tuple(
            get_attr(graph.nodes[node], aliases)
            for aliases in (ALIAS_EPI, ALIAS_THETA, ALIAS_VF)
        )
        for node in nodes
    )


def test_true_local_acceleration_domain_does_not_authenticate_manufactured_records():
    graph = _source()
    source = _connected_fields(graph)
    phase_rates = tuple(float(value.midpoint) for value in source.phase_rates)
    acceleration = 2 / pi * cos(1 / 8) * (phase_rates[1] - phase_rates[0])
    assert acceleration == pytest.approx(2 * cos(1 / 8) / pi**2)
    assert acceleration > 0.14
    observed = observe_structural_acceleration(graph, 0)
    assert observed.available and observed.value == 0.25
    assert observed.value != pytest.approx(acceleration)


def test_each_prefix_has_explicit_law_admission_and_complete_live_word(complete_word):
    states, stages = complete_word
    assert tuple(stage.glyph for stage in stages) == ("OZ", "THOL", "UM", "SHA")
    assert all(stage.schedule == "two_phase_jacobi" for stage in stages)
    assert [node_debt(graph.nodes[0]) for graph in states] == [0, 1, 0, 0, 0]
    assert all(graph.graph["_t"] == 3.0 for graph in states)
    assert all(_fields(graph).nodes == tuple(graph) for graph in states)
    for index, graph in enumerate(states):
        if index == 2:
            with pytest.raises(ValueError, match="connected|isolated"):
                _connected_fields(graph)
        else:
            connected, completed = _connected_fields(graph), _fields(graph)
            assert connected.epi == completed.epi
            assert connected.pressure == completed.pressure
            assert connected.form_rates == completed.form_rates
            assert connected.phase_rates == completed.phase_rates
            assert connected.storage == completed.storage
    final = states[-1]
    assert tuple(final.nodes[0]["glyph_history"])[-4:] == ("OZ", "THOL", "UM", "SHA")
    assert final.nodes[0]["latent"] is True
    assert final.nodes[CHILD]["glyph_history"] == []


def test_nominal_u6_observations_require_same_node_support_and_actual_baselines(
    complete_word,
):
    states, _ = complete_word
    observations = tuple(
        observe_structural_potential_confinement(
            after,
            compute_structural_potential(before),
            compute_structural_potential(after),
            reference=f"synthetic_hierarchy_contact_prefix_{index}",
        )
        for index, (before, after) in enumerate(zip(states, states[1:]))
        if index != 1
    )
    # Birth has no before observation for its new child. The shared monitor
    # explicitly rejects that domain; no synthetic zero baseline is inserted.
    with pytest.raises(ValueError, match="exactly the graph nodes"):
        observe_structural_potential_confinement(
            states[2],
            compute_structural_potential(states[1]),
            compute_structural_potential(states[2]),
            reference="unsupported_birth_baseline",
        )
    # These finite checks are not a whole-word/family U6 theorem or energy budget.
    assert all(observation.confined for observation in observations)


def test_birth_is_frozen_and_keeps_actual_metabolic_and_hierarchy_inputs(complete_word):
    states, _ = complete_word
    original, before, born = states[:3]
    assert _fields(original).storage == _fields(before).storage == _fields(born).storage
    assert tuple(born) == (0, 1, CHILD)
    assert tuple(born.edges(data=True)) == tuple(before.edges(data=True))
    assert _physical(before, (0, 1)) == _physical(born, (0, 1))
    child = born.nodes[CHILD]
    expected_form = 0.3 * 0.5 + (0.75 - 0.5) / (2 * pi)
    assert child["EPI"] == pytest.approx(expected_form, abs=1e-16)
    assert get_attr(child, ALIAS_VF) == 0.95 * 2.0
    assert get_attr(child, ALIAS_THETA) == 0.0
    assert child["parent_node"] == 0
    assert born.graph["hierarchy"] == {0: [CHILD]}
    source = _fields(born)
    index = source.nodes.index(CHILD)
    assert source.isolate_indices == (index,)
    assert source.inverse_phase_metric[index] is None
    for row in (source.pressure, source.form_rates, source.phase_rates):
        assert row[index].lo == row[index].hi == 0
    record = born.nodes[0]["sub_epis"][0]
    assert record["d2_epi"] == 0.25
    assert record["metabolized"]
    assert record["network_signals"]["neighbor_count"] == 1


def test_actual_um_weight_and_degree_normalization_come_from_new_support(complete_word):
    states, _ = complete_word
    born, attached, final = states[2:]
    assert set(attached.edges) == {(0, 1), (0, CHILD)}
    assert [attached.degree[node] for node in attached] == [2, 1, 1]
    assert get_attr(attached.nodes[0], ALIAS_THETA) == pytest.approx(1 / 128)
    assert get_attr(attached.nodes[1], ALIAS_THETA) == pytest.approx(15 / 128)
    assert get_attr(attached.nodes[CHILD], ALIAS_THETA) == 0.0
    child_form = born.nodes[CHILD]["EPI"]
    expected_weight = (
        1 - (1 / 128) / (2 * pi) - (0.5 - child_form) / (4 * (0.5 + child_form + 1e-9))
    )
    weight = attached.edges[0, CHILD]["weight"]
    assert weight == pytest.approx(expected_weight, abs=2e-16)
    assert weight > 4 / 5 > UM_COMPAT_THRESHOLD
    # Capacity blending consumes the old actual neighbor, not the new child.
    assert get_attr(attached.nodes[0], ALIAS_VF) == 2.0
    assert get_attr(final.nodes[0], ALIAS_VF) == 1.0
    assert get_attr(final.nodes[CHILD], ALIAS_VF) == 1.9
    source = _connected_fields(final)
    for index, node in enumerate(source.nodes):
        phase = get_attr(final.nodes[node], ALIAS_THETA)
        gradient = sum(
            final.edges[node, peer]["weight"]
            * (
                get_attr(final.nodes[node], ALIAS_EPI)
                - get_attr(final.nodes[peer], ALIAS_EPI)
            )
            for peer in final[node]
        )
        expected_pressure = sum(
            sin(get_attr(final.nodes[peer], ALIAS_THETA) - phase)
            for peer in final[node]
        ) / (pi * final.degree[node])
        capacity = get_attr(final.nodes[node], ALIAS_VF)
        assert float(source.pressure[index].midpoint) == pytest.approx(
            expected_pressure
        )
        assert float(source.form_rates[index].midpoint) == pytest.approx(
            capacity * expected_pressure
        )
        assert float(source.phase_rates[index].midpoint) == pytest.approx(
            capacity * gradient / (pi * final.degree[node])
        )


def test_complete_support_work_includes_old_phase_relaxation_and_birth_edge(
    complete_word,
):
    states, _ = complete_word
    born, attached, final = states[2:]
    child_form = get_attr(born.nodes[CHILD], ALIAS_EPI)
    weight = attached.edges[0, CHILD]["weight"]
    old_phase_work = cos(1 / 8) - cos(7 / 64)
    new_form_work = weight * (0.5 - child_form) ** 2 / 2
    new_phase_work = 1 - cos(1 / 128)
    work = float((_fields(attached).storage - _fields(born).storage).midpoint)
    assert old_phase_work < 0 < new_form_work
    assert work == pytest.approx(
        old_phase_work + new_form_work + new_phase_work, abs=2e-15
    )
    assert work > 1 / 50
    assert _fields(final).storage == _fields(attached).storage
    source = _connected_fields(states[0])
    initial_form_mean = sum(
        Q(d) / nu * x for d, nu, x in zip(source.degrees, source.capacity, source.epi)
    ) / sum(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    initial_phase_mean = sum(
        Q(d) / nu * theta
        for d, nu, theta in zip(source.degrees, source.capacity, source.phase)
    ) / sum(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    assert initial_form_mean == Q(5, 8)
    assert initial_phase_mean == Q(1, 16)
    parent_form, neighbor_form = source.epi
    parent_phase, neighbor_phase = source.phase
    child_form = Q(get_attr(born.nodes[CHILD], ALIAS_EPI))
    child_phase = Q(get_attr(born.nodes[CHILD], ALIAS_THETA))
    for endpoint in (attached, final):
        successor = _connected_fields(endpoint)
        weights = tuple(
            Q(d) / nu for d, nu in zip(successor.degrees, successor.capacity)
        )
        form_mean = sum(rho * x for rho, x in zip(weights, successor.epi)) / sum(
            weights
        )
        phase_mean = sum(
            rho * theta for rho, theta in zip(weights, successor.phase)
        ) / sum(weights)
        # Independently reconstruct the changed degree/capacity weights from
        # the actual three-node endpoint, including SHA's second parent reset.
        r0 = Q(2) / Q(get_attr(endpoint.nodes[0], ALIAS_VF))
        r1 = 1 / source.capacity[1]
        rc = 1 / Q(get_attr(endpoint.nodes[CHILD], ALIAS_VF))
        total = r0 + r1 + rc
        assert (
            form_mean
            == (r0 * parent_form + r1 * neighbor_form + rc * child_form) / total
        )
        assert (
            form_mean - initial_form_mean
            == (
                (r0 - 1 / source.capacity[0]) * (parent_form - initial_form_mean)
                + rc * (child_form - initial_form_mean)
            )
            / total
        )
        # All phases stay in this one declared lift. Use actual represented
        # increments rather than assuming ideal consensus/modulo arithmetic.
        assert all(0 <= theta < Q(1, 4) for theta in successor.phase)
        delta0 = Q(get_attr(endpoint.nodes[0], ALIAS_THETA)) - parent_phase
        delta1 = Q(get_attr(endpoint.nodes[1], ALIAS_THETA)) - neighbor_phase
        deltac = Q(get_attr(endpoint.nodes[CHILD], ALIAS_THETA)) - child_phase
        assert (
            phase_mean - initial_phase_mean
            == (
                (r0 - 1 / source.capacity[0]) * (parent_phase - initial_phase_mean)
                + rc * (child_phase - initial_phase_mean)
                + r0 * delta0
                + r1 * delta1
                + rc * deltac
            )
            / total
        )
        assert form_mean != initial_form_mean
        assert phase_mean != initial_phase_mean


@pytest.mark.parametrize("candidate_policy", ["disabled_links", "stale_inventory"])
def test_disconnected_control_keeps_same_old_resets_but_freezes_child(
    complete_word, candidate_policy
):
    states, _ = complete_word
    control = copy_graph_state(states[2])
    if candidate_policy == "disabled_links":
        control.graph["UM_FUNCTIONAL_LINKS"] = False
    else:
        control.graph["_node_sample"] = (0, 1)
    _stage(control, 2)
    assert _physical(control, (0, 1, CHILD)) == _physical(states[3], (0, 1, CHILD))
    _stage(control, 3)
    assert _physical(control, (0, 1, CHILD)) == _physical(states[4], (0, 1, CHILD))
    assert not nx.is_connected(control)
    null = _fields(control)
    active = _connected_fields(states[4])
    child_index = active.nodes.index(CHILD)
    assert null.form_rates[child_index].lo == null.form_rates[child_index].hi == 0
    assert null.phase_rates[child_index].lo == null.phase_rates[child_index].hi == 0
    assert active.form_rates[child_index].lo > 0
    assert active.phase_rates[child_index].hi < 0
    # Different phase velocities supply child form acceleration, even without
    # evaluating a continuous trajectory or inventing a sampled derivative.
    rates = tuple(float(value.midpoint) for value in active.phase_rates)
    acceleration = 1.9 / pi * cos(1 / 128) * (rates[0] - rates[child_index])
    assert acceleration > 3 / 200


def test_explicit_fresh_candidate_inventory_matches_live_inventory(complete_word):
    states, _ = complete_word
    graph = copy_graph_state(states[2])
    graph.graph["_node_sample"] = tuple(graph)
    _stage(graph, 2)
    assert tuple(graph.edges(data=True)) == tuple(states[3].edges(data=True))
    assert _fields(graph) == _fields(states[3])


def test_connected_refresh_rejects_birth_and_retains_only_committed_oz_prefix(
    complete_word,
):
    states, _ = complete_word
    graph = copy_graph_state(states[1])
    before = copy_graph_state(graph)
    with pytest.raises(ValueError, match="connected|isolated"):
        _stage(graph, 1, refresh=_connected_refresh)
    assert nx.utils.graphs_equal(graph, before)
    assert CHILD not in graph
    assert tuple(graph.nodes[0]["glyph_history"])[-1:] == ("OZ",)


def test_isolated_child_cannot_be_the_first_um_target(complete_word):
    states, _ = complete_word
    graph = copy_graph_state(states[2])
    before = copy_graph_state(graph)
    with pytest.raises(
        (OperatorPreconditionError, TNFRValueError),
        match="neighbor|neighbour|U3|degree",
    ):
        _stage(graph, 2, targets=(CHILD,))
    assert nx.utils.graphs_equal(graph, before)


def test_unknown_candidate_rejects_contact_but_preserves_committed_birth(complete_word):
    states, _ = complete_word
    graph = copy_graph_state(states[2])
    graph.graph["_node_sample"] = (0, 1, "absent-child")
    before = copy_graph_state(graph)
    with pytest.raises(TNFRValueError, match="unknown node"):
        _stage(graph, 2)
    assert nx.utils.graphs_equal(graph, before)
    assert CHILD in graph and graph.degree(CHILD) == 0
