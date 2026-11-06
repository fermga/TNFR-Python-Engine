"""Static controls for the fixed local VAL/temporal-admission question.

The twenty-node fixture directly declares the nominal post-contact algebraic
state; it neither replays a frozen producer nor authenticates its ancestry.
Only VAL at node 2 executes. Synthetic histories below test independent live
gates and are explicitly not samples of this source's continuous trajectory.
No THOL proposal, child event, or continuous evolution is evaluated.
"""

from fractions import Fraction as Q
from math import cos, pi, sin

import networkx as nx
import pytest

from tests.operators.test_collective_reception import MODEL, _SuppliedClock
from tnfr.alias import get_attr, set_dnfr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics import relational
from tnfr.errors import TNFRValueError
from tnfr.operators import network_stage
from tnfr.operators._thol_config import resolve_thol_bifurcation_threshold
from tnfr.operators.definitions import Expansion, SelfOrganization, Silence
from tnfr.operators.grammar_debt import node_debt, reset_grammar_state_from_history
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.grammar_patterns import validate_sequence
from tnfr.operators.grammar_u6 import observe_structural_potential_confinement
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.operators.preconditions.self_organization import (
    validate_self_organization_strict,
)
from tnfr.operators.word_execution import execute_network_operator_stage
from tnfr.physics.fields import compute_structural_potential
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange
from tnfr.sdk._state import copy_graph_state

_PARENT = 2
_CONTEXT = {"initial_epi_nonzero": True, "compatibility_profile": "core"}


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_child():
    def forbidden(*args, **kwargs):
        pytest.fail("local admission controls permit no trajectory or THOL proposal")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        patch.setattr(SelfOrganization, "_prepare_execution", forbidden)
        yield


def _source(factor):
    """Declare an algebraic static control, with the correct target histories."""
    graph = nx.Graph(
        RANDOM_SEED=17,
        _t=7.0,
        GAMMA={"type": "none"},
        GRAMMAR_REJECTION_MODE="raise",
        VALIDATE_OPERATOR_PRECONDITIONS=True,
        EDGE_AWARE_ENABLED=False,
        EPI_MIN=-1.0,
        EPI_MAX=1.0,
        CLIP_MODE="hard",
        GLYPH_FACTORS={"VAL_scale": factor},
    )
    mean, boost, phase_push = 5 / 16, 1 / 1024, 1 / 1024
    alpha = 2 * pi / 5
    for node in range(20):
        pair = (node % 10) // 2
        phase = pair * alpha
        if node in (2, 3):
            phase = (1 - phase_push) * alpha
        elif node in (8, 9):
            phase = (4 + phase_push) * alpha
        graph.add_node(
            node,
            EPI=mean + (boost if node in (0, 1) else 0),
            theta=phase,
            nu_f=1 / 2 if node in (0, 1) else 1.0,
            Si=0.5,
            EPI_kind="declared_post_contact_algebraic_control",
            glyph_history=(
                ["AL", "UM", "SHA"]
                if node == 0
                else ["AL", "AL", "AL", "UM", "SHA"] if node == 1 else []
            ),
        )
        reset_grammar_state_from_history(graph.nodes[node])
    for offset in (0, 10):
        for pair in range(5):
            graph.add_edges_from(
                (offset + 2 * pair + left, offset + 2 * ((pair + 1) % 5) + right)
                for left in (0, 1)
                for right in (0, 1)
            )
    nx.set_edge_attributes(graph, 1.0, "weight")
    nx.set_edge_attributes(graph, 1.0, "length")
    graph.add_edge(0, 1, weight=1.0)
    weight = 1 - boost / (4 * (2 * mean + boost + 1e-9))
    graph.add_edges_from(
        (left, right, {"weight": weight}) for left in (0, 1) for right in (10, 11)
    )
    report = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    for node, pressure in zip(report.nodes, report.pressure):
        set_dnfr(graph, node, float(pressure.midpoint))
    return graph


def _stage(graph):
    # Batch validation describes a prospective word. Only its VAL prefix runs;
    # the owed stabilizer and closure remain unexecuted after this function.
    operators = (Expansion(), SelfOrganization(), Silence())
    assert validate_sequence([op.name for op in operators], context=_CONTEXT).passed
    sequence = ValidatedSequence(operators, context=_CONTEXT)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(network_stage, "datetime", _SuppliedClock)
        return execute_network_operator_stage(
            graph, operators[0], (_PARENT,), sequence_context=sequence.step(0)
        )


def _values(graph, aliases):
    return tuple(get_attr(graph.nodes[node], aliases) for node in graph)


def _mean(report, values):
    weights = tuple(
        Q(degree) / nu for degree, nu in zip(report.degrees, report.capacity)
    )
    return sum((weight * value for weight, value in zip(weights, values)), Q(0)) / sum(
        weights, Q(0)
    )


def _assert_read_only(graph, before):
    # The shared copier excludes rebuildable NodeNX caches from both sides.
    current = copy_graph_state(graph)
    assert current.graph == before.graph
    assert dict(current.nodes) == dict(before.nodes)
    assert dict(current.edges) == dict(before.edges)


@pytest.mark.parametrize("factor", (128.0, 256.0))
def test_local_val_preserves_physical_coordinates_and_reweights_complete_rows(factor):
    graph = _source(factor)
    initial = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    potential = compute_structural_potential(graph)
    before = copy_graph_state(graph)
    assert graph.number_of_edges() == 45
    assert graph.degree(_PARENT) == 4
    expected_pressure = cos(2 * pi / 5) * sin(2 * pi / (5 * 1024)) / pi
    assert get_attr(graph.nodes[_PARENT], ALIAS_DNFR) == pytest.approx(
        expected_pressure, abs=1e-16
    )
    assert get_attr(graph.nodes[_PARENT], ALIAS_DNFR) > 2**-14 > 1e-6

    stage = _stage(graph)
    changed = bound_relational_sine_support_exchange(graph, reference_model=MODEL)

    assert stage.glyph == "VAL" and stage.nodes_processed == 1
    assert stage.schedule == "two_phase_jacobi"
    for aliases in (ALIAS_EPI, ALIAS_THETA, ALIAS_DNFR):
        assert _values(graph, aliases) == _values(before, aliases)
    assert dict(graph.edges) == dict(before.edges)
    assert graph.graph["_t"] == 7.0
    assert get_attr(graph.nodes[_PARENT], ALIAS_VF) == factor
    for node in graph:
        if node != _PARENT:
            assert graph.nodes[node] == before.nodes[node]
            assert changed.form_rates[node] == initial.form_rates[node]
            assert changed.phase_rates[node] == initial.phase_rates[node]
    assert changed.form_rates[_PARENT] == initial.form_rates[_PARENT] * Q(factor)
    assert changed.phase_rate_numerators()[_PARENT] == -Q(factor) / 2048
    assert changed.phase_rate_numerators()[_PARENT] == (
        initial.phase_rate_numerators()[_PARENT] * Q(factor)
    )
    assert (
        changed.phase_rates[_PARENT] - initial.phase_rates[_PARENT] * Q(factor)
    ).contains(0)
    assert changed.form_storage == initial.form_storage
    assert changed.phase_storage == initial.phase_storage
    assert changed.storage == initial.storage
    assert changed.continuous_loss == 0
    assert changed.storage_rate.contains(0)
    # Local acceleration transport differs even at an unchanged-capacity node.
    assert (
        changed.form_acceleration_bounds()[0].hi
        < initial.form_acceleration_bounds()[0].lo
    )
    assert "VAL_MAX_VF" not in graph.graph
    assert "VAL_MIN_DNFR" not in graph.graph
    assert "VAL_MIN_EPI" not in graph.graph
    assert "edge_aware_interventions" not in graph.graph
    after_potential = compute_structural_potential(graph)
    assert after_potential == potential
    u6 = observe_structural_potential_confinement(graph, potential, after_potential)
    assert u6.confined and u6.mean_absolute_drift == 0.0
    assert u6.time_coverage == "two_snapshot_finite_observation"

    old_mass, new_mass = Q(104), Q(100) + 4 / Q(factor)
    delta_weight = 4 / Q(factor) - 4
    for values in (changed.epi, changed.phase):
        expected = (
            old_mass * _mean(initial, values) + delta_weight * values[_PARENT]
        ) / new_mass
        assert _mean(changed, values) == expected
        assert _mean(changed, values) != _mean(initial, values)
    # This exact statement uses the fixture's declared [0, 2*pi) phase lifts.
    assert _mean(changed, changed.epi) == Q(5, 16) + Q(28, 1024) / new_mass


def test_only_actual_local_val_supplies_the_target_grammar_context():
    graph = _source(128.0)
    assert graph.nodes[_PARENT]["glyph_history"] == []
    assert node_debt(graph.nodes[_PARENT]) == 0
    blocked = validate_candidate(graph, _PARENT, "THOL")
    assert not blocked.allowed
    assert {item.rule for item in blocked.violations} == {"U4b"}
    assert validate_candidate(graph, _PARENT, "VAL").allowed

    _stage(graph)

    assert tuple(graph.nodes[_PARENT]["glyph_history"]) == ("VAL",)
    assert graph.nodes[_PARENT]["_operator_step"] == 1
    assert node_debt(graph.nodes[_PARENT]) == 1
    assert validate_candidate(graph, _PARENT, "THOL").allowed
    assert len(graph) == 20 and "hierarchy" not in graph.graph
    assert all("sub_epis" not in graph.nodes[node] for node in graph)
    assert all("THOL" not in graph.nodes[node]["glyph_history"] for node in graph)


def test_admitted_capacity_and_static_acceleration_do_not_supply_observed_history():
    graph = _source(128.0)
    _stage(graph)
    report = bound_relational_sine_support_exchange(graph, reference_model=MODEL)
    assert report.form_acceleration_bounds()[_PARENT].lo > Q(1, 8)
    graph.nodes[_PARENT]["D2_EPI"] = 1.0
    before = copy_graph_state(graph)

    observation = observe_structural_acceleration(graph, _PARENT)
    assert not observation.available
    with pytest.raises(OperatorPreconditionError, match="samples"):
        validate_self_organization_strict(graph, _PARENT, emit_warnings=False)

    _assert_read_only(graph, before)


def test_strict_capacity_limit_blocks_a_second_val_without_consuming_more_context():
    graph = _source(128.0)
    _stage(graph)
    before = copy_graph_state(graph)
    with pytest.raises(OperatorPreconditionError, match="at maximum"):
        _stage(graph)
    _assert_read_only(graph, before)
    assert tuple(graph.nodes[_PARENT]["glyph_history"]) == ("VAL",)
    assert node_debt(graph.nodes[_PARENT]) == 1


def _synthetic_history(graph, *, acceleration=Q(1, 4)):
    """A quadratic gate fixture, never a same-law source record."""
    step = Q(1, 2**20)
    endpoint = Q(get_attr(graph.nodes[_PARENT], ALIAS_EPI))
    times = (Q(7), Q(7) + step, Q(7) + 2 * step)
    values = tuple(
        endpoint + acceleration * (time - times[-1]) ** 2 / 2 for time in times
    )
    graph.nodes[_PARENT]["epi_time_history"] = tuple(
        (float(time), float(value)) for time, value in zip(times, values)
    )
    graph.graph["_t"] = float(times[-1])


@pytest.mark.parametrize("acceleration", (Q(0), Q(1, 4)))
def test_synthetic_history_distinguishes_thol_live_gate_from_crossing(acceleration):
    graph = _source(128.0)
    _stage(graph)
    _synthetic_history(graph, acceleration=acceleration)
    before = copy_graph_state(graph)
    observation = observe_structural_acceleration(graph, _PARENT)
    assert observation.available and observation.current_endpoint_matches_state
    assert observation.value == float(acceleration)
    assert "THOL_BIFURCATION_THRESHOLD" not in graph.graph
    threshold = resolve_thol_bifurcation_threshold(graph.graph)
    assert (abs(observation.value) > threshold) is (acceleration > 0)
    # The optional strict gate warns on low acceleration; it does not require
    # child creation. This read-only pass supplies no actual event admission.
    validate_self_organization_strict(graph, _PARENT, emit_warnings=False)
    assert validate_candidate(graph, _PARENT, "THOL").allowed
    _assert_read_only(graph, before)


@pytest.mark.parametrize("pressure", (0.0, -1e-6, True))
def test_crossing_history_cannot_replace_thol_positive_finite_pressure(pressure):
    graph = _source(128.0)
    _stage(graph)
    _synthetic_history(graph)
    graph.nodes[_PARENT][ALIAS_DNFR[0]] = pressure
    if len(ALIAS_DNFR) > 1:
        graph.nodes[_PARENT][ALIAS_DNFR[1]] = 1 / 8
    before = copy_graph_state(graph)
    assert observe_structural_acceleration(graph, _PARENT).value > 0.1
    with pytest.raises(OperatorPreconditionError):
        validate_self_organization_strict(graph, _PARENT, emit_warnings=False)
    _assert_read_only(graph, before)


def test_synthetic_crossing_and_live_scalars_do_not_replace_actual_context():
    graph = _source(128.0)
    _synthetic_history(graph)
    validate_self_organization_strict(graph, _PARENT, emit_warnings=False)
    result = validate_candidate(graph, _PARENT, "THOL")
    assert not result.allowed
    assert {item.rule for item in result.violations} == {"U4b"}
    assert graph.nodes[_PARENT]["glyph_history"] == []


@pytest.mark.parametrize("gate", ("form", "capacity", "degree"))
def test_crossing_and_context_do_not_replace_other_strict_thol_gates(gate):
    graph = _source(128.0)
    _stage(graph)
    if gate == "form":
        graph.nodes[_PARENT][ALIAS_EPI[0]] = 3 / 16
    elif gate == "capacity":
        graph.nodes[_PARENT][ALIAS_VF[0]] = 1 / 16
    else:
        graph.remove_edges_from(tuple(graph.edges(_PARENT)))
    _synthetic_history(graph)
    before = copy_graph_state(graph)
    assert observe_structural_acceleration(graph, _PARENT).value > 0.1
    assert validate_candidate(graph, _PARENT, "THOL").allowed
    with pytest.raises(OperatorPreconditionError):
        validate_self_organization_strict(graph, _PARENT, emit_warnings=False)
    _assert_read_only(graph, before)


def test_history_reader_does_not_authenticate_endpoint_clock_or_producer():
    graph = _source(128.0)
    _stage(graph)
    _synthetic_history(graph)
    observation = observe_structural_acceleration(graph, _PARENT)
    graph.graph["_t"] = 100.0
    # API admission retains increasing timestamps and a matching live EPI.
    # Association with a same-law trajectory and its clock is an additional
    # evidence obligation; changing _t cannot authenticate these fake samples.
    assert observe_structural_acceleration(graph, _PARENT) == observation


@pytest.mark.parametrize("defect", ("stale_endpoint", "repeated_time", "boolean_time"))
def test_invalid_physical_history_cannot_fall_through_to_legacy_samples(defect):
    graph = _source(128.0)
    _stage(graph)
    _synthetic_history(graph)
    history = list(graph.nodes[_PARENT]["epi_time_history"])
    if defect == "stale_endpoint":
        history[-1] = (history[-1][0], history[-1][1] + 1 / 16)
    elif defect == "repeated_time":
        history[1] = (history[0][0], history[1][1])
    else:
        history[0] = (True, history[0][1])
    graph.nodes[_PARENT]["epi_time_history"] = history
    graph.nodes[_PARENT]["epi_history"] = [0.5, 0.25, 0.5]
    before = copy_graph_state(graph)
    with pytest.raises(TNFRValueError):
        observe_structural_acceleration(graph, _PARENT)
    with pytest.raises(OperatorPreconditionError):
        validate_self_organization_strict(graph, _PARENT, emit_warnings=False)
    _assert_read_only(graph, before)
