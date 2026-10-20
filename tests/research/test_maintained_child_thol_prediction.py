"""Independent child-reset algebra; no reserved THOL proposal is evaluated."""

from dataclasses import replace
from fractions import Fraction as Q
from math import nextafter
from pathlib import Path

import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.constants.canonical import COUPLING_GENTLE, COUPLING_MODERATE
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import metabolism, network_stage
from tnfr.operators.definitions import SelfOrganization
from tnfr.research import maintained_child_thol_prediction as owner
from tnfr.research.maintained_parent_thol import _signature
from tnfr.sdk._state import copy_graph_state

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_child_history"


def _forbidden(*args, **kwargs):
    pytest.fail(
        "analytical child admission must not execute a proposal, event or trajectory"
    )


@pytest.fixture(scope="module")
def admitted():
    # Source admission may reconstruct inherited proof arithmetic; it may not
    # rerun any numerical acquisition, operator proposal or metabolic reader.
    from tnfr.research import (
        local_capacity_history,
        maintained_child_history_acquisition,
    )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SelfOrganization, "_prepare_execution", _forbidden)
        patch.setattr(metabolism, "capture_network_signals", _forbidden)
        patch.setattr(metabolism, "compose_subepi_amplitude", _forbidden)
        patch.setattr(network_stage, "execute_self_organization_stage", _forbidden)
        patch.setattr(local_capacity_history, "numerical_field", _forbidden)
        patch.setattr(local_capacity_history, "refresh_pressure", _forbidden)
        patch.setattr(
            maintained_child_history_acquisition, "rk4_coupled_step", _forbidden
        )
        source = owner.load_child_history_source(EVIDENCE, repository_root=ROOT)
        yield source


@pytest.fixture(scope="module")
def prediction(admitted):
    before = _signature(admitted.graph)
    result = owner._predict_child_thol(admitted)
    assert _signature(admitted.graph) == before
    return result


def test_public_reader_reloads_each_call_and_does_not_accept_a_cached_report(
    monkeypatch,
):
    calls, fresh = [], object()

    def load(path, *, repository_root):
        calls.append((path, repository_root))
        return fresh

    monkeypatch.setattr(owner, "load_child_history_source", load)
    monkeypatch.setattr(owner, "_predict_child_thol", lambda source: source)
    assert owner.predict_child_thol(EVIDENCE, repository_root=ROOT) is fresh
    assert owner.predict_child_thol(EVIDENCE, repository_root=ROOT) is fresh
    assert calls == [(EVIDENCE, ROOT), (EVIDENCE, ROOT)]


def test_own_history_and_live_context_supply_the_trigger(admitted, prediction):
    rows = admitted.graph.nodes[owner.ACTOR]["epi_time_history"]
    times, forms = zip(*rows, strict=True)
    h = Q(times[1]) - Q(times[0])
    exact = (Q(forms[2]) - 2 * Q(forms[1]) + Q(forms[0])) / h**2
    assert tuple(map(Q, times)) == prediction.sample_times
    assert h == Q(1, 1024) and Q(times[2]) - Q(times[1]) == h
    assert prediction.observed_acceleration == exact > 120
    assert prediction.threshold == Q(0.1)
    assert prediction.actor_form > prediction.minimum_form == Q(0.2)
    assert prediction.strict_preconditions_admitted and prediction.grammar_allowed
    assert prediction.expected_actor_glyph_history == ("OZ", "THOL")
    assert prediction.expected_actor_u2_debt == 0
    assert not prediction.complete_word_executed


def test_degree_one_exact_variance_removes_an_interval_dependency(prediction):
    expected = Q(0.3) * prediction.source_forms[20] + Q(COUPLING_MODERATE) * (
        prediction.source_forms[2] - prediction.source_forms[20]
    )
    assert prediction.neighbors == (2,)
    assert prediction.metabolic.neighbor_count == 1
    assert prediction.metabolic.mean_neighbor_epi == prediction.source_forms[2]
    assert prediction.ideal_phase_variance == 0
    assert (
        prediction.metabolic.phase_variance.lo
        <= 0
        <= prediction.metabolic.phase_variance.hi
    )
    assert prediction.ideal_child_form == expected
    assert (
        prediction.metabolic.child_form.lo
        <= expected
        <= prediction.metabolic.child_form.hi
    )
    assert (
        prediction.child_form_bounds.lo
        == prediction.metabolic.child_form.lo - owner.NUMERICAL_ALLOWANCE
    )
    assert (
        prediction.child_form_bounds.hi
        == prediction.metabolic.child_form.hi + owner.NUMERICAL_ALLOWANCE
    )
    assert Q(78, 1000) < prediction.child_form_bounds.lo
    assert prediction.child_form_bounds.hi < Q(79, 1000) < prediction.minimum_form


def test_capacity_is_rounded_not_falsely_declared_an_exact_product(prediction):
    exact = Q(121.6) * Q(0.95)
    represented = prediction.child_capacity
    assert prediction.child_capacity_ideal == exact
    assert represented == Q(float(121.6 * 0.95))
    assert represented != exact
    assert prediction.capacity_rounding_defect == represented - exact
    # Independent nearest-neighbor characterization of the chosen binary64.
    low = Q(nextafter(float(represented), float("-inf")))
    high = Q(nextafter(float(represented), float("inf")))
    assert (low + represented) / 2 < exact < (represented + high) / 2
    assert 0 < abs(prediction.capacity_rounding_defect) < Q(683, 10**17)


def test_raw_pressure_uses_two_rounded_operations_and_a_distinct_storage(prediction):
    gain, acceleration, pressure = (
        Q(COUPLING_GENTLE),
        prediction.observed_acceleration,
        prediction.actor_pressure,
    )
    expected_increment = Q(float(gain) * float(acceleration))
    expected_raw = Q(float(pressure) + float(expected_increment))
    assert prediction.pressure_gain == gain
    assert prediction.exact_pressure_increment == gain * acceleration
    assert prediction.represented_pressure_increment == expected_increment
    assert prediction.raw_pressure_represented == expected_raw
    assert prediction.raw_pressure_ideal == pressure + gain * acceleration
    assert (
        prediction.pressure_rounding_defect
        == expected_raw - prediction.raw_pressure_ideal
    )
    assert (
        prediction.raw_pressure_bounds.lo
        <= expected_raw
        <= prediction.raw_pressure_bounds.hi
    )
    assert prediction.raw_pressure_bounds.lo > 9
    assert (
        prediction.pressure_square_storage_jump
        == (prediction.raw_pressure_ideal**2 - pressure**2) / 2
        > 45
    )
    assert prediction.event_storage_jump == 0


def test_nested_identity_is_supplied_without_transport_or_inherited_history(prediction):
    assert prediction.source_nodes == (*range(20), owner.ACTOR)
    assert prediction.successor_nodes == (*prediction.source_nodes, owner.CHILD)
    assert prediction.successor_edges == prediction.source_edges
    assert len(prediction.successor_edges) == 53
    assert prediction.child_degree == 0
    assert prediction.child_hierarchy_level == 2
    assert prediction.child_hierarchy_path == (2, owner.ACTOR)
    assert prediction.expected_hierarchy == (
        (2, (owner.ACTOR,)),
        (owner.ACTOR, (owner.CHILD,)),
    )
    assert prediction.child_record_operator_step == 2
    assert prediction.child_phase == prediction.actor_phase
    assert prediction.child_pressure == 0
    assert prediction.child_glyph_history == ()
    assert prediction.child_legacy_history_length == 1
    assert prediction.child_timed_history_length == 0
    assert not prediction.child_immediate_thol_admitted


def test_old_component_equalities_do_not_transfer_a_future_tube(prediction):
    expected_weights = tuple(
        Q(d) / nu
        for d, nu in zip(
            prediction.source_degrees, prediction.source_capacities, strict=True
        )
    )
    assert prediction.old_component_mean_weights == expected_weights
    assert all(value > 0 for value in expected_weights)
    assert prediction.old_component_mean_jumps == (0, 0)
    assert prediction.old_component_rows_unchanged
    assert len(prediction.source_principal_margins) == 53
    assert min(prediction.source_principal_margins) > Q(3, 2)
    assert prediction.isolate_form_rate == prediction.isolate_phase_rate == 0
    assert prediction.successor_law == "supplied_frozen_isolate_completion"
    assert not prediction.global_positive_mean_metric_available
    assert not prediction.original_retention_transferred_to_acquired_point
    assert not prediction.future_from_acquired_point_certified
    assert not prediction.cross_birth_u5_u6_certified
    assert not prediction.actual_event


def test_principal_bounds_have_their_own_explicit_edge_orientation(prediction):
    positions = {node: index for index, node in enumerate(prediction.source_nodes)}
    graph_edges = tuple(
        (positions[a], positions[b]) for a, b in prediction.source_edges
    )
    assert graph_edges != prediction.source_principal_edges
    assert (
        tuple(sorted(tuple(sorted(edge)) for edge in graph_edges))
        == prediction.source_principal_edges
    )
    pi = pi_interval()
    for (left, right), shift, margin in zip(
        prediction.source_principal_edges,
        prediction.source_principal_shifts,
        prediction.source_principal_margins,
        strict=True,
    ):
        gap = (
            I(prediction.source_phases[right] - prediction.source_phases[left])
            - 2 * shift * pi
        )
        assert margin == pi.lo - gap.abs_max > 0


def test_zero_edge_work_comes_from_unchanged_arguments_not_interval_subtraction(
    prediction,
):
    positions = {node: index for index, node in enumerate(prediction.source_nodes)}
    edges = tuple((positions[a], positions[b]) for a, b in prediction.source_edges)

    def arguments(forms, phases):
        return tuple(
            (weight * (forms[j] - forms[i]) ** 2, phases[j] - phases[i])
            for (i, j), weight in zip(
                edges, prediction.source_conductances, strict=True
            )
        )

    original = arguments(prediction.source_forms, prediction.source_phases)
    # Even distinct isolate coordinates cannot enter the original edge-only H.
    # This is a structural identity, not a claim that these are event outputs.
    for child_form, child_phase in ((Q(-2), Q(8)), (Q(1, 16), Q(-3))):
        extended = arguments(
            (*prediction.source_forms, child_form),
            (*prediction.source_phases, child_phase),
        )
        assert extended == original
    assert all(21 not in edge for edge in edges)
    assert prediction.event_storage_jump == 0


def test_cached_derived_fields_and_observations_are_not_premises(admitted, prediction):
    changed = replace(
        admitted,
        fields=None,
        observation=None,
        assessment=None,
        sequence=None,
        threshold=Q(1000),
    )
    assert owner._predict_child_thol(changed) == prediction


@pytest.mark.parametrize(
    "key,value",
    [
        ("THOL_METABOLIC_ENABLED", False),
        ("THOL_MAX_BIFURCATION_DEPTH", 1),
        ("BIFURCATION_THRESHOLD_TAU", 0.0),
        ("GLYPH_FACTORS", {"VAL_scale": True}),
        ("VALIDATE_OPERATOR_PRECONDITIONS", 1),
        ("EDGE_AWARE_ENABLED", 0),
        ("VALIDATE_NODAL_EQUATION", True),
        ("GRAMMAR_REJECTION_MODE", "substitute"),
        ("EPI_MIN", True),
        ("_t", 7.0),
        ("hierarchy", {2: (owner.ACTOR,)}),
    ],
)
def test_live_configuration_cannot_be_replaced_by_cached_admission(
    admitted, key, value
):
    graph = copy_graph_state(admitted.graph)
    graph.graph[key] = value
    with pytest.raises((TypeError, ValueError)):
        owner._predict_child_thol(replace(admitted, graph=graph))


@pytest.mark.parametrize(
    "key,value",
    [
        ("hierarchy_level", True),
        ("_bifurcation_level", 5),
        ("parent_node", True),
        ("_hierarchy_path", [2, owner.ACTOR]),
        ("sub_nodes", ["absent"]),
        ("sub_epis", [{}]),
        ("glyph_history", ["VAL"]),
        ("_operator_step", True),
        ("_grammar_u2_debt", 0),
    ],
)
def test_live_nested_identity_and_context_are_admitted(admitted, key, value):
    graph = copy_graph_state(admitted.graph)
    graph.nodes[owner.ACTOR][key] = value
    with pytest.raises((TypeError, ValueError)):
        owner._predict_child_thol(replace(admitted, graph=graph))


@pytest.mark.parametrize(
    "alias,value",
    [
        (ALIAS_EPI, True),
        (ALIAS_VF, 0),
        (ALIAS_DNFR, float("nan")),
        (ALIAS_DNFR, 0.5),
    ],
)
def test_invalid_or_inconsistent_live_coordinate_rejects(admitted, alias, value):
    graph = copy_graph_state(admitted.graph)
    set_attr(graph.nodes[owner.ACTOR], alias, value)
    with pytest.raises((TypeError, ValueError)):
        owner._predict_child_thol(replace(admitted, graph=graph))


def test_stale_history_is_not_replaced_by_a_cached_acceleration(admitted):
    graph = copy_graph_state(admitted.graph)
    graph.nodes[owner.ACTOR]["epi_time_history"] = tuple(
        (float(Q(time) - 1), form)
        for time, form in graph.nodes[owner.ACTOR]["epi_time_history"]
    )
    with pytest.raises((TypeError, ValueError)):
        owner._predict_child_thol(replace(admitted, graph=graph))
