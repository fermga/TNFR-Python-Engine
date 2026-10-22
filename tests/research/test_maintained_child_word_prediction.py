"""Independent instantaneous child-word bounds without reserved events."""

from dataclasses import replace
from fractions import Fraction as Q
from math import pi
from pathlib import Path

import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.operators import al_sha_stage_proposals, metabolism, word_execution
from tnfr.operators.definitions import SelfOrganization
from tnfr.research import maintained_child_word_prediction as owner
from tnfr.research.maintained_parent_thol import _signature
from tnfr.sdk._state import copy_graph_state

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_child_history"


def _forbidden(*args, **kwargs):
    pytest.fail("word prediction cannot execute a reserved event, proposal or flow")


@pytest.fixture(scope="module", autouse=True)
def no_event_or_flow():
    from tnfr.research import (
        local_capacity_history,
        maintained_child_history_acquisition,
        maintained_parent_thol_execution,
    )

    with pytest.MonkeyPatch.context() as patch:
        for obj, name in (
            (SelfOrganization, "_prepare_execution"),
            (al_sha_stage_proposals, "propose_silence_stage"),
            (word_execution, "execute_network_operator_stage"),
            (metabolism, "capture_network_signals"),
            (metabolism, "compose_subepi_amplitude"),
            (local_capacity_history, "numerical_field"),
            (local_capacity_history, "refresh_pressure"),
            (maintained_parent_thol_execution, "refresh_frozen_successor"),
            (maintained_child_history_acquisition, "rk4_coupled_step"),
        ):
            patch.setattr(obj, name, _forbidden)
        yield


@pytest.fixture(scope="module")
def admitted():
    return owner.load_child_history_source(EVIDENCE, repository_root=ROOT)


@pytest.fixture(scope="module")
def prediction(admitted):
    before = _signature(admitted.graph)
    result = owner._predict_child_word(admitted)
    assert _signature(admitted.graph) == before
    return result


def test_public_entry_reloads_source_instead_of_accepting_a_mutable_prediction(
    monkeypatch,
):
    calls, source, result = [], object(), object()

    def load(path, *, repository_root):
        calls.append((path, repository_root))
        return source

    def predict(value):
        assert value is source
        return result

    monkeypatch.setattr(owner, "load_child_history_source", load)
    monkeypatch.setattr(owner, "_predict_child_word", predict)
    assert owner.predict_child_word(EVIDENCE, repository_root=ROOT) is result
    assert owner.predict_child_word(EVIDENCE, repository_root=ROOT) is result
    assert calls == [(EVIDENCE, ROOT), (EVIDENCE, ROOT)]


def test_sha_scalar_product_keeps_rounding_distinct_and_only_changes_actor(prediction):
    expected_factor = Q(1.0 - 1.0 / (4 * pi))
    ideal = Q(121.6) * expected_factor
    represented = Q(121.6 * float(expected_factor))
    assert prediction.sha_factor == expected_factor
    assert prediction.ideal_actor_capacity_after == ideal
    assert prediction.actor_capacity_after == represented
    assert prediction.capacity_rounding_defect == represented - ideal
    assert prediction.capacity_absolute_error == abs(represented - ideal)
    assert prediction.capacity_absolute_error > 0
    assert (
        Q(9, 10) < prediction.represented_capacity_ratio == represented / Q(121.6) < 1
    )
    for index, capacity in enumerate(prediction.capacity_before):
        if index == prediction.actor_index:
            assert prediction.ideal_capacity_after[index] == ideal
            assert prediction.represented_capacity_after[index] == represented
        else:
            assert prediction.ideal_capacity_after[index] == capacity
            assert prediction.represented_capacity_after[index] == capacity
    assert prediction.successor_capacities[:-1] == prediction.represented_capacity_after
    assert prediction.successor_capacities[-1] == prediction.thol.child_capacity
    assert (
        prediction.successor_capacities[-1]
        != prediction.thol.child_capacity * expected_factor
    )


def _node_rows(result, index, capacity):
    # Independent node-centered assembly, rather than the production edge loop.
    neighbors = {}
    for (left, right), weight in zip(result.edges, result.conductances, strict=True):
        if left == index:
            neighbors[right] = weight
        elif right == index:
            neighbors[left] = weight
    degree = len(neighbors)
    phases, forms = result.thol.source_phases, result.thol.source_forms
    sine_sum = sum((sin(I(phases[j] - phases[index])) for j in neighbors), I(0))
    gradient = sum(
        (weight * (forms[index] - forms[j]) for j, weight in neighbors.items()), Q(0)
    )
    pressure = sine_sum / (degree * pi_interval())
    return pressure, capacity * pressure, capacity * gradient / (degree * pi_interval())


def test_complete_old_rows_obey_the_same_pressure_with_local_capacity_scaling(
    prediction,
):
    assert prediction.constitutive_pressure_unchanged
    assert (
        prediction.before.pressure
        == prediction.ideal.pressure
        == prediction.after.pressure
    )
    ratio, actor = prediction.represented_capacity_ratio, prediction.actor_index
    assert (
        prediction.actor_form_rate_multiplier
        == prediction.actor_phase_rate_multiplier
        == ratio
    )
    for index in range(21):
        independent = _node_rows(
            prediction, index, prediction.represented_capacity_after[index]
        )
        for actual, expected in zip(
            (
                prediction.after.pressure[index],
                prediction.after.form_rate[index],
                prediction.after.phase_rate[index],
            ),
            independent,
            strict=True,
        ):
            assert (actual - expected).contains(0)
        if index != actor:
            assert (
                prediction.after.form_rate[index] == prediction.before.form_rate[index]
            )
            assert (
                prediction.after.phase_rate[index]
                == prediction.before.phase_rate[index]
            )
    assert (
        prediction.after.form_rate[actor] - ratio * prediction.before.form_rate[actor]
    ).contains(0)
    assert (
        prediction.after.phase_rate[actor] - ratio * prediction.before.phase_rate[actor]
    ).contains(0)
    assert prediction.after.form_rate[actor].lo > 0
    assert prediction.after.phase_rate[actor].hi < 0
    # At least one unchanged nonzero remote row rules out a global clock rescale.
    assert any(not prediction.before.form_rate[i].contains(0) for i in range(20))
    assert prediction.other_node_instantaneous_rows_unchanged
    assert prediction.expected_actor_latent


def test_reciprocal_capacity_means_change_by_the_exact_single_weight_identity(
    prediction,
):
    actor = prediction.actor_index
    for i, degree in enumerate(prediction.before.degrees):
        assert (
            prediction.mean_weights_before[i]
            == Q(degree) / prediction.capacity_before[i]
        )
        assert (
            prediction.mean_weights_after[i]
            == Q(degree) / prediction.represented_capacity_after[i]
        )
        if i != actor:
            assert prediction.mean_weights_before[i] == prediction.mean_weights_after[i]
    delta = prediction.mean_weights_after[actor] - prediction.mean_weights_before[actor]
    assert delta > 0
    total_after = sum(prediction.mean_weights_after, Q(0))
    assert (
        prediction.form_mean_jump
        == delta
        * (prediction.thol.actor_form - prediction.form_mean_before)
        / total_after
    )
    assert (
        prediction.phase_mean_jump
        == delta
        * (prediction.thol.actor_phase - prediction.phase_mean_before)
        / total_after
    )
    assert (
        prediction.form_mean_after - prediction.form_mean_before
        == prediction.form_mean_jump
        != 0
    )
    assert (
        prediction.phase_mean_after - prediction.phase_mean_before
        == prediction.phase_mean_jump
        != 0
    )
    assert prediction.after.form_mean.contains(prediction.form_mean_after)
    assert prediction.after.phase_mean.contains(prediction.phase_mean_after)
    # Use retained real lifts directly, with no phase wrapping or circular mean.
    assert (
        prediction.phase_mean_after
        == sum(
            (
                phase * weight
                for phase, weight in zip(
                    prediction.thol.source_phases,
                    prediction.mean_weights_after,
                    strict=True,
                )
            ),
            Q(0),
        )
        / total_after
    )


def test_full_successor_preserves_child_interval_and_declares_frozen_isolate(
    prediction,
):
    assert prediction.nodes == (*prediction.thol.source_nodes, owner.CHILD)
    assert prediction.isolate_index == 21
    assert prediction.forms[-1] == prediction.thol.child_form_bounds
    assert prediction.forms[-1].lo < prediction.forms[-1].hi
    assert prediction.successor.degrees == (*prediction.before.degrees, 0)
    assert prediction.successor.form_rate[:-1] == prediction.after.form_rate
    assert prediction.successor.phase_rate[:-1] == prediction.after.phase_rate
    assert prediction.successor.pressure[:-1] == prediction.after.pressure
    for rows in (
        prediction.successor.form_rate,
        prediction.successor.phase_rate,
        prediction.successor.pressure,
        prediction.successor.mean_weights,
    ):
        assert rows[-1] == I(0)
    assert (
        prediction.isolate_pressure
        == prediction.isolate_form_rate
        == prediction.isolate_phase_rate
        == 0
    )
    assert (
        prediction.before.positive_mean_metric and prediction.after.positive_mean_metric
    )
    assert not prediction.successor.positive_mean_metric
    assert not prediction.global_positive_mean_metric_available


def test_storage_arguments_and_old_pressure_are_preserved_at_both_handoffs(prediction):
    # H depends on the edges' forms and phase differences, never capacities.
    form_storage = sum(
        (
            weight * (prediction.forms[j] - prediction.forms[i]) ** 2 / 2
            for (i, j), weight in zip(
                prediction.edges, prediction.conductances, strict=True
            )
        ),
        I(0),
    )
    old_form_storage = sum(
        (
            weight
            * (prediction.thol.source_forms[j] - prediction.thol.source_forms[i]) ** 2
            / 2
            for (i, j), weight in zip(
                prediction.edges, prediction.conductances, strict=True
            )
        ),
        Q(0),
    )
    assert form_storage.contains(old_form_storage)
    assert all(prediction.isolate_index not in edge for edge in prediction.edges)
    assert all(
        prediction.phases[j] - prediction.phases[i]
        == prediction.thol.source_phases[j] - prediction.thol.source_phases[i]
        for i, j in prediction.edges
    )
    assert (
        prediction.thol_storage_jump
        == prediction.sha_storage_jump
        == prediction.total_storage_jump
        == 0
    )
    assert prediction.expected_stored_pressures == (
        *prediction.thol.source_stored_pressures,
        Q(0),
    )
    assert (
        prediction.thol.raw_pressure_bounds.lo > 9
    )  # Raw THOL relief is separately refreshed.


def test_sha_context_is_hypothetical_readonly_and_uses_no_newborn_graph(
    admitted, prediction, monkeypatch
):
    original = owner.validate_candidate
    signature = _signature(admitted.graph)
    seen = []

    def observe(graph, node, glyph, **kwargs):
        assert graph is not admitted.graph and glyph == "SHA" and node == owner.ACTOR
        assert tuple(graph) == tuple(admitted.graph) and owner.CHILD not in graph
        for old_node in graph:
            for alias in (ALIAS_EPI, ALIAS_THETA, ALIAS_VF, ALIAS_DNFR):
                assert get_attr(graph.nodes[old_node], alias) == get_attr(
                    admitted.graph.nodes[old_node], alias
                )
        actor = graph.nodes[node]
        assert tuple(actor["glyph_history"]) == ("OZ", "THOL")
        assert actor["_operator_step"] == 2 and actor["_grammar_u2_debt"] == 0
        assert (
            actor["epi_time_history"] == admitted.graph.nodes[node]["epi_time_history"]
        )
        seen.append(True)
        return original(graph, node, glyph, **kwargs)

    monkeypatch.setattr(owner, "validate_candidate", observe)
    assert owner._sha_context(admitted.graph, prediction.thol) == (("OZ", "THOL"), 2, 0)
    assert seen == [True] and _signature(admitted.graph) == signature


def test_expected_word_closure_does_not_claim_a_new_history_or_future_certificate(
    prediction,
):
    assert prediction.structural_time == Q(3683329, 524288)
    assert (
        prediction.sha_preconditions_admitted
        and prediction.hypothetical_sha_grammar_allowed
    )
    assert prediction.expected_actor_glyph_history == ("OZ", "THOL", "SHA")
    assert (
        prediction.expected_actor_operator_step == 3
        and prediction.expected_actor_u2_debt == 0
    )
    assert prediction.preserved_epi == prediction.thol.actor_form
    assert (
        prediction.expected_silence_duration == 0
        and not prediction.was_initial_on_silence
    )
    assert prediction.actor_timed_history_preserved
    assert prediction.expected_new_child_glyph_history == ()
    for flag in (
        prediction.new_child_timed_history_available,
        prediction.complete_word_executed,
        prediction.whole_word_atomic,
        prediction.future_from_acquired_point_certified,
        prediction.old_retention_transferred_after_capacity_reset,
        prediction.numerical_u5_observation_available,
        prediction.cross_birth_u6_observation_available,
        prediction.autonomous_occurrence_derived,
        prediction.latency_wall_timestamp_available,
    ):
        assert flag is False
    assert prediction.events_executed == 0


def test_cached_fields_cannot_supply_word_premises(admitted, prediction):
    changed = replace(
        admitted,
        fields=None,
        observation=None,
        assessment=None,
        sequence=None,
        threshold=Q(1000),
    )
    assert owner._predict_child_word(changed) == prediction


@pytest.mark.parametrize(
    "key,value",
    [
        ("SHA_MIN_VF", True),
        ("SHA_MIN_VF", float("nan")),
        ("SHA_MIN_VF", 1000),
        ("SHA_vf_factor", 0.5),
        ("GLYPH_FACTORS", {"VAL_scale": 128.0, "SHA_vf_factor": 0.5}),
        ("_t", True),
        ("GRAMMAR_REJECTION_MODE", "substitute"),
    ],
)
def test_altered_execution_configuration_cannot_use_cached_readiness(
    admitted, key, value
):
    graph = copy_graph_state(admitted.graph)
    graph.graph[key] = value
    with pytest.raises((TypeError, ValueError)):
        owner._predict_child_word(replace(admitted, graph=graph))


@pytest.mark.parametrize("change", ("capacity", "history", "debt", "time_history"))
def test_live_actor_must_still_match_the_actual_source_admission(admitted, change):
    graph = copy_graph_state(admitted.graph)
    actor = graph.nodes[owner.ACTOR]
    if change == "capacity":
        actor[ALIAS_VF[0]] = True  # Bypass the generic float-coercing setter.
    elif change == "history":
        actor["glyph_history"].append("SHA")
    elif change == "debt":
        actor["_grammar_u2_debt"] = True
    else:
        actor["epi_time_history"] = [
            (t - 1, value) for t, value in actor["epi_time_history"]
        ]
    with pytest.raises((TypeError, ValueError)):
        owner._predict_child_word(replace(admitted, graph=graph))
