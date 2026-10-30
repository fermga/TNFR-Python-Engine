"""Detached SHA algebra and retained committed-UM source controls."""

from dataclasses import replace
from fractions import Fraction as Q
from math import pi
from pathlib import Path

import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_VF
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.operators import al_sha_stage_proposals, network_stage, word_execution
from tnfr.research import local_capacity_history
from tnfr.research import maintained_parent_sha as owner
from tnfr.research import maintained_parent_um_execution
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_parent_um_corrected"


@pytest.fixture(scope="module", autouse=True)
def no_response():
    def forbidden(*args, **kwargs):
        pytest.fail("SHA prediction must not execute a proposal, event or flow")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (al_sha_stage_proposals, "propose_silence_stage"),
            (word_execution, "execute_network_operator_stage"),
            (network_stage, "execute_coupling_stage"),
            (maintained_parent_um_execution, "evaluate_um_event"),
            (local_capacity_history, "evaluate_local_history"),
            (local_capacity_history, "declared_preparation"),
            (local_capacity_history, "refresh_pressure"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def retained():
    from tnfr.research.maintained_parent_sha_source import load_um_successor

    admitted = load_um_successor(EVIDENCE, repository_root=ROOT)
    signature = _signature(admitted.graph)
    prediction = owner.predict_parent_sha(admitted)
    assert _signature(admitted.graph) == signature
    return admitted, prediction


def test_exact_ideal_product_and_represented_product_have_separate_evidence():
    capacity = 117.87376674577816
    factor = 1.0 - 1.0 / (4.0 * pi)
    ideal, represented, error = owner._capacity_reset(capacity, factor)
    assert ideal == Q(capacity) * Q(factor)
    assert represented == Q(float(ideal))
    assert error == abs(represented - ideal)
    assert 0 < error < Q(1, 2**40)
    assert Q(9, 10) < factor < 1


@pytest.mark.parametrize("capacity,factor", ((0, 0.5), (2, 0), (2, 0.5)))
def test_general_scalar_kernel_retains_intentional_zero_and_exact_products(
    capacity, factor
):
    ideal, represented, error = owner._capacity_reset(capacity, factor)
    assert ideal == represented == Q(capacity) * Q(factor)
    assert error == 0


@pytest.mark.parametrize("bad", (True, False, "2", float("nan"), float("inf"), -1))
def test_capacity_admission_rejects_invalid_or_negative_scalars(bad):
    with pytest.raises((TypeError, ValueError)):
        owner._capacity_reset(bad, 0.5)


@pytest.mark.parametrize("bad", (True, False, "0.5", float("nan"), -0.5, 1.0, 2.0))
def test_factor_admission_precedes_scalar_product(bad):
    with pytest.raises((TypeError, ValueError)):
        owner._capacity_reset(2.0, bad)


@pytest.mark.parametrize("capacity,factor", ((Q(1, 10), 0.5), (2, Q(1, 10))))
def test_selected_representation_cannot_silently_round_exact_rational_inputs(
    capacity, factor
):
    with pytest.raises(ValueError, match="already be exactly represented"):
        owner._capacity_reset(capacity, factor)


def test_positive_capacity_product_cannot_underflow_to_zero():
    with pytest.raises(ValueError, match="underflows"):
        owner._capacity_reset(float.fromhex("0x0.0000000000001p-1022"), 0.5)


def test_retained_source_default_closes_only_a_predicted_word(retained):
    admitted, prediction = retained
    assert prediction.factor == Q(1.0 - 1.0 / (4.0 * pi))
    assert prediction.factor != Q(1, 2)
    assert tuple(admitted.graph.nodes[2]["glyph_history"]) == ("VAL", "THOL", "UM")
    assert prediction.expected_parent_history == ("VAL", "THOL", "UM", "SHA")
    assert prediction.expected_parent_operator_step == 4
    assert not prediction.complete_word_executed
    assert not prediction.autonomous_occurrence_derived
    assert not prediction.numerical_u5_observation_available
    assert not prediction.cross_birth_u6_observation_available
    assert prediction.event_storage_jump == Q(0)


def test_only_parent_capacity_changes_and_the_old_motion_family_is_excluded(retained):
    _, result = retained
    parent = result.parent_index
    assert (
        result.ideal_parent_capacity == result.capacity_before[parent] * result.factor
    )
    assert result.represented_parent_capacity == Q(float(result.ideal_parent_capacity))
    assert result.represented_capacity_ratio == (
        result.represented_parent_capacity / result.capacity_before[parent]
    )
    assert result.source_um_parent_capacity.contains(result.capacity_before[parent])
    assert result.represented_parent_capacity < result.source_um_parent_capacity.lo
    assert not result.represented_capacity_inside_um_family
    for index in range(len(result.nodes)):
        if index != parent:
            assert result.capacity_before[index] == result.ideal_capacity_after[index]
            assert (
                result.capacity_before[index]
                == result.represented_capacity_after[index]
            )


def test_pressure_is_unchanged_but_both_parent_rows_stay_nonzero(retained):
    _, result = retained
    parent = result.parent_index
    assert (
        result.before.pressure == result.ideal.pressure == result.represented.pressure
    )
    for index in range(len(result.nodes)):
        if index != parent:
            assert result.before.form_rate[index] == result.represented.form_rate[index]
            assert (
                result.before.phase_rate[index] == result.represented.phase_rate[index]
            )
    # Independently construct the selected node's bare-sine and weighted-form
    # numerators; the latency label is not a zero-rate law.
    sine_sum, form_gradient = I(0), Q(0)
    for (left, right), weight in zip(result.edges, result.conductances, strict=True):
        if parent not in (left, right):
            continue
        neighbor = right if parent == left else left
        sine_sum += sin(I(result.phases[neighbor] - result.phases[parent]))
        form_gradient += weight * (result.epi[parent] - result.epi[neighbor])
    rate_factor = result.represented_parent_capacity / (
        pi_interval() * result.before.degrees[parent]
    )
    assert (result.represented.form_rate[parent] - rate_factor * sine_sum).contains(0)
    assert (
        result.represented.phase_rate[parent] - rate_factor * form_gradient
    ).contains(0)
    assert not result.represented.form_rate[parent].contains(0)
    assert not result.represented.phase_rate[parent].contains(0)


def test_exact_means_use_updated_reciprocal_capacity_and_retained_lifts(retained):
    _, result = retained
    parent = result.parent_index
    ratio = result.represented_capacity_ratio
    assert (
        result.mean_weights_after[parent] == result.mean_weights_before[parent] / ratio
    )
    for index in range(len(result.nodes)):
        assert result.mean_weights_after[index] == (
            Q(result.before.degrees[index]) / result.represented_capacity_after[index]
        )
        if index != parent:
            assert result.mean_weights_before[index] == result.mean_weights_after[index]
    delta_weight = (
        result.mean_weights_after[parent] - result.mean_weights_before[parent]
    )
    total_after = sum(result.mean_weights_after, Q(0))
    assert result.form_mean_after - result.form_mean_before == (
        delta_weight * (result.epi[parent] - result.form_mean_before) / total_after
    )
    assert result.phase_mean_after - result.phase_mean_before == (
        delta_weight * (result.phases[parent] - result.phase_mean_before) / total_after
    )
    assert result.represented.form_mean.contains(result.form_mean_after)
    assert result.represented.phase_mean.contains(result.phase_mean_after)
    assert result.phases[10] < 0 and result.phases[11] < 0


@pytest.mark.parametrize("change", ("capacity", "factor", "history", "clock"))
def test_mutated_live_admission_cannot_reuse_a_cached_source_certificate(
    retained, change
):
    admitted, _ = retained
    graph = restore_recorded_graph_state(record_graph_state(admitted.graph)).graph
    if change == "capacity":
        set_attr(graph.nodes[2], ALIAS_VF, True)
    elif change == "factor":
        graph.graph["GLYPH_FACTORS"]["SHA_vf_factor"] = 0.5
    elif change == "history":
        graph.nodes[2]["glyph_history"].append("SHA")
    else:
        graph.graph["_t"] = True
    with pytest.raises((TypeError, ValueError)):
        owner.predict_parent_sha(replace(admitted, graph=graph))
