"""Bounded typed-observation contracts; no historical producer imports.

Fixtures execute two-node AL/SHA words with declared held pressure and supplied
durations. They do not activate Mutation or claim a continuous operator limit.
"""

from copy import copy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.constants import inject_defaults
from tnfr.operators import (
    OperatorEPIContribution,
    OperatorEPIGrowthObservation,
    observe_operator_epi_growth,
)
from tnfr.operators.event_runtime import execute_operator_event_schedule
from tnfr.operators.event_timing import build_operator_event_schedule


def _execute(
    *,
    durations=(0.5, 0.0, 0.5),
    start=0.0,
    capture=True,
    pressure=0.0,
    edge_weight=1.0,
    word=("emission", "silence"),
):
    graph = nx.path_graph(2)
    graph.edges[0, 1]["weight"] = edge_weight
    inject_defaults(graph)
    graph.graph.update(
        _t=start,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        EPI_LATENT_MAX=0.875,
        CLIP_MODE="hard",
        RANDOM_SEED=23,
        GRAMMAR_CANON={"enabled": True},
        GRAMMAR_REJECTION_MODE="raise",
        IL_PRECONDITIONS={"warn_zero_dnfr": False},
    )
    graph.graph["GLYPH_FACTORS"].update(AL_boost=0.125, SHA_vf_factor=0.5)
    for node, epi in enumerate((0.25, 0.5)):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="same-form",
            nu_f=1.0,
            theta=0.0,
            delta_nfr=pressure,
            latent=False,
            glyph_history=[],
        )
    schedule = build_operator_event_schedule(
        word, start_time=start, flow_durations=durations
    )
    result = execute_operator_event_schedule(
        graph, schedule, include_stage_certificates=capture
    )
    return graph, result


@pytest.fixture(scope="module")
def positive_source():
    return _execute()


def test_public_reader_separates_actual_jump_from_zero_flow(positive_source):
    graph, result = positive_source
    observation = observe_operator_epi_growth(result)
    assert type(observation) is OperatorEPIGrowthObservation
    assert observation.source == "operator_schedule_net_growth"
    assert observation.available and observation.unavailability_reasons == ()
    assert observation.nodes == (0, 1)
    assert observation.exact_epi_before == (Fraction(1, 4), Fraction(1, 2))
    assert observation.exact_epi_after == (Fraction(3, 8), Fraction(5, 8))
    assert observation.exact_duration == 1
    assert observation.exact_flow_increment == (0, 0)
    assert observation.exact_jump_increment == (Fraction(1, 8),) * 2
    assert observation.exact_net_increment == observation.exact_jump_increment
    assert observation.exact_net_growth_rate == (Fraction(1, 8),) * 2
    assert observation.exact_jump_variation == observation.exact_jump_increment
    assert all(type(value) is Fraction for value in observation.exact_net_growth_rate)
    assert graph.graph["_t"] == 1.0


def test_capture_does_not_require_affine_gain_or_diffusive_pressure(positive_source):
    _, result = positive_source
    composition = result.represented_epi_schedule_composition
    assert not composition.represented_affine_composition_gain_certified
    assert composition.failed_conditions
    assert observe_operator_epi_growth(result).available


def test_included_operator_without_epi_map_certificate_still_has_observed_endpoints():
    _, result = _execute(
        word=("emission", "coherence", "silence"),
        durations=(0.5, 0.0, 0.0, 0.5),
    )
    stage = result.glyph_stage_evidence[1]
    assert stage.event.glyph.value == "IL"
    assert stage.certificate is None and stage.endpoint_capture_complete
    report = observe_operator_epi_growth(result)
    assert report.available
    assert report.exact_net_increment == (Fraction(1, 8),) * 2
    contribution = next(
        item for item in report.contributions if item.operator_name == "coherence"
    )
    assert contribution.exact_increment == (0, 0)


def test_contributions_keep_actual_kind_indices_and_coincident_order(positive_source):
    _, result = positive_source
    report = observe_operator_epi_growth(result)
    assert (
        tuple(type(item) for item in report.contributions)
        == (OperatorEPIContribution,) * 4
    )
    assert tuple(
        (item.operation_kind, item.operation_index) for item in report.contributions
    ) == (("flow", 0), ("glyph", 0), ("glyph", 1), ("flow", 2))
    assert report.included_event_indices == (0, 1)
    assert report.excluded_initial_event_indices == ()
    assert report.contributions[1].exact_start_time == Fraction(1, 2)
    assert report.contributions[1].exact_end_time == Fraction(1, 2)
    assert report.contributions[2].exact_start_time == Fraction(1, 2)
    assert report.contributions[1].operator_name == "emission"
    assert report.contributions[2].operator_name == "silence"
    assert all(
        left.exact_epi_after == right.exact_epi_before
        for left, right in zip(report.contributions, report.contributions[1:])
    )


@pytest.mark.parametrize(
    ("durations", "excluded", "included"),
    [((0.0, 0.5, 0.5), (0,), (1,)), ((0.0, 0.0, 1.0), (0, 1), ())],
)
def test_initial_right_boundary_excludes_all_initial_glyphs(
    durations, excluded, included
):
    _, result = _execute(durations=durations)
    report = observe_operator_epi_growth(result)
    assert report.available
    assert report.excluded_initial_event_indices == excluded
    assert report.included_event_indices == included
    assert (
        report.exact_epi_before
        == report.exact_epi_after
        == (Fraction(3, 8), Fraction(5, 8))
    )
    assert report.exact_jump_increment == report.exact_jump_variation == (0, 0)
    assert report.exact_net_growth_rate == (0, 0)


def test_terminal_glyphs_are_included():
    _, result = _execute(durations=(1.0, 0.0, 0.0))
    report = observe_operator_epi_growth(result)
    assert report.available
    assert report.included_event_indices == (0, 1)
    assert report.exact_jump_increment == (Fraction(1, 8),) * 2
    assert tuple(item.operation_kind for item in report.contributions) == (
        "flow",
        "glyph",
        "glyph",
    )
    assert report.contributions[-1].exact_end_time == report.exact_end_time == 1


def test_zero_duration_has_no_rate_or_fabricated_zero_totals():
    _, result = _execute(durations=(0.0, 0.0, 0.0))
    report = observe_operator_epi_growth(result)
    assert not report.available
    assert "nonpositive_window_duration" in report.unavailability_reasons
    assert report.excluded_initial_event_indices == (0, 1)
    assert report.included_event_indices == ()
    assert report.exact_net_growth_rate is report.exact_net_increment is None
    assert report.exact_flow_increment is report.exact_jump_increment is None


def test_intact_execution_without_endpoint_capture_is_typed_unavailable():
    _, result = _execute(capture=False)
    report = observe_operator_epi_growth(result)
    assert not report.available
    assert report.unavailability_reasons == ("operation_endpoint_trace_unavailable",)
    assert report.contributions == ()
    assert report.exact_net_growth_rate is None


def test_intact_execution_with_unsupported_capture_domain_is_unavailable():
    # Negative conductance is not a canonical organization. The declared
    # zero-driver AL/SHA control does not consume it, whereas optional nodal
    # endpoint capture does. Missing capture must never become zero evidence.
    _, result = _execute(edge_weight=-1.0)
    assert result._proof_fields_are_intact()
    assert result.stage_certification_requested
    report = observe_operator_epi_growth(result)
    assert not report.available
    assert any(
        "endpoints_unavailable" in reason for reason in report.unavailability_reasons
    )
    assert report.exact_net_growth_rate is report.exact_net_increment is None


def test_declared_clock_translation_preserves_rates(positive_source):
    _, original = positive_source
    _, translated = _execute(start=7.0)
    first = observe_operator_epi_growth(original)
    second = observe_operator_epi_growth(translated)
    assert second.available
    assert second.exact_start_time == 7 and second.exact_end_time == 8
    assert second.exact_duration == first.exact_duration == 1
    assert second.exact_net_growth_rate == first.exact_net_growth_rate


def test_positive_clock_rescaling_changes_only_window_average(positive_source):
    _, original = positive_source
    _, slower = _execute(durations=(1.0, 0.0, 1.0))
    first = observe_operator_epi_growth(original)
    second = observe_operator_epi_growth(slower)
    assert second.available
    assert second.exact_net_increment == first.exact_net_increment
    assert second.exact_duration == 2
    assert second.exact_net_growth_rate == tuple(
        value / 2 for value in first.exact_net_growth_rate
    )


def test_flow_is_accounted_separately_without_reconstructing_pressure():
    _, result = _execute(pressure=0.125)
    report = observe_operator_epi_growth(result)
    assert report.available
    # First half at nu=1, second half after SHA at nu=1/2.
    assert report.exact_flow_increment == (Fraction(3, 32),) * 2
    assert report.exact_jump_increment == (Fraction(1, 8),) * 2
    assert (
        report.exact_net_increment
        == report.exact_net_growth_rate
        == (Fraction(7, 32),) * 2
    )


@pytest.mark.parametrize("value", [None, {}, (), object()])
def test_wrong_source_type_is_not_an_unavailable_observation(value):
    with pytest.raises(TypeError, match="OperatorEventExecutionResult"):
        observe_operator_epi_growth(value)


def test_unsealed_result_cannot_be_promoted(positive_source):
    _, result = positive_source
    forged = copy(result)
    object.__setattr__(forged, "_proof_stamp", ())
    with pytest.raises(ValueError, match="intact committed"):
        observe_operator_epi_growth(forged)


@pytest.mark.parametrize("field", ["endpoints", "conditions", "order"])
def test_altered_nested_primitive_or_cached_summary_rejects(positive_source, field):
    _, result = positive_source
    forged = copy(result)
    composition = copy(result.represented_epi_schedule_composition)
    object.__setattr__(forged, "represented_epi_schedule_composition", composition)
    if field == "conditions":
        object.__setattr__(composition, "conditions", ())
    elif field == "order":
        object.__setattr__(
            composition, "operations", tuple(reversed(composition.operations))
        )
    else:
        operation = copy(composition.operations[0])
        object.__setattr__(operation, "exact_epi_after", (Fraction(99), Fraction(99)))
        object.__setattr__(
            composition, "operations", (operation, *composition.operations[1:])
        )
    with pytest.raises(ValueError, match="intact committed"):
        observe_operator_epi_growth(forged)
    assert observe_operator_epi_growth(result).available


def test_repeated_reading_does_not_touch_graph_history_or_source(positive_source):
    graph, result = positive_source
    retained = tuple(
        (
            node,
            graph.nodes[node]["epi_time_history"],
            tuple(graph.nodes[node]["epi_time_history"]),
            tuple(graph.nodes[node]["glyph_history"]),
            graph.nodes[node].get("_operator_step"),
        )
        for node in graph
    )
    log = tuple(dict(item) for item in graph.graph["hybrid_event_log"])
    first, second = observe_operator_epi_growth(result), observe_operator_epi_growth(
        result
    )
    assert first == second
    assert result._proof_fields_are_intact()
    assert tuple(graph.graph["hybrid_event_log"]) == log
    for node, history, samples, glyphs, step in retained:
        assert graph.nodes[node]["epi_time_history"] is history
        assert tuple(history) == samples
        assert tuple(graph.nodes[node]["glyph_history"]) == glyphs
        assert graph.nodes[node].get("_operator_step") == step
        assert "_native_zhir_consumed_intervals" not in graph.nodes[node]


def test_detached_observation_cannot_replace_execution_source(positive_source):
    _, result = positive_source
    observation = observe_operator_epi_growth(result)
    with pytest.raises(TypeError):
        observe_operator_epi_growth(observation)
