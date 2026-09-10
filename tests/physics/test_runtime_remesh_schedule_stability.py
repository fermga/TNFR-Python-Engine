"""Adjacent-cycle runtime REMESH/schedule history balances."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import networkx as nx
import pytest

import tnfr.physics.runtime_remesh_schedule_stability as bridge_module
from tnfr.errors import TNFRValueError
from tnfr.operators.event_remesh_runtime import execute_event_remesh_cycle
from tnfr.operators.event_remesh_sequence import (
    compose_event_remesh_cycle_observations,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.physics.runtime_remesh_schedule_stability import (
    RuntimeRemeshScheduleBoundaryObservation,
    RuntimeRemeshScheduleSequenceObservation,
    observe_runtime_remesh_schedule_sequence,
)


def _set_pure_epi_pressure(graph: nx.Graph) -> None:
    values = {node: float(graph.nodes[node]["EPI"]) for node in graph}
    for node in graph:
        neighbours = tuple(graph.neighbors(node))
        graph.nodes[node]["delta_nfr"] = (
            sum(values[item] for item in neighbours) / len(neighbours)
            - values[node]
        )


def _graph(
    *,
    current: tuple[float, float] = (2.0, 0.0),
    past: tuple[float, float] = (0.0, 2.0),
    vf: tuple[float, float] = (1.0, 1.0),
    alpha: float = 0.5,
) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=23,
        _gamma_spec={"type": "none"},
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=alpha,
        REMESH_ALPHA_HARD=True,
        REMESH_LOG_EVENTS=False,
        EPI_MIN=-10.0,
        EPI_MAX=10.0,
        CLIP_MODE="hard",
        DT_MIN=0.0,
        compute_delta_nfr=_set_pure_epi_pressure,
    )
    for node, epi in enumerate(current):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=vf[node],
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )
    graph.graph["_epi_hist"] = deque(
        [{node: value for node, value in enumerate(past)}],
        maxlen=64,
    )
    _set_pure_epi_pressure(graph)
    return graph


def _cycle(
    graph: nx.Graph,
    *,
    metric: tuple[float, float] = (1.0, 1.0),
    duration: float = 0.125,
):
    schedule = build_operator_event_schedule(
        (),
        start_time=graph.graph["_t"],
        flow_durations=(duration,),
    )
    return execute_event_remesh_cycle(
        graph,
        schedule,
        metric_weights=metric,
        refresh_pressure_after_remesh=True,
        include_stage_certificates=True,
    )


def _sequence(
    *,
    metrics: tuple[tuple[float, float], ...] = (
        (1.0, 1.0),
        (1.0, 1.0),
    ),
    vf: tuple[float, float] = (1.0, 1.0),
):
    graph = _graph(vf=vf)
    cycles = tuple(_cycle(graph, metric=metric) for metric in metrics)
    return compose_event_remesh_cycle_observations(cycles)


@pytest.fixture(scope="module")
def two_cycle_observation() -> RuntimeRemeshScheduleSequenceObservation:
    return observe_runtime_remesh_schedule_sequence(_sequence())


def test_public_facade_and_stub_expose_adjacent_cycle_api() -> None:
    import tnfr.physics as physics

    expected = {
        "RuntimeRemeshScheduleBoundaryObservation",
        "RuntimeRemeshScheduleSequenceObservation",
        "observe_runtime_remesh_schedule_sequence",
    }
    assert expected <= set(physics.__all__)
    assert (
        physics.RuntimeRemeshScheduleBoundaryObservation
        is RuntimeRemeshScheduleBoundaryObservation
    )
    assert (
        physics.RuntimeRemeshScheduleSequenceObservation
        is RuntimeRemeshScheduleSequenceObservation
    )
    assert (
        physics.observe_runtime_remesh_schedule_sequence
        is observe_runtime_remesh_schedule_sequence
    )

    package = Path(physics.__file__).parent
    stub = (package / "runtime_remesh_schedule_stability.pyi").read_text(
        encoding="utf-8"
    )
    assert "class RuntimeRemeshScheduleBoundaryObservation" in stub
    assert "class RuntimeRemeshScheduleSequenceObservation" in stub
    assert "def observe_runtime_remesh_schedule_sequence" in stub


def test_two_cycles_bind_schedule_endpoint_to_next_history(
    two_cycle_observation: RuntimeRemeshScheduleSequenceObservation,
) -> None:
    observation = two_cycle_observation
    boundary = observation.boundaries[0]

    assert type(observation) is RuntimeRemeshScheduleSequenceObservation
    assert type(boundary) is RuntimeRemeshScheduleBoundaryObservation
    assert boundary.exact_common_normalized_metric == (
        Fraction(1, 2),
        Fraction(1, 2),
    )
    assert boundary.exact_schedule_input_head == (
        Fraction(7, 16),
        Fraction(25, 16),
    )
    assert boundary.exact_scheduled_head == (
        Fraction(37, 64),
        Fraction(91, 64),
    )
    assert boundary.exact_scheduled_post_history == (
        boundary.exact_scheduled_head,
        (Fraction(7, 4), Fraction(1, 4)),
    )
    assert boundary.exact_next_cycle_history == (
        boundary.exact_scheduled_post_history
    )
    assert boundary.exact_recorded_history_advance_certified
    assert boundary.failed_conditions == ()


def test_two_cycle_exact_energy_balance_has_no_hidden_gain_product(
    two_cycle_observation: RuntimeRemeshScheduleSequenceObservation,
) -> None:
    observation = two_cycle_observation
    boundary = observation.boundaries[0]

    assert boundary.exact_augmented_energy_before == Fraction(3, 8)
    assert boundary.exact_augmented_energy_after == Fraction(351, 2048)
    assert boundary.exact_energy_drop == Fraction(417, 2048)
    assert (
        boundary.exact_gain_based_energy_drop_lower_bound
        == boundary.exact_energy_drop
    )
    assert boundary.exact_schedule_augmented_energy_gain_slack == 0
    assert boundary.exact_finite_schedule_remesh_balance_certified
    assert observation.exact_total_energy_drop == boundary.exact_energy_drop
    assert observation.exact_total_gain_based_energy_drop_lower_bound == (
        boundary.exact_gain_based_energy_drop_lower_bound
    )
    assert observation.exact_total_schedule_augmented_energy_gain_slack == 0
    assert observation.exact_finite_energy_telescope_certified
    assert not observation.runtime_global_gain_certified
    assert not observation.shared_graph_execution_provenance_certified
    assert not observation.whole_sequence_atomicity_certified
    assert not observation.repeated_runtime_stability_certified
    assert not observation.future_stability_certified


def test_three_cycles_telescope_the_evolving_history_exactly() -> None:
    observation = observe_runtime_remesh_schedule_sequence(
        _sequence(metrics=((1.0, 1.0),) * 3)
    )

    assert len(observation.boundaries) == 2
    left, right = observation.boundaries
    assert left.exact_scheduled_post_history == right.exact_transition.exact_history
    assert left.exact_augmented_energy_after == right.exact_augmented_energy_before
    assert observation.exact_total_energy_drop == (
        observation.exact_augmented_energy_initial
        - observation.exact_augmented_energy_final
    )
    assert observation.exact_total_energy_drop == sum(
        (item.exact_energy_drop for item in observation.boundaries),
        Fraction(0),
    )
    assert observation.sequence_observation_certified


def test_proportional_cycle_metrics_use_the_common_normalized_ray() -> None:
    observation = observe_runtime_remesh_schedule_sequence(
        _sequence(
            metrics=((1.0, 2.0), (2.0, 4.0)),
            vf=(1.0, 0.5),
        )
    )

    assert observation.exact_common_normalized_metric == (
        Fraction(1, 3),
        Fraction(2, 3),
    )
    assert observation.boundaries[0].exact_transition.exact_metric_weights == (
        Fraction(1, 3),
        Fraction(2, 3),
    )
    assert observation.sequence_observation_certified


def test_changed_remesh_configuration_is_rejected() -> None:
    graph = _graph()
    left = _cycle(graph)
    graph.graph["REMESH_ALPHA"] = 0.25
    right = _cycle(graph)
    sequence = compose_event_remesh_cycle_observations((left, right))

    assert sequence.exact_common_metric_cycle_sequence_certified
    assert not sequence.remesh_configurations_equal
    with pytest.raises(TNFRValueError, match="one REMESH configuration"):
        observe_runtime_remesh_schedule_sequence(sequence)


def test_missing_represented_schedule_composition_is_rejected() -> None:
    sequence = _sequence()
    graph = _graph()
    cycles = (_cycle(graph, duration=0.0), _cycle(graph, duration=0.0))
    no_schedule_sequence = compose_event_remesh_cycle_observations(cycles)

    assert sequence.exact_common_metric_cycle_sequence_certified
    assert not no_schedule_sequence.exact_common_metric_cycle_sequence_certified
    with pytest.raises(TNFRValueError, match="common schedule metric"):
        observe_runtime_remesh_schedule_sequence(no_schedule_sequence)


def test_tampered_nested_cycle_invalidates_the_outer_observation() -> None:
    sequence = _sequence()
    observation = observe_runtime_remesh_schedule_sequence(sequence)
    object.__setattr__(
        sequence.cycles[0],
        "exact_total_weighted_mean_drift",
        Fraction(99),
    )

    assert not observation.sequence_observation_certified
    assert observation.failed_conditions == (
        "runtime_remesh_schedule_sequence_proof_fields_intact",
    )


def test_private_reseal_cannot_promote_a_changed_boundary_derivative() -> None:
    observation = observe_runtime_remesh_schedule_sequence(_sequence())
    boundary = observation.boundaries[0]
    forged = replace(
        boundary,
        exact_energy_drop=Fraction(99),
        _proof_stamp=(),
    )
    values = bridge_module._object_values(
        forged,
        RuntimeRemeshScheduleBoundaryObservation,
    )
    object.__setattr__(
        forged,
        "_proof_stamp",
        bridge_module._stamp_from_values(
            bridge_module._BOUNDARY_PROOF_VERSION,
            bridge_module._BOUNDARY_FIELD_NAMES,
            values,
        ),
    )

    assert not forged.boundary_observation_certified
    assert forged.failed_conditions == (
        "runtime_remesh_schedule_boundary_proof_fields_intact",
    )


def test_hostile_resealed_payload_fails_without_equality_dispatch() -> None:
    equality_calls: list[str] = []

    class AlwaysEqual:
        def __eq__(self, other: object) -> bool:
            del other
            equality_calls.append("called")
            return True

    observation = observe_runtime_remesh_schedule_sequence(_sequence())
    forged = replace(
        observation,
        exact_total_energy_drop=AlwaysEqual(),
        _proof_stamp=(),
    )
    values = bridge_module._object_values(
        forged,
        RuntimeRemeshScheduleSequenceObservation,
    )
    object.__setattr__(
        forged,
        "_proof_stamp",
        bridge_module._stamp_from_values(
            bridge_module._SEQUENCE_PROOF_VERSION,
            bridge_module._SEQUENCE_FIELD_NAMES,
            values,
        ),
    )

    assert not forged.sequence_observation_certified
    assert equality_calls == []


def test_wrong_public_input_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="ObservedEventRemeshCycleSequence"):
        observe_runtime_remesh_schedule_sequence(object())  # type: ignore[arg-type]
