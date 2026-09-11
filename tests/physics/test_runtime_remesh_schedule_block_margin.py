"""Finite block margins from causally executed event/REMESH sequences."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import networkx as nx
import pytest

import tnfr.physics.runtime_remesh_schedule_block_margin as block_module
import tnfr.physics.runtime_remesh_schedule_stability as telescope_module
from tnfr.errors import TNFRValueError
from tnfr.operators.event_remesh_causal_runtime import (
    EventRemeshCycleExecutionSpec,
    ExecutedEventRemeshCycleSequence,
    execute_event_remesh_cycle_sequence,
)
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.physics.runtime_remesh_schedule_block_margin import (
    RuntimeRemeshScheduleBlockMarginObservation,
    observe_executed_event_remesh_block_margin,
)
from tnfr.physics.runtime_remesh_schedule_stability import (
    RuntimeRemeshScheduleBoundaryObservation,
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
) -> nx.Graph:
    graph = nx.path_graph(2)
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=23,
        _gamma_spec={"type": "none"},
        REMESH_TAU_GLOBAL=1,
        REMESH_TAU_LOCAL=1,
        REMESH_ALPHA=0.5,
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
            nu_f=1.0,
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


def _specs(count: int = 2) -> tuple[EventRemeshCycleExecutionSpec, ...]:
    result: list[EventRemeshCycleExecutionSpec] = []
    start = 0.0
    for _index in range(count):
        schedule = build_operator_event_schedule(
            (),
            start_time=start,
            flow_durations=(0.125,),
        )
        result.append(EventRemeshCycleExecutionSpec(schedule))
        start = schedule.end_time
    return tuple(result)


def _execution(
    *,
    count: int = 2,
    current: tuple[float, float] = (2.0, 0.0),
    past: tuple[float, float] = (0.0, 2.0),
) -> ExecutedEventRemeshCycleSequence:
    return execute_event_remesh_cycle_sequence(
        _graph(current=current, past=past),
        _specs(count),
        metric_weights=(1.0, 1.0),
    )


@pytest.fixture(scope="module")
def two_cycle_execution() -> ExecutedEventRemeshCycleSequence:
    return _execution()


@pytest.fixture(scope="module")
def three_cycle_execution() -> ExecutedEventRemeshCycleSequence:
    return _execution(count=3)


def test_direct_module_and_stub_expose_block_margin_api() -> None:
    assert block_module.__all__ == (
        "RuntimeRemeshScheduleBlockMarginObservation",
        "observe_executed_event_remesh_block_margin",
    )
    package = Path(block_module.__file__).parent
    stub = (package / "runtime_remesh_schedule_block_margin.pyi").read_text(
        encoding="utf-8"
    )
    assert "class RuntimeRemeshScheduleBlockMarginObservation" in stub
    assert "def observe_executed_event_remesh_block_margin" in stub


def test_single_boundary_has_exact_positive_block_margin(
    two_cycle_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    observation = observe_executed_event_remesh_block_margin(
        two_cycle_execution
    )

    assert type(observation) is RuntimeRemeshScheduleBlockMarginObservation
    assert observation.source_execution is two_cycle_execution
    assert observation.boundaries[0] is (
        two_cycle_execution.runtime_telescope.boundaries[0]
    )
    assert observation.start_boundary == 0
    assert observation.boundary_count == 1
    assert observation.exact_augmented_energy_before == Fraction(3, 8)
    assert observation.exact_augmented_energy_after == Fraction(351, 2048)
    assert observation.exact_gain_based_energy_drop_lower_bound == Fraction(
        417,
        2048,
    )
    assert observation.exact_schedule_augmented_energy_gain_slack == 0
    assert observation.exact_energy_drop == Fraction(417, 2048)
    assert (
        observation.exact_gain_based_energy_drop_fraction_lower_bound
        == Fraction(139, 256)
    )
    assert observation.exact_observed_energy_drop_fraction == Fraction(139, 256)
    assert observation.exact_endpoint_energy_gain_upper_bound == Fraction(
        117,
        256,
    )

    assert observation.block_observation_certified
    assert observation.exact_finite_block_balance_certified
    assert observation.energy_nonincrease_observed
    assert observation.energy_nonincrease_sufficiently_certified
    assert observation.positive_normalized_block_margin_certified
    assert observation.strict_energy_contraction_observed
    assert not observation.zero_energy_preservation_observed
    assert observation.failed_conditions == ()

    assert not observation.uniform_class_coercivity_certified
    assert not observation.uniform_repeated_margin_certified
    assert not observation.repeated_runtime_stability_certified
    assert not observation.future_stability_certified
    assert not observation.runtime_global_gain_certified
    assert not observation.full_tnfr_stability_certified
    assert not observation.solver_accuracy_certified
    assert not observation.solver_order_certified
    assert not observation.mesh_convergence_certified


def test_contiguous_subblock_reuses_authoritative_boundary_identities(
    three_cycle_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    execution = three_cycle_execution
    whole = observe_executed_event_remesh_block_margin(execution)
    right = observe_executed_event_remesh_block_margin(
        execution,
        start_boundary=1,
        boundary_count=1,
    )

    assert whole.boundary_count == 2
    assert whole.boundaries == execution.runtime_telescope.boundaries
    assert whole.exact_energy_drop == (
        whole.exact_augmented_energy_before
        - whole.exact_augmented_energy_after
    )
    assert whole.exact_energy_drop == (
        whole.exact_gain_based_energy_drop_lower_bound
        + whole.exact_schedule_augmented_energy_gain_slack
    )
    assert right.boundaries[0] is execution.runtime_telescope.boundaries[1]
    assert right.exact_augmented_energy_before == (
        whole.boundaries[1].exact_augmented_energy_before
    )
    assert right.block_observation_certified


def test_zero_initial_energy_keeps_normalized_diagnostics_undefined() -> None:
    execution = _execution(current=(1.0, 1.0), past=(1.0, 1.0))
    observation = observe_executed_event_remesh_block_margin(execution)

    assert observation.exact_augmented_energy_before == 0
    assert observation.exact_augmented_energy_after == 0
    assert observation.exact_energy_drop == 0
    assert observation.exact_gain_based_energy_drop_fraction_lower_bound is None
    assert observation.exact_observed_energy_drop_fraction is None
    assert observation.exact_endpoint_energy_gain_upper_bound is None
    assert observation.energy_nonincrease_sufficiently_certified
    assert not observation.positive_normalized_block_margin_certified
    assert not observation.strict_energy_contraction_observed
    assert observation.zero_energy_preservation_observed


@pytest.mark.parametrize(
    ("start", "count", "error"),
    (
        (True, None, TypeError),
        (0, True, TypeError),
        (-1, None, TNFRValueError),
        (2, None, TNFRValueError),
        (0, 0, TNFRValueError),
        (1, 2, TNFRValueError),
    ),
)
def test_invalid_or_empty_boundary_ranges_are_rejected(
    start: int,
    count: int | None,
    error: type[Exception],
    three_cycle_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    with pytest.raises(error):
        observe_executed_event_remesh_block_margin(
            three_cycle_execution,
            start_boundary=start,
            boundary_count=count,
        )


def test_nested_boundary_private_reseal_cannot_promote_changed_energy() -> None:
    execution = _execution()
    observation = observe_executed_event_remesh_block_margin(execution)
    boundary = execution.runtime_telescope.boundaries[0]
    object.__setattr__(boundary, "exact_energy_drop", Fraction(99))
    values = telescope_module._object_values(
        boundary,
        RuntimeRemeshScheduleBoundaryObservation,
    )
    object.__setattr__(
        boundary,
        "_proof_stamp",
        telescope_module._stamp_from_values(
            telescope_module._BOUNDARY_PROOF_VERSION,
            telescope_module._BOUNDARY_FIELD_NAMES,
            values,
        ),
    )

    assert not boundary.boundary_observation_certified
    assert not observation.block_observation_certified
    assert observation.failed_conditions == (
        "runtime_remesh_schedule_block_margin_proof_fields_intact",
    )


def test_private_outer_reseal_cannot_promote_changed_margin(
    two_cycle_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    observation = observe_executed_event_remesh_block_margin(
        two_cycle_execution
    )
    forged = replace(
        observation,
        exact_gain_based_energy_drop_lower_bound=Fraction(0),
        exact_gain_based_energy_drop_fraction_lower_bound=Fraction(0),
        exact_endpoint_energy_gain_upper_bound=Fraction(1),
        _proof_stamp=(),
    )
    values = block_module._observation_values(forged)
    object.__setattr__(
        forged,
        "_proof_stamp",
        block_module._proof_stamp_from_values(values),
    )

    assert not forged.block_observation_certified
    assert not forged.energy_nonincrease_sufficiently_certified


def test_private_reseal_cannot_replace_a_boundary_by_equal_value(
    two_cycle_execution: ExecutedEventRemeshCycleSequence,
) -> None:
    observation = observe_executed_event_remesh_block_margin(
        two_cycle_execution
    )
    copied_boundary = replace(observation.boundaries[0])
    assert copied_boundary.boundary_observation_certified
    forged = replace(
        observation,
        boundaries=(copied_boundary,),
        _proof_stamp=(),
    )
    values = block_module._observation_values(forged)
    object.__setattr__(
        forged,
        "_proof_stamp",
        block_module._proof_stamp_from_values(values),
    )

    assert not forged.block_observation_certified


def test_wrong_public_input_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="ExecutedEventRemeshCycleSequence"):
        observe_executed_event_remesh_block_margin(object())  # type: ignore[arg-type]
