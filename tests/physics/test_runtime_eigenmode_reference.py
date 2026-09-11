"""Executor-bound reversible single-eigenmode Euler reference contracts."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math

import networkx as nx
import pytest

import tnfr.physics.runtime_eigenmode_reference as runtime_module
from tnfr.errors import TNFRValueError
from tnfr.mathematics._neighbor_differences import mean_neighbor_difference
from tnfr.operators.event_runtime import (
    ExecutedPressureRefreshedFlowPartition,
    execute_operator_event_schedule,
)
from tnfr.operators.event_timing import (
    build_operator_event_schedule,
    build_physical_flow_partition,
)
from tnfr.physics.runtime_eigenmode_reference import (
    ExecutedReversibleSingleEigenmodeEulerPartitionObservation,
    ExecutedReversibleSingleEigenmodeEulerReferenceObservation,
    observe_executed_reversible_single_eigenmode_euler_reference,
)


F = Fraction

P3_EDGES = ((0, 1), (1, 2))
C3_EDGES = ((0, 1), (1, 2), (2, 0))
C4_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


def _refresh_pure_epi_pressure(graph: nx.Graph) -> None:
    """Use the canonical binary64 neighbor-difference reduction."""

    epi = {node: float(graph.nodes[node]["EPI"]) for node in graph}
    for node in graph:
        neighbors = tuple(graph.adj[node])
        graph.nodes[node]["delta_nfr"] = mean_neighbor_difference(
            epi[node],
            tuple(epi[neighbor] for neighbor in neighbors),
            tuple(
                float(graph.edges[node, neighbor].get("weight", 1.0))
                for neighbor in neighbors
            ),
        )


def _executed_partition(
    *,
    edges: tuple[tuple[int, int], ...],
    initial_epi: tuple[float, ...],
    nu_f: tuple[float, ...],
    durations: tuple[float, ...],
    use_default_pressure: bool = False,
) -> ExecutedPressureRefreshedFlowPartition:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(initial_epi)))
    graph.add_edges_from(edges, weight=1.0)
    graph.graph.update(
        _t=0.0,
        DT_MIN=0.0,
        GAMMA={"type": "none"},
        _gamma_spec={"type": "none"},
        EPI_MIN=-1.0e300,
        EPI_MAX=1.0e300,
        DNFR_WEIGHTS={
            "phase": 0.0,
            "epi": 1.0,
            "vf": 0.0,
            "topo": 0.0,
        },
    )
    if not use_default_pressure:
        graph.graph["compute_delta_nfr"] = _refresh_pure_epi_pressure
    for node, epi, capacity in zip(
        graph,
        initial_epi,
        nu_f,
        strict=True,
    ):
        graph.nodes[node].update(
            EPI=epi,
            epi_kind="test",
            nu_f=capacity,
            theta=0.0,
            delta_nfr=0.0,
            glyph_history=[],
        )

    total_duration = float(sum((F.from_float(item) for item in durations), F(0)))
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(total_duration,),
    )
    partition = build_physical_flow_partition(
        schedule.intervals[0],
        durations,
    )
    result = execute_operator_event_schedule(
        graph,
        schedule,
        physical_flow_partitions=(partition,),
        suppress_birth_warnings=True,
    )
    return result.physical_flow_partition_evidence[0]


def _observe(
    *partitions: ExecutedPressureRefreshedFlowPartition,
) -> ExecutedReversibleSingleEigenmodeEulerReferenceObservation:
    return observe_executed_reversible_single_eigenmode_euler_reference(partitions)


@pytest.fixture(scope="module")
def regular_c4_observation():
    execution = _executed_partition(
        edges=C4_EDGES,
        initial_epi=(1.0, 0.0, -1.0, 0.0),
        nu_f=(1.0, 1.0, 1.0, 1.0),
        durations=(0.25, 0.25),
    )
    return execution, _observe(execution)


@pytest.fixture(scope="module")
def nonregular_p3_observation():
    execution = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(4.0, 3.0, 2.0),
        nu_f=(1.0, 1.0, 1.0),
        durations=(0.25, 0.25),
    )
    return _observe(execution)


@pytest.fixture(scope="module")
def heterogeneous_p3_observation():
    execution = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(0.1, 0.0, -0.1),
        nu_f=(0.1, 0.5, 0.1),
        durations=(0.25, 0.25),
    )
    return _observe(execution)


@pytest.fixture(scope="module")
def rounded_p3_family():
    initial = (1.0e16, 1.0, -9_999_999_999_999_998.0)
    capacities = (1.0, 1.0, 1.0)
    return _observe(
        _executed_partition(
            edges=P3_EDGES,
            initial_epi=initial,
            nu_f=capacities,
            durations=(0.25, 0.25),
        ),
        _executed_partition(
            edges=P3_EDGES,
            initial_epi=initial,
            nu_f=capacities,
            durations=(0.125,) * 4,
        ),
    )


def test_regular_c4_runtime_binding_recovers_the_exact_mode(
    regular_c4_observation,
) -> None:
    execution, observation = regular_c4_observation
    reference = observation.reference_certificate

    assert type(observation) is (
        ExecutedReversibleSingleEigenmodeEulerReferenceObservation
    )
    assert observation.runtime_reference_binding_certified
    assert observation.failed_conditions == ()
    assert observation.nodes == (0, 1, 2, 3)
    assert reference.exact_degrees == (F(2),) * 4
    assert reference.exact_reversible_metric == (F(2),) * 4
    assert reference.exact_weighted_mean == 0
    assert reference.exact_centered_mode == (F(1), F(0), F(-1), F(0))
    assert reference.exact_mode_eigenvalue == 1
    assert observation.partition_observations[0].execution is execution


def test_nonregular_p3_runtime_binding_uses_the_reversible_metric(
    nonregular_p3_observation,
) -> None:
    observation = nonregular_p3_observation
    reference = observation.reference_certificate
    row = observation.partition_observations[0]

    assert observation.runtime_reference_binding_certified
    assert reference.exact_degrees == (F(1), F(2), F(1))
    assert reference.exact_reversible_metric == (F(1), F(2), F(1))
    assert reference.exact_weighted_mean == 3
    assert reference.exact_centered_mode == (F(1), F(0), F(-1))
    assert reference.exact_mode_eigenvalue == 1
    assert row.nodes == (0, 1, 2)
    assert row.execution.physical_pressure_reevaluated_partition_established


def test_heterogeneous_capacity_is_bound_without_assuming_a_scalar_nu(
    heterogeneous_p3_observation,
) -> None:
    observation = heterogeneous_p3_observation
    reference = observation.reference_certificate

    q = F.from_float(0.1)
    assert observation.runtime_reference_binding_certified
    assert reference.exact_nu_f == (q, F(1, 2), q)
    assert reference.exact_reversible_metric == (1 / q, F(4), 1 / q)
    assert reference.exact_mode_eigenvalue == q
    assert reference.exact_mode_residual == (F(0), F(0), F(0))


def test_large_off_mode_capacity_does_not_replace_the_certified_mode_rate() -> None:
    mean = float(2**45)
    scale = math.ulp(mean)
    mode = (1022.0, 511.0, -1.0)
    execution = _executed_partition(
        edges=P3_EDGES,
        initial_epi=tuple(mean + scale * value for value in mode),
        nu_f=(2.0, 1022.0, 1.0 / 512.0),
        durations=(0.25, 0.25),
        use_default_pressure=True,
    )
    observation = _observe(execution)
    reference = observation.reference_certificate
    row = observation.partition_observations[0]

    assert max(reference.exact_nu_f) == 1022
    assert reference.exact_mode_eigenvalue == 1
    assert max(reference.exact_nu_f) > 1000 * reference.exact_mode_eigenvalue
    assert not execution.all_segment_modal_decisions_stable
    assert execution.all_boundaries_binary64_pure_epi_pressure_realized
    assert execution.all_segment_binary64_replays_identified
    assert not execution.all_segment_exact_affine_maps_identified
    assert row.runtime_partition_binding_certified
    assert row.exact_runtime_defect_decomposition_certified
    assert any(
        any(value != 0 for value in defect)
        for defect in row.exact_local_runtime_defects
    )

    scalar_propagated = (F(0),) * len(row.nodes)
    for duration, defect in zip(
        row.exact_segment_durations,
        row.exact_local_runtime_defects,
        strict=True,
    ):
        scalar_factor = 1 - duration * reference.exact_mode_eigenvalue
        scalar_propagated = tuple(
            scalar_factor * previous + local
            for previous, local in zip(
                scalar_propagated,
                defect,
                strict=True,
            )
        )
    assert scalar_propagated != row.exact_endpoint_runtime_defect


def test_binary64_pressure_and_execution_residuals_are_kept_separate(
    rounded_p3_family,
) -> None:
    observation = rounded_p3_family
    row = observation.partition_observations[0]

    assert observation.runtime_reference_binding_certified
    assert row.runtime_partition_binding_certified
    assert row.execution.all_segment_binary64_replays_identified
    assert row.execution.all_boundaries_binary64_pure_epi_pressure_realized
    assert not row.execution.all_segment_exact_affine_maps_identified
    assert not row.all_segment_exact_affine_maps_identified
    assert row.exact_pressure_realization_residuals[0] == (
        F(-1),
        F(0),
        F(1),
    )
    assert any(
        any(value != 0 for value in residual)
        for residual in row.exact_pressure_realization_residuals
    )
    assert row.exact_held_input_execution_residuals[1] == (
        F(-1, 4),
        F(0),
        F(1, 4),
    )
    assert any(
        any(value != 0 for value in residual)
        for residual in row.exact_held_input_execution_residuals
    )


def test_local_and_propagated_runtime_defect_identities(
    rounded_p3_family,
) -> None:
    row = rounded_p3_family.partition_observations[0]
    reference = rounded_p3_family.reference_certificate
    durations = row.exact_segment_durations
    capacity = reference.exact_nu_f

    expected_local = tuple(
        tuple(
            duration * nu * rho_value + eta_value
            for nu, rho_value, eta_value in zip(
                capacity,
                rho,
                eta,
                strict=True,
            )
        )
        for duration, rho, eta in zip(
            durations,
            row.exact_pressure_realization_residuals,
            row.exact_held_input_execution_residuals,
            strict=True,
        )
    )
    assert row.exact_local_runtime_defects == expected_local

    propagated = (F(0),) * len(row.nodes)
    for duration, local in zip(durations, expected_local, strict=True):
        propagated = tuple(
            sum(
                (
                    (
                        (F(1) if source == target else F(0))
                        - duration * reference.exact_generator[source][target]
                    )
                    * propagated[target]
                    for target in range(len(row.nodes))
                ),
                F(0),
            )
            + local[source]
            for source in range(len(row.nodes))
        )

    assert row.exact_endpoint_runtime_defect == propagated
    assert row.exact_endpoint_runtime_defect == tuple(
        runtime - exact
        for runtime, exact in zip(
            row.exact_runtime_boundary_epi[-1],
            row.exact_reference_boundary_epi[-1],
            strict=True,
        )
    )
    assert row.exact_endpoint_runtime_defect_linf == max(
        abs(value) for value in propagated
    )


def test_signed_continuous_intervals_and_norm_bounds_are_propagated(
    rounded_p3_family,
) -> None:
    observation = rounded_p3_family
    reference = observation.reference_certificate
    row = observation.partition_observations[0]
    runtime_endpoint = row.exact_runtime_boundary_epi[-1]

    expected_lower = tuple(
        runtime - continuous_upper
        for runtime, continuous_upper in zip(
            runtime_endpoint,
            reference.exact_continuous_endpoint_upper_bound,
            strict=True,
        )
    )
    expected_upper = tuple(
        runtime - continuous_lower
        for runtime, continuous_lower in zip(
            runtime_endpoint,
            reference.exact_continuous_endpoint_lower_bound,
            strict=True,
        )
    )
    assert row.exact_runtime_minus_continuous_endpoint_lower_bound == (
        expected_lower
    )
    assert row.exact_runtime_minus_continuous_endpoint_upper_bound == (
        expected_upper
    )

    coordinate_minima = tuple(
        F(0) if lower <= 0 <= upper else min(abs(lower), abs(upper))
        for lower, upper in zip(expected_lower, expected_upper, strict=True)
    )
    coordinate_maxima = tuple(
        max(abs(lower), abs(upper))
        for lower, upper in zip(expected_lower, expected_upper, strict=True)
    )
    metric = reference.exact_reversible_metric
    assert row.exact_runtime_continuous_linf_error_lower_bound == max(
        coordinate_minima
    )
    assert row.exact_runtime_continuous_linf_error_upper_bound == max(
        coordinate_maxima
    )
    assert row.exact_runtime_continuous_h_energy_error_lower_bound == sum(
        (
            weight * value * value
            for weight, value in zip(metric, coordinate_minima, strict=True)
        ),
        F(0),
    ) / 2
    assert row.exact_runtime_continuous_h_energy_error_upper_bound == sum(
        (
            weight * value * value
            for weight, value in zip(metric, coordinate_maxima, strict=True)
        ),
        F(0),
    ) / 2
    assert row.exact_runtime_defect_decomposition_certified
    assert row.exact_continuous_error_enclosure_certified
    assert observation.exact_runtime_defect_decomposition_certified
    assert observation.exact_continuous_error_enclosures_certified


def test_rows_remain_bound_to_one_derived_reference_family(
    rounded_p3_family,
) -> None:
    observation = rounded_p3_family
    reference = observation.reference_certificate

    assert len(observation.partition_observations) == 2
    assert reference.exact_partitions == tuple(
        row.exact_segment_durations for row in observation.partition_observations
    )
    assert all(
        type(row) is ExecutedReversibleSingleEigenmodeEulerPartitionObservation
        and row.partition_index == index
        and row.execution.physical_pressure_reevaluated_partition_established
        and row.runtime_partition_binding_certified
        for index, row in enumerate(observation.partition_observations)
    )


def test_runtime_binding_rejects_mixed_modes() -> None:
    execution = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(2.0, -1.0, 0.0),
        nu_f=(1.0, 1.0, 1.0),
        durations=(0.25, 0.25),
    )

    with pytest.raises(TNFRValueError, match="one exact positive eigenmode"):
        _observe(execution)


def test_runtime_binding_rejects_incompatible_family_sources() -> None:
    first = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(1.0, 0.0, -1.0),
        nu_f=(1.0, 1.0, 1.0),
        durations=(0.25, 0.25),
    )
    changed_capacity = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(1.0, 0.0, -1.0),
        nu_f=(0.5, 1.0, 0.5),
        durations=(0.125,) * 4,
    )

    with pytest.raises(TNFRValueError, match="capacity|source|common"):
        _observe(first, changed_capacity)


def test_runtime_binding_rejects_changed_conductance_across_the_family() -> None:
    path = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(1.0, 0.0, -1.0),
        nu_f=(1.0, 1.0, 1.0),
        durations=(0.25, 0.25),
    )
    triangle = _executed_partition(
        edges=C3_EDGES,
        initial_epi=(1.0, 0.0, -1.0),
        nu_f=(1.0, 1.0, 1.0),
        durations=(0.125,) * 4,
    )

    with pytest.raises(TNFRValueError, match="conductance|generator|source|common"):
        _observe(path, triangle)


def test_runtime_binding_rejects_changed_initial_state_across_the_family() -> None:
    first = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(1.0, 0.0, -1.0),
        nu_f=(1.0, 1.0, 1.0),
        durations=(0.25, 0.25),
    )
    changed_initial = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(2.0, 0.0, -2.0),
        nu_f=(1.0, 1.0, 1.0),
        durations=(0.125,) * 4,
    )

    with pytest.raises(TNFRValueError, match="initial|source|common"):
        _observe(first, changed_initial)


def test_runtime_binding_rejects_changed_total_duration() -> None:
    half = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(1.0, 0.0, -1.0),
        nu_f=(1.0, 1.0, 1.0),
        durations=(0.25, 0.25),
    )
    quarter = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(1.0, 0.0, -1.0),
        nu_f=(1.0, 1.0, 1.0),
        durations=(0.125, 0.125),
    )

    with pytest.raises(TNFRValueError, match="duration|partition|common"):
        _observe(half, quarter)


def test_runtime_binding_rejects_incomparable_equal_horizon_partitions() -> None:
    first = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(1.0, 0.0, -1.0),
        nu_f=(1.0, 1.0, 1.0),
        durations=(0.25, 0.25),
    )
    incomparable = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(1.0, 0.0, -1.0),
        nu_f=(1.0, 1.0, 1.0),
        durations=(0.125, 0.375),
    )

    with pytest.raises(TNFRValueError, match="subdivision"):
        _observe(first, incomparable)


def test_runtime_binding_rejects_closed_euler_stability_boundary() -> None:
    boundary = _executed_partition(
        edges=P3_EDGES,
        initial_epi=(1.0, 0.0, -1.0),
        nu_f=(1.0, 1.0, 1.0),
        durations=(1.0, 0.25),
    )

    with pytest.raises(TNFRValueError, match="modal step"):
        _observe(boundary)


@pytest.mark.parametrize("bad_input", ((), {"unordered"}, (object(),)))
def test_runtime_binding_rejects_missing_unordered_or_wrong_evidence(
    bad_input,
) -> None:
    with pytest.raises(
        (TypeError, TNFRValueError),
        match="partition|sequence|evidence",
    ):
        observe_executed_reversible_single_eigenmode_euler_reference(bad_input)


def test_nested_execution_tampering_invalidates_both_observation_layers() -> None:
    observation = _observe(
        _executed_partition(
            edges=P3_EDGES,
            initial_epi=(4.0, 3.0, 2.0),
            nu_f=(1.0, 1.0, 1.0),
            durations=(0.25, 0.25),
        )
    )
    row = observation.partition_observations[0]
    nested = row.execution.segment_flow_evidence[0].certificate
    assert nested is not None

    object.__setattr__(nested.left, "exact_epi", (F(99), F(3), F(2)))

    assert not row.runtime_partition_binding_certified
    assert not observation.runtime_reference_binding_certified


def test_direct_nested_row_tampering_cannot_promote_outer_exactness() -> None:
    observation = _observe(
        _executed_partition(
            edges=P3_EDGES,
            initial_epi=(1.0e16, 1.0, -9_999_999_999_999_998.0),
            nu_f=(1.0, 1.0, 1.0),
            durations=(0.25, 0.25),
        )
    )
    row = observation.partition_observations[0]
    zero = (F(0),) * len(row.nodes)
    assert row.exact_endpoint_runtime_defect != zero
    assert not observation.all_observed_endpoints_equal_exact_euler_reference

    object.__setattr__(row, "exact_endpoint_runtime_defect", zero)

    assert not row.runtime_partition_binding_certified
    assert not observation.runtime_reference_binding_certified
    assert not observation.all_observed_endpoints_equal_exact_euler_reference


def test_derived_row_reference_and_order_tampering_fail_closed(
    rounded_p3_family,
) -> None:
    observation = rounded_p3_family
    first = observation.partition_observations[0]
    altered_row = replace(
        first,
        exact_endpoint_runtime_defect=(F(0),) * len(first.nodes),
    )
    altered_reference = replace(
        observation.reference_certificate,
        exact_mode_eigenvalue=F(99),
    )
    reordered = replace(
        observation,
        partition_observations=tuple(
            reversed(observation.partition_observations)
        ),
    )

    assert not altered_row.runtime_partition_binding_certified
    assert not replace(
        observation,
        reference_certificate=altered_reference,
    ).runtime_reference_binding_certified
    assert not reordered.runtime_reference_binding_certified


def test_scope_does_not_promote_finite_binary64_observation(
    rounded_p3_family,
) -> None:
    observation = rounded_p3_family

    assert not observation.binary64_asymptotic_convergence_certified
    assert not observation.runtime_mesh_convergence_certified
    assert not observation.solver_accuracy_certified
    assert not observation.solver_order_certified
    assert not observation.glyph_or_remesh_dynamics_certified
    assert not observation.future_or_repeated_stability_certified
    assert not observation.common_causal_execution_provenance_certified
    assert not observation.full_tnfr_stability_certified
    assert "binary64" in observation.scope


def test_private_reseal_cannot_legitimize_a_changed_derived_row(
    nonregular_p3_observation,
) -> None:
    observation = nonregular_p3_observation
    row = observation.partition_observations[0]
    forged = replace(
        row,
        exact_endpoint_runtime_defect=(F(7), F(7), F(7)),
        _proof_stamp=(),
    )
    resealed = runtime_module._seal_partition(forged)

    assert not resealed.runtime_partition_binding_certified

    forged_reference = replace(
        observation,
        nodes=tuple(reversed(observation.nodes)),
        _proof_stamp=(),
    )
    resealed_reference = runtime_module._seal_reference(forged_reference)
    assert not resealed_reference.runtime_reference_binding_certified
