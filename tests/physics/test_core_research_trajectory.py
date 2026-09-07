"""Time-resolved and mesh-agreement checks for the restricted S16 chain."""

from __future__ import annotations

import copy
from fractions import Fraction
import math

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.core_research_trajectory import (
    CoreResearchRefinementComparison,
    CoreResearchRefinementSample,
    CoreResearchTrajectoryCertificate,
    CoreResearchTrajectoryIntervalCertificate,
    certify_core_research_trajectory,
    compare_core_research_trajectory_refinement,
)
from tnfr.physics.structural_diffusion import structural_diffusion_operator
from tnfr.physics.structural_state_distance import StructuralChannelScales


SCALES = StructuralChannelScales(
    epi=1.0,
    frequency=1.0,
    phase=math.pi,
    pressure=1.0,
    epi_rate=1.0,
    edge_conductance=1.0,
    edge_length=1.0,
)
EXACT_PARTITION = ((0, 3), (1, 2))
BASE_FREQUENCY = np.array([0.5, 1.5, 1.5, 0.5])
BASE_PHASE = np.array([0.0, 0.2, 0.4, 0.6])
BASE_WEIGHTS = np.array([2.0, 0.75, 2.0])


def _state(
    epi,
    *,
    frequency=BASE_FREQUENCY,
    phase=BASE_PHASE,
    weights=BASE_WEIGHTS,
) -> nx.Graph:
    graph = nx.path_graph(4)
    for edge, weight in zip(graph.edges(), weights):
        graph.edges[edge]["weight"] = float(weight)
    for node, value, rate, angle in zip(graph, epi, frequency, phase):
        graph.nodes[node].update(
            EPI=float(value),
            nu_f=float(rate),
            theta=float(angle),
        )

    nodes, laplacian = structural_diffusion_operator(graph)
    field = np.asarray([graph.nodes[node]["EPI"] for node in nodes], dtype=float)
    pressure = -(laplacian @ field)
    for node, value in zip(nodes, pressure):
        graph.nodes[node]["delta_nfr"] = float(value)
        graph.nodes[node]["dEPI_dt"] = float(
            graph.nodes[node]["nu_f"] * value
        )
    return graph


def _advance(left: nx.Graph, dt: float, **next_state_kwargs) -> nx.Graph:
    epi = np.asarray([left.nodes[node]["EPI"] for node in left], dtype=float)
    rate = np.asarray(
        [
            left.nodes[node]["nu_f"] * left.nodes[node]["delta_nfr"]
            for node in left
        ],
        dtype=float,
    )
    return _state(epi + dt * rate, **next_state_kwargs)


def _constant_trajectory():
    times = (0.0, 0.05, 0.1)
    first = _state([2.0, -1.0, 3.0, 0.5])
    second = _advance(first, times[1] - times[0])
    third = _advance(second, times[2] - times[1])
    return (first, second, third), times


def _trajectory_on_grid(times):
    snapshots = [_state([2.0, -1.0, 3.0, 0.5])]
    for left_time, right_time in zip(times, times[1:]):
        snapshots.append(_advance(snapshots[-1], right_time - left_time))
    return tuple(snapshots)


def _graph_payload(graph: nx.Graph):
    return (
        copy.deepcopy(graph.graph),
        copy.deepcopy(tuple(graph.nodes(data=True))),
        copy.deepcopy(tuple(graph.edges(data=True))),
    )


def test_forward_euler_trajectory_passes_all_temporal_conditions():
    snapshots, times = _constant_trajectory()

    result = certify_core_research_trajectory(
        (snapshot for snapshot in snapshots),
        (time for time in times),
        (iter(block) for block in EXACT_PARTITION),
        scales=SCALES,
    )

    assert isinstance(result, CoreResearchTrajectoryCertificate)
    assert len(result.intervals) == 2
    assert all(
        isinstance(interval, CoreResearchTrajectoryIntervalCertificate)
        for interval in result.intervals
    )
    assert result.integration_rule == "forward_euler"
    assert result.spectral_tolerance == pytest.approx(1e-12)
    assert result.nodes == tuple(snapshots[0])
    assert result.times == times
    assert result.partition == EXACT_PARTITION
    assert result.fixed_transport_operator
    assert result.maximum_transport_operator_entry_difference == 0.0
    assert result.switching_stability.supports_exact_switching_theorem
    assert result.joint_temporal_conditions_pass
    assert result.failed_conditions == ()
    assert result.first_failed_interval_index is None
    assert result.first_failed_condition is None
    assert result.first_failed_conditions == ()
    assert all(interval.interval_conditions_pass for interval in result.intervals)
    assert all(
        interval.euler_relaxation.is_euler_stable for interval in result.intervals
    )
    assert all(
        interval.scaled_epi_update_residual_linf <= result.tolerance
        for interval in result.intervals
    )
    assert len(result.common_lyapunov_values) == len(snapshots)
    assert len(result.common_lyapunov_scaled_increments) == len(snapshots) - 1
    assert result.maximum_scaled_common_lyapunov_increase <= result.tolerance
    assert (
        result.cumulative_scaled_common_lyapunov_positive_variation
        <= result.tolerance
    )
    assert all(
        increment <= result.tolerance
        for increment in result.common_lyapunov_scaled_increments
    )
    assert all(
        interval.euler_relaxation.spectral_relative_tolerance
        == pytest.approx(result.spectral_tolerance)
        for interval in result.intervals
    )
    assert all(
        interval.euler_relaxation.spectral_zero_threshold
        == pytest.approx(
            result.spectral_tolerance
            * interval.euler_relaxation.fastest_decay_rate
        )
        for interval in result.intervals
    )
    assert result.scales is SCALES
    assert "sampled regimes only" in result.scope
    assert "Intermediate unobserved regimes" in result.scope
    assert "OPEN" in result.scope


def test_exact_common_metric_allows_sampled_switching_without_fixed_transport():
    first = _state([2.0, -1.0, 3.0, 0.5])
    second = _advance(
        first,
        0.05,
        frequency=2.0 * BASE_FREQUENCY,
        weights=2.0 * BASE_WEIGHTS,
    )
    third = _advance(second, 0.05)

    result = certify_core_research_trajectory(
        (first, second, third),
        (0.0, 0.05, 0.1),
        EXACT_PARTITION,
        scales=SCALES,
    )

    assert result.switching_stability.supports_exact_switching_theorem
    assert not result.fixed_transport_operator
    assert result.maximum_transport_operator_entry_difference > 0.0
    assert result.joint_temporal_conditions_pass


def test_common_lyapunov_reprojects_the_weighted_mean_at_every_snapshot():
    first_epi = np.array([0.25, -0.5, 0.75, 0.0])
    first = _state(first_epi)
    translated = _state(first_epi + 0.25)

    result = certify_core_research_trajectory(
        (first, translated),
        (0.0, 0.05),
        EXACT_PARTITION,
        scales=SCALES,
    )

    assert result.common_lyapunov_values[1] == pytest.approx(
        result.common_lyapunov_values[0]
    )


def test_common_lyapunov_uses_exactly_normalized_proved_reference_metric():
    epi = np.array([0.0, 1.0, 4.0, -2.0])
    weights = np.ones(3)
    frequency = np.array([1.0, 4.0 / 3.0, 1.0, 0.4])
    first = _state(epi, frequency=frequency, weights=weights)
    second = _advance(
        first,
        0.01,
        frequency=frequency,
        weights=weights,
    )

    result = certify_core_research_trajectory(
        (first, second),
        (0.0, 0.01),
        EXACT_PARTITION,
        scales=SCALES,
    )

    reference = tuple(
        Fraction.from_float(float(value))
        for value in result.switching_stability.reference_metric_weights
    )
    total = sum(reference, Fraction())
    proved_weights = tuple(weight / total for weight in reference)
    field = tuple(Fraction.from_float(float(value)) for value in epi)
    center = sum(
        (weight * value for weight, value in zip(proved_weights, field)),
        Fraction(),
    )
    expected = sum(
        (
            weight * (value - center) ** 2 / 2
            for weight, value in zip(proved_weights, field)
        ),
        Fraction(),
    )

    rounded = tuple(
        Fraction.from_float(float(value))
        for value in result.switching_stability.normalized_metric_weights
    )
    rounded_total = sum(rounded, Fraction())
    rounded_center = sum(
        (weight * value for weight, value in zip(rounded, field)),
        Fraction(),
    ) / rounded_total
    old_nearby_value = sum(
        (
            weight * (value - rounded_center) ** 2 / 2
            for weight, value in zip(rounded, field)
        ),
        Fraction(),
    )

    assert result.switching_stability.reference_metric_weights.tolist() == [
        1.0,
        1.5,
        2.0,
        2.5,
    ]
    assert expected != old_nearby_value
    assert result.common_lyapunov_values[0] == float(expected)


def test_common_lyapunov_boundary_is_decided_before_float_rounding():
    first = _state([0.0, 0.0, 0.0, 0.0])
    second_epi = [0.2, 0.0, -0.1, 0.3]
    second = _state(second_epi)

    probe = certify_core_research_trajectory(
        (first, second),
        (0.0, 0.01),
        EXACT_PARTITION,
        scales=SCALES,
        tolerance=0.1,
    )
    reference = tuple(
        Fraction.from_float(float(value))
        for value in probe.switching_stability.reference_metric_weights
    )
    total = sum(reference, Fraction())
    weights = tuple(weight / total for weight in reference)
    field = tuple(Fraction.from_float(value) for value in second_epi)
    center = sum(
        (weight * value for weight, value in zip(weights, field)), Fraction()
    )
    exact_increase = sum(
        (
            weight * (value - center) ** 2 / 2
            for weight, value in zip(weights, field)
        ),
        Fraction(),
    )
    tolerance = float(exact_increase)
    assert Fraction.from_float(tolerance) < exact_increase

    result = certify_core_research_trajectory(
        (first, second),
        (0.0, 0.01),
        EXACT_PARTITION,
        scales=SCALES,
        tolerance=tolerance,
    )

    assert result.maximum_scaled_common_lyapunov_increase == tolerance
    conditions = dict(result.numerical_conditions)
    assert not conditions["observed_common_lyapunov_nonincrease"]
    assert not conditions["cumulative_common_lyapunov_positive_variation"]


def test_spectral_resolution_is_relative_under_global_frequency_scaling():
    frequency = 1e-12 * BASE_FREQUENCY
    first = _state([2.0, -1.0, 3.0, 0.5], frequency=frequency)
    second = _advance(first, 1.0, frequency=frequency)

    result = certify_core_research_trajectory(
        (first, second),
        (0.0, 1.0),
        EXACT_PARTITION,
        scales=SCALES,
        tolerance=1e-10,
        spectral_tolerance=1e-12,
    )

    diagnostic = result.intervals[0].euler_relaxation
    assert result.joint_temporal_conditions_pass
    assert diagnostic.slowest_decay_rate == pytest.approx(
        1e-12 * 0.18384038305065373
    )
    assert diagnostic.spectral_relative_tolerance == pytest.approx(1e-12)
    assert diagnostic.spectral_zero_threshold == pytest.approx(
        1e-12 * diagnostic.fastest_decay_rate
    )


def test_discrete_residual_blocks_first_interval_without_rejecting_endpoints():
    first = _state([2.0, -1.0, 3.0, 0.5])
    second = _advance(first, 0.05)
    second.nodes[0]["EPI"] += 1e-3
    second = _state(
        [second.nodes[node]["EPI"] for node in second],
    )

    result = certify_core_research_trajectory(
        (first, second),
        (0.0, 0.05),
        EXACT_PARTITION,
        scales=SCALES,
    )

    interval = result.intervals[0]
    assert interval.endpoint_certificate.joint_numerical_conditions_pass
    assert not interval.interval_conditions_pass
    assert interval.failed_conditions == ("forward_euler_epi_update",)
    assert interval.scaled_epi_update_residual_linf > result.tolerance
    assert not result.joint_temporal_conditions_pass
    assert result.first_failed_interval_index == 0
    assert result.first_failed_condition == "forward_euler_epi_update"
    assert result.first_failed_conditions == ("forward_euler_epi_update",)


def test_forward_euler_boundary_is_decided_before_float_rounding():
    tolerance = 1e-10
    dt = np.nextafter(0.0, 1.0)
    first = _state([0.0, -1.0, 0.0, 0.0])
    second = _state([tolerance, -1.0, 0.0, 0.0])
    for node in first:
        pressure = -1.0 if node == 0 else 0.0
        first.nodes[node]["delta_nfr"] = pressure
        first.nodes[node]["dEPI_dt"] = first.nodes[node]["nu_f"] * pressure

    result = certify_core_research_trajectory(
        (first, second),
        (0.0, dt),
        EXACT_PARTITION,
        scales=SCALES,
        tolerance=tolerance,
    )

    interval = result.intervals[0]
    # The exact residual is tolerance + dt/2, but nearest binary64 rounds it
    # back to tolerance. The certificate condition must still reject it.
    assert interval.scaled_epi_update_residual_linf == tolerance
    assert not dict(interval.numerical_conditions)["forward_euler_epi_update"]


def test_modally_unstable_forward_euler_step_is_not_promoted():
    first = _state([2.0, -1.0, 3.0, 0.5])
    second = _advance(first, 2.0)

    result = certify_core_research_trajectory(
        (first, second),
        (0.0, 2.0),
        EXACT_PARTITION,
        scales=SCALES,
    )

    interval = result.intervals[0]
    assert interval.scaled_epi_update_residual_linf <= result.tolerance
    assert not interval.euler_relaxation.is_euler_stable
    assert "forward_euler_modal_stability" in interval.failed_conditions
    assert result.first_failed_interval_index == 0
    assert result.first_failed_condition == "forward_euler_modal_stability"
    assert result.first_failed_conditions == ("forward_euler_modal_stability",)
    assert not result.joint_temporal_conditions_pass


def test_endpoint_failure_precedes_later_discrete_conditions():
    first = _state([2.0, -1.0, 3.0, 0.5])
    second = _advance(first, 0.05)
    second.nodes[0]["dEPI_dt"] += 0.25

    result = certify_core_research_trajectory(
        (first, second),
        (0.0, 0.05),
        EXACT_PARTITION,
        scales=SCALES,
    )

    assert result.intervals[0].scaled_epi_update_residual_linf <= result.tolerance
    assert result.first_failed_interval_index == 0
    assert result.first_failed_condition == "right_nodal_equation_consistency"
    assert not result.joint_temporal_conditions_pass


def test_missing_common_lyapunov_is_a_global_noninterval_failure():
    first = _state([2.0, -1.0, 3.0, 0.5])
    second = _advance(
        first,
        0.05,
        frequency=np.array([0.5, 2.0, 2.0, 0.5]),
    )

    result = certify_core_research_trajectory(
        (first, second),
        (0.0, 0.05),
        EXACT_PARTITION,
        scales=SCALES,
    )

    assert result.intervals[0].interval_conditions_pass
    assert not result.switching_stability.supports_exact_switching_theorem
    assert not result.joint_temporal_conditions_pass
    assert result.first_failed_interval_index is None
    assert result.first_failed_condition == "sampled_switching_common_lyapunov"
    assert "sampled_switching_common_lyapunov" in result.first_failed_conditions


def test_observed_common_lyapunov_catches_small_step_residual_at_large_amplitude():
    first = _state([2000.0, -1000.0, 3000.0, 500.0])
    second = _advance(first, 1e-12)
    perturbed_epi = [second.nodes[node]["EPI"] for node in second]
    perturbed_epi[2] += 5e-4
    second = _state(perturbed_epi)

    result = certify_core_research_trajectory(
        (first, second),
        (0.0, 1e-12),
        EXACT_PARTITION,
        scales=SCALES,
        tolerance=1e-3,
    )

    assert result.intervals[0].interval_conditions_pass
    assert result.switching_stability.supports_exact_switching_theorem
    assert result.maximum_scaled_common_lyapunov_increase > result.tolerance
    assert not result.joint_temporal_conditions_pass
    assert result.first_failed_interval_index is None
    assert result.first_failed_conditions == (
        "observed_common_lyapunov_nonincrease",
        "cumulative_common_lyapunov_positive_variation",
    )


def test_cumulative_lyapunov_budget_blocks_many_individually_small_increases():
    tolerance = 1e-3
    step_count = 60
    uniform_increment = 9e-4
    snapshots = tuple(
        _state(
            [
                1.0 + index * uniform_increment,
                1.0 - index * uniform_increment,
                1.0 + index * uniform_increment,
                1.0 - index * uniform_increment,
            ]
        )
        for index in range(step_count + 1)
    )
    times = tuple(1e-4 * index for index in range(step_count + 1))

    result = certify_core_research_trajectory(
        snapshots,
        times,
        EXACT_PARTITION,
        scales=SCALES,
        tolerance=tolerance,
    )

    assert all(interval.interval_conditions_pass for interval in result.intervals)
    assert result.maximum_scaled_common_lyapunov_increase < tolerance
    assert result.cumulative_scaled_common_lyapunov_positive_variation > tolerance
    assert result.failed_conditions == (
        "cumulative_common_lyapunov_positive_variation",
    )
    assert result.first_failed_interval_index is None
    assert result.first_failed_condition == (
        "cumulative_common_lyapunov_positive_variation"
    )
    assert not result.joint_temporal_conditions_pass


def test_certificate_does_not_mutate_snapshot_or_container_inputs():
    snapshots, times = _constant_trajectory()
    before = tuple(_graph_payload(snapshot) for snapshot in snapshots)
    times_list = list(times)
    partition_list = [list(block) for block in EXACT_PARTITION]

    certify_core_research_trajectory(
        list(snapshots),
        times_list,
        partition_list,
        scales=SCALES,
    )

    assert tuple(_graph_payload(snapshot) for snapshot in snapshots) == before
    assert times_list == list(times)
    assert partition_list == [list(block) for block in EXACT_PARTITION]


@pytest.mark.parametrize("bad_time", [True, np.bool_(False), math.nan, math.inf])
def test_times_reject_booleans_nan_and_infinity(bad_time):
    snapshots, _ = _constant_trajectory()

    with pytest.raises(ValueError, match="finite real timestamps"):
        certify_core_research_trajectory(
            snapshots[:2],
            (0.0, bad_time),
            EXACT_PARTITION,
            scales=SCALES,
        )


@pytest.mark.parametrize(
    "times, message",
    [
        ((0.0,), "length"),
        ((0.0, 0.0), "strictly increasing"),
        ((1.0, 0.0), "strictly increasing"),
        ((-1e308, 1e308), "interval length"),
    ],
)
def test_times_require_matching_length_and_strict_finite_intervals(times, message):
    snapshots, _ = _constant_trajectory()

    with pytest.raises(ValueError, match=message):
        certify_core_research_trajectory(
            snapshots[:2], times, EXACT_PARTITION, scales=SCALES
        )


@pytest.mark.parametrize(
    "tolerance", [True, np.bool_(True), 0.0, 1.0, math.nan, math.inf]
)
def test_relative_tolerance_has_one_strict_validated_contract(tolerance):
    snapshots, times = _constant_trajectory()

    with pytest.raises(ValueError, match="open interval"):
        certify_core_research_trajectory(
            snapshots, times, EXACT_PARTITION, scales=SCALES, tolerance=tolerance
        )


@pytest.mark.parametrize(
    "spectral_tolerance", [True, np.bool_(True), 0.0, 1.0, math.nan, math.inf]
)
def test_spectral_tolerance_is_a_separate_relative_contract(spectral_tolerance):
    snapshots, times = _constant_trajectory()

    with pytest.raises(ValueError, match="spectral_tolerance.*open interval"):
        certify_core_research_trajectory(
            snapshots,
            times,
            EXACT_PARTITION,
            scales=SCALES,
            spectral_tolerance=spectral_tolerance,
        )


@pytest.mark.parametrize("rule", [True, None, 1])
def test_integration_rule_requires_a_string(rule):
    snapshots, times = _constant_trajectory()

    with pytest.raises(TypeError, match="integration_rule must be a string"):
        certify_core_research_trajectory(
            snapshots,
            times,
            EXACT_PARTITION,
            scales=SCALES,
            integration_rule=rule,
        )


def test_undeclared_integration_rule_is_rejected():
    snapshots, times = _constant_trajectory()

    with pytest.raises(ValueError, match="forward_euler"):
        certify_core_research_trajectory(
            snapshots,
            times,
            EXACT_PARTITION,
            scales=SCALES,
            integration_rule="trapezoidal",
        )


def test_snapshot_and_scale_types_are_validated_before_composition():
    snapshots, times = _constant_trajectory()
    with pytest.raises(ValueError, match="at least two snapshots"):
        certify_core_research_trajectory(
            (snapshots[0],), (0.0,), EXACT_PARTITION, scales=SCALES
        )
    with pytest.raises(TypeError, match="iterable of NetworkX"):
        certify_core_research_trajectory(
            snapshots[0], times, EXACT_PARTITION, scales=SCALES
        )
    with pytest.raises(TypeError, match="only NetworkX"):
        certify_core_research_trajectory(
            (snapshots[0], object()), (0.0, 0.1), EXACT_PARTITION, scales=SCALES
        )
    with pytest.raises(TypeError, match="StructuralChannelScales"):
        certify_core_research_trajectory(
            snapshots, times, EXACT_PARTITION, scales=object()
        )


def test_fixed_support_and_persistent_node_ids_are_required():
    snapshots, _ = _constant_trajectory()
    changed_edge = snapshots[1].copy()
    changed_edge.add_edge(0, 2, weight=1.0)
    changed_node = snapshots[1].copy()
    nx.relabel_nodes(changed_node, {0: "zero"}, copy=False)

    with pytest.raises(ValueError, match="fixed bare edge support"):
        certify_core_research_trajectory(
            (snapshots[0], changed_edge),
            (0.0, 0.05),
            EXACT_PARTITION,
            scales=SCALES,
        )
    with pytest.raises(ValueError, match="persistent node identifiers"):
        certify_core_research_trajectory(
            (snapshots[0], changed_node),
            (0.0, 0.05),
            EXACT_PARTITION,
            scales=SCALES,
        )


def test_boolean_and_nonfinite_state_channels_are_rejected():
    snapshots, _ = _constant_trajectory()
    boolean_epi = snapshots[0].copy()
    boolean_epi.nodes[0]["EPI"] = True
    infinite_pressure = snapshots[0].copy()
    infinite_pressure.nodes[0]["delta_nfr"] = math.inf

    with pytest.raises(ValueError, match="finite"):
        certify_core_research_trajectory(
            (boolean_epi, snapshots[1]),
            (0.0, 0.05),
            EXACT_PARTITION,
            scales=SCALES,
        )
    with pytest.raises(ValueError, match="finite"):
        certify_core_research_trajectory(
            (infinite_pressure, snapshots[1]),
            (0.0, 0.05),
            EXACT_PARTITION,
            scales=SCALES,
        )


def test_nested_fine_grid_reports_common_time_agreement_without_convergence_claim():
    equilibrium = _state([1.0, 1.0, 1.0, 1.0])
    fine_snapshots = (equilibrium.copy(), equilibrium.copy(), equilibrium.copy())
    fine_times = (0.0, 0.05, 0.1)
    coarse_snapshots = (fine_snapshots[0], fine_snapshots[-1])
    coarse_times = (fine_times[0], fine_times[-1])

    result = compare_core_research_trajectory_refinement(
        coarse_snapshots,
        coarse_times,
        fine_snapshots,
        fine_times,
        EXACT_PARTITION,
        scales=SCALES,
        agreement_tolerance=1e-8,
        same_dynamics_declared=True,
    )

    assert isinstance(result, CoreResearchRefinementComparison)
    assert all(
        isinstance(sample, CoreResearchRefinementSample) for sample in result.samples
    )
    assert result.common_time_pairs == ((0.0, 0.0), (0.1, 0.1))
    assert result.maximum_direct_epi_error == 0.0
    assert result.maximum_scaled_direct_epi_error == 0.0
    assert result.all_coarse_times_matched
    assert result.fine_grid_is_strict_refinement
    assert result.common_time_agreement_within_tolerance
    assert result.coarse_trajectory.joint_temporal_conditions_pass
    assert result.fine_trajectory.joint_temporal_conditions_pass
    assert result.maximum_fine_step < result.maximum_coarse_step
    assert result.joint_refinement_conditions_pass
    assert result.failed_conditions == ()
    assert result.trajectory_tolerance == pytest.approx(1e-10)
    assert result.spectral_tolerance == pytest.approx(1e-12)
    assert result.agreement_tolerance == pytest.approx(1e-8)
    assert result.same_dynamics_declared
    assert result.time_tolerance == 0.0
    assert "does not prove convergence" in result.scope


def test_nontrivial_euler_refinement_compares_two_certified_paths():
    coarse_times = (0.0, 0.04, 0.08)
    fine_times = (0.0, 0.02, 0.04, 0.06, 0.08)
    coarse_snapshots = _trajectory_on_grid(coarse_times)
    fine_snapshots = _trajectory_on_grid(fine_times)

    result = compare_core_research_trajectory_refinement(
        coarse_snapshots,
        coarse_times,
        fine_snapshots,
        fine_times,
        EXACT_PARTITION,
        scales=SCALES,
        agreement_tolerance=1e-2,
        same_dynamics_declared=True,
    )

    assert result.coarse_trajectory.joint_temporal_conditions_pass
    assert result.fine_trajectory.joint_temporal_conditions_pass
    assert result.fine_grid_is_strict_refinement
    assert result.maximum_fine_step < result.maximum_coarse_step
    assert 0.0 < result.maximum_scaled_direct_epi_error < 1e-2
    assert result.common_time_agreement_within_tolerance
    assert result.joint_refinement_conditions_pass
    assert "convergence" in result.scope


def test_extra_fine_sample_without_smaller_maximum_step_is_not_strict_refinement():
    equilibrium = _state([1.0, 1.0, 1.0, 1.0])
    coarse_times = (0.0, 0.5, 1.0)
    fine_times = (0.0, 0.25, 0.5, 1.0)

    result = compare_core_research_trajectory_refinement(
        tuple(equilibrium.copy() for _ in coarse_times),
        coarse_times,
        tuple(equilibrium.copy() for _ in fine_times),
        fine_times,
        EXACT_PARTITION,
        scales=SCALES,
        agreement_tolerance=1e-8,
        same_dynamics_declared=True,
    )

    assert result.all_coarse_times_matched
    assert len(result.fine_times) > len(result.coarse_times)
    assert result.maximum_fine_step == result.maximum_coarse_step
    assert not result.fine_grid_is_strict_refinement
    assert not result.joint_refinement_conditions_pass
    assert "fine_grid_is_strict_refinement" in result.failed_conditions


def test_refinement_requires_explicit_same_dynamics_declaration_for_promotion():
    equilibrium = _state([1.0, 1.0, 1.0, 1.0])
    result = compare_core_research_trajectory_refinement(
        (equilibrium.copy(), equilibrium.copy()),
        (0.0, 0.1),
        (equilibrium.copy(), equilibrium.copy(), equilibrium.copy()),
        (0.0, 0.05, 0.1),
        EXACT_PARTITION,
        scales=SCALES,
        agreement_tolerance=1e-8,
        same_dynamics_declared=False,
    )

    assert result.coarse_trajectory.joint_temporal_conditions_pass
    assert result.fine_trajectory.joint_temporal_conditions_pass
    assert not result.same_dynamics_declared
    assert result.failed_conditions == ("same_dynamics_declared",)
    assert not result.joint_refinement_conditions_pass


@pytest.mark.parametrize("declaration", [None, 0, "yes", np.bool_(True)])
def test_refinement_same_dynamics_declaration_requires_a_boolean(declaration):
    equilibrium = _state([1.0, 1.0, 1.0, 1.0])

    with pytest.raises(TypeError, match="same_dynamics_declared must be a boolean"):
        compare_core_research_trajectory_refinement(
            (equilibrium.copy(), equilibrium.copy()),
            (0.0, 0.1),
            (equilibrium.copy(), equilibrium.copy(), equilibrium.copy()),
            (0.0, 0.05, 0.1),
            EXACT_PARTITION,
            scales=SCALES,
            agreement_tolerance=1e-8,
            same_dynamics_declared=declaration,
        )


def test_persistent_id_error_detects_swap_hidden_by_quotient_distance():
    original = _state([2.0, -1.0, 3.0, 0.5])
    reversed_state = _state(
        [original.nodes[node]["EPI"] for node in reversed(tuple(original))],
        frequency=np.asarray(
            [original.nodes[node]["nu_f"] for node in reversed(tuple(original))]
        ),
        phase=np.asarray(
            [original.nodes[node]["theta"] for node in reversed(tuple(original))]
        ),
        weights=BASE_WEIGHTS,
    )

    result = compare_core_research_trajectory_refinement(
        (original.copy(), original.copy()),
        (0.0, 1.0),
        (reversed_state.copy(), reversed_state.copy()),
        (0.0, 1.0),
        EXACT_PARTITION,
        scales=SCALES,
        agreement_tolerance=1e-10,
        same_dynamics_declared=True,
    )

    assert all(
        sample.quotient_structural_distance.distance == pytest.approx(0.0)
        for sample in result.samples
    )
    assert result.maximum_scaled_direct_epi_error > result.agreement_tolerance
    assert not result.common_time_agreement_within_tolerance
    assert not result.joint_refinement_conditions_pass
    assert not hasattr(result, "converged")


def test_declared_time_tolerance_is_reported_for_unique_matches():
    snapshots, _ = _constant_trajectory()
    result = compare_core_research_trajectory_refinement(
        (snapshots[0], snapshots[-1]),
        (0.0, 0.1),
        snapshots,
        (1e-12, 0.05, 0.1 + 1e-12),
        EXACT_PARTITION,
        scales=SCALES,
        agreement_tolerance=1e-8,
        same_dynamics_declared=True,
        time_tolerance=2e-12,
    )

    assert result.time_tolerance == pytest.approx(2e-12)
    assert result.common_time_pairs == ((0.0, 1e-12), (0.1, 0.1 + 1e-12))
    assert all(
        sample.absolute_time_offset <= result.time_tolerance
        for sample in result.samples
    )


def test_time_matching_rejects_ambiguity_and_different_intervals():
    snapshots, _ = _constant_trajectory()
    with pytest.raises(ValueError, match="ambiguous"):
        compare_core_research_trajectory_refinement(
            (snapshots[0], snapshots[-1]),
            (0.0, 1.0),
            snapshots,
            (0.0, 0.1, 1.0),
            EXACT_PARTITION,
            scales=SCALES,
            agreement_tolerance=1e-8,
            same_dynamics_declared=True,
            time_tolerance=0.11,
        )
    with pytest.raises(ValueError, match="same time interval"):
        compare_core_research_trajectory_refinement(
            (snapshots[0], snapshots[-1]),
            (0.0, 1.0),
            (snapshots[0], snapshots[-1]),
            (0.1, 1.0),
            EXACT_PARTITION,
            scales=SCALES,
            agreement_tolerance=1e-8,
            same_dynamics_declared=True,
        )


@pytest.mark.parametrize("time_tolerance", [True, -1.0, math.nan, math.inf])
def test_refinement_time_tolerance_must_be_finite_nonnegative(time_tolerance):
    snapshots, _ = _constant_trajectory()

    with pytest.raises(ValueError, match="finite nonnegative"):
        compare_core_research_trajectory_refinement(
            (snapshots[0], snapshots[-1]),
            (0.0, 1.0),
            (snapshots[0], snapshots[-1]),
            (0.0, 1.0),
            EXACT_PARTITION,
            scales=SCALES,
            agreement_tolerance=1e-8,
            same_dynamics_declared=True,
            time_tolerance=time_tolerance,
        )


@pytest.mark.parametrize(
    "name,bad_value",
    [
        ("agreement_tolerance", True),
        ("agreement_tolerance", math.nan),
        ("agreement_tolerance", math.inf),
        ("agreement_tolerance", 0.0),
        ("trajectory_tolerance", np.bool_(True)),
        ("trajectory_tolerance", math.nan),
        ("trajectory_tolerance", math.inf),
        ("trajectory_tolerance", 1.0),
        ("spectral_tolerance", np.bool_(True)),
        ("spectral_tolerance", math.nan),
        ("spectral_tolerance", math.inf),
        ("spectral_tolerance", 0.0),
        ("spectral_tolerance", 1.0),
    ],
)
def test_refinement_numeric_tolerances_reject_bool_and_nonfinite(name, bad_value):
    equilibrium = _state([1.0, 1.0, 1.0, 1.0])
    kwargs = {
        "agreement_tolerance": 1e-8,
        "same_dynamics_declared": True,
        "trajectory_tolerance": 1e-10,
        "spectral_tolerance": 1e-12,
    }
    kwargs[name] = bad_value

    with pytest.raises(ValueError, match=f"{name}.*open interval"):
        compare_core_research_trajectory_refinement(
            (equilibrium.copy(), equilibrium.copy()),
            (0.0, 1.0),
            (equilibrium.copy(), equilibrium.copy(), equilibrium.copy()),
            (0.0, 0.5, 1.0),
            EXACT_PARTITION,
            scales=SCALES,
            **kwargs,
        )


def test_refinement_comparison_does_not_mutate_graphs():
    snapshots, times = _constant_trajectory()
    before = tuple(_graph_payload(snapshot) for snapshot in snapshots)

    compare_core_research_trajectory_refinement(
        (snapshots[0], snapshots[-1]),
        (times[0], times[-1]),
        snapshots,
        times,
        EXACT_PARTITION,
        scales=SCALES,
        agreement_tolerance=1e-8,
        same_dynamics_declared=True,
    )

    assert tuple(_graph_payload(snapshot) for snapshot in snapshots) == before
