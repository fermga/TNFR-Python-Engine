"""Exact stability checks for frozen heterogeneous EPI diffusion."""

from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.physics import (
    derive_time_varying_diffusion_stability_bound,
    diagnose_euler_relaxation_window,
    verify_heterogeneous_diffusion_stability,
    verify_switching_diffusion_stability,
)


def _set_state(graph, epi, frequency):
    for node, value, nu_f in zip(graph, epi, frequency):
        graph.nodes[node].update(EPI=value, nu_f=nu_f, theta=0.0)
    return graph


def _exact_constant_capacity_step(graph, state, frequency, duration):
    """Independently evolve one constant-capacity interval."""
    adjacency = nx.to_numpy_array(graph, nodelist=list(graph), weight="weight")
    strength = adjacency.sum(axis=1)
    laplacian = np.diag(strength) - adjacency
    mobility = np.asarray(frequency, dtype=float) / strength
    root = np.sqrt(mobility)
    symmetric_generator = root[:, None] * laplacian * root[None, :]
    rates, modes = np.linalg.eigh(symmetric_generator)
    propagator = root[:, None] * (
        modes @ np.diag(np.exp(-rates * duration)) @ modes.T
    ) / root[None, :]
    return propagator @ state


def test_two_node_certificate_matches_closed_form_equality():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=2.0)
    _set_state(graph, [1.0, -1.0], [0.5, 2.0])

    result = verify_heterogeneous_diffusion_stability(graph)

    np.testing.assert_allclose(result.metric_weights, [4.0, 1.0])
    assert result.conserved_total == pytest.approx(3.0)
    assert result.equilibrium_value == pytest.approx(0.6)
    assert result.lyapunov_value == pytest.approx(1.6)
    assert result.generalized_gap == pytest.approx(2.5)
    assert result.exact_quotient_gap_lower_bound == Fraction(5, 2)
    assert result.certified_exponential_rate_lower_bound == 5.0
    assert result.lyapunov_derivative == pytest.approx(-8.0)
    assert result.derivative_upper_bound == pytest.approx(-8.0)
    assert result.balance_residual < 1e-12
    assert result.is_certified
    assert result.exact_weighted_mean_preservation
    assert not result.is_equilibrium


def test_stability_certificate_rejects_a_single_node_domain():
    graph = _set_state(nx.empty_graph(1), [0.0], [1.0])

    with pytest.raises(ValueError, match="at least two nodes"):
        verify_heterogeneous_diffusion_stability(graph)


def test_weighted_path_satisfies_generalized_poincare_bound():
    graph = nx.path_graph(5)
    for edge, weight in zip(graph.edges, [0.5, 3.0, 1.5, 4.0]):
        graph.edges[edge]["weight"] = weight
    _set_state(graph, [2.0, -1.0, 4.0, 0.5, -3.0], [0.2, 0.7, 1.1, 2.0, 3.0])

    result = verify_heterogeneous_diffusion_stability(graph)

    assert result.generalized_gap > 0.0
    assert result.lyapunov_derivative <= result.derivative_upper_bound + 1e-12
    assert result.lyapunov_value > 0.0
    assert result.is_certified


def test_uniform_field_is_the_unique_certified_equilibrium():
    graph = _set_state(nx.cycle_graph(4), [7.0] * 4, [0.3, 0.8, 1.4, 2.2])

    result = verify_heterogeneous_diffusion_stability(graph)

    assert result.equilibrium_value == pytest.approx(7.0)
    assert result.lyapunov_value == pytest.approx(0.0)
    assert result.lyapunov_derivative == pytest.approx(0.0)
    assert result.is_equilibrium
    assert result.is_certified


def test_rounded_laplacian_rows_cannot_fake_a_global_diffusion_theorem():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 0.1), (0, 2, 0.2), (1, 2, 0.3)]
    )
    _set_state(graph, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])

    result = verify_heterogeneous_diffusion_stability(graph)

    assert result.is_consensus
    assert not result.is_equilibrium
    assert not result.exact_consensus_subspace_preservation
    assert not result.exact_uniform_fixed_point_preservation
    assert result.exact_quotient_gap_lower_bound == 0
    assert result.certified_exponential_rate_lower_bound == 0.0
    assert not result.is_certified


def test_common_epi_shift_only_shifts_the_equilibrium():
    graph = _set_state(nx.path_graph(3), [1.0, -2.0, 4.0], [0.5, 1.0, 3.0])
    original = verify_heterogeneous_diffusion_stability(graph)
    for node in graph:
        graph.nodes[node]["EPI"] += 11.0

    shifted = verify_heterogeneous_diffusion_stability(graph)

    assert shifted.equilibrium_value == pytest.approx(original.equilibrium_value + 11.0)
    assert shifted.lyapunov_value == pytest.approx(original.lyapunov_value)
    assert shifted.lyapunov_derivative == pytest.approx(original.lyapunov_derivative)
    assert shifted.generalized_gap == pytest.approx(original.generalized_gap)


@pytest.mark.parametrize("seed", range(5))
def test_exact_flow_obeys_exponential_envelope_on_seeded_weighted_graphs(seed):
    rng = np.random.default_rng(seed)
    graph = nx.path_graph(8)
    graph.add_edges_from([(0, 3), (2, 6), (4, 7)])
    for edge in graph.edges:
        graph.edges[edge]["weight"] = rng.uniform(0.1, 4.0)
    epi = rng.normal(size=len(graph))
    frequency = np.exp(rng.uniform(-2.0, 2.0, size=len(graph)))
    _set_state(graph, epi, frequency)

    result = verify_heterogeneous_diffusion_stability(graph)
    adjacency = nx.to_numpy_array(graph, nodelist=result.nodes, weight="weight")
    strength = adjacency.sum(axis=1)
    laplacian = np.diag(strength) - adjacency
    inverse_root = 1.0 / np.sqrt(result.metric_weights)
    generator = inverse_root[:, None] * laplacian * inverse_root[None, :]
    rates, modes = np.linalg.eigh(generator)
    centered = epi - result.equilibrium_value
    initial = np.sqrt(result.metric_weights) * centered

    for time in [0.1, 1.0, 5.0]:
        evolved = modes @ (np.exp(-rates * time) * (modes.T @ initial))
        value = 0.5 * float(evolved @ evolved)
        envelope = np.exp(-result.exponential_rate * time) * result.lyapunov_value
        assert value <= envelope + 1e-11


@pytest.mark.parametrize(
    "graph, message",
    [
        (_set_state(nx.empty_graph(2), [0.0, 1.0], [1.0, 1.0]), "row strength"),
        (_set_state(nx.path_graph(3), [0.0, 1.0, 2.0], [1.0, 0.0, 1.0]), "positive structural frequency"),
    ],
)
def test_degenerate_transport_is_outside_the_theorem(graph, message):
    with pytest.raises(ValueError, match=message):
        verify_heterogeneous_diffusion_stability(graph)


def test_zero_weight_bridge_does_not_fake_connectivity():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=0.0)
    graph.add_edge(2, 3, weight=1.0)
    _set_state(graph, [0.0, 1.0, 2.0, 3.0], [1.0] * 4)

    with pytest.raises(ValueError, match="connected positive conductance"):
        verify_heterogeneous_diffusion_stability(graph)


def test_asymmetric_transport_is_outside_the_theorem():
    graph = nx.DiGraph([(0, 1), (1, 0), (1, 2)])
    _set_state(graph, [0.0, 1.0, 2.0], [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="symmetric adjacency"):
        verify_heterogeneous_diffusion_stability(graph)


def test_time_varying_bound_matches_closed_two_node_rate():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=2.0)
    _set_state(graph, [1.0, -1.0], [1.0, 1.0])

    result = derive_time_varying_diffusion_stability_bound(
        graph, frequency_lower_bounds=[0.5, 2.0], frequency_upper_bounds=[1.0, 3.0]
    )

    assert result.combinatorial_gap == pytest.approx(4.0)
    assert result.minimum_mobility == pytest.approx(0.25)
    assert result.maximum_mobility == pytest.approx(1.5)
    assert result.dirichlet_energy == pytest.approx(4.0)
    assert result.spectral_energy_decay_rate_estimate == pytest.approx(2.0)
    assert result.spectral_consensus_distance_estimate == pytest.approx(
        np.sqrt(2.0)
    )
    assert result.exact_combinatorial_gap_lower_bound == Fraction(4)
    assert result.certified_combinatorial_gap_lower_bound == 4.0
    assert result.exact_minimum_mobility_lower_bound == Fraction(1, 4)
    assert result.certified_minimum_mobility_lower_bound == 0.25
    assert result.exact_maximum_mobility_upper_bound == Fraction(3, 2)
    assert result.certified_maximum_mobility_upper_bound == 1.5
    assert result.exact_energy_decay_rate_lower_bound == Fraction(2)
    assert result.exact_dirichlet_energy == Fraction(4)
    assert result.dirichlet_energy_lower_bound == 4.0
    assert result.dirichlet_energy_upper_bound == 4.0
    assert result.energy_decay_rate == pytest.approx(2.0)
    assert result.energy_derivative_upper_bound == pytest.approx(-8.0)
    assert result.consensus_distance_bound == pytest.approx(np.sqrt(2.0))
    assert Fraction.from_float(result.energy_decay_rate) <= (
        result.exact_energy_decay_rate_lower_bound
    )
    assert Fraction.from_float(result.energy_derivative_upper_bound) >= (
        -Fraction.from_float(result.energy_decay_rate)
        * result.exact_dirichlet_energy
    )
    assert Fraction.from_float(result.consensus_distance_bound) ** 2 >= (
        2
        * result.exact_dirichlet_energy
        / result.exact_combinatorial_gap_lower_bound
    )
    assert isinstance(result.nodes, tuple)
    assert result.exact_real_uniform_fixed_point_preservation
    assert result.materialized_binary64_uniform_fixed_point_preservation
    assert result.exact_real_continuous_time_model_certified
    assert result.operational_binary64_rate_available
    assert not result.runtime_integration_certified
    assert result.is_certified


def test_time_varying_exact_real_theorem_is_separate_from_binary64_laplacian():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [(0, 1, 0.1), (0, 2, 0.2), (1, 2, 0.3)]
    )
    _set_state(graph, [2.0, -1.0, 4.0], [1.0, 1.0, 1.0])

    result = derive_time_varying_diffusion_stability_bound(
        graph, frequency_lower_bounds=1.0, frequency_upper_bounds=2.0
    )

    assert result.exact_real_uniform_fixed_point_preservation
    assert not result.materialized_binary64_uniform_fixed_point_preservation
    assert result.exact_combinatorial_gap_lower_bound > 0
    assert result.exact_energy_decay_rate_lower_bound > 0
    assert result.energy_decay_rate > 0.0
    assert result.exact_real_continuous_time_model_certified
    assert result.operational_binary64_rate_available
    assert result.is_certified
    assert not result.runtime_integration_certified


def test_time_varying_rate_underflow_causes_safe_operational_abstention():
    graph = _set_state(
        nx.path_graph(7), np.arange(7.0), np.ones(7)
    )
    smallest_positive = np.nextafter(0.0, 1.0)

    result = derive_time_varying_diffusion_stability_bound(
        graph,
        frequency_lower_bounds=smallest_positive,
        frequency_upper_bounds=1.0,
    )

    assert result.exact_real_uniform_fixed_point_preservation
    assert result.exact_energy_decay_rate_lower_bound > 0
    assert result.energy_decay_rate == 0.0
    assert result.spectral_energy_decay_rate_estimate == 0.0
    assert result.exact_real_continuous_time_model_certified
    assert not result.operational_binary64_rate_available
    assert not result.is_certified


def test_spectral_overflow_does_not_suppress_exact_real_certificate():
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1e308)
    _set_state(graph, [1.0, -1.0], [1.0, 1.0])

    result = derive_time_varying_diffusion_stability_bound(
        graph, frequency_lower_bounds=1.0, frequency_upper_bounds=2.0
    )

    assert not np.isfinite(result.dirichlet_energy)
    assert result.exact_dirichlet_energy > Fraction.from_float(
        np.finfo(float).max
    )
    assert result.dirichlet_energy_upper_bound == float("inf")
    assert result.exact_energy_decay_rate_lower_bound == Fraction(4)
    assert result.energy_decay_rate == 4.0
    assert result.exact_real_continuous_time_model_certified
    assert result.operational_binary64_rate_available
    assert result.is_certified


def test_time_varying_payload_arrays_are_detached_and_read_only():
    graph = _set_state(nx.path_graph(3), [1.0, 0.0, -1.0], [1.0] * 3)
    lower = np.array([0.5, 0.75, 1.0])
    upper = np.array([1.5, 1.75, 2.0])

    result = derive_time_varying_diffusion_stability_bound(
        graph, lower, upper
    )
    lower[:] = 99.0
    upper[:] = 101.0

    np.testing.assert_allclose(
        result.frequency_lower_bounds, [0.5, 0.75, 1.0]
    )
    np.testing.assert_allclose(
        result.frequency_upper_bounds, [1.5, 1.75, 2.0]
    )
    for payload in (
        result.frequency_lower_bounds,
        result.frequency_upper_bounds,
        result.mobility_lower_bounds,
        result.mobility_upper_bounds,
        result.certified_mobility_lower_bounds,
        result.certified_mobility_upper_bounds,
    ):
        with pytest.raises(ValueError, match="read-only"):
            payload.flat[0] = payload.flat[0]


def test_piecewise_capacity_schedule_obeys_common_energy_envelope():
    graph = nx.path_graph(4)
    for edge, weight in zip(graph.edges, [0.5, 2.0, 3.0]):
        graph.edges[edge]["weight"] = weight
    initial = np.array([2.0, -1.0, 3.5, 0.0])
    _set_state(graph, initial, [1.0] * 4)
    lower = np.array([0.2, 0.3, 0.5, 0.7])
    upper = np.array([1.5, 2.0, 1.8, 2.4])
    bound = derive_time_varying_diffusion_stability_bound(graph, lower, upper)
    adjacency = nx.to_numpy_array(graph, weight="weight")
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    state = initial.copy()
    elapsed = 0.0
    schedules = [lower, upper, (lower + upper) / 2.0, upper[::-1]]

    for frequency in schedules:
        duration = 0.35
        state = _exact_constant_capacity_step(graph, state, frequency, duration)
        elapsed += duration
        energy = 0.5 * float(state @ laplacian @ state)
        envelope = (
            np.exp(-bound.energy_decay_rate * elapsed)
            * bound.dirichlet_energy_upper_bound
        )
        assert energy <= envelope + 1e-11


def test_capacity_order_changes_consensus_value_when_ratios_change():
    graph = nx.path_graph(2)
    initial = np.array([1.0, 0.0])
    first = np.array([4.0, 0.5])
    second = first[::-1]

    forward = _exact_constant_capacity_step(graph, initial, first, 0.6)
    forward = _exact_constant_capacity_step(graph, forward, second, 0.6)
    reverse = _exact_constant_capacity_step(graph, initial, second, 0.6)
    reverse = _exact_constant_capacity_step(graph, reverse, first, 0.6)

    # A final common-capacity phase conserves the arithmetic mean.  The two
    # orders therefore converge to different uniform values, disproving a
    # schedule-independent weighted mean for changing capacity ratios.
    assert np.mean(forward) != pytest.approx(np.mean(reverse), abs=1e-4)


@pytest.mark.parametrize(
    "lower, upper, message",
    [
        ([0.0, 1.0], [1.0, 1.0], "finite and positive"),
        ([2.0, 1.0], [1.0, 1.0], "cannot exceed"),
        ([1.0], [1.0], "one value per"),
    ],
)
def test_time_varying_bound_rejects_invalid_assumptions(lower, upper, message):
    graph = _set_state(nx.path_graph(2), [1.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError, match=message):
        derive_time_varying_diffusion_stability_bound(graph, lower, upper)


def test_time_varying_bound_rejects_positive_frequency_input_underflow():
    graph = _set_state(nx.path_graph(2), [1.0, 0.0], [1.0, 1.0])

    with pytest.raises(ValueError, match="finite and positive"):
        derive_time_varying_diffusion_stability_bound(
            graph,
            frequency_lower_bounds=Fraction(1, 10**1000),
            frequency_upper_bounds=1.0,
        )


def test_path_modal_window_exposes_three_position_policy_gap():
    graph = _set_state(nx.path_graph(21), np.arange(21.0), np.ones(21))

    result = diagnose_euler_relaxation_window(graph, dt=0.5)

    assert result.policy_window == 3
    assert result.modal_steps == 231
    assert result.is_euler_stable
    assert result.maximum_modal_factor**230 >= result.target_fraction
    assert result.maximum_modal_factor**231 < result.target_fraction


def test_complete_graph_relaxes_inside_policy_numeric_window():
    graph = _set_state(nx.complete_graph(4), [1.0, -1.0, 0.5, 2.0], [1.0] * 4)

    result = diagnose_euler_relaxation_window(graph, dt=0.5)

    assert result.slowest_decay_rate == pytest.approx(4.0 / 3.0)
    assert result.fastest_decay_rate == pytest.approx(4.0 / 3.0)
    assert result.maximum_modal_factor == pytest.approx(1.0 / 3.0)
    assert result.modal_steps == 2
    assert result.policy_window == 3


def test_euler_diagnostic_detects_unstable_timestep():
    graph = _set_state(nx.path_graph(5), np.arange(5.0), np.ones(5))

    result = diagnose_euler_relaxation_window(graph, dt=1.1)

    assert result.euler_stability_limit == pytest.approx(1.0)
    assert not result.is_euler_stable
    assert result.modal_steps is None
    assert result.maximum_modal_factor > 1.0
    assert result.policy_window == 3


def test_euler_window_uses_actual_heterogeneous_spectrum():
    graph = _set_state(nx.path_graph(4), [0.0, 1.0, -1.0, 2.0], [0.2, 0.7, 1.3, 2.1])

    result = diagnose_euler_relaxation_window(graph, dt=0.2, target_fraction=0.1)

    assert len(result.decay_rates) == len(graph) - 1
    assert result.slowest_decay_rate > 0.0
    assert result.is_euler_stable
    assert result.modal_steps is not None
    assert result.maximum_modal_factor**result.modal_steps < 0.1
    assert result.policy_window == 3
    assert result.spectral_relative_tolerance == pytest.approx(1e-12)
    assert result.spectral_zero_threshold == pytest.approx(
        result.spectral_relative_tolerance * result.fastest_decay_rate
    )


def test_euler_spectral_cutoff_is_relative_to_global_frequency_scale():
    base_frequency = np.array([0.2, 0.7, 1.3, 2.1])
    graph = _set_state(
        nx.path_graph(4),
        [0.0, 1.0, -1.0, 2.0],
        1e-12 * base_frequency,
    )

    result = diagnose_euler_relaxation_window(
        graph, dt=1.0, tolerance=1e-12
    )

    assert len(result.decay_rates) == len(graph) - 1
    assert result.slowest_decay_rate > 0.0
    assert result.fastest_decay_rate < 1e-11
    assert result.spectral_zero_threshold == pytest.approx(
        1e-12 * result.fastest_decay_rate
    )
    assert result.is_euler_stable


@pytest.mark.parametrize(
    "dt, target, message",
    [
        (0.0, None, "dt"),
        (0.1, 1.0, "target_fraction"),
        (float("inf"), 0.5, "dt"),
    ],
)
def test_euler_window_rejects_invalid_parameters(dt, target, message):
    graph = _set_state(nx.path_graph(3), [0.0, 1.0, 2.0], [1.0] * 3)
    with pytest.raises(ValueError, match=message):
        diagnose_euler_relaxation_window(graph, dt=dt, target_fraction=target)


@pytest.mark.parametrize(
    "boolean", [True, False, np.bool_(True), np.bool_(False)]
)
@pytest.mark.parametrize("parameter", ["dt", "target_fraction", "tolerance"])
def test_euler_window_rejects_boolean_numeric_controls(boolean, parameter):
    graph = _set_state(nx.path_graph(3), [0.0, 1.0, 2.0], [1.0] * 3)
    controls = {"dt": 0.1, "target_fraction": 0.5, "tolerance": 1e-12}
    controls[parameter] = boolean

    with pytest.raises(ValueError, match=rf"{parameter}.*not boolean"):
        diagnose_euler_relaxation_window(graph, **controls)


@pytest.mark.parametrize(
    "boolean", [True, False, np.bool_(True), np.bool_(False)]
)
def test_diffusion_stability_certificates_reject_boolean_tolerance(boolean):
    graph = _set_state(nx.path_graph(3), [0.0, 1.0, 2.0], [1.0] * 3)
    other = _set_state(nx.cycle_graph(3), [0.0, 1.0, 2.0], [2.0] * 3)

    with pytest.raises(ValueError, match="tolerance.*not boolean"):
        verify_heterogeneous_diffusion_stability(graph, tolerance=boolean)
    with pytest.raises(ValueError, match="tolerance.*not boolean"):
        verify_switching_diffusion_stability(
            [graph, other], tolerance=boolean
        )


@pytest.mark.parametrize(
    "boolean", [True, False, np.bool_(True), np.bool_(False)]
)
@pytest.mark.parametrize("parameter", ["lower", "upper"])
def test_time_varying_frequency_limits_reject_boolean_scalars(
    boolean, parameter
):
    graph = _set_state(nx.path_graph(3), [0.0, 1.0, 2.0], [1.0] * 3)
    bounds = {"lower": 0.5, "upper": 2.0}
    bounds[parameter] = boolean

    with pytest.raises(ValueError, match=rf"frequency_{parameter}_bounds.*not boolean"):
        derive_time_varying_diffusion_stability_bound(
            graph, bounds["lower"], bounds["upper"]
        )


@pytest.mark.parametrize(
    "bounds",
    [
        ([0.5, True, 0.5], [2.0] * 3),
        ({0: 0.5, 1: np.bool_(True), 2: 0.5}, {0: 2.0, 1: 2.0, 2: 2.0}),
    ],
)
def test_time_varying_frequency_limits_reject_boolean_collections(bounds):
    graph = _set_state(nx.path_graph(3), [0.0, 1.0, 2.0], [1.0] * 3)

    with pytest.raises(ValueError, match="frequency_lower_bounds.*not booleans"):
        derive_time_varying_diffusion_stability_bound(graph, *bounds)


def _switching_regime(graph, epi, common_metric, *, metric_scale=1.0):
    adjacency = nx.to_numpy_array(graph, nodelist=list(graph), weight="weight")
    strength = adjacency.sum(axis=1)
    frequency = strength / (metric_scale * np.asarray(common_metric, dtype=float))
    return _set_state(graph, epi, frequency)


def test_switching_certificate_rejects_single_node_regimes():
    first = _set_state(nx.empty_graph(1), [0.0], [1.0])
    second = _set_state(nx.empty_graph(1), [0.0], [2.0])

    with pytest.raises(ValueError, match="at least two nodes"):
        verify_switching_diffusion_stability([first, second])


def test_consensus_invariance_without_uniform_fixed_points_cannot_certify_switching():
    rounded = nx.Graph()
    rounded.add_edge(0, 0, weight=1e308)
    rounded.add_edge(1, 1, weight=1e308)
    rounded.add_edge(0, 1, weight=5e-324)
    _set_state(rounded, [0.0, 1.0], [1e308, 1e308])
    ordinary = _set_state(nx.path_graph(2), [0.0, 1.0], [1.0, 1.0])

    result = verify_switching_diffusion_stability([rounded, ordinary])

    assert result.shares_exact_common_metric
    assert result.exact_consensus_subspace_preservation_by_regime == (True, True)
    assert result.exact_common_consensus_subspace_preservation
    assert result.exact_uniform_fixed_point_preservation_by_regime == (False, True)
    assert not result.exact_common_uniform_fixed_point_preservation
    assert result.exact_weighted_mean_preservation_by_regime == (False, True)
    assert np.all(result.certified_exponential_rate_lower_bounds > 0.0)
    assert not result.supports_exact_switching_theorem


def test_common_metric_certifies_arbitrary_fixed_node_topology_switching():
    epi = np.array([2.0, -1.0, 4.0, 0.5])
    metric = np.array([1.0, 2.0, 4.0, 8.0])
    path = _switching_regime(nx.path_graph(4), epi, metric)
    cycle = _switching_regime(nx.cycle_graph(4), epi, metric, metric_scale=2.0)

    result = verify_switching_diffusion_stability([path, cycle])

    np.testing.assert_allclose(result.normalized_metric_weights, metric / metric.sum())
    assert result.regime_count == 2
    assert result.shares_exact_common_metric
    assert result.common_metric_within_tolerance
    assert result.common_metric_residual == 0.0
    assert result.uniform_exponential_rate > 0.0
    assert result.supports_exact_switching_theorem
    assert result.numerical_hypotheses_pass
    assert len(result.exact_quotient_gap_lower_bounds) == 2
    assert np.all(result.certified_exponential_rate_lower_bounds > 0.0)
    assert result.certified_uniform_exponential_rate_lower_bound > 0.0
    assert result.certified_uniform_exponential_rate_lower_bound <= (
        result.uniform_exponential_rate
    )
    assert isinstance(result.nodes, tuple)
    assert result.exact_consensus_subspace_preservation_by_regime == (True, True)
    assert result.exact_common_consensus_subspace_preservation
    assert result.exact_uniform_fixed_point_preservation_by_regime == (True, True)
    assert result.exact_common_uniform_fixed_point_preservation
    assert result.exact_weighted_mean_preservation_by_regime == (True, True)
    assert result.exact_common_weighted_mean_preservation
    for payload in (
        result.reference_metric_weights,
        result.normalized_metric_weights,
        result.metric_mismatches,
        result.generalized_gaps,
        result.certified_exponential_rate_lower_bounds,
    ):
        with pytest.raises(ValueError, match="read-only"):
            payload.flat[0] = payload.flat[0]


def test_exact_switched_flow_obeys_the_common_exponential_envelope():
    epi = np.array([3.0, -2.0, 1.0, 4.0])
    metric = np.array([1.0, 1.5, 2.0, 2.5])
    regimes = [
        _switching_regime(nx.path_graph(4), epi, metric),
        _switching_regime(nx.cycle_graph(4), epi, metric, metric_scale=0.7),
        _switching_regime(nx.complete_graph(4), epi, metric, metric_scale=1.8),
    ]
    certificate = verify_switching_diffusion_stability(regimes)
    state = epi.copy()
    elapsed = 0.0

    for graph, duration in zip([regimes[1], regimes[0], regimes[2]], [0.2, 0.4, 0.3]):
        frequency = np.array([graph.nodes[node]["nu_f"] for node in graph])
        state = _exact_constant_capacity_step(graph, state, frequency, duration)
        elapsed += duration
        centered = state - certificate.equilibrium_value
        value = 0.5 * np.sum(certificate.normalized_metric_weights * centered**2)
        envelope = (
            np.exp(-certificate.uniform_exponential_rate * elapsed)
            * certificate.lyapunov_value
        )
        assert value <= envelope + 1e-11


def test_topology_family_without_common_metric_is_not_promoted_to_theorem():
    epi = [1.0, 0.0, -1.0, 2.0]
    path = _set_state(nx.path_graph(4), epi, [1.0] * 4)
    cycle = _set_state(nx.cycle_graph(4), epi, [1.0] * 4)

    result = verify_switching_diffusion_stability([path, cycle])

    assert not result.shares_exact_common_metric
    assert not result.supports_exact_switching_theorem
    assert np.max(result.metric_mismatches) > 0.0


def test_large_tolerance_does_not_promote_approximate_metric_to_exact_theorem():
    epi = np.array([2.0, -1.0, 4.0, 0.5])
    first_metric = np.array([1.0, 2.0, 3.0, 4.0])
    second_metric = np.array([1.0, 2.01, 3.0, 4.0])
    first = _switching_regime(nx.path_graph(4), epi, first_metric)
    second = _switching_regime(nx.cycle_graph(4), epi, second_metric)

    result = verify_switching_diffusion_stability(
        [first, second], tolerance=0.5
    )

    assert result.common_metric_residual > 0.0
    assert result.common_metric_within_tolerance
    assert result.numerical_hypotheses_pass
    assert not result.shares_exact_common_metric
    assert not result.supports_exact_switching_theorem


def test_metric_normalization_underflow_cannot_create_false_exact_theorem():
    first = _set_state(
        nx.path_graph(2), [0.0, 1.0], [1e-308, 1e308]
    )
    second = _set_state(
        nx.path_graph(2), [0.0, 1.0], [1e-308, 5e307]
    )

    # The raw metric vectors are [1e308, 1e-308] and
    # [1e308, 2e-308], so they are not proportional.  Direct normalization
    # would round both small coordinates to zero and falsely identify them.
    with pytest.raises(ValueError, match="normalization.*dynamic range"):
        verify_switching_diffusion_stability([first, second])


def test_rounded_normalization_cannot_create_false_exact_theorem():
    upper = np.nextafter(1.0, np.inf)
    lower = np.nextafter(1.0, -np.inf)
    first = _set_state(nx.path_graph(2), [0.0, 1.0], [1.0, upper])
    second = _set_state(nx.path_graph(2), [0.0, 1.0], [lower, 1.0])

    result = verify_switching_diffusion_stability([first, second])

    # Both normalized vectors round to [0.5, 0.5], but d/nu is not exactly
    # proportional in rational arithmetic on the supplied binary64 inputs.
    assert result.common_metric_residual == 0.0
    assert result.common_metric_within_tolerance
    assert not result.shares_exact_common_metric
    assert not result.supports_exact_switching_theorem


def test_switching_certificate_rejects_node_set_changes():
    first = _set_state(nx.path_graph(3), [0.0, 1.0, 2.0], [1.0] * 3)
    second = _set_state(nx.path_graph(4), [0.0, 1.0, 2.0, 3.0], [1.0] * 4)
    with pytest.raises(ValueError, match="one node set"):
        verify_switching_diffusion_stability([first, second])
