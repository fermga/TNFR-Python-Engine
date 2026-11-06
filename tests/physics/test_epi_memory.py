"""Derived hidden-state memory in a fixed reversible pure-EPI observation.

The P5 reference uses its independently reduced scalar equation
``d'' + 3 d' + 2 d = 0``. No matrix-exponential helper is used to compute
the expected trajectories or memory convolution in these tests.
"""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_EPI
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.integrators import update_epi_via_nodal_equation
from tnfr.physics.epi_memory import observe_epi_memory

P5_BLOCKS = ((0, 4), (1, 2, 3))
SAMPLE_TIMES = (0.0, 0.125, 0.5, 1.0, 3.0)


def _graph(size, capacity=None):
    graph = nx.path_graph(size)
    if capacity is None:
        capacity = [1.0] * size
    for node, frequency in zip(graph, capacity, strict=True):
        # Initial EPI is an explicit observation input, not this live value.
        graph.nodes[node].update(EPI=99.0, nu_f=frequency, theta=0.0)
    return graph


def _p5_reference(initial, time):
    """Solve the two visible coordinates and one coupled hidden coordinate."""
    a0 = (initial[0] + initial[4]) / 2.0
    b0 = (initial[1] + initial[2] + initial[3]) / 3.0
    u0 = initial[2] - (initial[1] + initial[3]) / 2.0
    d0 = a0 - b0
    derivative0 = -4.0 * d0 / 3.0 - 4.0 * u0 / 9.0
    coefficient1 = 2.0 * d0 + derivative0
    coefficient2 = -d0 - derivative0
    decay1, decay2 = math.exp(-time), math.exp(-2.0 * time)
    hidden_decay = math.exp(-5.0 * time / 3.0)
    difference = coefficient1 * decay1 + coefficient2 * decay2
    derivative = -coefficient1 * decay1 - 2.0 * coefficient2 * decay2
    conserved = (a0 + 3.0 * b0) / 4.0
    macro = np.array([conserved + 0.75 * difference, conserved - 0.25 * difference])
    rate = np.array([0.75 * derivative, -0.25 * derivative])
    # Integral of exp(-5(t-s)/3) * (c1 exp(-s) + c2 exp(-2s)).
    integral = 1.5 * coefficient1 * (decay1 - hidden_decay) - 3.0 * coefficient2 * (
        decay2 - hidden_decay
    )
    convolution = integral * np.array([1.0 / 6.0, -1.0 / 18.0])
    source = u0 * hidden_decay * np.array([-1.0 / 3.0, 1.0 / 9.0])
    return macro, rate, convolution, source


def test_equitable_p4_has_no_memory_even_with_nonzero_hidden_initial_epi():
    initial = np.array([2.0, -1.0, 3.0, 0.5])
    result = observe_epi_memory(
        _graph(4), ((0, 3), (1, 2)), initial, times=SAMPLE_TIMES
    )

    assert result.closure_within_tolerance
    assert np.linalg.norm(result.initial_hidden) > 1.0
    np.testing.assert_allclose(
        result.instantaneous_generator, [[1.0, -1.0], [-0.5, 0.5]]
    )
    a0, b0 = 1.25, 1.0
    conserved = (a0 + 2.0 * b0) / 3.0
    for sample in result.samples:
        difference = (a0 - b0) * math.exp(-1.5 * sample.time)
        expected = [conserved + 2.0 * difference / 3.0, conserved - difference / 3.0]
        np.testing.assert_allclose(sample.projected_epi, expected, atol=2e-12)
        np.testing.assert_allclose(sample.markov_epi, expected, atol=2e-12)
        np.testing.assert_allclose(sample.source_free_epi, expected, atol=2e-12)
        np.testing.assert_allclose(sample.kernel, 0.0, atol=2e-13)
        np.testing.assert_allclose(sample.initial_source, 0.0, atol=2e-13)
        np.testing.assert_allclose(sample.convolution, 0.0, atol=2e-13)
        assert sample.identity_residual < 2e-12
        assert sample.fine_semigroup_residual < result.numerical_tolerance
        assert sample.markov_semigroup_residual < result.numerical_tolerance


@pytest.mark.parametrize(
    "initial",
    [
        [0.0, 0.0, 3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [2.0, -1.0, 0.25, 3.0, -0.5],
    ],
)
def test_p5_matches_analytic_memory_source_and_two_decay_modes(initial):
    result = observe_epi_memory(_graph(5), P5_BLOCKS, initial, times=SAMPLE_TIMES)

    assert not result.closure_within_tolerance
    np.testing.assert_allclose(result.macro_metric_weights, [2.0, 6.0])
    np.testing.assert_allclose(
        result.instantaneous_generator, [[1.0, -1.0], [-1.0 / 3.0, 1.0 / 3.0]]
    )
    for sample in result.samples:
        macro, rate, convolution, source = _p5_reference(initial, sample.time)
        expected_kernel = math.exp(-5.0 * sample.time / 3.0) * np.array(
            [[1.0 / 6.0, -1.0 / 6.0], [-1.0 / 18.0, 1.0 / 18.0]]
        )
        np.testing.assert_allclose(sample.kernel, expected_kernel, atol=2e-12)
        np.testing.assert_allclose(sample.initial_source, source, atol=2e-12)
        np.testing.assert_allclose(sample.convolution, convolution, atol=2e-12)
        np.testing.assert_allclose(sample.projected_epi, macro, atol=2e-12)
        np.testing.assert_allclose(sample.projected_rate, rate, atol=2e-12)
        np.testing.assert_allclose(sample.reconstructed_rate, rate, atol=2e-12)
        assert sample.identity_residual < 2e-12
        assert sample.relative_identity_residual < 2e-12
        assert sample.fine_semigroup_residual < result.numerical_tolerance
        assert sample.markov_semigroup_residual < result.numerical_tolerance


def test_identical_p5_macro_initial_states_have_different_nodal_derivatives():
    initial_a = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    initial_b = np.array([0.0, 0.0, 3.0, 0.0, 0.0])
    observed = [
        observe_epi_memory(_graph(5), P5_BLOCKS, state, times=[0.0, 1.0])
        for state in (initial_a, initial_b)
    ]
    np.testing.assert_allclose(observed[0].initial_macro, [0.0, 1.0])
    np.testing.assert_allclose(observed[1].initial_macro, [0.0, 1.0])
    np.testing.assert_allclose(observed[0].samples[0].projected_rate, [1.0, -1.0 / 3.0])
    np.testing.assert_allclose(
        observed[1].samples[0].projected_rate, [0.0, 0.0], atol=1e-14
    )
    assert (
        np.linalg.norm(
            observed[0].samples[1].projected_epi - observed[1].samples[1].projected_epi
        )
        > 0.1
    )

    # The production pressure reader and shared Euler step realize these
    # initial derivatives; this does not assert an exact finite-time flow.
    for state, observation in zip((initial_a, initial_b), observed, strict=True):
        graph = _graph(5)
        graph.graph.update(
            DNFR_WEIGHTS={"phase": 0.0, "epi": 1.0, "vf": 0.0, "topo": 0.0},
            GAMMA={"type": "none"},
            use_extended_dynamics=False,
            DT_MIN=0.0,
            EPI_MIN=-4.0,
            EPI_MAX=4.0,
            CLIP_MODE="hard",
        )
        for node, value in zip(graph, state, strict=True):
            graph.nodes[node]["EPI"] = value
        default_compute_delta_nfr(graph)
        step = 0.125
        update_epi_via_nodal_equation(graph, dt=step, t=0.0, method="euler")
        after = np.array(
            [get_attr(graph.nodes[node], ALIAS_EPI, None) for node in graph]
        )
        np.testing.assert_allclose(
            observation.projection @ ((after - state) / step),
            observation.samples[0].projected_rate,
            atol=2e-13,
        )


def test_omitting_hidden_initial_source_loses_the_observed_trajectory():
    initial = [0.0, 0.0, 3.0, 0.0, 0.0]
    result = observe_epi_memory(_graph(5), P5_BLOCKS, initial, times=[0.0, 1.0])
    lifted_initial = [0.0, 1.0, 1.0, 1.0, 0.0]
    expected_source_free, _, _, _ = _p5_reference(lifted_initial, 1.0)

    np.testing.assert_allclose(
        result.samples[1].source_free_epi, expected_source_free, atol=2e-12
    )
    assert np.linalg.norm(result.samples[0].initial_source) > 1.0
    assert np.linalg.norm(result.samples[1].projected_epi - expected_source_free) > 0.1


def test_zero_hidden_initial_state_does_not_remove_subsequently_generated_memory():
    result = observe_epi_memory(
        _graph(5), P5_BLOCKS, [1.0, 0.0, 0.0, 0.0, 1.0], times=[0.0, 1.0]
    )
    np.testing.assert_array_equal(result.initial_hidden, np.zeros(5))
    for sample in result.samples:
        np.testing.assert_allclose(sample.initial_source, 0.0, atol=2e-14)
        np.testing.assert_allclose(
            sample.source_free_epi, sample.projected_epi, atol=2e-12
        )
    assert np.linalg.norm(result.samples[1].convolution) > 0.01
    assert (
        np.linalg.norm(result.samples[1].projected_epi - result.samples[1].markov_epi)
        > 0.01
    )


def test_p3_half_projection_has_derived_capacity_dependent_memory_rate():
    result = observe_epi_memory(
        _graph(3, [1.0, 2.0, 1.0]),
        ((0, 1), (2,)),
        [1.0, -1.0, 0.0],
        times=SAMPLE_TIMES,
    )
    assert result.right_inverse_residual == 0.0
    assert result.projector_residual == 0.0
    np.testing.assert_array_equal(result.initial_macro, [0.0, 0.0])
    for sample in result.samples:
        decay = math.exp(-2.5 * sample.time)
        np.testing.assert_allclose(
            sample.kernel, decay * np.array([[0.25, -0.25], [-0.5, 0.5]]), atol=2e-12
        )
        np.testing.assert_allclose(
            sample.initial_source, decay * np.array([0.5, -1.0]), atol=2e-12
        )
        np.testing.assert_allclose(sample.markov_epi, 0.0, atol=2e-13)
        np.testing.assert_allclose(sample.source_free_epi, 0.0, atol=2e-13)


def test_uniform_capacity_rescaling_determines_the_memory_clock():
    initial = [0.0, 0.0, 3.0, 0.0, 0.0]
    factor = 2.0
    base = observe_epi_memory(_graph(5), P5_BLOCKS, initial, times=[factor * 0.5])
    scaled = observe_epi_memory(
        _graph(5, [factor] * 5), P5_BLOCKS, initial, times=[0.5]
    )
    reference, sample = base.samples[0], scaled.samples[0]

    np.testing.assert_allclose(scaled.projection, base.projection)
    np.testing.assert_allclose(
        sample.projected_epi, reference.projected_epi, atol=2e-12
    )
    np.testing.assert_allclose(sample.kernel, factor**2 * reference.kernel, atol=2e-12)
    np.testing.assert_allclose(
        sample.initial_source, factor * reference.initial_source, atol=2e-12
    )
    np.testing.assert_allclose(
        sample.convolution, factor * reference.convolution, atol=2e-12
    )


def test_loose_closure_tolerance_does_not_erase_resolved_memory_terms():
    result = observe_epi_memory(
        _graph(5),
        P5_BLOCKS,
        [0.0, 0.0, 3.0, 0.0, 0.0],
        times=[0.0],
        tolerance=10.0,
    )
    assert result.closure_within_tolerance
    assert result.tolerance == 10.0
    assert result.samples[0].kernel[0, 0] == pytest.approx(1.0 / 6.0)
    np.testing.assert_allclose(result.samples[0].initial_source, [-1.0, 1.0 / 3.0])


def test_weighted_memory_kernel_is_metric_symmetric_positive_semidefinite():
    graph = _graph(5, [0.5, 2.0, 1.5, 4.0, 1.0])
    for edge, weight in zip(graph.edges, [1.0, 3.0, 2.0, 0.5], strict=True):
        graph.edges[edge]["weight"] = weight
    result = observe_epi_memory(graph, P5_BLOCKS, [0.0] * 5, times=[0.0, 0.5, 2.0])
    metric = np.diag([2.5, 2.0 + 5.0 / 1.5 + 2.5 / 4.0])

    assert not result.closure_within_tolerance
    for sample in result.samples:
        weighted_kernel = metric @ sample.kernel
        np.testing.assert_allclose(weighted_kernel, weighted_kernel.T, atol=2e-12)
        assert np.linalg.eigvalsh(weighted_kernel).min() >= -2e-12
        np.testing.assert_allclose(sample.kernel @ np.ones(2), 0.0, atol=2e-12)


def test_connected_quotient_does_not_require_connected_micro_support():
    graph = nx.Graph([(0, 1), (2, 3)])
    for node in graph:
        graph.nodes[node]["nu_f"] = 1.0
    result = observe_epi_memory(
        graph, ((0, 2), (1, 3)), [1.0, 1.0, -1.0, -1.0], times=[0.0, 1.0]
    )
    assert result.closure_within_tolerance
    np.testing.assert_array_equal(result.initial_hidden, [1.0, 1.0, -1.0, -1.0])
    np.testing.assert_allclose(
        result.hidden_generator @ result.initial_hidden, 0.0, atol=2e-13
    )
    for sample in result.samples:
        np.testing.assert_allclose(sample.kernel, 0.0, atol=2e-13)
        np.testing.assert_allclose(sample.initial_source, 0.0, atol=2e-13)
        np.testing.assert_allclose(sample.projected_epi, 0.0, atol=2e-13)


def test_relabeling_and_node_insertion_order_preserve_macro_observations():
    initial = [2.0, -1.0, 0.25, 3.0, -0.5]
    original = _graph(5)
    labels = {0: "left", 1: ("inner", 1), 2: "center", 3: 7, 4: ("right",)}
    reordered = nx.Graph()
    order = (3, 1, 4, 0, 2)
    for node in order:
        reordered.add_node(labels[node], **original.nodes[node])
    reordered.add_edges_from(
        (labels[left], labels[right]) for left, right in original.edges
    )
    partition = tuple(tuple(labels[node] for node in block) for block in P5_BLOCKS)
    base = observe_epi_memory(original, P5_BLOCKS, initial, times=SAMPLE_TIMES)
    renamed = observe_epi_memory(
        reordered, partition, [initial[node] for node in order], times=SAMPLE_TIMES
    )

    assert renamed.nodes == tuple(reordered)
    assert renamed.blocks == partition
    for left, right in zip(base.samples, renamed.samples, strict=True):
        for name in (
            "kernel",
            "initial_source",
            "convolution",
            "projected_epi",
            "projected_rate",
        ):
            np.testing.assert_allclose(
                getattr(left, name), getattr(right, name), atol=2e-12
            )


def test_observation_is_detached_immutable_and_does_not_write_graph_state():
    graph = _graph(5)
    graph.graph["caller_metadata"] = {"nested": [1, 2, 3]}
    graph.nodes[0]["history"] = [0.25, 0.5]
    before = deepcopy(
        (graph.graph, list(graph.nodes(data=True)), list(graph.edges(data=True)))
    )
    initial = np.array([0.0, 0.0, 3.0, 0.0, 0.0])
    result = observe_epi_memory(graph, P5_BLOCKS, initial, times=[0.0, 0.5])
    assert (
        graph.graph,
        list(graph.nodes(data=True)),
        list(graph.edges(data=True)),
    ) == before
    retained = result.samples[1].projected_epi.copy()
    initial[:] = 20.0
    graph.nodes[0]["nu_f"] = 10.0
    graph.edges[0, 1]["weight"] = 3.0
    np.testing.assert_array_equal(result.initial_epi, [0.0, 0.0, 3.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.samples[1].projected_epi, retained)

    for record in (result, *result.samples):
        for field in fields(record):
            value = getattr(record, field.name)
            if isinstance(value, np.ndarray):
                assert not value.flags.writeable
                with pytest.raises(ValueError):
                    value.setflags(write=True)
                with pytest.raises(ValueError):
                    value.flat[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        result.scope = "changed"


def test_supplied_sample_order_and_repeated_times_are_preserved():
    times = [1.0, 0.0, 1.0, 0.25]
    result = observe_epi_memory(_graph(5), P5_BLOCKS, [0.0] * 5, times=times)
    assert tuple(sample.time for sample in result.samples) == tuple(times)


@pytest.mark.parametrize(
    "times",
    [
        [],
        [-0.1],
        [math.inf],
        [math.nan],
        [True],
        [np.bool_(False)],
        [1.0j],
        ["0.5"],
        0.5,
    ],
)
def test_invalid_observation_times_are_rejected(times):
    with pytest.raises(ValueError):
        observe_epi_memory(_graph(5), P5_BLOCKS, [0.0] * 5, times=times)


@pytest.mark.parametrize(
    "initial",
    [
        [0.0] * 4,
        [[0.0] * 5],
        [0.0, 0.0, math.inf, 0.0, 0.0],
        [0.0, 0.0, math.nan, 0.0, 0.0],
        [0.0, 0.0, True, 0.0, 0.0],
        np.zeros(5, dtype=bool),
        [0.0, 0.0, 0.0j, 0.0, 0.0],
        ["0.0"] * 5,
    ],
)
def test_initial_epi_requires_an_unambiguous_finite_real_vector(initial):
    with pytest.raises(ValueError):
        observe_epi_memory(_graph(5), P5_BLOCKS, initial, times=[0.0])


@pytest.mark.parametrize("capacity", [0.0, -1.0, math.inf, math.nan, True])
def test_reversible_memory_inherits_positive_finite_capacity_scope(capacity):
    graph = _graph(5)
    graph.nodes[2]["nu_f"] = capacity
    with pytest.raises(ValueError):
        observe_epi_memory(graph, P5_BLOCKS, [0.0] * 5, times=[0.0])


def test_asymmetric_transport_is_outside_reversible_memory_scope():
    graph = nx.DiGraph(_graph(5))
    graph.remove_edge(1, 0)
    with pytest.raises(ValueError, match="symmetric"):
        observe_epi_memory(graph, P5_BLOCKS, [0.0] * 5, times=[0.0])


def test_unrepresentable_exponential_arguments_fail_closed():
    with pytest.raises(ValueError, match="finite numeric range"):
        observe_epi_memory(
            _graph(3, [1.0, 2.0, 1.0]),
            ((0, 1), (2,)),
            [1.0, -1.0, 0.0],
            times=[np.finfo(float).max],
        )


@pytest.mark.parametrize("closure_tolerance", [1e-10, 10.0])
def test_large_time_consensus_drift_is_rejected_independently_of_closure(
    closure_tolerance,
):
    # The projected rate identity can have zero residual even when a numerical
    # propagator corrupts consensus. These necessary checks are not accuracy
    # enclosures; closure tolerance must not weaken them.
    with pytest.raises(ValueError, match="stochastic semigroup checks"):
        observe_epi_memory(
            _graph(4),
            ((0, 3), (1, 2)),
            np.ones(4),
            times=[1e16],
            tolerance=closure_tolerance,
        )


@pytest.mark.parametrize("time", [0.0, 1.0])
def test_large_self_loop_cancellation_cannot_certify_a_corrupted_generator(time):
    graph = _graph(4, [1e20] * 4)
    for node in graph:
        graph.add_edge(node, node, weight=1e20)
    # Materializing D-W loses the small off-loop contribution to its diagonal.
    # A time-zero identity flow cannot rescue a generator that moves constants.
    with pytest.raises(ValueError, match="generator"):
        observe_epi_memory(graph, ((0, 3), (1, 2)), np.ones(4), times=[time])


def test_moderate_self_loops_preserve_the_derived_reversible_quotient():
    graph = _graph(4)
    for node in graph:
        graph.add_edge(node, node, weight=1.0)
    result = observe_epi_memory(
        graph, ((0, 3), (1, 2)), [2.0, -1.0, 3.0, 0.5], times=[0.0, 1.0]
    )
    assert result.generator_residual < result.numerical_tolerance
    assert result.closure_within_tolerance
    np.testing.assert_allclose(result.macro_metric_weights, [4.0, 6.0])
    np.testing.assert_allclose(
        result.instantaneous_generator, [[0.5, -0.5], [-1.0 / 3.0, 1.0 / 3.0]]
    )
    for sample in result.samples:
        difference = 0.25 * math.exp(-5.0 * sample.time / 6.0)
        expected = [1.1 + 0.6 * difference, 1.1 - 0.4 * difference]
        np.testing.assert_allclose(sample.projected_epi, expected, atol=2e-12)
        np.testing.assert_allclose(sample.markov_epi, expected, atol=2e-12)
        np.testing.assert_allclose(sample.kernel, 0.0, atol=2e-13)
        assert sample.fine_semigroup_residual < result.numerical_tolerance
        assert sample.markov_semigroup_residual < result.numerical_tolerance


@pytest.mark.parametrize(
    "numerical_tolerance", [True, 0.0, -1.0, math.inf, math.nan, 1.0j, "1e-10"]
)
def test_invalid_numerical_tolerance_is_rejected(numerical_tolerance):
    with pytest.raises(ValueError):
        observe_epi_memory(
            _graph(5),
            P5_BLOCKS,
            [0.0] * 5,
            times=[0.0],
            numerical_tolerance=numerical_tolerance,
        )


@pytest.mark.parametrize("tolerance", [True, 0.0, -1.0, math.inf, math.nan])
def test_invalid_closure_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError):
        observe_epi_memory(
            _graph(5), P5_BLOCKS, [0.0] * 5, times=[0.0], tolerance=tolerance
        )
