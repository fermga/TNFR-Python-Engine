"""Linear pressure retains representable differences and active-channel scope."""

from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.backends.optimized_numpy import OptimizedNumPyBackend
from tnfr.constants.aliases import ALIAS_DEPI, ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
import tnfr.dynamics.dnfr as dnfr
from tnfr.dynamics.fused_dnfr import compute_fused_gradients_symmetric
from tnfr.dynamics.integrators import update_epi_via_nodal_equation


PATHS = ("fused", "fallback", "python", "dense", "optimized")


def _graph(values, channel="epi", kind=nx.Graph):
    graph = nx.path_graph(len(values), create_using=kind)
    graph.graph.update(
        RANDOM_SEED=17,
        DNFR_WEIGHTS={key: float(key == channel) for key in ("phase", "epi", "vf", "topo")},
        EPI_MIN=-1e308, EPI_MAX=1e308,
    )
    for node, value in zip(graph, values):
        graph.nodes[node].update(theta=0.0)
        set_attr(graph.nodes[node], ALIAS_EPI, value if channel != "vf" else 0.5)
        set_attr(graph.nodes[node], ALIAS_VF, value if channel == "vf" else 1.0)
    return graph


def _run(graph, path, monkeypatch):
    if path == "optimized":
        OptimizedNumPyBackend().compute_delta_nfr(graph)
    elif path == "dense":
        # Exercise the legacy matrix accumulator rather than the fused dispatcher.
        graph.graph["dnfr_force_dense"] = True
        data = dnfr._prepare_dnfr_data(graph)
        assert data["A"] is not None
        stats = dnfr._build_neighbor_sums_common(graph, data, use_numpy=True)
        x, y, epi_sum, vf_sum, count, deg_sum, degs = stats
        dnfr._compute_dnfr_common(
            graph, data, x=x, y=y, epi_sum=epi_sum, vf_sum=vf_sum,
            count=count, deg_sum=deg_sum, degs=degs,
        )
    elif path == "python":
        import tnfr.mathematics.unified_numerical as numerical
        with monkeypatch.context() as context:
            context.setattr(dnfr, "np", None)
            context.setattr(numerical, "np", None)
            context.setattr(numerical, "NUMPY_AVAILABLE", False)
            graph.graph["vectorized_dnfr"] = False
            dnfr.default_compute_delta_nfr(graph)
    else:
        graph.graph["vectorized_dnfr"] = path == "fused"
        dnfr.default_compute_delta_nfr(graph)
    return np.array([get_attr(graph.nodes[n], ALIAS_DNFR) for n in graph])


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("channel", ["epi", "vf"])
def test_large_common_offset_retains_exact_path_differences(path, channel, monkeypatch):
    graph = _graph([1e16 + 2, 1e16, 1e16 + 4], channel)
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(_run(graph, path, monkeypatch), [-2.0, 3.0, -4.0])


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("channel", ["epi", "vf"])
def test_large_uniform_field_does_not_overflow_neighbor_sum(path, channel, monkeypatch):
    graph = _graph([1e308] * 3, channel)
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(_run(graph, path, monkeypatch), 0.0)


@pytest.mark.parametrize("path", PATHS)
def test_overflowing_pair_difference_can_have_representable_weighted_pressure(path, monkeypatch):
    graph = _graph([0.9e308, -1.7e308, 1.7e308], kind=nx.DiGraph)
    graph.remove_edges_from(list(graph.edges()))
    graph.add_weighted_edges_from([(0, 1, 0.7), (0, 2, 0.3)])
    np.testing.assert_allclose(_run(graph, path, monkeypatch), [-1.58e308, 0, 0], rtol=5e-16)


@pytest.mark.parametrize("path", PATHS)
def test_channel_coefficient_is_applied_before_rejecting_unrepresentable_component(path, monkeypatch):
    graph = _graph([1e308, -1e308])
    graph.graph["DNFR_WEIGHTS"].update(epi=1e-300, phase=1.0)
    np.testing.assert_allclose(_run(graph, path, monkeypatch), [-2e8, 2e8], rtol=5e-16)


@pytest.mark.parametrize("path", PATHS)
def test_disabled_linear_channels_and_zero_edges_do_not_evaluate_extreme_pairs(path, monkeypatch):
    graph = _graph([1e308, -1e308, 1e308])
    graph.graph["DNFR_WEIGHTS"].update(epi=0.0, phase=1.0)
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_VF, (-1) ** node * 1e308)
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(_run(graph, path, monkeypatch), 0.0)
    zero = _graph([1e308, -1e308])
    zero[0][1]["weight"] = 0.0
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_array_equal(_run(zero, path, monkeypatch), 0.0)


@pytest.mark.parametrize("path", PATHS)
def test_unrepresentable_active_pressure_rejected_before_pressure_write(path, monkeypatch):
    graph = _graph([1e308, -1e308])
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_DNFR, 7.0)
    with pytest.raises(ValueError, match="floating-point range"):
        _run(graph, path, monkeypatch)
    assert [get_attr(graph.nodes[n], ALIAS_DNFR) for n in graph] == [7.0, 7.0]


@pytest.mark.parametrize("kind", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph])
def test_weighted_loops_parallel_neighbors_and_aliases_match_exact_rational_means(kind, monkeypatch):
    graph = _graph([1e16 + 2, 1e16, 1e16 + 4, 1e308], kind=kind)
    graph.remove_edges_from(list(graph.edges()))
    graph.add_weighted_edges_from([(0, 0, 0.5), (0, 1, 0.25), (0, 2, 0.75)])
    if graph.is_multigraph():
        graph.add_edge(0, 1, weight=0.25)
    # Conflicting lower-precedence alias must not replace the canonical value.
    for node in graph:
        graph.nodes[node]["epi"] = 99.0
    matrix = nx.to_numpy_array(graph, nodelist=list(graph))
    epi = [Fraction.from_float(get_attr(graph.nodes[n], ALIAS_EPI)) for n in graph]
    expected = []
    for row in range(len(graph)):
        weights = [Fraction.from_float(float(w)) for w in matrix[row]]
        denominator = sum(weights)
        expected.append(float(sum(w * (value - epi[row]) for w, value in zip(weights, epi)) / denominator)
                        if denominator else 0.0)
    np.testing.assert_allclose(_run(graph, "fused", monkeypatch), expected, atol=1e-14)
    np.testing.assert_allclose(_run(graph, "fallback", monkeypatch), expected, atol=1e-14)


@pytest.mark.parametrize("use_jit", [False, True])
def test_fused_jit_dispatch_uses_the_same_linear_reducer(use_jit):
    graph = nx.path_graph(70)
    edges = [(u, v) for u in graph for v in graph.neighbors(u)]
    source, target = np.array(edges, dtype=np.intp).T
    values = np.array([1e16 + 2 * (n % 3) for n in graph])
    expected = [sum(values[v] - values[u] for v in graph.neighbors(u)) / len(list(graph.neighbors(u)))
                for u in graph]
    result = compute_fused_gradients_symmetric(
        edge_src=source, edge_dst=target, phase=np.zeros(len(graph)), epi=values,
        vf=np.ones(len(graph)), weights={"w_epi": 1.0}, use_jit=use_jit,
        accumulate_both_directions=False,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("frequency", [0.0, 1.0])
def test_nodal_integration_applies_capacity_once_and_zero_capacity_freezes(frequency, monkeypatch):
    graph = _graph([1e16 + 2, 1e16, 1e16 + 4])
    graph.graph["DT_MIN"] = 0.0  # One explicit step; no sub-ULP subdivided increments.
    nx.set_node_attributes(graph, frequency, ALIAS_VF[0])
    pressure = _run(graph, "fused", monkeypatch)
    before = [get_attr(graph.nodes[n], ALIAS_EPI) for n in graph]
    update_epi_via_nodal_equation(graph, dt=1.0, method="euler")
    np.testing.assert_array_equal([get_attr(graph.nodes[n], ALIAS_DEPI) for n in graph], frequency * pressure)
    np.testing.assert_array_equal([get_attr(graph.nodes[n], ALIAS_EPI) for n in graph],
                                  np.array(before) + frequency * pressure)


def test_example_linear_hooks_retain_small_differences():
    graph = _graph([1e16 + 2, 1e16, 1e16 + 4])
    dnfr.dnfr_laplacian(graph)
    np.testing.assert_array_equal([get_attr(graph.nodes[n], ALIAS_DNFR) for n in graph], [-2, 3, -4])


@pytest.mark.parametrize("path", PATHS)
def test_underflowing_probability_keeps_representable_pressure(path, monkeypatch):
    graph = _graph([0.0, 1e308, 0.0], kind=nx.DiGraph)
    graph.remove_edges_from(list(graph.edges()))
    graph.add_weighted_edges_from([(0, 1, 1e-200), (0, 2, 1e200)])
    np.testing.assert_allclose(_run(graph, path, monkeypatch), [1e-92, 0, 0], rtol=5e-16, atol=0)


@pytest.mark.parametrize("coefficient", [1e-20, 1e200])
def test_exceptional_products_round_only_after_weight_and_channel_application(coefficient):
    from tnfr.mathematics._neighbor_differences import edge_mean_differences, mean_neighbor_difference

    center, neighbors, weights = 0.0, [1e308, 0.0], [1e-300, 1e300]
    expected = float(Fraction.from_float(weights[0]) * Fraction.from_float(neighbors[0])
                     * Fraction.from_float(coefficient)
                     / sum(Fraction.from_float(w) for w in weights))
    scalar = mean_neighbor_difference(center, neighbors, weights, coefficient=coefficient)
    array = edge_mean_differences([center, *neighbors], [0, 0], [1, 2], weights,
                                 coefficient=coefficient)
    assert scalar == expected
    assert array[0] == expected


@pytest.mark.parametrize("path", PATHS)
@pytest.mark.parametrize("neighbors, weights, expected", [
    ([1e16, 1.0, -1e16], [1.0, 1.0, 1.0], 1.0 / 3.0),
    ([7e16, 1.0, -3e16], [3.0, 1.0, 7.0], 1.0 / 11.0),
])
def test_mixed_sign_cancellation_retains_weighted_residual(path, neighbors, weights, expected, monkeypatch):
    graph = _graph([0.0, *neighbors], kind=nx.DiGraph)
    graph.remove_edges_from(list(graph.edges()))
    graph.add_weighted_edges_from((0, target, weight) for target, weight in enumerate(weights, 1))
    np.testing.assert_array_equal(_run(graph, path, monkeypatch), [expected, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("order", [range(6), [5, 2, 3, 0, 4, 1], reversed(range(6))])
def test_compensated_rows_are_independent_of_interleaved_edge_order(order):
    from tnfr.mathematics._neighbor_differences import edge_mean_differences, mean_neighbor_difference

    values = [0.0, 7e16, 1.0, -3e16, 0.0, 1e16, 1.0, -1e16]
    source = np.array([0, 0, 0, 4, 4, 4])
    target = np.array([1, 2, 3, 5, 6, 7])
    weights = np.array([3.0, 1.0, 7.0, 1.0, 1.0, 1.0])
    order = list(order)
    expected = np.zeros(len(values))
    expected[0], expected[4] = 1.0 / 11.0, 1.0 / 3.0
    result = edge_mean_differences(values, source[order], target[order], weights[order])
    np.testing.assert_array_equal(result, expected)
    assert mean_neighbor_difference(0.0, values[1:4], weights[:3]) == expected[0]
    assert mean_neighbor_difference(0.0, values[5:8], weights[3:]) == expected[4]


@pytest.mark.parametrize("path", PATHS)
def test_overflow_of_assembled_channels_rejects_before_any_pressure_write(path, monkeypatch):
    graph = _graph([0.0, -1.7e308, 1.7e308])
    graph.remove_edges_from(list(graph.edges()))
    graph.add_edge(1, 2)  # Leading isolate must also retain its previous pressure.
    for node, frequency in enumerate([0.0, 0.0, 1.7e308]):
        set_attr(graph.nodes[node], ALIAS_VF, frequency)
        set_attr(graph.nodes[node], ALIAS_DNFR, 7.0)
    graph.graph["DNFR_WEIGHTS"].update(epi=0.5, vf=0.5)
    with np.errstate(over="raise", invalid="raise"):
        with pytest.raises(ValueError, match="Assembled pressure"):
            _run(graph, path, monkeypatch)
    assert [graph.nodes[node][ALIAS_DNFR[0]] for node in graph] == [7.0, 7.0, 7.0]


def test_exact_cancellation_matches_fraction_oracle_for_seeded_weighted_rows():
    from tnfr.mathematics._neighbor_differences import edge_mean_differences, mean_neighbor_difference

    rng = np.random.default_rng(29)
    for _ in range(40):
        center = float(rng.uniform(-4.0, 4.0))
        neighbors = [float(center + rng.uniform(0.1, 4.0)),
                     float(center - rng.uniform(0.1, 4.0)),
                     float(rng.uniform(-4.0, 4.0))]
        weights = [float(rng.uniform(0.1, 4.0)) for _ in neighbors]
        coefficient = float(rng.uniform(0.01, 1.0))
        exact = float(sum(Fraction.from_float(w) * (Fraction.from_float(v) - Fraction.from_float(center))
                          for v, w in zip(neighbors, weights))
                      * Fraction.from_float(coefficient)
                      / sum(Fraction.from_float(w) for w in weights))
        assert mean_neighbor_difference(center, neighbors, weights, coefficient=coefficient) == exact
        array = edge_mean_differences([center, *neighbors], [0, 0, 0], [1, 2, 3], weights,
                                     coefficient=coefficient)
        assert array[0] == exact
