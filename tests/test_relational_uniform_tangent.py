"""Uniform-form ideal derivatives, observed without executing dynamics."""

import math
import pickle
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_exchange,
    evaluate_relational_uniform_tangent,
)


def _graph(graph, phases, capacities=None):
    graph.graph.update(_t=7.0, retained={"history": ["unchanged"]})
    for i, node in enumerate(graph):
        graph.nodes[node].update(
            EPI=-0.75,
            theta=phases[i],
            nu_f=1.0 if capacities is None else capacities[i],
            delta_nfr=999.0,
        )
    return graph


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def test_heterogeneous_consensus_pair_has_independent_analytic_blocks():
    graph = _graph(nx.path_graph(2), (0.0, 0.0), (1.0, 2.0))
    model = RelationalExchangeModel(2.0, epi_weight=3.0, phase_weight=1.0)
    report = evaluate_relational_uniform_tangent(graph, model=model)
    laplacian = np.array([[1.0, -1.0], [-2.0, 2.0]])
    expected = np.block(
        [
            [-0.75 * laplacian, -0.25 / math.pi * laplacian],
            [0.125 / math.pi * laplacian, np.zeros((2, 2))],
        ]
    )
    np.testing.assert_allclose(report.generator, expected, rtol=2e-15, atol=0)
    assert report.field.form_rate == report.field.phase_rate == (0.0, 0.0)


def test_twisted_c5_reuses_its_cosine_metric_and_has_two_offset_directions():
    graph = _graph(nx.cycle_graph(5), tuple(math.tau * i / 5 for i in range(5)))
    report = evaluate_relational_uniform_tangent(
        graph, model=RelationalExchangeModel(1.0)
    )
    adjacency = nx.to_numpy_array(graph)
    laplacian = 2 * np.eye(5) - adjacency
    expected = np.block(
        [
            [-0.25 * laplacian, -0.25 / math.pi * laplacian],
            [0.25 / (math.pi * math.cos(math.tau / 5)) * laplacian, np.zeros((5, 5))],
        ]
    )
    np.testing.assert_allclose(report.generator, expected, rtol=3e-15, atol=1e-16)
    assert max(
        abs(value) for row in report.common_offset_residuals for value in row
    ) < Q(1, 10**14)
    assert report.common_offset_residuals == tuple(
        (sum(map(Q, row[:5])), sum(map(Q, row[5:]))) for row in report.generator
    )
    assert report.phase_source_row_sum_residuals == tuple(
        sum(map(Q, row)) for row in report.phase_source_jacobian
    )


@pytest.mark.parametrize(
    "domain,phases,capacities",
    (
        ("acute", (-0.3, 0.1, 0.55), (1.0, 0.5, 2.0)),
        ("positive_resultant", (-0.3, 0.1, 0.55), (1.0, 0.5, 2.0)),
        ("positive_resultant", (-0.9, -0.9, 0.9, 0.9), (1.0, 0.5, 2.0, 1.5)),
    ),
)
def test_noncritical_uniform_state_matches_native_centered_differences_and_is_read_only(
    domain,
    phases,
    capacities,
):
    size = len(phases)
    graph = _graph(
        nx.path_graph(3) if size == 3 else nx.complete_graph(4), phases, capacities
    )
    model = RelationalExchangeModel(1.7, phase_domain=domain)
    before = _snapshot(graph)
    report = evaluate_relational_uniform_tangent(graph, model=model)
    assert _snapshot(graph) == before
    assert any(report.field.form_rate)
    assert not any(report.field.phase_rate)
    observed = np.empty((2 * size, 2 * size))
    increment = 2.0**-18
    for column in range(2 * size):
        attribute = "EPI" if column < size else "theta"
        samples = []
        for sign in (-1, 1):
            probe = graph.copy()
            probe.nodes[column % size][attribute] += sign * increment
            field = evaluate_relational_exchange(probe, model=model)
            samples.append(np.array(field.form_rate + field.phase_rate))
        observed[:, column] = (samples[1] - samples[0]) / (2 * increment)
    np.testing.assert_allclose(report.generator, observed, rtol=0, atol=2e-10)


def test_one_ulp_of_nonuniform_form_is_rejected_without_modifying_graph():
    graph = _graph(nx.path_graph(2), (0.0, 0.2))
    graph.nodes[1]["EPI"] = math.nextafter(-0.75, 0.0)
    before = _snapshot(graph)
    with pytest.raises(ValueError, match="exactly uniform represented EPI"):
        evaluate_relational_uniform_tangent(graph, model=RelationalExchangeModel(1.0))
    assert _snapshot(graph) == before


def test_zero_capacity_freezes_both_local_rows_but_retains_neighbor_dependence():
    graph = _graph(nx.path_graph(3), (0.0, 0.15, 0.3), (1.0, 0.0, 2.0))
    report = evaluate_relational_uniform_tangent(
        graph, model=RelationalExchangeModel(1.0)
    )
    assert report.generator[1] == report.generator[4] == (0.0,) * 6
    assert report.generator[0][1] != 0
    assert report.generator[0][4] != 0


@pytest.mark.parametrize(
    "key,value",
    (("nu_f", True), ("nu_f", -1.0), ("theta", float("nan")), ("EPI", True)),
)
def test_native_scalar_admission_is_retained(key, value):
    graph = _graph(nx.path_graph(2), (0.0, 0.0))
    graph.nodes[0][key] = value
    with pytest.raises((TypeError, ValueError)):
        evaluate_relational_uniform_tangent(graph, model=RelationalExchangeModel(1.0))


def test_finite_field_does_not_admit_unrepresentable_tangent_coefficients():
    graph = _graph(nx.path_graph(2), (0.0, 0.0), (1e308, 1e308))
    model = RelationalExchangeModel(1e-308)
    assert not any(evaluate_relational_exchange(graph, model=model).phase_rate)
    with pytest.raises((OverflowError, ValueError), match="form-to-phase tangent"):
        evaluate_relational_uniform_tangent(graph, model=model)


def test_nonzero_tangent_product_lost_on_materialization_is_rejected():
    graph = _graph(nx.path_graph(2), (0.0, 0.0), (math.ulp(0.0), 1.0))
    with pytest.raises(ValueError, match="underflows"):
        evaluate_relational_uniform_tangent(graph, model=RelationalExchangeModel(1.0))
