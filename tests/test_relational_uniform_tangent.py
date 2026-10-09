"""Shared joint derivatives under exact form or phase admission."""

import math
import pickle
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_consensus_tangent,
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


def test_acute_tangent_uses_the_actual_phasor_at_a_large_retained_raw_lift():
    raw = math.tau * 2**56
    gap = math.atan2(math.sin(raw), math.cos(raw))
    graph = _graph(nx.path_graph(2), (0.0, raw))
    report = evaluate_relational_uniform_tangent(
        graph, model=RelationalExchangeModel(1.0)
    )
    metric = math.pi * math.sin(raw) / gap
    laplacian = np.array([[1.0, -1.0], [-1.0, 1.0]])
    expected = np.block(
        [
            [-0.5 * laplacian, -0.5 / math.pi * laplacian],
            [0.5 / metric * laplacian, np.zeros((2, 2))],
        ]
    )
    assert report.field.phase == (0.0, raw)
    assert report.field.form_rate[0] > 0 > report.field.form_rate[1]
    np.testing.assert_allclose(report.generator, expected, rtol=2e-15, atol=0)


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


@pytest.mark.parametrize("domain", ("acute", "positive_resultant", "regular"))
def test_phase_consensus_nonuniform_form_matches_independent_full_field_derivative(
    domain,
):
    graph = _graph(
        nx.Graph(((0, 1), (1, 2), (2, 0), (2, 3))),
        (0.25,) * 4,
        (1.0, 0.5, 2.0, 1.5),
    )
    for node, value in enumerate((0.75, -0.25, 0.125, -0.5)):
        graph.nodes[node]["EPI"] = value
    model = RelationalExchangeModel(
        1.7, epi_weight=2.0, phase_weight=3.0, phase_domain=domain
    )
    before = _snapshot(graph)
    report = evaluate_relational_consensus_tangent(graph, model=model)
    assert _snapshot(graph) == before
    assert any(report.field.form_rate) and any(report.field.phase_rate)
    with pytest.raises(ValueError, match="exactly uniform represented EPI"):
        evaluate_relational_uniform_tangent(graph, model=model)
    adjacency = nx.to_numpy_array(graph, nodelist=report.field.nodes)
    degrees = adjacency.sum(axis=1)
    laplacian = np.diag(degrees) - adjacency
    nu = np.array(report.field.capacity)
    K = np.diag(nu / degrees)
    e, w = model.effective_weights
    expected = np.block(
        [
            [-e * K @ laplacian, -w / math.pi * K @ laplacian],
            [w / (model.storage_scale * math.pi) * K @ laplacian, np.zeros((4, 4))],
        ]
    )
    np.testing.assert_allclose(report.generator, expected, rtol=2e-15, atol=0)

    def independent_field(state):
        form, phase = state[:4], state[4:]
        q = laplacian @ form
        relative = phase[None, :] - phase[:, None]
        real = (adjacency * np.cos(relative)).sum(axis=1)
        imaginary = (adjacency * np.sin(relative)).sum(axis=1)
        argument = np.arctan2(imaginary, real)
        metric = np.array(
            [
                math.pi * R if angle == 0 else math.pi * S / angle
                for R, S, angle in zip(real, imaginary, argument)
            ]
        )
        return np.concatenate(
            (
                nu * (-e * q / degrees + w * argument / math.pi),
                w / model.storage_scale * nu * q / metric,
            )
        )

    state = np.array(report.field.epi + report.field.phase)
    np.testing.assert_allclose(
        report.field.form_rate + report.field.phase_rate,
        independent_field(state),
        rtol=2e-15,
        atol=0,
    )
    increment = 2**-18
    observed = np.column_stack(
        [
            (
                independent_field(state + increment * direction)
                - independent_field(state - increment * direction)
            )
            / (2 * increment)
            for direction in np.eye(8)
        ]
    )
    np.testing.assert_allclose(report.generator, observed, rtol=0, atol=2e-10)
    assert report.common_offset_residuals == tuple(
        (sum(map(Q, row[:4])), sum(map(Q, row[4:]))) for row in report.generator
    )


def test_phase_consensus_tangent_reuses_one_capture_and_retains_offsets_and_labels(
    monkeypatch,
):
    from tnfr.dynamics import relational as owner

    graph = _graph(nx.path_graph(3), (0.25,) * 3, (1.0, 0.5, 2.0))
    for node, value in enumerate((0.5, -0.25, 0.125)):
        graph.nodes[node]["EPI"] = value
    before = _snapshot(graph)
    evaluate = owner.evaluate_relational_exchange
    calls = []

    def observed(*args, **kwargs):
        calls.append(1)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(owner, "evaluate_relational_exchange", observed)
    model = RelationalExchangeModel(1.0)
    original = evaluate_relational_consensus_tangent(graph, model=model)
    assert calls == [1] and _snapshot(graph) == before
    labels = {node: ("node", 10 - node) for node in graph}
    shifted = nx.relabel_nodes(graph, labels)
    for node in shifted:
        shifted.nodes[node]["EPI"] += 8
        shifted.nodes[node]["theta"] += 3
    translated = evaluate_relational_consensus_tangent(shifted, model=model)
    assert translated.field.nodes == tuple(labels[node] for node in graph)
    assert translated.generator == original.generator
    assert translated.field.form_rate == original.field.form_rate
    assert translated.field.phase_rate == original.field.phase_rate
    assert translated.field.epi != original.field.epi
    assert translated.field.phase != original.field.phase
    assert translated.phase_source_row_sum_residuals == tuple(
        sum(map(Q, row)) for row in translated.phase_source_jacobian
    )
    graph.nodes[0]["EPI"] = 123.0
    assert original.field.epi[0] == 0.5


def test_phase_and_form_tangent_owners_agree_only_when_both_premises_hold():
    graph = _graph(nx.path_graph(3), (0.25,) * 3, (1.0, 0.0, 2.0))
    model = RelationalExchangeModel(2.0)
    uniform = evaluate_relational_uniform_tangent(graph, model=model)
    consensus = evaluate_relational_consensus_tangent(graph, model=model)
    for name in (
        "field",
        "generator",
        "phase_source_jacobian",
        "common_offset_residuals",
    ):
        assert getattr(uniform, name) == getattr(consensus, name)
    assert "exactly_uniform_represented_form" in uniform.scope
    assert (
        "baseline_nonzero_rates_retained_without_equilibrium_projection"
        in consensus.scope
    )
    graph.nodes[0]["EPI"] = 1.0
    nonuniform = evaluate_relational_consensus_tangent(graph, model=model)
    assert nonuniform.generator[1] == nonuniform.generator[4] == (0.0,) * 6
    assert nonuniform.field.pressure[1] != 0
    assert nonuniform.field.form_rate[1] == nonuniform.field.phase_rate[1] == 0


def test_phase_consensus_tangent_rejects_exact_nonzero_phase_defect_without_projection():
    graph = _graph(nx.path_graph(2), (0.25, math.nextafter(0.25, 1.0)))
    graph.nodes[0]["EPI"] = 1.0
    before = _snapshot(graph)
    with pytest.raises(ValueError, match="exactly equal represented raw phases"):
        evaluate_relational_consensus_tangent(graph, model=RelationalExchangeModel(1.0))
    assert _snapshot(graph) == before


@pytest.mark.parametrize("attribute", ("EPI", "theta", "nu_f"))
def test_phase_consensus_tangent_preserves_authoritative_scalar_admission(attribute):
    graph = _graph(nx.path_graph(2), (0.0, 0.0))
    graph.nodes[0][attribute] = True
    with pytest.raises((ValueError, TypeError)):
        evaluate_relational_consensus_tangent(graph, model=RelationalExchangeModel(1.0))
