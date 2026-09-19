"""Independent held-law controls for inherited joint nodal quotients.

The fixtures declare phase, capacity, conductance and channel coefficients
before evaluating pressure. Exact projections below use primitive graph data;
the independent circular calculation checks the counted phase realization
numerically. Neither phase rounding nor stored pressure is used to fit a source.
These controls do not establish a phase law or select a coherent partition.
"""

import math
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import networkx as nx
import pytest

from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.physics.joint_quotient import observe_joint_nodal_quotient

EQUAL_WEIGHTS = {"phase": 0.25, "epi": 0.25, "vf": 0.25, "topo": 0.25}


def _prepared(
    graph, blocks, phase, capacity, epi, *, weights=EQUAL_WEIGHTS, refresh=True
):
    for block, angle, nu in zip(blocks, phase, capacity, strict=True):
        for node in block:
            graph.nodes[node].update(theta=float(angle), nu_f=float(nu))
    for node, value in zip(graph, epi, strict=True):
        graph.nodes[node].update(EPI=float(value), delta_nfr=0.0)
    if weights is not None:
        graph.graph["DNFR_WEIGHTS"] = dict(weights)
    if refresh:
        default_compute_delta_nfr(graph)
    return graph


def _k23(*, epi=(-0.5, 0.5, 0.25, 1.25, 2.25), refresh=True):
    blocks = ((0, 1), (2, 3, 4))
    graph = _prepared(
        nx.complete_bipartite_graph(2, 3),
        blocks,
        (0.125, 0.875),
        (0.5, 2.0),
        epi,
        refresh=refresh,
    )
    return graph, blocks


def _project(graph, blocks, values):
    """Direct metric projection, independent of either quotient observer."""
    nodes = tuple(graph)
    lookup = dict(zip(nodes, values, strict=True))
    metric = {
        node: sum((F(data.get("weight", 1)) for data in graph[node].values()), F(0))
        / F(graph.nodes[node]["nu_f"])
        for node in nodes
    }
    return tuple(
        sum((metric[node] * lookup[node] for node in block), F(0))
        / sum((metric[node] for node in block), F(0))
        for block in blocks
    )


def _primitive_channels(graph, blocks):
    """Compute neighbor counts and analytic phasor direction from the graph."""
    membership = {node: a for a, block in enumerate(blocks) for node in block}
    counts = tuple(
        tuple(
            sum(membership[j] == b for j in graph[block[0]]) for b in range(len(blocks))
        )
        for block in blocks
    )
    phase = tuple(graph.nodes[block[0]]["theta"] for block in blocks)
    capacity = tuple(F(graph.nodes[block[0]]["nu_f"]) for block in blocks)
    degrees = tuple(sum(row) for row in counts)
    # All these fixtures lie in one small positive open semicircle. There
    # is no wrap ambiguity and no vanishing trigonometric resultant.
    circular = tuple(
        (
            math.atan2(
                math.fsum(n * math.sin(theta) for n, theta in zip(row, phase)),
                math.fsum(n * math.cos(theta) for n, theta in zip(row, phase)),
            )
            - phase[a]
        )
        / math.pi
        for a, row in enumerate(counts)
    )

    def gradient(values):
        return tuple(
            sum((n * values[b] for b, n in enumerate(row)), F(0)) / degrees[a]
            - values[a]
            for a, row in enumerate(counts)
        )

    return counts, circular, gradient(capacity), gradient(degrees)


def _check_actual_pressure_accounting(graph, blocks, result):
    """Bind exact bookkeeping to the actual public pressure hook output."""
    fine = result.fine_capture
    actual = tuple(F(graph.nodes[node]["delta_nfr"]) for node in graph)
    assert fine.snapshot.stored_pressure == actual
    rates = tuple(F(graph.nodes[node]["nu_f"]) * p for node, p in zip(graph, actual))
    assert result.projected_stored_rate == _project(graph, blocks, rates)
    model_pressure = tuple(
        fine.epi_weight * gradient + force
        for gradient, force in zip(fine.snapshot.epi_gradient, fine.forcing)
    )
    model_rates = tuple(nu * p for nu, p in zip(fine.snapshot.capacity, model_pressure))
    assert result.projected_model_rate == _project(graph, blocks, model_rates)
    assert result.projected_model_rate == tuple(
        a + b
        for a, b in zip(result.effective_rate, result.phase_materialization_rate_defect)
    )
    assert result.projected_fresh_rate == tuple(
        a + b
        for a, b in zip(
            result.projected_model_rate, result.projected_kernel_rate_defect
        )
    )
    assert result.projected_stored_rate == tuple(
        a + b
        for a, b in zip(
            result.projected_fresh_rate, result.projected_stored_rate_residual
        )
    )
    expected_phase_defect = tuple(
        value - result.counted_phase_gradient[a]
        for value, a in zip(fine.phase_gradient, result.structure.node_blocks)
    )
    assert result.counted_phase_materialization_defect == expected_phase_defect
    assert result.phase_materialization_rate_defect == _project(
        graph,
        blocks,
        tuple(
            nu * dict(fine.normalized_weights)["phase"] * value
            for nu, value in zip(fine.snapshot.capacity, expected_phase_defect)
        ),
    )


def test_k23_all_positive_channels_reconstruct_from_independent_primitives():
    graph, blocks = _k23()
    counts, phase, vf, topo = _primitive_channels(graph, blocks)
    result = observe_joint_nodal_quotient(graph, blocks)
    assert result.structure.multiplicity == counts == ((0, 3), (2, 0))
    assert result.block_phase == (F(1, 8), F(7, 8))
    assert (
        result.structure.block_capacity
        == result.structure.effective_capacity
        == (F(1, 2), F(2))
    )
    assert result.structure.source_scale == (1, 1)
    assert result.structure.capacity_gradient == vf == (F(3, 2), F(-3, 2))
    assert result.structure.topology_gradient == topo == (-1, 1)
    assert tuple(map(float, result.counted_phase_gradient)) == pytest.approx(
        phase, abs=2e-16
    )
    assert phase == pytest.approx((0.75 / math.pi, -0.75 / math.pi), abs=2e-16)
    channels = dict(result.counted_components)
    assert channels["phase"] == tuple(
        value / 4 for value in result.counted_phase_gradient
    )
    assert channels["vf"] == (F(3, 8), F(-3, 8))
    assert channels["topo"] == (F(-1, 4), F(1, 4))
    assert result.inherited_components == result.counted_components
    assert result.effective_forcing == tuple(
        value / 4 + offset
        for value, offset in zip(result.counted_phase_gradient, (F(1, 8), F(-1, 8)))
    )
    assert result.closure.projected_epi == (0, F(5, 4))
    # Exact macro transport is independently fixed by the inherited edge
    # strength (6,6) and metric (12,3), not by the observed rates.
    assert result.structure.macro_metric_weights == (12, 3)
    assert result.effective_pressure == tuple(
        force + diffusion
        for force, diffusion in zip(result.effective_forcing, (F(5, 16), F(-5, 16)))
    )
    assert result.effective_rate == tuple(
        nu * p for nu, p in zip((F(1, 2), F(2)), result.effective_pressure)
    )
    assert result.closure.all_state_affine_closed
    assert any(result.closure.hidden_epi)
    assert result.closure.hidden_rate_contribution == (0, 0)
    assert result.fine_capture.stored_pressure_residual == (0,) * 5
    _check_actual_pressure_accounting(graph, blocks, result)


def test_k3_default_capacity_channel_needs_internal_counts_and_retained_capacity():
    blocks = ((0,), (1, 2))
    graph = _prepared(
        nx.complete_graph(3), blocks, (0, 0), (1, 1), (0, 0, 0), weights=None
    )
    result = observe_joint_nodal_quotient(graph, blocks)
    assert result.structure.multiplicity == ((0, 2), (1, 1))
    assert result.structure.aggregated_conductance == ((0, 2), (2, 2))
    assert result.structure.macro_conductance == ((0, 2), (2, 0))
    assert result.structure.block_capacity == (1, 1)
    assert result.structure.effective_capacity == (1, F(1, 2))
    assert result.structure.source_scale == (1, 2)
    assert result.structure.capacity_gradient == (0, 0)
    assert result.effective_forcing == result.projected_stored_rate == (0, 0)
    weights = dict(result.fine_capture.normalized_weights)
    assert weights["topo"] == 0 and weights["vf"] > 0
    # This deliberately naive loop-free graph substitutes effective capacity
    # into the original capacity channel. Its default recipe creates motion
    # although every fine channel vanishes in the uniform fine state.
    macro = nx.Graph()
    macro.add_edge(0, 1, weight=2.0)
    macro = _prepared(macro, ((0,), (1,)), (0, 0), (1, 0.5), (0, 0), weights=None)
    actual_macro_rate = tuple(
        F(macro.nodes[node]["nu_f"]) * F(macro.nodes[node]["delta_nfr"])
        for node in macro
    )
    assert actual_macro_rate == (weights["vf"] * F(-1, 2), weights["vf"] * F(1, 4))
    assert actual_macro_rate != result.projected_stored_rate
    _check_actual_pressure_accounting(graph, blocks, result)


def test_tripartite_counted_phase_keeps_unequal_neighbor_multiplicities():
    blocks = ((0, 1), (2, 3, 4), (5, 6, 7, 8))
    original = nx.complete_multipartite_graph(2, 3, 4)
    graph = nx.Graph()
    graph.add_nodes_from(original)
    # Reverse the actual fine neighbor accumulation relative to ascending
    # grouped macro counts; binary64 phase materialization stays observable.
    graph.add_edges_from(reversed(tuple(original.edges)))
    graph = _prepared(graph, blocks, (0.125, 0.625, 1.375), (0.5, 1, 2), (0,) * 9)
    counts, phase, vf, topo = _primitive_channels(graph, blocks)
    result = observe_joint_nodal_quotient(graph, blocks)
    assert result.structure.multiplicity == counts == ((0, 3, 4), (2, 0, 4), (2, 3, 0))
    assert result.structure.capacity_gradient == vf
    assert result.structure.topology_gradient == topo
    assert tuple(map(float, result.counted_phase_gradient)) == pytest.approx(
        phase, abs=2e-16
    )
    uncounted_first_phase = ((0.625 + 1.375) / 2 - 0.125) / math.pi
    assert abs(phase[0] - uncounted_first_phase) > 0.01
    assert result.structure.source_scale == (1, 1, 1)
    assert any(result.counted_phase_materialization_defect)
    assert any(result.phase_materialization_rate_defect)
    assert any(result.projected_kernel_rate_defect)
    assert result.projected_stored_rate_residual == (0, 0, 0)
    _check_actual_pressure_accounting(graph, blocks, result)


def test_hidden_epi_changes_do_not_refit_or_change_the_held_macro_prediction():
    graph_a, blocks = _k23()
    graph_b, _ = _k23(epi=(-1.5, 1.5, -0.75, 1.25, 3.25))
    a = observe_joint_nodal_quotient(graph_a, blocks)
    b = observe_joint_nodal_quotient(graph_b, blocks)
    assert a.fine_capture.snapshot.epi != b.fine_capture.snapshot.epi
    assert a.fine_capture.full_kernel_pressure != b.fine_capture.full_kernel_pressure
    assert a.closure.projected_epi == b.closure.projected_epi == (0, F(5, 4))
    assert a.fine_capture.forcing == b.fine_capture.forcing
    assert a.inherited_components == b.inherited_components
    assert a.closure.macro_generator == b.closure.macro_generator
    assert a.projected_model_rate == b.projected_model_rate
    assert a.effective_rate == b.effective_rate
    assert (
        a.closure.hidden_rate_contribution
        == b.closure.hidden_rate_contribution
        == (0, 0)
    )
    _check_actual_pressure_accounting(graph_a, blocks, a)
    _check_actual_pressure_accounting(graph_b, blocks, b)


def test_fixed_k9_keeps_nonzero_grouped_phase_defect_separate_from_stale_pressure():
    # Interleaved fibers give a fixed adversarial summation order. The
    # different exact coefficient below is numerical materialization, not
    # unresolved EPI memory or a new constitutive source.
    blocks = ((0, 3, 6), (1, 4, 7), (2, 5, 8))
    graph = _prepared(
        nx.complete_graph(9),
        blocks,
        (0.1, 2.1, 4.1),
        (1, 2, 3),
        tuple(i / 8 for i in range(9)),
        refresh=False,
    )
    for node in graph:
        graph.nodes[node]["delta_nfr"] = 7.0
    result = observe_joint_nodal_quotient(graph, blocks)
    expected = tuple(F(-1, 2**52) if i == 4 else F(0) for i in range(9))
    assert result.counted_phase_materialization_defect == expected
    assert result.phase_materialization_rate_defect == (0, F(-1, 6 * 2**52), 0)
    assert result.projected_model_rate != result.effective_rate
    assert any(result.projected_stored_rate_residual)
    assert result.closure.all_state_affine_closed
    assert result.closure.hidden_rate_contribution == (0, 0, 0)
    _check_actual_pressure_accounting(graph, blocks, result)


def test_stale_pressure_is_retained_as_residual_not_reconstructed_as_forcing():
    graph, blocks = _k23()
    before = observe_joint_nodal_quotient(graph, blocks)
    graph.nodes[0]["delta_nfr"] = 17.0
    result = observe_joint_nodal_quotient(graph, blocks)
    assert result.fine_capture.snapshot.stored_pressure[0] == 17
    assert result.fine_capture.forcing == before.fine_capture.forcing
    assert result.effective_forcing == before.effective_forcing
    assert result.effective_rate == before.effective_rate
    assert result.projected_model_rate == before.projected_model_rate
    assert result.projected_kernel_rate_defect == before.projected_kernel_rate_defect
    assert result.projected_stored_rate_residual[0] != 0
    assert result.projected_stored_rate_residual[1] == 0
    _check_actual_pressure_accounting(graph, blocks, result)


def test_joint_observer_keeps_graph_inputs_and_held_rows_explicit():
    graph, blocks = _k23(refresh=False)
    graph.graph["research_note"] = {"values": [1, 2]}
    before_metadata = deepcopy(graph.graph)
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_edges = deepcopy(dict(graph.edges))
    result = observe_joint_nodal_quotient(graph, (iter(block) for block in blocks))
    assert graph.graph == before_metadata
    assert dict(graph.nodes(data=True)) == before_nodes
    assert dict(graph.edges) == before_edges
    assert "_dnfr_weights" not in graph.graph
    assert result.held_parameter_rows == (
        "primitive_phase",
        "fine_capacity",
        "unique_support",
        "conductance",
        "effective_channel_coefficients",
    )
    assert "no autonomous region selection" in result.scope
    with pytest.raises(FrozenInstanceError):
        result.effective_forcing = (0, 0)
    graph.nodes[0]["EPI"] = 999.0
    assert result.fine_capture.snapshot.epi[0] == F(-1, 2)


@pytest.mark.parametrize("field", ("theta", "nu_f"))
def test_near_equal_phase_or_capacity_is_not_exact_block_constancy(field):
    graph, blocks = _k23()
    graph.nodes[1][field] += 2.0**-35
    with pytest.raises(ValueError, match="exactly block-constant"):
        observe_joint_nodal_quotient(graph, blocks)


def test_equitable_unique_counts_do_not_replace_exact_weighted_epi_closure():
    blocks = ((0,), (1, 2))
    graph = nx.Graph()
    graph.add_weighted_edges_from(((0, 1, 1), (0, 2, 2), (1, 2, 1)))
    graph = _prepared(graph, blocks, (0.125, 0.5), (1, 1), (0, 0, 0))
    with pytest.raises(ValueError, match="exact all-state EPI projection closure"):
        observe_joint_nodal_quotient(graph, blocks)


def test_nonequitable_support_is_not_admitted_by_a_special_uniform_epi_state():
    blocks = ((0, 1), (2, 3))
    graph = _prepared(nx.path_graph(4), blocks, (0.125, 0.5), (1, 1), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="equitable support"):
        observe_joint_nodal_quotient(graph, blocks)


@pytest.mark.parametrize("bad", (True, "0.125", float("inf"), float("nan")))
def test_malformed_phase_cannot_be_used_as_a_held_constitutive_coordinate(bad):
    graph, blocks = _k23(refresh=False)
    graph.nodes[0]["theta"] = bad
    with pytest.raises((TypeError, ValueError)):
        observe_joint_nodal_quotient(graph, blocks)
