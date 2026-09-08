"""Fixed-branch circular phase coarse-graining and its obstruction."""

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr
from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.dynamics.dnfr import dnfr_phase_only
from tnfr.physics import certify_phase_nodal_coarse_graining


def _state(graph, phases, frequencies=None):
    if frequencies is None:
        frequencies = [1.0] * len(graph)
    for node, phase, frequency in zip(
        graph, phases, frequencies, strict=True
    ):
        graph.nodes[node].update(
            EPI=0.0,
            nu_f=float(frequency),
            theta=float(phase),
        )
    return graph


def _bipartite_state(phases):
    return _state(nx.complete_bipartite_graph(3, 3), phases)


def test_fixed_branch_pairwise_channel_closes_but_circular_fibers_do_not():
    graph = _bipartite_state([0.0, 0.2, 0.5, 0.7, 1.0, 1.4])

    result = certify_phase_nodal_coarse_graining(
        graph,
        [(0, 1, 2), (3, 4, 5)],
    )

    assert result.fixed_wrap_branch
    assert result.pairwise_strong_closure_certified
    assert (
        result.pairwise_operator_quotient
        .global_strong_closure_within_tolerance
    )
    assert result.canonical_operator_quotient.sampled_callables_repeatable
    assert not result.canonical_operator_quotient.sampled_fiber_independence
    assert result.relative_pairwise_projection_residual < 1e-12
    assert result.relative_pairwise_lift_residual < 1e-12
    assert result.canonical_lift_hypotheses_satisfied
    assert result.canonical_lift_closure_certified
    assert result.canonical_lift_residual == pytest.approx(0.0)

    # q and P R q have exactly the same macro chart coordinate.  Nevertheless,
    # the nonlinear phasor mean sees the unresolved within-block phases.
    assert result.same_macro_state_residual < 1e-12
    assert result.current_state_is_counterexample
    assert result.global_canonical_projected_closure is False
    assert result.sampled_fiber_dependence_residual > 1e-5
    assert not result.sampled_projected_closure_within_tolerance
    assert result.support_status == "lift_closed_with_global_counterexample"
    assert "COUNTEREXAMPLE" in result.claim_status


def test_block_constant_state_certifies_only_the_lifted_subspace():
    graph = _bipartite_state([0.2, 0.2, 0.2, 1.1, 1.1, 1.1])

    result = certify_phase_nodal_coarse_graining(
        graph,
        [(0, 1, 2), (3, 4, 5)],
    )

    assert result.canonical_lift_closure_certified
    assert result.sampled_projected_closure_within_tolerance
    assert not result.current_state_is_counterexample
    assert result.global_canonical_projected_closure is None
    assert result.support_status == "lift_subspace_closed"
    assert "global canonical projected autonomy remains unproved" in (
        result.claim_status
    )


def test_certificate_matches_the_engine_phase_only_pressure_hook():
    phases = [0.0, 0.2, 0.5, 0.7, 1.0, 1.4]
    frequencies = [0.5, 0.5, 0.5, 1.5, 1.5, 1.5]
    graph = _state(nx.complete_bipartite_graph(3, 3), phases, frequencies)
    result = certify_phase_nodal_coarse_graining(
        graph,
        [(0, 1, 2), (3, 4, 5)],
    )

    dnfr_phase_only(graph)
    observed_rate = np.array(
        [
            graph.nodes[node]["nu_f"]
            * float(
                get_attr(
                    graph.nodes[node],
                    ALIAS_DNFR,
                    strict=True,
                )
            )
            for node in graph
        ]
    )

    np.testing.assert_allclose(
        result.micro_canonical_nodal_rate,
        observed_rate,
        atol=1e-14,
    )
    assert not result.phase_chart.flags.writeable
    assert not result.micro_canonical_nodal_rate.flags.writeable


def test_phase_geometry_does_not_read_or_constrain_epi():
    graph = _bipartite_state([0.0, 0.2, 0.5, 0.7, 1.0, 1.4])
    for node in graph:
        graph.nodes[node]["EPI"] = 1.0 + 2.0j

    result = certify_phase_nodal_coarse_graining(
        graph,
        [(0, 1, 2), (3, 4, 5)],
    )

    assert result.fixed_wrap_branch
    assert result.pairwise_strong_closure_certified
    assert result.current_state_is_counterexample
    assert all(graph.nodes[node]["EPI"] == 1.0 + 2.0j for node in graph)


def test_canonical_phasor_mean_is_unweighted_but_projection_uses_conductance():
    phases = [0.0, 0.2, 0.5, 0.7, 1.0, 1.4]
    uniform = _bipartite_state(phases)
    weighted = _bipartite_state(phases)
    weighted.edges[0, 3]["weight"] = 7.0
    partition = [(0, 1, 2), (3, 4, 5)]

    first = certify_phase_nodal_coarse_graining(uniform, partition)
    second = certify_phase_nodal_coarse_graining(weighted, partition)

    assert second.canonical_neighbor_averaging == "unweighted_support"
    assert second.projection_metric == "conductance_degree_over_capacity"
    np.testing.assert_allclose(
        first.micro_canonical_nodal_rate,
        second.micro_canonical_nodal_rate,
    )
    assert not np.allclose(
        first.projection,
        second.projection,
    )

    dnfr_phase_only(weighted)
    observed_weighted_rate = np.array(
        [
            weighted.nodes[node]["nu_f"]
            * float(
                get_attr(
                    weighted.nodes[node],
                    ALIAS_DNFR,
                    strict=True,
                )
            )
            for node in weighted
        ]
    )
    np.testing.assert_allclose(
        second.micro_canonical_nodal_rate,
        observed_weighted_rate,
        atol=1e-14,
    )


def test_zero_weight_edges_define_canonical_macro_support():
    graph = nx.Graph()
    graph.add_nodes_from(range(6))
    graph.add_edges_from([(0, 2), (1, 3), (2, 4), (3, 5)])
    graph.add_edge(0, 4, weight=0.0)
    graph.add_edge(1, 5, weight=0.0)
    graph = _state(graph, [0.1, 0.1, 0.6, 0.6, 1.1, 1.1])

    result = certify_phase_nodal_coarse_graining(
        graph,
        [(0, 1), (2, 3), (4, 5)],
    )

    # The weighted pairwise quotient is a path, while the canonical phase
    # channel sees the zero-weight A-C edges and therefore has triangle support.
    assert result.macro_conductance[0, 2] == 0.0
    assert result.macro_neighbor_support == ((1, 2), (0, 2), (0, 1))
    np.testing.assert_allclose(
        result.neighbor_multiplicity,
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
    )
    assert result.pairwise_strong_closure_certified
    assert result.canonical_lift_hypotheses_satisfied
    assert result.canonical_lift_closure_certified
    assert result.relative_canonical_lift_residual < 1e-12
    assert result.support_status == "lift_subspace_closed"

    dnfr_phase_only(graph)
    observed_rate = np.array(
        [
            graph.nodes[node]["nu_f"]
            * float(get_attr(graph.nodes[node], ALIAS_DNFR, strict=True))
            for node in graph
        ]
    )
    np.testing.assert_allclose(
        result.micro_canonical_nodal_rate,
        observed_rate,
        atol=1e-14,
    )


@pytest.mark.parametrize("graph_type", [nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph])
def test_canonical_support_follows_networkx_neighbor_semantics(graph_type):
    graph = graph_type()
    graph.add_nodes_from(range(6))
    positive_edges = [(0, 2), (1, 3), (2, 4), (3, 5)]
    if graph.is_directed():
        for source, target in positive_edges:
            graph.add_edge(source, target)
            graph.add_edge(target, source)
    else:
        graph.add_edges_from(positive_edges)

    for source, target in [(0, 4), (1, 5)]:
        graph.add_edge(source, target, weight=0.0)
        if graph.is_multigraph():
            graph.add_edge(source, target, weight=0.0)
    graph = _state(graph, [0.1, 0.1, 0.6, 0.6, 1.1, 1.1])

    result = certify_phase_nodal_coarse_graining(
        graph,
        [(0, 1), (2, 3), (4, 5)],
    )

    expected = (
        ((1, 2), (0, 2), (1,))
        if graph.is_directed()
        else ((1, 2), (0, 2), (0, 1))
    )
    assert result.macro_neighbor_support == expected
    assert result.canonical_lift_hypotheses_satisfied
    assert result.canonical_lift_closure_certified
    assert result.relative_canonical_lift_residual < 1e-12


def test_unequal_macro_multiplicities_block_canonical_family_closure():
    graph = nx.Graph()
    graph.add_nodes_from(range(8))
    graph.add_edges_from(
        [
            (0, 2),
            (1, 3),
            (0, 4),
            (0, 5),
            (1, 6),
            (1, 7),
        ]
    )
    graph = _state(
        graph,
        [0.1, 0.1, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9],
    )

    result = certify_phase_nodal_coarse_graining(
        graph,
        [(0, 1), (2, 3), (4, 5, 6, 7)],
    )

    assert result.pairwise_strong_closure_certified
    assert result.equitable_neighbor_profiles
    assert result.no_internal_fiber_edges
    assert not result.uniform_active_macro_multiplicity
    assert not result.canonical_lift_hypotheses_satisfied
    assert not result.canonical_lift_closure_certified
    assert result.canonical_lift_residual > 1e-3
    np.testing.assert_allclose(
        result.neighbor_multiplicity,
        [[0.0, 1.0, 2.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )


def test_nonequitable_partition_fails_the_pairwise_global_identity():
    graph = _state(
        nx.path_graph(5),
        [0.0, 0.2, 0.4, 0.6, 0.8],
    )

    result = certify_phase_nodal_coarse_graining(
        graph,
        [(0, 4), (1, 2, 3)],
    )

    assert result.fixed_wrap_branch
    assert not result.reversible_partition_closure_within_tolerance
    assert not result.pairwise_strong_closure_certified
    assert result.relative_pairwise_projection_residual > 1e-3
    assert result.relative_pairwise_lift_residual > 1e-3
    assert not result.equitable_neighbor_profiles


def test_non_semicircular_state_abstains_from_fixed_branch_claims():
    graph = _bipartite_state([0.0, 2.0, -2.0, 0.2, 1.8, -1.8])

    result = certify_phase_nodal_coarse_graining(
        graph,
        [(0, 1, 2), (3, 4, 5)],
    )

    assert not result.fixed_wrap_branch
    assert result.wrap_branch_margin < 0.0
    assert not result.pairwise_strong_closure_certified
    assert not result.canonical_lift_closure_certified
    assert not result.current_state_is_counterexample
    assert result.global_canonical_projected_closure is None
    assert result.support_status == "abstained_wrap_branch"


def test_phase_rate_residuals_are_invariant_under_global_rotation():
    phases = [6.0, 6.08, 6.24, 0.07, 0.19, 0.42]
    rotated = [math.remainder(value + 1.234, math.tau) for value in phases]
    partition = [(0, 1, 2), (3, 4, 5)]

    first = certify_phase_nodal_coarse_graining(
        _bipartite_state(phases),
        partition,
    )
    second = certify_phase_nodal_coarse_graining(
        _bipartite_state(rotated),
        partition,
    )

    np.testing.assert_allclose(
        first.micro_canonical_nodal_rate,
        second.micro_canonical_nodal_rate,
        atol=1e-14,
    )
    assert first.sampled_projection_residual == pytest.approx(
        second.sampled_projection_residual,
        abs=1e-14,
    )
    assert first.sampled_fiber_dependence_residual == pytest.approx(
        second.sampled_fiber_dependence_residual,
        abs=1e-14,
    )
    assert first.current_state_is_counterexample
    assert second.current_state_is_counterexample


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("theta", True, "not boolean"),
        ("nu_f", np.bool_(True), "not boolean"),
        ("theta", math.inf, "finite real"),
    ],
)
def test_phase_quotient_rejects_invalid_nodal_scalars(field, value, message):
    graph = _bipartite_state([0.0, 0.2, 0.5, 0.7, 1.0, 1.4])
    graph.nodes[0][field] = value

    with pytest.raises(ValueError, match=message):
        certify_phase_nodal_coarse_graining(
            graph,
            [(0, 1, 2), (3, 4, 5)],
        )


@pytest.mark.parametrize("value", [True, np.bool_(False), math.nan])
def test_phase_quotient_rejects_invalid_chart_reference(value):
    graph = _bipartite_state([0.0, 0.2, 0.5, 0.7, 1.0, 1.4])

    with pytest.raises(ValueError, match="chart_reference.*finite real"):
        certify_phase_nodal_coarse_graining(
            graph,
            [(0, 1, 2), (3, 4, 5)],
            chart_reference=value,
        )