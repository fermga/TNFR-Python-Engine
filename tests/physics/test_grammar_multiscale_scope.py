"""Exact quotient closure does not impose a U5 coherence-average inequality."""

from fractions import Fraction as Q

import networkx as nx
import numpy as np

from tnfr.metrics.common import structural_coherence
from tnfr.physics.epi_memory import observe_forced_support_closure
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.structural_morphism import _build_reversible_partition_geometry
from tnfr.physics.support_transport import observe_support_transport


_BLOCKS = ((0, 1), (2, 3))


def _quotient(epi):
    graph = nx.complete_graph(4)
    for node, value in enumerate(epi):
        graph.nodes[node].update(EPI=float(value), nu_f=1.0, theta=0.0)
    snapshot = observe_support_transport(graph)
    reference = derive_forced_support_balance(
        snapshot, epi_weight=Q(1), forcing=(Q(0),) * 4,
    )
    closure = observe_forced_support_closure(reference, _BLOCKS)
    return graph, snapshot, closure


def _coherence(pressure, rate):
    exact = Q(1) / (1 + abs(pressure) + abs(rate))
    assert structural_coherence(pressure, rate) == float(exact)
    return exact


def test_k4_quotient_closes_exactly_with_derived_capacity_two_thirds():
    graph, snapshot, closure = _quotient((Q(1), Q(1, 4), Q(11, 8), Q(11, 8)))
    assert snapshot.epi_gradient == (0, 1, Q(-1, 2), Q(-1, 2))
    assert closure.metric_weights == (3,) * 4
    assert closure.macro_metric_weights == (6, 6)
    assert closure.all_state_affine_closed
    assert closure.lifted_affine_subspace_invariant
    assert all(value == 0 for row in closure.hidden_to_macro for value in row)
    assert all(value == 0 for row in closure.instantaneous_kernel for value in row)
    assert closure.witness is None
    assert closure.macro_generator == (
        (Q(2, 3), Q(-2, 3)), (Q(-2, 3), Q(2, 3)),
    )

    # The shared quotient owner aggregates four inter-block unit edges and
    # divides external strength four by the macro H weight six. This pure
    # geometry builder performs no matrix-exponential or trajectory probes.
    geometry = _build_reversible_partition_geometry(graph, _BLOCKS, tolerance=1e-12)
    np.testing.assert_array_equal(geometry.macro_conductance, ((0, 4), (4, 0)))
    np.testing.assert_array_equal(geometry.macro_metric_weights, (6, 6))
    np.testing.assert_array_equal(geometry.macro_frequency, (float(Q(2, 3)),) * 2)
    assert geometry.nodal_closure_within_tolerance
    assert Q(4) / closure.macro_metric_weights[0] == Q(2, 3)


def test_exact_autonomous_parent_can_have_less_coherence_than_child_average():
    _, snapshot, closure = _quotient((Q(1), Q(1, 4), Q(11, 8), Q(11, 8)))
    assert closure.all_state_affine_closed
    assert closure.projected_epi == (Q(5, 8), Q(11, 8))
    assert closure.projected_nodal_rate == (Q(1, 2), Q(-1, 2))
    assert closure.affine_macro_rate == closure.projected_nodal_rate
    parent_pressure = closure.projected_epi[1] - closure.projected_epi[0]
    assert parent_pressure == Q(3, 4)
    assert Q(2, 3) * parent_pressure == closure.projected_nodal_rate[0]
    parent = _coherence(parent_pressure, closure.projected_nodal_rate[0])
    children = tuple(_coherence(p, p) for p in snapshot.epi_gradient[:2])
    assert parent == Q(4, 9)
    assert children == (Q(1), Q(1, 3))
    assert parent < sum(children) / 2 == Q(2, 3)


def test_exact_quotient_can_also_increase_coherence_by_signed_cancellation():
    _, snapshot, closure = _quotient((Q(1, 4), Q(7, 4), Q(1), Q(1)))
    assert closure.all_state_affine_closed
    assert snapshot.epi_gradient == (1, -1, 0, 0)
    assert closure.projected_epi == (1, 1)
    assert closure.projected_nodal_rate == (0, 0)
    parent = _coherence(Q(0), Q(0))
    children = tuple(_coherence(p, p) for p in snapshot.epi_gradient[:2])
    assert parent == 1
    assert children == (Q(1, 3), Q(1, 3))
    assert parent > sum(children) / 2
