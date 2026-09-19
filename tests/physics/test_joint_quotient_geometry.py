"""Exact EPI quotient closure and inherited potential on a declared P3.

In the P3 controls, unit conductance, capacity and EPI coefficient are held
fixed; phase is zero and the other pressure channels are absent. Changing explicit path lengths
changes the field observation without changing this nodal law. The proof
concerns unrestricted scalar EPI and the averaged structural potential, not
the entire tetrad, an emergent metric law or a runtime trajectory. A separate
star control distinguishes represented phase-pressure arithmetic from the
availability of the geometric circular mean.
"""

import math
from fractions import Fraction as F

import networkx as nx
import pytest

from tnfr.mathematics.krylov import exact_rank
from tnfr.physics._cycle_algebra import dot
from tnfr.physics.canonical import compute_structural_potential
from tnfr.physics.epi_memory import observe_forced_support_closure
from tnfr.physics.fields import observe_phase_curvature
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.hybrid_operator_stability import _exact_matrix_product as product
from tnfr.physics.joint_quotient import observe_joint_nodal_quotient
from tnfr.physics.support_transport import observe_support_transport

NODES = (0, 1, 2)
BLOCKS = ((0, 2), (1,))
HIDDEN = (F(1), F(0), F(-1))
ZERO_OUTPUT = ((F(0),) * 3,) * 2


def _apply(matrix, vector):
    return tuple(dot(row, vector) for row in matrix)


def _negative(matrix):
    return tuple(tuple(-value for value in row) for row in matrix)


def _identity(size):
    return tuple(tuple(F(i == j) for j in range(size)) for i in range(size))


def _graph(second_length=2, epi=(F(3, 4), F(1, 4), F(1, 2))):
    graph = nx.path_graph(3)
    graph.edges[0, 1].update(weight=1, length=1)
    graph.edges[1, 2].update(weight=1, length=second_length)
    # Declared fresh pressure of this isolated channel, independently of the
    # closure and potential observers. All test states are exactly dyadic.
    pressure = (epi[1] - epi[0], (epi[0] + epi[2]) / 2 - epi[1], epi[1] - epi[2])
    for node, value, source in zip(NODES, epi, pressure, strict=True):
        graph.nodes[node].update(EPI=value, nu_f=1, theta=0, delta_nfr=source)
    return graph


def _closure(graph):
    source = observe_support_transport(graph)
    assert source.nodes == NODES
    assert source.stored_pressure == source.epi_gradient
    reference = derive_forced_support_balance(
        source, epi_weight=F(1), forcing=(F(0),) * 3
    )
    return observe_forced_support_closure(reference, BLOCKS)


def _potential_kernel(graph):
    """Independent rational shortest paths for these three-node fixtures.

    No binary64 distance/kernel is rationalized after rounding. Explicit
    lengths are metric input already present on the supplied graph.
    """
    distance = [[F(0) if i == j else None for j in NODES] for i in NODES]
    for left, right, data in graph.edges(data=True):
        distance[left][right] = distance[right][left] = F(data["length"])
    for middle in NODES:
        for left in NODES:
            for right in NODES:
                first, second = distance[left][middle], distance[middle][right]
                if first is not None and second is not None:
                    candidate = first + second
                    old = distance[left][right]
                    if old is None or candidate < old:
                        distance[left][right] = candidate
    assert all(value is not None for row in distance for value in row)
    return tuple(
        tuple(F(0) if i == j else 1 / distance[i][j] ** 2 for j in NODES) for i in NODES
    )


def _output(closure, kernel):
    # Here A=L because capacity and EPI coefficient are both one. The field
    # observes pressure -Lx, not an independently supplied stored pressure.
    return _negative(
        product(product(closure.projection, kernel), closure.micro_generator)
    )


def test_closed_epi_quotient_can_hide_a_nonzero_inherited_potential_difference():
    graph = _graph()
    closure = _closure(graph)
    r, p, q, a = (
        closure.projection,
        closure.lift,
        closure.hidden_projector,
        closure.micro_generator,
    )
    kernel = _potential_kernel(graph)
    output = _output(closure, kernel)
    assert r == ((F(1, 2), F(0), F(1, 2)), (F(0), F(1), F(0)))
    assert kernel == (
        (F(0), F(1), F(1, 9)),
        (F(1), F(0), F(1, 4)),
        (F(1, 9), F(1, 4), F(0)),
    )
    assert closure.all_state_affine_closed
    assert product(product(r, a), q) == ZERO_OUTPUT
    assert product(r, a) == product(closure.macro_generator, r)
    assert product(product(output, p), r) != output
    assert product(output, q) == (
        (F(0), F(0), F(0)),
        (F(-3, 8), F(0), F(3, 8)),
    )
    assert _apply(r, HIDDEN) == _apply(product(r, a), HIDDEN) == (0, 0)
    assert _apply(q, HIDDEN) == HIDDEN
    assert _apply(output, HIDDEN) == (F(0), F(-3, 4))

    shifted = tuple(x + h / 8 for x, h in zip(closure.epi, HIDDEN, strict=True))
    after = observe_forced_support_closure(closure.reference, BLOCKS, epi=shifted)
    assert after.projected_epi == closure.projected_epi
    assert after.projected_nodal_rate == closure.projected_nodal_rate
    assert tuple(
        later - before
        for later, before in zip(
            _apply(output, shifted), _apply(output, closure.epi), strict=True
        )
    ) == (F(0), F(-3, 32))


def test_one_extra_coordinate_is_necessary_and_sufficient_for_joint_output():
    graph = _graph()
    closure = _closure(graph)
    r, a = closure.projection, closure.micro_generator
    output = _output(closure, _potential_kernel(graph))
    assert exact_rank(r) == 2
    assert exact_rank((*r, *output)) == 3
    # Any all-state linear realization retaining both outputs must have rank
    # at least three. The independent coordinates below attain that bound.
    c = (*r, (F(1, 2), F(0), F(-1, 2)))
    lift = ((F(1), F(0), F(1)), (F(0), F(1), F(0)), (F(1), F(0), F(-1)))
    generator = ((F(1), F(-1), F(0)), (F(-1), F(1), F(0)), (F(0), F(0), F(1)))
    decoder = ((F(37, 72), F(-37, 72), F(0)), (F(-5, 4), F(5, 4), F(-3, 4)))
    assert exact_rank(c) == 3
    assert product(c, lift) == product(lift, c) == _identity(3)
    assert product(c, a) == product(generator, c)
    assert product(decoder, c) == output
    assert exact_rank((*c, *product(c, a))) == 3

    # x=(u+h,v,u-h): u'=v-u, v'=u-v, h'=-h. These are instantaneous
    # consequences of the held nodal law, with no selected phase dynamics.
    for u, v, hidden in (
        (F(5, 8), F(1, 4), F(1, 8)),
        (F(1, 2), F(1, 2), F(-1, 4)),
        (F(3, 4), F(1, 8), F(0)),
    ):
        state = (u, v, hidden)
        epi = (u + hidden, v, u - hidden)
        assert _apply(c, epi) == state
        assert _apply(_negative(product(c, a)), epi) == (v - u, u - v, -hidden)
        expected = (F(-37, 72) * (v - u), F(5, 4) * (v - u) - F(3, 4) * hidden)
        assert _apply(output, epi) == _apply(decoder, state) == expected


def test_equal_lengths_close_the_field_through_an_inherited_two_state_kernel():
    graph = _graph(second_length=1)
    closure = _closure(graph)
    r, q = closure.projection, closure.hidden_projector
    kernel = _potential_kernel(graph)
    inherited = ((F(1, 4), F(1)), (F(2), F(0)))
    output = _output(closure, kernel)
    assert closure.all_state_affine_closed
    assert product(r, kernel) == product(inherited, r)
    assert product(product(r, kernel), q) == ZERO_OUTPUT
    assert product(output, q) == ZERO_OUTPUT
    assert output == _negative(product(inherited, product(closure.macro_generator, r)))
    assert exact_rank((*r, *output)) == 2
    assert _apply(output, HIDDEN) == (0, 0)
    # The self-block term and unequal off-diagonal coefficients are inherited
    # fine observations, not the ordinary inverse-square kernel of a P2.


@pytest.mark.parametrize("hidden_shift", (F(0), F(1, 8)))
def test_canonical_readout_matches_exact_geometry_without_changing_the_epi_law(
    hidden_shift,
):
    epi = tuple(
        x + hidden_shift * h
        for x, h in zip((F(3, 4), F(1, 4), F(1, 2)), HIDDEN, strict=True)
    )
    graphs = (_graph(second_length=1, epi=epi), _graph(second_length=2, epi=epi))
    closures = tuple(_closure(graph) for graph in graphs)
    equal, asymmetric = closures
    assert equal.reference.source == asymmetric.reference.source
    assert equal.micro_generator == asymmetric.micro_generator
    assert equal.projected_epi == asymmetric.projected_epi
    assert equal.projected_nodal_rate == asymmetric.projected_nodal_rate

    potentials = []
    for graph, closure in zip(graphs, closures, strict=True):
        exact = _apply(_potential_kernel(graph), closure.reference.source.epi_gradient)
        observed = compute_structural_potential(graph)
        assert observed == pytest.approx(
            dict(zip(NODES, map(float, exact), strict=True)), rel=1e-14, abs=1e-15
        )
        expected_macro = _apply(_output(closure, _potential_kernel(graph)), epi)
        actual_macro = ((observed[0] + observed[2]) / 2, observed[1])
        assert actual_macro == pytest.approx(
            tuple(map(float, expected_macro)), rel=1e-14, abs=1e-15
        )
        potentials.append(exact)
    assert potentials[0] != potentials[1]


def test_joint_kernel_accounting_does_not_certify_a_nonzero_phase_resultant():
    graph = nx.star_graph(4)
    blocks = ((0,), (1, 2), (3,), (4,))
    for block, phase in zip(blocks, (0.37, 0.0, math.pi, -math.pi), strict=True):
        for node in block:
            graph.nodes[node].update(EPI=node / 8, nu_f=1, theta=phase, delta_nfr=7)
    nx.set_edge_attributes(graph, 1, "weight")
    graph.graph["DNFR_WEIGHTS"] = dict.fromkeys(("phase", "epi", "vf", "topo"), 1)

    result = observe_joint_nodal_quotient(graph, blocks)
    assert result.closure.all_state_affine_closed
    assert result.structure.multiplicity[0] == (0, 2, 1, 1)
    assert result.counted_phase_materialization_defect == (0,) * 5
    assert result.projected_model_rate == result.effective_rate
    assert (
        tuple(
            a + b
            for a, b in zip(
                result.projected_model_rate,
                result.projected_kernel_rate_defect,
                strict=True,
            )
        )
        == result.projected_fresh_rate
    )

    # Signed finite phase representatives are admitted by the captured NumPy
    # branch. At this exact represented cancellation it uses atan2(0,0), while
    # the geometric reader correctly reports no available circular curvature.
    # This is neither a strict-U3 fixture nor a transcendental-zero proof.
    center = next(row for row in observe_phase_curvature(graph).rows if row.node == 0)
    assert center.status == "undefined_represented_resultant"
    assert center.resultant.joint_zero and center.curvature is None
    assert result.counted_phase_gradient[0] != 0
    assert "not certified" in result.phase_geometry_scope
    assert "atan2(0,0)" in result.phase_geometry_scope
