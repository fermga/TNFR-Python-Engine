"""Exact internal-mode pushforward on one supplied P2 x C3 scalar graph.

This is a projection of a declared pure-EPI model, not a new engine law or
an executed trajectory. Internal modal orientation is not canonical nodal
phase. Exact rational graph algebra and materialized binary64 matrices are
tested separately; the latter do not represent 1/3 exactly.
"""

from copy import deepcopy
from fractions import Fraction as Q
from math import sqrt

import networkx as nx
import pytest

from tests.physics._internal_mode_fixture import (
    GRAM,
    INDUCED,
    LIFT,
    NODES,
    PROJECTION,
    Q_MODE,
    P,
    _apply,
    _exact_generator,
    _graph,
    _inner,
)
from tnfr.mathematics.phasor_resultant import reduce_phasor_components
from tnfr.physics._cycle_algebra import dot
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.hybrid_operator_stability import _exact_matrix_product as product
from tnfr.physics.structural_diffusion import structural_diffusion_operator
from tnfr.physics.support_transport import observe_support_transport


def test_internal_basis_has_its_actual_metric_and_an_exact_left_inverse():
    graph = _graph()
    assert len(graph) == 6 and graph.number_of_edges() == 9
    assert tuple(dict(graph.degree()).values()) == (3,) * 6
    assert dot(P, Q_MODE) == sum(P) == sum(Q_MODE) == 0
    assert (dot(P, P), dot(Q_MODE, Q_MODE)) == GRAM
    assert product(PROJECTION, LIFT) == tuple(
        tuple(Q(i == j) for j in range(4)) for i in range(4)
    )
    # Every component of a fiberwise constant form is omitted, not invented.
    assert _apply(PROJECTION, (Q(2),) * 3 + (Q(-1),) * 3) == (0,) * 4
    assert product(tuple(zip(*LIFT, strict=True)), LIFT) == (
        (2, 0, 0, 0),
        (0, 6, 0, 0),
        (0, 0, 2, 0),
        (0, 0, 0, 6),
    )


def test_actual_graph_generator_intertwines_projection_and_lift_exactly():
    source = observe_support_transport(_graph())
    fine = _exact_generator(source)
    assert source.capacity == (1,) * 6
    assert product(fine, LIFT) == product(LIFT, INDUCED)
    # This identity holds for every fine scalar input, including omitted means.
    assert product(PROJECTION, fine) == product(INDUCED, PROJECTION)
    assert product(product(PROJECTION, fine), LIFT) == INDUCED
    assert _apply(fine, (Q(1),) * 6) == (0,) * 6
    # Both internal components decay at rate 1 in the macro-uniform mode.
    assert _apply(INDUCED, (Q(1), Q(0), Q(1), Q(0))) == (-1, 0, -1, 0)


def test_binary64_reader_matrix_is_not_the_exact_rational_generator():
    graph = _graph()
    nodes, laplacian = structural_diffusion_operator(graph)
    assert tuple(nodes) == NODES
    represented = tuple(tuple(-Q(float(value)) for value in row) for row in laplacian)
    exact = _exact_generator(observe_support_transport(graph))
    third = Q(float(1 / 3))
    assert third != Q(1, 3)
    assert represented != exact
    assert _apply(represented, (Q(1),) * 6) == (3 * third - 1,) * 6
    represented_induced = tuple(
        tuple(
            -(1 + third) if i == j else third if abs(i - j) == 2 else Q(0)
            for j in range(4)
        )
        for i in range(4)
    )
    assert product(PROJECTION, represented) == product(represented_induced, PROJECTION)
    assert represented_induced != INDUCED
    # No exact continuous or fresh-kernel claim follows from rounding this matrix.


def test_internal_restoring_energy_is_inherited_from_fine_dirichlet_energy():
    coefficients = (Q(1, 8), Q(1, 16), Q(-1, 8), Q(1, 32))
    source = observe_support_transport(_graph(coefficients))
    left, right = coefficients[:2], coefficients[2:]
    difference = tuple(a - b for a, b in zip(left, right, strict=True))
    inherited = (
        _inner(difference, difference)
        + 3 * _inner(left, left)
        + 3 * _inner(right, right)
    ) / 2
    assert source.dirichlet_energy == inherited > 0
    # Hessian of the restricted fine energy: diagonal 4G, off-diagonal -G.
    hessian = tuple(
        tuple(
            4 * GRAM[i % 2] if i == j else -GRAM[i % 2] if abs(i - j) == 2 else Q(0)
            for j in range(4)
        )
        for i in range(4)
    )
    metric = tuple(
        tuple(3 * GRAM[i % 2] if i == j else Q(0) for j in range(4)) for i in range(4)
    )
    assert product(metric, INDUCED) == tuple(tuple(-x for x in row) for row in hessian)
    assert dot(coefficients, _apply(hessian, coefficients)) / 2 == inherited
    rate = _apply(INDUCED, coefficients)
    energy_rate = dot(_apply(hessian, coefficients), rate)
    assert energy_rate == -dot(rate, _apply(metric, rate)) < 0


def test_weighted_polar_pushforward_has_the_derived_cosine_and_sine_terms():
    symbolic = pytest.importorskip("sympy")
    u0, v0, u1, v1 = symbolic.symbols("u0 v0 u1 v1", real=True)
    state = symbolic.Matrix([u0, v0, u1, v1])
    generator = symbolic.Matrix(INDUCED)
    rate = generator * state
    for a, b in ((0, 1), (1, 0)):
        u, v, other_u, other_v = (
            state[2 * a],
            state[2 * a + 1],
            state[2 * b],
            state[2 * b + 1],
        )
        du, dv = rate[2 * a], rate[2 * a + 1]
        squared = 2 * u**2 + 6 * v**2
        cross = 2 * u * other_u + 6 * v * other_v
        determinant = u * other_v - v * other_u
        # On squared>0, z=sqrt(2)u+i sqrt(6)v has these polar derivatives.
        # cross=r_a*r_b*cos(delta); sqrt(12)*det=r_a*r_b*sin(delta).
        squared_rate = symbolic.diff(squared, u) * du + symbolic.diff(squared, v) * dv
        angle_rate = symbolic.sqrt(12) * (u * dv - v * du) / squared
        assert symbolic.simplify(squared_rate - 2 * (cross - 4 * squared) / 3) == 0
        assert (
            symbolic.simplify(
                angle_rate - symbolic.sqrt(12) * determinant / (3 * squared)
            )
            == 0
        )
        # The 4 consists of the macro-edge loss 1 and internal relaxation 3.


def test_equal_amplitudes_do_not_erase_internal_orientation_response():
    aligned = (Q(1, 4), Q(0), Q(1, 4), Q(0))
    rotated = (Q(1, 4), Q(0), Q(1, 8), Q(1, 8))
    assert tuple(_inner(z[:2], z[:2]) for z in (aligned, rotated)) == (Q(1, 8),) * 2
    assert tuple(_inner(z[2:], z[2:]) for z in (aligned, rotated)) == (Q(1, 8),) * 2
    # The second internal angle is pi/3, since z=(sqrt(2)/8)(1+i sqrt(3)).
    assert _inner(rotated[:2], rotated[2:]) == Q(1, 16)
    rates = tuple(_apply(INDUCED, values) for values in (aligned, rotated))
    squared_rates = tuple(
        2 * _inner(values[:2], rate[:2])
        for values, rate in zip((aligned, rotated), rates, strict=True)
    )
    assert squared_rates == (Q(-1, 4), Q(-7, 24))
    # Angular speed divided by sqrt(3), retaining the nonunit basis metric.
    u, v = rotated[:2]
    du, dv = rates[1][:2]
    assert 2 * (u * dv - v * du) / _inner(rotated[:2], rotated[:2]) == Q(1, 6)


def test_fresh_fine_pressure_realizes_internal_decay_without_evolving_state():
    graph = _graph()
    before = deepcopy(graph)
    observation = capture_non_epi_forcing(graph)
    assert observation.epi_weight == 1
    assert observation.forcing == observation.kernel_pressure_defect == (0,) * 6
    assert observation.full_kernel_pressure == (Q(-1, 4), Q(1, 4), 0) * 2
    assert _apply(PROJECTION, observation.full_kernel_pressure) == (
        Q(-1, 4),
        0,
        Q(-1, 4),
        0,
    )
    assert dict(graph.nodes(data=True)) == dict(before.nodes(data=True))
    assert dict(graph.edges) == dict(before.edges)
    assert graph.graph == before.graph


def test_internal_decay_is_not_same_form_canonical_macro_pressure():
    # Uniform amplitude and modal phase erase every current macro-P2 channel.
    # Materialize the correctly normalized radius sqrt(2)/4 at both nodes.
    # The exact zero-gradient result holds for every common finite value;
    # this float is not asserted equal to the ideal irrational radius.
    macro = nx.path_graph(2)
    macro.graph["DNFR_WEIGHTS"] = {key: 0.25 for key in ("epi", "phase", "vf", "topo")}
    for node in macro:
        macro.nodes[node].update(EPI=sqrt(2) / 4, nu_f=1.0, theta=0.0, delta_nfr=0.0)
    captured = capture_non_epi_forcing(macro)
    assert (
        captured.snapshot.epi_gradient == captured.snapshot.capacity_gradient == (0, 0)
    )
    assert captured.snapshot.topology_gradient == captured.phase_gradient == (0, 0)
    assert captured.full_kernel_pressure == captured.forcing == (0, 0)
    coefficients = (Q(1, 4), Q(0), Q(1, 4), Q(0))
    induced = _apply(INDUCED, coefficients)
    assert tuple(induced[i] / coefficients[i] for i in (0, 2)) == (-1, -1)
    # This is a pressure-pushforward obstruction, not a failed runtime operator.


def test_zero_amplitude_has_no_polar_direction_even_with_a_nonzero_fine_rate():
    coefficients = (Q(0), Q(0), Q(1, 4), Q(0))
    source = observe_support_transport(_graph(coefficients))
    rate = _apply(PROJECTION, _apply(_exact_generator(source), source.epi))
    assert rate == (Q(1, 12), 0, Q(-1, 3), 0)
    assert _inner(coefficients[:2], coefficients[:2]) == 0
    zero_direction = reduce_phasor_components(((0.0, 0.0),))
    assert zero_direction.joint_zero and zero_direction.angle is None
    # Component evolution is defined; no angle or angular speed is assigned
    # at the singular polar origin, and no zero derivative is fabricated.
