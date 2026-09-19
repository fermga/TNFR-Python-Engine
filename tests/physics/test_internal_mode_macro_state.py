"""Minimal ideal pressure state on the fixed unit triangular prism.

All ranks and identities are rational. Five EPI coordinates determine the
ideal pure-EPI pressure; six reconstruct EPI. This is not a compression of
independently stored pressure, primitive phase, capacities, or engine history,
and does not identify the rounding of a binary64 pressure implementation.
"""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tests.physics._internal_mode_fixture import (
    GRAM,
    INDUCED,
    LIFT,
    NODES,
    PROJECTION,
    _apply,
    _exact_generator,
    _graph,
    _inner,
)
from tnfr.mathematics.krylov import exact_rank
from tnfr.physics._cycle_algebra import dot
from tnfr.physics.epi_memory import observe_forced_support_realization
from tnfr.physics.forced_support import derive_forced_support_balance
from tnfr.physics.hybrid_operator_stability import _exact_matrix_product as product
from tnfr.physics.support_transport import observe_support_transport

MEANS = tuple(tuple(Q(a == b, 3) for a, _ in NODES) for b in range(2))
SIGN = (Q(1),) * 3 + (Q(-1),) * 3
C5 = (*PROJECTION, tuple(value / 3 for value in SIGN))
T5 = tuple((*row, sign / 2) for row, sign in zip(LIFT, SIGN, strict=True))
B5 = tuple((*row, Q(0)) for row in INDUCED) + ((Q(0),) * 4 + (Q(-2, 3),),)
C6 = ((Q(1, 6),) * 6, *C5)
T6 = tuple((Q(1), *row) for row in T5)
B6 = ((Q(0),) * 6,) + tuple((Q(0), *row) for row in B5)


def _identity(size):
    return tuple(tuple(Q(i == j) for j in range(size)) for i in range(size))


def _potential_matrix(graph):
    """Exact unit-distance inverse-square readout on this one supplied graph."""
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    return tuple(
        tuple(
            Q(0) if left == right else Q(1, distances[left][right] ** 2)
            for right in NODES
        )
        for left in NODES
    )


def test_five_coordinates_and_common_mean_form_a_complete_exact_chart():
    assert exact_rank(C5) == exact_rank(T5) == 5
    assert product(C5, T5) == _identity(5)
    assert _apply(C5, (Q(1),) * 6) == (0,) * 5
    assert product(C6, T6) == product(T6, C6) == _identity(6)
    source = observe_support_transport(
        _graph(
            (Q(1, 8), Q(1, 16), Q(-1, 16), Q(1, 32)),
            means=(Q(5, 8), Q(3, 8)),
        )
    )
    retained = _apply(C5, source.epi)
    assert retained == (Q(1, 8), Q(1, 16), Q(-1, 16), Q(1, 32), Q(1, 4))
    mean = sum(source.epi, Q(0)) / 6
    assert mean == Q(1, 2)
    assert tuple(mean + value for value in _apply(T5, retained)) == source.epi
    assert _apply(T6, _apply(C6, source.epi)) == source.epi


def test_five_state_closure_reconstructs_every_ideal_pressure_and_is_minimal():
    fine = _exact_generator(observe_support_transport(_graph()))
    assert product(C5, fine) == product(B5, C5)
    assert product(fine, T5) == product(T5, B5)
    assert product(product(T5, B5), C5) == fine
    assert product(C6, fine) == product(B6, C6)
    assert exact_rank(fine) == exact_rank(C5) == 5
    # If the full linear pressure map factors through another linear state C,
    # rank(fine) <= rank(C). Thus five is minimal for all-state ideal pressure.
    assert _apply(fine, (Q(1),) * 6) == (0,) * 6


def test_omitted_mean_contrast_changes_pressure_at_the_same_internal_state():
    first = observe_support_transport(_graph())
    second = observe_support_transport(_graph(means=(Q(5, 8), Q(3, 8))))
    assert _apply(PROJECTION, first.epi) == _apply(PROJECTION, second.epi)
    fine = _exact_generator(first)
    difference = tuple(b - a for a, b in zip(first.epi, second.epi, strict=True))
    pressure_change = _apply(fine, difference)
    assert pressure_change == (Q(-1, 12),) * 3 + (Q(1, 12),) * 3
    assert _apply(C5, difference) == (0, 0, 0, 0, Q(1, 4))
    assert _apply(_potential_matrix(_graph()), pressure_change) == (
        (Q(-1, 24),) * 3 + (Q(1, 24),) * 3
    )


def test_global_mean_is_invisible_to_ideal_pressure_but_not_to_full_epi():
    first = observe_support_transport(_graph())
    shifted = observe_support_transport(_graph(means=(Q(3, 4), Q(3, 4))))
    assert first.epi != shifted.epi
    assert _apply(C5, first.epi) == _apply(C5, shifted.epi)
    fine = _exact_generator(first)
    assert _apply(fine, first.epi) == _apply(fine, shifted.epi)
    assert first.dirichlet_energy == shifted.dirichlet_energy
    # This is exact-model translation invariance, not equality of every
    # freshly rounded kernel or an assertion about independently stored p.


def test_full_chart_inherits_metric_and_energy_without_an_added_potential():
    coefficients = (Q(1, 8), Q(1, 16), Q(-1, 16), Q(1, 32))
    source = observe_support_transport(_graph(coefficients, means=(Q(5, 8), Q(3, 8))))
    transpose = tuple(zip(*T6, strict=True))
    metric = product(transpose, tuple(tuple(3 * value for value in row) for row in T6))
    weights = (Q(18), Q(6), Q(18), Q(6), Q(18), Q(9, 2))
    assert metric == tuple(
        tuple(value if i == j else Q(0) for j in range(6))
        for i, value in enumerate(weights)
    )
    hessian = tuple(tuple(-value for value in row) for row in product(metric, B6))
    assert hessian == tuple(zip(*hessian, strict=True))
    assert hessian[-1][-1] == 3 and hessian[0] == (0,) * 6
    state = _apply(C6, source.epi)
    delta = tuple(
        a - b for a, b in zip(coefficients[:2], coefficients[2:], strict=True)
    )
    internal = (
        _inner(delta, delta)
        + 3 * _inner(coefficients[:2], coefficients[:2])
        + 3 * _inner(coefficients[2:], coefficients[2:])
    ) / 2
    energy = internal + Q(3, 2) * state[-1] ** 2
    assert dot(state, _apply(hessian, state)) / 2 == source.dirichlet_energy == energy
    rate = _apply(B6, state)
    assert dot(_apply(hessian, state), rate) == -dot(rate, _apply(metric, rate)) < 0
    assert tuple(hessian[i][i] for i in range(1, 5)) == tuple(4 * x for x in GRAM) * 2


def test_shared_realization_closes_fiber_means_with_zero_memory_but_loses_pressure():
    graph = _graph(means=(Q(5, 8), Q(3, 8)))
    source = observe_support_transport(graph)
    reference = derive_forced_support_balance(
        source, epi_weight=Q(1), forcing=(Q(0),) * 6
    )
    realization = observe_forced_support_realization(reference, (NODES[:3], NODES[3:]))
    closure = realization.closure
    assert closure.projection == MEANS
    assert closure.macro_generator == ((Q(1, 3), Q(-1, 3)), (Q(-1, 3), Q(1, 3)))
    assert closure.macro_metric_weights == (9, 9)
    assert closure.all_state_affine_closed and closure.lifted_affine_subspace_invariant
    assert closure.hidden_to_macro == ((0,) * 6,) * 2
    assert closure.macro_to_hidden == ((0, 0),) * 6
    assert closure.instantaneous_kernel == closure.coupling_gram == ((0, 0),) * 2
    assert realization.dimension == 2 and realization.rank_progression == (2, 2)
    assert realization.extra_coordinates == 0 and realization.reduced_source == (0, 0)
    assert realization.projected_state == (Q(5, 8), Q(3, 8))
    assert realization.projected_rate == (Q(-1, 12), Q(1, 12))
    internal = _apply(LIFT, (Q(1, 8), Q(0), Q(0), Q(0)))
    assert _apply(MEANS, internal) == (0, 0)
    assert any(_apply(_exact_generator(source), internal))
    # Minimality in the existing owner retains its declared means, not every
    # micro pressure or field. Zero memory does not restore omitted outputs.


def test_inverse_square_potential_keeps_rank_five_and_has_an_exact_mean_quotient():
    graph = _graph()
    potential = _potential_matrix(graph)
    fine = _exact_generator(observe_support_transport(graph))
    assert exact_rank(potential) == 6
    assert exact_rank(product(potential, fine)) == 5
    averaged = ((Q(2), Q(3, 2)), (Q(3, 2), Q(2)))
    assert product(MEANS, potential) == product(averaged, MEANS)
    assert product(product(potential, product(T5, B5)), C5) == product(potential, fine)
    # Internal sources supply the positive self-block 2; cross-block sources
    # contribute 1 + 1/4 + 1/4. These terms survive exact field aggregation.


def test_no_positive_macro_edge_length_restores_same_scalar_potential():
    symbolic = pytest.importorskip("sympy")
    length = symbolic.Symbol("length", positive=True)
    contrast = symbolic.Symbol("contrast", nonzero=True, real=True)
    inherited_pressure = symbolic.Matrix([-contrast / 3, contrast / 3])
    mean_kernel = symbolic.Matrix(
        [[2, symbolic.Rational(3, 2)], [symbolic.Rational(3, 2), 2]]
    )
    scalar_kernel = symbolic.Matrix([[0, 1 / length**2], [1 / length**2, 0]])
    inherited_potential = mean_kernel * inherited_pressure
    scalar_potential = scalar_kernel * inherited_pressure
    assert inherited_potential == symbolic.Matrix([-contrast / 6, contrast / 6])
    ratio = symbolic.simplify(scalar_potential[0] / inherited_potential[0])
    assert ratio == -2 / length**2 and ratio.is_negative
    # Even supplying the correct inherited mean pressure to the scalar P2
    # readout reverses this nonzero contrast for every positive edge length.
    # No rescaling by positive mobility/length repairs the lost internal term.
