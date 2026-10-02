"""Power, symmetry and finite locality do not select a primitive phase row.

These exact controls keep the original prism preparation and fresh nodal form
row. Beta is a declared comparison-storage scale, not a new clock or fitted
TNFR parameter. The constructed radius-two field is a counterexample to
constitutive uniqueness, not a phase law installed in the engine. No trajectory,
Poisson structure, domain invariance or sustained pattern is certified.
"""

import pytest

from tests.physics._internal_mode_fixture import (
    P,
    _exact_generator,
    _metric_differential,
    _nonrepeated_phase_geometry,
)
from tnfr.physics.support_transport import observe_support_transport


def _power_null_matrix(s, adjacency, covector):
    """A comparison matrix with a local edge factorization, not an operator."""
    squares = s.matrix_multiply_elementwise(covector, covector)
    return s.diag(*(adjacency * squares)) - (
        s.diag(*covector) * adjacency * s.diag(*covector)
    )


@pytest.fixture(scope="module")
def retained():
    s, graph, rows, phases = _nonrepeated_phase_geometry()
    adjacency = s.Matrix(6, 6, lambda i, j: int(j in rows[i]))
    laplacian = -s.Matrix(_exact_generator(observe_support_transport(graph)))
    metric, _, source, jacobian = _metric_differential(s, rows, phases)
    covector = (metric * source).applyfunc(s.simplify)
    form = s.ones(6, 1) / 2 + s.Matrix(P * 2) / 4
    return (
        s,
        rows,
        phases,
        adjacency,
        laplacian,
        metric,
        source,
        jacobian,
        covector,
        form,
    )


def test_reciprocal_edge_factor_is_positive_semidefinite_and_annihilates_power(
    retained,
):
    s, rows, _, _, _, _, _, _, _, _ = retained
    edges = tuple((i, j) for i, row in enumerate(rows) for j in row if i < j)
    weights = s.symbols(f"weight0:{len(edges)}", nonnegative=True)
    adjacency = s.zeros(6)
    for (i, j), weight in zip(edges, weights, strict=True):
        adjacency[i, j] = adjacency[j, i] = weight
    a = s.Matrix(s.symbols("a0:6", real=True))
    v = s.Matrix(s.symbols("v0:6", real=True))
    factor = _power_null_matrix(s, adjacency, a)
    edge_energy = sum(
        weight * (a[j] * v[i] - a[i] * v[j]) ** 2
        for (i, j), weight in zip(edges, weights, strict=True)
    )
    assert factor == factor.T
    assert (factor * a).applyfunc(s.expand) == s.zeros(6, 1)
    assert s.expand(v.dot(factor * v) - edge_energy) == 0
    assert edge_energy.is_nonnegative
    assert s.expand(a.dot(factor * v)) == 0
    # Each output needs neighboring a and v only. With a=-grad(V_phi) and
    # v=L*x, both already use one neighborhood, so the composed state field
    # has radius two. The positive matrix here is not a dynamical mobility:
    # its output is tangent to the phase-potential level set.


def test_remaining_direction_respects_phase_gauge_labels_and_form_offset(retained):
    s, rows, phases, adjacency, lap, _, _, _, a, x = retained
    direct = s.Matrix(
        [sum(s.sin(phases[j] - phases[i]) for j in row) for i, row in enumerate(rows)]
    )
    assert (a - direct).applyfunc(s.simplify) == s.zeros(6, 1)
    direction = _power_null_matrix(s, adjacency, a) * lap * x
    shift, offset = s.symbols("phase_shift form_offset", real=True)
    shifted = tuple(value + shift for value in phases)
    shifted_a = s.Matrix(
        [sum(s.sin(shifted[j] - shifted[i]) for j in row) for i, row in enumerate(rows)]
    )
    assert (shifted_a - a).applyfunc(s.simplify) == s.zeros(6, 1)
    assert lap * (x + offset * s.ones(6, 1)) == lap * x

    order = (4, 0, 5, 2, 1, 3)
    permutation = s.Matrix(6, 6, lambda i, j: int(order[i] == j))
    relabeled = _power_null_matrix(
        s, permutation * adjacency * permutation.T, permutation * a
    )
    relabeled_direction = (
        relabeled * (permutation * lap * permutation.T) * (permutation * x)
    )
    assert (relabeled_direction - permutation * direction).applyfunc(
        s.simplify
    ) == s.zeros(6, 1)
    # Simultaneous centered-form/phase reversal is also respected. Phase
    # reflection alone is not a symmetry of the unchanged signed form row.
    reflected = _power_null_matrix(s, adjacency, -a) * lap * (-x)
    assert (reflected + direction).applyfunc(s.simplify) == s.zeros(6, 1)


def test_same_lossless_exchange_has_distinct_form_and_mean_acceleration(retained):
    s, _, _, adjacency, lap, metric, source, jacobian, a, x = retained
    center = s.eye(6) - s.ones(6) / 6
    y = center * x
    assert y == s.Matrix(P * 2) / 4 and y.dot(y) == s.Rational(1, 4)
    direction = (_power_null_matrix(s, adjacency, a) * lap * x).applyfunc(s.simplify)
    assert s.simplify(a.dot(direction)) == 0
    assert s.simplify((center * direction).dot(center * direction)) == (
        s.Rational(131, 96) + 283 * s.sqrt(3) / 768
    )

    e, w, beta = s.symbols("e w beta", positive=True)
    # This lossless baseline uses the whole-field mean. Only its added null
    # direction is radius-two local; the separate local pair below has its
    # own generally nonzero residual away from the retained preparation.
    omega0 = (w / beta) * metric.inv() * y
    omega1 = omega0 + (w / beta) * direction
    for omega in (omega0, omega1):
        residual = w * y.dot(source) - beta * a.dot(omega)
        assert s.simplify(residual) == 0
    # Both rows have identical fresh p and identical instantaneous storage
    # loss -e*y^T*L*y. Their next form response differs in relative
    # directions; common phase rotation cannot account for this distinction.
    pressure = -e * lap * x + w * source
    acceleration0 = -e * lap * pressure + w * jacobian * omega0
    acceleration1 = -e * lap * pressure + w * jacobian * omega1
    difference = (acceleration1 - acceleration0).applyfunc(s.simplify)
    assert (
        s.simplify(
            sum(difference) / 6
            - (w**2 / beta) * (155 * s.sqrt(3) - 267) / (2688 * s.pi)
        )
        == 0
    )
    assert (
        s.simplify(
            2 * y.dot(difference)
            - (w**2 / beta) * (13 * s.sqrt(3) - 333) / (448 * s.pi)
        )
        == 0
    )
    assert s.simplify(jacobian * s.ones(6, 1)) == s.zeros(6, 1)
    assert difference != s.zeros(6, 1)
    # Choosing either row would be an extra constitutive assumption. Both
    # remain in the passive class; neither replenishes its integrated loss.


def test_entire_local_pair_shares_a_declared_residual_and_matches_retained_state(
    retained,
):
    s, _, _, adjacency, lap, metric, source, _, a, original = retained
    ones = s.ones(6, 1)
    center = s.eye(6) - s.ones(6) / 6
    x = s.Matrix(s.symbols("form0:6", real=True))
    y, local_form = center * x, lap * x
    factor = _power_null_matrix(s, adjacency, a)
    w, beta = s.symbols("w beta", positive=True)
    baseline = (w / beta) * metric.inv() * local_form
    addition = (w / beta) * factor * local_form
    target = w * source.dot(y - local_form)
    for omega in (baseline, baseline + addition):
        residual = w * y.dot(source) - beta * a.dot(omega)
        assert s.simplify(residual - target) == 0
    # The residual is specified from the state before selecting either row.
    # It is not generally zero; no passive-trajectory result is transferred
    # from the globally centered lossless family to these local fields.
    assert any(s.simplify(s.diff(target, coordinate)) != 0 for coordinate in x)
    assert lap * ones == s.zeros(6, 1)
    assert metric.inv() * lap * ones == s.zeros(6, 1)
    assert factor * lap * ones == s.zeros(6, 1)
    assert lap * original == center * original
    assert (
        baseline.subs(dict(zip(x, original, strict=True)))
        - ((w / beta) * metric.inv() * center * original)
    ).applyfunc(s.simplify) == s.zeros(6, 1)
    assert s.simplify(target.subs(dict(zip(x, original, strict=True)))) == 0
    # M=H^-1 uses one phase neighborhood, L*x one form neighborhood, and
    # B(a)*L*x at most two neighborhoods. No common mean enters either row.
    # Their common storage residual does involve the whole-field diagnostic.


def test_prescribed_power_leaves_four_prism_quotient_directions_and_zero_degeneracy(
    retained,
):
    s, rows, _, _, _, metric, source, _, a, x = retained
    ones = s.ones(6, 1)
    y = x - ones * sum(x) / 6
    assert a != s.zeros(6, 1) and s.simplify(a.dot(ones)) == 0
    assert a.T.rank() == 1
    constraints = a.T.col_join(ones.T)
    basis = constraints.nullspace()
    assert constraints.rank() == 2 and len(basis) == 4
    coefficients = s.symbols("free0:4", real=True)
    tangent = sum(
        (
            coefficient * vector
            for coefficient, vector in zip(coefficients, basis, strict=True)
        ),
        s.zeros(6, 1),
    )
    w, beta = s.symbols("w beta", positive=True)
    target, gauge = s.symbols("target_residual common_rotation", real=True)
    omega = (
        (w / beta) * metric.inv() * y
        - (target / beta) * a / a.dot(a)
        + tangent
        + gauge * ones
    )
    assert s.simplify(w * y.dot(source) - beta * a.dot(omega)) == target
    assert s.simplify(sum(tangent)) == 0
    # For any nonzero a, one scalar power condition leaves n-1 directions;
    # quotienting the common rotation leaves n-2. This is instantaneous
    # constitutive freedom, not an existence or invariant-domain theorem.
    _, _, zero_source, _ = _metric_differential(s, rows, (0,) * 6)
    assert zero_source == s.zeros(6, 1)
    arbitrary_row = s.Matrix(s.symbols("omega0:6", real=True))
    assert w * y.dot(zero_source) - beta * zero_source.dot(arbitrary_row) == 0
    # At a=0 the only attainable residual is zero, and that identity places
    # no instantaneous restriction on any of the six phase rates.
