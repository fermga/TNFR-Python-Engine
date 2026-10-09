"""Fixed-source tangent freedom at the prepared internal prism phase pattern.

The current angles are handled symbolically: their exact cosine Gram contains
surds and is not silently rounded into the rational production API. A separate
rational planar control reuses that API on the same support. Neither fixture
supplies a phase law, a runtime derivative, or a maintenance trajectory.
"""

from fractions import Fraction as Q

import networkx as nx

from tests.physics._internal_mode_fixture import _phase_support as _support
from tests.physics._internal_mode_fixture import (
    _prepared_phase_geometry as _prepared_geometry,
)
from tnfr.physics.phase_response import (
    derive_phase_response,
    observe_phase_source_geometry,
)


def test_prepared_surds_give_the_actual_regular_phasor_source_derivative():
    s, _, rows, phases, center, _, gram, response = _prepared_geometry()
    variables = s.symbols("theta0:6", real=True)
    substitution = dict(zip(variables, phases, strict=True))
    direct = s.zeros(6)
    for i, neighbors in enumerate(rows):
        real = sum(s.cos(variables[j]) for j in neighbors)
        imaginary = sum(s.sin(variables[j]) for j in neighbors)
        actual_real = s.simplify(real.subs(substitution))
        actual_imaginary = s.simplify(imaginary.subs(substitution))
        # Rotate the resultant by -pi/6: positive real and zero imaginary.
        assert s.simplify(
            actual_real * s.cos(center) + actual_imaginary * s.sin(center)
        ) == (1 + s.sqrt(3))
        assert (
            s.simplify(actual_imaginary * s.cos(center) - actual_real * s.sin(center))
            == 0
        )
        mean = s.atan2(imaginary, real)
        for j in range(6):
            direct[i, j] = s.simplify(s.diff(mean, variables[j]).subs(substitution))
        squared = sum(gram[j, k] for j in neighbors for k in neighbors)
        assert s.simplify(squared - (1 + s.sqrt(3)) ** 2) == 0
        for j in neighbors:
            gram_coefficient = sum(gram[j, k] for k in neighbors) / squared
            assert s.simplify(gram_coefficient - response[i, j]) == 0
    assert s.simplify(direct - response) == s.zeros(6)
    assert (
        tuple(s.simplify((center - phase) / s.pi) for phase in phases)
        == (s.Rational(1, 6), -s.Rational(1, 6), 0) * 2
    )
    assert (
        max(abs(phases[i] - phases[j]) for i, row in enumerate(rows) for j in row)
        == s.pi / 3
    )
    # All edges are strictly inside U3's pi/2 branch, and every source
    # resultant is nonzero; this is not a derivative through a branch cut.


def test_only_common_rotation_is_invisible_to_the_full_prepared_phase_source():
    s, graph, rows, _, _, cosine, _, response = _prepared_geometry()
    assert nx.is_connected(graph)
    for i, row in enumerate(rows):
        for j in range(6):
            assert response[i, j].is_positive if j in row else response[i, j] == 0
    assert s.simplify(response * s.ones(6, 1)) == s.ones(6, 1)
    reversible = s.diag(*cosine) * (s.eye(6) - response)
    assert s.simplify(reversible - reversible.T) == s.zeros(6)
    scaled_source_jacobian = response - s.eye(6)  # pi*Dg, not a phase evolution law.
    assert scaled_source_jacobian.rank() == 5
    assert scaled_source_jacobian.nullspace() == [s.ones(6, 1)]
    assert s.simplify(scaled_source_jacobian[:5, :5].det()) != 0
    internal_direction = s.Matrix((1, -1, 0) * 2)
    assert (
        s.simplify(scaled_source_jacobian * internal_direction) == -internal_direction
    )
    # Positivity persists in a regular neighborhood on this fixed connected
    # support. A differentiable path preserving the complete g there can only
    # rotate commonly. Maintaining a changing EPI pattern need not hold g fixed.


def test_rational_planar_control_reuses_source_owner_without_rounding_the_current_gram():
    _, rows = _support()
    vectors = ((Q(4, 5), Q(-3, 5)), (Q(4, 5), Q(3, 5)), (Q(1), Q(0))) * 2
    gram = tuple(
        tuple(
            sum((a * b for a, b in zip(left, right, strict=True)), Q(0))
            for right in vectors
        )
        for left in vectors
    )
    assert min(gram[i][j] for i, row in enumerate(rows) for j in row) == Q(7, 25) > 0
    reference = derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=rows,
        receiver_sources=tuple((i,) for i in range(6)),
        phase_factor=0,
    )
    expected = tuple(
        tuple(vectors[j][0] * Q(5, 13) if j in rows[i] else Q(0) for j in range(6))
        for i in range(6)
    )
    assert reference.mean_resultant_squared == (Q(169, 25),) * 6
    assert reference.mean_response == expected
    assert reference.jacobian == tuple(
        tuple(Q(i == j) for j in range(6)) for i in range(6)
    )
    geometry = observe_phase_source_geometry(reference)
    assert geometry.rank == 5 and geometry.tangent_dimension == 1
    assert geometry.only_common_rotation and geometry.mean_is_nonnegative
    assert geometry.scaled_source_jacobian != reference.jacobian
    # A zero operator-stage factor does not erase the underlying source
    # derivative. This rational-angle witness is not the original pi/3 capture.
