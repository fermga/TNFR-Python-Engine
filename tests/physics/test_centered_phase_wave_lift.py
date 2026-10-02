"""Local primitive-state lift of a separately chosen centered graph wave.

Canonical pressure supplies coordinates, not a conservative evolution law.
The wave, its time convention and its Hamiltonian remain additional premises.
Exact ideal phase derivatives below use the prepared surd geometry, not a
rounded Gram matrix passed into the rational production API. No trajectory,
global chart invariance, autonomous selector or new engine law is asserted.
"""

from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from tests.physics._internal_mode_fixture import (
    NODES,
    Q_MODE,
    P,
    _prepared_pressure_coordinates,
)
from tnfr.physics.structural_diffusion import structural_diffusion_operator


@pytest.fixture(scope="module")
def prepared():
    return _prepared_pressure_coordinates()


def test_prepared_pressure_coordinates_have_an_invertible_centered_phase_chart(
    prepared,
):
    s, graph, rows, phases, _, cosine, lap, project, scaled, inverse = prepared
    assert (
        nx.is_connected(graph) and not graph.is_directed() and not graph.is_multigraph()
    )
    assert all(len(row) == len(set(row)) == 3 for row in rows)
    # A global narrow chart ensures positive cosine response on every supported
    # entry. Per-edge U3 alone would not justify this general positivity claim.
    assert max(phases) - min(phases) == s.pi / 3 < s.pi / 2
    response = scaled + s.eye(6)
    assert all(response[i, j] > 0 for i, row in enumerate(rows) for j in row)
    assert (response * s.ones(6, 1)).applyfunc(s.expand) == s.ones(6, 1)
    left = s.Matrix(cosine)
    assert all(value > 0 for value in left)
    assert (left.T * scaled).applyfunc(s.simplify) == s.zeros(1, 6)
    assert scaled.rank() == (project * scaled).rank() == 5
    # If Dg*u=c*1, the positive left stationary vector forces c=0;
    # connectedness then leaves only common rotation. Fixing its mean gives
    # the displayed two-sided inverse, valid on all five centered directions.
    assert (project * scaled * inverse - project).applyfunc(s.simplify) == s.zeros(6)
    assert (inverse * project * scaled - project).applyfunc(s.simplify) == s.zeros(6)
    assert inverse * s.ones(6, 1) == s.zeros(6, 1)

    nodes, actual_lap = structural_diffusion_operator(graph)
    assert tuple(nodes) == NODES
    np.testing.assert_array_equal(actual_lap, np.asarray(lap, dtype=float))
    # The shared wave reader uses materialized binary64 normalization. The
    # exact algebra is reconstructed from its effective conductances instead.
    i, j = next((i, row[0]) for i, row in enumerate(rows))
    assert Fraction(float(actual_lap[i, j])) != Fraction(lap[i, j])
    assert lap * s.ones(6, 1) == s.zeros(6, 1)


def test_centered_wave_lifts_but_requires_a_nonzero_full_mean_acceleration(prepared):
    s, _, _, phases, center, cosine, lap, project, scaled, inverse = prepared
    e, w, amplitude = s.symbols("e w amplitude", positive=True)
    y = amplitude * s.Matrix(Q_MODE * 2)
    source = s.Matrix([(center - phase) / s.pi for phase in phases])
    assert source == s.Matrix(P * 2) / 6
    assert sum(source) == 0 and lap * y == y
    velocity = -e * lap * y + w * source
    assert sum(velocity) == 0  # mu_dot=0 at this point, not along the lift.
    required = -lap * y + e * lap * velocity
    phase_velocity = (s.pi * inverse * required / w).applyfunc(s.simplify)
    assert s.simplify(sum(phase_velocity)) == 0
    assert (w * project * scaled * phase_velocity / s.pi - required).applyfunc(
        s.simplify
    ) == s.zeros(6, 1)
    acceleration = (-e * lap * velocity + w * scaled * phase_velocity / s.pi).applyfunc(
        s.simplify
    )
    mean_acceleration = (1 + e**2) * amplitude * (s.sqrt(3) - 2) / (1 + s.sqrt(3))
    assert (acceleration + lap * y - mean_acceleration * s.ones(6, 1)).applyfunc(
        s.simplify
    ) == s.zeros(6, 1)
    assert mean_acceleration < 0
    # The full wave x_ddot=-L*x would require an impossible source tangent.
    # This obstruction holds for all six phase velocities, not only repeated
    # ones; adding a common phase rotation cannot change it.
    obstruction = (s.Matrix(cosine).T * required)[0]
    assert s.simplify(obstruction - 2 * (1 + e**2) * amplitude * (2 - s.sqrt(3))) == 0
    assert 2 - s.sqrt(3) > 0  # The remaining factors are positive by assumption.
    gauge_speed = s.symbols("gauge_speed", real=True)
    assert (scaled * (gauge_speed * s.ones(6, 1))).applyfunc(s.expand) == s.zeros(6, 1)
    actual_epi = s.ones(6, 1) / 2 + y.subs(amplitude, s.Rational(1, 16))
    assert all(0 < value < 1 for value in actual_epi)


def test_pressure_coordinate_pullback_conserves_the_declared_centered_wave_energy(
    prepared,
):
    s, _, _, phases, center, _, lap, project, scaled, inverse = prepared
    e, w = s.symbols("e w", positive=True)
    y = project * s.Matrix(s.symbols("x0:6", real=True))
    source = s.Matrix([(center - phase) / s.pi for phase in phases])
    velocity = (-e * lap * y + w * project * source).applyfunc(s.expand)
    phase_velocity = (s.pi * inverse * (-lap * y + e * lap * velocity) / w).applyfunc(
        s.expand
    )
    projected_source_derivative = project * scaled / s.pi
    velocity_rate = (
        -e * lap * velocity + w * projected_source_derivative * phase_velocity
    ).applyfunc(s.expand)
    assert (velocity_rate + lap * y).applyfunc(s.expand) == s.zeros(6, 1)
    # H_wave=(3/2)||v||^2+(1/2)y^T B y, B=3L: actual prism degree metric.
    # This is not the tetrad energy or the unforced EPI Lyapunov flow.
    energy_rate = 3 * velocity.dot(velocity_rate) + 3 * (lap * y).dot(velocity)
    assert s.expand(energy_rate) == 0
    assert lap == lap.T and all(value >= 0 for value in lap.eigenvals())

    # Pull back the standard wave two-form through (y,theta)->(y,v).
    # The -eL contribution cancels because L is symmetric. Rank ten leaves
    # both common coordinates outside this centered Hamiltonian chart.
    zero = s.zeros(6)
    jacobian = project.row_join(zero).col_join(
        (-e * lap).row_join(w * projected_source_derivative)
    )
    canonical_form = 3 * zero.row_join(s.eye(6)).col_join((-s.eye(6)).row_join(zero))
    pulled = (jacobian.T * canonical_form * jacobian).applyfunc(s.expand)
    expected = 3 * zero.row_join(w * projected_source_derivative).col_join(
        (-w * projected_source_derivative.T).row_join(zero)
    )
    assert (pulled - expected).applyfunc(s.expand) == s.zeros(12)
    assert pulled + pulled.T == s.zeros(12) and pulled.rank() == 10
    gradient = (3 * lap * y - 3 * e * lap * velocity).col_join(
        3 * w * projected_source_derivative.T * velocity
    )
    lifted_velocity = velocity.col_join(phase_velocity)
    assert (pulled * lifted_velocity + gradient).applyfunc(s.expand) == s.zeros(12, 1)
    # Closedness comes from an actual coordinate pullback on a regular
    # neighborhood; pointwise skewness alone would not prove that theorem.
    # Choosing this Hamiltonian and flow remains an additional premise.


def test_uniform_phase_chart_wave_requires_the_active_phase_feedback_row(prepared):
    s, _, rows, _, _, _, lap, project, _, _ = prepared
    e, k = s.symbols("e k", positive=True)
    uniform_response = s.Matrix(
        6, 6, lambda i, j: s.Rational(1, 3) if j in rows[i] else 0
    )
    assert uniform_response - s.eye(6) == -lap
    # Linearized canonical pressure is -L*(e*y+k*theta), k=w/pi. Requiring
    # y_ddot=-L*y then fixes theta_dot=(y-e*y_dot)/k modulo common rotation.
    upper = (-e * lap).row_join(-k * lap)
    lower = (project / k + e**2 * lap / k).row_join(e * lap)
    generator = upper.col_join(lower)
    assert (upper * generator).applyfunc(s.expand) == (-lap).row_join(s.zeros(6))
    internal = s.Matrix(P * 2)
    assert lap * internal == internal
    required_block = s.Matrix([[-e, -k], [(1 + e**2) / k, e]])
    lift = internal.row_join(s.zeros(6, 1)).col_join(s.zeros(6, 1).row_join(internal))
    assert (generator * lift - lift * required_block).applyfunc(s.expand) == s.zeros(
        12, 2
    )
    assert required_block.trace() == 0 and s.simplify(required_block.det()) == 1
    assert (required_block**2).applyfunc(s.simplify) == -s.eye(2)
    # The +e phase self-response cancels EPI damping in this selected wave.
    # It is the previously classified neutral boundary b=e, not a result
    # inferred from the existing configured dissipative phase-response row.
