"""Full-prism nonlinear storage for the conditional fresh-pressure model.

The repeated-triple restriction is removed, but support, unit capacity,
pressure weights, and common phase-alignment coefficient remain fixed.
The optional phase formula is still a supplied comparison law, not its
independent-pressure runtime. No trajectory or invariant domain is claimed.
"""

from fractions import Fraction as Q

import pytest

from tests.physics._internal_mode_fixture import _exact_generator, _graph
from tnfr.physics.forcing_realization import capture_non_epi_forcing
from tnfr.physics.phase_response import derive_phase_response
from tnfr.physics.support_transport import observe_support_transport


def _full_phase_geometry():
    """Exact nonrepeated unit phasors; their entire spread is below pi/2."""
    source = observe_support_transport(_graph())
    phasors = ((Q(3, 5), Q(4, 5)),) + ((Q(1), Q(0)),) * 5
    gram = tuple(
        tuple(left[0] * right[0] + left[1] * right[1] for right in phasors)
        for left in phasors
    )
    reference = derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=source.support_neighbors,
        receiver_sources=tuple((i,) for i in range(6)),
        phase_factor=0,
    )
    current = tuple(
        sum(
            (
                phasors[j][1] * phasors[i][0] - phasors[j][0] * phasors[i][1]
                for j in neighbors
            ),
            Q(0),
        )
        / len(neighbors)
        for i, neighbors in enumerate(source.support_neighbors)
    )
    assert sum(current) == 0
    assert reference.is_nonnegative
    return source, reference, current


def test_full_pressure_chain_rule_has_a_spatial_response_term_beyond_mean_work():
    s = pytest.importorskip("sympy")
    source, reference, current = _full_phase_geometry()
    laplacian = -s.Matrix(_exact_generator(source))
    response = s.Matrix(reference.mean_response)
    ones, projection = s.ones(6, 1), s.eye(6) - s.ones(6) / 6
    assert laplacian * ones == s.zeros(6, 1)
    assert response * ones == ones
    assert response != s.ones(6) / 6
    e, k, alignment = s.symbols("e k alignment", positive=True)
    pressure = s.Matrix(s.symbols("p0:6", real=True))
    force = s.Matrix(s.symbols("f0:6", real=True))
    force_at_mean = s.Symbol("force_at_mean", real=True)
    phase_rate = force + alignment * s.Matrix(current)
    centered_phase_rate = projection * phase_rate
    pressure_rate = -e * laplacian * pressure + k * (response - s.eye(6)) * phase_rate
    # This is the exact derivative of declared fresh p=-e*L*x+w*g(theta),
    # with Dg=(R-I)/pi and k=w/pi. The detached Gram does not authenticate
    # live phases or a pressure capture; it realizes this regular derivative.
    storage_gradient = force - force_at_mean * ones
    actual = storage_gradient.dot(pressure_rate) - k * alignment * s.Matrix(
        current
    ).dot(phase_rate)
    expected = (
        -e * force.dot(laplacian * pressure)
        - k * centered_phase_rate.dot(centered_phase_rate)
        + k * storage_gradient.dot(response * centered_phase_rate)
    )
    assert s.expand(actual - expected) == 0
    mean_rate = s.simplify(sum(pressure_rate) / 6)
    assert s.simplify(mean_rate - k * sum(response * centered_phase_rate) / 6) == 0
    # The storage is sum F(p_i)-6F(mean p)+(k*alignment/3)V(theta).
    # Its phase gradient is -k*alignment*J because the actual degree is 3.
    # Unlike the repeated sector, R*centered_phase_rate need not be common.
    assert projection * response * projection != s.zeros(6)


def test_fresh_intertriangle_mode_has_positive_cross_work_but_net_storage_loss():
    s = pytest.importorskip("sympy")
    amplitude, e, weight = Q(1, 16), Q(1, 2), Q(1, 4)
    graph = _graph(
        (0, 0, 0, 0),
        means=(Q(1, 2) - 3 * amplitude / (2 * e), Q(1, 2) + 3 * amplitude / (2 * e)),
    )
    graph.graph["DNFR_WEIGHTS"] = dict(epi=0.5, phase=0.25, vf=0.25, topo=0)
    captured = capture_non_epi_forcing(graph)
    direction = s.Matrix([1, 1, 1, -1, -1, -1])
    pressure = s.Matrix(captured.full_kernel_pressure)
    assert pressure == amplitude * direction
    assert captured.kernel_pressure_defect == (0,) * 6
    assert all(0 < value < 1 for value in captured.snapshot.epi)
    laplacian = -s.Matrix(_exact_generator(captured.snapshot))
    assert laplacian * direction == s.Rational(2, 3) * direction
    reference = derive_phase_response(
        cosine_gram=((Q(1),) * 6,) * 6,
        mean_neighbors=captured.snapshot.support_neighbors,
        receiver_sources=tuple((i,) for i in range(6)),
        phase_factor=0,
    )
    response = s.Matrix(reference.mean_response)
    assert response == s.eye(6) - laplacian
    alpha, beta = s.symbols("alpha beta", positive=True)
    scalar_force = alpha * s.sin(s.pi * amplitude) + beta * amplitude
    force = scalar_force * direction
    k = weight / s.pi
    # All primitive phases are equal, so J=0 and the actual conditional
    # phase velocity is f(p). This fresh pressure is captured, not fitted.
    cross_work = s.simplify(k * force.dot(response * force))
    assert s.simplify(cross_work - 2 * k * scalar_force**2) == 0
    assert cross_work > 0
    total = -e * force.dot(laplacian * pressure) - k * force.dot(force) + cross_work
    assert (
        s.simplify(total + 4 * e * amplitude * scalar_force + 4 * k * scalar_force**2)
        == 0
    )
    assert s.simplify(total) < 0
    # The missing inter-triangle mode changes the balance, but positive
    # cross work here does not imply net production or a sustained orbit.


def test_actual_full_response_admits_the_row_stochastic_norm_and_gap_bounds():
    s = pytest.importorskip("sympy")
    source, reference, _ = _full_phase_geometry()
    response = s.Matrix(reference.mean_response)
    vector = s.Matrix(s.symbols("u0:6", real=True))
    assert all(value >= 0 for value in response)
    assert all(sum(response[i, j] for j in range(6)) == 1 for i in range(6))
    column_sum = [sum(response[i, j] for i in range(6)) for j in range(6)]
    assert all(value <= 6 for value in column_sum)
    decomposition = sum((6 - column_sum[j]) * vector[j] ** 2 for j in range(6)) + sum(
        response[i, j] * response[i, other] * (vector[j] - vector[other]) ** 2
        for i in range(6)
        for j in range(6)
        for other in range(j + 1, 6)
    )
    image = response * vector
    assert s.expand(6 * vector.dot(vector) - image.dot(image) - decomposition) == 0
    # Jensen's displayed sum-of-squares identity holds for every nonnegative
    # row-stochastic R of size 6, not only this exact nonrepeated fixture.
    # Global phase spread<pi/2 suffices for that positivity. Edgewise U3
    # alone is not silently substituted for this stronger phase hypothesis.
    laplacian = -s.Matrix(_exact_generator(source))
    eigenvalue = s.Symbol("eigenvalue")
    assert laplacian == laplacian.T
    characteristic = s.factor(laplacian.charpoly(eigenvalue).as_expr())
    expected = (
        eigenvalue
        * (eigenvalue - s.Rational(2, 3))
        * (eigenvalue - 1) ** 2
        * (eigenvalue - s.Rational(5, 3)) ** 2
    )
    assert s.expand(characteristic - expected) == 0
    # Connected symmetry supplies p^T L p >= (2/3)||p-mean(p)||^2.
    # Monotonicity gives f(p)^T L p >= ell*p^T L p edge by edge.


def test_pressure_quarter_class_controls_all_modes_and_the_complete_mean():
    s = pytest.importorskip("sympy")
    e, weight = s.Rational(1, 2), s.Rational(1, 4)
    alpha, beta, gap = s.Rational(1, 2), s.Rational(3, 20), s.Rational(2, 3)
    k = weight / s.pi
    lower_slope, upper_slope = beta + alpha * s.pi / s.sqrt(2), beta + alpha * s.pi
    pressure = s.Symbol("pressure", real=True)
    assert s.calculus.util.function_range(
        s.cos(s.pi * pressure),
        pressure,
        s.Interval(-s.Rational(1, 4), s.Rational(1, 4)),
    ) == s.Interval(s.sqrt(2) / 2, 1)
    # Strong convexity gives storage >=ell*||h||^2/2+(k*g0/3)*V.
    # On the pressure interval, ||f(p)-f(q)1||<=upper_slope*||h||.
    cross = k * upper_slope * s.sqrt(6)
    damping = e * lower_slope * gap
    assert s.simplify(cross**2 / k - 6 * k * upper_slope**2) == 0
    assert s.simplify(4 * damping - 4 * lower_slope / 3) == 0
    # Exact coarse bounds suffice: 3<pi<16/5 and sqrt(2)<3/2 give
    # 6*k*upper_slope^2 <49/32 <23/15 <4*e*ell*lambda_2.
    assert 3 < s.pi < s.Rational(16, 5)
    assert s.sqrt(2) < s.Rational(3, 2)
    assert k < s.Rational(1, 12)
    assert upper_slope < s.Rational(7, 4)
    assert lower_slope > s.Rational(23, 20)
    assert 6 * s.Rational(1, 12) * s.Rational(7, 4) ** 2 == s.Rational(49, 32)
    assert s.Rational(49, 32) < s.Rational(23, 15)
    assert 735 < 736
    norm_h, norm_u = s.symbols("norm_h norm_u", nonnegative=True)
    loss = damping * norm_h**2 + k * norm_u**2 - cross * norm_h * norm_u
    completed = (
        damping * (norm_h - cross * norm_u / (2 * damping)) ** 2
        + (k - cross**2 / (4 * damping)) * norm_u**2
    )
    assert s.simplify(loss - completed) == 0
    # Hence storage strictly decreases off h=u=0 in the stated domain.
    # There h=0 gives u=g0*J; strict phase gaps and connected support make
    # J=0 imply phase consensus, then fresh p=-e*L*x has zero mean as well.
    # No invariance of this pressure/phase domain, binary64 stability,
    # global convergence, or actual optional-runtime closure is inferred.
