"""Exact cancellation and conservative integral bounds, without trajectories."""

from fractions import Fraction as Q

import pytest

from tnfr._exact_time import exp_unit_bounds
from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse
from tnfr.mathematics._rational_interval import cos, pi_interval
from tnfr.physics._cycle_algebra import laplacian_matrix


def test_full_metric_remainder_is_the_derivative_of_the_approximate_invariant():
    s = pytest.importorskip("sympy")
    shift = s.zeros(5)
    for i in range(5):
        shift[i, (i + 1) % 5] = 1
    B, J = 2 * s.eye(5) - shift - shift.T, shift - shift.T
    projection = s.eye(5) - s.ones(5) / 5
    u, v, remainder = (s.Matrix(s.symbols(f"{name}:5")) for name in ("u", "v", "r"))
    a, b, d, tangent = s.symbols("a b d tangent", positive=True)
    q = B * u
    departure = s.diag(*(tangent * J * v / 2 + remainder))
    phase_rate = d * (s.eye(5) + departure) * q
    du = -a * q - b * B * v
    dv = projection * phase_rate
    coefficient = d * tangent / (10 * a)
    actual = sum(phase_rate) / 5 + coefficient * (du.T * J * v + u.T * J * dv)[0]
    residual = d * q.dot(remainder) / 5 + coefficient * (u.T * J * d * departure * q)[0]
    assert s.expand(actual - residual) == 0
    # No decay approximation enters the cancellation. Setting the inverse
    # metric remainder to zero still retains its first-order departure.
    assert not s.expand(residual).equals(0)


def test_elementary_metric_storage_and_integral_budgets_are_strict():
    pi = pi_interval()
    c = cos(2 * pi / 5)
    assert Q(3, 10) < c.lo < c.hi < Q(1, 2)
    assert 3 < pi.lo < pi.hi < 4
    radius = Q(1, 100)
    tangent_bound = Q(10, 3)
    # |Jv/2|<=||v||, |Bv/2|<=2||v||. Taylor with bounded
    # cosine curvature and the sinc remainder encloses the denominator.
    denominator_floor = (1 - radius / Q(3, 10)) * (1 - (2 * radius) ** 2 / 6)
    denominator_quadratic = Q(1, 2) / Q(3, 10) + (1 + tangent_bound * radius) * Q(2, 3)
    assert denominator_floor > Q(9, 10)
    assert denominator_quadratic < 3
    linear = (tangent_bound + 3 * radius) / Q(9, 10)
    quadratic = tangent_bound * 4 + 3 / Q(9, 10)
    assert linear < 4 and quadratic < 17
    assert (Q(3, 10) - 2 * radius) / 2 > Q(1, 8)
    amplitude_limit = Q(1, 1024)
    assert 6 * amplitude_limit**2 < radius**2 / 8
    assert 7 * amplitude_limit < radius
    # Cauchy--Schwarz and the cross-storage inequality, with no empirical
    # decay rate, produce an integrable cubic error over the entire future.
    assert 32 * (2 + Q(5, 2) * 24) == 1984 < 2048
    assert 24 * 2048 < 224**2
    assert 14 * 7 * 224 + 6 * 7 * 24 == 22960 < 2**15


def test_inverse_metric_derivatives_supply_independent_majorants():
    s = pytest.importorskip("sympy")
    c, kappa, t = s.symbols("c kappa t", real=True)
    f = c / s.cos(kappa + t)
    assert s.simplify(s.diff(f, t) - c * s.sin(kappa + t) / s.cos(kappa + t) ** 2) == 0
    assert (
        s.simplify(
            s.diff(f, t, 2) / 2 - c / s.cos(kappa + t) ** 3 + c / (2 * s.cos(kappa + t))
        )
        == 0
    )
    # This derivative route is independent of the denominator expansion
    # above and checks the constants consumed by the static certificate.
    pi = pi_interval()
    cosine_upper, cosine_floor, radius = Q(31, 100), Q(299, 1000), Q(1, 100)
    assert cos(2 * pi / 5).hi < cosine_upper
    assert cos(2 * pi / 5 + radius).lo > cosine_floor
    assert cosine_upper / cosine_floor < Q(11, 10)
    assert cosine_upper / cosine_floor**2 < Q(7, 2)
    assert cosine_upper / cosine_floor**3 < 12
    assert 1 / (6 * (1 - (2 * radius) ** 2 / 6)) < Q(1, 5)
    assert Q(7, 2) + Q(11, 10) * 4 * radius / 5 < 4
    assert 12 + Q(11, 10) * 4 / 5 < 17


def test_cycle_pseudoinverse_cross_balance_and_spectrum_use_the_existing_support():
    s = pytest.importorskip("sympy")
    rows = laplacian_matrix(5)
    B = 2 * s.Matrix(rows)
    common = s.ones(5) / 5
    regular_inverse = exact_matrix_inverse(
        tuple(tuple(2 * value + Q(1, 5) for value in row) for row in rows)
    )
    inverse = s.Matrix(regular_inverse) - common
    assert B * inverse == inverse * B == s.eye(5) - common
    assert inverse * s.ones(5, 1) == s.zeros(5, 1)
    u, v = s.Matrix((1, -1, 0, 0, 0)), s.Matrix((0, 1, -1, 0, 0))
    assert abs((u.T * inverse * v)[0]) < 2
    positive = [value for value in B.eigenvals() if value != 0]
    assert all(bool(value > 1) and bool(value < 4) for value in positive)
    us, vs = s.symbols("u0:4"), s.symbols("v0:4")
    u, v = s.Matrix((*us, -sum(us))), s.Matrix((*vs, -sum(vs)))
    mobility = s.diag(*s.symbols("M0:5", positive=True))
    a, b = s.symbols("a b", positive=True)
    u_rate = -a * B * u - b * B * v
    v_rate = (s.eye(5) - common) * mobility * B * u
    derivative = (u_rate.T * inverse * v + u.T * inverse * v_rate)[0]
    independent = (
        -a * (u.T * v)[0] - b * (v.T * v)[0] + (u.T * inverse * mobility * B * u)[0]
    )
    assert s.expand(derivative - independent) == 0
    # The Young bound absorbs one half of the negative phase-square term.
    unorm, vnorm = s.symbols("unorm vnorm", real=True)
    assert s.expand(unorm**2 / 2 + vnorm**2 / 32 - unorm * vnorm / 4) == (
        s.expand((4 * unorm - vnorm) ** 2 / 32)
    )


def test_cross_corrected_storage_gives_a_finite_time_and_tail_without_sampling():
    # E lies between z^2/8 and 2*z^2; |F|<=z^2/2. The preceding
    # cross-balance identity supplies F'<=-v^2/32+5*u^2/2.
    correction = Q(1, 32)
    lower = Q(1, 8) - correction / 2
    upper = 2 + correction / 2
    form_decay = Q(1, 4) - correction * Q(5, 2)
    phase_decay = correction / 32
    assert lower == Q(7, 64) > 0
    assert form_decay > phase_decay > 0
    energy_time = upper / min(form_decay, phase_decay)
    assert energy_time == 2064
    # The preparation has E(0)<=6*epsilon^2 and |F(0)|<=2*epsilon^2.
    initial_coefficient = 6 + 2 * correction
    assert initial_coefficient / lower < 8**2
    # |m'|<=16/(3*sqrt(5))*||u||||v||<3/2*||z||^2.
    assert Q(16**2, 3**2 * 5) < 3**2
    tail_factor = Q(3, 2) * energy_time
    assert tail_factor == 3096 < 4096
    epsilon, blocks = Q(1, 2**20), 40
    # e>2 gives exp(-n)<2^-n for positive integer n; no exp rounding
    # or evolved response enters this deliberately conservative witness.
    radius = 8 * epsilon / 2**blocks
    tail = 4096 * radius**2
    assert 2 * energy_time * blocks == 165120
    assert radius == Q(1, 2**57) and tail == Q(1, 2**102)
    minimum_offset = Q(91, 160) * epsilon**2
    assert minimum_offset > Q(1, 2**41)
    # sin(delta)>delta/2, denominator<2, atan(t)>t/2 and pi<4
    # give an ideal port-rate magnitude >delta/64 on the admitted chart.
    minimum_rate = minimum_offset / 64
    assert minimum_rate > Q(1, 2**47)
    assert 4 * radius + tail < minimum_rate
    assert 5 * radius + tail < Q(1, 2)


def test_short_contact_growth_integral_and_prospective_signal_margin():
    s = pytest.importorskip("sympy")
    time, duration, rho = s.symbols("time duration rho", positive=True)
    # Growth controls displacement; only the form row needs a Lipschitz
    # bound to integrate its rate error. This is not an Euler error claim.
    integral = s.integrate(3 * rho * (s.exp(3 * time) - 1), (time, 0, duration))
    assert s.simplify(integral - rho * (s.exp(3 * duration) - 1 - 3 * duration)) == 0
    x, y = s.symbols("x y", positive=True)
    mobility = s.atan(y / x) / (2 * s.pi * y)
    assert s.limit(mobility, y, 0) == 1 / (2 * s.pi * x)
    # The positive integral representation has integrand <=1/x, so the
    # same mobility bound applies at zero source and on either signed side.
    integration_coordinate = s.symbols("a", real=True)
    assert (
        s.simplify(
            1 / x
            - x / (x**2 + integration_coordinate**2 * y**2)
            - integration_coordinate**2
            * y**2
            / (x * (x**2 + integration_coordinate**2 * y**2))
        )
        == 0
    )
    edge_floor, radius = Q(7, 25), Q(1, 100)
    pi = pi_interval()
    assert cos(2 * pi / 5).lo - 2 * radius > edge_floor
    assert 1 + 1 / (pi.lo * edge_floor) < 3
    h, state_radius_upper = Q(1, 4096), Q(1, 2**39)
    exponential = exp_unit_bounds(3 * h)
    assert exponential[1] < 2
    assert state_radius_upper * exponential[1] < radius
    error = state_radius_upper * (exponential[1] - 1 - 3 * h)
    assert 0 < error < 9 * state_radius_upper * h**2
    residual, tail = Q(1, 2**57), Q(1, 2**102)
    prior_rate_margin = Q(1, 2**47) - (2 * residual + tail)
    # Subtracting the still-moving isolated left ring costs its whole-time
    # background bound; the untouched right reference remains exactly zero.
    assert h * (prior_rate_margin - 4 * residual / 3) - error > 0


def test_isolated_cycle_mean_and_storage_taylor_terms_cancel_without_centering():
    s = pytest.importorskip("sympy")
    x, v = s.symbols("x0:5"), s.symbols("v0:5")
    e, w, capacity, kappa = s.symbols("e w capacity kappa", positive=True)
    q = tuple(2 * x[i] - x[(i + 1) % 5] - x[(i - 1) % 5] for i in range(5))
    phase_gap = tuple(v[(i + 1) % 5] - v[i] for i in range(5))
    # On the acute degree-two chart, the argument of the two-neighbor
    # resultant is their mean relative angle. No phase-row assumption is
    # used here; the common capacity and unchanged form row are essential.
    g = tuple((phase_gap[i] - phase_gap[(i - 1) % 5]) / (2 * s.pi) for i in range(5))
    mean_rate = sum(capacity * (-e * q[i] / 2 + w * g[i]) for i in range(5)) / 5
    assert s.expand(mean_rate) == 0
    # Nonzero component means do not affect edge storage or its first
    # variation at the twist. The phase Hessian is bounded above by the
    # edge-square form because cosine <=1 along every interpolation.
    parameter = s.symbols("t", real=True)
    potential = sum(1 - s.cos(kappa + parameter * gap) for gap in phase_gap)
    assert s.simplify(s.diff(potential, parameter).subs(parameter, 0)) == 0
    hessian = sum(gap**2 * s.cos(kappa + parameter * gap) for gap in phase_gap)
    assert s.simplify(s.diff(potential, parameter, 2) - hessian) == 0
    # Independently bounded geometry supplies a strict capture gap. The
    # same supnorm tube gives ten edge-square contributions, each <=2rho².
    pi = pi_interval()
    gap = 5 * cos(2 * pi / 5).lo - 4 * cos(3 * pi / 8).hi
    assert gap > Q(13, 1000)
    assert 10 * (2 * Q(1, 100)) ** 2 / 2 < gap


def test_receiver_mean_sign_survives_five_independent_continuous_remainders():
    # A coarse prospective proof independent of the evaluated certificate:
    # the initial receiver sum is the port rate; averaging five remainders
    # of magnitude B costs B, not B/5. Use rational prior lower bounds.
    h = Q(1, 4096)
    rho = Q(1, 2**40)
    residual, tail = Q(1, 2**57), Q(1, 2**102)
    # sin(kappa)<=1, pi>3, cos(kappa)>3/10 and cubic error
    # epsilon²/32 put the full initial offset below this sharper radius.
    assert (Q(20, 27) + Q(1, 32)) / 2**40 + residual + tail < rho
    minimum_offset = Q(91, 160) / 2**40
    minimum_rate = minimum_offset / 64 - 2 * residual - tail
    remainder = rho * (exp_unit_bounds(3 * h)[1] - 1 - 3 * h)
    assert h * minimum_rate / 5 - remainder > 0
