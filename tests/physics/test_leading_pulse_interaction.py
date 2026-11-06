"""Eight constructed controls for a fixed-window pulse interaction expansion.

Amplitude jets are formal parameter derivatives, never time integrations.
The cubic Hamiltonian has its own stability statement; the fixed fast-time
remainder does not transfer that statement to the full nonlinear engine.
"""

from fractions import Fraction as Q
from itertools import product

import mpmath as mp
import pytest
import sympy as sp

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import cos, pi_interval
from tnfr.mathematics._rational_interval import sin as interval_sin
from tnfr.mathematics._rational_interval import sqrt
from tnfr.physics._sine_flow import _held_sine_neighbor_field

RADICAL = sp.sqrt(5)
C = (RADICAL - 1) / 4
TANGENT = sp.sqrt(5 + 2 * RADICAL)
FIFTH_ROOT = C + sp.I * sp.sqrt((5 + RADICAL) / 8)
EDGES = tuple(
    sorted(
        {
            tuple(sorted((2 * i + s, 2 * ((i + 1) % 5) + t)))
            for i in range(5)
            for s, t in product((0, 1), repeat=2)
        }
    )
)


def _clean(value):
    return sp.simplify(sp.expand(value))


def _zero(matrix):
    assert sp.Matrix(matrix).applyfunc(_clean) == sp.zeros(*matrix.shape)


def _inverse(matrix):
    rational = tuple(
        tuple(Q(int(value.p), int(value.q)) for value in row) for row in matrix.tolist()
    )
    return sp.Matrix(exact_matrix_inverse(rational))


def _contains(interval, expression):
    with mp.workdps(90):
        expected = mp.mpf(str(sp.N(expression, 85)))
        lower = mp.mpf(interval.lo.numerator) / interval.lo.denominator
        upper = mp.mpf(interval.hi.numerator) / interval.hi.denominator
        if interval.lo == interval.hi:
            assert abs(expected - lower) < mp.mpf("1e-80")
        else:
            assert lower <= expected <= upper
        assert interval.width < Q(1, 2**65)


def _fourier(index):
    # Use one sine radical throughout: independent trigonometric reductions
    # introduce equivalent nested radicals that simplify need not identify.
    return sp.Matrix([sp.expand(FIFTH_ROOT ** ((index * j) % 5)) for j in range(5)])


@pytest.fixture(scope="module")
def algebra():
    adjacency, difference = sp.zeros(5), sp.zeros(5)
    for i in range(5):
        adjacency[i, (i + 1) % 5] = adjacency[i, (i - 1) % 5] = 1
        difference[i, (i + 1) % 5], difference[i, (i - 1) % 5] = 1, -1
    laplacian, origin = 2 * sp.eye(5) - adjacency, sp.ones(5) / 5
    inverse = _inverse(laplacian + origin) - origin
    second = _inverse(laplacian**2 - 16 * sp.eye(5))
    n = (
        sp.eye(5) + adjacency - TANGENT**2 * difference * inverse * difference
    ).applyfunc(_clean)
    m = (
        adjacency - TANGENT**2 * difference * second * laplacian * difference
    ).applyfunc(_clean)
    a, b = sp.symbols("a0:5"), sp.symbols("b0:5")
    intensity, square = sp.Matrix([x * y for x, y in zip(a, b)]), sp.Matrix(
        [x**2 for x in a]
    )
    law = sp.Matrix(
        [a[i] * (n * intensity)[i] / 16 + b[i] * (m * square)[i] / 32 for i in range(5)]
    )
    return dict(
        L=laplacian,
        D=difference,
        Adj=adjacency,
        P=origin,
        R0=inverse,
        R2=second,
        N=n,
        M=m,
        a=a,
        b=b,
        law=law,
        aa=law.jacobian(a),
        ab=law.jacobian(b),
    )


def _symbols(algebra):
    if "spectra" in algebra:
        return algebra["spectra"]
    assert sp.re(FIFTH_ROOT).is_positive is True
    assert sp.im(FIFTH_ROOT).is_positive is True
    assert _clean(FIFTH_ROOT**5 - 1) == 0
    assert _clean(FIFTH_ROOT * sp.conjugate(FIFTH_ROOT) - 1) == 0
    n = (sp.S(3), 13 + 6 * RADICAL, sp.S(3), sp.S(3), 13 + 6 * RADICAL)
    m1, m2 = -(108 + 2 * RADICAL) / 41, -(358 + 168 * RADICAL) / 41
    m = (sp.S(2), m1, m2, m2, m1)
    for mode in range(5):
        vector = _fourier(mode)
        _zero(algebra["N"] * vector - n[mode] * vector)
        _zero(algebra["M"] * vector - m[mode] * vector)
    algebra["spectra"] = n, m
    return n, m


def test_both_full_rows_have_the_declared_weighted_expansion_and_original_clock(
    algebra,
):
    z = sp.Matrix((1, -2, 3, -1, 2)) / 8
    v = sp.Matrix((-2, 1, 1, 2, -1)) / 8
    h = sp.Matrix((2, -1, 0, 1, -2)) / 16
    p = sp.Matrix((-1, 2, -2, 0, 1)) / 16
    pi, curvature = pi_interval(), cos(2 * pi_interval() / 5)
    root = sqrt(curvature)
    nodes = []
    for i in range(5):
        for sign in (1, -1):
            nodes.append(Jet((0, root * Q(sign * v[i]), root * Q(p[i]), 0, 0)))
    for i in range(5):
        for sign in (1, -1):
            nodes.append(Jet((2 * pi * Q(i, 5), Q(sign * z[i]), Q(h[i]), 0, 0)))
    neighbors = tuple(
        tuple((j, 1, Q(1)) for j in range(10) if tuple(sorted((i, j))) in EDGES)
        for i in range(10)
    )
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    field = _held_sine_neighbor_field(model, (4,) * 10, (Q(1),) * 10, neighbors)
    original = field(tuple(nodes))
    # Transform every row by dt/dtau=pi/sqrt(c), not just the internal row.
    rates = tuple(value * (pi / root) for value in original)
    squared = sp.Matrix([value**2 for value in z])
    mean_forcing = -algebra["L"] * h / 2 - TANGENT * algebra["D"] * squared / 4
    cubic = sp.Matrix(
        [
            z[i] ** 3 / 6
            + z[i] * (algebra["Adj"] * squared)[i] / 4
            + TANGENT * z[i] * (algebra["D"] * h)[i] / 2
            for i in range(5)
        ]
    )
    for i in range(5):
        mean_form = (rates[2 * i] + rates[2 * i + 1]) / (2 * root)
        internal_form = (rates[2 * i] - rates[2 * i + 1]) / (2 * root)
        mean_phase = (rates[10 + 2 * i] + rates[11 + 2 * i]) / 2
        internal_phase = (rates[10 + 2 * i] - rates[11 + 2 * i]) / 2
        for degree in range(4):
            _contains(mean_form.coeffs[degree], mean_forcing[i] if degree == 2 else 0)
            _contains(
                mean_phase.coeffs[degree],
                (algebra["L"] * p)[i] / 2 if degree == 2 else 0,
            )
            _contains(internal_phase.coeffs[degree], v[i] if degree == 1 else 0)
            _contains(
                internal_form.coeffs[degree],
                -z[i] if degree == 1 else cubic[i] if degree == 3 else 0,
            )
        _contains(internal_form.coeffs[4], 0)  # odd internal parity
    # Independent elementary Taylor identity, before the fixed numerical probe.
    e, x, y, g, s, c = sp.symbols("epsilon x y h sin_alpha cos_alpha", real=True)
    shift = e * (y - x) + e**2 * g
    expansion = s * (1 - shift**2 / 2) + c * (shift - shift**3 / 6)
    assert (
        _clean(
            sp.expand(expansion).coeff(e, 3) + s * g * (y - x) + c * (y - x) ** 3 / 6
        )
        == 0
    )


def test_collective_response_retains_actual_denominators_and_homogeneous_initial_data(
    algebra,
):
    L, D, P = algebra["L"], algebra["D"], algebra["P"]
    _zero(L * algebra["R0"] - sp.eye(5) + P)
    _zero(D * P)
    _zero(D**2 + L * (4 * sp.eye(5) - L))
    lam = sp.symbols("lambda")
    assert sp.factor(L.charpoly(lam).as_expr()) == lam * (lam**2 - 5 * lam + 5) ** 2
    signal = sp.Matrix(sp.symbols("I0:5"))
    harmonic = sp.Matrix(sp.symbols("B0:5"))
    h0 = -TANGENT * algebra["R0"] * D * signal / 4
    h2 = -TANGENT * algebra["R2"] * L * D * harmonic / 4
    _zero(L**2 * h0 / 4 + TANGENT * L * D * signal / 16)
    _zero((L**2 / 4 - 4 * sp.eye(5)) * h2 + TANGENT * L * D * harmonic / 16)
    # h=h_h+H0+Re(H2*exp(2it)); arbitrary centered initial data need h_h.
    time = sp.symbols("tau", real=True)
    ell = sp.symbols("ell", positive=True)
    h_initial, v_initial, constant, cosine, sine = sp.symbols(
        "h_initial v_initial H0 Hc Hs", real=True
    )
    particular = constant + cosine * sp.cos(2 * time) + sine * sp.sin(2 * time)
    homogeneous = (h_initial - constant - cosine) * sp.cos(ell * time) + (
        v_initial - 2 * sine
    ) * sp.sin(ell * time) / ell
    assert _clean((particular + homogeneous).subs(time, 0) - h_initial) == 0
    assert (
        _clean(sp.diff(particular + homogeneous, time).subs(time, 0) - v_initial) == 0
    )
    assert _clean(sp.diff(homogeneous, time, 2) + ell**2 * homogeneous) == 0
    # This nonresonant zero-IC corrector retains an actual homogeneous mode;
    # it is not erased by averaging a noninteger frequency for one period.
    correction = sum(
        (sp.cos((1 + sign * ell) * time) - sp.cos(time))
        / (2 * (1 - (1 + sign * ell) ** 2))
        for sign in (1, -1)
    )
    assert (
        sp.trigsimp(
            sp.diff(correction, time, 2)
            + correction
            - sp.cos(time) * sp.cos(ell * time)
        )
        == 0
    )
    assert correction.subs(time, 0) == sp.diff(correction, time).subs(time, 0) == 0
    assert _clean(sp.diff(correction, time, 2).subs(time, 0)) == 1
    for frequency in ((5 - RADICAL) / 4, (5 + RADICAL) / 4):
        assert frequency != 0 and frequency != 2
        assert _clean(1 - (1 + frequency) ** 2) != 0
        assert _clean(1 - (1 - frequency) ** 2) != 0
    for sign in (-1, 1):
        frequency_box = (5 + sign * sqrt(5)) / 4
        assert interval_sin(2 * pi_interval() * frequency_box).hi < 0


def test_resonant_cubic_projection_includes_constant_and_second_harmonic_feedback(
    algebra,
):
    a, b, D, L = algebra["a"], algebra["b"], algebra["D"], algebra["L"]
    carrier = sp.symbols("carrier", nonzero=True)
    oscillation = sp.Matrix([(x * carrier + y / carrier) / 2 for x, y in zip(a, b)])
    intensity = sp.Matrix([x * y for x, y in zip(a, b)])
    square, conjugate_square = sp.Matrix([x**2 for x in a]), sp.Matrix(
        [y**2 for y in b]
    )
    h0 = -TANGENT * algebra["R0"] * D * intensity / 4
    h2 = -TANGENT * algebra["R2"] * L * D * square / 4
    h2bar = -TANGENT * algebra["R2"] * L * D * conjugate_square / 4
    mean_response = h0 + (h2 * carrier**2 + h2bar / carrier**2) / 2
    square_wave = sp.Matrix([value**2 for value in oscillation])
    for i in range(5):
        forcing = (
            oscillation[i] ** 3 / 6
            + oscillation[i] * (algebra["Adj"] * square_wave)[i] / 4
            + TANGENT * oscillation[i] * (D * mean_response)[i] / 2
        )
        coefficient = sp.expand(forcing).coeff(carrier, 1)
        assert _clean(coefficient - algebra["law"][i]) == 0
    # The first-harmonic solvability condition is i*A_T=coefficient.
    amplitude_derivative, phase = sp.symbols("A_T tau", real=True)
    assert sp.diff(
        amplitude_derivative * sp.exp(sp.I * phase), phase
    ) == sp.I * amplitude_derivative * sp.exp(sp.I * phase)


def test_cubic_hamiltonian_conserves_action_and_respects_pair_and_phase_symmetries(
    algebra,
):
    a, b, n, m = algebra["a"], algebra["b"], algebra["N"], algebra["M"]
    intensity = sp.Matrix([x * y for x, y in zip(a, b)])
    square, conjugate_square = sp.Matrix([x**2 for x in a]), sp.Matrix(
        [y**2 for y in b]
    )
    hamiltonian = (
        intensity.dot(n * intensity) / 32 + conjugate_square.dot(m * square) / 64
    )
    _zero(n - n.T)
    _zero(m - m.T)
    for i in range(5):
        assert _clean(sp.diff(hamiltonian, b[i]) - algebra["law"][i]) == 0
    conjugate = dict(zip(a + b, b + a))
    assert _clean(hamiltonian.subs(conjugate, simultaneous=True) - hamiltonian) == 0
    rhs = -sp.I * algebra["law"]
    rhs_bar = sp.I * algebra["law"].subs(conjugate, simultaneous=True)
    assert _clean(sum(b[i] * rhs[i] + a[i] * rhs_bar[i] for i in range(5))) == 0
    gauge = sp.symbols("gauge", nonzero=True)
    transformation = {
        **dict(zip(a, (gauge * x for x in a))),
        **dict(zip(b, (y / gauge for y in b))),
    }
    assert (
        _clean(hamiltonian.subs(transformation, simultaneous=True) - hamiltonian) == 0
    )
    for index in range(5):
        flipped = {a[index]: -a[index], b[index]: -b[index]}
        assert _clean(hamiltonian.subs(flipped, simultaneous=True) - hamiltonian) == 0
        for i in range(5):
            assert (
                _clean(
                    rhs[i].subs(flipped, simultaneous=True)
                    - (-1 if i == index else 1) * rhs[i]
                )
                == 0
            )


def test_common_pattern_recovers_period_correction_and_existing_full_pulse_splitting(
    algebra,
):
    n, m = _symbols(algebra)
    common = dict(zip(algebra["a"] + algebra["b"], (sp.S(1),) * 10))
    _zero(algebra["law"].subs(common) - sp.ones(5, 1) / 4)
    _zero(
        algebra["aa"].subs(common)
        - sp.eye(5) / 4
        - (algebra["N"] + algebra["M"] - sp.eye(5)) / 16
    )
    _zero(algebra["ab"].subs(common) - (algebra["N"] + sp.eye(5)) / 16)
    expected_p = ((95 + RADICAL) / 164, (55 + 21 * RADICAL) / 41)
    expected_q = (-(479 + 245 * RADICAL) / 164, (14 + 21 * RADICAL) / 41)
    expected_square = (
        (23365 + 11877 * RADICAL) / 53792,
        -(2975 + 1449 * RADICAL) / 6724,
    )
    swap = sp.Matrix(((0, 1), (1, 0)))
    for k in (1, 2):
        p, q = (2 - m[k]) / 8, -(2 * n[k] + m[k]) / 8
        assert _clean(p - expected_p[k - 1]) == 0
        assert _clean(q - expected_q[k - 1]) == 0
        block = sp.Matrix(((0, -p / 2), (q / 2, 0)))
        _zero(swap * block * swap - sp.Matrix(((0, q / 2), (-p / 2, 0))))
        assert _clean(-p * q / 4 - expected_square[k - 1]) == 0
    time = sp.symbols("tau", real=True)
    correction = 3 * sp.sin(time) / 16 + sp.sin(3 * time) / 48 - time * sp.cos(time) / 4
    assert (
        sp.trigsimp(
            sp.diff(correction, time, 2) + correction - 2 * sp.sin(time) ** 3 / 3
        )
        == 0
    )
    assert correction.subs(time, 0) == sp.diff(correction, time).subs(time, 0) == 0
    # The secular term is the fixed-window signature of frequency1-epsilon²/4.
    amplitude = sp.symbols("epsilon", real=True)
    assert (
        sp.diff(sp.sin((1 - amplitude**2 / 4) * time), amplitude, 2).subs(amplitude, 0)
        / 2
        == -time * sp.cos(time) / 4
    )


def test_predeclared_phase_advance_is_a_cubic_relative_equilibrium_with_full_jacobian(
    algebra,
):
    n, m = _symbols(algebra)
    carrier = _fourier(1)
    substituted = dict(
        zip(algebra["a"] + algebra["b"], tuple(carrier) + tuple(carrier.conjugate()))
    )
    frequency = (6 + m[2]) / 32
    assert _clean(frequency + 7 * (2 + 3 * RADICAL) / 164) == 0
    _zero(algebra["law"].subs(substituted) - frequency * carrier)
    assert _clean((carrier.conjugate().T * carrier)[0] - 5) == 0
    gauge = sp.diag(*carrier)
    inverse = sp.diag(*carrier.conjugate())
    actual_a = (
        inverse * (algebra["aa"].subs(substituted) - frequency * sp.eye(5)) * gauge
    )
    actual_b = inverse * algebra["ab"].subs(substituted) * inverse
    _zero(actual_b - algebra["N"] / 16 - m[2] * sp.eye(5) / 32)
    fourier = sp.Matrix.hstack(*(_fourier(k) for k in range(5)))
    _zero(fourier.conjugate().T * fourier - 5 * sp.eye(5))
    for k in range(5):
        vector = _fourier(k)
        diagonal = (2 * n[k] + 2 * m[(k + 2) % 5] - m[2]) / 32
        pairing = (2 * n[k] + m[2]) / 32
        _zero(actual_a * vector - diagonal * vector)
        _zero(actual_b * vector - pairing * vector)
    # The conjugate equation completes all ten real directions; it cannot be
    # replaced by the k block with its unequal partner diagonal discarded.


def test_exact_sideband_hessians_give_only_cubic_constrained_energy_stability(algebra):
    n, m = _symbols(algebra)
    frequency = (6 + m[2]) / 32
    expected_delta = (
        (136515 + 60007 * RADICAL) / 215168,
        (39775 + 19283 * RADICAL) / 215168,
    )
    expected_det = ((13935 + 6204 * RADICAL) / 26896, (8815 + 4797 * RADICAL) / 53792)
    expected_mean = (
        (958 + 490 * RADICAL) / (32 * 41),
        (578 + 166 * RADICAL) / (32 * 41),
    )
    spectral = sp.symbols("lambda")
    for k in (1, 2):
        plus = (2 * n[k] + 2 * m[(k + 2) % 5] - m[2]) / 32
        minus = (2 * n[k] + 2 * m[(2 - k) % 5] - m[2]) / 32
        coupling = (2 * n[k] + m[2]) / 32
        mean, drift = (plus + minus) / 2, (plus - minus) / 2
        discriminant = mean**2 - coupling**2
        hessian = sp.Matrix(((plus, coupling), (coupling, minus)))
        generator = -sp.I * sp.Matrix(((plus, coupling), (-coupling, -minus)))
        assert _clean(discriminant - expected_delta[k - 1]) == 0
        assert _clean(mean - expected_mean[k - 1]) == 0
        assert _clean(hessian.det() - expected_det[k - 1]) == 0
        assert expected_delta[k - 1].is_positive is True
        assert expected_mean[k - 1].is_positive is True
        assert expected_det[k - 1].is_positive is True
        assert (
            _clean(
                generator.charpoly(spectral).as_expr()
                - ((spectral + sp.I * drift) ** 2 + discriminant)
            )
            == 0
        )
        assert _clean(hessian.det() - discriminant + drift**2) == 0
    neutral = sp.Matrix(((frequency, frequency), (-frequency, -frequency)))
    assert neutral.rank() == 1 and neutral**2 == sp.zeros(2)
    assert neutral * sp.Matrix((sp.I, -sp.I)) == sp.zeros(2, 1)
    # The action-five constraint removes the real radial direction; global
    # phase is then quotiented. The remaining constrained Hessian is positive.
    radial, phase = sp.symbols("radial phase", real=True)
    assert (
        sp.diff(5 * ((1 + radial) ** 2 + phase**2), radial).subs({radial: 0, phase: 0})
        == 10
    )
    assert (
        sp.diff(5 * ((1 + radial) ** 2 + phase**2), phase).subs({radial: 0, phase: 0})
        == 0
    )


def test_timing_chart_and_fixed_window_remainder_do_not_supply_full_flow_stability():
    time = sp.symbols("tau", real=True)
    cosine, sine = sp.cos(time), sp.sin(time)
    angle_rate = (
        cosine * sp.diff(sp.sqrt(C) * sine, time)
        - sp.sqrt(C) * sine * sp.diff(cosine, time)
    ) / (cosine**2 + C * sine**2)
    assert sp.trigsimp(angle_rate - sp.sqrt(C) / (cosine**2 + C * sine**2)) == 0
    assert _clean(angle_rate.subs(time, 0) - sp.sqrt(C)) == 0
    assert _clean(angle_rate.subs(time, sp.pi / 2) - 1 / sp.sqrt(C)) == 0
    assert _clean(sp.sqrt(C) * angle_rate.subs(time, 0) / sp.pi - C / sp.pi) == 0
    assert (
        _clean(sp.sqrt(C) * angle_rate.subs(time, sp.pi / 2) / sp.pi - 1 / sp.pi) == 0
    )
    # Constant auxiliary phase offsets need not give equal instantaneous chi rates.
    denominator = C**2 + C * (1 - C**2)
    assert _clean(1 - denominator - (1 - C) * (1 - C**2)) == 0
    assert (1 - C).is_positive is True and (1 - C**2).is_positive is True
    # A zero-IC cubic corrector cancels internal residuals through degree3;
    # collective residuals start at4. This is a fixed-time comparison only.
    epsilon, z, z3, zl, zr, zl3, zr3, dh = sp.symbols(
        "epsilon z z3 zl zr zl3 zr3 Dh", real=True
    )
    internal = epsilon * z + epsilon**3 * z3
    left, right = epsilon * zl + epsilon**3 * zl3, epsilon * zr + epsilon**3 * zr3
    cubic = z**3 / 6 + z * (zl**2 + zr**2) / 4 + TANGENT * z * dh / 2
    exact_through_cubic = (
        -internal
        + internal**3 / 6
        + internal * (left**2 + right**2) / 4
        + TANGENT * internal * epsilon**2 * dh / 2
    )
    approximate_acceleration = -epsilon * z + epsilon**3 * (cubic - z3)
    residual = sp.expand(exact_through_cubic - approximate_acceleration)
    assert all(residual.coeff(epsilon, degree) == 0 for degree in range(5))
    # In original fine x/theta coordinates, every scaled-time Jacobian row
    # is bounded by2/sqrt(c), from degree4 and |cos|<=1.
    assert _clean(8 / (4 * sp.sqrt(C)) - 2 / sp.sqrt(C)) == 0
    lipschitz, budget = sp.symbols("L R", positive=True)
    bound = budget * epsilon**4 * (sp.exp(lipschitz * time) - 1) / lipschitz
    assert bound.subs(time, 0) == 0
    assert (
        sp.simplify(sp.diff(bound, time) - lipschitz * bound - budget * epsilon**4) == 0
    )
    # Neither this Gronwall bound nor the cubic minimum is a uniform
    # tau=O(epsilon^-2) comparison or an exact nonlinear traveling orbit.
