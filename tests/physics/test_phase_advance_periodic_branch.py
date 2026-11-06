"""Seven finite controls for the analytic periodic-branch construction.

Formal amplitude jets are not integrations. These controls check identities
consumed by the implicit-function proof; they do not replace its functional
analysis or certify a numerical amplitude radius or transverse stability.
"""

from fractions import Fraction as Q
from itertools import product

import mpmath as mp
import pytest
import sympy as sp

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.physics._sine_flow import _held_sine_neighbor_field

R = sp.sqrt(5)
C = (R - 1) / 4
S = sp.sqrt((5 + R) / 8)
ROOT = C + sp.I * S
KAPPA = -(14 + 21 * R) / 164
HARMONIC = (95 + 40 * R) / 164
THIRD = (124 + 63 * R) / 1968
EDGES = tuple(
    sorted(
        {
            tuple(sorted((2 * i + a, 2 * ((i + 1) % 5) + b)))
            for i in range(5)
            for a, b in product((0, 1), repeat=2)
        }
    )
)


def _clean(value):
    return sp.simplify(sp.expand(value))


def _zero(matrix):
    assert sp.Matrix(matrix).applyfunc(_clean) == sp.zeros(*matrix.shape)


def _root(power):
    # A single sine radical avoids inequivalent symbolic presentations of zero.
    return sp.expand(ROOT ** (power % 5))


def _harmonic(variable, order, pair=0, sine=False):
    forward = variable**order * _root(order * pair)
    backward = variable ** (-order) * _root(-order * pair)
    return (forward - backward) / (2 * sp.I) if sine else (forward + backward) / 2


def _derivative(expression, variable):
    return sp.I * variable * sp.diff(expression, variable)


def _contains(interval, expression):
    # High-precision cross-check of a symbolic value against an outward box;
    # exact identities elsewhere, not this comparison, establish the proof.
    with mp.workdps(90):
        expected = mp.mpf(str(sp.N(expression, 85)))
        lower = mp.mpf(interval.lo.numerator) / interval.lo.denominator
        upper = mp.mpf(interval.hi.numerator) / interval.hi.denominator
        if interval.lo == interval.hi:
            assert abs(expected - lower) < mp.mpf("1e-80")
        else:
            assert lower <= expected <= upper
        assert interval.width < Q(1, 2**65)


def _field():
    neighbors = tuple(
        tuple((j, 1, Q(1)) for j in range(10) if tuple(sorted((i, j))) in EDGES)
        for i in range(10)
    )
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    return _held_sine_neighbor_field(model, (4,) * 10, (Q(1),) * 10, neighbors)


def _symbolic_field(state):
    forms, phases = state[:10], state[10:]
    form_rates, phase_rates = [], []
    for i in range(10):
        adjacent = [j for j in range(10) if tuple(sorted((i, j))) in EDGES]
        form_rates.append(sum(sp.sin(phases[j] - phases[i]) for j in adjacent) / 4)
        phase_rates.append(sum(forms[i] - forms[j] for j in adjacent) / 4)
    return tuple(form_rates + phase_rates)  # pi times the original-time field


@pytest.fixture(scope="module")
def matrices():
    adjacency = sp.zeros(5)
    for i in range(5):
        adjacency[i, (i - 1) % 5] = adjacency[i, (i + 1) % 5] = 1
    laplacian, origin = 2 * sp.eye(5) - adjacency, sp.ones(5) / 5
    rational = tuple(
        tuple(Q(int(x.p), int(x.q)) for x in row)
        for row in (laplacian + origin).tolist()
    )
    inverse = sp.Matrix(exact_matrix_inverse(rational)) - origin
    return laplacian, inverse, origin


@pytest.fixture(scope="module")
def profile_jets():
    """One formal amplitude preparation, including its declared frequency."""
    pi = pi_interval()
    radical = sqrt(I(5))
    curvature = (radical - 1) / 4
    root = sqrt(curvature)
    kappa = -(14 + 21 * radical) / 164
    harmonic = (95 + 40 * radical) / 164
    third = (124 + 63 * radical) / 1968
    eigenvalue = (5 + radical) / 2
    frequency = Jet((1, 0, -kappa, 0, 0))
    forms, phases, phase_derivatives = [], [], []
    for i in range(5):
        angle = 2 * pi * Q(i, 5)
        d = Jet((0, cos(angle), 0, third * cos(3 * angle), 0))
        ds = Jet((0, -sin(angle), 0, -3 * third * sin(3 * angle), 0))
        h = Jet((0, 0, -harmonic * sin(2 * angle), 0, 0))
        hs = Jet((0, 0, -2 * harmonic * cos(2 * angle), 0, 0))
        internal_form = frequency * ds * root
        mean_form = frequency * hs * (2 * root / eigenvalue)
        for sign in (1, -1):
            forms.append(mean_form + sign * internal_form)
            phases.append(h + angle + sign * d)
            phase_derivatives.append(hs + sign * ds)
    original = _field()(tuple(forms + phases))
    # s=Omega*sqrt(c)*t/pi: every original row receives this same factor.
    rates = tuple(value * (pi / root) / frequency for value in original)
    return rates, tuple(phase_derivatives)


def test_complete_field_respects_cycle_member_exchange_and_twist_reversal():
    state = tuple(sp.Rational((7 * i) % 19 - 9, 16) for i in range(20))
    original = _symbolic_field(state)
    cycle = tuple(2 * ((i + 1) % 5) + sign for i in range(5) for sign in (0, 1))
    exchange = tuple(i ^ 1 for i in range(10))
    reverse = tuple(2 * ((-i) % 5) + (1 - sign) for i in range(5) for sign in (0, 1))
    for permutation in (cycle, exchange, reverse):
        assert {
            tuple(sorted((permutation[i], permutation[j]))) for i, j in EDGES
        } == set(EDGES)
    for permutation in (cycle, exchange):
        offset = -2 * sp.pi / 5 if permutation == cycle else 0
        transformed = tuple(state[i] for i in permutation) + tuple(
            state[10 + i] + offset for i in permutation
        )
        expected = tuple(original[i] for i in permutation) + tuple(
            original[10 + i] for i in permutation
        )
        _zero(sp.Matrix(_symbolic_field(transformed)) - sp.Matrix(expected))
    # Reflection, primitive-phase inversion and member exchange preserve the
    # positive twist; this is a reversor, not a commuting state symmetry.
    transformed = tuple(state[i] for i in reverse) + tuple(
        -state[10 + i] for i in reverse
    )
    expected = tuple(-original[i] for i in reverse) + tuple(
        original[10 + i] for i in reverse
    )
    _zero(sp.Matrix(_symbolic_field(transformed)) - sp.Matrix(expected))
    for i in range(5):
        assert _clean(_root(i + 1) - ROOT * _root(i)) == 0
        assert _clean(sp.conjugate(_root(-i)) - _root(i)) == 0
    # The actual shared owner independently consumes the same constructed state.
    actual = _field()(tuple(Q(value) for value in state))
    for enclosure, symbolic in zip(actual, original):
        _contains(enclosure, symbolic / sp.pi)


def test_profile_filters_leave_one_kernel_and_uniform_inverse_bounds(matrices):
    laplacian, inverse, origin = matrices
    _zero(laplacian * inverse - sp.eye(5) + origin)
    _zero(inverse * sp.ones(5, 1))
    assert _clean(ROOT**5 - 1) == 0
    assert _clean(ROOT * sp.conjugate(ROOT) - 1) == 0
    assert C.is_positive is True and S.is_positive is True
    ell = (sp.S(0), (5 - R) / 4, (5 + R) / 4, (5 + R) / 4, (5 - R) / 4)
    for n in range(11):
        vector = sp.Matrix([_root(n * i) for i in range(5)])
        _zero(laplacian * vector - 2 * ell[n % 5] * vector)
        assert _clean(sum(vector) - (5 if n % 5 == 0 else 0)) == 0
        assert sp.cos(n * sp.pi) == (-1) ** n
        if n % 2 == 1:
            assert (1 - n**2 == 0) is (n == 1)
        if n > 0 and n % 2 == 0 and n % 5:
            assert (n**2 - ell[n % 5] ** 2).is_positive is True
    # These identities, with t>=0, bound every remaining harmonic, not just
    # the finite examples above: n^2=9+t internally and n^2=4+t collectively.
    gap = (17 - 5 * R) / 8
    assert gap.is_positive is True
    assert _clean(4 - ell[2] ** 2 - gap) == 0
    t = sp.symbols("t", nonnegative=True)
    assert _clean(sp.Rational(5, 4) * (8 + t) - (10 + t) - t / 4) == 0
    assert _clean(5 * (gap + t) / gap - (5 + t) - t * (5 / gap - 1)) == 0
    assert (5 / gap - 1).is_positive is True


def test_amplitude_projection_and_scalar_transversality_have_no_hidden_kernel():
    z = sp.symbols("z", nonzero=True)
    epsilon, omega, b3, b5 = sp.symbols("epsilon omega b3 b5", real=True)
    d = epsilon * _harmonic(z, 1) + b3 * _harmonic(z, 3) + b5 * _harmonic(z, 5)
    residual = (1 + omega) * _derivative(_derivative(d, z), z) + d
    projection = sp.expand(residual).coeff(z, 1) + sp.expand(residual).coeff(z, -1)
    assert _clean(projection + omega * epsilon) == 0
    divided = sp.cancel(projection / epsilon)
    assert divided == -omega and sp.diff(divided, omega) == -1
    amplitude = sp.expand(d).coeff(z, 1) + sp.expand(d).coeff(z, -1)
    assert amplitude == epsilon
    complement = sp.expand(residual - projection * _harmonic(z, 1))
    assert complement.coeff(z, 1) == complement.coeff(z, -1) == 0
    for n, coefficient in ((3, b3), (5, b5)):
        equation = 2 * complement.coeff(z, n)
        assert _clean(sp.diff(equation, coefficient).subs(omega, 0) - (1 - n**2)) == 0
    # The nonlinear range solution is supplied by the analytic proof. Here
    # the exact removable scalar derivative and its normalization are checked.


def test_collective_second_harmonic_matches_the_complete_field(profile_jets):
    eigenvalue = (5 + R) / 2
    gap = (17 - 5 * R) / 8
    tangent = S / C
    assert _clean(HARMONIC * gap - tangent * eigenvalue * (2 * S * C) / 8) == 0
    assert _clean(4 * HARMONIC / eigenvalue - (55 + 21 * R) / 82) == 0
    rates, _ = profile_jets
    for i in range(5):
        mean = (rates[2 * i] + rates[2 * i + 1]) / 2
        expected = sp.sqrt(C) * 8 * HARMONIC / eigenvalue * sp.im(_root(2 * i))
        for degree, value in ((0, 0), (1, 0), (2, expected), (3, 0)):
            _contains(mean.coeffs[degree], value)


def test_cubic_solvability_gives_the_frequency_and_periodic_corrector(profile_jets):
    z = sp.symbols("z", nonzero=True)
    epsilon = sp.symbols("epsilon", real=True)
    fundamental = _harmonic(z, 1)
    neighbor_square = _harmonic(z, 1, 1) ** 2 + _harmonic(z, 1, -1) ** 2
    mean_difference = -HARMONIC * (
        _harmonic(z, 2, 1, sine=True) - _harmonic(z, 2, -1, sine=True)
    )
    force = sp.expand(
        fundamental**3 / 6
        + fundamental * neighbor_square / 4
        + (S / C) * fundamental * mean_difference / 2
    )
    assert _clean(2 * force.coeff(z, 1) - 2 * KAPPA) == 0
    assert _clean(2 * force.coeff(z, 3) - 2 * KAPPA + sp.Rational(1, 3)) == 0
    assert _clean(THIRD - sp.Rational(1, 24) + KAPPA / 4) == 0
    d = epsilon * fundamental + epsilon**3 * THIRD * _harmonic(z, 3)
    residual = (
        (1 - 2 * KAPPA * epsilon**2) * _derivative(_derivative(d, z), z)
        + d
        - epsilon**3 * force
    )
    assert _clean(sp.expand(residual).coeff(epsilon, 3)) == 0
    assert KAPPA.is_negative is True
    rates, _ = profile_jets
    for i in range(5):
        internal = (rates[2 * i] - rates[2 * i + 1]) / 2
        for degree, expected in (
            (0, 0),
            (1, -sp.sqrt(C) * sp.re(_root(i))),
            (2, 0),
            (
                3,
                sp.sqrt(C)
                * (KAPPA * sp.re(_root(i)) - 9 * THIRD * sp.re(_root(3 * i))),
            ),
        ):
            _contains(internal.coeffs[degree], expected)


def test_full_phase_reconstruction_keeps_clock_and_member_return(
    matrices, profile_jets
):
    laplacian, inverse, _ = matrices
    z = sp.symbols("z", nonzero=True)
    frequency, epsilon = sp.symbols("Omega epsilon", positive=True)
    d = sp.Matrix(
        [
            epsilon * _harmonic(z, 1, i) + epsilon**3 * THIRD * _harmonic(z, 3, i)
            for i in range(5)
        ]
    )
    h = sp.Matrix(
        [-(epsilon**2) * HARMONIC * _harmonic(z, 2, i, sine=True) for i in range(5)]
    )
    ds = d.applyfunc(lambda x: _derivative(x, z))
    hs = h.applyfunc(lambda x: _derivative(x, z))
    u = sp.sqrt(C) * frequency * ds
    forms = 2 * sp.sqrt(C) * frequency * inverse * hs
    _zero(laplacian * forms / (2 * sp.pi) - sp.sqrt(C) * frequency * hs / sp.pi)
    _zero(u / sp.pi - sp.sqrt(C) * frequency * ds / sp.pi)
    # The exact neighbor current is an oriented edge divergence. Its missing
    # constant mode must vanish to recover the mean form row after elimination.
    currents = sp.symbols("j0:5", real=True)
    divergence = sp.Matrix([currents[i] - currents[(i - 1) % 5] for i in range(5)])
    assert sum(divergence) == 0
    mean_second = laplacian * divergence / (4 * C * frequency**2)
    recovered = 2 * frequency**2 * inverse * mean_second
    _zero(recovered - divergence / (2 * C))
    _zero(C * recovered / sp.pi - divergence / (2 * sp.pi))
    _zero(d.subs(z, -z) + d)
    _zero(u.subs(z, -z) + u)
    _zero(h.subs(z, -z) - h)
    _zero(forms.subs(z, -z) - forms)
    assert sp.expand(d[0]).coeff(z, 1) == epsilon / 2
    # This nonzero first harmonic forces minimum labeled period 2*pi. The
    # half shift returns only after the simultaneous within-pair exchange.
    assert (
        _clean(
            2 * sp.expand(d[0] ** 2).coeff(z, 2) / epsilon**2
            - 1 / sp.S(2)
            - epsilon**2 * THIRD
        )
        == 0
    )
    period = 2 * sp.pi**2 / (sp.sqrt(C) * frequency)
    assert _clean(period * frequency * sp.sqrt(C) / sp.pi - 2 * sp.pi) == 0
    rates, expected_phase = profile_jets
    for actual, expected in zip(rates[10:], expected_phase):
        for left, right in zip(actual.coeffs, expected.coeffs):
            difference = left - right
            assert difference.contains(0) and difference.width < Q(1, 2**65)


def test_full_storage_and_nonzero_pulses_support_only_existential_admission():
    z = sp.symbols("z", nonzero=True)
    phases, forms = [], []
    for i in range(5):
        for sign in (1, -1):
            phases.append(sign * _harmonic(z, 1, i))
            forms.append(-sign * sp.sqrt(C) * _harmonic(z, 1, i, sine=True))
    energy = sum(
        ((forms[i] - forms[j]) ** 2 + C * (phases[i] - phases[j]) ** 2) / 2
        for i, j in EDGES
    )
    radius = sum(value**2 for value in phases + forms)
    assert _clean(energy - 20 * C) == 0
    assert _clean(radius - 5 * (1 + C)) == 0
    cosine, sine = _harmonic(z, 1), _harmonic(z, 1, sine=True)
    assert _clean(cosine**2 + C * sine**2 - C - (1 - C) * cosine**2) == 0
    assert C.is_positive is True and (1 - C).is_positive is True
    assert _clean(sp.pi / 2 - 2 * sp.pi / 5 - sp.pi / 10) == 0
    # These are the existing active-family constants, not a computed new
    # amplitude cutoff. Analytic remainders must first become small enough.
    alpha = 2 * pi_interval() / 5
    radius_bound = Q(1, 10)
    acute = cos(alpha + sqrt(I(2)) * radius_bound)
    coercivity = (5 - sqrt(I(5))) * acute / 2
    assert acute.lo > 0 and coercivity.lo > 0
    assert 20 * C > 0
    # The proof gives rho_i >= c*epsilon^2/2 for sufficiently small nonzero
    # epsilon, and storage/radius O(epsilon^2); no numerical radius is inferred.
