"""Seven exact controls for the full periodic-branch spectral proof.

No variational trajectory or numerical multiplier is computed. Formal
coefficient projections check the analytic spectral reduction; they do not
replace its remainder argument or establish nonlinear orbital stability.
"""

from fractions import Fraction as Q
from functools import lru_cache
from itertools import product

import mpmath as mp
import pytest
import sympy as sp

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval, sqrt
from tnfr.physics._sine_flow import _held_sine_neighbor_field

R = sp.sqrt(5)
C = (R - 1) / 4
S = sp.sqrt((5 + R) / 8)
ROOT = C + sp.I * S
KAPPA = -(14 + 21 * R) / 164
HARMONIC = (95 + 40 * R) / 164
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
    return sp.expand(ROOT ** (power % 5))


@lru_cache(maxsize=1)
def _fourier_field():
    return sp.QQ.algebraic_field(ROOT)


def _fourier_zero(matrix):
    # The derived matrix can contain different radical presentations of the
    # same cyclotomic number. Field coefficients decide equality exactly.
    for entry in matrix:
        value = _clean(entry)
        if value != 0:
            assert not value.free_symbols
            field = _fourier_field()
            assert field.from_sympy(value) == field.zero


def _laplacian(weights):
    result = sp.zeros(10)
    for (i, j), weight in zip(EDGES, weights):
        result[i, i] += weight
        result[j, j] += weight
        result[i, j] -= weight
        result[j, i] -= weight
    return result


def _inverse(matrix):
    rows = tuple(tuple(Q(int(x.p), int(x.q)) for x in row) for row in matrix.tolist())
    return sp.Matrix(exact_matrix_inverse(rows))


def _coefficient(vector, variable, degree):
    return vector.applyfunc(lambda value: sp.expand(value).coeff(variable, degree))


def _contains(interval, expression):
    # This is a cross-check, not a replacement for exact algebraic identities.
    with mp.workdps(90):
        expected = mp.mpf(str(sp.N(expression, 85)))
        lower = mp.mpf(interval.lo.numerator) / interval.lo.denominator
        upper = mp.mpf(interval.hi.numerator) / interval.hi.denominator
        if interval.lo == interval.hi:
            assert abs(expected - lower) < mp.mpf("1e-80")
        else:
            assert lower <= expected <= upper
        assert interval.width < Q(1, 2**65)


@pytest.fixture(scope="module")
def geometry():
    fine = _laplacian((1,) * 20)
    adjacency, difference = sp.zeros(5), sp.zeros(5)
    plus, minus = sp.zeros(10, 5), sp.zeros(10, 5)
    for i in range(5):
        adjacency[i, (i - 1) % 5] = adjacency[i, (i + 1) % 5] = 1
        difference[i, (i + 1) % 5], difference[i, (i - 1) % 5] = 1, -1
        plus[2 * i, i] = plus[2 * i + 1, i] = 1
        minus[2 * i, i], minus[2 * i + 1, i] = 1, -1
    laplacian, origin = 2 * sp.eye(5) - adjacency, sp.ones(5) / 5
    inverse = _inverse(laplacian + origin) - origin
    second = _inverse(laplacian**2 - 16 * sp.eye(5))
    canonical = sp.BlockMatrix(
        [[sp.zeros(10), sp.eye(10)], [-sp.eye(10), sp.zeros(10)]]
    ).as_explicit()
    reference = sp.BlockMatrix(
        [[sp.zeros(10), fine / 4], [-fine / 4, sp.zeros(10)]]
    ).as_explicit()
    return dict(
        Lf=fine,
        L=laplacian,
        Adj=adjacency,
        D=difference,
        plus=plus,
        minus=minus,
        R0=inverse,
        R2=second,
        J=canonical,
        A0=reference,
    )


@pytest.fixture(scope="module")
def variation(geometry):
    """Edgewise Hessian coefficients along the established branch, not a new law."""
    z = sp.symbols("z", nonzero=True)
    carrier = sp.Matrix([_root(i) for i in range(5)])
    base = (carrier * z + carrier.conjugate() / z) / 2
    mean = sp.Matrix(
        [
            -HARMONIC * (z**2 * _root(2 * i) - z**-2 * _root(-2 * i)) / (2 * sp.I)
            for i in range(5)
        ]
    )
    first, second = [], []
    for left, right in EDGES:
        i, j = left // 2, right // 2
        internal_gap = (-1) ** (right % 2) * base[j] - (-1) ** (left % 2) * base[i]
        mean_gap = mean[j] - mean[i]
        sine_gap = sp.im(_root(j - i))
        first.append(-sine_gap * internal_gap)
        second.append(-sine_gap * mean_gap - C * internal_gap**2 / 2)
    first, second = _laplacian(first), _laplacian(second)
    fine = geometry["Lf"]
    acceleration0 = -(fine**2) / 16
    acceleration1 = -fine * first / (16 * C)
    # Omega^-2 = 1+2*kappa*epsilon^2+O(epsilon^4) affects both rows.
    acceleration2 = -fine * second / (16 * C) + 2 * KAPPA * acceleration0
    return dict(
        z=z,
        carrier=carrier,
        base=base,
        mean=mean,
        H1=first,
        H2=second,
        Q0=acceleration0,
        Q1=acceleration1,
        Q2=acceleration2,
    )


@pytest.fixture(scope="module")
def bridge(geometry, variation):
    """Resonant derivative derived from the full fine Hessian and its range equation."""
    z, carrier = variation["z"], variation["carrier"]
    real, imag = sp.symbols("x0:5", real=True), sp.symbols("y0:5", real=True)
    w = sp.Matrix([x + sp.I * y for x, y in zip(real, imag)])
    b = sp.diag(*carrier) * w
    perturbation = (b * z + b.conjugate() / z) / 2
    plus, minus = geometry["plus"], geometry["minus"]
    forcing = plus.T * variation["Q1"] * minus * perturbation / 2
    expected = (
        -(S / C)
        * geometry["L"]
        * geometry["D"]
        * variation["base"].multiply_elementwise(perturbation)
        / 4
    )
    _zero(forcing - expected)
    coefficients = {n: _coefficient(forcing, z, n) for n in (0, 2, -2)}
    collective = 4 * geometry["R0"] ** 2 * coefficients[0]
    for n in (2, -2):
        collective += 4 * geometry["R2"] * coefficients[n] * z**n
    _zero(geometry["L"] ** 2 * _coefficient(collective, z, 0) / 4 - coefficients[0])
    for n in (2, -2):
        _zero(
            (geometry["L"] ** 2 / 4 - 4 * sp.eye(5)) * _coefficient(collective, z, n)
            - coefficients[n]
        )
    resonant_force = (
        minus.T
        * (variation["Q2"] * minus * perturbation + variation["Q1"] * plus * collective)
        / 2
    )
    # If forcing = Re(f*exp(is)), then 2*i*B_T=f. The exponential
    # coefficient is f/2, retaining the distinction in the previous gate.
    wdot = -sp.I * sp.diag(*carrier.conjugate()) * _coefficient(resonant_force, z, 1)
    jx, jy = wdot.jacobian(real), wdot.jacobian(imag)
    actual_a = ((sp.I * jx + jy) / 2).applyfunc(_clean)
    actual_b = ((sp.I * jx - jy) / 2).applyfunc(_clean)
    return actual_a, actual_b


def test_full_jacobian_is_hamiltonian_and_matches_all_original_rows(geometry):
    phases = tuple(Q((7 * i) % 17 - 8, 16) for i in range(10))
    curvatures = tuple(sp.cos(phases[j] - phases[i]) for i, j in EDGES)
    hessian = _laplacian(curvatures)
    fine, canonical = geometry["Lf"], geometry["J"]
    jacobian = sp.BlockMatrix(
        [[sp.zeros(10), fine / 4], [-hessian / (4 * C), sp.zeros(10)]]
    ).as_explicit()
    _zero(jacobian.T * canonical + canonical * jacobian)
    for origin in (
        sp.ones(10, 1).col_join(sp.zeros(10, 1)),
        sp.zeros(10, 1).col_join(sp.ones(10, 1)),
    ):
        _zero(jacobian * origin)
    neighbors = tuple(
        tuple((j, 1, Q(1)) for j in range(10) if tuple(sorted((i, j))) in EDGES)
        for i in range(10)
    )
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    field = _held_sine_neighbor_field(model, (4,) * 10, (Q(1),) * 10, neighbors)
    curvature = (sqrt(I(5)) - 1) / 4
    root, pi = sqrt(curvature), pi_interval()
    for seed in (1, 3, 5):
        direction = tuple(Q((seed * i + 3) % 13 - 6, 16) for i in range(20))
        state = tuple(
            Jet((Q(i - 4, 16), root * direction[10 + i])) for i in range(10)
        ) + tuple(Jet((phases[i], direction[i])) for i in range(10))
        rates = field(state)
        actual = tuple(value.coeffs[1] * pi / root for value in rates[10:]) + tuple(
            value.coeffs[1] * pi / curvature for value in rates[:10]
        )
        expected = jacobian * sp.Matrix(direction)
        for interval, value in zip(actual, expected):
            _contains(interval, value)


def test_reference_spectrum_retains_internal_fast_and_origin_multiplicities(geometry):
    fine, laplacian = geometry["Lf"], geometry["L"]
    plus, minus = geometry["plus"], geometry["minus"]
    _zero(fine * plus - 2 * plus * laplacian)
    _zero(fine * minus - 4 * minus)
    ell1, ell2 = (5 - R) / 4, (5 + R) / 4
    spectral = sp.symbols("lambda")
    expected = (
        spectral**2
        * (spectral**2 + 1) ** 5
        * (spectral**2 + ell1**2) ** 2
        * (spectral**2 + ell2**2) ** 2
    )
    assert _clean(geometry["A0"].charpoly(spectral).as_expr() - expected) == 0
    assert len(geometry["A0"].nullspace()) == 2
    # These exact intervals exclude all full-period collisions, including
    # opposite Krein signs; the noninteger frequencies also exclude +1.
    for value, lower, upper in (
        (ell1, 0, 1),
        (ell2, 1, 2),
        (2 * ell1, 1, 2),
        (2 * ell2, 3, 4),
        (ell2 - ell1, 1, 2),
    ):
        assert (value - lower).is_positive is True and (
            upper - value
        ).is_positive is True
    assert _clean(ell1 + ell2 - sp.Rational(5, 2)) == 0
    assert sp.cos(sp.pi) == -1 and sp.cos(2 * sp.pi) == 1


def test_actual_resonant_bridge_recovers_the_slow_generator(
    geometry, variation, bridge
):
    actual_a, actual_b = bridge
    laplacian, difference = geometry["L"], geometry["D"]
    n = (
        sp.eye(5)
        + geometry["Adj"]
        - (S / C) ** 2 * difference * geometry["R0"] * difference
    )
    m = (
        geometry["Adj"]
        - (S / C) ** 2 * difference * geometry["R2"] * laplacian * difference
    )
    carrier = variation["carrier"]
    m2 = -(358 + 168 * R) / 41
    rotated = (
        sp.diag(*(carrier.conjugate().applyfunc(lambda x: x**2)))
        * m
        * sp.diag(*(carrier.applyfunc(lambda x: x**2)))
    )
    _zero(actual_a - n / 16 - rotated / 16 + m2 * sp.eye(5) / 32)
    _zero(actual_b - n / 16 - m2 * sp.eye(5) / 32)
    _zero(actual_a - actual_a.conjugate().T)
    _zero(actual_b - actual_b.T)
    # Both constant and second-harmonic periodic collective inverses were
    # solved in bridge; the free fast modes are retained in separate clusters.


def test_slow_eigenvalues_are_exactly_nonzero_simple_and_separated(bridge):
    actual_a, actual_b = bridge
    n = (3, 13 + 6 * R, 3, 3, 13 + 6 * R)
    m1, m2 = -(108 + 2 * R) / 41, -(358 + 168 * R) / 41
    m = (2, m1, m2, m2, m1)
    fourier = sp.Matrix.hstack(
        *(sp.Matrix([_root(k * i) for i in range(5)]) for k in range(5))
    )
    _zero(fourier.conjugate().T * fourier - 5 * sp.eye(5))
    for k in range(5):
        _fourier_zero(
            actual_a * fourier[:, k]
            - (2 * n[k] + 2 * m[(k + 2) % 5] - m2) * fourier[:, k] / 32
        )
        _fourier_zero(actual_b * fourier[:, k] - (2 * n[k] + m2) * fourier[:, k] / 32)
    bounds = (
        (Q(28, 25), Q(113, 100), Q(47, 100), Q(12, 25)),
        (Q(31, 50), Q(63, 100), Q(7, 50), Q(3, 20)),
    )
    rlo, rhi = Q(223, 100), Q(56, 25)
    assert rlo**2 < 5 < rhi**2
    for k, (slo, shi, dlo, dhi) in zip((1, 2), bounds):
        plus = (2 * n[k] + 2 * m[(k + 2) % 5] - m2) / 32
        minus = (2 * n[k] + 2 * m[(2 - k) % 5] - m2) / 32
        coupling = (2 * n[k] + m2) / 32
        mean, drift = (plus + minus) / 2, (plus - minus) / 2
        delta = _clean(mean**2 - coupling**2)
        spectral = sp.symbols("lambda")
        block = -sp.I * sp.Matrix(((plus, coupling), (-coupling, -minus)))
        assert (
            _clean(
                block.charpoly(spectral).as_expr()
                - (spectral + sp.I * drift) ** 2
                - delta
            )
            == 0
        )
        expected = ((136515 + 60007 * R) if k == 1 else (39775 + 19283 * R)) / 215168
        assert _clean(delta - expected) == 0
        # Affine expressions in sqrt(5), bounded by rational endpoint algebra.
        affine = sp.symbols("r", real=True)
        lower = Q(
            delta.xreplace({R: affine}).subs(
                affine, sp.Rational(rlo.numerator, rlo.denominator)
            )
        )
        upper = Q(
            delta.xreplace({R: affine}).subs(
                affine, sp.Rational(rhi.numerator, rhi.denominator)
            )
        )
        assert slo**2 < lower <= upper < shi**2
        positive_drift = _clean(-drift)
        lower_d = Q(
            positive_drift.xreplace({R: affine}).subs(
                affine, sp.Rational(rlo.numerator, rlo.denominator)
            )
        )
        upper_d = Q(
            positive_drift.xreplace({R: affine}).subs(
                affine, sp.Rational(rhi.numerator, rhi.denominator)
            )
        )
        assert dlo < lower_d <= upper_d < dhi
    # Disjoint positive frequency intervals derived from those bounds.
    assert Q(2, 5) < bounds[1][0] - bounds[1][3] < bounds[1][1] - bounds[1][2] < Q(1, 2)
    assert (
        Q(3, 5) < bounds[0][0] - bounds[0][3] < bounds[0][1] - bounds[0][2] < Q(7, 10)
    )
    assert Q(3, 4) < bounds[1][0] + bounds[1][2] < bounds[1][1] + bounds[1][3] < Q(4, 5)
    assert Q(3, 2) < bounds[0][0] + bounds[0][2] < bounds[0][1] + bounds[0][3] < Q(5, 3)
    neutral = -sp.I * sp.Matrix(((KAPPA, KAPPA), (-KAPPA, -KAPPA)))
    assert neutral.rank() == 1 and neutral**2 == sp.zeros(2)


def test_each_repeated_fast_cluster_has_a_definite_krein_form(geometry):
    plus, canonical = geometry["plus"], geometry["J"]
    for mode, ell in ((1, (5 - R) / 4), (2, (5 + R) / 4)):
        spatial = plus * sp.Matrix.hstack(
            *(sp.Matrix([_root(k * i) for i in range(5)]) for k in (mode, 5 - mode))
        )
        for sign in (1, -1):
            vectors = spatial.col_join(sign * sp.I * spatial)
            _zero(geometry["A0"] * vectors - sign * sp.I * ell * vectors)
            gram = -sp.I * vectors.conjugate().T * canonical * vectors
            _zero(gram - sign * 20 * sp.eye(2))
    # Definite restrictions protect a repeated cluster; simplicity is not
    # silently asserted for either of its two spatial copies.


def test_member_corrected_half_return_retains_parity_and_original_clock(
    geometry, variation
):
    swap = sp.zeros(10)
    for i in range(10):
        swap[i, i ^ 1] = 1
    full_swap = sp.diag(swap, swap)
    _zero(full_swap.T * geometry["J"] * full_swap - geometry["J"])
    _zero(swap * variation["H1"] * swap + variation["H1"])
    _zero(swap * variation["H2"] * swap - variation["H2"])
    _zero(variation["H1"].subs(variation["z"], -variation["z"]) + variation["H1"])
    _zero(variation["H2"].subs(variation["z"], -variation["z"]) - variation["H2"])
    _zero(swap * geometry["minus"] + geometry["minus"])
    _zero(swap * geometry["plus"] - geometry["plus"])
    # A constructed noncommuting symplectic half map checks the endpoint
    # association algebra, not an unevaluated monodromy of the TNFR branch.
    identity, zero = sp.eye(2), sp.zeros(2)
    kick = sp.BlockMatrix(
        [[identity, zero], [sp.Matrix(((2, -1), (-1, 1))), identity]]
    ).as_explicit()
    drift = sp.BlockMatrix(
        [[identity, sp.Matrix(((1, 2), (2, 3)))], [zero, identity]]
    ).as_explicit()
    half, exchange = drift * kick, sp.diag(1, -1, 1, -1)
    second_half = exchange * half * exchange
    corrected, full = exchange * half, second_half * half
    assert corrected**2 == full and half**2 != full
    canonical = sp.BlockMatrix([[zero, identity], [-identity, zero]]).as_explicit()
    for mapping in (half, exchange, corrected):
        _zero(mapping.T * canonical * mapping - canonical)
    frequency, exponent, epsilon = sp.symbols("Omega sigma epsilon", positive=True)
    period = 2 * sp.pi**2 / (sp.sqrt(C) * frequency)
    original_rate = sp.sqrt(C) * frequency * epsilon**2 * exponent / sp.pi
    assert _clean(period * original_rate - 2 * sp.pi * epsilon**2 * exponent) == 0
    assert _clean(period * original_rate / 2 - sp.pi * epsilon**2 * exponent) == 0


def test_period_energy_derivatives_expose_neutral_shear_and_dimension_count(geometry):
    epsilon, p4, e4 = sp.symbols("epsilon p4 e4", real=True)
    period0 = 2 * sp.pi**2 / sp.sqrt(C)
    period = period0 * (1 + KAPPA * epsilon**2 + p4 * epsilon**4)
    energy = 20 * C * epsilon**2 + e4 * epsilon**4
    assert (
        _clean(
            sp.expand(sp.diff(period, epsilon)).coeff(epsilon, 1) - 2 * KAPPA * period0
        )
        == 0
    )
    assert _clean(sp.expand(sp.diff(energy, epsilon)).coeff(epsilon, 1) - 40 * C) == 0
    assert KAPPA.is_negative is True and C.is_positive is True
    period_derivative = sp.symbols("period_derivative", real=True, nonzero=True)
    nilpotent = sp.Matrix(((0, -period_derivative), (0, 0)))
    assert nilpotent.rank() == 1 and nilpotent**2 == sp.zeros(2)
    # (M-I)*branch_derivative=-P'*flow. Nonzero energy derivative makes
    # those vectors independent; the complete return is not semisimple.
    cosines = sp.Matrix([sp.re(_root(i)) for i in range(5)])
    sines = sp.Matrix([sp.im(_root(i)) for i in range(5)])
    minus = geometry["minus"]
    amplitude = (minus * cosines).col_join(-minus * sines)
    tangent = (-minus * sines).col_join(-minus * cosines)
    origins = (
        sp.ones(10, 1).col_join(sp.zeros(10, 1)),
        sp.zeros(10, 1).col_join(sp.ones(10, 1)),
    )
    basis = sp.Matrix.hstack(*origins, tangent, amplitude)
    gram = (basis.T * geometry["J"] * basis).applyfunc(_clean)
    assert gram.rank() == 4
    assert (
        _clean(
            (amplitude.T * sp.diag(C * geometry["Lf"], C * geometry["Lf"]) * amplitude)[
                0
            ]
            - 40 * C
        )
        == 0
    )
    assert geometry["A0"].rows - gram.rank() == 16
    # The sixteen separated transverse directions exclude the two origins
    # and the phase/family Jordan pair; no uniform epsilon->0 bound follows.
