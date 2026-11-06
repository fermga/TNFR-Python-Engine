"""Exact finite pulse-interaction tensor for prospective mathematical controls.

This reusable test helper implements the actual edge Hamiltonian in
GLOBAL_PULSE_HYPOTHESIS Section 14. It factors H3=tan(alpha)*h3 and constructs
Z4=Res(H4+{H3,S3}/2) over Q(zeta5), with all nine complex amplitudes.
The previously evaluated private fixture is preserved byte-for-byte as
historical evidence. Importing this helper does not construct the interaction tensor or evaluate
controls. Callers must admit and freeze their own calculation before requesting
the tensor.
"""

from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement, product
from math import factorial

import sympy as sp

R = sp.sqrt(5)
C = (R - 1) / 4
S = sp.sqrt((5 + R) / 8)
ZETA = C + sp.I * S
EDGES = tuple(
    sorted(
        {
            tuple(sorted((2 * i + a, 2 * ((i + 1) % 5) + b)))
            for i in range(5)
            for a, b in product((0, 1), repeat=2)
        }
    )
)
SPATIAL_MODES = (0, 0, 0, 0, 0, 1, 4, 2, 3)
NORMS = (2,) * 5 + (10,) * 4


def _add_term(polynomial, powers, coefficient):
    if not coefficient:
        return
    value = polynomial.get(powers, coefficient - coefficient) + coefficient
    if value:
        polynomial[powers] = value
    else:
        polynomial.pop(powers, None)


def _sum_polynomials(left, right, multiplier):
    result = dict(left)
    for powers, coefficient in right.items():
        _add_term(result, powers, multiplier * coefficient)
    return result


def _linear_power(linear, degree, field):
    result = {}
    indices = tuple(sorted(linear))
    for choice in combinations_with_replacement(indices, degree):
        counts = Counter(choice)
        powers = [0] * 18
        multiplicity = factorial(degree)
        coefficient = field.one
        for index, count in counts.items():
            powers[index] = count
            multiplicity //= factorial(count)
            coefficient *= linear[index] ** count
        _add_term(result, tuple(powers), field.convert(multiplicity) * coefficient)
    return result


def _differentiate(polynomial, index, field):
    result = {}
    for powers, coefficient in polynomial.items():
        if powers[index]:
            lowered = list(powers)
            lowered[index] -= 1
            result[tuple(lowered)] = coefficient * field.convert(powers[index])
    return result


def _multiply(left, right):
    result = {}
    for first, a in left.items():
        for second, b in right.items():
            powers = tuple(x + y for x, y in zip(first, second))
            _add_term(result, powers, a * b)
    return result


def _poisson_over_i(left, right, field):
    # {a_j,bar(a_j)}=2*i/||v_j||^2 for the unnormalized spatial basis.
    result = {}
    for index, norm in enumerate(NORMS):
        la, lb = (_differentiate(left, j, field) for j in (index, index + 9))
        ra, rb = (_differentiate(right, j, field) for j in (index, index + 9))
        factor = field.convert(sp.Rational(2, norm))
        result = _sum_polynomials(result, _multiply(la, rb), factor)
        result = _sum_polynomials(result, _multiply(lb, ra), -factor)
    return result


def _frequency_parts(powers):
    net = [powers[j] - powers[j + 9] for j in range(9)]
    internal, low, high = sum(net[:5]), sum(net[5:7]), sum(net[7:])
    return 4 * internal + 5 * low + 5 * high, -low + high


def _frequency(powers, field, radical):
    rational, irrational = _frequency_parts(powers)
    return (
        field.convert(sp.Rational(rational, 4))
        + field.convert(sp.Rational(irrational, 4)) * radical
    )


def _conjugate(coefficient, field, zeta):
    value = field.zero
    for rational in coefficient.to_list():
        value = value * zeta**4 + field.convert(rational)
    return value


def _intensity_power(first, second=None):
    powers = [0] * 18
    powers[first] += 1
    powers[first + 9] += 1
    if second is not None:
        powers[second] += 1
        powers[second + 9] += 1
    return tuple(powers)


@lru_cache(maxsize=1)
def build_interaction_tensor():
    field = sp.QQ.algebraic_field(ZETA)
    zeta = field.from_sympy(ZETA)
    radical = field.one + field.convert(2) * (zeta + zeta**4)
    tangent_squared = field.convert(5) + field.convert(2) * radical
    basis = [[field.zero for _ in range(9)] for _ in range(10)]
    for node in range(10):
        basis[node][node // 2] = field.convert(1 if node % 2 == 0 else -1)
        for mode in range(5, 9):
            basis[node][mode] = zeta ** ((SPATIAL_MODES[mode] * (node // 2)) % 5)
    h3, h4 = {}, {}
    for left, right in EDGES:
        sign = 1 if (right // 2 - left // 2) % 5 == 1 else -1
        linear = {}
        for index in range(9):
            value = (basis[right][index] - basis[left][index]) * field.convert(
                sp.Rational(1, 2)
            )
            if value:
                linear[index] = value
                linear[index + 9] = _conjugate(value, field, zeta)
        h3 = _sum_polynomials(
            h3, _linear_power(linear, 3, field), field.convert(sp.Rational(-sign, 24))
        )
        h4 = _sum_polynomials(
            h4, _linear_power(linear, 4, field), field.convert(sp.Rational(-1, 96))
        )
    divided = {
        powers: coefficient / _frequency(powers, field, radical)
        for powers, coefficient in h3.items()
    }
    k4 = _sum_polynomials(
        h4,
        _poisson_over_i(h3, divided, field),
        tangent_squared * field.convert(sp.Rational(1, 2)),
    )
    resonant = {p: a for p, a in k4.items() if _frequency_parts(p) == (0, 0)}
    return dict(
        field=field,
        basis=basis,
        zeta=zeta,
        radical=radical,
        tangent_squared=tangent_squared,
        h3=h3,
        h4=h4,
        divided=divided,
        k4=k4,
        resonant=resonant,
    )
