"""Finite Chebyshev polynomial algebra with shared outward intervals.

Coefficients mean ``sum(a[k]*T_k(x))`` on [-1,1], including an unhalved
constant coefficient. These are finite polynomials, not Taylor jets,
approximations to arbitrary functions, or an ODE solver. Callers own any
discarded polynomial tail and any change from x to physical time.

Products use the exact Laurent identity T_k=(z^k+z^-k)/2. Dyadic endpoint
integers are accumulated by carry-free packed integer convolution and
converted to the shared I representation only once per output coefficient.
The interval enclosure does not assume independent positive/negative
Laurent coefficients; treating their errors separately is conservative.

At most seven packed products, Laurent lengths <=385 per factor,
and <=769 coefficients per convolution are used. Bigint width depends
on coefficient magnitudes, with an explicit 2^22-bit allocation ceiling
per packed product. These work limits do not constrain a physical model.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from fractions import Fraction as Q
from functools import lru_cache

from .._exact_time import exact_or_represented_real as _exact
from ._dyadic_linear import _dot_encoded_rows, _encode_admitted_row
from ._rational_interval import INTERVAL_BITS, I, cos, pi_interval, sqrt

__all__ = (
    "MAX_FACTOR_DEGREE",
    "MAX_DEGREE",
    "MAX_PACKED_BITS",
    "admit_coefficients",
    "add",
    "subtract",
    "scale",
    "derivative",
    "multiply",
    "lobatto_nodes",
    "interpolate_lobatto_rows",
)

MAX_FACTOR_DEGREE = 192
MAX_DEGREE = 384
MAX_PACKED_BITS = 1 << 22
_MAX_LAURENT_LENGTH = 2 * MAX_FACTOR_DEGREE + 1
_MAX_LOBATTO_DEGREE = 96
_MAX_LOBATTO_ROWS = 3072
_GRID = 1 << INTERVAL_BITS
_PRODUCT_GRID = 16 * _GRID**2
_ZERO = I(0)


def _lobatto_degree(degree):
    if type(degree) is not int:
        raise TypeError("Lobatto degree must be an ordinary integer")
    if not 1 <= degree <= _MAX_LOBATTO_DEGREE:
        raise ValueError(f"Lobatto degree must lie in 1..{_MAX_LOBATTO_DEGREE}")
    return degree


@lru_cache(maxsize=8)
def _lobatto_nodes_admitted(degree):
    """Cache only immutable roots for an already admitted fixed degree."""
    nodes = [I(1)]
    for index in range(1, (degree + 1) // 2):
        nodes.append(cos(pi_interval() * Q(index, degree)))
    positive = tuple(nodes)
    if degree % 2 == 0:
        nodes.append(_ZERO)
    nodes.extend(-value for value in reversed(positive))
    return tuple(nodes)


def lobatto_nodes(degree=96):
    """Enclose cos(j*pi/n), j=0..n, in descending order for 1<=n<=96.

    Endpoints +1 and -1 and the even-degree middle zero are exact. Other
    intervals retain the shared mathematical-pi and cosine uncertainty.
    These nodes specify sampling locations, not function sample values.
    """
    return _lobatto_nodes_admitted(_lobatto_degree(degree))


@lru_cache(maxsize=8)
def _lobatto_weights_admitted(degree):
    """Integer centers/radii on grid 2*S, including sample endpoint weights."""
    nodes = _lobatto_nodes_admitted(degree)
    roots = _encode_admitted_row(nodes)
    result = []
    for mode in range(degree + 1):
        row = []
        for sample in range(degree + 1):
            index = mode * sample % (2 * degree)
            root = roots[min(index, 2 * degree - index)]
            factor = 1 if sample in (0, degree) else 2
            row.append(tuple(factor * value for value in root))
        result.append(tuple(row))
    return tuple(result)


def interpolate_lobatto_rows(samples, *, degree=96):
    """Enclose ordinary Chebyshev coefficients of prescribed Lobatto samples.

    Each of 1..3072 ordered rows contains n+1 samples at ``lobatto_nodes(n)``,
    with n an ordinary integer in 1..96. The full constant coefficient is
    retained: a_k=(2/n)*w_k*sum_j(w_j*f_j*cos(k*j*pi/n)), where endpoint
    weights are 1/2 and all others are one. The output always has n+1 entries.

    Complete primitive admission precedes fixed-weight preparation and all
    dot products. Exact dyadic integer products accumulate their signed
    centers and conservative radii before one shared I rounding per output.
    The work is O(rows*n**2), with O(n**2+rows*n) retained scalar storage;
    integer bit cost still depends on admitted magnitudes. Immutable weights
    are cached for at most eight degrees, never for supplied sample values.

    Sample and arithmetic uncertainty are retained. Approximation of any
    function between the nodes requires a separate interpolation-tail bound;
    no such bound, ODE candidate, or physical conclusion is returned here.
    """
    degree = _lobatto_degree(degree)
    raw_rows = _ordered(samples, _MAX_LOBATTO_ROWS, "Lobatto sample rows")
    admitted = []
    for index, values in enumerate(raw_rows):
        label = f"Lobatto samples[{index}]"
        row = _ordered(values, degree + 1, label)
        if len(row) != degree + 1:
            raise ValueError(f"{label} must contain exactly {degree + 1} entries")
        admitted.append(tuple(_interval(value, label) for value in row))
    weights = _lobatto_weights_admitted(degree)
    result = []
    for row in admitted:
        encoded = _encode_admitted_row(row)
        result.append(
            tuple(
                _dot_encoded_rows(
                    encoded,
                    weight_row,
                    numerator=1 if mode in (0, degree) else 2,
                    denominator=2 * degree,
                )
                for mode, weight_row in enumerate(weights)
            )
        )
    return tuple(result)


def _ordered(values, maximum, label):
    if isinstance(values, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError(f"{label} must be an ordered finite sequence")
    try:
        count = len(values)
    except TypeError as exc:
        raise TypeError(f"{label} must be an ordered finite sequence") from exc
    if not 1 <= count <= maximum:
        raise ValueError(f"{label} must contain between one and {maximum} entries")
    try:
        return tuple(values[index] for index in range(count))
    except (TypeError, KeyError, IndexError) as exc:
        raise TypeError(f"{label} must be an indexed finite sequence") from exc


def _interval(value, label):
    if isinstance(value, I):
        return I(_exact(value.lo, f"{label} lower"), _exact(value.hi, f"{label} upper"))
    return I(_exact(value, label))


def admit_coefficients(
    values, *, max_degree=MAX_DEGREE, label="Chebyshev coefficients"
):
    """Normalize every primitive coefficient, retaining explicit trailing zeros."""
    if type(max_degree) is not int:
        raise TypeError("maximum Chebyshev degree must be an ordinary integer")
    if not 0 <= max_degree <= MAX_DEGREE:
        raise ValueError(f"maximum Chebyshev degree must lie in 0..{MAX_DEGREE}")
    return tuple(
        _interval(value, f"{label}[{index}]")
        for index, value in enumerate(_ordered(values, max_degree + 1, label))
    )


def add(left, right):
    """Add finite polynomials, preserving the greater declared degree."""
    left, right = admit_coefficients(left), admit_coefficients(right)
    return _add_admitted(left, right)


def _add_admitted(left, right):
    """Add locally owned interval rows after complete boundary admission."""
    return tuple(
        (left[index] if index < len(left) else _ZERO)
        + (right[index] if index < len(right) else _ZERO)
        for index in range(max(len(left), len(right)))
    )


def subtract(left, right):
    """Subtract finite polynomials without trimming a canceled leading term."""
    left, right = admit_coefficients(left), admit_coefficients(right)
    return _subtract_admitted(left, right)


def _subtract_admitted(left, right):
    """Subtract locally owned interval rows after complete boundary admission."""
    return tuple(
        (left[index] if index < len(left) else _ZERO)
        - (right[index] if index < len(right) else _ZERO)
        for index in range(max(len(left), len(right)))
    )


def scale(coefficients, factor):
    """Multiply every coefficient by one re-admitted real interval scalar."""
    coefficients = admit_coefficients(coefficients)
    factor = _interval(factor, "Chebyshev scale")
    return _scale_admitted(coefficients, factor)


def _scale_admitted(coefficients, factor):
    """Scale an admitted row by an admitted interval scalar."""
    return tuple(value * factor for value in coefficients)


def _grid_bounds(value):
    return (
        value.lo.numerator * (_GRID // value.lo.denominator),
        value.hi.numerator * (_GRID // value.hi.denominator),
    )


def _norm_upper_admitted(values):
    """Euclidean upper bound on owned dyadic intervals, normalized before sqrt."""
    largest = squared_sum = 0
    for value in values:
        lower, upper = _grid_bounds(value)
        bound = max(abs(lower), abs(upper))
        largest = max(largest, bound)
        squared_sum += bound * bound
    if not largest:
        return Q(0)
    # Each absolute bound is bound/S. Cancel S before the sum of squares,
    # preserving the same rational normalized sum and final shared rounding.
    normalized_square = Q(squared_sum, largest * largest)
    return Q(largest, _GRID) * sqrt(I(normalized_square)).hi


def derivative(coefficients):
    """Differentiate in x; a degree-n row returns max(n,1) coefficients."""
    coefficients = admit_coefficients(coefficients)
    return _derivative_admitted(coefficients)


def _derivative_admitted(coefficients):
    """Differentiate a locally owned admitted row without repeated admission."""
    degree = len(coefficients) - 1
    if not degree:
        return (_ZERO,)
    bounds = tuple(_grid_bounds(value) for value in coefficients)
    lower, upper = [0] * (degree + 2), [0] * (degree + 2)
    for index in range(degree - 1, -1, -1):
        lower[index] = lower[index + 2] + 2 * (index + 1) * bounds[index + 1][0]
        upper[index] = upper[index + 2] + 2 * (index + 1) * bounds[index + 1][1]
    return tuple(
        I(
            Q(lower[index], (2 if index == 0 else 1) * _GRID),
            Q(upper[index], (2 if index == 0 else 1) * _GRID),
        )
        for index in range(degree)
    )


def _integer_row(values, label, *, nonnegative=False):
    row = _ordered(values, _MAX_LAURENT_LENGTH, label)
    if any(type(value) is not int for value in row):
        raise TypeError(f"{label} must contain ordinary integers")
    if nonnegative and any(value < 0 for value in row):
        raise ValueError(f"{label} must contain nonnegative integers")
    return row


def _nonnegative_convolution(left, right):
    """One exact carry-free Kronecker product, with bounded allocation."""
    left = _integer_row(left, "left convolution row", nonnegative=True)
    right = _integer_row(right, "right convolution row", nonnegative=True)
    count = len(left) + len(right) - 1
    largest_left, largest_right = max(left), max(right)
    if not largest_left or not largest_right:
        return (0,) * count
    coefficient_bound = min(len(left), len(right)) * largest_left * largest_right
    digit_bytes = (coefficient_bound.bit_length() + 7) // 8
    if count * digit_bytes * 8 > MAX_PACKED_BITS:
        raise ValueError("Chebyshev packed convolution exceeds its bit work limit")

    def pack(row):
        return int.from_bytes(
            b"".join(value.to_bytes(digit_bytes, "little") for value in row), "little"
        )

    # Every convolution coefficient is strictly below 2^(8*digit_bytes),
    # including when coefficient_bound itself is an exact power of two.
    packed = (pack(left) * pack(right)).to_bytes(count * digit_bytes, "little")
    return tuple(
        int.from_bytes(packed[start : start + digit_bytes], "little")
        for start in range(0, len(packed), digit_bytes)
    )


def _integer_convolution(left, right):
    """Signed convolution through positive/negative parts, without carries."""
    left, right = _integer_row(left, "left convolution row"), _integer_row(
        right, "right convolution row"
    )
    positive_left, negative_left = tuple(max(value, 0) for value in left), tuple(
        max(-value, 0) for value in left
    )
    positive_right, negative_right = tuple(max(value, 0) for value in right), tuple(
        max(-value, 0) for value in right
    )
    same_positive = _nonnegative_convolution(positive_left, positive_right)
    same_negative = _nonnegative_convolution(negative_left, negative_right)
    cross_positive = _nonnegative_convolution(positive_left, negative_right)
    cross_negative = _nonnegative_convolution(negative_left, positive_right)
    return tuple(
        a + b - c - d
        for a, b, c, d in zip(
            same_positive, same_negative, cross_positive, cross_negative
        )
    )


def _laurent_rows(coefficients):
    """Integer centers/radii on grid 4*S with the palindromic layout."""
    degree = len(coefficients) - 1
    centers, radii = [0] * (2 * degree + 1), [0] * (2 * degree + 1)
    for index, value in enumerate(coefficients):
        lower, upper = _grid_bounds(value)
        if index == 0:
            centers[degree], radii[degree] = 2 * (lower + upper), 2 * (upper - lower)
        else:
            centers[degree - index] = centers[degree + index] = lower + upper
            radii[degree - index] = radii[degree + index] = upper - lower
    return tuple(centers), tuple(radii)


def multiply(left, right):
    """Enclose the complete product, with one final I rounding per coefficient.

    Both factors have declared degree at most 192; the result has their
    summed degree, at most 384, even when its leading coefficients vanish.
    No terms are truncated. The packed bit cap is an explicit work boundary.
    """
    left = admit_coefficients(left, max_degree=MAX_FACTOR_DEGREE)
    right = admit_coefficients(right, max_degree=MAX_FACTOR_DEGREE)
    return _multiply_admitted(left, right)


def _multiply_admitted(left, right):
    """Multiply owned admitted factors whose degrees are each at most 192."""
    degree = len(left) + len(right) - 2
    if all(value.lo == value.hi == 0 for value in left) or all(
        value.lo == value.hi == 0 for value in right
    ):
        return (_ZERO,) * (degree + 1)
    left_center, left_radius = _laurent_rows(left)
    right_center, right_radius = _laurent_rows(right)
    center = _integer_convolution(left_center, right_center)
    first_radius = _nonnegative_convolution(
        tuple(abs(value) for value in left_center), right_radius
    )
    second_radius = _nonnegative_convolution(
        left_radius, tuple(abs(value) for value in right_center)
    )
    joint_radius = _nonnegative_convolution(left_radius, right_radius)
    result = []
    for index in range(degree + 1):
        position, factor = degree + index, 1 if index == 0 else 2
        radius = (
            first_radius[position] + second_radius[position] + joint_radius[position]
        )
        result.append(
            I(
                Q(factor * (center[position] - radius), _PRODUCT_GRID),
                Q(factor * (center[position] + radius), _PRODUCT_GRID),
            )
        )
    return tuple(result)
