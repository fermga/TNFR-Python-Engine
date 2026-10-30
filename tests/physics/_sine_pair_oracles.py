"""Independent doubled-C5 algebra shared by the sine response tests.

The Cartesian jets are local fine-row derivatives, never a trajectory solver
or a production report. Kernel differentiation and comparison-series bounds
remain test oracles. High-precision helpers use the caller's private context.
Frozen preparations, assessment calls and pytest fixtures stay in their suites;
this module imports neither production implementations nor other test modules.
"""

from fractions import Fraction as Q
from math import comb, factorial

PAIRS = tuple((2 * a, 2 * a + 1) for a in range(5))


EDGES = tuple((i, j) for a in range(5) for i in PAIRS[a] for j in PAIRS[(a + 1) % 5])


NEIGHBORS = tuple(
    tuple(j if i == node else i for i, j in EDGES if node in (i, j))
    for node in range(10)
)


ZERO = (Q(0), Q(0))


ONE = (Q(1), Q(0))


MINUS_ONE = (Q(-1), Q(0))


ORIENTATIONS = (ONE, (Q(0), Q(1)))


def _mul(left, right):
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _conj(value):
    return value[0], -value[1]


def _negative(value):
    return -value[0], -value[1]


def _preparation(rotation, orientation, *, control=False):
    neighbors = (
        (rotation, rotation, _negative(rotation), _negative(rotation))
        if control
        else (rotation, ONE, MINUS_ONE, MINUS_ONE)
    )
    return (orientation, _negative(orientation)) + tuple(
        value for phase in neighbors for value in (phase, phase)
    )


def _fine_jets(phasors, order=4, forms=None, epsilon=Q(0)):
    """Taylor coefficients of x and exp(i theta), in tau=t/pi.

    Start from zero forms unless explicitly supplied. Iterate convolution of the
    ten original fine form rows and ten phasor rows. No quotient field,
    reported rate, fitted coefficient or time step is used. Optional epsilon
    adds the declared cubic sine current using its exact power series.
    """
    forms = (Q(0),) * 10 if forms is None else forms
    x = [[Q(value)] for value in forms]
    z = [[value] for value in phasors]
    for k in range(order):
        next_x, next_z = [], []
        for i, neighbors in enumerate(NEIGHBORS):
            dx = Q(0)
            for j in neighbors:
                sine = tuple(
                    sum(
                        (
                            _mul(_conj(z[i][r]), z[j][degree - r])[1]
                            for r in range(degree + 1)
                        ),
                        Q(0),
                    )
                    for degree in range(k + 1)
                )
                dx += sine[k]
                if epsilon:
                    dx += epsilon * sum(
                        (
                            sine[p] * sine[q] * sine[k - p - q]
                            for p in range(k + 1)
                            for q in range(k - p + 1)
                        ),
                        Q(0),
                    )
            dx /= 4
            hz = tuple(
                sum(
                    (
                        (x[i][r] - sum((x[j][r] for j in neighbors), Q(0)) / 4)
                        * z[i][k - r][component]
                        for r in range(k + 1)
                    ),
                    Q(0),
                )
                for component in (0, 1)
            )
            next_x.append(dx / (k + 1))
            next_z.append((-hz[1] / (k + 1), hz[0] / (k + 1)))
        for i in range(10):
            x[i].append(next_x[i])
            z[i].append(next_z[i])
    return x, z


def _current_coefficient(z, degree, receiver, source):
    """Mean-form contribution: one eighth of the four directed edge sines."""
    return (
        sum(
            (
                _mul(_conj(z[i][r]), z[j][degree - r])[1]
                for i in PAIRS[receiver]
                for j in PAIRS[source]
                for r in range(degree + 1)
            ),
            Q(0),
        )
        / 8
    )


def _integrated_current_remainder(horizon):
    """Integrate one current's independently assembled derivative majorant."""
    phasor_bounds = ((1,), (0, 2), (2, 0, 4), (0, 20, 0, 8))
    third_current = [Q(0)] * 4
    for split in range(4):
        for left_power, left in enumerate(phasor_bounds[split]):
            for right_power, right in enumerate(phasor_bounds[3 - split]):
                third_current[left_power + right_power] += (
                    Q(1, 2) * comb(3, split) * left * right
                )
    return sum(
        (
            value * factorial(power) * horizon ** (power + 4) / factorial(power + 4)
            for power, value in enumerate(third_current)
        ),
        Q(0),
    )


def _receiver_jets(rotation, orientation, epsilon):
    x, _ = _fine_jets(_preparation(rotation, orientation), epsilon=epsilon)
    return tuple(sum((x[i][k] for i in PAIRS[1]), Q(0)) / 2 for k in range(5))


def _differentiate_phase_polynomial(polynomial):
    """Differentiate powers of sine and cosine, collecting exact monomials."""
    result = {}
    for (sine_power, cosine_power), value in polynomial.items():
        if sine_power:
            index = (sine_power - 1, cosine_power + 1)
            result[index] = result.get(index, Q(0)) + sine_power * value
        if cosine_power:
            index = (sine_power + 1, cosine_power - 1)
            result[index] = result.get(index, Q(0)) - cosine_power * value
    return result


def _independent_current_derivative_bounds(epsilon):
    # Begin with the declared edge law, not its report's derivative bounds.
    polynomial = {(1, 0): Q(1), (3, 0): epsilon}
    bounds = []
    for _ in range(4):
        bounds.append(sum((abs(value) for value in polynomial.values()), Q(0)))
        polynomial = _differentiate_phase_polynomial(polynomial)
    return tuple(bounds)


def _independent_receiver_remainder(horizon, epsilon):
    value, first, second, third = _independent_current_derivative_bounds(epsilon)
    # Original degree-four rows give |delta'| <= 4*M*tau,
    # |delta''| <= 4*M and |delta'''| <= 16*L*M*tau.
    velocity = 4 * value
    acceleration = 4 * value
    jerk = 4 * first * velocity
    # Differentiate j(delta) three times, keeping each chain-rule term.
    majorant = {
        1: 3 * second * velocity * acceleration + first * jerk,
        3: third * velocity**3,
    }
    # X1' averages eight directed edge pressures with factor 1/8.
    edge_normalization = sum((len(NEIGHBORS[i]) for i in PAIRS[1]), 0) * Q(1, 8)
    return edge_normalization * sum(
        (
            coefficient
            * factorial(power)
            * horizon ** (power + 4)
            / factorial(power + 4)
            for power, coefficient in majorant.items()
        ),
        Q(0),
    )


def _preparation_bound(horizon, epsilon, form, phase):
    """Sum the coupled fine-row comparison as a two-channel Neumann series."""
    lipschitz = _independent_current_derivative_bounds(epsilon)[1]
    form_from_phase = horizon * lipschitz * max(Q(2 * len(row), 4) for row in NEIGHBORS)
    phase_from_form = horizon * max(1 + Q(len(row), 4) for row in NEIGHBORS)
    term = (Q(form), Q(phase))
    partial = [Q(0), Q(0)]
    for _ in range(8):
        partial[0] += term[0]
        partial[1] += term[1]
        term = (form_from_phase * term[1], phase_from_form * term[0])
    block_ratio = (form_from_phase * phase_from_form) ** 4
    assert block_ratio < 1
    return partial[0] / (1 - block_ratio)


def _cubic_coefficients(rotation, orientation):
    samples = tuple(_receiver_jets(rotation, orientation, Q(e))[3] for e in range(3))
    quadratic = (samples[2] - 2 * samples[1] + samples[0]) / 2
    return samples[0], samples[1] - samples[0] - quadratic, quadratic


def _value(coefficients, epsilon):
    return sum(
        (value * epsilon**power for power, value in enumerate(coefficients)), Q(0)
    )


def _initial_rate_bound(rotation, orientation, upper):
    # At fixed initial phases each original fine form row is affine in epsilon.
    # Its absolute maximum on a closed interval therefore occurs at an endpoint.
    return max(
        abs(node[1])
        for epsilon in (Q(0), upper)
        for node in _fine_jets(
            _preparation(rotation, orientation), order=1, epsilon=epsilon
        )[0]
    )


def _refined_remainder(horizon, upper, initial_rate):
    """Integrate the original fine-row Lipschitz and chain-rule majorants."""
    _, first, second, third = _independent_current_derivative_bounds(upper)
    form_row_weight = max(sum((Q(1, len(row)) for _ in row), Q(0)) for row in NEIGHBORS)
    phase_row_norm = max(
        1 + sum((Q(1, len(row)) for _ in row), Q(0)) for row in NEIGHBORS
    )
    # Each gap rate is a difference of two complete phase rows. Integrating
    # x'' <= growth_coefficient*X*t twice gives the time-normalized form bound.
    growth_coefficient = form_row_weight * first * 2 * phase_row_norm
    ratio = growth_coefficient * horizon**2 / factorial(3)
    assert ratio < 1
    finite_series = sum((ratio**power for power in range(8)), Q(0))
    growth = initial_rate * finite_series / (1 - ratio**8)
    rate = initial_rate + growth_coefficient * growth * horizon**2 / factorial(2)
    gap_velocity = 2 * phase_row_norm * growth
    gap_acceleration = 2 * phase_row_norm * rate
    gap_jerk = 2 * phase_row_norm * form_row_weight * first * gap_velocity
    derivative_majorant = {
        1: 3 * second * gap_velocity * gap_acceleration + first * gap_jerk,
        3: third * gap_velocity**3,
    }
    receiver_normalization = sum(
        (Q(1, 2 * len(NEIGHBORS[i])) for i in PAIRS[1] for _ in NEIGHBORS[i]), Q(0)
    )
    error = receiver_normalization * sum(
        coefficient * factorial(power) * horizon ** (power + 4) / factorial(power + 4)
        for power, coefficient in derivative_majorant.items()
    )
    return growth, rate, error


def _mp(mp, value):
    return (
        mp.mpf(value.numerator) / value.denominator
        if isinstance(value, Q)
        else mp.mpf(value)
    )


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _energy(mp, forms, phases):
    return sum(
        (forms[i] - forms[j]) ** 2 / 2 + 1 - mp.cos(phases[j] - phases[i])
        for i, j in EDGES
    )


def _norm(mp, values):
    return mp.sqrt(sum(value * value for value in values))
