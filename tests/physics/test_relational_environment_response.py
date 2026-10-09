"""Static controls for conditional mediator poles, transfer and cancellation.

Pi intervals below concern the ideal default-law discriminants. The exact
matrix fixtures instead declare rational coefficients a and b independently;
they do not rationalize pi, authenticate a graph or evaluate a nonlinear
trajectory. Existing memory owners supply the block and observation algebra.
"""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics._exact_linear_algebra import exact_matrix_inverse as inverse
from tnfr.mathematics._exact_linear_algebra import exact_matrix_product as product
from tnfr.mathematics._rational_interval import I, pi_interval, sqrt
from tnfr.mathematics.linear_observation import (
    derive_coordinate_memory,
    derive_linear_observation,
)


def _identity(size):
    return tuple(tuple(Q(i == j) for j in range(size)) for i in range(size))


def _shift(matrix, frequency):
    return tuple(
        tuple(frequency * (i == j) - value for j, value in enumerate(row))
        for i, row in enumerate(matrix)
    )


def _scale(matrix, value):
    return tuple(tuple(value * entry for entry in row) for row in matrix)


def _hidden_generator(*, native=False, mu=Q(3), rho=Q(1, 4)):
    # An exact coefficient family, not an approximation of w/pi or w/(beta*pi).
    e, a, b = Q(2, 3), Q(2, 5), Q(3, 7)
    forward, reciprocal = (a, b / rho) if native else (a * rho, b)
    return ((-mu * e, -mu * forward), (mu * reciprocal, Q(0)))


def _path_four_inventory_memory():
    """Sine-consensus tangent with declared exact e,a,b and one hidden node."""
    e, a, b = Q(2, 3), Q(2, 5), Q(3, 7)
    laplacian = ((1, -1, 0, 0), (-1, 2, -1, 0), (0, -1, 2, -1), (0, 0, -1, 1))
    degrees = (1, 2, 2, 1)
    transport = tuple(
        tuple(Q(value, degree) for value in row)
        for row, degree in zip(laplacian, degrees)
    )
    generator = tuple(
        tuple(-e * value for value in row) + tuple(-a * value for value in row)
        for row in transport
    ) + tuple(tuple(b * value for value in row) + (Q(0),) * 4 for row in transport)
    # Retain form nodes 0,2,3 and phase nodes 0,2,3, hiding x_1,theta_1.
    return derive_coordinate_memory(generator, (0, 2, 3, 4, 6, 7)), (e, a, b)


def test_default_sine_hidden_discriminant_is_positive_on_the_whole_resultant_range():
    # Affine dependence on rho admits the entire interval, not sampled angles.
    discriminant = I(Q(1, 4)) - I(0, 1) / pi_interval() ** 2
    assert discriminant.lo > 0
    # At rho=0 the discriminant stays positive but one pole is zero. Strict
    # attraction also requires rho>0; a positive discriminant alone is not it.
    assert discriminant.contains(Q(1, 4))


@pytest.mark.parametrize("rho,sign", ((Q(1, 4), -1), (Q(1, 2), 1)))
def test_native_default_hidden_poles_have_geometry_dependent_character(rho, sign):
    discriminant = I(Q(1, 4)) - 1 / (pi_interval() ** 2 * rho)
    assert discriminant.hi < 0 if sign < 0 else discriminant.lo > 0
    # rho is an analytic coefficient here, not recognition of an ideal phase
    # configuration from a represented graph or an admission of its ports.


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("frequency", (Q(0), Q(3, 2)))
def test_hidden_transfer_and_full_schur_reduction_agree_without_static_closure(
    native, frequency
):
    mu, rho = Q(3), Q(1, 4)
    e, a, b = Q(2, 3), Q(2, 5), Q(3, 7)
    forward, reciprocal = (a, b / rho) if native else (a * rho, b)
    hidden = _hidden_generator(native=native, mu=mu, rho=rho)
    input_block = _scale(hidden, Q(-1))
    port = ((e / 3, a / 3), (-b / 3, Q(0)))
    visible = ((Q(-2), Q(0)), (Q(0), Q(-3)))
    generator = tuple(visible[i] + port[i] for i in range(2)) + tuple(
        input_block[i] + hidden[i] for i in range(2)
    )
    memory = derive_coordinate_memory(generator, (0, 1))
    denominator = frequency**2 + mu * e * frequency + mu**2 * forward * reciprocal
    expected = _scale(
        (
            (
                mu * e * frequency + mu**2 * forward * reciprocal,
                mu * forward * frequency,
            ),
            (-mu * reciprocal * frequency, mu**2 * forward * reciprocal),
        ),
        1 / denominator,
    )
    hidden_transfer = product(
        inverse(_shift(memory.hidden_generator, frequency)), memory.visible_to_hidden
    )
    assert hidden_transfer == expected
    feedback = product(memory.hidden_to_visible, hidden_transfer)
    effective = tuple(
        tuple(visible[i][j] + feedback[i][j] for j in range(2)) for i in range(2)
    )
    full_resolvent = inverse(_shift(generator, frequency))
    assert tuple(row[:2] for row in full_resolvent[:2]) == inverse(
        _shift(effective, frequency)
    )
    if frequency == 0:
        assert hidden_transfer == _identity(2)
    else:
        assert hidden_transfer != _identity(2)

    # The exact shared output-Krylov calculation retains both hidden
    # coordinates despite the identity static response at zero frequency.
    realization = derive_linear_observation(generator, ((1, 0, 0, 0), (0, 1, 0, 0)))
    assert realization.extra_coordinates == 2
    hidden_initial = ((Q(0),), (Q(1),))
    source = product(memory.hidden_to_visible, hidden_initial)
    assert source == ((a / 3,), (Q(0),))
    # This source survives at y(0)=0; removing it would change the full rate.
    assert tuple((row[3],) for row in generator[:2]) == source


def test_stable_hidden_oscillator_can_have_zero_initial_kernel_but_nonzero_memory():
    hidden = _hidden_generator()
    # A declared scalar observation/input of the same two-coordinate family.
    # Unlike the existing nilpotent control, this hidden block is stable.
    generator = ((Q(0), Q(1), Q(0)),) + tuple(
        (Q(i == 1),) + row for i, row in enumerate(hidden)
    )
    memory = derive_coordinate_memory(generator, (0,))
    assert memory.kernel_at_zero == ((0,),)
    first_derivative = product(
        product(memory.hidden_to_visible, memory.hidden_generator),
        memory.visible_to_hidden,
    )
    assert first_derivative[0][0] == Q(-3, 10)
    assert hidden[0][0] < 0
    assert hidden[0][0] * hidden[1][1] - hidden[0][1] * hidden[1][0] > 0
    source_initial = ((Q(0),), (Q(1),))
    assert product(memory.hidden_to_visible, source_initial) == ((0,),)
    source_derivative = product(
        product(memory.hidden_to_visible, hidden), source_initial
    )
    assert source_derivative == first_derivative
    assert derive_linear_observation(generator, ((1, 0, 0),)).dimension == 3
    # The signed causal transfer is independently obtained by exact inversion.
    frequency = Q(2)
    kernel_transform = product(
        product(memory.hidden_to_visible, inverse(_shift(hidden, frequency))),
        memory.visible_to_hidden,
    )
    determinant = (frequency - hidden[0][0]) * frequency - hidden[0][1] * hidden[1][0]
    assert kernel_transform == ((Q(-3, 10) / determinant,),)


@pytest.mark.parametrize("mu", (Q(1), Q(7)))
def test_outward_slow_pole_bounds_expose_resultant_dependent_relaxation(mu):
    e = Q(1, 2)
    stiffness = 1 / (4 * pi_interval() ** 2)
    previous_normalized = None
    for rho in (Q(1, 4), Q(1, 64), Q(1, 1024)):
        discriminant = I(e**2) - 4 * stiffness * rho
        # Rationalized root formula avoids subtracting nearly equal quantities.
        decay = 2 * mu * stiffness * rho / (e + sqrt(discriminant))
        assert 0 < decay.lo < decay.hi < mu * e / 2
        # Independent polynomial signs bracket the slow root for every
        # stiffness in the ideal-pi enclosure, on its decreasing branch.
        assert decay.lo**2 - mu * e * decay.lo + mu**2 * stiffness.lo * rho >= 0
        assert decay.hi**2 - mu * e * decay.hi + mu**2 * stiffness.hi * rho <= 0
        assert decay.lo > mu * stiffness.hi * rho / e
        assert decay.hi < 2 * mu * stiffness.lo * rho / e
        normalized = decay / (mu * rho)
        if previous_normalized is not None:
            assert normalized.hi < previous_normalized.lo
        previous_normalized = normalized
    # The brackets give decay=O(mu*rho), not a uniform rate proportional to
    # capacity alone. They certify poles, not nonlinear trajectory errors.


def test_exact_cancellation_retains_an_initial_phase_combination():
    e, b, mu = Q(2, 3), Q(3, 7), Q(5)
    # At zero port resultant the absolute hidden phase is retained; Psi is
    # undefined and must not be introduced as a coordinate. The frozen-port
    # sine law is exactly this affine contrast/absolute-phase system.
    generator = ((-mu * e, Q(0)), (mu * b, Q(0)))
    invariant = ((b / e, Q(1)),)
    result = derive_linear_observation(generator, invariant)
    assert result.dimension == 1
    assert result.reduced_generator == ((0,),)
    assert product(invariant, generator) == ((0, 0),)
    assert product(generator, ((Q(0),), (Q(1),))) == ((0,), (0,))
    # Two preparations with identical form and different hidden phase retain
    # different invariant values. Damping cannot select one phase at R=0.
    preparations = ((Q(2), Q(2)), (Q(-1), Q(1)))
    assert product(invariant, preparations) == ((Q(2, 7), Q(16, 7)),)


def test_undamped_cancellation_has_exact_affine_phase_drift():
    mu, b, time = Q(3), Q(2, 7), Q(5, 4)
    generator = ((Q(0), Q(0)), (mu * b, Q(0)))
    assert product(generator, generator) == ((0, 0), (0, 0))
    transition = ((Q(1), Q(0)), (mu * b * time, Q(1)))
    # T'=J*T and T(0)=I identify the exact flow without an ODE solver.
    assert product(generator, transition) == generator
    initial = ((Q(2),), (Q(-1),))
    assert product(transition, initial) == ((Q(2),), (Q(8, 7),))


def test_conserved_hidden_inventory_constrains_memory_and_initial_source():
    memory, (e, a, b) = _path_four_inventory_memory()
    full_charge = ((1, 2, 2, 1, 0, 0, 0, 0),)
    visible_charge, hidden_charge = ((1, 2, 1, 0, 0, 0),), ((2, 0),)
    assert product(full_charge, memory.generator) == ((0,) * 8,)
    assert product(visible_charge, memory.visible_generator) == _scale(
        product(hidden_charge, memory.visible_to_hidden), -1
    )
    assert product(visible_charge, memory.hidden_to_visible) == _scale(
        product(hidden_charge, memory.hidden_generator), -1
    )

    hidden_initial = ((Q(1),), (Q(0),))
    power = _identity(2)
    for _ in range(4):
        next_power = product(power, memory.hidden_generator)
        for right in (memory.visible_to_hidden, hidden_initial):
            visible_jet = product(
                product(product(visible_charge, memory.hidden_to_visible), power), right
            )
            hidden_inventory_jet = product(product(hidden_charge, next_power), right)
            assert visible_jet == _scale(hidden_inventory_jet, -1)
        power = next_power

    # y(0)=0, h(0)=(1,0) has nonzero visible rates entirely from the
    # retained source. Removing it would falsely freeze the visible charge.
    visible_rate = product(memory.hidden_to_visible, hidden_initial)
    assert visible_rate == ((e,), (e / 2,), (0,), (-b,), (-b / 2,), (0,))
    assert product(visible_charge, visible_rate) == ((2 * e,),)
    hidden_rate = product(memory.hidden_generator, hidden_initial)
    assert product(hidden_charge, hidden_rate) == ((-2 * e,),)
    # Hidden phase alone can transfer existing form inventory between the
    # two blocks even when the initial total form charge is zero.
    phase_initial = ((Q(0),), (Q(1),))
    assert product(
        product(visible_charge, memory.hidden_to_visible), phase_initial
    ) == ((2 * a,),)


def test_stationary_hidden_substitution_can_lose_reconstructed_total_inventory():
    memory, (e, a, b) = _path_four_inventory_memory()
    hidden_lift = _scale(
        product(inverse(memory.hidden_generator), memory.visible_to_hidden), -1
    )
    assert hidden_lift == (
        (Q(1, 2), Q(1, 2), 0, 0, 0, 0),
        (0, 0, 0, Q(1, 2), Q(1, 2), 0),
    )
    feedback = product(memory.hidden_to_visible, hidden_lift)
    reduced = tuple(
        tuple(left + right for left, right in zip(row, correction))
        for row, correction in zip(memory.visible_generator, feedback)
    )
    transport = ((-Q(1, 2), Q(1, 2), 0), (Q(1, 4), -Q(3, 4), Q(1, 2)), (0, 1, -1))
    expected = tuple(
        tuple(e * value for value in row) + tuple(a * value for value in row)
        for row in transport
    ) + tuple(tuple(-b * value for value in row) + (0,) * 3 for row in transport)
    assert reduced == expected

    visible_charge = ((1, 2, 1, 0, 0, 0),)
    inventory_correction = product(((2, 0),), hidden_lift)
    full_inventory = (
        tuple(
            left + right
            for left, right in zip(visible_charge[0], inventory_correction[0])
        ),
    )
    assert full_inventory == ((2, 3, 1, 0, 0, 0),)
    assert product(visible_charge, reduced) == ((0,) * 6,)
    assert product(full_inventory, reduced) == (
        (-e / 4, -e / 4, e / 2, -a / 4, -a / 4, a / 2),
    )

    # The actual hidden derivative starts at zero on the stationary graph,
    # but differentiating its algebraic reconstruction gives a nonzero rate.
    initial = ((Q(0),), (Q(0),), (Q(1),), (Q(0),), (Q(0),), (Q(0),))
    assert product(hidden_lift, initial) == ((0,), (0,))
    assert product(memory.visible_to_hidden, initial) == ((0,), (0,))
    initial_rate = product(reduced, initial)
    assert product(hidden_lift, initial_rate) == ((e / 4,), (-b / 4,))
    assert product(full_inventory, initial_rate) == ((e / 2,),)


def test_charge_neutral_hidden_source_still_drives_visible_sine_tangent():
    full, (e, _, b) = _path_four_inventory_memory()
    # Retain the two endpoints' form/phase and hide both interior nodes.
    # Reflection-odd hidden data has zero total form inventory, but it
    # transmits nonzero and opposite rates to the visible endpoints.
    memory = derive_coordinate_memory(full.generator, (0, 3, 4, 7))
    visible_charge = ((1, 1, 0, 0),)
    hidden_charge = ((2, 2, 0, 0),)
    hidden_initial = ((Q(1),), (Q(-1),), (Q(0),), (Q(0),))
    source = product(memory.hidden_to_visible, hidden_initial)
    assert source == ((e,), (-e,), (-b,), (b,))
    assert any(row[0] for row in source)
    assert product(visible_charge, source) == ((0,),)
    assert product(hidden_charge, hidden_initial) == ((0,),)

    # Cayley-Hamilton on this four-coordinate hidden block makes these
    # finite checks sufficient for all-time zero charge projection of the
    # exponential source, without declaring the vector source itself zero.
    jet = hidden_initial
    for _ in range(len(memory.hidden_indices)):
        assert product(hidden_charge, jet) == ((0,),)
        assert product(visible_charge, product(memory.hidden_to_visible, jet)) == (
            (0,),
        )
        jet = product(memory.hidden_generator, jet)


def test_nonlinear_asymptotic_origin_can_be_conserved_while_bare_mean_drifts():
    s = pytest.importorskip("sympy")
    y, mean, coefficient = s.symbols("y mean c", real=True)
    decay = s.Symbol("lambda", positive=True)
    time = s.Symbol("t", nonnegative=True)
    relative_flow = y * s.exp(-decay * time)
    correction = s.integrate(coefficient * relative_flow**2, (time, 0, s.oo))
    assert correction == coefficient * y**2 / (2 * decay)
    field = s.Matrix((-decay * y, coefficient * y**2))
    chart = s.Matrix((y, mean + correction))
    chart_derivative = chart.jacobian((y, mean))
    assert chart_derivative.det() == 1
    assert (chart_derivative * field).applyfunc(s.simplify) == s.Matrix((-decay * y, 0))
    assert s.simplify(s.diff(correction, y) * field[0] + field[1]) == 0
    # The invariant is not the old linear mean: it also consumes relative
    # state and the declared law coefficients. This is a generic local
    # chart control, not an added nodal pressure or a recovery trajectory.
    assert s.diff(chart[1], y) == coefficient * y / decay
