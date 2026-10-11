"""Independent algebra and finite-bound controls for the collective comparison.

Rational gamma values test identities valid for every positive gamma. They
are neither replacements for the model's fixed gamma nor physical bounds.
No trajectory, formation certificate or reserved response is evaluated.
"""

from fractions import Fraction as Q
import hashlib
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_square_matrix_power,
    exact_square_matrix_product,
    exact_symmetric_semidefinite,
)
from tnfr.mathematics._comparison_flow import scalar_affine_upper
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.physics._organized_c5_preparation import prepared_c5_initial_energy_upper
from tnfr.physics._organized_effective_law import bound_prepared_matrix_response


@pytest.fixture(scope="module", autouse=True)
def preserve_frozen_evidence_and_producers():
    root = Path(__file__).resolve().parents[2]
    paths = tuple(
        root / name
        for name in (
            "docs/assets/organized_finite_observation/attempt.json",
            "docs/assets/organized_finite_observation/result.json",
            "benchmarks/organized_finite_observation.py",
            "src/tnfr/physics/_organized_finite_observation_source.py",
            "docs/assets/organized_robust_current/attempt.json",
            "docs/assets/organized_robust_current/result.json",
            "benchmarks/organized_robust_current.py",
            "src/tnfr/physics/_organized_robust_current.py",
        )
    )
    before = tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths)
    with forbid_organized_response_evaluation():
        yield
    assert tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths) == before


_EDGES = tuple((c, (c + 1) % 5) for c in range(5))
_IDENTITY = tuple(tuple(Q(i == j) for j in range(5)) for i in range(5))


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _action(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _edge_form_gradient(form):
    gradient = [Q(0)] * 5
    for left, right in _EDGES:
        difference = form[left] - form[right]
        gradient[left] += difference
        gradient[right] -= difference
    return tuple(gradient)


def _quarter_turn_sine_source(phases):
    """Evaluate the edge sine sum exactly for integer multiples of pi/2."""
    values = (Q(0), Q(1), Q(0), Q(-1))
    source = [Q(0)] * 5
    for left, right in _EDGES:
        current = values[(phases[right] - phases[left]) % 4]
        source[left] += current
        source[right] -= current
    return tuple(source)


@pytest.fixture(scope="module")
def cycle_laplacian():
    matrix = [[Q(0)] * 5 for _ in range(5)]
    for left, right in _EDGES:
        matrix[left][left] += 1
        matrix[right][right] += 1
        matrix[left][right] -= 1
        matrix[right][left] -= 1
    return tuple(map(tuple, matrix))


def _response_matrices(laplacian, ratio):
    square = exact_square_matrix_power(laplacian, 2)
    metric = tuple(
        tuple(base + ratio * value for base, value in zip(unit, row))
        for unit, row in zip(_IDENTITY, square)
    )
    return metric, exact_matrix_inverse(metric)


@pytest.mark.parametrize("ratio", (Q(0), Q(1, 100), Q(2, 7), Q(5)))
def test_response_is_positive_symmetric_and_preserves_common_mode(
    cycle_laplacian, ratio
):
    metric, response = _response_matrices(cycle_laplacian, ratio)
    assert exact_symmetric_semidefinite(metric, strict=True)
    assert exact_symmetric_semidefinite(response, strict=True)
    assert exact_square_matrix_product(metric, response) == _IDENTITY
    assert exact_square_matrix_product(response, metric) == _IDENTITY
    storage_metric = exact_square_matrix_product(cycle_laplacian, metric)
    assert storage_metric == exact_square_matrix_product(metric, cycle_laplacian)
    assert exact_symmetric_semidefinite(storage_metric)
    assert not exact_symmetric_semidefinite(storage_metric, strict=True)
    common = (Q(1),) * 5
    assert _action(metric, common) == _action(response, common) == common
    assert _action(storage_metric, common) == (Q(0),) * 5
    if ratio == 0:
        assert metric == response == _IDENTITY
        assert storage_metric == cycle_laplacian


@pytest.mark.parametrize("ratio", (Q(0), Q(1, 100), Q(2, 7), Q(5)))
@pytest.mark.parametrize("gamma", (Q(1, 1023), Q(3, 7)))
def test_complete_rows_balance_storage_and_transform_the_driven_input(
    cycle_laplacian, ratio, gamma
):
    metric, response = _response_matrices(cycle_laplacian, ratio)
    form = tuple(Q(value, 17) for value in (3, -2, 1, -5, 3))
    forcing = tuple(Q(value, 19) for value in (-2, 5, 1, -3, -1))
    source = _quarter_turn_sine_source((0, 1, -1, 2, 0))
    gradient = _edge_form_gradient(form)
    assert gradient == _action(cycle_laplacian, form)
    unforced = tuple((-q + gamma * s) / 4 for q, s in zip(gradient, source))
    form_rate = _action(response, tuple(a + b for a, b in zip(unforced, forcing)))
    phase_rate = tuple(gamma * q / 4 for q in gradient)
    assert sum(form) == sum(forcing) == sum(form_rate) == sum(phase_rate) == 0

    # Differentiate U.L.U/2 + r*(L U).L.(L U)/2 directly. Its gradient is
    # L U + r*L^3 U; the potential gradient is the negative edge sine sum.
    cubic_gradient = _edge_form_gradient(_edge_form_gradient(gradient))
    storage_gradient = tuple(
        q + ratio * cube for q, cube in zip(gradient, cubic_gradient)
    )
    derivative = _dot(storage_gradient, form_rate) - _dot(source, phase_rate)
    loss = _dot(gradient, gradient) / 4
    work = _dot(gradient, forcing)
    assert loss > 0 and work != 0
    assert derivative == -loss + work
    assert derivative == _dot(forcing, forcing) - sum(
        (q - 2 * f) ** 2 / 4 for q, f in zip(gradient, forcing)
    )
    assert _action(metric, form_rate) == tuple(a + b for a, b in zip(unforced, forcing))

    wrong_input_rate = tuple(
        a + b for a, b in zip(_action(response, unforced), forcing)
    )
    wrong_derivative = _dot(storage_gradient, wrong_input_rate) - _dot(
        source, phase_rate
    )
    if ratio:
        assert wrong_input_rate != form_rate
        assert wrong_derivative != -loss + work
    else:
        assert form_rate == tuple(a + b for a, b in zip(unforced, forcing))
        assert wrong_input_rate == form_rate


@pytest.mark.parametrize("ratio", (Q(0), Q(1, 100), Q(2, 7)))
def test_charge_chart_transforms_phase_row_storage_and_work(cycle_laplacian, ratio):
    metric, response = _response_matrices(cycle_laplacian, ratio)
    gamma = Q(3, 7)
    form = tuple(Q(value, 17) for value in (3, -2, 1, -5, 3))
    forcing = tuple(Q(value, 19) for value in (-2, 5, 1, -3, -1))
    source = _quarter_turn_sine_source((0, 1, -1, 2, 0))
    gradient = _edge_form_gradient(form)
    charge = _action(metric, form)
    recovered = _action(response, charge)
    assert recovered == form
    charge_rate = tuple(
        (-q + gamma * s) / 4 + f for q, s, f in zip(gradient, source, forcing)
    )
    form_rate = _action(response, charge_rate)
    phase_rate = tuple(gamma * q / 4 for q in gradient)
    transformed_phase_rate = tuple(
        gamma * q / 4 for q in _edge_form_gradient(recovered)
    )
    wrong_phase_rate = tuple(gamma * q / 4 for q in _edge_form_gradient(charge))
    assert _action(metric, form_rate) == charge_rate
    assert transformed_phase_rate == phase_rate

    original_form_storage = _dot(form, _edge_form_gradient(charge)) / 2
    transformed_form_storage = _dot(charge, _edge_form_gradient(recovered)) / 2
    assert original_form_storage == transformed_form_storage
    # The changed chart has gradient L M Z. Its work conjugate stays L U.
    transformed_derivative = _dot(gradient, charge_rate) - _dot(
        source, transformed_phase_rate
    )
    assert transformed_derivative == -_dot(gradient, gradient) / 4 + _dot(
        gradient, forcing
    )
    if ratio:
        assert wrong_phase_rate != phase_rate
        assert _dot(source, wrong_phase_rate) != _dot(source, phase_rate)
    else:
        assert charge == form and wrong_phase_rate == phase_rate


def test_positive_definite_response_is_not_a_nonnegative_averaging_map(cycle_laplacian):
    metric, response = _response_matrices(cycle_laplacian, Q(1, 100))
    # Construct from edge incidence, invert, then compare the claimed row;
    # this is not a lookup table used to define the tested matrix itself.
    assert response[0] == tuple(Q(value, 2305) for value in (2181, 81, -19, -19, 81))
    assert exact_square_matrix_product(metric, response) == _IDENTITY
    assert exact_symmetric_semidefinite(response, strict=True)
    nonnegative = (Q(0), Q(0), Q(1), Q(0), Q(0))
    result = _action(response, nonnegative)
    assert result[0] == Q(-19, 2305) < 0
    assert _dot(nonnegative, result) > 0
    assert sum(result) == sum(nonnegative)


@pytest.mark.parametrize("ratio", (Q(1, 10**9), Q(1, 100), Q(2, 7), Q(5)))
@pytest.mark.parametrize("gamma", (Q(1, 1023), Q(3, 7)))
def test_old_storage_can_increase_while_own_storage_decreases(
    cycle_laplacian, ratio, gamma
):
    metric, response = _response_matrices(cycle_laplacian, ratio)
    # Phi=(0,pi/2,0,0,0) supplies this exact current from the actual edge law.
    source = _quarter_turn_sine_source((0, 1, 0, 0, 0))
    assert source == (Q(1), Q(-2), Q(1), Q(0), Q(0))
    form = tuple(-gamma * ratio * value / 2 for value in _edge_form_gradient(source))
    gradient = _edge_form_gradient(form)
    form_rate = _action(
        response, tuple((-q + gamma * s) / 4 for q, s in zip(gradient, source))
    )
    phase_rate = tuple(gamma * q / 4 for q in gradient)
    old_derivative = _dot(gradient, form_rate) - _dot(source, phase_rate)
    own_derivative = _dot(_action(metric, gradient), form_rate) - _dot(
        source, phase_rate
    )
    # Algebra gives these identities for every positive r and gamma, while
    # SPD of M makes the first expression strictly positive for q!=0.
    assert any(gradient)
    assert old_derivative == _dot(gradient, _action(response, gradient)) / 4 > 0
    assert own_derivative == -_dot(gradient, gradient) / 4 < 0
    extra_storage = ratio * _dot(gradient, _edge_form_gradient(gradient)) / 2
    assert extra_storage > 0


@pytest.fixture(scope="module")
def prepared_bound():
    return bound_prepared_matrix_response()


def test_neutral_polynomial_and_spectral_bounds_hold_as_matrix_identities(
    cycle_laplacian,
):
    laplacian = cycle_laplacian
    square = exact_square_matrix_power(laplacian, 2)
    projection = tuple(tuple(value - Q(1, 5) for value in row) for row in _IDENTITY)
    assert all(
        square[i][j] - 5 * laplacian[i][j] + 5 * projection[i][j] == 0
        for i in range(5)
        for j in range(5)
    )
    assert exact_symmetric_semidefinite(
        tuple(
            tuple(laplacian[i][j] - projection[i][j] for j in range(5))
            for i in range(5)
        )
    )
    assert exact_symmetric_semidefinite(
        tuple(
            tuple(4 * _IDENTITY[i][j] - laplacian[i][j] for j in range(5))
            for i in range(5)
        ),
        strict=True,
    )
    # Polynomial coefficient comparison proves the inverse formula for every r,
    # rather than only the numerical parameter points in the following control.
    linear = tuple(
        tuple(20 * projection[i][j] - 5 * laplacian[i][j] for j in range(5))
        for i in range(5)
    )
    quadratic = exact_square_matrix_product(square, linear)
    assert all(
        square[i][j] + linear[i][j] == 15 * projection[i][j]
        and quadratic[i][j] == 25 * projection[i][j]
        for i in range(5)
        for j in range(5)
    )


@pytest.mark.parametrize("response", (Q(1, 100), Q(1, 50), Q(1, 10), Q(2)))
def test_gain_comes_from_actual_matrix_input(cycle_laplacian, response):
    bound = bound_prepared_matrix_response(
        response_lower=response, response_upper=response
    )
    _, inverse = _response_matrices(cycle_laplacian, response)
    pulse = (Q(1), Q(-1), Q(0), Q(0), Q(0))
    gain = _dot(pulse, _action(inverse, pulse))
    assert gain == bound.response_gain_lower == bound.response_gain_upper
    assert 0 < gain < _dot(pulse, pulse)
    # Omitting M_r on the input incorrectly makes this gain equal to r=0.
    assert gain != 2


@pytest.mark.parametrize("winding", (-1, 1))
def test_identity_collar_has_independent_cosine_coercivity(winding):
    kappa = winding * 2 * pi_interval() / 5
    rho = Q(1, 10)
    assert cos(kappa + I(-rho, rho)).lo > Q(1, 5)
    assert (1 / (1023 * pi_interval())).hi < 1
    assert 2 * pi_interval().hi / 5 + rho < pi_interval().lo / 2
    boundary = (rho / 2, -rho / 2, Q(0), Q(0), Q(0))
    differences = tuple(boundary[j] - boundary[i] for i, j in _EDGES)
    assert sum(boundary) == sum(differences) == 0
    assert max(map(abs, differences)) == rho
    potential_excess = sum((cos(kappa) - cos(kappa + gap) for gap in differences), I(0))
    coercive_bound = sum(gap**2 for gap in differences) / 10
    assert potential_excess.lo >= coercive_bound >= Q(1, 1000)


@pytest.mark.parametrize("response", (Q(0), Q(1, 10)))
def test_shared_preparation_bound_covers_actual_form_and_phase_storage(
    cycle_laplacian, response
):
    radius = Q(1, 32)
    form = tuple(radius * value / 4 for value in (1, -1, 0, 0, 0))
    phase = tuple(radius * value / 8 for value in (1, 0, -1, 0, 0))
    assert sum(form) == sum(phase) == 0
    # The joint l1 bound independently suffices for the admitted l2 radii.
    assert sum(map(abs, form)) + sum(map(abs, phase)) < radius
    metric, _ = _response_matrices(cycle_laplacian, response)
    form_energy = _dot(form, _action(cycle_laplacian, _action(metric, form))) / 2
    kappa = 2 * pi_interval() / 5
    phase_energy = sum(
        (cos(kappa) - cos(kappa + phase[j] - phase[i]) for i, j in _EDGES),
        I(0),
    )
    bound = prepared_c5_initial_energy_upper(
        response_upper=response, preparation_radius=radius
    )
    assert form_energy > 0 and phase_energy.lo > 0
    assert form_energy + phase_energy.hi < bound


@pytest.mark.parametrize("parameter", ("response_upper", "preparation_radius"))
@pytest.mark.parametrize("invalid", (True, Q(-1), float("inf"), float("nan"), "1"))
def test_shared_preparation_bound_rejects_invalid_primitives(parameter, invalid):
    arguments = {"response_upper": Q(0), "preparation_radius": Q(0)}
    arguments[parameter] = invalid
    with pytest.raises((TypeError, ValueError)):
        prepared_c5_initial_energy_upper(**arguments)


def test_shared_preparation_admits_zero_and_normalizes_fraction_components():
    np = pytest.importorskip("numpy")
    assert prepared_c5_initial_energy_upper(response_upper=0, preparation_radius=0) == 0
    radius = Q(np.int64(2**40), np.int64(3))
    result = prepared_c5_initial_energy_upper(
        response_upper=Q(np.int64(1), np.int64(8)), preparation_radius=radius
    )
    assert result == Q(2**83, 9)
    assert type(result.numerator) is type(result.denominator) is int


def test_declared_finite_budgets_pass_without_a_response_evaluation(prepared_bound):
    bound = prepared_bound
    assert bound.identity_certified and bound.response_separated
    assert bound.energy_upper < Q(1, 2**16) < bound.storage_barrier
    assert bound.response_gap > Q(1, 2**21)
    assert bound.positive_observation_bounds[1] < bound.control_observation_bounds[0]
    # Independent positive-series exponential bound for the norm inequality.
    exponential_upper = scalar_affine_upper(
        bound.preparation_radius, 2, bound.horizon, 2 * bound.amplitude
    )
    assert exponential_upper <= bound.state_norm_upper < Q(1, 2**12)
    assert bound.pulse_absolute_work_upper < Q(1, 2**20)
    # A coarser spectral bound, independent of the rational C5 gain, suffices.
    coarse_gap = 2 * bound.amplitude * bound.horizon / 101 - 2 * (
        bound.response_remainder + bound.observation_error
    )
    assert Q(1, 2**21) < coarse_gap < bound.response_gap


def test_identity_and_response_flags_keep_separate_sufficient_claims(prepared_bound):
    noisy = bound_prepared_matrix_response(observation_error=Q(1))
    assert noisy.identity_certified and not noisy.response_separated
    assert noisy.energy_upper == prepared_bound.energy_upper
    assert noisy.pulse_absolute_work_upper == prepared_bound.pulse_absolute_work_upper
    assert noisy.response_gap < 0
    # High pulse energy can defeat this sufficient barrier while the finite
    # response intervals remain disjoint. Neither failure proves identity loss.
    energetic = bound_prepared_matrix_response(amplitude=Q(1))
    assert not energetic.identity_certified and energetic.response_separated
    assert energetic.energy_upper > energetic.storage_barrier
    broad_source = bound_prepared_matrix_response(preparation_radius=Q(1, 20))
    assert not broad_source.identity_certified and not broad_source.response_separated


@pytest.mark.parametrize("response", (Q(0), Q(1, 100), Q(1, 10)))
def test_equal_form_means_winding_and_storage_hide_opposite_live_rates(
    cycle_laplacian, response
):
    delta = Q(1, 2**26)
    shift = delta / 2
    plus = tuple(shift * (Q(i == 0) - Q(1, 5)) for i in range(5))
    minus = tuple(-value for value in plus)
    assert sum(plus) == sum(minus) == 0
    assert sum(value**2 for value in plus) < delta**2
    gap_plus = tuple(plus[j] - plus[i] for i, j in _EDGES)
    gap_minus = tuple(minus[j] - minus[i] for i, j in _EDGES)
    # The gap multisets coincide exactly: the full potential and winding do
    # too, without testing one stored energy literal against another.
    assert sorted(gap_plus) == sorted(gap_minus)
    _, inverse = _response_matrices(cycle_laplacian, response)
    kappa = 2 * pi_interval() / 5
    gamma = 1 / (1023 * pi_interval())

    def live_rate(gaps):
        source = [I(0) for _ in range(5)]
        for (i, j), gap in zip(_EDGES, gaps):
            current = sin(kappa + gap)
            source[i] += current
            source[j] -= current
        # Equal prescribed inputs add the same M_r*f and cancel in the rate
        # difference. This evaluates the actual edge-sine pressure contribution.
        return gamma * sum((m * s for m, s in zip(inverse[0], source)), I(0)) / 4

    positive, negative = live_rate(gap_plus), live_rate(gap_minus)
    assert positive.hi < 0 < negative.lo
    predicted_magnitude = (
        gamma * (inverse[0][0] - inverse[0][1]) * cos(kappa) * sin(I(shift)) / 2
    )
    assert predicted_magnitude.lo > 0
    assert (positive + predicted_magnitude).contains(0)
    assert (negative - predicted_magnitude).contains(0)


@pytest.mark.parametrize(
    "parameter",
    (
        "response_lower",
        "response_upper",
        "amplitude",
        "horizon",
        "preparation_radius",
        "observation_error",
    ),
)
@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), "1"))
def test_budget_scalars_are_admitted_before_arithmetic(parameter, invalid):
    with pytest.raises((TypeError, ValueError)):
        bound_prepared_matrix_response(**{parameter: invalid})


@pytest.mark.parametrize(
    "overrides",
    (
        {"response_lower": 0},
        {"response_lower": Q(1, 5)},
        {"response_upper": -1},
        {"amplitude": 0},
        {"horizon": 0},
        {"horizon": Q(1, 2)},
        {"preparation_radius": -1},
        {"observation_error": -1},
    ),
)
def test_invalid_proof_domains_reject(overrides):
    with pytest.raises(ValueError):
        bound_prepared_matrix_response(**overrides)


def test_normalized_exact_budgets_preserve_the_declared_certificate(prepared_bound):
    np = pytest.importorskip("numpy")
    payload = Q(np.int64(1), np.int64(2**26))
    assert bound_prepared_matrix_response(preparation_radius=payload) == prepared_bound
    assert (
        bound_prepared_matrix_response(observation_error=float(payload))
        == prepared_bound
    )
    zero_source = bound_prepared_matrix_response(
        preparation_radius=0, observation_error=0
    )
    assert zero_source.identity_certified and zero_source.response_separated
    assert zero_source.response_gap > prepared_bound.response_gap
