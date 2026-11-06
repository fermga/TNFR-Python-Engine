"""Finite-input software controls, separate from a variational response.

The module fixture reuses exactly three declared canonical input evaluations.
It checks their arithmetic widths, not a continuous error or retention claim.
All action, source, and derivative identities use independent synthetic data.
"""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from math import factorial
from time import perf_counter

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._rational_interval import I, pi_interval, sqrt
from tnfr.physics import _localized_variational_inputs as inputs
from tnfr.utils.io import json_dumps


@pytest.fixture(scope="module", autouse=True)
def no_prepared_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _matrix_product(left, right):
    return tuple(
        tuple(sum((a * b for a, b in zip(row, column)), Q(0)) for column in zip(*right))
        for row in left
    )


def _matrix_vector(matrix, vector):
    return tuple(sum((a * b for a, b in zip(row, vector)), Q(0)) for row in matrix)


def _dense_cycle(size, coefficient):
    incidence = tuple(
        tuple(
            Q(int(column == row) - int(column == (row + 1) % size))
            for column in range(size)
        )
        for row in range(size)
    )
    transpose = tuple(zip(*incidence))
    laplacian = _matrix_product(transpose, incidence)
    weighted = tuple(
        tuple(value * coefficient[index] for value in row)
        for index, row in enumerate(incidence)
    )
    feedback = _matrix_product(transpose, weighted)
    return (
        tuple(tuple(value / 2 for value in row) for row in laplacian),
        tuple(tuple(value / 2 for value in row) for row in feedback),
        incidence,
        transpose,
    )


def _norm_upper(values):
    """Normalize before squaring; the interval grid has absolute precision."""
    values = tuple(values)
    scale = max((abs(value) for value in values), default=Q(0))
    if not scale:
        return Q(0)
    return scale * sqrt(I(sum(((value / scale) ** 2 for value in values), Q(0)))).hi


def _radius_upper(values):
    return _norm_upper(value.radius for value in values)


def _complex_product(left, right):
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _complex_power(value, exponent):
    factor = value if exponent >= 0 else (value[0], -value[1])
    result = Q(1), Q(0)
    for _ in range(abs(exponent)):
        result = _complex_product(result, factor)
    return result


def _exact_rotated_profile(root, translation, coefficients, spacing):
    phase = _complex_product(root, translation)
    value, derivative = coefficients[0], Q(0)
    for mode, coefficient in enumerate(coefficients[1:], 1):
        cosine, sine = _complex_power(phase, mode)
        value += 2 * coefficient * cosine
        derivative -= 2 * mode * spacing * coefficient * sine
    return value, derivative


def _exact_reconstruction(profile, spatial, carrier, velocity, frequency, correctors):
    value, derivative = [Q(0), Q(0)], [Q(0), Q(0)]
    for harmonic, amplitude, amplitude_rate in (
        (1, profile, velocity * spatial),
        *(
            (
                harmonic,
                coefficient * profile**3,
                3 * coefficient * profile**2 * velocity * spatial,
            )
            for harmonic, coefficient in zip((-1, 3, -3), correctors)
        ),
    ):
        phase = _complex_power(carrier, harmonic)
        rate = _complex_product(
            (amplitude_rate, harmonic * frequency * amplitude), phase
        )
        for component in range(2):
            value[component] += amplitude * phase[component]
            derivative[component] += rate[component]
    return tuple(value + derivative)


@pytest.fixture(scope="module")
def canonical_factory():
    started = perf_counter()
    metadata, evaluate = inputs._prepare_localized_variational_inputs()
    return metadata, evaluate, perf_counter() - started


@pytest.fixture(scope="module")
def canonical_points(canonical_factory):
    metadata, evaluate, _ = canonical_factory
    points, diagnostics = [], []
    for fraction in (Q(0), Q(1, 2), Q(1)):
        time = metadata.horizon_bounds * fraction
        started = perf_counter()
        rows = evaluate(time)
        elapsed = perf_counter() - started
        radii = {
            "state": _radius_upper(rows.x + rows.theta),
            "time_derivative": _radius_upper(rows.dx_dt + rows.dtheta_dt),
            "phase_gap": _radius_upper(rows.phase_gap),
            "source": _radius_upper(rows.source_x + rows.source_theta),
        }
        points.append((fraction, rows, radii))
        diagnostics.append(
            {
                "normalized_time": str(fraction),
                "time_bounds": [str(time.lo), str(time.hi)],
                "elapsed_seconds": elapsed,
                "radius_upper_bounds": {
                    key: str(value) for key, value in radii.items()
                },
            }
        )
    return tuple(points), tuple(diagnostics)


def test_sparse_action_matches_independent_dense_incidence_and_keeps_means():
    size, sigma = 7, Q(1, 32)
    coefficient = tuple(Q(index + 1, 32) for index in range(size))
    form = tuple(Q(index - 3, 5) for index in range(size))
    phase = tuple(Q(index**2 % 11 - 4, 9) for index in range(size))
    principal, feedback, _, _ = _dense_cycle(size, coefficient)
    ax = _matrix_vector(principal, form)
    atheta = _matrix_vector(principal, phase)
    beta = _matrix_vector(feedback, phase)
    expected = tuple(
        -sigma * x - theta + value for x, theta, value in zip(ax, atheta, beta)
    )
    observed = inputs._apply_variation(
        tuple(map(I, form)), tuple(map(I, phase)), tuple(map(I, coefficient)), I(sigma)
    )
    assert all(box.contains(value) for box, value in zip(observed[0], expected))
    assert all(box.contains(value) for box, value in zip(observed[1], ax))
    assert all(sum(row, I(0)).contains(0) for row in observed)
    constants = inputs._apply_variation(
        (I(1),) * size, (I(-2),) * size, tuple(map(I, coefficient)), I(sigma)
    )
    assert all(box.lo == box.hi == 0 for row in constants for box in row)


@pytest.mark.parametrize("gap", [Q(0), Q(1, 16), Q(-2, 7), Q(3, 2)])
def test_polynomials_keep_exact_degrees_and_signs(gap):
    expected_sine = sum(
        (Q((-1) ** j, factorial(2 * j + 1)) * gap ** (2 * j + 1) for j in range(1, 9)),
        Q(0),
    )
    expected_cosine = sum(
        (Q((-1) ** (j + 1), factorial(2 * j)) * gap ** (2 * j) for j in range(1, 9)),
        Q(0),
    )
    sine, cosine = inputs._sine_cosine_polynomials(I(gap))
    assert sine.contains(expected_sine)
    assert cosine.contains(expected_cosine)


def test_source_uses_both_signed_rows_and_the_same_incidence_orientation():
    size, sigma = 5, Q(1, 64)
    x = tuple(Q(index - 2, 8) for index in range(size))
    theta = tuple(Q(index**2 - 3, 32) for index in range(size))
    dx = tuple(Q(2 - index, 16) for index in range(size))
    dtheta = tuple(Q(index % 2, 8) for index in range(size))
    principal, _, incidence, transpose = _dense_cycle(size, (Q(0),) * size)
    gap = _matrix_vector(incidence, theta)
    sine = tuple(
        sum(
            (
                Q((-1) ** j, factorial(2 * j + 1)) * value ** (2 * j + 1)
                for j in range(1, 9)
            ),
            Q(0),
        )
        for value in gap
    )
    divergence = _matrix_vector(transpose, sine)
    ax, atheta = _matrix_vector(principal, x), _matrix_vector(principal, theta)
    expected_x = tuple(
        rate + sigma * form + phase + nonlinear / 2
        for rate, form, phase, nonlinear in zip(dx, ax, atheta, divergence)
    )
    expected_theta = tuple(rate - form for rate, form in zip(dtheta, ax))
    rows = inputs._build_rows(
        I(0),
        tuple(map(I, x)),
        tuple(map(I, theta)),
        tuple(map(I, dx)),
        tuple(map(I, dtheta)),
        I(sigma),
    )
    assert all(box.contains(value) for box, value in zip(rows.phase_gap, gap))
    assert all(box.contains(value) for box, value in zip(rows.source_x, expected_x))
    assert all(
        box.contains(value) for box, value in zip(rows.source_theta, expected_theta)
    )
    perturbation = [Q(index, 8) for index in range(size)]
    old = rows.action(perturbation, theta)
    old_coefficients = rows.cosine_coefficient
    try:
        object.__setattr__(rows, "cosine_coefficient", (I(0),) * size)
        assert rows.action(perturbation, theta) == old
    finally:
        object.__setattr__(rows, "cosine_coefficient", old_coefficients)
    perturbation[0] = True
    with pytest.raises((TypeError, ValueError)):
        rows.action(perturbation, theta)
    with pytest.raises(FrozenInstanceError):
        rows.source_theta = ()


def test_reconstruction_derivative_matches_independent_complex_harmonics():
    profile, spatial, velocity, frequency = Q(2, 7), Q(-1, 9), Q(3, 8), Q(5, 11)
    carrier = Q(3, 5), Q(4, 5)
    correctors = Q(-2, 13), Q(3, 17), Q(-1, 19)

    expected = _exact_reconstruction(
        profile, spatial, carrier, velocity, frequency, correctors
    )
    observed = inputs._reconstruct_row(
        I(profile),
        I(spatial),
        tuple(map(I, carrier)),
        I(velocity),
        I(frequency),
        tuple(map(I, correctors)),
    )
    assert all(box.contains(target) for box, target in zip(observed, expected))


def test_profile_recurrence_matches_independent_triple_angle_identities():
    cosine, sine, spacing = Q(3, 5), Q(4, 5), Q(2, 7)
    coefficients = Q(1, 9), Q(1, 10), Q(-1, 11), Q(1, 12)
    cosine_modes = cosine, 2 * cosine**2 - 1, 4 * cosine**3 - 3 * cosine
    sine_modes = sine, 2 * sine * cosine, 3 * sine - 4 * sine**3
    expected = coefficients[0] + 2 * sum(
        value * coefficient
        for value, coefficient in zip(cosine_modes, coefficients[1:])
    )
    derivative = (
        -2
        * spacing
        * sum(
            mode * value * coefficient
            for mode, value, coefficient in zip(
                range(1, 4), sine_modes, coefficients[1:]
            )
        )
    )
    profile, observed_derivative = inputs._profile_at_phase(
        (I(cosine), I(sine)), tuple(map(I, coefficients)), I(spacing)
    )
    assert profile.contains(expected)
    assert observed_derivative.contains(derivative)


def test_separated_profile_keeps_temporal_modes_and_derivatives_beyond_spatial_aliases():
    roots = ((Q(1), Q(0)), (Q(0), Q(1)), (Q(-1), Q(0)), (Q(0), Q(-1)))
    coefficients = (Q(1, 7), Q(1, 16), Q(0), Q(-1, 32), Q(1, 64), Q(-1, 16))
    translation, spacing = (Q(3, 5), Q(4, 5)), Q(2, 9)
    basis = inputs._prepare_profile_basis(
        tuple(tuple(map(I, root)) for root in roots),
        tuple(map(I, coefficients)),
        I(spacing),
    )
    temporal = inputs._temporal_profile_vector(tuple(map(I, translation)), 5)
    expected = tuple(
        _exact_rotated_profile(root, translation, coefficients, spacing)
        for root in roots
    )
    for row, targets in zip(basis, expected):
        observed = inputs._profile_from_basis(row, temporal)
        assert all(box.contains(target) for box, target in zip(observed, targets))
    # Modes1 and5 share every spatial root, but their time phases differ.
    assert _complex_power(translation, 1) != _complex_power(translation, 5)
    assert expected[0][1] != -2 * spacing * sum(
        (mode % 4) * coefficient * _complex_power(translation, mode)[1]
        for mode, coefficient in enumerate(coefficients[1:], 1)
    )
    assert len(basis) == 4 and all(len(row) == 11 for pair in basis for row in pair)
    assert all(
        type(value) is int
        for pair in basis
        for row in pair
        for entry in row
        for value in entry
    )


def test_prepared_profile_basis_detaches_mutable_inputs_and_is_immutable():
    roots = [[I(1), I(0)], [I(-1), I(0)]]
    coefficients = [I(Q(1, 4)), I(Q(-1, 8)), I(Q(1, 16))]
    basis = inputs._prepare_profile_basis(roots, coefficients, I(Q(2, 7)))
    temporal = inputs._temporal_profile_vector((I(Q(3, 5)), I(Q(4, 5))), 2)
    before = tuple(inputs._profile_from_basis(row, temporal) for row in basis)
    roots[0][0], coefficients[1] = I(99), I(99)
    assert tuple(inputs._profile_from_basis(row, temporal) for row in basis) == before
    with pytest.raises(TypeError):
        basis[0] = ()
    with pytest.raises(TypeError):
        basis[0][0][0][0] = 1


def test_separated_evaluator_both_rows_rates_and_source_match_exact_harmonics(
    monkeypatch,
):
    roots = ((Q(1), Q(0)), (Q(0), Q(1)), (Q(-1), Q(0)), (Q(0), Q(-1)))
    coefficients = (Q(1, 8), Q(1, 32), Q(-1, 64), Q(1, 128), Q(1, 256), Q(-1, 512))
    spacing, velocity, frequency, sigma = Q(2, 9), Q(3, 8), Q(5, 11), Q(1, 32)
    correctors = (Q(-2, 13), Q(3, 17), Q(-1, 19))
    translation, rotation = (Q(3, 5), Q(4, 5)), (Q(5, 13), Q(12, 13))
    admitted_roots = tuple(tuple(map(I, root)) for root in roots)
    basis = inputs._prepare_profile_basis(
        admitted_roots, tuple(map(I, coefficients)), I(spacing)
    )
    evaluate = inputs._make_input_evaluator(
        basis,
        admitted_roots,
        I(spacing),
        I(velocity),
        I(frequency),
        tuple(map(I, correctors)),
        I(sigma),
        I(0, 1),
    )
    trigonometric_calls = []
    arithmetic_calls = {"complex_products": 0, "dots": 0}
    original_product, original_dot = inputs._complex_multiply, inputs._dot_encoded_rows

    def counted_product(left, right):
        arithmetic_calls["complex_products"] += 1
        return original_product(left, right)

    def counted_dot(left, right):
        arithmetic_calls["dots"] += 1
        return original_dot(left, right)

    def supplied_sine_cosine(argument):
        trigonometric_calls.append(argument)
        cosine, sine = (translation, rotation)[len(trigonometric_calls) - 1]
        return I(sine), I(cosine)

    monkeypatch.setattr(inputs, "_reduced_sine_cosine", supplied_sine_cosine)
    monkeypatch.setattr(inputs, "_complex_multiply", counted_product)
    monkeypatch.setattr(inputs, "_dot_encoded_rows", counted_dot)
    rows = evaluate(Q(1, 8))
    assert len(trigonometric_calls) == 2
    assert arithmetic_calls == {"complex_products": 5 - 1 + 3 * 4, "dots": 6 * 4}
    expected = tuple(
        _exact_reconstruction(
            *_exact_rotated_profile(root, translation, coefficients, spacing),
            _complex_product(root, rotation),
            velocity,
            frequency,
            correctors,
        )
        for root in roots
    )
    x, theta, dx, dtheta = tuple(zip(*expected))
    for observed, targets in zip(
        (rows.x, rows.theta, rows.dx_dt, rows.dtheta_dt), (x, theta, dx, dtheta)
    ):
        assert all(box.contains(target) for box, target in zip(observed, targets))
    principal, _, incidence, transpose = _dense_cycle(4, (Q(0),) * 4)
    gaps = _matrix_vector(incidence, theta)
    nonlinear = _matrix_vector(
        transpose,
        tuple(
            sum(
                (
                    Q((-1) ** j, factorial(2 * j + 1)) * gap ** (2 * j + 1)
                    for j in range(1, 9)
                ),
                Q(0),
            )
            for gap in gaps
        ),
    )
    ax, atheta = _matrix_vector(principal, x), _matrix_vector(principal, theta)
    assert all(
        box.contains(rate + sigma * form + phase + term / 2)
        for box, rate, form, phase, term in zip(
            rows.source_x, dx, ax, atheta, nonlinear
        )
    )
    assert all(
        box.contains(rate - form)
        for box, rate, form in zip(rows.source_theta, dtheta, ax)
    )


def test_evaluator_rejects_invalid_time_before_any_numerical_kernel(monkeypatch):
    roots = ((I(1), I(0)), (I(-1), I(0)))
    basis = inputs._prepare_profile_basis(roots, (I(1), I(Q(1, 8))), I(1))
    evaluate = inputs._make_input_evaluator(
        basis, roots, I(1), I(1), I(1), (I(0),) * 3, I(0), I(0, 1)
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid time reached a numerical kernel or cache")

    for name in (
        "_reduced_sine_cosine",
        "_temporal_profile_vector",
        "_carrier_reconstruction_weights",
        "_profile_from_basis",
        "_build_rows",
        "_canonical_profile_basis",
        "_spatial_roots",
    ):
        monkeypatch.setattr(inputs, name, forbidden)
    for bad in (
        True,
        False,
        float("nan"),
        float("inf"),
        "0",
        None,
        -1,
        I(-1, 0),
        I(0, 2),
    ):
        with pytest.raises((TypeError, ValueError)):
            evaluate(bad)


def test_agm_enclosure_matches_independent_positive_integral_series():
    # Integrating the binomial series gives K/(pi/2)=sum c_n^2 m^n.
    term = total = Q(1)
    for index in range(1, 161):
        term *= Q((2 * index - 1) ** 2, 8 * index**2)
        total += term
    first_omitted = term * Q(321**2, 8 * 161**2)
    pi = pi_interval()
    independent = I(pi.lo * total / 2, pi.hi * (total + 2 * first_omitted) / 2)
    observed = inputs._agm_elliptic_parameter_half()
    assert observed.lo <= independent.hi and independent.lo <= observed.hi
    assert observed.width < Q(1, 2**110)
    assert observed.lo > pi.hi / 2


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), "0", None])
def test_primitive_scalar_admission_precedes_interval_materialization(bad):
    with pytest.raises((TypeError, ValueError)):
        inputs._interval(bad, "synthetic time")


@pytest.mark.parametrize(
    "bad", [{0: 1, 1: 2}, {1, 2}, (value for value in (1, 2)), "12"]
)
def test_vector_admission_rejects_unordered_or_unbounded_rows(bad):
    with pytest.raises((TypeError, ValueError)):
        inputs._admit_vector(bad, 2, "synthetic state")


def test_vector_admission_rebuilds_and_detaches_mutable_input():
    source = [Q(1, 3), Q(-2, 7)]
    admitted = inputs._admit_vector(source, 2, "synthetic state")
    source[0] = 99
    assert admitted[0].contains(Q(1, 3)) and not admitted[0].contains(99)
    with pytest.raises(ValueError):
        inputs._admit_vector([0], 2, "synthetic state")


def test_canonical_metadata_and_time_domain_are_explicit(canonical_factory):
    metadata, evaluate, _ = canonical_factory
    assert metadata.node_count == 1024 and metadata.node_order == tuple(range(1024))
    assert metadata.edges[-1] == (1023, 0)
    assert metadata.form_loss == Q(1, 2**20)
    assert metadata.exchange == 1 - metadata.form_loss
    assert metadata.profile_cutoff == 24 and len(metadata.profile_coefficients) == 25
    assert metadata.sine_degree == 17 and metadata.cosine_degree == 16
    assert metadata.method == "canonical_localized_variational_inputs_dyadic128_v2"
    assert metadata.temporal_profile_modes == 24 and metadata.carrier_classes == 8
    assert metadata.complex_products_per_evaluation == 47
    assert metadata.profile_dot_products_per_evaluation == 2048
    assert metadata.reconstruction_dot_products_per_evaluation == 4096
    assert metadata.retained_profile_basis_rows == 2048
    assert metadata.retained_profile_basis_entries == 2048 * 49
    assert metadata.integers_per_encoded_basis_entry == 3
    assert metadata.dense_matrix_entries == 0
    assert metadata.evaluation_time_bounds.contains(metadata.horizon_bounds)
    for bad in (True, float("nan"), -1, I(-1, 0), metadata.horizon_bounds.hi + 1):
        with pytest.raises((TypeError, ValueError)):
            evaluate(bad)
    with pytest.raises(FrozenInstanceError):
        metadata.profile_cutoff = 1


def test_three_declared_points_meet_only_their_arithmetic_width_budget(
    canonical_points, canonical_factory, record_property
):
    points, diagnostics = canonical_points
    metadata, _, factory_seconds = canonical_factory
    report = {
        "scope": "Three finite-input software checks; no correction or nonlinear response",
        "method": metadata.method,
        "interval_method": metadata.interval_method,
        "node_count": metadata.node_count,
        "profile_cutoff": metadata.profile_cutoff,
        "sine_degree": metadata.sine_degree,
        "cosine_degree": metadata.cosine_degree,
        "agm_steps": metadata.agm_steps,
        "complex_products_per_evaluation": metadata.complex_products_per_evaluation,
        "profile_dot_products_per_evaluation": metadata.profile_dot_products_per_evaluation,
        "reconstruction_dot_products_per_evaluation": metadata.reconstruction_dot_products_per_evaluation,
        "retained_profile_basis_entries": metadata.retained_profile_basis_entries,
        "factory_seconds": factory_seconds,
        "point_radius_target": str(Q(1, 2**80)),
        "points": diagnostics,
    }
    record_property("input_preflight", json_dumps(report))
    assert tuple(point[0] for point in points) == (Q(0), Q(1, 2), Q(1))
    for _, rows, radii in points:
        assert len(rows.x) == len(rows.theta) == len(rows.source_x) == 1024
        assert all(value < Q(1, 2**80) for value in radii.values())


def test_componentwise_comparison_loses_the_actual_skew_cancellation():
    size, sigma = 6, Q(1, 16)
    principal, _, _, _ = _dense_cycle(size, (Q(0),) * size)
    tangent = tuple(
        tuple(-sigma * principal[i][j] for j in range(size))
        + tuple(-principal[i][j] for j in range(size))
        for i in range(size)
    ) + tuple(tuple(principal[i]) + (Q(0),) * size for i in range(size))
    comparison = tuple(
        tuple(value if i == j else abs(value) for j, value in enumerate(row))
        for i, row in enumerate(tangent)
    )
    form_mean, phase_mean = (Q(1),) * size + (Q(0),) * size, (Q(0),) * size + (
        Q(1),
    ) * size
    assert _matrix_vector(tangent, form_mean) == (Q(0),) * (2 * size)
    assert _matrix_vector(tangent, phase_mean) == (Q(0),) * (2 * size)
    assert _matrix_vector(comparison, form_mean) == tuple(
        2 * value for value in phase_mean
    )
    assert _matrix_vector(comparison, phase_mean) == tuple(
        2 * value for value in form_mean
    )


def test_tiny_norm_scaling_avoids_the_absolute_grid_square_floor():
    tiny = Q(1, 2**90)
    naive = sqrt(I(tiny) ** 2).hi
    assert naive >= Q(1, 2**64) > tiny
    assert _norm_upper((tiny,)) == tiny
    assert _norm_upper((3 * tiny, 4 * tiny)) >= 5 * tiny
    assert _norm_upper((3 * tiny, 4 * tiny)) < 6 * tiny


def test_three_zero_samples_do_not_certify_a_continuous_defect():
    def polynomial(value):
        return value**2 * (value - Q(1, 2)) ** 2 * (value - 1) ** 2

    def derivative(value):
        factors = value, value - Q(1, 2), value - 1
        return sum(
            (
                2
                * factor
                * factors[(index + 1) % 3] ** 2
                * factors[(index + 2) % 3] ** 2
                for index, factor in enumerate(factors)
            ),
            Q(0),
        )

    for sample in (Q(0), Q(1, 2), Q(1)):
        assert polynomial(sample) == derivative(sample) == 0
    assert polynomial(Q(1, 4)) == Q(9, 4096)
    assert derivative(Q(1, 4)) != 0
    # A constant spatial direction is annihilated by J+V. Along that direction
    # this polynomial's equation defect is its derivative, not its three samples.


def test_direct_taylor_defect_bound_preserves_the_integrated_derivative_factor():
    delta, amplitude = Q(1, 64), Q(2**40)
    # Every omitted c_n*s^n contributes exactly |c_n|*delta^n to the
    # integrated derivative estimate. Keeping an extra n would be incorrect.
    integrated_derivative = sum(
        (amplitude * degree * delta**degree / degree for degree in range(17, 81)),
        Q(0),
    )
    derivative_bound = amplitude * delta**17 / (1 - delta)
    assert integrated_derivative < derivative_bound
    integrated_state = sum(
        (amplitude * delta ** (degree + 1) / (degree + 1) for degree in range(17, 81)),
        Q(0),
    )
    assert integrated_state < delta * derivative_bound
    all_panels = amplitude * 2**19 * delta**16 * (1 + 3 * delta) / (1 - delta)
    assert all_panels == Q(67, 63 * 2**37)
    assert all_panels < Q(1, 2**36) < Q(1, 2**34)
    assert 27 * (2**35 + 96) < amplitude


def test_direct_taylor_sufficient_grid_has_excessive_coefficient_work():
    minimum_panels = Q(64 * 1024**2, 15)
    assert minimum_panels > 2**22
    assert 17 * 2048 * minimum_panels > 2**37
    # These count entries across a particular fixed-grid realization. They
    # are not a memory lower bound or a complexity lower bound for the ODE.
