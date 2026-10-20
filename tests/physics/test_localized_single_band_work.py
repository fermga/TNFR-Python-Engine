"""Static source and regional-work controls for one carrier with two lobes.

These controls admit algebra and observations, not a localized trajectory,
source materialization, independently evolving constituents or formation.
"""

from fractions import Fraction as Q

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin, sqrt
from tnfr.physics.relational_sine_comparison import _sine_rates


@pytest.fixture(scope="module", autouse=True)
def no_prepared_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _difference(values):
    return tuple(
        value - values[(j + 1) % len(values)] for j, value in enumerate(values)
    )


def _laplacian(values):
    return tuple(
        2 * value - values[j - 1] - values[(j + 1) % len(values)]
        for j, value in enumerate(values)
    )


def _assert_zero_interval(value):
    assert value.contains(0)
    assert value.abs_max < Q(1, 10**30)


@pytest.fixture(scope="module")
def source_coefficients():
    # Count the Fourier coefficients of the squared finite geometric sum.
    order = 16
    counts = {mode: 0 for mode in range(1 - order, order)}
    for left in range(order):
        for right in range(order):
            counts[left - right] += 1
    return {mode: Q(count, order) for mode, count in counts.items()}


def test_same_band_antipodal_lobes_overlap_and_phase_changes_total_norm(
    source_coefficients,
):
    coefficients = source_coefficients
    square_sum = sum(value**2 for value in coefficients.values())
    alternating_sum = sum(
        (-1) ** abs(mode) * value**2 for mode, value in coefficients.items()
    )
    assert square_sum == Q(171, 16)
    assert alternating_sum == Q(1, 16)
    overlap = alternating_sum / square_sum
    assert overlap == Q(1, 171)
    # Both translated lobes occupy the same band: orthogonal-band reasoning fails.
    modes = {128 + mode for mode in coefficients}
    assert modes == set(range(113, 144)) and 0 not in modes
    angle = sp.Symbol("psi", real=True)
    norm_squared = 2 + 2 * sp.Rational(overlap.numerator, overlap.denominator) * sp.cos(
        angle
    )
    assert sp.simplify(norm_squared.subs(angle, -angle) - norm_squared) == 0
    assert norm_squared.subs(angle, 0) != norm_squared.subs(angle, sp.pi)
    assert norm_squared.subs(angle, sp.pi) == sp.Rational(340, 171)
    assert norm_squared.subs(angle, 0) == sp.Rational(344, 171)


def test_whole_source_error_preserves_signal_and_absence_without_lobe_projections():
    size, order, radius = 1024, 16, 96
    tail = Q(size**3, 16 * order**3 * (radius - 1) ** 3)
    assert tail == Q(16384, 857375) < Q(1, 49)
    assert Q(141, 100) ** 2 < Q(340, 171)
    relative_error = Q(1, 1000)
    lower_norm = Q(141, 100) - relative_error
    upper_window = 1 + Q(1, 7) + relative_error
    assert lower_norm == Q(1409, 1000) > 1
    assert upper_window == Q(8007, 7000)
    screen_bound = (upper_window / lower_norm) ** 2
    assert screen_bound == Q(8007, 9863) ** 2 < Q(2, 3)
    assert Q(3, 4) - screen_bound > Q(1, 12)
    assert Q(344, 171) < (Q(3, 2) - relative_error) ** 2
    # Geometric absence survives the full-state error; it is not obtained by
    # artificially assigning the same-band observation to separate constituents.
    assert 2 * radius < Q(size, 4)
    # Both initial regions retain a signal, despite destructive interference.
    assert Q(48, 49) > (Q(4, 5) + Q(1, 7) + relative_error) ** 2
    assert Q(2, 7) + relative_error < Q(3, 10)


def test_antipodal_window_exclusion_includes_fractional_centers():
    size, radius = 1024, 96

    def distance(left, right):
        delta = abs(left - right) % size
        return min(delta, size - delta)

    # Half-grid values exercise the boundary convention, while the triangle
    # inequality gives the general real-center proof used by the owner.
    centers = (Q(index, 2) for index in range(2 * size))
    for center in centers:
        distances = (distance(center, 0), distance(center, 512))
        assert sum(distances) == 512
        assert max(distances) >= 256 > 2 * radius
    assert distance(Q(1, 2), 0) == Q(1, 2)
    # Collocated source windows do not meet this absence argument.
    assert max(distance(0, source) for source in (0, 0)) < radius


def test_source_phase_chart_and_error_mean_are_admitted_independently(
    source_coefficients,
):
    square_sum = sum(value**2 for value in source_coefficients.values())
    peak_squared = sum(source_coefficients.values()) ** 2 / (1024 * square_sum)
    assert peak_squared == Q(4, 171) < Q(2, 13) ** 2
    relative_error, amplitude = Q(1, 1000), Q(1, 2**12)
    assert 2 * Q(2, 13) + relative_error < Q(1, 3)
    phase_gap = 2 * amplitude / 3
    assert phase_gap == Q(1, 6144) < 1 < pi_interval().lo / 2
    mean_bound = amplitude * relative_error / 32
    assert mean_bound == amplitude / 32000
    # The error ball includes a nonzero constant phase/form mode. A zero-mean
    # projection would remove admitted states and change the source class.
    constant_error = amplitude * relative_error / 32
    assert 1024 * constant_error**2 == (amplitude * relative_error) ** 2
    assert constant_error > 0


@pytest.fixture(scope="module")
def shared_state_rows():
    model = RelationalExchangeModel(
        storage_scale=1,
        epi_weight=Q(1, 2**20),
        phase_weight=1 - Q(1, 2**20),
        phase_domain="regular",
    )
    size = 8
    forms = (
        tuple(Q(j % 3 - 1, 13) for j in range(size)),
        tuple(Q((-1) ** j * (j + 1), 71) for j in range(size)),
    )
    phases = (
        tuple(Q(j % 4 - 2, 17) for j in range(size)),
        tuple(Q((-1) ** j, 23) for j in range(size)),
    )
    clock = pi_interval() / Q(model.phase_weight)
    sigma = pi_interval() * Q(model.epi_weight) / Q(model.phase_weight)

    def rows(form, phase):
        current = tuple(
            sin(I(phase[j - 1] - phase[j])) + sin(I(phase[(j + 1) % size] - phase[j]))
            for j in range(size)
        )
        evaluated = _sine_rates(
            model, (2,) * size, _laplacian(form), (1,) * size, current
        )
        return tuple(
            tuple(clock * value for value in evaluated[key])
            for key in ("form_rates", "phase_rates")
        )

    joined_form = tuple(a + b for a, b in zip(*forms))
    joined_phase = tuple(a + b for a, b in zip(*phases))
    return {
        "forms": forms,
        "phases": phases,
        "form": joined_form,
        "phase": joined_phase,
        "rows": rows(joined_form, joined_phase),
        "separate_rows": tuple(rows(form, phase) for form, phase in zip(forms, phases)),
        "sigma": sigma,
    }


@pytest.mark.parametrize("region", ((0,), (0, 1, 2), tuple(range(8))))
def test_physical_storage_rate_excess_over_actual_damped_tangent(
    shared_state_rows, region
):
    data = shared_state_rows
    form, phase, sigma = data["form"], data["phase"], data["sigma"]
    y, u = _difference(form), _difference(phase)
    actual_y_rate, actual_u_rate = map(_difference, data["rows"])
    linear_form = tuple(
        -sigma * lap_x / 2 - lap_phase / 2
        for lap_x, lap_phase in zip(_laplacian(form), _laplacian(phase))
    )
    linear_y_rate = _difference(linear_form)
    linear_u_rate = _difference(tuple(I(value) / 2 for value in _laplacian(form)))
    sine_gap = tuple(sin(I(value)) for value in u)
    actual = sum(
        y[j] * actual_y_rate[j] + sine_gap[j] * actual_u_rate[j] for j in region
    )
    linear = sum(
        y[j] * linear_y_rate[j] + sine_gap[j] * linear_u_rate[j] for j in region
    )
    tail = tuple(value - sine for value, sine in zip(u, sine_gap))
    expected = sum(y[j] * _laplacian(tail)[j] / 2 for j in region)
    _assert_zero_interval(actual - linear - expected)
    assert not expected.contains(0)
    for actual_rate, linear_rate in zip(actual_u_rate, linear_u_rate):
        _assert_zero_interval(actual_rate - linear_rate)


def test_shared_state_mixed_work_uses_the_total_storage_gradient(shared_state_rows):
    data = shared_state_rows
    y, u = _difference(data["form"]), _difference(data["phase"])
    component_gaps = tuple(_difference(phase) for phase in data["phases"])
    region = (0, 1, 2)
    separate_rates = tuple(
        tuple(a + b for a, b in zip(*rows)) for rows in zip(*data["separate_rows"])
    )
    actual_edge_rates = tuple(map(_difference, data["rows"]))
    separate_edge_rates = tuple(map(_difference, separate_rates))
    observed = sum(
        y[j] * (actual_edge_rates[0][j] - separate_edge_rates[0][j])
        + sin(I(u[j])) * (actual_edge_rates[1][j] - separate_edge_rates[1][j])
        for j in region
    )
    mixed = tuple(
        sin(I(a + b)) - sin(I(a)) - sin(I(b)) for a, b in zip(*component_gaps)
    )
    expected = -sum(y[j] * _laplacian(mixed)[j] / 2 for j in region)
    product = tuple(
        sin(I(a) / 2) * sin(I(b) / 2) * sin(I(a + b) / 2)
        for a, b in zip(*component_gaps)
    )
    factored = 2 * sum(y[j] * _laplacian(product)[j] for j in region)
    _assert_zero_interval(observed - expected)
    _assert_zero_interval(observed - factored)
    assert not observed.contains(0)
    # Subtracting the separate storage derivatives instead changes gradients.
    wrong_separate_storage = sum(
        sum(
            _difference(form)[j] * _difference(rows[0])[j]
            + sin(I(_difference(phase)[j])) * _difference(rows[1])[j]
            for j in region
        )
        for form, phase, rows in zip(
            data["forms"], data["phases"], data["separate_rows"]
        )
    )
    joined_storage = sum(
        y[j] * actual_edge_rates[0][j] + sin(I(u[j])) * actual_edge_rates[1][j]
        for j in region
    )
    assert not (joined_storage - wrong_separate_storage - observed).contains(0)


def test_nonlinear_work_has_internal_and_boundary_terms():
    size, region = 5, {0, 1, 2}
    y = sp.symbols("y:5", real=True)
    tail = sp.symbols("q:5", real=True)
    direct = sum(y[j] * _laplacian(tail)[j] / 2 for j in region)
    internal = 0
    boundary = 0
    for left in range(size):
        right = (left + 1) % size
        if left in region and right in region:
            internal += (y[left] - y[right]) * (tail[left] - tail[right]) / 2
        elif left in region:
            boundary += y[left] * (tail[left] - tail[right]) / 2
        elif right in region:
            boundary += y[right] * (tail[right] - tail[left]) / 2
    assert sp.expand(direct - internal - boundary) == 0
    assert sp.expand(direct - boundary) != 0


def test_full_sine_work_remainders_bound_the_cubic_rate_comparison(shared_state_rows):
    data = shared_state_rows
    x, u = data["form"], _difference(data["phase"])
    y = _difference(x)
    edge_region, node_region = {0, 1, 2}, {0, 1}
    projected_y = tuple(
        value if j in edge_region else Q(0) for j, value in enumerate(y)
    )
    storage_weights = _laplacian(projected_y)
    storage_exact = sum(
        weight * (gap - sin(I(gap))) / 2 for weight, gap in zip(storage_weights, u)
    )
    storage_cubic = sum(weight * gap**3 / 12 for weight, gap in zip(storage_weights, u))
    storage_remainder = sum(
        abs(weight) * abs(gap) ** 5 / 240 for weight, gap in zip(storage_weights, u)
    )
    assert (storage_exact - storage_cubic).abs_max < storage_remainder
    bx = tuple((Q(j in node_region) - Q(3, 4)) * value for j, value in enumerate(x))
    screen_weights = _difference(bx)
    screen_exact = sum(
        weight * (gap - sin(I(gap))) for weight, gap in zip(screen_weights, u)
    )
    screen_cubic = sum(weight * gap**3 / 6 for weight, gap in zip(screen_weights, u))
    screen_remainder = sum(
        abs(weight) * abs(gap) ** 5 / 120 for weight, gap in zip(screen_weights, u)
    )
    assert (screen_exact - screen_cubic).abs_max < screen_remainder
    scale = sp.Symbol("eps", real=True)
    assert sp.expand(
        sum(
            scale * weight * (scale * gap) ** 3 / 6
            for weight, gap in zip(screen_weights, u)
        )
    ) == scale**4 * sp.Rational(screen_cubic)
    assert storage_cubic != 0 and screen_cubic != 0


def test_separate_history_screen_uses_total_state_cross_terms(shared_state_rows):
    data = shared_state_rows
    weights = tuple(Q(j in {0, 1}) - Q(3, 4) for j in range(8))
    bx = tuple(weight * value for weight, value in zip(weights, data["form"]))
    derivative_weights = _difference(bx)
    gaps = tuple(_difference(phase) for phase in data["phases"])
    total_insertion = sum(
        weight * (a + b - sin(I(a + b)))
        for weight, a, b in zip(derivative_weights, *gaps)
    )
    separate_insertion = sum(
        weight * (a - sin(I(a)) + b - sin(I(b)))
        for weight, a, b in zip(derivative_weights, *gaps)
    )
    exact_mixed = 4 * sum(
        weight * sin(I(a) / 2) * sin(I(b) / 2) * sin(I(a + b) / 2)
        for weight, a, b in zip(derivative_weights, *gaps)
    )
    _assert_zero_interval(total_insertion - separate_insertion - exact_mixed)
    assert not exact_mixed.contains(0)
    cross_screen = 2 * sum(
        weight * (first_x * second_x + first_phase * second_phase)
        for weight, first_x, second_x, first_phase, second_phase in zip(
            weights, *data["forms"], *data["phases"]
        )
    )
    assert cross_screen != 0


def test_nominal_fejer_pair_has_nonzero_signed_local_work_without_a_trajectory(
    source_coefficients,
):
    coefficients = source_coefficients
    normalization = sqrt(I(1024 * sum(value**2 for value in coefficients.values())))
    angle = pi_interval() / 512

    def profile(position, antipodal=False):
        return (
            sum(
                value
                * ((-1) ** abs(mode) if antipodal else 1)
                * cos(mode * position * angle)
                for mode, value in coefficients.items()
            )
            / normalization
        )

    f0, f1, f2, g1 = profile(0), profile(1), profile(2), profile(1, antipodal=True)
    _assert_zero_interval(profile(0, antipodal=True))
    root_half, amplitude = 1 / sqrt(I(2)), Q(1, 2**12)
    first = f0 - root_half * f1
    second = root_half * g1
    upper = root_half * f1
    lower = f2 - root_half * f1
    assert first.lo > 0 and second.lo > 0 and (upper - lower).lo > 0

    def tail(value):
        return value - sin(value)

    upper_minus, upper_plus = tail(amplitude * (upper - second)), tail(
        amplitude * (upper + second)
    )
    lower_minus, lower_plus = tail(amplitude * (lower - second)), tail(
        amplitude * (lower + second)
    )
    factored = (
        amplitude
        / 2
        * (
            first * (3 * (upper_minus - upper_plus) + lower_minus - lower_plus)
            - second * (upper_minus + upper_plus - lower_minus - lower_plus)
        )
    )
    assert factored.hi < 0
    # Direct edge-row contraction uses Dq_j=q_j-q_(j+1). The four
    # retained edges suffice for the fixed region {-1,0}; no field is evolved.
    form_gaps = (-amplitude * (first - second), amplitude * (first + second))
    phase_gaps = tuple(
        -amplitude * value
        for value in (lower + second, upper - second, upper + second, lower - second)
    )
    tails = tuple(tail(value) for value in phase_gaps)
    direct = sum(
        form_gaps[j] * (2 * tails[j + 1] - tails[j] - tails[j + 2]) / 2
        for j in range(2)
    )
    _assert_zero_interval(direct - factored)
    assert direct.hi < 0
    # Conjugate reflection keeps the primitive resources and reverses this
    # signed instantaneous contribution. No robustness to arbitrary r follows.
    reflected = sum(
        -form_gaps[1 - j] * (2 * tails[2 - j] - tails[3 - j] - tails[1 - j]) / 2
        for j in range(2)
    )
    _assert_zero_interval(direct + reflected)
    assert reflected.lo > 0


@pytest.mark.parametrize("region", ((0,), (0, 1, 2)))
def test_total_field_localization_gap_rate_retains_actual_rows_and_linear_drift(
    shared_state_rows, region
):
    data = shared_state_rows
    x, theta = data["form"], data["phase"]
    sigma = data["sigma"]
    weights = tuple(Q(j in region) - Q(3, 4) for j in range(len(x)))
    actual = 2 * sum(
        weight * (form * form_rate + phase * phase_rate)
        for weight, form, phase, form_rate, phase_rate in zip(
            weights, x, theta, *data["rows"]
        )
    )
    ax, atheta = tuple(value / 2 for value in _laplacian(x)), tuple(
        value / 2 for value in _laplacian(theta)
    )
    bx = tuple(weight * value for weight, value in zip(weights, x))
    abx = tuple(value / 2 for value in _laplacian(bx))
    linear = 2 * (
        -sigma * sum(form * weight * lap for form, weight, lap in zip(x, weights, ax))
        + sum(
            phase * (weight * lap - other)
            for phase, weight, lap, other in zip(theta, weights, ax, abx)
        )
    )
    nonlinear = sum(
        gap * (phase_gap - sin(I(phase_gap)))
        for gap, phase_gap in zip(_difference(bx), _difference(theta))
    )
    _assert_zero_interval(actual - linear - nonlinear)
    assert not nonlinear.contains(0)
    assert not linear.contains(0)
    # Using only the nonlinear insertion omits an actual, nonzero contribution
    # from the same-source damped tangent and its regional boundary exchange.
    assert not (actual - nonlinear).contains(0)
    assert any(value != 0 for value in atheta)


def test_initial_absence_supplies_only_a_necessary_signed_prefix_work_threshold():
    norm_squared = sp.Symbol("N0", positive=True)
    fraction, target = sp.Rational(2, 3), sp.Rational(3, 4)
    gap_upper = (fraction - target) * norm_squared
    assert gap_upper == -norm_squared / 12
    # Entry requires V(T)>=0 while V(0)<gap_upper. The fundamental theorem
    # then requires integral V' > N0/12; it is not an endpoint prediction.
    required_gain = -gap_upper
    assert required_gain.is_positive
    signal_at_entry = sp.Symbol("NT", nonnegative=True)
    assert (target * signal_at_entry - target * signal_at_entry) == 0
    # V=0 at the zero signal too, so the independent norm floor is essential.
    assert signal_at_entry.subs(signal_at_entry, 0) == 0
    time = sp.Symbol("t", real=True)
    comparison = norm_squared * (-sp.Rational(1, 8) + time - time**2)
    gained = sp.integrate(sp.diff(comparison, time), (time, 0, sp.Rational(1, 2)))
    assert gained > required_gain
    assert comparison.subs(time, sp.Rational(1, 2)).is_positive
    assert comparison.subs(time, 1).is_negative
    # This scalar calculus countercontrol is not a TNFR trajectory: a positive
    # accumulated prefix by itself supplies no later maintenance statement.


def test_conjugate_reflection_pairs_the_actual_same_carrier_sources_and_storage():
    f, g, carrier, relative = sp.symbols("f g k psi", real=True)
    source = sp.Matrix(
        [
            f * sp.cos(carrier) + g * sp.cos(carrier + relative),
            f * sp.sin(carrier) + g * sp.sin(carrier + relative),
        ]
    )
    reflected = sp.diag(1, -1) * source.subs(carrier, -carrier)
    target = source.subs(relative, -relative)
    assert all(sp.trigsimp(value) == 0 for value in reflected - target)
    total_intensity = sp.trigsimp(source.dot(source).expand(trig=True))
    assert (
        sp.trigsimp(total_intensity - f**2 - g**2 - 2 * f * g * sp.cos(relative)) == 0
    )
    # Even profile values are reflected; both full primitive channels remain.
    form_gaps = sp.symbols("y:5", real=True)
    phase_gaps = sp.symbols("u:5", real=True)
    original_storage = sum(
        y**2 / 2 + 1 - sp.cos(u) for y, u in zip(form_gaps, phase_gaps)
    )
    reflected_storage = sum(
        (-form_gaps[(-j - 1) % 5]) ** 2 / 2 + 1 - sp.cos(phase_gaps[(-j - 1) % 5])
        for j in range(5)
    )
    assert sp.expand(original_storage - reflected_storage) == 0


def test_actual_damped_linear_control_can_increase_regional_physical_storage():
    # This is an arbitrary admitted static cycle state, not the Fejer source.
    amplitude, sigma = sp.symbols("a sigma", positive=True)
    form = (0, 0, amplitude, 0, 0, 0, 0, 0)
    phase = (0, -amplitude, 0, 0, 0, 0, 0, 0)
    y, u = _difference(form), _difference(phase)
    linear_y = tuple(
        -sigma * first / 2 - second / 2
        for first, second in zip(_laplacian(y), _laplacian(u))
    )
    linear_u = tuple(value / 2 for value in _laplacian(y))
    regional_rate = sp.expand(y[0] * linear_y[0] + sp.sin(u[0]) * linear_u[0])
    assert regional_rate == amplitude * sp.sin(amplitude) / 2
    assert regional_rate.subs(amplitude, sp.pi / 6).is_positive
    assert sp.diff(regional_rate, sigma) == 0
    assert sum(y) == sum(u) == 0
    global_quadratic_rate = sp.expand(
        sum(a * da + b * db for a, da, b, db in zip(y, linear_y, u, linear_u))
    )
    assert global_quadratic_rate.is_negative


def test_conjugate_reflection_matches_resources_and_reverses_only_conservative_flux():
    size = 5
    y = sp.symbols("y:5", real=True)
    sine_gap = sp.symbols("s:5", real=True)
    sigma = sp.Symbol("sigma", positive=True)
    reflected_y = tuple(-y[(-j - 1) % size] for j in range(size))
    reflected_sine = tuple(sine_gap[(-j - 1) % size] for j in range(size))

    def reflection(j):
        return (-j - 1) % size

    region = {0, 4}  # The edge reflection preserves this observation.
    assert {reflection(j) for j in region} == region

    def fluxes(form_gap, current, j, k):
        return (
            (current[j] * form_gap[k] - form_gap[j] * current[k]) / 2,
            sigma * (form_gap[j] ** 2 - form_gap[k] ** 2) / 4,
        )

    conservative, reflected_conservative, loss_flux, reflected_loss_flux = (0,) * 4
    for j in region:
        for k in ((j - 1) % size, (j + 1) % size):
            if k not in region:
                first, second = fluxes(y, sine_gap, j, k)
                other_first, other_second = fluxes(reflected_y, reflected_sine, j, k)
                conservative += first
                reflected_conservative += other_first
                loss_flux += second
                reflected_loss_flux += other_second
    assert sp.expand(conservative + reflected_conservative) == 0
    assert sp.expand(loss_flux - reflected_loss_flux) == 0
    assert sp.expand(loss_flux) != 0
    assert (
        sp.expand(
            conservative + loss_flux + reflected_conservative + reflected_loss_flux
        )
        != 0
    )
    loss = sum(value**2 for value in _difference(y))
    reflected_loss = sum(value**2 for value in _difference(reflected_y))
    assert sp.expand(loss - reflected_loss) == 0


def test_matched_preparations_do_not_supply_a_symmetry_of_the_damped_flow():
    eigenvalue, sigma = sp.symbols("lambda sigma", positive=True)
    tangent = eigenvalue * sp.Matrix([[-sigma, -1], [1, 0]])
    conjugation = sp.diag(1, -1)
    multiplication_by_i = sp.Matrix([[0, -1], [1, 0]])
    assert tangent * conjugation - conjugation * tangent == 2 * eigenvalue * sp.Matrix(
        [[0, 1], [1, 0]]
    )
    assert tangent * conjugation + conjugation * tangent == sp.diag(
        -2 * sigma * eigenvalue, 0
    )
    assert (
        tangent * multiplication_by_i - multiplication_by_i * tangent
        == sigma * eigenvalue * sp.Matrix([[0, 1], [1, 0]])
    )
    # Real-linear superposition is valid. Moving a complex phase rotation
    # through the actual semigroup would instead require this commutator zero.
    assert (tangent * multiplication_by_i - multiplication_by_i * tangent).subs(
        sigma, 0
    ) == sp.zeros(2)
