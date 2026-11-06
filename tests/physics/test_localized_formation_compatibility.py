"""Static controls for the complete-law two-carrier compatibility question.

Exact Laurent characters retain independent temporal labels even when their
spatial characters coincide. These checks evaluate no prepared trajectory,
canonical variational input or frozen research producer.
"""

from fractions import Fraction as Q

import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval, sin
from tnfr.physics.relational_sine_comparison import _sine_rates


@pytest.fixture(scope="module", autouse=True)
def no_prepared_response_evaluation():
    with forbid_organized_response_evaluation():
        yield


def _exact(value):
    return sp.simplify(sp.expand(value))


def _laurent_coefficients(expression, variables):
    shifted = sp.Poly(
        sp.expand(expression * variables[0] ** 3 * variables[1] ** 3), *variables
    )
    coefficients = {}
    for powers, value in shifted.terms():
        coefficient = _exact(value)
        if coefficient != 0:
            coefficients[tuple(power - 3 for power in powers)] = coefficient
    return coefficients


@pytest.fixture(scope="module")
def carrier_algebra():
    u, v = sp.symbols("u v", nonzero=True)
    b, conjugate_b, c, conjugate_c = sp.symbols("B conjugate_B C conjugate_C")
    root = (1 + sp.I) / sp.sqrt(2)

    def shift(expression, direction):
        return sp.expand(
            expression.subs(
                {u: root**direction * u, v: root ** (3 * direction) * v},
                simultaneous=True,
            )
        )

    phases = (
        (b * u - conjugate_b / u) / (2 * sp.I),
        (c * v - conjugate_c / v) / (2 * sp.I),
    )
    gaps = tuple(sp.expand(phase - shift(phase, 1)) for phase in phases)

    def cubic(gap):
        return sp.expand((gap**3 - shift(gap**3, -1)) / 12)

    full = cubic(gaps[0] + gaps[1])
    separate = tuple(cubic(gap) for gap in gaps)
    mixed = sp.expand(full - sum(separate))
    return {
        "variables": (u, v),
        "amplitudes": (b, conjugate_b, c, conjugate_c),
        "root": root,
        "shift": shift,
        "phases": phases,
        "gaps": gaps,
        "full": full,
        "separate": separate,
        "mixed": mixed,
        "coefficients": _laurent_coefficients(full, (u, v)),
        "mixed_coefficients": _laurent_coefficients(mixed, (u, v)),
        "frequencies": (1 - sp.sqrt(2) / 2, 1 + sp.sqrt(2) / 2),
    }


def test_exact_sine_mixed_identity_and_its_cubic_limit():
    left, right, scale = sp.symbols("u v epsilon", real=True)
    exact = -(sp.sin(left + right) - sp.sin(left) - sp.sin(right)) / 2
    factored = 2 * sp.sin(left / 2) * sp.sin(right / 2) * sp.sin((left + right) / 2)
    assert sp.trigsimp(sp.expand_trig(exact - factored)) == 0
    scaled = factored.subs(
        {left: scale * left, right: scale * right}, simultaneous=True
    )
    cubic = sp.series(scaled, scale, 0, 4).removeO()
    assert sp.expand(cubic - scale**3 * left * right * (left + right) / 4) == 0
    assert exact.subs(right, -left) == 0
    assert exact.subs(right, 0) == 0


def test_sine_product_remainder_has_the_declared_telescoping_majorant():
    a, b, c, sa, sb, sc = sp.symbols("a b c sa sb sc", real=True)
    decomposition = (sa - a) * sb * sc + a * (sb - b) * sc + a * b * (sc - c)
    assert sp.expand(sa * sb * sc - a * b * c - decomposition) == 0
    left, right = sp.symbols("u v", real=True)
    # |sin(t)-t| <= |t|^3/6 and |sin(t)| <= |t| applied to that identity.
    factor = 2 * sp.Abs(a * b * c) * (a**2 + b**2 + c**2) / 6
    assert (
        _exact(
            factor.subs({a: left / 2, b: right / 2, c: (left + right) / 2})
            - sp.Abs(left * right * (left + right))
            * (left**2 + right**2 + (left + right) ** 2)
            / 96
        )
        == 0
    )


@pytest.mark.parametrize(
    "left,right", [(Q(1, 3), Q(2, 5)), (Q(-7, 4), Q(5, 6)), (Q(8), Q(-3))]
)
def test_sine_mixed_remainder_bound_has_independent_interval_controls(left, right):
    observed = 2 * sin(I(left) / 2) * sin(I(right) / 2) * sin(I(left + right) / 2)
    cubic = left * right * (left + right) / 4
    bound = (
        abs(left * right * (left + right))
        * (left**2 + right**2 + (left + right) ** 2)
        / 96
    )
    assert (observed - cubic).abs_max < bound


def test_mixed_cubic_follows_directly_from_incidence_and_keeps_all_labels(
    carrier_algebra,
):
    data = carrier_algebra
    gap_a, gap_b = data["gaps"]
    edge = sp.expand(gap_a * gap_b * (gap_a + gap_b))
    independent_mixed = sp.expand((edge - data["shift"](edge, -1)) / 4)
    assert _exact(data["mixed"] - independent_mixed) == 0
    all_labels = {
        (m, n) for m in range(-3, 4) for n in range(-3, 4) if abs(m) + abs(n) in (1, 3)
    }
    pure_thirds = {(3, 0), (-3, 0), (0, 3), (0, -3)}
    assert set(data["coefficients"]) == all_labels
    assert len(data["coefficients"]) == 16
    assert set(data["mixed_coefficients"]) == all_labels - pure_thirds
    assert len(data["mixed_coefficients"]) == 12
    u, v = data["variables"]
    reconstructed = sum(
        value * u**m * v**n for (m, n), value in data["coefficients"].items()
    )
    assert _exact(data["full"] - reconstructed) == 0


def test_resonant_coefficients_and_zero_amplitude_limits(carrier_algebra):
    data = carrier_algebra
    b, conjugate_b, c, conjugate_c = data["amplitudes"]
    frequency_a, frequency_b = data["frequencies"]
    self_a, self_b = frequency_a**2 / 8, frequency_b**2 / 8
    cross = frequency_a * frequency_b / 4
    assert _exact(cross - sp.Rational(1, 8)) == 0
    assert _exact(self_a - (3 - 2 * sp.sqrt(2)) / 16) == 0
    assert _exact(self_b - (3 + 2 * sp.sqrt(2)) / 16) == 0
    expected_a = -sp.I * (self_a * b * conjugate_b + cross * c * conjugate_c) * b
    expected_b = -sp.I * (self_b * c * conjugate_c + cross * b * conjugate_b) * c
    assert _exact(data["coefficients"][(1, 0)] - expected_a) == 0
    assert _exact(data["coefficients"][(0, 1)] - expected_b) == 0
    remaining = {
        (3, 0): -sp.I * (sp.sqrt(2) - 1) * b**3 / 48,
        (0, 3): sp.I * (sp.sqrt(2) + 1) * c**3 / 48,
        (2, 1): -sp.I * b**2 * c / 16,
        (1, 2): -sp.I * b * c**2 / 16,
        (2, -1): sp.I * (sp.sqrt(2) - 1) * b**2 * conjugate_c / 16,
        (-1, 2): -sp.I * (sp.sqrt(2) + 1) * conjugate_b * c**2 / 16,
    }
    for label, expected in remaining.items():
        assert _exact(data["coefficients"][label] - expected) == 0
    assert _exact(data["full"].subs({c: 0, conjugate_c: 0}) - data["separate"][0]) == 0
    assert _exact(data["full"].subs({b: 0, conjugate_b: 0}) - data["separate"][1]) == 0
    conjugacy = {
        sp.conjugate(b): conjugate_b,
        sp.conjugate(conjugate_b): b,
        sp.conjugate(c): conjugate_c,
        sp.conjugate(conjugate_c): c,
    }
    for (m, n), coefficient in data["coefficients"].items():
        assert (
            _exact(
                data["coefficients"][(-m, -n)]
                - sp.conjugate(coefficient).subs(conjugacy)
            )
            == 0
        )


def test_spatial_aliases_do_not_merge_temporal_labels_or_resonances(carrier_algebra):
    data = carrier_algebra
    root = data["root"]
    frequency_a, frequency_b = data["frequencies"]
    labels = tuple(data["coefficients"])
    assert {(m + 3 * n) % 8 for m, n in labels} == {1, 3, 5, 7}
    frequencies = {_exact(m * frequency_a + n * frequency_b) for m, n in labels}
    assert len(frequencies) == len(labels)
    plus_resonances, minus_resonances = set(), set()
    nonresonant_magnitudes = []
    for m, n in labels:
        spatial = m + 3 * n
        eigenvalue = _exact(1 - (root**spatial + root ** (-spatial)) / 2)
        temporal = m * frequency_a + n * frequency_b
        if _exact(temporal - eigenvalue) == 0:
            plus_resonances.add((m, n))
        else:
            nonresonant_magnitudes.append(_exact(sp.Abs(temporal - eigenvalue)))
        if _exact(temporal + eigenvalue) == 0:
            minus_resonances.add((m, n))
    assert plus_resonances == {(1, 0), (0, 1)}
    assert minus_resonances == {(-1, 0), (0, -1)}
    divisor_floor = 2 - sp.sqrt(2)
    assert len(nonresonant_magnitudes) == 14
    assert all(
        _exact(value - divisor_floor).is_nonnegative for value in nonresonant_magnitudes
    )
    assert divisor_floor in nonresonant_magnitudes
    alias_pair = ((3, 0), (0, 1))
    spatial_characters = [root**m * (root**3) ** n for m, n in alias_pair]
    assert _exact(spatial_characters[0] - spatial_characters[1]) == 0
    assert _exact(3 * frequency_a - frequency_b) != 0


def test_generated_coefficients_match_an_independent_eight_site_incidence(
    carrier_algebra,
):
    data = carrier_algebra
    b, conjugate_b, c, conjugate_c = data["amplitudes"]
    values = {
        b: (1 + 2 * sp.I) / 5,
        conjugate_b: (1 - 2 * sp.I) / 5,
        c: (2 - sp.I) / 7,
        conjugate_c: (2 + sp.I) / 7,
    }
    phase_roots = ((3 + 4 * sp.I) / 5, (5 + 12 * sp.I) / 13)
    root = data["root"]
    characters = tuple(
        (_exact(phase_roots[0] * root**j), _exact(phase_roots[1] * root ** (3 * j)))
        for j in range(8)
    )
    phases = tuple(
        _exact(
            (
                values[b] * u
                - values[conjugate_b] / u
                + values[c] * v
                - values[conjugate_c] / v
            )
            / (2 * sp.I)
        )
        for u, v in characters
    )
    gaps = tuple(_exact(phases[j] - phases[(j + 1) % 8]) for j in range(8))
    direct = tuple(_exact((gaps[j] ** 3 - gaps[j - 1] ** 3) / 12) for j in range(8))
    coefficients = {
        label: _exact(value.subs(values))
        for label, value in data["coefficients"].items()
    }
    for actual, (u, v) in zip(direct, characters):
        reconstructed = sum(
            value * u**m * v**n for (m, n), value in coefficients.items()
        )
        assert _exact(actual - reconstructed) == 0
    assert _exact(sum(direct)) == 0


def test_same_state_sine_defect_uses_actual_rows_and_linear_loss_cancels():
    size = 8
    model = RelationalExchangeModel(
        storage_scale=1,
        epi_weight=Q(1, 2**20),
        phase_weight=1 - Q(1, 2**20),
        phase_domain="regular",
    )
    forms = (
        tuple(Q(j - 3, 17) for j in range(size)),
        tuple(Q((-1) ** j, 13) for j in range(size)),
    )
    phases = (
        tuple(Q(j % 3 - 1, 11) for j in range(size)),
        tuple(Q((-1) ** j * (j + 1), 37) for j in range(size)),
    )

    def field(form, phase):
        gradient = tuple(
            2 * form[j] - form[j - 1] - form[(j + 1) % size] for j in range(size)
        )
        current = tuple(
            sin(I(phase[j - 1] - phase[j])) + sin(I(phase[(j + 1) % size] - phase[j]))
            for j in range(size)
        )
        return _sine_rates(model, (2,) * size, gradient, (1,) * size, current)

    separate = tuple(field(form, phase) for form, phase in zip(forms, phases))
    joined = field(
        tuple(a + b for a, b in zip(*forms)), tuple(a + b for a, b in zip(*phases))
    )
    gaps = tuple(
        tuple(phase[j] - phase[(j + 1) % size] for j in range(size)) for phase in phases
    )
    edge_mixed = tuple(
        2 * sin(I(a) / 2) * sin(I(b) / 2) * sin(I(a + b) / 2) for a, b in zip(*gaps)
    )
    clock = pi_interval() / Q(model.phase_weight)
    for j in range(size):
        actual = clock * (
            joined["form_rates"][j]
            - separate[0]["form_rates"][j]
            - separate[1]["form_rates"][j]
        )
        expected = edge_mixed[j] - edge_mixed[j - 1]
        assert (actual - expected).contains(0)
        assert (actual - expected).abs_max < Q(1, 10**32)
        phase_defect = (
            joined["phase_rates"][j]
            - separate[0]["phase_rates"][j]
            - separate[1]["phase_rates"][j]
        )
        assert phase_defect.contains(0) and phase_defect.abs_max < Q(1, 10**32)
    # This cancellation holds at supplied states, not between evolved histories.
    assert any(not value.contains(0) for value in edge_mixed)


def test_leading_damping_retains_counterrotating_loss_and_actual_two_rows():
    u, b, conjugate_b = sp.symbols("u B conjugate_B", nonzero=True)
    eigenvalue, sigma, epsilon = sp.symbols("lambda sigma epsilon", positive=True)
    form = epsilon * (b * u + conjugate_b / u) / 2
    phase = epsilon * (b * u - conjugate_b / u) / (2 * sp.I)
    actual_form_rate = -sigma * eigenvalue * form - eigenvalue * phase
    actual_phase_rate = eigenvalue * form
    actual_complex_rate = sp.expand(actual_form_rate + sp.I * actual_phase_rate)
    tangent = sp.I * eigenvalue * epsilon * b * u
    loss = sp.expand(actual_complex_rate - tangent)
    assert loss.coeff(u, 1) == -sigma * eigenvalue * epsilon * b / 2
    assert loss.coeff(u, -1) == -sigma * eigenvalue * epsilon * conjugate_b / 2
    gamma = sigma / epsilon**2
    resonant_loss = -gamma * eigenvalue * b / 2
    assert _exact(epsilon**3 * resonant_loss - loss.coeff(u, 1)) == 0
    corrector = -sp.I * gamma * conjugate_b / 4
    assert (
        _exact(
            -2 * sp.I * eigenvalue * corrector + gamma * eigenvalue * conjugate_b / 2
        )
        == 0
    )
    assert (
        _exact(actual_phase_rate - (eigenvalue * form - sigma * eigenvalue * phase / 2))
        != 0
    )


def test_leading_envelope_density_balance_has_no_direct_cross_mass_transfer():
    position = sp.Symbol("X", real=True)
    real, imaginary = sp.Function("u", real=True)(position), sp.Function(
        "v", real=True
    )(position)
    dispersion, potential, gamma, eigenvalue = sp.symbols("d V gamma lambda", real=True)
    real_rate = (
        dispersion * sp.diff(imaginary, position, 2)
        + potential * imaginary
        - gamma * eigenvalue * real / 2
    )
    imaginary_rate = (
        -dispersion * sp.diff(real, position, 2)
        - potential * real
        - gamma * eigenvalue * imaginary / 2
    )
    density_rate = sp.expand(2 * real * real_rate + 2 * imaginary * imaginary_rate)
    current = (
        -2
        * dispersion
        * (real * sp.diff(imaginary, position) - imaginary * sp.diff(real, position))
    )
    density = real**2 + imaginary**2
    assert (
        sp.expand(
            density_rate + sp.diff(current, position) + gamma * eigenvalue * density
        )
        == 0
    )
    assert sp.diff(density_rate, potential) == 0
    # Only periodic or zero-flux integration removes this boundary derivative.
    assert sp.diff(current, position) != 0


def test_independent_phase_symmetry_belongs_to_the_leading_coupled_system(
    carrier_algebra,
):
    b, conjugate_b, c, conjugate_c = carrier_algebra["amplitudes"]
    second_b, second_c = sp.symbols("Bxx Cxx")
    angle_a, angle_b = sp.symbols("alpha beta", real=True)
    dispersion_a, dispersion_b, gamma = sp.symbols("d1 d2 gamma", real=True)
    frequency_a, frequency_b = carrier_algebra["frequencies"]
    phase_a, phase_b = sp.exp(sp.I * angle_a), sp.exp(sp.I * angle_b)
    substitutions = {
        b: phase_a * b,
        conjugate_b: conjugate_b / phase_a,
        second_b: phase_a * second_b,
        c: phase_b * c,
        conjugate_c: conjugate_c / phase_b,
        second_c: phase_b * second_c,
    }
    rows = (
        dispersion_a * second_b
        + (frequency_a**2 * b * conjugate_b / 8 + c * conjugate_c / 8) * b
        - sp.I * gamma * frequency_a * b / 2,
        dispersion_b * second_c
        + (frequency_b**2 * c * conjugate_c / 8 + b * conjugate_b / 8) * c
        - sp.I * gamma * frequency_b * c / 2,
    )
    for row, phase in zip(rows, (phase_a, phase_b)):
        assert _exact(row.subs(substitutions, simultaneous=True) - phase * row) == 0
    assert carrier_algebra["coefficients"][(2, -1)] != 0


def test_bright_stationary_second_channel_obstruction_keeps_its_tail_hypothesis():
    position = sp.Symbol("X", real=True)
    first, second = sp.Function("f1", real=True)(position), sp.Function(
        "f2", real=True
    )(position)
    dispersion, cubic, cross, omega = sp.symbols("d2 g2 c Omega2", real=True)
    residual = (
        dispersion * sp.diff(second, position, 2)
        + cubic * second**3
        + cross * first**2 * second
        - omega * second
    )
    divergence = sp.diff(dispersion * second * sp.diff(second, position), position)
    integrated_density = (
        -dispersion * sp.diff(second, position) ** 2
        + cubic * second**4
        + cross * first**2 * second**2
        - omega * second**2
    )
    assert sp.expand(second * residual - divergence - integrated_density) == 0
    mass, quartic = sp.symbols("M Q", positive=True)
    gradient, overlap = sp.symbols("K J", nonnegative=True)
    actual_d = -sp.sqrt(2) / 4
    actual_g = (3 + 2 * sp.sqrt(2)) / 16
    forced_omega = (-actual_d * gradient + actual_g * quartic + overlap / 8) / mass
    assert forced_omega.is_positive
    assert (forced_omega / actual_d).is_negative
    decay = sp.Symbol("kappa", positive=True)
    tail = sp.exp(-decay * position)
    assert sp.diff(tail, position, 2) == decay**2 * tail
    # A nonzero exponential tail demands Omega2/d2=kappa^2>0, contradictory.
    assert (
        _exact(
            (dispersion * sp.diff(tail, position, 2) - omega * tail).subs(
                omega, dispersion * decay**2
            )
        )
        == 0
    )
    # The identity does not exclude periodic profiles or general time dependence.
    constant = sp.Symbol("a", positive=True)
    assert (
        _exact(
            residual.subs(
                {first: 0, second: constant, omega: cubic * constant**2}
            ).doit()
        )
        == 0
    )


def test_static_fejer_weights_and_tail_budget_follow_independent_counting():
    order, size, radius = 16, 1024, 96
    # |sum_{r=0}^{Q-1} u^r|^2/Q gives the triangular coefficients
    # independently of a packet constructor or any evaluated source.
    multiplicity = {}
    for first in range(order):
        for second in range(order):
            difference = first - second
            multiplicity[difference] = multiplicity.get(difference, 0) + 1
    weights = {mode: Q(count, order) for mode, count in multiplicity.items()}
    assert weights == {
        mode: 1 - Q(abs(mode), order) for mode in range(1 - order, order)
    }
    norm_squared = sum(weight**2 for weight in weights.values())
    assert norm_squared == Q(2 * order**2 + 1, 3 * order) == Q(171, 16)
    norm_lower = Q(2 * order, 3)
    assert norm_squared > norm_lower
    variable = sp.Symbol("s", positive=True)
    primitive = -1 / (3 * variable**3)
    assert sp.diff(primitive, variable) == variable**-4
    # The Fejer pointwise bound squared is N^3/(16 Q^2 S2 s^4).
    # Two tails and the integral comparison supply the remaining factor.
    tail = Q(size**3, 16 * order**2) * Q(2, 3 * (radius - 1) ** 3) / norm_lower
    assert tail == Q(size**3, 16 * order**3 * (radius - 1) ** 3)
    assert tail == Q(16384, 857375) < Q(1, 49)
    first_band = {128 + mode for mode in weights}
    second_band = {384 + mode for mode in weights}
    assert first_band.isdisjoint(second_band)
    assert 0 not in first_band | second_band
    assert len(first_band) == len(second_band) == 2 * order - 1


def test_initial_source_error_preserves_the_declared_absence_margin():
    amplitude = sp.Symbol("a", positive=True)
    relative_error = Q(1, 1000)
    norm_lower = amplitude * (1 - sp.Rational(1, 1000))
    assert (norm_lower - amplitude / 2).is_positive
    # Outside the source window, ||P_out(z0+e)|| <= a/7 + delta;
    # the complete norm is at least a-delta. Normalize before squaring.
    ratio = (Q(1, 7) + relative_error) / (1 - relative_error)
    assert ratio == Q(1007, 6993) < Q(1, 6)
    assert ratio**2 < Q(1, 36)
    identity_threshold = Q(3, 4)
    assert identity_threshold - Q(1, 36) == Q(13, 18)


def test_static_preparation_admits_primitive_phase_and_retains_error_means():
    size, order = 1024, 16
    amplitude = Q(1, 2**12)
    error_radius = amplitude / 1000
    weights = [1 - Q(abs(mode), order) for mode in range(1 - order, order)]
    square_sum = sum(weight**2 for weight in weights)
    peak_squared = sum(weights) ** 2 / (size * square_sum)
    assert peak_squared == Q(4, 171) < Q(4, 169)
    point_bound = 2 * amplitude * Q(2, 13) + error_radius
    assert point_bound < amplitude / 3
    assert 2 * point_bound < Q(1, 6144) < pi_interval().lo / 2
    # Orthogonal nominal bands give sqrt(2)*a; the complete error is added.
    assert (Q(3, 2) - Q(1, 1000)) ** 2 > 2
    # A permitted zero-mode error has nonzero mean and saturates its bound.
    constant_error = error_radius / 32
    assert size * constant_error**2 == error_radius**2
    assert constant_error == amplitude / 32000 > 0
    relative_error = sp.Symbol("eta", nonnegative=True)
    normalized_tail = (sp.Rational(1, 7) + relative_error) / (1 - relative_error)
    assert (
        _exact(
            sp.diff(normalized_tail, relative_error)
            - 8 / (7 * (relative_error - 1) ** 2)
        )
        == 0
    )


def test_antipodal_packet_windows_cannot_share_the_declared_common_center():
    size, radius = 1024, 96

    def circular_distance(first, second):
        difference = abs(first - second) % size
        return min(difference, size - difference)

    sources = (Q(0), Q(size, 2))
    centers = tuple(Q(index, 2) for index in range(2 * size))
    farthest = tuple(
        max(circular_distance(center, source) for source in sources)
        for center in centers
    )
    # Independent finite checks include every node and midpoint, especially
    # the quarter-cycle equality cases of the triangle-inequality proof.
    assert min(farthest) == Q(size, 4) > 2 * radius
    assert max(circular_distance(Q(size, 4), source) for source in sources) == Q(
        size, 4
    )
    for center in centers:
        distances = tuple(circular_distance(center, source) for source in sources)
        assert sum(distances) >= circular_distance(*sources)
        assert any(distance > 2 * radius for distance in distances)
    # The exclusion depends on antipodal preparation; co-location fails it.
    assert max(circular_distance(0, source) for source in (0, 0)) < radius


def test_same_band_cubic_mixing_retains_relative_phase_without_separate_laws():
    first, first_conjugate, second, second_conjugate = sp.symbols("B1 C1 B2 C2")
    joined = (first + second) ** 2 * (first_conjugate + second_conjugate)
    separate = first**2 * first_conjugate + second**2 * second_conjugate
    independent_mixed = (
        2 * first * first_conjugate * second
        + 2 * second * second_conjugate * first
        + first**2 * second_conjugate
        + second**2 * first_conjugate
    )
    mixed = sp.expand(joined - separate)
    assert sp.expand(mixed - independent_mixed) == 0
    magnitude = sp.Symbol("a", positive=True)
    aligned = {
        first: magnitude,
        first_conjugate: magnitude,
        second: magnitude,
        second_conjugate: magnitude,
    }
    opposed = {
        first: magnitude,
        first_conjugate: magnitude,
        second: -magnitude,
        second_conjugate: -magnitude,
    }
    assert mixed.subs(aligned) == 6 * magnitude**3
    assert mixed.subs(opposed) == 0
    assert (first * first_conjugate + second * second_conjugate).subs(aligned) == (
        first * first_conjugate + second * second_conjugate
    ).subs(opposed)
