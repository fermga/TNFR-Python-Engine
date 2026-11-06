"""Independent algebraic controls for a fixed lifted modal comparison.

Only constructed static states and the declared source algebra appear here.
No earlier trajectory is loaded or integrated. The harmonic reference is
not the complete TNFR law, and a large instantaneous defect is not itself
a lower bound on accumulated state error.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research import replica_modal_criterion as modal

ORACLE_DIGITS = 140
ORACLE_TOLERANCE = Q(1, 10**110)
POINT_WIDTH_CAP = Q(1, 10**30)


@pytest.fixture(scope="module")
def certificate():
    return modal.source_certificate()


def _context():
    context = mpmath.mp.clone()
    context.dps = ORACLE_DIGITS
    return context


def _mp(context, value):
    value = Q(value)
    return context.mpf(value.numerator) / value.denominator


def _encloses(context, interval, expected):
    tolerance = _mp(context, ORACLE_TOLERANCE)
    assert interval.width < POINT_WIDTH_CAP
    assert _mp(context, interval.lo) - tolerance <= expected
    assert expected <= _mp(context, interval.hi) + tolerance


def _matrix_product(left, right):
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(5)) for j in range(5))
        for i in range(5)
    )


def _matrix_sum(left, right, scale=1):
    return tuple(
        tuple(left[i][j] + scale * right[i][j] for j in range(5)) for i in range(5)
    )


def _pair_product(left, right):
    a, b = left
    c, d = right
    return (
        _matrix_sum(_matrix_product(a, c), _matrix_product(b, d), 5),
        _matrix_sum(_matrix_product(a, d), _matrix_product(b, c)),
    )


def _apply(matrix, vector):
    return tuple(sum(row[j] * vector[j] for j in range(5)) for row in matrix)


def _laplacian():
    return tuple(
        tuple(Q(2 if i == j else -1 if (i - j) % 5 in (1, 4) else 0) for j in range(5))
        for i in range(5)
    )


def _lap(vector):
    return tuple(
        2 * vector[i] - vector[(i - 1) % 5] - vector[(i + 1) % 5] for i in range(5)
    )


def _fine_state(forms, phases, internal_forms, internal_phases):
    return tuple(
        forms[i] + sign * internal_forms[i] for i in range(5) for sign in (1, -1)
    ) + tuple(
        phases[i] + sign * internal_phases[i] for i in range(5) for sign in (1, -1)
    )


def _project(values):
    return tuple((values[2 * i] + values[2 * i + 1]) / 2 for i in range(5))


def _oracle(context, state):
    rates = []
    for node in range(10):
        neighbors = tuple(
            other for other in range(10) if (node // 2 - other // 2) % 5 in (1, 4)
        )
        rates.append(
            sum(
                context.sin(state[10 + other] - state[10 + node]) for other in neighbors
            )
            / (4 * context.pi)
        )
    for node in range(10):
        neighbors = tuple(
            other for other in range(10) if (node // 2 - other // 2) % 5 in (1, 4)
        )
        rates.append(
            sum(state[node] - state[other] for other in neighbors) / (4 * context.pi)
        )
    return tuple(rates)


def _rational_state():
    return _fine_state(
        (Q(1, 5), Q(-1, 4), Q(3, 10), Q(-2, 5), Q(3, 20)),
        (Q(-1, 3), Q(1, 4), Q(-1, 5), Q(2, 7), Q(1, 9)),
        (Q(1, 8), Q(-1, 6), Q(1, 7), Q(-1, 5), Q(1, 9)),
        (Q(1, 10), Q(-1, 8), Q(1, 6), Q(-1, 9), Q(1, 12)),
    )


def _twist_state(pi):
    return _fine_state(
        (Q(1, 7),) * 5,
        tuple(2 * (i - 2) * pi / 5 for i in range(5)),
        (Q(1, 5),) * 5,
        (Q(1, 8),) * 5,
    )


def test_rational_cycle_projectors_have_the_correct_polynomial_and_multiplicities():
    geometry = modal.c5_modes()
    lap = _laplacian()
    identity = tuple(tuple(Q(i == j) for j in range(5)) for i in range(5))
    zero = tuple((Q(0),) * 5 for _ in range(5))
    constant = tuple((Q(1, 5),) * 5 for _ in range(5))
    centered = _matrix_sum(identity, constant, -1)
    square = _matrix_product(lap, lap)
    assert _matrix_sum(_matrix_sum(square, lap, -5), centered, 5) == zero
    assert (
        _matrix_sum(_matrix_sum(_matrix_product(square, lap), square, -5), lap, 5)
        == zero
    )
    assert geometry.laplacian == lap
    assert geometry.constant_projector == constant
    assert geometry.centered_projector == centered
    assert geometry.multiplicities == (1, 2, 2) and geometry.polynomial_certified
    parts = geometry.projector_rational_parts
    assert parts[0] == (constant, zero)
    for i, projector in enumerate(parts):
        assert _pair_product(projector, projector) == projector
        for j in range(i):
            assert _pair_product(projector, parts[j]) == (zero, zero)
        a, b = projector
        assert sum(a[k][k] for k in range(5)) == geometry.multiplicities[i]
        assert sum(b[k][k] for k in range(5)) == 0
    assert _matrix_sum(_matrix_sum(parts[0][0], parts[1][0]), parts[2][0]) == identity
    assert _matrix_sum(_matrix_sum(parts[0][1], parts[1][1]), parts[2][1]) == zero
    context = _context()
    # The Fourier kernel is independent of the implementation's quadratic
    # field representation of the two spectral projectors.
    for mode, projector in enumerate(geometry.projectors):
        for i in range(5):
            for j in range(5):
                expected = (
                    context.mpf(1) / 5
                    if mode == 0
                    else 2 * context.cos(2 * context.pi * mode * (i - j) / 5) / 5
                )
                _encloses(context, projector[i][j], expected)
    for actual, expected in zip(
        geometry.eigenvalues,
        (context.mpf(0), (5 - context.sqrt(5)) / 2, (5 + context.sqrt(5)) / 2),
    ):
        _encloses(context, actual, expected)
    with pytest.raises(FrozenInstanceError):
        geometry.polynomial_certified = False


def test_harmonic_reference_retains_initialization_clock_and_quadratic_storage():
    forms = (Q(1, 5), Q(-1, 4), Q(3, 10), Q(-2, 5), Q(3, 20))
    phases = (Q(-1, 3), Q(1, 4), Q(-1, 5), Q(2, 7), Q(1, 9))
    context = _context()
    x = tuple(_mp(context, value) for value in forms)
    theta = tuple(_mp(context, value) for value in phases)
    initial_store = 2 * sum(
        x[i] * _lap(x)[i] + theta[i] * _lap(theta)[i] for i in range(5)
    )
    initial_norm = sum(value**2 for value in x + theta)
    for time in (Q(0), Q(7, 5)):
        result = modal.harmonic_reference(forms, phases, time=time)
        t = _mp(context, time)
        expected_x = [sum(x) / 5] * 5
        expected_theta = [sum(theta) / 5] * 5
        for mode in (1, 2):
            projector = tuple(
                tuple(
                    2 * context.cos(2 * context.pi * mode * (i - j) / 5) / 5
                    for j in range(5)
                )
                for i in range(5)
            )
            px, pt = _apply(projector, x), _apply(projector, theta)
            frequency = (2 - 2 * context.cos(2 * context.pi * mode / 5)) / (
                2 * context.pi
            )
            sine, cosine = context.sin(frequency * t), context.cos(frequency * t)
            for i in range(5):
                expected_x[i] += cosine * px[i] - sine * pt[i]
                expected_theta[i] += sine * px[i] + cosine * pt[i]
        for i in range(5):
            _encloses(context, result.forms[i], expected_x[i])
            _encloses(context, result.phases[i], expected_theta[i])
            _encloses(
                context,
                result.form_rates[i],
                -_lap(expected_theta)[i] / (2 * context.pi),
            )
            _encloses(
                context, result.phase_rates[i], _lap(expected_x)[i] / (2 * context.pi)
            )
        final_store = 2 * sum(
            expected_x[i] * _lap(expected_x)[i]
            + expected_theta[i] * _lap(expected_theta)[i]
            for i in range(5)
        )
        assert abs(final_store - initial_store) < _mp(context, ORACLE_TOLERANCE)
        assert abs(
            sum(value**2 for value in expected_x + expected_theta) - initial_norm
        ) < _mp(context, ORACLE_TOLERANCE)
        assert abs(sum(expected_x) - sum(x)) < _mp(context, ORACLE_TOLERANCE)
        assert abs(sum(expected_theta) - sum(theta)) < _mp(context, ORACLE_TOLERANCE)


def test_full_twenty_rows_and_auxiliary_balance_match_independent_oracle():
    state = _rational_state()
    result = modal.observe_modal_state(state)
    context = _context()
    values = tuple(_mp(context, value) for value in state)
    fine = _oracle(context, values)
    x, theta = _project(values[:10]), _project(values[10:])
    dx, dt = _project(fine[:10]), _project(fine[10:])
    eta = tuple(dx[i] + _lap(theta)[i] / (2 * context.pi) for i in range(5))
    deltas = tuple((values[10 + 2 * i] - values[11 + 2 * i]) / 2 for i in range(5))
    gaps = tuple(theta[(i + 1) % 5] - theta[i] for i in range(5))
    residual = tuple(
        gaps[i]
        - context.cos(deltas[i])
        * context.cos(deltas[(i + 1) % 5])
        * context.sin(gaps[i])
        for i in range(5)
    )
    for actual, expected in zip(result.fine_rates, fine):
        _encloses(context, actual, expected)
    for i in range(5):
        _encloses(context, result.forms[i], x[i])
        _encloses(context, result.phases[i], theta[i])
        _encloses(context, result.form_rates[i], dx[i])
        _encloses(context, result.phase_rates[i], dt[i])
        _encloses(context, result.eta[i], eta[i])
        assert abs(
            eta[i] - (residual[(i - 1) % 5] - residual[i]) / (2 * context.pi)
        ) < _mp(context, ORACLE_TOLERANCE)
        _encloses(context, result.oriented_gaps[i], theta[(i + 1) % 5] - theta[i])
        di = (values[10 + 2 * i] - values[11 + 2 * i]) / 2
        following = (i + 1) % 5
        dj = (values[10 + 2 * following] - values[11 + 2 * following]) / 2
        _encloses(
            context, result.resultant_products[i], context.cos(di) * context.cos(dj)
        )
    energy = 2 * sum(x[i] * _lap(x)[i] + theta[i] * _lap(theta)[i] for i in range(5))
    derivative = 4 * sum(dx[i] * _lap(x)[i] + dt[i] * _lap(theta)[i] for i in range(5))
    work = 4 * sum(_lap(x)[i] * eta[i] for i in range(5))
    assert abs(derivative - work) < _mp(context, ORACLE_TOLERANCE)
    _encloses(context, result.harmonic_storage, energy)
    _encloses(context, result.harmonic_storage_rate, derivative)
    _encloses(context, result.harmonic_storage_rate_from_defect, work)
    # This second static state has an analytically positive auxiliary rate:
    # X=Theta=a*e0, delta=0 gives 12*a*(a-sin(a))/pi. It is not a change
    # in the full conserved storage or a new acquired preparation.
    a = Q(1, 2)
    positive = modal.observe_modal_state(
        _fine_state((a, 0, 0, 0, 0), (a, 0, 0, 0, 0), (Q(1, 8),) * 5, (Q(0),) * 5)
    )
    expected_positive = (
        12
        * _mp(context, a)
        * (_mp(context, a) - context.sin(_mp(context, a)))
        / context.pi
    )
    assert 3 * a * (a**3 / 6 - a**5 / 120) > Q(1, 100)
    assert positive.harmonic_storage_rate.lo > Q(1, 100)
    _encloses(context, positive.harmonic_storage_rate, expected_positive)
    _encloses(context, positive.harmonic_storage_rate_from_defect, expected_positive)


def test_source_profile_barrier_requires_finite_nonlinear_transport(certificate):
    context = _context()
    c = Q(570, 287)
    cover = Q(1, 2**20) + Q(1, 2**128)
    mu = Q(1, 100)
    profile = (-1, -1, 0, 1, 1)
    assert _lap(profile) == (-2, -1, 0, 1, 2)
    minus, plus = certificate.geometry.projector_rational_parts[1:]
    # Exact modal coefficients of a=Theta_4 and b=Theta_3. Their two
    # sinusoidal factors need not attain independent extremes physically.
    assert (_apply(minus[0], profile)[4], _apply(minus[1], profile)[4]) == (
        Q(1, 2),
        Q(1, 10),
    )
    assert (_apply(plus[0], profile)[4], _apply(plus[1], profile)[4]) == (
        Q(1, 2),
        Q(-1, 10),
    )
    assert (_apply(minus[0], profile)[3], _apply(minus[1], profile)[3]) == (
        Q(1, 2),
        Q(3, 10),
    )
    assert (_apply(plus[0], profile)[3], _apply(plus[1], profile)[3]) == (
        Q(1, 2),
        Q(-3, 10),
    )
    root5 = context.sqrt(5)
    a_minus, a_plus = (1 + root5) / (2 * root5), (root5 - 1) / (2 * root5)
    b_minus, b_plus = (3 + root5) / (2 * root5), (root5 - 3) / (2 * root5)
    assert 0 < a_minus < 1 and 0 < a_plus < 1 and b_plus < 0 < b_minus
    assert abs(a_minus + a_plus - 1) < _mp(context, ORACLE_TOLERANCE)
    assert abs(abs(b_minus) + abs(b_plus) - 3 / root5) < _mp(context, ORACLE_TOLERANCE)
    assert abs(abs(a_minus - b_minus) + abs(a_plus - b_plus) - 2 / root5) < _mp(
        context, ORACLE_TOLERANCE
    )
    assert 9 < 4 * 5  # 3/sqrt(5)<2; the closing gap has envelope2c.
    # The full20 independent coordinate error contracts under pair means.
    fine_errors = (cover,) * 20
    coarse_errors = _project(fine_errors[:10]) + _project(fine_errors[10:])
    assert sum(value**2 for value in coarse_errors) == 10 * cover**2
    assert (
        sum(value**2 for value in coarse_errors)
        == sum(value**2 for value in fine_errors) / 2
    )
    assert 20 < 25  # Edge norm sqrt2 times sqrt10 is strictly below5.
    assert certificate.reference_gap_upper_bound == 2 * c + 5 * cover
    target_gap = 3 * context.pi / 2 + _mp(context, mu)
    threshold = (target_gap - _mp(context, 2 * c + 5 * cover)) / context.sqrt(2)
    _encloses(context, certificate.transport_threshold_enclosure, threshold)
    assert (
        certificate.necessary_deviation_lower_bound
        == certificate.transport_threshold_enclosure.lo
    )
    assert (
        certificate.necessary_absolute_transport_lower_bound
        == certificate.necessary_deviation_lower_bound
        > Q(1, 2)
    )
    assert (
        certificate.harmonic_target_excluded
        and certificate.necessary_transport_certified
    )
    # These strict rational premises alone already give K>1/2; no free
    # evolution or prior response was sampled to choose the threshold.
    assert pi_interval().lo > Q(157, 50) and c < 2 and cover < Q(1, 10000)
    assert Q(10, 7) ** 2 > 2
    assert (Q(3, 2) * Q(157, 50) + mu - 4 - Q(5, 10000)) * Q(7, 10) > Q(1, 2)
    source_store = 12 * c**2 + 32 * c * cover + 80 * cover**2
    assert certificate.source_harmonic_storage_upper_bound == source_store < 48
    assert Q(405, 8) < certificate.target_harmonic_storage_lower_bound
    assert (
        _mp(context, certificate.target_harmonic_storage_lower_bound)
        <= Q(5, 2).numerator * target_gap**2 / Q(5, 2).denominator
    )
    assert (
        certificate.necessary_auxiliary_work_lower_bound
        == certificate.target_harmonic_storage_lower_bound - source_store
        > 0
    )
    assert certificate.auxiliary_work_certified
    # The reference envelope is above pi: this proves exclusion only of
    # the specified all-acute-margin target, not every winding change.
    assert _mp(context, certificate.reference_gap_upper_bound) > context.pi
    assert not certificate.arbitrary_winding_change_excluded


def test_prepared_twist_has_large_defect_and_zero_collective_actual_field(certificate):
    context = _context()
    state = _twist_state(pi_interval())
    result = modal.observe_modal_state(state)
    values = _fine_state(
        (_mp(context, Q(1, 7)),) * 5,
        tuple(2 * (i - 2) * context.pi / 5 for i in range(5)),
        (_mp(context, Q(1, 5)),) * 5,
        (_mp(context, Q(1, 8)),) * 5,
    )
    fine = _oracle(context, values)
    expected_eta = (-1, 0, 0, 0, 1)
    for actual, expected in zip(result.fine_rates, fine):
        _encloses(context, actual, expected)
    for i in range(5):
        _encloses(context, result.form_rates[i], context.mpf(0))
        _encloses(context, result.phase_rates[i], context.mpf(0))
        _encloses(context, result.eta[i], context.mpf(expected_eta[i]))
    # The large eta cancels the aligned harmonic form row. It does not
    # imply this prepared state's coarse coordinates are moving quickly.
    harmonic = modal.harmonic_reference(result.forms, result.phases, time=0)
    for i in range(5):
        _encloses(context, harmonic.form_rates[i], -context.mpf(expected_eta[i]))
        assert (harmonic.form_rates[i] + result.eta[i]).contains(0)
    assert result.harmonic_storage_rate.contains(0)
    assert any(abs(value) > _mp(context, Q(1, 100)) for value in fine)
    omega = 2 * context.pi / 5
    mu = _mp(context, Q(1, 100))
    assert mu < omega < context.pi / 2 - mu
    assert context.cos(_mp(context, Q(1, 8))) ** 2 > _mp(context, Q(1, 4))
    width = context.pi / 2 - 2 * mu - context.cos(mu) / 4 + context.sin(mu)
    lower = context.sqrt(2) * (1 - width / (2 * context.pi))
    _encloses(context, certificate.target_remainder_width_bound, width)
    _encloses(context, certificate.pointwise_lower_bound_enclosure, lower)
    assert (
        certificate.pointwise_defect_lower_bound
        == certificate.pointwise_lower_bound_enclosure.lo
        > 1
    )
    assert _mp(context, certificate.pointwise_defect_lower_bound) < context.sqrt(2)
    assert certificate.pointwise_target_defect_certified
    assert not result.source_membership_inferred and not result.acquisition_claimed
    assert not certificate.instantaneous_defect_is_accumulated_error


def test_continuous_lift_changes_reference_but_not_the_complete_periodic_field():
    context = _context()
    state = _rational_state()
    base = modal.observe_modal_state(state)
    common = state[:10] + tuple(value + Q(7, 3) for value in state[10:])
    translated = modal.observe_modal_state(common)
    turn = list(state)
    for index in (10, 11):
        turn[index] = turn[index] + 2 * pi_interval()
    rebranched = modal.observe_modal_state(turn)
    physical = _oracle(context, tuple(_mp(context, value) for value in state))
    delta_eta = _lap((1, 0, 0, 0, 0))
    for original, shifted, periodic, expected in zip(
        base.fine_rates, translated.fine_rates, rebranched.fine_rates, physical
    ):
        _encloses(context, original, expected)
        _encloses(context, shifted, expected)
        _encloses(context, periodic, expected)
    for i in range(5):
        assert (translated.eta[i] - base.eta[i]).contains(0)
        assert (rebranched.eta[i] - base.eta[i]).contains(delta_eta[i])
    original_reference = modal.harmonic_reference(base.forms, base.phases, time=0)
    changed_reference = modal.harmonic_reference(
        rebranched.forms, rebranched.phases, time=0
    )
    for i in range(5):
        assert (
            changed_reference.form_rates[i] - original_reference.form_rates[i]
        ).contains(-delta_eta[i])
    assert (
        base.continuous_lifts_retained and original_reference.continuous_lifts_retained
    )
    assert not base.harmonic_storage_is_complete_conserved_storage


def test_exact_input_admission_and_conditional_source_scope(certificate, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("raw admission occurred after scientific work")

    with monkeypatch.context() as patch:
        patch.setattr(modal, "complete_field", forbidden)
        patch.setattr(modal, "c5_modes", forbidden)
        patch.setattr(modal, "pi_interval", forbidden)
        for bad in (True, False, 0.0, float("nan"), float("inf"), "0", complex(0)):
            with pytest.raises((TypeError, ValueError)):
                modal.observe_modal_state((Q(0),) * 19 + (bad,))
            with pytest.raises((TypeError, ValueError)):
                modal.harmonic_reference((Q(0),) * 5, (Q(0),) * 4 + (bad,), time=0)
            with pytest.raises((TypeError, ValueError)):
                modal.harmonic_reference((Q(0),) * 5, (Q(0),) * 5, time=bad)
        for size in (0, 19, 21):
            with pytest.raises((TypeError, ValueError)):
                modal.observe_modal_state((Q(0),) * size)
        for size in (0, 4, 6):
            with pytest.raises((TypeError, ValueError)):
                modal.harmonic_reference((Q(0),) * size, (Q(0),) * 5, time=0)
        with pytest.raises((TypeError, ValueError)):
            modal.harmonic_reference((Q(0),) * 5, (Q(0),) * 5, time=I(-1, 0))
        forged = I(0)
        object.__setattr__(forged, "lo", False)
        with pytest.raises((TypeError, ValueError)):
            modal.observe_modal_state((Q(0),) * 19 + (forged,))
    source = certificate.source
    for changes in (
        dict(clock="rescaled structural clock"),
        dict(nominal=(source.nominal[0] + 1,) + source.nominal[1:]),
        dict(intended_coordinate_error=True),
        dict(numerical_enclosure_allowance=Q(0)),
        dict(proof_coordinate_error=source.proof_coordinate_error * 2),
        dict(initial_box=(I(0),) + source.initial_box[1:]),
        dict(storage_upper_bound=Q(49)),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(modal, "declared_source", lambda: replace(source, **changes))
            with pytest.raises((TypeError, ValueError)):
                modal.source_certificate()
    # Unrelated old verdicts are not the mathematical premises of this
    # source-specific criterion; consumed primitives are rebuilt above.
    with monkeypatch.context() as patch:
        patch.setattr(
            modal,
            "declared_source",
            lambda: replace(
                source,
                certified=False,
                global_chart_and_nontip_certified=False,
                early_exclusion_certified=False,
                original_control_exclusion_certified=False,
            ),
        )
        fresh = modal.source_certificate()
        assert fresh.certified
        assert fresh.reference_gap_upper_bound == certificate.reference_gap_upper_bound
        assert (
            fresh.necessary_deviation_lower_bound
            == certificate.necessary_deviation_lower_bound
        )
    assert certificate.certified and certificate.physical_chart_horizon == 6
    assert certificate.clock == "original structural t; both full rows retain 1/pi"
    assert not certificate.actual_acquisition_claimed
    assert not certificate.actual_transport_budget_evaluated
    assert not certificate.physical_complex_amplitude
    assert not certificate.harmonic_storage_is_complete_conserved_storage
    assert not certificate.empirical_validation and certificate.new_integrations == 0
    with pytest.raises(FrozenInstanceError):
        certificate.certified = False
