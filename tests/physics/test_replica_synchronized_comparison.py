"""Constructed controls for a source-based synchronized comparison allowance.

No actual/comparator trajectory or retained response is evaluated. Matrix
exponentials below are independent checks of the positive error majorant,
not solutions of the twenty-coordinate nonlinear law.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import mpmath
import pytest

from tnfr.mathematics._comparison_flow import nonnegative_exponential_upper
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research import replica_synchronized_comparison as comparison

ORACLE_DIGITS = 140
ORACLE_TOLERANCE = Q(1, 10**110)
POINT_WIDTH_CAP = Q(1, 10**30)


@pytest.fixture(scope="module")
def certificate():
    return comparison.source_certificate()


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


def _fine(forms, phases, internal_forms, internal_phases):
    return tuple(
        forms[i] + sign * internal_forms[i] for i in range(5) for sign in (1, -1)
    ) + tuple(
        phases[i] + sign * internal_phases[i] for i in range(5) for sign in (1, -1)
    )


def _means(values):
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


def _state():
    return _fine(
        (Q(1, 5), Q(-1, 4), Q(3, 10), Q(-2, 5), Q(3, 20)),
        (Q(-1, 3), Q(1, 4), Q(-1, 5), Q(2, 7), Q(1, 9)),
        (Q(1, 8), Q(-1, 6), Q(1, 7), Q(-1, 5), Q(1, 9)),
        (Q(1, 10), Q(-1, 8), Q(1, 6), Q(-1, 9), Q(1, 12)),
    )


def test_complete_two_state_rows_expose_only_the_collective_form_discrepancy():
    state = _state()
    result = comparison.observe_internal_modulation(state)
    context = _context()
    values = tuple(_mp(context, value) for value in state)
    x, theta = _means(values[:10]), _means(values[10:])
    synchronized = _fine(x, theta, (context.mpf(0),) * 5, (context.mpf(0),) * 5)
    actual_rates, reference_rates = _oracle(context, values), _oracle(
        context, synchronized
    )
    actual_x, actual_t = _means(actual_rates[:10]), _means(actual_rates[10:])
    reference_x, reference_t = _means(reference_rates[:10]), _means(
        reference_rates[10:]
    )
    deltas = tuple((values[10 + 2 * i] - values[11 + 2 * i]) / 2 for i in range(5))
    gaps = tuple(theta[(i + 1) % 5] - theta[i] for i in range(5))
    edge = tuple(
        (1 - context.cos(deltas[i]) * context.cos(deltas[(i + 1) % 5]))
        * context.sin(gaps[i])
        for i in range(5)
    )
    for bound, expected in zip(result.fine_rates, actual_rates):
        _encloses(context, bound, expected)
    for bound, expected in zip(result.synchronized_fine_rates, reference_rates):
        _encloses(context, bound, expected)
    for bound, expected in zip(result.synchronized_state, synchronized):
        _encloses(context, bound, expected)
    for i in range(5):
        _encloses(context, result.form_rates[i], actual_x[i])
        _encloses(context, result.phase_rates[i], actual_t[i])
        _encloses(context, result.synchronized_form_rates[i], reference_x[i])
        _encloses(context, result.synchronized_phase_rates[i], reference_t[i])
        _encloses(context, result.edge_modulation[i], edge[i])
        expected = (edge[(i - 1) % 5] - edge[i]) / (2 * context.pi)
        assert abs(actual_x[i] - reference_x[i] - expected) < _mp(
            context, ORACLE_TOLERANCE
        )
        assert abs(actual_t[i] - reference_t[i]) < _mp(context, ORACLE_TOLERANCE)
        _encloses(context, result.form_discrepancy[i], expected)
        _encloses(context, result.discrepancy_from_edges[i], expected)
        _encloses(context, result.phase_discrepancy[i], context.mpf(0))
    assert result.clock == "original structural t; both full rows retain 1/pi"
    assert (
        result.same_state_comparison and not result.evolved_prediction_error_evaluated
    )
    assert (
        not result.source_membership_inferred and not result.actual_acquisition_claimed
    )


def test_distinct_collective_states_retain_signed_secants_in_both_error_rows():
    # These are constructed static states, not a matched-initial-state
    # source pair or an evaluated comparison trajectory.
    state = _state()
    reference_state = _fine(
        (Q(2, 5), Q(-1, 10), Q(1, 2), Q(-3, 10), Q(1, 20)),
        (Q(3), Q(-2), Q(1), Q(-1), Q(2)),
        (Q(0),) * 5,
        (Q(0),) * 5,
    )
    actual = comparison.observe_internal_modulation(state)
    reference = comparison.observe_internal_modulation(reference_state)
    context = _context()
    tolerance = _mp(context, ORACLE_TOLERANCE)
    values = tuple(_mp(context, value) for value in state)
    reference_values = tuple(_mp(context, value) for value in reference_state)
    actual_rates = _oracle(context, values)
    reference_rates = _oracle(context, reference_values)
    for bound, expected in zip(actual.fine_rates, actual_rates):
        _encloses(context, bound, expected)
    for bound, expected in zip(reference.fine_rates, reference_rates):
        _encloses(context, bound, expected)
    x, theta = _means(values[:10]), _means(values[10:])
    reference_x, reference_theta = (
        _means(reference_values[:10]),
        _means(reference_values[10:]),
    )
    error_x = tuple(a - b for a, b in zip(x, reference_x))
    error_theta = tuple(a - b for a, b in zip(theta, reference_theta))
    gaps = tuple(theta[(i + 1) % 5] - theta[i] for i in range(5))
    reference_gaps = tuple(
        reference_theta[(i + 1) % 5] - reference_theta[i] for i in range(5)
    )
    gap_errors = tuple(a - b for a, b in zip(gaps, reference_gaps))
    assert all(value != 0 for value in gap_errors)
    secants = tuple(
        (context.sin(a) - context.sin(b)) / difference
        for a, b, difference in zip(gaps, reference_gaps, gap_errors)
    )
    assert min(secants) < 0 < max(secants)
    assert all(-1 <= value <= 1 for value in secants)
    deltas = tuple((values[10 + 2 * i] - values[11 + 2 * i]) / 2 for i in range(5))
    modulation = tuple(
        (1 - context.cos(deltas[i]) * context.cos(deltas[(i + 1) % 5]))
        * context.sin(gaps[i])
        for i in range(5)
    )
    actual_dx, actual_dt = _means(actual_rates[:10]), _means(actual_rates[10:])
    reference_dx, reference_dt = (
        _means(reference_rates[:10]),
        _means(reference_rates[10:]),
    )
    stiffness = context.matrix(5, 5)
    for i in range(5):
        preceding, following = (i - 1) % 5, (i + 1) % 5
        stiffness[i, i] = secants[preceding] + secants[i]
        stiffness[i, preceding] = -secants[preceding]
        stiffness[i, following] = -secants[i]
        signed_term = (
            secants[preceding] * gap_errors[preceding] - secants[i] * gap_errors[i]
        )
        residual = (modulation[preceding] - modulation[i]) / (2 * context.pi)
        expected_dx = -signed_term / (2 * context.pi) + residual
        expected_dt = (2 * error_x[i] - error_x[preceding] - error_x[following]) / (
            2 * context.pi
        )
        assert abs(actual_dx[i] - reference_dx[i] - expected_dx) < tolerance
        assert abs(actual_dt[i] - reference_dt[i] - expected_dt) < tolerance
        _encloses(context, actual.form_rates[i] - reference.form_rates[i], expected_dx)
        _encloses(
            context, actual.phase_rates[i] - reference.phase_rates[i], expected_dt
        )
        _encloses(context, actual.discrepancy_from_edges[i], residual)
    # The stiffness is symmetric but its edge secants are not all positive.
    # This independent spectral check exercises the global norm premise;
    # positivity of the actual dynamics is never assumed.
    eigenvalues = context.eigsy(stiffness, eigvals_only=True)
    lambda_plus = (5 + context.sqrt(5)) / 2
    assert max(abs(value) for value in eigenvalues) <= lambda_plus + tolerance
    applied = stiffness * context.matrix(error_theta)
    assert (
        sum(value**2 for value in applied)
        <= lambda_plus**2 * sum(value**2 for value in error_theta) + tolerance
    )


def test_synchronization_and_divergence_cancellation_are_not_activity_claims():
    context = _context()
    forms = (Q(1, 5), Q(-1, 4), Q(3, 10), Q(-2, 5), Q(3, 20))
    phases = (Q(1, 2), Q(0), Q(0), Q(0), Q(0))
    synchronized = comparison.observe_internal_modulation(
        _fine(forms, phases, (Q(0),) * 5, (Q(0),) * 5)
    )
    for i in range(5):
        assert synchronized.form_discrepancy[i].contains(0)
        assert synchronized.phase_discrepancy[i].contains(0)
        assert (
            synchronized.fine_rates[2 * i] - synchronized.fine_rates[2 * i + 1]
        ).contains(0)
        assert (
            synchronized.fine_rates[10 + 2 * i] - synchronized.fine_rates[11 + 2 * i]
        ).contains(0)
    # Zero phase separation gives zero instantaneous modulation even when
    # internal form is nonzero; that state is outside u=delta=0.
    u = (Q(1, 8), Q(1, 6), Q(-1, 7), Q(1, 5), Q(-1, 9))
    moving = comparison.observe_internal_modulation(
        _fine(forms, phases, u, (Q(0),) * 5)
    )
    for i in range(5):
        assert moving.form_discrepancy[i].contains(0)
        internal_phase_rate = (
            moving.fine_rates[10 + 2 * i] - moving.fine_rates[11 + 2 * i]
        ) / 2
        _encloses(context, internal_phase_rate, _mp(context, u[i]) / context.pi)
    # Uniform nonzero internal phase gives a signed, nonzero divergence.
    separated = comparison.observe_internal_modulation(
        _fine((Q(0),) * 5, phases, (Q(0),) * 5, (Q(1, 4),) * 5)
    )
    value = (
        context.sin(_mp(context, Q(1, 4))) ** 2
        * context.sin(_mp(context, Q(1, 2)))
        / context.pi
    )
    for i, expected in enumerate((value, -value / 2, 0, 0, -value / 2)):
        _encloses(context, separated.form_discrepancy[i], expected)
    assert separated.form_discrepancy[0].lo > 0
    assert separated.form_discrepancy[1].hi < 0
    # A uniform twist has nonzero edge modulation but zero divergence.
    # This constructed cancellation supplies no lower bound on an error.
    pi = pi_interval()
    twist = comparison.observe_internal_modulation(
        _fine(
            (Q(0),) * 5,
            tuple(2 * (i - 2) * pi / 5 for i in range(5)),
            (Q(1, 8),) * 5,
            (Q(1, 4),) * 5,
        )
    )
    expected_edge = context.sin(_mp(context, Q(1, 4))) ** 2 * context.sin(
        2 * context.pi / 5
    )
    for edge, divergence in zip(twist.edge_modulation, twist.form_discrepancy):
        _encloses(context, edge, expected_edge)
        _encloses(context, divergence, context.mpf(0))
    assert not twist.equal_internal_storage_claimed
    assert not twist.comparator_internal_activity_claimed


def test_source_profile_scales_control_each_quadratic_edge(certificate):
    source = certificate.source
    cover = Q(1, 2**20) + Q(1, 2**128)
    scale = Q(17, 287)
    d, v = Q(1, 50) + cover, 2 * scale + cover
    internal_u = tuple(
        (source.nominal[2 * i] - source.nominal[2 * i + 1]) / 2 for i in range(5)
    )
    internal_delta = tuple(
        (source.nominal[10 + 2 * i] - source.nominal[11 + 2 * i]) / 2 for i in range(5)
    )
    factors = tuple(
        max((value + cover) / v, (phase + cover) / d)
        for value, phase in zip(internal_u, internal_delta)
    )
    assert certificate.internal_phase_initial_upper == d
    assert certificate.internal_form_initial_upper == v
    assert certificate.internal_scale_factors == factors
    assert factors[-1] == 1 and all(0 < value < 1 for value in factors[:-1])
    for value, phase, factor in zip(internal_u, internal_delta, factors):
        assert value + cover <= factor * v
        assert phase + cover <= factor * d
    edges = tuple((factors[i] ** 2 + factors[(i + 1) % 5] ** 2) / 2 for i in range(5))
    assert certificate.edge_quadratic_factors == edges
    assert all(0 < value <= 1 for value in edges)
    assert sum(value**2 for value in edges) < 5
    context = _context()
    eigenvalue = (5 + context.sqrt(5)) / 2
    expected = context.sqrt(
        eigenvalue * sum(_mp(context, value**2) for value in edges)
    ) / (2 * context.pi)
    _encloses(context, certificate.forcing_coefficient_enclosure, expected)
    assert (
        certificate.forcing_coefficient_upper
        == certificate.forcing_coefficient_enclosure.hi
    )
    assert _mp(context, certificate.forcing_coefficient_upper) < context.sqrt(
        5 * eigenvalue
    ) / (2 * context.pi)
    assert _mp(context, certificate.alpha_upper) >= 1 / context.pi
    assert _mp(context, certificate.kappa_upper) >= eigenvalue / (2 * context.pi)
    alpha, kappa, forcing = (
        certificate.alpha_upper,
        certificate.kappa_upper,
        certificate.forcing_coefficient_upper,
    )
    assert certificate.comparison_matrix == (
        (0, kappa, forcing, 0, 0),
        (kappa, 0, 0, 0, 0),
        (0, 0, 0, 2 * alpha, 0),
        (0, 0, alpha, 0, alpha),
        (0, 0, 0, 2 * alpha, 0),
    )
    assert certificate.initial_envelope == (0, 0, d**2, d * v, v**2)
    # At any positive D,U, differentiating the quadratic monomials gives
    # the final three rows exactly, with no autonomous-model claim.
    D, U = Q(2, 7), Q(3, 8)
    moments = (Q(0), Q(0), D**2, D * U, U**2)
    derivative = tuple(
        sum(a * b for a, b in zip(row, moments))
        for row in certificate.comparison_matrix
    )
    assert derivative[2:] == (
        2 * D * alpha * U,
        alpha * (D**2 + U**2),
        2 * U * alpha * D,
    )


def test_positive_kernel_matches_an_independent_matrix_exponential(certificate):
    context = _context()
    tolerance = _mp(context, ORACLE_TOLERANCE)
    matrix = context.matrix(
        [
            [_mp(context, value) for value in row]
            for row in certificate.comparison_matrix
        ]
    )
    initial = context.matrix(
        [_mp(context, value) for value in certificate.initial_envelope]
    )
    assert tuple(bound.time for bound in certificate.bounds) == (Q(2), Q(6))
    for bound in certificate.bounds:
        kernel = bound.kernel
        expected = context.expm(_mp(context, bound.time) * matrix)
        assert kernel.matrix == certificate.comparison_matrix
        assert kernel.duration == bound.time
        for i in range(5):
            for j in range(5):
                assert (
                    _mp(context, kernel.partial_sum[i][j]) - tolerance <= expected[i, j]
                )
                assert expected[i, j] <= _mp(context, kernel.upper[i][j]) + tolerance
                assert (
                    kernel.upper[i][j]
                    == kernel.partial_sum[i][j] + kernel.tail_upper_bound[i][j]
                )
        propagated = expected * initial
        exact_upper = tuple(
            sum(a * b for a, b in zip(row, certificate.initial_envelope))
            for row in kernel.upper
        )
        assert bound.envelope_upper == exact_upper
        for i in range(5):
            assert propagated[i] <= _mp(context, bound.envelope_upper[i]) + tolerance
        alpha_t = _mp(context, certificate.alpha_upper * bound.time)
        D = _mp(context, certificate.internal_phase_initial_upper) * context.cosh(
            alpha_t
        ) + _mp(context, certificate.internal_form_initial_upper) * context.sinh(
            alpha_t
        )
        U = _mp(context, certificate.internal_form_initial_upper) * context.cosh(
            alpha_t
        ) + _mp(context, certificate.internal_phase_initial_upper) * context.sinh(
            alpha_t
        )
        for i, value in enumerate((D**2, D * U, U**2), 2):
            assert abs(propagated[i] - value) < tolerance
        assert bound.collective_form_error_upper == exact_upper[0]
        assert bound.collective_phase_error_upper == exact_upper[1]
        assert (
            context.sqrt(2) * propagated[1]
            <= _mp(context, bound.collective_gap_error_upper) + tolerance
        )
        assert (
            bound.target_erosion.error_bounds == (bound.collective_gap_error_upper,) * 2
        )
        assert bound.all_prior_times_covered
        assert (
            not bound.actual_error_evaluated
            and not bound.comparator_trajectory_evaluated
        )
    zero = nonnegative_exponential_upper(certificate.comparison_matrix, Q(0))
    identity = tuple(tuple(Q(i == j) for j in range(5)) for i in range(5))
    assert zero.upper == identity and zero.partial_sum == identity
    assert all(value == 0 for row in zero.tail_upper_bound for value in row)
    assert certificate.initial_envelope[:2] == (0, 0)


def test_joint_target_erosion_retains_equality_and_uncertain_contact():
    mu = Q(1, 100)
    zero = comparison.target_erosion(Q(0))
    exact = comparison.target_erosion(mu)
    assert zero.within_margin_certified and zero.jointly_nonempty_certified
    assert exact.error_bounds == (mu, mu)
    assert exact.within_margin_certified and not exact.margin_contact_unresolved
    rounded = comparison.target_erosion(I(mu))
    assert rounded.margin_contact_unresolved
    assert not rounded.within_margin_certified and not rounded.exceeds_margin_certified
    usable = comparison.target_erosion(Q(1, 4))
    empty = comparison.target_erosion(Q(1, 3))
    assert usable.exceeds_margin_certified and usable.jointly_nonempty_certified
    assert empty.jointly_empty_certified and not empty.joint_contact_unresolved
    threshold = pi_interval() / 10 - mu
    contact = comparison.target_erosion(threshold)
    assert contact.joint_contact_unresolved
    assert (
        not contact.jointly_nonempty_certified and not contact.jointly_empty_certified
    )
    # Exact symbolic coefficients of pi and mu prove the joint endpoint:
    # U-(pi/10-mu)=2pi/5, and five such gaps sum to2pi. Beyond it,
    # five upper limits cannot sum to2pi, although each interval can exist.
    assert Q(1, 2) - Q(1, 10) == Q(2, 5)
    assert -mu - (-mu) == 0
    assert 5 * Q(2, 5) == 2 and Q(1, 10) < Q(2, 5)
    context = _context()
    for result in (zero, exact, rounded, usable, empty, contact):
        _encloses(context, result.joint_threshold, context.pi / 10 - _mp(context, mu))
        assert not result.comparator_state_evaluated and not result.arrival_claimed
        assert not result.actual_error_lower_bound_claimed
        assert result.dilation_uses_integer_assignments
    with pytest.raises(FrozenInstanceError):
        empty.jointly_empty_certified = False


def test_raw_admission_precedes_fields_and_target_trigonometry(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("scientific work preceded raw input admission")

    monkeypatch.setattr(comparison, "complete_field", forbidden)
    monkeypatch.setattr(comparison, "pi_interval", forbidden)
    for bad in (True, False, 0.0, float("nan"), float("inf"), "0", complex(0)):
        with pytest.raises((TypeError, ValueError)):
            comparison.observe_internal_modulation((Q(0),) * 19 + (bad,))
        with pytest.raises((TypeError, ValueError)):
            comparison.target_erosion(bad)
    for size in (0, 19, 21):
        with pytest.raises((TypeError, ValueError)):
            comparison.observe_internal_modulation((Q(0),) * size)
    for lower, upper in ((False, Q(0)), (Q(1), Q(0))):
        forged = I(0)
        object.__setattr__(forged, "lo", lower)
        object.__setattr__(forged, "hi", upper)
        with pytest.raises((TypeError, ValueError)):
            comparison.observe_internal_modulation((Q(0),) * 19 + (forged,))
        with pytest.raises((TypeError, ValueError)):
            comparison.target_erosion(forged)
    with pytest.raises((TypeError, ValueError)):
        comparison.target_erosion(I(-1, 0))


def test_consumed_source_and_spectral_premises_are_not_cached_verdicts(
    certificate, monkeypatch
):
    source = certificate.source
    for changes in (
        dict(clock="another structural clock"),
        dict(nominal=(source.nominal[0] + 1,) + source.nominal[1:]),
        dict(intended_coordinate_error=True),
        dict(numerical_enclosure_allowance=Q(0)),
        dict(proof_coordinate_error=source.proof_coordinate_error * 2),
        dict(initial_box=(I(0),) + source.initial_box[1:]),
        dict(storage_upper_bound=Q(49)),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(
                comparison, "declared_source", lambda: replace(source, **changes)
            )
            with pytest.raises((TypeError, ValueError)):
                comparison.source_certificate()
    with monkeypatch.context() as patch:
        bad_geometry = replace(
            certificate.geometry,
            eigenvalues=certificate.geometry.eigenvalues[:2] + (I(4),),
        )
        patch.setattr(comparison, "c5_modes", lambda: bad_geometry)
        with pytest.raises((TypeError, ValueError)):
            comparison.source_certificate()
    with monkeypatch.context() as patch:
        patch.setattr(
            comparison,
            "declared_source",
            lambda: replace(
                source,
                certified=False,
                global_chart_and_nontip_certified=False,
                early_exclusion_certified=False,
            ),
        )
        rebuilt = comparison.source_certificate()
        assert (
            rebuilt.certified
            and rebuilt.comparison_matrix == certificate.comparison_matrix
        )
        assert rebuilt.initial_envelope == certificate.initial_envelope
        assert rebuilt.bounds == certificate.bounds
    assert certificate.certified and certificate.physical_chart_horizon == 6
    assert certificate.same_memberwise_collective_initial_state
    assert certificate.comparator_is_actual_invariant_restriction
    assert certificate.full_sine_retained and certificate.continuous_lifts_retained
    assert not certificate.equal_internal_storage_claimed
    assert not certificate.comparator_internal_activity_claimed
    assert not certificate.same_state_discrepancy_is_prediction_error
    assert not certificate.global_contraction_assumed
    assert (
        not certificate.actual_error_evaluated
        and not certificate.actual_acquisition_claimed
    )
    assert not certificate.retained_response_read and certificate.new_integrations == 0
    assert not certificate.empirical_validation
    with pytest.raises(FrozenInstanceError):
        certificate.certified = False
