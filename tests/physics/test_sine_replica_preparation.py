"""Static controls of one independently specified replica preparation class.

The unchanged conservative doubled-C5 field is evaluated only at declared
states and one formal tangent. No trajectory, acquisition, recurrence or
selected-source future endpoint is evaluated. The all-time obstruction is
the invariant-leaf and physical-phasor argument in SINE_REPLICA_PULSE.md.
"""

from fractions import Fraction as Q
from math import factorial

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sqrt
from tnfr.physics._sine_flow import _held_sine_neighbor_field
from tnfr.physics.relational_sine_scale import assess_sine_replica_scale

PAIRS = tuple((2 * i, 2 * i + 1) for i in range(5))
INTERNAL_FORM = (Q(1), Q(5, 4), Q(3, 2), Q(7, 4), Q(2))
INTERNAL_PHASE = (Q(1, 100), Q(3, 250), Q(7, 500), Q(2, 125), Q(1, 50))
EDGES = tuple(
    sorted(
        tuple(sorted((left, right)))
        for pair in range(5)
        for left in PAIRS[pair]
        for right in PAIRS[(pair + 1) % 5]
    )
)
NEIGHBORS = tuple(
    tuple(
        right if left == node else left
        for left, right in EDGES
        if node in (left, right)
    )
    for node in range(10)
)
# These allowances are fixed before the first evaluation. They concern only
# the independent 140-digit static oracle, not a source or trajectory error.
DECIMAL_PRECISION = 140
ORACLE_TOLERANCE = Q(1, 10**110)
INTERVAL_WIDTH_CAP = Q(1, 10**30)
SOURCE_STORAGE_CAP = Q(80) + Q(1, 125)
# Recorded in the prospective attempt-2 correction scope after preserving the
# first failed run. This is the fixed binary64 capture bridge, not a change to
# the original rational source or its independent-oracle tolerance.
PHASE_CAPTURE_ERROR_CAP = Q(1, 2**59)


def _mp(context, value):
    if isinstance(value, (int, Q)):
        value = Q(value)
        return context.mpf(value.numerator) / value.denominator
    return context.mpf(value)


def _oracle_context():
    context = mpmath.mp.clone()
    context.dps = DECIMAL_PRECISION
    return context


def _encloses(context, bound, value, *, width_cap=INTERVAL_WIDTH_CAP):
    tolerance = _mp(context, ORACLE_TOLERANCE)
    assert bound.hi - bound.lo < width_cap
    assert _mp(context, bound.lo) - tolerance <= value
    assert value <= _mp(context, bound.hi) + tolerance


def _captured_rate_width_cap():
    """Price the graph reader's existing 64-bit/12-term trig policy.

    At captured |delta|<1/40, the small-angle cosine width is bounded by C.
    S bounds the shifted-cosine sine width: its degree-22 series has omitted
    term at most 2**24/24!, plus the two 64-bit widenings and pi uncertainty.
    The factor eight covers the bounded coarse sums/products and original
    divisors; 2**-120 pays the finite 128-bit arithmetic. This width policy
    applies only to reader rates, never to the exact-source field or storage.
    """
    pi = pi_interval()
    cosine_width = Q(1, 40) ** 4 / 24 + Q(1, 2**63)
    sine_width = Q(2**24, factorial(24)) + Q(1, 2**62) + (pi.hi - pi.lo) / 2
    return 8 * (cosine_width + sine_width + Q(1, 2**120))


def _captured_companion(original, report):
    """Admit the separately represented graph capture, retaining both states."""
    comparison = report.comparison
    assert comparison.epi == original[:10]
    expected_phase = tuple(Q(float(value)) for value in original[10:])
    assert comparison.phase == expected_phase
    assert comparison.capacity == (Q(1),) * 10
    errors = tuple(
        captured - source for captured, source in zip(expected_phase, original[10:])
    )
    assert all(abs(value) <= PHASE_CAPTURE_ERROR_CAP for value in errors)
    assert all(errors[left] == -errors[right] for left, right in PAIRS)
    assert _means(comparison.epi) == _means(comparison.phase) == (Q(0),) * 5
    assert sum(errors) == sum(comparison.phase) == sum(comparison.epi) == 0
    deltas = _half_differences(comparison.phase)
    assert deltas == tuple(Q(float(value)) for value in INTERNAL_PHASE)
    assert all(0 < value < Q(1, 40) for value in deltas)
    # The rounded upper endpoint is outside the original closed source box.
    # This companion retains the common-mean leaf, not that box membership.
    assert deltas[-1] > Q(1, 50)
    return comparison.epi + comparison.phase


def _means(values):
    return tuple((values[left] + values[right]) / 2 for left, right in PAIRS)


def _half_differences(values):
    return tuple((values[left] - values[right]) / 2 for left, right in PAIRS)


def _state(means, phase_means, internal_forms, internal_phases):
    return tuple(
        means[pair] + sign * internal_forms[pair]
        for pair in range(5)
        for sign in (1, -1)
    ) + tuple(
        phase_means[pair] + sign * internal_phases[pair]
        for pair in range(5)
        for sign in (1, -1)
    )


def _fine_oracle(context, state):
    """Independent nodal sums, with fine degree four and original time t."""
    forms, phases = state[:10], state[10:]
    form_rates, phase_rates = [], []
    for node in range(10):
        pair = node // 2
        neighbors = tuple(
            2 * adjacent + member
            for adjacent in ((pair - 1) % 5, (pair + 1) % 5)
            for member in (0, 1)
        )
        form_rates.append(
            sum(context.sin(phases[other] - phases[node]) for other in neighbors)
            / (4 * context.pi)
        )
        phase_rates.append(
            sum(forms[node] - forms[other] for other in neighbors) / (4 * context.pi)
        )
    return tuple(form_rates + phase_rates)


def _fine_storage(context, state):
    return sum(
        (state[left] - state[right]) ** 2 / 2
        + 1
        - context.cos(state[10 + right] - state[10 + left])
        for left, right in EDGES
    )


def _storage_interval(state):
    state = tuple(I.coerce(value) for value in state)
    return sum(
        (
            (state[left] - state[right]) ** 2 / 2
            + 1
            - cos(state[10 + right] - state[10 + left])
            for left, right in EDGES
        ),
        I(0),
    )


@pytest.fixture(scope="module")
def prepared_source():
    model = RelationalExchangeModel(
        1, epi_weight=0, phase_weight=1, phase_domain="regular"
    )
    state = _state((Q(0),) * 5, (Q(0),) * 5, INTERNAL_FORM, INTERNAL_PHASE)
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    graph.add_edges_from(EDGES, weight=Q(1))
    for node in graph:
        graph.nodes[node].update(EPI=state[node], theta=state[10 + node], nu_f=Q(1))
    graph.graph["GAMMA"] = {"type": "none"}
    field = _held_sine_neighbor_field(
        model,
        (4,) * 10,
        (Q(1),) * 10,
        tuple(tuple((neighbor, 1, Q(1)) for neighbor in row) for row in NEIGHBORS),
    )
    rates = field(state)
    report = assess_sine_replica_scale(graph, reference_model=model, pairs=PAIRS)
    return state, field, rates, report


def test_complete_twenty_rows_match_independent_original_clock_oracle(prepared_source):
    state, _, rates, report = prepared_source
    context = _oracle_context()
    exact_state = tuple(_mp(context, value) for value in state)
    oracle = _fine_oracle(context, exact_state)
    captured = _captured_companion(state, report)
    captured_state = tuple(_mp(context, value) for value in captured)
    captured_oracle = _fine_oracle(context, captured_state)
    captured_width_cap = _captured_rate_width_cap()
    assert len(EDGES) == len(set(EDGES)) == 20
    assert tuple(map(len, NEIGHBORS)) == (4,) * 10
    assert report.comparison.nodes == tuple(range(10))
    assert report.comparison.edges == EDGES
    assert report.comparison.capacity == (Q(1),) * 10
    assert report.comparison.degrees == (4,) * 10
    assert len(rates) == len(oracle) == 20
    for actual, expected in zip(rates, oracle):
        _encloses(context, actual, expected)
    for actual, expected in zip(
        report.comparison.form_rates + report.comparison.phase_rates, captured_oracle
    ):
        _encloses(context, actual, expected, width_cap=captured_width_cap)
    capture_error = _mp(context, PHASE_CAPTURE_ERROR_CAP)
    for original_rate, captured_rate in zip(oracle[:10], captured_oracle[:10]):
        assert abs(captured_rate - original_rate) <= 2 * capture_error / context.pi
    assert captured_oracle[10:] == oracle[10:]
    # Both complete rows are active initially, even though their pair means
    # vanish. This is not an all-time monotonicity or pulse-period assertion.
    for pair in range(5):
        assert rates[2 * pair].hi < 0 < rates[2 * pair + 1].lo
        assert rates[10 + 2 * pair].lo > 0 > rates[11 + 2 * pair].hi
        assert report.comparison.form_rates[2 * pair].hi < 0
        assert report.comparison.form_rates[2 * pair + 1].lo > 0
        assert report.comparison.phase_rates[2 * pair].lo > 0
        assert report.comparison.phase_rates[2 * pair + 1].hi < 0
        _encloses(
            context,
            rates[10 + 2 * pair],
            _mp(context, INTERNAL_FORM[pair]) / context.pi,
        )
    assert abs(sum(oracle[:10])) < _mp(context, ORACLE_TOLERANCE)
    assert abs(sum(oracle[10:])) < _mp(context, ORACLE_TOLERANCE)


def test_replica_factorization_retains_activity_and_finite_source_budget(
    prepared_source, record_property
):
    state, _, rates, report = prepared_source
    context = _oracle_context()
    exact_state = tuple(_mp(context, value) for value in state)
    captured = _captured_companion(state, report)
    captured_state = tuple(_mp(context, value) for value in captured)
    captured_deltas = _half_differences(captured[10:])
    expected = _fine_oracle(context, captured_state)
    captured_width_cap = _captured_rate_width_cap()
    assert all(1 <= value <= 2 for value in INTERNAL_FORM)
    assert all(Q(1, 100) <= value <= Q(1, 50) for value in INTERNAL_PHASE)
    assert len(set(INTERNAL_FORM)) == len(set(INTERNAL_PHASE)) == 5
    assert report.form_means == report.phase_mean_radian_parts == (Q(0),) * 5
    assert report.phase_mean_turn_parts == (Q(0),) * 5
    assert report.form_half_differences == INTERNAL_FORM
    assert report.phase_half_difference_radian_parts == captured_deltas
    assert not report.source_in_synchronized_submanifold
    for actual, factored, residuals, values in (
        (
            report.mean_form_rates,
            report.factored_mean_form_rates,
            report.mean_form_factorization_residual,
            _means(expected[:10]),
        ),
        (
            report.mean_phase_rates,
            report.factored_mean_phase_rates,
            report.mean_phase_factorization_residual,
            _means(expected[10:]),
        ),
        (
            report.internal_form_rates,
            report.factored_internal_form_rates,
            report.internal_form_factorization_residual,
            _half_differences(expected[:10]),
        ),
        (
            report.internal_phase_rates,
            report.factored_internal_phase_rates,
            report.internal_phase_factorization_residual,
            _half_differences(expected[10:]),
        ),
    ):
        for direct, factor, residual, value in zip(actual, factored, residuals, values):
            _encloses(context, direct, value, width_cap=captured_width_cap)
            _encloses(context, factor, value, width_cap=captured_width_cap)
            assert residual.contains(0)
    assert all(bound.contains(0) for bound in _means(rates[:10]) + _means(rates[10:]))
    for pair in range(5):
        delta = _mp(context, captured_deltas[pair])
        neighbor_cosines = sum(
            context.cos(_mp(context, captured_deltas[other]))
            for other in ((pair - 1) % 5, (pair + 1) % 5)
        )
        internal_rate = -context.sin(delta) * neighbor_cosines / (2 * context.pi)
        _encloses(
            context,
            report.internal_form_rates[pair],
            internal_rate,
            width_cap=captured_width_cap,
        )
    storage = _fine_storage(context, exact_state)
    separated_storage = _mp(
        context, 4 * sum(value**2 for value in INTERNAL_FORM)
    ) + 4 * sum(
        1
        - context.cos(_mp(context, INTERNAL_PHASE[pair]))
        * context.cos(_mp(context, INTERNAL_PHASE[(pair + 1) % 5]))
        for pair in range(5)
    )
    assert abs(storage - separated_storage) < _mp(context, ORACLE_TOLERANCE)
    original_storage_bounds = _storage_interval(state)
    _encloses(context, original_storage_bounds, storage)
    assert 20 < original_storage_bounds.lo <= original_storage_bounds.hi
    assert original_storage_bounds.hi < SOURCE_STORAGE_CAP
    captured_storage = _fine_storage(context, captured_state)
    _encloses(context, report.comparison.storage, captured_storage)
    assert abs(captured_storage - storage) <= 40 * _mp(context, PHASE_CAPTURE_ERROR_CAP)
    # Fresh pointwise bounds for the represented companion. The exact source
    # family budget is independently checked above, not inferred from capture.
    assert 20 < report.comparison.storage.lo <= report.comparison.storage.hi
    assert report.comparison.storage.hi < SOURCE_STORAGE_CAP
    assert report.comparison.continuous_loss == 0
    assert report.comparison.storage_rate.contains(0)
    assert report.storage_decomposition_residual.contains(0)
    # The family ceiling comes from 4*sum(u_i**2)<=80 and
    # 4*sum_edges(1-cos(delta_i)*cos(delta_j))<=4*sum(delta_i**2).
    assert 4 * 5 * Q(1, 50) ** 2 == Q(1, 125)
    record_property("declared_source_storage_upper", str(SOURCE_STORAGE_CAP))
    record_property("source_storage_lower", str(original_storage_bounds.lo))
    record_property("source_storage_upper", str(original_storage_bounds.hi))
    record_property("captured_storage_lower", str(report.comparison.storage.lo))
    record_property("captured_storage_upper", str(report.comparison.storage.hi))
    record_property("phase_capture_error_upper", str(PHASE_CAPTURE_ERROR_CAP))
    record_property("captured_rate_width_cap", str(captured_width_cap))
    record_property("captured_companion_in_original_closed_box", "false")
    record_property("evidence_scope", "static_source_class_obstruction_no_trajectory")


def test_collinear_pair_phasors_survive_chart_changes_and_exclude_acute_winding(
    prepared_source,
):
    _, field, _, _ = prepared_source
    context = _oracle_context()
    tolerance = _mp(context, ORACLE_TOLERANCE)
    pi = pi_interval()
    origin = Q(2, 7)
    # Explicit continuous lifts, including cases outside the initial family
    # and outside the old reader's short-pair chart. They are not trajectories.
    patterns = (
        ((Q(0),) * 5, (1,) * 5),
        ((Q(1),) * 5, (-1,) * 5),
        ((Q(0), Q(1), Q(0), Q(1), Q(0)), (1, -1, 1, -1, 1)),
        ((Q(1, 2), Q(0), Q(0), Q(0), Q(0)), (0, 1, 1, 1, 1)),
    )
    for coefficients, signs in patterns:
        deltas = tuple(value * pi for value in coefficients)
        state = _state((Q(0),) * 5, (origin,) * 5, INTERNAL_FORM, deltas)
        rates = field(state)
        exact_deltas = tuple(_mp(context, value) * context.pi for value in coefficients)
        exact_state = _state(
            (context.mpf(0),) * 5,
            (_mp(context, origin),) * 5,
            tuple(_mp(context, value) for value in INTERNAL_FORM),
            exact_deltas,
        )
        oracle = _fine_oracle(context, exact_state)
        for actual, expected in zip(rates, oracle):
            _encloses(context, actual, expected)
        assert all(
            bound.contains(0) for bound in _means(rates[:10]) + _means(rates[10:])
        )
        phasors = tuple(
            (
                context.exp(1j * exact_state[10 + left])
                + context.exp(1j * exact_state[10 + right])
            )
            / 2
            for left, right in PAIRS
        )
        for phasor, sign in zip(phasors, signs):
            expected = sign * context.exp(1j * _mp(context, origin))
            assert abs(phasor - expected) < tolerance
        if 0 in signs:
            assert cos(deltas[signs.index(0)]).contains(0)
            assert abs(phasors[signs.index(0)]) < tolerance
        elif len(set(signs)) > 1:
            assert any(signs[pair] * signs[(pair + 1) % 5] < 0 for pair in range(5))
            assert any(
                context.re(context.conj(phasors[pair]) * phasors[(pair + 1) % 5]) < 0
                for pair in range(5)
            )
        else:
            gaps = tuple(
                context.arg(context.conj(phasors[pair]) * phasors[(pair + 1) % 5])
                for pair in range(5)
            )
            assert all(abs(gap) < tolerance for gap in gaps)
            assert abs(sum(gaps) / (2 * context.pi)) < tolerance
    # A chart change retains the physical phases: Theta -> Theta+pi,
    # delta -> delta-pi leaves the plus lift and adds 2pi to the minus lift.
    delta, theta = _mp(context, Q(1, 50)), _mp(context, origin)
    original = (
        context.exp(1j * (theta + delta)) + context.exp(1j * (theta - delta))
    ) / 2
    shifted = (
        context.exp(1j * (theta + context.pi + delta - context.pi))
        + context.exp(1j * (theta + context.pi - delta + context.pi))
    ) / 2
    assert abs(original - shifted) < tolerance
    reconstructed = context.exp(1j * (theta + context.pi)) * context.cos(
        delta - context.pi
    )
    assert abs(shifted - reconstructed) < tolerance


def test_equal_energy_and_global_means_do_not_remove_the_source_obstruction(
    prepared_source, record_property
):
    state, _, _, _ = prepared_source
    context = _oracle_context()
    tolerance = _mp(context, ORACLE_TOLERANCE)
    source = tuple(_mp(context, value) for value in state)
    energy = _fine_storage(context, source)
    alpha = 2 * context.pi / 5
    minimum = 20 * (1 - context.cos(alpha))
    assert 0 < minimum < 20 < energy
    v = context.sqrt((energy - minimum) / 20)
    target = _state(
        (context.mpf(0),) * 5,
        tuple((pair - 2) * alpha for pair in range(5)),
        (v,) * 5,
        (context.mpf(0),) * 5,
    )
    assert abs(_fine_storage(context, target) - energy) < tolerance
    assert abs(sum(source[:10]) - sum(target[:10])) < tolerance
    assert abs(sum(source[10:]) - sum(target[10:])) < tolerance
    phasors = tuple(
        (context.exp(1j * target[10 + left]) + context.exp(1j * target[10 + right])) / 2
        for left, right in PAIRS
    )
    gaps = tuple(
        context.arg(context.conj(phasors[pair]) * phasors[(pair + 1) % 5])
        for pair in range(5)
    )
    assert all(0 < gap < context.pi / 2 for gap in gaps)
    assert abs(sum(gaps) - 2 * context.pi) < tolerance
    assert all(abs(rate) > 0 for rate in _fine_oracle(context, target)[10:])
    # This is an algebraic equal-invariant state, not a reached endpoint or a
    # retained pulse. Its internal energy exceeds the pulse's libration barrier.
    assert v**2 > context.cos(alpha)
    pi = pi_interval()
    minimum_bounds = 20 * (1 - cos(2 * pi / 5))
    # Match the original rational source, not the separately represented
    # graph capture; outward arithmetic retains the original energy relation.
    v_bounds = sqrt((_storage_interval(state) - minimum_bounds) / 20)
    target_bounds = _state(
        (Q(0),) * 5,
        tuple(2 * Q(pair - 2, 5) * pi for pair in range(5)),
        (v_bounds,) * 5,
        (Q(0),) * 5,
    )
    _encloses(context, v_bounds, v)
    _encloses(context, _storage_interval(target_bounds), energy)
    record_property("target_minimum_lower", str(minimum_bounds.lo))
    record_property("target_minimum_upper", str(minimum_bounds.hi))
    record_property("matched_target_is_reached_endpoint", "false")


def test_nonuniform_mean_form_releases_the_constraint_without_proving_formation(
    prepared_source,
):
    _, field, _, _ = prepared_source
    context = _oracle_context()
    # A separate directional control at delta=0, outside the declared source
    # box. Only the common-X restriction is released; no duration is advanced.
    means = (Q(1, 16), Q(-1, 16), Q(0), Q(0), Q(0))
    state = _state(means, (Q(0),) * 5, INTERNAL_FORM, (Q(0),) * 5)
    rates = field(state)
    laplacian = tuple(
        2 * means[pair] - means[(pair - 1) % 5] - means[(pair + 1) % 5]
        for pair in range(5)
    )
    laplacian_squared = tuple(
        2 * laplacian[pair] - laplacian[(pair - 1) % 5] - laplacian[(pair + 1) % 5]
        for pair in range(5)
    )
    assert sum(means) == 0 and any(value != 0 for value in laplacian)
    for bound, value in zip(_means(rates[10:]), laplacian):
        _encloses(context, bound, _mp(context, value) / (2 * context.pi))
        if value != 0:
            assert not bound.contains(0)
    assert all(bound.contains(0) for bound in rates[:10])
    direction = tuple(Jet((I(value), rate)) for value, rate in zip(state, rates))
    tangent = field(direction)
    exact_state = tuple(_mp(context, value) for value in state)
    oracle = _fine_oracle(context, exact_state)
    acceleration = []
    for node in range(10):
        pair = node // 2
        neighbors = tuple(
            2 * adjacent + member
            for adjacent in ((pair - 1) % 5, (pair + 1) % 5)
            for member in (0, 1)
        )
        acceleration.append(
            sum(
                context.cos(exact_state[10 + other] - exact_state[10 + node])
                * (oracle[10 + other] - oracle[10 + node])
                for other in neighbors
            )
            / (4 * context.pi)
        )
    acceleration.extend(
        sum(oracle[node] - oracle[other] for other in NEIGHBORS[node])
        / (4 * context.pi)
        for node in range(10)
    )
    for result, first, second in zip(tangent, oracle, acceleration):
        _encloses(context, result.coeffs[0], first)
        _encloses(context, result.coeffs[1], second)
    mean_acceleration = _means(tuple(result.coeffs[1] for result in tangent[:10]))
    for bound, value in zip(mean_acceleration, laplacian_squared):
        _encloses(context, bound, -_mp(context, value) / (4 * context.pi**2))
    assert any(not bound.contains(0) for bound in mean_acceleration)


def test_near_symmetry_growth_bound_uses_actual_signed_collective_rows(prepared_source):
    _, field, _, _ = prepared_source
    context = _oracle_context()
    tolerance = _mp(context, ORACLE_TOLERANCE)
    means = (Q(1, 500), Q(-1, 500), Q(0), Q(0), Q(0))
    phase_means = (Q(0), Q(1, 500), Q(-1, 500), Q(0), Q(0))
    # The differential estimate admits arbitrary internal phases. Mixed signs
    # are tested outside the source box; its energy/activity claims do not
    # transfer to this independently constructed contrast state.
    deltas = (Q(1, 100), Q(2), Q(-3), Q(1, 50), Q(4))
    state = _state(means, phase_means, INTERNAL_FORM, deltas)
    rates = field(state)
    exact_state = tuple(_mp(context, value) for value in state)
    oracle = _fine_oracle(context, exact_state)
    for actual, expected in zip(rates, oracle):
        _encloses(context, actual, expected)
    x = tuple(_mp(context, value) for value in means)
    theta = tuple(_mp(context, value) for value in phase_means)
    resultants = tuple(context.cos(_mp(context, value)) for value in deltas)
    assert min(resultants) < 0 < max(resultants)
    signed_laplacian = [[context.mpf(0) for _ in range(5)] for _ in range(5)]
    for pair in range(5):
        neighbor = (pair + 1) % 5
        gap = theta[neighbor] - theta[pair]
        sinc = context.sin(gap) / gap if gap else context.mpf(1)
        coefficient = resultants[pair] * resultants[neighbor] * sinc
        assert abs(coefficient) <= 1
        signed_laplacian[pair][pair] += coefficient
        signed_laplacian[neighbor][neighbor] += coefficient
        signed_laplacian[pair][neighbor] -= coefficient
        signed_laplacian[neighbor][pair] -= coefficient
    assert all(
        abs(sum(row)) < tolerance and sum(abs(value) for value in row) <= 4
        for row in signed_laplacian
    )
    assert all(
        signed_laplacian[row][column] == signed_laplacian[column][row]
        for row in range(5)
        for column in range(5)
    )
    mean_form_rates, mean_phase_rates = _means(oracle[:10]), _means(oracle[10:])
    for pair in range(5):
        expected_form = -sum(
            signed_laplacian[pair][column] * theta[column] for column in range(5)
        ) / (2 * context.pi)
        expected_phase = (2 * x[pair] - x[(pair - 1) % 5] - x[(pair + 1) % 5]) / (
            2 * context.pi
        )
        assert abs(mean_form_rates[pair] - expected_form) < tolerance
        assert abs(mean_phase_rates[pair] - expected_phase) < tolerance
    assert abs(sum(mean_form_rates)) < tolerance
    assert abs(sum(mean_phase_rates)) < tolerance
    x_squared = sum(value**2 for value in x)
    theta_squared = sum(value**2 for value in theta)
    assert (
        sum(value**2 for value in mean_form_rates) <= 4 * theta_squared / context.pi**2
    )
    assert sum(value**2 for value in mean_phase_rates) <= 4 * x_squared / context.pi**2
    norm_squared_rate = 2 * sum(
        value * rate
        for value, rate in zip(x + theta, mean_form_rates + mean_phase_rates)
    )
    assert norm_squared_rate <= 4 * (x_squared + theta_squared) / context.pi
    initial_norm_squared = sum(value**2 for value in means + phase_means)
    assert 0 < initial_norm_squared < Q(1, 100) ** 2
    # No exponential or temporal response is evaluated. The geometric-series
    # majorant exp(z)<=1/(1-z), 0<=z<1, and pi>3 give exp(2/pi)<3.
    pi = pi_interval()
    assert pi.lo > 3
    assert Q(2) / pi.lo < Q(2, 3)
    assert 1 / (1 - Q(2, 3)) == 3
    assert Q(3, 100) < 1 and pi.lo**2 > 8
