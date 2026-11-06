"""Independent crossing-time and saltation algebra at the exact carrier.

No first hit is located and no hybrid trajectory is evaluated. The controls
check local derivatives of an admitted crossing, including both continuous
rows and the state/level dependence of its occurrence time. Exact energy
jets also distinguish equal collective summaries under one fixed guard.
"""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT
from tnfr.operators.definitions_base import Operator
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics import _organized_source_guard_sensitivity as owner

# Independent incidence, not a consumer of the report's geometry inventory.
EDGES = tuple(
    (9 * child + slot, 9 * child + (slot + 1) % 9)
    for child in range(5)
    for slot in range(9)
) + tuple(
    (9 * a + slot, 9 * b + slot)
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (0, 4))
    for slot in range(9)
)
NEIGHBORS = tuple(
    tuple(b if a == node else a for a, b in EDGES if node in (a, b))
    for node in range(45)
)
ZERO = (Q(0),) * 45
MAIN = (Q(1, 64), -Q(1, 64)) + ZERO[2:]
ASYMMETRIC = ZERO[:2] + (Q(1, 1000), -Q(1, 1000)) + ZERO[4:]
PROFILE_A = (0, 0, 0, 1, 0, 0, -1, 0, 0)
PROFILE_B = (0, 0, 0, 1, 0, 0, 0, -1, 0)
PAIR_SCALE = Q(1, 128)
PAIR_A = tuple(PAIR_SCALE * value for value in PROFILE_A) * 5
PAIR_B = tuple(PAIR_SCALE * value for value in PROFILE_B) * 5
GENERAL = tuple(Q((3 * node) % 7 - 3, 4096) for node in range(45))


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Guard sensitivity cannot execute events or trajectories")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def factors():
    registered = canonical_glyph_factor_defaults()
    return {
        "reception_mix": Q(registered["EN_mix"]),
        "phase_coefficient": Q(DEFAULT_PHASE_LOCKING_COEFFICIENT),
        "silence_capacity_factor": Q(registered["SHA_vf_factor"]),
    }


@pytest.fixture(scope="module")
def report(factors):
    return owner._assess_organized_source_carrier_guard_sensitivity(MAIN, **factors)


def _gradient(values):
    result = [Q(0)] * 45
    for left, right in EDGES:
        result[left] += values[left] - values[right]
        result[right] += values[right] - values[left]
    return tuple(result)


def _dot(left, right):
    return sum((a * b for a, b in zip(left, right)), Q(0))


def _cycle_gradient(values):
    return tuple(
        2 * values[j] - values[(j - 1) % 9] - values[(j + 1) % 9] for j in range(9)
    )


def _reception(values, beta):
    result = list(values)
    for target in (0, 1):
        result[target] = (1 - beta) * values[target] + beta * sum(
            (values[j] for j in NEIGHBORS[target]), Q(0)
        ) / len(NEIGHBORS[target])
    return tuple(result)


def _mp(context, value):
    return context.mpf(value.numerator) / int(value.denominator)


def _phase_reset_derivative(context, vector, coefficient):
    carrier = tuple(2 * context.pi * (node % 9 - 4) / 9 for node in range(45))
    result = list(vector)
    for target in (0, 1):
        phasors = tuple(
            context.exp(context.j * (carrier[j] - carrier[target]))
            for j in NEIGHBORS[target]
        )
        relative_derivative = context.fsum(
            phasor * (vector[j] - vector[target])
            for phasor, j in zip(phasors, NEIGHBORS[target])
        ) / context.fsum(phasors)
        result[target] += _mp(context, coefficient) * relative_derivative.real
    return tuple(result)


def _normalized_phase(context, base, matching, cycle):
    cosine = context.cos(2 * context.pi / 9)
    return tuple(
        _mp(context, a)
        + (_mp(context, b) + cosine * _mp(context, c)) / (3 + 2 * cosine)
        for a, b, c in zip(base, matching, cycle)
    )


def test_exact_main_crossing_and_fixed_level_response(report, factors):
    q = _gradient(MAIN)
    assert report.timing.form_gradient == q
    assert (
        report.guard_energy_level
        == sum(((MAIN[b] - MAIN[a]) ** 2 / 2 for a, b in EDGES), Q(0))
        == Q(3, 2048)
    )
    assert report.energy_loss_rate == _dot(q, q) / 4 == Q(5, 1024)
    assert report.event_time_form_covector == tuple(
        value / report.energy_loss_rate for value in q
    )
    assert report.event_time_phase_covector == ZERO
    assert report.level_time_derivative == -Q(1024, 5)
    contrast = report.level_form_derivative[8] - report.level_form_derivative[2]
    assert contrast == -48 * factors["reception_mix"] / 25 < 0
    assert report.level_phase_base[8] - report.level_phase_base[2] == -contrast
    for node in (8, 2):
        assert report.level_phase_matching[node] == 0
        assert report.level_phase_cycle[node] == 0
    phase_contrast = report.level_phase_bounds[8] - report.level_phase_bounds[2]
    assert phase_contrast.lo > 0
    mp = mpmath.mp.clone()
    mp.dps = 100
    expected_phase = 48 * _mp(mp, factors["reception_mix"]) / (25 * 1023 * mp.pi)
    assert _mp(mp, phase_contrast.lo) <= expected_phase <= _mp(mp, phase_contrast.hi)

    # A perturbation of node zero changes this remote observable only through
    # its occurrence time: the reset Jacobian alone has zero contrast here.
    direction = (Q(1),) + ZERO[1:]
    time_change = _dot(report.event_time_form_covector, direction)
    assert time_change == Q(96, 5)
    reset_direction = _reception(direction, factors["reception_mix"])
    assert reset_direction[8] - reset_direction[2] == 0
    reset_phase_direction = _phase_reset_derivative(
        mp, (mp.mpf(0),) * 45, factors["phase_coefficient"]
    )
    assert reset_phase_direction[8] - reset_phase_direction[2] == 0
    correction = time_change * (
        report.timing.form_commutator[8] - report.timing.form_commutator[2]
    )
    assert correction == 9 * factors["reception_mix"] / 50 > 0
    normalized_phase = _normalized_phase(
        mp,
        report.timing.phase_commutator_base,
        report.timing.phase_commutator_matching,
        report.timing.phase_commutator_cycle,
    )
    phase_correction = (
        _mp(mp, time_change)
        * (normalized_phase[8] - normalized_phase[2])
        / (1023 * mp.pi)
    )
    assert abs(phase_correction + _mp(mp, correction) / (1023 * mp.pi)) < mp.mpf(
        "1e-90"
    )
    assert phase_correction < 0


@pytest.mark.parametrize(
    "direction",
    (
        (Q(1),) + ZERO[1:],
        ZERO[:8] + (Q(1),) + ZERO[9:],
        tuple(Q((3 * i) % 7 - 3, 8) for i in range(45)),
        (Q(1),) * 45,
    ),
)
def test_implicit_crossing_constraint_has_the_complete_time_correction(
    report, direction
):
    q = _gradient(MAIN)
    before_form_rate = tuple(-value / 4 for value in q)
    time_change = _dot(report.event_time_form_covector, direction)
    crossing_direction = tuple(
        v + rate * time_change for v, rate in zip(direction, before_form_rate)
    )
    # Differentiating E(Phi_t(s)) = fixed level gives this zero, not q.v=0.
    assert _dot(q, crossing_direction) == 0
    assert _dot(q, before_form_rate) * report.level_time_derivative == 1


@pytest.mark.parametrize("form", (MAIN, ASYMMETRIC))
@pytest.mark.parametrize("coefficient", (Q(0), Q(0.3), Q(1)))
def test_saltation_maps_both_incoming_rows_to_actual_successor_rows(
    factors, form, coefficient
):
    result = owner._assess_organized_source_carrier_guard_sensitivity(
        form, **{**factors, "phase_coefficient": coefficient}
    )
    q = _gradient(form)
    after_form = _reception(form, factors["reception_mix"])
    after_q = _gradient(after_form)
    capacities = (factors["silence_capacity_factor"],) * 2 + (Q(1),) * 43
    before_x_rate = tuple(-value / 4 for value in q)
    after_x_rate = tuple(-nu * value / 4 for nu, value in zip(capacities, after_q))
    reset_x_rate = _reception(before_x_rate, factors["reception_mix"])
    time_change = _dot(result.event_time_form_covector, before_x_rate)
    assert time_change == -1
    complete_x_rate = tuple(
        reset + timing * time_change
        for reset, timing in zip(reset_x_rate, result.timing.form_commutator)
    )
    assert complete_x_rate == after_x_rate
    assert reset_x_rate != after_x_rate

    mp = mpmath.mp.clone()
    mp.dps = 100
    gamma = 1 / (1023 * mp.pi)
    before_phase_rate = tuple(gamma * _mp(mp, value) / 4 for value in q)
    after_phase_rate = tuple(
        gamma * _mp(mp, nu * value) / 4 for nu, value in zip(capacities, after_q)
    )
    reset_phase_rate = _phase_reset_derivative(mp, before_phase_rate, coefficient)
    normalized_timing = _normalized_phase(
        mp,
        result.timing.phase_commutator_base,
        result.timing.phase_commutator_matching,
        result.timing.phase_commutator_cycle,
    )
    normalized_level = _normalized_phase(
        mp,
        result.level_phase_base,
        result.level_phase_matching,
        result.level_phase_cycle,
    )
    for index, expected in enumerate(after_phase_rate):
        actual = reset_phase_rate[index] + gamma * normalized_timing[index] * _mp(
            mp, time_change
        )
        assert abs(actual - expected) < mp.mpf("1e-90")
        # Rebuild the level response directly from independent fields and the
        # phasor derivative, instead of only comparing two report fields.
        level_response = (expected - reset_phase_rate[index]) / _mp(
            mp, result.energy_loss_rate
        )
        assert abs(gamma * normalized_level[index] - level_response) < mp.mpf("1e-90")
        bounds = result.level_phase_bounds[index]
        assert _mp(mp, bounds.lo) <= level_response <= _mp(mp, bounds.hi)


def test_phase_only_direction_has_zero_first_order_time_change(report):
    phase_direction = tuple(Q((5 * node) % 11 - 5, 16) for node in range(45))
    assert _dot(report.event_time_phase_covector, phase_direction) == 0
    assert _dot(report.event_time_form_covector, ZERO) == 0
    # This is a carrier crossing statement, not phase-independent timing on
    # a neighborhood: the exact carrier makes the phase energy gradient zero.
    mp = mpmath.mp.clone()
    mp.dps = 100
    for node, neighbors in enumerate(NEIGHBORS):
        current = mp.fsum(
            mp.sin(2 * mp.pi * ((neighbor % 9) - (node % 9)) / 9)
            for neighbor in neighbors
        )
        assert abs(current) < mp.mpf("1e-95")


def test_positive_energy_drop_uses_the_opposite_level_sign(report, factors):
    # ell=E_star-eta uses one fixed reference E_star, not each input's energy.
    drop_time_derivative = -report.level_time_derivative
    drop_response = tuple(-value for value in report.level_form_derivative)
    assert drop_time_derivative == Q(1024, 5) > 0
    assert drop_response[8] - drop_response[2] == 48 * factors["reception_mix"] / 25 > 0
    q = _gradient(MAIN)
    before_x_rate = tuple(-value / 4 for value in q)
    assert _dot(q, before_x_rate) * drop_time_derivative == -1


def test_equal_collective_summaries_do_not_fix_the_same_guard_time_jet(factors):
    reports = []
    observations = []
    for profile, form, third_moment in (
        (PROFILE_A, PAIR_A, Q(42)),
        (PROFILE_B, PAIR_B, Q(40)),
    ):
        cycle_q = _cycle_gradient(profile)
        assert (
            _dot(profile, cycle_q),
            _dot(cycle_q, cycle_q),
            _dot(cycle_q, _cycle_gradient(cycle_q)),
        ) == (Q(4), Q(12), third_moment)
        q = _gradient(form)
        assert q == tuple(PAIR_SCALE * value for value in cycle_q) * 5
        child_means = tuple(
            sum(form[start : start + 9], Q(0)) / 9 for start in range(0, 45, 9)
        )
        energy = sum(((form[a] - form[b]) ** 2 / 2 for a, b in EDGES), Q(0))
        loss = _dot(q, q) / 4
        assert child_means == (Q(0),) * 5
        assert energy == 10 * PAIR_SCALE**2 < Q(1, 100)
        assert loss == 15 * PAIR_SCALE**2 > 0
        # Target values and both initial rows also agree. Phase rate elsewhere
        # is nonzero; the shared exact initial carrier is not frozen in time.
        target_rows = tuple((form[i], q[i], -q[i] / 4, q[i] / 4) for i in (0, 1))
        assert target_rows == ((Q(0),) * 4,) * 2
        assert any(q)
        observations.append((child_means, energy, loss, target_rows))
        result = owner._assess_organized_source_carrier_guard_sensitivity(
            form, **factors
        )
        assert result.guard_energy_level == energy
        assert result.energy_loss_rate == loss
        # qdot follows the actual incoming form row. This derivative of the
        # complete loss is independent of the implementation's L^3 formula.
        qdot = _gradient(tuple(-value / 4 for value in q))
        assert result.energy_second_time_derivative == -_dot(q, qdot) / 2
        reports.append(result)

    first, second = reports
    assert observations[0] == observations[1]
    assert first.energy_second_time_derivative == Q(105, 4) * PAIR_SCALE**2
    assert second.energy_second_time_derivative == 25 * PAIR_SCALE**2
    # rho'=-E'' differs despite the equal observed rho. At the same fixed
    # level E_star-eta, the first nonzero inverse-series difference is positive.
    assert (
        first.drop_time_quadratic_coefficient - second.drop_time_quadratic_coefficient
        == 1 / (5400 * PAIR_SCALE**4)
        > 0
    )


@pytest.mark.parametrize("form", (MAIN, ASYMMETRIC, PAIR_A, PAIR_B, GENERAL))
def test_energy_acceleration_retains_both_rows_and_reverts_the_local_drop_series(
    factors, form
):
    result = owner._assess_organized_source_carrier_guard_sensitivity(form, **factors)
    q = _gradient(form)
    x_rate = tuple(-value / 4 for value in q)
    q_rate = _gradient(x_rate)
    loss = _dot(q, q) / 4
    energy_acceleration = -_dot(q, q_rate) / 2
    assert result.energy_second_time_derivative == energy_acceleration > 0

    linear = -result.level_time_derivative
    quadratic = result.drop_time_quadratic_coefficient
    # Substitution of tau=a*eta+b*eta^2 into
    # E(tau)-E_star+eta=-rho*tau+E''*tau^2/2+eta cancels both coefficients.
    assert 1 - loss * linear == 0
    assert -loss * quadratic + energy_acceleration * linear**2 / 2 == 0

    mp = mpmath.mp.clone()
    mp.dps = 100
    gamma = 1 / (1023 * mp.pi)
    phase_rate = tuple(gamma * _mp(mp, value) / 4 for value in q)
    assert any(phase_rate)
    phase_hessian_rate = [mp.mpf(0)] * 45
    for left, right in EDGES:
        cosine = mp.cos(2 * mp.pi * ((right % 9) - (left % 9)) / 9)
        term = cosine * (phase_rate[left] - phase_rate[right])
        phase_hessian_rate[left] += term
        phase_hessian_rate[right] -= term
    # DS=-H on the carrier. In the direct second derivative of joint energy,
    # phase curvature and the phase-driven form acceleration cancel exactly.
    phase_curvature = mp.fsum(
        rate * value for rate, value in zip(phase_rate, phase_hessian_rate)
    )
    assert phase_curvature > 0
    x_acceleration = tuple(
        (-_mp(mp, rate) - gamma * phase_term) / 4
        for rate, phase_term in zip(q_rate, phase_hessian_rate)
    )
    direct = (
        _mp(mp, _dot(x_rate, q_rate))
        + phase_curvature
        + mp.fsum(_mp(mp, value) * rate for value, rate in zip(q, x_acceleration))
    )
    assert abs(direct - _mp(mp, energy_acceleration)) < mp.mpf("1e-90")
    omitted_phase_work = direct - phase_curvature
    assert abs(omitted_phase_work - _mp(mp, energy_acceleration)) > mp.mpf("1e-25")


@pytest.mark.parametrize("scale", (Q(1, 2), Q(-2)))
def test_form_scaling_changes_loss_and_energy_curvature_but_inverts_time_jet(
    report, factors, scale
):
    result = owner._assess_organized_source_carrier_guard_sensitivity(
        tuple(scale * value for value in MAIN), **factors
    )
    assert result.energy_loss_rate == scale**2 * report.energy_loss_rate
    assert (
        result.energy_second_time_derivative
        == scale**2 * report.energy_second_time_derivative
    )
    assert (
        result.drop_time_quadratic_coefficient
        == report.drop_time_quadratic_coefficient / scale**4
    )


def test_common_form_shift_preserves_crossing_and_full_response_derivatives(
    report, factors
):
    shifted = owner._assess_organized_source_carrier_guard_sensitivity(
        tuple(value + Q(1, 100) for value in MAIN), **factors
    )
    for field in (
        "energy_loss_rate",
        "energy_second_time_derivative",
        "drop_time_quadratic_coefficient",
        "guard_energy_level",
        "event_time_form_covector",
        "event_time_phase_covector",
        "level_time_derivative",
        "level_form_derivative",
        "level_phase_base",
        "level_phase_matching",
        "level_phase_cycle",
        "level_phase_bounds",
    ):
        assert getattr(shifted, field) == getattr(report, field)
    assert shifted.timing.initial_form_mean == Q(1, 100)
    assert sum(report.event_time_form_covector, Q(0)) == 0


@pytest.mark.parametrize("level", (Q(0), Q(1, 100), -Q(1, 100)))
def test_constant_form_is_admitted_by_timing_but_not_transverse(factors, level):
    with pytest.raises(ValueError, match="transverse"):
        owner._assess_organized_source_carrier_guard_sensitivity(
            (level,) * 45, **factors
        )


def test_small_exact_nonzero_loss_is_not_rounded_to_an_equilibrium(factors):
    h = Q(1, 2**600)
    result = owner._assess_organized_source_carrier_guard_sensitivity(
        (h, -h) + ZERO[2:], **factors
    )
    assert result.energy_loss_rate == 20 * h**2 > 0
    assert result.level_time_derivative == -1 / (20 * h**2)
    unscaled_q = _gradient((Q(1), Q(-1)) + ZERO[2:])
    unscaled_moment = _dot(unscaled_q, _gradient(unscaled_q))
    assert unscaled_moment == 566
    assert result.energy_second_time_derivative == Q(283, 4) * h**2 > 0
    assert result.drop_time_quadratic_coefficient == Q(283, 64000) / h**4
    phase_contrast = result.level_phase_bounds[8] - result.level_phase_bounds[2]
    assert phase_contrast.lo > 0
    mp = mpmath.mp.clone()
    mp.dps = 100
    expected = 3 * _mp(mp, factors["reception_mix"]) / (1023 * mp.pi * 100 * _mp(mp, h))
    assert _mp(mp, phase_contrast.lo) <= expected <= _mp(mp, phase_contrast.hi)


@pytest.mark.parametrize(
    "form,overrides",
    (
        ((True,) + ZERO[1:], {}),
        ((float("nan"),) + ZERO[1:], {}),
        (ZERO[:44], {}),
        ((Q(1, 30),) * 45, {}),
        ((Q(1),) + ZERO[1:], {}),
        (MAIN, {"reception_mix": Q(1, 3)}),
        (MAIN, {"phase_coefficient": True}),
        (MAIN, {"silence_capacity_factor": 0}),
    ),
)
def test_guard_reader_preserves_primitive_factor_and_domain_rejections(
    factors, form, overrides
):
    with pytest.raises((TypeError, ValueError)):
        owner._assess_organized_source_carrier_guard_sensitivity(
            form, **{**factors, **overrides}
        )
