"""Four regional contact coefficients, retained source bounds and information loss.

The exact profile witness is a larger-class algebraic construction. No
original source is replayed and no time-dependent response is integrated.
"""

import os
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_phase_sensitive_interface as phase_owner
from tnfr.physics import _mixed_regional_loop as owner
from tnfr.physics import _sine_flow as flow_owner
from tnfr.research.artifact_io import encode_exact_tree, write_json_once

ERROR = Q(1, 10**10)
DELTA, ETA, WEIGHT_ERROR = Q(1, 1024), Q(2, 10**69), Q(2, 10**60)
MATURE, G, GL = Q(1, 2**28), Q(1, 3000), Q(1, 3300)
SECTION_ERROR = Q(3, 2) * (14 * ETA + 2 * 10**10 * WEIGHT_ERROR)
QLO, QHI = 3 * DELTA / 16 - SECTION_ERROR / 2, 3 * DELTA / 16 + SECTION_ERROR / 2
REGIONS = tuple(tuple(range(k, k + 9)) for k in (0, 9, 18, 27))
CYCLE = ((0, 1), (1, 3), (3, 2), (2, 0))
TOL = Q(1, 10**190)


def _mp(mp, value):
    if isinstance(value, (int, Q, np.integer)):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second, strict=True))


def _center(values):
    mean = sum(values) / len(values)
    return tuple(value - mean for value in values)


def _turns():
    return tuple(Q((1 + j // 9) * (j % 9 - 4), 9) for j in range(18))


def _support():
    bins = tuple(int((9 * turn) % 9) for turn in _turns())
    old = tuple(
        sorted(
            tuple(sorted((9 * p + j, 9 * p + (j + 1) % 9)))
            for p in (0, 1)
            for j in range(9)
        )
    )
    edges = {
        (a, b)
        for a in range(18)
        for b in range(a + 1, 18)
        if min((bins[b] - bins[a]) % 9, (bins[a] - bins[b]) % 9) in (1, 2)
        or (a // 9 != b // 9 and bins[a] == bins[b])
    }
    return old, tuple(sorted(edges - set(old)))


def _nominal_weights(mp):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + (-1 if j < 9 else 1) * _mp(mp, DELTA) / 2
        for j, turn in enumerate(_turns())
    )
    return tuple(
        1 - abs((theta[b] - theta[a] + mp.pi) % (2 * mp.pi) - mp.pi) / (2 * mp.pi)
        for a, b in _support()[1]
    )


def _weights(mp, side):
    values = tuple(
        Q(mp.nstr(value, 210))
        + (1 if value < _mp(mp, Q(7, 8)) else -1)
        * WEIGHT_ERROR
        * Q(1 + (j + side) % 3, 6)
        for j, value in enumerate(_nominal_weights(mp))
    )
    assert all(Q(3, 4) <= value <= 1 for value in values)
    return values


def _edges(weights):
    old, new = _support()
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, weight) for (a, b), weight in zip(new, weights, strict=True)
    )


def _h_apply(mp, values, weights):
    out = [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, weight in _edges(weights[side]):
            a, b = a + 18 * side, b + 18 * side
            current = _mp(mp, weight) * (values[a] - values[b])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _sine(mp, phases):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, phase)
        for turn, phase in zip(_turns() * 2, phases, strict=True)
    )
    out = [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, _ in _edges((Q(1),) * 63):
            a, b = a + 18 * side, b + 18 * side
            current = mp.sin(theta[b] - theta[a])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _exchange(values, slots=18):
    first = tuple(values[j] - values[18 + j] if j < slots else 0 for j in range(18))
    return first + tuple(-value for value in first)


def _field(mp, forms, phases, weights):
    x = tuple(_mp(mp, value) for value in forms)
    hx, sine, gx = _h_apply(mp, x, weights), _sine(mp, phases), _exchange(x)
    gamma = 1 / (1023 * mp.pi)
    cross = tuple(
        mp.sin(_mp(mp, phases[j + 18]) - _mp(mp, phases[j])) for j in range(18)
    )
    cross = cross + tuple(-value for value in cross)
    lx = tuple(a + 9 * b for a, b in zip(hx, gx, strict=True))
    splus = tuple(a + 9 * b for a, b in zip(sine, cross, strict=True))
    return tuple((-a + gamma * b) / 9 for a, b in zip(lx, splus, strict=True)), tuple(
        gamma * a / 9 for a in lx
    )


def _state(forms, phases):
    return tuple(I(value) for value in forms) + tuple(
        2 * pi_interval() * turn + I(value)
        for turn, value in zip(_turns() * 2, phases, strict=True)
    )


def _new_field(weights):
    return phase_owner._mixed_phase_sensitive_field(
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
    )


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_retained_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("regional-loop controls cannot replay or advance a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (_coupling_stage_kernel, "propose_coupling_target"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (network_stage, "_commit_coupling_structure"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(module, name, forbidden)
        # Permit the existing row builder for static full-state comparisons;
        # every integrator, acquisition producer and operator remains blocked.
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", static_factory)
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 220

    def forbidden(*args, **kwargs):
        pytest.fail("regional-loop proof cannot evaluate an exponential or root")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("exp", "expm", "eig", "polyroots", "odefun", "invertlaplace"):
            patch.setattr(context, name, forbidden)
        yield context


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _replace_at(value, path, **changes):
    if not path:
        return replace(value, **changes)
    return replace(
        value, **{path[0]: _replace_at(getattr(value, path[0]), path[1:], **changes)}
    )


def _regional_center(values):
    return tuple(
        value
        for region in REGIONS
        for value in _center(tuple(values[j] for j in region))
    )


def _profiles(offsets):
    return tuple(
        2 * pi_interval() * turn + I(offset)
        for turn, offset in zip(_turns() * 2, offsets, strict=True)
    )


def _phase_edges():
    old, new = _support()
    cross = tuple((a, b, Q(1)) for a, b in old + new if a < 9 <= b)
    return (
        cross,
        tuple((j, j + 18, Q(9)) for j in range(9, 18)),
        tuple((b + 18, a + 18, Q(1)) for a, b, _ in cross),
        tuple((j + 18, j, Q(9)) for j in range(9)),
    )


def _complex(mp, angle):
    return mp.mpc(mp.cos(angle), mp.sin(angle))


def _coefficient_oracle(mp, offsets):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, offset)
        for turn, offset in zip(_turns() * 2, offsets, strict=True)
    )
    return tuple(
        sum(
            (
                _mp(mp, weight) * _complex(mp, theta[b] - theta[a])
                for a, b, weight in row
            ),
            mp.mpc(0),
        )
        for row in _phase_edges()
    )


def _loop(values):
    result = 1
    for value in values:
        result *= value / abs(value)
    return result


def _complex_contains(mp, intervals, value):
    _contains(mp, intervals[0], value.real)
    _contains(mp, intervals[1], value.imag)


def _mean_phase_force(mp, coefficients, means):
    out = [mp.mpf(0)] * 4
    gamma = 1 / (1023 * mp.pi)
    for (a, b), coefficient in zip(CYCLE, coefficients, strict=True):
        flow = (
            gamma * (_complex(mp, _mp(mp, means[b] - means[a])) * coefficient).imag / 81
        )
        out[a] += flow
        out[b] -= flow
    return tuple(out)


def _witness_offsets(amplitude, swapped=False):
    bins = tuple(int((9 * turn) % 9) for turn in _turns())
    first = (amplitude, -amplitude) + (Q(0),) * 7
    second = (Q(0), amplitude, -amplitude) + (Q(0),) * 6
    if swapped:
        first, second = second, first
    return (
        tuple((first if j < 9 else second)[bins[j]] for j in range(18)) + (Q(0),) * 18
    )


def _assess(**changes):
    inputs = dict(form_error_bounds=(ERROR, ERROR), phase_error_bounds=(ERROR, ERROR))
    inputs.update(changes)
    return owner._assess_mixed_regional_loop(**inputs)


def _observation_payload(report):
    names = (
        "certified",
        "status",
        "unresolved_conditions",
        "source_association_certified",
        "complete_law_certified",
        "regional_geometry_certified",
        "source_comparison_certified",
        "nominal_factorization_certified",
        "source_nonzero_certified",
        "source_loop_bound_certified",
        "same_state_response_bound_certified",
        "witness_membership_certified",
        "matched_magnitude_obstruction_certified",
        "switching_work_certified",
        "source_comparison_upper_bound",
        "metric_root_upper_bound",
        "scaled_error_upper_bound",
        "comparison_phase_error_upper_bound",
        "hidden_profile_squared_upper_bound",
        "internal_coefficient_defect_upper_bound",
        "matching_coefficient_defect_upper_bound",
        "internal_coefficient_modulus_lower_bound",
        "matching_coefficient_modulus_lower_bound",
        "loop_deviation_upper_bound",
        "regional_mean_rate_error_upper_bound",
        "marked_gap_upper_bound",
        "registration_margin",
        "phase_switching_work_allowance",
        "aligned_storage_jump_bounds",
        "registered_storage_jump_bounds",
        "witness_amplitude",
        "witness_scaled_norm_squared",
        "witness_marked_perturbation_upper_bound",
        "source_bound_for_all_postattachment_times",
        "source_actual_loop_nontrivial_certified",
        "source_exact_factorization_certified",
        "witness_is_acquired_source",
        "witness_discriminates_whole_copy_transfer",
        "response_units",
        "loop_units",
        "new_observation_acquired",
        "source_trajectory_evaluated",
        "empirical_validation",
    )
    values = {name: getattr(report, name) for name in names}
    for name in (
        "gamma_bounds",
        "nominal_internal_sum",
        "nominal_internal_coefficient",
        "witness_imaginary_coefficient_bounds",
        "witness_matching_coefficient_bounds",
        "witness_positive_regional_mean_rate_bounds",
        "witness_regional_mean_rate_separation_bounds",
    ):
        interval = getattr(report, name)
        values[name] = {"lo": interval.lo, "hi": interval.hi}
    return {
        "schema": "tnfr.regional-loop-control-observations.v1",
        "assessment": values,
    }


@pytest.fixture(scope="module")
def report():
    result = _assess()
    destination = os.environ.get("TNFR_REGIONAL_LOOP_RECORD_PATH")
    if destination:
        write_json_once(destination, encode_exact_tree(_observation_payload(result)))
    return result


def test_nominal_geometry_uses_bare_phase_coefficients_and_one_rectangle(mp):
    observation = owner._mixed_regional_contact_coefficients(_profiles((Q(0),) * 36))
    omega = 2 * mp.pi / 9
    c0 = 1 + 2 * mp.cos(omega) + 2 * mp.cos(2 * omega)
    expected = (9 * c0, mp.mpf(81), 9 * c0, mp.mpf(81))
    assert c0 > 2 and tuple(len(row) for row in _phase_edges()) == (45, 9, 45, 9)
    for interval, value in zip(observation.coefficients, expected, strict=True):
        _complex_contains(mp, interval, mp.mpc(value))
    assert observation.nonzero_certified == (True,) * 4
    _complex_contains(mp, observation.loop_product, mp.mpc(1))
    _contains(mp, observation.loop_argument, mp.mpf(0))
    assert observation.region_nodes == REGIONS and observation.cycle == CYCLE
    # Independently reconstruct both complex row and column sums in phase-bin order.
    matrix = tuple(
        tuple(
            (
                _complex(mp, omega * (j - i))
                if (j - i) % 9 in (0, 1, 2, 7, 8)
                else mp.mpc(0)
            )
            for j in range(9)
        )
        for i in range(9)
    )
    for j in range(9):
        assert abs(sum(matrix[j]) - c0) < _mp(mp, TOL)
        assert abs(sum(matrix[i][j] for i in range(9)) - c0) < _mp(mp, TOL)
    assert not observation.supplied_profiles_authenticated_as_source


def test_four_regional_rates_match_every_actual_row_with_unequal_form_weights(mp):
    hidden = _regional_center(tuple(Q((7 * j) % 11 - 5, 10**6) for j in range(36)))
    means = (Q(1, 11), Q(-1, 13), Q(1, 17), Q(-1, 19))
    offsets = tuple(hidden[j] + means[j // 9] for j in range(36))
    forms = _center(tuple(Q((5 * j) % 13 - 6, 10**5) for j in range(36)))
    weights = (_weights(mp, 0), _weights(mp, 1))
    expected_x, expected_y = _field(mp, forms, offsets, weights)
    actual = _new_field(weights)(_state(forms, offsets))
    for interval, value in zip(actual, expected_x + expected_y, strict=True):
        _contains(mp, interval, value)
    coefficients = _coefficient_oracle(mp, hidden)
    observed = owner._mixed_regional_contact_coefficients(_profiles(hidden))
    for interval, value in zip(observed.coefficients, coefficients, strict=True):
        _complex_contains(mp, interval, value)
    sine_rates = _mean_phase_force(mp, coefficients, means)
    hx = _h_apply(mp, tuple(_mp(mp, x) for x in forms), weights)
    gx = _exchange(forms)
    linear = tuple(
        -sum(hx[j] + 9 * _mp(mp, gx[j]) for j in region) / 81 for region in REGIONS
    )
    for region, linear_rate, sine_rate in zip(REGIONS, linear, sine_rates, strict=True):
        assert abs(
            sum(expected_x[j] for j in region) / 9 - linear_rate - sine_rate
        ) < _mp(mp, TOL)
    # Held form conductances belong to the first term, not to the phase coefficient.
    theta = tuple(
        2 * mp.pi * _mp(mp, t) + _mp(mp, h)
        for t, h in zip(_turns(), hidden[:18], strict=True)
    )
    wrong = sum(
        (
            _mp(mp, w) * _complex(mp, theta[b] - theta[a])
            for a, b, w in _edges(weights[0])
            if a < 9 <= b
        ),
        mp.mpc(0),
    )
    assert abs(wrong - coefficients[0]) > _mp(mp, Q(1, 10))


def test_regional_frame_changes_preserve_state_force_and_loop(mp):
    hidden = _witness_offsets(Q(1, 100))
    means = (Q(1, 7), Q(-1, 5), Q(1, 11), Q(-1, 13))
    shifts = (Q(1, 3), Q(-1, 4), Q(1, 8), Q(-1, 9))
    changed_means = tuple(a + b for a, b in zip(means, shifts, strict=True))
    changed_hidden = tuple(hidden[j] - shifts[j // 9] for j in range(36))
    assert tuple(hidden[j] + means[j // 9] for j in range(36)) == tuple(
        changed_hidden[j] + changed_means[j // 9] for j in range(36)
    )
    original = _coefficient_oracle(mp, hidden)
    changed = _coefficient_oracle(mp, changed_hidden)
    observed = owner._mixed_regional_contact_coefficients(_profiles(changed_hidden))
    for (a, b), before, after, intervals in zip(
        CYCLE, original, changed, observed.coefficients, strict=True
    ):
        assert abs(after - _complex(mp, _mp(mp, shifts[a] - shifts[b])) * before) < _mp(
            mp, TOL
        )
        _complex_contains(mp, intervals, after)
    assert abs(_loop(original) - _loop(changed)) < _mp(mp, TOL)
    first = _mean_phase_force(mp, original, means)
    second = _mean_phase_force(mp, changed, changed_means)
    assert max(abs(a - b) for a, b in zip(first, second, strict=True)) < _mp(mp, TOL)
    physical_intervention = _mean_phase_force(mp, original, changed_means)
    assert max(
        abs(a - b) for a, b in zip(first, physical_intervention, strict=True)
    ) > _mp(mp, Q(1, 10**6))


def test_cycle_orientation_and_factorization_keep_each_link_magnitude(mp):
    values = _coefficient_oracle(mp, _witness_offsets(Q(1, 100)))
    forward = _loop(values)
    reverse = _loop(tuple(value.conjugate() for value in reversed(values)))
    assert abs(reverse - forward.conjugate()) < _mp(mp, TOL)
    assert forward.imag > _mp(mp, Q(1, 10**8))
    gauges = {0: mp.mpc(1)}
    for (a, b), value in zip(CYCLE[:3], values[:3], strict=True):
        gauges[b] = gauges[a] * (value / abs(value)).conjugate()
    transformed = tuple(
        gauges[a].conjugate() * value / abs(value) * gauges[b]
        for (a, b), value in zip(CYCLE, values, strict=True)
    )
    assert max(abs(value - 1) for value in transformed[:3]) < _mp(mp, TOL)
    assert abs(transformed[3] - forward) < _mp(mp, TOL)
    scaled = tuple((j + 2) * value for j, value in enumerate(values))
    assert abs(_loop(scaled) - forward) < _mp(mp, TOL)
    means = (Q(1, 10), Q(0), Q(0), Q(0))
    assert _mean_phase_force(mp, scaled, means) != _mean_phase_force(mp, values, means)


def test_zero_or_unresolved_coefficient_does_not_become_zero_curvature(monkeypatch):
    profiles = [I(0)] * 36
    # Three copies of the three roots of unity cancel this matching sum.
    for j in range(18, 27):
        profiles[j] = 2 * pi_interval() * Q((j - 18) % 3 - 1, 3)
    zero = owner._mixed_regional_contact_coefficients(profiles)
    assert not zero.nonzero_certified[3]
    assert zero.unit_coefficients is zero.loop_product is zero.loop_argument is None
    assert not zero.argument_available
    assert not zero.zero_coefficient_inferred_from_unavailable_chart
    assert len(zero.coefficients) == len(zero.magnitudes) == 4
    broad = owner._mixed_regional_contact_coefficients((I(-4, 4),) * 36)
    assert not any(broad.nonzero_certified)
    assert broad.loop_product is None and broad.raw_loop_product is not None

    def unavailable(*args):
        raise ValueError("the principal argument rectangle meets its branch cut")

    monkeypatch.setattr(owner, "arg", unavailable)
    angle_only = owner._mixed_regional_contact_coefficients(_profiles((Q(0),) * 36))
    assert all(angle_only.nonzero_certified) and angle_only.loop_product is not None
    assert angle_only.loop_argument is None and not angle_only.argument_available


def test_linear_defects_cancel_and_quadratic_bounds_use_centered_hidden_profiles(mp):
    hidden = _regional_center(tuple(Q((11 * j) % 17 - 8, 10**5) for j in range(36)))
    baseline = _coefficient_oracle(mp, (Q(0),) * 36)
    perturbed = _coefficient_oracle(mp, hidden)
    theta = tuple(2 * mp.pi * _mp(mp, turn) for turn in _turns() * 2)
    for (a, b), edges, before, after, factor in zip(
        CYCLE, _phase_edges(), baseline, perturbed, (5, 9, 5, 9), strict=True
    ):
        linear = sum(
            (
                _mp(mp, w * (hidden[j] - hidden[i])) * _complex(mp, theta[j] - theta[i])
                for i, j, w in edges
            ),
            mp.mpc(0),
        )
        assert abs(linear) < _mp(mp, TOL)
        square = sum(w * (hidden[j] - hidden[i]) ** 2 for i, j, w in edges)
        local = sum(hidden[j] ** 2 for j in REGIONS[a] + REGIONS[b])
        assert square <= 2 * factor * local
        assert abs(after - before) <= _mp(mp, square / 2)
    total = _dot(hidden, hidden)
    assert abs(_loop(perturbed) - 1) <= _mp(mp, 7 * total / 9)
    error = tuple(hidden[j] + Q(j // 9 - 1, 10000) for j in range(36))
    assert _regional_center(error) == hidden
    assert _dot(hidden, hidden) <= _dot(error, error)


def test_matched_magnitude_profiles_have_opposite_regional_rates_not_copy_transfer(
    mp, report
):
    amplitude = DELTA / 10000
    first, second = _witness_offsets(amplitude), _witness_offsets(amplitude, True)
    assert tuple(sum(first[j] for j in region) for region in REGIONS) == (Q(0),) * 4
    assert 4 * _dot(first, first) == 4 * _dot(second, second) == 16 * amplitude**2
    values = _coefficient_oracle(mp, first)
    swapped = _coefficient_oracle(mp, second)
    a, omega = _mp(mp, amplitude), 2 * mp.pi / 9
    imaginary = 8 * mp.sin(a / 2) ** 2 * mp.sin(omega) * (1 - mp.cos(omega - a))
    assert imaginary > 0 and abs(values[0].imag - imaginary) < _mp(mp, TOL)
    assert abs(swapped[0] - values[0].conjugate()) < _mp(mp, TOL)
    matching = 9 * (7 + 2 * mp.cos(a))
    for j in (1, 3):
        assert abs(values[j] - matching) < _mp(mp, TOL)
    for before, after in zip(values, swapped, strict=True):
        assert abs(abs(before) - abs(after)) < _mp(mp, TOL)
    assert _loop(values).imag > 0 and _loop(swapped).imag < 0
    weights = (_weights(mp, 0), _weights(mp, 1))
    rate = imaginary / (81 * 1023 * mp.pi)
    for offsets, expected_sign in ((first, 1), (second, -1)):
        observation = owner._mixed_regional_contact_coefficients(_profiles(offsets))
        exact = _coefficient_oracle(mp, offsets)
        for interval, value in zip(observation.coefficients, exact, strict=True):
            _complex_contains(mp, interval, value)
        dx, _ = _field(mp, (Q(0),) * 36, offsets, weights)
        assert abs(sum(dx[:9]) / 9 - expected_sign * rate) < _mp(mp, TOL)
        assert abs(sum(dx[:18])) < _mp(mp, TOL)
        for left, right in _support()[0]:
            assert abs(offsets[right] - offsets[left]) <= 2 * amplitude
        bins = tuple(int((9 * turn) % 9) for turn in _turns())
        for left in range(9):
            right = next(j for j in range(9, 18) if bins[j] == bins[left])
            assert abs(offsets[right] - offsets[left]) <= 2 * amplitude
    _contains(mp, report.witness_imaginary_coefficient_bounds, imaginary)
    _contains(mp, report.witness_positive_regional_mean_rate_bounds, rate)
    _contains(mp, report.witness_regional_mean_rate_separation_bounds, 2 * rate)
    assert (
        report.witness_membership_certified
        and report.matched_magnitude_obstruction_certified
    )
    assert report.witness_whole_copy_transfer_rate == 0
    assert not report.witness_is_acquired_source


def test_same_state_magnitude_comparator_preserves_weighted_term_and_prices_rate_only(
    mp,
):
    hidden = _regional_center(tuple(Q((13 * j) % 19 - 9, 10**5) for j in range(36)))
    means = (Q(1, 7), Q(1, 5), Q(-1, 11), Q(-1, 13))
    coefficients = _coefficient_oracle(mp, hidden)
    exact = _mean_phase_force(mp, coefficients, means)
    blind = _mean_phase_force(
        mp, tuple(mp.mpc(abs(value)) for value in coefficients), means
    )
    bound = _mp(mp, 28 * G * _dot(hidden, hidden) / 81)
    difference = tuple(a - b for a, b in zip(exact, blind, strict=True))
    assert 0 < max(abs(value) for value in difference) <= bound
    weighted_form_term = tuple(_mp(mp, Q(j - 2, 97)) for j in range(4))
    assert max(
        abs((a + w) - (b + w) - d)
        for a, b, w, d in zip(exact, blind, weighted_form_term, difference, strict=True)
    ) < _mp(mp, TOL)
    # This is a same-state instantaneous comparison, with no error integration.


def test_original_source_error_gives_coefficient_domains_without_new_readings(report):
    # Rebuild from original registered source errors, not the preceding verdict.
    eb = 3 * (24 * MATURE + 84 * WEIGHT_ERROR * 10 * QHI / GL) / (1 - 135 * QHI)
    pstar = 2 * (1 + G) * eb + 14 * WEIGHT_ERROR * QHI / (3 * GL)
    difference = pstar / (1 - 2 * G)
    hidden_square = difference**2 / 4
    assert report.source_comparison_upper_bound == eb
    assert report.metric_root_upper_bound == pstar
    assert report.scaled_error_upper_bound == difference
    assert report.comparison_phase_error_upper_bound == difference / 2
    assert report.hidden_profile_squared_upper_bound == hidden_square
    assert report.internal_coefficient_defect_upper_bound == 5 * hidden_square
    assert report.matching_coefficient_defect_upper_bound == 9 * hidden_square
    assert report.internal_coefficient_modulus_lower_bound == 18 - 5 * hidden_square > 0
    assert report.matching_coefficient_modulus_lower_bound == 81 - 9 * hidden_square > 0
    assert report.loop_deviation_upper_bound == 7 * hidden_square / 9
    assert report.regional_mean_rate_error_upper_bound == 28 * G * hidden_square / 81
    assert report.marked_gap_upper_bound == QHI + 3 * difference / 4
    assert report.registration_margin == DELTA / 4 - report.marked_gap_upper_bound > 0
    # Marked identity implies a sufficient nonzero domain, but is not an equality.
    assert difference < DELTA / 3
    assert report.nominal_internal_sum.lo > 2
    assert report.nominal_matching_coefficient == 81
    assert report.certified and not report.unresolved_conditions
    for name in (
        "source_association_certified",
        "complete_law_certified",
        "regional_geometry_certified",
        "source_comparison_certified",
        "nominal_factorization_certified",
        "source_nonzero_certified",
        "source_loop_bound_certified",
        "same_state_response_bound_certified",
        "source_bound_for_all_postattachment_times",
    ):
        assert getattr(report, name)
    assert (
        report.source.reference_relative_phase
        == "same actual continuous psi(t), analytically"
    )
    assert not report.same_actual_phase_reference_is_forecast
    entry = 4 * QHI + 200 * MATURE
    allowance = 972 * ETA + Q(9, 2) * entry**3
    assert report.phase_switching_work_allowance == allowance
    assert report.aligned_storage_jump_bounds == (Q(0), 9 * (entry + 18 * ETA) ** 2)
    assert report.registered_storage_jump_bounds == (
        162 - allowance,
        162 + allowance + 9 * entry**2,
    )
    assert report.switching_work_certified
    assert (
        report.original_attachment_work_retained_once
        and report.original_source_error_paid_once
    )


def test_consumed_source_primitives_and_actual_cross_edges_are_readmitted(
    monkeypatch, report
):
    source = report.source
    cases = (
        ((), "source_association_certified", False),
        ((), "prior_domain_certified", False),
        ((), "source_cone_certified", False),
        ((), "reference_cone_certified", False),
        ((), "reference_integrals_certified", False),
        ((), "incremental_stiffness_certified", False),
        ((), "common_metric_certified", False),
        ((), "forced_comparison_certified", False),
        ((), "cycle_windings_certified", False),
        ((), "source_error_paid_once", False),
        ((), "initial_metric_root_upper_bound", Q(0)),
        ((), "phase_error_upper_bound", Q(0)),
        ((), "forcing_integral_upper_bound", Q(0)),
        ((), "reference_form_integral_upper_bound", Q(0)),
        ((), "mean_error_forcing_discarded", True),
        ((), "reference_is_autonomous_predictor", True),
        ((), "phase_readout_input_required", True),
        ((), "clock", "untransported clock"),
        ((), "initial_registrations_turns", (Q(0), Q(0))),
        ((), "phase_switching_work_allowance", Q(0)),
        (("source",), "original_attachment_state_retained", False),
        (("source",) * 3, "original_complete_field_unchanged", False),
        (("source",) * 4, "complete_precontact_history_translated", False),
        (("source",) * 5, "original_phase_lifts_retained", False),
        (("source",) * 5, "phase_mean_error_upper_bound", Q(0)),
        (("source",) * 7, "early_age", Q(3375001)),
        (("source",) * 9, "uniform_comparison_upper_bound", Q(0)),
        (("source",) * 11, "exact_zero_entry_total_from_prior_laws", False),
    )
    for path, name, value in cases:
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_marked_identity", lambda **kwargs: changed
        )
        assert not _assess().certified, (path, name)
    path = ("source",) * 11 + ("handoff", "prefix", "reference")
    reference = source
    for name in path:
        reference = getattr(reference, name)
    for name, value in (
        ("phase_reset_error_bound", 4 * ETA),
        ("new_weight_range", (Q(1, 2), Q(1))),
        ("form_error_bound", ERROR / 2),
    ):
        original = replace(reference.sources[0], **{name: value})
        changed = _replace_at(
            source,
            path,
            sources=(original, reference.sources[1]),
            donor=replace(reference.donor, source=original),
        )
        monkeypatch.setattr(
            owner, "_assess_mixed_marked_identity", lambda **kwargs: changed
        )
        assert not _assess().source_association_certified
    for name, value in (
        ("original_supplied_phase_origins", (Q(0), DELTA)),
        ("actual_weight_deviation_upper_bound", Q(0)),
        ("inherited_donor_velocity_retained", False),
    ):
        changed = _replace_at(source, path, **{name: value})
        monkeypatch.setattr(
            owner, "_assess_mixed_marked_identity", lambda **kwargs: changed
        )
        assert not _assess().certified
    monkeypatch.setattr(owner, "_assess_mixed_marked_identity", lambda **kwargs: source)
    registered = owner._derive_registered_phase_probe_geometry()
    geometry = registered.successor
    actual_cross = next(edge for edge in geometry.edge_pairs if edge[0] < 9 <= edge[1])
    absent_cross = next(
        (a, b)
        for a in range(9)
        for b in range(9, 18)
        if (a, b) not in geometry.edge_pairs
    )
    changed_edges = tuple(
        sorted((set(geometry.edge_pairs) - {actual_cross}) | {absent_cross})
    )
    # A cached incidence certificate must not mask a different actual edge set.
    assert len(changed_edges) == len(geometry.edge_pairs)
    for changes in (
        dict(edge_pairs=changed_edges),
        dict(bare_gap_lower_bound=Q(3)),
        dict(weighted_gap_lower_bound=Q(2)),
        dict(spectral_gap_certified=False),
    ):
        changed_geometry = replace(registered, successor=replace(geometry, **changes))
        monkeypatch.setattr(
            owner, "_derive_registered_phase_probe_geometry", lambda: changed_geometry
        )
        with pytest.raises(ValueError):
            owner._mixed_regional_contact_coefficients((Q(0),) * 36)
        rejected = _assess()
        assert (
            not rejected.regional_geometry_certified
            and not rejected.source_loop_bound_certified
        )
    monkeypatch.setattr(
        owner, "_derive_registered_phase_probe_geometry", lambda: registered
    )
    assert _assess().certified


def test_raw_fixed_width_admission_precedes_geometry_trigonometry_and_source_work(
    monkeypatch, report
):
    def forbidden(*args, **kwargs):
        pytest.fail("raw invalid input reached retained source or coefficient work")

    invalid = (
        True,
        np.bool_(False),
        None,
        "0",
        float("nan"),
        float("inf"),
        complex(1, 2),
    )
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_derive_registered_phase_probe_geometry", forbidden)
        patch.setattr(owner, "sin", forbidden)
        patch.setattr(owner, "cos", forbidden)
        for bad in invalid:
            for position in (0, 35):
                values = tuple(bad if j == position else Q(0) for j in range(36))
                with pytest.raises((TypeError, ValueError, OverflowError)):
                    owner._mixed_regional_contact_coefficients(values)
        for bad in ((0,) * 35, (0,) * 37, repeat(0), {0, 1}, {0: 0}, "0" * 36):
            with pytest.raises((TypeError, ValueError)):
                owner._mixed_regional_contact_coefficients(bad)
    with monkeypatch.context() as patch:
        patch.setattr(owner, "_assess_mixed_marked_identity", forbidden)
        for name in ("form_error_bounds", "phase_error_bounds"):
            for bad in invalid + (-Q(1, 10**100),):
                for side in (0, 1):
                    with pytest.raises((TypeError, ValueError, OverflowError)):
                        _assess(
                            **{
                                name: tuple(
                                    bad if j == side else Q(0) for j in range(2)
                                )
                            }
                        )
            for bad in ((0,), (0, 0, 0), {0, 1}, {0: 0, 1: 0}, repeat(0)):
                with pytest.raises((TypeError, ValueError)):
                    _assess(**{name: bad})
    consumed = []

    def cached(**kwargs):
        consumed.append(kwargs)
        return report.source

    monkeypatch.setattr(owner, "_assess_mixed_marked_identity", cached)
    repeated = _assess(
        form_error_bounds=np.array([ERROR, ERROR], dtype=object),
        phase_error_bounds=[ERROR, ERROR],
    )
    assert repeated == report and len(consumed) == 1
    assert consumed[0] == dict(
        form_error_bounds=(ERROR, ERROR),
        phase_error_bounds=(ERROR, ERROR),
        registered_observation_error_bound=Q(0),
        aligned_observation_error_bound=Q(0),
    )
    for name in ("form_error_bounds", "phase_error_bounds"):
        assert not _assess(**{name: (ERROR + Q(1, 10**100), ERROR)}).certified


def test_unused_verdicts_do_not_supply_source_bounds_or_broaden_claims(
    monkeypatch, report, record_property
):
    source = replace(
        report.source,
        certified=False,
        strict_marked_identity_certified=False,
        all_postattachment_times_covered=False,
        uniform_collective_prediction_certified=True,
        status="unused summary",
        unresolved_conditions=("unused",),
        source=replace(
            report.source.source,
            certified=False,
            transverse_decay_certified=False,
            integrable_moment_certified=False,
        ),
    )
    monkeypatch.setattr(owner, "_assess_mixed_marked_identity", lambda **kwargs: source)
    repeated = _assess()
    assert repeated.certified and repeated.source_bound_for_all_postattachment_times
    assert repeated.loop_deviation_upper_bound == report.loop_deviation_upper_bound
    assert (
        repeated.regional_mean_rate_error_upper_bound
        == report.regional_mean_rate_error_upper_bound
    )
    assert (
        repeated.registered_storage_jump_bounds == report.registered_storage_jump_bounds
    )
    assert report.region_nodes == REGIONS and report.cycle == CYCLE
    assert (
        report.full_state_dimension == 72 and report.physical_capacities == (Q(1),) * 36
    )
    assert report.full_contact_pairs == tuple((j, j + 18) for j in range(18))
    assert report.activation_time == 2010000 and report.switch_age == 3375000
    assert report.switch_time == 5385000 and report.internal_divisor == 9
    assert report.response_units == "regional mean form per structural tau"
    assert report.loop_units == "dimensionless"
    assert (
        report.response_comparator == "same state and magnitudes; zero contact phases"
    )
    assert (
        report.clock
        == "tau=(1023/1024)*t; unchanged complete regional interaction rows"
    )
    for name in (
        "source_actual_loop_nontrivial_certified",
        "source_exact_factorization_certified",
        "regional_frame_change_is_physical_intervention",
        "coefficient_magnitudes_discarded",
        "held_form_weights_used_as_phase_coefficients",
        "same_actual_phase_reference_is_forecast",
        "comparison_is_autonomous_predictor",
        "rate_bound_integrated_as_trajectory_error",
        "original_form_tolerance_reassigned",
        "witness_is_acquired_source",
        "witness_discriminates_whole_copy_transfer",
        "witness_refutes_existing_aggregate_prediction",
        "new_field_installed",
        "new_observation_acquired",
        "source_trajectory_evaluated",
        "original_source_replayed",
        "state_reset_performed",
        "independent_link_field_identified",
        "electromagnetic_sector_identified",
        "empirical_validation",
    ):
        assert getattr(report, name) is False
    assert not hasattr(report, "post_duration") and not hasattr(
        report, "actual_source_state"
    )
    with pytest.raises(FrozenInstanceError):
        report.hidden_profile_squared_upper_bound = Q(0)
    for name in (
        "source",
        "state",
        "weights",
        "actual_phase",
        "initial_reading",
        "horizon",
        "noise",
        "threshold",
        "registration",
        "clock",
    ):
        with pytest.raises(TypeError):
            _assess(**{name: Q(0)})
    payload = _observation_payload(report)
    for name, value in payload["assessment"].items():
        record_property(name, str(value))
