"""Static ideal-word algebra and conditional event-aligned error controls.

The off-carrier phasor control is a supplied static map, not an evaluated
state at either guard hit. No events, trajectories or hit searches are run.
"""

import math
from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.config.thresholds import DNFR_RECEPTION_MAX, EPI_SATURATION_MAX
from tnfr.dynamics import integrators, runtime
from tnfr.operators import network_stage, word_execution
from tnfr.operators._coherence_stage_kernel import DEFAULT_PHASE_LOCKING_COEFFICIENT
from tnfr.operators.definitions_base import Operator
from tnfr.operators.factor_contracts import canonical_glyph_factor_defaults
from tnfr.physics import _organized_source_child_word_error as owner
from tnfr.physics._organized_source_capture import _organized_source_offset_edge_angles
from tnfr.physics._organized_source_child_quotient import (
    _restrict_organized_source_child_quotient,
)

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
    tuple(b if a == i else a for a, b in EDGES if i in (a, b)) for i in range(45)
)
SWAP = tuple(9 * (0, 2, 1, 3, 4)[i // 9] + i % 9 for i in range(45))
ZERO = (Q(0),) * 45
UNIT = (Q(1),) * 45
MAIN = (Q(1, 64), -Q(1, 64)) + ZERO[2:]
EPSILON = Q(1, 2**20)
HORIZON = Q(1, 100)
LEVEL = Q(59, 40960)


def _exchange(values):
    return tuple(values[j] for j in SWAP)


def _project(values):
    return tuple((a + b) / 2 for a, b in zip(values, _exchange(values)))


def _perturb(values, epsilon):
    result = list(values)
    result[9] += epsilon
    result[18] -= epsilon
    return tuple(result)


def _bound(**overrides):
    return owner._bound_organized_source_child_word_error(
        **{
            "form": _perturb(MAIN, EPSILON),
            "phase_offsets": _perturb(ZERO, EPSILON),
            "capacities": UNIT,
            "horizon": HORIZON,
            "guard_level": LEVEL,
            **overrides,
        }
    )


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Child word bounds cannot execute events or trajectories")

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
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module")
def report():
    return _bound()


def _mp(context, value):
    if isinstance(value, (Q, int)):
        return context.mpf(value.numerator) / int(value.denominator)
    return value


def _contains(context, interval, value):
    assert _mp(context, interval.lo) <= value <= _mp(context, interval.hi)


def _ideal_word(context, form, offsets, beta, coefficient, kappa):
    """Independent simultaneous fine-target maps and continuous phase lifts."""
    forms = tuple(_mp(context, x) for x in form)
    phase = tuple(_mp(context, x) for x in offsets)
    result_form, result_phase = list(forms), list(phase)
    alpha = 2 * context.pi / 9
    for target in (0, 1):
        peers = NEIGHBORS[target]
        result_form[target] = (1 - _mp(context, beta)) * forms[target] + _mp(
            context, beta
        ) * context.fsum(forms[j] for j in peers) / len(peers)
        resultant = context.fsum(
            context.exp(
                context.j * (alpha * ((j % 9) - target) + phase[j] - phase[target])
            )
            for j in peers
        )
        result_phase[target] += _mp(context, coefficient) * context.arg(resultant)
    capacities = (kappa, kappa) + UNIT[2:]
    return tuple(result_form), tuple(result_phase), capacities


def _storage(context, form, offsets):
    alpha = 2 * context.pi / 9
    return context.fsum(
        (_mp(context, form[b]) - _mp(context, form[a])) ** 2 / 2
        + 1
        - context.cos(
            alpha * ((b % 9) - (a % 9))
            + _mp(context, offsets[b])
            - _mp(context, offsets[a])
        )
        for a, b in EDGES
    )


def test_prepared_offset_chart_retains_closing_edges_and_raw_offsets(report, mp):
    geometry = report.barrier.geometry
    offsets = tuple(Q((7 * i) % 13 - 6, 512) for i in range(45))
    angles = _organized_source_offset_edge_angles(geometry, offsets)
    by_edge = {
        (a, b): interval for (a, b, _), interval in zip(geometry.weighted_edges, angles)
    }
    assert set(by_edge) == set(EDGES)
    alpha = 2 * mp.pi / 9
    for a, b in EDGES:
        expected = _mp(mp, offsets[b] - offsets[a])
        if a // 9 == b // 9:
            expected += alpha
        _contains(mp, by_edge[a, b], expected)
    for child in range(5):
        a, b = 9 * child + 8, 9 * child
        raw_unadjusted = -8 * alpha + _mp(mp, offsets[b] - offsets[a])
        assert _mp(mp, by_edge[a, b].lo) > raw_unadjusted
    shifted = tuple(x + Q(5, 7) for x in offsets)
    shifted_angles = _organized_source_offset_edge_angles(geometry, shifted)
    for (a, b, _), interval in zip(geometry.weighted_edges, shifted_angles):
        expected = _mp(mp, offsets[b] - offsets[a])
        if a // 9 == b // 9:
            expected += alpha
        _contains(mp, interval, expected)


def test_fixed_guard_control_has_domain_and_strict_word_policy_margins(report):
    assert report.incoming.crossing_certified
    assert report.geometry_certified
    assert report.domain_certified and report.certified
    assert report.unavailable_reasons == ()
    assert report.fine_initial_mean == report.core_initial_mean == 0
    for name in (
        "boundary_ceiling_margin",
        "fine_acute_margin",
        "core_acute_margin",
        "fine_energy_margin",
        "core_energy_margin",
        "mean_margin",
        "form_rail_margin",
        "reception_form_margin",
        "reception_pressure_margin",
    ):
        assert getattr(report, name) > 0
    assert report.silence_capacity_margin >= 0
    assert report.resultant_real_lower_bound == 4
    assert report.budget.word_count == 1
    assert report.budget.initial_pair_capacity == 1
    assert report.reception_form_margin == (
        report.reception_form_ceiling - report.budget.form_absolute_upper_bound
    )
    assert report.reception_pressure_margin == (
        report.reception_pressure_ceiling
        - report.budget.reception_pressure_absolute_upper_bound
    )


def test_word_factors_and_policies_reuse_shared_owners(report):
    defaults = canonical_glyph_factor_defaults()
    assert report.reception_mix == Q(defaults["EN_mix"])
    assert report.pressure_retention == Q(defaults["IL_dnfr_factor"])
    assert report.silence_capacity_factor == Q(defaults["SHA_vf_factor"])
    assert report.phase_coefficient == Q(DEFAULT_PHASE_LOCKING_COEFFICIENT)
    assert report.reception_form_ceiling == Q(EPI_SATURATION_MAX)
    assert report.reception_pressure_ceiling == Q(DNFR_RECEPTION_MAX)
    assert report.silence_min_capacity == Q(0.01)
    assert (
        report.successor_capacities
        == (
            report.silence_capacity_factor,
            report.silence_capacity_factor,
        )
        + UNIT[2:]
    )


def test_event_alignment_keeps_timing_and_hidden_phase_terms(report):
    incoming = report.incoming
    x, g = incoming.form_radius_bound, incoming.gamma_upper
    degree = max(map(len, NEIGHBORS))
    velocity = max((2 * degree * x + degree * g) / 4, degree * g * x / 2)
    assert report.velocity_bound == velocity
    assert report.timestamp_error_bound == incoming.crossing_time_error_bound > 0
    aligned = (
        incoming.projected_state_error_bound + velocity * report.timestamp_error_bound
    )
    hidden = report.phase_coefficient * incoming.hidden_radius_bound**2 / 4
    assert report.aligned_input_error_bound == aligned
    assert aligned > incoming.projected_state_error_bound
    assert report.direct_hidden_phase_error_bound == hidden > 0
    assert report.post_form_error_bound == aligned
    assert report.post_phase_error_bound == (
        (1 + Q(5, 2) * report.phase_coefficient) * aligned + hidden
    )
    assert report.post_primary_error_bound == max(
        aligned, report.post_phase_error_bound
    )
    assert report.full_state_error_bound == (
        incoming.hidden_radius_bound + report.post_primary_error_bound
    )
    assert report.post_energy_error_bound == (
        108 * incoming.hidden_radius_bound**2
        + 360 * x * report.post_form_error_bound
        + 180 * report.post_phase_error_bound
    )


def test_static_noncarrier_word_has_nonzero_even_hidden_phase_defect(report, mp):
    phi, b = Q(1, 128), EPSILON
    core_phase = list(ZERO)
    core_phase[9] = core_phase[18] = phi
    core_phase = tuple(core_phase)
    fine_phase = _perturb(core_phase, b)
    factors = (
        report.reception_mix,
        report.phase_coefficient,
        report.silence_capacity_factor,
    )
    fine = _ideal_word(mp, ZERO, fine_phase, *factors)
    core = _ideal_word(mp, ZERO, core_phase, *factors)
    opposite = _ideal_word(mp, ZERO, _exchange(fine_phase), *factors)
    r = 1 + 2 * mp.cos(2 * mp.pi / 9)
    z_core = r + 2 * mp.exp(mp.j * _mp(mp, phi))
    z_fine = r + 2 * mp.exp(mp.j * _mp(mp, phi)) * mp.cos(_mp(mp, b))
    cross_imaginary = (z_fine * mp.conj(z_core)).imag
    cross_formula = 2 * r * (-2 * mp.sin(_mp(mp, b) / 2) ** 2) * mp.sin(_mp(mp, phi))
    assert abs(cross_imaginary - cross_formula) < mp.mpf("1e-95")
    assert cross_imaginary < 0
    phase_difference = fine[1][0] - core[1][0]
    assert phase_difference < 0
    expected = _mp(mp, report.phase_coefficient) * (mp.arg(z_fine) - mp.arg(z_core))
    assert abs(phase_difference - expected) < mp.mpf("1e-95")
    assert abs(phase_difference) <= _mp(mp, report.phase_coefficient * b**2 / 4)
    assert max(
        abs(a - c) for a, c in zip(_project(fine[1]), _project(opposite[1]))
    ) < mp.mpf("1e-95")
    assert fine[0] == core[0] == (mp.mpf(0),) * 45
    assert fine[2] == core[2] == report.successor_capacities
    assert abs(_storage(mp, fine[0], fine[1]) - _storage(mp, core[0], core[1])) <= _mp(
        mp, 108 * b**2 + 180 * report.phase_coefficient * b**2 / 4
    )


def test_static_phasor_second_order_coefficient_is_independent(mp):
    phi, b = mp.mpf(1) / 128, _mp(mp, EPSILON)
    r = 1 + 2 * mp.cos(2 * mp.pi / 9)
    z_core = r + 2 * mp.exp(mp.j * phi)
    coefficient = -r * mp.sin(phi) / abs(z_core) ** 2

    def angle(value):
        return mp.arg(r + 2 * mp.exp(mp.j * phi) * mp.cos(value))

    assert abs(mp.diff(angle, 0, 2) / 2 - coefficient) < mp.mpf("1e-95")
    quotient = (angle(b) - angle(0)) / b**2
    assert abs(quotient - coefficient) < abs(coefficient) * mp.mpf("1e-10")
    # On the exact carrier, the angle response vanishes despite a changed modulus.
    assert mp.arg(r + 2 * mp.cos(b)) == mp.arg(r + 2) == 0


def test_static_reception_commutes_with_projection_and_full_word_bound(report, mp):
    eta = Q(1, 100000)
    core_form = MAIN
    core_phase = list(ZERO)
    core_phase[9] = core_phase[18] = Q(1, 128)
    fine_form = _perturb(core_form, EPSILON)
    fine_phase = _perturb(tuple(core_phase), EPSILON)
    other_form, other_phase = list(core_form), list(core_phase)
    other_form[2] += eta
    other_phase[0] += eta
    factors = (
        report.reception_mix,
        report.phase_coefficient,
        report.silence_capacity_factor,
    )
    fine = _ideal_word(mp, fine_form, fine_phase, *factors)
    projected = _ideal_word(mp, _project(fine_form), _project(fine_phase), *factors)
    other = _ideal_word(mp, tuple(other_form), tuple(other_phase), *factors)
    assert max(abs(a - b) for a, b in zip(_project(fine[0]), projected[0])) < mp.mpf(
        "1e-95"
    )
    form_error = max(abs(a - b) for a, b in zip(_project(fine[0]), other[0]))
    phase_error = max(abs(a - b) for a, b in zip(_project(fine[1]), other[1]))
    assert form_error <= _mp(mp, eta)
    assert phase_error <= _mp(
        mp,
        (1 + Q(5, 2) * report.phase_coefficient) * eta
        + report.phase_coefficient * EPSILON**2 / 4,
    )
    assert fine[2] == other[2]


def test_exact_symmetry_has_zero_event_aligned_error_without_relaxing_restriction():
    result = _bound(form=MAIN, phase_offsets=ZERO)
    assert result.certified
    for name in (
        "timestamp_error_bound",
        "aligned_input_error_bound",
        "direct_hidden_phase_error_bound",
        "post_form_error_bound",
        "post_phase_error_bound",
        "post_primary_error_bound",
        "full_state_error_bound",
        "post_energy_error_bound",
    ):
        assert getattr(result, name) == 0
    with pytest.raises(ValueError, match="exactly equal"):
        _restrict_organized_source_child_quotient(
            form=_perturb(MAIN, Q(1, 2**600)), phase_offsets=ZERO, capacities=UNIT
        )


def test_tiny_hidden_data_retains_quadratic_projected_but_linear_full_error():
    epsilon = Q(1, 2**600)
    small = _bound(form=_perturb(MAIN, epsilon), phase_offsets=_perturb(ZERO, epsilon))
    doubled = _bound(
        form=_perturb(MAIN, 2 * epsilon), phase_offsets=_perturb(ZERO, 2 * epsilon)
    )
    assert small.certified and doubled.certified
    for name in (
        "timestamp_error_bound",
        "aligned_input_error_bound",
        "direct_hidden_phase_error_bound",
        "post_form_error_bound",
        "post_phase_error_bound",
        "post_primary_error_bound",
        "post_energy_error_bound",
    ):
        value = getattr(small, name)
        assert 0 < value < Q(1, 2**1000)
        assert getattr(doubled, name) == 4 * value
    assert doubled.full_state_error_bound == (
        2 * small.incoming.hidden_radius_bound + 4 * small.post_primary_error_bound
    )
    assert doubled.full_state_error_bound != 4 * small.full_state_error_bound


def test_low_energy_wrong_continuous_lift_cannot_borrow_word_domain():
    offsets = list(ZERO)
    offsets[9] = offsets[18] = Q(2 * math.pi)
    result = _bound(phase_offsets=tuple(offsets))
    assert result.incoming.crossing_certified
    assert result.incoming.fine_initial_excess_bounds.hi < Q(1, 100)
    assert result.incoming.core_initial_excess_bounds.hi < Q(1, 100)
    assert result.fine_acute_margin < 0 and result.core_acute_margin < 0
    assert not result.domain_certified and not result.certified
    assert result.post_phase_error_bound is None
    assert result.successor_capacities is None
    assert result.unavailable_reasons


def test_strict_initial_mean_boundary_is_not_silently_centered():
    form = tuple(value + Q(1, 30) for value in _perturb(MAIN, EPSILON))
    result = _bound(form=form)
    assert result.incoming.form == form
    assert result.fine_initial_mean == result.core_initial_mean == Q(1, 30)
    assert result.mean_margin == 0
    assert result.fine_energy_margin > 0 and result.core_energy_margin > 0
    assert not result.domain_certified and not result.certified
    assert result.post_form_error_bound is None


def test_acute_high_energy_source_is_unavailable():
    result = _bound(form=(Q(1, 8), -Q(1, 8)) + ZERO[2:], phase_offsets=ZERO)
    assert result.fine_acute_margin > 0 and result.core_acute_margin > 0
    assert result.fine_energy_margin < 0 and result.core_energy_margin < 0
    assert not result.domain_certified and not result.certified
    assert result.post_energy_error_bound is None


def test_admitted_word_domain_does_not_supply_missing_guard_crossings():
    result = _bound(guard_level=Q(0))
    assert result.domain_certified
    assert not result.incoming.crossing_certified
    assert not result.certified
    assert "incoming_crossing_unavailable" in result.unavailable_reasons
    assert result.timestamp_error_bound is None
    assert result.aligned_input_error_bound is None
    assert result.resultant_real_lower_bound is None


@pytest.mark.parametrize(
    "overrides",
    (
        {"form": (True,) + ZERO[1:]},
        {"phase_offsets": (float("nan"),) + ZERO[1:]},
        {"phase_offsets": ZERO[:-1]},
        {"capacities": (Q(1) + Q(1, 2**600),) + UNIT[1:]},
        {"horizon": Q(0)},
        {"guard_level": False},
    ),
)
def test_invalid_primitives_reject_before_an_ideal_word_claim(overrides):
    with pytest.raises((TypeError, ValueError)):
        _bound(**overrides)
