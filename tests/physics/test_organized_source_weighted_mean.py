"""Arbitrary conserved means without event, source or trajectory execution."""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_source_capture as capture
from tnfr.physics import _organized_source_port as port

CAPACITIES = (Q(1, 2), Q(1, 2)) + (Q(1),) * 43
TURNS = (0,) * 45
FORM = tuple(
    Q(3, 10)
    + (Q(1, 128) if n == 0 else -Q(1, 64) if n == 9 else Q(0))
    + (Q(1, 256) if n == 12 else Q(0))
    for n in range(45)
)


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Mean-leaf admission cannot execute events, flow or sources")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (word_execution, "execute_network_operator_stage"),
            (capture, "_assess_organized_source_capture"),
            (capture, "_assess_organized_source_attachment"),
            (port, "_assess_organized_source_attachment"),
            (port, "_assess_organized_source_emission"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


@pytest.fixture(scope="module")
def phases(mp):
    return tuple(
        Q(
            float(
                2 * mp.pi / 9 * (n % 9 - 4)
                + mp.mpf(1) / 40
                + (mp.mpf(1) / 100 if n == 0 else 0)
            )
        )
        for n in range(45)
    )


def _assess(form, phases, capacities=CAPACITIES, turns=TURNS):
    return capture._assess_organized_source_weighted_mean_retention(
        form=form, phases=phases, phase_turns=turns, capacities=capacities
    )


@pytest.fixture(scope="module")
def report(phases):
    return _assess(FORM, phases)


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def test_arbitrary_mean_is_retained_without_centering_raw_state(report, phases):
    mean = Q(3, 10) + Q(1, 47 * 256)
    assert report.weighted_form_total == Q(141, 10) + Q(1, 256)
    assert report.weighted_form_mean == mean != sum(FORM) / 45
    assert report.observation.unit_capacity_reference.form == FORM
    assert report.observation.unit_capacity_reference.phases == phases
    assert report.observation.inverse_capacity_weighted_form_total == (
        report.weighted_form_total
    )
    assert report.asymptotic_form == (mean,) * 45
    assert report.admitted and report.converges_to_target
    assert report.geometric_retention_certified and report.form_rails_certified
    assert report.retained_windings == (1,) * 5
    assert not report.unresolved_conditions
    with pytest.raises(ValueError, match="zero sum"):
        capture._assess_organized_source_raw_capacity_retention(
            form=FORM, phases=phases, capacities=CAPACITIES, phase_turns=TURNS
        )


def test_weighted_centered_spectral_bound_and_absolute_rails(report):
    weights = tuple(1 / nu for nu in CAPACITIES)
    mean = sum(w * x for w, x in zip(weights, FORM)) / sum(weights)
    centered = tuple(x - mean for x in FORM)
    ordinary_centered = tuple(x - sum(FORM) / 45 for x in FORM)
    norm = sum(z**2 for z in centered)
    weighted_norm = sum(w * z**2 for w, z in zip(weights, centered))
    comparison = sum(w * y**2 for w, y in zip(weights, ordinary_centered))
    ordinary_norm = sum(y**2 for y in ordinary_centered)
    energy = sum(report.observation.unit_capacity_reference.form_edge_storage)
    assert sum(w * z for w, z in zip(weights, centered)) == 0
    assert min(weights) * norm <= weighted_norm <= comparison
    assert comparison <= max(weights) * ordinary_norm
    assert Q(4, 15) * ordinary_norm <= 2 * energy
    assert norm <= 15 * energy
    assert energy < report.excess_bounds.lo
    bound = report.centered_form_squared_upper_bound
    assert bound == 15 * report.excess_bounds.hi < Q(3, 20)
    radius = report.centered_form_absolute_upper_bound
    assert radius**2 >= bound
    assert report.form_absolute_upper_bound == abs(mean) + radius < 1
    assert max(abs(x) for x in FORM) < report.form_absolute_upper_bound


def test_independent_incidence_rows_conserve_both_weighted_totals(report, phases, mp):
    observation = report.observation
    reference = observation.unit_capacity_reference
    gradient, current = [Q(0)] * 45, [mp.mpf(0)] * 45
    storage = mp.mpf(0)
    for a, b, weight in reference.geometry.weighted_edges:
        difference = FORM[a] - FORM[b]
        gradient[a] += weight * difference
        gradient[b] -= weight * difference
        angle = _mp(mp, phases[b] - phases[a])
        sine = _mp(mp, weight) * mp.sin(angle)
        current[a] += sine
        current[b] -= sine
        storage += _mp(mp, weight * difference**2 / 2)
        storage += _mp(mp, weight) * (1 - mp.cos(angle))
    gamma = 1 / (1023 * mp.pi)
    pressure = tuple((-_mp(mp, q) + gamma * s) / 4 for q, s in zip(gradient, current))
    dx = tuple(_mp(mp, nu) * p for nu, p in zip(CAPACITIES, pressure))
    dtheta = tuple(gamma * _mp(mp, nu * q) / 4 for nu, q in zip(CAPACITIES, gradient))
    loss = sum(nu * q**2 / 4 for nu, q in zip(CAPACITIES, gradient))
    assert reference.form_gradient == tuple(gradient)
    assert observation.dissipation_tau == loss
    _contains(mp, reference.joint_storage_bounds, storage)
    for bounds, value in zip(observation.pressure_tau_bounds, pressure):
        _contains(mp, bounds, value)
    for clock, factor in (
        ("tau", mp.mpf(1)),
        ("original", mp.mpf(1023) / 1024),
        ("slow", 2 / gamma**2),
    ):
        for channel, values in (("form", dx), ("phase", dtheta)):
            bounds = getattr(observation, f"{channel}_{clock}_rate_bounds")
            for interval, value in zip(bounds, values):
                _contains(mp, interval, factor * value)
            assert sum(
                (interval / nu for interval, nu in zip(bounds, CAPACITIES)), I(0)
            ).contains(0)
        _contains(
            mp,
            getattr(observation, f"storage_{clock}_rate_bounds"),
            -factor * _mp(mp, loss),
        )
    assert sum(gradient) == 0
    assert sum(nu * q for nu, q in zip(CAPACITIES, gradient)) != 0
    _contains(
        mp,
        observation.storage_rate_from_rows_bounds,
        sum(
            _mp(mp, q) * a - s * b for q, s, a, b in zip(gradient, current, dx, dtheta)
        ),
    )


def test_weighted_phase_target_uses_carrier_subtraction(report, phases, mp):
    weights = tuple(1 / nu for nu in CAPACITIES)
    weighted_phase = sum(_mp(mp, w * theta) for w, theta in zip(weights, phases))
    carrier = tuple(2 * mp.pi / 9 * (n % 9 - 4) for n in range(45))
    weighted_carrier = sum(_mp(mp, w) * psi for w, psi in zip(weights, carrier))
    assert abs(weighted_carrier) > 1
    origin = (weighted_phase - weighted_carrier) / _mp(mp, sum(weights))
    _contains(mp, report.weighted_phase_origin_bounds, origin)
    for bounds, psi in zip(report.asymptotic_phase_lift_bounds, carrier):
        _contains(mp, bounds, psi + origin)
    shifted = _assess(FORM, phases, turns=(7,) * 45)
    _contains(mp, shifted.weighted_phase_origin_bounds, origin + 14 * mp.pi)
    assert shifted.edge_angle_bounds == report.edge_angle_bounds
    assert shifted.observation == report.observation


@pytest.mark.parametrize("shift", (Q(-4, 5), Q(2)))
def test_uniform_form_shift_keeps_fields_and_changes_only_form_limit(
    report, phases, shift
):
    shifted = _assess(tuple(x + shift for x in FORM), phases)
    before, after = report.observation, shifted.observation
    assert after.unit_capacity_reference.form == tuple(x + shift for x in FORM)
    assert shifted.weighted_form_mean == report.weighted_form_mean + shift
    assert shifted.asymptotic_form == (shifted.weighted_form_mean,) * 45
    assert shifted.geometric_retention_certified and shifted.converges_to_target
    assert shifted.excess_bounds == report.excess_bounds
    assert shifted.centered_form_squared_upper_bound == (
        report.centered_form_squared_upper_bound
    )
    assert shifted.asymptotic_phase_lift_bounds == report.asymptotic_phase_lift_bounds
    for clock in ("tau", "original", "slow"):
        for channel in ("form", "phase"):
            assert getattr(after, f"{channel}_{clock}_rate_bounds") == getattr(
                before, f"{channel}_{clock}_rate_bounds"
            )
    assert after.pressure_tau_bounds == before.pressure_tau_bounds
    assert after.storage_tau_rate_bounds == before.storage_tau_rate_bounds


def test_insufficient_rail_bound_does_not_deny_geometry_or_infer_exit(phases):
    form = (Q(99, 100),) * 45
    report = _assess(form, phases)
    assert max(form) < 1
    assert report.form_absolute_upper_bound > 1
    assert not report.form_rails_certified and not report.admitted
    assert report.geometric_retention_certified and report.converges_to_target
    assert report.retained_windings == (1,) * 5
    assert report.asymptotic_form == form
    assert report.unresolved_conditions == (
        "weighted_mean_and_excess_do_not_certify_unit_form_rails",
    )


def test_common_capacity_scaling_preserves_mean_and_centered_bound(report, phases):
    scaled = _assess(FORM, phases, tuple(7 * nu for nu in CAPACITIES))
    assert scaled.weighted_form_mean == report.weighted_form_mean
    assert scaled.weighted_form_total == report.weighted_form_total / 7
    assert scaled.capacity_ratio == report.capacity_ratio
    assert scaled.centered_form_squared_upper_bound == (
        report.centered_form_squared_upper_bound
    )
    assert scaled.form_absolute_upper_bound == report.form_absolute_upper_bound
    for old, new in zip(
        report.observation.phase_tau_rate_bounds,
        scaled.observation.phase_tau_rate_bounds,
    ):
        assert max((7 * old).lo, new.lo) <= min((7 * old).hi, new.hi)


@pytest.mark.parametrize("control", ("wrong_turn", "high_energy", "wrong_winding"))
def test_outside_domain_retains_mean_but_withholds_future_bounds(control, phases):
    form, turns = FORM, TURNS
    if control == "wrong_turn":
        turns = (1,) + (0,) * 44
    elif control == "high_energy":
        form = tuple(32 * (x - Q(3, 10)) + Q(3, 10) for x in FORM)
    else:
        phases = (Q(0),) * 45
    report = _assess(form, phases, turns=turns)
    assert not report.geometric_retention_certified
    assert not report.admitted and not report.form_rails_certified
    assert report.weighted_form_mean != 0
    assert report.centered_form_squared_upper_bound is None
    assert report.form_absolute_upper_bound is None
    assert report.asymptotic_form is None
    assert report.asymptotic_phase_lift_bounds is None
    assert report.converges_to_target is None
    assert report.retained_windings is None


@pytest.mark.parametrize("field", ("form", "phases", "capacities"))
@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), 1j, "length"))
def test_invalid_primitives_reject_before_geometry(monkeypatch, field, invalid):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid primitive reached observation")

    monkeypatch.setattr(capture, "_observe_organized_source_raw_capacity", forbidden)
    values = dict(form=FORM, phases=(Q(0),) * 45, capacities=CAPACITIES)
    values[field] = (0,) * 44 if invalid == "length" else (invalid,) + (1,) * 44
    with pytest.raises((ValueError, TypeError)):
        capture._assess_organized_source_weighted_mean_retention(
            **values, phase_turns=TURNS
        )


@pytest.mark.parametrize("capacity", (Q(0), -Q(1, 10**300)))
def test_nonpositive_capacity_rejects_before_geometry(monkeypatch, capacity):
    def forbidden(*args, **kwargs):
        pytest.fail("Nonpositive capacity reached observation")

    monkeypatch.setattr(capture, "_observe_organized_source_raw_capacity", forbidden)
    with pytest.raises(ValueError, match="nonnegative|strictly positive"):
        _assess(FORM, (0,) * 45, (capacity,) + (Q(1),) * 44)


@pytest.mark.parametrize("turn", (True, Q(0), 0.0, "length"))
def test_invalid_turns_reject_before_geometry(monkeypatch, turn):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid turn reached observation")

    monkeypatch.setattr(capture, "_observe_organized_source_raw_capacity", forbidden)
    turns = (0,) * 44 if turn == "length" else (turn,) + (0,) * 44
    with pytest.raises((ValueError, TypeError)):
        _assess(FORM, (0,) * 45, turns=turns)


def test_negative_excess_upper_bound_is_a_consistency_error(monkeypatch, phases):
    original = capture._organized_source_raw_chart

    def inconsistent_chart(*args, **kwargs):
        chart = original(*args, **kwargs)
        assert chart["admitted"]
        chart["excess_bounds"] = I(-Q(1, 1000))
        return chart

    monkeypatch.setattr(capture, "_organized_source_raw_chart", inconsistent_chart)
    with pytest.raises(ValueError, match="cannot be negative"):
        _assess(FORM, phases)
