"""Raw-phase held-capacity continuation, without event or flow execution.

The paired half-capacity control uses rendered winding-one phases with one
fixed local tilt. Its weighted-zero form has a nonzero ordinary total, so the
new leaf cannot be replaced by the unit-capacity theorem. A separate small
antisymmetric form discriminates shared from unchanged phase mobility.
"""

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
FORM = tuple(Q(1, 128) if n == 0 else -Q(1, 64) if n == 9 else Q(0) for n in range(45))
TURNS = (0,) * 45


@pytest.fixture(scope="module", autouse=True)
def no_event_flow_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Raw capacity admission cannot execute events, flow or sources")

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
def geometry():
    return port._derive_organized_source_port()


@pytest.fixture(scope="module")
def phases(geometry):
    return tuple(
        Q(
            float(
                geometry.alpha_bounds.midpoint * (n % 9 - 4)
                + Q(1, 40)
                + (Q(1, 100) if n == 0 else 0)
            )
        )
        for n in range(45)
    )


@pytest.fixture(scope="module")
def report(phases):
    return capture._assess_organized_source_raw_capacity_retention(
        form=FORM, phases=phases, phase_turns=TURNS, capacities=CAPACITIES
    )


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _fields(mp, geometry, form, phases):
    gradient, current = [Q(0)] * 45, [mp.mpf(0)] * 45
    storage = mp.mpf(0)
    for a, b, weight in geometry.weighted_edges:
        q = weight * (form[a] - form[b])
        gradient[a] += q
        gradient[b] -= q
        angle = _mp(mp, phases[b] - phases[a])
        sine = _mp(mp, weight) * mp.sin(angle)
        current[a] += sine
        current[b] -= sine
        storage += _mp(mp, weight * (form[a] - form[b]) ** 2 / 2)
        storage += _mp(mp, weight) * (1 - mp.cos(angle))
    return tuple(gradient), tuple(current), storage


def test_weighted_leaf_does_not_require_zero_ordinary_form(report, phases, mp):
    assert sum(FORM) == -Q(1, 128)
    assert sum(x / nu for x, nu in zip(FORM, CAPACITIES)) == 0
    assert report.admitted and report.geometric_retention_certified
    assert report.form_rails_certified and report.converges_to_target
    assert report.retained_windings == (1,) * 5
    assert report.capacity_ratio == 2
    assert report.form_squared_upper_bound == Q(3, 20)
    assert Q(3, 20) < report.form_absolute_upper_bound**2 < 1
    assert not report.unresolved_conditions
    assert report.observation.inverse_capacity_weighted_form_total == 0
    with pytest.raises(ValueError, match="zero total form"):
        capture._assess_organized_source_raw_retention(
            form=FORM, phases=phases, phase_turns=TURNS
        )
    weights = tuple(1 / nu for nu in CAPACITIES)
    assert report.inverse_capacity_weights == weights
    assert report.inverse_capacity_weight_sum == 47
    assert report.weighted_carrier_slot_sum == -7
    weighted_mean = (
        sum(_mp(mp, angle / nu) for angle, nu in zip(phases, CAPACITIES)) / 47
    )
    origin = weighted_mean + 14 * mp.pi / (9 * 47)
    _contains(mp, report.weighted_phase_origin_bounds, origin)
    assert not (
        _mp(mp, report.weighted_phase_origin_bounds.lo)
        <= weighted_mean
        <= _mp(mp, report.weighted_phase_origin_bounds.hi)
    )
    for n, bounds in enumerate(report.asymptotic_phase_lift_bounds):
        _contains(mp, bounds, 2 * mp.pi / 9 * (n % 9 - 4) + origin)


def test_weighted_spectral_estimate_keeps_the_constant_mode(report):
    """Weighted least squares controls the hidden ordinary constant component."""
    weights = report.inverse_capacity_weights
    mean = sum(FORM) / 45
    centered = tuple(x - mean for x in FORM)
    weighted_x = sum(w * x**2 for w, x in zip(weights, FORM))
    weighted_y = sum(w * y**2 for w, y in zip(weights, centered))
    norm_x, norm_y = sum(x**2 for x in FORM), sum(y**2 for y in centered)
    energy = sum(report.observation.unit_capacity_reference.form_edge_storage)
    assert norm_x > norm_y
    assert min(weights) * norm_x <= weighted_x <= weighted_y <= max(weights) * norm_y
    assert report.spectral_gap_lower_bound * norm_y <= 2 * energy
    assert (
        norm_x <= report.capacity_ratio * 2 * energy / report.spectral_gap_lower_bound
    )
    assert energy < report.excess_bounds.lo < report.energy_ceiling


def test_raw_rows_balance_and_all_clocks_match_independent_incidence(
    report, phases, mp, geometry
):
    observation = report.observation
    q, current, storage = _fields(mp, geometry, FORM, phases)
    gamma = 1 / (1023 * mp.pi)
    pressure = tuple((-_mp(mp, qi) + gamma * si) / 4 for qi, si in zip(q, current))
    dx = tuple(_mp(mp, nu) * p for nu, p in zip(CAPACITIES, pressure))
    dtheta = tuple(gamma * _mp(mp, nu * qi) / 4 for nu, qi in zip(CAPACITIES, q))
    loss = sum(nu * qi**2 / 4 for nu, qi in zip(CAPACITIES, q))
    assert observation.dissipation_tau == loss
    assert observation.unit_capacity_reference.form_gradient == q
    _contains(mp, observation.unit_capacity_reference.joint_storage_bounds, storage)
    _contains(mp, observation.storage_tau_rate_bounds, -_mp(mp, loss))
    _contains(
        mp,
        observation.storage_rate_from_rows_bounds,
        sum(_mp(mp, qi) * a - si * b for qi, si, a, b in zip(q, current, dx, dtheta)),
    )
    assert observation.balance_residual_bounds.contains(0)
    for bounds, value in zip(observation.pressure_tau_bounds, pressure):
        _contains(mp, bounds, value)
    for clock, factor in (
        ("tau", mp.mpf(1)),
        ("original", mp.mpf(1023) / 1024),
        ("slow", 2 / gamma**2),
    ):
        for channel, expected in (("form", dx), ("phase", dtheta)):
            for bounds, value in zip(
                getattr(observation, f"{channel}_{clock}_rate_bounds"), expected
            ):
                _contains(mp, bounds, factor * value)
        _contains(
            mp,
            getattr(observation, f"storage_{clock}_rate_bounds"),
            -factor * _mp(mp, loss),
        )
    # Capacity-weighted rows conserve the declared totals, not ordinary means.
    assert sum(q) == 0
    assert sum(nu * qi for nu, qi in zip(CAPACITIES, q)) != 0
    weighted_dx = sum(
        (rate / nu for rate, nu in zip(observation.form_tau_rate_bounds, CAPACITIES)),
        I(0),
    )
    weighted_dt = sum(
        (rate / nu for rate, nu in zip(observation.phase_tau_rate_bounds, CAPACITIES)),
        I(0),
    )
    assert weighted_dx.contains(0) and weighted_dt.contains(0)


def test_unshared_phase_row_has_positive_storage_control(geometry):
    small, tilt = Q(1, 10**7), Q(1, 100)
    form = (small, -small) + (Q(0),) * 43
    phases = tuple(
        Q(
            float(
                geometry.alpha_bounds.midpoint * (n % 9 - 4)
                + Q(1, 40)
                + (tilt if n == 0 else -tilt if n == 1 else 0)
            )
        )
        for n in range(45)
    )
    shared, unchanged = (
        port._observe_organized_source_raw_capacity(
            form=form, phases=phases, capacities=CAPACITIES, phase_mobility=branch
        )
        for branch in ("shared_capacity", "unit_capacity")
    )
    assert shared.pressure_tau_bounds == unchanged.pressure_tau_bounds
    assert shared.form_tau_rate_bounds == unchanged.form_tau_rate_bounds
    assert shared.phase_tau_rate_bounds != unchanged.phase_tau_rate_bounds
    assert shared.storage_tau_rate_bounds.hi < 0 < unchanged.storage_tau_rate_bounds.lo
    assert unchanged.inverse_capacity_weighted_form_total is None
    assert unchanged.inverse_capacity_weighted_raw_phase_total is None


def test_unit_capacity_recovers_existing_raw_law_and_continuation(phases):
    form = (Q(1, 64), -Q(1, 64)) + (Q(0),) * 43
    previous = capture._assess_organized_source_raw_retention(
        form=form, phases=phases, phase_turns=TURNS
    )
    actual = capture._assess_organized_source_raw_capacity_retention(
        form=form, phases=phases, phase_turns=TURNS, capacities=(Q(1),) * 45
    )
    assert actual.admitted and previous.admitted
    assert actual.observation.unit_capacity_reference == previous.observation
    assert actual.excess_bounds == previous.excess_bounds
    assert actual.edge_angle_bounds == previous.edge_angle_bounds
    assert actual.weighted_phase_origin_bounds == previous.lifted_phase_mean_bounds
    assert actual.asymptotic_phase_lift_bounds == previous.asymptotic_phase_lift_bounds
    assert actual.form_squared_upper_bound == previous.form_squared_upper_bound
    for clock in ("tau", "original", "slow"):
        for channel in ("form", "phase"):
            field = f"{channel}_{clock}_rate_bounds"
            assert getattr(actual.observation, field) == getattr(
                previous.observation, field
            )


def test_common_turn_shift_moves_only_weighted_origin(report, phases, mp):
    shifted = capture._assess_organized_source_raw_capacity_retention(
        form=FORM, phases=phases, phase_turns=(2,) * 45, capacities=CAPACITIES
    )
    assert shifted.admitted
    assert shifted.observation == report.observation
    assert shifted.edge_angle_bounds == report.edge_angle_bounds
    assert shifted.weighted_phase_turn_sum == 94
    origin = (_mp(mp, report.weighted_raw_phase_sum) + 14 * mp.pi / 9) / 47 + 4 * mp.pi
    _contains(mp, shifted.weighted_phase_origin_bounds, origin)


def test_common_capacity_scale_preserves_leaf_geometry_and_scales_rates(report, phases):
    scaled = capture._assess_organized_source_raw_capacity_retention(
        form=FORM,
        phases=phases,
        phase_turns=TURNS,
        capacities=tuple(4 * nu for nu in CAPACITIES),
    )
    assert scaled.admitted and scaled.capacity_ratio == report.capacity_ratio
    assert scaled.edge_angle_bounds == report.edge_angle_bounds
    assert scaled.excess_bounds == report.excess_bounds
    assert scaled.form_squared_upper_bound == report.form_squared_upper_bound
    assert scaled.weighted_phase_origin_bounds == report.weighted_phase_origin_bounds
    before, after = report.observation, scaled.observation
    assert before.pressure_tau_bounds == after.pressure_tau_bounds
    assert after.dissipation_tau == 4 * before.dissipation_tau
    for clock in ("tau", "original", "slow"):
        for channel in ("form", "phase"):
            for old, new in zip(
                getattr(before, f"{channel}_{clock}_rate_bounds"),
                getattr(after, f"{channel}_{clock}_rate_bounds"),
            ):
                # Independent enclosure midpoints are unnecessary: the same
                # exact scaling of dyadic enclosures has a nonempty overlap.
                expected = 4 * old
                assert max(new.lo, expected.lo) <= min(new.hi, expected.hi)


def test_large_ratio_does_not_infer_unit_form_rails(phases):
    capacities = (Q(1, 20),) + (Q(1),) * 44
    report = capture._assess_organized_source_raw_capacity_retention(
        form=(Q(0),) * 45, phases=phases, phase_turns=TURNS, capacities=capacities
    )
    assert report.geometric_retention_certified and report.converges_to_target
    assert report.retained_windings == (1,) * 5
    assert report.form_squared_upper_bound == Q(3, 2)
    assert not report.form_rails_certified and not report.admitted
    assert report.unresolved_conditions == (
        "capacity_ratio_does_not_certify_unit_form_rails",
    )


@pytest.mark.parametrize("control", ("wrong_turn", "high_energy", "wrong_winding"))
def test_unavailable_domains_do_not_inherit_retention(control, phases):
    form, turns = FORM, TURNS
    if control == "wrong_turn":
        turns = (1,) + (0,) * 44
    elif control == "high_energy":
        form = tuple(32 * x for x in FORM)
    else:
        form, phases = (Q(0),) * 45, (Q(0),) * 45
    result = capture._assess_organized_source_raw_capacity_retention(
        form=form, phases=phases, phase_turns=turns, capacities=CAPACITIES
    )
    assert not result.admitted and not result.geometric_retention_certified
    assert result.retained_windings is None and result.converges_to_target is None
    assert result.asymptotic_phase_lift_bounds is None


def test_zero_capacity_observer_is_not_inverse_leaf_admission(phases):
    capacities = (Q(0),) + (Q(1),) * 44
    observation = port._observe_organized_source_raw_capacity(
        form=FORM,
        phases=phases,
        capacities=capacities,
        phase_mobility="shared_capacity",
    )
    assert observation.form_tau_rate_bounds[0] == I(0)
    assert observation.phase_tau_rate_bounds[0] == I(0)
    assert observation.inverse_capacity_weighted_form_total is None
    assert observation.inverse_capacity_weighted_raw_phase_total is None
    with pytest.raises(ValueError, match="strictly positive"):
        capture._assess_organized_source_raw_capacity_retention(
            form=FORM, phases=phases, phase_turns=TURNS, capacities=capacities
        )


def test_negative_capacity_rejects_before_raw_observation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Negative capacity reached geometry")

    monkeypatch.setattr(port, "_observe_organized_source_phase_state", forbidden)
    capacities = (-Q(1, 10**300),) + (Q(1),) * 44
    with pytest.raises(ValueError, match="nonnegative"):
        port._observe_organized_source_raw_capacity(
            form=FORM,
            phases=(0,) * 45,
            capacities=capacities,
            phase_mobility="shared_capacity",
        )
    with pytest.raises(ValueError, match="nonnegative"):
        capture._assess_organized_source_raw_capacity_retention(
            form=FORM, phases=(0,) * 45, phase_turns=TURNS, capacities=capacities
        )


@pytest.mark.parametrize("field", ("form", "phases", "capacities"))
@pytest.mark.parametrize("invalid", (True, float("nan"), 1j, "wrong_length"))
def test_invalid_primitives_precede_all_geometry(monkeypatch, field, invalid):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid primitive reached observation")

    monkeypatch.setattr(port, "_derive_organized_source_port", forbidden)
    values = dict(form=(Q(0),) * 45, phases=(Q(0),) * 45, capacities=(Q(1),) * 45)
    values[field] = (
        (Q(0),) * 44 if invalid == "wrong_length" else (invalid,) + (Q(1),) * 44
    )
    with pytest.raises((ValueError, TypeError)):
        port._observe_organized_source_raw_capacity(
            **values, phase_mobility="shared_capacity"
        )
    with pytest.raises((ValueError, TypeError)):
        capture._assess_organized_source_raw_capacity_retention(
            **values, phase_turns=TURNS
        )


@pytest.mark.parametrize("invalid", (True, Q(0), 0.0, "wrong_length"))
def test_invalid_turns_precede_geometry(monkeypatch, invalid):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid branch data reached observation")

    monkeypatch.setattr(capture, "_observe_organized_source_raw_capacity", forbidden)
    turns = (0,) * 44 if invalid == "wrong_length" else (invalid,) + (0,) * 44
    with pytest.raises((ValueError, TypeError)):
        capture._assess_organized_source_raw_capacity_retention(
            form=FORM, phases=(0,) * 45, phase_turns=turns, capacities=CAPACITIES
        )


def test_nonzero_weighted_leaf_rejects_before_observation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Wrong form leaf reached observation")

    monkeypatch.setattr(capture, "_observe_organized_source_raw_capacity", forbidden)
    # Ordinary zero form does not suffice when the nonzero pair has unequal capacities.
    form = (Q(1, 64),) + (Q(0),) * 8 + (-Q(1, 64),) + (Q(0),) * 35
    assert sum(form) == 0
    with pytest.raises(ValueError, match="zero sum"):
        capture._assess_organized_source_raw_capacity_retention(
            form=form, phases=(0,) * 45, phase_turns=TURNS, capacities=CAPACITIES
        )


@pytest.mark.parametrize("branch", (None, True, "inferred", []))
def test_phase_mobility_is_explicit_and_validated(branch):
    with pytest.raises(ValueError, match="phase_mobility"):
        port._observe_organized_source_raw_capacity(
            form=FORM, phases=(0,) * 45, capacities=CAPACITIES, phase_mobility=branch
        )
