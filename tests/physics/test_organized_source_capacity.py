"""Complete held-capacity continuation, without executing Silence or any flow.

Frozen discriminator: e=e0-e1, x=10^-7 e, offsets=1/40+(1/100)e,
capacity1/2 on nodes0,1 and1 elsewhere. Retaining the old phase row has
positive storage derivative; sharing capacity in both rows cancels that term.
The prior Reception preparation supplies a separate retained-domain control.
An unreflected impulse controls weighted totals. No evaluated trajectory,
operator occurrence, acquired source or uniquely selected law is claimed.
"""

from fractions import Fraction as Q

import mpmath
import pytest
import sympy as sp

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I, sin
from tnfr.operators import network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _organized_source_capture as capture
from tnfr.physics import _organized_source_port as owner

ORIGIN, TILT, SMALL_FORM = Q(1, 40), Q(1, 100), Q(1, 10**7)
COUNTER_FORM = (SMALL_FORM, -SMALL_FORM) + (Q(0),) * 43
COUNTER_OFFSETS = (ORIGIN + TILT, ORIGIN - TILT) + (ORIGIN,) * 43
PAIR_CAPACITIES = (Q(1, 2), Q(1, 2)) + (Q(1),) * 43
UNIT_CAPACITIES = (Q(1),) * 45
RETAINED_FORM = (Q(1, 64), -Q(1, 64)) + (Q(0),) * 43
RETAINED_OFFSETS = tuple(
    ORIGIN
    + (TILT if c == 1 else -TILT if c == 2 else 0)
    * (1 if j == 0 else -1 if j == 1 else 0)
    for c in range(5)
    for j in range(9)
)


@pytest.fixture(scope="module", autouse=True)
def no_flow_event_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Capacity controls cannot execute flow, events or source producers")

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
            (owner, "_assess_organized_source_attachment"),
            (owner, "_assess_organized_source_emission"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def geometry():
    return owner._derive_organized_source_port()


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


@pytest.fixture(scope="module")
def counter_observations():
    return {
        (label, branch): owner._observe_organized_source_capacity(
            form=COUNTER_FORM,
            phase_offsets=COUNTER_OFFSETS,
            capacities=capacities,
            phase_mobility=branch,
        )
        for label, capacities in (("pair", PAIR_CAPACITIES), ("unit", UNIT_CAPACITIES))
        for branch in ("shared_capacity", "unit_capacity")
    }


@pytest.fixture(scope="module")
def retained():
    return {
        eta: capture._assess_organized_source_shared_capacity_retention(
            form=RETAINED_FORM, phase_offsets=RETAINED_OFFSETS, pair_capacity=eta
        )
        for eta in (Q(0), Q(1, 2), Q(1))
    }


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, value):
    assert _mp(mp, bounds.lo) <= value <= _mp(mp, bounds.hi)


def _fine_fields(geometry, form, offsets, mp):
    q, current = [Q(0)] * 45, [mp.mpf(0)] * 45
    alpha = 2 * mp.pi / 9
    for a, b, weight in geometry.weighted_edges:
        difference = weight * (form[a] - form[b])
        q[a] += difference
        q[b] -= difference
        angle = alpha * (b % 9 - a % 9) + _mp(mp, offsets[b] - offsets[a])
        sine = _mp(mp, weight) * mp.sin(angle)
        current[a] += sine
        current[b] -= sine
    return tuple(q), tuple(current)


def test_the_phase_mobility_premise_controls_storage_cancellation():
    q, s, gamma, nu, phase_nu = sp.symbols("q s gamma nu phase_nu", real=True)
    dx, dtheta = nu * (-q + gamma * s) / 4, phase_nu * gamma * q / 4
    work = sp.expand(q * dx - s * dtheta)
    assert sp.expand(work + nu * q**2 / 4 - gamma * (nu - phase_nu) * q * s / 4) == 0
    assert sp.simplify(work.subs(phase_nu, nu)) == -nu * q**2 / 4
    assert (
        sp.simplify(work.subs(phase_nu, 1) + nu * q**2 / 4)
        == gamma * q * s * (nu - 1) / 4
    )


@pytest.mark.parametrize("branch", ("shared_capacity", "unit_capacity"))
def test_capacity_rows_and_storage_match_independent_full_support_fields(
    geometry, mp, counter_observations, branch
):
    observation = counter_observations["pair", branch]
    q, current = _fine_fields(geometry, COUNTER_FORM, COUNTER_OFFSETS, mp)
    gamma = 1 / (1023 * mp.pi)
    pressure = tuple((-_mp(mp, qi) + gamma * si) / 4 for qi, si in zip(q, current))
    dx = tuple(_mp(mp, nu) * p for nu, p in zip(PAIR_CAPACITIES, pressure))
    phase_nu = PAIR_CAPACITIES if branch == "shared_capacity" else UNIT_CAPACITIES
    dtheta = tuple(gamma * _mp(mp, nu * qi) / 4 for nu, qi in zip(phase_nu, q))
    loss = sum(nu * qi**2 / 4 for nu, qi in zip(PAIR_CAPACITIES, q))
    direct = sum(
        _mp(mp, qi) * a - si * b for qi, si, a, b in zip(q, current, dx, dtheta)
    )
    assert observation.capacities == PAIR_CAPACITIES
    assert observation.phase_mobility == branch
    assert observation.unit_capacity_reference.form_gradient == q
    assert observation.dissipation_tau == loss == 11 * SMALL_FORM**2
    for clock, factor in (
        ("tau", mp.mpf(1)),
        ("original", mp.mpf(1023) / 1024),
        ("slow", 2 / gamma**2),
    ):
        for index in range(45):
            _contains(
                mp,
                getattr(observation, f"form_{clock}_rate_bounds")[index],
                factor * dx[index],
            )
            _contains(
                mp,
                getattr(observation, f"phase_{clock}_rate_bounds")[index],
                factor * dtheta[index],
            )
    for bounds, expected in zip(observation.pressure_tau_bounds, pressure):
        _contains(mp, bounds, expected)
    _contains(mp, observation.storage_tau_rate_bounds, direct)
    _contains(mp, observation.storage_rate_from_rows_bounds, direct)
    assert observation.balance_residual_bounds.contains(0)
    if branch == "shared_capacity":
        assert observation.inverse_capacity_weighted_form_total == 0
        assert observation.inverse_capacity_weighted_phase_offset_total == 47 * ORIGIN
    else:
        assert observation.inverse_capacity_weighted_form_total is None
        assert observation.inverse_capacity_weighted_phase_offset_total is None


def test_fixed_in_domain_counterexample_excludes_old_phase_row_decay_transfer(
    geometry, mp, counter_observations
):
    alpha, t = geometry.alpha_bounds, I(TILT)
    amplitude = sin(alpha + t) - sin(alpha - 2 * t) + 3 * sin(t)
    assert amplitude.lo > 3 * TILT
    assert geometry.gamma_bounds.lo > Q(1, 4000)
    assert -11 * SMALL_FORM**2 + Q(3, 2) * Q(1, 4000) * SMALL_FORM * 3 * TILT > 0
    shared = counter_observations["pair", "shared_capacity"]
    unchanged = counter_observations["pair", "unit_capacity"]
    assert shared.pressure_tau_bounds == unchanged.pressure_tau_bounds
    assert shared.form_tau_rate_bounds == unchanged.form_tau_rate_bounds
    assert shared.phase_tau_rate_bounds != unchanged.phase_tau_rate_bounds
    assert shared.storage_tau_rate_bounds.hi < 0 < unchanged.storage_tau_rate_bounds.lo
    a = 2 * mp.pi / 9
    tilt = _mp(mp, TILT)
    expected_a = mp.sin(a + tilt) - mp.sin(a - 2 * tilt) + 3 * mp.sin(tilt)
    expected_rate = -11 * _mp(mp, SMALL_FORM) ** 2 + 3 * _mp(
        mp, SMALL_FORM
    ) * expected_a / (2 * 1023 * mp.pi)
    _contains(mp, unchanged.storage_tau_rate_bounds, expected_rate)
    domain = capture._assess_organized_source_reflection_retention(
        form=COUNTER_FORM, phase_offsets=COUNTER_OFFSETS
    )
    upper = 6 * SMALL_FORM**2 + 6 * TILT**2
    assert domain.admitted and domain.excess_bounds.hi < upper < Q(1, 100)


def test_all_unit_capacities_recover_both_original_rows_and_loss(counter_observations):
    shared = counter_observations["unit", "shared_capacity"]
    unchanged = counter_observations["unit", "unit_capacity"]
    reference = shared.unit_capacity_reference
    paired = counter_observations["pair", "shared_capacity"]
    assert (
        paired.unit_capacity_reference.joint_storage_bounds
        == reference.joint_storage_bounds
    )
    assert paired.pressure_tau_bounds == shared.pressure_tau_bounds
    for clock in ("tau", "original", "slow"):
        for field in ("form", "phase"):
            name = f"{field}_{clock}_rate_bounds"
            assert (
                getattr(shared, name)
                == getattr(unchanged, name)
                == getattr(reference, name)
            )
    assert shared.dissipation_tau == reference.joint_dissipation_tau
    assert shared.storage_tau_rate_bounds.contains(reference.joint_storage_tau_rate)
    assert unchanged.storage_tau_rate_bounds.contains(reference.joint_storage_tau_rate)


def test_positive_held_capacity_totals_are_weighted_raw_sums_not_ordinary_means(
    geometry, mp
):
    h = Q(1, 64)
    form, offsets = (h,) + (Q(0),) * 44, (ORIGIN,) * 45
    capacities = (Q(1, 2),) + (Q(1),) * 44
    observation = owner._observe_organized_source_capacity(
        form=form,
        phase_offsets=offsets,
        capacities=capacities,
        phase_mobility="shared_capacity",
    )
    assert observation.inverse_capacity_weighted_form_total == 2 * h
    assert observation.inverse_capacity_weighted_phase_offset_total == 46 * ORIGIN
    q, current = _fine_fields(geometry, form, offsets, mp)
    assert sum(q) == 0 and q[0] == 5 * h
    assert max(abs(value) for value in current) < mp.mpf("1e-75")
    ordinary_form = sum(observation.form_tau_rate_bounds, I(0))
    ordinary_phase = sum(observation.phase_tau_rate_bounds, I(0))
    _contains(mp, ordinary_form, _mp(mp, 5 * h / 8))
    _contains(mp, ordinary_phase, -_mp(mp, 5 * h / 8) / (1023 * mp.pi))
    assert ordinary_form.lo > 0 and ordinary_phase.hi < 0
    assert sum(
        (rate / nu for rate, nu in zip(observation.form_tau_rate_bounds, capacities)),
        I(0),
    ).contains(0)
    assert sum(
        (rate / nu for rate, nu in zip(observation.phase_tau_rate_bounds, capacities)),
        I(0),
    ).contains(0)


@pytest.mark.parametrize("eta", (Q(1, 2), Q(1)))
def test_mirrored_positive_capacity_retains_the_old_domain_with_new_complete_rows(
    retained, eta
):
    report = retained[eta]
    continuation = report.continuation
    assert report.pair_capacity == eta and report.domain.admitted
    assert report.retained_windings == (1,) * 5
    assert report.form_absolute_upper_bound == Q(3, 10)
    assert report.converges_to_target is True
    assert continuation.capacities == (eta, eta) + (Q(1),) * 43
    assert continuation.phase_mobility == "shared_capacity"
    for node in range(45):
        mirror = 9 * (node // 9) + (1 - node % 9) % 9
        assert continuation.capacities[node] == continuation.capacities[mirror]
        assert (
            continuation.form_tau_rate_bounds[node]
            + continuation.form_tau_rate_bounds[mirror]
        ).contains(0)
        assert (
            continuation.phase_tau_rate_bounds[node]
            + continuation.phase_tau_rate_bounds[mirror]
        ).contains(0)
    assert continuation.storage_tau_rate_bounds.hi < 0
    assert continuation.inverse_capacity_weighted_form_total == 0
    assert continuation.inverse_capacity_weighted_phase_offset_total == ORIGIN * (
        43 + 2 / eta
    )


def test_zero_capacity_freezes_both_selected_rows_only_for_the_shared_premise(retained):
    report = retained[Q(0)]
    continuation = report.continuation
    assert report.domain.admitted and report.retained_windings == (1,) * 5
    assert report.form_absolute_upper_bound == Q(3, 10)
    assert report.converges_to_target is None
    for node in (0, 1):
        assert continuation.form_tau_rate_bounds[node] == I(0)
        assert continuation.phase_tau_rate_bounds[node] == I(0)
        assert not continuation.pressure_tau_bounds[node].contains(0)
        assert not continuation.unit_capacity_reference.phase_tau_rate_bounds[
            node
        ].contains(0)
    assert continuation.inverse_capacity_weighted_form_total is None
    assert continuation.inverse_capacity_weighted_phase_offset_total is None
    assert any(not row.contains(0) for row in continuation.form_tau_rate_bounds[2:])


def test_high_energy_state_keeps_observation_but_no_retention_or_convergence_claim():
    report = capture._assess_organized_source_shared_capacity_retention(
        form=(Q(1, 4), -Q(1, 4)) + (Q(0),) * 43,
        phase_offsets=RETAINED_OFFSETS,
        pair_capacity=Q(1, 2),
    )
    assert not report.domain.admitted and report.domain.in_acute_cell
    assert report.retained_windings is report.form_absolute_upper_bound is None
    assert report.converges_to_target is None and report.unresolved_conditions
    assert report.continuation.capacities == PAIR_CAPACITIES


@pytest.mark.parametrize("field", ("form", "phase_offsets", "capacities"))
@pytest.mark.parametrize("bad", (True, float("inf"), "short"))
def test_raw_capacity_law_inputs_reject_before_fine_observation(
    field, bad, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid capacity-law input reached the fine observer")

    arguments = dict(
        form=COUNTER_FORM,
        phase_offsets=COUNTER_OFFSETS,
        capacities=PAIR_CAPACITIES,
        phase_mobility="shared_capacity",
    )
    values = arguments[field]
    arguments[field] = values[:-1] if bad == "short" else (bad,) + values[1:]
    monkeypatch.setattr(owner, "_observe_organized_source_port", forbidden)
    with pytest.raises((TypeError, ValueError)):
        owner._observe_organized_source_capacity(**arguments)


@pytest.mark.parametrize("selector", (None, True, "shared", ""))
def test_phase_premise_is_explicit_and_admitted_before_geometry(selector, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid phase premise reached geometry")

    monkeypatch.setattr(owner, "_derive_organized_source_port", forbidden)
    with pytest.raises(ValueError, match="phase_mobility"):
        owner._observe_organized_source_capacity(
            form=COUNTER_FORM,
            phase_offsets=COUNTER_OFFSETS,
            capacities=PAIR_CAPACITIES,
            phase_mobility=selector,
        )


def test_negative_capacity_and_missing_phase_premise_are_not_coerced(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid capacity reached fine observation")

    monkeypatch.setattr(owner, "_observe_organized_source_port", forbidden)
    with pytest.raises(ValueError, match="nonnegative"):
        owner._observe_organized_source_capacity(
            form=COUNTER_FORM,
            phase_offsets=COUNTER_OFFSETS,
            capacities=(-Q(1),) + PAIR_CAPACITIES[1:],
            phase_mobility="shared_capacity",
        )
    with pytest.raises(TypeError, match="phase_mobility"):
        owner._observe_organized_source_capacity(
            form=COUNTER_FORM, phase_offsets=COUNTER_OFFSETS, capacities=PAIR_CAPACITIES
        )


@pytest.mark.parametrize("bad", (True, float("inf"), -Q(1), Q(2), (Q(1, 2),)))
def test_pair_capacity_domain_rejects_before_barrier_and_fine_state(bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid pair capacity reached geometry or fine observation")

    monkeypatch.setattr(capture, "_derive_organized_source_capture_barrier", forbidden)
    monkeypatch.setattr(owner, "_observe_organized_source_port", forbidden)
    with pytest.raises((TypeError, ValueError)):
        capture._assess_organized_source_shared_capacity_retention(
            form=RETAINED_FORM, phase_offsets=RETAINED_OFFSETS, pair_capacity=bad
        )


@pytest.mark.parametrize("field", ("form", "phase_offsets"))
def test_retention_wrapper_cannot_bypass_primitive_state_admission(field, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid retained state reached geometry")

    arguments = dict(
        form=RETAINED_FORM, phase_offsets=RETAINED_OFFSETS, pair_capacity=Q(1, 2)
    )
    arguments[field] = arguments[field][:-1] + (True,)
    monkeypatch.setattr(capture, "_derive_organized_source_capture_barrier", forbidden)
    monkeypatch.setattr(owner, "_derive_organized_source_port", forbidden)
    with pytest.raises((TypeError, ValueError)):
        capture._assess_organized_source_shared_capacity_retention(**arguments)
